"""The JevAI runner: waits in the background for Hearts of Iron IV, reads the mod's state lines from game.log, asks the
model which posture each AI country should follow, and hands the choices to the mod. No Python install, launch option,
dump or console input needed; players only use the Paradox launcher. Ships as jevai.exe.

    jevai.exe --install      start JevAI with Windows (a hidden Startup-folder shortcut) and right away
    jevai.exe --uninstall    remove that and stop the background runner
    jevai.exe                run in this window: wait for HOI4, steer it while it runs, repeat

The mod logs each country's state to game.log (scripted_effects/jevai_state.txt: JEV|S/T/G/H/N/E/A lines): every
country on the 1st of each month and, in the "major powers only" mode, the major powers every week. When HOI4 starts,
the runner loads the model (NPU if present, else CPU; the first NPU compile takes ~2 min, later loads come from cache),
rebuilds the model's state text from each update and decides for the countries in it: every AI country monthly, or
the major powers weekly. It writes the choices to <userdir>/mod/jevai/history/units/JEVAI_orders.txt; the mod reloads
that file daily with load_oob and applies each posture through its scripted effects. When the game closes it frees
the model and waits for the next one. It also keeps JevAI loading after every installed overhaul mod
(patch_load_order).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

from .pack import Packer
from .postures import NAMES as POSTURES, VERSION
from .text import GROWTH_LEVELS, posture_questions, state_text

ORDERS = "JEVAI_orders"  # load_oob name: history/units/JEVAI_orders.txt inside the mod
IDEOLOGY = {"fascism": "fascist", "communism": "communist", "democratic": "democratic", "neutrality": "non-aligned"}
LINE = re.compile(r"^\[[^\]]*\]\[(\d+)\.(\d+)\.(\d+)\.\d+\]\[[^\]]*\]: (JEV\|.*?)\s*$")
KINDS = {"S", "T", "G", "H", "N", "E", "A", "MAJOR"}
LOC_NAME = re.compile(r'^\s*([A-Z][A-Z0-9]{2}(?:_(?:fascism|communism|democratic|neutrality))?):\d*\s*"(.*)"\s*$')
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW for helper processes


def default_userdir() -> str:
    return os.path.join(os.path.expanduser("~"), "Documents", "Paradox Interactive", "Hearts of Iron IV")


def find_game() -> str | None:
    """hoi4.exe from Steam's library list (every library folder), or None."""
    steam = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Steam")
    libs = [steam]
    try:
        with open(os.path.join(steam, "steamapps", "libraryfolders.vdf"), encoding="utf-8") as f:
            libs += [p.replace("\\\\", "\\") for p in re.findall(r'"path"\s+"([^"]+)"', f.read())]
    except OSError:
        pass
    for lib in libs:
        exe = os.path.join(lib, "steamapps", "common", "Hearts of Iron IV", "hoi4.exe")
        if os.path.isfile(exe):
            return exe
    return None


def pids(image: str) -> list[int]:
    """PIDs of running processes with this image name."""
    r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                       creationflags=NO_WINDOW)
    return [int(m) for m in re.findall(r'^"[^"]+","(\d+)"', r.stdout, re.M)]


STARTUP_LINK = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "Startup",
                            "JevAI.lnk")


def install(exe: str):
    """A Startup-folder shortcut that runs `jevai.exe --hidden` at every logon, then start it now."""
    q = lambda s: s.replace("'", "''")  # noqa: E731 - PowerShell single-quoted string
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{q(STARTUP_LINK)}');$s.TargetPath='{q(exe)}';"
                    f"$s.Arguments='--hidden';$s.WorkingDirectory='{q(os.path.dirname(exe))}';$s.WindowStyle=7;$s.Save()"],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    if not pids_of_others("jevai.exe"):
        os.startfile(STARTUP_LINK)
    print(f"JevAI starts with Windows ({STARTUP_LINK}) and is running now. Just play from the Paradox launcher.")


def uninstall():
    if os.path.exists(STARTUP_LINK):
        os.remove(STARTUP_LINK)
    for pid in pids_of_others("jevai.exe"):
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=NO_WINDOW)
    print("JevAI no longer starts with Windows, and the background runner is stopped.")


def pids_of_others(image: str) -> list[int]:
    return [p for p in pids(image) if p != os.getpid()]


def here() -> str:
    """Folder of the runner: next to jevai.exe when frozen, else this package."""
    return os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _mod_root(userdir: str, descriptor_text: str) -> str | None:
    p = re.search(r'^\s*path\s*=\s*"([^"]+)"', descriptor_text, re.M)
    return None if not p else p.group(1) if os.path.isabs(p.group(1)) else os.path.join(userdir, p.group(1))


def patch_load_order(userdir: str) -> list[str]:
    """Overhaul mods list folders under replace_path, which drops those folders (on_actions, events, scripted effects,
    history/units) from every mod loaded before them, and mods load alphabetically unless a dependency says otherwise.
    So JevAI declares every installed mod that uses replace_path as a dependency and loads after it. The launcher and
    the game read the change at their next start."""
    moddir = os.path.join(userdir, "mod")
    names = set()
    for f in glob.glob(os.path.join(moddir, "*.mod")):
        text = _read(f)
        name = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.M)
        if not name or name.group(1) == "JevAI":
            continue
        root = _mod_root(userdir, text)
        if "replace_path" in text or (root and "replace_path" in _read(os.path.join(root, "descriptor.mod"))):
            names.add(name.group(1))
    block_re = re.compile(r"\n?dependencies\s*=\s*\{([^}]*)\}\s*")
    shipped = block_re.search(_read(os.path.join(moddir, "jevai", "descriptor.mod")))
    names |= set(re.findall(r'"([^"]+)"', shipped.group(1))) if shipped else set()  # the well-known list stays too
    block = "dependencies = {\n" + "".join(f'\t"{n}"\n' for n in sorted(names)) + "}\n"
    for f in (os.path.join(moddir, "jevai.mod"), os.path.join(moddir, "jevai", "descriptor.mod")):
        text = _read(f)
        if not text:
            continue
        new = block_re.sub("\n", text).rstrip() + "\n" + block
        if new != text:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new)
    return sorted(names)


def active_mods(userdir: str) -> list[str]:
    """Folders of the mods enabled in the launcher (dlc_load.json)."""
    try:
        enabled = json.loads(_read(os.path.join(userdir, "dlc_load.json"))).get("enabled_mods", [])
    except ValueError:
        return []
    return [r for r in (_mod_root(userdir, _read(os.path.join(userdir, m))) for m in enabled) if r]


def english_names(roots: list[str]) -> dict[str, str]:
    """TAG and TAG_<ideology group> -> English country name, from the English localisation of the game and the mods
    (later roots win). The model was trained on English names; this keeps them English in any game language."""
    names = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "localisation", "**", "*_l_english.yml"), recursive=True):
            for line in _read(path).splitlines():
                m = LOC_NAME.match(line)
                if m:
                    names[m.group(1)] = m.group(2)
    return names


class GameLog:
    """The mod's JEV lines, read from <userdir>/logs/game.log as the game writes them. Lines come in bursts (every
    country on the 1st of each month; major powers weekly in the "major powers only" mode), grouped here by game date.
    `latest` keeps each country's most recent lines, so a weekly burst of major powers still sees last month's
    neighbours. The game recreates the file when it starts, so a file shorter than what was read means a new session."""

    def __init__(self, userdir: str):
        self.path = os.path.join(userdir, "logs", "game.log")
        self.pos, self.mode, self.bursts, self.latest, self.last = 0, None, {}, {}, 0.0

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.pos:
            self.pos, self.mode, self.bursts, self.latest = 0, None, {}, {}
        if size == self.pos:
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read(size - self.pos)
        end = chunk.rfind(b"\n") + 1  # a line still being written waits for the next poll
        self.pos += end
        for line in chunk[:end].decode("utf-8", "replace").splitlines():
            m = LINE.match(line)
            if not m:
                continue
            p = m.group(4).split("|")
            if p[1] == "MODE" and len(p) > 2:
                self.mode = p[2]
            elif p[1] in KINDS and len(p) > 2:
                self.bursts.setdefault(tuple(map(int, m.groups()[:3])), []).append(p)
                self.last = time.time()

    def complete(self, quiet: float = 5.0) -> list[tuple[str, set[str]]]:
        """Bursts whose lines have all arrived (a later burst began, or nothing new came for `quiet` seconds), oldest
        first, as (date, tags that reported in it); each one updates `latest`."""
        done = sorted(self.bursts)
        if done and time.time() - self.last < quiet:
            done = done[:-1]
        out = []
        for d in done:
            by: dict[str, list] = {}
            for p in self.bursts.pop(d):
                by.setdefault(p[2], []).append(p)
            self.latest.update(by)
            out.append((".".join(map(str, d)), {t for t, ls in by.items() if any(p[1] == "S" for p in ls)}))
        return out


def _num(kv: dict, k: str) -> float:
    try:
        return float(kv.get(k) or 0)
    except ValueError:
        return 0.0


def log_records(lines: list[list[str]], english: dict[str, str], date: str) -> tuple[dict[str, dict], dict[str, str], set[str], set[str]]:
    """One month of JEV lines -> records shaped like runtime.game.country_record (what state_text reads), names for every
    tag the lines mention, the human-played tags and the major powers. Units at the front, fleets and equipment requests
    are not in the log: 0 / empty."""
    recs, group, humans, majors = {}, {}, set(), set()

    def rec(tag: str) -> dict:
        return recs.setdefault(tag, {"tag": tag, "name": tag, "date": date, "mil": 0, "civ": 0, "nav": 0, "n_states": 0,
                                     "divisions": 0, "front_units": 0, "fronts": 0, "fleets": 0, "enemies": [],
                                     "allies": [], "neighbors": [], "manpower": {}, "equipment_fill": {}, "jev": {}})

    for p in lines:
        kind, tag = p[1], p[2]
        if kind in ("S", "T"):
            kv = {}
            for x in p[3:]:
                k, _, v = x.partition("=")
                kv[k] = v
            r = rec(tag)
            if kind == "S":
                r.update(name=kv.get("name") or tag, mil=int(_num(kv, "mil")), civ=int(_num(kv, "civ")), nav=int(_num(kv, "nav")),
                         n_states=int(_num(kv, "st")), divisions=int(_num(kv, "div")))
                r["manpower"]["max"] = _num(kv, "mpmax") * 1000
            else:
                r["jev"].update({k: (_num(kv, k) if k not in ("ideo", "fac") else v) for k, v in kv.items()})
                r["manpower"]["available"] = _num(kv, "mp") * 1000
        elif kind == "G" and len(p) > 3:
            group[tag] = p[3]
        elif kind == "H":
            humans.add(tag)
        elif kind == "MAJOR":
            majors.add(tag)
        elif kind in ("N", "E", "A") and len(p) > 3:
            rec(tag)[{"N": "neighbors", "E": "enemies", "A": "allies"}[kind]].append(p[3])
    for tag, r in recs.items():
        g = group.get(tag)
        if g in IDEOLOGY:
            r["jev"]["ideo"] = IDEOLOGY[g]  # the training text's words, whatever the game language
        r["name"] = english.get(f"{tag}_{g}") or english.get(tag) or r["name"]
    # countries without states log no S line; name them from the localisation (enemies in exile, for example)
    names = {t: english.get(t, t) for r in recs.values() for t in r["enemies"] + r["allies"] + r["neighbors"]}
    names.update({t: r["name"] for t, r in recs.items()})
    return {t: r for t, r in recs.items() if r["n_states"] > 0}, names, humans, majors


class Model:
    """The exported IR on one device: NPU if present (its own graph, compiled once and cached), else CPU."""

    def __init__(self, model_dir: str, device: str = "auto"):
        import openvino as ov
        with open(os.path.join(model_dir, "jev.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        core = ov.Core()
        devices = core.available_devices
        if device == "auto":
            device = "NPU" if "NPU" in devices else "CPU"
        self.device = device
        cache = os.path.join(os.environ.get("LOCALAPPDATA") or model_dir, "jevai", "ov_cache")
        os.makedirs(cache, exist_ok=True)
        core.set_property({"CACHE_DIR": cache})  # the NPU compile happens once per model and driver
        graph = self.cfg["graphs"].get(device, self.cfg["graphs"]["CPU"])
        t = time.time()
        self.req = core.compile_model(os.path.join(model_dir, graph), device, {"PERFORMANCE_HINT": "LATENCY"}).create_infer_request()
        print(f"model on {device} ({graph}) ready in {time.time() - t:.0f}s", flush=True)
        self.packer = Packer(model_dir, self.cfg["max_state_tokens"], self.cfg["seq"])

    def scores(self, state: str, horizon: int, weights=(1.0, 1.0, 0.5)) -> list[float]:
        """U per posture (POSTURES order), the same score jevd uses."""
        w_ok, w_gain, w_grow = weights
        top = len(GROWTH_LEVELS) - 1
        out = []
        for p in POSTURES:
            b = self.packer([(state, posture_questions(p, VERSION, horizon))], length=self.cfg["seq"],
                            questions=self.cfg["questions"], options=self.cfg["options"])
            probs = self.req.infer({k: b[k] for k in ("input_ids", "attention_mask", "pool", "opt_mask")})["probs"][0]
            grow = float(np.dot(np.arange(len(GROWTH_LEVELS)), probs[0, : len(GROWTH_LEVELS)]))
            out.append(w_ok * float(probs[1, 1]) + w_gain * float(probs[2, 1]) + w_grow * grow / top)
        return out


def orders_file(postures: dict[str, int], date: str, device: str) -> str:
    """An OOB file whose instant_effect sets each country's wanted posture; the mod applies it to AI countries.
    It holds no units, so loading it changes nothing but those variables."""
    lines = [f"# JevAI orders for {date} ({device}); rewritten by the runner, reloaded daily by the mod\n",
             "instant_effect = {\n"]
    for tag, k in sorted(postures.items()):
        lines.append(f"\tif = {{ limit = {{ country_exists = {tag} }} {tag} = {{ set_variable = {{ jev_want = {k} }} }} }}\n")
    lines.append("}\n")
    return "".join(lines)


def write_atomic(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class Tee:
    """Runner output goes to the console and to jevai.log next to jevai.exe, readable after the window closes."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def session(a, out_dir: str, pid: int):
    """Steer one running game until its process `pid` exits: load the English names and the model (from the NPU cache
    after the first time) while the game loads, then decide on each update the mod logs."""
    ready: dict = {}

    def prepare():
        try:
            exe = find_game()
            ready["names"] = english_names(([os.path.dirname(exe)] if exe else []) + active_mods(a.userdir))
            ready["model"] = Model(a.model, a.device)
        except Exception as e:  # noqa: BLE001 - reported below; the game keeps running without the model
            ready["error"] = e

    loader = threading.Thread(target=prepare, daemon=True)
    loader.start()
    log = GameLog(a.userdir)
    model = names_en = None
    checked = time.time()
    while True:
        if time.time() - checked > 5:
            checked = time.time()
            if pid not in pids("hoi4.exe"):
                return
        log.poll()
        if model is None:
            if "error" in ready:
                print(f"model failed to load: {ready['error']}", flush=True)
                return
            if loader.is_alive():
                time.sleep(1)
                continue
            model, names_en = ready["model"], ready["names"]
            log.complete(quiet=0)  # what was logged before the model was ready is history; decide from the next update
            print("waiting for the next update (every country on the 1st of the month; major powers weekly in that mode)",
                  flush=True)
        done = log.complete()
        if not done:
            time.sleep(1)
            continue
        date, reported = done[-1]  # if the game ran ahead of the model, the older bursts are stale: decide the newest
        now = f"[{time.strftime('%H:%M:%S')}] {date}"
        if log.mode in (None, "off"):
            print(f"{now}: " + ("waiting for the JevAI choice in the start-of-game event" if log.mode is None
                                else "JevAI is off for this game (vanilla AI)"), flush=True)
            continue
        recs, names, humans, majors = log_records([p for ls in log.latest.values() for p in ls], names_en, date)
        humans |= set(filter(None, a.player.split(",")))
        elig = [t for t in reported if t in recs and t not in humans and recs[t]["mil"] + recs[t]["civ"] >= a.min_factories
                and (log.mode == "all" or t in majors)]
        if not elig:
            continue
        t0 = time.time()
        chosen = {}
        for t in elig:
            u = model.scores(state_text(recs[t], names, recs), a.horizon)
            best, cur = int(np.argmax(u)), int(recs[t]["jev"].get("pos") or 0) - 1  # cur: posture in force, -1 none
            # near-ties flip with tiny changes (even the date): switch only for a clear gain
            chosen[t] = (cur if 0 <= cur < len(u) and u[best] - u[cur] < a.stick else best) + 1
        write_atomic(os.path.join(out_dir, ORDERS + ".txt"), orders_file(chosen, date, model.device))
        top = sorted(chosen.items(), key=lambda kv: -(recs[kv[0]]["mil"] + recs[kv[0]]["civ"]))[:6]
        print(f"{now} ({log.mode}): {len(chosen)} countries in {time.time() - t0:.0f}s on {model.device}; "
              + ", ".join(f"{t} {POSTURES[k - 1]}" for t, k in top), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="JevAI runner: the model picks AI postures in your HOI4 games")
    ap.add_argument("--install", action="store_true", help="start JevAI with Windows (hidden) and right away")
    ap.add_argument("--uninstall", action="store_true", help="stop starting with Windows and stop the background runner")
    ap.add_argument("--hidden", action="store_true", help="no window (how the Startup shortcut runs it)")
    ap.add_argument("--userdir", default=default_userdir())
    ap.add_argument("--model", default=os.path.join(here(), "model"))
    ap.add_argument("--device", default="auto", choices=["auto", "NPU", "CPU", "GPU"])
    ap.add_argument("--horizon", type=int, default=6, help="months the posture questions look ahead")
    ap.add_argument("--stick", type=float, default=0.01, help="keep the current posture unless another scores this much higher")
    ap.add_argument("--min-factories", type=int, default=20, help="skip countries with fewer military + civilian factories")
    ap.add_argument("--player", default="", help="extra tag(s) never to steer, comma separated (humans are excluded anyway)")
    a = ap.parse_args(argv)
    if a.install or a.uninstall:
        if a.uninstall:
            return uninstall()
        if not getattr(sys, "frozen", False):
            sys.exit("--install works from jevai.exe")
        return install(sys.executable)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if a.hidden:
        ctypes.windll.user32.ShowWindow(k32.GetConsoleWindow(), 0)
    mutex = k32.CreateMutexW(None, False, r"Local\JevAIRunner")  # one runner per user
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        sys.exit("JevAI is already running")
    sys.stdout.reconfigure(errors="replace")  # mod names can be in any script; a redirected cp1252 stdout must not crash
    sys.stdout = Tee(sys.stdout, open(os.path.join(here(), "jevai.log"), "a", encoding="utf-8"))
    print(f"\n=== JevAI runner started {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    out_dir = os.path.join(a.userdir, "mod", "jevai", "history", "units")
    if not os.path.isdir(out_dir):
        sys.exit(f"JevAI mod not found in {os.path.join(a.userdir, 'mod', 'jevai')}; install it first")
    n_deps = None
    while True:
        deps = patch_load_order(a.userdir)  # before each game, so a newly installed overhaul is covered
        if len(deps) != n_deps:
            n_deps = len(deps)
            print(f"JevAI loads after {n_deps} overhaul mods (read by the launcher at its next start)", flush=True)
        print("waiting for HOI4 (start it from the Paradox launcher)", flush=True)
        while not (running := pids("hoi4.exe")):
            time.sleep(5)
        print(f"[{time.strftime('%H:%M:%S')}] HOI4 started (pid {running[0]}); loading the model", flush=True)
        session(a, out_dir, running[0])
        print(f"[{time.strftime('%H:%M:%S')}] HOI4 closed", flush=True)
    del mutex


if __name__ == "__main__":
    main()
