"""The JevAI runner: starts Hearts of Iron IV (or attaches to it), reads the mod's monthly state lines from game.log,
asks the model which posture each AI country should follow, and hands the choices to the mod. No Python install,
launch option, dump or console input needed at play time; ships as jevai.exe.

    jevai.exe [--game "...\\Hearts of Iron IV\\hoi4.exe"] [--userdir ...] [--device auto|NPU|CPU|GPU] [--no-launch]

The mod logs each country's state on the 1st of every month (on_actions/jevai.txt: JEV|S/T/G/H/N/E/A lines); the runner
rebuilds the model's state text from them. It compiles the model for the NPU while the game loads (the first compile
takes ~2 min, later starts load it from cache). Every `--every` game months it scores all six postures for the AI
countries the player chose in the start-of-game event and writes them to the mod's order file,
<userdir>/mod/jevai/history/units/JEVAI_orders.txt; the mod reloads it weekly with load_oob and applies each posture
through its scripted effects. At startup it also makes JevAI load after every installed overhaul mod (see
patch_load_order). The runner exits when a game it launched closes; --no-launch attaches to a game started from the
Paradox launcher.
"""
from __future__ import annotations

import argparse
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


def launch_game(exe: str, userdir: str, extra: list[str]) -> subprocess.Popen:
    """Start HOI4 without its launcher, so the mod set is the one the launcher last saved in dlc_load.json."""
    args = [exe, "-nolauncher", *extra]
    if os.path.normcase(os.path.abspath(userdir)) != os.path.normcase(default_userdir()):
        args.append(f"-userdir={userdir}")
    return subprocess.Popen(args, cwd=os.path.dirname(exe))


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
    """The mod's JEV lines, read from <userdir>/logs/game.log as the game writes them, grouped by game date. The game
    recreates the file when it starts, so a file shorter than what was read means a new session."""

    def __init__(self, userdir: str):
        self.path = os.path.join(userdir, "logs", "game.log")
        self.pos, self.mode, self.months, self.last = 0, None, {}, 0.0

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.pos:
            self.pos, self.mode, self.months = 0, None, {}
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
                self.months.setdefault(tuple(map(int, m.groups()[:3])), []).append(p)
                self.last = time.time()

    def complete(self, quiet: float = 5.0) -> list[tuple[str, list[list[str]]]]:
        """Months whose burst of lines has ended: a later month has begun, or nothing new came for `quiet` seconds."""
        done = sorted(self.months)
        if done and time.time() - self.last < quiet:
            done = done[:-1]
        return [(".".join(map(str, d)), self.months.pop(d)) for d in done]


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
    lines = [f"# JevAI orders for {date} ({device}); rewritten by the runner, reloaded weekly by the mod\n",
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="JevAI runner: the model picks AI postures in your HOI4 game")
    ap.add_argument("--game", default=None, help="path to hoi4.exe (default: found through Steam)")
    ap.add_argument("--no-launch", action="store_true", help="attach to a game started from the Paradox launcher")
    ap.add_argument("--game-args", default="", help="extra HOI4 launch options, space separated")
    ap.add_argument("--userdir", default=default_userdir())
    ap.add_argument("--model", default=os.path.join(here(), "model"))
    ap.add_argument("--device", default="auto", choices=["auto", "NPU", "CPU", "GPU"])
    ap.add_argument("--every", type=int, default=3, help="decide every N game months")
    ap.add_argument("--horizon", type=int, default=6, help="months the posture questions look ahead")
    ap.add_argument("--min-factories", type=int, default=20, help="skip countries with fewer military + civilian factories")
    ap.add_argument("--player", default="", help="extra tag(s) never to steer, comma separated (humans are excluded anyway)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # mod names can be in any script; a redirected cp1252 stdout must not crash
    out_dir = os.path.join(a.userdir, "mod", "jevai", "history", "units")
    if not os.path.isdir(out_dir):
        sys.exit(f"JevAI mod not found in {os.path.join(a.userdir, 'mod', 'jevai')}; install it first")
    deps = patch_load_order(a.userdir)
    if deps:
        print(f"JevAI loads after {len(deps)} overhaul mods (from the next game start): {', '.join(deps)}", flush=True)

    exe = a.game or find_game()
    game = None
    if not a.no_launch:
        if not exe:
            sys.exit("hoi4.exe not found through Steam; pass --game <path to hoi4.exe> (or --no-launch)")
        game = launch_game(exe, a.userdir, a.game_args.split())
        print(f"started HOI4 (pid {game.pid}); preparing the model while it loads", flush=True)
    ready: dict = {}

    def prepare():  # the model compiles while the game loads
        try:
            ready["names"] = english_names(([os.path.dirname(exe)] if exe else []) + active_mods(a.userdir))
            ready["model"] = Model(a.model, a.device)
        except Exception as e:  # noqa: BLE001 - reported below; the game keeps running without the model
            ready["error"] = e

    loader = threading.Thread(target=prepare, daemon=True)
    loader.start()
    log = GameLog(a.userdir)
    print(f"reading {log.path}", flush=True)
    model = names_en = None
    while True:
        if game is not None and game.poll() is not None:
            print("HOI4 closed; JevAI runner exits", flush=True)
            return
        log.poll()
        if model is None:
            if "error" in ready:
                sys.exit(f"model failed to load: {ready['error']}")
            if loader.is_alive():
                time.sleep(1)
                continue
            model, names_en = ready["model"], ready["names"]
            log.complete(quiet=0)  # months logged while the model compiled are history; decide from the next one on
        for date, lines in log.complete():
            y, mo = map(int, date.split(".")[:2])
            if (y * 12 + mo) % a.every:
                continue
            if log.mode in (None, "off"):
                print(f"[{time.strftime('%H:%M:%S')}] {date}: " + ("waiting for the JevAI choice in the start-of-game event"
                      if log.mode is None else "JevAI is off for this game (vanilla AI)"), flush=True)
                continue
            recs, names, humans, majors = log_records(lines, names_en, date)
            humans |= set(filter(None, a.player.split(",")))
            elig = [t for t, r in recs.items() if t not in humans and r["mil"] + r["civ"] >= a.min_factories
                    and (log.mode == "all" or t in majors)]
            t0 = time.time()
            chosen = {t: int(np.argmax(model.scores(state_text(recs[t], names, recs), a.horizon))) + 1 for t in elig}
            write_atomic(os.path.join(out_dir, ORDERS + ".txt"), orders_file(chosen, date, model.device))
            top = sorted(chosen.items(), key=lambda kv: -(recs[kv[0]]["mil"] + recs[kv[0]]["civ"]))[:6]
            print(f"[{time.strftime('%H:%M:%S')}] {date} ({log.mode}): {len(chosen)} countries in {time.time() - t0:.0f}s on "
                  f"{model.device}; " + ", ".join(f"{t} {POSTURES[k - 1]}" for t, k in top), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
