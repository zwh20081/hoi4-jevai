r"""The JevAI runner: waits in the background for Hearts of Iron IV, reads the mod's state lines from game.log, asks the
model which posture each AI country should follow, and hands the choices to the mod. No Python install, launch option,
dump or console input needed; players only use the Paradox launcher. Ships as jevai.exe.

    jevai.exe --install      copy the runner to %LOCALAPPDATA%\jevai, start it with Windows (hidden) and right away
    jevai.exe --uninstall    remove that and stop the background runner
    jevai.exe                run in this window: wait for HOI4, steer it while it runs, repeat

At the start of a game the player picks the range (major powers or all AI countries) and the period (every week, month
or 3 months; events/jevai.txt). The mod logs each country's state to game.log (scripted_effects/jevai_state.txt:
JEV|S/T/G/H/N/E/A lines): every country on the 1st of each month and, with the weekly period, the countries in range
every week. The model is compiled before any game (prepare: at install, when the runner starts and again when a game
starts; NPU if present and it compiles within the time limit, else CPU, each in a child process that can be stopped),
so when HOI4 starts the runner loads it from the cache in seconds. It rebuilds the model's state text from each update
and, once the period has passed since its last decision, decides for the countries in range. It writes the choices to
history/units/JEVAI_orders.txt in the JevAI copy the game uses (the one enabled in the launcher's playset); the mod
reloads that file daily with load_oob and applies each posture through its scripted effects. When the game closes it
frees the model and waits for the next one. It also keeps JevAI loading after every installed overhaul mod (patch_load_order), finds HOI4's user folder in
the Windows Documents folder wherever OneDrive or a folder move put it, and says in its log why a game stays silent.

The installed runner is a copy outside the mod folder, so it never locks the mod's files and Steam can update the
Workshop item; when HOI4 starts and the mod's jevai.exe differs from the copy, the copy reinstalls from it.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zlib

import numpy as np

from . import multiplayer, steamlobby
from .pack import Packer
from .postures import NAMES as POSTURES, VERSION
from .text import GROWTH_LEVELS, posture_questions, state_text

ORDERS = "JEVAI_orders"  # load_oob name: history/units/JEVAI_orders.txt inside the mod
NAME = "JevAI"  # the mod's name in its descriptor: how the launcher's .mod files for it are found
IDEOLOGY = {"fascism": "fascist", "communism": "communist", "democratic": "democratic", "neutrality": "non-aligned"}
LINE = re.compile(r"^\[[^\]]*\]\[(\d+)\.(\d+)\.(\d+)\.\d+\]\[[^\]]*\]: (.*?)\s*$")  # a game.log line with a game date
KINDS = {"S", "T", "G", "H", "N", "E", "A", "MAJOR"}
PERIOD_DAYS = {"week": 7, "month": 28, "quarter": 89}  # fewest game days between decisions (months have 28 to 31)
LIMIT = {"NPU": 600, "CPU": 300}  # seconds a compile may take before JevAI stops it (NPU ~2.5 min on a Core Ultra 300)
PROGRESS = 60  # seconds between "still compiling" lines
MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)  # HOI4's calendar has no leap years
MOD_NAME = re.compile(r'^\s*name\s*=\s*"([^"]+)"', re.M)
LOC_NAME = re.compile(r'^\s*([A-Z][A-Z0-9]{2}(?:_(?:fascism|communism|democratic|neutrality))?):\d*\s*"(.*)"\s*$')
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW for helper processes
HOME = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "jevai")  # runner copy, NPU cache, log


def default_userdir() -> str:
    """HOI4's user folder in the Windows Documents folder, found the way the game finds it: OneDrive or a folder move
    can put Documents elsewhere than %USERPROFILE%\\Documents."""
    docs, path = os.path.join(os.path.expanduser("~"), "Documents"), ctypes.c_wchar_p()
    folder = ctypes.create_string_buffer(uuid.UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7").bytes_le)  # FOLDERID_Documents
    if ctypes.windll.shell32.SHGetKnownFolderPath(folder, 0, None, ctypes.byref(path)) == 0:
        docs = path.value
    ctypes.windll.ole32.CoTaskMemFree(path)
    return os.path.join(docs, "Paradox Interactive", "Hearts of Iron IV")


def game_day(date: str) -> int:
    """A game date "y.m.d" as a day count."""
    y, m, d = map(int, date.split("."))
    return 365 * y + MONTH_START[m - 1] + d


def due(period: str, last: int | None, day: int) -> bool:
    """Whether the update of `day` gets a decision: the first one, one after loading an earlier save, and then each one
    once the period has passed since the last decision."""
    return last is None or day < last or day - last >= PERIOD_DAYS.get(period, PERIOD_DAYS["month"])


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


def install(mod: str, model: str | None = None):
    """Copy the runner (not the model) to HOME/runner, add a Startup-folder shortcut that runs the copy hidden for this
    mod folder at every logon, compile the model into the cache (prepare, with progress; a reinstall tries the NPU
    again), and start the runner."""
    print(f"\n=== JevAI install {time.strftime('%Y-%m-%d %H:%M:%S')} from {mod}", flush=True)
    stop_others()
    src, dst = here(), os.path.join(HOME, "runner")
    if os.path.normcase(src) != os.path.normcase(dst):
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("model", "*.log", "*.cmd"), dirs_exist_ok=True)
    exe = os.path.join(dst, "jevai.exe")
    q = lambda s: s.replace("'", "''")  # noqa: E731 - PowerShell single-quoted string
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{q(STARTUP_LINK)}');$s.TargetPath='{q(exe)}';"
                    f"$s.Arguments='--hidden --mod \"{q(mod)}\"';$s.WorkingDirectory='{q(dst)}';$s.WindowStyle=7;$s.Save()"],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    if os.path.exists(_npu_skip_path()):
        os.remove(_npu_skip_path())
    print("preparing the model: about 2 minutes on an Intel NPU the first time, seconds on a CPU", flush=True)
    device = prepare(model or os.path.join(mod, "runner", "model"))
    os.startfile(STARTUP_LINK)
    print(f"JevAI is installed in {dst}, starts with Windows and is running now"
          + (f", with the model on the {device}" if device else "; the model did not load (see above)")
          + ". Just play from the Paradox launcher.", flush=True)


def uninstall():
    if os.path.exists(STARTUP_LINK):
        os.remove(STARTUP_LINK)
    stop_others()

    def writable(func, path, exc):  # OpenVINO writes its cache blobs read-only
        if not isinstance(exc, FileNotFoundError):
            os.chmod(path, stat.S_IWRITE)
            func(path)

    for d in ("runner", "ov_cache"):  # only what the runner made: HOME can also hold the collector's inst/
        shutil.rmtree(os.path.join(HOME, d), onexc=writable)
    print("JevAI no longer starts with Windows; the background runner, its copy and its model cache are removed.")


def pids_of_others(image: str) -> list[int]:
    return [p for p in pids(image) if p != os.getpid()]


def stop_others():
    """Stop every other jevai.exe (the background runner) and wait until they are gone and their files unlocked."""
    for pid in pids_of_others("jevai.exe"):
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=NO_WINDOW)
    for _ in range(40):
        if not pids_of_others("jevai.exe"):
            return
        time.sleep(0.25)


def updated_runner(mod: str) -> str | None:
    """The mod's jevai.exe if this process is the installed copy and the mod's runner has changed since (a Workshop
    update): copytree keeps size and mtime, so any difference means a new version."""
    src = os.path.join(mod, "runner", "jevai.exe")
    if not getattr(sys, "frozen", False) or not os.path.isfile(src) or os.path.normcase(here()) == os.path.normcase(os.path.dirname(src)):
        return None
    a, b = os.stat(src), os.stat(sys.executable)
    return src if (a.st_size, int(a.st_mtime)) != (b.st_size, int(b.st_mtime)) else None


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
    """A .mod file's folder in one Windows spelling: the launcher writes forward slashes, and OpenVINO's cache key is the
    model path as written, so D:/x and D:\\x would compile the model twice."""
    p = re.search(r'^\s*path\s*=\s*"([^"]+)"', descriptor_text, re.M)
    return None if not p else os.path.normpath(p.group(1) if os.path.isabs(p.group(1)) else os.path.join(userdir, p.group(1)))


def enabled_mods(userdir: str) -> list[str]:
    """The launcher's .mod files enabled in dlc_load.json (the playset the game starts with), in its order."""
    try:
        files = json.loads(_read(os.path.join(userdir, "dlc_load.json"))).get("enabled_mods", [])
    except ValueError:
        return []
    return [os.path.normcase(os.path.normpath(os.path.join(userdir, f))) for f in files]


def patch_load_order(userdir: str, mod: str) -> list[str]:
    """Overhaul mods list folders under replace_path, which drops those folders (on_actions, events, scripted effects,
    history/units) from every mod loaded before them, and mods load alphabetically unless a dependency says otherwise.
    Declare only enabled replace_path mods as dependencies so JevAI loads after them. Write the descriptor.mod of every
    JevAI copy (--mod, and each one a launcher .mod file named JevAI points to) and those .mod files (jevai.mod, or
    ugc_<id>.mod for the Workshop item). The launcher and the game read the change at their next start."""
    names, ours = set(), [os.path.join(mod, "descriptor.mod")]
    enabled = set(enabled_mods(userdir))
    for f in glob.glob(os.path.join(userdir, "mod", "*.mod")):
        text = _read(f)
        name = MOD_NAME.search(text)
        if not name:
            continue
        root = _mod_root(userdir, text)
        if name.group(1) == NAME:
            ours += [f] + ([os.path.join(root, "descriptor.mod")] if root else [])
            continue
        if os.path.normcase(os.path.normpath(f)) not in enabled:
            continue
        if "replace_path" in text or (root and "replace_path" in _read(os.path.join(root, "descriptor.mod"))):
            names.add(name.group(1))
    block_re = re.compile(r"\n?dependencies\s*=\s*\{([^}]*)\}\s*")
    block = ("dependencies = {\n" + "".join(f'\t"{n}"\n' for n in sorted(names)) + "}\n") if names else ""
    for f in {os.path.normcase(os.path.abspath(p)): p for p in ours}.values():
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
    return [r for r in (_mod_root(userdir, _read(f)) for f in enabled_mods(userdir)) if r]


def jevai_mods(userdir: str) -> list[tuple[str, str | None, bool]]:
    """The launcher's .mod files named JevAI (a Workshop subscription, a local copy): file, mod folder, enabled."""
    enabled, out = set(enabled_mods(userdir)), []
    for f in sorted(glob.glob(os.path.join(userdir, "mod", "*.mod"))):
        text = _read(f)
        name = MOD_NAME.search(text)
        if name and name.group(1) == NAME:
            out.append((f, _mod_root(userdir, text), os.path.normcase(os.path.normpath(f)) in enabled))
    return out


def game_mod(a) -> str:
    """The JevAI folder the game uses (the enabled .mod named JevAI), else --mod. Says when JevAI is not in the
    playset, or when several .mod files carry its name: HOI4 then skipped the enabled Workshop copy in a test while a
    local jevai.mod of the same name existed (temp/checks/modtest.py)."""
    mods = jevai_mods(a.userdir)
    if len(mods) > 1:
        print(f"{len(mods)} mods are named {NAME} ({', '.join(os.path.basename(f) for f, _, _ in mods)}): HOI4 can skip "
              f"the enabled one; keep one, or rename the others in their .mod and descriptor.mod files", flush=True)
    for _, root, on in mods:
        if on and root and os.path.isdir(os.path.join(root, "history", "units")):
            return root
    print(f"{NAME} is not enabled in the launcher's playset; enable it there and start the game again", flush=True)
    return a.mod


def active_mod_names(userdir: str) -> list[str] | None:
    """The mods HOI4 runs this session, from system.log's Active Mod lines; None until it has logged them."""
    text = _read(os.path.join(userdir, "logs", "system.log"))
    return re.findall(r"\]: Active Mod: (.*?)\s*$", text, re.M) if "Active Mod Count:" in text else None


def silence(userdir: str, log: GameLog, waited: float) -> str | None:
    """Why the game logs no JevAI lines, once that is clear: no game.log a minute after HOI4 started, HOI4 not running
    JevAI (system.log lists the active mods before the game loads), or JevAI active while the game ran two days without
    a JEV line (an overhaul mod loaded after it and its replace_path dropped JevAI's scripts)."""
    if not os.path.isfile(log.path):
        return (f"no game log at {log.path} a minute after HOI4 started: its user folder is elsewhere; start jevai.exe "
                f'with --userdir "<that folder>"') if waited > 60 else None
    if log.jev or not log.days:
        return None
    active = active_mod_names(userdir)
    if active is not None and NAME not in active:
        return (f"HOI4 did not load {NAME} in this game (its system.log lists {len(active)} active mods, not {NAME}): "
                f"enable {NAME} in the launcher's playset, keep only one mod named {NAME}, and start the game again")
    if len(log.days) < 3:
        return None
    if active is not None:
        return (f"{NAME} is loaded, but its scripts do not run: an overhaul mod replaces them and loads after it. {NAME} "
                "now loads after the overhaul mods it found; restart the Paradox launcher, then the game")
    return (f"the game has run 2 days without a {NAME} line: HOI4 did not load {NAME} (check the launcher's playset), or "
            "an overhaul mod loaded after it and replaced its scripts (restart the Paradox launcher)")


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


def day_date(day: int) -> str:
    """game_day's inverse: the game date "y.m.d" of a day count."""
    y, r = divmod(day - 1, 365)
    m = max(i for i, s in enumerate(MONTH_START) if s <= r)
    return f"{y}.{m + 1}.{r - MONTH_START[m] + 1}"

class GameLog:
    """The mod's JEV lines, read from <userdir>/logs/game.log as the game writes them. Lines come in bursts (every
    country on the 1st of each month; the countries in range weekly with the weekly period), grouped here by game date.
    `latest` keeps each country's most recent lines, so a weekly burst of major powers still sees last month's
    neighbours. The game recreates the file when it starts, so a file shorter than what was read means a new session.
    `mode` and `period` are the player's choices (JEV|MODE, JEV|PERIOD); `days` (up to 3 game dates seen) and `jev` (a
    JEV line seen) tell a running game that logs nothing for JevAI."""

    def __init__(self, userdir: str):
        self.path = os.path.join(userdir, "logs", "game.log")
        self.last = 0.0
        self.reset()

    def reset(self):
        self.pos, self.mode, self.period, self.bursts, self.latest, self.days, self.jev = 0, None, None, {}, {}, set(), False
        self.now, self.multiplayer = 0, None
        self.generation = getattr(self, "generation", 0) + 1

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.pos:
            self.reset()
        if size == self.pos:
            return
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read(size - self.pos)
        end = chunk.rfind(b"\n") + 1  # a line still being written waits for the next poll
        self.pos += end
        for line in chunk[:end].decode("utf-8", "replace").splitlines():
            if "[[ Launching " in line:
                pos = self.pos
                self.reset()
                self.pos = pos
                self.multiplayer = "MULTIPLAYER" in line
                continue
            m = LINE.match(line)
            if not m:
                continue
            date = tuple(map(int, m.groups()[:3]))
            day = 365 * date[0] + MONTH_START[date[1] - 1] + date[2]
            if day < self.now:
                pos, kind = self.pos, self.multiplayer
                self.reset()
                self.pos, self.multiplayer = pos, kind
            self.now = day
            if len(self.days) < 3:
                self.days.add(date)
            if not m.group(4).startswith("JEV|"):
                continue
            self.jev = True
            p = m.group(4).split("|")
            if len(p) < 3:
                continue
            if p[1] == "MODE":
                self.mode = p[2]
            elif p[1] == "PERIOD":
                self.period = p[2]
            elif p[1] in KINDS:
                self.bursts.setdefault(date, []).append(p)
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
            tags = rec(tag)[{"N": "neighbors", "E": "enemies", "A": "allies"}[kind]]
            if p[3] not in tags:  # a day with both the weekly and the monthly update logs a country twice
                tags.append(p[3])
    for tag, r in recs.items():
        g = group.get(tag)
        if g in IDEOLOGY:
            r["jev"]["ideo"] = IDEOLOGY[g]  # the training text's words, whatever the game language
        r["name"] = english.get(f"{tag}_{g}") or english.get(tag) or r["name"]
    # countries without states log no S line; name them from the localisation (enemies in exile, for example)
    names = {t: english.get(t, t) for r in recs.values() for t in r["enemies"] + r["allies"] + r["neighbors"]}
    names.update({t: r["name"] for t, r in recs.items()})
    return {t: r for t, r in recs.items() if r["n_states"] > 0}, names, humans, majors


def compile_graph(core, model_dir: str, device: str) -> tuple[dict, str, object]:
    """The model's config, its graph for `device` and an infer request, compiled through HOME/ov_cache with the settings
    every caller uses: the child's compile (prepare) and the runner's load (Model) share one cache entry."""
    with open(os.path.join(model_dir, "jev.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cache = os.path.join(HOME, "ov_cache")
    os.makedirs(cache, exist_ok=True)
    core.set_property({"CACHE_DIR": cache})  # the NPU compile happens once per model and driver
    graph = cfg["graphs"].get(device, cfg["graphs"]["CPU"])
    path = os.path.normpath(os.path.abspath(os.path.join(model_dir, graph)))  # the cache key is this string
    return cfg, graph, core.compile_model(path, device, {"PERFORMANCE_HINT": "LATENCY"}).create_infer_request()


def compile_only(device: str, model_dir: str) -> int:
    """Child mode (jevai.exe --compile DEVICE): compile the model for DEVICE into the cache and print one JSON line for
    the runner (prepare): ok and seconds, missing (no such device) or the error."""
    t = time.time()
    try:
        import openvino as ov
        core = ov.Core()
        if device not in core.available_devices:
            print(json.dumps({"ok": False, "missing": True}), flush=True)
            return 2
        compile_graph(core, model_dir, device)
    except Exception as e:  # noqa: BLE001 - reported to the runner
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps({"ok": True, "seconds": round(time.time() - t, 1)}), flush=True)
    return 0


def me() -> list[str]:
    """This runner again, as a child process: jevai.exe itself when frozen, else this module."""
    return [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", __spec__.name]


def child_command(model_dir: str, device: str) -> list[str]:
    return me() + ["--compile", device, "--model", model_dir]


def compile_child(model_dir: str, device: str, limit: float) -> tuple[bool, str]:
    """Compile the model for `device` in a child process, which (unlike a compile in this process) can be stopped:
    (True, how long) or (False, why; "missing" when there is no such device). Says every PROGRESS seconds that it is
    still compiling, and stops the child after `limit` seconds."""
    os.makedirs(os.path.join(HOME, "ov_cache"), exist_ok=True)
    out = os.path.join(HOME, "ov_cache", f"compile_{device}.txt")
    with open(out, "w", encoding="utf-8") as f:
        p = subprocess.Popen(child_command(model_dir, device), stdout=f, stderr=subprocess.STDOUT,
                             creationflags=NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    span = lambda s: f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f}s"  # noqa: E731
    t = time.time()
    note = t + PROGRESS
    while p.poll() is None:
        time.sleep(1)
        if time.time() - t > limit:
            p.kill()
            p.wait()
            return False, f"not done after {span(limit)}, stopped"
        if time.time() >= note:
            print(f"still compiling the model for the {device}: {span(time.time() - t)} (JevAI stops at {span(limit)})",
                  flush=True)
            note += PROGRESS
    lines = _read(out).strip().splitlines() or [f"exit code {p.returncode}, no output"]
    try:
        r = json.loads(lines[-1])
    except ValueError:
        return False, lines[-1]
    return (True, f"{r.get('seconds', 0):.0f}s") if r.get("ok") else (False, "missing" if r.get("missing") else r.get("error", "?"))


def ov_version() -> str:
    try:
        from importlib.metadata import version
        return version("openvino")
    except Exception:  # noqa: BLE001 - only part of a cache key
        return "?"


def _model_key(model_dir: str) -> str:
    """This model's NPU graph (folder, size, time) and the OpenVINO version: a new model or runtime tries the NPU again."""
    try:
        with open(os.path.join(model_dir, "jev.json"), encoding="utf-8") as f:
            graph = json.load(f)["graphs"]["NPU"]
        st = os.stat(os.path.join(model_dir, os.path.splitext(graph)[0] + ".bin"))
        stamp = f"{st.st_size}|{int(st.st_mtime)}"
    except (OSError, ValueError, KeyError):
        stamp = "?"
    return f"{os.path.normcase(os.path.abspath(model_dir))}|{stamp}|{ov_version()}"


def _npu_skip_path() -> str:
    return os.path.join(HOME, "ov_cache", "npu_skip.json")


def prepare(model_dir: str, device: str = "auto") -> str | None:
    """Compile the model into the cache and return the device the runner will use (None: none worked). With "auto" the
    NPU comes first unless it failed or timed out before for this model and OpenVINO (npu_skip.json), then the CPU.
    Each device compiles in a child process stopped after LIMIT; from a warm cache this takes seconds, so the runner
    does it at install, when it starts and when a game starts, and the model is ready before HOI4 has loaded."""
    if not os.path.isfile(os.path.join(model_dir, "jev.json")):
        print(f"no model in {model_dir}", flush=True)
        return None
    try:
        skip = json.loads(_read(_npu_skip_path()) or "{}")
    except ValueError:
        skip = {}
    key = _model_key(model_dir)
    if device != "auto":
        tries = [device]
    elif key in skip:
        tries = ["CPU"]
        print(f"the NPU failed before for this model ({skip[key]}); using the CPU (jevai.exe --device NPU retries)", flush=True)
    else:
        tries = ["NPU", "CPU"]
    for dev in tries:
        ok, detail = compile_child(model_dir, dev, LIMIT.get(dev, LIMIT["NPU"]))
        if ok:
            print(f"model prepared on {dev} in {detail}", flush=True)
            if dev == "NPU" and skip.pop(key, None):
                write_atomic(_npu_skip_path(), json.dumps(skip, indent=1))
            return dev
        if detail == "missing":
            continue
        print(f"the model could not be prepared on {dev}: {detail}", flush=True)
        if dev == "NPU" and device == "auto":
            skip[key] = f"{detail}, {time.strftime('%Y-%m-%d')}"
            write_atomic(_npu_skip_path(), json.dumps(skip, indent=1))
            print("later games use the CPU for this model; jevai.exe --device NPU retries the NPU", flush=True)
    return None


class Model:
    """The exported IR on one device, loaded from the cache that prepare filled; if the NPU fails here anyway, the CPU."""

    def __init__(self, model_dir: str, device: str = "auto"):
        import openvino as ov
        core = ov.Core()
        if device == "auto":
            device = "NPU" if "NPU" in core.available_devices else "CPU"
        tries = [device] + (["CPU"] if device == "NPU" else [])
        for i, dev in enumerate(tries):
            print(f"loading the model on {dev}", flush=True)
            t = time.time()
            try:
                self.cfg, graph, self.req = compile_graph(core, model_dir, dev)
            except Exception as e:  # noqa: BLE001 - try the next device; the last one's error goes to the caller
                if i == len(tries) - 1:
                    raise
                print(f"the model failed on {dev}: {e}", flush=True)
                continue
            self.device = dev
            print(f"model on {dev} ({graph}) ready in {time.time() - t:.0f}s", flush=True)
            break
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


def orders_file(postures: dict[str, int], date: str, device: str, target: str | None = None) -> str:
    """An OOB file whose instant_effect sets each country's wanted posture; the mod applies it to AI countries.
    It holds no units, so loading it changes nothing but those variables. The set takes effect from `target` (default:
    the update's own day, at once): every game of a multiplayer session reloads the file daily and applies the set
    on the same game day, since the postures are synchronized state."""
    lines = [f"# JevAI orders for {date} ({device}), from {target or date}; rewritten by the runner, reloaded daily by "
             "the mod\n", "instant_effect = {\n"]
    if postures:
        lines.append(f"\tif = {{ limit = {{ date > {day_date(game_day(target or date) - 1)} }}\n")
        for tag, k in sorted(postures.items()):
            lines.append(f"\t\tif = {{ limit = {{ country_exists = {tag} }} {tag} = {{ set_variable = {{ jev_want = {k} }} }} }}\n")
        lines.append("\t}\n")
    lines.append("}\n")
    return "".join(lines)


def write_atomic(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def default_steam_api() -> str:
    """The game's own steam_api64.dll, or "" when HOI4 is not in a Steam library."""
    exe = find_game()
    return os.path.join(os.path.dirname(exe), "steam_api64.dll") if exe else ""


def start_relay(a, role: dict, orders: str) -> subprocess.Popen:
    """The Steam relay of this multiplayer game (jevai.exe --relay), stopped when the game closes."""
    dll = role.get("steam_api") or a.steam_api or default_steam_api()
    cmd = me() + ["--relay", role["role"], "--orders", orders, "--steam-api", dll]
    return subprocess.Popen(cmd + (["--code", role["code"]] if role.get("code") else []),
                            creationflags=NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)


def publish(lobby, orders: str, last: str | None) -> str | None:
    """Host relay, one poll: put the orders file in the lobby when it changed; returns what the lobby holds."""
    text = _read(orders)
    if text and text != last:
        if not lobby.set("orders", multiplayer.pack(text)):
            raise steamlobby.SteamError("Steam refused the orders update")
        return text
    return last


def fetch(lobby, orders: str, last: str | None) -> str | None:
    """Client relay, one poll: write the host's newest orders file when it changed; returns the last set seen."""
    wire = lobby.get("orders")
    if not wire or wire == last:
        return last
    write_atomic(orders, multiplayer.unpack(wire))
    return wire


def relay(role: str, orders: str, code: str | None, dll: str) -> int:
    """Child mode (jevai.exe --relay host|client): the only process that loads steam_api64.dll, so a replaced or old DLL
    can only crash this one. The host's relay puts each new orders file in its lobby; a client's relay finds the host's
    lobby by code and writes each set to its own orders file, byte for byte. Runs until the runner stops it; logs to
    jevai.log."""
    os.makedirs(HOME, exist_ok=True)
    with open(os.path.join(HOME, "jevai.log"), "a", encoding="utf-8") as out:
        def say(msg: str):
            out.write(f"[{time.strftime('%H:%M:%S')}] relay: {msg}\n")
            out.flush()

        try:
            lobby = steamlobby.Lobby(dll)
            lobby.open()
            if role == "host":
                code = multiplayer.code_for(lobby.account_id())
                lobby.create(code)
                say(f"hosting; the other players run jevai.exe --join {code}")
            else:
                if not multiplayer.valid_code(code or ""):
                    raise ValueError("invalid host pairing code")
                code = code.upper()
                waited = False
                while not (found := lobby.find(code)):
                    if not waited:
                        say(f"waiting for the host {code} (its JevAI opens the lobby when its game starts)")
                        waited = True
                    time.sleep(10)
                lobby.join(found)
                say(f"following the host {code}")
            last = None
            while True:
                lobby.tick()
                if multiplayer.code_for(lobby.owner()) != code:
                    raise steamlobby.SteamError("lobby owner no longer matches the pairing code")
                new = publish(lobby, orders, last) if role == "host" else fetch(lobby, orders, last)
                if new != last:
                    say(("sent " if role == "host" else "received ") + _read(orders).split("\n", 1)[0].lstrip("# "))
                    last = new
                time.sleep(1)
        except (steamlobby.SteamError, OSError, ValueError, RuntimeError, AttributeError, zlib.error) as e:
            say(f"stopped: {e}")
            return 1


class Tee:
    """Runner output goes to the console and to HOME/jevai.log, readable after the window closes."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def model_dir(a, mod: str) -> str:
    """--model, else the runner/model of the JevAI copy the game uses, else the one of --mod."""
    d = os.path.join(mod, "runner", "model")
    return a.model or (d if os.path.isfile(os.path.join(d, "jev.json")) else os.path.join(a.mod, "runner", "model"))


def session(a, mod: str, pid: int):
    """Steer one running game until its process `pid` exits: once the game session starts, load the English names and
    the model (prepared into the cache already, normally: seconds), then decide on the updates the mod logs, once per
    period the player chose. In a multiplayer game only the host (jevai.exe --host) decides: every order set applies
    multiplayer.MARGIN_DAYS after the host's newest game date, and the relay carries it to the other players' games.
    Says once why, if the game logs nothing for JevAI."""
    orders = os.path.join(mod, "history", "units", ORDERS + ".txt")
    write_atomic(orders, orders_file({}, "no update yet", "-"))  # not the last game's
    log = GameLog(a.userdir)
    role = multiplayer.load_role(HOME)
    print(f"reading {log.path}; orders go to {orders}", flush=True)
    ready: dict = {}

    def load():
        try:
            exe = find_game()
            ready["names"] = english_names(([os.path.dirname(exe)] if exe else []) + active_mods(a.userdir))
            md = model_dir(a, mod)
            device = prepare(md, a.device)  # a warm cache: seconds; else compiles within the time limits
            if device is None:
                raise RuntimeError("no device could run the model (see above)")
            ready["model"] = Model(md, device)
        except Exception as e:  # noqa: BLE001 - reported below; the game keeps running without the model
            ready["error"] = e

    loader = proc = None
    model = names_en = last = None  # last: game day of the last decision
    per_country = 0.0  # seconds, from the last batch
    started = checked = time.time()
    explained = told = False
    generation = log.generation
    try:
        while True:
            if time.time() - checked > 5:
                checked = time.time()
                if pid not in pids("hoi4.exe"):
                    return
                if not explained and (why := silence(a.userdir, log, checked - started)):
                    print(why, flush=True)
                    explained = True
            log.poll()
            if log.generation != generation:
                if proc:
                    proc.kill()
                    proc.wait()
                    proc = None
                write_atomic(orders, orders_file({}, "new game session", "-"))
                generation, last, told = log.generation, None, False
                role = multiplayer.load_role(HOME)
            if log.multiplayer is None:  # the menus: no game session yet
                time.sleep(1)
                continue
            if log.multiplayer and role and proc is None:
                proc = start_relay(a, role, orders)
                print("multiplayer game: JevAI " + ("decides for everyone" if role["role"] == "host"
                                                    else f"applies the orders of the host {role['code']}"), flush=True)
            if proc and proc.poll() is not None:
                print(f"multiplayer relay exited ({proc.returncode}); JevAI stops issuing orders for this game",
                      flush=True)
                write_atomic(orders, orders_file({}, "relay stopped", "-"))
                while pid in pids("hoi4.exe"):
                    time.sleep(5)
                return
            if log.multiplayer and (role is None or role["role"] == "client"):
                if loader is not None and not loader.is_alive():
                    ready.clear()
                    model = names_en = loader = None
                if role is None and not told:
                    print("multiplayer game without a JevAI role, so JevAI stays off: the host runs jevai.exe --host, "
                          "the other players jevai.exe --join CODE", flush=True)
                    told = True
                time.sleep(1)
                continue
            if loader is None:
                loader = threading.Thread(target=load, daemon=True)
                loader.start()
            if model is None:
                if "error" in ready:
                    print(f"model failed to load: {ready['error']}; JevAI sits out this game", flush=True)
                    while pid in pids("hoi4.exe"):
                        time.sleep(5)
                    return
                if loader.is_alive():
                    time.sleep(1)
                    continue
                model, names_en = ready["model"], ready["names"]
                per_country = 1.5 if model.device == "NPU" else 10.0
                log.complete(quiet=0)  # what was logged before the model was ready is history; decide from the next update
                print("waiting for the game's next update (the 1st of each month, or weekly with the weekly period)",
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
            period = log.period or ("week" if log.mode == "majors" else "month")  # saves from JevAI 0.2 log no period
            if not due(period, last, game_day(date)):
                continue
            recs, names, humans, majors = log_records([p for ls in log.latest.values() for p in ls], names_en, date)
            humans |= set(filter(None, a.player.split(",")))
            elig = [t for t in reported if t in recs and t not in humans and recs[t]["mil"] + recs[t]["civ"] >= a.min_factories
                    and (log.mode == "all" or t in majors)]
            if not elig:
                continue
            head = f"{now} ({log.mode}, {period})"
            if (est := len(elig) * per_country) > 20:  # a CPU takes minutes for a big batch: say so before it starts
                print(f"{head}: deciding {len(elig)} countries on {model.device}, about "
                      + (f"{est:.0f}s" if est < 120 else f"{est / 60:.0f} min"), flush=True)
            t0 = time.time()
            chosen = {}
            batch_generation = log.generation
            for t in elig:
                horizon = a.horizon if a.horizon is not None else getattr(model, "cfg", {}).get("horizon", 6)
                u = model.scores(state_text(recs[t], names, recs), horizon)
                best, cur = int(np.argmax(u)), int(recs[t]["jev"].get("pos") or 0) - 1  # cur: posture in force, -1 none
                # near-ties flip with tiny changes (even the date): switch only for a clear gain
                chosen[t] = (cur if 0 <= cur < len(u) and u[best] - u[cur] < a.stick else best) + 1
            log.poll()
            if log.generation != batch_generation:
                print("game session changed while deciding; discarded the old batch", flush=True)
                continue
            if log.mode == "off":
                write_atomic(orders, orders_file({}, "JevAI is off", "-"))
                continue
            if proc and proc.poll() is not None:
                continue
            target = day_date(log.now + multiplayer.MARGIN_DAYS) if log.multiplayer else date
            write_atomic(orders, orders_file(chosen, date, model.device, target))
            last, per_country = game_day(date), (time.time() - t0) / len(elig)
            top = sorted(chosen.items(), key=lambda kv: -(recs[kv[0]]["mil"] + recs[kv[0]]["civ"]))[:6]
            skipped = f"; {len(done) - 1} older updates skipped" if len(done) > 1 else ""
            print(f"{head}: {len(chosen)} countries in {time.time() - t0:.0f}s on {model.device}{skipped}; "
                  + ", ".join(f"{t} {POSTURES[k - 1]}" for t, k in top), flush=True)
    finally:
        if proc:
            proc.kill()
            proc.wait()


def set_role(a) -> int:
    """jevai.exe --host / --join CODE: this PC's part in multiplayer games, kept in HOME/multiplayer.json for the running
    JevAI, which reads it when a game starts; single-player games ignore it."""
    if a.join is not None:
        if not a.join:
            try:
                a.join = input("JevAI code from the host: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("No pairing code entered; multiplayer role unchanged")
                return 1
        a.join = a.join.strip()
        if not multiplayer.valid_code(a.join):
            print(f"{a.join} is not a JevAI code (7 letters and digits, printed by the host's jevai.exe --host)")
            return 1
        multiplayer.save_role(HOME, "client", a.join, a.steam_api)
        print(f"multiplayer games: this PC applies the orders of the host {a.join.upper()} (from the next game)")
        return 0
    multiplayer.save_role(HOME, "host", steam_api=a.steam_api)
    lobby = steamlobby.Lobby(a.steam_api or default_steam_api())
    try:
        lobby.open()
        code = multiplayer.code_for(lobby.account_id())
    except (steamlobby.SteamError, OSError, AttributeError) as e:
        print(f"multiplayer games: this PC hosts. Steam did not answer ({e}); the code to share appears in jevai.log "
              "when a multiplayer game starts")
        return 0
    print(f"multiplayer games: this PC hosts. The other players run: jevai.exe --join {code}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="JevAI runner: the model picks AI postures in your HOI4 games")
    ap.add_argument("--install", action="store_true", help="start JevAI with Windows (hidden) and right away")
    ap.add_argument("--uninstall", action="store_true", help="stop starting with Windows and stop the background runner")
    ap.add_argument("--hidden", action="store_true", help="no window (how the Startup shortcut runs it)")
    ap.add_argument("--userdir", default=default_userdir(), help="HOI4's user folder (default: the one in Documents)")
    ap.add_argument("--mod", default=os.path.dirname(here()),
                    help="the JevAI mod folder (default: the one jevai.exe is in); games use the copy enabled in the playset")
    ap.add_argument("--model", help="exported model folder (default: runner/model of the JevAI copy the game uses)")
    ap.add_argument("--device", default="auto", choices=["auto", "NPU", "CPU", "GPU"])
    ap.add_argument("--horizon", type=int, help="months to look ahead (default: model config, or 6 for older models)")
    ap.add_argument("--stick", type=float, default=0.01, help="keep the current posture unless another scores this much higher")
    ap.add_argument("--min-factories", type=int, default=20, help="skip countries with fewer military + civilian factories")
    ap.add_argument("--player", default="", help="extra tag(s) never to steer, comma separated (humans are excluded anyway)")
    ap.add_argument("--compile", metavar="DEVICE", help=argparse.SUPPRESS)  # child mode, see prepare
    roles = ap.add_mutually_exclusive_group()
    roles.add_argument("--host", action="store_true", help="multiplayer: host and print a pairing code")
    roles.add_argument("--join", metavar="CODE", nargs="?", const="", help="multiplayer: join a host (prompts if omitted)")
    ap.add_argument("--steam-api", help="multiplayer: steam_api64.dll path (default: the game's)")
    ap.add_argument("--relay", choices=["host", "client"], help=argparse.SUPPRESS)
    ap.add_argument("--orders", help=argparse.SUPPRESS)
    ap.add_argument("--code", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.horizon is not None and a.horizon <= 0:
        ap.error("--horizon must be positive")
    if a.relay and (not a.orders or (a.relay == "client" and not multiplayer.valid_code(a.code or ""))):
        ap.error("--relay requires --orders and clients require a valid --code")
    a.mod = os.path.abspath(a.mod)
    if a.relay:
        return relay(a.relay, a.orders, a.code, a.steam_api or default_steam_api())
    if a.compile:
        return compile_only(a.compile, a.model or os.path.join(a.mod, "runner", "model"))
    if hasattr(sys.stdout, "reconfigure"):  # a process started without a console has none
        sys.stdout.reconfigure(errors="replace")  # mod names can be in any script; a cp1252 stdout must not crash
    os.makedirs(HOME, exist_ok=True)
    if a.host or a.join is not None:
        try:
            return set_role(a)
        except OSError as e:
            print(f"Could not save multiplayer role: {e}")
            return 1
    sys.stdout = Tee(*filter(None, [sys.stdout]), open(os.path.join(HOME, "jevai.log"), "a", encoding="utf-8"))
    if a.install or a.uninstall:
        if a.uninstall:
            return uninstall()
        if not getattr(sys, "frozen", False):
            sys.exit("--install works from jevai.exe")
        return install(a.mod, a.model)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if a.hidden:
        ctypes.windll.user32.ShowWindow(k32.GetConsoleWindow(), 0)
    mutex = k32.CreateMutexW(None, False, r"Local\JevAIRunner")  # one runner per user
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        sys.exit("JevAI is already running")
    print(f"\n=== JevAI runner started {time.strftime('%Y-%m-%d %H:%M:%S')} for {a.mod}", flush=True)
    if not os.path.isdir(os.path.join(a.mod, "history", "units")):
        sys.exit(f"JevAI mod not found in {a.mod}; run install.cmd from the mod's runner folder")
    role = multiplayer.load_role(HOME)
    if not role or role["role"] != "client":
        prepare(model_dir(a, a.mod), a.device)  # a warm cache before the next game
    n_deps = None
    while True:
        deps = patch_load_order(a.userdir, a.mod)  # before each game, so a newly installed overhaul is covered
        if len(deps) != n_deps:
            n_deps = len(deps)
            print(f"JevAI loads after {n_deps} overhaul mods (read by the launcher at its next start)", flush=True)
        print(f"waiting for HOI4 (start it from the Paradox launcher; user folder {a.userdir})", flush=True)
        while not (running := pids("hoi4.exe")):
            time.sleep(5)
        print(f"[{time.strftime('%H:%M:%S')}] HOI4 started (pid {running[0]})", flush=True)
        mod = game_mod(a)  # the launcher wrote the playset just before starting the game
        if src := updated_runner(mod):  # Steam updated the mod: the new runner reinstalls itself and takes over
            print(f"the mod's runner was updated; reinstalling from {src}", flush=True)
            subprocess.Popen([src, "--install", "--mod", mod], creationflags=NO_WINDOW)
            return
        try:
            session(a, mod, running[0])
        except Exception:  # noqa: BLE001 - the background runner must outlive a bug: log it, sit out this game
            print(traceback.format_exc(), flush=True)
            while running[0] in pids("hoi4.exe"):
                time.sleep(5)
        print(f"[{time.strftime('%H:%M:%S')}] HOI4 closed", flush=True)
    del mutex


if __name__ == "__main__":
    sys.exit(main())
