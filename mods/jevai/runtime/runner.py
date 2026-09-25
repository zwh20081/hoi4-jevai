"""The JevAI runner: starts Hearts of Iron IV, reads its monthly dumps, asks the model which posture each AI country
should follow, and hands the choices to the mod. No Python install or console needed at play time; ships as jevai.exe.

    jevai.exe [--game "...\\Hearts of Iron IV\\hoi4.exe"] [--userdir ...] [--device auto|NPU|CPU|GPU] [--no-launch]

Launch the game through jevai.exe: it starts HOI4 with -dump_history (the launch option that writes
history_dump/N.txt every game month) and compiles the model for the NPU while the game loads, so the ~75 s first
compile overlaps HOI4's own load (later starts load the compiled model from cache). Every `--every` game months it
scores all six postures for every AI country the player does not control and writes them to the mod's order file,
<userdir>/mod/jevai/history/units/JEVAI_orders.txt. The mod reloads that file weekly with load_oob and applies each
country's posture through its scripted effects, so the model steers the AI without any input into the game window.
The runner exits when the game closes. --no-launch attaches to a game started some other way (with -dump_history).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

from .game import load_invariants, month_files, month_records, read_month
from .pack import Packer
from .postures import NAMES as POSTURES, VERSION
from .text import GROWTH_LEVELS, posture_questions, state_text

ORDERS = "JEVAI_orders"  # load_oob name: history/units/JEVAI_orders.txt inside the mod


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
    """Start HOI4 with -dump_history (the runner reads its monthly dumps); the launcher is skipped, so the mod set
    is the one last saved by the launcher in <userdir>/dlc_load.json."""
    args = [exe, "-dump_history", "-nolauncher", *extra]
    if os.path.normcase(os.path.abspath(userdir)) != os.path.normcase(default_userdir()):
        args.append(f"-userdir={userdir}")
    return subprocess.Popen(args, cwd=os.path.dirname(exe))


def here() -> str:
    """Folder of the runner: next to jevai.exe when frozen, else this package."""
    return os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))


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
        core.set_property({"CACHE_DIR": cache})  # the NPU compile (~75 s) happens once per model and driver
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


def game_mode(userdir: str) -> tuple[str | None, set[str]]:
    """The mode the player picked in the start-of-game event (JEV|MODE|all/majors/off, or None before the choice) and
    the major powers (JEV|MAJOR|TAG, logged monthly by the mod), read from game.log."""
    try:
        with open(os.path.join(userdir, "logs", "game.log"), encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None, set()
    modes = re.findall(r"JEV\|MODE\|(all|majors|off)", text)
    # majors of the last two months: enough to cover a month still being written; old majors drop out over time
    majors = set(re.findall(r"JEV\|MAJOR\|([A-Z0-9]{3})", text[-2_000_000:]))
    return (modes[-1] if modes else None), majors


def main(argv=None):
    ap = argparse.ArgumentParser(description="JevAI runner: starts HOI4; the model picks AI postures in your game")
    ap.add_argument("--game", default=None, help="path to hoi4.exe (default: found through Steam)")
    ap.add_argument("--no-launch", action="store_true", help="attach to a game started some other way (with -dump_history)")
    ap.add_argument("--game-args", default="", help="extra HOI4 launch options, space separated")
    ap.add_argument("--userdir", default=default_userdir())
    ap.add_argument("--model", default=os.path.join(here(), "model"))
    ap.add_argument("--device", default="auto", choices=["auto", "NPU", "CPU", "GPU"])
    ap.add_argument("--every", type=int, default=3, help="decide every N game months")
    ap.add_argument("--horizon", type=int, default=6, help="months the posture questions look ahead")
    ap.add_argument("--min-factories", type=int, default=20, help="skip countries with fewer military + civilian factories")
    ap.add_argument("--player", default="", help="tag(s) the player controls, comma separated (read from the dump when empty)")
    a = ap.parse_args(argv)
    dump = os.path.join(a.userdir, "history_dump")
    out_dir = os.path.join(a.userdir, "mod", "jevai", "history", "units")
    if not os.path.isdir(out_dir):
        sys.exit(f"JevAI mod not found in {os.path.join(a.userdir, 'mod', 'jevai')}; install it first")

    # the model compiles while the game loads: start the game first, then compile in the background
    game = None
    if not a.no_launch:
        exe = a.game or find_game()
        if not exe:
            sys.exit("hoi4.exe not found through Steam; pass --game <path to hoi4.exe> (or --no-launch)")
        game = launch_game(exe, a.userdir, a.game_args.split())
        print(f"started HOI4 (pid {game.pid}) with -dump_history; preparing the model while it loads", flush=True)
    ready: dict = {}

    def prepare():
        try:
            ready["model"] = Model(a.model, a.device)
        except Exception as e:  # noqa: BLE001 - reported below; the game keeps running without the model
            ready["error"] = e

    loader = threading.Thread(target=prepare, daemon=True)
    loader.start()
    print(f"watching {dump}", flush=True)
    done, st_center, model = set(), None, None
    while True:
        if game is not None and game.poll() is not None:
            print("HOI4 closed; JevAI runner exits", flush=True)
            return
        if model is None:
            if "error" in ready:
                sys.exit(f"model failed to load: {ready['error']}")
            if loader.is_alive():
                time.sleep(1)
                continue
            model = ready["model"]
            # months the game wrote while the model compiled are history; decide from the newest one on
            done = {os.path.basename(p) for p in (month_files(dump)[:-1] if os.path.isdir(dump) else [])}
        fs = month_files(dump) if os.path.isdir(dump) else []
        if not fs or os.path.basename(fs[-1]) in done:
            time.sleep(2)
            continue
        j = read_month(fs[-1])
        if j is None:
            time.sleep(2)
            continue
        done.add(os.path.basename(fs[-1]))
        n = int(os.path.basename(fs[-1])[:-4])
        if n % a.every:
            continue
        if st_center is None:
            st_center = load_invariants(dump)
        mode, majors = game_mode(a.userdir)
        if mode in (None, "off"):
            print(f"[{time.strftime('%H:%M:%S')}] {j['date']}: " + ("waiting for the JevAI choice in the start-of-game event"
                  if mode is None else "JevAI is off for this game (vanilla AI)"), flush=True)
            continue
        players = set(t for t in a.player.split(",") if t) or set(j.get("player_countries") or [])
        recs = month_records(j, st_center)
        names = {t: r["name"] for t, r in recs.items()}
        elig = [t for t, r in recs.items() if t not in players and r["n_states"] > 0 and r["mil"] + r["civ"] >= a.min_factories
                and (mode == "all" or t in majors)]
        t0 = time.time()
        chosen = {}
        for t in elig:
            u = model.scores(state_text(recs[t], names, recs), a.horizon)
            chosen[t] = int(np.argmax(u)) + 1
        write_atomic(os.path.join(out_dir, ORDERS + ".txt"), orders_file(chosen, j["date"], model.device))
        top = sorted(chosen.items(), key=lambda kv: -(recs[kv[0]]["mil"] + recs[kv[0]]["civ"]))[:6]
        print(f"[{time.strftime('%H:%M:%S')}] {j['date']} ({mode}): {len(chosen)} countries in {time.time() - t0:.0f}s on {model.device}; "
              + ", ".join(f"{t} {POSTURES[k - 1]}" for t, k in top), flush=True)


if __name__ == "__main__":
    main()
