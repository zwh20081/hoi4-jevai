#!/usr/bin/env python
"""Parallel HOI4 data collector.

Keeps N game instances running at once (default 2 — each needs ~5.6 GB), each in its own userdir,
all on the OpenGL renderer (the D3D11 path crashes under VMware when a second instance loads).
Each slot runs a queue of games: prepare userdir -> launch with -continuelastsave (continue_game.json
points at the start save) -> observe / reseed / norender / nogui / speed 5 -> watch until the end
year, a crash or a stall -> archive history_dump (+ Jev actions) to runs/games/<run> -> next game.

Jev participation: when a game's `jev` flag is set, the daemon (jevai.jevd) is told about the
instance and controls a random half of the eligible countries with eps-exploration; vanilla AI
keeps the rest. Games without the flag are the pure-vanilla control.

    python -m jevai.collector --slots 2 --games 8 --jev-every-other
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from .console import Console

# Every path is derived from where this file lives, so the project runs from any folder
# (VM local disk, the Z: shared folder, or the host).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P0 = os.path.join(ROOT, "jev_hoi4", "p0")
BASE_USER = os.path.join(ROOT, "hoi4user")  # template userdir: settings, dlc list, start saves
MOD = os.path.join(ROOT, "mod", "jevai_probe")
# The games' working userdirs (logs, autosaves, live dumps) live on a local disk, not in the project:
# a network/shared project folder is too slow for them. Finished games are archived into runs/games.
SCRATCH = os.environ.get("JEVAI_SCRATCH") or os.path.join(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(), "jevai", "inst")
SAVES = {"hist": "jev_1936_06.hoi4", "nonhist": "jev_1936_06_nonhist.hoi4"}
LOG = os.path.join(ROOT, "runs", "collect.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def ps(script, *args, timeout=120):
    return subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", os.path.join(P0, script), *map(str, args)],
                          capture_output=True, text=True, timeout=timeout)


def alive(pid):
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def last_date(userdir):
    p = os.path.join(userdir, "logs", "game.log")
    try:
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 20000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    import re
    ds = re.findall(r"\]\[(1[89]\d\d\.\d+\.\d+)\.\d+\]", tail)
    return ds[-1] if ds else ""


def prepare_userdir(slot: int) -> str:
    """A working userdir for one slot under SCRATCH, filled from the template userdir hoi4user
    (settings, dlc list, start saves). The template itself is never run."""
    u = os.path.join(SCRATCH, f"u{slot}")
    os.makedirs(os.path.join(u, "save games"), exist_ok=True)
    os.makedirs(os.path.join(u, "mod"), exist_ok=True)
    for f in ["settings.txt", "dlc_load.json", "gameplaysettings.txt", "naval_dist.cache", "naval_dist_checksum.cache",
              "dlc_signature", "game_data.json"]:
        if os.path.exists(os.path.join(BASE_USER, f)):
            shutil.copy2(os.path.join(BASE_USER, f), u)
    for s in SAVES.values():
        dst = os.path.join(u, "save games", s)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(BASE_USER, "save games", s), dst)
    return u


def install_mod(u: str):
    """Copy the current mod into the userdir and point the descriptor at it with a path relative to the
    userdir (the way the game itself installs mods). Done at every launch, so each game runs the mod
    as it was when the game started."""
    dst = os.path.join(u, "mod", "jevai_probe")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(MOD, dst)
    desc = open(os.path.join(MOD, "descriptor.mod"), encoding="utf-8-sig").read().rstrip()
    open(os.path.join(u, "mod", "jevai_probe.mod"), "w", encoding="utf-8").write(desc + '\npath="mod/jevai_probe"\n')


class Slot:
    def __init__(self, i: int):
        self.i, self.name = i, f"u{i}"
        self.userdir = prepare_userdir(i)
        self.pid = 0
        self.game = None
        self.t_start = 0.0
        self.last, self.last_change = "", 0.0
        self.console = Console(self.userdir, lambda: self.pid, range(0, 5))  # jevd uses acks 5-9

    def start(self, game: dict):
        self.game = game
        u = self.userdir
        shutil.rmtree(os.path.join(u, "history_dump"), ignore_errors=True)
        for f in ["jev_apply.txt"]:
            if os.path.exists(os.path.join(u, f)):
                os.remove(os.path.join(u, f))
        install_mod(u)
        json.dump({"title": "JevAI start", "desc": "", "date": "", "filename": SAVES[game["save"]], "is_remote": False},
                  open(os.path.join(u, "continue_game.json"), "w"), indent=1)
        pidfile = os.path.join(ROOT, "runs", "inst", self.name + ".pid")
        r = ps("launch.ps1", "-gameArgs", "-ogl;-dump_history;-continuelastsave", "-userdir", u, "-pidfile", pidfile)
        self.pid = int(open(pidfile).read().strip())
        log(f"{self.name} launch {game['run']} save={game['save']} seed={game['seed']} jev={game['jev']} pid={self.pid}")
        # wait for the save to be in (log date leaves the 1936.01.01 front-end stamp)
        for _ in range(100):
            time.sleep(3)
            if not alive(self.pid):
                raise RuntimeError("died while loading")
            d = last_date(u)
            if d and not d.startswith("1936.1.1"):
                break
        time.sleep(20)  # map/graphics finish loading after the date appears
        ps("movewin.ps1", "-procid", self.pid, "-x", 60 * (self.i - 1), "-y", 40 * (self.i - 1))
        # every setup command is a toggle, so each must land exactly once: send them one per verified batch
        for cmd in ["observe", f"random_seed {game['seed']}", "debug_norender", "debug_nogui", "gamespeed 5"]:
            if not self.console.send([cmd]):
                raise RuntimeError(f"console did not ack '{cmd}'")
        d0 = last_date(u)
        for _ in range(6):
            time.sleep(10)
            if last_date(u) != d0:
                break
            self.console.send(["gamespeed 5"])
        else:
            raise RuntimeError("clock does not move after gamespeed 5")
        self.t_start = time.time()
        self.last, self.last_change = last_date(u), time.time()
        # register with the daemon (policy settings travel with the game, see jevd.Instance)
        if game["jev"]:
            reg = os.path.join(ROOT, "runs", "inst", self.name + ".jev")
            json.dump({"inst": self.name, "userdir": u, "run": game["run"], **game.get("policy", {})}, open(reg, "w"))

    def adopt(self, game: dict) -> bool:
        """Take over a game a previous collector process started in this slot (pid file + live process)."""
        pidfile = os.path.join(ROOT, "runs", "inst", self.name + ".pid")
        if not os.path.exists(pidfile):
            return False
        pid = int(open(pidfile).read().strip() or 0)
        if not pid or not alive(pid):
            return False
        self.pid, self.game = pid, game
        self.t_start = time.time()
        self.last, self.last_change = last_date(self.userdir), time.time()
        log(f"{self.name} adopted running {game['run']} pid={pid} at {self.last}")
        return True

    def poll(self) -> str | None:
        """Return a finish reason when the game is over, else None."""
        if not alive(self.pid):
            return "process gone"
        now = time.time()
        # a gap between polls far longer than the poll period means the whole VM was suspended; the
        # game clock stopped too, so don't count that time as a stall (seen twice: 17 and 29 min)
        if now - getattr(self, "polled", now) > 180:
            self.last_change = now
        self.polled = now
        d = last_date(self.userdir)
        if d != self.last:
            self.last, self.last_change = d, now
        elif now - self.last_change > 120 and now - getattr(self, "kicked", 0) > 60:
            self.kicked = now
            self.console.send(["gamespeed 5"])
            if now - self.last_change > 420:
                return f"stalled at {d}"
        if d and int(d.split(".")[0]) >= self.game["end"]:
            return f"reached {d}"
        return None

    def finish(self, reason: str):
        g = self.game
        out = os.path.join(ROOT, "runs", "games", g["run"])
        os.makedirs(out, exist_ok=True)
        if reason.startswith("stalled") and alive(self.pid):
            # keep evidence: the next launch overwrites the userdir logs
            ps("shot.ps1", "-procid", self.pid, "-out", os.path.join(out, "stall.png"))
            cpu = subprocess.run(["powershell", "-Command", f"(Get-Process -Id {self.pid}).TotalProcessorTime.TotalSeconds"],
                                 capture_output=True, text=True).stdout.strip()
            time.sleep(10)
            cpu2 = subprocess.run(["powershell", "-Command", f"(Get-Process -Id {self.pid}).TotalProcessorTime.TotalSeconds"],
                                  capture_output=True, text=True).stdout.strip()
            open(os.path.join(out, "stall.txt"), "w").write(f"cpu seconds {cpu} -> {cpu2} over 10 s\n")
            for f in ["game.log", "error.log", "system.log"]:
                src = os.path.join(self.userdir, "logs", f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(out, "stall_" + f))
        subprocess.run(["taskkill", "/PID", str(self.pid), "/F"], capture_output=True)
        time.sleep(4)
        reg = os.path.join(ROOT, "runs", "inst", self.name + ".jev")
        if os.path.exists(reg):
            os.remove(reg)
        out = os.path.join(ROOT, "runs", "games", g["run"])
        os.makedirs(out, exist_ok=True)
        src = os.path.join(self.userdir, "history_dump")
        shutil.rmtree(os.path.join(out, "history_dump"), ignore_errors=True)
        n = 0
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, "history_dump"), ignore=shutil.ignore_patterns("*.png"))
            n = sum(1 for f in os.listdir(src) if f[:-4].isdigit())
        meta = dict(g, last_date=self.last, months=n, reason=reason, minutes=round((time.time() - self.t_start) / 60, 1), slot=self.name)
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        log(f"{self.name} done {g['run']}: {reason}, {n} months")
        self.game = None


def make_queue(n: int, start_idx: int, jev_mode: str, end: int, seed0: int, saves=("hist",), policy=None, pv=1,
               vanilla_first: bool = False):
    """Games alternate Jev/vanilla in pairs that share a save, so Jev vs vanilla is never confounded with
    the save (until 2026-09-25 Jev games all used 'hist' and vanilla games 'nonhist')."""
    q = []
    for k in range(n):
        i = start_idx + k
        jev = {"all": True, "none": False, "alt": (k % 2 == 0) != vanilla_first}[jev_mode]
        q.append({"run": f"g{i:02d}", "save": saves[(k // 2) % len(saves)], "seed": seed0 + 1009 * i, "jev": jev,
                  "end": end, "pv": pv, **({"policy": policy} if jev and policy else {})})
    return q


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--start", type=int, default=0, help="first game index (default: after existing runs)")
    ap.add_argument("--jev", choices=["all", "none", "alt"], default="alt")
    ap.add_argument("--end", type=int, default=1945)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--adopt", action="append", default=[], help="slot:run:save:seed:jev - keep a game already running")
    ap.add_argument("--saves", default="hist", help="comma list of start saves, cycled per Jev/vanilla pair")
    ap.add_argument("--pv", type=int, default=2, help="posture version of the mod (jevai/gen_mod.py VERSION)")
    ap.add_argument("--every", type=int, default=6, help="Jev decides every N months")
    ap.add_argument("--sync", action="store_true", default=True, help="all countries decide on the same months")
    ap.add_argument("--eps", type=float, default=0.5, help="exploration rate for Jev games")
    ap.add_argument("--vanilla-first", action="store_true", help="alternation starts with a vanilla game")
    ap.add_argument("--rule", choices=["value", "choice"], default="value", help="Jev decision rule (jevd)")
    ap.add_argument("--horizon", type=int, default=6, help="look-ahead of the posture-outcome questions (rule value)")
    a = ap.parse_args(argv)
    games_dir = os.path.join(ROOT, "runs", "games")
    existing = [int(d[1:]) for d in os.listdir(games_dir) if d[0] == "g" and d[1:].isdigit()] if os.path.isdir(games_dir) else []
    start = a.start or (max(existing, default=0) + 1)
    policy = {"every": a.every, "sync": a.sync, "eps": a.eps, "pv": a.pv, "rule": a.rule, "horizon": a.horizon}
    queue = make_queue(a.games, start, a.jev, a.end, a.seed, tuple(a.saves.split(",")), policy, a.pv, a.vanilla_first)
    log(f"collector: {a.slots} slots, {len(queue)} games {queue[0]['run']}..{queue[-1]['run']}, jev={a.jev}")
    slots = [Slot(i + 1) for i in range(a.slots)]
    for spec in a.adopt:
        si, run, save, seed, jev = spec.split(":")
        g = {"run": run, "save": save, "seed": int(seed), "jev": jev == "1", "end": a.end, "pv": a.pv}
        slots[int(si) - 1].adopt(g)
        queue = [q for q in queue if q["run"] != run]
    while queue or any(s.game for s in slots):
        for s in slots:
            if s.game:
                r = s.poll()
                if r:
                    s.finish(r)
            if not s.game and queue:
                g = queue.pop(0)
                try:
                    s.start(g)
                except Exception as e:  # noqa: BLE001 - a failed launch just goes back on the queue once
                    log(f"{s.name} start failed for {g['run']}: {e}")
                    subprocess.run(["taskkill", "/PID", str(s.pid), "/F"], capture_output=True)
                    s.game = None
                    if not g.get("retried"):
                        queue.insert(0, dict(g, retried=True))
                time.sleep(30)  # stagger launches: two instances loading at once is the riskiest moment
        time.sleep(20)
    log("collector: queue empty")


if __name__ == "__main__":
    main(sys.argv[1:])
