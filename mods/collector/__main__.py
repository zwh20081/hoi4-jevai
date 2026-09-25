"""Parallel HOI4 data collector (VM only).

Keeps N games running at once (each needs ~5.6 GB), each in its own userdir, all on the OpenGL renderer (D3D11
crashed under VMware whenever a second instance loaded). Each slot runs a queue of games: prepare the userdir,
launch with -continuelastsave (continue_game.json points at the start save, so no menu clicks), then observe,
reseed, norender, nogui, speed 5; watch until the end year, a crash or a stall; archive history_dump and Jev's
actions to temp/games/<run>; next game.

Jev games register with jevd (temp/inst/<slot>.jev, carrying the game's policy), which controls a random half of
the eligible countries with eps-exploration; vanilla AI keeps the rest. Games alternate Jev/vanilla in pairs that
share a start save, so Jev vs vanilla is never confounded with the save.

    python -m mods.collector --slots 3 --games 9     (defaults: --jev alt --saves hist --every 6 --eps 0.5)
    python -m mods.collector --status
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time

from mods.jevai.runtime.console import Console
from mods.jevai.runtime.game import last_date
from mods.jevai.runtime.postures import VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEMP = os.path.join(ROOT, "temp")
GAMES, INST = os.path.join(TEMP, "games"), os.path.join(TEMP, "inst")
BASE_USER = os.path.join(TEMP, "hoi4user")  # template userdir: settings, DLC list, start saves; never run itself
MOD = os.path.join(ROOT, "mods", "jevai")
LOG = os.path.join(TEMP, "collect.log")
# live userdirs (logs, autosaves, dumps) stay on a local disk: the shared project folder is too slow for them
SCRATCH = os.environ.get("JEVAI_SCRATCH") or os.path.join(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(), "jevai", "inst")
SAVES = {"hist": "jev_1936_06.hoi4", "nonhist": "jev_1936_06_nonhist.hoi4"}


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ps(script: str, *args, timeout: int = 120):
    return subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, script), *map(str, args)],
                          capture_output=True, text=True, timeout=timeout)


def alive(pid: int) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def read_pid(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip() or 0)


def prepare_userdir(slot: int) -> str:
    """A working userdir for one slot under SCRATCH, filled from the template (settings, DLC list, start saves)."""
    u = os.path.join(SCRATCH, f"u{slot}")
    os.makedirs(os.path.join(u, "save games"), exist_ok=True)
    os.makedirs(os.path.join(u, "mod"), exist_ok=True)
    for f in ["settings.txt", "dlc_load.json", "gameplaysettings.txt", "naval_dist.cache", "naval_dist_checksum.cache",
              "dlc_signature", "game_data.json"]:
        if os.path.exists(os.path.join(BASE_USER, f)):
            shutil.copy2(os.path.join(BASE_USER, f), u)
    for s in SAVES.values():
        if not os.path.exists(os.path.join(u, "save games", s)):
            shutil.copy2(os.path.join(BASE_USER, "save games", s), os.path.join(u, "save games", s))
    return u


def install_mod(u: str):
    """Copy the mod's game files (descriptor.mod and common/, not the Python runtime) into the userdir and point a
    descriptor at them, the way the game installs mods. Done at every launch, so each game runs the mod as it was
    when the game started."""
    dst = os.path.join(u, "mod", "jevai")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(MOD, "common"), os.path.join(dst, "common"))
    shutil.copy2(os.path.join(MOD, "descriptor.mod"), dst)
    with open(os.path.join(MOD, "descriptor.mod"), encoding="utf-8-sig") as f:
        desc = f.read().rstrip()
    with open(os.path.join(u, "mod", "jevai.mod"), "w", encoding="utf-8") as f:
        f.write(desc + '\npath="mod/jevai"\n')


class Slot:
    def __init__(self, i: int):
        self.i, self.name = i, f"u{i}"
        self.userdir = prepare_userdir(i)
        self.pid, self.game = 0, None
        self.t_start, self.last, self.last_change, self.polled, self.kicked = 0.0, "", 0.0, 0.0, 0.0
        self.console = Console(self.userdir, lambda: self.pid, range(0, 5))  # jevd uses acks 5-9

    def start(self, game: dict):
        self.game = game
        u = self.userdir
        shutil.rmtree(os.path.join(u, "history_dump"), ignore_errors=True)
        if os.path.exists(os.path.join(u, "jev_apply.txt")):
            os.remove(os.path.join(u, "jev_apply.txt"))
        install_mod(u)
        with open(os.path.join(u, "continue_game.json"), "w") as f:
            json.dump({"title": "JevAI start", "desc": "", "date": "", "filename": SAVES[game["save"]], "is_remote": False}, f, indent=1)
        pidfile = os.path.join(INST, self.name + ".pid")
        ps("launch.ps1", "-gameArgs", "-ogl;-dump_history;-continuelastsave", "-userdir", u, "-pidfile", pidfile)
        self.pid = read_pid(pidfile)
        log(f"{self.name} launch {game['run']} save={game['save']} seed={game['seed']} jev={game['jev']} pid={self.pid}")
        for _ in range(100):  # the save is in once the log date leaves the 1936.01.01 front-end stamp
            time.sleep(3)
            if not alive(self.pid):
                raise RuntimeError("died while loading")
            d = last_date(u)
            if d and not d.startswith("1936.1.1"):
                break
        time.sleep(20)  # map and graphics finish loading after the date appears
        ps("movewin.ps1", "-procid", self.pid, "-x", 60 * (self.i - 1), "-y", 40 * (self.i - 1))
        # the setup commands are toggles (except gamespeed), so each must land exactly once: one verified send each
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
        if game["jev"]:  # register with jevd; the policy travels with the game
            with open(os.path.join(INST, self.name + ".jev"), "w") as f:
                json.dump({"inst": self.name, "userdir": u, "run": game["run"], **game.get("policy", {})}, f)

    def adopt(self, game: dict) -> bool:
        """Take over a game an earlier collector process started in this slot (pid file + live process)."""
        pidfile = os.path.join(INST, self.name + ".pid")
        pid = read_pid(pidfile) if os.path.exists(pidfile) else 0
        if not pid or not alive(pid):
            return False
        self.pid, self.game, self.t_start = pid, game, time.time()
        self.last, self.last_change = last_date(self.userdir), time.time()
        log(f"{self.name} adopted running {game['run']} pid={pid} at {self.last}")
        return True

    def poll(self) -> str | None:
        """A finish reason once the game is over, else None."""
        if not alive(self.pid):
            return "process gone"
        now = time.time()
        # a gap between polls far longer than the poll period means the whole VM was suspended; the game clock
        # stopped too, so that time is not a stall
        if self.polled and now - self.polled > 180:
            self.last_change = now
        self.polled = now
        d = last_date(self.userdir)
        if d != self.last:
            self.last, self.last_change = d, now
        elif now - self.last_change > 120 and now - self.kicked > 60:
            self.kicked = now
            self.console.send(["gamespeed 5"])
            if now - self.last_change > 420:
                return f"stalled at {d}"
        if d and int(d.split(".")[0]) >= self.game["end"]:
            return f"reached {d}"
        return None

    def finish(self, reason: str):
        g = self.game
        out = os.path.join(GAMES, g["run"])
        os.makedirs(out, exist_ok=True)
        if reason.startswith("stalled") and alive(self.pid):  # keep evidence: the next launch overwrites the logs
            ps("shot.ps1", "-procid", self.pid, "-out", os.path.join(out, "stall.png"))
            cpu = lambda: subprocess.run(["powershell", "-Command", f"(Get-Process -Id {self.pid}).TotalProcessorTime.TotalSeconds"],
                                         capture_output=True, text=True).stdout.strip()
            c0 = cpu()
            time.sleep(10)
            with open(os.path.join(out, "stall.txt"), "w") as f:
                f.write(f"cpu seconds {c0} -> {cpu()} over 10 s\n")
            for f in ["game.log", "error.log", "system.log"]:
                if os.path.exists(os.path.join(self.userdir, "logs", f)):
                    shutil.copy2(os.path.join(self.userdir, "logs", f), os.path.join(out, "stall_" + f))
        subprocess.run(["taskkill", "/PID", str(self.pid), "/F"], capture_output=True)
        time.sleep(4)
        if os.path.exists(os.path.join(INST, self.name + ".jev")):
            os.remove(os.path.join(INST, self.name + ".jev"))
        src = os.path.join(self.userdir, "history_dump")
        shutil.rmtree(os.path.join(out, "history_dump"), ignore_errors=True)
        n = 0
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, "history_dump"), ignore=shutil.ignore_patterns("*.png"))
            n = sum(1 for f in os.listdir(src) if f[:-4].isdigit())
        meta = dict(g, last_date=self.last, months=n, reason=reason, minutes=round((time.time() - self.t_start) / 60, 1), slot=self.name)
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        log(f"{self.name} done {g['run']}: {reason}, {n} months")
        self.game = None


def make_queue(n: int, start_idx: int, jev_mode: str, end: int, seed0: int, saves: tuple, policy: dict, pv: int,
               vanilla_first: bool = False) -> list[dict]:
    """Games alternate Jev/vanilla in pairs that share a save, so Jev vs vanilla is never confounded with the save."""
    q = []
    for k in range(n):
        i = start_idx + k
        jev = {"all": True, "none": False, "alt": (k % 2 == 0) != vanilla_first}[jev_mode]
        q.append({"run": f"g{i:02d}", "save": saves[(k // 2) % len(saves)], "seed": seed0 + 1009 * i, "jev": jev,
                  "end": end, "pv": pv, **({"policy": policy} if jev else {})})
    return q


def status():
    """Where each slot's game is, recent collector events, archived games, Jev activity, free RAM."""
    slots = sorted(glob.glob(os.path.join(SCRATCH, "u*")))
    print(" | ".join(f"{os.path.basename(u)} {last_date(u) or '-'}" for u in slots) or "no slots yet")
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip() for line in f if " done " in line or " launch " in line or "failed" in line]
        print("\n".join(lines[-4:]))
    for g in sorted(glob.glob(os.path.join(GAMES, "g*"))):
        n = len(glob.glob(os.path.join(g, "history_dump", "[0-9]*.txt")))
        a = os.path.join(g, "actions.jsonl")
        acts = sum(1 for _ in open(a, encoding="utf-8")) if os.path.exists(a) else 0
        m = json.load(open(os.path.join(g, "meta.json"))) if os.path.exists(os.path.join(g, "meta.json")) else None
        print(f"{os.path.basename(g)} months={n} actions={acts} pv={m.get('pv', 1) if m else '?'} {'done (' + m.get('reason', '') + ')' if m else 'running'}")
    jl = os.path.join(TEMP, "jevd.log")
    if os.path.exists(jl):
        with open(jl, encoding="utf-8", errors="replace") as f:
            last = [line.rstrip() for line in f if "countries" in line]
        if last:
            print(last[-1])
    try:
        print(subprocess.run(["powershell", "-Command", "$o=Get-CimInstance Win32_OperatingSystem; "
                              "'free RAM {0:N1} GB of {1:N1}' -f ($o.FreePhysicalMemory/1MB), ($o.TotalVisibleMemorySize/1MB)"],
                             capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:  # noqa: BLE001 - status is best effort
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Parallel HOI4 data collector (VM only)")
    ap.add_argument("--status", action="store_true", help="print the collection status and exit")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--start", type=int, default=0, help="first game number (default: after the existing runs)")
    ap.add_argument("--jev", choices=["all", "none", "alt"], default="alt")
    ap.add_argument("--end", type=int, default=1945, help="stop a game at this year")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--adopt", action="append", default=[], help="slot:run:save:seed:jev - keep a game that is already running")
    ap.add_argument("--saves", default="hist", help="comma list of start saves (hist, nonhist), cycled per Jev/vanilla pair")
    ap.add_argument("--pv", type=int, default=VERSION, help="posture version of the mod (runtime/postures.py VERSION)")
    ap.add_argument("--every", type=int, default=6, help="Jev decides every N months")
    ap.add_argument("--eps", type=float, default=0.5, help="exploration rate in Jev games")
    ap.add_argument("--horizon", type=int, default=6, help="months the posture questions look ahead")
    ap.add_argument("--vanilla-first", action="store_true", help="the alternation starts with a vanilla game")
    a = ap.parse_args(argv)
    if a.status:
        return status()
    os.makedirs(INST, exist_ok=True)
    existing = [int(d[1:]) for d in os.listdir(GAMES) if d[0] == "g" and d[1:].isdigit()] if os.path.isdir(GAMES) else []
    start = a.start or (max(existing, default=0) + 1)
    policy = {"every": a.every, "eps": a.eps, "pv": a.pv, "horizon": a.horizon}
    queue = make_queue(a.games, start, a.jev, a.end, a.seed, tuple(a.saves.split(",")), policy, a.pv, a.vanilla_first)
    log(f"collector: {a.slots} slots, {len(queue)} games {queue[0]['run']}..{queue[-1]['run']}, jev={a.jev}")
    slots = [Slot(i + 1) for i in range(a.slots)]
    for spec in a.adopt:
        si, run, save, seed, jev = spec.split(":")
        slots[int(si) - 1].adopt({"run": run, "save": save, "seed": int(seed), "jev": jev == "1", "end": a.end, "pv": a.pv})
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
                except Exception as e:  # noqa: BLE001 - a failed launch goes back on the queue once
                    log(f"{s.name} start failed for {g['run']}: {e}")
                    subprocess.run(["taskkill", "/PID", str(s.pid), "/F"], capture_output=True)
                    s.game = None
                    if not g.get("retried"):
                        queue.insert(0, dict(g, retried=True))
                time.sleep(30)  # stagger launches: two instances loading at once is the riskiest moment
        time.sleep(20)
    log("collector: queue empty")


if __name__ == "__main__":
    main()
