"""Reliable console input for a running HOI4 instance (shared by the collector and jevd).

console.ps1 types "` <cmd> Enter ... `" into the window. If one toggle key is lost the console state
inverts: the next send closes the console, the text lands on the map as hotkeys (space = pause) and
the final toggle leaves the console open. Every send therefore ends with `e <TAG> jev_ack_<n>`; if the
ack does not appear in game.log the console is assumed open and the send is retried without the
opening toggle, which also re-closes it. Retries add `gamespeed 5` to undo a stray pause.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PS1 = os.path.join(ROOT, "jev_hoi4", "p0", "console.ps1")
PREFERRED = ["USA", "SOV", "ENG", "GER", "JAP", "ITA", "FRA", "CHI", "BRA", "TUR", "SPR", "SWE"]


class Console:
    def __init__(self, userdir: str, pid_fn, ack_ids: range):
        """pid_fn() returns the current game pid (it changes when a slot relaunches).
        ack_ids: which jev_ack_<n> effects this process uses (collector and jevd use disjoint ranges)."""
        self.userdir, self.pid_fn, self.ack_ids = userdir, pid_fn, list(ack_ids)
        self.i = 0
        self.failures = 0

    def _log_size(self) -> int:
        try:
            return os.path.getsize(os.path.join(self.userdir, "logs", "game.log"))
        except OSError:
            return 0

    def _seen(self, offset: int, needle: str) -> bool:
        try:
            with open(os.path.join(self.userdir, "logs", "game.log"), "rb") as f:
                f.seek(offset)
                return needle.encode() in f.read()
        except OSError:
            return False

    def ack_tag(self) -> str:
        """A country that surely exists: the `e TAG` must resolve or the ack never logs."""
        fs = sorted(glob.glob(os.path.join(self.userdir, "history_dump", "[0-9]*.txt")), key=lambda p: int(os.path.basename(p)[:-4]))
        if not fs:
            return "GER"
        try:
            cs = json.load(open(fs[-1], encoding="utf-8", errors="replace"))["countries"]
        except (OSError, ValueError, KeyError):
            return "GER"
        for t in PREFERRED:
            c = cs.get(t)
            if c and c.get("exists", True) is not False and c.get("fully_controlled_states"):
                return t
        return "GER"

    def _type(self, line: str, opened: bool) -> int:
        args = ["powershell", "-ExecutionPolicy", "Bypass", "-File", PS1, "-procid", str(self.pid_fn()), "-cmds", line]
        if opened:
            args.append("-open")
        try:
            return subprocess.run(args, capture_output=True, timeout=120).returncode
        except subprocess.TimeoutExpired:
            return 1

    def send(self, cmds: list[str], wait: float = 10.0, tries: int = 4) -> bool:
        """Run `cmds` (console lines) exactly once, verified. Returns False if no ack after `tries`."""
        n = self.ack_ids[self.i % len(self.ack_ids)]
        self.i += 1
        needle = f"JEV|ACK|{n}"
        tag = self.ack_tag()
        opened = False
        for k in range(tries):
            batch = list(cmds) + (["gamespeed 5"] if k > 0 else []) + [f"e {tag} jev_ack_{n}"]
            off = self._log_size()
            rc = self._type(";".join(batch), opened)
            if rc == 2:  # could not focus the window: nothing was typed, state unchanged
                time.sleep(3)
                continue
            t0 = time.time()
            while time.time() - t0 < wait:
                time.sleep(0.5)
                if self._seen(off, needle):
                    return True
            # no ack: assume the console is now open (inverted state) and type without the opening toggle
            opened = not opened
        self.failures += 1
        return False
