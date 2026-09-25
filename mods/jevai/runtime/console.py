"""Reliable console input for a running game (the collector, jevd and the mod runtime share it).

console.ps1 focuses the game window and types "` <cmd> Enter ... `" with SendInput (the game ignores posted window
messages). If one toggle key is lost the console state inverts: the next send closes the console, the text lands on
the map as hotkeys (space = pause) and the final toggle leaves the console open. So every send ends with
`e <TAG> jev_ack_<n>`; if JEV|ACK|<n> does not show up in game.log the console is assumed open and the send is
retried without the opening toggle, which also closes it again. Retries add `gamespeed 5` to undo a stray pause.
"""
from __future__ import annotations

import os
import subprocess
import time

from .game import month_files, read_month

PS1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "console.ps1")
PREFERRED = ["USA", "SOV", "ENG", "GER", "JAP", "ITA", "FRA", "CHI", "BRA", "TUR", "SPR", "SWE"]


class Console:
    def __init__(self, userdir: str, pid_fn, ack_ids: range):
        """pid_fn() returns the current game pid (it changes when a slot relaunches). ack_ids: the jev_ack_<n>
        effects this process uses; processes typing into the same game need disjoint ranges."""
        self.userdir, self.pid_fn, self.ack_ids = userdir, pid_fn, list(ack_ids)
        self.i = 0
        self.failures = 0

    def _log(self) -> str:
        return os.path.join(self.userdir, "logs", "game.log")

    def _log_size(self) -> int:
        try:
            return os.path.getsize(self._log())
        except OSError:
            return 0

    def _seen(self, offset: int, needle: str) -> bool:
        try:
            with open(self._log(), "rb") as f:
                f.seek(offset)
                return needle.encode() in f.read()
        except OSError:
            return False

    def ack_tag(self) -> str:
        """A country that surely exists: `e TAG` must resolve or the ack never logs."""
        fs = month_files(os.path.join(self.userdir, "history_dump"))
        try:
            cs = (read_month(fs[-1], tries=1) or {}).get("countries", {}) if fs else {}
        except OSError:
            cs = {}
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
        """Run console lines exactly once, confirmed by the ack. False if no ack after `tries`."""
        n = self.ack_ids[self.i % len(self.ack_ids)]
        self.i += 1
        needle = f"JEV|ACK|{n}"
        tag = self.ack_tag()
        opened = False
        for k in range(tries):
            batch = list(cmds) + (["gamespeed 5"] if k > 0 else []) + [f"e {tag} jev_ack_{n}"]
            off = self._log_size()
            if self._type(";".join(batch), opened) == 2:  # could not focus the window: nothing was typed
                time.sleep(3)
                continue
            t0 = time.time()
            while time.time() - t0 < wait:
                time.sleep(0.5)
                if self._seen(off, needle):
                    return True
            opened = not opened  # no ack: the console is now open (inverted), so type without the opening toggle
        self.failures += 1
        return False

    def run_file(self, cmds: list[str], name: str = "jev_apply.txt") -> bool:
        """Write the commands to a file in the userdir and `run` it: one console line whatever their number."""
        with open(os.path.join(self.userdir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(cmds) + "\n")
        return self.send([f"run {name}"])
