"""Collection status: where each slot's game is, recent collector events, archived games, Jev activity.

    python -m jevai.status        (or: bash jev_hoi4/status.sh)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess

from .collector import LOG, ROOT, SCRATCH, last_date


def main():
    slots = sorted(glob.glob(os.path.join(SCRATCH, "u*")))
    print(" | ".join(f"{os.path.basename(u)} {last_date(u) or '-'}" for u in slots) or "no slots yet")
    if os.path.exists(LOG):
        lines = [l.rstrip() for l in open(LOG, encoding="utf-8", errors="replace") if " done " in l or " launch " in l or "failed" in l]
        print("\n".join(lines[-4:]))
    games = sorted(glob.glob(os.path.join(ROOT, "runs", "games", "g*")))
    for g in games:
        n = len(glob.glob(os.path.join(g, "history_dump", "[0-9]*.txt")))
        a = os.path.join(g, "actions.jsonl")
        acts = sum(1 for _ in open(a, encoding="utf-8")) if os.path.exists(a) else 0
        m = json.load(open(os.path.join(g, "meta.json"))) if os.path.exists(os.path.join(g, "meta.json")) else None
        state = f"done ({m.get('reason', '')})" if m else "running"
        print(f"{os.path.basename(g)} months={n} actions={acts} pv={m.get('pv', 1) if m else '?'} {state}")
    jl = os.path.join(ROOT, "runs", "jevd.log")
    if os.path.exists(jl):
        last = [l.rstrip() for l in open(jl, encoding="utf-8", errors="replace") if "countries" in l]
        if last:
            print(last[-1])
    try:
        out = subprocess.run(["powershell", "-Command", "$o=Get-CimInstance Win32_OperatingSystem; "
                              "'free RAM {0:N1} GB of {1:N1}' -f ($o.FreePhysicalMemory/1MB), ($o.TotalVisibleMemorySize/1MB)"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        print(out)
    except Exception:  # noqa: BLE001 - status is best effort
        pass


if __name__ == "__main__":
    main()
