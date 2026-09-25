"""Build the HOI4 training set from all archived observer games.

    python -m jevai.build_dataset [--games runs/games] [--out runs/dataset] [--horizons 3,6,12]

For every runs/games/<run>/history_dump: parse -> records (runs/dataset/records/<run>.jsonl), then
examples for each horizon. Split is by GAME (whole runs held out) so no month of a test game is seen
in training. Writes train/val/test.jsonl + stats.json.
"""
from __future__ import annotations

import collections
import concurrent.futures
import glob
import json
import os
import random
import sys

from . import dump_parse, examples

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_run(d: str, rec_path: str, hz: tuple):
    """Parse one game's dumps into records (run in a worker process)."""
    run = os.path.basename(d)
    with open(rec_path, "w", encoding="utf-8") as f:
        for r in dump_parse.parse_dir(os.path.join(d, "history_dump"), tuple(int(h) for h in hz)):
            r["run"] = run
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return run


def main(argv):
    games = argv[argv.index("--games") + 1] if "--games" in argv else os.path.join(ROOT, "runs", "games")
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(ROOT, "runs", "dataset")
    hz = tuple(argv[argv.index("--horizons") + 1].split(",")) if "--horizons" in argv else ("3", "6", "12")
    os.makedirs(os.path.join(out, "records"), exist_ok=True)
    runs = sorted(d for d in glob.glob(os.path.join(games, "*")) if os.path.isdir(os.path.join(d, "history_dump")))
    per_run: dict[str, list] = {}
    stats = {"runs": {}, "horizons": list(hz)}
    todo = [d for d in runs if not os.path.exists(os.path.join(out, "records", os.path.basename(d) + ".jsonl"))]
    if todo:  # parsing is the slow part: one worker per game
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(todo), os.cpu_count() or 4)) as ex:
            futs = [ex.submit(parse_run, d, os.path.join(out, "records", os.path.basename(d) + ".jsonl"), hz) for d in todo]
            for f in concurrent.futures.as_completed(futs):
                print("parsed", f.result(), flush=True)
    act_count = collections.Counter()
    for d in runs:
        run = os.path.basename(d)
        rec_path = os.path.join(out, "records", run + ".jsonl")
        recs = [json.loads(l) for l in open(rec_path, encoding="utf-8")]
        act_path = os.path.join(d, "actions.jsonl")
        acts = [json.loads(l) for l in open(act_path, encoding="utf-8") if l.strip()] if os.path.exists(act_path) else []
        examples.attach_actions(recs, acts)
        for r in recs:
            m = r.get("meta") or {}
            if m.get("action_fresh"):
                a = m["action"]
                act_count[(f"v{a.get('pv', 1)}", a["posture"], "applied" if m.get("action_applied", True) else "not applied")] += 1
        exs = []
        for h in hz:
            exs += examples.build(recs, h, seed=hash(run) & 0xFFFF)
        per_run[run] = exs
        dates = sorted({r["date"] for r in recs}, key=lambda s: [int(x) for x in s.split(".")])
        stats["runs"][run] = {"months": len(dates), "first": dates[0] if dates else None, "last": dates[-1] if dates else None,
                              "records": len(recs), "examples": len(exs), "questions": sum(len(e.questions) for e in exs),
                              "jev_actions": len(acts)}
        print(run, stats["runs"][run])
    names = sorted(per_run)
    # hold out whole games (no month of a held-out game is seen in training). Stable: a game's split
    # depends only on its name, so adding games never moves an old one between splits.
    def part_of(run):
        k = int(run[1:]) if run[1:].isdigit() else sum(map(ord, run))
        return "test" if k % 5 == 4 else "val" if k % 5 == 3 else "train"
    split = {"train": [], "val": [], "test": []}
    for r in names:
        split[part_of(r) if len(names) >= 3 else "train"].append(r)
    stats["split"] = split
    for part, rs in split.items():
        exs = [e for r in rs for e in per_run[r]]
        random.Random(0).shuffle(exs)
        with open(os.path.join(out, part + ".jsonl"), "w", encoding="utf-8") as f:
            for e in exs:
                f.write(e.to_json() + "\n")
        lab = collections.Counter()
        for e in exs:
            for q in e.questions:
                lab[f"{q.qid}={q.options[q.gold] if q.kind != 'choice' or q.qid == 'grow' else 'named'}"] += 1
        stats[part] = {"states": len(exs), "questions": sum(len(e.questions) for e in exs), "labels": dict(sorted(lab.items()))}
        print(part, stats[part]["states"], "states", stats[part]["questions"], "questions")
    stats["decisions"] = {" / ".join(k): v for k, v in sorted(act_count.items())}
    print("decisions:", stats["decisions"])
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
