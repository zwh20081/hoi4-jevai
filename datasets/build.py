"""Build the training set from the archived games (temp/games/<run>/history_dump and actions.jsonl).

    python -m datasets.build [--games temp/games] [--out temp/dataset] [--horizons 3,6,12]

Each game month becomes one record per country (runtime.game) with what happened 3/6/12 months later attached;
records are cached in <out>/records/<run>.jsonl (delete a file to re-parse that game). Each record becomes one
example per horizon: the state text plus labelled questions.
- Forecast questions, every country-month: power trend, capitulation, territory gained or lost, new war, the next
  enemy, territory taken from a named country.
- Posture questions (runtime.text.posture_questions), for months where Jev applied a posture and the game confirmed
  it. The posture was drawn with known odds, so the outcome is a causal label for it.
Games are split whole by run number (% 5: 4 test, 3 validation, else train), so no month of a held-out game is
trained on. Writes train/val/test.jsonl and stats.json.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import random
import zlib

from mods.jevai.runtime.game import load_invariants, month_files, month_records
from mods.jevai.runtime.text import GROWTH_LEVELS, Example, Question, noul, posture_questions, state_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def power(r: dict) -> float:
    return r["mil"] * 1.0 + r["civ"] * 0.5 + r["divisions"] * 0.3


def growth_level(g: float) -> int:
    """Index into GROWTH_LEVELS for a relative power change."""
    return 0 if g < -0.02 else 1 if g < 0.03 else 2 if g < 0.12 else 3


def outcome(now: dict, later: dict | None) -> dict:
    """What happened between two observations of the same country. `later` None = the country is gone."""
    if later is None or (later["n_states"] == 0 and now["n_states"] > 0):
        return {"exists": 0}
    p0, p1 = power(now), power(later)
    return {"exists": 1, "d_states": later["n_states"] - now["n_states"], "d_mil": later["mil"] - now["mil"],
            "d_civ": later["civ"] - now["civ"], "d_div": later["divisions"] - now["divisions"],
            "power_growth": round((p1 - p0) / max(p0, 1.0), 4),
            "new_enemies": sorted(set(later["enemies"]) - set(now["enemies"])), "capitulated": 0}


def parse_dir(dump_dir: str, horizons=(3, 6, 12)):
    """Records of one game, each with outcome[str(h)] for every horizon that still falls inside the game."""
    st_center = load_invariants(dump_dir)
    months = []
    for p in month_files(dump_dir):
        with open(p, encoding="utf-8", errors="replace") as f:
            j = json.load(f)
        months.append((j["date"], month_records(j, st_center)))
    owner = [{s: t for t, r in recs.items() for s in r["states"]} for _, recs in months]  # who holds each state
    for i, (_, recs) in enumerate(months):
        for t, r in recs.items():
            r["outcome"] = {}
            for h in horizons:
                if i + h >= len(months):
                    continue
                o = outcome(r, months[i + h][1].get(t))
                took, lost = collections.defaultdict(int), collections.defaultdict(int)
                for s, t0 in owner[i].items():
                    t1 = owner[i + h].get(s)
                    if t1 and t1 != t0:
                        if t1 == t:
                            took[t0] += 1
                        if t0 == t:
                            lost[t1] += 1
                o["took_from"], o["lost_to"] = dict(took), dict(lost)
                o["capitulated"] = int(o["exists"] == 0)
                r["outcome"][str(h)] = o
            yield r


def forecast_questions(r: dict, h: str, names: dict[str, str], rng: random.Random) -> list[Question]:
    o = r["outcome"].get(h)
    if not o:
        return []
    months = f"{h} months"
    qs = [Question("grow", "choice", f"How will this country's military-industrial power change over the next {months}?",
                   list(GROWTH_LEVELS), growth_level(o.get("power_growth", 0.0)) if o["exists"] else 0),
          noul("capit", f"This country will capitulate or be annexed within {months}.", o["capitulated"])]
    if not o["exists"]:
        return qs
    qs.append(noul("gain", f"This country will gain territory within {months}.", o["d_states"] > 0))
    qs.append(noul("lose", f"This country will lose territory within {months}.", o["d_states"] < 0))
    qs.append(noul("newwar", f"This country will enter a new war within {months}.", bool(o["new_enemies"])))
    if o["new_enemies"]:  # who it fights next: a choice among neighbours, only when there is a new enemy to name
        cands = [t for t in r.get("neighbors", []) if t not in r["enemies"]]
        gold = [t for t in o["new_enemies"] if t in cands]
        if gold and len(cands) >= 2:
            cands = cands[:9] if gold[0] in cands[:9] else cands[:8] + [gold[0]]
            rng.shuffle(cands)
            qs.append(Question("whowar", "choice", "Which country will this country be at war with next?",
                               [names.get(t, t) for t in cands], cands.index(gold[0])))
    for t in list(o.get("took_from", {}))[:1]:  # territory flow with a named country
        qs.append(noul("took", f"This country will take territory from {names.get(t, t)} within {months}.", True))
    if not o.get("took_from") and r["enemies"]:
        t = rng.choice(r["enemies"])
        qs.append(noul("took", f"This country will take territory from {names.get(t, t)} within {months}.", False))
    return qs


def action_questions(r: dict, h: str) -> list[Question]:
    """Labelled posture questions for a fresh Jev decision the game confirmed (see attach_actions)."""
    m = r.get("meta") or {}
    a, o = m.get("action"), r["outcome"].get(h)
    if not a or not o or not m.get("action_fresh") or not m.get("action_applied", True):
        return []
    grow = growth_level(o.get("power_growth", 0.0)) if o["exists"] else 0
    ok, gain = bool(o["exists"] and o["d_states"] >= 0), bool(o["exists"] and o["d_states"] > 0)
    return posture_questions(a["posture"], a.get("pv", 1), h, (grow, int(ok), int(gain)))


def attach_actions(records: list[dict], actions: list[dict]):
    """Put each Jev decision on the record of its (tag, date). A posture stays in force until the next decision, so
    the records in between inherit it (action_fresh False). action_applied: the mod's telemetry (`pos`, posture in
    force) shows the posture within the next two months; games without that telemetry count every decision."""
    key = lambda d: [int(x) for x in d.split(".")]
    dates = sorted({r["date"] for r in records}, key=key)
    nxt = {d: dates[i + 1:i + 3] for i, d in enumerate(dates)}
    pos = {(r["tag"], r["date"]): (r.get("jev") or {}).get("pos") for r in records}
    by = {(a["tag"], a["date"]): a for a in actions}
    cur: dict[str, dict] = {}
    for r in sorted(records, key=lambda r: key(r["date"])):
        a = by.get((r["tag"], r["date"]))
        if a:
            cur[r["tag"]] = a
        if r["tag"] in cur:
            m = r.setdefault("meta", {})
            m["action"], m["action_fresh"] = cur[r["tag"]], bool(a)
            if a:
                known = [p for p in (pos.get((r["tag"], d)) for d in nxt[r["date"]]) if p is not None]
                m["action_applied"] = (not known) or any(int(p) == a["k"] for p in known)


def build(records: list[dict], h: str = "3", seed: int = 0, min_states: int = 1) -> list[Example]:
    rng = random.Random(seed)
    by_date: dict[str, dict[str, dict]] = {}
    for r in records:
        by_date.setdefault(r["date"], {})[r["tag"]] = r
    names = {r["tag"]: r["name"] for r in records}
    out = []
    for r in records:
        if r["n_states"] < min_states:
            continue
        qs = forecast_questions(r, h, names, rng) + action_questions(r, h)
        if not qs:
            continue
        meta = {"tag": r["tag"], "date": r["date"], "run": r.get("run", "")}
        a = (r.get("meta") or {}).get("action")
        if a:
            meta.update(posture=a["posture"], propensity=a["propensity"], explore=a["explore"], fresh=r["meta"].get("action_fresh"))
        out.append(Example(state_text(r, names, by_date[r["date"]]), qs, f"hoi4:{h}m", meta))
    return out


def parse_run(game_dir: str, rec_path: str, horizons: tuple[int, ...]) -> str:
    """Parse one game into its records file (runs in a worker process)."""
    run = os.path.basename(game_dir)
    with open(rec_path, "w", encoding="utf-8") as f:
        for r in parse_dir(os.path.join(game_dir, "history_dump"), horizons):
            r["run"] = run
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return run


def split_of(run: str) -> str:
    """Stable per game (depends only on its name), so adding games never moves an old one between splits."""
    k = int(run[1:]) if run[1:].isdigit() else sum(map(ord, run))
    return "test" if k % 5 == 4 else "val" if k % 5 == 3 else "train"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the HOI4 training set from archived games")
    ap.add_argument("--games", default=os.path.join(ROOT, "temp", "games"))
    ap.add_argument("--out", default=os.path.join(ROOT, "temp", "dataset"))
    ap.add_argument("--horizons", default="3,6,12")
    a = ap.parse_args(argv)
    hz = tuple(a.horizons.split(","))
    rec_dir = os.path.join(a.out, "records")
    os.makedirs(rec_dir, exist_ok=True)
    games = sorted(d for d in (os.path.join(a.games, n) for n in os.listdir(a.games)) if os.path.isdir(os.path.join(d, "history_dump")))
    todo = [d for d in games if not os.path.exists(os.path.join(rec_dir, os.path.basename(d) + ".jsonl"))]
    if todo:  # parsing is the slow part: one worker per game
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(todo), os.cpu_count() or 4)) as ex:
            futs = [ex.submit(parse_run, d, os.path.join(rec_dir, os.path.basename(d) + ".jsonl"), tuple(map(int, hz))) for d in todo]
            for f in concurrent.futures.as_completed(futs):
                print("parsed", f.result(), flush=True)
    per_run: dict[str, list[Example]] = {}
    stats: dict = {"runs": {}, "horizons": list(hz)}
    decisions = collections.Counter()
    for d in games:
        run = os.path.basename(d)
        with open(os.path.join(rec_dir, run + ".jsonl"), encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        act_path = os.path.join(d, "actions.jsonl")
        acts = []
        if os.path.exists(act_path):
            with open(act_path, encoding="utf-8") as f:
                acts = [json.loads(line) for line in f if line.strip()]
        attach_actions(recs, acts)
        for r in recs:
            m = r.get("meta") or {}
            if m.get("action_fresh"):
                a_ = m["action"]
                decisions[(f"v{a_.get('pv', 1)}", a_["posture"], "applied" if m.get("action_applied", True) else "not applied")] += 1
        per_run[run] = [e for h in hz for e in build(recs, h, seed=zlib.crc32(run.encode()) & 0xFFFF)]
        dates = sorted({r["date"] for r in recs}, key=lambda s: [int(x) for x in s.split(".")])
        stats["runs"][run] = {"months": len(dates), "first": dates[0] if dates else None, "last": dates[-1] if dates else None,
                              "records": len(recs), "examples": len(per_run[run]),
                              "questions": sum(len(e.questions) for e in per_run[run]), "jev_actions": len(acts)}
        print(run, stats["runs"][run], flush=True)
    split = {"train": [], "val": [], "test": []}
    for run in sorted(per_run):
        split[split_of(run) if len(per_run) >= 3 else "train"].append(run)
    stats["split"] = split
    for part, runs in split.items():
        exs = [e for r in runs for e in per_run[r]]
        random.Random(0).shuffle(exs)
        with open(os.path.join(a.out, part + ".jsonl"), "w", encoding="utf-8") as f:
            for e in exs:
                f.write(e.to_json() + "\n")
        labels = collections.Counter(f"{q.qid}={q.options[q.gold] if q.kind != 'choice' or q.qid == 'grow' else 'named'}"
                                     for e in exs for q in e.questions)
        stats[part] = {"states": len(exs), "questions": sum(len(e.questions) for e in exs), "labels": dict(sorted(labels.items()))}
        print(part, stats[part]["states"], "states", stats[part]["questions"], "questions", flush=True)
    stats["decisions"] = {" / ".join(k): v for k, v in sorted(decisions.items())}
    print("decisions:", stats["decisions"])
    with open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
