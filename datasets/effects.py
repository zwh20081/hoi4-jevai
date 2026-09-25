"""Which Jev postures are good or bad? Measured from the collected games, three ways.

    python -m datasets.effects [--h 6] [--pv 2]

1. Manipulation check: does a posture change what the country builds? For each Jev-controlled
   country in a game, monthly production per equipment group is compared with that country's own
   average in the same game (ratio > 1 = the posture raised it). A posture that does not move
   production cannot have a real effect, whatever the outcome numbers say.
2. Posture effects (randomised): when Jev's top choice was X, the applied posture was a lottery with
   fixed odds (0.75 X, 0.05 each other) that does not depend on the country's situation, so comparing
   mean outcomes between postures inside that group is a fair causal comparison. Outcomes are
   residuals: the change over the next h months minus the same country's average change over the
   same months in the other games (all games start from the same save), which removes most of the
   noise that comes from who the country is and what year it is.
3. Jev vs vanilla: in Jev games a coin flip decided which countries Jev controls; compare controlled
   and uncontrolled countries' residual outcomes.

Confidence intervals come from resampling whole games (outcomes inside one game are correlated).
Only games on the historical-focus save are used (the non-historical save is a different world).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

from mods.jevai.runtime.game import country_record, month_files, parse_logs
from mods.jevai.runtime.postures import NAMES as POSTURES, VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = os.path.join(ROOT, "temp", "games")
CACHE = os.path.join(ROOT, "temp", "analysis")
GROUPS = {"infantry": ["inf"], "support": ["art", "at", "aa"], "armor": ["arm", "mot"],
          "air": ["fig", "cas", "tac", "nav"], "naval": ["scr", "cap", "sub"],
          "civ factories built": ["b_civ"], "mil factories built": ["b_mil"]}
TARGET = {1: ["support"], 2: ["civ factories built"], 3: ["armor"], 4: ["air"], 5: ["naval"],
          6: ["mil factories built", "infantry"]}


def meta(run):
    p = os.path.join(GAMES, run, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load(run):
    """[[date, {tag: minimal record}], ...] per month, cached in temp/analysis/<run>.json."""
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, run + ".json")
    d = os.path.join(GAMES, run, "history_dump")
    fs = month_files(d)
    if os.path.exists(cp):
        c = json.load(open(cp))
        if c["n"] == len(fs):
            return c["months"]
    months = []
    for p in fs:
        j = json.load(open(p, encoding="utf-8", errors="replace"))
        lg = parse_logs(j.get("logs", []))
        recs = {}
        for t, c in j["countries"].items():
            if c.get("exists", True) is False:
                continue
            r = country_record(t, c, lg.get(t, {}), j["date"])
            prod = dict(r.get("prod") or {})
            b = str(prod.pop("bld", "")).split("/")
            if len(b) == 3:
                prod["b_civ"], prod["b_mil"] = float(b[0]), float(b[1])
            recs[t] = {"mil": r["mil"], "civ": r["civ"], "div": r["divisions"], "st": r["n_states"],
                       "pos": (r.get("jev") or {}).get("pos"), "prod": prod}
        months.append([j["date"], recs])
    json.dump({"n": len(fs), "months": months}, open(cp, "w"))
    return months


def power(r):
    return r["mil"] + 0.5 * r["civ"] + 0.3 * r["div"]


def outcome(months, i, tag, h):
    """(power growth, change in states) over h months from month i; None if not observable."""
    if i + h >= len(months):
        return None
    r0 = months[i][1].get(tag)
    if not r0 or r0["st"] == 0:
        return None
    r1 = months[i + h][1].get(tag)
    if not r1 or r1["st"] == 0:
        return (-1.0, -r0["st"])
    return ((power(r1) - power(r0)) / max(power(r0), 1.0), r1["st"] - r0["st"])


def baselines(games, h):
    """(tag, date) -> {game: outcome} over all games given."""
    by = collections.defaultdict(dict)
    for g, months in games.items():
        for i, (date, recs) in enumerate(months):
            for t in recs:
                o = outcome(months, i, t, h)
                if o:
                    by[(t, date)][g] = o
    return by


def residual(by, g, tag, date, o):
    """Outcome minus the same country's mean outcome over the same months in the other games."""
    others = [v for k, v in by.get((tag, date), {}).items() if k != g]
    if not others:
        return None
    return (o[0] - sum(x[0] for x in others) / len(others), o[1] - sum(x[1] for x in others) / len(others))


def boot(per_game, stat, B=2000, seed=0):
    """per_game: game -> aggregate; stat(list of aggregates) -> float or None. Point + 95% over games."""
    rng = random.Random(seed)
    gs = list(per_game)
    point = stat([per_game[g] for g in gs])
    vals = sorted(v for v in (stat([per_game[rng.choice(gs)] for _ in gs]) for _ in range(B)) if v is not None)
    if point is None or not vals:
        return point, None, None
    return point, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def mean_of(aggs, key):
    s = sum(a[key][0] for a in aggs if key in a)
    n = sum(a[key][1] for a in aggs if key in a)
    return s / n if n else None


def manipulation(games, jev):
    """lag -> game -> (posture, group) -> [actual production, expected production]."""
    agg = {0: {}, 1: {}}
    for g in jev:
        months = games[g]
        ctl = {t for t, v in json.load(open(os.path.join(GAMES, g, "control.json"))).items() if v}
        for lag in (0, 1):
            agg[lag][g] = collections.defaultdict(lambda: [0.0, 0.0])
        for t in ctl:
            inc, pos = [], []
            for i in range(1, len(months)):
                a, b = months[i - 1][1].get(t), months[i][1].get(t)
                if not a or not b or not a["prod"] or not b["prod"]:
                    inc.append(None)
                    pos.append(None)
                    continue
                inc.append({grp: sum(b["prod"].get(k, 0) - a["prod"].get(k, 0) for k in ks) for grp, ks in GROUPS.items()})
                pos.append(b["pos"])  # posture in force during this month
            valid = [x for x in inc if x]
            if len(valid) < 6:
                continue
            mean = {grp: sum(x[grp] for x in valid) / len(valid) for grp in GROUPS}
            for j, p in enumerate(pos):
                if not p:
                    continue
                for lag in (0, 1):
                    x = inc[j + lag] if j + lag < len(inc) else None
                    if not x:
                        continue
                    for grp in GROUPS:
                        cell = agg[lag][g][(int(p), grp)]
                        cell[0] += x[grp]
                        cell[1] += mean[grp]
    return agg


def posture_effects(games, jev, by, h):
    """game -> (posture, metric) -> [sum residual, n] for applied decisions where Jev's top pick was
    'total war economy' (the stratum where the posture was a fixed-odds lottery)."""
    agg, counts = {}, collections.Counter()
    for g in jev:
        months = games[g]
        idx = {d: i for i, (d, _) in enumerate(months)}
        a = collections.defaultdict(lambda: [0.0, 0])
        for line in open(os.path.join(GAMES, g, "actions.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            i = idx.get(r["date"])
            if i is None or i + 1 >= len(months):
                continue
            # applied = the posture shows in the telemetry of one of the next two months (a decision taken
            # while a game is still starting up, or across a VM pause, lands a few days after the next dump)
            seen = [months[j][1].get(r["tag"]) for j in (i + 1, i + 2) if j < len(months)]
            if not any(x and x["pos"] is not None and int(x["pos"]) == r["k"] for x in seen):
                counts["not applied"] += 1
                continue
            greedy = max(range(len(r["probs"])), key=r["probs"].__getitem__) + 1
            if greedy != 6:
                counts["other stratum"] += 1
                continue
            o = outcome(months, i, r["tag"], h)
            res = residual(by, g, r["tag"], r["date"], o) if o else None
            if not res:
                counts["no outcome yet"] += 1
                continue
            counts["used"] += 1
            for m, v in (("growth", res[0]), ("states", res[1])):
                c = a[(r["k"], m)]
                c[0] += v
                c[1] += 1
        agg[g] = a
    return agg, counts


def policy_effect(games, jev, by, h):
    """game -> (jev|vanilla, metric) -> [sum residual, n] over eligible country-months."""
    agg = {}
    for g in jev:
        months = games[g]
        ctl = json.load(open(os.path.join(GAMES, g, "control.json")))
        a = collections.defaultdict(lambda: [0.0, 0])
        for i, (date, recs) in enumerate(months):
            for t, r in recs.items():
                if t not in ctl or r["mil"] + r["civ"] < 20:
                    continue
                o = outcome(months, i, t, h)
                res = residual(by, g, t, date, o) if o else None
                if not res:
                    continue
                for m, v in (("growth", res[0]), ("states", res[1])):
                    c = a[("jev" if ctl[t] else "vanilla", m)]
                    c[0] += v
                    c[1] += 1
        agg[g] = a
    return agg


def fmt(p, lo, hi, scale=1.0, unit=""):
    if p is None:
        return "n/a"
    if lo is None:
        return f"{p * scale:+.2f}{unit}"
    return f"{p * scale:+.2f}{unit} [{lo * scale:+.2f}, {hi * scale:+.2f}]"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=6, help="outcome horizon in months")
    ap.add_argument("--pv", type=int, default=VERSION, help="posture version to analyse (games launched with that mod)")
    a = ap.parse_args(argv)
    runs = sorted(d for d in os.listdir(GAMES) if os.path.isdir(os.path.join(GAMES, d, "history_dump")))
    hist = [r for r in runs if "nonhist" not in str(meta(r).get("save", "hist"))]
    games = {}
    for r in hist:
        games[r] = load(r)
        print(f"loaded {r}: {len(games[r])} months", file=sys.stderr, flush=True)
    jev = [r for r in hist if meta(r).get("jev") and int(meta(r).get("pv", 1)) == a.pv
           and os.path.exists(os.path.join(GAMES, r, "control.json"))
           and any(x.get("pos") is not None for _, recs in games[r][:3] for x in recs.values())]
    print(f"historical-save games: {len(hist)} ({', '.join(hist)}); Jev games with posture telemetry: {', '.join(jev)}")
    by = baselines(games, a.h)

    print("\n1) Does each posture change production? (actual / the country's usual; >1 = more)")
    man = manipulation(games, jev)
    for k in range(1, 7):
        for grp in TARGET[k]:
            cells = []
            for lag in (0, 1):
                per = {g: man[lag][g] for g in jev}
                st = lambda s, k=k, grp=grp: (lambda act, exp: act / exp if exp else None)(
                    sum(x[(k, grp)][0] for x in s if (k, grp) in x), sum(x[(k, grp)][1] for x in s if (k, grp) in x))
                p, lo, hi = boot(per, st)
                cells.append("n/a" if p is None else (f"{p:.2f}" + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None else "")))
            print(f"  {k} {POSTURES[k - 1]:21s} {grp:20s} same month {cells[0]:22s} next month {cells[1]}")

    print(f"\n2) Posture vs 'total war economy' when Jev's top pick was total war (randomised), {a.h}-month outcomes")
    pe, counts = posture_effects(games, jev, by, a.h)
    print(f"   decisions: {dict(counts)}")
    for k in range(1, 6):
        n = sum(x[(k, "growth")][1] for x in pe.values() if (k, "growth") in x)
        out = []
        for m, scale, unit in (("growth", 100, " pp"), ("states", 1, " states")):
            st = lambda s, k=k, m=m: (lambda x, y: None if x is None or y is None else x - y)(mean_of(s, (k, m)), mean_of(s, (6, m)))
            out.append(fmt(*boot(pe, st), scale=scale, unit=unit))
        print(f"  {POSTURES[k - 1]:21s} n={n:4d}  power growth {out[0]:30s} territory {out[1]}")
    n6 = sum(x[(6, "growth")][1] for x in pe.values() if (6, "growth") in x)
    print(f"  (reference: total war economy n={n6})")

    print(f"\n3) Jev-controlled vs uncontrolled countries in Jev games, {a.h}-month outcomes")
    po = policy_effect(games, jev, by, a.h)
    for m, scale, unit in (("growth", 100, " pp"), ("states", 1, " states")):
        st = lambda s, m=m: (lambda x, y: None if x is None or y is None else x - y)(mean_of(s, ("jev", m)), mean_of(s, ("vanilla", m)))
        nj = sum(x[("jev", m)][1] for x in po.values() if ("jev", m) in x)
        nv = sum(x[("vanilla", m)][1] for x in po.values() if ("vanilla", m) in x)
        print(f"  {m:7s} jev - vanilla {fmt(*boot(po, st), scale=scale, unit=unit):32s} (country-months: jev {nj}, vanilla {nv})")


if __name__ == "__main__":
    main(sys.argv[1:])
