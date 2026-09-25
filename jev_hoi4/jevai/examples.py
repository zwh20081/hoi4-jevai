"""Turn parsed country-month records into open-jev Examples (state text + typed questions + gold).

Two question families, both labelled only by what the game later did:

1. Forecast questions ("what will happen to this country"), gold = observed outcome h months later.
   These teach the encoder HOI4 dynamics; they need no intervention and every observer game yields
   ~90 countries x N months of them.
2. Action-value questions ("if this country follows posture P, does it gain?"), gold = observed outcome
   of runs where the daemon randomly assigned posture P (jev_posture recorded in the record). Only
   records with meta.action are used, so the label is causal for that randomised action.

    python -m jevai.examples runs/data/run0.jsonl runs/data/run0.examples.jsonl --horizon 3
"""
from __future__ import annotations

import json
import random
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from typed_decisions.schema import Example, Question, NOUL_OPTIONS  # noqa: E402

GROWTH_LEVELS = ["shrinks", "stagnates", "grows slowly", "grows fast"]


def growth_level(g: float) -> int:
    return 0 if g < -0.02 else 1 if g < 0.03 else 2 if g < 0.12 else 3


def state_text(r: dict, names: dict[str, str], by_tag: dict[str, dict]) -> str:
    """<=256 DeBERTa tokens: who we are, strength, war situation, and the neighbours that matter."""
    def nm(t):
        return names.get(t, t)
    mp = r.get("manpower", {})
    j = r.get("jev") or {}
    head = f"Country: {r['name']}. Date: {r['date']}."
    if j.get("ideo"):
        head += f" Government: {j['ideo']}."
    if j.get("fac"):
        head += f" Faction: {j['fac']}."
    parts = [head,
             f"Factories: {r['mil']} military, {r['civ']} civilian, {r['nav']} naval. States: {r['n_states']}.",
             f"Army: {r['divisions']} divisions, {r['front_units']} on {r['fronts']} fronts. Fleets: {r['fleets']}."]
    if "stab" in j:
        parts.append(f"Stability {int(100 * j['stab'])}%, war support {int(100 * j.get('ws', 0))}%.")
    if r["enemies"] and "str" in j:
        parts.append(f"Army strength vs enemies x{j['str']:.1f}, with allies x{j.get('astr', 0):.1f}. Surrender progress {int(100 * j.get('surr', 0))}%.")
    if mp.get("max"):
        parts.append(f"Manpower available {int(mp.get('available', 0) / 1000)}k of {int(mp['max'] / 1000)}k.")
    fill = r.get("equipment_fill", {})
    short = [k.replace("_equipment", "") for k, v in fill.items() if v < 0.5]
    if short:
        parts.append("Equipment shortage: " + ", ".join(short[:4]) + ".")
    if r["enemies"]:
        parts.append("At war with: " + ", ".join(nm(t) for t in r["enemies"][:5]) + ".")
    else:
        parts.append("At peace.")
    if r["allies"]:
        parts.append("Allies: " + ", ".join(nm(t) for t in r["allies"][:5]) + (f" and {len(r['allies']) - 5} more" if len(r["allies"]) > 5 else "") + ".")
    my = r["mil"] + 0.3 * r["divisions"]
    nbs = sorted((t for t in r.get("neighbors", []) if t in by_tag), key=lambda t: -(by_tag[t]["mil"] + 0.3 * by_tag[t]["divisions"]))[:6]
    if nbs:
        desc = []
        for t in nbs:
            o = by_tag[t]
            ratio = (o["mil"] + 0.3 * o["divisions"]) / max(my, 1)
            rel = "enemy" if t in r["enemies"] else "ally" if t in r["allies"] else "at war" if o["enemies"] else "neutral"
            desc.append(f"{nm(t)} ({rel}, {o['mil']} mil, {o['divisions']} div, strength x{ratio:.1f})")
        parts.append("Neighbours: " + "; ".join(desc) + ".")
    return " ".join(parts)


def noul(qid, text, yes: bool) -> Question:
    return Question(qid, "noul", text, list(NOUL_OPTIONS), int(bool(yes)))


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
    # who it fights next: a choice among candidates, only when there is a new enemy to name
    if o["new_enemies"]:
        cands = [t for t in r.get("neighbors", []) if t not in r["enemies"]]
        gold = [t for t in o["new_enemies"] if t in cands]
        if gold and len(cands) >= 2:
            cands = cands[:9] if gold[0] in cands[:9] else cands[:8] + [gold[0]]
            rng.shuffle(cands)
            qs.append(Question("whowar", "choice", f"Which country will this country be at war with next?",
                               [names.get(t, t) for t in cands], cands.index(gold[0])))
    # territory flow with a named neighbour
    for t, n in list(o.get("took_from", {}).items())[:1]:
        qs.append(noul("took", f"This country will take territory from {names.get(t, t)} within {months}.", True))
    if not o.get("took_from") and r["enemies"]:
        t = rng.choice(r["enemies"])
        qs.append(noul("took", f"This country will take territory from {names.get(t, t)} within {months}.", False))
    return qs


def action_texts(posture: str, pv: int, h) -> tuple[str, str, str]:
    """The three posture-outcome question texts (growth, no territory lost, territory gained). Training
    (action_questions) and deciding (jevd, rule 'value') must use the exact same wording, so both call this."""
    s = f"strategy set v{pv}"  # v1/v2 postures share names but differ in content
    return (f"This country follows the posture '{posture}' ({s}) for {h} months. How does its power change?",
            f"Following the posture '{posture}' ({s}), this country avoids losing territory for {h} months.",
            f"Following the posture '{posture}' ({s}), this country gains territory within {h} months.")


def action_questions(r: dict, h: str) -> list[Question]:
    """Questions about a posture Jev applied at this month (one set per decision, only decisions the game
    confirmed). The posture was drawn with known odds (eps-greedy), so the outcome is causal for it:
    "this posture, then Jev's policy, for h months". The question names the posture set (v1/v2 differ
    in content under the same names), so the model can tell them apart; deployment asks with v2."""
    m = r.get("meta") or {}
    a = m.get("action")
    o = r["outcome"].get(h)
    if not a or not o or not m.get("action_fresh") or not m.get("action_applied", True):
        return []
    g = growth_level(o.get("power_growth", 0.0)) if o["exists"] else 0
    t_grow, t_ok, t_gain = action_texts(a["posture"], a.get("pv", 1), h)
    return [Question("act_grow", "choice", t_grow, list(GROWTH_LEVELS), g),
            noul("act_ok", t_ok, o["exists"] and o["d_states"] >= 0),
            noul("act_gain", t_gain, o["exists"] and o["d_states"] > 0)]


def attach_actions(records: list[dict], actions: list[dict]):
    """Put each daemon action on the record of the same (tag, date); a posture stays in force until the
    next decision, so records in between inherit it (action_fresh = False). action_applied: the game's
    own telemetry (posture in force, `pos`) shows the posture within the next two months; games without
    that telemetry count every decision as applied."""
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
            m["action"] = cur[r["tag"]]
            m["action_fresh"] = bool(a)
            if a:
                seen = [pos.get((r["tag"], d)) for d in nxt[r["date"]]]
                known = [p for p in seen if p is not None]
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
        out.append(Example(state=state_text(r, names, by_date[r["date"]]), questions=qs, source=f"hoi4:{h}m", meta=meta))
    return out


def main(argv):
    src, dst = argv[0], argv[1]
    h = argv[argv.index("--horizon") + 1] if "--horizon" in argv else "3"
    recs = [json.loads(l) for l in open(src, encoding="utf-8")]
    ex = build(recs, h)
    with open(dst, "w", encoding="utf-8") as f:
        for e in ex:
            f.write(e.to_json() + "\n")
    nq = sum(len(e.questions) for e in ex)
    print(f"{len(ex)} states / {nq} questions -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
