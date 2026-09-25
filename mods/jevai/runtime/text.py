"""Everything the model reads, as text: one country in one month, and the question wording. Training data
(datasets.build) and play-time decisions (the collector's jevd) call these same functions, so the model is asked
exactly what it was trained on."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

NOUL_OPTIONS = ("no", "yes")
GROWTH_LEVELS = ["shrinks", "stagnates", "grows slowly", "grows fast"]


@dataclass
class Question:
    qid: str
    kind: str  # "choice": one of the options | "noul": yes/no with options NOUL_OPTIONS, answer = p(yes)
    instructions: str
    options: list[str]
    gold: int = 0  # index into options (0 when asking rather than training)

    def __post_init__(self):
        assert self.kind != "noul" or tuple(self.options) == NOUL_OPTIONS, "noul options are (no, yes)"
        assert 0 <= self.gold < len(self.options), (self.qid, self.gold, len(self.options))


@dataclass
class Example:
    state: str
    questions: list[Question]
    source: str
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"state": self.state, "source": self.source, "meta": self.meta,
                           "questions": [asdict(q) for q in self.questions]}, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Example":
        d = json.loads(line)
        return Example(d["state"], [Question(**q) for q in d["questions"]], d["source"], d.get("meta", {}))


def read_jsonl(path: str) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [Example.from_json(line) for line in f if line.strip()]


def noul(qid: str, text: str, yes: bool = False) -> Question:
    return Question(qid, "noul", text, list(NOUL_OPTIONS), int(bool(yes)))


def state_text(r: dict, names: dict[str, str], by_tag: dict[str, dict]) -> str:
    """<=256 DeBERTa tokens: who the country is, its strength, its war situation, and the neighbours that matter."""
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


def posture_questions(posture: str, pv: int, h, gold: tuple[int, int, int] = (0, 0, 0)) -> list[Question]:
    """The three posture-outcome questions Jev decides with: power growth level over h months, no territory lost,
    territory gained. They name the posture set, because v1 and v2 share names but not content."""
    s = f"strategy set v{pv}"
    return [Question("act_grow", "choice", f"This country follows the posture '{posture}' ({s}) for {h} months. How does its power change?",
                     list(GROWTH_LEVELS), gold[0]),
            noul("act_ok", f"Following the posture '{posture}' ({s}), this country avoids losing territory for {h} months.", gold[1]),
            noul("act_gain", f"Following the posture '{posture}' ({s}), this country gains territory within {h} months.", gold[2])]
