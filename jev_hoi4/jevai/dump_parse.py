"""Parse HOI4 `-dump_history` output (history_dump/N.txt, one JSON per game month) into
per-country monthly records, and attach outcome labels from later months.

    python -m jevai.dump_parse <history_dump dir> <out.jsonl> [--horizons 3,6,12]

A record is the observation for one country at one month (what the AI saw) plus, for each
horizon h, the change h months later (what happened). Labels come only from the game itself.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

LOG_KV = re.compile(r"^(\w+)\|([A-Z0-9]{3})\|(.*)$")


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def parse_logs(lines: list[str]) -> dict[str, dict]:
    """Group the dump's `logs` lines by country. Per-country AI internals (manpower, equipment
    requests, convoys, supply, active strategy plans) plus the month's events (focus, decision, war)."""
    out: dict[str, dict] = defaultdict(lambda: {"strats": [], "events": [], "manpower": {}, "equipment": {},
                                                 "convoy": {}, "supply": {}, "garrison": {}})
    for l in lines:
        m = LOG_KV.match(l)
        if not m:
            continue
        kind, tag, rest = m.groups()
        c = out[tag]
        if kind == "JEV":
            # our mod's telemetry: "JEV|TAG|date: T|TAG|k=v|k=v..." (the dump prefixes kind|tag|date: )
            _, _, text = rest.partition(": ")
            f = text.split("|")
            if f and f[0] in ("T", "P"):
                key = "jev" if f[0] == "T" else "prod"
                for kv in f[2:]:
                    k, _, v = kv.partition("=")
                    c.setdefault(key, {})[k] = _num(v) if _num(v) is not None else v
            continue
        if kind == "strat":
            c["strats"].append(rest)
        elif kind in ("manpower", "garrison", "supply"):
            k, _, v = rest.partition(":")
            c[kind][k.strip()] = _num(v.strip())
        elif kind == "convoy":
            k, _, v = rest.partition("=")
            c["convoy"][k] = _num(v)
        elif kind == "equipment":
            eq, _, kv = rest.partition("|")
            k, _, v = kv.partition(":")
            c["equipment"].setdefault(eq, {})[k.strip()] = _num(v.strip())
        elif kind in ("focus", "decision", "idea", "untagged", "update_template", "diplomacy"):
            _, _, text = rest.partition(": ")
            c["events"].append((kind, text))
    return out


def load_invariants(d: str):
    inv = json.load(open(os.path.join(d, "invariants.json")))
    centers = inv["provinces_center_point"]
    st_center = {}
    for sid, prov in enumerate(inv["state_id_to_central_province"]):
        if prov and prov < len(centers):
            st_center[sid] = centers[prov]
    return st_center


def country_record(tag: str, c: dict, lg: dict, date: str) -> dict:
    f = c.get("factories") or {}
    states = c.get("fully_controlled_states") or []
    orders = c.get("order_data") or {}
    eq_def = {}
    for eq, v in lg.get("equipment", {}).items():
        req, stock = v.get("requested") or 0, v.get("in stock") or 0
        if req > 0:
            eq_def[eq] = round(min(1.0, stock / req), 3)
    return {
        "tag": tag, "name": c.get("name", tag), "date": date,
        "mil": f.get("mil_controlled", 0), "civ": f.get("civ_controlled", 0), "nav": f.get("nav_controlled", 0),
        "civ_consumer": f.get("civ_consumer_goods", 0), "civ_trade": f.get("civ_trade_export", 0) + f.get("civ_trade_import", 0),
        "states": sorted(states), "n_states": len(states),
        "divisions": len(c.get("armies") or []), "fleets": len(c.get("navies") or []),
        "air_missions": len(c.get("air_missions") or []),
        "fronts": sum(1 for o in orders.values() if o.get("type") == 2),
        "front_units": sum(o.get("num_units_at_front", 0) for o in orders.values() if o.get("type") == 2),
        "enemies": c.get("enemies") or [], "allies": c.get("allies") or [], "potential_enemies": c.get("potential_enemies") or [],
        "occupied_states": len(c.get("resistance_data") or {}),
        "manpower": lg.get("manpower", {}), "equipment_fill": eq_def,
        "convoys_free": lg.get("convoy", {}).get("free"), "strats": lg.get("strats", []),
        "events": lg.get("events", []), "jev": lg.get("jev", {}), "prod": lg.get("prod", {}),
    }


def neighbors(recs: dict[str, dict], st_center: dict, radius: float = 70.0) -> dict[str, list[str]]:
    """Approximate land adjacency: two countries neighbour if any pair of their state centres is within
    `radius` map pixels (map is 5632x2048). Cheap and good enough to pick candidate targets/allies."""
    pts = {t: [st_center[s] for s in r["states"] if s in st_center] for t, r in recs.items()}
    tags = [t for t in pts if pts[t]]
    nb = {t: set() for t in tags}
    r2 = radius * radius
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < r2 for p in pts[a] for q in pts[b]):
                nb[a].add(b); nb[b].add(a)
    return {t: sorted(v) for t, v in nb.items()}


def power(r: dict) -> float:
    return r["mil"] * 1.0 + r["civ"] * 0.5 + r["divisions"] * 0.3


def outcome(now: dict, later: dict | None) -> dict:
    """What happened between two observations of the same country. `later` None = country is gone."""
    if later is None or (later["n_states"] == 0 and now["n_states"] > 0):
        return {"exists": 0}
    d_states = later["n_states"] - now["n_states"]
    p0, p1 = power(now), power(later)
    return {"exists": 1, "d_states": d_states, "d_mil": later["mil"] - now["mil"], "d_civ": later["civ"] - now["civ"],
            "d_div": later["divisions"] - now["divisions"],
            "power_growth": round((p1 - p0) / max(p0, 1.0), 4),
            "new_enemies": sorted(set(later["enemies"]) - set(now["enemies"])),
            "capitulated": 0}


def parse_dir(d: str, horizons=(3, 6, 12)):
    st_center = load_invariants(d)
    months = []
    for p in sorted(glob.glob(os.path.join(d, "[0-9]*.txt")), key=lambda p: int(os.path.basename(p)[:-4])):
        j = json.load(open(p, encoding="utf-8", errors="replace"))
        lg = parse_logs(j.get("logs", []))
        recs = {t: country_record(t, c, lg.get(t, {}), j["date"]) for t, c in j["countries"].items() if c.get("exists", True) is not False}
        nb = neighbors(recs, st_center)
        for t, r in recs.items():
            r["neighbors"] = nb.get(t, [])
        months.append((j["date"], recs))
    # state ownership per month so we can say who took whose states
    owner = [{s: t for t, r in recs.items() for s in r["states"]} for _, recs in months]
    for i, (date, recs) in enumerate(months):
        for t, r in recs.items():
            r["outcome"] = {}
            for h in horizons:
                if i + h >= len(months):
                    continue
                later = months[i + h][1].get(t)
                o = outcome(r, later)
                took = defaultdict(int); lost = defaultdict(int)
                for s, t0 in owner[i].items():
                    t1 = owner[i + h].get(s)
                    if t1 and t1 != t0:
                        if t1 == t: took[t0] += 1
                        if t0 == t: lost[t1] += 1
                o["took_from"] = dict(took); o["lost_to"] = dict(lost)
                o["capitulated"] = int(o["exists"] == 0)
                r["outcome"][str(h)] = o
            yield r


def main(argv):
    src, dst = argv[0], argv[1]
    hz = (3, 6, 12)
    if "--horizons" in argv:
        hz = tuple(int(x) for x in argv[argv.index("--horizons") + 1].split(","))
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        for r in parse_dir(src, hz):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} country-months -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
