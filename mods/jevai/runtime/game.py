"""What the game writes, read back: -dump_history months (<userdir>/history_dump/N.txt, one JSON per game month,
N = months since the start save) and the clock in <userdir>/logs/game.log. A month becomes one record per country:
the observation Jev decides on, and what the dataset labels."""
from __future__ import annotations

import glob
import json
import os
import re
import time
from collections import defaultdict

LOG_KV = re.compile(r"^(\w+)\|([A-Z0-9]{3})\|(.*)$")
LOG_DATE = re.compile(r"\]\[(1[89]\d\d\.\d+\.\d+)\.\d+\]")


def month_files(dump_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(dump_dir, "[0-9]*.txt")), key=lambda p: int(os.path.basename(p)[:-4]))


def read_month(path: str, tries: int = 5) -> dict | None:
    """One month; a file the game is still writing fails to parse, so retry a few times."""
    for i in range(tries):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return json.load(f)
        except json.JSONDecodeError:
            if i + 1 < tries:
                time.sleep(2)
    return None


def last_date(userdir: str) -> str:
    """Newest game date in game.log ('' if none): shows whether a game has loaded, runs, or stalls."""
    path = os.path.join(userdir, "logs", "game.log")
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 20000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    ds = LOG_DATE.findall(tail)
    return ds[-1] if ds else ""


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def parse_logs(lines: list[str]) -> dict[str, dict]:
    """Group the dump's `logs` lines by country: AI internals (manpower, equipment requests, convoys, supply, active
    strategy plans), the month's events, and the mod's telemetry (JEV|T -> jev, JEV|P -> prod)."""
    out: dict[str, dict] = defaultdict(lambda: {"strats": [], "events": [], "manpower": {}, "equipment": {},
                                                 "convoy": {}, "supply": {}, "garrison": {}})
    for line in lines:
        m = LOG_KV.match(line)
        if not m:
            continue
        kind, tag, rest = m.groups()
        c = out[tag]
        if kind == "JEV":
            # the dump prefixes every log line with kind|tag|date: , so ours read "JEV|TAG|date: T|TAG|k=v|..."
            f = rest.partition(": ")[2].split("|")
            if f and f[0] in ("T", "P"):
                key = "jev" if f[0] == "T" else "prod"
                for kv in f[2:]:
                    k, _, v = kv.partition("=")
                    c.setdefault(key, {})[k] = _num(v) if _num(v) is not None else v
        elif kind == "strat":
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
            c["events"].append((kind, rest.partition(": ")[2]))
    return out


def load_invariants(dump_dir: str) -> dict[int, list]:
    """State id -> map position of its central province (the dump's invariants.json)."""
    with open(os.path.join(dump_dir, "invariants.json"), encoding="utf-8") as f:
        inv = json.load(f)
    centers = inv["provinces_center_point"]
    return {sid: centers[prov] for sid, prov in enumerate(inv["state_id_to_central_province"]) if prov and prov < len(centers)}


def country_record(tag: str, c: dict, lg: dict, date: str) -> dict:
    f = c.get("factories") or {}
    states = c.get("fully_controlled_states") or []
    orders = c.get("order_data") or {}
    eq_fill = {}
    for eq, v in lg.get("equipment", {}).items():
        req, stock = v.get("requested") or 0, v.get("in stock") or 0
        if req > 0:
            eq_fill[eq] = round(min(1.0, stock / req), 3)
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
        "manpower": lg.get("manpower", {}), "equipment_fill": eq_fill,
        "convoys_free": lg.get("convoy", {}).get("free"), "strats": lg.get("strats", []),
        "events": lg.get("events", []), "jev": lg.get("jev", {}), "prod": lg.get("prod", {}),
    }


def neighbors(recs: dict[str, dict], st_center: dict, radius: float = 70.0) -> dict[str, list[str]]:
    """Approximate land adjacency: two countries neighbour if any pair of their state centres is within `radius` map
    pixels (the map is 5632x2048). Cheap, and good enough to pick the neighbours that matter."""
    pts = {t: [st_center[s] for s in r["states"] if s in st_center] for t, r in recs.items()}
    tags = [t for t in pts if pts[t]]
    nb = {t: set() for t in tags}
    r2 = radius * radius
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < r2 for p in pts[a] for q in pts[b]):
                nb[a].add(b)
                nb[b].add(a)
    return {t: sorted(v) for t, v in nb.items()}


def month_records(j: dict, st_center: dict) -> dict[str, dict]:
    """Every existing country of one dump month, with its approximate land neighbours."""
    lg = parse_logs(j.get("logs", []))
    recs = {t: country_record(t, c, lg.get(t, {}), j["date"])
            for t, c in j["countries"].items() if c.get("exists", True) is not False}
    nb = neighbors(recs, st_center)
    for t, r in recs.items():
        r["neighbors"] = nb.get(t, [])
    return recs
