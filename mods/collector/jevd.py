"""jevd: Jev takes part in the collector's games (VM only).

For every game the collector registers (temp/inst/<slot>.jev), each decision month: read the newest history_dump
month and, for each eligible country Jev controls, score all six postures with the posture-outcome questions the
model is trained on (runtime.text.posture_questions: power growth level, no territory lost, territory gained,
`horizon` months ahead), then pick the best:

    U = w_ok * P(no territory lost) + w_gain * P(territory gained) + w_grow * E[growth level] / 3

With probability eps the posture is drawn uniformly instead. Every decision goes to temp/games/<run>/actions.jsonl
with the per-posture scores, the propensity of the posture taken and whether it explored, so the outcomes read from
the dump later (datasets.build) are causal labels, with inverse-propensity weights available. Which countries Jev
controls is a coin flip per (game, country) saved in control.json; the others stay vanilla as the control group.

    python -m mods.collector.jevd [--model temp/train/<run>/best] [--eps 0.3] [--frac 0.5] [--horizon 6]
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import random
import time

import torch

from mods.jevai.runtime.console import Console
from mods.jevai.runtime.game import last_date, load_invariants, month_files, month_records, read_month
from mods.jevai.runtime.pack import Packer
from mods.jevai.runtime.postures import NAMES as POSTURES, VERSION
from mods.jevai.runtime.text import GROWTH_LEVELS, posture_questions, state_text
from trainer.model import load, to_torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMP = os.path.join(ROOT, "temp")
INST = os.path.join(TEMP, "inst")
MODEL = os.path.join(ROOT, "models", "open-jev-deberta-v3-large")


def pid_of(inst: str) -> int:
    p = os.path.join(INST, inst + ".pid")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return int(f.read().strip() or 0)


class Game:
    """One registered game: its userdir and dump, action log, control coin flips and policy settings."""

    def __init__(self, reg: dict):
        self.name, self.userdir, self.run = reg["inst"], reg["userdir"], reg["run"]
        self.settings = {k: reg[k] for k in ("every", "eps", "pv", "horizon") if k in reg}
        self.dump = os.path.join(self.userdir, "history_dump")
        self.out = os.path.join(TEMP, "games", self.run)
        os.makedirs(self.out, exist_ok=True)
        self.actions = open(os.path.join(self.out, "actions.jsonl"), "a", encoding="utf-8")
        self.console = Console(self.userdir, lambda: pid_of(self.name), range(5, 10))  # the collector uses acks 0-4
        # a fresh game acts on its newest month too; a resumed one (actions already logged) waits for the next
        resumed = os.path.getsize(os.path.join(self.out, "actions.jsonl")) > 0
        fs = month_files(self.dump)
        self.done = {os.path.basename(p) for p in (fs if resumed else fs[:-1])}
        self.month_n = -1  # dump index of the month in hand (months since the start save)
        self.st_center = None
        cp = os.path.join(self.out, "control.json")
        self.control: dict[str, bool] = json.load(open(cp)) if os.path.exists(cp) else {}

    def controlled(self, tag: str, frac: float) -> bool:
        """Coin flip seeded by (game, country): independent across games and the same after a restart."""
        if tag not in self.control:
            self.control[tag] = random.Random(f"{self.run}|{tag}|control").random() < frac
            with open(os.path.join(self.out, "control.json"), "w") as f:
                json.dump(self.control, f, indent=0)
        return self.control[tag]

    def new_month(self) -> dict | None:
        fs = month_files(self.dump)
        if not fs or os.path.basename(fs[-1]) in self.done:
            return None
        j = read_month(fs[-1])
        if j is None:
            return None
        self.done.add(os.path.basename(fs[-1]))
        self.month_n = int(os.path.basename(fs[-1])[:-4])
        if self.st_center is None:
            self.st_center = load_invariants(self.dump)
        return j


@torch.no_grad()
def decide_value(net, packer, states: list[str], pv: int, horizon: int, weights: tuple[float, float, float],
                 bs: int = 12, device: str = "cpu") -> list[list[dict]]:
    """Score every posture for every state: one sequence per (state, posture) with its three questions (all six
    postures in one sequence would not fit the 512-token window). Per state, one dict per posture (POSTURES order):
    ok = P(no territory lost), gain = P(territory gained), grow = expected growth level 0..3, U = the score."""
    w_ok, w_gain, w_grow = weights
    top = len(GROWTH_LEVELS) - 1
    items = [(s, posture_questions(p, pv, horizon)) for s in states for p in POSTURES]
    parts = []
    for i in range(0, len(items), bs):
        b = to_torch(packer(items[i:i + bs]), device)
        logits = net(b["input_ids"], b["attention_mask"], b["pool"], b["opt_mask"]).float()
        for row in (logits / net.temperature).softmax(-1).cpu():
            level = sum(k * x for k, x in enumerate(row[0, : len(GROWTH_LEVELS)].tolist()))
            ok, gain = float(row[1, 1]), float(row[2, 1])  # noul options are (no, yes)
            parts.append({"ok": round(ok, 4), "gain": round(gain, 4), "grow": round(level, 3),
                          "U": round(w_ok * ok + w_gain * gain + w_grow * level / top, 5)})
    n = len(POSTURES)
    return [parts[i * n:(i + 1) * n] for i in range(len(states))]


def step(g: Game, net, packer, cfg: dict) -> int:
    every = int(g.settings.get("every", cfg["every"]))
    eps = float(g.settings.get("eps", cfg["eps"]))
    pv = int(g.settings.get("pv", VERSION))
    horizon = int(g.settings.get("horizon", cfg["horizon"]))
    j = g.new_month()
    if j is None or g.month_n % every:  # every country is re-decided on the same months (by dump index)
        return 0
    recs = month_records(j, g.st_center)
    names = {t: r["name"] for t, r in recs.items()}
    elig = [t for t, r in recs.items() if r["n_states"] > 0 and r["mil"] + r["civ"] >= cfg["min_mil"] and g.controlled(t, cfg["frac"])]
    if not elig:
        return 0
    states = [state_text(recs[t], names, recs) for t in elig]
    t0 = time.time()
    scored = decide_value(net, packer, states, pv, horizon, cfg["weights"])
    cmds, rows = [], []
    for t, st, sc in zip(elig, states, scored):
        tot = sum(x["U"] for x in sc) or 1.0
        probs = [x["U"] / tot for x in sc]  # Jev's preference over postures: the scores normalised to sum 1
        greedy = max(range(len(POSTURES)), key=probs.__getitem__)
        rr = random.Random(f"{g.run}|{g.month_n}|{t}|posture")  # per game, month and country
        explore = rr.random() < eps
        k = rr.randrange(len(POSTURES)) if explore else greedy
        prop = eps / len(POSTURES) + (1 - eps) * (k == greedy)  # explore draws can land on the greedy posture too
        cmds.append(f"e {t} jev_set_posture_{k + 1}")
        rows.append({"date": j["date"], "tag": t, "posture": POSTURES[k], "k": k + 1, "explore": explore and k != greedy,
                     "pv": pv, "eps": eps, "propensity": round(prop, 4), "probs": [round(x, 4) for x in probs],
                     "rule": "value", "model": cfg["model_id"], "scores": sc, "horizon": horizon, "state": st})
    ok = g.console.run_file(cmds) if pid_of(g.name) else False
    # the game runs on while Jev thinks, so record when the postures actually took effect
    when = last_date(g.userdir) if ok else ""
    applied_date = ".".join(str(int(x)) for x in when.split(".")) if when else None
    for r in rows:
        r["applied"], r["applied_date"] = ok, applied_date  # not applied: the previous posture stays in force
        g.actions.write(json.dumps(r, ensure_ascii=False) + "\n")
    g.actions.flush()
    print(f"[{time.strftime('%H:%M:%S')}] {g.name} {j['date']}: {len(cmds)} countries, jev+apply {time.time() - t0:.1f}s"
          f"{'' if ok else ' NOT CONFIRMED'}", flush=True)
    return len(cmds)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Jev in the loop for the collector's games (VM only)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--eps", type=float, default=0.3, help="exploration rate (a game's registration overrides it)")
    ap.add_argument("--frac", type=float, default=0.5, help="share of eligible countries Jev controls")
    ap.add_argument("--every", type=int, default=2, help="months between decisions (a game's registration overrides it)")
    ap.add_argument("--horizon", type=int, default=6, help="months the posture questions look ahead")
    ap.add_argument("--min-mil", type=int, default=20, help="skip countries with fewer military + civilian factories")
    ap.add_argument("--w-ok", type=float, default=1.0, help="weight of P(no territory lost)")
    ap.add_argument("--w-gain", type=float, default=1.0, help="weight of P(territory gained)")
    ap.add_argument("--w-grow", type=float, default=0.5, help="weight of the expected growth level (0..1)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    model_dir = os.path.abspath(a.model)
    cfg = {"eps": a.eps, "every": a.every, "horizon": a.horizon, "min_mil": a.min_mil, "frac": a.frac,
           "weights": (a.w_ok, a.w_gain, a.w_grow),
           "model_id": "base" if model_dir == MODEL else os.path.relpath(model_dir, ROOT).replace(os.sep, "/")}
    if os.name == "nt":  # below-normal priority: inference bursts must not make the desktop lag
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(a.threads or max(1, (os.cpu_count() or 4) // 2))  # leave cores for the games
    net, mcfg = load(model_dir, "cpu")
    packer = Packer(model_dir, mcfg.get("max_state_tokens", 256), mcfg.get("max_len", 512))
    games: dict[str, Game] = {}
    print(f"jevd up: model={cfg['model_id']} horizon={a.horizon} weights={cfg['weights']} postures={POSTURES}", flush=True)
    while True:
        live = set()
        for p in glob.glob(os.path.join(INST, "*.jev")):
            try:
                with open(p) as f:
                    reg = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            live.add(reg["run"])
            if reg["run"] not in games:
                games[reg["run"]] = Game(reg)
                print(f"attached {reg['run']} on {reg['inst']} {games[reg['run']].settings}", flush=True)
        for run in [r for r in games if r not in live]:
            games.pop(run).actions.close()
            print(f"detached {run}", flush=True)
        n = 0
        for g in list(games.values()):
            try:
                n += step(g, net, packer, cfg)
            except Exception as e:  # noqa: BLE001 - one bad month must not kill the daemon
                print(f"{g.run}: step failed: {e!r}", flush=True)
        if n == 0:
            time.sleep(3)


if __name__ == "__main__":
    main()
