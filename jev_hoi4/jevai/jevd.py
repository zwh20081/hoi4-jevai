"""jevd: let the Jev model take part in running HOI4 instances.

Every game month, for each instance: read the newest history_dump month, build each controlled
country's state text, let Jev pick a posture per country, and apply it through the console
(`e TAG jev_set_posture_k`). Two decision rules:

- value (default): for every posture, ask the three posture-outcome questions the model is trained on
  (examples.action_texts: growth, no territory lost, territory gained, over `horizon` months) and pick
  the posture with the best score U = w_ok*P(no loss) + w_gain*P(gain) + w_grow*E[growth level]/3.
  This is what training improves: those questions carry labels from the collected games.
- choice: one zero-shot question "which posture should this country follow?" (no training label;
  used for games g05-g33).

Exploration makes the data useful for learning: with probability eps the posture is drawn uniformly
at random instead of Jev's argmax, and every applied action is written to runs/games/<run>/actions.jsonl
with the full Jev distribution, the propensity of the action taken, and whether it was explore/exploit.
Outcomes come later from the dump (dump_parse), so each action becomes a causal training example
(action-value question in examples.py) and inverse-propensity weights are available.

Control is randomised per country per game: `frac` of eligible countries are Jev-controlled; the rest
stay vanilla (the control group).

    python -m jevai.jevd --inst u1:hoi4user:g06 --inst u2:runs/inst/u2:g07 [--eps 0.3] [--frac 0.5] [--model .]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from .dump_parse import country_record, load_invariants, neighbors, parse_logs  # noqa: E402
from .examples import GROWTH_LEVELS, action_texts, state_text  # noqa: E402
from .collector import last_date  # noqa: E402
from .console import Console  # noqa: E402
from .gen_mod import NAMES as POSTURES  # noqa: E402

CONSOLE = os.path.join(ROOT, "jev_hoi4", "p0", "console.ps1")
Q_POSTURE = "Which strategic posture should this country follow for the next 6 months?"


def month_files(d):
    return sorted(glob.glob(os.path.join(d, "[0-9]*.txt")), key=lambda p: int(os.path.basename(p)[:-4]))


def read_month(path):
    for _ in range(5):  # the game may still be writing the file
        try:
            return json.load(open(path, encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            time.sleep(2)
    return None


def pid_of(inst):
    p = os.path.join(ROOT, "runs", "inst", inst + ".pid")
    return int(open(p).read().strip()) if os.path.exists(p) else 0


def run_console(inst: "Instance", cmds: list[str]) -> bool:
    """Write the commands to a file in the userdir and `run` it (one console line regardless of count),
    verified by an ack and retried on a lost key. Returns whether the game confirmed it."""
    path = os.path.join(inst.userdir, "jev_apply.txt")
    open(path, "w", encoding="utf-8").write("\n".join(cmds) + "\n")
    return inst.console.send(["run jev_apply.txt"])


class Instance:
    def __init__(self, spec: str, rng: random.Random, frac: float, settings: dict | None = None):
        """settings (from the collector's registration) override the daemon defaults per game:
        every (months between decisions), sync (all countries decide on the same months),
        eps (exploration), pv (posture version of the mod the game was launched with)."""
        self.settings = settings or {}
        self.name, rest = spec.split(":", 1)
        udir, self.run = rest.rsplit(":", 1)  # userdir may itself contain a drive colon
        self.userdir = udir if os.path.isabs(udir) else os.path.join(ROOT, udir)
        self.dump = os.path.join(self.userdir, "history_dump")
        self.out = os.path.join(ROOT, "runs", "games", self.run)
        os.makedirs(self.out, exist_ok=True)
        self.actions = open(os.path.join(self.out, "actions.jsonl"), "a", encoding="utf-8")
        self.console = Console(self.userdir, lambda: pid_of(self.name), range(5, 10))  # collector uses acks 0-4 (acks 10-19 exist only in games launched after 15:10)
        # fresh game: act on the newest month too; resumed game (actions already logged): wait for the next one
        resumed = os.path.getsize(os.path.join(self.out, "actions.jsonl")) > 0
        fs = month_files(self.dump)
        self.done = {os.path.basename(p) for p in (fs if resumed else fs[:-1])}
        self.month_n = -1  # dump index of the month being processed (months since the start save)
        self.st_center = None
        self.rng, self.frac = rng, frac
        self.control: dict[str, bool] = {}  # tag -> Jev-controlled (fixed per game)
        cp = os.path.join(self.out, "control.json")
        if os.path.exists(cp):
            self.control = json.load(open(cp))

    def controlled(self, tag: str) -> bool:
        # the coin flip is seeded by (game, country): independent across games and the same after a restart
        # (a daemon-wide RNG gave every game attached first after a restart the same flips, 2026-09-25)
        if tag not in self.control:
            self.control[tag] = random.Random(f"{self.run}|{tag}|control").random() < self.frac
            json.dump(self.control, open(os.path.join(self.out, "control.json"), "w"), indent=0)
        return self.control[tag]

    def new_month(self):
        fs = month_files(self.dump)
        if not fs or os.path.basename(fs[-1]) in self.done:
            return None
        p = fs[-1]
        j = read_month(p)
        if j is None:
            return None
        self.done.add(os.path.basename(p))
        self.month_n = int(os.path.basename(p)[:-4])
        if self.st_center is None:
            self.st_center = load_invariants(self.dump)
        return j


def decide(model, states: list[str], bs: int = 16):
    """One posture distribution per state, batched (one forward per `bs` states)."""
    import torch
    q = model._question(0, {"type": "choice", "instructions": Q_POSTURE, "options": POSTURES})
    out = []
    with torch.no_grad():
        for i in range(0, len(states), bs):
            b = model.collator([(s, [q]) for s in states[i:i + bs]], model.device)
            lg = model.model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"]).float()
            p = (lg / model.model.temperature).softmax(-1)[:, 0, : len(POSTURES)]
            out += [dict(zip(POSTURES, row.tolist())) for row in p]
    return out


def decide_value(model, states: list[str], pv: int, horizon: int, weights: tuple[float, float, float], bs: int = 12):
    """Rule 'value': score every posture for every state with the trained posture-outcome questions.
    One sequence per (state, posture) with three questions (all six postures in one sequence would not
    fit the 512-token window). Returns, per state, one dict per posture (POSTURES order):
    ok = P(no territory lost), gain = P(territory gained), grow = expected growth level 0..3, U = score."""
    import torch
    w_ok, w_gain, w_grow = weights
    top = len(GROWTH_LEVELS) - 1
    qsets = []
    for p in POSTURES:
        t_grow, t_ok, t_gain = action_texts(p, pv, horizon)
        qsets.append([model._question(0, {"type": "choice", "instructions": t_grow, "options": list(GROWTH_LEVELS)}),
                      model._question(1, {"type": "noul", "instructions": t_ok}),
                      model._question(2, {"type": "noul", "instructions": t_gain})])
    items = [(s, qs) for s in states for qs in qsets]
    parts = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            b = model.collator(items[i:i + bs], model.device)
            lg = model.model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"]).float()
            pr = (lg / model.model.temperature).softmax(-1)
            for row in pr:
                g = row[0, : len(GROWTH_LEVELS)].tolist()
                level = sum(k * x for k, x in enumerate(g))
                ok, gain = float(row[1, 1]), float(row[2, 1])  # noul options are (no, yes)
                parts.append({"ok": round(ok, 4), "gain": round(gain, 4), "grow": round(level, 3),
                              "U": round(w_ok * ok + w_gain * gain + w_grow * level / top, 5)})
    n = len(POSTURES)
    return [parts[i * n:(i + 1) * n] for i in range(len(states))]


def step(inst: Instance, model, cfg: dict):
    every = int(inst.settings.get("every", cfg["every"]))
    eps = float(inst.settings.get("eps", cfg["eps"]))
    sync = bool(inst.settings.get("sync", False))
    pv = int(inst.settings.get("pv", 1))
    rule = inst.settings.get("rule", cfg["rule"])
    horizon = int(inst.settings.get("horizon", cfg["horizon"]))
    min_mil = cfg["min_mil"]
    j = inst.new_month()
    if j is None:
        return 0
    lg = parse_logs(j.get("logs", []))
    recs = {t: country_record(t, c, lg.get(t, {}), j["date"]) for t, c in j["countries"].items() if c.get("exists", True) is not False}
    nb = neighbors(recs, inst.st_center)
    for t, r in recs.items():
        r["neighbors"] = nb.get(t, [])
    names = {t: r["name"] for t, r in recs.items()}
    elig = [t for t, r in recs.items() if r["n_states"] > 0 and r["mil"] + r["civ"] >= min_mil and inst.controlled(t)]
    # each country is re-decided every `every` months, either all together (sync: one console send per
    # period, a posture holds exactly `every` months) or staggered by tag. Timing uses the dump index, so
    # a daemon restart does not shift it.
    n = inst.month_n
    elig = [t for t in elig if (n % every == 0 if sync else (sum(map(ord, t)) + n) % every == 0)]
    if not elig:
        return 0
    states = [state_text(recs[t], names, recs) for t in elig]
    t0 = time.time()
    if rule == "value":
        scored = decide_value(model, states, pv, horizon, cfg["weights"])
        prefs = [[x["U"] for x in sc] for sc in scored]
    else:
        prefs = [[d[p] for p in POSTURES] for d in decide(model, states)]
        scored = [None] * len(states)
    cmds, rows = [], []
    for t, st, pref, sc in zip(elig, states, prefs, scored):
        # probs: Jev's preference over postures (the answer distribution, or the scores normalised to sum 1)
        tot = sum(pref) or 1.0
        probs = [x / tot for x in pref] if rule == "value" else pref
        greedy = max(range(len(POSTURES)), key=probs.__getitem__)
        rr = random.Random(f"{inst.run}|{n}|{t}|posture")  # per game, month and country
        explore = rr.random() < eps
        k = rr.randrange(len(POSTURES)) if explore else greedy
        # probability that this policy picks k (explore draws can land on the greedy posture too)
        prop = eps / len(POSTURES) + (1 - eps) * (k == greedy)
        explore = explore and k != greedy
        cmds.append(f"e {t} jev_set_posture_{k + 1}")
        rows.append({"date": j["date"], "tag": t, "posture": POSTURES[k], "k": k + 1, "explore": explore, "pv": pv, "eps": eps,
                     "propensity": round(prop, 4), "probs": [round(x, 4) for x in probs], "rule": rule,
                     "model": cfg["model_id"], **({"scores": sc, "horizon": horizon} if sc else {}), "state": st})
    ok = run_console(inst, cmds) if pid_of(inst.name) else False
    # the game keeps running while Jev thinks (value rule: ~1-2 game months for a full sync round on CPU),
    # so record when the postures actually took effect; outcomes can be measured from that date
    when = last_date(inst.userdir) if ok else ""
    applied_date = ".".join(str(int(x)) for x in when.split(".")) if when else None
    for r in rows:
        r["applied"] = ok  # False: the game never confirmed it; the posture from before stays in force
        r["applied_date"] = applied_date
        inst.actions.write(json.dumps(r, ensure_ascii=False) + "\n")
    inst.actions.flush()
    print(f"[{time.strftime('%H:%M:%S')}] {inst.name} {j['date']}: {len(cmds)} countries ({rule}), jev+apply {time.time() - t0:.1f}s{'' if ok else ' NOT CONFIRMED'}", flush=True)
    return len(cmds)


def main(argv):
    """Serve every instance the collector registers (runs/inst/<name>.jev), plus any --inst given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", action="append", default=[], help="name:userdir:run_id (optional, static)")
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--min-mil", type=int, default=20, help="skip small countries (mil+civ factories)")
    ap.add_argument("--every", type=int, default=2, help="re-decide each country every N months")
    ap.add_argument("--model", default=ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rule", choices=["value", "choice"], default="value", help="how Jev picks a posture (see module doc)")
    ap.add_argument("--horizon", type=int, default=6, help="months the posture-outcome questions look ahead (rule value)")
    ap.add_argument("--w-ok", type=float, default=1.0, help="weight of P(no territory lost)")
    ap.add_argument("--w-gain", type=float, default=1.0, help="weight of P(territory gained)")
    ap.add_argument("--w-grow", type=float, default=0.5, help="weight of expected growth level (0..1)")
    a = ap.parse_args(argv)
    model_path = os.path.abspath(a.model)
    model_id = "base" if model_path == ROOT else os.path.relpath(model_path, ROOT).replace(os.sep, "/")
    cfg = {"eps": a.eps, "every": a.every, "min_mil": a.min_mil, "rule": a.rule, "horizon": a.horizon,
           "weights": (a.w_ok, a.w_gain, a.w_grow), "model_id": model_id}
    import ctypes
    import torch
    from typed_decisions.open_jev import OpenJev
    if os.name == "nt":  # below-normal priority: inference bursts must not make the desktop lag
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(a.threads or max(1, (os.cpu_count() or 4) // 2))  # leave cores for the games
    model = OpenJev.from_pretrained(a.model, device="cpu")
    rng = random.Random(a.seed)
    insts: dict[str, Instance] = {}
    for spec in a.inst:
        i = Instance(spec, random.Random(rng.random()), a.frac)
        insts[i.run] = i
    reg_dir = os.path.join(ROOT, "runs", "inst")
    print(f"jevd up: rule={a.rule} model={model_id} horizon={a.horizon} weights={cfg['weights']} postures={POSTURES}", flush=True)
    while True:
        live = set()
        for p in glob.glob(os.path.join(reg_dir, "*.jev")):
            try:
                r = json.load(open(p))
            except (OSError, json.JSONDecodeError):
                continue
            live.add(r["run"])
            if r["run"] not in insts:
                st = {k: r[k] for k in ("every", "sync", "eps", "pv", "rule", "horizon") if k in r}
                insts[r["run"]] = Instance(f"{r['inst']}:{r['userdir']}:{r['run']}", random.Random(rng.random()), a.frac, st)
                print(f"attached {r['run']} on {r['inst']} {st}", flush=True)
        for run in [k for k in insts if k not in live and not a.inst]:
            insts.pop(run).actions.close()
            print(f"detached {run}", flush=True)
        n = 0
        for i in list(insts.values()):
            try:
                n += step(i, model, cfg)
            except Exception as e:  # noqa: BLE001 - one bad month must not kill the daemon
                print(f"{i.run}: step failed: {e!r}", flush=True)
        if n == 0:
            time.sleep(3)


if __name__ == "__main__":
    main(sys.argv[1:])
