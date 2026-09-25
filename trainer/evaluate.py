"""Score a bundle on HOI4 examples, per question id: accuracy next to the majority-class rate, Brier score, and
Brier skill (1 - Brier / Brier of always predicting the label frequencies: 1 = perfect, 0 = no better, < 0 worse).
Training keeps the checkpoint with the best macro-average skill over question ids.

    python -m trainer.evaluate temp/dataset/test.jsonl [--model DIR] [--n 2000] [--device xpu] [--bf16]
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "open-jev-deberta-v3-large")


def _brier(probs, gold) -> float:
    return sum((p - (i == gold)) ** 2 for i, p in enumerate(probs))


def score(recs: list[dict]) -> list[tuple]:
    """(qid, n, accuracy, majority rate, Brier, Brier skill) per question id."""
    by = collections.defaultdict(list)
    for r in recs:
        by[r["qid"]].append(r)
    rows = []
    for qid, rs in sorted(by.items()):
        golds = collections.Counter(r["gold"] for r in rs)
        maj = golds.most_common(1)[0][0]
        acc = sum(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in rs) / len(rs)
        brier = sum(_brier(r["probs"], r["gold"]) for r in rs) / len(rs)
        # reference: this question id's label frequencies, cut to each question's options
        freq = [[golds.get(i, 0) for i in range(len(r["probs"]))] for r in rs]
        base = sum(_brier([x / sum(f) for x in f], r["gold"]) for f, r in zip(freq, rs)) / len(rs)
        rows.append((qid, len(rs), acc, golds[maj] / len(rs), brier, 1 - brier / base if base > 0 else 0.0))
    return rows


def skill(recs: list[dict]) -> float:
    rows = score(recs)
    return sum(r[5] for r in rows) / len(rows) if rows else float("-inf")


def table(recs: list[dict]) -> str:
    lines = [f"  {'qid':9s} {'n':>6s} {'acc':>6s} {'major':>6s} {'brier':>6s} {'skill':>6s}"]
    lines += [f"  {q:9s} {n:6d} {a:6.3f} {m:6.3f} {b:6.3f} {s:6.3f}" for q, n, a, m, b, s in score(recs)]
    return "\n".join(lines + [f"  macro skill {skill(recs):.4f}"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a bundle on HOI4 examples")
    ap.add_argument("data")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=2000, help="states to score (random sample, fixed seed)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args(argv)
    import torch
    from mods.jevai.runtime.pack import Packer
    from mods.jevai.runtime.text import read_jsonl
    from .model import load, predict
    if a.device == "cpu":
        torch.set_num_threads(os.cpu_count())
    net, cfg = load(a.model, a.device)
    packer = Packer(a.model, cfg.get("max_state_tokens", 256), cfg.get("max_len", 512))
    ex = read_jsonl(a.data)
    random.Random(0).shuffle(ex)
    ex = [e for e in ex[: a.n] if packer.fits(e.state, e.questions)]
    t = time.time()
    recs = predict(net, packer, ex, a.bs, a.device, a.bf16)
    print(f"{a.model}: {len(ex)} states, {len(recs)} questions in {time.time() - t:.0f}s\n{table(recs)}")


if __name__ == "__main__":
    main()
