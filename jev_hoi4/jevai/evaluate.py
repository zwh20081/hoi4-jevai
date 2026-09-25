"""Score an open-jev bundle on HOI4 examples: accuracy per question id vs the majority-class baseline.

    python -m jevai.evaluate runs/data/run0.ex.jsonl [--model .] [--n 200]
"""
from __future__ import annotations

import collections
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from typed_decisions.encoder import predict  # noqa: E402
from typed_decisions.open_jev import OpenJev  # noqa: E402
from typed_decisions.schema import read_jsonl  # noqa: E402


def score(recs, train_recs=None):
    """recs: predict() output. Majority baseline is taken from train_recs (or recs itself if none)."""
    base_src = train_recs or recs
    maj = {}
    for qid, rs in _group(base_src).items():
        maj[qid] = collections.Counter(r["gold"] for r in rs).most_common(1)[0][0]
    rows = []
    for qid, rs in sorted(_group(recs).items()):
        acc = sum(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in rs) / len(rs)
        base = sum(r["gold"] == maj.get(qid, 0) for r in rs) / len(rs)
        brier = sum(sum((p - (i == r["gold"])) ** 2 for i, p in enumerate(r["probs"])) for r in rs) / len(rs)
        rows.append((qid, len(rs), acc, base, brier))
    return rows


def _group(recs):
    g = collections.defaultdict(list)
    for r in recs:
        g[r["qid"]].append(r)
    return g


def main(argv):
    path = argv[0]
    model = argv[argv.index("--model") + 1] if "--model" in argv else ROOT
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 200
    ex = read_jsonl(path)
    random.Random(0).shuffle(ex)
    ex = ex[:n]
    torch.set_num_threads(os.cpu_count())
    m = OpenJev.from_pretrained(model, device="cpu")
    t = time.time()
    recs = predict(m.model, m.collator, ex, batch_size=8, device="cpu")
    print(f"{len(ex)} states, {len(recs)} questions in {time.time() - t:.0f}s")
    print(f"{'qid':10s} {'n':>5s} {'acc':>6s} {'major':>6s} {'brier':>6s}")
    for qid, k, acc, base, brier in score(recs):
        print(f"{qid:10s} {k:5d} {acc:6.3f} {base:6.3f} {brier:6.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
