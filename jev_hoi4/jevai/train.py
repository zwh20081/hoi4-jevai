"""Fine-tune open-jev on HOI4 examples on CPU.

    python -m jevai.train --train runs/dataset/train.jsonl --val runs/dataset/val.jsonl \
        --out runs/ckpt/hoi4-v1 [--device cpu|xpu|cuda] [--freeze 18] [--bs 8] [--lr 2e-5] [--steps 1500] [--max-states N]

Freezes the embeddings and the lowest `--freeze` of 24 DeBERTa layers (CPU budget); trains the rest
plus the scoring head with the original loss (CE + Brier). Rare positive labels are upsampled by
repeating states that carry them. Saves an open-jev bundle (loadable by OpenJev.from_pretrained)
with a temperature refit on the validation split.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from typed_decisions.encoder import decision_loss, predict  # noqa: E402
from typed_decisions.open_jev import OpenJev  # noqa: E402
from typed_decisions.schema import read_jsonl  # noqa: E402

from .evaluate import score  # noqa: E402

RARE = {"capit", "gain", "lose", "newwar", "took", "whowar"}


def arg(argv, k, d=None, t=str):
    return t(argv[argv.index(k) + 1]) if k in argv else d


def upsample(ex, factor: int, rng):
    """Repeat states that carry a rare positive label (noul yes on a rare question, or a whowar)."""
    out = list(ex)
    for e in ex:
        if any(q.qid in RARE and (q.kind == "choice" or q.gold == 1) for q in e.questions):
            out += [e] * (factor - 1)
    rng.shuffle(out)
    return out


def fit_temperature(recs):
    best, bt = 1e9, 1.0
    for t in [0.6 + 0.05 * i for i in range(30)]:
        nll = 0.0
        for r in recs:
            lg = [x / t for x in r["logits"]]
            m = max(lg)
            z = m + math.log(sum(math.exp(x - m) for x in lg))
            nll += z - lg[r["gold"]]
        if nll < best:
            best, bt = nll, t
    return bt


def main(argv):
    tr_path, va_path, out = arg(argv, "--train"), arg(argv, "--val"), arg(argv, "--out")
    freeze, bs, lr = arg(argv, "--freeze", 18, int), arg(argv, "--bs", 8, int), arg(argv, "--lr", 2e-5, float)
    steps, max_states = arg(argv, "--steps", 1500, int), arg(argv, "--max-states", 0, int)
    up = arg(argv, "--upsample", 3, int)
    dev = arg(argv, "--device", "cpu")  # cpu | xpu | cuda
    if dev == "cpu":
        torch.set_num_threads(os.cpu_count())
    torch.manual_seed(0)
    rng = random.Random(0)

    m = OpenJev.from_pretrained(ROOT, device=dev)
    net = m.model
    bb = net.backbone
    for p in bb.embeddings.parameters():
        p.requires_grad = False
    for layer in bb.encoder.layer[:freeze]:
        for p in layer.parameters():
            p.requires_grad = False
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable params {n_tr / 1e6:.1f}M (frozen emb + {freeze} layers)")

    train = read_jsonl(tr_path)
    if max_states:
        rng.shuffle(train)
        train = train[:max_states]
    val = read_jsonl(va_path) if va_path and os.path.exists(va_path) else []
    rng.shuffle(val)
    val = val[:400]
    def fits(e):
        try:
            m.collator.encode_one(e.state, e.questions)
            return True
        except ValueError:
            return False
    n0 = len(train)
    train = [e for e in train if fits(e)]
    val = [e for e in val if fits(e)]
    if len(train) < n0:
        print(f"dropped {n0 - len(train)} training states longer than the model window")
    train = upsample(train, up, rng)
    print(f"train {len(train)} states (after upsampling x{up}), val {len(val)}")

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps))
    net.train()
    t0, i, run_loss = time.time(), 0, None
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    log = open(os.path.join(os.path.dirname(out) or ".", os.path.basename(out) + ".log"), "a")
    for step in range(steps):
        if i + bs > len(train):
            rng.shuffle(train); i = 0
        chunk = train[i:i + bs]; i += bs
        b = m.collator([(e.state, e.questions) for e in chunk], m.device)
        logits = net(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"])
        loss, info = decision_loss(logits, b["gold"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if step % 20 == 0:
            msg = f"step {step} loss {run_loss:.4f} ce {info['ce']:.3f} {(time.time() - t0) / (step + 1):.1f}s/step"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if val and (step + 1) % 500 == 0 or step + 1 == steps:
            net.eval()
            recs = predict(net, m.collator, val, batch_size=8, device=m.device, temperature=1.0)
            for qid, k, acc, base, brier in score(recs):
                msg = f"  val {qid:8s} n={k:5d} acc {acc:.3f} major {base:.3f} brier {brier:.3f}"
                print(msg); log.write(msg + "\n")
            log.flush()
            net.train()
    # save bundle
    net.eval()
    T = fit_temperature(predict(net, m.collator, val, batch_size=8, device=m.device, temperature=1.0)) if val else 1.0
    os.makedirs(out, exist_ok=True)
    bb.save_pretrained(out, safe_serialization=True)
    m.tok.save_pretrained(out)
    save_file({k: v.detach().cpu().contiguous() for k, v in net.head.state_dict().items()}, os.path.join(out, "head.safetensors"))
    cfg = dict(m.config)
    cfg["temperature"] = T
    cfg["train_hoi4"] = {"train": tr_path, "states": len(train), "steps": steps, "bs": bs, "lr": lr, "freeze": freeze, "upsample": up}
    json.dump(cfg, open(os.path.join(out, "open_jev_config.json"), "w"), indent=1)
    print(f"saved {out} (T={T:.2f}) in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
