"""Fine-tune a Jev bundle on HOI4 examples and keep the checkpoint that scores best on validation.

    python -m trainer.train --out temp/train/<name> [--device xpu] [--bf16] [--freeze 0] [--bs 16] [--lr 2e-5]
        [--steps 3000] [--eval-every 500] [--val-states 2000] [--max-states N] [--upsample 3]
        [--train temp/dataset/train.jsonl] [--val temp/dataset/val.jsonl] [--init models/open-jev-deberta-v3-large]

Embeddings and the lowest --freeze of the 24 encoder layers stay frozen. Loss: cross-entropy + Brier. States that
carry a rare positive label are repeated --upsample times. Every --eval-every steps the model is scored on a fixed
validation subset; when the macro Brier skill (trainer.evaluate) improves, the bundle goes to <out>/best with its
temperature refitted on that subset. --bf16 runs forward and backward under bfloat16 autocast; weights and optimizer
state stay fp32. Writes <out>/train.log and <out>/report.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from mods.jevai.runtime.pack import Packer
from mods.jevai.runtime.text import read_jsonl

from . import model as jm
from .evaluate import MODEL, ROOT, skill, table

RARE = {"capit", "gain", "lose", "newwar", "took", "whowar"}


def upsample(ex: list, factor: int, rng) -> list:
    """Repeat states that carry a rare positive label (a yes on a rare question, or any whowar)."""
    out = list(ex)
    for e in ex:
        if any(q.qid in RARE and (q.kind == "choice" or q.gold == 1) for q in e.questions):
            out += [e] * (factor - 1)
    rng.shuffle(out)
    return out


def fit_temperature(recs: list[dict]) -> float:
    """The temperature (0.6..2.05) with the lowest log-loss on these predictions."""
    best, bt = math.inf, 1.0
    for t in [0.6 + 0.05 * i for i in range(30)]:
        nll = 0.0
        for r in recs:
            lg = [x / t for x in r["logits"]]
            m = max(lg)
            nll += m + math.log(sum(math.exp(x - m) for x in lg)) - lg[r["gold"]]
        if nll < best:
            best, bt = nll, t
    return bt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune a Jev bundle on HOI4 examples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default=os.path.join(ROOT, "temp", "dataset", "train.jsonl"))
    ap.add_argument("--val", default=os.path.join(ROOT, "temp", "dataset", "val.jsonl"))
    ap.add_argument("--init", default=MODEL, help="bundle to start from")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--freeze", type=int, default=0, help="frozen encoder layers (of 24), from the bottom")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-states", type=int, default=2000)
    ap.add_argument("--max-states", type=int, default=0, help="train on a random subset of states (0 = all)")
    ap.add_argument("--upsample", type=int, default=3)
    a = ap.parse_args(argv)

    torch.manual_seed(0)
    rng = random.Random(0)
    if a.device == "cpu":
        torch.set_num_threads(os.cpu_count())
    net, cfg = jm.load(a.init, a.device)
    packer = Packer(a.init, cfg.get("max_state_tokens", 256), cfg.get("max_len", 512))
    bb = net.backbone
    for p in list(bb.embeddings.parameters()) + [p for layer in bb.encoder.layer[: a.freeze] for p in layer.parameters()]:
        p.requires_grad = False
    params = [p for p in net.parameters() if p.requires_grad]

    train = read_jsonl(a.train)
    if a.max_states:
        rng.shuffle(train)
        train = train[: a.max_states]
    val = read_jsonl(a.val)
    rng.shuffle(val)
    val = [e for e in val[: a.val_states] if packer.fits(e.state, e.questions)]
    n0 = len(train)
    train = [e for e in train if packer.fits(e.state, e.questions)]
    n_fit = len(train)
    train = upsample(train, a.upsample, rng)

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train.log"), "a", encoding="utf-8")

    def say(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"args {vars(a)}")
    say(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M params; train {n_fit} states "
        f"({n0 - n_fit} too long dropped) -> {len(train)} after rare x{a.upsample}; val {len(val)}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / a.steps))
    best, best_step, i, run_loss, t0 = -math.inf, 0, 0, None, time.time()
    net.train()
    for step in range(a.steps):
        if i + a.bs > len(train):
            rng.shuffle(train)
            i = 0
        chunk = train[i:i + a.bs]
        i += a.bs
        b = jm.to_torch(packer([(e.state, e.questions) for e in chunk]), a.device)
        with jm.autocast(a.device, a.bf16):
            logits = net(b["input_ids"], b["attention_mask"], b["pool"], b["opt_mask"])
        loss, info = jm.decision_loss(logits, b["gold"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if step % 20 == 0:
            say(f"step {step} loss {run_loss:.4f} ce {info['ce']:.3f} {(time.time() - t0) / (step + 1):.2f}s/step")
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            recs = jm.predict(net, packer, val, a.bs, a.device, a.bf16, temperature=1.0)
            s = skill(recs)
            say(f"eval step {step + 1}: macro skill {s:.4f} (best so far {best:.4f})\n{table(recs)}")
            if s > best:
                best, best_step = s, step + 1
                net.temperature = fit_temperature(recs)
                prov = {"train": a.train, "states": n_fit, "step": best_step, "bs": a.bs, "lr": a.lr, "freeze": a.freeze,
                        "upsample": a.upsample, "bf16": a.bf16, "val_macro_skill": round(s, 4)}
                jm.save(net, dict(cfg, train_hoi4=prov), os.path.join(a.out, "best"), a.init)
                say(f"saved {os.path.join(a.out, 'best')} (T={net.temperature:.2f})")
            net.train()
    report = {"best_macro_skill": best, "best_step": best_step, "minutes": round((time.time() - t0) / 60, 1), "args": vars(a)}
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say(f"done in {report['minutes']} min; best macro skill {best:.4f} at step {best_step}")


if __name__ == "__main__":
    main()
