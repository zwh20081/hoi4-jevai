"""Fine-tune a Jev bundle on HOI4 examples, keep the checkpoint that scores best on validation, survive interruptions.

    python -m trainer.train --out temp/train/<name> --bf16 [--epochs 1 | --steps N] [--bs 16] [--accum 2] [--lr 2e-5]
        [--freeze 0] [--grad-ckpt] [--eval-every 1000] [--val-states 1500] [--save-every 500] [--upsample 3]
        [--max-states N] [--device xpu] [--train ...] [--val ...] [--init models/open-jev-deberta-v3-large]
    python -m trainer.train --out temp/train/<name> --resume [--stop-after N]

States with a rare positive label or with posture questions are repeated --upsample times. An optimizer step takes
--accum batches of --bs states. Loss: cross-entropy + Brier. --bf16 is autocast with fp32 weights and optimizer;
--grad-ckpt recomputes activations in the backward pass (less memory, slower). Every --eval-every steps the model is
scored on a fixed validation subset (trainer.evaluate.subset); a better macro Brier skill saves the bundle to
<out>/best with its temperature refitted on that subset. The whole training state (weights, optimizer, schedule, data
order, RNGs) goes to <out>/state.pt every --save-every steps, at --stop-after, on Ctrl+C and at the end; --resume
continues from it with the arguments the run started with.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import time

import torch

from mods.jevai.runtime.pack import Packer
from mods.jevai.runtime.text import read_jsonl

from . import model as jm
from .evaluate import MODEL, ROOT, skill, subset, table

RARE = {"capit", "gain", "lose", "newwar", "took", "whowar"}


def upsampled(e) -> bool:
    """Worth repeating: a rare positive forecast label (a yes on a rare question, or any whowar), or posture questions."""
    return any(q.qid.startswith("act_") or (q.qid in RARE and (q.kind == "choice" or q.gold == 1)) for q in e.questions)


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


def device_rng(device: str):
    """torch.xpu / torch.cuda for their RNG state, None on CPU."""
    mod = getattr(torch, torch.device(device).type, None)
    return mod if device != "cpu" and hasattr(mod, "get_rng_state") else None


def trim_cache(device: str, step: int, every: int):
    """Give the device's cached blocks back every `every` steps. Batch shapes change every step, so the caching
    allocator keeps adding blocks of new sizes; on an iGPU that cache is host RAM (measured: 36 GB reserved for 4 GB
    in use after 40 steps, 4.7 GB with a trim every 10 steps at ~4% of the speed). Trimming on every step, or
    whenever the cache passes a size, stalls the device: 1.7x slower."""
    mod = device_rng(device)
    if every and step % every == 0 and mod and hasattr(mod, "empty_cache"):
        mod.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune a Jev bundle on HOI4 examples (resumable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="continue <out>/state.pt with the run's original arguments")
    ap.add_argument("--stop-after", type=int, default=0, help="save the state and stop after N steps of this invocation")
    ap.add_argument("--train", default=os.path.join(ROOT, "temp", "dataset", "train.jsonl"))
    ap.add_argument("--val", default=os.path.join(ROOT, "temp", "dataset", "val.jsonl"))
    ap.add_argument("--init", default=MODEL, help="bundle to start from")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing: less memory, slower")
    ap.add_argument("--freeze", type=int, default=0, help="frozen encoder layers (of 24), from the bottom")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="batches per optimizer step (effective batch = bs * accum)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (default: --epochs over the training set)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-states", type=int, default=1500, help="validation subset: up to this many posture states + this many others")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-states", type=int, default=0, help="train on a random subset of states (0 = all)")
    ap.add_argument("--upsample", type=int, default=3)
    ap.add_argument("--trim-every", type=int, default=10, help="empty the device memory cache every N steps (0 = never)")
    a = ap.parse_args(argv)
    print(f"loading model and data for {a.out} (a few minutes) ...", flush=True)
    state_path = os.path.join(a.out, "state.pt")
    saved = torch.load(state_path, map_location="cpu", weights_only=False) if a.resume else None
    if saved:
        a = argparse.Namespace(**dict(saved["args"], resume=True, stop_after=a.stop_after, trim_every=a.trim_every))
    run_args = {k: v for k, v in vars(a).items() if k not in ("resume", "stop_after", "trim_every")}
    if os.name == "nt":  # a long run must not stall on system sleep (the display may still turn off)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

    torch.manual_seed(0)
    rng = random.Random(0)
    if a.device == "cpu":
        torch.set_num_threads(os.cpu_count())
    net, cfg = jm.load(a.init, a.device)
    packer = Packer(a.init, cfg.get("max_state_tokens", 256), cfg.get("max_len", 512))
    bb = net.backbone
    for p in list(bb.embeddings.parameters()) + [p for layer in bb.encoder.layer[: a.freeze] for p in layer.parameters()]:
        p.requires_grad = False
    if a.grad_ckpt:
        bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in net.parameters() if p.requires_grad]

    train = read_jsonl(a.train)
    if a.max_states:
        rng.shuffle(train)
        train = train[: a.max_states]
    n0 = len(train)
    train = [e for e in train if packer.fits(e.state, e.questions)]
    n_fit = len(train)
    train = [e for e in train for _ in range(a.upsample if upsampled(e) else 1)]
    val = [e for e in subset(read_jsonl(a.val), a.val_states) if packer.fits(e.state, e.questions)]
    steps = a.steps or math.ceil(a.epochs * len(train) / (a.bs * a.accum))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps))
    if saved:
        assert saved["prog"]["n_train"] == len(train), "the training data changed since this run started"
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["dev_rng"] is not None:
            device_rng(a.device).set_rng_state(saved["dev_rng"])
        prog = saved["prog"]
        del saved
    else:
        order = list(range(len(train)))
        rng.shuffle(order)
        prog = {"step": 0, "pos": 0, "order": order, "n_train": len(train), "best": -math.inf, "best_step": 0,
                "loss": None, "seconds": 0.0}

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train.log"), "a", encoding="utf-8")

    def say(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    def save_state():
        """Atomic: a crash while writing leaves the previous state.pt intact."""
        dev = device_rng(a.device)
        torch.save({"args": run_args, "model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "dev_rng": dev.get_rng_state() if dev else None,
                    "prog": dict(prog, seconds=prog["seconds"] + time.time() - t0)}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    def batch() -> list:
        if prog["pos"] + a.bs > len(prog["order"]):  # epoch boundary: a new order
            rng.shuffle(prog["order"])
            prog["pos"] = 0
        idx = prog["order"][prog["pos"]: prog["pos"] + a.bs]
        prog["pos"] += a.bs
        return [train[k] for k in idx]

    say(f"{'resume at step ' + str(prog['step']) if a.resume else 'start'}: args {run_args}")
    say(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M params; train {n_fit} states ({n0 - n_fit} too long "
        f"dropped) -> {len(train)} with rare and posture states x{a.upsample}; val {len(val)}; {steps} steps of {a.bs}x{a.accum}")
    t0, done = time.time(), 0
    net.train()
    try:
        while prog["step"] < steps:
            for _ in range(a.accum):
                b = jm.to_torch(packer([(e.state, e.questions) for e in batch()]), a.device)
                with jm.autocast(a.device, a.bf16):
                    logits = net(b["input_ids"], b["attention_mask"], b["pool"], b["opt_mask"])
                loss, info = jm.decision_loss(logits, b["gold"])
                (loss / a.accum).backward()
                prog["loss"] = loss.item() if prog["loss"] is None else 0.98 * prog["loss"] + 0.02 * loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            prog["step"] += 1
            done += 1
            step = prog["step"]
            trim_cache(a.device, step, a.trim_every)
            if step % 20 == 0 or done == 1:
                sps = (time.time() - t0) / done
                say(f"step {step}/{steps} ({100 * step / steps:.1f}%) loss {prog['loss']:.4f} ce {info['ce']:.3f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {sps:.2f}s/step ETA {(steps - step) * sps / 3600:.1f}h")
            if step % a.eval_every == 0 or step == steps:
                recs = jm.predict(net, packer, val, a.bs, a.device, a.bf16, temperature=1.0)
                s = skill(recs)
                say(f"eval step {step}: macro skill {s:.4f} (best {prog['best']:.4f} at step {prog['best_step']})\n{table(recs)}")
                if s > prog["best"]:
                    prog["best"], prog["best_step"] = s, step
                    net.temperature = fit_temperature(recs)
                    prov = {"train": a.train, "states": n_fit, "step": step, "steps": steps, "bs": a.bs, "accum": a.accum,
                            "lr": a.lr, "freeze": a.freeze, "upsample": a.upsample, "bf16": a.bf16, "val_macro_skill": round(s, 4)}
                    jm.save(net, dict(cfg, train_hoi4=prov), os.path.join(a.out, "best"), a.init)
                    say(f"saved {os.path.join(a.out, 'best')} (T={net.temperature:.2f})")
                net.train()
            if step % a.save_every == 0 or step == steps or done == a.stop_after:
                save_state()
            if done == a.stop_after and step < steps:
                say(f"stopped at step {step}; continue with --resume")
                return
    except KeyboardInterrupt:  # saved mid-step: the batches of an unfinished step are skipped on resume
        save_state()
        say(f"interrupted at step {prog['step']}; state saved, continue with --resume")
        return
    report = {"best_macro_skill": prog["best"], "best_step": prog["best_step"], "steps": steps,
              "hours": round((prog["seconds"] + time.time() - t0) / 3600, 2), "args": run_args}
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    say(f"done: {steps} steps in {report['hours']} h; best macro skill {prog['best']:.4f} at step {prog['best_step']}")


if __name__ == "__main__":
    main()
