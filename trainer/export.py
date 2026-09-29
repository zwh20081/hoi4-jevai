"""Export a Jev bundle to OpenVINO IR for the mod's runner, and check it against torch.

    python -m trainer.export --model temp/train/<name>/best --out temp/ov/<name> [--seq 512] [--check 24]

The runner only asks the three posture questions (runtime.text.posture_questions): 3 questions with at most 4
options. Each IR has fixed shapes: [1, seq] tokens, pool [1, 3*4+3, seq], opt_mask [1, 3, 4]. Two graphs of the same
model, FP16 weights: jev_npu.xml with gather-free, select-free attention (trainer.npu_attention; 7x faster on the Intel
NPU than the stock graph, but 1.7x slower on CPU) and jev_cpu.xml with the stock attention (CPU fallback). The runner
picks by device. The tokenizer and config go next to them.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil

import numpy as np
import torch

from mods.jevai.runtime.pack import Packer
from mods.jevai.runtime.postures import NAMES as POSTURES, VERSION
from mods.jevai.runtime.text import GROWTH_LEVELS, posture_questions, read_jsonl

from . import model as jm
from .evaluate import MODEL, ROOT
from .npu_attention import patch

QM, OM = 3, len(GROWTH_LEVELS)  # 3 posture questions, the growth question has the most options (4)


class Scorer(torch.nn.Module):
    """The IR's graph: probabilities per question, temperature applied (softmax over each question's options)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, input_ids, attention_mask, pool, opt_mask):
        logits = self.net(input_ids, attention_mask, pool, opt_mask)
        return (logits / self.net.temperature).softmax(-1)


def inputs(packer, state: str, posture: str, seq: int, horizon: int = 6) -> dict[str, np.ndarray]:
    b = packer([(state, posture_questions(posture, VERSION, horizon))], length=seq, questions=QM, options=OM)
    return {k: b[k] for k in ("input_ids", "attention_mask", "pool", "opt_mask")}


def main(argv=None):
    import openvino as ov
    ap = argparse.ArgumentParser(description="Export a Jev bundle to OpenVINO IR")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--check", type=int, default=24, help="posture states from the test split to compare against torch")
    ap.add_argument("--data", default=os.path.join(ROOT, "temp", "dataset", "test.jsonl"))
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--max-error", type=float, default=0.02)
    ap.add_argument("--min-agreement", type=float, default=0.98)
    a = ap.parse_args(argv)
    if min(a.horizon, a.check, a.cpu_threads) < 1:
        ap.error("horizon, check and cpu-threads must be positive")
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing bundle: {out}")
    print("loading source model", a.model, flush=True)
    ref_net, cfg = jm.load(a.model, "cpu")
    trained_horizon = cfg.get("posttrain_v2", {}).get("horizon_months")
    if trained_horizon is not None and trained_horizon != a.horizon:
        raise ValueError(f"Requested horizon {a.horizon} differs from trained horizon {trained_horizon}")
    ref_net = Scorer(ref_net.float().eval())
    packer = Packer(a.model, cfg.get("max_state_tokens", 256), a.seq)
    tests = []
    seen = set()
    for e in read_jsonl(a.data):
        if any(q.qid == "act_grow" for q in e.questions) and e.state not in seen:
            tests.append(e)
            seen.add(e.state)
            if len(tests) == a.check:
                break
    if not tests:
        raise ValueError("No posture samples found in validation data")
    example = inputs(packer, tests[0].state, POSTURES[0], a.seq, a.horizon)
    # Keep the small reference outputs, then export both graphs from a single model.
    references = []
    print(f"torch reference: {len(tests)} states x {len(POSTURES)} postures", flush=True)
    with torch.inference_mode():
        for e in tests:
            for posture in POSTURES:
                x = inputs(packer, e.state, posture, a.seq, a.horizon)
                references.append(ref_net(*[torch.from_numpy(x[k]) for k in x]).numpy().copy())
    out.mkdir(parents=True, exist_ok=True)
    for name in ("jev_cpu", "jev_npu"):
        if name == "jev_npu":
            patch(ref_net.net.backbone, a.seq)
        print("exporting", name, flush=True)
        with torch.no_grad():
            ovm = ov.convert_model(ref_net, example_input={k: torch.from_numpy(v) for k, v in example.items()},
                                   input=[(k, list(v.shape)) for k, v in example.items()])
        for port_name, port in zip(example, ovm.inputs):
            port.get_tensor().set_names({port_name})
        ovm.outputs[0].get_tensor().set_names({"probs"})
        ov.save_model(ovm, str(out / (name + ".xml")), compress_to_fp16=True)
        del ovm
        gc.collect()
    del ref_net
    gc.collect()
    shutil.copy2(os.path.join(a.model, "tokenizer.json"), out)
    bundle = {"seq": a.seq, "questions": QM, "options": OM,
              "max_state_tokens": cfg.get("max_state_tokens", 256), "horizon": a.horizon,
              "model_version": cfg.get("posttrain_v2", {}).get("version", "v1"),
              "postures_version": VERSION, "postures": POSTURES, "growth_levels": GROWTH_LEVELS,
              "graphs": {"NPU": "jev_npu.xml", "CPU": "jev_cpu.xml", "GPU": "jev_npu.xml"},
              "source": os.path.relpath(os.path.abspath(a.model), ROOT).replace(os.sep, "/"),
              "train": cfg.get("posttrain_v2") or cfg.get("train_hoi4") or cfg.get("train"),
              "base_train": cfg.get("train"),
              "posttrain_v2": cfg.get("posttrain_v2"), "parent_train_hoi4": cfg.get("parent_train_hoi4")}
    (out / "jev.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    with open(a.data, "rb") as f:
        data_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    source_hashes = {}
    for name in ("model.safetensors", "head.safetensors", "open_jev_config.json"):
        with (Path(a.model) / name).open("rb") as f:
            source_hashes[name] = hashlib.file_digest(f, "sha256").hexdigest()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "openvino_version": ov.__version__,
              "torch_version": torch.__version__, "horizon": a.horizon, "states": len(tests),
              "postures_per_state": len(POSTURES), "max_abs_probability_error_threshold": a.max_error,
              "argmax_agreement_threshold": a.min_agreement, "devices": {},
              "data_sha256": data_sha256, "data": os.path.relpath(a.data, ROOT).replace(os.sep, "/"),
              "source_sha256": source_hashes}
    core = ov.Core()
    for dev, graph in (("CPU", "jev_cpu.xml"), ("NPU", "jev_npu.xml")):
        if dev not in core.available_devices:
            report["devices"][dev] = {"status": "unavailable", "passed": False}
            continue
        try:
            print("validating", dev, flush=True)
            props = {"INFERENCE_NUM_THREADS": a.cpu_threads} if dev == "CPU" else {}
            comp = core.compile_model(str(out / graph), dev, props)
            worst, agreed, total = 0.0, 0, 0
            for i, e in enumerate(tests):
                for j, posture in enumerate(POSTURES):
                    x = inputs(packer, e.state, posture, a.seq, a.horizon)
                    probs = comp(x)["probs"]
                    ref = references[i * len(POSTURES) + j]
                    if not np.isfinite(probs).all():
                        raise ValueError("Non-finite OpenVINO probabilities")
                    mask = x["opt_mask"].astype(bool)
                    worst = max(worst, float(np.abs(probs[mask] - ref[mask]).max()))
                    agreed += int((np.where(mask, probs, -np.inf).argmax(-1) ==
                                   np.where(mask, ref, -np.inf).argmax(-1)).sum())
                    total += QM
            agreement = agreed / total
            report["devices"][dev] = {"status": "checked", "graph": graph, "device_name": core.get_property(dev, "FULL_DEVICE_NAME"),
                "max_abs_probability_error": worst, "argmax_agreement": agreement,
                "questions": total, "passed": worst <= a.max_error and agreement >= a.min_agreement}
            del comp
        except Exception as exc:
            report["devices"][dev] = {"status": "error", "passed": False, "error": str(exc)}
        (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(dev, report["devices"][dev], flush=True)
    (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    hashes = {}
    for path in sorted(out.iterdir()):
        if path.is_file():
            with path.open("rb") as f:
                hashes[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    (out / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    if not all(d["passed"] for d in report["devices"].values()):
        raise RuntimeError(f"Validation incomplete or failed; see {out / 'validation.json'}")
    return report


if __name__ == "__main__":
    main()
