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


def inputs(packer, state: str, posture: str, seq: int) -> dict[str, np.ndarray]:
    b = packer([(state, posture_questions(posture, VERSION, 6))], length=seq, questions=QM, options=OM)
    return {k: b[k] for k in ("input_ids", "attention_mask", "pool", "opt_mask")}


def main(argv=None):
    import openvino as ov
    ap = argparse.ArgumentParser(description="Export a Jev bundle to OpenVINO IR")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--check", type=int, default=24, help="posture states from the test split to compare against torch")
    a = ap.parse_args(argv)
    ref_net, cfg = jm.load(a.model, "cpu")
    ref_net = Scorer(ref_net.float().eval())  # stock attention: the torch reference, and the CPU graph
    npu_net, _ = jm.load(a.model, "cpu")
    patch(npu_net.float().eval().backbone, a.seq)  # gather-free, select-free attention: same numbers
    npu_net = Scorer(npu_net)
    packer = Packer(a.model, cfg.get("max_state_tokens", 256), a.seq)
    tests = [e for e in read_jsonl(os.path.join(ROOT, "temp", "dataset", "test.jsonl")) if any(q.qid == "act_grow" for q in e.questions)]
    tests = tests[: a.check]
    example = inputs(packer, tests[0].state, POSTURES[0], a.seq)
    os.makedirs(a.out, exist_ok=True)
    for name, net in (("jev_cpu", ref_net), ("jev_npu", npu_net)):
        with torch.no_grad():
            ovm = ov.convert_model(net, example_input={k: torch.from_numpy(v) for k, v in example.items()},
                                   input=[(k, list(v.shape)) for k, v in example.items()])
        for port_name, port in zip(example, ovm.inputs):
            port.get_tensor().set_names({port_name})
        ovm.outputs[0].get_tensor().set_names({"probs"})
        ov.save_model(ovm, os.path.join(a.out, name + ".xml"), compress_to_fp16=True)
        del ovm
    shutil.copy2(os.path.join(a.model, "tokenizer.json"), a.out)
    with open(os.path.join(a.out, "jev.json"), "w", encoding="utf-8") as f:
        json.dump({"seq": a.seq, "questions": QM, "options": OM, "max_state_tokens": cfg.get("max_state_tokens", 256),
                   "postures_version": VERSION, "postures": POSTURES, "growth_levels": GROWTH_LEVELS,
                   "graphs": {"NPU": "jev_npu.xml", "CPU": "jev_cpu.xml", "GPU": "jev_npu.xml"},
                   "source": os.path.relpath(os.path.abspath(a.model), ROOT).replace(os.sep, "/"),
                   "train": cfg.get("train_hoi4")}, f, indent=1)
    print("saved", a.out, "|", sorted(os.listdir(a.out)))

    core = ov.Core()
    ref, got = {}, {}
    for dev, graph in (("CPU", "jev_cpu.xml"), ("NPU", "jev_npu.xml")):
        if dev not in core.available_devices:
            continue
        comp = core.compile_model(os.path.join(a.out, graph), dev)
        worst = 0.0
        for e in tests:
            for p in POSTURES[:2]:
                x = inputs(packer, e.state, p, a.seq)
                if (e.state, p) not in ref:
                    with torch.no_grad():
                        ref[(e.state, p)] = ref_net(*[torch.from_numpy(x[k]) for k in x]).numpy()
                probs = comp(x)["probs"]
                mask = x["opt_mask"]
                worst = max(worst, float(np.abs(np.where(mask, probs, 0) - np.where(mask, ref[(e.state, p)], 0)).max()))
        got[dev] = worst
        print(f"{dev} ({graph}): max |p_openvino - p_torch| over {len(tests)} states x 2 postures = {worst:.4f}", flush=True)
    return got


if __name__ == "__main__":
    main()
