"""The Jev model in torch: a DeBERTa-v3 encoder reads the state and every question in one pass, and a small head
scores each option of each question from [mean(question text); mean(option text); product]. A softmax within a
question's options is its answer. Loss: cross-entropy + Brier.

A bundle is a folder: the backbone in Hugging Face format, the tokenizer files, head.safetensors and
open_jev_config.json (pooling, temperature, limits, provenance). The stock model and every checkpoint use it."""
from __future__ import annotations

import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "spm.model")


class DecisionEncoder(nn.Module):
    def __init__(self, backbone, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.temperature = 1.0  # fitted on validation after training; not a parameter

    def forward(self, input_ids, attention_mask, pool, opt_mask):
        """pool (B, Qm*Om+Qm, L) and opt_mask (B, Qm, Om) come from runtime.pack. Returns logits (B, Qm, Om),
        -inf where a question has no such option."""
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, Qm, Om = opt_mask.shape
        mean = torch.bmm(pool.to(h.dtype), h)
        opt = mean[:, : Qm * Om].reshape(B, Qm, Om, -1)
        q = mean[:, Qm * Om:].unsqueeze(2).expand_as(opt)
        return self.head(torch.cat([q, opt, q * opt], -1)).squeeze(-1).masked_fill(~opt_mask, float("-inf"))


def load(model_dir: str, device: str = "cpu") -> tuple[DecisionEncoder, dict]:
    from transformers import AutoModel
    with open(os.path.join(model_dir, "open_jev_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("pool", "span") == "span", "only span pooling is supported"
    net = DecisionEncoder(AutoModel.from_pretrained(model_dir, attn_implementation=cfg.get("attn_implementation", "eager")), cfg["hidden"])
    net.head.load_state_dict(load_file(os.path.join(model_dir, "head.safetensors")))
    net.temperature = float(cfg.get("temperature", 1.0))
    return net.to(device).eval(), cfg


def save(net: DecisionEncoder, cfg: dict, out_dir: str, tokenizer_dir: str):
    """Write a bundle; the tokenizer files are copied from `tokenizer_dir` (training never changes them)."""
    os.makedirs(out_dir, exist_ok=True)
    net.backbone.save_pretrained(out_dir, safe_serialization=True)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in net.head.state_dict().items()},
              os.path.join(out_dir, "head.safetensors"))
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tokenizer_dir, name)):
            shutil.copy2(os.path.join(tokenizer_dir, name), out_dir)
    with open(os.path.join(out_dir, "open_jev_config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cfg, temperature=net.temperature), f, indent=1)


def decision_loss(logits, gold, brier_weight: float = 1.0):
    """logits (B, Qm, Om) with -inf on missing options; gold (B, Qm) with -100 on missing questions."""
    valid = gold >= 0
    lg, g = logits[valid].float(), gold[valid]
    ce = F.cross_entropy(lg, g)
    p = lg.softmax(-1)
    brier = (((p - F.one_hot(g, lg.size(-1)).to(p.dtype)) ** 2) * torch.isfinite(lg)).sum(-1).mean()
    return ce + brier_weight * brier, {"ce": ce.item(), "brier": brier.item(), "n": int(valid.sum())}


def to_torch(batch: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def autocast(device, bf16: bool):
    """bfloat16 autocast for forward passes (weights stay fp32); a no-op when bf16 is False."""
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=bf16)


@torch.no_grad()
def predict(net, packer, examples, batch_size: int = 16, device: str = "cpu", bf16: bool = False,
            temperature: float | None = None) -> list[dict]:
    """One record per question: qid, kind, source, gold, probs, logits."""
    net.eval()
    T = net.temperature if temperature is None else temperature
    out = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        b = to_torch(packer([(e.state, e.questions) for e in chunk]), device)
        with autocast(device, bf16):
            logits = net(b["input_ids"], b["attention_mask"], b["pool"], b["opt_mask"])
        logits = logits.float().cpu()
        probs = (logits / T).softmax(-1)
        for bi, e in enumerate(chunk):
            for qi, q in enumerate(e.questions):
                n = len(q.options)
                out.append({"qid": q.qid, "kind": q.kind, "source": e.source, "gold": q.gold,
                            "probs": probs[bi, qi, :n].tolist(), "logits": logits[bi, qi, :n].tolist()})
    return out
