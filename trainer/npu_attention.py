"""NPU-friendly DeBERTa-v2 attention for export (numerically the same model).

Profiling the stock graph on the Intel NPU showed three ops taking 75% of the time: GatherElements (the relative-
position lookups c2p/p2c, 20%), and BitwiseNot + Select (masked_fill with ~mask on a [heads, L, L] tensor, 41%). The
sequence length is fixed at export, so the relative positions are constants: each gather becomes a batched matmul
with a constant one-hot selector, and the mask becomes an additive bias. The scale sqrt(head_dim * 3) is a Python
constant: transformers computes it with a TorchScript function, and its traced form made the NPU wrong (cosine 0.63
against CPU on layer 0) while every other piece of the block matched.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def patch(backbone, seq: int):
    """Swap every layer's self-attention forward for the gather-free, select-free one, for inputs of length `seq`.
    The selectors depend only on the sequence length and the (shared) bucket settings, so all layers share one pair."""
    from transformers.models.deberta_v2.modeling_deberta_v2 import build_relative_position

    sel = {}
    for layer in backbone.encoder.layer:
        att = layer.attention.self
        assert att.share_att_key and att.relative_attention and set(att.pos_att_type) == {"c2p", "p2c"}
        span = att.pos_ebd_size
        key = (span, att.position_buckets, att.max_relative_positions)
        if key not in sel:
            probe = torch.zeros(1, seq, 1)
            rel = build_relative_position(probe, probe, bucket_size=att.position_buckets,
                                          max_position=att.max_relative_positions).reshape(seq, seq).long()
            c2p = torch.clamp(rel + span, 0, 2 * span - 1)  # index for [q, k]
            p2c = torch.clamp(-rel + span, 0, 2 * span - 1)  # index for [k, q] (query and key lengths are equal)
            sel[key] = (F.one_hot(c2p, 2 * span).half(), F.one_hot(p2c, 2 * span).half())  # 0/1 are exact in fp16
        att._c2p_sel, att._p2c_sel = sel[key]
        att._scale = math.sqrt(att.attention_head_size * 3)  # 1 + c2p + p2c; all three scores share it
        att.forward = _forward.__get__(att)


def _pick(scores, sel):
    """scores [BH, L, 2k], sel [L, L, 2k] one-hot -> out [BH, L, L] with out[b, i, j] = scores[b, i, idx[i, j]]."""
    return torch.einsum("bij,ikj->bik", scores, sel.to(scores.dtype))


def _forward(self, hidden_states, attention_mask, output_attentions=False, query_states=None, relative_pos=None,
             rel_embeddings=None):
    if query_states is None:
        query_states = hidden_states
    query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
    key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
    value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
    attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / self._scale)
    rel_embeddings = self.pos_dropout(rel_embeddings)
    span = self.pos_ebd_size
    rel_embeddings = rel_embeddings[0: span * 2, :].unsqueeze(0)
    reps = query_layer.size(0) // self.num_attention_heads
    pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads).repeat(reps, 1, 1)
    pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(reps, 1, 1)
    c2p = _pick(torch.bmm(query_layer, pos_key_layer.transpose(-1, -2)), self._c2p_sel)
    p2c = _pick(torch.bmm(key_layer, pos_query_layer.transpose(-1, -2)), self._p2c_sel).transpose(-1, -2)
    attention_scores = attention_scores + (c2p + p2c) / self._scale
    L = attention_scores.size(-1)
    attention_scores = attention_scores.view(-1, self.num_attention_heads, L, L)
    # attention_mask is [B, 1, L, L] of 0/1: an additive bias instead of masked_fill(~mask, min). -1e4 underflows to
    # exactly 0 after softmax in fp16 and fp32, and stays finite in fp16 (the NPU computes in fp16; finfo.max overflows)
    bias = (attention_mask.to(attention_scores.dtype) - 1.0) * 1e4
    attention_probs = torch.softmax(attention_scores + bias, dim=-1)
    attention_probs = self.dropout(attention_probs)
    context_layer = torch.bmm(attention_probs.view(-1, L, L), value_layer)
    context_layer = (context_layer.view(-1, self.num_attention_heads, L, context_layer.size(-1))
                     .permute(0, 2, 1, 3).contiguous())
    context_layer = context_layer.view(*context_layer.size()[:-2], -1)
    return (context_layer, attention_probs if output_attentions else None)
