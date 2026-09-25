"""State + questions -> the arrays the model takes. One sequence answers every question on it:

    [CLS] [STATE] state [Q] question_1 [OPT] option [OPT] option ... [Q] question_2 ... [SEP]

`pool` averages the text tokens of option o of question q (slot q*Om + o) and of question q's own text (slot
Qm*Om + q), so the model scores each option from [mean(question); mean(option); product]. It is a matrix
(pool @ hidden) rather than a gather/scatter, so the same graph exports to OpenVINO and to fixed-shape devices."""
from __future__ import annotations

import os

import numpy as np
from tokenizers import Tokenizer

MARKERS = ("[STATE]", "[Q]", "[OPT]")


class Packer:
    def __init__(self, model_dir: str, max_state: int = 256, max_len: int = 512):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.max_state, self.max_len = max_state, max_len
        self.cls, self.sep, self.pad = (self.tok.token_to_id(t) for t in ("[CLS]", "[SEP]", "[PAD]"))
        self.m_state, self.m_q, self.m_opt = (self.tok.token_to_id(t) for t in MARKERS)
        self._cache: dict[str, list[int]] = {}

    def ids(self, text: str) -> list[int]:
        r = self._cache.get(text)
        if r is None:
            r = self.tok.encode(text, add_special_tokens=False).ids
            if len(self._cache) < 200_000:
                self._cache[text] = r
        return r

    def encode(self, state: str, questions) -> tuple[list[int], list[tuple[int, int, int, int]]]:
        """Token ids (state cut to max_state tokens) and text spans (question, option or -1 for the question's
        own text, start, end). Raises ValueError above max_len."""
        ids = [self.cls, self.m_state] + self.ids(state)[: self.max_state]
        spans = []
        for qi, q in enumerate(questions):
            t = self.ids(q.instructions)
            spans.append((qi, -1, len(ids) + 1, len(ids) + 1 + len(t)))
            ids += [self.m_q] + t
            for oi, o in enumerate(q.options):
                t = self.ids(o)
                spans.append((qi, oi, len(ids) + 1, len(ids) + 1 + len(t)))
                ids += [self.m_opt] + t
        ids.append(self.sep)
        if len(ids) > self.max_len:
            raise ValueError(f"sequence {len(ids)} > max_len {self.max_len}: state + {len(questions)} questions")
        return ids, spans

    def fits(self, state: str, questions) -> bool:
        try:
            self.encode(state, questions)
            return True
        except ValueError:
            return False

    def __call__(self, items, length: int | None = None, questions: int | None = None, options: int | None = None) -> dict[str, np.ndarray]:
        """Pack [(state, questions), ...]. `length`, `questions` and `options` fix the padded size of the sequence,
        the question count and the option count (fixed shapes); otherwise each is the batch maximum."""
        enc = [self.encode(s, qs) for s, qs in items]
        L = length or max(len(ids) for ids, _ in enc)
        Qm = questions or max(len(qs) for _, qs in items)
        Om = options or max(len(q.options) for _, qs in items for q in qs)
        B = len(items)
        input_ids = np.full((B, L), self.pad, np.int64)
        attention_mask = np.zeros((B, L), np.int64)
        pool = np.zeros((B, Qm * Om + Qm, L), np.float32)
        opt_mask = np.zeros((B, Qm, Om), bool)
        gold = np.full((B, Qm), -100, np.int64)
        for b, ((ids, spans), (_, qs)) in enumerate(zip(enc, items)):
            if len(ids) > L:
                raise ValueError(f"sequence {len(ids)} > length {L}")
            input_ids[b, : len(ids)] = ids
            attention_mask[b, : len(ids)] = 1
            for qi, oi, st, en in spans:
                if en > st:
                    pool[b, Qm * Om + qi if oi < 0 else qi * Om + oi, st:en] = 1.0 / (en - st)
            for qi, q in enumerate(qs):
                opt_mask[b, qi, : len(q.options)] = True
                gold[b, qi] = q.gold
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pool": pool, "opt_mask": opt_mask, "gold": gold}
