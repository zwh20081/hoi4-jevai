---
license: apache-2.0
language:
- en
base_model: microsoft/deberta-v3-large
pipeline_tag: text-classification
library_name: transformers
tags:
- typed-decisions
- calibrated
- decision-model
- open-jev
- deberta-v3
datasets:
- mteb/banking77
- SetFit/sst5
- google/boolq
---

# open-jev-deberta-v3-large

**An open, Jev-shaped typed-decision model.** One program *state* and any number of typed *questions*
(`choice` over up to 255 options / `score` over 2–10 ordered levels / `noul` yes-no) go in; a calibrated
probability distribution per question comes back from **one forward pass**. Nothing is generated, so the
structured-output error rate is 0 by construction.

This is an independent reproduction of the *shape* of TypeSafe AI's Jev ("System One Model"); it is not
affiliated with TypeSafe, uses none of their data or code, and its numbers are not comparable to theirs
(they report agreement with frontier models on private workflows; this model is measured on public gold
labels). Code, corpus builder, ablations and the full measured record:
[kotoba-lang/typed-decisions](https://github.com/kotoba-lang/typed-decisions) and the ADRs it cites.

## Use

```python
# pip install torch transformers safetensors huggingface_hub sentencepiece protobuf
import sys; sys.path.insert(0, "<path to this repo snapshot>")   # or: pip install git+https://github.com/kotoba-lang/typed-decisions
from typed_decisions.open_jev import OpenJev

m = OpenJev.from_pretrained("com-kotobalabs/open-jev-deberta-v3-large")
m.decide(
    "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    [{"type": "choice", "instructions": "Which product area is the message about?",
      "options": ["fees & charges", "pin & security", "refund & dispute", "top-up", "exchange & fiat", "atm & cash", "transfer", "card", "account & identity", "other"]},
     {"type": "score",  "instructions": "How positive is the sentiment of this message?",
      "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
     {"type": "noul",   "instructions": "The customer is asking for a refund."}])
# [{'choice': 'fees & charges', 'probabilities': {...}, 'confidence': 0.733},
#  {'score': 0.846, 'probabilities': {...}, 'confidence': 0.527},
#  {'noul': 0.892}]
```

`score` is the expected level index (may fall between levels); `noul` is p(yes). Every answer carries the
full distribution. Context: 512 tokens total (state is cut to 256 tokens). CPU fp32 on an M1 Max: 1.8 s for
4 questions; H100 bf16: 28 ms end-to-end for 10 questions on one state (25 ms forward), 518 questions/s at
batch 8.

## How it works

```
[CLS] [STATE] state [Q] instructions [OPT] option_1 [OPT] option_2 … [Q] … [SEP]
```

A DeBERTa-v3-large encoder reads state and all questions at once. For each option the head scores
`[mean(question tokens); mean(option tokens); product]`; a softmax **within each question's option group**
is that question's distribution. The three marker tokens are added to the tokenizer; `head.safetensors`
holds the 3-layer scoring head; `open_jev_config.json` holds the pooling mode, the post-hoc temperature
(fitted on a validation split) and the training provenance. Loss was cross-entropy + Brier; training used
gold-preserving question augmentation (option shuffling, paraphrase templates, distractor dropping,
level relabelling, noul negation with flipped gold) at p = 0.7.

## Training data

Public gold labels only — no synthetic answers, no teacher model:

| source | state | questions |
|---|---|---|
| `mteb/banking77` | customer message | choice: intent (77) · choice: product area (10, keyword rule over intent names) · noul: "asks about a card" |
| `SetFit/sst5` | review sentence | score: sentiment level (5) · choice: polarity (3) · noul: "expresses a positive opinion" |
| `google/boolq` | passage | noul: the dataset's question |

18,000 train states / 42,000 questions, 1 epoch, seed 2, one H100 (229 s, ≈ $0.25).

## Measured (test: 1,500 states / 3,508 questions; OOD: 4,012 never-seen questions on the same states)

| | accuracy | Brier | ECE |
|---|---|---|---|
| in-domain (question types seen in training) | **0.854** | 0.213 | 0.022 |
| banking77 intent (77 options) | 0.916 | | |
| boolq | 0.879 | | |
| sst5 level (5 ordered) | 0.599 acc / 0.51 MAE | | |
| **OOD** (new instructions and new option sets) | **0.690** | 0.399 | 0.035 |
| OOD: negated boolq noul | 0.83 | | |
| OOD: banking77 topic (new 5-way partition) | 0.61 | | |
| OOD: sst5 stars / disappointed (new level sets) | 0.45 (majority 0.26) | | |

Three seeds of this configuration: in-domain 0.847 ± 0.005, OOD 0.678 ± 0.012. Larger context-free
sweeps, the head ablation (a fresh marker-token head does not learn; the span head does), the comparison
against ModernBERT-base/large and against a LoRA-tuned LLaDA-MoE-7B-A1B (0.835 in-domain at 16× the
latency), and the code-decision variant (which definition does this one reference, 0.63 on held-out
namespaces, chance 0.18) are in the repository README.

## Limitations

- **It reads the question only partly.** In-domain 0.85 vs OOD 0.69 is the honest gap; unseen ordered
  scales are the weakest (score on new level sets is barely above majority).
- English only; 512-token context; three public domains (banking support, movie reviews, Wikipedia yes/no).
  Anything else is out of distribution and should be measured before use.
- Confidence is the max probability after temperature scaling on the validation split (in-domain ECE
  0.022); on OOD questions it is over-confident by ~0.03 mean and should be re-calibrated on your data.
- It cannot generate text or arguments; it chooses among the options you give it.

## Provenance

Built in the `kotoba-lang` workspace with the recipe and measurements recorded in
ADR-2609181544 / ADR-2609181715 (superproject `com-junkawasaki/root`). Base model
`microsoft/deberta-v3-large` (MIT). Datasets: banking77 (CC-BY-4.0), SST-5, BoolQ (CC-BY-SA-3.0).
