# Layout and training on the host

The project folder is self-contained: every script finds files relative to its own location, so the
same folder works in the VM (as `Z:\hoi4jevai`, a VMware shared folder) and on the host (the folder
behind that share).

## What lives where

| Path (relative to the project folder) | What |
|---|---|
| `model.safetensors`, `head.safetensors`, `tokenizer.*`, `open_jev_config.json` | base open-jev model |
| `typed_decisions/` | model code (encoder, loader) |
| `jev_hoi4/jevai/` | pipeline code: collector, jevd, dump_parse, examples, build_dataset, train, evaluate, effects, status |
| `jev_hoi4/p0/` | PowerShell helpers that drive the game window (VM only) |
| `mod/jevai_probe/` | the HOI4 mod (telemetry + postures); copied into each game's userdir at launch |
| `hoi4user/` | template game userdir: settings, start saves `save games/jev_1936_06*.hoi4` |
| `runs/games/gNN/` | **the collected data**: `history_dump/` (one JSON per game month), `actions.jsonl` (Jev decisions), `control.json`, `meta.json` |
| `runs/dataset/` | built training set (`train/val/test.jsonl`, `stats.json`) |
| `runs/collect.log`, `runs/jevd.log` | collection logs |

Only two things are machine specific, and neither is inside the project:
- the game install, `HOI4_EXE` (default: the VM's `C:\Users\zwh20081\Desktop\h4\hoi4.exe`);
- the games' working userdirs, `JEVAI_SCRATCH` (default `%LOCALAPPDATA%\jevai\inst`), kept on a local disk
  because the shared folder is too slow for live game files. Finished games are archived into `runs/games`.

`.venv/` is the VM's Python environment (CPU torch, base Python at C:\Python312). On the host use your
own environment instead: `torch` (XPU or CUDA build), `transformers`, `safetensors`, `sentencepiece`, `protobuf`.

## Training on the host (after collection is done)

From the project folder:

```
set PYTHONPATH=jev_hoi4
python -m jevai.build_dataset
python -m jevai.train --train runs/dataset/train.jsonl --val runs/dataset/val.jsonl --out runs/ckpt/hoi4-v1 --device xpu --freeze 0 --bs 16 --steps 3000
python -m jevai.evaluate runs/dataset/test.jsonl --model runs/ckpt/hoi4-v1
```

- `--device xpu` (Intel Arc) or `cuda`; `--freeze 0` trains all layers (the CPU default freezes 18 of 24).
- `build_dataset` splits by whole game (game ids ending in 4 go to test, 3 to validation).
- `python -m jevai.effects --pv 2` reports which postures helped or hurt in the version-2 games.

## Using the trained model in games (back in the VM)

Jev's default decision rule is `value`: for every posture it asks the three posture-outcome questions the
model was trained on (same wording as the training data, `examples.action_texts`) and picks the posture
with the best score

    U = 1.0 * P(no territory lost in 6 months) + 1.0 * P(territory gained) + 0.5 * expected growth level (0..1)

(weights: `--w-ok`, `--w-gain`, `--w-grow`; look-ahead: `--horizon`). `--rule choice` restores the old
zero-shot question used for games g05-g33. Every decision logs its rule, model, per-posture scores and the
game date it took effect (`applied_date`).

    python -m jevai.jevd --model runs/ckpt/hoi4-v1
    python -m jevai.collector --slots 3 --games 6 --start 40 --jev all --eps 0.2

With `--jev all` every game has Jev controlling a random half of the eligible countries, so each game
compares trained-Jev countries with vanilla ones (`python -m jevai.effects --pv 2`, section 3).

Speed: the value rule runs one model pass per country and posture (6 per country). On the VM's CPU that
is ~0.7 s each, so a full decision round for ~20 countries takes 1.5-2.5 minutes, i.e. 1-2 game months;
the actual date is logged. 8-bit quantization was 1.8x faster but picked the same posture only 2/12 times,
so it is not used.
