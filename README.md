# JevAI

JevAI lets a typed-decision model steer Hearts of Iron IV AI countries. Every few game months, the model (a
fine-tuned [open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)) reads a
country's situation as text, scores six strategic postures, and the best one is applied to that country's AI through
the game console. The games also produce the training data: decisions are partly random with known odds, so what
happened afterwards is a causal label for the posture taken.

## How it works

- **The mod** (`mods/jevai`) adds six postures, each a bundle of AI-only levers (`add_ai_strategy` entries and `ai_*`
  modifiers for unit mix, production, factory ratios and construction). None gives a gameplay bonus; they change what
  the AI decides. The mod also logs monthly telemetry: stability, war support, strength ratios, the posture in force
  and cumulative production.
- **The game** runs with `-dump_history`, which writes one JSON per game month: factories, divisions, fronts, wars,
  the AI's own strategy plans and the mod's telemetry.
- **The runtime** (`mods/jevai/runtime`) turns a month into one text per country and asks the model three questions
  per posture, 6 months ahead: power trend, no territory lost, territory gained. It picks the posture with the best
  `P(no loss) + P(gain) + 0.5 * expected growth (0..1)`. Commands reach the game through its console, typed with
  SendInput (the game ignores posted window messages) and confirmed by an acknowledgement effect in the log.
- **Collection** (`mods/collector`, in a Windows VM) runs several games at once. In Jev games a coin flip per country
  decides whether Jev or the vanilla AI controls it, and 30 to 50% of Jev's decisions are random, each logged with its
  odds.
- **Datasets** (`datasets/`) label every country-month with what happened 3, 6 and 12 months later (forecast
  questions) and every confirmed decision with its outcome (posture questions). Games are split whole into
  train, validation and test.
- **Training** (`trainer/`) fine-tunes the model with cross-entropy + Brier loss and keeps the checkpoint with the
  best validation Brier skill.

## Layout

```
models/          download pointers (*.txt); the weights are downloaded next to them and not committed
mods/jevai/      the mod: HOI4 script files + runtime/ (the Python side, shipped with the mod)
mods/collector/  VM-only data collection: game queue (__main__.py), Jev in the loop (jevd.py), window helpers
datasets/        build.py (games -> train/val/test), effects.py (posture effect analysis)
trainer/         model.py, train.py, evaluate.py
temp/            everything generated: games, dataset, training runs, the template game userdir (not committed)
```

## Setup

Python 3.13 with a torch build for your device, then `requirements.txt`. On an Intel GPU:

```
conda create -n py313 python=3.13 -y && conda activate py313
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install -r requirements.txt
hf download com-kotobalabs/open-jev-deberta-v3-large --local-dir models/open-jev-deberta-v3-large
```

Collection also needs HOI4 1.19 on Windows (`HOI4_EXE`), the template userdir `temp/hoi4user` with the start saves
`save games/jev_1936_06.hoi4` and `jev_1936_06_nonhist.hoi4`, and optionally `JEVAI_SCRATCH` for the games' live
userdirs (default `%LOCALAPPDATA%\jevai\inst`).

## Commands

From the repo root:

```
python -m mods.collector --slots 3 --games 9                    # collect (VM); --status shows progress
python -m mods.collector.jevd --model temp/train/<name>/best    # Jev in the loop, next to the collector
python -m datasets.build                                        # temp/games -> temp/dataset
python -m trainer.train --out temp/train/<name> --bf16          # fine-tune; keeps temp/train/<name>/best
python -m trainer.evaluate temp/dataset/test.jsonl --model temp/train/<name>/best --device xpu --bf16
python -m datasets.effects --pv 2                               # which postures helped or hurt
python -m mods.jevai.runtime.postures                           # regenerate the mod's posture files
```

## What was measured

Feasibility, on a 12-core 16 GB VM (2026-09-23):
- Console `run <file>` plus `e TAG <scripted_effect>` works from outside the game, and `add_ai_strategy` takes effect.
  Posted key and mouse messages are ignored; SendInput with the window in the foreground works.
- Fresh 1936 starts crashed 8 times out of 8 between 1936-06-01 and 06-24, with the same access violation whatever
  the mod or launch flags. Loading the 1936-06-01 autosave gets past it, so every game starts from there.
- D3D11 crashed when a second instance loaded under VMware; `-ogl` does not. One instance needs about 5.6 GB.
- Observer mode at speed 5 takes about 22 s per game month, or 30 minutes for 1936-06 to 1943.
- The stock model is at or below the majority baseline on every HOI4 question: it knows nothing about the game.

Collection up to 2026-09-25: 33 games (g01-g24 with posture set v1, g25-g33 with v2), 1,817 game-months and 6,890
Jev decisions (5,917 v1, 973 v2). The dataset has 446k states and 2.4M questions (train 249k, validation 114k,
test 83k states), including 17,910 posture question sets. Rare events: capitulation 2.4%, territory gained 5.6%,
territory lost 5.4%, a new war 19.1%.

Posture effects in the v2 games (`datasets.effects --pv 2`, 95% intervals from resampling whole games):
- Production follows the posture: air power 1.49x planes [1.23, 1.60], armored offensive 1.30x armor [1.14, 1.39],
  defensive buildup 1.19x support equipment [1.08, 1.27], total war economy 1.12x military factories [1.10, 1.18].
  Naval power is unclear: 1.08x ships [0.86, 1.17].
- Compared with total war economy over 6 months, industrial expansion gained +2.4 points of power growth
  [+0.9, +3.3] and +0.6 states [+0.35, +0.86], from 72 decisions in 4 games. Power counts factories and divisions,
  not planes or ships, so the air and naval postures look worse by construction.
- The untrained model did no better than the vanilla AI: -1.3 points of growth [-4.6, +1.7] in v2 games, and -0.17
  states per 6 months [-0.26, -0.05] in v1 games.
