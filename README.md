# JevAI

JevAI lets a trained model steer the Hearts of Iron IV AI. Every game month (every week for major powers, if you
choose), a fine-tuned [open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
reads each AI country's situation as text, scores six strategic postures, and the mod applies the best one to that
country's AI. The games JevAI collects are also its training data: decisions there are partly random with known odds,
so what happened afterwards is a causal label for the posture taken.

- The mod: [Steam Workshop](https://steamcommunity.com/sharedfiles/filedetails/?id=3808093078)
- The model: [huggingface.co/zwh20081/hoi4-jevai](https://huggingface.co/zwh20081/hoi4-jevai) (`torch/` the PyTorch
  bundle, `openvino/` the graphs the mod runs)

## Play

Needs Windows 10 or 11 (64-bit) and Hearts of Iron IV 1.19. The model runs on an Intel NPU (Core Ultra) if there is
one, otherwise on the CPU.

1. Subscribe to the Workshop item (about 2.5 GB with the model). Or build the release (`mods.jevai.package`, below)
   and copy `temp/release/jevai` into `Documents\Paradox Interactive\Hearts of Iron IV\mod\`, with `jevai.mod` next
   to it.
2. Start the companion runner from the mod's `runner` folder. For a Workshop subscription, that folder is
   `<Steam library>\steamapps\workshop\content\394360\3808093078\runner`:
   - Run `install.cmd` once to copy the runner to `%LOCALAPPDATA%\jevai`, start it now, and start it at Windows sign-in.
   - Or run `jevai.exe` before each game, keep its console open while playing, and press Ctrl+C to stop it afterward.
     This does not add a Windows Startup entry.
   SmartScreen may warn about the unsigned `jevai.exe` (More info, Run anyway).
3. Play from the Paradox launcher with JevAI in the playset. An event at the start of each game asks who the model
   commands: major powers (decided weekly), all AI countries with 20+ factories (monthly), or none.

The first game compiles the model for the NPU (about 2 minutes, while the game loads); later games load it from the
cache. A decision takes about 1.5 s per country on the NPU. `%LOCALAPPDATA%\jevai\jevai.log` has one line per update,
and `runner\uninstall.cmd` removes the installed runner, its model cache and the Startup entry. Multiplayer is not supported:
each player's game would read its own orders. With overhaul mods, JevAI loads after them, but the model has only seen
vanilla 1936 games.

No game console, launch option or memory access is involved. The mod logs each country's state to `game.log`
(`scripted_effects/jevai_state.txt`); the runner (`mods/jevai/runtime/runner.py`, frozen as `jevai.exe`) reads the
log, scores the postures and writes `history/units/JEVAI_orders.txt` in the mod folder, which the mod reloads daily
with `load_oob`.

## How the model is made

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
trainer/         model.py, train.py, evaluate.py, export.py (OpenVINO)
temp/            everything generated: games, dataset, training runs, the template game userdir (not committed)
```

## Setup

Python 3.13 with a torch build for your device, then `requirements.txt`. On an Intel GPU:

```
conda create -n py313 python=3.13 -y && conda activate py313
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install -r requirements.txt
hf download com-kotobalabs/open-jev-deberta-v3-large --local-dir models/open-jev-deberta-v3-large
hf download zwh20081/hoi4-jevai --local-dir models/hoi4-jevai    # the trained model (torch/, openvino/)
```

Export and packaging use a separate env with CPU torch, `openvino`, `nncf` and `pyinstaller`.

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
python -m trainer.export --model temp/train/<name>/best --out temp/ov/<name>   # OpenVINO: NPU and CPU graphs
python -m mods.jevai.package --model temp/ov/<name>             # release: the mod + jevai.exe + the model
python -m mods.jevai.workshop temp/release/jevai --note "..."   # create or update the Workshop item (Steam running)
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

The trained model hoi4-v1 (one A100, 2.9 h, 12,116 steps; kept step 10,000, validation macro skill 0.672), Brier
skill on the test games against predicting the label frequencies (`trainer.evaluate`):
- All ten question types, macro: 0.625 (stock model -0.067). Power trend 0.45, territory gained 0.60, territory lost
  0.45, new war 0.69. Capitulation scores 1.000 because capitulated states carry fewer questions, which gives the
  answer away; to fix before the next training.
- The three posture questions the runner acts on: 0.485 on all test states (stock -0.124), 0.302 on the 5,460 states
  with a decision (stock -0.271).
- OpenVINO on an Intel Core Ultra NPU: 233 ms per sequence (a country needs 6), probabilities within 0.0025 of
  PyTorch.

## License

Apache-2.0, for the code and the model. The model is fine-tuned from open-jev-deberta-v3-large (Apache-2.0), which is
based on microsoft/deberta-v3-large (MIT).
