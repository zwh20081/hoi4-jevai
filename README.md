# JevAI

JevAI lets a trained model steer the Hearts of Iron IV AI. Every week, month or 3 months (you choose, and whether it
commands the major powers or all AI countries), a fine-tuned [open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
reads each AI country's situation as text, scores six strategic postures, and the mod applies the best one to that
country's AI. The games JevAI collects are also its training data: decisions there are partly random with known odds,
so what happened afterwards is a causal label for the posture taken.

- The mod: [Steam Workshop](https://steamcommunity.com/sharedfiles/filedetails/?id=3808093078)
- The model: [huggingface.co/zwh20081/hoi4-jevai](https://huggingface.co/zwh20081/hoi4-jevai) (`torch/` the PyTorch
  bundle, `openvino/` the graphs the mod runs)

The `0.4.0-experimental` package includes the v2 CPU/NPU OpenVINO model with a three-month prediction horizon.
Versioned model files and evaluation are under `v2/` on Hugging Face; the root model folders retain v1.
v2 is a small supervised continuation on verified overhaul-game outcomes, not a demonstrated increase in gameplay
strength. CPU/NPU outputs were checked against PyTorch on four real states and all six postures; both devices chose
the same options on that sample. This numerical check is separate from multiplayer gameplay acceptance.

## Play

Needs Windows 10 or 11 (64-bit) and Hearts of Iron IV 1.19. The model runs on an Intel NPU (Core Ultra) if there is
one, otherwise on the CPU.

1. Subscribe to the Workshop item (about 2.5 GB with the model). Or build the release (`mods.jevai.package`, below)
   and copy `temp/release/jevai` into `Documents\Paradox Interactive\Hearts of Iron IV\mod\`, with `jevai.mod` next
   to it.
2. Start the companion runner from the mod's `runner` folder. For a Workshop subscription, that folder is
   `<Steam library>\steamapps\workshop\content\394360\3808093078\runner`:
   - Run `install.cmd` once to copy the runner to `%LOCALAPPDATA%\jevai`, prepare the model (about 2 minutes on an
     Intel NPU, seconds on a CPU), start it now, and start it at Windows sign-in.
   - Or run `jevai.exe` before each game, keep its console open while playing, and press Ctrl+C to stop it afterward.
     This does not add a Windows Startup entry.
   SmartScreen may warn about the unsigned `jevai.exe` (More info, Run anyway).
3. Play from the Paradox launcher with JevAI in the playset. Keep only one mod named JevAI: with a local copy of the
   same name next to the Workshop item, HOI4 skipped the enabled Workshop copy. Two events at the start of each game
   ask who the model commands (major powers, all AI countries with 20+ factories, or none) and how often it decides
   (every week, month or 3 months).

The model is compiled once before any game: by `install.cmd`, or by `jevai.exe` when it starts (about 2 minutes on an
Intel NPU, seconds on a CPU). The runner checks it again whenever it starts and when a game starts, so games load it
from the cache in seconds, before HOI4 has finished loading. An NPU compile that fails or takes more than 10 minutes
is stopped and JevAI uses the CPU from then on (`jevai.exe --device NPU`, a reinstall or a new model tries the NPU
again). A decision takes about 1.5 s per country on the NPU and about 10 s on a CPU, so on a CPU weekly decisions for
all AI countries fall behind the game (the runner then decides the newest update and says how many it skipped).
`%LOCALAPPDATA%\jevai\jevai.log` has one line per update, says which `game.log` it reads and where the orders go, and
explains a game that logs nothing for JevAI (JevAI not loaded, or its scripts replaced by an overhaul mod).
`runner\uninstall.cmd` removes the installed runner, its model cache and the Startup entry. With overhaul mods,
JevAI loads after them. v2 has seen several overhaul settings, but compatibility and policy quality are not guaranteed
for every mod. Its three-month prediction horizon is separate from the weekly/monthly/quarterly decision frequency.

No game console, launch option or memory access is involved. The mod logs each country's state to `game.log`
(`scripted_effects/jevai_state.txt`); the runner (`mods/jevai/runtime/runner.py`, frozen as `jevai.exe`) reads the
log in HOI4's user folder (in the Windows Documents folder, also when OneDrive moved it), scores the postures and
writes `history/units/JEVAI_orders.txt` in the JevAI copy the playset enables, which the mod reloads daily with
`load_oob`.

## Experimental multiplayer

The `0.4.0-experimental` build includes experimental multiplayer. Two-machine gameplay acceptance has not passed,
so this is not a claim of desync-free play. Use matching game versions, playsets and
JevAI builds on every machine, with Steam running and the companion runner installed or running manually.

1. Before starting the multiplayer game, the host runs `runner\host.cmd` or `jevai.exe --host` and shares the code.
2. Other players run `runner\join.cmd` and enter the code in JevAI's prompt, or run `jevai.exe --join CODE`.
3. Leave each player's runner running. Only the host loads the model and decides; clients receive orders through
   a Steam lobby. A multiplayer game without a configured role receives no JevAI decisions.

The role is saved in `%LOCALAPPDATA%\jevai\multiplayer.json`; single-player ignores it. Configure roles before
launching a game. Orders target the host's newest logged date plus three game days. That margin is unverified on
real two-machine games: late or missing orders can cause desync. Host migration and mid-game joining are not
supported. Relay failures are recorded in `%LOCALAPPDATA%\jevai\jevai.log`; stop the experiment if delivery fails.
The single-player runner operates locally; multiplayer requires the Steam connection.

If the default Steam API DLL cannot initialize, `--steam-api "C:\path\to\steam_api64.dll"` selects a compatible
Steam API library when configuring the role. The relay loads it in a child process. Do not download arbitrary DLLs.

Build the experimental package separately so the published build stays available:

```text
python -m mods.jevai.package --model temp/ov/hoi4-v2 --out temp/release-mp/jevai
```

Before claiming validated multiplayer support, two legitimate Steam accounts on two machines must verify matching
`JEV|ACT` records across three game months, relay interruption and game resync. The current offline checks cannot
establish that result. Follow the [acceptance guide](docs/multiplayer-acceptance.md) and its offline log comparison.

## How the model is made

- **The mod** (`mods/jevai`) adds six postures, each a bundle of AI-only levers (`add_ai_strategy` entries and `ai_*`
  modifiers for unit mix, production, factory ratios and construction). None gives a gameplay bonus; they change what
  the AI decides. The mod also logs monthly telemetry: stability, war support, strength ratios, the posture in force
  and cumulative production.
- **The game** runs with `-dump_history`, which writes one JSON per game month: factories, divisions, fronts, wars,
  the AI's own strategy plans and the mod's telemetry.
- **The runtime** (`mods/jevai/runtime`) turns a month into one text per country and asks the model three questions
  per posture at the model's configured horizon (three months for v2, six for legacy v1): power trend, no territory
  lost, territory gained. It picks the posture with the best
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
- OpenVINO on the CPU (Core Ultra X9 388H, HOI4 running alongside): the CPU graph compiles in 4.4 s (0.6 s from the
  cache), 1.4 to 1.6 s per sequence, about 9 s per country, 2.3 GB of memory. In vanilla games 11 countries have 20+
  factories in 1936, 29 in 1938 and 40 to 47 from 1940.

## License

Apache-2.0, for the code and the model. The model is fine-tuned from open-jev-deberta-v3-large (Apache-2.0), which is
based on microsoft/deberta-v3-large (MIT).
