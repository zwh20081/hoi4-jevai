# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

JevAI: a typed-decision model ("Jev", a DeBERTa-v3-large encoder with an option-scoring head) picks one of six AI
postures for Hearts of Iron IV countries in running games. The randomized decisions and their outcomes are the
training data. `README.md` has the pipeline and every measured result so far; read it before changing collection,
labels or training.

## Two machines

- **VM** (VMware Windows guest) runs the games: `python -m mods.collector` and `python -m mods.collector.jevd`, CPU
  inference. It mounts this folder as `Z:\hoi4jevai`. `.venv/` is the VM's env (uv, CPU torch, base Python only in the
  VM); it does not run on the host.
- **Host** (this machine) trains on an Intel Arc B390 iGPU (torch XPU, shares host RAM) in the conda env `py313`.
  Setup: `conda activate py313`, `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu`,
  then `uv pip install -r requirements.txt` (`requirements.txt` has no index line, so torch must come first).

The folder is shared live with the VM. Before moving or deleting anything in `temp/`, check that no collection is
running (`temp/collect.log`, `temp/jevd.log`, `python -m mods.collector --status`). The collector copies the mod's game
files into each game's userdir at launch, so mod edits reach the next game the VM starts; code edits reach the VM
when it restarts the collector or jevd.

## Commands

Everything runs from the repo root as `python -m <package>.<module>`; there are no `sys.path` edits, and CLI paths
are relative to the cwd. No test suite or linter.

```
python -m datasets.build                        # temp/games -> temp/dataset/{train,val,test}.jsonl + stats.json
python -m trainer.train --out temp/train/<name> --bf16 [--device xpu] [--freeze 0] [--bs 16] [--steps 3000]
                                                #   [--eval-every 500] [--val-states 2000] [--max-states N]
python -m trainer.evaluate temp/dataset/test.jsonl --model temp/train/<name>/best --device xpu --bf16 [--n 2000]
python -m datasets.effects --h 6 --pv 2         # posture effects from the games (bootstrap CIs over games)
python -m mods.jevai.runtime.postures           # regenerate the mod's posture files
python -m mods.collector --slots 3 --games 9    # VM; --status; restart with --adopt slot:run:save:seed:jev per live game
python -m mods.collector.jevd --model temp/train/<name>/best   # VM, next to the collector
python -m trainer.export --model temp/train/<name>/best --out temp/ov/<name>   # OpenVINO IR (conda env `ov`)
python -m mods.jevai.package --model temp/ov/<name>           # release: mod + jevai.exe runner + model (env `ov`)
```

## Player runner (mods/jevai/runtime/runner.py)

Players run `jevai.exe` (the frozen runner): it starts HOI4 (or attaches with `--no-launch`) and compiles the model in a
background thread while the game loads (cold NPU compile ~2 min, overlapping the game load; later starts load from
cache). No launch option, dump or console input: on the 1st of every month the mod logs each country's state to
`game.log` (`on_actions/jevai.txt`: `JEV|S` numbers, `T` stability/war support/ratios, `G` ideology group key, `H`
human-played, `N`/`E`/`A` one line per neighbour/enemy/ally), and `runner.log_records` rebuilds the training-format
state text from them. Country names come from the game's and active mods' English localisation (`TAG_<group>`, then
`TAG`) and ideologies from the group key, so the text stays English in any game language. Units at the front, fleets
and equipment requests are not available to scripts: the text shows 0 / omits them (a known difference from the
dump-built training text; everything else matched on 91/91 countries of a real month, `temp/checks/check_logpath.py`).
Each `--every` game months it scores the six postures for the AI countries the player chose in the start-of-game event
(`events/jevai.txt`: majors / all / off) and writes `mod/jevai/history/units/JEVAI_orders.txt`; the mod's `on_weekly`
reloads it with `load_oob` (once per week, whatever the tags) and `jev_follow_orders` applies the posture to AI
countries only. Unverified in-game so far: that `load_oob` re-reads the file from disk on every call (the probe mod in
`temp/oobprobe/` tests it; it needs an unlocked desktop so the game can run).

Load order: overhaul mods `replace_path` `common/on_actions`, `events`, `common/scripted_effects` and `history/units`,
which drops those folders from every mod loaded before them, and mods load alphabetically unless a dependency says
otherwise. `descriptor.mod` lists well-known overhauls as dependencies, and `runner.patch_load_order` adds every
installed mod that uses `replace_path` to `mod/jevai.mod` and `mod/jevai/descriptor.mod` at each start.

- `trainer/export.py` writes two IR graphs of the same model, FP16, fixed shapes (1 x 512 tokens, 3 posture
  questions): `jev_npu.xml` uses `trainer/npu_attention.py` (relative-position gathers as one-hot matmuls, mask as an
  additive bias, the attention scale as a Python constant; the traced TorchScript scale made the NPU wrong) and runs
  ~7x faster on the Intel NPU; `jev_cpu.xml` keeps the stock attention, which is faster on CPU. The runner uses the
  NPU if present, else CPU, and caches the NPU compile (~75 s) in `%LOCALAPPDATA%\jevai\ov_cache`.
- The `ov` conda env (CPU torch, openvino, nncf, pyinstaller) is separate from `py313` so exports never disturb
  training. Training itself runs fastest on a CUDA box: `trainer/train.sh <name>` starts or resumes it in tmux.

## Layout rules

- `models/`: only `*.txt` download pointers are committed (`models/*/` is ignored). Weights never go into git; the
  trained and OpenVINO models go to a Hugging Face repo later. The stock bundle lives in
  `models/open-jev-deberta-v3-large/` and is the default `--init` / `--model`.
- `mods/jevai/runtime/` is the torch-free core: game parsing (`game.py`), text and question wording (`text.py`),
  token packing (`pack.py`), postures (`postures.py`), console input (`console.py` + `console.ps1`). `datasets`,
  `trainer` and `mods/collector` import it; it uses relative imports only, because it also ships inside the mod.
- `trainer/model.py` is the only torch model code. `datasets/` is a local package that shadows Hugging Face
  `datasets`; the project does not use that library.
- `temp/` holds everything generated (gitignored): `games/`, `dataset/`, `train/`, `analysis/`, `inst/`, `hoi4user/`
  (template userdir with the start saves), `collect.log`, `jevd.log`, and throwaway `checks/`, `specs/`, `plans/`.

## Architecture

```
VM  mods.collector   one hoi4.exe per slot (-ogl -dump_history -continuelastsave), userdir under JEVAI_SCRATCH
      │  temp/inst/<slot>.pid; for Jev games temp/inst/<slot>.jev (registration + policy every/eps/pv/horizon)
      ▼
    mods.collector.jevd   each decision month: state text per controlled country -> 6 postures x 3 questions
      │                   -> best score (eps-greedy) -> console `run jev_apply.txt` (`e TAG jev_set_posture_k`)
      │                   -> temp/games/<run>/actions.jsonl, control.json
      ▼
    game + mod       applies the posture, logs JEV|ACT; monthly JEV|T / JEV|P telemetry into the dump's `logs`
      ▼
    collector        at game end: history_dump (minus PNGs) + meta.json -> temp/games/<run>/
Host  datasets.build -> trainer.train (keeps <out>/best) -> trainer.evaluate / datasets.effects -> jevd --model
```

Contracts that span files:

- **Wording parity.** `text.state_text` and `text.posture_questions` produce what the model reads both in training
  (`datasets.build`) and in play (`jevd`). Change them only together with a dataset rebuild and retraining.
- **Postures.** `postures.NAMES` is the only source: index + 1 = `k` in `jev_set_posture_k` and in the `jev_posture`
  country variable, 0 = none. Changing postures means editing `POSTURES`/`MODIFIERS`, bumping `VERSION`,
  regenerating, and launching games with the new `--pv`; posture questions name the set ("strategy set v2") because
  v1 and v2 share names but not content. The generated `common/*/jevai_postures.txt` are never edited by hand;
  `common/on_actions/jevai.txt` (telemetry) is.
- **Telemetry.** `JEV|T|TAG|k=v|...` and `JEV|P|TAG|...` lines from `on_actions/jevai.txt` are parsed by
  `game.parse_logs` into each record's `jev` and `prod`, which `text.state_text` and `datasets.effects.load` read.
- **Console.** The game ignores posted window messages, so `console.ps1` focuses the window and uses SendInput,
  serialized across processes by the `Global\JevAIConsoleInput` mutex (exit code 2 = focus failed). Every send ends
  with `e <TAG> jev_ack_<n>` and counts only once `JEV|ACK|<n>` appears in game.log; a missing ack means the console
  toggle inverted, so the retry skips the opening toggle. Collector ack ids 0-4, jevd 5-9 (the mod defines 0-19).
  `observe`, `debug_norender`, `debug_nogui` are toggles that must land once; only `gamespeed 5` is safe to resend.
- **Causal data.** Each action row keeps the state text, per-posture scores, `explore`, `propensity`, `applied` and
  `applied_date`. Control is a coin flip per (run, tag) in `control.json`; decisions are seeded by run, month and tag,
  so a jevd restart reproduces them. Anything that picks actions without recording its odds breaks the labels.
- **Dataset.** Records are cached in `temp/dataset/records/<run>.jsonl` and never refreshed: delete a file when that
  game was re-collected or the horizons changed. The split is by whole game from the run number (`gNN`: NN % 5 == 4
  test, 3 validation). Example seeds use `zlib.crc32(run)`, so builds are reproducible.
- **Model input.** `[CLS] [STATE] state [Q] question [OPT] option ... [SEP]`. `Packer` cuts the state to 256 tokens
  and raises above 512. Pooling is `pool @ hidden` (`pool` rows average each option's and question's text tokens),
  so the graph has no gather/scatter. A bundle = HF backbone + tokenizer files + `head.safetensors` +
  `open_jev_config.json` (temperature, limits, provenance).
- **Training.** `--bf16` is autocast with fp32 weights and optimizer. Every `--eval-every` steps the model is scored
  on a fixed validation subset; the best macro Brier skill (per question id, against predicting the label
  frequencies) is saved to `<out>/best` with a refitted temperature. The stock model is at or below the majority
  baseline on every HOI4 question, so judge results by skill and by the `major` column that `evaluate` prints.

## Why the collection code looks odd

- `-ogl`: D3D11 crashed when a second instance loaded under VMware. About 5.6 GB per instance; launches are staggered.
- Fresh 1936 starts crash around June 1936, so games start from 1936-06-01 saves via `continue_game.json` +
  `-continuelastsave`, with no menu clicks.
- `launch.ps1` finds the game through `$env:HOI4_EXE` (the default is the VM's install path). Live userdirs stay on
  the VM's local disk (`JEVAI_SCRATCH`) because the shared folder is too slow; `temp/hoi4user/` is only copied from,
  and its `dlc_load.json` enables `mod/jevai.mod`.
