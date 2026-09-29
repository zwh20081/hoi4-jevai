# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

JevAI: a typed-decision model ("Jev", a DeBERTa-v3-large encoder with an option-scoring head) picks one of six AI
postures for Hearts of Iron IV countries in running games. The randomized decisions and their outcomes are the
training data. `README.md` has the pipeline and every measured result so far; read it before changing collection,
labels or training.

## Two machines

- **VM** (VMware Windows guest) runs the games: `python -m mods.collector` and `python -m mods.collector.jevd`, CPU
  inference. It mounts this folder as a shared drive. `.venv/` is the VM's env (uv, CPU torch, base Python only in the
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
python -m mods.jevai.workshop temp/release/jevai --note "..."  # create/update the Workshop item via the Steam client
```

## Player runner (mods/jevai/runtime/runner.py)

Players start the runner manually with `jevai.exe` before each game, or run `jevai.exe --install` once (the release's
`install.cmd`): it copies the runner without the model to `%LOCALAPPDATA%\jevai\runner`, adds a Startup-folder shortcut
that runs the copy with `--hidden --mod <mod folder>` at logon, prepares the model and starts it; `--uninstall` removes
the shortcut, the copy and the NPU cache. `runner.prepare` compiles the model into `%LOCALAPPDATA%\jevai\ov_cache` in a
child `jevai.exe --compile DEVICE` (a compile in the runner's own process could not be stopped): the NPU first, each
device stopped after `runner.LIMIT` (NPU 10 min, CPU 5; the NPU takes ~2.5 min on a Core Ultra 300); an NPU failure
or time-out is recorded in `ov_cache\npu_skip.json` (keyed by model files and OpenVINO version; `--device NPU`, a
reinstall or a new model retries) and later runs use the CPU. It runs at install (progress in the install window), when
the runner starts and when a game starts; from a warm cache that takes seconds, so `runner.Model` loads before HOI4 has
finished loading. OpenVINO's cache key is the model path as written (`D:/x` and `D:\x` compile twice) and its blobs are
read-only, so `runner.compile_graph` normalizes the path and `uninstall` clears the read-only bit before deleting. Running from the copy keeps the mod folder unlocked, so Steam can update the Workshop item;
when HOI4 starts and the mod's `jevai.exe` differs from the copy (size, mtime), the copy runs the mod's
`jevai.exe --install` and exits (`runner.updated_runner`). The runner (one per user, named mutex) waits for
`hoi4.exe`, then works with the JevAI copy the playset enables (`runner.game_mod`: the `.mod` named JevAI in
`dlc_load.json`, for the orders, the model and the self-update; it warns when none is enabled or two `.mod` files carry
the name), prepares and loads the model (naming the device; if the NPU fails in-process anyway, the CPU), steers that
game until the process exits, frees the model
and waits again (`runner.session`; an exception is logged and the runner sits out that game). The user folder is
`<Documents>\Paradox Interactive\Hearts of Iron IV` with Documents from `SHGetKnownFolderPath` (OneDrive can move it;
`--userdir` overrides). `runner.silence` says once why a game logs nothing for JevAI: no `game.log` after a minute,
JevAI missing from `system.log`'s Active Mod lines, or two game days without a `JEV|` line (an overhaul replaced its
scripts). No launch option, dump or console input: `scripted_effects/jevai_state.txt` (`jev_log_state`) logs a country's
state to `game.log` (`JEV|S` numbers, `T` stability/war support/ratios/posture, `G` ideology group key, `H`
human-played, `MAJOR`, `N`/`E`/`A` one line per neighbour/enemy/ally), monthly for every country and, with the weekly
period, weekly for the countries in range. `runner.GameLog` groups lines into bursts by game date and keeps each
country's latest lines, and `runner.log_records` rebuilds the training-format state text. Country names come from the
game's and active mods' English localisation (`TAG_<group>`, then `TAG`) and ideologies from the group key, so the text
stays English in any game language (modded ideology groups keep their localized name). Units at the front, fleets
and equipment requests are not available to scripts: the text shows 0 / omits them (a known difference from the
dump-built training text; everything else matched on 91/91 countries of a real month, `temp/checks/check_logpath.py`).
Two start-of-game events (`events/jevai.txt`) set the range (`jevai.1`: major powers, all AI countries with
`--min-factories` 20+, or none; `JEV|MODE|majors|all|off`) and the period (`jevai.2`: `JEV|PERIOD|week|month|quarter`).
`on_startup` only runs for a new game, never for a loaded save (measured 2026-09-28), so `jev_announce`
(`scripted_effects/jevai_state.txt`) runs from it and from every weekly and monthly update, at most once a day: it logs
both, gives saves from 0.2 (no period) their old pace (majors weekly, all AI monthly), and asks the questions in a
campaign that never answered them (an unanswered event times out after 13 days to its first option). With `-debug`,
the game glues an empty log line in front of each `log` effect line, so test parsers search for `]: JEV|`. The newest completed burst is decided for the countries in range once `runner.due` allows it
(at least 7/28/89 game days since the last decision; at once after loading an earlier save), keeping the current
posture unless another scores `--stick` (0.01) higher; on a CPU a batch longer than ~20 s is announced first, and
updates that arrived meanwhile are reported as skipped. The orders go to `history/units/JEVAI_orders.txt` of that
JevAI copy (emptied when a new HOI4 process starts); the mod's `on_daily`
reloads it with `load_oob` (once per day, whatever the tags) and `jev_follow_orders` applies the posture to AI
countries only. The runner logs to `%LOCALAPPDATA%\jevai\jevai.log`. Unverified in-game so far: that `load_oob` re-reads the file from disk on every call (`JEV|ACT`
lines in game.log show postures being applied).

Experimental multiplayer (`docs/superpowers/specs/2026-09-28-multiplayer-design.md`): every game must apply identical
orders on the same day. `jevai.exe --host` / `--join CODE` store the role in
`%LOCALAPPDATA%\jevai\multiplayer.json`; bare `--join` reads and validates a code in Python. `join.cmd` passes only
that fixed switch; never interpolate untrusted input through CMD (`%CODE%` is command injection). Configure the role
before launching the game. The multiplayer launch marker gates host-only inference; clients receive orders through
a Steam lobby relay child. Unconfigured multiplayer runs make no decisions. Order sets target the host's newest
logged date plus `multiplayer.MARGIN_DAYS`; the guard uses `date >` the preceding day. Single-player ignores roles.
The relay is isolated from inference and accepts a Steam API path override (`--steam-api`); failures must remain
visible in the runner log. This is experimental: real two-machine timing, matching action receipts, relay interruption
and resync are not yet accepted. Do not claim desync-free play, bump to 0.4.0 or publish before that acceptance.
Package smoke tests use `--out temp/release-mp/jevai`, never overwrite the published `temp/release/jevai` build.

Load order: overhaul mods `replace_path` `common/on_actions`, `events`, `common/scripted_effects` and `history/units`,
which drops those folders from every mod loaded before them, and mods load alphabetically unless a dependency says
otherwise. The shipped `descriptor.mod` has no hard dependencies. `runner.patch_load_order` adds only enabled mods
that use `replace_path` to the `descriptor.mod` of every JevAI copy (`--mod` and each folder a `.mod` named JevAI
points to) and to those launcher `.mod` files (`jevai.mod`, or `ugc_<id>.mod` for the Workshop item) at each start,
effective at the next launch. The launcher rewrites every `ugc_<id>.mod` from the Workshop folder's `descriptor.mod`
when it starts, so for the Workshop item only the latter counts. Two `.mod` files with the same `name`: HOI4 dropped
the enabled Workshop copy while a local `jevai.mod` named JevAI existed (no Active Mod line, no events), and loaded it
alone, next to a renamed local copy, or from a plain `.mod` (`temp/checks/modtest.py`, 2026-09-27).

Releases: code on GitHub `zwh20081/hoi4-jevai`, models on Hugging Face `zwh20081/hoi4-jevai` (`torch/`, `openvino/`),
the mod on the Steam Workshop (item 3808093078). `mods.jevai.workshop` uploads `temp/release/jevai` through the running
Steam client with a stock `steam_api64.dll` of Steamworks SDK 1.48+ (`--steam-api`, default the game's: a replaced or
older one crashes or hits the wrong interface version); the item id is `remote_file_id` in `descriptor.mod`, the page
text `workshop.txt`.

- `trainer/export.py` writes two IR graphs of the same model, FP16, fixed shapes (1 x 512 tokens, 3 posture
  questions): `jev_npu.xml` uses `trainer/npu_attention.py` (relative-position gathers as one-hot matmuls, mask as an
  additive bias, the attention scale as a Python constant; the traced TorchScript scale made the NPU wrong) and runs
  ~7x faster on the Intel NPU; `jev_cpu.xml` keeps the stock attention, which is faster on CPU. The runner uses the
  NPU if present and its compile (75-160 s here) succeeds within the limit, else the CPU (compile ~5 s, ~9 s per
  country); both compiles are cached in `%LOCALAPPDATA%\jevai\ov_cache` (`runner.prepare`).
- The `ov` conda env (CPU torch, openvino, nncf, pyinstaller) is separate from `py313` so exports never disturb
  training. Training itself runs fastest on a CUDA box: `trainer/train.sh <name>` starts or resumes it in tmux.

## Layout rules

- `models/`: only `*.txt` download pointers are committed (`models/*/` is ignored). Weights never go into git; the
  trained and OpenVINO models are on Hugging Face (`models/hoi4-jevai.txt`). The stock bundle lives in
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
- `launch.ps1` finds the game through `$env:HOI4_EXE` (set this to the full path of `hoi4.exe`). Live userdirs stay on
  the VM's local disk (`JEVAI_SCRATCH`) because the shared folder is too slow; `temp/hoi4user/` is only copied from,
  and its `dlc_load.json` enables `mod/jevai.mod`.
