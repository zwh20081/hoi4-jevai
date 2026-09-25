# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

JevAI: a typed-decision model ("Jev") picks strategic postures for Hearts of Iron IV AI countries in running games, and the randomized decisions plus their outcomes become training data for fine-tuning the same model. Zero-shot, the stock model is at or below the majority baseline on every HOI4 question, so the current phase collects that data.

- Repo root: a Hugging Face snapshot of `com-kotobalabs/open-jev-deberta-v3-large` (weights, tokenizer, `head.safetensors`, `open_jev_config.json`). The root `README.md` is that model card, not project docs. These files are the stock model: the base of every training run and the default `--model` for inference. Don't overwrite them; write checkpoints under `runs/ckpt/`.
- `typed_decisions/`: the model's library, shipped inside the snapshot.
- `jev_hoi4/jevai/`: the project package. `jev_hoi4/DATA_DESIGN.md` (P0 measurements, modeling plan) and `jev_hoi4/COLLECTION.md` (collection setup) explain the reasoning but are older than the code; argparse definitions win over docs and docstrings (the collector docstring's `--jev-every-other` is now `--jev alt`).
- `jev_hoi4/p0/`: PowerShell window/input automation for the game, plus `bench_model.py`.
- `mod/jevai_probe/`: the HOI4 mod (game 1.19.*).
- `hoi4user/`: template HOI4 userdir (settings, DLC list, start saves `jev_1936_06.hoi4` and `jev_1936_06_nonhist.hoi4`).
- `runs/`: all outputs. `games/<run>/` (history_dump, meta.json, actions.jsonl, control.json), `dataset/`, `analysis/` (effects cache), `inst/` (pid and registration files), `collect.log`, `jevd.log`.

There are no tests or linter. `requirements.txt` lists the Python deps unpinned.

## Two machines

- **VM** (VMware Windows guest) runs the games: `jevai.collector` and `jevai.jevd`, with the model on CPU. It mounts this folder as `Z:\hoi4jevai`. `.venv/` is the VM's env (uv, CPU torch); its base interpreter exists only in the VM, so it does not run on the host.
- **Host** trains on an Intel Arc B390 GPU (torch XPU) in the conda env `py313`.

This folder is shared live with the VM, and a collection is usually running:
- Leave `runs/inst/` and the `runs/games/<run>/` of unfinished games alone.
- The collector copies `mod/jevai_probe` into a game's userdir at every launch, so mod edits reach the next game the VM starts.
- Code edits reach the VM when it restarts the collector or jevd.

## Commands

Run from the repo root with `jev_hoi4` on `PYTHONPATH` (the package uses relative imports, so use `-m`). CLI paths are relative to the cwd; module defaults are relative to the repo root.

Host setup (the XPU wheels are not on PyPI, so torch goes first):
```powershell
conda create -n py313 python=3.13 -y; conda activate py313
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install -r requirements.txt
```

Host, training and analysis:
```powershell
conda activate py313; $env:PYTHONPATH = "jev_hoi4"
python -m jevai.build_dataset        # runs/games/*/history_dump -> runs/dataset/{train,val,test}.jsonl + stats.json
python -m jevai.train --train runs/dataset/train.jsonl --val runs/dataset/val.jsonl --out runs/ckpt/<name> --device xpu
                                     # [--freeze 18] [--bs 8] [--lr 2e-5] [--steps 1500] [--upsample 3] [--max-states N]
python -m jevai.evaluate runs/dataset/test.jsonl --model runs/ckpt/<name> [--n 200]   # CPU only
python -m jevai.effects --h 6 --pv 2 # do postures change production and outcomes (bootstrap CIs over games)
python jev_hoi4/p0/bench_model.py xpu xpu:bf16   # smoke test: load the model, time decide()
```
Collection progress, readable from the host: `runs/collect.log` (collector events), `runs/jevd.log` (jevd stdout, one line per decision round), `runs/games/<run>/meta.json` (written when a game ends). `jevai.status` / `status.sh` are VM-side.

VM, collection:
```
python -m jevai.collector --slots 3 --games 9   # defaults: --jev alt --saves hist --every 6 --eps 0.5 --pv 2
python -m jevai.jevd                            # separate process; serves every game registered in runs/inst/*.jev
bash jev_hoi4/status.sh
python jev_hoi4/jevai/gen_mod.py mod/jevai_probe   # regenerate the posture library in the mod
```
- The policy in a game's registration (the collector's `--every/--eps/--pv`) overrides jevd's own CLI defaults, which only apply to static `--inst` games.
- Restarting the collector while games run: pass `--adopt <slot>:<run>:<save>:<seed>:<jev 0|1>` for each live game, or that slot launches a new game into the same userdir. `--start` defaults to the number after the highest `runs/games/gNN`.

## Architecture

```
VM  jevai.collector  one hoi4.exe per slot (-ogl -dump_history -continuelastsave), each with its own userdir
      │  runs/inst/<slot>.pid; for Jev games runs/inst/<slot>.jev (registration + policy)
      ▼
    jevai.jevd       each new history_dump month: state text per controlled country
      │              → one posture question to the model → eps-greedy pick
      │              → <userdir>/jev_apply.txt (`e TAG jev_set_posture_k`) + console `run jev_apply.txt`
      │              → runs/games/<run>/actions.jsonl, control.json
      ▼
    game + mod       applies the posture, logs JEV|ACT; monthly JEV|T / JEV|P telemetry into the dump's `logs`
      ▼
    collector        at game end: history_dump (minus PNGs) + meta.json → runs/games/<run>/
Host  build_dataset (dump_parse → examples) → train → evaluate / effects → new bundle for `jevd --model`
```

Contracts that span files:

- **Posture ids.** `gen_mod.NAMES` is the only source: index + 1 = `k` in `jev_set_posture_k` and in the `jev_posture` country variable, 0 = none. jevd and effects import it; `examples.POSTURES` is an older, unused list. A posture is a bundle of additive `add_ai_strategy` entries (plus an AI-only dynamic modifier) with an exact inverse, so switching = undo old + apply new. To change postures, edit `POSTURES`/`MODIFIERS` in `gen_mod.py`, bump `VERSION`, regenerate, and launch games with `--pv <VERSION>` so analyses can tell versions apart (`effects.py --pv`). Never hand-edit the generated `scripted_effects/jev_postures.txt` and `dynamic_modifiers/jev_postures.txt`; `on_actions/jev_probe_on_actions.txt` is hand-written.
- **Telemetry.** The `JEV|T|TAG|k=v|...` (stability, war support, strength ratios, posture) and `JEV|P|TAG|...` (cumulative production) lines from `on_monthly` land in the dump's `logs`. `dump_parse.parse_logs` turns them into each record's `jev` and `prod`, which `examples.state_text` and `effects.load` read. Change these together.
- **Console input** (`jevai/console.py` + `p0/console.ps1`). The game ignores posted window messages, so console.ps1 focuses the window and uses SendInput, serialized across processes by the `Global\JevAIConsoleInput` mutex; exit code 2 means focus failed. Every send ends with `e <TAG> jev_ack_<n>` and counts only once `JEV|ACK|<n>` appears in `<userdir>/logs/game.log`. A missing ack means the console toggle inverted, so the retry skips the opening toggle. The collector uses ack ids 0-4, jevd 5-9 (the mod defines 0-19). `observe`, `debug_norender` and `debug_nogui` are toggles that must land exactly once; only `gamespeed 5` is safe to resend. `-cmds` takes one `;`-separated string because arrays don't survive a bash → PowerShell call.
- **Causal data.** Each `actions.jsonl` row stores the state text, the full distribution, the action, `explore`, `propensity` and `applied`. Which countries Jev controls is a coin flip per (run, tag) saved in `control.json`; decisions are seeded by run, month and tag, so a jevd restart reproduces them. Jev and vanilla games alternate in pairs that share a start save. Keep all of this intact: an action chosen by anything unrecorded breaks the causal labels and the inverse-propensity weights.
- **Dataset.** `dump_parse` makes one record per country-month with outcomes 3/6/12 months later. `examples` turns records into forecast questions (`grow`, `capit`, `gain`, `lose`, `newwar`, `whowar`, `took`) and, for records under a Jev action, action-value questions (`act_grow`, `act_ok`, `act_gain`), one Example per record per horizon. `build_dataset` splits by whole game from the run number (`gNN`: NN % 5 == 4 → test, 3 → val, else train). Parsed records are cached in `runs/dataset/records/<run>.jsonl` and never refreshed: delete that file when a run was re-collected or the horizons changed.
- **Model input.** `[CLS] [STATE] state [Q] question [OPT] option ... [SEP]`; one forward pass answers every question, each as a softmax over its own options (`noul` has the fixed options `(no, yes)`). The Collator silently cuts the state to 256 tokens and raises if the whole sequence passes 512; `examples.state_text` is sized to fit, so recheck the token count when adding fields. jevd, train and evaluate reach into `typed_decisions` internals (`OpenJev._question`, `.collator`, `.model(...)`, `encoder.predict`, `decision_loss`), so changes there ripple.
- **Training.** `train.py` always starts from the stock bundle at the repo root (no resume). It freezes the embeddings and the lowest `--freeze` of 24 layers, trains with CE + Brier, repeats states carrying a rare positive label `--upsample` times, validates on up to 400 val states every 500 steps, refits the temperature on val, and writes an OpenJev bundle plus `<out>.log`. Judge results against the majority-class column that train and evaluate print; capitulation, territory and new-war positives are rare.

## Why the collection code looks odd

- `-ogl`: the D3D11 renderer crashed when a second instance loaded under VMware.
- Fresh 1936 starts crash around June 1936 every time, so games start from 1936-06-01 saves through `continue_game.json` + `-continuelastsave`, with no menu clicks.
- About 5.6 GB RAM per instance; launches are staggered by 30 s.
- `launch.ps1` finds the game through `$env:HOI4_EXE` (the default is the VM's install path). Live userdirs sit on the VM's local disk (`JEVAI_SCRATCH`, default `%LOCALAPPDATA%\jevai\inst\u<N>`) because the shared folder is too slow; `hoi4user/` is only copied from.
- `collect.sh`, `collect_all.sh` and the pixel-clicking `p0/loadgame.ps1` / `p0/newgame.ps1` are the older single-instance path.
