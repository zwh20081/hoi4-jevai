# How to model and collect data (P0 results + design)

Measured in this VM on 2026-09-23. Numbers are from this machine, not estimates.

## 1. What P0 established

| Item | Result |
|---|---|
| Model load/inference (CPU fp32, 16 cores) | load 4.6 s; 1 state × 4 questions 0.44–0.50 s; batch of 8 3.9 s. 200 HOI4 states × ~5 questions = 164 s |
| Training speed (CPU, full fine-tune) | bs4 ≈ 10 s/step, bs8 ≈ 25 s/step (seq ≈ 180); head only with frozen backbone: bs8 ≈ 7 s |
| Game → outside | `log = "JEV\|..."` shows up in game.log, and also in the `logs` field of history_dump |
| Outside → game | Console `run jev_t1.txt` (file sits in the userdir root) + `e TAG scripted_effect` **works**. `add_ai_strategy` took effect: ITA `ai_strategy_prepare_for_war@POL` went from 0 to 150 |
| Input injection | PostMessage keys and clicks are **ignored** by the game. It needs SetForegroundWindow + SendInput/mouse_event (`console.ps1`, `realclick.ps1`) |
| Observer / speed | Console `observe` + `gamespeed 5`. In this VM speed 5 ≈ 1.4 game days/s ≈ 22 s per game month |
| Built-in data source | `-dump_history` writes one JSON per month to `history_dump/N.txt` (see §2) |
| **Crash** | A fresh 1936 start crashed 8/8 times between 1936-06-01 and 06-24, with the same stack every time (ACCESS_VIOLATION in a worker thread). It did not depend on the mod, `-debug`, `-dump_history` or `-hands_off`. **Loading the 1936-06-01 autosave (`runs/pre_crash_1936_06.hoi4`) gets past it and has now run to 1937-10.** Collection runs therefore start from that save |

## 2. history_dump: the game's own structured data, used as the data layer

Each month, `-dump_history` writes this for every country:
- `factories` (mil/civ/nav, consumer goods, trade), `fully_controlled_states`, `armies` (one entry per division, with its location),
  `navies`, `air_missions`, `order_data` (front/invasion orders: unit count, units at the front, path),
  `enemies / allies / potential_enemies`, `resistance_data` (occupied territory)
- The `logs` lines contain the **AI's internal state**: `strat|TAG|<active ai_strategy plan>`, `manpower|TAG|available/max/in field`,
  `equipment|TAG|<equipment>|in stock/requested`, `convoy`, `supply`, `garrison`, plus the month's focuses, decisions, declarations of war and our own `JEV|` lines
- `invariants.json`: province coordinates and the state→province mapping, which I use to compute adjacency

So **the game already exports nearly all of the state I need.** The mod's `on_monthly` log only has to add the few fields the dump lacks
(stability, war support, `enemies_strength_ratio`, surrender_progress, current values of `ai_strategy_*@TAG`, and the Jev action applied this month).

Code: `jev_hoi4/jevai/dump_parse.py` → one record per country per month, with outcomes h months later attached
(change in state count, factories and divisions, power growth, new enemies, who took whose territory, capitulation).
Measured: 16 months of dump from one game → 1,081 country-month records.

## 3. Modeling: three kinds of training signal, from cheap to valuable

### A. Forecast questions (no intervention, collected at scale)
State text = this month's situation (≤256 tokens; measured max 195). Questions:
- choice: power trend over the next 3 months [shrinks / stagnates / grows slowly / grows fast]
- noul: gain territory / lose territory / capitulate / enter a new war
- choice: next war opponent (candidates = neighbours)
- noul: take territory from X

The gold label is what the game actually did. This teaches the encoder HOI4 dynamics ("reading the map") and needs no mod intervention:
an observer game yields ~90 countries × ~100 months ≈ 9,000 states.

**Zero-shot baseline (200 states, stock model):** every question is **at or below the majority baseline**
(grow 0.425 vs majority 0.44; took 0.64 vs 0.93). The model knows nothing about HOI4. Training is required, and
performance must always be judged against the majority baseline.

### B. Action-value questions (randomized intervention = causal labels)
On each run, the daemon picks some countries at random and, with ε-exploration, assigns them a posture
(an action from the §5 catalogue) and records it in `JEV|ACT|TAG|posture` logs. Questions take the form
"This country follows posture P for 3 months. How does its power change?" / "…avoids losing territory".
Because the action is randomly assigned, the label is a **causal effect**, not a correlation. At decision time, one forward pass asks about all candidate postures together
and the one with the best expected outcome is chosen. This is the only signal that can make the AI "stronger".

### C. Imitation (optional)
The `strat|` lines are the plans vanilla AI activated in that month. They can be used as a weak teacher (what vanilla would do) for warm-starting.
Low priority: the goal is to beat vanilla, not copy it.

### Label imbalance is the main problem
From 778 samples on a 3-month horizon: capitulate 6 / gain territory 15 / new war 8 (all rare events); only grow is balanced.
Remedies:
- The 1936–37 run is mostly peacetime; 1939–43 has far more positive samples. **Prioritize collecting 1938–1943**
  (with the 14 Aug 1939 scenario as a second starting point).
- Use a 6/12-month horizon for rare events. Oversample positives during training. Report Brier scores and balanced accuracy.

## 4. Collection pipeline

```
launch.ps1 (-debug -dump_history, load the June 1936 save / 1939 scenario)
  → console.ps1: observe; gamespeed 5
  → [daemon, every month] tail game.log → build state → (exploration) choose posture → write jev_apply.txt → console: run jev_apply.txt
  → game runs to the end date → save history_dump/ as runs/<run_id>/
dump_parse.py → examples.py → train (CPU) → evaluate.py (compare against majority baseline)
```
Throughput in this VM: ~22 s per game month. 1936.6 → 1943.1 ≈ 80 months ≈ 30 min per game (without inference).
With the daemon sequential, 20–30 Jev decisions per month × 0.1 s (batched) adds ~3 s/month.
**10 games ≈ 5–6 hours ≈ 70k country-months** (after filtering for n_states ≥ 1, roughly 50k states / 250k questions),
which is on the order of the original model's training set of 18k states.

## 5. Training on CPU

- Full fine-tune at bs8 ≈ 25 s/step. 1 epoch over 50k states ≈ 6,000 steps ≈ **42 hours**. Too slow.
- Options, ordered:
  1. **Freeze the lower 18 of 24 layers and train the top 6 + head**: estimated ~3× faster. Try this first
  2. Use a **frozen backbone + cached encodings for forecast questions** and train only the head. The state encoding is computed once and reused. But the span head needs to see the question tokens, so the cache has to include questions. Suitable for fixed question templates
  3. Train on a subset (e.g. 10k states) and check that it beats the majority baseline before scaling up
  4. Rent a GPU (the original training ran for 229 s on an H100)
- Evaluate by game, not by random split: hold out whole games so the same game's months never land on both sides of the split.

## 6. Next steps
1. Mod: add the missing fields to `on_monthly` (stability / war support / strength ratio / strategy values / JEV|ACT)
2. The daemon `jevd.py`: tail + state building + ε-exploration posture assignment + console injection
3. Run 1 full game from the June 1936 save with no intervention → the first batch of forecast data. Run a first training job with partial freezing to see whether it beats the majority baseline
4. Only if it does: turn on randomized intervention and collect action-value data
