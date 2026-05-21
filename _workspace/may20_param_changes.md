# 2026-05-20 Parkour Experiments: Param Change Timeline

Source: `logs/rsl_rl/go2_parkour/2026-05-20_*/params/{env,agent}.yaml`
Method: sequential `diff` between adjacent runs (cosmetic `log_dir`/`run_name`/`usd_path` Isaac-version path differences omitted from semantic columns).

Line→terrain mapping (confirmed from env.yaml): 337=`parkour_flat`, 376=`parkour_hurdle`, 415=`parkour_step`, 452=`parkour_gap`, 490=`parkour_stair`.

## env.yaml Timeline

### 1 → 2 (09-50-37 → 09-50-51)
- Only `usd_path` Isaac 4.5 vs 5.1 (+log_dir). **No semantic change.** [pair]

### 2 → 3 (09-50-51 → 10-29-03)
- `max_tilt: 1.2 → 1.309` (~68.7° → ~75°). Classification: **termination**
- usd_path 5.1→4.5 (Isaac version flip).

### 3 → 4 (10-29-03 → 10-29-06)
- Only usd_path 4.5→5.1. **No semantic change.** [pair]

### 4 → 5 (10-29-06 → 12-27-42, "change_reward")
- `parkour_flat.proportion: 0.1 → 0.2`
- `parkour_gap.proportion: 0.3 → 0.2`
- usd_path 5.1→4.5.
- Classification: **terrain** (NOT reward — despite run name `change_reward`, only terrain proportions changed in env.yaml).

### 5 → 6 (12-27-42 → 12-27-46)
- Only usd_path 4.5→5.1. **No semantic change.** [pair]

### 6 → 7 (12-27-46 → 14-25-07, "no_stair_env")
- **No diff at all** (only log_dir). Despite the name "no_stair_env", env.yaml is identical to run #6 — stair proportion still 0.2.

### 7 → 8 (14-25-07 → 14-43-58, "change_seed")
- `seed: 42 → 100`. Classification: **seed**

### 8 → 9 (14-43-58 → 15-02-27, "no_stair_env")
- `parkour_step.proportion: 0.2 → 0.3`
- `parkour_gap.proportion: 0.2 → 0.3`
- `parkour_stair.proportion: 0.2 → 0.0`  ← actual stair removal here
- `seed: 100 → 42`
- Classification: **terrain + seed**

### 9 → 10 (15-02-27 → 16-07-33)
- `parkour_step.proportion: 0.3 → 0.2`
- `parkour_gap.proportion: 0.3 → 0.2`
- `parkour_stair.proportion: 0.0 → 0.2` (stair reinstated)
- Classification: **terrain**

### 10 → 11 (16-07-33 → 17-14-12, "clip_actions_10")
- `clip_actions: 4.8 → 10.0`. Classification: **action**

### 11 → 12 (17-14-12 → 18-02-12, "goal_idx_visualization")
- Only log_dir. **No semantic env change.**

## agent.yaml Timeline

| Pair | Semantic change |
|------|----------------|
| 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 11→12 | only `run_name` |
| 7→8 | `seed: 42 → 100` |
| 8→9 | `seed: 100 → 42` |
| 9→10 | `load_run: .* → change_termination_bootstraping`, `run_name: '' ` |
| 10→11 | `load_run: change_termination_bootstraping → .*`, run_name back |
| 11→12 | only `run_name` |

No PPO hyperparameter (lr/clip/gamma/num_steps/mini_batches) changed across the 12 runs.

## Core Change Summary

| # | Time | Semantic delta vs prior | Class |
|---|------|------------------------|-------|
| 1 | 09-50-37 | baseline (Isaac 4.5) | — |
| 2 | 09-50-51 | Isaac 5.1 pair of #1 | pair |
| 3 | 10-29-03 | `max_tilt 1.2→1.309` | termination |
| 4 | 10-29-06 | Isaac 5.1 pair of #3 | pair |
| 5 | 12-27-42 | flat 0.1→0.2, gap 0.3→0.2 | terrain (despite "change_reward" name) |
| 6 | 12-27-46 | Isaac 5.1 pair of #5 | pair |
| 7 | 14-25-07 | identical to #6 (mislabeled "no_stair_env") | none |
| 8 | 14-43-58 | `seed 42→100` | seed |
| 9 | 15-02-27 | stair 0.2→0.0, step/gap 0.2→0.3, seed 100→42 | terrain+seed (real no-stair) |
| 10 | 16-07-33 | stair restored 0.2, step/gap back 0.2, `load_run=change_termination_bootstraping` | terrain+resume |
| 11 | 17-14-12 | `clip_actions 4.8→10.0` | action |
| 12 | 18-02-12 | none (viz-only run) | none |

## Unexpected Findings

1. **Run #5 ("change_reward") changes no reward.** env.yaml diff vs #4 shows only `parkour_flat 0.1→0.2` and `parkour_gap 0.3→0.2` (terrain proportions), plus the Isaac path. No `reward_scales` key changed. agent.yaml only changed `run_name`.

2. **Run #7 ("no_stair_env") is identical to #6.** Zero semantic diff. The actual stair-removal occurs in **run #9 (15-02-27)**, where `parkour_stair.proportion: 0.2 → 0.0` while step and gap absorb the mass (0.2→0.3 each).

3. **Pair verification:** All 3 "isaac 4.5 ↔ 5.1 pair" sets (#1/#2, #3/#4, #5/#6) differ ONLY in `usd_path` host (`http://…/4.5/…` vs `https://…/5.1/…`) and log/run name. Same seed (42), same hyperparams, same terrain proportions — true parameter pairs.

4. **Run #10 (16-07-33) is the only resume run.** `load_run: .* → change_termination_bootstraping` and empty `run_name`. Otherwise identical PPO hyperparams.

5. **No reward_scales delta detected anywhere in 12 runs.** No `_get_rewards`-relevant config changed via env.yaml across this day.

6. No file was missing; all 12 runs have both `env.yaml` and `agent.yaml`.
