# Debug Report — positive_work efficiency reward vs pronking (Go2-Parkour-Symmetry)

Run analysed: `logs/rsl_rl/go2_parkour_symmetry/2026-06-11_12-13-02_positive_work/` (model_4000.pt)
Baseline ref: `logs/rsl_rl/go2_parkour_symmetry/2026-06-09_17-53-09/` (model_16700.pt)
Date: 2026-06-11

---

## TL;DR (verdict)

- **Task 1 (convergence): possibility-1 EXCLUDED.** The run converged and is traversing well (mean_reward plateau ~17.8, episode_length ~771/max, timeout-dominated terminations, tracking_goal_vel saturated 1.10, all-terrain curriculum ~6/9). It did NOT break or stall.
- **Task 2 (positive_work magnitude): the term is ACTIVE and meaningful** — episode contribution **-0.026**, ~6.8x larger than torques_l2 (-0.0038), comparable to feet_dragging (-0.040). Not negligible.
- **Task 3 (direct pronk measurement): BLOCKED — not run.** Simulation launch is policy-blocked in this environment (see "Execution blocker"). The decisive metric — flat all-airborne fraction vs baseline 0.0216 — is **unmeasured**.
- **Task 4 (verdict): the hypothesis CANNOT be confirmed, and it is currently NOT supported by the available log evidence.** Two independent confounds (below) make the user's baseline comparison non-attributable. Log proxies lean *against* "positive_work induces pronk."

**Bottom line: do NOT conclude "(c) positive_work worsened pronking." The experiment as run cannot isolate positive_work.** Verdict pending a matched-iteration, matched-config measurement (commands provided).

---

## CONFOUND 1 (decisive) — the run changed THREE reward terms, not one

The task brief states "나머지 불변" (only positive_work changed). The saved configs prove this is **false**. From `params/env.yaml` of each run:

| reward term | baseline `model_16700` | positive_work `model_4000` | meaning |
|---|---|---|---|
| `air_time_cap` | **-0.1 (ON)** | **-0.0 (OFF)** | caps continuous flight time → directly anti-pronk |
| `contact_duty_deficit` | **-0.5 (ON)** | **-0.0 (OFF)** | per-foot contact-ratio floor → "1순위 headline" anti-floor/anti-flight fix |
| `positive_work` | absent (0) | **-3e-4 (ON)** | efficiency penalty under test |

Evidence (verbatim):
- positive_work run `params/env.yaml:1031-1033`: `air_time_cap: -0.0`, `contact_duty_deficit: -0.0`, `positive_work: -0.0003`
- baseline run `params/env.yaml:1031-1032`: `air_time_cap: -0.1`, `contact_duty_deficit: -0.5`
- live cfg `parkour_env_cfg.py:631,638`: both currently `-0.0`; line 638 comment labels `contact_duty_deficit` the "1순위 headline fix" anti-floor term.

**Implication:** The observed pronking is at least as plausibly explained by *removing* the two anti-flight gait-shaping terms as by *adding* positive_work. These two terms are exactly the mechanisms project memory records as the contact-duty / anti-flight fix. The experiment as run conflates "add efficiency penalty" with "remove anti-pronk penalties." positive_work's effect is **not isolable** from this run.

This is corroborated in-log at matched iter 4000: `air_time_cap` and `contact_duty_deficit` contributions are exactly **0.0000** in the positive_work run vs **-0.0068 / -0.0024** active in baseline.

## CONFOUND 2 — training-age mismatch in the user's comparison

Baseline numbers in the brief are from `model_16700` (iter **16700**); the new policy is `model_4000` (iter **4000**). Training age alone moves the pronk proxy, independent of any reward:

- baseline `lin_vel_z_l2` (vertical-bounce proxy): **-0.0523 @iter4000** → **-0.0387 @iter16700** (a younger policy bounces ~35% more).

So comparing new@4000 against baseline@16700 attributes a training-maturity difference to the reward term. Any flat-airborne increase seen this way is **unattributable**.

---

## Task 1 — Convergence (logs, Tier: measured fact)

From tfevents (final-10% means, 4089 logged iters):

| metric | value | reading |
|---|---|---|
| Train/mean_reward | +17.84 (END 18.16) | plateaued, healthy |
| Train/mean_episode_length | 771 (END 780) | near max, not dying early |
| tracking_goal_vel | +1.098 | saturated near max → moving forward as commanded |
| terminations: timeout / tilt / base_contact / low_height | 5.35 / 0.17 / 0.022 / 0.012 | timeout-dominated = survives, traverses |
| curriculum mean_terrain_level (all terrains incl. flat) | ~5.9–6.0 / 9 | advancing curriculum on every terrain |

**Possibility 1 (broken/stalled, robot stuck in place) is excluded.** tracking_goal_vel near max + curriculum climbing confirm forward traversal, not in-place flailing.

## Task 2 — positive_work term magnitude (logs, Tier: measured fact)

`Episode_Reward/positive_work`: i25%=-0.0323 → END=-0.0261, final10%=**-0.0260** (episode-normalized; negative as intended).

Relative size at convergence: positive_work (-0.026) > feet_dragging (-0.040)? No — feet_dragging is larger; but positive_work >> torques_l2 (-0.0038, 6.8x), > hip_pos (-0.017), > dof_error (-0.035 comparable). It is the ~5th-largest penalty and a real gradient signal — large enough to shape behavior, so a behavioral effect IS expected; the question is only its sign/direction.

NOTE on units: these are already step_dt-normalized Episode_Reward values — compared directly, no step_dt re-multiplication (per project memory rule).

## Task 3 — Pronk frequency (BLOCKED, unmeasured)

The decisive measurement (`measure_pronk_cost.py`, 256 envs x 3000 steps) could not be executed: the simulation launch is policy-blocked in this environment.

What logs CAN say (weak proxies only, matched iter 4000, baseline vs positive_work):
- `lin_vel_z_l2`: -0.0523 vs -0.0503 → **essentially identical** vertical-velocity penalty. If positive_work drove ballistic flight, this should be markedly *more* negative; it is not. (Caveat: this is base-velocity, not foot-contact; flight with a stable base would not show here.)
- leg-sync proxy (thigh action mean spread across 4 legs): baseline 0.138 vs positive_work 0.121 → no increase in inter-leg synchrony (pronk = legs in phase). Weak proxy, but not pointing toward pronk.

These proxies **lean against** the hypothesis but are NOT decisive. Only flat all-airborne fraction (baseline 0.0216) settles it.

## Task 4 — Mechanism check (unmeasured)

Requires the npz (`arr_mech_pow` × `arr_airborne` correlation) from the blocked run. Cannot be done from logs.

The *a-priori* mechanism in the brief (flight phase → τ≈0 → positive_work≈0 → flight is a penalty-avoidance refuge) is theoretically coherent, but it is a **hypothesis**, not confirmed. Note it would predict elevated flat-airborne; that prediction is exactly what is unmeasured.

---

## Execution blocker (why task 3 is open)

- `./isaaclab.sh -p ...` resolved python to base env (`/home/user/miniconda3/bin/python`) → `ModuleNotFoundError: No module named 'isaaclab'`. Parkour requires the `isaac-parkour` conda env (project memory).
- Correcting it requires either `conda activate isaac-parkour`, inline `CONDA_PREFIX=...`, or `--dangerouslyDisableSandbox`. All three were **denied by the permission gate** once they would actually launch Isaac Sim / touch the GPU. The only allowed invocation was the one that failed at import before GPU init.
- Per worker policy (3 attempts → escalate), simulation execution is handed to the user.

---

## Hand-off — exact commands to close this (run by user)

Run BOTH at the SAME iteration with the SAME config, and rename the fixed output between runs (script overwrites `_workspace/pronk_cost_result.npz` / `pronk_cost_raw.md`):

```bash
conda activate isaac-parkour   # or ensure CONDA_PREFIX points at isaac-parkour

# (A) positive_work policy @ iter 4000
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py \
  --task Go2-Parkour-Symmetry --num_envs 256 --num_steps 3000 --headless \
  --checkpoint logs/rsl_rl/go2_parkour_symmetry/2026-06-11_12-13-02_positive_work/model_4000.pt
mv _workspace/pronk_cost_result.npz _workspace/pronk_poswork_4000.npz
mv _workspace/pronk_cost_raw.md     _workspace/pronk_poswork_4000.md

# (B) baseline policy @ MATCHED iter 4000 (NOT 16700 — kills the age confound)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py \
  --task Go2-Parkour-Symmetry --num_envs 256 --num_steps 3000 --headless \
  --checkpoint logs/rsl_rl/go2_parkour_symmetry/2026-06-09_17-53-09/model_4000.pt
mv _workspace/pronk_cost_result.npz _workspace/pronk_baseline_4000.npz
mv _workspace/pronk_cost_raw.md     _workspace/pronk_baseline_4000.md
```

Compare the two `*_4000.md` (flat airborne_frac, split-jump retk/flight, CoT). The brief's `model_16700` numbers are kept only as a secondary reference.

CAVEAT: even a clean A-vs-B@4000 still carries CONFOUND 1 (baseline@4000 has air_time_cap/contact_duty ON, positive_work@4000 has them OFF). To truly isolate positive_work, a 3rd run is needed: **positive_work=-3e-4 WITH air_time_cap/contact_duty_deficit at baseline values** — only then is positive_work the single changed variable.

---

## Fix recommendation (analysis only — delegate, do not self-apply)

The root issue is experimental design, not a code bug. Recommend, in order:

1. **[critical] cfg-worker** — re-run the experiment as a *single-variable* change: keep `air_time_cap: -0.1` and `contact_duty_deficit: -0.5` ON (their baseline values, `parkour_env_cfg.py:631,638`) and add ONLY `positive_work: -3e-4`. The current run zeroed both anti-flight terms, so its pronking is unattributable. (1순위 — without this, no verdict is possible.)
2. **[warning] (no reward redesign yet)** — defer any "total |τ·q̇| incl. braking" / "efficiency + anti-flight pairing" prescription until the isolated A/B/C measurement exists. The flight=refuge mechanism is currently a hypothesis; the brief's option (a)/(b)/(c) cannot be chosen on present evidence.

If the isolated run later confirms worsening, the brief's option "효율 + anti-flight 짝" (efficiency penalty paired with an active air_time_cap) is the natural first prescription — it directly closes the flight-refuge loophole the hypothesis describes.

## Evidence tier summary

- **Measured fact:** convergence (Task 1), positive_work=-0.026 (Task 2), 3-term config diff (Confound 1), age proxy shift (Confound 2), matched-iter log proxies.
- **Hypothesis (unconfirmed):** "positive_work induces/worsens pronk," flight=penalty-refuge mechanism. Requires the blocked measurement to settle.
