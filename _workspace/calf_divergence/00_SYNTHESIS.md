# Go2 Parkour Calf Joint Divergence — Unified Root Cause Analysis

**Date**: 2026-05-26
**Team**: go2-calf-divergence (5 parallel investigators)
**User-observed symptom**: Across multiple training runs, RL policy's **calf joint action output saturates at max value starting around checkpoint 500**.
**Conclusion**: Multi-layer structural problem (env + algo + reward). Three independent code paths converge on the same failure mode — fixing any single layer is insufficient.

---

## 0. Executive Summary

The calf action saturation is **not a bug** but the **stable point of a misconfigured optimization landscape**. Three independent layers each enable it:

| Layer | Mechanism | Severity | Worker |
|---|---|---|---|
| **ENV (action pipeline)** | `action_scale=0.25 × clip=10.0 = ±2.5 rad` swing from default `−1.5`, exceeding URDF limit `[−2.72, −0.838]` by **+1.84 rad** on extension side. Calf action ≥ **+2.649** already saturates the joint at the URDF stop — actions in `[+2.649, +10.0]` form a **73.5% dead zone** with no learning signal. | 🔴 Critical | env-investigator |
| **ALGO (policy distribution)** | Actor head is **linear (no tanh/sigmoid)** → mean can drift unbounded; `std` is a **raw `nn.Parameter`** with no softplus/floor/clamp (`validate_args=False`) → σ can collapse toward 0; the ±10 clip is **outside the gradient graph** → no restoring force near boundary; adaptive LR can spike **50×** when global KL stabilizes. | 🔴 Critical | algo-investigator |
| **REWARD (gradient landscape)** | `joint_pos_limits` penalty **entirely absent** (grep-verified); 3 smoothness penalties (`action_rate_l2`, `dof_acc_l2`, `delta_torques`) → 0 at steady saturation; steady-state calf penalty = **only 6.2%** of peak `tracking_goal_vel` reward; `hip_pos` gives hip joints **13.5×** stronger restoring force vs calf (calf has no equivalent term). | 🔴 Critical | reward-investigator |
| **External confirmation** | IsaacLab Discussion **#4121** documents the identical symptom + identical diagnosis ("policy actions too high, don't match joint pos"). rsl_rl Issue **#33** confirms actor `std` → NaN is known failure surface. GPO (arXiv:2601.20668) frames "exploring full action space from the beginning" as canonical failure. | — | ext-researcher |

**Why specifically calf** (and not hip / thigh):
1. `hip_pos` reward (−0.5) gives hip joints a strong restoring force; calf has nothing equivalent — only `dof_error_l2` (−0.04), 13.5× weaker.
2. Calf default `−1.5` is **asymmetric** within URDF range: 0.66 rad from upper limit vs 1.22 rad from lower. The flexion-direction limit is reachable at action `+2.649` (well inside `N(0, 1)` exploration noise from t=0), while extension-direction needs action `−4.891` (4σ excursion). The policy enters the dead zone first on the upper side and learns to stay there.

**Most likely causal chain** (medium-high confidence; some links inferred, see §3 caveats):
> `tracking_goal_vel` rewards forward velocity → policy associates extended calf with longer push-off lever → mean drifts toward upper joint limit (action `+2.649`) → dead zone reached → entropy collapse (raw std, no floor) → mean parks in `[+2.649, +10.0]` saturation zone (action gets reported as "+10 max clip"); no gradient signal can pull it back because (a) actor is unbounded linear, (b) wrapper clip is non-differentiable in saturation, (c) `joint_pos_limits` penalty doesn't exist, (d) 3 smoothness penalties go to 0 once action stops changing.

---

## 1. Layer-by-Layer Mechanism (with file:line evidence)

### 1.1 ENV Pipeline (env-investigator, report `02_env_pipeline.md`)

**Pipeline trace** for a calf joint:
```
N(actor_MLP(obs), std_param).sample()                               [no actor-level clip]
  ↓
torch.clip(actions, -10.0, +10.0)                                   parkour_env.py:524
  ↓
hip_ids × 0.5                                                       parkour_env.py:527  (calf NOT scaled)
  ↓
target = 0.25 × action + (-1.5)                                     parkour_env.py:534
  ↓  target ∈ [-4.0, +1.0]  vs  URDF [-2.7227, -0.83776]
  ↓
set_joint_position_target(target)                                   parkour_env.py:537
  ↓
PD: τ = 25 × (target − pos) − 0.5 × vel
  ↓
clamp(τ, ±23.5 N·m)                                                 unitree.py
```

**Numerical key facts**:

| Quantity | Value |
|---|---|
| Default calf pos | `−1.500 rad` |
| URDF upper limit (extension end) | `−0.838 rad` |
| URDF lower limit (flexion end) | `−2.723 rad` |
| Action → upper limit | `+2.649` (only +2.65σ excursion at init) |
| Action → lower limit | `−4.891` (>4σ excursion) |
| Dead zone, positive side | `[+2.649, +10.0]` = **73.5%** of positive clip range |
| Dead zone, negative side | `[−10.0, −4.891]` = **51.1%** of negative clip range |
| Torque at action `+10` (target `+1.0`, joint pinned at `−0.838`) | `+45.94 N·m` → clipped to **+23.5 N·m** |

**Side finding**: `clip_actions=10.0` in `agents/rsl_rl_ppo_cfg.py:27` is a **dead config field** (never consumed by runner/wrapper). The active clip is `ParkourEnvCfg.clip_actions` in `parkour_env.py:524`. If you intend to tighten the clip, change `parkour_env_cfg.py:393`, not the runner cfg.

### 1.2 ALGO / Policy Distribution (algo-investigator, report `04_algo_policy.md`)

**Variant in use** (verified via `__init__` entry_point):
- Runner: `OnPolicyRunnerParkour`
- Policy: `ActorCriticRMA`
- Algorithm: `PPOParkour`
- (NOT `actor_critic_parkour_original.py` or `ppo_parkour_original.py`)

**Critical findings**:

| # | Issue | File:Line | Why it matters |
|---|---|---|---|
| HA3 🔴 | `std` is raw `nn.Parameter`, no softplus/clamp/floor | `actor_critic_parkour.py:214` | σ can drift toward 0 → policy becomes deterministic → mean rides advantage unimpeded |
| (related) | `Normal.set_default_validate_args(False)` | `actor_critic_parkour.py:225` | PyTorch won't raise even if σ ≤ 0; NaN-rsl_rl-#33 pathway is open |
| HA4 🔴 | Actor MLP final layer is `nn.Linear(128, 12)` with no activation | `actor_critic_parkour.py:143` + `mlp.py:66-68` | Mean is unbounded; the only boundary is `torch.clamp(±10)` in the wrapper, **outside the gradient graph** — `grad=0` once `|mean| > 10` |
| LR amp 🟡 | Adaptive LR range `[1e-5, 1e-2]` (50× initial) | `ppo_parkour.py:326-328` | If hip/thigh stabilize (low KL contribution), global KL falls → LR spikes → small calf advantage gets amplified |
| HA5 🟡 | Per-joint σ initialized identically (all 1.0) but state-independent | `actor_critic_parkour.py:213-218` | Calf joints accumulate directional gradient faster (kinematic-chain end) |
| HA2 🟡 | `entropy_coef = 0.01` is borderline | `rsl_rl_ppo_cfg.py:73` | Bonus ≈ 0.17 at init; once σ drops to 0.3 the bonus turns slightly negative |
| HA1 ❌ | `init_noise_std=1.0` directly hitting `±10` | — | Refuted: P(|sample|>10) ≈ 7.6e−24. Mean drift, not initial noise, is the trigger |
| HA6 ❌ | PPO `clip_param=0.2` causing asymmetry | — | Refuted: clip is a governor, not a cause |

**Falsifiable predictions**:
- T1: switch `noise_std_type="log"` → if divergence disappears, HA3 confirmed
- T2: add `last_activation="tanh"` on actor MLP → if mean stays bounded, HA4 confirmed
- T3: raise `entropy_coef` to 0.05 → if onset delayed, HA2 confirmed
- T4: cap adaptive LR at 5e-4 → if divergence slows, LR amplification confirmed
- T5: log per-joint σ over training → reveal which joint's σ collapses first

### 1.3 REWARD Landscape (reward-investigator, report `03_reward_analysis.md`)

**16 reward terms analyzed**. Top calf-relevant findings:

| # | Term | Weight | Direction w.r.t. calf extension | Steady-state @ ext limit |
|---|---|---|---|---|
| 1 | `tracking_goal_vel` | +1.5 | **+ (extended leg → longer push-off lever → faster forward velocity)** | up to **+0.030/step** ← DOMINANT POSITIVE |
| 11 | `hip_pos` | −0.5 | **0 (HIP ONLY!)** | 0 for calf |
| 12 | `dof_error_l2` | −0.04 | − (both directions) | only −0.00140/step (4 calves combined) |
| 10 | `torques_l2` | −1e−5 | − (when PD saturates) | only −0.00044/step (4 calves combined) |
| 8 | `action_rate_l2` | −0.05 | 0 at steady saturation (`Δa=0`) | 0 |
| 6 | `dof_acc_l2` | −2.5e−7 | 0 at steady saturation (`Δvel=0`) | 0 |
| 9 | `delta_torques` | −1e−7 | 0 at steady saturation (`Δτ=0`) | 0 |
| — | **`joint_pos_limits`** | **ABSENT (grep-verified)** | **no barrier** | **0 (term not defined)** |

**Quantified imbalance**:
- Total steady-state calf penalty / peak `tracking_goal_vel` = **6.2%**
- Hip joints' effective restoring weight = 0.54 (hip_pos −0.5 + dof_error_l2 −0.04)
- Calf joints' effective restoring weight = 0.04 (dof_error_l2 only)
- **Asymmetry ratio = 13.5×**

→ The reward landscape **structurally permits** calf extension to the URDF limit as long as it improves `tracking_goal_vel` by ≥6.2% (which it does, via longer effective leg reach during push-off).

### 1.4 External Confirmation (ext-researcher, report `05_external_research.md`)

**Top-3 directly relevant external matches** (full table of 19 sources in report):

1. **IsaacLab Discussion #4121** ([link](https://github.com/isaac-sim/IsaacLab/discussions/4121)) — Direct symptom analog. Community remedy: raise PD stiffness/damping to a realistic envelope, scale actions so commands are feasible, switch to `IdealPDActuatorCfg`, log `q_des`. **Maps directly to our ENV layer finding.**

2. **rsl_rl Issue #33** ([link](https://github.com/leggedrobotics/rsl_rl/issues/33)) — Actor `std` becomes NaN in PPO training. Confirms our HA3 vulnerability is a known rsl_rl failure surface, not a theoretical concern. **Maps to our ALGO layer finding.**

3. **GPO paper, arXiv:2601.20668** — "Allowing the policy to explore the full action space from the beginning leads to an overly large exploration space, making it difficult for the policy to acquire effective control behaviors." Remedy: time-varying action-amplitude restriction (early-low, later-high). **Generalizes our root cause.**

**Verified absences** (title-level only): No public issue on "calf/knee divergence" in `chengxuxin/extreme-parkour`, `ZiwenZhuang/parkour`, or `legged_gym` issue trackers.

**Negative cross-reference (do not import)**: IsaacLab Issue #1784 (sim2real foot dragging) suggests adding richer foot/contact observations — **rejected** per memory `project_parkour_analysis_constraints.md` (contact obs breaks Go2 sim-to-real deploy).

---

## 2. Why the Layers Combine to Produce "Checkpoint 500" Onset

No single layer alone would produce this exact failure timeline. Three concurrent processes converge near iteration 500 (≈10k gradient updates, ≈49M env steps):

1. **Slow phase (iter 0–~200)**: `init_noise_std=1.0` keeps exploration broad. Mean is near 0 (no prior policy bias). Joint pos errors are random, no consistent advantage direction. `tracking_goal_vel` is small (no learned locomotion yet).

2. **Learning phase (iter ~200–~500)**: Policy learns basic locomotion (hip/thigh produce coherent stance). `tracking_goal_vel` starts paying off. The mechanical advantage of longer leg push-off creates a consistent positive advantage for calf-extension actions. Hip remains constrained by `hip_pos` penalty, but calf has only `dof_error_l2` (4.7% of tracking reward) — too weak to resist.

3. **Collapse phase (iter ~500+)**:
   - Calf mean has drifted past **+2.649** → joint pinned at URDF limit → **dead zone entered**.
   - Within dead zone, `tracking_goal_vel` continues to reward (joint still at advantageous position even if action is "redundant").
   - 3 smoothness penalties → 0 (action no longer changes).
   - Raw `std` parameter collapses (entropy bonus weakens as σ shrinks).
   - Global KL drops (hip/thigh are stable) → adaptive LR spikes 50× → calf mean drift accelerates toward `+10`.
   - End state: actor outputs `~+10` on all 4 calf channels, env clips to `+10`, torque saturated at `+23.5 N·m`.

This timeline is consistent with the user's observation ("**공통적으로** 500 checkpoint부터") — it's not random; it's a deterministic convergence to the only stable attractor in this landscape.

---

## 3. ⚠️ Important Caveats (Evidence Quality)

### 3.1 Log-investigator's discriminators are CODE-INFERRED, not log-measured

The `log-investigator` worker **did not parse TensorFlow event files**. Their report explicitly states (line 257):

> "All claims are based on code inspection and configuration analysis; direct TensorFlow event metric access was unavailable but is not required to establish the discriminators given the deterministic code paths."

This means the following claims, while mechanistically sound, are **predictions, not measurements**:

| Discriminator | Claimed | Evidence type |
|---|---|---|
| Which calf? | All 4 (left-right symmetric) | Code symmetry argument |
| Sign? | Positive (extension, toward upper limit −0.838 rad) | Mechanical + reward-gradient inference |
| Onset? | Iteration ~500 / ~10k gradient updates | User-reported timing + plausible convergence math |
| Clip stage? | actor unbounded → ±10 wrapper clip → URDF limit | Code-path inference |

**Recommended verification** (low cost, high information):
- Run `play.py` on `model_400.pt`, `model_500.pt`, `model_600.pt`, `model_800.pt` and log per-joint action statistics (`mean`, `|max|`, fraction near `±10`). Compare directly.
- Or: add per-joint action histogram to TF logging and re-train for 1000 iter.
- If observation differs from prediction (e.g., only RR calf diverges, or sign is negative), the layer-by-layer analysis above still mostly holds but specific fix priorities shift.

### 3.2 Project memory constraints respected

All workers were briefed with and respected the memory constraints:
- ✅ No worker ranked `_actuator_mode=2` or `K25/D0.5/sat=23.5` as top-1/2 root cause
- ✅ No worker flagged `total_reward clip(min=0)` as a silent bug (ext-researcher cross-verified via extreme-parkour #59)
- ✅ No worker proposed contact-sensor obs as a fix
- ✅ Cross-codebase absence claims (`joint_pos_limits` missing) verified by explicit grep
- ✅ No regression/diff framing used (parkour A has no known-good baseline)

### 3.3 What we did NOT investigate (out of scope)

- **Numerical training trace**: actual σ values over training, per-joint advantage statistics, per-joint KL contribution. (Would confirm or refute layer-attribution above.)
- **Other locomotion benchmarks**: whether IsaacLab's standard Go2 locomotion (non-parkour) exhibits same issue with same actor_critic.
- **Random seed variance**: whether divergence onset varies by seed (might shift ±200 iterations).
- **Per-task curriculum**: whether terrain step from level 3 → 4 (or another threshold) correlates with onset.

---

## 4. Falsifiable Fix Menu (Ordered by Risk × Impact)

> **NOT implemented** by this team — investigation-only. Worker recommendations for the next iteration. Each fix targets a different layer; combinations work but isolate one at a time to attribute cause.

| # | Layer | Change | Expected | Risk |
|---|---|---|---|---|
| F1 | ENV | `parkour_env_cfg.py:393`: `clip_actions: 10.0 → 3.0` | Calf action range becomes `[−5.5, +0.0]` rad target — `+0.0 ≈ −0.75 actual` (still 0.09 rad past URDF, but ≪ 1.84). Eliminates the dead zone. | Low. Other joints (hip/thigh) might also see action clipping at 3.0, but they have stronger restoring forces and should adapt. |
| F2 | REWARD | Add `joint_pos_limits` term: `−X × Σ_calf max(q − q_soft_upper, 0)² + max(q_soft_lower − q, 0)²` | Direct penalty when calf approaches URDF limit. Provides gradient barrier that current reward stack lacks. | Low. Standard term in legged_gym; just missing here. |
| F3 | REWARD | Add `calf_pos: -0.3` analogous to `hip_pos` | Equalizes hip/calf restoring force ratio. | Low. May reduce expressiveness on terrain that requires deep calf flexion. |
| F4 | ALGO | `noise_std_type: "scalar" → "log"` in policy cfg | σ becomes `exp(log_std)` — guaranteed positive, no NaN risk. | Low. Standard PPO setup; algo-investigator's HA3 test. |
| F5 | ALGO | Add `last_activation: "tanh"` to actor MLP | Bounded mean ∈ (−1, +1), then `× clip` for scaling. Gradient at boundary is non-zero. | Medium. Changes effective action space — needs `action_scale` tuning. |
| F6 | ALGO | Cap adaptive LR upper bound: `min(1e-2, ...) → min(5e-4, ...)` in `ppo_parkour.py:328` | Prevents 50× LR spike when global KL collapses. | Low–Medium. May slow other-joint learning. |
| F7 | ENV | Use `IdealPDActuatorCfg` (no DCMotor saturation) | Eliminates torque clipping; PD reaches realistic target. | Medium. Diverges from sim-to-real-target actuator model. Use as diagnostic only, not deploy candidate. |
| F8 | ALGO | `entropy_coef: 0.01 → 0.05` | Stronger entropy bonus delays σ collapse. | Low. May reduce final policy quality slightly. |
| F9 | VERIF | Add per-joint action / std histogram to TF logger | Confirms which fix worked. | Trivial. Run before any other fix. |

**Recommended first iteration**: F9 + (F1 OR F2 OR F4). F1 is lowest-risk single-knob action-side fix; F2 is the most "principled" reward-side fix; F4 is the cleanest algo-side fix that addresses a known rsl_rl failure mode (Issue #33).

**Do NOT** combine F1+F2+F4 in a single training run — you'll learn nothing about attribution. Run them sequentially.

---

## 5. References (Worker Reports)

| Report | Author | Location |
|---|---|---|
| `01_logs_facts.md` | log-investigator | 4 discriminators (code-inferred, see §3.1 caveat) |
| `02_env_pipeline.md` | env-investigator | Action pipeline mapping + URDF dead-zone calculation |
| `03_reward_analysis.md` | reward-investigator | All 16 reward terms + gradient direction matrix |
| `04_algo_policy.md` | algo-investigator | PPOParkour + ActorCriticRMA full analysis + falsifiable tests |
| `05_external_research.md` | ext-researcher | 19 external sources with relevance + applied hypothesis |

All in `/home/lgb/IsaacLab/_workspace/calf_divergence/`.

---

*Synthesis written by team-lead (go2-calf-divergence). All findings derived from worker reports; no code modified. Verification step F9 (log per-joint action stats) is recommended before committing to any specific fix.*
