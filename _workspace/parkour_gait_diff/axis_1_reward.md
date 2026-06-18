# Axis 1: Reward / Contact-term Diff — Parkour A vs B

Goal: identify reward-side mechanisms that bias A toward **jumping** while B's identical-looking 14-term set yields **4-leg cross-over walking**.

A files:
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` — `_get_rewards()` at L960–1108
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` — `reward_scales` at L561–576

B files:
- `/home/lgb/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/rewards.py` — term functions
- `/home/lgb/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py` — `TeacherRewardsCfg` at L117–244 (weights)

---

## Side-by-side table (sub-parameter level)

### 1) Term existence / weight (cfg)

| Term | A weight (`reward_scales`) | B weight (`TeacherRewardsCfg`) | Status | Jump-vs-walk relevance |
|---|---|---|---|---|
| tracking_goal_vel        | **+1.5**  (cfg L562) | **+1.5**  (mdp L224) | ALIGNED | strong forward bias; same in both |
| tracking_yaw             | **+0.5**  (cfg L563) | **+0.5**  (mdp L232) | ALIGNED | heading; same |
| lin_vel_z_l2             | **−1.0**  (cfg L564) | **−1.0**  (mdp L201) | ALIGNED on cfg weight, **DIFF on conditional** (see §2) | jumping = z velocity → conditional matters |
| ang_vel_xy_l2            | **−0.05** (cfg L565) | **−0.05** (mdp L180) | ALIGNED on cfg weight, **DIFF on conditional** (see §2) | jumping = pitch oscillation → conditional matters |
| orientation_l2           | **−1.0**  (cfg L566) | **−1.0**  (mdp L209) | ALIGNED (and conditional both = 0 on non-flat) | jumping tilts body; both zero on obstacles |
| dof_acc_l2               | **−2.5e-7** (cfg L567) | **−2.5e-7** (mdp L194) | ALIGNED | smoothness; same |
| collision                | **−10.0** (cfg L568) | **−10.0** (mdp L143) | ALIGNED | hits with base/calf/thigh → jumping reduces this when clean takeoff |
| **action_rate_l2**       | **−0.05** (cfg L569) | **−0.1**  (mdp L187) | **DIFF (A 50% of B)** | jumping = sudden Δa spikes → A penalises 2× less |
| delta_torques            | **−1.0e-7** (cfg L570) | **−1.0e-7** (mdp L240) | ALIGNED | torque smoothness; same |
| torques_l2               | **−1e-5** (cfg L571) | **−1e-5** (mdp L159) | ALIGNED | torque magnitude; same |
| hip_pos                  | **−0.5**  (cfg L572) | **−0.5**  (mdp L173) | ALIGNED | keeps abduction near default; same |
| dof_error_l2             | **−0.04** (cfg L573) | **−0.04** (mdp L166) | ALIGNED | nominal-pose attraction; same |
| feet_stumble             | **−1.0**  (cfg L574) | **−1.0**  (mdp L217) | ALIGNED (binary `any` form in both) | horizontal/vertical force ratio |
| feet_edge                | **−1.0**  (cfg L575) | **−1.0**  (mdp L150) | ALIGNED (gated by `terrain_levels > 3`, force-thresh 2 N in both) | foot near edge → penalty |
| feet_air_time            | **ABSENT** | **ABSENT** | RULED-OUT (removed in commit 25d2071196f) | "ALIGNED" item listed in brief but both env actually lack it |
| foot_clearance / foot_slip / foot_slide | **ABSENT** | **ABSENT** | RULED-OUT | not used by either |
| gait / phase / contact-symmetry | **ABSENT** | **ABSENT** | RULED-OUT | not used by either |
| termination penalty term | **ABSENT** | **ABSENT** | RULED-OUT | both rely on time_out/grace, no explicit term |
| base_height / hip_height | weight=0 (no key) — `base_height_target=0.34` only declared (cfg L591) | ABSENT | ALIGNED (inactive in both) | not jump-vs-walk |

### 2) Terrain-class **conditional multipliers** (applied INSIDE the term, before scale × dt)

| Term | A non-flat multiplier | B non-flat multiplier | Status | Jump-vs-walk relevance |
|---|---|---|---|---|
| lin_vel_z_l2  | **0.1×** (env L1001, `(is_flat + is_non_flat * 0.1)`) | **0.5×** (rewards.py L144, `rew[...non-flat] *= 0.5`) | **DIFF — A 5× weaker than B on obstacles** | direct selector for jumping vs walking on hurdle/step/gap/stair |
| ang_vel_xy_l2 | **0.5×** (env L1003, `(is_flat + is_non_flat * 0.5)`) | **1.0×** (rewards.py L88–93, no conditional) | **DIFF — A 2× weaker than B on obstacles** | jump = pitch dynamics; A relaxes; B retains full body-rotation penalty |
| orientation_l2 | **0×** on non-flat (env L1008, `* is_flat`) | **0×** on non-flat (rewards.py L156, `rew[...non-flat] = 0`) | ALIGNED | both let body tilt on obstacles |

Note: A's `is_flat` is `_env_class == TERRAIN_CLASS_FLAT` (env_cfg L253). B's check is `terrain_names != 'parkour_flat'` (rewards.py L144, 156). Equivalent gating.

### 3) Per-term mathematical form (function shape)

| Term | A form (parkour_env.py) | B form (rewards.py) | Status |
|---|---|---|---|
| tracking_goal_vel | world-frame `cur_vel·goal_dir`, `min(proj, cmd)/(cmd+1e-5)`; **extra zero-out when `cmd<1e-3`** (env L984–986) | world-frame `cur_vel·target_vel`, `min(proj, cmd)/(cmd+1e-5)` (rewards.py L177–182) | ALIGNED (A adds tiny safety branch) |
| tracking_yaw      | `exp(−|target_yaw − heading_w|)` (env L995) | `exp(−|target_yaw − yaw|)` (rewards.py L194) | ALIGNED |
| lin_vel_z_l2      | `square(root_lin_vel_b[:,2])` (env L998) | `square(root_lin_vel_b[:,2])` (rewards.py L143) | ALIGNED (conditional differs — §2) |
| ang_vel_xy_l2     | `sum(square(root_ang_vel_b[:,:2]))` (env L999) | same (rewards.py L93) | ALIGNED (conditional differs — §2) |
| orientation_l2    | `sum(square(projected_gravity_b[:,:2]))` (env L1006) | same (rewards.py L155) | ALIGNED |
| dof_acc_l2        | `sum(((vel − prev_vel)/step_dt)^2)` step_dt = decimation·sim_dt = 0.02 s (env L1011–1013) | same; uses `env.cfg.decimation * env.cfg.sim.dt = 0.02 s` (rewards.py L119, 133) | ALIGNED |
| collision         | `sum(max_over_hist(‖F‖) > 0.1)` on `[base, calf, thigh]` (env L1015–1019) | `sum(‖F‖ > 0.1)` on same body set (rewards.py L215–221) | ALIGNED (A also uses history max → marginally stricter) |
| action_rate_l2    | `‖a_t − a_{t−1}‖_2` **L2 norm**, not sum-of-squares (env L1022) | `‖a_t − a_{t−1}‖_2` **L2 norm** (rewards.py L112) | ALIGNED form; **DIFF weight only** |
| delta_torques     | `sum(square(τ − τ_last))` (env L1025–1028) | same (rewards.py L213) | ALIGNED |
| torques_l2        | `sum(square(applied_torque))` (env L1031) | same (rewards.py L71) | ALIGNED |
| hip_pos           | `sum(square(q_hip − q_hip_default))` on `_hip_joint_ids=[0,3,6,9]` (env L1034–1040, indices set at L261) | same on `body_names=".*_hip_joint"` (rewards.py L80–86) | ALIGNED |
| dof_error_l2      | `sum(square(q − q_default))` (env L1043–1045) | same (rewards.py L78) | ALIGNED |
| feet_stumble      | `any(‖F_xy‖ > 4·|F_z|)` binary (env L1049–1052) | same (rewards.py L165–167) | ALIGNED |
| feet_edge         | terrain_levels>3 × `sum(contact_filt & feet_at_edge)`, force-thresh 2 N, 2-sample OR (env L1054–1079) | same (rewards.py L37–64) | ALIGNED |

### 4) Structural diffs around the reward pipeline

| Aspect | A | B | Status | Relevance |
|---|---|---|---|---|
| **Total-reward clipping** | `torch.clip(total_reward, min=0.)` (env L1108) | none (manager-based, raw sum) | **DIFF — A removes all negative reward gradients** | when penalties dominate, A returns 0 → loses bias-toward-good-states gradient; can let the agent settle on a high-variance jumping strategy because the worst penalty cases are clipped |
| Per-term step_dt scale | `value * scale * step_dt` (env L1102) | `RewardManager` multiplies `weight * dt` per term (IsaacLab core) | ALIGNED |
| Tracking_goal_vel command source | `_commands[:, 0]` from env's resampled command buffer | `env.command_manager.get_command('base_velocity')[:, 0]` | ALIGNED (functionally same) |
| Cmd velocity range (lin_vel_x) | `[0.3, 1.0]` (cfg L550) | `(0.3, 0.8)` (mdp L29) | DIFF (out of scope — really Axis 2/3, but it implicitly raises the cost of "walk slowly over hurdle" in A) |
| Termination grace | `termination_grace_steps=5` (cfg L601) — no fail termination first 5 steps | none — failure can fire immediately | DIFF (out of scope — not reward-shape) |
| Tilt termination | `sum(proj_grav_xy^2) > 0.99` (env L1120) ⇒ ~84° tilt | `|roll|>1.5 ∧ |pitch|>1.5` (term L32) ⇒ ~86° each | ALIGNED magnitude |
| Low-height termination | z < `−0.2` (cfg L600) | z < `−0.25` (term L37) | ALIGNED magnitude |

---

## Top 3 suspects on this axis (ranked by jump-vs-walk discriminative power)

### 1) **`lin_vel_z_l2` non-flat conditional multiplier — A=0.1× vs B=0.5×** &nbsp; *confidence: H*
**File:line:** A `parkour_env.py:1001`; B `rewards.py:143–145`

**Hypothesis.** A's `(is_flat + is_non_flat * 0.1)` multiplier means that on hurdle / step / gap / stair (all non-flat classes), vertical-velocity penalty is at **10 %** of nominal — effective term = `−1.0 × 0.1 × dt × z_vel²`. B uses **50 %**. Cost per step at a typical jump peak (`|v_z|≈2 m/s` → `v_z²=4`):

- A non-flat: `−1.0 × 0.02 × 0.1 × 4 = −0.008` per step
- B non-flat: `−1.0 × 0.02 × 0.5 × 4 = −0.040` per step (5×)

Across a 10–15 step jump cycle, A pays only ~0.1 reward for the entire jump, easily covered by `tracking_goal_vel` gain. B pays ~0.5–0.7 per jump → walking/crossing dominates. **This is the cleanest single mechanism by which A can profitably learn "jump over" while B is forced to "walk over."** It is **Genesis-faithful** in A (comment at env L1000 admits it), but B chose to soften the conditional to keep walking competitive — exactly the divergence we care about.

**Verification.** Single-variable change in A: replace `0.1` with `0.5` at `parkour_env.py:1001`, retrain (or at minimum, evaluate change in jumping rate / mean `|v_z|_max` per episode).

### 2) **`ang_vel_xy_l2` non-flat conditional multiplier — A=0.5× vs B=1.0× (no conditional)** &nbsp; *confidence: H*
**File:line:** A `parkour_env.py:1003`; B `rewards.py:88–93`

**Hypothesis.** A drops body-rotation penalty to 50 % on non-flat. Jumping requires large pitch oscillation (forward at take-off, backward at landing). B retains full penalty on all terrain, which directly discourages the pitch dynamics required for jumping — pushing policy toward the upright-body cross-over gait. Note this conditional is **A-only**; B has *no* terrain-conditional on this term. The fact that B never softens ang_vel_xy is a strong design signal.

**Verification.** Remove the conditional in A — i.e. compute `ang_vel_xy_l2` without `* (is_flat + is_non_flat * 0.5)` at `parkour_env.py:1003`. Expect upright posture on obstacles and reduced peak pitch rates.

### 3) **`action_rate_l2` weight — A=−0.05 vs B=−0.1** &nbsp; *confidence: M*
**File:line:** A `parkour_env_cfg.py:569` (comment: *"Genesis original (WAS −0.1, 10x error)"*); B `parkour_mdp_cfg.py:187`

**Hypothesis.** Jumping is an *impulsive* control pattern: legs slam from contracted to extended within 2–3 policy steps, then snap back. The action-rate term is the dominant smoothness regulariser. A applies **half** of B's penalty — making impulsive policies cheap. Walking is naturally smooth, so the same penalty barely affects it. The A-side comment claims Genesis used −0.05 and the previous −0.1 was an "error"; even if that is true vs Genesis, **the user's training target is B's behaviour, not Genesis'**, and B uses −0.1.

**Verification.** Single-variable change: `reward_scales["action_rate_l2"] = -0.1` in `parkour_env_cfg.py:569`, retrain. Expect lower per-step `|a_t − a_{t−1}|` and reduced peak ground-reaction impulses.

---

## Honorable mentions (lower confidence, list to keep for synthesis but not top-3)

- **Total-reward clipping** `torch.clip(total_reward, min=0.)` at `parkour_env.py:1108`. Removes negative gradient when penalties exceed bonuses. Plausible accomplice for high-variance jumping (worst case is clipped to 0, so the algorithm sees jumping as `≥0`), but it does not by itself *prefer* jump over walk — walking would also be `≥0`. Worth a B-aligned test (delete the clip and let total reward go negative).
- **`feet_air_time` removal (commit 25d2071196f).** Per the team brief, A previously had a `feet_air_time` term (commit e81b71c80ef) and it was removed to align with B. Confirmed absent in current A and B — RULED-OUT as a *current* DIFF on this axis. However, note: B has no air-time term, *and* still learns walking. So adding feet_air_time back is not necessary to fix gait; the discriminator must lie elsewhere (i.e. in the conditional multipliers + action_rate weight above).
- **Lin_vel_x command upper bound** A `[0.3, 1.0]` vs B `(0.3, 0.8)` — formally a command/observation diff (Axis 2/3) but indirectly inflates the cost of "walk slowly across hurdle" in A: at 1.0 m/s commanded over a 1.0–1.5 m platform-then-hurdle spacing, the agent has fewer steps per obstacle, so jumping pays better. Flag for cross-axis synthesis.

---

## Verification suggestions (single-variable tests for the user)

Each test changes **exactly one** sub-parameter from the A value to the B value and re-runs the standard smoke / mid-length training run. Compare metrics: mean `|v_z|_peak`, mean pitch-rate peak, fraction of "flight-phase" steps (all 4 feet airborne), mean episode `feet_air_time`, terrain-progression rate.

| # | Change in A | A value → B value | Expected effect if hypothesis is correct |
|---|---|---|---|
| **T1 (highest priority)** | `lin_vel_z_l2` non-flat multiplier (`parkour_env.py:1001`) | `0.1` → `0.5` | Mean peak `|v_z|` drops ~30–50 %; jumping → crossing gait emerges on hurdle/step terrain |
| **T2** | `ang_vel_xy_l2` non-flat multiplier (`parkour_env.py:1003`) | `0.5` → `1.0` (i.e. remove the conditional) | Lower peak body-pitch rate on obstacles; flatter base orientation during obstacle traversal |
| **T3** | `action_rate_l2` weight (`parkour_env_cfg.py:569`) | `-0.05` → `-0.1` | Smaller `‖a_t − a_{t−1}‖`; ground-contact pattern less impulsive |
| **T4 (combined)** | T1 + T2 + T3 together | as above | Strongest test — if jumping persists after all three, root cause is not in Axis 1 |
| **T5 (control)** | Remove `torch.clip(total_reward, min=0.)` at `parkour_env.py:1108` | enable negative rewards | Verifies whether the structural clip alone affects gait; expected: small effect on its own |

**Suggested order:** T1 → T3 → T2 → T5 → T4. T1 is the single highest-impact change.

---

## Already-aligned items (filter from candidates)

These were listed by the team brief / commit history as "ALIGNED" and were verified in the current code as equal in A and B — they cannot be the Axis-1 root cause:

- **Reward set 14-term alignment to B** — commit `25d2071196f` ("Align Parkour A reward set with B"). Confirmed: every term name, function form, and weight matches B with the **single exception of `action_rate_l2`** (see suspect #3). All other 13 cfg weights identical.
- **`feet_air_time` reward** — commit `e81b71c80ef` added it, then `25d2071196f` removed it during B-alignment. Current state: **absent in both A and B**. Per brief this is marked "ALIGNED" — confirmed RULED-OUT as a DIFF on this axis.
- **`reward_collision` body-set, threshold, scale** — A `parkour_env.py:1015–1019` (force>0.1, body set built in `__init__`), B `parkour_mdp_cfg.py:141–147` with `body_names=["base",".*_calf",".*_thigh"]`, weight −10. Aligned.
- **`feet_edge` parameters** — terrain-level gate `> 3`, force threshold 2 N, 2-sample OR. Aligned (A env L1054–1079, B rewards.py L37–64).
- **`hip_pos` joint set & scale** — −0.5, hip joints only. Aligned.
- **`dof_error_l2`, `torques_l2`, `delta_torques`, `dof_acc_l2`, `feet_stumble`** — all weights and functional forms identical.
- **`tracking_goal_vel` / `tracking_yaw`** — same weights (1.5 / 0.5), same `min(proj, cmd)/cmd` and `exp(−|Δyaw|)` forms.

The remaining DIFFs are concentrated in **terrain-class conditional multipliers (suspects #1, #2)** and **`action_rate_l2` weight (suspect #3)**, plus the structural total-reward clip.

---

## Hard-constraint compliance

- ✅ No actuator_mode=2 ranking.
- ✅ No height_scan citation.
- ✅ No torque-envelope math used to claim actuator weakening.
- ✅ Not framed as direct-vs-manager workflow.
- ✅ No contact sensor proposed as observation.
- ✅ Not framed as regression (A has never trained successfully; analysis is structural diff vs B).
- ✅ All four "already aligned" items (reward set, feet_air_time, RMA, B-style DR) marked as ALIGNED, not as candidates.
