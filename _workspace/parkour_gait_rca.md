# Parkour Gait RCA Report

**Author**: synth (worker in team `parkour-gait-debug`)
**Date**: 2026-05-12
**Scope**: Root-cause analysis for three observed gait failures on `Go2-Parkour-Direct-v0`
**Tier discipline**: A = quantitative log evidence · B = code-surface structural diff vs references · C = theoretical inference (must not be the #1 ranked hypothesis for any symptom)

---

## Executive Summary

| Symptom | Top Fix | Tier |
|---|---|---|
| **S3** Episode-start twitch | Reset `_yaw_diff` / `_next_yaw_diff` / `_scan` inside `_reset_idx()` (currently stale across episode boundary) | **B** |
| **S2** Yaw oscillation toward goal | Stop zeroing yaw dims in proprio history (`proprio_for_history[:, :2] = 0`) + refresh yaw_diff every step (decouple from the global %5 gate) | **B** |
| **S1** Rear-leg drag / over-jump / 2-leg stair jump | Add foot-contact bits (4 dims) into the policy proprio observation; corroborated by Tier-A worsening `dof_error_l2` | **A + B** |

All three top fixes are **structural** (obs layout / reset logic). No actuator changes, no reward-scale tuning, no hyperparameter changes are proposed.

---

## S3 — Episode-start twitch

### Observed evidence
- **Log (Tier A, via log-analyzer task #5)**: episode-start frames show large action-rate spikes and outlier `action_smoothness_*` penalties in the first 1–3 steps after reset; spike disappears by step ≈5.
- **Code (Tier B)**: `parkour_env.py:1083-1117` — `_reset_idx` resets `_actions`, `_previous_actions`, `_processed_actions`, `_last_processed_actions`, `_last_last_processed_actions`, `_last_applied_torque`, `_prev_joint_vel`, `_last_contacts`, `_proprio_history` (zeroed). It does **NOT** reset `_yaw_diff`, `_next_yaw_diff`, or `_scan`.
- **Code (Tier B)**: `parkour_env.py:583` — these three buffers are refreshed only when `self.common_step_counter % 5 == 0`.
- **Verified**: `common_step_counter` is a **GLOBAL** counter (`source/isaaclab/isaaclab/envs/direct_rl_env.py:389` — `self.common_step_counter += 1  # total step (common for all envs)`). It is **NOT** per-env.

### Root-cause hypotheses (ranked by tier)

#### H3-1 [Tier B] Stale `_yaw_diff` / `_next_yaw_diff` / `_scan` carried across episode boundary
- **Why**: After an env resets at iteration `t`, the next `_get_observations()` call returns proprio assembled from `self._yaw_diff` / `self._next_yaw_diff` (proprio dims 0,1) and `self._scan`. Those tensors are only overwritten when `common_step_counter % 5 == 0`. Because `common_step_counter` is GLOBAL, the env that just reset does not get a dedicated refresh — it inherits whatever stale value the previous episode left, for up to 4 policy steps (~80 ms at 50 Hz). The very first action after reset is therefore conditioned on the previous-episode goal direction → wrong heading → corrective spike on step 2 → the observed twitch.
- **Concrete change** (NO actual edit — proposal only):

  `parkour_env.py:1116-1117` (inside `_reset_idx`, after the `_proprio_history` reset):

  ```python
  # BEFORE
  # Reset proprioceptive history (Task #3)
  self._proprio_history[env_ids] = 0.0
  ```

  ```python
  # AFTER (proposed)
  # Reset proprioceptive history (Task #3)
  self._proprio_history[env_ids] = 0.0

  # Reset goal-yaw deltas and height scan so the first observation after
  # reset does not contain stale values from the previous episode.
  # (common_step_counter is GLOBAL, so the %5 refresh gate may skip this env.)
  self._yaw_diff[env_ids] = 0.0
  self._next_yaw_diff[env_ids] = 0.0
  self._scan[env_ids] = 0.0
  ```
  **Note**: a cleaner alternative is to refresh yaw_diff / scan unconditionally every step (see H2-3 fix), which subsumes this fix. If that option is chosen, this fix becomes redundant.
- **Risk**: low. Zero-init is a safe placeholder (matches the masked-out history convention used for yaw dims).
- **Verification plan**: log `proprio[env, 0:2]` (yaw_diff / next_yaw_diff) and `proprio[env, :]` mean-abs at episode step 1 for the first 10 resets per run; expect the values to drop from ~0.5–π rad (stale) to ~0 after the fix, and the action_rate/action_smoothness spike on steps 1–3 should disappear or reduce ≥50 %.

#### H3-2 [Tier B] First-step `_proprio_history` is filled from the (stale) current proprio
- **Why**: `parkour_env.py:839-843` — when `episode_length_buf <= 1`, the history ring buffer is filled by stacking the current `proprio_for_history` `history_len` times. If the current proprio carries the stale yaw_diff from H3-1, the entire 10-step history at episode start encodes the wrong goal direction. Even if the next %5 refresh corrects `_yaw_diff`, the policy sees that wrong direction in 9/10 history slots for ≥9 steps after reset.
- **Concrete change**: subsumed by H3-1 (fixing the stale yaw_diff also fixes the history fill). No standalone change required.
- **Risk**: n/a — derivative of H3-1.
- **Verification plan**: same as H3-1; additionally inspect `history[env, :, 0:2]` row variance across early steps — expect monotonic shift from "all-same stale" to "monotonic refresh" pattern.

#### H3-3 [Tier C] 0.05 m settling-buffer drop combined with `termination_grace_steps=5`
- **Why**: `parkour_env.py:1133` adds `+0.05 m` to the spawn z to avoid initial penetration; combined with `termination_grace_steps=5` (`parkour_env_cfg.py:521`), the first ~5 policy steps experience small free-fall transients. This is **expected** (not a bug) and is consistent across all 3 references.
- **EXCLUDED from top fix per tier discipline** (theoretical, no log signature distinct from H3-1).
- **Verification plan**: if H3-1 fix does NOT eliminate the spike, log `root_pos_w[:, 2]` and `root_lin_vel_w[:, 2]` for steps 1–5 after reset; if z-vel exceeds 0.3 m/s downward and bounces back, the settling drop is the residual cause.

---

## S2 — Yaw oscillation toward goal (despite GT yaw direction in observation)

### Observed evidence
- **Log (Tier A)**: per log-analyzer task #5, `tracking_yaw` reward (the ACTIVE yaw term, scale=0.5) is non-zero but exhibits high run-to-run variance and slow improvement.
  - ⚠️ **Clarification — log-analyzer note correction**: log-analyzer reported `tracking_ang_vel_z_exp: 0.0` and tagged it as "yaw reward inactive." That is **a misreading** of the cfg. `tracking_ang_vel_z_exp` is INTENTIONALLY disabled (scale=0.0 at `parkour_env_cfg.py:478`). The active yaw reward is `tracking_yaw` (scale=0.5, `parkour_env_cfg.py:476`), computed at `parkour_env.py:886-899`. Any S2 analysis must reference `tracking_yaw`, not `tracking_ang_vel_z_exp`.
- **Code (Tier B)**: `parkour_env.py:838` — `proprio_for_history[:, :2] = 0`. Dims 0 and 1 of proprio are `_yaw_diff` and `_next_yaw_diff` (lines 777-778). The history ring buffer therefore **never** contains yaw information — the policy can only see *current-step* yaw, never temporal context.
- **Code (Tier B)**: `parkour_env.py:583, 593-596` — `_yaw_diff` / `_next_yaw_diff` are refreshed only when `common_step_counter % 5 == 0`. At 50 Hz policy rate, the policy receives a yaw value that is up to **80 ms stale**.
- **Code (Tier B)**: `parkour_env.py:892-899` — `moving_mask = ((horizontal_speed − 0.05) / 0.10).clamp(0,1)`. This smoothstep on `tracking_yaw` is **not present** in any of the 3 reference implementations (ref-genesis, ref-isaaclab, ref-robotsw). The reward magnitude depends non-linearly on body-frame horizontal speed, introducing a coupled gradient w.r.t. speed in a term that should depend on yaw only.

### Root-cause hypotheses (ranked by tier)

#### H2-1 [Tier B] Yaw dims masked out of proprio history → policy has no temporal yaw context
- **Why**: With yaw zeroed in 9 of 10 history slots and current yaw refreshed only every 5 steps, the policy cannot form a temporal derivative of heading error. A controller without `d(yaw_error)/dt` will tend to overshoot then correct — the textbook bang-bang of P-only control = oscillation. References RobotSW and Isaaclab_Parkour both keep yaw in the policy obs history.
- **Concrete change** (proposal only):

  `parkour_env.py:836-843`:

  ```python
  # BEFORE
  proprio_for_history = proprio.clone()
  proprio_for_history[:, :2] = 0
  self._proprio_history = torch.where(
          (self.episode_length_buf <= 1)[:, None, None],
          torch.stack([proprio_for_history] * self.cfg.history_len, dim=1),
          torch.cat([self._proprio_history[:, 1:], proprio_for_history.unsqueeze(1)], dim=1),
  )
  ```

  ```python
  # AFTER (proposed)
  # Keep yaw_diff / next_yaw_diff in the history so the policy can form a
  # temporal derivative of heading error (prevents P-only oscillation).
  proprio_for_history = proprio.clone()
  # (no masking — leave dims 0 and 1 intact)
  self._proprio_history = torch.where(
          (self.episode_length_buf <= 1)[:, None, None],
          torch.stack([proprio_for_history] * self.cfg.history_len, dim=1),
          torch.cat([self._proprio_history[:, 1:], proprio_for_history.unsqueeze(1)], dim=1),
  )
  ```
- **Risk**: medium. Increases information content of history input; any policy network that was already overfitting to the masked layout will need re-training. Pretrained weights cannot be loaded as-is.
- **Verification plan**: train 50 iterations with the change; expect mean `tracking_yaw` to rise ≥15 % and yaw-error variance (post-hoc from rollout logs) to drop ≥30 %.

#### H2-2 [Tier B] `moving_mask` gating on `tracking_yaw` is not present in any reference
- **Why**: The smoothstep multiplier (`parkour_env.py:898-899`) zeroes the yaw reward below 0.05 m/s and ramps it linearly with body speed up to 0.15 m/s. This creates a coupled gradient: the policy can increase `tracking_yaw` reward by speeding up rather than by aligning yaw. Near the 0.05–0.15 m/s zone (typical at episode start, near-goal slowdown, and any pause), the yaw-reward gradient w.r.t. heading flickers on/off as speed crosses the threshold — perfect setup for oscillation.
- **Concrete change** (proposal only):

  `parkour_env.py:892-899`:

  ```python
  # BEFORE
  horizontal_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
  speed_lo = self.cfg.yaw_reward_speed_lower
  speed_hi = self.cfg.yaw_reward_speed_upper
  moving_mask = ((horizontal_speed - speed_lo) / (speed_hi - speed_lo)).clamp(min=0.0, max=1.0)
  tracking_yaw = tracking_yaw * moving_mask
  ```

  ```python
  # AFTER (proposed) — remove gating; revert to reference behavior
  # tracking_yaw is left as exp(-|yaw_diff|); references (ref-genesis,
  # ref-isaaclab, ref-robotsw) do not gate on speed. The "stand-still + face
  # goal" local optimum that motivated the gate is better addressed by the
  # tracking_goal_vel term (1.5 weight), which rewards forward velocity along
  # the goal direction and is zero for a stationary robot.
  ```
  Note: the commit `c3a7c4395af` that introduced this gating did so to prevent a stand-still local optimum, but `tracking_goal_vel` already provides the forward-velocity gradient with weight 1.5 vs `tracking_yaw` weight 0.5. Removing the gate restores the pure heading-tracking signal.
- **Risk**: medium. Removes the user's earlier mitigation against the stand-still local optimum. Should be paired with monitoring of `tracking_goal_vel` to confirm no regression to standing.
- **Verification plan**: train 50 iterations with gate removed; expect `tracking_yaw` to rise smoothly (no oscillation in its episode-mean curve) AND `tracking_goal_vel` to stay flat or rise. If `tracking_goal_vel` drops > 10 %, the local optimum is back → reconsider.

#### H2-3 [Tier B] `_yaw_diff` refreshed only every 5 global steps (≤ 80 ms stale)
- **Why**: Same `common_step_counter % 5 == 0` gate as in S3. The policy effectively does heading control on a yaw signal that is sampled at 10 Hz while emitting actions at 50 Hz. The Nyquist mismatch alone is enough to cause sub-100 ms oscillation around the goal direction.
- **Concrete change** (proposal only):

  `parkour_env.py:583-596`:

  ```python
  # BEFORE
  if self.common_step_counter % 5 == 0:
      scan = (
          self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.3
      ).clip(-1.0, 1.0)
      self._scan = scan.clone()
      yaw_diff = self._target_yaw - self._robot.data.heading_w
      self._yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
      next_yaw_diff = self._next_target_yaw - self._robot.data.heading_w
      self._next_yaw_diff = torch.atan2(torch.sin(next_yaw_diff), torch.cos(next_yaw_diff))
  ```

  ```python
  # AFTER (proposed) — refresh yaw every step; keep the %5 gate ONLY for the heavy height scan
  yaw_diff = self._target_yaw - self._robot.data.heading_w
  self._yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
  next_yaw_diff = self._next_target_yaw - self._robot.data.heading_w
  self._next_yaw_diff = torch.atan2(torch.sin(next_yaw_diff), torch.cos(next_yaw_diff))

  if self.common_step_counter % 5 == 0:
      scan = (
          self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.3
      ).clip(-1.0, 1.0)
      self._scan = scan.clone()
  ```
  Yaw computation is 4 cheap ops (2× atan2 + 2× sub) — there is no perf reason to gate it. Also subsumes the S3 H3-1 fix (yaw_diff is now always fresh after reset, assuming `self._target_yaw` is properly initialized in `_init_env_goals` which is called inside `_reset_idx` at line 1126).
- **Risk**: low. Tiny extra compute per step; height-scan gating preserved for perf.
- **Verification plan**: log standard deviation of `proprio[:, 0]` across consecutive policy steps; expect it to drop (current value oscillates between sampled and stale snapshots).

---

## S1 — Rear-leg drag (flat) / over-aggressive jumping (terrain) / 2-leg stair jump

### Observed evidence
- **Log (Tier A, log-analyzer task #5)**: `dof_error_l2` is **WORSENING** across training in all 3 runs (-28 % to -57 % degradation across iterations). This is anomalous — joint-tracking penalty should improve, not regress, as the policy learns.
- **Log (Tier A)**: `feet_dragging` reward (scale=-0.1) is active and non-zero; needs quantitative magnitude check per run but the existence of the active term means the dragging signal is reaching the policy gradient.
- **Code (Tier B)**: `parkour_env.py:774-787` — policy proprio (42 dims) = `_yaw_diff`(1) + `_next_yaw_diff`(1) + `projected_gravity_b`(3) + `cmd_vx`(1) + `joint_pos_rel`(12) + `joint_vel*0.05`(12) + `prev_actions`(12). **There is no foot-contact channel.**
- **Code (Tier B)**: `parkour_env.py:148` — `self._last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, ...)`. Only a single-step boolean contact memory exists, and it is used **only** internally for the `feet_dragging` reward (line 978-979). The policy never sees it.
- **Reference (Tier B)**: ref-isaaclab proprio carries 5-dim contact obs (4 contact bits + 1 threshold). ref-robotsw policy obs carries `contact_filt - 0.5` (4 dims). Genesis does not (Genesis is also simpler and uses different reward shaping).
- **Reference (Tier B)**: ref-robotsw maintains a 100-step `contact_buf` history — current IsaacLab impl has zero contact history.

### Root-cause hypotheses (ranked by tier)

#### H1-1 [Tier A + B] No foot-contact observation in the policy proprio → blind gait coordination
- **Why**: The policy must decide which feet to lift/plant each step. Without contact bits, it can only infer ground contact indirectly from `joint_pos` / `joint_vel` / `projected_gravity` — a much weaker signal than the binary "is foot X loaded right now." The observable symptoms all match "blind gait":
  - **Rear-leg drag on flat**: front legs drive forward, rear legs are not actively lifted because the policy doesn't know they are still on the ground.
  - **2-leg stair jump**: the policy commits to a jump push-off without knowing which two feet are loaded → uses only the pair that happens to be in contact.
  - **Over-aggressive jumping (left rear)**: when the policy decides to clear an obstacle, it issues a swing command with no feedback on whether the foot is still loaded — combined with the high-saturation actuators (contextual fact, not a knob), the resulting swing is exaggerated.
- The Tier-A `dof_error_l2` worsening is consistent with "policy is using action range to override default pose to over-correct for the lack of contact feedback" — the policy is learning to ride the action limit, which inflates `dof_error_l2`.
- **Concrete change** (proposal only):

  `parkour_env.py:774-787` (policy proprio) — add 4 contact bits, with the value held over one policy step in `_last_contacts`:

  ```python
  # BEFORE
  proprio = torch.cat(
      [
          self._yaw_diff[:, None],                                             # 1
          self._next_yaw_diff[:, None],                                        # 1
          self._robot.data.projected_gravity_b,                                # 3
          self._commands[:, 0:1],                                              # 1
          self._robot.data.joint_pos - self._robot.data.default_joint_pos,    # 12
          self._robot.data.joint_vel * 0.05,                                   # 12
          self._actions,                                                       # 12
      ],
      dim=-1,
  )
  # proprio dim = 3 + 1 + 1 + 1 + 12 + 12 + 12 = 42
  ```

  ```python
  # AFTER (proposed)
  # Foot contact bits: shifted to [-0.5, +0.5] to match RobotSW convention
  # (zero-mean → does not bias linear layer at init).
  contact_obs = self._last_contacts.float() - 0.5  # (N, 4)
  proprio = torch.cat(
      [
          self._yaw_diff[:, None],                                             # 1
          self._next_yaw_diff[:, None],                                        # 1
          self._robot.data.projected_gravity_b,                                # 3
          self._commands[:, 0:1],                                              # 1
          self._robot.data.joint_pos - self._robot.data.default_joint_pos,    # 12
          self._robot.data.joint_vel * 0.05,                                   # 12
          self._actions,                                                       # 12
          contact_obs,                                                         # 4 (NEW)
      ],
      dim=-1,
  )
  # proprio dim = 1+1+3+1+12+12+12+4 = 46
  ```
  **Companion config edits** (must be synchronized):
  - `parkour_env_cfg.py:331` — `observation_space: int = 42` → `46`
  - `parkour_env_cfg.py:335` — `num_proprio: int = 42` → `46`
  Both go through `cfg-worker` for normal review (no logic change, only declared-dim sync).
  **History initialization caveat**: `_last_contacts` is reset to `False` at `parkour_env.py:1098`, so the first proprio after reset contains `0 - 0.5 = -0.5` for all 4 feet (indicating "all airborne"). Since the robot is on the ground at spawn this is slightly incorrect for ~1 step but resolves on the next step when contact is detected. Acceptable.
- **Risk**: medium. Changes the policy input dim (42 → 46); requires retraining from scratch (no transfer of old checkpoints). Network's first linear layer width must accommodate the change — `network-worker` should confirm `actor_critic.py` reads `num_proprio` from cfg.
- **Verification plan**: train 100 iterations with contact obs; expect (a) `dof_error_l2` to stabilize or improve (no worsening trend), (b) `feet_dragging` reward magnitude to drop ≥30 %, (c) qualitative gait check on flat terrain (play.py) shows 4-foot swing instead of dragging rear legs.

#### H1-2 [Tier B] No contact history → policy cannot detect "stuck on ground" patterns
- **Why**: RobotSW keeps 100-step `contact_buf` so the policy can learn long-horizon gait patterns (e.g., "rear-right has been in contact for the last 20 steps → it must be lifted now"). The current IsaacLab impl has zero contact history. With the H1-1 fix in place, the contact bits ARE present in the policy proprio, and since `_proprio_history` length is 10, they will appear in the 10-step history automatically once H1-1 is applied. So this fix is partially subsumed by H1-1; the remaining gap (100-step vs 10-step history) is a secondary consideration.
- **Concrete change**: no additional change required if H1-1 is applied; consider extending `history_len` from 10 to a larger value in a separate experiment. Out of scope for the immediate top fix.
- **Risk**: n/a — derivative of H1-1.
- **Verification plan**: post H1-1, inspect whether `history[:, :, -4:]` (the contact slice across 10 steps) carries informative variance; if yes, current `history_len=10` is sufficient. If the policy still drags after H1-1, revisit with history_len=20.

#### H1-3 [Tier C] Worsening `dof_error_l2` driven by high actuator headroom
- **Why** (CONTEXTUAL FACT, NOT A FIX KNOB): the actuator override at `parkour_env_cfg.py:402-411` sets `stiffness=40`, `saturation_effort=35`, `effort_limit` up to 40 N·m. Higher saturation enables larger joint deviations from `default_joint_pos` per unit action → the policy can hit `dof_error_l2` peaks cheaply. Combined with no contact feedback (H1-1), the optimizer exploits this headroom for over-aggressive jumps.
- **EXCLUDED from top fix per user constraints** — actuator changes are NOT acceptable as fix proposals. Documented here purely as context for why H1-1's effect may be amplified.
- **Verification plan**: after H1-1, `dof_error_l2` trend should reverse even with current actuators. If it does not, escalate to the user for actuator discussion (not for an automated fix).

---

## Appendix: Non-fix observations & clarifications

### A. log-analyzer `tracking_ang_vel_z_exp` clarification
log-analyzer task #5 reported `tracking_ang_vel_z_exp: 0.0` and labeled it "yaw reward inactive."
**This is a misread.** The cfg `parkour_env_cfg.py:478` deliberately sets `tracking_ang_vel_z_exp: 0.0` (the term is disabled). The active yaw reward is `tracking_yaw` (scale=0.5, `parkour_env_cfg.py:476`), computed at `parkour_env.py:886-899`. Any downstream consumer of the log-analyzer report should substitute `tracking_yaw` whenever yaw reward signal is discussed.

### B. Actuator settings are a contextual fact
`stiffness=40`, `saturation_effort=35`, `effort_limit` up to 40 N·m (`parkour_env_cfg.py:400-416`, mode 1).
These values give the robot more torque headroom than the stock Go2 config (mode 2: stiffness=25, sat=23.5). They enable sharp leg lifts, which is consistent with the over-aggressive jumping observed. **Per user constraint, no fix here proposes changing these values.** They are documented purely so that downstream consumers do not mistake "current behavior is partly enabled by current actuators" for "the fix is to change actuators."

### C. `_yaw_diff` wrapping via `atan2(sin, cos)` is correct
`parkour_env.py:594, 596` wraps yaw delta to `[-π, π]` using `atan2(sin(Δ), cos(Δ))`. RobotSW uses an unwrapped delta which is incorrect near ±π; the current wrap is **good practice**, not a problem. Do not "fix" this.

### D. Height-scan formula `pos_w[:, 2] - ray_hits_w[..., 2] - 0.3`
Per project memory, this has been independently verified as normal. Comments in `parkour_env.py:578-582` warn that the formula "appears" wrong (sensor offset (0,0,20) suggests saturation), but the height_scanner uses `attach_yaw_only` with default-frame conversion and the offset is treated relative to robot base. **Do not propose this as a root cause** for any of S1/S2/S3.

### E. Stale comment in `parkour_env_cfg.py`
Lines 327-329 in `parkour_env_cfg.py` reference `num_proprio = 46` in the obs-space breakdown comment, while the actual cfg field is `num_proprio: int = 42` (line 335) and matches code. This is a stale comment, not a bug. If H1-1 is applied (which raises proprio to 46), this comment will accidentally become correct again — but a synchronized update of cfg + comment + code is still required.

### F. History `history_len = 10` is consistent
`parkour_env_cfg.py:338` → `history_len: int = 10`. `parkour_env.py:155, 841` both use `self.cfg.history_len`. No inconsistency between code and cfg. (Earlier ref-current note about "10 vs 6" appears to be conflating an obs-space breakdown comment with the actual field.)

### G. No gait/feet_air_time reward is missing from any impl
None of the 4 references (current, Genesis, Isaaclab_Parkour, RobotSW) implement a `feet_air_time` or explicit gait-period reward. **Do not propose adding one** without strong evidence; the current `feet_dragging` (-0.1) + `feet_stumble` (-1.0) + `feet_edge` (-1.0) cover the foot-contact cost surface.

---

## Cross-symptom synthesis

Two of the three root causes (S2-H2-3 and S3-H3-1) share the same underlying defect: **the global `common_step_counter % 5 == 0` gate at `parkour_env.py:583` controls both yaw and height-scan refresh, but yaw is a cheap, fast-changing signal that should not be gated**.

A single change at `parkour_env.py:583` (move yaw refresh out of the gate, leave height scan gated) fixes both S3-H3-1 (no more stale yaw across reset boundary) and S2-H2-3 (no more 80 ms stale yaw during normal stepping). This is recommended as the **first concrete code change** to apply, before trying H1-1 (contact obs) and H2-1 (un-mask history yaw) — because it requires no obs-space dim change and therefore no retraining of the network width.

Recommended fix-application order:
1. **First** — `parkour_env.py:583` yaw-refresh decoupling (H2-3 fix, also subsumes H3-1). No network change.
2. **Second** — `parkour_env.py:838` un-mask yaw in history (H2-1 fix). No network change.
3. **Third** — `parkour_env.py:774-787` + cfg dim sync, add contact bits (H1-1 fix). Requires network input-dim change → retrain from scratch.
4. **Fourth** (only if 1–3 do not fix S3) — explicit `_reset_idx` reset of `_yaw_diff`/`_next_yaw_diff`/`_scan` (H3-1 standalone). Belt-and-suspenders.

If retraining capacity allows, items 1–3 can be batched into a single retraining run; otherwise apply 1+2 first (zero-dim-change) and validate before committing to item 3.

---

## Worker delegation map

| Fix | Owner worker | Files |
|---|---|---|
| Decouple yaw refresh from %5 gate (S2-H2-3, S3-H3-1) | `obs-worker` | `parkour_env.py:583-596` |
| Un-mask yaw in proprio history (S2-H2-1) | `obs-worker` | `parkour_env.py:836-843` |
| Remove `moving_mask` gating on `tracking_yaw` (S2-H2-2) | `reward-worker` | `parkour_env.py:892-899` |
| Add contact bits to policy proprio (S1-H1-1) | `obs-worker` + `cfg-worker` | `parkour_env.py:774-787`, `parkour_env_cfg.py:331,335` |
| Explicit reset of stale buffers (S3-H3-1, fallback) | `obs-worker` | `parkour_env.py:1116-1117` |

---

*End of report.*

## Critic Verdict

**Reviewer**: critic (worker in team `parkour-gait-debug`)
**Date**: 2026-05-12
**Mode**: THOROUGH (no escalation to ADVERSARIAL — no CRITICAL findings; gates all pass)
**Result**: **PASS** (all 7 constraint gates C1–C7 satisfied)

### Independent verification of Tier B "common_step_counter is GLOBAL" claim

The single most load-bearing Tier B claim in the report (used by H3-1, H3-2, H2-3, and the cross-symptom synthesis) is:
> `common_step_counter` is GLOBAL (`source/isaaclab/isaaclab/envs/direct_rl_env.py:389`), so the `% 5 == 0` refresh gate at `parkour_env.py:583` may skip a freshly-reset env.

Independently re-read by critic:
- `source/isaaclab/isaaclab/envs/direct_rl_env.py:388-389` →
  ```
  self.episode_length_buf += 1  # step in current episode (per env)
  self.common_step_counter += 1  # total step (common for all envs)
  ```
  This is a single scalar incremented once per `env.step()` call, NOT a per-env tensor. The synth claim is **VERIFIED**, not surface inference. Tier B classification stands.
- `source/isaaclab_tasks/.../parkour/parkour_env.py:583-596` → `if self.common_step_counter % 5 == 0:` block guards exactly the assignments to `self._scan`, `self._yaw_diff`, `self._next_yaw_diff`. Gate semantics confirmed.
- `_reset_idx` at `parkour_env.py:1083-1140` resets `_actions`, `_previous_actions`, `_prev_joint_vel`, `_processed_actions`, `_last_processed_actions`, `_last_last_processed_actions`, `_last_applied_torque`, `_last_contacts`, `_term_goal_reached`, `_reach_goal_timer`, `_target_pos_rel`, `_next_target_pos_rel`, `_target_yaw`, `_next_target_yaw`, `_proprio_history` — but **does NOT reset** `_yaw_diff`, `_next_yaw_diff`, or `_scan`. Synth claim confirmed verbatim.

Project memory `feedback_dont_assert_unverified_bugs.md` requirement is satisfied: the claim is grounded in two concrete line references in two files, both re-read by critic.

### Gate-by-gate result

| Gate | Criterion | Result | Notes |
|---|---|---|---|
| **C1** | Every hypothesis has explicit tier tag | **PASS** | 9/9 hypotheses tagged: H3-1=B, H3-2=B, H3-3=C, H2-1=B, H2-2=B, H2-3=B, H1-1=A+B, H1-2=B, H1-3=C |
| **C2** | No #1-ranked hypothesis is Tier C | **PASS** | S3#1=H3-1 [B], S2#1=H2-1 [B], S1#1=H1-1 [A+B]. H3-3 and H1-3 (both Tier C) are explicitly demoted and excluded from top-fix slots |
| **C3** | No fix proposes stiffness/damping changes | **PASS** | H1-3 (actuator headroom) is explicitly marked "EXCLUDED from top fix per user constraints — actuator changes are NOT acceptable as fix proposals". No other hypothesis touches `actuator_stiffness/damping` |
| **C4** | Reward_scale fixes have quantitative log evidence | **PASS (vacuous)** | No `reward_scales` dict value is changed. H2-2 removes a `moving_mask` multiplier (code-logic, Tier B vs reference diff), not a scale value — out of C4's scope |
| **C5** | Hyperparameter fixes have explicit evidence | **PASS (vacuous)** | Report explicitly states "no hyperparameter changes are proposed"; none found in the fix list |
| **C6** | Every hypothesis has measurable 1-line verification plan | **PASS** | All 9 hypotheses carry a verification line with numbers (≥50 % spike reduction, ≥15 % tracking_yaw rise, ≥30 % drag drop, ≥30 % yaw-error variance drop, etc.) or post-hoc inspection criteria |
| **C7** | S3 section traces reset → first-step path with line numbers | **PASS** | Trace chain present: `_reset_idx` 1083-1117 (lists buffers reset) → does NOT reset `_yaw_diff/_next_yaw_diff/_scan` → next `_get_observations()` reads them at proprio dims 0-1 (777-778) and as `_scan` → refresh gated by `%5` at 583 → counter is GLOBAL (direct_rl_env.py:389) → therefore stale. Plus H3-2 traces history-fill at 839-843 |

### Minor observations (non-blocking)

These do NOT cause a gate failure, but the synth worker may wish to be aware:

1. **Line-range precision** — H3-1 says "`parkour_env.py:1083-1117` — `_reset_idx` resets … _proprio_history". The proprio_history reset is at line 1117; the function actually extends to line 1140+. Range is accurate for the reset section but not the whole function. Cosmetic.
2. **H2-2 reward-semantics framing** — The `moving_mask` removal is a reward-computation change. While it does not touch `reward_scales` dict (so C4 vacuous-PASS holds), a strict reading of "reward fix proposals" could ask for quantitative log evidence of the oscillation/local-optimum trade-off. The verification plan (≥10 % drop in `tracking_goal_vel` as a regression trigger) is acceptable for now, but the synth should flag this for the team-lead so reward-worker is aware the change is structural-not-scalar.
3. **Stale comment lines 325-329 in cfg** — The report correctly identifies this as a stale comment (appendix E). Confirmed by critic re-read.
4. **No false-positive on height-scan** — Appendix D respects project memory `project_parkour_height_scan_verified.md` by explicitly forbidding height-scan from being raised as a root cause. Compliant.

### Executive summary (user-facing)

The Parkour RCA report passes all seven critic gates. **S3 (episode-start twitch)** top fix is to decouple yaw_diff/next_yaw_diff refresh from the global `common_step_counter % 5` gate at `parkour_env.py:583` (Tier B; the synth's "GLOBAL counter" claim was independently re-verified at `direct_rl_env.py:389`, and this single change also subsumes the explicit-reset-on-`_reset_idx` fallback). **S2 (yaw oscillation toward goal)** top fix is to stop masking yaw_diff/next_yaw_diff out of `_proprio_history` at `parkour_env.py:838` (Tier B), restoring temporal heading-error context so the policy can form a derivative term and stop bang-banging; secondary fix is removing the `moving_mask` smoothstep on `tracking_yaw` (892-899), which is not present in any of the three references. **S1 (rear-leg drag / over-aggressive jump / 2-leg stair jump)** top fix is to add 4-dim foot-contact bits to the policy proprio at `parkour_env.py:774-787` with synchronized cfg dim bumps 42→46 (Tier A+B), supported by the worsening `dof_error_l2` trend (-28 % to -57 %) reported in log-analyzer task #5. All three fixes are structural — **no actuator stiffness/damping, reward_scale, or hyperparameter changes are proposed**, consistent with the project constraints. Recommended application order: (1) decouple yaw refresh from %5 gate, (2) un-mask history yaw, (3) add contact bits (this third step requires retraining from scratch because obs-dim grows 42→46).

*End of Critic Verdict.*
