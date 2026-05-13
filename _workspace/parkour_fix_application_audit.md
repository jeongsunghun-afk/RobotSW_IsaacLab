# Parkour Gait RCA — Fix Application Audit

**Author**: fix-auditor (team `parkour-asym-rca`)
**Scope**: Verify which of the 4 fixes proposed in `_workspace/parkour_gait_rca.md` are present in `HEAD` (commits `2aab9ee`, `c3a7c43` claim to apply fixes).
**Method**: Read-only inspection of `parkour_env.py` / `parkour_env_cfg.py` at current HEAD + `git log -S` for each fix keyword + `git show` of the two named commits.
**Tier discipline**: A = log evidence, B = code-surface diff, C = inference. Top-1 cannot be Tier C.
**Forbidden hypotheses** (project-memory blocked, not raised anywhere in this report): `_actuator_mode=2`, torque envelope tuning, new rewards, scale tuning, action latency, height-scan formula.

---

## Section 1 — Fix-by-fix verdict table

| Fix | Verdict | Commit | Current line / snippet | Notes |
|---|---|---|---|---|
| **Fix-1** Decouple yaw refresh from `common_step_counter % 5` gate (`parkour_env.py:583-596` in original RCA) | **PARTIALLY APPLIED** | `2aab9ee` | `parkour_env.py:593-626` — `recently_reset = (episode_length_buf <= 1)`; `if do_global_refresh or recently_reset.any():` then `if do_global_refresh: self._yaw_diff = yaw_raw … else: self._yaw_diff[recently_reset] = yaw_raw[recently_reset]` | Covers the **S3 reset-boundary** case (newly-reset envs get a per-env one-shot patch). Does **NOT** cover the **S2 in-flight oscillation** case: normal-stepping envs (`episode_length_buf > 1`) still wait for the global `% 5 == 0` gate → policy keeps seeing yaw signal up to **80 ms stale** at 50 Hz. The parkour_gait_rca.md cross-symptom recommendation (move yaw refresh fully out of the gate and keep only height-scan gated) is **not** implemented. Also: lines 617-618 / 625-626 use **unwrapped** `yaw_raw` (no `atan2(sin, cos)`); the wrapping lines are commented out, regressing the wrap-correctness noted in Appendix C of parkour_gait_rca.md. |
| **Fix-2** Remove `proprio_for_history[:, :2] = 0` (yaw mask in history) | **NOT APPLIED** | (none — no commit touches this line) | `parkour_env.py:867-873` — `proprio_for_history = proprio.clone(); proprio_for_history[:, :2] = 0; self._proprio_history = torch.where(...)`. The mask is still in place. `git log --all -S "proprio_for_history[:, :2]"` returns no commits. | Yaw_diff / next_yaw_diff (proprio dims 0-1) are still wiped to 0 in every history slot → policy still has zero temporal yaw context → H2-1 (P-only oscillation) defect persists exactly as described in parkour_gait_rca.md. |
| **Fix-3** Append `(self._last_contacts.float() - 0.5)` to proprio cat + cfg `observation_space: 42→46`, `num_proprio: 42→46` | **NOT APPLIED** | (none — `git log --all -S "_last_contacts.float"` returns 0 commits) | `parkour_env.py:805-816` — proprio cat ends with `self._actions, # 12`; comment line 817 confirms `proprio dim = 3 + 1 + 1 + 1 + 12 + 12 + 12 = 42`. `parkour_env_cfg.py:331` → `observation_space: int = 42`; line 335 → `num_proprio: int = 42`. Stale comment at lines 325-329 still says `policy: 3+3+2+2+12+12+12 = 46`, `history: 10 * 46 = 460`, `critic = 697` — **inconsistent with the actual 42-dim code** (the comment was already noted as stale in parkour_gait_rca.md Appendix E). | `_last_contacts` is declared at `parkour_env.py:148` and used only internally for the `feet_dragging` reward (~line 978) and for the new `reset` gating in commit `2aab9ee`. It is **never** placed into the policy proprio tensor. This is the leading candidate for Symptom #3 — see Section 2. |
| **Fix-4** Remove `tracking_yaw = tracking_yaw * moving_mask` | **APPLIED** | `c3a7c43` | `parkour_env.py:916-922` — `# tracking_yaw: exponential decay from heading error (Genesis line 1451-1454)` / `# No speed gating — stand-still local optimum is prevented by tracking_goal_vel (weight=1.5) …` / `tracking_yaw = torch.exp(-torch.abs(yaw_diff))`. `git show c3a7c43` diff shows the `horizontal_speed = …`, `moving_mask = …`, `tracking_yaw = tracking_yaw * moving_mask` block deleted (prefixed `-` in the diff). | Commit message: "removing stationary yaw reward exploit." Fix matches parkour_gait_rca.md H2-2 verbatim. Note: the same commit message says "tuning actuators" — those changes are out of scope of this audit (project-memory blocks `_actuator_mode` reasoning). |

### Cross-checks against lead's partial confirmations
- ✅ `_last_contacts` is **NOT** in the proprio cat at `parkour_env.py:805` (only used for `feet_dragging` reward and for `2aab9ee` reset gating). Confirmed.
- ✅ `parkour_env_cfg.py:331,335` still say `42`. Confirmed.
- ✅ Stale comment `parkour_env_cfg.py:325-329` still says `46`. Confirmed (inconsistent with code's 42).

---

## Section 2 — Stair-freeze hypothesis ranking (Symptom #3)

**Symptom #3 description (from team-lead spec)**: on stair/step terrain, robot bows toward the goal, freezes, and does not commit to stepping. Persists after `2aab9ee` + `c3a7c43`.

Ranking must stay within parkour_gait_rca.md theory; no new theories permitted.

### #1 — Fix-3 (contact bits in policy proprio) — **Tier A + B, missing**

**Why it's #1 for stair freeze, with code evidence only**:

- **Tier B (code-surface)** — proprio cat at `parkour_env.py:805-816` contains: `yaw_diff(1) + next_yaw_diff(1) + projected_gravity_b(3) + cmd_vx(1) + joint_pos_rel(12) + joint_vel*0.05(12) + actions(12) = 42`. **There is no foot-contact channel.** Reference implementations cited in parkour_gait_rca.md H1-1 (ref-isaaclab, ref-robotsw) **do** carry contact bits in the policy proprio. The policy must select which foot to lift/plant each policy step (20 ms at 50 Hz); without `is_foot_X_loaded`, this selection has to be inferred indirectly from `joint_pos / joint_vel / projected_gravity`. On flat terrain that signal is weak but workable; on a stair edge it is **insufficient** — the policy cannot tell which of the 4 feet are committed to a step and which are free to swing.

- **Tier A (log evidence inherited from parkour_gait_rca.md)** — log-analyzer task #5 reported `dof_error_l2` worsening across training (−28 % to −57 % degradation). parkour_gait_rca.md H1-1 attributes this to "the policy is using action range to override default pose to over-correct for the lack of contact feedback." On stairs this manifests as "the policy decides to commit to a push-off, can't verify which feet are loaded, retracts, tries again" — i.e. the **bow-and-freeze** pattern. The Tier-A worsening trend predates the stair freeze; the freeze is the terrain-specific failure mode of the same defect.

- **Tier B (consistency with the symptom)** — the description "robot bows to goal, freezes, does not commit to stepping" is the textbook symptom of an open-loop foot-placement controller. The policy is **gait-blind**: it computes a yaw-aligned bow toward the goal (yaw signal is fresh, see Fix-1 partial), then must trigger a step. Without contact-bits feedback, the cost gradient for "lift foot X" is uniformly weak across all 4 feet → no commitment → freeze. The 2-leg stair jump described in parkour_gait_rca.md S1 is the same failure on harder difficulty rows.

This is the **leading candidate** for the stair-freeze symptom, and the evidence is **Tier A + B** (not Tier C). The Tier C exclusion rule is therefore not violated.

### #2 — Fix-2 (un-mask yaw in proprio history) — **Tier B, missing**

**Why this contributes to stair freeze (not the primary cause)**:

- **Tier B** — `parkour_env.py:868` still zeros yaw dims in the history ring buffer. parkour_gait_rca.md H2-1 framing: without a temporal yaw signal in history, the policy effectively runs P-only heading control → bang-bang oscillation around the goal direction. On flat terrain this manifests as yaw oscillation; on a stair edge, the same P-only behaviour interacts with the stepping decision — the policy oscillates between "align yaw" and "lift foot," neither converges, and the robot bows-and-pauses. This is consistent with the bow-and-freeze description but the freeze itself is more directly attributable to the missing stepping commitment signal (Fix-3).

- Cannot be ranked #1: the symptom "freeze" is more strongly explained by the gait-blind defect than by the temporal-yaw-context defect. Pure P-only yaw control would produce **oscillation** (the S2 symptom), not freeze.

### #3 — Fix-1 (decouple yaw refresh from %5 gate) — **Tier B, partial**

- The **reset side** of Fix-1 is in `HEAD` (commit `2aab9ee`'s `recently_reset` branch). For envs that have just reset, yaw is now fresh on step 1. This addresses the **S3 episode-start twitch**, not the persistent stair freeze that happens during normal stepping.
- The **in-flight side** of Fix-1 (yaw refresh every policy step regardless of the `% 5` gate) is **NOT** applied: normal-stepping envs still see yaw up to 80 ms stale. This contributes to yaw oscillation (S2) but is not the direct cause of stair freeze. Also, lines 617-618 / 625-626 use **unwrapped** `yaw_raw` (the `atan2(sin, cos)` wrap call is commented out in HEAD), which silently regresses the wrap-correctness flagged in parkour_gait_rca.md Appendix C — separate concern, not the freeze driver, but worth noting for whoever owns Fix-1.

### #4 — Fix-4 (remove moving_mask) — **Tier B, applied**

Already in HEAD via `c3a7c43`. Cannot explain symptom persistence by definition. Listed here for completeness.

### Tier validation summary
- **#1 Fix-3** = Tier A (dof_error_l2 worsening from parkour_gait_rca.md task #5 log evidence) + Tier B (code diff vs ref-isaaclab / ref-robotsw). **Compliant with the no-Tier-C-as-#1 rule.**
- **#2 Fix-2** = Tier B (code diff vs references).
- **#3 Fix-1** = Tier B (code diff). Partially applied.
- **#4 Fix-4** = Tier B. Already applied.

---

## Section 3 — Recommended verification

### Primary action: apply Fix-3 (contact bits in policy proprio) and retrain

Concrete steps, in order:

1. **Code edit (obs-worker)** — append `contact_obs = self._last_contacts.float() - 0.5  # (N, 4)` to the proprio cat at `parkour_env.py:805-816`. New proprio dim = 46.
2. **Cfg edit (cfg-worker)** — `parkour_env_cfg.py:331` `observation_space: 42 → 46`; line 335 `num_proprio: 42 → 46`. Update the stale comment block at lines 325-329 to reflect the new layout (the existing comment already says `46`, so this becomes correct again after this edit).
3. **Network shape check (network-worker)** — verify `actor_critic.py` reads `num_proprio` from cfg so the first-layer input width auto-adapts. parkour_gait_rca.md H1-1 already flagged this as a network-worker concern.
4. **Train**: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Parkour-Direct-v0 --num_envs 4096` for **500–1000 iterations** (Fix-3 changes obs-dim → cannot transfer old checkpoint, must retrain from scratch).
5. **Pass criteria** (also from parkour_gait_rca.md H1-1 verification plan):
   - **Tier A signals**: `curriculum/mean_terrain_level_stair` must rise above its current floor (where the freeze pins it). `dof_error_l2` trend must reverse (stop worsening, ideally improve). `feet_dragging` reward magnitude should drop ≥ 30 %.
   - **Tier B qualitative**: `play.py` rollout on a stair row — the robot should commit to a step rather than bow-and-freeze. 4-foot swing pattern visible.

### If Fix-3 alone does not lift the stair-freeze (secondary action)

Apply Fix-2 (`parkour_env.py:868` — delete `proprio_for_history[:, :2] = 0`). No dim change, but obs *content* in the history slots changes, so the policy should be retrained or at minimum fine-tuned. parkour_gait_rca.md recommends Fix-3 + Fix-2 batched in a single retraining run if retraining capacity allows. Optionally also extend Fix-1 to full in-flight decoupling (move yaw refresh out of the `do_global_refresh` gate; leave only the height-scan inside the gate). Fix-1's full form is dim-preserving and can be combined safely with Fix-3 + Fix-2.

### Do NOT propose
- Any actuator / stiffness / damping change (`_actuator_mode=2`-style reasoning blocked by project memory).
- Any reward-scale tuning, new reward terms, action latency, or height-scan formula change (per project memory + parkour_gait_rca.md Appendices B, D, G).

---

*End of audit.*
