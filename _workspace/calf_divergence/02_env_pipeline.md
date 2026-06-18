# [HYP-ENV] Parkour Action Pipeline — Calf Joint Analysis

**Date:** 2026-05-26
**Task:** #2 — Map raw → scale → clip → joint target → PD → torque pipeline for calf joints
**Author:** env-investigator

---

## 1. Files Examined

| File | Role |
|------|------|
| `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` | `action_scale`, `clip_actions`, actuator cfg, `_actuator_mode` |
| `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` | `_pre_physics_step`, `_apply_action`, joint indexing, obs assembly |
| `source/isaaclab_assets/isaaclab_assets/robots/unitree.py` | `UNITREE_GO2_CFG` — default joint pos, actuator K/D/sat/vel |
| `agents/rsl_rl_ppo_cfg.py` | Runner `clip_actions`, `init_noise_std`, actor head |
| `rsl_rl/rsl_rl/modules/actor_critic.py` | `act()` — distribution sample, no actor-level clip |
| `rsl_rl/rsl_rl/runners/on_policy_runner_parkour.py` | Runner — no `clip_actions` usage found |
| `RobotSW_Genesis/genesis/assets/urdf/go2/urdf/go2.urdf` | Authoritative joint limits (parsed by `xml.etree`) |

---

## 2. Full Action Pipeline for Calf Joints

### Step 1: Policy Raw Output (actor head)

**File:** `rsl_rl/rsl_rl/modules/actor_critic.py:152–156`

```python
def act(self, obs, **kwargs):
    obs = self.get_actor_obs(obs)
    obs = self.actor_obs_normalizer(obs)
    self._update_distribution(obs)
    return self.distribution.sample()   # ← Normal(mean, std).sample(), NO clip
```

- Distribution: `Normal(mean, std)`, `init_noise_std = 1.0` (`agents/rsl_rl_ppo_cfg.py:59`)
- **No tanh, no clip at actor-head level.** Raw actions are unbounded Gaussian samples.
- At inference: `act_inference()` returns the mean directly (`actor_critic.py:164`), also unclipped.

### Step 2: Runner-Level Clip

**File:** `agents/rsl_rl_ppo_cfg.py:27` — `clip_actions = 10.0`

- This field exists on `Go2ParkourPPORunnerCfg` but is **never read** by any runner code.
- Verified: zero hits in `on_policy_runner_parkour.py`, `on_policy_runner.py`, `vec_env.py` for `clip_actions`.
- **Runner clip: NONE (dead config field).**

### Step 3: Env-Level Clip

**File:** `parkour_env.py:524`

```python
self._actions = torch.clip(actions.clone(), -self.cfg.clip_actions, self.cfg.clip_actions)
```

- `cfg.clip_actions = 10.0` (`parkour_env_cfg.py:393`)
- **Active clip: ±10.0 on all 12 DOF uniformly.**

### Step 4: Hip Scale Reduction

**File:** `parkour_env.py:525–527`

```python
scaled = self._actions.clone()
scaled[:, self._hip_joint_ids] *= 0.5   # hip joints only
```

- Hip joint indices found dynamically: `[i for i, n in enumerate(all_joint_names) if "hip" in n]` (`parkour_env.py:273–277`)
- **Calf joints are NOT scaled here.** Effective calf clip remains ±10.0.

### Step 5: Scaling + Default Position Offset

**File:** `parkour_env.py:534`

```python
self._processed_actions = self.cfg.action_scale * scaled + self._robot.data.default_joint_pos
```

- `action_scale = 0.25` (`parkour_env_cfg.py:389`)
- `default_calf_pos = -1.5 rad` (all 4 calf joints, `unitree.py:165`: `".*_calf_joint": -1.5`)
- **Joint target = 0.25 × clipped_action + (−1.5)**

### Step 6: Joint Position Target Sent to Sim

**File:** `parkour_env.py:537`

```python
self._robot.set_joint_position_target(self._processed_actions)
```

- Target is the full 12-DOF processed tensor, in `self._robot.data` ordering (no manual reindex).

### Step 7: PD Torque Computation

**Active actuator config** (`_actuator_mode = 2`, `parkour_env_cfg.py:477–517`):

| Parameter | Value |
|-----------|-------|
| `stiffness` (K) | 25.0 N·m/rad |
| `damping` (D) | 0.5 N·m·s/rad |
| `saturation_effort` | 23.5 N·m |
| `effort_limit` (calf) | 23.5 N·m |
| `velocity_limit` (calf) | 30.0 rad/s |
| `armature` | 0.01 kg·m² |

Torque formula: `τ = clip(K × (target − pos) − D × vel, −sat, +sat)`

### Step 8: Effort Limit / Saturation

Final torque clipped to ±23.5 N·m per calf joint.

---

## 3. Numerical Table: Per-Joint Pipeline Values

> All 4 calf joints (FL/FR/RL/RR) are **identical** in limits and default position.

| Parameter | FL/FR Calf | RL/RR Calf | Notes |
|-----------|------------|------------|-------|
| `default_joint_pos` | −1.5000 rad | −1.5000 rad | `unitree.py:165` |
| `action_scale` | 0.25 | 0.25 | `parkour_env_cfg.py:389` |
| `clip_actions` | ±10.0 | ±10.0 | `parkour_env_cfg.py:393` |
| **Scaled max displacement** | ±2.500 rad | ±2.500 rad | `0.25 × 10.0` |
| **Target range** | [−4.000, +1.000] rad | [−4.000, +1.000] rad | `default + scale × clip` |
| URDF hard lower limit | −2.7227 rad | −2.7227 rad | go2.urdf (parsed) |
| URDF hard upper limit | −0.83776 rad | −0.83776 rad | go2.urdf (parsed) |
| **URDF range** | 1.8849 rad | 1.8849 rad | upper − lower |
| Soft lower limit (×0.9) | −2.6285 rad | −2.6285 rad | center ± 0.9×half_range |
| Soft upper limit (×0.9) | −0.9320 rad | −0.9320 rad | |
| **Headroom to UPPER limit** | +0.6622 rad | +0.6622 rad | `upper − default` |
| Action to reach UPPER limit | +2.649 raw | +2.649 raw | `0.6622 / 0.25` |
| **Headroom to LOWER limit** | −1.2227 rad | −1.2227 rad | `lower − default` |
| Action to reach LOWER limit | −4.891 raw | −4.891 raw | `−1.2227 / 0.25` |
| Effort limit (calf) | 23.5 N·m | 23.5 N·m | `parkour_env_cfg.py:506–510` |
| K (stiffness) | 25.0 | 25.0 | mode=2 |
| D (damping) | 0.5 | 0.5 | mode=2 |

### Key Computed: PD Torque at Extreme Actions

Assuming joint is physically clamped at URDF hard limit when target exceeds it:

| Action value | Target pos | Joint actual | Position error | Raw PD torque | After sat clip |
|-------------|------------|--------------|---------------|--------------|----------------|
| +10.0 (max clip) | +1.0000 rad | −0.83776 rad | +1.8378 rad | +45.94 N·m | **+23.5 N·m** (maxed) |
| −10.0 (min clip) | −4.0000 rad | −2.72270 rad | −1.2773 rad | −31.93 N·m | **−23.5 N·m** (maxed) |
| +2.649 (upper limit) | −0.8378 rad | −0.83776 rad | ~0 | ~0 | ~0 |
| −4.891 (lower limit) | −2.7228 rad | −2.72270 rad | ~0 | ~0 | ~0 |
| 0.0 (default) | −1.5000 rad | −1.5000 rad | 0 | 0 | 0 |

**→ Any calf action beyond ±2.649 / −4.891 drives the actuator to max torque output.**

---

## 4. Joint Indexing Verification (H5)

### Ordering in URDF (go2.urdf — parsed in joint definition order)

```
idx=0  FL_hip_joint      idx=1  FL_thigh_joint    idx=2  FL_calf_joint
idx=3  FR_hip_joint      idx=4  FR_thigh_joint    idx=5  FR_calf_joint
idx=6  RL_hip_joint      idx=7  RL_thigh_joint    idx=8  RL_calf_joint
idx=9  RR_hip_joint      idx=10 RR_thigh_joint    idx=11 RR_calf_joint
```
(URDF element indices after removing fixed joints; actual runtime ordering depends on USD import order)

### Obs vs. Action Alignment

| Location | Mechanism | Source |
|----------|-----------|--------|
| Obs `joint_pos` | `self._robot.data.joint_pos` (native IsaacLab ordering) | `parkour_env.py:868` |
| Obs `joint_vel` | `self._robot.data.joint_vel` (same ordering) | `parkour_env.py:869` |
| Action target | `self._processed_actions` derived from `self._robot.data.default_joint_pos` (same ordering) | `parkour_env.py:534` |
| `set_joint_position_target` | same ordering | `parkour_env.py:537` |
| Hip scaling | `self._hip_joint_ids` dynamically found via string match `"hip" in name` | `parkour_env.py:273–277` |

**No manual index remapping found anywhere in the pipeline.**
Both obs and action use `self._robot.data` ordering — they are internally consistent by construction.

The only potential H5 risk would be if the USD joint ordering differs from expectation, but since no re-mapping exists, obs and action would still be self-consistent (all channels shift together).

---

## 5. Hypothesis Evaluation

### H1: action_scale × clip_actions > headroom to joint limit (asymmetric range)
**Status: ✅ STRONGLY CONSISTENT**

**Evidence:**
- Scaled action range: ±2.5 rad from default (`parkour_env_cfg.py:389,393` + `unitree.py:165`)
- Calf joint range (URDF): 1.8849 rad total (`go2.urdf`)
- Target range = [−4.0, +1.0] rad vs. URDF limits [−2.7227, −0.83776] rad
- **Both ends of the target range exceed physical limits:**
  - Positive (flexion) overshoot: clip hits at action=+10.0, limit reached at action=+2.649 → **7.351 unused units (73.5% of +clip range)** drive joint to max torque
  - Negative (extension) overshoot: clip hits at action=−10.0, limit reached at action=−4.891 → **5.109 unused units (51.1% of −clip range)** drive joint to max torque

**Asymmetry:** The calf joint default (−1.5 rad) is NOT centered in the joint range (center = −1.780 rad). The default is closer to the upper (flexion) limit by 0.66 rad vs. 1.22 rad to the extension limit. This means:
- Flexion saturation occurs at action = **+2.649** (well within Normal(0,1) std=1.0 territory)
- Extension saturation occurs at action = **−4.891** (requires >4σ excursion from a fresh policy)

**Implication:** The policy encounters the flexion clamping zone first and more frequently. If any reward signal nudges calf actions positive, the policy quickly learns to push actions past +2.649, where maximum torque (23.5 N·m) is delivered continuously with no additional gradient feedback from joint position.

**Falsification test:** If calf actions were found to saturate in the extension direction (very negative) instead of the flexion direction, H1 would be partially weakened (extension saturation requires larger action magnitude).

---

### H2: default_joint_pos for calf is near a joint limit, range asymmetric
**Status: ⚠️ PARTIALLY CONSISTENT**

**Evidence:**
- `default_calf_pos = −1.5 rad` (`unitree.py:165`)
- URDF range center = −1.780 rad
- Default is 0.28 rad toward the upper (flexion) end of the range
- Distance to upper limit: 0.662 rad (35.1% of range)
- Distance to lower limit: 1.223 rad (64.9% of range)

The default is not "near the limit" in an absolute sense (not within 5–10% of limit), but the asymmetry means the policy has significantly less room in the flexion direction. Combined with H1, this asymmetry makes flexion saturation structurally more likely.

**Refuting evidence:** The default is not extreme (e.g., not within 0.1 rad of the limit). The asymmetry is moderate. H2 alone would not cause saturation; it amplifies H1.

**Falsification:** If a symmetric default (−1.780 rad) were used with the same action_scale/clip, flexion and extension saturation would be equally likely.

---

### H3: PD gains insufficient to hold body → requires extreme calf commands
**Status: ⚠️ UNDETERMINED (project memory constraint: cannot rank top-1/2)**

**Evidence FOR:**
- K=25 with max position error within range: worst case 25 × 1.88 = 47 N·m → clipped to 23.5 N·m even at max legal position error
- With effort_limit = saturation_effort = 23.5 N·m, the actuator is always at or near saturation whenever the joint is pulled to its limits
- For parkour (jumping, obstacle clearance), high-bandwidth torque demands may exceed what K=25/D=0.5/sat=23.5 can provide

**Evidence AGAINST:**
- Project memory: "saturation 23.5/vel 30/K25/D0.5/armature 0.01 actuator 약화 설정은 이전 학습 성공 사례 있음" — learning with these exact parameters succeeded previously. This confirms the gains are not categorically insufficient.
- The calf saturation appears as a consequence of the action range exceeding physical limits (H1), not a failure of the PD to generate needed torque.

**Note:** The saturation at 23.5 N·m is reached for any position error > 23.5/25 = 0.94 rad. Within the calf's 1.88-rad range, any action that moves the joint more than 0.94 rad from actual position saturates the actuator. This is a structural property of the DCMotorCfg, not a malfunction.

---

### H4: effort_limit/saturation forces policy to max action to get moderate torque
**Status: ⚠️ UNDETERMINED (related to H3; same constraint)**

**Evidence:**
- At action=0 (default pos), torque = 0 (joint at target)
- At action=+1.0 (target = −1.25 rad, within range), error if joint at −1.5 rad = 0.25 rad, torque = 25 × 0.25 = 6.25 N·m — well below saturation
- Saturation only occurs when position error > 0.94 rad, i.e., when action drives target far outside actual joint position
- **There is no incentive to push action beyond the limit just to get more torque** — torque from a moderate action (say +2.0) with the joint near default is 25 × 0.5 = 12.5 N·m, adequate for stance

**The saturation seen at calf joints is more consistent with the policy actively choosing large actions** (driven by reward) **rather than needing more torque from small actions.** H4 is a weaker explanation than H1.

---

### H5: Joint name ordering mismatch between obs / action / reward
**Status: ❌ INCONSISTENT WITH AVAILABLE EVIDENCE**

**Evidence:**
- Obs assembly (`parkour_env.py:868–870`): uses `self._robot.data.joint_pos` and `joint_vel` — native IsaacLab ordering, no remapping
- Action target (`parkour_env.py:534,537`): `self._processed_actions = action_scale * scaled + self._robot.data.default_joint_pos` then `set_joint_position_target(self._processed_actions)` — same ordering
- Hip scaling (`parkour_env.py:273–277,527`): index found dynamically by name string match, not hardcoded
- No `joint_names` reordering, no explicit index tables for calf vs. other joints

**Both obs and action use the identical `self._robot.data` ordering throughout.** An indexing mismatch between obs and action is structurally impossible unless IsaacLab's `joint_pos` and `set_joint_position_target` use different orderings internally (not observed in IsaacLab's design).

**Caveat (verified absence):** No joint-index remapping found in parkour_env.py. If a future grep reveals explicit index selection (e.g., `[:, 2::3]` style slicing), this hypothesis should be re-evaluated.

---

## 6. Summary: Pipeline from Raw → Torque

```
Policy output (Normal sample, unbounded)
    ↓
[NO actor clip — actor_critic.py:156]
    ↓
[NO runner clip — on_policy_runner_parkour.py: no clip_actions usage]
    ↓
ENV CLIP: torch.clip(actions, -10.0, +10.0)     [parkour_env.py:524]
    ↓
Hip scale ×0.5 (hip joints only)                 [parkour_env.py:527]
    ↓
target = 0.25 × action + default_joint_pos       [parkour_env.py:534]
    ↓  (for calf: default = -1.5 rad)
    ↓  target ∈ [-4.0, +1.0] rad
    ↓  URDF hard limit ∈ [-2.7227, -0.83776] rad
    ↓  ← target CAN exceed limits at action > +2.649 or < -4.891
    ↓
PD: τ = K*(target - pos) - D*vel                 [DCMotorCfg]
    K=25, D=0.5
    ↓
Saturation clip: τ ∈ [-23.5, +23.5] N·m         [parkour_env_cfg.py:503,506-510]
    ↓
Applied torque to joint
```

---

## 7. Key Finding: Dead Zone at High Calf Action Values

The combination of `action_scale=0.25`, `clip_actions=10.0`, and `default_calf_pos=−1.5` with URDF limits `[−2.7227, −0.83776]` creates a significant **dead zone**:

- For calf action ∈ [+2.649, +10.0]: joint is physically clamped at upper limit (−0.83776 rad). Additional action increase beyond +2.649 has **zero effect on joint position** — it only maintains max torque output.
- For calf action ∈ [−10.0, −4.891]: joint is physically clamped at lower limit (−2.7227 rad). Same zero-effect zone.

The dead zone spans **73.5% of the positive clip range** and **51.1% of the negative clip range**.

This creates a policy gradient problem: once actions are in the dead zone, the PD error signal (reward terms like `dof_error_l2`) sees a constant residual regardless of how the policy changes its calf output. The only signal comes from rewards that don't depend on joint position (e.g., velocity tracking, which could drive the calf to stay clamped if that stance helps locomotion).

**The "calf divergence" pattern — where calf actions grow large and stay there — is mechanistically explained by H1**: the action space for calf joints extends well beyond physical reachability, so the policy can enter a high-action regime that remains rewarded (or at least not penalized) due to the torque floor at the joint limit.

---

## 8. Note on `clip_actions=10.0` in Runner Cfg

`agents/rsl_rl_ppo_cfg.py:27` contains `clip_actions = 10.0` on `Go2ParkourPPORunnerCfg`. This value is **never consumed** by `on_policy_runner_parkour.py`, `on_policy_runner.py`, or `vec_env.py`. It is a dead config field with no runtime effect. The only active clip is in `parkour_env.py:524` reading `self.cfg.clip_actions` from `ParkourEnvCfg`.

---

*Report written by task #2 env-investigator. Research only — no code was modified.*
