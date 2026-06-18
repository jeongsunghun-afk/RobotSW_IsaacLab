# Parkour Reward 구현 비교 분석 보고서

> **A**: `/home/lgb/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/rewards.py` (Manager-based MDP, 14 term)
> **B**: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` (Direct-style RL env, 22 term)
> **목적**: reward 계산 방법(센서 데이터 획득 방식 + 수식) 정밀 비교. reward_scale은 표로만 병기, 가치 판단 없음.

---

## 0. 구조적 차이 (TL;DR)

| 항목 | A (`parkour_isaaclab`) | B (`IsaacLab parkour_env.py`) |
|---|---|---|
| Env baseclass | `ManagerBasedRLEnv` 계열 (`ParkourManagerBasedRLEnv`) | `DirectRLEnv` (`Go2ParkourEnv`) |
| Reward 정의 방식 | 각 reward = 독립 함수/클래스, `RewTerm(func, weight, params)`로 cfg에서 wiring | 모든 reward 로직이 `parkour_env.py:_get_rewards()` 한 메서드 안 + 헬퍼 메서드 |
| Stateful reward 처리 | `ManagerTermBase` 서브클래스 — `__init__` / `reset(env_ids)` 자체 보유 | env attribute (`self._prev_*`, `self._last_*`) 로 보관, `_reset_idx`에서 일괄 reset |
| State 접근 경로 | `env.scene["robot"].data.*`, `env.scene.sensors["contact_forces"].data.*` (string key lookup) | `self._robot.data.*`, `self._contact_sensor.data.*` (직접 attribute) |
| Per-env terrain context | `env.parkour_manager.get_term(parkour_name)` → `ParkourEvent` (target_pos_rel, target_yaw, terrain_levels, env_per_terrain_name 등) | env attribute 직접 (`self._target_pos_rel`, `self._target_yaw`, `self._terrain_levels`, `self._env_class`) |
| Reward 누적 | `RewardManager`가 `weight * value` 자동 합산 | `total += scale * step_dt * value` 명시 loop (`step_dt=0.02` 모든 term에 곱) |
| Term 수 | 14 | 22 (B는 Genesis/anymal-c/parkour 혼합 포팅) |
| Frame mixing | tracking_goal_vel=world xy, lin_vel_z/ang_vel_xy=body | 동일 패턴 + tracking_lin_vel_xy_exp(body)/tracking_ang_vel_z_exp(body) 추가 |

---

## 1. 계산 결과에 적용되는 스칼라 multiplier 차이 (구조적)

A는 `RewardManager`가 단순히 `value * weight`를 합산.

B는 모든 term에 대해 한 자리에서:
```python
scaled = self.cfg.reward_scales[key] * self.step_dt * value
total_reward += scaled
```
즉 B는 추가로 **`step_dt=0.02`가 모든 term에 곱**해진다. 같은 "weight=−1.0"이라도 effective per-step contribution은 A가 1.0배, B가 0.02배이다 (단위적으로 B는 "per-second weight"에 가깝다). 이 점은 동일 scale 숫자를 비교할 때 반드시 인지해야 한다.

---

## 2. Reward 매핑 — 어느 term이 양쪽에 존재하는가

| A side                    | B side (cfg key)                | 비교 가능? |
|---------------------------|----------------------------------|-----------|
| `reward_tracking_goal_vel`  | `tracking_goal_vel`             | ✅ 1:1 |
| `reward_tracking_yaw`       | `tracking_yaw`                  | ✅ 1:1 (B에 wrap 미적용 동일) |
| `reward_lin_vel_z`          | `lin_vel_z_l2`                  | ✅ 1:1 (gating 방식만 다름) |
| `reward_ang_vel_xy`         | `ang_vel_xy_l2`                 | ✅ 1:1 (gating 방식만 다름) |
| `reward_orientation`        | `orientation_l2`                | ✅ 1:1 |
| `reward_dof_acc`            | `dof_acc_l2`                    | ✅ 1:1 |
| `reward_torques`            | `torques_l2`                    | ✅ 1:1 |
| `reward_delta_torques`      | `delta_torques`                 | ✅ 1:1 |
| `reward_dof_error`          | `dof_error_l2`                  | ✅ 1:1 (B는 flat-only mask 추가) |
| `reward_hip_pos`            | `hip_pos`                       | ✅ 1:1 |
| `reward_action_rate`        | `action_rate_l2`                | ✅ 동일하게 둘 다 L2 *norm* (squared가 아닌 점 양쪽 일치) |
| `reward_feet_stumble`       | `feet_stumble`                  | ✅ 1:1 |
| `reward_feet_edge`          | `feet_edge`                     | ✅ 1:1 (B는 `_last_contacts` 활용으로 contact_filt 약간 다름) |
| `reward_collision`          | `collision`                     | ⚠️ 거의 동일하지만 history reduction 다름 (A: idx 0만, B: max over history) + 대상 body 집합 다름 |
| — | `tracking_lin_vel_xy_exp`      | B 전용 (현재 scale 0) |
| — | `tracking_ang_vel_z_exp`       | B 전용 (현재 scale 0) |
| — | `termination`                  | B 전용 (early-term penalty) |
| — | `feet_dragging`                | B 전용 |
| — | `feet_air_time`                | B 전용 (anymal-c 스타일) |
| — | `action_smoothness_1`          | B 전용 (a_t − a_{t−1})² |
| — | `action_smoothness_2`          | B 전용 (a_t − 2a_{t−1} + a_{t−2})² |
| — | `base_height`                  | B 전용 |

---

## 3. Reward Scale 병기 (값만, 분석 없음)

> A는 Teacher stage 기준 (Student는 거의 비활성화 — `reward_collision -0.0`만). B는 현재 디버깅 중 튜닝값.

| Term | A (Teacher weight) | B (`reward_scales`) | 비고 |
|---|---:|---:|---|
| tracking_goal_vel       | `+1.5`     | `+1.0`        | |
| tracking_yaw            | `+0.5`     | `+0.5`        | |
| tracking_lin_vel_xy_exp | —          | `0.0`         | B 전용 |
| tracking_ang_vel_z_exp  | —          | `0.0`         | B 전용 |
| lin_vel_z (`_l2`)       | `-1.0`     | `-1.0`        | gating ×0.5(A non-flat) / ×0.1(B non-flat) |
| ang_vel_xy (`_l2`)      | `-0.05`    | `-0.05`       | A: 무조건 / B: ×0.5 on non-flat |
| orientation (`_l2`)     | `-1.0`     | `-1.0`        | 양쪽 모두 flat-only |
| dof_acc (`_l2`)         | `-2.5e-7`  | `-2.5e-7`     | |
| collision               | `-10.0`    | `-10.0`       | history reduction 다름 (§6) |
| action_rate (`_l2`)     | `-0.1`     | `-0.0`        | B에서는 현재 off |
| delta_torques           | `-1.0e-7`  | `-1.0e-7`     | |
| torques (`_l2`)         | `-1.0e-5`  | `-1.0e-5`     | |
| hip_pos                 | `-0.5`     | `-0.0`        | B에서는 현재 off |
| dof_error (`_l2`)       | `-0.04`    | `-0.0`        | B에서는 현재 off, flat-only |
| feet_stumble            | `-1.0`     | `-0.0`        | B에서는 현재 off |
| feet_edge               | `-1.0`     | `-0.0`        | B에서는 현재 off |
| termination             | —          | `-100.0`      | B 전용 |
| feet_dragging           | —          | `-0.0`        | B 전용, 현재 off |
| action_smoothness_1     | —          | `-0.0`        | B 전용 |
| action_smoothness_2     | —          | `-0.0`        | B 전용 |
| base_height             | —          | `0.0`         | B 전용, flat-only |
| feet_air_time           | —          | `0.0`         | B 전용 |

**효과 단위 환산 주의**: B는 모든 항목에 `× step_dt(0.02)`가 추가로 곱해진다. 즉 A "weight=−1.0"의 step-당 영향은 `value`, B "scale=−1.0"의 step-당 영향은 `0.02 × value`.

---

## 4. 공통 reward — Sensor data 획득 방식 비교

### 4.1 `tracking_goal_vel`

| 항목 | A | B |
|---|---|---|
| 로봇 속도 소스 | `asset.data.root_vel_w[:, :2]` (alias of `root_lin_vel_w`) — world frame xy | `self._robot.data.root_lin_vel_w[:, :2]` — world frame xy |
| 목표 방향 소스 | `env.parkour_manager.get_term(parkour_name).target_pos_rel` `(N,2)` | `self._target_pos_rel` `(N,2)` (env attribute, `_update_goals`에서 갱신) |
| 커맨드 소스 | `env.command_manager.get_command('base_velocity')[:, 0]` `(N,)` | `self._commands[:, 0]` `(N,)` (env attribute) |
| 단위 벡터화 | `target_pos_rel / (‖·‖ + 1e-5)` | `target_pos_rel / (‖·‖ + 1e-5)` |
| 투영 | `<unit_dir, vel_w_xy>` | `Σ_xy (vel_w_xy * unit_dir)` (수학적으로 동일) |
| Saturation | `min(proj_vel, cmd) / (cmd + 1e-5)` | `min(proj_forward, |cmd|) / (|cmd| + 1e-5)` |
| 커맨드 zero guard | 없음 (분모에 `+1e-5`만) | `where(|cmd|>1e-3, r, 0)` 명시 (정지 명령일 때 신호 차단) |
| Return shape | `(N,)` | `(N,)` |

**핵심 차이**: 데이터 접근 경로가 manager(string key) vs direct attribute뿐, **수식은 사실상 동일**. B는 추가로 `where(|cmd|>1e-3, r, 0)`로 정지 명령 시 0 보장.

---

### 4.2 `tracking_yaw`

| 항목 | A | B |
|---|---|---|
| 현재 yaw | `root_quat_w`(w,x,y,z 순)에서 직접 `atan2(2(w·z + x·y), 1 − 2(y² + z²))` 계산 | `self._robot.data.heading_w` 직접 사용 (IsaacLab가 사전 계산) |
| target yaw | `env.parkour_manager.get_term(parkour_name).target_yaw` | `self._target_yaw` |
| wrap | **없음** — raw `|target − yaw|` | **없음** — raw `|target − yaw|` (`wrap_to_pi`/`atan2(sin,cos)` 코드 주석 처리됨) |
| 수식 | `exp(−|target_yaw − yaw|)` | `exp(−|target_yaw − heading_w|)` |
| Return shape | `(N,)` | `(N,)` |

**핵심 차이**: yaw 추출 경로 — A는 quaternion에서 직접, B는 IsaacLab가 미리 노출한 `heading_w` 사용. 수식 동일. ±π 경계 discontinuity 문제는 양쪽 모두 동일하게 존재.

---

### 4.3 `lin_vel_z` (penalty)

| 항목 | A | B |
|---|---|---|
| 데이터 소스 | `asset.data.root_lin_vel_b[:, 2]` (body z) | `self._robot.data.root_lin_vel_b[:, 2]` (body z) |
| 수식 (raw) | `v_z²` | `v_z²` |
| Terrain modulation | `parkour_manager.env_per_terrain_name[:, -1]`로 terrain 이름 조회, `!= 'parkour_flat'`이면 `×= 0.5` | `self._env_class`(int LUT)로 `is_flat`/`is_non_flat` 분리, `× (is_flat + 0.1*is_non_flat)` |
| Non-flat 시 효과 배율 | **0.5** | **0.1** (5배 더 약함) |
| Return shape | `(N,)` | `(N,)` |

**핵심 차이**:
1. terrain 분류 방식 — A는 이름 문자열 비교(현재-step의 last terrain name), B는 `_env_class` int 인덱싱(LUT 기반, 더 빠름).
2. non-flat에서 penalty 가중치 — A=0.5 vs B=0.1.

---

### 4.4 `ang_vel_xy` (penalty)

| 항목 | A | B |
|---|---|---|
| 데이터 소스 | `asset.data.root_ang_vel_b[:, :2]` (body x,y) | `self._robot.data.root_ang_vel_b[:, :2]` (body x,y) |
| 수식 (raw) | `Σ_i ω_i²` over xy | `Σ_i ω_i²` over xy |
| Terrain modulation | **없음** — 모든 terrain에서 동일 가중 | `× (is_flat + 0.5*is_non_flat)` (non-flat에서 2배 약함) |
| Return shape | `(N,)` | `(N,)` |

**핵심 차이**: A는 unconditional, B는 non-flat에서 ×0.5 modulation 적용.

---

### 4.5 `orientation` (penalty)

| 항목 | A | B |
|---|---|---|
| 데이터 소스 | `asset.data.projected_gravity_b[:, :2]` | `self._robot.data.projected_gravity_b[:, :2]` |
| 수식 (raw) | `Σ_i g_proj_i²` | `Σ_i g_proj_i²` |
| Mask | `rew[terrain != parkour_flat] = 0.` (조건부 zero, hard) | `× is_flat` (multiplicative, equivalent) |
| 효과 | 양쪽 모두 **flat에서만** 작동 | 동일 |

**핵심 차이**: 구현 방법(인덱싱 vs 곱하기)만 다르고 결과 동일.

---

### 4.6 `dof_acc` (penalty)

| 항목 | A | B |
|---|---|---|
| 데이터 소스 | `asset.data.joint_vel` `(N, J)` | `self._robot.data.joint_vel` `(N, 12)` |
| Previous velocity buffer | `ManagerTermBase` 내부 `previous_joint_vel[N, 2, J]` (window=2, slot 0/1 사용) | env attribute `self._prev_joint_vel[N, 12]` (단일 slot) |
| Buffer 업데이트 | 호출 시점에 `prev[:,0] = prev[:,1]; prev[:,1] = joint_vel`, **계산 후 prev로 다음 step 사용** | `_get_rewards` 진입 즉시 `prev_joint_vel = joint_vel.clone()` 전에 `acc` 계산 후 갱신 (B는 코드 상단에서 `joint_acc = (joint_vel - prev) / step_dt` 계산 → 마지막에 `self._prev_joint_vel = joint_vel.clone()`) |
| `dt` 출처 | `self.dt = env.cfg.decimation * env.cfg.sim.dt` (init 시 캐시) | `self.step_dt` (env attribute) |
| 수식 | `acc = (curr − prev) / dt`; `Σ_j acc_j²` | `acc = (curr − prev) / step_dt`; `Σ_j acc_j²` |
| Reset | `reset(env_ids)`에서 prev 양 slot 0으로 | `_reset_idx`에서 `_prev_joint_vel[env_ids] = 0` |

**핵심 차이**:
- A는 2-슬롯 window를 유지하지만 실제로는 항상 인접 2-step 차분만 사용 → B와 수식적으로 동일.
- 양쪽 모두 reset 시 prev=0 (첫 step에서 acc가 `joint_vel/dt`로 큰 값 → 1 step "spike" 가능)는 동일한 잠재 특성.

---

### 4.7 `torques` / `delta_torques` (penalty)

| 항목 | A | B |
|---|---|---|
| 데이터 소스 | `asset.data.applied_torque` | `self._robot.data.applied_torque` |
| `torques` 수식 | `Σ_j τ_j²` | `Σ_j τ_j²` |
| `delta_torques` prev buffer | class 내부 `previous_torque[N, 2, J]` (window 2) | env `self._last_applied_torque[N, 12]` (단일 slot) |
| `delta_torques` 수식 | `Σ_j (τ_t − τ_{t−1})²` | `Σ_j (τ_t − τ_{t−1})²` |
| Reset | `reset(env_ids)` 0 초기화 | `_reset_idx`에서 0 초기화 |

**핵심 차이**: 사실상 동일. 데이터 출처도 동일 attribute `applied_torque`.

---

### 4.8 `dof_error` / `hip_pos` (penalty)

| 항목 | A | B |
|---|---|---|
| 데이터 소스 | `asset.data.joint_pos`, `asset.data.default_joint_pos` | `self._robot.data.joint_pos`, `self._robot.data.default_joint_pos` |
| `dof_error` 수식 | `Σ_j (q_j − q_default,j)²` over all joints | `Σ_j (q_j − q_default,j)²` over all joints, then `× is_flat` (flat-only) |
| `hip_pos` 수식 | `Σ_hip (q − q_default)²`; hip 집합은 `SceneEntityCfg(joint_names=".*_hip_joint")`로 정의 | `Σ_hip (q − q_default)²`; hip 집합은 `__init__`에서 `find_joints(".*hip.*")`로 캐시 (`_hip_joint_ids`) |
| Return shape | `(N,)` | `(N,)` |

**핵심 차이**:
- B의 `dof_error_l2`만 추가로 `is_flat` 마스킹(non-flat에서 0). A는 unconditional.
- hip 매칭 패턴이 A=`.*_hip_joint`(엄밀한 joint suffix), B=`.*hip.*`(더 느슨, ‘thigh_hip’ 같은 잘못된 매칭 가능성). Go2의 경우 결과는 동일할 수 있으나 정규식 자체는 다름.

---

### 4.9 `action_rate` (penalty)

| 항목 | A | B |
|---|---|---|
| 현재 action 소스 | `env.action_manager.get_term('joint_pos').raw_actions` | `self._actions` (policy raw output, `_pre_physics_step`에서 update) |
| Previous action buffer | class 내부 `previous_actions[N, 2, J]` (window) | env `self._previous_actions[N, 12]` (단일 slot, `_pre_physics_step`에서 갱신) |
| 수식 | `‖prev[:,1] − prev[:,0]‖₂` over joints (`torch.norm(..., dim=1)`) | `‖actions − previous_actions‖₂` (`torch.norm(diff, dim=-1)`) |
| Reduction | **L2 norm (제곱근 포함)** | **L2 norm (제곱근 포함)** |
| Reset | `reset(env_ids)` window 양 slot 0 | `_reset_idx`에서 `_previous_actions[env_ids] = 0` |

**핵심 차이**: 양쪽 모두 **sum-of-squares가 아닌 L2 norm**으로 일치. B 코드의 주석은 "Genesis sum-square와 의도적으로 다르다"는 점을 명시. action 소스는 manager vs env attribute뿐 의미적으로 동일.

---

### 4.10 `feet_stumble` (penalty)

| 항목 | A | B |
|---|---|---|
| Contact force 소스 | `contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]` (history idx 0 = 최신 substep) | `self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_ids]` (history idx 0) |
| Body 집합 | sensor_cfg(`body_names`)로 4개 foot 매칭 | `find_bodies(".*foot")` 결과 `self._feet_ids` (4개) |
| 수식 | `F_xy = ‖F_xy‖₂; F_z = |F_z|; flag = F_xy > 4·F_z; rew = any_feet(flag).float()` | 동일 — `any_feet(‖F_xy‖ > 4·|F_z|)` → float |
| Return shape | `(N,)` 0/1 | `(N,)` 0/1 |

**핵심 차이**: 본질적으로 동일. body id 캐시 방법(`SceneEntityCfg` vs `find_bodies`)만 다름.

---

### 4.11 `feet_edge` (penalty, gated by terrain_levels)

| 항목 | A | B |
|---|---|---|
| Foot world position | `asset.data.body_state_w[:, asset_cfg.body_ids, 0:2]` | `self._robot.data.body_pos_w[:, self._feet_ids, :2]` |
| 현재 contact | `‖net_forces_w_history[:, 0, body_ids]‖ > 2.0` | `‖net_forces_w_history[:, 0, feet_ids]‖ > 2.0` |
| Previous contact | `‖net_forces_w_history[:, -1, body_ids]‖ > 2.0` (history buffer의 가장 오래된 entry) | `self._last_contacts[N, 4]` bool buffer (지난 step에서 저장된 값) |
| contact_filt | `contact_now OR contact_prev` | `contact_now OR last_contacts`, 직후 `last_contacts ← contact_now` |
| Edge mask 소스 | `self.x_edge_masks_tensor` (init 시 `terrain_generator_class.x_edge_maskes`에서 빌드) | `self.x_edge_mask` (init 시 `_build_edge_mask`로 빌드) |
| Grid 인덱싱 | `fx = round((x + rows_offset) / horizontal_scale)`, `fy = round((y + cols_offset) / horizontal_scale)`, clip(0, W-1)/(0, H-1) | `gx = round((xy − origin) * inv_scale − 0.5)`, clamp(0, dim-1) |
| 게이팅 | `(terrain_levels > 3) * Σ_foot(feet_at_edge)` | `(terrain_levels > 3).float() * Σ_foot(feet_at_edge.float())` |
| Return shape | `(N,)` | `(N,)` |

**핵심 차이**:
- **이전 step contact 정의 방식이 다름**:
  - A = sensor의 *history buffer 가장 오래된 슬라이스* (`history[:, -1]`) — sensor가 마지막 H step의 force를 들고 있으면 “현 step과 H step 전”의 OR.
  - B = *직전 step의 finalized contact 결과* (`_last_contacts` bool buffer) — physical하게 1 policy step 전.
- 두 방식이 정확히 같은 시점을 가리키지는 않음 (sensor의 history 길이와 policy step의 관계에 따라).
- Edge grid 인덱스 계산식의 offset/scale 표현이 다르지만 의도는 동일 (월드 XY → 그리드 셀).

---

### 4.12 `collision` (penalty)

| 항목 | A | B |
|---|---|---|
| Contact force 소스 | `contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]` `(N, B, 3)` (idx 0 = 최신 substep만) | `self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids]` `(N, hist=3, B, 3)` (전체 history) |
| History reduction | **idx 0만 사용** → 최신 1 substep | `max_{hist} ‖F‖` → history(3) 중 최댓값 |
| Threshold | `‖F‖ > 0.1` | `‖F‖ > 0.1` |
| 대상 body 집합 | cfg `["base", ".*_calf", ".*_thigh"]` (Go2 → 9 body) | `["base", ".*thigh", ".*calf", ".*hip", "Head_upper", "Head_lower"]` (Go2 → 더 많은 body) |
| 수식 | `Σ_b 1{‖F_b‖ > 0.1}` | `Σ_b 1{max_hist‖F_b‖ > 0.1}` |
| Return shape | `(N,)` int → float cast | `(N,)` float |

**핵심 차이**:
1. **History reduction이 다름** — A는 latest substep 단발, B는 3-substep max → B가 짧은 충돌 spike에 더 민감 (substep aliasing 회피).
2. **대상 body 집합이 다름** — A: base + thigh + calf. B: 위에 hip + Head_upper/Head_lower 추가. B는 머리 충돌까지 penalize.
3. Threshold(0.1 N)와 수식 형태(sum of indicators)는 동일.

---

## 5. B에만 있는 reward (계산 방식)

### 5.1 `tracking_lin_vel_xy_exp` (현재 scale 0)
- 데이터: `self._commands[:, :2]`, `self._robot.data.root_lin_vel_b[:, :2]` (**body frame**)
- 수식: `err = Σ_xy (cmd − v_b)²`; `r = exp(−err / tracking_sigma)`, `tracking_sigma=0.2`
- A와의 관계: A에는 body-frame 명령 추종이 없음 (A는 `tracking_goal_vel`이 world-frame goal 방향 투영만). 즉 B는 추가로 explicit body-frame velocity tracking 항을 보유 (현재 off).

### 5.2 `tracking_ang_vel_z_exp` (현재 scale 0)
- 데이터: `self._commands[:, 2]`, `self._robot.data.root_ang_vel_b[:, 2]`
- 수식: `r = exp(−(cmd − ω_b_z)² / tracking_sigma)`
- A와의 관계: A는 explicit yaw rate tracking 없음 (`tracking_yaw`는 yaw position 매칭).

### 5.3 `termination` (penalty, scale −100)
- 데이터: `_get_dones`에서 set되는 3 bool — `_term_base_contact` (base contact force max-hist > 5 N), `_term_tilt` (Σ projected_gravity_b[:,:2]² > 0.99), `_term_low_height` (root_link z < termination_height).
- 수식: `r = (term_base_contact | term_tilt | term_low_height).float()`
- 주의: `_get_dones`의 grace period(5 step) 마스크는 reward에는 **적용되지 않음** — grace 동안에도 `−100 × 0.02 = −2.0`/step 누적.
- A와의 관계: A에는 동등한 term이 없음 (A는 termination signal을 reward로 변환하지 않음).

### 5.4 `feet_dragging` (penalty, 현재 scale 0)
- 데이터: `self._robot.data.body_link_lin_vel_w[:, _feet_ids, :2]` (각 발의 world-XY link 속도), `feet_edge` 블록에서 만든 `contact_filt`, `cfg.dragging_velocity_threshold=0.05`.
- 수식: `Σ_feet (‖v_xy_foot‖ * contact_filt * (‖v_xy_foot‖ > 0.05))`
- 의미: 발이 접지(또는 직전 접지)된 상태에서 horizontal speed > 5cm/s면 그 속도를 penalize.
- A에는 없음.

### 5.5 `feet_air_time` (bonus, 현재 scale 0)
- 데이터: `contact_sensor.compute_first_contact(step_dt)[:, feet_ids]` (rising edge bool), `contact_sensor.data.last_air_time[:, feet_ids]` (이전 비행 시간 sec), `self._commands[:, :2]` (게이팅용).
- 수식: `Σ_feet ((last_air_time − 0.5) * first_contact_bool) × 1{‖cmd_xy‖>0.1} × is_flat`
- A에는 없음. anymal-c 스타일 trot inducer로, B만 보유 (현재 off).

### 5.6 `action_smoothness_1` / `_2` (penalty, 현재 scale 0)
- 데이터: `self._processed_actions`, `self._last_processed_actions`, `self._last_last_processed_actions` — 모두 `_pre_physics_step`에서 snapshot.
- `_1` 수식: `Σ_j ((a_t − a_{t−1})² * mask_{a_{t-1} != 0})`
- `_2` 수식: `Σ_j ((a_t − 2a_{t-1} + a_{t-2})² * mask1 * mask2)` (이산 2차 도함수)
- 마스크 `(a_{t-k} != 0)`는 reset 직후 spurious 큰 값 방지용.
- A에는 `reward_action_rate` 한 개만 존재. B는 1st difference(action_rate_l2) + 1st diff with mask(action_smoothness_1) + 2nd diff(action_smoothness_2) 세 종류를 분리해 둠.

### 5.7 `base_height` (penalty, flat-only, 현재 scale 0)
- 데이터: `self._robot.data.root_link_pos_w[:, 2]`, `self._terrain.env_origins[:, 2]`, `cfg.base_height_target=0.34`.
- 수식: `(z_base − z_origin − 0.34)² × is_flat`
- A에는 없음 (A는 base height 항을 등록하지 않음).

---

## 6. Sensor data 접근 패턴 — 정리

| 데이터 종류 | A 접근 경로 | B 접근 경로 |
|---|---|---|
| Root world lin vel | `env.scene["robot"].data.root_vel_w` (=root_lin_vel_w) | `self._robot.data.root_lin_vel_w` |
| Root body lin vel | `asset.data.root_lin_vel_b` | `self._robot.data.root_lin_vel_b` |
| Root body ang vel | `asset.data.root_ang_vel_b` | `self._robot.data.root_ang_vel_b` |
| Heading | `root_quat_w`에서 atan2 직접 계산 | `self._robot.data.heading_w` 직접 |
| Projected gravity | `asset.data.projected_gravity_b` | `self._robot.data.projected_gravity_b` |
| Joint pos/vel/torque/default | `asset.data.{joint_pos, joint_vel, applied_torque, default_joint_pos}` | `self._robot.data.{joint_pos, joint_vel, applied_torque, default_joint_pos}` |
| Foot world pos | `asset.data.body_state_w[:, body_ids, 0:2]` | `self._robot.data.body_pos_w[:, _feet_ids, :2]` |
| Foot link vel | (사용 안함) | `self._robot.data.body_link_lin_vel_w[:, _feet_ids, :2]` |
| Contact force history | `contact_sensor.data.net_forces_w_history` (idx 0 / idx -1만 사용) | 동일 attribute, **history max** (collision/termination) 또는 idx 0 (stumble/edge) 둘 다 사용 |
| Air-time helper | (사용 안함) | `contact_sensor.compute_first_contact(step_dt)`, `contact_sensor.data.last_air_time` |
| Goal context | `env.parkour_manager.get_term(...)` → `target_pos_rel`, `target_yaw`, `terrain_levels`, `env_per_terrain_name` 등 | env attribute: `_target_pos_rel`, `_target_yaw`, `_terrain_levels`, `_env_class` |
| Command | `env.command_manager.get_command('base_velocity')` | `self._commands` |
| Action | `env.action_manager.get_term('joint_pos').raw_actions` | `self._actions` / `self._processed_actions` |

**일관된 패턴**:
- A = manager pattern (string lookup, term 객체 경유). 모듈성/재사용성 ↑.
- B = direct attribute access (`self._*`). 구현이 간결하지만 환경 전용 attribute에 종속.

---

## 7. 양쪽이 수식적으로 동일한지 가장 헷갈리기 쉬운 곳

| 항목 | 양쪽 결과가 step별로 같은가? | 사유 |
|---|---|---|
| `tracking_yaw` | ✅ 동일 | A는 quat→atan2 직접, B는 IsaacLab pre-computed heading. IsaacLab의 `heading_w` 정의는 `atan2(2(wz+xy), 1-2(y²+z²))` 동일. wrap 미적용도 동일. |
| `tracking_goal_vel` | ⚠️ A는 cmd=0일 때 분모 1e-5, B는 `where(cmd>1e-3, r, 0)`. cmd≈0 환경에서 부호/스파이크 차이 가능. 그 외 동일. |
| `lin_vel_z`, `ang_vel_xy` | ⚠️ raw 수식 동일하지만 terrain modulation 계수가 다름 (§4.3, §4.4). |
| `orientation` | ✅ flat-only 동일 (zero out vs ×is_flat) |
| `dof_acc` | ✅ 동일. 양쪽 모두 reset 후 첫 step에서 acc spike 가능. |
| `torques`, `delta_torques` | ✅ 동일. |
| `hip_pos` | ⚠️ joint 집합 매칭 패턴이 다름. Go2에서는 결과 같을 가능성 있으나 정규식 수준에서는 다름. |
| `dof_error` | ⚠️ B는 flat-only 마스킹 추가, A는 unconditional. |
| `action_rate` | ✅ 양쪽 모두 L2 norm (squared 아님). 데이터 출처 다름. |
| `feet_stumble` | ✅ 동일. |
| `feet_edge` | ⚠️ contact_prev 정의가 다름 (history `[-1]` vs `_last_contacts` buffer). 시점이 일치하지 않음. |
| `collision` | ❌ History reduction과 body 집합이 모두 다름 (§4.12). |
| `termination` | — (A에 없음) |
| `feet_air_time`, `feet_dragging`, `tracking_*_exp`, `action_smoothness_*`, `base_height` | — (B 전용) |

---

## 8. 결론 (관찰 사실만)

1. **공통 12개 term은 거의 동일한 수식**이고, 차이의 대부분은:
   - **데이터 접근 패턴** (manager string-lookup vs direct attribute) — 결과 수치는 동일.
   - **Terrain modulation 방식** (terrain name 문자열 비교 vs `_env_class` int LUT) — 의미는 같지만 계수 자체는 다름 (`lin_vel_z`/`ang_vel_xy`).
   - **History 처리 방식** — `collision`은 A=latest only / B=max over history, `feet_edge`의 prev contact 정의도 sensor history vs persistent buffer로 다름.

2. **B는 anymal-c 계열 항(feet_air_time, dof tracking exp, base_height, action_smoothness) + termination penalty + feet_dragging을 추가**로 보유. 현재 대부분 scale=0으로 비활성화 상태.

3. **Scale 환산 시 주의**: B는 모든 term에 `step_dt=0.02`가 곱해진다 — 동일 숫자라도 per-step 영향은 A의 0.02배.

4. **Stateful term의 reset semantics**: A는 `ManagerTermBase.reset(env_ids)`로 각 term이 자율 관리, B는 `_reset_idx`에서 env attribute 일괄 초기화. 둘 다 reset 후 첫 step의 차분 항(dof_acc, action_rate, delta_torques)은 prev=0에서 시작하므로 `cur/dt` 또는 `‖cur‖` 만큼의 1-step spike 가능 (양쪽 공통 특성).

5. **Frame 일치**: tracking_goal_vel은 양쪽 모두 world-frame xy lin vel, tracking_yaw는 yaw position만 비교 (yaw rate 아님). `lin_vel_z`/`ang_vel_xy`/`orientation`은 양쪽 모두 body frame. 이 부분은 완전히 일치.

---

*end of comparison report*
