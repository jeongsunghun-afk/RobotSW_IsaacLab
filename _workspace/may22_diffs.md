# May 22 Parkour — 코드 / Reward Scale / Hyperparameter 변경사항

> 비교 대상 3개 run
> - **A (baseline)**: `2026-05-21_15-48-19_feet_dragging` (May 21 series 최고 성능 24.88)
> - **B**: `2026-05-22_17-02-57_change_collision_candidate`
> - **C**: `2026-05-22_17-35-28_change_spot_trot`
>
> 데이터 소스: 각 run dir의 `params/{env,agent}.yaml` 및 `git/IsaacLab.diff` 실제 비교 결과.

---

## Baseline: May 21 feet_dragging (15-48-19)

### Reward Scales (env.yaml `reward_scales:` 섹션 직접 추출)

| Term              | weight   |
|-------------------|----------|
| tracking_goal_vel | 1.5      |
| tracking_yaw      | 0.5      |
| lin_vel_z_l2      | -1.0     |
| ang_vel_xy_l2     | -0.05    |
| orientation_l2    | -1.0     |
| dof_acc_l2        | -2.5e-07 |
| collision         | -10.0    |
| action_rate_l2    | -0.05    |
| delta_torques     | -1.0e-07 |
| torques_l2        | -1.0e-05 |
| hip_pos           | -0.5     |
| dof_error_l2      | -0.04    |
| feet_stumble      | -1.0     |
| feet_edge         | -1.0    |
| feet_dragging     | -0.1     |

### 관련 env cfg 파라미터 (변동 추적용)
```yaml
tracking_sigma: 0.2
dragging_velocity_threshold: 0.05
base_height_target: 0.34
max_tilt: 1.309
termination_height: -0.2
termination_grace_steps: 5
_actuator_mode: 2
clip_actions: 10.0
```

### Algorithm Hyperparameters (agent.yaml 직접 추출)

| Param                | value             |
|----------------------|-------------------|
| algorithm.class_name | PPO               |
| learning_rate        | 0.0002            |
| gamma                | 0.99              |
| lam                  | 0.95              |
| entropy_coef         | 0.01              |
| clip_param           | 0.2               |
| value_loss_coef      | 1.0               |
| max_grad_norm        | 1.0               |
| num_learning_epochs  | 5                 |
| num_mini_batches     | 4                 |
| actor_hidden_dims    | [512, 256, 128]   |
| critic_hidden_dims   | [512, 256, 128]   |
| activation           | elu               |
| init_noise_std       | 1.0               |
| num_steps_per_env    | 24                |
| max_iterations       | 50000             |
| save_interval        | 100               |
| empirical_normalization | false          |

### Code state (`git/IsaacLab.diff` 요약, origin/main 대비)
- `parkour_env.py`: feet_dragging 추가, hind feet ID 어서션, undesired contact list = `["base", ".*thigh", ".*calf", ".*hip", "Head_upper", "Head_lower"]` (calf 포함).
- `parkour_env_cfg.py`: `feet_dragging: -0.1`, `dragging_velocity_threshold: 0.05`, `max_tilt: 1.309`.
- 기타 누적 working-tree drift: `go2_imitation` 환경(AMP 43-dim × 10-history 구성), Go2_4.5/5.1 asset 등 — parkour 실험과 무관.

---

## B: 05-22 `change_collision_candidate` vs A (baseline)

### 변경된 Reward Scales

| Term               | A        | B        | Δ                  |
|--------------------|----------|----------|--------------------|
| feet_gait_pairing  | (없음)   | **0.0**  | **신규 항목 추가, 비활성** |

> 다른 모든 reward weight는 동일.

### 변경된 Env cfg

| Field                          | A      | B      | 의미 |
|--------------------------------|--------|--------|------|
| feet_gait_std                  | (없음) | **0.2** | 신규: gait pairing reward의 error tolerance |
| feet_gait_max_err              | (없음) | **0.3** | 신규: squared-error clamp 상한 |
| feet_gait_velocity_threshold   | (없음) | **0.3** | 신규: cmd_speed gate (m/s) |

> `clip_actions`, `max_tilt`, `dragging_velocity_threshold` 등 다른 모든 env cfg 동일.

### 변경된 Hyperparameters

**없음.** agent.yaml 의 알고리즘/네트워크 파라미터는 A와 100% 동일 (`run_name`만 다름).

### 변경된 Code (git diff delta, parkour 영역만)

**1) `parkour_env.py` — undesired contact list에서 `.*calf` 제거 (run name 기원):**
```diff
 self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(
-    ["base", ".*thigh", ".*calf", ".*hip", "Head_upper", "Head_lower"]
+    ["base", ".*thigh", ".*hip", "Head_upper", "Head_lower"]
+    # ["base", ".*thigh", ".*calf", ".*hip", "Head_upper", "Head_lower"]
 )
```
→ **collision penalty 트리거 바디 후보에서 4개 calf (FL/FR/RL/RR_calf)가 빠짐.** `collision = -10.0` reward의 active 영역 축소. (run 이름 "change_collision_candidate"의 정확한 의미.)

**2) `parkour_env.py` `__init__` — diagonal gait pair ID 캐시 추가:**
```diff
+ # Gait pairing — diagonal pairs (FL+RR, FR+RL) using URDF order [FL(0), FR(1), RL(2), RR(3)]
+ self._gait_synced_pair_0 = (self._feet_ids[0], self._feet_ids[3])  # FL + RR
+ self._gait_synced_pair_1 = (self._feet_ids[1], self._feet_ids[2])  # FR + RL
```

**3) `parkour_env.py` `_get_rewards()` — Spot GaitReward `sync_only` 구현 추가 (scale=0 이므로 dict 키만 생성):**
```python
air_time     = self._contact_sensor.data.current_air_time
contact_time = self._contact_sensor.data.current_contact_time
std      = self.cfg.feet_gait_std        # 0.2
max_err2 = self.cfg.feet_gait_max_err ** 2  # 0.09

def _sync_pair(pair):
    a, b = pair
    se_air     = torch.clamp((air_time[:, a]     - air_time[:, b])     ** 2, max=max_err2)
    se_contact = torch.clamp((contact_time[:, a] - contact_time[:, b]) ** 2, max=max_err2)
    return torch.exp(-(se_air + se_contact) / std)

sync_reward = _sync_pair(self._gait_synced_pair_0) * _sync_pair(self._gait_synced_pair_1)
gate = (torch.norm(self._commands[:, :2], dim=-1) > 0.3).float()
feet_gait_pairing = sync_reward * gate * is_flat
# rewards dict 등록:
"feet_gait_pairing": feet_gait_pairing,
```
→ scale=0이므로 학습 신호로는 비활성. 단순히 인프라/메트릭만 추가됨 (로깅 목적 추정).

**4) `parkour_env_cfg.py` — reward_scales + gait 파라미터 추가:**
```diff
 reward_scales = {
   ...
   "feet_dragging": -0.1,
+  "feet_gait_pairing": 0.,   # Spot GaitReward sync-only: trot 대각 쌍 phase 동기 (양수 only)
 }
+
+# Gait pairing reward parameters (Spot GaitReward style, sync-only)
+feet_gait_std: float = 0.2
+feet_gait_max_err: float = 0.3
+feet_gait_velocity_threshold: float = 0.3
```

**5) 무관 변경 (parkour 학습에 영향 없음):**
- `go2_imitation/agents/rsl_rl_amp_cfg.py`: `clip_actions 1.0 → 4.0`, `amp_observation_space 430 → 98`.
- `go2_imitation/go2_imitation_env.py`: AMP disc obs를 43-dim×10 history → 49-dim×2 history (R4 ablation, root_rot_tan_norm 6D 추가).
- `go2_imitation/go2_imitation_env_cfg.py`: `MOTION_FILES_DIR`을 `imitation/go2` → `imitation/smr_mirror_pkl`로 교체.
- B는 추가로 `_workspace/amp_compare/`, `_workspace/plans/parkour_gait_pairing_plan.md`, `_workspace/plans/parkour_trot_gait_shaping_plan.md` untracked 파일들이 working tree에 등장.
- Git 상태: A는 origin/main과 동기, B는 "ahead of origin/main by 1 commit".

### 의도 해석

run 이름 "change_collision_candidate"의 실제 변경은 **`parkour_env.py:_undesired_contact_body_ids`의 정규식 패턴 리스트에서 `.*calf`를 제거**한 것. 즉 collision penalty(-10.0)가 트리거되는 body 후보 집합을 좁힌 것이다. 본 변경은 Go2 4개 calf 링크가 더 이상 collision penalty를 받지 않도록 함 → parkour 장애물에서 calf 충돌이 자주 발생하는 stair/step terrain 학습 신호 완화를 의도한 것으로 보임.

부수적으로 **feet_gait_pairing 인프라(코드+cfg+reward_scales 키)가 함께 추가되었으나 scale=0** 이라서 학습 신호로는 작동하지 않음. C 실험을 위한 준비 단계로 보임 (즉 본 run은 collision list 축소 단독 효과 + gait pairing 로깅만).

go2_imitation 측 변경은 별개의 AMP ablation(R4: tan_norm 6D 추가)으로, 본 parkour run의 학습에는 사용되지 않음 — diff에 같이 올라온 working-tree drift일 뿐.

---

## C: 05-22 `change_spot_trot` vs A (baseline)

### 변경된 Reward Scales

| Term               | A        | C        | Δ                |
|--------------------|----------|----------|------------------|
| feet_gait_pairing  | (없음)   | **0.5**  | **신규 항목, 활성** |

> 다른 모든 reward weight는 동일.

### 변경된 Env cfg

| Field                          | A      | C      | 의미 |
|--------------------------------|--------|--------|------|
| feet_gait_std                  | (없음) | **0.2** | 신규: gait pairing error tolerance |
| feet_gait_max_err              | (없음) | **0.3** | 신규: squared-error clamp 상한 |
| feet_gait_velocity_threshold   | (없음) | **0.3** | 신규: cmd_speed gate |

### 변경된 Hyperparameters

**없음.** agent.yaml 은 A/B/C 모두 동일 (`run_name`만 다름).

### 변경된 Code (git diff delta, parkour 영역만)

C는 B의 모든 코드 변경을 포함하며, 추가로 **`feet_gait_pairing` reward에 Spot 원본의 4-term async 곱셈 항이 복원**됨:

```diff
 sync_reward = _sync_pair(self._gait_synced_pair_0) * _sync_pair(self._gait_synced_pair_1)
+
+def _async_pair(a, b):  # a, b: single foot indices (NOT pair tuples)
+    se_0 = torch.clamp((air_time[:, a]     - contact_time[:, b]) ** 2, max=max_err2)
+    se_1 = torch.clamp((contact_time[:, a] - air_time[:, b])     ** 2, max=max_err2)
+    return torch.exp(-(se_0 + se_1) / std)
+
+# Spot 원본 async 4항 — 다른 대각 쌍 소속 발의 cross 비교 (정지/끌기 false-positive 차단)
+p0_0, p0_1 = self._gait_synced_pair_0   # (FL, RR)
+p1_0, p1_1 = self._gait_synced_pair_1   # (FR, RL)
+async_reward = (
+    _async_pair(p0_0, p1_0) *   # FL vs FR
+    _async_pair(p0_1, p1_1) *   # RR vs RL
+    _async_pair(p0_0, p1_1) *   # FL vs RL
+    _async_pair(p0_1, p1_0)     # RR vs FR
+)
+
 gate = (torch.norm(self._commands[:, :2], dim=-1) > 0.3).float()
-feet_gait_pairing = sync_reward * gate * is_flat
+feet_gait_pairing = sync_reward * async_reward * gate * is_flat
```

그리고 `parkour_env_cfg.py`의 weight:
```diff
- "feet_gait_pairing": 0.,
+ "feet_gait_pairing": 0.5,
```

**B와 동일하게 포함된 변경:**
- `parkour_env.py` undesired contact list에서 `.*calf` 제거 (즉 C도 collision_candidate 축소 적용).
- gait pair ID 캐시 추가.
- `feet_gait_std/max_err/velocity_threshold` cfg 추가.
- B와 동일하게 go2_imitation 측 R4 ablation 코드도 working tree에 존재 (parkour 학습엔 무관).
- `_workspace/plans/parkour_gait_pairing_async_restore.md` 추가됨.

### 의도 해석

run 이름 "change_spot_trot"의 실제 변경은 **AMP reference motion 교체가 아님** — `go2_imitation` 측 `MOTION_FILES_DIR`이 `smr_mirror_pkl`로 바뀐 변경은 working tree drift일 뿐 parkour 환경에서는 사용되지 않음 (parkour run은 PPO 학습이며 AMP discriminator를 쓰지 않음, env.yaml에 motion_file 관련 필드 없음).

실제 의미는 **"Spot의 trot gait reward 형식을 채택"** — 즉 Boston Dynamics Spot의 `GaitReward`(`sync × async × gate × is_flat`)를 4족 trot 동기화 보상으로 도입한 것. 대각선 쌍 (FL+RR), (FR+RL)이 air/contact 시간을 동기화하도록, 동시에 다른 대각 쌍과는 phase가 어긋나도록 보상.

scale **+0.5**(B의 0.0에서 활성화)는 May 21 Exp 7 `feet_air_time +0.5` 망가짐 사례와 동일 magnitude. 단 본 reward는:
- `is_flat` 게이트 (flat terrain만)
- `cmd_speed > 0.3` 게이트
- 모든 항이 `exp(-x)` 형태로 **항상 양수 [0, 1]**

따라서 Exp 7의 "+0.5 의도가 음수로 작동" 함정은 구조적으로 제거되어 있음. 단 [추정] flat terrain에서만 보상되는 점은 동일하므로 curriculum 진행 시 hurdle/step terrain으로 넘어가면 신호 끊김 가능성 있음.

---

## C vs B (두 May 22 실험 직접 비교)

`diff B/env.yaml C/env.yaml` 와 `diff B/IsaacLab.diff C/IsaacLab.diff` 의 실제 결과 요약:

| 카테고리 | B | C | 차이 |
|----------|---|---|------|
| `reward_scales.feet_gait_pairing` | **0.0** | **0.5** | C가 활성화 |
| `feet_gait_pairing` 식 | `sync_reward * gate * is_flat` | `sync_reward * async_reward * gate * is_flat` | C가 4-term async 곱셈 복원 |
| undesired contact list (`.*calf` 제거) | ✓ | ✓ | 동일 |
| `feet_gait_std/max_err/velocity_threshold` 신규 | 0.2 / 0.3 / 0.3 | 0.2 / 0.3 / 0.3 | 동일 |
| `_undesired_contact_body_ids` | calf 빠짐 | calf 빠짐 | 동일 |
| go2_imitation R4 ablation (parkour 무관) | ✓ | ✓ | 동일 |
| Git 상태 | ahead by 1 commit | ahead by 1 commit | 동일 |
| Algorithm hyperparam | A와 동일 | A와 동일 | 동일 |
| Plans 디렉토리 | `parkour_gait_pairing_plan.md`, `parkour_trot_gait_shaping_plan.md` | + `parkour_gait_pairing_async_restore.md` | C만 async_restore 계획 추가 |

→ **C는 B의 strict superset.** B = sync-only @ scale 0 (인프라만), C = sync × async @ scale 0.5 (실제 활성).

---

## 종합 (변경 축 분류)

| 축 | A | B | C |
|----|---|---|---|
| **Collision candidate (calf 포함 여부)** | calf 포함 | **calf 제거** | **calf 제거** |
| **feet_gait_pairing scale** | 없음 | 0.0 (코드만) | **0.5 (활성)** |
| **gait 식 (sync vs sync×async)** | N/A | sync-only | **sync × async (Spot 원본)** |
| feet_dragging | -0.1 | -0.1 | -0.1 |
| 그 외 reward / cfg / hyperparam | (baseline) | A와 동일 | A와 동일 |

- **B = collision candidate 축 단독 변경** (calf 빠짐 + gait 인프라는 비활성 로깅용).
- **C = collision candidate 축 + Spot GaitReward 축의 동시 변경** (sync×async @ +0.5).
- **공통 변경 = `.*calf`를 undesired contact list에서 제거 + gait pairing 코드/cfg infrastructure (B/C 모두 적용).**
- **변경되지 않은 축**: 모든 PPO 하이퍼파라미터(lr, clip, entropy, batch, hidden), terrain cfg, observation cfg, action_rate/feet_dragging/feet_stumble/feet_edge weight, max_tilt, clip_actions.

## 주의 사항 (project memory 제약 점검)

- B/C 모두 contact sensor 기반 새 reward(`feet_gait_pairing`)를 추가했으나 **이는 reward signal 입력일 뿐 observation 추가가 아님** — sim-to-real 금지 사항인 "contact sensor obs 추가"에는 해당하지 않음.
- B/C 모두 `_actuator_mode: 2` 유지 (변경 없음).
- May 21 Exp 6 `feet_dragging: -0.1` 은 A/B/C 전부 유지 (project memory에 따르면 net positive이므로 유지 정상).

## 자기 검증
- [x] params/*.yaml diff는 실제 `diff` 명령 결과 (추측 아님; 출력 상단 4개 명령 block 참조)
- [x] git/IsaacLab.diff는 실제 파일 내용 (각 run dir의)
- [x] reward weight 표는 env.yaml의 `reward_scales:` 섹션에서 직접 추출 (line 1016-1031 of A, +1 line for B/C)
- [x] 의도 해석은 git diff + yaml diff 증거에 근거 ([추정] 표기는 flat-gate 신호 끊김 가능성 한 곳에만 사용)
