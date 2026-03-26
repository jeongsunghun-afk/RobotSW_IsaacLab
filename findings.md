# Findings

---

## [현재 세션] Go2 AMP Discriminator 실패 원인 심층 분석 (2026-03-26)

### Loss 그래프 해석

| Loss | 수렴값 | 의미 |
|------|--------|------|
| disc_expert_loss | ~0.545 | discriminator가 expert 샘플을 "진짜"로 분류하는 자신감 낮음 |
| disc_policy_loss | ~0.545 | discriminator가 policy 샘플을 "가짜"로 분류하는 자신감 낮음 |

두 손실이 **동일한 값(~0.545)**에 수렴 → discriminator가 expert와 policy를 구분하지 못하는 상태.

이진 교차 엔트로피에서 p=e^(-0.545)≈0.58 → 모든 샘플에 58% 확률 할당 (랜덤에 가까움).

#### 두 가지 해석 경우

**경우 A (정상 수렴)**: policy가 discriminator를 속이는 데 성공 → expert ≈ policy 분포.
**경우 B (비정상 수렴)**: policy가 모션을 모방하지 않고 discriminator의 구조적 artifact를 exploit.

사용자 관찰(실제 모방 안 됨)과 task reward 상승 → **경우 B**가 확실.

---

### ★★★★★ 원인 1: toe velocity 좌표계 불일치 (신규 발견 - 가장 심각)

#### 코드 비교

**Policy AMP 관측** (`_get_observations`):
```python
rel_vel = body_lin_vel_w[:, key_indexes] - root_lin_vel_w.unsqueeze(1)  # 상대 속도
local_key_body_vel = quat_apply_inverse(root_quat, rel_vel)              # 로컬 프레임
```

**Expert AMP 관측** (`collect_reference_motions`):
```python
body_linear_velocities[:, self.motion_key_body_indexes]  # = toe_vel_local (raw from data)
```

#### motion loader에서 toe_vel_local의 의미

`go2_motion_loader.py` 주석:
```python
# [49:61]  toe_vel_local (발 4개 × 3, [FL, RL, FR, RR] 순서)
```

stmr_go2.py는 `toe_vel_local`을 **base frame에서 발의 절대 속도**로 저장 (루트 속도 미포함).

→ **Policy**: `quat_apply_inverse(quat, world_toe_vel - world_root_vel)` (상대 속도)
→ **Expert**: `quat_apply_inverse(quat, world_toe_vel)` (절대 속도)

차이 = `root_lin_vel_b` ≈ [1.0, 0, 0] m/s (보행 속도). **AMP obs 55차원 중 12차원이 체계적으로 오염.**

#### 영향

discriminator는 이 오프셋을 expert/policy를 구분하는 유일한 신호로 학습.
policy는 발을 실제로 움직이지 않고 루트 속도를 조절하는 방식으로 discriminator를 속임.
→ task reward로 속도 tracking은 성공하나 실제 보행 스타일 모방 실패.

---

### ★★★★☆ 원인 2: FrameDuration vs policy step_dt 불일치 (기존)

| 항목 | 값 |
|------|----|
| 데이터셋 FrameDuration | 0.01667s (≈60fps) |
| policy step_dt | 0.02s (50Hz) |
| 차이 | 1.2배 |

`collect_reference_motions`의 AMP obs pair 간격:
- Expert pair: 0.01667s 기준으로 계산된 속도값
- Policy pair: 0.02s 간격의 실제 시뮬레이션 속도

→ 같은 물리적 동작도 속도 분포가 다름. discriminator 학습 오염.

---

### ★★★☆☆ 원인 3: task reward vs AMP reward 불균형 (신규)

`go2_amp_env_cfg.py`:
```python
lin_vel_reward_scale = 1.0
yaw_rate_reward_scale = 0.5
```

RSL-RL AMP에서 style reward는 discriminator score로 계산됨. discriminator가 제대로 학습되지 않으면 style reward = 0.5 (랜덤).
task reward는 명확한 gradient를 제공 → policy는 task reward만 최적화.

→ reward 상승 = velocity tracking 향상. AMP 모방 X.

---

### ★★★☆☆ 원인 4: blend 클리핑 버그 (기존, 수정됨)

음수 시간으로 AMP obs 계산 시 외삽 발생 → discriminator 훈련 오염.
`go2_motion_loader.py`의 `_compute_frame_blend`에 `np.clip(blend, 0.0, 1.0)` 추가 완료.

---

### ★★☆☆☆ 원인 5: reset_strategy (기존, 수정됨)

`"default"` → `"random"` 변경 완료.

---

### 해결 방안 (우선순위 순)

#### 방안 1: Expert AMP obs에서 root 속도 빼기 ★★★★★ (가장 중요, 즉시 적용)

`collect_reference_motions`에서 toe velocity 계산 수정:

```python
# 현재 (버그):
body_linear_velocities[:, self.motion_key_body_indexes]  # 절대 속도

# 수정:
root_vel = body_linear_velocities[:, self.motion_ref_body_index].unsqueeze(1)  # (N,1,3)
toe_vel_relative = body_linear_velocities[:, self.motion_key_body_indexes] - root_vel  # (N,4,3)
# 그 다음 compute_obs에 toe_vel_relative 전달
```

OR stmr_go2.py에서 저장 방식 수정 (근본 해결).

#### 방안 2: 데이터셋 50fps 재생성 ★★★★★ (단기, 가장 근본적)

```bash
python stmr_go2.py --dt 0.02  # 50fps = policy step_dt
```

#### 방안 3: AMP reward weight 강화 ★★★☆☆

RSL-RL runner config에서:
```python
amp_reward_coef = 2.0  # 기본값보다 높게 (style reward 비중 증가)
```

---

### 권장 실행 순서

1. **즉시 (오늘)**: 방안 1 (toe velocity offset 수정) → 재학습 → loss 패턴 확인
2. **단기**: 방안 2 (50fps 데이터 재생성) → 재학습
3. **확인**: expert_loss와 policy_loss가 **다르게** 수렴하는지 확인 (expert_loss < policy_loss이어야 정상)

---

## [이전 세션] Go2 AMP 학습 결과 vs 레퍼런스 비디오 불일치 원인 분석 (2026-03-25)

### 핵심 수치 확인

| 항목 | 값 | 비고 |
|------|-----|------|
| 데이터셋 FrameDuration | 0.01667s (≈60fps) | stmr_go2.py `--dt` 인자 |
| 시뮬레이션 physics dt | 0.005s (200Hz) | go2_amp_env_cfg.py |
| policy step_dt | 0.02s (50Hz) | dt × decimation = 0.005 × 4 |
| dt 비율 (policy/data) | **1.20배** | 20% 차이 |

---

### 원인 1: FrameDuration vs policy step_dt 불일치 (★★★★☆ - 가장 중요)

`collect_reference_motions`에서 AMP obs pair 샘플링:
```python
times = current_times - step_dt * [0, 1]  # step_dt = 0.02s
```

- 레퍼런스: t, t-0.02s 두 시점의 obs를 구성
- 시뮬레이션: 0.02s 간격의 두 policy step obs

레퍼런스 모션의 **속도값은 0.01667s 간격 finite diff**로 계산:
```python
lin_vel = (root_pos_next - root_pos) / dt   # dt = 0.01667s
joint_vel = (joint_pos_next - joint_pos) / dt
```

시뮬레이션의 `root_lin_vel_b`는 Isaac Sim이 0.005s physics step으로 계산하는 **연속 물리 속도**.

→ 같은 물리적 움직임에서도 두 속도값의 시간 기준이 달라 discriminator가 비교하는 분포 자체가 달라짐.

---

### 원인 2: `_compute_frame_blend` 음수 blend 버그 (★★★☆☆)

`collect_reference_motions`에서:
```python
times = current_times - step_dt * [0, 1]
```

`current_times` 중 일부가 `step_dt(0.02s)` 미만일 때, `times[1]`이 **음수**가 됨.

`_compute_frame_blend`에서:
```python
phase = np.clip(times / self.duration, 0.0, 1.0)  # 음수 → 0으로 클리핑
index_0 = 0
blend = (times - 0) / self.dt  # 음수 그대로! → 외삽(extrapolation) 발생
```

→ 음수 시간의 AMP obs는 물리적으로 의미 없는 외삽값 → discriminator 훈련 오염

---

### 원인 3: `reset_strategy = "default"` 설정 (★★★☆☆)

`go2_amp_env_cfg.py`:
```python
reset_strategy = "default"  # ← R_Skeleton_amp는 "random"
```

- `"default"`: 기본 서있는 자세에서 시작 → 레퍼런스 모션과 초기 상태가 전혀 다름
- `"random"`: 레퍼런스 모션의 랜덤 지점에서 시작 → discriminator가 더 적절한 positive sample 봄

AMP 초기 학습에서 시뮬레이션 obs가 레퍼런스와 너무 달라 discriminator가 쉽게 구분 → task reward가 0에 수렴해도 AMP reward가 discriminator를 이기기 어려움.

---

### 원인 4: TMR 적용 후 FrameDuration 불일치 가능성 (★★☆☆☆)

`stmr_go2.py`에서 TMR(Temporal Motion Retargeting) 적용:
```python
final_motion = tmr_optimizer.resample_motion(smr_q_traj, best_alpha)
dataset_features = compute_dataset_features(model, data, final_motion, dt_source)
output_motion(dataset_features, txt_output_path, dt_source)
```

TMR이 `best_alpha != 1.0`으로 모션을 시간 스케일링했으나, 속도 계산은 여전히 원본 `dt_source`로 함.
→ 실제 모션 속도와 저장된 속도값 불일치 가능성 (alpha에 따라 다름).

---

### 해결 방안 (우선순위 순)

#### 방안 A: 데이터셋 재생성 - 50fps로 맞추기 ★★★★★ (권장)

```bash
cd /home/lgb/Dog_Motion_data_3D/mujoco_retarget_go2
python stmr_go2.py --dt 0.02 ...   # 50fps = policy step_dt와 정확히 일치
```

- 효과: 속도 스케일 완벽 일치, FrameDuration = step_dt
- 주의: 기존 60fps 데이터를 50fps로 리샘플링하는 것과 다름 (새로 retargeting 필요)

#### 방안 B: blend 클리핑 버그 수정 ★★★★☆ (즉시 적용 가능)

[go2_amp/go2_motion_loader.py](source/isaaclab_tasks/isaaclab_tasks/direct/go2_amp/go2_motion_loader.py)의 `_compute_frame_blend` 수정:

```python
def _compute_frame_blend(self, times: np.ndarray):
    phase = np.clip(times / self.duration, 0.0, 1.0)
    index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
    index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
    blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
    blend = np.clip(blend, 0.0, 1.0)  # ← 이 줄 추가
    return index_0, index_1, blend
```

#### 방안 C: reset_strategy = "random" 변경 ★★★☆☆ (즉시 적용 가능)

[go2_amp/go2_amp_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/go2_amp/go2_amp_env_cfg.py):
```python
reset_strategy = "random"  # "default" → "random"
```

#### 방안 D: decimation 조정으로 policy step을 FrameDuration에 맞추기 ★★☆☆☆

```python
# go2_amp_env_cfg.py
sim: SimulationCfg = SimulationCfg(dt=1/200, ...)
decimation = 3  # 0.005 × 3 = 0.015s ← 0.01667s에 더 가깝지만 완벽하지 않음
# 또는
sim: SimulationCfg = SimulationCfg(dt=1/60, ...)
decimation = 1  # 정확히 0.01667s ← 성능 저하 우려
```

#### 방안 E: 모션 로더에서 속도 재계산 ★★☆☆☆ (임시방편)

저장된 velocity를 버리고 `body_positions`로부터 재계산:
- 단, 이미 보간된 body_positions에서의 finite diff이므로 원래보다 부드러워짐
- 근본 해결책은 아님

---

### 권장 실행 순서

1. **즉시**: 방안 B (blend 클리핑) + 방안 C (reset_strategy=random) 적용 후 재학습
2. **단기**: 방안 A (50fps 데이터 재생성) 후 재학습
3. **확인**: 재학습 후 비디오와 비교

---

## [현재 세션] Deploy 시 발 착지 충격 (Foot Impact) 원인 분석 (2026-03-26)

### 핵심 문제

실제 하드웨어 deploy 시 발이 바닥에 쾅쾅 부딪히는 현상.
학습 환경(`go2_wtw_env.py`)에서 착지 충격을 직접 패널티하는 메커니즘이 부재함.

---

### 원인 1: Bezier c2 제어점의 높은 z값 → 착지 직전 급하강

**코드 위치**: `go2_wtw_env.py` L607~L612

```python
c1 = c0.clone()
c1[:, :, 2:3] = height_cmd   # c1 z = height_cmd (예: 0.1m)
c2 = c3.clone()
c2[:, :, 2:3] = height_cmd   # c2 z = height_cmd ← 문제
# c3 z = 0.0 (지면)
```

3차 Bezier 착지 직전 z속도:
```
dP/ds|s=1 = 3*(c3 - c2) = 3*(0 - height_cmd)
```

height_cmd = 0.1m → 착지 시 z방향 속도 크기 ≈ 0.3 (normalized).
→ **발이 마지막 구간에서 수직 낙하**

**해결**: c2[:, :, 2:3]을 `height_cmd * 0.1` 수준으로 낮춰 착지 전 지면 근접 유지

---

### 원인 2: 착지 순간 z속도 페널티 부재

`tracking_contacts_shaped_vel`은 stance 구간 **전체** 발속도 페널티.
착지 **순간**(첫 접촉, swing→stance 전환)의 z방향 속도에 대한 별도 패널티 없음.

**관련 API**: `self._contact_sensor.compute_first_contact(self.step_dt)` → 첫 접촉 스텝 감지

---

### 원인 3: 충격 force 모니터링 없음

`tracking_contacts_shaped_force`는 swing 중 접촉 force를 패널티.
착지 시 충격 force(impact peak)를 패널티하지 않음.

---

### 해결 방안 (우선순위 순)

#### 방안 A: Bezier c2 z값 수정 ★★★★★

**수정 위치**: `go2_wtw_env.py` L610

```python
# 현재:
c2[:, :, 2:3] = height_cmd

# 수정 (착지 전 지면 근접):
c2[:, :, 2:3] = height_cmd * 0.1   # 10%만 유지 → 완만한 하강
```

효과: 발이 swing 후반부에 이미 지면에 가까워지므로 착지 충격 최소화.
재학습 없이 즉시 적용 가능 (궤적 형상 변경, bezier_reward는 새 c2 기준 재계산됨).

---

#### 방안 B: Foot Touchdown z-velocity Penalty 추가 ★★★★☆

**수정 위치**: `go2_wtw_env.py` `_get_rewards()` 내부 + `go2_env_cfg.py`

```python
# _get_rewards() 내부에 추가:
first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
foot_z_vel = self._robot.data.body_link_lin_vel_w[:, self._feet_ids, 2]  # [num_envs, 4]
foot_landing_vel = torch.sum(torch.square(foot_z_vel) * first_contact, dim=1)

# rewards dict에 추가:
"foot_landing_vel": foot_landing_vel * self.cfg.foot_landing_vel_reward_scale * self.step_dt,
```

```python
# go2_env_cfg.py에 추가:
foot_landing_vel_reward_scale = -3.0   # 음수 = 패널티
```

```python
# __init__의 _episode_sums에도 "foot_landing_vel" 추가 필요
```

효과: 발이 빠르게 내려오는 행동에 직접 패널티 → policy가 soft landing 학습.

---

#### 방안 C: Impact Force Penalty 추가 ★★★☆☆ (보조)

```python
# 첫 접촉 스텝의 force 크기 패널티
first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
foot_impact_forces = torch.sum(foot_forces * first_contact, dim=1)

# rewards dict에 추가:
"foot_impact_force": foot_impact_forces * self.cfg.foot_impact_force_reward_scale * self.step_dt,
```

주의: `gait_force_sigma=100`인 기존 force reward가 일부 커버하므로 중복 고려.

---

### 권장 실행 순서

1. **즉시**: 방안 A (c2 z값 수정) → 재학습 없이도 부분 효과
2. **재학습 포함**: 방안 A + 방안 B → 가장 강력한 soft landing 학습
3. **확인**: deploy 후 foot contact force 측정으로 개선 검증

---

## [현재 세션] Bezier Swing Start 편향 문제 분석 (2026-03-26)

### 문제 요약

Bezier curve 기반 foot trajectory reward가 raibert_heuristic보다 velocity tracking이 낮은 이유:
swing 시작점(c0)이 nominal hip 위치에 고정되어 있어 전체 궤적이 앞으로 편향됨.

### 현재 c0, c3 계산 분석

**코드 위치**: `go2_wtw_env.py` line 585~607

| 제어점 | x 계산 | 의미 |
|--------|--------|------|
| c0 (liftoff) | `hip_xs` | nominal hip 위치 (보정 없음) |
| c3 (landing) | `hip_xs + x_corr` | Raibert 보정: 앞으로 `x_corr` |

x_corr 정의:
```python
x_corr = 0.5 * x_vel_des * (0.5 / freq_clamped)  # = x_vel * T_period / 4
```

결과적으로 궤적 중심 ≈ `hip + x_corr/2` → **앞으로 치우침**

### 문제 메커니즘

Raibert heuristic의 철학:
- 발이 hip 전방 `+x_corr`에 착지하고, hip 후방 `+x_corr`에서 이륙
- 이렇게 하면 stance phase 동안 hip이 발 위를 지나가며 대칭적으로 지지력 생성

현재 Bezier:
- 이륙: hip_nominal (보정 없음)
- 착지: hip + x_corr
- → 발이 항상 앞쪽에만 있게 됨 → 뒤에서 밀어주는 힘 부족 → velocity tracking 손실

### 제안 수정

c0를 `hip_xs - x_corr`로 변경하여 궤적을 hip 중심으로 대칭화:

```python
# 현재:
c0_body[:, :, 0] = hip_xs
c0_body[:, :, 1] = hip_ys

# 수정:
c0_body[:, :, 0] = hip_xs - x_corr          # x: 뒤로 x_corr
start_ys = hip_ys.clone()
start_ys[:, 0:2] = start_ys[:, 0:2] - y_corr  # front: -y_corr (landing +y_corr와 대칭)
start_ys[:, 2:4] = start_ys[:, 2:4] + y_corr  # rear:  +y_corr (landing -y_corr와 대칭)
c0_body[:, :, 1] = start_ys
```

수정 후 궤적 중심 = `(hip - x_corr + hip + x_corr) / 2 = hip` ✓

### 수치 검증

전형적인 파라미터:
- v_x = 1.0 m/s, freq = 2.0 Hz → T_period = 0.5s
- x_corr = 0.5 * 1.0 * (0.5/2.0) = 0.125 m
- 현재: c0 = hip+0, c3 = hip+0.125 → 중심 hip+0.0625
- 수정: c0 = hip-0.125, c3 = hip+0.125 → 중심 hip ✓

### 기대 효과

1. **velocity tracking 개선**: 발이 hip 후방에서 이륙 → 전방으로 지면 반력 생성 가능
2. **Raibert heuristic과 일관성**: foot placement 패턴이 Raibert와 동일한 대칭 구조
3. **c1, c2(높이 제어점)도 자동 보정**: c1=c0, c2=c3 기반이므로 수정 즉시 반영

---

## [이전 세션] Go2 WTW 중간 결과 문제 분석

### 문제 1: 명령 tracking 불안정

**[1-A] raibert_heuristic 완전 비활성화**
```python
# go2_env_cfg.py:172
raibert_heuristic_reward_scale = 0.
```
발 착지 위치를 명시적으로 유도하는 가장 직접적인 reward가 꺼져 있음.

**[1-B] tracking_sigma가 너무 작음**
```python
tracking_sigma = 0.25
```
`tracking_lin_vel = exp(-error / 0.25)`
→ error = 0.5 m/s 이면 reward = exp(-2) ≈ 0.135 (거의 0)
→ 학습 초기에 tracking error가 크면 gradient signal 자체가 사라짐

**[1-C] Bezier reward의 큰 패널티가 tracking을 방해**
```python
feet_clearance_bezier_reward_scale = -10.0
```

**[1-D] 명령 resampling 주기**
```python
sample_interval = int(self.cfg.resampling_time / self.dt)
```

### 수정 방향

| 항목 | 현재 값 | 제안 값 | 이유 |
|------|---------|---------|------|
| `raibert_heuristic_reward_scale` | 0. | -1.0 ~ -5.0 | 발 위치 직접 유도 |
| `tracking_sigma` | 0.25 | 0.5 ~ 1.0 | 초기 학습 gradient 확보 |
| `feet_clearance_bezier_reward_scale` | -10.0 | -1.0 ~ -3.0 | neg reward 억제 완화 |
| `lin_vel_reward_scale` | 1.2 | 1.5 ~ 2.0 | tracking 상대적 중요도 상승 |

---

### 문제 2: Bezier front feet target이 멀어 보임

현재 Bezier c0, c3 계산:
```
c0_world = quat_apply(base_quat, [±L/2, ±W/2, 0]) + base_pos_current
c3_world = quat_apply(base_quat, [±L/2 + x_corr, ±W/2 + y_corr, 0]) + base_pos_at_landing
x_corr = 0.5 * x_vel_des * (0.5 / freq)          # 모든 발 동일
base_pos_at_landing = base_pos + base_vel * T_swing  # T_swing = (1-dur)/freq
```

T_swing이 모든 발에 동일하게 적용되어 잘못된 target 계산 발생.

수정 방향:
```python
remaining_phase = 1.0 - self.foot_indices  # [num_envs, 4]
remaining_phase = remaining_phase.clamp(0.0, 0.5)
T_swing_per_foot = remaining_phase / freq_clamped.unsqueeze(1)  # [num_envs, 4]
```

---

### 문제 3: Deploy 시 부드러운 보행 방법

추천 순서:
1. **즉시**: Action EMA 필터 (alpha=0.8)
2. **단기**: 학습에 EMA 포함 + dof_vel scale 강화
3. **장기**: Jerk reward 추가 + Kd 튜닝
