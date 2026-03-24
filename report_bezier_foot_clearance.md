# 보고서: Bezier Curve 기반 발 궤적 보상 구현

**작성일**: 2026-03-24
**대상 파일**:
- `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_env_cfg.py`

---

## 1. 배경 및 목표

### 문제 정의
Go2 로봇이 RL 학습 후 실제 하드웨어에 배포했을 때 발이 **쾅쾅 찍히는(stomping)** 현상이 발생.

**원인**: 기존 `feet_clearance_cmd_linear` 보상이 삼각파(triangular wave) 높이 프로파일을 사용하여, swing 중간에서 최대 높이에 도달한 뒤 빠르게 내려와야 하지만 착지 직전까지 높이를 강제함 → 발이 지면에 충격을 주며 착지.

```python
# 기존: 삼각파 기반 (foot_indices ∈ [0.5, 1.0) → swing)
phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
target_height = commands[:, 9].unsqueeze(1) * phases + 0.02
feet_clearance_cmd_linear = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
```

### 목표
3차 Bezier 곡선을 이용해 **부드러운 아크 궤적**을 정의하고, 발이 이 궤적을 추종하도록 보상을 설계.
- 기존 보상(`feet_clearance_cmd_linear`)은 비활성화(scale=0)
- 새 보상(`feet_clearance_bezier`)을 추가하여 x, y, z 3D 궤적 전체를 최적화

---

## 2. 핵심 개념 정리

### 2.1 foot_indices 구조

| 구간 | 의미 | desired_contact_states |
|------|------|----------------------|
| [0.0, 0.5) | stance | ≈ 1.0 (접지) |
| [0.5, 1.0) | swing | ≈ 0.0 (공중) |

swing 진행률(단조 증가): `s = clamp((foot_indices - 0.5) × 2.0, 0.0, 1.0)` → [0, 1]

### 2.2 발 순서 (foot order)
| 인덱스 | 발 |
|-------|-----|
| 0 | FL (Front Left) |
| 1 | FR (Front Right) |
| 2 | RL (Rear Left) |
| 3 | RR (Rear Right) |

### 2.3 3차 Bezier 곡선
```
B(s) = (1-s)³c0 + 3(1-s)²s·c1 + 3(1-s)s²·c2 + s³·c3
```
- `c0`: swing 시작 시 발 위치 (world frame)
- `c1`: c0 + height_cmd (이륙 가속)
- `c2`: c3 + height_cmd (착지 감속)
- `c3`: 착지 목표 위치 (world frame, z=0)

s=0.5일 때 높이:
```
B(0.5) ≈ 0.5·c0_z + 0.75·height_cmd
```
c0_z ≈ 0이면 B(0.5) ≈ 0.75 × height_cmd → 합리적인 아크 형성

---

## 3. 구현 내용

### 3.1 새 버퍼 추가 (`__init__`)

```python
# go2_wtw_env.py:74-77
self.swing_start_pos = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
self._prev_desired_contact_states = torch.ones(self.num_envs, 4, device=self.device, requires_grad=False)
self.swing_landing_target = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
self.bezier_target_pos = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
```

### 3.2 Config 파라미터 추가 (`go2_env_cfg.py`)

```python
feet_clearance_cmd_linear_reward_scale = 0.       # 기존 보상 비활성화
feet_clearance_bezier_reward_scale = -10.0         # Bezier 보상 활성화
```

### 3.3 Swing 시작 감지 및 착지 목표 계산 (`_get_rewards`)

#### Swing 시작 감지
```python
was_stance = self._prev_desired_contact_states > 0.5
is_swing = self.desired_contact_states < 0.5
transition_to_swing = was_stance & is_swing          # [num_envs, 4]
transition_mask_3d = transition_to_swing.unsqueeze(-1).expand_as(foot_positions)

# swing 시작 위치 저장
self.swing_start_pos = torch.where(transition_mask_3d, foot_positions, self.swing_start_pos)
```

#### 착지 목표(c3) 계산 — Raibert Heuristic 독립 구현

`desired_footsteps_world_frame`을 사용하지 않는 이유:
1. 해당 변수는 현재 phase(-0.5) 기준 Raibert 목표 → landing phase(+0.5) 기준이어야 함
2. `base_pos_현재` 기준 world 좌표 → 착지 시점의 base 위치는 `base_pos + base_vel × T_swing` 으로 예측해야 올바름

```python
# swing 전체 시간
T_swing = (1.0 - commands[:, 8].clamp(0.1, 0.9)) / freq_clamped

# 착지 시점 base 위치 예측
base_vel_w = self._robot.data.root_lin_vel_w
base_pos_at_landing = base_pos.clone()
base_pos_at_landing[:, :2] = base_pos[:, :2] + base_vel_w[:, :2] * T_swing.unsqueeze(1)

# body frame 착지 목표 (hip nominal + Raibert 보정)
hip_xs = torch.cat([s_len/2, s_len/2, -s_len/2, -s_len/2], dim=1)
hip_ys = torch.cat([s_wid/2, -s_wid/2, s_wid/2, -s_wid/2], dim=1)
x_corr = 0.5 * x_vel_des * (0.5 / freq_clamped.unsqueeze(1))
yaw_v = commands[:, 2:3]
y_corr = 0.5 * (yaw_v * s_len / 2.0) * (0.5 / freq_clamped.unsqueeze(1))
landing_xs = hip_xs + x_corr
landing_ys = hip_ys.clone()
landing_ys[:, 0:2] += y_corr   # front legs
landing_ys[:, 2:4] -= y_corr   # rear legs

# body → world frame 변환
land_w = quat_apply(q_land, land_body.reshape(-1, 3)).view(num_envs, 4, 3)
landing_target_new = land_w + base_pos_at_landing.unsqueeze(1)
landing_target_new[:, :, 2] = 0.0  # z = 지면

# swing 전환 시점에만 업데이트
self.swing_landing_target = torch.where(transition_mask_3d, landing_target_new, self.swing_landing_target)
```

#### Bezier 궤적 계산 및 보상

```python
s = torch.clamp((self.foot_indices - 0.5) * 2.0, 0.0, 1.0).unsqueeze(-1)  # [N, 4, 1]
height_cmd = commands[:, 9].unsqueeze(1).unsqueeze(1)                       # [N, 1, 1]

c0 = self.swing_start_pos    # [N, 4, 3]
c3 = self.swing_landing_target
c1 = c0.clone(); c1[:, :, 2:3] = c0[:, :, 2:3] + height_cmd
c2 = c3.clone(); c2[:, :, 2:3] = c3[:, :, 2:3] + height_cmd

bezier = (1-s)**3*c0 + 3*(1-s)**2*s*c1 + 3*(1-s)*s**2*c2 + s**3*c3

err_foot_3d = torch.norm(bezier - foot_positions, dim=-1)                   # [N, 4]
swing_mask = 1.0 - self.desired_contact_states
feet_clearance_bezier = torch.exp(-err_foot_3d**2 / tracking_sigma) * swing_mask
feet_clearance_bezier = torch.sum(feet_clearance_bezier, dim=1)
```

### 3.4 Debug 시각화

Orange sphere 시각화 (`_set_debug_vis_impl`, `_debug_vis_callback`):
- Stance 구간: 현재 발 위치 표시 (시각적 혼란 방지)
- Swing 구간: Bezier 목표 위치 표시

```python
is_stance = (self.desired_contact_states > 0.5).unsqueeze(-1)
self.bezier_target_pos = torch.where(is_stance, foot_positions, bezier.detach())
```

---

## 4. 발견된 버그 및 수정 내역

### 버그 1: Shape mismatch — s 브로드캐스팅
- **문제**: `s = [N, 4]`, `c0 = [N, 4, 3]` → 브로드캐스팅 불가
- **수정**: `s = swing_progress.unsqueeze(-1)` → `[N, 4, 1]`로 변경

### 버그 2: c1 z축 절대값 사용
- **문제**: `c1[:,:,2] = height_cmd` → 절대 높이(발 현재 z 무시)
- **수정**: `c1[:,:,2:3] = c0[:,:,2:3] + height_cmd` → 현재 발 높이에서 상대적 상승

### 버그 3: err_foot_3d shape 불일치
- **문제**: `bezier_z[N,4] - foot_positions[N,4,3]` → shape 오류
- **수정**: 3D 전체 norm 사용 `torch.norm(bezier - foot_positions, dim=-1)`

### 버그 4: variable name mismatch
- **문제**: `reward_tracking_3d`를 rewards dict에서 `feet_clearance_bezier` 키로 참조
- **수정**: 변수명을 `feet_clearance_bezier`로 통일

### 버그 5: World frame 초기화 오류 (Reward scale이 예상보다 큰 원인)
- **문제**: `swing_start_pos`와 `swing_landing_target`을 `(0, 0, 0)` (world origin)으로 초기화.
  로봇은 `env_spacing`에 따라 world frame에서 `(4, 8, 0.5)` 같은 위치에 배치되므로, reset 직후 late-stance 발의 swing_mask > 0일 때:
  ```
  err = (0,0,0) - (4, -0.175, 0.05) → err² ≈ 16 per foot
  Bezier 기여: -1 × 0.5 × 16 × 3dims × 0.02 ≈ -0.48/step
  Raibert 기여: -10 × 0.01 × 8 × 0.02 = -0.016/step  (30배 차이!)
  ```
  Raibert는 body frame(상대좌표)이라 world position에 영향받지 않지만, Bezier는 world frame 절대좌표이므로 terrain offset이 오차에 직접 포함됨.

- **수정** (`_reset_idx`): `default_root_state` 계산 이후로 초기화 이동, base world 위치(z=0)로 초기화
  ```python
  base_pos_reset = default_root_state[:, :3]                               # terrain origin 포함
  default_foot_xy = base_pos_reset.unsqueeze(1).expand(-1, 4, -1).clone()
  default_foot_xy[:, :, 2] = 0.0                                           # z = 지면
  self.swing_start_pos[env_ids] = default_foot_xy
  self.swing_landing_target[env_ids] = default_foot_xy
  ```
  초기 오차 감소: `|world_pos|` (4-8m) → `|base_pos - foot_pos|` (0.15-0.25m) → **약 300-1000배 개선**

---

## 5. Reward Scale 비교

| 보상 항목 | Scale | 오차 기준 | 값 범위 (정상) |
|-----------|-------|----------|----------------|
| `raibert_heuristic` | -10.0 (현재 0) | body frame x,y, 4발×2차원 | 0.0008–0.08 m² |
| `feet_clearance_cmd_linear` | 0.0 (비활성화) | z축만, swing 구간 | — |
| `feet_clearance_bezier` | -10.0 | world frame x,y,z 3D norm, swing 구간 | 수정 후 0.02–0.05 m² |

---

## 6. 최종 코드 상태

### go2_env_cfg.py
```python
feet_clearance_cmd_linear_reward_scale = 0.
feet_clearance_bezier_reward_scale = -10.0
```

### go2_wtw_env.py 주요 변경
- `__init__`: 4개 버퍼 추가 (swing_start_pos, _prev_desired_contact_states, swing_landing_target, bezier_target_pos)
- `_reset_idx`: base world 위치로 swing 버퍼 초기화 (default_root_state 계산 이후)
- `_get_rewards`: Bezier 보상 블록 추가 (540-624번째 줄)
- `_episode_sums`: `"feet_clearance_bezier"` 키 추가
- `_set_debug_vis_impl` / `_debug_vis_callback`: orange sphere 시각화 추가

---

## 7. 검증 방법

1. **학습 로그**: `Episode_Reward/feet_clearance_bezier` 에피소드 합계가 `raibert_heuristic`과 비슷한 크기인지 확인
2. **play.py 시각화**: orange sphere가 smooth arc를 그리는지 확인
3. **정성 평가**: 실제 하드웨어 배포 후 stomping 감소 확인
