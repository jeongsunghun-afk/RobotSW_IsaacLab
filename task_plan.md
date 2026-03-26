# Task Plan: Go2 WTW 발 착지 부드럽게 만들기 (Soft Landing)

## 목표
실제 deploy 시 Go2 로봇의 발이 바닥에 세게 충격을 주지 않고
최대한 부드럽게 착지(soft landing)하도록 학습 환경을 개선.

## 전체 상태: 계획 수립 완료 / 구현 대기

---

## 문제 진단

현재 코드(`go2_wtw_env.py`)에서 발이 쾅쾅 거리는 원인:

### 원인 1: Bezier 궤적의 급격한 하강
- c2 (착지 직전 제어점) z = `height_cmd` (예: 0.1m)
- c3 (착지점) z = 0.0 (지면)
- 착지 시점 속도: `dP/ds|s=1 = 3*(c3-c2)` → z 방향 급속 하강
- height_cmd = 0.1이면 착지 직전 z속도가 0.3 규모로 커짐

### 원인 2: 착지 순간 z속도 페널티 부재
- `tracking_contacts_shaped_vel`은 stance 구간 전체 발속도를 패널티
- 착지 순간(첫 접촉)의 충격 속도를 별도로 패널티하지 않음

### 원인 3: 충격 force 패널티 부재
- `tracking_contacts_shaped_force`는 swing 중 접촉 force를 패널티
- 착지 순간의 충격 force(impact force) 크기를 패널티하지 않음

---

## 개선 방안 (우선순위 순)

### 방안 A: Bezier c2 z값 수정 ⭐ [가장 효과적, 즉각 적용 가능]

**현재**:
```python
c2 = c3.clone()
c2[:, :, 2:3] = height_cmd  # c2 z = height_cmd (높음)
```

**수정안**:
```python
c2 = c3.clone()
c2[:, :, 2:3] = height_cmd * 0.1  # c2 z = 10%만 남겨 착지 전 접지 준비
```
또는 고정값:
```python
c2[:, :, 2:3] = torch.clamp(height_cmd * 0.1, min=0.01)
```

- **효과**: 착지 직전 발이 낮게 유지되어 착지 충격 감소
- **위험**: bezier reward(`feet_clearance_bezier`)가 c2 변경에 맞춰 훈련된 경우 재학습 필요

---

### 방안 B: Foot Touchdown Velocity Penalty ⭐ [직접적, 새 보상 추가]

착지 순간(첫 접촉) z속도에 패널티.

**구현**:
```python
# compute_first_contact: swing → stance 전환 감지
first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
# 발의 z방향 속도 (음수 = 하강)
foot_z_vel = self._robot.data.body_link_lin_vel_w[:, self._feet_ids, 2]  # [num_envs, 4]
# 착지 시 z속도 크기 패널티 (하강 속도가 클수록 큰 페널티)
foot_landing_vel = torch.sum(torch.square(foot_z_vel) * first_contact, dim=1)
```

- **스케일 제안**: `-2.0` ~ `-5.0` (step_dt 곱 포함)
- **효과**: 발이 내려오는 속도를 줄이도록 policy가 학습

---

### 방안 C: Impact Force Penalty [보조적]

착지 후 첫 stance 스텝의 contact force 크기 패널티.

```python
# first_contact에서 force 측정
first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
impact_forces = foot_forces * first_contact  # [num_envs, 4]
foot_impact_penalty = torch.sum(impact_forces, dim=1)
```

- **스케일 제안**: `-0.001` ~ `-0.005`
- **주의**: gait_force_sigma로 이미 부분적으로 커버됨

---

### 방안 D: 착지 직전 발 속도 보상 구조 개선 (tracking_contacts_shaped_vel 강화)

기존 `tracking_contacts_shaped_vel`은 stance 전체에서 발 속도 패널티.
**착지 초기** (desired_contact > 0.8 등) 구간에만 더 강한 패널티 부과.

```python
# 강한 stance 초기 (contact transition zone)
strong_stance_mask = (self.desired_contact_states > 0.8).float()
tracking_contacts_shaped_vel_landing = 0
for i in range(4):
    tracking_contacts_shaped_vel_landing += -(
        strong_stance_mask[:, i] * (1 - torch.exp(-foot_velocities[:, i]**2 / (self.cfg.gait_vel_sigma * 0.1)))
    )
```

---

## 구현 계획 (단계별)

### Phase 1: 빠른 개선 (Bezier c2 수정)
- [ ] `go2_wtw_env.py` L610: c2 z값 수정 (`height_cmd * 0.1`)
- [ ] `go2_env_cfg.py`: 새 config 파라미터 `bezier_landing_ratio = 0.1` 추가 (선택)
- [ ] 시뮬레이션 확인

### Phase 2: 착지 속도 보상 추가
- [ ] `go2_wtw_env.py` `_get_rewards()`: `foot_landing_vel` 계산 추가
- [ ] `go2_env_cfg.py`: `foot_landing_vel_reward_scale = -3.0` 추가
- [ ] 보상 dict에 추가 + episode sums logging

### Phase 3: 검증
- [ ] 학습 후 deploy에서 발 충격 완화 확인
- [ ] 기존 gait quality 지표(lin_vel tracking 등) 저하 없는지 확인

---

## 파일 위치
- 환경: `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py`
- 설정: `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_env_cfg.py`
