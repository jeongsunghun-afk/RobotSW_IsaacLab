# Task Plan: Bezier Curve Foot Clearance Reward

## 목표
go2 WTW 환경에서 발 궤적을 Bezier 곡선으로 제어하는 새 보상 함수 추가.
기존 `feet_clearance_cmd_linear`는 유지하고 `feet_clearance_bezier`를 추가.

## 배경
로봇이 학습 후 실제 적용 시 쾅쾅거리는 문제 → 기존 삼각파 높이 프로파일이 착지 직전까지
높은 높이를 요구하기 때문. Bezier 곡선으로 부드러운 착지(c2=ground+h → c3=ground) 유도.

---

## 구현 단계

### Phase 1: go2_wtw_env_cfg.py 수정 [complete]
- `feet_clearance_bezier_scale: float` 파라미터 추가
- 초기값: 기존 `feet_clearance_cmd_linear_scale`과 동일하게 설정

### Phase 2: go2_wtw_env.py - __init__ 버퍼 추가 [complete]
추가할 버퍼:
```python
self.swing_start_pos = torch.zeros(self.num_envs, 4, 3, device=self.device)
self._prev_desired_contact_states = torch.ones(self.num_envs, 4, device=self.device)
```

### Phase 3: go2_wtw_env.py - _reset_idx 초기화 [complete]
리셋 시:
```python
self.swing_start_pos[env_ids] = 0.0
self._prev_desired_contact_states[env_ids] = 1.0
```

### Phase 4: go2_wtw_env.py - _get_rewards에서 swing 감지 및 swing_start_pos 업데이트 [complete]
```python
# swing 시작 감지 (stance→swing 전환)
was_stance = self._prev_desired_contact_states > 0.5   # [num_envs, 4]
is_swing = self.desired_contact_states < 0.5            # [num_envs, 4]
transition_to_swing = was_stance & is_swing              # [num_envs, 4]

# swing_start_pos 업데이트
transition_mask = transition_to_swing.unsqueeze(-1).expand_as(foot_positions)
self.swing_start_pos = torch.where(transition_mask, foot_positions, self.swing_start_pos)
self._prev_desired_contact_states = self.desired_contact_states.clone()
```

### Phase 5: go2_wtw_env.py - Bezier 보상 계산 [complete]
```python
# swing 진행률 (단조 [0, 1])
swing_progress = torch.clamp((self.foot_indices - 0.5) * 2.0, 0.0, 1.0)  # [num_envs, 4]

# Bezier 제어점
c0 = self.swing_start_pos[:, :, 2]                 # [num_envs, 4]
c3 = self.desired_footsteps_world_frame[:, :, 2]   # [num_envs, 4], 모두 0
height_cmd = commands[:, 9].unsqueeze(1)           # [num_envs, 1]
c1 = c0 + height_cmd
c2 = c3 + height_cmd

# 3차 Bezier B(s)
s = swing_progress
bezier_z = (1-s)**3 * c0 + 3*(1-s)**2 * s * c1 + 3*(1-s) * s**2 * c2 + s**3 * c3

# 보상 계산
foot_height_z = foot_positions[:, :, 2]
feet_clearance_bezier = torch.square(bezier_z - foot_height_z) * (1 - self.desired_contact_states)
feet_clearance_bezier = torch.sum(feet_clearance_bezier, dim=1)
feet_clearance_bezier[both_low] = 0.0
```

### Phase 6: go2_wtw_env.py - rewards 딕셔너리에 추가 [complete]
```python
"feet_clearance_bezier": feet_clearance_bezier * self.cfg.feet_clearance_bezier_scale * self.step_dt,
```

---

## 결정 사항
- `swing_start_pos` 업데이트는 `_get_rewards` 내부에서 수행
  (이미 desired_contact_states, foot_positions가 계산된 후이므로 순서 보장)
- `_prev_desired_contact_states`는 매 reward 계산마다 갱신
- 초기에는 `feet_clearance_bezier_scale`을 기존 선형 보상과 동일한 크기로 시작 후 튜닝

## 검증 방법
1. 학습 로그에서 `feet_clearance_bezier` 에피소드 합계 확인
2. play.py로 시각적 확인 (발이 부드러운 아크 그리는지)
3. 기존 `feet_clearance_cmd_linear` 값과 비교해 발산 여부 확인
