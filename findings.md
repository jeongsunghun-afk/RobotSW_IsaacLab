# Findings: Bezier Curve Foot Trajectory Reward

## 현재 코드 분석

### feet_clearance_cmd_linear 기존 구현 (go2_wtw_env.py:430-436)

```python
phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
foot_height = (foot_positions[:, :, 2]).view(self.num_envs, -1)
target_height = commands[:, 9].unsqueeze(1) * phases + 0.02
feet_clearance_cmd_linear = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
feet_clearance_cmd_linear = torch.sum(feet_clearance_cmd_linear, dim=1)
feet_clearance_cmd_linear[both_low] = 0.
```

- `phases`: foot_indices 기반 **삼각파** [0→1→0], swing 중간에서 최대
- 기존 방식은 높이가 선형적으로 증가했다가 감소 → 착지 직전에도 높이를 유지하려 해서 쾅쾅거림 유발

### foot_indices 구조
- [0.0, 0.5): **stance** 구간
- [0.5, 1.0): **swing** 구간
- swing 진행률(monotonic) = `(foot_indices - 0.5) * 2.0` → [0, 1]

### swing_start_pos 버퍼
- **존재하지 않음** → 새로 추가 필요
- swing 시작 시점의 발 위치를 저장해야 Bezier 시작점(c0) 사용 가능

### desired_footsteps_world_frame
- Raibert Heuristic 기반 목표 발 위치
- **z = 0.0** (항상 지면으로 설정됨, 라인 530)
- shape: `[num_envs, 4, 3]`

### desired_contact_states
- 정규분포 CDF 기반 부드러운 접촉 상태 [0, 1]
- 0 ≈ swing, 1 ≈ stance
- swing 마스크: `(1 - self.desired_contact_states)`

### _episode_sums 초기화 위치
- `_get_rewards`에서 `rewards` dict 키로 자동 등록됨
- `__init__`에서 별도 초기화 불필요 (DirectRLEnv가 처리)
- 단, cfg에 scale 파라미터를 추가해야 로깅 가능

---

## Bezier 구현 타당성 검토

### 제안된 제어점
```
c0 = swing 시작 시 발 z (실제 높이)
c1 = c0 + height_cmd   (이륙 가속)
c2 = 0 + height_cmd    (c3=0이므로, 착지 감속)
c3 = desired_footstep z = 0.0 (지면)
```

### s=0.5에서의 높이 분석
```
B(0.5) = 0.125*c0 + 0.375*(c0+h) + 0.375*(0+h) + 0.125*0
       = 0.5*c0 + 0.75*h
```
c0 ≈ 0이면 B(0.5) ≈ 0.75 * height_cmd → 합리적인 아크 형성 ✅

### 핵심 수정 필요사항
1. **phase 변수**: 삼각파가 아닌 **단조증가 [0→1]** 필요
   - 현재: `phases = 1 - |1 - clip((foot_indices*2)-1, 0, 1)*2|` → 삼각파
   - 수정: `swing_progress = clamp((foot_indices - 0.5) * 2.0, 0.0, 1.0)` → 단조

2. **c0 (시작점)**: swing 시작 순간의 실제 발 위치 필요 → `swing_start_pos` 버퍼 추가

3. **swing 감지**: `desired_contact_states > 0.5` (stance) → `< 0.5` (swing) 전환 감지

### 잠재적 문제
- `swing_start_pos` 초기화: 환경 리셋 시 0으로 초기화되면 첫 스텝에서 c0=0 사용
  → 첫 swing은 다소 부정확할 수 있음 (허용 가능)
- 가파른 지형에서 c0 ≠ 0인 경우: Bezier가 지형 높이를 고려함 → 오히려 좋음

---

## 구현에 필요한 파일 목록

| 파일 | 변경 내용 |
|------|---------|
| `go2_wtw_env.py` | swing_start_pos 버퍼, 감지 로직, Bezier 계산, 새 보상 |
| `go2_wtw_env_cfg.py` | `feet_clearance_bezier_scale` 파라미터 추가 |
