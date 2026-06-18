# Parkour A — 평지 Trot Gait Shaping 계획

**작성일**: 2026-05-21
**대상 env**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/` (A env)
**문제**: feet_dragging penalty (-0.1, 뒷발) 도입 후 부작용 — 정책이 dragging을 회피하기 위해 다리를 일부러 들어 이동
**해결 방향**: **평지에서 자연스러운 trot 보행을 학습**하도록 보조 reward 추가

---

## 1. 진단 — 부작용의 원인

| 상태 | 신호 |
|---|---|
| 정책이 발을 끌면 (contact ∧ vel>0.05) | `feet_dragging × -0.1` 페널티 |
| 정책이 발을 들면 (contact=False) | `feet_dragging = 0` (회피 가능) |
| 그러나 적절한 **발 들기 시간**이 있는지/대각 쌍 동기인지에 대한 **명시 보상 없음** |
| 결과 | dragging 회피 = 발 들기 (높이/타이밍 무관) — local minimum |

→ **trot gait의 두 가지 핵심 특성**을 명시 보상해야 함:
1. 발이 일정 시간(예 ≥0.4s) 공중에 있어야 함 (적절한 swing duration)
2. 발 들기 높이가 비정상적으로 과하지 않아야 함 (over-lift 억제)

---

## 2. 참고 패턴 비교 → 선택

| 패턴 | 출처 | 효과 | dragging과의 양립성 | 채택 |
|---|---|---|---|---|
| **feet_air_time** (ANYmal/ETH) | `mdp/rewards.py:27-46` | 발이 threshold 이상 공중 → +reward (first_contact 시점) | ✅ 직교 신호 (dragging은 contact 시, air_time은 swing 시) | **Stage 1 채택** |
| foot_clearance (Spot, target~5cm) | `…/spot/mdp/rewards.py:182-190` | swing 중 발 z를 target 유지 | ❌ over-lift 부작용 강화 위험 (target 위로도 penalty 매우 약함) | 제외 |
| **foot swing-height cap** (over-lift penalty) | 자작 | 발 z가 cap (~0.15m) 초과 시 (z-cap)² penalty | ✅ Stage 1 보완 — over-lift만 억제 | **Stage 2 채택** |
| GaitReward 대각 쌍 (Spot) | `…/spot/mdp/rewards.py:87-179` | FL+RR / FR+RL air-time 차이 minimize | ✅ trot 강제 강력하나 buffer/복잡도 증가 | **Stage 3 보류** |
| WTW phase clock | `direct/go2/go2_wtw_env.py` | 명시 phase variable | 큰 구조 변경, sim-to-real 추가 검토 필요 | 보류 |

### 선택 전략: **Stage 1 → 평가 → Stage 2** (한 번에 한 변경 원칙)

이유:
- 한 번에 여러 reward 동시 추가 시 효과/부작용 분리 불가
- 사용자가 이미 dragging 도입 직후 부작용을 발견 — 단계적 도입이 필수
- feet_air_time이 ANYmal 표준 + parkour A에 사전 검증 문서 (`parkour_feet_air_time_validation.md`) 존재 → Stage 1 안전

---

## 3. Stage 1 — `feet_air_time` 추가 (본 작업)

### 3.1 변경 1: `parkour_env_cfg.py` — `reward_scales`에 항 추가
**위치**: `parkour_env_cfg.py:566-582` (reward_scales dict 내)
**추가 (1줄)**:
```python
"feet_air_time": 0.5,   # ANYmal-style: reward swing duration > threshold; gates by cmd speed
```

### 3.2 변경 2: `parkour_env_cfg.py` — 관련 cfg field 추가
**위치**: `dragging_velocity_threshold` 인접 (~line 592 근처)
**추가 (2줄)**:
```python
feet_air_time_threshold: float = 0.3   # seconds; Go2 size (15kg). ANYmal=0.5, Go1=0.25 → Go2 권장 0.3 (validate-method G-4)
feet_air_time_cmd_speed_threshold: float = 0.1   # m/s; 정지 명령 시 발 들기 보상 차단 (IsaacLab 표준과 동일)
```

### 3.3 변경 3: `parkour_env.py` — `_get_rewards()`에 계산 추가
**위치**: 기존 `feet_dragging` 계산 인접 (~line 1077 직후, contact_filt 활용 영역 내)
**필수 조건**:
- `self._contact_sensor.data.last_air_time[:, self._feet_ids]` (shape: N,4) — 직전 swing duration
- `self._contact_sensor.data.current_contact_time[:, self._feet_ids]` — 현재 contact 지속 시간 (0이면 still in air)
- `first_contact` 시점 추출: `current_contact_time` 가 dt 이하인 step (방금 착지)

**계산 (개념적 6~8줄)** — validate-method G-1 반영, 공식 API 사용:
```python
# === feet_air_time reward — ALL 4 feet, ANYmal pattern (IsaacLab standard) ===
# Use ContactSensor's official compute_first_contact() — robust across IsaacLab versions
last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]  # (N, 4)
first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]  # (N, 4) bool
air_time_bonus = (last_air_time - self.cfg.feet_air_time_threshold) * first_contact.float()  # (N, 4)
# Gate: only when commanded speed is non-trivial — IsaacLab feet_air_time() 표준과 동일 (mdp/rewards.py:45)
cmd_speed = torch.norm(self._commands[:, :2], dim=-1)  # (N,)
gate = (cmd_speed > self.cfg.feet_air_time_cmd_speed_threshold).float()
feet_air_time = torch.sum(air_time_bonus, dim=-1) * gate  # (N,)
```

> ✅ **확인 완료** (validate-method G-3):
> - `self._commands` shape (N,3), index 0=vx, 1=vy, 2=wz (parkour env line 105 검증)
> - `parkour_env_cfg.py:541`에 `track_air_time=True` 이미 활성화 → `last_air_time` / `compute_first_contact` 사용 가능
> - `compute_first_contact()` 는 IsaacLab `ContactSensor` 공식 API (contact_sensor.py:182-216)

### 3.4 변경 4: `parkour_env.py` — `reward_values` dict 등록
`feet_dragging`과 동일 패턴으로 `"feet_air_time": feet_air_time` 추가 (~line 1114).

### 3.5 변경 영향
- **`_get_observations`**: 변경 없음 (contact_sensor 데이터는 reward 내부 전용 → sim-to-real 안전)
- **`_reset_idx`**: 변경 없음 (`last_air_time`, `current_contact_time`은 `ContactSensor` 내부 관리)
- **obs space dim**: 변경 없음
- **신규 self.* buffer**: 없음

---

## 4. Stage 2 — Over-lift penalty (Stage 1 학습 후 부작용 잔존 시)

> **본 계획에서는 Stage 1만 즉시 구현. Stage 2는 학습 결과 보고 후 사용자 결정 시 적용.**

**개념**: foot z (world frame) > cap 시 penalty.

```python
# Stage 2 candidate (NOT in current implementation):
foot_z = self._robot.data.body_pos_w[:, self._feet_ids, 2]  # (N, 4)
over_lift = torch.clamp(foot_z - self.cfg.feet_overlift_cap, min=0.0)  # (N, 4)
feet_overlift = torch.sum(over_lift ** 2, dim=-1) * is_flat  # (N,) — flat only
```
- cap = 0.15m (trot natural step height ~5-10cm 위 여유)
- weight = -0.3 (가정, 학습 후 조정)
- `is_flat` mask 곱 → 비평지에서 장애물 넘기 학습 방해 회피

---

## 5. 검증 plan (validate-method 검증 항목)

| 항목 | 검증 |
|---|---|
| (a) `contact_sensor.data.last_air_time` 의미 | 직전 swing duration인지, current인지 — IsaacLab ContactSensor API 정확성 |
| (b) first_contact 추출 방법 | `current_contact_time > 0 ∧ ≤ dt`가 ANYmal `first_contact` API와 의미 동등한지 |
| (c) cmd speed attribute 이름 | parkour env의 실제 velocity command 변수명 (실 구현 시 확인) |
| (d) Reward magnitude 균형 | 0.5 × 0.4s(threshold 초과분 평균 ~0.2s) × 4발 × first_contact 빈도 (~step 5%) ≈ +0.04/step. tracking_goal_vel(+0.03/step)와 비슷 — 적절. |
| (e) Velocity gating의 정당성 | 정지 명령 시 발을 드는 부작용을 차단 — 사용자 관찰과 정합. cmd_speed_threshold=0.1m/s는 noise floor 위. |
| (f) dragging penalty와 conflict | swing 시 air_time +reward, contact + vel>thr 시 dragging -reward. **서로 다른 phase**에 작동 → 직교. 동시 만족 불가. ✅ |
| (g) sim-to-real 호환 | contact는 reward 내부 전용 (CLAUDE memory §1 준수). ✅ |
| (h) 학습 봉쇄 위험 | threshold=0.4s는 너무 길어 stall 위험? Go2 일반 trot stride ~0.3-0.5s → 0.4s는 약간 보수적, 안전 |
| (i) Genesis 원본과의 격차 | Genesis에 feet_air_time 있는지 — 보고서 G의 권고는 ANYmal/R_Skeleton 표준 (Genesis와 무관). 별도 영역 |
| (j) yaw 신호 선행 요구사항 | 보고서 G에서 yaw 신호 문제 언급. 본 계획 scope 외이나 사용자에게 노트로 보고 |

---

## 6. Risk & Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| threshold=0.4s가 Go2에 너무 길다 → 학습 봉쇄 | 중간 | smoke run 후 0.3s까지 낮출 옵션 보유, 또는 cfg field이므로 즉시 조정 가능 |
| velocity gating attribute 이름 오류 | 낮음 | reward-worker가 실제 env 변수명 (예: `self._commands[:,:2]` vs `self._goal_vel`) 확인 후 적용 |
| Stage 1만으로 over-lift 부작용 잔존 | 중간 | Stage 2 (over-lift penalty) 준비됨, 사용자 결정 시 즉시 적용 |
| `current_contact_time` field 부재 | 낮음 | IsaacLab ContactSensor 표준 API — manager_based `feet_air_time_positive_biped` 등에서 사용 검증됨. 부재 시 `last_air_time`만으로 대체 가능 |
| yaw 신호 문제 (보고서 G 언급) | 별도 issue | 본 계획 scope 외. 사용자에게 별도 보고 |

---

## 7. 제약 메모 self-check

| 제약 | 본 계획 |
|---|---|
| §1 Contact obs 추가 금지 | ✅ contact_sensor 데이터 reward 내부 전용 |
| §2 새 reward 도입 | ✅ 2026-05-13 해제됨. ANYmal 표준 패턴 채택 |
| §3 Reward scale 단독 조정 금지 | ✅ 신규 reward 추가 (구조 변경) |
| §4~7 (action latency / actuator / torque / height_scan) | 모두 무관 |

---

## 8. 사용자 확인 요청 사항

1. **Stage 1 (feet_air_time +0.5, threshold=0.4s, cmd gating=0.1m/s) 즉시 적용** 동의?
2. **4발 모두 적용** 동의? (dragging은 뒷발만이었으나 air_time은 4발 trot 전체 강제가 표준)
3. Stage 2 (over-lift penalty)은 **Stage 1 학습 결과 보고 후** 판단 동의?
4. validate-method 검증 후 reward-worker dispatch OK?
