# Parkour A — Gait Reward Duty-factor 재설계 (대기 중)

**작성일**: 2026-05-26
**상태**: **DEFERRED** — 사용자 결정으로 reward 수정 대기. 본 문서는 추후 재개 시 활용.

---

## 1. 배경 — yaw tracking 약화 분석

### 관찰
| 실험 | tracking_yaw @ iter 15527 | tracking_goal_vel | gait_pairing weight |
|---|---|---|---|
| A. feet_dragging | 0.40 | 1.27 | (없음) |
| B. collision_candidate | 0.38 | 1.04 | 0.0 (코드만, 비활성) |
| C. add_spot_trot_reward | 0.37 (-8%) | 0.90 (-29%) | 0.5 활성 |

사용자 정성 관찰: "로봇이 goal 방향을 안 바라본다"

### Background bug (별도)
- `parkour_env.py:1004` `yaw_diff` wrap 비활성 (atan2(sin, cos) 주석 처리)
- A/B/C 실험 시점 모두 동일 상태 (Explore 검증)
- **사용자가 이미 fix 적용 완료** (commit 629f32b42c5, 2026-05-26)
- 본 문서 scope 외

---

## 2. 사용자 핵심 통찰 — 회전이 phase를 깨지 않는다

이전 분석의 "step duration 비대칭" 주장은 부정확. 사용자 지적:

> "같은 시간이 필요하다면 각 회전 시에는 각 발의 속도 및 각속도가 안쪽/바깥쪽에 따라 달라지면 되는 거 아니야?"

**정정된 이해**:
- 이상적 turning trot: 안쪽 발 = 짧은 거리 × 낮은 선속도, 바깥쪽 발 = 긴 거리 × 높은 선속도, **step duration은 유지 가능**
- 기존 sync_pair (`(air_time[a] - air_time[b])²`)는 시간만 측정 → 이론적으로 회전과 무관
- 그러나 학습 초기 정책은 정확한 turning trot을 못 함 → step duration 흔들림이 metric을 일시 깨뜨림 → 회피 학습

### 진짜 메커니즘
| Mech | 설명 |
|---|---|
| **1. gait reward는 회전 정보 0** | 시간 기반 metric → "어떻게 회전해야 할지" gradient 없음 |
| **2. 학습 초기 imperfect turning이 sync 감소** | step duration 변동성이 metric을 일시 깨뜨림 → 회피 |
| **3. Reward landscape bias** | 직진에서 sync max → 정책 직진 편향 |
| **4. Speed reduction** | gait stride 안정화 위해 base speed ↓ → 회전 능력 ↓ |

---

## 3. 설계 — Duty Factor 기반 재설계

### ContactSensor API (검증 완료)
| 필드 | 활성 조건 | 비고 |
|---|---|---|
| `current_air_time` | `track_air_time=True` | 이미 사용 중 |
| `current_contact_time` | 동일 | 이미 사용 중 |
| `last_air_time` | 동일 | 이미 사용 중 |
| **`last_contact_time`** | 동일 | **존재함** (`contact_sensor_data.py:137-144`), 본 설계에 활용 |

`track_air_time=True` 한 플래그로 4개 모두 활성. parkour A env에 이미 활성.

### 새 sync_pair 식
```python
eps = 1e-6
period_a = last_air_time[a] + last_contact_time[a]   # 한 cycle 총 시간
period_b = last_air_time[b] + last_contact_time[b]
duty_a   = last_contact_time[a] / (period_a + eps)   # 0~1, stance 비율
duty_b   = last_contact_time[b] / (period_b + eps)

# 회전 robust: period는 의도적으로 제외
sync_pair = exp(-((duty_a - duty_b) ** 2) / std)
```

### 왜 회전에 robust
| 상황 | 기존 sync | 새 duty sync |
|---|---|---|
| 직진 trot | sync ≈ 1 | sync ≈ 1 |
| 회전, step length 비대칭, **duty 유지 (0.5)** | air_time 누적 비대칭 → **깨짐** | duty 같음 → **유지** |
| 회전 + 실제 duty 비대칭 | 깨짐 | 깨짐 (정상 검출) |
| 정지 (모두 stance) | sync ≈ 1 (false positive) | duty=1 동일 → sync ≈ 1 (여전) — velocity_gate가 막아야 |

### Async 처리 옵션 (사용자 결정 대기)
| 옵션 | 식 | 의미 | 추천 |
|---|---|---|---|
| (i) sync only, async 제거 | `gait = sync_1 × sync_2 × gate × is_flat` | 가장 단순, 단일 가설 검증 | 보수적 시작 |
| **(ii) duty 기반 async** | `async_pair = exp(-((duty[a] + duty[c] - 1)²) / std)` (한 쌍 stance ↔ 다른 쌍 swing) | duty 기반 일관 + trot phase 강제 | **권장** |
| (iii) sync 듀티, async 기존 time | 혼합 | 메커니즘 2 일부 잔존 | 비추 |

### 한계 (정직)
1. **phase timing 측정 못 함**: duty 같음 ≠ swing 시작 시점 동기. 순수 phase angle은 측정 X
2. **정지 false-positive 잔존**: velocity_gate (cmd_xy > 0.3)에 의존
3. async (ii) 형태는 학습 초기 duty 0/1 극단 → 신호 불안정 가능

---

## 4. 변경 사항 (재개 시 적용)

### parkour_env.py:_get_rewards()
기존 `_sync_pair(pair)` 함수를 duty 기반으로 교체. `_async_pair` 도 옵션 선택 후 재정의.

### parkour_env_cfg.py
- weight (`reward_scales["feet_gait_pairing"]`): 현재 0.0 (사용자 의도 대기 상태). 활성화 시 0.5 또는 사용자 결정값
- std, max_err: 기존 0.2 / 0.3 유지하되 duty 도메인 ([0,1])에 맞춰 추후 조정 검토 (max_err 0.5 정도가 자연스러울 수 있음)

---

## 5. 재개 절차 (대기 해제 시)

1. 사용자가 reward 수정 신호 → 본 문서 참조
2. async 옵션 (i/ii/iii) 결정
3. validate-method 검증 (특히 duty 0/1 극단 dead zone)
4. reward-worker dispatch
5. cfg-worker로 weight 활성화 (0.0 → 0.5 또는 결정값)
6. validate-code
7. 학습 시작

---

## 6. 현재 상태 (2026-05-26)

| 항목 | 상태 |
|---|---|
| yaw wrap fix | ✅ 사용자가 적용 완료 |
| feet_gait_pairing 코드 | sync × async × gate × is_flat (Spot 원본 형태) 적용됨 |
| reward_scales["feet_gait_pairing"] | **0.0** (사용자 의도, 다른 실험용) |
| duty factor 재설계 | 본 문서 — **대기 중** |

다음 학습은 wrap fix + 현재 gait 코드 (weight 0.0) 상태로 진행 가능. duty factor 도입은 사용자 trigger 시.
