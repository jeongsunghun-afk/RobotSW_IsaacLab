# Parkour A — Spot GaitReward async 4항 복원 (sync-only 실패 후속)

**작성일**: 2026-05-22
**대상**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/`
**문제**: 이전 sync-only 패턴 (weight=0.5, std=0.2, max_err=0.3, × velocity_gate × is_flat) 으로 학습한 결과 **trot 보행이 형성되지 않음** (사용자 정성 관찰)
**가설**: sync 식은 정지/끌기 상태에서도 max reward를 줌 (false positive). async 항이 빠지면 "trot 외 패턴이 모두 페널티" 강제가 사라짐

---

## 1. 진단

### sync-only의 false-positive 시나리오
| 상태 | sync_pair 값 |
|---|---|
| 4발 정지 + 접지 | air_time=0 모두 → se_air=0 → sync_pair ≈ 1.0 |
| 4발 끌기 (모두 contact 누적) | contact_time 비슷 → se_contact 작음 → sync_pair 높음 |
| 진짜 trot | 대각 쌍 내 시간 동기 → sync_pair 높음 (정상) |
| 한 다리 들기 | 들린 쪽 air_time 누적 → 쌍 내 차이 큼 → sync_pair 낮음 (페널티 작용) |

→ sync는 "정지/끌기"와 "진짜 trot"을 구별 못함. **정지에서도 high reward** → 정책이 trot으로 이동할 동기 부족.

### async 4항이 해결하는 것
async = `exp(-(se(air_a - contact_b) + se(contact_a - air_b)) / std)` — 다른 대각 쌍 소속 두 발(a, b)의 air vs contact 교차 비교.

| 상태 | async 값 |
|---|---|
| 4발 정지 + 접지 | a, b 모두 air=0, contact 동일하게 누적 → cross 차이 작음 → async ≈ 1.0... |
| **(정정)** 4발 정지 | 사실 air_a=0, contact_b 누적 → `(air_a - contact_b)² ≈ contact_b² > 0` → async 낮음 → 정지 차단 |
| 진짜 trot | a swing 중일 때 b stance → cross 일치 → async 높음 |

→ async가 정지 false-positive 차단 + trot 외 패턴 모두 페널티

---

## 2. 변경 사항 (1곳 추가만, 최소 침습)

### 현재 코드 상태 (사용자가 추가한 `× is_flat` 포함)
`parkour_env.py:_get_rewards()` ~line 1083-1097:
```python
# === feet_gait_pairing reward — sync-only ===
air_time     = self._contact_sensor.data.current_air_time
contact_time = self._contact_sensor.data.current_contact_time
std      = self.cfg.feet_gait_std
max_err2 = self.cfg.feet_gait_max_err ** 2

def _sync_pair(pair):
    a, b = pair
    se_air     = torch.clamp((air_time[:, a]     - air_time[:, b])     ** 2, max=max_err2)
    se_contact = torch.clamp((contact_time[:, a] - contact_time[:, b]) ** 2, max=max_err2)
    return torch.exp(-(se_air + se_contact) / std)

sync_reward = _sync_pair(self._gait_synced_pair_0) * _sync_pair(self._gait_synced_pair_1)
gate = (torch.norm(self._commands[:, :2], dim=-1) > self.cfg.feet_gait_velocity_threshold).float()
feet_gait_pairing = sync_reward * gate * is_flat  # 또는 사용자 추가 형태
```

### 변경: `_async_pair` 함수 + async 4항 곱 추가
```python
def _async_pair(a, b):  # a, b are single foot indices (NOT a pair)
    se_0 = torch.clamp((air_time[:, a]     - contact_time[:, b]) ** 2, max=max_err2)
    se_1 = torch.clamp((contact_time[:, a] - air_time[:, b])     ** 2, max=max_err2)
    return torch.exp(-(se_0 + se_1) / std)

# Spot 원본 async 4항 — 서로 다른 대각 쌍 소속 발의 cross 비교
p0_0, p0_1 = self._gait_synced_pair_0  # (FL, RR)
p1_0, p1_1 = self._gait_synced_pair_1  # (FR, RL)
async_reward = (
    _async_pair(p0_0, p1_0) *   # FL vs FR
    _async_pair(p0_1, p1_1) *   # RR vs RL
    _async_pair(p0_0, p1_1) *   # FL vs RL
    _async_pair(p0_1, p1_0)     # RR vs FR
)

# Final 곱셈에 async 추가
feet_gait_pairing = sync_reward * async_reward * gate * is_flat
```

### 변경 영향 요약
| 항목 | 변경 |
|---|---|
| `_async_pair` nested function | 새로 추가 |
| `async_reward` 계산 (4항 곱) | 새로 추가 |
| `feet_gait_pairing` 최종 식 | sync × **async** × gate × is_flat |
| cfg 변경 | **없음** (std, max_err, velocity_threshold 그대로) |
| weight | **0.5 그대로** (한 번에 하나 변경 원칙) |
| obs / reset | 변경 없음 |

---

## 3. 위험 & Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| 학습 초기 dead zone (sync × async 둘 다 0 근처) | 중간 | std=0.2, max_err=0.3 (Spot 0.1/0.2보다 완화) 이미 적용 → 초기 gradient 확보 |
| weight 0.5에 4항 곱이 추가되어 신호 약화 | 중간 | 결과 확인 후 weight 1.0~1.5로 sweep 후속 가능 |
| `is_flat` 곱셈으로 비평지 env에서 신호 0 | 의도된 설계 (사용자 결정) | 평지에서만 trot 학습이 목표이므로 유지 |
| Spot async 정의가 정확한지 | 낮음 | Explore 보고서 line 162-179, 142-150 정확 인용. 4항 곱 검증됨 |

---

## 4. 사용자 확인 사항

1. `× is_flat` 곱셈 형태 그대로 유지 (현재 코드 상태) — 동의?
2. weight=0.5, std=0.2, max_err=0.3 그대로 (async만 추가) — 동의?
3. 학습 후 trot 형성 확인되면 weight sweep / `is_flat` 제거 등 다음 단계 검토

---

## 5. 구현 절차

1. reward-worker: async 4항 nested function + 곱셈 추가
2. validate-code: 정합성 확인
3. **(필수) cfg-worker 사후 확인**: weight 값이 0.5인지 직접 검증 (이전 0.0 사고 재발 방지)
4. 사용자 학습 시작
