# Parkour A — Trot Diagonal Pair Gait Reward (Spot GaitReward 패턴 이식)

**작성일**: 2026-05-22
**대상 env**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/` (A env)
**문제**:
- feet_dragging penalty 도입 후 "다리 들기" 부작용 관찰
- feet_air_time (ANYmal pattern, weight=0.5) 시도 → 학습 붕괴 (음수 reward 누적 → "발 떼지 마라" gradient)
- 사용자 결정: feet_air_time 완전 제거, **Spot GaitReward 대각 쌍** 패턴으로 전환

**해결 방향**: **trot phase를 직접 강제**하는 양수-only reward. FL+RR / FR+RL 대각 쌍 air_time/contact_time 동기, 다른 발 페어는 비동기.

---

## 1. 핵심 설계 — Spot GaitReward Sync-only (validate-method 권고 옵션 A)

> **validate-method 발견**: 계획서 초안의 async가 Spot 원본 (4항 곱, 쌍 간 교차)과 의미 불일치 → 옵션 A 채택. **sync만 사용**, async 제거. 학습 초기 dead zone 위험 감소 + 구현 단순.

### Sync reward (대각 쌍 동기화)
```
se_air     = clip( (air_time[A]     - air_time[B])²,     max = max_err² )
se_contact = clip( (contact_time[A] - contact_time[B])², max = max_err² )
sync_pair  = exp( -(se_air + se_contact) / std )           # ∈ (0, 1]
sync_reward = sync_pair_1 × sync_pair_2                    # FL+RR × FR+RL
```

### Final
```python
gait_reward = sync_reward × velocity_gate   # ∈ [0, 1]
```
- 모두 양수 (exp 형태), 음수 누적 위험 없음 — feet_air_time 실패의 root cause 회피
- async 제거로 학습 초기 dead zone (`sync × async`가 0이 되는 false flat region) 위험 차단

---

## 2. parkour A 이식 매핑

### Foot index 매핑 (이미 검증됨)
`_feet_ids` URDF order = `[FL(0), FR(1), RL(2), RR(3)]`

| 페어 | 인덱스 |
|---|---|
| **Sync pair 1** (대각 1): FL + RR | `(0, 3)` |
| **Sync pair 2** (대각 2): FR + RL | `(1, 2)` |

### 사용 신호
- `self._contact_sensor.data.current_air_time[:, self._feet_ids]` — (N, 4) 현재 swing 지속 시간
- `self._contact_sensor.data.current_contact_time[:, self._feet_ids]` — (N, 4) 현재 contact 지속 시간
- 두 신호 모두 `track_air_time=True` 활성화 시 사용 가능 (parkour_env_cfg.py:541 확인됨)
- reset 시 0으로 자동 초기화 (env 측 처리 불필요)

### velocity gate
- `cmd_speed = torch.norm(self._commands[:, :2], dim=-1) > velocity_threshold` — 명령 속도 작으면 reward 0 (정지 시 발 들기 부작용 자동 차단)
- threshold = 0.3 m/s

---

## 3. 권장 파라미터 (Spot vs parkour A)

| 파라미터 | Spot (검증값) | parkour A 권장 | 근거 |
|---|---|---|---|
| `weight` | **10.0** | **0.5** | feet_air_time(0.5) 학습 붕괴 직후이므로 매우 보수적 시작. step당 max reward +0.01 (tracking_goal_vel 대비 3×). 양수-only이므로 학습 봉쇄 위험 낮음. 안정화 후 1.0→2.0 sweep |
| `std` | 0.1 | **0.2** | Spot 0.1은 매우 엄격 (error 100ms 초과 시 exp(-10)≈0). Go2 학습 초기 gradient 확보 위해 완화 |
| `max_err` | 0.2 | **0.3** | error 포화 cap. parkour 학습 초기 큰 phase 차이 허용 |
| `velocity_threshold` | 0.5 m/s | **0.3 m/s** | parkour goal-following 속도 범위 (0.3~1.0). lower bound 일치 |
| `synced_pairs` | name-based | **index-based** `[(0,3), (1,2)]` | parkour A는 `_feet_ids` 검증된 인덱스 직접 사용 (이전 hind feet assertion으로 안전성 보장) |

---

## 4. 변경 사항 (4곳)

### 변경 1: `parkour_env_cfg.py` — `reward_scales`에 항 추가
**위치**: line 566~582 (reward_scales dict 내, `feet_dragging` 인접)
**추가 (1줄)**:
```python
"feet_gait_pairing": 0.5,   # Spot GaitReward 패턴: trot 대각 쌍 phase 동기 (sync × async × velocity_gate)
```

### 변경 2: `parkour_env_cfg.py` — GaitReward 파라미터 cfg field 추가
**위치**: `dragging_velocity_threshold` 인접 (line 592 근처)
**추가 (3~4줄)**:
```python
# Gait pairing reward parameters (Spot GaitReward style)
feet_gait_std: float = 0.2   # error tolerance (Spot=0.1, parkour 완화)
feet_gait_max_err: float = 0.3   # max squared-error cap (Spot=0.2, parkour 완화)
feet_gait_velocity_threshold: float = 0.3   # m/s; cmd_speed gate
```

### 변경 3: `parkour_env.py` — `__init__`에 synced_pairs 정의 (compile-time)
**위치**: `_feet_ids` 정의 직후 (line 193~198, hind feet assertion 인접)
**추가 (3~5줄)**:
```python
# Gait pairing — diagonal pairs (FL+RR, FR+RL) using URDF order [FL(0), FR(1), RL(2), RR(3)]
self._gait_synced_pair_0 = (self._feet_ids[0], self._feet_ids[3])  # FL + RR
self._gait_synced_pair_1 = (self._feet_ids[1], self._feet_ids[2])  # FR + RL
```

### 변경 4: `parkour_env.py` — `_get_rewards()`에 gait reward 계산 (sync-only)
**위치**: feet_dragging 계산 직후, `self._last_contacts = contact` 이전 (이전 feet_air_time 자리)
**추가 (10~12줄)**:
```python
# === feet_gait_pairing reward — Spot GaitReward sync-only (대각 쌍 air/contact 시간 동기) ===
air_time     = self._contact_sensor.data.current_air_time      # (N, B)
contact_time = self._contact_sensor.data.current_contact_time  # (N, B)
std      = self.cfg.feet_gait_std
max_err2 = self.cfg.feet_gait_max_err ** 2

def _sync_pair(pair):
    a, b = pair
    se_air     = torch.clamp((air_time[:, a]     - air_time[:, b])     ** 2, max=max_err2)
    se_contact = torch.clamp((contact_time[:, a] - contact_time[:, b]) ** 2, max=max_err2)
    return torch.exp(-(se_air + se_contact) / std)

sync_reward = _sync_pair(self._gait_synced_pair_0) * _sync_pair(self._gait_synced_pair_1)
gate = (torch.norm(self._commands[:, :2], dim=-1) > self.cfg.feet_gait_velocity_threshold).float()
feet_gait_pairing = sync_reward * gate   # (N,), all in [0, 1]
```

> ⚠️ **validate-method 권고 옵션 A 적용**: Spot 원본의 async 항(4항 곱, 쌍 간 교차)을 제거하고 sync만 사용. 이유: (1) 계획서 초안의 async 정의가 Spot 원본과 의미 불일치, (2) sync만으로도 trot 대각 동기 신호 충분, (3) `sync × async` dead zone 위험 차단.

> ⚠️ **삽입 위치 주의** (직전 feet_dragging 작업과 동일 패턴):
> `self._last_contacts = contact` 대입 **이전** 영역에 위치. contact_filt와는 무관하므로 dragging 계산 직후 어디든 OK.

### 변경 5: `parkour_env.py` — `reward_values` dict 등록
**위치**: reward_values dict (~line 1124, `"feet_dragging": feet_dragging` 다음)
**추가 (1줄)**:
```python
"feet_gait_pairing": feet_gait_pairing,
```

---

## 5. 영향 없음 확인

| 항목 | 영향 |
|---|---|
| `_get_observations` | **변경 없음** — current_air_time/contact_time은 reward 내부 전용 (sim-to-real 안전) |
| obs space dim | 변화 없음 |
| `_reset_idx` | **변경 없음** — current_air_time/contact_time은 ContactSensor 내부 reset 처리 |
| 신규 self.* buffer | 없음 (synced_pairs는 인덱스 tuple, buffer 아님) |
| network input/output | 영향 없음 |
| agent yaml | 영향 없음 |
| 기존 feet_dragging | **그대로 유지** |

---

## 6. 검증 plan (validate-method 항목)

| 항목 | 검증 |
|---|---|
| (a) sync/async 수식 정확성 | Spot GaitReward (`spot/mdp/rewards.py:162-179`)와 동일 형식인지 확인 |
| (b) synced pair 매핑 | `_feet_ids[0]=FL`, `[3]=RR`, `[1]=FR`, `[2]=RL` 가정 검증 (기존 hind feet assertion으로 일부 보장) |
| (c) `current_air_time` / `current_contact_time` 활성화 | `track_air_time=True`만으로 둘 다 사용 가능한지 확인 |
| (d) reward magnitude 균형 | weight 0.5 × max 1.0 × step_dt 0.02 = +0.01/step. tracking_goal_vel (+0.003)의 3× — 보수적 (Spot의 1/20) |
| (e) 음수 누적 위험 | 모든 항이 exp(-...) 양수 [0,1], 곱셈도 양수 → **음수 누적 불가**. feet_air_time 실패의 핵심 원인 회피. ✅ |
| (f) gradient dead zone | `sync × async` 곱셈이라 둘 중 하나가 0이면 전체 0. std=0.2 / max_err=0.3 (Spot 대비 완화)로 dead zone 위험 감소 |
| (g) velocity gating | `cmd_speed > 0.3` 정지 시 reward 0 → 정지 발 들기 부작용 자동 차단 |
| (h) sim-to-real | current_air_time/contact_time은 reward 내부 전용 → CLAUDE memory §1 준수 ✅ |
| (i) dragging과 conflict | dragging: contact ∧ vel>0.05 시 -, gait: 4발 air/contact 시간 동기 시 +. 서로 다른 신호. 직교 ✅ |
| (j) Spot은 manager_based, parkour는 direct env | 직접 reward 계산 코드 인라인. 함수 구조는 본질 동일 |

---

## 7. Risk & Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| weight 0.5도 강해 학습 봉쇄 | 낮음 | 양수-only이므로 봉쇄 가능성 매우 낮음. smoke run에서 `episode/feet_gait_pairing` 추세 모니터링, 비활성 (=항상 0) 또는 너무 빨리 saturate 시 std 조정 |
| gradient dead zone (sync×async=0) | 중간 | std=0.2 (Spot 0.1보다 완화), max_err=0.3 (Spot 0.2보다 완화) → 초기 학습에서 0에 빠르게 빠지지 않도록 설정 |
| async reward가 학습 초기에 항상 0 | 중간 | air_time vs contact_time 교차 비교는 학습 초기 어렵. sync만으로도 trot 형성 가능 → 필요 시 async 제거하고 sync만 사용 (옵션) |
| index-based pair 매핑 오류 | 낮음 | 기존 hind feet assertion으로 부분 검증. 추가 init-time assertion 가능 (FL/FR/RL/RR 4개 모두 명시 확인) |
| Spot weight=10.0 대비 0.5는 너무 약함 | 중간 | 첫 학습 결과 확인 후 1.0→2.0→5.0 sweep 권장 |

---

## 8. 제약 메모 self-check (project_parkour_analysis_constraints)

| 제약 | 본 계획 |
|---|---|
| §1 Contact obs 추가 금지 | ✅ contact_sensor 데이터 reward 내부 전용 |
| §2 새 reward 도입 | ✅ 2026-05-13 해제됨. IsaacLab Spot 검증 패턴 |
| §3 Reward scale 단독 조정 금지 | ✅ 신규 reward 추가 (구조 변경) |
| §4~7 (action latency / actuator / torque / height_scan) | 모두 무관 |

---

## 9. 사용자 확인 요청 사항

1. **weight = 0.5로 시작** 동의? (Spot 10.0의 1/20, 매우 보수적. feet_air_time 학습 봉쇄 교훈)
2. **std=0.2, max_err=0.3, velocity_threshold=0.3** 동의?
3. **sync × async 모두 적용** 동의? (또는 async 제거하고 sync만으로 단순화?)
4. validate-method 검증 후 reward-worker dispatch OK?
