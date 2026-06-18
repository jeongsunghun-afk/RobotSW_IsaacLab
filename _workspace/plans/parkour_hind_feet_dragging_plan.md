# Parkour A — 뒷발 Feet Dragging Penalty 추가 계획

**작성일**: 2026-05-21
**대상 env**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/` (A env)
**문제**: 학습 결과에서 뒷다리가 지면을 질질 끄는 (dragging) 현상 관찰
**해결 방향**: **뒷발 (RL_foot, RR_foot)** 에 한정한 dragging penalty 신설

---

## 1. 배경 — 현재 상태

| 항목 | 현황 |
|---|---|
| `dragging_velocity_threshold` cfg field | **존재** (`parkour_env_cfg.py:592-593`, 값=0.05 m/s). Genesis 원본 의도. |
| `reward_scales["feet_dragging"]` | **없음** (566-581에 미정의) |
| `_get_rewards()` 내 계산 코드 | **없음** (1082-1108) |
| 결론 | **dead config** — cfg field는 남아있으나 reward 함수 자체가 미구현 |

→ 본 작업은 사용자 의도(Genesis 원본)와 달리 누락된 reward 항을 **뒷발 한정으로** 복원하는 것.

### 인프라 사전 확인 (이미 존재 → 새 buffer 추가 불필요)

| 자원 | 위치 | 비고 |
|---|---|---|
| `self._feet_ids` | `parkour_env.py:193` | `find_bodies(".*foot")` 순서 = [FL, FR, RL, RR] |
| `self._last_contacts` (N,4) bool | `parkour_env.py:169` | 2-step debounce용. `_reset_idx`에서 초기화됨 |
| `self._contact_sensor.data.net_forces_w_history` | parkour env 표준 사용 | 시간축 max로 stable contact 판정 가능 |
| `self._robot.data.body_lin_vel_w[:, feet_ids, :2]` | 표준 articulation API | 발 xy 선형 속도 |

→ **신규 buffer 추가 없음**. `_reset_idx` 수정 불필요.

---

## 2. 참고 패턴 비교 및 선택

| Pattern | 출처 | 핵심 식 | 장점 | 단점 |
|---|---|---|---|---|
| **A. Genesis 원본** | `RobotSW_Genesis/parkour/legged_env_parkour.py:1597-1610` | `Σ ‖xy_vel‖ × (contact_filt ∧ ‖xy_vel‖>thr)` | 사용자 의도 = Genesis 충실 복원. cfg threshold 재사용 | velocity threshold(0.05)로 small drag 누락 |
| B. IsaacLab `feet_slide` | `manager_based/.../mdp/rewards.py:71-85` | `Σ ‖xy_vel‖ × (max_history_force.norm > 1.0N)` | IsaacLab 표준, 노이즈 강건 (시간축 max) | velocity threshold 없어 weight 더 보수적 필요 |
| C. Spot `foot_slip_penalty` | `…/spot/mdp/rewards.py:237-251` | B와 동일 + threshold 파라미터화 | 명시적 contact threshold | A와 본질 차이 없음 |

### 선택: **Pattern A (Genesis-fidelity)** + 뒷발 한정

이유:
1. `dragging_velocity_threshold=0.05` cfg field가 이미 Genesis 원본 의미로 정의됨 → **dead config 활성화**
2. 과거 실험 (`_workspace/parkour_experiment_review/REPORT.md`)에서 weight **-0.1**이 검증된 sweet spot
3. `_last_contacts` 2-step debounce가 이미 존재 → Genesis의 `contact_filt`와 동일 의도

---

## 3. 변경 사항 (정확한 위치, 최소 침습)

### 3.1 `parkour_env_cfg.py` — `reward_scales` dict에 항 추가

**위치**: `parkour_env_cfg.py:566-581` (reward_scales dict)

**추가 항 (1줄)**:
```python
"feet_dragging": -0.1,   # hind feet only; threshold = dragging_velocity_threshold
```

- 위치: 기존 `feet_edge`, `feet_stumble` 인접에 배치 (의미적 그룹)
- weight = **-0.1** (과거 실험 검증된 sweet spot)
- threshold는 기존 `dragging_velocity_threshold=0.05` 재사용 (이미 cfg field)

### 3.2 `parkour_env.py` — `_get_rewards()` 내 reward 계산 추가

**위치**: `_get_rewards()` 함수 내, 기존 `feet_edge` 계산부의 **`contact_filt` 생성 직후, `self._last_contacts = contact` 대입 이전**
(현재 코드 기준 line ~1061 직후)

> ⚠️ **validate-method H-2 (Red) 반영**: 반드시 `contact_filt[:, 2:4]`를 사용해야 함.
> `self._last_contacts`는 `contact_filt` 계산 **이후** `contact`로 덮어쓰기되므로, 그 이후에 참조하면 **debounce가 적용 안 된 raw 현재-step contact**가 됨. Genesis 의미론과 어긋남.

**추가 코드 (개념적 6~8줄)**:
```python
# === feet_dragging penalty — HIND FEET ONLY (RL, RR) ===
# MUST be placed BEFORE `self._last_contacts = contact` overwrite.
# Uses contact_filt (2-step debounced) for Genesis-equivalent semantics.
hind_feet_ids = self._feet_ids[2:4]  # URDF order: [FL(0), FR(1), RL(2), RR(3)]
hind_xy_vel_norm = torch.norm(
    self._robot.data.body_lin_vel_w[:, hind_feet_ids, :2], dim=-1
)  # (N, 2)
hind_contact = contact_filt[:, 2:4]  # (N, 2) bool — debounced
is_dragging = hind_contact & (hind_xy_vel_norm > self.cfg.dragging_velocity_threshold)
feet_dragging = torch.sum(hind_xy_vel_norm * is_dragging.float(), dim=-1)  # (N,)
```
그리고 reward_values dict에 `"feet_dragging": feet_dragging * reward_scales["feet_dragging"]` 등록.

### 3.2.1 `__init__` 에 인덱스 assertion 추가 (validate-method H-1 반영)

`_feet_ids` 정의 직후 (parkour_env.py:193 근처)에 1회성 검증 추가:
```python
hind_foot_names = [self._contact_sensor.body_names[i] for i in self._feet_ids[2:4]]
assert all(("RL" in n) or ("RR" in n) for n in hind_foot_names), (
    f"_feet_ids[2:4] expected to be hind feet (RL/RR), got: {hind_foot_names}"
)
```
init time에 1회 비용 — runtime 영향 없음.

**핵심 설계 결정**:
- **뒷발만** (`_feet_ids[2:4]`) — 앞발은 dragging penalty 대상 아님 (사용자 명시 요구)
- **`contact_filt` 재사용** (NOT `_last_contacts`) — Genesis `contact_filt` 의미론과 일치하는 debounce
- `contact ∧ vel>thr` 두 조건 모두 충족 시에만 penalty (Genesis 패턴)
- 매그니튜드는 `xy_vel_norm` 자체 (절대값 누적) — Genesis와 동일
- 신규 buffer 없음, `_reset_idx` 수정 불필요

### 3.3 영향 없음 확인

- `_reset_idx`: **수정 불필요** (신규 buffer 없음)
- `_get_observations`: **수정 불필요** (obs 공간 변화 없음, sim-to-real 안전)
- network input/output shape: 변화 없음
- agents/*.yaml: 변화 없음

---

## 4. 검증 계획 (validate-method 호출 항목)

| 항목 | 검증 방법 |
|---|---|
| (a) Index 매핑 정확성 | `_feet_ids[2:4]`가 실제 RL/RR인지 — body name dump 확인. URDF iteration order ↔ Go2 body name 매핑 검증. |
| (b) `_last_contacts` shape/timing | (N,4) bool, `_get_rewards` 호출 시점에 **현재 step contact**인지, 이전 step인지 (debounce 정의). 매그니튜드와 일치 시점인지. |
| (c) `body_lin_vel_w` 사용 적절성 | Genesis가 world frame xy velocity 사용 → IsaacLab도 동일. yaw rotated frame 변환 불필요. |
| (d) Reward magnitude 균형 | -0.1 weight × xy_vel ~ 1 m/s × 2발 contact ≈ -0.2/step. 다른 negative reward (orientation -1, collision -10) 대비 적절한 보조 신호인지. |
| (e) sim-to-real 호환성 | contact 정보가 reward 내부에서만 사용, observation 노출 없음 → §1 (CLAUDE memory) 준수 |
| (f) cfg-code 일관성 | `feet_dragging` ↔ `dragging_velocity_threshold` 모두 cfg에 존재, runtime에서 동기화. |
| (g) 학습 봉쇄 위험 | 과거 -1.0에서 봉쇄, -0.1에서 회복 — 본 작업은 검증된 -0.1로 시작. |

---

## 5. 구현 절차 (사용자 승인 후)

1. **reward-worker dispatch**: 위 §3.1, §3.2 정확히 적용
2. **validate-code 호출**: shape, buffer, cfg-code 일관성 자동 검증 (체크리스트 A~D)
3. **smoke run** (선택): `--num_envs 64 --max_iterations 500` 단기 학습으로 NaN/explode 없는지 확인
4. **본 학습**: 사용자 트리거 시 시작

---

## 6. Risk & Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| 뒷발 인덱스 매핑 오류 (FL/RL 혼동) | 낮음 | URDF order 검증 (Explore 보고서 확인됨), reward-worker 구현 시 assert/print 추가 가능 |
| weight -0.1도 강해 stair/step 진행 방해 | 중간 | 과거 검증값이나, 본 작업은 **뒷발만**이라 effective magnitude는 50% (4발→2발) → 더 안전 |
| Genesis `contact_filt` 2.0N vs IsaacLab `_last_contacts` 정의 불일치 | 중간 | `_last_contacts` 정의(parkour_env.py 주변)를 reward-worker가 재확인하여 debounce 의미 검증 |
| `body_lin_vel_w`는 world frame — base 회전 시 추가 보정 필요? | 낮음 | Genesis 원본도 world frame xy 사용. 일관성 유지. |

---

## 7. 제약 메모 self-check (project_parkour_analysis_constraints)

| 제약 | 본 계획 준수 여부 |
|---|---|
| §1 Contact obs 추가 금지 | ✅ contact는 reward 내부 전용 |
| §2 새 reward 도입 | ✅ 2026-05-13 해제됨. Genesis 정의 존재 → 그대로 도입 |
| §3 Reward scale 단독 조정 | ✅ 본 작업은 신규 reward 추가 (단순 scale 변경 아님) |
| §4 Action latency | ✅ 무관 |
| §5 Actuator 약화 blame | ✅ 무관 |
| §6 Torque envelope 추론 | ✅ 무관 |
| §7 Height_scan 의심 | ✅ 무관 |

---

## 8. 사용자 확인 요청 사항 (구현 전)

1. **weight = -0.1로 시작** 동의? (검증된 sweet spot이나 뒷발만이므로 더 안전)
2. **뒷발만** 한정 동의? (앞발 dragging은 무시) — 사용자 요구사항대로
3. **threshold = 0.05 m/s** (기존 cfg 값) 그대로 사용 동의?
4. validate-method 검증 후 구현 진입 OK?
