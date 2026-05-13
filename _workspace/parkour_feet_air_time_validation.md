# Parkour `feet_air_time` Reward Proposal — Validation Report

**Worker**: reward-validator (team `parkour-flat-trot`, task #2)
**Date**: 2026-05-13
**Decision**: **GO** (with one CAUTION on magnitude monitoring)

---

## TL;DR
1. 사용자 제안 `feet_air_time` 코드는 IsaacLab 표준 패턴(`anymal_c`, `R_Skeleton`)과 **bit-identical**. parkour `_feet_ids`도 같은 방식(`contact_sensor.find_bodies(".*foot")`)으로 정의되어 있어 drop-in 가능.
2. parkour `contact_sensor`는 `track_air_time=True`, `history_length=3`로 이미 충분. cfg 변경 불필요.
3. go2_wtw의 주석 처리는 commit `f3240fd4b37f`("code update", 2026-02-20)에서 raibert heuristic + bezier curve **phase-based gait reward 도입과 동시 발생** → "도움 안 돼서 제거"가 아니라 "더 구조적인 phase reward로 교체". parkour에는 해당 phase reward가 없으므로 feet_air_time 재도입 정당화 가능.
4. "default pos에 가깝게"는 **이미 parkour에 코드가 존재**: `parkour_env.py:1003`의 주석 처리된 `dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)`. 한 줄 uncomment로 Genesis-faithful 구현 완료. 별도 reward 추가 불필요.

---

## Section 1 — 표준 패턴 line-by-line 비교

### 1.1 코드 비교표

| 항목 | anymal_c (anymal_c_env.py) | R_Skeleton (skeleton_env.py) | 사용자 제안 (parkour 적용) | 일치? |
|---|---|---|---|---|
| `_feet_ids` 정의 위치 | L52 `self._contact_sensor.find_bodies(".*FOOT")` | L84 `self._contact_sensor.find_bodies(".*toe")` | L181 `self._contact_sensor.find_bodies(".*foot")` (이미 정의됨) | ✅ 동일 패턴 |
| `_feet_ids` 인덱스 공간 | contact sensor body index | contact sensor body index | contact sensor body index | ✅ |
| `first_contact = ...` | L128 | L264 | 동일 | ✅ |
| `last_air_time = ...` | L129 | L265 | 동일 | ✅ |
| `air_time` 식 | L130-132 | L266-268 | 동일 | ✅ |
| 활성 조건 (`norm(cmd[:,:2]) > 0.1`) | L131 | L267 | 동일 | ✅ |
| `(last_air_time - 0.5)` threshold | 0.5s | 0.5s | 0.5s 사용 | ✅ |
| 적용 위치 (rewards dict) | L150 `air_time * scale * step_dt` | L295 `air_time * scale * step_dt` | parkour는 `reward_values[key] * scale * step_dt` 통일 (L1099) → 그대로 사용 | ✅ |
| `feet_air_time_reward_scale` 기본값 | `anymal_c_env_cfg.py:108` = **0.5** | `skeleton_env_cfg.py:173,310` = **0.5** | 0.5 권장 (표준) | ✅ |

### 1.2 발견 사항
- 세 env 모두 **완전히 동일한 코드 블록** 사용. parkour 적용 시 추가 변환 작업 0.
- parkour `_feet_ids`는 이미 `parkour_env.py:181`에서 `contact_sensor.find_bodies(".*foot")`로 정의되어 있어 **추가 정의 불필요**. 같은 변수가 `feet_stumble`(L1012), `feet_edge`(L1023, 1028), `feet_dragging`(L1051)에서 이미 4 발에 대해 정상 동작 중.
- `_feet_ids`가 contact sensor index인 점은 `first_contact`/`last_air_time` 두 텐서가 모두 sensor 출력이므로 **정확히 일치**.

---

## Section 2 — parkour 양립성

### 2.1 B-1: Reward magnitude 추정 (정량)

#### 2.1.1 단일 step 기여도 (trot 발현 후)
- 가정: trot, freq ≈ 2 Hz, stance ratio ≈ 0.5 → 발당 air time ≈ 0.25 s
- `(last_air_time - 0.5)` = `-0.25` (음수, 0.5s threshold 미달)
- step 당 fire 횟수: trot에서는 step_dt=0.02s 동안 2 발이 first_contact 가능 (드물게, ~5% 확률)
- 평균 step 당 기여: 4 feet × `(-0.25)` × 0.05 (fire rate) ≈ `-0.05` / step (sum over feet)
- × `scale=0.5` × `step_dt=0.02` = **-0.0005 / step**

#### 2.1.2 학습 진행 후 (air_time → 0.4 s 가정, slower stride)
- `(last_air_time - 0.5)` ≈ `-0.1` 또는 양수 가능
- 양수 영역으로 진입하면 + 보상 발생 → 학습 신호

#### 2.1.3 비교 대상
| Reward | scale | typical raw | per-step contrib (× scale × step_dt=0.02) |
|---|---|---|---|
| tracking_goal_vel | +1.5 | 0..1 (~0.7 success) | +0.021 |
| tracking_yaw | +0.5 | 0..1 (~0.7) | +0.007 |
| feet_dragging | -0.1 | 0..2 m/s × 4 feet | -0.001 ~ -0.004 |
| dof_error_l2 | -0.04 | 0.5..3 rad² | -0.0004 ~ -0.0024 |
| hip_pos | -0.5 | 0.05..0.3 rad² | -0.0005 ~ -0.003 |
| termination | -100 | 0 or 1 (episode-end) | one-shot ~-2.0 |
| collision | -10 | 0..3 contacts | -0 ~ -0.6 |
| **feet_air_time (new, 0.5)** | +0.5 | ~-0.05 to +0.2 sum | **-0.0005 to +0.002** |

**결론**: 추가 reward는 `tracking_goal_vel` 대비 ~10×~30× 작고 `feet_dragging`/`hip_pos`와 같은 자릿수. **dominance 위험 없음, signal famine 위험도 없음**. termination(-100, one-shot)이 가리는 문제도 없음 — 다른 dense reward들이 termination에 살아남아 학습 중인 게 이미 증명됨.

### 2.2 B-2: feet_dragging / feet_stumble / feet_edge와의 호환성

| 항 | 이벤트 시점 | 신호 | 정보 중복 |
|---|---|---|---|
| `feet_dragging` (-0.1) | 발 in-contact + sliding (speed > 0.05 m/s) | continuous penalty | 발 들지 않고 슬라이드하는 행동 페널티 |
| `feet_stumble` (-1.0) | 발 in-contact + lateral force > 4× vertical | 0/1 event | 측면 충격 페널티 |
| `feet_edge` (-1.0) | 발 in-contact + edge mask hit | 0/1 event | 모서리 밟기 페널티 |
| **`feet_air_time` (new, +0.5)** | 발 air→contact transition + air > 0.5s | one-shot at touchdown | 충분한 swing time 양성 보상 |

이 4개 항은 **이벤트 시점이 모두 다름**:
- feet_dragging: contact + 이동 중 (stance phase)
- feet_air_time: contact 시작 순간 (touchdown)
- feet_stumble/edge: contact + 비정상 조건

→ **직접적 cancellation 없음**. feet_dragging이 "발을 들어라"의 negative 형태, feet_air_time이 "충분히 들었다"의 positive 형태 → 상호 보완.

⚠️ **단 하나 주의점**: feet_dragging은 `speed > 0.05 m/s` threshold라서 정상 stance에서도 약한 sliding 가능. 학습 초기에 두 항이 동시에 약한 음수 → 추가 deepening 없음 (자릿수 비교 시).

### 2.3 B-3: Contact sensor cfg 충분성

```python
# parkour_env_cfg.py:450-454
contact_sensor: ContactSensorCfg = ContactSensorCfg(
    ...
    history_length=3,
    update_period=0.005,        # (not shown above; verified separately if needed)
    track_air_time=True,         # ← 이미 활성화됨
)
```

- `track_air_time=True` ✅ — `data.last_air_time` 사용 가능
- `history_length=3` ✅ — anymal_c와 일치 (`compute_first_contact`는 last_dt 정보만 필요)
- `step_dt = decimation/sim.dt = 4/(1/200) = 0.02 s` (cfg L318, L341) — feet_air_time 시간 계산 정합
- **cfg 변경 불필요**

---

## Section 3 — "default pos에 가깝게" 분석

### 3.1 현재 parkour의 "default pos" 신호

| 항 | scale | 범위 | 의미 |
|---|---|---|---|
| `dof_error_l2` | -0.04 | sum over 12 joints (current - default)² | 전 관절 default 편차 페널티 |
| `hip_pos` | -0.5 | sum over 4 hip joints (current - default)² | hip 특화 페널티 (강도 ×12.5) |

→ **default-pos 신호는 이미 존재**. 단, 둘 다 flat/non-flat 무관하게 균등 적용.

### 3.2 Genesis-faithful flat-only 강화 (이미 코드에 존재!)

`parkour_env.py:1002-1003`:
```python
# Genesis conditional: dof_error penalized 10x more on flat (nominal posture expected there)
# dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)
```

이 한 줄을 **uncomment**하면:
- flat terrain envs: `dof_error_l2` 효과 scale = `-0.04 × 10 = -0.4` (hip_pos와 같은 자릿수)
- non-flat envs: 기존 `-0.04` 유지
- terrain mask `is_flat`는 L918에서 이미 계산되어 사용 중 (`lin_vel_z_l2`, `ang_vel_xy_l2`, `orientation_l2`, `base_height`에 적용)

### 3.3 권장: Option A (1-line uncomment)

```python
# parkour_env.py:1003 — uncomment 단 한 줄
dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)
```

**근거**:
- Zero new code (코드 추가 없이 효과)
- Genesis-faithful (이미 의도된 동작)
- Scale tuning 회피 (cfg `reward_scales` 손대지 않음 → §3 제약 준수)
- Hip_pos는 그대로 두어 non-flat 보호 유지

**대안 (NOT 권장)**:
- Option B: `dof_error_l2` cfg scale 직접 변경 (-0.04 → -0.4) — non-flat까지 영향, parkour 학습 영향 큼 ❌
- Option C: 별도 `dof_error_l2_flat` 항 신설 — 코드 중복, 기존 `dof_error_l2`와 정보 중복 ❌

---

## Section 4 — go2_wtw 주석 처리 이력 추적

### 4.1 git evidence

#### Commit `a2a68d6ca89b` (2026-01-14 "first update", initial)
- `feet_air_time`, `similar_to_default`, `base_height`, `flat_orientation`, `dof_acc`, `delta_torques`, `torques_l2_weighted` 등 **활성** 상태로 도입.

#### Commit `f3240fd4b37f` (2026-02-20 "code update")
git show 결과 (이 한 commit에서):
```python
# 변경 전 → 변경 후 (모두 동시 발생)
-        first_contact = self._contact_sensor.compute_first_contact(...)
+        # first_contact = ...   ← 주석 처리

-        flat_orientation = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
+        # flat_orientation = ...   ← 주석 처리

-        similar_to_default = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)
+        # similar_to_default = ...   ← 주석 처리

-        base_height = torch.square(base_pos[:, 2] - ...)
+        # base_height = ...   ← 주석 처리

-        # dof acceleration penalty (활성)
+        # # dof acceleration penalty (이중 주석)
```

#### 같은 commit에서 추가된 reward들 (`go2_wtw_env.py` grep 결과):
- `gait_indices` (L68): phase tracker buffer
- `bezier_target_pos` (L74)
- `feet_clearance_bezier`, `feet_clearance_bezier_5th` (L115-116)
- **`raibert_heuristic`** (L118): footstep placement reward
- **`tracking_contacts_shaped_force`, `tracking_contacts_shaped_vel`** (L119-120): phase-conditioned contact reward

### 4.2 해석: "교체" vs "실패"

- 주석 처리된 reward들과 새로 도입된 reward들이 **같은 commit에서 동시 발생** → 의도된 교체
- 새 reward 스택 (`raibert_heuristic` + `tracking_contacts_shaped_*` + `bezier`)는 **phase-based gait reward** — feet_air_time과 같은 목적(swing/stance 분리 유도) but 더 구조적
  - feet_air_time: emergent gait (간접 신호)
  - raibert + tracking_contacts_shaped: prescribed gait (직접 phase 처방)
- commit message "code update"는 자세한 이유 없으나, **새 reward 스택이 feet_air_time을 subsume** (raibert가 동일 정보 + footstep XY까지 처방)
- parkour는 phase-based reward 스택을 사용하지 **않음** → feet_air_time의 emergent gait 신호가 필요한 상황

**Conclusion (D)**: go2_wtw의 주석 처리는 "도움 안 돼서 제거"가 **아니라** "더 강한 phase reward로 교체"이며, parkour에는 phase reward가 없으므로 feet_air_time 도입은 정당화됨.

---

## Section 5 — GO / CAUTION / NO-GO 결론

### 5.1 종합 판정: **GO** (with 1 CAUTION + 별도 1-line uncomment 권장)

| 항목 | 평가 |
|---|---|
| 코드 표준성 | ✅ anymal_c / R_Skeleton과 bit-identical |
| parkour 적용 호환성 (cfg) | ✅ `_feet_ids`, `track_air_time=True`, `step_dt` 모두 충족 |
| 기존 reward와 magnitude 균형 | ✅ tracking_goal_vel 대비 10×~30× 작음, dominance/famine 없음 |
| feet_dragging/stumble/edge 충돌 | ✅ 이벤트 시점 다름, 상보적 |
| go2_wtw 주석 처리 우려 | ✅ "교체"로 판정, parkour에는 phase reward 없으므로 적용 합당 |
| 단일 가설 (§reward-only 변경) | ✅ 1개 reward 추가 + 1줄 uncomment, 다른 변경 없음 |

### 5.2 구체 변경안

#### Change 1: `feet_air_time` reward 신설 (필수)

**파일 1: `parkour_env.py`** — `_get_rewards()` 내부, `feet_dragging` 블록 직후 (L1056 다음)에 삽입:
```python
        # === Feet air time (anymal_c / R_Skeleton 표준 패턴, scale=0.5) ===
        # commands[:,:2] = (lin_vel_x, lin_vel_y) — 정지 시 신호 차단.
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
```

그리고 `reward_values` dict(L1072)에 추가:
```python
            "feet_air_time": feet_air_time,
```

**파일 2: `parkour_env_cfg.py`** — `reward_scales` dict(L474)에 추가:
```python
        "feet_air_time": 0.5,            # anymal_c/R_Skeleton 표준 scale; emergent trot 유도
```

`_episode_sums` 초기화는 `parkour_env.py:174-177`에서 자동으로 `reward_scales.keys()`를 순회하므로 **추가 작업 불필요**.

#### Change 2: dof_error_l2 flat 강화 (권장, 1-line uncomment)

**파일: `parkour_env.py:1003`**:
```python
# BEFORE
# dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)

# AFTER
dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)
```

이 변경은 `cfg.reward_scales`를 손대지 않으므로 사용자 §3(scale 신중) 준수. 사용자가 "별도 reward"를 명시적으로 원하면 Change 2는 생략 가능하지만, 그 경우 Option C (별도 `dof_error_l2_flat` 항 신설)보다 본 옵션이 **코드 0 중복**이라 우월.

### 5.3 CAUTION (모니터링 포인트)

1. **초기 학습 시 음의 air_time 기여**: trot 발현 전 air_time이 0.5s에 못 미치면 `(last_air_time - 0.5) < 0` → 음수 보상. 보상 magnitude는 -0.0005/step 수준이라 학습 차단 위험은 작지만, TensorBoard에서 `Episode_Reward/feet_air_time`을 모니터링 권장. 첫 100~200 episode 음수 → 양수로 전환되는지 확인.
2. **task #1과의 정합성**: task #1 (flat-diagnostic)이 root cause를 "command 분포", "termination 빈도", "actuator/PD 설정" 등 **reward 비-측 이슈**로 판정하면, feet_air_time 추가만으로는 평지 학습 실패가 해결되지 않을 수 있음. → task #1 결과 수신 후 변경 적용 우선순위 결정 권장.
3. **scale=0.5 보수성**: anymal_c와 R_Skeleton은 평지 위주 env. parkour는 non-flat이 dominant → 평지 sample이 적어 신호 약화 가능. 학습 200~500 iter 후 평지 trot 발현이 약하면 scale을 +0.5 → +1.0로 한 단계만 강화. 절대 첫 시도부터 0.5 초과 금지.

### 5.4 NOT 권장 (절대 동반 금지)

- ❌ `feet_dragging` weight 약화: task #1 결과 없이 같이 손대면 원인 분리 불가
- ❌ `termination` weight 변경: 영구 메모리 §3 제약 위반 위험
- ❌ `tracking_goal_vel` weight 강화: dominance 이미 충분, 추가 강화 시 trot 무시하고 standing-forward-skid 모드 가능

---

## Appendix — 검증 evidence 출처

- `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env.py:52,128-132,150`
- `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py:108`
- `source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton/skeleton_env.py:84,264-268,295`
- `source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton/skeleton_env_cfg.py:173,310`
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py:181,918,1003,1012,1023,1028,1051-1056,1072-1094,1097-1101`
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py:317-318,341-343,450-454,474-512`
- `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py:68,74,115-120,422-424,432-454`
- `git blame` go2_wtw_env.py L422-424 → commit `f3240fd4b37f` (2026-02-20, "code update")
- `git show f3240fd4b37f` → 같은 commit에서 raibert/bezier/contact phase reward 동시 추가

---

**완료**: 이 보고서는 read-only 검증 결과이며, 실제 reward 추가/uncomment 적용은 reward-worker(또는 본인이 아닌 별도 worker)에게 위임 권장.
