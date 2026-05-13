# Parkour 잔존 결함 RCA v2 — 좌측 뒷다리 비대칭 + 계단 정지

**Date**: 2026-05-13 (v2: 같은 날 deploy 제약 반영)
**대상 run**: `logs/rsl_rl/go2_parkour/2026-05-12_18-23-51_change_various_things_0.5_yaw_diff`
**Team**: `parkour-asym-rca` (lead = team-lead, workers = asym-auditor + fix-auditor)
**입력 보고서**:
- `_workspace/parkour_indexing_audit.md` (asym-auditor)
- `_workspace/parkour_fix_application_audit.md` (fix-auditor)

**v1 → v2 변경 사유**: 사용자가 "Go2 실제 deploy 시 contact 센서 사용 불가"를 추가 제약으로 명시. v1의 1순위 권장 Fix-3 (contact obs를 policy proprio에 추가)는 sim-to-real 양립 불가로 **무효화**. 대안 액션으로 재정렬.

**Tier discipline**: A = log evidence, B = code-surface diff, C = inference. 1순위 가설은 Tier C 단독 금지.

**Hard constraints (사용자 + 프로젝트 메모리, 영구 기록: `project_parkour_analysis_constraints.md`)**:
1. **Contact sensor 기반 obs 추가 금지** (sim-to-real deploy 제약, v2 신규)
2. Genesis(`/home/lgb/RobotSW_Genesis/parkour`)에 없는 새 reward 도입 금지
3. Reward / hyperparameter scale 튜닝 단독 액션 금지 (사용자 이미 충분 시도)
4. Action latency 무관
5. `_actuator_mode=2` blame 금지
6. 토크 envelope 정량 추론 Tier-2 금지
7. Height_scan 의심 금지
- 차단된 이전 보고서: `_workspace/parkour_step_stair_gap_RCA_summary.md`, `_workspace/parkour_step_stair_gap_rca_codeside.md` (1순위 = actuator_mode → §5 위반)

---

## 0. TL;DR (v2)

**v1과의 차이**: contact-obs 권장은 deploy 제약으로 제거. Sim-to-real 양립 가능한 dim-preserving 패치 + Genesis와의 reward/obs 구현 divergence 조사로 액션 축이 이동.

| 증상 | 원인 분류 | 1순위 가설 (v2) | Tier | 액션 |
|------|---------|---------------|------|------|
| #1 평지 왼쪽 뒷다리 비사용 | 학습 dynamics (3-leg local optimum) | Fix-2 미적용으로 yaw temporal context 부재 + Fix-1 in-flight stale로 yaw 진동 → 정책이 3-leg 안정점 락-인 | **B (간접)** | Fix-2 + Fix-1 in-flight + atan2 wrap 복구 batch 적용 후 재학습 |
| #2 허들 왼쪽 뒷다리 과도 들어올림 | 학습 dynamics | 동일 (yaw context 부재로 swing 위상 부정확) | **B (간접)** | 동일 |
| #3 계단/step 정지 | **미해결 root cause** (deploy 제약으로 contact-obs 경로 닫힘) | Genesis는 contact obs 없이도 동작 → Isaac과의 reward 계산 / obs 표현 / terrain init divergence가 진짜 원인. v1의 contact-obs 가설은 **deploy 제약과 양립 불가하므로 폐기** | **B (지표만)** | **Genesis ↔ Isaac line-by-line divergence 조사 (다음 이터레이션 신규 worker dispatch)** |

**1순위 액션 (즉시, 위험 낮음)** — dim-preserving 3종 batch:
- **Fix-2**: `parkour_env.py:868`의 `proprio_for_history[:, :2] = 0` 제거 (yaw history mask 해제)
- **Fix-1 in-flight**: `parkour_env.py:593-626`의 yaw refresh를 `do_global_refresh` gate 밖으로 이동 (height_scan만 gate 안에 유지)
- **Fix-1 regression 복구**: line 617-618 / 625-626의 `atan2(sin, cos)` wrap 주석 해제

→ obs dim 변화 없음(checkpoint 호환). 500-1000 iter 재학습. 증상 #1/#2는 어느 정도 완화 기대, #3은 부분 완화 (가능성).

**2순위 액션 (다음 이터레이션)** — Genesis divergence 조사 (사용자 제약 내):
- Genesis `legged_env_parkour.py`와 Isaac `parkour_env.py`의 **같은 이름 reward 함수**(feet_dragging, feet_stumble, feet_edge, tracking_goal_vel, tracking_yaw 등)의 **구현 line-by-line 비교**. 같은 이름이 같은 계산을 보장하지 않음. divergence가 있으면 그건 **버그 수정**(새 reward 도입 아님)이므로 사용자 §2 제약과 양립.
- Observation 표현 — Genesis의 12-joint state가 Isaac과 동일하게 인코딩되는지 (scale 0.05, default_pos 차감 등 정확히 같은지).
- Terrain curriculum / spawn 자세 분포 — Isaac이 stair에 더 일찍/자주 노출되는지.

---

## 1. 두 클래스로 분리

증상 #1, #2 (좌측 뒷다리 한정)와 #3 (계단 정지)은 다른 메커니즘일 가능성이 높음. 한 묶음으로 보면 진단이 흐려진다.

### 1.1 좌측 뒷다리 한정 비대칭 (#1, #2)

대칭 로봇 + 대칭 reward에서 **한 쪽 다리만** 실패하는 패턴의 후보:

| 후보 | 평가 (v2) |
|------|------|
| (a) joint/foot 인덱싱 mismatch (intra-Isaac 불일치) | 코드 구조상 강한 부정 — 모든 lookup이 URDF iteration order 따름. 자세한 분석 §2. **단, runtime 검증 미완** (training 일시 중단 후 §4.4.4) |
| (b) 학습 dynamics — 3-leg local optimum | **여전히 유력**. PPO에서 4-leg gait를 균형 있게 학습하기 어려운 환경에서 흔히 발생. 일단 락되면 reward gradient가 4-leg 회복을 약하게만 유도. **단, 단독으로는 "왜 Genesis는 안 락-인되는가?"를 설명 못함** → §5의 reward/obs/terrain divergence가 진짜 메커니즘일 가능성 |
| (c) USD asset 비대칭 (mass/inertia/COM) | 가능하나 표준 Go2 asset이라 가능성 낮음. §4.4.4에서 검증 |
| (d) Fix-2 미적용으로 yaw history 마스킹 + 정지 시 yaw 진동 | 부분 기여 가능 — 진동이 한쪽으로 치우치면 그쪽 다리 부담 증가. §4.1 batch가 이 부분을 해소 |
| (e) Genesis와의 reward 계산 divergence | **v2 새 후보**. 같은 이름 reward가 다르게 계산되면 한쪽 다리 reward가 다른 한쪽보다 더 강하게 페널티 → 비대칭. §5.1 |

→ **v2 1순위 후보**: (d) + (e). (d)는 §4.1 batch로 즉시 처리, (e)는 다음 이터레이션 Genesis diff worker. (b)는 메커니즘 설명이지 단독 fix가 없음(scale 튜닝 금지 §3). (a)는 부정 우세이나 마지막 backup 검증 대상.

### 1.2 계단/step 정지 (#3) — v2 재평가

v1에서는 "Fix-3 미적용 = contact obs 부재 → gait-blind → freeze"로 설명했으나 v2에서 **deploy 제약 추가 + Genesis는 contact obs 없이 동작**이라는 두 사실로 이 설명은 무효:
- Fix-3 적용 불가 (sim-to-real 제약)
- Genesis가 contact obs 없이도 stair에서 정상 동작 → contact obs 부재가 root cause라면 Genesis도 같은 증상을 보였어야 함

→ **v2 가설**: stair freeze는 Isaac-only 결함이며, Genesis와의 reward/obs/terrain divergence가 진짜 원인. 구체 후보:
- (i) Isaac의 `tracking_goal_vel` reward 계산이 Genesis와 다른 방식으로 계산되어, 정지 상태(v=0)에서도 충분히 좋은 reward를 받음 → 정지 안정점 형성
- (ii) Isaac의 stair terrain difficulty 곡선이 Genesis보다 가파름 → 정책이 stepping을 학습할 reward gradient를 못 만남
- (iii) Reset 자세가 stair 위에 떨어지는 경우, 정책이 학습할 수 없는 상태에서 시작해 reward floor에 갇힘
- (iv) `feet_dragging` 또는 `feet_edge` reward가 Isaac에서 stair 위에서 항상 페널티 영역을 만들어, 정책이 발을 들어 올리는 시도 자체를 회피

→ **§4.4.1 (reward diff) + §4.4.3 (terrain/spawn)이 1순위 검증 대상**. 본 RCA의 §4.1 dim-preserving batch는 #3에 직접 효과 없음(yaw 신호 품질만 개선), 다만 batch가 끝난 시점에서 freeze 패턴이 약간이라도 변하는지 관찰하면 yaw 신호의 기여도 측정 가능.

---

## 2. 인덱싱 가설의 부정 (asym-auditor 결과 재해석)

asym-auditor는 **Genesis(`FR→FL→RR→RL`) vs Isaac(`FL→FR→RL→RR`)**의 joint order mismatch를 "smoking gun"으로 결론지었다 (`_workspace/parkour_indexing_audit.md` §4). 그러나 이는 **framing error**이며, 본 RCA의 결론에 반영하지 않는다.

### 2.1 왜 Genesis vs Isaac 차이는 무관한가

- 이 run은 **Isaac에서 from-scratch로 학습**한 정책. Genesis와의 transfer가 아니다.
- 학습은 "Isaac 환경의 obs 분포 → Isaac 환경의 action 적용 → Isaac 환경의 reward" 닫힌 루프.
- Genesis가 어떤 순서로 dof를 등록하든 무관. 중요한 건 **intra-Isaac 일관성**: `joint_names` 순서, `action` 적용 순서, `_feet_ids` 순서, contact 텐서 dim 순서가 서로 일치하는가.

### 2.2 intra-Isaac 일관성 — 코드 구조상 강한 추정

| 텐서/리스트 | 결정 메커니즘 | 결과 순서 |
|------------|---------------|----------|
| `self._robot.data.joint_names` | URDF 파싱 후 PhysX articulation iteration order | FL → FR → RL → RR (3 joint씩) |
| Action 적용 | DCMotor가 joint_names_expr 정규식으로 매칭 → 매칭 순서 = joint_names 순서 | 동일 (URDF order) |
| `self._feet_ids = _contact_sensor.find_bodies(".*foot")` | PhysX body iteration order | URDF body order — [FL_foot(8), FR_foot(14), RL_foot(20), RR_foot(26)] |
| `_foot_body_ids = _robot.find_bodies(".*foot")` | 동일 메커니즘 | 동일 순서 |
| `_last_contacts[:, i]` | `net_contact_forces[:, 0, self._feet_ids]`에서 indexing → dim 1 = `_feet_ids` 순서 | 동일 |
| `feet_dragging`, `feet_stumble`, `feet_edge` reward | 모두 `self._feet_ids` 또는 동일 인덱싱 사용 | 동일 |

→ **모두 URDF iteration order로 정렬**되며 내부적으로 일관 (Tier B from code structure). PhysX articulation의 body iteration이 USD/URDF 순서를 따른다는 것은 documented 동작.

### 2.3 검증 시도 결과 (실패)

advisor 가이드에 따라 `parkour_env.py:228-235`의 runtime print를 캡처하려 시도:
- `/tmp/inspect_parkour_indices.py` 작성 — env 인스턴스 후 `joint_names`, `_feet_ids`, `body_names[_feet_ids]` 출력.
- 두 번 실행:
  1. GPU 0: CUDA OOM (671MB 할당 실패 — train.py가 점유)
  2. GPU 3: kvdb plugin 충돌로 env 생성 무성공 종료 (PID 627623 train.py가 5/11부터 활성)
- **결론**: 사용자 training이 활성 상태에서는 단독 env 인스턴스화가 안정적이지 않음. **future verification**: 학습 일시 중단 후 inspect 스크립트 재실행 (`./isaaclab.sh -p /tmp/inspect_parkour_indices.py`).

→ 본 RCA는 코드 구조 기반 강한 추론(Tier B)으로 인덱싱 일관성을 가정한다. runtime capture로 비대칭이 발견되면 본 결론을 1순위로 격상해야 함.

### 2.4 그래도 좌측 뒷다리 한정 비대칭은 왜?

Tier B 가정 하에서 가장 설득력 있는 설명:
- **3-leg local optimum**: PPO에서 흔한 실패. 4-leg gait는 위상이 4개 위상을 조율해야 하는 고차 결합 문제. 정책이 한 발을 ground에 박아두는 "tripod" gait를 발견하면 단기 reward가 안정적 → policy gradient가 거기로 락.
- 좌/우 중 어느 쪽이 락될지는 **early-iter random perturbation** (seed, initial action noise) — 일관된 좌측 패턴은 이번 run에 한정.
- **Fix-3가 적용됐다면** 정책은 어느 발이 들려 있는지 명시적 신호를 받음 → 3-leg gait의 contact 패턴이 "1발 항상 ground" → reward gradient가 4-leg로의 transition을 강화. Fix-3 부재가 락-인을 풀어내지 못하게 함.

이 설명은 Tier C(이론적 추론)이므로 단독 1순위로 쓰지 않고, **Fix-3 미적용이라는 Tier B 사실에 부수**되어 제시된다.

---

## 3. Fix 적용 상태 (fix-auditor 결과)

| Fix | 위치 | 적용 상태 | 영향 |
|-----|------|----------|------|
| Fix-1 (yaw refresh 분리) | `parkour_env.py:583-596` → 현재 593-626 | **PARTIAL** (`2aab9ee`) — reset-side만, in-flight side 미적용 | 정상 step 중 yaw 80ms stale 유지. 정지 시 진동(증상 #2의 yaw 부분) 미해소 |
| Fix-1 regression | line 617-618 / 625-626 | `atan2(sin, cos)` wrap이 주석 처리됨 → unwrapped `yaw_raw` 사용 | ±π 경계에서 yaw_diff 부정확. 추가 위험 |
| Fix-2 (history yaw unmask) | `parkour_env.py:868` (`proprio_for_history[:, :2] = 0`) | **NOT APPLIED** | yaw 진동 정책 학습 불가 (P-only 게이트) |
| **Fix-3 (contact obs in proprio)** | `parkour_env.py:805-816` (cat) + `parkour_env_cfg.py:331,335` (dim) | **NOT APPLIED — 그리고 적용 금지 (v2)**. `_last_contacts`는 reward 내부 계산에만 허용. proprio/policy obs에 추가하는 것은 **sim-to-real deploy 제약 위반** (Go2 실기에 contact 센서 없음). 영구 기록: `project_parkour_analysis_constraints.md §1` | v1에서 1순위로 잘못 권장. **v2에서 폐기**. Genesis가 contact obs 없이도 동작하므로 진짜 root cause는 다른 곳 |
| Fix-4 (moving_mask 제거) | `parkour_env.py:892-899` | **APPLIED** (`c3a7c43`) | 정상 |

→ **4개 fix 중 1.5개만 적용**. Fix-3 미적용이 가장 큰 결손.

---

## 4. 1순위 검증 실험 (v2 — dim-preserving batch)

contact-obs 경로가 deploy 제약으로 닫혔으므로, **obs dim을 유지하면서 정책 입력 품질을 개선**하는 3개 변경을 batch로 적용한다. 모두 `parkour_gait_rca.md`에서 이미 정의·검증된 fix이며, sim-to-real 양립.

### 4.1 변경 (1회 batch)

| # | 위치 | 변경 | 효과 |
|---|------|------|------|
| Fix-2 | `parkour_env.py:868` | `proprio_for_history[:, :2] = 0` **제거** | 정책 history에 yaw temporal 신호 복원 → P-only 진동 해소 |
| Fix-1 in-flight | `parkour_env.py:593-626` | yaw refresh를 `do_global_refresh` gate **밖으로** 이동 (height_scan만 안에 유지). 의사코드: <br>```<br># BEFORE (currently)<br>if do_global_refresh or recently_reset.any():<br>    if do_global_refresh:<br>        self._scan = ...<br>        self._yaw_diff = yaw_raw<br>        self._next_yaw_diff = next_yaw_raw<br>    else:<br>        self._yaw_diff[recently_reset] = ...<br>        self._next_yaw_diff[recently_reset] = ...<br><br># AFTER<br># yaw refresh: every step<br>self._yaw_diff = yaw_raw_wrapped<br>self._next_yaw_diff = next_yaw_raw_wrapped<br># scan refresh: gated<br>if do_global_refresh:<br>    self._scan = ...<br>``` | 80ms stale → 0ms. 정상 step 중 yaw 신선도 회복 |
| Fix-1 wrap 복구 | `parkour_env.py:617-618, 625-626` | `atan2(sin, cos)` wrap 주석 해제 → wrapped yaw를 위 `yaw_raw_wrapped`로 사용 | ±π 경계 부정확 제거 |

워커 위임:
- `obs-worker` — `parkour_env.py` 3곳 모두

### 4.2 학습

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Parkour-Direct-v0 --num_envs 4096 --max_iterations 1000 \
  --logger wandb --wandb-project IsaacLab-parkour --run_name dimpreserve_yaw_batch
```

obs dim 불변이므로 기존 checkpoint **fine-tune 가능**(`--resume` 옵션). 단, history mask 해제로 obs *content* 변경 → 재학습 처음부터가 더 깨끗.

### 4.3 판정 기준

**Tier A 메트릭** (wandb/tensorboard):
- `tracking_yaw` reward — 진동 감소, 평균 상승 (≥15%)
- 평지 episode_length — 증가 (정책이 안정적으로 직진)
- `dof_error_l2` 추세 — 악화 멈춤
- `curriculum/mean_terrain_level_{flat,hurdle}` — 상승

**Tier B 정성 평가** (play.py):
- 평지 rollout: 좌측 뒷다리 사용 회복 여부 — **단, 3-leg local optimum이 강하게 락-인 됐다면 본 batch만으로 부족할 수 있음** (그 경우 §4.4)
- 허들 rollout: 좌측 뒷다리 swing 정상 범위
- 계단 rollout: 본 batch만으로는 #3 완전 해소 기대 어려움 — 부분 완화 정도

### 4.4 If batch 적용 후에도 증상 잔존 (다음 이터레이션)

본 batch는 yaw 신호 품질만 개선. 증상 #3(stair freeze)과 #1/#2의 3-leg 락-인은 **다른 메커니즘이 root cause**일 가능성이 큼. 다음 worker dispatch:

**4.4.1 Genesis ↔ Isaac reward 구현 line-by-line diff** (`debug-worker` + `Explore`):
- Genesis(`/home/lgb/RobotSW_Genesis/parkour/legged_env_parkour.py`)의 `_reward_feet_dragging`, `_reward_feet_stumble`, `_reward_feet_edge`, `_reward_tracking_goal_vel`, `_reward_tracking_yaw` 함수 본문
- Isaac(`parkour_env.py`)의 동일 reward 계산 부분
- **divergence 발견 시 그건 버그 수정 대상**(새 reward 도입 아님 → 사용자 §2 제약과 양립)

**4.4.2 Observation 표현 divergence**:
- joint_vel scale (Isaac `*0.05`) vs Genesis 해당 값
- `joint_pos - default_joint_pos` 계산 시 default_joint_pos가 양 프로젝트에서 동일 dict인지
- prev_actions 정의 (last applied vs last policy output)

**4.4.3 Terrain curriculum / spawn**:
- `parkour_terrains.py`의 11×40 grid 비율 — stair/step 비율이 Genesis 대비 큰지
- spawn 자세 (default_joint_pos)가 stair 위에서 안정적인지

**4.4.4 USD asset 비대칭** (assertions 마지막 단계):
- training 일시 중단 후 `/tmp/inspect_parkour_indices.py` 실행 → 인덱싱 + body mass/inertia 캡처
- 좌/우 leg의 USD mass/COM이 비대칭이면 asset-side 수정 필요

→ 다음 이터레이션에서 위 4개를 병렬 worker로 dispatch 권장. 본 RCA의 범위 외.

---

## 5. Genesis-는-잘됨 (v2 핵심 분석 경로)

`parkour_gait_rca.md` Appendix G: **"Genesis does not [carry contact bits in policy obs]"**. 즉, Genesis는 contact obs 없이도 동작.

v1에서는 이 모순을 "open question"으로 미뤄두고 Fix-3를 권장했으나, v2에서는 **deploy 제약이 Fix-3를 닫았으므로 이 모순이 곧 root cause 분석의 주축**이 된다.

**핵심 질문**: Genesis와 Isaac이 같은 robot, 같은 reward 항 이름들, 같은 task 정의를 쓰는데, 왜 Isaac만 stair freeze + 좌측 뒷다리 락-인이 발생하는가?

가능한 분기 (사용자 §1-§7 제약 내):

### 5.1 Reward 함수 구현 divergence (가장 유력, 사용자 §2 양립)

같은 이름(`feet_dragging`, `feet_stumble`, `feet_edge`, `tracking_goal_vel`, `tracking_yaw`)이 같은 계산을 보장하지 않음. 두 프로젝트가 합쳐지지 않고 독립 진화했으므로 한쪽에 버그/divergence 가능성 큼. divergence 발견 시 그건 **버그 수정**이지 새 reward 도입이 아니므로 §2 제약과 양립.

→ §4.4.1에서 자세히 다룸.

### 5.2 Observation 표현 divergence

- joint_vel scale (Isaac `*0.05`)이 Genesis와 같은가?
- `joint_pos - default_joint_pos`의 default_joint_pos가 양 프로젝트 동일 dict인가?
- prev_actions 정의 (Isaac은 raw action vs Genesis는 last applied)?

→ §4.4.2

### 5.3 Terrain curriculum / spawn 분포

- Isaac의 11×40 terrain grid에서 stair/step 비율이 Genesis 대비 큰가?
- difficulty curriculum 곡선이 더 가파른가?
- spawn 자세가 stair 위에 떨어지는 빈도가 높은가?

→ §4.4.3

### 5.4 USD asset 비대칭 (좌우 비대칭 의심 시)

- 좌/우 leg의 mass/COM/inertia 비대칭? 표준 Go2 asset에서 매우 드물지만 좌측 뒷다리 한정 패턴이라 검증 가치 있음.

→ §4.4.4

### 5.5 PhysX vs Genesis 솔버 차이 (분석 영역 외 → 닫힘)

- contact 응답 dynamics 차이로 implicit contact 신호의 SNR이 다를 수 있음.
- 그러나 솔버는 사용자가 바꿀 수 없는 영역. **분석/수정 대상 아님**.

---

## 6. 되돌리지 말 것 (이전 commit 보존)

- `2aab9ee`의 reset-side stale buffer 패치 (`_yaw_diff`, `_next_yaw_diff`, `_scan` 초기화) — bug fix, 유지
- `c3a7c43`의 `moving_mask` 제거 — Fix-4, 유지
- `c3a7c43`의 actuator 조정 (`_actuator_mode`) — 본 RCA 영역 외 (메모리 차단). 이 RCA에서 액션 미제시.

---

## 7. 차후 (이터레이션 계획)

### 이터레이션 N (즉시) — §4.1 dim-preserving batch
- `obs-worker`에게 Fix-2 + Fix-1 in-flight + atan2 wrap 복구 3종 dispatch
- 500–1000 iter 재학습
- training-evaluator로 §4.3 메트릭 평가
- 예상 효과: 증상 #2의 yaw 부분 완화. #1/#3은 부분 또는 무효

### 이터레이션 N+1 (필요 시) — Genesis divergence 조사
이터레이션 N 결과에 따라 다음을 병렬 dispatch (`/team`):
- worker-1 (`debug-worker` + `Explore`): §4.4.1 reward 구현 line-by-line diff
- worker-2 (`Explore`): §4.4.2 observation 표현 + §4.4.3 terrain curriculum/spawn diff
- worker-3 (`Explore`, training 중단 시점): §4.4.4 USD asset 비대칭 + `/tmp/inspect_parkour_indices.py` 인덱싱 runtime capture

divergence 발견 시 그건 **버그 수정**(새 reward/scale 도입 아님)이므로 사용자 제약 §1-§7과 양립.

### 항상 영구 참조
- 새 worker dispatch 시 `~/.claude/projects/-home-lgb-IsaacLab/memory/project_parkour_analysis_constraints.md`를 worker 프롬프트에 반드시 포함시켜 금지 영역(특히 contact obs 추가, scale 튜닝 등) 위반을 차단.

---

## 8. References

- 이번 RCA 입력:
  - `_workspace/parkour_indexing_audit.md` (asym-auditor — 본 RCA에서 framing 재해석)
  - `_workspace/parkour_fix_application_audit.md` (fix-auditor — 채택)
- 이전 RCA (재참조 가능):
  - `_workspace/parkour_gait_rca.md` (구조적 분석, 4개 fix 정의)
- 차단된 보고서 (메모리 위반):
  - `_workspace/parkour_step_stair_gap_RCA_summary.md`
  - `_workspace/parkour_step_stair_gap_rca_codeside.md`
- 프로젝트 메모리:
  - `~/.claude/projects/-home-lgb-IsaacLab/memory/project_parkour_actuator_mode2_verified.md`
  - `~/.claude/projects/-home-lgb-IsaacLab/memory/project_parkour_height_scan_verified.md`
  - `~/.claude/projects/-home-lgb-IsaacLab/memory/feedback_torque_envelope_not_tier2_against_empirical.md`

*End of RCA.*
