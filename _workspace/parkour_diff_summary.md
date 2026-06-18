# parkour(A) vs Isaaclab_Parkour(B) 환경 차이 — 통합 비교 보고서

**작성일**: 2026-05-19
**팀**: parkour-env-diff (team-lead + worker-1~5)
**대상**: A = `source/isaaclab_tasks/.../direct/parkour/` (Direct RL, **학습 초반 학습 안 됨**)
         B = `Isaaclab_Parkour/` (Manager-based, extreme_parkour 포팅 원본, **학습 잘 됨**)
**범위**: observation 자체 구성은 비교 제외 (사용자 지시). reward / curriculum / terrain / goal / episode / control / hyperparam.

> **주의**: 본 보고서의 모든 "원인" 표현은 **가설**이다. 코드 표면 비교 기반이며 실험 검증 전이다.
> 사용자 금지 영역(actuator 약화 blame, scale 단독 튜닝, contact-obs 추가, height_scan 의심 등)은 RCA ranking에서 제외했다.

하위 보고서:
- `parkour_diff_reward.md` (worker-1)
- `parkour_diff_terrain_curriculum.md` (worker-2)
- `parkour_diff_goal.md` (worker-3)
- `parkour_diff_episode_dr.md` (worker-4)
- `parkour_diff_control_hyperparam.md` (worker-5)

---

## 1. 주요 구조 차이 — 검증 대상

**worker-1(reward) + worker-4(episode/termination) 두 축에서 독립적으로 수렴한 단일 메커니즘:**

### H1 (구조적 차이 — 검증 필요, "유력 단정" 아님) — "음수 reward 누적 + 조기 종료 출구" = suicide local optimum

| 요소 | A (학습 안 됨) | B (학습 잘 됨) |
|------|---------------|---------------|
| step total reward 집계 | clip 없음 → 음수 가능 (`parkour_env.py:1100-1107`) | `clip(min=0)` (`parkour_reward_manager.py:38`) |
| base/몸통 접촉 처리 | **종료 조건** `base_contact>5N` (`parkour_env.py:1114-1117`) | 종료 안 함, collision penalty(-10)로 벌점만 |
| 실패 종료의 value bootstrap | terminated → bootstrap=0 (`parkour_env.py:1129-1132`) | 단일 `time_out` DoneTerm → 항상 truncated → bootstrap 수행 |

**메커니즘**: iter 0의 미숙한 정책에서 A는 ① 정지·탐색 중 penalty 합으로 step reward가 **강하게 음수**가 되고, ② `base_contact` 종료라는 **즉시 탈출 경로**가 존재한다. 음수 reward 스트림에서는 에피소드를 빨리 끝낼수록 누적 return이 덜 음수가 되므로, 정책이 "빨리 넘어져 종료"하는 방향으로 수렴할 수 있다(early-termination / suicide optimum). B는 (1) reward를 0 미만으로 떨어뜨리지 않고 (2) 몸통 접촉으로 종료시키지 않아 이 함정이 구조적으로 봉쇄돼 있다.

이는 **reward scale 튜닝 문제가 아니라 reward 집계 구조 + 종료 조건 설계의 차이**다. (금지 영역 §3과 무관.)

> 검증 방법(코드 수정 전 진단): A 학습 로그에서 **에피소드 길이가 초반에 짧게 정체/감소**하는지, **base_contact 종료 비율**이 지배적인지, **mean step reward가 음수**인지 확인. 세 가지가 동시에 나타나면 H1이 강하게 지지된다.

### ⚠ 정정 — "regression"이 아니라 "처음부터 안 됨" (사용자 확인)

초안에서는 `parkour_smoke500_analysis.md`(2026-05-13)의 "PASS"를 학습 성공으로 해석했으나, **사용자 확인 결과 A 환경은 어느 시점에도 원하는 학습 결과를 낸 적이 없다.** smoke500의 "PASS"는 500-iter 자동 smoke 게이트의 관대한 통과였을 뿐 — 당시 에피소드 길이 27~28 step, `tracking_goal_vel` 0.017→0.042의 미미한 절대값, 종료 분포가 `low_height`(142/146)·`tilt` 지배 — 실제로 보행/파쿠르를 학습한 상태가 아니었다.

**함의 (프레임 교체)**:
- **"last-known-good 대비 regression" 프레임은 성립하지 않는다** — known-good이 없다. `git diff`는 "무엇이 깨졌나"가 아니라 변경 이력 참고용으로만 의미.
- 따라서 **H1을 포함한 A-vs-B 구조 차이는 선행 성공 사례에 의해 반증되지 않는다.** 모두 유효한 후보로 살아 있다.
- 본 문제의 질문은 "어느 커밋이 깼나"가 아니라 **"A의 어떤 구조가 B와 달라 처음부터 학습을 막는가"** 이다. → **A-vs-B 비교가 본 문제의 1차 도구로 맞다.**
- 단 H1·H2·C2·C3 등은 여전히 *가설*이다. 코드 표면 비교 기반이며, 아래 §4 진단으로 참/거짓을 가린 뒤 수정에 착수한다.

---

## 2. 축별 차이 요약

### 2-1. Reward (worker-1)
- 14개 reward 항 집합은 A/B **완전 일치**. 식도 tracking_goal_vel·tracking_yaw·feet_stumble·feet_edge 실질 동등.
- **구조 차이**: B만 step total reward `clip(min=0)` → H1.
- collision penalty: A는 substep history max 집계 + undesired body에 hip·head 포함, B는 최신 substep 1개 + base/calf/thigh만 → A가 더 자주·세게 발화.
- scale 차이 1건: `action_rate` A=-0.05 / B=-0.1.
- 비평지 conditional 계수 불일치(lin_vel_z, ang_vel_xy) — 평지/iter0엔 무영향, 커리큘럼 진행 후 영향.
- A cfg의 `tracking_sigma`·`yaw_reward_speed_*`·`dragging_velocity_threshold`·`base_height_target`는 미참조 dead config.

### 2-2. Terrain + Curriculum (worker-2)
- **표현 차이**: A=trimesh mesh(노이즈 없는 평탄 박스), B=height-field + 전 지형 `apply_roughness` 노이즈 → 지면 표면·접촉 분포가 다름.
- 장애물 mix 불일치: A는 B 시그니처 `parkour`(경사 stepping-stone)를 빼고 `parkour_stair`를 추가. 평지 비중 A 0.1 / B 0.2(절반), gap A 0.3 / B 0.2.
- difficulty 곡선: A의 hurdle·gap이 B보다 다소 쉬움(A "overflow fix"로 gap 범위 축소).
- 시작 레벨: A `max_init=3`(d≈0.30) / B `max_init=2`(d≈0.22) — A가 약간 높으나 장애물이 더 쉬워 대체로 상쇄 추정.
- **curriculum 진행 규칙(0.8/0.4 임계, ±1 step)은 A/B 완전 동일** — 로직 차이 없음.
- spawn: A는 env_origin 직접, B는 origin−7m 보정(각자 일관). **A의 terrain origin→env_origins 전파 정합성은 검증 권장** — 어긋나면 spawn/curriculum거리/goal좌표가 동시 편향.

### 2-3. Goal / Command (worker-3)
- A/B 모두 "goal waypoint(방향) + velocity command(전진속도)" 2-트랙 구조 동일. 도달 임계 0.2m, goal 8개/future 2, resample 6s 동일.
- **이전 진단 정정**: `parkour_flat_run_diagnostic.md`의 "A는 yaw wrapping 비활성" 의심은 **현재 코드에서 사실 아님**. 오히려 A가 `atan2(sin,cos)` wrapping을 적용하고, B(extreme_parkour 원본)가 미적용. 방향이 반대.
- A는 `_yaw_diff`(goal heading-error) obs를 **5 step(~100ms)마다만 refresh**, B는 매 step → A에 최대 ~80ms 신호 지연(의도된 10Hz gate일 수 있음).
- A는 ang_vel(wz) command 없음(`ang_vel_range=[0,0]`), B는 heading-PD로 생성.
- A의 goal 좌표는 registry 기반 world-frame 재구성 → size mismatch 시 직선 goal로 **silent fallback** 위험(B는 terrain-local 직접 로드, 위험 없음).

### 2-4. Episode / Termination / Reset / DR (worker-4)
- H1 관련 항목(§1) 외:
- A는 reset 시 **joint pos 랜덤화 없음**(항상 default 자세), B는 `reset_joints_by_scale` ±5% → A는 초기 상태 다양성 한 축 결여.
- A는 spawn 후 5 step 실패종료 grace 면제(B엔 없음) — 이 점은 A가 더 관대.
- 종료 임계: low_height A −0.2 / B −0.25, tilt 표현 다름(A proj_gravity_xy 결합 / B roll·pitch 독립 1.5rad).
- DR(friction/mass/com/push)은 대체로 정합 — A가 더 가혹하다 볼 근거 약함.
- episode 길이 구조 동일(20s, decim 4, 1000 step).

### 2-5. Control + PPO Hyperparam (worker-5)
- **PPO 하이퍼파라미터 전 항목 A=B 완전 동일** (lr 2e-4 adaptive, kl 0.01, epochs 5, mini_batch 4, gamma 0.99, lam 0.95, clip 0.2, entropy 0.01, value_coef 1.0, grad_norm 1.0, rollout 24).
- **알고리즘 구조 동일** — 둘 다 RMA-style PPO(ActorCriticRMA + scan/priv encoder + StateHistoryEncoder + estimator + priv_reg schedule). network hidden dims 동일. 클래스 이름만 다름(PPOParkour vs PPOWithExtractor).
- **A 고유 control 차이**: hip 관절 action에 추가 ×0.5 스케일(`parkour_env.py:516`) → 유효 action_scale A hip=0.125 / B 전 관절 0.25. B teacher엔 이 hip 스케일 없음.
- friction `combine_mode`: A=multiply / B=average → 유효 접지 마찰 산출식 차이.
- 차이 기타: num_envs A 4096 / B 6144, estimator lr A 1e-3 / B 1e-4(10배).
- sim dt 0.005 / decim 4 / 50Hz / action_scale 0.25 / clip ±4.8 / default offset 방식은 A=B 동일.

---

## 3. 후보 가설 — A의 "처음부터 학습 불가" 구조 차이

A는 known-good이 없으므로(§1 정정), 아래는 모두 "B와 다른, 학습을 막을 수 있는 구조 후보"다. 검증 비용이 낮은 것을 위로 둔다.

| 가설 | 내용 | 근거 worker | 상태 |
|------|------|-------------|------|
| ~~C2~~ | ~~terrain origin→env_origins 전파 정합성~~ | w2 | **기각 — §3.2 (코드 검증: 정합성 정상)** |
| ~~C3~~ | ~~goal map registry size mismatch → 직선 goal silent fallback~~ | w3 | **기각 — §3.1 (코드 검증)** |
| ~~H1~~ | ~~음수 reward 누적(clip 부재) + base_contact 종료 → suicide optimum~~ | w1+w4 | **기각 — §3.3 (최신 run empirical: A 정상 학습)** |
| H2 | reset joint pos 랜덤화 부재 → 초기 상태 다양성 결여 | w4 | 미검증 — 코드 확정 사실, 효과 미측정 |
| H5 | terrain 표현(mesh vs heightfield+roughness), 장애물 mix, 평지 비중 절반 | w2 | 미검증 — 단순 차이 |
| H6 | hip action ×0.5 스케일, friction combine_mode, estimator lr 10배 | w5 | 미검증 — 단순 차이 |
| — | goal yaw 5-step 갱신 지연, ang_vel command 부재 | w3 | 미검증 — 보조 |

> **검증 후 상태**: w1~w5가 제기한 "유력 가설"(C2/C3/H1)은 코드·로그 검증 결과 **3개 모두 기각**됐다. 남은 H2/H5/H6은 A/B 간 "단순 구조 차이"일 뿐 실패 원인으로 확인된 바 없다. → 본 보고서는 "A 실패의 원인 규명"이 아니라 **"A vs B 구조 차이 카탈로그"**로 활용해야 한다(§5 참조).

### 3.1. C3 검증 결과 — 기각 (2026-05-19, general-purpose 코드 검증)

worker-3가 제기한 "goal map registry size mismatch → 직선 goal silent fallback" 가설을 실제 코드(`parkour_env.py`, `parkour_terrains.py`, core `mesh_terrains.py`, `terrain_generator.py`)로 검증한 결과 **사실 아님**:
- 현재 cfg(`num_rows=11 × num_cols=40` = 440 타일)에서 terrain generator는 정확히 440회 terrain 함수를 호출하고, 모든 parkour 함수는 goal이 없어도 origin pad로 **무조건 1회 append** → registry 길이는 항상 정확히 440. mismatch가 발생할 구조적 경로 없음(`proportion=0` 지형은 호출조차 안 됨).
- `_build_terrain_goals_map()`(`parkour_env.py:368-426`)에 mismatch 검사·경고(`print` "[Parkour] WARN: ...")는 실재하나, `_init_env_goals()`의 fallback 분기는 `parkour_env.py:545-546` `print('no'); return`으로 즉시 종료 → 그 아래 "직선 goal 생성" 코드는 **dead code**. 즉 worker-3 보고서의 "직선 goal fallback" 묘사 자체가 코드와 불일치.
- **부수 발견 (latent bug, 현재 미발동)**: 만약 실제 mismatch가 발생하면 fallback이 무력화되어 모든 goal이 `(0,0,0)`으로 남는다. 현재 cfg에선 트리거 안 되지만 cfg 변경 시 위험. `print('no')`+`return`은 디버깅 잔재로 제거 권장.
- **남은 런타임 확인 1건**: A 기동 로그에 `[Parkour] WARN: PARKOUR_GOALS_REGISTRY size` 또는 `no`가 출력됐는지. 없으면 goal 시스템은 본 문제와 무관 확정.

### 3.2. C2 검증 결과 — 기각 (2026-05-19, 코드 검증)

"terrain origin→env_origins 전파 정합성" 가설을 `parkour_env.py`/`terrain_importer.py`/`terrain_generator.py` 코드로 검증한 결과 **정합성 정상**:
- spawn 위치(`parkour_env.py:1185`), curriculum 거리 기준(`:1264-1265`), goal world 좌표(`:415-419`) — 셋 모두 동일 anchor `terrain_origins[row,col]`로 정렬. `env_origins`는 `terrain_origins`를 `(level,type)` 인덱싱한 값 그대로(`terrain_importer.py:353`), 오프셋 없음.
- B의 `−(size[1]+offset)` 보정은 A에 "누락"된 게 아니라, A/B가 다른 goal 좌표 규약을 써서 A엔 구조적으로 불필요.
- (별개 사실) A는 로봇을 플랫폼 중앙에 spawn → B보다 첫 장애물까지 런웨이가 짧음. 좌표 버그 아닌 설계 차이.

### 3.3. H1 검증 결과 — empirical 기각 (2026-05-19, 학습 로그)

최신 A run `2026-05-19_17-20-51_isaac_4.5`(580 iter)의 TensorBoard 로그:
| 메트릭 | 초반 | 후반 |
|--------|------|------|
| `Train/mean_reward` | 2.75 | **10.69** |
| `Train/mean_episode_length` | 482 | **665** |
| `Episode_Reward/tracking_goal_vel` | 0.18 | 0.86 |
| `curriculum/mean_terrain_level` | 0.25 | 3.94 |
| `Episode_Termination/cause_base_contact` | 0.11 | 0.06 |
| `Episode_Termination/cause_tilt` | 8.0 | 0.85 |

→ 이 run은 **정상 학습**한다. 에피소드는 길고(665 step) reward는 상승하며 base_contact 종료는 지배적이지 않다. H1이 말한 "짧은 에피소드 + base_contact 종료 지배 + 음수 reward" 패턴이 **관측되지 않음**. H1 메커니즘은 최신 A에서 발생하지 않는다.

> **검증 종합**: w1~w5의 유력 가설 3개(C2/C3/H1) 모두 기각. "A의 어떤 구조가 학습을 막는가"라는 질문은, **A가 (최신 run 기준) 실제로 학습되므로** 전제부터 성립하지 않는다. §5 참조.

## 4. 권장 진단 순서 (코드 수정 전)

1. **A 기동 로그 확인 (최우선·최저비용)** — `[Parkour] WARN: PARKOUR_GOALS_REGISTRY size` / `no` (C3 잔여 확인, 없으면 goal 무관 확정) + terrain origin 불일치 경고 검색 → C2 판정. C3 가설 자체는 §3.1에서 코드로 기각됨.
2. **A 학습 로그 확인** — 초반 mean episode length, 종료 원인 분포(base_contact 비중), mean step reward 부호, reward 항별 추이. → H1 판정 + "학습이 전혀 안 되는지 / 미미하게 되는지" 구분.
3. **A-vs-B 구조 차이를 B 기준으로 1개씩 정렬** — C2가 음성이면, H1(reward clip / base_contact 종료) → H2(reset 랜덤화) 순으로 B에 맞춰보고 각각 단독 효과 측정. **한 번에 하나만** 변경(원인 분리).
4. (참고) `git log -- .../direct/parkour/` — regression 프레임은 아니지만, 최근 커밋(`25d2071`/`e81b71c`/`c7261bd`)이 어떤 의도였는지 맥락 파악용.

> **현 단계에서는 코드 수정 없음** — 본 보고서는 차이 식별 + 진단 계획까지가 범위.

---

## 5. ⚠ 가장 중요한 empirical 발견 — A의 실패는 "구조"가 아니라 "run/config" 문제

`logs/rsl_rl/go2_parkour/` 최근 run 전수 스캔 결과(2026-05-19):

| Run | iter | reward(끝) | ep_len(끝) | terrain_level(끝) | 판정 |
|-----|------|-----------|-----------|-------------------|------|
| 17-20-51 isaac_4.5 | 580 | **10.7** | 669 | 4.4 | ✅ 정상 학습 |
| 17-20-44 isaac-5.1 | 674 | **10.6** | 647 | 4.8 | ✅ 정상 학습 |
| 17-00-42 isaac_4.5 | 213 | 1.17 | 105 | **0.005** | ❌ 실패 |
| 17-00-32 isaac-5.1 | 260 | 0.91 | 86 | **0.012** | ❌ 실패 |
| 16-40-26 isaac-5.1 | 287 | 1.09 | 99 | **0.015** | ❌ 실패 |
| 16-37-32 isaac_4.5 | 291 | **−0.47** | 92 | **0.019** | ❌ 실패 |
| 05-18 / 05-14 (구버전) | 7k~50k | 8~18 | 770~920 | 3~5 | ✅ 정상 학습 |

**핵심**: A는 학습이 **되기도 하고 안 되기도 한다.** 같은 `direct/parkour` 환경인데 오늘 16:37~17:00 run들은 실패(terrain_level 0 고정 = curriculum 0단계 탈출 실패)했고, 17:20 run들은 정상 학습한다. → **A의 실패는 A vs B의 고정된 구조 차이로 설명되지 않는다.** 만약 H1/H5 같은 구조 차이가 원인이면 *모든* A run이 실패해야 하지만, 그렇지 않다.

**함의**:
- 본 A-vs-B 비교 보고서는 "왜 A가 학습 안 되나"의 답이 **아니다**. A-vs-B 차이는 모두 "상수"라서 가변적인 성공/실패를 설명할 수 없다.
- 실제 원인은 16:37~17:00(실패) ↔ 17:20(성공) 사이에 바뀐 **무언가**다. 각 run은 `<run>/git/IsaacLab.diff`(실행 시점 코드 스냅샷)와 `<run>/params/{env,agent}.yaml`(cfg 덤프)을 저장한다.
- **올바른 다음 단계**: `2026-05-19_17-00-42_isaac_4.5`(실패)와 `2026-05-19_17-20-51_isaac_4.5`(성공)의 `git/IsaacLab.diff` + `params/env.yaml`을 diff → 학습을 고친 변경을 정확히 국소화. 이것이 A-vs-B 구조 비교보다 직접적이고 결정적이다.

> 본 보고서(§1~§4)의 A-vs-B 차이 카탈로그는 참고 자료로 유효하나, **A 실패의 root cause는 fail-run vs pass-run 비교에서 찾아야 한다.**

---

*세부 근거·파일:line 인용은 §0의 5개 하위 보고서 참조.*
