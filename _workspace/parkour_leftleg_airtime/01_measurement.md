# 01 Measurement — air_time_cap run 실측 (lead가 직접 추출, 정본)

> ⚠️ 이 파일은 lead의 **summary_iterator 스트리밍 실측본**이 정본이다. measure-worker는 312MB event_accumulator 로딩 블로킹 + conda env(py3.11/3.12) 버전 충돌로 TensorBoard 파싱에 실패했고, 코드 구조 추론만으로 "weight=-0.25/threshold=0.6s escalate, 1400 iter early stop" 등 **부정확한 결론**을 냈다(그 버전은 폐기). 아래는 실제 event 파일 4,567,038 events 스트리밍 추출 결과.

**대상 run**: `logs/rsl_rl/go2_parkour/2026-06-02_16-12-18_air_time_cap_reward`
**상태**: 학습 진행 중 (event 312MB, **last_step=49999** 시점 추출). 커밋 90f82298ef1(2026-06-02 16:11, air_time_cap 추가) 직후 16:12 시작 → 사용자의 "여전히 왼쪽 다리 들림" 관찰은 이 run.
**추출 방법**: event_accumulator 통째 로드 불가(312MB OOM) → `summary_iterator` 스트리밍 (`_parse_events.py`, `_events_airtimecap.json`).

---

## A. air_time_cap penalty는 실측상 무력 (advisor의 (A) 판별 확정)

reward term별 episode 평균 기여 (early 0–10% / mid 45–55% / late 90–100%):

| term | early | mid | late |
|------|-------|-----|------|
| **air_time_cap** | -0.0025 | -0.0029 | **-0.0033** |
| tracking_goal_vel | +0.985 | +1.069 | **+1.082** |
| feet_edge | -0.076 | -0.036 | -0.034 |
| feet_dragging (hind-only) | -0.030 | -0.020 | -0.020 |
| feet_stumble | -0.006 | -0.001 | -0.002 |
| feet_gait_pairing | 0.000 | 0.000 | **0.000 (비활성)** |
| collision | -0.340 | -0.071 | -0.076 |
| hip_pos | -0.026 | -0.022 | -0.021 |

- `Train/mean_reward`: 14.4 → 19.6, `Train/mean_episode_length`: 714 → **750 step** (≈15s@0.02dt). (measure-worker의 "1400 iter early stop"은 오류 — 50k step까지 진행.)
- **판별**: air_time_cap 기여 -0.0033 = mean_reward(19.6)의 **0.017%**. episode가 750 step으로 충분히 긴데도 penalty가 거의 안 걸림 → air_time이 max_air=1.0s를 거의 안 넘긴다(주기적 micro-tap 리셋으로 escape, diagnose-worker 가설 2와 정합). "짧은 에피소드로 air_time 못 쌓음"(메모리 27–28 step)은 이 run엔 **무관**.
- 회피 유인 비교: feet_edge(-0.034)+feet_dragging(-0.020) ≈ -0.054, air_time_cap(-0.0033)의 **16배**. 다리를 들면 이 contact-gated penalty들을 회피 + tracking +1.08 획득 → 3-leg가 reward-positive. air_time_cap은 이를 못 막음. **weight가 약해서가 아니라 신호가 escape 가능 + 회피 유인에 압도됨** (→ weight/threshold 단독 튜닝은 해법 아님, §3).

## B. "왼쪽 다리" = RL(rear-left) 확정 — RL_calf action 발산 (단, downstream 증상)

per-leg action_stats `/mean` (late 90–100%):

| Leg | hip | thigh | calf |
|-----|-----|-------|------|
| FL | -1.276 | -0.417 | +1.485 |
| FR | -0.245 | +0.208 | +0.121 |
| **RL** | +0.341 | +0.622 | **+32517.449** |
| RR | +1.075 | -0.200 | +1.534 |

policy_std (late): RL_calf = **10609.672** (다른 11관절 0.4–0.7).

- `RL_calf_joint`만 선택적으로 action mean +32517, std 10609 — 나머지 11개 관절 대비 **4~5 자릿수 발산**. RL = rear-left = **왼쪽 뒷다리**. **들린 다리 = RL 확정.**
- **인과 = downstream 증상 (advisor 확정)**: `parkour_env.py:557`에서 `clip_actions=10.0`으로 applied action은 ±10 clip. 만약 32517이 실제 적용됐다면 sim이 발산해 terrain_level 6 불가. 즉 32517·std10609는 **clip 이후 reward에 무관해진 죽은 action 차원의 unclipped policy-output artifact**(항상 clip→gradient 0→mean random-walk, entropy가 std 부풀림). 정책이 RL을 안 쓰기로 한 결과지, calf를 못 써서 다리를 든 게 아님. ⇒ **actuator/calf-limit/토크 처방 금지(§5/§6). fix target 아니라 모니터링 지표.**

## C. 학습은 표면상 진행 — 3-leg로 커리큘럼 통과

curriculum/mean_terrain_level (late): all ≈ 5.8–6.1 (flat 6.06 / gap 5.72 / hurdle 6.03 / stair 6.06 / step 5.85), 균등.
Termination(late): goal_reached 5.30, time_out 5.50, tilt 0.049, base_contact 0.006, low_height 0.003.

→ 정책은 RL을 사실상 버린 3-leg gait로 모든 지형 커리큘럼을 통과 중. 사용자의 "3-leg gait" gait-level 관찰과 정합.

---

## 측정 종합 (root cause)
1. **3-leg가 reward-positive 평형**: contact-gated penalty(feet_edge/dragging/stumble) 회피 + tracking 유지. air_time_cap(-0.0033)은 이를 못 막음.
2. **air_time_cap 무력 메커니즘**: ① raw current_air_time이 micro-tap에 0 리셋되어 escape ② 회피 유인(-0.054)이 penalty의 16배. → weight/max_air 튜닝(§3)으로 풀 문제 아님, **신호 설계 수정** 필요.
3. **RL_calf 발산 = 증상**(clip+entropy). fix 후 정상 회복 여부를 성공 지표로.
