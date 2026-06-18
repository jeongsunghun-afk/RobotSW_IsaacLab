# 03 · 학습곡선·Loss 분석 — "로봇이 왜 망가지는가" 진단

**대상**: go2_parkour_symmetry 3개 런 (tensorboard tfevents)
**작성**: worker-3 (debug-worker)
**방법**: `tensorboard.backend.event_processing.event_accumulator`로 스칼라 전량 파싱 (conda `isaac-parkour`).
파싱 스크립트: `_workspace/log_analysis_3runs/_parse_runs.py`, `_traj.py`, `_zoom.py`. 원시 요약: `_parsed_summary.json`.

---

## 결론 (TL;DR)

> **런3(`positive_work_0.01`)은 ~iter 1100에서 시작해 ~iter 1580에 완전 붕괴했다. 원인은 `positive_work` 효율 penalty의 weight가 런1 대비 ~33–38배 과대(iter0 episodic 기여 -0.0115 vs 런1 -0.0003)로, 커리큘럼이 지형을 올릴수록 penalty(에피소드 -1.5 ~ -3.0)가 tracking 보상(+1.1 상한)을 압도 → total_reward가 `clip(min=0)` floor(0)에 박힘 → advantage 신호 소멸 → 남은 gradient인 entropy bonus가 policy std를 무한 증폭(0.55→25.8) → 정책 발산.** 두 메커니즘이 **순차 결합**: ① tracking 압도 → ② clip floor → ③ entropy/std runaway. 이는 단순 "안 움직임"이 아니라 **완전한 정책 붕괴(irreversible)**다.
>
> 대조군: 런2(`positive_work`=0, 완전 비활성)는 전 구간 안정. 런1(같은 reward, weight ~33배 작음)은 중반 일시 dip 후 **회복**. → positive_work weight가 단일 원인임을 분리 확인. **Evidence Tier 1 (tfevents 직접 수치 + 통제변인 분리).**

---

## 1. 스칼라 태그 목록

3개 런 모두 **동일한 96개 스칼라 태그** (태그 set 차이 없음). 주요 그룹:

- `Train/`: mean_reward, mean_episode_length (+/time variants)
- `Episode_Reward/`: action_rate_l2, air_time_cap, ang_vel_xy_l2, collision, contact_duty_deficit, delta_torques, dof_acc_l2, dof_error_l2, feet_dragging, feet_edge, feet_gait_pairing, feet_stumble, hip_pos, lin_vel_z_l2, orientation_l2, **positive_work**, torques_l2, **tracking_goal_vel**, tracking_yaw
- `Episode_Termination/`: base_contact, cause_base_contact, cause_goal_reached, cause_low_height, cause_tilt, time_out
- `Loss/`: entropy, estimator, hist_latent_loss, learning_rate, priv_reg_loss, surrogate, symmetry, value
- `Policy/mean_noise_std`, `policy_std/<joint>` (12), `action_stats/<joint>/{mean,abs_max,sample_std}` (36)
- `curriculum/mean_terrain_level` (+ flat/gap/hurdle/stair/step)

> 비활성(전 구간 0) 태그: `air_time_cap`, `contact_duty_deficit`, `feet_gait_pairing`, `Episode_Length/mean_at_reset` — 3런 공통. (런2명 "no_duty_time_cap"과 정합: duty/time_cap 항 off.)

런별 step 범위: 런1 = 0..18763 (18764 pt) · 런2 = 0..15334 (15335 pt) · 런3 = 0..14012 (14013 pt).

---

## 2. 3런 핵심 메트릭 비교표 (late = 90–100% 구간 평균, 별도 표기 없으면)

| 메트릭 | 런1 positive_work (~3e-4) | 런2 no_duty_time_cap (pw=0) | 런3 positive_work_0.01 | 판정 |
|---|---|---|---|---|
| **Train/mean_reward** (late) | **18.99** (last 19.40) | **18.31** (last 19.12) | **0.0000** (last 0.0000, max 13.17) | 런3 붕괴 |
| **Train/mean_episode_length** (late) | 768 | 742 | **998** (last 999=상한, timeout 고착) | 런3 비정상 |
| **tracking_goal_vel** (late) | 1.100 | 1.060 | **0.0026** (≈0) | 런3 정지 |
| tracking_yaw (late) | 0.359 | 0.346 | 0.406 | — |
| **positive_work** (late) | **-0.0249** | 0.0 (비활성) | **-0.4839** (min -3.17) | 런3 ~19배 |
| **collision** (late) | -0.064 | -0.119 | **-48.09** (min -51.1) | 런3 지속충돌 |
| hip_pos (late) | -0.016 | -0.016 | **-1.941** | 런3 자세붕괴 |
| torques_l2 (late) | -0.0040 | -0.0044 | -0.0257 | 런3 ↑ |
| **Loss/entropy** (late) | 10.17 | 10.81 | **54.19** (last 55.5, 단조↑) | 런3 폭발 |
| **Policy/mean_noise_std** (late) | 0.570 | 0.601 | **22.5** (last 25.15, 단조↑) | 런3 발산 |
| **Loss/value** (late) | 6.4e-3 | 8.9e-3 | **5.1e-9** (→0) | 런3 critic 붕괴 |
| Loss/learning_rate (late) | 2.0e-4 | 1.5e-4 | **1.0e-5** (min floor 고착) | 런3 LR floor |
| **curriculum/mean_terrain_level** (late) | 5.93 | 5.89 | **0.0** (max 4.07) | 런3 진전 0 |
| termination 주원인 (late) | goal_reached 4.97 | goal_reached 5.18 | **time_out 4.17, goal_reached 0** | 런3 목표 미달 |

> ⚠️ `Episode_Reward/*`는 episode-normalized(step_dt 반영) 누적치 → 표 안에서 직접 비교 가능, weight 재곱 금지.
> 런1의 mid(45–55%) 구간 값이 낮게 보이는 건 중반 일시 dip(아래 §4) 평균화 아티팩트이며 late에서 완전 회복됨.

---

## 3. 붕괴 메커니즘 — 슬로모션 death spiral (런3, iter 1100–1580)

`_zoom.py` 20-iter 간격 추출. **모든 변수가 단조로 한 방향**으로 움직이는 인과 사슬:

| iter | mean_reward | trkvel | positive_work | std | entropy | terrain |
|---|---|---|---|---|---|---|
| 1100 | 11.20 | 0.650 | **-0.795** | 0.549 | 9.65 | 2.95 |
| 1200 | 7.75 | 0.678 | -0.912 | 0.705 | 12.65 | 3.38 |
| 1280 | 4.00 | 0.477 | -1.096 | 0.887 | 15.42 | 2.65 |
| 1360 | 1.33 | 0.451 | -1.468 | 1.129 | 18.31 | 3.07 |
| 1440 | 0.197 | 0.264 | -1.672 | 1.400 | 20.89 | 1.87 |
| 1540 | 0.001 | 0.065 | -1.856 | 1.912 | 24.55 | 0.72 |
| 1580 | **0.000** | 0.078 | -2.981 | 2.198 | 26.29 | 0.37 |
| 1700 | 0.000 | -0.007 | -0.174 | 3.359 | 31.44 | 0.00 |
| 14167 | 0.000 | 0.004 | -0.405 | **25.80** | **55.85** | 0.00 |

읽는 법 (인과 순서):
1. **트리거** — `positive_work` weight 과대(0.01). iter0 기여 **-0.0115** (런1 -0.0003 대비 **38배**). 커리큘럼이 지형을 올리며 요구 일량↑ → penalty가 **-0.79 → -1.5 → -3.0**으로 단조 증가. tracking 상한(+1.1)을 초과.
2. **net 음수화** — "전진" 행동의 한계 보상이 음수가 됨. trkvel 0.65→0.06로 정책이 **움직임을 줄이는 방향**으로 학습(penalty 회피). 동시에 tracking 보상도 상실.
3. **clip floor 도달** — total_reward가 net 음수 → `clip(min=0)` → mean_reward가 iter~1580에 **0.000 고착**. (parkour total_reward=clip(min=0)은 의도된 설계 — bug 아님.)
4. **gradient starvation → entropy runaway** — 모든 return≈0이라 advantage 신호 소멸, surrogate gradient 무력화. 남은 유일한 gradient는 **entropy bonus** → std/entropy를 **무한 증폭**: std 0.55→25.8, entropy 9.65→55.85 (단조, 회복 없음).
5. **critic 붕괴** — 모든 target이 0 → value function이 V≈0 학습 → `Loss/value` 6e-3 → **5e-10**.
6. **물리 붕괴 2단계** — (A) iter 1700–~8000: 정책=순수 발산 노이즈 → 짧은 에피소드(13–471)로 즉시 넘어짐. (B) iter ~8264+: 비종료 붕괴 자세 발견 → 에피소드가 **999(상한) timeout 고착**하지만 다리/몸통 지속 접촉으로 **collision -48** 누적. goal_reached=0. 영구.

---

## 4. 통제변인 분리 — 같은 메커니즘인데 왜 런1·런2는 멀쩡한가

**런2 (positive_work 완전 비활성, pw=0.0000 전 구간)**: 전 구간 매끈. reward 13→19.3 단조 상승, std 0.58–0.76 안정, terrain ~6 유지, dip 없음. → positive_work 항 자체가 사라지면 붕괴 없음. **원인이 positive_work임을 직접 증명하는 control.**

**런1 (positive_work ON, weight ~33배 작음)**: positive_work 기여가 항상 작음(-0.025 ~ -0.05)으로 tracking(+1.1)을 **압도하지 못함**. 단, iter ~7000–9500에 **일시 dip** 존재(reward 18→1.3→0.03, terrain→0, value_loss 97 스파이크) — 그러나 std가 0.83에서 **멈추고**(폭발 안 함), iter 10248에 reward 16.3으로 **완전 회복**.
→ 결정적 차이: **std/entropy runaway의 발생 여부**. 런1은 floor에 잠깐 닿아도 tracking gradient가 살아있어 std가 bounded → 탈출. 런3은 penalty가 floor를 **깊고 영구히** 만들어 entropy bonus만 남음 → std 폭발 → **비가역**.

| 항목 | 런1 (작은 pw) | 런3 (큰 pw) |
|---|---|---|
| pw 기여 vs tracking | -0.025 ≪ +1.10 (압도 못함) | -1.5~-3.0 > +1.1 (압도) |
| floor 도달 | 일시(iter 7–9.5k) | 영구(iter 1.58k~끝) |
| std 거동 | 0.83서 정점→복귀 | 0.55→25.8 발산 |
| 결과 | **회복** | **영구 붕괴** |

---

## 5. 가설 검증 결과

| 가설 (작업지시) | 판정 | 근거 |
|---|---|---|
| 런3가 런1 대비 망가졌다 | **CONFIRMED** | mean_reward late 0.0 vs 18.99; terrain 0 vs 5.93 |
| positive_work weight 과대(0.01≈-1e-2, 33배) | **CONFIRMED** | iter0 episodic pw -0.0115 vs 런1 -0.0003 (38배) |
| total_reward가 clip(min=0) floor 박힘 | **CONFIRMED** | iter~1580부터 mean_reward 0.000 고착, min=0 |
| gradient 소멸 → 붕괴 | **CONFIRMED** | value_loss→5e-10, surrogate 무력, entropy/std 폭발 |
| 효율 penalty가 tracking 압도("안 움직임") | **CONFIRMED** | trkvel 0.65→0.003, pw -3.0 > tracking +1.1 |
| entropy collapse(탐색 죽음)? | **REJECTED(반대현상)** | entropy/std는 **폭발**(9.65→55.85, std→25.8). collapse 아닌 **runaway** |

핵심 정정: 흔한 실패는 "entropy collapse"지만 런3는 **반대 — entropy explosion**. clip floor가 advantage를 없애 entropy bonus만 남긴 결과다.

---

## 6. "망가짐" 정량 증거 (런/iter/메트릭 특정)

- **붕괴 시점**: 런3, iter **1100(개시)–1580(floor 도달)**. (`_zoom.py` 단조 추이)
- **붕괴 지표**: `Train/mean_reward` 11.2→0.000 / `tracking_goal_vel` 0.65→0.003 / `Policy/mean_noise_std` 0.55→**25.8** / `Loss/entropy` 9.65→**55.85** / `Loss/value` 6e-3→**5e-10** / `curriculum/mean_terrain_level` 3.0→**0.0**.
- **트리거 수치**: `Episode_Reward/positive_work` iter0 **-0.0115**(런1 -0.0003의 38배), 커리큘럼 진행 중 **-3.17(min)**까지 심화, tracking 상한 +1.1 초과.
- **2차 증상**: `collision` late **-48.09**(붕괴 자세 지속접촉), `hip_pos` -1.94, eplen 999 timeout 고착, goal_reached 0.

---

## 7. 진단 분류 및 수정 권고 (담당 worker)

**메커니즘 진단**: `clip floor → entropy/std runaway` (단일 root, tracking 압도가 floor를 유발). entropy collapse / value 폭증 아님.

1. **[critical] reward-worker** — `positive_work` reward scale를 런3의 0.01에서 런1 수준(~3e-4)으로 **되돌리거나 그 이하**로. 효율 penalty의 에피소드 기여가 `tracking_goal_vel`(+1.1 상한)을 **절대 초과하지 않도록** 상한 설정(예: episodic |pw| ≤ 0.3·tracking). 단일 변경.
   - 검증: 재학습 시 `positive_work` 기여가 커리큘럼 max terrain에서도 tracking 대비 작게 유지되는지, `Policy/mean_noise_std`가 bounded(<1.0) 유지되는지 모니터.
2. **[참고/별도 트랙] loss-worker(선택)** — 구조적 강건화로 reward가 floor에 닿아도 entropy runaway를 막도록 `entropy_coef` 적응 또는 std 상한(예: max noise_std clamp) 고려. **단, 1번이 root fix이며 이 항은 동시 변경 금지**(원인 분리 위해 1번 먼저 단독 검증).

> ⚠️ 동시 변경 금지: positive_work scale **하나만** 먼저 되돌려 재현/해소 확인. clip(min=0)은 의도된 설계이므로 제거 권고 아님.

**Evidence Tier 1** — 3런 tfevents 직접 수치, 단조 인과 사슬 가시, 통제변인(런2 pw=0, 런1 작은 pw)으로 변인 분리. 추측 없음.

---

## 부록 · 재현 단계

```bash
conda activate isaac-parkour   # /home/user/miniconda3
cd /home/lgb/IsaacLab
python _workspace/log_analysis_3runs/_parse_runs.py   # 태그목록 + 요약 JSON
python _workspace/log_analysis_3runs/_traj.py         # 25-pt 런별 궤적
python _workspace/log_analysis_3runs/_zoom.py         # 런3 iter1100-2200 붕괴 확대
```
