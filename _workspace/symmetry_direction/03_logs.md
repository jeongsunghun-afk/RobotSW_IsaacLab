# 03 — go2_parkour_symmetry 8런 tfevents 로그 분석

**작성**: worker-3 | **소스**: `logs/rsl_rl/go2_parkour_symmetry/*/events.out.tfevents`
**파서**: `tensorboard.backend.event_processing.event_accumulator` (env `isaac-parkour`, 전체 스칼라 로드)
**수치 = T1(검증된 tfevents 값)**. 해석/메커니즘은 T2(수치로부터 강한 추론)·T3(가설)로 표기.

> reward는 episode-normalized (step_dt 반영). parkour total_reward = `clip(min=0)` → 정책이 붕괴하면 `Train/mean_reward`가 정확히 **0.0**으로 바닥침. 이 "reward=0 + curriculum/terrain_level=0" 동시 발생이 본 분석의 **붕괴 시그니처**.

---

## 1. 8런 핵심 메트릭 비교표 (최종값, T1)

| # | 런 (라벨) | iters | final reward | max reward | terrain_lv | std | entropy | goal_reached | tilt | base_contact | 판정 |
|---|-----------|-------|-------------|-----------|-----------|-----|---------|-------------|------|-------------|------|
| R1 | baseline | 30075 | **19.61** | 20.75 | 6.04 | 0.561 | 9.99 | 4.96 | 0.08 | 0.00 | ✅ 안정 |
| R2 | gap_step_more_difficult | 21743 | **19.54** | 21.12 | 5.97 | 0.568 | 10.15 | 4.92 | 0.04 | 0.00 | ✅ 안정 (중간 1회 transient dip) |
| R3 | torque_penalty 1e-4 | 14014 | **17.84** | 20.16 | 6.09 | 0.579 | 10.35 | 5.92 | 0.13 | 0.00 | ⚠️ 회복 (step~10.5k 1회 full crash→recover) |
| R4 | no_duty_time_cap | 5077 | **0.00** | 17.26 | 0.00 | 0.845 | 14.94 | 0.00 | 108.5 | 0.00 | ❌ 붕괴 |
| **R5** | **positive_work −3e-4** | **50000** | **21.36** | **22.47** | **5.99** | **0.518** | **9.02** | **6.08** | **0.00** | 0.00 | ✅✅ **최고·최안정** |
| R6 | no_duty_time_cap (pw 0) | 18919 | **0.00** | 20.55 | 0.00 | **1422.9** | 104.1 | 0.00 | 104.4 | 0.29 | ❌ 폭주 붕괴 |
| R7 | positive_work −1e-2 | 14375 | **0.00** | 13.17 | 0.00 | 26.68 | 56.25 | 0.00 | 0.04 | 0.00 | ❌ 초반 붕괴 |
| **R8** | **positive_work −1e-3** | **50000** | **10.24** | 20.64 | 6.07 | **0.800** | **14.26** | 3.88 | 0.96 | 0.17 | ⚠️ **완주했으나 중반 full crash + 부분회복** |

> 정상 수렴 기준선(R1/R2/R5): reward ≈ 19–21, terrain_lv ≈ 6, std ≈ 0.52–0.57, entropy ≈ 9–10, goal_reached ≈ 5–6, tilt < 0.1.
> termination 값은 `Episode_Termination/cause_*`의 최종 스칼라(에피소드당 발생 카운트 비율). 붕괴 런은 tilt가 100+로 폭증 = 거의 모든 env가 매 step 넘어짐.

---

## 2. 두 완주 런 정밀 비교 — R5(−3e-4) vs R8(−1e-3)  ⭐핵심

두 런 모두 50000 iter 도달. 그러나 **R5는 깨끗이 완주, R8은 완주하지 못함**(중반 붕괴 후 부분 회복).

### 2.1 tail 궤적 (step별, T1)

**R5 (−3e-4) — 단조 안정/상승, tail 무이벤트:**
| step | reward | std | entropy | terrain | collision | pos_work |
|------|--------|-----|---------|---------|-----------|----------|
| 35000 | 19.81 | 0.542 | 9.56 | 5.85 | −0.061 | −0.024 |
| 40000 | 19.70 | 0.540 | 9.51 | 6.59 | −0.030 | −0.025 |
| 42500 | 19.38 | 0.547 | 9.69 | 5.66 | −0.034 | −0.024 |
| 45000 | 20.56 | 0.538 | 9.48 | 6.13 | −0.036 | −0.023 |
| 47500 | 20.61 | 0.524 | 9.15 | 5.92 | −0.034 | −0.024 |
| 49999 | **21.36** | **0.518** | **9.02** | 5.99 | −0.039 | −0.023 |
→ std는 끝까지 **하강**(0.542→0.518), entropy 하강, reward 최고치 갱신. **건강한 수렴.**

**R8 (−1e-3) — step ~42500 full crash → 부분 회복:**
| step | reward | std | entropy | terrain | collision | pos_work |
|------|--------|-----|---------|---------|-----------|----------|
| 35000 | 19.17 | 0.540 | 9.55 | 5.47 | −0.083 | −0.071 |
| 40000 | 19.01 | 0.531 | 9.34 | 6.29 | −0.075 | −0.075 |
| **42500** | **0.000** | 0.759 | 13.60 | **0.000** | −0.361 | −0.013 |
| **45000** | **0.000** | **1.053** | **17.52** | **0.000** | **−2.568** | −0.047 |
| 47500 | 6.84 | 0.925 | 16.00 | 5.33 | −2.295 | −0.178 |
| 49999 | 10.24 | 0.800 | 14.26 | 6.07 | −0.999 | −0.145 |
→ step 40000까지 R5와 사실상 동급(reward 19, std 0.53). 그러다 **42500에서 reward·terrain이 동시에 0으로 붕괴**(클립 바닥), std 0.53→1.05 폭주, entropy 9.3→17.5, collision −0.08→−2.57. −1e-3 penalty가 충분히 약해 step 47500부터 부분 회복했으나, **최종 reward 10.24 = R5의 절반, std 0.80·entropy 14.3은 건강 basin(0.52/9.0) 미복귀.**

### 2.2 판정 (T1+T2)

- **R5(−3e-4)가 명백한 승자.** 최종 reward 21.36(8런 최고), max 22.47, std 0.518(8런 최저=가장 수렴), entropy 9.02(최저), terrain 5.99, goal_reached 6.08, **fall 계열 termination 전부 0**(tilt 0.0/base_contact 0.0/low_height 0.0). collision −0.039로 baseline(−0.097)보다도 우수. (T1)
- **R8(−1e-3)은 "붕괴 없이 완주"가 아니다.** iter 카운트만 50000일 뿐, step ~42500에서 R6/R7과 **동일한 클립-바닥 붕괴 모드**(reward→0 + curriculum 리셋 + std/entropy 폭주)를 겪었고 penalty가 약해 우연히 부분 회복했을 뿐. 최종 상태도 reward 절반·std/entropy 고착·goal_reached 3.88(vs 6.08)·tilt 0.96·base_contact 0.17로 **불안정 잔존.** (T1; 메커니즘 T2)
- **calibration 가설 정량 확정**: −3e-4 안전(무이벤트) / −1e-3 경계(crash 1회+부분회복) / −1e-2 붕괴(초반부터 회복불가). positive_work weight가 클수록 클립-바닥 붕괴 위험 단조 증가. (T2)

---

## 3. 망가진 런 상세

### R7 (positive_work −1e-2) — 대조군, 초반 즉시 붕괴 (T1)
- `Train/mean_reward`: 전 구간 0.0 (max 13.17은 step 923에서만 잠깐). terrain_level 3593 step부터 끝까지 0.0 = 커리큘럼 한 번도 못 올라감.
- std 1.00→5.03(3593)→8.71→15.03→**26.68**, entropy 17→36→43→49→**56.25** = 단조 폭주.
- reward 항: `collision −48.93`, `hip_pos −1.91`, `dof_error −1.32`, `positive_work −0.38` = penalty가 reward를 압도 → 총합 음수 → clip(min=0) 바닥 → gradient 소실 → std runaway. **positive_work −1e-2는 즉사.** (T1)

### R6 (no_duty_time_cap, pw 0) — 최악의 폭주 (T1)
- step 14189까지는 정상(reward 18.87, terrain 5.84, std 0.587). step 18919에서 **완전 폭발**: std **1422.9**, entropy 104.1, `Loss/symmetry` **1258.6**, reward 0.0, terrain 0.0, ep_length 42, tilt 104.4.
- 라벨상 positive_work=0이지만 **이 붕괴는 pw 탓이 아니라 `no_duty_time_cap`(duty/time cap 항 제거) 탓**(R4와 동일 패턴, §4 참조). pw=0의 단독 효과는 본 런으로 분리 불가(confound). (T2)

### R4 (no_duty_time_cap, 5k) — 조기 붕괴 (T1)
- step 2538까지 정상(reward 14.73, terrain 6.49). step 3807에서 reward·terrain 0.0, value_loss 1.93 스파이크, 이후 회복 못 함. 최종 ep_length 34, tilt 108.5(거의 모든 env 매 step 전도). std 0.70→0.85 상승. (T1)

---

## 4. 초기·성공 런 비교 (T1)

- **R1 baseline (30k)**: 가장 깨끗한 레퍼런스. reward 19.61, terrain 6.04(flat 5.5/hurdle 5.9/step 6.5/stair 6.4 균형), goal_reached 4.96. 지배 reward 항: `tracking_goal_vel +1.08`, `tracking_yaw +0.35` (양수 신호 원천), 최대 penalty `action_rate −0.24`.
- **R2 gap_step (21.7k)**: baseline과 동급(reward 19.54, terrain 5.97). 단 step 16307에서 reward 18.5→14.78 **transient dip** 후 회복 — 난이도 상향(gap/step)에 따른 일시 적응, 붕괴 아님. step terrain만 4.96으로 약간 낮음(난이도 상향 흔적). (T1+T2)
- **R3 torque_penalty 1e-4 (14k)**: 최종 reward 17.84(약간 낮음), terrain 6.09. **step ~10510에서 1회 full crash**(reward 0.0013, terrain 0.0, std 0.93, entropy 16.0) 후 step 14014에 17.84로 회복. torque penalty 추가가 일시 불안정 유발했으나 치명적이진 않음. `torques_l2` penalty가 −0.004→**−0.039**로 10× 강화된 것이 유일한 항별 차이. (T1)

### 두 "no_duty_time_cap" 런의 공통 붕괴 = 강한 패턴 (T2)
R4·R6 **둘 다** duty/time cap 항 제거 후 붕괴(2/2). → **duty/time cap 관련 항은 학습 안정성에 기여하는 regularizer일 가능성**. 단, 본 worker는 cfg diff 미확인 → 정확한 제거 항목·인과는 task #1/#2(코드·cfg diff)와 cross-check 필요. (T2, 가설)

---

## 5. 종합 결론

1. **가장 안정·고성능 설정 = R5 (positive_work weight −3e-4).** 8런 중 최고 final reward(21.36)·최저 std(0.518)·fall 0·max terrain 균형. 49999 step 전 구간 무이벤트 단조 수렴. (T1)
2. **positive_work 안전 임계는 −3e-4와 −1e-3 사이.** −3e-4=무사고, −1e-3=완주표기되나 중반 클립-바닥 붕괴+부분회복(최종 reward 절반), −1e-2=초반 즉사. weight↑ → 붕괴위험 단조↑. (T1+T2)
3. **R8(−1e-3)을 "성공한 완주"로 분류하면 안 됨.** iter만 채웠을 뿐 R6/R7과 동일 붕괴 모드를 거쳤고 최종 상태 불안정. 권장 설정에서 제외. (T2)
4. **clip(min=0) total_reward가 붕괴 증폭기.** penalty가 reward 압도 → 총합 0 클립 → gradient 소실 → std/entropy 폭주. 모든 붕괴 런(R4/R6/R7/R8-중반)이 "reward=0 + terrain=0" 동시 시그니처 공유. (T2; clip 설계는 의도된 것 — 메모리 참조)
5. **`no_duty_time_cap` 제거는 2/2 붕괴** → 회귀 위험 항목. 추가 검증 권고. (T2)

### 망가진 런 식별 요약
- **완전 붕괴(회복 불가)**: R4, R6, R7
- **중반 붕괴 + 부분 회복(불완전)**: R8
- **일시 dip/crash 후 회복(최종 정상)**: R2(dip), R3(crash)
- **깨끗한 성공**: R1, R5  ← **R5 권장**

### Evidence tier 범례
- **T1 검증사실**: 본 보고서 모든 수치 표·궤적 = tfevents 직접 추출값.
- **T2 강한 추론**: 붕괴 메커니즘(clip-바닥→gradient소실→std폭주), calibration 단조성, no_duty 패턴 — 수치로부터 직접 도출.
- **T3 가설(미검증)**: duty/time cap 항의 정확한 인과·제거 항목(코드 diff 미확인, task #1/#2 cross-check 필요), pw=0 단독 효과(R6 confound).
