# 04 · 최종 학습 모델 걸음걸이(gait) 측정·비교

**작성**: worker-4 | **일자**: 2026-06-15 | **task #4**
**도구**: `scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py` (pronk_v1), 256 envs × 3000 step, `--task Go2-Parkour-Symmetry --headless`, seed 42
**conda env**: `isaac-5.1` (※ task 지시의 `isaac-parkour`는 오류 — 하단 [실행 메모] 참조)

---

## 0. TL;DR — "효율 reward가 걸음을 자연스럽게(pronk↓ trot↑) 만들었나?"

| 질문 | 정량 답 |
|------|---------|
| 전신 pronk(4발 동시 비행) 줄었나? | **YES.** penalty ↑일수록 monotonic ↓ (stair 0.277→0.237→0.132) |
| trot/교대보행으로 갔나? | **NO.** split-jump(앞뒤 분리 다단 hop) **폭증** (stair retk/flight 0.060→0.078→**0.290**) |
| 에너지 효율(CoT) 개선됐나? | **NO, 오히려 악화.** pw -1e-3은 측정 CoT를 **2.2× 증가**시킴 (stair abs CoT 0.576→**1.281**) — penalty 대상량을 줄이지 못하고 **역효과** |
| actuator 부담? | pw -1e-3에서 **포화 폭증** (flat all-step satfrac 0.009→**0.96**) |
| 주행 성공/강건성 유지? | run8(-1e-3) 실패율 0.7%→**3.1%** (4×↑), 단 둘 다 timeout 주행·terrain level 동등 |

**결론**: 효율 penalty는 **전신 pronk는 억제**했으나 **교대보행을 유도하지 못했다.** -1e-3은 과도(overshoot)하여 정책을 *포화·고에너지·파편화 hop*이라는 더 부자연스러운 local optimum으로 밀어넣었고, 자기 목적(positive_work 최소화)마저 역행했다. **-3e-4가 전 지표에서 더 균형적**이며, "효율로 trot이 emergent하지 않는다"는 것이 측정 사실. (trot 유도엔 명시적 gait-shaping 필요 — 메모 `project_parkour_gait_reward_design`와 일치.)

---

## 1. 측정 대상

| 라벨 | run | checkpoint | positive_work | 기타 reward |
|------|-----|-----------|---------------|-------------|
| baseline (pw0) | 2026-06-09_17-53-09 | model_16700 | 0 | **air_time_cap -0.1 ON, contact_duty -0.5 ON** |
| run5 (pw -3e-4) | 2026-06-11_12-13-02_positive_work | model_49999 | **-0.0003** | air_time_cap -0.0, contact_duty -0.0 |
| run8 (pw -1e-3) | 2026-06-12_09-53-06_positive_work_0.001 | model_49999 | **-0.001** | air_time_cap -0.0, contact_duty -0.0 |

> **검증된 사실**: run5 ↔ run8 의 reward cfg(env.yaml) diff는 **positive_work 값 하나뿐** — `feet_gait_pairing 0.0 / air_time_cap -0.0 / contact_duty_deficit -0.0 / contact_duty_target 0.5` 전부 동일. 따라서 **run5↔run8은 단일변수(효율 penalty 크기) 비교**로 깨끗함.
> **주의(confound)**: baseline은 air_time_cap·contact_duty가 **ON**이고 iter도 16700(vs 49999). 즉 baseline↔pw 비교는 "효율항 추가"와 "anti-flight 2항 제거"가 섞인 **비단일변수** 비교 → baseline은 *secondary reference*로만 사용. (출처: `_workspace/debug_report_positive_work_pronk.md`)
> baseline npz 원본은 본 세션 실행 중 출력경로 덮어쓰기로 유실 — baseline 수치는 task 브리프/메모 `project_parkour_pronk_measurement` 기록값 사용. run5/run8은 본 세션 신규 측정(원본 md 보존: `pronk_run5_pw3e-4.md`, `pronk_run8_pw1e-3.md`).

---

## 2. 핵심 측정표 (terrain별, pw 0 → -3e-4 → -1e-3 추세)

### (1) Pronk — all-airborne frac (4발 동시 비행 비율) — **낮을수록 좋음**
| terrain | baseline pw0 | run5 pw-3e-4 | run8 pw-1e-3 | run5→run8 |
|---------|-------------:|-------------:|-------------:|----------:|
| flat    | 0.0216 | 0.0128 | **0.0077** | −40% |
| gap     |   n/a  | 0.0589 | 0.0494 | −16% |
| hurdle  |   n/a  | 0.0786 | 0.0672 | −15% |
| step    |   n/a  | 0.0990 | 0.0707 | −29% |
| stair   | 0.277  | 0.2373 | **0.1316** | −44% |

→ **전신 pronk은 효율 penalty가 강할수록 단조 감소.** 가장 심하던 stair에서 baseline 0.277 → run8 0.132로 절반 이하. (단 baseline은 anti-flight ON이었음에도 0.277, pw run은 anti-flight OFF인데 더 낮음 → 효율항이 anti-flight 전용항보다도 pronk 억제에 효과적으로 작동.)

### (4) Split-jump — retk/flight (앞뒤 비협응 다단 hop) — **낮을수록 좋음(자연스러움)**
| terrain | baseline pw0 | run5 pw-3e-4 | run8 pw-1e-3 | run5→run8 |
|---------|-------------:|-------------:|-------------:|----------:|
| flat    | 0.002 | 0.001 | 0.000 | — |
| gap     |  n/a  | 0.023 | 0.089 | ×3.9 |
| hurdle  |  n/a  | 0.059 | 0.213 | ×3.6 |
| step    |  n/a  | 0.055 | **0.295** | ×5.4 |
| stair   | 0.060 | 0.078 | **0.290** | ×3.7 |

→ **split-jump은 효율 penalty가 강할수록 폭증.** pronk(전신 비행)을 누른 대가로 정책은 "앞발 착지→뒷발 착지 전 재도약"식 **파편화 hop**으로 도피. **교대보행(trot)이 아니라 협응 붕괴.** 이게 핵심 역설: pronk↓이 곧 자연보행↑이 아니다.

### (3) CoT(에너지효율) — Σ(positive τ·q̇)·dt / (m·g·dist) — **낮을수록 효율적**
| terrain | run5 abs (×flat) | run8 abs (×flat) | baseline(×flat) |
|---------|-----------------:|-----------------:|----------------:|
| flat    | 0.384 (1.00) | 0.486 (1.00) | — |
| gap     | 0.374 (0.97) | 0.582 (1.20) | — |
| hurdle  | 0.390 (1.02) | 0.870 (1.79) | — |
| step    | 0.437 (1.14) | 1.084 (2.23) | — |
| stair   | 0.576 (1.50) | **1.281 (2.64)** | (1.49×) |

→ **효율 penalty를 3.3× 키웠는데 측정 positive 기계일량(=penalty가 줄이려는 바로 그 양)이 오히려 증가.** stair 에너지 run5 266k J → run8 596k J (거리 ~동일 3134 vs 3160 m). **명백한 backfire**: -1e-3은 효율을 개선하기는커녕 더 비싼 gait로 정책을 밀어넣음. (baseline stair 1.49× ≈ run5 1.50× — run5는 baseline 수준 효율 유지, run8만 붕괴.)

### (2a) Actuator 포화 — satfrac — **낮을수록 좋음**
| terrain | run5 all-step / landing | run8 all-step / landing |
|---------|------------------------:|------------------------:|
| flat    | 0.0085 / 0.000 | **0.960** / 0.344 |
| gap     | 0.045 / 0.001 | 0.229 / 0.186 |
| step    | 0.069 / 0.040 | 0.532 / **0.705** |
| stair   | 0.088 / 0.033 | 0.578 / **0.625** |
| hurdle  | 0.058 / 0.013 | 0.710 / 0.502 |

→ run8(-1e-3)은 **거의 상시 토크 포화** (평지 96% step). 효율을 노린 penalty가 역으로 정책을 **near-limit 난폭 구동**으로 몰았다.

### (2b) 착지 충격 — peak Fz [bodyweight], p50 / p95
| terrain | run5 p50 / p95 | run8 p50 / p95 | baseline |
|---------|---------------:|---------------:|---------:|
| stair   | 3.12 / 8.20 | 1.54 / 5.02 | (~3.1) |
| step    | 2.73 / 7.29 | 1.88 / 5.31 | — |
| gap     | 2.68 / 4.75 | 2.28 / 5.12 | — |

→ run8은 **회당 충격은 더 작지만**(작은 hop 다수로 분산), 대신 **착지 횟수·포화·총에너지가 폭증**. "부드러운 착지"가 아니라 "잘게 쪼갠 난타"에 가까움. (사전등록 임계 X_sat=0.20·Y_impact=4.0BW 기준: run8 stair/step landing satfrac은 0.62/0.71로 임계 초과 → 안전성도 run8에서 악화.)

### 주행/강건성 (terrain 진행·실패)
| 지표 | run5 pw-3e-4 | run8 pw-1e-3 |
|------|-------------:|-------------:|
| episodes total / failure | 958 / 7 (**0.7%**) | 965 / 30 (**3.1%**) |
| valid(비실패) frac | 0.993 | 0.972 |
| terrain_level 마지막5% | 3.80 | 3.84 |
| goal_reached(종료원인) | **0** | **0** |

→ 둘 다 timeout 지배 주행, 도달 terrain level 동등(~3.8). run8 실패율 4×↑(여전히 절대값은 낮음).
> **goal 성공률 caveat**: 본 스크립트의 goal 지표는 "goal_reached 종료원인" 카운트인데 **두 런 모두 0** (에피소드가 goal 도달 전 timeout/리셋에 지배되는 측정 설정). 따라서 **goal 성공률은 이 지표로 두 런을 구분 불가** — terrain_level과 실패율을 주행능력 proxy로 사용. (goal 성공률 정밀비교가 필요하면 별도 play/eval 필요 — handoff.)

---

## 3. pw 0 → -3e-4 → -1e-3 추세 종합

| 지표 (stair 기준) | pw0 baseline | pw -3e-4 | pw -1e-3 | 추세 해석 |
|------|------:|------:|------:|------|
| pronk(all-airborne) | 0.277 | 0.237 | 0.132 | ↓↓ (개선) |
| split-jump(retk/flight) | 0.060 | 0.078 | 0.290 | ↑↑ (악화) |
| CoT(×flat) | 1.49 | 1.50 | 2.64 | -3e-4 유지, -1e-3 급악화 |
| 포화(all-step) | (낮음) | 0.088 | 0.578 | ↑↑ (악화) |
| 실패율 | — | 0.7% | 3.1% | ↑ (악화) |

**해석**: 효율 penalty는 pronk 억제엔 monotonic하게 듣지만, **-3e-4 → -1e-3 구간에서 split-jump·CoT·포화·실패율이 모두 비선형적으로 붕괴**. 즉 penalty 크기에 **sweet spot이 존재하고 -3e-4가 그 근처**, -1e-3은 명백한 overshoot.

---

## 4. 정량 결론 (task #5 종합 입력)

1. **"효율로 걸음이 자연스러워졌나" = 부분적 YES / 핵심 NO.**
   - pronk(전신 비행) 억제: ✅ 효율 penalty가 직접적으로 작동(monotonic).
   - trot(교대보행) 유도: ❌ 전혀. 오히려 split-jump 협응붕괴로 도피. **효율항 단독으로는 trot이 emergent하지 않음.**
2. **-1e-3은 backfire.** 자기 목적(positive 기계일량 최소화)을 측정상 역행(CoT 2.2×↑), 포화·실패율 동반 악화. **권장 상한 ~-3e-4** (분석 의견, 구현 위임 대상).
3. **자연 trot이 목표라면 명시적 gait-shaping 필요** — pronk 억제(효율/anti-flight)만으로는 부족. 메모 `project_parkour_gait_reward_design`의 anti-pronk min-stance(양수보상) + feet_gait_pairing 재활성 스택이 split-jump 협응 문제를 직접 겨냥하는 후보. (reward-worker 영역, 본 task 범위 밖 권고.)

---

## 5. 산출물 / 실행 메모

**측정 산출물** (`_workspace/symmetry_direction/`):
- `pronk_run8_pw1e-3.{npz,md}` — run8 (pw -1e-3, model_49999) 신규 측정
- `pronk_run5_pw3e-4.{npz,md}` — run5 (pw -3e-4, model_49999) 신규 측정
- 측정 wrapper: `_run_gait_measure.sh`, 로그: `_gait_measure.log`

**[실행 메모 — env 정정]**
- task 지시는 conda `isaac-parkour`였으나 **틀림**. isaac-parkour는 site-packages에 **public rsl_rl**가 설치돼 repo custom rsl_rl(`OnPolicyRunnerParkour`)을 가려 `ImportError`. 정상 env는 **`isaac-5.1`** (repo `/home/lgb/IsaacLab/rsl_rl` editable + isaacsim 보유). 메모 `project_conda_env`(범용 IsaacLab=isaac-5.1)와 일치.
- 실행 명령(정상 동작 확인):
  ```bash
  conda activate isaac-5.1 && cd /home/lgb/IsaacLab
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py \
    --task Go2-Parkour-Symmetry --num_envs 256 --num_steps 3000 --headless \
    --checkpoint <model_49999.pt>
  # 출력 고정경로(_workspace/pronk_cost_result.npz, pronk_cost_raw.md)는 런마다 rename 필수
  ```

**[데이터 무결성 caveat]**
- 측정 자체는 양 런 모두 완료·정상(실패율 0.7%/3.1%, valid 0.99/0.97). 수치는 신뢰 가능.
- baseline(pw0) **원본 npz는 유실**(고정 출력경로 덮어쓰기). baseline 수치는 기록값(task 브리프 + 메모) 인용이며 **secondary reference**. 정밀 baseline 재측정이 필요하면 `model_16700.pt`로 위 명령 재실행(원본 config가 air_time_cap ON이므로 단일변수 비교엔 부적합 — 메모 참조).

**[가설 vs 검증 구분]**
- ✅ 검증사실: run5↔run8 단일변수(pw만 다름), pronk monotonic↓, split-jump↑↑, run8 CoT/포화/실패율 악화, goal_reached 양런 0.
- 🔶 가설(미검증): "trot 유도엔 명시적 gait-shaping 필요"는 본 측정이 직접 증명한 게 아니라 *효율 단독으론 trot 미발생*이라는 측정에서 도출한 권고. 구현·재학습으로만 확인 가능.
