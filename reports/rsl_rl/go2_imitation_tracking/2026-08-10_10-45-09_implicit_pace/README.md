# 2026-08-10_10-45-09_implicit_pace

한 줄 목적: `DCMotorCfg` → `ImplicitActuatorCfg` 로 **토크-속도 곡선을 제거**했을 때
`cmd 3.5` 벽이 열리는지 본다. **PACE 플랜트**(`use_pace_params=true`) 쪽 A/B 짝.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-10_10-45-09_implicit_pace`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-10 10:45 · `--num_envs 4096 --max_iterations 60000` (PACE 기본 ON) · GPU3
- 액추에이터: `ImplicitActuatorCfg(joint_names_expr=[".*"], effort_limit=None, kp 25, kd 0.5)`
  → USD `drive maxForce` 를 그대로 캡으로 쓴다 (hip/thigh **23.70**, calf **45.43** N·m)
- 직전 세대(`yaw10vn15_pace`) 대비 **바뀐 것은 액추에이터 모델 하나**

## 결과 요약

**진행 중 (24k+ / 60k).** 램프 3 점(8k·16k·24k). 달성률 매치드 Δ(%p):

| cmd | DC@8k | IMP@8k | Δ | DC@16k | IMP@16k | Δ | DC@24k | IMP@24k | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 80% | 92% | +12 | 97% | 94% | −3 | 72% | 80% | +8 |
| 1.0 | 83% | 91% | +8 | 94% | 89% | −5 | 80% | 73% | −6 |
| 1.5 | 84% | 100% | +16 | 97% | 98% | +2 | 84% | 100% | +16 |
| 2.0 | 88% | 98% | +11 | 97% | 95% | −2 | 84% | 95% | +11 |
| **2.5** | 86% | 81% | **−5** | 97% | 77% | **−20** | 84% | **62%** | **−22** |
| 3.0 | 14% | 3% | −11 | 20% | 25% | +5 | 48% | 9% | −39 |
| 3.5 | 0% | 0% | 0 | 0% | 0% | 0 | 0% | 0% | 0 |

`cmd 2.5` 중앙값 — DC 1.394 / 1.452 / **1.497**(상승) vs IMP 1.336 / 1.385 / **1.298**(하락).
격차가 −4% → −5% → **−13%** 로 벌어진다. stock 과 같은 모양이다.

★★ `cmd 2.5` 달성률은 **두 플랜트 × 3 체크포인트 = 6 개 매치드 점 전부 음수**다
(stock −27/−39/−41, pace −5/−20/−22). pace 는 8k 가 −5%p 라 "양쪽 |Δ| > 15%p" 규칙 자체는
미달이지만, **부호 6/6 일치는 재현 변동으로 설명되지 않는다.**

⚠ `cmd 3.0` 은 pace 에서도 부호가 엇갈린다(−11 / +5 / −39). ★ `cmd 3.5` 는 여기서도 전부 0%.

토크(8k, cmd 2.5, 달린 env 62/64): 앞다리 thigh 는 p95 6.1/6.9 로 놀고 있고 **뒷다리 thigh 만**
21.95/23.70 으로 캡에 붙는다. calf RL/RR 은 p95 39.7/42.2 로 새로 열린 45.43 헤드룸을
실제로 쓰고 있다. calf `|q̇|p95` 4.6~5.6 → USD 클립 15.70 은 안 걸린다.

전체 표·토크: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/implicit_vs_dcmotor_ramp.md`

## 산출물

### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_14000__implicit_pace_14k__20260810-170454.mp4`](videos/model_14000__implicit_pace_14k__20260810-170454.mp4) | `model_14000.pt` | `implicit_pace_14k` | 2026-08-10 17:05:49 |
<!-- /report-video:videos -->
