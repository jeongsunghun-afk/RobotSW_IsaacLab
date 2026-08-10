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

**진행 중 (16k+ / 60k).** 램프 2 점(8k·16k). 달성률 매치드 Δ(%p):

| cmd | DC@8k | IMP@8k | Δ | DC@16k | IMP@16k | Δ | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 80% | 92% | +12 | 97% | 94% | −3 | 부호 엇갈림 |
| 1.0 | 83% | 91% | +8 | 94% | 89% | −5 | 부호 엇갈림 |
| 1.5 | 84% | 100% | +16 | 97% | 98% | +2 | 밴드 안 |
| 2.0 | 88% | 98% | +11 | 97% | 95% | −2 | 부호 엇갈림 |
| **2.5** | 86% | 81% | −5 | 97% | **77%** | **−20** | 밴드 안(8k 미달) |
| 3.0 | 14% | 3% | −11 | 20% | 25% | +5 | 부호 엇갈림 |
| 3.5 | 0% | 0% | 0 | 0% | 0% | 0 | 밴드 안 |

`IMP@16k` 중앙값: 0.354 / 0.590 / 1.000 / 1.218 / **1.385** / 1.372 / 0.145.

**엄격한 규칙(양쪽 매치드 점에서 부호 일치 + 최소 |Δ| > 15%p)을 통과하는 항목이 pace 엔 없다.**
`cmd 2.5` 하락은 16k 에서 −20%p 로 크지만 8k 가 −5%p 라 미달이다 — stock 에서 확정된
도달률 하락을 **방향으로만 지지**한다. `cmd 3.5` 는 여기서도 0% 다.

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
