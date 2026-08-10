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

**진행 중 (14.5k / 60k).** 램프는 **8k 한 점뿐**이라 방향을 말할 수 없다.

| cmd | DCMotor@8k | **Implicit@8k** |
|---:|---:|---:|
| 0.5 | 0.361 (80%) | 0.348 (**92%**) |
| 1.5 | 0.998 (84%) | 1.020 (**100%**) |
| 2.0 | 1.270 (88%) | 1.232 (**98%**) |
| 2.5 | **1.394 (86%)** | 1.336 (81%) |
| 3.0 | **1.341 (14%)** | 0.886 (3%) |
| 3.5 | 0.173 (0%) | 0.132 (**0%**) |

저·중속 달성률(92/91/100/98%)은 어느 세대보다 좋고 고속 꼬리는 낮다. 다만 8k 는
직전 세대에서도 최저점이었고(pace 8k 1.394 → 56k 1.724), **한 점으로 판정하지 않는다.**

토크(cmd 2.5, 달린 env 62/64): 앞다리 thigh 는 p95 6.1/6.9 로 놀고 있고 **뒷다리 thigh 만**
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
