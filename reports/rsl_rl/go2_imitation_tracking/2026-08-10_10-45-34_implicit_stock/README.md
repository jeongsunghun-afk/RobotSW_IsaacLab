# 2026-08-10_10-45-34_implicit_stock

한 줄 목적: `DCMotorCfg` → `ImplicitActuatorCfg` 로 **토크-속도 곡선을 제거**했을 때
`cmd 3.5` 벽이 열리는지 본다. **스톡 플랜트**(`use_pace_params=false`) 쪽 A/B 짝.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-10_10-45-34_implicit_stock`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-10 10:45 · `--num_envs 4096 --max_iterations 60000 env.use_pace_params=false` · GPU1
- 액추에이터: `ImplicitActuatorCfg(joint_names_expr=[".*"], effort_limit=None, kp 25, kd 0.5)`
  → USD `drive maxForce` 를 그대로 캡으로 쓴다 (hip/thigh **23.70**, calf **45.43** N·m)
- 직전 세대(`yaw10vn15_stock`, `DCMotorCfg` calf 35.5/`velocity_limit` 30.0) 대비
  **바뀐 것은 액추에이터 모델 하나** — yaw ±1.0 · `joint_vel_noise` 0.15 동일

## 결과 요약

**진행 중 (16.5k / 60k).** 속도 램프 2 점(8k·16k)만 있고 **판정 기준인 `cmd 3.5` 는 여전히 0%** 다.

| cmd | DCMotor@16k | **Implicit@16k** |
|---:|---:|---:|
| 0.5 | 0.376 (67%) | 0.371 (**83%**) |
| 1.5 | 1.061 (100%) | 1.102 (94%) |
| 2.5 | **1.554 (95%)** | 1.524 (**56%**) |
| 3.0 | 0.743 (23%) | **1.347 (38%)** |
| 3.5 | 0.147 (0%) | 0.194 (**0%**) |

★ 최고 구간(cmd 2.5)의 토크를 보면 **thigh 4 개가 전부 p95 = 23.70** 으로 캡에 붙어 있다
(캡 도달 9.8~14.8%). 곡선을 없앤 대가로 벽이 **평탄한 thigh maxForce** 로 옮겨갔고,
23.7 은 Unitree 공식 peak 라 올릴 근거가 없다. calf `|q̇|p95` 는 4.5~6.6 이라
경고했던 USD 하드 클립 15.70 rad/s 는 아직 안 걸린다.

전체 표·토크: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/implicit_vs_dcmotor_ramp.md`

## 산출물

### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_16000__implicit_stock_16k__20260810-170343.mp4`](videos/model_16000__implicit_stock_16k__20260810-170343.mp4) | `model_16000.pt` | `implicit_stock_16k` | 2026-08-10 17:04:37 |
<!-- /report-video:videos -->
