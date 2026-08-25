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

**56k / 60k.** 램프 7 점 전부 확보. **★ 이 플랜트에서는 DCMotor 가 확실히 낫다 — stock 과 반대다.**

| cmd | 지표 | 8k | 16k | 24k | 32k | 40k | 48k | 56k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | DC 중앙 | 1.394 | 1.452 | 1.497 | 1.533 | 1.585 | 1.602 | 1.633 |
| 2.5 | IMP 중앙 | 1.336 | 1.385 | 1.298 | 1.396 | 1.375 | 1.418 | 1.406 |
| 2.5 | Δ 달성률 | −5 | −20 | −22 | −17 | −3 | −9 | −8 |
| 3.0 | DC 중앙 | 1.341 | 1.213 | 1.455 | 1.596 | 1.651 | 1.657 | **1.724** |
| 3.0 | IMP 중앙 | 0.886 | 1.372 | 1.063 | 1.332 | 1.294 | 1.497 | 1.438 |
| 3.0 | **Δ 달성률** | −11 | +5 | **−39** | **−42** | **−38** | **−36** | **−41** |
| 3.5 | IMP 달성률 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **`cmd 3.0` 이 24k 이후 5 개 연속 −36~−42%p 로 뒤진다.** DC 는 56k 에서 **1.724 m/s (80%)** 까지
  가는데 IMP 는 1.438 (39%)에 그친다. 1.724 는 이 세대군 전체에서 가장 빠른 값이다.
- `cmd 2.5` 는 7 점 전부 음수(−3 ~ −22)이고 중앙값도 전 구간 뒤진다.
- ★ `cmd 3.5` 는 여기서도 7 점 전부 0%.

★★ **같은 액추에이터 변경이 stock 에서는 이득(`cmd 3.0` +14~+30%p), pace 에서는 손해**다.
과거 8k 한 점으로 "플랜트별 부호 반전"을 주장했다가 철회한 적이 있는데, 이번엔 각 방향이
**4~6 개 연속 체크포인트**로 지지되고 폭도 20~40%p 라 근거의 질이 다르다.

토크(48k, cmd 3.0, 달린 env 31/64): **앞다리 thigh 가 p95 4.8/5.5 로 논다**(뒷다리만 캡 도달
12.3/16.5%). 같은 구간 stock 은 네 thigh 가 전부 캡에 붙는다 — 더 빠른 쪽이 네 다리를 다 쓰는
쪽이고, pace 열세의 후보 설명이지만 **인과는 미검증**이다.

전체 표·그림: `../../_comparisons/mimickit_vs_60_actuator_limit/metrics/implicit_full_trajectory.md`
· `.../figures/implicit_full_trajectory.png`

## 산출물

### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_14000__implicit_pace_14k__20260810-170454.mp4`](videos/model_14000__implicit_pace_14k__20260810-170454.mp4) | `model_14000.pt` | `implicit_pace_14k` | 2026-08-10 17:05:49 |
<!-- /report-video:videos -->
