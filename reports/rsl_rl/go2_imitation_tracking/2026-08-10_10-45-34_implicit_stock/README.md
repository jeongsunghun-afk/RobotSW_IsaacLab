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

**진행 중 (24k+ / 60k).** 속도 램프 3 점(8k·16k·24k). **판정 기준인 `cmd 3.5` 는 세 점 모두 0%**.
달성률 매치드 Δ(%p):

| cmd | DC@8k | IMP@8k | Δ | DC@16k | IMP@16k | Δ | DC@24k | IMP@24k | Δ | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 48% | 52% | +3 | 67% | 83% | +16 | 59% | 52% | −8 | 엇갈림 |
| 1.0 | 56% | 73% | +17 | 92% | 88% | −5 | 81% | 66% | −16 | 엇갈림 |
| 1.5 | 69% | 77% | +8 | 100% | 94% | −6 | 91% | 72% | −19 | 엇갈림 |
| 2.0 | 78% | 72% | −6 | 100% | 91% | −9 | 92% | 70% | −22 | 부호 일치(8k 밴드 안) |
| **2.5** | 78% | 52% | **−27** | 95% | 56% | **−39** | 92% | 52% | **−41** | ★3 점 전부 밴드 밖 |
| 3.0 | 42% | 22% | −20 | 23% | 38% | +14 | 17% | 33% | +16 | 엇갈림 |
| 3.5 | **8%** | 0% | −8 | 0% | 0% | 0 | 0% | 0% | 0 | implicit 전부 0% |

`cmd 2.5` 중앙값: DC 1.540 / 1.554 / **1.556**(평탄) vs IMP 1.398 / 1.524 / **1.256**.
16k 까지는 "천장은 같고 도달률만 하락"이었는데 **24k 에서는 중앙값도 −19% 로 벌어졌다**
(`cmd 3.0` 중앙값도 1.347 → 0.366).

⚠ 단일 체크포인트 하락에는 전례가 있다(`clip10` 32k 붕괴 → 40k 회복) — **32k 를 봐야
추세 하락과 진동이 갈린다.** 다만 `cmd 2.5` 달성률만은 −27 / −39 / −41 로 3 점 모두 같은
방향이고 전부 밴드 밖이라 진동으로 설명되지 않는다.

⚠ `cmd 3.0` 은 부호가 엇갈려(−20 / +14 / +16) 개선으로 못 쓴다. ⚠ `cmd 3.5` 가 "모든 세대에서
0%" 라는 서술은 **틀렸다** — `DC@8k` 이 8% 다.

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
