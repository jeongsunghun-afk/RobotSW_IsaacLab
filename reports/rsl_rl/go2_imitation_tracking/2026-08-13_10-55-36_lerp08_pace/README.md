# 2026-08-13_10-55-36_lerp08_pace

한 줄 목적: 스톡에서 나온 **`task_reward_lerp` 0.8 (style 0.2) → `cmd 4.0` 4.01 m/s** 가
**PACE 플랜트에서도 재현되는지** 본다. `implicit_pace` 대비 **변수 1 개**.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-13_10-55-36_lerp08_pace`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-13 10:55 · GPU0 (전용) · 60000 iter
- 플랜트: **PACE** (`use_pace_params=true`, 소스 기본값)
- 액추에이터: `ImplicitActuatorCfg(effort_limit=None, kp 25, kd 0.5)` — `implicit_pace` 와 동일
- spawn: `max_depenetration_velocity` 1.0 (소스 기본값)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name lerp08_pace agent.amp.task_reward_lerp=0.8
```

설정·런타임 확인: `params/agent.yaml:82-84` → `0.8 / 0.5 / 5000`,
`params/env.yaml:386` → `use_pace_params: true`.

## 왜 이 arm 이 필요한가

스톡에서 style 비중을 0.5 → 0.2 로 줄이자 `cmd 4.0` 달성률이 **0% → 98%**, 중앙값
**4.013 m/s** 가 나왔다([`../2026-08-12_10-19-34_lerp08_stock/`](../2026-08-12_10-19-34_lerp08_stock/)).
6.0 종전 최고 1.724 의 2 배가 넘는다.

⚠ 그런데 **이 task 는 플랜트마다 부호가 반대로 나온 전례가 있다** — 액추에이터를
`DCMotor → ImplicitActuator` 로 바꿨을 때 stock 은 `cmd 3.0` +14~+30%p, pace 는 −36~−42%p 로
정반대였다([`../_comparisons/mimickit_vs_60_actuator_limit/`](../_comparisons/mimickit_vs_60_actuator_limit/)).
따라서 **스톡 결과 하나만으로 "style 가중치가 천장이었다"고 일반화하면 안 된다.**

PACE 는 식별된 armature 0.17~0.20 · viscous 2.3~2.5 를 쓰는 무거운 플랜트라, style 압력을 줄여
얻는 여유가 스톡만큼 있을 이유가 없다.

## 비교 대상 (`2026-08-10_10-45-09_implicit_pace`, lerp 0.5, 같은 플랜트)

| cmd | 지표 | 40k | 48k | 56k |
|---|---|---:|---:|---:|
| 2.5 | 중앙 | 1.375 | 1.418 | 1.406 |
| 3.0 | 중앙 | 1.294 | **1.497** | 1.438 |
| 3.0 | 달성률 | 28 | 47 | 39 |
| 3.5 | 달성률 | 0 | 0 | 0 |

같은 플랜트의 `DCMotor` 세대는 `cmd 3.0` 에서 **1.724 (80%)** 까지 갔다 — PACE 에서는 DCMotor 가
ImplicitActuator 보다 나았다. 이 arm 이 그 값을 넘으면 의미가 크다.

## 판정 규칙

★ **40k 이후 점으로만** 판정한다. 재현 산포는 달성률 ±8~15%p, 중앙값 ±0.02 m/s.

★★ **속도와 걸음 품질을 반드시 같이 읽는다.** `noamp_stock`(style 0)은 3.589 m/s 를 냈지만
34 Hz 진동 · 높이 0.18 m 의 실기 불가 모드였다. 램프 npz 에서 다음을 함께 뽑는다:

| 지표 | 판정 기준 (스톡 기준값) |
|---|---|
| `base_h` [m] | 0.25 이상 유지 (붕괴 시 0.18) |
| 관절속도 부호반전 [Hz] | 15 이하 (정상 8~10, 붕괴 34) |
| `\|τ\|` p95 최대 | calf 캡 45.4 에 상시 포화되지 않을 것 |

램프 커맨드 (PACE 플랜트이므로 `--no_pace` 를 **주지 않는다**):

```bash
python _workspace/go2_tracking/speed_ramp_record.py \
  --checkpoint logs/rsl_rl/go2_imitation_tracking/2026-08-13_10-55-36_lerp08_pace/model_40000.pt \
  --task Go2-Imitation-Tracking-v0 --num_envs 64 --no_video \
  --out_dir reports/rsl_rl/go2_imitation_tracking/_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_lerp08_pace/pace_40000
```

## 결과 요약 (중간, 47.8k / 60k) — ★★ 플랜트 반전 없음

**PACE 에서도 재현된다.** 40k 판정점 하나:

| cmd | `implicit_pace`(0.5) 중앙 | **0.8 중앙** | base % | **0.8 %** |
|---|---:|---:|---:|---:|
| 2.5 | 1.375 | 2.396 | 78 | **97** |
| 3.0 | 1.294 | 2.904 | 28 | **97** |
| 3.5 | 0.174 | 3.419 | 0 | **97** |
| 4.0 | 0.101 | **3.969** | 0 | **97** |

top: base 1.375 @cmd 2.5 → **3.969 @cmd 4.0**.

걸음 품질 (`cmd 4.0`): `base_h` **0.335 m**, 관절속도 부호반전 **12.8 Hz**,
캡 도달률 최대 8% — 스톡 40k(0.295 / 12.9 Hz / 10%)와 사실상 같다.

★ **이 arm 을 띄운 이유였던 "플랜트별 부호 반전" 우려는 해소됐다.** 액추에이터 변경 때는
stock +14~+30%p / pace −36~−42%p 로 정반대였지만, style 가중치는 **두 플랜트에서 같은
방향으로 같은 크기**다. 오히려 PACE 쪽 `cmd 4.0` 중앙값(3.969)이 스톡 40k(3.893)보다 약간 높다.

같은 플랜트 DCMotor 세대의 최고 기록 **1.724 @cmd 3.0** 도 넘는다.

⚠ 40k 한 점이다. 48k·56k 로 확인한다.

원자료: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_lerp08_pace/`
