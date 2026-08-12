# 2026-08-12_10-11-50_lerp03_stock

한 줄 목적: AMP **style term 비중을 0.5 → 0.7 로 올렸을 때**(`task_reward_lerp` 0.5 → 0.3)
걸음 품질과 속도가 어떻게 움직이는지 본다. **스톡 플랜트**. `implicit_stock` 대비 **변수 1 개**.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-12_10-11-50_lerp03_stock`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-12 10:11 · GPU0 (전용) · 60000 iter · ETA 약 24 h
- 액추에이터: `ImplicitActuatorCfg(effort_limit=None, kp 25, kd 0.5)` — `implicit_stock` 과 동일
- spawn: `max_depenetration_velocity` 1.0 (소스 기본값) — `depen3_stock` 과 다름

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name lerp03_stock env.use_pace_params=false \
  agent.amp.task_reward_lerp=0.3
```

## ★ 방향 주의 — `lerp` 를 **낮추면 style 이 세진다**

```
total_reward = task_reward_lerp · task + (1 − task_reward_lerp) · style
```
(`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:174`)

| `task_reward_lerp` | task : style | 런 |
|---|---|---|
| 1.0 | 100 : 0 | `noamp_stock` — **붕괴**(34 Hz 진동 포복, 중단) |
| 0.5 | 50 : 50 | `implicit_stock` — baseline |
| **0.3** | **30 : 70** | **이 런 — style 을 더 세게** |
| 0.7 | 70 : 30 | (미실행) style 을 줄이는 쪽 |

`task_reward_lerp_start=0.5` / `_anneal_iters=5000` 은 baseline 그대로 두었으므로
**Stage 1 은 baseline 과 동일**하고 5000 iter 에 걸쳐 0.5 → 0.3 으로 내려간다. 변수는 최종값 하나다.

설정·런타임 둘 다 확인:
- `params/agent.yaml:82-84` → `0.3 / 0.5 / 5000`, `params/env.yaml:386` → `use_pace_params: false`
- 런타임 `Loss/amp_task_reward_lerp`: iter 0 에서 0.5, iter 14 에서 0.4994 — 선형 스케줄
  (`0.5 − 0.2·14/5000 = 0.49944`) 위에 정확히 올라가 있다

## 무엇을 기대하나

`noamp_stock`(lerp 1.0)에서 **style 을 없애면 속도는 2 배가 되지만 실기에 못 올리는 진동 모드로
간다**는 것이 나왔다([`../2026-08-11_11-14-27_noamp_stock/`](../2026-08-11_11-14-27_noamp_stock/)).
이 런은 그 반대 방향이다 — style 을 더 세게 걸면 걸음 품질 지표가 baseline 보다 좋아지는지,
그 대가로 속도를 얼마나 잃는지를 잰다.

램프 속도와 함께 **걸음 품질 지표를 같이 본다**(AMP off 에서 병리를 잡아낸 그 지표들):

| 지표 | AMP off (1.0) | baseline (0.5) | 이 런 (0.3) |
|---|---:|---:|---|
| `base_h` [m] | 0.177 | 0.295 | ? |
| `\|q̇\|` p95 [rad/s] | 16.60 | 5.59 | ? |
| `\|τ\|` p95 최대 | 45.4 (캡) | 31.2 | ? |
| 관절속도 부호반전 [Hz] | 33.5 | 9.1 | ? |

(전부 `cmd 2.5` 구간, 24k 기준)

## 판정 규칙

★ 이 세대군은 **40k 이후 점으로만** 판정한다. 8k~24k 차이는 48k 이후를 예측하지 못한다는 것이
`implicit_stock` 에서 실측됐다. 재현 산포는 달성률 **±8~15%p**, 중앙값 ±0.02 m/s.
근거: `../_comparisons/mimickit_vs_60_actuator_limit/README.md`

## 비교 대상 (`2026-08-10_10-45-34_implicit_stock`, lerp 0.5)

| cmd | 지표 | 40k | 48k | 56k |
|---|---|---:|---:|---:|
| 2.5 | 중앙 | 1.389 | 1.598 | 1.569 |
| 3.0 | 중앙 | 0.686 | **1.686** | 1.623 |
| 3.5 | 달성률 | 0 | 0 | 0 |

## 결과 요약

학습 진행 중 — 램프 미측정.
