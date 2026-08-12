# 2026-08-12_10-19-34_lerp08_stock

한 줄 목적: AMP **style term 비중을 0.5 → 0.2 로 줄였을 때**(`task_reward_lerp` 0.5 → 0.8)
속도 천장이 열리는지, 그리고 `noamp_stock`(style 0)의 붕괴가 여기서도 시작되는지 본다.
**스톡 플랜트**. `implicit_stock` 대비 **변수 1 개**.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-12_10-19-34_lerp08_stock`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-12 10:19 · GPU1 (전용) · 60000 iter
- 액추에이터: `ImplicitActuatorCfg(effort_limit=None, kp 25, kd 0.5)` — `implicit_stock` 과 동일
- spawn: `max_depenetration_velocity` 1.0 (소스 기본값) — `depen3_stock` 과 다름

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name lerp08_stock env.use_pace_params=false \
  agent.amp.task_reward_lerp=0.8
```

## style 비중 축 위에서 이 런의 위치

```
total_reward = task_reward_lerp · task + (1 − task_reward_lerp) · style
```
(`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:174`) — **`lerp` 는 task 쪽 계수**라 올릴수록
style 이 약해진다.

| `task_reward_lerp` | task : style | 런 | 결과 |
|---|---|---|---|
| 0.5 | 50 : 50 | `implicit_stock` | baseline (top 1.686 @ cmd 3.0, 48k) |
| **0.8** | **80 : 20** | **이 런** | ? |
| 1.0 | 100 : 0 | `noamp_stock` | **붕괴** — 3.589 m/s 지만 34 Hz 진동 포복, 29.6k 에서 중단 |

`task_reward_lerp_start=0.5` / `_anneal_iters=5000` 은 baseline 그대로라 **Stage 1 은 baseline 과
동일**하고 5000 iter 에 걸쳐 0.5 → 0.8 로 올라간다. 변수는 최종 혼합비 하나다.

설정·런타임 둘 다 확인:
- `params/agent.yaml:82-84` → `0.8 / 0.5 / 5000`, `params/env.yaml:386` → `use_pace_params: false`
- 런타임 `Loss/amp_task_reward_lerp`: iter 0 → 0.5, iter 1 → 0.50006
  (`0.5 + 0.3·1/5000 = 0.50006`) — 선형 스케줄 위에 정확히 올라가 있다

## 이 런이 답해야 하는 질문

`noamp_stock` 에서 **style 을 완전히 없애면 속도가 2 배가 되지만 실기에 못 올리는 진동 모드로
간다**는 것이 나왔다([`../2026-08-11_11-14-27_noamp_stock/`](../2026-08-11_11-14-27_noamp_stock/)).
style 0.5 는 멀쩡하고 style 0 은 붕괴다. **style 0.2 는 어느 쪽인가** — 붕괴가 이미 시작됐는지,
아니면 걸음을 유지한 채 속도만 얻는 구간이 존재하는지.

따라서 램프에서 **속도와 걸음 품질을 반드시 같이 읽는다**. 속도만 보면 `noamp_stock` 을
"6.0 최고 기록"으로 오독하게 된다.

| 지표 (`cmd 2.5`, 24k) | style 0 (lerp 1.0) | baseline (lerp 0.5) | 이 런 (lerp 0.8) |
|---|---:|---:|---|
| `base_h` [m] | 0.177 | 0.295 | ? |
| pitch [°] | +7.6 | −1.0 | ? |
| `\|q̇\|` p95 [rad/s] | 16.60 | 5.59 | ? |
| `\|τ\|` p95 최대 [N·m] | 45.4 (캡 상시) | 31.2 | ? |
| 관절속도 부호반전 [Hz] | 33.5 | 9.1 | ? |

**속도가 올라도 위 지표가 style 0 쪽으로 붙으면 실패로 읽는다.**

## 판정 규칙

★ 이 세대군은 **40k 이후 점으로만** 판정한다(8k~24k 차이가 48k 이후를 예측하지 못한다는 것이
`implicit_stock` 에서 실측됨). 재현 산포는 달성률 **±8~15%p**, 중앙값 ±0.02 m/s.
근거: `../_comparisons/mimickit_vs_60_actuator_limit/README.md`

⚠ 단 걸음 품질 지표는 예외다 — `noamp_stock` 의 병리(34 Hz·높이 0.18)는 24k 에서 이미 질적으로
명백했다. **붕괴 여부만은 24k 에서 조기 판단해도 된다.**

## 비교 대상 (`2026-08-10_10-45-34_implicit_stock`, lerp 0.5)

| cmd | 지표 | 40k | 48k | 56k |
|---|---|---:|---:|---:|
| 2.5 | 중앙 | 1.389 | 1.598 | 1.569 |
| 3.0 | 중앙 | 0.686 | **1.686** | 1.623 |
| 3.5 | 달성률 | 0 | 0 | 0 |

## 이력

`lerp=0.3`(style 0.7, 반대 방향) 런을 2026-08-12 10:11 에 띄웠다가 **6 분 만에 사용자 지시로
중단**하고 이 런으로 대체했다. 체크포인트 3 개(0/100/200)뿐이라 보고서를 남기지 않는다.
로그만 `logs/rsl_rl/go2_imitation_tracking/2026-08-12_10-11-50_lerp03_stock/` 에 있다.

## 결과 요약

학습 진행 중 — 램프 미측정.
