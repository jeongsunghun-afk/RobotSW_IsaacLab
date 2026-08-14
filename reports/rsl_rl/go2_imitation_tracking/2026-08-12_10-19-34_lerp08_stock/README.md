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

## 결과 요약 (완주, 60k) — ★★★ 확정

**`cmd 4.0` 에서 3.80~3.94 m/s · 달성률 97~98%, 판정 구간(40k 이후) 네 점 전부.**
그리고 걸음 품질이 baseline 대비 나빠지지 않았다.

| iter | base(0.5) top | **0.8 `cmd 4.0` 중앙** | 달성률 | `base_h`@4.0 | 부호반전 Hz | 캡도달 최대 |
|---|---|---:|---:|---:|---:|---:|
| 40k | 1.389 @2.5 | **3.893** | 98 | 0.295 | 12.9 | 10% |
| 48k | 1.686 @3.0 | **3.852** | 98 | 0.338 | 12.5 | 8% |
| 56k | 1.623 @3.0 | **3.798** | 97 | 0.365 | 12.2 | 12% |
| 59999 | 1.440 @2.5 | **3.937** | 97 | 0.279 | 12.7 | 7% |

- 6.0 종전 최고 **1.724** → **3.94 m/s**. 5.1 최고 2.058 의 1.9 배.
- implicit 13 개 체크포인트가 전부 0% 였던 `cmd 3.5`·`4.0` 을 **97~98%** 로 통과.
- 40k 이후 네 점이 3.80~3.94 로 흔들림이 거의 없다. 24k(3.998)·32k(4.013)와도 일관된다.
  **이 세대군에서 24k 이하가 뒤집혔던 전례에 걸리지 않는다.**

### 걸음 품질 — baseline 과 같은 대역이다

| | base 0.5 @48k (cmd 3.0) | **0.8 @40k (cmd 4.0)** | noamp 1.0 @24k (cmd 4.0) |
|---|---:|---:|---:|
| `base_h` [m] | 0.305 | **0.295** | 0.177 |
| 관절속도 부호반전 [Hz] | 8.4 | **12.9** | 34.4 |
| 토크 캡 도달률 (최대 관절) | 14% | **10%** | **90%** |

★ **캡 도달률로 보면 0.8 이 baseline 보다 오히려 덜 포화돼 있다** — 2.8 배 빠른데
thigh 캡 도달이 10% 대 14% 다. `|τ|` p95 가 45.4 를 찍는 것은 calf 한둘이 순간적으로
닿는 것이고(6~8%), `noamp` 의 54~90% 상시 포화와는 성격이 다르다.

`base_h` 는 40k~56k 에서 0.295 → 0.365 로 **오히려 올라간다**. `cmd 2.5` 에서는 0.363 으로
baseline(0.305)보다 높다.

### 눈으로 확인

![style 가중치별 보행](../_comparisons/mimickit_vs_60_actuator_limit/figures/style_weight_gait_frames.png)

0.08 s 간격 프레임. 위(style 0.2)는 다리가 스윙/스탠스 위상을 오가는 사족 주행이고,
아래(style 0)는 몸통이 눌린 채 다리가 벌어져 끌린다.

### ⚠ 남은 것

- `cmd 4.0` 은 **명령 범위 상한**이다(`lin_vel_x` 0~4.0). 즉 이 값이 정책의 최고 속도라는
  뜻이 아니라 **범위 안을 다 채웠다**는 뜻이다. 더 위를 보려면 범위를 넓혀야 하고, 그러면
  기존 램프들과 상단 비교가 깨진다.
- 실기 이관 가능성은 미검증. sim 지표상 baseline 과 같은 대역이라는 것까지만 확인됐다.

원자료: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_lerp08/`

### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_32000__lerp08_stock_32k_4ms__20260813-110251.mp4`](videos/model_32000__lerp08_stock_32k_4ms__20260813-110251.mp4) | `model_32000.pt` | `lerp08_stock_32k_4ms` | 2026-08-13 11:09:38 |
<!-- /report-video:videos -->
