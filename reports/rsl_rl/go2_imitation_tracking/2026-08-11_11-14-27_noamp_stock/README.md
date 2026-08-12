# 2026-08-11_11-14-27_noamp_stock

한 줄 목적: **AMP style reward 를 정책에서 완전히 떼어내고**(pure task reward) 학습했을 때
`go2_imitation_tracking` 의 속도 천장이 어떻게 움직이는지 본다. **스톡 플랜트** 쪽.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-11_11-14-27_noamp_stock`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-11 11:14 · GPU1 (leg run 과 공유) · ETA 약 15 h · 60000 iter
- 액추에이터: `ImplicitActuatorCfg(effort_limit=None, kp 25, kd 0.5)` — `implicit_stock` 과 동일
- **직전 세대(`2026-08-10_10-45-34_implicit_stock`) 대비 바뀐 것은 AMP 혼합 하나** — 깨끗한 A/B

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name noamp_stock env.use_pace_params=false \
  agent.amp.task_reward_lerp=1.0 agent.amp.task_reward_lerp_start=1.0 \
  agent.amp.task_reward_lerp_anneal_iters=0
```

### AMP 를 끄는 방식

러너는 `if "amp_obs" in extras:` 로 게이트되어 있고 env 가 항상 `amp_obs` 를 넣어 주므로
"AMP 를 끈다"는 스위치는 없다. 대신 혼합식

```
total_reward = task_reward_lerp * rewards + (1 - task_reward_lerp) * style_term
```

에서 `task_reward_lerp = 1.0` 으로 고정하면 **style term 의 계수가 정확히 0** 이 되어 정책이
받는 보상은 순수 task reward 다 (`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:174`).
`_start` 와 `_anneal_iters` 까지 같이 넘겨야 Stage 1 (`_start=0.5`) 이 살아나지 않는다.

**검증 완료** — hydra 오버라이드가 조용히 씹히는 사고(`agent.resume` 전례)를 배제하기 위해 둘 다 확인:

- `params/agent.yaml` 82~84 줄: `task_reward_lerp: 1.0` / `_start: 1.0` / `_anneal_iters: 0`
- TensorBoard **런타임** 값 `Loss/amp_task_reward_lerp` = **1.0 @ iter 0~4** (설정값이 아니라
  실제로 곱해지는 값)

⚠ **`Episode_Reward/amp_reward` 는 계속 기록된다** — discriminator 는 그대로 학습하기 때문이다.
정책 보상에는 **기여가 0** 이므로 이 스칼라를 보상 성분으로 읽지 말 것.

## 판정 규칙

★ 이 세대군은 **40k 이후 점으로만** 판정한다. 8k~24k 차이는 48k 이후를 예측하지 못한다는 것이
`implicit_stock` 에서 실측됐다(Δ −41 → +6 으로 반전). 근거:
`../_comparisons/mimickit_vs_60_actuator_limit/README.md`

## 비교 대상 (`implicit_stock`, AMP on, 동일 플랜트·액추에이터)

| cmd | 지표 | 40k | 48k | 56k |
|---|---|---:|---:|---:|
| 2.5 | 중앙 | 1.389 | 1.598 | 1.569 |
| 3.0 | 중앙 | 0.686 | **1.686** | 1.623 |
| 3.5 | 달성률 | 0 | 0 | 0 |

## 결과 요약 (중간, 29.2k / 60k)

**★★★ 속도는 폭발했지만 걸음이 아니다 — 전형적인 reward hacking 이다.**

24k 램프(판정용이 아니라 "정책이 살아 있나" 확인용으로 측정):

| cmd | AMP on 중앙 | **AMP off 중앙** | on 달성률 | **off 달성률** |
|---|---:|---:|---:|---:|
| 0.5 | 0.268 | 0.635 | 52 | **95** |
| 1.0 | 0.634 | 1.080 | 66 | **95** |
| 2.5 | 1.256 | 2.325 | 52 | **95** |
| 3.0 | 0.366 | 2.776 | 33 | **95** |
| 3.5 | 0.174 | 3.121 | 0 | **95** |
| 4.0 | 0.023 | **3.589** | 0 | **95** |

`cmd 4.0` 에서 **3.589 m/s**. 6.0 종전 최고(1.724)의 2 배, 5.1 최고(2.058)보다도 74% 빠르고,
implicit 13 개 체크포인트가 전부 0% 였던 `cmd 3.5`·`4.0` 을 95% 로 통과한다.

### ★ 그런데 거동을 보면 로봇이 아니다

| 지표 | AMP on (cmd 2.5) | **AMP off (cmd 2.5)** | AMP off (cmd 4.0) |
|---|---:|---:|---:|
| `base_h` [m] | 0.295 | **0.177** | 0.183 |
| pitch [°] | −1.0 | **+7.6** | +7.0 |
| `\|q̇\|` p95 [rad/s] | 5.59 | **16.60** | 20.62 |
| `\|τ\|` p95 최대 [N·m] | 31.2 | **45.4** | **45.4** |
| 관절속도 부호반전 [Hz] | 9.1 | **33.5** | 34.4 |

- **높이 0.18 m** — nominal 기립 0.27~0.32 대비 10 cm 낮은 포복 자세로 달린다.
- **초당 33~34 회 관절 진동.** AMP on 은 9 Hz(정상 보행 스텝 주파수대)다. 이건
  `r2s` 에서 "허우적댐"으로 관측했던 병리와 같은 서명이다([[project_r2s_sim_missing_pace_plant_gap]]).
- **`|τ|` p95 가 전 명령 구간에서 45.4 = calf USD 캡에 정확히 붙어 있다.** 상시 포화다.
- `|q̇|` p95 15.70 은 **calf USD `physxJoint:maxJointVelocity` 15.70 rad/s 그 값**이다.
  implicit 세대 내내 "안 걸린다"고 확인했던 하드 클립이 여기서는 상시로 걸린다.

즉 **3.589 m/s 는 속도 기록이 아니라 물리 상한을 전부 긁는 진동 모드**이고, 실기에 올릴 수
없다. task reward 만 남기면 정책이 이 모드를 찾아낸다는 것 — **AMP style term 이 실제로 그것을
막고 있었다**는 것이 이 실험의 결론이다.

### 학습 지표에도 서명이 남아 있었다

| | 8k | 16k | 24k | 29k |
|---|---:|---:|---:|---:|
| `Policy/mean_noise_std` (AMP off) | 3.055 | 3.484 | 3.767 | **3.953** |
| `Policy/mean_noise_std` (AMP on) | 0.199 | 0.175 | 0.157 | 0.152 |
| `Episode_Reward/standing_base_h` (off / on) | 0.86 / 1.43 | 0.83 / 1.55 | 0.80 / 1.50 | 0.79 / 1.53 |

std 가 0.15 로 수렴하는 대신 **3.9 까지 단조 발산**하고, `standing_base_h` 는 baseline 의 절반이다.
⚠ 단 std 는 과거 램프 붕괴를 못 잡은 전력이 있어 **단독 근거로는 못 쓴다**
([[project_no_train_metric_detects_ramp_collapse]]). 여기서는 램프 실측이 먼저고 std 는 방증이다.

⚠ `Train/mean_reward`(855 vs 680)는 **비교 불가**다 — AMP on 은 `0.5·task + 0.5·style`,
off 는 순수 task 라 정의가 다른 값이다.

### 남은 판단

24k 는 판정 구간(40k+)이 아니다. 다만 여기서 본 것은 "어느 쪽이 몇 %p 낫다"가 아니라 **34 Hz
진동 · 높이 0.18 · 토크 상시 포화라는 질적 병리**라서, 40k 를 기다린다고 성격이 바뀔 종류가
아니다. 계속 돌릴지는 GPU 배분 문제다.

원자료: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_noamp/`
