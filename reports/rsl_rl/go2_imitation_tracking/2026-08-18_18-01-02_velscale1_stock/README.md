# 2026-08-18_18-01-02_velscale1_stock

한 줄 목적: **MimicKit 설정에 맞추기** — vel_err_scale 0.5 → **1.0** (MimicKit 값).
**`task_reward_lerp` 은 baseline 그대로 0.5.** `implicit_stock` 대비 변수만 다르다.

## 왜 이 arm 인가

MimicKit 은 **우리와 완전히 같은 18 개 클립**과 **같은 50/50 style 혼합**
(`task_reward_weight 0.5 / disc_reward_weight 0.5`)으로 `cmd 4.0` 을 추종한다
(§13). 같은 조건에서 우리 baseline 은 `cmd 4.0` 에서 0.03 m/s · 0% 다.

→ **`lerp` 를 0.8 로 올려 얻은 3.9 m/s 는 MimicKit 의 레시피가 아니라 우회로**이고,
진짜 차이는 다른 곳에 있다. 속도 추종 보상 **식은 동일**하고(`exp(-scale·err²)`)
값만 다르다:

| | 우리 | MimicKit |
|---|---:|---:|
| `vel_err_scale` | 0.5 | **1.0** |
| `dr.push_robot` | 1 m/s 킥 / 5 s | **없음** |

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-18_18-01-02_velscale1_stock`
- 시작: 2026-08-18 · GPU2 · 60000 iter · 스톡 플랜트

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name velscale1_stock env.use_pace_params=false \
  env.vel_err_scale=1.0
```

### ⚠ hydra 경로 함정 — `push_robot` 은 `env.dr.push_robot` 이다

처음에 `env.push_robot=false` 로 띄웠더니 **오류 없이 통과**했는데, cfg 최상위에
`push_robot: false` 라는 **새 키가 생겼을 뿐** env 가 읽는 `dr.push_robot` 은 `true` 로
남아 있었다. 두 런이 baseline 과 동일한 조건으로 돌 뻔했다(발견 후 폐기·재실행).

→ 검증은 `params/env.yaml` 을 **경로로** 확인해야 한다. 최상위에 같은 이름의 키가 있는지
보는 것만으로는 안 된다. 이 런의 확인값:

| 키 | 값 |
|---|---|
| `vel_err_scale` | 1.0 |
| `dr.push_robot` | true |
| `use_pace_params` | false |
| `agent.amp.task_reward_lerp` | 0.5 |

## 판정 규칙

★ **40k 이후 점으로만** 판정. 재현 산포 ±8~15%p / ±0.02 m/s.
★★ 속도와 함께 **보행 종류**(위상차)를 본다 — `logs/gait_phase.py`. §12 에서 고정 lerp 0.8 이
`cmd 4.0` 에서 4 Hz trot 으로 회귀하는 것을 `base_h` 만 보다가 놓쳤다.

## 비교 대상 (`2026-08-10_10-45-34_implicit_stock`, lerp 0.5, 기본 cfg)

| cmd | 40k | 48k | 56k | 59999 |
|---|---:|---:|---:|---:|
| 2.5 중앙 | 1.389 | 1.598 | 1.569 | 1.440 |
| 3.0 중앙 | 0.686 | 1.686 | 1.623 | 1.240 |
| 4.0 달성률 | 0 | 0 | 0 | 0 |

## 결과 요약

학습 진행 중 — 램프 미측정.
