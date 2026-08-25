# 2026-08-25_10-06-22_schedfixed_stock

한 줄 목적: **adaptive LR 스케줄을 끄고 상수 LR** 로 학습한다(MimicKit 방식).
나머지는 전부 baseline — 학습 std, `vel_err_scale` 0.5, `lerp` 0.5, push on. **변수 하나.**

## 왜 이 arm 인가 — 앞 arm 이 여기서 죽었다

`fixedstd_stock` / `fixedstd_vel1_stock`(고정 std 0.1)은 **둘 다 붕괴**해서 48k/51k 에서 중단했다.
ep_len 이 1k 에서 667 까지 정상적으로 올라갔다가 2k 이후 무너져 40k 에 **13~18 step** 이 됐다.

원인은 고정 std 자체가 아니라 **adaptive LR 스케줄러와의 상호작용**이었다:

| | LR @1k | LR @40k | surrogate @1k → @40k | std |
|---|---:|---:|---|---:|
| baseline | 1.1e-4 | 3.4e-5 | −0.0014 → −0.0029 | 0.34 → 0.15 |
| fixedstd 0.1 | **1e-5 (바닥)** | **1e-5** | **0.0135 → 0.29 (발산)** | 0.1 고정 |

고정 σ 에서 KL 은 `Δμ²/(2σ²)` 다. baseline 의 std 는 초기에 **0.42 까지 올라갔다가** 0.15 로
내려오는데, 0.1 로 못 박으면 같은 평균 변화가 훨씬 큰 KL 로 찍힌다. 스케줄러가 `desired_kl=0.01`
초과로 판단해 LR 을 **바닥 1e-5 에 고정**하고, 거기서 정책이 못 따라가 surrogate 가 발산한다.

★ **MimicKit 에는 이 문제가 없다** — SGD **상수 LR**(actor 2e-4 / critic 1e-4)이라 adaptive KL
스케줄이 아예 없다. 그래서 그쪽에서는 `FIXED 0.1` 이 성립한다.

→ 고정 std 를 다시 시도하기 전에 **상수 LR 단독**이 baseline 을 어떻게 바꾸는지 먼저 본다.
이게 깨끗한 순서다.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-25_10-06-22_schedfixed_stock`
- 시작: 2026-08-25 10:06 · GPU0 (전용) · 60000 iter · 스톡 플랜트

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name schedfixed_stock env.use_pace_params=false \
  agent.algorithm.schedule=fixed
```

`schedule` 이 `"adaptive"` 가 아니면 LR 조정 블록 전체가 건너뛰어진다
(`rsl_rl/rsl_rl/algorithms/ppo_parkour.py:347` — `PPOAMP` 는 `PPOParkour` 를 상속).

### 설정·런타임 확인

| 키 | 값 |
|---|---|
| `agent.algorithm.schedule` | **fixed** |
| `agent.algorithm.learning_rate` | 2e-4 (상수로 유지됨) |
| `agent.policy.noise_std_type` / `init_noise_std` | scalar / 0.25 (**baseline — std 는 학습**) |
| `agent.amp.task_reward_lerp` | 0.5 |
| `vel_err_scale` · `dr.push_robot` · `use_pace_params` | 0.5 · true · false |

런타임: `Loss/learning_rate` 217 점 전부 **2.00e-04** (min=max). `Policy/mean_noise_std` 는
0.251 → 0.336 으로 **학습되고 있다**(고정 아님). ep_len 은 216 iter 에서 이미 893.

⚠ **상수 2e-4 는 baseline 의 adaptive 가 수렴하는 값(3~5e-5)보다 4~6 배 높다.** MimicKit 은
같은 2e-4 를 **SGD** 로 쓰는데 우리는 **Adam** 이라 실효 步幅이 훨씬 크다. 발산 가능성이 있고,
발산하면 그 자체가 "상수 LR 을 쓰려면 optimizer 도 같이 봐야 한다"는 결과다.

## 판정 규칙

★ **40k 이후 점으로만**. 재현 산포 ±8~15%p / ±0.02 m/s.
★★ 속도와 **보행 종류**(`logs/gait_phase.py`)를 같이 본다 — style 0.5 arm 들은 지금까지 전부
trot 을 못 벗어났다. 이 arm 도 `cmd 2.5` 부근 전이 여부가 핵심이다.

## 비교 대상

| arm | cmd 2.5 vx | cmd 3.0 vx | cmd 3.5 % | cmd 4.0 % | top |
|---|---:|---:|---:|---:|---:|
| baseline (adaptive LR) | 1.499 | 1.309 | 0 | 0 | 1.686 |
| vel_scale 1.0 | 1.866 | 1.329 | 0 | 0 | 1.896 |
| vel1 + nopush | 1.880 | 1.608 | 0 | 0 | 1.914 |
| fixed std 0.1 | **붕괴** | 붕괴 | — | — | −0.03 |
| *lerp 0.8 (style 약화)* | *2.370* | *2.886* | *98* | *98* | *3.937* |

## 결과 요약

학습 진행 중 — 램프 미측정.
