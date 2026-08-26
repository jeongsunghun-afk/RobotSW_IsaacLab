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

## 결과 (60k 완주) — 붕괴 없음. **달성률은 열렸고 전이는 안 열렸다**

### 학습 지표 — 전 구간 baseline 위

`Loss/learning_rate` 는 60000 점 전부 2.00e-04(min=max). Adam + 상수 2e-4 가 발산할 거라던
사전 우려는 빗나갔다 — surrogate 는 0.044 → 0.0005 로 단조 수렴했다.

| iter | 1k | 5k | 20k | 40k | 59999 |
|---|---:|---:|---:|---:|---:|
| ep_len (이 런 / baseline) | 932 / 873 | 940 / 923 | 969 / 951 | 980 / 956 | **996 / 980** |
| mean_reward | 523 / 473 | 687 / 653 | 753 / 733 | 785 / 744 | **810 / 750** |
| `lin_vel_reward` | 20.7 / 17.3 | 27.7 / 28.7 | 29.1 / 30.2 | 34.9 / 30.4 | **33.4 / 28.9** |
| `amp_reward` | 23.7 / 21.4 | 34.1 / 32.0 | 39.6 / 38.1 | 39.8 / 38.5 | **43.3 / 40.4** |

★ task 항과 style 항이 **동시에** 올랐다 — 둘을 맞바꾼 게 아니다.

### 램프 (40k 이후 4 점, `--num_envs 64 --no_pace`)

```
                       arm |   cmd 2.5    cmd 3.0    cmd 3.5    cmd 4.0
    baseline (adaptive LR) | 1.598(97)  1.686(58)  0.592( 0)  0.049( 0)
 const LR (schedule=fixed) | 1.586(95)  1.673(95)  1.626( 9)  0.131( 0)
```

★ `cmd 3.0` 에서 **median vx 는 사실상 동일한데(1.673 vs 1.686) 달성률만 58% → 95%** 다.
빨라진 게 아니라 넘어지거나 멈추던 env 가 안 넘어진다. §14 의 세 arm 은 정반대로 속도만 올리고
달성률은 58~64% 그대로였으니, **서로 다른 축**을 건드리고 있다.

`cmd 4.0` 은 여전히 0%. 다만 걸음 품질은 붕괴 모드가 아니다(`base_h` 0.271, flip 8.4 Hz) —
무너진 게 아니라 그냥 안 달린다.

### ★★ 판정: §14 와 같은 결말 — 전이가 없다

위상차(`../_comparisons/mimickit_vs_60_actuator_limit/metrics/gait_phase.md`)는 **전 명령 구간
trot** 이다. 56k 의 `cmd 3.5` 1.626 도 2.67 Hz trot 이고, 대각 위상차 FL-RR 은 0.00~0.01 로
끝까지 붙어 있다. 같은 표에서 PACE lerp 0.8 은 `cmd 3.0` 부터 FL-FR 0.11 의 **bound** 로 넘어간다.

→ **style 0.5 는 trot 에 잠근다**는 §14 의 관측이 다시 확인됐다. 상수 LR 은 그 trot 을 더 튼튼하게
만들 뿐 벗어나게 하지 못한다. 이 arm 은 천장을 열지 못했다.

### 이 arm 이 남긴 것

1. 고정 std 붕괴의 원인 진단(**std 자체가 아니라 adaptive 스케줄러와의 상호작용**)이 뒷받침됐다.
   → 상수 LR + 단위 정합(0.4)으로 고정 std 재시도가 가능해졌다.
2. `cmd 3.0` 달성률 +37%p 는 산포(±8~15%p)를 넘는다. **속도 축과 독립인 안정성 축**이 있다.

원자료: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_schedfixed/`
