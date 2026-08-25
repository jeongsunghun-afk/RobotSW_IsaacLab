# 2026-08-24_10-18-28_fixedstd_vel1_stock

한 줄 목적: **고정 std 0.1 + `vel_err_scale` 1.0** — MimicKit 설정에 가장 가까운 조합.
`task_reward_lerp` 은 baseline 0.5 그대로.

## 2×2 설계

이 arm 으로 격자가 채워진다. 전부 스톡 플랜트 · lerp 0.5 · 60k:

| | `vel_err_scale` 0.5 (우리) | `vel_err_scale` **1.0** (MimicKit) |
|---|---|---|
| **학습 std** (우리) | `implicit_stock` (baseline) | `velscale1_stock` (§14) |
| **고정 std 0.1** (MimicKit) | `fixedstd_stock` | **이 런** |

단독 arm 두 개가 이미 있으므로 이 조합으로 **상호작용**까지 읽을 수 있다. 두 요인이
따로는 안 되는데 같이 걸면 열리는 경우가 이 task 에서 드물지 않다.

## 왜 이 두 개인가

- §14: `vel_err_scale` 1.0 단독은 `cmd 2.5` 를 1.499 → 1.866 으로 올렸지만
  **`cmd 3.5`·`4.0` 은 0% 그대로**. 위상차로 보면 **trot 을 벗어나지 못한다.**
- 남은 MimicKit 차이 중 **탐색량**에 걸리는 것이 action std 다. 학습 std 는 스스로 줄어들어
  (0.25 → 0.15) trot 국소해 탈출 확률이 계속 낮아지고, 고정 std 는 끝까지 같다.

가설: `vel_err_scale` 1.0 이 고속 쪽 **보상 기울기**를 세우고, 고정 std 가 그 기울기를 탈
**탐색량**을 유지한다 — 둘이 같이 있어야 trot → 비대칭 전이가 일어난다.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-24_10-18-28_fixedstd_vel1_stock`
- 시작: 2026-08-24 10:18 · GPU1 (전용) · 60000 iter

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name fixedstd_vel1_stock env.use_pace_params=false \
  env.vel_err_scale=1.0 agent.policy.noise_std_type=fixed agent.policy.init_noise_std=0.1
```

`noise_std_type="fixed"` 구현·검증은 [`../2026-08-24_09-07-16_fixedstd_stock/`](../2026-08-24_09-07-16_fixedstd_stock/) 참조
(buffer 등록 / 단위 테스트 / 런타임 확인).

### 설정 반영 확인

| 키 | 값 |
|---|---|
| `agent.policy.noise_std_type` | **fixed** |
| `agent.policy.init_noise_std` | **0.1** |
| `vel_err_scale` | **1.0** |
| `agent.amp.task_reward_lerp` | 0.5 (baseline) |
| `dr.push_robot` | true (baseline — §14 에서 off 는 무효과로 확인) |
| `use_pace_params` | false |

런타임 `Policy/mean_noise_std` = **0.10000** (min=max, 106 점).
같은 시점 `fixedstd_stock` 은 3107 점 전부 0.10000 — 긴 구간에서도 buffer 가 안 흔들린다.

## 판정 규칙

★ **40k 이후 점으로만**. 재현 산포 ±8~15%p / ±0.02 m/s.
★★ 핵심 질문은 속도가 아니라 **`cmd 2.5` 부근 trot → 비대칭 전이 여부**다
(`logs/gait_phase.py`). 전이 없이 속도만 오르면 §14 의 세 arm 과 같은 결말이다.

## 비교 대상

| arm | cmd 2.5 vx | cmd 3.0 vx | cmd 3.5 % | cmd 4.0 % | top |
|---|---:|---:|---:|---:|---:|
| baseline (0.5 / 학습 std) | 1.499 | 1.309 | 0 | 0 | 1.686 |
| vel_scale 1.0 | 1.866 | 1.329 | 0 | 0 | 1.896 |
| vel1 + nopush | 1.880 | 1.608 | 0 | 0 | 1.914 |
| *lerp 0.8 (style 약화)* | *2.370* | *2.886* | *98* | *98* | *3.937* |

## 결과 (중단, model_51100) — ★ 붕괴

**전 명령 구간에서 못 걷는다.** 40k 램프: `cmd 2.5`~`4.0` 전부 vx ≈ 0 (음수),
`base_h` **0.10~0.15 m** — 기립 0.27~0.32 대비 배를 깔고 누운 자세다.

학습 지표도 같은 말을 한다 (40k): ep_len **13~18** (baseline 970),
mean_reward **2.6~4.9** (743), `standing_base_h` **0.02~0.03** (1.57).

### 처음엔 학습됐다가 2k 이후 무너진다

| iter | 0 | 500 | 1k | 2k | 5k | 20k | 40k |
|---|---:|---:|---:|---:|---:|---:|---:|
| ep_len (fixedstd) | 260 | 664 | 667 | 518 | 48 | 10 | 15 |
| ep_len (+vel1) | 242 | 695 | 725 | 218 | 48 | 203 | 14 |

두 런이 **같은 모양**이다. 탐색이 모자라 못 일어선 게 아니라, 일어선 뒤 무너졌다.

### 원인 — 고정 std × adaptive LR 스케줄러

| | LR @1k | LR @40k | surrogate @1k → @40k | std |
|---|---:|---:|---|---:|
| baseline | 1.1e-4 | 3.4e-5 | −0.0014 → −0.0029 | 0.34 → 0.15 |
| **이 런** | **1e-5 (바닥)** | **1e-5** | **0.0135 → 0.29 (발산)** | 0.1 고정 |

고정 σ 에서 KL 은 `Δμ²/(2σ²)`. baseline std 는 초기 **0.42** 까지 올랐다가 0.15 로 내려오는데
0.1 로 못 박으면 같은 평균 변화가 훨씬 큰 KL 로 찍힌다 → 스케줄러가 LR 을 바닥 1e-5 에 고정 →
정책이 못 따라가 surrogate 발산.

★ MimicKit 은 SGD **상수 LR** 이라 adaptive KL 스케줄이 없다. 그래서 그쪽에서는 성립한다.

### ⚠ 단위도 안 맞았다

MimicKit `pos` 모드는 **action 이 곧 관절 목표각**이다(`char_env.py:472` — 클립 후 그대로
`set_cmd`). 우리는 `action_scale = 0.25` 를 곱해 기본자세에 더한다.
→ MimicKit std 0.1 = **0.1 rad**, 우리 std 0.1 = **0.025 rad** 로 **4 배 작다.**
등가값은 `0.1 / 0.25 = 0.4`.

### 다음

고정 std 를 다시 하려면 **상수 LR + 단위 정합(0.4)** 을 같이 가야 한다. 다만 상수 LR 단독의
효과를 먼저 분리하는 게 순서라
[`../2026-08-25_10-06-22_schedfixed_stock/`](../2026-08-25_10-06-22_schedfixed_stock/) 를 먼저 돌린다.

원자료: `../../_comparisons/mimickit_vs_60_actuator_limit/metrics/` 아래
`ramp_fixedstd/` · `ramp_fixedstd_vel1/`
