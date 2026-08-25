# 2026-08-24_09-07-16_fixedstd_stock

한 줄 목적: action std 를 **학습하지 않고 0.1 로 고정**했을 때(MimicKit 방식) 고속 천장이
열리는지 본다. **`task_reward_lerp` 은 baseline 0.5 그대로.** 변수 하나.

## 왜 이 arm 인가

§13~14 에서 확인된 것:

- MimicKit 은 **우리와 같은 18 개 클립**, **같은 50/50 style 혼합**으로 `cmd 4.0` 을 추종한다.
- 그 차이라고 지목했던 `vel_err_scale`(0.5→1.0)과 `push_robot` off 는 **셋 다 60k 완주 후
  `cmd 3.5`·`4.0` 0%** — 천장을 못 열었다(§14). 가설 기각.
- 위상차로 보면 style 0.5 arm 들은 **전 구간 trot 을 벗어나지 못한다.** lerp 0.8 만
  `cmd 2.5` 부근에서 비대칭 보행으로 전이한다. `vel_err_scale` 1.0 은 그 trot 의 상한을
  1.50 → 1.87 m/s 로 올릴 뿐 **전이를 만들지 못했다.**

남은 MimicKit 차이 중 **탐색량**에 직접 걸리는 것이 action std 다:

| | 우리 | MimicKit |
|---|---|---|
| action std | **학습** (`init 0.25` → 0.15 수렴) | **FIXED 0.1** (`actor_std_type: "FIXED"`) |

학습 std 는 정책이 스스로 탐색량을 줄여 나가므로 trot 국소해에 갇히면 빠져나올 확률이 계속
낮아진다. 고정 std 는 끝까지 같은 크기로 탐색한다. **이게 전이 실패의 원인인지**를 본다.

⚠ MimicKit 은 `FIXED 0.1` 을 `action_bound_weight 10.0`(경계 이탈 소프트 페널티)과 함께 쓴다.
우리는 `clip_actions 10.0`(하드 클립)만 있고 그 항이 없다. 변수를 하나로 두려고 **이번 arm 에는
넣지 않았다** — 고정 std 만으로 열리는지 먼저 본다.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-24_09-07-16_fixedstd_stock`
- 시작: 2026-08-24 09:07 · GPU0 (전용) · 60000 iter · 스톡 플랜트

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name fixedstd_stock env.use_pace_params=false \
  agent.policy.noise_std_type=fixed agent.policy.init_noise_std=0.1
```

### 구현 — `noise_std_type="fixed"` 를 새로 추가했다

rsl_rl 의 `ActorCriticRMA` 는 `"scalar"`/`"log"` 둘 다 std 를 **`nn.Parameter`** 로 만들어
학습시킨다. 고정 옵션이 없어서 추가했다(`rsl_rl/rsl_rl/modules/actor_critic_parkour.py`):

- `"fixed"` 는 `register_buffer("std", ...)` — Parameter 가 아니라 **buffer** 라서 optimizer 가
  건드리지 않고, `state_dict` 에는 남아 **resume 시 값이 그대로 복원**된다.
- 사용부 두 곳(`_update_distribution`, `get_actions_log_prob`)은 `"scalar"` 와 같은 경로를 탄다.
- `state_dependent_std=True` 와 조합하면 즉시 에러(무음 무시 방지).

**단위 테스트로 확인**(GPU 학습 시작 전):

| `noise_std_type` | std ∈ parameters | std ∈ buffers | Adam 1 step 후 변화 |
|---|---|---|---|
| `scalar` (기존) | True | False | **0.1 → 0.2 / 0.0** (학습됨) |
| **`fixed`** (신규) | **False** | **True** | **0.000000** (불변) |

**런타임 확인**: TensorBoard `Policy/mean_noise_std` 가 122 개 점 전부 **정확히 0.1**
(최소=최대=0.1). 설정값이 아니라 실제로 곱해지는 값이다.

### 설정 반영 확인

| 키 | 값 |
|---|---|
| `agent.policy.noise_std_type` | **fixed** |
| `agent.policy.init_noise_std` | **0.1** |
| `agent.amp.task_reward_lerp` | 0.5 (baseline) |
| `vel_err_scale` | 0.5 (baseline) |
| `dr.push_robot` | true (baseline) |
| `use_pace_params` | false |

## 판정 규칙

★ **40k 이후 점으로만** 판정. 재현 산포 ±8~15%p / ±0.02 m/s.
★★ 속도와 **보행 종류**(`logs/gait_phase.py` 위상차)를 같이 본다.
이 arm 의 핵심 질문은 속도 자체가 아니라 **`cmd 2.5` 부근에서 trot → 비대칭 전이가
일어나는가**다. 전이 없이 속도만 조금 오르면 §14 의 세 arm 과 같은 결과다.

## 비교 대상

| arm | cmd 2.5 vx | cmd 3.5 % | cmd 4.0 % | top |
|---|---:|---:|---:|---:|
| baseline (0.5, 학습 std) | 1.499 | 0 | 0 | 1.686 |
| vel_scale 1.0 | 1.866 | 0 | 0 | 1.896 |
| 둘 다 (vel1+nopush) | 1.880 | 0 | 0 | 1.914 |
| *lerp 0.8 (style 약화)* | *2.370* | *98* | *98* | *3.937* |

## 결과 (중단, model_48200) — ★ 붕괴

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
