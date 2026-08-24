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

## 결과 요약

학습 진행 중 — 램프 미측정.
