# 2026-08-26_14-04-56_tannorm_stock

한 줄 목적: 관절 각도를 **raw 라디안 12** 대신 **관절별 회전의 tan-norm 72**(MimicKit 표현)로
넣는다. 나머지는 전부 baseline. **변수 하나.**

## 왜 이 arm 인가 — 관측 축은 한 번도 실험한 적이 없다

§16 에서 두 쪽 관측을 처음 대조했고, 차이가 컸다(차원은 양쪽 **체크포인트 가중치 shape** 으로 검산):

| | 우리 | MimicKit |
|---|---|---|
| **관절 위치** | `joint_pos − default` **12** (raw rad) | 관절별 quat → tan-norm **96** |
| 몸통 자세 | `projected_gravity_b` 3 | tan-norm 6 |
| 이전 action | 12 | 없음 |
| proprio 합 | **42** | **117** |

우리 관절 입력은 **무한 정의역의 선형 라디안**이다. 이런 입력은 학습 분포의 **가장자리**에서
성질이 나빠진다(정규화 통계 이동·활성 포화) — 그리고 고속 천장이 있는 지점이 정확히 그 가장자리다.
tan-norm 은 [−1,1] 유계이고 θ 에 대해 주기적이다. §16-f 에서 후보 1 순위로 적은 것이 이것이다.

⚠ estimator 오차는 §16-d 에서 **이미 기각**했다(실제로 3.9 m/s 를 내는 arm 에서 오차가 1% 로 가장
작았다). 이 arm 은 그다음 후보다.

## 구현 — `joint_pos_tan_norm` 플래그 (기본 False)

`q = angle_axis(a_j, θ_j)` 를 만들고 `[R(q)·(1,0,0), R(q)·(0,0,1)]` 6 차원을 낸다
(MimicKit `torch_util.quat_to_tan_norm` 과 같은 식). 축은 Go2 실측 — hip=x, thigh/calf=y.

설계에서 조심한 것 셋:

1. **`observation_space` 는 cfg 상수가 아니라 env `__init__` 이 `super().__init__` 전에 다시 계산한다.**
   hydra 오버라이드가 확정되는 시점이 거기다(cfg `__post_init__` 은 너무 이르다).
2. **관절각 노이즈·`encoder_bias` 는 라디안 공간에서 더한 뒤 인코딩한다.** tan-norm 출력에 라디안을
   더하면 물리적 의미가 사라진다. raw 경로는 인코딩이 항등이라 **bit-identical** 이다.
3. **`_apply_obs_dr` 인덱스를 하드코딩하지 않는다.** `self._obs_idx_joint_vel` 에서 유도한다 —
   이 인덱스가 밀린 사고가 이미 한 번 있었다(σ=0.2 가 σ=0.05 자리에 주입돼 4 배 증폭).

θ 는 **기본자세 상대**를 쓴다. 절대각과는 `R(a, θ_rel+θ_def) = R(a,θ_def)·R(a,θ_rel)` 로
관절마다 고정 회전 하나만큼만 다르고 그 차이는 첫 선형층이 흡수하므로, 상대각을 쓰면 DR 의미가
raw 경로와 그대로 같아진다.

### 학습 전 실측 검증 (env 를 실제로 굴려서)

| | raw (baseline) | tan-norm (이 arm) |
|---|---|---|
| `cfg.observation_space` / 실제 shape | 42 / (N, 42) | **102 / (N, 102)** |
| 상수 차원 (std < 1e-6) | 13 | **45** |
| policy obs finite | True | True |
| 정규화 출력 finite | True | **True** |
| `\|tan\|² + \|norm\|²` | — | **2.0000~2.0000** |
| 값 범위 | — | **[−0.776, 1.000]** |

★ 상수 45 = 원래 dead 인 13(`actions` 12 + `vy` 1) + **32**. 이 32 는 Go2 축 때문에 구조적으로
θ 와 무관하다: hip(축 x)은 `tan=(1,0,0)` 전체와 `norm` 의 x 성분이, thigh/calf(축 y)는 두 벡터의
y 성분이 상수다. **NaN 이 아니다** — `EmpiricalNormalization` 이 `std + eps`(eps=1e-2)로 나눠
정규화 후 정확히 0 이 된다. NaN 이었다면 `act()` 의 `torch.nan_to_num` 이 조용히 0 으로 덮어
**멀쩡해 보이는 채로 쓰레기를 학습**했을 것이라 먼저 확인했다.

## ⚠ 교란 요인 — 표현만 바뀐 게 아니다

42 → 102 는 **actor 1 층(68→128)과 history encoder 입력(420→1020)도 같이 키운다.** 이 arm 이
이기면 표현 덕인지 용량 덕인지 이 실험만으로는 못 가른다.

★ 정보량은 관절당 (cos θ, sin θ) **2 개뿐**이다. 성공하면 차원을 맞춘 대조군은 또 다른 72 가
아니라 **cos/sin 24** 다 — 그게 표현 대 용량을 가르는 실험이다.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-26_14-04-56_tannorm_stock`
- 시작: 2026-08-26 14:04 · GPU0 (전용) · 60000 iter · 스톡 플랜트

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name tannorm_stock \
  env.use_pace_params=false env.joint_pos_tan_norm=true
```

### 설정 반영 확인 (`params/env.yaml`)

| 키 | 값 |
|---|---|
| `joint_pos_tan_norm` | **true** |
| `observation_space` | **102** (env 가 다시 계산한 값이 그대로 덤프됨) |
| `amp_observation_space` | 49 (**불변** — disc 는 이 arm 의 변수가 아니다) |
| `agent.amp.task_reward_lerp` · `vel_err_scale` · `dr.push_robot` | 0.5 · 0.5 · true (baseline) |
| `agent.algorithm.schedule` | adaptive (**baseline** — 상수 LR 은 별개 축이라 겹치지 않는다) |

## 판정 규칙

★ **40k 이후 점으로만**. 재현 산포 ±8~15%p / ±0.02 m/s.
★★ 속도와 **보행 종류**(`logs/gait_phase.py`)를 같이 본다. style 0.5 arm 은 지금까지 전부 trot 을
못 벗어났다(`velscale1`·`nopush`·`vel1+nopush`·`schedfixed` 4/4). 이 arm 의 핵심 질문도
**`cmd 2.5` 부근 trot → 비대칭 전이 여부**다. 전이 없이 속도만 오르면 같은 결말이다.

램프는 run 의 `params/env.yaml` 에서 이 플래그를 읽어 자동으로 맞춘다
(`speed_ramp_record.py` — 어긋나면 actor 1 층 크기가 달라 `load_state_dict` 가 그 자리에서 터지므로
조용히 틀릴 위험이 없어 멈춰 세우지 않고 맞춘다).

## 비교 대상 (전부 lerp 0.5 · 스톡 · 40k 이후 4 점)

| arm | cmd 2.5 | cmd 3.0 | cmd 3.5 | cmd 4.0 |
|---|---:|---:|---:|---:|
| baseline | 1.598(97) | 1.686(58) | 0.592(0) | 0.049(0) |
| vel_err_scale 1.0 | 1.896(89) | 1.843(58) | 0.277(0) | 0.213(0) |
| vel1 + nopush | 1.896(100) | 1.914(64) | 0.285(0) | 0.214(0) |
| const LR | 1.586(95) | 1.673(95) | 1.626(9) | 0.131(0) |
| *lerp 0.8 (참고선)* | *2.404(98)* | *2.908(98)* | *3.370(98)* | *3.937(97)* |

## 결과 요약

학습 진행 중 — 램프 미측정.

초기 지표 (iter 14): ep_len 22 → **136**, mean_reward 20 → **89**, surrogate 0.055 → 0.028.
`Loss/learning_rate` 1e-5 는 baseline 도 초반·1k 에서 같은 값이라 이상 신호가 아니다.
