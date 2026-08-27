# 2026-08-27_10-54-18_ampTanNorm_stock

한 줄 목적: **discriminator** 관측의 관절 각도를 raw 라디안 12 → tan-norm 72 로 바꾼다.
정책 관측은 raw 그대로. 나머지 전부 baseline. **변수 하나.**

## 왜 disc 인가 — 정책 쪽 노브가 6/6 실패했다

| arm | 바꾼 것 | `cmd 3.5` | 보행 |
|---|---|---:|---|
| baseline | — | 0% | trot |
| vel_err_scale 1.0 | 보상 스케일 | 0% | trot |
| push off | 외란 | 0% | trot |
| vel1 + nopush | 둘 다 | 0% | trot |
| const LR | 옵티마이저 스케줄 | 9% | trot |
| tan-norm (policy) | 관측 표현 | 0% | trot |

서로 다른 축을 건드린 여섯 arm 이 전부 같은 자리로 수렴한다 — 산포가 아니라 구조다.
반면 `lerp` 를 올려 **style 을 약화**하면 한 번에 열린다(`cmd 4.0` 97%, bound 로 전이).
→ 벽을 만드는 것이 style 신호라면, **그 신호를 만드는 disc** 를 봐야 한다.

그리고 §17-c 에서 disc 관측을 성분까지 분해하니 차이가 한 곳에 몰려 있었다:

| 성분 | 우리 | MimicKit |
|---|---:|---:|
| **관절 각도** | **12** (raw rad) | **96** (tan-norm, 16 관절) |
| root 위치 | 높이 1 | 참조 기준 상대 xy 2 |
| root 회전 | tan-norm 6 | tan-norm 6 |
| key body(발) | 12 | 12 |
| root 선속도 / 각속도 | 3 / 3 | 3 / 3 |
| 관절 속도 | 12 | 12 |
| 합 | **49** | **134** |

`49 + 84 + 1 = 134` 로 정확히 닫힌다 — **차이 85 중 84 가 관절 각도 표현 하나**다.

## 구현 — `amp_joint_tan_norm` (기본 False)

인코딩 식은 `joint_pos_tan_norm` 과 같다(`_joint_tan_norm` **함수 하나를 공유**한다).
policy 와 disc 가 따로 구현되면 언젠가 갈라지는데, disc 는 정책 obs 와 달리 **expert(참조 모션)
쪽과도 대칭**이어야 해서 그 사고가 치명적이다.

★ **live · terminal · expert 세 경로가 전부 `_compute_amp_obs` 하나를 지난다.** 인코딩을 그 함수
안에서만 갈라지게 두면 정책과 expert 를 다른 자로 재는 일이 구조적으로 생기지 않는다.
(`_compute_amp_obs` 는 jit 함수라 `Optional` 을 못 받는다 → **빈 텐서**를 off 신호로 쓴다.)

`_AMP_BASE_DIM` 은 상수 43 이 아니라 `cfg.amp_observation_space − 6` 으로 유도한다 — 두 상수가
갈라지지 않게. `cfg.amp_observation_space` 자체는 env `__init__` 이 `super().__init__` **전에**
다시 계산한다(hydra 오버라이드가 확정되는 시점).

θ 는 **절대각**을 쓴다. policy 쪽(상대각)과 다르지만, disc 는 정책·expert 양쪽이 같은 규약이면
되고 참조 모션이 절대각을 주므로 이쪽이 자연스럽다. MimicKit 도 절대각이다.

### 학습 전 실측 검증

**인코더 단위 테스트** — 로드리게스 구현이 쿼터니언 경로와 같은 값인가:

| 항목 | 결과 |
|---|---|
| `quat_from_angle_axis` + `quat_apply` 대비 최대 절대오차 | **2.4e-7** |
| `\|tan\|² + \|norm\|²` | **2.000000~2.000000** |
| 값 범위 (θ ±4 rad, 관절 범위 밖까지) | **[−1.000, 1.000]** |
| 상수 차원 | 32 / 72 |

★ 정책 tan-norm arm 은 쿼터니언 경로로 돌았는데, 공유 함수로 묶은 뒤에도 2.4e-7 일치라
**그 arm 의 결과는 이 리팩터로 바뀌지 않는다.**

**env 스모크** — 두 경로 폭이 같은가, 값이 유한한가:

| | `amp_joint_tan_norm=False` | `=True` |
|---|---|---|
| per-step disc obs | 49 | **109** |
| 총 disc 입력 (×10) | 490 | **1090** |
| 내부 버퍼 shape | (N, 10, 43) | **(N, 10, 103)** |
| expert 샘플 shape | (64, 490) | **(64, 1090)** ← 총 차원과 일치 |
| expert finite | True | **True** |
| expert 값 범위 | [−353.141, 41.574] | **[−9.628, 8.378]** |
| live 버퍼 finite | True | True |

★ off 경로가 49/490/43 으로 **완전히 동일**하다 — 기존 arm 에 회귀 없음.

⚠ raw 쪽 `−353` 은 이 arm 의 문제가 아니라 **참조 데이터 결함**이다 → §17-d, 아래 "따로 발견한 것".

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-27_10-54-18_ampTanNorm_stock`
- 시작: 2026-08-27 10:54 · GPU0 (전용) · 60000 iter · 스톡 플랜트

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name ampTanNorm_stock \
  env.use_pace_params=false env.amp_joint_tan_norm=true
```

### 설정 반영 확인 (`params/env.yaml`)

| 키 | 값 |
|---|---|
| `amp_joint_tan_norm` | **true** |
| `amp_observation_space` | **109** (env 가 다시 계산한 값이 그대로 덤프됨) |
| `joint_pos_tan_norm` / `observation_space` | **false / 42** (정책은 raw — 변수 하나) |
| `agent.amp.task_reward_lerp` · `vel_err_scale` · `dr.push_robot` | 0.5 · 0.5 · true (baseline) |
| `agent.algorithm.schedule` | adaptive (baseline) |

## 판정 규칙

★ **40k 이후 점으로만**. 재현 산포 ±8~15%p / ±0.02 m/s. 우열은 **같은 iter 끼리** 비교할 것 —
요약표의 `best` 는 점 개수가 다르면 4 점 arm 에 유리하다.
★★ 핵심 질문은 속도가 아니라 **`cmd 2.5` 부근 trot → 비대칭 전이 여부**다(`logs/gait_phase.py`).
style 0.5 arm 이 지금까지 6/6 으로 trot 에 갇혔다.

⚠ `amp_reward` **절대값을 이전 arm 과 직접 비교하지 말 것** — disc 입력 차원이 490 → 1090 이라
판별기 자체가 다른 함수다. 비교는 램프와 위상차로 한다.

## 비교 대상 (전부 lerp 0.5 · 스톡 · 40k 이후 4 점)

| arm | cmd 2.5 | cmd 3.0 | cmd 3.5 | cmd 4.0 |
|---|---:|---:|---:|---:|
| baseline | 1.598(97) | 1.686(58) | 0.592(0) | 0.049(0) |
| vel1 + nopush | 1.896(100) | 1.914(64) | 0.285(0) | 0.214(0) |
| const LR | 1.586(95) | 1.673(95) | 1.626(9) | 0.131(0) |
| tan-norm (policy, 1 점) | 1.379(86) | 1.306(0) | 0.395(0) | 0.125(0) |
| *lerp 0.8 (참고선)* | *2.404(98)* | *2.908(98)* | *3.370(98)* | *3.937(97)* |

## ⚠ 따로 발견한 것 — 참조 모션 각속도 yaw 랩 결함

이 arm 의 스모크에서 raw expert 값 범위가 `[−353, 41]` 로 나와 추적했다.

```
motion_lib.py:445   euler_rates = _finite_diff(root_euler, dt)   # yaw 를 unwrap 하지 않는다
```

`go2_walk_turn` 과 그 mirror 에 yaw 랩이 1 회씩 있고, 그 프레임에서 `Δyaw ≈ 2π` 가 그대로 미분돼
각속도가 **373.4 rad/s** 로 튄다(실제 2.26). `2π × 60 fps = 377.0` 과 일치한다.
`ppo_amp.py:187` 이 `update_normalization(cat([expert, policy]))` 라 이 값이 정규화 통계에 들어간다:

```
                  std        wx       wy        wz     |max|
현재 코드                 0.526    1.572    10.280     373.4
np.unwrap 적용            0.586    1.006     0.544      13.6
팽창률                    0.90x    1.56x    18.9x
```

→ **disc 가 보는 expert yaw rate 가 실제의 5% 로 눌린다.** 2628 프레임 중 2 개(0.08%)의 결과다.

⚠ **이 arm 에서는 고치지 않는다.** 이전 arm 들과 같은 expert 분포를 써야 단일 변수 비교가 성립한다.
`np.unwrap` 적용은 **별도 arm** 으로 돌릴 것. 천장의 원인일 가능성은 낮다(눌리는 것이 주로 yaw 인데
고속 전이는 pitch 축 문제이고, 같은 데이터로 lerp 0.8 은 잘 열린다).

## 결과 요약

학습 진행 중 — 램프 미측정.
