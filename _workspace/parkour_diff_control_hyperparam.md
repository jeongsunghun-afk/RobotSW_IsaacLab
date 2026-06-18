# Parkour A vs B — Control + PPO 하이퍼파라미터 비교

- **A** = `parkour` (Direct RL, 학습 초반 학습 안 됨)
  `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/`
- **B** = `Isaaclab_Parkour` (Manager-based, 학습 잘 됨, extreme_parkour 포팅 원본)
  `Isaaclab_Parkour/`

> 본 문서는 **control 경로**(action→target joint pos 변환, actuator, sim/decimation)와
> **PPO 하이퍼파라미터/알고리즘 구조**만 다룬다. 모든 값은 파일:line 인용. 검증 안 된 추론은 "가설" 표기.
> 금지 규칙 준수: actuator 약화/reward·hyperparam scale 단독을 root cause로 ranking하지 않음 — A/B 값 차이는 사실로만 기록.

---

## 1. Control 타이밍 (sim dt / decimation / policy 주파수)

| 항목 | A | B | 차이 |
|------|---|---|------|
| sim dt | `1/200 = 0.005s` (`parkour_env_cfg.py:362, 365`) | `0.005s` (`parkour_teacher_cfg.py:55`) | 동일 |
| decimation | `4` (`parkour_env_cfg.py:340`) | `4` (`parkour_teacher_cfg.py:51`) | 동일 |
| render_interval | `4` (`:366`) | `= decimation` (`:56`) | 동일 |
| policy(control) dt | `4 × 0.005 = 0.02s` → **50 Hz** | `0.02s` → **50 Hz** | 동일 |
| episode_length_s | `20.0` (`:339`) | `20.0` (`:53`) | 동일 |

→ **시간 축(주파수)은 A/B 완전 동일.**

---

## 2. Action → target joint position 변환식

### A (`parkour_env.py:503-527`, `parkour_env_cfg.py:341-345`)
```
action_scale = 0.25,  clip_actions = 4.8
_actions = clip(raw_action, -4.8, +4.8)
scaled   = _actions.clone()
scaled[:, hip_joint_ids] *= 0.5            #  ★ A 고유: hip 축만 추가 0.5배
_processed_actions = action_scale * scaled + default_joint_pos
set_joint_position_target(_processed_actions)
```

### B (`actions_cfg.py`, `joint_actions.py`, `parkour_mdp_cfg.py:329-340`)
```
scale = 0.25,  clip = (-4.8, +4.8),  use_default_offset = True
raw  = clip(action, -4.8, +4.8)            # action delay teacher에선 OFF (아래 §5)
_processed_actions = raw * 0.25 + _offset  # _offset = default_joint_pos
set_joint_position_target(_processed_actions)
```

| 항목 | A | B | 차이 |
|------|---|---|------|
| action_scale | 0.25 | 0.25 | 동일 |
| raw action clip | ±4.8 (`clip_actions`, `:345`) | ±4.8 (`clip={'.*':(-4.8,4.8)}`) | 동일 |
| default pos offset | `+ default_joint_pos` | `+ _offset(=default_joint_pos)` | 동일 (방식 동일) |
| **hip 추가 스케일** | **`hip *= 0.5`** (`parkour_env.py:516`) | **없음** | ★ **A만 존재** |

→ **유효 action_scale: A hip = 0.25×0.5 = 0.125, thigh/calf = 0.25 / B 전 관절 = 0.25.**
  A의 hip 가동범위가 B 대비 절반. (사실 기록; 원인 단정 아님)

추가 메모: A는 PPO 러너 cfg에도 `clip_actions = 10.0`(`rsl_rl_ppo_cfg.py:21`)이 따로 있어
rsl_rl 래퍼가 ±10.0으로 1차 clip → env가 ±4.8로 2차 clip하는 **이중 clip** 구조.
유효 내부 clip은 ±4.8로 B와 동일하므로 동작상 결과는 같음(가설: 무영향).

---

## 3. Actuator 설정 (PD gain / limit / armature)

> ⚠️ actuator 약화(K/D/saturation/vel/armature) 자체를 root cause로 ranking하지 않음.
> 사용자가 A의 `_actuator_mode=2` 설정으로 이전 학습 성공을 검증함 (MEMORY). 아래는 **값 차이 사실 기록**.

A: stock `DCMotorCfg`, `_actuator_mode = 2` 분기 적용 (`parkour_env_cfg.py:430-450`, `__post_init__ :511-528`)
B: 커스텀 `ParkourDCMotorCfg` (`parkour_actuator_cfg.py`), `default_cfg.py:60-83`에서 `base_legs` 교체

| 파라미터 | A (mode 2) | B | 비고 |
|----------|-----------|---|------|
| stiffness (K) | 25.0 | 40.0 | |
| damping (D) | 0.5 | 1.0 | |
| friction | 0.0 | 0.0 | 동일 |
| effort_limit | hip/thigh/calf = 23.5 / 23.5 / 23.5 | 35 / 40 / 40 | |
| saturation_effort | **23.5 (스칼라, 전 관절 동일)** | **{hip 35, thigh 45, calf 45} (관절별 dict)** | A는 스칼라만 허용(주석 :425) |
| velocity_limit | 30.0 / 30.0 / 30.0 | 52.4 / 30.1 / 30.1 | A는 hip도 30 |
| armature | **0.01 (명시)** (`:528`) | **미지정** (ParkourDCMotorCfg에 armature 전달 안 함 → 기본값) | |
| actuator class | `DCMotorCfg` (stock) | `ParkourDCMotorCfg` (커스텀) | §3.1 |

### 3.1 Actuator 토크 계산식 차이
- A `DCMotorCfg`/stock `DCMotor`: 속도 의존 토크 saturation 적용 — `max_effort = saturation_effort·(1 − vel/vel_limit)`, `effort_limit`로 clip.
- B `ParkourDCMotor` (`parkour_actuator_pd.py:46-71`): `IdealPDActuator` 상속, `compute()`에서
  `effort = K·error_pos + D·error_vel + joint_efforts` 계산 후 `_clip_effort()`에서
  **A와 동일한 형태**의 속도 의존 saturation(`saturation_effort·(1−vel/vel_limit)`, `effort_limit` clip) 적용.
- → **토크 saturation 메커니즘 형태는 A/B 동일.** 차이는 위 표의 파라미터 값과,
  A=스칼라 saturation_effort / B=관절별 dict 라는 점.

### 3.2 기타 robot cfg
| 항목 | A | B | 차이 |
|------|---|---|------|
| base USD | `UNITREE_GO2_CFG` | `UNITREE_GO2_CFG` | 동일 |
| enabled_self_collisions | 미설정 (stock 기본값) | **`True` 명시** (`default_cfg.py:59`) | B만 명시 |
| reset joint pos | `default_joint_pos` 그대로 (`parkour_env.py:1180`) | `reset_joints_by_scale (0.95,1.05)` (`parkour_mdp_cfg.py:264-270`) | B는 reset 시 관절 스케일 노이즈 (DR worker #4 영역) |

---

## 4. PPO 하이퍼파라미터

A: `agents/rsl_rl_ppo_cfg.py`  /  B: `agents/rsl_teacher_ppo_cfg.py` + `parkour_rl_cfg.py`

| 파라미터 | A | B | 차이 |
|----------|---|---|------|
| num_steps_per_env (rollout) | 24 | 24 | 동일 |
| max_iterations | 50000 | 50000 | 동일 |
| save_interval | 100 | 100 | 동일 |
| learning_rate | 2.0e-4 | 2.0e-4 | 동일 |
| schedule | adaptive | adaptive | 동일 |
| desired_kl | 0.01 | 0.01 | 동일 |
| num_learning_epochs | 5 | 5 | 동일 |
| num_mini_batches | 4 | 4 | 동일 |
| gamma | 0.99 | 0.99 | 동일 |
| lam | 0.95 | 0.95 | 동일 |
| clip_param | 0.2 | 0.2 | 동일 |
| entropy_coef | 0.01 | 0.01 | 동일 |
| value_loss_coef | 1.0 | 1.0 | 동일 |
| use_clipped_value_loss | True | True | 동일 |
| max_grad_norm | 1.0 | 1.0 | 동일 |
| empirical_normalization | False | False | 동일 |
| actor/critic obs normalization | False / False | (cfg 미지정 → 기본 False) | 동일 |
| **num_envs** | **4096** (`parkour_env_cfg.py:413`) | **6144** (`parkour_teacher_cfg.py:35`) | ★ 차이 |
| dagger_update_freq | cfg 미지정 → 러너 코드 하드코딩 `it%20` (`on_policy_runner_parkour.py:93`) | 20 (cfg 명시) | 실효 동일(20) |
| priv_reg_coef_schedual | cfg 미지정 → 코드 기본 `[0,0.1,2000,3000]` (`ppo_parkour.py:124`) | `[0.0,0.1,2000,3000]` (명시) | 동일 |
| estimator hidden_dims | [128, 64] | [128, 64] | 동일 |
| **estimator learning_rate** | **1.0e-3** (`rsl_rl_ppo_cfg.py:54`) | **1.0e-4** (`parkour_rl_cfg.py:50`) | ★ 10배 차이 |
| estimator train_with_estimated_states | True | True | 동일 |

→ **순수 PPO 하이퍼파라미터(lr/clip/epoch/batch/gamma/lam/kl/entropy/grad_norm)는 A=B 완전 동일.**
  control 외 차이는 (1) num_envs 4096 vs 6144, (2) estimator lr 1e-3 vs 1e-4 두 개뿐.

---

## 5. Action delay / filter / smoothing

| 항목 | A | B |
|------|---|---|
| action delay 인프라 | 없음 | `DelayedJointPositionAction` (`joint_actions.py`) — history buffer + delay step |
| **teacher 학습 시 delay** | 해당없음 (delay 미구현) | **OFF** — `parkour_teacher_cfg.py:62-63`: `use_delay=False`, `history_length=1` |
| EMA / low-pass filter | 없음 | 없음 |
| action_rate 패널티 | reward로 처리 (`action_rate_l2 = -0.05`) | reward로 처리 (`reward_action_rate = -0.1`) |

→ B의 action delay는 student/배포용 인프라이며 **teacher 학습에선 꺼져 있음.**
  따라서 control 경로상 A와 B teacher는 **둘 다 delay 없음 = 차이 없음.**
  (action_rate weight 차이 -0.05 vs -0.1은 reward worker #1 영역)

---

## 6. 학습 알고리즘 구조

| 항목 | A | B | 차이 |
|------|---|---|------|
| 알고리즘 클래스 | `PPOParkour` (`ppo_parkour.py`) | `PPOWithExtractor` (`parkour_rl_cfg.py:67`) | 이름 다름, 구조 동등(가설) |
| 러너 클래스 | `OnPolicyRunnerParkour` | (B 자체 rsl_rl runner) | 둘 다 dagger update 오케스트레이션 |
| Actor/Critic | `ActorCriticRMA` | `ActorCriticRMA` | 동일 |
| RMA priv encoder (adaptation) | 있음 | 있음 | 동일 |
| StateHistoryEncoder (history) | 있음 (`actor_critic_parkour.py:21`) | 있음 (`ParkourRslRlStateHistEncoderCfg`) | 동일 |
| scan encoder | 있음 | 있음 | 동일 |
| estimator (base_lin_vel 예측) | `DefaultEstimator` 류, hidden [128,64] | `DefaultEstimator`, hidden [128,64] | 동일 |
| priv_reg loss scheduling | 있음 (`[0,0.1,2000,3000]`) | 있음 (`[0,0.1,2000,3000]`) | 동일 |
| dagger / history-encoder 업데이트 | `update_dagger()`, `it%20` | `dagger_update_freq=20` | 동일 |
| distillation(student/depth) | teacher cfg에선 미사용 | teacher cfg에선 미사용 (student/depth 별도) | 동일 |

→ **A와 B teacher는 동일한 RMA-style PPO 파이프라인** (ActorCritic + scan/priv encoder + history encoder + estimator + priv_reg).
  구조적 알고리즘 차이는 발견되지 않음. 클래스 이름만 다름.

### 6.1 Network 구조 (개략)
| 항목 | A | B | 차이 |
|------|---|---|------|
| actor_hidden_dims | [512, 256, 128] | [512, 256, 128] | 동일 |
| critic_hidden_dims | [512, 256, 128] | [512, 256, 128] | 동일 |
| scan_encoder_dims | cfg 미지정 → `ActorCriticRMA` 기본 `[128,64,32]` (`actor_critic_parkour.py:93`) | [128, 64, 32] (명시) | **실효 동일** |
| priv_encoder_dims | cfg 미지정 → 기본 `[64,20]` (`:94`) | [64, 20] (명시) | **실효 동일** |
| activation | elu | elu | 동일 |
| init_noise_std | 1.0 | 1.0 | 동일 |
| RNN 사용 | 없음 (`is_recurrent=False`) | 없음 | 동일 |

→ A의 cfg가 `scan_encoder_dims`/`priv_encoder_dims`를 명시하지 않지만,
  `ActorCriticRMA` 코드 기본값이 B 명시값과 정확히 일치 → **네트워크 shape는 A=B 동일.**

---

## 7. 기타 control 관련 환경 차이

| 항목 | A | B | 비고 |
|------|---|---|------|
| sim physics_material friction_combine | `multiply` (`parkour_env_cfg.py:369`) | `average` (`default_cfg.py:48`) | 유효 마찰 계수 산출식 차이 |
| terrain physics_material friction_combine | `multiply` (`:391`) | `average` (`default_cfg.py:48`) | 동상 |
| static/dynamic friction | 1.0 / 1.0 | 1.0 / 1.0 | 값 자체는 동일 |

가설: `multiply` vs `average` combine mode는 robot–terrain 접촉 시 유효 마찰을 다르게 산출한다.
robot foot 마찰이 startup DR로 랜덤화될 때(A: 0.4–1.5, B: 0.6–2.0) combine mode가 다르면
실제 접지 마찰 분포가 달라짐. (사실 기록 — 단정 아님)

---

## 8. 학습 초반 영향 가설 (우선순위)

> 아래는 **control/하이퍼파라미터 관점의 차이 목록**이며, root cause 단정/RCA ranking이 아니다.
> "학습 초반(early training)" 동작에 영향을 줄 수 있는 정도 순으로 정렬한 **가설**.

**[가설 1] A의 hip 관절 추가 `×0.5` 스케일 (A만 존재)**
- `parkour_env.py:516` — A는 hip action을 0.5배 추가 축소 → 유효 hip action_scale = 0.125 (B = 0.25).
- 학습 초반 hip 가동범위가 B 대비 절반 → 자세 교정/측면 균형/장애물 회피 동작 폭이 제한될 수 있음.
- B teacher에는 이 스케일이 전혀 없음. **A/B control 변환식의 가장 명확한 단일 차이.**
- 검증 제안: A에서 hip 0.5 스케일 제거 후 학습 초반 추적/넘어짐 추이 비교 (worker가 아닌 사용자 결정).

**[가설 2] sim/terrain friction_combine_mode (`multiply` A vs `average` B)**
- combine mode가 다르면 동일 `static_friction=1.0`이라도 robot–terrain 접지 유효 마찰이 달라짐.
- 학습 초반 접지/추진력에 영향 가능. friction DR 범위까지 다르므로(§7) 분포 자체가 어긋남.

**[가설 3] estimator learning_rate 10배 차이 (A 1e-3 vs B 1e-4)**
- estimator는 proprio history로 base_lin_vel을 예측하고 `train_with_estimated_states=True`로
  그 추정값이 actor 입력에 들어감. lr 10배가 크면 학습 초반 estimator 추정이 불안정/발산해
  actor 입력 노이즈가 커질 수 있음 (가설). B는 1e-4로 보수적.

**[가설 4] num_envs 4096(A) vs 6144(B)**
- 동일 num_steps_per_env=24에서 배치 크기 = num_envs×24. A 배치가 약 33% 작음.
- gradient 추정 분산이 다소 커질 수 있으나, 둘 다 충분히 큰 배치라 영향은 작을 것으로 추정(가설).

**[참고: 원인으로 단정하지 않는 차이 — 사실만 기록]**
- Actuator 파라미터 차이(K/D/effort/saturation/vel/armature, §3): A `_actuator_mode=2`는
  사용자가 이전 학습 성공을 검증한 설정 → root cause ranking에서 제외.
- reward weight 차이(action_rate −0.05 vs −0.1 등)는 reward worker(#1) 영역.

**[control/PPO 관점 결론]**
- 순수 PPO 하이퍼파라미터와 알고리즘 파이프라인 구조는 **A와 B teacher가 사실상 동일**하다.
- control 경로에서 **A에만 존재하는 명확한 차이는 `hip ×0.5` 스케일**이며, 그 외에는
  friction combine mode, estimator lr, num_envs가 차이점. 학습 초반 문제의 단서는
  control 변환식(가설 1) 쪽이 PPO 설정보다 우선 검토 대상으로 보임.
