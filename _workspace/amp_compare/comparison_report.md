# IsaacLab `go2_imitation` vs MimicKit `amp_go2_track` 비교·진단 보고서

> **작성**: team `go2-amp-compare` lead가 worker-isaac / worker-mimickit / worker-theory 산출물 + 추가 verify를 통합.
> **목표**: IsaacLab에서 학습이 잘 안 되는 이유를 *확정*이 아니라 *검증 가능한 후보들(ranked)*로 제시.
> **참조 산출물**:
> - `_workspace/amp_compare/isaac_side.md` (worker-isaac)
> - `_workspace/amp_compare/mimickit_side.md` (worker-mimickit)
> - `_workspace/amp_compare/theory_notes.md` (worker-theory, 14항목 체크리스트 C1~C14)

---

## 0. 한 줄 요약

두 구현 모두 **AMP 골격(LSGAN/BCE disc + zero-centered GP + replay + 0.5 task / 0.5 style + RSI + early termination)** 자체는 갖췄다. 차이는 **(a) policy가 motion에 도달할 수 있는가**, **(b) discriminator가 walk/trot/run gait를 식별할 수 있는가**, **(c) task 명령이 motion data가 표현할 수 있는 범위 안인가** 세 축에서 IsaacLab 쪽이 모두 불리하게 셋팅돼 있다. 가장 유력한 후보는 **action 가용 범위 부족 (clip_actions=1.0 × action_scale=0.25)**, **task 명령 범위 초과 (360° tar_dir vs forward-only)**, **AMP disc window H=2 (MimicKit는 10)**.

---

## 1. 공통 구조 (일치 항목 — 정상 동작에 무관)

| 항목 | IsaacLab | MimicKit |
|---|---|---|
| Algorithm 골격 | PPOAMPBase + ActorCritic + AMPDiscriminator (rsl_rl) | AMPAgent (PPOAgent) + AMPModel (mimickit) |
| Engine | Isaac Lab | Isaac Lab (engine adapter) |
| Sim dt / Decimation / Control freq | 0.005s / 4 / 50 Hz | 0.005s / 4 / 50 Hz |
| Episode length | 10s (500 step) | 10s |
| num_envs default | 4096 | 4096 |
| Reference State Initialization | ✓ (motion에서 random phase) | ✓ (motion에서 random phase) |
| Demo data | `imitation/smr_mirror_pkl/` 18 PKL (run/trot/walk × mirror) | `data/datasets/dataset_go2_locomotion_smr_mirror3.yaml` 18 motion (run/trot/walk, pace weight=0) |
| Disc input에 task/phase feature | 없음 ✓ (theory C2 ✓) | 없음 ✓ (C2 ✓) |
| Disc output | linear logit | linear logit |
| GP coefficient | 5.0 양측 (expert+policy) | 5.0 양측 (demo+agent) |
| Replay buffer size | 200,000 | 200,000 |
| Discount γ / GAE λ | 0.99 / 0.95 | 0.99 / 0.95 |
| Task reward formulation | velocity 추종 exp(-err²) | velocity 추종 exp(-err²) |
| Task vs Style mix (end state) | 0.5 / 0.5 | 0.5 / 0.5 |
| Obs/Disc-obs normalization | EmpiricalNormalization | Normalizer(clip=10) |

→ **이 표의 항목들은 두 구현이 거의 동일하므로 학습 격차의 원인이 될 수 없다.** 진단은 아래 §2의 차이에서만 일어난다.

---

## 2. 결정적 차이 표 (학습 격차의 후보군)

| # | 영역 | IsaacLab | MimicKit | 차이의 의미 |
|---|------|----------|----------|-------------|
| D1 | **Action 의미** | `0.25 · a + default_qpos` (default offset, RL standard) | 절대 joint position target (`mid + scale·a`, motion frame과 동도메인) | IsaacLab은 default 기준 추가 변위. motion data가 절대 좌표라 RSI 후 차분 필요 |
| D2 | **Action 범위 (effective)** | **default ± 0.25 rad ≈ ±14.3°** (clip_actions=1.0 × action_scale=0.25) | 각 joint range mid ± 1.4 × halfrange (예: thigh `[-2.58, 4.50]`) | Go2 trot/run의 thigh 진폭이 0.25rad 넘으면 **도달 불가** |
| D3 | **Task 명령 분포** | `tar_dir ∈ [-π, π]` 전방향 + `tar_speed ∈ [0.5, 3.0]` + `face_dir = tar_dir` | `lin_x ∈ [0, 4]`, `lin_y = 0` (forward only), `ang_yaw ∈ [-1, 1]` | IsaacLab은 lateral·후방 명령 포함. Go2 quadruped는 lateral 동작이 어렵고 demo도 forward+turn 위주 → 명령이 motion capability 밖이면 style reward signal 0 |
| D4 | **AMP disc window (`num_amp_observations`)** | **2** (코드 값) / 주석은 "10" | **10** (`num_disc_obs_steps: 10`) | H=2는 gait cycle(0.2s @ 50Hz = 10 frame) 식별 불가. theory C14 위반 |
| D5 | **AMP disc per-step feature** | dof_pos(12) + dof_vel(12) + **root_h(1)** + root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) = **43** | root_xy_rel(2) + **root_rot_tan_norm(6)** + joint_rot_tan_norm(72) + key_pos(12) + root_vel(3) + root_ang_vel(3) + dof_vel(12) = **110** | IsaacLab은 **root orientation을 disc가 못 봄** (z-height만 있음). 기울어진 trot과 똑바른 trot이 disc 입력상 동일. theory C1 부분 위반 |
| D6 | **AMP disc 내 within-window 변위** | 없음 (H=2 + root_xy_rel 없음) | `root_xy_rel`로 window 내 다른 step들이 last-frame root 기준 상대 좌표 | "이번 0.2s 동안 robot이 얼마나 이동했나"가 disc 입력. IsaacLab은 이 정보 없음 |
| D7 | **AMP disc total input** | **86** (43 × 2) — cfg/문서는 430 주장하나 runner가 env 값으로 override (on_policy_runner_amp.py:262-264) | 1100 (110 × 10) | 명백한 capacity 격차이나 차원만 키운다고 해결 X. D4, D5, D6 함께 봐야 |
| D8 | **Style ↔ Task lerp warm-up** | `start=0.5, end=0.5` → **anneal 사실상 비활성** (cfg docstring은 "1.0→0.5 anneal" 주장) | 0.5 fixed | IsaacLab cfg 자체가 일관되긴 함 (둘 다 0.5). 다만 *의도된* warm-up이 broken → cold-start disc score 0 상태에서 즉시 50%로 진입 |
| D9 | **Hip-scale reduction** | 없음 | hip joint 명령에 ×0.5 (engine.py:220-222) | Go2 yaw-drift 억제용 표준 trick. 없으면 정책이 hip을 과하게 휘둘러 학습 불안정 |
| D10 | **Early termination 엄격도** | base contact > **500 N** + base_h < 0.15 + roll/pitch > 70° | **non-foot body가 ground와 접촉 시 즉시 종료** | IsaacLab은 thigh/calf scraping이 살아남음 → 그 noisy state가 replay에 쌓여 disc가 "scraping = policy"를 학습 → 정상 motion까지 음의 score 가능 (theory C8 부분 위반) |
| D11 | **Domain randomization** | 없음 | friction[0.4,1.2], base mass+[-0.5,5], CoM, kp/kd scale | sim-to-real 외 학습 자체엔 영향 작음 |
| D12 | **Policy 출력 noise std** | learnable, init 0.25 | **FIXED 0.1** + ent_coef=0 | IsaacLab은 더 큰 초기 exploration. 통상 학습에 도움이나 disc-style 환경에선 양날 |
| D13 | **PPO entropy_coef** | 0.007 | 0.0 (action_std fixed로 대체) | learnable std에서 ent_coef 양수는 std 폭주 방지. 그러나 값 자체는 보통 |
| D14 | **Critic obs** | policy obs와 동일 (`obs_groups: critic=[policy]`) | 동일 | 차이 없음 |
| D15 | **PPO optimizer** | Adam, lr 2e-4 adaptive (KL 0.01) | SGD, actor 2e-4 / critic 1e-4 fixed | 학습 안정성 차이 가능. Adam이 보통 우월하므로 IsaacLab 불리한 점 아님 |
| D16 | **disc_loss / style reward form** | LSGAN: `MSE(D,±1)`, `r_s = clamp(1−0.25(D−1)², 0)·2` | BCE: `BCE(logit,{0,1})`, `r_s = −log(1−σ(D))·2` | 둘 다 유효 AMP 변형. theory §1.9 노트 |

> *위 표의 사실들은 모두 산출물의 file:line 인용에 근거. 자세한 라인은 `isaac_side.md`/`mimickit_side.md` 참조.*

---

## 3. Ranked Root-Cause Candidates (학습 실패의 가장 유력한 후보)

> **본 ranking은 단정이 아니다.** 각 후보는 (a) 명확한 코드 evidence, (b) 신뢰도, (c) **검증용 ablation**을 함께 명시.
> 사용자 메모리 규칙(`feedback_dont_assert_unverified_bugs.md`)에 따라 한 가지를 *원인이다*라고 못 박지 않는다.

### 🔴 Tier 1 — Critical (학습이 *원천적으로* 망가질 수 있는 후보)

#### R1. Action 가용 범위가 motion 진폭보다 좁다 (clip_actions=1.0 + action_scale=0.25)

- **Evidence**:
  - `vecenv_wrapper.py:163-164`: `actions = torch.clamp(actions, -clip_actions, clip_actions)` (env로 들어가기 전에 ±1.0 clip)
  - `go2_imitation_env.py:160`: `processed_actions = 0.25 * actions + default_joint_pos`
  - `rsl_rl_ppo_cfg.py:34`: `clip_actions: float = 1.0`
  - 결과: 최종 joint target ∈ `default_qpos ± 0.25 rad ≈ default ± 14.3°`
  - Go2 default(`unitree.py:159-167`): hip 0±0.1, F-thigh 0.8, R-thigh 1.0, calf -1.5
  - 비교: MimicKit thigh action range `[-2.58, 4.50]` → **약 4 rad** 폭.
  - Go2 trot/run motion의 thigh/calf 진폭은 통상 ±0.3 ~ ±0.5 rad → IsaacLab의 ±0.25 rad는 **부족할 가능성이 매우 큼**.
- **위반 시 증상**: policy가 RSI 직후 motion이 명시하는 다음 joint pose에 도달하지 못함 → discriminator가 demo와 policy를 trivially 분리 → style reward 항상 낮음 → 학습 정체.
- **신뢰도**: **High**. action clip 적용 위치는 verified, motion 통계는 미실측이나 일반 quadruped trot/run 진폭으로 보아 plausible.
- **검증**:
  ```python
  # rsl_rl_ppo_cfg.py
  clip_actions: float = 4.0       # 또는 action_scale을 0.5/1.0으로 올림
  # 또는 env_cfg.action_scale = 0.5
  ```
  500 iter smoke + episode return + amp_reward 추이 비교. action saturation 비율 (|action|=1인 비율) 로깅 추가.

#### R2. Task 명령 분포가 motion data capability를 초과 (360° tar_dir + lateral 포함)

- **Evidence**:
  - `go2_imitation_env.py:467-470`: `θ = uniform(-π, π); tar_dir = [cos θ, sin θ]` → **전방향 360° 명령**
  - `go2_imitation_env.py:479`: `face_dir = tar_dir` (헤딩까지 임의 방향)
  - `env_cfg.py:75-76`: `tar_speed_min=0.5, tar_speed_max=3.0`
  - MimicKit (`amp_go2_tracking_env.yaml:28-33`): `lin_x ∈ [0, 4]`, `lin_y ∈ [0, 0]`, `ang_yaw ∈ [-1, 1]` — **forward only + yaw rate**
  - Go2 demo motion(walk/walk_turn/trot/run): 모두 전진+회전 중심. lateral pure side-step motion은 데이터셋에 거의 없음.
- **위반 시 증상**: lateral 또는 후방 명령 시 policy가 demo와 다른 모션(crab walk 같은 임시방편)을 만들어야 함 → disc style reward = 0 → 그 명령 sample들이 학습을 끌어내림. theory C2 그룹의 task-style mismatch 변종.
- **신뢰도**: **High**. 명령 분포와 motion capability 차이는 명백.
- **검증**:
  ```python
  # _resample_steering 수정
  theta = torch.empty(...).uniform_(-math.pi/4, math.pi/4)  # forward ± 45° 만
  # face_dir 도 forward 우선 또는 더 좁은 범위
  ```
  또는 lin_y를 명시적으로 0으로 강제. 200~500 iter smoke 후 baseline vs 비교.

#### R3. AMP discriminator window H=2 (gait cycle 식별 불가)

- **Evidence**:
  - `env_cfg.py:63`: `num_amp_observations: int = 2` (cfg 주석 "10"과 모순)
  - 동일 cfg docstring(`env_cfg.py:42`): "AMP History (num_amp_observations = 10): amp_observation_size = 43 × 10 = 430" — 의도는 10
  - git log: 첫 commit부터 코드는 2였음 (regression이 아니라 **처음부터 의도와 mismatch**)
  - MimicKit: `num_disc_obs_steps: 10` — 0.2s window
  - 50Hz에서 trot 1 사이클 ≈ 8~12 frame → H=2는 한 stride의 끝과 끝만 보고 중간 swing은 못 봄.
- **위반 시 증상**: discriminator가 "이 두 시점의 pose pair가 motion에 있는가"만 평가. gait coordination, swing phase, foot strike sequence 모두 capture 불가 → style signal weak → policy가 정상 보행을 학습할 incentive 부족 (theory C14).
- **신뢰도**: **High**. cfg 자체가 의도와 mismatch라는 강한 메타 시그널.
- **검증**:
  ```python
  # env_cfg.py
  num_amp_observations: int = 10
  # 동시에 R5(rel_track_obs)도 켤지 결정
  ```
  500~1000 iter smoke. amp_reward 평균 추이 + episode length 비교.

---

### 🟠 Tier 2 — High (Tier 1과 결합 시 critical, 단독으론 학습 가능)

#### R4. AMP disc 입력에 root orientation 누락

- **Evidence**:
  - IsaacLab disc per-step feature (`go2_imitation_env.py:597-620`): `dof_pos(12) + dof_vel(12) + root_height(1) + root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12)` = 43
    → **root z만 있고 root orientation(rot)이 없음**.
  - MimicKit (`compute_tar_obs`): root_rot을 last-frame root 기준 local 변환 후 `quat_to_tan_norm` → 6 dims/step
  - AMP 논문 (theory §1.1, C1): root linear/angular velocity in *character local frame* — 이를 위해선 root_rot이 disc에 있거나 obs가 이미 local frame이어야 함.
  - 확인: IsaacLab의 root_lin_vel, root_ang_vel은 *body frame*(`go2_imitation_env.py:597-620`에서 `_robot.data.root_lin_vel_b` 사용). 이건 OK. 그러나 *얼마나 기울어졌는가*를 disc가 못 봄.
- **위반 시 증상**: 동일 dof pattern + 동일 body-frame 속도라면 robot이 똑바로 trot하든 30° 기울어서 trot하든 disc는 구분 못함 → style reward는 자세 안정성을 가르치지 못함.
- **신뢰도**: **Medium-High**.
- **검증**:
  ```python
  # _compute_amp_obs에 root rotation 추가
  # 옵션 a) projected_gravity (3-dim, 이미 _get_observations에서 사용)
  # 옵션 b) MimicKit처럼 tan_norm 6D
  ```
  smoke 500 iter, amp_reward 분포 비교.

#### R5. lerp warm-up 사실상 비활성 (의도된 1.0→0.5 anneal 미작동)

- **Evidence**:
  - `rsl_rl_ppo_cfg.py:75-77`: `task_reward_lerp=0.5`, `task_reward_lerp_start=0.5`, `task_reward_lerp_anneal_iters=5000`
  - `rsl_rl_ppo_cfg.py:7-12` docstring: "Stage 1 (0~5000 iter): task_reward_lerp=**1.0** → 순수 task reward / Stage 2: ... → 50% task + 50% AMP" — 의도는 명백히 1.0 → 0.5
  - git log: 첫 commit부터 start=end=0.5 (한 번도 의도대로 작동한 적 없음).
- **위반 시 증상**: discriminator가 random init된 채로 reward 절반을 담당 → 초기 disc score ≈ 0 → reward 절반이 0 → policy는 task reward만 따라가다 motion에서 멀어진 채로 RSI 분포와 mismatch 누적.
- **신뢰도**: **Medium-High**. 효과는 *delay* 정도(영구 차단은 아님). Tier 2.
- **검증**:
  ```python
  task_reward_lerp_start = 1.0   # 의도대로 복원
  task_reward_lerp = 0.5
  task_reward_lerp_anneal_iters = 5000
  ```
  smoke 1000 iter. cold-start phase에서 episode return 추이 비교.

#### R6. Early termination이 MimicKit보다 너무 관대

- **Evidence**:
  - IsaacLab (`env.py:294-317`, cfg `:88-91`): base_h<0.15 OR projected_gravity_z>0 OR |roll|>70° OR |pitch|>70° OR **base contact >500 N**
  - MimicKit (`deepmimic_env.py:892-898`): non-foot body 어디든 ground contact (force>0.1) → 즉시 FAIL
  - 차이: thigh/calf가 ground에 닿아 미끄러지는 (scraping) state는 IsaacLab에선 살아남고 MimicKit에선 즉시 종료
- **위반 시 증상**: noisy fallen-ish state들이 replay buffer에 누적 → disc가 "scraping pose"를 policy distribution으로 학습 → policy가 회복 시도해도 이미 그 pose 자체가 음의 style reward → 학습 trap (theory C8 부분 위반).
- **신뢰도**: **Medium**. 단독 critical은 아니지만 R1·R3와 결합 시 증폭.
- **검증**:
  ```python
  # _get_dones에 non-foot contact 추가
  non_foot_contact = (contact_forces[..., non_foot_idx].norm(dim=-1) > 1.0).any(-1)
  died = died | non_foot_contact
  ```
  smoke 500 iter, episode length 분포 + amp_reward 평균 비교.

#### R7. Hip-scale reduction 부재 (Go2-specific trick)

- **Evidence**:
  - MimicKit (`amp_go2_tracking_env.yaml:55-56`, `isaac_lab_engine.py:220-222`): hip DOF action × 0.5
  - IsaacLab: 동일 처리 없음
- **위반 시 증상**: Go2가 hip을 과하게 휘둘러 yaw drift / heading 변동 증가 → R2의 face_reward 추정 어려움 + disc input의 root_ang_vel 노이즈 증가.
- **신뢰도**: **Medium**. Go2 quadruped RL 표준 trick. 직접 검증된 성공 사례 다수.
- **검증**: env._pre_physics_step에서 hip joint index의 action에 0.5 곱셈 추가.

---

### 🟡 Tier 3 — Low (튜닝 영역, 단독으로 학습 실패 원인 아님)

| # | 후보 | 근거 |
|---|------|------|
| R8 | Domain randomization 부재 | MimicKit는 friction/mass/CoM/PD 모두 randomize. sim 학습 자체보다 sim-to-real에 영향. |
| R9 | init_noise_std 0.25 learnable (MimicKit 0.1 fixed) | exploration 폭의 차이. learnable이 보통 더 안정적이라 IsaacLab 불리한 점 아님. |
| R10 | LSGAN vs BCE disc loss / 다른 style reward form | 둘 다 valid AMP 변형. theory §1.9 노트. |
| R11 | Optimizer (Adam adaptive KL vs SGD fixed) | Adam이 통상 우월. IsaacLab 불리 X. |
| R12 | obs 차원 / 표현 (50 vs 93, joint_pos vs tan_norm) | 학습 가능 범위 내. critical 아님. |
| R13 | num_amp_observations=2일 때 `amp_observation_space=430` cfg dead code | runner가 env 값(86)으로 override해 실제 작동엔 무해. 다만 cfg drift의 메타 시그널. |
| R14 | hip default ±0.1 등 init pose 미세 차이 | RSI가 즉시 덮으므로 무관. |

---

## 4. 권장 검증 순서 (제일 빠르게 답이 나오는 순서)

> 한 번에 한 가지만 바꿔야 어느 후보가 효과를 냈는지 분리 가능. 사용자 메모리 `feedback_dont_assert_unverified_bugs.md` 정신.

1. **R1 (action 범위)**: `clip_actions=4.0` 또는 `action_scale=0.5`로 변경 → 500 iter smoke. `|action|≥1` 비율 + amp_reward 평균 + episode return 추이 로깅. **가장 cheap한 ablation이면서 가장 큰 변화 가능**.
2. **R3 (disc window H=10)**: `num_amp_observations=10`으로 변경 → 1000 iter smoke. amp_reward 평균 + disc accuracy(있다면) 확인.
3. **R5 (lerp 복원)**: `task_reward_lerp_start=1.0`. 1000 iter smoke로 cold-start phase 비교.
4. **R2 (task 명령 제한)**: `tar_dir`을 forward ± 45°로 좁힘. 500 iter smoke.
5. **R4 (root orientation 추가)**: `_compute_amp_obs`에 `projected_gravity` 또는 root_rot tan_norm 6D 추가. 500 iter smoke.
6. **R6 (non-foot contact termination)**: 추가 → 500 iter smoke.
7. **R7 (hip scale 0.5)**: 추가 → 500 iter smoke.

위 1~3은 거의 cfg 1줄 변경이라 단일 commit으로 빠르게 verify 가능. 4~7은 env.py / disc obs 수정 동반.

---

## 5. 확인된 *비-원인* (drop from suspect list)

- **DirectRLEnv timeout bootstrap 매핑 누락 가설**: `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:172`에서 `extras["time_outs"] = truncated` 확인. PPO timeout bootstrap 정상 작동.
- **AMP disc 입력에 phase/cmd 누락 (theory C2)**: IsaacLab도 disc에는 cmd/phase 안 넣음 (`go2_imitation_env.py:597-620` `_compute_amp_obs` 참조). theory C2 위반 아님.
- **Demo:policy 샘플 비대칭 (theory C9)**: IsaacLab도 1:1 (`on_policy_runner_amp.py:160-169`).
- **Disc obs normalization 누락 (theory C12)**: 양쪽 모두 활성.
- **RSI 미사용 (theory C7)**: 양쪽 모두 활성.
- **AMP disc input 정의의 policy vs demo 일치성**: IsaacLab의 `_compute_amp_obs`가 양 경로에서 동일 함수로 호출됨 (`go2_imitation_env.py:188-195` policy / `:525-532` reference). 일치성 OK.
  - 단 `root_lin_vel`/`foot_pos`의 계산 source가 (sim 측정 vs motion finite-diff + FK)로 다름 → 노이즈 특성 차이는 있으나 정의 차이는 아님.

---

## 6. 메타 관찰 (cfg ↔ docstring drift)

| 항목 | 코드 값 | docstring/주석 | 의도 추정 |
|---|---|---|---|
| `num_amp_observations` | 2 | "= 10, total 430" (env_cfg.py:42) | 10이었어야 함 |
| `task_reward_lerp_start` | 0.5 | "Stage 1: 1.0 → 순수 task" (rsl_rl_ppo_cfg.py:7-12) | 1.0이었어야 함 |
| `amp_observation_space` | 430 (cfg) → runner가 86으로 override | "= 43 × 10 = 430" | 의도 정합 시 430(=43×10), 현재는 86(=43×2) |
| `MOTION_FILES_DIR` | `imitation/smr_mirror_pkl` | "imitation/go2 폴더의 7개 PKL" (env_cfg.py:21 comment) | 한 번 갈아엎음 |
| Policy obs comment | "= 44" | 실제는 50 (b62559d 커밋에서 root_lin_vel+ang_vel 6 추가 시 cfg 값은 50으로 고침, 주석 미수정) | 50이 맞음 |

→ **이 drift들은 "한 번도 의도대로 셋팅되어 학습된 적이 없을 가능성"을 시사**한다. 사용자 메모리 `project_parkour_A_never_trained.md`의 "known-good 시점 없음" 패턴과 유사. 즉 *regression debugging*이 아니라 *fresh tuning*으로 접근해야 함.

---

## 7. 사용자 메모리 적합성

본 분석은 다음 규칙을 준수:

- `feedback_dont_assert_unverified_bugs.md`: 모든 후보를 *확정 버그*가 아니라 *후보 + 신뢰도 + 검증 방법*으로 표시.
- `feedback_torque_envelope_not_tier2_against_empirical.md`: R1(action 범위)는 framework 내부 동작(`vecenv_wrapper.clip_actions`)을 *verified*로 표기, motion 진폭 부분은 *추정*으로 분리.
- `feedback_worker_absence_claim_must_verify.md`: "X 없다" 단정 전 grep 재검증 (R4 root_rot, R7 hip-scale 등 모두 grep 확인 후 기재).

---

## 부록 A: 핵심 수치 한눈에 (양쪽 비교)

| 항목 | IsaacLab | MimicKit |
|---|---|---|
| **Action effective range** | default ± 0.25 rad | mid ± 1.4 × half_range |
| **Task tar_dir 범위** | 360° + speed [0.5, 3.0] | forward only + yaw_rate [-1, 1] |
| **AMP disc input dim** | **86** (43×2) | **1100** (110×10) |
| **AMP disc per-step features** | dof_pos·dof_vel·root_h·root_vel(b)·root_ang_vel(b)·foot_pos_local | root_xy_rel·root_rot_tan·joint_rot_tan·key_pos·root_vel·root_ang_vel·dof_vel |
| **Root orientation in disc** | 없음 (z만) | 6D tan_norm |
| **Within-window 변위** | 없음 | root_xy_rel |
| **Hip scale reduction** | 없음 | 0.5 |
| **Early termination 엄격도** | base 중심 (관대) | non-foot 전체 (엄격) |
| **Domain randomization** | 없음 | 4종 |
| **lerp warm-up** | 비활성 (start=end=0.5) | 0.5 fixed (동일하지만 *의도된 warm-up*이 IsaacLab엔 broken) |
| **disc_loss / style reward** | LSGAN / clip form | BCE / -log(1-σ(D)) |
| **PPO optimizer** | Adam adaptive(KL 0.01) | SGD fixed |
| **action_std** | 0.25 learnable | 0.1 fixed |
| **ent_coef** | 0.007 | 0.0 |

---

## 부록 B: ablation 우선순위 정리 (체크리스트)

- [ ] 1. `clip_actions=4.0` (또는 `action_scale=0.5`) → 500 iter smoke
- [ ] 2. `num_amp_observations=10` → 1000 iter smoke (Tier 1)
- [ ] 3. `task_reward_lerp_start=1.0` → 1000 iter smoke (warm-up 복원)
- [ ] 4. tar_dir 범위를 forward ± 45°로 제한 → 500 iter smoke
- [ ] 5. `_compute_amp_obs`에 projected_gravity(3) 또는 root_rot tan_norm(6) 추가 → 500 iter smoke
- [ ] 6. non-foot body contact termination 추가 → 500 iter smoke
- [ ] 7. hip DOF action × 0.5 → 500 iter smoke

각 단계마다 다음 메트릭 비교 (가능한 경우): `Episode/return_mean`, `Episode_Reward/amp_reward`, `Disc/{agent,demo}_acc`, `Episode/length_mean`, `|action|≥1 비율`.

---

*— team-lead@go2-amp-compare*
