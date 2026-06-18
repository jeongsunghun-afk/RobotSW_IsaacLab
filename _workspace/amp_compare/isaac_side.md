# IsaacLab `go2_imitation` 분석 보고서

> READ-ONLY. 모든 사실은 실제 파일 라인 인용. 추측은 별도 표기.

---

## 0. 진입점 & 등록

- **Task ID**: `Go2-Imitation-v0` (`source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation/__init__.py:21`)
- **Env entry_point**: `go2_imitation_env:Go2ImitationEnv` (`__init__.py:22`)
- **env_cfg_entry_point**: `go2_imitation_env_cfg:Go2ImitationEnvCfg` (`__init__.py:25`)
- **rsl_rl_cfg_entry_point**: `agents.rsl_rl_ppo_cfg:Go2ImitationPPORunnerCfg` (`__init__.py:26`)
- **학습 진입 스크립트**: `scripts/reinforcement_learning/rsl_rl/train.py` — `class_name == "OnPolicyRunnerAMPBase"`로 분기 (`train.py:223-224`)
- **학습 command (CLAUDE.md:171-173 권장)**:
  ```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Imitation-v0 --num_envs 4096 --headless \
    --logger wandb --wandb-project IsaacLab-locomotion
  ```
- skrl 사용 안 함 (`agents/__init__.py`에 rsl_rl 외 등록 없음, `agents/`에는 `rsl_rl_ppo_cfg.py` 단 1개).

---

## 1. 환경(ENV)

### 1.1 시뮬레이션/타이밍
- `episode_length_s = 10.0` (`go2_imitation_env_cfg.py:51`)
- `sim_dt_hz = 200`, `policy_dt_hz = 50`, `decimation = 200/50 = 4` (`env_cfg.py:54-56`)
- `sim.dt = 1/200 = 0.005s`, `sim.render_interval = decimation` (`env_cfg.py:97-99`)
- 따라서 `step_dt = 0.02s` (50 Hz), `max_episode_length = 500 steps`.
- `scene.num_envs = 4096`, `env_spacing = 5.0`, `replicate_physics=True` (`env_cfg.py:123`)
- 시뮬레이션 physics material: static/dynamic friction = 1.0, restitution = 0.0 (`env_cfg.py:97-107`).
- Terrain: `terrain_type="plane"` (`env_cfg.py:109-121`) — 단, env에서는 `spawn_ground_plane()`로 별도 ground 생성 (`go2_imitation_env.py:132-141`).

### 1.2 로봇 / Actuator
- 로봇: `UNITREE_GO2_CFG` (`env_cfg.py:125`).
- USD: `Unitree/Go2/go2.usd` (`source/isaaclab_assets/.../unitree.py:142`).
- 초기 자세: pos=(0,0,0.27), hip ±0.1, F·R thigh 0.8 / 1.0, calf -1.5 (`unitree.py:159-167`).
- Actuator (단일 `DCMotorCfg`, `unitree.py:170-181`):
  - `effort_limit = 23.5`
  - `saturation_effort = 23.5`
  - `velocity_limit = 30.0`
  - `stiffness = 25.0`
  - `damping = 0.5`
  - `friction = 0.0`
  - `armature = 0.01`
- `soft_joint_pos_limit_factor = 0.9` (`unitree.py:169`).
- Self-collision enabled (`unitree.py:155`).
- Contact sensor: `prim_path="/World/envs/env_.*/Robot/.*"`, `history_length=3`, `update_period=0.005`, `track_air_time=True` (`env_cfg.py:129-134`).

### 1.3 Observation
- `observation_space: int = 44 + 6 = 50` (`env_cfg.py:59`) — cfg 주석은 "44" 라고 표기하지만 실제 값은 **50**.
- `_get_observations` 실제 concat (`go2_imitation_env.py:238-251`):
  | 요소 | dim | 위치 |
  |------|-----|------|
  | `root_lin_vel_b` | 3 | `:240` |
  | `root_ang_vel_b` | 3 | `:241` |
  | `projected_gravity_b` | 3 | `:242` |
  | `local_tar_dir` (heading-relative) | 2 | `:243` |
  | `tar_speed` | 1 | `:244` |
  | `local_face_dir` (heading-relative) | 2 | `:245` |
  | `joint_pos - default_joint_pos` | 12 | `:246` |
  | `joint_vel` | 12 | `:247` |
  | `self.actions` (이전 액션) | 12 | `:248` |
  | **합계** | **50** | |
- **정규화**: actor/critic 모두 `EmpiricalNormalization` ON. `actor_obs_normalization=True`, `critic_obs_normalization=True` (`agents/rsl_rl_ppo_cfg.py:48-49`).
- **Clip**: env 단계의 obs clip 없음. 외부 `clip_actions=1.0`은 액션 측 (`rsl_rl_ppo_cfg.py:34`).
- **Noise**: env에서 obs noise 추가 없음 (`_get_observations`에 noise 코드 부재).
- **History**: policy obs 자체에는 history 없음. AMP disc obs만 history 사용 (1.10 / 2.3 참고).
- **Phase**: policy obs에 motion phase 미포함 (코드상 phase는 reset/disc-only). → 정책은 어느 motion·어느 phase를 추종하는지 알 수 없음. Steering 명령(tar_dir/tar_speed/face_dir)만 입력.

### 1.4 Action
- `action_space: int = 12` (`env_cfg.py:60`).
- `action_scale = 0.25` (`env_cfg.py:94`).
- 처리: `processed_actions = action_scale * actions + default_joint_pos` (`go2_imitation_env.py:160`).
- 적용: `_robot.set_joint_position_target(processed_actions)` (`go2_imitation_env.py:171`) — joint **position target** (P-D 제어, low-level은 DCMotor).
- **Low-pass filter 없음**.
- 액션 클립: 러너 외부 `clip_actions=1.0` (`rsl_rl_ppo_cfg.py:34`). `_pre_physics_step`에는 clip 없음 (`go2_imitation_env.py:158-160`).

### 1.5 Reward
- `tar_reward_w = 0.7`, `face_reward_w = 0.3` (`env_cfg.py:82-83`).
- `vel_err_scale = 0.5` (`env_cfg.py:84`).
- **tar_reward** (`go2_imitation_env.py:256-270`):
  ```
  root_vel_xy = (root_pos_w - prev_root_pos_w)[:, :2] / step_dt
  tar_vel     = tar_speed * tar_dir       # [N,2]
  tar_vel_err = ||tar_vel - root_vel_xy||²
  tar_reward  = exp(-0.5 * tar_vel_err)
  if dot(tar_dir, root_vel_xy) < 0: tar_reward = 0
  ```
- **face_reward** (`:273-278`):
  ```
  heading_rot = yaw-only quat(root_quat)
  char_fwd    = quat_apply(heading_rot, [1,0,0])[:, :2]
  face_reward = max(0, dot(face_dir, char_fwd))   # [0, 1]
  ```
- **최종 task reward** (`:280`):
  ```
  reward = 0.7 * tar_reward + 0.3 * face_reward
  ```
  → clip 없음. **min=0 clip 없음**.
- Imitation pose/vel/key_body 에러 reward 항 **없음** — 모방 신호는 전적으로 AMP discriminator를 통해 들어옴 (즉 reward 구조는 'task(steering) + style(AMP)' 분리형).
- Episode 통계 누적: `_episode_sums["tar_reward"]`, `_episode_sums["face_reward"]` (`:282-283`); reset 시 평균으로 로깅 (`:402-406`).

### 1.6 Termination / Early termination
- `early_termination = True` (`env_cfg.py:87`).
- `_get_dones` (`go2_imitation_env.py:290-324`):
  - `time_out = (episode_length_buf ≥ max_episode_length - 1)` → 500 step.
  - `died` 합집합:
    - `base_height < termination_height (0.15 m)` (`:294-295`, cfg `:88`)
    - `projected_gravity_b[:,2] > 0` (뒤집힘) (`:297-298`)
    - `|roll| > 70°` 또는 `|pitch| > 70°` — gravity에서 atan2로 계산 (`:300-306`, cfg `:90-91`)
    - base contact force > 500 N (contact sensor의 `body_names`에 "base" 키워드 매칭) (`:308-317`, cfg `:89`)
  - **첫 1 step 면제**: `died = died & (episode_length_buf > 1)` (`:320`).
- `_get_dones` 반환: `(died, time_out)` — DirectRLEnv 표준 (terminated, truncated).
- DirectRLEnv 표준에 따라 `extras["time_outs"]`는 자동 채워지며, PPO base가 timeout bootstrap 수행 (`rsl_rl/rsl_rl/algorithms/ppo.py:165-168`). 확인: DirectRLEnv는 `reset_terminated`, `reset_time_outs`를 분리 반환 (`source/isaaclab/isaaclab/envs/direct_rl_env.py:393`, `:420`); wrapper가 extras["time_outs"]에 매핑하는지 직접 라인은 확인 안 됨 (wrapper 파일 비검사).

### 1.7 Reset / Reference State Initialization (RSI)
- `reset_strategy = "random"` (`env_cfg.py:72`) — `"random"` 또는 `"random_start"`. 분기는 `_reset_strategy_rsi`(`go2_imitation_env.py:416-457`).
- **항상 RSI 사용** (default state 없음). `_reset_idx`에서 `_reset_strategy_rsi(env_ids)`만 호출 (`:386`).
- 흐름 (`_reset_strategy_rsi`):
  1. `motion_ids = self._motion_lib.sample_motions(n)` — 길이 비례 가중치 (1.8 참고) (`:419`).
  2. `times = sample_times(motion_ids)` ([0, motion_length] 균등); `"start" in reset_strategy`면 `times=0` (`:421-424`).
  3. `calc_motion_frame(motion_ids, times)` → root_pos/quat/lin_vel/ang_vel/dof_pos/dof_vel/foot_pos (`:426-428`).
  4. `root_state[:, :3] = root_pos + env_origins` (`:432`).
  5. `root_state[:, 3:7] = root_quat` (wxyz).
  6. `root_state[:, 7:10] = quat_apply(root_quat, lin_vel_b)` — body→world.
  7. `root_state[:, 10:13] = quat_apply(root_quat, ang_vel_b)` — body→world.
  8. `joint_pos_out[:, :12] = dof_pos[:, _motion_dof_indices]`, 동일하게 joint_vel.
  9. RSI 시점 AMP 버퍼를 채우기 위해 `pre_shift_times = times - step_dt`로 reference buffer 새로 계산 (`:448-451`) — `_compute_reference_buffers(n, pre_shift_times, motion_ids=motion_ids)`로 동일 motion에 대해 H-step 히스토리 산출.
- **Terminal AMP obs 캡처**: `_reset_idx` 진입 시 RSI 이전의 마지막 obs를 `_terminal_amp_obs[env_ids]`에 저장 (`go2_imitation_env.py:330-379`) → runner의 done-step amp_reward 교체용 (2.5 참고).
- `_prev_root_pos_w[env_ids] = root_state[:, :3]` (RSI 직후 속도 계산 0이 되도록) (`:396`).
- Steering 재샘플링: reset 시점에 동시 수행 (`:393`).

### 1.8 Motion library
- 위치: `imitation/smr_mirror_pkl/` (`env_cfg.py:30` — `MOTION_FILES_DIR`).
  - 주의: `env_cfg.py:29`의 `imitation/go2/` 경로는 **주석 처리**되어 있고 활성 경로는 `smr_mirror_pkl`. CLAUDE.md 문서(line 24, 217)와 불일치.
  - 실제 디렉토리 내용: `go2_run0(_mirror).pkl`, `go2_run1(_mirror).pkl`, `go2_run2(_mirror).pkl`, `go2_trot0(_mirror).pkl`, `go2_trot1(_mirror).pkl`, `go2_walk(_mirror).pkl`, `go2_walk1(_mirror).pkl`, `go2_walk2(_mirror).pkl`, `go2_walk_turn(_mirror).pkl` (총 18 파일, mirror 9 + 원본 9).
- 포맷: 18-col PKL, `data["frames"] (N, 18)`, `data["fps"] = float` (`motion_lib.py:431-434`).
  - `frames[:, 0:3]` = root_pos (world), `[:, 3:6]` = root euler (rpy), `[:, 6:18]` = dof_pos (12).
- 속도/가속도/발 위치 자동 계산:
  - root_quat: ZYX Euler→wxyz (`_euler_to_quat_wxyz`, `motion_lib.py:141-150`).
  - linear vel (body): `_finite_diff(root_pos, dt) → world → body` (`motion_lib.py:441-445`).
  - angular vel (body): `_euler_rates_to_body_angvel` (`motion_lib.py:182-189`).
  - dof_vel: forward finite diff (`_finite_diff(dof_pos, dt)`).
  - foot_pos: 손코딩 Go2 FK (`_go2_fk_foot_pos`, `motion_lib.py:97-138`) — base-local frame, [FL, FR, RL, RR].
- DOF 순서 매핑: `Go2MotionLib.get_dof_index(robot_joint_names)` (`motion_lib.py:381-394`) → `_motion_dof_indices` (`go2_imitation_env.py:80-87`). 이름 불일치 시 1:1 fallback.
- 샘플링:
  - `sample_motions(n)`: `torch.multinomial(weights)` (`motion_lib.py:324`). weights는 모션 길이(초) 비례 정규화 (`motion_lib.py:286-292`).
  - `sample_times(motion_ids, truncate_time=0.0)`: `[0, motion_length]` 균등 (`motion_lib.py:336-340`).
  - `calc_motion_frame`: 선형 보간 + quaternion SLERP (`motion_lib.py:342-371`).
- **Retarget 없음** (Go2 → Go2 직접 사용. FK도 Go2 URDF 파라미터 하드코딩).

### 1.9 Domain randomization
- **미사용**. env 코드/cfg 어디에도 push, mass, friction, gravity, actuator randomization 호출이 없음. (검색 결과 env 파일에 `push_`, `apply_external`, `randomize_` 키워드 부재.)
- 시뮬레이션 friction은 1.0 고정 (`env_cfg.py:103-105`).

### 1.10 Command / Goal conditioning
- 정책 입력에 motion id / phase / future target frame **없음** (1.3 참고).
- 대신 **steering 명령** (`_resample_steering`, `go2_imitation_env.py:463-485`):
  - `tar_dir`: 각도 θ∈[-π, π] 균등 → 단위벡터 [cosθ, sinθ] (`:467-470`).
  - `tar_speed ∈ [tar_speed_min=0.5, tar_speed_max=3.0]` 균등 (`:472-476`, cfg `:75-76`).
  - `face_dir = tar_dir` 동일 (`:479`) — `rand_face_dir` 미구현. CLAUDE.md 33-39, 113과 동일.
  - 다음 재샘플까지 `tar_timer ∈ [tar_change_time_min=4.0, tar_change_time_max=7.0]` s 균등 (`:482-485`, cfg `:77-78`).
- 타이머 감소 및 만료 시 재샘플: `_post_physics_step` (`go2_imitation_env.py:162-168`).
- 즉 정책은 "어느 모션을 모방하라" 정보 없이 오직 "어느 방향·속도·자세 방향으로 가라"는 steering 신호만 받음. AMP discriminator는 이 신호와 **무관**하게 reference motion 분포와의 일치도를 평가.

---

## 2. 알고리즘(ALGO)

### 2.1 사용 알고리즘
- Runner: **`OnPolicyRunnerAMPBase`** (`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:249`).
- Algorithm: **`PPOAMPBase`** (`rsl_rl/rsl_rl/algorithms/ppo_amp.py:14`) — `PPO`(기본) 상속, history/DAGGER/priv-encoder 없는 "단순" AMP.
- Policy: **`ActorCritic`** (단순 MLP). Recurrent / CNN / RMA 미사용.

### 2.2 Actor / Critic 네트워크 (`agents/rsl_rl_ppo_cfg.py:45-53`)
- `actor_hidden_dims = [512, 256, 128]`, `critic_hidden_dims = [512, 256, 128]`.
- `activation = "elu"`.
- `init_noise_std = 0.25` (state-independent scalar `nn.Parameter(std)`, `actor_critic.py:97`).
- `actor_obs_normalization = True`, `critic_obs_normalization = True` (`EmpiricalNormalization`, `actor_critic.py:67-81`).
- Critic은 정책과 동일한 obs group을 사용: `obs_groups = {"policy": ["policy"], "critic": ["policy"]}` (`rsl_rl_ppo_cfg.py:40-43`) — critic이 추가 priv 정보를 받지 못함.
- `estimator = None` (`rsl_rl_ppo_cfg.py:55`).
- 액션 분포: `Normal(mean, std)`, std는 `nn.Parameter(0.25 * 1₁₂)` (`actor_critic.py:96-97`).

### 2.3 AMP Discriminator (`rsl_rl/rsl_rl/modules/amp_discriminator.py`, `algorithms/ppo_amp.py:33-58`)
- 클래스 `AMPDiscriminator`.
- 입력 차원: env가 결정 → **86** (`amp_observation_space=43` × `num_amp_observations=2`, runner가 env 값으로 덮어씀; `on_policy_runner_amp.py:261-264`). cfg의 `amp_observation_space=430`은 **무시됨** (override 됨).
- Hidden: `[1024, 512]` (`rsl_rl_ppo_cfg.py:83`).
- 출력: `1` (logit), **출력 활성 함수 없음**(raw logit) (`MLP(input_dim, 1, hidden_dims, activation)`; `last_activation=None` 기본 — `networks/mlp.py:36-37, 77-78`).
- 은닉 activation: `relu` (AMPDiscriminator 생성자 기본값 `activation="relu"`, `amp_discriminator.py:17`) — cfg에서 override 없음.
- Input pre-normalization: `EmpiricalNormalization(input_dim)` (`amp_discriminator.py:32`), 매 `update_amp` 종료 시 `discriminator.update_normalization(cat([expert, policy]))` (`ppo_amp.py:155`).
- `norm_clip = None` (`rsl_rl_ppo_cfg.py:106`) → MimicKit식 ±10 클램프 적용 안 됨.

#### Discriminator input feature 정의 (per-step 43-dim)
- 같은 함수 `_compute_amp_obs(dof_pos, dof_vel, root_pos, root_lin_vel, root_ang_vel, foot_pos_local)`가 policy/agent와 expert(reference motion) 양쪽에서 사용됨 — 정의 일치.
- 구성 (`go2_imitation_env.py:597-620`):
  | 항목 | dim |
  |------|-----|
  | `dof_pos` | 12 |
  | `dof_vel` | 12 |
  | `root_pos[:, 2:3]` (root height) | 1 |
  | `root_lin_vel` (body frame) | 3 |
  | `root_ang_vel` (body frame) | 3 |
  | `foot_pos_local.view(N,-1)` (FL,FR,RL,RR × 3) | 12 |
  | **합계** | **43** |
- 히스토리: `num_amp_observations = 2` (`env_cfg.py:63`) — **2 step만 stack**. CLAUDE.md(:60) 가 적은 "10 history × 43 = 430"은 **현재 cfg와 불일치**.
- 상대 trajectory feature: `include_rel_track_obs = False` (`env_cfg.py:65`) → MimicKit식 `root_pos_obs_xy` 추가 **off**. 켜진다면 차원이 `(43+2) × H`로 증가하지만 현재 비활성.
- 최종 discriminator obs (env가 extras로 expose, `go2_imitation_env.py:221-225`): `amp_obs.view(N, -1)` shape `(N, 86)`.

#### Loss / GP / 정규화
- `disc_loss_type = "ls_gan"` (`rsl_rl_ppo_cfg.py:103`).
  - `expert_loss = MSE(expert_logits, +1)`, `policy_loss = MSE(policy_logits, -1)` (`ppo_amp.py:112-113`).
  - 합: `0.5 * (expert_loss + policy_loss)` (`ppo_amp.py:151`).
- Gradient penalty 양측 (expert + policy):
  - `grad_penalty = 0.5 * gp_coef * (||∇_x D(x_expert)||² + ||∇_x D(x_policy)||²)` (`ppo_amp.py:115-140`).
  - `gradient_penalty_coef = 5.0` (`rsl_rl_ppo_cfg.py:81`).
- Logit reg: `disc_logit_reg_type = "logit"` (`rsl_rl_ppo_cfg.py:105`):
  - `logit_reg = 0.01 * (expert_logits.pow(2).mean() + policy_logits.pow(2).mean())` (`ppo_amp.py:149`).
- 최적화: `Adam(disc.parameters(), lr=2.5e-4)` (`ppo_amp.py:48`, lr `rsl_rl_ppo_cfg.py:80`). **별도 옵티마이저** (PPO와 분리).
- Weight decay 없음.
- `disc_norm_clip = None` (`rsl_rl_ppo_cfg.py:106`).

### 2.4 AMP demo / replay buffer
- `enable_replay_buffer = True` (`rsl_rl_ppo_cfg.py:88`).
- `replay_buffer_size = 200_000` (`rsl_rl_ppo_cfg.py:89`) — flat circular buffer of policy AMP obs (per-step, dim 86).
- Buffer 추가: `add_to_replay_buffer(policy_amp_obs_batch)` — iteration 당 `num_steps_per_env × num_envs = 24 × 4096 = 98_304` 새 샘플 추가 (`ppo_amp.py:64-81`, runner `on_policy_runner_amp.py:155`).
- Disc 학습 시 policy pool 구성 (`on_policy_runner_amp.py:160-166`):
  ```
  num_samples = policy_amp_obs_batch.shape[0]   # = 24*4096 = 98_304
  replayed    = sample_replay_buffer(num_samples)   # 98_304
  policy_pool = cat(policy_amp_obs_batch, replayed) # ~196_608
  ```
- Expert pool: `env.get_amp_observations(num_pool)` 호출 (`on_policy_runner_amp.py:169`):
  - 환경 `Go2ImitationEnv.get_amp_observations(N)` → `collect_reference_motions(N)` (`go2_imitation_env.py:568-570`).
  - 내부: `_compute_reference_buffers(N)` — `sample_motions(N)` + 균등 `sample_times` + `calc_motion_frame` → 동일 43-dim feature를 H-step 적층 (`go2_imitation_env.py:491-537`).
  - **expert 샘플 수 = policy_pool 수**. 즉 demo:policy 비율 ≈ 1:1.
- Mini-batch 학습 (`on_policy_runner_amp.py:172-178`):
  - `disc_num_epochs = 2`, `disc_mini_batch_size = 4096` (`rsl_rl_ppo_cfg.py:84-85`).
  - 매 epoch 마다 `randperm(num_pool)`으로 policy/expert pool을 동일 perm으로 동기 셔플하여 mini-batch 추출.
  - 따라서 iter당 disc step 수 ≈ `2 epoch × ceil(196_608 / 4096) = 2 × 48 = 96`.

### 2.5 Reward 합성 (task + style)
- Style (amp) reward 계산: `AMPDiscriminator.compute_amp_reward(amp_obs)` (`amp_discriminator.py:49-72`):
  - `disc_reward_type = "ls_gan"` (`rsl_rl_ppo_cfg.py:104`):
    ```
    reward = clamp(1 - 0.25*(logit - 1)², min=0) * amp_reward_coef
    amp_reward_coef = 2.0   (rsl_rl_ppo_cfg.py:82)
    ```
  - 입력은 normalized + (선택) clamp([-norm_clip, norm_clip]) — 현재 norm_clip=None.
- 합성식 (`on_policy_runner_amp.py:331-332`):
  ```
  task_reward_lerp = alg.amp_task_reward_lerp
  total_reward = task_reward_lerp * rewards + (1 - task_reward_lerp) * amp_reward
  ```
- `task_reward_lerp` 스케줄 (`on_policy_runner_amp.py:296-298`):
  ```
  if enable_lerp_schedule and anneal_iters > 0:
      progress = min(1.0, (it - start_it) / anneal_iters)
      lerp = lerp_start + (lerp_end - lerp_start) * progress
  ```
- cfg 값 (`rsl_rl_ppo_cfg.py:75-78`):
  - `task_reward_lerp = 0.5` (end)
  - `task_reward_lerp_start = 0.5` (start)
  - `task_reward_lerp_anneal_iters = 5000`
  - `enable_lerp_schedule = True`
- **즉 lerp는 시작/끝 동일 0.5 → 사실상 anneal 없이 항상 50%/50% 합성**. CLAUDE.md(:37, :103-104, :155)이 기술한 "1.0 → 0.5 anneal"은 현재 cfg와 불일치.
- Terminal step 보정 (`on_policy_runner_amp.py:319-322`): done env의 `amp_obs`는 RSI 직후 obs이므로, env가 보존한 `extras["terminal_amp_obs"]`(=pre-reset 마지막 obs)로 교체 후 amp_reward 계산.
- 보고: episode 종료 시 `Episode_Reward/amp_reward = sum(amp_reward)/ep_lens` (`on_policy_runner_amp.py:336-347`).

### 2.6 PPO 하이퍼파라미터 (`agents/rsl_rl_ppo_cfg.py:57-71`)
| 항목 | 값 |
|------|-----|
| `learning_rate` (actor+critic) | `2e-4` |
| `schedule` | `"adaptive"` (KL 기반 adapt; `desired_kl=0.01`, PPO base `ppo.py:262-296`) |
| `clip_param` | `0.2` |
| `entropy_coef` | `0.007` |
| `value_loss_coef` | `1.0` |
| `use_clipped_value_loss` | `True` |
| `gamma` | `0.99` |
| `lam` (GAE λ) | `0.95` |
| `desired_kl` | `0.01` |
| `max_grad_norm` | `1.0` |
| `num_learning_epochs` | `5` |
| `num_mini_batches` | `4` |
| `normalize_advantage_per_mini_batch` | (미설정, default `False` — `ppo.py:47`) |
| `num_steps_per_env` | `24` (`rsl_rl_ppo_cfg.py:30`) |
| `max_iterations` | `50000` |
| `save_interval` | `100` |
| `experiment_name` | `"go2_imitation"` |
| `clip_actions` | `1.0` |
| `class_name` (algo) | `"PPOAMPBase"` |
| `class_name` (runner) | `"OnPolicyRunnerAMPBase"` |
| Discriminator `learning_rate` | `2.5e-4` |
| Discriminator `disc_num_epochs` | `2` |
| Discriminator `disc_mini_batch_size` | `4096` |
| Discriminator `gradient_penalty_coef` | `5.0` |
| Discriminator `reward_coef` | `2.0` |
| Discriminator `disc_logit_reg` | `0.01` |
| Discriminator `replay_buffer_size` | `200_000` |
- 실제 학습 command 권장 `--num_envs 4096`(CLAUDE.md:172) → batch = 4096 × 24 = 98_304 transitions / iter, mini-batch ≈ 24_576.

### 2.7 Normalization
- **Policy obs**: `EmpiricalNormalization` ON for actor와 critic (2.2).
- **Value normalization** (returns scale): PPO base에는 별도 value normalizer 없음 (`ppo.py:307-315` — clipped value loss만).
- **Advantage normalization**: 전체 batch 단위로 1회 (`ppo.py:194-196`). `normalize_advantage_per_mini_batch=False`.
- **Discriminator input normalization**: `EmpiricalNormalization`, expert+policy 합쳐 매 disc step마다 통계 업데이트 (`ppo_amp.py:155`).
- Reward normalization (return scaling) 없음 — task reward + amp reward는 raw로 합산되어 PPO에 전달.

---

## 3. 데이터 흐름 자체 검증

1. **Expert vs Policy AMP obs 정의 일치성**:
   - 동일 함수 `_compute_amp_obs(dof_pos, dof_vel, root_pos, root_lin_vel, root_ang_vel, foot_pos_local)` 사용:
     - Policy (live): `go2_imitation_env.py:188-195`.
     - Expert (motion): `go2_imitation_env.py:525-532` (`_compute_reference_buffers`).
   - 차원·필드 모두 동일. → **policy 측 frame 정의와 expert 측 frame 정의가 매칭됨**.
   - 단, expert의 `root_lin_vel`, `root_ang_vel`은 모션 PKL에서 finite diff + Euler-rate 변환으로 만든 body-frame 값 (`motion_lib.py:441-446`). Policy 측은 `_robot.data.root_lin_vel_b`/`root_ang_vel_b` (시뮬 측정 body-frame). 정의는 동일하나 **노이즈 특성/계산 경로**는 다름(시뮬 PD 노이즈 vs PKL fin-diff).
   - `foot_pos_local`도 동일 정의(base-local)이지만 expert는 손코딩 FK(`_go2_fk_foot_pos`, `motion_lib.py:97-138`)에서, policy는 Isaac 측정 `body_pos_w[...]` + `quat_apply_inverse`로 계산 (`go2_imitation_env.py:180-185`) → 동일 정의에서 측정/계산이 다름.

2. **DOF 순서 매핑**:
   - 모션 데이터 → IsaacLab 순서: `dof_pos[:, _motion_dof_indices]`로 재배열 (`_compute_reference_buffers`: `:522-523`, `_reset_strategy_rsi`: `:442-443`).
   - Policy live obs는 IsaacLab 순서 그대로 (`self._robot.data.joint_pos`). → **양측이 IsaacLab joint order로 통일됨**.

3. **Steering / phase 정보의 disc 누락 여부**:
   - Discriminator 입력 `_compute_amp_obs(...)`에는 `tar_dir`, `tar_speed`, `face_dir`, motion_id, phase 어느 것도 포함되지 않음.
   - Policy obs에는 steering 명령(5-dim)이 포함되나 motion_id/phase는 없음.
   - 즉 **discriminator는 "이 행동이 어떤 motion을 흉내내고 있는가"에 대한 라벨 없이 unconditional 분포 일치만 평가**. Reference 모션의 walk/trot/run/pace/turn 9종 mirror 포함 18종 전체 집합에 대해 marginal 분포만 supervisable.
   - Policy는 steering 명령은 받지만 어떤 모션을 따라야 할지 명령 없음.

4. **Terminal step 처리**:
   - env가 reset 직전에 `_terminal_amp_obs[env_ids]` 저장 (`go2_imitation_env.py:330-379`), runner가 done env의 `agent_amp_obs`를 `terminal_amp_obs`로 교체 후 amp_reward 계산 (`on_policy_runner_amp.py:319-322`). 정합.

5. **Replay buffer 일관성**:
   - Replay에 들어가는 obs는 terminal-corrected policy obs (`amp_obs_buffer.append(corrected_amp_obs.detach())`, `on_policy_runner_amp.py:326`). → 일관됨.

---

## 4. 불확실 / 미확인 항목

- DirectRLEnv의 `(terminated, truncated)` 반환이 wrapper(`isaaclab_rl/.../RslRlVecEnvWrapper`)에서 `extras["time_outs"]`로 매핑되는 정확한 라인은 본 분석에서 직접 확인하지 않음(파일 미열람). 통상 wrapper가 처리하지만, time_outs 부재 시 PPO bootstrap이 비활성화되어 cumulative 보상 학습이 달라질 수 있음 → **확인 필요**.
- `clip_actions=1.0` (`rsl_rl_ppo_cfg.py:34`) 이 runner 내부에서 어디서 적용되는지(env 전달 전 clip vs 단순 메타) 직접 라인은 본 보고서에서 미확인.
- `prev_root_pos_w` 초기값은 `torch.zeros(...)`로 생성되나 (`go2_imitation_env.py:96`), 첫 episode 시작 시 `_reset_idx` 내 `_prev_root_pos_w[env_ids] = root_state[:, :3]`로 즉시 RSI root 위치로 채워짐 (`:396`). → 초기 step에서 비정상 큰 속도가 발생하지는 않을 것으로 보이나, IsaacLab DirectRLEnv 첫 `_get_rewards` 호출 시점은 직접 트레이스하지 않음.
- `imitation/smr_mirror_pkl/*.pkl` 모션 데이터의 fps / 총 길이는 PKL 내부 메타로 결정 (`motion_lib._load_pkl`); 본 보고서에서는 PKL 내부 값을 실제 읽지 않음(파일 바이너리 미실행). 로그에 `[Go2MotionLib] 로드: ...` 출력으로 검증 가능 (`motion_lib.py:279`).
- `include_rel_track_obs` 토글 시 `amp_observation_size = (43+2) × H = 90`이 되며 cfg `amp_observation_space=430`(`rsl_rl_ppo_cfg.py:93`)과 모두 불일치. runner override 로직 (`on_policy_runner_amp.py:262-264`)이 env의 `amp_observation_space.shape[0]`(env가 `__init__`에서 `gym.spaces.Box(... shape=(amp_observation_size,))`로 구성, `go2_imitation_env.py:99-102`)을 우선시하므로 실제 disc 입력 크기는 환경 cfg 기준으로만 결정됨. → 현재 86. cfg의 430은 dead code.
- `agents/rsl_rl_ppo_cfg.py:91-92` 주석은 "amp_observation_space = 450 (rel_track=True) / 430 (rel_track=False)"로 계산하나, 이는 `H=10` 가정. 실제 `num_amp_observations=2` 하에서는 90/86.
- CLAUDE.md 문서가 기술한 "AMP disc 430-dim / 10 history / lerp 1.0→0.5 / motion dir=go2/" 등은 현재 코드 상태와 다수 불일치. 문서가 옛 버전 기준일 가능성 있으나 본 보고서는 코드 우선.

---

## 부록: 핵심 수치 한눈에

| 항목 | 값 | 근거 |
|---|---|---|
| Policy obs dim | **50** (cfg 주석 "44"와 상이) | `env_cfg.py:59`, `env.py:238-251` |
| AMP per-step obs dim | 43 | `env_cfg.py:64`, `env.py:597-620` |
| AMP history `H` | **2** (문서 "10"과 상이) | `env_cfg.py:63` |
| AMP disc input dim (실효) | **86** (cfg 430은 override됨) | `env.py:99-102`, `runner:262-264` |
| `include_rel_track_obs` | **False** | `env_cfg.py:65` |
| Reset 전략 | 항상 RSI (random time, length-weighted motion) | `env.py:386`, `motion_lib.py:286-292` |
| Termination | h<0.15, gz>0, |roll|>70°, |pitch|>70°, base contact>500N | `env.py:294-317` |
| Episode | 10 s / 500 step | `env_cfg.py:51` |
| Decimation / dt | 4 (200→50 Hz) | `env_cfg.py:54-56` |
| Action | scale 0.25, target = scale·a + default_qpos | `env_cfg.py:94`, `env.py:160` |
| Actuator | DCMotor: K=25, D=0.5, eff=23.5, vel=30, arm=0.01 | `unitree.py:170-181` |
| Domain rand | **없음** | (코드 부재) |
| Task reward | 0.7·tar + 0.3·face (no clip) | `env.py:280`, `env_cfg.py:82-83` |
| Style reward | LS-GAN: clamp(1−0.25(d−1)²,0)·2.0 | `amp_discriminator.py:69-72` |
| Lerp | start=end=0.5 (anneal no-op) | `rsl_rl_ppo_cfg.py:75-78` |
| Actor/Critic | MLP [512,256,128] ELU, init_std=0.25 | `rsl_rl_ppo_cfg.py:45-53` |
| Discriminator | MLP [1024,512] ReLU, output linear logit, EmpNorm | `amp_discriminator.py:13-36`, `rsl_rl_ppo_cfg.py:83` |
| PPO lr / clip / ent / γ / λ | 2e-4 / 0.2 / 0.007 / 0.99 / 0.95 | `rsl_rl_ppo_cfg.py:60-67` |
| PPO epochs / mini_batch | 5 / 4 | `rsl_rl_ppo_cfg.py:61-62` |
| Disc lr / epochs / mini_batch | 2.5e-4 / 2 / 4096 | `rsl_rl_ppo_cfg.py:80, 84-85` |
| GP coef / logit reg | 5.0 / 0.01 (logit-mean²) | `rsl_rl_ppo_cfg.py:81, 86` |
| Replay buffer | 200_000, expert:policy=1:1 (≈196k mix) | `rsl_rl_ppo_cfg.py:88-89`, `runner:161-169` |
| Motion file dir (활성) | `imitation/smr_mirror_pkl` (CLAUDE.md의 go2/ 대신) | `env_cfg.py:30` |
| Motion list | 9 base + 9 mirror = 18 PKL (walk/walk1/walk2/walk_turn/run0~2/trot0~1) | (디렉토리 ls) |
| Sampling weight | motion length(s) 비례 | `motion_lib.py:286-292` |
