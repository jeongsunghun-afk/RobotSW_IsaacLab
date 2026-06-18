# MimicKit `amp_go2_track_args` 분석 보고서

> 본 문서는 **MimicKit 측**의 환경/알고리즘 구현을 코드 그대로 정리한다. IsaacLab `go2_imitation`과의 비교/원인 분석은 lead가 수행한다. 모든 항목은 `file:line` 인용. 확인 불가는 "확인 안 됨"으로 표기.

---

## 0. 진입점 & 학습 command

### 0.1 argfile
- 경로: `/home/lgb/MimicKit/args/amp_go2_track_args.txt`
- 내용 (`args/amp_go2_track_args.txt:1-5`):

  ```
  --num_envs 4096
  --engine_config data/engines/isaac_lab_engine.yaml
  --env_config data/envs/amp_go2_tracking_env.yaml
  --agent_config data/agents/amp_go2_task_agent.yaml
  ```

- argfile 자체에는 `--mode` 가 들어있지 않으므로 default(`train`)가 사용됨 (`mimickit/run.py:103` — `mode = args.parse_string("mode", "train")`). `--logger`, `--visualize`, `--out_dir`, `--max_samples` 등은 CLI에서 추가로 넘긴다.

### 0.2 재구성된 학습 command
```bash
python mimickit/run.py --arg_file args/amp_go2_track_args.txt
# 동등:
python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_go2_tracking_env.yaml \
  --agent_config data/agents/amp_go2_task_agent.yaml
```
- Trainer 엔트리 `mimickit/run.py:run()` (`mimickit/run.py:99-184`) → `build_env()` (line 35) → `build_agent()` (line 42) → `agent.train_model(...)` (line 49 / `mimickit/learning/base_agent.py:52`).

### 0.3 클래스/파일 매핑
- **Env class**: `TaskTrackingEnv` — `mimickit/envs/task_tracking_env.py:5`
  - MRO: `TaskTrackingEnv → TaskTrackingMixin → AMPEnv → DeepMimicEnv → CharEnv → SimEnv → BaseEnv`
  - 등록자: `mimickit/envs/env_builder.py` (env_name="task_tracking" → 본 클래스, env_config:1)
- **Agent class**: `AMPAgent` — `mimickit/learning/amp_agent.py:11`
  - 상속: `PPOAgent` (`mimickit/learning/ppo_agent.py:12`) → `BaseAgent` (`mimickit/learning/base_agent.py:27`)
  - 등록자: `mimickit/learning/agent_builder.py` (agent_name="AMP" → 본 클래스)
- **Model class**: `AMPModel` — `mimickit/learning/amp_model.py:7` (상속 `PPOModel`)
- **Motion file**: `data/datasets/dataset_go2_locomotion_smr_mirror3.yaml` — pace(0.0) 제외 run/trot/walk 모션 18개 (weight=1.0). 실제 `.pkl` 경로 `data/motions/smr_mirror_pkl/go2_*.pkl`
- **Engine**: `IsaacLabEngine` (`mimickit/engines/isaac_lab_engine.py:63`), config `data/engines/isaac_lab_engine.yaml`

---

## 1. 환경(ENV)

### 1.1 시뮬레이션 / 타이밍
- `isaac_lab_engine.yaml:6-7`:
  - `control_freq: 50` → policy step Δt = `1/50 = 0.02 s` (`mimickit/engines/isaac_lab_engine.py:73`)
  - `sim_freq: 200` → physics dt = `1/200 = 0.005 s` → **decimation = sim_freq/control_freq = 4** (`isaac_lab_engine.py:74` `self._sim_steps = int(sim_freq / control_freq)`)
- `episode_length: 10.0` 초 (`amp_go2_tracking_env.yaml:6`) ⇒ 약 500 control step
- `env_spacing: 5` m (yaml:8) — 환경 격자 간격
- `num_envs default = 4096` (argfile)
- PhysX 설정 (`isaac_lab_engine.py:770-775`): `bounce_threshold_velocity=0.2`, `max_position_iter=4`, `max_velocity_iter=0`, `gpu_max_rigid_contact_count=8M`, `static_friction=1.0`, `dynamic_friction=1.0` (DR로 override됨)

### 1.2 로봇 / Actuator
- **Asset**: `data/assets/go2/go2.xml` (MJCF). USD가 같은 디렉토리에 존재(`go2.usd`)하나 `IsaacLabEngine._parse_usd_path()` (`isaac_lab_engine.py:788-792`)가 확장자만 `.usd`로 바꿔서 로드 → 실제 시뮬은 `go2.usd` 사용. Kinematic model(`MJCFCharModel`, `char_env.py:131-133`)은 `.xml` 사용.
- **DOF**: 12 (4 다리 × hip/thigh/calf, `go2.xml:44-49…`)
- **Joint ranges & xml-defined gains** (`go2.xml`):
  - hip: range `[-1.0472, 1.0472]`, `actuatorfrcrange=±23.7`, xml `stiffness=15, damping=2, armature=0.01, frictionloss=0.2`
  - thigh: range `[-1.5708, 3.4907]`, `actuatorfrcrange=±23.7`, 동일 xml gains
  - calf: range `[-2.7227, -0.83776]`, `actuatorfrcrange=±35.5`, 동일 xml gains
  - 글로벌 `motor ctrlrange="-33.5 33.5"`
- **실제 적용 PD gains** (`isaac_lab_engine.yaml:13-15`):
  - `stiffness: 25.0 (kp)`, `damping: 1.0 (kd)` — 모든 12 joint에 동일 스칼라
  - `_build_actuator_cfg()` (`isaac_lab_engine.py:794-813`)에서 `control_mode=pos` → `ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=kp, damping=kd, effort_limit=None)` 사용. **effort_limit=None** (즉, ImplicitActuator는 USD/MJCF의 actuatorfrcrange를 따른다고 봐야 하나 명시적 지정 없음 — Isaac Lab default 동작에 위임).
- **init_pose** (`amp_go2_tracking_env.yaml:22`): `[0, 0, 0.27, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]`
  - root pos `(0,0,0.27)`, root rot exp_map `(0,0,0)`, joints `[hip=0, thigh=0.9, calf=-1.8] × FL/FR/RR/RL`
- **start_joint_pos dict** for hinge joints는 `CharEnv._build_start_joint_pos_dict()` (`char_env.py:104-114`)에서 init_dof_pos로부터 생성되어 `ArticulationCfg.InitialStateCfg(joint_pos=…)`에 전달 (`isaac_lab_engine.py:994-995`).
- **Hip scale reduction**: `hip_scale_reduction: true`, factor `0.5` (`env yaml:55-56`). engine에서 set_cmd 시 hip DOF에 해당하는 action에 0.5를 곱함 (`isaac_lab_engine.py:220-222`, `setup_hip_scale_reduction()` line 1113-1133).

### 1.3 Observation (policy obs)

> `amp_go2_task_agent.yaml`에는 `enable_priv_encoder`가 없으므로 **legacy path** 사용 (`ppo_agent.py:40`, default False). priv_obs cfg dict는 actor용 obs에서 특정 항목을 **끄는** 플래그로 동작.

- 계산 진입: `TaskTrackingMixin._compute_obs()` (`task_tracking_mixin.py:66`) → `super()._compute_obs()` = `DeepMimicEnv._compute_obs()` (`deepmimic_env.py:332`, legacy 분기 line 391-413) → `compute_deepmimic_obs()` (`deepmimic_env.py:776`)
- 활성 플래그 (env yaml & `char_env.py:21-27`):
  - `global_obs: False` (yaml:7) → 모든 회전·속도를 heading-local frame으로 변환
  - `root_height_obs: False` (yaml:8, 그리고 `priv_obs.root_height_obs: False` yaml:18)
  - `priv_obs.root_vel_obs: False` (yaml:15) → `_root_vel_obs = False`
  - `priv_obs.root_ang_vel_obs: False` (yaml:16) → `_root_ang_vel_obs = False`
  - `priv_obs.key_pos: False` (yaml:17) → `_key_pos_obs = False`
  - `enable_phase_obs: False` (yaml:10)
  - `enable_tar_obs: False` (yaml:11)
- **`compute_char_obs()`** (`char_env.py:625-665`) 산출:
  - root_height: 제외 (root_height_obs=False)
  - root_rot tan_norm (local): **6**
  - root_vel (local): 제외
  - root_ang_vel (local): 제외
  - joint_rot tan_norm × 12 joints (1D은 stub spherical로 변환): `12 × 6 = 72`
  - dof_vel: **12**
  - key_pos: 제외
  → char_obs dim = **6 + 72 + 12 = 90**
- DeepMimic phase/tar obs 모두 비활성 → DeepMimic obs = 90
- **TaskTrackingMixin._compute_obs() (legacy path line 72-74)**: `task_obs = [cmd_lin_x, cmd_lin_y, cmd_ang_yaw]` (3)를 concat
- **최종 policy obs dim = 90 + 3 = 93**
- Normalization (`base_agent.py:171, 343`): 전체 obs를 단일 `Normalizer(shape=(93,), clip=10.0)`로 running mean/std 정규화. `normalizer_samples: 1e8` (`amp_go2_task_agent.yaml:30`)까지 통계 갱신.
- Noise: **추가되지 않음** (확인 안 됨 — env 코드 어디서도 obs에 가우시안 노이즈를 더하는 호출 없음)
- History/phase: **없음** (legacy path, priv_encoder 비활성)

### 1.4 Action
- Action space: `CharEnv._build_action_space()` (`char_env.py:162-193`) → `control_mode=pos` 분기 `_build_action_bounds_pos()` (`char_env.py:203-245`)
  - 각 1D 관절: `mid = 0.5*(high+low)` (zero_center_action=False, env yaml에 키 없음 → default False), `scale = max(|high-mid|, |low-mid|) * 1.4`, `[mid-scale, mid+scale]`
  - 결과: 12-dim Box. (e.g. hip ±1.466, thigh `[-2.58, 4.50]`, calf `[-3.10, -0.46]`)
  - **즉 action = 절대 joint position target 직접 출력**. default-pose offset 없음.
- Action normalizer (`base_agent.py:195-216`): `init_mean = 0.5*(high+low)`, `init_std = 0.5*(high-low)` → 정책 출력 norm_a는 [-1,1] 스케일로 학습, `unnormalize` 후 절대 joint pos.
  - 정책 init: actor 마지막 layer weight `~U(-0.01, 0.01)` (`base_model.py` via `actor_init_output_scale`), `action_std=0.1` FIXED → 초기 norm_a ≈ 0 → 초기 절대 target ≈ joint range 중점. (init_pose `[0, 0.9, -1.8]`이 thigh mid 0.96 / calf mid -1.78에 가까움.)
- 클립: `_apply_action()` (`char_env.py:472-476`)에서 `torch.minimum/maximum`으로 action bound 클립 후 `engine.set_cmd()`.
- **Low-pass filter / EMA: 없음** (확인 안 됨 — code path에 필터 없음).
- **Hip scale reduction** (`isaac_lab_engine.py:220-222`): set_cmd 직전 hip DOF index에 0.5 곱셈.
- **목적지**: `engine.set_cmd()` → `obj.set_joint_position_target(sim_cmd)` (`isaac_lab_engine.py:227`) — PD target. 4 physics substep 동안 동일 target 유지.

### 1.5 Reward

> **중요**: AMPEnv가 `_update_reward()`를 **빈 함수로 override** 한다 (`amp_env.py:275-276`):
> ```python
> def _update_reward(self):
>     return
> ```
> 따라서 DeepMimicEnv의 imitation reward (pose/vel/root_pose/root_vel/key_pos)는 **호출되지 않는다**.
> 단, MRO에 의해 `TaskTrackingEnv._update_reward()` → `TaskTrackingMixin._update_reward()` (`task_tracking_mixin.py:78-96`)가 우선 호출되어 task tracking reward를 기록한다 (MRO order: TaskTrackingMixin이 AMPEnv보다 앞에 위치, `task_tracking_env.py:5`).
> ⇒ env가 `_reward_buf`에 쓰는 값은 **task tracking reward만**.

- **Task tracking reward** (`task_tracking_mixin.py:152-180`, jit):
  - heading-local linear velocity `body_lin_vel = quat_rotate(heading_inv_rot, world_lin_vel)`
  - `lin_vel_err² = (cmd_x - body_x)² + (cmd_y - body_y)²`
  - `lin_vel_reward = exp(-lin_vel_scale * lin_vel_err²)`, scale=`1.0` (env yaml:52)
  - `ang_err² = (cmd_yaw - world_ang_vel_z)²`
  - `ang_vel_reward = exp(-ang_vel_scale * ang_err²)`, scale=`0.5` (yaml:53)
  - `reward = lin_vel_w * lin_vel_reward + ang_vel_w * ang_vel_reward`
  - `lin_vel_w = 0.5`, `ang_vel_w = 0.5` (yaml:50-51)
  - **클립 없음** (모든 항이 양수 [0,1] exp이므로 자연히 [0,1])
- **`reward_pose_w / reward_vel_w / reward_root_pose_w / reward_root_vel_w / reward_key_pos_w` (env yaml:38-42)는 파싱은 되지만(`deepmimic_env.py:28-38`) AMPEnv가 reward 계산을 건너뛰므로 학습 시 사용되지 않는 dead config.**
- Imitation은 전적으로 **AMP discriminator의 style reward** (Section 2.5)로만 제공된다.
- `get_reward_succ() = get_reward_fail() = 0.0` (`deepmimic_env.py:47-53`)

### 1.6 Termination
- `AMPEnv._update_done()` (`amp_env.py:244-273`) → `deepmimic_env.compute_done()` (`deepmimic_env.py:872-933`, jit) 호출. 단, `motion_len_term`을 **항상 False**로 강제(`amp_env.py:247`) — motion 종료가 SUCC 트리거 안 함.
- 적용 조건:
  - **TIME (timeout)**: `time >= episode_length(=10.0)` → `DoneFlags.TIME`
  - **SUCC (motion end)**: 비활성 (motion_len_term=False)
  - **FAIL (early termination, `enable_early_termination=True` yaml:24)**:
    - `contact_bodies = ["FR_foot","FL_foot","RR_foot","RL_foot"]` (yaml:26)
    - logic (`deepmimic_env.py:892-898`): ground contact force의 contact_body index를 **0으로 마스킹**한 뒤 나머지 body에서 `|force|>0.1`이 있으면 fall로 간주
    - ⇒ **non-foot body가 지면에 닿으면 즉시 fail**
    - `pose_termination: False` (yaml:9) → pose 거리 종료 비활성
    - `not_first_step` 가드: time>0 일 때만 fail 활성화 (line 929)
- `pose_termination_dist`: default 1.0 (`deepmimic_env.py:18`) — 사용 안 됨

### 1.7 Reset / Reference State Initialization
- 흐름:
  1. `TaskTrackingMixin._reset_envs()` (`task_tracking_mixin.py:111-116`) → `super()._reset_envs(env_ids)` → `AMPEnv._reset_envs()` (`amp_env.py:278-283`) → `DeepMimicEnv._reset_envs()` → `CharEnv._reset_envs()` (`char_env.py:434-442`) → `SimEnv._reset_envs()`
  2. `CharEnv._reset_envs`는 `self._reset_char(env_ids)` (DeepMimic 버전 line 151-157) → `_reset_ref_motion(env_ids)` (line 175-197) → `_ref_state_init(env_ids)` (line 202-215)
  3. 그 다음 `self._reset_char_rigid_body_state(env_ids)` (line 459) → `_engine.apply_reset_randomization(env_ids)` (line 441)
  4. `AMPEnv._reset_envs`는 추가로 `_reset_disc_hist(env_ids)` (line 285-297)로 10-step disc history 버퍼를 motion에서 fetch한 과거 frame들로 채움
  5. `TaskTrackingMixin._reset_envs` 추가로 `_reset_task(env_ids)` (line 118-125): cmd 샘플 + 다음 변경 시각 = 현재 time + uniform([4,7])
- **RSI (Reference State Initialization)**:
  - `_sample_motion_times(n)` (`deepmimic_env.py:277-285`):
    - motion_id = `torch.multinomial(motion_weights, n, replacement=True)` (`motion_lib.py:30-32`)
    - `rand_reset: True` (yaml:13) → `motion_time = phase_uniform * motion_length` (`motion_lib.py:34-43`)
  - 그 시점의 motion state를 char에 직접 set: root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel (`deepmimic_env.py:202-214`)
- ref_char (visualization)는 `_enable_ref_char() = self._visualize and self._visualize_ref_char` (`deepmimic_env.py:138-139`) — train 시 visualize=False면 안 만들어짐.

### 1.8 Motion library
- File: `data/datasets/dataset_go2_locomotion_smr_mirror3.yaml` — `motion_lib.MotionLib._fetch_motion_files()` (`motion_lib.py:192-225`)가 yaml의 `motions:` 리스트를 읽음.
- 실제 motion = `data/motions/smr_mirror_pkl/go2_*.pkl` (24개, pace 6개 weight 0.0, run/trot/walk 18개 weight 1.0).
- Format: `mimickit/anim/motion.py`의 `Motion` 클래스가 pkl 로드. 각 frame = `[root_pos(3), root_rot(3 exp_map), joint_dof(12)]` (`README.md:127-134`, `motion_lib.py:11-15`).
- FPS는 모션별로 pkl에 저장, runtime에서 frame-rate에 맞춰 finite-difference로 `root_vel`, `root_ang_vel`, `dof_vel` 자동 계산 (`motion_lib.py:170-181`).
- Sampling:
  - motion id: `torch.multinomial(weights)` (weight 0.0인 pace는 절대 샘플되지 않음)
  - time: uniform [0, motion_len]
  - frame blend: 선형 interp(root_pos) + slerp(root_rot, joint_rot), dof_vel/root_vel은 frame_idx0 직접 사용 (`motion_lib.py:66-92`)
- 모션 길이/통계: 확인 안 됨 (pkl 파일을 열어보지 않음)
- Retargeting: motion이 GMR 기반이라는 README 언급(`README.md:138`). smr_mirror_pkl이라는 디렉토리명으로 보아 mirror augmentation이 적용된 SMR 데이터셋. 정확한 retarget 파이프라인 확인 안 됨.

### 1.9 Domain randomization
- `isaac_lab_engine.yaml:20-74`, 어댑터 `mimickit/engines/isaac_lab_dr_adapter.py` (코드 직접 미확인, cfg 의미만):
  - **physics_material** (startup, 1회): static_friction `[0.4,1.2]`, dynamic_friction `[0.3,1.0]`, restitution 0, num_buckets 64
  - **add_base_mass** (startup): base body mass에 uniform `[-0.5, +5.0]` kg 추가
  - **randomize_com** (startup): base CoM offset x±0.15, y±0.05, z±0.05 m
  - **joint stiffness/damping** (reset마다, log-uniform): kp scale `[0.75, 1.5]`, kd scale `[0.5, 1.5]`
- push force / observation noise / action noise: **설정 없음**

### 1.10 Command / Goal conditioning
- Policy 입력에 들어가는 task 정보 = `[cmd_lin_x, cmd_lin_y, cmd_ang_yaw]` (3-dim, body frame 스칼라, heading 회전 변환 없이 그대로 concat — `task_tracking_mixin.py:144-149` 주석에 "이미 body frame 스칼라").
- Range (yaml:28-33): `lin_x ∈ [0, 4] m/s` (전진 only), `lin_y ∈ [0, 0]` (lateral 항상 0), `ang_yaw ∈ [-1, 1] rad/s`
- cmd 변경 주기: 매 env마다 uniform `[4, 7]` 초마다 재샘플 (`task_tracking_mixin.py:103-125`)
- **motion id / phase / target frame window는 policy에 들어가지 않는다** (enable_phase_obs=False, enable_tar_obs=False) — 즉 어떤 motion을 따라가야 하는지 정책이 모르고, "원하는 속도를 추종하라"만 알 수 있음. 모션 의존성은 AMP discriminator를 통해서만 주입됨.
- Discriminator 입력에는 cmd / phase / tar 모두 **들어가지 않음** (Section 2.3).

---

## 2. 알고리즘(ALGO)

### 2.1 사용 알고리즘
- 클래스: `AMPAgent` extends `PPOAgent` (`mimickit/learning/amp_agent.py:11`)
- 등록명: "AMP" (agent yaml line 1)
- 학습 루프 진입: `BaseAgent.train_model()` (`base_agent.py:52-102`) → 반복 `_train_iter()` (`base_agent.py:259-280`):
  - `_rollout_train(steps_per_iter)` (line 285-296)
  - `_build_train_data()` (PPO line 157 → AMP override line 81-89 — disc demo fetch + disc replay push + reward 합성)
  - `_update_model()` (AMP line 134-146 — PPO update + disc update)
  - `_update_normalizers()` (line 76-79 — obs_norm + disc_obs_norm)

### 2.2 Actor / Critic 네트워크
- Net builder: `learning/nets/net_builder.py` (yaml에서 `actor_net: "fc_2layers_1024units"` 문자열로 모듈 선택)
- `fc_2layers_1024units.build_net()` (`nets/fc_2layers_1024units.py:5-22`):
  - `layer_sizes = [1024, 512]`
  - `Linear(in, 1024) → ReLU → Linear(1024, 512) → ReLU` (Sequential)
  - bias zero init, weight default torch (Kaiming uniform via Linear default)
- Activation은 `BaseModel._activation = torch.nn.ReLU` (`base_model.py:12`)
- **Actor head**: `DistributionGaussianDiag` (`base_model.py:24-25`), `actor_init_output_scale: 0.01` → 마지막 linear weight `U(-0.01, 0.01)`, `actor_std_type: FIXED`, `action_std: 0.1` (학습 안 됨, 상수)
- **Critic head**: 마지막 `Linear(512, 1)` (`ppo_model.py:184-193`), `init.zeros_(bias)`, weight default.
- Normalization: 입력 obs는 `Normalizer(clip=10.0)`로 standardize → 네트워크에 들어가기 전에 적용 (`ppo_agent.py:120, 327, 342`).
- **enable_priv_encoder=False** (agent yaml에 키 없음) → RMA priv/history encoder, dagger 경로 모두 **비활성**.

### 2.3 AMP Discriminator
- 네트워크: `disc_net: fc_2layers_1024units` (agent yaml:11) → 동일 `Linear(in,1024)→ReLU→Linear(1024,512)→ReLU`. 위에 `_disc_logits = Linear(512, 1)` (`amp_model.py:38-40`), weight `U(-1.0, 1.0)`, bias zero.
- **Input feature** (`compute_disc_obs()`, `amp_env.py:327-348`, 그리고 `_compute_disc_obs_demo`/`_update_disc_obs` 둘 다 같은 함수 호출):
  - 입력 차원은 `num_disc_obs_steps = 10` (yaml:12) 프레임 윈도우 × per-step feature.
  - per-step feature 구성 (`compute_disc_obs` → `compute_tar_obs` + `compute_disc_vel_obs`):
    - **위치/회전 부분** (`compute_tar_obs`, `deepmimic_env.py:722-774`):
      - `ref_root_pos / ref_root_rot` = window의 **마지막** frame root state (track_global_root=False이므로, `amp_env.py:41-43`)
      - 각 step의 root_pos를 ref_root_pos 기준 상대 → heading-local 회전 적용 → `root_height_obs=False` ⇒ `root_pos_obs = root_pos_obs[..., :2]` (xy 만, **2 dims/step**)
      - root_rot: ref frame 기준 local 변환 후 `quat_to_tan_norm` → **6 dims/step**
      - joint_rot tan_norm: 12 joint × 6 = **72 dims/step**
      - key_pos: `key_bodies=["FR_foot","FL_foot","RR_foot","RL_foot"]` (yaml:25) → `_has_key_bodies()=True`, 각 step의 4 foot world position을 root 기준 상대 + heading-local 변환 → 4×3 = **12 dims/step**
      - pos_obs subtotal/step = `2 + 6 + 72 + 12 = 92`
    - **속도 부분** (`compute_disc_vel_obs`, `amp_env.py:300-325`):
      - root_vel (heading-local, ref frame 기준): **3 dims/step**
      - root_ang_vel (heading-local): **3 dims/step**
      - dof_vel: **12 dims/step**
      - vel subtotal/step = `3 + 3 + 12 = 18`
    - per-step = `92 + 18 = 110`
  - 10 step × 110 = **1100-dim** flattened disc_obs (확인: `_disc_obs_buf = torch.zeros([N, *disc_obs_space.shape])` line 152-153, shape는 fetch_disc_obs_demo(1) 결과로 동적 결정)
  - **즉 disc input은 "최근 10 step (= 0.2 s @ 50Hz) 동안의 root 위치/회전/속도 + 12 dof rot/vel + 4 foot 위치"의 시간 윈도우.**
  - Policy obs와 달리 root_vel, root_ang_vel, key_pos가 **포함됨** (priv_obs 플래그는 disc에는 적용되지 않음 — `compute_disc_obs`는 priv_obs 플래그를 받지 않고 `global_obs`, `root_height_obs`만 사용).
  - phase, cmd, motion id, future tar frame 등은 **포함되지 않음**.
- Activation: hidden ReLU, output **linear (logit, sigmoid 없음)** — BCEWithLogitsLoss로 처리.
- **Loss formulation** (`amp_agent.py:162-218`):
  - `BCEWithLogitsLoss(disc_agent_logit, 0)` (agent target 0)
  - `BCEWithLogitsLoss(disc_demo_logit, 1)` (demo target 1)
  - `disc_loss = 0.5 * (loss_agent + loss_demo)` (line 182)
  - **즉 sigmoid cross-entropy (binary, BCE).** Least-squares 아님.
- **Gradient penalty** (line 184-196):
  - `grad_demo = ∂logit_demo / ∂norm_disc_obs_demo`, `grad_agent = ∂logit_agent / ∂norm_disc_obs`
  - `gp = 0.5 * (mean(||grad_demo||²) + mean(||grad_agent||²))`
  - `disc_loss += disc_grad_penalty * gp`, `disc_grad_penalty = 5` (agent yaml:49)
  - **참고**: 원조 AMP 논문은 demo에만 GP를 거는 1-side WGAN-GP 형태이지만, MimicKit은 agent + demo 양측 모두에 GP 적용. (양측 모두 `requires_grad_(True)` line 167, 173)
- **Logit regularization** (line 212-216):
  - `disc_logits.weight` (final linear) L2 → `disc_loss += 0.01 * ||W_logit||²` (`disc_logit_reg: 0.01`, yaml:48)
- **Spectral norm / weight clipping: 없음**
- **Weight decay**: `disc_optimizer.weight_decay: 0.0001` (yaml:24)
- **Optimizer**: `SGD`, lr `2.5e-4` (yaml:21-24)

### 2.4 AMP demo / replay buffer
- **Demo buffer**: 명시적 사전 캐시 없음. 매 train iter마다 `env.fetch_disc_obs_demo(n)` (`amp_env.py:29-32`, `amp_agent.py:91-98`)로 **n = rollout flat size**(= `steps_per_iter * num_envs` = 32 * 4096 = 131072)개의 demo를 motion에서 sampling+forward kinematics로 생성. 매번 새로 추출.
- **Replay buffer**: `ExperienceBuffer(buffer_length=200000, batch_size=1)` (`amp_agent.py:53-55`, agent yaml:46)
  - capacity = 200,000 sample. push 후 FIFO ring.
  - 매 iter `_store_disc_replay_data()` (line 100-115)에서 현재 rollout disc_obs를 random permutation 한 뒤 `min(n, disc_replay_samples=1000)` 개만 push (yaml:47).
- **demo:policy 비율**:
  - `_compute_disc_loss` (line 162-218):
    - `disc_demo_obs = batch["disc_obs_demo"]` — 매 disc step에서 batch_size 만큼 fresh demo
    - `disc_obs = batch["disc_obs"]` (현재 rollout) + replay에서 같은 크기만큼 sample → concat (line 169-171) ⇒ **agent batch = 2 × demo batch** (현재 rollout의 fake + replay의 fake)
  - 즉 demo : (current_policy + replay) = 1 : 2.
- **update 빈도**:
  - PPO/disc update는 매 iter 1회 호출(`_update_model` line 134-146)
  - disc_batch_size = 2 * num_envs = 8192, num_samples per iter = 131072, num_disc_batches = ceil(131072 / 8192) = 16, num_disc_steps = 16 * disc_epochs(=2) = **32 disc step / iter**

### 2.5 Reward 합성
- 위치: `AMPAgent._compute_rewards()` (`amp_agent.py:117-132`), 매 iter `_build_train_data` 안에서 1회 호출 (line 85).
- 수식:
  - `task_r = exp_buffer["reward"]` (env가 기록한 task tracking reward, Section 1.5)
  - `norm_disc_obs = disc_obs_norm.normalize(disc_obs)`
  - `disc_logits = model.eval_disc(norm_disc_obs)` (`amp_model.py:12-15`)
  - `prob = sigmoid(disc_logits) = 1 / (1 + exp(-D))`
  - `disc_r = -log(max(1 - prob, 1e-4)) * disc_reward_scale`, `disc_reward_scale = 2` (yaml:50)
    - **이 형태는 AMP 논문의 style reward `r_s = -log(1 - D(s,s'))`와 동일** (sigmoid를 거친 확률 기반). max-clip 0.0001은 log explosion 방지.
    - **`r_s = max(0, 1 - 0.25*(D-1)²)` 형태는 사용하지 않음** (그것은 ASE/일부 변형 reward).
  - **최종 reward (PPO advantage 계산용)**:
    - `r = task_reward_weight * task_r + disc_reward_weight * disc_r`
    - `task_reward_weight = 0.5`, `disc_reward_weight = 0.5` (yaml:52-53)
    - `exp_buffer["reward"] := r` (in-place 덮어쓰기, line 125)
- 이 합성된 reward가 GAE/TD(λ) 계산(`ppo_agent.py:179`)에 들어가고, value function은 이 합성 reward의 return을 학습.

### 2.6 PPO 하이퍼파라미터 (정확한 숫자)
모두 `data/agents/amp_go2_task_agent.yaml`:

| 항목 | 값 | 출처 |
|---|---|---|
| `discount (γ)` | 0.99 | line 26 |
| `steps_per_iter` (rollout length) | 32 | line 27 |
| `iters_per_output` | 100 | line 28 |
| `test_episodes` | 32 | line 29 |
| `normalizer_samples` | 1e8 | line 30 |
| `td_lambda (GAE λ)` | 0.95 | line 39 |
| `ppo_clip_ratio (ε)` | 0.2 | line 40 |
| `norm_adv_clip` | 4.0 | line 41 |
| `action_bound_weight` | 10.0 | line 42 |
| `action_entropy_weight` | 0.0 | line 43 |
| `action_reg_weight` | 0.0 | line 44 |
| `actor_epochs` | 5 | line 32 |
| `actor_batch_size` | 4 (× num_envs = 16384) | line 33 |
| `critic_epochs` | 2 | line 34 |
| `critic_batch_size` | 2 (× num_envs = 8192) | line 35 |
| `disc_epochs` | 2 | line 36 |
| `disc_batch_size` | 2 (× num_envs = 8192) | line 37 |
| `actor_optimizer` | SGD, lr 2e-4 | line 13-15 |
| `critic_optimizer` | SGD, lr 1e-4 | line 17-19 |
| `disc_optimizer` | SGD, lr 2.5e-4, weight_decay 1e-4 | line 21-24 |
| `disc_buffer_size` | 200000 | line 46 |
| `disc_replay_samples` | 1000 | line 47 |
| `disc_logit_reg` | 0.01 | line 48 |
| `disc_grad_penalty (λ_gp)` | 5 | line 49 |
| `disc_reward_scale` | 2 | line 50 |
| `task_reward_weight` | 0.5 | line 52 |
| `disc_reward_weight` | 0.5 | line 53 |
| `model_checkpoint_interval` | default 1000 | `base_agent.py:165` (override 없음) |
| **`max_grad_norm`** | **명시 안 됨** — `_update_actor/_update_critic`은 grad clipping 호출하지 않음 (`ppo_agent.py:298-324`). `mp_optimizer.MPOptimizer.step()`이 자체적으로 clip을 하는지는 확인 안 됨. |
| **`learning rate schedule`** | **없음** (yaml에 scheduler 키 없음, SGD optimizer는 자체 schedule 없음). `actor_optimizer/critic_optimizer/disc_optimizer` 모두 fixed lr. |
| **`KL early stopping`** | **없음** (`_update_actor`는 num_steps만큼 무조건 반복) |

- Sample budget per iter ≈ `steps_per_iter * num_envs = 32 * 4096 = 131,072` transitions.
- Actor update step 수 = `ceil(131072 / 16384) * 5 = 8 * 5 = 40 steps/iter`
- Critic update step 수 = `ceil(131072 / 8192) * 2 = 16 * 2 = 32 steps/iter`
- Disc update step 수 = `ceil(131072 / 8192) * 2 = 32 steps/iter`

### 2.7 Normalization
- **Obs normalizer** (`base_agent.py:168-193`):
  - `_obs_norm = Normalizer(shape=(93,), clip=10.0)` — running mean/std on full policy obs
  - actor & critic 모두 동일 `_obs_norm.normalize(obs)` 사용 (`ppo_agent.py:120, 327, 342`)
  - enable_priv_encoder=False 이므로 `_proprio_norm/_priv_norm/_history_norm`은 모두 None (line 188-190)
- **Disc obs normalizer** (`amp_agent.py:58-64`):
  - `_disc_obs_norm = Normalizer(shape=(1100,), clip=10.0)` — disc input 별도 정규화
  - demo와 agent disc_obs 모두 normalize 후 disc에 입력 (`amp_agent.py:166, 172`, `_calc_disc_rewards` line 237-244)
  - demo도 normalizer update에 기여 (`_record_disc_demo_data` line 97)
- **Action normalizer** (`base_agent.py:195-216`):
  - init_mean = action bound 중점, init_std = action bound 절반 (즉 `[-1,1]`로 정규화)
  - 학습 동안 update 안 됨 (record/update 호출 없음, init only)
- **Value/return normalization: 없음** (TD(λ) return을 그대로 critic target으로 사용)
- **Advantage normalization** (`ppo_agent.py:189-192`):
  - rand_action_mask 적용된 adv로 mean/std 계산 후 `(adv - mean) / max(std, 1e-5)`
  - 그 다음 `torch.clamp(norm_adv, -4.0, 4.0)` (`norm_adv_clip: 4.0`)

---

## 3. 데이터 흐름 자체 검증

### 3.1 motion → discriminator feature vs policy → discriminator feature
- **동일한 `compute_disc_obs()` 함수 사용** (`amp_env.py:328`). 호출 경로 두 곳:
  - **Demo**: `_compute_disc_obs_demo()` (`amp_env.py:34-61`) — motion_lib에서 `calc_motion_frame(motion_ids, motion_times)` → body_pos는 `kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)`로 계산 (sim 안 거침).
  - **Agent**: `_update_disc_obs()` (`amp_env.py:194-242`) — sim의 `_disc_hist_*` circular buffer에서 가져옴 (실시간 sim state).
- 두 경로 모두 동일한 변수 세트(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos[key_body_ids]) + 동일한 reference frame 처리(window의 last frame root) + 동일한 heading-local 변환을 거침.
- **불일치 우려 지점**:
  - `body_pos`: agent는 sim의 actual body world pos, demo는 motion frame의 forward kinematics 결과. forward_kinematics는 kin_char_model.dof_to_rot / forward_kinematics가 sim과 정확히 일치한다는 가정에 의존. mjcf model이 sim USD와 동일 토폴로지/링크 순서를 가지는지 `_validate_envs()` (`char_env.py:270-278`)에서 link 이름만 비교(순서는 sim2common 매핑으로 재배치).
  - `root_vel`, `root_ang_vel`, `dof_vel`: motion lib에서는 **finite difference**로 계산 (`motion_lib.py:170-181`). 실제 sim의 root_vel은 PhysX가 계산한 ground-truth velocity. fps와 control_freq가 다른 경우 (motion fps는 pkl에 저장된 원본, sim은 50Hz) **속도 magnitude/discrete 차이가 본질적**으로 발생 가능. 확인 안 됨.
  - root_pos는 demo에서 motion의 절대 좌표를 그대로 쓰고, agent는 sim의 (env_offset 뺀) world pos. `compute_tar_obs`는 ref_root_pos(=window last frame root)를 빼므로 절대 좌표 영향은 상쇄됨 (xy만 보고, z는 height obs=False).
- **결론**: feature 정의 자체는 동일. 시간 윈도우(10 step)도 동일 dt(`engine.get_timestep()=0.02`, `amp_env.py:68`)로 backward 샘플. 단 demo의 motion_times는 `motion_times0 - dt*[0..9]`로 backward (`amp_env.py:64-73`) — 즉 마지막 step이 현재 motion_time.

### 3.2 phase / goal 정보가 discriminator에 빠지고 policy에만 들어가는가?
- **Policy**: cmd (lin_x, lin_y, ang_yaw) 3-dim은 들어감. motion id, motion phase, future target frame은 **들어가지 않음** (enable_phase_obs=False, enable_tar_obs=False).
- **Discriminator**: cmd, motion id, phase, future tar 모두 **들어가지 않음**. 순수 (s_t-9 … s_t) 시퀀스만.
- ⇒ **Discriminator는 cmd-agnostic 하게 "natural Go2 locomotion인가?"만 판정**하고, policy는 cmd를 추가로 받아서 cmd 추종 + disc style reward 최대화 양쪽을 동시에 학습.

---

## 4. 불확실 / 미확인 항목

1. **Motion data 통계**: 각 pkl의 frame 수, fps, motion length 합 — pkl 열어보지 않음 (binary). MotionLib 로딩 후 `Loaded {N} motions with total length {T}s` 로그가 찍히지만 실 데이터 미확인.
2. **Retargeting 파이프라인**: `smr_mirror_pkl`이라는 디렉토리 이름과 `_mirror` 파일명 suffix는 좌우 대칭 augmentation을 시사하지만, SMR(?) 원본의 retarget source/방법은 README 참조 외 미검증.
3. **Effort limit**: `ImplicitActuatorCfg(effort_limit=None)` (`isaac_lab_engine.py:803`)이 Isaac Lab에서 USD/MJCF의 `actuatorfrcrange`(hip/thigh ±23.7, calf ±35.5)를 그대로 따르는지, 아니면 무제한인지 확인 안 됨.
4. **`mp_optimizer.MPOptimizer.step()`의 내부 grad clipping**: 코드 직접 미확인. yaml에 `max_grad_norm` 키 없음.
5. **lr scheduler**: yaml에 schedule 없음, 코드에 scheduler 호출 없음. 학습 끝까지 fixed SGD lr로 추정 (확인 안 됨).
6. **action low-pass filter / EMA**: 코드 경로에 없음. 확인 안 됨.
7. **Obs noise**: 호출 없음. 확인 안 됨.
8. **`go2_mimickit_isaac_lab/` 디렉토리** (top-of-repo): MimicKit과 별도 위치한 IsaacLab 호환 환경 사본일 가능성 (이름상). 본 보고서 범위 밖이라 미조사. 참고용으로만 언급.
9. **patch_*.py 파일들** (`patch_agent_cfg.py`, `patch_env_cfg.py`, `patch_isaac_lab.py`, `patch_revert_cfg.py`, `patch_toggle.py`, `PATCH_SUMMARY.md`): 학습 전후 cfg toggle용 헬퍼로 추정. amp_go2_track 학습 직접 호출 경로에는 사용되지 않음 (argfile/agent yaml에 참조 없음). 미조사.
10. **PPO grad flow에서 priv_reg loss 등 RMA 항목**: `enable_priv_encoder=False`이므로 모두 비활성. amp_go2_track**_priv**_args.txt (`amp_go2_task_priv_agent.yaml`, `enable_priv_encoder: True`)는 별도 variant — 본 문서는 non-priv만 다룸.
