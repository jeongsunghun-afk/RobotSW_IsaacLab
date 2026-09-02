# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Imitation Tracking 환경 — body-frame 속도추종 command (vx, vy, yaw_rate).

go2_imitation에서 steering (tar_dir · tar_speed · face_dir) 구조를
body-frame 선속도/각속도 추종(vx, vy=0, yaw_rate)으로 교체한 환경.

  - AMP Discriminator: 49-dim obs × 10 history = 490-dim  (R4: root_rot_tan_norm 6D 추가)
  - Task reward: lin_vel_reward(0.7) + yaw_vel_reward(0.3)
  - Reset: 항상 RSI (Reference State Initialization)
  - 알고리즘: RMA(ActorCriticRMA) + estimator + PPOAMP + OnPolicyRunnerAMP (rsl_rl)

Policy observation dict (실배포 가능 RMA 구조 — estimator가 policy(42)로부터 root 선속도·각속도를
추정하고, priv_explicit(6)는 학습 시 GT critic/estimator target 용):
  policy(42)        = projected_gravity_b(3) +
                       lin_vel_cmd(2) + yaw_vel_cmd(1) +
                       joint_pos - default(12) + joint_vel(12) + actions(12)
  ⚠ `joint_pos_tan_norm=True` 면 관절 블록이 12 → 72(관절별 회전 tan-norm) 로 늘어
    policy 가 **102**, history 가 (10, 102) 가 된다. §16 의 MimicKit 대조 arm 이다.
  priv_explicit(6)  = root_lin_vel_b * priv_explicit_lin_vel_scale +
                       root_ang_vel_b * priv_explicit_ang_vel_scale

  priv_explicit로 분리한 신호는 policy obs에서 제외한다 — estimator의 추정 대상을 actor에게
  직접 보여주면 추정 구조가 무의미해진다. root_ang_vel_b는 실기 IMU로 측정 가능하지만
  이 설계 일관성을 위해 obs에서 뺐다(2026-07-30 변경).
  priv_latent(19)   = armature_scale(1) + joint_friction_scale(1) + base_mass_offset(1) +
                       foot_friction_offset(1) + kp_scale(1) + kd_scale(1) +
                       action_delay_norm(1) + encoder_bias_norm(12)
  history(10, 42)   = policy proprio ring buffer (noised)

[R4 ablation] MimicKit compute_tar_obs 방식 이식:
  - 각 disc window frame의 root_quat을 window[-1](현재) frame의 heading-inv 기준 local로 변환
  - 변환식: relative_quat = quat_mul(heading_inv_of_ref, frame_quat)
    (MimicKit deepmimic_env.py:733,742 — heading-only inverse, 전체 quat inverse 아님)
  - 변환된 quat → quat_to_tan_norm: [tan(x), norm(z)] 6D feature
    (MimicKit torch_util.py:218-224 — ref_tan=[1,0,0], ref_norm=[0,0,1])
  - per-step disc feature: 43 → 49 (base_43 + root_rot_tan_norm_6)
"""

from __future__ import annotations

import glob
import math
import os

import gymnasium as gym
import numpy as np
import torch
import warp as wp  # IsaacLab 3.0: _ALL_INDICES is a warp array → wp.to_torch()

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_inverse, quat_mul

from isaaclab_tasks.direct.amp_command_condition import AMPCommandConditionMixin

from .go2_imitation_tracking_env_cfg import (
    PACE_ARMATURE,
    PACE_COULOMB,
    PACE_ENCODER_BIAS_MAG,
    PACE_VISCOUS,
    Go2ImitationTrackingEnvCfg,
)
from .motion_lib import Go2MotionLib


class Go2ImitationTrackingEnv(AMPCommandConditionMixin, DirectRLEnv):
    """Go2 AMP + body-frame 속도추종 Imitation 환경 (RMA + estimator 아키텍처).

    Policy proprio (42-dim — priv_explicit로 분리한 root 속도 항은 제외):
        projected_gravity_b(3) +
        lin_vel_cmd(2) + yaw_vel_cmd(1) +
        joint_pos - default(12) + joint_vel(12) + actions(12)

    priv_explicit(6-dim, 노이즈 없는 GT — 학습 시 critic 입력 / estimator target):
        root_lin_vel_b * priv_explicit_lin_vel_scale
        root_ang_vel_b * priv_explicit_ang_vel_scale
        둘 다 policy obs에 없으므로 estimator는 **미관측 상태 추정**을 학습한다
        (각속도가 obs에 있던 이전 버전의 denoising 과제가 아니다).

    priv_latent(19-dim, quasi-static domain-rand 파라미터 — RMA priv encoder 입력):
        armature_scale(1) + joint_friction_scale(1) + base_mass_offset(1) +
        foot_friction_offset(1) + kp_scale(1) + kd_scale(1) +
        action_delay_norm(1) + encoder_bias_norm(12)

    history(10, 42): policy proprio ring buffer (노이즈 포함, 최신이 index 0).

    AMP disc observation (49-dim per step):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [R4: heading-relative 6D rotation]
    """

    cfg: Go2ImitationTrackingEnvCfg

    # key body 이름 (발 4개) — motion_lib foot 순서와 일치
    KEY_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    #: 관절 회전축 — 이름 접미사로 고른다. Go2 는 hip 이 x, thigh/calf 가 y 축이다
    #: (MimicKit `data/assets/go2/go2.xml` 의 12 개 `<joint axis=...>` 로 확인).
    #: tan-norm 인코딩에서 축 선택은 **어느 성분이 상수가 되느냐**만 바꾼다 — 정보량은
    #: 어느 축을 쓰든 (cos θ, sin θ) 로 같고 첫 선형층이 고정 재배치를 흡수한다.
    JOINT_AXIS_BY_SUFFIX = {"hip": (1.0, 0.0, 0.0), "thigh": (0.0, 1.0, 0.0), "calf": (0.0, 1.0, 0.0)}

    def __init__(self, cfg: Go2ImitationTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        # policy obs 폭은 관절 인코딩에 달려 있다. `super().__init__` 이 `cfg.observation_space` 로
        # gym space 를 만들므로 **그 전에** 확정해야 하고, hydra 오버라이드는 env 생성 시점엔 이미
        # cfg 에 반영돼 있으므로 여기가 유일하게 맞는 자리다(cfg `__post_init__` 은 너무 이르다).
        # ⚠ 아래 두 키는 **파생값이다** — 여기서 무조건 덮어쓰므로 hydra 로 직접 오버라이드해도
        #   조용히 무시된다. 폭을 바꾸려면 `*_tan_norm` 플래그를 쓸 것.
        joint_pos_obs_dim = 12 * 6 if cfg.joint_pos_tan_norm else 12
        cfg.observation_space = 3 + 2 + 1 + joint_pos_obs_dim + 12 + 12
        # disc obs 도 같은 이유로 여기서 확정한다. base(43 또는 103) + root_rot_tan_norm(6).
        amp_joint_dim = 12 * 6 if cfg.amp_joint_tan_norm else 12
        cfg.amp_observation_space = amp_joint_dim + 12 + 1 + 3 + 3 + 12 + 6

        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg

        # ── policy proprio 레이아웃 ──────────────────────────────
        # `_apply_obs_dr` 는 인덱스를 하드코딩하지 않고 이 값에서 유도한다. 예전에 각속도를
        # 뺐을 때 슬라이스만 밀린 채 남겨두면 σ 가 4 배로 잘못 주입되는 사고가 있었다.
        self._joint_pos_obs_dim = joint_pos_obs_dim
        self._obs_idx_joint_vel = 6 + joint_pos_obs_dim  # joint_vel 블록 시작
        self._obs_idx_actions = self._obs_idx_joint_vel + 12

        # ── 모션 라이브러리 ─────────────────────────────────────
        motion_file = self.cfg.motion_file
        if os.path.isdir(motion_file):
            pkl_files = sorted(glob.glob(os.path.join(motion_file, "*.pkl")))
            assert pkl_files, f"PKL 파일 없음: {motion_file}"
            motion_files = pkl_files
        else:
            motion_files = [motion_file]
        # `weights=None` 이면 MotionLib 이 **길이 비례**로 뽑는다. MimicKit 은 dataset YAML 에서
        # 전 클립에 `weight: 1.0` 을 줘 **클립 균등**으로 뽑으므로, 대조하려면 여기서 넘겨야 한다
        # (`motion_uniform_weights` docstring 참조 — 고속 클립 `go2_run2` 가 2.09 배 차이난다).
        motion_weights = [1.0] * len(motion_files) if self.cfg.motion_uniform_weights else None
        self._motion_lib = Go2MotionLib(motion_files=motion_files, device=self.device, weights=motion_weights)

        # ── body / joint 인덱스 ──────────────────────────────────
        self.ref_body_index = self._robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self._robot.data.body_names.index(name) for name in self.KEY_BODY_NAMES]

        # hip(abduction) 관절 인덱스 — 액션 범위 축소용
        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(self._robot.data.joint_names) if "hip" in n], dtype=torch.long, device=self.device
        )

        # ── tan-norm 인코딩용 관절 회전축 [12, 3] ────────────────
        # policy(`_joint_axis`)와 disc(`_amp_joint_axis`)가 따로 켜질 수 있어 둘로 나눈다.
        # 꺼진 쪽은 **빈 텐서**다 — `_compute_amp_obs` 는 jit 함수라 Optional 을 못 받으므로
        # `numel() == 0` 을 off 신호로 쓴다.
        axes = []
        for name in self._robot.data.joint_names:
            match = [ax for suffix, ax in self.JOINT_AXIS_BY_SUFFIX.items() if suffix in name]
            if len(match) != 1:
                raise ValueError(
                    f"관절 '{name}' 의 회전축을 정할 수 없다 (매칭 {len(match)} 건). "
                    f"`JOINT_AXIS_BY_SUFFIX` 에 접미사를 추가할 것."
                )
            axes.append(match[0])
        _axis = torch.tensor(axes, dtype=torch.float32, device=self.device)  # [12, 3]
        _empty = torch.zeros(0, 3, dtype=torch.float32, device=self.device)
        self._joint_axis = _axis if self.cfg.joint_pos_tan_norm else _empty
        self._amp_joint_axis = _axis if self.cfg.amp_joint_tan_norm else _empty

        # ── motion_lib ↔ IsaacLab joint 순서 매핑 ────────────────
        # IsaacLab joint 순서(알파벳 등)와 PKL DOF_NAMES 순서가 다를 수 있음.
        # go2_amp_env.py의 motion_dof_indexes 패턴과 동일.
        robot_joint_names = list(self._robot.data.joint_names)
        try:
            self._motion_dof_indices = self._motion_lib.get_dof_index(robot_joint_names)
            print(f"[Go2ImitationTrackingEnv] IsaacLab joint 순서: {robot_joint_names}")
            print(f"[Go2ImitationTrackingEnv] motion_dof_indices: {self._motion_dof_indices}")
        except AssertionError:
            print("[Go2ImitationTrackingEnv] DOF 이름 불일치 — 순서대로 1:1 매핑 사용")
            self._motion_dof_indices = list(range(12))

        # ── Velocity tracking command state ─────────────────────
        self._lin_vel_cmd = torch.zeros(self.num_envs, 2, device=self.device)  # (vx, vy) — vy 항상 0
        # ④ 정지 명령으로 강제된 env (`rel_standing_envs`). `_resample_steering` 이 갱신한다.
        self._standing_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._yaw_vel_cmd = torch.zeros(self.num_envs, device=self.device)  # yaw rate [rad/s]
        self._tar_timer = torch.zeros(self.num_envs, device=self.device)  # [N] seconds to change

        # ── AMP 관측 버퍼 ─────────────────────────────────────────
        # amp_observation_space (cfg=49) = base(43) + root_rot_tan_norm(6)  [R4]
        # amp_observation_buffer 내부 저장은 base 43-dim만; tan_norm 6D는 소비 시점에 계산됨.
        # 내부 버퍼 차원 = per-step disc obs 에서 root_rot_tan_norm(6)을 뺀 것.
        # tan-norm 이면 103, 아니면 43. cfg 값에서 유도해 두 상수가 갈라지지 않게 한다.
        _AMP_BASE_DIM = self.cfg.amp_observation_space - 6
        self.amp_observation_size = self.cfg.num_amp_observations * (
            self.cfg.amp_observation_space + (2 if self.cfg.include_rel_track_obs else 0)
        )
        # 조건부 disc: 조건 열(+valid) 을 obs 끝에 붙인다 (mode="none" 이면 0).
        self.amp_observation_size += self._init_amp_condition()
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, _AMP_BASE_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        # [R4] per-step root_quat history buffer (xyzw, identity-initialized).
        # Kept separate from amp_observation_buffer so base 43-dim stays untouched.
        # The heading-relative tan_norm 6D is computed at consumption time.
        self._amp_quat_buf = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, 4),
            dtype=torch.float32,
            device=self.device,
        )
        self._amp_quat_buf[..., 3] = 1.0  # identity quat xyzw: w=1

        if self.cfg.include_rel_track_obs:
            self._hist_root_pos_w = torch.zeros(
                (self.num_envs, self.cfg.num_amp_observations, 3),
                dtype=torch.float32,
                device=self.device,
            )

        # ── Terminal amp obs 버퍼 ─────────────────────────────────
        # _reset_idx() 진입 시점(RSI 이전)의 실패 에피소드 마지막 상태를 보존.
        # 러너에서 terminal step의 amp_reward 계산 시 post-reset RSI obs 대신 이 값을 사용.
        self._terminal_amp_obs = torch.zeros(
            (self.num_envs, self.amp_observation_size),
            dtype=torch.float32,
            device=self.device,
        )

        # ── PACE 식별 파라미터 + Domain Randomization 초기화 ─────
        self._init_domain_rand()

        # ── RMA proprio history 링버퍼 (policy obs 42-dim × history_len) ────
        self._proprio_history = torch.zeros(
            self.num_envs, self.cfg.history_len, self.cfg.observation_space, device=self.device
        )

    # ──────────────────────────────────────────────────────────
    # IsaacLab DirectRLEnv 필수 메서드
    # ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        # Env isolation: filter cross-env collisions unconditionally (GPU too). In IsaacLab 3.0
        # the auto-filter path (interactive_scene:218) is skipped when the scene cfg declares no
        # entities (has_scene_cfg_entities=False), so the old cpu-only guard left GPU runs
        # unfiltered — robots from different envs physically collide. Ref: IsaacLab #1918.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self._episode_sums = {
            "lin_vel_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "yaw_vel_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        if self.cfg.standing_style_substitute:
            # ④ 진단용 — **보상이 아니다.** style 대체가 실제로 물리는지 학습 중에 보려면 필요하다
            # (pose 보상은 extras 경로라 `Episode_Reward/*` 에 안 잡힌다). 판정은
            #   standing_base_h / standing_frac = 정지 중 평균 base 높이
            # 로 하며, 0.22(웅크림) → 0.32(default 기립) 로 올라가야 기전이 작동한 것이다.
            for _k in ("standing_pose_reward", "standing_frac", "standing_base_h"):
                self._episode_sums[_k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        actions = self._actions.clone()
        if self.cfg.hip_scale_reduction:
            actions[:, self._hip_joint_ids] *= 0.5
        target = self.cfg.action_scale * actions + self._robot.data.default_joint_pos
        # DR: action(토크 명령) 지연 — per-env 지연 스텝만큼 과거 target 을 적용
        if self.cfg.domain_rand and self.cfg.dr.randomize_action_delay:
            self._action_delay_buf = torch.cat([target.unsqueeze(1), self._action_delay_buf[:, :-1]], dim=1)
            rows = torch.arange(self.num_envs, device=self.device)
            self._processed_actions = self._action_delay_buf[rows, self._action_delay_steps]
        else:
            self._processed_actions = target

    def _post_physics_step(self):
        # 목표 속도 타이머 업데이트
        self._tar_timer -= self.step_dt
        change_mask = self._tar_timer <= 0.0
        if change_mask.any():
            change_ids = change_mask.nonzero(as_tuple=False).flatten()
            self._resample_steering(change_ids)

        # DR: 주기적 외란 push (base 수평 속도 킥)
        if self.cfg.domain_rand and self.cfg.dr.push_robot:
            self._push_timer -= self.step_dt
            push_ids = (self._push_timer <= 0.0).nonzero(as_tuple=False).flatten()
            if push_ids.numel() > 0:
                self._push_robots(push_ids)
                self._push_timer[push_ids] = self.cfg.dr.push_interval_s

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        root_pos_w = self._robot.data.body_pos_w[:, self.ref_body_index]  # [N,3]
        # IsaacLab 3.0+: body_quat_w is (x,y,z,w). Heading helpers and _amp_quat_buf are
        # now fully xyzw-aware; motion_lib quats (wxyz) are converted at the consumption
        # boundary inside _compute_reference_buffers and _reset_strategy_rsi.
        root_quat_w = self._robot.data.body_quat_w[:, self.ref_body_index]  # [N,4] xyzw
        # IsaacLab 3.0: ArticulationData props return ProxyArray; .torch needed at
        # @torch.jit.script boundaries (_compute_amp_obs). Indexed accessors above
        # (body_pos_w[:, i], body_quat_w[:, i]) already unwrap to torch via __getitem__.
        root_lin_vel_b = self._robot.data.root_lin_vel_b.torch  # [N,3]
        root_ang_vel_b = self._robot.data.root_ang_vel_b.torch  # [N,3]

        # ── 발 위치 (local frame) ─────────────────────────────
        rel_pos = self._robot.data.body_pos_w[:, self.key_body_indexes] - root_pos_w.unsqueeze(1)
        N, K = rel_pos.shape[:2]
        local_key_body_pos = quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
            rel_pos.reshape(-1, 3),
        ).view(N, K, 3)

        # ── AMP disc 관측 계산 및 버퍼 업데이트 ─────────────────
        amp_obs_step = _compute_amp_obs(
            self._robot.data.joint_pos.torch,
            self._robot.data.joint_vel.torch,
            root_pos_w,
            root_lin_vel_b,
            root_ang_vel_b,
            local_key_body_pos,
            self._amp_joint_axis,
        )
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
            self._amp_quat_buf[:, i + 1] = self._amp_quat_buf[:, i]  # [R4]
            if self.cfg.include_rel_track_obs:
                self._hist_root_pos_w[:, i + 1] = self._hist_root_pos_w[:, i]

        self.amp_observation_buffer[:, 0] = amp_obs_step.clone()
        self._amp_quat_buf[:, 0] = root_quat_w.clone()  # [R4] xyzw (from body_quat_w)

        if self.cfg.include_rel_track_obs:
            self._hist_root_pos_w[:, 0] = root_pos_w.clone()

            # ── 상대 위치 궤적 피처 추가 (MimicKit 방식) ───────────
            N, H = self.num_envs, self.cfg.num_amp_observations
            heading_rot_inv = _calc_heading_quat_inv(root_quat_w)  # [N, 4]
            root_pos_diff = self._hist_root_pos_w - root_pos_w.unsqueeze(1)  # [N, H, 3]

            heading_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, H, -1).reshape(N * H, 4)
            local_root_pos_diff = quat_apply(heading_inv_expand, root_pos_diff.reshape(N * H, 3)).view(N, H, 3)
            root_pos_obs_xy = local_root_pos_diff[:, :, :2]  # [N, H, 2]

            base_amp = torch.cat([root_pos_obs_xy, self.amp_observation_buffer], dim=-1)  # [N, H, 45]
        else:
            N, H = self.num_envs, self.cfg.num_amp_observations
            base_amp = self.amp_observation_buffer  # [N, H, 43]

        # [R4] heading-relative root_rot → tan_norm 6D (MimicKit compute_tar_obs 방식)
        # ref = 현재(window 첫 번째=최신) frame의 heading_inv
        # 각 window frame quat을 heading_inv 기준 local로 변환 후 tan_norm 적용
        rot_tan_norm = _apply_root_rot_tan_norm(self._amp_quat_buf, N, H)  # [N, H, 6]
        final_amp_obs = torch.cat([base_amp, rot_tan_norm], dim=-1)  # [N, H, 49]

        self.extras = {
            "amp_obs": self._append_amp_cond(final_amp_obs.view(self.num_envs, -1), self._policy_amp_cond()).clone(),
            # terminal step에서 러너가 post-reset RSI obs 대신 이 값으로 amp_reward 계산
            "terminal_amp_obs": self._terminal_amp_obs.clone(),
        }

        # ④ 정지 구간 style 대체 — 러너의 보상 융합에 쓰인다:
        #     total = lerp·task + (1−lerp)·(style_weight·amp + (1−style_weight)·style_substitute)
        # 두 키가 없으면 러너는 style_weight=1 / substitute=0 으로 보므로 다른 AMP task 는 불변.
        if self.cfg.standing_style_substitute:
            _pose_r = self._compute_standing_pose_reward()
            self.extras["style_weight"] = (~self._standing_mask).float()
            self.extras["style_substitute"] = _pose_r
            # 진단 누적. 여기서 하는 이유: `_reset_idx`(누적 초기화)가 이 시점보다 먼저 돌므로
            # 리셋된 env 는 새 에피소드분부터 쌓인다. `standing_base_h / standing_frac` 가 정지 중
            # 평균 base 높이다(둘 다 같은 상수로 정규화되므로 비율은 그대로 [m]).
            _stand_f = self._standing_mask.float()
            self._episode_sums["standing_pose_reward"] += _pose_r
            self._episode_sums["standing_frac"] += _stand_f
            self._episode_sums["standing_base_h"] += _stand_f * root_pos_w[:, 2]

        # ── Policy proprio (42-dim) ────────────────────────────────────────────
        # priv_explicit로 분리된 신호는 policy obs에서 **제외**한다 — estimator가 추정하는
        # 대상을 actor에게 직접 보여주면 추정 구조가 무의미해지기 때문이다.
        # root_lin_vel_b(실측 불가)와 root_ang_vel_b(priv_explicit로 이동)가 모두 빠졌다.
        # 관절각 노이즈(`joint_pos_noise`)와 엔코더 오프셋(`encoder_bias`)은 **라디안 공간에서**
        # 더한 뒤 인코딩한다. tan-norm 출력에 라디안을 더하면 물리적 의미가 사라지기 때문이다.
        # raw 경로에서는 인코딩이 항등이라 예전처럼 블록에 더하는 것과 bit-identical 하다.
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos  # [N, 12] rad
        if self.cfg.domain_rand:
            if self.cfg.dr.obs_noise:
                joint_pos_rel = joint_pos_rel + torch.randn_like(joint_pos_rel) * self.cfg.dr.joint_pos_noise
            if self.cfg.dr.encoder_bias:
                joint_pos_rel = joint_pos_rel + self._encoder_bias

        proprio = torch.cat(
            [
                self._robot.data.projected_gravity_b,  # 3
                self._lin_vel_cmd,  # 2 (vx, vy)
                self._yaw_vel_cmd.unsqueeze(-1),  # 1
                self._encode_joint_pos(joint_pos_rel),  # 12 (raw) 또는 72 (tan-norm)
                self._robot.data.joint_vel,  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )  # total = 42 (raw) 또는 102 (tan-norm)
        policy_obs = self._apply_obs_dr(proprio)

        # ── priv_explicit(6) — GT root 선속도/각속도 (critic/estimator target, 노이즈 없음) ──
        # 두 블록 모두 policy obs에 없다 → estimator는 proprio(42)+history로부터
        # **미관측 상태 추정**을 학습하고, critic은 GT를 본다(asymmetric actor-critic).
        priv_explicit = torch.cat(
            [
                root_lin_vel_b * self.cfg.priv_explicit_lin_vel_scale,  # 3
                root_ang_vel_b * self.cfg.priv_explicit_ang_vel_scale,  # 3
            ],
            dim=-1,
        )  # total = 6

        # ── priv_latent(19) — quasi-static domain-rand 파라미터 ──
        priv_latent = self._get_priv_latent()

        # ── proprio history 링버퍼 갱신 (노이즈 포함된 policy_obs 저장) ──
        self._proprio_history = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([policy_obs] * self.cfg.history_len, dim=1),
            torch.cat([self._proprio_history[:, 1:], policy_obs.unsqueeze(1)], dim=1),
        )

        return {
            "policy": policy_obs,
            "priv_explicit": priv_explicit,
            "priv_latent": priv_latent,
            "history": self._proprio_history,
        }

    def _compute_standing_pose_reward(self) -> torch.Tensor:
        """정지 명령 env 에서 style reward 를 대신할 **default pose 유지 보상**.

        ``r = w · exp(−scale · mean((q − q_default)²))``, 정지 env 만 값이 있고 나머지는 0 이다.

        왜 style 을 대체하는가: cmd 0 구간에서 discriminator 가 **웅크린 정지를 빠른 보행보다
        참조답다고 평가**해(0.904 vs 0.735) style 이 흡수 상태를 강화하고 있었다. 참조 18클립이
        전부 걷는 중이라 disc 를 고칠 데이터가 없으므로, 그 구간만 튜닝 가능한 신호로 바꾼다.

        기체 높이 목표를 따로 두지 않는다 — default 관절각이 서 있는 높이를 물리적으로 결정하고,
        관측된 실패("관절이 굽어 base_h 0.217")가 곧 관절각 오차이기 때문이다.
        """
        err = torch.mean((self._robot.data.joint_pos - self._robot.data.default_joint_pos) ** 2, dim=-1)
        r = self.cfg.standing_pose_reward_w * torch.exp(-self.cfg.standing_pose_err_scale * err)
        return torch.where(self._standing_mask, r, torch.zeros_like(r))

    def _get_rewards(self) -> torch.Tensor:
        # ── (1) lin_vel tracking — body frame 직접 비교 ──────────
        lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]  # (vx, vy) actual
        lin_vel_err = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
        lin_vel_reward = torch.exp(-self.cfg.vel_err_scale * lin_vel_err)

        # ── (2) yaw_vel tracking ─────────────────────────────────
        yaw_vel_b = self._robot.data.root_ang_vel_b[:, 2]
        yaw_vel_err = (self._yaw_vel_cmd - yaw_vel_b) ** 2
        yaw_vel_reward = torch.exp(-self.cfg.yaw_vel_err_scale * yaw_vel_err)

        reward = self.cfg.lin_vel_reward_w * lin_vel_reward + self.cfg.yaw_vel_reward_w * yaw_vel_reward

        self._episode_sums["lin_vel_reward"] += lin_vel_reward
        self._episode_sums["yaw_vel_reward"] += yaw_vel_reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ★ `_post_physics_step` 은 `DirectRLEnv` 의 훅이 **아니다** — 정의만 해 두면 아무도 부르지
        #   않는다. 6.0 마이그레이션에서 이 사실이 누락돼 명령 재샘플과 push 외란이 **한 번도 돌지
        #   않았다**(`tar_change_time_*` 2~7 s 는 dead config 였고, 명령은 20 s 에피소드 내내 고정).
        #   `_get_dones` 가 물리 step 직후 처음 불리는 훅이라 여기에 건다 —
        #   `go2_pedipulation_env.py` 와 `skeleton_wtw_env.py` 가 쓰는 것과 같은 우회다.
        #
        # ⚠ 되살리면 **push 외란도 같이 살아난다**(같은 메서드 안에 있다). 그것까지 한꺼번에 켜면
        #   변수가 둘이 되므로, 명령 축만 보려면 실행 시 `env.dr.push_robot=false` 를 명시할 것.
        #   기존 `nopush_stock` arm 의 "무효과" 는 push-off 를 **push-없음**과 비교한 것이었다.
        self._post_physics_step()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            base_height = self._robot.data.body_pos_w[:, self.ref_body_index, 2]
            died = base_height < self.cfg.termination_height

            flipped = self._robot.data.projected_gravity_b[:, 2] > 0.0
            died = died | flipped

            grav_b = self._robot.data.projected_gravity_b
            roll = torch.atan2(grav_b[:, 1], -grav_b[:, 2])
            pitch = torch.atan2(-grav_b[:, 0], torch.sqrt(grav_b[:, 1] ** 2 + grav_b[:, 2] ** 2))
            roll_limit = self.cfg.roll_termination_deg * math.pi / 180.0
            pitch_limit = self.cfg.pitch_termination_deg * math.pi / 180.0
            bad_orientation = (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)
            died = died | bad_orientation

            if hasattr(self, "contact_sensor"):
                contact_forces = self.contact_sensor.data.net_forces_w
                if contact_forces is not None and contact_forces.numel() > 0:
                    bad_contacts = self._get_body_contact(
                        contact_forces,
                        self._robot.data.body_names,
                        "base",
                        threshold=self.cfg.contact_force_threshold,
                    )
                    died = died | bad_contacts

            # 첫 스텝 직후부터만 termination 적용 (physics 정착 시간)
            died = died & (self.episode_length_buf > 1)
        else:
            died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)  # IsaacLab 3.0: _ALL_INDICES is wp.array

        # ── Terminal amp_obs 캡처 (RSI/reset 이전 — 로봇이 아직 실패 에피소드 상태) ──
        # robot.data는 마지막 physics step 결과를 보유. _robot.reset() 전에 읽어야 함.
        n = len(env_ids)  # type: ignore[arg-type]
        root_pos_w_t = self._robot.data.body_pos_w[env_ids][:, self.ref_body_index]  # [n,3]
        root_quat_w_t = self._robot.data.body_quat_w[env_ids][:, self.ref_body_index]  # [n,4]
        root_lin_vel_b_t = self._robot.data.root_lin_vel_b[env_ids]  # [n,3]
        root_ang_vel_b_t = self._robot.data.root_ang_vel_b[env_ids]  # [n,3]
        rel_pos_t = self._robot.data.body_pos_w[env_ids][:, self.key_body_indexes] - root_pos_w_t.unsqueeze(
            1
        )  # [n,K,3]
        K = rel_pos_t.shape[1]
        local_key_body_pos_t = quat_apply_inverse(
            root_quat_w_t.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
            rel_pos_t.reshape(-1, 3),
        ).view(n, K, 3)
        amp_obs_terminal_step = _compute_amp_obs(
            self._robot.data.joint_pos[env_ids],
            self._robot.data.joint_vel[env_ids],
            root_pos_w_t,
            root_lin_vel_b_t,
            root_ang_vel_b_t,
            local_key_body_pos_t,
            self._amp_joint_axis,
        )  # [n, amp_obs_space=43]
        # 히스토리를 terminal state 기준으로 구성 (_get_observations와 동일한 시프트)
        terminal_buf = self.amp_observation_buffer[env_ids].clone()  # [n, H, 43]
        terminal_quat_buf = self._amp_quat_buf[env_ids].clone()  # [n, H, 4]  [R4]
        if self.cfg.include_rel_track_obs:
            terminal_hist_pos = self._hist_root_pos_w[env_ids].clone()  # [n, H, 3]

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            terminal_buf[:, i + 1] = terminal_buf[:, i]
            terminal_quat_buf[:, i + 1] = terminal_quat_buf[:, i]  # [R4]
            if self.cfg.include_rel_track_obs:
                terminal_hist_pos[:, i + 1] = terminal_hist_pos[:, i]

        terminal_buf[:, 0] = amp_obs_terminal_step
        terminal_quat_buf[:, 0] = root_quat_w_t  # [R4]

        if self.cfg.include_rel_track_obs:
            terminal_hist_pos[:, 0] = root_pos_w_t

            H = self.cfg.num_amp_observations
            heading_rot_inv_t = _calc_heading_quat_inv(root_quat_w_t)
            root_pos_diff_t = terminal_hist_pos - root_pos_w_t.unsqueeze(1)
            heading_inv_expand_t = heading_rot_inv_t.unsqueeze(1).expand(-1, H, -1).reshape(n * H, 4)
            local_root_pos_diff_t = quat_apply(heading_inv_expand_t, root_pos_diff_t.reshape(n * H, 3)).view(n, H, 3)
            root_pos_obs_xy_t = local_root_pos_diff_t[:, :, :2]

            base_terminal = torch.cat([root_pos_obs_xy_t, terminal_buf], dim=-1)  # [n, H, 45]
        else:
            base_terminal = terminal_buf  # [n, H, 43]

        # [R4] 동일한 tan_norm 변환 적용 (live path와 대칭)
        rot_tan_norm_t = _apply_root_rot_tan_norm(terminal_quat_buf, n, self.cfg.num_amp_observations)  # [n, H, 6]
        final_terminal_amp_obs = torch.cat([base_terminal, rot_tan_norm_t], dim=-1)  # [n, H, 49]

        self._terminal_amp_obs[env_ids] = self._append_amp_cond(
            final_terminal_amp_obs.view(n, -1), self._policy_amp_cond(env_ids)
        )
        # ────────────────────────────────────────────────────────────────────────

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 항상 RSI
        root_state, joint_pos, joint_vel = self._reset_strategy_rsi(env_ids)  # type: ignore[arg-type]

        # IsaacLab 3.0: partial-env writes use the *_index API with keyword args.
        self._robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)

        # 속도 명령 재샘플링 (버퍼 초기화 — 불변규칙 §3)
        self._resample_steering(env_ids)  # type: ignore[arg-type]

        # RMA proprio history 리셋 (실질적인 재구성은 다음 _get_observations의
        # episode_length_buf<=1 분기에서 현재 proprio로 복제되지만, 방어적으로 0 초기화)
        self._proprio_history[env_ids] = 0.0

        # NOTE: amp_observation_buffer는 _reset_strategy_rsi 내부에서 RSI 데이터로 채워짐.
        # 여기서 0으로 덮어쓰지 않음 — 덮어쓰면 RSI 효과가 완전히 사라짐.

        # Episode 통계 리셋 및 로깅
        extras: dict = {}
        for key in self._episode_sums:
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])  # type: ignore[index]
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0  # type: ignore[index]

        if "log" not in self.extras:
            self.extras["log"] = {}  # type: ignore[assignment]
        self.extras["log"].update(extras)  # type: ignore[union-attr]

        # PACE 플랜트 반영 + per-env Domain Randomization 재샘플
        self._apply_domain_rand(env_ids)  # type: ignore[arg-type]

    # ──────────────────────────────────────────────────────────
    # PACE 파라미터 + Domain Randomization
    # ──────────────────────────────────────────────────────────

    def _init_domain_rand(self):
        """PACE nominal 플랜트 텐서 + DR per-env 상태 버퍼 초기화.

        관절 타입(hip/thigh/calf)별 식별값을 robot joint order 로 펼쳐두고, per-env 스케일·
        encoder bias·action delay·push 타이머 버퍼를 만든다. 실제 sim write 는 리셋마다
        :meth:`_apply_domain_rand` 에서 수행한다 (각 env = 서로 다른 로봇).
        """
        device = self.device
        names = list(self._robot.data.joint_names)
        nj = self._robot.num_joints

        # PACE nominal (robot joint order). 미식별 관절은 nominal(armature 0.01, 마찰 0) 유지.
        self._pace_armature = torch.full((nj,), 0.01, device=device)
        self._pace_viscous = torch.zeros(nj, device=device)
        self._pace_coulomb = torch.zeros(nj, device=device)
        for key in ("hip", "thigh", "calf"):
            ids = [i for i, n in enumerate(names) if n.endswith(f"_{key}_joint")]
            self._pace_armature[ids] = PACE_ARMATURE[key]
            self._pace_viscous[ids] = PACE_VISCOUS[key]
            self._pace_coulomb[ids] = PACE_COULOMB[key]
        self._all_joint_ids = torch.arange(nj, dtype=torch.int32, device=device)

        # 게인 원본 (DCMotor 는 actuator.stiffness/damping 텐서로 토크를 계산 → 이 텐서를 직접 스케일)
        # calf 는 effort_limit 이 달라 별도 actuator 그룹이므로 **전 그룹**을 돌아야 한다.
        # 한 그룹만 잡으면 게인 DR 이 calf 에 걸리지 않는다.
        self._acts = list(self._robot.actuators.values())
        self._base_kp = [a.stiffness.clone() for a in self._acts]
        self._base_kd = [a.damping.clone() for a in self._acts]
        self._act = self._acts[0]  # 하위호환: 단일 그룹을 가정하던 외부 도구용

        # per-env DR 상태
        self._encoder_bias = torch.zeros(self.num_envs, 12, device=device)
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        d_cap = max(1, int(self.cfg.dr.max_action_delay_steps))
        self._action_delay_buf = self._robot.data.default_joint_pos.unsqueeze(1).repeat(1, d_cap + 1, 1).clone()
        self._push_timer = torch.full((self.num_envs,), self.cfg.dr.push_interval_s, device=device)

        # mass DR 기준값 (base 링크 원본 질량)
        self._base_body_id = self._robot.data.body_names.index("base")
        _bid = self._base_body_id
        self._default_base_mass = self._robot.data.body_mass.torch[:, _bid : _bid + 1].clone()

        # ── RMA priv_latent 버퍼 (per-env, quasi-static DR 파라미터) ────────────
        # nominal 값(해당 DR 항목이 off 이거나 아직 샘플되지 않은 env)은 항상 이 초기값 유지:
        # scale류 = 1.0(무작위화 없음), offset류 = 0.0(nominal 대비 편차 없음).
        # ActorCriticRMA는 priv_latent 키를 무조건 요구하므로 DR off 여도 항상 내보내야 함.
        self._priv_armature_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_joint_friction_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_base_mass_offset = torch.zeros(self.num_envs, 1, device=device)  # [kg]
        self._priv_foot_friction_offset = torch.zeros(self.num_envs, 1, device=device)  # nominal(1.0) 대비 오프셋
        self._priv_kp_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_kd_scale = torch.ones(self.num_envs, 1, device=device)

        self._dr_ready = False  # 첫 전체 reset 에서 startup material DR 적용 후 True

    def _sample(self, rng: tuple[float, float], n: int, d: int) -> torch.Tensor:
        """[n, d] uniform 샘플 (rng=(lo, hi))."""
        lo, hi = rng
        return torch.rand(n, d, device=self.device) * (hi - lo) + lo

    def _apply_domain_rand(self, env_ids: torch.Tensor):
        """PACE 플랜트 반영 + per-env DR 재샘플 (리셋된 env_ids 대상)."""
        env_ids_long = env_ids.to(torch.long)
        env_ids_int = env_ids.to(torch.int32)
        n = int(env_ids_long.numel())
        dr = self.cfg.dr

        # ── PACE armature / viscous / Coulomb (+ per-env 스케일) ──
        if self.cfg.use_pace_params:
            arm_s = (
                self._sample(dr.armature_scale_range, n, 1) if (self.cfg.domain_rand and dr.randomize_armature) else 1.0
            )
            jf_s = (
                self._sample(dr.joint_friction_scale_range, n, 1)
                if (self.cfg.domain_rand and dr.randomize_joint_friction)
                else 1.0
            )
            armature = self._pace_armature.unsqueeze(0).repeat(n, 1) * arm_s
            viscous = self._pace_viscous.unsqueeze(0).repeat(n, 1) * jf_s
            coulomb = self._pace_coulomb.unsqueeze(0).repeat(n, 1) * jf_s
            self._robot.write_joint_armature_to_sim_index(
                armature=armature, joint_ids=self._all_joint_ids, env_ids=env_ids_int
            )
            self._robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=coulomb,
                joint_dynamic_friction_coeff=coulomb,
                joint_viscous_friction_coeff=viscous,
                joint_ids=self._all_joint_ids,
                env_ids=env_ids_int,
            )
            # priv_latent 버퍼 갱신: 실제로 랜덤화된 env_ids만 (스칼라 nominal=1.0 인 경우는 미기록,
            # 초기값이 이미 nominal 이므로 무의미한 write 생략)
            if self.cfg.domain_rand and dr.randomize_armature:
                self._priv_armature_scale[env_ids_long] = arm_s
            if self.cfg.domain_rand and dr.randomize_joint_friction:
                self._priv_joint_friction_scale[env_ids_long] = jf_s

        if not self.cfg.domain_rand:
            return

        # ── startup-only: 마찰(material) 랜덤화 (첫 전체 reset 에서 모든 env) ──
        if not self._dr_ready:
            self._randomize_material_startup()
            self._dr_ready = True

        # ── base payload 질량 ──
        if dr.randomize_mass:
            add = self._sample(dr.added_base_mass_range, n, 1)
            new_mass = torch.clamp(self._default_base_mass[env_ids_long] + add, min=1e-6)
            body_ids = torch.tensor([self._base_body_id], dtype=torch.int32, device=self.device)
            self._robot.set_masses_index(masses=new_mass, body_ids=body_ids, env_ids=env_ids_int)
            # priv_latent: 실제 적용된(clamp 반영) nominal 대비 오프셋 [kg]
            self._priv_base_mass_offset[env_ids_long] = new_mass - self._default_base_mass[env_ids_long]

        # ── PD 게인 스케일 (DCMotor actuator 텐서 직접 수정) ──
        if dr.randomize_gains:
            kp_scale = self._sample(dr.kp_scale_range, n, 1)
            kd_scale = self._sample(dr.kd_scale_range, n, 1)
            for _a, _kp0, _kd0 in zip(self._acts, self._base_kp, self._base_kd):
                _a.stiffness[env_ids_long] = _kp0[env_ids_long] * kp_scale
                _a.damping[env_ids_long] = _kd0[env_ids_long] * kd_scale
            self._priv_kp_scale[env_ids_long] = kp_scale
            self._priv_kd_scale[env_ids_long] = kd_scale

        # ── encoder bias / action delay / push 타이머 ──
        if dr.encoder_bias:
            rand_bias = torch.rand(n, 12, device=self.device) * 2.0 - 1.0
            self._encoder_bias[env_ids_long] = rand_bias * PACE_ENCODER_BIAS_MAG
        if dr.randomize_action_delay:
            self._action_delay_steps[env_ids_long] = torch.randint(
                0, dr.max_action_delay_steps + 1, (n,), device=self.device
            )
            self._action_delay_buf[env_ids_long] = self._robot.data.default_joint_pos[env_ids_long].unsqueeze(1)
        self._push_timer[env_ids_long] = dr.push_interval_s

    def _randomize_material_startup(self):
        """모든 env 의 강체 마찰(static·dynamic)을 per-env 값으로 랜덤화 (startup 1회).

        코어 ``events.randomize_rigid_body_material`` 의 PhysX 경로와 동일한 API 사용
        (root_view + warp). 전체 shape 에 동일 계수를 적용해 "로봇별 접지 마찰"을 근사한다.
        """
        dr = self.cfg.dr
        materials = wp.to_torch(self._robot.root_view.get_material_properties())  # [num_envs, num_shapes, 3] CPU
        lo, hi = dr.foot_friction_range
        fr = torch.rand(materials.shape[0], 1, device="cpu") * (hi - lo) + lo  # per-env 마찰
        materials[..., 0] = fr  # static friction
        materials[..., 1] = fr  # dynamic friction
        env_ids = torch.arange(materials.shape[0], device="cpu", dtype=torch.int32)
        self._robot.root_view.set_material_properties(
            wp.from_torch(materials.contiguous(), dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
        )
        # priv_latent: nominal 마찰(env_cfg static_friction=1.0) 대비 오프셋으로 저장 (전 env, startup 1회뿐)
        self._priv_foot_friction_offset[:, 0] = fr.to(self.device).squeeze(-1) - 1.0

    def _push_robots(self, env_ids: torch.Tensor):
        """base 에 랜덤 수평 속도 킥을 주어 외란(impulse)을 흉내낸다."""
        n = int(env_ids.numel())
        lin = self._robot.data.root_lin_vel_w[env_ids].clone()  # [n,3] world
        ang = self._robot.data.root_ang_vel_w[env_ids].clone()  # [n,3] world
        kick = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self.cfg.dr.max_push_vel_xy
        lin[:, 0:2] += kick
        vel = torch.cat([lin, ang], dim=-1)  # [n,6]
        self._robot.write_root_com_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids.to(torch.int32))

    def _encode_joint_pos(self, joint_pos_rel: torch.Tensor) -> torch.Tensor:
        """관절각 [N, 12] rad → policy obs 블록. raw 는 항등, tan-norm 은 [N, 72].

        tan-norm 은 MimicKit ``torch_util.quat_to_tan_norm`` 과 같은 식이다: 관절 회전
        ``q = angle_axis(a_j, θ_j)`` 로 기준 tan(1,0,0)·norm(0,0,1) 을 돌려 6 차원을 만든다.

        Args:
            joint_pos_rel: 기본자세 상대 관절각 [rad], shape [num_envs, 12].

        Returns:
            shape [num_envs, 12] (raw) 또는 [num_envs, 72] (tan-norm).
        """
        if not self.cfg.joint_pos_tan_norm:
            return joint_pos_rel
        return _joint_tan_norm(joint_pos_rel, self._joint_axis)

    def _apply_obs_dr(self, obs: torch.Tensor) -> torch.Tensor:
        """policy proprio 에 **중력·관절속도** 관측 노이즈를 더한다. AMP obs/priv_explicit 는 불변.

        레이아웃 — root_lin_vel_b(실측 불가)와 root_ang_vel_b(priv_explicit로 이동)가 모두
        policy obs에서 빠져 있다. GT priv_explicit는 노이즈 없이 그대로 사용한다:
            projected_gravity_b[0:3] +
            lin_vel_cmd[3:5] + yaw_vel_cmd[5:6] +
            joint_pos 블록[6 : 6+D] + joint_vel[6+D : 18+D] + actions[18+D : 30+D]
        여기서 D 는 12(raw) 또는 72(tan-norm)다.

        ⚠ 관절각 노이즈와 ``encoder_bias`` 는 여기가 아니라 ``_get_observations`` 에서 **라디안
        공간**에 더한다 — tan-norm 인코딩 후에는 라디안을 더할 수 없기 때문이다.

        ⚠ 인덱스를 하드코딩하지 말 것. 예전에 각속도를 obs 에서 뺐을 때 슬라이스를 밀린 채
        남겨두면 σ=0.2 노이즈가 σ=0.05 인 projected_gravity_b 에 주입돼 4 배 증폭됐다.
        ``dr.ang_vel_noise`` 는 주입할 자리가 없어 dead config 다.
        """
        if not self.cfg.domain_rand:
            return obs
        dr = self.cfg.dr
        noise = torch.zeros_like(obs)
        if dr.obs_noise:
            n = self.num_envs
            jv = self._obs_idx_joint_vel
            noise[:, 0:3] = torch.randn(n, 3, device=self.device) * dr.gravity_noise  # projected_gravity_b
            noise[:, jv : jv + 12] = torch.randn(n, 12, device=self.device) * dr.joint_vel_noise  # joint_vel
        return obs + noise

    def _get_priv_latent(self) -> torch.Tensor:
        """RMA priv_latent(19-dim) 조립 — quasi-static domain-rand 파라미터, 매 env 항상 반환.

        DR 항목이 off 이거나 아직 샘플되지 않은 env는 버퍼 nominal 값(scale=1.0, offset=0.0)을
        그대로 반환한다 (ActorCriticRMA가 priv_latent 키를 무조건 요구하므로).
        """
        return torch.cat(
            [
                self._priv_armature_scale,  # 1
                self._priv_joint_friction_scale,  # 1
                self._priv_base_mass_offset,  # 1  [kg]
                self._priv_foot_friction_offset,  # 1
                self._priv_kp_scale,  # 1
                self._priv_kd_scale,  # 1
                (self._action_delay_steps.float() / max(1, self.cfg.dr.max_action_delay_steps)).unsqueeze(-1),  # 1
                self._encoder_bias / PACE_ENCODER_BIAS_MAG,  # 12
            ],
            dim=-1,
        )  # (N, 19)

    # ──────────────────────────────────────────────────────────
    # 리셋 전략
    # ──────────────────────────────────────────────────────────

    def _reset_strategy_rsi(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference State Initialization — 모션 데이터로 초기화."""
        n = len(env_ids)
        motion_ids = self._motion_lib.sample_motions(n)
        start = "start" in self.cfg.reset_strategy
        if start:
            times = torch.zeros(n, device=self.device)
        else:
            times = self._motion_lib.sample_times(motion_ids)

        root_pos, root_quat, lin_vel_b, ang_vel_b, dof_pos, dof_vel, _ = self._motion_lib.calc_motion_frame(
            motion_ids, times
        )

        # root_state [N, 13]: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        # motion_lib returns wxyz; IsaacLab 6.0 sim and quat_apply expect xyzw.
        root_quat_xyzw = convert_quat(root_quat, to="xyzw")
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = root_pos + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_quat_xyzw  # xyzw

        # motion_lib 속도는 body frame → world frame 변환 필요
        root_state[:, 7:10] = quat_apply(root_quat_xyzw, lin_vel_b)
        root_state[:, 10:13] = quat_apply(root_quat_xyzw, ang_vel_b)

        joint_pos_out = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel_out = self._robot.data.default_joint_vel[env_ids].clone()
        # motion data → IsaacLab joint 순서로 재배열
        joint_pos_out[:, :12] = dof_pos[:, self._motion_dof_indices]
        joint_vel_out[:, :12] = dof_vel[:, self._motion_dof_indices]

        # AMP 버퍼를 RSI 시작점으로 채우기 (동일 motion_ids 전달 — Bug 1 수정)
        # 1-step 오프셋: _get_observations()에서 버퍼 시프트가 한 번 더 일어나므로,
        # RSI 버퍼를 1 step 앞서 채워야 시프트 후 [ref@t, ref@t-dt, ..., ref@t-9dt]가 정렬됨.
        pre_shift_times = (times - self.step_dt).clamp(min=0.0)
        amp_obs_buf, root_pos_hist, quat_hist = self._compute_reference_buffers(
            n, pre_shift_times.cpu().numpy(), motion_ids=motion_ids
        )

        self.amp_observation_buffer[env_ids] = amp_obs_buf
        self._amp_quat_buf[env_ids] = quat_hist  # [R4] fill quat history from motion data
        if self.cfg.include_rel_track_obs:
            self._hist_root_pos_w[env_ids] = root_pos_hist

        # ── rest-init: 일부 env 는 참조 프레임 대신 **기립 정지**로 시작한다 ──────────────
        # 참조 18 클립은 전부 "이미 걷고 있는" 상태라(가장 느린 go2_walk 도 평균 0.15 m/s),
        # RSI 만 쓰면 정지가 초기 조건이 된 적이 없다 → 정지→이동 전이를 배울 기회가 없다.
        if self.cfg.rel_rest_init > 0.0:
            rest = torch.rand(n, device=self.device) < self.cfg.rel_rest_init
            if rest.any():
                default_root = self._robot.data.default_root_state[env_ids][rest].clone()
                # default_root_state 는 env-local 좌표라 origin 을 더해야 한다(RSI 경로와 동일).
                default_root[:, 0:3] = default_root[:, 0:3] + self.scene.env_origins[env_ids][rest]
                default_root[:, 7:13] = 0.0  # lin/ang velocity = 0
                root_state[rest] = default_root
                joint_pos_out[rest] = self._robot.data.default_joint_pos[env_ids][rest]
                joint_vel_out[rest] = 0.0
                # NOTE: amp_observation_buffer 는 위에서 참조 모션으로 채워진 채 둔다. 로봇은
                # 정지 상태이므로 첫 ~10 step 동안만 history 앞부분이 실제 상태와 어긋나는데,
                # `_get_observations` 가 매 step 실측 amp_obs 를 밀어 넣어 그 안에 씻겨 나간다.
                # 영향 규모: rel_rest_init × (10 step / 에피소드 1000 step) ≈ 0.1% 의 disc 샘플.
                # 정지 상태의 amp_obs 를 정확히 만들려면 reset 시점에 FK 가 필요해 비용이 크다.

        return root_state, joint_pos_out, joint_vel_out

    # ──────────────────────────────────────────────────────────
    # 속도 명령 관리 (재샘플링)
    # ──────────────────────────────────────────────────────────

    def _resample_steering(self, env_ids: torch.Tensor):
        """lin_vel_cmd(vx, vy), yaw_vel_cmd, 타이머를 재샘플링합니다."""
        n = len(env_ids)

        # vx: [lin_vel_x_min, lin_vel_x_max]
        self._lin_vel_cmd[env_ids, 0] = (
            torch.rand(n, device=self.device) * (self.cfg.lin_vel_x_max - self.cfg.lin_vel_x_min)
            + self.cfg.lin_vel_x_min
        )
        # vy: [lin_vel_y_min, lin_vel_y_max] — 기본 (0.0, 0.0) 이므로 항상 0
        self._lin_vel_cmd[env_ids, 1] = (
            torch.rand(n, device=self.device) * (self.cfg.lin_vel_y_max - self.cfg.lin_vel_y_min)
            + self.cfg.lin_vel_y_min
        )

        # yaw rate: [yaw_vel_min, yaw_vel_max]
        self._yaw_vel_cmd[env_ids] = (
            torch.rand(n, device=self.device) * (self.cfg.yaw_vel_max - self.cfg.yaw_vel_min) + self.cfg.yaw_vel_min
        )

        # 일부 env 는 **정확히 0**(정지) 명령으로 강제한다. uniform 샘플의 near-zero tail 만으로는
        # 정지가 뚜렷한 모드로 학습되지 않는다(cmd 0 에서 task reward 는 v=0 일 때 1.0 최대인데도
        # 정책이 그 상태에서 빠져나오지 못했다). IsaacLab 표준 velocity command 의
        # `rel_standing_envs` 와 같은 장치.
        if self.cfg.rel_standing_envs > 0.0:
            standing = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
            self._lin_vel_cmd[env_ids[standing]] = 0.0
            self._yaw_vel_cmd[env_ids[standing]] = 0.0
            # ④ style 대체용 마스크. **명령 기준**이라 정책이 스스로 멈춰서 pose 보상을 수확할 수
            # 없다(실측 속도로 게이트하면 그게 가능해진다). 다음 재샘플까지 유지된다.
            self._standing_mask[env_ids] = standing
        else:
            self._standing_mask[env_ids] = False

        # 다음 변경까지 남은 시간
        self._tar_timer[env_ids] = (
            torch.rand(n, device=self.device) * (self.cfg.tar_change_time_max - self.cfg.tar_change_time_min)
            + self.cfg.tar_change_time_min
        )

    # ──────────────────────────────────────────────────────────
    # AMP 인터페이스 (OnPolicyRunnerAMPBase 필수)
    # ──────────────────────────────────────────────────────────

    def _compute_reference_buffers(
        self,
        num_samples: int,
        current_times: np.ndarray | None = None,
        motion_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(num_samples)

        if current_times is None:
            current_times_t = self._motion_lib.sample_times(motion_ids)
        else:
            current_times_t = torch.tensor(current_times, dtype=torch.float32, device=self.device)

        n_hist = self.cfg.num_amp_observations
        time_offsets = torch.arange(n_hist, device=self.device).float() * self.step_dt  # [H]
        times_flat = (current_times_t.unsqueeze(-1) - time_offsets.unsqueeze(0)).reshape(-1)
        times_flat = torch.clamp(times_flat, min=0.0)
        motion_ids_flat = motion_ids.unsqueeze(-1).expand(-1, n_hist).reshape(-1)

        frame = self._motion_lib.calc_motion_frame(motion_ids_flat, times_flat)
        root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos = (
            frame[0],
            frame[1],
            frame[2],
            frame[3],
            frame[4],
            frame[5],
            frame[6],
        )

        dof_pos = dof_pos[:, self._motion_dof_indices]
        dof_vel = dof_vel[:, self._motion_dof_indices]

        amp_obs = _compute_amp_obs(
            dof_pos,
            dof_vel,
            root_pos,
            lin_vel,
            ang_vel,
            foot_pos,
            self._amp_joint_axis,
        )

        amp_obs_buf = amp_obs.view(num_samples, n_hist, -1)  # [N, H, 43]
        root_pos_hist = root_pos.view(num_samples, n_hist, 3)
        # [R4] return full quat history (N, H, 4) — convert wxyz (motion_lib) → xyzw (IsaacLab 6.0)
        # so all consumers (_reset_strategy_rsi, collect_reference_motions) receive xyzw directly.
        quat_hist_wxyz = root_quat.view(num_samples, n_hist, 4)
        quat_hist = convert_quat(quat_hist_wxyz.reshape(-1, 4), to="xyzw").reshape(num_samples, n_hist, 4)
        return amp_obs_buf, root_pos_hist, quat_hist

    def collect_reference_motions(
        self,
        num_samples: int,
        current_times: np.ndarray | None = None,
        motion_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """레퍼런스 모션 AMP 관측값 수집. (상대 궤적 피처 + R4 root_rot_tan_norm 포함)"""
        if motion_ids is None:
            # 조건부 disc 라벨을 위해 클립 id 를 여기서 뽑아 들고 있는다.
            motion_ids = self._motion_lib.sample_motions(num_samples)
        amp_obs_buf, root_pos_hist, quat_hist = self._compute_reference_buffers(num_samples, current_times, motion_ids)
        curr_root_quat = quat_hist[:, 0, :]  # [N, 4]
        n_hist = self.cfg.num_amp_observations

        if self.cfg.include_rel_track_obs:
            curr_root_pos = root_pos_hist[:, 0, :]
            root_pos_diff = root_pos_hist - curr_root_pos.unsqueeze(1)

            heading_rot_inv = _calc_heading_quat_inv(curr_root_quat)
            heading_inv_expand = heading_rot_inv.unsqueeze(1).expand(-1, n_hist, -1).reshape(num_samples * n_hist, 4)
            local_root_pos_diff = quat_apply(heading_inv_expand, root_pos_diff.reshape(num_samples * n_hist, 3)).view(
                num_samples, n_hist, 3
            )
            root_pos_obs_xy = local_root_pos_diff[:, :, :2]

            base_amp = torch.cat([root_pos_obs_xy, amp_obs_buf], dim=-1)  # [N, H, 45]
        else:
            base_amp = amp_obs_buf  # [N, H, 43]

        # [R4] expert 측 동일한 tan_norm 변환 (live path와 완전 대칭)
        rot_tan_norm = _apply_root_rot_tan_norm(quat_hist, num_samples, n_hist)  # [N, H, 6]
        final_amp_obs = torch.cat([base_amp, rot_tan_norm], dim=-1)  # [N, H, 49]

        return self._append_amp_cond(final_amp_obs.view(num_samples, -1), self._expert_amp_cond(motion_ids))

    def get_amp_observations(self, num_samples: int) -> torch.Tensor:
        """Runner가 Discriminator 업데이트 시 호출하는 Expert 관측 샘플러."""
        return self.collect_reference_motions(num_samples)

    # ──────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────

    def _get_body_contact(
        self,
        contact_forces: torch.Tensor,
        body_names: list[str],
        keyword: str,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        indexes = [i for i, name in enumerate(body_names) if keyword in name.lower()]
        if not indexes:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if contact_forces.shape[1] >= max(indexes) + 1:
            forces = contact_forces[:, indexes, :]
            return torch.norm(forces, dim=-1).max(dim=-1).values > threshold
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


# ──────────────────────────────────────────────────────────────
# JIT 컴파일 함수
# ──────────────────────────────────────────────────────────────


@torch.jit.script
def _joint_tan_norm(theta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """관절각 → 관절별 회전의 tan-norm (MimicKit ``quat_to_tan_norm`` 과 같은 값).

    회전축 a 둘레로 θ 만큼 돌린 회전을 기준 tan(1,0,0)·norm(0,0,1) 에 적용해 관절당 6 차원을
    만든다. 쿼터니언을 거치지 않고 로드리게스 식 ``v·cosθ + (a×v)·sinθ + a(a·v)(1−cosθ)`` 를
    직접 쓴다 — 결과는 같고(단위 테스트로 확인) jit 안에서 안전하다.

    **policy obs 와 disc obs 가 이 함수 하나를 공유한다.** 두 경로가 따로 구현되면 언젠가
    갈라지는데, disc 는 정책 obs 와 달리 expert(참조 모션) 쪽과도 대칭이어야 해서 치명적이다.

    Args:
        theta: 관절각 [rad], shape [batch, num_joints].
        axis: 관절별 단위 회전축, shape [num_joints, 3].

    Returns:
        shape [batch, num_joints * 6] — 관절마다 [tan(3), norm(3)].
    """
    n, j = theta.shape[0], theta.shape[1]
    c = torch.cos(theta).unsqueeze(-1)  # [n, j, 1]
    s = torch.sin(theta).unsqueeze(-1)
    a = axis.unsqueeze(0).expand(n, j, 3)  # [n, j, 3]
    out = torch.zeros(n, j, 6, dtype=theta.dtype, device=theta.device)
    for k in range(2):
        ref = torch.zeros(3, dtype=theta.dtype, device=theta.device)
        ref[0 if k == 0 else 2] = 1.0  # tan=(1,0,0), norm=(0,0,1)
        v = ref.view(1, 1, 3).expand(n, j, 3)
        rot = v * c + torch.cross(a, v, dim=-1) * s + a * (a * v).sum(-1, keepdim=True) * (1.0 - c)
        out[:, :, 3 * k : 3 * k + 3] = rot
    return out.reshape(n, j * 6)


@torch.jit.script
def _compute_amp_obs(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    foot_pos_local: torch.Tensor,
    joint_axis: torch.Tensor,
) -> torch.Tensor:
    """AMP disc 기본 관측 벡터 계산 (43-dim, tan-norm 이면 103-dim).

    dof_pos(12) + dof_vel(12) + root_height(1) + lin_vel(3) + ang_vel(3) + foot_pos(12)
    root_rot_tan_norm(6)은 _apply_root_rot_tan_norm()에서 window stacking 후 추가됨 (→ 49-dim).

    ★ live(정책) · terminal · expert(참조 모션) **세 경로가 전부 이 함수를 지난다.** disc 는
    정책과 expert 를 같은 자로 재야 의미가 있으므로, 인코딩을 여기 한 곳에만 두는 게 필수다.

    Args:
        joint_axis: 비어 있으면 ``dof_pos`` 를 raw 라디안 12 로 그대로 넣는다. [12, 3] 이면
            관절별 회전의 tan-norm 72 로 바꾼다 (``amp_joint_tan_norm=True``).
    """
    dof_pos_obs = _joint_tan_norm(dof_pos, joint_axis) if joint_axis.numel() > 0 else dof_pos
    return torch.cat(
        [
            dof_pos_obs,  # 12 (raw) 또는 72 (tan-norm)
            dof_vel,  # 12
            root_pos[:, 2:3],  # 1 (root 높이)
            root_lin_vel,  # 3
            root_ang_vel,  # 3
            foot_pos_local.view(foot_pos_local.shape[0], -1),  # 12
        ],
        dim=-1,
    )


def _apply_root_rot_tan_norm(
    quat_buf: torch.Tensor,
    num_envs: int,
    n_hist: int,
) -> torch.Tensor:
    """Window 내 각 frame의 root_quat을 현재 frame heading-inv 기준 local로 변환 후 6D tan_norm 반환.

    MimicKit compute_tar_obs (deepmimic_env.py:733,742) 방식:
      - ref = quat_buf[:, 0, :] (window index 0 = 가장 최근 frame)
      - heading_inv_rot = _calc_heading_quat_inv(ref)  (yaw-only inverse)
      - relative_quat[h] = quat_mul(heading_inv_expand, quat_buf[:, h, :])
      - tan_norm: [quat_rotate(q, [1,0,0]), quat_rotate(q, [0,0,1])] (MimicKit torch_util.py:216-227)

    Args:
        quat_buf: [N, H, 4] xyzw — per-step root_quat history (index 0 = newest)
        num_envs: N
        n_hist:   H

    Returns:
        rot_tan_norm: [N, H, 6]
    """
    # ref heading-inv: yaw-only inverse of the current (newest) frame
    ref_quat = quat_buf[:, 0, :]  # [N, 4]
    heading_inv = _calc_heading_quat_inv(ref_quat)  # [N, 4]

    # expand heading_inv over history dimension
    heading_inv_exp = heading_inv.unsqueeze(1).expand(-1, n_hist, -1).reshape(num_envs * n_hist, 4)  # [N*H, 4]
    quats_flat = quat_buf.reshape(num_envs * n_hist, 4)  # [N*H, 4]

    # relative_quat = heading_inv * frame_quat  (MimicKit deepmimic_env.py:742)
    rel_quat = quat_mul(heading_inv_exp, quats_flat)  # [N*H, 4]

    # tan_norm: rotate x-axis and z-axis (MimicKit torch_util.py:218-224)
    tan_ref = torch.zeros(num_envs * n_hist, 3, dtype=quat_buf.dtype, device=quat_buf.device)
    tan_ref[:, 0] = 1.0  # [1, 0, 0]
    norm_ref = torch.zeros(num_envs * n_hist, 3, dtype=quat_buf.dtype, device=quat_buf.device)
    norm_ref[:, 2] = 1.0  # [0, 0, 1]

    tan = quat_apply(rel_quat, tan_ref)  # [N*H, 3]
    norm = quat_apply(rel_quat, norm_ref)  # [N*H, 3]

    tan_norm = torch.cat([tan, norm], dim=-1)  # [N*H, 6]
    return tan_norm.view(num_envs, n_hist, 6)


@torch.jit.script
def _calc_heading_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only quaternion (heading) 추출. quat: [N,4] xyzw."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = yaw * 0.5
    heading = torch.stack(
        [torch.zeros_like(half_yaw), torch.zeros_like(half_yaw), torch.sin(half_yaw), torch.cos(half_yaw)],
        dim=-1,
    )
    return heading  # [N,4] xyzw


@torch.jit.script
def _calc_heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Heading quaternion 역원 (heading 기준 로컬 변환용). quat: [N,4] xyzw."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = -yaw * 0.5  # inverse = negative yaw
    heading_inv = torch.stack(
        [torch.zeros_like(half_yaw), torch.zeros_like(half_yaw), torch.sin(half_yaw), torch.cos(half_yaw)],
        dim=-1,
    )
    return heading_inv  # [N,4] xyzw
