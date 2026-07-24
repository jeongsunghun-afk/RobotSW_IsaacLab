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

Policy observation dict (실배포 가능 RMA 구조 — root_lin_vel_b는 실측 불가하므로
estimator가 policy(45)로부터 추정하고, priv_explicit(3)는 학습 시 GT critic/estimator target 용):
  policy(45)        = root_ang_vel_b(3) + projected_gravity_b(3) +
                       lin_vel_cmd(2) + yaw_vel_cmd(1) +
                       joint_pos - default(12) + joint_vel(12) + actions(12)
  priv_explicit(3)  = root_lin_vel_b * priv_explicit_lin_vel_scale
  priv_latent(19)   = armature_scale(1) + joint_friction_scale(1) + base_mass_offset(1) +
                       foot_friction_offset(1) + kp_scale(1) + kd_scale(1) +
                       action_delay_norm(1) + encoder_bias_norm(12)
  history(10, 45)   = policy proprio ring buffer (noised)

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

from .go2_imitation_tracking_env_cfg import (
    PACE_ARMATURE,
    PACE_COULOMB,
    PACE_ENCODER_BIAS_MAG,
    PACE_VISCOUS,
    Go2ImitationTrackingEnvCfg,
)
from .motion_lib import Go2MotionLib


class Go2ImitationTrackingEnv(DirectRLEnv):
    """Go2 AMP + body-frame 속도추종 Imitation 환경 (RMA + estimator 아키텍처).

    Policy proprio (45-dim, 실측 가능한 신호만):
        root_ang_vel_b(3) + projected_gravity_b(3) +
        lin_vel_cmd(2) + yaw_vel_cmd(1) +
        joint_pos - default(12) + joint_vel(12) + actions(12)

    priv_explicit(3-dim, 실측 불가 — 학습 시 critic/estimator target GT):
        root_lin_vel_b * priv_explicit_lin_vel_scale

    priv_latent(19-dim, quasi-static domain-rand 파라미터 — RMA priv encoder 입력):
        armature_scale(1) + joint_friction_scale(1) + base_mass_offset(1) +
        foot_friction_offset(1) + kp_scale(1) + kd_scale(1) +
        action_delay_norm(1) + encoder_bias_norm(12)

    history(10, 45): policy proprio ring buffer (노이즈 포함, 최신이 index 0).

    AMP disc observation (49-dim per step):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [R4: heading-relative 6D rotation]
    """

    cfg: Go2ImitationTrackingEnvCfg

    # key body 이름 (발 4개) — motion_lib foot 순서와 일치
    KEY_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def __init__(self, cfg: Go2ImitationTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg

        # ── 모션 라이브러리 ─────────────────────────────────────
        motion_file = self.cfg.motion_file
        if os.path.isdir(motion_file):
            pkl_files = sorted(glob.glob(os.path.join(motion_file, "*.pkl")))
            assert pkl_files, f"PKL 파일 없음: {motion_file}"
            motion_files = pkl_files
        else:
            motion_files = [motion_file]
        self._motion_lib = Go2MotionLib(motion_files=motion_files, device=self.device)

        # ── body / joint 인덱스 ──────────────────────────────────
        self.ref_body_index = self._robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self._robot.data.body_names.index(name) for name in self.KEY_BODY_NAMES]

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
        self._yaw_vel_cmd = torch.zeros(self.num_envs, device=self.device)  # yaw rate [rad/s]
        self._tar_timer = torch.zeros(self.num_envs, device=self.device)  # [N] seconds to change

        # ── AMP 관측 버퍼 ─────────────────────────────────────────
        # amp_observation_space (cfg=49) = base(43) + root_rot_tan_norm(6)  [R4]
        # amp_observation_buffer 내부 저장은 base 43-dim만; tan_norm 6D는 소비 시점에 계산됨.
        _AMP_BASE_DIM = 43  # 내부 버퍼 차원 (고정)
        self.amp_observation_size = self.cfg.num_amp_observations * (
            self.cfg.amp_observation_space + (2 if self.cfg.include_rel_track_obs else 0)
        )
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

        # ── RMA proprio history 링버퍼 (policy obs 45-dim × history_len) ────
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

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        target = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
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
            "amp_obs": final_amp_obs.view(self.num_envs, -1).clone(),
            # terminal step에서 러너가 post-reset RSI obs 대신 이 값으로 amp_reward 계산
            "terminal_amp_obs": self._terminal_amp_obs.clone(),
        }

        # ── Policy proprio (45-dim) — 실측 가능한 신호만. root_lin_vel_b는 실배포 시
        # 직접 측정 불가하므로 policy obs에서 제외하고 priv_explicit로 분리(estimator가 추정) ──
        proprio = torch.cat(
            [
                root_ang_vel_b,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._lin_vel_cmd,  # 2 (vx, vy)
                self._yaw_vel_cmd.unsqueeze(-1),  # 1
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel,  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )  # total = 45
        policy_obs = self._apply_obs_dr(proprio)

        # ── priv_explicit(3) — GT root_lin_vel_b (critic/estimator target, 노이즈 없음) ──
        priv_explicit = root_lin_vel_b * self.cfg.priv_explicit_lin_vel_scale

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

        self._terminal_amp_obs[env_ids] = final_terminal_amp_obs.view(n, -1)
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
        self._act = self._robot.actuators["base_legs"]
        self._base_kp = self._act.stiffness.clone()
        self._base_kd = self._act.damping.clone()

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
            self._act.stiffness[env_ids_long] = self._base_kp[env_ids_long] * kp_scale
            self._act.damping[env_ids_long] = self._base_kd[env_ids_long] * kd_scale
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

    def _apply_obs_dr(self, obs: torch.Tensor) -> torch.Tensor:
        """policy proprio(45-dim) 에만 관측 노이즈 + encoder bias 를 더한다. AMP obs/priv_explicit 는 불변.

        45-dim 레이아웃 (root_lin_vel_b 는 policy obs에서 분리되어 priv_explicit로 이동했으므로
        여기엔 없음 — GT priv_explicit는 노이즈 없이 그대로 사용):
            root_ang_vel_b[0:3] + projected_gravity_b[3:6] +
            lin_vel_cmd[6:8] + yaw_vel_cmd[8:9] +
            (joint_pos-default)[9:21] + joint_vel[21:33] + actions[33:45]
        """
        if not self.cfg.domain_rand:
            return obs
        dr = self.cfg.dr
        noise = torch.zeros_like(obs)
        if dr.obs_noise:
            n = self.num_envs
            noise[:, 0:3] = torch.randn(n, 3, device=self.device) * dr.ang_vel_noise  # root_ang_vel_b
            noise[:, 3:6] = torch.randn(n, 3, device=self.device) * dr.gravity_noise  # projected_gravity_b
            noise[:, 9:21] = torch.randn(n, 12, device=self.device) * dr.joint_pos_noise  # joint_pos - default
            noise[:, 21:33] = torch.randn(n, 12, device=self.device) * dr.joint_vel_noise  # joint_vel
        if dr.encoder_bias:
            noise[:, 9:21] += self._encoder_bias  # per-env 고정 엔코더 오프셋
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

        return final_amp_obs.view(num_samples, self.amp_observation_size)

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
def _compute_amp_obs(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    foot_pos_local: torch.Tensor,
) -> torch.Tensor:
    """AMP disc 기본 관측 벡터 계산 (43-dim).

    dof_pos(12) + dof_vel(12) + root_height(1) + lin_vel(3) + ang_vel(3) + foot_pos(12)
    root_rot_tan_norm(6)은 _apply_root_rot_tan_norm()에서 window stacking 후 추가됨 (→ 49-dim).
    """
    return torch.cat(
        [
            dof_pos,  # 12
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
