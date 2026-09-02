# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Leg(17-DOF) Imitation Tracking 환경 — body-frame 속도추종 command (vx, vy, yaw_rate).

go2_imitation에서 steering (tar_dir · tar_speed · face_dir) 구조를
body-frame 선속도/각속도 추종(vx, vy=0, yaw_rate)으로 교체한 환경.

  - AMP Discriminator: 59-dim obs × 10 history = 590-dim  (R4: root_rot_tan_norm 6D 추가)
  - Task reward: lin_vel_reward(0.7) + yaw_vel_reward(0.3)
  - Reset: 항상 RSI (Reference State Initialization)
  - 알고리즘: PPOAMPBase + OnPolicyRunnerAMPBase (rsl_rl)

Policy observation (63-dim):
  root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3) +
  lin_vel_cmd(2) + yaw_vel_cmd(1) +
  joint_pos - default(17) + joint_vel(17) + actions(17)

[R4 ablation] MimicKit compute_tar_obs 방식 이식:
  - 각 disc window frame의 root_quat을 window[-1](현재) frame의 heading-inv 기준 local로 변환
  - 변환식: relative_quat = quat_mul(heading_inv_of_ref, frame_quat)
    (MimicKit deepmimic_env.py:733,742 — heading-only inverse, 전체 quat inverse 아님)
  - 변환된 quat → quat_to_tan_norm: [tan(x), norm(z)] 6D feature
    (MimicKit torch_util.py:218-224 — ref_tan=[1,0,0], ref_norm=[0,0,1])
  - per-step disc feature: 53 → 59 (base_53 + root_rot_tan_norm_6)
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

from .leg_imitation_tracking_env_cfg import LegImitationTrackingEnvCfg
from .motion_lib import LegMotionLib


class LegImitationTrackingEnv(AMPCommandConditionMixin, DirectRLEnv):
    """Leg(17-DOF 4족+허리) AMP + body-frame 속도추종 Imitation 환경.

    Policy observation (63-dim):
        root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3) +
        lin_vel_cmd(2) + yaw_vel_cmd(1) +
        joint_pos - default(17) + joint_vel(17) + actions(17)

    AMP disc observation (59-dim per step):
        dof_pos(17) + dof_vel(17) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [R4: heading-relative 6D rotation]
    """

    cfg: LegImitationTrackingEnvCfg

    # key body 이름 (발 4개) — motion_lib foot 순서와 일치
    KEY_BODY_NAMES = ["FL_foot_link", "FR_foot_link", "HL_foot_link", "HR_foot_link"]

    def __init__(self, cfg: LegImitationTrackingEnvCfg, render_mode: str | None = None, **kwargs):
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
        weight_mode = self.cfg.motion_weight_mode
        valid_modes = ("length", "command_uniform", "command_uniform_mirror")
        if weight_mode not in valid_modes:
            raise ValueError(f"motion_weight_mode 는 {valid_modes} 중 하나여야 한다: {weight_mode}")
        self._motion_lib = LegMotionLib(
            motion_files=motion_files,
            device=self.device,
            mirror_complete=weight_mode == "command_uniform_mirror",
        )
        if weight_mode != "length":
            w = self._motion_lib.set_motion_weights_command_uniform(self.cfg.lin_vel_x_max)
            speeds = self._motion_lib.motion_mean_speeds
            print(f"[LegMotionLib] weight_mode={weight_mode} (vel_max={self.cfg.lin_vel_x_max:.2f} m/s)")
            for name, v, p in zip(self._motion_lib.motion_names, speeds.tolist(), w.tolist()):
                print(f"  {name:32s} {v:5.2f} m/s  weight {100 * p:5.2f}%")

        # ── body / joint 인덱스 ──────────────────────────────────
        self.ref_body_index = self._robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self._robot.data.body_names.index(name) for name in self.KEY_BODY_NAMES]

        # ── motion_lib ↔ IsaacLab joint 순서 매핑 ────────────────
        # IsaacLab joint 순서(알파벳 등)와 PKL DOF_NAMES 순서가 다를 수 있음.
        # go2_amp_env.py의 motion_dof_indexes 패턴과 동일.
        robot_joint_names = list(self._robot.data.joint_names)
        try:
            self._motion_dof_indices = self._motion_lib.get_dof_index(robot_joint_names)
            print(f"[LegImitationTrackingEnv] IsaacLab joint 순서: {robot_joint_names}")
            print(f"[LegImitationTrackingEnv] motion_dof_indices: {self._motion_dof_indices}")
        except AssertionError:
            print("[LegImitationTrackingEnv] DOF 이름 불일치 — 순서대로 1:1 매핑 사용")
            self._motion_dof_indices = list(range(17))

        # ── Velocity tracking command state ─────────────────────
        self._lin_vel_cmd = torch.zeros(self.num_envs, 2, device=self.device)  # (vx, vy) — vy 항상 0
        self._yaw_vel_cmd = torch.zeros(self.num_envs, device=self.device)  # yaw rate [rad/s]
        self._tar_timer = torch.zeros(self.num_envs, device=self.device)  # [N] seconds to change

        # ── AMP 관측 버퍼 ─────────────────────────────────────────
        # amp_observation_space (cfg=59) = base(53) + root_rot_tan_norm(6)  [R4]
        # amp_observation_buffer 내부 저장은 base 53-dim만; tan_norm 6D는 소비 시점에 계산됨.
        _AMP_BASE_DIM = 53  # 내부 버퍼 차원 (17+17+1+3+3+12)
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
        # Kept separate from amp_observation_buffer so base 53-dim stays untouched.
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
            "torque_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _resample_steering_on_timer(self):
        """`tar_change_time` 이 지난 env 의 속도 명령을 다시 뽑는다.

        ★ 이 메서드는 원래 `_post_physics_step()` 이라는 이름이었고 **한 번도 호출되지 않았다**
        — `DirectRLEnv.step()` 에 그런 훅이 없다(`_pre_physics_step`/`_apply_action`/`_get_dones`/
        `_get_rewards`/`_reset_idx`/`_get_observations` 뿐). 그래서 `tar_change_time_min/max` 는
        죽은 설정이었고 명령은 리셋 때 한 번 뽑혀 **에피소드 내내 고정**됐다(4096 env·8 s 실측:
        명령이 바뀐 env 0/4091). 정책이 "달리는 중 명령 변화"를 겪은 적이 없어 속도에 따른
        보행 전환을 학습할 신호가 아예 없었다.

        살아 있는 `_get_rewards()` 끝에서 부른다 — 그 step 의 보상은 **옛 명령**으로 계산되고
        이어지는 `_get_observations()` 가 **새 명령**을 싣는다. `_get_dones()` 에서 부르면 보상이
        한 step 어긋난 명령으로 계산된다.

        기본값은 OFF(`resample_command_in_episode=False`)라 기존 run 의 재현성은 유지된다.
        """
        if not self.cfg.resample_command_in_episode:
            return
        self._tar_timer -= self.step_dt
        change_mask = self._tar_timer <= 0.0
        if change_mask.any():
            change_ids = change_mask.nonzero(as_tuple=False).flatten()
            self._resample_steering(change_ids)

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
        # `root_lin_vel_b`/`root_ang_vel_b` 는 COM 기준이다. motion_lib 레퍼런스는 base link 기준이므로
        # AMP disc 가 두 분포를 같은 좌표계에서 보도록 link 기준 속도를 쓴다.
        root_lin_vel_b = self._robot.data.root_link_lin_vel_b.torch  # [N,3]
        root_ang_vel_b = self._robot.data.root_link_ang_vel_b.torch  # [N,3]

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

            base_amp = torch.cat([root_pos_obs_xy, self.amp_observation_buffer], dim=-1)  # [N, H, 57]
        else:
            N, H = self.num_envs, self.cfg.num_amp_observations
            base_amp = self.amp_observation_buffer  # [N, H, 53]

        # [R4] heading-relative root_rot → tan_norm 6D (MimicKit compute_tar_obs 방식)
        # ref = 현재(window 첫 번째=최신) frame의 heading_inv
        # 각 window frame quat을 heading_inv 기준 local로 변환 후 tan_norm 적용
        rot_tan_norm = _apply_root_rot_tan_norm(self._amp_quat_buf, N, H)  # [N, H, 6]
        final_amp_obs = torch.cat([base_amp, rot_tan_norm], dim=-1)  # [N, H, 59]

        self.extras = {
            "amp_obs": self._append_amp_cond(final_amp_obs.view(self.num_envs, -1), self._policy_amp_cond()).clone(),
            # terminal step에서 러너가 post-reset RSI obs 대신 이 값으로 amp_reward 계산
            "terminal_amp_obs": self._terminal_amp_obs.clone(),
        }

        # ── Policy 관측 (63-dim) — body-frame 속도 명령을 그대로 삽입 ──
        policy_obs = torch.cat(
            [
                root_lin_vel_b,  # 3
                root_ang_vel_b,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._lin_vel_cmd,  # 2 (vx, vy)
                self._yaw_vel_cmd.unsqueeze(-1),  # 1
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 17
                self._robot.data.joint_vel,  # 17
                self.actions,  # 17
            ],
            dim=-1,
        )  # total = 63

        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        # ── (1) lin_vel tracking — body frame 직접 비교 ──────────
        lin_vel_b = self._robot.data.root_link_lin_vel_b[:, :2]  # (vx, vy) actual
        lin_vel_err = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
        lin_vel_reward = torch.exp(-self.cfg.vel_err_scale * lin_vel_err)

        # ── (2) yaw_vel tracking ─────────────────────────────────
        yaw_vel_b = self._robot.data.root_link_ang_vel_b[:, 2]
        yaw_vel_err = (self._yaw_vel_cmd - yaw_vel_b) ** 2
        yaw_vel_reward = torch.exp(-self.cfg.yaw_vel_err_scale * yaw_vel_err)

        # ── (3) 관절 토크 페널티 (opt-in, torque_penalty_w=0.0 이면 no-op) ──
        # applied_torque = 액추에이터/effort-limit 적용 후 실제 토크 [N, 17].
        torque_penalty = -self.cfg.torque_penalty_w * torch.sum(self._robot.data.applied_torque.torch**2, dim=-1)

        reward = (
            self.cfg.lin_vel_reward_w * lin_vel_reward + self.cfg.yaw_vel_reward_w * yaw_vel_reward + torque_penalty
        )

        self._episode_sums["lin_vel_reward"] += lin_vel_reward
        self._episode_sums["yaw_vel_reward"] += yaw_vel_reward
        self._episode_sums["torque_penalty"] += torque_penalty

        # 보상을 옛 명령으로 다 쓴 뒤에 명령을 갱신한다 — 다음 관측이 새 명령을 싣는다.
        self._resample_steering_on_timer()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            base_height = self._robot.data.body_pos_w[:, self.ref_body_index, 2]
            died = base_height < self.cfg.termination_height

            # 총 기울기(total tilt) 종료: gz = projected_gravity_b[2] 는 완전 기립 시 -1, 90° 기울면 0.
            # tilt > max_tilt_deg 이면 gz > -cos(max_tilt) → 종료. 방향 무관하게 대각 사각지대까지 잡는다.
            # (기존 축별 roll/pitch + flipped 조합은 roll≈49° 반쯤 누운 자세가 새는 사각지대가 있었다.)
            gz = self._robot.data.projected_gravity_b[:, 2]
            tilted = gz > -math.cos(math.radians(self.cfg.max_tilt_deg))
            died = died | tilted

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
        root_lin_vel_b_t = self._robot.data.root_link_lin_vel_b[env_ids]  # [n,3]
        root_ang_vel_b_t = self._robot.data.root_link_ang_vel_b[env_ids]  # [n,3]
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
        )  # [n, amp_obs_space=53]
        # 히스토리를 terminal state 기준으로 구성 (_get_observations와 동일한 시프트)
        terminal_buf = self.amp_observation_buffer[env_ids].clone()  # [n, H, 53]
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

            base_terminal = torch.cat([root_pos_obs_xy_t, terminal_buf], dim=-1)  # [n, H, 57]
        else:
            base_terminal = terminal_buf  # [n, H, 53]

        # [R4] 동일한 tan_norm 변환 적용 (live path와 대칭)
        rot_tan_norm_t = _apply_root_rot_tan_norm(terminal_quat_buf, n, self.cfg.num_amp_observations)  # [n, H, 6]
        final_terminal_amp_obs = torch.cat([base_terminal, rot_tan_norm_t], dim=-1)  # [n, H, 59]

        self._terminal_amp_obs[env_ids] = self._append_amp_cond(
            final_terminal_amp_obs.view(n, -1), self._policy_amp_cond(env_ids)
        )
        # ────────────────────────────────────────────────────────────────────────

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 리셋 전략 분기: reset_strategy 에 "stand" 포함 시 rel_stand_envs 비율만큼 정지(default_pos+noise,
        # root 속도 0)로 리셋(정지 출발·저속 안정 학습), 나머지는 RSI(모션 프레임). 기본은 전부 RSI.
        stand_ids = env_ids[:0]
        rsi_ids = env_ids
        if "stand" in self.cfg.reset_strategy and self.cfg.rel_stand_envs > 0.0:
            is_stand = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_stand_envs
            stand_ids = env_ids[is_stand]
            rsi_ids = env_ids[~is_stand]

        # ★ 속도 명령을 **RSI 앞에서** 재샘플링한다. `rsi_match_command` 가 켜지면 초기 클립을 명령
        # 속도에 맞춰 골라야 하므로 명령이 먼저 있어야 한다. `_resample_steering` 은 로봇 상태에
        # 의존하지 않는 순수 난수 샘플링이라 순서를 앞당겨도 안전하다(버퍼 초기화 — 불변규칙 §3).
        self._resample_steering(env_ids)  # type: ignore[arg-type]

        # IsaacLab 3.0: partial-env writes use the *_index API with keyword args.
        # root_state[:, 7:] 는 (RSI 경로에서) 이미 COM 기준으로 보정해 두었다.
        if len(rsi_ids) > 0:
            root_state, joint_pos, joint_vel = self._reset_strategy_rsi(rsi_ids)
            self._robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=rsi_ids)
            self._robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=rsi_ids)
            self._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=rsi_ids)

        if len(stand_ids) > 0:
            root_state, joint_pos, joint_vel = self._reset_strategy_stand(stand_ids)
            self._robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=stand_ids)
            self._robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=stand_ids)
            self._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=stand_ids)

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

    def _reset_strategy_rsi(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference State Initialization — 모션 데이터로 초기화."""
        n = len(env_ids)
        if self.cfg.rsi_match_command:
            # 명령 속도에 가까운 클립에서 시작한다. 명령은 이 함수 호출 **전에** 재샘플링돼 있다.
            motion_ids = self._motion_lib.sample_motions_near_speed(
                self._lin_vel_cmd[env_ids, 0].abs(), self.cfg.rsi_match_temperature
            )
        else:
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

        # motion_lib 속도는 base **link** frame 기준 → world frame 으로 변환한 뒤,
        # sim write API 가 요구하는 **COM** 기준으로 v_com = v_link + ω × r_com 보정을 넣는다.
        # (Leg 는 base COM 이 (-0.130, 0, 0.046) 만큼 떨어져 있어 ω=4 rad/s 에서 0.5 m/s 급 오차가 난다.)
        lin_vel_w = quat_apply(root_quat_xyzw, lin_vel_b)
        ang_vel_w = quat_apply(root_quat_xyzw, ang_vel_b)
        com_offset_w = quat_apply(root_quat_xyzw, self._robot.data.body_com_pos_b[env_ids][:, self.ref_body_index])
        root_state[:, 7:10] = lin_vel_w + torch.cross(ang_vel_w, com_offset_w, dim=-1)
        root_state[:, 10:13] = ang_vel_w

        joint_pos_out = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel_out = self._robot.data.default_joint_vel[env_ids].clone()
        # motion data → IsaacLab joint 순서로 재배열
        joint_pos_out[:, :17] = dof_pos[:, self._motion_dof_indices]
        joint_vel_out[:, :17] = dof_vel[:, self._motion_dof_indices]

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

    def _reset_strategy_stand(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """정지(standing) 초기화 — default_joint_pos + noise, root 속도 0.

        정지 출발·저속 안정성 학습용. AMP agent-history 버퍼는 참조 모션으로 seed 한다
        (discriminator expert 샘플은 collect_reference_motions 에서 별도로 뽑으므로 무관하며,
        seed 는 이후 H(=num_amp_observations) 스텝 내 실제 관측으로 교체된다).
        """
        n = len(env_ids)
        # default_root_state pos 는 env-local → world 로 env_origins 더함, 속도는 0(정지).
        root_state = self._robot.data.default_root_state[env_ids].clone()  # [n,13], 직립
        root_state[:, 0:3] = root_state[:, 0:3] + self.scene.env_origins[env_ids]
        root_state[:, 7:13] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        noise = (torch.rand_like(joint_pos) * 2.0 - 1.0) * self.cfg.stand_reset_joint_noise
        joint_pos = joint_pos + noise
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        # AMP agent-history seed (참조 모션; expert 와 무관, H 스텝 내 씻김)
        amp_obs_buf, root_pos_hist, quat_hist = self._compute_reference_buffers(n)
        self.amp_observation_buffer[env_ids] = amp_obs_buf
        self._amp_quat_buf[env_ids] = quat_hist
        if self.cfg.include_rel_track_obs:
            self._hist_root_pos_w[env_ids] = root_pos_hist

        return root_state, joint_pos, joint_vel

    # ──────────────────────────────────────────────────────────
    # 속도 명령 관리 (재샘플링)
    # ──────────────────────────────────────────────────────────

    def _resample_steering(self, env_ids: torch.Tensor):
        """lin_vel_cmd(vx, vy), yaw_vel_cmd, 타이머를 재샘플링합니다."""
        n = len(env_ids)

        # vx: [lin_vel_x_min, lin_vel_x_max]
        vx = (
            torch.rand(n, device=self.device) * (self.cfg.lin_vel_x_max - self.cfg.lin_vel_x_min)
            + self.cfg.lin_vel_x_min
        )
        # command deadzone: |vx| <= cmd_deadzone → 0 (저속서 정지/default 유지 유도)
        if self.cfg.cmd_deadzone > 0.0:
            vx = torch.where(vx.abs() <= self.cfg.cmd_deadzone, torch.zeros_like(vx), vx)
        self._lin_vel_cmd[env_ids, 0] = vx
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

        amp_obs_buf = amp_obs.view(num_samples, n_hist, -1)  # [N, H, 53]
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
        expert_cond = None
        if motion_ids is None:
            # 조건부 disc: 클립 id 와 조건 라벨을 함께 뽑는다 (`amp_cond_expert_sampling` 참조).
            motion_ids, expert_cond = self._sample_expert_motions(num_samples)
        elif self.amp_cond_dim > 0:
            expert_cond = self._expert_amp_cond(motion_ids)
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

            base_amp = torch.cat([root_pos_obs_xy, amp_obs_buf], dim=-1)  # [N, H, 57]
        else:
            base_amp = amp_obs_buf  # [N, H, 53]

        # [R4] expert 측 동일한 tan_norm 변환 (live path와 완전 대칭)
        rot_tan_norm = _apply_root_rot_tan_norm(quat_hist, num_samples, n_hist)  # [N, H, 6]
        final_amp_obs = torch.cat([base_amp, rot_tan_norm], dim=-1)  # [N, H, 59]

        return self._append_amp_cond(final_amp_obs.view(num_samples, -1), expert_cond)

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
    """AMP disc 기본 관측 벡터 계산 (53-dim).

    dof_pos(17) + dof_vel(17) + root_height(1) + lin_vel(3) + ang_vel(3) + foot_pos(12)
    root_rot_tan_norm(6)은 _apply_root_rot_tan_norm()에서 window stacking 후 추가됨 (→ 59-dim).
    """
    return torch.cat(
        [
            dof_pos,  # 17
            dof_vel,  # 17
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
