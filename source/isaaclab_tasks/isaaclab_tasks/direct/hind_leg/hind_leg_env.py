# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .hind_leg_env_cfg import HindLegFlatEnvCfg, HindLegRoughEnvCfg


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class HindLegEnv(DirectRLEnv):
    cfg: HindLegFlatEnvCfg | HindLegRoughEnvCfg

    def __init__(self, cfg: HindLegFlatEnvCfg | HindLegRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.command_curriculum = self.cfg.command_curriculum
        self.curriculum_rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        if self.command_curriculum:
            self.curriculum_lin_vel_x = torch.full((self.num_envs,), 0.0, device=self.device)
            self.curriculum_ang_vel = torch.full((self.num_envs,), 0.0, device=self.device)
            self.curriculum_step = self.cfg.curriculum_step
            self.curriculum_threshold = self.cfg.curriculum_threshold
            self.max_lin_vel_x = self.cfg.command_cfg["lin_vel_x_range"][1]
            self.max_ang_vel = self.cfg.command_cfg["ang_vel_range"][1]

        if self.cfg.history_observation:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.history_len, self.cfg.num_prio_obs, device=self.device, dtype=torch.float
            )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "similar_to_default",
                "base_height",
                "termination",
                "gait_stance",
                "gait_swing",
                "foot_slip",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        # Contact sensor body ids for swing/stance gating (.*foot_contact.*)
        self._feet_ids, feet_contact_names = self._contact_sensor.find_bodies(".*foot_contact.*")
        # Robot articulation body ids for sole position and velocity (.*foot_contact.*)
        # These are in a different index space than _feet_ids (sensor vs. robot body indices).
        self._sole_body_ids, sole_body_names = self._robot.find_bodies(".*foot_contact.*")
        assert len(self._sole_body_ids) == 2, (
            f"Expected exactly 2 sole bodies matching '.*foot_contact.*', got {len(self._sole_body_ids)}: {sole_body_names}"
        )
        # Verify foot ordering is consistent between contact sensor and robot body indices.
        # A mismatch would silently swap swing/stance gating between feet, corrupting training.
        assert [n.split("/")[-1] for n in feet_contact_names] == [n.split("/")[-1] for n in sole_body_names], (
            f"Foot name order mismatch — contact sensor: {feet_contact_names}, sole bodies: {sole_body_names}"
        )
        # Sole rest-z is captured lazily on the first _get_rewards call (body_pos_w not valid in __init__).
        # It is a flat-ground constant, reset-invariant, so it is NOT initialized in _reset_idx.
        self._sole_rest_z: torch.Tensor | None = None

        # Gait phase clock ∈ [0, 1) — one scalar per env, advances each step by step_dt / gait_period.
        # Initialized to zero here; _reset_idx randomizes it per-episode for decorrelation.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalzied_body_names)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, HindLegRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        # Advance gait phase by one env-step before reward/obs read this step's value.
        self._gait_phase = (self._gait_phase + self.step_dt / self.cfg.gait_period) % 1.0
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # print(self._robot.joint_names)
        self._previous_actions = self._actions.clone()

        # Phase clock obs (4-dim) — HL at phase φ, HR at anti-phase φ+0.5.
        # _sole_body_ids order is [HL, HR] (asserted in __init__); index 1 (HR) gets 0.5 offset.
        phi_hl = self._gait_phase  # (N,)
        phi_hr = (self._gait_phase + 0.5) % 1.0  # (N,) — anti-phase
        clock_obs = torch.stack(
            [
                torch.sin(2.0 * torch.pi * phi_hl),
                torch.cos(2.0 * torch.pi * phi_hl),
                torch.sin(2.0 * torch.pi * phi_hr),
                torch.cos(2.0 * torch.pi * phi_hr),
            ],
            dim=1,
        )  # (N, 4)

        if self.cfg.history_observation:
            obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        # self._robot.data.root_lin_vel_b,
                        # self._robot.data.root_ang_vel_b,
                        self._robot.data.projected_gravity_b,
                        self._commands,
                        self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                        self._robot.data.joint_vel,
                        # height_data,
                        self._actions,
                        clock_obs if self.cfg.clock_inputs else None,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
        else:
            obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        self._robot.data.root_lin_vel_b,
                        self._robot.data.root_ang_vel_b,
                        self._robot.data.projected_gravity_b,
                        self._commands,
                        self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                        self._robot.data.joint_vel,
                        # height_data,
                        self._actions,
                        clock_obs if self.cfg.clock_inputs else None,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
        observations = {"policy": obs}

        height_data = None
        if isinstance(self.cfg, HindLegRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)

        if height_data is not None:
            observations["scan"] = height_data

        if self.cfg.history_observation:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([obs] * self.cfg.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], obs.unsqueeze(1)], dim=1),
            )
            observations["history"] = self.obs_history_buf

        if self.cfg.priv_explicit:
            priv_explicit = torch.cat(
                [
                    self._robot.data.root_lin_vel_b * 2.0,  # 3
                    self._robot.data.root_ang_vel_b * 0.25,  # 3
                ],
                dim=-1,
            )
            observations["priv_explicit"] = priv_explicit
        if self.cfg.priv_latent:
            priv_obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        torch.tensor(self._robot.root_physx_view.get_masses(), device=self.device),
                        torch.tensor(
                            self._robot.root_physx_view.get_material_properties(),
                            device=self.device,
                        ).reshape(self.num_envs, -1),
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
            observations["priv_latent"] = priv_obs

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.1)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.1)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.2) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # Similar to default
        similar_to_default = torch.sum(
            torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1
        )

        # base height
        base_height = torch.square(self._robot.data.root_link_pos_w[:, 2] - self._robot.data.default_root_state[:, 2])

        # termination
        termination = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        ).float()

        # Phase clock — HL at gait_phase, HR at anti-phase (offset 0.5).
        # _sole_body_ids order is [HL, HR] (asserted in __init__).
        phi_hl = self._gait_phase  # (N,)
        phi_hr = (self._gait_phase + 0.5) % 1.0  # (N,)
        theta = 2.0 * torch.pi * torch.stack([phi_hl, phi_hr], dim=1)  # (N, 2)

        # Smooth swing/stance coefficients via tanh (50/50 duty, gait_phase_sharpness controls ramp steepness).
        # E_swing ≈ 1 during swing half-cycle, ≈ 0 during stance; E_stance is complement.
        E_swing = 0.5 * (1.0 + torch.tanh(self.cfg.gait_phase_sharpness * torch.sin(theta)))  # (N, 2)
        E_stance = 1.0 - E_swing  # (N, 2)

        # Command-gated standing override: when command is near-zero, zero out E_swing so the robot
        # stands still instead of marching in place. E_stance remains untouched (two feet in contact
        # scores ~2.0 via gait_stance, strictly beating any march pattern). yaw is included in the
        # gate so that a pure-yaw command still allows swing. Phase clock is NOT touched — obs/checkpoint
        # compatibility is preserved.
        cmd_xy = torch.norm(self._commands[:, :2], dim=1)  # (N,)
        standing = (cmd_xy < self.cfg.standing_vel_threshold) & (
            self._commands[:, 2].abs() < self.cfg.standing_yaw_threshold
        )  # (N,) bool
        E_swing = torch.where(standing.unsqueeze(1), torch.zeros_like(E_swing), E_swing)
        E_stance = 1.0 - E_swing

        # sole_rest_z: lazily captured on first physics step (body_pos_w invalid during __init__).
        # Flat-ground constant; reset-invariant; NOT re-zeroed in _reset_idx.
        if self._sole_rest_z is None:
            self._sole_rest_z = self._robot.data.body_pos_w[:, self._sole_body_ids, 2].mean(dim=0).detach()
        # Contact gate: True = foot in contact (force > 1N).
        # net_contact_forces shape: (N, history, n_contact_bodies); _feet_ids selects sole contact bodies.
        contact_filt = (
            torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1.0
        )  # (N, n_feet)
        # Sole world-z minus rest-z = lift above ground (reset-invariant flat-ground baseline).
        sole_z = self._robot.data.body_pos_w[:, self._sole_body_ids, 2]  # (N, n_feet)
        lift = sole_z - self._sole_rest_z  # (N, n_feet)

        # (A) Phase-scheduled stance reward — bonus when scheduled-stance foot is in contact (scale > 0).
        gait_stance = torch.sum(E_stance * contact_filt.float(), dim=1)  # (N,) ∈ [0, 2]

        # (B) Phase-scheduled swing clearance reward — bonus when scheduled-swing foot is lifted (scale > 0).
        # clearance ∈ [0, 1]: saturates at gait_swing_height; partial lift already earns partial reward.
        clearance = torch.clamp(lift / self.cfg.gait_swing_height, 0.0, 1.0)  # (N, n_feet)
        gait_swing = torch.sum(E_swing * clearance, dim=1)  # (N,) ∈ [0, 2]

        # (C) Contact-gated anti-slip penalty (scale < 0).
        # Penalises stance feet that are sliding; complementary to swing clearance above.
        sole_vel_xy = torch.norm(self._robot.data.body_lin_vel_w[:, self._sole_body_ids, :2], dim=-1)  # (N, n_feet)
        slip_pen = torch.sum(contact_filt.float() * sole_vel_xy**2, dim=1)  # (N,) ≥ 0; scale < 0 → penalty

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "similar_to_default": similar_to_default * self.cfg.similar_to_default_reward_scale * self.step_dt,
            "base_height": base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            "gait_stance": gait_stance * self.cfg.gait_stance_reward_scale * self.step_dt,
            "gait_swing": gait_swing * self.cfg.gait_swing_reward_scale * self.step_dt,
            "foot_slip": slip_pen * self.cfg.foot_slip_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        self.curriculum_rew_buf += reward
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Randomize gait phase for reset envs to decorrelate episodes across parallel envs.
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        if self.cfg.history_observation:
            self.obs_history_buf[env_ids, :, :] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_state[:, 7:], env_ids=env_ids)
        self._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        if self.command_curriculum:
            extras["mean_lin_vel_x"] = torch.mean(self._commands[:, 0])
            extras["max_lin_vel_x"] = torch.max(self._commands[:, 0])
            extras["min_lin_vel_x"] = torch.min(self._commands[:, 0])
            extras["mean_ang_vel"] = torch.mean(torch.abs(self._commands[:, 2]))
            extras["curriculum_rew"] = torch.mean(self.curriculum_rew_buf[env_ids])
            success_mask = self.curriculum_rew_buf[env_ids] > self.curriculum_threshold
            self.curriculum_lin_vel_x[env_ids] += success_mask * self.curriculum_step
            self.curriculum_ang_vel[env_ids] += success_mask * self.curriculum_step
            self.curriculum_rew_buf[env_ids] = 0.0

        self.extras["log"].update(extras)

        # Sample new commands
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        if self.command_curriculum:
            command_keys_in_order = [
                "lin_vel_x_range",
                "lin_vel_y_range",
                "ang_vel_range",
            ]
            for i in range(self.cfg.num_commands):
                if i < len(command_keys_in_order):
                    key = command_keys_in_order[i]
                    if key in self.cfg.command_cfg:
                        lower, upper = self.cfg.command_cfg[key]
                        if i == 0:  # lin_vel_x에 curriculum 적용
                            curr = self.curriculum_lin_vel_x[env_ids]
                            use_curriculum = curr < upper
                            low = torch.where(use_curriculum, curr - self.curriculum_step, torch.full_like(curr, lower))
                            high = torch.where(use_curriculum, curr, torch.full_like(curr, upper))
                            self._commands[env_ids, i] = torch.lerp(
                                low, high, torch.rand(len(env_ids), device=self.device)
                            )
                        elif i == 1 or i == 2:  # ang_vel에 curriculum 적용
                            curr = self.curriculum_ang_vel[env_ids]
                            use_curriculum = curr < upper
                            # random sign 선택
                            direction = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1  # {-1, +1}
                            signed_curr = curr * direction.float()
                            low = torch.where(
                                use_curriculum, signed_curr - self.curriculum_step, torch.full_like(curr, lower)
                            )
                            high = torch.where(use_curriculum, signed_curr, torch.full_like(curr, upper))
                            self._commands[env_ids, i] = torch.lerp(
                                low, high, torch.rand(len(env_ids), device=self.device)
                            )
        else:
            self._commands[env_ids, 0] = torch_rand_float(
                *self.cfg.command_cfg["lin_vel_x_range"], (len(env_ids),), self.device
            )
            self._commands[env_ids, 1] = torch_rand_float(
                *self.cfg.command_cfg["lin_vel_y_range"], (len(env_ids),), self.device
            )
            self._commands[env_ids, 2] = torch_rand_float(
                *self.cfg.command_cfg["ang_vel_range"], (len(env_ids),), self.device
            )

        # 일부 env는 정지(standing) 명령으로 강제 → 정책이 cmd=0 평형을 학습하게 함
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_standing_envs
        self._commands[env_ids[standing], :] = 0.0
