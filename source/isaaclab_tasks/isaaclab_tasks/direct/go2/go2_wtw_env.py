# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, SPHERE_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_yaw,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    quat_apply_inverse,
)

from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


axis_angle_to_quat = quat_from_angle_axis
transform_quat_by_quat = quat_mul
transform_by_quat = quat_apply
inv_quat = quat_inv


class WTWEnv(DirectRLEnv):
    cfg: Go2FlatEnvCfg | Go2RoughEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._last_processed_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._last_last_jrocessed_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        # X/Y linear velocity and yaw angular velocity commands
        self.num_commands = self.cfg.num_commands
        self._commands = torch.zeros(self.num_envs, self.num_commands, device=self.device)
        self.dt = self.step_dt

        # For walk-these-ways
        self.gait_indices = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

        if self.cfg.history_observation:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.history_len, self.cfg.num_prio_obs, device=self.device, dtype=torch.float
            )

        self.rew_buf_pos = torch.zeros((self.num_envs,), device=self.device)
        self.rew_buf_neg = torch.zeros((self.num_envs,), device=self.device)
        self.command_curriculum = self.cfg.command_curriculum
        self.curriculum_rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        if self.command_curriculum:
            self.curriculum_lin_vel_x = torch.full((self.num_envs,), 0.0, device=self.device)
            self.curriculum_ang_vel = torch.full((self.num_envs,), 0.0, device=self.device)
            self.curriculum_step = self.cfg.curriculum_step
            self.curriculum_threshold = self.cfg.curriculum_threshold
            self.max_lin_vel_x = self.cfg.command_cfg["lin_vel_x_range"][1]
            self.max_ang_vel = self.cfg.command_cfg["ang_vel_range"][1]

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
                "undesired_contacts",
                "similar_to_default",
                "feet_clearance_cmd_linear",
                "orientation_control",
                "raibert_heuristic",
                "tracking_contacts_shaped_force",
                "tracking_contacts_shaped_vel",
                "dof_vel_l2",
                "jump",
                "action_smoothness1",
                "action_smoothness2",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        # # Explicitly order feet to ensure [FL, FR, RL, RR] correspondence
        # self._feet_contact_ids = []
        # self._feet_ids = []
        # for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
        #     sensor_ids, _ = self._contact_sensor.find_bodies(name)
        #     robot_ids, _ = self._robot.find_bodies(name)
        #     self._feet_contact_ids.append(sensor_ids[0])
        #     self._feet_ids.append(robot_ids[0])
        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._feet_ids, _ = self._robot.find_bodies(".*foot")

        all_joint_names = self._robot.data.joint_names
        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if "hip" in n],
            dtype=torch.long, device=self.device
        )

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalized_body_names)
        self.set_debug_vis(getattr(self.cfg, "debug_vis", True))

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, Go2RoughEnvCfg):
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
        self._actions = torch.clip(actions.clone(), -self.cfg.clip_actions, self.cfg.clip_actions).to(self.device)
        actions = self._actions.clone()
        if self.cfg.hip_scale_reduction:
            actions[:, self._hip_joint_ids] *= 0.5
        self._processed_actions = self.cfg.action_scale * actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _post_physics_step(self):
        sample_interval = int(self.cfg.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._contact_target_step()

    def _contact_target_step(self):
        if self.num_commands > 4:
            frequencies = self._commands[:, 4]
            phases = self._commands[:, 5]
            offsets = self._commands[:, 6]
            bounds = self._commands[:, 7]
            durations = self._commands[:, 8]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases,
            ]

            # self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) >= durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs])
                )

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = 0.07
            smoothing_cdf_start = torch.distributions.normal.Normal(
                0, kappa
            ).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)
            ) + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)
            )
            smoothing_multiplier_FR = smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)
            ) + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)
            )
            smoothing_multiplier_RL = smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)
            ) + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)
            )
            smoothing_multiplier_RR = smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)
            ) + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)
            )

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.num_commands > 9:
            self.desired_footswing_height = self._commands[:, 9]

    def _get_observations(self) -> dict:
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()

        self._last_last_jrocessed_actions = self._last_processed_actions.clone()
        self._last_processed_actions = self._processed_actions.clone()

        height_data = None
        if isinstance(self.cfg, Go2RoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        if self.cfg.prev_actions:
            obs = torch.cat([obs, self._previous_actions], dim=-1)

        if self.cfg.timing_parameter:
            obs = torch.cat([obs, self.gait_indices.unsqueeze(1)], dim=-1)

        if self.cfg.clock_inputs:
            obs = torch.cat([obs, self.clock_inputs], dim=-1)

        observations = {"policy": obs}

        if height_data is not None:
            observations["scan"] = height_data

        if self.cfg.history_observation:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([obs] * self.cfg.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], obs.unsqueeze(1)], dim=1),
            )
            observations["history"] = self.obs_history_buf

        if self.cfg.priv_latent:
            priv_obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        self._robot.data.root_lin_vel_b,
                        self._robot.data.root_ang_vel_b,
                        torch.tensor(self._robot.root_physx_view.get_masses(), device=self.device),
                        torch.tensor(
                            self._robot.root_physx_view.get_material_properties().reshape(self.num_envs, -1),
                            device=self.device,
                        ),
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
            observations["priv"] = priv_obs
        return observations

    def _get_rewards(self) -> torch.Tensor:
        commands = getattr(self, "commands", self._commands)
        base_lin_vel = getattr(self, "base_lin_vel", self._robot.data.root_lin_vel_b)
        base_ang_vel = getattr(self, "base_ang_vel", self._robot.data.root_ang_vel_b)
        base_pos = getattr(self, "base_pos", self._robot.data.root_link_pos_w)
        projected_gravity = getattr(self, "projected_gravity", self._robot.data.projected_gravity_b)
        # dof_pos = getattr(self, "dof_pos", self._robot.data.joint_pos)
        # default_dof_pos = getattr(self, "default_dof_pos", self._robot.data.default_joint_pos)
        dof_vel = getattr(self, "dof_vel", self._robot.data.joint_vel)
        # torques = getattr(self, "torques", self._robot.data.applied_torque)
        # actions = getattr(self, "actions", self._actions)
        # previous_actions = getattr(self, "previous_actions", self._previous_actions)

        processed_actions = self._processed_actions
        last_processed_actions = self._last_processed_actions
        last_last_processed_actions = self._last_last_jrocessed_actions

        foot_positions = self._robot.data.body_link_pos_w[:, self._feet_ids, :]

        # print(self._robot.data.body_com_pos_w[:, self._feet_ids, :])
        # print(self._contact_sensor.find_bodies(".*foot"))

        foot_velocities = self._robot.data.body_link_lin_vel_w[:, self._feet_ids, :]
        base_quat = self._robot.data.root_com_quat_w

        low_lin_vel = torch.norm(self._commands[:, :2], dim=1) < 0.1
        low_ang_vel = torch.abs(self._commands[:, 2]) < 0.05
        both_low = low_lin_vel & low_ang_vel

        def _scale(name: str, default: float = 1.0) -> float:
            return getattr(self.cfg, f"{name}_reward_scale", default)

        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel = torch.exp(-lin_vel_error / self.cfg.tracking_sigma)
        # yaw rate tracking
        yaw_rate_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
        tracking_ang_vel = torch.exp(-yaw_rate_error / self.cfg.tracking_sigma)
        # z velocity tracking
        z_vel_error = torch.square(base_lin_vel[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        # first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
        # last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids]
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (torch.norm(commands[:, :2], dim=1) > 0.1)
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        # flat_orientation = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        # # Similar to default
        # similar_to_default = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)
        # # similar_to_default[~both_low] = 0.

        # # base height
        # base_height = torch.square(base_pos[:, 2] - self._robot.data.default_root_state[:, 2])

        # # dof acceleration penalty
        # last_dof_vel = getattr(self, "last_dof_vel", dof_vel)
        # dt = getattr(self, "dt", self.step_dt)
        # dof_acc = torch.sum(torch.square((last_dof_vel - dof_vel) / dt), dim=1)

        # # action rate penalty (alt)
        # action_rate_alt = torch.norm(getattr(self, "previous_actions", previous_actions) - actions, dim=1)

        # # torque change penalty
        # last_torques = getattr(self, "last_torques", torques)
        # delta_torques = torch.sum(torch.square(torques - last_torques), dim=1)

        # # torque magnitude penalty (optionally weighted)
        # torques_l2_weighted = torch.sum(torch.square(torques), dim=1)

        # action_smoothness1
        diff1 = torch.square(processed_actions - last_processed_actions)
        action_smoothness1 = torch.sum(diff1 * (self._previous_actions != 0), dim=1)

        diff2 = torch.square(processed_actions - 2 * last_processed_actions + last_last_processed_actions)
        diff2 = diff2 * (self._previous_actions != 0)
        action_smoothness2 = torch.sum(diff2 * (self._previous_previous_actions != 0), dim=1)

        # penalize clearance error against command
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (foot_positions[:, :, 2]).view(self.num_envs, -1)
        target_height = commands[:, 9].unsqueeze(1) * phases + 0.02
        feet_clearance_cmd_linear = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        feet_clearance_cmd_linear = torch.sum(feet_clearance_cmd_linear, dim=1)
        feet_clearance_cmd_linear[both_low] = 0.

        # orientation control tracking from commands
        roll_pitch_commands = commands[:, 10:12]
        quat_roll = axis_angle_to_quat(
            -roll_pitch_commands[:, 1], torch.tensor([1, 0, 0], device=self.device, dtype=torch.float)
        )
        quat_pitch = axis_angle_to_quat(
            -roll_pitch_commands[:, 0], torch.tensor([0, 1, 0], device=self.device, dtype=torch.float)
        )
        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_apply_inverse(desired_base_quat, self.global_gravity)
        orientation_control = torch.sum(
            torch.square(projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1
        )

        # raibert heuristic foot placement error
        cur_footsteps_translated = foot_positions - base_pos.unsqueeze(1)
        # print(cur_footsteps_translated[0])
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(inv_quat(base_quat), cur_footsteps_translated[:, i, :])

        x_vel_des = commands[:, 0:1]
        frequencies = commands[:, 4]

        if self.num_commands >= 13:
            desired_stance_width = commands[:, 12:13]
            desired_ys_nom = torch.cat(
                [
                    desired_stance_width / 2,
                    -desired_stance_width / 2,
                    desired_stance_width / 2,
                    -desired_stance_width / 2,
                ],
                dim=1,
            )
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor(
                [
                    -desired_stance_width / 2,
                    desired_stance_width / 2,
                    -desired_stance_width / 2,
                    desired_stance_width / 2,
                ],
                device=self.device,
            ).unsqueeze(0)

        if self.num_commands >= 14:
            desired_stance_length = commands[:, 13:14]
            desired_xs_nom = torch.cat(
                [
                    desired_stance_length / 2,
                    desired_stance_length / 2,
                    -desired_stance_length / 2,
                    -desired_stance_length / 2,
                ],
                dim=1,
            )
        else:
            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor(
                [
                    desired_stance_length / 2,
                    desired_stance_length / 2,
                    -desired_stance_length / 2,
                    -desired_stance_length / 2,
                ],
                device=self.device,
            ).unsqueeze(0)

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        yaw_vel_des = commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        desired_footsteps_body_frame_xy = desired_footsteps_body_frame
        desired_footsteps_body_frame_z = torch.zeros_like(desired_footsteps_body_frame_xy[..., :1])
        desired_footsteps_body_frame_3d = torch.cat(
            (desired_footsteps_body_frame_xy, desired_footsteps_body_frame_z), dim=2
        )

        num_envs = self.num_envs
        num_feet = 4
        q_reshaped = base_quat.unsqueeze(1).repeat(1, num_feet, 1).reshape(num_envs * num_feet, 4)
        v_reshaped = desired_footsteps_body_frame_3d.reshape(num_envs * num_feet, 3)
        rotated_footsteps_world_orientation = quat_apply(q_reshaped, v_reshaped).view(num_envs, num_feet, 3)
        self.desired_footsteps_world_frame = rotated_footsteps_world_orientation + base_pos.unsqueeze(1)
        self.desired_footsteps_world_frame[:, :, 2] = 0.0

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        raibert_heuristic = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        self._visualize_desired_footsteps()

        # contact shaping (force)

        foot_forces = torch.mean(torch.norm(net_contact_forces[:, :, self._feet_contact_ids], dim=-1), dim=1)

        desired_contact = self.desired_contact_states
        tracking_contacts_shaped_force = 0
        for i in range(4):
            tracking_contacts_shaped_force += -(
                (1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.gait_force_sigma))
            )
        tracking_contacts_shaped_force = tracking_contacts_shaped_force / 4
        tracking_contacts_shaped_force[both_low] = 0.

        # contact shaping (velocity)
        foot_velocities = torch.norm(foot_velocities, dim=2).view(self.num_envs, -1)
        tracking_contacts_shaped_vel = 0
        for i in range(4):
            tracking_contacts_shaped_vel += -(
                desired_contact[:, i] * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.gait_vel_sigma))
            )
        tracking_contacts_shaped_vel = tracking_contacts_shaped_vel / 4
        tracking_contacts_shaped_vel[both_low] = 0.

        # dof velocity penalty
        dof_vel_penalty = dof_vel[:]
        dof_vel_l2 = torch.sum(torch.square(dof_vel_penalty), dim=1)

        # jump height tracking reward
        body_height = base_pos[:, 2]
        jump_height_target = commands[:, 3] + self.cfg.base_height_target
        jump = -torch.square(body_height - jump_height_target)

        rewards = {
            "track_lin_vel_xy_exp": tracking_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": tracking_ang_vel * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            # "similar_to_default": similar_to_default * self.cfg.similar_to_default_reward_scale * self.step_dt,
            "feet_clearance_cmd_linear": feet_clearance_cmd_linear * _scale("feet_clearance_cmd_linear") * self.step_dt,
            "orientation_control": orientation_control * _scale("orientation_control") * self.step_dt,
            "raibert_heuristic": raibert_heuristic * _scale("raibert_heuristic") * self.step_dt,
            "tracking_contacts_shaped_force": tracking_contacts_shaped_force
            * _scale("tracking_contacts_shaped_force")
            * self.step_dt,
            "tracking_contacts_shaped_vel": tracking_contacts_shaped_vel
            * _scale("tracking_contacts_shaped_vel")
            * self.step_dt,
            "dof_vel_l2": dof_vel_l2 * self.cfg.dof_vel_reward_scale * self.step_dt,
            "jump": jump * _scale("jump") * self.step_dt,
            "action_smoothness1": action_smoothness1 * _scale("action_smoothness1") * self.step_dt,
            "action_smoothness2": action_smoothness2 * _scale("action_smoothness2") * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        self.rew_buf_pos[:] = 0.0
        self.rew_buf_neg[:] = 0.0
        for key, value in rewards.items():
            self._episode_sums[key] += value
            if torch.sum(value) >= 0:
                self.rew_buf_pos += value
            elif torch.sum(value) <= 0:
                self.rew_buf_neg += value

        reward = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.sigma_rew_neg)
        self.curriculum_rew_buf += reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0
        if self.cfg.history_observation:
            self.obs_history_buf[env_ids, :, :] = 0.0
        self.gait_indices[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
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

    def _visualize_desired_footsteps(self):
        if hasattr(self, "footsteps_visualizer"):
            self.footsteps_visualizer.visualize(self.desired_footsteps_world_frame.flatten(0, 1))

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "link_pos_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.02
                marker_cfg.prim_path = "/Visuals/Debug/link_positions"
                self.link_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.link_pos_visualizer.set_visibility(True)
            if not hasattr(self, "footsteps_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.02
                marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Debug/footsteps"
                self.footsteps_visualizer = VisualizationMarkers(marker_cfg)
            self.footsteps_visualizer.set_visibility(True)
        else:
            if hasattr(self, "link_pos_visualizer"):
                self.link_pos_visualizer.set_visibility(False)
            if hasattr(self, "footsteps_visualizer"):
                self.footsteps_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        link_positions = self._robot.data.body_link_pos_w[:, self._feet_ids, :].reshape(-1, 3)
        self.link_pos_visualizer.visualize(link_positions)

    def _resample_commands(self, env_ids: torch.Tensor):
        command_keys_in_order = [
            "lin_vel_x_range",
            "lin_vel_y_range",
            "ang_vel_range",
            "body_height_cmd_range",
            "gait_frequency_cmd_range",
            "gait_phase_cmd_range",
            "gait_offset_cmd_range",
            "gait_bound_cmd_range",
            "gait_duration_cmd_range",
            "footswing_height_range",
            "body_pitch_range",
            "body_roll_range",
            "stance_width_range",
            "stance_length_range",
        ]
        if self.command_curriculum:
            for i in range(self.cfg.num_commands):
                if i < len(command_keys_in_order):
                    key = command_keys_in_order[i]
                    if key in self.cfg.command_cfg:
                        lower, upper = self.cfg.command_cfg[key]
                        if i == 0:  # lin_vel_x에 curriculum 적용
                            curr = self.curriculum_lin_vel_x[env_ids]
                            use_curriculum = curr < upper
                            low = torch.where(use_curriculum, curr - self.curriculum_step, torch.full_like(curr, lower))
                            high = torch.where(
                                use_curriculum, curr + self.curriculum_step, torch.full_like(curr, upper)
                            )
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
                            self._commands[env_ids, i] = torch_rand_float(lower, upper, (len(env_ids),), self.device)
            low_lin_vel = torch.norm(self._commands[env_ids, :2], dim=1) < 0.1
            low_ang_vel = torch.abs(self._commands[env_ids, 2]) < 0.05
            both_low = low_lin_vel & low_ang_vel

            self._commands[env_ids[both_low], 0] = 0.0
            self._commands[env_ids[both_low], 1] = 0.0
            self._commands[env_ids[both_low], 2] = 0.0

        else:
            for i in range(self.cfg.num_commands):
                if i < len(command_keys_in_order):
                    key = command_keys_in_order[i]
                    if key in self.cfg.command_cfg:
                        lower, upper = self.cfg.command_cfg[key]
                        self._commands[env_ids, i] = torch_rand_float(lower, upper, (len(env_ids),), self.device)

        if self.num_commands > 5:
            strategy_indices = torch.randint(0, 5, (len(env_ids),), device=self.device)

            # Strategy 1
            idx = env_ids[strategy_indices == 0]
            if len(idx) > 0:
                self._commands[idx, 5] = (self._commands[idx, 5] / 2.0 - 0.25) % 1.0
                self._commands[idx, 6] = (self._commands[idx, 6] / 2.0 - 0.25) % 1.0
                self._commands[idx, 7] = (self._commands[idx, 7] / 2.0 - 0.25) % 1.0

            # Strategy 2
            idx = env_ids[strategy_indices == 1]
            if len(idx) > 0:
                self._commands[idx, 5] = self._commands[idx, 5] / 2.0 + 0.25
                self._commands[idx, 6] = 0.0
                self._commands[idx, 7] = 0.0

            # Strategy 3
            idx = env_ids[strategy_indices == 2]
            if len(idx) > 0:
                self._commands[idx, 5] = 0.0
                self._commands[idx, 6] = self._commands[idx, 6] / 2.0 + 0.25
                self._commands[idx, 7] = 0.0

            # Strategy 4
            idx = env_ids[strategy_indices == 3]
            if len(idx) > 0:
                self._commands[idx, 5] = 0.0
                self._commands[idx, 6] = 0.0
                self._commands[idx, 7] = self._commands[idx, 7] / 2.0 + 0.25

            # Strategy 5
            idx = env_ids[strategy_indices == 4]
            if len(idx) > 0:
                self._commands[idx, 5] = self._commands[idx, 5] / 2.0 + 0.25
                self._commands[idx, 6] = self._commands[idx, 6] / 2.0 + 0.25
                self._commands[idx, 7] = self._commands[idx, 7] / 2.0 + 0.25
