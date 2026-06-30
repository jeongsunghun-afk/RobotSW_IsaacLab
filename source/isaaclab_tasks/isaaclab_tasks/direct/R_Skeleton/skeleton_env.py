# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import socket

import gymnasium as gym
import torch

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor, FrameTransformer, RayCaster

from .skeleton_env_cfg import SkeletonEnvCfg, SkeletonHistoryEnvCfg, SkeletonHistoryFixedEnvCfg, SkeletonRoughEnvCfg


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class SkeletonEnv(DirectRLEnv):
    cfg: SkeletonEnvCfg | SkeletonHistoryEnvCfg | SkeletonRoughEnvCfg | SkeletonHistoryFixedEnvCfg

    def __init__(
        self,
        cfg: SkeletonEnvCfg | SkeletonHistoryEnvCfg | SkeletonRoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
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
        self.rew_buf_pos = torch.zeros((self.num_envs,), device=self.device)
        self.rew_buf_neg = torch.zeros((self.num_envs,), device=self.device)
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
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._neck_ids, _ = self._contact_sensor.find_bodies([".*neck_p", ".*neck_r", ".*neck_y"])
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*toe")
        self._feet_body_ids, _ = self._robot.find_bodies(".*toe")
        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*ankle_r")
        # self._feet_body_ids, _ = self._robot.find_bodies(".*ankle_r")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalized_contact_link_names)

        # Debug Visualization
        marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ContactForces"
        marker_cfg.markers["arrow"].scale = (0.2, 0.02, 0.02)
        self._contact_forces_visualizer = VisualizationMarkers(marker_cfg)

        # UDP Socket for Debugging
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_addr = ("127.0.0.1", 5005)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # FrameTransformer: SkeletonHistoryFixedEnvCfg처럼 foot_frame이 설정된 경우에만 초기화
        if hasattr(self.cfg, "foot_frame"):
            self._foot_frame_sensor = FrameTransformer(self.cfg.foot_frame)
            self.scene.sensors["foot_frame"] = self._foot_frame_sensor
        else:
            self._foot_frame_sensor = None
        if isinstance(self.cfg, SkeletonRoughEnvCfg):
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

    @property
    def foot_pos_w(self) -> torch.Tensor:
        """발끝의 월드 좌표 위치를 반환합니다. shape: (num_envs, 4, 3) [FL, FR, HL, HR 순서].

        FrameTransformerSensor가 설정된 경우(Fixed joint USD) 사용합니다.
        미설정 시 ankle_r / wrist_r 링크 위치로 폴백합니다.

        Returns:
            Tensor of shape (num_envs, 4, 3): 각 발끝의 월드 좌표 (x, y, z).
        """
        if self._foot_frame_sensor is not None:
            # FrameTransformerSensor 사용 (Fixed joint 환경)
            # target_pos_w: (num_envs, num_targets, 3) — FL, FR, HL, HR 순서
            return self._foot_frame_sensor.data.target_pos_w
        else:
            # 폴백: ankle_r / wrist_r 링크 위치 반환
            return self._robot.data.body_pos_w[:, self._feet_body_ids, :]

    @property
    def foot_pos_b(self) -> torch.Tensor:
        """발끝의 로봇 기준 좌표(base frame) 위치를 반환합니다. shape: (num_envs, 4, 3).

        Returns:
            Tensor of shape (num_envs, 4, 3): 각 발끝의 base frame 좌표 (x, y, z).
        """
        if self._foot_frame_sensor is not None:
            return self._foot_frame_sensor.data.target_pos_source
        else:
            # 폴백: body_pos_w를 base frame으로 변환
            base_pos = self._robot.data.root_pos_w[:, :3].unsqueeze(1)
            return self._robot.data.body_pos_w[:, self._feet_body_ids, :] - base_pos

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clip(actions.clone(), -self.cfg.clip_actions, self.cfg.clip_actions).to(self.device)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # print(self._robot.joint_names)
        self._previous_actions = self._actions.clone()
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
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
        # 관측치 구성 요소별 세부 체크
        observations = {"policy": obs}

        height_data = None
        if isinstance(self.cfg, SkeletonRoughEnvCfg):
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
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
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
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        self.rew_buf_pos[:] = 0.0
        self.rew_buf_neg[:] = 0.0
        for key, value in rewards.items():
            self._episode_sums[key] += value
            # Extract positive and negative components explicitly
            pos_val = torch.clamp(value, min=0.0)
            neg_val = torch.clamp(value, max=0.0)
            self.rew_buf_pos += pos_val
            self.rew_buf_neg += neg_val

        # Linear combination + Survival Bonus

        # Scale down penalties severely for initial learning (can be tuned later)
        penalty_scale = 0.1

        reward = self.rew_buf_pos + (self.rew_buf_neg * penalty_scale)
        self.curriculum_rew_buf += reward
        return reward

    def _post_physics_step(self):
        self._visualize_contact_forces()

        # --- Debug Start ---
        # 1. 가장 높은 토크를 받는 관절 출력
        torques = self._robot.data.applied_torque[0]  # env 0
        max_torque_val, max_torque_idx = torch.max(torch.abs(torques), dim=0)
        joint_names = self._robot.data.joint_names

        # 2. 가장 큰 외부 힘(Contact Force)을 받는 링크 출력
        # net_contact_forces_w: [num_envs, num_bodies, 3]
        net_contact_forces = self._contact_sensor.data.net_forces_w[0]  # env 0
        force_norms = torch.norm(net_contact_forces, dim=-1)
        max_force_val, max_force_idx = torch.max(force_norms, dim=0)
        # sensor_body_names = self._contact_sensor.body_names

        # print(f"[Debug] Max Torque: {max_torque_val.item():.2f} Nm at {joint_names[max_torque_idx]} | "
        #       f"Max Force: {max_force_val.item():.2f} N at {sensor_body_names[max_force_idx]}")
        # --- Debug End ---

        # Send Debug Data via UDP (only for env 0 to avoid flooding)
        if self._udp_sock:
            try:
                # Send data for ALL joints
                # joint_names is a list of strings
                joint_names = self._robot.data.joint_names

                # Convert tensors to lists
                ref_list = self._processed_actions[0, :].tolist()
                pos_list = self._robot.data.joint_pos[0, :].tolist()
                vel_list = self._robot.data.joint_vel[0, :].tolist()
                trq_list = self._robot.data.applied_torque[0, :].tolist()

                data = {
                    "t": self.common_step_counter,
                    "names": joint_names,
                    "ref": ref_list,
                    "pos": pos_list,
                    "vel": vel_list,
                    "trq": trq_list,
                }

                # Use a larger buffer size on receiver side, or compress if needed.
                # JSON serialization might increase size, but for ~38 joints it should be fine (< MTU usually, or fragmented by IP)
                # 38 joints * 4 floats * ~10 bytes + names... might exceed 1 packet if names are long.
                # But localhost handles large packets usually.

                msg = json.dumps(data).encode()
                self._udp_sock.sendto(msg, self._udp_addr)
            except Exception:
                # print(f"UDP Error: {e}")
                pass

    def _visualize_contact_forces(self):
        # Create mapping if not exists
        if not hasattr(self, "_sensor_to_robot_body_map"):
            sensor_body_names = self._contact_sensor.body_names
            robot_body_names = self._robot.data.body_names

            # Map sensor body index to robot body index
            # This assumes sensor bodies are a subset of robot bodies
            mapping = []
            for name in sensor_body_names:
                try:
                    idx = robot_body_names.index(name)
                    mapping.append(idx)
                except ValueError:
                    # print(f"[Warning] Body {name} found in sensor but not in robot data.")
                    mapping.append(-1)

            self._sensor_to_robot_body_map = torch.tensor(mapping, device=self.device, dtype=torch.long)

        # Get contact forces from sensor
        # net_contact_forces_w: [num_envs, num_bodies, 3]
        net_contact_forces = self._contact_sensor.data.net_forces_w

        # Visualize only for env 0
        forces = net_contact_forces[0]  # [num_sensor_bodies, 3]

        # Filter: only show markers if force magnitude > 1.0
        force_mags = torch.norm(forces, dim=-1)
        contact_mask = force_mags > 1.0

        # Get active sensor indices
        active_sensor_indices = torch.nonzero(contact_mask).squeeze(-1)

        if len(active_sensor_indices) > 0:
            # Map to robot body indices to get positions
            active_robot_indices = self._sensor_to_robot_body_map[active_sensor_indices]

            # Filter out invalid mappings (-1)
            valid_mask = active_robot_indices >= 0
            valid_robot_indices = active_robot_indices[valid_mask]

            if len(valid_robot_indices) > 0:
                # Get positions of contacting bodies
                positions = self._robot.data.body_link_pos_w[0, valid_robot_indices, :]

                self._contact_forces_visualizer.set_visibility(True)
                self._contact_forces_visualizer.visualize(positions)
                return

        # If no valid contacts or no contacts at all
        self._contact_forces_visualizer.set_visibility(False)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_base = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        died_neck = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._neck_ids], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        # Base roll/pitch termination (if projected gravity Z > -0.5, means angle > 60 degrees)
        died_ang = self._robot.data.projected_gravity_b[:, 2] > -0.5

        died = died_base | died_neck | died_ang

        # # Debugging death condition for env 0
        # if died[0] and not time_out[0]:
        #     print(f"[DEBUG] Env 0 Died. Base: {died_base[0].item()}, Neck: {died_neck[0].item()}, Ang: {died_ang[0].item()}")

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
