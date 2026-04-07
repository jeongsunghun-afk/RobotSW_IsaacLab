# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 AMP (Adversarial Motion Prior) 환경.

skeleton_amp_env.py 를 기반으로 Go2(12 DOF)에 맞게 조정되었습니다.
모션 참조 데이터는 stmr_go2.py가 생성하는 DeepMimic JSON txt 파일에서 로드합니다.
"""

from __future__ import annotations

import glob
import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.utils.math import sample_uniform as torch_rand_float

from .go2_motion_loader import Go2MotionLoader
from .go2_amp_env_cfg import Go2AmpEnvCfg


class Go2AmpEnv(DirectRLEnv):
    """Go2 AMP imitation 환경.

    AMP 관측 벡터 (amp_observation_space = 43):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        lin_vel(3) + ang_vel(3) + key_body_pos(12)
    """

    cfg: Go2AmpEnvCfg

    # key body 이름 (발 4개) — Go2MotionLoader.BODY_NAMES 와 일치해야 함
    KEY_BODY_NAMES = [
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot",
    ]

    def __init__(self, cfg: Go2AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg

        # 모션 로더 초기화
        motion_file = self.cfg.motion_file
        if os.path.isdir(motion_file):
            txt_files = sorted(glob.glob(os.path.join(motion_file, "*.txt")))
            pkl_files = sorted(glob.glob(os.path.join(motion_file, "*.pkl")))
            motion_files = txt_files + pkl_files
        else:
            motion_files = [motion_file]
        self._motion_loader = Go2MotionLoader(motion_files=motion_files, device=self.device)

        # 로봇에서 body/joint 인덱스 구하기
        self.ref_body_index = self._robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [
            self._robot.data.body_names.index(name) for name in self.KEY_BODY_NAMES
        ]

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # RSI reset 시 선택된 frame의 forward velocity 저장 (in-episode curriculum용)
        self._rsi_ref_vel = torch.zeros(self.num_envs, device=self.device)

        # Pose termination용: 에피소드 내 현재 모션 시간 추적
        self._episode_motion_times = torch.zeros(self.num_envs, device=self.device)
        # RSI로 리셋된 환경만 pose termination 적용 (default 전략 환경 제외)
        self._rsi_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 모션 로더에서 DOF 인덱스 매핑
        robot_joint_names = list(self._robot.data.joint_names)
        try:
            self.motion_dof_indexes = self._motion_loader.get_dof_index(robot_joint_names)
        except AssertionError:
            print("[Go2AmpEnv] DOF 이름 불일치 — 순서대로 1:1 매핑 사용")
            self.motion_dof_indexes = list(range(min(len(robot_joint_names), self._motion_loader.num_dofs)))

        # Go2MotionLoader.BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot", "base"]
        # KEY_BODY_NAMES = [FL, FR, RL, RR] 와 완전히 대응됨
        self.motion_ref_body_index = 4           # base = index 4
        self.motion_key_body_indexes = [0, 1, 2, 3]  # FL, FR, RL, RR

        # AMP 관측 버퍼
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.amp_observation_size,)
        )
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            dtype=torch.float32,
            device=self.device,
        )

        # History Buffer
        if self.cfg.history_observation:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.history_len, self.cfg.observation_space,
                device=self.device, dtype=torch.float,
            )

    # ------------------------------------------------------------------
    # Isaac Lab DirectRLEnv 필수 메서드
    # ------------------------------------------------------------------

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
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self._episode_sums = {
            "tracking_lin_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "tracking_ang_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _post_physics_step(self):
        self._episode_motion_times += self.step_dt

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # In-episode command 재샘플링 (curriculum delta 기반)
        if self.cfg.command_resample_interval > 0:
            resample_mask = (
                (self.episode_length_buf % self.cfg.command_resample_interval == 0)
                & (self.episode_length_buf > 0)
            )
            resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
            if len(resample_ids) > 0:
                self._resample_commands_in_episode(resample_ids)

        root_pos_w = self._robot.data.body_pos_w[:, self.ref_body_index]
        root_quat_w = self._robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        # 월드 기준 상대 거리
        rel_pos = self._robot.data.body_pos_w[:, self.key_body_indexes] - root_pos_w.unsqueeze(1)

        num_keys = rel_pos.shape[1]
        root_quat_expanded = root_quat_w.unsqueeze(1).expand(-1, num_keys, -1)

        # 로컬 프레임으로 변환
        local_key_body_pos = quat_apply_inverse(
            root_quat_expanded.reshape(-1, 4),
            rel_pos.reshape(-1, 3),
        ).view(*rel_pos.shape)

        obs = compute_obs(
            self._robot.data.joint_pos,
            self._robot.data.joint_vel,
            root_pos_w,
            root_lin_vel_b,
            root_ang_vel_b,
            local_key_body_pos,
        )

        # AMP 히스토리 버퍼 업데이트
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs.clone()

        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)
        }

        # Policy 관측치
        policy_obs = torch.cat(
            [
                self._robot.data.projected_gravity_b,                             # 3
                self._commands,                                                    # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel,                                        # 12
                self.actions,                                                      # 12
            ],
            dim=-1,
        )

        observations = {"policy": policy_obs}

        # History
        if self.cfg.history_observation:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([policy_obs] * self.cfg.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], policy_obs.unsqueeze(1)], dim=1),
            )
            observations["history"] = self.obs_history_buf

        # Privileged
        if self.cfg.priv_latent:
            priv_obs = torch.cat(
                [
                    self._robot.data.root_lin_vel_b,                                                              # 3
                    self._robot.data.root_ang_vel_b,                                                              # 3
                    torch.tensor(self._robot.root_physx_view.get_masses(), device=self.device).reshape(self.num_envs, -1),
                    torch.tensor(self._robot.root_physx_view.get_material_properties(), device=self.device).reshape(self.num_envs, -1),
                ],
                dim=-1,
            )
            observations["priv"] = priv_obs

        return observations

    def _get_rewards(self) -> torch.Tensor:
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel = torch.exp(-lin_vel_error / self.cfg.tracking_sigma)

        yaw_rate_error = torch.square(self._commands[:, 2] - base_ang_vel[:, 2])
        tracking_ang_vel = torch.exp(-yaw_rate_error / self.cfg.tracking_sigma)

        lin_vel_reward = tracking_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
        ang_vel_reward = tracking_ang_vel * self.cfg.yaw_rate_reward_scale * self.step_dt

        self._episode_sums["tracking_lin_vel"] += lin_vel_reward
        self._episode_sums["tracking_ang_vel"] += ang_vel_reward

        return lin_vel_reward + ang_vel_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            base_height = self._robot.data.body_pos_w[:, self.ref_body_index, 2]
            died = base_height < self.cfg.termination_height

            flipped = self._robot.data.projected_gravity_b[:, 2] > 0.0
            died = died | flipped

            # Roll / Pitch 각도 기반 termination
            grav_b = self._robot.data.projected_gravity_b  # (N, 3), 정규화된 중력 벡터
            roll = torch.atan2(grav_b[:, 1], -grav_b[:, 2])
            pitch = torch.atan2(-grav_b[:, 0], torch.sqrt(grav_b[:, 1] ** 2 + grav_b[:, 2] ** 2))
            roll_limit = self.cfg.roll_termination_deg * torch.pi / 180.0
            pitch_limit = self.cfg.pitch_termination_deg * torch.pi / 180.0
            bad_orientation = (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)
            died = died | bad_orientation

            if hasattr(self, "contact_sensor"):
                contact_forces = self.contact_sensor.data.net_forces_w
                if contact_forces is not None and contact_forces.numel() > 0:
                    body_names = self._robot.data.body_names
                    bad_contacts = self._get_body_contact(
                        contact_forces, body_names, "base",
                        threshold=self.cfg.contact_force_threshold,
                    )
                    died = died | bad_contacts

            # Pose divergence termination (MimicKit 방식)
            # RSI로 리셋된 환경에서만 reference와 key body 위치 비교
            if self.cfg.pose_termination and self._rsi_active.any():
                times_np = self._episode_motion_times.cpu().numpy()
                (_, _, ref_body_positions, _, _, _) = self._motion_loader.sample(
                    num_samples=self.num_envs, times=times_np
                )
                # motion loader: body_positions[0:4] = 발, root-local 프레임
                ref_key_pos = ref_body_positions[:, self.motion_key_body_indexes]  # (N, 4, 3)

                root_pos_w = self._robot.data.body_pos_w[:, self.ref_body_index]
                root_quat_w = self._robot.data.body_quat_w[:, self.ref_body_index]
                key_pos_w = self._robot.data.body_pos_w[:, self.key_body_indexes]
                rel_pos = key_pos_w - root_pos_w.unsqueeze(1)
                N, K = rel_pos.shape[:2]
                local_key_pos = quat_apply_inverse(
                    root_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
                    rel_pos.reshape(-1, 3),
                ).view(N, K, 3)

                body_diff = ref_key_pos - local_key_pos
                max_dist_sq = torch.sum(body_diff * body_diff, dim=-1).max(dim=-1).values
                pose_fail = (max_dist_sq > self.cfg.pose_termination_dist ** 2) & self._rsi_active
                died = died | pose_fail

            # MimicKit과 동일: 첫 스텝 직후부터만 termination 적용 (physics 정착 시간 확보)
            not_first_step = self.episode_length_buf > 1
            died = died & not_first_step
        else:
            died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        rsi_times = None
        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel, rsi_times = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # pose termination 추적용 모션 시간 초기화
        if rsi_times is not None:
            self._episode_motion_times[env_ids] = torch.tensor(
                rsi_times, dtype=torch.float32, device=self.device
            )
            self._rsi_active[env_ids] = True
        else:
            self._episode_motion_times[env_ids] = 0.0
            self._rsi_active[env_ids] = False

        self._resample_commands(env_ids)

        # RSI frame 속도로 command override (자세와 명령 속도 일치)
        if rsi_times is not None:
            frame_indices = np.clip(
                np.round(rsi_times / self._motion_loader.dt).astype(int),
                0,
                self._motion_loader.num_frames - 1,
            )
            ref_vel_x = self._motion_loader._frame_lin_vel_x[frame_indices]
            ref_vel_tensor = torch.tensor(ref_vel_x, dtype=torch.float32, device=self.device)
            self._commands[env_ids, 0] = ref_vel_tensor
            self._rsi_ref_vel[env_ids] = ref_vel_tensor

        if self.cfg.history_observation:
            self.obs_history_buf[env_ids, :, :] = 0.0

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"].update(extras)

    # ------------------------------------------------------------------
    # 리셋 전략
    # ------------------------------------------------------------------

    def _reset_strategy_default(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self.amp_observation_buffer[env_ids] = 0.0
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)

        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = (
            body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        )
        root_state[:, 2] += 0.05  # 지면 충돌 방지
        root_rot = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 3:7] = root_rot

        root_state[:, 7:10] = quat_apply(root_rot, body_linear_velocities[:, self.motion_ref_body_index])
        root_state[:, 10:13] = quat_apply(root_rot, body_angular_velocities[:, self.motion_ref_body_index])

        n_dofs = len(self.motion_dof_indexes)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, :n_dofs] = dof_positions[:, self.motion_dof_indexes[:n_dofs]]
        joint_vel[:, :n_dofs] = dof_velocities[:, self.motion_dof_indexes[:n_dofs]]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(
            num_samples, self.cfg.num_amp_observations, -1
        )

        return root_state, joint_pos, joint_vel, times

    # ------------------------------------------------------------------
    # AMP 인터페이스
    # ------------------------------------------------------------------

    def collect_reference_motions(
        self, num_samples: int, current_times: np.ndarray | None = None
    ) -> torch.Tensor:
        """레퍼런스 모션 AMP 관측값 수집."""
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)

        times = (
            np.expand_dims(current_times, axis=-1)
            - self.step_dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()

        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples * self.cfg.num_amp_observations, times=times)

        amp_obs = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_obs.view(-1, self.amp_observation_size)

    def get_amp_observations(self, num_samples: int) -> torch.Tensor:
        """RSL-RL Runner가 Discriminator 업데이트 시 호출하는 Expert 관측 샘플러."""
        return self.collect_reference_motions(num_samples)

    def _resample_commands(self, env_ids: torch.Tensor):
        if self.cfg.command_curriculum:
            command_keys_in_order = ["lin_vel_x_range", "lin_vel_y_range", "ang_vel_range"]
            for i in range(self.cfg.num_commands):
                if i < len(command_keys_in_order):
                    key = command_keys_in_order[i]
                    if key in self.cfg.command_cfg:
                        lower, upper = self.cfg.command_cfg[key]
                        if i == 0:
                            if not hasattr(self, "curriculum_lin_vel_x"):
                                self.curriculum_lin_vel_x = torch.full(
                                    (self.num_envs,), lower, device=self.device
                                )
                            curr = self.curriculum_lin_vel_x[env_ids]
                            use_curriculum = curr < upper
                            low = torch.where(use_curriculum, curr - self.cfg.curriculum_step, torch.full_like(curr, lower))
                            high = torch.where(use_curriculum, curr, torch.full_like(curr, upper))
                            self._commands[env_ids, i] = torch.lerp(
                                low, high, torch.rand(len(env_ids), device=self.device)
                            )
                        else:
                            if not hasattr(self, "curriculum_ang_vel"):
                                self.curriculum_ang_vel = torch.full(
                                    (self.num_envs,), lower, device=self.device
                                )
                            curr = self.curriculum_ang_vel[env_ids]
                            use_curriculum = curr < upper
                            direction = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1
                            signed_curr = curr * direction.float()
                            low = torch.where(
                                use_curriculum, signed_curr - self.cfg.curriculum_step, torch.full_like(curr, lower)
                            )
                            high = torch.where(use_curriculum, signed_curr, torch.full_like(curr, upper))
                            self._commands[env_ids, i] = torch.lerp(
                                low, high, torch.rand(len(env_ids), device=self.device)
                            )
        else:
            self._commands[env_ids, 0] = torch_rand_float(
                self.cfg.command_cfg["lin_vel_x_range"][0],
                self.cfg.command_cfg["lin_vel_x_range"][1],
                (len(env_ids),), self.device,
            )
            self._commands[env_ids, 1] = torch_rand_float(
                self.cfg.command_cfg["lin_vel_y_range"][0],
                self.cfg.command_cfg["lin_vel_y_range"][1],
                (len(env_ids),), self.device,
            )
            self._commands[env_ids, 2] = torch_rand_float(
                self.cfg.command_cfg["ang_vel_range"][0],
                self.cfg.command_cfg["ang_vel_range"][1],
                (len(env_ids),), self.device,
            )

    def _resample_commands_in_episode(self, env_ids: torch.Tensor):
        """에피소드 중 command 재샘플링 — curriculum delta 기반.

        Stage 1 (step < start):  command 변경 없음
        Stage 2 (start ~ end):   RSI 기준 ± delta 점진 증가
        Stage 3 (step > end):    RSI 기준 ± delta_end (전체 범위)
        """
        start_step = self.cfg.command_curriculum_start_step
        end_step = self.cfg.command_curriculum_end_step

        if self.common_step_counter < start_step:
            return  # Stage 1: curriculum 시작 전, command 변경 없음

        progress = min(
            (self.common_step_counter - start_step) / max(end_step - start_step, 1), 1.0
        )
        delta = self.cfg.command_delta_start + (
            self.cfg.command_delta_end - self.cfg.command_delta_start
        ) * progress

        vel_min = float(self.cfg.command_cfg["lin_vel_x_range"][0])
        vel_max = float(self.cfg.command_cfg["lin_vel_x_range"][1])

        ref_vel = self._rsi_ref_vel[env_ids]
        noise = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * delta
        new_vel = torch.clamp(ref_vel + noise, vel_min, vel_max)

        alpha = self.cfg.command_soft_update_alpha
        self._commands[env_ids, 0] = alpha * new_vel + (1.0 - alpha) * self._commands[env_ids, 0]

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def _get_body_contact(
        self,
        contact_forces: torch.Tensor,
        body_names: list[str],
        keyword: str,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        """특정 body 이름에 keyword가 포함된 body들의 접촉 감지."""
        indexes = [i for i, name in enumerate(body_names) if keyword in name.lower()]
        if not indexes:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if contact_forces.shape[1] >= max(indexes) + 1:
            forces = contact_forces[:, indexes, :]
            contact = torch.norm(forces, dim=-1).max(dim=-1).values > threshold
        else:
            contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return contact


# ------------------------------------------------------------------
# JIT 컴파일 함수
# ------------------------------------------------------------------


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    local_key_body_positions: torch.Tensor,
) -> torch.Tensor:
    """AMP 관측 벡터 계산.

    출력 크기: 12+12+1+3+3+12 = 43
    """
    obs = torch.cat(
        (
            dof_positions,               # 12
            dof_velocities,              # 12
            root_positions[:, 2:3],      # 1 (root 높이)
            root_linear_velocities,      # 3
            root_angular_velocities,     # 3
            local_key_body_positions.view(local_key_body_positions.shape[0], -1),   # 4×3=12
        ),
        dim=-1,
    )
    return obs
