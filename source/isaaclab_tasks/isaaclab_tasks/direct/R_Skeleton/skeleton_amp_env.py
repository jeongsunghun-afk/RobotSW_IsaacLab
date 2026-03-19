# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""R_Skeleton AMP (Adversarial Motion Prior) 환경.

humanoid_amp_env.py 를 기반으로 R_Skeleton(38 DOF)에 맞게 조정되었습니다.
모션 참조 데이터는 DeepMimic JSON 형식의 txt 파일에서 로드합니다.
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
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, quat_apply_inverse
from isaaclab.utils.math import sample_uniform as torch_rand_float

from .motion_loader import SkeletonMotionLoader
from .skeleton_amp_env_cfg import SkeletonAmpEnvCfg


class SkeletonAmpEnv(DirectRLEnv):
    """R_Skeleton AMP imitation 환경.

    AMP 관측 벡터 (amp_observation_space = 105):
        dof_pos(34) + dof_vel(34) + root_height(1) +
        tangent_normal(6) + lin_vel(3) + ang_vel(3) + key_body_pos(12) + key_body_lin_vel(12)
    """

    cfg: SkeletonAmpEnvCfg

    # key body 이름 (발끝 4개) — SkeletonMotionLoader.BODY_NAMES 와 일치해야 함
    KEY_BODY_NAMES = [
        "FL_link7_toe",
        "FR_link7_toe",
        "HL_link7_toe",
        "HR_link7_toe",
    ]

    def __init__(self, cfg: SkeletonAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 액션 오프셋/스케일 (관절 범위 기반)
        # dof_lower = self._robot.data.soft_joint_pos_limits[0, :, 0]
        # dof_upper = self._robot.data.soft_joint_pos_limits[0, :, 1]
        # self.action_offset = 0.5 * (dof_upper + dof_lower)
        # self.action_scale = dof_upper - dof_lower

        self.cfg = cfg

        # 모션 로더 초기화
        motion_file = self.cfg.motion_file
        if os.path.isdir(motion_file):
            motion_files = sorted(glob.glob(os.path.join(motion_file, "*.txt")))
        else:
            motion_files = [motion_file]
        self._motion_loader = SkeletonMotionLoader(motion_files=motion_files, device=self.device)

        # 로봇에서 body/joint 인덱스 구하기
        self.ref_body_index = self._robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [
            self._robot.data.body_names.index(name) for name in self.KEY_BODY_NAMES
        ]
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # 모션 로더에서 body/DOF 인덱스 구하기
        # (모션의 joint 순서가 로봇과 다를 수 있으므로 매핑)
        robot_joint_names = list(self._robot.data.joint_names)
        try:
            self.motion_dof_indexes = self._motion_loader.get_dof_index(robot_joint_names)
        except AssertionError:
            # 로봇 joint 이름이 모션 로더 DOF 이름과 맞지 않으면 순서대로 1:1 매핑
            print("[SkeletonAmpEnv] DOF 이름 불일치 — 순서대로 1:1 매핑 사용")
            self.motion_dof_indexes = list(range(min(len(robot_joint_names), self._motion_loader.num_dofs)))

        # motion_loader.BODY_NAMES = ["FL_link7_toe", "FR_link7_toe", "HL_link7_toe", "HR_link7_toe", "base"]
        # stmr.py가 [FL, HL, FR, HR] 순서로 저장하지만 motion_loader 로딩 시 [0,2,1,3] 재정렬로
        # [FL(0), FR(1), HL(2), HR(3), base(4)] 순서로 맞춰져 있음.
        # KEY_BODY_NAMES = [FL, FR, HL, HR] 와 완전히 대응됨.
        self.motion_ref_body_index = 4          # base = index 4
        self.motion_key_body_indexes = [0, 1, 2, 3]  # FL, FR, HL, HR

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
        
        # History Buffer 추가
        if self.cfg.history_observation:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.history_len, self.cfg.observation_space, device=self.device, dtype=torch.float
            )

    # ------------------------------------------------------------------
    # Isaac Lab DirectRLEnv 필수 메서드
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # 지면
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

        # 씬 복제
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 씬에 등록
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        
        # 에피소드 로깅용 딕셔너리
        self._episode_sums = {
            "tracking_lin_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "tracking_ang_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _post_physics_step(self):
        """물리 스텝 이후 처리 (필요시 추가 구현)."""
        pass

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # AMP 관측 벡터 계산
        # compute_obs는 이제 local 좌표를 받으므로, 사전에 변환하여 전달합니다.
        
        root_pos_w = self._robot.data.body_pos_w[:, self.ref_body_index]
        root_quat_w = self._robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_vel_w = self._robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_ang_vel_w = self._robot.data.body_ang_vel_w[:, self.ref_body_index]
        
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        
        # 월드 기준 상대 거리/속도
        rel_pos = self._robot.data.body_pos_w[:, self.key_body_indexes] - root_pos_w.unsqueeze(1)
        rel_vel = self._robot.data.body_lin_vel_w[:, self.key_body_indexes] - root_lin_vel_w.unsqueeze(1)

        # 루트 쿼터니언 형상 맞추기 (N, num_keys, 4)
        num_keys = rel_pos.shape[1]
        root_quat_expanded = root_quat_w.unsqueeze(1).expand(-1, num_keys, -1)
        
        # 로컬 프레임으로 완전하게 회전 변환 (quat_rotate_inverse)
        local_key_body_pos = quat_apply_inverse(
            root_quat_expanded.reshape(-1, 4), 
            rel_pos.reshape(-1, 3)
        ).view(*rel_pos.shape)
        
        local_key_body_vel = quat_apply_inverse(
            root_quat_expanded.reshape(-1, 4), 
            rel_vel.reshape(-1, 3)
        ).view(*rel_vel.shape)

        obs = compute_obs(
            self._robot.data.joint_pos,
            self._robot.data.joint_vel,
            root_pos_w,
            root_lin_vel_b,
            root_ang_vel_b,
            local_key_body_pos,
            local_key_body_vel,
        )
        # print('-----Simulation-----')
        # print(root_pos_w[0])
        # print(local_key_body_pos[0])

        # AMP 히스토리 버퍼 업데이트 (최신 obs = index 0)
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs.clone()

        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)
        }
        
        # 정책 네트워크용 관측치 (AMP obs + commands -> 삭제: 이제 projected gravity, commands, pos, vel, actions 위주)
        # Actor/Critic RMA 구조 관측
        policy_obs = torch.cat(
            [
                self._robot.data.projected_gravity_b, # 3
                self._commands, # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos, # 34
                self._robot.data.joint_vel, # 34
                self.actions, # 34
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
                    self._robot.data.root_lin_vel_b, # 3
                    self._robot.data.root_ang_vel_b, # 3
                    torch.tensor(self._robot.root_physx_view.get_masses(), device=self.device).reshape(self.num_envs, -1),
                    torch.tensor(self._robot.root_physx_view.get_material_properties(), device=self.device).reshape(self.num_envs, -1),
                ],
                dim=-1,
            )
            observations["priv"] = priv_obs

        # 디스크리미네이터에서 사용하기 위해 extras['amp_obs']에 AMP 관측을 유지합니다. 
        # rsl_rl Runner가 step 시 extras['amp_obs'] 부분을 떼어냅니다.
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Command Tracking Reward 계산 (Task Reward)
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b

        # Linear Velocity (x, y) Tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel = torch.exp(-lin_vel_error / self.cfg.tracking_sigma)
        
        # Yaw Rate (z) Tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - base_ang_vel[:, 2])
        tracking_ang_vel = torch.exp(-yaw_rate_error / self.cfg.tracking_sigma)

        # Tracking Rewards
        lin_vel_reward = tracking_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
        ang_vel_reward = tracking_ang_vel * self.cfg.yaw_rate_reward_scale * self.step_dt
        # lin_vel_reward = tracking_lin_vel * self.cfg.lin_vel_reward_scale
        # ang_vel_reward = tracking_ang_vel * self.cfg.yaw_rate_reward_scale

        # Total Task Reward
        task_reward = lin_vel_reward + ang_vel_reward

        # 누적 보상 기록 (로깅용)
        self._episode_sums["tracking_lin_vel"] += lin_vel_reward
        self._episode_sums["tracking_ang_vel"] += ang_vel_reward

        return task_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            # base 높이가 너무 낮으면 넘어진 것
            base_height = self._robot.data.body_pos_w[:, self.ref_body_index, 2]
            died = base_height < self.cfg.termination_height

            # base 방향 검사 (거꾸로 뒤집힌 경우 방지)
            # 정상적인 직립 상태에서는 지면 방향(Z축)으로 중력(벡터 값 음수)이 작용함.
            # Z축을 기준으로 기울어져 양수가 되면 뒤집힌 상태임. (허용 오차 고려 0.0)
            flipped = self._robot.data.projected_gravity_b[:, 2] > 0.0
            died = died | flipped

            # 불법적인 신체 부위 접촉 감지
            if hasattr(self, "contact_sensor"):
                contact_forces = self.contact_sensor.data.net_forces_w
                if contact_forces is not None and contact_forces.numel() > 0:
                    body_names = self._robot.data.body_names
                    
                    # 지면과 닿으면 안 되는 부위들
                    # bad_contact_keywords = ["base", "neck", "arm", "shoulder", "thigh", "waist", "knee"]
                    bad_contact_keywords = ["base"]
                    bad_contacts = torch.zeros_like(died)
                    for keyword in bad_contact_keywords:
                        bad_contacts |= self._get_body_contact(contact_forces, body_names, keyword)
                        
                    died = died | bad_contacts
        else:
            died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 명령 리샘플링
        self._resample_commands(env_ids)

        # History buffer 초기화
        if self.cfg.history_observation:
            self.obs_history_buf[env_ids, :, :] = 0.0

        # 로깅(Extras) 처리
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            # 에피소드가 끝난 환경은 버퍼 초기화
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

        # print(root_state)
        # print(joint_pos)
        # print(joint_vel)
        # print(self.action_scale)
        # AMP 히스토리 버퍼 초기화
        self.amp_observation_buffer[env_ids] = 0.0
        
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # root 상태 설정 (base body 사용)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = (
            body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        )
        root_state[:, 2] += 0.05  # 지면 충돌 방지를 위해 약간 들어올림
        root_rot = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 3:7] = root_rot
        
        # motion_loader의 속도 데이터는 Base 로컬 좌표계 기준이므로, 월드 좌표계 속도로 변환하여 할당합니다.
        root_state[:, 7:10] = quat_apply(root_rot, body_linear_velocities[:, self.motion_ref_body_index])
        root_state[:, 10:13] = quat_apply(root_rot, body_angular_velocities[:, self.motion_ref_body_index])
        
        # print(root_state)
        
        # DOF 상태
        n_dofs = len(self.motion_dof_indexes)
        robot_ndof = self._robot.data.default_joint_pos.shape[1]
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, : n_dofs] = dof_positions[:, self.motion_dof_indexes[: n_dofs]]
        joint_vel[:, : n_dofs] = dof_velocities[:, self.motion_dof_indexes[: n_dofs]]

        # AMP 히스토리 초기화
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(
            num_samples, self.cfg.num_amp_observations, -1
        )

        return root_state, joint_pos, joint_vel

    # ------------------------------------------------------------------
    # skrl AMP 인터페이스
    # ------------------------------------------------------------------

    def collect_reference_motions(
        self, num_samples: int, current_times: np.ndarray | None = None
    ) -> torch.Tensor:
        """레퍼런스 모션 AMP 관측값 수집 (skrl이 호출)."""
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
        # print('-----data-----')
        # print(body_positions[0])
        amp_obs = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
            body_linear_velocities[:, self.motion_key_body_indexes],
        )
        return amp_obs.view(-1, self.amp_observation_size)

    def get_amp_observations(self, num_samples: int) -> torch.Tensor:
        """RSL-RL Runner가 Discriminator 업데이트 시 호출하는 Expert 관측 샘플러.

        on_policy_runner_amp.py가 hasattr(env, 'get_amp_observations') 로 존재 여부를 확인합니다.
        내부적으로 collect_reference_motions를 호출합니다.
        """
        return self.collect_reference_motions(num_samples)

    def _resample_commands(self, env_ids: torch.Tensor):
        if self.cfg.command_curriculum:
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
                            if not hasattr(self, 'curriculum_lin_vel_x'):
                                self.curriculum_lin_vel_x = torch.full((self.num_envs,), lower, device=self.device)
                            curr = self.curriculum_lin_vel_x[env_ids]
                            use_curriculum = curr < upper
                            low = torch.where(use_curriculum, curr - self.cfg.curriculum_step, torch.full_like(curr, lower))
                            high = torch.where(use_curriculum, curr, torch.full_like(curr, upper))
                            self._commands[env_ids, i] = torch.lerp(
                                low, high, torch.rand(len(env_ids), device=self.device)
                            )
                        elif i == 1 or i == 2:  # ang_vel에 curriculum 적용
                            if not hasattr(self, 'curriculum_ang_vel'):
                                self.curriculum_ang_vel = torch.full((self.num_envs,), lower, device=self.device)
                            curr = self.curriculum_ang_vel[env_ids]
                            use_curriculum = curr < upper
                            # random sign 선택
                            direction = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1  # {-1, +1}
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
                self.cfg.command_cfg["lin_vel_x_range"][0], self.cfg.command_cfg["lin_vel_x_range"][1], (len(env_ids),), self.device
            )
            self._commands[env_ids, 1] = torch_rand_float(
                self.cfg.command_cfg["lin_vel_y_range"][0], self.cfg.command_cfg["lin_vel_y_range"][1], (len(env_ids),), self.device
            )
            self._commands[env_ids, 2] = torch_rand_float(
                self.cfg.command_cfg["ang_vel_range"][0], self.cfg.command_cfg["ang_vel_range"][1], (len(env_ids),), self.device
            )

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
        """특정 body 이름에 keyword가 포함된 body들의 접촉 감지 (env별 OR)."""
        indexes = [i for i, name in enumerate(body_names) if keyword in name.lower()]
        if not indexes:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # contact_forces shape: (num_envs, num_bodies, 3) 또는 (num_envs, num_sensors, 3)
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
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    """쿼터니언에서 tangent 및 normal 벡터를 계산합니다."""
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    local_key_body_positions: torch.Tensor,
    local_key_body_linear_velocities: torch.Tensor,
) -> torch.Tensor:
    """AMP 관측 벡터 계산.

    출력 크기: 34+34+1+3+3+12+12 = 99
    (key_body_positions 가 4개라면 12차원)
    """
    obs = torch.cat(
        (
            dof_positions,               # 34
            dof_velocities,              # 34
            root_positions[:, 2:3],      # 1 (root 높이)
            root_linear_velocities,      # 3
            root_angular_velocities,     # 3
            # key body 상대 위치 (root 기준)
            local_key_body_positions.view(
                local_key_body_positions.shape[0], -1
            ),  # 4×3 = 12
            # key body 상대 선속도 (root 기준)
            local_key_body_linear_velocities.view(
                local_key_body_linear_velocities.shape[0], -1
            ),  # 4×3 = 12
        ),
        dim=-1,
    )
    return obs