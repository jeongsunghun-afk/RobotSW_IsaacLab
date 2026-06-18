# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 skrl AMP 환경.

humanoid_amp_env.py 의 skrl AMP 구조 + go2_amp_env.py 의 Go2 관측 구성을 결합.

NPZ 데이터 규약 (convert_smr_to_npz.py 출력):
  body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot", "base"]
  body_positions[:, 0:4]  — 발끝 위치, body-local frame
  body_positions[:, 4]    — 베이스 위치, world frame
  body_linear_velocities  — body frame (toes + base)
  body_angular_velocities — body frame (toes=0, base)
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
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.utils.math import sample_uniform as torch_rand_float

from .go2_skrl_amp_env_cfg import Go2SkrlAmpEnvCfg
from .motions.motion_loader import MotionLoader


class Go2SkrlAmpEnv(DirectRLEnv):
    """Go2 skrl AMP 모방학습 환경.

    AMP observation (43-dim):
        dof_pos(12) + dof_vel(12) + root_height(1)
        + lin_vel_b(3) + ang_vel_b(3) + key_body_pos_local(12)
    """

    cfg: Go2SkrlAmpEnvCfg

    KEY_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def __init__(self, cfg: Go2SkrlAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 모션 로더 초기화 (NPZ 파일 디렉토리 또는 단일 파일)
        # MotionLoader 자체가 디렉토리 처리를 지원하므로 단일 인스턴스로 모든 클립 로드
        motion_file = self.cfg.motion_file
        if os.path.isdir(motion_file):
            npz_files = sorted(glob.glob(os.path.join(motion_file, "*.npz")))
            if not npz_files:
                raise FileNotFoundError(f"NPZ 파일 없음: {motion_file}\nconvert_smr_to_npz.py 를 먼저 실행하세요.")
        self._motion_loader = MotionLoader(motion_file, device=self.device)

        # 로봇 body/joint 인덱스
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in self.KEY_BODY_NAMES]

        # 모션 로더 body/DOF 인덱스
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(self.KEY_BODY_NAMES)
        robot_joint_names = list(self.robot.data.joint_names)
        try:
            self.motion_dof_indexes = self._motion_loader.get_dof_index(robot_joint_names)
        except AssertionError:
            print("[Go2SkrlAmpEnv] DOF 이름 불일치 — 1:1 순서 매핑 사용")
            self.motion_dof_indexes = list(range(min(len(robot_joint_names), self._motion_loader.num_dofs)))

        # X/Y linear velocity + yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # AMP 관측 버퍼
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # DirectRLEnv 필수 메서드
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
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
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.cfg.action_scale * self.actions + self.robot.data.default_joint_pos
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        root_pos_w = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b

        # 발끝 위치를 body-local 프레임으로 변환
        key_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes]
        rel_pos = key_pos_w - root_pos_w.unsqueeze(1)
        num_keys = rel_pos.shape[1]
        root_quat_exp = root_quat_w.unsqueeze(1).expand(-1, num_keys, -1)
        local_key_body_pos = quat_apply_inverse(root_quat_exp.reshape(-1, 4), rel_pos.reshape(-1, 3)).view(
            *rel_pos.shape
        )

        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            root_pos_w,
            root_lin_vel_b,
            root_ang_vel_b,
            local_key_body_pos,
        )

        # AMP 히스토리 버퍼 업데이트
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs.clone()

        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        # Policy 관측치
        policy_obs = torch.cat(
            [
                self.robot.data.projected_gravity_b,  # 3
                self._commands,  # 3
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # 12
                self.robot.data.joint_vel,  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel = torch.exp(-lin_vel_error / self.cfg.tracking_sigma)

        yaw_rate_error = torch.square(self._commands[:, 2] - base_ang_vel[:, 2])
        tracking_ang_vel = torch.exp(-yaw_rate_error / self.cfg.tracking_sigma)

        lin_vel_rew = tracking_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
        ang_vel_rew = tracking_ang_vel * self.cfg.yaw_rate_reward_scale * self.step_dt

        self.extras["log"] = {
            "reward/tracking_lin_vel": lin_vel_rew.mean().item(),
            "reward/tracking_ang_vel": ang_vel_rew.mean().item(),
            "reward/task_total": (lin_vel_rew + ang_vel_rew).mean().item(),
        }

        return lin_vel_rew + ang_vel_rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            base_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
            died = base_height < self.cfg.termination_height
            died = died | (self.robot.data.projected_gravity_b[:, 2] > 0.0)

            grav_b = self.robot.data.projected_gravity_b
            roll = torch.atan2(grav_b[:, 1], -grav_b[:, 2])
            pitch = torch.atan2(-grav_b[:, 0], torch.sqrt(grav_b[:, 1] ** 2 + grav_b[:, 2] ** 2))
            roll_limit = self.cfg.roll_termination_deg * torch.pi / 180.0
            pitch_limit = self.cfg.pitch_termination_deg * torch.pi / 180.0
            died = died | (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)

            # physics 정착 시간 확보 (첫 1 step 제외)
            died = died & (self.episode_length_buf > 1)
        else:
            died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # env_ids는 위에서 None이 아닌 Tensor로 보장됨
        assert env_ids is not None

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._resample_commands(env_ids)

    # ------------------------------------------------------------------
    # 리셋 전략
    # ------------------------------------------------------------------

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
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

        # body_positions[:, motion_ref_body_index] = base world-frame position
        root_rot = body_rotations[:, self.motion_ref_body_index]  # (N, 4) wxyz
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_rot
        # body frame velocities → world frame
        root_state[:, 7:10] = quat_apply(root_rot, body_linear_velocities[:, self.motion_ref_body_index])
        root_state[:, 10:13] = quat_apply(root_rot, body_angular_velocities[:, self.motion_ref_body_index])

        n_dofs = len(self.motion_dof_indexes)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, :n_dofs] = dof_positions[:, self.motion_dof_indexes[:n_dofs]]
        joint_vel[:, :n_dofs] = dof_velocities[:, self.motion_dof_indexes[:n_dofs]]

        # AMP 버퍼 초기화 (RSI 시점의 reference motion으로)
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, joint_pos, joint_vel

    # ------------------------------------------------------------------
    # AMP 인터페이스 (skrl AMP agent 필수)
    # ------------------------------------------------------------------

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        """레퍼런스 모션 AMP 관측값 수집.

        NPZ의 body_positions[:, key_indexes]는 body-local frame으로 저장됨.
        → compute_obs에 그대로 전달 가능 (별도 좌표 변환 불필요).
        """
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)

        # 각 샘플에 대해 num_amp_observations개의 연속 프레임 시간 생성
        # 순환 모션이므로 음수 시간은 duration으로 wrap-around 처리
        # np.maximum(times, 0.0) 사용 시 t=0 근처 샘플에서 frame 0이 과다 표현됨
        times = (
            np.expand_dims(current_times, axis=-1)  # type: ignore[arg-type]
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        times = times % self._motion_loader.duration

        (
            dof_positions,
            dof_velocities,
            body_positions,
            _,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples * self.cfg.num_amp_observations, times=times)

        # key body positions: NPZ에 body-local frame으로 저장됨 → 그대로 사용
        amp_obs = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],  # world pos (height용)
            body_linear_velocities[:, self.motion_ref_body_index],  # body frame
            body_angular_velocities[:, self.motion_ref_body_index],  # body frame
            body_positions[:, self.motion_key_body_indexes],  # body-local toe pos
        )
        return amp_obs.view(-1, self.amp_observation_size)

    def _resample_commands(self, env_ids: torch.Tensor):
        self._commands[env_ids, 0] = torch_rand_float(
            self.cfg.command_cfg["lin_vel_x_range"][0],
            self.cfg.command_cfg["lin_vel_x_range"][1],
            (len(env_ids),),
            self.device,
        )
        self._commands[env_ids, 1] = torch_rand_float(
            self.cfg.command_cfg["lin_vel_y_range"][0],
            self.cfg.command_cfg["lin_vel_y_range"][1],
            (len(env_ids),),
            self.device,
        )
        self._commands[env_ids, 2] = torch_rand_float(
            self.cfg.command_cfg["ang_vel_range"][0],
            self.cfg.command_cfg["ang_vel_range"][1],
            (len(env_ids),),
            self.device,
        )


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

    출력 크기: 12 + 12 + 1 + 3 + 3 + 12 = 43
    """
    obs = torch.cat(
        (
            dof_positions,  # 12
            dof_velocities,  # 12
            root_positions[:, 2:3],  # 1
            root_linear_velocities,  # 3
            root_angular_velocities,  # 3
            local_key_body_positions.view(local_key_body_positions.shape[0], -1),  # 12
        ),
        dim=-1,
    )
    return obs
