# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Interaction 환경 - Genesis InteractionEnv를 IsaacLab DirectRLEnv 기반으로 포팅.

모션 모방(Motion Imitation) 방식으로 Go2 로봇의 상호작용 행동(앉기, 눕기, 일어서기 등)을 학습합니다.
"""

from __future__ import annotations

import os

import gymnasium as gym
import pandas as pd
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
)

##
# Pre-defined configs
##


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------


def torch_rand_float(lower: float, upper: float, shape: tuple, device: str) -> torch.Tensor:
    """균일 분포에서 랜덤 float 텐서를 샘플링합니다."""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


from .go2_interaction_cfg import Go2InteractionCfg

# ---------------------------------------------------------------------------
# 환경 클래스
# ---------------------------------------------------------------------------


class Go2InteractionEnv(DirectRLEnv):
    """Go2 상호작용 학습 환경.

    Genesis `InteractionEnv`를 IsaacLab `DirectRLEnv` 기반으로 포팅한 환경입니다.
    CSV 모션 파일을 참조 데이터로 활용하여 로봇의 hip/foot 위치와 base 자세를 추적합니다.
    """

    cfg: Go2InteractionCfg

    def __init__(
        self,
        cfg: Go2InteractionCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ------------------------------------------------------------------ #
        # 액션 / 이전 액션 버퍼
        # ------------------------------------------------------------------ #
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros_like(self._actions)

        # ------------------------------------------------------------------ #
        # 커맨드 버퍼 (lin_vel_x, lin_vel_y, ang_vel)
        # ------------------------------------------------------------------ #
        # self._commands = torch.zeros(self.num_envs, 3, device=self.device)  # 삭제

        # ------------------------------------------------------------------ #
        # Interaction 커맨드: 모션 ID (0~3)
        # ------------------------------------------------------------------ #
        self._interaction_command = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)

        # ------------------------------------------------------------------ #
        # 모션 프레임 카운터
        # ------------------------------------------------------------------ #
        self._current_frames = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)

        # ------------------------------------------------------------------ #
        # 관측 히스토리 버퍼 [num_envs, history_len, num_prio_obs]
        # ------------------------------------------------------------------ #
        self._obs_history = torch.zeros(
            self.num_envs,
            self.cfg.history_len,
            self.cfg.num_prio_obs,
            device=self.device,
        )

        # ------------------------------------------------------------------ #
        # 토크 버퍼 (이전 토크 추적)
        # ------------------------------------------------------------------ #
        self._prev_torques = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # ------------------------------------------------------------------ #
        # 참조 위치 (target_world_positions)
        # ------------------------------------------------------------------ #
        self._target_world_positions: dict = {}

        # ------------------------------------------------------------------ #
        # 에피소드 보상 로그
        # ------------------------------------------------------------------ #
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "hip_positions",
                "foot_positions",
                "base_height",
                "base_pitch",
                "feet_contact",
                "stand_penalty",
                "dof_acc",
                "action_rate",
                "delta_torques",
                "torques",
                "similar_to_default",
                "lin_vel_z",
                "ang_vel_xy",
                "torques_balance",
                "stand_front_vel",
                "front_stillness",
            ]
        }

        # ------------------------------------------------------------------ #
        # Contact Sensor 바디 ID 조회
        # ------------------------------------------------------------------ #
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalized_body_names)

        # ------------------------------------------------------------------ #
        # 로봇 바디 ID 조회 (world position용)
        # ------------------------------------------------------------------ #
        # regex 매칭 시 순서 보장을 위해 명시적으로 검색합니다 (FL, FR, RL, RR)
        self._hip_body_ids = []
        self._foot_body_ids = []
        for prefix in ["FL", "FR", "RL", "RR"]:
            hip_id, _ = self._robot.find_bodies(f"{prefix}_hip")
            foot_id, _ = self._robot.find_bodies(f"{prefix}_foot")
            self._hip_body_ids.append(hip_id[0])
            self._foot_body_ids.append(foot_id[0])

        # ------------------------------------------------------------------ #
        # 좌/우 관절 인덱스
        # ------------------------------------------------------------------ #
        all_joint_names = self._robot.data.joint_names
        left_names = ["FL_hip", "FL_thigh", "FL_calf", "RL_hip", "RL_thigh", "RL_calf"]
        right_names = ["FR_hip", "FR_thigh", "FR_calf", "RR_hip", "RR_thigh", "RR_calf"]
        self._left_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if any(ln in n for ln in left_names)],
            dtype=torch.long,
            device=self.device,
        )
        self._right_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if any(rn in n for rn in right_names)],
            dtype=torch.long,
            device=self.device,
        )

        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if "hip" in n],
            dtype=torch.long,
            device=self.device,
        )

        # 앞다리(FL, FR) 관절 인덱스 (stand 진동 억제용)
        front_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
        self._front_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if any(fn in n for fn in front_names)],
            dtype=torch.long,
            device=self.device,
        )

        # ------------------------------------------------------------------ #
        # 모션 데이터 로딩
        # ------------------------------------------------------------------ #
        self.all_preprocessed_data: dict = {}
        self._motion_lengths_tensor = torch.zeros(len(self.cfg.motion_files), dtype=torch.long, device=self.device)
        self._preload_all_motions()

    # ---------------------------------------------------------------------- #
    # 씬 구성
    # ---------------------------------------------------------------------- #

    def _setup_scene(self):
        """씬에 로봇, 컨택 센서, 지형을 추가합니다."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 클론 & 복제
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------------- #
    # 액션 처리
    # ---------------------------------------------------------------------- #

    def _pre_physics_step(self, actions: torch.Tensor):
        """액션을 클리핑하고 목표 관절 위치를 계산합니다."""
        self._actions = actions.clone().clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        actions = self._actions.clone()
        if self.cfg.hip_scale_reduction:
            actions[:, self._hip_joint_ids] *= 0.5
        self._processed_actions = self.cfg.action_scale * actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        """목표 관절 위치를 로봇에 적용합니다."""
        self._robot.set_joint_position_target(self._processed_actions)

    def _post_physics_step(self):
        """물리 스텝 이후 상태를 업데이트합니다."""
        self._step_motion_frames()

    # ---------------------------------------------------------------------- #
    # 관측 계산
    # ---------------------------------------------------------------------- #

    def _get_observations(self) -> dict:
        """관측을 계산하고 policy/priv/history 그룹으로 분리하여 반환합니다."""

        # ------------------------------------------------------------------ #
        # Base 오일러각 계산 (roll, pitch, yaw)
        # ------------------------------------------------------------------ #
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        base_euler = torch.stack([roll, pitch, yaw], dim=-1)  # [N, 3]

        # ------------------------------------------------------------------ #
        # Base 선속도/각속도 (body frame)
        # ------------------------------------------------------------------ #
        base_lin_vel_b = self._robot.data.root_lin_vel_b
        base_ang_vel_b = self._robot.data.root_ang_vel_b

        # ------------------------------------------------------------------ #
        # Prio 관측 (정책 입력)
        # projected_gravity(3) + interaction_cmd(1)
        # + dof_pos(12) + dof_vel(12) + actions(12) + roll_pitch(2) = 42
        # ------------------------------------------------------------------ #
        prio_obs = torch.cat(
            [
                self._robot.data.projected_gravity_b,  # 3
                self._interaction_command.float(),  # 1
                (self._robot.data.joint_pos - self._robot.data.default_joint_pos)
                * self.cfg.obs_scales["dof_pos"],  # 12
                self._robot.data.joint_vel * self.cfg.obs_scales["dof_vel"],  # 12
                self._actions,  # 12
                base_euler[:, :2],  # 2 (roll, pitch)
            ],
            dim=-1,
        )  # [N, 42]

        # ------------------------------------------------------------------ #
        # Priv 관측 (특권 정보: 오일러각 3)
        # ------------------------------------------------------------------ #
        priv_obs = base_euler  # [N, 3]

        # ------------------------------------------------------------------ #
        # Priv Latent 관측
        # height(1)+lin_vel(3)+ang_vel(3)+dof_pos[:6](6)+dof_vel[:6](6)+torques[:3](3) = 22
        # ------------------------------------------------------------------ #
        current_torques = self._robot.data.applied_torque  # [N, 12]
        priv_latent = torch.cat(
            [
                self._robot.data.root_link_pos_w[:, 2:3],  # base height (1)
                base_lin_vel_b,  # lin_vel (3)
                base_ang_vel_b,  # ang_vel (3)
                self._robot.data.joint_pos[:, :6],  # dof_pos[:6] (6)
                self._robot.data.joint_vel[:, :6],  # dof_vel[:6] (6)
                current_torques[:, :3],  # torques[:3] (3)
            ],
            dim=-1,
        )  # [N, 22]

        # ------------------------------------------------------------------ #
        # 히스토리 버퍼 업데이트 (roll 방식)
        # ------------------------------------------------------------------ #
        self._obs_history = torch.roll(self._obs_history, shifts=1, dims=1)
        self._obs_history[:, 0] = prio_obs
        history_obs = self._obs_history  # [N, history_len, num_prio_obs]

        # ------------------------------------------------------------------ #
        # 이전 액션 업데이트
        # ------------------------------------------------------------------ #
        self._previous_actions = self._actions.clone()
        self._prev_torques = current_torques.clone()

        return {
            "policy": prio_obs,  # [N, 42]
            "priv": priv_latent,  # [N, 22]
            "history": history_obs,  # [N, 10, 42]
        }

    # ---------------------------------------------------------------------- #
    # 보상 계산
    # ---------------------------------------------------------------------- #

    def _get_rewards(self) -> torch.Tensor:
        """보상을 계산합니다."""

        # ------------------------------------------------------------------ #
        # 현재 상태 변수
        # ------------------------------------------------------------------ #
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        base_euler = torch.stack([roll, pitch, yaw], dim=-1)

        # Hip / Foot world positions
        hip_positions = self._robot.data.body_pos_w[:, self._hip_body_ids, :]  # [N, 4, 3]
        foot_positions = self._robot.data.body_pos_w[:, self._foot_body_ids, :]  # [N, 4, 3]

        # 현재 토크
        current_torques = self._robot.data.applied_torque  # [N, 12]

        # ------------------------------------------------------------------ #
        # 참조 위치 업데이트
        # ------------------------------------------------------------------ #
        self._calculate_reference_positions()

        cmd = self._interaction_command.squeeze(-1)  # [N]
        mask_default = cmd == 0

        # ------------------------------------------------------------------ #
        # 1. Hip/Foot 위치 추적 보상 (기준 좌표계 선택)
        # ------------------------------------------------------------------ #
        if self.cfg.reward_frame == "base":
            # [Base Frame] 현재 로봇 부위를 body frame으로 변환하고 body-frame 타겟과 비교
            root_pos = self._robot.data.root_link_pos_w.unsqueeze(1)  # [N, 1, 3]
            root_quat = self._robot.data.root_link_quat_w  # [N, 4]
            root_quat_rep = root_quat.unsqueeze(1).repeat(1, 4, 1)  # [N, 4, 4]

            # 현재 hip/foot를 로봇 body frame으로 변환
            cur_hip_local = quat_apply_inverse(root_quat_rep, hip_positions - root_pos)  # [N, 4, 3]
            cur_foot_local = quat_apply_inverse(root_quat_rep, foot_positions - root_pos)  # [N, 4, 3]

            # 타겟은 이미 body-frame 상대 오프셋이므로 그대로 사용
            target_hip_local = self._target_reference_positions["hip_local"]  # [N, 4, 3]
            target_foot_local = self._target_reference_positions["foot_local"]  # [N, 4, 3]

            hip_pos_error = torch.norm(cur_hip_local - target_hip_local, dim=2)
            foot_pos_error = torch.norm(cur_foot_local - target_foot_local, dim=2)
        else:
            # [World Frame] 기존 방식
            target_hip = self._target_reference_positions["hip"]
            hip_pos_error = torch.norm(hip_positions - target_hip, dim=2)

            target_foot = self._target_reference_positions["foot"]
            foot_pos_error = torch.norm(foot_positions - target_foot, dim=2)

        # 공통 보상 계산
        rew_hip = torch.mean(torch.exp(-hip_pos_error / self.cfg.reward_sigma), dim=1)
        rew_hip[mask_default] = 0.0

        rew_foot = torch.mean(torch.exp(-foot_pos_error / self.cfg.reward_sigma), dim=1)
        rew_foot[mask_default] = 0.0

        # ------------------------------------------------------------------ #
        # 3. Base height 추적 보상
        # ------------------------------------------------------------------ #
        if "base_height" in self._target_reference_positions:
            current_h = self._robot.data.root_link_pos_w[:, 2]
            # interaction_command == 2 (눕기)인 환경은 목표 높이를 0.05로 고정
            mask_lay = cmd == 2
            target_h = self._target_reference_positions["base_height"].clone()
            target_h[mask_lay] = 0.05
            height_diff = torch.abs(target_h - current_h)
            rew_height = torch.exp(-height_diff / self.cfg.reward_sigma)
            rew_height[mask_default] = 0.0
        else:
            rew_height = torch.zeros(self.num_envs, device=self.device)

        # ------------------------------------------------------------------ #
        # 4. Base pitch 추적 보상
        # ------------------------------------------------------------------ #
        if "base_rotation" in self._target_reference_positions:
            current_pitch = base_euler[:, 1]
            target_pitch = self._target_reference_positions["base_rotation"][:, 1]
            pitch_diff = torch.abs(current_pitch - target_pitch)
            rew_pitch = torch.exp(-pitch_diff / self.cfg.reward_sigma)
            rew_pitch[mask_default] = 0.0
        else:
            rew_pitch = torch.zeros(self.num_envs, device=self.device)

        # ------------------------------------------------------------------ #
        # 5. Feet contact 보상 (stand_up 시 앞발 들기)
        # ------------------------------------------------------------------ #
        net_forces_w = self._contact_sensor.data.net_forces_w  # [N, num_bodies, 3]
        foot_contact_z = net_forces_w[:, self._feet_ids, 2]  # [N, 4]
        in_contact = foot_contact_z > 1.0  # [N, 4]

        # stand_up(3)인 경우 앞발(FL=0, FR=1)은 떼야 함, 나머지는 붙어야 함
        fl_target = torch.where(
            cmd == 3, torch.zeros_like(cmd, dtype=torch.bool), torch.ones_like(cmd, dtype=torch.bool)
        )
        fr_target = fl_target.clone()
        contact_mask = torch.stack(
            [
                fl_target,
                fr_target,
                torch.ones_like(fl_target, dtype=torch.bool),  # RL 항상 접촉
                torch.ones_like(fl_target, dtype=torch.bool),  # RR 항상 접촉
            ],
            dim=1,
        )  # [N, 4]

        # Smooth contact reward: 0.25 per correct foot
        correct_contacts = torch.where(contact_mask, in_contact, ~in_contact)
        rew_contact = torch.sum(correct_contacts.float(), dim=1) * 0.25

        # ------------------------------------------------------------------ #
        # 6. Stand-up 특정 강한 페널티 (앞발 접촉 시)
        # ------------------------------------------------------------------ #
        rew_stand_penalty = torch.zeros(self.num_envs, device=self.device)
        mask_stand = cmd == 3
        # 앞발(0, 1) 중 하나라도 닿아있으면 강력한 음수 보상 부여
        front_contact = in_contact[:, 0] | in_contact[:, 1]
        rew_stand_penalty[mask_stand & front_contact] = -2.0  # 강한 페널티

        # Stand-up 시 목표 자세(높이, 기울기)에 대한 추가 가중치 동적 부여
        rew_height[mask_stand] *= 2.0
        rew_pitch[mask_stand] *= 2.0

        # ------------------------------------------------------------------ #
        # 정규화 보상들
        # ------------------------------------------------------------------ #
        rew_dof_acc = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        rew_action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        rew_delta_torques = torch.sum(torch.square(current_torques - self._prev_torques), dim=1)
        rew_torques = torch.sum(torch.square(current_torques), dim=1)

        # interaction_command == 0(기본 자세)일 때만 default 유사도 보상
        rew_similar_to_default = torch.zeros(self.num_envs, device=self.device)
        diff_abs = torch.sum(
            torch.abs(self._robot.data.joint_pos[mask_default] - self._robot.data.default_joint_pos[mask_default]),
            dim=1,
        )
        rew_similar_to_default[mask_default] = torch.exp(-diff_abs / self.cfg.reward_sigma)

        rew_lin_vel_z = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        rew_ang_vel_xy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        # 좌/우 토크 밸런스
        if len(self._left_joint_ids) > 0 and len(self._right_joint_ids) > 0:
            left_t = current_torques[:, self._left_joint_ids]
            right_t = current_torques[:, self._right_joint_ids]
            rew_torques_balance = torch.norm(left_t - right_t, dim=1)
        else:
            rew_torques_balance = torch.zeros(self.num_envs, device=self.device)

        # ------------------------------------------------------------------ #
        # [A] Stand 앞다리 관절 속도 페널티
        #     stand(cmd=3)일 때 FL/FR 관절 속도²를 추가로 페널티하여 진동 억제
        # ------------------------------------------------------------------ #
        rew_stand_front_vel = torch.zeros(self.num_envs, device=self.device)
        if mask_stand.any():
            front_vel_sq = torch.sum(torch.square(self._robot.data.joint_vel[:, self._front_joint_ids]), dim=1)  # [N]
            rew_stand_front_vel[mask_stand] = front_vel_sq[mask_stand]

        # ------------------------------------------------------------------ #
        # [B] Stand 앞발 근방 정지 보상
        #     FL/FR 발이 타겟 8cm 이내에 들어오면 속도가 낮을수록 보상
        #     → 타겟 도달 후 떨지 않고 자세 유지 유도
        # ------------------------------------------------------------------ #
        rew_front_stillness = torch.zeros(self.num_envs, device=self.device)
        if mask_stand.any():
            # foot_pos_error[:,0]=FL, foot_pos_error[:,1]=FR
            front_near = (foot_pos_error[:, 0] < 0.08) & (foot_pos_error[:, 1] < 0.08)
            active = mask_stand & front_near
            if active.any():
                front_vel_mag = torch.norm(self._robot.data.joint_vel[:, self._front_joint_ids], dim=1)  # [N]
                rew_front_stillness[active] = torch.exp(-front_vel_mag[active] / 0.5)

        # ------------------------------------------------------------------ #
        # 보상 합산 (스케일 × step_dt 처리)
        # ------------------------------------------------------------------ #
        rewards = {
            "hip_positions": rew_hip * self.cfg.hip_positions_reward_scale * self.step_dt,
            "foot_positions": rew_foot * self.cfg.foot_positions_reward_scale * self.step_dt,
            "base_height": rew_height * self.cfg.base_height_reward_scale * self.step_dt,
            "base_pitch": rew_pitch * self.cfg.base_pitch_reward_scale * self.step_dt,
            "feet_contact": rew_contact * self.cfg.feet_contact_reward_scale * self.step_dt,
            "stand_penalty": rew_stand_penalty * self.cfg.stand_penalty_reward_scale * self.step_dt,
            "dof_acc": rew_dof_acc * self.cfg.dof_acc_reward_scale * self.step_dt,
            "action_rate": rew_action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "delta_torques": rew_delta_torques * self.cfg.delta_torques_reward_scale * self.step_dt,
            "torques": rew_torques * self.cfg.torques_reward_scale * self.step_dt,
            "similar_to_default": rew_similar_to_default * self.cfg.similar_to_default_reward_scale * self.step_dt,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale * self.step_dt,
            "torques_balance": rew_torques_balance * self.cfg.torques_balance_reward_scale * self.step_dt,
            "stand_front_vel": rew_stand_front_vel * self.cfg.stand_front_vel_reward_scale * self.step_dt,
            "front_stillness": rew_front_stillness * self.cfg.front_stillness_reward_scale * self.step_dt,
        }
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, val in rewards.items():
            self._episode_sums[key] += val

        return total_reward

    # ---------------------------------------------------------------------- #
    # 종료 조건
    # ---------------------------------------------------------------------- #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """에피소드 종료 조건을 계산합니다."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Base 접촉 종료
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )

        # Roll 각도 초과 종료
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        base_euler = torch.stack([roll, pitch, yaw], dim=-1)
        roll_exceeded = torch.abs(base_euler[:, 0]) > self.cfg.termination_if_roll_greater_than

        died = base_contact | roll_exceeded
        return died, time_out

    # ---------------------------------------------------------------------- #
    # 리셋
    # ---------------------------------------------------------------------- #

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """지정된 환경들을 초기 상태로 리셋합니다."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 에피소드 길이 분산 (초기 스파이크 방지)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # 액션 리셋
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._prev_torques[env_ids] = 0.0

        # 로봇 상태 리셋
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 모션 프레임 리셋
        self._current_frames[env_ids] = 0

        # 히스토리 버퍼 리셋
        self._obs_history[env_ids] = 0.0

        # 참조 위치 초기화
        self._target_world_positions = {}

        # 에피소드 보상 로그
        extras = {}
        for key in self._episode_sums:
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()

        # 커맨드 재샘플
        self._resample_commands(env_ids)

    # ---------------------------------------------------------------------- #
    # 커맨드 재샘플
    # ---------------------------------------------------------------------- #

    def _resample_commands(self, env_ids: torch.Tensor):
        """interaction 모션 ID를 새로 샘플링합니다."""
        n = len(env_ids)
        # 속도 커맨드 샘플링 삭제

        # 모션 ID 랜덤 샘플링
        num_motions = len(self.all_preprocessed_data)
        if num_motions > 0:
            self._interaction_command[env_ids] = torch.randint(
                0, num_motions, (n, 1), device=self.device, dtype=torch.long
            )
        self._current_frames[env_ids] = 0

    # ---------------------------------------------------------------------- #
    # 모션 데이터 관련 메서드
    # ---------------------------------------------------------------------- #

    def _preload_all_motions(self):
        """CSV 모션 파일을 미리 로드하고 전처리합니다."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ref_motion_path = os.path.join(current_dir, "interaction", "ref_motion")

        for motion_id, filename in self.cfg.motion_files.items():
            csv_path = os.path.join(ref_motion_path, filename)
            if not os.path.exists(csv_path):
                print(f"[Go2InteractionEnv] 경고: 모션 파일을 찾을 수 없습니다: {csv_path}")
                continue
            motion_data = pd.read_csv(csv_path)
            self.all_preprocessed_data[motion_id] = self._preprocess_motion_data(motion_data)

        print("=" * 50)
        print(f"[Go2InteractionEnv] {len(self.all_preprocessed_data)}개 모션 로드 완료")
        print("=" * 50)

        motion_lengths = [
            len(self.all_preprocessed_data[str(i)])
            for i in range(len(self.cfg.motion_files))
            if str(i) in self.all_preprocessed_data
        ]
        self._motion_lengths_tensor = torch.tensor(motion_lengths, dtype=torch.long, device=self.device)
        print(f"[Go2InteractionEnv] 모션 길이: {motion_lengths}")

    def _preprocess_motion_data(self, motion_data: pd.DataFrame) -> dict:
        """단일 모션 DataFrame을 프레임별 hip/foot 상대 좌표로 전처리합니다."""
        preprocessed: dict = {}
        max_frames = int(motion_data["frame"].max())

        # 일어서기 모션 등의 기준점(뒷발 X좌표)을 기본 서 있는 자세 기준으로 정렬
        frame0 = motion_data[motion_data["frame"] == 0]
        hind_foot_x = []
        for foot_idx in [7, 11]:
            row = frame0[frame0["joint_idx"] == foot_idx]
            if not row.empty:
                hind_foot_x.append(float(row["base_rel_x"].values[0]) * self.cfg.size_prop + self.cfg.x_offset)

        target_default_hind_x = self.cfg.target_default_hind_x
        if len(hind_foot_x) > 0:
            current_hind_x = sum(hind_foot_x) / len(hind_foot_x)
            x_shift = target_default_hind_x - current_hind_x
        else:
            x_shift = 0.0

        for frame in range(max_frames + 1):
            frame_data = motion_data[motion_data["frame"] == frame]
            preprocessed[frame] = {}

            # Base rotation & height
            base_row = frame_data[frame_data["joint_idx"] == 0]
            if not base_row.empty:
                preprocessed[frame]["roll"] = float(base_row["roll"].values[0])
                preprocessed[frame]["pitch"] = float(base_row["pitch"].values[0])
                preprocessed[frame]["yaw"] = float(base_row["yaw"].values[0])
                preprocessed[frame]["base_height"] = float(base_row["base_frame_z"].values[0])

            # Hip positions
            for hip_idx in self.cfg.ref_hip_indices:
                hip_data = frame_data[frame_data["joint_idx"] == hip_idx]
                if not hip_data.empty:
                    x_rel = float(hip_data["base_rel_x"].values[0]) * self.cfg.size_prop + self.cfg.x_offset + x_shift
                    y_rel = -float(hip_data["base_rel_y"].values[0]) * self.cfg.size_prop
                    if hip_idx in self.cfg.left_indices:
                        y_rel += self.cfg.y_offset
                    elif hip_idx in self.cfg.right_indices:
                        y_rel -= self.cfg.y_offset
                    z_rel = float(hip_data["base_rel_z"].values[0]) * self.cfg.size_prop + self.cfg.z_offset
                    preprocessed[frame][hip_idx] = {"x_rel": x_rel, "y_rel": y_rel, "z_rel": z_rel}

            # Foot positions
            for foot_idx in self.cfg.ref_foot_indices:
                foot_data = frame_data[frame_data["joint_idx"] == foot_idx]
                if not foot_data.empty:
                    x_rel = float(foot_data["base_rel_x"].values[0]) * self.cfg.size_prop + self.cfg.x_offset + x_shift
                    y_rel = -float(foot_data["base_rel_y"].values[0]) * self.cfg.size_prop
                    if foot_idx in self.cfg.left_indices:
                        y_rel += self.cfg.y_offset
                    elif foot_idx in self.cfg.right_indices:
                        y_rel -= self.cfg.y_offset
                    z_rel = float(foot_data["base_rel_z"].values[0]) * self.cfg.size_prop + self.cfg.z_offset
                    preprocessed[frame][foot_idx] = {"x_rel": x_rel, "y_rel": y_rel, "z_rel": z_rel}

        # 좌우 대칭화: 원본 데이터의 비대칭을 제거하고 좌우 y값을 평균값으로 통일
        lr_pairs = list(zip(self.cfg.left_indices, self.cfg.right_indices))
        for frame in preprocessed:
            for left_idx, right_idx in lr_pairs:
                if left_idx in preprocessed[frame] and right_idx in preprocessed[frame]:
                    avg_y = (preprocessed[frame][left_idx]["y_rel"] - preprocessed[frame][right_idx]["y_rel"]) / 2.0
                    preprocessed[frame][left_idx]["y_rel"] = avg_y
                    preprocessed[frame][right_idx]["y_rel"] = -avg_y
                    avg_x = (preprocessed[frame][left_idx]["x_rel"] + preprocessed[frame][right_idx]["x_rel"]) / 2.0
                    preprocessed[frame][left_idx]["x_rel"] = avg_x
                    preprocessed[frame][right_idx]["x_rel"] = avg_x
                    avg_z = (preprocessed[frame][left_idx]["z_rel"] + preprocessed[frame][right_idx]["z_rel"]) / 2.0
                    preprocessed[frame][left_idx]["z_rel"] = avg_z
                    preprocessed[frame][right_idx]["z_rel"] = avg_z

        return preprocessed

    def _get_target_reference_data(self) -> dict:
        """각 환경의 현재 프레임/모션 ID에 해당하는 참조 데이터를 반환합니다."""
        motion_ids = self._interaction_command.squeeze(-1).tolist()  # [N]
        frames = self._current_frames.squeeze(-1).tolist()  # [N]

        motion_data_list = []
        for mid, f in zip(motion_ids, frames):
            mid_str = str(int(mid))
            f_int = int(f)
            if mid_str in self.all_preprocessed_data and f_int in self.all_preprocessed_data[mid_str]:
                motion_data_list.append(self.all_preprocessed_data[mid_str][f_int])
            else:
                # 유효하지 않은 경우 모션 "0", 프레임 0 사용
                motion_data_list.append(self.all_preprocessed_data.get("0", {}).get(0, {}))

        def _safe_xyz(d: dict, idx: int) -> list:
            entry = d.get(idx, {})
            return [entry.get("x_rel", 0.0), entry.get("y_rel", 0.0), entry.get("z_rel", 0.0)]

        # Hip positions [N, 4, 3]
        hip_pos_local = torch.tensor(
            [[_safe_xyz(md, hi) for hi in self.cfg.ref_hip_indices] for md in motion_data_list],
            device=self.device,
            dtype=torch.float,
        )

        # Foot positions [N, 4, 3]
        foot_pos_local = torch.tensor(
            [[_safe_xyz(md, fi) for fi in self.cfg.ref_foot_indices] for md in motion_data_list],
            device=self.device,
            dtype=torch.float,
        )

        # root에 더해서 world 좌표로 변환
        root_pos = self._robot.data.root_link_pos_w.unsqueeze(1)  # [N, 1, 3]
        hip_pos_world = root_pos + hip_pos_local
        foot_pos_world = root_pos + foot_pos_local

        # Base rotation [N, 3]
        base_rot = torch.tensor(
            [[md.get("roll", 0.0), -md.get("pitch", 0.0), md.get("yaw", 0.0)] for md in motion_data_list],
            device=self.device,
            dtype=torch.float,
        )

        # Base height [N]
        base_height = torch.tensor(
            [md.get("base_height", 0.34) for md in motion_data_list],
            device=self.device,
            dtype=torch.float,
        )

        return {
            "hip_positions": hip_pos_world,
            "foot_positions": foot_pos_world,
            "hip_positions_local": hip_pos_local,
            "foot_positions_local": foot_pos_local,
            "base_rotation": base_rot,
            "base_height": base_height,
        }

    # Stand-up 자세 body-frame 타겟 (뒷발 지지, 앞발 들기)
    # 순서: FL, FR, RL, RR  /  x=전후, y=좌우, z=상하
    STAND_HIP_LOCAL = [
        [0.18, 0.07, 0.03],  # FL hip: 앞-상
        [0.18, -0.07, 0.03],  # FR hip: 앞-상
        [-0.18, 0.07, -0.03],  # RL hip: 뒤-하
        [-0.18, -0.07, -0.03],  # RR hip: 뒤-하
    ]
    STAND_FOOT_LOCAL = [
        [0.15, 0.09, 0.28],  # FL foot: 들어올림
        [0.15, -0.09, 0.28],  # FR foot: 들어올림
        [-0.10, 0.09, -0.48],  # RL foot: 땅에 닿도록
        [-0.10, -0.09, -0.48],  # RR foot: 땅에 닿도록
    ]
    STAND_BASE_HEIGHT = 0.65  # 기립 시 base 높이 (m)
    STAND_BASE_PITCH = 1.1  # 기립 시 pitch (rad, ~63도)

    def _calculate_reference_positions(self):
        """참조 위치(월드 및 로컬)를 계산하고 내부 버퍼에 저장합니다."""
        ref = self._get_target_reference_data()

        # Stand 커맨드(cmd=3): CSV 모션 대신 하드코딩된 body-frame 기립 자세 타겟 사용
        mask_stand = self._interaction_command.squeeze(-1) == 3
        if mask_stand.any():
            n_stand = int(mask_stand.sum().item())
            stand_hip = (
                torch.tensor(self.STAND_HIP_LOCAL, device=self.device, dtype=torch.float)
                .unsqueeze(0)
                .expand(n_stand, -1, -1)
            )  # [n_stand, 4, 3]
            stand_foot = (
                torch.tensor(self.STAND_FOOT_LOCAL, device=self.device, dtype=torch.float)
                .unsqueeze(0)
                .expand(n_stand, -1, -1)
            )  # [n_stand, 4, 3]

            ref["hip_positions_local"][mask_stand] = stand_hip
            ref["foot_positions_local"][mask_stand] = stand_foot
            ref["base_height"][mask_stand] = self.STAND_BASE_HEIGHT
            ref["base_rotation"][mask_stand, 1] = self.STAND_BASE_PITCH  # pitch

            # world 좌표도 갱신 (root_pos 기준 단순 오프셋 – world 모드용)
            root_pos_stand = self._robot.data.root_link_pos_w[mask_stand].unsqueeze(1)
            ref["hip_positions"][mask_stand] = root_pos_stand + stand_hip
            ref["foot_positions"][mask_stand] = root_pos_stand + stand_foot

        self._target_reference_positions = {
            "hip": ref["hip_positions"],
            "foot": ref["foot_positions"],
            "hip_local": ref["hip_positions_local"],
            "foot_local": ref["foot_positions_local"],
            "base_rotation": ref["base_rotation"],
            "base_height": ref["base_height"],
        }

        # ---- 디버그 시각화 (hip/foot 부위별 4색, base 흰색) ----
        if hasattr(self, "_ref_hip_vis"):
            robot_hip_pos = self._robot.data.body_pos_w[:, self._hip_body_ids, :]
            robot_foot_pos = self._robot.data.body_pos_w[:, self._foot_body_ids, :]

            if self.cfg.reward_frame == "base":
                # base frame 타겟을 world space로 변환하여 시각화
                # hip_local / foot_local 은 이미 로봇 body frame 상대 오프셋이므로
                # 현재 로봇 회전을 적용한 뒤 world 위치를 더하면 됩니다
                root_pos_w = self._robot.data.root_link_pos_w  # [N, 3]
                root_quat = self._robot.data.root_link_quat_w  # [N, 4]

                hip_local = ref["hip_positions_local"]  # [N, 4, 3]
                foot_local = ref["foot_positions_local"]  # [N, 4, 3]

                root_quat_rep = root_quat.unsqueeze(1).repeat(1, 4, 1)  # [N, 4, 4]

                # body frame offset → world frame
                vis_hip_world = root_pos_w.unsqueeze(1) + quat_apply(root_quat_rep, hip_local)  # [N, 4, 3]
                vis_foot_world = root_pos_w.unsqueeze(1) + quat_apply(root_quat_rep, foot_local)  # [N, 4, 3]
            else:
                vis_hip_world = ref["hip_positions"]  # [N, 4, 3]
                vis_foot_world = ref["foot_positions"]  # [N, 4, 3]

            for i in range(4):
                self._ref_hip_vis[i].visualize(vis_hip_world[:, i, :])
                self._ref_foot_vis[i].visualize(vis_foot_world[:, i, :])
                self._robot_hip_vis[i].visualize(robot_hip_pos[:, i, :])
                self._robot_foot_vis[i].visualize(robot_foot_pos[:, i, :])

            base_pos = self._robot.data.root_link_pos_w.clone()
            target_h = ref["base_height"].clone()
            # 눕기(interaction_command == 2)일 경우 base_height 0.05로 고정
            mask_lay = self._interaction_command.squeeze(-1) == 2
            target_h[mask_lay] = 0.05
            base_pos[:, 2] = target_h
            self._base_target_vis.visualize(base_pos)

    def _step_motion_frames(self):
        """모션 프레임 카운터를 업데이트합니다 (post_physics_step에서 호출)."""
        if len(self._motion_lengths_tensor) == 0:
            return
        # 각 env의 interaction_command에 해당하는 모션 길이로 mod
        motion_len = self._motion_lengths_tensor[self._interaction_command.squeeze(-1)].unsqueeze(-1)  # [N, 1]
        self._current_frames = (self._current_frames + 1) % motion_len

    # ---------------------------------------------------------------------- #
    # 디버그 시각화 (Debug Visualization)
    # ---------------------------------------------------------------------- #

    def _set_debug_vis_impl(self, debug_vis: bool):
        """디버그 마커 시각화를 설정합니다."""
        if debug_vis:
            if not hasattr(self, "_ref_hip_vis"):
                # FL(0), FR(1), RL(2), RR(3) 색상 매핑
                # 빨강, 초록, 파랑, 노랑
                colors = [
                    (1.0, 0.0, 0.0),  # FL: Red
                    (0.0, 1.0, 0.0),  # FR: Green
                    (0.0, 0.0, 1.0),  # RL: Blue
                    (1.0, 1.0, 0.0),  # RR: Yellow
                ]
                names = ["FL", "FR", "RL", "RR"]

                self._ref_hip_vis = []
                self._ref_foot_vis = []
                self._robot_hip_vis = []
                self._robot_foot_vis = []

                for i, color in enumerate(colors):
                    # Ref Hip
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.markers["sphere"].radius = 0.03
                    marker_cfg.markers["sphere"].visual_material.diffuse_color = color
                    marker_cfg.prim_path = f"/Visuals/Debug/ref_hip_{names[i]}"
                    self._ref_hip_vis.append(VisualizationMarkers(marker_cfg))

                    # Ref Foot
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.markers["sphere"].radius = 0.03
                    marker_cfg.markers["sphere"].visual_material.diffuse_color = color
                    marker_cfg.prim_path = f"/Visuals/Debug/ref_foot_{names[i]}"
                    self._ref_foot_vis.append(VisualizationMarkers(marker_cfg))

                    # Robot Hip
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.markers["sphere"].radius = 0.04
                    marker_cfg.markers["sphere"].visual_material.diffuse_color = color
                    marker_cfg.prim_path = f"/Visuals/Debug/robot_hip_{names[i]}"
                    self._robot_hip_vis.append(VisualizationMarkers(marker_cfg))

                    # Robot Foot
                    marker_cfg = SPHERE_MARKER_CFG.copy()
                    marker_cfg.markers["sphere"].radius = 0.04
                    marker_cfg.markers["sphere"].visual_material.diffuse_color = color
                    marker_cfg.prim_path = f"/Visuals/Debug/robot_foot_{names[i]}"
                    self._robot_foot_vis.append(VisualizationMarkers(marker_cfg))

                # Base (Body) - 흰색으로 기준점 표시
                marker_cfg_base = SPHERE_MARKER_CFG.copy()
                marker_cfg_base.markers["sphere"].radius = 0.05
                marker_cfg_base.markers["sphere"].visual_material.diffuse_color = (1.0, 1.0, 1.0)
                marker_cfg_base.prim_path = "/Visuals/Debug/target_base"
                self._base_target_vis = VisualizationMarkers(marker_cfg_base)

            for i in range(4):
                self._ref_hip_vis[i].set_visibility(True)
                self._ref_foot_vis[i].set_visibility(True)
                self._robot_hip_vis[i].set_visibility(True)
                self._robot_foot_vis[i].set_visibility(True)
            self._base_target_vis.set_visibility(True)
        else:
            if hasattr(self, "_ref_hip_vis"):
                for i in range(4):
                    self._ref_hip_vis[i].set_visibility(False)
                    self._ref_foot_vis[i].set_visibility(False)
                    self._robot_hip_vis[i].set_visibility(False)
                    self._robot_foot_vis[i].set_visibility(False)
                self._base_target_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        """디버그 마커를 렌더 스텝마다 업데이트하기 위한 콜백.
        실제 마커 위치는 _calculate_world_positions 내부에서 매 물리 스텝마다 갱신되므로 여기서는 추가 작업 불필요.
        """
        pass
