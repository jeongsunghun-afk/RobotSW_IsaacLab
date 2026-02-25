# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers, SPHERE_MARKER_CFG
import isaaclab.utils.math as math_utils

from .go2_interaction_env import Go2InteractionEnv
from .go2_neck_interaction_cfg import Go2NeckInteractionCfg

def torch_rand_float(lower: float, upper: float, shape: tuple, device: str) -> torch.Tensor:
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Go2NeckInteractionEnv(Go2InteractionEnv):
    """Go2 로봇에 7-DOF 목 모듈이 부착된 상호작용 환경."""

    cfg: Go2NeckInteractionCfg

    def __init__(self, cfg: Go2NeckInteractionCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -------------------------------------------------------------------
        # 조인트 / 바디 인덱스 파악
        # -------------------------------------------------------------------
        neck_joint_names = [
            "N_joint1_neck_y",
            "N_joint2_neck_p",
            "N_joint3_neck_y",
            "N_joint4_neck_r",
            "N_joint5_neck_y",
            "N_joint6_neck_p",
            "N_joint7_neck_y",
        ]
        self._neck_joint_ids, found = self._robot.find_joints(neck_joint_names)
        self._action_neck_ids = list(self._neck_joint_ids)  # 실제 joint ID와 동일
        self._non_neck_ids = [i for i in range(19) if i not in self._action_neck_ids]

        # head 바디 인덱스 (physics jacobian, EE pose 계산용)
        neck_body_names = [".*neck_p", ".*neck_r", ".*neck_y"]
        self._neck_ids, body_names = self._robot.find_bodies(neck_body_names)
        self._head_body_idx = self._neck_ids[-1]

        self._jacobi_head_body_idx = self._head_body_idx  # non-fixed base
        self._jacobi_neck_joint_ids = [i + 6 for i in self._neck_joint_ids]

        # DifferentialIKController
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self._ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=self.num_envs, device=self.device)

        # 상태 변수
        self._current_neck_q = torch.zeros(self.num_envs, 7, device=self.device)
        self._ik_target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._ik_target_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ik_target_duration_range = (100, 300)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

        # IK 타겟 재선정 타이머 관리
        self._ik_target_timer -= 1
        reset_mask = self._ik_target_timer <= 0
        if reset_mask.any():
            reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
            self._resample_ik_targets(reset_ids)

        # EE(head) pose
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._head_body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._head_body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        self._ik_controller.set_command(
            self._ik_target_pos_b,
            ee_pos=ee_pos_b,
            ee_quat=ee_quat_b,
        )

        jacobian_w = self._robot.root_physx_view.get_jacobians()[
            :, self._jacobi_head_body_idx, :, self._jacobi_neck_joint_ids
        ]

        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        neck_joint_pos = self._robot.data.joint_pos[:, self._neck_joint_ids]
        target_neck_q = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, neck_joint_pos)

        # 조인트 제한 클램핑
        if hasattr(self._robot.data, "soft_joint_pos_limits"):
            lower = self._robot.data.soft_joint_pos_limits[:, self._neck_joint_ids, 0]
            upper = self._robot.data.soft_joint_pos_limits[:, self._neck_joint_ids, 1]
            target_neck_q = torch.clamp(target_neck_q, lower, upper)

        alpha = 0.15
        self._current_neck_q = (1.0 - alpha) * self._current_neck_q + alpha * target_neck_q

        # action vector에 반영
        neck_default = self._robot.data.default_joint_pos[:, self._neck_joint_ids]
        neck_action = (self._current_neck_q - neck_default) / self.cfg.action_scale
        self._actions[:, self._action_neck_ids] = neck_action

        # 상호작용 행동 스케일 적용
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        self._update_neck_target_vis(root_pos_w, root_quat_w)

    def _resample_ik_targets(self, env_ids: torch.Tensor):
        n = len(env_ids)
        x = torch_rand_float(0.15, 0.40, (n,), self.device)
        y = torch_rand_float(-0.20, 0.20, (n,), self.device)
        z = torch_rand_float(0.00, 0.25, (n,), self.device)

        self._ik_target_pos_b[env_ids, 0] = x
        self._ik_target_pos_b[env_ids, 1] = y
        self._ik_target_pos_b[env_ids, 2] = z

        low, high = self._ik_target_duration_range
        self._ik_target_timer[env_ids] = torch.randint(low, high, (n,), device=self.device)

    def _update_neck_target_vis(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor):
        if not hasattr(self, "_neck_target_visualizer"):
            return
        target_w = root_pos_w + math_utils.quat_apply(root_quat_w, self._ik_target_pos_b)
        self._neck_target_visualizer.visualize(target_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)

        if debug_vis:
            if not hasattr(self, "_neck_target_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.04
                marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Debug/neck_ik_target"
                self._neck_target_visualizer = VisualizationMarkers(marker_cfg)
            self._neck_target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_neck_target_visualizer"):
                self._neck_target_visualizer.set_visibility(False)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            self._current_neck_q[:] = 0.0
            self._ik_target_timer[:] = 0
            self._ik_target_pos_b[:] = 0.0
            self._ik_controller.reset()
        else:
            self._current_neck_q[env_ids] = 0.0
            self._ik_target_timer[env_ids] = 0
            self._ik_target_pos_b[env_ids] = 0.0
            self._ik_controller.reset(env_ids)
