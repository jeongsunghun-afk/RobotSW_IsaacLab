# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers, SPHERE_MARKER_CFG
import isaaclab.utils.math as math_utils

from .go2_wtw_env import WTWEnv, torch_rand_float
from .go2_env_cfg import Go2NeckFlatEnvCfg


class Go2NeckEnv(WTWEnv):
    """Go2 로봇에 7-DOF 목 모듈이 부착된 환경.

    IsaacLab의 DifferentialIKController와 physics jacobian을 활용하여
    base_frame 기준 타겟 위치로 목을 자연스럽게 제어합니다.
    """

    cfg: Go2NeckFlatEnvCfg

    def __init__(self, cfg: Go2NeckFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -------------------------------------------------------------------
        # 조인트 / 바디 인덱스 파악
        # -------------------------------------------------------------------
        joint_names = getattr(self._robot.data, "joint_names", None)
        if joint_names is None:
            print("[WARN] Robot joint names are unavailable; skipping action mapping print.")
            return

        for idx, name in enumerate(joint_names):
            print(f"  {idx:02d}: {name}")

        # neck 조인트 이름 (URDF 순서: y, p, y, r, y, p, y)
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
        print(f"  Found neck joints: {found}, IDs: {self._neck_joint_ids}")

        # action vector (19-DOF) 내에서 neck 조인트의 인덱스
        # Go2 기본 12-DOF(hip,thigh,calf × 4) + neck 7-DOF
        # URDF joint 순서에서 action index는 joint_pos 순서와 동일
        self._action_neck_ids = list(self._neck_joint_ids)  # 실제 joint ID와 동일
        self._non_neck_ids = [i for i in range(19) if i not in self._action_neck_ids]

        # -------------------------------------------------------------------
        # head 바디 인덱스 (physics jacobian, EE pose 계산용)
        # N_joint7_neck_y 뒤에 붙는 head link를 end-effector로 사용
        # 마지막 neck body = N_joint7_neck_y (또는 그 child head body)
        # -------------------------------------------------------------------
        # 먼저 모든 neck 바디를 확인하고 마지막을 head로 사용
        neck_body_names = [".*neck_p", ".*neck_r", ".*neck_y"]
        self._neck_ids, body_names = self._robot.find_bodies(neck_body_names)
        print(f"  Neck body IDs: {self._neck_ids}, names: {body_names}")

        # head는 neck 체인의 마지막 body (N_joint7_neck_y)
        self._head_body_idx = self._neck_ids[-1]
        print(f"  Head body index: {self._head_body_idx}")

        # floating-base: jacobian에서 6-DOF 기반 자유도가 앞에 추가됨
        # body idx(physX): 0 = root body이므로 그대로 사용 (floating base는 _head_body_idx 그대로)
        self._jacobi_head_body_idx = self._head_body_idx  # non-fixed base
        # joint jacobian columns: 6(floating dof) + actual_joint_id
        self._jacobi_neck_joint_ids = [i + 6 for i in self._neck_joint_ids]

        # -------------------------------------------------------------------
        # IsaacLab DifferentialIKController (DLS 방식, position-only)
        # -------------------------------------------------------------------
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",  # 위치만 제어 (orientation 무시)
            use_relative_mode=False,  # 절대 위치 타겟
            ik_method="dls",
            ik_params={"lambda_val": 0.05},  # damping 계수
        )
        self._ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=self.num_envs, device=self.device)

        # -------------------------------------------------------------------
        # 상태 변수
        # -------------------------------------------------------------------
        # 현재 스무딩된 neck target 조인트 각도 (절대값)
        self._current_neck_q = torch.zeros(self.num_envs, 7, device=self.device)

        # base_frame(root frame) 기준 IK 타겟 위치
        # neck의 EE(head)가 도달할 목표 좌표 (root_frame 기준)
        self._ik_target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

        # IK 타겟 재선정 타이머
        self._ik_target_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ik_target_duration_range = (100, 300)  # 2~6초 @ 50Hz

        # neck default joint pos (action 변환용)
        # _robot이 초기화된 후 사용 가능
        self._neck_joint_default = self._robot.data.default_joint_pos[:, self._neck_joint_ids].clone()

        print("[Go2NeckEnv] Initialized with IsaacLab DifferentialIKController (DLS, position mode)")

    # -----------------------------------------------------------------------
    # Pre-physics step: IsaacLab IK 컨트롤러 기반 목 제어
    # -----------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

        # === 1. 타겟 재선정 타이머 관리 ===
        self._ik_target_timer -= 1
        reset_mask = self._ik_target_timer <= 0
        if reset_mask.any():
            reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
            self._resample_ik_targets(reset_ids)

        # === 2. 현재 EE(head) pose를 root_frame(base_frame) 기준으로 계산 ===
        # task_space_actions.py의 _compute_frame_pose 패턴 사용
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._head_body_idx]  # (N, 3)
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._head_body_idx]  # (N, 4)
        root_pos_w = self._robot.data.root_link_pos_w  # (N, 3)
        root_quat_w = self._robot.data.root_link_quat_w  # (N, 4)

        # world → root_frame 변환
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )  # (N, 3), (N, 4)

        # === 3. IK 타겟을 컨트롤러에 설정 ===
        # set_command: command는 (N, 3) 절대 위치, position 모드에서 ee_quat 필요
        self._ik_controller.set_command(
            self._ik_target_pos_b,
            ee_pos=ee_pos_b,
            ee_quat=ee_quat_b,
        )

        # === 4. Physics Jacobian 획득 및 root_frame 변환 ===
        # task_space_actions.py의 jacobian_b 패턴 사용
        # get_jacobians() shape: (N, num_bodies, 6, 6+num_joints)
        jacobian_w = self._robot.root_physx_view.get_jacobians()[
            :, self._jacobi_head_body_idx, :, self._jacobi_neck_joint_ids
        ]  # (N, 6, 7)

        # world frame → root frame으로 회전 변환
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))  # (N, 3, 3)
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        # position-only IK이므로 jacobian의 상위 3행(translational)만 사용
        # DifferentialIKController는 command_type="position"일 때 자동으로 [:, 0:3]만 사용
        # (differential_ik.py L165: jacobian_pos = jacobian[:, 0:3])

        # === 5. IK 계산: 현재 neck 조인트 각도 → 목표 조인트 각도 ===
        neck_joint_pos = self._robot.data.joint_pos[:, self._neck_joint_ids]  # (N, 7)

        # DifferentialIKController.compute() 호출
        # 반환: (N, 7) 목표 조인트 각도 (절대값)
        target_neck_q = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, neck_joint_pos)  # (N, 7)

        # === 6. 조인트 제한 클램핑 ===
        if hasattr(self._robot.data, "soft_joint_pos_limits"):
            lower = self._robot.data.soft_joint_pos_limits[:, self._neck_joint_ids, 0]  # (N, 7)
            upper = self._robot.data.soft_joint_pos_limits[:, self._neck_joint_ids, 1]  # (N, 7)
            target_neck_q = torch.clamp(target_neck_q, lower, upper)

        # === 7. 스무딩 (1차 저역통과 필터) ===
        alpha = 0.15  # 응답성/부드러움 트레이드오프 (낮을수록 부드러움)
        self._current_neck_q = (1.0 - alpha) * self._current_neck_q + alpha * target_neck_q

        # === 8. action vector에 반영 ===
        # processed_actions = action_scale * actions + default_joint_pos
        # → neck_action = (target_q - default_joint_pos) / action_scale
        neck_default = self._robot.data.default_joint_pos[:, self._neck_joint_ids]  # (N, 7)
        neck_action = (self._current_neck_q - neck_default) / self.cfg.action_scale
        self._actions[:, self._action_neck_ids] = neck_action

        actions = self._actions.clone()
        if self.cfg.hip_scale_reduction:
            actions[:, self._non_neck_ids[:4]] *= 0.5
        self._processed_actions = self.cfg.action_scale * actions + self._robot.data.default_joint_pos

        # === 9. 타겟 위치 시각화 ===
        self._update_neck_target_vis(root_pos_w, root_quat_w)

    # -----------------------------------------------------------------------
    # IK 타겟 재선정 (root_frame = base_frame 기준)
    # -----------------------------------------------------------------------

    def _resample_ik_targets(self, env_ids: torch.Tensor):
        """root_frame(base_frame) 기준으로 새 IK 타겟 위치를 샘플링합니다.

        head가 도달할 수 있는 범위는 neck chain 총 링크 길이(≈0.265m)를 고려합니다.
        root_frame의 원점은 로봇 base의 COM이므로, neck이 유체 앞쪽에 있음을 고려한 오프셋 적용.
        """
        n = len(env_ids)

        # root_frame 기준 타겟 위치 샘플링
        # neck root가 base 앞쪽 약 0.2m, 위쪽 약 0.05m에 위치
        # head가 neck root에서 약 0.25m 반경 내에 도달 가능
        x = torch_rand_float(0.15, 0.40, (n,), self.device)  # 앞 방향 (base 기준)
        y = torch_rand_float(-0.20, 0.20, (n,), self.device)  # 좌우
        z = torch_rand_float(0.00, 0.25, (n,), self.device)  # 상하 (base 기준, 위가 양수)

        self._ik_target_pos_b[env_ids, 0] = x
        self._ik_target_pos_b[env_ids, 1] = y
        self._ik_target_pos_b[env_ids, 2] = z

        # 지속 시간 설정
        low, high = self._ik_target_duration_range
        self._ik_target_timer[env_ids] = torch.randint(low, high, (n,), device=self.device)

    # -----------------------------------------------------------------------
    # 타겟 마커 시각화 업데이트
    # -----------------------------------------------------------------------

    def _update_neck_target_vis(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor):
        """IK 타겟 위치를 world 좌표로 변환하여 마커를 업데이트합니다.

        Args:
            root_pos_w: 로봇 root의 world 위치 (N, 3)
            root_quat_w: 로봇 root의 world 회전 (N, 4)
        """
        if not hasattr(self, "_neck_target_visualizer"):
            return

        # root_frame 기준 타겟 → world frame 변환
        # p_world = root_pos_w + R(root_quat_w) @ target_b
        target_w = root_pos_w + math_utils.quat_apply(root_quat_w, self._ik_target_pos_b)

        self._neck_target_visualizer.visualize(target_w)

    # -----------------------------------------------------------------------
    # Debug Visualization
    # -----------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        """부모의 visualizer에 추가로 IK 타겟 마커를 관리합니다."""
        super()._set_debug_vis_impl(debug_vis)

        if debug_vis:
            if not hasattr(self, "_neck_target_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.04  # 4cm 빨간 구
                marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.1, 0.1)  # 빨간색
                marker_cfg.prim_path = "/Visuals/Debug/neck_ik_target"
                self._neck_target_visualizer = VisualizationMarkers(marker_cfg)
            self._neck_target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_neck_target_visualizer"):
                self._neck_target_visualizer.set_visibility(False)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            self._current_neck_q[:] = 0.0
            self._ik_target_timer[:] = 0  # 즉시 새 타겟 선정
            self._ik_target_pos_b[:] = 0.0
            self._ik_controller.reset()
        else:
            self._current_neck_q[env_ids] = 0.0
            self._ik_target_timer[env_ids] = 0
            self._ik_target_pos_b[env_ids] = 0.0
            self._ik_controller.reset(env_ids)

    # -----------------------------------------------------------------------
    # Rewards / Dones (부모 재사용)
    # -----------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        return super()._get_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._get_dones()
