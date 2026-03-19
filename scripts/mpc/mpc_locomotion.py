"""
MPC(Model Predictive Control) 기반 4족 보행 제어기

참고 문헌:
  - Kim et al. (2019) "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control
    and Model Predictive Control", IROS 2019.
  - Di Carlo et al. (2018) "Dynamic Locomotion in the MIT Cheetah 3 Through Convex
    Model-Predictive Control", IROS 2018.

이 모듈은 IsaacLab 시뮬레이터 환경(Go2, Go2Neck, R_Skeleton)에서
MPC 기반 4족 보행을 수행하기 위한 제어기를 제공합니다.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch


# =============================================================================
# 1. 보행 패턴 스케줄러 (Gait Scheduler)
# =============================================================================

class GaitScheduler:
    """Trot 및 Stand 보행 패턴을 관리합니다."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        gait_type: str = "trot",
        gait_period: float = 0.5,
        stance_ratio: float = 0.5,
        dt: float = 0.02,
    ):
        self.num_envs = num_envs
        self.device = device
        self.gait_type = gait_type
        self.gait_period = gait_period
        self.stance_ratio = stance_ratio
        self.dt = dt

        self._phase_offsets = {
            "trot":  torch.tensor([0.0, 0.5, 0.5, 0.0], device=device),
            "stand": torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
            "walk":  torch.tensor([0.0, 0.5, 0.25, 0.75], device=device),
            "pace":  torch.tensor([0.0, 0.0, 0.5, 0.5], device=device),
        }

        self._phase = torch.zeros(num_envs, 1, device=device)
        self._phase_vel = 1.0 / gait_period

    def set_gait(self, gait_type: str):
        if gait_type in self._phase_offsets:
            self.gait_type = gait_type
            print(f"[MPC] GaitScheduler: Gait 변경됨 -> {gait_type}")
        else:
            print(f"[MPC] GaitScheduler: 지원하지 않는 Gait -> {gait_type}")

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self._phase = torch.rand(self.num_envs, 1, device=self.device)
        else:
            new_phase = self._phase.clone()
            new_phase[env_ids] = torch.rand(len(env_ids), 1, device=self.device)
            self._phase = new_phase

    def update(self, dt: float) -> torch.Tensor:
        self.dt = dt
        self._phase = (self._phase + self._phase_vel * dt) % 1.0
        offsets = self._phase_offsets[self.gait_type]
        abs_phase = (self._phase + offsets.unsqueeze(0)) % 1.0

        if self.gait_type == "stand":
            contact = torch.ones(self.num_envs, 4, dtype=torch.bool, device=self.device)
        else:
            contact = abs_phase < self.stance_ratio
        
        self._last_mask = torch.stack([contact, ~contact], dim=-1) # (N, 4, 2)
        return self._last_mask

    def get_swing_phase(self) -> torch.Tensor:
        offsets = self._phase_offsets[self.gait_type]
        abs_phase = (self._phase + offsets.unsqueeze(0)) % 1.0
        is_swing = abs_phase >= self.stance_ratio
        swing_dur = 1.0 - self.stance_ratio
        swing_phase = torch.where(
            is_swing,
            (abs_phase - self.stance_ratio) / (swing_dur + 1e-6),
            torch.zeros_like(abs_phase),
        )
        return swing_phase.clamp(0.0, 1.0)


# =============================================================================
# 2. 스윙 다리 궤적 생성기 (Swing Leg Trajectory)
# =============================================================================

class SwingLegTrajectory:
    def __init__(self, swing_height: float = 0.08, device: str = "cuda"):
        self.swing_height = swing_height
        self.device = device

    def compute_trajectory(
        self,
        p_curr: torch.Tensor,
        p_target: torch.Tensor,
        phase: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        t = phase.unsqueeze(-1)
        h = torch.zeros(1, 1, 3, device=self.device)
        h[0, 0, 2] = self.swing_height
        P0, P1, P2, P3 = p_curr, p_curr + h, p_target + h, p_target
        p_des = (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3
        return p_des


# =============================================================================
# 3. 단순화된 MPC 솔버 (Simplified SRBD-MPC)
# =============================================================================

class SimpleMPCSolver:
    def __init__(self, mass: float, gravity: float = 9.81, mu: float = 0.6, device: str = "cuda"):
        self.mass = mass
        self.gravity = gravity
        self.mu = mu
        self.device = device
        self.g_vec = torch.tensor([0.0, 0.0, -gravity], device=device)

    def compute_grf(self, contact, base_quat, base_lin_vel_b, base_ang_vel_b, target_lin_vel_b, target_ang_vel_b, feet_pos_w, base_pos_w, inertia=None) -> torch.Tensor:
        N = contact.shape[0]
        kp_lin, kd_lin = 80.0, 20.0
        kp_ang, kd_ang = 40.0, 10.0
        vel_error = target_lin_vel_b - base_lin_vel_b
        ang_error = target_ang_vel_b - base_ang_vel_b
        a_des_b = kp_lin * vel_error - kd_lin * base_lin_vel_b
        R_bw = quat_to_rotation_matrix(base_quat)
        a_des_w = torch.bmm(R_bw, a_des_b.unsqueeze(-1)).squeeze(-1)
        F_total = self.mass * (a_des_w - self.g_vec.unsqueeze(0))
        n_contact = contact.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        fz_each = F_total[:, 2:3] / n_contact
        grf = torch.zeros(N, 4, 3, device=self.device)
        grf[:, :, 2] = fz_each * contact.float()
        grf[:, :, 0] = (F_total[:, 0:1] / n_contact) * contact.float()
        grf[:, :, 1] = (F_total[:, 1:2] / n_contact) * contact.float()
        grf[:, :, 2] = grf[:, :, 2].clamp(min=0.0)
        f_xy_norm = torch.norm(grf[:, :, :2], dim=-1, keepdim=True) + 1e-6
        max_f_xy = self.mu * grf[:, :, 2:3]
        scale = (max_f_xy / f_xy_norm).clamp(max=1.0)
        grf[:, :, :2] *= scale
        return grf * contact.float().unsqueeze(-1)


# =============================================================================
# 4. 발 배치 계획기 (Foot Placement Planner — Raibert Heuristic)
# =============================================================================

class FootPlacementPlanner:
    def __init__(self, device: str, nominal_foot_offsets_b: Optional[torch.Tensor] = None):
        self.device = device
        self.stance_duration = 0.25 # 기본 스탠스 지속 시간
        self.k_raibert = 0.30 # 더욱 공격적인 가속 유도
        self.foot_clearance = 0.05 
        self.nominal_foot_offsets_b = nominal_foot_offsets_b if nominal_foot_offsets_b is not None else torch.tensor([[0.18, 0.15, 0], [0.18, -0.15, 0], [-0.18, 0.15, 0], [-0.18, -0.15, 0]], device=device)

    def compute_target(self, base_pos_w, base_quat_w, base_lin_vel_b, target_lin_vel_b, contact_schedule, floor_height=0.0) -> torch.Tensor:
        # 1. 실제 Orientation에서 Yaw만 추출 (Virtual Orientation)
        yaws = quat_to_yaw(base_quat_w)
        R_v_wb = yaw_to_rotation_matrix(yaws)
        
        # 2. Virtual Base 기준 중립 발 위치 → World
        foot_b = self.nominal_foot_offsets_b.unsqueeze(0).expand(base_pos_w.shape[0], -1, -1)
        p_v_foot_w = torch.bmm(foot_b, R_v_wb.transpose(1, 2)) + base_pos_w.unsqueeze(1)
        
        # 3. Raibert Heuristic (World Coordinate)
        v_w = torch.bmm(R_v_wb, base_lin_vel_b.unsqueeze(-1)).squeeze(-1)
        v_des_w = torch.bmm(R_v_wb, target_lin_vel_b.unsqueeze(-1)).squeeze(-1)
        p_target = p_v_foot_w + (v_w * self.stance_duration / 2).unsqueeze(1) + self.k_raibert * (v_w - v_des_w).unsqueeze(1)
        p_target[:, :, 2] = floor_height + self.foot_clearance
        return p_target


# =============================================================================
# 로봇 사양 정의 (Robot Specifications)
# =============================================================================

@dataclass
class RobotSpec:
    name: str
    l_hip: float = 0.095
    l_thigh: float = 0.213
    l_calf: float = 0.213
    # Skeleton은 전방(Front)과 후방(Rear) 링크 길이가 다름
    l_up_f: float = 0.25; l_lo_f: float = 0.25; l_h_f: float = 0.10
    l_up_r: float = 0.30; l_lo_r: float = 0.30; l_h_r: float = 0.10
    nominal_height: float = 0.30
    hip_joint_offsets: List[List[float]] = field(default_factory=list)
    nominal_foot_offsets: List[List[float]] = field(default_factory=list)
    action_dof: int = 12

ROBOT_SPECS = {
    "go2": RobotSpec(
        name="go2",
        nominal_height=0.32, # 가동 범위 확보를 위해 0.34에서 소폭 하향
        hip_joint_offsets=[[0.19, 0.045, 0], [0.19, -0.045, 0], [-0.19, 0.045, 0], [-0.19, -0.045, 0]],
        nominal_foot_offsets=[[0.18, 0.16, 0], [0.18, -0.16, 0], [-0.18, 0.16, 0], [-0.18, -0.16, 0]],
        action_dof=12
    ),
    "go2_neck": RobotSpec(
        name="go2_neck",
        nominal_height=0.34,
        hip_joint_offsets=[[0.19, 0.045, 0], [0.19, -0.045, 0], [-0.19, 0.045, 0], [-0.19, -0.045, 0]],
        nominal_foot_offsets=[[0.18, 0.16, 0], [0.18, -0.16, 0], [-0.18, 0.16, 0], [-0.18, -0.16, 0]],
        action_dof=12
    ),
    "skeleton": RobotSpec(
        name="skeleton",
        nominal_height=0.50,
        hip_joint_offsets=[[0.23, 0.15, 0], [0.23, -0.15, 0], [-0.23, 0.15, 0], [-0.23, -0.15, 0]],
        nominal_foot_offsets=[[0.20, 0.25, 0], [0.20, -0.25, 0], [-0.20, 0.25, 0], [-0.20, -0.25, 0]],
        action_dof=38
    )
}

# =============================================================================
# 상위 보행 제어기 (MPC Locomotion Controller)
# =============================================================================

class MPCLocomotionController:
    def __init__(self, robot_type: str, num_envs: int = 1, device: str = "cuda"):
        self.device, self.robot_type, self.num_envs = device, robot_type, num_envs
        self.spec = ROBOT_SPECS.get(robot_type, ROBOT_SPECS["go2"])
        
        self.gait = GaitScheduler(num_envs, device)
        self.swing = SwingLegTrajectory(device=device)
        self.mpc = SimpleMPCSolver(mass=12.0, device=device)
        self.foot_planner = FootPlacementPlanner(device, 
            nominal_foot_offsets_b=torch.tensor(self.spec.nominal_foot_offsets, device=device))
        
        self.action_scale = 0.6 # 기립과 추진의 타협점 (0.5에서 검증된 추진력 활용)
        self._target_pos_w = torch.zeros(num_envs, 4, 3, device=device)
        self._p_lift_w = torch.zeros(num_envs, 4, 3, device=device)
        self._prev_contact = torch.ones(num_envs, 4, dtype=torch.bool, device=device)
        self._is_initialized = False
        self._joint_idx_map = {}
        self._ik_offsets = torch.zeros(4, 3, device=device) # [FL, FR, RL, RR] x [abd, up, lo]

    def initialize(self, env):
        self._default_joint_pos = env._robot.data.default_joint_pos.clone()
        joint_names = env._robot.data.joint_names
        
        patterns = [["FL_hip", "FL_thigh", "FL_calf"], ["FR_hip", "FR_thigh", "FR_calf"],
                    ["RL_hip", "RL_thigh", "RL_calf", "HL_hip", "HL_thigh", "HL_knee", "HL_ankle"],
                    ["RR_hip", "RR_thigh", "RR_calf", "HR_hip", "HR_thigh", "HR_knee", "HR_ankle"]]
        
        for i, leg_patterns in enumerate(patterns):
            self._joint_idx_map[i] = []
            for p in leg_patterns:
                for idx, name in enumerate(joint_names):
                    if p in name: self._joint_idx_map[i].append(idx)
        
        ids, found = env._robot.find_bodies(".*foot|.*toe")
        self._foot_indices = ids
        
        # --- IK Self-Calibration ---
        # 초기 자세(default_joint_pos)에서의 발 위치를 계산하고 이를 수식의 0점과 동기화
        root_pos = env._robot.data.root_pos_w[0:1, :3]
        root_quat = env._robot.data.root_quat_w[0:1, :]
        yaws = quat_to_yaw(root_quat)
        R_v_bw = yaw_to_rotation_matrix(yaws).transpose(1, 2)
        
        feet_w = env._robot.data.body_pos_w[0:1, self._foot_indices, :]
        feet_v = torch.bmm(R_v_bw, (feet_w - root_pos).transpose(1, 2)).transpose(1, 2)
        
        # 오프셋 없이 IK 계산 시도
        self._ik_offsets = torch.zeros(4, 3, device=self.device)
        raw_targets = self._go2_ik(feet_v, torch.ones(1, 4, device=self.device), self.spec.action_dof)
        # action_scale 및 default_joint_pos 가 보정된 상태이므로 역산하여 순수 q_ik 추출
        q_raw = raw_targets * self.action_scale + self._default_joint_pos[0:1, :self.spec.action_dof]
        
        for i in range(4):
            idx = self._joint_idx_map[i]
            if len(idx) >= 3:
                q_def = self._default_joint_pos[0, idx[:3]]
                self._ik_offsets[i] = q_def - q_raw[0, idx[:3]]
        
        print(f"[MPC] IK Self-Calibration 완료 (Offsets): \n{self._ik_offsets}")
        self._is_initialized = True

    def reset(self, env_ids=None):
        self.gait.reset(env_ids)
        if env_ids is None: self._prev_contact[:] = True
        else: self._prev_contact[env_ids] = True

    def compute_action(self, env, observations):
        if not self._is_initialized:
            self.initialize(env)
            # 초기 기립을 위해 목표 위치를 nominal 위치로 설정 (World frame)
            base_pos_w = env._robot.data.root_pos_w[:, :3]
            base_quat_w = env._robot.data.root_quat_w
            yaws = quat_to_yaw(base_quat_w)
            R_v_wb = yaw_to_rotation_matrix(yaws)
            nom_offsets_w = torch.bmm(R_v_wb, torch.tensor(self.spec.nominal_foot_offsets, device=self.device).t().unsqueeze(0).expand(self.num_envs, -1, -1)).transpose(1, 2)
            self._target_pos_w = base_pos_w.unsqueeze(1) + nom_offsets_w
        
        base_pos_w = env._robot.data.root_pos_w[:, :3]
        base_quat_w = env._robot.data.root_quat_w
        base_lin_vel_b = env._robot.data.root_lin_vel_b
        target_lin_vel_b = env._commands[:, :3]
        
        mask = self.gait.update(env.step_dt)
        contact, swing = mask[:, :, 0], mask[:, :, 1]
        
        # 1. State Transition Check
        liftoff = self._prev_contact & (~contact.bool())
        p_foot_curr_w = env._robot.data.body_pos_w[:, self._foot_indices, :]
        if liftoff.any():
            self._p_lift_w[liftoff] = p_foot_curr_w[liftoff].detach().clone()
        
        # 2. 착지점/스윙 궤적 계획
        p_swing_tgt_w = self.foot_planner.compute_target(base_pos_w, base_quat_w, base_lin_vel_b, target_lin_vel_b, contact)
        p_des_w = self.swing.compute_trajectory(self._p_lift_w, p_swing_tgt_w, self.gait.get_swing_phase(), swing)
        
        # 3. Target Position 업데이트 (Stance Latching & Virtual Support)
        # 3. Target Position 업데이트 (Stance Latching & Virtual Support)
        # touchdown: 이전 프레임에서 swing이었으나 현재 contact인 상태
        touchdown = (~self._prev_contact) & contact.bool()
        
        for i in range(4):
            c_m, s_m = contact[:, i].bool(), swing[:, i].bool()
            td_m = touchdown[:, i]
            
            if c_m.any(): 
                # [Physics Fix] 착지 시점에만 위치를 고정(Latch)하여 지지대로 활용
                if td_m.any():
                    self._target_pos_w[td_m, i, 0:2] = p_foot_curr_w[td_m, i, 0:2].detach().clone()
                
                # Z는 항상 바닥을 누르도록 하여 기립 유지
                self._target_pos_w[c_m, i, 2] = -0.02
                
            if s_m.any(): 
                self._target_pos_w[s_m, i] = p_des_w[s_m, i].detach().clone()
        
        self._prev_contact = contact.bool().clone()
        return self._feet_pos_to_joint_action(env, self._target_pos_w, base_pos_w, base_quat_w, env._robot.data.joint_pos, contact)

    def _feet_pos_to_joint_action(self, env, p_des_w, base_pos_w, base_quat_w, joint_pos, contact):
        """Virtual Base(가상 베이스) 및 수직 복원력을 적용한 IK 엔진"""
        # 1. Orientation에서 Yaw만 추출 (수평 자세 기준)
        yaws = quat_to_yaw(base_quat_w)
        R_v_wb = yaw_to_rotation_matrix(yaws)
        R_v_bw = R_v_wb.transpose(1, 2)
        
        # 2. 높이 복원력 (Height Restoration) 강력 적용
        # 몸체가 낮을수록 '가상 베이스'를 높게 유지
        h_err = self.spec.nominal_height - base_pos_w[:, 2]
        p_virtual_w = base_pos_w.clone()
        p_virtual_w[:, 2] = self.spec.nominal_height + 3.0 * h_err # 높이 유지를 위해 3.0으로 상항
        
        # 3. 가상 기울기 (Virtual Leaning) — 전진 추진력 확보
        # 속도 오차가 클수록 몸체를 앞으로 숙여(가상 베이스를 전방으로) 다리를 뒤로 박차게 함
        v_err = env._commands[:, 0] - env._robot.data.root_lin_vel_b[:, 0]
        p_virtual_w[:, 0] += 0.2 * v_err # 기울기 게인 0.1 -> 0.2 상향
        
        # 4. World -> Virtual Body 변환
        p_des_b = torch.einsum('nij,nkj->nki', R_v_bw, p_des_w - p_virtual_w.unsqueeze(1))
        
        if self.robot_type == "skeleton": 
            return self._skeleton_ik(p_des_b, contact, self.spec.action_dof)
        return self._go2_ik(p_des_b, contact, self.spec.action_dof)

    def _go2_ik(self, p_des_b, contact, dof):
        l_thigh, l_calf, l_hip = self.spec.l_thigh, self.spec.l_calf, self.spec.l_hip
        hip_b = torch.tensor(self.spec.hip_joint_offsets, device=self.device)
        joint_targets = self._default_joint_pos[:, :dof].clone()
        side_signs = [1.0, -1.0, 1.0, -1.0]

        for i in range(4):
            idx = self._joint_idx_map[i]
            if not idx: continue
            
            # 1. Coordinate Transform to Hip Frame
            dp = p_des_b[:, i, :] - hip_b[i].unsqueeze(0)
            s = side_signs[i]
            dy_eff = dp[:, 1] - s * l_hip
            r_sag = torch.sqrt(dy_eff**2 + dp[:, 2]**2).clamp(min=1e-4)
            r_2l = torch.sqrt(dp[:, 0]**2 + r_sag**2).clamp(min=1e-3, max=l_thigh+l_calf-0.001)
            
            # 2. Geometry IK (Standard Knee Forward config)
            # Abduction
            q_abd = torch.atan2(dy_eff, -dp[:, 2])
            
            # Knee bend (Calf)
            cos_lo = ((l_thigh**2 + l_calf**2 - r_2l**2) / (2 * l_thigh * l_calf)).clamp(-1, 1)
            q_knee = torch.acos(cos_lo)
            q_lo = -(3.14159 - q_knee) # Straight=0, Bent=negative
            
            # Hip Pitch (Thigh)
            cos_up = ((r_2l**2 + l_thigh**2 - l_calf**2) / (2 * r_2l * l_thigh)).clamp(-1, 1)
            alpha = torch.acos(cos_up)
            gamma = torch.atan2(dp[:, 0], r_sag)
            
            # Standard Go2: +Pitch moves BACK. 
            # All legs unified to standard knee-forward geometry.
            # Calibration offsets will handle the rest.
            q_up = -(gamma + alpha)
            
            # 3. Apply Calibration Offsets
            q_final = torch.stack([q_abd, q_up, q_lo], dim=-1) + self._ik_offsets[i]
            
            joint_targets[:, idx[0]] = q_final[:, 0]
            joint_targets[:, idx[1]] = q_final[:, 1]
            joint_targets[:, idx[2]] = q_final[:, 2]
            
        return (joint_targets - self._default_joint_pos[:, :dof]) / self.action_scale

    def _skeleton_ik(self, p_des_b, contact, dof):
        # RobotSpec에서 전/후방 다리 길이 가져오기
        hip_b = torch.tensor(self.spec.hip_joint_offsets, device=self.device)
        joint_targets = self._default_joint_pos[:, :dof].clone()
        side_signs = [1.0, -1.0, 1.0, -1.0]

        for i in range(4):
            idx = self._joint_idx_map[i]
            if not idx: continue
            # 전후방 다리 길이 분리 적용
            l_up, l_lo, l_h = (self.spec.l_up_f, self.spec.l_lo_f, self.spec.l_h_f) if i < 2 else \
                              (self.spec.l_up_r, self.spec.l_lo_r, self.spec.l_h_r)
            
            dp = p_des_b[:, i, :] - hip_b[i].unsqueeze(0)
            dy_e = dp[:, 1] - side_signs[i]*l_h
            r_sag = torch.sqrt(dy_e**2 + dp[:, 2]**2).clamp(min=1e-4)
            r_2l = torch.sqrt(dp[:, 0]**2 + r_sag**2).clamp(min=1e-3, max=l_up + l_lo - 0.01)
            
            q_abd = torch.atan2(dy_e, -dp[:, 2])
            
            # Use unified Go2 IK logic for Skeleton (Standard Knee Forward)
            cos_lo = ((l_up**2 + l_lo**2 - r_2l**2) / (2 * l_up * l_lo)).clamp(-1, 1)
            q_knee = torch.acos(cos_lo)
            q_lo = -(3.14159 - q_knee)
            
            cos_up = ((r_2l**2 + l_up**2 - l_lo**2) / (2 * r_2l * l_up)).clamp(-1, 1)
            alpha = torch.acos(cos_up)
            gamma = torch.atan2(dp[:, 0], r_sag)
            q_up = -(gamma + alpha)
            
            # Apply Calibration Offsets
            q_final = torch.stack([q_abd, q_up, q_lo], dim=-1) + self._ik_offsets[i]
            
            joint_targets[:, idx[0]] = q_final[:, 0]
            # Skeleton uses different Pitch/Calf indices. 
            # Patterns: ["RL_hip", "RL_thigh", "RL_calf", ...]
            # Assuming idx[1] is Pitch, idx[2] is Knee/Calf as per manual check
            if len(idx) >= 3:
                joint_targets[:, idx[1]] = q_final[:, 1]
                joint_targets[:, idx[2]] = q_final[:, 2]
                
        return (joint_targets - self._default_joint_pos[:, :dof]) / self.action_scale

def quat_to_rotation_matrix(q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    R = torch.zeros(q.shape[0], 3, 3, device=q.device)
    R[:,0,0], R[:,0,1], R[:,0,2] = 1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)
    R[:,1,0], R[:,1,1], R[:,1,2] = 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)
    R[:,2,0], R[:,2,1], R[:,2,2] = 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    return R

def quat_to_yaw(q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return yaw

def yaw_to_rotation_matrix(yaw):
    N = yaw.shape[0]
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    R = torch.zeros(N, 3, 3, device=yaw.device)
    R[:, 0, 0], R[:, 0, 1] = cos_y, -sin_y
    R[:, 1, 0], R[:, 1, 1] = sin_y, cos_y
    R[:, 2, 2] = 1.0
    return R

def quat_rotate_vector(q, v):
    R = quat_to_rotation_matrix(q)
    if v.dim()==1: v = v.unsqueeze(0).expand(q.shape[0], -1)
    return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)

def quat_inverse(q):
    res = q.clone(); res[:, 1:] = -res[:, 1:]; return res
