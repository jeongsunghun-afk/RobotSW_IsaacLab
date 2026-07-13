# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R_Skeleton Hind Leg Real2Sim 환경 설정 (r2s_go2 패턴 기반).

RL 없음. 외부(sim_runner_hindleg.py)가 UDP로 받은 PD 목표를 set_setpoint()로 주입하면
slew rate limiter를 통해 안전하게 적용한다. 고정베이스 단일 뒷다리(5-DOF).

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_hind_leg/CONTRACT.md
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.rga import R_SKELETON_HIND_LEG_CFG  # isort: skip

# ---------------------------------------------------------------------------
# 관절 계약 (CONTRACT.md §2) — USD 로드 순서 독립, find_joints(preserve_order=True)로 고정.
# 실측 관절명(2026-07-13): HL_joint2_thigh_r, HL_joint3_thigh_p, HL_joint4_knee_p,
# HL_joint5_ankle_p, HL_joint6_toe_p.
# ---------------------------------------------------------------------------

NUM_JOINTS: int = 5

JOINT_NAME_PATTERNS: list[str] = [
    ".*_thigh_r",
    ".*_thigh_p",
    ".*_knee_p",
    ".*_ankle_p",
    ".*_toe_p",
]

# GUI/로그 표시용 짧은 레이블 (JOINT_NAME_PATTERNS와 동일 순서).
JOINT_LABELS: list[str] = ["thigh_r", "thigh_p", "knee_p", "ankle_p", "toe_p"]

# 관절별 PD 게인 — R_SKELETON_HIND_LEG_CFG(rga.py)의 legs/feet 액추에이터와 일치.
# GUI가 이 값을 기본으로 발행하고, faithful_pd=True면 슬라이더로 런타임 변경 가능.
DEFAULT_KP: list[float] = [300.0, 300.0, 300.0, 100.0, 100.0]
DEFAULT_KD: list[float] = [5.0, 5.0, 5.0, 5.0, 5.0]

# 관절 최대 속도 [rad/s] — slew rate limiter용 (rga.py THIGH/KNEE/ANKLE_VEL).
V_MAX_RAD: list[float] = [41.0, 25.0, 25.0, 51.0, 51.0]

# 실측 soft joint position limit [rad] (soft_joint_pos_limit_factor=0.9 반영, 2026-07-13).
# ⚠ scripts/real2sim/r2s_hind_leg/motions.py 의 SOFT_LIMITS_RAD 와 값 일치 필수(다른 패키지라 중복).
#   sim이 position target을 이 범위로 silently 클램프하므로 GUI 범위도 여기에 맞춘다.
SOFT_LIMITS_RAD: list[tuple[float, float]] = [
    (-2.826, 2.826),  # thigh_r
    (-1.413, 1.413),  # thigh_p
    (-1.413, 1.413),  # knee_p
    (-1.413, 1.413),  # ankle_p
    (-1.413, 1.413),  # toe_p
]

# 기본(중립) 자세 — 실측 default_joint_pos 전부 0.0.
DEFAULT_POSE: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]


@configclass
class R2SHindLegEnvCfg(DirectRLEnvCfg):
    """R_Skeleton Hind Leg Real2Sim 테스트 환경 (고정베이스 5-DOF).

    Observation (15-dim): joint_pos(5) + joint_vel(5) + applied_torque(5).
    Action (5-dim): 사용 안 함(setpoint은 set_setpoint()로 주입, action은 no-op).
    """

    # 에피소드 — 10분 (조기 종료 없음)
    episode_length_s: float = 600.0
    decimation: int = 4  # 200Hz physics / 50Hz control

    observation_space: int = 15  # 5 × (pos + vel + torque)
    action_space: int = NUM_JOINTS
    state_space: int = 0

    # faithful PD: True면 set_setpoint의 kp/kd를 write_joint_stiffness/damping_to_sim으로
    # 실제 반영(GUI 슬라이더가 살아있음). False면 cfg 액추에이터 PD 고정(r2s_go2 seam 방식).
    faithful_pd: bool = True

    # 로봇 — R_SKELETON_HIND_LEG_CFG는 고정베이스(USD Fixed_Filpped), init z=0.6.
    robot: ArticulationCfg = R_SKELETON_HIND_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation)
