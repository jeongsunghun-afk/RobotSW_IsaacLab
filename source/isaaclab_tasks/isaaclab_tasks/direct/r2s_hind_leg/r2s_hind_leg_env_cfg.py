# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""R_Skeleton Hind Leg Real2Sim 환경 설정."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.rga import R_SKELETON_HIND_LEG_CFG  # isort: skip

# ---------------------------------------------------------------------------
# 관절 파라미터 (rga.py 기준)
# ---------------------------------------------------------------------------

# USD 로드 후 articulation.data.joint_names로 실제 이름 확인 필요
# 패턴: .*_thigh_r, .*_thigh_p, .*_knee_p, .*_ankle_p, .*_toe_p
JOINT_NAME_PATTERNS: list[str] = [
    ".*_thigh_r",
    ".*_thigh_p",
    ".*_knee_p",
    ".*_ankle_p",
    ".*_toe_p",
]

# 슬라이더 표시용 레이블
JOINT_LABELS: list[str] = ["thigh_r", "thigh_p", "knee_p", "ankle_p", "toe_p"]

# 관절 각도 한계 (degree) — GUI 슬라이더 범위
JOINT_LIMITS_DEG: list[tuple[float, float]] = [
    (-60.0, 60.0),  # thigh_r
    (-90.0, 90.0),  # thigh_p
    (0.0, 120.0),  # knee_p
    (-45.0, 45.0),  # ankle_p
    (-30.0, 30.0),  # toe_p
]

# 관절 최대 속도 (rad/s) — slew rate limiter용
V_MAX_RAD: list[float] = [41.0, 25.0, 25.0, 51.0, 51.0]

# 최대 토크 (Nm) — 참조용 (ImplicitActuator가 자체 제한)
TAU_MAX_NM: list[float] = [22.0, 53.0, 53.0, 48.0, 48.0]

# ZMQ 포트
ZMQ_SETPOINT_PORT: int = 5555
ZMQ_STATE_PORT: int = 5556

NUM_JOINTS: int = 5


@configclass
class R2SHindLegEnvCfg(DirectRLEnvCfg):
    """R_Skeleton Hind Leg Real2Sim 테스트 환경.

    RL 없음. 순수 포지션 제어 테스트용.

    Observation (15-dim):
        joint_pos(5) + joint_vel(5) + applied_torque(5)

    Action (5-dim):
        joint position targets (rad)
    """

    # 에피소드 — 10분 (조기 종료 없음)
    episode_length_s: float = 600.0
    decimation: int = 4  # 200Hz physics / 50Hz control

    # 공간
    observation_space: int = 15  # 5 × (pos + vel + torque)
    action_space: int = NUM_JOINTS
    state_space: int = 0

    # 로봇
    robot: ArticulationCfg = R_SKELETON_HIND_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # 시뮬레이션
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation)
