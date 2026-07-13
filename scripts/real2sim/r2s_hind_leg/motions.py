# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-HindLeg 목표각 상수 + 보간/스텝/사인 유틸 (순수 함수, ROS/torch 비의존).

관절 순서는 CONTRACT.md §2 (thigh_r, thigh_p, knee_p, ankle_p, toe_p) 고정.
"""

from __future__ import annotations

import math

NUM_JOINTS: int = 5

# GUI/plot 레이블 (CONTRACT §2 순서). r2s_hind_leg_env_cfg.py JOINT_LABELS 와 일치.
JOINT_NAMES: list[str] = ["thigh_r", "thigh_p", "knee_p", "ankle_p", "toe_p"]

# 관절별 기본 PD 게인 — rga.py R_SKELETON_HIND_LEG_CFG(legs/feet)와 일치.
# ⚠ r2s_hind_leg_env_cfg.py DEFAULT_KP/KD 와 값 일치 필수(다른 패키지라 중복 정의).
DEFAULT_KP: list[float] = [300.0, 300.0, 300.0, 100.0, 100.0]
DEFAULT_KD: list[float] = [5.0, 5.0, 5.0, 5.0, 5.0]

# 실측 soft joint position limit [rad] (2026-07-13). GUI 명령을 이 범위로 클램프해
# sim의 silent 클램핑(soft_joint_pos_limit_factor=0.9)으로 인한 추종 혼란을 방지한다.
# ⚠ r2s_hind_leg_env_cfg.py SOFT_LIMITS_RAD 와 값 일치 필수.
SOFT_LIMITS_RAD: list[tuple[float, float]] = [
    (-2.826, 2.826),  # thigh_r
    (-1.413, 1.413),  # thigh_p
    (-1.413, 1.413),  # knee_p
    (-1.413, 1.413),  # ankle_p
    (-1.413, 1.413),  # toe_p
]

# 기본(중립) 자세 — 실측 default_joint_pos 전부 0.0.
DEFAULT_POSE: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]


def clamp_to_soft(pose: list[float]) -> list[float]:
    """각 관절을 SOFT_LIMITS_RAD 범위로 클램프."""
    return [min(SOFT_LIMITS_RAD[i][1], max(SOFT_LIMITS_RAD[i][0], pose[i])) for i in range(NUM_JOINTS)]


def lerp(a: list[float], b: list[float], alpha: float) -> list[float]:
    """두 관절각 벡터를 선형 보간 (alpha 0~1 클램프)."""
    t = min(1.0, max(0.0, alpha))
    return [a[i] + (b[i] - a[i]) * t for i in range(NUM_JOINTS)]


def interpolate_sequence(start: list[float], goal: list[float], num_steps: int) -> list[list[float]]:
    """start에서 goal까지 num_steps개 프레임으로 선형 보간(마지막이 goal)."""
    num_steps = max(1, num_steps)
    return [lerp(start, goal, (i + 1) / num_steps) for i in range(num_steps)]


def step_pose(current: list[float], joint_idx: int, delta: float) -> list[float]:
    """선택 관절만 delta[rad] 이동한 자세(soft limit 클램프)."""
    pose = list(current)
    pose[joint_idx] += delta
    return clamp_to_soft(pose)


def sine_offset(base: list[float], joint_idx: int, amplitude: float, frequency: float, t: float) -> list[float]:
    """base 자세에서 선택 관절에 사인 오프셋(soft limit 클램프)."""
    pose = list(base)
    pose[joint_idx] += amplitude * math.sin(2.0 * math.pi * frequency * t)
    return clamp_to_soft(pose)


if __name__ == "__main__":
    assert len(JOINT_NAMES) == NUM_JOINTS
    assert clamp_to_soft([9.0] * 5) == [SOFT_LIMITS_RAD[i][1] for i in range(5)]
    assert interpolate_sequence([0.0] * 5, [1.0] * 5, 10)[-1] == [1.0] * 5
    print("motions(hindleg) OK")
