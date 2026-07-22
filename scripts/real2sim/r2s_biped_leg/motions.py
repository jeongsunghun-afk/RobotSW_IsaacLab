# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg 목표각 상수 + 보간/스텝/사인 유틸 (순수 함수, ROS/torch 비의존).

관절 순서는 **leg-major** 고정 — HL 4개(hip, thigh, calf, foot) → HR 4개.
(`find_joints(preserve_order=True)`로 실측한 순서.)
"""

from __future__ import annotations

import math

NUM_JOINTS: int = 8

# GUI/plot 레이블 (leg-major 순서).
# ⚠ r2s_biped_leg_env_cfg.py JOINT_LABELS 와 값 일치 필수(다른 패키지라 중복 정의).
JOINT_NAMES: list[str] = [
    "HL_hip",
    "HL_thigh",
    "HL_calf",
    "HL_foot",
    "HR_hip",
    "HR_thigh",
    "HR_calf",
    "HR_foot",
]

# 관절별 기본 PD 게인 — HIND_LEG_CFG 실측값(2026-07-21 probe).
# go2/R_Skeleton(300/5)과 달리 12~65로 훨씬 낮다. GUI 슬라이더 상한도 여기에 맞춰져 있다.
# ⚠ source/isaaclab_tasks/isaaclab_tasks/direct/r2s_biped_leg/r2s_biped_leg_env_cfg.py
#    DEFAULT_KP/DEFAULT_KD 와 값 일치 필수.
DEFAULT_KP: list[float] = [65.0, 53.0, 12.0, 20.0, 65.0, 53.0, 12.0, 20.0]
DEFAULT_KD: list[float] = [6.0, 4.8, 1.1, 1.0, 6.0, 4.8, 1.1, 1.0]

# 실측 soft joint position limit [rad] (2026-07-21 probe). GUI 명령을 이 범위로 클램프해
# sim의 silent 클램핑(soft_joint_pos_limit_factor)으로 인한 추종 혼란을 방지한다.
# ⚠ r2s_biped_leg_env_cfg.py SOFT_LIMITS_RAD 와 값 일치 필수.
# ⚠ foot 관절은 좌우가 **미러**라 비대칭이다(오타 아님). 대칭이라 가정하지 말 것.
SOFT_LIMITS_RAD: list[tuple[float, float]] = [
    (-0.5498, 0.5498),  # HL_hip
    (-1.9024, 1.5533),  # HL_thigh
    (-1.3836, 0.4236),  # HL_calf
    (-0.4192, 1.4662),  # HL_foot
    (-0.5498, 0.5498),  # HR_hip
    (-1.9024, 1.5533),  # HR_thigh
    (-1.3836, 0.4236),  # HR_calf
    (-1.4662, 0.4192),  # HR_foot  (HL_foot의 미러 — 의도된 비대칭)
]

# 기본(중립) 자세 — 실측 default_joint_pos 전부 0.0.
DEFAULT_POSE: list[float] = [0.0] * NUM_JOINTS


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
    assert len(DEFAULT_KP) == NUM_JOINTS
    assert len(DEFAULT_KD) == NUM_JOINTS
    assert len(SOFT_LIMITS_RAD) == NUM_JOINTS
    assert len(DEFAULT_POSE) == NUM_JOINTS
    assert clamp_to_soft([9.0] * NUM_JOINTS) == [SOFT_LIMITS_RAD[i][1] for i in range(NUM_JOINTS)]
    assert clamp_to_soft([-9.0] * NUM_JOINTS) == [SOFT_LIMITS_RAD[i][0] for i in range(NUM_JOINTS)]
    assert interpolate_sequence([0.0] * NUM_JOINTS, [0.4] * NUM_JOINTS, 10)[-1] == [0.4] * NUM_JOINTS
    # foot 좌우 비대칭(미러)이 보존되는지 — 대칭으로 "고쳐지면" 여기서 걸린다.
    assert SOFT_LIMITS_RAD[3] == (-SOFT_LIMITS_RAD[7][1], -SOFT_LIMITS_RAD[7][0])
    print("motions(bipedleg) OK")
