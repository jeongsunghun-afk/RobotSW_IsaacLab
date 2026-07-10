# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 GUI 모션 정의 (관절각/시퀀스/보간).

순수 함수/상수만 담는다 — ros/rclpy 의존 없음. 조인트 순서는
CONTRACT.md §2(Unitree 표준 idx0..11)를 그대로 따른다.

- idx 0: FR_hip, 1: FR_thigh, 2: FR_calf
- idx 3: FL_hip, 4: FL_thigh, 5: FL_calf
- idx 6: RR_hip, 7: RR_thigh, 8: RR_calf
- idx 9: RL_hip, 10: RL_thigh, 11: RL_calf
"""

from __future__ import annotations

import math

NUM_MOTORS: int = 12

# CONTRACT.md §6 / unitree.py UNITREE_GO2_CFG base_legs 액추에이터와 일치.
DEFAULT_KP: float = 25.0
DEFAULT_KD: float = 0.5

JOINT_NAMES: list[str] = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# 기본 자세(선 자세) — GUI 지시서 §"기본자세" 목표각.
DEFAULT_POSE: list[float] = [
    -0.1,
    0.8,
    -1.5,  # FR: hip, thigh, calf
    0.1,
    0.8,
    -1.5,  # FL
    -0.1,
    1.0,
    -1.5,  # RR
    0.1,
    1.0,
    -1.5,  # RL
]

# 앉기(웅크린) 자세 — thigh를 더 굽히고 calf를 더 접는다.
SIT_POSE: list[float] = [
    -0.1,
    1.35,
    -2.6,  # FR
    0.1,
    1.35,
    -2.6,  # FL
    -0.1,
    1.55,
    -2.6,  # RR
    0.1,
    1.55,
    -2.6,  # RL
]

# go2_stand_example(unitree_ros2) 기립 궤적 자세 — Unitree idx 순서(JOINT_NAMES와 동일).
# ⚠ r2s_go2_env_cfg.py 의 _STAND_FOLDED(prone 초기자세)와 값이 반드시 일치해야 함(다른 패키지라 중복 정의).
# calf 는 example 원값 -2.65 대신 -2.6 사용 — -2.65 는 GO2 calf soft limit(≈-2.628)를 벗어나
# sim 이 position target 을 클램프하므로 soft limit 안쪽 값으로 완화(r2s_go2_env_cfg.py 참조).
STAND_FOLDED: list[float] = [
    0.0,
    1.36,
    -2.6,
    0.0,
    1.36,
    -2.6,
    -0.2,
    1.36,
    -2.6,
    0.2,
    1.36,
    -2.6,
]  # 엎드림(target_pos_1)
STAND_UP: list[float] = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3, 0.0, 0.67, -1.3]  # 기립(target_pos_2)


def lerp(a: list[float], b: list[float], alpha: float) -> list[float]:
    """두 관절각 벡터를 선형 보간.

    Args:
        a: 시작 관절각, 길이 12.
        b: 목표 관절각, 길이 12.
        alpha: 보간 비율, 0.0(a)~1.0(b)로 클램프.

    Returns:
        보간된 관절각, 길이 12.
    """
    t = min(1.0, max(0.0, alpha))
    return [a[i] + (b[i] - a[i]) * t for i in range(NUM_MOTORS)]


def interpolate_sequence(start: list[float], goal: list[float], num_steps: int) -> list[list[float]]:
    """start에서 goal까지 num_steps개 프레임으로 선형 보간한 시퀀스를 생성.

    Args:
        start: 시작 관절각, 길이 12.
        goal: 목표 관절각, 길이 12.
        num_steps: 생성할 프레임 수(마지막 프레임이 goal과 일치).

    Returns:
        길이 num_steps의 관절각 리스트(각 원소는 길이 12).
    """
    n = max(1, num_steps)
    return [lerp(start, goal, (i + 1) / n) for i in range(n)]


def step_pose(current: list[float], joint_idx: int, delta: float) -> list[float]:
    """current에서 joint_idx 관절만 delta만큼 이동한 새 관절각을 반환.

    Args:
        current: 현재 관절각, 길이 12.
        joint_idx: 스텝을 적용할 관절 인덱스(0..11).
        delta: 목표각 변화량 [rad].

    Returns:
        수정된 관절각, 길이 12.
    """
    pose = list(current)
    pose[joint_idx] = pose[joint_idx] + delta
    return pose


def sine_offset(base: list[float], joint_idx: int, amplitude: float, frequency: float, t: float) -> list[float]:
    """base 자세에서 joint_idx 관절에 사인파를 주입한 관절각을 반환.

    Args:
        base: 기준 관절각(보통 DEFAULT_POSE), 길이 12.
        joint_idx: 사인파를 주입할 관절 인덱스(0..11).
        amplitude: 사인파 진폭 [rad].
        frequency: 사인파 주파수 [Hz].
        t: 경과 시간 [s].

    Returns:
        수정된 관절각, 길이 12.
    """
    pose = list(base)
    pose[joint_idx] = pose[joint_idx] + amplitude * math.sin(2.0 * math.pi * frequency * t)
    return pose


def stand_up_sequence(
    start: list[float],
    frame_hz: float,
    fold_s: float = 1.0,
    rise_s: float = 1.0,
    hold_s: float = 1.0,
) -> list[list[float]]:
    """go2_stand_example 방식의 기립 시퀀스를 생성한다: start→FOLDED→UP→(UP 홀드).

    unitree_ros2 go2_stand_example 의 다단계 궤적을 그대로 따른다(마지막 splay 자세는 제외 —
    "일어서기" 버튼이므로 선 자세로 끝낸다).

    Args:
        start: 현재 관절각, 길이 12.
        frame_hz: 프레임 생성 주파수 [Hz].
        fold_s: start → STAND_FOLDED(엎드림) 보간 시간 [s].
        rise_s: STAND_FOLDED → STAND_UP(기립) 보간 시간 [s].
        hold_s: 기립 자세 유지 시간 [s].

    Returns:
        관절각 프레임 리스트(각 원소 길이 12). 마지막 프레임은 STAND_UP.
    """
    seq: list[list[float]] = []
    seq += interpolate_sequence(start, STAND_FOLDED, max(1, int(fold_s * frame_hz)))
    seq += interpolate_sequence(STAND_FOLDED, STAND_UP, max(1, int(rise_s * frame_hz)))
    seq += [list(STAND_UP) for _ in range(max(1, int(hold_s * frame_hz)))]
    return seq
