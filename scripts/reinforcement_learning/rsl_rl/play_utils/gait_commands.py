"""
gait_commands.py
================
Go2WTW 보행 파라미터 딕셔너리 및 14차원 커맨드 벡터 빌더.

지원 보행 목록:
    pronking, trotting, bounding, pacing, galloping,
    walking, ambling, cantering, half-bounding, gallop_rot

빌드되는 커맨드 인덱스 (Go2WTW 기준, 14개):
    [0] lin_vel_x      [1] lin_vel_y      [2] ang_vel(yaw)
    [3] body_height    [4] gait_frequency
    [5] gait_phase     [6] gait_offset    [7] gait_bound
    [8] gait_duration  [9] footswing_height
    [10] body_pitch    [11] body_roll
    [12] stance_width  [13] stance_length
"""

from __future__ import annotations

# ── 보행별 파라미터 딕셔너리 ───────────────────────────────────────────

# gait_phase, gait_offset, gait_bound
GAITS: dict[str, list[float]] = {
    "pronking":     [0,      0,    0    ],
    "trotting":     [0.5,    0,    0    ],
    "bounding":     [0,      0.5,  0    ],
    "pacing":       [0,      0,    0.5  ],
    "galloping":    [0.25,   0.,   0.   ],
    "walking":      [0.,     0.25, 0.5  ],
    "ambling":      [0.,     0.25, 0.5  ],
    "cantering":    [0.0,    0.3,  0.3  ],
    "half-bounding":[0.,     0.25, 0.   ],
    "gallop_rot":   [0.4646, 0.0,  0.7677],
}

STEP_FREQUENCY: dict[str, float] = {
    "pronking":     2.0,
    "trotting":     1.5,
    "bounding":     2.0,
    "pacing":       2.0,
    "galloping":    3.0,
    "walking":      1.5,
    "ambling":      2.0,
    "cantering":    2.0,
    "half-bounding":2.0,
    "gallop_rot":   3.0,
}

FOOTSWING_HEIGHT: dict[str, float] = {
    "pronking":     0.2,
    "trotting":     0.1,
    "bounding":     0.2,
    "pacing":       0.1,
    "galloping":    0.1,
    "walking":      0.1,
    "ambling":      0.1,
    "cantering":    0.15,
    "half-bounding":0.15,
    "gallop_rot":   0.1,
}

DURATIONS: dict[str, float] = {
    "pronking":     0.5,
    "trotting":     0.5,
    "bounding":     0.5,
    "pacing":       0.5,
    "galloping":    0.5,
    "walking":      0.5,
    "ambling":      0.5,
    "cantering":    0.5,
    "half-bounding":0.5,
    "gallop_rot":   0.5,
}

STANCE_WIDTHS: dict[str, float] = {
    "pronking":     0.25,
    "trotting":     0.25,
    "bounding":     0.25,
    "pacing":       0.26,
    "galloping":    0.25,
    "walking":      0.25,
    "ambling":      0.25,
    "cantering":    0.25,
    "half-bounding":0.25,
    "gallop_rot":   0.25,
}

STANCE_LENGTHS: dict[str, float] = {
    "pronking":     0.45,
    "trotting":     0.45,
    "bounding":     0.45,
    "pacing":       0.45,
    "galloping":    0.45,
    "walking":      0.45,
    "ambling":      0.45,
    "cantering":    0.45,
    "half-bounding":0.45,
    "gallop_rot":   0.45,
}

# 지원 보행 이름 목록
GAIT_NAMES: list[str] = sorted(GAITS.keys())


def build_go2wtw_command(
    gait: str,
    x_vel: float,
    y_vel: float,
    yaw_vel: float,
    body_height: float = 0.0,
    body_pitch: float = 0.0,
    body_roll: float = 0.0,
) -> list[float]:
    """Go2WTW 환경용 14차원 커맨드 벡터를 빌드합니다.

    Parameters
    ----------
    gait : str
        GAIT_NAMES 중 하나.
    x_vel, y_vel, yaw_vel : float
        선속도(전후/좌우) 및 yaw 각속도.
    body_height, body_pitch, body_roll : float
        선택적 자세 파라미터.

    Returns
    -------
    list[float]
        길이 14의 커맨드 벡터.
    """
    if gait not in GAITS:
        raise ValueError(
            f"알 수 없는 보행: '{gait}'. "
            f"사용 가능: {GAIT_NAMES}"
        )
    g = GAITS[gait]
    return [
        x_vel,                   # 0: lin_vel_x
        y_vel,                   # 1: lin_vel_y
        yaw_vel,                 # 2: ang_vel
        body_height,             # 3: body_height
        STEP_FREQUENCY[gait],    # 4: gait_frequency
        g[0],                    # 5: gait_phase
        g[1],                    # 6: gait_offset
        g[2],                    # 7: gait_bound
        DURATIONS[gait],         # 8: gait_duration
        FOOTSWING_HEIGHT[gait],  # 9: footswing_height
        body_pitch,              # 10: body_pitch
        body_roll,               # 11: body_roll
        STANCE_WIDTHS[gait],     # 12: stance_width
        STANCE_LENGTHS[gait],    # 13: stance_length
    ]


def build_simple_command(
    x_vel: float,
    y_vel: float,
    yaw_vel: float,
) -> list[float]:
    """R_Skeleton 등 3차원 커맨드 환경용 벡터를 빌드합니다."""
    return [x_vel, y_vel, yaw_vel]
