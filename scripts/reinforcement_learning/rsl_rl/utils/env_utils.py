"""
env_utils.py
===========
환경(env) 관련 헬퍼 함수들.
- 액션 인덱스 → 관절 이름 매핑 출력
- 환경에서 커맨드 정보(레이블, 범위) 자동 감지
"""

from __future__ import annotations

import gymnasium as gym


def print_action_joint_mapping(env) -> None:
    """액션 인덱스 → 관절 이름 매핑을 출력합니다.

    Articulation 로봇을 사용하는 환경에서만 동작합니다.
    """
    base_env = getattr(env, "unwrapped", env)
    robot = getattr(base_env, "_robot", None)
    if robot is None or not hasattr(robot, "data"):
        print("[env_utils] WARN: 로봇 인스턴스를 찾을 수 없어 액션 매핑 출력을 건너뜁니다.")
        return

    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is None:
        print("[env_utils] WARN: 관절 이름을 가져올 수 없습니다.")
        return

    action_dim = None
    if hasattr(base_env, "single_action_space"):
        action_dim = gym.spaces.flatdim(base_env.single_action_space)

    print("[INFO] 액션 인덱스 → 관절 이름 매핑 (Articulation 관절 순서):")
    for idx, name in enumerate(joint_names):
        print(f"  {idx:02d}: {name}")

    if action_dim is not None and action_dim != len(joint_names):
        print(
            f"[env_utils] WARN: 액션 차원({action_dim})과 관절 수({len(joint_names)})가"
            " 일치하지 않습니다."
        )


def get_env_command_info(env) -> tuple[list[str], list[tuple[float, float]]]:
    """환경에서 커맨드 레이블과 (min, max) 범위를 자동으로 추출합니다.

    우선순위:
    1. `env.unwrapped.cfg.command_cfg` 딕셔너리 (Go2WTW, R_Skeleton 등)
    2. `env.unwrapped._commands` 텐서 크기 (fallback: cmd_0 ~ cmd_N)

    Returns
    -------
    labels : list[str]
        각 커맨드에 대한 사람이 읽기 좋은 레이블.
    ranges : list[tuple[float, float]]
        각 커맨드의 (min, max) 범위. 범위를 알 수 없으면 (-1.0, 1.0) 기본값.
    """
    base_env = getattr(env, "unwrapped", env)
    cfg = getattr(base_env, "cfg", None)
    command_cfg: dict | None = getattr(cfg, "command_cfg", None) if cfg is not None else None

    # command_cfg가 있으면 키 순서대로 레이블/범위 추출
    if command_cfg and isinstance(command_cfg, dict):
        labels: list[str] = []
        ranges: list[tuple[float, float]] = []
        for key, val in command_cfg.items():
            # 사람이 읽기 좋은 레이블: 접미사 _range/_cmd_range 제거
            label = key.replace("_cmd_range", "").replace("_range", "")
            labels.append(label)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                lo, hi = float(val[0]), float(val[1])
            else:
                lo, hi = -1.0, 1.0
            ranges.append((lo, hi))
        return labels, ranges

    # fallback: _commands 텐서 크기 사용
    commands_tensor = getattr(base_env, "_commands", None)
    if commands_tensor is not None:
        num_commands = commands_tensor.shape[-1]
        labels = [f"cmd_{i}" for i in range(num_commands)]
        ranges = [(-1.0, 1.0)] * num_commands
        return labels, ranges

    # 환경에 커맨드 정보가 전혀 없는 경우
    return [], []
