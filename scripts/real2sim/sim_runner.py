# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real2Sim 시뮬레이션 러너.

R_Skeleton Hind Leg를 IsaacLab에 올리고 ZMQ로 외부 컨트롤러와 통신.
ROS2는 controller.py에서 별도 conda 환경으로 사용.

실행:
  ./isaaclab.sh -p scripts/real2sim/sim_runner.py --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="R2S Hind Leg 시뮬레이션 러너")
parser.add_argument("--num_envs", type=int, default=1, help="환경 수 (기본값: 1)")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Fabric 비활성화")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import atexit
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks.direct.r2s_hind_leg  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# 로컬 유틸 (scripts/real2sim/ 기준 실행)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_UTILS_DIR = _SCRIPT_DIR / "utils"
sys.path.insert(0, str(_UTILS_DIR))
from data_logger import DataLogger  # noqa: E402
from zmq_bridge import ZMQSimBridge  # noqa: E402

# 환경 클래스 (타입 어노테이션용)
from isaaclab_tasks.direct.r2s_hind_leg.r2s_hind_leg_env import R2SHindLegEnv  # noqa: E402

# ---------------------------------------------------------------------------
# 공유 버퍼 (메인 스레드 ↔ 폴링 스레드)
# ---------------------------------------------------------------------------
_NUM_JOINTS = 5
_shared_setpoint = np.zeros(_NUM_JOINTS, dtype=np.float32)
_setpoint_lock = threading.Lock()
_new_setpoint = threading.Event()
_stop_event = threading.Event()  # 폴링 스레드 종료 신호
_poll_thread_ref: threading.Thread | None = None


def _poll_thread(bridge: "ZMQSimBridge") -> None:
    """setpoint 폴링 스레드 — sim 루프와 독립 (ZMQ)."""
    while not _stop_event.is_set():
        q = bridge.recv_setpoint(timeout_ms=20)
        if q is not None and len(q) == _NUM_JOINTS:
            with _setpoint_lock:
                _shared_setpoint[:] = q
            _new_setpoint.set()


def _shutdown(r2s_env: R2SHindLegEnv, bridge: "ZMQSimBridge", logger: DataLogger) -> None:
    # 폴링 스레드 먼저 종료 후 socket close
    _stop_event.set()
    if _poll_thread_ref is not None:
        _poll_thread_ref.join(timeout=1.0)
    path = logger.save()
    if path:
        print(f"[R2S] 데이터 저장 완료: {path}")
    bridge.close()
    r2s_env.close()


def main() -> None:
    global _running

    # 환경 설정 로드
    env_cfg = parse_env_cfg(
        "Isaac-R2S-HindLeg-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env: gym.Env = gym.make("Isaac-R2S-HindLeg-v0", cfg=env_cfg, render_mode="rgb_array")
    r2s_env: R2SHindLegEnv = env.unwrapped  # type: ignore[assignment]

    # ZMQ 브릿지 + 데이터 로거 초기화
    bridge = ZMQSimBridge()
    logger = DataLogger()
    atexit.register(_shutdown, r2s_env, bridge, logger)

    # 폴링 스레드 시작 (ZMQ/ROS2 공용)
    global _poll_thread_ref
    poll_thread = threading.Thread(target=_poll_thread, args=(bridge,), daemon=True)
    poll_thread.start()
    _poll_thread_ref = poll_thread

    print("[R2S] 시뮬레이션 시작. 컨트롤러: python scripts/real2sim/controller.py")

    # 환경 초기화
    env.reset()
    last_setpoint = np.zeros(_NUM_JOINTS, dtype=np.float32)

    # 메인 시뮬레이션 루프
    while simulation_app.is_running():
        # 최신 setpoint 읽기
        if _new_setpoint.is_set():
            with _setpoint_lock:
                last_setpoint = _shared_setpoint.copy()
            # TEMP: HL_joint2_thigh_r URDF axis가 -1 0 0이라 모션 데이터(v10 +X convention)와 회전 부호가 반대.
            # 정식 fix는 URDF axis +1 0 0 + USD 재변환. 그 전까진 thigh_r 채널만 부호 반전.
            # last_setpoint[0] = -last_setpoint[0]
            _new_setpoint.clear()

        # setpoint 주입 (환경이 slew rate 적용)
        q_tensor = torch.from_numpy(last_setpoint).to(r2s_env.device)
        r2s_env.set_setpoint(q_tensor)

        # 시뮬레이션 step (action은 무시, setpoint으로 제어)
        zero_action = torch.zeros(1, _NUM_JOINTS, device=r2s_env.device)
        _, _, terminated, truncated, _ = env.step(zero_action)

        # state 발행 + 로깅
        state = r2s_env.get_state()
        bridge.send_state(state)
        applied = r2s_env.get_applied_setpoint().cpu().numpy()
        logger.log(state, setpoint=applied)

        # 타임아웃 리셋
        done = terminated if isinstance(terminated, bool) else bool(terminated.any())
        tout = truncated if isinstance(truncated, bool) else bool(truncated.any())
        if done or tout:
            env.reset()

    _stop_event.set()
    path = logger.save()
    if path:
        print(f"[R2S] 데이터 저장: {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
