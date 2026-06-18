# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""ZMQ 기반 IPC 브릿지 — Phase 2에서 ROS2Transport로 교체 가능."""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np

try:
    import zmq
except ImportError as e:
    raise ImportError("pyzmq가 설치되지 않았습니다. pip install pyzmq") from e


# ---------------------------------------------------------------------------
# 추상 인터페이스 (Phase 2에서 ROS2Transport로 교체)
# ---------------------------------------------------------------------------


@runtime_checkable
class TransportBridge(Protocol):
    def recv_setpoint(self, timeout_ms: int = 0) -> np.ndarray | None: ...
    def send_state(self, state: dict) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class ControllerTransport(Protocol):
    def send_setpoint(self, q: np.ndarray) -> None: ...
    def recv_state(self, timeout_ms: int = 0) -> dict | None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# ZMQ 구현 — Sim 측 (setpoint PULL + state PUB)
# ---------------------------------------------------------------------------


class ZMQSimBridge:
    """sim_runner.py 에서 사용하는 ZMQ 브릿지.

    setpoint: PULL + CONFLATE=1 (최신 setpoint만 유지)
    state:    PUB  + CONFLATE=1 (최신 state만 발행)
    """

    def __init__(
        self,
        setpoint_port: int = 5555,
        state_port: int = 5556,
    ) -> None:
        self._ctx = zmq.Context()

        # setpoint 수신 (PULL)
        self._setpoint_sock = self._ctx.socket(zmq.PULL)
        self._setpoint_sock.setsockopt(zmq.CONFLATE, 1)
        self._setpoint_sock.setsockopt(zmq.RCVTIMEO, 0)  # non-blocking
        self._setpoint_sock.bind(f"tcp://*:{setpoint_port}")

        # state 발행 (PUB) — CONFLATE는 SUB 측에서만 설정, PUB는 SNDHWM=1
        self._state_sock = self._ctx.socket(zmq.PUB)
        self._state_sock.setsockopt(zmq.SNDHWM, 1)
        self._state_sock.bind(f"tcp://*:{state_port}")

    def recv_setpoint(self, timeout_ms: int = 20) -> np.ndarray | None:
        """최신 관절 setpoint(rad)를 수신. 없으면 None 반환."""
        if not self._setpoint_sock.poll(timeout_ms):
            return None
        raw = self._setpoint_sock.recv(zmq.NOBLOCK)
        if len(raw) != 5 * 4:  # 5 joints × float32
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def send_state(self, state: dict) -> None:
        """관절 state를 JSON으로 발행."""
        payload = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()})
        self._state_sock.send_string(payload, zmq.NOBLOCK)

    def close(self) -> None:
        self._setpoint_sock.close()
        self._state_sock.close()
        self._ctx.term()


# ---------------------------------------------------------------------------
# ZMQ 구현 — Controller 측 (setpoint PUSH + state SUB)
# ---------------------------------------------------------------------------


class ZMQControllerBridge:
    """controller.py 에서 사용하는 ZMQ 브릿지.

    setpoint: PUSH (sim의 PULL로 전달)
    state:    SUB + CONFLATE=1 (최신 state만 수신)
    """

    def __init__(
        self,
        host: str = "localhost",
        setpoint_port: int = 5555,
        state_port: int = 5556,
    ) -> None:
        self._ctx = zmq.Context()

        # setpoint 전송 (PUSH)
        self._setpoint_sock = self._ctx.socket(zmq.PUSH)
        self._setpoint_sock.setsockopt(zmq.SNDHWM, 1)
        self._setpoint_sock.connect(f"tcp://{host}:{setpoint_port}")

        # state 수신 (SUB)
        self._state_sock = self._ctx.socket(zmq.SUB)
        self._state_sock.setsockopt(zmq.CONFLATE, 1)
        self._state_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._state_sock.connect(f"tcp://{host}:{state_port}")

    def send_setpoint(self, q: np.ndarray) -> None:
        """관절 setpoint(rad) 전송."""
        self._setpoint_sock.send(q.astype(np.float32).tobytes(), zmq.NOBLOCK)

    def recv_state(self, timeout_ms: int = 100) -> dict | None:
        """최신 state 수신. 없으면 None 반환."""
        if self._state_sock.poll(timeout_ms):
            raw = self._state_sock.recv_string()
            data = json.loads(raw)
            return {k: np.array(v, dtype=np.float32) if isinstance(v, list) else v for k, v in data.items()}
        return None

    def close(self) -> None:
        self._setpoint_sock.close()
        self._state_sock.close()
        self._ctx.term()
