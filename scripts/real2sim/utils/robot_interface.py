"""실제 로봇 통신용 ROS2 인터페이스 skeleton.

현재 미구현 상태. 향후 실제 로봇 도입 시 여기에 ROS2 노드 추가.

**사용 방식:**
1. controller.py (Python 3.10 환경)에서 로봇 인터페이스 임포트
2. 현재: ZMQ만 사용
3. 향후: ZMQ + ROS2 동시 운영 (같은 setpoint 이중 송신)

**확장 순서:**
Phase 2: ROS2 service/topic으로 실제 로봇과 통신
- JointCommand service (setpoint 전송)
- JointState subscription (state 수신)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class RobotInterface(ABC):
    """로봇 통신 인터페이스 (ZMQ와 동일한 프로토콜)."""

    @abstractmethod
    def send_setpoint(self, q: np.ndarray) -> None:
        """관절 명령 전송.

        Args:
            q: 관절 각도 (rad), shape (5,)
        """

    @abstractmethod
    def recv_state(self, timeout_ms: int = 0) -> Optional[dict]:
        """로봇 상태 수신.

        Returns:
            {"pos": [5], "vel": [5], "torque": [5], "timestamp": float}
            또는 None (타임아웃)
        """

    @abstractmethod
    def close(self) -> None:
        """연결 종료."""


class MockRobotInterface(RobotInterface):
    """Skeleton: 실제 로봇 없이 dummy state 반환.

    현재 상태: 실제 로봇이 없으므로 더미 구현만 제공.
    향후 ROS2 노드로 교체.
    """

    def __init__(self):
        """Mock 로봇 초기화."""
        self._setpoint = np.zeros(5, dtype=np.float32)
        self._step = 0

    def send_setpoint(self, q: np.ndarray) -> None:
        """Dummy: setpoint 저장만 함."""
        if len(q) == 5:
            self._setpoint = q.copy()

    def recv_state(self, timeout_ms: int = 0) -> Optional[dict]:
        """Dummy: 미리 정해진 상태 반환 (실제 로봇 움직임 없음)."""
        import time

        state = {
            "pos": self._setpoint.copy().tolist(),  # setpoint를 그대로 반환
            "vel": np.zeros(5, dtype=np.float32).tolist(),
            "torque": np.zeros(5, dtype=np.float32).tolist(),
            "timestamp": time.time(),
        }
        return state

    def close(self) -> None:
        """Dummy: 특별한 정리 필요 없음."""
        pass


class ROS2RobotInterface(RobotInterface):
    """Phase 2: 실제 ROS2 로봇 인터페이스 (구현 대기).

    **사전 조건:**
    - Python 3.10 conda 환경 (isaac-r2s-py310)
    - ROS2 Humble 설치
    - real2sim_msgs 패키지 빌드됨

    **구현 예정:**
    ```python
    import rclpy
    from rclpy.node import Node
    from real2sim_msgs.msg import JointCommand, JointState
    from real2sim_msgs.srv import GetJointState

    class ROS2RobotBridge(Node):
        def __init__(self):
            super().__init__('robot_bridge')

            # Setpoint 발행
            self.cmd_pub = self.create_publisher(
                JointCommand, 'robot/joint_command', qos_profile_sensor_data)

            # State 구독
            self.state_sub = self.create_subscription(
                JointState, 'robot/joint_state', self._on_state, qos_profile_sensor_data)

            self._latest_state = None
    ```
    """

    def __init__(self):
        """ROS2 로봇 인터페이스 초기화 (미구현)."""
        raise NotImplementedError(
            "ROS2RobotInterface는 Phase 2에서 구현됩니다. "
            "현재는 ZMQ 또는 MockRobotInterface를 사용하세요."
        )

    def send_setpoint(self, q: np.ndarray) -> None:
        """ROS2 service로 setpoint 전송 (미구현)."""
        raise NotImplementedError()

    def recv_state(self, timeout_ms: int = 0) -> Optional[dict]:
        """ROS2 subscription에서 state 수신 (미구현)."""
        raise NotImplementedError()

    def close(self) -> None:
        """ROS2 노드 종료 (미구현)."""
        raise NotImplementedError()


class DualTransportBridge:
    """ZMQ + 로봇 인터페이스 동시 운영.

    향후 Phase 2에서 사용:
    - ZMQ: 시뮬레이션과 통신 (항상 활성)
    - Robot: 실제 로봇과 통신 (선택적)

    같은 setpoint를 두 채널 모두에 송신하여 동시 검증 가능.
    """

    def __init__(self, zmq_bridge, robot_interface: Optional[RobotInterface] = None):
        """
        Args:
            zmq_bridge: ZMQ 브릿지 인스턴스
            robot_interface: 로봇 인터페이스 (선택사항)
        """
        self.zmq_bridge = zmq_bridge
        self.robot_interface = robot_interface

    def send_setpoint(self, q: np.ndarray) -> None:
        """ZMQ 및 로봇 인터페이스 모두에 setpoint 송신."""
        # ZMQ: 시뮬레이션
        self.zmq_bridge.send_setpoint(q)

        # Robot: 실제 로봇 (있으면)
        if self.robot_interface:
            try:
                self.robot_interface.send_setpoint(q)
            except Exception as e:
                print(f"[Robot] Send failed: {e}")

    def recv_state(self, source: str = "zmq", timeout_ms: int = 0) -> Optional[dict]:
        """지정된 소스에서 state 수신.

        Args:
            source: "zmq" (시뮬레이션) 또는 "robot" (실제 로봇)
            timeout_ms: 수신 타임아웃 (ms)

        Returns:
            state dict 또는 None
        """
        if source == "zmq":
            return self.zmq_bridge.recv_state(timeout_ms)
        elif source == "robot" and self.robot_interface:
            return self.robot_interface.recv_state(timeout_ms)
        return None

    def close(self) -> None:
        """양쪽 인터페이스 종료."""
        self.zmq_bridge.close()
        if self.robot_interface:
            self.robot_interface.close()


# ============================================================================
# 사용 예제 (현재: 미활성화)
# ============================================================================
if __name__ == "__main__":
    # Phase 1: ZMQ만 사용
    # from zmq_bridge import ZMQControllerBridge
    # bridge = ZMQControllerBridge()
    # bridge.send_setpoint(q)
    # state = bridge.recv_state()

    # Phase 2 (향후): ZMQ + 실제 로봇
    # from zmq_bridge import ZMQControllerBridge
    # zmq_bridge = ZMQControllerBridge()
    # robot_bridge = ROS2RobotInterface()  # ROS2 노드
    # dual_bridge = DualTransportBridge(zmq_bridge, robot_bridge)
    #
    # while True:
    #     q = get_command_from_gui()
    #     dual_bridge.send_setpoint(q)
    #
    #     sim_state = dual_bridge.recv_state("zmq")
    #     robot_state = dual_bridge.recv_state("robot")
    #
    #     # 시뮬레이션과 로봇 동시 모니터링
    #     compare_states(sim_state, robot_state)

    print("✓ Robot Interface Skeleton Ready")
    print("  Phase 1: ZMQ only (current)")
    print("  Phase 2: ZMQ + ROS2 dual transport (future)")
