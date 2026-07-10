#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 sim_bridge — `/lowcmd` <-> UDP <-> `/lowstate` 브릿지 노드.

시스템 Python 3.10 + ROS2 humble + unitree_go overlay 로 실행한다 (CONTRACT.md §1):

    source /opt/ros/humble/setup.bash
    source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
    /usr/bin/python3 scripts/real2sim/r2s_go2/sim_bridge.py

역할:
    1. `/lowcmd`(unitree_go/msg/LowCmd) 구독 -> motor_cmd[0..11] 추출 -> UDP로 sim_runner에 전송.
    2. UDP로 sim_runner가 보낸 상태 패킷 수신(별도 스레드, latest-wins) -> `/lowstate` 발행(50Hz).

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md
"""

from __future__ import annotations

import math
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
import r2s_udp  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from unitree_go.msg import LowCmd, LowState  # noqa: E402

STATE_PUBLISH_HZ: float = 50.0
UDP_HOST: str = "127.0.0.1"

# 토픽명(환경변수로 오버라이드 가능). 동시(sim+real) 구동 시 sim 상태는 실로봇 /lowstate 와
# 충돌하지 않도록 /sim/lowstate 로 remap 한다. 명령 /lowcmd 는 remap 하지 않는다 —
# sim 과 실로봇이 같은 명령을 받는 것이 동시 구동의 목적이므로 공유해야 한다.
CMD_TOPIC: str = os.environ.get("R2S_CMD_TOPIC", "/lowcmd")
STATE_TOPIC: str = os.environ.get("R2S_STATE_TOPIC", "/lowstate")


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """쿼터니언(w,x,y,z)을 roll/pitch/yaw [rad]로 변환."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = min(1.0, max(-1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class SimBridge(Node):
    """`/lowcmd`를 UDP로 sim_runner에 전달하고, UDP 상태를 `/lowstate`로 발행."""

    def __init__(self) -> None:
        super().__init__("r2s_sim_bridge")

        self._cmd_seq: int = 0
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_addr = (UDP_HOST, r2s_udp.CMD_PORT)

        self._state_lock = threading.Lock()
        self._latest_state: dict | None = None

        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind((UDP_HOST, r2s_udp.STATE_PORT))
        self._state_sock.settimeout(1.0)
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        self._lowcmd_sub = self.create_subscription(LowCmd, CMD_TOPIC, self._on_lowcmd, 10)
        self._lowstate_pub = self.create_publisher(LowState, STATE_TOPIC, 10)
        self._publish_timer = self.create_timer(1.0 / STATE_PUBLISH_HZ, self._publish_lowstate)

        self.get_logger().info(
            f"r2s_sim_bridge ready: {CMD_TOPIC} -> UDP:{r2s_udp.CMD_PORT}, UDP:{r2s_udp.STATE_PORT} -> {STATE_TOPIC}"
        )

    def _recv_loop(self) -> None:
        """UDP 상태 패킷 수신 루프(별도 스레드). 최신 패킷만 보관(latest-wins)."""
        while rclpy.ok():
            try:
                data, _ = self._state_sock.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            parsed = r2s_udp.unpack_state(data)
            if parsed is None:
                continue
            with self._state_lock:
                self._latest_state = parsed

    def _on_lowcmd(self, msg: LowCmd) -> None:
        """`/lowcmd` 콜백: motor_cmd[0..11]을 UDP 명령 패킷으로 직렬화해 전송."""
        q = [msg.motor_cmd[i].q for i in range(r2s_udp.NUM_MOTORS)]
        dq = [msg.motor_cmd[i].dq for i in range(r2s_udp.NUM_MOTORS)]
        kp = [msg.motor_cmd[i].kp for i in range(r2s_udp.NUM_MOTORS)]
        kd = [msg.motor_cmd[i].kd for i in range(r2s_udp.NUM_MOTORS)]
        tau = [msg.motor_cmd[i].tau for i in range(r2s_udp.NUM_MOTORS)]

        self._cmd_seq += 1
        packet = r2s_udp.pack_cmd(self._cmd_seq, q, dq, kp, kd, tau)
        self._cmd_sock.sendto(packet, self._cmd_addr)

    def _publish_lowstate(self) -> None:
        """최신 UDP 상태를 `/lowstate`로 발행(타이머, STATE_PUBLISH_HZ)."""
        with self._state_lock:
            state = self._latest_state
        if state is None:
            return

        msg = LowState()
        for i in range(r2s_udp.NUM_MOTORS):
            motor = msg.motor_state[i]
            motor.q = state["q"][i]
            motor.dq = state["dq"][i]
            motor.ddq = state["ddq"][i]
            motor.tau_est = state["tau_est"][i]

        quat_w, quat_x, quat_y, quat_z, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = state["imu"]
        msg.imu_state.quaternion = [quat_w, quat_x, quat_y, quat_z]
        msg.imu_state.gyroscope = [gyro_x, gyro_y, gyro_z]
        msg.imu_state.accelerometer = [acc_x, acc_y, acc_z]
        msg.imu_state.rpy = list(_quat_to_rpy(quat_w, quat_x, quat_y, quat_z))

        self._lowstate_pub.publish(msg)

    def destroy_node(self) -> bool:
        self._cmd_sock.close()
        self._state_sock.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
