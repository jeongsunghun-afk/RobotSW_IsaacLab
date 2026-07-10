#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 read_lowstate — 실기 GO2의 `/lowstate`를 rclpy로 구독해 파싱·검증한다.

실기 통신 점검 및 sim↔real 비교(M2)의 씨앗. 시스템 python3.10 + CycloneDDS RMW 필요:

    source scripts/real2sim/r2s_go2/robot_env.sh   # 또는 unitree setup_local.sh
    /usr/bin/python3 scripts/real2sim/r2s_go2/read_lowstate.py [seconds]

seconds: 수신 관찰 시간 [s], 기본 3.0. 미수신이면 exit 1(FAIL).
"""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState


class LowStateReader(Node):
    """`/lowstate` 최신 메시지와 수신 카운트만 보관하는 최소 구독 노드."""

    def __init__(self) -> None:
        super().__init__("r2s_lowstate_reader")
        self.count = 0
        self.last: LowState | None = None
        self.create_subscription(LowState, "/lowstate", self._cb, 10)

    def _cb(self, msg: LowState) -> None:
        self.count += 1
        self.last = msg


def main() -> None:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    rclpy.init()
    node = LowStateReader()
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        rclpy.spin_once(node, timeout_sec=0.1)

    m = node.last
    if node.count == 0 or m is None:
        print("RESULT: FAIL — /lowstate 미수신. robot_env(RMW=cyclonedds/iface), 로봇 전원, ros2 daemon 확인.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    q = [round(m.motor_state[i].q, 3) for i in range(12)]
    dq = [round(m.motor_state[i].dq, 3) for i in range(12)]
    quat = [round(x, 3) for x in m.imu_state.quaternion]  # wxyz
    foot_force = list(m.foot_force)
    hz = node.count / duration

    print(f"RESULT: PASS — /lowstate {node.count}건 수신 (~{hz:.0f}Hz)")
    print(f"  head           = {list(m.head)}  (0xFE,0xEF=254,239 기대)")
    print(f"  motor q[0:12]  = {q}")
    print(f"  motor dq[0:12] = {dq}")
    print(f"  imu quat(wxyz) = {quat}")
    print(f"  foot_force     = {foot_force}")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
