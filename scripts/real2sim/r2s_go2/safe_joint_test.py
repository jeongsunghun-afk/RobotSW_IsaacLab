#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 safe_joint_test — 앉은 자세 그대로 한 관절만 살짝 움직이는 저수준 테스트.

``low_level_ctrl`` 예제는 제어하지 않는 관절을 kp=kd=0(힘빠짐)으로 두어 로봇이 주저앉는다.
이 스크립트는 반대로:

  1. ``/lowstate`` 를 받아 **현재 12관절 자세 q0 를 캡처**한다.
  2. **전 관절을 q0 에 부드러운 PD 로 홀드**한다(캡처 자세 유지 → 앉은 채 안 무너짐).
  3. kp 를 0→목표로 서서히 램프-인(engage)해 인수 시 저크를 없앤다.
  4. 선택한 **한 관절만** q0 주변으로 작은 사인파를 준다.
  5. 종료(Ctrl+C/시간초과) 시 kp 를 서서히 램프-다운(부드러운 해제).

안전:
  - 앉은 자세 유지 → **매달 필요 없음**. 단, ``/lowcmd`` 가 먹으려면 **sport(고수준) 서비스를
    먼저 내려야** 한다(리모컨 damping: L2+A → L2+B). 안 내리면 명령이 무시/충돌될 수 있다.
  - 모든 목표각은 q0±max_dev 로 clamp, kp/amp 상한 clamp → 오타/버그로도 큰 동작 불가.
  - ``--dry-run`` 은 ``/lowcmd`` 를 **전혀 발행하지 않고** 상태만 읽어 계획을 출력(모션 0, 검증용).

실행(시스템 python3.10 + CycloneDDS RMW):
    source scripts/real2sim/r2s_go2/robot_env.sh
    /usr/bin/python3 scripts/real2sim/r2s_go2/safe_joint_test.py --dry-run   # 먼저 무모션 확인
    /usr/bin/python3 scripts/real2sim/r2s_go2/safe_joint_test.py --joint 0 --amp 0.15
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import lowcmd_crc  # noqa: E402
import motions  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from unitree_go.msg import LowCmd, LowState  # noqa: E402

NUM = 12
RATE_HZ = 200.0

# 안전 상한 (인수로 더 낮출 순 있어도 이 값을 넘지 못함)
MAX_AMP = 0.4  # rad
MAX_KP = 40.0
MAX_KD = 2.0
MAX_DEV_CAP = 0.5  # rad, per-joint clamp 밴드 상한


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class SafeJointTest(Node):
    """현재 자세 홀드 + 한 관절 살짝 흔들기."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("r2s_safe_joint_test")
        self.args = args
        self.q0: list[float] | None = None
        self._pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.create_subscription(LowState, "/lowstate", self._on_state, 10)

    def _on_state(self, msg: LowState) -> None:
        if self.q0 is None:
            self.q0 = [float(msg.motor_state[i].q) for i in range(NUM)]

    def _build_cmd(self, target: list[float], kp: float, kd: float) -> LowCmd:
        msg = LowCmd()
        msg.head[0] = lowcmd_crc.HEAD[0]
        msg.head[1] = lowcmd_crc.HEAD[1]
        msg.level_flag = lowcmd_crc.LEVEL_FLAG_LOWLEVEL
        msg.gpio = 0
        for i in range(NUM):
            m = msg.motor_cmd[i]
            m.mode = lowcmd_crc.MOTOR_MODE_SERVO
            m.q = target[i]
            m.dq = 0.0
            m.tau = 0.0
            m.kp = kp
            m.kd = kd
        lowcmd_crc.set_crc(msg)
        return msg

    def run(self) -> int:
        a = self.args
        joint = a.joint
        amp = _clamp(a.amp, 0.0, MAX_AMP)
        kp = _clamp(a.kp, 0.0, MAX_KP)
        kd = _clamp(a.kd, 0.0, MAX_KD)
        max_dev = _clamp(a.max_dev, 0.0, MAX_DEV_CAP)
        engage = max(0.1, a.engage)
        period = 1.0 / RATE_HZ

        # 1) 현재 자세 캡처 대기
        print("[safe_test] /lowstate 대기 중 (현재 자세 캡처)...", flush=True)
        t_wait = time.monotonic()
        while self.q0 is None and time.monotonic() - t_wait < 5.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.q0 is None:
            print("[safe_test] ✗ /lowstate 미수신. robot_env(RMW/iface)·로봇 전원 확인.", flush=True)
            return 1

        q0 = self.q0
        jname = motions.JOINT_NAMES[joint]
        print(f"[safe_test] 캡처 자세 q0 = {[round(x, 3) for x in q0]}", flush=True)
        print(
            f"[safe_test] 대상: [{joint}] {jname},  amp={amp:.3f} rad,  freq={a.freq:.2f} Hz,  "
            f"kp={kp:.1f}, kd={kd:.1f},  engage={engage:.1f}s,  duration={a.duration:.1f}s",
            flush=True,
        )
        print(f"[safe_test] 안전 clamp: 각 관절 |q-q0| <= {max_dev:.2f} rad", flush=True)

        if a.dry_run:
            print("[safe_test] --dry-run: /lowcmd 발행 안 함(모션 0). 계획만 출력하고 종료.", flush=True)
            return 0

        print(
            "[safe_test] ⚠ 이제부터 /lowcmd 발행. sport 서비스가 내려가 있는지 확인!  (Ctrl+C=부드러운 해제)",
            flush=True,
        )

        t0 = time.monotonic()
        hold_s = a.hold
        motion_start = engage + hold_s
        total = motion_start + a.duration
        try:
            while True:
                rclpy.spin_once(self, timeout_sec=0.0)
                t = time.monotonic() - t0
                if t >= total:
                    break
                ramp = _clamp(t / engage, 0.0, 1.0)  # kp 0→1 램프-인
                kp_now = kp * ramp
                target = list(q0)
                if t >= motion_start:
                    tm = t - motion_start
                    target[joint] = q0[joint] + amp * math.sin(2.0 * math.pi * a.freq * tm)
                # 안전 clamp
                for i in range(NUM):
                    target[i] = _clamp(target[i], q0[i] - max_dev, q0[i] + max_dev)
                self._pub.publish(self._build_cmd(target, kp_now, kd))
                time.sleep(period)
            self._ramp_down(q0, kp, kd, engage, period)
            print("[safe_test] ✅ 완료(정상 종료).", flush=True)
            return 0
        except KeyboardInterrupt:
            print("\n[safe_test] Ctrl+C — kp 램프-다운(부드러운 해제)...", flush=True)
            self._ramp_down(q0, kp, kd, engage, period)
            print("[safe_test] 해제 완료.", flush=True)
            return 0

    def _ramp_down(self, q0: list[float], kp: float, kd: float, engage: float, period: float) -> None:
        """마지막에 kp 를 kp→0 으로 서서히 낮춰 부드럽게 손을 뗀다(현재 q0 홀드)."""
        steps = max(1, int(engage * RATE_HZ))
        for s in range(steps, -1, -1):
            kp_now = kp * (s / steps)
            self._pub.publish(self._build_cmd(list(q0), kp_now, kd))
            time.sleep(period)


def main() -> None:
    p = argparse.ArgumentParser(description="앉은 자세 유지 + 한 관절 살짝 흔들기(안전 저수준 테스트)")
    p.add_argument(
        "--joint", type=int, default=0, help="관절 인덱스 0..11 (기본 0=FR_hip). 0/3/6/9=hip, 1/4/7/10=thigh"
    )
    p.add_argument("--amp", type=float, default=0.15, help=f"진폭 [rad], 기본 0.15 (상한 {MAX_AMP})")
    p.add_argument("--freq", type=float, default=0.2, help="사인 주파수 [Hz], 기본 0.2(느림)")
    p.add_argument("--kp", type=float, default=20.0, help=f"홀드 kp, 기본 20 (상한 {MAX_KP})")
    p.add_argument("--kd", type=float, default=0.5, help=f"홀드 kd, 기본 0.5 (상한 {MAX_KD})")
    p.add_argument("--engage", type=float, default=2.0, help="kp 램프-인/아웃 시간 [s], 기본 2")
    p.add_argument("--hold", type=float, default=1.5, help="흔들기 전 자세 홀드 시간 [s], 기본 1.5")
    p.add_argument("--duration", type=float, default=10.0, help="사인 흔들기 시간 [s], 기본 10")
    p.add_argument(
        "--max-dev",
        dest="max_dev",
        type=float,
        default=0.3,
        help=f"관절별 |q-q0| clamp [rad], 기본 0.3 (상한 {MAX_DEV_CAP})",
    )
    p.add_argument("--dry-run", action="store_true", help="/lowcmd 발행 없이 상태만 읽어 계획 출력(모션 0)")
    args = p.parse_args()
    if not 0 <= args.joint < NUM:
        p.error("--joint 은 0..11")

    rclpy.init()
    node = SafeJointTest(args)
    rc = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
