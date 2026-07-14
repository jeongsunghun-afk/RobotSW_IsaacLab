#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 게인 규약 검증 — Unitree kp가 IsaacLab과 같은 단위(N·m/rad)인가?

PACE는 PD 게인을 **알려진 입력**으로 가정하고 식별하지 않는다. 논문이 밝히듯
``{armature, damping, kp, kd}``의 **공통 스케일이 폐루프 거동을 보존**하기 때문이다. 따라서:

    실제 토크가 ``α·kp·e`` 인데 우리가 ``kp`` 라고 믿으면,
    CMA-ES는 armature/마찰을 모두 ``1/α`` 배로 줄여서 **위치 궤적은 완벽히 재현**한다.
    score는 훌륭하게 나오고, 물리값은 α배 틀린다.

즉 **이 오차는 식별 데이터로 잡을 수 없다.** 사전에 재는 것이 유일한 방어이며, 틀렸을 때 깨지는 것은
토크 크기 → effort limit(23.5 N·m) 포화 거동과 에너지 지표다.

원리 (모델 불필요):
    PD 법칙 자체가 ``τ = kp·(q_des − q) + kd·(0 − dq)`` 이다. 정착 상태에서 ``dq ≈ 0`` 이므로
    ``τ = kp·e``. 목표각을 조금씩 옮겨 ``e`` 를 바꿔가며 ``(e, tau_est)`` 를 모으고 **직선의 기울기**를
    구하면 그게 실효 kp다. 중력은 ``e`` 가 어디서 정착할지만 정하지 이 관계식은 건드리지 않는다
    (그래서 링크 질량/CAD가 필요 없다).

    kd는 이 방법으로 재지 않는다 — kd 오차는 어차피 viscous 마찰과 수학적으로 구분되지 않아
    식별이 알아서 흡수한다(CONTRACT §12.4).

⚠ 로봇을 **매달고**(다리 자유) sport 서비스를 내린 뒤 실행할 것.

실행 (시스템 python3.10 + CycloneDDS)::

    source scripts/real2sim/r2s_go2/robot_env.sh
    /usr/bin/python3 scripts/real2sim/r2s_go2/gain_check.py --selftest        # 추정기 자체 검증(로봇 불필요)
    /usr/bin/python3 scripts/real2sim/r2s_go2/gain_check.py --dry-run
    /usr/bin/python3 scripts/real2sim/r2s_go2/gain_check.py --suspended --joint 1 --kp 25
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

NUM = 12
RATE_HZ = 200.0

MAX_KP = 60.0
MAX_KD = 2.0
MAX_OFFSET = 0.20  # rad, 목표각 오프셋 상한


def fit_slope(errors: list[float], torques: list[float]) -> tuple[float, float, float]:
    """``tau = slope * e + intercept`` 최소자승 직선 적합.

    Args:
        errors: 정착 추종오차 ``q_des − q`` [rad].
        torques: 정착 토크 추정값 [N·m].

    Returns:
        ``(slope, intercept, r2)`` — slope가 실효 kp [N·m/rad].

    Raises:
        ValueError: 표본이 2개 미만이거나 오차 분산이 0일 때.
    """
    n = len(errors)
    if n < 2:
        raise ValueError("표본이 2개 미만이다.")
    mx, my = statistics.fmean(errors), statistics.fmean(torques)
    sxx = sum((x - mx) ** 2 for x in errors)
    if sxx <= 0:
        raise ValueError("추종오차의 분산이 0이다 — 오프셋이 실제로 적용되지 않았다.")
    sxy = sum((x - mx) * (y - my) for x, y in zip(errors, torques))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(errors, torques))
    ss_tot = sum((y - my) ** 2 for y in torques)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def report(kp_cmd: float, errors: list[float], torques: list[float]) -> int:
    """측정 결과를 판정해 출력한다.

    Args:
        kp_cmd: 명령한 kp.
        errors: 정착 추종오차 [rad].
        torques: 정착 토크 [N·m].

    Returns:
        종료 코드 (0=합격).
    """
    slope, intercept, r2 = fit_slope(errors, torques)
    ratio = slope / kp_cmd if kp_cmd else float("nan")

    print("\n=== 측정 ===")
    print(f"{'e = q_des - q [rad]':>22s} {'tau_est [N·m]':>14s}")
    for e, t in zip(errors, torques):
        print(f"{e:22.4f} {t:14.3f}")

    print("\n=== 적합 (tau = kp_eff · e + b) ===")
    print(f"명령 kp        : {kp_cmd:.2f}")
    print(f"실효 kp (기울기): {slope:.2f}  N·m/rad")
    print(f"절편 b         : {intercept:+.3f} N·m   (0에 가까워야 정상 — 크면 tau_est 오프셋/마찰)")
    print(f"R²             : {r2:.4f}          (1에 가까워야 PD 법칙대로 동작)")
    print(f"\n▶ α = 실효/명령 = {ratio:.3f}")

    if r2 < 0.9:
        print("✗ R²가 낮다. 정착이 덜 됐거나 마찰/포화가 지배한다. --settle 을 늘리거나 오프셋을 키울 것.")
        return 1
    if abs(ratio - 1.0) <= 0.1:
        print("✅ α ≈ 1 — Unitree kp가 IsaacLab과 같은 N·m/rad 규약이다. 그대로 진행.")
        return 0
    print(
        f"⚠ α = {ratio:.3f} — 게인 규약이 어긋난다.\n"
        f"  이 상태로 식별하면 위치 궤적은 맞게 재현되지만 **armature/마찰이 {1 / ratio:.3f}배 편향**되고\n"
        "  토크 크기가 틀어져 effort limit 포화·에너지 지표가 오염된다.\n"
        f"  대응: sysid/학습 cfg의 stiffness를 실효값({slope:.1f})으로 쓰거나, 원인(단위/기어비)을 먼저 규명할 것."
    )
    return 2


def selftest() -> int:
    """로봇 없이 추정기 자체를 검증한다 (tau = kp·e + 잡음)."""
    import random

    random.seed(0)
    kp_true = 25.0
    errors = [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15]
    torques = [kp_true * e + random.gauss(0, 0.05) for e in errors]
    slope, intercept, r2 = fit_slope(errors, torques)
    assert abs(slope - kp_true) < 0.5, slope
    assert r2 > 0.99, r2
    print(f"selftest OK — kp_true={kp_true}, 추정={slope:.2f}, b={intercept:+.3f}, R²={r2:.4f}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Unitree kp가 IsaacLab과 같은 단위인지 검증한다.")
    p.add_argument("--joint", type=int, default=1, help="대상 관절 0..11 (기본 1 = FR_thigh).")
    p.add_argument("--kp", type=float, default=25.0, help=f"검증할 kp (상한 {MAX_KP}).")
    p.add_argument("--kd", type=float, default=0.5, help=f"kd (상한 {MAX_KD}). 정착 후 dq≈0이라 영향 적음.")
    p.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=[-0.15, -0.10, -0.05, 0.05, 0.10, 0.15],
        help=f"q0 대비 목표각 오프셋 [rad] (각 |값| <= {MAX_OFFSET}).",
    )
    p.add_argument("--settle", type=float, default=1.5, help="오프셋마다 정착 대기 시간 [s].")
    p.add_argument("--window", type=float, default=0.5, help="정착 후 평균낼 구간 [s].")
    p.add_argument("--engage", type=float, default=2.0, help="kp 램프-인/아웃 시간 [s].")
    p.add_argument("--suspended", action="store_true", help="로봇이 매달려 있음을 확인. 없으면 발행하지 않는다.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="/lowcmd 발행 없이 계획만 출력.")
    p.add_argument("--selftest", action="store_true", help="로봇 없이 추정기만 검증하고 종료.")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if not 0 <= args.joint < NUM:
        p.error("--joint 은 0..11")
    if any(abs(o) > MAX_OFFSET for o in args.offsets):
        p.error(f"--offsets 의 각 |값| 은 {MAX_OFFSET} 이하")

    sys.path.insert(0, os.path.dirname(__file__))
    import lowcmd_crc  # noqa: PLC0415
    import motions  # noqa: PLC0415
    import rclpy  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from unitree_go.msg import LowCmd, LowState  # noqa: PLC0415

    kp = min(args.kp, MAX_KP)
    kd = min(args.kd, MAX_KD)
    period = 1.0 / RATE_HZ

    class GainCheck(Node):
        def __init__(self) -> None:
            super().__init__("r2s_gain_check")
            self.q0: list[float] | None = None
            self.q: list[float] | None = None
            self.tau: list[float] | None = None
            self._pub = self.create_publisher(LowCmd, "/lowcmd", 10)
            self.create_subscription(LowState, "/lowstate", self._on_state, 10)

        def _on_state(self, msg: LowState) -> None:
            self.q = [float(msg.motor_state[i].q) for i in range(NUM)]
            self.tau = [float(msg.motor_state[i].tau_est) for i in range(NUM)]
            if self.q0 is None:
                self.q0 = list(self.q)

        def _publish(self, target: list[float], kp_now: float) -> None:
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
                m.kp = kp_now
                m.kd = kd
            lowcmd_crc.set_crc(msg)
            self._pub.publish(msg)

        def hold(self, target: list[float], kp_from: float, kp_to: float, seconds: float) -> None:
            steps = max(1, int(seconds * RATE_HZ))
            for s in range(steps + 1):
                rclpy.spin_once(self, timeout_sec=0.0)
                self._publish(target, kp_from + (s / steps) * (kp_to - kp_from))
                time.sleep(period)

        def measure(self, target: list[float], joint: int, window: float) -> tuple[float, float]:
            """정착 구간 동안 (e, tau)를 평균낸다."""
            es: list[float] = []
            ts: list[float] = []
            steps = max(1, int(window * RATE_HZ))
            for _ in range(steps):
                rclpy.spin_once(self, timeout_sec=0.0)
                self._publish(target, kp)
                if self.q is not None and self.tau is not None:
                    es.append(target[joint] - self.q[joint])
                    ts.append(self.tau[joint])
                time.sleep(period)
            return statistics.fmean(es), statistics.fmean(ts)

    rclpy.init()
    node = GainCheck()

    print("[gain] /lowstate 대기 중...", flush=True)
    t_wait = time.monotonic()
    while node.q0 is None and time.monotonic() - t_wait < 5.0:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.q0 is None:
        print("[gain] ✗ /lowstate 미수신. robot_env(RMW/iface)·로봇 전원 확인.", flush=True)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    q0 = list(node.q0)
    jname = motions.JOINT_NAMES[args.joint]
    print(f"[gain] q0 = {[round(x, 3) for x in q0]}", flush=True)
    print(f"[gain] 대상 [{args.joint}] {jname},  kp={kp} kd={kd},  오프셋 {args.offsets}", flush=True)

    if args.dry_run:
        print("[gain] --dry-run: /lowcmd 발행 안 함. 계획만 출력하고 종료.", flush=True)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    if not args.suspended:
        print("[gain] ✗ --suspended 가 없다. 다리가 자유롭게 움직일 수 있어야 한다.", flush=True)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print("[gain] ⚠ /lowcmd 발행 시작. sport 서비스 해제 확인!", flush=True)
    errors: list[float] = []
    torques: list[float] = []
    rc = 1
    try:
        node.hold(q0, 0.0, kp, args.engage)  # kp 램프-인
        for off in args.offsets:
            target = list(q0)
            target[args.joint] = q0[args.joint] + off
            node.hold(target, kp, kp, args.settle)  # 정착
            e, t = node.measure(target, args.joint, args.window)
            print(f"[gain]   offset {off:+.3f} → e={e:+.4f} rad, tau_est={t:+.3f} N·m", flush=True)
            errors.append(e)
            torques.append(t)
        node.hold(q0, kp, 0.0, args.engage)  # 램프-다운
        rc = report(kp, errors, torques)
    except KeyboardInterrupt:
        print("\n[gain] Ctrl+C — 램프-다운...", flush=True)
        node.hold(q0, kp, 0.0, args.engage)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
