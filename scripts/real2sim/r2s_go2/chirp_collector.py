#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 실기 chirp 수집 — PACE 시스템 식별용 여기 데이터 (Phase 2).

**매달린** GO2에 500 Hz로 chirp 위치 명령을 발행하고 ``/lowstate``를 받아 원시 스트림을 저장한다.
여기신호는 sim 재생과 **같은 함수**(``chirp.build_chirp``)에서 나온다 — 명령이 다르면 식별이 성립하지 않는다.

출력은 ``.npz``(원시 스트림)다. 시스템 python3.10(ROS2)에는 torch가 없으므로, PACE가 먹는
``chirp_data.pt``는 Isaac conda 쪽 ``scripts/real2sim/convert_capture_to_pt.py``가 만든다
(CONTRACT §1의 두 파이썬 세계 분리).

⚠ **로봇은 반드시 공중에 매달아야 한다.** PACE는 base 고정(``fix_root_link``)을 전제로 하고, 발이
지면에 닿으면 접촉력이 모델 밖 항으로 들어가 식별이 오염된다.

⚠ **매다는 리그가 공진한다.** 논문은 매단 ANYmal의 chirp를 2 Hz까지만 올렸다("structural
constraints"). 그래서 이 스크립트는 **IMU(자세/각속도/가속도)를 함께 기록**한다 — 변환 단계에서
base가 크게 흔들린 주파수 구간이 보이면 그 데이터는 관절이 아니라 리그를 잰 것이다.
본 수집 전에 **``--amplitude_scale 0.3`` 저진폭 스윕으로 공진을 먼저 확인**할 것.

안전:
  - ``/lowcmd``가 먹으려면 sport(고수준) 서비스를 먼저 내려야 한다 (리모컨 L2+A → L2+B).
  - 전 관절 목표각은 chirp 중심 ± ``--max_dev``로 clamp되고, 발행 전에 GO2 soft limit을 검사한다.
  - 시작 시 현재 자세 q0 → chirp 중심으로 보간 이동(``--approach``), kp는 0에서 램프-인(``--engage``).
  - 추종오차가 ``--abort_dev``를 넘으면 즉시 중단하고 부드럽게 해제한다(부분 데이터는 저장).
  - ``--dry-run``은 ``/lowcmd``를 **전혀 발행하지 않고** 계획/리밋 검사만 출력한다.

실행 (시스템 python3.10 + CycloneDDS)::

    source scripts/real2sim/r2s_go2/robot_env.sh
    # 0) 무모션 계획 확인
    /usr/bin/python3 scripts/real2sim/r2s_go2/chirp_collector.py --dry-run
    # 1) 리그 공진 확인 (저진폭)
    /usr/bin/python3 scripts/real2sim/r2s_go2/chirp_collector.py --suspended \
        --amplitude_scale 0.3 --out sweep_check.npz
    # 2) 본 수집 (게인 세트별로 반복)
    /usr/bin/python3 scripts/real2sim/r2s_go2/chirp_collector.py --suspended --kp 25 --kd 0.5 --out chirp_kp25.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import chirp  # noqa: E402
import lowcmd_crc  # noqa: E402
import motions  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from unitree_go.msg import LowCmd, LowState  # noqa: E402

NUM = chirp.NUM_MOTORS

# 안전 상한 (인수로 더 낮출 순 있어도 넘지 못한다)
MAX_KP = 80.0  # 게인 스윕용으로 safe_joint_test(40)보다 높지만, 매단 상태 전제
MAX_KD = 3.0
MAX_AMPLITUDE_SCALE = 1.0  # chirp.CHIRP_AMPLITUDE 대비 배율
MAX_DEV_CAP = 0.8  # rad, chirp 중심 대비 clamp 밴드 상한
DEFAULT_ABORT_DEV = 0.6  # rad, 추종오차가 이걸 넘으면 중단


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class ChirpCollector(Node):
    """매달린 GO2에 chirp를 발행하고 /lowstate 원시 스트림을 기록한다."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("r2s_chirp_collector")
        self.args = args
        self.q0: list[float] | None = None

        # 상태 스트림 (수신 콜백에서만 append — 발행 루프를 막지 않는다)
        self.t_state: list[float] = []
        self.q_state: list[list[float]] = []
        self.dq_state: list[list[float]] = []
        self.tau_state: list[list[float]] = []
        self.imu_quat: list[list[float]] = []  # wxyz
        self.imu_gyro: list[list[float]] = []
        self.imu_acc: list[list[float]] = []
        self.tick: list[int] = []

        # 명령 스트림 (발행 시점에 append)
        self.t_cmd: list[float] = []
        self.q_cmd: list[list[float]] = []

        self._t0 = time.monotonic()
        self._pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.create_subscription(LowState, "/lowstate", self._on_state, 50)

    # ------------------------------------------------------------------
    # 수신
    # ------------------------------------------------------------------

    def _on_state(self, msg: LowState) -> None:
        if self.q0 is None:
            self.q0 = [float(msg.motor_state[i].q) for i in range(NUM)]
        if not self._recording:
            return
        self.t_state.append(time.monotonic() - self._t0)
        self.q_state.append([float(msg.motor_state[i].q) for i in range(NUM)])
        self.dq_state.append([float(msg.motor_state[i].dq) for i in range(NUM)])
        self.tau_state.append([float(msg.motor_state[i].tau_est) for i in range(NUM)])
        imu = msg.imu_state
        self.imu_quat.append([float(x) for x in imu.quaternion])  # wxyz
        self.imu_gyro.append([float(x) for x in imu.gyroscope])
        self.imu_acc.append([float(x) for x in imu.accelerometer])
        self.tick.append(int(msg.tick))

    _recording = False

    # ------------------------------------------------------------------
    # 발행
    # ------------------------------------------------------------------

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

    def _publish(self, target: list[float], kp: float, kd: float, record: bool = False) -> None:
        self._pub.publish(self._build_cmd(target, kp, kd))
        if record:
            self.t_cmd.append(time.monotonic() - self._t0)
            self.q_cmd.append(list(target))

    def _current_q(self) -> list[float]:
        return list(self.q_state[-1]) if self.q_state else list(self.q0 or chirp.CHIRP_CENTER)

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------

    def run(self) -> int:
        a = self.args
        kp = _clamp(a.kp, 0.0, MAX_KP)
        kd = _clamp(a.kd, 0.0, MAX_KD)
        amp_scale = _clamp(a.amplitude_scale, 0.0, MAX_AMPLITUDE_SCALE)
        max_dev = _clamp(a.max_dev, 0.0, MAX_DEV_CAP)
        abort_dev = _clamp(a.abort_dev, 0.0, MAX_DEV_CAP)
        period = 1.0 / a.rate

        # ---- 궤적 생성 + 리밋 검사 (발행 전에) ----
        amplitude = [x * amp_scale for x in chirp.CHIRP_AMPLITUDE]
        _, traj = chirp.build_chirp(
            duration=a.duration,
            rate_hz=a.rate,
            f0=a.min_frequency,
            f1=a.max_frequency,
            amplitude=amplitude,
        )
        violations = chirp.check_within_soft_limits(traj)
        if violations:
            for j, lo, hi in violations:
                print(
                    f"[chirp] ✗ soft limit 위반: [{j}] {motions.JOINT_NAMES[j]} "
                    f"[{lo:+.3f}, {hi:+.3f}] vs {chirp.SOFT_LIMITS[j]}",
                    flush=True,
                )
            return 1

        center = list(chirp.CHIRP_CENTER)
        print(
            f"[chirp] chirp {a.min_frequency}->{a.max_frequency} Hz, {a.duration}s @ {a.rate:.0f} Hz "
            f"({len(traj)} steps), amplitude x{amp_scale:.2f}",
            flush=True,
        )
        print(f"[chirp] kp={kp:.1f} kd={kd:.2f}  (이 게인은 데이터와 함께 저장되어 sim 재생 때 복원된다)", flush=True)
        print(
            f"[chirp] clamp |q - center| <= {max_dev:.2f} rad,  abort if |q - q_des| > {abort_dev:.2f} rad", flush=True
        )
        print("[chirp] soft limit 검사 통과 (전 12관절)", flush=True)

        # ---- 현재 자세 캡처 ----
        print("[chirp] /lowstate 대기 중 (현재 자세 캡처)...", flush=True)
        t_wait = time.monotonic()
        while self.q0 is None and time.monotonic() - t_wait < 5.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.q0 is None:
            print("[chirp] ✗ /lowstate 미수신. robot_env(RMW/iface)·로봇 전원 확인.", flush=True)
            return 1
        q0 = list(self.q0)
        print(f"[chirp] q0 = {[round(x, 3) for x in q0]}", flush=True)

        if a.dry_run:
            print("[chirp] --dry-run: /lowcmd 발행 안 함(모션 0). 계획만 출력하고 종료.", flush=True)
            return 0

        if not a.suspended:
            print(
                "[chirp] ✗ --suspended 가 없다. PACE는 base 고정을 전제한다 — 로봇을 공중에 매달고,\n"
                "        발이 지면/장애물에 닿지 않는지 확인한 뒤 --suspended 를 붙여 다시 실행할 것.",
                flush=True,
            )
            return 1

        print("[chirp] ⚠ 이제부터 /lowcmd 발행. sport 서비스가 내려가 있는지 확인!  (Ctrl+C=부드러운 해제)", flush=True)

        try:
            # ---- 1) engage: q0 홀드, kp 0 → kp 램프-인 ----
            self._ramp(q0, q0, 0.0, kp, kd, a.engage, period)
            # ---- 2) approach: q0 → chirp 중심으로 보간 이동 ----
            self._ramp(q0, center, kp, kp, kd, a.approach, period)
            # ---- 3) hold: 정착 ----
            self._ramp(center, center, kp, kp, kd, a.hold, period)

            # ---- 4) chirp 본 구간 (여기만 기록) ----
            print(f"[chirp] 수집 시작 ({a.duration:.0f}s)...", flush=True)
            self._recording = True
            aborted = self._run_chirp(traj, center, kp, kd, max_dev, abort_dev, period)
            self._recording = False

            # ---- 5) 해제: 중심 홀드 후 kp 램프-다운 ----
            self._ramp(center, center, kp, 0.0, kd, a.engage, period)
            print("[chirp] 해제 완료.", flush=True)
        except KeyboardInterrupt:
            self._recording = False
            print("\n[chirp] Ctrl+C — kp 램프-다운(부드러운 해제)...", flush=True)
            hold_q = self._current_q()
            self._ramp(hold_q, hold_q, kp, 0.0, kd, a.engage, period)
            aborted = True

        self._save(kp, kd, amp_scale, aborted)
        return 1 if aborted else 0

    def _ramp(
        self,
        q_from: list[float],
        q_to: list[float],
        kp_from: float,
        kp_to: float,
        kd: float,
        seconds: float,
        period: float,
    ) -> None:
        """자세와 kp를 동시에 선형 보간하며 발행한다 (기록하지 않음)."""
        steps = max(1, int(seconds / period))
        t0 = time.monotonic()
        for s in range(steps + 1):
            rclpy.spin_once(self, timeout_sec=0.0)
            u = s / steps
            target = [q_from[i] + u * (q_to[i] - q_from[i]) for i in range(NUM)]
            self._publish(target, kp_from + u * (kp_to - kp_from), kd)
            self._sleep_until(t0 + (s + 1) * period)

    def _run_chirp(
        self,
        traj: list[list[float]],
        center: list[float],
        kp: float,
        kd: float,
        max_dev: float,
        abort_dev: float,
        period: float,
    ) -> bool:
        """chirp 궤적을 절대시각 스케줄로 발행하며 기록한다.

        Returns:
            중단되었으면 True.
        """
        t0 = time.monotonic()
        for k, row in enumerate(traj):
            rclpy.spin_once(self, timeout_sec=0.0)
            target = [_clamp(row[i], center[i] - max_dev, center[i] + max_dev) for i in range(NUM)]
            self._publish(target, kp, kd, record=True)

            # 추종오차 감시 — 관절이 기계 한계에 박히거나 리그에 걸리면 즉시 중단.
            if self.q_state:
                q_now = self.q_state[-1]
                dev = max(abs(q_now[i] - target[i]) for i in range(NUM))
                if dev > abort_dev:
                    worst = max(range(NUM), key=lambda i: abs(q_now[i] - target[i]))
                    print(
                        f"\n[chirp] ✗ ABORT at t={k * period:.2f}s — [{worst}] {motions.JOINT_NAMES[worst]} "
                        f"추종오차 {dev:.3f} rad > {abort_dev:.3f}. 부분 데이터를 저장하고 해제한다.",
                        flush=True,
                    )
                    return True
            self._sleep_until(t0 + (k + 1) * period)
        return False

    @staticmethod
    def _sleep_until(deadline: float) -> None:
        """절대시각까지 대기 — ``sleep(period)`` 누적 드리프트를 막는다(500 Hz에서 치명적)."""
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------

    def _save(self, kp: float, kd: float, amp_scale: float, aborted: bool) -> None:
        a = self.args
        out_dir = a.out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, a.out)

        if not self.t_cmd or not self.t_state:
            print("[chirp] ✗ 기록된 샘플이 없다. 저장하지 않는다.", flush=True)
            return

        np.savez(
            out_path,
            # 명령 스트림 (발행 시각, 우리 monotonic 시계)
            t_cmd=np.asarray(self.t_cmd, dtype=np.float64),
            q_cmd=np.asarray(self.q_cmd, dtype=np.float32),
            # 상태 스트림 (수신 시각, 같은 시계)
            t_state=np.asarray(self.t_state, dtype=np.float64),
            q=np.asarray(self.q_state, dtype=np.float32),
            dq=np.asarray(self.dq_state, dtype=np.float32),
            tau_est=np.asarray(self.tau_state, dtype=np.float32),
            # 리그 공진 판정용 — base가 크게 흔들렸다면 관절이 아니라 리그를 잰 것이다.
            imu_quat=np.asarray(self.imu_quat, dtype=np.float32),  # wxyz
            imu_gyro=np.asarray(self.imu_gyro, dtype=np.float32),
            imu_acc=np.asarray(self.imu_acc, dtype=np.float32),
            tick=np.asarray(self.tick, dtype=np.int64),
            # 메타
            kp=np.full(NUM, kp, dtype=np.float32),
            kd=np.full(NUM, kd, dtype=np.float32),
            joint_order=np.asarray(motions.JOINT_NAMES),
            rate_hz=np.float64(a.rate),
            f0_hz=np.float64(a.min_frequency),
            f1_hz=np.float64(a.max_frequency),
            duration_s=np.float64(a.duration),
            amplitude_scale=np.float64(amp_scale),
            aborted=np.bool_(aborted),
        )

        cmd_dt = np.diff(np.asarray(self.t_cmd))
        st_dt = np.diff(np.asarray(self.t_state))
        print(f"[chirp] 저장: {out_path}", flush=True)
        print(
            f"[chirp]   명령 {len(self.t_cmd)} 샘플, 실효 {1.0 / cmd_dt.mean():.1f} Hz, "
            f"max gap {cmd_dt.max() * 1e3:.1f} ms",
            flush=True,
        )
        print(
            f"[chirp]   상태 {len(self.t_state)} 샘플, 실효 {1.0 / st_dt.mean():.1f} Hz, "
            f"max gap {st_dt.max() * 1e3:.1f} ms",
            flush=True,
        )
        gyro = np.asarray(self.imu_gyro)
        print(
            f"[chirp]   base 각속도 |w| max={np.linalg.norm(gyro, axis=1).max():.3f} rad/s "
            "(크면 리그가 공진한 것 — 그 데이터로는 관절을 식별할 수 없다)",
            flush=True,
        )
        print(
            "[chirp] 다음: ./isaaclab.sh -p scripts/real2sim/convert_capture_to_pt.py --capture " + out_path, flush=True
        )


def main() -> None:
    p = argparse.ArgumentParser(description="매달린 GO2에서 PACE 식별용 chirp 데이터를 수집한다.")
    p.add_argument("--kp", type=float, default=25.0, help=f"위치 게인 (상한 {MAX_KP}). 데이터와 함께 저장된다.")
    p.add_argument("--kd", type=float, default=0.5, help=f"속도 게인 (상한 {MAX_KD}).")
    p.add_argument("--duration", type=float, default=chirp.DEFAULT_DURATION_S, help="chirp 길이 [s].")
    p.add_argument("--min_frequency", type=float, default=chirp.DEFAULT_F0_HZ, help="시작 주파수 [Hz].")
    p.add_argument(
        "--max_frequency",
        type=float,
        default=chirp.DEFAULT_F1_HZ,
        help="종료 주파수 [Hz]. ⚠ 리그 공진을 넘지 말 것 — 논문은 매단 ANYmal에서 2 Hz로 제한했다.",
    )
    p.add_argument(
        "--amplitude_scale",
        type=float,
        default=1.0,
        help=f"chirp 진폭 배율 (상한 {MAX_AMPLITUDE_SCALE}). 공진 확인은 0.3 권장.",
    )
    p.add_argument("--rate", type=float, default=chirp.DEFAULT_RATE_HZ, help="/lowcmd 발행률 [Hz] = sim 제어율.")
    p.add_argument("--engage", type=float, default=2.0, help="kp 램프-인/아웃 시간 [s].")
    p.add_argument("--approach", type=float, default=3.0, help="q0 → chirp 중심 보간 이동 시간 [s].")
    p.add_argument("--hold", type=float, default=1.5, help="chirp 전 정착 홀드 시간 [s].")
    p.add_argument("--max_dev", type=float, default=0.7, help=f"목표각 clamp 밴드 [rad] (상한 {MAX_DEV_CAP}).")
    p.add_argument("--abort_dev", type=float, default=DEFAULT_ABORT_DEV, help="추종오차가 이 값을 넘으면 중단 [rad].")
    p.add_argument("--out", type=str, default="chirp_capture.npz", help="출력 파일명.")
    p.add_argument("--out_dir", type=str, default="data/go2_real", help="출력 디렉터리 (repo 기준).")
    p.add_argument("--suspended", action="store_true", help="로봇이 공중에 매달려 있음을 확인. 없으면 발행하지 않는다.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="/lowcmd 발행 없이 계획/리밋 검사만 출력.")
    args = p.parse_args()

    rclpy.init()
    node = ChirpCollector(args)
    rc = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
