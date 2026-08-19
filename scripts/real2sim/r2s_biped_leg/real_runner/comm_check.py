# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""real_runner_bipedleg UDP seam 통신 검사 (워크스테이션 쪽).

policy_runner 없이 ACT(9887)를 파이로 송신하고 STATE(9888) 회신을 받아
왕복 경로·페이싱·패킷 무결성을 검증한다. 브링업 사다리 어느 단계에서든
real_runner가 떠 있으면 사용 가능하다.

안전:
    real_runner를 ``--probe`` 또는 ``--hold`` 로 실행한 상태에서 쓰는 것을
    전제로 한다 (두 모드는 ACT를 받아도 모터 목표로 쓰지 않음). 기본 브리지
    모드 상대로 실행하면 첫 ACT가 ENGAGE를 트리거하므로 금지. 그래도 사고를
    줄이기 위해 첫 STATE 수신 후에는 target_q를 수신 q로 echo한다(=현재 자세).

사용:
    ./isaaclab.sh -p scripts/real2sim/r2s_biped_leg/real_runner/comm_check.py \
        --host 192.168.60.5 --duration 10
"""

import argparse
import math
import os
import select
import socket
import sys
import time
from collections.abc import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import r2s_udp  # noqa: E402

JOINT_NAMES = ["HL_hip", "HR_hip", "HL_thigh", "HR_thigh", "HL_calf", "HR_calf", "HL_foot", "HR_foot"]

# real_runner `kTelemDtSec` 와 짝 (2026-08-19: 5 ms). 한쪽만 바꾸면 아래 경고가 오작동한다.
TELEM_EXPECT_HZ: float = 200.0


def report_telem_echo(ticks: "Sequence[int | None]", clamp_seen: int, last: dict) -> None:
    """TELEM 의 시간축·명령 에코를 판정한다 (2026-08-19 확장 필드).

    캡처 품질의 핵심이라 여기서 명시적으로 본다:

    * ``telem_tick`` 이 없으면 파이가 **구 빌드**다 — 시간축이 도착시각(파이썬 스케줄 지터 포함)
      으로만 남아 5 ms 급 현상을 분해할 수 없다.
    * tick 이 건너뛰면 그만큼 **UDP 유실**이다. 지금까지는 유실을 모른 채 선형보간했다.
    * ``clamp_mask`` 가 서면 목표가 soft limit 에 잘린 것 — §18 류 오염을 캡처가 자진신고한다.
    """
    if last.get("telem_tick") is None:
        print("  ⚠ telem_tick 없음 — 파이 real_runner 가 구 빌드(≤121 B). 시간축 에코가 없다 (re-scp + rebuild)")
        return
    seen = [t for t in ticks if t is not None]
    span = seen[-1] - seen[0] + 1 if seen else 0
    missing = span - len(seen)
    print(
        f"  tick {seen[0]}~{seen[-1]}  span {span}  수신 {len(seen)}"
        f"  유실 {missing} ({100 * missing / max(span, 1):.2f}%)"
    )
    if last.get("cmd_q") is None:
        print("  ⚠ cmd_q 무효 — 브리지가 아직 명령을 전송하지 않음(warmup/HOLD). bridge 모드 + ACT 필요")
    if clamp_seen:
        joints = [JOINT_NAMES[j] for j in range(8) if clamp_seen & (1 << j)]
        print(f"  ⚠ 목표가 soft limit 에 클램프됨: {', '.join(joints)} — 이 상태의 캡처는 명령이 오염된다")


def warn_telem_rate(hz: float) -> None:
    """TELEM 실측 레이트가 기대치에 못 미치면 경고한다.

    TELEM 레이트는 **sysid 캡처의 시간 해상도**다 — ``gui_controller`` 의 ``RealMonitorThread``
    가 수신 즉시 기록하므로 이 값이 그대로 npz 의 ``t_real`` 레이트가 된다. 50 Hz 로 캡처하면
    chirp 을 5 Hz 로 올렸을 때 선형보간 오차가 진폭의 5.2 % 에 달해 armature 를 식별할 수 없다
    (reports/_comparisons/pace_bipedleg_foot_coupling_probe/README.md §23-j).
    """
    if hz < TELEM_EXPECT_HZ * 0.9:
        print(
            f"  ⚠ TELEM 레이트 {hz:.1f} Hz 가 기대치 {TELEM_EXPECT_HZ:.0f} Hz 의 90 % 미만 —"
            " 파이 real_runner 가 구 빌드(50 Hz)이거나 손실 발생."
            " 이 상태로 캡처하면 sysid 해상도가 부족하다 (re-scp + rebuild 확인)"
        )


def parse_gain_arg(text: str) -> list[float]:
    """스칼라 하나면 8관절 전체에 broadcast, 콤마 8개면 관절별. articulation 순서."""
    parts = [p for p in text.split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) == 1:
        return vals * 8
    if len(vals) == 8:
        return vals
    raise argparse.ArgumentTypeError(f"게인은 값 1개(broadcast) 또는 8개(콤마구분) — 받은 개수 {len(vals)}")


def percentile(sorted_vals: "Sequence[float]", p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="real_runner UDP seam 통신 검사")
    parser.add_argument("--host", default="192.168.60.5", help="real_runner 주소 (유선 파이 기본)")
    parser.add_argument("--act_port", type=int, default=r2s_udp.REAL_ACT_PORT)
    parser.add_argument("--state_port", type=int, default=r2s_udp.REAL_STATE_PORT)
    parser.add_argument("--duration", type=float, default=10.0, help="테스트 시간 [s]")
    parser.add_argument("--rate", type=float, default=50.0, help="ACT 송신 주파수 [Hz] (학습 STEP과 동일 50)")
    parser.add_argument(
        "--sine_joint",
        type=int,
        default=None,
        help="⚠ 이 관절(articulation 0~7)에 sine 목표 송신 — bridge 모드면 실제로 움직인다!",
    )
    parser.add_argument("--sine_amp", type=float, default=0.2, help="sine 진폭 [rad]")
    parser.add_argument("--sine_hz", type=float, default=0.5, help="sine 주파수 [Hz]")
    parser.add_argument(
        "--kp",
        type=parse_gain_arg,
        default=None,
        help="POLICY_GAIN 송신: kp 스칼라(전관절) 또는 콤마 8개(articulation 순서). --kd 와 함께 사용",
    )
    parser.add_argument(
        "--kd",
        type=parse_gain_arg,
        default=None,
        help="POLICY_GAIN 송신: kd 스칼라(전관절) 또는 콤마 8개(articulation 순서). --kp 와 함께 사용",
    )
    args = parser.parse_args()
    if (args.kp is None) != (args.kd is None):
        print("✗ [comm_check] --kp 와 --kd 는 함께 지정해야 한다 (kp/kd 쌍으로 송신)")
        return 1
    if args.sine_joint is not None:
        print(
            f"⚠ [comm_check] sine 모드: joint {args.sine_joint} amp {args.sine_amp} rad @ {args.sine_hz} Hz "
            "— real_runner가 BRIDGE 모드면 실제 액추에이션 발생 (probe/hold에선 무시됨)"
        )

    print(f"[comm_check] ACT → {args.host}:{args.act_port} @ {args.rate:.0f} Hz, STATE ← *:{args.state_port}")
    print("[comm_check] 전제: 파이 real_runner가 --probe 또는 --hold 상태 (ACT 무시 모드)")

    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        rx_sock.bind(("", args.state_port))
    except OSError as e:
        print(f"[comm_check] 포트 {args.state_port} bind 실패({e}) — policy_runner가 이미 떠 있으면 종료 후 재시도")
        return 1
    rx_sock.setblocking(False)
    telem_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    telem_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    telem_cnt = 0
    telem_last: dict | None = None
    telem_ticks: list[int | None] = []  # 브리지 틱 — 유실은 번호가 건너뛰는 것으로 드러난다
    telem_clamp = 0  # 구간 내내 OR — 한 번이라도 목표가 잘렸으면 그 캡처는 오염이다
    try:
        telem_sock.bind(("", r2s_udp.REAL_TELEM_PORT))
        telem_sock.setblocking(False)
    except OSError as e:
        print(f"[comm_check] TELEM 포트 bind 실패({e}) — gui monitor 실행 중이면 TELEM 검사는 생략")
        telem_sock = None

    send_dt = 1.0 / args.rate
    t0 = time.monotonic()
    next_send = t0
    seq = 0
    tx_cnt = 0
    last_q: list[float] | None = None  # 첫 STATE 후 echo-back 목표
    rx_times: list[float] = []
    seq_lags: list[int] = []
    bad_pkts = 0
    q_min = [math.inf] * 8
    q_max = [-math.inf] * 8
    dq_absmax = [0.0] * 8
    grav_first: list[float] | None = None
    grav_last: list[float] | None = None
    grav_all_upright = True  # 전 패킷이 정확히 (0,0,-1)이면 IMU 미수신 의심

    # POLICY_GAIN 송신 — UDP 유실 대비 시작 시 몇 번 반복 전송. real_runner 는 수신 즉시 kp/kd 갱신.
    if args.kp is not None:
        print(f"[comm_check] GAIN 송신: kp={args.kp} kd={args.kd} (articulation 순서) → {args.host}:{args.act_port}")
        for i in range(5):
            tx_sock.sendto(r2s_udp.pack_policy_gain(i, args.kp, args.kd), (args.host, args.act_port))
            time.sleep(0.01)

    while True:
        now = time.monotonic()
        if now - t0 >= args.duration:
            break
        if now >= next_send:
            target = list(last_q) if last_q is not None else [0.0] * 8
            if args.sine_joint is not None and 0 <= args.sine_joint < 8:
                target[args.sine_joint] = args.sine_amp * math.sin(2.0 * math.pi * args.sine_hz * (now - t0))
            tx_sock.sendto(r2s_udp.pack_policy_act(seq, target), (args.host, args.act_port))
            seq += 1
            tx_cnt += 1
            next_send += send_dt
        if telem_sock is not None:
            while True:
                try:
                    tdata, _ = telem_sock.recvfrom(2048)
                except BlockingIOError:
                    break
                t_pkt = r2s_udp.unpack_policy_telem(tdata)
                if t_pkt is not None:
                    telem_cnt += 1
                    telem_last = t_pkt
                    # 분기 없이 담고 판정은 report_telem_echo 가 한다 (구 브리지면 None 이 섞인다).
                    telem_ticks.append(t_pkt["telem_tick"])
                    telem_clamp |= t_pkt["clamp_mask"]
        timeout = max(0.0, min(next_send - time.monotonic(), 0.005))
        readable, _, _ = select.select([rx_sock], [], [], timeout)
        if not readable:
            continue
        while True:
            try:
                data, _ = rx_sock.recvfrom(2048)
            except BlockingIOError:
                break
            st = r2s_udp.unpack_policy_state(data)
            if st is None:
                bad_pkts += 1
                continue
            rx_times.append(time.monotonic())
            seq_lags.append((seq - 1) - st["seq"])
            last_q = st["q"]
            for j in range(8):
                q_min[j] = min(q_min[j], st["q"][j])
                q_max[j] = max(q_max[j], st["q"][j])
                dq_absmax[j] = max(dq_absmax[j], abs(st["dq"][j]))
            g = st["gravity"]
            grav_last = g
            if grav_first is None:
                grav_first = g
            if abs(g[0]) > 1e-9 or abs(g[1]) > 1e-9 or abs(g[2] + 1.0) > 1e-9:
                grav_all_upright = False

    # ---- 리포트 ----
    rx_cnt = len(rx_times)
    expected_rx = args.duration / 0.020  # real_runner 20 ms 페이싱
    print(f"\n[comm_check] === {args.duration:.1f} s 결과 ===")
    if args.kp is not None:
        print(f"GAIN 송신 : kp={args.kp} kd={args.kd} (articulation 순서, 5회 반복)")
    print(f"TX(ACT)  : {tx_cnt}  (송신 seq 0~{seq - 1})")
    print(f"RX(STATE): {rx_cnt}  (20 ms 페이싱 기대치 ~{expected_rx:.0f}, 수신율 {100.0 * rx_cnt / expected_rx:.1f}%)")
    if bad_pkts:
        print(f"⚠ 파싱 실패 패킷: {bad_pkts}")
    if rx_cnt < 2:
        print("✗ STATE 회신이 없거나 부족 — 파이 real_runner 상태/방화벽/경로 확인")
        return 1

    dts = sorted((rx_times[i + 1] - rx_times[i]) * 1e3 for i in range(rx_cnt - 1))
    mean_dt = sum(dts) / len(dts)
    print(f"회신 간격 : mean {mean_dt:.2f} ms  min {dts[0]:.2f}  p95 {percentile(dts, 95):.2f}  max {dts[-1]:.2f}")
    lag_sorted = sorted(seq_lags)
    print(f"seq echo 지연: median {percentile(lag_sorted, 50)}  p95 {percentile(lag_sorted, 95)}  (ACT seq 대비)")

    print("관절 q [rad] (articulation 순서):")
    for j in range(8):
        span = q_max[j] - q_min[j]
        print(f"  {JOINT_NAMES[j]:9s} [{q_min[j]:+.3f}, {q_max[j]:+.3f}]  span {span:.4f}  |dq|max {dq_absmax[j]:.3f}")
    if grav_first is not None and grav_last is not None:
        print(
            f"gravity 첫/끝: ({grav_first[0]:+.3f},{grav_first[1]:+.3f},{grav_first[2]:+.3f})"
            f" / ({grav_last[0]:+.3f},{grav_last[1]:+.3f},{grav_last[2]:+.3f})"
        )
    if grav_all_upright:
        print(
            "⚠ gravity가 전 구간 정확히 (0,0,-1) — real_runner IMU 미수신 fallback 의심"
            " (파이 로그의 'IMU 미수신!' 확인)"
        )
    if telem_cnt and telem_last is not None:
        tau = telem_last["tau"]
        rpy = telem_last["rpy"]
        mask = telem_last["valid_mask"]
        n_valid = sum(1 for j in range(8) if mask & (1 << j))
        imu_ok = bool(mask & (1 << 8))
        telem_hz = telem_cnt / args.duration if args.duration > 0 else 0.0
        print(
            f"TELEM(9889): {telem_cnt}개 수신 ({telem_hz:.1f} Hz, 기대 {TELEM_EXPECT_HZ:.0f})"
            f"  모터 유효 {n_valid}/8  IMU {'OK' if imu_ok else '미수신'}"
            + "  마지막 tau[N·m]: "
            + " ".join(f"{v:+.2f}" for v in tau)
            + f"  rpy: [{rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}]"
        )
        warn_telem_rate(telem_hz)
        report_telem_echo(telem_ticks, telem_clamp, telem_last)
        if n_valid < 8:
            print("  ⚠ 모터 상태 미유효 — RobotEmbedded 기동/EtherCAT/모터 전원 확인 (링크는 정상)")
    else:
        # 2026-08-14: 의미가 뒤집혔다. r2s_udp.unpack_policy_telem 은 legacy TELEM(120 B·116 B)도
        # 받으므로 **구버전 러너는 여전히 파싱된다**(규약 버전 0 으로). 이제 크기 불일치로 수신이
        # 끊기는 쪽은 러너가 r2s_udp.py 보다 **최신**일 때다 — 러너가 121 B 를 보내는데 이 워크스테이션의
        # r2s_udp.py 가 아직 120 B 만 알면 전부 거부된다.
        print(
            "TELEM(9889): 수신 없음 — real_runner 미기동/네트워크 두절이거나, gui monitor가 포트 점유 중,\n"
            "  또는 real_runner가 이 워크스테이션의 r2s_udp.py보다 최신(패킷 크기 불일치) — 워크스테이션 쪽을 갱신할 것"
        )

    ok_pacing = 18.0 <= mean_dt <= 23.0
    ok_rx = rx_cnt >= 0.9 * expected_rx
    verdict = "PASS" if (ok_pacing and ok_rx and bad_pkts == 0) else "CHECK"
    print(f"\n[comm_check] 판정: {verdict} (페이싱 {'OK' if ok_pacing else 'NG'}, 수신율 {'OK' if ok_rx else 'NG'})")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
