#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""역구동 임피던스 프로브 — 발목을 손으로 움직일 때 무릎이 따라 도는가.

**무엇을 재는가**: sim 은 foot↔calf 전치를 calf 관절의 effort target 으로 주입한다. 그 결과
calf 를 무토크로 두고 foot 을 구동하면 **sim 에서는 무릎이 움직인다**(2026-08-14 실측: 중력 OFF,
전치 ON 6.33° / OFF 정확히 0.00°). 실기가 같은지는 확인되지 않았다 — 실기에서는 "모든 토크를 푼 채
foot 을 움직여도 무릎은 안 움직인다"고 보고됐다.

이 프로브는 그 차이를 **발목→무릎 경로의 임피던스**로 잰다. 전 관절 무토크(RELAX)로 두고 발을 손으로
움직이면서, 발목 관절각이 변하는 동안 **무릎 관절각이 얼마나 따라 도는지** 기록한다.

    Δq_calf ≈ 0   → 발목→무릎 경로가 고임피던스. sim 이 무릎을 과하게 움직이는 것.
    Δq_calf 유의  → 저임피던스. sim 의 전치 주입이 실기와 부합.

⚠ **이 테스트가 재지 않는 것**: 손으로 움직이는 것은 *운동을 강제*하는 것이지 알려진 토크를 거는 게
   아니다. 따라서 "전치 토크가 무릎에 도달하는가"를 직접 재지는 못한다. 그건 calf 무토크 + **foot
   드라이브에 토크 지령**으로 재야 한다. 다만 sim 의 자유-calf 거동이 곧 임피던스 가정이라, 이 프로브가
   그 가정을 직접 반증/확인할 수 있다.

준비
----
1. GUI(``r2s_bl_gui``)를 실기에 붙인 채 **RELAX(무토크)** 상태로 둔다 — 기동 기본 상태다.
2. ``monitor.py`` 는 **끈다** (같은 포트를 쓴다).
3. 이 스크립트를 실행하고, 화면 카운트다운 뒤 **발만** 손으로 천천히 왕복시킨다.

⚠ 교란 주의
   · **허벅지를 고정**할 것. 안 그러면 다리 전체가 흔들려 무릎각이 그것 때문에 변한다.
   · **정강이를 손으로 잡지 말 것** — 잡으면 무릎을 구속해서 답이 정해진다.
   · 중력토크가 무릎에 거의 안 걸리는 자세(정강이가 수직에 가까운)에서 할 것.
   · 눈으로 "안 움직인다"는 몇 도까지 놓친다 — 판정은 이 로그로 한다.

실행::

    python3 scripts/real2sim/r2s_biped_leg/backdrive_probe.py --leg HL --seconds 20
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motions  # noqa: E402
import r2s_udp  # noqa: E402

RAD2DEG = 180.0 / math.pi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=["HL", "HR"], default="HL", help="관측할 다리")
    ap.add_argument("--seconds", type=float, default=20.0, help="기록 길이 [s]")
    ap.add_argument("--countdown", type=float, default=3.0, help="시작 전 카운트다운 [s]")
    ap.add_argument("--port", type=int, default=r2s_udp.REAL_MON_PORT, help="수신 포트 (GUI 중계)")
    ap.add_argument("--out", default="", help="npz/json 저장 경로 (기본: 저장 안 함)")
    args = ap.parse_args()

    names = motions.JOINT_NAMES
    i_calf = names.index(f"{args.leg}_calf")
    i_foot = names.index(f"{args.leg}_foot")
    i_hip = names.index(f"{args.leg}_hip")
    i_thigh = names.index(f"{args.leg}_thigh")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", args.port))
    except OSError as e:
        raise SystemExit(
            f"포트 {args.port} 바인드 실패: {e}\n  monitor.py 가 떠 있으면 끄고 다시 실행할 것 (같은 포트를 쓴다)."
        ) from e
    sock.settimeout(0.5)

    print("=" * 78)
    print(f"역구동 임피던스 프로브 — {args.leg} 다리, {args.seconds:.0f} 초")
    print("=" * 78)
    print("전 관절 RELAX(무토크) 상태인지 확인하고, 허벅지를 고정한 채 **발만** 천천히 왕복시키세요.")
    print("정강이는 잡지 마세요 — 잡으면 무릎이 구속됩니다.\n")

    for k in range(int(args.countdown), 0, -1):
        print(f"  {k} ...", flush=True)
        time.sleep(1.0)
    print("  시작 — 발을 움직이세요\n", flush=True)

    rows: list[tuple[float, list[float]]] = []
    t_start = time.monotonic()
    last_print = 0.0
    while True:
        now = time.monotonic() - t_start
        if now >= args.seconds:
            break
        try:
            data, _ = sock.recvfrom(4096)
        except TimeoutError:
            continue
        m = r2s_udp.unpack_monitor(data)
        if m is None:
            continue
        rows.append((now, list(m["sim_q"])))  # real 중계는 sim_q 슬롯에 실기 q 가 담긴다
        if now - last_print > 0.5:
            last_print = now
            q = m["sim_q"]
            print(
                f"  t={now:5.1f}s  calf={q[i_calf] * RAD2DEG:+7.2f}°  foot={q[i_foot] * RAD2DEG:+7.2f}°",
                flush=True,
            )

    sock.close()
    if len(rows) < 10:
        raise SystemExit(
            f"표본이 {len(rows)} 개뿐이다 — GUI 가 실기에 붙어 중계 중인지 확인할 것.\n"
            "  (GUI 가 RELAX 라도 TELEM 은 계속 와야 한다)"
        )

    def span(idx: int) -> float:
        v = [r[1][idx] for r in rows]
        return (max(v) - min(v)) * RAD2DEG

    d_calf, d_foot = span(i_calf), span(i_foot)
    d_hip, d_thigh = span(i_hip), span(i_thigh)
    # raw(모터축) foot = 관절 foot + 관절 calf — 실제로 back-drive 된 foot 드라이브 양
    raw_foot = [(r[1][i_foot] + r[1][i_calf]) for r in rows]
    d_raw_foot = (max(raw_foot) - min(raw_foot)) * RAD2DEG
    ratio = d_calf / d_foot if d_foot > 1e-9 else float("nan")

    print("\n" + "=" * 78)
    print(f"표본 {len(rows)} 개 · {rows[-1][0]:.1f} 초")
    print(f"{'관절':16s}{'가동폭(°)':>12s}")
    for lbl, v in (
        (f"{args.leg}_calf (무릎)", d_calf),
        (f"{args.leg}_foot (발목)", d_foot),
        ("  └ raw foot 드라이브", d_raw_foot),
        (f"{args.leg}_hip (교란)", d_hip),
        (f"{args.leg}_thigh (교란)", d_thigh),
    ):
        print(f"{lbl:16s}{v:12.3f}")
    print(f"\n무릎/발목 비 = {ratio:.4f}")

    print("\n[판정]")
    if max(d_hip, d_thigh) > 0.3 * d_foot:
        print("  ⚠ hip/thigh 도 함께 움직였다 — 허벅지가 안 잡혀 다리 전체가 흔들렸을 수 있다.")
        print("    허벅지를 고정하고 다시 잴 것. 아래 판정은 신뢰하지 말 것.")
    if d_foot < 2.0:
        print("  ⚠ 발목이 거의 안 움직였다 — 더 크게 왕복시킬 것 (최소 10° 이상 권장).")
    elif ratio < 0.05:
        print("  → **무릎이 따라 돌지 않는다** (비 < 0.05).")
        print("    발목→무릎 경로가 고임피던스 ⇒ sim 이 자유 calf 에서 무릎을 과하게 움직이는 것이다.")
        print("    sim 의 전치 effort 주입 방식을 손봐야 한다.")
    elif ratio > 0.3:
        print("  → **무릎이 상당히 따라 돈다** (비 > 0.3).")
        print("    저임피던스 ⇒ sim 의 전치 주입이 실기와 부합한다.")
    else:
        print(f"  → 중간 (비 {ratio:.3f}). 반복 측정 권장 — 자세·속도를 바꿔 가며 3 회 이상.")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "leg": args.leg,
            "n": len(rows),
            "duration_s": rows[-1][0],
            "span_deg": {
                "calf": d_calf,
                "foot": d_foot,
                "raw_foot": d_raw_foot,
                "hip": d_hip,
                "thigh": d_thigh,
            },
            "ratio_calf_over_foot": ratio,
            "joint_names": names,
            "t": [r[0] for r in rows],
            "q": [r[1] for r in rows],
        }
        p.write_text(json.dumps(payload, indent=1))
        print(f"\n저장: {p}")


if __name__ == "__main__":
    main()
