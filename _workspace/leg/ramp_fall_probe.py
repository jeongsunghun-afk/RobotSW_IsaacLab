# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""램프 npz에서 **롤아웃이 도중에 넘어졌는지**와 그 지점을 판정한다.

`ramp_onset_duty.py` 와 `ramp_summary.py` 는 명령 단계별 hold 구간만 본다. 그래서 로봇이
정점 직후 넘어져 남은 구간을 통째로 누워서 보내면, 그 구간이 **"duty 0.00 = 정지 자세 유지"**
로 집계된다 — 서 있는 것과 누워 있는 것이 같은 숫자가 된다. 실제로 2026-08-28 `vel_err_scale`
판정에서 arm A 의 하강 duty 0.00 을 하마터면 "저속에서 안 걷는다"로 읽을 뻔했다.
(같은 함정의 sim 쪽 판본은 [[project_hindleg_backward_fall_never_terminates]] — 넘어져도
에피소드가 안 끝나 학습 분포가 오염된 사건이다. 여기서는 학습이 아니라 **평가**가 오염된다.)

판정은 세 가지를 함께 본다. 하나만으로는 "정상적으로 명령 0 에서 멈춘 것"과 구분이 안 된다.

  전진 정지 : 이후 남은 구간 전체의 전진이 `--min_adv` 미만이 되는 첫 스텝
  자세      : 종료 시점의 thigh/calf 각 — 접혀 있으면(≈ -1.0 rad) 서 있는 게 아니다
  heading   : 종료 yaw — 넘어지면 크게 돌아간다(정상 주행은 heading_hold 로 ±10° 이내)

★ 정지 시각이 명령 0 부근(램프 끝)이면 **정상 종료**다. 정점 근처면 붕괴다.

사용:
    ./isaaclab.sh -p _workspace/leg/ramp_fall_probe.py "라벨=글롭" ["라벨=글롭" ...]
"""

import argparse
import glob

import numpy as np

FOLDED_RAD = -0.6  # thigh/calf 가 이보다 접혀 있으면 기립 자세가 아니다 [rad]
YAW_LOST_DEG = 30.0  # heading_hold 가 걸린 정상 주행에서 나올 수 없는 편차 [deg]


def _probe(path: str, min_adv: float, tail: int):
    d = np.load(path)
    cmd = np.asarray(d["vx_cmd"], dtype=float).ravel()
    px = np.asarray(d["px"], dtype=float).ravel()
    t = np.asarray(d["t"], dtype=float).ravel()
    yaw = np.asarray(d["yaw_ang"], dtype=float).ravel()
    jpos = np.asarray(d["jpos"], dtype=float)
    names = [str(s) for s in d["joint_names"]]
    peak = int(np.argmax(cmd))

    stop = None
    for i in range(peak, len(px) - tail):
        if px[-1] - px[i] < min_adv:
            stop = i
            break

    end = slice(-tail, None)
    legs = [i for i, n in enumerate(names) if n.endswith(("_thigh_joint", "_calf_joint"))]
    folded = float(np.mean(jpos[end][:, legs].mean(axis=0) < FOLDED_RAD)) if legs else float("nan")
    yaw_end = float(np.degrees(yaw[end]).mean())
    return {
        "stop_t": None if stop is None else float(t[stop]),
        "stop_cmd": None if stop is None else float(cmd[stop]),
        "peak_t": float(t[peak]),
        "adv_post_peak": float(px[-1] - px[peak]),
        "folded_frac": folded,
        "yaw_end": yaw_end,
    }


def _verdict(r: dict) -> str:
    """넘어짐 / 정상 종료 / 애매 중 하나."""
    if r["stop_t"] is None:
        return "OK (끝까지 전진)"
    posture = r["folded_frac"] > 0.5 or abs(r["yaw_end"]) > YAW_LOST_DEG
    late = r["stop_cmd"] is not None and r["stop_cmd"] < 0.6
    if posture and not late:
        return "FALL"
    if late and not posture:
        return "OK (램프 끝 감속)"
    return "?? (증거 상충 — 영상 확인)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="라벨=글롭")
    ap.add_argument("--min_adv", type=float, default=0.3, help="이후 전진이 이 값[m] 미만이면 정지로 본다")
    ap.add_argument("--tail", type=int, default=300, help="종료 자세를 평균낼 마지막 스텝 수")
    args = ap.parse_args()

    print(f"folded = 종료 시 thigh/calf 가 {FOLDED_RAD} rad 보다 접힌 관절 비율 · yaw = 종료 heading\n")
    print(f"{'run':24s} {'stop_t':>7s} {'cmd':>5s} {'adv':>8s} {'folded':>7s} {'yaw':>8s}  판정")
    print("-" * 76)
    for spec in args.specs:
        label, pattern = spec.split("=", 1)
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"{label:24s} (매칭된 npz 없음: {pattern})")
            continue
        print(f"{label}  (peak t={_probe(hits[0], args.min_adv, args.tail)['peak_t']:.1f}s)")
        for p in hits:
            r = _probe(p, args.min_adv, args.tail)
            st = "  -  " if r["stop_t"] is None else f"{r['stop_t']:7.1f}"
            sc = "  -  " if r["stop_cmd"] is None else f"{r['stop_cmd']:5.2f}"
            print(
                f"  {p.split('/')[-2]:22s} {st} {sc} {r['adv_post_peak']:7.2f}m"
                f" {100 * r['folded_frac']:6.0f}% {r['yaw_end']:+7.1f}°  {_verdict(r)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
