# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""램프 npz에서 명령별 **상승 vs 하강** duty·달성률을 분리해 낸다.

`ramp_summary.py` 의 `per_rollout` 은 `range(1, n_up)` 으로 **상승만** 집계하므로 보행 개시
히스테리시스(상승은 실패, 하강은 성공)를 볼 수 없다. 이 스크립트는 두 방향을 나란히 낸다.

구간은 `vx_cmd` 트레이스에서 직접 잘라내므로 npz 스키마 차이(`ramp_s` 유무)를 흡수한다.
hold 구간의 뒤쪽 `tail` 비율만 써서 램프 전이의 과도응답을 배제한다.

`duty` = hold 구간에서 순간 `|jvel| RMS > 0.4` 인 스텝의 비율. 1.0=연속 보행, 0=정지 자세 유지.
`ach`  = median(vx) / 명령. 100% 초과는 **과잉**이며, 저속에서는 "천천히 못 걷는다"는 별개의 실패다.

사용:
    python _workspace/leg/ramp_onset_duty.py "라벨=글롭" ["라벨=글롭" ...] [--cmd 0.5]
"""

import argparse
import glob

import numpy as np

DUTY_JVEL_RMS = 0.4  # 보행 판정 임계 [rad/s]


def _segments(cmd, target, tol=1e-3, min_len=20):
    """`vx_cmd` 가 target 으로 유지되는 연속 구간들의 인덱스 배열."""
    idx = np.flatnonzero(np.abs(cmd - target) < tol)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [g for g in groups if g.size > min_len]


def probe(path, target, tail=0.6):
    """(방향, duty, 달성률) 목록. 방향은 명령 정점 기준 'up'/'down'."""
    d = np.load(path)
    cmd = np.asarray(d["vx_cmd"], dtype=float).ravel()
    vx = np.asarray(d["vx"], dtype=float).ravel()
    jvel = np.asarray(d["jvel"], dtype=float)
    peak = int(np.argmax(cmd))
    rows = []
    for g in _segments(cmd, target):
        g = g[int(len(g) * (1.0 - tail)) :]
        rms = np.sqrt((jvel[g] ** 2).mean(axis=1))
        rows.append(
            ("up" if g[0] < peak else "down", float((rms > DUTY_JVEL_RMS).mean()), float(np.median(vx[g])) / target)
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="라벨=글롭")
    ap.add_argument("--cmd", type=float, default=0.5, help="분석할 속도 명령 [m/s]")
    ap.add_argument("--tail", type=float, default=0.6, help="hold 구간 뒤쪽 사용 비율")
    args = ap.parse_args()

    print(f"cmd {args.cmd} m/s · duty = |jvel|RMS > {DUTY_JVEL_RMS} 비율 · ach = median(vx)/cmd\n")
    print(f"{'run':28s} {'dir':5s} {'duty':>6s} {'ach':>7s} {'n':>4s}")
    print("-" * 54)
    for spec in args.specs:
        label, pattern = spec.split("=", 1)
        rows = []
        for p in sorted(glob.glob(pattern)):
            rows += probe(p, args.cmd, args.tail)
        if not rows:
            print(f"{label:28s} (매칭된 npz 없음: {pattern})")
            continue
        for direction in ("up", "down"):
            x = [(duty, ach) for k, duty, ach in rows if k == direction]
            if not x:
                continue
            print(
                f"{label if direction == 'up' else '':28s} {direction:5s}"
                f" {np.mean([i[0] for i in x]):6.2f}"
                f" {100 * np.mean([i[1] for i in x]):6.0f}%"
                f" {len(x):4d}"
            )


if __name__ == "__main__":
    main()
