# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""`gait_survey_multienv.py` 산출물 → **(속도 × 선회) 격자별 보행 분포**.

램프는 명령 공간의 직진 선 하나만 훑으므로, 거기서 pace 만 나왔다고 정책이 pace 만 한다고
말할 수 없다. 여기서는 env 자체 분포로 뿌린 수천 개 표본을 명령 격자에 얹어, 어떤 명령에서
어떤 보행이 나오는지 지도를 만든다.

버리는 표본 세 가지 — 남기면 전부 "뭔가로" 분류돼 지도를 오염시킨다:
  리셋 걸침 : 에피소드 경계에서 위상이 끊긴다
  낙상      : 누워 버둥거려도 순시 위상은 나온다
  명령 변동 : 창 안에서 명령이 바뀌면 어느 격자에 넣을지 모호하다

사용:
    ./isaaclab.sh -p _workspace/leg/gait_survey_analyze.py <npz> [--win_s 2.0]
"""

import argparse
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gait_classify import LEGS, MIN_AMP_RAD, _circ_mean_frac, _classify, _inst_phase  # noqa: E402

FALL_H = 0.35  # 이보다 낮으면 낙상으로 본다 [m] (termination_height)
MIN_CYCLES = 3.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--win_s", type=float, default=2.0, help="분류 창 길이 [s] — 끝에서 이만큼 쓴다")
    ap.add_argument("--vx_bins", type=float, nargs="+", default=[0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
    ap.add_argument("--yaw_bins", type=float, nargs="+", default=[-1.5, -0.5, 0.5, 1.5])
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    jpos = np.asarray(d["jpos"], dtype=np.float32)  # [T, N, J]
    vxc, yawc = np.asarray(d["vx_cmd"]), np.asarray(d["yaw_cmd"])
    hgt, elen = np.asarray(d["height"]), np.asarray(d["ep_len"])
    names = [str(s) for s in d["joint_names"]]
    dt = float(d["dt"])
    fs = 1.0 / dt
    T, N, _ = jpos.shape
    w = int(args.win_s * fs)
    sl = slice(T - w, T)  # 마지막 창

    idx = {lg: names.index(f"{lg}_thigh_joint") for lg in LEGS}

    reset = (np.diff(elen[sl], axis=0) < 0).any(axis=0)  # 창 안에서 리셋
    fell = (hgt[sl] < FALL_H).any(axis=0)
    cmd_moved = (vxc[sl].std(axis=0) > 1e-3) | (yawc[sl].std(axis=0) > 1e-3)
    keep = ~(reset | fell | cmd_moved)
    print(f"표본 {N} → 사용 {keep.sum()} (리셋 {reset.sum()} · 낙상 {fell.sum()} · 명령변동 {cmd_moved.sum()} 제외)")
    if keep.sum() == 0:
        return 1

    labels = np.array(["-"] * N, dtype=object)
    for e in np.flatnonzero(keep):
        sig = {lg: jpos[sl, e, idx[lg]] for lg in LEGS}
        if max(float(np.ptp(s)) for s in sig.values()) < MIN_AMP_RAD:
            labels[e] = "stand"
            continue
        ph = {lg: _inst_phase(sig[lg], fs) for lg in LEGS}
        freq = float(np.polyfit(np.arange(w) / fs, ph["FL"], 1)[0]) / (2 * np.pi)
        if abs(freq) * args.win_s < MIN_CYCLES:
            continue  # '-' 유지
        phi = {lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS}
        labels[e] = _classify(phi)

    vx_e, yaw_e = vxc[T - 1], yawc[T - 1]
    print("\n[명령 격자별 gallop 비율]  행=vx [m/s], 열=yaw [rad/s]")
    print("칸 = gallop% / pace% (n).  최빈값만 쓰면 소수 보행이 안 보여 비율로 낸다.\n")
    hdr = "vx \\ yaw".ljust(14) + "".join(
        f"[{args.yaw_bins[j]:+.1f},{args.yaw_bins[j + 1]:+.1f})".rjust(22) for j in range(len(args.yaw_bins) - 1)
    )
    print(hdr)
    print("-" * len(hdr))
    for i in range(len(args.vx_bins) - 1):
        lo, hi = args.vx_bins[i], args.vx_bins[i + 1]
        line = f"[{lo:.1f},{hi:.1f})".ljust(14)
        for j in range(len(args.yaw_bins) - 1):
            ylo, yhi = args.yaw_bins[j], args.yaw_bins[j + 1]
            m = keep & (vx_e >= lo) & (vx_e < hi) & (yaw_e >= ylo) & (yaw_e < yhi)
            labs = [x for x in labels[m] if x != "-"]
            if len(labs) < 5:
                line += f"n={len(labs)}".rjust(22)
                continue
            c = Counter(labs)
            gl = 100 * (c["gallop"] + c["bound"]) / len(labs)
            pc = 100 * c["pace"] / len(labs)
            line += f"{gl:.0f}% / {pc:.0f}% (n={len(labs)})".rjust(22)
        print(line)

    labs_all = [x for x in labels[keep] if x != "-"]
    print(f"\n[전체 분포] n={len(labs_all)}")
    for k, v in Counter(labs_all).most_common():
        print(f"  {k:8s} {v:5d}  {100 * v / len(labs_all):5.1f}%")

    # gallop/bound 가 나온 표본의 명령 범위
    gb = np.array([labels[e] in ("gallop", "bound") for e in range(N)]) & keep
    if gb.any():
        print(
            f"\n★ gallop/bound n={gb.sum()} — vx {vx_e[gb].min():.2f}~{vx_e[gb].max():.2f} "
            f"(중앙 {np.median(vx_e[gb]):.2f}) · |yaw| {np.abs(yaw_e[gb]).min():.2f}~{np.abs(yaw_e[gb]).max():.2f} "
            f"(중앙 {np.median(np.abs(yaw_e[gb])):.2f})"
        )
    else:
        print("\n★ gallop/bound 표본 0개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
