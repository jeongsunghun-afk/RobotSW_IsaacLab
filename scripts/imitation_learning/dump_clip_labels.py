# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""참조 모션 클립 → 속도·yaw·gait 라벨 표와 명령 범위 커버리지 그림을 만든다.

조건부 AMP discriminator(``amp_cond_mode``)가 expert 샘플에 붙이는 라벨이 바로 이 표의
``mean_speed`` / ``mean_yaw_rate`` 다. 시뮬레이터 없이 motion_lib 만 읽는다.

.. code-block:: bash

    python scripts/imitation_learning/dump_clip_labels.py --task leg --out reports/.../metrics
    python scripts/imitation_learning/dump_clip_labels.py --task go2 --v_max 4.0
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TASK_DIR = os.path.join(_REPO, "source", "isaaclab_tasks", "isaaclab_tasks", "direct")
_DEFAULT_MOTION = {
    "leg": os.path.join(_TASK_DIR, "leg_imitation_tracking", "imitation", "merged_leg_pkl"),
    "go2": os.path.join(_TASK_DIR, "go2_imitation_tracking", "imitation", "smr_mirror_pkl"),
}
_GAIT_RE = re.compile(r"(walk_turn|walk|trot|pace|gallop|bound|run|ramp|stand)", re.IGNORECASE)


def _gait_from_name(name: str) -> str:
    m = _GAIT_RE.search(name)
    return m.group(1).lower() if m else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["leg", "go2"], required=True)
    ap.add_argument("--motion_file", default=None, help="pkl 디렉터리 또는 파일. 기본은 task 기본 데이터셋")
    ap.add_argument("--v_max", type=float, default=4.0, help="명령 상한 [m/s] (커버리지 축)")
    ap.add_argument("--weight_mode", default="length", choices=["length", "uniform", "command_uniform"])
    ap.add_argument("--out", default=None, help="출력 디렉터리 (csv + png). 기본은 현재 디렉터리")
    args = ap.parse_args()

    motion_file = args.motion_file or _DEFAULT_MOTION[args.task]
    if args.task == "leg":
        from isaaclab_tasks.direct.leg_imitation_tracking.motion_lib import LegMotionLib as Lib
    else:
        from isaaclab_tasks.direct.go2_imitation_tracking.motion_lib import Go2MotionLib as Lib

    lib = Lib(motion_files=motion_file, device="cpu")
    if args.weight_mode == "uniform":
        lib._motion_weights = torch.full_like(lib._motion_weights, 1.0 / lib.num_motions)
    elif args.weight_mode == "command_uniform":
        lib.set_motion_weights_command_uniform(args.v_max)

    names = lib.motion_names
    speeds = lib.motion_mean_speeds
    yaws = lib.motion_mean_yaw_rates
    lengths = lib._motion_lengths
    frames = lib._motion_num_frames
    weights = lib._motion_weights

    out_dir = args.out or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"clip_labels_{args.task}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "gait", "mean_speed_mps", "mean_yaw_rate_rps", "length_s", "frames", "weight"])
        for i, n in enumerate(names):
            w.writerow(
                [
                    n,
                    _gait_from_name(n),
                    f"{float(speeds[i]):.4f}",
                    f"{float(yaws[i]):.4f}",
                    f"{float(lengths[i]):.3f}",
                    int(frames[i]),
                    f"{float(weights[i]):.5f}",
                ]
            )
    print(f"[dump_clip_labels] {len(names)} clips → {csv_path}")
    for i, n in enumerate(names):
        gait, sp, yw, wt = _gait_from_name(n), float(speeds[i]), float(yaws[i]), 100 * float(weights[i])
        print(f"  {n:32s} {gait:10s} {sp:5.2f} m/s  {yw:+5.2f} rad/s  w={wt:5.2f}%")

    # ── 커버리지 그림: 속도축 위 클립 위치(가중치=마커 크기), 명령 범위 [0, v_max] 음영 ──
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[dump_clip_labels] matplotlib 없음 — 그림 생략")
        return

    gaits = sorted({_gait_from_name(n) for n in names})
    cmap = plt.get_cmap("tab10")
    color = {g: cmap(i % 10) for i, g in enumerate(gaits)}
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax0.axvspan(0.0, args.v_max, color="0.92", label=f"command range [0, {args.v_max:g}]")
    for i, n in enumerate(names):
        g = _gait_from_name(n)
        ax0.scatter(
            float(speeds[i]), float(yaws[i]), s=60 + 2000 * float(weights[i]), color=color[g], alpha=0.75, edgecolor="k"
        )
        ax0.annotate(n, (float(speeds[i]), float(yaws[i])), fontsize=6, xytext=(3, 3), textcoords="offset points")
    for g in gaits:
        ax0.scatter([], [], color=color[g], label=g)
    ax0.set_ylabel("clip mean yaw rate [rad/s]")
    ax0.set_title(
        f"{args.task}: reference clips on the command axes (marker size = sampling weight, mode={args.weight_mode})"
    )
    ax0.legend(fontsize=7, ncol=3)
    ax0.grid(alpha=0.3)

    # 속도축 expert 질량 히스토그램 (가중치 합)
    bins = torch.linspace(0.0, max(args.v_max, float(speeds.max()) + 0.25), 25)
    mass = torch.zeros(len(bins) - 1)
    for i in range(len(names)):
        b = int(torch.bucketize(speeds[i], bins).clamp(1, len(bins) - 1)) - 1
        mass[b] += weights[i]
    ax1.bar(bins[:-1].numpy(), mass.numpy(), width=float(bins[1] - bins[0]), align="edge", color="0.4")
    ax1.axvspan(0.0, args.v_max, color="0.92", zorder=0)
    ax1.set_xlabel("clip mean speed [m/s]")
    ax1.set_ylabel("expert mass")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"cmd_coverage_{args.task}.png")
    fig.savefig(png_path, dpi=130)
    print(f"[dump_clip_labels] figure → {png_path}")


if __name__ == "__main__":
    main()
