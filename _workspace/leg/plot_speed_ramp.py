# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""speed_ramp_record.py 가 저장한 ramp_data.npz 를 4장의 png plot 으로 그린다.

Run:
    python _workspace/leg/plot_speed_ramp.py --npz _workspace/leg/ramp_out/ramp_data.npz
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--npz", type=str, default="_workspace/leg/ramp_out/ramp_data.npz")
args = parser.parse_args()

d = np.load(args.npz, allow_pickle=True)
t = d["t"]
names = [str(n) for n in d["joint_names"]]
limit = d["effort_limit"]
hold_s = float(d["hold_s"])
profile = d["vx_profile"]
out_dir = os.path.dirname(args.npz)

# 관절을 다리별로 그룹화 (보기 쉽게). waist 는 단독.
GROUPS = {
    "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint"],
    "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint"],
    "HL": ["HL_hip_joint", "HL_thigh_joint", "HL_calf_joint", "HL_foot_joint"],
    "HR": ["HR_hip_joint", "HR_thigh_joint", "HR_calf_joint", "HR_foot_joint"],
    "waist": ["FB_waist_joint"],
}
JCOLOR = {"hip": "tab:blue", "thigh": "tab:orange", "calf": "tab:green", "foot": "tab:red", "waist": "tab:purple"}


def jtype(name):
    for k in JCOLOR:
        if k in name:
            return k
    return "waist"


# 한 단계 = 명령 상승(ramp_s) + 유지(hold_s). ramp_s 를 무시하면 경계선이 어긋난다.
ramp_s = float(d["ramp_s"]) if "ramp_s" in d.files else 0.0
stage_s = ramp_s + hold_s


def shade_stages(ax):
    """각 vx 명령 단계 경계를 세로선 + 상단에 명령값 라벨로 표시."""
    for i in range(len(profile)):
        ax.axvline(i * stage_s, color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(t[0], t[-1])


def stage_label_axis(ax):
    for i, v in enumerate(profile):
        ax.text(
            (i + 0.5) * stage_s, ax.get_ylim()[1] * 0.98, f"{v:.1f}",
            ha="center", va="top", fontsize=7, color="0.4",
        )


# ── 1. 속도 추종 ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes[0].plot(t, d["vx_cmd"], "k--", lw=1.5, label="vx command")
axes[0].plot(t, d["vx"], "tab:blue", lw=1.2, label="vx actual")
axes[0].set_ylabel("x velocity [m/s]")
axes[0].legend(loc="upper right")
axes[0].set_title("Leg-Imitation-Tracking: speed command ramp (0->4->0 m/s) tracking")
stage_label_axis(axes[0])

axes[1].axhline(0.0, color="k", ls="--", lw=1.0, label="vy command (=0)")
axes[1].plot(t, d["vy"], "tab:green", lw=1.2, label="vy actual")
axes[1].set_ylabel("y velocity [m/s]")
axes[1].legend(loc="upper right")

if "yaw_cmd" in d.files:
    axes[2].plot(t, d["yaw_cmd"], "k--", lw=1.2, label="yaw rate command")
else:
    axes[2].axhline(0.0, color="k", ls="--", lw=1.0, label="yaw command (=0)")
axes[2].plot(t, d["yaw"], "tab:red", lw=1.2, label="yaw rate actual")
axes[2].set_ylabel("yaw rate [rad/s]")
axes[2].set_xlabel("time [s]")
axes[2].legend(loc="upper right")

for ax in axes:
    shade_stages(ax)
fig.tight_layout()
p1 = os.path.join(out_dir, "1_velocity_tracking.png")
fig.savefig(p1, dpi=130)
plt.close(fig)


# ── 관절 3종 (position / torque / velocity) 공통 그리기 ──────────
def plot_joint_grid(key, ylabel, title, fname, show_limit=False):
    data = d[key]  # [T, 17]
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for row, (grp, js) in enumerate(GROUPS.items()):
        ax = axes[row]
        for jn in js:
            idx = names.index(jn)
            ax.plot(t, data[:, idx], lw=1.0, color=JCOLOR[jtype(jn)], label=jtype(jn))
            if show_limit:
                lim = limit[idx]
                ax.axhline(lim, color=JCOLOR[jtype(jn)], ls=":", lw=0.7, alpha=0.6)
                ax.axhline(-lim, color=JCOLOR[jtype(jn)], ls=":", lw=0.7, alpha=0.6)
        ax.set_ylabel(f"{grp}\n{ylabel}")
        ax.legend(loc="upper right", ncol=4, fontsize=7)
        shade_stages(ax)
        ax.axhline(0.0, color="0.7", lw=0.5)
    axes[0].set_title(title)
    stage_label_axis(axes[0])
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


p2 = plot_joint_grid(
    "jpos", "pos [rad]",
    "Joint position [rad] - per leg, color = joint type (hip/thigh/calf/foot)",
    "2_joint_position.png",
)
p3 = plot_joint_grid(
    "jtau", "τ [N·m]",
    "Applied torque [N.m] - dotted = effort limit",
    "3_joint_torque.png", show_limit=True,
)
p4 = plot_joint_grid(
    "jvel", "ω [rad/s]",
    "Joint velocity [rad/s]",
    "4_joint_velocity.png",
)

outs = [p1, p2, p3, p4]

# ── 5. 주행 방향 유지 (heading) ──────────────────────────────────
# heading_hold 를 쓰면 목표 방향에서 벗어날 때마다 yaw rate 명령으로 되돌린다.
# 로봇이 "한쪽 방향으로만" 가는지는 world 평면 궤적과 heading 오차로 판정한다.
if "px" in d.files:
    px, py = d["px"], d["py"]
    yaw_ang, yaw_err = d["yaw_ang"], d["yaw_err"]
    hold = bool(d["heading_hold"]) if "heading_hold" in d.files else False
    tgt = float(d["heading_target"]) if "heading_target" in d.files else 0.0
    kp = float(d["heading_kp"]) if "heading_kp" in d.files else 0.0

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0])

    # (좌) world 평면 궤적 — 목표 방향 직선과 나란하면 직진 유지
    axt = fig.add_subplot(gs[:, 0])
    # 시작점 기준 상대 좌표, 목표 heading 축으로 회전해 "전진/횡변위"로 본다
    dx, dy = px - px[0], py - py[0]
    fwd = dx * np.cos(tgt) + dy * np.sin(tgt)
    lat = -dx * np.sin(tgt) + dy * np.cos(tgt)
    sc = axt.scatter(fwd, lat, c=t, cmap="viridis", s=6)
    axt.axhline(0.0, color="k", ls="--", lw=1.0, label="target line (heading)")
    axt.set_xlabel("forward [m] (along target heading)")
    axt.set_ylabel("lateral [m]")
    axt.set_aspect("equal", adjustable="datalim")
    axt.legend(loc="upper left")
    axt.grid(alpha=0.3)
    axt.set_title(
        f"XY path (world) — heading_hold={'ON' if hold else 'OFF'}"
        + (f" (kp={kp})" if hold else "")
        + f"\nfinal lateral {lat[-1]:+.2f} m, max |lateral| {np.abs(lat).max():.2f} m"
    )
    fig.colorbar(sc, ax=axt, label="time [s]")

    # (우) heading 각도 / 오차 / yaw rate 명령
    ax0 = fig.add_subplot(gs[0, 1])
    ax0.axhline(tgt, color="k", ls="--", lw=1.0, label=f"target {tgt:+.3f}")
    ax0.plot(t, yaw_ang, "tab:blue", lw=1.2, label="heading actual")
    ax0.set_ylabel("heading [rad]")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.set_title("heading hold")

    ax1 = fig.add_subplot(gs[1, 1], sharex=ax0)
    ax1.axhline(0.0, color="k", ls="--", lw=1.0)
    ax1.plot(t, yaw_err, "tab:red", lw=1.2)
    ax1.set_ylabel("heading error [rad]")
    ax1.text(
        0.02, 0.9, f"RMS {np.sqrt((yaw_err ** 2).mean()):.3f} rad, |max| {np.abs(yaw_err).max():.3f} rad",
        transform=ax1.transAxes, fontsize=8, va="top",
    )

    ax2 = fig.add_subplot(gs[2, 1], sharex=ax0)
    if "yaw_cmd" in d.files:
        ax2.plot(t, d["yaw_cmd"], "k--", lw=1.2, label="yaw rate cmd")
    ax2.plot(t, d["yaw"], "tab:green", lw=1.0, label="yaw rate actual")
    ax2.set_ylabel("yaw rate [rad/s]")
    ax2.set_xlabel("time [s]")
    ax2.legend(loc="upper right", fontsize=8)

    for ax in (ax0, ax1, ax2):
        shade_stages(ax)
    fig.tight_layout()
    p5 = os.path.join(out_dir, "5_heading_hold.png")
    fig.savefig(p5, dpi=130)
    plt.close(fig)
    outs.append(p5)

print("저장 완료:")
for p in outs:
    print(f"  {p}")
