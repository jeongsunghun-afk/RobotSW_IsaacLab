# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""arch_scaling_study 판정 그림 — (a) priv vs history A/B, (b) 보상 gradient.

왼쪽:  같은 체크포인트에서 actor latent 경로만 바꾼 램프 A/B (4 repeat).
       증류를 제거한 priv 경로에서도 cmd 0.5 상승 실패가 남는다 → 증류 오차 가설 기각.
오른쪽: 정지(v=0)가 얻는 task reward 비율을 vel_err_scale 별로. cmd 0.5 에서 baseline(0.5)은
       0.88 로 거의 만점이라 걸을 유인이 없다. 1.0/1.5 arm 이 이 값을 끌어내린다.

라벨은 영어로 둔다 — 렌더 환경에 CJK 폰트가 없어 한글이 □ 로 깨진다.

    python _workspace/leg/plot_arch_scaling_verdict.py
"""

import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AB_DIR = "_workspace/leg/ab_privhist"
OUT = "reports/leg_imitation/_comparisons/arch_scaling_study/figures/arch_scaling_verdict.png"
# 3.2 는 vx_max 클립 지점이라 hold 구간이 짧아 세그먼트가 안 잡힌다 — 제외.
CMDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DUTY_JVEL_RMS = 0.4


def _segments(cmd, target, tol=1e-3, min_len=20):
    idx = np.flatnonzero(np.abs(cmd - target) < tol)
    if idx.size == 0:
        return []
    return [g for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if g.size > min_len]


def duty_by_cmd(pattern, tail=0.6):
    """{cmd: (상승 duty 평균, 하강 duty 평균)}"""
    rows = {c: {"up": [], "down": []} for c in CMDS}
    for path in sorted(glob.glob(pattern)):
        d = np.load(path)
        cmd = np.asarray(d["vx_cmd"], dtype=float).ravel()
        jvel = np.asarray(d["jvel"], dtype=float)
        peak = int(np.argmax(cmd))
        for c in CMDS:
            for g in _segments(cmd, c):
                g = g[int(len(g) * (1.0 - tail)) :]
                rms = np.sqrt((jvel[g] ** 2).mean(axis=1))
                rows[c]["up" if g[0] < peak else "down"].append(float((rms > DUTY_JVEL_RMS).mean()))
    return {c: (np.mean(v["up"]) if v["up"] else np.nan, np.mean(v["down"]) if v["down"] else np.nan)
            for c, v in rows.items()}


hist = duty_by_cmd(f"{AB_DIR}/hist_*/ramp_data.npz")
priv = duty_by_cmd(f"{AB_DIR}/priv_*/ramp_data.npz")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# ── (a) A/B ───────────────────────────────────────────────────────────────
x = np.arange(len(CMDS))
for arm, data, color in (("history latent (deploy path)", hist, "tab:blue"),
                         ("priv latent (training path)", priv, "tab:red")):
    ax1.plot(x, [data[c][0] for c in CMDS], "o-", color=color, label=f"{arm} — ramp UP")
    ax1.plot(x, [data[c][1] for c in CMDS], "s--", color=color, alpha=0.45, label=f"{arm} — ramp DOWN")
ax1.axvspan(-0.35, 0.35, color="gold", alpha=0.25, zorder=0)
ax1.annotate("failure confined\nto cmd 0.5", xy=(0.0, 0.35), xytext=(1.2, 0.35),
             arrowprops=dict(arrowstyle="->", color="0.35"), color="0.25", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{c:g}" for c in CMDS])
ax1.set_xlabel("velocity command [m/s]")
ax1.set_ylabel("gait duty  (fraction of stepping steps)")
ax1.set_ylim(-0.05, 1.08)
ax1.set_title("(a) Removing distillation error does not fix onset\n"
              "same checkpoint, 4 repeats, only the actor latent path differs", fontsize=10)
ax1.legend(fontsize=7.5, loc="lower right")
ax1.grid(alpha=0.3)

# ── (b) 보상 gradient ─────────────────────────────────────────────────────
cc = np.linspace(0.0, 3.2, 200)
for k, style in ((0.5, "-"), (1.0, "--"), (1.5, "-.")):
    ax2.plot(cc, np.exp(-k * cc**2), style, label=f"vel_err_scale = {k}"
             + ("  (baseline)" if k == 0.5 else "  (new arm)"))
ax2.axvline(0.5, color="0.6", lw=0.8)
ax2.axvline(1.0, color="0.6", lw=0.8)
for k in (0.5, 1.0, 1.5):
    ax2.plot([0.5], [math.exp(-k * 0.25)], "o", color="k", ms=4, zorder=5)
ax2.annotate(f"standing already earns {math.exp(-0.5*0.25):.3f}\nof max task reward at cmd 0.5",
             xy=(0.5, math.exp(-0.5 * 0.25)), xytext=(1.15, 0.70),
             arrowprops=dict(arrowstyle="->", color="0.35"), fontsize=9, color="0.25")
ax2.annotate("cmd 1.0 (0.607): 19/19 rollouts walk", xy=(1.0, math.exp(-0.5 * 1.0)),
             xytext=(1.30, 0.42), arrowprops=dict(arrowstyle="->", color="0.35"),
             fontsize=9, color="0.25")
ax2.set_ylim(-0.03, 1.16)
ax2.set_xlabel("velocity command [m/s]")
ax2.set_ylabel("task reward while standing  exp(-k c^2)")
ax2.set_title("(b) Why only cmd 0.5 fails: the reward is nearly flat there\n"
              "and how the two new arms steepen it", fontsize=10)
ax2.legend(fontsize=8.5)
ax2.grid(alpha=0.3)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
for c in CMDS:
    print(f"cmd {c:<4g} hist up {hist[c][0]:.2f} down {hist[c][1]:.2f} | "
          f"priv up {priv[c][0]:.2f} down {priv[c][1]:.2f}")
