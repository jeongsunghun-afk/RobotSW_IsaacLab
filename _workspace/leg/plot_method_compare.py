# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""imitation 방법 비교 — 능력 축을 분리해서 그린다.

단일 평균으로 순위를 매기면 안 되는 이유가 이 그림의 요지다. 정상 주행(steady)과 정지 출발(start)은
서로 다른 능력이고, 자세 대칭성(skew)은 둘 중 어느 것으로도 드러나지 않는다.
"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIRS = [("FL_hip_joint", "FR_hip_joint"), ("HL_hip_joint", "HR_hip_joint")]

# (라벨, 글롭, 계열)  계열: base=구프로토콜 재측정, mod=현대
METHODS = [
    ("torque1e5\n(smr)", "M_torqsmr", "reset=random"),
    ("nosym", "M_nosym", "reset=random"),
    ("stand+dz", "M_standdz", "reset=random_stand"),
    ("trotmirror", "M_trotmir", "reset=random_stand"),
    ("ds14", "ds14_0810_final", "reset=random_stand"),
    ("ds14\n+vmax32", "ds14_vmax32_final", "reset=random_stand"),
    ("trot110150", "trot110150_final", "reset=random_stand"),
    ("trot110150\n+wcmd", "wcmd_final", "reset=random_stand"),
    ("waistfix", "waistfix_final", "reset=random_stand"),
    ("waistfix\n+wcmd", "waistfix_wcmd_final", "reset=random_stand"),
    ("ds14+wcmd", "ds14wcmd_it49999", "reset=random"),
]
COL = {"reset=random": "#c0392b", "reset=random_stand": "#2980b9"}


def per_rollout(path):
    d = np.load(path)
    prof = d["vx_profile"]
    dt = float(d["dt"])
    span = int(round((float(d["hold_s"]) + float(d["ramp_s"])) / dt))
    rs = int(round(float(d["ramp_s"]) / dt))
    nm = [str(x) for x in d["joint_names"]]
    w = np.sqrt((d["jvel"] ** 2).mean(axis=1))
    n = len(d["vx"])
    lv, ach, sk = [], [], []
    for i in range(1, (len(prof) + 1) // 2):
        s, e = min(i * span + rs, n), min((i + 1) * span, n)
        if s >= e:
            continue
        lv.append(float(prof[i]))
        ach.append(float(np.median(d["vx"][s:e])) / float(prof[i]))
        j = d["jpos"][s:e]
        sk.append(abs(j[:, nm.index(PAIRS[0][0])].mean() + j[:, nm.index(PAIRS[0][1])].mean())
                  + abs(j[:, nm.index(PAIRS[1][0])].mean() + j[:, nm.index(PAIRS[1][1])].mean()))
    lv, ach = np.array(lv), np.array(ach)
    s, e = min(span + rs, n), min(2 * span, n)
    return dict(lv=lv, ach=ach, steady=float(np.mean(ach[(lv >= 2.0) & (lv <= 4.0)])),
                start=float(np.median(d["vx"][s:e])) / 0.5,
                duty=float((w[s:e] > 0.4).mean()), skew=float(np.mean(sk)))


def collect():
    out = []
    for lab, pat, grp in METHODS:
        R = [per_rollout(p) for p in sorted(glob.glob(f"_workspace/leg/{pat}_*/ramp_data.npz"))]
        if not R:
            print(f"  (없음) {lab} <- {pat}")
            continue
        out.append((lab, grp, dict(
            steady=np.median([r["steady"] for r in R]), start=np.median([r["start"] for r in R]),
            duty=np.median([r["duty"] for r in R]), skew=np.median([r["skew"] for r in R]),
            lv=R[0]["lv"], ach=np.median([r["ach"] for r in R], axis=0), n=len(R))))
    return out


D = collect()
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
x = np.arange(len(D))

# (a) 세 축을 나란히
w = 0.27
ax[0].bar(x - w, [100 * d[2]["steady"] for d in D], w, label="steady (cmd 2.0-4.0)", color="#27ae60")
ax[0].bar(x, [100 * d[2]["start"] for d in D], w, label="start (cmd 0.5 from rest)", color="#c0392b")
ax[0].bar(x + w, [1000 * d[2]["skew"] for d in D], w, label="skew x1000 (lower=better)", color="#8e44ad")
ax[0].set_xticks(x)
ax[0].set_xticklabels([d[0] for d in D], fontsize=7, rotation=45, ha="right")
ax[0].set_ylabel("[%]  /  skew x1000")
ax[0].set_title("(a) Three capabilities do not move together")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=0.3, axis="y")

# (b) steady vs start — reset 전략으로 색 구분
for lab, grp, d in D:
    ax[1].scatter(100 * d["steady"], 100 * d["start"], s=90, color=COL[grp],
                  edgecolor="k", linewidth=0.5, zorder=3)
    ax[1].annotate(lab.replace("\n", " "), (100 * d["steady"], 100 * d["start"]),
                   fontsize=6.5, xytext=(4, 3), textcoords="offset points")
for g, c in COL.items():
    ax[1].scatter([], [], s=90, color=c, edgecolor="k", label=g)
ax[1].set_xlabel("steady tracking, cmd 2.0-4.0 [%]")
ax[1].set_ylabel("start from rest, cmd 0.5 [%]")
ax[1].set_title("(b) Start-from-rest tracks the reset strategy,\nnot the tracking quality")
ax[1].legend(fontsize=7)
ax[1].grid(alpha=0.3)

# (c) 명령속도별 달성 곡선
for lab, grp, d in D:
    ax[2].plot(d["lv"], 100 * d["ach"], marker="o", ms=3, lw=1.2,
               label=f"{lab.replace(chr(10), ' ')} (n={d['n']})")
ax[2].set_xlabel("commanded vx [m/s]")
ax[2].set_ylabel("achievement [%]")
ax[2].set_title("(c) Per-command achievement (ascending)")
ax[2].legend(fontsize=6, ncol=2)
ax[2].grid(alpha=0.3)

plt.tight_layout()
p = "reports/leg_imitation/_comparisons/imitation_method_comparison/figures/method_comparison.png"
plt.savefig(p, dpi=150)
print("saved", p)
