# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""저속(cmd 0.5) 실패가 '속도를 못 낸다'가 아니라 '정지에서 보행으로 진입을 못 한다'임을 보인다.

같은 정책·같은 명령인데 램프 상승(정지출발)과 하강(이동중)의 결과가 갈린다는 것이 핵심 증거다.
duty = |jvel|RMS > 0.4 인 스텝 비율. 1.0 이면 연속 보행, 0 이면 정지 자세 유지.
"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = [("8k", "ds14wcmd_it8000"), ("15k", "ds14wcmd_it15000"), ("25k", "ds14wcmd_it25000"),
        ("40k", "ds14wcmd_it40000"), ("50k", "ds14wcmd_it49999"),
        ("ds14\n(length)", "ds14_0810_final"), ("wcmd\n(ds18)", "wcmd_it40000")]
UP, DOWN = 1, 15  # vx_profile 인덱스: 상승 0.5 / 하강 0.5


def hold(d, idx):
    dt = float(d["dt"])
    span = int(round((float(d["hold_s"]) + float(d["ramp_s"])) / dt))
    rs = int(round(float(d["ramp_s"]) / dt))
    s, e = idx * span + rs, min((idx + 1) * span, len(d["vx"]))
    w = np.sqrt((d["jvel"] ** 2).mean(axis=1))
    return float((w[s:e] > 0.4).mean()), float(np.median(d["vx"][s:e])) / 0.5


def collect():
    out = {}
    for lab, pat in RUNS:
        rows = []
        for f in sorted(glob.glob(f"_workspace/leg/{pat}_*/ramp_data.npz")):
            d = np.load(f)
            rows.append((hold(d, UP), hold(d, DOWN)))
        out[lab] = rows
    return out


data = collect()
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

# (a) 상승 vs 하강 달성률
x = np.arange(len(RUNS))
up = [100 * np.median([r[0][1] for r in data[l]]) for l, _ in RUNS]
dn = [100 * np.median([r[1][1] for r in data[l]]) for l, _ in RUNS]
ax[0].bar(x - 0.2, up, 0.4, label="ascending (from rest)", color="#c0392b")
ax[0].bar(x + 0.2, dn, 0.4, label="descending (already moving)", color="#2980b9")
ax[0].set_xticks(x); ax[0].set_xticklabels([l for l, _ in RUNS], fontsize=8)
ax[0].set_ylabel("cmd 0.5 achievement [%]"); ax[0].legend(fontsize=8)
ax[0].set_title("(a) Same policy, same command 0.5 m/s\nonly the approach differs")
ax[0].grid(alpha=0.3, axis="y")

# (b) 이봉성: duty vs 달성률
for lab, _ in RUNS:
    u = np.array([r[0] for r in data[lab]])
    ax[1].scatter(u[:, 0], 100 * u[:, 1], s=60, alpha=0.8, label=lab.replace("\n", " "))
    v = np.array([r[1] for r in data[lab]])
    ax[1].scatter(v[:, 0], 100 * v[:, 1], s=30, marker="x", color="#2980b9", alpha=0.6)
ax[1].set_xlabel("stepping duty  (fraction of steps with |jvel|RMS > 0.4)")
ax[1].set_ylabel("achievement [%]")
ax[1].set_title("(b) Achievement is set by whether it steps at all\n(x = descending: every rollout at duty ~1.0)")
ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=0.3)

# (c) 대표 시계열
d = np.load("_workspace/leg/ds14wcmd_it49999_2/ramp_data.npz")
t = np.arange(len(d["vx"])) * float(d["dt"])
w = np.sqrt((d["jvel"] ** 2).mean(axis=1))
ax[2].plot(t, d["vx_cmd"], "k--", lw=1.0, label="command")
ax[2].plot(t, d["vx"], color="#c0392b", lw=0.9, label="actual vx")
ax[2].plot(t, w, color="#7f8c8d", lw=0.6, alpha=0.6, label="|jvel| RMS")
_s, _e = 1 * 250 + 100, 2 * 250
ax[2].axvspan(t[_s], t[_e], color="#c0392b", alpha=0.15)
ax[2].annotate("cmd 0.5 up:\nSTANDS", (t[(_s + _e) // 2], 4.6), color="#c0392b",
               fontsize=8, ha="center", fontweight="bold")
_s, _e = 15 * 250 + 100, min(16 * 250, len(t) - 1)
ax[2].axvspan(t[_s], t[_e], color="#2980b9", alpha=0.15)
ax[2].annotate("cmd 0.5 down:\nWALKS", (t[(_s + _e) // 2], 4.6), color="#2980b9",
               fontsize=8, ha="center", fontweight="bold")
ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("[m/s]  /  [rad/s]")
ax[2].set_title("(c) 50k rollout 2: same 0.5 m/s command,\nopposite behaviour depending on approach")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

plt.tight_layout()
p = "reports/leg_imitation/_comparisons/lowspeed_gait_onset_hysteresis/figures/gait_onset_hysteresis.png"
plt.savefig(p, dpi=150)
print("saved", p)
