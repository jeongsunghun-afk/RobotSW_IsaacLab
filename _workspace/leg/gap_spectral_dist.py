# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task A 보강 — 분류기 없는 분포 거리 (C2ST 가 음성 대조에서 무의미로 판명된 뒤).

per-frame C2ST 는 같은 정책 시드 간 대조에서도 0.86~0.99 가 나와 이 표본 크기에서
분리도를 판정할 수 없다. 대신 marginal 1-D Wasserstein 과 RBF MMD 로
"저역통과 후 B' 가 A' 수준으로 가까워지는가"를 직접 잰다.
"""

from __future__ import annotations

import json

import numpy as np

from gap_spectral import (  # noqa: E402
    FS_POL, FS_REF, MATCHES, POLICIES, SEEDS, TRIM_S,
    load_ramp, lowpass, read_clip, resample_to, stage_slices,
)

SCRATCH = "/tmp/claude-1002/-home-lgb-IsaacLab-6-0/0acd1c66-e03c-47df-9285-08e899ea3b28/scratchpad"


def w1(u: np.ndarray, v: np.ndarray) -> float:
    q = np.linspace(0.0, 1.0, 201)
    return float(np.mean(np.abs(np.quantile(u, q) - np.quantile(v, q))))


def mmd_rbf(x, y, rng, cap=600):
    if len(x) > cap:
        x = x[np.sort(rng.choice(len(x), cap, replace=False))]
    if len(y) > cap:
        y = y[np.sort(rng.choice(len(y), cap, replace=False))]
    z = np.concatenate([x, y])
    mu, sd = z.mean(0), z.std(0) + 1e-8
    x, y = (x - mu) / sd, (y - mu) / sd
    z = np.concatenate([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    gamma = 1.0 / max(np.median(d2[d2 > 0]), 1e-8)
    K = np.exp(-gamma * d2)
    n, m = len(x), len(y)
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
                 + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())


def main():
    rng = np.random.default_rng(0)
    ramps = {pol: {s: load_ramp(tag, s) for s in SEEDS} for pol, tag in POLICIES.items()}
    r0 = ramps["A_condmlp50k"][0]
    jn, prof = r0["joint_names"], r0["vx_profile"]
    sl = stage_slices(len(prof), r0["ramp_s"], r0["hold_s"], FS_POL)
    tp, tr = int(TRIM_S * FS_POL), int(TRIM_S * FS_REF)
    clips = {n: read_clip(n, jn) for n in sorted({c for v in MATCHES.values() for c in v})}

    rows = []
    for cmd, cnames in MATCHES.items():
        sids = [i for i, v in enumerate(prof) if abs(v - cmd) < 1e-9]
        for cond, fc in (("raw", None), ("lp5", 5.0), ("lp3", 3.0)):
            ref = []
            for cn in cnames:
                x = np.concatenate([clips[cn]["jpos"], clips[cn]["jvel"]], axis=1)
                if fc is not None:
                    x = lowpass(x, fc, FS_REF)
                    t = min(tr, len(x) // 5)
                    x = x[t : len(x) - t]
                ref.append(resample_to(x, FS_REF, FS_POL))
            ref = np.concatenate(ref)
            for pol in POLICIES:
                segs = []
                for s in SEEDS:
                    r = ramps[pol][s]
                    for si in sids:
                        x = np.concatenate([r["jpos"][sl[si]], r["jvel"][sl[si]]], axis=1)
                        if fc is not None:
                            x = lowpass(x, fc, FS_POL)
                            x = x[tp : len(x) - tp]
                        segs.append(x)
                pol_x = np.concatenate(segs)
                nj = len(jn)
                w_pos = float(np.mean([w1(ref[:, j], pol_x[:, j]) for j in range(nj)]))
                w_vel = float(np.mean([w1(ref[:, nj + j], pol_x[:, nj + j]) for j in range(nj)]))
                rows.append(dict(cmd=cmd, cond=cond, pol=pol, w_jpos=w_pos, w_jvel=w_vel,
                                 mmd=mmd_rbf(ref, pol_x, rng),
                                 mmd_vel=mmd_rbf(ref[:, nj:], pol_x[:, nj:], rng),
                                 n_ref=len(ref), n_pol=len(pol_x)))
    with open(f"{SCRATCH}/gap_dist.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{'cmd':>4} {'cond':5s} {'policy':16s} {'W1(jpos)':>9} {'W1(jvel)':>9} {'MMD(all)':>9} {'MMD(vel)':>9}")
    for r in rows:
        print(f"{r['cmd']:4.1f} {r['cond']:5s} {r['pol']:16s} {r['w_jpos']:9.4f} {r['w_jvel']:9.4f} "
              f"{r['mmd']:9.4f} {r['mmd_vel']:9.4f}")


if __name__ == "__main__":
    main()


def plot():
    """gap_dist.json → figures/gap_w1_lowpass.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(f"{SCRATCH}/gap_dist.json"))
    cmds = sorted({r["cmd"] for r in rows}, reverse=True)
    conds = ["raw", "lp5", "lp3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, lab in ((axes[0], "w_jvel", "W1(joint vel) ref vs policy [rad/s]"),
                         (axes[1], "w_jpos", "W1(joint pos) ref vs policy [rad]"),
                         (axes[2], "mmd_vel", "RBF MMD^2 (joint vel)")):
        w = 0.13
        for i, (pol, c) in enumerate((("A_condmlp50k", "C0"), ("B_conddrail45k", "C3"))):
            for j, cond in enumerate(conds):
                vals = [next(r[key] for r in rows if r["cmd"] == cm and r["cond"] == cond and r["pol"] == pol)
                        for cm in cmds]
                ax.bar(np.arange(len(cmds)) + (i * 3 + j - 2.5) * w, vals, w,
                       color=c, alpha=0.35 + 0.32 * j,
                       label=f"{pol.split('_')[0]}' {cond}")
        ax.set_xticks(range(len(cmds)))
        ax.set_xticklabels([f"cmd {c}" for c in cmds])
        ax.set_ylabel(lab, fontsize=9)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("B' excess distance from reference is confined above 5 Hz (joint velocity only)", fontsize=10)
    fig.tight_layout()
    out = "/home/lgb/IsaacLab-6.0/reports/leg_imitation/_comparisons/conditional_discriminator/figures/gap_w1_lowpass.png"
    fig.savefig(out, dpi=130)
    print("saved", out)
