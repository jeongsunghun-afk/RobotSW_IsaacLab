# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task A — 두 AMP 판별기가 592 차원 입력의 **어느 채널**을 보고 판정하는지 잰다.

두 축으로 잰다.
  1. saliency: ``|∂logit/∂x_raw_j| · σ_j`` (σ_j = 판별기 normalizer 의 std+eps). 정규화가 선형이라
     이 값은 **표준화 입력 기준** 민감도와 같다. 필드별 합·히스토리 스텝별 합.
  2. occlusion: policy 표본의 한 필드를 **같은 index 의 expert 값**으로 갈아끼웠을 때 D(policy) 변화,
     그리고 그 반대. 갈아끼운 표본은 실제 궤적이 아닌 **키메라**라 인과가 아니다 — 그래서 같은 필드를
     다른 policy 행 값으로 바꾸는 **널 대조**를 같이 잰다.

주의: ``norm_clip=10.0`` 이라 클립된 열의 gradient 는 정확히 0 이다. 필드별 클립 적중률을 같이 낸다.
서로 다른 disc 의 gradient 크기 절대값은 비교 불가(MLP logit 무계 vs DRAIL MSE 차분) — 비율만 본다.
"""

from __future__ import annotations

import json
import numpy as np
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disc_offline as D  # noqa: E402

CLIPS = ["leg_run1", "leg_trot0", "leg_walk1", "leg_walk"]
POLICIES = ["condmlp50k", "conddrail45k"]
N_SAMP = 1000
REPS = 32
SEED = 4321
OUT_DIR = os.path.join(
    D.REPO, "reports/leg_imitation/_comparisons/conditional_discriminator/metrics"
)
FIG_DIR = os.path.join(
    D.REPO, "reports/leg_imitation/_comparisons/conditional_discriminator/figures"
)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    discs = D.build_both(dev)
    fs = D.field_slices()
    hs = D.hist_slices()
    fnames = [f for f, _ in D.FIELDS]

    rows_sal, rows_occ, rows_hist = [], [], []
    rng = np.random.default_rng(SEED)

    for pol in POLICIES:
        for clip in CLIPS:
            path = os.path.join(D.PROBE_ROOT, pol, f"{clip}_rsi.npz")
            if not os.path.exists(path):
                print(f"!!! 없음: {path}")
                continue
            z, lay = D.load_npz(path)
            assert lay["step_dim"] == 59 and lay["n_hist"] == 10, lay
            po = z["policy_obs"]
            eo = z["expert_obs"]
            if po.shape[0] == 0:
                print(f"!!! 빈 표본: {path}")
                continue
            n = min(N_SAMP, po.shape[0], eo.shape[0])
            ip = rng.choice(po.shape[0], n, replace=False)
            ie = rng.choice(eo.shape[0], n, replace=False)
            P = torch.tensor(po[ip], dtype=torch.float32)
            E = torch.tensor(eo[ie], dtype=torch.float32)
            # 널 대조용: policy 를 한 칸 섞은 짝
            P_shuf = torch.tensor(po[rng.permutation(ip)], dtype=torch.float32)

            for arch, disc in discs.items():
                sigma = D.norm_sigma(disc).cpu()  # [590]
                torch.manual_seed(SEED)
                for src, X in (("policy", P), ("expert", E)):
                    g = D.grad_logit(disc, X, reps=REPS).cpu()  # [n, 592]
                    contrib = (g[:, : disc._kin_dim].abs() * sigma).mean(0).numpy()  # [590]
                    tot = contrib.sum()
                    clip_hit = D.clip_hit_frac(disc, X).numpy()
                    row = {"disc": arch, "policy": pol, "clip": clip, "src": src, "total": float(tot)}
                    for f in fnames:
                        cols = fs[f]
                        row[f] = float(contrib[cols].sum() / tot)
                        row[f"{f}_clip"] = float(clip_hit[cols].mean())
                    # 조건 열 2개의 절대 기여 (정규화 안 됨 → σ=1 로 본다)
                    row["cond_abs"] = float(g[:, disc._kin_dim :].abs().mean(0).sum())
                    rows_sal.append(row)
                    if src == "policy":
                        hrow = {"disc": arch, "policy": pol, "clip": clip}
                        for h in range(10):
                            hrow[f"h{h}"] = float(contrib[hs[h]].sum() / tot)
                        rows_hist.append(hrow)

                # ── occlusion ──
                torch.manual_seed(SEED)
                d_p = float(D.score(disc, P, REPS).mean())
                d_e = float(D.score(disc, E, REPS).mean())
                l_p = float(D.logits(disc, P, REPS).mean())
                l_e = float(D.logits(disc, E, REPS).mean())
                base = {"disc": arch, "policy": pol, "clip": clip, "d_policy": d_p, "d_expert": d_e,
                        "logit_policy": l_p, "logit_expert": l_e}
                for f in fnames + ["_null"]:
                    if f == "_null":
                        Pm = P.clone()
                        Pm[:, fs["dof_vel"]] = P_shuf[:, fs["dof_vel"]]
                        Em = E.clone()
                        idx = torch.randperm(E.shape[0])
                        Em[:, fs["dof_vel"]] = E[idx][:, fs["dof_vel"]]
                    else:
                        Pm = P.clone()
                        Pm[:, fs[f]] = E[:, fs[f]]
                        Em = E.clone()
                        Em[:, fs[f]] = P[:, fs[f]]
                    r = dict(base)
                    r["field"] = f
                    r["d_policy_sub"] = float(D.score(disc, Pm, REPS).mean())
                    r["d_expert_sub"] = float(D.score(disc, Em, REPS).mean())
                    r["dD_policy"] = r["d_policy_sub"] - d_p
                    r["dD_expert"] = r["d_expert_sub"] - d_e
                    r["dlogit_policy"] = float(D.logits(disc, Pm, REPS).mean()) - l_p
                    r["dlogit_expert"] = float(D.logits(disc, Em, REPS).mean()) - l_e
                    rows_occ.append(r)
            print(f"done {pol}/{clip}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"saliency": rows_sal, "occlusion": rows_occ, "hist": rows_hist,
           "n_samples": N_SAMP, "reps": REPS, "seed": SEED}
    with open(os.path.join(OUT_DIR, "disc_saliency.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(OUT_DIR, "disc_saliency.json"))
    make_plots(out)


def make_plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fnames = [f for f, _ in D.FIELDS]
    os.makedirs(FIG_DIR, exist_ok=True)

    # saliency: 필드별 비율, policy 표본 기준, 클립 평균
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True)
    for i, arch in enumerate(["mlp", "drail"]):
        for j, pol in enumerate(POLICIES):
            ax = axes[i][j]
            rs = [r for r in out["saliency"] if r["disc"] == arch and r["policy"] == pol and r["src"] == "policy"]
            if not rs:
                continue
            w = 0.8 / len(rs)
            for k, r in enumerate(rs):
                ax.bar(np.arange(len(fnames)) + k * w - 0.4, [r[f] for f in fnames], w, label=r["clip"])
            ax.set_xticks(range(len(fnames)))
            ax.set_xticklabels(fnames, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"D={arch}  policy={pol}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    axes[0][0].set_ylabel("saliency share")
    axes[1][0].set_ylabel("saliency share")
    fig.suptitle("|dlogit/dx|·sigma, field share (policy samples)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "disc_saliency.png"), dpi=130)
    plt.close(fig)

    # occlusion
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True)
    for i, arch in enumerate(["mlp", "drail"]):
        for j, pol in enumerate(POLICIES):
            ax = axes[i][j]
            rs = [r for r in out["occlusion"] if r["disc"] == arch and r["policy"] == pol]
            clips = sorted({r["clip"] for rs_ in [rs] for r in rs_})
            flds = fnames + ["_null"]
            w = 0.8 / max(len(clips), 1)
            for k, c in enumerate(clips):
                vals = [next((r["dD_policy"] for r in rs if r["clip"] == c and r["field"] == f), np.nan) for f in flds]
                ax.bar(np.arange(len(flds)) + k * w - 0.4, vals, w, label=c)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xticks(range(len(flds)))
            ax.set_xticklabels(flds, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"D={arch}  policy={pol}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    axes[0][0].set_ylabel("dD(policy)")
    axes[1][0].set_ylabel("dD(policy)")
    fig.suptitle("occlusion: policy field <- expert field  (_null = policy field <- shuffled policy)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "disc_occlusion.png"), dpi=130)
    plt.close(fig)
    print("saved figures ->", FIG_DIR)


if __name__ == "__main__":
    main()
