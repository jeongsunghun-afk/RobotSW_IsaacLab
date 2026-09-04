# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task C — 조건부 D 가 **조건 열을 실제로 쓰는가** 를 policy 표본으로 검정한다.

조건 열 구성(코드 확인, `source/isaaclab_tasks/isaaclab_tasks/direct/amp_command_condition.py`):

.. code-block:: text

    amp_cond_mode = "speed"  →  cond_values_dim = 1, amp_cond_dim = 2
    590 열 = |lin_vel_cmd| / v_max      591 열 = valid (1)
    v_max  = amp_cond_v_max or lin_vel_x_max = 3.2 m/s   (두 run 다 amp_cond_v_max=None)
    정지 명령 env 는 cond=0 · valid=0 (무조건부 판별)

따라서 상수 0.1 / 0.5 / 0.9 는 각각 명령 0.32 / 1.60 / 2.88 m/s 라벨이다.

09-03 게이트는 expert 표본만 썼다. 여기서는 policy·expert 둘 다에 같은 처치를 건다.
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
N_SAMP = 4000
REPS = 32
SEED = 5150
V_MAX = 3.2  # env.yaml lin_vel_x_max (amp_cond_v_max 가 None 이라 이 값이 스케일)
OUT_DIR = os.path.join(D.REPO, "reports/leg_imitation/_comparisons/conditional_discriminator/metrics")
FIG_DIR = os.path.join(D.REPO, "reports/leg_imitation/_comparisons/conditional_discriminator/figures")

VARIANTS = ["orig", "c0.1", "c0.5", "c0.9", "shuffle", "shuffle_pool", "drop"]


def apply_variant(X: torch.Tensor, name: str, kin: int, gen: torch.Generator,
                  pool: torch.Tensor | None = None) -> torch.Tensor:
    out = X.clone()
    if name == "orig":
        return out
    if name == "shuffle_pool":
        # ★ 클립 안에서 cond 는 거의 상수(표준편차 0.03~0.14)라 클립 내부 셔플은 약한 검정이다.
        # 4 클립의 cond 를 모두 모아 다시 뽑으면 주변분포는 보존하면서 짝만 완전히 끊는다.
        idx = torch.randint(0, pool.shape[0], (X.shape[0],), generator=gen)
        out[:, kin] = pool[idx]
        return out
    if name == "drop":
        out[:, kin:] = 0.0  # cond=0 AND valid=0 — env 의 정지 명령 처리와 같다
        return out
    if name == "shuffle":
        idx = torch.randperm(X.shape[0], generator=gen)
        out[:, kin] = X[idx, kin]
        return out
    out[:, kin] = float(name[1:])
    return out


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    discs = D.build_both(dev)
    rng = np.random.default_rng(SEED)
    rows = []

    # 클립 간 풀링 cond (주변분포 보존, 짝만 끊는 강한 셔플용)
    POOL = {}
    for pol in POLICIES:
        acc = {"policy": [], "expert": []}
        for clip in CLIPS:
            fp = os.path.join(D.PROBE_ROOT, pol, f"{clip}_rsi.npz")
            if not os.path.exists(fp):
                continue
            zz = np.load(fp, allow_pickle=True)
            acc["policy"].append(zz["policy_obs"][:, 590])
            acc["expert"].append(zz["expert_obs"][:, 590])
        POOL[pol] = {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in acc.items()}
        print(f"{pol} pooled cond: n={POOL[pol]['policy'].numel()} "
              f"{POOL[pol]['policy'].min():.3f}~{POOL[pol]['policy'].max():.3f}")

    for pol in POLICIES:
        for clip in CLIPS:
            path = os.path.join(D.PROBE_ROOT, pol, f"{clip}_rsi.npz")
            if not os.path.exists(path):
                print(f"!!! 없음: {path}")
                continue
            z, lay = D.load_npz(path)
            assert lay["cond_dim"] == 2 and lay["cond_values_dim"] == 1, lay
            po, eo = z["policy_obs"], z["expert_obs"]
            if po.shape[0] == 0:
                continue
            n = min(N_SAMP, po.shape[0], eo.shape[0])
            P = torch.tensor(po[rng.choice(po.shape[0], n, replace=False)], dtype=torch.float32)
            E = torch.tensor(eo[rng.choice(eo.shape[0], n, replace=False)], dtype=torch.float32)
            native_p = float(P[:, 590].mean())
            native_e = float(E[:, 590].mean())

            for arch, disc in discs.items():
                kin = disc._kin_dim
                pool = POOL[pol]
                base = {}
                for src, X in (("policy", P), ("expert", E)):
                    gen = torch.Generator().manual_seed(SEED)
                    for v in VARIANTS:
                        Xv = apply_variant(X, v, kin, gen, pool[src])
                        torch.manual_seed(SEED)
                        d = D.score(disc, Xv, REPS)
                        if v == "orig":
                            base[src] = float(d.mean())
                        rows.append({
                            "disc": arch, "policy": pol, "clip": clip, "src": src, "variant": v,
                            "n": n, "d_mean": float(d.mean()), "d_median": float(np.median(d)),
                            "dD": float(d.mean()) - base[src],
                            "native_cond": native_p if src == "policy" else native_e,
                            "native_speed": (native_p if src == "policy" else native_e) * V_MAX,
                        })
            pr = [r for r in rows if r["clip"] == clip and r["policy"] == pol and r["src"] == "policy"]
            print(f"{pol}/{clip} (native cond {native_p:.3f} = {native_p*V_MAX:.2f} m/s)")
            for arch in discs:
                seg = {r["variant"]: r["d_mean"] for r in pr if r["disc"] == arch}
                print(f"   D_{arch:5s} policy: " + "  ".join(f"{v}={seg[v]:.3f}" for v in VARIANTS))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"rows": rows, "v_max": V_MAX, "reps": REPS, "seed": SEED, "n_samples": N_SAMP,
           "cond_layout": "col590=|lin_vel_cmd|/3.2, col591=valid"}
    with open(os.path.join(OUT_DIR, "disc_cond_swap.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(OUT_DIR, "disc_cond_swap.json"))
    make_plot(out)


def make_plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True)
    for i, arch in enumerate(["mlp", "drail"]):
        for j, pol in enumerate(POLICIES):
            ax = axes[i][j]
            w = 0.8 / len(CLIPS)
            for k, c in enumerate(CLIPS):
                vals = [next((r["d_mean"] for r in rows if r["disc"] == arch and r["policy"] == pol
                              and r["clip"] == c and r["src"] == "policy" and r["variant"] == v), np.nan)
                        for v in VARIANTS]
                nat = next((r["native_cond"] for r in rows if r["disc"] == arch and r["policy"] == pol
                            and r["clip"] == c and r["src"] == "policy"), np.nan)
                ax.bar(np.arange(len(VARIANTS)) + k * w - 0.4, vals, w, label=f"{c} (cond {nat:.2f})")
            ax.set_xticks(range(len(VARIANTS)))
            ax.set_xticklabels(VARIANTS, rotation=20, ha="right", fontsize=8)
            ax.set_title(f"D={arch}  policy={pol}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    axes[0][0].set_ylabel("D(policy)")
    axes[1][0].set_ylabel("D(policy)")
    fig.suptitle("condition-column swap: c0.1=0.32 m/s, c0.5=1.60, c0.9=2.88; drop = cond 0 + valid 0", fontsize=11)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "disc_cond_swap.png"), dpi=130)
    plt.close(fig)
    print("saved figure")


if __name__ == "__main__":
    main()
