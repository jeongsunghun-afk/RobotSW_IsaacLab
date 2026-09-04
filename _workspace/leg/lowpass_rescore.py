# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task B — 정책 AMP 관측의 고주파 성분만 지우고 두 판별기로 **반사실 채점**한다.

입력: worker-5 가 만드는 ``{tag}/{clip}_rsi_seq.npz``
  ``policy_obs_seq`` [N, T, 592] float16 · ``policy_jvel`` [N, T, 17] · ``valid`` [N, T]

AMP 관측의 히스토리 슬롯 k 는 시각 t−k 의 값이다. 그래서 **슬롯 0 의 시계열만** 필터한 뒤 슬롯 k 에
t−k 의 필터값을 다시 채워 히스토리를 일관되게 재구성한다. 이 전제(구르는 복사본)는 실행 전에
``obs[n, t, slot1] == obs[n, t-1, slot0]`` 로 검증하고, 깨지면 **필터본을 만들지 않고 멈춘다**.

무엇이 인과가 아닌지: 필터본은 정책이 실제로 낼 수 있는 궤적이 아니라 관측을 후처리한 것이다.
"D 가 얼마나 오르는가" 는 판별기가 그 대역을 얼마나 보는지의 측정이지, 떨림을 없앤 정책의 예측이 아니다.
"""

from __future__ import annotations

import json
import numpy as np
import os
import sys
import torch
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disc_offline as D  # noqa: E402

CLIPS = ["leg_run1", "leg_trot0", "leg_walk1"]
POLICIES = ["condmlp50k", "conddrail45k"]
REPS = 32
REWARD_COEF = 2.0  # agent.yaml amp.reward_coef (모듈 기본 1.5 를 알고리즘이 덮어쓴다)
MAX_SAMPLES = 4000
OUT_DIR = os.path.join(D.REPO, "reports/leg_imitation/_comparisons/conditional_discriminator/metrics")
FIG_DIR = os.path.join(D.REPO, "reports/leg_imitation/_comparisons/conditional_discriminator/figures")

VARIANTS = [
    ("dof_vel", 3.0), ("dof_vel", 5.0), ("dof_vel", 8.0),
    ("ang_vel", 5.0),
    ("both", 5.0),
]


def _slot_cols(field: str, step_dim=59, n_hist=10):
    """[n_hist, width] — 슬롯별 그 필드의 열 인덱스."""
    off, w = 0, None
    for name, width in D.FIELDS:
        if name == field:
            w = width
            break
        off += width
    assert w is not None, field
    return np.array([[h * step_dim + off + j for j in range(w)] for h in range(n_hist)])


def check_shift(obs: np.ndarray, cols: np.ndarray, valid: np.ndarray) -> float:
    """slot1[t] 과 slot0[t-1] 의 최대 절대 차. 구르는 히스토리면 fp16 반올림 수준이어야 한다."""
    a = obs[:, 1:, cols[1]]      # slot 1 at t
    b = obs[:, :-1, cols[0]]     # slot 0 at t-1
    m = (valid[:, 1:] & valid[:, :-1])[..., None]
    d = np.abs(a - b)[np.broadcast_to(m, a.shape)]
    return float(d.max()) if d.size else float("nan")


def lowpass_segments(x: np.ndarray, valid: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """[T, C] 를 연속 valid 구간마다 zero-phase 저역통과. 너무 짧은 구간은 원본 유지."""
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    padlen = 3 * max(len(a), len(b))
    out = x.copy()
    t = 0
    T = x.shape[0]
    while t < T:
        if not valid[t]:
            t += 1
            continue
        s = t
        while t < T and valid[t]:
            t += 1
        if t - s > padlen:
            out[s:t] = filtfilt(b, a, x[s:t], axis=0)
    return out


def build_filtered(obs: np.ndarray, valid: np.ndarray, fields: list[str], fs: float, fc: float) -> np.ndarray:
    """슬롯 0 시계열을 필터한 뒤 슬롯 k ← t−k 로 히스토리 재구성. [N, T, 592] 반환."""
    out = obs.copy()
    N, T, _ = obs.shape
    for field in fields:
        cols = _slot_cols(field)
        for n in range(N):
            s = obs[n][:, cols[0]]  # [T, w]
            sf = lowpass_segments(s, valid[n], fs, fc)
            for k in range(cols.shape[0]):
                if k == 0:
                    out[n][:, cols[0]] = sf
                else:
                    out[n][k:, cols[k]] = sf[:-k]
    return out


def flatten_valid(obs: np.ndarray, valid: np.ndarray, n_hist: int = 10) -> np.ndarray:
    """히스토리 창이 다 찬 스텝만 [M, 592] 로 펼친다."""
    m = valid.copy()
    m[:, : n_hist - 1] = False
    return obs[m]


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    discs = D.build_both(dev)
    rows, checks = [], []
    rng = np.random.default_rng(20260904)

    for pol in POLICIES:
        for clip in CLIPS:
            path = os.path.join(D.PROBE_ROOT, pol, f"{clip}_rsi_seq.npz")
            if not os.path.exists(path):
                print(f"!!! 없음 — 건너뜀: {path}")
                continue
            z = np.load(path, allow_pickle=True)
            obs = z["policy_obs_seq"].astype(np.float32)
            valid = z["valid"].astype(bool)
            dt = float(z["dt"]) if "dt" in z.files else 0.02
            fs = 1.0 / dt
            cols_dv = _slot_cols("dof_vel")
            shift_err = check_shift(obs, cols_dv, valid)
            checks.append({"policy": pol, "clip": clip, "shift_max_abs_err": shift_err,
                           "obs_scale": float(np.abs(obs[:, :, cols_dv[0]]).mean()), "dt": dt,
                           "shape": list(obs.shape)})
            print(f"{pol}/{clip}: shift check max|slot1[t]-slot0[t-1]| = {shift_err:.4g}")
            if not np.isfinite(shift_err) or shift_err > 0.05:
                print("!!! 구르는 히스토리 전제 위반 — 이 파일은 필터하지 않는다")
                continue

            # 참조선: 같은 클립 expert
            ez, _ = D.load_npz(os.path.join(D.PROBE_ROOT, pol, f"{clip}_rsi.npz"))
            base_rows = {}
            for arch in discs:
                base_rows[arch] = float(ez[f"d_{arch}_expert"].mean())

            variants = {"orig": obs}
            for field, fc in VARIANTS:
                flds = ["dof_vel", "ang_vel"] if field == "both" else [field]
                variants[f"{field}@{fc:g}Hz"] = build_filtered(obs, valid, flds, fs, fc)

            for vname, vobs in variants.items():
                X = flatten_valid(vobs, valid)
                if X.shape[0] > MAX_SAMPLES:
                    X = X[rng.choice(X.shape[0], MAX_SAMPLES, replace=False)]
                Xt = torch.tensor(X, dtype=torch.float32)
                for arch, disc in discs.items():
                    torch.manual_seed(99)
                    d = D.score(disc, Xt, REPS)
                    rew = -np.log(np.clip(1.0 - d, 1e-4, None)) * REWARD_COEF
                    rows.append({
                        "policy": pol, "clip": clip, "disc": arch, "variant": vname,
                        "n": int(X.shape[0]),
                        "d_mean": float(d.mean()), "d_median": float(np.median(d)),
                        "reward_mean": float(rew.mean()),
                        "d_expert_ref": base_rows[arch],
                    })
                print(f"  {vname}: " + "  ".join(
                    f"D_{r['disc']}={r['d_mean']:.3f}" for r in rows[-len(discs):]))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"rows": rows, "checks": checks, "reps": REPS, "reward_coef": REWARD_COEF}
    with open(os.path.join(OUT_DIR, "lowpass_rescore.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(OUT_DIR, "lowpass_rescore.json"))
    if rows:
        make_plot(out)


def make_plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=True)
    for i, arch in enumerate(["mlp", "drail"]):
        for j, pol in enumerate(POLICIES):
            ax = axes[i][j]
            clips = []
            for r in rows:
                if r["clip"] not in clips:
                    clips.append(r["clip"])
            w = 0.8 / max(len(clips), 1)
            for k, c in enumerate(clips):
                vals = [next((r["d_mean"] for r in rows
                              if r["disc"] == arch and r["policy"] == pol and r["clip"] == c
                              and r["variant"] == v), np.nan) for v in variants]
                ax.bar(np.arange(len(variants)) + k * w - 0.4, vals, w, label=c)
                ref = next((r["d_expert_ref"] for r in rows
                            if r["disc"] == arch and r["policy"] == pol and r["clip"] == c), None)
                if ref is not None:
                    ax.axhline(ref, ls="--", lw=0.8, color=f"C{k}")
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"D={arch}  policy={pol}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    axes[0][0].set_ylabel("D(policy)")
    axes[1][0].set_ylabel("D(policy)")
    fig.suptitle("low-pass counterfactual rescoring (dashed = same-clip expert D)", fontsize=11)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "lowpass_rescore.png"), dpi=130)
    plt.close(fig)
    print("saved figure")


if __name__ == "__main__":
    main()
