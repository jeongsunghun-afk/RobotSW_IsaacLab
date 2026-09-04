# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""조건부 disc 갭 조사 — Task B: AMP 관측 공간에서의 분리도.

worker-3 가 만드는 clip_probe npz 를 읽어 판별기 구조에 독립적인 분리도(C2ST)와
특징 그룹별 거리(1-D Wasserstein, RBF MMD), 그리고 판별기 출력 분포를 비교한다.

npz 스키마 (worker-3 규정):
  policy_obs   [S, D]      D = n_hist*step_dim + cond_dim
  expert_obs   [E, D]
  layout       JSON str    {n_hist:10, step_dim:59, cond_dim, fields:[[name, size], ...]}
                           (fields 가 dict 나 name 리스트로 와도 파싱한다)
                           hist_order: index 0 = newest
  d_mlp_policy / d_mlp_expert / d_drail_policy / d_drail_expert   (sigmoid 출력)
  policy_jvel  [N, T, 17], expert_jvel [T, 17], dt
  clip_name, clip_mean_speed, cmd, meas

사용법:
  python _workspace/leg/gap_analysis_ampobs.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

REPO = "/home/lgb/IsaacLab-6.0"
PROBE_DIR = f"{REPO}/reports/leg_imitation/_comparisons/conditional_discriminator/metrics/clip_probe"
OUT_MET = f"{REPO}/reports/leg_imitation/_comparisons/conditional_discriminator/metrics"
OUT_FIG = f"{REPO}/reports/leg_imitation/_comparisons/conditional_discriminator/figures"

POLICIES = {"A_condmlp50k": "condmlp50k", "B_conddrail45k": "conddrail45k"}
FOLDS = 5
SEED = 0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
CAP = 2000  # 클래스당 표본 상한 (계산량 제한)
# C2ST 는 비미러 클립만 (계산량); 거리/판별기 출력은 전 파일

# 기본 field 레이아웃 (layout 이 없거나 이름만 줄 때의 fallback; step_dim 59 와 일치)
DEFAULT_FIELDS = [
    ("dof_pos", 17), ("dof_vel", 17), ("root_h", 1), ("lin_vel", 3),
    ("ang_vel", 3), ("foot_pos", 12), ("rot_tan_norm", 6),
]


# ── 레이아웃 파싱 ──────────────────────────────────────────────────────────


def parse_layout(raw) -> dict:
    """layout 을 dict 로 정규화하고 field 별 (start, size) 를 채운다."""
    if isinstance(raw, (bytes, str)):
        lay = json.loads(raw if isinstance(raw, str) else raw.decode())
    elif isinstance(raw, np.ndarray):
        lay = json.loads(str(raw.item()) if raw.ndim == 0 else str(raw[0]))
    else:
        lay = dict(raw)
    fields = lay.get("fields", DEFAULT_FIELDS)
    norm = []
    if isinstance(fields, dict):
        norm = [(k, int(v)) for k, v in fields.items()]
    else:
        for f in fields:
            if isinstance(f, (list, tuple)) and len(f) == 2 and not isinstance(f[1], str):
                norm.append((str(f[0]), int(f[1])))
            else:  # 이름만 준 경우 → 기본 크기 사용
                name = str(f[0] if isinstance(f, (list, tuple)) else f)
                size = dict(DEFAULT_FIELDS).get(name)
                assert size is not None, f"field '{name}' 크기를 알 수 없다"
                norm.append((name, size))
    lay["fields"] = norm
    lay["n_hist"] = int(lay.get("n_hist", 10))
    lay["step_dim"] = int(lay.get("step_dim", sum(s for _, s in norm)))
    lay["cond_dim"] = int(lay.get("cond_dim", 0))
    assert lay["step_dim"] == sum(s for _, s in norm), (
        f"step_dim {lay['step_dim']} != field 합 {sum(s for _, s in norm)}"
    )
    off, spans = 0, {}
    for name, size in norm:
        spans[name] = (off, off + size)
        off += size
    lay["spans"] = spans
    return lay


def field_columns(lay: dict, name: str) -> np.ndarray:
    """모든 history 스텝에 걸친 해당 field 의 전체 열 인덱스."""
    lo, hi = lay["spans"][name]
    cols = []
    for h in range(lay["n_hist"]):
        base = h * lay["step_dim"]
        cols.extend(range(base + lo, base + hi))
    return np.array(cols, dtype=int)


def obs_columns(lay: dict) -> np.ndarray:
    """조건 열을 제외한 관측 열 (= 앞 n_hist*step_dim)."""
    return np.arange(lay["n_hist"] * lay["step_dim"])


def hist_mean(x: np.ndarray, lay: dict) -> np.ndarray:
    """history 축 평균 저역 버전 [S, step_dim]."""
    n, sd = lay["n_hist"], lay["step_dim"]
    return x[:, : n * sd].reshape(len(x), n, sd).mean(axis=1)


# ── C2ST (Task A 와 동일 구조: 표준화 + 로지스틱/MLP256, 연속 블록 5-fold) ──


def _fit_eval(xtr, ytr, xte, yte, kind: str) -> tuple[float, float]:
    g = torch.Generator(device=DEV).manual_seed(SEED)
    torch.manual_seed(SEED)
    xtr, ytr, xte, yte = xtr.to(DEV), ytr.to(DEV), xte.to(DEV), yte.to(DEV)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    d = xtr.shape[1]
    if kind == "logreg":
        model = torch.nn.Linear(d, 1).to(DEV)
        wd, epochs, lr = 1e-2, EPOCHS, 0.05
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        ).to(DEV)
        wd, epochs, lr = 1e-3, EPOCHS, 0.005
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    n, bs = xtr.shape[0], min(256, xtr.shape[0])
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            lossf(model(xtr[idx]).squeeze(-1), ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        s = model(xte).squeeze(-1)
    acc = ((s > 0).float() == yte).float().mean().item()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype, device=s.device)
    npos, nneg = yte.sum().item(), (1 - yte).sum().item()
    auroc = float("nan") if npos == 0 or nneg == 0 else (
        ranks[yte == 1].sum().item() - npos * (npos + 1) / 2) / (npos * nneg)
    return acc, auroc


def c2st(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """연속 블록 5-fold. 표본이 시간 순서일 수 있으므로 랜덤 fold 를 쓰지 않는다."""
    def chunks(x):
        if len(x) > CAP:
            x = x[np.linspace(0, len(x) - 1, CAP).astype(int)]
        e = np.linspace(0, len(x), FOLDS + 1).astype(int)
        return [x[e[k] : e[k + 1]] for k in range(FOLDS)]

    fa, fb = chunks(a), chunks(b)
    res = {k: [] for k in ("acc_logreg", "acc_mlp", "auc_logreg", "auc_mlp")}
    for k in range(FOLDS):
        tr_a = np.concatenate([fa[j] for j in range(FOLDS) if j != k])
        tr_b = np.concatenate([fb[j] for j in range(FOLDS) if j != k])
        te_a, te_b = fa[k], fb[k]
        if min(len(tr_a), len(tr_b), len(te_a), len(te_b)) < 8:
            continue

        def bal(u, v):
            m = min(len(u), len(v))
            return u[np.sort(rng.choice(len(u), m, replace=False))], v[np.sort(rng.choice(len(v), m, replace=False))]

        tr_a, tr_b = bal(tr_a, tr_b)
        te_a, te_b = bal(te_a, te_b)
        xtr = torch.tensor(np.concatenate([tr_a, tr_b]), dtype=torch.float32)
        ytr = torch.tensor(np.r_[np.zeros(len(tr_a)), np.ones(len(tr_b))], dtype=torch.float32)
        xte = torch.tensor(np.concatenate([te_a, te_b]), dtype=torch.float32)
        yte = torch.tensor(np.r_[np.zeros(len(te_a)), np.ones(len(te_b))], dtype=torch.float32)
        for kind in ("logreg", "mlp"):
            acc, auc = _fit_eval(xtr, ytr, xte, yte, kind)
            res[f"acc_{kind}"].append(acc)
            res[f"auc_{kind}"].append(auc)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in res.items()} | {
        "n_a": int(len(a)), "n_b": int(len(b))
    }


# ── 거리 지표 ──────────────────────────────────────────────────────────────


def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """1-D Wasserstein-1 (같은 분위 격자에서 적분)."""
    q = np.linspace(0.0, 1.0, 201)
    return float(np.mean(np.abs(np.quantile(u, q) - np.quantile(v, q))))


def mmd_rbf(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, cap: int = 800) -> float:
    """RBF MMD^2 (median heuristic, unbiased)."""
    if len(x) > cap:
        x = x[np.sort(rng.choice(len(x), cap, replace=False))]
    if len(y) > cap:
        y = y[np.sort(rng.choice(len(y), cap, replace=False))]
    z = np.concatenate([x, y])
    mu, sd = z.mean(0), z.std(0) + 1e-8
    x, y = (x - mu) / sd, (y - mu) / sd
    z = np.concatenate([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / max(med, 1e-8)
    K = np.exp(-gamma * d2)
    n, m = len(x), len(y)
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return float(
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2 * Kxy.mean()
    )


# ── 메인 ───────────────────────────────────────────────────────────────────


MIRROR_SKIP = "_mirror"


def load_meta(d, fname, pol) -> dict:
    def sc(k, default=float("nan")):
        return float(d[k]) if k in d.files and d[k].size == 1 else default

    return dict(
        file=fname, pol=pol,
        clip=str(d["clip_name"]) if "clip_name" in d.files else "?",
        start=str(d["start_mode"]) if "start_mode" in d.files else "?",
        clip_speed=sc("clip_mean_speed"),
        cmd_vx=float(np.mean(d["cmd"][..., 0])) if "cmd" in d.files else float("nan"),
        meas_vx=float(np.mean(d["meas"][..., 0])) if "meas" in d.files else float("nan"),
    )


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_MET, exist_ok=True)
    rng = np.random.default_rng(0)
    files = {}
    for pol, tag in POLICIES.items():
        for p in sorted(glob.glob(f"{PROBE_DIR}/{tag}/*.npz")):
            files.setdefault(os.path.basename(p), {})[pol] = p
    assert files, f"probe npz 없음: {PROBE_DIR}"
    print(f"[dev] {DEV} — {len(files)} 파일")

    rows_c2st, rows_dist, rows_d = [], [], []
    d_hists = {}
    lay_ref = None
    for fname, per_pol in sorted(files.items()):
        do_c2st = MIRROR_SKIP not in fname  # 계산량 제한: 미러 클립은 거리/판별기만
        cached = {}
        for pol, path in sorted(per_pol.items()):
            d = np.load(path, allow_pickle=True)
            lay = parse_layout(d["layout"])
            lay_ref = lay_ref or lay
            po = d["policy_obs"].astype(np.float64)
            eo = d["expert_obs"].astype(np.float64)
            oc = obs_columns(lay)
            meta = load_meta(d, fname, pol)
            cached[pol] = eo[:, oc]

            if do_c2st:
                dv = set(field_columns(lay, "dof_vel").tolist())
                keep = np.array([c for c in oc if c not in dv])
                variants = {
                    "all": (eo[:, oc], po[:, oc]),
                    "no_dof_vel": (eo[:, keep], po[:, keep]),
                    "hist_mean": (hist_mean(eo, lay), hist_mean(po, lay)),
                }
                for vname, (a, b) in variants.items():
                    rows_c2st.append(meta | {"variant": vname, "pair": "expert_vs_policy"}
                                     | c2st(a, b, rng))

            for fld, _ in lay["fields"]:
                cols = field_columns(lay, fld)
                rows_dist.append(meta | {
                    "field": fld,
                    "wasserstein_mean": float(np.mean([wasserstein_1d(eo[:, c], po[:, c]) for c in cols])),
                    "mmd_rbf": mmd_rbf(eo[:, cols], po[:, cols], rng),
                })

            for dk in ("d_mlp", "d_drail"):
                kp, ke = f"{dk}_policy", f"{dk}_expert"
                if kp not in d.files:
                    continue
                dp, de = d[kp].ravel().astype(np.float64), d[ke].ravel().astype(np.float64)
                d_hists[(fname, pol, dk)] = (dp, de)
                rows_d.append(meta | {
                    "disc": dk,
                    "policy_mean": float(dp.mean()), "policy_median": float(np.median(dp)),
                    "expert_mean": float(de.mean()), "expert_median": float(np.median(de)),
                    "policy_lt05": float((dp < 0.05).mean()), "expert_gt95": float((de > 0.95).mean()),
                    "sep_auroc": float(_auroc_np(de, dp)),
                })

        # 음성 대조: 같은 클립의 expert_obs (두 파일에서 각각 독립 추출) — 참 null
        if do_c2st and len(cached) == 2:
            a, b = [cached[k] for k in sorted(cached)]
            rows_c2st.append({"file": fname, "pol": "ctrl", "clip": fname, "start": "?",
                              "clip_speed": float("nan"), "cmd_vx": float("nan"), "meas_vx": float("nan"),
                              "variant": "all", "pair": "ctrl_expert_A_vs_B"} | c2st(a, b, rng))

    with open(f"{OUT_MET}/gap_ampobs_raw.json", "w") as f:
        json.dump({"c2st": rows_c2st, "dist": rows_dist, "disc": rows_d}, f, indent=1)
    print(f"[done] c2st={len(rows_c2st)} dist={len(rows_dist)} disc={len(rows_d)}")

    _plots(rows_c2st, rows_dist, d_hists, lay_ref)


def _auroc_np(pos: np.ndarray, neg: np.ndarray) -> float:
    """expert(pos) 가 policy(neg) 보다 높은 D 값을 받는 정도."""
    z = np.concatenate([pos, neg])
    r = np.empty(len(z))
    r[np.argsort(z)] = np.arange(1, len(z) + 1)
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _plots(rows_c2st, rows_dist, d_hists, lay):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clips = sorted({r["file"] for r in rows_c2st if r["pol"] != "ctrl"})
    n = max(len(clips), 1)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 4.2), sharey=True, squeeze=False)
    vs = ["all", "no_dof_vel", "hist_mean"]
    for ax, cf in zip(axes[0], clips):
        w = 0.28
        for i, pol in enumerate(sorted(POLICIES)):
            vals = [next((r["acc_mlp"] for r in rows_c2st if r["file"] == cf and r["pol"] == pol
                          and r["variant"] == v and r["pair"] == "expert_vs_policy"), np.nan) for v in vs]
            ax.bar(np.arange(len(vs)) + (i - 0.5) * w, vals, w, label=pol, color=f"C{0 if i == 0 else 3}")
        ctrl = next((r["acc_mlp"] for r in rows_c2st if r["file"] == cf and r["pair"] == "ctrl_expert_A_vs_B"), np.nan)
        ax.axhline(ctrl, c="g", ls="--", lw=1, label="ctrl expert/expert")
        ax.axhline(0.5, c="k", ls=":")
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels(vs, rotation=30, ha="right", fontsize=6)
        ax.set_title(cf.replace(".npz", ""), fontsize=6)
        ax.set_ylim(0.4, 1.02)
    axes[0][0].set_ylabel("C2ST held-out acc (MLP, block 5-fold)")
    axes[0][0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/gap_c2st_ampobs.png", dpi=130)

    fields = [f for f, _ in lay["fields"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, key, labl in ((axes[0], "wasserstein_mean", "mean 1-D Wasserstein (expert vs policy)"),
                          (axes[1], "mmd_rbf", "RBF MMD^2 (expert vs policy)")):
        w = 0.35
        for i, pol in enumerate(sorted(POLICIES)):
            vals = [float(np.mean([r[key] for r in rows_dist if r["pol"] == pol and r["field"] == fl] or [np.nan]))
                    for fl in fields]
            ax.bar(np.arange(len(fields)) + (i - 0.5) * w, vals, w, label=pol, color=f"C{0 if i == 0 else 3}")
        ax.set_xticks(range(len(fields)))
        ax.set_xticklabels(fields, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(labl, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/gap_feature_distance.png", dpi=130)

    keys = [k for k in sorted(d_hists) if "_mirror" not in k[0]][:12]
    if keys:
        fig, axes = plt.subplots(2, (len(keys) + 1) // 2, figsize=(2.6 * ((len(keys) + 1) // 2), 5.4), squeeze=False)
        for ax, k in zip(axes.ravel(), keys):
            dp, de = d_hists[k]
            ax.hist(de, bins=40, range=(0, 1), alpha=0.6, label="expert", color="C2")
            ax.hist(dp, bins=40, range=(0, 1), alpha=0.6, label="policy", color="C1")
            ax.set_title(f"{k[0].replace('.npz','')}\n{k[1]} {k[2]}", fontsize=5)
            ax.tick_params(labelsize=5)
            ax.legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(f"{OUT_FIG}/gap_d_hist.png", dpi=130)


if __name__ == "__main__":
    main()
