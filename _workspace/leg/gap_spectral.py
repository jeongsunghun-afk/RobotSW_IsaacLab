# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""조건부 disc 갭 조사 — Task A: 주파수 분해 분리도.

A' cond-mlp(model_49999) vs B' cond+drail(model_45000) 의 긴 램프 npz 와 참조 pkl 을
비교해 (1) 관절속도 PSD 대역별 파워 비율, (2) 저역통과 전/후 C2ST 분리도를 잰다.

판별기 구조에 독립적인 분리도를 재는 것이 목적이므로 D 출력은 쓰지 않는다.

주의:
  - 정책 dt=0.02 s (50 Hz, Nyquist 25), 참조 fps=60 (Nyquist 30).
    PSD 는 각자 native fs 로 계산하고 대역 적분은 25 Hz 까지만 한다.
    C2ST 는 참조를 50 Hz 로 선형 보간해 같은 격자에 올린다 (motion_lib 런타임과 동일).
  - 참조 dof_vel 은 유한차분(고역통과 성격), 정책 dof_vel 은 시뮬 상태. 연산자가 다르다.
  - 시계열이라 랜덤 fold 는 무효(이웃 프레임 누설). 연속 블록 5-fold 를 쓴다.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import torch
from scipy import signal

REPO = "/home/lgb/IsaacLab-6.0"
RAMP_DIR = f"{REPO}/reports/leg_imitation/_comparisons/conditional_discriminator/metrics/long_ramp"
PKL_DIR = f"{REPO}/source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/imitation/new_smr_leg_pkl"
OUT_MET = f"{REPO}/reports/leg_imitation/_comparisons/conditional_discriminator/metrics"
OUT_FIG = f"{REPO}/reports/leg_imitation/_comparisons/conditional_discriminator/figures"

DOF_NAMES = [
    "HL_hip_joint", "HL_thigh_joint", "HL_calf_joint", "HL_foot_joint",
    "HR_hip_joint", "HR_thigh_joint", "HR_calf_joint", "HR_foot_joint",
    "FB_waist_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
]

POLICIES = {"A_condmlp50k": "condmlp50k", "B_conddrail45k": "conddrail45k"}
SEEDS = [0, 1, 2]
FS_POL = 50.0
FS_REF = 60.0
BANDS = [(0.0, 3.0), (3.0, 8.0), (8.0, 25.0)]
BAND_NAMES = ["0-3Hz", "3-8Hz", "8-25Hz"]
TRIM_S = 0.3  # filtfilt 엣지 과도응답 절단 [s]
FOLDS = 5
SEED_TORCH = 0


# ── 데이터 로드 ────────────────────────────────────────────────────────────


def load_ramp(tag: str, seed: int) -> dict:
    p = f"{RAMP_DIR}/{tag}_s{seed}/ramp_data.npz"
    d = np.load(p, allow_pickle=True)
    return {
        "jpos": d["jpos"].astype(np.float64),
        "jvel": d["jvel"].astype(np.float64),
        "vx": d["vx"].astype(np.float64),
        "vx_cmd": d["vx_cmd"].astype(np.float64),
        "joint_names": [str(s) for s in d["joint_names"]],
        "vx_profile": d["vx_profile"].astype(np.float64),
        "hold_s": float(d["hold_s"]),
        "ramp_s": float(d["ramp_s"]),
        "dt": float(d["dt"]),
    }


def read_clip(name: str, joint_names: list[str]) -> dict:
    """pkl → dof_pos/dof_vel (ramp joint_names 순서), 평균 root 속도."""
    with open(f"{PKL_DIR}/{name}.pkl", "rb") as f:
        data = pickle.load(f)
    frames = np.array(data["frames"], dtype=np.float64)
    fps = float(data["fps"])
    dt = 1.0 / fps
    root_pos = frames[:, 0:3]
    dof_pos = frames[:, 6:23]
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt
    dof_vel[-1] = dof_vel[-2]
    idx = [DOF_NAMES.index(n) for n in joint_names]
    spd = np.linalg.norm(np.diff(root_pos, axis=0), axis=1) / dt
    return {
        "jpos": dof_pos[:, idx],
        "jvel": dof_vel[:, idx],
        "fps": fps,
        "mean_speed": float(spd.mean()),
        "max_speed": float(spd.max()),
        "name": name,
    }


def stage_slices(n_stage: int, ramp_s: float, hold_s: float, fs: float) -> list[slice]:
    """stage 별 hold 후반 4 s 슬라이스."""
    per = int(round((ramp_s + hold_s) * fs))
    late = int(round(4.0 * fs))
    out = []
    for k in range(n_stage):
        end = (k + 1) * per
        out.append(slice(end - late, end))
    return out


# ── 신호 처리 ──────────────────────────────────────────────────────────────


def lowpass(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """zero-phase 2차 butterworth 저역통과. x [T, D]."""
    b, a = signal.butter(2, fc / (fs / 2.0), btype="low")
    return signal.filtfilt(b, a, x, axis=0)


def band_power(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """관절별 Welch PSD → (freqs, psd [F, D], 대역파워 [3, D])."""
    nps = min(nperseg, x.shape[0])
    f, p = signal.welch(x, fs=fs, nperseg=nps, noverlap=nps // 2, detrend="constant", axis=0)
    bp = np.zeros((len(BANDS), x.shape[1]))
    for i, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & (f < hi)
        bp[i] = np.trapezoid(p[m], f[m], axis=0)
    return f, p, bp


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """선형 보간 재표본화 (motion_lib 런타임 보간과 동일한 연산)."""
    t_in = np.arange(x.shape[0]) / fs_in
    t_out = np.arange(0.0, t_in[-1], 1.0 / fs_out)
    return np.stack([np.interp(t_out, t_in, x[:, j]) for j in range(x.shape[1])], axis=1)


# ── C2ST ───────────────────────────────────────────────────────────────────


def _fit_eval(xtr, ytr, xte, yte, kind: str) -> tuple[float, float]:
    """고정 하이퍼파라미터. kind in {logreg, mlp}. 반환 (acc, auroc)."""
    g = torch.Generator().manual_seed(SEED_TORCH)
    torch.manual_seed(SEED_TORCH)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd
    d = xtr.shape[1]
    if kind == "logreg":
        model = torch.nn.Linear(d, 1)
        wd, epochs, lr = 1e-2, 400, 0.05
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        wd, epochs, lr = 1e-3, 400, 0.005
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = xtr.shape[0]
    bs = min(256, n)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = lossf(model(xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        s = model(xte).squeeze(-1)
    pred = (s > 0).float()
    acc = (pred == yte).float().mean().item()
    # AUROC (rank 기반)
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    npos = yte.sum().item()
    nneg = len(yte) - npos
    if npos == 0 or nneg == 0:
        auroc = float("nan")
    else:
        auroc = ((ranks[yte == 1].sum().item() - npos * (npos + 1) / 2) / (npos * nneg))
    return acc, auroc


def c2st(segs_a: list[np.ndarray], segs_b: list[np.ndarray], rng: np.random.Generator) -> dict:
    """연속 블록 5-fold C2ST. segs_* 는 연속 구간 리스트 [T_i, D].

    각 구간을 시간순 5등분해 fold 를 배정한다(이웃 프레임 누설 방지).
    클래스 균형은 많은 쪽을 적은 쪽 n 으로 fold 내 서브샘플링해 맞춘다.
    """
    def chunk(segs):
        folds = [[] for _ in range(FOLDS)]
        for s in segs:
            n = s.shape[0]
            if n < FOLDS * 4:
                continue
            edges = np.linspace(0, n, FOLDS + 1).astype(int)
            for k in range(FOLDS):
                folds[k].append(s[edges[k] : edges[k + 1]])
        return [np.concatenate(f, axis=0) if f else np.zeros((0, segs[0].shape[1])) for f in folds]

    fa, fb = chunk(segs_a), chunk(segs_b)
    accs, aucs = {"logreg": [], "mlp": []}, {"logreg": [], "mlp": []}
    for k in range(FOLDS):
        tr_a = np.concatenate([fa[j] for j in range(FOLDS) if j != k], axis=0)
        tr_b = np.concatenate([fb[j] for j in range(FOLDS) if j != k], axis=0)
        te_a, te_b = fa[k], fb[k]
        if min(len(tr_a), len(tr_b), len(te_a), len(te_b)) < 8:
            continue
        # 클래스 균형 (train / test 각각)
        def bal(u, v):
            m = min(len(u), len(v))
            iu = rng.choice(len(u), m, replace=False)
            iv = rng.choice(len(v), m, replace=False)
            return u[np.sort(iu)], v[np.sort(iv)]

        tr_a, tr_b = bal(tr_a, tr_b)
        te_a, te_b = bal(te_a, te_b)
        xtr = torch.tensor(np.concatenate([tr_a, tr_b]), dtype=torch.float32)
        ytr = torch.tensor(np.concatenate([np.zeros(len(tr_a)), np.ones(len(tr_b))]), dtype=torch.float32)
        xte = torch.tensor(np.concatenate([te_a, te_b]), dtype=torch.float32)
        yte = torch.tensor(np.concatenate([np.zeros(len(te_a)), np.ones(len(te_b))]), dtype=torch.float32)
        for kind in ("logreg", "mlp"):
            a, u = _fit_eval(xtr, ytr, xte, yte, kind)
            accs[kind].append(a)
            aucs[kind].append(u)
    out = {}
    for kind in ("logreg", "mlp"):
        out[f"acc_{kind}"] = float(np.mean(accs[kind])) if accs[kind] else float("nan")
        out[f"acc_{kind}_sd"] = float(np.std(accs[kind])) if accs[kind] else float("nan")
        out[f"auc_{kind}"] = float(np.mean(aucs[kind])) if aucs[kind] else float("nan")
    out["n_a"] = int(sum(len(s) for s in segs_a))
    out["n_b"] = int(sum(len(s) for s in segs_b))
    return out


# ── 메인 ───────────────────────────────────────────────────────────────────

# 속도 매칭: 램프 cmd → 참조 클립 목록 (미러 포함, 미러도 expert 분포의 일부)
MATCHES = {
    3.0: ["leg_run0", "leg_run0_mirror", "leg_run1", "leg_run1_mirror"],
    2.0: ["leg_trot0", "leg_trot0_mirror"],
    1.0: ["leg_walk1"],
    0.5: ["leg_walk1_stmr"],
}

JOINT_GROUPS = ["hip", "thigh", "calf", "foot", "waist"]


def group_of(name: str) -> str:
    for g in JOINT_GROUPS:
        if g in name:
            return g
    return "other"


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    ramps = {pol: {s: load_ramp(tag, s) for s in SEEDS} for pol, tag in POLICIES.items()}
    ref0 = ramps["A_condmlp50k"][0]
    jn = ref0["joint_names"]
    prof = ref0["vx_profile"]
    slices = stage_slices(len(prof), ref0["ramp_s"], ref0["hold_s"], FS_POL)
    trim_p = int(round(TRIM_S * FS_POL))
    trim_r = int(round(TRIM_S * FS_REF))

    clips = {n: read_clip(n, jn) for n in sorted({c for v in MATCHES.values() for c in v})}

    result = {"clips": {n: {"mean_speed": c["mean_speed"], "max_speed": c["max_speed"],
                            "n_frames": int(c["jpos"].shape[0]), "fps": c["fps"]}
                        for n, c in clips.items()}}

    # ── 1. PSD / 대역 파워 ────────────────────────────────────────────────
    psd_store = {}
    band_rows = []
    for cmd, cnames in MATCHES.items():
        stage_ids = [i for i, v in enumerate(prof) if abs(v - cmd) < 1e-9]
        # 참조
        ref_bp = np.zeros((len(BANDS), len(jn)))
        ref_psd = None
        nps_ref = min(120, min(clips[cn]["jpos"].shape[0] for cn in cnames))
        for cn in cnames:
            f, p, bp = band_power(clips[cn]["jvel"], FS_REF, nperseg=nps_ref)
            ref_bp += bp / len(cnames)
            ref_psd = (f, p) if ref_psd is None else (f, ref_psd[1] + p)
        ref_psd = (ref_psd[0], ref_psd[1] / len(cnames))
        psd_store[("ref", cmd)] = ref_psd
        tot = ref_bp.sum(axis=0)
        band_rows.append(dict(cmd=cmd, who="ref", meas_vx=float("nan"),
                              frac=[float(ref_bp[i].sum() / ref_bp.sum()) for i in range(3)],
                              abs_hf=float(ref_bp[2].sum()),
                              grp_hf={g: float(ref_bp[2][[j for j, n in enumerate(jn) if group_of(n) == g]].sum()
                                               / max(tot[[j for j, n in enumerate(jn) if group_of(n) == g]].sum(), 1e-12))
                                      for g in JOINT_GROUPS}))
        # 정책
        for pol in POLICIES:
            bp_acc = np.zeros((len(BANDS), len(jn)))
            psd_acc = None
            vxs = []
            cnt = 0
            for s in SEEDS:
                r = ramps[pol][s]
                for si in stage_ids:
                    seg = r["jvel"][slices[si]]
                    vxs.append(float(r["vx"][slices[si]].mean()))
                    f, p, bp = band_power(seg, FS_POL, nperseg=100)
                    bp_acc += bp
                    psd_acc = p.copy() if psd_acc is None else psd_acc + p
                    cnt += 1
            bp_acc /= cnt
            psd_store[(pol, cmd)] = (f, psd_acc / cnt)
            tot = bp_acc.sum(axis=0)
            band_rows.append(dict(cmd=cmd, who=pol, meas_vx=float(np.mean(vxs)),
                                  frac=[float(bp_acc[i].sum() / bp_acc.sum()) for i in range(3)],
                                  abs_hf=float(bp_acc[2].sum()),
                                  grp_hf={g: float(bp_acc[2][[j for j, n in enumerate(jn) if group_of(n) == g]].sum()
                                                   / max(tot[[j for j, n in enumerate(jn) if group_of(n) == g]].sum(), 1e-12))
                                          for g in JOINT_GROUPS}))
    result["bands"] = band_rows

    # ── 3. 정지(cmd 0) PSD — stage 0(램프 전) / stage 14(램프 후) 분리 ────
    stand_rows = []
    stand_psd = {}
    for pol in POLICIES:
        for label, si in (("stage0_pre", 0), ("stage14_post", len(prof) - 1)):
            bp_acc = np.zeros((len(BANDS), len(jn)))
            psd_acc = None
            peak = 0.0
            for s in SEEDS:
                seg = ramps[pol][s]["jvel"][slices[si]]
                peak = max(peak, float(np.abs(seg).max()))
                f, p, bp = band_power(seg, FS_POL, nperseg=100)
                bp_acc += bp / len(SEEDS)
                psd_acc = p.copy() if psd_acc is None else psd_acc + p
            stand_psd[(pol, label)] = (f, psd_acc / len(SEEDS))
            tot = bp_acc.sum(axis=0)
            stand_rows.append(dict(pol=pol, stage=label, peak_jvel=peak,
                                   frac=[float(bp_acc[i].sum() / bp_acc.sum()) for i in range(3)],
                                   rms=float(np.sqrt((bp_acc.sum(axis=0)).sum())),
                                   grp_hf={g: float(bp_acc[2][[j for j, n in enumerate(jn) if group_of(n) == g]].sum()
                                                    / max(tot[[j for j, n in enumerate(jn) if group_of(n) == g]].sum(), 1e-12))
                                           for g in JOINT_GROUPS}))
    result["stand"] = stand_rows

    # ── 2. 저역통과 C2ST ──────────────────────────────────────────────────
    conds = [("raw", None), ("lp5", 5.0), ("lp3", 3.0)]
    c2st_rows = []
    for cmd, cnames in MATCHES.items():
        stage_ids = [i for i, v in enumerate(prof) if abs(v - cmd) < 1e-9]
        for cond, fc in conds:
            # 참조 특징: 필터(native 60Hz) → 50Hz 재표본화 → 엣지 절단
            ref_segs = []
            for cn in cnames:
                x = np.concatenate([clips[cn]["jpos"], clips[cn]["jvel"]], axis=1)
                if fc is not None:
                    x = lowpass(x, fc, FS_REF)
                x = x[trim_r : x.shape[0] - trim_r] if x.shape[0] > 2 * trim_r + 10 else x
                x = resample_to(x, FS_REF, FS_POL)
                ref_segs.append(x)
            pol_segs = {}
            for pol in POLICIES:
                segs = []
                for s in SEEDS:
                    r = ramps[pol][s]
                    for si in stage_ids:
                        x = np.concatenate([r["jpos"][slices[si]], r["jvel"][slices[si]]], axis=1)
                        if fc is not None:
                            x = lowpass(x, fc, FS_POL)
                        x = x[trim_p : x.shape[0] - trim_p]
                        segs.append(x)
                pol_segs[pol] = segs
                res = c2st(ref_segs, segs, rng)
                c2st_rows.append(dict(cmd=cmd, cond=cond, pair=f"ref_vs_{pol}", **res))
            # 음성/참조 대조
            #  (a) 같은 정책 시드 간 (memorization floor)
            for pol in POLICIES:
                half = len(pol_segs[pol]) // 2
                res = c2st(pol_segs[pol][:half], pol_segs[pol][half:], rng)
                c2st_rows.append(dict(cmd=cmd, cond=cond, pair=f"ctrl_seed_{pol}", **res))
            #  (b) 정책 A' vs B'
            res = c2st(pol_segs["A_condmlp50k"], pol_segs["B_conddrail45k"], rng)
            c2st_rows.append(dict(cmd=cmd, cond=cond, pair="ctrl_A_vs_B", **res))
            #  (c) 참조 클립 전반부 vs 후반부 (비정상성 포함 — 상한 아님)
            if len(ref_segs) >= 1:
                h1 = [s[: len(s) // 2] for s in ref_segs]
                h2 = [s[len(s) // 2 :] for s in ref_segs]
                res = c2st(h1, h2, rng)
                c2st_rows.append(dict(cmd=cmd, cond=cond, pair="ctrl_ref_halves", **res))
    result["c2st"] = c2st_rows

    with open("/tmp/claude-1002/-home-lgb-IsaacLab-6-0/0acd1c66-e03c-47df-9285-08e899ea3b28/scratchpad/gap_spectral.json", "w") as f:
        json.dump(result, f, indent=1)

    # ── 플롯 ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MATCHES) + 1, figsize=(5 * (len(MATCHES) + 1), 4))
    for ax, cmd in zip(axes[:-1], MATCHES):
        for who, lab, c in (("ref", "reference", "k"), ("A_condmlp50k", "A' cond-mlp", "C0"),
                            ("B_conddrail45k", "B' cond+drail", "C3")):
            f, p = psd_store[(who, cmd)]
            m = f > 0
            ax.loglog(f[m], p[m].sum(axis=1), color=c, label=lab)
        ax.set_title(f"cmd {cmd} vs {'+'.join(MATCHES[cmd])}", fontsize=8)
        ax.set_xlabel("freq [Hz]")
        ax.set_ylabel("joint-vel PSD sum [(rad/s)^2/Hz]")
        ax.axvline(3, ls=":", c="gray")
        ax.axvline(8, ls=":", c="gray")
        ax.legend(fontsize=7)
    ax = axes[-1]
    for (pol, label), (f, p) in stand_psd.items():
        ls = "-" if "pre" in label else "--"
        c = "C0" if pol.startswith("A") else "C3"
        m = f > 0
        ax.loglog(f[m], p[m].sum(axis=1), ls, color=c, label=f"{pol} {label}", lw=1.2)
    ax.set_title("cmd 0 (standstill), no reference", fontsize=8)
    ax.set_xlabel("freq [Hz]")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/gap_psd.png", dpi=130)

    # C2ST 막대
    fig, axes = plt.subplots(1, len(MATCHES), figsize=(4.2 * len(MATCHES), 4), sharey=True)
    for ax, cmd in zip(np.atleast_1d(axes), MATCHES):
        pairs = ["ref_vs_A_condmlp50k", "ref_vs_B_conddrail45k", "ctrl_seed_A_condmlp50k",
                 "ctrl_seed_B_conddrail45k", "ctrl_A_vs_B", "ctrl_ref_halves"]
        w = 0.26
        for i, (cond, _) in enumerate(conds):
            vals = []
            for pr in pairs:
                r = [x for x in c2st_rows if x["cmd"] == cmd and x["cond"] == cond and x["pair"] == pr]
                vals.append(r[0]["acc_mlp"] if r else np.nan)
            ax.bar(np.arange(len(pairs)) + (i - 1) * w, vals, w, label=cond)
        ax.axhline(0.5, c="k", ls=":")
        ax.set_xticks(np.arange(len(pairs)))
        ax.set_xticklabels([p.replace("ref_vs_", "ref/").replace("ctrl_", "ctl:") for p in pairs],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(f"cmd {cmd}", fontsize=9)
        ax.set_ylim(0.4, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("C2ST held-out acc (MLP, block 5-fold)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/gap_c2st_lowpass.png", dpi=130)
    print("done")


if __name__ == "__main__":
    main()
