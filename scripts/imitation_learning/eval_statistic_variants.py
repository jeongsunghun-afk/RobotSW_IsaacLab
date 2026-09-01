# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""동결 인코더는 그대로 두고 **보상 통계량만** 갈아 끼워 판별력을 비교한다.

배경 (2026-09-01): 같은 latent 인데 스텝별 kNN 은 ``frontswap`` 을 0.729 로 잡고
배포 중인 창 marginal KL 은 0.499(우연)다. 정보는 latent 에 있고 통계량이 버린다.
반대로 ``static``(정지)은 창 marginal KL 만 1.000 이고 스텝별은 0.411 로 **반전**된다.
그래서 필요한 것은 "정지도 잡고 국소도 잡는" 통계량이며, 이 스크립트가 그 후보를 잰다.

비교 대상 (전부 **창 N개 전이쌍** 단위 — 그래야 보상으로 쓸 수 있다):
  mkl_diag   현행. KL( N(mu_w, diag var_w) || N(mu_ref, diag var_ref) )
  mkl_full   전체 공분산 판. 창 공분산은 N < D 라 특이하므로 수축(shrinkage)한다
  gmm        참조를 대각 GMM(k개 mode)로 두고 창 내 스텝별 -log p 평균
  knn        참조 표본에 대한 창 내 스텝별 k-번째 최근접 거리 평균
  hybrid     mkl_diag 와 gmm 을 각각 expert 분포로 표준화해 더한 것

AUROC 는 **점수가 클수록 negative** 방향이다 (validator 와 동일).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────────────────────
# 대각 공분산 GMM (sklearn 이 isaac-6.0 env 에 없어 EM 을 직접 쓴다)
# ──────────────────────────────────────────────────────────────
def fit_diag_gmm(z: torch.Tensor, k: int, iters: int = 120, seed: int = 0, var_floor: float = 1e-4):
    """대각 공분산 GMM 을 EM 으로 적합. 반환 (log_w [k], mu [k,D], var [k,D])."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, d = z.shape
    # k-means++ 대신 무작위 초기화 + 전체 분산 — 단순하지만 재현 가능
    idx = torch.randperm(n, generator=g)[:k]
    mu = z[idx.to(z.device)].clone()
    var = z.var(0, unbiased=False).clamp(min=var_floor).repeat(k, 1)
    log_w = torch.full((k,), -np.log(k), device=z.device, dtype=z.dtype)
    for _ in range(iters):
        # E-step — log N(z; mu_j, var_j)
        lp = -0.5 * (
            ((z[:, None, :] - mu[None]) ** 2 / var[None]).sum(-1)
            + torch.log(var).sum(-1)[None]
            + d * np.log(2 * np.pi)
        )
        lp = lp + log_w[None]
        lse = torch.logsumexp(lp, dim=1, keepdim=True)
        r = torch.exp(lp - lse)                                   # [n, k]
        nk = r.sum(0).clamp(min=1e-8)
        log_w = torch.log(nk / n)
        mu = (r.T @ z) / nk[:, None]
        var = ((r.T @ (z ** 2)) / nk[:, None] - mu ** 2).clamp(min=var_floor)
    return log_w, mu, var


def gmm_neg_logprob(z: torch.Tensor, log_w, mu, var) -> torch.Tensor:
    d = z.shape[1]
    lp = -0.5 * (
        ((z[:, None, :] - mu[None]) ** 2 / var[None]).sum(-1)
        + torch.log(var).sum(-1)[None]
        + d * np.log(2 * np.pi)
    )
    return -torch.logsumexp(lp + log_w[None], dim=1)


def knn_dist(z: torch.Tensor, ref: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    out = []
    for i in range(0, len(z), chunk):
        d = torch.cdist(z[i : i + chunk], ref)
        out.append(d.topk(k, largest=False).values[:, -1])
    return torch.cat(out)


# ──────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rep = f"{REPO}/reports/go2_imitation/go2_imitation_tracking"
    root = f"{REPO}/source/isaaclab_tasks/isaaclab_tasks/direct"
    p.add_argument("--ckpt", default=f"{rep}/2026-08-27_phase1_motion_vae/gaussfix_reg/runs/base_s0/motion_vae.pt")
    p.add_argument("--validator", default=os.path.join(HERE, "validate_latent_style_reward.py"))
    p.add_argument("--pkl_dir", default=f"{root}/go2_imitation_latent/imitation/motion_pkl")
    p.add_argument("--motion_lib", default=f"{root}/go2_imitation_latent/motion_lib.py")
    p.add_argument("--trainer", default=os.path.join(HERE, "train_go2_motion_vae.py"))
    p.add_argument("--split_json", default=f"{rep}/2026-08-27_phase0_dataset_build/metrics/session_split.json")
    p.add_argument("--window_n", type=int, default=8, help="창 길이 (배포값 8)")
    p.add_argument("--windows_per_clip", type=int, default=24)
    p.add_argument("--gmm_k", nargs="*", type=int, default=[4, 8, 16])
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--ref_max", type=int, default=20000)
    p.add_argument("--shrink_sweep", nargs="*", type=float, default=[],
                   help="여러 수축계수의 mkl_full 을 한 번의 인코딩으로 동시에 잰다 (민감도용).")
    p.add_argument("--shrink", type=float, default=0.20, help="창 공분산 수축 계수 (N<D 특이성 방어)")
    p.add_argument(
        "--policy_npz", nargs="*", default=[],
        help="`이름=경로.npz` 형식. 정책이 실제로 만든 x_vae 시퀀스를 negative 로 추가한다. "
             "합성 negative 와 달리 **정책의 실제 상태**라 '떠는 정지' 를 그대로 잰다.")
    p.add_argument(
        "--pair_dt", type=float, default=None,
        help="전이 쌍 간격 [s]. 미지정이면 ref_stats 의 값. policy npz 는 50 Hz 라 "
             "env 와 같은 선형보간으로 이 간격을 맞춘다.")
    p.add_argument("--out_json", default=f"{rep}/2026-09-01_pose_dropout_and_statistic/statistic_variants.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    v = load_module(a.validator, "v")
    ml, tr = v.load_modules(a.motion_lib, a.trainer)

    ck = torch.load(a.ckpt, weights_only=False)
    targs = ck["args"]
    stride = targs.get("stride", 1)
    model = tr.MotionVAE(
        targs["latent_dim"], targs["hidden"], targs["num_experts"],
        latent_dist=targs.get("latent_dist", "gauss"),
        kappa_fixed=targs.get("kappa_fixed", 0.0),
        gauss_fixed_logvar=targs.get("gauss_fixed_logvar", None),
    ).to(dev)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in list(miss) + list(unexp) if not k.startswith("speed_head")]
    if bad:
        raise SystemExit(f"체크포인트 키 불일치: {bad}")
    model.eval()
    mean, std = ck["mean"], ck["std"]

    val_sessions = set(json.load(open(a.split_json))["val_sessions"])
    train_clips, val_clips = v.load_expert_clips(ml, tr, a.pkl_dir, val_sessions)
    print(f"[데이터] train 클립 {len(train_clips)} / val 클립 {len(val_clips)}")

    @torch.no_grad()
    def encode(x60: np.ndarray) -> np.ndarray:
        x = x60[::stride]
        if len(x) < 2:
            return np.zeros((0, targs["latent_dim"]), np.float32)
        xn = torch.from_numpy((x - mean) / std).to(dev)
        mu, _ = model.encode(xn[:-1], xn[1:])
        return mu.cpu().numpy().astype(np.float32)

    # ── 참조 (train 세션) ────────────────────────────────────
    ref = np.concatenate([encode(c["x"]) for c in train_clips])
    if len(ref) > a.ref_max:
        ref = ref[rng.choice(len(ref), a.ref_max, replace=False)]
    ref_t = torch.from_numpy(ref).to(dev)
    mu_ref = ref_t.mean(0)
    var_ref = ref_t.var(0, unbiased=False).clamp(min=1e-4)
    cov_ref = torch.from_numpy(np.cov(ref.T)).float().to(dev)
    D = ref.shape[1]
    cov_ref = cov_ref + 1e-4 * torch.eye(D, device=dev)
    L_ref = torch.linalg.cholesky(cov_ref)
    logdet_ref = 2.0 * torch.log(torch.diagonal(L_ref)).sum()
    inv_ref = torch.cholesky_inverse(L_ref)
    print(f"[참조] 전이쌍 {len(ref):,}개, latent {D}-D")

    gmms = {}
    for k in a.gmm_k:
        gmms[k] = fit_diag_gmm(ref_t, k, seed=a.seed)
        print(f"[GMM] k={k} 적합 완료")

    # ── 창 통계량 ────────────────────────────────────────────
    N = a.window_n

    def window_scores(z_np: np.ndarray) -> dict[str, list[float]]:
        """클립 하나 → 창별 점수. 창은 겹치지 않게 균등 표본."""
        out: dict[str, list[float]] = {}
        if len(z_np) < N:
            return out
        starts = np.arange(0, len(z_np) - N + 1)
        if len(starts) > a.windows_per_clip:
            starts = starts[np.linspace(0, len(starts) - 1, a.windows_per_clip).astype(int)]
        z = torch.from_numpy(z_np).to(dev)
        # 스텝별 점수는 클립 전체에 대해 한 번만 계산하고 창으로 자른다
        step_gmm = {k: gmm_neg_logprob(z, *gmms[k]).cpu().numpy() for k in a.gmm_k}
        step_knn = knn_dist(z, ref_t, a.knn_k).cpu().numpy()
        for s in starts:
            W = z[s : s + N]
            mu_w = W.mean(0)
            var_w = W.var(0, unbiased=False).clamp(min=1e-4)
            out.setdefault("mkl_diag", []).append(float(
                0.5 * (torch.log(var_ref / var_w) + (var_w + (mu_w - mu_ref) ** 2) / var_ref - 1.0).sum()
            ))
            Wc = W - mu_w
            cov_raw = (Wc.T @ Wc) / N
            cov_w = (1 - a.shrink) * cov_raw + a.shrink * torch.diag(var_ref)
            dmu = (mu_w - mu_ref)[:, None]
            sign, logdet_w = torch.linalg.slogdet(cov_w)
            out.setdefault("mkl_full", []).append(float(
                0.5 * ((inv_ref @ cov_w).trace() + (dmu.T @ inv_ref @ dmu).squeeze()
                       - D + logdet_ref - logdet_w)
            ))
            for _sh in a.shrink_sweep:
                _cw = (1 - _sh) * cov_raw + _sh * torch.diag(var_ref)
                _sg, _ld = torch.linalg.slogdet(_cw)
                out.setdefault(f"mklfull_sh{_sh:g}", []).append(float(
                    0.5 * ((inv_ref @ _cw).trace() + (dmu.T @ inv_ref @ dmu).squeeze()
                           - D + logdet_ref - _ld)
                ))
            for k in a.gmm_k:
                out.setdefault(f"gmm{k}", []).append(float(step_gmm[k][s : s + N].mean()))
            out.setdefault("knn", []).append(float(step_knn[s : s + N].mean()))
        return out

    # ── 정책 실측 시퀀스 (선택) ──────────────────────────────
    pair_dt = a.pair_dt
    if pair_dt is None:
        _rs = os.path.join(os.path.dirname(a.ckpt), "latent_ref_stats.pt")
        pair_dt = float(torch.load(_rs, weights_only=False)["pair_dt"]) if os.path.exists(_rs) else 1 / 30
    POLICY_DT = 0.02
    lag = pair_dt / POLICY_DT
    lo = int(np.floor(lag))
    frac = float(lag - lo)
    print(f"[정책 시퀀스] pair_dt {pair_dt*1000:.1f} ms → 50 Hz 에서 lag {lag:.3f} step "
          f"(lo {lo}, frac {frac:.3f}) — env `_update_style_reward` 와 같은 보간")

    @torch.no_grad()
    def encode_policy_seq(X: np.ndarray) -> np.ndarray:
        """[T, 49] 정책 실측 → z [T-lo-1, D]. env 와 같이 x_prev 를 선형보간한다."""
        if len(X) < lo + 2:
            return np.zeros((0, targs["latent_dim"]), np.float32)
        cur = X[lo + 1 :]
        prv = (1.0 - frac) * X[1 : len(X) - lo] + frac * X[: len(X) - lo - 1]
        xc = torch.from_numpy(((cur - mean) / std).astype(np.float32)).to(dev)
        xp = torch.from_numpy(((prv - mean) / std).astype(np.float32)).to(dev)
        mu, _ = model.encode(xp, xc)
        return mu.cpu().numpy().astype(np.float32)

    policy_sets: dict[str, list[np.ndarray]] = {}
    for spec in a.policy_npz:
        name, path = spec.split("=", 1)
        Z = np.load(path, allow_pickle=True)
        seqs = Z["x_vae"]                                    # [N_env, T, 49]
        eplen = Z["eplen"] if "eplen" in Z else None
        cut = []
        for e in range(len(seqs)):
            X = seqs[e]
            if eplen is not None:                            # 에피소드 경계에서 자른다
                b = np.flatnonzero(np.diff(eplen[e]) < 0) + 1
                for seg in np.split(np.arange(len(X)), b):
                    if len(seg) >= 64:
                        cut.append(X[seg])
            elif len(X) >= 64:
                cut.append(X)
        policy_sets[name] = cut
        print(f"[정책 시퀀스] {name}: env {len(seqs)}개 → 에피소드 조각 {len(cut)}개 "
              f"(중앙 길이 {int(np.median([len(c) for c in cut]))} 프레임)")

    NEGS = ["shuffle", "reverse", "speed2x", "static", "legswap", "frontswap", "rearswap", "desync2", "desync5"]
    NEGS += list(policy_sets.keys())

    def clip_scores(clips: list[dict], kind: str | None) -> dict[str, np.ndarray]:
        agg: dict[str, list[float]] = {}
        if kind in policy_sets:
            for X in policy_sets[kind]:
                ws = window_scores(encode_policy_seq(X))
                for key, vals in ws.items():
                    if vals:
                        agg.setdefault(key, []).append(float(np.median(vals)))
            return {k: np.asarray(vv) for k, vv in agg.items()}
        for c in clips:
            x = c["x"] if kind is None else v.make_negative(c["x"], kind, stride, rng)
            if x is None or len(x) < 4:
                continue
            ws = window_scores(encode(x))
            for key, vals in ws.items():
                if vals:
                    agg.setdefault(key, []).append(float(np.median(vals)))
        return {k: np.asarray(vv) for k, vv in agg.items()}

    exp_s = clip_scores(val_clips, None)
    stats = sorted(exp_s.keys())
    print(f"[expert] 클립 {len(next(iter(exp_s.values()))):,}개, 통계량 {stats}")

    # hybrid 표준화 상수는 **expert 분포**에서만 뽑는다 (negative 를 보지 않는다)
    norm = {k: (float(exp_s[k].mean()), float(exp_s[k].std() + 1e-12)) for k in stats}

    def add_hybrid(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for k in a.gmm_k:
            if "mkl_diag" in d and f"gmm{k}" in d:
                za = (d["mkl_diag"] - norm["mkl_diag"][0]) / norm["mkl_diag"][1]
                zb = (d[f"gmm{k}"] - norm[f"gmm{k}"][0]) / norm[f"gmm{k}"][1]
                d[f"hybrid_gmm{k}"] = za + zb
        if "mkl_diag" in d and "knn" in d:
            za = (d["mkl_diag"] - norm["mkl_diag"][0]) / norm["mkl_diag"][1]
            zb = (d["knn"] - norm["knn"][0]) / norm["knn"][1]
            d["hybrid_knn"] = za + zb
        return d

    exp_s = add_hybrid(exp_s)
    all_stats = sorted(exp_s.keys())
    table = {}
    for neg in NEGS:
        ns = add_hybrid(clip_scores(val_clips, neg))
        table[neg] = {k: v.auroc(exp_s[k], ns[k]) for k in all_stats if k in ns}
        print(f"[{neg}] " + "  ".join(f"{k} {table[neg][k]:.3f}" for k in all_stats if k in table[neg]))

    hdr = f"{'negative':11}" + "".join(f"{k:>13}" for k in all_stats)
    print("\n" + hdr)
    print("-" * len(hdr))
    for neg in NEGS:
        print(f"{neg:11}" + "".join(f"{table[neg].get(k, float('nan')):>13.3f}" for k in all_stats))
    worst = {k: min(table[n][k] for n in NEGS if k in table[n]) for k in all_stats}
    meanv = {k: float(np.mean([table[n][k] for n in NEGS if k in table[n]])) for k in all_stats}
    print(f"{'최솟값':11}" + "".join(f"{worst[k]:>13.3f}" for k in all_stats))
    print(f"{'평균':11}" + "".join(f"{meanv[k]:>13.3f}" for k in all_stats))

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump({"ckpt": a.ckpt, "window_n": N, "table": table, "worst": worst, "mean": meanv,
               "n_ref": int(len(ref)), "n_val_clips": len(val_clips)},
              open(a.out_json, "w"), indent=1)
    print(f"\n[저장] {a.out_json}")


if __name__ == "__main__":
    main()
