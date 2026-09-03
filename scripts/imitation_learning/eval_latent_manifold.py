# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""latent 다양체 게이트 — 곡선 자로 재는 판정 (Phase 1 재분석).

기존 게이트는 ``walk`` 로 적합한 **선형** 프로브를 ``run``/``trot`` 에 적용해 상관을 봤다.
그 지표는 **속도축이 전역적으로 직선**이라고 가정한다. latent 이 곡선이면 서로 다른 지점의
접선 방향이 다르고, 곡선이 90도 넘게 휘면 내적이 음수가 되므로 **올바른 latent 도 탈락한다.**

여기서는 선형성을 가정하지 않고 잰다. 재학습은 하지 않으며 기존 체크포인트만 읽는다.

측정 항목::

  M1  거리-속도 상관    쌍 (i,j) 의 latent 거리와 |vx_i - vx_j| 의 Spearman
                        전체 / 카테고리 내 / 카테고리 간을 나눠 낸다
  M2  betweenness       속도로 정렬한 삼중쌍 (i<j<k) 에서 d(i,j)+d(j,k) 가 d(i,k) 에 얼마나 가까운가
                        1.0 에 가까울수록 z_j 가 두 점 사이에 놓인다 = 1-D 순서 구조
  M3  경로 통과         가장 느린 walk -> 가장 빠른 run 측지 경로가 trot 군집을 지나는가
  M4  디코더 응답       그 경로를 따라 한 스텝 디코드했을 때 예측 vx 의 단조성과 효과크기
  G1' 절대 일반화       val_rec / train_rec / gap. 비율 게이트가 가리는 것을 드러낸다

거리와 보간은 latent 기하에 맞춘다::

  vmf   : 단위 구.  d = arccos(z_i . z_j) 측지거리,  보간 = slerp
  gauss : 유클리드. d = ||z_i - z_j||,               보간 = 선형
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Geometry:
    """latent 분포에 맞는 거리·보간·평균."""

    def __init__(self, is_sphere: bool):
        self.is_sphere = is_sphere

    def dist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """행 단위 거리. a, b: [N, d]"""
        if self.is_sphere:
            return np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))
        return np.linalg.norm(a - b, axis=-1)

    def mean(self, z: np.ndarray) -> np.ndarray:
        """대표점. 구에서는 평균 후 재정규화(resultant 방향)."""
        m = z.mean(0)
        if self.is_sphere:
            n = np.linalg.norm(m)
            m = m / n if n > 1e-9 else np.eye(len(m))[0]
        return m

    def interp(self, a: np.ndarray, b: np.ndarray, alphas: np.ndarray) -> np.ndarray:
        """a -> b 경로. 구에서는 slerp (선형보간은 노름이 줄어 크기 아티팩트가 섞인다)."""
        if not self.is_sphere:
            return (1 - alphas)[:, None] * a[None, :] + alphas[:, None] * b[None, :]
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        omega = np.arccos(dot)
        if omega < 1e-6:
            return np.repeat(a[None, :], len(alphas), axis=0)
        s = np.sin(omega)
        w0 = np.sin((1 - alphas) * omega) / s
        w1 = np.sin(alphas * omega) / s
        return w0[:, None] * a[None, :] + w1[:, None] * b[None, :]


# ──────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="motion_vae.pt 경로")
    p.add_argument("--categories", nargs="*", default=["walk", "trot", "run"])
    p.add_argument("--num_alpha", type=int, default=21, help="경로 표본 수")
    p.add_argument("--num_triples", type=int, default=4000, help="betweenness 표본 삼중쌍 수")
    p.add_argument("--num_starts", type=int, default=8, help="M4 에서 쓸 시작 상태 수 (편향 확인용)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_json", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, weights_only=False)
    targs = ck["args"]

    here = os.path.dirname(os.path.abspath(__file__))
    tmod = load_module(os.path.join(here, "train_go2_motion_vae.py"), "tmod")
    ml = load_module(targs["motion_lib"], "ml")

    is_sphere = targs.get("latent_dist", "gauss") == "vmf"
    geo = Geometry(is_sphere)

    model = tmod.MotionVAE(
        targs["latent_dim"],
        targs["hidden"],
        targs["num_experts"],
        latent_dist=targs.get("latent_dist", "gauss"),
        kappa_fixed=targs.get("kappa_fixed", 0.0),
        gauss_fixed_logvar=targs.get("gauss_fixed_logvar", None),
    ).to(dev)
    # `speed_head` 는 축 정렬 손실용 보조 헤드이고 encode/decode 경로에 관여하지 않는다.
    # 그 이전에 저장된 체크포인트에는 없으므로 strict=False 로 읽되, 빠진 키가 보조 헤드
    # 이외라면 즉시 멈춘다 (조용히 랜덤 가중치로 도는 것을 막는다).
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith("speed_head")]
    if bad:
        raise SystemExit(f"체크포인트 키 불일치: {bad}")
    model.eval()
    mean, std = ck["mean"], ck["std"]
    vx_i = tmod.FEATURE_SLICES["lin_vel"].start
    stride = targs.get("stride", 1)

    # ── val 세션 클립 수집 ──────────────────────────────────────
    with open(targs["split_json"]) as fh:
        split = json.load(fh)
    val_sessions = set(split["val_sessions"])
    clips: list[dict] = []
    for cat in args.categories:
        for path in sorted(glob.glob(os.path.join(targs["pkl_dir"], f"{cat}__*.pkl"))):
            m = tmod.CLIP_NAME_RE.match(os.path.splitext(os.path.basename(path))[0])
            if m is None or m.group(2) not in val_sessions:
                continue
            x = tmod.build_x_vae(ml, path)[::stride]
            if len(x) < 4:
                continue
            clips.append({"cat": cat, "x": x, "vx": float(x[:, vx_i].mean())})
    if len(clips) < 6:
        raise SystemExit(f"val 클립이 부족하다 ({len(clips)}개).")

    @torch.no_grad()
    def encode_clip(x: np.ndarray) -> np.ndarray:
        xn = torch.from_numpy((x - mean) / std).to(dev)
        mu, _ = model.encode(xn[:-1], xn[1:])
        return geo.mean(mu.cpu().numpy().astype(np.float64))

    for c in clips:
        c["z"] = encode_clip(c["x"])

    Z = np.stack([c["z"] for c in clips])
    V = np.array([c["vx"] for c in clips])
    C = np.array([c["cat"] for c in clips])
    n = len(clips)
    print(f"[데이터] val 클립 {n}개 — " + ", ".join(f"{cat} {int((cat == C).sum())}" for cat in args.categories))
    print(
        f"[기하] {'구면 (slerp / arccos)' if is_sphere else '유클리드 (선형 / L2)'}"
        f",  latent {Z.shape[1]}-D,  stride {stride},  predict_ahead {targs.get('predict_ahead', 1)}"
    )

    # ── M1 거리-속도 상관 ───────────────────────────────────────
    iu, ju = np.triu_indices(n, k=1)
    d_pair = geo.dist(Z[iu], Z[ju])
    dv_pair = np.abs(V[iu] - V[ju])
    same_cat = C[iu] == C[ju]
    m1_all = spearman(d_pair, dv_pair)
    m1_in = spearman(d_pair[same_cat], dv_pair[same_cat]) if same_cat.sum() > 2 else float("nan")
    m1_cross = spearman(d_pair[~same_cat], dv_pair[~same_cat]) if (~same_cat).sum() > 2 else float("nan")
    print("\n[M1] 거리-속도 상관 (선형성 가정 없음)")
    print(f"     전체 {m1_all:+.3f}   카테고리 내 {m1_in:+.3f}   **카테고리 간 {m1_cross:+.3f}**   기준 >= +0.5")

    # ── M2 betweenness ─────────────────────────────────────────
    order = np.argsort(V)
    ratios = []
    for _ in range(args.num_triples):
        a, b, c_ = np.sort(rng.choice(n, 3, replace=False))
        i_, j_, k_ = order[a], order[b], order[c_]
        dik = geo.dist(Z[i_][None], Z[k_][None])[0]
        if dik < 1e-6:
            continue
        ratios.append((geo.dist(Z[i_][None], Z[j_][None])[0] + geo.dist(Z[j_][None], Z[k_][None])[0]) / dik)
    ratios = np.array(ratios)
    m2_med = float(np.median(ratios))
    m2_frac = float((ratios <= 1.25).mean())
    print(f"\n[M2] betweenness — 속도로 정렬한 삼중쌍 {len(ratios)}개")
    print(f"     (d_ij + d_jk) / d_ik  중앙값 {m2_med:.3f}   <=1.25 인 비율 {m2_frac:.0%}   (1.0 = 완전한 1-D 순서)")

    # ── M3 경로가 trot 을 지나는가 ──────────────────────────────
    walk_idx = np.where(C == "walk")[0]
    run_idx = np.where(C == "run")[0]
    trot_idx = np.where(C == "trot")[0]
    m3 = None
    if len(walk_idx) and len(run_idx) and len(trot_idx):
        i_slow = walk_idx[np.argmin(V[walk_idx])]
        i_fast = run_idx[np.argmax(V[run_idx])]
        alphas = np.linspace(0.0, 1.0, args.num_alpha)
        path = geo.interp(Z[i_slow], Z[i_fast], alphas)
        if is_sphere:
            path = path / np.linalg.norm(path, axis=-1, keepdims=True)

        trot_c = geo.mean(Z[trot_idx])
        d_to_trot = geo.dist(path, np.repeat(trot_c[None], len(path), 0))
        a_star = float(alphas[int(np.argmin(d_to_trot))])
        trot_radius = float(np.median(geo.dist(Z[trot_idx], np.repeat(trot_c[None], len(trot_idx), 0))))
        closest = float(d_to_trot.min())
        # walk / run 군집 중심까지의 거리와 비교해 "가까움" 을 상대화한다
        walk_c, run_c = geo.mean(Z[walk_idx]), geo.mean(Z[run_idx])
        span = float(geo.dist(walk_c[None], run_c[None])[0])
        m3 = {"alpha_star": a_star, "closest": closest, "trot_radius": trot_radius, "span": span}
        print("\n[M3] 경로 통과 — 가장 느린 walk -> 가장 빠른 run")
        print(f"     trot 중심에 가장 가까워지는 지점  alpha* = {a_star:.2f}   (0=walk, 1=run)")
        print(f"     그때 거리 {closest:.3f}   trot 군집 반경 {trot_radius:.3f}   walk-run 간격 {span:.3f}")
        print(
            f"     판정: alpha* 가 중간(0.2~0.8) 이고 거리 <= 반경  ->  "
            f"{'통과' if 0.2 <= a_star <= 0.8 and closest <= trot_radius else '미통과'}"
        )

    # ── M4 경로 위 디코더 응답 ─────────────────────────────────
    @torch.no_grad()
    def decode_vx(x0: np.ndarray, zrow: np.ndarray) -> float:
        xc = torch.from_numpy((x0 - mean) / std).to(dev).unsqueeze(0).float()
        zt = torch.from_numpy(zrow).to(dev).unsqueeze(0).float()
        out = model.decode(xc, zt)
        return float(out[0, vx_i].item() * std[vx_i] + mean[vx_i])

    m4 = None
    if m3 is not None:
        alphas = np.linspace(0.0, 1.0, args.num_alpha)
        path = geo.interp(Z[i_slow], Z[i_fast], alphas)
        if is_sphere:
            path = path / np.linalg.norm(path, axis=-1, keepdims=True)
        # 시작 상태를 여러 클립에서 뽑아 편향을 확인한다
        starts = [clips[k]["x"][0] for k in rng.choice(n, min(args.num_starts, n), replace=False)]
        rhos, effs, curves = [], [], []
        for x0 in starts:
            sp = np.array([decode_vx(x0, path[t]) for t in range(len(alphas))])
            rhos.append(spearman(alphas, sp))
            effs.append((sp.max() - sp.min()) / max(V[i_fast] - V[i_slow], 1e-9))
            curves.append(sp.tolist())
        m4 = {
            "rho_median": float(np.median(rhos)),
            "rho_min": float(np.min(rhos)),
            "eff_median": float(np.median(effs)),
            "curves": curves,
            "vx_span": float(V[i_fast] - V[i_slow]),
        }
        ok4 = np.median(rhos) >= 0.7 and np.median(effs) >= 0.10
        print(f"\n[M4] 경로 위 디코더 응답 — 시작 상태 {len(starts)}개")
        print(f"     단조성 rho 중앙값 {np.median(rhos):+.3f} [최소 {np.min(rhos):+.3f}]")
        print(f"     효과크기 중앙값 {np.median(effs):.1%}   (경로 양끝 실측 속도차 {V[i_fast] - V[i_slow]:.2f} m/s)")
        print(f"     기준 rho >= 0.7 이고 효과 >= 10%  ->  {'PASS' if ok4 else 'FAIL'}")

    # ── M5 gait 분리도 (인코더) ─────────────────────────────────
    # z 만 보고 걸음걸이를 맞출 수 있는가. 못 맞추면 정책이 gait 를 고를 근거가 없다.
    # walk 141 / trot 16 / run 16 로 불균형하므로 **클래스 균형 정확도**로 낸다.
    cats_present = [c for c in args.categories if (c == C).sum() >= 3]
    per_cls: dict[str, float] = {}
    for cat in cats_present:
        idx = np.where(cat == C)[0]
        hit = 0
        for i_ in idx:
            # 자기 자신을 뺀 중심들과 비교 (leave-one-out)
            best, bestd = None, np.inf
            for cc in cats_present:
                sel = np.where(cc == C)[0]
                sel = sel[sel != i_]
                if len(sel) == 0:
                    continue
                d_ = geo.dist(Z[i_][None], geo.mean(Z[sel])[None])[0]
                if d_ < bestd:
                    best, bestd = cc, d_
            hit += int(best == cat)
        per_cls[cat] = hit / len(idx)
    m5 = float(np.mean(list(per_cls.values()))) if per_cls else float("nan")
    print("\n[M5] gait 분리도 (인코더, LOO 최근접중심, 균형 정확도)")
    print(
        "     "
        + "   ".join(f"{c} {per_cls[c]:.0%}" for c in cats_present)
        + f"   ->  균형 {m5:.0%}   기준 >= 70% (우연 {1 / max(len(cats_present), 1):.0%})"
    )

    # ── M6 gait 조종 가능성 (디코더) ────────────────────────────
    # z 를 trot 중심 쪽으로 옮기면 디코드 결과가 실제로 trot 쪽 상태로 가는가.
    # 표적 방향 = (trot 평균상태 - walk 평균상태), 정규화 x 공간에서.
    m6 = None
    if len(walk_idx) and len(trot_idx):
        xw = np.concatenate([clips[k]["x"] for k in walk_idx]).mean(0)
        xt = np.concatenate([clips[k]["x"] for k in trot_idx]).mean(0)
        tgt = (xt - xw) / std
        tgt_n2 = float((tgt * tgt).sum())
        z_trot_c = geo.mean(Z[trot_idx])
        alphas6 = np.linspace(0.0, 1.0, args.num_alpha)
        projs = []
        for k in rng.choice(walk_idx, min(args.num_starts, len(walk_idx)), replace=False):
            x0 = clips[k]["x"][0]
            pth = geo.interp(Z[k], z_trot_c, alphas6)
            if is_sphere:
                pth = pth / np.linalg.norm(pth, axis=-1, keepdims=True)
            row = []
            for t in range(len(alphas6)):
                xc = torch.from_numpy((x0 - mean) / std).to(dev).unsqueeze(0).float()
                zt_ = torch.from_numpy(pth[t]).to(dev).unsqueeze(0).float()
                with torch.no_grad():
                    xh = model.decode(xc, zt_)[0].cpu().numpy().astype(np.float64)
                # walk 평균에서 출발해 trot 평균 쪽으로 얼마나 갔는가. 1.0 = 완전 도달
                row.append(float(((xh - (xw - mean) / std) * tgt).sum() / max(tgt_n2, 1e-12)))
            projs.append(row)
        P = np.array(projs)
        m6_rho = float(np.median([spearman(alphas6, r) for r in P]))
        m6_reach = float(np.median(P[:, -1] - P[:, 0]))
        m6 = {"rho_median": m6_rho, "reach_median": m6_reach, "curves": P.tolist()}
        ok6 = m6_rho >= 0.7 and m6_reach >= 0.30
        print("\n[M6] gait 조종 가능성 (디코더) — z 를 trot 중심으로 옮길 때")
        print(
            f"     walk→trot 방향 진행률 rho 중앙값 {m6_rho:+.3f}   도달률 {m6_reach:+.2f}"
            f"   (1.0 = trot 평균상태 완전 도달)"
        )
        print(f"     기준 rho >= 0.7 이고 도달률 >= 0.30  ->  {'PASS' if ok6 else 'FAIL'}")

    # ── G1' 절대 일반화 ─────────────────────────────────────────
    g1p = {k: ck.get("g1", {}).get(k) for k in ("train_rec", "val_rec", "ratio")}
    if g1p["val_rec"] is not None:
        gap = g1p["val_rec"] - g1p["train_rec"]
        print("\n[G1'] 절대 일반화 (비율 게이트가 가리는 것)")
        print(
            f"     train_rec {g1p['train_rec']:.5f}   val_rec {g1p['val_rec']:.5f}   "
            f"gap {gap:.5f}   비율 {g1p['ratio']:.3f}"
        )
        print("     ※ 비율은 예측 지평(stride, predict_ahead)이 같은 설정끼리만 비교 가능하다")
        g1p["gap"] = gap

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as _out_fh:
            json.dump(
                {
                    "ckpt": args.ckpt,
                    "latent_dist": targs.get("latent_dist", "gauss"),
                    "stride": stride,
                    "predict_ahead": targs.get("predict_ahead", 1),
                    "num_clips": n,
                    "counts": {c: int((c == C).sum()) for c in args.categories},
                    "m1": {"all": m1_all, "within": m1_in, "cross": m1_cross},
                    "m2": {"median_ratio": m2_med, "frac_le_1.25": m2_frac},
                    "m3": m3,
                    "m4": m4,
                    "m5": {"balanced_acc": m5, "per_class": per_cls},
                    "m6": m6,
                    "g1_absolute": g1p,
                },
                _out_fh,
                indent=2,
            )
        print(f"\n[저장] {args.out_json}")


if __name__ == "__main__":
    main()
