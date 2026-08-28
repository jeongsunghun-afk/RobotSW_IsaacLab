# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion VAE 게이트 G2 — latent 보간 단조성 (Phase 1).

G0(latent 활용도)와 G1(held-out 세션 재구성)은 "z 가 정보를 나르는가"만 잰다.
**그 정보가 무엇인가**는 구분하지 못한다. 특히 ``--cond_dropout`` 으로 학습하면
z 가 "다음에 무엇을 할지"가 아니라 "지금 상태가 무엇인지"를 담을 수 있는데,
그러면 synthesis 정책이 조종할 대상이 되지 못한다.

G2 는 그것을 직접 잰다:

    1. 같은 카테고리에서 평균 전진속도가 서로 다른 클립 A, B 를 고른다.
    2. 각 클립의 전이들을 인코딩해 대표 latent z_A, z_B 를 얻는다.
    3. alpha ∈ [0, 1] 로 z 를 보간하고, **공통 시작 상태**에서 autoregressive 롤아웃한다.
    4. 생성된 시퀀스의 평균 vx 가 alpha 에 대해 단조인지 본다.

z 가 상태만 담고 있다면 alpha 를 바꿔도 롤아웃 속도가 체계적으로 변하지 않는다.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="motion_vae.pt 경로")
    p.add_argument("--category", default="walk", help="비교에 쓸 카테고리")
    p.add_argument("--num_pairs", type=int, default=12, help="평가할 (느린 클립, 빠른 클립) 쌍 수")
    p.add_argument("--num_alpha", type=int, default=9)
    p.add_argument("--rollout", type=int, default=60, help="autoregressive 롤아웃 스텝")
    p.add_argument("--min_speed_gap", type=float, default=0.4, help="쌍의 최소 평균속도 차 [m/s]")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--mode",
        choices=["one_step", "rollout"],
        default="one_step",
        help="one_step: 학습 분포 안. rollout: z 를 고정한 채 장기 롤아웃 — "
        "논문의 synthesis 정책은 매 스텝 새 z 를 내므로 분포 밖 사용이다.",
    )
    p.add_argument(
        "--category_b",
        default=None,
        help="★ 교차 카테고리 진단. 지정하면 (느린 --category 클립, 빠른 --category_b 클립) "
        "으로 짝을 만든다. 명령 추종은 gait 를 넘나들므로 이 축이 실제로 중요하다.",
    )
    p.add_argument(
        "--x0_source",
        choices=["A", "B"],
        default="A",
        help="★ 대칭 검사. 공통 시작 상태를 느린 클립(A) 것으로 쓸지 빠른 클립(B) 것으로 "
        "쓸지. 부호가 x0 에 따라 뒤집히면 그것은 latent 이 아니라 조건 상태의 효과다.",
    )
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, weights_only=False)
    targs = ck["args"]

    spec = importlib.util.spec_from_file_location(
        "t", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_go2_motion_vae.py")
    )
    assert spec is not None and spec.loader is not None
    tmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tmod)

    spec2 = importlib.util.spec_from_file_location("ml", targs["motion_lib"])
    assert spec2 is not None and spec2.loader is not None
    ml = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ml)

    # ★ 구 체크포인트에는 이 키들이 없다 — .get 으로 gauss 기본값을 준다.
    latent_dist = targs.get("latent_dist", "gauss")
    model = tmod.MotionVAE(
        targs["latent_dim"],
        targs["hidden"],
        targs["num_experts"],
        latent_dist=latent_dist,
        kappa_fixed=targs.get("kappa_fixed", 0.0),
        gauss_fixed_logvar=targs.get("gauss_fixed_logvar", None),
    ).to(dev)
    # 구 체크포인트에는 speed_head(축 정렬 보조 헤드)가 없다. 그 키만 누락되는 것은 허용하되
    # 다른 키가 어긋나면 조용히 넘기지 않는다.
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("speed_head.")] + list(unexpected)
    if bad:
        raise SystemExit(f"체크포인트 키가 어긋난다: {bad}")
    model.eval()
    mean, std = ck["mean"], ck["std"]
    vx_slice = tmod.FEATURE_SLICES["lin_vel"]

    # ── val 세션의 지정 카테고리 클립 수집 ────────────────────────
    split = json.load(open(targs["split_json"]))
    val_sessions = set(split["val_sessions"])

    def collect(cat: str) -> list[dict]:
        out = []
        for path in sorted(glob.glob(os.path.join(targs["pkl_dir"], f"{cat}__*.pkl"))):
            m = tmod.CLIP_NAME_RE.match(os.path.splitext(os.path.basename(path))[0])
            if m is None or m.group(2) not in val_sessions:
                continue
            x = tmod.build_x_vae(ml, path)[:: targs["stride"]]
            if len(x) < args.rollout // 2:
                continue
            out.append({"path": path, "x": x, "vx": float(x[:, vx_slice.start].mean()), "cat": cat})
        return out

    clips = collect(args.category)
    clips_b = collect(args.category_b) if args.category_b else None
    if len(clips) < 2:
        raise SystemExit(f"val 세션에 '{args.category}' 클립이 부족하다 ({len(clips)}개).")
    clips.sort(key=lambda c: c["vx"])
    print(
        f"[데이터] val 세션 '{args.category}' 클립 {len(clips)}개, "
        f"평균 vx {clips[0]['vx']:.2f} ~ {clips[-1]['vx']:.2f} m/s"
    )

    k_ahead = max(1, targs.get("predict_ahead", 1))
    resultant: list[float] = []

    def encode_clip(x: np.ndarray) -> torch.Tensor:
        xn = torch.from_numpy((x - mean) / std).to(dev)
        with torch.no_grad():
            mu, _ = model.encode(xn[:-k_ahead], xn[k_ahead:])
        m = mu.mean(0, keepdim=True)
        # ★ 진단: ||mean(mu)|| — 클립 안에서 스텝별 mu 가 서로 상쇄되는 정도.
        #   1.0 에 가까우면 클립 전체가 한 방향, 0 에 가까우면 대표 방향이 사실상 임의라
        #   그 위에서 잰 보간 곡선은 의미가 없다. G2b 를 읽기 전에 반드시 확인한다.
        resultant.append(float(m.norm()))
        # vmf: 단위벡터의 평균은 단위벡터가 아니다. 반드시 다시 정규화한다.
        return torch.nn.functional.normalize(m, dim=-1) if latent_dist == "vmf" else m

    def interp(za: torch.Tensor, zb: torch.Tensor, al: float) -> torch.Tensor:
        """latent 보간. vmf 는 구 위에 머물러야 한다 (slerp).

        선형보간을 그대로 쓰면 alpha=0.5 에서 ||z|| 가 cos(theta/2) 로 줄어들어,
        효과 크기가 '방향 변화'가 아니라 '크기 축소' 아티팩트와 섞인다.
        """
        if latent_dist != "vmf":
            return (1 - al) * za + al * zb
        dot = torch.clamp((za * zb).sum(-1, keepdim=True), -1.0, 1.0)
        th = torch.arccos(dot)
        if float(th.abs().max()) < 1e-6:
            return za
        return (torch.sin((1 - al) * th) * za + torch.sin(al * th) * zb) / torch.sin(th)

    @torch.no_grad()
    def rollout(x0: np.ndarray, z: torch.Tensor, steps: int) -> float:
        """공통 시작 상태에서 z 고정 롤아웃 → 생성 시퀀스의 평균 vx [m/s]."""
        xc = torch.from_numpy((x0 - mean) / std).to(dev).unsqueeze(0)
        vals = []
        for _ in range(steps):
            xc = model.decode(xc, z)
            vals.append(float(xc[0, vx_slice.start].item()))
        return float(np.mean(vals) * std[vx_slice.start] + mean[vx_slice.start])

    @torch.no_grad()
    def one_step(x0: np.ndarray, z: torch.Tensor) -> float:
        """한 스텝만 디코드 → 예측 vx [m/s]. 학습 분포 안에서의 z 민감도."""
        xc = torch.from_numpy((x0 - mean) / std).to(dev).unsqueeze(0)
        out = model.decode(xc, z)
        return float(out[0, vx_slice.start].item() * std[vx_slice.start] + mean[vx_slice.start])

    # 느린쪽 / 빠른쪽에서 짝을 만든다. category_b 가 있으면 A 의 느린쪽 ↔ B 의 빠른쪽.
    src_b = clips if clips_b is None else sorted(clips_b, key=lambda c: c["vx"])
    half = min(len(clips), len(src_b)) // 2 if clips_b is None else min(len(clips), len(src_b))
    pairs = []
    for i in range(min(args.num_pairs, max(half, 1))):
        if i >= len(clips) or i >= len(src_b):
            break
        a, b = clips[i], src_b[-(i + 1)]
        if b["vx"] - a["vx"] >= args.min_speed_gap:
            pairs.append((a, b))
    if not pairs:
        raise SystemExit(f"평균속도 차 >= {args.min_speed_gap} m/s 인 쌍이 없다.")

    # ── G2a: 선형 프로브 — z 가 속도 정보를 담고 있는가 ─────────────
    # 롤아웃 없이 인코딩만으로 재는 in-distribution 지표다.
    Z = torch.cat([encode_clip(c["x"]) for c in clips]).cpu().numpy()
    y = np.array([c["vx"] for c in clips])
    # ★ leave-one-out 교차검증. in-sample R^2 는 클립 수 <= latent 차원일 때 항상 1.0 이 되는
    #   미결정계라 무의미하다 (run/trot 은 클립 16개 vs latent 18차원).
    preds = np.empty(len(y))
    for i in range(len(y)):
        m_ = np.ones(len(y), bool)
        m_[i] = False
        Zt = np.concatenate([Z[m_] - Z[m_].mean(0), np.ones((m_.sum(), 1))], -1)
        cf, *_ = np.linalg.lstsq(Zt, y[m_] - y[m_].mean(), rcond=None)
        preds[i] = np.concatenate([Z[i] - Z[m_].mean(0), [1.0]]) @ cf + y[m_].mean()
    r2 = 1.0 - float(((y - preds) ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
    res = np.array(resultant[: len(clips)])
    print(
        f"\n[진단] 클립 대표방향의 resultant ||mean(mu)||  중앙값 {np.median(res):.3f}  "
        f"범위 {res.min():.3f}~{res.max():.3f}   (낮으면 보간 곡선의 의미가 약해진다)"
    )
    # ★ LOO 라도 표본 수 <= 특징 수 + 1 이면 lstsq 는 회귀가 아니라 최소노름 해를 낸다.
    #   R^2 가 음수로 크게 나오는 것이 그 신호다. 이때 **무효가 되는 것은 G2a 뿐이다** —
    #   G2b 는 회귀를 하지 않고 (보간 예측 vx 변화폭)/(실측 A-B 속도차) 만 재므로
    #   클립이 16개라도 정의가 성립한다. 즉 이 경우 종합 PASS 는 선언할 수 없지만
    #   G2b 의 FAIL 은 유효한 결과다. "진단이라 무효"라며 넘기지 말 것.
    under = len(clips) <= Z.shape[1] + 1
    print(
        f"\n[G2a] 선형 프로브 z -> 클립 평균 vx  (LOO 교차검증)  R^2 = {r2:.3f}   "
        f"클립 {len(clips)}, latent {Z.shape[1]}"
        f"{'   ★ 미결정계 (표본 <= 차원+1) — 게이트 아님, 진단으로만 읽을 것' if under else ''}"
    )
    print(f"      기준 >= 0.5  ->  {'PASS' if r2 >= 0.5 else 'FAIL'}{'  [무효]' if under else ''}")

    alphas = np.linspace(0.0, 1.0, args.num_alpha)
    rows, rhos = [], []
    print(f"\n{'쌍':>3s}  {'vx_A':>6s} {'vx_B':>6s} | " + " ".join(f"a={a:.2f}" for a in alphas) + "   rho")
    print("-" * (30 + 7 * args.num_alpha))
    for k, (ca, cb) in enumerate(pairs):
        za, zb = encode_clip(ca["x"]), encode_clip(cb["x"])
        # 공통 시작 상태 — 속도 변화가 z 에서만 오도록. --x0_source 로 A/B 를 바꿔 대칭 검사한다.
        x0 = (ca if args.x0_source == "A" else cb)["x"][0]
        if args.mode == "rollout":
            speeds = [rollout(x0, interp(za, zb, al), args.rollout) for al in alphas]
        else:
            speeds = [one_step(x0, interp(za, zb, al)) for al in alphas]
        rho = spearman(alphas, np.array(speeds))
        rhos.append(rho)
        rows.append({"vx_A": ca["vx"], "vx_B": cb["vx"], "alphas": alphas.tolist(), "speeds": speeds, "spearman": rho})
        print(f"{k:3d}  {ca['vx']:6.2f} {cb['vx']:6.2f} | " + " ".join(f"{s:6.2f}" for s in speeds) + f"  {rho:+.2f}")

    rho_arr = np.array(rhos)
    mean_rho = float(rho_arr.mean())
    frac_mono = float((rho_arr >= 0.7).mean())
    # ★ 효과 크기 없이 순위상관만 보면 안 된다. 값 차이가 0.001 m/s 수준이면
    #   Spearman 은 ±1.0 이 나오지만 z 는 사실상 아무 것도 하지 않는 것이다.
    eff = np.array([(max(r["speeds"]) - min(r["speeds"])) / max(r["vx_B"] - r["vx_A"], 1e-9) for r in rows])
    med_eff = float(np.median(eff))
    print(f"\n[G2b] latent 보간 단조성 (mode={args.mode}) — 쌍 {len(pairs)}개")
    print(f"     Spearman 평균 {mean_rho:+.3f},  rho >= 0.7 인 쌍 {frac_mono:.0%}")
    print(f"     효과 크기 (예측 vx 변화폭 / A-B 속도 간격) 중앙값 {med_eff:.1%}   기준 >= 10%")
    ok = mean_rho >= 0.7 and frac_mono >= 0.7 and med_eff >= 0.10
    print(
        f"     판정 -> {'PASS' if ok else 'FAIL'}"
        f"{'  (순위는 맞지만 효과가 없다 = decoder 가 z 를 무시)' if not ok and mean_rho >= 0.7 else ''}"
        f"{'  [G2a 미결정계 → 종합 PASS 선언 불가. 단 위 G2b 수치 자체는 유효하다]' if under else ''}"
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump(
            {
                "ckpt": args.ckpt,
                "category": args.category,
                "rollout": args.rollout,
                "latent_dist": latent_dist,
                "kappa_fixed": targs.get("kappa_fixed", 0.0),
                "gauss_fixed_logvar": targs.get("gauss_fixed_logvar", None),
                "predict_ahead": k_ahead,
                "resultant_median": float(np.median(res)),
                "resultant_min": float(res.min()),
                "underdetermined": bool(under),
                "num_clips": len(clips),
                "mode": args.mode,
                "g2a_probe_r2": r2,
                "g2a_pass": bool(r2 >= 0.5),
                "mean_spearman": mean_rho,
                "frac_monotonic": frac_mono,
                "median_effect_size": med_eff,
                "pass": bool(mean_rho >= 0.7 and frac_mono >= 0.7 and med_eff >= 0.10 and not under),
                "pairs": rows,
            },
            open(args.out_json, "w"),
            indent=2,
        )
        print(f"[저장] {args.out_json}")


if __name__ == "__main__":
    main()
