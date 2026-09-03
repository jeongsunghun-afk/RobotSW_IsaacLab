# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent style reward 의 참조 통계 산출 (Phase 2).

Phase 1 의 동결 인코더로 **train 세션 expert** 의 latent 분포를 적합해
``latent_ref_stats.pt`` 를 만든다. env 는 이 파일만 읽으므로, val 세션 누수 여부가
코드 독해가 아니라 **파일 내용**으로 확인된다 (게이트 V5).

보상은 스텝 로그밀도가 아니라 **env 별 시간창 marginal KL** 이다::

    env e 의 최근 N 전이쌍에서   mu_e, var_e   (대각)
    D_e = KL( N(mu_e, diag var_e) || N(mu_ref, diag var_ref) )
    r_style_e = exp( -c_kl * (D_e - offset) )

기호:

===========  =============================================  ==========
기호         의미                                           단위
===========  =============================================  ==========
``x_vae``    프레임 상태 특징 49-D                          혼합
``z``        인코더 사후분포 평균 18-D                       무차원
``N``        env 별 시간창 길이 (전이 쌍 수)                 개
``mu_ref``   train expert latent 평균                        무차원
``var_ref``  train expert latent 대각 분산                   무차원
``D_e``      창 marginal KL                                  nat
``c_kl``     보상 민감도                                     1/nat
``offset``   ``D_e`` 오프셋 (expert 중앙값)                   nat
===========  =============================================  ==========

``D_e`` 는 ``d_step`` 과 **완전히 다른 스칼라**다. Phase 2 검증 보고서 §5 의 ``c_kl``
스윕은 ``d_step`` 기준이라 그대로 쓸 수 없다. 그래서 여기서 길이 정확히 ``N`` 인 expert
창을 대량으로 뽑아 ``D_e`` 의 분위수를 함께 저장하고, ``c_kl`` 을 데이터에서 정한다.

실행::

    CUDA_VISIBLE_DEVICES=1 /home/user/miniconda3/envs/isaac-6.0/bin/python \\
      scripts/imitation_learning/build_go2_latent_ref_stats.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from rsl_rl.modules.motion_encoder import LatentStyleScorer, MotionEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _load_vae_trainer():
    """``train_go2_motion_vae.py`` 를 모듈로 읽는다 (수정하지 않고 재사용만 한다)."""
    path = os.path.join(_HERE, "train_go2_motion_vae.py")
    spec = importlib.util.spec_from_file_location("_go2_vae_trainer", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_go2_vae_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt",
        default=os.path.join(
            _REPO,
            "reports/go2_imitation/go2_imitation_tracking/2026-08-27_phase1_motion_vae",
            "gaussfix_reg/runs/base_s0/motion_vae.pt",
        ),
        help="Phase 1 motion VAE 체크포인트 (인코더만 쓴다)",
    )
    p.add_argument(
        "--pkl_dir",
        default=os.path.join(
            _REPO, "source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation_latent/imitation/motion_pkl"
        ),
    )
    p.add_argument(
        "--motion_lib",
        default=os.path.join(_REPO, "source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation_latent/motion_lib.py"),
    )
    p.add_argument(
        "--split_json",
        default=os.path.join(
            _REPO,
            "reports/go2_imitation/go2_imitation_tracking/2026-08-27_phase0_dataset_build/metrics/session_split.json",
        ),
    )
    p.add_argument("--out", default=None, help="기본값은 체크포인트 옆의 latent_ref_stats.pt")
    p.add_argument("--window_n", type=int, default=8, help="env 별 시간창 길이 (전이 쌍 수)")
    p.add_argument("--cov_reg", type=float, default=1e-4, help="참조 전체 공분산의 대각 정칙화 (Cholesky 안정성).")
    p.add_argument("--full_shrink", type=float, default=0.20, help="창 공분산 수축계수. N<D 라 표본 공분산이 특이하다.")
    p.add_argument("--var_floor", type=float, default=1e-4)
    p.add_argument("--max_windows", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    ckpt_meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vae_args = ckpt_meta.get("args", {})
    stride = int(vae_args.get("stride", 1))
    k_ahead = max(1, int(vae_args.get("predict_ahead", 1)))
    print(
        f"[체크포인트] {args.ckpt}\n  stride {stride} · predict_ahead {k_ahead} · latent {vae_args.get('latent_dim')}"
    )

    enc = MotionEncoder.from_checkpoint(args.ckpt, device=dev, freeze=True)

    # ── 데이터: train 세션만 ────────────────────────────────────
    trainer = _load_vae_trainer()
    ds_args = argparse.Namespace(
        pkl_dir=args.pkl_dir, motion_lib=args.motion_lib, split_json=args.split_json, stride=stride
    )
    train, val = trainer.load_dataset(ds_args)
    with open(args.split_json) as fh:
        split = json.load(fh)
    print(f"[데이터] train 클립 {len(train['clips'])} / val 클립 {len(val['clips'])} (val 은 쓰지 않는다)")

    mean = np.asarray(ckpt_meta["mean"], dtype=np.float32)
    std = np.asarray(ckpt_meta["std"], dtype=np.float32)

    def encode(prev: np.ndarray, curr: np.ndarray, chunk: int = 65536) -> torch.Tensor:
        out = []
        for i in range(0, len(prev), chunk):
            xp = torch.from_numpy((prev[i : i + chunk] - mean) / std).to(dev)
            xc = torch.from_numpy((curr[i : i + chunk] - mean) / std).to(dev)
            out.append(enc.inference(xp, xc, normalized=True))
        return torch.cat(out)

    # ── 1) 전역 참조 통계 (전 train 전이쌍) ─────────────────────
    tr_prev, tr_curr = trainer.make_pairs(train["clips"], k_ahead)
    z_ref = encode(tr_prev, tr_curr)
    scorer = LatentStyleScorer(var_floor=args.var_floor)
    scorer.fit_reference(z_ref)
    mu_ref, var_ref = scorer.mu_ref, scorer.var_ref
    print(f"[참조] 전이쌍 {len(z_ref):,} · mu_ref |·| {mu_ref.abs().mean():.4f} · var_ref 평균 {var_ref.mean():.4f}")
    n_floored = int((z_ref.var(dim=0, unbiased=False) < args.var_floor).sum())
    print(f"[참조] var_floor({args.var_floor}) 에 걸린 차원 {n_floored}/{enc.latent_dim}")

    # ── 2) 길이 N 창의 D_e 분포 ─────────────────────────────────
    # env 시간창의 대리물 — 한 클립 안의 연속 N 쌍. 클립 경계를 넘지 않는다.
    starts: list[int] = []
    offset = 0
    for c in train["clips"]:
        n_pairs = len(c["x"]) - k_ahead
        if n_pairs >= args.window_n:
            for s in range(0, n_pairs - args.window_n + 1):
                starts.append(offset + s)
        offset += max(n_pairs, 0)
    idx = np.array(starts, dtype=np.int64)
    if len(idx) > args.max_windows:
        idx = rng.choice(idx, size=args.max_windows, replace=False)
    win = torch.from_numpy(idx).to(dev).unsqueeze(1) + torch.arange(args.window_n, device=dev).unsqueeze(0)
    d_e = []
    floor_hits = 0
    for i in range(0, len(win), 8192):
        zw = z_ref[win[i : i + 8192]]  # [B, N, D]
        mu_w = zw.mean(dim=1)
        var_raw = zw.var(dim=1, unbiased=False)
        floor_hits += int((var_raw < args.var_floor).sum())
        var_w = var_raw.clamp(min=args.var_floor)
        d_e.append(0.5 * (torch.log(var_ref / var_w) + (var_w + (mu_w - mu_ref).pow(2)) / var_ref - 1.0).sum(dim=-1))
    d_e_t = torch.cat(d_e)
    q = {f"p{k}": float(torch.quantile(d_e_t, k / 100.0)) for k in (1, 5, 25, 50, 75, 95, 99)}
    print(f"[D_e] expert 창 {len(d_e_t):,}개 (N={args.window_n})")
    print("  " + " · ".join(f"{k} {v:.3f}" for k, v in q.items()))
    print(f"  var_floor 클램프 비율 {floor_hits / (len(d_e_t) * enc.latent_dim):.4%}")

    # ── 3) c_kl 을 데이터에서 정한다 ────────────────────────────
    # offset = expert 중앙값 → expert 중앙 창의 r_style 이 정확히 1.0
    # c_kl 은 "expert 창의 **상사분위**(p75) 가 r_style 0.9 를 받는" 민감도로 잡는다.
    #
    # ★ p95 를 기준으로 잡으면 안 된다. N=8 창의 D_e 분포는 꼬리가 매우 무겁고(p50 10.9 대
    #   p95 49.6) 그 꼬리는 스타일 차이가 아니라 **짧은 창의 분산 추정 잡음**이다. 꼬리로
    #   보정하면 c_kl 이 지나치게 작아져 정책이 실제로 머무는 구간(D_e 15~25)에서 보상
    #   격차가 1% 수준으로 눌린다 — 실측으로 확인했다.
    # (Phase 2 §5 의 c_kl=0.01 은 d_step 기준값이라 이 스칼라에 쓸 수 없다)
    iqr_hi = max(q["p75"] - q["p50"], 1e-6)
    c_kl = float(-np.log(0.9) / iqr_hi)
    print(f"[보정] offset {q['p50']:.3f} nat · p75-p50 {iqr_hi:.3f} → c_kl {c_kl:.5f}")
    for name, d in (("p50", q["p50"]), ("p75", q["p75"]), ("p95", q["p95"])):
        print(f"    expert {name} (D_e {d:7.3f}) → r_style {float(np.exp(-c_kl * (d - q['p50']))):.4f}")

    # ── 2b) 전체 공분산 판 (`style_statistic="full"`) ──────────
    # 대각판은 차원 간 상관을 버려서 "떠는 정지"를 expert 보다 좋게 친다
    # (2026-09-01 배터리: 정책_정지(노이즈) AUROC 0.139). 전체 공분산은 같은 창 형태로
    # 그 상관까지 본다 (같은 배터리 0.845, 11개 negative 최솟값 0.579 로 최고).
    cov_ref = torch.from_numpy(np.cov(z_ref.cpu().numpy().T)).float().to(dev)
    cov_ref = cov_ref + args.cov_reg * torch.eye(enc.latent_dim, device=dev)
    L = torch.linalg.cholesky(cov_ref)
    inv_ref = torch.cholesky_inverse(L)
    logdet_ref = float(2.0 * torch.log(torch.diagonal(L)).sum())
    diag_ref = torch.diag(var_ref)
    d_f = []
    for i in range(0, len(win), 8192):
        zw = z_ref[win[i : i + 8192]]
        mu_w = zw.mean(dim=1)
        wc = zw - mu_w.unsqueeze(1)
        cov_w = wc.transpose(1, 2) @ wc / zw.shape[1]
        cov_w = (1.0 - args.full_shrink) * cov_w + args.full_shrink * diag_ref
        dmu = mu_w - mu_ref
        _sg, ldw = torch.linalg.slogdet(cov_w)
        tr_term = torch.einsum("ij,bji->b", inv_ref, cov_w)
        maha = torch.einsum("bi,ij,bj->b", dmu, inv_ref, dmu)
        d_f.append(0.5 * (tr_term + maha - enc.latent_dim + logdet_ref - ldw))
    d_f_t = torch.cat(d_f)
    qf = {f"p{k}": float(torch.quantile(d_f_t, k / 100.0)) for k in (1, 5, 25, 50, 75, 95, 99)}
    iqr_f = max(qf["p75"] - qf["p50"], 1e-6)
    c_kl_full = float(-np.log(0.9) / iqr_f)
    print("[D_e full] expert 창 분위수 " + " · ".join(f"{k} {v:.3f}" for k, v in qf.items()))
    print(
        f"[보정 full] shrink {args.full_shrink} · offset {qf['p50']:.3f} nat · "
        f"p75-p50 {iqr_f:.3f} → c_kl {c_kl_full:.5f}"
    )

    # ── 2c) power 매핑 지수 ────────────────────────────────────
    # exp 매핑 `exp(-c(D-off))` 는 D 가 offset 에서 수십 배 멀어지면 보상과 gradient 가
    # 수치적으로 0 이 된다 (2026-09-01 실측: D~1225 에서 r=2e-25, |dr/dD|=1e-26,
    # env 간 std 0.10 으로 V3 게이트 0.391 미달 → 신호 소멸). KL 은 로그 스케일 양이므로
    # 같은 보정 규칙(expert p75 -> 0.9)을 **로그축**에서 적용한다:
    #     r = (offset / D_e) ** a,   a = -ln(0.9) / (ln p75 - ln p50)
    # 보정점에서 exp 판과 정확히 같은 값을 주고, 꼬리에서만 완만해진다.
    # ★ D_e 의 **단조 변환**이므로 AUROC 기반 판정(§8 통계량 선택)은 그대로 유효하다.
    a_kl = float(-np.log(0.9) / max(np.log(q["p75"]) - np.log(q["p50"]), 1e-9))
    a_kl_full = float(-np.log(0.9) / max(np.log(qf["p75"]) - np.log(qf["p50"]), 1e-9))
    print(f"[보정 power] a(diag) {a_kl:.4f} · a(full) {a_kl_full:.4f} (무차원)")
    for _n, _d in (("p50", qf["p50"]), ("p75", qf["p75"]), ("p95", qf["p95"])):
        print(f"    full expert {_n} (D_e {_d:7.3f}) -> r_style {(qf['p50'] / _d) ** a_kl_full:.4f}")

    out = args.out or os.path.join(os.path.dirname(args.ckpt), "latent_ref_stats.pt")
    torch.save(
        {
            "mu_ref": mu_ref.cpu(),
            "var_ref": var_ref.cpu(),
            "latent_dim": int(enc.latent_dim),
            "x_dim": int(enc.x_dim),
            "window_n": int(args.window_n),
            "var_floor": float(args.var_floor),
            "d_e_quantiles": q,
            "c_kl": c_kl,
            "offset": q["p50"],
            # 전체 공분산 판 — env 가 `style_statistic="full"` 일 때 쓴다
            "cov_ref": cov_ref.cpu(),
            "cov_reg": float(args.cov_reg),
            "full_shrink": float(args.full_shrink),
            "d_e_full_quantiles": qf,
            "c_kl_full": c_kl_full,
            "a_kl": a_kl,
            "a_kl_full": a_kl_full,
            "offset_full": qf["p50"],
            "n_pairs": int(len(z_ref)),
            "n_windows": int(len(d_e_t)),
            "stride": stride,
            "k_ahead": k_ahead,
            "pair_dt": stride * k_ahead / 60.0,
            "ckpt_path": os.path.relpath(args.ckpt, _REPO),
            "ckpt_sha256": _sha256(args.ckpt),
            "split_json": os.path.relpath(args.split_json, _REPO),
            "train_sessions": sorted(split["train_sessions"]),
            "val_sessions_excluded": sorted(split["val_sessions"]),
            "n_train_clips": len(train["clips"]),
        },
        out,
    )
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
