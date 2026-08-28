# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""안 B (latent style reward) 오프라인 판별력 검증 — Phase 2 게이트.

시뮬레이터 없이, 학습된 motion VAE **인코더만** 써서 다음을 잰다::

    mu_ref, Sigma_ref  = expert 배치의 z 통계 (대각 근사)
    d_step[t]          = -log N( z[t] ; mu_ref, Sigma_ref )
    r_style            = exp( -c_kl * d_step )

질문은 하나다 — 이 점수가 **expert 와 expert 아닌 것을 구분하는가**.
구분하지 못하면 (AUROC ~ 0.5) 안 B 도 실패다.

★ 참조 통계는 **train 세션**의 latent 로 적합하고, positive 는 **held-out val 세션**만
쓴다. train 세션으로 재면 인코더가 본 데이터라 과대평가된다.

★★ 모든 negative 에 대해 **raw x_vae 대각 가우시안 baseline** 을 나란히 잰다.
latent 가 이 baseline 을 못 이기는 negative 는 "인코더가 스타일을 안다" 는 증거가 아니라
"정규화 한 줄로도 되는 판별" 이다. 판정은 baseline 대비 우위로 한다.

negative 구성:

=====================  ==================================================================
이름                   내용
=====================  ==================================================================
``cat_*``              데이터셋에서 제외된 카테고리 (spin, jump, stand, lie, sit)
``shuffle``            프레임 순서 무작위 셔플 — 채널 주변분포는 **완전 보존**, 전이만 파괴
``reverse``            시간 반전
``speed2x``            2배속 리샘플 (프레임 서브샘플 + 속도 블록 x2)
``speed0.5x``          0.5배속 리샘플 (프레임 조밀화 + 속도 블록 x0.5)
``static``             한 프레임 반복 (속도 채널은 그 프레임 값을 유지)
``jnoise_s{sigma}``    관절각에 가우시안 잡음 [rad] — 민감도 스윕
``legswap``            좌우 다리 교환 (base 채널은 그대로)
``frontswap``          **앞다리만** 좌우 교환 — trot 을 pace 로 바꾼다. ★ 진짜 어려운 negative
``rearswap``           뒷다리만 좌우 교환
``desync{k}``          다리 블록만 k 프레임 시간이동 — base-다리 위상 결합을 깬다
=====================  ==================================================================

★ ``legswap`` 은 **negative 로서 결함이 있다**. 좌우대칭 직진보행에서 좌우 다리를 통째로
바꾸면 같은 gait 의 반주기 상태가 되어 물리적으로 유효한 모션이다. 낮은 AUROC 를
"판별 실패" 로 읽으면 안 된다. 대신 ``frontswap`` (trot -> pace) 이 gait **종류**를 바꾸는
진짜 어려운 negative 다.

``shuffle``/``reverse``/``static`` 은 x_vae 를 만든 **뒤** 프레임 순서만 바꾼다. 프레임을
재배열한 뒤 속도를 다시 계산하면 속도 채널이 폭발해 판별이 트리비얼해지므로 그렇게 하지
않는다. 이 구성에서는 49개 채널의 주변분포가 정확히 보존되고 **전이 구조만** 깨진다.

``legswap`` 의 관절/발 인덱스 치환과 부호는 데이터셋의 미러 클립(``M_`` 접두사)과 원본을
대조해 **실측으로 역산**했다 (maxerr 0.0). 추측하지 않았다.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import pickle
import re
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rsl_rl"))

from rsl_rl.modules.motion_encoder import LatentStyleScorer, MotionEncoder  # noqa: E402

CLIP_NAME_RE = re.compile(r"^[a-z_]+__(M_)?(D\d+_[A-Za-z0-9]+_[A-Za-z0-9]+_\d+)_F\d+_F\d+_\d+$")

FEATURE_SLICES = {
    "base_height": slice(0, 1),
    "rot6d": slice(1, 7),
    "lin_vel": slice(7, 10),
    "ang_vel": slice(10, 13),
    "foot_pos": slice(13, 25),
    "joint_pos": slice(25, 37),
    "joint_vel": slice(37, 49),
}
#: 속도 성분 — 리샘플 시 배율을 곱해야 하는 채널.
VEL_IDX = list(range(7, 13)) + list(range(37, 49))
JOINT_POS_IDX = np.arange(25, 37)

#: 좌우 다리 교환 — 미러 클립 대조로 역산한 치환/부호 (다리당 3관절). 다리 순서는
#: 발 위치 평균으로 실측 확인했다: 0 = FL(x+, y+), 1 = FR, 2 = RL, 3 = RR.
#: hip 만 부호 반전, thigh/calf 는 그대로. 발 위치는 y 성분만 반전.
_LEG_SWAPS = {"legswap": [1, 0, 3, 2], "frontswap": [1, 0, 2, 3], "rearswap": [0, 1, 3, 2]}
_JOINT_SIGN_ONE = np.array([-1.0, 1.0, 1.0])
_FOOT_SIGN_ONE = np.array([1.0, -1.0, 1.0])


def _leg_maps(order: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """다리 순열 → (채널 치환, 관절 부호, 발 부호). 교환된 다리만 부호가 뒤집힌다."""
    perm = np.concatenate([np.arange(3 * leg, 3 * leg + 3) for leg in order])
    swapped = np.array([o != i for i, o in enumerate(order)])
    jsign = np.concatenate([_JOINT_SIGN_ONE if sw else np.ones(3) for sw in swapped])
    fsign = np.concatenate([_FOOT_SIGN_ONE if sw else np.ones(3) for sw in swapped])
    return perm, jsign, fsign


# ──────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────
def load_modules(motion_lib_path: str, trainer_path: str):
    """``motion_lib`` 과 VAE 학습 스크립트를 경로로 로드한다 (``build_x_vae`` 재사용)."""
    mods = []
    for name, path in (("ml", motion_lib_path), ("tr", trainer_path)):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods[0], mods[1]


def quat_to_exp_map(qxyzw: np.ndarray) -> np.ndarray:
    """단위 쿼터니언 (N, 4) [x, y, z, w] → exponential map (N, 3). ``w >= 0`` 최단호."""
    q = np.asarray(qxyzw, dtype=np.float64).copy()
    q[q[:, 3] < 0] *= -1.0
    length = np.linalg.norm(q[:, :3], axis=-1)
    angle = 2.0 * np.arctan2(length, q[:, 3])
    axis = np.where(
        (length > 1e-5)[:, None],
        q[:, :3] / np.maximum(length, 1e-12)[:, None],
        np.array([0.0, 0.0, 1.0])[None, :],
    )
    return axis * angle[:, None]


def x_vae_from_txt(ml, tr, txt_path: str, tmp_dir: str) -> np.ndarray | None:
    """61열 리타게팅 txt → ``x_vae`` (N, 49).

    expert 데이터셋과 **완전히 같은 경로**를 타도록, 빌더와 동일한 방식으로 pkl 로 바꾼 뒤
    ``build_x_vae`` 를 태운다. 파싱 실패나 비유한값이면 None.
    """
    try:
        with open(txt_path) as fh:
            raw = json.load(fh)
        frames = np.array(raw["Frames"], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None
    if frames.ndim != 2 or frames.shape[1] < 19 or len(frames) < 8:
        return None
    out = np.concatenate([frames[:, 0:3], quat_to_exp_map(frames[:, 3:7]), frames[:, 7:19]], axis=-1)
    fps = round(1.0 / float(raw["FrameDuration"]))
    tmp = os.path.join(tmp_dir, "tmp.pkl")
    with open(tmp, "wb") as fh:
        pickle.dump({"loop_mode": 0, "fps": fps, "frames": out.tolist()}, fh)
    x = tr.build_x_vae(ml, tmp)
    return x if np.isfinite(x).all() else None


def load_expert_clips(ml, tr, pkl_dir: str, val_sessions: set[str]) -> tuple[list[dict], list[dict]]:
    """expert pkl → train / val(held-out 세션) 클립 목록. ``x`` 는 **stride 1 (60fps)** 원본."""
    train, val = [], []
    unparsed = []
    for p in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl"))):
        m = CLIP_NAME_RE.match(os.path.splitext(os.path.basename(p))[0])
        if m is None:
            unparsed.append(p)
            continue
        x = tr.build_x_vae(ml, p)
        if len(x) < 8 or not np.isfinite(x).all():
            unparsed.append(p)
            continue
        rec = {"name": os.path.basename(p), "session": m.group(2), "cat": os.path.basename(p).split("__")[0], "x": x}
        (val if m.group(2) in val_sessions else train).append(rec)
    if unparsed:
        raise SystemExit(f"클립 {len(unparsed)}개를 처리하지 못했다 (조용히 넘기지 않는다): {unparsed[:3]}")
    return train, val


# ──────────────────────────────────────────────────────────────
# negative 생성
# ──────────────────────────────────────────────────────────────
def make_negative(x60: np.ndarray, kind: str, stride: int, rng: np.random.Generator) -> np.ndarray | None:
    """stride 1 원본 ``x_vae`` → negative ``x_vae`` (학습 stride 기준)."""
    x = x60[::stride].copy()
    if kind == "expert":
        return x
    if kind == "shuffle":
        return x[rng.permutation(len(x))]
    if kind == "reverse":
        return x[::-1].copy()
    if kind == "static":
        return np.repeat(x[len(x) // 2 : len(x) // 2 + 1], len(x), axis=0)
    if kind == "speed2x":
        y = x60[:: stride * 2].copy()
        y[:, VEL_IDX] *= 2.0
        return y if len(y) >= 4 else None
    if kind == "speed0.5x":
        y = x60.copy()
        y[:, VEL_IDX] *= 0.5
        return y
    if kind in _LEG_SWAPS:
        perm, jsign, fsign = _leg_maps(_LEG_SWAPS[kind])
        y = x.copy()
        fp = y[:, FEATURE_SLICES["foot_pos"]]
        y[:, FEATURE_SLICES["foot_pos"]] = fp[:, perm] * fsign
        for sl in (FEATURE_SLICES["joint_pos"], FEATURE_SLICES["joint_vel"]):
            blk = y[:, sl]
            y[:, sl] = blk[:, perm] * jsign
        return y
    if kind.startswith("desync"):
        k = int(kind[len("desync") :])
        if len(x) <= k + 2:
            return None
        y = x[: len(x) - k].copy()
        leg = np.r_[13:25, 25:37, 37:49]
        y[:, leg] = x[k:, leg]
        return y
    if kind.startswith("jnoise_s"):
        sigma = float(kind[len("jnoise_s") :])
        y = x.copy()
        y[:, JOINT_POS_IDX] += rng.normal(0.0, sigma, size=(len(y), 12)).astype(np.float32)
        return y
    raise ValueError(f"알 수 없는 negative 종류: {kind}")


# ──────────────────────────────────────────────────────────────
# 점수
# ──────────────────────────────────────────────────────────────
def pair_stack(clips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """클립 목록 → (x_prev, x_curr, clip_id) 상태전이 쌍."""
    prev, curr, cid = [], [], []
    for i, x in enumerate(clips):
        if len(x) < 2:
            continue
        prev.append(x[:-1])
        curr.append(x[1:])
        cid.append(np.full(len(x) - 1, i, dtype=np.int64))
    return np.concatenate(prev), np.concatenate(curr), np.concatenate(cid)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUROC. **점수가 클수록 negative** 라는 방향으로 정의한다."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    # 동점 처리 — 평균 순위
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    n_neg, n_pos = len(neg), len(pos)
    r_neg = ranks[:n_neg].sum()
    return float((r_neg - n_neg * (n_neg + 1) / 2.0) / (n_neg * n_pos))


def quantiles(v: np.ndarray) -> dict:
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p5": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def clip_means(v: np.ndarray, cid: np.ndarray) -> np.ndarray:
    """클립 단위 평균 점수."""
    n = int(cid.max()) + 1 if len(cid) else 0
    return np.array([v[cid == i].mean() for i in range(n) if (cid == i).any()])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = "source/isaaclab_tasks/isaaclab_tasks/direct"
    rep = "reports/go2_imitation/go2_imitation_tracking"
    p.add_argument("--ckpt", default=f"{rep}/2026-08-27_phase1_motion_vae/final_s2_strat/motion_vae.pt")
    p.add_argument("--pkl_dir", default=f"{root}/go2_imitation_latent/imitation/motion_pkl")
    p.add_argument("--motion_lib", default=f"{root}/go2_imitation_latent/motion_lib.py")
    p.add_argument("--trainer", default="scripts/imitation_learning/train_go2_motion_vae.py")
    p.add_argument("--split_json", default=f"{rep}/2026-08-27_phase0_dataset_build/metrics/session_split.json")
    p.add_argument(
        "--src_dir", default="/home/lgb/Dog_Motion_data_3D/mujoco_retarget_go2/smr_txt_retarget_dataset_go2_v0"
    )
    p.add_argument("--neg_categories", nargs="*", default=["spin", "jump", "stand", "lie", "sit"])
    p.add_argument("--neg_cat_max_clips", type=int, default=60, help="제외 카테고리당 최대 클립 수")
    p.add_argument(
        "--noise_sigmas", nargs="*", type=float, default=[0.01, 0.02, 0.05, 0.1, 0.2], help="관절각 잡음 σ [rad] 스윕"
    )
    p.add_argument("--c_kl_sweep", nargs="*", type=float, default=[0.001, 0.01, 0.1])
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument(
        "--mkl_batch",
        type=int,
        default=64,
        help="클립 단위 배치 marginal KL 을 잴 때 클립당 표본 쌍 수 (길이 효과 제거)",
    )
    p.add_argument("--knn_ref_max", type=int, default=20000, help="k-NN 참조 표본 상한")
    p.add_argument("--out_dir", default=f"{rep}/2026-08-27_phase2_style_reward_validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    ml, tr = load_modules(args.motion_lib, args.trainer)
    ckpt_args = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("args", {})
    stride = int(ckpt_args.get("stride", 1))
    print(f"[체크포인트] {args.ckpt}")
    print(
        f"             stride {stride} ({60 // stride} fps), latent {ckpt_args.get('latent_dim')}, "
        f"cond_dropout {ckpt_args.get('cond_dropout')}"
    )

    encoder = MotionEncoder.from_checkpoint(args.ckpt, dev)
    print(f"[인코더] x_dim {encoder.x_dim}, latent {encoder.latent_dim}, frozen")

    # ── expert ────────────────────────────────────────────────
    with open(args.split_json) as fh:
        val_sessions = set(json.load(fh)["val_sessions"])
    train_clips, val_clips = load_expert_clips(ml, tr, args.pkl_dir, val_sessions)
    print(
        f"[expert] train 클립 {len(train_clips)} (세션 {len({c['session'] for c in train_clips})}), "
        f"held-out val 클립 {len(val_clips)} (세션 {len({c['session'] for c in val_clips})})"
    )

    def encode_np(xp: np.ndarray, xc: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(xp), 8192):
            a = torch.from_numpy(np.ascontiguousarray(xp[i : i + 8192])).to(dev)
            b = torch.from_numpy(np.ascontiguousarray(xc[i : i + 8192])).to(dev)
            out.append(encoder.inference(a, b).cpu().numpy())
        return np.concatenate(out)

    # 참조 통계는 **train 세션** latent 로 적합한다.
    ref_prev, ref_curr, _ = pair_stack([c["x"][::stride] for c in train_clips])
    z_ref = encode_np(ref_prev, ref_curr)
    scorer = LatentStyleScorer(c_kl=0.01)
    sub = rng.choice(len(z_ref), size=min(args.knn_ref_max, len(z_ref)), replace=False)
    scorer.fit_reference(torch.from_numpy(z_ref[sub]).to(dev), keep_samples=True)
    print(f"[참조] train 상태전이 쌍 {len(z_ref):,} → mu_ref/Sigma_ref 적합 (k-NN 표본 {len(sub):,})")

    # raw x_vae baseline — latent 와 **같은 입력**(전이 쌍 98-D)을 보는 대각 가우시안.
    raw_pair = np.concatenate([ref_prev, ref_curr], axis=-1)
    raw_mean = raw_pair.mean(0)
    raw_var = raw_pair.var(0)
    raw_var[raw_var < 1e-8] = 1e-8
    raw_const = 0.5 * float(np.log(2.0 * math.pi * raw_var).sum())

    def raw_score(xp: np.ndarray, xc: np.ndarray) -> np.ndarray:
        d = np.concatenate([xp, xc], axis=-1) - raw_mean
        return 0.5 * (d * d / raw_var).sum(-1) + raw_const

    # ── negative 세트 구성 ─────────────────────────────────────
    perturb_kinds = [
        "shuffle",
        "reverse",
        "speed2x",
        "speed0.5x",
        "static",
        "legswap",
        "frontswap",
        "rearswap",
        "desync2",
        "desync5",
    ]
    perturb_kinds += [f"jnoise_s{s}" for s in args.noise_sigmas]

    sets: dict[str, list[np.ndarray]] = {}
    sets["expert"] = [c["x"][::stride] for c in val_clips]
    # 통제군 — 참조를 적합한 바로 그 train 세션. held-out 세션의 점수가 이것보다 얼마나
    # 나쁜지가 "세션 일반화 격차" 이고, negative 판정은 그 격차와 분리해서 읽어야 한다.
    tr_sub = rng.choice(len(train_clips), size=min(200, len(train_clips)), replace=False)
    sets["expert_train"] = [train_clips[i]["x"][::stride] for i in tr_sub]
    for kind in perturb_kinds:
        out = []
        for c in val_clips:
            y = make_negative(c["x"], kind, stride, rng)
            if y is not None and len(y) >= 2:
                out.append(y)
        sets[kind] = out

    # 제외 카테고리 — 원 txt 에서 직접 만든다 (데이터셋에 pkl 이 없다).
    with tempfile.TemporaryDirectory() as tmp_dir:
        for cat in args.neg_categories:
            files = sorted(glob.glob(os.path.join(args.src_dir, cat, "*.txt")))
            if not files:
                print(f"  [경고] 카테고리 '{cat}' 에 파일이 없다 — 건너뜀")
                continue
            pick = (
                files
                if len(files) <= args.neg_cat_max_clips
                else [files[i] for i in rng.choice(len(files), args.neg_cat_max_clips, replace=False)]
            )
            out = []
            for f in pick:
                x = x_vae_from_txt(ml, tr, f, tmp_dir)
                if x is not None and len(x) >= 2 * stride + 2:
                    out.append(x[::stride])
            if out:
                sets[f"cat_{cat}"] = out
            print(f"  [카테고리 negative] {cat}: 파일 {len(files)} 중 {len(out)} 사용")

    # ── 점수 계산 ─────────────────────────────────────────────
    results: dict[str, dict] = {}
    for name, clips in sets.items():
        if not clips:
            continue
        xp, xc, cid = pair_stack(clips)
        z = torch.from_numpy(encode_np(xp, xc)).to(dev)
        d_step = scorer.neg_log_prob(z).cpu().numpy()
        maha = scorer.mahalanobis(z).cpu().numpy()
        knn = scorer.knn_distance(z, k=args.knn_k).cpu().numpy()
        raw = raw_score(xp, xc)
        # 클립 단위 배치 marginal KL — 스텝별 보상이 아니라 **분포 수준** 지표다.
        # 클립 길이가 KL 을 바꾸므로 클립당 고정 개수(mkl_batch)로 표본을 맞춘다.
        mkl = []
        for ci in range(int(cid.max()) + 1):
            sel = np.flatnonzero(cid == ci)
            if len(sel) < args.mkl_batch:
                continue
            pick = rng.choice(sel, args.mkl_batch, replace=False)
            mkl.append(float(scorer.marginal_kl(z[torch.from_numpy(pick).to(dev)]).item()))
        results[name] = {
            "num_clips": len(clips),
            "num_pairs": int(len(xp)),
            "marginal_kl": float(scorer.marginal_kl(z).item()),
            "frame": {"d_step": d_step, "maha": maha, "knn": knn, "raw": raw},
            "clip": {
                k: clip_means(v, cid) for k, v in (("d_step", d_step), ("maha", maha), ("knn", knn), ("raw", raw))
            },
            "clip_mkl": np.array(mkl),
        }
        print(
            f"  [점수] {name:16s} 클립 {len(clips):4d} 쌍 {len(xp):7,d}  "
            f"d_step p50 {np.percentile(d_step, 50):9.2f}  marginal_KL {results[name]['marginal_kl']:8.3f}"
        )

    exp = results["expert"]
    d_exp_med = float(np.median(exp["frame"]["d_step"]))

    # ── AUROC 표 ──────────────────────────────────────────────
    rows = []
    for name, r in results.items():
        if name in ("expert", "expert_train"):
            continue
        row = {
            "negative": name,
            "num_clips": r["num_clips"],
            "num_pairs": r["num_pairs"],
            "marginal_kl": r["marginal_kl"],
        }
        for lvl in ("frame", "clip"):
            for met in ("d_step", "maha", "knn", "raw"):
                row[f"auroc_{lvl}_{met}"] = auroc(exp[lvl][met], r[lvl][met])
        # 통제 — train 세션 expert 를 positive 로 썼을 때. held-out 격차와 분리해 읽는다.
        row["auroc_clip_d_step_vs_train"] = auroc(results["expert_train"]["clip"]["d_step"], r["clip"]["d_step"])
        row["auroc_clip_marginal_kl"] = auroc(exp["clip_mkl"], r["clip_mkl"])
        row["num_mkl_clips"] = int(len(r["clip_mkl"]))
        row["d_step_stats"] = quantiles(r["frame"]["d_step"])
        row["d_step_excess_p50"] = float(np.median(r["frame"]["d_step"]) - d_exp_med)
        rows.append(row)

    # ── c_kl 동적 범위 ────────────────────────────────────────
    ckl_table = []
    for c in args.c_kl_sweep:
        entry = {"c_kl": c, "expert": {}, "negatives": {}}
        for tag, off in (("absolute", 0.0), ("offset_removed", d_exp_med)):
            entry["expert"][tag] = quantiles(np.exp(-c * (exp["frame"]["d_step"] - off)))
        for name, r in results.items():
            if name in ("expert", "expert_train"):
                continue
            entry["negatives"][name] = {
                tag: quantiles(np.exp(-c * (r["frame"]["d_step"] - off)))
                for tag, off in (("absolute", 0.0), ("offset_removed", d_exp_med))
            }
        ckl_table.append(entry)

    # ── 저장 ──────────────────────────────────────────────────
    summary = {
        "ckpt": args.ckpt,
        "stride": stride,
        "fps": 60 // stride,
        "latent_dim": encoder.latent_dim,
        "seed": args.seed,
        "val_sessions": sorted(val_sessions),
        "num_train_clips": len(train_clips),
        "num_val_clips": len(val_clips),
        "num_ref_pairs": int(len(z_ref)),
        "expert": {
            "d_step": quantiles(exp["frame"]["d_step"]),
            "num_pairs": exp["num_pairs"],
            "num_clips": exp["num_clips"],
            "d_step_median": d_exp_med,
            "marginal_kl": exp["marginal_kl"],
        },
        "auroc": rows,
        "c_kl_sweep": ckl_table,
        "noise_sigmas": args.noise_sigmas,
    }
    with open(os.path.join(metrics_dir, "style_reward_validation.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    import csv

    with open(os.path.join(metrics_dir, "auroc_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "negative",
                "num_clips",
                "num_pairs",
                "auroc_frame_d_step",
                "auroc_clip_d_step",
                "auroc_frame_raw",
                "auroc_clip_raw",
                "auroc_frame_maha",
                "auroc_clip_maha",
                "auroc_frame_knn",
                "auroc_clip_knn",
                "auroc_clip_d_step_vs_train",
                "auroc_clip_marginal_kl",
                "num_mkl_clips",
                "d_step_p50",
                "d_step_excess_p50",
                "marginal_kl",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["negative"],
                    r["num_clips"],
                    r["num_pairs"],
                    f"{r['auroc_frame_d_step']:.4f}",
                    f"{r['auroc_clip_d_step']:.4f}",
                    f"{r['auroc_frame_raw']:.4f}",
                    f"{r['auroc_clip_raw']:.4f}",
                    f"{r['auroc_frame_maha']:.4f}",
                    f"{r['auroc_clip_maha']:.4f}",
                    f"{r['auroc_frame_knn']:.4f}",
                    f"{r['auroc_clip_knn']:.4f}",
                    f"{r['auroc_clip_d_step_vs_train']:.4f}",
                    f"{r['auroc_clip_marginal_kl']:.4f}",
                    r["num_mkl_clips"],
                    f"{r['d_step_stats']['p50']:.3f}",
                    f"{r['d_step_excess_p50']:.3f}",
                    f"{r['marginal_kl']:.4f}",
                ]
            )

    np.savez_compressed(
        os.path.join(metrics_dir, "raw_scores.npz"),
        **{f"{n}__{met}": r["frame"][met] for n, r in results.items() for met in ("d_step", "maha", "knn", "raw")},
    )

    # ── 플롯 ──────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # (1) d_step 분포
        order = ["expert", "expert_train"] + [r["negative"] for r in rows]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.boxplot([results[n]["frame"]["d_step"] for n in order], tick_labels=order, showfliers=False)
        ax.set_ylabel("d_step  [nat]")
        ax.set_title("latent neg-log-density: expert (held-out sessions) vs negatives")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "d_step_distribution.png"), dpi=130)
        plt.close(fig)

        # (2) 잡음 민감도
        sig = args.noise_sigmas
        med = [np.median(results[f"jnoise_s{s}"]["frame"]["d_step"]) for s in sig if f"jnoise_s{s}" in results]
        au_l = [next(r["auroc_frame_d_step"] for r in rows if r["negative"] == f"jnoise_s{s}") for s in sig]
        au_r = [next(r["auroc_frame_raw"] for r in rows if r["negative"] == f"jnoise_s{s}") for s in sig]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(sig, med, "o-", label="negative")
        axes[0].axhline(d_exp_med, color="k", ls="--", label="expert median")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("joint noise sigma  [rad]")
        axes[0].set_ylabel("median d_step  [nat]")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(sig, au_l, "o-", label="latent d_step")
        axes[1].plot(sig, au_r, "s--", label="raw x_vae baseline")
        axes[1].axhline(0.5, color="k", ls=":", label="chance")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("joint noise sigma  [rad]")
        axes[1].set_ylabel("AUROC (frame)")
        axes[1].set_ylim(0.4, 1.02)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.suptitle("joint-noise sensitivity")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "noise_sensitivity.png"), dpi=130)
        plt.close(fig)

        # (3) c_kl 동적 범위 (오프셋 제거 기준)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        order = [n for n in order if n != "expert_train"]  # c_kl 표는 held-out expert 기준
        width = 0.8 / len(args.c_kl_sweep)
        xs = np.arange(len(order))
        for i, e in enumerate(ckl_table):
            vals = [
                e["expert"]["offset_removed"]["p50"]
                if n.startswith("expert")
                else e["negatives"][n]["offset_removed"]["p50"]
                for n in order
            ]
            ax.bar(xs + i * width, vals, width, label=f"c_kl={e['c_kl']}")
        ax.set_xticks(xs + 0.4 - width / 2)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel("median r_style (offset removed)")
        ax.set_title("style reward dynamic range")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "c_kl_dynamic_range.png"), dpi=130)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[경고] 플롯 실패: {exc}")

    # ── 콘솔 표 ───────────────────────────────────────────────
    print(
        f"\n[expert held-out] d_step p50 {d_exp_med:.2f}  "
        f"p5 {np.percentile(exp['frame']['d_step'], 5):.2f}  p95 {np.percentile(exp['frame']['d_step'], 95):.2f}"
    )
    print(f"[expert train]    d_step p50 {np.median(results['expert_train']['frame']['d_step']):.2f}")
    print(
        f"\n{'negative':18s} {'AUROC frame':>12s} {'AUROC clip':>11s} {'clip(vs tr)':>12s} | "
        f"{'raw frame':>10s} {'raw clip':>9s} | {'maha clip':>10s} {'knn clip':>9s} "
        f"{'margKL clip':>12s} | {'d_step p50':>11s}"
    )
    print("-" * 134)
    for r in rows:
        print(
            f"{r['negative']:18s} {r['auroc_frame_d_step']:12.3f} {r['auroc_clip_d_step']:11.3f} "
            f"{r['auroc_clip_d_step_vs_train']:12.3f} | "
            f"{r['auroc_frame_raw']:10.3f} {r['auroc_clip_raw']:9.3f} | "
            f"{r['auroc_clip_maha']:10.3f} {r['auroc_clip_knn']:9.3f} {r['auroc_clip_marginal_kl']:12.3f} | "
            f"{r['d_step_stats']['p50']:11.2f}"
        )
    print(f"\n[저장] {metrics_dir}  /  {figures_dir}")


if __name__ == "__main__":
    main()
