# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 latent-imitation env 검증 (Phase 2 게이트 V1a · V1b · V3 · V4).

학습을 띄우기 전에 통과해야 하는 항목만 잰다. 판정 기준은 실행 전에 고정했다.

======  =========================================================  ===================================
게이트  묻는 것                                                    기준
======  =========================================================  ===================================
V1a     motion 경로 두 개(AMP obs vs ``x_vae``)의 성분 대조         순열 일치 또는 차이의 명시
V1b     **sim 실측** ``x_vae`` vs **motion** ``x_vae``               블록별 편차가 물리적으로 설명됨
V3      한 스텝 ``r_style`` 의 env 간 산포                           ``std/mean`` > 0 (효과크기 병기)
V4      정지 env 의 ``r_style`` 이 유의하게 낮은가                   expert 군 대비 ``>= 2`` std 낮음
======  =========================================================  ===================================

V3 을 "std > 0" 으로만 재면 부동소수 잡음으로도 통과하므로 **효과크기**(변동계수)를 함께
낸다. V4 도 눈으로 보지 않고 **expert 군의 across-env std 단위**로 잰다 — PPO advantage 가
이 항을 볼 수 있는지를 결정하는 수치가 그것이다.

세 군을 비교한다:

===========  =====================================================================
군           만드는 법
===========  =====================================================================
``expert``   **held-out val 세션** expert 창 (sim 아님). 정책이 도달해야 할 값
``static``   sim, 액션 0 — PD 가 기본 자세를 잡아 로봇이 **실제로 서 있다**
``free``     sim, 랜덤 액션 — 움직인다
===========  =====================================================================

sim 군에 상태를 인위로 주입하지 않는다. 액션만 바꾼다.

실행::

    CUDA_VISIBLE_DEVICES=1 /home/user/miniconda3/envs/isaac-6.0/bin/python \\
      scripts/imitation_learning/verify_go2_latent_env.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=96, help="검증용 env 수 (절반씩 static/free 로 나눈다)")
parser.add_argument("--steps", type=int, default=200, help="굴릴 policy 스텝 수 (앞 절반은 전이 구간으로 버린다)")
parser.add_argument("--out_dir", default=None, help="수치를 JSON 으로 저장할 디렉터리")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ruff: noqa: E402
import glob
import importlib.util
import json
import os
import sys

import gymnasium as gym
import torch

from isaaclab.utils.math import convert_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

X_VAE_BLOCKS = [
    ("base_height", 0, 1, "m"),
    ("rot6d", 1, 7, "-"),
    ("lin_vel", 7, 10, "m/s"),
    ("ang_vel", 10, 13, "rad/s"),
    ("foot_pos", 13, 25, "m"),
    ("joint_pos", 25, 37, "rad"),
    ("joint_vel", 37, 49, "rad/s"),
]


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _block_report(a: torch.Tensor, b: torch.Tensor) -> list[dict]:
    """블록별 최대·중앙 절대오차와 상대 스케일."""
    rows = []
    for name, lo, hi, unit in X_VAE_BLOCKS:
        d = (a[:, lo:hi] - b[:, lo:hi]).abs()
        rows.append(
            {
                "block": name,
                "unit": unit,
                "max_abs_err": float(d.max()),
                "median_abs_err": float(d.median()),
                "p95_abs_err": float(torch.quantile(d.reshape(-1), 0.95)),
                "ref_rms": float(b[:, lo:hi].pow(2).mean().sqrt()),
            }
        )
    return rows


def _print_blocks(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"  {'block':<12} {'unit':<6} {'max|err|':>12} {'p95|err|':>12} {'median|err|':>12} {'ref RMS':>12}")
    for r in rows:
        print(
            f"  {r['block']:<12} {r['unit']:<6} {r['max_abs_err']:>12.3e} {r['p95_abs_err']:>12.3e}"
            f" {r['median_abs_err']:>12.3e} {r['ref_rms']:>12.3e}"
        )


def main() -> None:
    results: dict = {}

    # ══════════════════════════════════════════════════════════
    # V1a — motion 경로 두 개의 성분 대조 (sim 불필요)
    # ══════════════════════════════════════════════════════════
    trainer = _load_module(os.path.join(_HERE, "train_go2_motion_vae.py"), "_go2_vae_trainer")
    # baseline AMP env (수정 금지 대상) — 패키지로 임포트해야 상대 임포트가 풀린다.
    import importlib

    base_env_mod = importlib.import_module("isaaclab_tasks.direct.go2_imitation_tracking.go2_imitation_tracking_env")
    latent_pkg = "source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation_latent"
    ml = _load_module(os.path.join(_REPO, latent_pkg, "motion_lib.py"), "_go2_latent_motion_lib")

    pkl = sorted(glob.glob(os.path.join(_REPO, latent_pkg, "imitation/motion_pkl/walk__*.pkl")))[0]
    x_vae_np = trainer.build_x_vae(ml, pkl)  # [T, 49] — 인코더가 학습된 순서
    root_pos, root_quat_wxyz, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos, _fps = ml.Go2MotionLib._load_pkl(pkl)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    n_fr = len(x_vae_np)
    empty_axis = torch.zeros(0, 3, device=dev)
    amp_base = base_env_mod._compute_amp_obs(
        t(dof_pos), t(dof_vel), t(root_pos), t(lin_vel), t(ang_vel), t(foot_pos), empty_axis
    )  # [T, 43]
    quat_xyzw = convert_quat(t(root_quat_wxyz), to="xyzw")
    amp_rot = base_env_mod._apply_root_rot_tan_norm(quat_xyzw.unsqueeze(1), n_fr, 1)[:, 0, :]  # [T, 6]
    amp_obs = torch.cat([amp_base, amp_rot], dim=-1)  # [T, 49]
    x_vae = t(x_vae_np)

    # AMP 레이아웃: [q 12][qd 12][h 1][lin 3][ang 3][foot 12][rot_tan_norm 6]
    amp_perm = {
        "base_height": (24, 25),
        "lin_vel": (25, 28),
        "ang_vel": (28, 31),
        "foot_pos": (31, 43),
        "joint_pos": (0, 12),
        "joint_vel": (12, 24),
    }
    v1a_rows = []
    for name, lo, hi, unit in X_VAE_BLOCKS:
        if name == "rot6d":
            continue
        a_lo, a_hi = amp_perm[name]
        d = (x_vae[:, lo:hi] - amp_obs[:, a_lo:a_hi]).abs()
        v1a_rows.append({"block": name, "unit": unit, "max_abs_err": float(d.max()), "amp_slice": [a_lo, a_hi]})
    # rot6d 는 순열이 아니다 — 두 표현이 같은 회전을 나타내는지 외적으로 확인한다.
    c0, c1 = x_vae[:, 1:4], x_vae[:, 4:7]
    amp_tan, amp_norm = amp_obs[:, 43:46], amp_obs[:, 46:49]
    col2_from_x_vae = torch.cross(c0, c1, dim=-1)  # R·z = (R·x) × (R·y)
    rot_consistent = float((col2_from_x_vae - amp_norm).abs().max())
    rot_first_col = float((c0 - amp_tan).abs().max())
    rot_direct = float((x_vae[:, 1:7] - amp_obs[:, 43:49]).abs().max())

    print("\n" + "=" * 78)
    print(f"V1a — motion 경로 대조 (AMP obs 49 vs x_vae 49), 프레임 {n_fr}")
    print("=" * 78)
    print(f"  {'block':<12} {'unit':<6} {'AMP slice':>12} {'max|err|':>12}")
    for r in v1a_rows:
        print(f"  {r['block']:<12} {r['unit']:<6} {str(r['amp_slice']):>12} {r['max_abs_err']:>12.3e}")
    print(f"\n  rot6d 는 순열이 아니다 — 직접 대조 max|err| {rot_direct:.3e}")
    print(f"    1 열 일치 (x_vae[1:4] vs AMP tan)          max|err| {rot_first_col:.3e}")
    print(f"    (R·x)×(R·y) vs AMP norm  →  같은 회전인가  max|err| {rot_consistent:.3e}")
    results["V1a"] = {
        "n_frames": int(n_fr),
        "pkl": os.path.basename(pkl),
        "blocks": v1a_rows,
        "rot6d_direct_max_err": rot_direct,
        "rot6d_first_col_max_err": rot_first_col,
        "rot6d_cross_vs_amp_norm_max_err": rot_consistent,
    }

    # ══════════════════════════════════════════════════════════
    # env 생성
    # ══════════════════════════════════════════════════════════
    env_cfg = parse_env_cfg("Go2-Imitation-Latent-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.rel_rest_init = 0.0  # 전 env 를 RSI 로 — 참조 프레임과 대조하려면 필수
    env_cfg.rel_standing_envs = 0.0
    env_cfg.early_termination = False  # 군을 유지한 채로 굴린다
    env_cfg.domain_rand = False
    env_cfg.standing_style_substitute = False  # 스타일 보상 자체를 보려면 대체 경로를 꺼야 한다
    env = gym.make("Go2-Imitation-Latent-v0", cfg=env_cfg).unwrapped
    torch.manual_seed(0)
    env.reset()

    # ══════════════════════════════════════════════════════════
    # V1b — sim 실측 x_vae vs motion x_vae (같은 RSI 프레임)
    # ══════════════════════════════════════════════════════════
    # 리셋 직후 발 위치(body_pos_w)는 physics view 를 한 번 굴려야 갱신된다. 그 substep 동안
    # 액추에이터가 관절을 끌지 않도록 PD 목표를 **리셋 자세 그대로** 준다 — 그러지 않으면
    # 기본 자세로 끌려가는 토크가 5 ms 만에 관절속도를 크게 바꿔 채널 편차와 뒤섞인다.
    env._robot.set_joint_position_target(env._robot.data.joint_pos.torch.clone())
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env._get_observations()
    x_sim = env._x_vae_hist[:, 0].clone()
    x_ref = env.reference_x_vae(env._last_rsi_motion_ids, env._last_rsi_times)
    v1b_rows = _block_report(x_sim, x_ref)
    print("\n" + "=" * 78)
    print(f"V1b — sim 실측 x_vae vs motion x_vae (RSI 직후, physics substep {env.physics_dt * 1e3:.1f} ms 경과)")
    print("=" * 78)
    _print_blocks("", v1b_rows)
    results["V1b"] = {"n_envs": int(x_sim.shape[0]), "physics_dt_ms": env.physics_dt * 1e3, "blocks": v1b_rows}

    # ══════════════════════════════════════════════════════════
    # V3 / V4 — sim 두 군 + held-out expert 기준선
    # ══════════════════════════════════════════════════════════
    # sim 군은 **물리적으로 진짜**다 — 액션 0 이면 PD 가 기본 자세를 잡아 로봇이 실제로 서
    # 있고(정지), 랜덤 액션이면 움직인다. 키네마틱 재생 같은 인위적 상태 주입을 쓰지 않는다.
    n = env.num_envs
    half = n // 2
    idx_static = torch.arange(0, half, device=env.device)
    idx_free = torch.arange(half, n, device=env.device)

    hist: list[torch.Tensor] = []
    kl_hist: list[torch.Tensor] = []
    for _k in range(args_cli.steps):
        act = torch.zeros(n, env.cfg.action_space, device=env.device)
        act[idx_free] = 0.4 * torch.randn(len(idx_free), env.cfg.action_space, device=env.device)
        env.step(act)
        hist.append(env._style_reward.clone())
        kl_hist.append(env._latent_kl.clone())

    R = torch.stack(hist)  # [steps, n]
    KL = torch.stack(kl_hist)
    warm = args_cli.steps // 2  # RSI 전이 구간을 버린다 (정지 군이 자세를 잡을 시간)
    R, KL = R[warm:], KL[warm:]

    # ── held-out expert 기준선 (val 세션, sim 아님) ──────────────
    # 정책이 도달해야 하는 값이 무엇인지 보여주는 군이다. 참조 통계는 train 으로만 적합했으므로
    # 여기 val 을 쓰는 것은 누수가 아니다 (V5 참조).
    ds_args = argparse.Namespace(
        pkl_dir=os.path.join(_REPO, latent_pkg, "imitation/motion_pkl"),
        motion_lib=os.path.join(_REPO, latent_pkg, "motion_lib.py"),
        split_json=os.path.join(_REPO, env._ref_stats["split_json"]),
        stride=int(env._ref_stats["stride"]),
    )
    _tr, val_ds = trainer.load_dataset(ds_args)
    k_ahead = int(env._ref_stats["k_ahead"])
    wn = env._window_n
    va_prev, va_curr = trainer.make_pairs(val_ds["clips"], k_ahead)
    mean_np = env._motion_encoder.obs_mean.cpu().numpy()
    std_np = env._motion_encoder.obs_std.cpu().numpy()
    zs = []
    for i in range(0, len(va_prev), 65536):
        xp = torch.from_numpy((va_prev[i : i + 65536] - mean_np) / std_np).to(env.device)
        xc = torch.from_numpy((va_curr[i : i + 65536] - mean_np) / std_np).to(env.device)
        zs.append(env._motion_encoder.inference(xp, xc, normalized=True))
    z_val = torch.cat(zs)
    starts, off = [], 0
    for c in val_ds["clips"]:
        npair = len(c["x"]) - k_ahead
        if npair >= wn:
            starts.extend(range(off, off + npair - wn + 1))
        off += max(npair, 0)
    widx = torch.tensor(starts, device=env.device).unsqueeze(1) + torch.arange(wn, device=env.device).unsqueeze(0)
    zw = z_val[widx]
    mu_w = zw.mean(dim=1)
    var_w = zw.var(dim=1, unbiased=False).clamp(min=env._var_floor)
    kl_val = 0.5 * (torch.log(env._var_ref / var_w) + (var_w + (mu_w - env._mu_ref).pow(2)) / env._var_ref - 1.0).sum(
        dim=-1
    )
    r_val = torch.exp(-env._c_kl * (kl_val - env._kl_offset))

    def sim_stat(ix: torch.Tensor) -> dict:
        r = R[:, ix]
        return {
            "r_mean": float(r.mean()),
            "r_std_across_env": float(r.std(dim=1).mean()),
            "r_cv": float(r.std(dim=1).mean() / max(abs(float(r.mean())), 1e-12)),
            "kl_median": float(KL[:, ix].median()),
        }

    s_sta, s_free = sim_stat(idx_static), sim_stat(idx_free)
    s_exp = {
        "r_mean": float(r_val.mean()),
        "r_std_across_env": float(r_val.std()),
        "r_cv": float(r_val.std() / max(abs(float(r_val.mean())), 1e-12)),
        "kl_median": float(kl_val.median()),
        "n_windows": int(len(r_val)),
    }

    step_std = R.std(dim=1)
    step_mean = R.mean(dim=1)
    v3 = {
        "steps_used": int(R.shape[0]),
        "min_step_std": float(step_std.min()),
        "mean_step_std": float(step_std.mean()),
        "mean_step_cv": float((step_std / step_mean.abs().clamp(min=1e-12)).mean()),
        "static_group_step_std_min": float(R[:, idx_static].std(dim=1).min()),
    }
    sep_exp = (s_exp["r_mean"] - s_sta["r_mean"]) / max(s_exp["r_std_across_env"], 1e-12)
    sep_free = (s_free["r_mean"] - s_sta["r_mean"]) / max(s_free["r_std_across_env"], 1e-12)
    v4 = {
        "expert_holdout_r_mean": s_exp["r_mean"],
        "static_r_mean": s_sta["r_mean"],
        "free_r_mean": s_free["r_mean"],
        "expert_holdout_std": s_exp["r_std_across_env"],
        "separation_static_vs_expert_in_expert_std": float(sep_exp),
        "separation_static_vs_free_in_free_std": float(sep_free),
        "expert_kl_median": s_exp["kl_median"],
        "static_kl_median": s_sta["kl_median"],
        "free_kl_median": s_free["kl_median"],
    }

    print("\n" + "=" * 78)
    print(f"V3 / V4 — r_style (sim: 마지막 {R.shape[0]} 스텝 × env / expert: val 창 {s_exp['n_windows']:,}개)")
    print("=" * 78)
    print(f"  {'군':<22} {'r_mean':>10} {'std':>10} {'변동계수':>10} {'D_e 중앙':>12}")
    for nm, st in (("expert (held-out val)", s_exp), ("static (sim, act=0)", s_sta), ("free (sim, random)", s_free)):
        print(
            f"  {nm:<22} {st['r_mean']:>10.4f} {st['r_std_across_env']:>10.4f}"
            f" {st['r_cv']:>10.4f} {st['kl_median']:>12.3f}"
        )
    print(
        f"\n  V3  한 스텝 env 간 std — 최소 {v3['min_step_std']:.3e} · 평균 {v3['mean_step_std']:.3e}"
        f" · 평균 변동계수 {v3['mean_step_cv']:.4f}"
    )
    print(f"  V4  static 이 expert 보다 {sep_exp:.2f} × (expert std) 낮다   [기준 >= 2]")
    print(f"      static 이 free 보다   {sep_free:.2f} × (free std) 낮다")

    results["V3"] = v3
    results["V4"] = v4
    results["groups"] = {"expert_holdout": s_exp, "static": s_sta, "free": s_free}
    results["config"] = {
        "num_envs": n,
        "steps": args_cli.steps,
        "window_n": int(env._window_n),
        "c_kl": float(env._c_kl),
        "kl_offset": float(env._kl_offset),
        "pair_dt_s": float(env._pair_dt),
        "policy_dt_s": float(env.step_dt),
        "x_hist_depth": int(env._x_vae_hist.shape[1]),
        "x_lag_lo": int(env._x_lag_lo),
        "x_lag_frac": float(env._x_lag_frac),
    }

    # ══════════════════════════════════════════════════════════
    # V6 — 리셋 warm-up 이 보상을 에피소드 나이와 상관시키는가
    # ══════════════════════════════════════════════════════════
    # 창이 찰 때까지(약 11 step) env 는 `neutral_reward` 를 받는다. 그 값이 실제로 달성
    # 가능한 값보다 높으면 "종료 → 리셋 → 공짜 보상" 이 exploit 이 된다. V3/V4 는
    # `early_termination=False` 로 쟀으므로 이 항을 구조적으로 볼 수 없다 — 여기서 켠다.
    env.cfg.early_termination = True
    env.reset()
    r_hist, ready_hist, age_hist = [], [], []
    for _k in range(args_cli.steps):
        act = 0.4 * torch.randn(n, env.cfg.action_space, device=env.device)
        env.step(act)
        r_hist.append(env._style_reward.clone())
        ready_hist.append((env._z_count >= env._window_n).clone())
        age_hist.append(env.episode_length_buf.clone().float())
    Rt = torch.stack(r_hist).reshape(-1)
    RD = torch.stack(ready_hist).reshape(-1)
    AG = torch.stack(age_hist).reshape(-1)
    warm_frac = float((~RD).float().mean())
    r_warm = float(Rt[~RD].mean()) if (~RD).any() else float("nan")
    r_ready = float(Rt[RD].mean()) if RD.any() else float("nan")
    rc = torch.stack([AG, Rt])
    corr = float(torch.corrcoef(rc)[0, 1])
    v6 = {
        "steps": int(args_cli.steps),
        "warmup_step_fraction": warm_frac,
        "r_style_warmup_mean": r_warm,
        "r_style_ready_mean": r_ready,
        "neutral_reward": float(env.cfg.latent.neutral_reward),
        "corr_episode_age_vs_r_style": corr,
        "mean_episode_len_steps": float(AG.mean()),
    }
    print("\n" + "=" * 78)
    print("V6 — 리셋 warm-up 편향 (early_termination=True)")
    print("=" * 78)
    print(f"  warm-up step 비율 {warm_frac:.3%} · 평균 에피소드 나이 {v6['mean_episode_len_steps']:.1f} step")
    print(f"  r_style — warm-up {r_warm:.4f} (neutral {v6['neutral_reward']:.3f}) vs 창이 찬 뒤 {r_ready:.4f}")
    print(f"  에피소드 나이 ↔ r_style 상관 {corr:+.4f}   [0 에 가까울수록 좋다]")
    results["V6"] = v6

    # ══════════════════════════════════════════════════════════
    # V5 — 참조 통계의 출처 (파일 내용으로 확인)
    # ══════════════════════════════════════════════════════════
    ref = env._ref_stats
    v5 = {
        "ckpt_path": ref["ckpt_path"],
        "ckpt_sha256": ref["ckpt_sha256"][:16],
        "split_json": ref["split_json"],
        "n_train_sessions": len(ref["train_sessions"]),
        "n_val_sessions_excluded": len(ref["val_sessions_excluded"]),
        "session_overlap": sorted(set(ref["train_sessions"]) & set(ref["val_sessions_excluded"])),
        "n_pairs": int(ref["n_pairs"]),
        "n_windows": int(ref["n_windows"]),
        "d_e_quantiles": ref["d_e_quantiles"],
    }
    print("\n" + "=" * 78)
    print("V5 — 참조 통계 출처")
    print("=" * 78)
    print(
        f"  train 세션 {v5['n_train_sessions']} · 제외된 val 세션 {v5['n_val_sessions_excluded']}"
        f" · 교집합 {len(v5['session_overlap'])}"
    )
    print(f"  전이쌍 {v5['n_pairs']:,} · 창 {v5['n_windows']:,} · ckpt {v5['ckpt_path']}")
    results["V5"] = v5

    if args_cli.out_dir:
        os.makedirs(args_cli.out_dir, exist_ok=True)
        out = os.path.join(args_cli.out_dir, "latent_env_verification.json")
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\n[저장] {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
