# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
순위 0 검증 스크립트 v3: per-foot 접지율 + ContactSensor vs 운동학 접지 비교.

ground_z 소스: height_scanner RayCaster의 ray_hits_w (world-frame, (N,C,3)).
sole offset 자동 추정: warm-up 구간 sensor=True 프레임의 (foot_z - ground_z) 중앙값.
진단 항목: scanner 커버리지 bbox, 발별 ray_valid_ratio, clearance 통계.

실행:
  cd /home/lgb/IsaacLab && conda run -n isaac-parkour ./isaaclab.sh -p \
      scripts/reinforcement_learning/rsl_rl/verify_3leg_contact.py \
      --task Go2-Parkour-Direct-v0 --num_envs 64 --num_steps 2000 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="3-leg gait contact diagnostic for Go2 Parkour.")
parser.add_argument("--task", type=str, default="Go2-Parkour-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=2000)
# --checkpoint provided by cli_args.add_rsl_rl_args()
parser.add_argument(
    "--force_thresh", type=float, default=2.0, help="Contact force threshold [N] (default 2.0, matches env reward)"
)
parser.add_argument(
    "--z_thresh", type=float, default=0.03, help="Extra clearance above sole_offset for kin contact [m]"
)
parser.add_argument("--v_thresh", type=float, default=0.2, help="Foot vertical velocity threshold [m/s]")
parser.add_argument(
    "--kin_only_flag_thresh",
    type=float,
    default=0.10,
    help="kin_only excess above which edge-underreporting is flagged",
)
parser.add_argument("--cal_steps", type=int, default=200, help="Warm-up steps for sole-offset calibration")
parser.add_argument("--seed", type=int, default=42)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunnerParkour

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

SCRIPT_VERSION = "v3"


# ── helpers ───────────────────────────────────────────────────────────────────


def _ground_z_from_ray_hits(
    feet_pos_w: torch.Tensor,  # (N, 4, 3) world-frame foot positions
    ray_hits_w: torch.Tensor,  # (N, C, 3) world-frame ray hit positions
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find ground_z under each foot as z of the nearest finite ray hit.

    Returns:
        ground_z   : (N, 4) float32 — terrain z under each foot (NaN if no valid ray)
        valid_mask : (N, 4) bool    — True where a finite ray was found
        min_xy_dist: (N, 4) float32 — xy distance to the nearest used ray (diagnostic)
    """
    N, num_feet, _ = feet_pos_w.shape
    device = feet_pos_w.device

    # Finite mask: (N, C)
    finite_mask = torch.isfinite(ray_hits_w).all(dim=-1)

    # Squared xy distances: (N, 4, C)
    ray_xy = ray_hits_w[..., :2]  # (N, C, 2)
    foot_xy = feet_pos_w[..., :2]  # (N, 4, 2)
    diff = foot_xy.unsqueeze(2) - ray_xy.unsqueeze(1)  # (N, 4, C, 2)
    sq_dist = (diff * diff).sum(dim=-1)  # (N, 4, C)

    INF_DIST = 1e9
    sq_dist = torch.where(
        finite_mask.unsqueeze(1).expand_as(sq_dist),
        sq_dist,
        torch.full_like(sq_dist, INF_DIST),
    )

    nearest_idx = sq_dist.argmin(dim=-1)  # (N, 4)
    min_sq_dist = sq_dist.min(dim=-1).values  # (N, 4)
    valid_mask = min_sq_dist < (INF_DIST / 2.0)
    min_xy_dist = min_sq_dist.sqrt()  # (N, 4)

    # Gather z for nearest ray
    ray_z = ray_hits_w[..., 2]  # (N, C)
    ground_z = ray_z.gather(1, nearest_idx.reshape(N, -1)).reshape(N, num_feet)

    ground_z = torch.where(valid_mask, ground_z, torch.full_like(ground_z, float("nan")))

    return ground_z, valid_mask, min_xy_dist


def _print_table(foot_names: list[str], metrics: dict[str, np.ndarray], title: str) -> str:
    col_w = 14
    lines = [
        title,
        f"{'metric':<24}" + "".join(f"{n:>{col_w}}" for n in foot_names),
        "-" * (24 + col_w * len(foot_names)),
    ]
    for key, vals in metrics.items():
        lines.append(f"{key:<24}" + "".join(f"{v:>{col_w}.4f}" for v in vals))
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # FAIL-LOUD version stamp — first thing printed after Isaac boot
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"=== verify_3leg {SCRIPT_VERSION} (height_scanner ground_z) ===")
    print(f"=== started {ts} ===")
    print(f"{'=' * 60}\n")

    # ── Environment setup ──────────────────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = False
    for attr in ("debug_vis_edge_mask", "enable_keyboard_view_switch"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    # Checkpoint
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[{SCRIPT_VERSION}] Checkpoint : {resume_path}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    assert agent_cfg.class_name == "OnPolicyRunnerParkour", agent_cfg.class_name
    runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    base_env = env.unwrapped
    device = base_env.device
    N = args_cli.num_envs
    T = args_cli.num_steps
    CAL = args_cli.cal_steps

    # Foot ids and names (runtime, no hard-coding)
    feet_ids = list(base_env._feet_ids)
    foot_names = [base_env._contact_sensor.body_names[i] for i in feet_ids]
    print(f"[{SCRIPT_VERSION}] Foot body names : {foot_names}")
    print(f"[{SCRIPT_VERSION}] Foot body ids   : {feet_ids}")

    from isaaclab_tasks.direct.parkour.parkour_env import _TERRAIN_CLASS_NAMES

    # ── height_scanner check ───────────────────────────────────────────────────
    hs = getattr(base_env, "_height_scanner", None)
    hs_valid = False
    if hs is not None:
        _probe = hs.data.ray_hits_w
        if _probe.dim() == 3 and _probe.shape[-1] >= 3:
            hs_valid = True
            print(f"[{SCRIPT_VERSION}] height_scanner : ray_hits_w shape = {tuple(_probe.shape)}")
        else:
            print(f"[{SCRIPT_VERSION}] WARNING: ray_hits_w shape unexpected: {tuple(_probe.shape)}. Kin DISABLED.")
    else:
        print(f"[{SCRIPT_VERSION}] WARNING: _height_scanner not found. Kin DISABLED.")

    # ── SMOKING GUN: scanner coverage bbox vs foot positions ──────────────────
    # Run one step to get real foot positions and ray hit bbox, then print both.
    obs = env.get_observations()
    with torch.inference_mode():
        base_env._commands[:, 0] = 1.0
        base_env._commands[:, 1] = 0.0
        base_env._commands[:, 2] = 0.0
        _actions = policy(obs)
        obs, _, _dones, _ = env.step(_actions)
        policy_nn.reset(_dones)

    if hs_valid:
        with torch.no_grad():
            rh = hs.data.ray_hits_w  # (N, C, 3)
            # Use env 0 for the bbox (representative)
            finite_mask_0 = torch.isfinite(rh[0]).all(dim=-1)  # (C,)
            rh0_fin = rh[0][finite_mask_0]  # (C_fin, 3)

            # feet for env 0 (body frame xy relative to robot base for readability)
            fp0_w = base_env._robot.data.body_pos_w[0, feet_ids, :]  # (4, 3) world
            base0 = base_env._robot.data.root_pos_w[0, :2]  # (2,) world base xy
            fp0_b = fp0_w[:, :2] - base0  # (4, 2) body-relative xy

            print(f"\n[{SCRIPT_VERSION}] ======= SCANNER COVERAGE DIAGNOSTIC (env 0) =======")
            print("  height_scanner cfg: offset_x=+0.375m, size=[1.6, 1.0]m")
            print(f"  => scanner grid (body-relative x): [{0.375 - 0.8:.3f}, {0.375 + 0.8:.3f}] m")
            print(f"  => scanner grid (body-relative y): [{-0.5:.3f}, {+0.5:.3f}] m")

            if rh0_fin.shape[0] > 0:
                rh0_b_xy = rh0_fin[:, :2] - base0  # body-relative xy of hit points
                print(
                    f"  ray_hits bbox (body-relative x): [{rh0_b_xy[:, 0].min().item():.3f}, "
                    f"{rh0_b_xy[:, 0].max().item():.3f}] m"
                )
                print(
                    f"  ray_hits bbox (body-relative y): [{rh0_b_xy[:, 1].min().item():.3f}, "
                    f"{rh0_b_xy[:, 1].max().item():.3f}] m"
                )
            else:
                print("  ray_hits (env 0): ALL INVALID (all inf) — scanner not hitting terrain!")

            print("  Foot body-relative xy (vs scanner grid):")
            for fi, fn in enumerate(foot_names):
                bx, by = fp0_b[fi, 0].item(), fp0_b[fi, 1].item()
                in_x = -0.425 <= bx <= 1.175
                in_y = -0.5 <= by <= 0.5
                coverage = "IN grid" if (in_x and in_y) else f"OUT OF GRID (x_ok={in_x}, y_ok={in_y})"
                print(f"    {fn}: body_rel_x={bx:+.3f}m  body_rel_y={by:+.3f}m  -> {coverage}")

            # Also show world-frame foot z vs scanner ray hit z distribution
            gzv, valid_v, dxy_v = _ground_z_from_ray_hits(
                base_env._robot.data.body_pos_w[:, feet_ids, :],
                rh,
            )
            ray_valid_frac = valid_v.float().mean(dim=0).cpu().numpy()  # (4,) fraction over envs
            mean_dxy = dxy_v.cpu().numpy().mean(axis=0)  # (4,) mean nearest-ray distance
            foot_z_mean = base_env._robot.data.body_pos_w[:, feet_ids, 2].mean(dim=0).cpu().numpy()
            gzv_np = gzv.cpu().numpy()
            valid_np = valid_v.cpu().numpy()
            ground_z_mean = np.where(valid_np, gzv_np, np.nan)
            ground_z_mean = np.nanmean(ground_z_mean, axis=0)  # (4,)
            clearance_mean = foot_z_mean - ground_z_mean  # (4,)

            print("\n  Per-foot scanner validity (all envs, step 1):")
            print(
                f"  {'foot':<12} {'ray_valid%':>12} {'mean_nearest_m':>16} "
                f"{'mean_foot_z':>13} {'mean_gnd_z':>12} {'mean_clr':>10}"
            )
            for fi, fn in enumerate(foot_names):
                print(
                    f"  {fn:<12} {ray_valid_frac[fi] * 100:>11.1f}% "
                    f"{mean_dxy[fi]:>16.3f}m "
                    f"{foot_z_mean[fi]:>13.3f}m "
                    f"{ground_z_mean[fi]:>12.3f}m "
                    f"{clearance_mean[fi]:>10.3f}m"
                )
            print(f"[{SCRIPT_VERSION}] =====================================================\n")

    # ── Phase 1: sole-offset calibration ─────────────────────────────────────
    sole_offset = np.zeros(4, dtype=np.float32)
    cal_clr: list[list[float]] = [[] for _ in range(4)]
    # Track ray_valid counts across calibration steps (per foot, across envs*steps)
    cal_ray_valid_count = np.zeros(4, dtype=np.int64)
    cal_total_count = np.zeros(4, dtype=np.int64)

    if hs_valid:
        print(f"[{SCRIPT_VERSION}] === Sole-offset calibration ({CAL} steps) ===")

        for step in range(CAL):
            with torch.inference_mode():
                base_env._commands[:, 0] = 1.0
                base_env._commands[:, 1] = 0.0
                base_env._commands[:, 2] = 0.0
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            ep_len = base_env.episode_length_buf.cpu().numpy()
            valid_env = ep_len > 1

            net_cf = base_env._contact_sensor.data.net_forces_w_history
            sensor_np = (torch.norm(net_cf[:, 0, feet_ids], dim=-1) > args_cli.force_thresh).cpu().numpy()  # (N, 4)

            fp_w = base_env._robot.data.body_pos_w[:, feet_ids, :]  # (N, 4, 3)
            rh = hs.data.ray_hits_w
            gz_t, rv_t, _ = _ground_z_from_ray_hits(fp_w, rh)
            fz_t = fp_w[..., 2]
            clr_np = (fz_t - gz_t).cpu().numpy()
            rv_np = rv_t.cpu().numpy()

            for fi in range(4):
                for ei in range(N):
                    cal_total_count[fi] += 1
                    if not valid_env[ei]:
                        continue
                    if rv_np[ei, fi]:
                        cal_ray_valid_count[fi] += 1
                    if sensor_np[ei, fi] and rv_np[ei, fi] and np.isfinite(clr_np[ei, fi]):
                        cal_clr[fi].append(float(clr_np[ei, fi]))

        print(f"[{SCRIPT_VERSION}] Sole offset calibration results:")
        print("  (expected offset ~0.01-0.05 m = foot body origin above sole surface)")
        print(
            f"  {'foot':<12} {'ray_valid%':>11} {'cal_n':>7} {'sole_offset':>12} {'mean_clr':>10} {'p5':>8} {'p95':>8}"
        )
        for fi, fn in enumerate(foot_names):
            rv_pct = 100.0 * cal_ray_valid_count[fi] / max(cal_total_count[fi], 1)
            vals = cal_clr[fi]
            n_cal = len(vals)
            if n_cal >= 10:
                med = float(np.median(vals))
                sole_offset[fi] = med
                print(
                    f"  {fn:<12} {rv_pct:>10.1f}% {n_cal:>7} {med:>12.4f}m "
                    f"{np.mean(vals):>10.4f}m {np.percentile(vals, 5):>8.4f} "
                    f"{np.percentile(vals, 95):>8.4f}"
                )
            else:
                sole_offset[fi] = 0.0
                print(f"  {fn:<12} {rv_pct:>10.1f}% {n_cal:>7}  *** INSUFFICIENT — using 0.0m ***")

        print(f"[{SCRIPT_VERSION}] sole_offset applied: {sole_offset}")
        print(f"[{SCRIPT_VERSION}] kin threshold = sole_offset[fi] + z_thresh({args_cli.z_thresh}m)")
        print(f"[{SCRIPT_VERSION}] =====================================================\n")
    else:
        print(f"[{SCRIPT_VERSION}] Skipping sole-offset calibration (hs_valid=False).\n")

    # ── Phase 2: main data collection ─────────────────────────────────────────
    print(f"[{SCRIPT_VERSION}] Collecting {T} steps × {N} envs ...")

    arr_sensor = np.zeros((T, N, 4), dtype=bool)
    arr_kin = np.zeros((T, N, 4), dtype=bool)
    arr_class = np.zeros((T, N), dtype=np.int64)

    # Per-step diagnostic accumulators (mean over envs)
    # For per-foot analysis: accumulate across all steps when sensor=True
    diag_rv_count = np.zeros(4, dtype=np.int64)  # ray_valid count (all envs, all steps)
    diag_total = np.zeros(4, dtype=np.int64)  # total count
    diag_clr_sum = np.zeros(4, dtype=np.float64)  # clearance sum (finite, sensor=True)
    diag_clr_n = np.zeros(4, dtype=np.int64)  # clearance count (finite, sensor=True)
    diag_fz_sum = np.zeros(4, dtype=np.float64)  # foot_z sum (all valid envs)
    diag_gz_sum = np.zeros(4, dtype=np.float64)  # ground_z sum (where ray_valid)
    diag_gz_n = np.zeros(4, dtype=np.int64)

    sole_t = torch.tensor(sole_offset, dtype=torch.float32, device=device)  # (4,)
    thresh_t = sole_t + args_cli.z_thresh  # (4,)

    t_start = time.time()
    for step in range(T):
        with torch.inference_mode():
            base_env._commands[:, 0] = 1.0
            base_env._commands[:, 1] = 0.0
            base_env._commands[:, 2] = 0.0
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # sensor contact
        net_cf = base_env._contact_sensor.data.net_forces_w_history
        sensor_contact = (torch.norm(net_cf[:, 0, feet_ids], dim=-1) > args_cli.force_thresh).cpu().numpy()  # (N, 4)

        # kinematic contact
        if hs_valid:
            fp_w = base_env._robot.data.body_pos_w[:, feet_ids, :]  # (N, 4, 3)
            rh = hs.data.ray_hits_w
            gz_t, rv_t, _ = _ground_z_from_ray_hits(fp_w, rh)
            fz_t = fp_w[..., 2]  # (N, 4)
            clr_t = fz_t - gz_t  # (N, 4)

            fv_w = base_env._robot.data.body_lin_vel_w[:, feet_ids, :]
            fvz = fv_w[..., 2].abs()  # (N, 4)

            kin_contact = (rv_t & (clr_t < thresh_t) & (fvz < args_cli.v_thresh)).cpu().numpy()

            # Diagnostics accumulation
            rv_np = rv_t.cpu().numpy()
            fz_np = fz_t.cpu().numpy()
            gz_np = gz_t.cpu().numpy()
            clr_np = clr_t.cpu().numpy()
            sn_np = sensor_contact
        else:
            kin_contact = np.zeros((N, 4), dtype=bool)

        # terrain class
        env_class = base_env._env_class.cpu().numpy()

        # Skip spawn transient
        ep_len = base_env.episode_length_buf.cpu().numpy()
        valid = ep_len > 1
        sensor_contact[~valid] = False
        kin_contact[~valid] = False

        arr_sensor[step] = sensor_contact
        arr_kin[step] = kin_contact
        arr_class[step] = env_class

        # Accumulate diagnostics
        if hs_valid:
            for fi in range(4):
                for ei in range(N):
                    if not valid[ei]:
                        continue
                    diag_total[fi] += 1
                    diag_fz_sum[fi] += fz_np[ei, fi]
                    if rv_np[ei, fi]:
                        diag_rv_count[fi] += 1
                        if np.isfinite(gz_np[ei, fi]):
                            diag_gz_sum[fi] += gz_np[ei, fi]
                            diag_gz_n[fi] += 1
                    if sn_np[ei, fi] and rv_np[ei, fi] and np.isfinite(clr_np[ei, fi]):
                        diag_clr_sum[fi] += clr_np[ei, fi]
                        diag_clr_n[fi] += 1

        if (step + 1) % 200 == 0:
            print(f"  step {step + 1}/{T}  ({time.time() - t_start:.1f}s)")

    total_time = time.time() - t_start
    print(f"[{SCRIPT_VERSION}] Collection done in {total_time:.1f}s\n")

    # ── Per-foot diagnostic report ─────────────────────────────────────────────
    diag_lines: list[str] = []
    diag_lines.append(f"[{SCRIPT_VERSION}] === PER-FOOT DIAGNOSTIC (main collection) ===")
    hdr = (
        f"  {'foot':<12} {'ray_valid%':>11} {'mean_foot_z':>13} "
        f"{'mean_gnd_z':>11} {'mean_clr(planted)':>19} {'cal_n':>7}"
    )
    diag_lines.append(hdr)
    diag_lines.append("  " + "-" * 80)
    for fi, fn in enumerate(foot_names):
        rv_pct = 100.0 * diag_rv_count[fi] / max(diag_total[fi], 1)
        mfz = diag_fz_sum[fi] / max(diag_total[fi], 1)
        mgz = diag_gz_sum[fi] / max(diag_gz_n[fi], 1) if diag_gz_n[fi] > 0 else float("nan")
        mclr = diag_clr_sum[fi] / max(diag_clr_n[fi], 1) if diag_clr_n[fi] > 0 else float("nan")
        n_cal = diag_clr_n[fi]
        line = f"  {fn:<12} {rv_pct:>10.1f}% {mfz:>13.3f}m {mgz:>11.3f}m {mclr:>19.4f}m {n_cal:>7}"
        diag_lines.append(line)

    diag_lines.append("")
    diag_lines.append("  INTERPRETATION:")
    diag_lines.append("  - ray_valid% low (~0) → scanner does NOT cover this foot → nearest-ray")
    diag_lines.append("    distance too large → ray_valid=False → kin always False.")
    diag_lines.append("  - ray_valid% high but mean_clr >> sole_offset → clearance estimate wrong.")
    diag_lines.append("  - ray_valid% high and mean_clr ≈ sole_offset → scanner working correctly.")
    diag_str = "\n".join(diag_lines)
    print(diag_str)
    print()

    # ── Analysis ──────────────────────────────────────────────────────────────
    S = arr_sensor.reshape(-1, 4)
    K = arr_kin.reshape(-1, 4)
    C = arr_class.reshape(-1)
    print(f"[{SCRIPT_VERSION}] Total samples: {S.shape[0]} ({T} steps × {N} envs)\n")

    def _compute_metrics(sensor: np.ndarray, kin: np.ndarray) -> dict[str, np.ndarray]:
        if sensor.shape[0] == 0:
            nan4 = np.full(4, np.nan)
            return {
                k: nan4
                for k in (
                    "sensor_ratio",
                    "kin_ratio",
                    "kin_only_ratio",
                    "sensor_only_ratio",
                    "both_ratio",
                    "neither_ratio",
                )
            }
        return {
            "sensor_ratio": sensor.mean(axis=0),
            "kin_ratio": kin.mean(axis=0),
            "kin_only_ratio": ((~sensor) & kin).mean(axis=0),
            "sensor_only_ratio": (sensor & (~kin)).mean(axis=0),
            "both_ratio": (sensor & kin).mean(axis=0),
            "neither_ratio": ((~sensor) & (~kin)).mean(axis=0),
        }

    overall = _compute_metrics(S, K)
    overall_table = _print_table(foot_names, overall, "=== OVERALL per-foot contact metrics ===")
    print(overall_table)
    print()

    # Integrity check
    print(f"[{SCRIPT_VERSION}] Integrity check — both_ratio (expect >0.3 for active feet):")
    for fi, fn in enumerate(foot_names):
        sr = overall["sensor_ratio"][fi]
        bt = overall["both_ratio"][fi]
        rv_pct = 100.0 * diag_rv_count[fi] / max(diag_total[fi], 1)
        flag = ""
        if sr > 0.3 and bt < 0.1:
            if rv_pct < 50.0:
                flag = " *** scanner coverage issue (ray_valid% low)"
            else:
                flag = " *** kin broken despite valid rays — check clearance/offset"
        print(f"  {fn}: sensor={sr:.3f}  both={bt:.3f}  ray_valid={rv_pct:.1f}%{flag}")
    print()

    # Per-terrain
    active_classes = sorted(set(C.tolist()))
    terrain_tables: list[str] = []
    per_terrain: dict[str, dict[str, np.ndarray]] = {}
    for cls_id in active_classes:
        mask = cls_id == C
        cls_name = _TERRAIN_CLASS_NAMES.get(int(cls_id), f"class_{cls_id}")
        cls_m = _compute_metrics(S[mask], K[mask])
        per_terrain[cls_name] = cls_m
        tbl = _print_table(foot_names, cls_m, f"--- Terrain: {cls_name} (class {cls_id})  n={int(mask.sum())} ---")
        terrain_tables.append(tbl)
        print(tbl)
        print()

    flat_kin_only = np.zeros(4)
    if "flat" in per_terrain:
        flat_kin_only = per_terrain["flat"].get("kin_only_ratio", np.zeros(4))
        print(
            "[verify_3leg] Flat terrain kin_only (noise floor): "
            + ", ".join(f"{foot_names[i]}={flat_kin_only[i]:.4f}" for i in range(4))
        )
        print()

    # Flags
    non_flat = [k for k in per_terrain if k != "flat"]
    flag_lines: list[str] = []
    suspect_feet: list[str] = []
    if hs_valid:
        for cls_name in non_flat:
            cls_ko = per_terrain[cls_name]["kin_only_ratio"]
            for fi, fn in enumerate(foot_names):
                excess = float(cls_ko[fi]) - float(flat_kin_only[fi])
                if excess > args_cli.kin_only_flag_thresh:
                    flag_lines.append(
                        f"  *** edge-underreporting 의심: foot={fn}, terrain={cls_name}, "
                        f"kin_only={cls_ko[fi]:.4f}, noise_floor={flat_kin_only[fi]:.4f}, "
                        f"excess={excess:.4f}"
                    )
                    if fn not in suspect_feet:
                        suspect_feet.append(fn)
    else:
        flag_lines.append("  [SKIP] hs_valid=False — kinematic contact not computed.")

    print("=== FLAGS ===")
    if suspect_feet:
        print(f"  edge-underreporting 의심 발: {suspect_feet}")
        for ln in flag_lines:
            print(ln)
        print("\n  -> 권장: contact 대신 운동학 신호(foot_z-ground_z) 사용.\n")
    else:
        if hs_valid:
            print(
                f"  kin_only excess < {args_cli.kin_only_flag_thresh} 모든 발 → "
                "edge-underreporting 근거 약함. 3-leg gait 원인은 reward/policy 쪽."
            )
        print()

    # Asymmetry + cross-verdict
    print("=== per-foot sensor_ratio ASYMMETRY ===")
    sr = overall["sensor_ratio"]
    sr_mean = sr.mean()
    sr_std = sr.std()
    for fi, fn in enumerate(foot_names):
        print(f"  {fn}: sensor_ratio={sr[fi]:.4f}  (mean={sr_mean:.4f}, dev={sr[fi] - sr_mean:+.4f})")

    verdict_text = ""
    if sr_std > 0.05:
        ui = int(np.argmin(sr))
        un = foot_names[ui]
        print(f"\n  -> 비대칭 감지 (std={sr_std:.4f}). 가장 덜 딛는 발: {un} (sensor={sr[ui]:.4f})\n")

        print("=== CROSS-VERDICT ===")
        if hs_valid:
            kr = overall["kin_ratio"]
            ko = overall["kin_only_ratio"]
            nf = float(flat_kin_only[ui]) if "flat" in per_terrain else 0.0
            exc = float(ko[ui]) - nf
            rv_pct_u = 100.0 * diag_rv_count[ui] / max(diag_total[ui], 1)
            print(f"  Underused foot : {un}")
            print(f"    sensor_ratio   = {sr[ui]:.4f}")
            print(f"    kin_ratio      = {kr[ui]:.4f}")
            print(f"    kin_only_ratio = {ko[ui]:.4f}")
            print(f"    noise_floor    = {nf:.4f}  (flat kin_only)")
            print(f"    excess         = {exc:.4f}")
            print(f"    ray_valid%     = {rv_pct_u:.1f}%")

            if rv_pct_u < 50.0:
                verdict_text = (
                    f"**VERDICT**: {un}의 ray_valid={rv_pct_u:.1f}% — "
                    "scanner가 이 발을 거의 커버하지 못함. kin 신호 무효. "
                    "kin_ratio/kin_only 수치는 신뢰할 수 없음 (scanner coverage 문제). "
                    "ground_z 대안 필요 (terrain heightmap 직접 조회 또는 발별 별도 ray cast)."
                )
                print(f"\n  VERDICT: {un}의 ray_valid%={rv_pct_u:.1f}% — scanner coverage 부족. kin 수치 무효.")
                print("  ground_z 대안 필요 전까지 sensor_ratio 비대칭만으로 진단 가능.")
            elif kr[ui] < sr_mean - 0.05:
                verdict_text = (
                    f"**VERDICT**: {un}의 kin_ratio({kr[ui]:.4f})도 낮음 → "
                    "정책이 이 발을 실제로 공중에 유지. edge-underreporting 근거 없음. "
                    "원인: reward shaping / gait pairing 부재."
                )
                print(f"\n  VERDICT: {un}의 kin_ratio도 낮음 → kinematic choice. edge-underreporting 근거 없음.")
                print("  권장: feet_air_time 상한 패널티 + gait_pairing 재활성.")
            elif exc > args_cli.kin_only_flag_thresh:
                verdict_text = (
                    f"**VERDICT**: {un}의 kin_ratio 높고 sensor_ratio 낮음 → "
                    f"edge-underreporting 지지 (kin_only excess={exc:.4f}). "
                    "순위 1/2 fix를 운동학 신호로 구현 권장."
                )
                print(f"\n  VERDICT: edge-underreporting 지지. kin_only excess={exc:.4f}.")
                print("  권장: 순위 1/2 fix를 contact 대신 운동학 신호로 구현.")
            else:
                verdict_text = "**VERDICT**: 혼재 신호 — 추가 측정 필요."
                print("\n  VERDICT: 혼재 신호. 추가 스텝/envs 권장.")
        else:
            verdict_text = "height_scanner 비활성 → kinematic 비교 불가."
            print("  height_scanner 비활성 → sensor_ratio 비대칭만 확인됨.")
        print()
    else:
        verdict_text = "sensor_ratio 대칭적 — 3-leg gait 없음."
        print(f"\n  -> 대칭적 접지 (std={sr_std:.4f} ≤ 0.05)\n")

    # ── Save NPZ (sole_offset 포함, no fallback path) ──────────────────────────
    out_dir = "/home/lgb/IsaacLab/_workspace/3leg_gait_research"
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "rank0_verify_result.npz")
    np.savez_compressed(
        npz_path,
        version=np.array([SCRIPT_VERSION]),
        arr_sensor=arr_sensor.astype(np.uint8),
        arr_kin=arr_kin.astype(np.uint8),
        arr_class=arr_class,
        foot_names=np.array(foot_names),
        sole_offset=sole_offset,  # present in v3
        ray_valid_ratio=(diag_rv_count / np.maximum(diag_total, 1)).astype(np.float32),
        sensor_ratio=overall["sensor_ratio"],
        kin_ratio=overall["kin_ratio"],
        kin_only_ratio=overall["kin_only_ratio"],
        sensor_only_ratio=overall["sensor_only_ratio"],
        both_ratio=overall["both_ratio"],
        neither_ratio=overall["neither_ratio"],
        force_thresh=np.float32(args_cli.force_thresh),
        z_thresh=np.float32(args_cli.z_thresh),
        v_thresh=np.float32(args_cli.v_thresh),
        num_steps=np.int32(T),
        num_envs=np.int32(N),
    )
    print(f"[{SCRIPT_VERSION}] NPZ saved  : {npz_path}  (keys include version, sole_offset, ray_valid_ratio)")

    # ── Markdown report ────────────────────────────────────────────────────────
    md_path = os.path.join(out_dir, "rank0_verify_result.md")

    # sole_offset table
    sole_rows = ["| foot | sole_offset (m) | cal_n |", "|------|----------------|-------|"]
    for fi, fn in enumerate(foot_names):
        n_c = len(cal_clr[fi]) if hs_valid else 0
        sole_rows.append(f"| {fn} | {sole_offset[fi]:.4f} | {n_c} |")

    # ray_valid table for md
    rv_rows = [
        "| foot | ray_valid% (main) | mean_clearance_planted (m) |",
        "|------|------------------|---------------------------|",
    ]
    for fi, fn in enumerate(foot_names):
        rv_pct = 100.0 * diag_rv_count[fi] / max(diag_total[fi], 1)
        mclr = diag_clr_sum[fi] / max(diag_clr_n[fi], 1) if diag_clr_n[fi] > 0 else float("nan")
        rv_rows.append(f"| {fn} | {rv_pct:.1f}% | {mclr:.4f} |")

    report = [
        f"# 3-Leg Gait 순위 0 검증 결과 ({SCRIPT_VERSION} — height_scanner ground_z)",
        "",
        f"- 체크포인트: `{resume_path}`",
        f"- 실행 시각: {ts}",
        f"- num_steps: {T}  num_envs: {N}  cal_steps: {CAL}",
        f"- force_thresh: {args_cli.force_thresh} N  z_thresh: {args_cli.z_thresh} m  "
        f"v_thresh: {args_cli.v_thresh} m/s",
        "- ground_z 소스: height_scanner ray_hits_w (nearest finite ray)",
        "",
        "## Scanner Coverage Diagnostic",
        "",
        "```",
        "height_scanner cfg: offset_x=+0.375m, size=[1.6, 1.0]m",
        "scanner grid (body-relative x): [-0.425, 1.175] m",
        "scanner grid (body-relative y): [-0.500, 0.500] m",
        "(실제 bbox는 stdout 참조)",
        "```",
        "",
        "## Ray Valid Ratio (발별 scanner 커버리지)",
        "",
        *rv_rows,
        "",
        "**ray_valid% 해석**: 낮으면 scanner가 해당 발을 커버하지 못해 kin 신호 무효.",
        "",
        "## Sole Offset (자동 추정)",
        "",
        *sole_rows,
        "",
        "## Overall per-foot metrics",
        "",
        "```",
        overall_table,
        "```",
        "",
        "## Per-terrain breakdown",
        "",
    ]
    for tbl in terrain_tables:
        report += ["```", tbl, "```", ""]

    report += ["## Flags", ""]
    if suspect_feet:
        report.append(f"**edge-underreporting 의심 발: {suspect_feet}**")
        for ln in flag_lines:
            report.append(ln)
        report.append("\n**권장**: 운동학 신호(foot_z-ground_z) 사용.")
    else:
        report.append("edge-underreporting 근거 약함.")

    report += [
        "",
        "## 3-leg gait asymmetry",
        "",
    ]
    for fi, fn in enumerate(foot_names):
        report.append(
            f"- {fn}: sensor_ratio={sr[fi]:.4f}  "
            f"kin_ratio={overall['kin_ratio'][fi]:.4f}  "
            f"both_ratio={overall['both_ratio'][fi]:.4f}"
        )

    report += ["", "## Cross-verdict", "", verdict_text]

    with open(md_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"[{SCRIPT_VERSION}] MD  saved  : {md_path}")

    ts_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"=== verify_3leg {SCRIPT_VERSION} DONE @ {ts_end} ===")
    print(f"{'=' * 60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
