# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Front- vs hind-foot landing-accuracy diagnostic for the 50k Teacher3D parkour policy. v2

Hypothesis under test
---------------------
The teacher's 3D clearance sensor casts rays only into the FORWARD hemisphere
(azimuth ±100°), so the ground *under / behind the hind legs* is absent from the
privileged terrain GT. If that matters, the HIND feet should land less accurately
than the FRONT feet on DIFFICULT terrain, while FLAT (a control where forward-only
sensing is irrelevant) should show front ≈ hind.

v2 changes vs v1
-----------------
- terrain_h source: raycast (default) instead of height_field lookup.
  ray_start = (foot_x, foot_y, foot_z + 0.03), dir = (0,0,-1),
  mesh = height_scanner.meshes[first_key] (combined warp mesh built from /World/ground).
  ray_start EPS=0.03 m above foot_z: at landing, foot_z ≈ floor + 0.023 m (foot-sphere radius),
  so start ≈ floor + 0.053 m — well below crawl ceiling (~0.4 m clearance). First downward hit = floor.
  Misses (inf) → event excluded.
  `--floor_source heightfield` flag restores v1 behaviour for comparison.
- median is primary stat; mean is secondary.
- (hind-front) gap: median_gap primary, mean_gap secondary.
- impact peakFz table: baseline-subtracted column: (hind-front delta for terrain) - (hind-front delta for flat).
  (flat already shows hind +6.6 N rear-heavy structural bias; this removes it.)
- Crawl sanity block: reports |placement_error|>0.5 m fraction for raycast vs heightfield.
- No auto-verdict / re-training justification text. Numbers only.

Per-foot landing = individual foot airborne→contact transition (NOT all-4-airborne pronk).

Index discipline (DO NOT cross-use)
-----------------------------------
  position : _amp_foot_body_ids (articulation), _robot.data.body_pos_w[:, ids, :] → (N,4,3)
  contact  : _feet_ids (contact-sensor), _contact_sensor.data.net_forces_w_history[:,:,ids,:]
Both reordered to canonical [FL,FR,RL,RR] BY NAME. Foot k = same physical foot in both.

Mesh handle
-----------
  base_env._height_scanner.meshes  is a dict {mesh_prim_path_key: wp.Mesh}.
  The height_scanner cfg has mesh_prim_paths=["/World/ground"] → key = "/World/ground".
  We use the first value (robust to potential key variation) for raycast.

raycast_mesh API (isaaclab.utils.warp.ops)
------------------------------------------
  raycast_mesh(ray_starts: Tensor(N,3), ray_directions: Tensor(N,3), mesh: wp.Mesh,
               max_dist: float) -> (ray_hits: Tensor(N,3), None, None, None)
  Misses → ray_hits = inf. Input must be contiguous float32 on mesh.device.

Run (headless, GPU 2 — coordinator will run on GPU 2):
  cd /home/lgb/IsaacLab && CUDA_VISIBLE_DEVICES=2 conda run -n isaac-5.1 --no-capture-output \\
      python scripts/reinforcement_learning/rsl_rl/measure_hind_foot_placement.py \\
      --task Go2-ParkourImitation-Teacher3D-v0 --num_envs 1024 --num_steps 1500 --headless \\
      --checkpoint logs/rsl_rl/parkour_imitation_go2_teacher3d/2026-06-26_18-04-01_teacher3d_resume50k/model_49999.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Front vs hind foot landing-accuracy diagnostic (v2).")
parser.add_argument("--task", type=str, default="Go2-ParkourImitation-Teacher3D-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_steps", type=int, default=1500)
# --checkpoint provided by cli_args.add_rsl_rl_args()
parser.add_argument(
    "--force_thresh", type=float, default=2.0, help="Contact force threshold [N] (default 2.0, matches env reward)"
)
parser.add_argument(
    "--spawn_transient",
    type=int,
    default=15,
    help="Drop first N steps of each episode (spawn settle).",
)
parser.add_argument(
    "--max_init_level",
    type=int,
    default=9,
    help="Force max_init_terrain_level to sample hard rows.",
)
parser.add_argument(
    "--floor_source",
    type=str,
    default="raycast",
    choices=["raycast", "heightfield"],
    help="terrain_h source: 'raycast' (v2, default, fixes crawl artefacts) or 'heightfield' (v1).",
)
parser.add_argument(
    "--ray_eps",
    type=float,
    default=0.03,
    help="Ray-start offset above foot_z [m] (default 0.03). foot_z at landing ≈ floor+0.023 m.",
)
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
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerParkour
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP, OnPolicyRunnerAMPBase
from rsl_rl.runners.on_policy_runner_parkour_amp import OnPolicyRunnerParkourAMP, OnPolicyRunnerParkourAMPVoxel

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.warp import raycast_mesh

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

SCRIPT_VERSION = "hind_foot_v2"
OUT_DIR = "/home/lgb/IsaacLab/_workspace/parkour_imitation_lidar/hind_foot_check"
CANONICAL = ["FL", "FR", "RL", "RR"]  # canonical foot order; front={0,1}, hind={2,3}


def _build_runner(agent_cfg, env, device):
    """Mirror play.py's class_name branch (no onnx/jit export)."""
    cn = agent_cfg.class_name
    if cn == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    if cn == "OnPolicyRunnerParkour":
        return OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=device)
    if cn == "OnPolicyRunnerAMP":
        return OnPolicyRunnerAMP(env, agent_cfg.to_dict(), log_dir=None, device=device)
    if cn == "OnPolicyRunnerAMPBase":
        return OnPolicyRunnerAMPBase(env, agent_cfg.to_dict(), log_dir=None, device=device)
    if cn == "OnPolicyRunnerParkourAMP":
        return OnPolicyRunnerParkourAMP(env, agent_cfg.to_dict(), log_dir=None, device=device)
    if cn == "OnPolicyRunnerParkourAMPVoxel":
        return OnPolicyRunnerParkourAMPVoxel(env, agent_cfg.to_dict(), log_dir=None, device=device)
    raise ValueError(f"Unsupported runner class: {cn}")


def _canonical_reorder(names_list, ids_list, tag):
    """Reorder so result[k] picks the entry whose name matches CANONICAL[k] = {FL,FR,RL,RR}.
    Matches by substring 'FL'/'FR'/'RL'/'RR' + 'FOOT' (case-insensitive).
    """
    order = []
    for canon in CANONICAL:
        matched = None
        for local_i, nm in enumerate(names_list):
            up = nm.upper()
            if canon in up and "FOOT" in up:
                matched = local_i
                break
        assert matched is not None, f"[{tag}] could not find canonical foot '{canon}' in {names_list}"
        order.append(matched)
    assert len(set(order)) == 4, f"[{tag}] non-unique canonical mapping: order={order} names={names_list}"
    return order


# ── main ──────────────────────────────────────────────────────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 64}")
    print(f"=== measure_hind_foot_placement {SCRIPT_VERSION} ===")
    print(f"=== floor_source={args_cli.floor_source}  ray_eps={args_cli.ray_eps} ===")
    print(f"=== started {ts} ===")
    print(f"{'=' * 64}\n")

    # ── Environment setup ──────────────────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = False
    for attr in ("debug_vis_edge_mask", "enable_keyboard_view_switch"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    # Force hard terrain rows for difficult-terrain sample coverage.
    if hasattr(env_cfg, "terrain") and hasattr(env_cfg.terrain, "max_init_terrain_level"):
        old = env_cfg.terrain.max_init_terrain_level
        env_cfg.terrain.max_init_terrain_level = args_cli.max_init_level
        print(f"[{SCRIPT_VERSION}] max_init_terrain_level: {old} -> {args_cli.max_init_level}")

    # Checkpoint
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[{SCRIPT_VERSION}] Task       : {args_cli.task}")
    print(f"[{SCRIPT_VERSION}] Checkpoint : {resume_path}")
    print(f"[{SCRIPT_VERSION}] Runner cls : {agent_cfg.class_name}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = _build_runner(agent_cfg, env, agent_cfg.device)
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

    # ── Foot index canonicalization (position vs contact, by name) ─────────────
    pos_ids_raw = list(base_env._amp_foot_body_ids)  # articulation order
    pos_names_raw = [base_env._robot.data.body_names[i] for i in pos_ids_raw]
    contact_ids_raw = list(base_env._feet_ids)  # contact-sensor order
    contact_names_raw = [base_env._contact_sensor.body_names[i] for i in contact_ids_raw]

    pos_order = _canonical_reorder(pos_names_raw, pos_ids_raw, "position")
    contact_order = _canonical_reorder(contact_names_raw, contact_ids_raw, "contact")

    # canonical id tensors: index k -> physical foot CANONICAL[k]
    foot_pos_ids = torch.tensor([pos_ids_raw[i] for i in pos_order], device=device, dtype=torch.long)
    foot_contact_ids = torch.tensor([contact_ids_raw[i] for i in contact_order], device=device, dtype=torch.long)

    print(f"\n[{SCRIPT_VERSION}] ── FOOT CANONICALIZATION (k -> physical foot) ──")
    for k in range(4):
        print(
            f"  k={k} ({CANONICAL[k]:<2})  position: id={int(foot_pos_ids[k])}"
            f" name='{pos_names_raw[pos_order[k]]}'"
            f"   contact: id={int(foot_contact_ids[k])}"
            f" name='{contact_names_raw[contact_order[k]]}'"
        )
    print(f"  front = k{{0,1}} = {{FL,FR}}   hind = k{{2,3}} = {{RL,RR}}\n")

    # ── Terrain height sources ─────────────────────────────────────────────────
    # Height-field (v1, kept for roughness and comparison).
    hf = base_env._edge_mask_height_field  # (H, W) float32 world metres
    hf_origin = base_env._edge_mask_origin  # (2,) world-frame lower-left corner (x, y)
    hf_inv_scale = base_env._edge_mask_inv_scale  # cells / metre (scalar)
    H, W = hf.shape
    print(f"[{SCRIPT_VERSION}] height_field shape=({H},{W}) origin={hf_origin.cpu().numpy()} inv_scale={hf_inv_scale:.4f}")

    # Pre-pad hf for safe 3×3 window std (replicate edges).
    hf_pad = torch.nn.functional.pad(hf[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]  # (H+2, W+2)

    # Warp mesh for raycast (v2).
    # _height_scanner.meshes is a dict; key = first mesh_prim_path = "/World/ground".
    # We use the first value so we're robust to minor key differences.
    _hs_meshes = base_env._height_scanner.meshes
    assert len(_hs_meshes) > 0, "[v2] _height_scanner.meshes is empty — terrain mesh not built yet?"
    _warp_mesh_key = next(iter(_hs_meshes))
    warp_mesh = _hs_meshes[_warp_mesh_key]
    print(f"[{SCRIPT_VERSION}] warp mesh key='{_warp_mesh_key}'  device='{warp_mesh.device}'")

    def _hf_lookup(foot_xy: torch.Tensor):
        """(N,4,2) -> (terrain_h (N,4), roughness (N,4)). Mirrors feet_edge formula (round-0.5)."""
        grid = ((foot_xy - hf_origin) * hf_inv_scale - 0.5).round().long()  # (N,4,2)
        gx = grid[..., 0].clamp(0, H - 1)
        gy = grid[..., 1].clamp(0, W - 1)
        th = hf[gx, gy]  # (N,4)
        # 3×3 std using padded field.
        gxp, gyp = gx + 1, gy + 1
        windows = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                windows.append(hf_pad[gxp + dx, gyp + dy])  # (N,4)
        rough = torch.stack(windows, dim=-1).std(dim=-1)  # (N,4)
        return th, rough

    # ── Terrain class names ────────────────────────────────────────────────────
    from isaaclab_tasks.direct.parkour.parkour_env import _TERRAIN_CLASS_NAMES

    # ── Allocate streaming buffers (CPU numpy) ─────────────────────────────────
    arr_class = np.zeros((T, N), dtype=np.int64)
    arr_level = np.zeros((T, N), dtype=np.int16)
    arr_foot_contact = np.zeros((T, N, 4), dtype=bool)
    arr_foot_xyz = np.zeros((T, N, 4, 3), dtype=np.float32)  # foot world position (x,y,z)
    arr_terrain_h_hf = np.zeros((T, N, 4), dtype=np.float32)  # height_field-based terrain height
    arr_rough = np.zeros((T, N, 4), dtype=np.float32)  # 3×3 hf std (roughness)
    arr_peakfz = np.zeros((T, N, 4), dtype=np.float32)  # per-foot peak |Fz| over contact history
    arr_eplen = np.zeros((T, N), dtype=np.int32)
    arr_dones = np.zeros((T, N), dtype=bool)
    arr_c_tilt = np.zeros((T, N), dtype=bool)
    arr_c_low = np.zeros((T, N), dtype=bool)
    arr_c_base = np.zeros((T, N), dtype=bool)
    arr_c_goal = np.zeros((T, N), dtype=bool)

    obs = env.get_observations()

    print(f"[{SCRIPT_VERSION}] Collecting {T} steps x {N} envs (NATIVE policy) ...")
    t_start = time.time()
    for step in range(T):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # contact forces (canonical contact order)
        net_cf = base_env._contact_sensor.data.net_forces_w_history  # (N, hist, bodies, 3)
        feet_cf_hist = net_cf[:, :, foot_contact_ids, :]  # (N, hist, 4, 3)
        peak_fz = feet_cf_hist[..., 2].abs().max(dim=1).values  # (N, 4)
        contact = torch.norm(net_cf[:, 0, foot_contact_ids], dim=-1) > args_cli.force_thresh  # (N, 4)

        # foot positions (canonical position order)
        foot_pos_w = base_env._robot.data.body_pos_w[:, foot_pos_ids, :]  # (N, 4, 3)
        foot_xy = foot_pos_w[..., :2]  # (N, 4, 2)
        terrain_h_hf, rough = _hf_lookup(foot_xy)

        env_class = base_env._env_class
        level = base_env._terrain_levels
        ep_len = base_env.episode_length_buf

        arr_class[step] = env_class.cpu().numpy()
        arr_level[step] = level.cpu().numpy().astype(np.int16)
        arr_foot_contact[step] = contact.cpu().numpy()
        arr_foot_xyz[step] = foot_pos_w.cpu().numpy()
        arr_terrain_h_hf[step] = terrain_h_hf.cpu().numpy()
        arr_rough[step] = rough.cpu().numpy()
        arr_peakfz[step] = peak_fz.cpu().numpy()
        arr_eplen[step] = ep_len.cpu().numpy().astype(np.int32)
        arr_dones[step] = dones.cpu().numpy().astype(bool)
        arr_c_tilt[step] = base_env._term_tilt.cpu().numpy()
        arr_c_low[step] = base_env._term_low_height.cpu().numpy()
        arr_c_base[step] = base_env._term_base_contact.cpu().numpy()
        arr_c_goal[step] = base_env._term_goal_reached.cpu().numpy()

        if (step + 1) % 250 == 0:
            print(f"  step {step + 1}/{T}  ({time.time() - t_start:.1f}s)")

    total_time = time.time() - t_start
    print(f"[{SCRIPT_VERSION}] Collection done in {total_time:.1f}s\n")

    # ════════════════════════════════════════════════════════════════════════════
    #  POST-PROCESSING
    # ════════════════════════════════════════════════════════════════════════════

    # (a) valid mask: failure-terminated episodes dropped; spawn transient dropped.
    valid = np.ones((T, N), dtype=bool)
    n_fail = n_total = n_goal = 0
    for n in range(N):
        done_steps = np.nonzero(arr_dones[:, n])[0]
        seg_start = 0
        for ds in done_steps:
            n_total += 1
            is_fail = bool(arr_c_tilt[ds, n] or arr_c_low[ds, n] or arr_c_base[ds, n])
            is_goal = bool(arr_c_goal[ds, n])
            if is_goal:
                n_goal += 1
            if is_fail and not is_goal:
                n_fail += 1
                valid[seg_start : ds + 1, n] = False
            seg_start = ds + 1
    valid &= arr_eplen > args_cli.spawn_transient

    print(f"[{SCRIPT_VERSION}] episodes: total={n_total} failure={n_fail} goal={n_goal}")
    if n_total > 0:
        print(
            f"[{SCRIPT_VERSION}]   failure_rate={100.0 * n_fail / n_total:.1f}%"
            f"  goal_rate={100.0 * n_goal / n_total:.1f}%"
        )
    print(f"[{SCRIPT_VERSION}] valid sample fraction = {valid.mean():.3f}")
    lvl_start = arr_level[: max(1, T // 20)].mean()
    lvl_end = arr_level[-max(1, T // 20) :].mean()
    print(f"[{SCRIPT_VERSION}] mean terrain_level: first5%={lvl_start:.2f}  last5%={lvl_end:.2f}\n")

    # (b) per-foot landing events: airborne→contact per foot, gated by valid.
    prev_contact = np.zeros_like(arr_foot_contact)
    prev_contact[1:] = arr_foot_contact[:-1]
    landing = (~prev_contact) & arr_foot_contact & valid[:, :, None]  # (T,N,4)

    # (c) RAYCAST terrain height at landing positions (v2).
    # We do this post-hoc: gather all landing (t,n,k) positions, batch-raycast downward.
    ev_t, ev_n, ev_k = np.nonzero(landing)
    n_events_raw = ev_t.size
    print(f"[{SCRIPT_VERSION}] raw landing events (pre-NaN filter) = {n_events_raw}")

    ev_foot_xyz = arr_foot_xyz[ev_t, ev_n, ev_k]  # (M, 3) float32
    ev_foot_xyz_hf_th = arr_terrain_h_hf[ev_t, ev_n, ev_k]  # (M,) height_field terrain h
    ev_peakfz = arr_peakfz[ev_t, ev_n, ev_k]  # (M,)
    ev_rough = arr_rough[ev_t, ev_n, ev_k]  # (M,)
    ev_class = arr_class[ev_t, ev_n]  # (M,)
    ev_level = arr_level[ev_t, ev_n]  # (M,)

    # Compute terrain height with selected floor_source.
    if args_cli.floor_source == "raycast" and n_events_raw > 0:
        # ray_start: foot_xyz + (0, 0, ray_eps), direction: (0, 0, -1)
        ray_starts = torch.tensor(ev_foot_xyz, device=device, dtype=torch.float32)
        ray_starts[:, 2] += args_cli.ray_eps  # shift above foot sphere by EPS
        ray_dirs = torch.zeros_like(ray_starts)
        ray_dirs[:, 2] = -1.0  # downward

        # raycast_mesh expects (N,3) flat tensors; returns (ray_hits (N,3), None, None, None)
        # mesh.device may be a string like "cuda:0"; ensure inputs are on same device.
        import warp as wp
        _mesh_torch_dev = wp.device_to_torch(warp_mesh.device)
        ray_starts_d = ray_starts.to(_mesh_torch_dev).contiguous()
        ray_dirs_d = ray_dirs.to(_mesh_torch_dev).contiguous()

        with torch.inference_mode():
            ray_hits, _, _, _ = raycast_mesh(
                ray_starts_d,
                ray_dirs_d,
                warp_mesh,
                max_dist=2.0,
            )
        ray_hits = ray_hits.to(device)  # (M, 3) — misses = inf

        ev_terrain_h_rc = ray_hits[:, 2].cpu().numpy()  # (M,) — floor Z from downward hit
        ev_terrain_h = ev_terrain_h_rc
        floor_source_label = f"raycast(eps={args_cli.ray_eps}m,max_dist=2.0m)"
        print(
            f"[{SCRIPT_VERSION}] raycast done: hits_finite={np.isfinite(ev_terrain_h_rc).sum()}/{n_events_raw}"
        )
    else:
        ev_terrain_h = ev_foot_xyz_hf_th
        floor_source_label = "heightfield"
        if n_events_raw == 0:
            print(f"[{SCRIPT_VERSION}] WARNING: 0 landing events — nothing to raycast.")

    # (d) placement_error = foot_z - terrain_h; NaN mask.
    ev_foot_z = ev_foot_xyz[:, 2]  # (M,)
    ev_placement_err = ev_foot_z - ev_terrain_h  # (M,)
    # Also compute height_field-based error for crawl sanity comparison.
    ev_placement_err_hf = ev_foot_z - ev_foot_xyz_hf_th  # (M,) always available

    finite_mask = (
        np.isfinite(ev_placement_err)
        & np.isfinite(ev_peakfz)
        & np.isfinite(ev_rough)
        & np.isfinite(ev_foot_z)
    )
    ev_ishind = ev_k >= 2  # RL=2, RR=3

    # Print crawl sanity (artefact rate before vs after).
    crawl_id = next((c for c, nm in _TERRAIN_CLASS_NAMES.items() if nm == "crawl"), None)
    if crawl_id is not None:
        crawl_m = ev_class == crawl_id
        n_crawl = int(crawl_m.sum())
        if n_crawl > 0:
            big_hf = float((np.abs(ev_placement_err_hf[crawl_m]) > 0.5).mean())
            if args_cli.floor_source == "raycast":
                big_rc = float((np.abs(ev_placement_err[crawl_m]) > 0.5).mean())
                print(
                    f"\n[{SCRIPT_VERSION}] CRAWL SANITY (|placement_err|>0.5m artefact rate):"
                    f"\n  heightfield: {big_hf:.3f} ({100 * big_hf:.1f}%)"
                    f"\n  raycast    : {big_rc:.3f} ({100 * big_rc:.1f}%)"
                    f"  (before/after artefact resolution)  n_crawl={n_crawl}"
                )
            else:
                print(
                    f"\n[{SCRIPT_VERSION}] CRAWL SANITY (|placement_err|>0.5m, heightfield):"
                    f" {big_hf:.3f}  n_crawl={n_crawl}"
                )
        else:
            print(f"\n[{SCRIPT_VERSION}] no crawl landing events to report.\n")

    # Apply finite mask.
    ev_t = ev_t[finite_mask]
    ev_n = ev_n[finite_mask]
    ev_k = ev_k[finite_mask]
    ev_class = ev_class[finite_mask]
    ev_level = ev_level[finite_mask]
    ev_placement_err = ev_placement_err[finite_mask]
    ev_peakfz = ev_peakfz[finite_mask]
    ev_rough = ev_rough[finite_mask]
    ev_ishind = ev_ishind[finite_mask]

    print(f"\n[{SCRIPT_VERSION}] valid+finite per-foot landing events = {ev_placement_err.size}")
    flat_id = next((c for c, nm in _TERRAIN_CLASS_NAMES.items() if nm == "flat"), 0)
    n_flat = int((ev_class == flat_id).sum())
    print(f"[{SCRIPT_VERSION}] flat landings = {n_flat}  (control group)")

    # ── Save NPZ ──
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, "hind_foot_placement_v2.npz")
    np.savez_compressed(
        npz_path,
        version=np.array([SCRIPT_VERSION]),
        checkpoint=np.array([resume_path]),
        task=np.array([args_cli.task]),
        floor_source=np.array([floor_source_label]),
        canonical=np.array(CANONICAL),
        pos_names=np.array([pos_names_raw[i] for i in pos_order]),
        contact_names=np.array([contact_names_raw[i] for i in contact_order]),
        ev_class=ev_class,
        ev_level=ev_level,
        ev_foot_k=ev_k.astype(np.int8),
        ev_is_hind=ev_ishind,
        ev_placement_err=ev_placement_err.astype(np.float32),
        ev_peak_fz=ev_peakfz.astype(np.float32),
        ev_roughness=ev_rough.astype(np.float32),
        n_total_episodes=np.int32(n_total),
        n_fail_episodes=np.int32(n_fail),
        n_goal_episodes=np.int32(n_goal),
        num_steps=np.int32(T),
        num_envs=np.int32(N),
        seed=np.int32(args_cli.seed),
        force_thresh=np.float32(args_cli.force_thresh),
    )
    print(f"\n[{SCRIPT_VERSION}] NPZ saved : {npz_path}")

    # ── Build stdout summary tables ────────────────────────────────────────────
    class_ids = sorted({int(c) for c in np.unique(ev_class)})

    def emit(s=""):
        print(s)

    def stats(mask):
        v = ev_placement_err[mask]
        if v.size == 0:
            return None
        return {
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p90_abs": float(np.percentile(np.abs(v), 90)),
            "std": float(np.std(v)),
            "fz": float(np.mean(ev_peakfz[mask])),
            "rough": float(np.mean(ev_rough[mask])),
        }

    def fz_stats(mask):
        v = ev_peakfz[mask]
        if v.size == 0:
            return None
        return {"n": int(v.size), "mean": float(np.mean(v)), "median": float(np.median(v))}

    emit(f"\n{'=' * 70}")
    emit(f"  measure_hind_foot_placement {SCRIPT_VERSION}")
    emit(f"  floor_source : {floor_source_label}")
    emit(f"  task         : {args_cli.task}")
    emit(f"  checkpoint   : {resume_path}")
    emit(f"  run          : {ts}  N={N} T={T} seed={args_cli.seed}")
    emit(f"  foot canonical: {CANONICAL}  front={{FL,FR}}  hind={{RL,RR}}")
    emit(f"  pos_names    : {[pos_names_raw[i] for i in pos_order]}")
    emit(f"  contact_names: {[contact_names_raw[i] for i in contact_order]}")
    emit(f"  episodes total={n_total} failure={n_fail} goal={n_goal}  valid_frac={valid.mean():.3f}")
    emit(f"  valid+finite landings = {ev_placement_err.size}  flat={n_flat}")
    emit(f"{'=' * 70}\n")

    emit("placement_error = foot_z - terrain_h [m]")
    emit("  (foot-radius bias is same for front+hind; cancels in hind-front gap.)")
    emit("  PRIMARY stat: median. Secondary: mean. p90 = 90th percentile of |error|.")
    emit("")

    # ── (1) Per-terrain × {front, hind} placement table ──────────────────────
    emit("─" * 70)
    emit("(1) PLACEMENT ERROR  terrain × {front, hind}")
    emit("─" * 70)
    emit(
        f"{'terrain':<18}{'grp':<7}{'n':>7}{'median':>9}{'mean':>9}"
        f"{'p90(|.|)':>11}{'std':>8}{'peakFz':>10}{'rough':>9}"
    )
    stat_cache = {}
    for cls in class_ids:
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"cls{cls}")
        cmask = ev_class == cls
        sf = stats(cmask & ~ev_ishind)
        sh = stats(cmask & ev_ishind)
        stat_cache[cname] = {"front": sf, "hind": sh}
        for grp, s in (("front", sf), ("hind", sh)):
            if s is None:
                continue
            emit(
                f"  {cname:<16}{grp:<7}{s['n']:>7}{s['median']:>+9.4f}{s['mean']:>+9.4f}"
                f"{s['p90_abs']:>11.4f}{s['std']:>8.4f}{s['fz']:>10.1f}{s['rough']:>9.4f}"
            )
    emit("")

    # ── (2) Hind − Front gap table ────────────────────────────────────────────
    emit("─" * 70)
    emit("(2) HIND − FRONT GAP  (primary: median_gap; secondary: mean_gap)")
    emit("─" * 70)
    emit(
        f"{'terrain':<18}{'n_front':>8}{'n_hind':>8}"
        f"{'median_gap':>12}{'mean_gap':>10}{'p90_gap':>10}"
    )
    gap_cache = {}
    for cls in class_ids:
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"cls{cls}")
        sf = stat_cache[cname]["front"]
        sh = stat_cache[cname]["hind"]
        if sf is None or sh is None:
            continue
        mg = sh["median"] - sf["median"]
        meang = sh["mean"] - sf["mean"]
        p90g = sh["p90_abs"] - sf["p90_abs"]
        gap_cache[cname] = {"median_gap": mg, "mean_gap": meang, "p90_gap": p90g, "n_f": sf["n"], "n_h": sh["n"]}
        emit(f"  {cname:<18}{sf['n']:>8}{sh['n']:>8}{mg:>+12.4f}{meang:>+10.4f}{p90g:>+10.4f}")
    emit("")
    flat_mg = gap_cache.get("flat", {}).get("median_gap", float("nan"))
    emit(f"  flat control median_gap = {flat_mg:+.4f} m  (reference for all others)")
    emit("")

    # ── (3) peakFz table with baseline-subtracted column ──────────────────────
    emit("─" * 70)
    emit("(3) PEAK Fz at landing  terrain × {front, hind}  +  baseline-subtracted delta")
    emit("─" * 70)
    emit("  baseline = flat (hind-front) peakFz delta (structural rear-heavy bias)")
    emit(f"{'terrain':<18}{'grp':<7}{'n':>7}{'mean_Fz':>10}{'median_Fz':>11}")
    fz_cache = {}
    for cls in class_ids:
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"cls{cls}")
        cmask = ev_class == cls
        sfz = fz_stats(cmask & ~ev_ishind)
        shz = fz_stats(cmask & ev_ishind)
        fz_cache[cname] = {"front": sfz, "hind": shz}
        for grp, s in (("front", sfz), ("hind", shz)):
            if s is None:
                continue
            emit(f"  {cname:<16}{grp:<7}{s['n']:>7}{s['mean']:>10.1f}{s['median']:>11.1f}")
    emit("")

    # flat hind-front delta (structural bias baseline)
    flat_sfz = fz_cache.get("flat", {}).get("front")
    flat_shz = fz_cache.get("flat", {}).get("hind")
    flat_fz_delta = float("nan")
    if flat_sfz and flat_shz:
        flat_fz_delta = flat_shz["mean"] - flat_sfz["mean"]
    emit(f"  flat (hind-front) peakFz delta = {flat_fz_delta:+.1f} N  (structural baseline to subtract)")
    emit("")
    emit(f"{'terrain':<18}{'hind-front Fz':>16}{'baseline-sub Fz':>18}")
    for cls in class_ids:
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"cls{cls}")
        sfz = fz_cache[cname]["front"]
        shz = fz_cache[cname]["hind"]
        if sfz is None or shz is None:
            continue
        raw_delta = shz["mean"] - sfz["mean"]
        sub_delta = raw_delta - flat_fz_delta if np.isfinite(flat_fz_delta) else float("nan")
        emit(f"  {cname:<18}{raw_delta:>+16.1f}{sub_delta:>+18.1f}")
    emit("")

    # ── (4) Crawl sanity block (already printed above, repeat inline) ──────────
    if crawl_id is not None:
        crawl_mask_ev = (ev_class == crawl_id)
        n_crawl_valid = int(crawl_mask_ev.sum())
        emit("─" * 70)
        emit(f"(4) CRAWL SANITY  (valid+finite landing events = {n_crawl_valid})")
        emit("─" * 70)
        if n_crawl_valid > 0 and args_cli.floor_source == "raycast":
            # Post-filter raycast artefact rate for crawl.
            # ev_placement_err already uses raycast terrain_h (finite mask applied).
            big_rc_final = float((np.abs(ev_placement_err[crawl_mask_ev]) > 0.5).mean())
            emit(f"  |placement_err|>0.5m fraction (raycast, post-filter) = {big_rc_final:.3f}")
            emit("  (Compare to v1 heightfield crawl artefact rate printed earlier in stdout.)")
        elif n_crawl_valid > 0:
            big_hf_final = float((np.abs(ev_placement_err[crawl_mask_ev]) > 0.5).mean())
            emit(f"  |placement_err|>0.5m fraction (heightfield, post-filter) = {big_hf_final:.3f}")
        emit("")

    ts_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 64}")
    print(f"=== measure_hind_foot_placement {SCRIPT_VERSION} DONE @ {ts_end} ===")
    print(f"{'=' * 64}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
