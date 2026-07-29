# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualise the 3D-voxel policy input as Sim-scene markers, with a binary-vs-ternary A/B.

Ported from the 5.1 ``parkour_perception_viz.py`` (voxel modality only) and extended with the
comparison that matters for the gap-terrain question: the *same* ray cast is filled twice —
once in binary mode (``{0,1}``, what the policy currently sees) and once in legacy ternary mode
(``{-1 unknown, 0 free, 1 occupied}``) — so the information dropped by the binary encoding is
visible in one frame rather than inferred.

Markers
-------
* orange cuboids — occupied cells (``grid == 1``). Identical in both modes.
* cyan spheres   — cells that ternary marks *free* (ray-traversed empty space) but binary
  collapses to ``0``, indistinguishable from never-sampled space.

Why this works headless: ``VisualizationMarkers`` are ordinary ``/Visuals/...`` USD prims, so the
``/OmniverseKit_Persp`` render product captures them without a GUI viewport. ``DebugViewer`` key
toggles are no-ops headless, so the env's ``_draw_voxel_occupied`` is called directly.

Usage
-----
.. code-block:: bash

    env -u DISPLAY CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
        ./isaaclab.sh -p scripts/demos/parkour_voxel_viz.py --headless --enable_cameras \
        --terrains gap
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualise the Go2 parkour 3D-voxel policy input.")
parser.add_argument("--task", type=str, default="Go2-ParkourImitation-Teacher3DVoxel-EasyEntry-v0", help="Task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Envs to spawn (only env 0 is used).")
parser.add_argument(
    "--terrains",
    type=str,
    default="gap",
    help="Comma-separated terrain classes to visit: gap,crawl,hurdle,stair,step,flat.",
)
parser.add_argument("--out_dir", type=str, default="_workspace/voxel_viz", help="Output directory.")
parser.add_argument("--settle_steps", type=int, default=25, help="Steps to settle after respawn.")
parser.add_argument("--record_steps", type=int, default=40, help="Frames recorded per terrain.")
parser.add_argument("--fps", type=int, default=15, help="Output video fps.")
parser.add_argument("--png_every", type=int, default=10, help="Save a PNG every N recorded frames.")
parser.add_argument("--side_view", action="store_true", help="Camera from the side (better for gap voids).")
parser.add_argument("--x_offset", type=float, default=0.0, help="Teleport the robot this far forward before recording.")
parser.add_argument(
    "--show_free",
    action="store_true",
    help="Also draw the ternary-free cells (cyan). Off by default: they swamp the frame, and what "
    "the policy actually sees is the orange occupied set alone.",
)
parser.add_argument("--cam_dist", type=float, default=1.9, help="Camera distance from the robot (m).")
parser.add_argument(
    "--stats_only",
    action="store_true",
    help="Skip rendering; print the clearance-ray → voxel-occupied loss breakdown per terrain.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras are mandatory for rgb_array capture. Disabling them also breaks the 6.0 headless
# physics-view creation path, so stats-only runs keep the camera pipeline and just skip capture.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.parkour.parkour_env_cfg import (
    TERRAIN_CLASS_CRAWL,
    TERRAIN_CLASS_FLAT,
    TERRAIN_CLASS_GAP,
    TERRAIN_CLASS_HURDLE,
    TERRAIN_CLASS_STAIR,
    TERRAIN_CLASS_STEP,
)
from isaaclab_tasks.direct.parkour.voxel_occupancy import fill_voxel_grid, voxel_grid_shape
from isaaclab_tasks.utils import parse_env_cfg

CLASS_BY_NAME = {
    "gap": TERRAIN_CLASS_GAP,
    "crawl": TERRAIN_CLASS_CRAWL,
    "hurdle": TERRAIN_CLASS_HURDLE,
    "stair": TERRAIN_CLASS_STAIR,
    "step": TERRAIN_CLASS_STEP,
    "flat": TERRAIN_CLASS_FLAT,
}


def make_point_markers(prim_path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    """Single-prototype sphere marker set (mirrors the env's clearance-vis pattern)."""
    cfg = SPHERE_MARKER_CFG.copy()
    cfg.prim_path = prim_path
    cfg.markers["sphere"].radius = radius
    cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkers(cfg)


def place_env_on_class(env, class_id: int, env_id: int = 0) -> bool:
    """Force env ``env_id`` onto a mid-difficulty tile of terrain ``class_id`` and reset it.

    Returns False if no column of the requested class exists.
    """
    cols = (env._col_to_class == class_id).nonzero(as_tuple=False).flatten()
    if cols.numel() == 0:
        return False
    col = int(cols[cols.numel() // 2].item())  # middle column of that class

    origins = env._terrain.terrain_origins  # (num_rows, num_cols, 3)
    lvl = int(origins.shape[0]) // 2  # mid difficulty row

    env._terrain_levels[env_id] = lvl
    env._terrain_types[env_id] = col
    env._env_class[env_id] = env._col_to_class[col]
    env._terrain.env_origins[env_id] = origins[lvl, col]
    env._skip_curriculum[env_id] = True
    env._reset_idx(torch.tensor([env_id], device=env.device))
    return True


def aim_camera(env, env_id: int = 0, side_view: bool = False) -> None:
    """Aim the persp render camera so the robot AND its 2 m forward voxel volume both fit."""
    pos = env._robot.data.root_pos_w[env_id].detach().cpu().numpy()
    d = args_cli.cam_dist
    if side_view:
        # Pure side profile: the gap void and the grid's z extent read clearly.
        eye = (float(pos[0] + 0.7), float(pos[1] - d), float(pos[2] + 0.35))
    else:
        # Over-the-shoulder: robot in frame, grid extending forward.
        eye = (float(pos[0] - d), float(pos[1] - 0.8 * d), float(pos[2] + 0.6 * d))
    target = (float(pos[0] + 0.7), float(pos[1]), float(pos[2] - 0.1))
    env.sim.set_camera_view(eye=eye, target=target)


def push_robot_forward(env, dx: float, env_id: int = 0) -> None:
    """Teleport the robot ``dx`` m along its heading so the terrain feature enters the grid.

    Zero-action playback never walks, so the first gap sits outside the 2 m forward voxel
    volume unless the robot is placed next to it.
    """
    if abs(dx) < 1e-6:
        return
    root = env._robot.data.root_state_w[env_id : env_id + 1].clone()
    fwd = math_utils.quat_apply_yaw(root[:, 3:7], torch.tensor([[dx, 0.0, 0.0]], device=env.device))
    root[:, 0:3] += fwd
    env._robot.write_root_pose_to_sim(root[:, 0:7], env_ids=torch.tensor([env_id], device=env.device))
    env._robot.write_root_velocity_to_sim(root[:, 7:13] * 0.0, env_ids=torch.tensor([env_id], device=env.device))


def ternary_grid(env) -> torch.Tensor:
    """Refill the voxel grid from the current ray cast in legacy ternary mode.

    Mirrors ``parkour_env._update_perception`` exactly (same sanitisation, same is_hit rule),
    changing only ``binary_occupancy`` — so any difference is attributable to the encoding.
    """
    sensor_pos = env._clearance_scanner.data.pos_w  # (N, 3)
    hits = env._clearance_scanner.data.ray_hits_w  # (N, R, 3), inf/NaN on miss
    dist = torch.norm(hits - sensor_pos.unsqueeze(1), dim=-1)
    dist = torch.nan_to_num(dist, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    is_hit = dist < env._voxel_cfg.max_distance

    cfg_t = copy.deepcopy(env._voxel_cfg)
    cfg_t.binary_occupancy = False
    return fill_voxel_grid(env._voxel_ray_dirs, dist, is_hit, cfg_t, env.num_envs, env.device)


def ray_to_voxel_breakdown(env, env_id: int = 0) -> dict:
    """Count how the 294 clearance rays shrink to N occupied voxels, stage by stage.

    clearance feeds all 294 rays to the policy as a distance vector (misses clamped to
    max_distance, so every element carries a value). The voxel arm keeps only the hits that
    land inside the grid, then collapses duplicates that quantise into the same 0.1 m cell.
    """
    sensor_pos = env._clearance_scanner.data.pos_w
    hits = env._clearance_scanner.data.ray_hits_w
    dist = torch.norm(hits - sensor_pos.unsqueeze(1), dim=-1)
    dist = torch.nan_to_num(dist, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    max_d = env._voxel_cfg.max_distance
    is_hit = (dist < max_d)[env_id]  # (R,)

    cfg = env._voxel_cfg
    res = cfg.resolution
    nx, ny, nz = voxel_grid_shape(cfg)
    pts = dist[env_id].clamp(max=max_d).unsqueeze(-1) * env._voxel_ray_dirs  # (R, 3) sensor-local
    ix = ((pts[:, 0] - cfg.x_range[0]) / res).round().long()
    iy = ((pts[:, 1] - cfg.y_range[0]) / res).round().long()
    iz = ((pts[:, 2] - cfg.z_range[0]) / res).round().long()
    inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)

    kept = is_hit & inb
    flat = (ix * (ny * nz) + iy * nz + iz)[kept]
    return {
        "n_rays": int(is_hit.numel()),
        "n_hit": int(is_hit.sum().item()),
        "n_miss": int((~is_hit).sum().item()),
        "n_out_of_grid": int((is_hit & ~inb).sum().item()),
        "n_kept": int(kept.sum().item()),
        "n_unique": int(torch.unique(flat).numel()) if flat.numel() else 0,
        "n_cells": nx * ny * nz,
    }


def grid_to_world(env, idx: torch.Tensor, env_id: int = 0) -> torch.Tensor:
    """Voxel indices (M, 3) → world positions, using the env's own fill convention."""
    x_min, y_min, z_min = (
        env._voxel_cfg.x_range[0],
        env._voxel_cfg.y_range[0],
        env._voxel_cfg.z_range[0],
    )
    res = env._voxel_cfg.resolution
    local = torch.stack(
        [
            x_min + idx[:, 0].float() * res,
            y_min + idx[:, 1].float() * res,
            z_min + idx[:, 2].float() * res,
        ],
        dim=-1,
    )
    quat = env._robot.data.root_quat_w[env_id].unsqueeze(0).expand(local.shape[0], -1)
    return math_utils.quat_apply_yaw(quat, local) + env._clearance_scanner.data.pos_w[env_id]


def capture_frame(base, free_vis, side_view: bool):
    """Draw binary-occupied + ternary-free markers, render, return (frame, stats)."""
    base._draw_voxel_occupied()  # orange cuboids = binary occupied (== policy input's 1s)

    g_bin = base._voxel_grid[0]
    g_ter = ternary_grid(base)[0]

    # Cells ternary calls free but binary collapses into the same 0 as never-sampled space.
    free_idx = ((g_ter == 0) & (g_bin == 0)).nonzero(as_tuple=False)
    if args_cli.show_free and free_idx.shape[0] > 0:
        free_vis.set_visibility(True)
        free_vis.visualize(translations=grid_to_world(base, free_idx))
    else:
        free_vis.set_visibility(False)

    stats = {
        "bin_occ": int((g_bin == 1).sum().item()),
        "ter_occ": int((g_ter == 1).sum().item()),
        "ter_free": int((g_ter == 0).sum().item()),
        "ter_unknown": int((g_ter == -1).sum().item()),
        "total": int(g_bin.numel()),
        # Posture sanity: zero-action playback on gap terrain can topple the robot, which would
        # make the rendered grid meaningless. grav_z ≈ -1 upright, → 0 fallen.
        "root_z": float(base._robot.data.root_pos_w[0, 2].item()),
        "grav_z": float(base._robot.data.projected_gravity_b[0, 2].item()),
    }

    aim_camera(base, side_view=side_view)
    base.sim.render()  # force a render pass AFTER markers/camera update
    return base.render(), stats


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped

    zero_act = torch.zeros((base.num_envs, int(env_cfg.action_space)), device=base.device)
    free_vis = make_point_markers("/Visuals/Viz/voxel_free", (0.0, 0.9, 0.9), radius=0.022)  # cyan

    env.reset()
    for _ in range(6):  # renderer warm-up: first frames come back empty
        base.sim.render()
        _ = base.render()

    names = [n.strip() for n in args_cli.terrains.split(",") if n.strip()]
    for name in names:
        if name not in CLASS_BY_NAME:
            print(f"[voxel-viz] unknown terrain '{name}' — skipped.")
            continue
        if not place_env_on_class(base, CLASS_BY_NAME[name]):
            print(f"[voxel-viz] terrain '{name}' has no column in this cfg — skipped.")
            continue
        for _ in range(args_cli.settle_steps):
            env.step(zero_act)
        if args_cli.x_offset:
            push_robot_forward(base, args_cli.x_offset)
            for _ in range(args_cli.settle_steps):
                env.step(zero_act)

        if args_cli.stats_only:
            # Average over a few steps so a single unlucky frame doesn't define the numbers.
            acc, reps = {}, 8
            for _ in range(reps):
                env.step(zero_act)
                b = ray_to_voxel_breakdown(base)
                b["occupied"] = int((base._voxel_grid[0] == 1).sum().item())
                b["ter_free"] = int((ternary_grid(base)[0] == 0).sum().item())
                for k, v in b.items():
                    acc[k] = acc.get(k, 0) + v
            a = {k: v / reps for k, v in acc.items()}
            grav = float(base._robot.data.projected_gravity_b[0, 2].item())
            print(
                f"[voxel-stats] {name}  (grav_z={grav:.2f} {'upright' if grav < -0.85 else 'TILTED'})\n"
                f"    clearance : {a['n_rays']:.0f} rays → policy gets {a['n_rays']:.0f} distance values"
                f" (hit {a['n_hit']:.1f}, miss→clamp {a['n_miss']:.1f})\n"
                f"    voxel     : {a['n_hit']:.1f} hits −{a['n_out_of_grid']:.1f} outside grid"
                f" = {a['n_kept']:.1f} kept −{a['n_kept'] - a['n_unique']:.1f} duplicate cells"
                f" = {a['n_unique']:.1f} occupied\n"
                f"    → clearance {a['n_rays']:.0f} values vs voxel {a['occupied']:.1f} occupied"
                f" / {a['n_cells']:.0f} cells ({100 * a['occupied'] / a['n_cells']:.2f}% non-zero);"
                f" ternary would add {a['ter_free']:.0f} free cells"
            )
            continue

        out_mp4 = os.path.join(args_cli.out_dir, f"voxel_{name}.mp4")
        writer = imageio.get_writer(out_mp4, fps=args_cli.fps, macro_block_size=None)
        wrote, last = 0, None
        for i in range(args_cli.record_steps):
            env.step(zero_act)
            frame, stats = capture_frame(base, free_vis, args_cli.side_view)
            if frame is None or int(np.asarray(frame).sum()) == 0:
                continue
            arr = np.asarray(frame)
            writer.append_data(arr)
            if wrote % max(1, args_cli.png_every) == 0:
                imageio.imwrite(os.path.join(args_cli.out_dir, f"voxel_{name}_{wrote:03d}.png"), arr)
            wrote += 1
            last = stats
        writer.close()

        if last:
            pct = 100.0 * last["ter_free"] / last["total"]
            print(
                f"[voxel-viz] {name}: {wrote} frames -> {out_mp4}\n"
                f"           posture: root_z = {last['root_z']:.3f} m, grav_z = {last['grav_z']:.2f}"
                f" ({'upright' if last['grav_z'] < -0.85 else 'FALLEN/TILTED — grid may be meaningless'})\n"
                f"           binary occupied = {last['bin_occ']} / {last['total']}\n"
                f"           ternary  occupied = {last['ter_occ']}, free = {last['ter_free']} ({pct:.1f}%),"
                f" unknown = {last['ter_unknown']}\n"
                f"           → binary drops {last['ter_free']} free cells into the same 0 as unknown."
            )
        else:
            print(f"[voxel-viz] {name}: no non-empty frames captured (GPU render contention?).")

    print("[voxel-viz] done.")


if __name__ == "__main__":
    main()
