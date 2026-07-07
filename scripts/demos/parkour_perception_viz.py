# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay each perception modality as Sim-scene markers and record one mp4 per modality.

Renders the four perception modalities of the Go2 parkour-imitation LiDAR env — height_scan
(2D heightfield), clearance (3D ray-cast), voxel (occupancy grid), and Mid-360 LiDAR (raw hit
points) — as ``VisualizationMarkers`` overlaid on the Isaac Sim scene, captured by the
``/OmniverseKit_Persp`` render product (``render_mode="rgb_array"``), and written to mp4 with
``imageio``.  Approach "A" (Sim-scene marker overlay + camera recording).

One env (env 0) is driven with a zero action (Go2 default standing pose) and repositioned onto
each of the 4 target terrain classes (crawl / hurdle / gap / stair) in turn; each modality video
shows all 4 terrains sequentially.

Why this works headless
------------------------
- ``VisualizationMarkers`` are ordinary ``/Visuals/...`` USD prims, so the ``/OmniverseKit_Persp``
  render product captures them even with ``--headless --enable_cameras`` (no GUI viewport needed).
- ``DebugViewer`` key-toggle draw callbacks are no-ops headless (``debug_viewer.py:114``), so we
  call the env's ``_draw_clearance_rays`` / ``_draw_voxel_occupied`` **directly** (bypassing the
  callback gating) and build our own markers for height_scan / lidar.
- The persp camera is aimed per-frame via ``sim.set_camera_view`` (independent of the
  ``viewport_camera_controller``, which is ``None`` headless).

Usage
-----
.. code-block:: bash

    conda activate isaac-5.1
    env -u DISPLAY CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
        ./isaaclab.sh -p scripts/demos/parkour_perception_viz.py --headless --enable_cameras

    # Step-1 render-pipeline validation (saves a few PNGs, no mp4):
    ... ./isaaclab.sh -p scripts/demos/parkour_perception_viz.py --headless --enable_cameras --prototype
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record per-modality perception marker overlays for the Go2 parkour env.")
parser.add_argument("--task", type=str, default="Go2-ParkourImitation-Lidar-VizAllModality-v0", help="Task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to spawn (only env 0 is used).")
parser.add_argument(
    "--modality",
    type=str,
    default="all",
    choices=["all", "height_scan", "clearance", "voxel", "lidar"],
    help="Which modality (or all) to record.",
)
parser.add_argument("--out_dir", type=str, default="_workspace/perception_viz", help="Output directory for mp4/png.")
parser.add_argument("--settle_steps", type=int, default=25, help="Steps to let the robot settle after respawn.")
parser.add_argument("--record_steps", type=int, default=60, help="Frames recorded per terrain per modality.")
parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
parser.add_argument("--max_lidar_pts", type=int, default=6000, help="Cap on LiDAR hit markers drawn per frame.")
parser.add_argument("--prototype", action="store_true", help="Step-1 validation: save PNGs of clearance overlay only.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras are mandatory for rgb_array capture.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.parkour.parkour_env_cfg import (
    TERRAIN_CLASS_CRAWL,
    TERRAIN_CLASS_GAP,
    TERRAIN_CLASS_HURDLE,
    TERRAIN_CLASS_STAIR,
)
from isaaclab_tasks.utils import parse_env_cfg

# Ordered (class_id, short_name) list of terrains to visit within each modality video.
TERRAIN_SEQUENCE = [
    (TERRAIN_CLASS_CRAWL, "crawl"),
    (TERRAIN_CLASS_HURDLE, "hurdle"),
    (TERRAIN_CLASS_GAP, "gap"),
    (TERRAIN_CLASS_STAIR, "stair"),
]

MODALITIES = ["height_scan", "clearance", "voxel", "lidar"]


def make_point_markers(prim_path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    """Create a single-prototype sphere marker set (mirrors the env's clearance-vis pattern)."""
    cfg = SPHERE_MARKER_CFG.copy()
    cfg.prim_path = prim_path
    cfg.markers["sphere"].radius = radius
    cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkers(cfg)


def place_env_on_class(env, class_id: int, env_id: int = 0) -> bool:
    """Force env ``env_id`` onto a mid-difficulty tile of terrain ``class_id`` and reset it.

    Terrain curriculum is frozen for this cfg, so the forced level/type persists.
    Returns False if no column of the requested class exists.
    """
    col_to_class = env._col_to_class  # (num_cols,)
    cols = (col_to_class == class_id).nonzero(as_tuple=False).flatten()
    if cols.numel() == 0:
        return False
    col = int(cols[cols.numel() // 2].item())  # middle column of that class

    origins = env._terrain.terrain_origins  # (num_rows, num_cols, 3)
    num_rows = int(origins.shape[0])
    lvl = num_rows // 2  # mid difficulty row

    env._terrain_levels[env_id] = lvl
    env._terrain_types[env_id] = col
    env._env_class[env_id] = env._col_to_class[col]
    env._terrain.env_origins[env_id] = origins[lvl, col]
    env._skip_curriculum[env_id] = True
    env._reset_idx(torch.tensor([env_id], device=env.device))
    return True


def aim_camera(env, env_id: int = 0, back: float = 2.6, side: float = 2.2, up: float = 1.7) -> None:
    """Aim the persp render camera at env ``env_id``'s robot base (world-frame diagonal view)."""
    pos = env._robot.data.root_pos_w[env_id].detach().cpu().numpy()
    eye = (float(pos[0] - back), float(pos[1] - side), float(pos[2] + up))
    target = (float(pos[0]), float(pos[1]), float(pos[2] + 0.2))
    env.sim.set_camera_view(eye=eye, target=target)


def hide_all(env, hs_vis, lidar_vis) -> None:
    """Hide every modality marker set so only the active modality is visible."""
    hs_vis.set_visibility(False)
    lidar_vis.set_visibility(False)
    for attr in (
        "_clearance_hit_up_vis",
        "_clearance_hit_down_vis",
        "_clearance_miss_up_vis",
        "_clearance_miss_down_vis",
        "_voxel_occupied_visualizer",
    ):
        v = getattr(env, attr, None)
        if v is not None:
            v.set_visibility(False)


def draw_modality(env, modality: str, hs_vis, lidar_vis, max_lidar_pts: int) -> int:
    """Draw markers for the given modality from freshly-updated env buffers. Returns #points drawn."""
    if modality == "height_scan":
        pts = env._height_scanner.data.ray_hits_w[0]  # (187, 3)
        fin = torch.isfinite(pts).all(dim=-1)
        pts = pts[fin]
        if pts.shape[0] > 0:
            hs_vis.set_visibility(True)
            hs_vis.visualize(translations=pts)
        return int(pts.shape[0])

    if modality == "clearance":
        env._draw_clearance_rays()
        hits = env._clearance_scanner.data.ray_hits_w[0]
        return int(torch.isfinite(hits).all(dim=-1).sum().item())

    if modality == "voxel":
        env._draw_voxel_occupied()
        return int((env._voxel_grid[0] == 1).sum().item())

    if modality == "lidar":
        hits = env._mid360.data.ray_hits_w[0]  # (R, 3)
        dist = env._mid360.data.distances[0]  # (R,)
        max_range = float(env.cfg.mid360_lidar.max_distance)
        valid = torch.isfinite(hits).all(dim=-1) & (dist < max_range - 0.1)
        pts = hits[valid]
        if pts.shape[0] > max_lidar_pts:
            idx = torch.randperm(pts.shape[0], device=pts.device)[:max_lidar_pts]
            pts = pts[idx]
        if pts.shape[0] > 0:
            lidar_vis.set_visibility(True)
            lidar_vis.visualize(translations=pts)
        return int(pts.shape[0])

    raise ValueError(f"unknown modality {modality!r}")


def capture_frame(base, hs_vis, lidar_vis, modality: str, max_lidar_pts: int):
    """Hide others, draw the active modality, aim camera, force a render, read rgb. Returns (frame, npts).

    ``base`` must be the *unwrapped* env: gymnasium ``Wrapper.__getattr__`` blocks access to
    private (``_``-prefixed) attributes/methods, so all marker/draw/camera calls go through the
    unwrapped env directly.
    """
    hide_all(base, hs_vis, lidar_vis)
    npts = draw_modality(base, modality, hs_vis, lidar_vis, max_lidar_pts)
    aim_camera(base)
    # Force a render pass AFTER markers/camera update so the annotator sees them.
    base.sim.render()
    frame = base.render()  # (H, W, 3) uint8 or None
    return frame, npts


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped

    act_dim = int(env_cfg.action_space)
    zero_act = torch.zeros((base.num_envs, act_dim), device=base.device)

    # Marker sets we own (clearance/voxel are created lazily by the env's own draw fns).
    hs_vis = make_point_markers("/Visuals/Viz/height_scan", (1.0, 0.85, 0.0), radius=0.03)  # yellow
    lidar_vis = make_point_markers("/Visuals/Viz/lidar", (0.7, 0.1, 0.9), radius=0.02)  # purple

    env.reset()

    # Renderer warm-up: first frames come back empty until the RTX renderer settles.
    for _ in range(6):
        base.sim.render()
        _ = base.render()

    if args_cli.prototype:
        run_prototype(env, base, zero_act, hs_vis, lidar_vis)
    else:
        run_full(env, base, zero_act, hs_vis, lidar_vis)

    # NOTE: simulation_app.close() is intentionally skipped — it segfaults on Isaac 5.1 headless.
    print("[viz] done (app not closed on purpose to avoid Isaac 5.1 headless segfault).")


def run_prototype(env, base, zero_act, hs_vis, lidar_vis):
    """Step-1 validation: place robot on crawl, overlay clearance markers, save a few PNGs."""
    proto_dir = os.path.join(args_cli.out_dir, "prototype")
    os.makedirs(proto_dir, exist_ok=True)
    print("[viz][prototype] validating render pipeline on 'crawl' with clearance overlay ...")

    ok = place_env_on_class(base, TERRAIN_CLASS_CRAWL)
    print(f"[viz][prototype] place_env_on_class(crawl) -> {ok}")
    for _ in range(args_cli.settle_steps):
        env.step(zero_act)

    saved = 0
    for i in range(8):
        env.step(zero_act)
        frame, npts = capture_frame(base, hs_vis, lidar_vis, "clearance", args_cli.max_lidar_pts)
        nonzero = frame is not None and int(np.asarray(frame).sum()) > 0
        print(f"[viz][prototype] frame {i}: clearance_hits={npts} rgb_nonzero={nonzero} shape={None if frame is None else frame.shape}")
        if nonzero and i % 2 == 0:
            path = os.path.join(proto_dir, f"proto_clearance_{i:02d}.png")
            imageio.imwrite(path, np.asarray(frame))
            saved += 1
            print(f"[viz][prototype] saved {path}")
    print(f"[viz][prototype] DONE — saved {saved} PNGs to {proto_dir}. Inspect them for visible markers.")


def run_full(env, base, zero_act, hs_vis, lidar_vis):
    """Step-2: one mp4 per modality, each sweeping the 4 target terrains."""
    modalities = MODALITIES if args_cli.modality == "all" else [args_cli.modality]

    for modality in modalities:
        out_path = os.path.join(args_cli.out_dir, f"{modality}.mp4")
        writer = imageio.get_writer(out_path, fps=args_cli.fps, macro_block_size=None)
        print(f"[viz] === recording modality '{modality}' -> {out_path} ===")
        total_frames = 0

        for class_id, name in TERRAIN_SEQUENCE:
            ok = place_env_on_class(base, class_id)
            if not ok:
                print(f"[viz]   terrain '{name}' (class {class_id}) has no column — skipped.")
                continue
            for _ in range(args_cli.settle_steps):
                env.step(zero_act)

            wrote = 0
            for _ in range(args_cli.record_steps):
                env.step(zero_act)
                frame, npts = capture_frame(base, hs_vis, lidar_vis, modality, args_cli.max_lidar_pts)
                if frame is not None and int(np.asarray(frame).sum()) > 0:
                    writer.append_data(np.asarray(frame))
                    wrote += 1
            total_frames += wrote
            print(f"[viz]   terrain '{name}': wrote {wrote} frames (last #pts={npts}).")

        writer.close()
        print(f"[viz] === modality '{modality}' DONE: {total_frames} frames -> {out_path} ===")


if __name__ == "__main__":
    main()
