# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo for the Livox Mid-360 LiDAR implemented as a custom warp RayCaster pattern.

A Unitree Go2 robot carries a Mid-360 LiDAR (custom :class:`LivoxPatternCfg`) over a ground
plane. The script steps the simulation and reports point-cloud statistics (valid finite hits,
x/y/z range, miss/NaN/inf counts).

.. note::
    The Mid-360 FOV is mostly *upward* (elevation phi in [-7.2deg, +52.2deg]), so with zero tilt
    over flat ground most rays point up and miss (returned as ``inf``). Apply a forward-down tilt
    via ``--tilt_deg`` so the upward cone rotates below the horizon and sweeps the ground.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/sensors/livox_mid360_demo.py --headless --num_envs 1
    ./isaaclab.sh -p scripts/demos/sensors/livox_mid360_demo.py --headless --tilt_deg 30
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Livox Mid-360 LiDAR (custom RayCaster pattern) demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_rays", type=int, default=20000, help="Number of LiDAR rays to subsample.")
parser.add_argument(
    "--tilt_deg",
    type=float,
    default=0.0,
    help="Forward-down pitch of the sensor in degrees (rotation about the y-axis).",
)
parser.add_argument("--mount_x", type=float, default=0.3, help="Sensor offset position on Robot/base link [m].")
parser.add_argument("--mount_y", type=float, default=0.0, help="Sensor offset position on Robot/base link [m].")
parser.add_argument("--mount_z", type=float, default=0.1, help="Sensor offset position on Robot/base link [m].")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
parser.add_argument(
    "--dump",
    action="store_true",
    default=True,
    help="Dump the stabilized point cloud (.npy) to _workspace/sim2real/ for visualization.",
)
parser.add_argument("--no-dump", dest="dump", action="store_false", help="Disable point-cloud dump.")
parser.add_argument(
    "--dump_step", type=int, default=40, help="Simulation step at which to dump the point cloud (once)."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct._common.sensors import LivoxPatternCfg, livox_pattern

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


def tilt_quaternion(tilt_deg: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a forward-down pitch about the y-axis.

    A positive ``tilt_deg`` rotates the sensor's local frame so the forward (mostly-upward) rays
    sweep downward toward the ground.
    """
    half = math.radians(tilt_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


@configclass
class LivoxDemoSceneCfg(InteractiveSceneCfg):
    """Scene with a Go2 robot carrying a Mid-360 LiDAR over ground + obstacle boxes."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # NOTE: obstacle boxes are intentionally omitted. The core RayCaster supports exactly ONE
    # mesh prim path (it raises NotImplementedError for len(mesh_prim_paths) != 1), so a single
    # sensor cannot ray-cast against both the ground and separate obstacle prims. We ray-cast
    # against the ground plane and rely on a forward-down tilt to sweep the mostly-upward FOV
    # onto the ground.

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Mid-360 LiDAR (custom warp RayCaster pattern)
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(
            pos=(args_cli.mount_x, args_cli.mount_y, args_cli.mount_z), rot=tilt_quaternion(args_cli.tilt_deg)
        ),
        pattern_cfg=LivoxPatternCfg(num_rays=args_cli.num_rays),
        ray_alignment="base",
        mesh_prim_paths=["/World/ground"],
        max_distance=70.0,
        debug_vis=not args_cli.headless,
    )


def report_stats(hits: torch.Tensor, step: int) -> None:
    """Print point-cloud statistics over finite (valid) hits only."""
    pts = hits.reshape(-1, 3)
    total = pts.shape[0]
    finite_mask = torch.isfinite(pts).all(dim=-1)
    n_nan = torch.isnan(pts).any(dim=-1).sum().item()
    n_inf = torch.isinf(pts).any(dim=-1).sum().item()
    valid = pts[finite_mask]
    n_valid = valid.shape[0]

    print(f"--- step {step}: total_rays={total} valid_hits={n_valid} miss/inf={n_inf} nan={n_nan}")
    if n_valid > 0:
        mn = valid.min(dim=0).values
        mx = valid.max(dim=0).values
        print(f"    x[{mn[0]:.3f}, {mx[0]:.3f}] y[{mn[1]:.3f}, {mx[1]:.3f}] z[{mn[2]:.3f}, {mx[2]:.3f}]")
    else:
        print("    (no finite hits this step)")


def dump_point_cloud(scene: InteractiveScene, step: int) -> None:
    """Dump the stabilized point cloud (env 0) to ``_workspace/sim2real/`` as .npy files."""
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "_workspace", "sim2real")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    sensor = scene["ray_caster"]
    # world-frame hits for env 0, finite (valid) only
    hits = sensor.data.ray_hits_w[0]  # (num_rays, 3)
    finite_mask = torch.isfinite(hits).all(dim=-1)
    hits_world = hits[finite_mask].detach().cpu().numpy().astype(np.float32)
    # sensor origin world position for env 0
    sensor_pos = sensor.data.pos_w[0].detach().cpu().numpy().astype(np.float32)
    # sensor-local pattern directions (raw beam set, no offset/tilt) for beam-distribution analysis
    _, dirs_local = livox_pattern(sensor.cfg.pattern_cfg, str(sensor.data.pos_w.device))
    dirs_local = dirs_local.detach().cpu().numpy().astype(np.float32)

    path_hits = os.path.join(out_dir, "mid360_hits_world.npy")
    path_pos = os.path.join(out_dir, "mid360_sensor_pos.npy")
    path_dirs = os.path.join(out_dir, "mid360_dirs_local.npy")
    np.save(path_hits, hits_world)
    np.save(path_pos, sensor_pos)
    np.save(path_dirs, dirs_local)

    print(f"[DUMP] step {step}: wrote point cloud to {out_dir}")
    print(f"[DUMP]   {path_hits}  shape={hits_world.shape}")
    print(f"[DUMP]   {path_pos}  shape={sensor_pos.shape}")
    print(f"[DUMP]   {path_dirs}  shape={dirs_local.shape}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Run the simulator and report LiDAR statistics."""
    sim_dt = sim.get_physics_dt()
    count = 0
    dumped = False

    print(
        f"[INFO]: Mid-360 LiDAR: num_rays={scene['ray_caster'].num_rays}"
        f", mount=({args_cli.mount_x}, {args_cli.mount_y}, {args_cli.mount_z})"
        f", tilt_deg={args_cli.tilt_deg}"
    )

    while simulation_app.is_running() and count < args_cli.steps:
        if count % 100 == 0:
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_vel = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # hold default pose
        scene["robot"].set_joint_position_target(scene["robot"].data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # report every 20 steps (after a few warmup steps)
        if count % 20 == 0:
            report_stats(scene["ray_caster"].data.ray_hits_w, count)

        # one-time point-cloud dump at a stabilized step
        if args_cli.dump and not dumped and count >= args_cli.dump_step:
            dump_point_cloud(scene, count)
            dumped = True


def main() -> None:
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene_cfg = LivoxDemoSceneCfg(num_envs=args_cli.num_envs, env_spacing=8.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
