# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo for the Livox Mid-360 LiDAR over a procedural rough terrain with static obstacles.

Uses :class:`MultiMeshRayCasterCfg` so the sensor casts against both the terrain mesh and
obstacle meshes simultaneously, giving much higher hit density than a bare ground plane.

Obstacle cubes and spheres are spawned under ``/World/static/`` so they can be referenced
as a single prim-expression target in the sensor config.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/sensors/livox_mid360_scene_demo.py --headless --num_envs 1
    ./isaaclab.sh -p scripts/demos/sensors/livox_mid360_scene_demo.py --headless --tilt_deg 40
    ./isaaclab.sh -p scripts/demos/sensors/livox_mid360_scene_demo.py --headless \\
        --mount_x 0.32 --mount_z 0.04 --tilt_deg 40 --num_rays 24000
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Livox Mid-360 LiDAR scene demo (terrain + obstacles, MultiMeshRayCaster)."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_rays", type=int, default=24000, help="Number of LiDAR rays to subsample.")
parser.add_argument(
    "--tilt_deg",
    type=float,
    default=40.0,
    help="Forward-down pitch of the sensor in degrees (rotation about the y-axis).",
)
parser.add_argument("--mount_x", type=float, default=0.32, help="Sensor offset position on Robot/base link [m].")
parser.add_argument("--mount_y", type=float, default=0.0, help="Sensor offset position on Robot/base link [m].")
parser.add_argument("--mount_z", type=float, default=0.04, help="Sensor offset position on Robot/base link [m].")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
parser.add_argument(
    "--dump",
    action="store_true",
    default=True,
    help="Dump the stabilized point cloud (.npy) to _workspace/sim2real/ for visualization.",
)
parser.add_argument("--no-dump", dest="dump", action="store_false", help="Disable point-cloud dump.")
parser.add_argument(
    "--dump_step", type=int, default=60, help="Simulation step at which to dump the point cloud (once)."
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
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct._common.sensors import LivoxPatternCfg, livox_pattern

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


def tilt_quaternion(tilt_deg: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a forward-down pitch about the y-axis.

    A positive ``tilt_deg`` rotates the sensor's local frame so the forward (mostly-upward) rays
    sweep downward toward the ground.
    """
    half = math.radians(tilt_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


@configclass
class LivoxSceneCfg(InteractiveSceneCfg):
    """Scene with rough terrain, static obstacles, and a Go2 carrying a Mid-360 LiDAR."""

    # procedural rough terrain — prim_path becomes the mesh target for the height scanner
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # static obstacles under /World/static/ — all referenced by a single prim_expr in the sensor
    # NOTE: These are AssetBaseCfg (static, not RigidObject) so they stay in place as meshes.
    obs_cube_a = AssetBaseCfg(
        prim_path="/World/static/cube_a",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 1.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1), metallic=0.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, 1.5, 0.6)),
    )
    obs_cube_b = AssetBaseCfg(
        prim_path="/World/static/cube_b",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.8, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.2), metallic=0.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.5, 3.0, 0.4)),
    )
    obs_cube_c = AssetBaseCfg(
        prim_path="/World/static/cube_c",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.3, 0.6),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.9), metallic=0.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.0, -2.0, 0.3)),
    )
    obs_sphere_a = AssetBaseCfg(
        prim_path="/World/static/sphere_a",
        spawn=sim_utils.SphereCfg(
            radius=0.6,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.7, 0.1), metallic=0.3),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-3.5, -2.5, 0.6)),
    )
    obs_sphere_b = AssetBaseCfg(
        prim_path="/World/static/sphere_b",
        spawn=sim_utils.SphereCfg(
            radius=0.4,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.1, 0.8), metallic=0.3),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -4.0, 0.4)),
    )

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Mid-360 LiDAR — MultiMeshRayCaster against terrain + all /World/static/* obstacles
    # Static obstacles use RaycastTargetCfg with track_mesh_transforms=False (they never move)
    # for better runtime performance.
    ray_caster = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(args_cli.mount_x, args_cli.mount_y, args_cli.mount_z),
            rot=tilt_quaternion(args_cli.tilt_deg),
        ),
        pattern_cfg=LivoxPatternCfg(num_rays=args_cli.num_rays, sampling="window"),
        ray_alignment="base",
        mesh_prim_paths=[
            # terrain mesh — static, shared across envs
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",
                is_shared=True,
                track_mesh_transforms=False,
            ),
            # all static obstacles under /World/static/ — single regex, static
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/static/.*",
                is_shared=True,
                track_mesh_transforms=False,
            ),
        ],
        max_distance=20.0,
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

    path_hits = os.path.join(out_dir, "mid360_scene_hits_world.npy")
    path_pos = os.path.join(out_dir, "mid360_scene_sensor_pos.npy")
    path_dirs = os.path.join(out_dir, "mid360_scene_dirs_local.npy")
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
        f"[INFO]: Mid-360 LiDAR (scene): num_rays={scene['ray_caster'].num_rays}"
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

        # report every 20 steps
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
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])

    scene_cfg = LivoxSceneCfg(num_envs=args_cli.num_envs, env_spacing=10.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
