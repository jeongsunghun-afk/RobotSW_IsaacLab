# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a per-terrain gait video for a trained RSL-RL parkour policy.

Forked from ``scripts/reinforcement_learning/rsl_rl/play.py`` with three changes:

1. Terrain isolation: before ``gym.make`` the target sub-terrain proportion is set
   to 1.0 and all others to 0.0, so the whole grid is a single terrain type.
2. JIT/ONNX export is skipped (it can crash on the voxel/lidar networks and is
   unnecessary for video capture).
3. A single video is recorded to an exact ``--out_path`` and a forward-velocity
   liveness metric is logged so a collapsed policy (near-zero forward motion) is
   distinguishable from a walking one even in headless runs.

Runner-class resolution reuses play.py's ``agent_cfg.class_name`` switch so each
of the four methods (height_scan / clearance / voxel / lidar) loads correctly.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# local imports — reuse the RSL-RL CLI arg helpers from the play/train scripts.
import os as _os_boot
import sys

from isaaclab.app import AppLauncher

sys.path.insert(
    0,
    _os_boot.path.join(
        _os_boot.path.dirname(_os_boot.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"
    ),
)
import cli_args  # isort: skip  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(description="Record a per-terrain video for a trained RSL-RL parkour agent.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=12, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--terrain",
    type=str,
    required=True,
    choices=["hurdle", "step", "gap", "stair", "flat", "crawl"],
    help="Which single sub-terrain to isolate (proportion 1.0, all others 0.0).",
)
parser.add_argument("--out_path", type=str, required=True, help="Exact output mp4 path for the recorded video.")
parser.add_argument(
    "--max_init_level",
    type=int,
    default=8,
    help="Cap on the initial terrain level spread. The camera tracks the highest-level target env, "
    "so this sets the difficulty of the recorded course (8 surfaces near-failure difficulty; "
    "lower it to film levels the policy reliably clears).",
)
parser.add_argument(
    "--sensor",
    type=str,
    default="none",
    choices=["none", "height_scan", "clearance", "voxel", "lidar", "lidar_grid", "voxel_compare"],
    help="Which perception-sensor visualization to overlay (goal markers stay OFF). 'none' = no overlay.",
)
# append RSL-RL cli arguments (adds --load_run, --checkpoint, --experiment_name, ...)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
args_cli.enable_cameras = True
args_cli.video = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import os
import shutil

import gymnasium as gym
import torch
from rsl_rl.runners import (
    DistillationRunner,
    OnPolicyRunner,
    OnPolicyRunnerParkour,
    OnPolicyRunnerParkourDistill,
)
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP, OnPolicyRunnerAMPBase
from rsl_rl.runners.on_policy_runner_parkour_amp import (
    OnPolicyRunnerParkourAMP,
    OnPolicyRunnerParkourAMPLidar,
    OnPolicyRunnerParkourAMPVoxel,
)

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# CLI terrain name -> sub_terrains dict key in PARKOUR_TERRAINS_CFG.
TERRAIN_KEY = {
    "flat": "parkour_flat",
    "hurdle": "parkour_hurdle",
    "step": "parkour_step",
    "gap": "parkour_gap",
    "stair": "parkour_stair",
    "crawl": "parkour_crawl",
}


def _isolate_terrain(env_cfg, terrain_name: str) -> None:
    """Isolate a single sub-terrain so ~all of the grid is that terrain type (in place).

    The RandomGoal env family hard-asserts at least one ``parkour_flat`` column exists
    (it is used for AMP flat-column masking), so we cannot set flat to exactly 0.0.
    Instead we reserve exactly one flat column: ``_col_to_class`` (parkour_env.py) maps
    column 0 to flat iff ``flat_prop > 1/num_cols`` boundary — setting flat = 1/num_cols
    yields exactly one flat column (col 0), with the remaining num_cols-1 columns mapped
    to the target terrain. Proportions must sum to 1.0 (±1e-6, asserted in the env).

    When the target itself is ``parkour_flat`` we simply set flat = 1.0.
    """
    target_key = TERRAIN_KEY[terrain_name]
    tg = env_cfg.terrain.terrain_generator
    if target_key not in tg.sub_terrains:
        raise KeyError(
            f"Terrain '{terrain_name}' -> key '{target_key}' not found in sub_terrains "
            f"(available: {list(tg.sub_terrains.keys())})."
        )
    for key in list(tg.sub_terrains.keys()):
        tg.sub_terrains[key].proportion = 0.0

    if target_key == "parkour_flat":
        tg.sub_terrains["parkour_flat"].proportion = 1.0
    else:
        # One reserved flat column (satisfies the AMP flat-column assert); rest = target.
        flat_prop = 1.0 / float(tg.num_cols)
        tg.sub_terrains["parkour_flat"].proportion = flat_prop
        tg.sub_terrains[target_key].proportion = 1.0 - flat_prop

    resolved = {k: v.proportion for k, v in tg.sub_terrains.items() if v.proportion > 0.0}
    print(f"[per-terrain] isolated terrain='{terrain_name}' key='{target_key}'. active proportions: {resolved}")


def use_noninstanceable_go2_for_render(env_cfg) -> None:
    """Swap the stock instanceable ``go2.usd`` for a flattened non-instanceable copy.

    The stock instanceable Unitree Go2 asset triggers an Isaac Sim 6.0 render-path issue
    (IsaacLab #2925 / IsaacSim #227) where visual meshes expand only partially in the
    play/viewer camera path, so recorded videos show a fragmented robot (body / some legs
    missing). Flattening the asset and clearing ``instanceable`` makes every link carry its
    own visual geometry, which renders reliably. Visualization path only; training keeps the
    instanceable asset.
    """
    from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

    robot_cfg = getattr(env_cfg, "robot", None)
    if robot_cfg is None and hasattr(env_cfg, "scene"):
        robot_cfg = getattr(env_cfg.scene, "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None) if robot_cfg is not None else None
    usd_path = getattr(spawn_cfg, "usd_path", None) if spawn_cfg is not None else None
    if spawn_cfg is None or not (isinstance(usd_path, str) and usd_path.endswith("/Robots/Unitree/Go2/go2.usd")):
        return
    noninst_path = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Go2_noninstanceable", "go2.usd")
    if not os.path.isfile(noninst_path):
        print(
            f"[WARN] Non-instanceable Go2 asset not found at {noninst_path}; robot may render fragmented. "
            "Generate it with: ./isaaclab.sh -p scripts/tools/make_go2_noninstanceable.py"
        )
        return
    spawn_cfg.usd_path = noninst_path
    print(f"[per-terrain] Rendering with non-instanceable Go2 asset (Isaac Sim 6.0 render fix): {noninst_path}")


def _pick_target_view_env(env_unwrapped, terrain_name: str) -> int:
    """Return an env id guaranteed to sit on the *target* terrain (not the reserved flat col).

    ``TerrainImporter`` assigns ``terrain_types = floor(arange(num_envs) / (num_envs/num_cols))``,
    so **env 0 always lands in terrain column 0**, which ``_isolate_terrain`` reserves for the
    single flat column (AMP flat-mask assert).  The parkour camera follows
    ``cfg.viewer.env_index`` (default 0) via ``_camera_follow_callback``, so it records that lone
    flat column — the observed bug.

    We instead pick a target-terrain env with the *highest* terrain level so the obstacle is
    prominent: level-0 patches are nearly flat and would still look like flat on camera.
    Returns 0 for ``terrain_name == 'flat'`` (env 0 is the flat column we want).
    """
    env_class = env_unwrapped._env_class  # [num_envs] class id per env
    flat_class = int(env_unwrapped._col_to_class[0].item())  # class of column 0 == flat
    if terrain_name == "flat":
        return 0
    target_mask = env_class != flat_class
    if not bool(target_mask.any()):
        return 0
    levels = env_unwrapped._terrain_levels.clone().float()
    levels[~target_mask] = -1.0  # exclude the flat-column env(s) from the argmax
    return int(torch.argmax(levels).item())


def _make_point_markers(prim_path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    """Single-prototype sphere marker set (mirrors the env's clearance-vis / perception-viz pattern)."""
    cfg = SPHERE_MARKER_CFG.copy()
    cfg.prim_path = prim_path
    cfg.markers["sphere"].radius = radius
    cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkers(cfg)


def _draw_sensor_overlay(env_unwrapped, sensor: str, view_idx: int, hs_vis, lidar_vis, max_lidar_pts: int = 6000):
    """Draw the per-method perception-sensor markers for the *tracked* env (goal markers stay off).

    - clearance / voxel: use the env's own ``_draw_clearance_rays`` / ``_draw_voxel_occupied``,
      which target ``_get_active_viewer_env_id()`` — equal to ``view_idx`` (we set the viewport
      env-index there). They create their own markers lazily and update every call.
    - height_scan / lidar: custom sphere point clouds at the tracked env's ray hits.
    """
    if sensor == "height_scan":
        hs = getattr(env_unwrapped, "_height_scanner", None)
        if hs is None:
            return
        pts = hs.data.ray_hits_w[view_idx]  # (187, 3)
        pts = pts[torch.isfinite(pts).all(dim=-1)]
        if pts.shape[0] > 0:
            hs_vis.set_visibility(True)
            hs_vis.visualize(translations=pts)
    elif sensor == "clearance":
        if hasattr(env_unwrapped, "_draw_clearance_rays"):
            env_unwrapped._draw_clearance_rays()
    elif sensor == "voxel":
        if hasattr(env_unwrapped, "_draw_voxel_occupied"):
            env_unwrapped._draw_voxel_occupied()
    elif sensor == "voxel_compare":
        # GT volume (orange) plus the ray-sampled cells it replaces (magenta), same frame.
        if hasattr(env_unwrapped, "_draw_voxel_compare"):
            env_unwrapped._draw_voxel_compare()
    elif sensor == "lidar_grid":
        # The LiDAR student's own metric occupancy grid (cyan), as opposed to "voxel" which
        # draws the teacher's privileged grid (orange). Only the LiDAR env defines it.
        if hasattr(env_unwrapped, "_draw_lidar_grid"):
            env_unwrapped._draw_lidar_grid()
    elif sensor == "lidar":
        mid = getattr(env_unwrapped, "_mid360", None)
        if mid is None:
            return
        hits = mid.data.ray_hits_w[view_idx]  # (R, 3)
        valid = torch.isfinite(hits).all(dim=-1)
        if hasattr(mid.data, "distances"):
            mid_cfg = getattr(env_unwrapped.cfg, "mid360_lidar", None)
            max_range = float(
                getattr(mid_cfg, "max_distance", None) or getattr(env_unwrapped.cfg, "lidar_max_range", 20.0)
            )
            valid = valid & (mid.data.distances[view_idx] < max_range - 0.1)
        pts = hits[valid]
        if pts.shape[0] > max_lidar_pts:
            pts = pts[torch.randperm(pts.shape[0], device=pts.device)[:max_lidar_pts]]
        if pts.shape[0] > 0:
            lidar_vis.set_visibility(True)
            lidar_vis.visualize(translations=pts)


_RUNNER_CLASSES = {
    "OnPolicyRunner": OnPolicyRunner,
    "OnPolicyRunnerParkour": OnPolicyRunnerParkour,
    "OnPolicyRunnerAMP": OnPolicyRunnerAMP,
    "OnPolicyRunnerAMPBase": OnPolicyRunnerAMPBase,
    "OnPolicyRunnerParkourAMP": OnPolicyRunnerParkourAMP,
    "OnPolicyRunnerParkourAMPVoxel": OnPolicyRunnerParkourAMPVoxel,
    "OnPolicyRunnerParkourAMPLidar": OnPolicyRunnerParkourAMPLidar,
    "DistillationRunner": DistillationRunner,
    "OnPolicyRunnerParkourDistill": OnPolicyRunnerParkourDistill,
}
"""Runner classes selectable through ``agent_cfg.class_name``, mirroring play.py's switch."""

_RAW_CFG_RUNNERS = frozenset({"OnPolicyRunner", "DistillationRunner", "OnPolicyRunnerParkourDistill"})
"""Runners that must receive the unfiltered agent cfg.

The others forward their ``algorithm`` block straight into ``PPOParkour``, so unknown keys raise;
these three read theirs with ``.get()`` defaults instead, and filtering would strip fields they
still need.
"""


def _build_runner(env, agent_cfg: RslRlBaseRunnerCfg):
    """Construct the runner named by ``agent_cfg.class_name``.

    Args:
        env: Wrapped environment the runner will drive.
        agent_cfg: Resolved runner configuration.

    Returns:
        The constructed runner, with no log directory (playback only).

    Raises:
        ValueError: If ``class_name`` is not a supported runner.
    """
    import inspect

    from rsl_rl.algorithms.ppo_parkour import PPOParkour

    name = agent_cfg.class_name
    if name not in _RUNNER_CLASSES:
        raise ValueError(f"Unsupported runner class: {name}")

    cfg = agent_cfg.to_dict()
    if name not in _RAW_CFG_RUNNERS:
        accepted = set(inspect.signature(PPOParkour.__init__).parameters) - {"self"}
        cfg["algorithm"] = {k: v for k, v in cfg["algorithm"].items() if k in accepted or k == "class_name"}
    return _RUNNER_CLASSES[name](env, cfg, log_dir=None, device=agent_cfg.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with an RSL-RL agent on a single isolated terrain and record a video."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve the checkpoint path directly from --load_run/--checkpoint (bypass get_checkpoint_path
    # so the "experiment/run" style --load_run given by the harness resolves unambiguously).
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint) and os.path.isfile(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        # --load_run may be "<experiment>/<run>" or just "<run>"; --checkpoint is the model file name.
        ckpt_name = args_cli.checkpoint if args_cli.checkpoint else "model_49999.pt"
        run_rel = agent_cfg.load_run
        cand = os.path.join("logs", "rsl_rl", run_rel, ckpt_name)
        if not os.path.isfile(cand):
            cand = os.path.join(log_root_path, run_rel, ckpt_name)
        resume_path = os.path.abspath(cand)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # ── terrain isolation (must run before gym.make) ──────────────────────────
    _isolate_terrain(env_cfg, args_cli.terrain)

    # ── Isaac Sim 6.0 render fix: swap to non-instanceable Go2 so the recorded
    #    video shows a whole robot (stock instanceable asset renders fragmented). ──
    use_noninstanceable_go2_for_render(env_cfg)

    # ── Spawn at high difficulty so the recorded gap is prominent (wide gaps that
    #    require real leaping), not the near-flat level-0..3 patches. The camera picks
    #    the highest-level target env, so raising the init spread surfaces a hard gap. ──
    _high = args_cli.max_init_level
    if hasattr(env_cfg, "terrain_max_init_level"):
        env_cfg.terrain_max_init_level = _high
    if hasattr(env_cfg.terrain, "max_init_terrain_level"):
        env_cfg.terrain.max_init_terrain_level = _high
    print(f"[per-terrain] max_init_terrain_level set to {_high} (surface hard gaps for the camera)")

    # Prevent the RandomGoal env from teleporting graduated random-goal envs onto the
    # reserved flat column mid-rollout (``random_goal_force_flat``), which would drag the
    # tracked camera env back onto flat.  Disabling random-goal pins every env to its
    # natural terrain column; the flat-column assert is independent of this flag so the
    # env still constructs.  No-op for tasks that lack the attribute.
    if hasattr(env_cfg, "enable_random_goal"):
        env_cfg.enable_random_goal = False
        print("[per-terrain] enable_random_goal=False (pin envs to natural terrain column)")

    # Turn OFF all parkour goal/edge/height debug markers: the goal-direction dot arrows
    # (cur_goal / future_goals / HeadingDots / TargetDots) and edge-mask spheres are gated
    # by ``debug_vis`` / ``debug_vis_edge_mask``. We replace them with a per-method sensor
    # overlay drawn manually in the rollout loop (see _draw_sensor_overlay).
    env_cfg.debug_vis = False
    if hasattr(env_cfg.scene, "height_scanner") and hasattr(env_cfg.scene.height_scanner, "debug_vis"):
        env_cfg.scene.height_scanner.debug_vis = False
    if hasattr(env_cfg, "debug_vis_edge_mask"):
        env_cfg.debug_vis_edge_mask = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.set_debug_vis(False)  # ensure goal-marker visualizers stay hidden

    # ── retarget the follow camera onto a target-terrain env ──────────────────
    # (env 0 is the reserved flat column; the camera would otherwise record flat.)
    view_idx = _pick_target_view_env(env.unwrapped, args_cli.terrain)
    env.unwrapped.cfg.viewer.env_index = view_idx  # read live each frame by _camera_follow_callback
    _vcc = getattr(env.unwrapped, "viewport_camera_controller", None)
    if _vcc is not None:
        _vcc.cfg.env_index = view_idx  # also drives edge-mask viz + fallback path
    # Headless has no viewport_camera_controller, so the env's _get_active_viewer_env_id()
    # falls back to env 0 — the reserved flat column. The sensor-overlay draw functions use
    # that id to place their markers, which would put every marker on a robot the camera is
    # not filming (silently overlay-free videos). Pin it to the tracked env.
    env.unwrapped._get_active_viewer_env_id = lambda _v=view_idx: _v
    tracked_class = int(env.unwrapped._env_class[view_idx].item())
    flat_class = int(env.unwrapped._col_to_class[0].item())
    tracked_level = int(env.unwrapped._terrain_levels[view_idx].item())
    print(
        f"[per-terrain] camera tracks env={view_idx}: terrain_class={tracked_class} "
        f"(flat_class={flat_class}), terrain_level={tracked_level}, terrain='{args_cli.terrain}'"
    )

    # ── perception-sensor overlay markers (goal markers already OFF) ───────────
    # height_scan / lidar need custom point-cloud markers; clearance / voxel use the env's
    # own lazily-created marker sets inside _draw_clearance_rays / _draw_voxel_occupied.
    hs_vis = lidar_vis = None
    if args_cli.sensor == "height_scan":
        hs_vis = _make_point_markers("/Visuals/Viz/height_scan", (1.0, 0.85, 0.0), radius=0.03)  # yellow
    elif args_cli.sensor == "lidar":
        lidar_vis = _make_point_markers("/Visuals/Viz/lidar", (0.7, 0.1, 0.9), radius=0.02)  # purple
    print(f"[per-terrain] sensor overlay = '{args_cli.sensor}'")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ── video recording into a temp folder, then rename to the exact out_path ──
    out_path = os.path.abspath(args_cli.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_video_dir = os.path.join(
        os.path.dirname(out_path), ".rec_tmp_" + os.path.basename(out_path).replace(".mp4", "")
    )
    if os.path.isdir(tmp_video_dir):
        shutil.rmtree(tmp_video_dir)
    os.makedirs(tmp_video_dir, exist_ok=True)
    video_kwargs = {
        "video_folder": tmp_video_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print(f"[per-terrain] recording video -> {out_path}")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[per-terrain] loading checkpoint: {resume_path} (runner={agent_cfg.class_name})")

    # 6.0 migration: IsaacLab 3.0 (rsl-rl-lib 5.0.1) adds algorithm-cfg fields (e.g. optimizer,
    # share_cnn_encoders) that the vendored 3.2.0 PPOParkour.__init__ does not accept. Filter the
    # algorithm cfg to the vendored PPOParkour signature (mirrors play.py's ParkourAMP branch).
    runner = _build_runner(env, agent_cfg)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # NOTE: JIT/ONNX export intentionally skipped (crashes on voxel/lidar nets; unneeded for video).

    # Go2Recovery-style settle disable guard (no-op for parkour tasks lacking the attribute).
    if hasattr(env.unwrapped.cfg, "settle_max_steps"):
        env.unwrapped.cfg.settle_max_steps = 0

    # reset environment
    obs = env.get_observations()

    # ── forward-motion liveness metric ─────────────────────────────────────────
    # Reset-invariant: average body-frame forward velocity over the rollout.
    # A collapsed/never-standing policy stays near 0; a walking one is ~1 m/s.
    fwd_vel_sum = 0.0
    fwd_vel_count = 0

    # ── tracked-env base-z series ───────────────────────────────────────────────
    # Proves the *recorded* env traverses an obstacle: flat gait keeps z ~constant,
    # climbing/crawling makes z vary. Uses view_idx only (NOT a global mean).
    z_series: list[float] = []

    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # goal-based parkour tasks generate goals on reset; velocity command block is a no-op
            # for them but retained for command-based tasks that may reuse this script.
            if hasattr(env.unwrapped, "_commands") and env.unwrapped._commands.shape[1] >= 3:
                env.unwrapped._commands[:, 0] = 1.0
                env.unwrapped._commands[:, 1] = 0.0
                env.unwrapped._commands[:, 2] = 0.0

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            # accumulate forward-velocity liveness metric
            robot = getattr(env.unwrapped, "_robot", None)
            if robot is not None and hasattr(robot.data, "root_lin_vel_b"):
                fwd_vel_sum += float(robot.data.root_lin_vel_b[:, 0].mean().item())
                fwd_vel_count += 1
            if robot is not None and hasattr(robot.data, "root_pos_w"):
                z_series.append(float(robot.data.root_pos_w[view_idx, 2].item()))

            # draw the method's perception-sensor overlay for the tracked env.
            # The marker prims must be followed by an explicit render pass: without it the
            # overlay never reaches the recorded frames (the capture happens inside env.step,
            # before these writes land), which silently produces overlay-free videos.
            if args_cli.sensor != "none":
                _draw_sensor_overlay(env.unwrapped, args_cli.sensor, view_idx, hs_vis, lidar_vis)
                env.unwrapped.sim.render()

        timestep += 1
        if timestep == args_cli.video_length:
            break

    mean_fwd_vel = fwd_vel_sum / max(fwd_vel_count, 1)
    print(
        f"[per-terrain] LIVENESS terrain={args_cli.terrain} steps={fwd_vel_count} "
        f"mean_forward_vel_b_x={mean_fwd_vel:.3f} m/s"
    )

    # ── tracked-env base-z verdict (obstacle traversal on the recorded env) ─────
    if z_series:
        z_t = torch.tensor(z_series)
        z_min = float(z_t.min())
        z_max = float(z_t.max())
        z_std = float(z_t.std())
        z_rng = z_max - z_min
        # flat gait keeps z within a narrow crouch band (range < ~0.05m). Stepping over an
        # obstacle (hurdle/step/stair climb, gap dip, crawl crouch) lifts/drops the base well
        # beyond that; 0.06m separates the two robustly (a level-3 hurdle already gives ~0.08m).
        verdict = "OBSTACLE-VARIATION" if z_rng > 0.06 else "FLAT-LIKE(SUSPECT)"
        print(
            f"[per-terrain] TRACKED_Z terrain={args_cli.terrain} env={view_idx} "
            f"class={tracked_class} level={tracked_level} z_min={z_min:.3f} z_max={z_max:.3f} "
            f"z_range={z_rng:.3f} z_std={z_std:.3f} -> {verdict}"
        )

    env.close()

    # move the recorded mp4 to the exact requested path
    produced = sorted(glob.glob(os.path.join(tmp_video_dir, "*.mp4")))
    if produced:
        if os.path.isfile(out_path):
            os.remove(out_path)
        shutil.move(produced[0], out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[per-terrain] SAVED {out_path} ({size_mb:.2f} MB)")
        shutil.rmtree(tmp_video_dir, ignore_errors=True)
    else:
        print(f"[per-terrain] ERROR: no mp4 produced in {tmp_video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
