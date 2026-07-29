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
import sys

from isaaclab.app import AppLauncher

# local imports — reuse the RSL-RL CLI arg helpers from the play/train scripts.
import os as _os_boot

sys.path.insert(0, _os_boot.path.join(_os_boot.path.dirname(_os_boot.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"))
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
parser.add_argument("--out_path", type=str, required=True, help="Output JSON path for the probe statistics.")
parser.add_argument("--level", type=int, required=True, help="Terrain level to pin ALL envs to (0..num_rows-1).")
parser.add_argument("--steps", type=int, default=3000, help="Rollout steps (episodes are counted as they end).")
parser.add_argument(
    "--sensor",
    type=str,
    default="none",
    choices=["none", "height_scan", "clearance", "voxel", "lidar"],
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

import json
import os

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerParkour
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
from isaaclab_tasks.direct.parkour.parkour_env_cfg import (
    TERRAIN_CLASS_CRAWL,
    TERRAIN_CLASS_FLAT,
    TERRAIN_CLASS_GAP,
    TERRAIN_CLASS_HURDLE,
    TERRAIN_CLASS_STAIR,
    TERRAIN_CLASS_STEP,
)
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


_TERRAIN_CLASS_BY_NAME = {
    "flat": TERRAIN_CLASS_FLAT,
    "hurdle": TERRAIN_CLASS_HURDLE,
    "step": TERRAIN_CLASS_STEP,
    "gap": TERRAIN_CLASS_GAP,
    "stair": TERRAIN_CLASS_STAIR,
    "crawl": TERRAIN_CLASS_CRAWL,
}


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
    elif sensor == "lidar":
        mid = getattr(env_unwrapped, "_mid360", None)
        if mid is None:
            return
        hits = mid.data.ray_hits_w[view_idx]  # (R, 3)
        valid = torch.isfinite(hits).all(dim=-1)
        if hasattr(mid.data, "distances"):
            mid_cfg = getattr(env_unwrapped.cfg, "mid360_lidar", None)
            max_range = float(getattr(mid_cfg, "max_distance", None) or getattr(env_unwrapped.cfg, "lidar_max_range", 20.0))
            valid = valid & (mid.data.distances[view_idx] < max_range - 0.1)
        pts = hits[valid]
        if pts.shape[0] > max_lidar_pts:
            pts = pts[torch.randperm(pts.shape[0], device=pts.device)[:max_lidar_pts]]
        if pts.shape[0] > 0:
            lidar_vis.set_visibility(True)
            lidar_vis.visualize(translations=pts)


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
    _high = 8
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

    out_path = os.path.abspath(args_cli.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[per-terrain] loading checkpoint: {resume_path} (runner={agent_cfg.class_name})")

    # 6.0 migration: IsaacLab 3.0 (rsl-rl-lib 5.0.1) adds algorithm-cfg fields (e.g. optimizer,
    # share_cnn_encoders) that the vendored 3.2.0 PPOParkour.__init__ does not accept. Filter the
    # algorithm cfg to the vendored PPOParkour signature (mirrors play.py's ParkourAMP branch).
    import inspect as _inspect

    from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPOParkour

    _accepted = set(_inspect.signature(_VendoredPPOParkour.__init__).parameters.keys()) - {"self"}
    _cfg_dict = agent_cfg.to_dict()
    _cfg_dict["algorithm"] = {
        k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted or k == "class_name"
    }

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkour":
        runner = OnPolicyRunnerParkour(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAMP":
        runner = OnPolicyRunnerAMP(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAMPBase":
        runner = OnPolicyRunnerAMPBase(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMP":
        runner = OnPolicyRunnerParkourAMP(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMPVoxel":
        runner = OnPolicyRunnerParkourAMPVoxel(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMPLidar":
        runner = OnPolicyRunnerParkourAMPLidar(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
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

    # ── pin every env to the requested difficulty row of the target terrain ────
    # The whole point of the probe: curriculum level is the confounder we are removing.
    # Both policies get the same rows/columns, so any difference is the policy, not its
    # self-selected difficulty.
    u = env.unwrapped
    origins = u._terrain.terrain_origins  # (num_rows, num_cols, 3)
    num_rows = int(origins.shape[0])
    level = max(0, min(args_cli.level, num_rows - 1))
    target_class = _TERRAIN_CLASS_BY_NAME[args_cli.terrain]
    cols = (u._col_to_class == target_class).nonzero(as_tuple=False).flatten()
    if cols.numel() == 0:
        raise RuntimeError(f"no column of class {target_class} ({args_cli.terrain}) exists")
    for i in range(u.num_envs):
        col = int(cols[i % cols.numel()].item())
        u._terrain_levels[i] = level
        u._terrain_types[i] = col
        u._env_class[i] = u._col_to_class[col]
        u._terrain.env_origins[i] = origins[level, col]
        u._skip_curriculum[i] = True
    u._reset_idx(torch.arange(u.num_envs, device=u.device))
    print(f"[probe] pinned {u.num_envs} envs to terrain='{args_cli.terrain}' level={level} (rows={num_rows})")

    obs = env.get_observations()

    # ── accumulate episode outcomes ────────────────────────────────────────────
    # Read from extras["log"], which the env fills at reset time. Reading the _term_*
    # flags after env.step() would be wrong: _reset_idx clears them before we see them
    # (documented gotcha from the Teacher3D goal-rate work).
    TERM_KEYS = [
        "Episode_Termination/cause_goal_reached",
        "Episode_Termination/cause_tilt",
        "Episode_Termination/cause_low_height",
        "Episode_Termination/cause_base_contact",
        "Episode_Termination/time_out",
    ]
    REW_KEYS = ["Episode_Reward/collision", "Episode_Reward/tracking_goal_vel", "Episode_Reward/feet_edge"]
    counts = {k: 0.0 for k in TERM_KEYS}
    rew_sum = {k: 0.0 for k in REW_KEYS}
    rew_n = 0
    fwd_vel_sum, fwd_vel_count = 0.0, 0
    goal_idx_sum, goal_idx_n = 0.0, 0

    for _ in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            policy_nn.reset(dones)

            robot = getattr(u, "_robot", None)
            if robot is not None:
                fwd_vel_sum += float(robot.data.root_lin_vel_b[:, 0].mean().item())
                fwd_vel_count += 1
            # how far along the goal chain the envs are (progress proxy, pre-reset)
            if hasattr(u, "_current_goal_idx"):
                goal_idx_sum += float(u._current_goal_idx.float().mean().item())
                goal_idx_n += 1

            # extras["log"] is only refreshed inside _reset_idx, but the dict itself persists,
            # so reading it every step would re-count the same episodes. Gate on an actual reset.
            if int(dones.sum().item()) > 0:
                log = (extras or {}).get("log", {})
                for k in TERM_KEYS:
                    if k in log:
                        counts[k] += float(log[k])
                hit = False
                for k in REW_KEYS:
                    if k in log:
                        rew_sum[k] += float(log[k])
                        hit = True
                if hit:
                    rew_n += 1

    mean_fwd_vel = fwd_vel_sum / max(fwd_vel_count, 1)
    total_ep = sum(counts.values())
    stats = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "terrain": args_cli.terrain,
        "level": level,
        "num_envs": int(u.num_envs),
        "steps": args_cli.steps,
        "episodes": total_ep,
        "mean_forward_vel": mean_fwd_vel,
        "mean_goal_idx": goal_idx_sum / max(goal_idx_n, 1),
        "terminations": {k.split("/")[-1]: counts[k] for k in TERM_KEYS},
        "termination_frac": {
            k.split("/")[-1]: (counts[k] / total_ep if total_ep else 0.0) for k in TERM_KEYS
        },
        "rewards": {k.split("/")[-1]: (rew_sum[k] / max(rew_n, 1)) for k in REW_KEYS},
    }
    print("[probe] " + json.dumps(stats, ensure_ascii=False))
    with open(out_path, "w") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(f"[probe] SAVED {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
