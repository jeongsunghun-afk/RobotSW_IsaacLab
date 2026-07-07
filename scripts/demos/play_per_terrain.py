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
parser.add_argument("--out_path", type=str, required=True, help="Exact output mp4 path for the recorded video.")
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
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerParkour
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP, OnPolicyRunnerAMPBase
from rsl_rl.runners.on_policy_runner_parkour_amp import (
    OnPolicyRunnerParkourAMP,
    OnPolicyRunnerParkourAMPLidar,
    OnPolicyRunnerParkourAMPVoxel,
)

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

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

    # enable debug visualizations exactly like play.py (edge mask / height scanner rays)
    env_cfg.debug_vis = True
    if hasattr(env_cfg.scene, "height_scanner") and hasattr(env_cfg.scene.height_scanner, "debug_vis"):
        env_cfg.scene.height_scanner.debug_vis = True
    if hasattr(env_cfg, "debug_vis_edge_mask"):
        env_cfg.debug_vis_edge_mask = True

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.set_debug_vis(getattr(env_cfg, "debug_vis", True))

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ── video recording into a temp folder, then rename to the exact out_path ──
    out_path = os.path.abspath(args_cli.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_video_dir = os.path.join(os.path.dirname(out_path), ".rec_tmp_" + os.path.basename(out_path).replace(".mp4", ""))
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
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkour":
        runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAMP":
        runner = OnPolicyRunnerAMP(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAMPBase":
        runner = OnPolicyRunnerAMPBase(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMP":
        runner = OnPolicyRunnerParkourAMP(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMPVoxel":
        runner = OnPolicyRunnerParkourAMPVoxel(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMPLidar":
        runner = OnPolicyRunnerParkourAMPLidar(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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

    # reset environment
    obs = env.get_observations()

    # ── forward-motion liveness metric ─────────────────────────────────────────
    # Reset-invariant: average body-frame forward velocity over the rollout.
    # A collapsed/never-standing policy stays near 0; a walking one is ~1 m/s.
    fwd_vel_sum = 0.0
    fwd_vel_count = 0

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

        timestep += 1
        if timestep == args_cli.video_length:
            break

    mean_fwd_vel = fwd_vel_sum / max(fwd_vel_count, 1)
    print(
        f"[per-terrain] LIVENESS terrain={args_cli.terrain} steps={fwd_vel_count} "
        f"mean_forward_vel_b_x={mean_fwd_vel:.3f} m/s"
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
