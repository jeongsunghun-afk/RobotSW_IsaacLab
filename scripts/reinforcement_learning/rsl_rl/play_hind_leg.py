# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play / visualise a trained RSL-RL checkpoint for the HindLeg biped task.

This is a dedicated test/visualisation script for ``HindLeg-Direct-v0``:
  * loads a trained checkpoint and runs the policy in inference,
  * keeps the camera tracking the robot base (so the recorded video follows it),
  * optionally records an mp4 of the rollout,
  * collects per-joint position / velocity / torque every step and plots them
    (each joint labelled by its articulation joint name) after the rollout.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RSL-RL HindLeg checkpoint and plot joint data.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--steps", type=int, default=500, help="Number of inference steps to run / collect for plotting."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments (adds --checkpoint, --load_run, --experiment_name, ...)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video (and to drive the tracking camera under headless)
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerParkour
from rsl_rl.runners.on_policy_runner_parkour_amp import OnPolicyRunnerParkourAMP

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    ViewerCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_jit_parkour,
    export_policy_as_onnx,
    export_policy_as_onnx_parkour,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def plot_joint_data(joint_names, joint_pos, joint_vel, joint_torque, save_path):
    """Plot per-joint position / velocity / torque vs. timestep and save as a png.

    Layout: one row per joint, three columns (position, velocity, torque). Every subplot title
    names the joint so it is unambiguous which joint each curve belongs to.

    Parameters:
        joint_names (list[str]): Joint names in articulation data-tensor column order.
        joint_pos (np.ndarray): Shape (num_steps, num_joints), joint positions [rad].
        joint_vel (np.ndarray): Shape (num_steps, num_joints), joint velocities [rad/s].
        joint_torque (np.ndarray): Shape (num_steps, num_joints), applied torques [N*m].
        save_path (str): Full path to write the png.
    """
    num_joints = len(joint_names)
    if num_joints == 0 or joint_pos.shape[0] == 0:
        print("[WARN] No joint data collected; skipping plot generation.")
        return

    timesteps = np.arange(joint_pos.shape[0])
    metrics = [
        ("joint position", joint_pos, "position [rad]"),
        ("joint velocity", joint_vel, "velocity [rad/s]"),
        ("torque", joint_torque, "torque [N*m]"),
    ]

    fig, axes = plt.subplots(num_joints, 3, figsize=(15, 2.6 * num_joints), squeeze=False)
    for row, name in enumerate(joint_names):
        for col, (metric_label, data, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            ax.plot(timesteps, data[:, row], linewidth=1.0)
            ax.set_title(f"{name} - {metric_label}")
            ax.set_xlabel("timestep")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved joint data plot to: {save_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with the trained RSL-RL HindLeg agent and collect joint telemetry."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")  # noqa: F841

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # camera tracks the robot base: built-in ViewportCameraController follows asset "robot" in env 0.
    # origin_type="asset_root" keeps the eye/lookat offset relative to the robot root each frame, so
    # both the live viewport and the recorded video follow the robot. No env-file change required.
    env_cfg.viewer = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        eye=(0.0, -2.5, 0.8),
        lookat=(0.0, 0.0, 0.3),
    )

    # create isaac environment
    env_cfg.debug_vis = True
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if hasattr(env.unwrapped, "set_debug_vis"):
        env.unwrapped.set_debug_vis(getattr(env_cfg, "debug_vis", True))

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video of the rollout.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    print(agent_cfg.class_name)
    # load previously trained model (HindLeg default cfg uses OnPolicyRunnerParkour)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkour":
        runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMP":
        runner = OnPolicyRunnerParkourAMP(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module (try-except for backwards compatibility)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if agent_cfg.class_name in ("OnPolicyRunnerParkour", "OnPolicyRunnerParkourAMP"):
        estimator = getattr(runner.alg, "estimator", None)
        export_policy_as_jit_parkour(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt", estimator=estimator
        )
        export_policy_as_onnx_parkour(
            policy_nn, path=export_model_dir, normalizer=normalizer, filename="policy.onnx", estimator=estimator
        )
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # robot articulation handle + joint names in data-tensor column order (safe runtime query).
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    print(f"[INFO] Articulation joint order ({len(joint_names)} joints):")
    for idx, jname in enumerate(joint_names):
        print(f"  {idx:02d}: {jname}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # per-step telemetry for env 0 (collected as python lists, stacked after the loop).
    pos_history: list[np.ndarray] = []
    vel_history: list[np.ndarray] = []
    torque_history: list[np.ndarray] = []

    # total steps to run: at least --steps, and never shorter than the requested video length.
    total_steps = max(args_cli.steps, args_cli.video_length if args_cli.video else 0)
    print(f"[INFO] Running {total_steps} inference steps...")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # forward command (the env zeroes swing/gait when the command is near-zero, so the
            # robot would just stand still without an explicit non-zero forward command).
            env.unwrapped._commands[:, 0] = 0.0
            env.unwrapped._commands[:, 1] = 0.0
            env.unwrapped._commands[:, 2] = 0.0

            actions = policy(obs)

            # collect joint telemetry for env 0 before stepping.
            pos_history.append(robot.data.joint_pos[0].cpu().numpy().copy())
            vel_history.append(robot.data.joint_vel[0].cpu().numpy().copy())
            torque = getattr(robot.data, "applied_torque", None)
            if torque is None:
                torque = robot.data.computed_torque
            torque_history.append(torque[0].cpu().numpy().copy())

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        timestep += 1
        if timestep >= total_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    # plot collected joint telemetry.
    joint_pos = np.asarray(pos_history)
    joint_vel = np.asarray(vel_history)
    joint_torque = np.asarray(torque_history)
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "hind_leg_joint_data.png")
    plot_joint_data(joint_names, joint_pos, joint_vel, joint_torque, plot_path)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
