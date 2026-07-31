# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DTC Phase-2 ablation gate (Step 8) for Quad17-Velocity-Direct-v0.

Loads a trained foothold-tracking policy and measures, under a fixed forward command, the per-
touchdown foothold-tracking reward and XY placement error. The 28-dim foothold obs block is
zeroed/scrambled via the env var QUAD17_FOOTHOLD_ABLATE ("" | "zero" | "scramble"), which the env
reads at construction. The reward is always scored against the TRUE target buffer, so a DROP in
foothold_track / RISE in placement error under ablation proves the policy actually consumes the
reference channel (rather than exploiting a redundant cmd/clock shortcut).

Run the same checkpoint twice (ablate unset vs "scramble") and compare the printed [ABLATION] line.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Quad17 DTC-P2 foothold ablation gate.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Quad17-Velocity-Direct-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed (same for both ablation runs).")
parser.add_argument("--warmup", type=int, default=150, help="Settle steps before metrics start accumulating.")
parser.add_argument("--steps", type=int, default=1500, help="Metric-accumulation steps.")
parser.add_argument("--cmd_vx", type=float, default=0.8, help="Fixed forward command (m/s) applied every step.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunnerParkour

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    base = env.unwrapped
    ablate_mode = getattr(base, "_foothold_ablate", "") or "none"
    print(f"[ABLATION] checkpoint={resume_path}")
    print(f"[ABLATION] ablate_mode={ablate_mode} cmd_vx={args_cli.cmd_vx} warmup={args_cli.warmup} steps={args_cli.steps}")

    obs = env.get_observations()

    def _apply_cmd():
        base._commands[:, 0] = args_cli.cmd_vx
        base._commands[:, 1] = 0.0
        base._commands[:, 2] = 0.0

    with torch.inference_mode():
        # settle under the fixed command
        for _ in range(args_cli.warmup):
            _apply_cmd()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        # zero the (non-reset) eval accumulators, then measure
        base._eval_td_count.zero_()
        base._eval_track_sum.zero_()
        base._eval_err_sum.zero_()
        for _ in range(args_cli.steps):
            _apply_cmd()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

    td = base._eval_td_count.item()
    track = base._eval_track_sum.item()
    err2 = base._eval_err_sum.item()
    mean_track = track / td if td > 0 else float("nan")
    mean_err2 = err2 / td if td > 0 else float("nan")
    mean_err_cm = (mean_err2**0.5) * 100.0 if td > 0 else float("nan")
    print(
        f"[ABLATION-RESULT] mode={ablate_mode} touchdowns={int(td)} "
        f"mean_foothold_track={mean_track:.4f} mean_err2_m2={mean_err2:.5f} rms_err_cm={mean_err_cm:.2f}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
