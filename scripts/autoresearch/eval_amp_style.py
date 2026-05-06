# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP Style-Primary Evaluator for autoresearch.

CLI:
    ./isaaclab.sh -p scripts/autoresearch/eval_amp_style.py \
        --task Isaac-Go2-AMP-Direct-v0 \
        --checkpoint <PATH> \
        --cmd-grid "vx=0.5,1.0,1.5,2.0;vy=0;wz=-0.5,0,0.5" \
        --num-seeds 4 \
        --rollout-steps 500 \
        --seed-list-mode primary \
        --output-json <PATH>

Outputs:
    JSON file at --output-json with required `pass: bool` and `score: float` keys
    (autoresearch SKILL contract).
    Stdout: EVAL_SCORE=<float>, DISC_COLLAPSED=<bool>

Evaluator version: 1.0.0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AMP style-primary evaluator for autoresearch.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. Isaac-Go2-AMP-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Absolute or relative path to checkpoint .pt file.")
parser.add_argument(
    "--cmd-grid",
    type=str,
    default="vx=0.5,1.0,1.5,2.0;vy=0;wz=-0.5,0,0.5",
    help="Command grid string, e.g. 'vx=0.5,1.0;vy=0;wz=-0.5,0,0.5'",
)
parser.add_argument("--num-seeds", type=int, default=4, help="Number of seeds per command combination.")
parser.add_argument("--rollout-steps", type=int, default=500, help="Number of policy steps per (cmd, seed) rollout.")
parser.add_argument(
    "--seed-list-mode",
    type=str,
    default="primary",
    choices=["primary", "disjoint"],
    help="'primary' uses seeds [0,1,2,3], 'disjoint' uses [10,11,12,13].",
)
parser.add_argument("--output-json", type=str, required=True, help="Path to write JSON output.")

# Headless is forced; no video/cameras needed.
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Force headless — evaluator never renders.
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import itertools
import json
import math
import os
import tempfile
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):  # type: ignore[attr-defined]
    skrl.logger.error(  # type: ignore[attr-defined]
        f"Unsupported skrl version: {skrl.__version__}. "  # type: ignore[attr-defined]
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    sys.exit(1)

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

EVALUATOR_VERSION = "1.0.0"
LAMBDA_TRACK = 0.1  # tracking penalty weight (style dominant ~6×)
DISC_STD_COLLAPSE_THRESH = 1e-3
DISC_ABS_COLLAPSE_THRESH = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_cmd_grid(grid_str: str) -> list[dict[str, float]]:
    """Parse 'vx=0.5,1.0;vy=0;wz=-0.5,0,0.5' into list of (vx,vy,wz) dicts."""
    axis_values: dict[str, list[float]] = {}
    for segment in grid_str.strip().split(";"):
        segment = segment.strip()
        if not segment:
            continue
        key, vals_str = segment.split("=", 1)
        key = key.strip()
        axis_values[key] = [float(v) for v in vals_str.split(",")]

    vx_list = axis_values.get("vx", [0.0])
    vy_list = axis_values.get("vy", [0.0])
    wz_list = axis_values.get("wz", [0.0])

    combos = []
    for vx, vy, wz in itertools.product(vx_list, vy_list, wz_list):
        combos.append({"vx": vx, "vy": vy, "wz": wz})
    return combos


def seed_list(mode: str, num_seeds: int) -> list[int]:
    """Return the fixed seed list based on mode."""
    if mode == "primary":
        return list(range(num_seeds))           # [0, 1, 2, 3]
    elif mode == "disjoint":
        return list(range(10, 10 + num_seeds))  # [10, 11, 12, 13]
    else:
        raise ValueError(f"Unknown seed_list_mode: {mode}")


def atomic_write_json(data: dict, path: str) -> None:
    """Write JSON atomically via tempfile + rename."""
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, allow_nan=True)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Main (wrapped by hydra_task_config)
# ---------------------------------------------------------------------------

# AMP entry point uses "skrl_amp_cfg_entry_point"
agent_cfg_entry_point = "skrl_amp_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, experiment_cfg: dict):
    """Run the AMP style evaluator."""
    t_start = time.time()

    # ------------------------------------------------------------------
    # Parse CLI arguments
    # ------------------------------------------------------------------
    cmd_combos = parse_cmd_grid(args_cli.cmd_grid)
    seeds = seed_list(args_cli.seed_list_mode, args_cli.num_seeds)
    rollout_steps = args_cli.rollout_steps
    checkpoint_path = os.path.abspath(args_cli.checkpoint)

    print(f"[EVAL] Task:         {args_cli.task}")
    print(f"[EVAL] Checkpoint:   {checkpoint_path}")
    print(f"[EVAL] Cmd combos:   {len(cmd_combos)}")
    print(f"[EVAL] Seeds:        {seeds}")
    print(f"[EVAL] Rollout steps:{rollout_steps}")

    # ------------------------------------------------------------------
    # Override env_cfg for evaluation
    # ------------------------------------------------------------------
    # Use 1 env; command curriculum off; resample disabled.
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Fix command range to a single default value; we override _commands directly at rollout time.
    # Disable curriculum (command_cfg range narrowed to zero width — doesn't matter since we
    # override self._commands post-reset directly).
    # command_resample_interval: the env resamples in _reset_idx only, not mid-rollout,
    # so fixing _commands after each reset is sufficient.

    # Disable early termination during evaluation to allow full rollout steps.
    env_cfg.early_termination = False

    # ------------------------------------------------------------------
    # Build environment (single sim instance, reused across all rollouts)
    # ------------------------------------------------------------------
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")

    # ------------------------------------------------------------------
    # Load runner + checkpoint
    # ------------------------------------------------------------------
    runner = Runner(env, experiment_cfg)
    print(f"[EVAL] Loading checkpoint: {checkpoint_path}")
    runner.agent.load(checkpoint_path)
    runner.agent.enable_training_mode(False)

    # Grab references for discriminator forward
    agent = runner.agent
    discriminator = agent.discriminator
    amp_preprocessor = agent._amp_observation_preprocessor
    device = agent.device

    # ------------------------------------------------------------------
    # Evaluation loop: outer=cmd, inner=seed
    # ------------------------------------------------------------------
    per_cmd_seed_results: dict[str, dict[str, dict]] = {}

    for cmd in cmd_combos:
        vx, vy, wz = cmd["vx"], cmd["vy"], cmd["wz"]
        cmd_key = f"vx_{vx}_wz_{wz}"
        per_cmd_seed_results[cmd_key] = {}

        for seed in seeds:
            torch.manual_seed(seed)

            # Reset env
            obs, info = env.reset()

            # Fix the command to (vx, vy, wz) by directly writing to the unwrapped env's buffer.
            # env -> SkrlVecEnvWrapper (IsaacLabWrapper) -> gym.Wrapper -> DirectRLEnv
            unwrapped_env = gym_env.unwrapped
            unwrapped_env._commands[:] = torch.tensor(
                [[vx, vy, wz]], dtype=torch.float32, device=unwrapped_env.device
            )

            # Accumulation buffers
            logits_buffer: list[torch.Tensor] = []
            style_rewards: list[float] = []
            tracking_errs: list[float] = []

            with torch.inference_mode():
                for _step in range(rollout_steps):
                    # Policy step — deterministic (mean_actions)
                    actions, outputs = agent.act(obs, states=None, timestep=0, timesteps=0)
                    mean_actions = outputs.get("mean_actions", actions)

                    # Env step
                    obs, _reward, _terminated, _truncated, info = env.step(mean_actions)

                    # Re-fix command after every step (in case reset happened mid-rollout)
                    unwrapped_env._commands[:] = torch.tensor(
                        [[vx, vy, wz]], dtype=torch.float32, device=unwrapped_env.device
                    )

                    # --- Extract amp_obs ---
                    # env._get_observations() sets self.extras["amp_obs"] which becomes info["amp_obs"]
                    # after gym step. info here is the raw extras dict from DirectRLEnv.
                    amp_obs = info.get("amp_obs", None)
                    if amp_obs is None:
                        # Fallback: read directly from unwrapped env's buffer
                        amp_obs = unwrapped_env.amp_observation_buffer.view(-1, unwrapped_env.amp_observation_size)

                    # --- Discriminator forward (pattern from amp.py:387-392) ---
                    amp_logits, _ = discriminator.act(
                        {"observations": amp_preprocessor(amp_obs)}, role="discriminator"
                    )
                    # amp_logits shape: (1, 1) or (1,) — squeeze to scalar for accumulation
                    logit_val = amp_logits.squeeze()
                    logits_buffer.append(logit_val.detach().cpu())

                    # --- Style reward (amp.py:390-392 exact formula) ---
                    style_rew = -torch.log(
                        torch.maximum(
                            1 - 1 / (1 + torch.exp(-amp_logits)),
                            torch.tensor(1e-4, device=device),
                        )
                    )
                    style_rewards.append(style_rew.mean().item())

                    # --- Tracking error ---
                    root_lin_vel_b = unwrapped_env.robot.data.root_lin_vel_b  # (1, 3)
                    root_ang_vel_b = unwrapped_env.robot.data.root_ang_vel_b  # (1, 3)

                    v_xy_actual = root_lin_vel_b[0, :2]  # (2,)
                    w_z_actual = root_ang_vel_b[0, 2]    # scalar

                    v_xy_cmd = torch.tensor([vx, vy], dtype=torch.float32, device=device)
                    w_z_cmd = torch.tensor(wz, dtype=torch.float32, device=device)

                    track_err = (
                        torch.norm(v_xy_actual - v_xy_cmd).item() + torch.abs(w_z_actual - w_z_cmd).item()
                    )
                    tracking_errs.append(track_err)

            # --- Per (cmd, seed) aggregation ---
            logits_tensor = torch.stack(logits_buffer)  # (rollout_steps,)
            logits_std = logits_tensor.std().item()
            logits_abs_mean = logits_tensor.abs().mean().item()

            disc_collapsed = (logits_std < DISC_STD_COLLAPSE_THRESH) or (logits_abs_mean > DISC_ABS_COLLAPSE_THRESH)

            style_mean = float(sum(style_rewards) / len(style_rewards))
            track_mean = float(sum(tracking_errs) / len(tracking_errs))

            per_cmd_seed_results[cmd_key][str(seed)] = {
                "style_mean": style_mean,
                "track_mean": track_mean,
                "disc_collapsed": disc_collapsed,
                "logits_std": logits_std,
                "logits_abs_mean": logits_abs_mean,
            }

    # ------------------------------------------------------------------
    # Aggregate across all (cmd, seed) combinations
    # ------------------------------------------------------------------
    any_disc_collapsed = False
    per_cmd_breakdown: dict[str, dict] = {}

    # Per-seed score: mean over cmds of (style - 0.1 * track) for each seed
    per_seed_scores: dict[str, list[float]] = {str(s): [] for s in seeds}

    all_style_means: list[float] = []
    all_track_means: list[float] = []

    for cmd_key, seed_results in per_cmd_seed_results.items():
        cmd_styles = []
        cmd_tracks = []
        cmd_collapsed = False

        for seed_str, metrics in seed_results.items():
            cmd_styles.append(metrics["style_mean"])
            cmd_tracks.append(metrics["track_mean"])
            if metrics["disc_collapsed"]:
                cmd_collapsed = True
                any_disc_collapsed = True

            per_seed_scores[seed_str].append(metrics["style_mean"] - LAMBDA_TRACK * metrics["track_mean"])

        cmd_style_mean = float(sum(cmd_styles) / len(cmd_styles)) if cmd_styles else float("nan")
        cmd_track_mean = float(sum(cmd_tracks) / len(cmd_tracks)) if cmd_tracks else float("nan")

        per_cmd_breakdown[cmd_key] = {
            "style": cmd_style_mean,
            "track": cmd_track_mean,
            "disc_collapsed": cmd_collapsed,
        }

        all_style_means.append(cmd_style_mean)
        all_track_means.append(cmd_track_mean)

    global_style_mean = float(sum(all_style_means) / len(all_style_means)) if all_style_means else float("nan")
    global_track_mean = float(sum(all_track_means) / len(all_track_means)) if all_track_means else float("nan")

    # Final scalar
    if any_disc_collapsed:
        score = float("-inf")
    else:
        score = global_style_mean - LAMBDA_TRACK * global_track_mean

    # Std across seeds (per-seed score = mean over cmds of (style - 0.1*track))
    seed_score_means = []
    for seed_str in [str(s) for s in seeds]:
        vals = per_seed_scores[seed_str]
        if vals:
            seed_score_means.append(sum(vals) / len(vals))
    if len(seed_score_means) >= 2:
        n = len(seed_score_means)
        mean_s = sum(seed_score_means) / n
        variance = sum((v - mean_s) ** 2 for v in seed_score_means) / (n - 1)
        score_std = math.sqrt(variance)
    elif len(seed_score_means) == 1:
        score_std = 0.0
    else:
        score_std = float("nan")

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Build JSON output (autoresearch contract)
    # ------------------------------------------------------------------
    # `pass` is always false here — autoresearch compares to baseline and sets pass.
    # Exception: if disc_collapsed, explicitly false (and score=-inf already signals it).
    result = {
        "pass": False,
        "score": score,
        "score_style": global_style_mean,
        "score_track": global_track_mean,
        "score_std_across_seeds": score_std,
        "disc_collapsed": any_disc_collapsed,
        "seeds_used": seeds,
        "cmd_grid": [args_cli.cmd_grid],
        "rollout_steps": rollout_steps,
        "per_cmd_breakdown": per_cmd_breakdown,
        "checkpoint": checkpoint_path,
        "task": args_cli.task,
        "elapsed_seconds": elapsed,
        "evaluator_version": EVALUATOR_VERSION,
    }

    output_path = os.path.abspath(args_cli.output_json)
    atomic_write_json(result, output_path)

    # ------------------------------------------------------------------
    # Human-readable stdout
    # ------------------------------------------------------------------
    print(f"EVAL_SCORE={score}")
    print(f"DISC_COLLAPSED={any_disc_collapsed}")
    print(f"[EVAL] style_mean={global_style_mean:.4f}  track_mean={global_track_mean:.4f}  std={score_std:.4f}")
    print(f"[EVAL] JSON written to: {output_path}")
    print(f"[EVAL] Elapsed: {elapsed:.1f}s")

    env.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore[call-arg]
    finally:
        simulation_app.close()
