"""Safety baseline measurement script for Go2Recovery-v0.

Measures peak joint velocity, peak applied torque, peak joint acceleration,
peak CoM speed, action rate, and recovery time for fallen-start environments.

Also measures upright attainment vs. strict success, per-condition bottleneck
analysis (conditions 1/2/3 separately), and vel_RMS distribution ("trembling")
to diagnose "stands up but trembles" failure mode.

Usage:
    # settle OFF (pure policy behavior, no settle phase):
    ./isaaclab.sh -p _workspace/go2_recovery/measure_safety.py --headless \\
        --settle_steps 0

    # settle ON (use cfg default = 100 steps):
    ./isaaclab.sh -p _workspace/go2_recovery/measure_safety.py --headless \\
        --settle_steps -1

    # settle ON with explicit override:
    ./isaaclab.sh -p _workspace/go2_recovery/measure_safety.py --headless \\
        --settle_steps 50

    # full example with checkpoint:
    ./isaaclab.sh -p _workspace/go2_recovery/measure_safety.py --headless \\
        --checkpoint logs/rsl_rl/go2_recovery_direct/2026-06-16_18-25-50_update_recovery/model_9999.pt \\
        --num_envs 256 --num_steps 1000 --settle_steps 0

Purpose: Stage II (safe, slow recovery) design quantitative baseline.
         Diagnose "upright but trembling" failure mode separating:
           - upright attainment rate (ever crossed threshold)
           - per-condition bottleneck (cond1 upright / cond2 pose / cond3 vel_rms)
           - vel_RMS distribution vs. success threshold 2.0 rad/s
"""

import argparse
import sys

# ── 1. Argparse before AppLauncher (AppLauncher.add_app_launcher_args must come first) ──

parser = argparse.ArgumentParser(description="Go2Recovery safety baseline measurement.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rsl_rl/go2_recovery_direct/2026-06-16_18-25-50_update_recovery/model_9999.pt",
    help="Path to checkpoint (.pt). Relative to IsaacLab root or absolute.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel envs.")
parser.add_argument("--num_steps", type=int, default=1000, help="Rollout length (policy steps).")
parser.add_argument(
    "--settle_steps",
    type=int,
    default=-1,
    help=(
        "settle_max_steps override. "
        "-1 = use cfg default (settle ON, cfg.settle_max_steps). "
        "0  = disable settle (eval mode, measure pure recovery). "
        ">0 = override to this value."
    ),
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# clear extra args so Hydra/OmegaConf is not confused
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Post-launch imports ────────────────────────────────────────────────────

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401  (registers all tasks)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# ── 3. Output directory ───────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_FILE = os.path.join(_SCRIPT_DIR, "_safety_baseline.txt")

# ── 4. Main ──────────────────────────────────────────────────────────────────


@hydra_task_config("Go2Recovery-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Measure safety KPIs for the trained Go2Recovery policy."""

    # ── 4-a. Env cfg overrides ────────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs

    # settle_steps argument controls settle_max_steps:
    #   -1 → use cfg default (settle enabled as per training)
    #   0  → disable settle (eval mode: measure pure recovery behavior)
    #   >0 → override to given value
    if args_cli.settle_steps >= 0:
        env_cfg.settle_max_steps = args_cli.settle_steps
    actual_settle = env_cfg.settle_max_steps

    # Keep terminate_on_success=True so recovery_time is measurable (episode ends on success)
    env_cfg.terminate_on_success = True

    # Device
    device = agent_cfg.device if hasattr(agent_cfg, "device") else "cuda:0"
    env_cfg.sim.device = device

    # ── 4-b. Create env ───────────────────────────────────────────────────────
    env = gym.make("Go2Recovery-v0", cfg=env_cfg)
    raw_env = env.unwrapped  # type: ignore[attr-defined]

    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", 100.0))

    # ── 4-c. Load checkpoint ──────────────────────────────────────────────────
    checkpoint_path = args_cli.checkpoint
    if not os.path.isabs(checkpoint_path):
        # Resolve relative to IsaacLab root (cwd when launched via isaaclab.sh)
        checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

    print(f"[measure_safety] Loading checkpoint: {checkpoint_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path)

    # Deterministic inference: disable action noise
    policy = runner.get_inference_policy(device=device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Disable exploration noise for deterministic evaluation
    if hasattr(policy_nn, "std"):
        policy_nn.std.data.fill_(0.0)

    step_dt: float = raw_env.step_dt  # seconds per policy step (decimation already folded in)
    success_hold_steps: int = raw_env.cfg.success_hold_steps
    num_envs: int = raw_env.num_envs

    # Success condition thresholds (mirror env._update_success exactly)
    success_cos_threshold: float = raw_env.cfg.success_cos_threshold   # 0.809
    success_pose_eps: float = raw_env.cfg.success_pose_eps              # 0.5  → threshold = eps^2
    success_vel_eps: float = raw_env.cfg.success_vel_eps               # 2.0 rad/s

    # joint_weights: (1, 12) same as env self._joint_weights
    joint_weights: torch.Tensor = raw_env._joint_weights  # (1, 12) on device

    # Window for "episode tail" statistics (~last 50 policy steps or episode length, whichever is smaller)
    TAIL_WINDOW: int = 50

    # Joint names for labeling (read once — same across all envs)
    joint_names: list[str] = raw_env._robot.data.joint_names  # list of 12 strings
    num_joints: int = len(joint_names)
    # default_joint_pos: (num_envs, 12) → take row 0 as the universal default (same for all envs)
    default_joint_pos_1d: torch.Tensor = raw_env._robot.data.default_joint_pos[0].clone()  # (12,)

    print(f"[measure_safety] step_dt={step_dt:.4f}s  success_hold_steps={success_hold_steps}")
    print(f"[measure_safety] num_envs={num_envs}  num_steps={args_cli.num_steps}")
    print(f"[measure_safety] settle_max_steps={actual_settle}")
    print(f"[measure_safety] success thresholds: cos>{success_cos_threshold:.3f}  "
          f"pose_err<{success_pose_eps**2:.4f}  vel_rms<{success_vel_eps:.2f} rad/s")

    # ── 4-d. Tracking buffers ─────────────────────────────────────────────────
    # Per-env episode tracking
    recovery_start_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    in_episode = torch.zeros(num_envs, dtype=torch.bool, device=device)
    max_joint_vel = torch.zeros(num_envs, device=device)
    max_applied_torque = torch.zeros(num_envs, device=device)
    max_dof_acc = torch.zeros(num_envs, device=device)
    max_root_speed = torch.zeros(num_envs, device=device)
    max_action_rate = torch.zeros(num_envs, device=device)
    prev_joint_vel = torch.zeros(num_envs, 12, device=device)
    prev_action = torch.zeros(num_envs, 12, device=device)

    # Upright tracking: "ever reached upright threshold during episode"
    upright_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Tail window accumulators for per-condition bottleneck and vel_RMS
    # We maintain a rolling buffer of last TAIL_WINDOW steps for each env.
    # For efficiency: accumulate running sums for the last TAIL_WINDOW steps.
    # Buffer shape: (num_envs, TAIL_WINDOW) — circular buffer approach would be complex;
    # instead we track the most recent TAIL_WINDOW readings in a deque-style tensor.
    tail_cos: torch.Tensor = torch.zeros(num_envs, TAIL_WINDOW, device=device)
    tail_pose_err: torch.Tensor = torch.zeros(num_envs, TAIL_WINDOW, device=device)
    tail_vel_rms: torch.Tensor = torch.zeros(num_envs, TAIL_WINDOW, device=device)
    tail_ptr = torch.zeros(num_envs, dtype=torch.long, device=device)   # circular write pointer
    tail_count = torch.zeros(num_envs, dtype=torch.long, device=device) # how many valid entries

    # Per-joint pose deviation buffer: stores joint_pos for the tail window.
    # Shape: (num_envs, TAIL_WINDOW, num_joints)
    # We also track per-step cos_dist to filter upright-only steps during aggregation.
    # (tail_cos already holds cos_dist — reused at episode end for upright masking)
    tail_joint_pos: torch.Tensor = torch.zeros(num_envs, TAIL_WINDOW, num_joints, device=device)

    # Episode-level result lists (fallen-start episodes only)
    results_recovery_time: list[float] = []
    results_peak_jvel: list[float] = []
    results_peak_torque: list[float] = []
    results_peak_dof_acc: list[float] = []
    results_peak_root_speed: list[float] = []
    results_peak_action_rate: list[float] = []
    results_success: list[int] = []
    results_timeout_peak_jvel: list[float] = []

    # New: upright attainment per episode
    results_upright_ever: list[int] = []        # 1 if upright was reached at any point
    results_upright_final: list[float] = []     # cos_dist at episode end (last tail step)

    # New: per-condition satisfaction rate at episode end (tail average)
    results_cond1_frac: list[float] = []   # fraction of tail steps where cond1 (upright) met
    results_cond2_frac: list[float] = []   # fraction of tail steps where cond2 (pose) met
    results_cond3_frac: list[float] = []   # fraction of tail steps where cond3 (vel_rms) met

    # New: tail vel_RMS statistics
    results_tail_vel_rms_mean: list[float] = []   # mean vel_RMS over tail window
    results_tail_vel_rms_max: list[float] = []    # max vel_RMS over tail window

    # Per-joint pose deviation: collect (mean signed dev, mean abs dev, weighted err contrib) per episode.
    # Each entry is a (num_joints,) numpy array; aggregated after rollout.
    results_joint_signed_dev: list[np.ndarray] = []   # mean(actual - default) per joint [rad]
    results_joint_abs_dev: list[np.ndarray] = []      # mean|actual - default| per joint [rad]
    results_joint_weighted_err: list[np.ndarray] = [] # mean(w_i^2 * (actual - default)^2) per joint

    # ── 4-e. Initial reset ────────────────────────────────────────────────────
    obs = env.get_observations()
    policy_nn.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    # Identify fallen-start envs after initial reset
    started_fallen: torch.Tensor = raw_env._started_fallen.clone()  # (num_envs,) bool
    in_episode[started_fallen] = True
    recovery_start_step[started_fallen] = 0

    # ── 4-f. Rollout loop ─────────────────────────────────────────────────────
    print("[measure_safety] Starting rollout ...")
    t0 = time.time()

    for step_idx in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)  # (num_envs, 12)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # ── 4-f-i. Gather raw env state ──────────────────────────────────────
        robot_data = raw_env._robot.data

        joint_vel: torch.Tensor = robot_data.joint_vel          # (num_envs, 12) [rad/s]
        applied_torque: torch.Tensor = robot_data.applied_torque  # (num_envs, 12) [Nm]

        if hasattr(robot_data, "root_lin_vel_w"):
            root_lin_vel: torch.Tensor = robot_data.root_lin_vel_w
        else:
            root_lin_vel = robot_data.root_lin_vel_b
        root_speed = root_lin_vel.norm(dim=1)  # (num_envs,)

        dof_acc = (joint_vel - prev_joint_vel) / step_dt  # (num_envs, 12) [rad/s²]
        prev_joint_vel = joint_vel.clone()

        act: torch.Tensor = actions
        action_rate = (act - prev_action).abs().max(dim=1).values
        prev_action = act.clone()

        # ── 4-f-i-b. Compute success condition signals (matches env._update_success) ──
        # cos_dist = -projected_gravity_b[:,2]
        cos_dist: torch.Tensor = -robot_data.projected_gravity_b[:, 2]  # (num_envs,)

        # cond1: upright
        cond1: torch.Tensor = cos_dist > success_cos_threshold  # (num_envs,) bool

        # cond2: near-default-pose — pose_err = sum(weights^2 * (default - pos)^2)
        #   threshold = success_pose_eps^2  (same as env: cond_pose = pose_err < eps^2)
        pose_diff = robot_data.default_joint_pos - robot_data.joint_pos  # (num_envs, 12)
        pose_err: torch.Tensor = torch.sum(joint_weights**2 * pose_diff**2, dim=1)  # (num_envs,)
        cond2: torch.Tensor = pose_err < (success_pose_eps**2)  # (num_envs,) bool

        # cond3: low joint vel RMS — sqrt(mean(vel^2))
        vel_rms: torch.Tensor = torch.sqrt(torch.mean(joint_vel**2, dim=1))  # (num_envs,)
        cond3: torch.Tensor = vel_rms < success_vel_eps  # (num_envs,) bool

        # ── 4-f-ii. Update per-env running maxima (only for tracked episodes) ─
        active = in_episode

        max_joint_vel = torch.where(active, torch.maximum(max_joint_vel, joint_vel.abs().max(dim=1).values), max_joint_vel)
        max_applied_torque = torch.where(active, torch.maximum(max_applied_torque, applied_torque.abs().max(dim=1).values), max_applied_torque)
        max_dof_acc = torch.where(active, torch.maximum(max_dof_acc, dof_acc.abs().max(dim=1).values), max_dof_acc)
        max_root_speed = torch.where(active, torch.maximum(max_root_speed, root_speed), max_root_speed)
        max_action_rate = torch.where(active, torch.maximum(max_action_rate, action_rate), max_action_rate)

        # upright_ever: set True once cos_dist exceeds threshold (for tracked episodes)
        upright_ever = upright_ever | (active & cond1)

        # ── 4-f-ii-b. Update tail circular buffer for tracked envs ───────────
        # Only update for active (tracked) envs to avoid contaminating episode boundary.
        # We write to the slot pointed by tail_ptr for each active env.
        # Vectorized: scatter via advanced indexing.
        active_ids = active.nonzero(as_tuple=False).squeeze(1)  # indices of tracked envs
        if len(active_ids) > 0:
            ptrs = tail_ptr[active_ids]  # (n_active,)
            tail_cos[active_ids, ptrs] = cos_dist[active_ids]
            tail_pose_err[active_ids, ptrs] = pose_err[active_ids]
            tail_vel_rms[active_ids, ptrs] = vel_rms[active_ids]
            # Per-joint position for pose deviation analysis
            tail_joint_pos[active_ids, ptrs] = robot_data.joint_pos[active_ids]
            # Advance pointer (circular, mod TAIL_WINDOW)
            tail_ptr[active_ids] = (ptrs + 1) % TAIL_WINDOW
            # Increment count (cap at TAIL_WINDOW)
            tail_count[active_ids] = torch.clamp(tail_count[active_ids] + 1, max=TAIL_WINDOW)

        # ── 4-f-iii. Check for episode completions (dones) ───────────────────
        completed = dones.bool()
        fallen_completed = completed & in_episode

        if fallen_completed.any():
            ids_completed = fallen_completed.nonzero(as_tuple=False).squeeze(1)
            success_now: torch.Tensor = raw_env._current_success

            ids_computed = ids_completed.tolist()
            for i in ids_computed:
                env_i = int(i)
                succeeded = bool(success_now[env_i].item())
                results_success.append(int(succeeded))

                results_peak_jvel.append(float(max_joint_vel[env_i].item()))
                results_peak_torque.append(float(max_applied_torque[env_i].item()))
                results_peak_dof_acc.append(float(max_dof_acc[env_i].item()))
                results_peak_root_speed.append(float(max_root_speed[env_i].item()))
                results_peak_action_rate.append(float(max_action_rate[env_i].item()))

                if succeeded:
                    start = int(recovery_start_step[env_i].item())
                    recovery_steps = (step_idx - start + 1)
                    recovery_time_s = recovery_steps * step_dt
                    results_recovery_time.append(recovery_time_s)
                else:
                    results_timeout_peak_jvel.append(float(max_joint_vel[env_i].item()))

                # Upright attainment
                results_upright_ever.append(int(upright_ever[env_i].item()))

                # Tail window analysis: use valid entries from circular buffer
                n_valid = int(tail_count[env_i].item())
                if n_valid > 0:
                    ptr = int(tail_ptr[env_i].item())
                    # The valid slots in circular buffer (most recent n_valid entries).
                    # All TAIL_WINDOW slots are valid if count==TAIL_WINDOW; otherwise first n_valid.
                    # Circular buffer: last n_valid entries end at ptr-1 (mod TAIL_WINDOW).
                    # Easiest: just use all valid entries (tail_cos[env_i, :n_valid] when not wrapped,
                    # or all TAIL_WINDOW when wrapped). Since we only need the last TAIL_WINDOW,
                    # all entries in the buffer are valid tail data.
                    tail_c = tail_cos[env_i, :n_valid if n_valid < TAIL_WINDOW else TAIL_WINDOW]
                    tail_p = tail_pose_err[env_i, :n_valid if n_valid < TAIL_WINDOW else TAIL_WINDOW]
                    tail_v = tail_vel_rms[env_i, :n_valid if n_valid < TAIL_WINDOW else TAIL_WINDOW]

                    results_cond1_frac.append(float((tail_c > success_cos_threshold).float().mean().item()))
                    results_cond2_frac.append(float((tail_p < success_pose_eps**2).float().mean().item()))
                    results_cond3_frac.append(float((tail_v < success_vel_eps).float().mean().item()))
                    results_tail_vel_rms_mean.append(float(tail_v.mean().item()))
                    results_tail_vel_rms_max.append(float(tail_v.max().item()))

                    # final cos_dist: last entry written before episode end
                    last_slot = (ptr - 1) % TAIL_WINDOW
                    results_upright_final.append(float(tail_cos[env_i, last_slot].item()))

                    # ── Per-joint pose deviation (upright steps only) ─────────
                    # Filter tail steps where cos_dist > success_cos_threshold (standing posture).
                    # tail_c already holds the per-slot cos_dist values.
                    n_slots = n_valid if n_valid < TAIL_WINDOW else TAIL_WINDOW
                    upright_mask = tail_c > success_cos_threshold  # (n_slots,) bool
                    n_upright_steps = int(upright_mask.sum().item())
                    if n_upright_steps > 0:
                        # joint_pos for upright slots: (n_upright_steps, num_joints)
                        jp_slots = tail_joint_pos[env_i, :n_slots]  # (n_slots, num_joints)
                        jp_upright = jp_slots[upright_mask]          # (n_upright_steps, num_joints)
                        # signed deviation: actual - default  (positive = overshoots default)
                        dev = jp_upright - default_joint_pos_1d.unsqueeze(0)  # (n_upright_steps, 12)
                        signed_dev = dev.mean(dim=0).cpu().numpy()   # (12,)
                        abs_dev = dev.abs().mean(dim=0).cpu().numpy()  # (12,)
                        # weighted error contribution per joint: w_i^2 * mean((actual-default)^2)
                        w = joint_weights[0].cpu().numpy()  # (12,)
                        weighted_err_per_joint = (w ** 2) * (dev ** 2).mean(dim=0).cpu().numpy()  # (12,)
                        results_joint_signed_dev.append(signed_dev)
                        results_joint_abs_dev.append(abs_dev)
                        results_joint_weighted_err.append(weighted_err_per_joint)
                    # If no upright steps in this episode tail, skip (don't append nan arrays —
                    # the lists will be shorter than results_success, which is intentional).
                else:
                    # No tail data (episode was too short)
                    results_cond1_frac.append(float("nan"))
                    results_cond2_frac.append(float("nan"))
                    results_cond3_frac.append(float("nan"))
                    results_tail_vel_rms_mean.append(float("nan"))
                    results_tail_vel_rms_max.append(float("nan"))
                    results_upright_final.append(float("nan"))

        # ── 4-f-iv. Reset tracking for newly reset envs ───────────────────────
        if completed.any():
            reset_ids = completed.nonzero(as_tuple=False).squeeze(1)
            in_episode[reset_ids] = False
            max_joint_vel[reset_ids] = 0.0
            max_applied_torque[reset_ids] = 0.0
            max_dof_acc[reset_ids] = 0.0
            max_root_speed[reset_ids] = 0.0
            max_action_rate[reset_ids] = 0.0
            upright_ever[reset_ids] = False
            tail_cos[reset_ids] = 0.0
            tail_pose_err[reset_ids] = 0.0
            tail_vel_rms[reset_ids] = 0.0
            tail_joint_pos[reset_ids] = 0.0
            tail_ptr[reset_ids] = 0
            tail_count[reset_ids] = 0

            new_fallen = raw_env._started_fallen[reset_ids]
            newly_fallen_global = reset_ids[new_fallen]

            if len(newly_fallen_global) > 0:
                in_episode[newly_fallen_global] = True
                recovery_start_step[newly_fallen_global] = step_idx + 1

        if (step_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_ep = len(results_success)
            print(
                f"  step {step_idx + 1:5d}/{args_cli.num_steps}  "
                f"episodes_collected={n_ep:4d}  elapsed={elapsed:.1f}s"
            )

    total_time = time.time() - t0
    print(f"[measure_safety] Rollout done in {total_time:.1f}s")

    # ── 4-g. Statistics ───────────────────────────────────────────────────────

    def _stats(arr: list[float], name: str = "") -> dict[str, float]:
        if not arr:
            return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"),
                    "max": float("nan"), "min": float("nan"), "n": 0}
        a = np.array(arr, dtype=np.float64)
        valid = a[~np.isnan(a)]
        if len(valid) == 0:
            return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"),
                    "max": float("nan"), "min": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "p95": float(np.percentile(valid, 95)),
            "p50": float(np.percentile(valid, 50)),
            "p5": float(np.percentile(valid, 5)),
            "max": float(np.max(valid)),
            "min": float(np.min(valid)),
            "n": int(len(valid)),
        }

    n_total_fallen = len(results_success)
    n_success = sum(results_success)
    n_timeout = n_total_fallen - n_success
    success_rate = n_success / n_total_fallen if n_total_fallen > 0 else float("nan")

    # Upright ever: episodes where upright was reached at any point
    n_upright_ever = sum(results_upright_ever)
    upright_ever_rate = n_upright_ever / n_total_fallen if n_total_fallen > 0 else float("nan")

    rt_stats = _stats(results_recovery_time)
    jv_stats = _stats(results_peak_jvel)
    tq_stats = _stats(results_peak_torque)
    da_stats = _stats(results_peak_dof_acc)
    rs_stats = _stats(results_peak_root_speed)
    ar_stats = _stats(results_peak_action_rate)

    # Per-condition fractions (tail window, fallen episodes only)
    c1_mean = float(np.nanmean(results_cond1_frac)) if results_cond1_frac else float("nan")
    c2_mean = float(np.nanmean(results_cond2_frac)) if results_cond2_frac else float("nan")
    c3_mean = float(np.nanmean(results_cond3_frac)) if results_cond3_frac else float("nan")

    # Vel RMS tail distribution
    vrms_stats = _stats(results_tail_vel_rms_mean)
    # Distribution of tail vel_rms_mean relative to threshold 2.0
    if results_tail_vel_rms_mean:
        vrms_arr = np.array([v for v in results_tail_vel_rms_mean if not np.isnan(v)])
        n_vrms = len(vrms_arr)
        n_below = int(np.sum(vrms_arr < success_vel_eps))
        # Buckets for vel_RMS mean
        vrms_buckets = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]
    else:
        vrms_arr = np.array([], dtype=np.float64)
        n_vrms = 0
        n_below = 0
        vrms_buckets = []

    # Final cos_dist distribution
    cos_final_stats = _stats(results_upright_final)

    # ── 4-h. Report ───────────────────────────────────────────────────────────

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  Go2Recovery-v0  Safety Baseline")
    lines.append(f"  Checkpoint : {args_cli.checkpoint}")
    lines.append(f"  num_envs   : {num_envs}   num_steps  : {args_cli.num_steps}")
    lines.append(f"  step_dt    : {step_dt:.4f} s   (policy 50 Hz, decimation 4)")
    if args_cli.settle_steps >= 0:
        lines.append(f"  settle_max_steps overridden to {actual_settle}")
    else:
        lines.append(f"  settle_max_steps = {actual_settle} (cfg default, not overridden)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Fallen-start episodes collected : {n_total_fallen}")
    lines.append(f"  Strict success (3 conditions)   : {n_success}   ({success_rate * 100:.1f}%)")
    lines.append(f"  Timeout (no strict success)     : {n_timeout}")
    lines.append("")

    # ── UPRIGHT ATTAINMENT — new section ─────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  UPRIGHT ATTAINMENT  (separated from strict success)")
    lines.append("=" * 72)
    lines.append(f"  Upright-ever rate (cos>{success_cos_threshold:.3f} reached once) : "
                 f"{n_upright_ever}/{n_total_fallen}  ({upright_ever_rate * 100:.1f}%)")
    lines.append(f"  Strict success rate (all 3 cond, {success_hold_steps} consecutive steps):  "
                 f"{n_success}/{n_total_fallen}  ({success_rate * 100:.1f}%)")
    if n_total_fallen > 0:
        gap = upright_ever_rate - success_rate
        lines.append(f"  Gap (upright_ever - strict_success)                  : "
                     f"{gap * 100:.1f} pp")
        lines.append(f"  → If gap is large, robot reaches upright but fails")
        lines.append(f"    to hold all 3 conditions for {success_hold_steps} steps.")
    lines.append("")
    lines.append(f"  Final cos_dist (last tail step, mean/p50/p95/min, all fallen episodes):")
    lines.append(f"    mean={cos_final_stats.get('mean', float('nan')):.3f}  "
                 f"p50={cos_final_stats.get('p50', float('nan')):.3f}  "
                 f"p95={cos_final_stats.get('p95', float('nan')):.3f}  "
                 f"min={cos_final_stats.get('min', float('nan')):.3f}")
    lines.append(f"    (1.0 = fully upright, 0.809 = success_cos_threshold, 0.0 = horizontal)")
    lines.append("")

    # ── PER-CONDITION BOTTLENECK — new section ────────────────────────────────
    lines.append("=" * 72)
    lines.append("  PER-CONDITION SATISFACTION  (tail window = last ~50 policy steps)")
    lines.append("  Measures fraction of tail steps where each condition is met.")
    lines.append("  Computed per fallen-start episode, then averaged across episodes.")
    lines.append("=" * 72)
    lines.append(f"  Cond 1 — Upright   (cos_dist > {success_cos_threshold:.3f})    avg frac : {c1_mean * 100:.1f}%")
    lines.append(f"  Cond 2 — Pose      (pose_err < {success_pose_eps**2:.4f})     avg frac : {c2_mean * 100:.1f}%")
    lines.append(f"  Cond 3 — Vel RMS   (vel_rms < {success_vel_eps:.1f} rad/s)    avg frac : {c3_mean * 100:.1f}%")
    lines.append("")
    lines.append("  Bottleneck diagnosis:")
    conds = [("Cond 1 (upright)", c1_mean), ("Cond 2 (pose)", c2_mean), ("Cond 3 (vel_rms)", c3_mean)]
    sorted_conds = sorted(conds, key=lambda x: x[1])
    lines.append(f"    Primary bottleneck (lowest satisfaction): {sorted_conds[0][0]} ({sorted_conds[0][1]*100:.1f}%)")
    if len(sorted_conds) > 1:
        lines.append(f"    Secondary bottleneck                   : {sorted_conds[1][0]} ({sorted_conds[1][1]*100:.1f}%)")
    lines.append("")
    lines.append("  Success condition thresholds used (matches env._update_success exactly):")
    lines.append(f"    cos_threshold = {success_cos_threshold}  (cfg.success_cos_threshold)")
    lines.append(f"    pose_err threshold = {success_pose_eps}^2 = {success_pose_eps**2:.4f}  "
                 f"(cfg.success_pose_eps squared)")
    lines.append(f"    vel_rms threshold  = {success_vel_eps}  (cfg.success_vel_eps)")
    lines.append(f"    joint_weights      = hip {raw_env._joint_weights.max().item():.1f} / "
                 f"thigh {raw_env._joint_weights[0, raw_env._thigh_joint_ids[0]].item():.2f} / "
                 f"calf {raw_env._joint_weights[0, raw_env._calf_joint_ids[0]].item():.2f}  "
                 f"(matches env self._joint_weights)")
    lines.append("")

    # ── TREMBLING (vel_RMS) DISTRIBUTION — new section ───────────────────────
    lines.append("=" * 72)
    lines.append("  JOINT VELOCITY RMS  (vel_RMS, 'trembling' indicator)")
    lines.append(f"  tail-window mean vel_RMS per episode vs. threshold {success_vel_eps:.1f} rad/s")
    lines.append("=" * 72)
    lines.append(f"    n episodes with tail data : {vrms_stats.get('n', 0)}")
    lines.append(f"    mean                      : {vrms_stats.get('mean', float('nan')):.3f} rad/s")
    lines.append(f"    p50 (median)              : {vrms_stats.get('p50', float('nan')):.3f} rad/s")
    lines.append(f"    p95                       : {vrms_stats.get('p95', float('nan')):.3f} rad/s")
    lines.append(f"    max                       : {vrms_stats.get('max', float('nan')):.3f} rad/s")
    lines.append(f"    min                       : {vrms_stats.get('min', float('nan')):.3f} rad/s")
    lines.append(f"    below threshold (<{success_vel_eps:.1f})     : "
                 f"{n_below}/{n_vrms}  ({n_below/n_vrms*100:.1f}% of episodes)" if n_vrms > 0 else "    (no data)")
    lines.append("")
    if len(vrms_arr) > 0:
        lines.append("  Distribution of tail mean vel_RMS:")
        prev_b = 0.0
        for b in vrms_buckets:
            cnt = int(np.sum(vrms_arr <= b)) if b != float("inf") else len(vrms_arr)
            pct = cnt / len(vrms_arr) * 100
            label = f"<={b:.1f}" if b != float("inf") else "  all"
            lines.append(f"    {label:7s} rad/s : {cnt:4d} / {len(vrms_arr):4d}  ({pct:5.1f}%)")
    lines.append("")
    lines.append("  NOTE: vel_RMS = sqrt(mean(joint_vel²)) over 12 joints,")
    lines.append("        computed at each policy step, then averaged over tail window.")
    lines.append("        Cond 3 uses per-step vel_rms; here we report the tail-window mean.")
    lines.append("")

    # ── RECOVERY TIME ─────────────────────────────────────────────────────────
    lines.append("─" * 72)
    lines.append("  RECOVERY TIME  (success-only)   [seconds]")
    lines.append("─" * 72)
    if results_recovery_time:
        lines.append(f"    n         : {rt_stats['n']}")
        lines.append(f"    mean      : {rt_stats['mean']:.2f} s")
        lines.append(f"    median    : {rt_stats.get('p50', float('nan')):.2f} s")
        lines.append(f"    p95       : {rt_stats['p95']:.2f} s")
        lines.append(f"    max       : {rt_stats['max']:.2f} s")
        lines.append(f"    min       : {rt_stats['min']:.2f} s")
        lines.append(f"    [design target: 6~8 s]")
        buckets = [2, 4, 6, 8, 10, 12, float("inf")]
        arr_rt = np.array(results_recovery_time)
        lines.append("    distribution (cumulative %):")
        for b in buckets:
            cnt = int(np.sum(arr_rt <= b))
            pct = cnt / len(arr_rt) * 100
            label = f"<={b:.0f}s" if b != float("inf") else "  all"
            lines.append(f"      {label:8s}: {cnt:4d} / {len(arr_rt):4d}  ({pct:5.1f}%)")
    else:
        lines.append("    (no successful episodes)")
    lines.append("")

    # ── PEAK JOINT VELOCITY ────────────────────────────────────────────────────
    lines.append("─" * 72)
    lines.append("  PEAK JOINT VELOCITY            [rad/s]   (all fallen episodes)")
    lines.append("─" * 72)
    lines.append(f"    n         : {jv_stats.get('n', 0)}")
    lines.append(f"    mean      : {jv_stats.get('mean', float('nan')):.3f}")
    lines.append(f"    median    : {jv_stats.get('p50', float('nan')):.3f}")
    lines.append(f"    p95       : {jv_stats.get('p95', float('nan')):.3f}")
    lines.append(f"    max       : {jv_stats.get('max', float('nan')):.3f}")
    lines.append(f"    [actuator vel_limit: 18.0 rad/s (Tier-0)]")
    lines.append("")

    # ── PEAK APPLIED TORQUE ───────────────────────────────────────────────────
    lines.append("─" * 72)
    lines.append("  PEAK APPLIED TORQUE            [Nm]      (all fallen episodes)")
    lines.append("─" * 72)
    lines.append(f"    n         : {tq_stats.get('n', 0)}")
    lines.append(f"    mean      : {tq_stats.get('mean', float('nan')):.3f}")
    lines.append(f"    median    : {tq_stats.get('p50', float('nan')):.3f}")
    lines.append(f"    p95       : {tq_stats.get('p95', float('nan')):.3f}")
    lines.append(f"    max       : {tq_stats.get('max', float('nan')):.3f}")
    lines.append(f"    [effort_limit: 23.5 Nm (DCMotor)]")
    lines.append("")

    # ── PEAK JOINT ACCELERATION ───────────────────────────────────────────────
    lines.append("─" * 72)
    lines.append("  PEAK JOINT ACCELERATION (|Δvel/dt|) [rad/s²] (all fallen)")
    lines.append("─" * 72)
    lines.append(f"    n         : {da_stats.get('n', 0)}")
    lines.append(f"    mean      : {da_stats.get('mean', float('nan')):.1f}")
    lines.append(f"    median    : {da_stats.get('p50', float('nan')):.1f}")
    lines.append(f"    p95       : {da_stats.get('p95', float('nan')):.1f}")
    lines.append(f"    max       : {da_stats.get('max', float('nan')):.1f}")
    lines.append(f"    note: computed as (joint_vel_t - joint_vel_{{t-1}}) / {step_dt:.4f}")
    lines.append("")

    # ── PEAK BASE (CoM) SPEED ─────────────────────────────────────────────────
    lines.append("─" * 72)
    lines.append("  PEAK BASE (CoM) SPEED          [m/s]    (all fallen episodes)")
    lines.append("─" * 72)
    lines.append(f"    n         : {rs_stats.get('n', 0)}")
    lines.append(f"    mean      : {rs_stats.get('mean', float('nan')):.3f}")
    lines.append(f"    median    : {rs_stats.get('p50', float('nan')):.3f}")
    lines.append(f"    p95       : {rs_stats.get('p95', float('nan')):.3f}")
    lines.append(f"    max       : {rs_stats.get('max', float('nan')):.3f}")
    lines.append(f"    source    : root_lin_vel_w norm (world-frame CoM speed)")
    lines.append("")

    # ── ACTION RATE ───────────────────────────────────────────────────────────
    # ── Per-joint pose deviation aggregation ──────────────────────────────────
    # Each list entry is a (12,) array; stack into (N_episodes, 12) then average.
    if results_joint_signed_dev:
        jdev_signed_mat = np.stack(results_joint_signed_dev, axis=0)   # (N, 12)
        jdev_abs_mat    = np.stack(results_joint_abs_dev,    axis=0)   # (N, 12)
        jdev_werr_mat   = np.stack(results_joint_weighted_err, axis=0) # (N, 12)

        mean_signed = jdev_signed_mat.mean(axis=0)   # (12,)
        mean_abs    = jdev_abs_mat.mean(axis=0)       # (12,)
        mean_werr   = jdev_werr_mat.mean(axis=0)      # (12,)

        # Sort by mean abs deviation (descending) for "top offender" ranking
        abs_rank = np.argsort(mean_abs)[::-1]         # indices, largest first
        werr_rank = np.argsort(mean_werr)[::-1]

        n_upright_eps = len(results_joint_signed_dev)
    else:
        mean_signed = mean_abs = mean_werr = None
        abs_rank = werr_rank = None
        n_upright_eps = 0

    lines.append("=" * 72)
    lines.append("  PER-JOINT POSE DEVIATION  (standing, tail window, upright steps only)")
    lines.append(f"  Episodes with ≥1 upright tail step : {n_upright_eps}")
    lines.append("  Upright filter: cos_dist > {:.3f}  (success_cos_threshold)".format(success_cos_threshold))
    lines.append("  Deviation = actual_joint_pos − default_joint_pos  [rad]")
    lines.append("  joint_weights: hip=1.0 / thigh=0.75 / calf=0.5")
    lines.append("=" * 72)

    if mean_signed is not None:
        # Table header
        lines.append(
            f"  {'Joint':<22s}  {'mean(act-def)':>14s}  {'mean|dev|':>10s}  {'w²·dev²':>10s}"
        )
        lines.append("  " + "-" * 62)
        for j in range(num_joints):
            jname = joint_names[j]
            lines.append(
                f"  {jname:<22s}  {mean_signed[j]:>+14.4f}  {mean_abs[j]:>10.4f}  {mean_werr[j]:>10.5f}"
            )
        lines.append("")

        lines.append("  Ranked by |deviation| (largest first):")
        for rank_i, j in enumerate(abs_rank):
            lines.append(
                f"    {rank_i + 1:2d}. {joint_names[j]:<22s}  |dev|={mean_abs[j]:.4f} rad"
                f"  signed={mean_signed[j]:+.4f} rad"
            )
        lines.append("")

        lines.append("  Ranked by weighted error contribution w²·dev² (largest first):")
        for rank_i, j in enumerate(werr_rank):
            lines.append(
                f"    {rank_i + 1:2d}. {joint_names[j]:<22s}  w²·dev²={mean_werr[j]:.5f}"
                f"  |dev|={mean_abs[j]:.4f} rad"
            )
        lines.append("")

        # Left-Right symmetry check: group by segment name, compare L vs R
        lines.append("  Left-Right symmetry check  (signed deviation: L − R):")
        lines.append("  Positive = L joint deflects more positive than R counterpart")
        # Build a mapping: strip leading leg-prefix to find L/R pairs.
        # Joint name convention: FL_hip / FR_hip / RL_hip / RR_hip etc.
        seg_to_pairs: dict[str, dict[str, int]] = {}  # segment -> {"FL":j, "FR":j, ...}
        for j, jname in enumerate(joint_names):
            # Expected pattern: <LEG>_<segment>  e.g. FL_hip, RR_calf
            parts = jname.split("_", 1)
            if len(parts) == 2:
                leg_prefix, segment = parts[0], parts[1]
                if segment not in seg_to_pairs:
                    seg_to_pairs[segment] = {}
                seg_to_pairs[segment][leg_prefix] = j

        for segment, leg_map in sorted(seg_to_pairs.items()):
            # Front pair: FL vs FR
            if "FL" in leg_map and "FR" in leg_map:
                fl_j, fr_j = leg_map["FL"], leg_map["FR"]
                diff = mean_signed[fl_j] - mean_signed[fr_j]
                lines.append(
                    f"    Front {segment:<8s}: FL={mean_signed[fl_j]:+.4f}  FR={mean_signed[fr_j]:+.4f}"
                    f"  diff(FL-FR)={diff:+.4f} rad"
                )
            # Rear pair: RL vs RR
            if "RL" in leg_map and "RR" in leg_map:
                rl_j, rr_j = leg_map["RL"], leg_map["RR"]
                diff = mean_signed[rl_j] - mean_signed[rr_j]
                lines.append(
                    f"    Rear  {segment:<8s}: RL={mean_signed[rl_j]:+.4f}  RR={mean_signed[rr_j]:+.4f}"
                    f"  diff(RL-RR)={diff:+.4f} rad"
                )
        lines.append("")
    else:
        lines.append("  (no upright tail data collected — all episodes ended without reaching upright)")
        lines.append("")

    lines.append("─" * 72)
    lines.append("  ACTION RATE (max |Δaction|)    [-]      (all fallen episodes)")
    lines.append("─" * 72)
    lines.append(f"    n         : {ar_stats.get('n', 0)}")
    lines.append(f"    mean      : {ar_stats.get('mean', float('nan')):.4f}")
    lines.append(f"    median    : {ar_stats.get('p50', float('nan')):.4f}")
    lines.append(f"    p95       : {ar_stats.get('p95', float('nan')):.4f}")
    lines.append(f"    max       : {ar_stats.get('max', float('nan')):.4f}")
    lines.append(f"    note: L-inf per step (max over 12 joints), not L2 sum")
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  Measurement date : 2026-06-17")
    lines.append(f"  Total wall-clock  : {total_time:.1f} s")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(_SCRIPT_DIR, exist_ok=True)
    with open(_OUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\n[measure_safety] Report saved to: {_OUT_FILE}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
