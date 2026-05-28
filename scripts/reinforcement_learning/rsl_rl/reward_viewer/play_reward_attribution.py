# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward attribution analysis script for Go2 Parkour.

Records per-step reward-term breakdown and per-foot contact data for N envs
(one per active terrain class) at a fixed difficulty level, then renders
offline matplotlib plots for visual attribution of 3-leg gait behaviour.

Conda env: isaac-parkour  (NOT the generic 'isaac' env)

Usage:
    conda activate isaac-parkour
    cd /home/lgb/IsaacLab
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reward_attribution.py \\
        --task Go2-Parkour-Direct-v0 \\
        --num_envs 5 \\
        --difficulty 5 \\
        [--load_run 2026-05-26_16-33-17_add_spot_trot_reward] \\
        [--checkpoint model_16800.pt]

Outputs (two locations):
    {checkpoint_run_dir}/reward_attribution/   ← primary (versioned with model)
    _workspace/parkour_reward_attribution/results/  ← convenience copy

    Files per location:
        reward_attribution_{terrain_name}.npz   × 5 (per-terrain buffer)
        reward_attribution_{terrain_name}.png   × 5 (per-terrain plot)
        reward_attribution_all.npz              (full multi-env buffer)
        run_meta.json                           (checkpoint, params, names)

Design reference:
    _workspace/parkour_reward_attribution/DESIGN.md §3-§8
    _workspace/parkour_reward_attribution/SUBTASKS.md §S2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os as _os_boot
import sys

# Ensure the parent rsl_rl/ directory is on sys.path so that cli_args.py
# (which lives there) can be imported by bare name after the move into
# reward_viewer/.
sys.path.insert(0, _os_boot.path.dirname(_os_boot.path.dirname(_os_boot.path.abspath(__file__))))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Reward attribution analysis for Go2 Parkour.")
parser.add_argument(
    "--task",
    type=str,
    default="Go2-Parkour-Direct-v0",
    help="Task name (must be Go2-Parkour-Direct-v0).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Envs to spawn (default: auto = number of active terrain classes, normally 5).",
)
parser.add_argument(
    "--difficulty",
    type=int,
    default=5,
    help="Terrain difficulty row to pin (0 = easiest, 10 = hardest; default 5).",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RL agent config entry-point key.",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--live-viz",
    action="store_true",
    default=False,
    help=(
        "Stream per-step reward + contact data to a live ZMQ viewer process "
        "(reward_attribution_viewer.py). OFF by default; has zero overhead when absent. "
        "See PYQT_IPC_SPEC.md for the wire format and launch instructions."
    ),
)
parser.add_argument(
    "--zmq-endpoint",
    type=str,
    default=None,
    metavar="ENDPOINT",
    help=(
        "Override ZMQ PUB endpoint used with --live-viz "
        "(e.g. tcp://127.0.0.1:5557 for TCP loopback). "
        "Default: ipc:///tmp/parkour_reward.sock (from reward_pub_protocol.ZMQ_ENDPOINT). "
        "Useful for smoke-testing without IPC socket cleanup issues."
    ),
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra.
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse app.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import os as _os
import sys as _sys

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunnerParkour

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# ── Module-level state for deferred offline plotting ─────────────────────────
# Populated inside main(); consumed in __main__ after simulation_app.close().
# (Matplotlib Agg backend must be imported after Isaac's Qt loop exits —
#  see DESIGN.md §8 "Why offline + matplotlib".)
_deferred_plots: list[tuple[str, str, str]] = []   # (npz_path, png_primary, png_results)

# ── Constants ─────────────────────────────────────────────────────────────────
_SUPPORTED_TASK = "Go2-Parkour-Direct-v0"
_FOOT_NAMES = ["FL", "FR", "RL", "RR"]
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_RESULTS_DIR = _os.path.normpath(
    _os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", "_workspace", "parkour_reward_attribution", "results")
)


# ── Terrain pinning helper (DESIGN.md §4) ─────────────────────────────────────
def _pin_terrain_per_env(raw, active_class_ids: list[int], difficulty: int) -> None:
    """Pin each env to a distinct terrain class at the requested difficulty row.

    Generalises the existing ``_change_terrain_for_viewer`` keyboard hook
    (parkour_env.py:1686-1738) from "one env" to "all N envs", applied once at
    script start.  No new code paths are exercised; the env is unmodified.

    Args:
        raw:               ``env.unwrapped`` (Go2ParkourEnv instance).
        active_class_ids:  Terrain class IDs, one per env (len == raw.num_envs).
        difficulty:        Row index to pin; clamped to [0, num_rows-1].
    """
    device = raw.device

    # For each active class find the first column assigned to that class.
    # _col_to_class[c] = terrain class for column c (built at env init, parkour_env.py:160).
    col_per_env = torch.tensor(
        [int((raw._col_to_class == k).nonzero()[0].item()) for k in active_class_ids],
        dtype=torch.long,
        device=device,
    )

    num_rows = int(raw._terrain.terrain_origins.shape[0])
    level = max(0, min(difficulty, num_rows - 1))
    print(
        f"[pin_terrain] Pinning {len(active_class_ids)} envs → level={level}/{num_rows - 1}, "
        f"cols={col_per_env.tolist()}, class_ids={active_class_ids}"
    )

    # Write into TerrainImporter's aliased tensors (same tensors as keyboard hook).
    raw._terrain_levels[:] = level
    raw._terrain_types[:] = col_per_env
    raw._env_class[:] = raw._col_to_class[raw._terrain_types]
    raw._terrain.env_origins[:] = raw._terrain.terrain_origins[
        raw._terrain_levels, raw._terrain_types
    ]

    # Belt-and-suspenders: skip curriculum advance even if terrain_curriculum=True.
    raw._skip_curriculum[:] = True

    # Full reset — spawns robots at the assigned tile origin.
    raw._reset_idx(None)
    # Cancel _reset_idx's random episode-spread (parkour_env.py:1215).
    raw.episode_length_buf[:] = 0


# ── Recording helpers ─────────────────────────────────────────────────────────
def _capture_contact(raw) -> torch.Tensor:
    """Read current-step per-foot contact mask (N, 4) bool — same threshold as env."""
    forces = raw._contact_sensor.data.net_forces_w_history   # (N, hist, B, 3)
    return (torch.norm(forces[:, 0, raw._feet_ids], dim=-1) > 2.0).cpu()  # (N, 4)


def _capture_foot_xy_speed(raw) -> torch.Tensor:
    """Read per-foot XY speed (N, 4) float32 for dragging indicator."""
    return torch.norm(
        raw._robot.data.body_lin_vel_w[:, raw._feet_ids, :2], dim=-1
    ).detach().cpu()  # (N, 4)


# ── Main ──────────────────────────────────────────────────────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Reward attribution recording loop for Go2 Parkour."""
    global _deferred_plots

    # ── Validate task ──────────────────────────────────────────────────────────
    if args_cli.task != _SUPPORTED_TASK:
        raise ValueError(
            f"play_reward_attribution.py only supports '{_SUPPORTED_TASK}' "
            f"(terrain-pinning is parkour-specific); got '{args_cli.task}'."
        )

    # ── Detect active terrains from cfg ───────────────────────────────────────
    terrain_gen_cfg = env_cfg.terrain.terrain_generator
    sub_terrain_names: list[str] = list(terrain_gen_cfg.sub_terrains.keys())  # insertion order = class IDs
    active_info: list[tuple[int, str]] = [
        (i, name.removeprefix("parkour_"))
        for i, name in enumerate(sub_terrain_names)
        if terrain_gen_cfg.sub_terrains[name].proportion > 0
    ]
    if not active_info:
        raise RuntimeError("No active terrain classes found (all proportions == 0).")
    active_class_ids = [t[0] for t in active_info]
    terrain_names = [t[1] for t in active_info]      # "flat", "hurdle", "step", …
    num_active = len(active_class_ids)
    print(f"[INFO] Active terrains ({num_active}): {list(zip(active_class_ids, terrain_names))}")

    # num_envs: default to num_active; override mismatches with a warning.
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else num_active
    if num_envs != num_active:
        print(
            f"[WARN] --num_envs {num_envs} != num_active_terrains {num_active}; "
            f"setting num_envs = {num_active} (one env per terrain class)."
        )
        num_envs = num_active

    # ── Cfg overrides (before gym.make) ───────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = num_envs
    env_cfg.terrain_curriculum = False                              # §4: disable curriculum
    env_cfg.terrain.max_init_terrain_level = args_cli.difficulty    # cosmetic; §4 overrides
    env_cfg.debug_vis = False                                       # suppress viewport overlays

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # ── Checkpoint discovery (mirrors play.py:218-234) ────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Looking for checkpoints in: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # ── Env construction (mirrors play.py:247-270) ────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        from isaaclab.envs import multi_agent_to_single_agent
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Runner + policy (mirrors play.py:275-296) ─────────────────────────────
    if agent_cfg.class_name != "OnPolicyRunnerParkour":
        raise ValueError(
            f"play_reward_attribution.py requires OnPolicyRunnerParkour, "
            f"got '{agent_cfg.class_name}'."
        )
    runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ── Live-viz ZMQ publisher setup (PYQT_IPC_SPEC.md §4-§6) ───────────────
    # All ZMQ state is held in local variables so it is never reachable when
    # --live-viz is absent, guaranteeing bit-identical offline behaviour.
    _zmq_ctx = None
    _zmq_pub = None
    _zmq_noblock = None  # zmq.NOBLOCK constant (int 1); captured to avoid re-import in loop
    if args_cli.live_viz:
        try:
            import zmq as _zmq  # noqa: PLC0415
            from reward_pub_protocol import ZMQ_ENDPOINT, ZMQ_HWM, encode_step, make_pub_socket  # noqa: PLC0415
            _zmq_ctx = _zmq.Context()
            _effective_endpoint = args_cli.zmq_endpoint or ZMQ_ENDPOINT
            if args_cli.zmq_endpoint:
                # Custom endpoint (e.g. TCP for smoke-testing): create socket manually
                # because make_pub_socket always binds to the default IPC endpoint.
                _zmq_pub = _zmq_ctx.socket(_zmq.PUB)
                _zmq_pub.setsockopt(_zmq.SNDHWM, ZMQ_HWM)
                _zmq_pub.bind(_effective_endpoint)
            else:
                _zmq_pub = make_pub_socket(_zmq_ctx)
            _zmq_noblock = _zmq.NOBLOCK
            print(f"[live-viz] ZMQ PUB socket bound to {_effective_endpoint}")
            print("[live-viz] Start reward_attribution_viewer.py in a separate terminal.")
        except Exception as _exc:
            print(f"[live-viz] WARN: ZMQ setup failed ({_exc}); continuing without live viz.")
            _zmq_ctx = None
            _zmq_pub = None
            _zmq_noblock = None

    # ── Terrain pinning (DESIGN.md §4) ────────────────────────────────────────
    raw = env.unwrapped
    _pin_terrain_per_env(raw, active_class_ids, args_cli.difficulty)
    obs = env.get_observations()

    # ── Record env metadata ───────────────────────────────────────────────────
    step_dt: float = float(raw.step_dt)
    term_names: list[str] = list(raw.cfg.reward_scales.keys())         # 16, canonical order
    reward_scales_dict: dict[str, float] = dict(raw.cfg.reward_scales)
    dragging_threshold: float = float(raw.cfg.dragging_velocity_threshold)
    K = len(term_names)                                                # 16
    N = raw.num_envs                                                   # 5
    T_max = int(raw.cfg.episode_length_s / step_dt)                    # 1000

    print(f"[INFO] Buffers: T_max={T_max}, N={N}, K={K}, step_dt={step_dt}")
    print(f"[INFO] Reward terms: {term_names}")

    # ── Pre-allocate recording buffers ────────────────────────────────────────
    buf_rewards = torch.zeros(T_max, N, K, dtype=torch.float32)             # (T, N, 16)
    buf_contacts = torch.zeros(T_max, N, 4, dtype=torch.bool)               # (T, N, 4)
    buf_foot_xy_speed = torch.zeros(T_max, N, 4, dtype=torch.float32)       # (T, N, 4)
    buf_commands = torch.zeros(T_max, N, 3, dtype=torch.float32)            # (T, N, 3)
    ep_len = torch.full((N,), -1, dtype=torch.int32)
    env_done = torch.zeros(N, dtype=torch.bool)

    # ── Step loop ─────────────────────────────────────────────────────────────
    t = 0
    print("[INFO] Starting recording loop …")
    while simulation_app.is_running() and t < T_max and not env_done.all():
        with torch.inference_mode():
            # Forward-only command: 1 m/s, no lateral, no yaw.
            raw._commands[:, 0] = 1.0
            raw._commands[:, 1] = 0.0
            raw._commands[:, 2] = 0.0

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # Capture reward breakdown (S1 buffer, §5).
        r_step = raw._last_reward_breakdown_per_env.detach().cpu()      # (N, K)

        # Capture foot contact (§6).
        c_step = _capture_contact(raw)                                   # (N, 4) bool

        # Capture foot XY speed (for dragging indicator).
        spd_step = _capture_foot_xy_speed(raw)                           # (N, 4)

        # Capture commands.
        cmd_step = raw._commands[:, :3].detach().cpu()                   # (N, 3)

        # Live-viz publish — non-blocking; silently swallowed if viewer absent.
        # Runs only when --live-viz is active; zero overhead otherwise.
        if _zmq_pub is not None:
            try:
                _live_payloads = [
                    {
                        "env_id": i,
                        "rewards": r_step[i].tolist(),
                        "contact": c_step[i].tolist(),
                        "commands": cmd_step[i].tolist(),
                        "done": bool(dones.cpu()[i].item()),
                        "terrain_id": int(raw._env_class[i].item()),
                    }
                    for i in range(N)
                ]
                _zmq_pub.send(encode_step(t, _live_payloads), flags=_zmq_noblock)
            except Exception:
                pass  # never stall the sim (HWM drop or viewer disconnect)

        # Write into buffers only for envs that have not yet terminated.
        keep = ~env_done
        buf_rewards[t, keep] = r_step[keep]
        buf_contacts[t, keep] = c_step[keep]
        buf_foot_xy_speed[t, keep] = spd_step[keep]
        buf_commands[t, keep] = cmd_step[keep]

        # Mark newly-terminated envs (freeze their slice).
        done_now = dones.cpu().bool() & ~env_done
        for i in done_now.nonzero(as_tuple=True)[0].tolist():
            ep_len[i] = t + 1
            env_done[i] = True

        if t % 100 == 0:
            n_active = int((~env_done).sum().item())
            print(f"[INFO] step={t:4d}  active_envs={n_active}/{N}")
        t += 1

    # Envs that hit T_max without terminating — set their ep_len to t.
    not_done_mask = ~env_done
    if not_done_mask.any():
        for i in not_done_mask.nonzero(as_tuple=True)[0].tolist():
            ep_len[i] = t
        print(f"[INFO] {int(not_done_mask.sum())} env(s) reached T_max without terminating.")

    T_used = t
    print(f"[INFO] Recording done. T_used={T_used}, ep_lens={ep_len.tolist()}")

    # ── Build output directories ───────────────────────────────────────────────
    out_primary = os.path.join(log_dir, "reward_attribution")
    results_abs = os.path.abspath(_RESULTS_DIR)
    for d in [out_primary, results_abs]:
        os.makedirs(d, exist_ok=True)

    # ── Save all-env buffer (DESIGN.md §2) ────────────────────────────────────
    all_npz_path = os.path.join(out_primary, "reward_attribution_all.npz")
    np.savez_compressed(
        all_npz_path,
        rewards=buf_rewards[:T_used].numpy(),              # (T_used, N, 16)
        foot_contact=buf_contacts[:T_used].numpy(),        # (T_used, N, 4)
        foot_xy_speed=buf_foot_xy_speed[:T_used].numpy(),  # (T_used, N, 4)
        commands=buf_commands[:T_used].numpy(),            # (T_used, N, 3)
        episode_len=ep_len.numpy(),                        # (N,)
        terrain_ids=np.array(active_class_ids, dtype=np.int32),
        terrain_names=np.array(terrain_names),
        term_names=np.array(term_names),
        foot_names=np.array(_FOOT_NAMES),
        reward_scales=np.array(list(reward_scales_dict.values()), dtype=np.float32),
        step_dt=np.float32(step_dt),
        difficulty=np.int32(args_cli.difficulty),
    )
    print(f"[INFO] Saved all-env buffer → {all_npz_path}")

    # ── Save run_meta.json ─────────────────────────────────────────────────────
    meta = {
        "checkpoint": str(resume_path),
        "difficulty": args_cli.difficulty,
        "command": [1.0, 0.0, 0.0],
        "term_names": term_names,
        "foot_names": _FOOT_NAMES,
        "step_dt": step_dt,
        "terrain_names": terrain_names,
        "terrain_ids": active_class_ids,
        "terrain_levels": [args_cli.difficulty] * N,
        "episode_lens": ep_len.tolist(),
        "dragging_velocity_threshold": dragging_threshold,
        "reward_scales": reward_scales_dict,
    }
    meta_path = os.path.join(out_primary, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved run metadata → {meta_path}")

    # ── Per-terrain save (+ defer plot) ───────────────────────────────────────
    for env_idx, (class_id, tname) in enumerate(zip(active_class_ids, terrain_names)):
        T_env = int(ep_len[env_idx].item())
        if T_env <= 0:
            print(f"[WARN] env {env_idx} ({tname}): ep_len={T_env}, skipping save.")
            continue

        # All values must be np.ndarray so **-unpacking into savez_compressed
        # matches the **kwds: ArrayLike signature without triggering a pyright
        # false-positive on the allow_pickle: bool parameter.
        per_env_data: dict[str, np.ndarray] = {
            "rewards": buf_rewards[:T_env, env_idx].numpy(),              # (T_env, 16)
            "foot_contact": buf_contacts[:T_env, env_idx].numpy(),        # (T_env, 4)
            "foot_xy_speed": buf_foot_xy_speed[:T_env, env_idx].numpy(),  # (T_env, 4)
            "commands": buf_commands[:T_env, env_idx].numpy(),            # (T_env, 3)
            "episode_len": np.array(T_env, dtype=np.int32),
            "terrain_id": np.array(class_id, dtype=np.int32),
            "terrain_name": np.array(tname),
            "term_names": np.array(term_names),
            "foot_names": np.array(_FOOT_NAMES),
            "reward_scales": np.array(list(reward_scales_dict.values()), dtype=np.float32),
            "step_dt": np.array(step_dt, dtype=np.float32),
            "difficulty": np.array(args_cli.difficulty, dtype=np.int32),
            "dragging_velocity_threshold": np.array(dragging_threshold, dtype=np.float32),
        }

        # Primary output (under log_dir/reward_attribution/).
        npz_primary = os.path.join(out_primary, f"reward_attribution_{tname}.npz")
        np.savez_compressed(npz_primary, **per_env_data)  # type: ignore[call-arg]
        print(f"[INFO] Saved {tname} buffer ({T_env} steps) → {npz_primary}")

        # Mirror copy to _workspace/results/ with ep_len in filename.
        npz_results = os.path.join(results_abs, f"reward_attribution_{tname}_ep{T_env}.npz")
        np.savez_compressed(npz_results, **per_env_data)  # type: ignore[call-arg]

        # Enqueue plots for offline rendering (after sim app closes).
        png_primary = os.path.join(out_primary, f"reward_attribution_{tname}.png")
        png_results = os.path.join(results_abs, f"reward_attribution_{tname}_ep{T_env}.png")
        _deferred_plots.append((npz_primary, png_primary, png_results))

    print(
        f"[INFO] Data saved.\n"
        f"  Primary : {out_primary}\n"
        f"  Results : {results_abs}\n"
        f"  Plots will be rendered after simulator closes."
    )

    # ── Live-viz teardown ─────────────────────────────────────────────────────
    if _zmq_ctx is not None and _zmq_pub is not None:
        try:
            _zmq_pub.close()
            _zmq_ctx.term()
            print("[live-viz] ZMQ publisher closed.")
        except Exception:
            pass

    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()  # type: ignore[call-arg]  # Hydra decorator changes the signature
    # Close Isaac Sim BEFORE rendering plots to avoid Qt/matplotlib backend conflict
    # (DESIGN.md §8: "Offline matplotlib after the episode is robust because the sim
    # app has exited by then.")
    simulation_app.close()

    if _deferred_plots:
        print(f"\n[INFO] Rendering {len(_deferred_plots)} plot(s) offline …")
        import matplotlib
        matplotlib.use("Agg")
        from reward_attribution_plot import plot_terrain_attribution
        import shutil

        for entry in _deferred_plots:
            npz_path, png_primary, png_results = entry
            try:
                plot_terrain_attribution(npz_path, png_primary)
                shutil.copy2(png_primary, png_results)
                print(f"[plot] {os.path.basename(png_primary)} → also copied to results/")
            except Exception as exc:
                print(f"[WARN] Plot failed for {npz_path}: {exc}")

        print("[INFO] All plots done.")
    else:
        print("[WARN] No plots to render (no episodes recorded).")
