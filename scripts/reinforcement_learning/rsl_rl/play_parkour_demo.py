# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive demo + headless recording sweep for Go2 Parkour policies.

Usage — interactive (GUI):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \\
        --checkpoint logs/rsl_rl/<run>/model_<N>.pt \\
        --task Go2-ParkourDemo-v0 \\
        --mode test          # or --mode playground

Usage — headless recording sweep (test terrain only):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \\
        --checkpoint logs/rsl_rl/<run>/model_<N>.pt \\
        --task Go2-ParkourDemo-v0 \\
        --record --headless \\
        [--record_classes stair,gap] \\
        [--record_levels 0,5,10]

num_envs is always forced to 1 (user arg ignored).
vx is NEVER hardcoded in this loop — it is set only via the slider → env._demo_vx → Hook 1
(_pre_physics_step in Go2ParkourDemoEnv), matching the plan spec and eliminating the
play.py:385-387 hardcode (C3 must-fix).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Go2 Parkour interactive demo / headless recording.")
parser.add_argument(
    "--task",
    type=str,
    default="Go2-ParkourDemo-v0",
    help="Task ID. 'Go2-ParkourDemo-v0' (test) or 'Go2-ParkourDemo-Playground-v0' (playground).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Absolute or relative path to the .pt checkpoint (trained with estimator runner, AC-X3).",
)
parser.add_argument(
    "--mode",
    type=str,
    default="test",
    choices=["test", "playground"],
    help="Demo mode: 'test' (goal-driven, test terrain) or 'playground' (free-drive, scatter terrain).",
)
parser.add_argument(
    "--record",
    action="store_true",
    default=False,
    help=(
        "Headless recording sweep. Sweeps test terrain ONLY (playground excluded — no goal for "
        "completion detection). Produces one mp4 per (class, level) tile."
    ),
)
parser.add_argument(
    "--record_classes",
    type=str,
    default=None,
    help=(
        "Comma-separated terrain class names to record, e.g. 'stair,gap'. "
        "Valid names: flat,hurdle,step,gap,stair. Default: all 5."
    ),
)
parser.add_argument(
    "--record_levels",
    type=str,
    default=None,
    help=("Comma-separated difficulty levels (0-10) to record, e.g. '0,5,10'. Default: 0 through 10 (all 11)."),
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=600,
    help=(
        "Max frames per recording clip (R7 cap). Clips that hit the cap are tagged 'timeout'. "
        "Default: 600 (~10 s at 60 Hz)."
    ),
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Output directory for mp4 files. Default: logs/demo_videos/ relative to cwd.",
)
parser.add_argument(
    "--cam_view",
    type=str,
    default="rear",
    choices=["rear", "side"],
    help="Recording chase camera angle: 'rear' (behind robot) or 'side' (robot's side profile, shows gait).",
)
# num_envs intentionally omitted — always forced to 1.
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed for the environment.",
)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Force headless when --record is requested so user doesn't need to remember.
if args_cli.record:
    args_cli.headless = True

# RecordVideo (fixed-length) not used. We do NOT set enable_cameras here;
# the follow camera sensor is declared in the demo cfg and loaded by the env.

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports follow (after AppLauncher)."""

import math
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners.on_policy_runner_parkour_amp import OnPolicyRunnerParkourAMP

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit_parkour,
    export_policy_as_onnx_parkour,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# ---------------------------------------------------------------------------
# Terrain class catalogue — must match _col_to_class order in parkour_env.py
# ---------------------------------------------------------------------------
_ALL_CLASS_NAMES: list[str] = ["flat", "hurdle", "step", "gap", "stair"]
_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(_ALL_CLASS_NAMES)}
_ALL_LEVELS: list[int] = list(range(11))  # 0..10


def _parse_record_subset(class_arg: str | None, level_arg: str | None) -> tuple[list[int], list[int]]:
    """Parse --record_classes / --record_levels and return (class_ids, levels).

    Raises ValueError on any invalid name or level.
    """
    if class_arg is None:
        class_ids = list(range(len(_ALL_CLASS_NAMES)))
    else:
        class_ids = []
        for part in class_arg.split(","):
            name = part.strip()
            if name not in _NAME_TO_ID:
                raise ValueError(
                    f"--record_classes: unknown terrain name '{name}'. Valid names: {', '.join(_ALL_CLASS_NAMES)}."
                )
            class_ids.append(_NAME_TO_ID[name])
        if not class_ids:
            raise ValueError("--record_classes: empty list after parsing.")

    if level_arg is None:
        levels = _ALL_LEVELS
    else:
        levels = []
        for part in level_arg.split(","):
            try:
                lvl = int(part.strip())
            except ValueError:
                raise ValueError(f"--record_levels: '{part.strip()}' is not an integer.") from None
            if lvl < 0 or lvl > 10:
                raise ValueError(f"--record_levels: level {lvl} is out of range [0, 10].")
            levels.append(lvl)
        if not levels:
            raise ValueError("--record_levels: empty list after parsing.")

    return class_ids, levels


def _load_runner(env_wrapped, checkpoint_path: str, device: str, task_id: str) -> OnPolicyRunnerParkourAMP:
    """Load an OnPolicyRunnerParkourAMP from checkpoint and assert estimator exists.

    AC-X3: must load via OnPolicyRunnerParkourAMP with estimator bundle.
    Loads the registered agent cfg via load_cfg_from_registry so the runner receives
    a valid train_cfg dict (matching play.py:279 pattern).
    """
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunnerParkourAMP(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path)

    # Assert that the runtime estimator is present (project memory: AMP runner
    # can be built without estimator if the wrong runner path is used — then actor
    # reads raw privileged obs instead of estimated values, breaking deployment).
    assert getattr(runner.alg, "estimator", None) is not None, (
        "checkpoint must be trained with estimator runner — runner.alg.estimator is None. "
        "Ensure the checkpoint was produced by OnPolicyRunnerParkourAMP with estimator enabled. "
        "See project memory: 'Parkour AMP 러너 estimator 누락 버그 fix'."
    )
    return runner


def _is_terminated(env_unwrapped) -> bool:
    """Return True if env index 0 has a terminal flag set."""
    for attr in ("_term_goal_reached", "_term_base_contact", "_term_tilt", "_term_low_height"):
        t = getattr(env_unwrapped, attr, None)
        if t is None:
            continue
        if isinstance(t, torch.Tensor):
            if bool(t[0].item()):
                return True
        elif bool(t):
            return True
    return False


def _goal_reached(env_unwrapped) -> bool:
    """Return True if the goal-reached terminal flag is set for env 0."""
    t = getattr(env_unwrapped, "_term_goal_reached", None)
    if t is None:
        return False
    if isinstance(t, torch.Tensor):
        return bool(t[0].item())
    return bool(t)


# ---------------------------------------------------------------------------
# Recording sweep (headless, test terrain only — OQ2)
# ---------------------------------------------------------------------------


def _run_recording_sweep(
    env,
    policy,
    policy_nn,
    class_ids: list[int],
    levels: list[int],
    max_frames: int,
    output_dir: str,
    cam_view: str = "rear",
) -> None:
    """Sweep over (class_id, level) tiles, record one mp4 per tile.

    Uses manual frame-capture loop (NOT RecordVideo — D3: RecordVideo is fixed-length).
    Follow-camera frames are read from the demo cfg's 'follow_camera' sensor
    (camera.data.output["rgb"]) — works headless (sensor-based rendering, R8).
    Falls back to a warning if the follow_camera sensor is unavailable.
    """
    try:
        import imageio  # type: ignore[import]

        _has_imageio = True
    except ImportError:
        _has_imageio = False
        print(
            "[recording] WARNING: imageio not available — mp4 output disabled. "
            "Install with: pip install imageio[ffmpeg]"
        )

    os.makedirs(output_dir, exist_ok=True)

    env_unwrapped = env.unwrapped

    # Locate the follow_camera sensor (declared in ParkourDemoEnvCfg).
    # 확인 필요: attribute name 'follow_camera' must match cfg field name in parkour_demo_env_cfg.py.
    follow_cam = getattr(env_unwrapped, "follow_camera", None)
    if follow_cam is None:
        print(
            "[recording] WARNING: env.follow_camera sensor not found. "
            "Frames will be black/skipped. Check parkour_demo_env_cfg.py CameraCfg field name."
        )

    total_clips = len(class_ids) * len(levels)
    clip_num = 0
    success_count = 0
    fail_count = 0

    for class_id in class_ids:
        class_name = _ALL_CLASS_NAMES[class_id]
        for level in levels:
            clip_num += 1
            print(f"[recording] clip {clip_num}/{total_clips} — class={class_name} lvl={level:02d}")

            # Force env onto this tile and reset.
            env_unwrapped.set_demo_tile(class_id, level)

            obs = env.get_observations()
            policy_nn.reset(torch.ones(1, dtype=torch.bool, device=env_unwrapped.device))

            frames: list = []
            status = "timeout"

            for frame_idx in range(max_frames):
                # Chase-camera: move follow_cam to 2.5m behind / 1.5m above robot, looking at robot.
                # Uses root_pos_w [N,3] and heading_w [N,] (world-frame yaw) from ArticulationData.
                if follow_cam is not None:
                    try:
                        root = env_unwrapped._robot.data.root_pos_w[0]  # (3,)
                        yaw = float(env_unwrapped._robot.data.heading_w[0])  # world yaw rad
                        if cam_view == "side":
                            # 로봇 왼쪽 측면에서 바라봄 (heading에 수직), 보행 프로파일 잘 보임
                            # side 방향 벡터: (-sin yaw, +cos yaw) — heading 왼쪽 수직
                            side, up_s = 2.2, 0.7
                            eye = torch.tensor(
                                [[root[0] - side * math.sin(yaw), root[1] + side * math.cos(yaw), root[2] + up_s]],
                                device=root.device,
                            )
                        else:  # rear (기존)
                            back, up = 2.5, 1.5
                            eye = torch.tensor(
                                [[root[0] - back * math.cos(yaw), root[1] - back * math.sin(yaw), root[2] + up]],
                                device=root.device,
                            )
                        target = torch.tensor(
                            [[root[0], root[1], root[2] + 0.2]],
                            device=root.device,
                        )
                        follow_cam.set_world_poses_from_view(eye, target)
                    except Exception as _chase_exc:
                        if frame_idx == 0:
                            print(f"[recording] chase-cam pose update failed (frame 0): {_chase_exc}")

                with torch.no_grad():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    policy_nn.reset(dones)

                # Capture follow-camera frame (R8: sensor-based, not viewport).
                if follow_cam is not None:
                    try:
                        rgb = follow_cam.data.output["rgb"]  # (H, W, 4) uint8 RGBA or (H,W,3) RGB
                        if isinstance(rgb, torch.Tensor):
                            rgb = rgb.squeeze(0).cpu().numpy()  # remove batch dim if present
                        # Drop alpha channel if present.
                        if rgb.ndim == 3 and rgb.shape[-1] == 4:
                            rgb = rgb[..., :3]
                        frames.append(rgb)
                    except Exception as exc:
                        # Sensor not yet populated — skip this frame's capture.
                        if frame_idx == 0:
                            print(f"[recording] follow_camera capture failed (frame 0): {exc}")

                # Check termination after step.
                if _goal_reached(env_unwrapped):
                    status = "success"
                    success_count += 1
                    break
                if _is_terminated(env_unwrapped):
                    status = "fail"
                    fail_count += 1
                    break
            else:
                # max_frames exhausted without terminal — timeout.
                fail_count += 1

            # Encode mp4 (AC-V4: filename encodes class + level + status).
            filename = f"parkour_demo_{class_name}_lvl{level:02d}_{status}.mp4"
            filepath = os.path.join(output_dir, filename)
            if _has_imageio and frames:
                try:
                    writer = imageio.get_writer(filepath, fps=30, format="ffmpeg", codec="libx264")
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()
                    print(f"[recording]   -> {filepath} ({len(frames)} frames, {status})")
                except Exception as exc:
                    print(f"[recording]   -> encode failed: {exc}")
            elif not frames:
                print(f"[recording]   -> no frames captured for {filename} (sensor unavailable)")
            else:
                # imageio not available — still report status.
                print(f"[recording]   -> {filename} ({len(frames)} frames, {status}) [no encoder]")

    print(
        f"[recording] sweep complete: {clip_num} clips, "
        f"{success_count} success, {fail_count} fail/timeout. "
        f"Output dir: {output_dir}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Demo entry point."""
    # ---- Validate recording subset args early (AC-V2: invalid → error, not silent skip) ----
    if args_cli.record:
        try:
            rec_class_ids, rec_levels = _parse_record_subset(args_cli.record_classes, args_cli.record_levels)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            simulation_app.close()
            return
        print(
            f"[recording] sweep plan: {len(rec_class_ids)} class(es) × "
            f"{len(rec_levels)} level(s) = {len(rec_class_ids) * len(rec_levels)} clips"
        )
    else:
        rec_class_ids, rec_levels = [], []

    # ---- Resolve checkpoint path ----
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # ---- Select task + cfg ----
    # Recording always uses test terrain (OQ2: playground is goal-less → no completion detection).
    # Interactive mode uses --task / --mode.
    if args_cli.record:
        # Force test task for recording regardless of --task / --mode.
        task_id = "Go2-ParkourDemo-v0"
        demo_mode = "test"
    else:
        task_id = args_cli.task
        demo_mode = args_cli.mode

    # ---- Import demo cfg after sim launch ----
    from isaaclab_tasks.direct.parkour_imitation.parkour_demo_env_cfg import (
        ParkourDemoEnvCfg,
        ParkourPlaygroundEnvCfg,
    )

    if demo_mode == "playground" and not args_cli.record:
        env_cfg = ParkourPlaygroundEnvCfg()
        # Playground registration uses a separate task id if registered separately;
        # otherwise we override the cfg and keep the test task id.
        # 확인 필요: if 'Go2-ParkourDemo-Playground-v0' is registered separately, use it.
        if "Playground" not in task_id:
            task_id = "Go2-ParkourDemo-v0"  # fall back to test task id, cfg overrides terrain
    else:
        env_cfg = ParkourDemoEnvCfg()

    # num_envs=1 ALWAYS (OQ7 resolved, user): override any cfg default.
    # InteractiveScene reads cfg.scene.num_envs — must force both
    # (runtime: scene built 4096 envs otherwise, GPU exhaustion under --enable_cameras).
    env_cfg.num_envs = 1
    env_cfg.scene.num_envs = 1

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # ---- Create env ----
    env = gym.make(task_id, cfg=env_cfg, render_mode=None)

    # ---- Set demo mode on the underlying Go2ParkourDemoEnv ----
    env.unwrapped._demo_mode = demo_mode

    # ---- Wrap for rsl_rl ----
    env_wrapped = RslRlVecEnvWrapper(env)

    # ---- Load runner + assert estimator (AC-X3) ----
    device = env.unwrapped.device
    try:
        runner = _load_runner(env_wrapped, checkpoint_path, device, task_id)
    except AssertionError as exc:
        print(f"[ERROR] Estimator assertion failed: {exc}")
        env.close()
        simulation_app.close()
        return
    except Exception as exc:
        print(f"[ERROR] Runner load failed: {exc}")
        env.close()
        simulation_app.close()
        return

    # ---- AC-X2: obs invariance check ----
    EXPECTED_POLICY_DIM = 46
    policy_dim = env.unwrapped.single_observation_space["policy"].shape[-1]
    assert policy_dim == EXPECTED_POLICY_DIM, (
        f"obs invariance FAIL: policy obs dim={policy_dim}, expected {EXPECTED_POLICY_DIM}."
    )
    print(f"[INFO] obs invariance OK: policy dim={policy_dim}")

    # ---- Get inference policy ----
    policy = runner.get_inference_policy(device=device)

    # Extract policy nn for reset on done
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ---- Export policy (estimator bundle for deployment) ----
    export_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
    estimator = getattr(runner.alg, "estimator", None)
    try:
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(
            policy_nn, "student_obs_normalizer", None
        )
        export_policy_as_jit_parkour(
            policy_nn,
            normalizer=normalizer,
            path=export_dir,
            filename="policy.pt",
            estimator=estimator,
        )
        export_policy_as_onnx_parkour(
            policy_nn,
            path=export_dir,
            normalizer=normalizer,
            filename="policy.onnx",
            estimator=estimator,
        )
    except Exception as exc:
        print(f"[INFO] Policy export skipped ({exc}) — inference will continue.")

    # =========================================================================
    # RECORDING PATH — headless sweep over test terrain (OQ2)
    # =========================================================================
    if args_cli.record:
        output_dir = args_cli.video_dir or os.path.join("logs", "demo_videos")
        output_dir = os.path.abspath(output_dir)
        _run_recording_sweep(
            env=env_wrapped,
            policy=policy,
            policy_nn=policy_nn,
            class_ids=rec_class_ids,
            levels=rec_levels,
            max_frames=args_cli.max_frames,
            output_dir=output_dir,
            cam_view=args_cli.cam_view,
        )
        env.close()
        return

    # =========================================================================
    # INTERACTIVE PATH — GUI loop with omni.ui panel
    # =========================================================================
    from isaaclab_tasks.direct.parkour_imitation.parkour_demo_panel import ParkourDemoPanel

    panel = ParkourDemoPanel(env.unwrapped)

    obs = env_wrapped.get_observations()
    dt = env.unwrapped.step_dt

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # NOTE: vx is NOT set here. It flows exclusively via:
                #   slider → env._demo_vx → Hook 1 (_pre_physics_step) → _commands[:,0]
                # Setting _commands[:,0] here (as play.py:385 does) is intentionally
                # removed (plan C3 must-fix). The demo env hooks handle injection.

                actions = policy(obs)
                obs, _, dones, _ = env_wrapped.step(actions)
                policy_nn.reset(dones)

            # Pump the omni.ui panel (no-op in headless / if no GUI).
            panel.update()

            # Real-time pacing — optional, non-blocking.
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        panel.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
