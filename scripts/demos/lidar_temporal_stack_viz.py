# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the effect of ``lidar_stack_at_sensor_rate`` on the LiDAR K=3 temporal frame stack.

Read-only diagnostic demo — does NOT modify any env logic. It records a single rollout with
the SL LiDAR policy while the ring buffer is pushed every control step (stride=1, i.e. the
default/OFF cadence), then *reconstructs* both the OFF-cadence stack (stride 1: k=0,1,2) and
the ON-cadence stack (stride 5: k=0,5,10) from that one recorded sequence. Because
``lidar_stack_at_sensor_rate`` changes nothing except the ring-buffer push stride, this
"method B" reconstruction isolates the flag's pure effect without any rollout-trajectory
confound (see project note: env.py:424-441 in
``parkour_imitation_random_goal_lidar_env.py``).

Usage
-----
.. code-block:: bash

    conda run -n isaac-5.1 --no-capture-output env -u DISPLAY \\
        ./isaaclab.sh -p scripts/demos/lidar_temporal_stack_viz.py \\
        --headless --enable_cameras --device cuda:2 --num_envs 16 --steps 60
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# reuse the RSL-RL CLI arg helpers from the play/train scripts.
import os as _os_boot

sys.path.insert(
    0, _os_boot.path.join(_os_boot.path.dirname(_os_boot.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl")
)
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser(
    description="Visualize LiDAR K=3 temporal stack under stride1 (OFF) vs stride5 (ON) cadence."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Go2-ParkourImitation-Symmetry-RandomGoal-Lidar-SL-v0",
    help=(
        "Task id used ONLY to resolve the correct agent_cfg (ActorCriticRMALidar / "
        "OnPolicyRunnerParkourAMPLidar network matching the SL checkpoint). The env_cfg's "
        "lidar_stack_at_sensor_rate flag is force-overridden to False below regardless of "
        "task default, so the rollout is always recorded at stride=1 (freshest cadence)."
    ),
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument(
    "--sl_checkpoint",
    type=str,
    default=(
        "logs/rsl_rl/parkour_imitation_go2_lidar_sl/2026-07-07_12-02-13_SL_lidar_stack/model_23400.pt"
    ),
    help="Absolute or repo-relative path to the SL_lidar_stack checkpoint (avoids clashing with cli_args' --checkpoint).",
)
parser.add_argument("--steps", type=int, default=60, help="Number of rollout control steps to record.")
parser.add_argument("--env_id", type=int, default=0, help="Which env's freshest LiDAR frame to record each step.")
parser.add_argument(
    "--num_captures", type=int, default=2, help="How many distinct capture timesteps T to render (top-N by diff)."
)
parser.add_argument(
    "--out_dir", type=str, default="logs/lidar_temporal_viz", help="Output directory for the comparison PNG(s)."
)
parser.add_argument(
    "--trail",
    action="store_true",
    help=(
        "Additionally render the K=3 temporal stack as a color-coded 'motion trail' point cloud "
        "(k0=red/newest, k1=green, k2=blue/oldest), OFF (stride=1) vs ON (stride=5) side-by-side. "
        "Always writes sim_trail_3d.png (matplotlib, robot-relative, guaranteed fallback). If "
        "--enable_cameras is ALSO passed, additionally attempts an in-scene Isaac Sim marker-overlay "
        "screenshot (sim_scene_OFF.png / sim_scene_ON.png) via a pose-teleport + camera capture trick — "
        "best-effort, falls back gracefully (prints the failure reason, keeps the matplotlib output) "
        "if the render pipeline misbehaves headless."
    ),
)
parser.add_argument(
    "--trail_max_pts",
    type=int,
    default=3000,
    help="Max LiDAR hit points sampled per time-slot (k0/k1/k2) for the trail visualization.",
)
parser.add_argument(
    "--trail_max_range",
    type=float,
    default=6.0,
    help=(
        "Sensor-range cap (m) for the trail viz, on top of the sensor's own max_distance. This is a "
        "coarse pre-filter applied to raw hit *ranges* (as measured by the sensor at each recorded "
        "step) before the robot-relative near-field filter below. Set <=0 to disable (use the sensor's "
        "full max_distance instead)."
    ),
)
parser.add_argument(
    "--near_radius",
    type=float,
    default=2.0,
    help=(
        "Robot-relative near-field filter (m) for the 3D motion-trail viz ONLY (does not affect the "
        "heatmap outputs). Keeps hit points with horizontal distance sqrt(x^2+y^2) < near_radius from "
        "the robot's position AT the capture instant T (robot-relative frame, x=forward). Far-range "
        "points are dominated by rotational (yaw) parallax rather than the translational near-field "
        "parallax the trail is meant to expose, which swamps the effect when plotted together. Set <=0 "
        "to disable."
    ),
)
parser.add_argument(
    "--near_fwd_min",
    type=float,
    default=-0.5,
    help=(
        "Additional robot-relative forward-axis filter (m) for the 3D motion-trail viz: drops points "
        "with x_fwd <= this value (keeps mostly-forward-and-side terrain, trims straight-behind hits). "
        "Set to a very negative number (e.g. -1e9) to disable."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# no video/onscreen viewport needed — cameras stay off, headless offscreen sim only.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerParkour
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP, OnPolicyRunnerAMPBase
from rsl_rl.runners.on_policy_runner_parkour_amp import (
    OnPolicyRunnerParkourAMP,
    OnPolicyRunnerParkourAMPLidar,
    OnPolicyRunnerParkourAMPVoxel,
)

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# ── trail-mode helpers (module scope; independent of the hydra-wrapped main()) ─────────────


def _gather_slot_points(
    hits_hist: list[torch.Tensor],
    dist_hist: list[torch.Tensor],
    idx: int,
    max_distance: float,
    max_pts: int,
    range_cap: float | None = None,
) -> torch.Tensor:
    """Filtered + optionally subsampled world-frame hit points for one recorded step ``idx``.

    Valid = finite hit AND measured distance below the sensor's miss sentinel (``max_distance``,
    with a 0.1 m margin — same convention as ``parkour_perception_viz.py``'s lidar overlay).

    ``range_cap`` (optional, trail-mode only): additionally drops hits beyond this radius. Far-range
    hits are dominated by ROTATIONAL parallax (a fixed angular yaw change sweeps a far-terrain hit
    point by a large absolute distance even though the sensor barely translated), which swamps the
    near-field TRANSLATIONAL parallax the trail viz is meant to expose (foot/terrain-level awareness,
    same near-field range the LiDAR is actually used for). Purely a display/analysis restriction —
    does not touch the sensor's own ``max_distance`` semantics.
    """
    hits = hits_hist[idx]
    dist = dist_hist[idx]
    valid = torch.isfinite(hits).all(dim=-1) & (dist < max_distance - 0.1)
    if range_cap is not None:
        valid = valid & (dist < range_cap)
    pts = hits[valid]
    if pts.shape[0] > max_pts:
        perm = torch.randperm(pts.shape[0], device=pts.device)[:max_pts]
        pts = pts[perm]
    return pts


def _to_robot_frame(pts_w: torch.Tensor, anchor_pos: torch.Tensor, anchor_quat: torch.Tensor) -> torch.Tensor:
    """World-frame points -> robot-body-frame at (anchor_pos, anchor_quat) [x=fwd, y=left, z=up]."""
    if pts_w.shape[0] == 0:
        return pts_w
    rel = pts_w - anchor_pos.unsqueeze(0)
    return quat_apply_inverse(anchor_quat, rel)


def _filter_near_field(pts_r: torch.Tensor, near_radius: float, fwd_min: float) -> torch.Tensor:
    """Robot-relative near-field filter for one time-slot's points (already in robot frame @T).

    Keeps points with horizontal distance ``sqrt(x^2+y^2) < near_radius`` from the robot AND
    (optionally) forward coordinate ``x > fwd_min``. This is the filter that actually makes the
    motion-trail effect visible: far-range LiDAR hits carry mostly ROTATIONAL (yaw) parallax whose
    absolute displacement swamps the near-field TRANSLATIONAL parallax the trail is meant to show
    once both are plotted on a shared euclidean scale (see module-level failure-mode note).
    """
    if pts_r.shape[0] == 0 or near_radius <= 0:
        return pts_r
    horiz = torch.sqrt(pts_r[:, 0] ** 2 + pts_r[:, 1] ** 2)
    mask = horiz < near_radius
    if fwd_min > -1e8:
        mask = mask & (pts_r[:, 0] > fwd_min)
    return pts_r[mask]


def _render_trail_matplotlib(
    off_pts_r: list[torch.Tensor],
    on_pts_r: list[torch.Tensor],
    out_path: str,
    T: int,
    env_id: int,
    near_radius: float,
    fwd_min: float,
) -> dict:
    """Robot-relative 3D scatter (+ top-down XY) of the OFF vs ON K=3 trail. Always succeeds (pure matplotlib)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

    colors = ["#d62728", "#2ca02c", "#1f77b4"]  # k0=red(newest), k1=green, k2=blue(oldest)
    off_labels = ["k0 (Δt=0s)", "k1 (Δt=-0.02s)", "k2 (Δt=-0.04s)"]
    on_labels = ["k0 (Δt=0s)", "k1 (Δt=-0.10s)", "k2 (Δt=-0.20s)"]

    def to_np(pts: torch.Tensor) -> np.ndarray:
        return pts.detach().cpu().numpy() if pts.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

    off_np = [to_np(p) for p in off_pts_r]
    on_np = [to_np(p) for p in on_pts_r]

    def centroid(pts: np.ndarray) -> np.ndarray:
        return pts.mean(axis=0) if pts.shape[0] > 0 else np.full(3, np.nan, dtype=np.float32)

    off_c = [centroid(p) for p in off_np]
    on_c = [centroid(p) for p in on_np]

    def shift_cm(c_list: list[np.ndarray]) -> float:
        if np.any(np.isnan(c_list[0])) or np.any(np.isnan(c_list[-1])):
            return float("nan")
        return float(np.linalg.norm(c_list[0] - c_list[-1]) * 100.0)

    off_shift_cm = shift_cm(off_c)
    on_shift_cm = shift_cm(on_c)
    ratio = on_shift_cm / off_shift_cm if off_shift_cm and off_shift_cm > 1e-6 else float("nan")

    fig = plt.figure(figsize=(14, 12))
    near_desc = f"near-field <{near_radius:.1f}m" + (f", x>{fwd_min:.1f}m" if fwd_min > -1e8 else "")
    fig.suptitle(
        f"LiDAR K=3 motion trail (robot-relative @T, x=forward, {near_desc}): OFF (stride=1) vs ON (stride=5)\n"
        f"capture T={T} control step, env_id={env_id}  |  centroid shift k0→k2: "
        f"OFF={off_shift_cm:.2f} cm   ON={on_shift_cm:.2f} cm   (ON/OFF={ratio:.2f}x)",
        fontsize=12,
    )

    # Fixed axis window, sized to the near-field filter (both OFF and ON plotted at the SAME scale
    # so the visual comparison is fair). Zooming to the filter range — instead of auto-scaling to
    # whatever points survived — keeps the trail effect legible instead of getting stretched/
    # squashed by however many near-field points happened to pass the filter this capture.
    x_lo = fwd_min if fwd_min > -1e8 else -near_radius - 0.3
    x_hi = near_radius + 0.5
    y_lim = near_radius + 0.3
    all_pts = np.concatenate([p for p in (off_np + on_np) if p.shape[0] > 0], axis=0) if any(
        p.shape[0] > 0 for p in off_np + on_np
    ) else np.zeros((1, 3), dtype=np.float32)
    z_lim = float(np.percentile(np.abs(all_pts[:, 2]), 98)) * 1.3 + 0.2 if all_pts.shape[0] > 0 else 1.0

    def scatter3d(ax, pts_list, labels, title):
        for p, c, lab in zip(pts_list, colors, labels):
            if p.shape[0] > 0:
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=28, c=c, label=lab, alpha=0.7, edgecolors="none")
        ax.scatter([0], [0], [0], s=180, c="black", marker="*", label="robot @T")
        ax.quiver(0, 0, 0, 0.4, 0, 0, color="black", linewidth=1.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_zlim(-z_lim, z_lim)
        ax.set_xlabel("x fwd (m)")
        ax.set_ylabel("y left (m)")
        ax.set_zlabel("z up (m)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    def scatter2d(ax, pts_list, labels, title, cvals):
        for p, c, lab in zip(pts_list, colors, labels):
            if p.shape[0] > 0:
                ax.scatter(p[:, 0], p[:, 1], s=32, c=c, label=lab, alpha=0.7, edgecolors="none")
        ax.scatter([0], [0], s=200, c="black", marker="*", label="robot @T")
        ax.annotate("", xy=(0.4, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5})
        for cc in cvals:
            if not np.any(np.isnan(cc)):
                ax.scatter([cc[0]], [cc[1]], s=110, facecolors="none", edgecolors="k", marker="o", linewidths=1.8)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xlabel("x fwd (m)")
        ax.set_ylabel("y left (m)")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="upper left")

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    scatter3d(ax1, off_np, off_labels, "OFF (stride=1) — 3D (near-duplicate, expect overlap)")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    scatter3d(ax2, on_np, on_labels, "ON (stride=5) — 3D (real temporal context, expect trail)")
    ax3 = fig.add_subplot(2, 2, 3)
    scatter2d(ax3, off_np, off_labels, f"OFF top-down (XY) | centroid shift k0→k2 = {off_shift_cm:.2f} cm", off_c)
    ax4 = fig.add_subplot(2, 2, 4)
    scatter2d(ax4, on_np, on_labels, f"ON top-down (XY) | centroid shift k0→k2 = {on_shift_cm:.2f} cm", on_c)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[lidar-viz][trail] SAVED {out_path}")
    return {"off_shift_cm": off_shift_cm, "on_shift_cm": on_shift_cm, "ratio": ratio}


def _capture_sim_scene(
    base,
    env_id: int,
    off_idx: list[int],
    on_idx: list[int],
    lidar_hits_hist: list[torch.Tensor],
    lidar_dist_hist: list[torch.Tensor],
    max_distance: float,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    out_dir: str,
    max_pts: int,
    range_cap: float | None = None,
) -> None:
    """Best-effort in-scene marker-overlay screenshot (pose-teleport + VisualizationMarkers + persp camera).

    Teleports env ``env_id``'s robot to its RECORDED pose at capture step T (``anchor_pos``/``anchor_quat``,
    default joint pose so the mesh doesn't clip), then overlays the historical world-frame LiDAR hit points
    for OFF (T,T-1,T-2) and ON (T,T-5,T-10) as three colored sphere marker sets (k0=red/k1=green/k2=blue)
    and captures two RGB screenshots. Pure post-hoc visualization (no physics stepping) — cannot desync
    from the already-recorded rollout, but the robot MESH pose is only exact at k0=T; older marker slots
    (k1/k2) intentionally sit wherever the sensor origin actually was at that earlier step, which is the
    point of the trail effect.

    Raises on any failure — caller is expected to catch and fall back to the matplotlib-only output.
    """
    env_ids_t = torch.tensor([env_id], device=base.device, dtype=torch.long)

    # Teleport root pose to the recorded T state; snap joints to default so the mesh doesn't clip terrain.
    pose = torch.cat([anchor_pos, anchor_quat]).unsqueeze(0)  # (1, 7) pos + quat(wxyz)
    base._robot.write_root_pose_to_sim(pose, env_ids=env_ids_t)
    default_jpos = base._robot.data.default_joint_pos[env_ids_t]
    base._robot.write_joint_state_to_sim(default_jpos, torch.zeros_like(default_jpos), env_ids=env_ids_t)

    colors = [(0.85, 0.1, 0.1), (0.1, 0.75, 0.1), (0.1, 0.35, 0.9)]  # k0 red, k1 green, k2 blue
    marker_sets = []
    for i, c in enumerate(colors):
        cfg = SPHERE_MARKER_CFG.copy()
        cfg.prim_path = f"/Visuals/Viz/trail_k{i}"
        cfg.markers["sphere"].radius = 0.025
        cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=c)
        marker_sets.append(VisualizationMarkers(cfg))

    # 3/4 rear-above camera, aimed using the robot's actual heading at T (not a world-fixed offset).
    fwd = quat_apply(anchor_quat, torch.tensor([1.0, 0.0, 0.0], device=base.device))
    fwd[2] = 0.0
    fwd = fwd / (fwd.norm() + 1e-6)
    right = torch.tensor([fwd[1], -fwd[0], 0.0], device=base.device)
    pos_np = anchor_pos.detach().cpu().numpy()
    fwd_np = fwd.detach().cpu().numpy()
    right_np = right.detach().cpu().numpy()
    # Steeper top-down-ish 3/4 angle (tall 'up', modest 'back'), pulled in close (~2-3m) so the
    # near-field (<=near_radius) ring reads clearly instead of being stretched toward a
    # receding-horizon vanishing point (a low eye-level camera on flat ground makes even a few-meter
    # grazing-angle hit look "far" in 2D — pure perspective, not a filter bug). Looking down and
    # slightly ahead keeps the whole range_cap'd ring inside the frame.
    back, side, up = 0.9, 0.8, 1.7
    eye = (
        float(pos_np[0] - fwd_np[0] * back + right_np[0] * side),
        float(pos_np[1] - fwd_np[1] * back + right_np[1] * side),
        float(pos_np[2] + up),
    )
    target = (
        float(pos_np[0] + fwd_np[0] * 0.6),
        float(pos_np[1] + fwd_np[1] * 0.6),
        float(pos_np[2] - 0.3),
    )

    def _aim() -> None:
        # parkour_env.py subscribes its OWN camera-follow callback to the physics-step "pop" event
        # (_camera_follow_callback), which re-aims the persp camera on every render — silently undoing
        # a one-shot set_camera_view() call made before a render loop. parkour_perception_viz.py's
        # aim_camera() sidesteps this by re-asserting the camera EVERY frame, immediately before that
        # frame's render() call; we do the same here.
        base.sim.set_camera_view(eye=eye, target=target)

    _aim()
    # Renderer warm-up: first frames after a teleport + new marker prims commonly come back blank.
    for _ in range(8):
        _aim()
        base.sim.render()
        _ = base.render()

    os.makedirs(out_dir, exist_ok=True)
    for name, idx_group in (("OFF", off_idx), ("ON", on_idx)):
        for marker_vis, idx in zip(marker_sets, idx_group):
            pts = _gather_slot_points(lidar_hits_hist, lidar_dist_hist, idx, max_distance, max_pts, range_cap)
            if pts.shape[0] > 0:
                marker_vis.set_visibility(True)
                marker_vis.visualize(translations=pts)
            else:
                marker_vis.set_visibility(False)
        _aim()
        base.sim.render()
        frame = base.render()
        if frame is None or int(np.asarray(frame).sum()) == 0:
            raise RuntimeError(f"sim-scene render returned an empty frame for {name} capture.")
        out_path = os.path.join(out_dir, f"sim_scene_{name}.png")
        imageio.imwrite(out_path, np.asarray(frame))
        print(f"[lidar-viz][trail] SAVED {out_path}")


def _build_runner(agent_cfg, env):
    class_name = agent_cfg.class_name
    cfg_dict = agent_cfg.to_dict()
    device = agent_cfg.device
    if class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, cfg_dict, log_dir=None, device=device)
    elif class_name == "OnPolicyRunnerParkour":
        return OnPolicyRunnerParkour(env, cfg_dict, log_dir=None, device=device)
    elif class_name == "OnPolicyRunnerAMP":
        return OnPolicyRunnerAMP(env, cfg_dict, log_dir=None, device=device)
    elif class_name == "OnPolicyRunnerAMPBase":
        return OnPolicyRunnerAMPBase(env, cfg_dict, log_dir=None, device=device)
    elif class_name == "OnPolicyRunnerParkourAMP":
        return OnPolicyRunnerParkourAMP(env, cfg_dict, log_dir=None, device=device)
    elif class_name == "OnPolicyRunnerParkourAMPVoxel":
        return OnPolicyRunnerParkourAMPVoxel(env, cfg_dict, log_dir=None, device=device)
    elif class_name == "OnPolicyRunnerParkourAMPLidar":
        return OnPolicyRunnerParkourAMPLidar(env, cfg_dict, log_dir=None, device=device)
    elif class_name == "DistillationRunner":
        return DistillationRunner(env, cfg_dict, log_dir=None, device=device)
    else:
        raise ValueError(f"Unsupported runner class: {class_name}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── force stride=1 (OFF/default) cadence for the ONE rollout we record ──────────────────
    # This is the whole point of "method B": lidar_stack_at_sensor_rate only changes the
    # ring-buffer push stride (env.py _get_observations, lines ~424-441). By recording at
    # stride=1 (freshest possible cadence) and then subsampling in POST-PROCESSING below
    # (k=0,1,2 for OFF vs k=0,5,10 for ON), we reconstruct both cadences from a single
    # trajectory with zero confound from stochastic rollout divergence.
    if hasattr(env_cfg, "lidar_stack_at_sensor_rate"):
        env_cfg.lidar_stack_at_sensor_rate = False
        print("[lidar-viz] env_cfg.lidar_stack_at_sensor_rate forced to False for rollout recording (read-only override).")

    env_cfg.debug_vis = False
    if hasattr(env_cfg.scene, "height_scanner") and hasattr(env_cfg.scene.height_scanner, "debug_vis"):
        env_cfg.scene.height_scanner.debug_vis = False

    resume_path = os.path.abspath(args_cli.sl_checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    env_cfg.log_dir = os.path.dirname(resume_path)

    # Sim-scene (A) capture needs an rgb_array-capable env; matplotlib-only (B) does not.
    want_sim_scene = bool(args_cli.trail) and bool(args_cli.enable_cameras)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if want_sim_scene else None)
    env.unwrapped.set_debug_vis(False)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped  # raw Go2ParkourImitationRandomGoalLidarEnv (delegates through wrappers)

    print(f"[lidar-viz] loading checkpoint: {resume_path} (runner={agent_cfg.class_name})")
    runner = _build_runner(agent_cfg, env)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(base.cfg, "settle_max_steps"):
        base.cfg.settle_max_steps = 0

    obs = env.get_observations()

    # ── rollout: record freshest LiDAR frame (k=0) every control step ───────────────────────
    # When --trail is set, ALSO record raw world-frame hit points (mid360.data.ray_hits_w /
    # .distances) + robot root pose every step — needed to reconstruct the OFF/ON motion-trail
    # point clouds and (optionally) the sim-scene screenshot in post-processing below.
    env_id = args_cli.env_id
    frames: list[np.ndarray] = []  # each (2, H, W) float32 cpu numpy
    lidar_hits_hist: list[torch.Tensor] = []  # each (R, 3) world-frame, GPU (only if --trail)
    lidar_dist_hist: list[torch.Tensor] = []  # each (R,) GPU
    root_pos_hist: list[torch.Tensor] = []  # each (3,) GPU
    root_quat_hist: list[torch.Tensor] = []  # each (4,) wxyz GPU
    with torch.inference_mode():
        for t in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            frame = base._lidar_frame_buf[env_id, 0].detach().clone().cpu().numpy()  # (2, H, W)
            frames.append(frame)
            if args_cli.trail:
                lidar_hits_hist.append(base._mid360.data.ray_hits_w[env_id].detach().clone())
                lidar_dist_hist.append(base._mid360.data.distances[env_id].detach().clone())
                root_pos_hist.append(base._robot.data.root_pos_w[env_id].detach().clone())
                root_quat_hist.append(base._robot.data.root_quat_w[env_id].detach().clone())

    el_min_deg = getattr(base, "_lidar_el_min_deg", -7.0)
    el_max_deg = getattr(base, "_lidar_el_max_deg", 52.0)
    print(f"[lidar-viz] recorded {len(frames)} frames. elevation span=[{el_min_deg:.2f}, {el_max_deg:.2f}] deg")

    # Sim-scene (A) capture needs the env alive for the post-hoc teleport + camera capture;
    # everything else (B, and the default heatmap path) only needs the already-recorded arrays.
    max_distance = float(env_cfg.mid360_lidar.max_distance) if args_cli.trail else None
    if not want_sim_scene:
        env.close()

    # ── pick capture timesteps T by mean-abs-diff(frame[T], frame[T-10]) on ch0 ─────────────
    n = len(frames)
    if n <= 10:
        raise RuntimeError(f"Recorded only {n} frames; need > 10 for a stride-10 comparison. Increase --steps.")
    diffs = []
    for t in range(10, n):
        d = float(np.abs(frames[t][0] - frames[t - 10][0]).mean())
        diffs.append((d, t))
    diffs.sort(key=lambda x: -x[0])

    # de-duplicate candidate T's so multiple images don't reuse near-identical windows.
    chosen: list[int] = []
    for _d, t in diffs:
        if all(abs(t - c) >= 8 for c in chosen):
            chosen.append(t)
        if len(chosen) >= args_cli.num_captures:
            break
    print(f"[lidar-viz] candidate T (by descending ch0 mean-abs-diff over Δt=0.2s): {[(t, round(d, 4)) for d, t in diffs[:5]]}")
    print(f"[lidar-viz] chosen capture timesteps: {chosen}")

    os.makedirs(args_cli.out_dir, exist_ok=True)

    az_extent = [0.0, 360.0]
    results = []
    for T in chosen:
        off_frames = [frames[T], frames[T - 1], frames[T - 2]]
        on_frames = [frames[T], frames[T - 5], frames[T - 10]]

        fig, axes = plt.subplots(3, 3, figsize=(15, 13), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
        fig.suptitle(
            "LiDAR K=3 temporal stack: OFF (stride=1, 0.06s span, near-duplicate frames) "
            "vs ON (stride=5, 0.3s span, real temporal context)\n"
            f"task-network={args_cli.task}  checkpoint={os.path.basename(args_cli.sl_checkpoint)}  "
            f"capture T={T} (control step)  env_id={env_id}",
            fontsize=13,
        )

        row_labels = ["OFF (stride=1)", "ON (stride=5)"]
        dt_labels = {
            0: ["k=0 (Δt=0s)", "k=1 (Δt=-0.02s)", "k=2 (Δt=-0.04s)"],
            1: ["k=0 (Δt=0s)", "k=1 (Δt=-0.10s)", "k=2 (Δt=-0.20s)"],
        }
        stack_sets = [off_frames, on_frames]

        im = None
        for row in range(2):
            for col in range(3):
                ax = axes[row, col]
                ch0 = stack_sets[row][col][0]
                im = ax.imshow(
                    ch0, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto",
                    extent=[az_extent[0], az_extent[1], el_min_deg, el_max_deg], origin="lower",
                )
                ax.set_title(f"{row_labels[row]} {dt_labels[row][col]}", fontsize=10)
                ax.set_xlabel("azimuth (deg)")
                if col == 0:
                    ax.set_ylabel(f"{row_labels[row]}\nelevation (deg)")
        fig.colorbar(im, ax=axes[0:2, :].ravel().tolist(), shrink=0.6, label="ch0 closeness = 1/(1+d)  [0,1]")

        # ── bottom row: |k0 - k2| temporal-diff maps ─────────────────────────────────────
        off_diff = np.abs(off_frames[0][0] - off_frames[2][0])
        on_diff = np.abs(on_frames[0][0] - on_frames[2][0])
        off_mad = float(off_diff.mean())
        on_mad = float(on_diff.mean())
        diff_vmax = max(off_diff.max(), on_diff.max(), 1e-6)

        for col, (label, dmap, mad) in enumerate(
            [("OFF |k0-k2| (Δt=0.04s)", off_diff, off_mad), ("ON |k0-k2| (Δt=0.20s)", on_diff, on_mad)]
        ):
            ax = axes[2, col]
            imd = ax.imshow(
                dmap, cmap="magma", vmin=0.0, vmax=diff_vmax, aspect="auto",
                extent=[az_extent[0], az_extent[1], el_min_deg, el_max_deg], origin="lower",
            )
            ax.set_title(f"{label}\nmean|Δ|={mad:.4f}", fontsize=10)
            ax.set_xlabel("azimuth (deg)")
            ax.set_ylabel("elevation (deg)")
            fig.colorbar(imd, ax=ax, shrink=0.8)
        axes[2, 2].axis("off")
        ratio = on_mad / max(off_mad, 1e-9)
        axes[2, 2].text(
            0.05, 0.5,
            f"ON mean|Δ| / OFF mean|Δ| =\n{ratio:.2f}x\n\n"
            f"OFF mean|Δ|={off_mad:.5f}\nON  mean|Δ|={on_mad:.5f}\n\n"
            "Larger ON diff => ring buffer\ncarries real temporal context;\n"
            "near-zero OFF diff => near-\nduplicate frames (no signal).",
            fontsize=11, va="center", ha="left", family="monospace",
        )

        out_path = os.path.join(args_cli.out_dir, f"lidar_stack_comparison_T{T}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[lidar-viz] SAVED {out_path}  OFF_mean|Δ|={off_mad:.5f}  ON_mean|Δ|={on_mad:.5f}  ratio={ratio:.2f}x")
        results.append((T, out_path, off_mad, on_mad, ratio))

    print("[lidar-viz] SUMMARY:")
    for T, out_path, off_mad, on_mad, ratio in results:
        print(f"  T={T}: {out_path}  OFF={off_mad:.5f}  ON={on_mad:.5f}  ratio={ratio:.2f}x")

    # ── --trail: motion-trail point-cloud viz (matplotlib always; sim-scene screenshot best-effort) ──
    if args_cli.trail:
        assert max_distance is not None  # only unset when --trail is False, see assignment above

        # Pick the trail capture T independently of the heatmap's `chosen` list: the trail effect
        # only shows up while the robot is actually translating, so select the step with the
        # largest planar (xy) displacement over the preceding 10 control steps (same window as the
        # ON cadence's k2 slot) rather than reusing the heatmap's diff-based T, which can land on a
        # near-stationary window and show no trail at all.
        fwd_disp = [
            (float(torch.norm((root_pos_hist[t] - root_pos_hist[t - 10])[:2]).item()), t) for t in range(10, n)
        ]
        fwd_disp.sort(key=lambda x: -x[0])
        T = fwd_disp[0][1]
        print(
            f"[lidar-viz][trail] capture T auto-selected by max planar displacement over last-10 "
            f"steps: T={T} (disp={fwd_disp[0][0]:.3f} m over 0.2s)  [heatmap-diff-based T was {chosen[0]}]"
        )

        off_idx = [T, T - 1, T - 2]
        on_idx = [T, T - 5, T - 10]
        range_cap = args_cli.trail_max_range if args_cli.trail_max_range > 0 else None

        off_pts_w = [
            _gather_slot_points(lidar_hits_hist, lidar_dist_hist, i, max_distance, args_cli.trail_max_pts, range_cap)
            for i in off_idx
        ]
        on_pts_w = [
            _gather_slot_points(lidar_hits_hist, lidar_dist_hist, i, max_distance, args_cli.trail_max_pts, range_cap)
            for i in on_idx
        ]

        anchor_pos = root_pos_hist[T]
        anchor_quat = root_quat_hist[T]
        off_pts_r = [_to_robot_frame(p, anchor_pos, anchor_quat) for p in off_pts_w]
        on_pts_r = [_to_robot_frame(p, anchor_pos, anchor_quat) for p in on_pts_w]

        # Robot-relative near-field filter — this is what actually makes the trail visible (see
        # `_filter_near_field` docstring): far-range hits are dominated by rotational (yaw) parallax
        # that swamps the near-field translational parallax once both are on a shared euclidean scale.
        off_pts_r = [_filter_near_field(p, args_cli.near_radius, args_cli.near_fwd_min) for p in off_pts_r]
        on_pts_r = [_filter_near_field(p, args_cli.near_radius, args_cli.near_fwd_min) for p in on_pts_r]
        print(
            f"[lidar-viz][trail] near-field filter (<{args_cli.near_radius:.1f}m, x_fwd>{args_cli.near_fwd_min:.1f}m) "
            f"point counts — OFF k0/k1/k2={[p.shape[0] for p in off_pts_r]}  "
            f"ON k0/k1/k2={[p.shape[0] for p in on_pts_r]}"
        )

        trail_metrics = _render_trail_matplotlib(
            off_pts_r,
            on_pts_r,
            os.path.join(args_cli.out_dir, "sim_trail_3d.png"),
            T,
            env_id,
            args_cli.near_radius,
            args_cli.near_fwd_min,
        )
        print(
            f"[lidar-viz][trail] T={T}  centroid shift k0->k2:  OFF={trail_metrics['off_shift_cm']:.2f} cm  "
            f"ON={trail_metrics['on_shift_cm']:.2f} cm  ratio(ON/OFF)={trail_metrics['ratio']:.2f}x"
        )

        if want_sim_scene:
            try:
                # NOTE: the rollout's env.step() calls ran inside torch.inference_mode(), which taints
                # the Articulation's internal cached buffers (root_state_w, joint_pos, ...) as "inference
                # tensors". write_root_pose_to_sim / write_joint_state_to_sim do in-place writes into those
                # SAME buffers, which PyTorch forbids outside inference_mode ("Inplace update to inference
                # tensor outside InferenceMode is not allowed") — so this post-hoc teleport+capture must
                # also run inside inference_mode to match.
                sim_scene_range_cap = args_cli.near_radius if args_cli.near_radius > 0 else range_cap
                with torch.inference_mode():
                    _capture_sim_scene(
                        base, env_id, off_idx, on_idx, lidar_hits_hist, lidar_dist_hist, max_distance,
                        anchor_pos, anchor_quat, args_cli.out_dir, args_cli.trail_max_pts, sim_scene_range_cap,
                    )
            except Exception as e:  # best-effort per task spec — matplotlib output above already saved
                print(f"[lidar-viz][trail] sim-scene capture FAILED ({e!r}) — falling back to matplotlib-only trail output.")
            finally:
                env.close()
        else:
            print(
                "[lidar-viz][trail] --enable_cameras not passed; skipping in-scene sim screenshot "
                "(matplotlib trail output above still produced)."
            )


if __name__ == "__main__":
    # `main()` is called with no args because `@hydra_task_config` injects `env_cfg`/`agent_cfg` at
    # runtime (same pattern as scripts/demos/parkour_perception_viz.py); pyright can't see through
    # the decorator's runtime signature rewrite, hence the ignore.
    main()  # type: ignore[call-arg]
    simulation_app.close()
