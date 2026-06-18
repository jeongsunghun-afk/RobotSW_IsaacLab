"""
Parkour RL kinematic diagnostic collector — v2 (bug-fixed).

Reads robot.data from a deterministic play loop (no env/reward modification).

Bug fixes vs v1:
  ① Loop: for-loop runs to num_steps; sim_app gate added as safety.
  ② Stride: world-frame foot positions stored at liftoff/touchdown;
             displacement projected onto robot heading at liftoff.
             contact debounce: require DEBOUNCE_STEPS consecutive contact
             before declaring touchdown (avoids bounce false-positives).
  ③ Sanity check: swing-count per leg vs user observation explicitly verified.

Metrics (4 feet [FL=0, FR=1, RL=2, RR=3], _feet_ids order verified by env assertion):
  1. contact_duty  — EMA buffer _foot_contact_duty (post warmup_steps)
  2. joint_pos     — hip/thigh/calf per leg (joint→leg mapping resolved at runtime)
  3. stride_length — world-frame fore-aft displacement liftoff→touchdown
                     projected onto robot heading at liftoff time
  4. foot_clearance— max (foot_z − liftoff_z) during swing (per-foot reference)
  5. air_time      — contact_sensor.data.last_air_time at confirmed touchdown

Also dumps world-frame foot XY time series for cross-check.

Usage:
  conda run -n isaac-5.1 ./isaaclab.sh -p \\
      _workspace/parkour_stride_gait/kinematic/collect_kinematics.py \\
      --task Go2-Parkour-Direct-v0 --num_envs 32 --num_steps 1200 \\
      --checkpoint logs/rsl_rl/go2_parkour/\\
2026-06-05_16-02-33_contact_duty_deficit_v1/model_49999.pt \\
      --headless
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_RSL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
    "scripts", "reinforcement_learning", "rsl_rl",
)
sys.path.insert(0, _RSL_DIR)
import cli_args  # noqa: E402

# ── Argument parsing (before AppLauncher) ────────────────────────────────────
parser = argparse.ArgumentParser(description="Kinematic diagnostic for Go2 Parkour")
parser.add_argument("--task",         type=str, default="Go2-Parkour-Direct-v0")
parser.add_argument("--num_envs",     type=int, default=32)
parser.add_argument("--num_steps",    type=int, default=1200,
                    help="Policy steps to collect. >=1000 recommended (EMA tau=1s=50 steps; "
                         "need >>3×tau=150 steps warmup + many gait cycles).")
parser.add_argument("--warmup_steps", type=int, default=200,
                    help="Steps discarded from contact_duty stats for EMA warmup. "
                         "EMA tau=1s≈50 steps; 4×tau=200 steps recommended.")
parser.add_argument("--seed",         type=int, default=0)
parser.add_argument("--agent",        type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--flat_terrain", action="store_true", default=False,
                    help="Override sub_terrain proportions to flat=1.0 (reproduce only_flat training config).")
parser.add_argument("--out_suffix",   type=str, default="",
                    help="Suffix appended to all output filenames, e.g. '_flat' → summary_flat.md.")
# --checkpoint is added by cli_args.add_rsl_rl_args below (do NOT re-add)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import numpy as np
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunnerParkour
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

OUT_DIR = _HERE
os.makedirs(OUT_DIR, exist_ok=True)

# Contact debounce: require this many consecutive contact steps before declaring touchdown.
# Prevents bounce artefacts (single-step force spike) from registering as strides.
DEBOUNCE_STEPS = 3

LEG_TAGS  = ["FL", "FR", "RL", "RR"]
TYPE_TAGS = ["hip", "thigh", "calf"]


# ─────────────────────────────────────────────────────────────────────────────
def resolve_joint_map(joint_names: list[str]) -> dict[str, dict[str, int]]:
    """Resolve {leg: {type: joint_index}} from joint_names at runtime.

    Prints the full mapping for audit. Asserts every leg×type is found.
    """
    mapping: dict[str, dict[str, int]] = {leg: {} for leg in LEG_TAGS}
    for idx, name in enumerate(joint_names):
        for leg in LEG_TAGS:
            if leg in name:
                for typ in TYPE_TAGS:
                    if typ in name.lower():
                        mapping[leg][typ] = idx
    print("\n[joint_map] Runtime-resolved joint name → index:")
    for leg in LEG_TAGS:
        for typ in TYPE_TAGS:
            idx = mapping[leg].get(typ, "MISSING")
            jname = joint_names[idx] if isinstance(idx, int) else "???"
            print(f"  {leg}_{typ}: [{idx}] = {jname}")
    for leg in LEG_TAGS:
        for typ in TYPE_TAGS:
            assert typ in mapping[leg], f"Joint {leg}_{typ} not found in {joint_names}"
    return mapping


def feet_labels_from_env(base_env) -> list[str]:
    """Return foot body-name labels in _feet_ids order."""
    labels = [base_env._contact_sensor.body_names[i] for i in base_env._feet_ids]
    print(f"\n[feet_ids] body names in _feet_ids order: {labels}")
    return labels


def heading_from_quat(quat_w: np.ndarray) -> np.ndarray:
    """Return unit forward (x-axis) vector in world frame from quaternion (w,x,y,z).

    Go2 uses Isaac convention: quat = (w, x, y, z).
    Forward vector (1,0,0) rotated by quat:
        v' = q * [0,1,0,0] * q_conj  (pure-quaternion sandwich)
    Simplified for unit x: returns (2*(x*z + w*y)*-... standard formula):
        fx = 1 - 2*(y^2 + z^2)
        fy = 2*(x*y + w*z)
    We only need the XY projection normalised.
    """
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    norm = np.sqrt(fx * fx + fy * fy) + 1e-8
    return np.stack([fx / norm, fy / norm], axis=-1)  # (N, 2)


# ─────────────────────────────────────────────────────────────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Main collection loop."""

    # ── Config overrides ──────────────────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.debug_vis = False

    # ── Flat terrain override (--flat_terrain) ───────────────────────────────
    # Reproduces the only_flat training config: parkour_flat proportion=1.0,
    # all other sub_terrains set to 0.0.
    # Must happen BEFORE gym.make (env_cfg is passed into the env constructor).
    # Access path: env_cfg.terrain.terrain_generator.sub_terrains (dict of cfg objects).
    if args_cli.flat_terrain:
        sub_terrains = env_cfg.terrain.terrain_generator.sub_terrains
        terrain_names = list(sub_terrains.keys())
        print(f"[INFO] --flat_terrain: sub_terrains found: {terrain_names}")
        # Set parkour_flat to 1.0, everything else to 0.0
        for name, tcfg in sub_terrains.items():
            if name == "parkour_flat":
                tcfg.proportion = 1.0
                print(f"[INFO]   {name}.proportion = 1.0")
            else:
                tcfg.proportion = 0.0
                print(f"[INFO]   {name}.proportion = 0.0")
        # Verify sum = 1.0 (env __init__ asserts this)
        total = sum(t.proportion for t in sub_terrains.values())
        assert abs(total - 1.0) < 1e-6, f"proportion sum = {total:.4f} ≠ 1.0"
        print(f"[INFO] --flat_terrain override complete: proportion sum = {total:.4f}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = (
            "logs/rsl_rl/go2_parkour/"
            "2026-06-05_16-02-33_contact_duty_deficit_v1/model_49999.pt"
        )
    print(f"[INFO] checkpoint: {resume_path}")

    # ── Environment + policy ──────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    runner = OnPolicyRunnerParkour(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ── Runtime mapping ───────────────────────────────────────────────────────
    joint_names = list(base_env._robot.data.joint_names)
    joint_map   = resolve_joint_map(joint_names)
    foot_labels = feet_labels_from_env(base_env)

    foot_leg_tags: list[str] = []
    for fl in foot_labels:
        matched = next((leg for leg in LEG_TAGS if leg in fl), None)
        assert matched is not None, f"Cannot derive leg tag from '{fl}'"
        foot_leg_tags.append(matched)
    print(f"[INFO] foot→leg: {list(zip(foot_labels, foot_leg_tags))}")

    joint_indices_per_foot = np.array(
        [[joint_map[leg]["hip"], joint_map[leg]["thigh"], joint_map[leg]["calf"]]
         for leg in foot_leg_tags],
        dtype=np.int64,
    )  # (4, 3)
    print(f"[INFO] joint_indices_per_foot:\n{joint_indices_per_foot}")

    feet_ids_list: list[int] = (
        base_env._feet_ids.tolist()
        if hasattr(base_env._feet_ids, "tolist")
        else list(base_env._feet_ids)
    )

    out_suffix = args_cli.out_suffix  # e.g. "_flat"

    N  = base_env.num_envs
    NF = 4
    num_steps    = args_cli.num_steps
    warmup_steps = args_cli.warmup_steps

    # ── Pre-allocate time-series buffers ─────────────────────────────────────
    ts_contact_duty = np.zeros((num_steps, N, NF), dtype=np.float32)
    ts_joint_pos    = np.zeros((num_steps, N, NF, 3), dtype=np.float32)
    ts_foot_z       = np.zeros((num_steps, N, NF), dtype=np.float32)
    ts_foot_xy_w    = np.zeros((num_steps, N, NF, 2), dtype=np.float32)  # world XY (cross-check)
    ts_contact      = np.zeros((num_steps, N, NF), dtype=np.bool_)
    ts_air_time     = np.zeros((num_steps, N, NF), dtype=np.float32)
    ts_env_class    = np.zeros((num_steps, N),     dtype=np.int64)

    # ── Stride / clearance state (script-local, NOT env buffers) ─────────────
    # BUG FIX ②: store WORLD-FRAME foot positions and robot heading at liftoff.
    # Stride = (touchdown_world_xy − liftoff_world_xy) · heading_at_liftoff
    # This is valid across body-frame rotations between liftoff and touchdown.
    in_swing           = np.zeros((N, NF), dtype=np.bool_)
    consec_contact_cnt = np.zeros((N, NF), dtype=np.int32)   # debounce counter
    liftoff_foot_z     = np.zeros((N, NF), dtype=np.float32)  # world-z at liftoff
    liftoff_foot_xy    = np.zeros((N, NF, 2), dtype=np.float32)  # world-XY at liftoff
    liftoff_heading    = np.zeros((N, NF, 2), dtype=np.float32)  # robot heading at liftoff
    swing_peak_z_rel   = np.zeros((N, NF), dtype=np.float32)  # max(foot_z - liftoff_z)

    # stride_events columns: env_idx, foot_idx, stride_x, clearance, air_time_s, env_class, step
    stride_events: list[tuple[int, int, float, float, float, int, int]] = []

    # ── Initial reset ─────────────────────────────────────────────────────────
    obs = env.get_observations()
    in_swing[:]           = False
    consec_contact_cnt[:] = 0
    liftoff_foot_xy[:]    = 0.0
    liftoff_heading[:]    = 0.0
    swing_peak_z_rel[:]   = 0.0

    print(
        f"\n[INFO] Starting collection: {num_steps} steps × {N} envs"
        f" | warmup_discard={warmup_steps} steps | debounce={DEBOUNCE_STEPS} steps"
    )
    print(f"[INFO] step_dt={base_env.step_dt:.4f}s | "
          f"EMA tau=1.0s ≈ {int(round(1.0/base_env.step_dt))} steps | "
          f"warmup covers {warmup_steps*base_env.step_dt:.1f}s = "
          f"{warmup_steps/int(round(1.0/base_env.step_dt)):.1f}×tau")

    # ── Main collection loop (BUG FIX ①: explicit for-loop; sim_app safety gate)
    for step in range(num_steps):
        if not simulation_app.is_running():
            print(f"[WARN] simulation_app stopped at step {step}. Saving collected data.")
            num_steps = step  # trim arrays
            ts_contact_duty = ts_contact_duty[:step]
            ts_joint_pos    = ts_joint_pos[:step]
            ts_foot_z       = ts_foot_z[:step]
            ts_foot_xy_w    = ts_foot_xy_w[:step]
            ts_contact      = ts_contact[:step]
            ts_air_time     = ts_air_time[:step]
            ts_env_class    = ts_env_class[:step]
            break

        with torch.inference_mode():
            # Replicate play.py: override commands every step so resample can't diverge.
            base_env._commands[:, 0] = 1.0
            base_env._commands[:, 1] = 0.0
            base_env._commands[:, 2] = 0.0

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # ── Read raw data ─────────────────────────────────────────────────────
        cduty_np      = base_env._foot_contact_duty.cpu().numpy()             # (N, 4)
        jpos_np       = base_env._robot.data.joint_pos.cpu().numpy()          # (N, 12)
        foot_pos_w_np = (
            base_env._robot.data.body_pos_w[:, feet_ids_list, :]
            .cpu().numpy()
        )  # (N, 4, 3)
        net_cf        = base_env._contact_sensor.data.net_forces_w_history    # (N, hist, B, 3)
        contact_np    = (
            torch.norm(net_cf[:, 0, :, :], dim=-1)[:, feet_ids_list] > 2.0
        ).cpu().numpy()  # (N, 4)
        air_time_np   = (
            base_env._contact_sensor.data.last_air_time[:, feet_ids_list]
            .cpu().numpy()
        )  # (N, 4)
        env_class_np  = base_env._env_class.cpu().numpy()                     # (N,)
        root_pos_w_np = base_env._robot.data.root_pos_w.cpu().numpy()         # (N, 3)
        root_quat_w_np = base_env._robot.data.root_quat_w.cpu().numpy()       # (N, 4) w,x,y,z

        # Robot heading (unit XY forward vector in world frame)
        heading_np = heading_from_quat(root_quat_w_np)  # (N, 2)

        # ── Store time series ─────────────────────────────────────────────────
        ts_contact_duty[step] = cduty_np
        for fi in range(NF):
            ts_joint_pos[step, :, fi, :] = jpos_np[:, joint_indices_per_foot[fi]]
        ts_foot_z[step]    = foot_pos_w_np[:, :, 2]
        ts_foot_xy_w[step] = foot_pos_w_np[:, :, :2]
        ts_contact[step]   = contact_np
        ts_air_time[step]  = air_time_np
        ts_env_class[step] = env_class_np

        # ── Stride / clearance tracking ───────────────────────────────────────
        dones_np = (
            dones.cpu().numpy().astype(np.bool_)
            if hasattr(dones, "cpu")
            else np.asarray(dones, dtype=np.bool_)
        )

        for fi in range(NF):
            foot_z    = foot_pos_w_np[:, fi, 2]     # (N,)
            foot_xy   = foot_pos_w_np[:, fi, :2]    # (N, 2)
            in_contact = contact_np[:, fi]           # (N,) bool

            # Invalidate trackers for reset envs
            if dones_np.any():
                in_swing[dones_np, fi]           = False
                consec_contact_cnt[dones_np, fi] = 0
                liftoff_foot_xy[dones_np, fi]    = foot_xy[dones_np]
                liftoff_heading[dones_np, fi]    = heading_np[dones_np]
                liftoff_foot_z[dones_np, fi]     = foot_z[dones_np]
                swing_peak_z_rel[dones_np, fi]   = 0.0

            # Update consecutive-contact debounce counter
            consec_contact_cnt[:, fi] = np.where(
                in_contact,
                consec_contact_cnt[:, fi] + 1,
                0,
            )
            # Debounced contact: True only after DEBOUNCE_STEPS consecutive contacts
            confirmed_contact = consec_contact_cnt[:, fi] >= DEBOUNCE_STEPS

            prev_in_swing = in_swing[:, fi].copy()

            # Liftoff: was NOT swinging, now airborne (not in contact)
            just_lifted = (~prev_in_swing) & (~in_contact)

            # Touchdown: was swinging, now confirmed contact
            just_landed = prev_in_swing & confirmed_contact

            # Process liftoffs
            lift_envs = np.where(just_lifted)[0]
            if len(lift_envs):
                in_swing[lift_envs, fi]        = True
                liftoff_foot_z[lift_envs, fi]  = foot_z[lift_envs]
                liftoff_foot_xy[lift_envs, fi] = foot_xy[lift_envs]
                liftoff_heading[lift_envs, fi] = heading_np[lift_envs]
                swing_peak_z_rel[lift_envs, fi] = 0.0

            # Update clearance for swinging feet
            swinging = in_swing[:, fi]
            if swinging.any():
                dz = foot_z - liftoff_foot_z[:, fi]
                swing_peak_z_rel[swinging, fi] = np.maximum(
                    swing_peak_z_rel[swinging, fi], dz[swinging]
                )

            # Process touchdowns (BUG FIX ②: world-frame displacement · heading)
            land_envs = np.where(just_landed)[0]
            for env_i in land_envs:
                in_swing[env_i, fi]          = False
                consec_contact_cnt[env_i, fi] = 0  # reset debounce

                # World-frame displacement from liftoff to touchdown
                delta_xy = foot_xy[env_i] - liftoff_foot_xy[env_i, fi]  # (2,)
                # Project onto robot's heading direction at liftoff
                h = liftoff_heading[env_i, fi]  # unit (2,) heading
                stride_x = float(np.dot(delta_xy, h))

                clearance = float(swing_peak_z_rel[env_i, fi])
                at_s      = float(air_time_np[env_i, fi])
                stride_events.append((
                    int(env_i), int(fi), stride_x, clearance,
                    at_s, int(env_class_np[env_i]), int(step),
                ))

        if step % 100 == 0:
            print(
                f"  step {step:4d}/{num_steps}"
                f"  contact_duty(mean): FL={cduty_np[:,0].mean():.3f}"
                f"  FR={cduty_np[:,1].mean():.3f}"
                f"  RL={cduty_np[:,2].mean():.3f}"
                f"  RR={cduty_np[:,3].mean():.3f}"
                f"  | swings_so_far: {len(stride_events)}"
            )

    # ── Save raw dumps ────────────────────────────────────────────────────────
    raw_path = os.path.join(OUT_DIR, f"raw_kinematics{out_suffix}.npz")
    np.savez_compressed(
        raw_path,
        contact_duty   = ts_contact_duty,
        foot_z         = ts_foot_z,
        foot_xy_w      = ts_foot_xy_w,
        contact        = ts_contact,
        air_time       = ts_air_time,
        env_class      = ts_env_class,
        foot_labels    = np.array(foot_labels),
        foot_leg_tags  = np.array(foot_leg_tags),
        warmup_steps   = np.int64(warmup_steps),
        num_steps      = np.int64(num_steps),
        debounce_steps = np.int64(DEBOUNCE_STEPS),
        note           = np.array(
            "v2: stride=world-XY-displacement·liftoff-heading; "
            "clearance=per-foot(z−liftoff_z); "
            "contact debounce=DEBOUNCE_STEPS consecutive steps; "
            "command: lin_vel_x=1.0 fixed each step"
        ),
    )
    print(f"[INFO] Saved raw kinematics → {raw_path}")

    raw_jpos_path = os.path.join(OUT_DIR, f"raw_joint_pos{out_suffix}.npz")
    np.savez_compressed(
        raw_jpos_path,
        joint_pos        = ts_joint_pos,
        joint_indices    = joint_indices_per_foot,
        foot_leg_tags    = np.array(foot_leg_tags),
        joint_names      = np.array(joint_names),
        joint_type_order = np.array(["hip", "thigh", "calf"]),
    )
    print(f"[INFO] Saved raw joint pos → {raw_jpos_path}")

    stride_path = os.path.join(OUT_DIR, f"raw_stride_events{out_suffix}.npz")
    if stride_events:
        se = np.array(stride_events, dtype=np.float64)
        np.savez_compressed(
            stride_path,
            stride_events = se,
            columns       = np.array([
                "env_idx", "foot_idx", "stride_x_m", "clearance_m",
                "air_time_s", "env_class", "step",
            ]),
        )
        print(f"[INFO] Saved stride events → {stride_path}  ({len(stride_events)} events)")
    else:
        print("[WARN] No stride events captured.")

    # ── Summary ───────────────────────────────────────────────────────────────
    _write_summary(
        ts_contact_duty        = ts_contact_duty,
        ts_joint_pos           = ts_joint_pos,
        ts_foot_xy_w           = ts_foot_xy_w,
        stride_events          = stride_events,
        foot_labels            = foot_labels,
        foot_leg_tags          = foot_leg_tags,
        joint_indices_per_foot = joint_indices_per_foot,
        joint_names            = joint_names,
        warmup_steps           = warmup_steps,
        num_steps              = num_steps,
        num_envs               = N,
        out_dir                = OUT_DIR,
        step_dt                = float(base_env.step_dt),
        out_suffix             = out_suffix,
        flat_terrain           = args_cli.flat_terrain,
        checkpoint_path        = resume_path,
    )

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
def _write_summary(
    ts_contact_duty: np.ndarray,
    ts_joint_pos: np.ndarray,
    ts_foot_xy_w: np.ndarray,
    stride_events: list,
    foot_labels: list[str],
    foot_leg_tags: list[str],
    joint_indices_per_foot: np.ndarray,
    joint_names: list[str],
    warmup_steps: int,
    num_steps: int,
    num_envs: int,
    out_dir: str,
    step_dt: float,
    out_suffix: str = "",
    flat_terrain: bool = False,
    checkpoint_path: str = "",
) -> None:
    foot_order = ["FL", "FR", "RL", "RR"]
    joint_types = ["hip", "thigh", "calf"]

    # Post-warmup slices
    wu = min(warmup_steps, num_steps - 1)
    duty_post = ts_contact_duty[wu:]           # (T, N, 4)
    jpos_post = ts_joint_pos[wu:]              # (T, N, 4, 3)
    duty_flat_arr = duty_post.reshape(-1, 4)   # (T*N, 4)  — "_arr" avoids name clash with flat_terrain bool
    jpos_flat_arr = jpos_post.reshape(-1, 4, 3)

    se = np.array(stride_events, dtype=np.float64) if stride_events else None
    tau_steps = max(1, int(round(1.0 / step_dt)))  # EMA tau=1s in steps

    lines: list[str] = []
    lines.append("# Go2 Parkour Kinematic Diagnostic — Summary (v2)")
    lines.append("")
    terrain_tag = "pure flat (parkour_flat proportion=1.0)" if flat_terrain else "default mixed parkour"
    lines.append(f"- Checkpoint: `{os.path.basename(os.path.dirname(checkpoint_path))}/{os.path.basename(checkpoint_path)}`")
    lines.append(f"- Terrain: {terrain_tag}")
    lines.append(f"- num_envs={num_envs}, total_steps={num_steps}, "
                 f"warmup_discarded={wu} ({wu*step_dt:.1f}s = {wu/tau_steps:.1f}×tau)")
    lines.append(f"- step_dt={step_dt:.4f}s | EMA tau≈1s≈{tau_steps} steps | "
                 f"contact debounce={DEBOUNCE_STEPS} steps")
    lines.append(f"- Feet in `_feet_ids` order (env assertion): {foot_labels}")
    lines.append(f"- Foot→leg: {list(zip(foot_labels, foot_leg_tags))}")
    lines.append("- Joint indices [hip, thigh, calf] per foot:")
    for fi, leg in enumerate(foot_order):
        idxs = joint_indices_per_foot[fi].tolist()
        names = [joint_names[i] for i in idxs]
        lines.append(f"  - {leg}: {idxs} = {names}")
    lines.append("")

    # ── 0. Sanity check: contact_duty + air_time lens ────────────────────────
    # "moves often" (swing count) ≠ "moves properly" — use duty/air_time instead.
    # Expected signature of RL abnormality:
    #   (a) contact_duty(RL) ≈ 0.30 floor (EMA pinned at target, never exceeds it)
    #       while duty(RR) > 0.40 (normal hind leg actually swings & lands)
    #   (b) air_time(RL) at touchdown ≈ 0 (foot barely leaves ground) or
    #       median air_time(RL) >> others (foot stays up too long — 3-leg gait)
    lines.append("## Sanity Check: contact_duty + air_time lens")
    lines.append("_User observation: RL 비정상 (나머지 3발은 앞으로 뻗음)_")
    lines.append("_Lens: (a) RL contact_duty ≈ 0.30 floor vs RR > 0.40  "
                 "(b) RL air_time distribution vs RR_")
    lines.append("")

    # (a) post-warmup per-foot duty mean
    duty_means = [float(duty_flat_arr[:, fi].mean()) for fi in range(4)]
    lines.append("Post-warmup contact_duty means:")
    lines.append("| Foot | duty_mean | pinned_at_floor? |")
    lines.append("|------|-----------|-----------------|")
    for fi, leg in enumerate(foot_order):
        pinned = "YES — floor" if duty_means[fi] < 0.34 else ("marginal" if duty_means[fi] < 0.40 else "no")
        lines.append(f"| {leg} | {duty_means[fi]:.4f} | {pinned} |")
    lines.append("")

    # (b) median air_time per foot at touchdown events
    if se is not None and len(se):
        lines.append("Air_time at touchdown (seconds), per foot:")
        lines.append("| Foot | N | median | mean | p90  |")
        lines.append("|------|---|--------|------|------|")
        at_medians: list[float] = []
        for fi, leg in enumerate(foot_order):
            at = se[se[:, 1] == fi, 4]
            if len(at):
                med = float(np.median(at))
                at_medians.append(med)
                p90 = float(np.percentile(at, 90))
                lines.append(f"| {leg} | {len(at)} | {med:.4f} | {at.mean():.4f} | {p90:.4f} |")
            else:
                at_medians.append(0.0)
                lines.append(f"| {leg} | 0 | N/A | N/A | N/A |")
        lines.append("")
    else:
        at_medians = [0.0, 0.0, 0.0, 0.0]
        lines.append("_No touchdown events — air_time lens unavailable._")
        lines.append("")

    # Sanity verdict: RL duty pinned AND (RL air_time outlier OR duty asymmetry vs RR)
    rl_duty  = duty_means[2]
    rr_duty  = duty_means[3]
    fl_duty  = duty_means[0]
    fr_duty  = duty_means[1]
    duty_asymmetry   = rr_duty - rl_duty          # positive = RR more contact than RL
    rl_airtime_med   = at_medians[2]
    rr_airtime_med   = at_medians[3]

    # Three signal conditions
    cond_duty_floor  = rl_duty < 0.34             # RL EMA sitting at/near 0.30 floor
    cond_duty_asym   = duty_asymmetry > 0.05      # RR meaningfully higher than RL
    cond_at_asym     = (rr_airtime_med > 0 and
                        abs(rl_airtime_med - rr_airtime_med) / max(rr_airtime_med, 1e-4) > 0.3)

    signals_fired = sum([cond_duty_floor, cond_duty_asym, cond_at_asym])
    if signals_fired >= 2:
        sanity_flag = (
            f"REPRODUCED ({signals_fired}/3 signals): "
            f"duty_floor={cond_duty_floor}, duty_asym(RR−RL={duty_asymmetry:.3f})={cond_duty_asym}, "
            f"at_asym={cond_at_asym} — RL 비정상 재현됨"
        )
        sanity_ok = True
    elif signals_fired == 1:
        sanity_flag = (
            f"PARTIAL (1/3 signals): "
            f"duty_floor={cond_duty_floor}, duty_asym(RR−RL={duty_asymmetry:.3f})={cond_duty_asym}, "
            f"at_asym={cond_at_asym} — marginal asymmetry"
        )
        sanity_ok = True  # proceed with caveat
    else:
        sanity_flag = (
            f"NOT_REPRODUCED (0/3 signals): all signals negative — "
            f"RL duty={rl_duty:.3f}, RR duty={rr_duty:.3f}, asym={duty_asymmetry:.3f}. "
            "측정 파이프라인 신뢰 불가 — 판정 보류."
        )
        sanity_ok = False

    lines.append(f"**Sanity check result**: {sanity_flag}")
    lines.append("")
    if not sanity_ok:
        lines.append("> **경고**: 측정이 사용자 관찰(RL 비정상)을 재현하지 못함.")
        lines.append("> 아래 모든 판정은 신뢰 불가. 추가 디버깅 필요.")
        lines.append("")

    # ── 1. Contact duty ───────────────────────────────────────────────────────
    lines.append("## Metric 1: contact_duty (EMA, post-warmup)")
    lines.append(f"_EMA tau=1s, warmup={wu} steps ({wu*step_dt:.1f}s) discarded_")
    lines.append("")
    lines.append("| Foot | mean | std  | min  | max  |")
    lines.append("|------|------|------|------|------|")
    for fi, leg in enumerate(foot_order):
        d = duty_flat_arr[:, fi]
        lines.append(f"| {leg} | {d.mean():.4f} | {d.std():.4f} | {d.min():.4f} | {d.max():.4f} |")
    lines.append("")

    # ── 2. Joint angle range ──────────────────────────────────────────────────
    for ti, jt in enumerate(joint_types):
        lines.append(f"## Metric 2{chr(ord('a')+ti)}: joint_pos — {jt} (rad)")
        lines.append("")
        lines.append("| Foot | mean  | min   | max   | range | center |")
        lines.append("|------|-------|-------|-------|-------|--------|")
        for fi, leg in enumerate(foot_order):
            v = jpos_flat_arr[:, fi, ti]
            vmin, vmax = float(v.min()), float(v.max())
            lines.append(
                f"| {leg} | {v.mean():.4f} | {vmin:.4f} | {vmax:.4f} "
                f"| {vmax-vmin:.4f} | {(vmin+vmax)/2:.4f} |"
            )
        lines.append("")

    # ── 3. Stride length ──────────────────────────────────────────────────────
    lines.append("## Metric 3: stride_length (world-frame displacement · liftoff-heading, meters)")
    lines.append("_v2 fix: world-frame liftoff/touchdown positions projected onto heading at liftoff._")
    lines.append("_Positive = forward, negative = backward._")
    lines.append("")
    if se is not None and len(se):
        lines.append("| Foot | N | mean  | std   | min   | max   |")
        lines.append("|------|---|-------|-------|-------|-------|")
        for fi, leg in enumerate(foot_order):
            sx = se[se[:, 1] == fi, 2]
            if len(sx):
                lines.append(
                    f"| {leg} | {len(sx)} | {sx.mean():.4f} | {sx.std():.4f}"
                    f" | {sx.min():.4f} | {sx.max():.4f} |"
                )
            else:
                lines.append(f"| {leg} | 0 | N/A | N/A | N/A | N/A |")
    else:
        lines.append("_No stride events._")
    lines.append("")

    # ── 4. Foot clearance ─────────────────────────────────────────────────────
    lines.append("## Metric 4: foot_clearance (max z − liftoff_z per swing, meters)")
    lines.append("_Per-foot liftoff_z baseline. NOT global terrain reference._")
    lines.append("")
    if se is not None and len(se):
        lines.append("| Foot | N | mean  | std   | min   | max   |")
        lines.append("|------|---|-------|-------|-------|-------|")
        for fi, leg in enumerate(foot_order):
            cl = se[se[:, 1] == fi, 3]
            if len(cl):
                lines.append(
                    f"| {leg} | {len(cl)} | {cl.mean():.4f} | {cl.std():.4f}"
                    f" | {cl.min():.4f} | {cl.max():.4f} |"
                )
            else:
                lines.append(f"| {leg} | 0 | N/A | N/A | N/A | N/A |")
    else:
        lines.append("_No swing events._")
    lines.append("")

    # ── 5. Air time at touchdown ──────────────────────────────────────────────
    lines.append("## Metric 5: air_time at touchdown (seconds)")
    lines.append("")
    if se is not None and len(se):
        lines.append("| Foot | N | mean  | std   | min   | max   |")
        lines.append("|------|---|-------|-------|-------|-------|")
        for fi, leg in enumerate(foot_order):
            at = se[se[:, 1] == fi, 4]
            if len(at):
                lines.append(
                    f"| {leg} | {len(at)} | {at.mean():.4f} | {at.std():.4f}"
                    f" | {at.min():.4f} | {at.max():.4f} |"
                )
            else:
                lines.append(f"| {leg} | 0 | N/A | N/A | N/A | N/A |")
    else:
        lines.append("_No events._")
    lines.append("")

    # ── Verdict ───────────────────────────────────────────────────────────────
    lines.append("## Verdict: 못 한다 vs 안 한다 (1차 판정)")
    lines.append("")
    lines.append("**핵심 비교: RL(index=2) vs RR(index=3) — 같은 뒷다리 역할, 반대쪽**")
    lines.append("")

    RL_idx, RR_idx = 2, 3
    thigh_ti = joint_types.index("thigh")

    for ti, jt in enumerate(joint_types):
        rl_v = jpos_flat_arr[:, RL_idx, ti]
        rr_v = jpos_flat_arr[:, RR_idx, ti]
        rl_r = float(rl_v.max() - rl_v.min())
        rr_r = float(rr_v.max() - rr_v.min())
        ratio = rl_r / (rr_r + 1e-6)
        rl_c  = float((rl_v.max() + rl_v.min()) / 2)
        rr_c  = float((rr_v.max() + rr_v.min()) / 2)
        lines.append(
            f"- **{jt}**: RL range={rl_r:.4f} rad, RR range={rr_r:.4f} rad, "
            f"ratio={ratio:.3f} | RL center={rl_c:.4f}, RR center={rr_c:.4f}"
        )

    rl_duty_mean = float(duty_flat_arr[:, RL_idx].mean())
    rr_duty_mean = float(duty_flat_arr[:, RR_idx].mean())
    lines.append(
        f"- **contact_duty**: RL={rl_duty_mean:.4f}, RR={rr_duty_mean:.4f}, "
        f"diff(RR−RL)={rr_duty_mean - rl_duty_mean:.4f}"
    )
    lines.append("")

    rl_thigh = jpos_flat_arr[:, RL_idx, thigh_ti]
    rr_thigh = jpos_flat_arr[:, RR_idx, thigh_ti]
    rl_thigh_range = float(rl_thigh.max() - rl_thigh.min())
    rr_thigh_range = float(rr_thigh.max() - rr_thigh.min())
    thigh_ratio = rl_thigh_range / (rr_thigh_range + 1e-6)

    lines.append("**판정 기준** (thigh range ratio = RL/RR):")
    lines.append("- ratio < 0.7 → RL range 30%+ 잘림 → **못 한다** (구조적 제약)")
    lines.append("- ratio ≥ 0.7 → range 유사, center만 다름 → **안 한다** (reward 교정 가능)")
    lines.append("")

    if not sanity_ok:
        verdict_str = "**판정 보류** — sanity check FAILED (측정이 사용자 관찰 재현 못 함)"
        primary     = "판정 보류"
    elif thigh_ratio < 0.7:
        verdict_str = (
            f"**못 한다** (thigh ratio={thigh_ratio:.3f} < 0.7; "
            "RL 관절 운동범위가 RR 대비 제한. 구조적 원인 가능성)"
        )
        primary = "못 한다"
    else:
        rl_c = float((rl_thigh.max() + rl_thigh.min()) / 2)
        rr_c = float((rr_thigh.max() + rr_thigh.min()) / 2)
        verdict_str = (
            f"**안 한다** (thigh ratio={thigh_ratio:.3f} ≥ 0.7; "
            f"range 유사, center 차이={abs(rl_c-rr_c):.4f} rad → policy가 RL을 다른 각도 영역 사용)"
        )
        primary = "안 한다"

    lines.append(f"### 1차 판정: {verdict_str}")
    lines.append("")
    lines.append("> 주의: 가설은 가설로 표기. actuator blame 금지(4발 대칭 → RL-only 설명 불가).")
    if rl_duty_mean < 0.35 and rr_duty_mean > 0.40:
        lines.append(
            f"> contact_duty RL({rl_duty_mean:.3f})≈0.30 고착 + RR({rr_duty_mean:.3f})>0.40 → "
            "local optimum 잔재 가설 지지 (가설)."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- v2 stride fix: world-frame liftoff→touchdown XY dot heading_at_liftoff.")
    lines.append(f"- contact debounce: {DEBOUNCE_STEPS} consecutive contact steps before touchdown.")
    lines.append(f"- warmup={wu} steps={wu*step_dt:.1f}s={wu/max(1,int(round(1/step_dt))):.1f}×tau discarded from EMA stats.")
    lines.append("- foot_xy_w dump in raw_kinematics.npz for cross-check of stride pipeline.")

    summary_path = os.path.join(out_dir, f"summary{out_suffix}.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[INFO] Summary → {summary_path}")

    # Stdout key numbers
    print("\n=== KEY DIAGNOSTIC NUMBERS ===")
    print(f"Contact duty (post-warmup): "
          f"FL={duty_flat_arr[:,0].mean():.4f}  FR={duty_flat_arr[:,1].mean():.4f}  "
          f"RL={duty_flat_arr[:,2].mean():.4f}  RR={duty_flat_arr[:,3].mean():.4f}")
    print(f"Thigh range (rad): "
          f"FL={jpos_flat_arr[:,0,1].max()-jpos_flat_arr[:,0,1].min():.4f}  "
          f"FR={jpos_flat_arr[:,1,1].max()-jpos_flat_arr[:,1,1].min():.4f}  "
          f"RL={jpos_flat_arr[:,2,1].max()-jpos_flat_arr[:,2,1].min():.4f}  "
          f"RR={jpos_flat_arr[:,3,1].max()-jpos_flat_arr[:,3,1].min():.4f}")
    print(f"Thigh center (rad): "
          f"FL={(jpos_flat_arr[:,0,1].max()+jpos_flat_arr[:,0,1].min())/2:.4f}  "
          f"FR={(jpos_flat_arr[:,1,1].max()+jpos_flat_arr[:,1,1].min())/2:.4f}  "
          f"RL={(jpos_flat_arr[:,2,1].max()+jpos_flat_arr[:,2,1].min())/2:.4f}  "
          f"RR={(jpos_flat_arr[:,3,1].max()+jpos_flat_arr[:,3,1].min())/2:.4f}")
    print(f"Thigh ratio RL/RR: {thigh_ratio:.3f}")
    print(f"Sanity signals fired: {signals_fired}/3 — {sanity_flag}")
    print(f"1차 판정: {primary}")
    print("==============================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
