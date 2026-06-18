"""
Extract generalized mass matrix diagonal (I_eff) for HIND_LEG_CFG actuated joints.

Usage:
    ./isaaclab.sh -p _workspace/extract_hind_leg_inertia.py --headless

Outputs I_eff = H[i,i] for each of the 8 actuated joints in two poses:
  Pose A: all joints at 0 (default)
  Pose B: natural stance angles
"""

# ── AppLauncher MUST be first, before any isaaclab/torch imports ──────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Extract HIND_LEG_CFG inertia diagonal")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── All isaaclab imports AFTER app launch ─────────────────────────────────────
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from isaaclab_assets.robots.rga import HIND_LEG_CFG  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
def build_scene(disable_gravity: bool = False):
    """Create a minimal sim scene with the HIND_LEG robot."""
    sim_cfg = SimulationCfg(dt=1 / 200)
    sim = SimulationContext(sim_cfg)

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # Distant light
    light_cfg = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/Light", light_cfg)

    # Robot — single env, no env-specific path
    robot_cfg = HIND_LEG_CFG.replace(prim_path="/World/Robot")
    if disable_gravity:
        robot_cfg = robot_cfg.replace(
            spawn=robot_cfg.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                )
            )
        )
    robot = Articulation(robot_cfg)

    sim.reset()
    return sim, robot


def get_mass_matrix(robot: Articulation):
    """Try available PhysX view APIs to get the generalized mass matrix."""
    view = robot.root_physx_view

    # Print available methods for diagnostics
    methods = [m for m in dir(view) if "mass" in m.lower() or "inertia" in m.lower()]
    print(f"    PhysX view mass/inertia methods: {methods}")

    mat = None
    used_api = None
    for api_name in ["get_generalized_mass_matrices", "get_mass_matrices"]:
        if hasattr(view, api_name):
            try:
                result = getattr(view, api_name)()
                mat = result
                used_api = api_name
                break
            except Exception as e:
                print(f"    {api_name} failed: {e}")

    if mat is None:
        print("    WARNING: No mass matrix API found — will use link mass fallback")
    else:
        print(f"    Used API: {used_api},  shape: {mat.shape},  dtype: {mat.dtype}")

    return mat, used_api


def anchor_joint_block(diag, total_mass):
    """
    Find where the 6 base DOFs end by looking for the 3 translational entries
    ≈ total_mass. Returns the start index of joint DOFs.
    """
    # Translational base DOFs (3) have diagonal ≈ total_mass; rotational (3) are much smaller.
    # Walk from index 0 and find a run of 3 consecutive values near total_mass.
    threshold = total_mass * 0.5
    for start in range(len(diag) - 2):
        if (diag[start] > threshold and
                diag[start + 1] > threshold and
                diag[start + 2] > threshold):
            # Found the 3 translational base entries
            joint_start = start + 6  # 3 trans + 3 rot = 6
            return joint_start, start
    # Fallback: assume standard floating-base layout (indices 0-5 = base)
    return 6, 0


def measure_pose(sim: SimulationContext, robot: Articulation, pose_name: str,
                 joint_positions: dict | None = None):
    """Set robot to a pose, step, and extract mass matrix diagonal."""
    print(f"\n{'='*60}")
    print(f"  POSE: {pose_name}")
    print(f"{'='*60}")

    joint_names = robot.data.joint_names
    num_joints = len(joint_names)
    print(f"  robot.data.joint_names ({num_joints}): {joint_names}")

    # ── Set joint positions ───────────────────────────────────────────────────
    joint_pos = torch.zeros(1, num_joints, device=robot.device)
    joint_vel = torch.zeros(1, num_joints, device=robot.device)

    if joint_positions is not None:
        mapping_applied = {}
        for target_name, angle_rad in joint_positions.items():
            # Case-insensitive substring match
            matched = [
                i for i, n in enumerate(joint_names)
                if target_name.lower() in n.lower()
            ]
            if len(matched) == 1:
                joint_pos[0, matched[0]] = angle_rad
                mapping_applied[joint_names[matched[0]]] = angle_rad
            elif len(matched) > 1:
                print(f"    WARNING: '{target_name}' matched multiple joints: "
                      f"{[joint_names[i] for i in matched]}; skipping")
            else:
                print(f"    WARNING: '{target_name}' matched no joint; skipping")
        print(f"  Pose B mapping applied:")
        for jn, ang in mapping_applied.items():
            print(f"    {jn:30s}: {ang:.4f} rad ({ang * 180/3.14159:.2f} deg)")

    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # ── Step sim and update ───────────────────────────────────────────────────
    sim.step()
    robot.update(sim.cfg.dt)

    # ── Verify pose ───────────────────────────────────────────────────────────
    actual_pos = robot.data.joint_pos[0].cpu()
    print(f"\n  Pose verification (target vs actual):")
    for i, jn in enumerate(joint_names):
        target = joint_pos[0, i].item()
        actual = actual_pos[i].item()
        drift = abs(actual - target)
        flag = " <<DRIFT>>" if drift > 0.02 else ""
        print(f"    [{i:2d}] {jn:30s}  target={target:+.4f}  actual={actual:+.4f}  "
              f"drift={drift:.4f}{flag}")

    # ── Link masses ───────────────────────────────────────────────────────────
    masses = robot.root_physx_view.get_masses()  # (num_envs, num_bodies)
    total_mass = masses[0].sum().item()
    print(f"\n  Body masses (kg):")
    for i, bname in enumerate(robot.data.body_names):
        print(f"    [{i:2d}] {bname:30s}: {masses[0, i].item():.4f} kg")
    print(f"  Total mass: {total_mass:.4f} kg")

    # ── Mass matrix ───────────────────────────────────────────────────────────
    mat, used_api = get_mass_matrix(robot)

    if mat is not None:
        # mat shape: (num_envs, ndof, ndof) or similar
        # Take env 0
        M = mat[0]  # (ndof, ndof)
        diag = torch.diag(M).cpu().numpy()
        ndof = len(diag)
        print(f"\n  Full mass matrix diagonal ({ndof} entries):")
        for i, val in enumerate(diag):
            print(f"    diag[{i:2d}] = {val:.6f}")

        # Anchor joint block
        joint_start, trans_start = anchor_joint_block(diag, total_mass)
        print(f"\n  Anchoring: translational base block starts at [{trans_start}], "
              f"joint DOFs start at [{joint_start}]")
        print(f"  (Entries [{trans_start}:{trans_start+3}] ≈ total mass {total_mass:.3f} kg: "
              f"{diag[trans_start]:.3f}, {diag[trans_start+1]:.3f}, {diag[trans_start+2]:.3f})")

        # Extract the 8 actuated joint diagonals
        joint_diag = diag[joint_start: joint_start + num_joints]

        print(f"\n  ┌{'─'*60}┐")
        print(f"  │  I_eff (H[i,i], kg·m²) — {pose_name:<30s}│")
        print(f"  ├{'─'*20}┬{'─'*15}┬{'─'*22}┤")
        print(f"  │ {'Joint name':<18s} │ {'I_eff (kg·m²)':>13s} │ {'Effort lim (Nm)':>20s} │")
        print(f"  ├{'─'*20}┬{'─'*15}┬{'─'*22}┤")

        # Effort limits from cfg
        effort_lims = {
            "hip":   28.0,
            "thigh": 28.0,
            "calf":  42.0,
            "foot":  56.0,
        }

        for i, jn in enumerate(joint_names):
            i_eff = joint_diag[i] if i < len(joint_diag) else float("nan")
            tau = next((v for k, v in effort_lims.items() if k in jn.lower()), float("nan"))
            print(f"  │ {jn:<18s} │ {i_eff:>13.6f} │ {tau:>20.1f} │")
        print(f"  └{'─'*20}┴{'─'*15}┴{'─'*22}┘")

        # Monotonicity sanity check
        print(f"\n  Monotonicity check (proximal > distal expected):")
        for i, jn in enumerate(joint_names):
            i_eff = joint_diag[i] if i < len(joint_diag) else float("nan")
            print(f"    {jn:<30s}: {i_eff:.6f} kg·m²")

        return joint_diag, joint_names

    else:
        # Fallback: compute distal subtree inertia from link masses + CoM positions
        print("\n  FALLBACK: mass matrix unavailable; using simplified distal-mass estimate.")
        print("  (This is approximate — only accounts for point masses, not rotational inertia)")
        # TODO: implement if needed
        return None, joint_names


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  HIND_LEG_CFG — Generalized Mass Matrix Diagonal (I_eff) Extraction")
    print("="*70)

    # Build scene with gravity disabled to prevent pose drift during measurement
    sim, robot = build_scene(disable_gravity=True)

    # Print actuator info from robot data
    print("\n  Actuator stiffness/damping from robot.data:")
    if hasattr(robot.data, "joint_stiffness"):
        stiff = robot.data.joint_stiffness[0].cpu()
        damp = robot.data.joint_damping[0].cpu()
        for i, jn in enumerate(robot.data.joint_names):
            print(f"    {jn:<30s}: Kp={stiff[i].item():.2f}  Kd={damp[i].item():.4f}")

    # ── Pose A: all zeros ─────────────────────────────────────────────────────
    diag_A, joint_names = measure_pose(sim, robot, "Pose A (all-zero)")

    # ── Pose B: natural stance ────────────────────────────────────────────────
    # Matching by substring against actual joint_names
    # rga.py init_state comment: HL_Hip:0, HL_Thigh:-0.2618, HL_Calf:-0.8727, HL_Foot:-0.8727
    #                             HR_Hip:0, HR_Thigh:+0.2618, HR_Calf:+0.8727, HR_Foot:+0.8727
    pose_b = {
        "HL_Hip":   0.0,
        "HL_Thigh": -0.2618,
        "HL_Calf":  -0.8727,
        "HL_Foot":  -0.8727,
        "HR_Hip":    0.0,
        "HR_Thigh":  0.2618,
        "HR_Calf":   0.8727,
        "HR_Foot":   0.8727,
    }
    diag_B, _ = measure_pose(sim, robot, "Pose B (natural stance)", joint_positions=pose_b)

    # ── Summary comparison ────────────────────────────────────────────────────
    if diag_A is not None and diag_B is not None:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: I_eff comparison (kg·m²)")
        print(f"{'='*70}")
        print(f"  {'Joint':<20s}  {'Pose A':>12s}  {'Pose B':>12s}  {'Diff%':>8s}")
        print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*8}")
        for i, jn in enumerate(joint_names):
            a = diag_A[i] if i < len(diag_A) else float("nan")
            b = diag_B[i] if i < len(diag_B) else float("nan")
            diff_pct = (b - a) / max(abs(a), 1e-9) * 100
            print(f"  {jn:<20s}  {a:>12.6f}  {b:>12.6f}  {diff_pct:>+7.1f}%")

    # ── Velocity limits and effort limits from cfg ───────────────────────────
    print(f"\n{'='*70}")
    print(f"  Actuator limits (from HIND_LEG_CFG)")
    print(f"{'='*70}")
    vel_lims = {"hip": 29.6, "thigh": 29.6, "calf": 19.7, "foot": 14.8}
    eff_lims = {"hip":  28.0, "thigh":  28.0, "calf":  42.0, "foot":  56.0}
    print(f"  {'Joint type':<12s}  {'vel_lim (rad/s)':>16s}  {'eff_lim (Nm)':>14s}  {'Kp':>8s}  {'Kd':>8s}")
    print(f"  {'-'*12}  {'-'*16}  {'-'*14}  {'-'*8}  {'-'*8}")
    for jtype in ["hip", "thigh", "calf", "foot"]:
        print(f"  {jtype:<12s}  {vel_lims[jtype]:>16.1f}  {eff_lims[jtype]:>14.1f}  "
              f"{'25.0':>8s}  {'0.5':>8s}")

    simulation_app.close()


if __name__ == "__main__":
    main()
