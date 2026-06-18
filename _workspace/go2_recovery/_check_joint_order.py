"""Go2 joint order ground-truth diagnostic script.

Usage:
    ./isaaclab.sh -p _workspace/go2_recovery/_check_joint_order.py --headless

Prints joint_names in PhysX/USD articulation order (index 0~11) and default_joint_pos.
This is the ground truth for Go2 DOF indexing — do NOT substitute with inference.
"""

"""Launch Isaac Sim first (AppLauncher must be called before any Isaac imports)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 joint order diagnostic")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── now safe to import Isaac / IsaacLab ──────────────────────────────────────
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # noqa: E402


def main():
    # Minimal sim context — CPU is fine for joint name query
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 0.0, 1.0], target=[0.0, 0.0, 0.0])

    # Spawn Go2 at origin
    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/Robot")
    robot: Articulation = Articulation(robot_cfg)

    # Ground plane (needed for stable articulation initialization)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # Light
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg_light.func("/World/Light", cfg_light)

    # Initialize sim (this triggers USD/PhysX articulation parsing)
    sim.reset()
    robot.reset()

    # ── GROUND TRUTH OUTPUT ──────────────────────────────────────────────────
    joint_names = robot.data.joint_names  # type: ignore[attr-defined]
    default_joint_pos = robot.data.default_joint_pos  # (1, 12)

    print("\n" + "=" * 60)
    print("Go2 joint_names (PhysX/USD articulation order)")
    print("=" * 60)
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}]  {name}")

    print("\n" + "-" * 60)
    print("default_joint_pos (rad):")
    dp = default_joint_pos[0].tolist()
    for i, (name, val) in enumerate(zip(joint_names, dp)):
        print(f"  [{i:2d}]  {name:30s}  {val:+.4f}")

    print("\n" + "-" * 60)
    print("Calf indices (where 'calf' in name):")
    calf_ids = [i for i, n in enumerate(joint_names) if "calf" in n]
    print(f"  calf_ids = {calf_ids}")
    print(f"  2::3     = {list(range(2, 12, 3))}")
    print(f"  Match: {calf_ids == list(range(2, 12, 3))}")

    print("\nHip indices (where 'hip' in name):")
    hip_ids = [i for i, n in enumerate(joint_names) if "hip" in n]
    print(f"  hip_ids  = {hip_ids}")
    print(f"  0,3,6,9  = [0, 3, 6, 9]")
    print(f"  Match: {hip_ids == [0, 3, 6, 9]}")

    print("\nThigh indices (where 'thigh' in name):")
    thigh_ids = [i for i, n in enumerate(joint_names) if "thigh" in n]
    print(f"  thigh_ids = {thigh_ids}")
    print(f"  1::3      = {list(range(1, 12, 3))}")
    print(f"  Match: {thigh_ids == list(range(1, 12, 3))}")

    # joint_weights pattern check
    print("\n" + "-" * 60)
    print("joint_weights pattern [hip=1.0, thigh=0.75, calf=0.5] x 4:")
    pattern = [1.0, 0.75, 0.5] * 4
    print("  Assumed (leg-by-leg grouping):", pattern)
    actual_pattern = []
    for i, name in enumerate(joint_names):
        if "hip" in name:
            actual_pattern.append(1.0)
        elif "thigh" in name:
            actual_pattern.append(0.75)
        elif "calf" in name:
            actual_pattern.append(0.5)
        else:
            actual_pattern.append(0.0)
    print("  Actual (by joint type):       ", actual_pattern)
    print(f"  joint_weights pattern Match: {pattern == actual_pattern}")

    print("=" * 60 + "\n")

    # One physics step to confirm no crash
    sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
