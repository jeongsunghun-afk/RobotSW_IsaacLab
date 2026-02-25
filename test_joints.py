import torch
import sys

# IsaacLab App
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import Articulation
from isaaclab_assets.robots.rga import R_SKELETON_CFG

sim_cfg = sim_utils.SimulationCfg()
sim = sim_utils.SimulationContext(sim_cfg)
sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())

robot = Articulation(R_SKELETON_CFG.replace(prim_path="/World/Robot"))
sim.reset()

with open("robot_joints.txt", "w") as f:
    for i, name in enumerate(robot.data.joint_names):
        f.write(f"{i}: {name}\n")

app_launcher.app.close()
