# HindLeg (8-DOF biped leg) PACE sim2real env cfg.
# Mirrors anymal_pace_env_cfg.py but uses HIND_LEG_CFG (our robot) with the
# actuator group swapped to a PaceDCMotor over the 8 leg joints, and GT/bounds
# shrunk to our small-joint scale. joint_order is left empty and derived from the
# articulation sim order at runtime by the collection/fit scripts.
import torch

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.rga import HIND_LEG_CFG
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg

# PaceDCMotor over the 8 leg joints. Kp/Kd kept from HIND_LEG_CFG (65/53/12/20, 6/4.8/1.1/1).
# saturation_effort / effort_limit / velocity_limit taken from HIND_LEG_CFG effort/velocity limits.
HINDLEG_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", ".*_foot_joint"],
    saturation_effort=60.0,  # scalar (DCMotor requires scalar); >= max effort_limit (56)
    effort_limit={".*_hip_joint": 28.0, ".*_thigh_joint": 28.0, ".*_calf_joint": 42.0, ".*_foot_joint": 56.0},
    velocity_limit={".*_hip_joint": 29.6, ".*_thigh_joint": 29.6, ".*_calf_joint": 19.7, ".*_foot_joint": 14.8},
    stiffness={".*_hip_joint": 65.0, ".*_thigh_joint": 53.0, ".*_calf_joint": 12.0, ".*_foot_joint": 20.0},
    damping={".*_hip_joint": 6.0, ".*_thigh_joint": 4.8, ".*_calf_joint": 1.1, ".*_foot_joint": 1.0},
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=10,
)

N_JOINTS = 8


@configclass
class HindLegPaceCfg(PaceCfg):
    robot_name: str = "hindleg_sim"
    data_dir: str = "hindleg_sim/chirp_data.pt"
    bounds_params: torch.Tensor = torch.zeros((4 * N_JOINTS + 1, 2))  # 33 params
    joint_order: list = []  # derived from articulation at runtime

    def __post_init__(self):
        N = N_JOINTS
        # armature (reflected rotor inertia) [kg m2]
        self.bounds_params[0:N, 0] = 1e-4
        self.bounds_params[0:N, 1] = 0.15
        # viscous friction [Nm s/rad]
        self.bounds_params[N:2 * N, 0] = 0.0
        self.bounds_params[N:2 * N, 1] = 0.5
        # coulomb / dry friction [Nm]
        self.bounds_params[2 * N:3 * N, 0] = 0.0
        self.bounds_params[2 * N:3 * N, 1] = 0.15
        # encoder bias [rad]
        self.bounds_params[3 * N:4 * N, 0] = -0.06
        self.bounds_params[3 * N:4 * N, 1] = 0.06
        # global actuation delay [sim steps]
        self.bounds_params[4 * N, 0] = 0.0
        self.bounds_params[4 * N, 1] = 10.0


@configclass
class HindLegPaceSceneCfg(PaceSim2realSceneCfg):
    robot: ArticulationCfg = HIND_LEG_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
        actuators={"legs": HINDLEG_PACE_ACTUATOR_CFG},
    )


@configclass
class HindLegPaceEnvCfg(PaceSim2realEnvCfg):
    scene: HindLegPaceSceneCfg = HindLegPaceSceneCfg()
    sim2real: PaceCfg = HindLegPaceCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.0025  # 400Hz
        self.decimation = 1
