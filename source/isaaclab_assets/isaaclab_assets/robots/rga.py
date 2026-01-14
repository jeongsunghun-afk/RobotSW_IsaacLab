import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


MOTION_JIG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/home/lgb/IsaacLab/source/isaaclab_assets/data/Robots/MotionJig/motion_jig.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
    ),  
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0., 0., 0.26),
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # damping=0.76,
            # stiffness=7.6,
            damping=1.0,
            stiffness=15.0,
            effort_limit_sim={
                "FL_joint_1": 8.4,
                "FL_joint_2": 8.4,
                "FL_joint_3": 7.1,
                "FL_joint_4": 7.1,
                "FL_joint_5": 7.1,
                "FL_joint_6": 2.5,
                "FL_joint_7": 2.5,
                "FR_joint_1": 8.4,
                "FR_joint_2": 8.4,
                "FR_joint_3": 7.1,
                "FR_joint_4": 7.1,
                "FR_joint_5": 7.1,
                "FR_joint_6": 2.5,
                "FR_joint_7": 2.5,
                "HL_joint_1": 8.4,
                "HL_joint_2": 8.4,
                "HL_joint_3": 7.1,
                "HL_joint_4": 7.1,
                "HL_joint_5": 2.5,
                "HL_joint_6": 2.5,
                "HR_joint_1": 8.4,
                "HR_joint_2": 8.4,
                "HR_joint_3": 7.1,
                "HR_joint_4": 7.1,
                "HR_joint_5": 2.5,
                "HR_joint_6": 2.5,
            },
        ),
    }
)