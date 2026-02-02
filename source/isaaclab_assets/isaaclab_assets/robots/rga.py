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


THIGH_VEL = 41.0
THIGH_TORQUE = 22.0
THIGH_P_VEL = 25
THIGH_P_TORQUE = 53.0
THIGH_KP = 500.0
THIGH_KD = 10.0

KNEE_VEL = 25.0
KNEE_TORQUE = 53.0
KNEE_KP = 300.0
KNEE_KD = 10.0

ANKLE_VEL = 51.0
ANKLE_TORQUE = 48.0 # 18.0
ANKLE_KP = 100.0
ANKLE_KD = 10.0

WAIST_VEL = 25.0
WAIST_TORQUE = 106.0
WAIST_KP = 300.0
WAIST_KD = 10.0

NECK_VEL = 16.0
NECK_TORQUE = 7.0
NECK_VEL2 = 51.0
NECK_TORQUE2 = 9.0
NECK_TORQUE3 = 18.0
NECK_KP = 500.0
NECK_KD = 10.0




R_SKELETON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/home/lgb/IsaacLab/source/isaaclab_assets/data/Robots/R.Skeleton/R_skeleton.usd",
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
        pos=(0., 0., 0.608),
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_y", ".*_shoulder_r", ".*_shoulder_p", ".*_elbow_p",
                               ".*_thigh_y", ".*_thigh_r", ".*_thigh_p", ".*_knee_p"],
            velocity_limit_sim={
                ".*_shoulder_y": THIGH_VEL,
                ".*_shoulder_r": THIGH_VEL,
                ".*_shoulder_p": THIGH_P_VEL,
                ".*_elbow_p": THIGH_P_VEL,
                ".*_thigh_y": THIGH_VEL,
                ".*_thigh_r": THIGH_VEL,
                ".*_thigh_p": THIGH_P_VEL,
                ".*_knee_p": KNEE_VEL
            },
            effort_limit_sim={
                ".*_shoulder_y": THIGH_TORQUE,
                ".*_shoulder_r": THIGH_TORQUE,
                ".*_shoulder_p": THIGH_P_TORQUE,
                ".*_elbow_p": THIGH_P_TORQUE,
                ".*_thigh_y": THIGH_TORQUE,
                ".*_thigh_r": THIGH_TORQUE,
                ".*_thigh_p": THIGH_P_TORQUE,
                ".*_knee_p": KNEE_TORQUE
            },
            stiffness={
                ".*_shoulder_y": THIGH_KP,
                ".*_shoulder_r": THIGH_KP,
                ".*_shoulder_p": THIGH_KP,
                ".*_elbow_p": THIGH_KP,
                ".*_thigh_y": THIGH_KP,
                ".*_thigh_r": THIGH_KP,
                ".*_thigh_p": THIGH_KP,
                ".*_knee_p": KNEE_KP
            },
            damping={
                ".*_shoulder_y": THIGH_KD,
                ".*_shoulder_r": THIGH_KD,
                ".*_shoulder_p": THIGH_KD,
                ".*_elbow_p": THIGH_KD,
                ".*_thigh_y": THIGH_KD,
                ".*_thigh_r": THIGH_KD,
                ".*_thigh_p": THIGH_KD,
                ".*_knee_p": KNEE_KD
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_p", ".*_ankle_r",
                              ".*_wrist_p", ".*_wrist_r"],
            velocity_limit_sim={
                ".*_ankle_p": ANKLE_VEL,
                ".*_ankle_r": ANKLE_VEL,
                ".*_wrist_p": ANKLE_VEL,
                ".*_wrist_r": ANKLE_VEL,
                },
            effort_limit_sim={
                ".*_ankle_p": ANKLE_TORQUE,
                ".*_ankle_r": ANKLE_TORQUE,
                ".*_wrist_p": ANKLE_TORQUE,
                ".*_wrist_r": ANKLE_TORQUE,
                },
            stiffness={
                ".*_ankle_p": ANKLE_KP,
                ".*_ankle_r": ANKLE_KP,
                ".*_wrist_p": ANKLE_KP,
                ".*_wrist_r": ANKLE_KP,
                },
            damping={
                ".*_ankle_p": ANKLE_KD,
                ".*_ankle_r": ANKLE_KD,
                ".*_wrist_p": ANKLE_KD,
                ".*_wrist_r": ANKLE_KD,
                },
        ),
        "toe": ImplicitActuatorCfg(
            joint_names_expr=[".*toe"],
            velocity_limit_sim=0.,
            effort_limit_sim=0.,
            stiffness=100.,
            damping=10.,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[".*_neck_p", ".*_neck_r", ".*_neck_y"],
            velocity_limit_sim={
                "N_joint1_neck_y": NECK_VEL,
                "N_joint2_neck_p": NECK_VEL,
                "N_joint3_neck_y": NECK_VEL,
                "N_joint4_neck_r": NECK_VEL2,
                "N_joint5_neck_y": NECK_VEL2,
                "N_joint6_neck_p": NECK_VEL2,
                "N_joint7_neck_y": NECK_VEL2,
            },
            effort_limit_sim={
                "N_joint1_neck_y": NECK_TORQUE,
                "N_joint2_neck_p": NECK_TORQUE,
                "N_joint3_neck_y": NECK_TORQUE,
                "N_joint4_neck_r": NECK_TORQUE2,
                "N_joint5_neck_y": NECK_TORQUE3,
                "N_joint6_neck_p": NECK_TORQUE3,
                "N_joint7_neck_y": NECK_TORQUE2,
            },
            stiffness={
                "N_joint1_neck_y": NECK_KP,
                "N_joint2_neck_p": NECK_KP,
                "N_joint3_neck_y": NECK_KP,
                "N_joint4_neck_r": NECK_KP,
                "N_joint5_neck_y": NECK_KP,
                "N_joint6_neck_p": NECK_KP,
                "N_joint7_neck_y": NECK_KP,
            },
            damping={
                "N_joint1_neck_y": NECK_KD,
                "N_joint2_neck_p": NECK_KD,
                "N_joint3_neck_y": NECK_KD,
                "N_joint4_neck_r": NECK_KD,
                "N_joint5_neck_y": NECK_KD,
                "N_joint6_neck_p": NECK_KD,
                "N_joint7_neck_y": NECK_KD,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[".*_waist_p", ".*_waist_r", ".*_waist_y"],
            velocity_limit_sim={
                ".*_waist_p": WAIST_VEL,
                ".*_waist_r": WAIST_VEL,
                ".*_waist_y": WAIST_VEL,
            },
            effort_limit_sim={
                ".*_waist_p": WAIST_TORQUE,
                ".*_waist_r": WAIST_TORQUE,
                ".*_waist_y": WAIST_TORQUE,
            },
            stiffness={
                ".*_waist_p": WAIST_KP,
                ".*_waist_r": WAIST_KP,
                ".*_waist_y": WAIST_KP,
            },
            damping={
                ".*_waist_p": WAIST_KD,
                ".*_waist_r": WAIST_KD,
                ".*_waist_y": WAIST_KD,
            },
        ),
    },
)