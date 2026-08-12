# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

MOTION_JIG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/MotionJig/motion_jig.usd",
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
        pos=(0.0, 0.0, 0.26),
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
    },
)


THIGH_VEL = 41.0
THIGH_TORQUE = 22.0
THIGH_P_VEL = 25
THIGH_P_TORQUE = 53.0
THIGH_KP = 300.0
THIGH_KD = 5.0

KNEE_VEL = 25.0
KNEE_TORQUE = 53.0
KNEE_KP = 300.0
KNEE_KD = 5.0

ANKLE_VEL = 51.0
ANKLE_TORQUE = 48.0  # 18.0
ANKLE_KP = 100.0
ANKLE_KD = 5.0

WAIST_VEL = 25.0
WAIST_TORQUE = 106.0
WAIST_KP = 100.0
WAIST_KD = 5.0

NECK_VEL = 16.0
NECK_TORQUE = 7.0
NECK_VEL2 = 51.0
NECK_TORQUE2 = 9.0
NECK_TORQUE3 = 18.0
NECK_KP = 100.0
NECK_KD = 5.0


R_SKELETON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R.Skeleton/R_skeleton.usd",
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R.SkeletonFixed2/R_skeleton.usd",
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R.SkeletonFixed2/R_Skeleton.usd",
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Light/R_Skeleton.usd",
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Collision/R_skeleton.usd",
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Collision2/R_skeleton.usd",
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Collision3/R_skeleton.usd",
        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Collision4/R_skeleton.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=16, solver_velocity_iteration_count=16
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.609),
    ),
    actuator_value_resolution_debug_print=True,  # type: ignore
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_y",
                ".*_shoulder_r",
                ".*_shoulder_p",
                ".*_elbow_p",
                ".*_thigh_y",
                ".*_thigh_r",
                ".*_thigh_p",
                ".*_knee_p",
            ],
            velocity_limit_sim={
                ".*_shoulder_y": THIGH_VEL,
                ".*_shoulder_r": THIGH_VEL,
                ".*_shoulder_p": THIGH_P_VEL,
                ".*_elbow_p": THIGH_P_VEL,
                ".*_thigh_y": THIGH_VEL,
                ".*_thigh_r": THIGH_VEL,
                ".*_thigh_p": THIGH_P_VEL,
                ".*_knee_p": KNEE_VEL,
            },
            effort_limit_sim={
                ".*_shoulder_y": THIGH_TORQUE,
                ".*_shoulder_r": THIGH_TORQUE,
                ".*_shoulder_p": THIGH_P_TORQUE,
                ".*_elbow_p": THIGH_P_TORQUE,
                ".*_thigh_y": THIGH_TORQUE,
                ".*_thigh_r": THIGH_TORQUE,
                ".*_thigh_p": THIGH_P_TORQUE,
                ".*_knee_p": KNEE_TORQUE,
            },
            stiffness={
                ".*_shoulder_y": THIGH_KP,
                ".*_shoulder_r": THIGH_KP,
                ".*_shoulder_p": THIGH_KP,
                ".*_elbow_p": THIGH_KP,
                ".*_thigh_y": THIGH_KP,
                ".*_thigh_r": THIGH_KP,
                ".*_thigh_p": THIGH_KP,
                ".*_knee_p": KNEE_KP,
            },
            damping={
                ".*_shoulder_y": THIGH_KD,
                ".*_shoulder_r": THIGH_KD,
                ".*_shoulder_p": THIGH_KD,
                ".*_elbow_p": THIGH_KD,
                ".*_thigh_y": THIGH_KD,
                ".*_thigh_r": THIGH_KD,
                ".*_thigh_p": THIGH_KD,
                ".*_knee_p": KNEE_KD,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_p", ".*_ankle_r", ".*_wrist_p", ".*_wrist_r"],
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
        # "toe": ImplicitActuatorCfg(
        #     joint_names_expr=[".*toe"],
        #     velocity_limit_sim=0.0,
        #     effort_limit_sim=0.0,
        #     stiffness=0.0,
        #     damping=0.0,
        # ),
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


RGA_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Go2Neck/go2_neck.usd",
        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Go2Neck2/go2_neck.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.34),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=[".*_neck_p", ".*_neck_r", ".*_neck_y"],
            velocity_limit_sim={
                "N_joint1_neck_y": 50,
                "N_joint2_neck_p": 25,
                "N_joint3_neck_y": 25,
                "N_joint4_neck_r": 50,
                "N_joint5_neck_y": 40,
                "N_joint6_neck_p": 40,
                "N_joint7_neck_y": 40,
            },
            effort_limit_sim={
                "N_joint1_neck_y": 10,
                "N_joint2_neck_p": 40,
                "N_joint3_neck_y": 40,
                "N_joint4_neck_r": 10,
                "N_joint5_neck_y": 8.2,
                "N_joint6_neck_p": 8.2,
                "N_joint7_neck_y": 4.1,
            },
            stiffness={
                "N_joint1_neck_y": 25,
                "N_joint2_neck_p": 25.0,
                "N_joint3_neck_y": 25.0,
                "N_joint4_neck_r": 25.0,
                "N_joint5_neck_y": 25.0,
                "N_joint6_neck_p": 25.0,
                "N_joint7_neck_y": 25.0,
            },
            damping={
                "N_joint1_neck_y": 0.5,
                "N_joint2_neck_p": 0.5,
                "N_joint3_neck_y": 0.5,
                "N_joint4_neck_r": 0.5,
                "N_joint5_neck_y": 0.5,
                "N_joint6_neck_p": 0.5,
                "N_joint7_neck_y": 0.5,
            },
        ),
    },
)


R_SKELETON_HIND_LEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Hind_Leg/Leg.usd",
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Hind_Leg_fixed/Leg.usd",
        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/R_Skeleton_Hind_Leg_Fixed_Filpped/Leg.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
    ),
    actuator_value_resolution_debug_print=True,  # type: ignore
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_thigh_r",
                ".*_thigh_p",
                ".*_knee_p",
            ],
            velocity_limit_sim={
                ".*_thigh_r": THIGH_VEL,
                ".*_thigh_p": THIGH_P_VEL,
                ".*_knee_p": KNEE_VEL,
            },
            effort_limit_sim={
                ".*_thigh_r": THIGH_TORQUE,
                ".*_thigh_p": THIGH_P_TORQUE,
                ".*_knee_p": KNEE_TORQUE,
            },
            stiffness={
                ".*_thigh_r": THIGH_KP,
                ".*_thigh_p": THIGH_KP,
                ".*_knee_p": KNEE_KP,
            },
            damping={
                ".*_thigh_r": THIGH_KD,
                ".*_thigh_p": THIGH_KD,
                ".*_knee_p": KNEE_KD,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_p",
                ".*_toe_p",
            ],
            velocity_limit_sim={
                ".*_ankle_p": ANKLE_VEL,
                ".*_toe_p": ANKLE_VEL,
            },
            effort_limit_sim={
                ".*_ankle_p": ANKLE_TORQUE,
                ".*_toe_p": ANKLE_TORQUE,
            },
            stiffness={
                ".*_ankle_p": ANKLE_KP,
                ".*_toe_p": ANKLE_KP,
            },
            damping={
                ".*_ankle_p": ANKLE_KD,
                ".*_toe_p": ANKLE_KD,
            },
        ),
    },
)


HIND_LEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Hind_Leg/hind_leg.usd",
        # usd_path=".../data/Robots/Hind_Leg_RLCal/Hind_Leg_RLCAL_260810/Hind_Leg_RLCAL_260810.usda",  # 구 CAD+RL_INTERFACE 한계 이식본(08-10)
        # usd_path=".../data/Robots/Hind_Leg_URDF2/Hind_Leg/Hind_Leg.usda",  # 2026-08-11: URDF2(신규 CAD 리비전, 08-11 이전 채택)
        #   원본 URDF: /home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF2/urdf/Hind_Leg.urdf
        #   링크 프레임/축(+Y)이 biped MJCF 모델각 규약과 동일(좌우 미러 아님) — sim q ≈ 모델각.
        #   effort 84/84/126/100.8·velocity는 URDF에 이미 실기값. 관절한계는 URDF2 자체값
        #   (hip ±14.9° 등 — RL_INTERFACE.md §3 실측표와 다름, 실기팀 신규 리비전 기준).
        #   질량도 신규(총 ~16.6 kg vs 구 13.57) — 구 모델 기준 kp/kd·정책은 재검토 대상.
        # 2026-08-11: URDF3(같은 날 배포된 추가 CAD 리비전)로 교체.
        #   원본 URDF: /home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF3/urdf/Hind_Leg.urdf
        #   URDF2 대비 관절명/타입/effort/lower/upper 전부 동일 — actuator cfg(velocity_limit 등)는 무수정.
        #   base 링크가 base_collision으로 리네임(질량 5.617→2.8kg)되며, 나머지 링크 10개도 전부
        #   "_link" -> "_link_collision"으로 리네임(코드에서 링크명 직접 참조 없어 무해).
        #   ⚠ URDF3 원본 <limit velocity>가 foot 관절만 24.7→14.8로 퇴행(RL_INTERFACE.md 실기 감속비
        #   재조사로 이미 폐기된 구값과 일치 — CAD 익스포터 쪽 gear ratio 가정이 미갱신인 것으로 추정).
        #   actuator DCMotorCfg의 velocity_limit(24.6)은 URDF와 무관하게 별도 하드코딩이라 영향 없지만,
        #   USD 변환 시 physx.usda의 physxJoint:maxJointVelocity가 14.8 rad/s(847.97754 deg/s)로
        #   baked-in되므로 임포트 후 **수동으로 URDF2 생성본 값(1415.2058 deg/s = 24.7 rad/s)에 맞춰
        #   패치**했다. asset 재생성 시 재적용 필요 — 아래 Newton fix와 함께 CLAUDE.md에 기록.
        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Hind_Leg_URDF3/Hind_Leg/Hind_Leg.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        # joint_pos={
        # # Left Leg (HL)
        # "HL_Hip_Joint": 0.,
        # "HL_Thigh_Joint": -0.2618,
        # "HL_Calf_Joint": -0.8727,
        # "HL_Foot_Joint": -0.8727,
        # # Right Leg (HR)
        # "HR_Hip_Joint": 0.,
        # "HR_Thigh_Joint": 0.2618,
        # "HR_Calf_Joint": 0.8727,
        # "HR_Foot_Joint": 0.8727,
        # }
    ),
    actuator_value_resolution_debug_print=True,  # type: ignore
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # DCMotor (explicit) — PACE sysid와 같은 플랜트 모델을 쓰기 위해 ImplicitActuator에서 교체됨
        # (2026-07-22). PACE가 식별하는 armature/마찰/바이어스/지연은 DC 모터 토크-속도 포화 곡선을
        # 전제로 적합되므로, ImplicitActuator로 두면 식별 결과를 그대로 옮길 수 없다.
        # 짝: source/isaaclab_tasks/.../direct/r2s_biped_leg/r2s_biped_leg_sysid_cfg.py
        #
        # ⚠ saturation_effort는 (관절별 dict가 아니라) 스칼라만 받는다. 여기서는 sysid cfg의
        #   SATURATION_EFFORT_NM과 **같은 값**을 써서 두 플랜트를 일치시킨다. 관절별 곡선이 필요하면
        #   τ_max별로 그룹을 3개(28/42/56)로 쪼개면 되지만, 그 경우 PACE 재적합이 필요하다.
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", ".*_foot_joint"],
            # 실기 무부하 속도한계 (RL_INTERFACE.md §6-c): 모터 207.2 rad/s ÷ 실제감속비 7/7/10.5/8.4.
            # 구값 foot 14.8은 감속비 오인(폐기).
            velocity_limit={
                ".*_hip_joint": 29.6,
                ".*_thigh_joint": 29.6,
                ".*_calf_joint": 19.7,
                ".*_foot_joint": 24.6,
            },
            # 실기 peak 토크 (RL_INTERFACE.md §6-c actuatorfrcrange): 모터 12 N·m × 감속비 7/7/10.5/8.4.
            # 구값 28/28/42/56은 1/3 오인(Leg_URDF2와 같은 함정) — 폐기.
            # ⚠ 런타임 토크 트립(보고토크 15 N·m 50 ms → limp 래치, §6-i)은 이보다 훨씬 낮다.
            #   플랜트 한계와 별개로 정책/보상 설계에서 다뤄야 한다.
            effort_limit={
                ".*_hip_joint": 84.0,
                ".*_thigh_joint": 84.0,
                ".*_calf_joint": 126.0,
                ".*_foot_joint": 100.8,
            },
            saturation_effort=126.0,
            # 실측 반사관성 ROTOR_I 7.4e-4 × 실제감속비² (RL_INTERFACE.md §6-a).
            # hip/thigh 7²=0.0363, calf 10.5²=0.0816, foot 8.4²=0.0522 kg·m².
            # PACE가 관절별 armature를 식별하면 이 값을 덮어쓰면 된다(현재는 실측 nominal).
            armature={
                ".*_hip_joint": 0.0363,
                ".*_thigh_joint": 0.0363,
                ".*_calf_joint": 0.0816,
                ".*_foot_joint": 0.0522,
            },
            # 실측 관절 마찰/감쇠 (hip 2축 실측, 타축 외삽 — RL_INTERFACE.md §6-a).
            # Isaac ≥5.0에서 friction/dynamic_friction은 계수가 아니라 effort [N·m]다.
            # URDF <dynamics friction>은 physx variant 레이어에 묻혀 sim에 반영되지 않는 것을
            # 스모크로 확인(해상표 "Not Specified") → cfg로 명시 주입한다.
            # ⚠ 실기는 Stribeck(정지 0.63~0.71 N·m, 동 0.505~0.575)이라 상수 0.38은
            #   저속을 25~35% 과소평가한다(§6-e) — DR 후보. PACE 식별값이 나오면 덮어쓸 것.
            friction={".*": 0.38},
            dynamic_friction={".*": 0.38},
            viscous_friction={".*": 0.09},
            # PD gains: inertia-scaled (ω_n=20 rad/s, ζ=0.9) from measured per-joint
            # effective inertia I_eff (hip 0.163 / thigh 0.133 / calf 0.030 / foot 0.0019 kg·m²).
            # Previous uniform Kp=25/Kd=0.5 (= Go2 quadruped default) left hip/thigh at ζ≈0.12
            # (under-damped) and foot at ω_n≈113 (over-stiff). foot=20 sized for contact authority
            # (env=HindLegHistoryEnvCfg, 200Hz physics). See _workspace/hind_leg_kp_kd_tuning_guide.md
            stiffness={
                ".*_hip_joint": 65.0,
                ".*_thigh_joint": 53.0,
                ".*_calf_joint": 12.0,
                ".*_foot_joint": 20.0,
            },
            damping={
                ".*_hip_joint": 6.0,
                ".*_thigh_joint": 4.8,
                ".*_calf_joint": 1.1,
                ".*_foot_joint": 1.0,
            },
        ),
    },
)


##
# Leg_URDF2 — 17-DOF 4족 로봇 (다리 4 × 4관절 + 허리 1)
##

# URDF(Leg_URDF2/urdf/Leg.urdf) 의 <limit effort> × 85% [N·m].
#   URDF effort: hip/thigh/waist = 84, calf = 126, foot = 168 (velocity 29.6/29.6/19.7/14.8 도 URDF 그대로).
# 2026-07-28 정정: 이전엔 URDF effort 의 1/3(28/42/56)을 nominal 로 오인해 그 85%(23.8/35.7/47.6)를 썼다.
#   실제 URDF effort 는 3배라, calf/foot 이 저토크 한계에 상시 포화 → 속도 ~3 m/s 천장이 발생했다.
#   이제 URDF effort 기준 85% 로 상향: 84×0.85=71.4, 126×0.85=107.1, 168×0.85=142.8.
LEG_HIP_TORQUE = 71.4
LEG_THIGH_TORQUE = 71.4
LEG_CALF_TORQUE = 107.1
LEG_FOOT_TORQUE = 142.8
LEG_WAIST_TORQUE = 71.4

# 관절별 게인 — `_workspace/leg/compute_leg_ieff.py` 로 USD 에서 유효관성 I_eff 를 실측해 도출.
# (총 질량 38.0 kg, I_eff 스프레드 922배 → 균일 게인은 관절마다 감쇠비가 제각각이 된다.)
#
#   Kd = 2ζ·√(Kp · I_tot),  ζ = 0.9,  I_tot = I_eff + armature
#   hip / thigh : inertia-scaled (Kp = ω_n²·I_tot, ω_n = 20 rad/s) — 자유 스윙 관절
#   calf / foot : 접지 관절. 관성 기반 Kp(15 / 4.6)는 스탠스 하중에 붕괴하므로
#                 trot(2다리 지지) 정적 토크 / 허용처짐 0.25 rad 로 floor
#   waist       : 관성 기반 Kp 513 은 τ_max=28 대비 0.055 rad 에서 포화 → τ_max/0.25 로 cap
#
#   type    I_eff+arm      Kp     Kd   ω_n
#   hip       0.17725      71    6.4  20.0
#   thigh     0.13302      53    4.8  20.0
#   calf      0.03782     134    4.1  59.5
#   foot      0.01157      59    1.5  71.2
#   waist     1.28139     112   21.6   9.3
LEG_ARMATURE = 0.01

LEG_HIP_KP, LEG_HIP_KD = 71.0, 6.4
LEG_THIGH_KP, LEG_THIGH_KD = 53.0, 4.8
LEG_CALF_KP, LEG_CALF_KD = 134.0, 4.1
LEG_FOOT_KP, LEG_FOOT_KD = 59.0, 1.5
LEG_WAIST_KP, LEG_WAIST_KD = 112.0, 21.6

LEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 원본 `Robots/Leg/Leg/Leg.usda` 는 물리 골격만 있고 visual/collision mesh 가 전혀 없어
        # 로봇이 지면을 통과한다. `scripts/tools/convert_urdf.py` 로 Leg_URDF2 를 재임포트한 자산을 쓴다.
        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Leg/Leg_gen/Leg.usd/Leg/Leg.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 다리 간 통과(교차·겹침)를 물리적으로 막는다. convex-hull collision 기준이라
            # 실제 mesh 보다 다소 부풀 수 있어(false-positive 가능) 초기 학습을 관찰해야 한다.
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    # 영자세는 앞/뒤 발끝이 모두 base 아래 약 0.50 m 로 정렬되는 좌우대칭 기립 자세다
    # (순운동학으로 검증). policy obs 의 `joint_pos - default_joint_pos` 기준점이 된다.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim={
                ".*_hip_joint": LEG_HIP_TORQUE,
                ".*_thigh_joint": LEG_THIGH_TORQUE,
                ".*_calf_joint": LEG_CALF_TORQUE,
            },
            velocity_limit_sim=30.0,
            stiffness={
                ".*_hip_joint": LEG_HIP_KP,
                ".*_thigh_joint": LEG_THIGH_KP,
                ".*_calf_joint": LEG_CALF_KP,
            },
            damping={
                ".*_hip_joint": LEG_HIP_KD,
                ".*_thigh_joint": LEG_THIGH_KD,
                ".*_calf_joint": LEG_CALF_KD,
            },
            armature=LEG_ARMATURE,
            friction=0.0,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=LEG_FOOT_TORQUE,
            velocity_limit_sim=30.0,
            stiffness=LEG_FOOT_KP,
            damping=LEG_FOOT_KD,
            armature=LEG_ARMATURE,
            friction=0.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["FB_waist_joint"],
            effort_limit_sim=LEG_WAIST_TORQUE,
            velocity_limit_sim=30.0,
            stiffness=LEG_WAIST_KP,
            damping=LEG_WAIST_KD,
            armature=LEG_ARMATURE,
            friction=0.0,
        ),
    },
)
"""17-DOF 4족 로봇(Leg_URDF2) 설정 — 다리 4개 × (hip, thigh, calf, foot) + 허리 1개."""
