# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

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
        # usd_path=".../data/Robots/Hind_Leg_RLCal/Hind_Leg_RLCAL_260810/Hind_Leg_RLCAL_260810.usda"
        #   — 구 CAD+RL_INTERFACE 한계 이식본(08-10)
        # usd_path=".../data/Robots/Hind_Leg_URDF2/Hind_Leg/Hind_Leg.usda"
        #   — 2026-08-11: URDF2(신규 CAD 리비전, 08-11 이전 채택)
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
        # 2026-08-12: URDF3_SignFix로 교체 — 실기 통신 실측에서 같은 목표에 **반대로 도는 관절**
        #   (HL_hip/HL_calf/HL_foot/HR_hip/HR_thigh)이 확인돼, sim을 실기에 맞추기 위해 URDF3에서
        #   이 5개 관절의 axis 부호와 limit([lo,hi]→[-hi,-lo])을 반전한 파생 자산.
        #   원본 URDF: /home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF3/urdf/Hind_Leg_SignFix.urdf
        #   (저장소 사본: data/Robots/Hind_Leg_URDF3_SignFix/urdf/Hind_Leg.urdf — 동일 내용)
        #   결과적으로 thigh/calf/foot은 좌우 미러 규약이 됐다(soft limit 좌우 다름 — motions.py 참고).
        #   ⚠ 재생성 시 URDF3와 동일한 2개 패치 필요(Newton API 제거 + foot maxJointVelocity 1415.2058)
        #   — 이 파일 위 주석과 task CLAUDE.md 참고. ⚠ 구(舊)규약으로 학습된 정책/수집 데이터는
        #   반전 관절의 부호가 달라 호환되지 않는다.
        # 2026-08-13: SignFix_Col 로 교체 — 충돌 형상을 전용 메시로 분리한 파생본.
        #   기존 SignFix 는 visual 과 collision 이 같은 정밀 메시(링크당 5만~26만 삼각형)를 가리켜
        #   충돌체가 "정밀 메시의 convex hull" 이었고, 실측 부피의 2.2~4.9배로 부풀어 있었다.
        #   구 리그 URDF(04_Hind_Leg_URDF/urdf/03_Leg_UFDF_260617_02.urdf)의 "visual 은 정밀 /
        #   collision 은 단순 볼록 덩어리" 구조를 현재 CAD 에 맞춰 재현했다.
        #   - visual: 정밀 메시 유지 / collision: meshes/collision_simple/ 의 볼록 조각 59개
        #   - 지면에 닿는 발 계열만 조각을 늘려 타이트하게(발 2.20→1.48, 발바닥 1.65→1.46),
        #     base/hip/thigh 는 단일 hull 이라 물리는 종전과 동일(극점 z 오차 0.000 mm)
        #   - 충돌 메시 총량 50 MB(104만 삼각형) → 1.2 MB(2.2만) 로 축소
        #   생성: _workspace/make_hindleg_collision_meshes.py + make_hindleg_collision_urdf.py
        #   원본 URDF: /home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF3/urdf/Hind_Leg_SignFix_Col.urdf
        #   ⚠ 재생성 시 URDF3 계열과 동일한 2개 패치 필요(Newton API 제거 + foot maxJointVelocity 1415.2058)
        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Hind_Leg_URDF3_SignFix_Col/Hind_Leg_SignFix_Col/Hind_Leg_SignFix_Col.usda",
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
            # ★★ 2026-08-26: **PACE 식별값으로 교체.** 여기부터 armature·마찰 세 항은 파생값이
            #   아니라 실기 chirp 재현으로 적합한 값이다.
            #
            #   출처: `logs/pace/bipedleg_0819_biasfrozen/26_08_25_15-27-59/mean_076.pt` 의
            #        **좌우 평균**. 25 차원 적합(encoder bias 8 개를 0 으로 고정)이다.
            #
            #   근거 — kd 5.0 hold-out 4 개(어느 적합에도 안 들어간 캡처)의 관절당 RMS:
            #        stock(종전 값)  0.770°     PACE 좌우평균  0.332°   → **5.4 배**
            #        마찰만 바꾸면 0.628° 이므로 이득의 큰 쪽은 armature 다.
            #        → reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/
            #          README.md §38, metrics/eval_stock_vs_pace.txt
            #
            #   ⚠ **좌우 평균을 쓴다.** 적합 원본은 viscous 좌우비가 hip 3.24 · foot 3.54 인데,
            #     이건 캡처 7 개가 kd 를 공유해 생긴 축퇴 아티팩트임이 밝혀졌다(§37-e: kd 2.5
            #     캡처를 넣으면 1.4 로 무너진다). 대칭화 비용은 hold-out **3.9 %**(0.326→0.332°)
            #     뿐이라, 알려진 아티팩트를 학습 플랜트에 굽는 것보다 싸다.
            #
            #   ⚠ armature 의 물리적 의미가 바뀌었다 — 이제 `I_r·N²` 이 아니라 식별값이다.
            #     파생값 대비 hip ×3.83 / thigh ×2.84 / calf ×1.72 / foot ×1.60.
            #     왜 파생값이 작은지는 §25 참조(로터만 센 값이라 그렇다).
            #   ⚠ **hip 은 아직 식별되지 않았다** — 캡처가 공진대(2.80 Hz)에 못 들어갔다(§34-b).
            #     0.1390 은 그 상태에서 나온 값이니 신뢰도가 다른 셋보다 낮다.
            armature={
                ".*_hip_joint": 0.1390,  # 파생 0.0363 × 3.83  ← ⚠ 미식별(§34-b)
                ".*_thigh_joint": 0.1029,  # 파생 0.0363 × 2.84
                # 벨트가 무릎을 건너므로 무릎이 돌면 foot 로터도 돈다(θ_f = N_f·(q_f + q_c)) ⇒
                # calf 관절은 foot 로터 관성도 짊어진다. 파생 형태는
                #     M_refl = I_r·[[N_c² + N_f²,  N_f²],   =  [[0.1338,  0.0522],
                #                   [N_f²,         N_f²]]       [0.0522,  0.0522]]
                # 이고, 대각 두 개가 여기 armature 다. off-diagonal 은 PhysX 가 표현할 수 없어
                # env 의 `foot_reflected_inertia` 항이 명시적 보정토크로 넣는다.
                # ⚠ env 는 I_off 를 **foot 대각 armature 에서 읽는다** — 파생값에서는 둘이 같은
                #   양(`I_r·N_f²`)이었기 때문이다. 식별값에서는 그 항등식이 더 이상 성립하지
                #   않지만, **sysid env 도 같은 규칙으로 재생하며 적합했으므로**(같은 슬롯을
                #   대각·off-diagonal 양쪽에 씀) 여기 옮겨도 적합 당시와 같은 플랜트가 된다.
                #   규칙을 바꿀 때는 PACE 재적합이 필요하다.
                ".*_calf_joint": 0.2304,  # 파생 0.1338 × 1.72
                ".*_foot_joint": 0.0834,  # 파생 0.0522 × 1.60  (= off-diagonal 로도 읽힌다)
            },
            # 관절 마찰 — 위 armature 와 같은 출처(PACE 0819 좌우평균). 종전 값은 전 관절 균일
            # 0.38 / 0.09 였는데, 그건 hip 2 축 실측 + 타축 외삽이었다(RL_INTERFACE.md §6-a).
            # Isaac ≥5.0에서 friction/dynamic_friction은 계수가 아니라 effort [N·m]다.
            # URDF <dynamics friction>은 physx variant 레이어에 묻혀 sim에 반영되지 않는 것을
            # 스모크로 확인(해상표 "Not Specified") → cfg로 명시 주입한다.
            #
            # ⚠ PACE 는 **정지/동마찰을 구분하지 않는다**(coulomb 한 항). 실기는 Stribeck 이라
            #   (정지 0.63~0.71 N·m, 동 0.505~0.575) 저속에서 여전히 갭이 남는다 — §37-e-2 가
            #   "kd 를 바꾸면 한 플랜트로 둘 다 못 맞춘다"의 원인 후보로 지목한 것이 이것이다.
            friction={
                ".*_hip_joint": 0.7768,
                ".*_thigh_joint": 0.5780,
                ".*_calf_joint": 0.9191,
                ".*_foot_joint": 0.8499,
            },
            dynamic_friction={
                ".*_hip_joint": 0.7768,
                ".*_thigh_joint": 0.5780,
                ".*_calf_joint": 0.9191,
                ".*_foot_joint": 0.8499,
            },
            viscous_friction={
                ".*_hip_joint": 0.6366,
                ".*_thigh_joint": 0.1957,
                ".*_calf_joint": 0.6186,
                ".*_foot_joint": 0.0337,  # ★ 유일하게 종전(0.09)보다 **작다** — ×0.37
            },
            # PD gains — **실기 드라이버 게인의 관절 공간 환산** (2026-08-18).
            #
            # 이전 값(65/53/12/20 + 6/4.8/1.1/1.0)은 구 모델 I_eff 기반 **설계값**이었고 실기와
            # 맞춘 적이 없다. `scripts/real2sim/r2s_biped_leg/motions.py:32-37` 이 그 값을 **폐기**로
            # 표시하고 실기팀 지정값을 기록해 뒀는데, GUI 쪽만 갱신되고 sim 플랜트가 안 따라가
            # 2026-08-12 이후로 어긋난 채였다.
            #
            # 실기 드라이버 게인(**채널** 좌표, `motions.DEFAULT_KP/KD`):
            #     hip(TR) 100/5,  thigh(TP) 50/5,  calf(KP) 50/5,  foot(AP) 20/5
            # GUI 가 이 값을 그대로 발행하고 real_runner 가 변환 없이 드라이버에 넣는다
            # (`real_runner_bipedleg.cpp:481`). 관절 공간 환산은 `kp_ch · gear_k^n`
            # (`convert_gui_chirp_bipedleg.py:203`), gear_k = hip·thigh 1.0 / calf 1.5 / foot 1.2.
            #
            # ⚠ **n = 2 는 채택이지 확정이 아니다** (`GAIN_GEAR_SCALE`/`DEFAULT_GAIN_EXPONENT` 와
            #   같은 베팅, 재적합에서 "약한 지지"). n 이 뒤집히면 calf 112.5→75, foot 28.8→24 다.
            #   calf 는 **어느 n 이든 옛 값 12 보다 4.2~9.4 배 단단하다** — 격차 자체는 n 과 무관.
            #   근거: reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/README.md §13
            stiffness={
                ".*_hip_joint": 100.0,  # 100 × 1.0²
                ".*_thigh_joint": 50.0,  # 50 × 1.0²
                ".*_calf_joint": 112.5,  # 50 × 1.5²
                ".*_foot_joint": 28.8,  # 20 × 1.2²
            },
            damping={
                ".*_hip_joint": 5.0,  # 5 × 1.0²
                ".*_thigh_joint": 5.0,  # 5 × 1.0²
                ".*_calf_joint": 11.25,  # 5 × 1.5²
                ".*_foot_joint": 7.2,  # 5 × 1.2²
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


##
# LEG_DTC — DTC/이산지형 트랙용 R.pet 17-DOF (LEG_CFG 파생)
##

# USD·관절 순서는 AMP 데이터셋(`leg_imitation_tracking`)과 정합을 유지하려고 LEG_CFG 를 그대로
# 물려받고, **액추에이터 한계만** 실기 실측으로 되돌린다. 두 항목 모두 USD 가 아니라 CFG 값이라
# 데이터셋 정합을 깨지 않는다.
#
# 1) foot(발목) effort 142.8 -> 100.8
#    LEG_CFG 는 Leg_URDF2 의 `<limit effort>` 168 에 85% 를 곱한 142.8 을 쓰는데, 168 은 stale 이다.
#    실값은 발목 8.4:1 재기어 실측인 **100.8** 이고, 회사 자신의 최신 8-DOF URDF
#    (`data/Robots/Hind_Leg_URDF3_SignFix/urdf/Hind_Leg.urdf`, 2026-08-12)도 `effort="100.8"` 을 쓴다.
#    142.8 은 실기의 1.42 배라 sim 에서 되는 것이 실기에서 안 된다.
#
# 2) velocity_limit_sim 30.0(전 관절 균일) -> URDF 실값 (hip/thigh 29.6, calf 19.7, foot 14.8)
#    LEG_CFG 는 전 관절 30.0 으로 평탄화했는데, URDF 는 관절마다 다르다. calf 는 1.5 배,
#    foot 은 2.0 배 과대 설정이다. 이산지형은 스윙 각속도가 평지보다 크므로 이 한계가 실제로 걸린다.
#
# 게인(calf Kp 134 / foot Kp 59 등)은 LEG_CFG 를 그대로 쓴다 — I_eff 실측 기반이라 근거가 있다.
# 다만 평지 AMP 기준으로 뽑힌 값이므로 지형 학습 후 재확인 대상이다.

LEG_DTC_FOOT_TORQUE = 100.8
"""발목 실효 토크 [N·m] — 8.4:1 재기어 실측. Leg_URDF2 의 168(=stale)이 아니다."""

LEG_DTC_LEG_VELOCITY_LIMIT = {".*_hip_joint": 29.6, ".*_thigh_joint": 29.6, ".*_calf_joint": 19.7}
LEG_DTC_FOOT_VELOCITY_LIMIT = 14.8
LEG_DTC_WAIST_VELOCITY_LIMIT = 29.6
"""URDF `<limit velocity>` 실값 [rad/s] — 액추에이터 그룹(legs / feet / waist)별로 나눠 둔다."""

# 명시적 deepcopy — configclass 에 copy() 가 없고, 얕은 복사면 actuators dict 를 공유해
# LEG_CFG(회사 AMP 태스크)의 액추에이터까지 같이 바뀐다.
LEG_DTC_CFG = deepcopy(LEG_CFG)
LEG_DTC_CFG.actuators["legs"].velocity_limit_sim = LEG_DTC_LEG_VELOCITY_LIMIT
LEG_DTC_CFG.actuators["feet"].effort_limit_sim = LEG_DTC_FOOT_TORQUE
LEG_DTC_CFG.actuators["feet"].velocity_limit_sim = LEG_DTC_FOOT_VELOCITY_LIMIT
LEG_DTC_CFG.actuators["waist"].velocity_limit_sim = LEG_DTC_WAIST_VELOCITY_LIMIT
"""DTC/이산지형용 17-DOF 설정 — LEG_CFG 파생, 발목 토크·관절별 속도한계만 실측값으로 교체."""
