# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from . import hind_leg_events

##
# Pre-defined configs
##
from isaaclab_assets.robots.rga import HIND_LEG_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# foot 링크의 관절축 기준 유효 관성 [kg·m²] — armature(로터 반사관성)를 뺀 링크분만.
# rga.py `HIND_LEG_CFG` kp/kd 주석의 실측 I_eff(foot 0.0019)에서 왔다.
# raw 좌표 마찰(`foot_raw_friction`)의 explicit 적분 안정 캡에만 쓴다.
# ⚠ r2s_biped_leg_env_cfg.FOOT_LINK_INERTIA_KGM2 와 값 일치 필수(다른 패키지라 중복 정의).
FOOT_LINK_INERTIA_KGM2: float = 0.0019


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # URDF3부터 base 링크가 base_collision으로 리네임됨 — 구·신 자산 겸용 패턴.
            "asset_cfg": SceneEntityCfg("robot", body_names="base.*"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # ------------------------------------------------------------------
    # 플랜트 DR — 게인·armature·관절마찰을 **한 인자로 연동** (2026-08-14)
    #
    # 이 한 항이 두 가지를 동시에 한다.
    #
    # (a) 지수 불확실성을 강건성 축으로: 드라이버가 전 축을 7:1 로 가정해 각도를 주고받으므로
    #     실효 관절강성이 명목 kp 의 **k^n** 배다(gear_k = 실제감속비/7 = hip·thigh 1.0 /
    #     calf 1.5 / foot 1.2). 지수 n 이 1 인지 2 인지는 **현 데이터로 원리적으로 판정 불가**다
    #     — 게인 오차가 armature·마찰의 재스케일로 흡수돼 잔차 신호가 잡음 바닥의 1/3~1/21 이다.
    #     ★ 진짜 n=1 은 (armature, viscous, coulomb, kp, kd) 가 **전부** ×1/k 인 점이다.
    #       게인만 ×1/k 하면 ω_n 이 √(1/k) 배 떨어져 **다른 로봇**이 된다 — 즉 게인만 흔드는 DR 은
    #       일반 강건성은 줘도 지수 불확실성을 **겨냥하지 못한다**. 그래서 다섯 물성에 같은 인자를
    #       곱해 그 다양체 위에서만 움직인다. (2026-08-14 이전의 게인 3항 분할안을 이것으로 대체)
    #
    # (b) 랜덤화 누락 갭: 종전 EventCfg 에는 `randomize_joint_parameters` 가 아예 없어서
    #     **PACE 가 식별하는 주요 물성 두 개(armature·관절마찰)가 전혀 랜덤화되지 않고 있었다.**
    #     지수 문제와 무관한 sim2real 갭이며 이 항이 함께 닫는다.
    #
    # factor[e, j] = s_exponent[e, kind(j)] × u_general[e, j]
    #   s_exponent : 베르누이(불확실성이 이진이므로 연속구간보다 정확).
    #                hip/thigh 1 고정 · calf {1/1.5, 1} · foot {1/1.2, 1}.
    #                ⚠ **좌우 다리는 같은 s** — 같은 로봇이니 지수가 다리마다 다를 수 없다.
    #   u_general  : 기존 플랜트 DR. log_uniform(0.75, 1.5), 관절마다 독립.
    #
    # ⚠ 항을 **하나로** 유지할 것. 게인 term 을 따로 덧붙이면 (ⓐ 지수 불확실성을 두 번 세고
    #   (ⓑ 겹치는 term 은 곱해지지 않고 나중 것이 이긴다(events.py:1240 — 캐시된 default 로
    #   리셋 후 scale). 어느 쪽이 이길지도 실행 순서에 달리게 된다.
    #
    # mode="reset" 유지: 이 env 는 priv_latent/history_observation 을 갖는 RMA 계열이라 적응 모듈이
    #   에피소드 안에서 플랜트를 추정한다 — 에피소드별 샘플링이 그 구조의 표준이다.
    # ------------------------------------------------------------------
    robot_coupled_plant_scale = EventTerm(
        func=hind_leg_events.randomize_coupled_plant_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "general_distribution_params": (0.75, 1.5),
            "distribution": "log_uniform",
            # True 면 calf·foot 이 같은 베르누이 비트를 쓴다. 지수 n 은 드라이버 펌웨어 하나의
            # 성질이라 물리적으로는 이쪽이 맞지만(좌우를 공통으로 두는 논거와 동일), 지정 스펙이
            # 종류별 독립 추첨이라 기본값은 False 로 둔다.
            "exponent_shared_across_kinds": False,
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base.*"),
            "com_range": {
                "x": (-0.08, 0.08),
                "y": (-0.04, 0.04),
                "z": (-0.02, 0.02),
            },
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class HindLegFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 1
    action_scale = 0.25
    action_space = 26

    priv_explicit = False
    priv_latent = False
    ang_vel = False
    friction_terrain = False
    timing_parameter = False
    clock_inputs = False
    prev_actions = False
    history_observation = False

    num_prio_obs = 3 + 3 + 3 + action_space * 3

    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4

    num_heights = 0

    num_priv = 3 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 0

    observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len

    state_space = 0

    penalzied_body_names = [
        "base",
        "FL_link_1",
        "FL_link_2",
        "FL_link_3",
        "FL_link_4",
        "FL_link_5",
        "FL_link_6",
        "FR_link_1",
        "FR_link_2",
        "FR_link_3",
        "FR_link_4",
        "FR_link_5",
        "FR_link_6",
        "HL_link_1",
        "HL_link_2",
        "HL_link_3",
        "HL_link_4",
        "HL_link_5",
        "HR_link_1",
        "HR_link_2",
        "HR_link_3",
        "HR_link_4",
        "HR_link_5",
    ]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 40,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = HIND_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.001
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -1.0
    similar_to_default_reward_scale = -0.01
    base_height_reward_scale = -10.0
    termination_reward_scale = -100.0
    # (A) Phase-scheduled stance reward: bonus when scheduled-stance foot is in contact (scale > 0)
    # (B) Phase-scheduled swing clearance reward: bonus when scheduled-swing foot is lifted (scale > 0)
    # (C) Contact-gated anti-slip penalty: stance foot horizontal velocity → penalty (scale < 0)
    foot_slip_reward_scale = -0.15  # < 0: penalty; at 0.93 m/s → ~0.13 per foot per step
    # Gait clock parameters
    gait_period = 0.6  # seconds; full gait cycle duration
    gait_swing_height = 0.07  # metres; clearance ramp saturates at this lift above sole_rest_z
    gait_phase_sharpness = 4.0  # tanh sharpness; higher = more square-wave swing/stance boundary
    gait_stance_reward_scale = 1.0  # > 0: bonus
    gait_swing_reward_scale = 1.0  # > 0: bonus
    # Standing detection thresholds: when command magnitude is below these values, gait is suppressed
    standing_vel_threshold = 0.1  # ‖cmd_xy‖ below this → standing candidate (matches feet_air_time gate)
    standing_yaw_threshold = 0.1  # |yaw command| below this → standing confirmed
    rel_standing_envs: float = 0.05  # fraction of resampled envs forced to cmd=0 (standing) each resample step

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # Command Definition
    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [0.0, 1.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [0.0, 1.0]
    lin_vel_y_range = [-0.0, 0.0]
    ang_vel_range = [-0.5, 0.5]


@configclass
class HindLegHistoryEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 8

    priv_explicit = True
    priv_latent = True
    ang_vel = False
    friction_terrain = True
    timing_parameter = False
    clock_inputs = True
    prev_actions = False
    history_observation = True

    num_prio_obs = 3 + 3 + action_space * 3

    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4

    num_heights = 0

    num_priv = 6 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 10

    # observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len
    observation_space = num_prio_obs

    state_space = 0

    penalzied_body_names = ["base.*", ".*hip.*", ".*thigh.*", ".*calf.*"]

    # foot↔calf 전달기구 커플링 (실기: foot 모터가 raw각 q_foot+q_calf를 구동, RL_INTERFACE coef=+1).
    # r2s_biped_leg live 모드에서 검증된 모델을 학습 액추에이션에도 적용해 sim-실기를 일치시킨다.
    foot_coupling: bool = True

    # 전치 토크 τ_calf += τ_foot_motor — r2s_biped_leg `foot_transpose`와 같은 스위치(기본 True).
    # 2026-08-14 실기 구조 확인: calf/foot 모터가 둘 다 허벅지에 있고 foot은 **무릎을 건너는 1:1
    # 벨트**로 발목을 돈다 ⇒ 모터 출력각 θ_f = q_foot + q_calf. 기구 구속이므로 일률 보존에서
    # Q_calf += τ_θf 가 강제된다 — 위치 커플링을 넣으면 이 항도 넣어야 한다.
    # (이전에는 hind_leg_env._apply_action에 플래그 없이 하드코딩돼 있었다. 기본값 True = 종전 동작.)
    # ★ 2026-08-18: 전치 토크는 foot 게인 재계산이 아니라 foot 액추에이터가 실제로 낸 직전 스텝
    #   토크(`data.applied_torque`)를 쓴다. 옛 식은 정적 effort_limit(100.8 N·m)로 클램프해
    #   DCMotor 속도 곡선 밖의 토크를 calf 에 실을 수 있었다.
    #   ⚠ **학습 플랜트가 바뀐다 — 기존 run 과 완전히 같은 플랜트가 아니다.** 완주 정책으로 학습
    #     env 를 계측한 실측(표본 194.6만): 접촉이 붙으면 foot q̇ 가 **24.70 rad/s** 까지 가고
    #     19.67% 의 스텝이 교차점을 넘는다. 구식이 실제로 곡선을 벗어난 건 **0.012%** 로 드물지만,
    #     벗어날 때 최대 **101 N·m** 를 calf 에 실었다 — `obs_clip` 을 넣게 만든 종류의 희귀
    #     폭주 이벤트다. 평균 거동은 사실상 같겠지만 동일 플랜트로 취급하면 안 된다.
    #     근거: reports/_comparisons/pace_bipedleg_foot_coupling_probe/README.md §11
    foot_transpose: bool = True

    # foot 마찰을 raw(모터축) 좌표로 — r2s_biped_leg `foot_raw_friction`과 같은 스위치(기본 True).
    # 감속기·벨트 마찰은 모터축 θ̇_f = q̇_f + q̇_c 에 앉아 있고, 같은 일률 보존 규칙에 의해 foot·calf
    # 양쪽에 같은 부호로 실린다. foot 관절의 PhysX 마찰(HIND_LEG_CFG friction 0.38 / viscous 0.09)을
    # 0으로 끄고 그 값을 b_raw/c_raw로 재해석해 raw 좌표에서 건다 — sysid 플랜트(PACE)와 동일 모델.
    # ⚠ 이 플래그는 **학습 플랜트를 바꾼다**. 이전 hind_leg run과는 같은 플랜트가 아니다.
    foot_raw_friction: bool = True
    foot_raw_friction_vel_eps: float = 0.2  # sign(w_raw) 완화 폭 [rad/s] (tanh(w/eps), 채터링 방지)

    # 학습 신호 클리핑 — 희귀 물리 폭주 이벤트의 극단 obs/reward가 GAE bootstrap을 타고
    # value loss 지수 발산을 일으키는 것을 차단 (2026-08-12 signfix_coupled v1/v2 파국).
    # 정상 신호 범위(|joint_vel|≲30, |reward|≲1/step) 밖에서만 작동.
    obs_clip: float = 100.0
    reward_clip: float = 10.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    # PD 게인은 실기팀 지정 운용값(2026-08-12, motions.py DEFAULT_KP/KD·real_runner calib_bipedleg.hpp와
    # 동일: hip 100/5, thigh 50/5, calf 50/5, foot 20/5)으로 덮어쓴다 — 실기 드라이버 kd 클램프가 [0,5]라
    # rga.py 기본값(hip kd 6.0 등)은 그대로 배포할 수 없다. 학습-배포 게인 일치가 sim2real 전제.
    robot: ArticulationCfg = HIND_LEG_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "legs": HIND_LEG_CFG.actuators["legs"].replace(
                stiffness={
                    ".*_hip_joint": 100.0,
                    ".*_thigh_joint": 50.0,
                    ".*_calf_joint": 50.0,
                    ".*_foot_joint": 20.0,
                },
                damping={".*": 5.0},
            )
        },
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.001
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -0.0
    similar_to_default_reward_scale = -0.1
    base_height_reward_scale = -10.0
    termination_reward_scale = -100.0
    # (A) Phase-scheduled stance reward: bonus when scheduled-stance foot is in contact (scale > 0)
    # (B) Phase-scheduled swing clearance reward: bonus when scheduled-swing foot is lifted (scale > 0)
    # (C) Contact-gated anti-slip penalty: stance foot horizontal velocity → penalty (scale < 0)
    foot_slip_reward_scale = -0.15  # < 0: penalty; at 0.93 m/s → ~0.13 per foot per step
    # Gait clock parameters
    gait_period = 0.6  # seconds; full gait cycle duration
    gait_swing_height = 0.07  # metres; clearance ramp saturates at this lift above sole_rest_z
    gait_phase_sharpness = 4.0  # tanh sharpness; higher = more square-wave swing/stance boundary
    gait_stance_reward_scale = 1.0  # > 0: bonus
    gait_swing_reward_scale = 1.0  # > 0: bonus
    # 접지 기준선 추정에 쓸 발별 접지 표본 수. 리셋 직후 공중 자세가 섞이지 않도록 접지 중인
    # 발만 표본하며, 이 개수가 모이면 기준선을 확정한다 (평지 상수라 이후 갱신 없음).
    sole_rest_min_samples: int = 20000
    # Standing detection thresholds: when command magnitude is below these values, gait is suppressed
    standing_vel_threshold = 0.1  # ‖cmd_xy‖ below this → standing candidate (matches feet_air_time gate)
    standing_yaw_threshold = 0.1  # |yaw command| below this → standing confirmed
    rel_standing_envs: float = 0.1  # fraction of resampled envs forced to cmd=0 (standing) each resample step

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # Command Definition
    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [-0.5, 2.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [-0.5, 2.0]
    lin_vel_y_range = [-0.0, 0.0]
    ang_vel_range = [-0.5, 0.5]


@configclass
class HindLegRoughEnvCfg(HindLegFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
