# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import isaaclab.envs.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# ---------------------------------------------------------------------------
# 이벤트 설정
# ---------------------------------------------------------------------------

@configclass
class InteractionEventCfg:
    """도메인 랜덤화 이벤트 설정."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


# ---------------------------------------------------------------------------
# 환경 설정 (Config)
# ---------------------------------------------------------------------------

@configclass
class Go2InteractionCfg(DirectRLEnvCfg):
    """Go2 상호작용 학습 환경 설정.

    Genesis InteractionEnv의 obs/reward/command 설정을 IsaacLab 형식으로 이식했습니다.
    """

    # ------------------------------------------------------------------ #
    # 기본 환경 파라미터
    # ------------------------------------------------------------------ #
    episode_length_s: float = 4.0
    dt: float = 1 / 200           # 물리 서브스텝 dt
    decimation: int = 4            # 제어 주파수 = 200/4 = 50 Hz
    action_scale: float = 0.25
    action_space: int = 12         # Go2 12개 관절
    clip_actions: float = 100.0
    hip_scale_reduction: bool = True
    debug_vis: bool = True

    # ------------------------------------------------------------------ #
    # 관측 공간 구성
    # - prio_obs  : gravity(3) + interaction_cmd(1)
    #               + dof_pos(12) + dof_vel(12) + actions(12) + roll_pitch(2)
    #               = 42
    # - num_priv  : base_euler xyz (3)
    # - num_priv_latent : height(1) + lin_vel(3) + ang_vel(3)
    #                     + dof_pos[:6](6) + dof_vel[:6](6) + torques[:3](3) = 22
    # - history   : prio_obs * history_len
    # ------------------------------------------------------------------ #
    num_prio_obs: int = 42
    num_priv: int = 3
    num_priv_latent: int = 22
    history_len: int = 10

    observation_space: int = num_prio_obs + num_priv + num_priv_latent + num_prio_obs * history_len
    # = 42 + 3 + 22 + 42*10 = 487
    state_space: int = 0

    # ------------------------------------------------------------------ #
    # 관측 스케일
    # ------------------------------------------------------------------ #
    obs_scales: dict = {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    }

    # ------------------------------------------------------------------ #
    # 시뮬레이션 설정
    # ------------------------------------------------------------------ #
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
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

    # ------------------------------------------------------------------ #
    # 씬 설정
    # ------------------------------------------------------------------ #
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # ------------------------------------------------------------------ #
    # 이벤트 (도메인 랜덤화)
    # ------------------------------------------------------------------ #
    events: InteractionEventCfg = InteractionEventCfg()

    # ------------------------------------------------------------------ #
    # 로봇
    # ------------------------------------------------------------------ #
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )

    # ------------------------------------------------------------------ #
    # 커맨드 설정
    # ------------------------------------------------------------------ #
    num_commands: int = 3
    command_cfg: dict = {
        # Interaction 테스크에서는 속도 커맨드를 사용하지 않으므로 비워둡니다.
    }

    # 보상 계산 기준 좌표계 ("world" 또는 "base")
    reward_frame: str = "world"

    # ------------------------------------------------------------------ #
    # 모션 데이터 설정
    # ------------------------------------------------------------------ #
    # 모션 파일은 interaction/ref_motion/ 폴더에 위치
    motion_files: dict = {
        "0": "default.csv",       # 기본 자세
        "1": "sit.csv",           # 앉기
        "2": "lay.csv",           # 눕기
        "3": "stand_cap_v2.csv",  # 일어서기
    }

    # Imitation 스케일 파라미터
    size_prop: float = 0.9
    y_offset: float = 0.07
    z_offset: float = 0.05
    x_offset: float = 0.0
    target_default_hind_x: float = -0.25  # 참조 데이터 정렬용 뒷발 X 좌표 (기존 -0.295에서 조정)

    # Genesis 참조 데이터의 관절 인덱스
    # FL_hip(15), FR_hip(20), RL_hip(4), RR_hip(8)
    ref_hip_indices: list = [15, 20, 4, 8]
    # FL_foot(18), FR_foot(23), RL_foot(7), RR_foot(11)
    ref_foot_indices: list = [18, 23, 7, 11]
    left_indices: list = [15, 18, 4, 7]
    right_indices: list = [20, 23, 8, 11]

    # ------------------------------------------------------------------ #
    # 종료 조건
    # ------------------------------------------------------------------ #
    termination_if_roll_greater_than: float = 1.484   # 85도 (rad)
    termination_if_pitch_greater_than: float = 1.484  # 85도 (rad)

    # 패널티 대상 바디 이름 (Contact 기반)
    penalized_body_names: list = ["base", ".*thigh"]
    termination_body_names: list = ["base"]
    feet_body_names: list = [".*foot"]
    hip_body_names: list = [".*hip"]

    # ------------------------------------------------------------------ #
    # 보상 스케일
    # ------------------------------------------------------------------ #
    reward_sigma: float = 0.2

    # 모션 추적 보상
    hip_positions_reward_scale: float = 0.5
    foot_positions_reward_scale: float = 0.5
    base_height_reward_scale: float = 1.5
    base_pitch_reward_scale: float = 1.5
    feet_contact_reward_scale: float = 0.5
    stand_penalty_reward_scale: float = 1.0

    # 정규화(패널티) 보상
    dof_acc_reward_scale: float = -2.5e-7
    action_rate_reward_scale: float = -0.01
    delta_torques_reward_scale: float = -1.0e-5
    torques_reward_scale: float = -0.0001
    similar_to_default_reward_scale: float = 5.0  # 정적 자세 유지 강화를 위해 상향 (기존 1.0)
    lin_vel_z_reward_scale: float = -1.0
    ang_vel_xy_reward_scale: float = -0.5
    torques_balance_reward_scale: float = -0.01

    # ------------------------------------------------------------------ #
    # 노이즈 모델
    # ------------------------------------------------------------------ #
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="add"),
    )

    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="add"),
    )
