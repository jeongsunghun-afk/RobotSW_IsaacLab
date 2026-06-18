# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 AMP 환경 설정."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# 모션 파일 디렉토리 (stmr_go2.py 출력 경로와 일치해야 함)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset")
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset_50")
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset_walk")
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset_smr")
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset_single")
MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "go2")  # pkl (walk/run/trot/pace)


@configclass
class Go2AmpEnvCfg(DirectRLEnvCfg):
    """Go2 AMP imitation 학습 환경 설정.

    AMP 관측 벡터 구성 (amp_observation_space = 43):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        lin_vel(3) + ang_vel(3) + key_body_pos(12)

    Policy 관측 벡터 구성 (observation_space = 42):
        projected_gravity_b(3) + commands(3) +
        joint_pos - default(12) + joint_vel(12) + actions(12)
    """

    # 에피소드
    episode_length_s = 20.0
    sim_dt = 200.0
    policy_dt = 50.0
    decimation = (int)(sim_dt / policy_dt)

    # 공간
    observation_space = 42
    action_space = 12  # Go2 DOF 수
    state_space = 0
    num_amp_observations = 10
    amp_observation_space = 43

    # History & Privileged
    history_observation = True
    history_len = 10
    priv_latent = True
    num_priv_obs = 74  # lin_vel(3)+ang_vel(3)+masses(17)+material(51) ≈ 74

    # 조기 종료
    early_termination = True
    termination_height = 0.1  # Go2 기본 높이 0.34m, 절반 이하이면 넘어짐으로 판정
    contact_force_threshold = 500.0  # base 접촉 판정 임계값 (N), MimicKit 기준
    pose_termination = False  # reference pose 이탈 시 조기 종료 (DeepMimic 방식, AMP에서는 비권장)
    pose_termination_dist = 0.5  # key body 최대 허용 이탈 거리 (m)
    roll_termination_deg = 70.0  # roll 각도 termination 임계값 (deg)
    pitch_termination_deg = 70.0  # pitch 각도 termination 임계값 (deg)
    action_scale = 0.25

    # 모션
    motion_file: str = MOTION_FILES_DIR
    reference_body = "base"
    reset_strategy = "default"  # {"default", "random", "random-start"}

    # 시뮬레이션 (200 Hz physics, 50 Hz policy) -> (60hz physics, 10hz policy)
    # sim: SimulationCfg = SimulationCfg(
    #     dt=1 / sim_dt,
    #     render_interval=decimation,
    #     # physx=PhysxCfg(
    #     #     gpu_found_lost_pairs_capacity=2**23,
    #     #     gpu_total_aggregate_pairs_capacity=2**23,
    #     # ),
    # )
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_dt,
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
    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # 로봇
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # 접촉 센서
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [0.0, 3.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }

    # RSI-Command Matching + In-episode Velocity Curriculum
    # Stage 1 (step < start):  RSI matching만, command 변동 없음
    # Stage 2 (start ~ end):   RSI ± delta 점진 증가
    # Stage 3 (step > end):    RSI ± delta_end (전체 범위)
    # 기본값: task_reward_lerp anneal(10000 iter × 24 steps = 240000) 이후 시작
    command_delta_start: float = 0.0  # Stage 1: 변동 없음
    command_delta_end: float = 5.0  # Stage 3: 전체 범위
    command_curriculum_start_step: int = 240000  # curriculum 시작 (common_step_counter 기준)
    command_curriculum_end_step: int = 600000  # curriculum 완료 (~25000 iter × 24 steps)
    command_resample_interval: int = 200  # 에피소드 중 재샘플링 주기 (env steps)
    command_soft_update_alpha: float = 0.5  # 새 command 반영 비율 (1=즉시, 0=유지)

    # Tracking 보상
    tracking_sigma = 0.25
    # lin_vel_reward_scale = 1.0
    # yaw_rate_reward_scale = 0.5
    lin_vel_reward_scale = 1.0 * 1.0 / (0.02 * 6)
    yaw_rate_reward_scale = 0.5 * 1.0 / (0.02 * 6)


@configclass
class Go2AmpSimpleEnvCfg(Go2AmpEnvCfg):
    """Go2 AMP 단순 버전 환경 설정.

    history_observation=False, ActorCritic(RMA 없음), PPOAMPBase(PPO 기반)와 함께 사용.
    priv_latent는 유지하여 비대칭 actor-critic(critic만 특권 관측 사용) 구조 유지.
    """

    history_observation = False
