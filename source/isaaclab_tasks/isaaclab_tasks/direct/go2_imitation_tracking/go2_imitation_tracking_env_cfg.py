# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Imitation Tracking (AMP + Body-frame Velocity Tracking) 환경 설정."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "smr_mirror_pkl")

# ---------------------------------------------------------------------------
# PACE 시스템 식별 결과 (mean_094, 26_07_20 재적합 · viscous 상한 5.0)
# ---------------------------------------------------------------------------
# scripts/real2sim/validate_go2.py 에서 hold-out RMSE 0.017 rad(nominal 대비 9.4×)로 검증된
# 실기 GO2 액추에이터 플랜트. 관절 타입별 값(다리별 편차 <5%라 타입 대표값으로 반영).
# ⚠ kp=25/kd=0.5 는 유지한다 — viscous 가 kd 오차를 흡수하도록 함께 식별된 조합이므로
#   (validate 가 재현한 조합) 게인을 바꾸면 안 된다.
PACE_ARMATURE: dict[str, float] = {"hip": 0.173, "thigh": 0.168, "calf": 0.200}  # [kg·m²]
PACE_VISCOUS: dict[str, float] = {"hip": 2.41, "thigh": 2.33, "calf": 2.49}  # [N·m·s/rad]
PACE_COULOMB: dict[str, float] = {"hip": 0.023, "thigh": 0.010, "calf": 0.020}  # [N·m]
PACE_ENCODER_BIAS_MAG: float = 0.05  # [rad] 식별된 |encoder bias| 대표값 → 관측 DR 범위의 근거
PACE_KP: float = 25.0  # 식별 당시 게인 (UNITREE_GO2_CFG 와 동일)
PACE_KD: float = 0.5


@configclass
class DomainRandCfg:
    """Domain Randomization 설정 — 식별된 PACE 값을 중심으로 ± 범위 랜덤화.

    잔여 sim2real gap(모델링 안 된 마찰 비선형·질량 오차·지연·센서 노이즈)에 정책이 강건해지도록
    한다. 물리 파라미터·질량·마찰은 env 별로 리셋마다 재샘플(각 env = 서로 다른 로봇),
    push·관측 노이즈는 스텝 단위 동적 적용.
    """

    # ── 물리 파라미터 (per-env, 리셋마다 재샘플) ──────────────────
    randomize_mass: bool = True
    added_base_mass_range: tuple[float, float] = (-1.0, 2.0)  # base 링크 payload [kg]
    randomize_material: bool = True
    foot_friction_range: tuple[float, float] = (0.4, 1.4)  # 발 static·dynamic 마찰 계수
    randomize_armature: bool = True
    armature_scale_range: tuple[float, float] = (0.8, 1.2)  # PACE armature 곱 스케일
    randomize_joint_friction: bool = True
    joint_friction_scale_range: tuple[float, float] = (0.8, 1.2)  # PACE viscous·Coulomb 공통 스케일

    # ── 액추에이터 게인·지연 ─────────────────────────────────────
    randomize_gains: bool = True
    kp_scale_range: tuple[float, float] = (0.9, 1.1)
    kd_scale_range: tuple[float, float] = (0.9, 1.1)
    randomize_action_delay: bool = True
    max_action_delay_steps: int = 1  # policy step(50Hz=20ms) 단위. 식별 delay~1ms 는 0 step에 근접

    # ── 지형·외란 ────────────────────────────────────────────────
    push_robot: bool = True
    push_interval_s: float = 5.0  # 외란 주입 주기 [s]
    max_push_vel_xy: float = 1.0  # base 수평 속도 킥 최대 [m/s]

    # ── 관측 노이즈 (policy obs 에만 적용, AMP obs 불변) ──────────
    obs_noise: bool = True
    joint_pos_noise: float = 0.01  # [rad]
    joint_vel_noise: float = 0.5  # [rad/s]
    lin_vel_noise: float = 0.1  # [m/s]
    ang_vel_noise: float = 0.2  # [rad/s]
    gravity_noise: float = 0.05  # projected_gravity 단위벡터 노이즈
    encoder_bias: bool = True  # per-env 고정 joint_pos 오프셋 (±PACE_ENCODER_BIAS_MAG)


@configclass
class Go2ImitationTrackingEnvCfg(DirectRLEnvCfg):
    """Go2 Imitation Tracking 환경 설정 — body-frame 속도추종 + RMA/estimator 아키텍처.

    Policy observation dict (실배포 가능 RMA 구조 — root_lin_vel_b는 실측 불가하므로
    estimator가 policy(45)로부터 추정하고, priv_explicit(3)는 학습 시 GT critic/estimator target 용):
        policy(45)        = root_ang_vel_b(3) + projected_gravity_b(3) +
                             lin_vel_cmd(2) + yaw_vel_cmd(1) +
                             joint_pos - default(12) + joint_vel(12) + actions(12)
        priv_explicit(3)  = root_lin_vel_b * priv_explicit_lin_vel_scale
        priv_latent(19)   = armature_scale(1) + joint_friction_scale(1) + base_mass_offset(1) +
                             foot_friction_offset(1) + kp_scale(1) + kd_scale(1) +
                             action_delay_norm(1) + encoder_bias_norm(12)
        history(10, 45)   = policy proprio ring buffer (noised)

    AMP Discriminator 관측 (amp_observation_space = 49, per step):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [R4: heading-relative 6D rotation, MimicKit compute_tar_obs 방식]

    AMP History (num_amp_observations = 10):
        amp_observation_size = 49 × 10 = 490
    """

    # ── 에피소드 ────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # sim: 200 Hz physics, 50 Hz policy (decimation=4)
    sim_dt_hz: int = 200
    policy_dt_hz: int = 50
    decimation: int = sim_dt_hz // policy_dt_hz  # 4

    # ── 공간 ────────────────────────────────────────────────────
    # policy(45) = root_ang_vel_b(3) + projected_gravity_b(3) + lin_vel_cmd(2) + yaw_vel_cmd(1)
    #            + joint_pos_offset(12) + joint_vel(12) + actions(12)
    observation_space: int = 3 + 3 + 2 + 1 + 12 + 12 + 12  # = 45
    action_space: int = 12
    state_space: int = 0

    # ── RMA (dict obs: policy/priv_explicit/priv_latent/history) ─
    num_priv_explicit: int = 3  # root_lin_vel_b * priv_explicit_lin_vel_scale
    num_priv_latent: int = 19  # armature/friction/mass/kp/kd/action_delay/encoder_bias(12)
    history_len: int = 10  # policy proprio ring buffer depth
    priv_explicit_lin_vel_scale: float = 2.0  # root_lin_vel_b 스케일 (parkour_imitation 관례)

    num_amp_observations: int = 10  # disc hist depth (ablation: 2→10, MimicKit 방향)
    amp_observation_space: int = 49  # per-step disc obs (R4: +6 root_rot_tan_norm)
    include_rel_track_obs: bool = False  # 상대적 2D 궤적 포함 여부 토글

    # ── 모션 데이터 ─────────────────────────────────────────────
    motion_file: str = MOTION_FILES_DIR
    reference_body: str = "base"

    # 항상 RSI (Reference State Initialization) 사용
    reset_strategy: str = "random"  # "random" | "random_start"

    # ── 속도추종 command 범위 ────────────────────────────────────
    lin_vel_x_min: float = 0.0  # vx 최소 (m/s)
    lin_vel_x_max: float = 4.0  # vx 최대 (m/s)
    lin_vel_y_min: float = 0.0  # vy 항상 0
    lin_vel_y_max: float = 0.0  # vy 항상 0
    yaw_vel_min: float = -1.5  # yaw rate 최소 (rad/s)
    yaw_vel_max: float = 1.5  # yaw rate 최대 (rad/s)
    tar_change_time_min: float = 4.0  # 목표 명령 변경 최소 주기 (s)
    tar_change_time_max: float = 7.0  # 목표 명령 변경 최대 주기 (s)

    # ── 보상 가중치 ─────────────────────────────────────────────
    # Task reward = lin_vel_reward_w * lin_vel_reward + yaw_vel_reward_w * yaw_vel_reward
    lin_vel_reward_w: float = 0.7  # 선속도 추종 가중치
    yaw_vel_reward_w: float = 0.3  # yaw 속도 추종 가중치
    vel_err_scale: float = 0.5  # lin_vel reward 지수 스케일
    yaw_vel_err_scale: float = 0.5  # yaw_vel reward 지수 스케일

    # ── 조기 종료 ───────────────────────────────────────────────
    early_termination: bool = True
    termination_height: float = 0.15  # base 높이 임계값 (m)
    contact_force_threshold: float = 500.0  # base 접촉 판정 (N)
    roll_termination_deg: float = 70.0
    pitch_termination_deg: float = 70.0

    # ── 액션 ────────────────────────────────────────────────────
    action_scale: float = 0.25

    # ── 시뮬레이션 ──────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / sim_dt_hz,
        render_interval=decimation,  # decimation
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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # ── Sim2Real: PACE 식별 파라미터 반영 + Domain Randomization ──
    # 둘 다 기본 on. 원래 baseline(nominal armature 0.01·마찰 0·DR 없음)을 재현하려면
    # use_pace_params=False, domain_rand=False 로 실행.
    # [A/B 진단 이력] PACE viscous=2.4 과감쇠가 awkward gait root cause로 확정(팀 분석, 26_07_22) →
    # 당시 OFF로 5.1 품질 복원 확증. 이후 viscous를 더 현실적인 ~0.2로 재보정하는 후속 작업은 별도.
    # RMA(ActorCriticRMA)+estimator+DR 아키텍처로 전환하면서 PACE/DR을 다시 기본 ON으로 되돌린다 —
    # priv_latent가 armature/friction/kp/kd/action_delay/encoder_bias 스케일을 명시적으로 인코딩하므로
    # 정책이 PACE 플랜트 편차에 강건해지도록 학습시키는 것이 이번 아키텍처의 목적이다.
    # gait 품질이 재차 저하되더라도 이는 이번 변경의 revert 신호가 아니라 별도 후속 진단 대상이다
    # (예: viscous 재보정, priv_encoder 용량 조정 등).
    use_pace_params: bool = True  # 식별된 armature/viscous/Coulomb 를 로봇 플랜트에 반영
    domain_rand: bool = True  # DR 활성화 (아래 dr 범위 사용)
    dr: DomainRandCfg = DomainRandCfg()
