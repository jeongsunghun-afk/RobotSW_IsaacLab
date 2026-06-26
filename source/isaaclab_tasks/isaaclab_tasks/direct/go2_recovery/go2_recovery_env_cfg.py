# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct._common import DebugViewerCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class EventCfg:
    """Randomization configuration for Go2 recovery."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class Go2RecoveryEnvCfg(DirectRLEnvCfg):
    """Configuration for Go2 fall-recovery environment (M1: skeleton + fall init).

    Observation space breakdown (42 total):
        root_ang_vel_b        [3]   angular velocity of base in body frame
        projected_gravity_b   [3]   gravity vector projected to body frame
        joint_pos_error       [12]  (joint_pos - default_joint_pos)
        joint_vel             [12]  joint velocities
        previous_actions      [12]  actions from previous step
        ─────────────────────────
        TOTAL                 42
    """

    # ── env core ──────────────────────────────────────────────────────────────
    episode_length_s: float = 10.0  # 복구 목표 6~8초 + 여유
    dt: float = 1 / 200
    decimation: int = 4  # policy 50 Hz
    action_scale: float = 0.25
    action_space: int = 12  # 12 DOF position targets
    # observation_space = 3+3+12+12+12 = 42  (cmd 없음 — 복구는 목표=default pose 고정)
    observation_space: int = 42
    state_space: int = 0
    clip_actions: float = (
        100.0  # unused — 실제 clip은 action_clip 필드 사용 (env.py:179). 삭제 보류(외부 참조 가능성 회피)
    )

    # ── fall init parameters (§5 사용자 확정: 임의 자세 전체 커버) ─────────────
    fall_height: float = 0.45  # 공중에서 낙하 시작할 z offset (m)
    fall_standing_ratio: float = 0.1  # default pose (standing) 비율
    fall_sitting_ratio: float = 0.1  # sitting pose 비율 (나머지 0.8 = fallen)
    # fallen 초기화 euler 범위 (rad)
    fall_roll_range: float = 2.356  # ±135 deg (2.356 rad)
    fall_pitch_range: float = 0.785  # ±45 deg
    fall_yaw_range: float = 3.1416  # ±180 deg
    # sitting pose 낮춤 z 오프셋
    sit_height_offset: float = (
        0.13  # Genesis: base_init_pos z=0.34, sit z=0.34-0.20=0.14m → IsaacLab default 0.27-0.14=0.13m
    )

    # ── obs scale (sim-to-real 대응: lin_vel 제거 → 고유감각만) ───────────────
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05

    # ── simulation ────────────────────────────────────────────────────────────
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

    # ── terrain: flat ground ──────────────────────────────────────────────────
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

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # ── events (randomization) ────────────────────────────────────────────────
    events: EventCfg = EventCfg()

    # ── robot ─────────────────────────────────────────────────────────────────
    # Tier-0 안전: velocity_limit 30→18 (60%), damping 0.5→1.0 (2×), stiffness 25 유지
    # 원본 UNITREE_GO2_CFG 공유 자산 직접 mutate 금지 → replace로 새 cfg 생성
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=18.0,  # 30→18 rad/s (60%), Tier-0 안전
                stiffness=25.0,
                damping=1.0,  # 0.5→1.0 (2×), 급격 토크 억제
                friction=0.0,
                armature=0.01,
            ),
        },
    )

    # ── sensors ───────────────────────────────────────────────────────────────
    # contact_sensor: base + foot 접촉 추적 (M1에서 obs 미사용; M2 success/safety 지표용)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # ── DebugViewer cfg (cfg-worker가 키/속도 조정 가능) ──────────────────────
    debug_viewer: DebugViewerCfg = DebugViewerCfg()

    # ── M2 action safety ──────────────────────────────────────────────────────
    # Tier-0: action clip (raw action 범위 제한, pre_physics_step env.py:179에서 적용)
    # 참조(Genesis): clip_actions=100 + action_scale=0.25 → 관절 full range 구동 가능
    # soft_joint_pos_limit 이 최종 clamp 담당하므로 100.0은 사실상 무제한
    action_clip: float = 100.0
    # hip joint(index 0,3,6,9) action scale 0.5× (참조 FR-Net 채택, 균형 권장)
    # True이면 hip 인덱스 action에 0.5를 곱한 후 scale 적용
    hip_action_scale: float = 0.5

    # ── M2 recovery reward ────────────────────────────────────────────────────
    # reward_reset = (roll_w·r_roll + stand_w·r_stand) × reward_reset_scale
    #              + r_roll_progress × roll_progress_weight  (별도 항, step_dt 1회 곱)
    #
    # 정량 근거 (step_dt=0.02):
    #   roll_reward_weight=2.0, reward_reset_scale=2.0 기준
    #   r_roll=0.25(fallen 중간): 2.0×0.25×2.0×0.02 = 0.020/step
    #     → 부드러운 복구 시 smoothness penalty(-0.003~-0.008/step) 상회 → net 양수
    #   r_roll=1.0(upright): 2.0×1.0×2.0×0.02 = 0.080/step
    #     → success_bonus(+5.0×0.02=0.100/step-equivalent)와 균형
    #   roll_progress(Δcos≈0.01/step): 5.0×0.01×0.02 = 0.001/step (즉각 방향 신호)
    #
    # r_roll weight (upright 정렬도): 참조(Genesis)값 0.5
    roll_reward_weight: float = 0.5
    # r_stand weight (height + pose + vel): 참조(Genesis)값 0.5
    stand_reward_weight: float = 0.5
    # potential-based progress shaping: Δcos_dist × weight × step_dt
    # fallen(-1)→upright(+1) 방향에 즉각 양의 신호, 후퇴 시 음수 (shaping 허용)
    # 5.0은 주 r_roll 항의 1/8 수준 — 보조 신호로 dominant 항 무력화 없음
    roll_progress_weight: float = 5.0
    # r_stand가 활성화되는 cos_dist 임계값 (cos(0.2π) ≈ 0.809)
    stand_cos_threshold: float = 0.809
    # Go2 default base height (m) — r_height 계산 기준
    # joint-forced kinematic 측정값 0.31 (baseline, 현재 best)
    # 0.286(PD 평형 실측)으로 낮춘 실험: Cond2 47%→4.4%, Cond1 77%→60% 악화 → 역효과 확인, 0.31 복원
    target_height: float = 0.31
    # r_pose 지수 감쇠 계수 (exp(-k·weighted_pose_err))
    # 참조(Genesis legged_env_recovery.py:1267): exp(-0.6·pose_err)
    # 1.0 → 0.6(참조)보다 완만히 강화. 1.5/3.0(과도, 앉기 유발) 미만.
    #   pose_err=0.25(success 임계): exp(-1.0×0.25)=0.78
    #   pose_err=0.5:                exp(-1.0×0.5) =0.61
    #   pose_err=1.0:                exp(-1.0×1.0) =0.37
    pose_exp_scale: float = 1.0
    # r_vel 지수 감쇠 계수 (exp(-k·vel²))
    vel_exp_scale: float = 0.02
    # r_stand 내부 가중치 (합=1.4, 재정규화 없음)
    # 기준(Genesis): 0.2·r_height + 0.6·r_pose + 0.2·r_vel (합=1.0)
    # 변경: stand_pose_weight 0.6→1.0 — 실효계수 0.5×0.6=0.3→0.5×1.0=0.5
    # 목적: 정착 standing이 default 자세에서 0.19 rad 벗어남 → pose gradient 강화
    # 재정규화 금지: 합=1.0 유지 시 pose 실효 0.357로 목표치(0.5) 미달
    stand_height_weight: float = 0.2
    stand_pose_weight: float = 1.0
    stand_vel_weight: float = 0.2
    # reward_reset 전체 스케일: 참조(Genesis)값 1.0
    reward_reset_scale: float = 1.0

    # ── M2 smoothness / regularization penalty ────────────────────────────────
    # Genesis 참조(train_recovery.py:201-223)값으로 원복.
    # 1차 action 변화율 (a_t - a_{t-1}) — 수식: 우리는 제곱합, 참조는 L2 norm(‖Δa‖); scale만 맞춤
    action_rate_l2_scale: float = -0.01
    # joint_pos_target 1차 차분: sum((target_t - target_{t-1})²)
    # 참조(legged_env_recovery.py:1284-1288, train:217): -0.1
    # action_rate_l2는 raw action 기반, 이 항은 target(=action_scale·action+default) 기반
    action_smoothness_1_scale: float = -0.1
    # 2차 action 변화율 (a_t - 2·a_{t-1} + a_{t-2})²
    action_smoothness_2_scale: float = -0.1
    # joint 가속도 (Δjoint_vel / dt)²
    dof_acc_l2_scale: float = -2.5e-7
    # joint 속도 제곱합 — 참조에 없음, 0.0으로 비활성화 (필드 유지)
    dof_vel_l2_scale: float = 0.0
    # applied torque 변화율 제곱합: sum((torque_t - torque_{t-1})²)
    # 참조(legged_env_recovery.py:1155, train:214): -1e-6
    delta_torques_scale: float = -1e-6
    # applied torque 제곱합
    dof_torques_l2_scale: float = -5e-5
    # joint position soft limit 위반 패널티
    dof_pos_limits_scale: float = -10.0

    # ── M2 success judgment ───────────────────────────────────────────────────
    # success 조건 판정용 임계값
    success_cos_threshold: float = 0.809  # upright cos_dist 기준
    success_pose_eps: float = 0.5  # weighted pose error 상한 (rad)
    success_vel_eps: float = 2.0  # joint vel RMS 상한 (rad/s)
    # 조건 연속 유지 step 수 (20 step = 0.4 s at 50 Hz policy)
    success_hold_steps: int = 20
    # success 달성 시 일회성 bonus reward
    success_reward_scale: float = 5.0
    # success 도달 env를 terminated에 포함 (early termination, 학습 효율↑)
    # Lever A: False로 전환 — crouch-to-timeout의 continuation value 우위 제거.
    # terminate=True 시 success bonus(+0.10 실효) < crouch r_stand 포기분(~1.8) → 성공 anti-incentive.
    # rsl_rl PPO는 time_out만 value-bootstrap하고 terminated는 bootstrap 없음(ppo.py:181-208) →
    # terminate-off 없이는 성공이 crouch보다 advantage 음수. (RECOVERY_POSE_PLAN.md §0)
    terminate_on_success: bool = False
    # 연속 in-region 보상 scale (Lever A): 순간 3조건(upright∧near-default∧low-vel) 동시충족 시 매 step 지급.
    # 실효값 = success_region_reward_scale × step_dt(0.02) ≈ 0.06/step.
    # 목적: crouch r_stand(≈0.0196/step)를 명확히 상회 → default-pose가 crouch보다 reward-최적점으로 전환.
    # 짧은 런 캘리브레이션 대상 (crouch 잔존 시 값 상향 조정).
    success_region_reward_scale: float = 3.0

    # ── Settle phase (학습 시 공중낙하→자연 안착 후 복구 시작) ──────────────────
    # 0이면 비활성(기존 동작). 양수이면 env별 randint(0, settle_max_steps+1) step
    # 동안 action=default_joint_pos로 override하여 자연 안착 후 복구 학습 시작.
    # settle_max_steps=100 (2s at 50Hz policy) — episode_length_s=10s와 충분한 여유
    settle_max_steps: int = 100

    # ── Fall lerp exponent (cubic lerp 지수) ─────────────────────────────────
    # rand^exponent: exponent=3 → 평균 α=0.25 (default 근방 밀도 ↑)
    #                exponent=1.5 → 평균 α=0.40, P(α<0.1) 46%→22% (더 펼쳐진 자세)
    fall_lerp_exponent: float = 1.5
