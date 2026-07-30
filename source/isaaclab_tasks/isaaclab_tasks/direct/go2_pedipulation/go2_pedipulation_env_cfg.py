# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Pedipulation (발 조작) 환경 설정.

로봇이 제자리에 선 채로 지정된 발을 base frame 목표 위치로 옮기고 유지하는 task.
명령 표현은 S1(도달·유지)부터 S4(접촉 품질)까지를 한 env에서 다룰 수 있도록 설계했고,
단계는 명령 생성기 분포로만 구분한다.

설계 근거는 ``.omc/research/pedipulation/PLAN.md`` 참조.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# ---------------------------------------------------------------------------
# PACE 시스템 식별 결과 (go2_imitation_tracking 과 동일 값)
# ---------------------------------------------------------------------------
PACE_ARMATURE: dict[str, float] = {"hip": 0.173, "thigh": 0.168, "calf": 0.200}  # [kg·m²]
PACE_VISCOUS: dict[str, float] = {"hip": 2.41, "thigh": 2.33, "calf": 2.49}  # [N·m·s/rad]
PACE_COULOMB: dict[str, float] = {"hip": 0.023, "thigh": 0.010, "calf": 0.020}  # [N·m]
PACE_ENCODER_BIAS_MAG: float = 0.05  # [rad]

# 다리 이름 — IsaacLab GO2 articulation 의 발 body 순서와 동일하게 유지할 것.
LEG_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")


@configclass
class DomainRandCfg:
    """Domain Randomization — go2_imitation_tracking 설정을 그대로 승계.

    접촉이 핵심인 task 이므로 마찰·질량·게인·지연 랜덤화를 유지한다.
    """

    randomize_mass: bool = True
    added_base_mass_range: tuple[float, float] = (-1.0, 2.0)  # base payload [kg]
    randomize_material: bool = True
    foot_friction_range: tuple[float, float] = (0.4, 1.4)
    randomize_armature: bool = True
    armature_scale_range: tuple[float, float] = (0.8, 1.2)
    randomize_joint_friction: bool = True
    joint_friction_scale_range: tuple[float, float] = (0.8, 1.2)

    randomize_gains: bool = True
    kp_scale_range: tuple[float, float] = (0.9, 1.1)
    kd_scale_range: tuple[float, float] = (0.9, 1.1)
    randomize_action_delay: bool = True
    max_action_delay_steps: int = 1

    # 외란 — P1 의 disturbance curriculum (base push + 발 외력)
    push_robot: bool = True
    push_interval_s: float = 3.0  # [s] P1: 3초 주기
    max_push_vel_xy: float = 0.6  # [m/s] P1: ±0.6
    max_push_ang_vel: float = 0.6  # [rad/s] P1: ±0.6

    push_foot_force: bool = True
    max_foot_force: float = 12.0  # [N] P1: 조작 발에 최대 12 N, 에피소드 내 상수

    obs_noise: bool = True
    joint_pos_noise: float = 0.01  # [rad]
    joint_vel_noise: float = 0.5  # [rad/s]
    gravity_noise: float = 0.05
    foot_pos_noise: float = 0.005  # [m] FK 추정 오차
    encoder_bias: bool = True


@configclass
class CommandCfg:
    """발 목표 명령 생성기 설정.

    단계 전환은 이 블록의 값만 바꿔서 수행한다 (env·obs·action 차원 불변).
    """

    # ── 조작 다리 선택 ────────────────────────────────────────
    num_manip_legs: int = 1  # 동시 조작 다리 개수. S3 에서 2 로 확장
    manip_leg_candidates: tuple[int, ...] = (0, 1, 2, 3)  # FL, FR, RL, RR

    # ── 목표 위치 샘플링 (nominal 발 위치 대비 base frame 오프셋) ──
    # P1 의 command-space curriculum: 작은 박스에서 시작해 성공하면 확장.
    # close-range 유지(D3) 이므로 P1 의 far-range(2.2×3.0×1.3 m)까지 가지 않는다.
    box_init: tuple[float, float, float] = (0.06, 0.05, 0.08)  # (±x, ±y, +z) [m]
    box_max: tuple[float, float, float] = (0.20, 0.14, 0.26)  # [m]
    box_step: tuple[float, float, float] = (0.02, 0.015, 0.02)  # 승급 시 확장량 [m]
    curriculum_err_threshold: float = 0.06  # [m] P1: 평균 추종 오차 < 0.06 m 이면 승급
    curriculum_min_episodes: int = 200  # 승급 판정에 필요한 최소 종료 에피소드 수

    # ── 궤적 모드 (S2) ────────────────────────────────────────
    # "static": 목표 고정 (S1) · "circle": base frame x-z 평면 원 궤적 (S2)
    #
    # ⚠ PLAN §5 의 S2-G1 은 반경 0.20 m 를 요구하지만 **Go2 다리로는 기구학적으로 불가능**하다.
    #   FL 발의 도달 영역을 관절 한계로 실측한 결과(IMPL_LOG §4h), 오프셋 범위는 x [-0.373,
    #   +0.405] / z [-0.090, +0.687] m 로 넓지만 그 영역 **안에 들어가는 최대 원은 반경
    #   0.140 m** 다 (관절 한계가 만드는 영역 형태 때문이며 z 범위 부족이 아니다).
    #   S1 이 검증한 명령 박스 안에서는 0.110 m 다. 0.20 m 는 팔 매니퓰레이터 벤치마크에서 온
    #   값으로, 사족 다리에 전이되지 않는다.
    trajectory_mode: str = "static"
    # 평가용 고정값 (randomize=False 일 때 사용)
    circle_radius: float = 0.10  # [m]
    circle_omega: float = 0.5  # [rad/s] 각속도. 접선속도 = omega * radius
    # 학습용 랜덤화 — 속도 sweep 평가를 하려면 정책이 여러 속도를 겪어야 한다.
    circle_randomize: bool = True
    circle_radius_range: tuple[float, float] = (0.05, 0.10)  # [m]
    # ⚠ sweep 축은 접선속도가 아니라 **각속도**다. 같은 접선속도라도 반경이 작으면 곡률이
    #   커져 더 어렵다. PLAN 의 임계값(0.035/0.045/0.06 m)은 반경 0.20 m 기준이므로,
    #   omega 를 맞춰야 그 숫자를 인용할 근거가 생긴다:
    #     PLAN  r=0.20, v=0.05/0.10/0.20 m/s  →  omega = 0.25 / 0.50 / 1.00 rad/s
    #     여기  r=0.10, v=0.025/0.05/0.10 m/s →  omega = 0.25 / 0.50 / 1.00 rad/s (동일)
    circle_omega_range: tuple[float, float] = (0.15, 1.2)  # [rad/s]
    # 원이 명령 박스를 벗어나지 않도록 중심 샘플링에 남길 여유
    circle_center_margin: float = 0.01  # [m]

    # ── 재샘플 주기 ───────────────────────────────────────────
    resample_time_min: float = 3.0  # [s]
    resample_time_max: float = 5.0  # [s]


@configclass
class Go2PedipulationEnvCfg(DirectRLEnvCfg):
    """Go2 Pedipulation 환경 설정.

    Observation dict (asymmetric actor-critic — actor 는 실기에서 얻을 수 있는 것만):
        policy(83)   = projected_gravity_b(3) + (joint_pos - default)(12) + joint_vel(12) +
                        prev_actions(28) + foot_pos_b(12) + leg_role(4) + foot_pos_err_b(12)
        history(830) = policy 링버퍼 10 step (flatten)
        priv(30)     = root_lin_vel_b(3) + root_ang_vel_b(3) + base_height(1) +
                        foot_contact_force(4) + dr_params(19)

    ``base_lin_vel`` / ``base_ang_vel`` 은 critic 전용이다 — actor 가 의존하면 실기 배포가
    막히거나(선속도) 불필요한 IMU 의존이 생긴다(각속도). estimator 는 쓰지 않는다.

    Action(28) = a_loc(12) + a_man(12) + a_stiffness(4)
        a_loc : 지지 다리용, 기본 자세 대비 절대 offset
        a_man : 조작 다리용, 직전 목표 대비 증분
        a_stiffness : 강성 채널. 현재 비활성이며 차원 유지를 위해 자리만 확보한다.
    """

    # ── 에피소드 ────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # sim: 200 Hz physics, 50 Hz policy
    sim_dt_hz: int = 200
    policy_dt_hz: int = 50
    decimation: int = sim_dt_hz // policy_dt_hz  # 4

    # ── 공간 ────────────────────────────────────────────────────
    num_legs: int = 4
    history_len: int = 10

    # policy = 3 + 12 + 12 + 28 + 12 + 4 + 12
    observation_space: int = 3 + 12 + 12 + 28 + 12 + 4 + 12  # = 83
    action_space: int = 12 + 12 + 4  # = 28
    state_space: int = 0
    num_priv: int = 3 + 3 + 1 + 4 + 19  # = 30

    # ── 액션 ────────────────────────────────────────────────────
    action_scale: float = 0.25
    use_stiffness_action: bool = False  # a_stiffness(4) 활성화 여부. 자리는 항상 유지
    stiffness_range: tuple[float, float] = (20.0, 60.0)  # [N·m/rad] 활성화 시
    # 조작 다리 증분 목표의 스텝당 상한 — 적분형 action 의 windup 방지
    manip_delta_clip: float = 0.15  # [rad/step]

    # ── 관절 속도 관측 필터 ─────────────────────────────────────
    # 1.0 = 필터 없음(원시값). EMA: v_filt ← (1-α)·v_filt + α·v_obs
    #
    # 유지 중 실제 관절 속도는 RMS 0.013 rad/s 인데 주입 노이즈는 σ=0.5 rad/s 다 — SNR 이 2.6%
    # 로, 이 채널은 사실상 순수 노이즈다. 그런데도 정책이 여기에 0 이 아닌 이득을 학습해
    # **떨림 분산의 78%** 를 이 채널이 만든다(측정: IMPL_LOG §4j).
    #
    # ⚠ 노이즈 크기(0.5)를 낮추는 것이 아니라 **필터를 넣는다.** 0.5 σ 는 IsaacLab 표준 velocity
    #   env(±1.5 균등, σ≈0.87)보다 이미 작아서, 줄이면 벤치마크를 쉽게 만드는 것이 된다. 반면
    #   실기의 관절 속도는 엔코더 차분값이라 어차피 필터를 거쳐 쓴다.
    # ⚠ **실기 배포 시 같은 필터를 반드시 같은 α 로 적용해야 한다.** 여기만 켜면 sim2real gap 이다.
    # ⚠ 기본값은 1.0(끔)이다. 이 값을 바꾸면 관측 의미가 바뀌므로 **학습과 평가가 같은 α 여야
    #   한다** — 기존 체크포인트를 필터 켠 채로 평가하면 분포 밖이다. 켜고 학습한 정책은
    #   평가 시에도 `env.jvel_filter_alpha` 를 같은 값으로 넘길 것.
    jvel_filter_alpha: float = 1.0

    # 안전: 발 끝 속도. ISO/TS 15066 과도 접촉 한계 역산에서 유도한 값
    # (원위 다리 기준 1.76 m/s, 전체 로봇 기준 1.33 m/s, 통증 회피 권장 0.5 m/s).
    # ⚠ 현재는 **페널티만** 건다. action 이 관절 위치 목표라 발 끝 속도를 직접 clip 하려면
    #    Jacobian 역산이 필요하므로, hard clip 은 사람 접촉을 다루는 S4/S5 에서 도입한다.
    foot_speed_penalty_threshold: float = 0.5  # [m/s] 이 위로 2차 페널티
    max_foot_speed: float = 1.3  # [m/s] 이 위로 barrier (급격히 커지는 페널티)
    w_foot_speed_barrier: float = -1.0

    # ── 보상 가중치 ─────────────────────────────────────────────
    # 추종 (P1/P3 합의 커널: w·exp(-||e||/sigma), sigma=0.8)
    w_track: float = 1.0
    track_sigma: float = 0.8
    # hold 보너스 (P4) — delta_ee 이내에 hold_min_steps 이상 머물면 가산
    w_hold: float = 0.5
    hold_delta_ee: float = 0.05  # [m]
    hold_min_steps: int = 10  # 50 Hz 기준 0.2 s

    # 지지 안정성 (P5) — CoF 기준 중력모멘트 + 지지 다각형 면적
    w_grav_moment: float = 0.5
    grav_moment_sigma: float = 10.0  # [N·m]
    w_support_area: float = 0.2
    support_area_sigma: float = 0.05  # [m²]

    # 자세 유지
    w_flat_orientation: float = -1.0
    w_base_height: float = -1.0
    base_height_target: float = 0.32  # [m]

    # 지지 다리 규제
    # ⚠ 너무 크면 counterbalance(앞발을 뻗을 때 몸통이 자세를 바꾸는 것)를 억제한다.
    w_stance_default: float = -0.1  # 지지 다리 기본 자세 이탈
    # base 표류는 직접 페널티하지 않는다(counterbalance 를 억제하므로). 대신 접지 발이
    # 미끄러지지 않게 하면 base 병진이 지지 다리의 kinematic envelope 안으로 묶인다.
    w_feet_slip: float = -3.0  # 접지 발 미끄러짐

    # action 규제 (절대 크기가 아니라 변화율/가속도/토크 — PLAN §3.4)
    #
    # ⚠ 스케일 주의: 28-dim 에 대한 제곱합이라 항당 기여가 쉽게 커진다. 초기 std=0.5 에서
    #   E[Σ(Δa)²] ≈ 14, E[Σ(a-2a'+a'')²] ≈ 42 이므로 **학습 초기에는** 추종 보상을 압도할 수
    #   있다. 첫 스모크에서 -0.05/-0.02 가 실제로 task 를 완전히 눌렀다.
    #
    # 아래 값은 그보다 낮지만, 처음 수렴시킨 -0.005/-0.002 보다는 4배 높다. 그 값에서는 정책이
    # 관측 노이즈를 액션으로 그대로 흘려보내 유지 중에도 관절 목표가 2.56°/step 흔들렸다.
    # 4배(현재)와 10배를 fine-tune 으로 비교한 결과가 이 값의 근거다:
    #   4배  — 떨림 2.56°→1.30°/step, hold 오차 12.4→10.2 mm, push 강건성 불변
    #   10배 — 떨림 0.91°/step 로 더 낮지만 **150 N push 성공률이 52%→31% 로 붕괴**
    # 즉 규제를 더 올리면 외란 회복 반응이 느려진다. 측정 근거는 IMPL_LOG §4g.
    w_action_rate: float = -0.02
    w_action_smooth: float = -0.008
    # 실제로 액추에이터에 보내는 관절 목표의 스텝당 변화량. **떠는 양 자체**를 벌하는 항이다.
    #
    # ⚠ w_action_rate 만으로는 조작 다리에 1차 페널티가 걸리지 않는다. a_man 이 이미 증분이라
    #   Δa_man 은 가속도이고, 목표가 일정한 속도로 표류하는 것에는 비용이 0 이다. 그래서 떨림이
    #   조작 다리 4.83°/step 대 지지 다리 2.16°/step 으로 갈렸다(측정: IMPL_LOG §4j).
    #   이 항은 두 경로에 동일하게 걸린다.
    w_joint_target_rate: float = -0.4
    w_joint_acc: float = -1e-6
    w_joint_vel: float = -5e-4
    w_torque: float = -2e-5

    # 접촉 / 안전
    w_contact_force: float = -2e-4  # 시뮬 측정 접촉력 페널티 (reward 전용 privileged)
    w_foot_speed: float = -0.05  # 임계 초과 발 끝 속도
    # 조작 다리를 낮은 목표로 보낼 때 calf 가 지면에 닿는 것을 과하게 벌하지 않도록 조정.
    w_collision: float = -0.5
    w_termination: float = -20.0
    w_joint_limit: float = -1.0

    # ── 조기 종료 ───────────────────────────────────────────────
    early_termination: bool = True
    termination_height: float = 0.15  # [m]
    contact_force_threshold: float = 300.0  # [N] base 접촉 판정
    roll_termination_deg: float = 55.0
    pitch_termination_deg: float = 55.0

    # ── 시뮬레이션 ──────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / sim_dt_hz,
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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # ── Sim2Real ────────────────────────────────────────────────
    use_pace_params: bool = True
    domain_rand: bool = True
    dr: DomainRandCfg = DomainRandCfg()
    command: CommandCfg = CommandCfg()
