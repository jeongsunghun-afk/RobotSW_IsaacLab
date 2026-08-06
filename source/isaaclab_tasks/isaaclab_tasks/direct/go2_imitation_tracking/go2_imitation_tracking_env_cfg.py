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
from isaaclab.actuators import DCMotorCfg
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
# PACE 시스템 식별 결과 (mean_199, 26_08_03 · 시간축 보정 데이터로 재적합)
# ---------------------------------------------------------------------------
# 실기 GO2 액추에이터 플랜트. 관절 타입별 값(다리별 편차가 작아 타입 대표값으로 반영).
# ⚠ kp=25/kd=0.5 는 유지한다 — 식별이 이 게인 조합에서 이뤄졌다.
#
# ★ 2026-08-03 값이 크게 바뀌었다. 이전 값(아래 _LEGACY)은 **잘못 식별된 값**이었다.
#   원인: 실기 캡처의 명령/상태 시간축이 104 ms(=52 step @500 Hz) 어긋나 있었고
#   (원인은 `/lowstate` 구독 큐 깊이 50 의 상시 포화), 적합기의 delay 파라미터 상한이
#   10 step 이라 그 위상지연을 표현할 수 없어 **armature/viscous 가 대신 흡수**했다.
#   그래서 viscous 가 실제의 ~100배, armature 가 ~10배, coulomb 은 거꾸로 ~30배 과소로 나왔다.
#   hold-out RMSE(0.017)는 세 캡처가 같은 결함을 공유해 이를 **걸러내지 못했다**.
#   검증: 식별된 마찰이 요구하는 소산 / 모터가 실제로 낸 정미 일 = 구 24.6배 → 신 1.1배.
#        `tau_est` 독립 회귀와도 coulomb 0.198/0.151/0.670 vs 0.213/0.156/0.655 로 일치.
#   상세: reports/_comparisons/pace_go2_sysid_excitation_audit/
PACE_ARMATURE: dict[str, float] = {"hip": 0.00101, "thigh": 0.00011, "calf": 0.01610}  # [kg·m²]
PACE_VISCOUS: dict[str, float] = {"hip": 0.00187, "thigh": 0.01617, "calf": 0.00152}  # [N·m·s/rad]
PACE_COULOMB: dict[str, float] = {"hip": 0.19832, "thigh": 0.15084, "calf": 0.67005}  # [N·m]

# 구 값 — **쓰지 말 것.** 위에 적은 이유로 물리적으로 불가능한 값이다.
# 2026-08-03 이전 run 을 재현/대조할 때만 참조한다(그 run 들은 이 플랜트에서 학습됐다).
PACE_ARMATURE_LEGACY: dict[str, float] = {"hip": 0.173, "thigh": 0.168, "calf": 0.200}
PACE_VISCOUS_LEGACY: dict[str, float] = {"hip": 2.41, "thigh": 2.33, "calf": 2.49}
PACE_COULOMB_LEGACY: dict[str, float] = {"hip": 0.023, "thigh": 0.010, "calf": 0.020}
# 2026-08-04 신규 캡처(`pitch_cancel` 패턴, 0.1→5 Hz, 20 s 완주) 적합값.
# **학습 플랜트로 채택하지 않았다** — coulomb 과 calf armature 는 독립 검사를 통과했지만
# viscous 가 독립 회귀 대비 3.5~10.8배, 에너지 수지 대비 1.6~3.0배로 기각됐다(원인 미상,
# 리그 오염과는 무상관 확인). r2s sim 에서 **눈으로 비교하기 위한 프리셋**으로만 쓴다.
# 파라미터는 결합 식별이므로 이 세트의 일부만 떼어 위 `PACE_*` 와 섞지 말 것.
# 상세: reports/_comparisons/pace_go2_sysid_excitation_audit/ (§신규 캡처 적합 물리 검사)
PACE_ARMATURE_SET3: dict[str, float] = {"hip": 0.00758, "thigh": 0.00531, "calf": 0.02106}  # [kg·m²]
PACE_VISCOUS_SET3: dict[str, float] = {"hip": 0.1806, "thigh": 0.1436, "calf": 0.1265}  # [N·m·s/rad]
PACE_COULOMB_SET3: dict[str, float] = {"hip": 0.1446, "thigh": 0.0997, "calf": 0.5766}  # [N·m]

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
    # DEAD (2026-07-30): 각속도가 policy obs에서 빠져 주입할 자리가 없다. 되살릴 때는
    # `_apply_obs_dr`의 인덱스를 반드시 함께 맞출 것 — 인덱스만 밀린 채 남겨두면 σ=0.2가
    # σ=0.05인 projected_gravity_b에 주입되어 4배 증폭된다.
    ang_vel_noise: float = 0.2  # [rad/s]
    gravity_noise: float = 0.05  # projected_gravity 단위벡터 노이즈
    encoder_bias: bool = True  # per-env 고정 joint_pos 오프셋 (±PACE_ENCODER_BIAS_MAG)


@configclass
class Go2ImitationTrackingEnvCfg(DirectRLEnvCfg):
    """Go2 Imitation Tracking 환경 설정 — body-frame 속도추종 + RMA/estimator 아키텍처.

    Policy observation dict (실배포 가능 RMA 구조 — estimator가 policy(42)로부터 root 선속도·
    각속도를 추정하고, priv_explicit(6)는 학습 시 GT critic/estimator target 용):
        policy(42)        = projected_gravity_b(3) +
                             lin_vel_cmd(2) + yaw_vel_cmd(1) +
                             joint_pos - default(12) + joint_vel(12) + actions(12)
        priv_explicit(6)  = root_lin_vel_b * priv_explicit_lin_vel_scale +
                             root_ang_vel_b * priv_explicit_ang_vel_scale
        priv_latent(19)   = armature_scale(1) + joint_friction_scale(1) + base_mass_offset(1) +
                             foot_friction_offset(1) + kp_scale(1) + kd_scale(1) +
                             action_delay_norm(1) + encoder_bias_norm(12)
        history(10, 42)   = policy proprio ring buffer (noised)

    **priv_explicit로 분리한 신호는 policy obs에서 제외한다** (2026-07-30 변경). estimator가
    추정하는 대상을 actor에게 직접 보여주면 추정 구조가 무의미하기 때문이다. root_ang_vel_b는
    실기 IMU로 측정 가능하지만 이 일관성을 위해 뺐다. 그 결과 estimator의 ang 블록 과제가
    "noisy→clean denoising"에서 **"미관측→GT 상태추정"** 으로 바뀌었다.

    AMP Discriminator 관측 (amp_observation_space = 49, per step):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [R4: heading-relative 6D rotation, MimicKit compute_tar_obs 방식]

    AMP History (num_amp_observations = 10):
        amp_observation_size = 49 × 10 = 490
    """

    # ── 에피소드 ────────────────────────────────────────────────
    #: 에피소드 길이 [s]. 2026-07-31 에 10.0 → 20.0 (정지→재출발 전이가 에피소드 안에서
    #: 완결되게 — 아래 "정지/재출발 학습" 절 참고).
    #: ⚠ ``Episode_Reward/*`` 는 ``episodic_avg / max_episode_length_s`` 로 정규화되므로
    #: per-step 환산 시 상수 10.0 이 아니라 **이 값**을 써야 한다
    #: (``per_step = Episode_Reward × episode_length_s / mean_episode_length``).
    #: 이전 런들과 `lin_ps` 를 비교할 때 이 상수를 런마다 맞춰야 한다.
    episode_length_s: float = 20.0

    # sim: 200 Hz physics, 50 Hz policy (decimation=4)
    sim_dt_hz: int = 200
    policy_dt_hz: int = 50
    decimation: int = sim_dt_hz // policy_dt_hz  # 4

    # ── 공간 ────────────────────────────────────────────────────
    # policy(42) = projected_gravity_b(3) + lin_vel_cmd(2) + yaw_vel_cmd(1)
    #            + joint_pos_offset(12) + joint_vel(12) + actions(12)
    # root_lin_vel_b·root_ang_vel_b는 priv_explicit로 분리되어 여기 없다.
    observation_space: int = 3 + 2 + 1 + 12 + 12 + 12  # = 42
    action_space: int = 12
    hip_scale_reduction: bool = True  # hip(abduction) 관절 액션을 0.5배로 축소
    state_space: int = 0

    # ── RMA (dict obs: policy/priv_explicit/priv_latent/history) ─
    num_priv_explicit: int = 6  # root_lin_vel_b(3)*lin_scale + root_ang_vel_b(3)*ang_scale
    num_priv_latent: int = 19  # armature/friction/mass/kp/kd/action_delay/encoder_bias(12)
    history_len: int = 10  # policy proprio ring buffer depth
    priv_explicit_lin_vel_scale: float = 2.0  # root_lin_vel_b 스케일 (parkour_imitation 관례)
    # root_ang_vel_b 스케일. actor/critic 입력은 어차피 normalizer를 통과하므로 이 값의 실제
    # 역할은 estimator MSE에서 lin 블록 대비 ang 블록의 **상대 gradient 가중치**다.
    #
    # s=2.0 런(2026-07-27 `angvel_priv_50k`, 16k iter) 결과: 각속도 추종 이득 0(8구간 모두
    # est_pace_dr과 동일), 선속도 추종 손해 없음, `Loss/estimator_lin`만 ~17% 높은 채 안정,
    # `Loss/estimator_ang`는 0.098에서 평탄(= 노이즈 분산 0.04 대비 39% 감소에서 정지).
    # 0.25는 **각속도가 policy obs에 있던 시절**의 8구간 비교로 고른 값이다(s=2.0 대비:
    # est_lin 손해 1.14배→1.00배, denoise 37%→29~32%, 정책 성능은 동일). 각속도를 obs에서
    # 빼면 추정 문제가 더 어려워지므로 최적값이 달라질 수 있다 — **상속된 값이며 재검증 안 됨.**
    # 비교 시 `est_ang` 절대값은 s²에 비례하므로 반드시 `est_ang/s²`로 정규화할 것.
    # 단, obs에서 각속도를 뺀 뒤로는 pass-through 경로가 없어져 `ang_vel_noise²`(=0.04)
    # 기준선은 **무효**다. 대신 target 자체의 분산(=평균 예측 시 MSE)과 비교할 것.
    priv_explicit_ang_vel_scale: float = 0.25

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
    tar_change_time_min: float = 2.0  # 목표 명령 변경 최소 주기 (s)
    tar_change_time_max: float = 7.0  # 목표 명령 변경 최대 주기 (s)

    # ── 정지/재출발 학습 (2026-07-31) ────────────────────────────
    # 배경: 이 정책은 **한 번 서면 재출발하지 못했다**(64 env 중 2~4개만 성공, 행동 경로
    # 2×2 전부 실패). 원인은 능력이 아니라 학습 신호였다 —
    #   ① `_reset_strategy_rsi` 가 항상 "이미 걷고 있는" 참조 프레임에서 시작해 **정지가
    #      초기 조건이 된 적이 없다**,
    #   ② cmd 0 에서 정지는 task reward 1.0(최대)이라 들어가는 것만 강화되고,
    #   ③ 명령 U[0,4] 를 4~7s 마다 재샘플하는 10s 에피소드에서 "정지 후 재출발"을 요구받는
    #      경우가 4.6% 뿐이며 그마저 리셋이 대신 해결해 준다.
    # 아래 세 값이 ①~③ 에 각각 대응한다. 상세: reports/rsl_rl/go2_imitation_tracking/
    # 2026-07-30_10-35-28_obs42_novideo/README.md "읽히는 것 4 → 원인 규명".
    rel_standing_envs: float = 0.1
    """재샘플 시 명령을 **정확히 0** 으로 강제할 env 비율 (②).

    near-zero tail 이 아니라 exact zero 라 정지가 뚜렷한 모드가 된다. IsaacLab 표준
    :class:`~isaaclab.envs.mdp.UniformVelocityCommandCfg` 와 같은 장치다.

    ⚠ 이 저장소의 ``hind_leg`` 에서 **0.2 는 보행을 전역 붕괴**시킨 이력이 있다
    (mean_reward 43→3, gait_swing −97.8%, from-scratch apples-to-apples). 0.2 이상으로
    올리지 말 것.
    """

    rel_rest_init: float = 0.1
    """리셋 시 참조 모션 대신 **기립 정지 상태**로 초기화할 env 비율 (①).

    default joint pos + 수평 자세 + 속도 0. 이 env 는 "항상 RSI" 라 정지가 초기 조건이 된
    적이 없었고, 그래서 정지→이동 전이가 에피소드 **시작부터** 요구된 적이 없다.
    명령은 강제하지 않는다 — 랜덤 명령을 그대로 받아야 실제로 출발을 요구받는다.
    """

    # ── 정지 시 style reward 대체 (④) ────────────────────────────────────────
    # 측정으로 확인된 문제: cmd 0 에서 **style(AMP) reward 가 오히려 가장 높다**(0.904).
    # discriminator 는 웅크린 정지(base_h 0.217)를 빠른 보행보다 더 참조답다고 본다
    # (bce 기준 σ 0.595 vs 0.518). task(0.988)와 style(0.904) 이 둘 다 정지를 선호해
    # **정지가 전역 최고 total(0.946)** 이 된다 — 흡수 상태의 정체.
    # 참조 모션에 "서 있다가 출발" 구간이 아예 없으니 discriminator 를 고칠 방법이 없다.
    # 그래서 정지 명령 구간에서만 style 을 끄고 **default pose 유지 보상으로 대체**한다.
    # 이러면 튜닝 불가능한 신호(disc 출력)가 튜닝 가능한 신호로 바뀐다.
    # 근거·원자료: reports/rsl_rl/go2_imitation_tracking/2026-07-31_12-27-01_reststand_ft/
    # README.md "★★ 보상 분해 측정".
    standing_style_substitute: bool = True
    """정지 명령(:attr:`rel_standing_envs` 로 강제된 env)에서 style reward 를 pose 보상으로 대체.

    False 면 이전 거동과 **완전히 동일**하다(러너가 ``style_weight`` 를 1.0 으로 본다).
    """

    standing_pose_reward_w: float = 0.85
    """정지 시 pose 보상의 포화값. style 자리(``1 − task_reward_lerp`` 예산)에 그대로 들어간다.

    ⚠ **1.0 으로 두면 안 된다.** 정지 total 이 ``0.5·0.988 + 0.5·1.0 = 0.994`` 가 되어 이동
    최고(cmd 1.0 에서 0.930)를 크게 넘고 흡수 상태가 **더 깊어진다**. 0.85 면 정지 total 이
    0.919 로 이동 최고보다 근소하게 낮아, 정지가 처음으로 보행보다 불리해진다.
    상한은 ``(0.930 − 0.5·0.988) / 0.5 = 0.871``.
    """

    standing_pose_err_scale: float = 5.0
    """pose 보상 지수 스케일. ``r_pose = w · exp(−scale · mean((q − q_default)²))``.

    관절각 오차만 쓴다 — 기체 높이는 별도 목표값을 두지 않아도 default 관절각이 물리적으로
    결정한다(하드코딩한 높이가 틀리면 오히려 잘못된 자세로 끌어당긴다). 관측된 실패가
    "관절이 굽어 base_h 0.217 로 주저앉음" 이므로 관절 오차가 곧 그 신호다.
    """

    # ③ 은 위 "에피소드" 절의 :attr:`episode_length_s` 10.0 → 20.0 으로 반영했다.
    # 에피소드당 명령 재샘플이 ~1.8 → ~3.6 회가 되어, 정지한 뒤에도 리셋 전에 높은 명령을
    # 다시 받는 상황이 실제로 발생한다. `tar_change_time_*` 는 건드리지 않았다 — 재샘플 주기를
    # 줄이면 각 명령의 정착 시간(정지까지 실측 ~2.5s)이 부족해져 추종 성능을 해칠 수 있다.

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

    # ── calf effort_limit 을 자산 값에 맞춰 복원 (2026-08-05) ────────────────────────
    # `UNITREE_GO2_CFG` 는 hip·thigh·calf 를 한 그룹으로 묶어 `effort_limit=23.5` 단일값을 준다.
    # 그런데 이 task 가 로드하는 `Go2_noninstanceable/go2.usd` 의 drive `maxForce` 는
    # hip/thigh 23.7, **calf 45.43** 이고, MimicKit 의 go2.usd 는 calf 35.5 다. 실측에서도
    # `|τ|max = 23.50` 이 thigh·calf 여러 관절에 정확히 찍혀 토크가 병목임이 확인됐다
    # (`reports/rsl_rl/go2_imitation_tracking/_comparisons/mimickit_vs_60_actuator_limit/`).
    #
    # `saturation_effort` 는 dict 를 못 받으므로(`actuator_pd_cfg.py:50` 이 float) 그룹을 둘로
    # 나눈다. 한 그룹에 `saturation_effort=35.5` 를 주고 `effort_limit` 만 관절별로 주면
    # `τ_max = clip(35.5·(1 − q̇/30), −∞, effort_limit)` 이 되어 hip·thigh 의 토크-속도 곡선까지
    # 완만해진다(q̇=8 에서 17.2 → 23.5, +37%). 그건 calf 만 바꾸는 변경이 아니다.
    #
    # ⚠ 그룹을 나눴으므로 env 의 `_acts` 처리(게인 DR)가 두 그룹을 모두 돌아야 한다.
    #
    # `velocity_limit` 은 **30.0 을 쓴다(URDF 의 calf 15.70 이 아니다).**
    #
    # Unitree 공식 URDF 의 `<limit>` 은 hip/thigh `effort 23.7 / velocity 30.1`,
    # calf `effort 45.43 / velocity 15.70` 이다. 2026-08-05 에 calf 를 15.70 으로 넣어 학습했는데
    # **effort 를 51% 올린 이득이 거의 그대로 상쇄됐다.** 참조 모션이 요구하는 calf 속도
    # `q̇ ≈ 7.9 rad/s` 에서의 실효 토크 `τ_max = sat·(1 − q̇/v_lim)` 를 비교하면:
    #
    #   구세대 (23.5 / 30.0) → 17.3
    #   (35.5 / 15.70)       → 17.6     ← +2% 뿐
    #   (35.5 / 30.0)        → 26.1     ← +51%, effort 증가분이 그대로 살아난다
    #
    # 두 값은 의미가 다르다. URDF `<limit velocity>` 는 **허용 최대 관절 속도**(정격/안전 한계)이고
    # `DCMotorCfg.velocity_limit` 은 토크-속도 곡선의 **x 절편, 즉 토크가 0 이 되는 무부하 속도**다.
    # 같은 값을 넣으면 "최대 속도에 닿는 순간 토크 0" 이 되어 그 속도에 도달할 수 없다.
    #
    # ★ 속도 상한 자체는 이 cfg 와 무관하게 **USD 자산에 이미 박혀 있다** —
    #   `Go2_noninstanceable/go2.usd` 의 `physxJoint:maxJointVelocity` 가
    #   hip/thigh 1724.60 deg/s(=30.10 rad/s), calf 899.54 deg/s(=**15.70 rad/s**)다.
    #   그래서 실행 로그의 `Simulation Joint Information` 표는 **어느 세대에서든** calf 15.700 을 찍는다
    #   (구세대 `clip10` 로그도 동일). 여기서 바꾸는 것은 **토크 곡선의 기울기뿐**이고 속도 벽은 그대로다.
    #   참조 모션의 calf 요구가 max 7.91 rad/s 이므로 15.70 벽은 재현에 지장이 없다.
    #
    # 참조 모션은 관절 속도로는 **전 구간 재현 가능**하다(calf max 7.91 / thigh 12.09 / hip 4.39,
    # 한계 초과 0.00%). 병목은 속도가 아니라 그 속도대에서 곡선이 깎는 **토크**다.
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.1,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
                armature=0.01,
            ),
            "calf": DCMotorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit=35.5,
                saturation_effort=35.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
                armature=0.01,
            ),
        },
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
