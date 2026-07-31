# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg Imitation Tracking + RMA Estimator (+ Domain Randomization) 환경 설정.

`LegImitationTrackingEnvCfg`(AMP + body-frame 속도추종)에 **RMA 학습 스택 + 도메인 랜덤화(DR)**를
얹은 변형이다. 학습 모듈은 `parkour_imitation`의 RMA 계열(`ActorCriticRMA` + `PPOAMP` +
`OnPolicyRunnerParkourAMP`)을 참고했다.

핵심 차이 (vs `LegImitationTrackingEnvCfg`):
  - **policy 관측에서 base 속도(6)를 분리**한다. 실기에서 base 선/각속도는 직접 측정할 수 없으므로,
    RMA Estimator 가 proprio 로부터 추정하도록 privileged state(priv_explicit)로 뺀다.
  - **도메인 랜덤화(DR)** 를 추가한다(`events`). priv_latent 이 실제 랜덤화된 물리 파라미터를
    담게 되어, RMA 의 나머지 절반(history_encoder → priv_latent 재현, DAgger 적응)이 실질적으로
    동작한다(sim2real). — 2026-07-24 사용자 결정.
  - 관측이 단일 "policy" 텐서가 아니라 **obs 그룹 dict** 로 바뀐다:
      policy(57)        : proprio (gravity + cmd + joint_pos_off + joint_vel + actions)
      priv_explicit(6)  : root_link_lin_vel_b(3) + root_link_ang_vel_b(3)  ← Estimator 예측 대상
      priv_latent(38)   : base_mass(1) + base_com(3) + joint_stiffness_ratio(17) + joint_damping_ratio(17)
      history(H×57)     : proprio 링버퍼 (adaptation module 입력)
      critic = policy + priv_explicit + priv_latent (obs_groups 에서 concat)

**priv_latent = mass + com + gains (friction 제외) 인 이유.** friction 은 per-shape 값이라
read-back 에 collision-shape 인덱스 계산이 필요한데, Leg 는 URDF 재임포트로 만든 **중첩 계층** USD 라
shape 인덱싱이 취약하다(과거 contact sensor 가 중첩 계층에서 base 만 매칭한 전례). mass/com/gains 는
`get_masses()` / `body_com_pos_b` / `joint_stiffness` 로 shape 인덱스 없이 안전하게 읽힌다.
friction 랜덤화 자체는 robustness 용으로 유지하되(events.body_physics_material), priv_latent 엔 넣지
않는다. mass/com/gains 만으로도 adaptation module 의 타당한 학습 대상이 된다.

AMP / 보상 / 종료 / 명령 범위 등은 부모에서 그대로 상속한다.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .leg_imitation_tracking_env_cfg import NUM_FEET, NUM_JOINTS, LegImitationTrackingEnvCfg


@configclass
class EventCfg:
    """Leg RMA 환경의 도메인 랜덤화(DR) 이벤트.

    표준 sim-to-real DR 세트 (RMA-style). priv_latent 이 read-back 하는 항목(mass/com/gains)과
    robustness 용 항목(friction/push)을 함께 둔다.
      - body_physics_material : 전 body static/dynamic 마찰 (robustness; priv_latent 미포함)
      - add_base_mass         : base 질량 섭동 → priv_latent.base_mass
      - randomize_com         : base CoM xyz 오프셋 → priv_latent.base_com
      - push_robot            : 주기적 속도 임펄스 (robustness)
      - randomize_actuator_gains : joint stiffness/damping ±10% → priv_latent.joint_*_ratio
    """

    # 전 body 마찰 랜덤화 (robustness). Leg base body = "Base", feet = ".*_foot_link".
    body_physics_material = EventTerm(
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

    # base 질량 섭동 (배터리/페이로드 변동). Leg 는 38 kg 급이라 Go2 표준(-1,+3)보다 약간 넓게.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
            "mass_distribution_params": (-1.5, 3.0),
            "operation": "add",
        },
    )

    # base CoM 오프셋 (질량 분포 드리프트 → 균형/관성 영향).
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
            "com_range": {
                "x": (-0.05, 0.05),
                "y": (-0.03, 0.03),
                "z": (-0.02, 0.02),
            },
        },
    )

    # push DR: base 에 주기적 랜덤 속도 임펄스 (sim-to-real robustness).
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    # actuator gains DR: joint stiffness/damping ±10% scale (startup).
    # priv_latent.joint_stiffness_ratio / joint_damping_ratio 에 비-상수 신호를 제공한다.
    # Leg 는 ImplicitActuator 라 sim-level joint_stiffness 가 곧 실제 PD 게인 → read-back 직접 가능
    # (parkour 의 DCMotor 우회 불필요). 고속 추종 안정성을 위해 범위는 ±10% 로 보수적.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class LegImitationTrackingRMAEnvCfg(LegImitationTrackingEnvCfg):
    """Leg Imitation Tracking + RMA Estimator + DR 환경 설정.

    관측 dict (obs_groups 로 라우팅; DirectRLEnv 는 int observation_space 로 Space 만 생성):
        policy:        proprio                              = 57
        priv_explicit: root_link_lin_vel_b + ang_vel_b      = 6
        priv:          base_mass + base_com + gains ratios  = 38
        history:       history_len × proprio                = 10 × 57
        critic total:  policy + priv_explicit + priv        = 57 + 6 + 38 = 101
    """

    # proprio(policy) 관측 차원. base 속도 6 을 뺀 값 = 63 - 6 = 57.
    #   gravity(3) + lin_vel_cmd(2) + yaw_vel_cmd(1) + joint_pos_off(17) + joint_vel(17) + actions(17)
    # DirectRLEnv 는 이 int 로 Space 를 만들지만, 러너는 obs_groups dict 로 관측을 라우팅한다.
    num_proprio: int = 3 + 2 + 1 + 3 * NUM_JOINTS  # = 57
    observation_space: int = 3 + 2 + 1 + 3 * NUM_JOINTS  # = 57 (runner overrides with dict obs_groups)

    # priv_explicit: Estimator 가 proprio 로부터 예측하는 privileged state (base 선/각속도).
    num_priv_explicit: int = 3 + 3  # = 6

    # priv_latent 차원: base_mass(1) + base_com(3) + joint_stiffness_ratio(17) + joint_damping_ratio(17).
    priv_latent_dim: int = 1 + 3 + NUM_JOINTS + NUM_JOINTS  # = 38

    # proprio 히스토리 길이. StateHistoryEncoder 는 10 / 20 / 50 만 허용한다.
    history_len: int = 10

    # 도메인 랜덤화 이벤트 (DirectRLEnv 가 EventManager 로 startup/interval 적용).
    events: EventCfg = EventCfg()

    def __post_init__(self):  # type: ignore[override]
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        assert self.num_proprio == 3 + 2 + 1 + 3 * NUM_JOINTS
        assert self.num_priv_explicit == 6
        assert self.priv_latent_dim == 1 + 3 + 2 * NUM_JOINTS
        assert NUM_FEET == 4
