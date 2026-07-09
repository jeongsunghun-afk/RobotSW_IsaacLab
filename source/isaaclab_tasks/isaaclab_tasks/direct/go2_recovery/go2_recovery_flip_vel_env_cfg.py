# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Go2Recovery-FlipVel-v0.

상속 전략: Go2RecoveryRiseSlowEnvCfg를 상속해 joint velocity 전용 파라미터만 추가.
base / RiseSlow 파일은 절대 수정하지 않는다.

신규 파라미터:
  joint_vel_thr: float = 8.0
    페널티 발동 임계값 [rad/s].
    측정 근거: 정상 기립 동작 max ≈ 7.8 rad/s, belly-up whip 8~22 rad/s → 8.0이 분리점.
    임계값 이하 동작에는 패널티 없음 (self-gating).
  joint_vel_scale: float = -0.15
    r_joint_vel weight (음수 → 페널티).
    magnitude: ep-sum belly-up 기준 ~-0.023 (초과분 감소 30.3%). sweet-spot 확정.
  joint_vel_warmup_steps: int = 48000
    curriculum ramp 시작 전 warmup 스텝 수.
    의도: 복구 동작을 먼저 학습한 후 속도 페널티를 점진적으로 도입.
  joint_vel_ramp_steps: int = 48000
    warmup 종료 후 페널티가 0→1으로 선형 증가하는 스텝 수.
    ramp 완료 시점: warmup_steps + ramp_steps = 96000 스텝.
"""

from isaaclab.utils import configclass

from .go2_recovery_rise_slow_env_cfg import Go2RecoveryRiseSlowEnvCfg


@configclass
class Go2RecoveryFlipVelEnvCfg(Go2RecoveryRiseSlowEnvCfg):
    """Go2 fall-recovery + rise-pace penalty + joint velocity whip suppression 환경 설정.

    Go2RecoveryRiseSlowEnvCfg를 상속:
      - rise_pace_* 파라미터: 변경 없음.
      - smooth_gate_* 파라미터: 변경 없음.
      - 신규: joint_vel_* 4개 파라미터 추가.
    """

    # ── joint velocity 페널티 파라미터 ──────────────────────────────────────

    # 페널티 발동 임계값 [rad/s].
    # 정상 기립 동작 max ≈ 7.8 rad/s / belly-up whip 8~22 rad/s → 8.0이 분리점.
    joint_vel_thr: float = 8.0

    # r_joint_vel weight (음수 → 페널티).
    # ep-sum belly-up 기준 ~-0.023 (초과분 감소 30.3%). sweet-spot 확정.
    joint_vel_scale: float = -0.15

    # curriculum ramp 시작 전 warmup 스텝 수.
    # 복구 동작 먼저 학습 후 속도 페널티 도입.
    joint_vel_warmup_steps: int = 48000

    # warmup 종료 후 페널티 0→1 선형 증가 스텝 수.
    # ramp 완료: warmup_steps + ramp_steps = 96000 스텝.
    joint_vel_ramp_steps: int = 48000
