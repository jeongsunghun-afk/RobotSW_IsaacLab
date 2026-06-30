# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Go2Recovery-RisePacing-v0.

상속 전략: Go2RecoveryEnvCfg를 상속해 state-gate 전용 파라미터만 추가/override.
base(Go2RecoveryEnvCfg)는 절대 수정하지 않는다.

새 파라미터:
  smooth_gate_cos_lo: float = 0.0  — gate 시작 cos 값 (이하: g=0, penalty 비활성)
  smooth_gate_cos_hi: float = 0.5  — gate 포화 cos 값 (이상: g=1, penalty 전활성)

  게이트 함수 (per-env):
    g = ((cos_dist - lo) / (hi - lo)).clamp(0, 1)
    cos=-1/0 → g=0 (뒤집힘/넘어짐: smoothness 면제, flip 임펄스 자유)
    cos=0.5+ → g=1 (거의 일어섬: smoothness 전활성, 정착만 부드럽게)

Override scale:
  dof_acc_l2_scale:          -2.5e-7 → -2.5e-6  (10× 강화, gate 적용)
  dof_vel_l2_scale:          0.0 유지            (1차 회귀 원인 — 0으로 되돌림)
  action_smoothness_1_scale: -0.1    → -0.2      (부드러움 강화, gate 적용)
  action_smoothness_2_scale: -0.1    → -0.2      (부드러움 강화, gate 적용)
"""

from isaaclab.utils.configclass import configclass

from .go2_recovery_env_cfg import Go2RecoveryEnvCfg


@configclass
class Go2RecoveryRisePacingEnvCfg(Go2RecoveryEnvCfg):
    """Go2 fall-recovery + uprightness state-gate 환경 설정.

    uprightness(cos_dist)로 smoothness penalty를 게이팅:
      - cos 낮음(뒤집힘/넘어짐): g=0 → penalty 비활성, flip 임펄스 자유
      - cos 높음(거의 일어섬): g=1 → penalty 전활성, 정착만 부드럽게
      → 1차 회귀(dof_vel penalty가 flip을 죽임)의 근본 해결.

    state-gate 적용 항: action_smoothness_1, action_smoothness_2, dof_acc_l2.
    r_success_region/success_bonus는 base와 동일(ramp 없음).
    """

    # ── state-gate 파라미터 ───────────────────────────────────────────────────
    # gate 시작 cos 값: cos≤lo → g=0 (penalty 완전 비활성)
    smooth_gate_cos_lo: float = 0.0

    # gate 포화 cos 값: cos≥hi → g=1 (penalty 전활성)
    # 기본값 0.5 → cos=0.25에서 g=0.5 (선형 보간 중간점)
    smooth_gate_cos_hi: float = 0.5

    # ── penalty override ──────────────────────────────────────────────────────
    # joint 가속도 penalty 강화: -2.5e-7 → -2.5e-6 (10× 강화, gate 적용)
    dof_acc_l2_scale: float = -2.5e-6

    # dof_vel_l2: 0.0 유지 (1차의 -0.01이 flip을 죽인 원인 → 되돌림)
    dof_vel_l2_scale: float = 0.0

    # target 기반 1차 smoothness 강화: -0.1 → -0.2 (gate 적용)
    action_smoothness_1_scale: float = -0.2

    # 2차 action smoothness 강화: -0.1 → -0.2 (gate 적용)
    action_smoothness_2_scale: float = -0.2
