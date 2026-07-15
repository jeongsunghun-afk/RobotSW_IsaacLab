# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Go2Recovery-RiseSlow-v0 (IsaacLab-6.0).

상속 전략: Go2RecoveryRisePacingEnvCfg를 상속해 rise-pace 전용 파라미터만 추가.
base / RisePacing 파일은 절대 수정하지 않는다.

신규 파라미터:
  rise_pace_gate_cos_lo: float = 0.5
    gate 시작 cos 값. cos ≤ lo → g_high=0 → r_rise_pace=0 (flip 구간 자유 보존).
  rise_pace_gate_cos_hi: float = 0.75
    gate 포화 cos 값. cos ≥ hi → g_high=1 → r_rise_pace 전활성.
  rise_pace_target_rate: float = 0.5
    허용 uprightness 상승률 (per-second).
    캘리브레이션: passive_baseline high-cos(>0.5) positive-only rise window median=0.0225/step
    step_dt=0.02s → 1.124/s. target = 45% × 1.124 ≈ 0.506 → 0.5/s.
  rise_pace_scale: float = -0.5
    r_rise_pace weight (음수 → 페널티).

gate 함수 (per-env):
    g_high = ((cos_dist - lo) / (hi - lo)).clamp(0, 1)
    cos < 0.5  → g_high=0 → 페널티 0  (flip 단계 자유 보존)
    cos ∈ [0.5, 0.75] → 선형 증가
    cos > 0.75 → g_high=1 → 페널티 전활성

floor 없음: 설계 명시. 절대 추가 금지.
"""

from isaaclab.utils.configclass import configclass

from .go2_recovery_rise_pacing_env_cfg import Go2RecoveryRisePacingEnvCfg


@configclass
class Go2RecoveryRiseSlowEnvCfg(Go2RecoveryRisePacingEnvCfg):
    """Go2 fall-recovery + smoothness state-gate + rise-pace penalty 환경 설정.

    Go2RecoveryRisePacingEnvCfg를 상속:
      - smooth_gate_* (g 계산): 변경 없음.
      - dof_acc/smoothness scale override: 변경 없음.
      - 신규: rise_pace_* 4개 파라미터 추가.
    """

    # ── rise-pace gate 파라미터 ───────────────────────────────────────────────

    # gate 시작 cos 값: cos ≤ lo → g_high=0 → r_rise_pace=0 (flip 구간 자유 보존)
    rise_pace_gate_cos_lo: float = 0.5

    # gate 포화 cos 값: cos ≥ hi → g_high=1 → r_rise_pace 전활성
    rise_pace_gate_cos_hi: float = 0.75

    # 허용 uprightness 상승률 (per-second).
    # 캘리브레이션 (probe_highcos_rate.py, passive_baseline, seed=12345, 512 envs):
    #   high-cos(>0.5) rise window positive-only median = 0.0225/step
    #   step_dt=0.02s → 1.124/s
    #   target = 45% × 1.124 ≈ 0.506 → 0.5/s  (안전범위: [0.393, 0.675]/s)
    #
    # 참고: 이전 값 1.7/s는 per-env peak median(0.0735/step)을 per-second로 환산한 값으로
    #       rush spike에 오염된 overestimate — 본 median 실측으로 대체.
    #
    # scale 안전 부등식 (scale=-0.5):
    #   break-even(penalty=success_region) rate ≈ 2.95/s (= 0.5 + √6)
    #   p90 high-cos rise_rate=3.325/s: penalty=-0.080/step > success_region=+0.06/step (⚠ flag)
    #   p99 high-cos rise_rate=5.815/s: penalty=-0.282/step  (스파이크, 수 스텝만)
    #   episodic: success_region 300step×0.06=+18.0 >> rush 5step×0.08=-0.40 → collapse 위험 없음
    rise_pace_target_rate: float = 0.5

    # r_rise_pace weight (음수 → 페널티).
    rise_pace_scale: float = -0.5
