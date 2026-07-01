# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2Recovery-RiseSlow-v0 환경.

Go2RecoveryRisePacingEnv 상속. 기존 base / RisePacing 파일은 절대 수정하지 않는다.

추가 메커니즘:
  r_rise_pace: high-cos 영역에서만 작동하는 uprightness-rate 초과 페널티.
    cos < gate_lo(0.5) → g_high=0 → 페널티 0 (flip 단계 자유 보존)
    cos > gate_hi(0.75) → g_high=1 → 페널티 전활성

수식:
    g_high      = ((cos_dist - lo) / (hi - lo)).clamp(0, 1)
    rise_rate   = delta_cos / step_dt              # per-second
    rise_excess = (rise_rate - target_rate).clamp(min=0)
    r_rise_pace = g_high × rise_excess² × scale   (scale < 0 → 페널티)

캘리브레이션:
    passive_baseline high-cos(>0.5) positive-only rise window median = 0.0225/step (per-step)
    step_dt = 0.02s → 1.124/s.  target_rate = 45% × 1.124 ≈ 0.506 → 0.5/s.

floor 없음: 설계 명시. cos < lo → g_high=0 → 페널티 0. 절대 floor 추가 금지.

신규 buffer: 없음 (_prev_cos_dist는 base _reset_idx에서 0 초기화).
신규 _episode_sums 키: "r_rise_pace" (__init__에서 시딩 — 미시딩 시 첫 step KeyError).
"""

from __future__ import annotations

import torch

from .go2_recovery_rise_pacing_env import Go2RecoveryRisePacingEnv
from .go2_recovery_rise_slow_env_cfg import Go2RecoveryRiseSlowEnvCfg


class Go2RecoveryRiseSlowEnv(Go2RecoveryRisePacingEnv):
    """Go2 fall-recovery + smoothness state-gate + rise-pace penalty 환경.

    Go2RecoveryRisePacingEnv를 상속:
      - 기존 smoothness state-gate (g, smooth_gate_*) 그대로 유지.
      - 신규 항: r_rise_pace — high-cos 구간에서 빠른 uprightness 상승률 억제.

    obs / buffer: 변경 없음 (observation_space=42 불변).
    _reset_idx / _get_observations: override 없음 (RisePacing/base 그대로).
    """

    cfg: Go2RecoveryRiseSlowEnvCfg

    def __init__(self, cfg: Go2RecoveryRiseSlowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # r_rise_pace episode 누적 버퍼 시딩.
        # base _episode_sums dict에 없는 키이므로, rewards loop의
        #   self._episode_sums[key] += weighted 가 첫 step에 KeyError를 일으킨다.
        # super().__init__() 직후(= _episode_sums dict 생성 직후)에만 삽입 가능.
        self._episode_sums["r_rise_pace"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # ── rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """_get_rewards 전체 override (option A — RisePacing 전체 복사 + r_rise_pace 삽입).

        super()._get_rewards() 호출 금지 — side-effect 이중실행(buffer 갱신, episode_sums 누적) 방지.
        RisePacing 대비 변경점 (★ 표시):
          1. delta_cos 계산 직후 r_rise_pace 블록 삽입.
          2. rewards dict에 "r_rise_pace": r_rise_pace 추가.
          그 외 모든 줄은 RisePacing과 byte-identical (side-effect 누락 방지).
        """
        # ── cos_dist: upright 정렬도 ──────────────────────────────────────────
        # base와 byte-identical
        from isaaclab.utils.math import quat_apply

        _up_world = torch.zeros(self.num_envs, 3, device=self.device)
        _up_world[:, 2] = 1.0
        root_up = quat_apply(self._robot.data.root_quat_w, _up_world)
        cos_dist = (root_up * _up_world).sum(dim=-1)

        # ★ smoothness state-gate g (RisePacing iter3: floor 포함, 변경 없음)
        # g_raw=0→g=floor: flip 구간(cos≤lo)에서도 floor만큼 smoothness penalty 유지
        # g_raw=1→g=1.0:   거의 일어섬(cos≥hi) → smoothness penalty 전활성
        g_raw = (
            (cos_dist - self.cfg.smooth_gate_cos_lo)
            / (self.cfg.smooth_gate_cos_hi - self.cfg.smooth_gate_cos_lo)
        ).clamp(0.0, 1.0)
        g = self.cfg.smooth_gate_floor + (1.0 - self.cfg.smooth_gate_floor) * g_raw

        # ── r_roll ───────────────────────────────────────────────────────────
        r_roll = (0.5 * cos_dist + 0.5) ** 2

        # ── r_roll_progress ──────────────────────────────────────────────────
        delta_cos = cos_dist - self._prev_cos_dist
        r_roll_progress = delta_cos * self.cfg.roll_progress_weight
        self._prev_cos_dist = cos_dist.clone()  # side-effect: 반드시 포함

        # ★ r_rise_pace: high-cos 영역에서만 작동하는 uprightness-rate 초과 페널티 (신규)
        # ──────────────────────────────────────────────────────────────────────
        # g_high: cos가 gate_lo ~ gate_hi 사이에서 0→1로 선형 증가.
        #   cos < lo(0.5)  → g_high=0 → 페널티 0 (flip 구간 자유 보존)
        #   cos > hi(0.75) → g_high=1 → 페널티 전활성
        # rise_rate: per-second (delta_cos / step_dt)
        # rise_excess: target_rate 초과분만 페널티. 느린 상승은 무해.
        # rise_pace_scale < 0 → 페널티 (설계 명시).
        # floor 없음: 절대 추가 금지 (설계 명시).
        g_high = (
            (cos_dist - self.cfg.rise_pace_gate_cos_lo)
            / (self.cfg.rise_pace_gate_cos_hi - self.cfg.rise_pace_gate_cos_lo)
        ).clamp(0.0, 1.0)
        rise_rate = delta_cos / self.step_dt
        rise_excess = (rise_rate - self.cfg.rise_pace_target_rate).clamp(min=0.0)
        r_rise_pace = g_high * rise_excess.square() * self.cfg.rise_pace_scale
        # ──────────────────────────────────────────────────────────────────────

        # ── r_stand ──────────────────────────────────────────────────────────
        r_stand = self._calc_r_stand()
        stand_active = cos_dist > self.cfg.stand_cos_threshold
        r_stand = torch.where(stand_active, r_stand, torch.zeros_like(r_stand))

        # ── reward_reset ─────────────────────────────────────────────────────
        reward_reset = (
            self.cfg.roll_reward_weight * r_roll + self.cfg.stand_reward_weight * r_stand
        ) * self.cfg.reward_reset_scale

        # ── smoothness / regularization penalties ─────────────────────────────
        # cfg.* scale은 Go2RecoveryRisePacingEnvCfg에서 override됨 → 자동 반영
        action_rate_l2 = (
            torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_l2_scale
        )

        first_step_mask = (self._previous_actions != 0).float()
        # ★ g 곱: smoothness state-gate (RisePacing 동일)
        action_smoothness_1 = (
            torch.sum(
                torch.square(self._processed_actions - self._last_joint_pos_target) * first_step_mask,
                dim=1,
            )
            * self.cfg.action_smoothness_1_scale
            * g
        )

        # ★ g 곱 (RisePacing 동일)
        action_smoothness_2 = (
            torch.sum(
                torch.square(self._actions - 2.0 * self._previous_actions + self._last_last_actions),
                dim=1,
            )
            * self.cfg.action_smoothness_2_scale
            * g
        )

        current_torque = self._robot.data.applied_torque
        delta_torques = (
            torch.sum(torch.square(current_torque - self._last_torque), dim=1) * self.cfg.delta_torques_scale
        )

        joint_vel = self._robot.data.joint_vel
        dof_acc = (joint_vel - self._previous_joint_vel) / self.step_dt
        # ★ g 곱 (RisePacing 동일)
        dof_acc_l2 = torch.sum(torch.square(dof_acc), dim=1) * self.cfg.dof_acc_l2_scale * g

        # dof_vel_l2: cfg.dof_vel_l2_scale = 0.0 → 자동 무효 (RisePacing 동일)
        dof_vel_l2 = torch.sum(torch.square(joint_vel), dim=1) * self.cfg.dof_vel_l2_scale

        dof_torques_l2 = torch.sum(torch.square(current_torque), dim=1) * self.cfg.dof_torques_l2_scale

        joint_pos = self._robot.data.joint_pos
        soft_lower = self._robot.data.soft_joint_pos_limits[:, :, 0]
        soft_upper = self._robot.data.soft_joint_pos_limits[:, :, 1]
        out_of_limits = (-(joint_pos - soft_lower)).clamp(min=0.0)
        out_of_limits += (joint_pos - soft_upper).clamp(min=0.0)
        dof_pos_limits = torch.sum(out_of_limits, dim=1) * self.cfg.dof_pos_limits_scale

        # ── buffer 갱신: side-effect — 반드시 포함 ───────────────────────────
        self._previous_joint_vel = joint_vel.clone()
        self._last_joint_pos_target = self._processed_actions.clone()
        self._last_torque = current_torque.clone()

        # ── success judgment ──────────────────────────────────────────────────
        success_bonus = self._update_success(cos_dist)  # side-effect: _success_region_mask 갱신

        # r_success_region: base와 동일 (ramp 없음)
        r_success_region = self._success_region_mask.float() * self.cfg.success_region_reward_scale

        # ── 전체 reward 합산 ──────────────────────────────────────────────────
        rewards = {
            "reward_reset": reward_reset,
            "r_roll_progress": r_roll_progress,
            "r_rise_pace": r_rise_pace,  # ★ 신규
            "action_rate_l2": action_rate_l2,
            "action_smoothness_1": action_smoothness_1,
            "action_smoothness_2": action_smoothness_2,
            "delta_torques": delta_torques,
            "dof_acc_l2": dof_acc_l2,
            "dof_vel_l2": dof_vel_l2,
            "dof_torques_l2": dof_torques_l2,
            "dof_pos_limits": dof_pos_limits,
            "success_bonus": success_bonus,
            "r_success_region": r_success_region,
        }

        reward = torch.zeros(self.num_envs, device=self.device)
        for key, value in rewards.items():
            weighted = value * self.step_dt
            reward += weighted
            self._episode_sums[key] += weighted

        # settle 마스킹
        if self.cfg.settle_max_steps > 0:
            settle_active = self._settle_counter < self._settle_steps
            reward = torch.where(settle_active, torch.zeros_like(reward), reward)

        return reward
