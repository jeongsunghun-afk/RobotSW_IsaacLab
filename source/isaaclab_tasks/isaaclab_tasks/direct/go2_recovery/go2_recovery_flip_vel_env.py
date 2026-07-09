# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2Recovery-FlipVel-v0 환경.

Go2RecoveryRiseSlowEnv 상속. base/RiseSlow 파일은 절대 수정하지 않는다.

추가 메커니즘:
  r_joint_vel: belly-up flip 시 발생하는 과도한 관절 속도(whip)를 억제하는
    threshold-excess 페널티.

수식:
    joint_speed_excess = (|q̇| - thr).clamp(min=0)      # (N, 12)
    r_joint_vel        = scale × sum(excess, dim=1)      # (N,), scale < 0 → 페널티
    _ct                = min(1, max(0, (step - warmup) / ramp))
    r_joint_vel       *= _ct                             # curriculum ramp

파라미터 (측정 근거):
  joint_vel_thr = 8.0 rad/s  — 정상 기립 동작 max 7.8 / whip 8~22 → 8이 분리점
  joint_vel_scale = -0.02    — 12관절 합산 초기 페널티 ~-4.6 ≈ r_rise_pace의 ~4배, 즉사 아님
  joint_vel_warmup_steps = 48000  — 복구 먼저 학습 후 ramp 시작
  joint_vel_ramp_steps   = 48000  — 48k~96k 스텝에 걸쳐 선형 증가

self-gating 설계: threshold=8이 정상동작(<8)을 자동 배제하므로 cos-gate 불필요.

신규 buffer: 없음 (common_step_counter 사용, joint_vel은 매 step 읽음).
신규 _episode_sums 키: "r_joint_vel" (__init__에서 시딩 — 미시딩 시 첫 step KeyError).
obs 42 불변.
"""

from __future__ import annotations

import torch

from .go2_recovery_rise_slow_env import Go2RecoveryRiseSlowEnv
from .go2_recovery_flip_vel_env_cfg import Go2RecoveryFlipVelEnvCfg


class Go2RecoveryFlipVelEnv(Go2RecoveryRiseSlowEnv):
    """Go2 fall-recovery + rise-pace penalty + joint velocity whip suppression 환경.

    Go2RecoveryRiseSlowEnv를 상속:
      - 기존 r_rise_pace (high-cos uprightness-rate 페널티) 그대로 유지.
      - 신규 항: r_joint_vel — threshold 초과 관절 속도 페널티.

    obs / buffer: 변경 없음 (observation_space=42 불변).
    _reset_idx / _get_observations: override 없음 (RiseSlow/base 그대로).
    """

    cfg: Go2RecoveryFlipVelEnvCfg

    def __init__(self, cfg: Go2RecoveryFlipVelEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # r_joint_vel episode 누적 버퍼 시딩.
        # base _episode_sums dict에 없는 키이므로, rewards loop의
        #   self._episode_sums[key] += weighted 가 첫 step에 KeyError를 일으킨다.
        # super().__init__() 직후(= _episode_sums dict 생성 직후)에만 삽입 가능.
        self._episode_sums["r_joint_vel"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # ── rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """_get_rewards 전체 override (option A — RiseSlow 전체 복사 + r_joint_vel 삽입).

        super()._get_rewards() 호출 금지 — side-effect 이중실행(buffer 갱신, episode_sums 누적) 방지.
        RiseSlow 대비 변경점 (★ 표시):
          1. joint_vel 계산 직후 r_joint_vel 블록 삽입 (기존 joint_vel 변수 재사용).
          2. rewards dict에 "r_joint_vel": r_joint_vel 추가.
          그 외 모든 줄은 RiseSlow와 byte-identical (side-effect 누락 방지).
        """
        # ── cos_dist: upright 정렬도 ──────────────────────────────────────────
        from isaaclab.utils.math import quat_apply

        _up_world = torch.zeros(self.num_envs, 3, device=self.device)
        _up_world[:, 2] = 1.0
        root_up = quat_apply(self._robot.data.root_quat_w, _up_world)
        cos_dist = (root_up * _up_world).sum(dim=-1)

        # smoothness state-gate g (RiseSlow과 동일)
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

        # ── r_rise_pace: high-cos 영역에서만 작동하는 uprightness-rate 초과 페널티 ──
        g_high = (
            (cos_dist - self.cfg.rise_pace_gate_cos_lo)
            / (self.cfg.rise_pace_gate_cos_hi - self.cfg.rise_pace_gate_cos_lo)
        ).clamp(0.0, 1.0)
        rise_rate = delta_cos / self.step_dt
        rise_excess = (rise_rate - self.cfg.rise_pace_target_rate).clamp(min=0.0)
        r_rise_pace = g_high * rise_excess.square() * self.cfg.rise_pace_scale

        # ── r_stand ──────────────────────────────────────────────────────────
        r_stand = self._calc_r_stand()
        stand_active = cos_dist > self.cfg.stand_cos_threshold
        r_stand = torch.where(stand_active, r_stand, torch.zeros_like(r_stand))

        # ── reward_reset ─────────────────────────────────────────────────────
        reward_reset = (
            self.cfg.roll_reward_weight * r_roll + self.cfg.stand_reward_weight * r_stand
        ) * self.cfg.reward_reset_scale

        # ── smoothness / regularization penalties ─────────────────────────────
        action_rate_l2 = (
            torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_l2_scale
        )

        first_step_mask = (self._previous_actions != 0).float()
        action_smoothness_1 = (
            torch.sum(
                torch.square(self._processed_actions - self._last_joint_pos_target) * first_step_mask,
                dim=1,
            )
            * self.cfg.action_smoothness_1_scale
            * g
        )

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
        dof_acc_l2 = torch.sum(torch.square(dof_acc), dim=1) * self.cfg.dof_acc_l2_scale * g

        dof_vel_l2 = torch.sum(torch.square(joint_vel), dim=1) * self.cfg.dof_vel_l2_scale

        dof_torques_l2 = torch.sum(torch.square(current_torque), dim=1) * self.cfg.dof_torques_l2_scale

        joint_pos = self._robot.data.joint_pos
        soft_lower = self._robot.data.soft_joint_pos_limits[:, :, 0]
        soft_upper = self._robot.data.soft_joint_pos_limits[:, :, 1]
        out_of_limits = (-(joint_pos - soft_lower)).clamp(min=0.0)
        out_of_limits += (joint_pos - soft_upper).clamp(min=0.0)
        dof_pos_limits = torch.sum(out_of_limits, dim=1) * self.cfg.dof_pos_limits_scale

        # ★ r_joint_vel: threshold-excess 관절 속도 페널티 (신규)
        # ──────────────────────────────────────────────────────────────────────
        # joint_vel은 위에서 이미 읽은 변수를 재사용 (중복 읽기 없음).
        # self-gating: threshold=8.0이 정상 기립 동작(<7.8)을 자동 배제 → cos-gate 불필요.
        # curriculum ramp: warmup 이후 ramp_steps에 걸쳐 0→1 선형 증가.
        joint_speed_excess = (joint_vel.abs() - self.cfg.joint_vel_thr).clamp(min=0.0)  # (N, 12)
        r_joint_vel = self.cfg.joint_vel_scale * joint_speed_excess.sum(dim=1)  # (N,), ≤0
        _ct = min(
            1.0,
            max(
                0.0,
                (float(self.common_step_counter) - float(self.cfg.joint_vel_warmup_steps))
                / max(1.0, float(self.cfg.joint_vel_ramp_steps)),
            ),
        )
        r_joint_vel = r_joint_vel * _ct
        # ──────────────────────────────────────────────────────────────────────

        # ── buffer 갱신: side-effect — 반드시 포함 ───────────────────────────
        self._previous_joint_vel = joint_vel.clone()
        self._last_joint_pos_target = self._processed_actions.clone()
        self._last_torque = current_torque.clone()

        # ── success judgment ──────────────────────────────────────────────────
        success_bonus = self._update_success(cos_dist)  # side-effect: _success_region_mask 갱신

        r_success_region = self._success_region_mask.float() * self.cfg.success_region_reward_scale

        # ── 전체 reward 합산 ──────────────────────────────────────────────────
        rewards = {
            "reward_reset": reward_reset,
            "r_roll_progress": r_roll_progress,
            "r_rise_pace": r_rise_pace,
            "r_joint_vel": r_joint_vel,  # ★ 신규
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
