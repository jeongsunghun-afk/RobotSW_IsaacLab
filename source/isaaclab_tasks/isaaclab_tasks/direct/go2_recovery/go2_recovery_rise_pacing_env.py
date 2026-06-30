# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2Recovery-RisePacing-v0 환경.

base(Go2RecoveryEnv) 상속. base 파일은 절대 수정하지 않는다.

추가 메커니즘:
  1. _get_rewards override (option A — 전체 복사):
     - super()._get_rewards() 호출 금지 (side-effect 이중실행 방지).
     - action_smoothness_1/2, dof_acc_l2 항에 uprightness state-gate(g) 곱.
     - 그 외 모든 줄은 base와 byte-identical (side-effect 누락 방지).
  2. cfg override(dof_acc/smoothness scale, smooth_gate_cos_lo/hi)는 cfg 파일에서 선언됨.
     → 복사된 _get_rewards가 self.cfg.*를 읽으므로 자동 반영.

state-gate:
  g = ((cos_dist - lo) / (hi - lo)).clamp(0, 1)
  cos 낮음 → g=0 → flip 단계 smoothness 면제 (1차 회귀 방지)
  cos 높음 → g=1 → 정착 단계 smoothness 전활성 (anti-snap)
"""

from __future__ import annotations

import torch

from .go2_recovery_env import Go2RecoveryEnv
from .go2_recovery_rise_pacing_env_cfg import Go2RecoveryRisePacingEnvCfg


class Go2RecoveryRisePacingEnv(Go2RecoveryEnv):
    """Go2 fall-recovery + uprightness state-gate 환경.

    uprightness(cos_dist)로 smoothness penalty를 게이팅:
      - cos 낮음(뒤집힘/넘어짐): g=0 → penalty 비활성, flip 임펄스 자유
      - cos 높음(거의 일어섬): g=1 → penalty 전활성, 정착만 부드럽게

    obs / amp: 변경 없음 (observation_space=42 불변).
    _reset_idx / _get_observations: override 없음 (base 그대로).
    """

    cfg: Go2RecoveryRisePacingEnvCfg

    # ── rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """_get_rewards 전체 override (option A).

        super()._get_rewards() 호출 금지 — side-effect 이중실행(buffer 갱신, episode_sums 누적) 방지.
        base의 모든 side-effect 줄을 포함하되, action_smoothness_1/2 와 dof_acc_l2 에만 g(gate) 곱.

        변경된 항 (3개):
            action_smoothness_1 *= g
            action_smoothness_2 *= g
            dof_acc_l2         *= g
        """
        # ── cos_dist: upright 정렬도 ──────────────────────────────────────────
        # base와 byte-identical
        from isaaclab.utils.math import quat_apply

        _up_world = torch.zeros(self.num_envs, 3, device=self.device)
        _up_world[:, 2] = 1.0
        root_up = quat_apply(self._robot.data.root_quat_w.torch, _up_world)
        cos_dist = (root_up * _up_world).sum(dim=-1)

        # ★ state-gate g 계산 (cos_dist 직후, rewards dict 구성 전)
        # g=0: 뒤집힘/넘어짐(cos≤lo) → smoothness penalty 면제, flip 임펄스 자유
        # g=1: 거의 일어섬(cos≥hi) → smoothness penalty 전활성, 정착만 부드럽게
        # (num_envs,) shape — 항과 elementwise 정합
        g = (
            (cos_dist - self.cfg.smooth_gate_cos_lo)
            / (self.cfg.smooth_gate_cos_hi - self.cfg.smooth_gate_cos_lo)
        ).clamp(0.0, 1.0)

        # ── r_roll ───────────────────────────────────────────────────────────
        r_roll = (0.5 * cos_dist + 0.5) ** 2

        # ── r_roll_progress ──────────────────────────────────────────────────
        delta_cos = cos_dist - self._prev_cos_dist
        r_roll_progress = delta_cos * self.cfg.roll_progress_weight
        self._prev_cos_dist = cos_dist.clone()  # side-effect: 반드시 포함

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
        # ★ g 곱: rewards dict 구성 전에 적용 → reward + episode_sums 양쪽 정직하게 반영
        action_smoothness_1 = (
            torch.sum(
                torch.square(self._processed_actions - self._last_joint_pos_target) * first_step_mask,
                dim=1,
            )
            * self.cfg.action_smoothness_1_scale
            * g
        )

        # ★ g 곱
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
        # ★ g 곱
        dof_acc_l2 = torch.sum(torch.square(dof_acc), dim=1) * self.cfg.dof_acc_l2_scale * g

        # dof_vel_l2: cfg.dof_vel_l2_scale = 0.0 → 자동 무효 (1차 회귀 원인 — 되돌림)
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
