# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg Imitation Tracking + RMA Estimator 환경.

`LegImitationTrackingEnv`(AMP + body-frame 속도추종)를 상속해 **RMA 관측 그룹**으로 재구성한다.
학습 스택(`ActorCriticRMA` + `PPOAMP` + `OnPolicyRunnerParkourAMP` + Estimator)은
`parkour_imitation`의 RMA 계열을 참고했다.

부모 대비 바뀌는 것은 **관측 구조뿐**이다. AMP disc 경로(extras["amp_obs"]),
보상, 종료, RSI, 모션 라이브러리는 모두 부모 그대로 재사용한다.

  - policy(57)        : proprio = gravity(3) + lin_vel_cmd(2) + yaw_vel_cmd(1) +
                        joint_pos_off(17) + joint_vel(17) + actions(17)
  - priv_explicit(6)  : root_link_lin_vel_b(3) + root_link_ang_vel_b(3)  ← Estimator 예측 대상
  - priv_latent(38)   : DR 랜덤화 파라미터 = base_mass(1) + base_com(3) +
                        joint_stiffness_ratio(17) + joint_damping_ratio(17)  ← adaptation 대상
  - history(H×57)     : proprio 링버퍼 (adaptation module 입력)

`_flat_env_mask`: `OnPolicyRunnerParkourAMP`가 AMP reward fusion 에서 무조건 참조한다
(`total = task + amp_weight * flat_mask * disc_reward`). 이 env 는 전부 평지 imitation 이므로
모든 env 에 True 를 주어 AMP 를 전 env 에 적용한다.
"""

from __future__ import annotations

import torch

from .leg_imitation_tracking_env import LegImitationTrackingEnv
from .leg_imitation_tracking_rma_env_cfg import LegImitationTrackingRMAEnvCfg


class LegImitationTrackingRMAEnv(LegImitationTrackingEnv):
    """Leg(17-DOF) AMP + body-frame 속도추종 + RMA Estimator 환경."""

    cfg: LegImitationTrackingRMAEnvCfg

    def __init__(self, cfg: LegImitationTrackingRMAEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── proprio 히스토리 링버퍼 [N, H, 57] ─────────────────────────────
        # adaptation module(history_encoder) 입력. AMP 버퍼(53/59-dim)와 내용이 다르므로 별도 유지.
        self._proprio_history = torch.zeros(
            self.num_envs, self.cfg.history_len, self.cfg.num_proprio, device=self.device
        )

        # ── flat_env_mask [N] — 전부 평지이므로 AMP 를 전 env 에 적용 ────────
        # OnPolicyRunnerParkourAMP 가 무조건 접근하므로 반드시 존재해야 한다.
        self._flat_env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # ── priv_latent read-back 용 base body index ───────────────────────
        # DR 로 랜덤화된 base 질량/CoM 을 shape 인덱스 없이 안전하게 읽기 위한 body id.
        self._robot_base_id, _base_names = self._robot.find_bodies(self.cfg.reference_body)
        assert len(self._robot_base_id) == 1, (
            f"find_bodies('{self.cfg.reference_body}') expected 1 body, got {_base_names}"
        )

    def _get_observations(self) -> dict:
        # 부모 호출: AMP 버퍼 시프트 + self.extras(amp_obs / terminal_amp_obs) 설정.
        # 반환되는 63-dim {"policy"} 는 버리고, 아래에서 RMA 그룹 dict 를 새로 만든다.
        super()._get_observations()

        # ── proprio (57) — base 속도 6 을 제외한 나머지 ────────────────────
        proprio = torch.cat(
            [
                self._robot.data.projected_gravity_b,  # 3
                self._lin_vel_cmd,  # 2 (vx, vy)
                self._yaw_vel_cmd.unsqueeze(-1),  # 1
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 17
                self._robot.data.joint_vel,  # 17
                self.actions,  # 17
            ],
            dim=-1,
        )  # [N, 57]

        # ── priv_explicit (6) — Estimator 예측 대상 (base link frame 속도) ──
        priv_explicit = torch.cat(
            [
                self._robot.data.root_link_lin_vel_b.torch,  # 3
                self._robot.data.root_link_ang_vel_b.torch,  # 3
            ],
            dim=-1,
        )  # [N, 6]

        # ── priv_latent (38) — DR 로 랜덤화된 준정적 물리 파라미터 ──────────
        # base_mass(1) + base_com(3) + joint_stiffness_ratio(17) + joint_damping_ratio(17).
        # shape 인덱스가 불필요한 항목만 사용 (Leg 중첩 계층 USD 안전).
        base_mass = torch.as_tensor(self._robot.root_physx_view.get_masses(), device=self.device)[
            :, self._robot_base_id
        ]  # [N, 1]
        base_com = self._robot.data.body_com_pos_b[:, self._robot_base_id, :].squeeze(1)  # [N, 3]
        # Leg 는 ImplicitActuator → sim-level joint_stiffness 가 실제 PD 게인이라 read-back 직접 가능.
        joint_stiffness_ratio = torch.nan_to_num(
            self._robot.data.joint_stiffness / self._robot.data.default_joint_stiffness - 1.0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )  # [N, 17]
        joint_damping_ratio = torch.nan_to_num(
            self._robot.data.joint_damping / self._robot.data.default_joint_damping - 1.0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )  # [N, 17]
        priv_latent = torch.cat([base_mass, base_com, joint_stiffness_ratio, joint_damping_ratio], dim=-1)  # [N, 38]

        # ── proprio 히스토리 링버퍼 업데이트 (parkour 방식) ────────────────
        # 에피소드 시작(<=1) 시 현재 proprio 로 전체를 채우고, 그 외에는 shift + append.
        self._proprio_history = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([proprio] * self.cfg.history_len, dim=1),
            torch.cat([self._proprio_history[:, 1:], proprio.unsqueeze(1)], dim=1),
        )

        # obs_groups["critic"] = [policy, priv_explicit, priv] 로 concat 되므로 "critic" 키는 불필요.
        return {
            "policy": proprio,
            "priv_explicit": priv_explicit,
            "priv": priv_latent,
            "history": self._proprio_history,
        }

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 부모: terminal amp_obs 캡처 + RSI + 명령 재샘플링 + 에피소드 로깅.
        super()._reset_idx(env_ids)

        # proprio 히스토리 리셋 (리셋된 env 만). None 이면 전체.
        if env_ids is None or len(env_ids) == self.num_envs:
            self._proprio_history[:] = 0.0
        else:
            self._proprio_history[env_ids] = 0.0
