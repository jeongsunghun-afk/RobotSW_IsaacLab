# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from rsl_rl.networks import MLP, EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    """AMP discriminator (MLP). 입력 뒤쪽 ``cond_dim`` 열은 **조건 벡터**로 취급한다.

    입력 레이아웃은 ``[kinematics (kin_dim) | condition (cond_dim)]`` 이다. 조건 열은 env 가 붙여 주며
    (예: 정규화한 명령 속도 + 유효 비트), 정규화기와 gradient penalty 는 kinematics 열에만 적용한다.
    ``cond_dim=0`` 이면 기존 무조건부 discriminator 와 완전히 같다.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[1024, 512],
        activation="relu",
        device="cuda",
        disc_reward_type="ls_gan",  # "ls_gan" | "bce" (MimicKit: -log(1-sigmoid(logit)))
        norm_clip=None,  # None=clip없음, 10.0=MimicKit 스타일
        cond_dim: int = 0,
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.cond_dim = int(cond_dim)
        self.kin_dim = input_dim - self.cond_dim
        assert self.kin_dim > 0, f"cond_dim({cond_dim}) 이 input_dim({input_dim}) 보다 작아야 한다"
        self.disc_reward_type = disc_reward_type
        self.norm_clip = norm_clip

        # AMP loss parameter
        self.amp_reward_coef = 1.5  # AMP style 보상 스케일
        # 조건부 D 에서 보상을 무조건부 평가와 섞는 비율 (0=조건부만). 알고리즘이 cfg 로 덮어쓴다.
        self.cond_reward_blend = 0.0

        # Empirical Normalizer — kinematics 열만 정규화 (조건 열은 이미 [−1, 1] 스케일)
        self.amp_obs_normalizer = EmpiricalNormalization(self.kin_dim).to(self.device)

        # WGAN reward normalizer: (D - μ̂) / σ̂  (Eq 4, arXiv:2206.11693)
        # 항상 생성되지만 disc_reward_type="wgan"일 때만 사용됨.
        # forward()는 순수 정규화만 수행하므로 update()를 별도 호출해야 함.
        self.reward_normalizer = EmpiricalNormalization(1).to(self.device)

        # discriminator network (input -> hidden_dims -> 1)
        self.trunk = MLP(input_dim, 1, hidden_dims, activation)
        self.trunk.to(self.device)

    # ── 조건 열 헬퍼 ────────────────────────────────────────────
    def drop_condition(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """조건 열(유효 비트 포함)을 0 으로 지운 사본을 돌려준다. ``cond_dim=0`` 이면 입력 그대로."""
        if self.cond_dim == 0:
            return amp_obs
        out = amp_obs.clone()
        out[:, self.kin_dim :] = 0.0
        return out

    def _normalize(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Kinematics 열 정규화 + 선택적 clip, 조건 열은 그대로 이어 붙인다."""
        norm = self.amp_obs_normalizer(amp_obs[:, : self.kin_dim])
        if self.norm_clip is not None:
            norm = torch.clamp(norm, -self.norm_clip, self.norm_clip)
        if self.cond_dim > 0:
            norm = torch.cat([norm, amp_obs[:, self.kin_dim :]], dim=-1)
        return norm

    def update_normalization(self, amp_obs: torch.Tensor) -> None:
        """Update the empirical normalizer statistics using real simulation observations."""
        self.amp_obs_normalizer.update(amp_obs[:, : self.kin_dim])

    def _reward_from_logits(self, disc_logits: torch.Tensor) -> torch.Tensor:
        if self.disc_reward_type == "bce":
            # MimicKit 방식: -log(1 - sigmoid(logit))
            # logit → +∞ (expert-like): reward → +∞
            # logit → -∞ (fake): reward → 0
            prob = torch.sigmoid(disc_logits)
            reward = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
        elif self.disc_reward_type == "wgan":
            # WASABI(WGAN) reward: (D - μ̂) / σ̂  — 논문 arXiv:2206.11693 Eq 4.
            # reward는 zero-mean·unit-variance이므로 약 절반이 음수 — signed 유지, clamp(min=0) 절대 금지.
            # 주의: parkour 경로의 total_reward.clip(min=0) (A env L1108, parkour_reward_manager.py L38)이
            # 활성인 경우 음수 AMP reward가 통째로 0으로 잘려 학습 신호 절반 소실.
            # 비-parkour Go2-Imitation 경로 우선 검증 권장.
            # EmpiricalNormalization.forward()는 통계를 갱신하지 않으므로 update()를 명시 호출.
            if self.training:
                self.reward_normalizer.update(disc_logits.detach())
            reward = self.reward_normalizer(disc_logits)
        else:
            # LS-GAN reward: clamp(1 - (1/4)*(d-1)^2, min=0) · coef
            reward = torch.clamp(1 - 0.25 * torch.square(disc_logits - 1), min=0)
        return reward.squeeze(-1)

    def compute_amp_reward(self, amp_obs):
        """판별자(Discriminator)를 통해 AMP 보상을 계산합니다.

        조건부 D 이고 ``cond_reward_blend`` > 0 이면 조건을 지운 무조건부 평가와 섞는다:
        ``(1-b)·r(x|c) + b·r(x|∅)``. 정책이 아직 그 명령 속도를 못 낼 때 style 신호가 0 으로 죽는 것을
        완화하는 안전판이다.

        Args:
            amp_obs (torch.Tensor): agent 또는 expert의 모션 관측치. [Batch, amp_observation_size]

        Returns:
            torch.Tensor: 에이전트의 모션이 전문가 모션과 얼마나 유사한지에 대한 스칼라 보상
        """
        amp_obs = amp_obs.to(self.device)
        reward = self._reward_from_logits(self.trunk(self._normalize(amp_obs)))
        if self.cond_dim > 0 and self.cond_reward_blend > 0.0:
            reward_u = self._reward_from_logits(self.trunk(self._normalize(self.drop_condition(amp_obs))))
            b = self.cond_reward_blend
            reward = (1.0 - b) * reward + b * reward_u
        return reward * self.amp_reward_coef

    def get_logits(self, amp_obs):
        amp_obs = amp_obs.to(self.device)
        norm_obs = self._normalize(amp_obs)
        return self.trunk(norm_obs)

    def get_output_layer_weights(self) -> torch.Tensor:
        """MimicKit 방식 logit regularization용 — trunk 마지막 Linear 레이어 가중치 반환."""
        for m in reversed(list(self.trunk.modules())):
            if isinstance(m, nn.Linear):
                return m.weight.flatten()
        raise RuntimeError("Linear layer not found in trunk")
