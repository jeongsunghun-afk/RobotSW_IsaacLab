# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DRAIL 확산 discriminator (LaCoLoco / NVlabs DRAIL 이식).

판별을 분류기 대신 **라벨 조건 확산 노이즈 예측 오차**로 한다.

.. code-block:: text

    x_t     = sqrt(a_t) * x + sqrt(1 - a_t) * eps            (x = 정규화한 kinematics)
    L(x, l) = | eps - eps_hat(x_t, [label(l), cond], t) |^2   (dim 평균)
    logit   = L(x, l_pi) - L(x, l_M)                          (l_M = 1.0 expert, l_pi = 0.0 policy)
    D       = sigmoid(logit)

``get_logits`` 가 이 logit 을 돌려주므로 기존 BCE 손실(``BCEWithLogitsLoss``)과 ``bce`` 보상
(``-log(1 - sigmoid(logit))``)을 **그대로** 쓴다. LaCoLoco 원본은 gradient penalty·logit reg 를 걸지
않으므로 알고리즘 쪽에서 ``disc_arch="drail"`` 이면 둘을 건너뛴다.

원본과 다른 점 하나: 두 라벨 평가에 **같은** ``t``·``eps`` 를 쓴다(paired). 원본은 라벨마다 따로 뽑는데,
차이만 쓰는 logit 에서는 공통 잡음을 쓰는 쪽이 분산이 작다. ``paired_noise=False`` 로 원본 동작.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (arXiv:2102.09672) cosine β 스케줄."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class _CondDiffusionMLP(nn.Module):
    """노이즈 예측기. 각 은닉층 출력에 step embedding 을 더한다 (LaCoLoco ``MLPConditionDiffusion``)."""

    def __init__(self, x_dim: int, cond_dim: int, hidden_dims: list[int], activation: str, diffusion_steps: int):
        super().__init__()
        act = resolve_nn_activation(activation)
        self._layers = nn.ModuleList()
        in_size = x_dim + cond_dim
        for unit in hidden_dims:
            self._layers.append(nn.Linear(in_size, unit))
            in_size = unit
        self._out = nn.Linear(in_size, x_dim)
        self._act = act
        self._step_embeddings = nn.ModuleList([nn.Embedding(diffusion_steps, unit) for unit in hidden_dims])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_t, cond], dim=-1)
        for layer, emb in zip(self._layers, self._step_embeddings):
            h = layer(h) + emb(t)
            h = self._act(h)
        return self._out(h)


class AMPDiffusionDiscriminator(nn.Module):
    """DRAIL 확산 discriminator. :class:`AMPDiscriminator` 와 같은 인터페이스를 낸다.

    입력 레이아웃 ``[kinematics (kin_dim) | condition (cond_dim)]``. 확산 노이즈는 kinematics 에만
    섞고, 조건 열은 라벨 벡터와 함께 깨끗한 조건 입력으로 들어간다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=[256, 256, 256, 256],
        activation: str = "elu",
        device: str = "cuda",
        disc_reward_type: str = "bce",
        norm_clip: float | None = None,
        cond_dim: int = 0,
        label_dim: int = 10,
        diffusion_steps: int = 1000,
        sample_strategy: str = "antithetic",  # "antithetic" | "uniform" | "constant"
        sample_strategy_value: int = 0,
        paired_noise: bool = True,
    ):
        super().__init__()
        assert disc_reward_type == "bce", "DRAIL 은 확률 기반이라 disc_reward_type='bce' 만 지원한다"
        self.device = device
        self.input_dim = input_dim
        self.cond_dim = int(cond_dim)
        self.kin_dim = input_dim - self.cond_dim
        assert self.kin_dim > 0, f"cond_dim({cond_dim}) 이 input_dim({input_dim}) 보다 작아야 한다"
        self.disc_reward_type = disc_reward_type
        self.norm_clip = norm_clip
        self.label_dim = int(label_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.sample_strategy = sample_strategy
        self.sample_strategy_value = int(sample_strategy_value)
        self.paired_noise = paired_noise

        self.amp_reward_coef = 1.5
        self.cond_reward_blend = 0.0

        self.amp_obs_normalizer = EmpiricalNormalization(self.kin_dim).to(self.device)
        self.model = _CondDiffusionMLP(
            self.kin_dim, self.label_dim + self.cond_dim, list(hidden_dims), activation, self.diffusion_steps
        ).to(self.device)

        betas = cosine_beta_schedule(self.diffusion_steps)
        alphas_prod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_bar_sqrt", torch.sqrt(alphas_prod).to(self.device))
        self.register_buffer("one_minus_alphas_bar_sqrt", torch.sqrt(1.0 - alphas_prod).to(self.device))

    # ── 조건/정규화 헬퍼 (AMPDiscriminator 와 동일 계약) ─────────
    def drop_condition(self, amp_obs: torch.Tensor) -> torch.Tensor:
        if self.cond_dim == 0:
            return amp_obs
        out = amp_obs.clone()
        out[:, self.kin_dim :] = 0.0
        return out

    def _split(self, amp_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.amp_obs_normalizer(amp_obs[:, : self.kin_dim])
        if self.norm_clip is not None:
            x = torch.clamp(x, -self.norm_clip, self.norm_clip)
        cond = amp_obs[:, self.kin_dim :]
        return x, cond

    def update_normalization(self, amp_obs: torch.Tensor) -> None:
        self.amp_obs_normalizer.update(amp_obs[:, : self.kin_dim])

    # ── 확산 손실 ──────────────────────────────────────────────
    def _sample_t(self, batch_size: int) -> torch.Tensor:
        T = self.diffusion_steps
        if self.sample_strategy == "constant":
            step = min(self.sample_strategy_value, T - 1)
            return torch.full((batch_size,), step, device=self.device, dtype=torch.long)
        if self.sample_strategy == "uniform":
            return torch.randint(0, T, (batch_size,), device=self.device)
        # antithetic: (t, T-1-t) 쌍 — 원본 DRAIL 의 기본
        half = (batch_size + 1) // 2
        t = torch.randint(0, T, (half,), device=self.device)
        return torch.cat([t, T - 1 - t], dim=0)[:batch_size]

    def _diffusion_loss(
        self, x: torch.Tensor, cond: torch.Tensor, label: float, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        a = self.alphas_bar_sqrt[t].unsqueeze(-1)
        aml = self.one_minus_alphas_bar_sqrt[t].unsqueeze(-1)
        x_t = x * a + eps * aml
        label_input = torch.full((x.shape[0], self.label_dim), float(label), device=self.device)
        out = self.model(x_t, torch.cat([label_input, cond], dim=-1), t)
        return (eps - out).square().mean(dim=1, keepdim=True)

    def get_logits(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """``logit = L(x, l_pi) − L(x, l_M)``. ``sigmoid(logit)`` 이 expert 확률 D 이다."""
        amp_obs = amp_obs.to(self.device)
        x, cond = self._split(amp_obs)
        t = self._sample_t(x.shape[0])
        eps = torch.randn_like(x)
        loss_expert_label = self._diffusion_loss(x, cond, 1.0, t, eps)
        if not self.paired_noise:
            t = self._sample_t(x.shape[0])
            eps = torch.randn_like(x)
        loss_policy_label = self._diffusion_loss(x, cond, 0.0, t, eps)
        return loss_policy_label - loss_expert_label

    def _reward_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        return (-torch.log(torch.clamp(1.0 - prob, min=1e-4))).squeeze(-1)

    def compute_amp_reward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        amp_obs = amp_obs.to(self.device)
        reward = self._reward_from_logits(self.get_logits(amp_obs))
        if self.cond_dim > 0 and self.cond_reward_blend > 0.0:
            reward_u = self._reward_from_logits(self.get_logits(self.drop_condition(amp_obs)))
            b = self.cond_reward_blend
            reward = (1.0 - b) * reward + b * reward_u
        return reward * self.amp_reward_coef

    def get_output_layer_weights(self) -> torch.Tensor:
        """DRAIL 에는 logit 출력층이 없다. 알고리즘은 ``disc_arch="drail"`` 이면 logit reg 를 건너뛴다."""
        return self.model._out.weight.flatten()
