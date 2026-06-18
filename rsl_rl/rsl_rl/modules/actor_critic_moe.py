# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixture-of-Experts actor-critic module (MoE-Loco, arXiv:2503.08564).

Design:
- Dense softmax gating (all experts participate every forward pass).
- No load-balancing aux loss required — every expert receives gradient every step.
- PPO loop is unchanged; the only difference vs ActorCritic is self.actor.
- critic remains a single MLP (actor-only MoE for 1st deployment).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.networks import MLP

from .actor_critic import ActorCritic


class MoEActor(nn.Module):
    """Dense-softmax Mixture-of-Experts actor head.

    Forward signature: obs [B, num_obs] -> mean [B, num_actions].
    This is a drop-in replacement for the single MLP that ActorCritic puts in
    self.actor. The parent's _update_distribution / act / act_inference all call
    self.actor(obs) expecting exactly this signature.

    Monitoring:
        last_gate_weights: stored on every forward as a plain attribute (detached).
            NOT registered as a buffer so it does not enter state_dict.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        num_experts: int,
        gating_hidden_dims: list[int],
        expert_hidden_dims: list[int],
        activation: str,
        gating_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.gating_temperature = gating_temperature

        # Gating network: obs -> logits [B, num_experts]
        self.gating = MLP(num_obs, num_experts, gating_hidden_dims, activation)

        # Expert networks: each maps obs -> action_mean [B, num_actions]
        self.experts = nn.ModuleList(
            [MLP(num_obs, num_actions, expert_hidden_dims, activation) for _ in range(num_experts)]
        )

        # Monitoring buffer — plain attribute, NOT in state_dict
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute weighted mixture of expert outputs.

        Args:
            obs: [B, num_obs]

        Returns:
            mean: [B, num_actions]
        """
        # Gating weights
        logits = self.gating(obs) / self.gating_temperature  # [B, N]
        w = F.softmax(logits, dim=-1)  # [B, N]  dense: all experts receive gradient

        # Detach for monitoring only — never enters loss
        self.last_gate_weights = w.detach()

        # Expert outputs stacked: [B, N, num_actions]
        outs = torch.stack([expert(obs) for expert in self.experts], dim=1)

        # Weighted sum: [B, num_actions]
        mean = (w.unsqueeze(-1) * outs).sum(dim=1)
        return mean


class ActorCriticMoE(ActorCritic):
    """ActorCritic with a dense softmax MoE actor.

    Only self.actor is replaced with MoEActor. Everything else — std parameters,
    normalizers, critic, all properties (action_mean / action_std / entropy),
    act / act_inference / _update_distribution / evaluate / get_actions_log_prob /
    update_normalization / reset — is inherited unchanged from ActorCritic.

    PPO loop remains entirely unmodified: no auxiliary loss, no new return values.

    Args:
        obs: TensorDict of observations (same as ActorCritic).
        obs_groups: Group-to-key mapping (same as ActorCritic).
        num_actions: Number of action dimensions.
        num_experts: Number of expert MLPs (default 6, per MoE-Loco paper).
        gating_hidden_dims: Hidden dims of the gating network (default [128]).
        expert_hidden_dims: Hidden dims of each expert MLP. Replaces the role of
            actor_hidden_dims from the parent. Must be provided explicitly.
        gating_temperature: Softmax temperature for gating (default 1.0).
            Increasing this during early training flattens the gate distribution
            and reduces expert collapse risk.
        **kwargs: Forwarded to ActorCritic (e.g. activation, init_noise_std,
            critic_hidden_dims, *_obs_normalization). actor_hidden_dims is stripped
            before forwarding because it applies to the throwaway parent actor only.

    Note on state_dependent_std:
        When state_dependent_std=True each expert must return [B, 2, num_actions].
        This is not supported in the first implementation — a ValueError is raised
        so the failure is explicit rather than producing silently wrong std tensors.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_experts: int = 6,
        gating_hidden_dims: list[int] | None = None,
        expert_hidden_dims: list[int] | None = None,
        gating_temperature: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if gating_hidden_dims is None:
            gating_hidden_dims = [128]
        if expert_hidden_dims is None:
            expert_hidden_dims = [256, 256, 256]

        # state_dependent_std is not supported in this implementation.
        if kwargs.get("state_dependent_std", False):
            raise NotImplementedError(
                "ActorCriticMoE does not support state_dependent_std=True in this implementation. "
                "Use state_dependent_std=False (default) or implement per-expert std init separately."
            )

        # Extract activation before super() so we can pass it to MoEActor.
        # ActorCritic does NOT store it as self._activation; we must capture here.
        activation: str = kwargs.get("activation", "elu")

        # Remove actor_hidden_dims from kwargs to avoid "multiple values" TypeError.
        # The parent will build a throwaway actor with whatever dims we give it;
        # we immediately overwrite self.actor with MoEActor below.
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "actor_hidden_dims"}

        # Initialise parent: std, normalizers, critic, distribution, reset.
        # actor_hidden_dims is not passed (throwaway); expert_hidden_dims is MoE-only.
        super().__init__(obs, obs_groups, num_actions, **kwargs_clean)

        # Re-compute num_actor_obs exactly as the parent does (L48-51 actor_critic.py).
        # The parent does NOT store this, so we recompute here.
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            num_actor_obs += obs[obs_group].shape[-1]

        # Replace the parent's single-MLP actor with the MoE actor.
        self.actor = MoEActor(
            num_obs=num_actor_obs,
            num_actions=num_actions,
            num_experts=num_experts,
            gating_hidden_dims=list(gating_hidden_dims),
            expert_hidden_dims=list(expert_hidden_dims),
            activation=activation,
            gating_temperature=gating_temperature,
        )
        print(f"MoEActor: {num_experts} experts, gating_hidden={gating_hidden_dims}, "
              f"expert_hidden={expert_hidden_dims}, temperature={gating_temperature}")

    # No method overrides — all of act / act_inference / _update_distribution /
    # evaluate / get_actions_log_prob / action_mean / action_std / entropy /
    # update_normalization / reset / load_state_dict are inherited from ActorCritic.
    #
    # The invariant that makes this work:
    #   self.actor(obs) returns [B, num_actions] for any obs [B, num_actor_obs].
    #   MoEActor satisfies this contract identically to the parent's MLP.
