# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCriticRMAMoE — RMA actor-critic with a dense-softmax MoE actor head.

Design:
- Inherits ActorCriticRMA: encoder / adaptation / history encoder / critic
  are all initialised by super().__init__() and remain unchanged.
- Only self.actor is replaced with MoEActor (imported from actor_critic_moe).
- Gating input is the full RMA actor input (proprio + priv_explicit + priv/hist
  latent + scan latent), which corresponds to ĥ_t in the MoE-Loco paper.
- dense softmax gating — no load-balancing aux loss, no PPO changes.

Binding contract (cfg worker must match these names exactly):
    num_experts          int   default 6
    gating_hidden_dims   list  default [128]
    expert_hidden_dims   list  default [512, 256, 128]
    gating_temperature   float default 1.0
"""

from __future__ import annotations

from tensordict import TensorDict
from typing import Any

from .actor_critic_moe import MoEActor
from .actor_critic_parkour import ActorCriticRMA


class ActorCriticRMAMoE(ActorCriticRMA):
    """ActorCriticRMA with the single-MLP actor replaced by a dense-softmax MoE actor.

    All RMA components (scan encoder, priv encoder, history encoder, critic,
    normalizers, distribution, adaptation/dagger) are inherited unchanged from
    ActorCriticRMA.  Only self.actor is swapped for a MoEActor instance.

    The actor input fed to self.actor (and therefore to every expert and the
    gating network) is the same concatenated vector that ActorCriticRMA builds:

        [actor_obs_norm, priv_explicit_norm, priv_or_hist_latent, scan_latent?]

    Its dimension is derived from the parent's throwaway actor (self.actor[0].in_features)
    so that any change to the parent formula is picked up automatically.

    Args:
        obs: TensorDict of observations (same as ActorCriticRMA).
        obs_groups: Group-to-key mapping (same as ActorCriticRMA).
        num_actions: Number of action dimensions.
        num_experts: Number of expert MLPs.  Default 6 (MoE-Loco paper).
        gating_hidden_dims: Hidden dims of the gating MLP.  Default [128].
        expert_hidden_dims: Hidden dims of each expert MLP.
            Replaces the role of actor_hidden_dims for the MoE actor.
            Default [512, 256, 128].
        gating_temperature: Softmax temperature for gating.  Default 1.0.
            Increasing this early in training flattens the gate distribution
            and reduces expert-collapse risk.
        **kwargs: Forwarded verbatim to ActorCriticRMA (activation,
            init_noise_std, noise_std_type, actor_hidden_dims,
            critic_hidden_dims, scan_encoder_dims, priv_encoder_dims,
            actor_obs_normalization, critic_obs_normalization, …).
            actor_hidden_dims still reaches the parent and governs the
            throwaway actor; the MoE expert dims are set by expert_hidden_dims.

    Note on state_dependent_std:
        Not supported.  A ValueError is raised before super().__init__() so
        the failure is immediate and explicit rather than producing silently
        wrong std tensors.  ActorCriticRMA's _update_distribution calls
        self.actor(obs)[..., 0, :] when state_dependent_std=True, which would
        index a non-existent dimension on MoEActor's [B, num_actions] output.
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
        **kwargs: Any,
    ) -> None:
        # Apply defaults here so they appear in __repr__ / print
        if gating_hidden_dims is None:
            gating_hidden_dims = [128]
        if expert_hidden_dims is None:
            expert_hidden_dims = [512, 256, 128]

        # Reject state_dependent_std before super() so we never build the
        # throwaway actor with the wrong output shape and confuse the init path.
        if kwargs.get("state_dependent_std"):
            raise NotImplementedError(
                "ActorCriticRMAMoE does not support state_dependent_std=True. "
                "MoEActor returns [B, num_actions]; the state_dependent_std path in "
                "ActorCriticRMA._update_distribution expects [B, 2, num_actions] from self.actor, "
                "which is incompatible."
            )

        # Extract activation string before super() so we can pass it to MoEActor.
        # ActorCriticRMA does NOT store it as self._activation.
        activation: str = kwargs.get("activation", "elu")

        # Initialise the full parent: encoder / adaptation / history encoder /
        # critic / normalizers / std parameter / distribution.
        # The parent also builds a throwaway single-MLP actor with actor_hidden_dims.
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        # Derive actor input dimension from the parent's throwaway actor.
        # MLP is nn.Sequential; [0] is the first nn.Linear.
        # This equals: num_actor_obs + num_priv_explicit + priv_encoder_dims[-1]
        #              + (scan_encoder_dims[-1] if scan present else 0)
        actor_input_dim: int = self.actor[0].in_features

        # Replace the throwaway actor with the MoE actor.
        # After this line self.actor is an MoEActor instance.
        self.actor = MoEActor(
            num_obs=actor_input_dim,
            num_actions=num_actions,
            num_experts=num_experts,
            gating_hidden_dims=list(gating_hidden_dims),
            expert_hidden_dims=list(expert_hidden_dims),
            activation=activation,
            gating_temperature=gating_temperature,
        )

        print(
            f"ActorCriticRMAMoE: actor_input_dim={actor_input_dim}, "
            f"num_experts={num_experts}, "
            f"gating_hidden={gating_hidden_dims}, "
            f"expert_hidden={expert_hidden_dims}, "
            f"temperature={gating_temperature}"
        )

    # -------------------------------------------------------------------------
    # No method overrides.
    #
    # act / act_inference / _update_distribution call self.actor(obs_actor)
    # where obs_actor is the concatenated [B, actor_input_dim] vector built by
    # the parent.  MoEActor satisfies the same forward contract:
    #     obs [B, actor_input_dim] -> mean [B, num_actions]
    #
    # evaluate / get_actor_obs / get_priv_latent / get_hist_latent /
    # update_normalization / load_state_dict / reset are all inherited.
    #
    # PPOParkour.update() accesses policy.parameters() for the optimiser —
    # MoE params (gating + experts) are part of self.actor which is registered
    # as a sub-module of this nn.Module, so they are included automatically.
    #
    # PPOParkour.update_dagger() accesses policy.history_encoder.parameters()
    # via a separate hist_encoder_optimizer — not touched here.
    # -------------------------------------------------------------------------
