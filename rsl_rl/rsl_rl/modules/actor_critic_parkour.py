# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import get_param, resolve_nn_activation

class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), resolve_nn_activation(self.activation_fn),
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), resolve_nn_activation(self.activation_fn),
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), resolve_nn_activation(self.activation_fn),
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), resolve_nn_activation(self.activation_fn), nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), resolve_nn_activation(self.activation_fn),
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), resolve_nn_activation(self.activation_fn),
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), resolve_nn_activation(self.activation_fn),
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), resolve_nn_activation(self.activation_fn),
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), resolve_nn_activation(self.activation_fn)
                )
    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class ActorCriticRMA(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        scan_encoder_dims: tuple[int] | list[int] = [128, 64, 32],
        priv_encoder_dims: tuple[int] | list[int] = [64, 20],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        num_scan_obs = 0
        # for obs_group in obs_groups["scan"]:
        #     assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
        #     num_scan_obs += obs[obs_group].shape[-1]
        num_history = 0
        for obs_group in obs_groups["history"]:
            assert len(obs[obs_group].shape) == 3, "The ActorCritic module only supports 2D observations."
            num_history += obs[obs_group].shape[-2]
        num_priv_obs = 0
        for obs_group in obs_groups["priv"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_priv_obs += obs[obs_group].shape[-1]


        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_scan_obs + num_actor_obs + priv_encoder_dims[-1], [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_scan_obs + num_actor_obs + priv_encoder_dims[-1], num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.scandot_encoder = None
        # Scandot Encoder
        if num_scan_obs > 0:
            self.scandot_encoder = MLP(num_scan_obs, scan_encoder_dims[-1], scan_encoder_dims, activation)
            # Scandot observation normalization
            self.scan_obs_normalization = actor_obs_normalization
            if actor_obs_normalization:
                self.scan_obs_normalizer = EmpiricalNormalization(num_scan_obs)
            else:
                self.scan_obs_normalizer = torch.nn.Identity()

        # Priv Encoder (Adaptation Module)
        self.priv_encoder = MLP(num_priv_obs, priv_encoder_dims[-1], priv_encoder_dims, activation)
        # Priv observation normalization
        self.priv_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.priv_obs_normalizer = EmpiricalNormalization(num_priv_obs)
        else:
            self.priv_obs_normalizer = torch.nn.Identity()

        # History Encoder
        self.history_encoder = StateHistoryEncoder(activation, num_actor_obs, num_history, priv_encoder_dims[-1])
        # History observation normalization
        self.history_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.history_obs_normalizer = EmpiricalNormalization(num_actor_obs * num_history)
        else:
            self.history_obs_normalizer = torch.nn.Identity()


        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)

            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)

            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, hist_encoding=False, **kwargs: dict[str, Any]) -> torch.Tensor:
        # obs = self.get_actor_obs(obs)
        obs_actor = self.get_actor_obs(obs)
        obs_actor = self.actor_obs_normalizer(obs_actor)
        if hist_encoding:
            history_latent = self.get_hist_latent(obs)
            obs_actor = torch.cat([obs_actor, history_latent], dim=-1)
        else:
            priv_latent = self.get_priv_latent(obs)
            obs_actor = torch.cat([obs_actor, priv_latent], dim=-1)

        if self.scandot_encoder is not None:
            obs_scan = self.get_scan_obs(obs)
            scandot_latent = self.scandot_encoder(obs_scan)
            obs_actor = torch.cat([obs_actor, scandot_latent], dim=-1)

        self._update_distribution(obs_actor)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs_actor = self.get_actor_obs(obs)
        obs_actor = self.actor_obs_normalizer(obs_actor)
        history_latent = self.get_hist_latent(obs)
        obs_actor = torch.cat([obs_actor, history_latent], dim=-1)

        if self.scandot_encoder is not None:
            obs_scan = self.get_scan_obs(obs)
            scandot_latent = self.scandot_encoder(obs_scan)
            obs_actor = torch.cat([obs_actor, scandot_latent], dim=-1)

        if self.state_dependent_std:
            return self.actor(obs_actor)[..., 0, :]
        else:
            return self.actor(obs_actor)
    
    def get_hist_latent(self, obs: TensorDict) -> torch.Tensor:
        obs_history = self.get_history_obs(obs)
        obs_history_flat = obs_history.reshape(obs_history.shape[0], -1)
        return self.history_encoder(self.history_obs_normalizer(obs_history_flat))

    def get_priv_latent(self, obs: TensorDict) -> torch.Tensor:
        obs_priv = self.get_priv_obs(obs)
        return self.priv_encoder(self.priv_obs_normalizer(obs_priv))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_scan_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["scan"]]
        return torch.cat(obs_list, dim=-1)

    def get_history_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["history"]]
        return torch.cat(obs_list, dim=-1)

    def get_priv_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["priv"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
            history_obs = self.get_history_obs(obs)
            self.history_obs_normalizer.update(history_obs.reshape(history_obs.shape[0], -1))
            priv_obs = self.get_priv_obs(obs)
            self.priv_obs_normalizer.update(priv_obs)
            if self.scandot_encoder is not None:
                scan_obs = self.get_scan_obs(obs)
                self.scan_obs_normalizer.update(scan_obs)

        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
