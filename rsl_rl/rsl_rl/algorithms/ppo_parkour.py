# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent, ActorCriticRMA
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPOParkour:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN | ActorCriticRMA
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN | ActorCriticRMA,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        # Entropy coefficient decay schedule (opt-in, iteration-based).
        # All four must be set together to enable; if any is None, entropy_coef stays
        # fixed at the value above for the whole run (default, backward-compatible).
        # Purpose: prevent late-training action_std runaway when the surrogate gradient
        # weakens and a fixed entropy bonus dominates the PPO objective (see LiDAR-SL
        # iter-8000+ divergence, debug-worker RCA).
        #   it < entropy_decay_start_iter                       -> entropy_coef_start
        #   entropy_decay_start_iter <= it <= entropy_decay_end_iter -> linear interpolation
        #   it > entropy_decay_end_iter                          -> entropy_coef_end
        entropy_coef_start: float | None = None,
        entropy_coef_end: float | None = None,
        entropy_decay_start_iter: int | None = None,
        entropy_decay_end_iter: int | None = None,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Estimator parameters
        estimator: nn.Module | None = None,
        estimator_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # SPO surrogate parameters
        surrogate_type: str = "ppo",
        spo_epsilon: float = 0.2,
        # LCP gradient penalty (arXiv:2410.11825 §3.4) — bounds Lipschitz constant of actor MLP
        # w.r.t. its immediate input (proprio + priv_explicit + priv_latent concat).
        # None disables LCP entirely (backward-compatible default).
        lcp_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Estimator components
        if estimator is not None and estimator_cfg is not None:
            self.estimator = estimator.to(self.device)
            self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_cfg["learning_rate"])
            self.train_with_estimated_states = estimator_cfg["train_with_estimated_states"]
        else:
            self.estimator = None
            self.estimator_optimizer = None
            self.train_with_estimated_states = False

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.policy.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        self.counter = 0
        # Create the optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Entropy coefficient decay schedule (opt-in; see __init__ docstring above).
        # Enabled only when all four schedule params are provided — otherwise entropy_coef
        # stays fixed (default, backward-compatible for parkour/parkour_symmetry/etc.).
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_decay_start_iter = entropy_decay_start_iter
        self.entropy_decay_end_iter = entropy_decay_end_iter
        # Guard first on a local (not self.attr) so pyright narrows int|None -> int inside
        # this block; only then is it safe to compare/subtract entropy_decay_end_iter and
        # entropy_decay_start_iter. If any of the four is None, decay stays disabled and no
        # arithmetic on the None-typed params ever executes (this is the default for every
        # other parkour task — required for the no-regression guarantee).
        if (
            entropy_coef_start is None
            or entropy_coef_end is None
            or entropy_decay_start_iter is None
            or entropy_decay_end_iter is None
        ):
            self._entropy_decay_enabled = False
        else:
            if entropy_decay_end_iter <= entropy_decay_start_iter:
                raise ValueError(
                    "entropy_decay_end_iter must be > entropy_decay_start_iter "
                    f"(got start={entropy_decay_start_iter}, end={entropy_decay_end_iter})."
                )
            self._entropy_decay_enabled = True
        # Current training iteration, set externally by the runner's learn() loop
        # (e.g. `self.alg.current_iteration = it` before calling update()). Stays 0 if the
        # runner never sets it, in which case the decay schedule (if enabled) behaves as if
        # it is always before entropy_decay_start_iter -> entropy_coef_start is used.
        self.current_iteration = 0

        # SPO parameters
        self.surrogate_type = surrogate_type
        self.spo_epsilon = spo_epsilon

        # LCP parameters (arXiv:2410.11825, Eq.7): λ_gp · E[‖∇_input log π(a|input)‖²]
        # None means LCP is disabled (lcp_cfg=None → no-op, backward-compatible).
        if lcp_cfg is not None:
            self.lcp_lambda_gp = lcp_cfg["lambda_gp"]
            self.lcp_penalize = lcp_cfg.get("penalize", "log_prob")
        else:
            self.lcp_lambda_gp = None
            self.lcp_penalize = "log_prob"

    def act(self, obs: TensorDict, hist_encoding: bool = False) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions — optionally replace priv_explicit with estimator output
        if self.train_with_estimated_states and self.estimator is not None:
            obs_est = obs.clone()
            obs_est["priv_explicit"] = self.estimator(obs["policy"])
            self.transition.actions = self.policy.act(obs_est, hist_encoding=hist_encoding).detach()
        else:
            self.transition.actions = self.policy.act(obs, hist_encoding=hist_encoding).detach()
        # Critic always uses original obs (ground-truth priv_explicit)
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record original observations before env.step() — preserves ground-truth priv_explicit in storage
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def _current_entropy_coef(self) -> float:
        """Return entropy_coef for the current iteration, applying the decay schedule if enabled.

        Disabled (any schedule param is None) -> returns the fixed self.entropy_coef unchanged
        (exact prior behavior). Enabled -> piecewise: hold / linear-decay / hold, keyed off
        self.current_iteration (set by the runner before update()).
        """
        # Read into locals first so pyright narrows int|None -> int/float for this whole
        # function body. The None-check on these four locals (not on the cached
        # self._entropy_decay_enabled bool) is what makes the narrowing valid: pyright cannot
        # infer that self._entropy_decay_enabled implies these self.attrs are non-None, but it
        # can infer that from a direct `x is None` check on the local itself.
        start = self.entropy_coef_start
        end = self.entropy_coef_end
        decay_start_iter = self.entropy_decay_start_iter
        decay_end_iter = self.entropy_decay_end_iter
        if start is None or end is None or decay_start_iter is None or decay_end_iter is None:
            # Disabled (default for every task except the LiDAR-SL cfg) -> fixed entropy_coef,
            # no arithmetic on the None-typed schedule params ever executes.
            return self.entropy_coef
        it = self.current_iteration
        if it <= decay_start_iter:
            return start
        if it >= decay_end_iter:
            return end
        frac = (it - decay_start_iter) / (decay_end_iter - decay_start_iter)
        return start + frac * (end - start)

    def update(self) -> dict[str, float]:
        # Entropy coefficient for this update() call — fixed for the whole rollout update
        # (self.current_iteration does not change mid-update). Decay disabled -> self.entropy_coef.
        entropy_coef_current = self._current_entropy_coef()

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Estimator loss
        mean_estimator_loss = 0 if self.estimator else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # LCP gradient penalty loss
        mean_lcp_loss = 0 if self.lcp_lambda_gp is not None else None
        # Adaptation Reg loss
        mean_priv_reg_loss = 0

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Adaptation module update
            priv_latent_batch = self.policy.get_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.policy.get_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(
                max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1
            )
            priv_reg_coef = (
                priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0])
                + self.priv_reg_coef_schedual[0]
            )

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            adv = torch.squeeze(advantages_batch)

            if self.surrogate_type == "spo":
                # Simple Policy Optimization (arXiv:2401.16025), Eq.16:
                #   f_spo = r·A − (|A| / 2ε)·(r − 1)^2      (maximize)
                surrogate_obj = ratio * adv - torch.abs(adv) * torch.square(ratio - 1.0) / (2.0 * self.spo_epsilon)
                surrogate_loss = -surrogate_obj.mean()  # maximize → minimize negative
            else:  # "ppo" — original ratio-clip (default, backward-compatible)
                surrogate = -adv * ratio
                surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - entropy_coef_current * entropy_batch.mean()
                + priv_reg_coef * priv_reg_loss
            )

            # LCP gradient penalty: λ_gp · E[‖∇_input log π(a|input)‖²]  (arXiv:2410.11825, Eq.7)
            # Gradient target: the *immediate actor MLP input* = [norm_proprio, norm_priv_explicit,
            # priv_latent, (scandot_latent)].  This bounds the Lipschitz constant of the actor MLP
            # w.r.t. its own concat input.  Encoders (priv_encoder, scandot_encoder) are excluded
            # from the gradient path because actor_input_leaf is detached from them — intentional
            # simplification per spec scope caveat (encoders are separately trained; LCP focuses on
            # the deploy-time MLP, which is the smoothness bottleneck).
            # update_dagger / hist_encoder_optimizer: separate method + separate optimizer → no contamination.
            if self.lcp_lambda_gp is not None:
                # Build post-norm / post-concat actor MLP input with encoders in no_grad, then
                # detach to make it a clean leaf before re-enabling grad on the *concat* tensor.
                with torch.no_grad():
                    actor_input_raw = self.policy.get_actor_input(obs_batch[:original_batch_size])
                obs_in = actor_input_raw.detach().requires_grad_(True)

                if self.lcp_penalize == "mean":
                    # Cheaper variant: penalize ‖∇_input μ(input)‖² (no actions needed).
                    # grad(μ.sum()) approximates the Frobenius norm of the actor Jacobian; the
                    # default "log_prob" mode is paper-faithful (arXiv:2410.11825 Eq.7).
                    if self.policy.state_dependent_std:
                        mean_from_leaf = self.policy.actor(obs_in)[..., 0, :]
                    else:
                        mean_from_leaf = self.policy.actor(obs_in)
                    grads = torch.autograd.grad(
                        outputs=mean_from_leaf.sum(),
                        inputs=obs_in,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                else:  # "log_prob" — paper-faithful (default)
                    actions_for_gp = actions_batch[:original_batch_size].detach()
                    logp = self.policy.log_prob_from_actor_input(obs_in, actions_for_gp)
                    grads = torch.autograd.grad(
                        outputs=logp.sum(),
                        inputs=obs_in,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]

                lcp_loss = grads.norm(2, dim=-1).pow(2).mean()
                loss = loss + self.lcp_lambda_gp * lcp_loss

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Estimator loss (separate optimizer — gradient isolated from PPO/RND)
            if self.estimator is not None:
                obs_orig = obs_batch[:original_batch_size]
                priv_explicit_pred = self.estimator(obs_orig["policy"])
                estimator_loss = (priv_explicit_pred - obs_orig["priv_explicit"]).pow(2).mean()
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Estimator loss
            if mean_estimator_loss is not None:
                mean_estimator_loss += estimator_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # LCP loss
            if mean_lcp_loss is not None:
                mean_lcp_loss += lcp_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_estimator_loss is not None:
            mean_estimator_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_lcp_loss is not None:
            mean_lcp_loss /= num_updates

        mean_priv_reg_loss /= num_updates

        # NOTE: storage.clear() and update_counter() are deferred to update_dagger()
        # which runs after this method, so the dagger generator can iterate over the
        # rollout one more time before storage is cleared.
        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "priv_reg_loss": mean_priv_reg_loss,
        }
        # Log the entropy coefficient actually applied this update — always emitted so the
        # decay schedule (when enabled) is directly observable in TensorBoard/WandB, and so
        # the fixed value is visible too when disabled (sanity check for other tasks).
        loss_dict["entropy_coef"] = entropy_coef_current
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.estimator:
            loss_dict["estimator"] = mean_estimator_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.lcp_lambda_gp is not None:
            loss_dict["lipschitz"] = mean_lcp_loss

        # --- F9: per-joint action statistics + policy noise std ---
        # Compute over the full rollout buffer (storage.actions shape: [num_steps, num_envs, num_actions]).
        # All reductions run inside no_grad; .item() prevents any graph from persisting.
        with torch.no_grad():
            actions = self.storage.actions  # [T, N, 12]
            num_joints = actions.shape[-1]
            for j in range(num_joints):
                a_j = actions[..., j]  # [T, N]
                loss_dict[f"action_stats/joint_{j:02d}/mean"] = a_j.mean().item()
                loss_dict[f"action_stats/joint_{j:02d}/abs_max"] = a_j.abs().amax().item()
                loss_dict[f"action_stats/joint_{j:02d}/sample_std"] = a_j.std().item()
            # Policy noise std (learnable parameter) — skip gracefully if architecture
            # uses state-dependent std (no self.std attribute on policy).
            if hasattr(self.policy, "std"):
                for j in range(num_joints):
                    loss_dict[f"policy_std/joint_{j:02d}"] = self.policy.std[j].item()
            elif hasattr(self.policy, "log_std"):
                for j in range(num_joints):
                    loss_dict[f"policy_std/joint_{j:02d}"] = self.policy.log_std[j].exp().item()

        return loss_dict

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            with torch.inference_mode():
                self.policy.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_state=hidden_states_batch[0])

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.policy.get_priv_latent(obs_batch)
            hist_latent_batch = self.policy.get_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()

            mean_hist_latent_loss += hist_latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel

    def update_counter(self):
        self.counter += 1
