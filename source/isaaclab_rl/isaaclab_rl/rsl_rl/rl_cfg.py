# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#####################
# LCP configuration #
#####################


@configclass
class RslRlLcpCfg:
    """Configuration for the Lipschitz-Constrained Policy (LCP) gradient penalty.

    Penalizes the L2-squared norm of the gradient of log π(a|obs) w.r.t. the observation,
    encouraging a Lipschitz-smooth policy for sim-to-real transfer.

    Reference: Chen et al., arXiv:2410.11825 — *Learning Smooth Humanoid Locomotion through
    Lipschitz-Constrained Policies* (X.B. Peng group), Eq. 7.
    """

    lambda_gp: float = 0.002
    """Gradient penalty coefficient λ_gp. Paper default is 0.002.

    Ablation grid from the paper: {0.0, 0.001, 0.002, 0.005, 0.01}.
    Larger values enforce stronger Lipschitz constraint but may reduce task performance.
    """

    penalize: str = "log_prob"
    """Which quantity to penalize the gradient of. Either 'log_prob' or 'mean'.

    - 'log_prob': paper-faithful — penalizes ‖∇_obs log π(a|obs)‖² (Eq. 7).
    - 'mean': cheaper variant — penalizes ‖∇_obs μ(obs)‖², directly bounds the
      Lipschitz constant of the deterministic policy map. No actions_batch needed.
    """


#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlPpoActorCriticMoECfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic network with a Mixture-of-Experts (MoE) actor.

    The actor is replaced by a dense-softmax MoE: N expert MLPs are each applied to the
    observation, and their outputs are weighted-summed by a learned gating network.
    No load-balancing auxiliary loss is needed (dense routing — all experts receive
    gradients every step). PPO loop is unmodified.

    Reference: Huang*, Zhu* et al., arXiv:2503.08564 — *MoE-Loco: Mixture of Experts
    for Multitask Locomotion*.
    """

    class_name: str = "ActorCriticMoE"
    """The policy class name. Must match the registered class in modules/__init__.py."""

    num_experts: int = 6
    """Number of expert MLPs in the MoE actor. Paper default is 6."""

    gating_hidden_dims: list[int] = [128]
    """Hidden dimensions of the gating network MLP. Paper uses [128] (Table VI)."""

    expert_hidden_dims: list[int] = MISSING
    """Hidden dimensions of each expert MLP. Replaces actor_hidden_dims for the MoE actor.

    Must be set explicitly. Example: [256, 256, 256].
    Note: actor_hidden_dims (inherited) is accepted by super().__init__ but is superseded
    by this field — ActorCriticMoE replaces self.actor with the MoE module after super init.
    """

    gating_temperature: float = 1.0
    """Temperature for the gating softmax. Values > 1.0 flatten the gating distribution
    (encourage expert diversity); values < 1.0 sharpen it (encourage specialization).
    Start at 1.0; tune if expert collapse is observed.
    """


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    surrogate_type: str = "ppo"
    """Surrogate objective type. Either 'ppo' (ratio-clip, default) or 'spo' (quadratic ratio penalty).

    SPO replaces PPO's clipped surrogate with a quadratic penalty:
        f_spo = r·A − (|A| / 2ε)·(r − 1)²
    where ε is :attr:`spo_epsilon`. Reference: Xie et al., arXiv:2401.16025, Eq. 16.
    Default 'ppo' preserves existing PPO behaviour exactly.
    """

    spo_epsilon: float = 0.2
    """Trust-region coefficient ε for SPO quadratic penalty. Only used when surrogate_type='spo'.

    Same role and scale as PPO's clip_param (≈ 0.2 recommended starting point).
    f_spo = r·A − (|A| / 2ε)·(r − 1)²  (arXiv:2401.16025, Eq. 16 / Eq. 18).
    Must be > 0; spo_epsilon → 0 causes the penalty term to blow up.
    """

    lcp_cfg: RslRlLcpCfg | None = None
    """The Lipschitz-Constrained Policy (LCP) gradient penalty configuration.

    Default is None, in which case LCP is disabled. When set, the gradient penalty
    λ_gp·E[‖∇_obs log π(a|obs)‖²] is added to the PPO loss to smooth the policy.
    Reference: Chen et al., arXiv:2410.11825, Eq. 7.
    """


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool | None = None
    """This parameter is deprecated and will be removed in the future.

    Use `actor_obs_normalization` and `critic_obs_normalization` instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "policy": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the policy will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Default is False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
