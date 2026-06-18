# RSL_RL Library Codebase Inventory

**Date**: 2026-06-08  
**Project**: IsaacLab rsl_rl library  
**Purpose**: Catalog existing network architectures, RL algorithms, loss functions, and IL techniques to prevent duplicate recommendations in future research.

---

## 1. NETWORK MODULES (rsl_rl/modules/)

### 1.1 Actor-Critic Architectures

| Module | File:Lines | Description | Key Features | Extension Points |
|--------|-----------|-------------|-------------|-----------------|
| **ActorCritic** | `actor_critic.py:22-250` | Basic feed-forward actor-critic | MLP-based, separate actor/critic with observation normalization, Gaussian policy | Add custom activation functions, state-dependent std, different normalizers |
| **ActorCriticRecurrent** | `actor_critic_recurrent.py:23-250` | LSTM/GRU-based actor-critic | Memory module (LSTM/GRU), recurrent policy with hidden state management | Swap RNN type, adjust hidden dims, add additional memory layers |
| **ActorCriticCNN** | `actor_critic_cnn.py:24-300` | CNN + MLP hybrid | Processes 2D observations (images) via CNN, 1D via MLP, separate branches for actor/critic | Modify CNN architecture, kernel/stride configs |
| **ActorCriticRMA** (Parkour) | `actor_critic_parkour.py:81-300` | Privileged training + Runtime Adaptation (RMA) | History encoder (Conv1D), privilege encoder (scan), adaptation module with history matching | History encoder design, privilege encoding method, latent dimension tuning |
| **ActorCriticRMA** (Original) | `actor_critic_parkour_original.py:81-400` | Original RMA variant | Similar to above but different implementation details | Same as above |

### 1.2 Student-Teacher Architectures (Distillation)

| Module | File:Lines | Description | Key Features | Extension Points |
|--------|-----------|-------------|-------------|-----------------|
| **StudentTeacher** | `student_teacher.py:22-210` | Non-recurrent student-teacher | Student (limited obs), Teacher (privileged obs), behavior cloning distillation | Separate student/teacher hidden dims, observation groups |
| **StudentTeacherRecurrent** | `student_teacher_recurrent.py:23-300` | Recurrent student-teacher | Student + Teacher both support LSTM/GRU, teacher can be recurrent or non-recurrent | RNN config, separate memory for student/teacher |

### 1.3 Specialized Modules

| Module | File:Lines | Description | Key Features | Extension Points |
|--------|-----------|-------------|-------------|-----------------|
| **RandomNetworkDistillation (RND)** | `rnd.py:22-216` | Curiosity-driven intrinsic rewards | Predictor + frozen target network, state/reward normalization, weight scheduling (constant/step/linear) | Predictor/target architecture, normalization strategy, weight schedule modes |
| **AMPDiscriminator** | `amp_discriminator.py:12-85` | Adversarial Motion Prior (AMP) | Binary classifier for expert vs policy motions, LS-GAN or BCE loss, gradient penalty, observation normalization | Loss type (LS-GAN/BCE), gradient penalty coefficient, norm clipping |
| **Estimator** | `estimator.py:13-38` | State estimation (proprioceptive → privileged) | MLP for predicting hidden/privileged states from proprioception | Hidden dims, activation functions |
| **Discriminator** | `estimator.py:40-73` | Skill/state discriminator | Multi-class classifier, optionally with spectral normalization | Hidden dims, num_skills parameter |
| **DepthOnlyFCBackbone58x87** | `depth_backbone.py:70-102` | Depth image encoder | Conv2D → FC, compresses 58×87 depth to latent (32-dim typical) | Kernel/stride, intermediate dims |
| **RecurrentDepthBackbone** | `depth_backbone.py:10-41` | Depth + GRU encoder | Combines depth CNN with GRU temporal modeling + output MLP | GRU hidden size, combination MLP dims |

---

## 2. NETWORK PRIMITIVES (rsl_rl/networks/)

### 2.1 Core Network Blocks

| Primitive | File:Lines | Description | Key Features | Extension Points |
|-----------|-----------|-------------|-------------|-----------------|
| **MLP** | `mlp.py:20-80` | Multi-layer perceptron | Flexible input/output dims, supports activation/last_activation, dimension inference with -1 | Activation function, output shape handling |
| **CNN** | `cnn.py:20-150` | Convolutional network | Supports 2D convolutions, normalization (batch/layer), pooling, global pooling, flattening | Norm type, padding mode, pooling strategy |
| **Memory (RNN)** | `memory.py:25-95` | GRU/LSTM wrapper | Supports GRU/LSTM, handles batched hidden states, trajectory unpadding for sequences | RNN type (GRU/LSTM), hidden size, num_layers |

### 2.2 Normalization Layers

| Normalizer | File:Lines | Description | Key Features | Extension Points |
|-----------|-----------|-------------|-------------|-----------------|
| **EmpiricalNormalization** | `normalization.py:19-73` | Running mean/variance normalization | Tracks mean/std incrementally, supports training-only updates, has `until` saturation | Epsilon value, saturation threshold |
| **EmpiricalDiscountedVariationNormalization** | `normalization.py:76-110` | Discounted reward normalization | Combines empirical norm with discounted reward averaging (Pathak et al.) | Gamma (discount), epsilon |

---

## 3. RL ALGORITHMS (rsl_rl/algorithms/)

### 3.1 Policy Gradient Algorithms

| Algorithm | File:Lines | Description | Loss Terms | Key Parameters |
|-----------|-----------|-------------|-----------|-----------------|
| **PPO** | `ppo.py:25-465` | Proximal Policy Optimization | Surrogate (clipped), value, entropy, optional RND, optional symmetry | clip_param=0.2, gamma=0.99, lam=0.95, entropy_coef=0.01, value_loss_coef=1.0 |
| **PPOParkour** (PPOParkour variant) | `ppo_parkour.py:25-596` | PPO + private state estimation + history encoder | Surrogate, value, entropy, RND, symmetry, **priv_reg_loss** (adaptation), estimator loss | Same as PPO + priv_reg_coef_schedule, estimator learning rate |
| **PPOParkourOriginal** | `ppo_parkour_original.py:63-457` | Original RMA-based PPO (legacy) | Surrogate, value, entropy, estimator, discriminator, priv_reg | Similar to PPOParkour but different structure |
| **PPOAMPBase** | `ppo_amp.py:14-206` | PPO + AMP (on basic ActorCritic) | Surrogate, value, entropy, **discriminator loss** (LS-GAN/BCE), **gradient penalty** | Same PPO params + amp_cfg dict (disc_loss_type, discriminator_lr, gradient_penalty_coef, reward_coef) |
| **PPOAMP** | `ppo_amp.py:208-402` | PPO + AMP (on ActorCriticRMA) | Surrogate, value, entropy, RND, symmetry, **discriminator loss**, **gradient penalty**, priv_reg | Same as PPOParkour + amp_cfg params |

### 3.2 Imitation Learning Algorithm

| Algorithm | File:Lines | Description | Loss Terms | Key Parameters |
|-----------|-----------|-------------|-----------|-----------------|
| **Distillation** | `distillation.py:20-182` | Behavior cloning via student-teacher | **Behavior loss** (MSE or Huber), optional recurrent handling | num_learning_epochs, gradient_length, loss_type (mse/huber), max_grad_norm |

### 3.3 Loss Terms Summary

| Loss Term | Algorithm(s) | Equation / Computation | Purpose |
|-----------|------------|----------------------|---------|
| **Surrogate (PPO)** | All PPO variants | `-A * clip(exp(logp_new / logp_old), 1±ε)` | Policy gradient with trust region |
| **Value Loss (Clipped)** | All PPO variants | `max((V-R)^2, (V_clipped-R)^2)` where V_clipped ∈ [V±ε] | Value function fitting with clipping |
| **Entropy** | All PPO variants | `sum(p*log(p))` | Exploration encouragement |
| **RND Loss** | PPO, PPOParkour, PPOAMP | `MSE(predictor(s), target(s))` | Curiosity-driven intrinsic reward |
| **Symmetry Loss** | PPO, PPOParkour, PPOAMP | `MSE(actor_mean(sym_obs)[sym_action_preds], sym_actions_expected)` | Mirror symmetry enforcement (optional) |
| **Priv Regularization** | PPOParkour, PPOAMP | `norm(priv_latent - hist_latent, p=2)` | History encoder to match privilege encoder (adaptation) |
| **Estimator Loss** | PPOParkour, PPOAMP | `MSE(estimator(obs_policy), priv_explicit)` | Predict privileged state from proprioception |
| **AMP Discriminator Loss** | PPOAMPBase, PPOAMP | LS-GAN: `MSE(D(expert), 1) + MSE(D(policy), -1)` or BCE | Adversarial motion matching |
| **Gradient Penalty** | PPOAMPBase, PPOAMP | `coef * (‖∇_x D(x)‖_2)^2` (for expert and policy) | Stabilize discriminator training |
| **Logit Regularization** | PPOAMPBase, PPOAMP | L2 on output layer weights or logits | Reduce discriminator overfitting |
| **Behavior Cloning** | Distillation | `MSE(student_actions, teacher_actions)` or `Huber(...)` | Student mimics teacher |

---

## 4. RUNNERS (rsl_rl/runners/)

### On-Policy Training Frameworks

| Runner | File:Lines | Supports | Algorithm Variants | Key Methods |
|--------|-----------|----------|-------------------|------------|
| **OnPolicyRunner** | `on_policy_runner.py:33-300` | Standard PPO | PPO (basic ActorCritic/CNN/Recurrent) | `learn()`, multi-GPU sync, logging |
| **OnPolicyRunnerParkour** | `on_policy_runner_parkour.py:40-350` | PPOParkour with RMA | ActorCriticRMA, RND, symmetry, estimator | `update_dagger()` for history encoder, `learn()` |
| **OnPolicyRunnerParkourOriginal** | `on_policy_runner_parkour_original.py:40-450` | PPOParkourOriginal (legacy) | ActorCriticRMA (original variant) | Depth encoder update, depth actor distillation |
| **OnPolicyRunnerAMP** | `on_policy_runner_amp.py:40-350` | PPOAMPBase on basic ActorCritic | PPOAMPBase | Discriminator buffer, discriminator update loop |
| **OnPolicyRunnerParkourAMP** | `on_policy_runner_parkour_amp.py:40-350` | PPOAMP on ActorCriticRMA | PPOAMP, RMA + AMP combined | Discriminator + history encoder + priv_reg |
| **DistillationRunner** | `distillation_runner.py:25-150` | Behavior cloning | StudentTeacher, StudentTeacherRecurrent | Student-only training loop |

### Storage / Rollout Buffer

| Storage | File:Lines | Type | Supports | Key Features |
|---------|-----------|------|----------|------------|
| **RolloutStorage** | `rollout_storage.py:21-250` | On-policy | Standard RL + Distillation | Stores transitions (obs/actions/rewards/dones/values/log_probs), mini-batch + recurrent generators, GAE computation |

---

## 5. FRAMEWORK ASSUMPTIONS & CONSTRAINTS

### Current Framework Characteristics

| Aspect | Status | Details |
|--------|--------|---------|
| **Default Training Paradigm** | **On-policy only** | Uses rollout storage, GAE advantage computation; no off-policy replay buffer for RL algorithms |
| **Privileged Information Support** | Yes (optional) | ActorCriticRMA allows privileged states during training; RMA/AMP designed for this |
| **Recurrent Policies** | Yes | GRU/LSTM supported, with trajectory padding/unpadding for mini-batch training |
| **Multi-GPU Training** | Yes | Distributed training support in PPO and derived algorithms with gradient averaging |
| **Multi-Task / Skill Learning** | No native support | Discriminator in estimator.py suggests exploration but no full multi-task framework |
| **Attention / Transformer** | **NOT implemented** | No transformer or attention mechanisms found; Memory module is RNN-only (GRU/LSTM) |
| **Off-Policy Methods** | **NOT implemented** | No DDPG, SAC, DQN, or replay buffer for continuous control |
| **Model-Based RL** | **NOT implemented** | No world model, dynamics model, or planning |
| **Hierarchical RL** | **NOT implemented** | No option-critic, HAMs, or feudal structures |

### Storage Design

- **Rollout Storage**: Fixed-size buffer (num_transitions_per_env × num_envs)
- **Replay Buffer**: AMP uses optional circular replay buffer (catastrophic forgetting mitigation only, not core RL)
- **Generator Interface**: Mini-batch sampling with optional recurrent unpadding
- **GAE Computation**: Manual in algorithms (not in storage) for flexibility

### Policy Architecture Flexibility

- **Actor-Critic Coupling**: Separate actor and critic allowed; shared trunk optional
- **Observation Groups**: Configurable via `obs_groups` dict (`policy`, `critic`, `teacher`, `rnd_state`, etc.)
- **Output Handling**: MLP supports scalar or tuple outputs (e.g., for state-dependent std)

---

## 6. ALREADY-IMPLEMENTED LIST

**This list documents all existing implementations to avoid duplicate research/engineering.**

### Network Architectures
- **MLP (Multi-Layer Perceptron)** — `networks/mlp.py` — Fully configurable dense feed-forward network
- **CNN (Convolutional Neural Network)** — `networks/cnn.py` — Conv2D + optional pooling/normalization/flattening
- **GRU/LSTM Memory Modules** — `networks/memory.py:Memory` — Recurrent sequence processing (GRU or LSTM)
- **ActorCritic (Basic)** — `modules/actor_critic.py` — Feed-forward actor-critic with Gaussian policy
- **ActorCriticRecurrent** — `modules/actor_critic_recurrent.py` — RNN-based actor-critic (GRU/LSTM)
- **ActorCriticCNN** — `modules/actor_critic_cnn.py` — Hybrid CNN+MLP for image + vector observations
- **ActorCriticRMA (Parkour Variant)** — `modules/actor_critic_parkour.py` — Privileged training with history encoder + runtime adaptation
- **ActorCriticRMA (Original)** — `modules/actor_critic_parkour_original.py` — Legacy RMA variant
- **StudentTeacher** — `modules/student_teacher.py` — Non-recurrent student (student obs) + teacher (privileged obs)
- **StudentTeacherRecurrent** — `modules/student_teacher_recurrent.py` — Recurrent student-teacher pair
- **StateHistoryEncoder** — `modules/actor_critic_parkour.py:StateHistoryEncoder` — Conv1D temporal history encoding (10/20/50 timestep options)

### RL Algorithms & Variants
- **Proximal Policy Optimization (PPO)** — `algorithms/ppo.py` — Standard PPO with clipped surrogate loss
- **PPO + RND (Random Network Distillation)** — `algorithms/ppo.py` — PPO + curiosity-driven intrinsic rewards
- **PPO + Symmetry Loss** — `algorithms/ppo.py` — PPO + mirror symmetry data augmentation and action matching loss
- **PPO + Multi-GPU Distributed Training** — `algorithms/ppo.py` — Gradient averaging across GPUs
- **PPOParkour** — `algorithms/ppo_parkour.py` — PPO + privileged state estimation + history encoder adaptation
- **PPOParkour + RND + Symmetry + Estimator + PrivReg** — `algorithms/ppo_parkour.py` — Full suite of features
- **PPOParkourOriginal** — `algorithms/ppo_parkour_original.py` — Legacy RMA implementation
- **PPOAMPBase** — `algorithms/ppo_amp.py:PPOAMPBase` — PPO + AMP discriminator (basic ActorCritic)
- **PPOAMP** — `algorithms/ppo_amp.py:PPOAMP` — PPO + AMP discriminator (ActorCriticRMA)

### Adversarial Motion Prior (AMP)
- **AMP Discriminator (LS-GAN variant)** — `modules/amp_discriminator.py` — Expert vs policy motion classification with LS-GAN loss
- **AMP Discriminator (BCE variant)** — `modules/amp_discriminator.py` — MimicKit-style BCE loss alternative
- **Gradient Penalty (AMP)** — `algorithms/ppo_amp.py` — WGAN-GP style regularization
- **Logit Regularization (AMP)** — `algorithms/ppo_amp.py` — Output layer weight L2 or logit value regularization
- **Replay Buffer (AMP)** — `algorithms/ppo_amp.py` — Circular buffer to mitigate discriminator catastrophic forgetting

### Curiosity & Exploration
- **Random Network Distillation (RND)** — `modules/rnd.py` — Predictor-target network for novelty detection
- **RND State Normalization** — `modules/rnd.py` — Optional empirical normalization of input state
- **RND Reward Normalization** — `modules/rnd.py` — Optional discounted variation normalization
- **RND Weight Scheduling** — `modules/rnd.py` — Constant / Step / Linear weight schedule for intrinsic reward decay

### Adaptation & Domain Randomization
- **History Encoder (Conv1D)** — `modules/actor_cricket_parkour.py:StateHistoryEncoder` — Temporal encoding of state history
- **Privilege Regularization Loss** — `algorithms/ppo_parkour.py` — L2 matching between history and privilege latents (DAGGER-style)
- **Estimator Network** — `modules/estimator.py:Estimator` — MLP for state estimation (proprioceptive → privileged)

### Imitation Learning & Distillation
- **Behavioral Cloning (Distillation)** — `algorithms/distillation.py` — Student mimics teacher actions
- **MSE vs Huber Loss** — `algorithms/distillation.py` — Configurable loss for distillation
- **Recurrent Distillation** — `algorithms/distillation.py:Distillation` — Supports StudentTeacherRecurrent

### Normalization & Observation Handling
- **Empirical Normalization (Running Mean/Var)** — `networks/normalization.py:EmpiricalNormalization` — Online whitening
- **Empirical Discounted Variation Normalization** — `networks/normalization.py:EmpiricalDiscountedVariationNormalization` — Reward normalization (Pathak et al.)
- **Observation Groups** — All modules — Flexible selection of policy / critic / teacher / rnd_state / scan obs

### Training Infrastructure
- **On-Policy Runner (Basic)** — `runners/on_policy_runner.py` — Standard rollout → learn loop
- **On-Policy Runner (Parkour)** — `runners/on_policy_runner_parkour.py` — Parkour variant with DAGGER
- **On-Policy Runner (AMP)** — `runners/on_policy_runner_amp.py` — AMP discriminator training loop
- **On-Policy Runner (Parkour+AMP)** — `runners/on_policy_runner_parkour_amp.py` — Combined RMA + AMP
- **Distillation Runner** — `runners/distillation_runner.py` — Student-only training with teacher in eval mode
- **Rollout Storage** — `storage/rollout_storage.py` — On-policy trajectory buffer with mini-batch generators
- **Multi-GPU Synchronization** — PPO and derived algorithms — Parameter broadcast and gradient reduction
- **Logger** — `utils/logger.py` — Wandb/Neptune integration

### Specialized Encoders & Backbones
- **Depth Image Encoder (FC)** — `modules/depth_backbone.py:DepthOnlyFCBackbone58x87` — Conv2D compression of 58×87 depth to latent
- **Recurrent Depth Encoder** — `modules/depth_backbone.py:RecurrentDepthBackbone` — Depth + GRU + output MLP
- **StackDepthEncoder** — `modules/depth_backbone.py:StackDepthEncoder` — Temporal stacking of depth frames via Conv1D

### Loss Functions & Regularization
- **Clipped Surrogate Loss** — PPO family — Trust region via action probability ratio clipping
- **Clipped Value Loss** — PPO family — Value function fitting with trust region
- **Entropy Regularization** — PPO family — Gaussian action entropy bonus
- **RND MSE Loss** — PPO + RND — Predictor vs target embedding distance
- **Symmetry Mirror Loss** — PPO family — MSE of symmetric action predictions
- **Privilege Regularization (L2)** — PPOParkour family — History latent ↔ privilege latent matching
- **State Estimation MSE** — PPOParkour family — Estimator network prediction error
- **AMP Discriminator Loss (LS-GAN)** — AMP family — Expert→+1, Policy→-1 targets
- **AMP Discriminator Loss (BCE)** — AMP family — MimicKit-style expert→1, policy→0
- **Gradient Penalty (LS)** — AMP family — WGAN-GP regularization on discriminator
- **Logit Regularization** — AMP family — Output layer weight or logit L2 norm
- **Behavior Cloning MSE/Huber** — Distillation — Student action prediction

### Advanced Training Features
- **Adaptive Learning Rate Schedule (KL-based)** — PPO family — Adjust LR if KL divergence crosses threshold
- **Per-Mini-Batch Advantage Normalization** — PPO family — Optional normalization within each mini-batch
- **Data Augmentation (Symmetry)** — PPO + symmetry_cfg — Augment observations/actions via environment symmetries
- **Mini-Batch Sampling with Recurrent Unpadding** — Storage — Handle variable-length sequences
- **GAE (Generalized Advantage Estimation)** — PPO family — λ-weighted TD residual accumulation

---

## 7. NOT IMPLEMENTED (To Avoid Redundant Proposals)

| Category | Example | Status |
|----------|---------|--------|
| **Off-Policy Algorithms** | DDPG, SAC, TD3, DQN | Not in rsl_rl |
| **Model-Based RL** | World models, PETS, MBPO | Not in rsl_rl |
| **Hierarchical RL** | Option-Critic, HAMs, Feudal RL | Not in rsl_rl |
| **Multi-Task Learning** | MTL, Meta-RL (MAML) | Not natively; discriminator concept exists but no framework |
| **Attention Mechanisms** | Transformer encoder, self-attention | Not in rsl_rl |
| **Large-Scale State Space** | Graph networks, set transformers | Not in rsl_rl |
| **Evolutionary Algorithms** | Genetic algorithms, CMA-ES | Not in rsl_rl |
| **Model Predictive Control** | MPC, planning | Not in rsl_rl |
| **Inverse RL / Reward Learning** | IRL, AIRL, GAIL | Not in rsl_rl (AMP is motion imitation, not inverse RL) |
| **Safe RL** | Constrained MDPs, Lagrangian methods | Not in rsl_rl |
| **Offline RL** | CQL, IQL, AWAC | Not in rsl_rl |

---

## 8. EXTENSION POINTS FOR FUTURE WORK

### By Difficulty Level

#### Easy (Minimal Changes)
1. **Add new activation functions** — Modify `resolve_nn_activation()` in utils
2. **Tune hyperparameters** — RND weight schedule, symmetry loss coefficient, AMP discriminator dims
3. **Change loss type** — Swap MSE ↔ Huber in distillation, BCE ↔ LS-GAN in AMP
4. **Adjust network dimensions** — Hidden layer sizes, RNN hidden state size

#### Medium (Module/Algorithm Modifications)
1. **Custom observation group** — Define new obs group in config (e.g., vision-only policy)
2. **Modify history encoder** — Replace Conv1D with different temporal architecture (e.g., Transformer)
3. **Add new discriminator regularization** — Extend AMP with spectral normalization, instance norm
4. **Estimator architecture variation** — Use different backbone for state estimation

#### Hard (Architectural Changes)
1. **Off-policy training** — Implement replay buffer, add off-policy algorithm (SAC, DDPG)
2. **Transformer-based policy** — Replace MLP actor with attention-based architecture
3. **Multi-task learning framework** — Extend algorithms to support task-specific heads and losses
4. **Hierarchical RL** — Add option/skill layers with skill discriminator
5. **Model-based components** — Add dynamics model and planning module

---

## 9. REFERENCES & CODE LOCATIONS

### Key Configuration Structures

```python
# Algorithm config (from train_cfg["algorithm"])
{
    "rnd_cfg": {
        "num_states": ...,
        "num_outputs": 128,
        "predictor_hidden_dims": [256, 256],
        "target_hidden_dims": [256, 256],
        "weight": 0.01,
        "state_normalization": True,
        "reward_normalization": True,
        "weight_schedule": {"mode": "linear", "initial_step": 0, "final_step": 10000, "final_value": 0.0}
    },
    "symmetry_cfg": {
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "data_augmentation_func": "symmetry_function_name",
        "mirror_loss_coeff": 0.1
    },
    "amp_cfg": {
        "task_reward_lerp": 0.5,
        "discriminator_learning_rate": 1e-4,
        "gradient_penalty_coef": 10.0,
        "reward_coef": 2.0,
        "disc_loss_type": "ls_gan",  # or "bce"
        "disc_logit_reg_type": "logit",  # or "weight"
        "enable_replay_buffer": True,
        "replay_buffer_size": 100000,
        "disc_num_epochs": 2
    }
}
```

### Typical Observation Groups Structure

```python
# obs_groups (from train_cfg["obs_groups"])
{
    "policy": ["obs_group_1", "obs_group_2"],
    "critic": ["obs_group_1", "obs_group_2"],
    "teacher": ["obs_group_1", "obs_group_2", "privileged_obs"],
    "rnd_state": ["obs_group_1"],  # if using RND
}
```

---

## 10. SUMMARY & GROUNDING

### Framework Character
- **Strongly on-policy**: Built around PPO; all variants extend PPO
- **Privilege-aware**: RMA/AMP designed for privileged information during training
- **Modular**: Easy to add new observation groups, algorithms, modules
- **Distributed-ready**: Multi-GPU gradient averaging built-in

### What's NOT Here (To Guide Literature Search)
- No off-policy methods → If considering SAC, DDPG, DQN, this is new
- No Transformers → If considering attention-based policies, this is new
- No Model-Based RL → If considering world models, planning, this is new
- No Hierarchical RL → If considering options, skills, feudal structures, this is new
- No Inverse RL → If considering reward learning (beyond motion imitation), this is new
- No Safe RL → If considering constraints, Lagrangians, this is new

### Recommended Next Steps for Research
1. **Check what you want to add** against the ALREADY-IMPLEMENTED LIST
2. **Identify where to hook in** using extension points in columns 4-5 of the tables
3. **Look at analogous implementations** (e.g., if adding SAC, study PPO closely)
4. **Test incrementally** using existing runners and logging infrastructure

---

**End of Inventory**
