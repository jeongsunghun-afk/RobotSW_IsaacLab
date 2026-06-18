# IsaacLab Go2 Parkour — Code Anchor Map

> File references are absolute paths with line numbers.  All facts are read directly from source; no inferences beyond trivial arithmetic.

---

## 1. Policy / Network Architecture

### Actor (`ActorCriticRMA`)

Source: `rsl_rl/rsl_rl/modules/actor_critic_parkour.py`

**Actor MLP** — `[512, 256, 128]` hidden dims, ELU activation (cfg: `rsl_rl_ppo_cfg.py:71`).

Actor input is a *concatenation* of four streams built at forward time (`actor_critic_parkour.py:139`):

```
actor_input_dim = num_actor_obs(46) + num_priv_explicit(6) + priv_encoder_out(20) + scan_latent(32)
               = 46 + 6 + 20 + 32 = 104
```

- `num_actor_obs = 46` — the raw `policy` obs (proprioception)
- `num_priv_explicit = 6` — passed directly, not encoded
- `priv_encoder_out = 20` — output of `priv_encoder` MLP or `history_encoder` (both output `priv_encoder_dims[-1]`)
- `scan_latent = 32` — output of `scandot_encoder` MLP

**Critic MLP** — `[512, 256, 128]` hidden dims, ELU, input = full privileged observation (272 dims).
`critic_obs = policy(46) + scan(187) + priv_explicit(6) + priv_latent(37) = 272`
(cfg comment: `rsl_rl_ppo_cfg.py:48`)

### Depth Backbone / Scan Encoder

Source: `rsl_rl/rsl_rl/modules/depth_backbone.py` and `actor_critic_parkour.py:153-162`

**In deployment (this implementation):** depth images are NOT used. The height scan (scandots, 187 dims) is encoded by a **scandot MLP encoder** — NOT a CNN.

`scandot_encoder = MLP(187 → [128, 64, 32], output_dim=32)` (`actor_critic_parkour.py:156`)

The `depth_backbone.py` file defines `DepthOnlyFCBackbone58x87` (CNN: Conv2d 1→32 k5, MaxPool2d, Conv2d 32→64 k3, Flatten, Linear→128, Linear→scandots_output_dim) and `RecurrentDepthBackbone` / `StackDepthEncoder` wrappers, but **none of these are instantiated** in the parkour runner or actor. They exist in the module file but are unused in the current training setup.

### RNN / GRU / Transformer

**No recurrent module in the actor or critic.** `ActorCriticRMA.is_recurrent = False` (`actor_critic_parkour.py:82`).

The `StateHistoryEncoder` processes proprioceptive history with **temporal 1D Conv layers** (not GRU/LSTM/Transformer):

- Per-timestep projection: `Linear(46 → 30)` (channel_size=10, output=3*channel_size)
- For `tsteps=10`: `Conv1d(30→20, k=4, s=2)` → `Conv1d(20→10, k=2)` → Flatten → `Linear(30 → priv_encoder_dims[-1]=20)` (`actor_critic_parkour.py:48-55, 67-69`)

The `RecurrentDepthBackbone` in `depth_backbone.py` contains a `GRU(32, 512)` but is **not used**.

### Latent / Encoder Dims

| Encoder | Input | Hidden | Output |
|---|---|---|---|
| `scandot_encoder` | 187 (height scan) | [128, 64] | **32** |
| `priv_encoder` | 37 (priv_latent) | [64] | **20** |
| `history_encoder` (StateHistoryEncoder) | 46 × 10 timesteps | conv1d | **20** |

`priv_encoder_dims` default: `[64, 20]` (parkour_cfg line 94).
`scan_encoder_dims` default: `[128, 64, 32]` (parkour_cfg line 93).

### Proprioception History Length

`history_len = 10` timesteps (`parkour_env_cfg.py:433`).
History buffer shape: `(num_envs, 10, 46)`.
Note: the first 2 elements (yaw_diff, next_yaw_diff) are zeroed in the history copy to avoid leaking goal info into the history encoder (`parkour_env.py:1031`).

---

## 2. Algorithm

### PPO Variant

`PPOParkour` (`rsl_rl/rsl_rl/algorithms/ppo_parkour.py`).  
Standard PPO ratio-clip surrogate (default `surrogate_type="ppo"`, clip=0.2).  
Optional variants registered: SPO (`Go2ParkourSPOPPORunnerCfg`), LCP Lipschitz penalty (`Go2ParkourLCPPPORunnerCfg`), MoE actor (`Go2ParkourMoEPPORunnerCfg`), L/R symmetry aug (`Go2ParkourSymmetryPPORunnerCfg`).

Key hypers (cfg: `rsl_rl_ppo_cfg.py:78-92`): lr=2e-4 adaptive (desired_kl=0.01), γ=0.99, λ=0.95, epochs=5, mini_batches=4, entropy_coef=0.01, value_loss_coef=1.0.  
All resets treated as time-outs (bootstrap-all): `_get_dones` always returns `terminated=zeros`, `time_out=reset_all` (`parkour_env.py:1292-1293`).

### Teacher–Student / DAgger Structure (RMA-style)

**Two-phase training in a single loop:**

**Phase 1 (PPO update, `ppo_parkour.py:241`):**  
During collection, actor uses `priv_latent` (ground-truth domain params, teacher signal) from `priv_encoder`.  
Priv-reg loss pushes `priv_encoder(priv_latent)` toward `history_encoder(proprio_history).detach()`:  
`priv_reg_loss = ‖priv_encoder(z) − history_encoder(hist).detach()‖₂`  
Coefficient ramps from 0 → 0.1 linearly between iteration 2000 and 3000 (`ppo_parkour.py:131`).

**Phase 2 (DAgger / history encoder update, `ppo_parkour.py:588`, `update_dagger()`):**  
Runs *after* PPO update on the same stored rollout.  
Trains `history_encoder` to match `priv_encoder(priv_latent).detach()`:  
`hist_latent_loss = ‖priv_latent.detach() − history_encoder(hist)‖₂`  
Separate optimizer (`hist_encoder_optimizer`, Adam lr=learning_rate).

**Inference / deploy time:** `act_inference` uses `history_encoder` only (no priv_latent, no priv_explicit oracle) (`actor_critic_parkour.py:290-305`).

**Estimator (optional):** A small MLP `[128, 64]` trained to predict `priv_explicit` (lin_vel + ang_vel, 6 dims) from `policy` obs alone (`rsl_rl_ppo_cfg.py:59-63`). When `train_with_estimated_states=True`, actor input uses estimated lin/ang vel instead of oracle during collection (`ppo_parkour.py:174`).

### Privileged Observations

| Name | Dims | Content | Who sees it |
|---|---|---|---|
| `priv_explicit` | 6 | `root_lin_vel_b * 2.0` (3) + `root_ang_vel_b * 0.25` (3) | Critic + actor during training (oracle); estimator at deploy |
| `priv_latent` | 37 | base_friction(1) + foot_friction(8) + base_mass(1) + base_com(3) + joint_stiffness_ratio(12) + joint_damping_ratio(12) | Critic + priv_encoder during training only; replaced by history_encoder at deploy |

Sources: `parkour_env.py:962-1016`, `parkour_env_cfg.py:419-424`, `rsl_rl_ppo_cfg.py:41-56`.

**No scandots/elevation map as privileged obs** — the height scan (187) goes through `scandot_encoder` and is available to both actor (via scan_latent) and critic. It is not labeled privileged.

**No oracle heading** — heading error is computed as a delta-yaw relative signal and passed to the *actor* as part of `policy` obs (not privileged).

### Distillation Phases

Not a staged curriculum (no separate "teacher checkpoint → student training" pipeline). Both priv_reg and dagger steps run simultaneously from iteration 0; priv_reg coefficient is ramped in after 2000 iterations.

---

## 3. Observation Space

### Actor (Deploy-time) Observation

Key: `"policy"` in env dict. Shape: `(num_envs, 46)`.

Source: `parkour_env.py:921-934` (assembled) + `parkour_env_cfg.py:426,430`.

| Component | Dim | Source |
|---|---|---|
| delta_yaw (target_yaw − heading_w, wrapped ±π) | 1 | `parkour_env.py:921` |
| delta_next_yaw | 1 | `parkour_env.py:922` |
| projected_gravity_b | 3 | `parkour_env.py:923` |
| lin_vel_x command | 1 | `parkour_env.py:924` |
| joint_pos − default_joint_pos | 12 | `parkour_env.py:925` |
| joint_vel × 0.05 | 12 | `parkour_env.py:926-929` |
| actions (current step) | 12 | `parkour_env.py:930` |
| contact_filt − 0.5 (4 feet, debounced) | 4 | `parkour_env.py:931` |
| **Total** | **46** | |

Note: stale comment at `parkour_env.py:935` says `proprio dim = 42`; the actual value is 46 (contact_filt added later). The cfg constants `num_proprio=42+4=46` and `observation_space=42+4=46` are correct.

### Privileged / Teacher-only obs

Both `priv_explicit` (6) and `priv_latent` (37) are in the env dict but **not in the actor's deploy-time input**. They are consumed only by the critic and the priv_encoder/estimator paths during training.

### Foot Contact in Observations vs Reward-only

**Foot contact IS in the actor observation** (deploy-time):  
`contact_filt` = debounced boolean per-foot contact (4 feet, `contact_sensor.net_forces_w_history` threshold=2.0 N, OR'd with previous step) appears as the last 4 dims of `policy` obs (value: float, `contact - 0.5`). `parkour_env.py:915-918,931`.

**Foot position is NOT in the observation.** Foot positions are only used inside `_get_rewards()` for the `feet_edge` penalty (world XY lookup in edge mask). No foot position or foot velocity enters any observation group.

---

## 4. Reward Terms

Source: `parkour_env.py:1230-1248`, scales from `parkour_env_cfg.py:607-637`.

All terms scaled by `cfg.reward_scales[key] * step_dt * value`. Total clipped at `min=0.0` (`parkour_env.py:1261`).

| Term | Scale | Sign | Notes |
|---|---|---|---|
| `tracking_goal_vel` | +1.5 | bonus | `min(proj_vel_toward_goal, commanded_speed) / commanded_speed` |
| `tracking_yaw` | +0.5 | bonus | `exp(−|target_yaw − heading|)` |
| `lin_vel_z_l2` | −1.0 | penalty | z-vel squared; halved on non-flat terrain |
| `ang_vel_xy_l2` | −0.05 | penalty | roll+pitch rate squared; halved on non-flat |
| `orientation_l2` | −1.0 | penalty | `‖gravity_b[:2]‖²`; zero on non-flat |
| `dof_acc_l2` | −2.5e-7 | penalty | `Σ((joint_vel − prev_joint_vel)/step_dt)²` |
| `collision` | −10.0 | penalty | undesired body contacts (base, thigh, calf, hip, head) |
| `action_rate_l2` | −0.1 | penalty | `‖actions − prev_actions‖₂` (L2 norm) |
| `delta_torques` | −1e-7 | penalty | `Σ(torque − prev_torque)²` |
| `torques_l2` | −1e-5 | penalty | `Σ(applied_torque)²` |
| `hip_pos` | −0.5 | penalty | hip joint deviation from default |
| `dof_error_l2` | −0.04 | penalty | all joint deviation from default |
| `feet_stumble` | −1.0 | penalty | lateral contact force > 4× vertical (`parkour_env.py:1132-1137`) |
| `feet_edge` | −1.0 | penalty | foot contact on terrain edge cells; gated terrain_level>3 |
| `feet_dragging` | −0.1 | penalty | hind feet (RL, RR) only; contact + xy_vel > 0.05 m/s |
| `feet_gait_pairing` | 0.0 | (disabled) | Spot-style diagonal pair trot sync; weight=0 |
| `air_time_cap` | −0.1 | penalty | per-foot excess air time > 1.0 s; graded |
| `contact_duty_deficit` | −0.5 | penalty | per-foot EMA contact duty < 0.5 target; flat terrain only |

**Foot-clearance/contact/gait/air-time terms already present:**
- `feet_dragging` (contact + velocity penalty, hind feet): present, active (−0.1)
- `feet_stumble` (contact force geometry): present, active (−1.0)
- `feet_edge` (contact at terrain edges): present, active (−1.0)
- `air_time_cap` (per-foot max air time): present, active (−0.1)
- `contact_duty_deficit` (EMA contact duty): present, active (−0.5)
- `feet_gait_pairing` (trot diagonal sync): present but **weight=0.0** (disabled)

**No positive air-time reward** (e.g. Genesis-style `air_time` bonus). The only air-time term is a cap-penalty.

---

## 5. What Is NOT Present

| Missing Feature | Evidence |
|---|---|
| **Recurrent encoder in actor/critic** | `is_recurrent = False` (`actor_critic_parkour.py:82`); `StateHistoryEncoder` uses 1D Conv, no GRU/LSTM/Transformer. The GRU in `RecurrentDepthBackbone` (`depth_backbone.py:22`) is not instantiated. |
| **Contrastive / auxiliary prediction heads** | No auxiliary loss beyond `priv_reg_loss` (L2 regression) and optional `hist_latent_loss`. No contrastive pairs, no self-supervised prediction targets. |
| **Gait / periodicity priors** | `feet_gait_pairing` reward term exists in code but `reward_scales["feet_gait_pairing"] = 0.0` — disabled. No CPG, clock signal, or phase-based gait reward is active. |
| **CaT-style termination constraints** | All resets use `time_out=True` (bootstrap-all). No curriculum-over-adversarial-terminations, no CaT termination shaping. |
| **Per-leg utilization penalties** | No per-leg asymmetry penalty beyond the symmetric `contact_duty_deficit` (applied uniformly to all 4 feet). No L/R leg imbalance term by default (symmetry aug is a separate variant `Go2ParkourSymmetryPPORunnerCfg`, inactive by default). |
| **Depth camera / RGB-D CNN** | `DepthOnlyFCBackbone58x87`, `RecurrentDepthBackbone`, `StackDepthEncoder` exist in `depth_backbone.py` but none are instantiated in the parkour runner. Height scan (scandots) is used instead. |
| **Positive air-time / step-frequency reward** | No positive bonus for swing time. Only negative cap penalty (`air_time_cap`). |
| **Oracle foot position in obs** | Foot positions are used in reward (`feet_edge`) but not fed to the policy or critic as an observation component. |

---

## Key File References

| File | Role |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` | Environment implementation (`Go2ParkourEnv`) |
| `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` | Environment config (`ParkourEnvCfg`, reward scales, terrain cfg) |
| `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/agents/rsl_rl_ppo_cfg.py` | Runner configs (obs_groups, network dims, algorithm hypers) |
| `rsl_rl/rsl_rl/modules/actor_critic_parkour.py` | `ActorCriticRMA` + `StateHistoryEncoder` |
| `rsl_rl/rsl_rl/algorithms/ppo_parkour.py` | `PPOParkour` (priv_reg + DAgger + optional LCP/SPO/symmetry) |
| `rsl_rl/rsl_rl/modules/depth_backbone.py` | CNN backbones (defined but unused in parkour) |
