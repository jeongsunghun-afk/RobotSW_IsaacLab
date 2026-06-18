# Task #4 — [HYP-ALGO] PPO Action Distribution Analysis
**Date**: 2026-05-26
**Scope**: ActorCriticRMA + PPOParkour — action distribution parameters and gradient flow

---

## 0. Variant Identification (Which Code Is Actually Running)

| Component | Class Used | Source File |
|-----------|-----------|-------------|
| Runner | `OnPolicyRunnerParkour` | `rsl_rl/rsl_rl/runners/on_policy_runner_parkour.py:35` |
| Actor-Critic | `ActorCriticRMA` | `rsl_rl/rsl_rl/modules/actor_critic_parkour.py:81` |
| Algorithm | `PPOParkour` | `rsl_rl/rsl_rl/algorithms/ppo_parkour.py:25` |
| Config entry | `Go2ParkourPPORunnerCfg` | `source/isaaclab_tasks/.../parkour/agents/rsl_rl_ppo_cfg.py:12` |

**Evidence**: `parkour/__init__.py:24` registers `"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourPPORunnerCfg"`. That cfg specifies `class_name = "OnPolicyRunnerParkour"` (line 30), `policy.class_name = "ActorCriticRMA"` (line 58), `algorithm.class_name = "PPOParkour"` (line 69).

> **Variants NOT used**: `ppo.py`, `ppo_parkour_original.py`, `actor_critic.py`, `actor_critic_parkour_original.py` — all excluded from further analysis.

---

## 1. Hyperparameter Table

| Parameter | Value | Source (file:line) | Range Check | Hypothesis Supported |
|-----------|-------|-------------------|-------------|----------------------|
| `init_noise_std` | **1.0** | `rsl_rl_ppo_cfg.py:59` | Moderate — yields σ=1.0 rad raw action | HA1 (partial) |
| `noise_std_type` | **"scalar"** (default) | `rl_cfg.py:31` (not overridden in parkour cfg) | No override in task cfg | HA3, HA5 |
| `state_dependent_std` | **False** (default) | `rl_cfg.py:34` (not overridden) | State-independent std | HA5 |
| `entropy_coef` | **0.01** | `rsl_rl_ppo_cfg.py:73` | Small; bonus ≈ 0.17 nats at init | HA2 |
| `clip_param` | **0.2** | `rsl_rl_ppo_cfg.py:72` | Standard | HA6 (refutes) |
| `max_grad_norm` | **1.0** | `rsl_rl_ppo_cfg.py:81` | Standard | — |
| `learning_rate` | **2.0e-4** (init) | `rsl_rl_ppo_cfg.py:76` | Adaptive range: [1e-5, 1e-2] | HA5 (interacts) |
| `schedule` | **"adaptive"** | `rsl_rl_ppo_cfg.py:77` | `desired_kl=0.01` | Adaptive amplification |
| `desired_kl` | **0.01** | `rsl_rl_ppo_cfg.py:79` | Standard | — |
| `value_loss_coef` | **1.0** | `rsl_rl_ppo_cfg.py:70` | Equal weight to surrogate | Low impact |
| `clip_actions` | **10.0** | `rsl_rl_ppo_cfg.py:27` | Very loose hard clip | HA4 (enables) |
| `activation` | **"elu"** | `rsl_rl_ppo_cfg.py:64` | Applied to all hidden layers | — |
| `actor_hidden_dims` | **[512, 256, 128]** | `rsl_rl_ppo_cfg.py:63` | Standard depth | — |
| `priv_reg_coef_schedual` | **[0, 0.1, 2000, 3000]** | `ppo_parkour.py:124` | Hardcoded, not in cfg | Extra gradient on priv_encoder |

---

## 2. Action Sampling Formula (Step-by-Step)

### 2.1 Actor Input Construction (`actor_critic_parkour.py:271-288`)

```
obs_actor = proprio_obs[42]                          # line 272
obs_actor = actor_obs_normalizer(obs_actor)           # line 273  (Identity, normalization=False)
priv_explicit = priv_explicit_obs_normalizer(...)     # line 274  (Identity)
priv_latent = priv_encoder(priv_obs[12])              # line 279  → dim=20 (priv_encoder_dims[-1])
obs_actor = cat([obs_actor(42), priv_explicit(6), priv_latent(20)], dim=-1)  # line 280 → dim=68
scan_latent = scandot_encoder(scan_obs[187])          # line 284  → dim=32 (scan_encoder_dims[-1])
obs_actor = cat([obs_actor(68), scan_latent(32)], dim=-1)  # line 285 → dim=100
```

**Total actor input**: 42 + 6 + 20 + 32 = **100 dims**
(Actor hidden: 512→256→128; output: 12 = num_actions)

### 2.2 Actor MLP Output (`rsl_rl/networks/mlp.py:66-68`)

```python
# MLP last layer: nn.Linear(128, 12), no last_activation (None by default)
mean = actor(obs_actor)   # shape [N, 12], UNBOUNDED — no tanh, no sigmoid
```

**Key fact**: `MLP.__init__` only appends `last_activation_mod` if `last_activation is not None` (mlp.py:77-78). `ActorCriticRMA` calls `MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)` without `last_activation` argument (actor_critic_parkour.py:143) → **linear final layer, no activation**.

### 2.3 Noise Standard Deviation (`actor_critic_parkour.py:213-218`)

```python
# noise_std_type="scalar", state_dependent_std=False → branch at line 213
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
# = nn.Parameter(1.0 * torch.ones(12))
# shape: [12], one per joint, all initialized to 1.0
```

- `std` is a **raw (non-log) learnable parameter**, per-joint, state-independent
- **No clamping, no softplus, no floor** anywhere in the forward path

### 2.4 Distribution Creation (`actor_critic_parkour.py:261-269`, `_update_distribution`)

```python
std = self.std.expand_as(mean)   # shape [N, 12], broadcast of [12] parameter
self.distribution = Normal(mean, std)
# Note: Normal.set_default_validate_args(False)  ← line 225, validation disabled
```

### 2.5 Action Sample (`actor_critic_parkour.py:288`)

```python
return self.distribution.sample()
# = mean + eps * std,  eps ~ N(0,I)
# shape [N, 12], unbounded
```

### 2.6 Clip at Wrapper (`vecenv_wrapper.py:163-164`)

```python
if self.clip_actions is not None:
    actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
# clip_actions=10.0 → torch.clamp(actions, -10.0, 10.0)
```

**Full chain**: `N(actor_MLP(obs), std_param)` → `sample()` → `clamp(±10)`

---

## 3. Hypothesis Evaluation

### HA1: `init_noise_std` too large → samples hit boundary from t=0

**Evidence**:
- `init_noise_std = 1.0` → `std = nn.Parameter(1.0 * ones(12))`
- Action clip boundary = ±10.0
- P(|action| > 10 | mean=0, std=1) ≈ 7.6×10⁻²⁴ → statistically impossible at init
- But after action_scale=0.25 in env, the effective joint target perturbation is ±1.0 * 0.25 = ±0.25 rad

**Status**: ❌ **Inconsistent as direct trigger** — at std=1.0, samples almost never reach the ±10 clip. The problem is not that samples start at the boundary; it's that the mean can _drift_ there without a gradient barrier.

**Falsifiability**: If std were lowered to 0.1 and divergence persisted at same onset, HA1 is fully refuted.

---

### HA2: `entropy_coef = 0.01` too small → entropy collapses → mean rides advantage

**Evidence**:
- Per-joint entropy for Normal: `H = 0.5 * log(2πe * σ²)`; at σ=1.0, H ≈ 1.42 nats
- Total policy entropy = sum over 12 dims ≈ 17 nats
- Entropy bonus in loss = 0.01 × 17 ≈ 0.17 — this is subtracted from the loss (maximized)
- Surrogate loss is typically O(0.01–0.1) per step; entropy bonus is comparable at init but shrinks as std shrinks
- Once std falls (e.g., to 0.3), entropy bonus ≈ 0.01 × 12 × 0.5 × log(2πe × 0.09) ≈ -0.33 — the bonus becomes slightly negative, providing even less resistance

**Status**: 🟡 **Partially consistent** — entropy_coef=0.01 is standard but provides only weak resistance to collapse. Once std starts shrinking from advantage pressure, the entropy term cannot stop it. This is an enabling condition, not a root cause.

**Falsifiability**: Increase `entropy_coef` to 0.01→0.05 and check if divergence onset is delayed or absent.

---

### HA3: `std` parameter (raw, not log) has no lower bound → can collapse toward 0 or go negative

**Evidence**:
- `noise_std_type = "scalar"` → `self.std = nn.Parameter(...)` (actor_critic_parkour.py:214)
- This is the **raw** standard deviation, not its logarithm
- Gradients: ∂L/∂σ flows directly from the surrogate loss through `Normal.log_prob(a)`:
  - `log_prob(a) = -0.5*((a-μ)/σ)² - log(σ) - 0.5*log(2π)`
  - ∂log_prob/∂σ = `(a-μ)²/σ³ - 1/σ` — this can push σ toward 0 if advantage favors determinism
- **No softplus, no clamp, no eps guard** before `Normal(mean, std)` is constructed
- `Normal.set_default_validate_args(False)` (line 225) means PyTorch won't raise an error even if `std ≤ 0`
- If std goes negative: `Normal.sample()` = `mean + eps * negative_std` → bimodal inversion, `log_prob` = NaN (log of negative)

**Status**: 🔴 **Consistent and mechanistically plausible** — this is a genuine vulnerability. The raw-std parameterization with no floor allows σ to approach 0, at which point the policy becomes nearly deterministic and the mean can drift far without noise. Common fix: `log_std` parameterization or softplus activation on std output.

**Falsifiability**: If `noise_std_type = "log"` (which adds `std = exp(log_std)`) is switched on and divergence disappears, HA3 is confirmed.

---

### HA4: Actor head output unbounded (linear), clip only at wrapper boundary — no gradient barrier

**Evidence**:
- Actor MLP final layer: `nn.Linear(128, 12)` with no activation (mlp.py:66-78)
- Clip at `±10.0` in `RslRlVecEnvWrapper.step()` (vecenv_wrapper.py:164)
- The clamp operation `torch.clamp(actions, -10, 10)` has **gradient = 0** where clamp is active:
  - For `a > 10`: clamp output = 10, gradient w.r.t. `a` = 0 → no gradient propagates back through the policy for out-of-range actions
- The rollout stores `self.transition.actions` = `policy.act(obs).detach()` (ppo_parkour.py:155-157) — this is the **clipped** action (after wrapper)
- But `old_actions_log_prob_batch` = `policy.get_actions_log_prob(clipped_action)` — the log-prob of the clipped value under the current distribution

**Analysis**: When policy mean drifts toward, say, +9.0 for a calf joint:
- Most samples fall in [7, 11], clipped to [7, 10]
- The log-prob `log_prob(10.0 | μ=9.0, σ=1.0)` is computed for a value at the extreme tail
- The PPO ratio `exp(log_prob_new - log_prob_old)` remains computable but the gradient back to μ is damped (since log_prob changes slowly in the tail)
- There is NO attractive gradient pulling μ back from ±10

**Status**: 🔴 **Consistent** — unbounded actor output + hard clip with no gradient through the boundary creates a "gravity well" near ±10.0. Once mean drifts there, gradient pressure is low to return it.

**Falsifiability**: If `tanh` is added as final actor activation (output ∈ (-1, 1), before action_scale), divergence to boundaries should stop. The gradient of tanh provides soft resistance near saturation.

---

### HA5: Per-joint std initialized identically (all 1.0) but advantage asymmetric across joints → calf drifts faster

**Evidence**:
- `self.std = nn.Parameter(1.0 * torch.ones(12))` — 12 separate parameters, all at 1.0
- Go2 joint ordering: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
- Calf joints (indices 2, 5, 8, 11) have a different kinematic role than hip/thigh — they provide the primary leg stiffness and are often at the end of the kinematic chain
- If the advantage signal for calf joints is consistently higher in one direction (e.g., "extend calf = more forward progress"), the surrogate gradient will push μ_calf in that direction more aggressively than μ_hip/thigh
- Since std is state-independent, σ_calf learns from the aggregated advantage of ALL states at once — which can average out the per-state signal for hip but reinforce the directional signal for calf
- The adaptive LR schedule can amplify this: if global KL is small (stable hip/thigh), LR increases up to 1e-2 (50× initial), accelerating calf drift

**Status**: 🟡 **Consistent as an amplifier** — not a root cause, but explains why calf joints in particular diverge faster than hip/thigh given the same algorithm parameters.

**Falsifiability**: If calf and hip joints have identical advantage distributions, no asymmetric drift would occur. Check advantage per joint over the rollout.

---

### HA6: PPO clip_param + large advantage → ratio clip asymmetric → breaks symmetry

**Evidence**:
- `clip_param = 0.2` → `ratio ∈ [0.8, 1.2]` (ppo_parkour.py:341-346)
- `surrogate_loss = max(surrogate, surrogate_clipped).mean()`
- This is the standard pessimistic PPO objective — it LIMITS policy updates per iteration
- If advantage is large and positive, ratio is clipped at 1.2 → gradient to policy mean is bounded

**Status**: ❌ **Inconsistent as cause** — PPO clipping acts as a governor that LIMITS divergence speed per iteration, not a cause. Over many iterations the mean can still drift if advantage is consistently one-sided.

---

## 4. Additional Finding: `priv_reg_loss` gradient injection

**Code** (`ppo_parkour.py:292-303, 363`):
```python
priv_latent_batch = self.policy.get_priv_latent(obs_batch)          # gradient flows through priv_encoder
with torch.inference_mode():
    hist_latent_batch = self.policy.get_hist_latent(obs_batch)       # detached (target)
priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
# priv_reg_coef ramps 0 → 0.1 between iterations 2000–3000 (hardcoded, not in cfg)

loss = surrogate_loss + value_loss_coef*value_loss - entropy_coef*entropy + priv_reg_coef*priv_reg_loss
```

**Analysis**:
- This additional loss term backpropagates through `priv_encoder` only (priv_latent depends on priv_encoder)
- It does NOT directly affect the actor MLP or the std parameter
- However, since `optimizer` covers `policy.parameters()` (all parameters including actor), and `priv_reg_loss` is part of the total `loss`, gradients from priv_reg_loss DO affect actor parameters through the shared optimizer
- The magnitude at max coef: `0.1 * L2_norm ≈ 0.1 * O(1) = 0.1` — comparable to entropy bonus
- The schedule is hardcoded in `PPOParkour.__init__:124`, not exposed in cfg → cannot be tuned without code change

**Status**: Potentially confounding but low-magnitude. More relevant: this loss is active only after iter 2000, which could correlate with a divergence onset if calf divergence appears around iter 2000-3000.

---

## 5. Adaptive LR Schedule Risk

**Code** (`ppo_parkour.py:325-338`):
```python
if kl_mean > self.desired_kl * 2.0:    # kl > 0.02
    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
elif kl_mean < self.desired_kl / 2.0:  # kl < 0.005
    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
```

**Analysis**:
- LR range: `[1e-5, 1e-2]` — 1000× spread
- Initial LR: 2e-4; max LR: 1e-2 (50× initial)
- The KL is computed GLOBALLY across all joints: if hip/thigh are stable (low KL contribution), the global KL may fall below 0.005 → LR spikes to 1e-2
- At LR=1e-2, the mean gradient step for a calf joint with a large advantage can be substantial
- This is especially concerning in combination with HA3 (raw std can shrink) and HA4 (no gradient barrier at boundary)

**Status**: 🟡 **Consistent as accelerant** — adaptive LR can dramatically increase update magnitude for ALL parameters when global KL appears small, even if specific joints (calf) have large advantage signals that would normally warrant smaller updates.

---

## 6. Summary Ranking of Risks

| Rank | Hypothesis | Mechanism | Status | Evidence Quality |
|------|-----------|-----------|--------|-----------------|
| 1 | **HA3**: raw `std` param — no floor or log reparameterization | σ can drift toward 0; policy becomes deterministic; mean rides advantage unimpeded | 🔴 Consistent | Strong — code path confirmed, no guard anywhere |
| 2 | **HA4**: unbounded actor output + hard clip ±10 with zero gradient | No gradient barrier near ±10; mean can park at boundary | 🔴 Consistent | Strong — MLP has no last_activation; clip is outside gradient graph |
| 3 | **Adaptive LR amplification**: global KL masks per-joint divergence | LR spikes when global KL low, even if calf joint is diverging | 🟡 Partially consistent | Plausible — math checks out; empirical onset unknown |
| 4 | **HA5**: identical init + asymmetric advantage per joint | Calf joints accumulate directional gradient faster | 🟡 Consistent | Plausible — requires advantage log to confirm |
| 5 | **HA2**: entropy_coef=0.01 too small | Weak entropy bonus cannot prevent σ collapse | 🟡 Enabling condition | Consistent but not sufficient alone |
| 6 | **HA1**: init_noise_std=1.0 | Not large enough to hit ±10 from init | ❌ Inconsistent as trigger | — |
| 7 | **HA6**: PPO clip_param breaks symmetry | Clip acts as governor, not cause | ❌ Inconsistent | — |

---

## 7. Falsifiable Test Proposals

> These are analysis suggestions only — NO code changes were made per task rules.

| Test | Change | Expected if hypothesis correct |
|------|--------|-------------------------------|
| T1: Switch to log-std | `noise_std_type="log"` in cfg | HA3 refuted if divergence disappears |
| T2: Add tanh on actor | `last_activation="tanh"` in MLP call (network-worker task) | HA4 refuted if mean stays bounded |
| T3: Raise entropy_coef | `entropy_coef: 0.01 → 0.05` | HA2 confirmed if onset delayed |
| T4: Cap adaptive LR | Change `min(1e-2, ...)` to `min(5e-4, ...)` | Adaptive LR hypothesis confirmed if divergence slows |
| T5: Log per-joint σ | Add per-joint std tracking to logger | Reveals which joint's σ collapses first |

---

## 8. Key File:Line Citations

| Claim | File:Line |
|-------|-----------|
| `init_noise_std = 1.0` | `rsl_rl_ppo_cfg.py:59` |
| `noise_std_type = "scalar"` (default, not overridden) | `rl_cfg.py:31` |
| `state_dependent_std = False` (default) | `rl_cfg.py:34` |
| `self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))` | `actor_critic_parkour.py:214` |
| No clamp/softplus on std | `actor_critic_parkour.py:212-218` (search: nothing) |
| `Normal.set_default_validate_args(False)` | `actor_critic_parkour.py:225` |
| Actor MLP call: no last_activation | `actor_critic_parkour.py:143` |
| MLP last layer is Linear when no last_activation | `mlp.py:67-68` |
| `_update_distribution` — scalar branch | `actor_critic_parkour.py:261-262` |
| `distribution.sample()` — unbounded | `actor_critic_parkour.py:288` |
| `clip_actions = 10.0` | `rsl_rl_ppo_cfg.py:27` |
| Clip applied in wrapper | `vecenv_wrapper.py:163-164` |
| `entropy_coef = 0.01` | `rsl_rl_ppo_cfg.py:73` |
| `clip_param = 0.2` | `rsl_rl_ppo_cfg.py:72` |
| `max_grad_norm = 1.0` | `rsl_rl_ppo_cfg.py:81` |
| Adaptive LR range `[1e-5, 1e-2]` | `ppo_parkour.py:326-328` |
| `priv_reg_coef_schedual = [0, 0.1, 2000, 3000]` (hardcoded) | `ppo_parkour.py:124` |
| `loss = surrogate + value_coef*value - entropy_coef*entropy + priv_reg_coef*priv_reg_loss` | `ppo_parkour.py:359-364` |
| No per-joint branch in actor head | `actor_critic_parkour.py:141-143` (single MLP for all joints) |
