# Action Smoothness Regularization for IsaacLab/rsl_rl PPO — Implementation Spec (CAPS vs LCP)

> Status: **specification only, no code changes**. All file/line references verified against the working tree on branch `main` (2026-06-08).
> Scope: integrate **CAPS** (temporal + spatial smoothness) and **LCP** (Lipschitz gradient penalty) into `rsl_rl`'s feed-forward PPO. Compare, specify both, and recommend which to build first.
>
> **Scope caveat (read first):** this spec targets the **base feed-forward** path — `PPO` (`ppo.py`) + `ActorCritic` (`actor_critic.py`). The project's flagship parkour stack uses **recurrent / history-encoder** policies (`PPOParkour`, `ActorCriticRMA`, `*_parkour_original.py`). Both the CAPS-temporal term (interacts with hidden-state carry-over) and the LCP obs-gradient penalty (input is the history/encoder feature, not raw obs) need **separate treatment there** — do **not** assume drop-in applicability to parkour. If the real training target is parkour, treat that as a follow-up design item.

---

## 0. TL;DR

| | CAPS | LCP |
|---|---|---|
| Paper | Mysore et al., ICRA 2021 (arXiv:2012.06644) | Chen et al. (X.B. Peng group), arXiv:2410.11825 |
| Loss | `λ_T·‖π(s_t)−π(s_{t+1})‖₂ + λ_S·‖π(s)−π(s+N(0,σ))‖₂` | `λ_gp·E[‖∇_obs log π(a\|obs)‖²]` |
| Knobs | 3 (λ_T, λ_S, σ) | 1 (λ_gp) |
| Extra forward passes | spatial: +1; temporal: +1 (and needs s_{t+1}) | +1 backward-through-input (`autograd.grad`) |
| In-repo template | `ppo.py` symmetry-loss block (L319–348) | `ppo_amp.py` gradient-penalty block (L115–140) |
| Main integration risk | **temporal term needs time-adjacent (s_t, s_{t+1}); shuffled minibatch destroys adjacency** | autograd.grad through normalized obs + double-backward cost |
| **Recommendation** | second | **build first** |

---

## 1. Paper verification (citations real and accurate)

### CAPS — arXiv:2012.06644 ✓
- **Authors:** Siddharth Mysore, Bassel Mabsout, Renato Mancuso, Kate Saenko. **Venue:** ICRA 2021. **Title:** *Regularizing Action Policies for Smooth Control with Reinforcement Learning*. Project page: `http://ai.bu.edu/caps/`.
- **Combined objective (Eq. 1):** `J^CAPS_π = J_π − λ_T·L_T − λ_S·L_S`
- **Temporal smoothness (Eq. 2):** `L_T = D(π_θ(s_t), π_θ(s_{t+1}))`
- **Spatial smoothness (Eq. 3):** `L_S = D(π_θ(s_t), π_θ(s̄_t))`, with `s̄ ~ φ(s_t)`, `φ(s)=N(s,σ)`.
- **Distance metric:** Euclidean / L2 — `D(a₁,a₂)=‖a₁−a₂‖₂`. (Verified in text: "we use euclidean distance measures".)
- **σ:** "set based on expected measurement noise and/or tolerance" — application-dependent, no universal constant in the paper body.
- **Default λ_T, λ_S:** The paper does **not** print universal numeric defaults in the main text; it states "an ablation study (available on our website) helped determine good regularization parameters." **No verifiable numeric default could be extracted** (web fetches confirmed the paper defers to the website ablation). Reported community/repo values vary by **orders of magnitude** depending on action/obs scaling, so do **not** copy a literal constant. Instead **calibrate by magnitude** (see §4): pick λ so the smoothness term is ~1–10% of the surrogate loss at init, then sweep around that. For a concrete anchor, the authors' released code (`http://ai.bu.edu/caps/` → GitHub) carries per-task λ values; pull from there if a literal starting point is wanted. σ should match the raw-obs measurement-noise scale.

### LCP — arXiv:2410.11825 ✓
- **Authors:** Zixuan Chen et al. (11 authors incl. **Xue Bin Peng**). **Title:** *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies*. Project page referenced in paper (`lipschitz-constrained-policy.github.io`).
- **Gradient penalty (Eq. 7):** `max_π J(π) − λ_gp·E_{(s,a)~D}[‖∇_s log π(a|s)‖²]`
  - Norm is **L2, squared**. Penalizes the gradient of the **log-probability** `log π(a|s)` w.r.t. the **observation** `s`, averaged over sampled (s,a).
- **Single hyperparameter:** `λ_gp`. **Default = 0.002.** Ablation grid: `{0.0, 0.001, 0.002, 0.005, 0.01}`.
- "Can be easily implemented in any RL framework, requiring only a few lines of code" — consistent with reusing the existing `torch.autograd.grad` pattern.

### Trade-off (verified)
- **CAPS = 2 terms / 2–3 knobs.** Temporal term is the conceptually correct smoothness objective (penalizes action change across *consecutive real timesteps*), but is the hardest to wire into a shuffled PPO minibatch (see §3.1). Spatial term is cheap and trivially compatible (one extra forward on noised obs, identical to the symmetry-loss pattern).
- **LCP = 1 term / 1 knob.** No temporal coupling, no dependence on storage ordering — operates purely on the current minibatch. Slightly higher per-sample compute (gradient-through-input + double backward) but structurally simpler and matches an existing in-repo pattern (`ppo_amp.py`).

---

## 2. Codebase mapping (verified file:line)

### 2.1 PPO loss assembly — `rsl_rl/rsl_rl/algorithms/ppo.py`
- `update()` defined L198. Minibatch loop L214–395.
- Recompute current-policy stats: `self.policy.act(obs_batch, …)` L254; `mu_batch = self.policy.action_mean[:original_batch_size]` L258.
- **Total loss assembled L317:** `loss = surrogate_loss + value_loss_coef*value_loss − entropy_coef*entropy`.
- **Symmetry-loss block L319–348** — the exact template for CAPS-spatial:
  - L325: re-augments obs via `data_augmentation_func` (CAPS analog: add Gaussian noise to obs).
  - L330: `mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())` — one extra forward returning the **mean action** (no sampling).
  - L342–345: `mse_loss(...)` between mean actions.
  - L348: `loss += coeff * symmetry_loss`. **CAPS-spatial would add its term identically here.**
- Backward/step L368–381 (`loss.backward()` L369). New loss terms must be folded into `loss` **before** L369.
- Logging dict L411–419 — add `caps_temporal`, `caps_spatial`, or `lipschitz` keys here.

### 2.2 Gradient-penalty template — `rsl_rl/rsl_rl/algorithms/ppo_amp.py`
- AMP disc gradient penalty L115–140 (mirrored at L307–330):
  - L116: `expert_data = expert_batch.detach().requires_grad_(True)`
  - L118–125: `torch.autograd.grad(outputs=logits, inputs=data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]`
  - L126: `gradients.norm(2, dim=1).pow(2).mean()` — **exactly the L2-squared-norm penalty LCP needs**, applied to the discriminator. LCP moves this from `disc.get_logits(x)` to `policy mean/log-prob(obs)`.
- Confirms `create_graph=True` (needed so the penalty is itself differentiable w.r.t. policy params — double backward).

### 2.3 Policy forward surface — `rsl_rl/rsl_rl/modules/actor_critic.py`
- `act_inference(obs)` L158–164 → returns the **policy mean** (`self.actor(obs)`); already used by symmetry block. Use for CAPS spatial/temporal mean comparison.
- `get_actor_obs(obs)` L171–173 → concatenates obs groups; `act()` L152–156 normalizes via `actor_obs_normalizer` then `_update_distribution`. **Normalization happens inside the policy**, so obs perturbation/grad must account for where it is injected (see §3.3).
- `action_mean` property returns `self.distribution.mean` after `act()`.

### 2.4 Rollout storage — `rsl_rl/rsl_rl/storage/rollout_storage.py`  ← **CAPS temporal crux**
- `observations` buffer shape `[num_transitions_per_env (T), num_envs, *obs]`, **time-ordered**, written per step at `self.observations[self.step].copy_(...)` L97.
- `mini_batch_generator()` L131: `observations = self.observations.flatten(0, 1)` L139, then indexed by **shuffled** `indices = torch.randperm(...)` L136 (`batch_idx`, L155–158).
- **No `next_observations` are stored** (grep confirmed: only L97/L128/L158 touch `observations`; `Transition` L28–42 has no next-obs field).
- ⇒ Inside a minibatch, `s_t` and `s_{t+1}` are **not** co-located and the time index is lost after shuffle.

### 2.5 Config dataclass — `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`
- `RslRlPpoAlgorithmCfg` L75–129. Nested optional cfgs (`rnd_cfg` L125, `symmetry_cfg` L128) are the pattern for a new `caps_cfg`/`lcp_cfg` field.
- Runner forwards `**self.alg_cfg` to PPO ctor: `on_policy_runner.py` L287–288 → any new ctor kwarg must have a matching cfg field (and PPO `__init__` kwarg).
- `symmetry_cfg.py` (full file) is the template for a small nested `@configclass`.

---

## 3. Implementation specs

### 3.1 CAPS — temporal term `λ_T·‖π(s_t)−π(s_{t+1})‖₂`  (THE integration problem)

**Requirement:** for each transition, the mean action at `s_t` and at the *real next* state `s_{t+1}` of the same env, then L2 distance.

**Why the current minibatch can't supply it:** `mini_batch_generator` flattens `[T,num_envs]→[T·num_envs]` and shuffles with `randperm` (L136/L139/L155). After shuffle, neighboring rows are unrelated; the `(t, env)` index is discarded. So `s_{t+1}` is not derivable inside the loop body.

**Three viable solutions (in preference order):**

1. **Store `next_observations` explicitly (recommended if temporal is wanted).**
   - Add a `next_observations` buffer to `RolloutStorage` (same shape/dtype as `observations`), populate it in `process_env_step`/`add_transition` from the post-step obs (the env's returned obs is already available in the runner's collection loop — it is the `obs` passed to the *next* `act`). Mask out rows where `dones[t]=1` (next state belongs to a fresh episode → exclude from temporal penalty).
   - Thread `next_obs_batch` through `mini_batch_generator`'s yield (same `batch_idx` gather as `observations`, L158-style) — shuffling is now **fine** because each row carries its own paired next-obs.
   - In `update()`: `mu_next = policy.act_inference(next_obs_batch.detach())`; `L_T = ((mu_batch − mu_next)*valid_mask).norm(dim=-1).mean()`; `loss += λ_T*L_T` near L348.
   - Cost: +1 obs-sized buffer (memory), +1 forward pass, generator signature change. **This is the clean, correct integration.**

2. **Next-obs approximation via stored adjacency in the *unshuffled* buffer (no new buffer, but a separate pass).**
   - Compute temporal loss in a small auxiliary pass over the **time-ordered** `self.observations` (before flatten): `mu[t]=act_inference(obs[t])`, penalize `‖mu[:-1]−mu[1:]‖` with `dones[:-1]==0` mask. Accumulate into the loss outside the shuffled minibatch loop, or as its own mini-loop. Avoids storing next-obs but breaks the "everything inside the minibatch loop" structure and recomputes forwards on the full rollout. Less clean; only if memory is tight.

3. **Drop the temporal term; ship CAPS-spatial only.** Spatial smoothness alone already removes most high-frequency jitter in practice and is trivial to integrate (§3.2). Reasonable MVP.

> **Conclusion on the task's key question:** CAPS-temporal **is integrable**, but **not** by reading `s_{t+1}` from the shuffled minibatch. It requires either (a) adding a paired `next_observations` buffer to `RolloutStorage` and threading it through the generator (recommended), or (b) a separate unshuffled pass over the time-ordered buffer. The naive "grab next row in the minibatch" approach is **impossible** because of the `randperm` shuffle at `rollout_storage.py:136`.

### 3.2 CAPS — spatial term `λ_S·‖π(s)−π(s+N(0,σ))‖₂`  (trivial — mirrors symmetry block)

- After the existing loss is assembled (~L317), add (template = symmetry block L330/L342/L348):
  ```
  mu_clean   = policy.act_inference(obs_batch.detach())            # or reuse mu_batch
  noisy_obs  = perturb(obs_batch, sigma)                            # add N(0,σ) to obs tensor(s)
  mu_noisy   = policy.act_inference(noisy_obs)
  L_S        = (mu_clean - mu_noisy).norm(dim=-1).mean()
  loss      += lambda_S * L_S
  ```
- **σ injection point (important):** obs is normalized *inside* `act_inference` (`actor_obs_normalizer`, L154/L160). Add the noise to the **raw obs** so σ is in raw-obs units (paper: noise reflects sensor/measurement scale). If you instead want σ in normalized units, perturb after normalization — but that requires touching the module; raw-obs perturbation needs no module change and is the recommended path.
- Perturb only the **policy obs groups** (`obs_groups["policy"]`), matching what the actor consumes.
- Cost: +1 forward pass per minibatch (identical order-of-magnitude to symmetry mirror loss).

### 3.3 CAPS — config additions
Add a nested `@configclass RslRlCapsCfg` (template: `symmetry_cfg.py`):
```
lambda_temporal: float = 0.0   # λ_T, 0 disables temporal
lambda_spatial:  float = 0.0   # λ_S, 0 disables spatial
sigma:           float = 0.05  # raw-obs-space std for spatial noise
```
- Add `caps_cfg: RslRlCapsCfg | None = None` to `RslRlPpoAlgorithmCfg` after L129 (next to `symmetry_cfg`).
- Add `caps_cfg: dict | None = None` kwarg to `PPO.__init__` (next to `symmetry_cfg`, L52); store parsed values.
- Log `caps_temporal`, `caps_spatial` into the loss dict (L411–419).

### 3.4 LCP — `λ_gp·E[‖∇_obs log π(a|obs)‖²]`  (mirrors AMP grad-penalty)

- Inside the minibatch loop, after `policy.act(obs_batch)` has built the distribution (so log-prob is available), compute the penalty using the **AMP template** (`ppo_amp.py` L116–126):
  ```
  obs_in = actor_obs(obs_batch).detach().requires_grad_(True)   # leaf w/ grad on the actor-input tensor
  logp   = log_prob_of_actions(obs_in, actions_batch)           # log π(a|obs) for THIS obs_in
  grads  = torch.autograd.grad(
              outputs=logp.sum(), inputs=obs_in,
              create_graph=True, retain_graph=True, only_inputs=True)[0]
  L_lip  = grads.norm(2, dim=-1).pow(2).mean()                  # ‖∇ log π‖², L2 squared
  loss  += lambda_gp * L_lip
  ```
- **Exact location:** add this block right before total `loss.backward()` (L369), folding `L_lip` into `loss` (like symmetry L348). `create_graph=True` is mandatory (penalty must be differentiable w.r.t. policy params — same as AMP L118-121).
- **Key wiring detail:** the penalty needs `log π(a|obs)` **as a function of a grad-enabled obs input**. The current `act()` normalizes obs internally and discards the pre-norm tensor. Cleanest is to make `obs_in` the **actor-input tensor** (`get_actor_obs(obs_batch)` output, optionally pre/post-normalizer per choice) marked `requires_grad_(True)`, run the actor forward on it to get `mean,std`, build `Normal`, and take `log_prob(actions_batch).sum(-1)`. This may need a tiny helper on the policy (e.g. `log_prob_from_actor_obs(obs_in, actions)`) — analogous to how AMP calls `discriminator.get_logits(data)` on a grad-enabled leaf. No change to the public `act()` path.
- **Verify the form against released code:** the Eq.7 extraction here is from a small-model web fetch of the PDF/HTML; before implementing, cross-check `λ_gp=0.002` and the `∇log π` (vs `∇μ`) form against the authors' released code at `lipschitz-constrained-policy.github.io`.
- **Paper-faithful vs cheap variant:** Eq.7 uses `∇ log π(a|s)`. A common simplification is penalizing `∇_obs μ(obs)` (the **mean** action) instead of the full log-prob — cheaper, no `actions_batch` needed, and directly bounds the Lipschitz constant of the deterministic policy map. If reproducing the paper exactly, use log-prob; if you only care about smoothing the action map for deploy, `∇μ` is a defensible, lighter choice. Spec both; default to **log-prob (paper-faithful)**.

### 3.5 LCP — config additions
```
@configclass RslRlLcpCfg:
    lambda_gp: float = 0.002      # paper default
    penalize: Literal["log_prob","mean"] = "log_prob"
```
- Add `lcp_cfg: RslRlLcpCfg | None = None` to `RslRlPpoAlgorithmCfg` after L129.
- Add `lcp_cfg` kwarg to `PPO.__init__`; log `lipschitz` into loss dict.

---

## 4. Verification / experiment plan + deploy

### A/B protocol
- **Baseline:** current PPO on the target task (e.g. Go2 velocity / parkour-flat). Fix seed set (≥3 seeds).
- **Arms:** (1) baseline, (2) LCP λ_gp∈{0.001, 0.002, 0.005} (paper-anchored), (3) CAPS-spatial **magnitude-calibrated λ_S** (see below) σ∈{0.02,0.05}, (4) CAPS full (+λ_T) if temporal implemented.
- **CAPS λ calibration (because λ is scale-dependent and has no verifiable literal default):** at iteration 0, measure the un-weighted smoothness term `L_S` (and `L_T`) and the surrogate loss; set λ so `λ·L ≈ 1–10%` of the surrogate-loss magnitude, then sweep ±1 order of magnitude (e.g. {0.1×, 1×, 10×} of the calibrated value). **Do not hardcode λ_S=1 or λ_S=400** — either may be off by ~100× for this codebase's action scaling and would make the CAPS arm look falsely weak (no smoothing) or falsely destructive (return collapse).
- **Smoothness metrics (log per arm):**
  - **Action rate** `‖a_t − a_{t−1}‖` mean/95th-pct (primary).
  - **Jerk / 2nd diff** `‖a_t − 2a_{t−1} + a_{t−2}‖`.
  - Action-signal FFT high-frequency power (CAPS paper's primary motivation) if cheap.
  - Torque/energy proxy and (sim) joint-velocity oscillation.
- **Reward retention:** task return must stay within ~X% of baseline; a smoothness method that tanks tracking is a fail. Watch command-tracking error specifically.

### Over-smoothing risk (agile/parkour) → schedulable λ
- Strong smoothing **slows policy reaction** — bad for parkour jumps / fast direction changes. Mitigation, **recommended**: make λ (λ_T/λ_S/λ_gp) **schedulable** (e.g. linear warm-up from 0 over the first N iters, or anneal up so early exploration is unconstrained and late-training motion is smoothed). Implement as a getter that reads iteration count, mirroring how AMP `task_reward_lerp` is annealed (`enable_lerp_schedule`, ppo_amp.py L57/L251). At minimum expose λ as a plain cfg float for manual sweeps before adding a scheduler.
- For parkour specifically: prefer **spatial-only / LCP with small λ** first; the temporal term most directly fights rapid intentional action changes.

### Deploy (sim-to-real) — quantitative expectation
- Both methods directly reduce **high-frequency actuator chatter** → lower mechanical wear, lower current draw, cooler motors. CAPS paper reports up to **~80% power reduction** on quadrotors; LCP paper demonstrates deployable smooth humanoid locomotion on hardware where the unconstrained policy was jittery/undeployable.
- Expect: reduced action-rate/jerk (target ≥30–50% drop in action-rate 95th-pct at a λ that costs <few-% return), lower joint-velocity oscillation, smaller sim-to-real gap from jitter that the real actuators can't track. This complements (does not replace) existing action-rate reward penalties; it adds a **gradient-shaping** smoothness pressure rather than a reward term.

---

## 5. Recommendation — build **LCP first**

**Why LCP before CAPS:**
1. **One knob (λ_gp=0.002) vs three (λ_T, λ_S, σ).** Far smaller tuning surface; the paper hands you a working default and an ablation grid.
2. **No storage/ordering dependency.** LCP operates entirely within the existing shuffled minibatch. CAPS-temporal needs a new `next_observations` buffer + generator signature change (§3.1) — strictly more invasive and the riskiest part of CAPS.
3. **A drop-in in-repo template already exists** — `ppo_amp.py`'s `torch.autograd.grad(... create_graph=True ...).norm(2).pow(2).mean()` is *exactly* the LCP penalty; it's a re-target from discriminator-logits to policy-log-prob, not new machinery.
4. **Strongest direct evidence for this codebase's domain** — LCP is from the X.B. Peng group, validated on *legged/humanoid locomotion* with Isaac-style massively-parallel PPO, i.e. the closest match to Go2/parkour here.

**Then add CAPS-spatial** as a cheap second method (it's just the symmetry-loss block with Gaussian obs noise — near-zero integration risk) for an A/B against LCP. **Defer CAPS-temporal** until spatial/LCP results justify the storage change; if pursued, do it via the `next_observations` buffer (§3.1 option 1), not via minibatch adjacency.

---

## Appendix — exact insertion points (quick reference)

| Change | File:line |
|---|---|
| Fold smoothness term into `loss` | `rsl_rl/rsl_rl/algorithms/ppo.py:317` (after) / before `:369` |
| Spatial template to copy | `ppo.py:330` (`act_inference`), `:342–348` (mse + `loss +=`) |
| Grad-penalty template to copy | `ppo_amp.py:116–126` |
| Loss-dict logging | `ppo.py:411–419` |
| Temporal needs next-obs buffer | `rollout_storage.py:97` (write), `:139/:155–158` (generator gather), `Transition` `:28–42` |
| PPO ctor kwargs | `ppo.py:31–55` (next to `symmetry_cfg`) |
| Cfg dataclass field | `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:128–129` (after `symmetry_cfg`) |
| Nested cfg template | `source/isaaclab_rl/isaaclab_rl/rsl_rl/symmetry_cfg.py` |
| Policy mean / log-prob hooks | `actor_critic.py:152–164` (`act`, `act_inference`), `:179` (`get_actions_log_prob`) |
