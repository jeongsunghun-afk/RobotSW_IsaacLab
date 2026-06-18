# RL Algorithm / Loss / Training-Technique Research for Legged Locomotion (rsl_rl)

**Date**: 2026-06-08
**Scope**: New RL algorithms, loss terms, and training techniques worth adding to this IsaacLab/rsl_rl codebase.
**Companion**: `00_codebase_inventory.md` (what already exists — used as the exclusion list).

---

## 0. Method (how candidates were ranked)

The decisive question for integration cost was read off the actual PPO `update()` surface
(`rsl_rl/rsl_rl/algorithms/ppo.py:298-369`):

> **Can the method be expressed as a term added to `loss` before `loss.backward()`, using quantities already in scope?**
> In-scope quantities: `mu_batch`, `sigma_batch`, `self.policy.action_mean/action_std`, `actions_log_prob_batch`,
> `old_actions_log_prob_batch`, `ratio`, `advantages_batch`, `value_batch`, `returns_batch`, `obs_batch`.

- **Yes** → **Group A** (drop-in loss term / advantage or normalization tweak). Low cost.
- **Needs replay buffer, off-policy sample regime, or a new runner** → **Group B**. High cost.

Ranking is by **impact / effort *for this on-policy, thousands-parallel-env, Go2-sim-to-real framework***, NOT by paper benchmark numbers. Off-policy methods are included but honestly down-ranked because they break the massively-parallel on-policy sample-collection story.

Each candidate states its **delta from the already-implemented near-miss** (the user pre-loaded the exclusion list with traps like "adaptive-KL-LR ≠ KL-penalty loss").

---

## 1. RANKED SHORTLIST

### #1 — CAPS (Conditioning for Action Policy Smoothness)

- **Method**: Two action-smoothness regularizers added to the policy loss — *temporal* (action at `s_t` ≈ action at `s_{t+1}`) and *spatial* (nearby states → nearby actions, evaluated at `s + N(0,σ)`).
- **Paper**: *Regularizing Action Policies for Smooth Control with Reinforcement Learning* — Siddharth Mysore, Bassel Mabsout, Renato Mancuso, Kate Saenko — ICRA 2021 — arXiv:2012.06644 — https://arxiv.org/abs/2012.06644
- **무엇을 바꾸나**: Adds `λ_T·‖π(s_t) − π(s_{t+1})‖ + λ_S·‖π(s_t) − π(s̄)‖` (s̄ = state + Gaussian noise) to the loss.
- **legged 적용성**: **High** — purpose-built for sim-to-real motor control; 80% power reduction + reliably flight-worthy controllers on a real quadrotor. Smooth actions are the #1 sim-to-real ask for legged robots (jitter destroys real-actuator transfer). Spatial term doubles as observation-noise robustness.
- **rsl_rl 통합 비용**: **Low (drop-in loss term).** Temporal term needs consecutive `mu_t, mu_{t+1}` — available because RolloutStorage is time-ordered; or approximate via mini-batch mean-action of next-step obs. Spatial term: one extra `act_inference(obs_batch + noise)` forward (the symmetry-loss block at `ppo.py:319` is the exact template — extra forward + MSE added to `loss`). Touches: `ppo.py` update loop + a `caps_cfg` block in the algorithm cfg.
- **이미 존재?**: **No.** Not the same as Symmetry loss (mirror equivariance, not temporal/spatial continuity) nor RND.
- **public 구현?**: Yes — project page https://ai.bu.edu/caps/ ; widely reproduced in legged-gym forks.
- **deploy 코멘트**: Strongest deploy upside of the whole list. Directly reduces motor oscillation / energy / wear on Go2. Risk: over-smoothing can blunt agile/parkour responses → keep `λ` small and schedulable.

---

### #2 — Lipschitz-Constrained Policies (LCP) for smooth locomotion

- **Method**: Replace explicit smoothness penalties with a **gradient penalty on the policy network** that bounds its local Lipschitz constant, giving differentiable smoothness without tuning separate temporal/spatial coefficients.
- **Paper**: *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies* — Zixuan Chen, Xialin He, Yen-Jen Wang, Qiayuan Liao, Yanjie Ze, Zhongyu Li, S. Shankar Sastry, Jiajun Wu, Koushil Sreenath, Saurabh Gupta, **Xue Bin Peng** — arXiv:2410.11825 (Oct 2024) — https://arxiv.org/abs/2410.11825 — project: https://lipschitz-constrained-policy.github.io/
- **무엇을 바꾸나**: Adds `λ·‖∇_obs π(obs)‖²`-style gradient penalty to the policy loss (1 hyperparameter) in place of CAPS' two terms.
- **legged 적용성**: **High** — born in legged locomotion; validated on **real humanoid** sim-to-real by the AMP/Xue Bin Peng group. Single knob is attractive vs CAPS' two.
- **rsl_rl 통합 비용**: **Low (drop-in loss term).** The gradient-penalty machinery already exists in this repo — AMP uses `torch.autograd.grad` gradient penalty (`ppo_amp.py`); reuse that pattern on the *policy* output w.r.t. obs. Touches: `ppo.py` + `lcp_cfg`.
- **이미 존재?**: **No** for the policy. Gradient penalty exists only on the **AMP discriminator**, not on the actor — different network, different purpose.
- **public 구현?**: Yes — official repo linked from project page.
- **deploy 코멘트**: Same sim-to-real smoothness win as CAPS with fewer knobs and a legged provenance. Slightly higher per-step compute (autograd grad). **#1 and #2 are alternatives — pick one to start (recommend CAPS for simplicity, LCP if you want a single knob).**

---

### #3 — SPO (Simple Policy Optimization)

- **Method**: Replace PPO's **ratio clipping** with **KL-divergence clipping**: reweight the surrogate by `d_clip/d` where `d = KL(π_old‖π_θ)`, `d_clip = clip(d, 0, d_max)`. Unlike PPO, samples outside the trust region keep a non-zero (penalized) gradient instead of being zeroed.
- **Paper**: *Simple Policy Optimization* — Zhengpeng Xie, Qiang Zhang, Fan Yang, **Marco Hutter**, Renjing Xu — ICML 2025 (poster) — arXiv:2401.16025 — https://arxiv.org/abs/2401.16025 — OpenReview: https://openreview.net/forum?id=SG8Yx1FyeU
- **무엇을 바꾸나**: Surrogate becomes (per paper Eq. 14): `Â>0 → ratio·Â·(d_clip/d)`; `Â<0 → ratio·Â·(−d_clip/d + 2)`. KL `d` computed from the (already-available) old/new Gaussian params.
- **legged 적용성**: **Med–High** — co-authored by **Marco Hutter** (ETH legged-robotics); claims robustness to deep/complex nets and lower KL at higher sample efficiency, which matters for the large RMA/parkour nets here. Locomotion-specific benchmark evidence is still thin (general-RL paper), hence not #1.
- **rsl_rl 통합 비용**: **Low (drop-in surrogate swap).** This repo **already computes the exact KL** (`ppo.py:264-272`) for the adaptive-LR schedule — reuse it to weight the surrogate at `ppo.py:298-304`. No new networks, no buffer. Touches: ~15 lines in `ppo.py` + a `surrogate_type` flag. Cleanest possible swap.
- **이미 존재?**: **No.** The repo has adaptive-**KL-based LR** (only nudges the LR) — SPO instead puts the KL **into the loss/surrogate**. Genuinely distinct from the exclusion list.
- **public 구현?**: Yes — https://github.com/MyRepositories-hub/Simple-Policy-Optimization
- **deploy 코멘트**: Algorithm-internal; no deploy-side change. Low-risk A/B vs current PPO since it shares all infra. **Best effort-adjusted bet for the user's explicit "PPO 변형" interest.**

---

### #4 — PopArt value/return normalization

- **Method**: Adaptively rescale value-function **targets** to zero-mean/unit-variance while **preserving the network outputs** (output layer weights/bias rescaled in lockstep), keeping value learning stable across reward-scale changes.
- **Paper**: *Learning Values Across Many Orders of Magnitude* — Hado van Hasselt, Arthur Guez, Matteo Hessel, Volodymyr Mnih, David Silver — NeurIPS 2016 — https://proceedings.neurips.cc/paper/6076-learning-values-across-many-orders-of-magnitude.pdf
- **무엇을 바꾸나**: Wraps the critic head with running target statistics; value loss is computed on normalized returns, undone at inference.
- **legged 적용성**: **Med** — legged reward functions are sums of many terms (tracking + smoothness + air-time + penalties) spanning very different scales and **changing under reward-shaping iterations** (exactly this project's workflow). PopArt removes manual `value_loss_coef` / reward-scale re-tuning each iteration.
- **rsl_rl 통합 비용**: **Low–Med.** Output normalization (`EmpiricalNormalization`) exists at `networks/normalization.py`; PopArt adds the **preserving-outputs weight rescale** on the critic's last layer + normalized value-loss in `ppo.py:306-315`. Touches: critic head wrapper + value-loss block. No buffer/runner change.
- **이미 존재?**: **No.** `EmpiricalDiscountedVariationNormalization` normalizes **rewards** (RND); PopArt normalizes **value targets with output preservation** — different mechanism and goal.
- **public 구현?**: Yes — many (CleanRL/SB3-contrib references, original DeepMind).
- **deploy 코멘트**: Neutral to deploy; a workflow-stability win given frequent reward re-shaping in this repo.

---

### #5 — PPG (Phasic Policy Gradient)

- **Method**: Separate policy and value optimization into **phases** — a policy phase, then an auxiliary distillation phase that lets the value head be optimized with higher sample reuse while distilling value features into a shared trunk (best of shared-vs-separate networks).
- **Paper**: *Phasic Policy Gradient* — Karl Cobbe, Jacob Hilton, Oleg Klimov, John Schulman — ICML 2021 — arXiv:2009.04416 — https://arxiv.org/abs/2009.04416 — code: https://github.com/openai/phasic-policy-gradient
- **무엇을 바꾸나**: Adds a periodic auxiliary phase reusing stored rollouts; needs an aux value head + behavior-clone-to-old-policy term.
- **legged 적용성**: **Med** — improves sample efficiency/feature sharing; benchmark evidence is Procgen (vision/generalization), with limited direct legged evidence. Most relevant if/when shared actor-critic trunks are adopted (current default uses separate actor/critic MLPs, blunting PPG's core benefit).
- **rsl_rl 통합 비용**: **Med.** No replay buffer, but needs an aux training phase over retained rollouts + KL/BC constraint to old policy + extra value head — more than a drop-in loss term, less than a new family. Touches: `ppo.py` update structure + a light runner change to trigger the aux phase.
- **이미 존재?**: **No.**
- **public 구현?**: Yes — official OpenAI repo + "PPG Reloaded" (ICML 2023) ablation.
- **deploy 코멘트**: Algorithm-internal, deploy-neutral. Medium effort with uncertain legged payoff → lower priority than #1-#4.

---

### #6 — Dual-Clip PPO

- **Method**: Add a **lower clip** `c` (c>1) on the surrogate when `Â<0`, preventing the policy gradient from exploding on large-negative-advantage samples (`max(min(ratio·Â, clip·Â), c·Â)`).
- **Paper**: *Mastering Complex Control in MOBA Games with Deep Reinforcement Learning* — Deheng Ye et al. — AAAI 2020 — https://arxiv.org/abs/1912.09729 (dual-clip is the contributed PPO modification). Concept also surfaced in DAPO (arXiv:2503.14476) as decoupled low/high clip.
- **무엇을 바꾸나**: One extra `torch.max(..., c*Â)` line in the surrogate when advantage is negative.
- **legged 적용성**: **Med** — cheap stability guard against occasional huge negative advantages (common with sparse penalties / early-termination spikes in locomotion). Low theoretical novelty; pure robustness.
- **rsl_rl 통합 비용**: **Low (drop-in, ~2 lines)** at `ppo.py:298-304`. Touches: surrogate block + a `dual_clip_c` cfg field.
- **이미 존재?**: **No** — repo has single (upper) clip only. Distinct from clipped surrogate.
- **public 구현?**: Yes — Tianshou, many PPO libs.
- **deploy 코멘트**: Deploy-neutral. A cheap "free stability" add-on; bundle with SPO trial, not a standalone headline.

---

### #7 — CrossQ (representative cheap off-policy, IF off-policy is ever pursued)

- **Method**: SAC + BatchNorm in the critic + **removal of target networks**, reaching REDQ/DroQ-level sample efficiency at UTD=1 (no 20× critic updates).
- **Paper**: *CrossQ: Batch Normalization in Deep RL for Greater Sample Efficiency and Simplicity* — Aditya Bhatt, Daniel Palenicek, Boris Belousov, Max Argus, Artemij Amiranashvili, Thomas Brox, Jan Peters — ICLR 2024 — arXiv:1902.05605 — https://arxiv.org/abs/1902.05605
- **무엇을 바꾸나**: Off-policy actor-critic; "few lines on top of SAC" but **requires the SAC family** (replay buffer, twin critics, entropy temperature).
- **legged 적용성**: **Med (real-robot), Low (this framework).** Off-policy SAC variants shine for *real-world* minute-scale learning (Minitaur, A1), but that regime is the opposite of this codebase's thousands-of-parallel-sim-env on-policy regime.
- **rsl_rl 통합 비용**: **High — new algorithm family.** Needs a replay buffer (none exists for RL — AMP's buffer is discriminator-only), twin Q-critics, entropy temperature, and a **new off-policy runner**. RolloutStorage's GAE pipeline is unused. Touches: new `storage/replay_buffer.py`, new `algorithms/sac.py`/`crossq.py`, new `runners/off_policy_runner.py`.
- **이미 존재?**: **No** (whole family absent).
- **public 구현?**: Yes — official JAX/PyTorch; SB3-contrib has SAC/TQC; DroQ/REDQ public.
- **deploy 코멘트**: Deploy-neutral once trained. **Do NOT over-rate on benchmark sample-efficiency** — with thousands of parallel envs the on-policy wall-clock is already excellent, so CrossQ's per-sample efficiency advantage mostly disappears here. Cheapest *entry point* if off-policy is ever wanted (UTD=1, no target net) — pick CrossQ over DroQ/REDQ to minimize compute.

---

### #8 — SAC / TD3 (the canonical off-policy baselines, for completeness)

- **Method**: Off-policy actor-critic. SAC = max-entropy stochastic policy + twin critics; TD3 = deterministic policy + twin critics + delayed actor + target smoothing.
- **Papers**: SAC — Haarnoja et al., ICML 2018, arXiv:1801.01290 — https://arxiv.org/abs/1801.01290 ; TD3 — Fujimoto et al., ICML 2018, arXiv:1802.09477 — https://arxiv.org/abs/1802.09477. Real-quadruped uses: *Learning to Walk via Deep RL* (Haarnoja et al., RSS 2019, arXiv:1812.11103); *Learning to Walk in the Real World with Minimal Human Effort* (Ha et al., CoRL 2020); *Learning to Walk in 20 Minutes* (Smith et al., RSS 2023, DroQ-based, https://www.roboticsproceedings.org/rss19/p056.pdf).
- **무엇을 바꾸나**: Entirely different sample/update regime (per-step replay updates vs synchronized rollouts).
- **legged 적용성**: **Med (real-world minute-scale)** but **Low for this sim-massively-parallel framework**.
- **rsl_rl 통합 비용**: **High — new family + replay buffer + new runner** (same surgery as #7).
- **이미 존재?**: **No.**
- **public 구현?**: Yes — SB3, CleanRL, rl_games.
- **deploy 코멘트**: Use only if the project pivots to **on-robot fine-tuning / real-world minute-scale learning**. For sim training at 4096 envs, PPO+drop-ins dominate on effort-adjusted value.

---

## 2. NOT RECOMMENDED (evaluated, dropped — with reason)

| Method | Why dropped |
|--------|-------------|
| **KL-penalty PPO (fixed/adaptive-β)** | Subsumed by SPO (#3), which is the modern, legged-authored KL-in-loss variant with public code. Recommending plain KL-penalty would be strictly weaker. |
| **AWR / V-trace** | AWR is off-policy advantage-weighted regression (replay) → Group-B cost without a clear win over on-policy PPO here. V-trace is for **off-policy/asynchronous** actors (IMPALA); this framework's rollouts are on-policy & synchronous, so V-trace's importance correction adds machinery for no benefit. |
| **TQC / REDQ / DroQ** | Same Group-B off-policy cost as CrossQ but **more** compute (TQC=quantile critics; REDQ/DroQ=high UTD ensembles). CrossQ (#7) is the cheaper representative; listing all would be padding. |
| **Gradient-penalty-on-policy (generic)** | Already captured by LCP (#2) as the legged-specific, validated instance. |
| **PPG-style "decouple value/policy" for generalization** (arXiv:2102.10330) | Targets Procgen generalization, not locomotion; overlaps PPG (#5). |

---

## 3. GROUP CLASSIFICATION + RECOMMENDATIONS

### Group A — Drop-in to existing PPO (cheap: loss term / surrogate / normalization)
Expressible as a term added to `loss` before `backward()` using in-scope quantities. No runner/buffer change.

| Rank | Method | Effort | Primary win |
|------|--------|--------|-------------|
| A1 | **CAPS** (#1) | Low | Sim-to-real action smoothness (deploy) |
| A2 | **LCP** (#2) | Low | Same smoothness, 1 knob, legged-validated |
| A3 | **SPO** (#3) | Low | Better PPO trust region, reuses existing KL |
| A4 | **PopArt** (#4) | Low–Med | Value stability across reward re-shaping |
| A5 | **Dual-Clip PPO** (#6) | Low (~2 lines) | Cheap negative-advantage stability guard |
| (A) | **PPG** (#5) | Med | Sample efficiency — only if shared trunk adopted |

**Group A top recommendation (impact/effort):**
1. **CAPS or LCP** — highest *deploy* impact (Go2 sim-to-real smoothness) at drop-in cost. Start with **CAPS** for transparency/control; switch to **LCP** if you want a single hyperparameter. Mutually substitutable — run one.
2. **SPO** — highest *algorithmic* leverage for the user's explicit "PPO 변형" interest, at the lowest possible effort (the KL is already computed at `ppo.py:264`). Ideal low-risk A/B against current PPO.
3. Bundle **Dual-Clip** + **PopArt** as cheap stability/robustness add-ons alongside the above.

### Group B — New algorithm family (expensive: replay buffer + new runner + sample-regime change)

| Method | Effort | Note |
|--------|--------|------|
| **CrossQ** (#7) | High | Cheapest off-policy entry (UTD=1, no target net) |
| **SAC / TD3** (#8) | High | Canonical baselines; real-world minute-scale niche |
| **TQC/REDQ/DroQ** | High+ | More compute than CrossQ for similar/less benefit here |

**Group B top recommendation:** **Defer all of Group B.** With thousands of parallel sim envs, on-policy PPO already achieves excellent wall-clock; off-policy's per-sample efficiency edge largely cancels, while the cost (new replay buffer, twin critics, off-policy runner, abandoning the GAE/RolloutStorage path) is large and touches core infra. If the project later pivots to **on-robot / real-world fine-tuning**, start with **CrossQ** (simplest, lowest compute) and consult the real-quadruped SAC/DroQ precedents (Smith 2023, Ha 2020).

---

## 4. ONE-LINE BOTTOM LINE

For this on-policy, massively-parallel, Go2-sim-to-real framework, the high-value adds are **drop-in PPO loss terms**: **CAPS/LCP** (action smoothness → sim-to-real) and **SPO** (better trust region, reuses the KL already computed), with **PopArt** + **dual-clip** as cheap stability. **Off-policy (SAC/TD3/CrossQ) is correctly down-ranked** — strong on paper but a poor effort/impact fit until the project pivots to real-world fine-tuning.

---

## 5. CITATIONS (verified from search/fetch this session)

- CAPS — arXiv:2012.06644 — https://arxiv.org/abs/2012.06644 ; https://ai.bu.edu/caps/
- LCP — arXiv:2410.11825 — https://arxiv.org/abs/2410.11825 ; https://lipschitz-constrained-policy.github.io/
- SPO — arXiv:2401.16025 — https://arxiv.org/abs/2401.16025 ; https://openreview.net/forum?id=SG8Yx1FyeU ; code https://github.com/MyRepositories-hub/Simple-Policy-Optimization
- PopArt — NeurIPS 2016 — https://proceedings.neurips.cc/paper/6076-learning-values-across-many-orders-of-magnitude.pdf
- PPG — arXiv:2009.04416 — https://arxiv.org/abs/2009.04416 ; code https://github.com/openai/phasic-policy-gradient
- L2C2 (related smoothness, Kobayashi 2022) — arXiv:2202.07152 — https://arxiv.org/abs/2202.07152
- Dual-Clip PPO — arXiv:1912.09729 ; DAPO arXiv:2503.14476 — https://arxiv.org/abs/2503.14476
- CrossQ — arXiv:1902.05605 (ICLR 2024) — https://arxiv.org/abs/1902.05605
- SAC — arXiv:1801.01290 ; TD3 — arXiv:1802.09477
- Real-quadruped off-policy: Learning to Walk via Deep RL arXiv:1812.11103 ; Learning to Walk in 20 Minutes (RSS 2023) https://www.roboticsproceedings.org/rss19/p056.pdf
- Context: Learning to Walk in Minutes (Rudin/Hutter) arXiv:2109.11978 — confirms this framework's on-policy massively-parallel lineage
