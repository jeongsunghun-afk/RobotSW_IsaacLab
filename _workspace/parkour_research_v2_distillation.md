# Training Algorithm Improvements for Extreme-Parkour + 2-Phase DAgger Pipeline

**Date:** 2026-06-10
**Context:** Go2 parkour policy is an Extreme Parkour derivative using a privileged teacher (elevation scandots + oracle heading) distilled to a depth-only student via 2-phase DAgger. Goal: better TRAINING ALGORITHMS for the privileged-learning / distillation pipeline.

**Hard gate applied to every candidate:** Does it touch the student/deploy observation space or require a sim environment rewrite? Only algorithm/loss/training-procedure changes that layer onto an existing PPO + teacher-student codebase pass. Privileged CRITIC changes are allowed (critic dropped at deploy); STUDENT input changes are not.

**Exclusion list confirmed applied (baselines only, not recommended):** Extreme Parkour, Robot Parkour Learning, ANYmal Parkour, DTC, SoloParkour, PIE, RMA, HIM, DreamWaQ, Walk These Ways.

---

## Thread 1 — Teacher-Student / DAgger Improvements

### CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion
- **Authors:** Hongxi Wang, Haoxiang Luo, Wei Zhang, Hua Chen — CLEARLAB, Southern University of Science and Technology (SUSTech)
- **Venue:** IEEE Robotics and Automation Letters (RA-L) 2024
- **arXiv:** 2405.10830

**Exact mechanism (formula-level).** Eliminates the 2-phase sequential structure. Teacher and student train simultaneously in a single PPO run, sharing the same policy network πθ and critic Vϕ but using different encoders:
- Teacher encoder Eθᵗ receives privileged state sᵗ (terrain height, contact forces, joint torques)
- Student encoder Eθˢ receives proprioceptive history o^{t−H:t}

Combined PPO-clip loss over both trajectory sets:
```
L_policy = L_PPO(Dᵗ) + L_PPO(Dˢ)
L_PPO = min(rₜ·Âₜ, clip(rₜ, 1−ε, 1+ε)·Âₜ)   (applied separately to teacher set Dᵗ and student set Dˢ)
```
Shared critic MSE: `L_value(ϕ) = (1/|D|T) Σ (Vϕ(sₜ,zₜ) − R̂ₜ)²` over combined trajectories.
Student supervised alignment (reconstruction): `L_rec(θˢ) = (1/|Dˢ|T) Σ ‖Eθˢ(oˢ) − Eθᵗ(sᵗ)‖₂²`
Rollout ratio: 6144 teacher : 2048 student agents (3:1). The student encoder is updated by BOTH the RL policy gradient AND the reconstruction loss simultaneously.

**Why it beats plain 2-phase DAgger.** In 2-phase, the teacher freezes at convergence, then the student distills from a fixed distribution that never covers states the student's own policy generates → compounding error OOD. CTS keeps both policies live simultaneously so the student continuously corrects against the live teacher while also receiving its own RL gradient (so it can handle states where the teacher representation is uninformative). Quantified: up to 20% reduction in average velocity tracking error vs. 2-stage baseline.

**Integration cost:** Medium. Restructure 2-phase runs into a single run with dual rollout buffers (split 4096-env pool ~3:1). Existing PPO + encoder architecture maps directly; key change is dual trajectory collection plus the L_rec term.
**Touches obs/env?** NO. Student deploy observation = proprioception only (unchanged). Teacher privileged state is training-only, dropped at deploy. **PASS.**

**Code URL:** No public repository confirmed as of June 2026. Project page (videos only): https://clearlab-sustech.github.io/concurrentTS/
**Real-robot evidence:** Yes — indoor and outdoor experiments on quadrupedal and point-foot bipedal hardware.
**Critical caveat:** No public code. The 3:1 rollout split reduces effective parallel throughput per group. Architecture fully specified for re-implementation.

---

### CritiQ + ReTRy: Distilling Realizable Students from Unrealizable Teachers
- **Authors:** Yujin Kim, Nathaniel Chin, Arnav Vasudev, Sanjiban Choudhury — Cornell PoRTaL Lab
- **Venue:** IROS 2025
- **arXiv:** 2505.09546
- **Code:** https://github.com/portal-cornell (CritiQ_ReTRy repo confirmed public)

**Exact mechanism.** Two complementary improvements to DAgger targeting the gap between a full-state teacher and a partial-observation student that cannot replicate it in unobservable regions.

**CritiQ (imitation path):** A critical-state classifier identifies states where the student is about to enter an unrecoverable trajectory. DAgger queries the teacher ONLY at these states:
```
query(s) = 1 if classifier(s) = critical ; else 0
```
This eliminates the sample waste of uniform querying on easy states.

**ReTRy (RL path — more directly relevant to gait collapse):** Iteratively builds a reset-state curriculum. At each iteration the teacher rolls out from states VISITED BY THE STUDENT, and those teacher-rollout states are added to the training reset distribution:
```
R_{i+1} = R_i ∪ { states from teacher rollouts starting at student-visited states_i }
```
The student is repeatedly initialized at the exact states where it deviates — staying on a recoverable path within its own observation space.

**Why it beats plain 2-phase DAgger.** Standard DAgger initializes at canonical easy states; the 3-leg-gait failure occurs mid-episode (the RL leg gradually stops swinging) and is never trained as a recovery problem. CritiQ focuses teacher supervision where it matters. ReTRy seeds RL training precisely at the onset of the RL leg's deviation, repeatedly exposing the recovery problem as an initialization.

**Integration cost:** Medium. CritiQ: lightweight critical-state classifier trained concurrently. ReTRy: log student trajectory states during rollout and sample from the augmented buffer at episode reset.
**Touches obs/env?** NO for observation. ReTRy requires resetting Isaac Sim to arbitrary states — standard functionality, not an env rewrite. **PASS.**

**Code URL:** https://github.com/portal-cornell (CritiQ_ReTRy). Project page: https://portal-cornell.github.io/CritiQ_ReTRy/
**Real-robot evidence:** Yes — validated in simulated and real-world robotic tasks per paper.
**Critical caveat:** IROS 2025, recent. Benchmark tasks may be manipulation-class, not legged parkour — verify scope before over-weighting. Codebase targets its own tasks, not Isaac Lab + RSL-RL; reset-API port required.

---

### To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable RL
- **Authors:** Yuda Song, Dhruv Rohatgi, Aarti Singh, J. Andrew Bagnell — CMU
- **Venue:** NeurIPS 2025
- **arXiv:** 2510.03207

**Mechanism / finding.** Theoretical characterization of when privileged distillation beats direct RL from observations. Key result: "the optimal latent policy is not always the best latent policy to distill"; the trade-off hinges on the stochasticity of latent dynamics (grounded in a "perturbed Block MDP" model).
**Touches obs/env?** N/A — simulation-only, no algorithm artifact.
**Code:** Not provided. **Real-robot:** None (simulation-only).
**Caveat:** No practical algorithm change. EXCLUDED from ranking — cite only when justifying phase-switching design decisions.

---

## Thread 2 — Asymmetric Actor-Critic Improvements

### RobotKeyframing: Learning Locomotion with High-Level Objectives via Mixture of Dense and Sparse Rewards (Multi-Critic PPO)
- **Authors:** Fatemeh Zargarbashi, Jin Cheng, Dongho Kang, Robert Sumner, Stelian Coros — ETH Zurich
- **Venue:** CoRL 2024
- **arXiv:** 2407.11562

**Exact mechanism.** Multi-critic PPO: N critic heads share a backbone, each estimating the value of ONE reward component independently. Advantage Aₖ is estimated per component; total policy gradient is `Σ_k Aₖ`. This prevents scale contamination — when a gait-obligation reward and a terrain reward differ by an order of magnitude, a shared single critic under-weights one or the other.

**Why it helps our pipeline.** If you add a contact-duty constraint reward (Thread 5), the scale mismatch with the terrain-traversal reward biases advantage estimation under a single critic. Multi-critic PPO makes each component independently learnable. Paper also reports it significantly reduces hyperparameter tuning effort vs. single-critic.

**Integration cost:** Low-Medium. Add N heads to the RSL-RL critic sharing a backbone, one per reward group. Final PPO update unchanged.
**Touches obs/env?** NO — pure critic architecture change. **PASS.**

**Code URL:** Not located in fetch (project page referenced but link not extracted).
**Real-robot evidence:** Yes — simulation and hardware experiments (CoRL 2024).
**Critical caveat:** Paper targets goal-conditioned keyframe locomotion, not parkour obstacle traversal. The multi-critic benefit for gait-vs-terrain decomposition is inferrable but untested in this context; if reward groups are mis-assigned the benefit disappears.

---

### Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Full-State Access
- **Authors:** Daniel Ebi, Damien Ernst, Klemens Böhm, Gaspard Lambrechts
- **Venue:** ICML 2026 (verified from arXiv abstract page)
- **arXiv:** 2509.26000

**Mechanism.** Extends asymmetric actor-critic theory so the critic can condition on arbitrary state-dependent privileged signals (not just full state) while still yielding unbiased policy gradients. Provides two informativeness criteria: a dependence-based test (pre-training) and a value-prediction-improvement test (post-training) for choosing which privileged signals to feed the critic.
**Touches obs/env?** NO — critic-only; actor unchanged. **PASS** on the gate.
**Integration cost:** Low (change which privileged obs feed the critic).
**Code URL:** Not provided. **Real-robot evidence:** None — experiments on partially-observable synthetic benchmarks only.
**Caveat:** Pure theory, no code, no hardware. Marginal practical gain over "give the critic everything" in a high-dim sim setting. NOT RANKED.

---

## Thread 3 — Exploration / Optimization to Escape Local Optima

### GPO: Growing Policy Optimization for Legged Robot Locomotion and Whole-Body Control
- **Authors:** Shuhao Liao, Peizhuo Li, Xinrong Yang, Linnan Chang, Zhaoxin Fan, Qing Wang, Lei Shi, Yuhong Cao, Wenjun Wu, Guillaume Sartoretti
- **Venue:** arXiv preprint, January 2026 (no venue yet)
- **arXiv:** 2601.20668

**Exact mechanism.** A time-varying action transformation τ_t restricts the effective action space early in training and progressively expands it:
```
a_eff = τ_t(a_raw)
```
τ_t starts narrow (low-torque regime only) and widens toward the full torque envelope over training. Proven to preserve the PPO update rule — "introduces only bounded, vanishing gradient distortion." Early exploration is forced into the low-torque regime.

**Why it attacks the 3-leg local optimum.** The 3-leg equilibrium is a high-reward shortcut reachable early, and sustaining one lifted leg requires sustained high torque. GPO's narrow-action early phase makes that shortcut hard to establish before the policy has learned to use all four legs — potentially bootstrapping into a 4-leg equilibrium before the full torque envelope unlocks.

**Integration cost:** Low. Apply τ_t as a wrapper around the RSL-RL action output in the PPO runner. No network changes.
**Touches obs/env?** NO — action-space transformation only. **PASS.**

**Code URL:** No public code found as of June 2026.
**Real-robot evidence:** Yes — zero-shot sim-to-real deployment on quadruped and hexapod hardware.
**Critical caveat:** Preprint only, no venue, no public code. The narrow early phase may slow terrain-curriculum progression if the robot can't generate enough force to surmount early obstacles. Expansion schedule needs calibration against your torque limits.

---

### A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning
- **Authors:** Arthur Juliani, Jordan T. Ash
- **Venue:** NeurIPS 2024
- **arXiv:** 2405.19153

**Exact mechanism / findings.** PPO accumulates dormant units under distribution shift (terrain curriculum). Critical finding: off-policy fixes (CReLU, plasticity injection) do NOT transfer to on-policy PPO and can perform worse than no intervention. What works: REGENERATIVE methods — continual regularizers / periodic re-initialization of dormant neurons (continual backpropagation). Related Nature 2024 result: PPO + continual backpropagation has few dormant units, high stable rank, near-constant weight magnitude, and avoids the performance collapse standard PPO shows on ant locomotion at ~20M steps.

**Why relevant.** If the RL stage suffers plasticity loss across curriculum stages, the stuck leg's representation can't update. A periodic reset of dormant units (near-zero activation) could unblock gradient flow.
**Integration cost:** Low (~50 lines — dormant-unit check every N updates, re-initialize those units, in the RSL-RL training loop).
**Touches obs/env?** NO — pure optimizer/network maintenance. **PASS.**
**Code URL:** Not extracted (NeurIPS 2024 proceedings + arXiv).
**Real-robot evidence:** None for the specific fix (benchmark study; ant-locomotion in sim).
**Caveat:** The 3-leg failure is a reward equilibrium problem (clip floor) per prior diagnosis, not primarily gradient stagnation. Apply as hygiene against secondary plasticity collapse, not the primary fix. NOT in top-3.

---

## Thread 4 — Symmetry as Critic/Value Constraint (NOT data-aug, NOT equivariant net)

**HONEST FINDING:** No 2024–2026 paper encodes symmetry as a critic/value bound (V(s) ≠ V(mirror(s))). The field splits into (1) data augmentation (already tried as Go2-Parkour-Symmetry, failed for 3-leg), (2) equivariant network architectures (require architectural rewrite), (3) symmetry-in-reward. The closest usable method below uses symmetry in the REWARD (not data-aug, not equivariant net).

### Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy (symmetry in reward function)
- **Authors:** Jiayu Ding, Xulin Chen, Garrett E. Katz, Zhenyu Gan
- **Venue:** arXiv:2403.10723 (v4 updated Feb 2026) — preprint, no top-venue publication confirmed. NOTE: arXiv:2510.10455 was a later version that was WITHDRAWN and redirects to 2403.10723.

**Exact mechanism.** Three symmetry terms baked directly into the reward function — not data augmentation, not equivariant networks:
- **Morphological symmetry reward:** penalizes left-right asymmetry in joint angles/forces during the same gait phase. NO explicit phase variable required.
- **Temporal symmetry reward:** penalizes phase offset between left/right leg cycles. Requires a phase variable.
- **Time-reversal symmetry reward:** penalizes non-reversible gait sequences (promotes periodic trot-like behavior).

**Why this differs from data-aug already tried.** Data-aug (Go2-Parkour-Symmetry) added mirrored rollouts to the batch passively. These reward terms apply LIVE per-step penalties at every gradient step. The morphological term penalizes the RL leg for deviating from the mirror posture of the RR leg even when the contact-duty reward is clamped to zero — providing the missing gradient where the duty-clip floor provides none.

**Integration cost:** Low. Add 1-3 reward terms to `_get_rewards()`. Morphological term has no phase-variable dependency and can be applied immediately.
**Touches obs/env?** NO — pure reward term additions. **PASS.**
**Code URL:** Not confirmed public.
**Real-robot evidence:** Yes — Unitree Go2 hardware tests confirmed.
**Critical caveat:** Preprint only (no top-venue publication). Temporal symmetry term requires an explicit gait phase variable the pipeline may not maintain. Apply the morphological term first as a no-dependency version; add temporal only if a phase signal is added.

---

## Thread 5 — Constraint-Based RL for Gait/Contact Obligation Without Obs Change

### Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion (KAIST IPO)
- **Authors:** Yunho Kim, Hyunsik Oh, Jeonghyun Lee, Jinhyeok Choi, Gwanghyeon Ji, Moonkyu Jung, Donghoon Youm, Jemin Hwangbo — KAIST Robotics and AI Lab (RaiLab)
- **Venue:** IEEE Transactions on Robotics, Vol. 40, pp. 2984–3003, 2024
- **arXiv:** 2308.12517
- **Code:** https://github.com/railabatkaist/legged-robot-constrained-rl

**Exact mechanism (formula-level).** Interior-Point Policy Optimization (IPO) with two constraint types and adaptive feasibility thresholds.

Probabilistic constraint — encodes gait contact pattern:
```
C_k(s,a,s') = 0      if foot contact state matches desired gait phase
C_k(s,a,s') = 1/n    otherwise   (n = n_legs)
E[C_k] ≤ D_k
Desired phase: (cos Φ(t), sin Φ(t)), Φ(t) = 2π f t + Φ_0
```
Average constraint — encodes foot clearance, contact velocity, orthogonal velocity:
```
E[f(s,a,s')] ≤ D_k
```
IPO objective (log-barrier augmented reward):
```
maximize  J(π) + Σ_k log(d_k − J_{C_k}(π)) / t
subject to  KL(π ‖ π_i) ≤ δ
```
where t is gradually increased (barrier tightens over training).
Adaptive threshold prevents infeasibility from random init:
```
d_k^i = max(d_k, J_{C_k}(π_i) + α·d_k)
```
Multi-head cost value function: one shared backbone predicts all constraint cost values simultaneously (11 constraints, negligible overhead). Implementations for both TRPO and PPO backends.

**Why it beats vanilla PPO for OUR pipeline.** The current contact-duty penalty uses `clip(min=0)` — intentional design per project memory — which eliminates the gradient when the correct legs are above their duty target (duty ≥ target → reward floor → gradient 0 → RL leg parks at the 3-leg residual). IPO's log-barrier on contact-duty has a NON-ZERO gradient from BOTH sides of the target. The gradient runs through the log-barrier term, structurally separate from the main reward, so the main-reward `clip(min=0)` cannot neutralize it. This is the algebraic patch to the diagnosed mechanism.

**Integration cost:** Medium. IPO replaces the PPO objective with the log-barrier augmented version; multi-head cost critic adds minor architecture. KAIST code covers PPO and TRPO; bridging to RSL-RL PPO is tractable. Constraint functions (contact duty, foot clearance) already computable from sim state.
**Touches obs/env?** NO. Gait phase target is a scalar parameter, not a student observation. Cost critic is training-only, dropped at deploy. **PASS.**

**Code URL:** https://github.com/railabatkaist/legged-robot-constrained-rl
**Real-robot evidence:** Yes — extensive real-world experiments across multiple legged robot morphologies. IEEE T-RO peer-reviewed.
**Critical caveat:** IPO has inner/outer tuning parameters (t and adaptive α). Constraints set too tightly cause the policy to sacrifice forward velocity entirely. Start with loose bounds (duty target ≈ 0.45) and tighten progressively.

---

### CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning
- **Authors:** Elliot Chane-Sane, Pierre-Alexandre Leziart, Thomas Flayols, Olivier Stasse, Philippe Souères, Nicolas Mansard — LAAS-CNRS
- **Venue:** IROS 2024
- **arXiv:** 2403.18765
- **Code/project:** https://constraints-as-terminations.github.io

**Exact mechanism (formula-level).** Constraints become stochastic episode TERMINATION probabilities rather than reward penalties or Lagrange multipliers.
```
δ = max_{i∈I}  p_i^max · clip(c_i^+ / c_i^max, 0, 1)

c_i^+   = max(0, c_i(s,a))                                  # violation magnitude
c_i^max ← τ^c · c_i^max + (1−τ^c) · max_{(s,a)∈batch} c_i^+(s,a)   # running normalization
```
PPO modification — exactly 3 lines:
```python
delta   = compute_delta(constraint_violations)   # eq. above
rewards = rewards * (1 - delta)
dones   = delta
```
Contact-duty / gait constraint: `c_i = max(0, target_duty − actual_duty)`. The paper explicitly supports foot contact force limits, air-time targets, number-of-foot-contacts (duty cycle), and collision avoidance as constraints c_i.

**Why it beats vanilla PPO for OUR pipeline.** The gradient signal runs through episode length (anticipated near-termination via δ), NOT through reward magnitude. This channel is structurally orthogonal to `clip(min=0)` on the main reward and cannot be neutralized by it — so the RL leg's duty deficit produces a non-zero learning signal (termination risk) even when the reward floor is saturated.

**Integration cost:** Low — 3-line change to the PPO rollout loop. No architecture change. Constraint functions already computable.
**Touches obs/env?** NO. Operates on standard state/action; height-scan obs in the paper's experiments are optional, not required. **PASS.**

**Code URL:** https://constraints-as-terminations.github.io (videos + code)
**Real-robot evidence:** Yes — Solo quadruped crossing stairs, slopes, and high obstacles. IROS 2024 peer-reviewed.
**Critical caveat:** Termination enforcement is probabilistic, not guaranteed. For gait-pattern constraints (vs. safety/torque limits), natural contact-timing variance in trotting may trigger spurious terminations if p_i^max is too high. Calibrate p_i^max on your terrain and duty-cycle definition before relying on it.

---

### A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards
- **Authors:** Gijeong Kim, Yong-Hoon Lee, Hae-Won Park — KAIST
- **Venue:** ICRA 2025
- **arXiv:** 2409.15780

**Exact mechanism.** Motion-style reward based on a RELAXED logarithmic barrier function as a soft constraint, biasing learning toward a desired gait/foot-clearance/joint-position/body-height style. Relaxed log-barrier is smooth everywhere (unlike hard log-barrier which is infinite at violation):
```
r_style ≈ relaxed_logbarrier(c(s,a) − c_target)
```
The predefined gait cycle is encoded in a flexible, phase-matched manner; deviation from the target contact state is penalized.

**Why relevant.** Soft-constraint alternative to IPO/CaT that enables quadruped/tripod/biped modes from a single framework — demonstrates explicit gait/contact pattern shaping without exteroceptive input.
**Integration cost:** Low — add one reward term. Works without explicit phase tracking if contact state is available.
**Touches obs/env?** NO — "without exteroceptive input." **PASS.**
**Code URL:** Not released.
**Real-robot evidence:** Yes — 45 kg KAIST HOUND: biped/tripod/quadruped, galloping 4.67 m/s, bipedal running 3.6 m/s, stairs, obstacles up to 58 cm — all from scratch.
**Critical caveat:** No code. KAIST-internal platform (HOUND, not Go2). Relaxed barrier is gentler than hard IPO — may not fully escape a deep 3-leg local optimum on its own.

---

## RANKED TOP-3 (training/algorithm changes most worth adopting)

Criteria in order: (1) obs/env gate PASS, (2) directly attacks the confirmed failure (3-leg local optimum / gradient elimination via `clip(min=0)` reward floor), (3) hardware evidence, (4) public code, (5) integration feasibility on RSL-RL + Isaac Lab.

**Rank 1 — KAIST IPO ("Not Only Rewards But Also Constraints")** — arXiv:2308.12517, IEEE T-RO 2024, code: github.com/railabatkaist/legged-robot-constrained-rl
One-line rationale: The only method whose log-barrier on contact-duty creates a non-zero gradient from both sides of the target, algebraically immune to the `clip(min=0)` floor that is paralysing the RL leg — peer-reviewed, hardware-validated, public code with a PPO backend.

**Rank 2 — CaT (Constraints as Terminations)** — arXiv:2403.18765, IROS 2024, code: constraints-as-terminations.github.io
One-line rationale: Same clip-floor bypass via an orthogonal gradient channel (episode-length/termination, not reward magnitude) at 3-line PPO integration cost — the cheapest fast-path experiment to validate the hypothesis before committing to full IPO.

**Rank 3 — CTS (Concurrent Teacher-Student RL)** — arXiv:2405.10830, IEEE RA-L 2024 (no public code)
One-line rationale: Once gait collapse is fixed, this is the most mature hardware-validated single-stage replacement for 2-phase DAgger, eliminating the frozen-teacher compounding-error gap (~20% tracking improvement) — apply after the constraint fix; re-implement from the fully specified paper.

---

## Adversarial kills / not-ranked
- **Informed Asymmetric AC** (ICML 2026, arXiv:2509.26000): no code, no hardware, pure theory. KILLED.
- **To Distill or Decide?** (NeurIPS 2025, arXiv:2510.03207): sim-only, no algorithm artifact. KILLED for ranking; design-reference only.
- **Plasticity loss / continual backprop** (NeurIPS 2024, arXiv:2405.19153): hygiene measure only; our failure is a reward equilibrium, not gradient stagnation. Not top-3.
- **GPO** (arXiv:2601.20668): would be Rank 4 — relevant action-space-expansion mechanism, hardware-validated, but PREPRINT with NO public code. Monitor for code release.
- **Symmetry-guided reward** (arXiv:2403.10723): preprint only; temporal term needs a phase variable. Experimental add-on, not a ranked pick. Thread 4 "symmetry as critic/value constraint" returned EMPTY — no such paper exists in 2024–2026.
- **Symmetry data-aug / equivariant nets** (e.g. MS-PPO arXiv:2512.00727): excluded per user constraint (already tried, can't fix 3-leg).

## All verified source URLs
- CTS: https://arxiv.org/abs/2405.10830 — project: https://clearlab-sustech.github.io/concurrentTS/
- CritiQ/ReTRy: https://arxiv.org/abs/2505.09546 — code: https://github.com/portal-cornell — project: https://portal-cornell.github.io/CritiQ_ReTRy/
- To Distill or Decide?: https://arxiv.org/abs/2510.03207
- RobotKeyframing: https://arxiv.org/abs/2407.11562
- Informed AAC: https://arxiv.org/abs/2509.26000
- GPO: https://arxiv.org/abs/2601.20668
- Plasticity Loss On-Policy: https://arxiv.org/abs/2405.19153
- KAIST IPO: https://arxiv.org/abs/2308.12517 — code: https://github.com/railabatkaist/legged-robot-constrained-rl
- CaT: https://arxiv.org/abs/2403.18765 — code: https://constraints-as-terminations.github.io
- Barrier-Based Style Rewards: https://arxiv.org/abs/2409.15780
- Symmetry-Guided RL Hierarchy: https://arxiv.org/abs/2403.10723
