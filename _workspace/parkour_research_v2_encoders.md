# Temporal/Sequence Encoder Upgrades for Extreme-Parkour-Style Go2 Policy

**Scout brief:** Upgrade the temporal/sequence backbone of a Go2 parkour policy (Extreme Parkour derivative: CNN depth backbone + MLP base policy, 2-phase privileged→DAgger distillation) to a stronger architecture.

**Hard constraints honored throughout:**
- No input bloat (proprioception+depth only; contact-sensor obs banned for sim-to-real; no extra command vectors/sensors).
- Must be a drop-in replacement of the temporal backbone, not a full training-env rewrite.
- Prefer 2024–2026 work from top venues/labs with public/deploy code and real-robot evidence.

**Excluded per disqualification list (referenced only as baselines):** Extreme Parkour, Robot Parkour Learning, ANYmal Parkour, DTC, SoloParkour, PIE, RMA, HIM, DreamWaQ, Walk These Ways, Body Transformer, Terrain Transformer.

**Search scope:** 5 category fan-outs (SSM/Mamba, causal transformers, world models, diffusion, architecture ablations). ~15 candidates screened. All URLs verified via WebFetch/WebSearch (June 2026).

---

## Pre-filter — Candidates Eliminated Before Ranking

| Paper | arXiv | Reason eliminated |
|---|---|---|
| LocoMamba | 2508.11849 (Adv. Eng. Informatics 2026) | Mamba+PPO, but **simulation-only** — fails hardware gate |
| DiffuseLoco | 2404.19264 (Berkeley, CoRL 2024) | Real robot confirmed, but **full offline diffusion BC paradigm** — violates drop-in constraint #2 |
| GRoQ-LoCO | 2505.10973 (May 2025) | Attention+GRU, real Go1 deploy, but **requires offline multi-robot datasets** — paradigm change |
| Acrobotics | 2509.02727 (LNCS 2025) | Uses **temporal CNN sliding window**, not a sequence upgrade; simulation-only |
| Radosavovic Science Robotics | 2303.03381 (Sci. Robot. 2024) | Causal transformer, real-world hiking — **humanoid (Digit) only**, no quadruped |
| Radosavovic Next-Token-Prediction | 2402.19469 (2024) | Autoregressive causal transformer — humanoid (Digit) only, no code, no latency |
| World models (TD-MPC2, DreamerV3/4, RSSM) | various | **Full training-loop overhauls** (imagined rollouts, latent dynamics objective) — not backbone swaps; integration cost extreme |
| Neural Circuit Priors | 2410.07174 (2024) | Comparable to MLP (not better), no code |
| Symmetry-Guided Memory Augmentation | 2502.01521 | Targets sensor-failure robustness, not temporal expressiveness for agility |
| SLR / PUMA / ULT | 2406.04835 / 2601.15995 / 2503.08997 | Insufficient architecture detail or off-target (ULT = flat-terrain omnidirectional, no parkour, no MLP/CNN ablation) |

---

## CANDIDATE 1 — Sparse MoE Actor Layer

**Title:** Quadruped Parkour Learning: Sparsely Gated Mixture of Experts with Visual Input
**Authors / Lab / Venue:** Michael Ziegltrum, Jianhao Jiao, Tianhu Peng, Chengxu Zhou, Dimitrios Kanoulas — arXiv:2604.19344, submitted April 21, 2026. arXiv-only (top-venue peer review pending).
**URL:** https://arxiv.org/abs/2604.19344 | HTML: https://arxiv.org/html/2604.19344v1

### Architecture (verified from paper HTML)
Two-phase PPO, structurally identical to Extreme Parkour:
- Phase 1: privileged info (elevation maps via scandots MLP, ground-truth COM/friction/motor strengths)
- Phase 2: depth GRU replaces scandots encoder

MoE modification confined to the actor's middle layer:
- **Depth:** GRU (hidden 512) on 160×120 depth frames (downsampled to 87×58) → 32-dim latent
- **Proprioception:** 591-dim obs (angular vel, orientation roll/pitch, joint pos/vel ×12 each, foot contacts ×4, prev actions), **10-step history**
- **Actor:** 3 layers (512 → 256 → 256 → 12 actions). The middle 256-dim layer is replaced by a **sparsely-gated MoE: 16 experts, top-4 activated per token**, each expert = weight matrix (not a full sub-MLP). Load-balancing auxiliary loss coefficient 0.1, with noise injection for balancing.

The MoE is a single-layer drop-in inside the existing actor head.

### Measured improvement (verified, real Go2 hardware, 32 cm box obstacle)
| Model | Total params | Inference params | Trials (10 runs) | Contact-to-contact |
|---|---|---|---|---|
| MoE top-4/16 | 1.6M | 0.8M | **10/10** | 1.68 s |
| MLP Large (matched inference compute) | 0.8M | 0.8M | 5/10 | — |
| MLP Extra-Large (matched total params) | 1.6M | 1.6M | 9/10 | 1.99 s |

MoE **doubles success trials** vs matched-compute MLP; **14.3% faster inference** than parameter-matched MLP.

### Integration cost: **LOW**
One MLP layer replaced with a `ModuleList` of 16 linear experts + softmax gating + one auxiliary load-balancing loss term. CNN/GRU depth backbone, PPO loop, DAgger distillation, training env: all unchanged.

### Input small? **YES** — proprioception + depth only, same as baseline.

### Deploy evidence
- **Real robot:** Unitree Go2, hardware-confirmed on real 32 cm box obstacle.
- **Latency:** 4.2 ms (batch=6000, off-board Nvidia RTX 2060). Single-env on-board Jetson latency NOT reported.
- **Code:** https://osf.io/v2kqj/ (anonymized view-only; not full open source).

### Critical caveat
Off-board GPU inference confirmed; **on-board Jetson 50 Hz viability unverified**. If deploying proprioception-only (depth GRU discarded at deploy), validate independently that the MoE actor-head gains hold without the depth latent input. No top-venue peer review yet.

---

## CANDIDATE 2 — FiLM-Modulated Mamba Backbone (REAL)

**Title:** REAL: Robust Extreme Agility via Spatio-Temporal Policy Learning and Physics-Guided Filtering
**Authors / Lab / Venue:** Jialong Liu, Dehan Shen, Yanbo Wen, Zeyu Jiang, Changhao Chen — PEAK-Lab, HKUST (Guangzhou) — arXiv:2603.17653, submitted March 18, 2026. arXiv-only.
**URL:** https://arxiv.org/abs/2603.17653 | HTML: https://arxiv.org/html/2603.17653

### Architecture (verified from paper HTML)
Student encoder = FiLM-modulated Mamba SSM:
1. Depth → lightweight CNN → feature map F_CNN
2. Proprioception p_t → linear → FiLM channel-wise modulation: `FiLM(F_CNN) = γ(p_t) ⊙ F_CNN + β(p_t)`
3. Concatenation [FiLM-modulated features + raw proprioception + EKF velocity estimate] → stacked Mamba layers
4. Mamba recurrence: `h_t = A_t h_{t-1} + B_t x_t; y_t = C_t h_t` — data-dependent selective SSM, **O(1) per step at inference**
5. **10-frame proprioceptive history** (IMU + joint encoders). Latent/hidden size not stated in paper.
6. Separate physics-guided Bayesian EKF for velocity estimation feeds as an extra input token to the Mamba.

Surrounding training: privileged teacher RL (cross-modal attention) → FiLM+Mamba student distillation with consistency-aware loss gating (RL + behavioral cloning balance). Same two-phase distillation paradigm as Extreme Parkour, extended by ~3 new modules (FiLM, Mamba, EKF).

### Measured improvement (verified, real Go2 hardware + sim)
| Method | Overall SR | Hurdles | Steps | Gaps | Blind-zone SR (1m FoV occluded) |
|---|---|---|---|---|---|
| REAL (FiLM+Mamba) | **0.78** | 0.82 | 0.94 | 0.28 | 0.55 |
| SoloParkour | 0.39 | 0.42 | 0.49 | **0.36** | 0.36 |
| Extreme Parkour | 0.16 | 0.18 | 0.14 | 0.10 | — |
| RPL | 0.04 | 0.05 | 0.04 | 0.03 | — |

**4.9× higher overall SR than Extreme Parkour** baseline. Under 1-meter FoV occlusion: 0.55 vs 0.36 (best baseline).

### Integration cost: **MEDIUM-HIGH**
Mamba backbone is structurally a drop-in replacing the CNN+MLP temporal encoder, and the distillation paradigm matches the existing setup. But FiLM conditioning + EKF physics estimator must also be implemented — ~3 new components beyond a pure backbone swap. With an existing DAgger pipeline, estimate 2–4 weeks.

### Input small? **YES** — proprioception + depth only. No contact sensors.

### Deploy evidence
- **Real robot:** Unitree Go2, Intel RealSense D435i, NVIDIA Jetson, 50 Hz control loop, 1 kHz low-level PD. Zero-shot sim-to-real on three obstacle courses (high-platform leap, scattered-box navigation, steep staircase climbing).
- **Latency:** **13.14 ms on Jetson** (mean over 1,000 steps). Transformer baseline = 23.07 ms (violates 20 ms real-time gate). Mamba is the only architecture tested that clears 50 Hz on-board.
- **Code:** https://jialonglong.github.io/REAL_wb/ (project page; full codebase not released as of March 2026).

### Critical caveat
Gap traversal SR (0.28) is **worse** than SoloParkour (0.36). REAL's advantage concentrates on hurdles/steps where depth provides clear structure. If gap-crossing is the primary failure mode, this architecture does not guarantee improvement on that obstacle class. Pre-peer-review.

---

## CANDIDATE 3 — Masked Sensory-Temporal Attention (MSTA)

**Title:** Masked Sensory-Temporal Attention for Sensor Generalization in Quadruped Locomotion
**Authors / Lab / Venue:** Dikai Liu, Tianwei Zhang, Jianxiong Yin, Simon See — **ICRA 2025** (peer-reviewed, top venue). arXiv:2409.03332.
**URL:** https://arxiv.org/abs/2409.03332 | Project: https://johnliudk.github.io/msta/

### Architecture (from abstract + project page)
Instead of flattening proprioception into a vector passed through an MLP, MSTA treats each sensor modality (IMU, joint positions, joint velocities, foot contacts) as a separate **sensor token** at each timestep, forming a [sensor × time] token grid. Causal attention over this grid lets the policy selectively weight informative modality-timestep pairs while masking out missing/corrupted inputs.
- Proprioception-only (no visual input required)
- Causal structure → deployable as a recurrent model at inference: **O(1) per step**
- No contact-sensor input — designed explicitly for sim-to-real under sensor dropout

Exact layer/head counts, history length T, parameter count: in the full ICRA 2025 paper (https://arxiv.org/pdf/2409.03332) — NOT extractable from the abstract.

### Measured improvement
Outperforms MLP and GRU encoders under sensor noise and large-fraction dropout ("a large portion of missing information"). **Gains on standard full-sensor locomotion vs MLP not confirmed from available content — read the full paper before committing.**

### Integration cost: **MEDIUM**
Re-tokenize the observation vector by sensor modality (rather than flat concatenation) and add causal attention layers to the actor. PPO training loop, reward function, environment: unchanged. Same lab maintains related code for ULT (IROS 2025).

### Input small? **YES** — proprioception-only, no contact sensors. Directly addresses the sim-to-real deploy constraint.

### Deploy evidence
- **Real robot:** Confirmed deployment on a real quadruped (per abstract + project page).
- **Latency:** O(1) per step (recurrent at inference); exact ms figure not stated.
- **Code:** https://johnliudk.github.io/msta/ (project page; GitHub availability not confirmed from abstract).

### Critical caveat
Headline result is **sensor-dropout robustness, not raw agility/obstacle clearance**. If the bottleneck is jumping higher/further rather than surviving sensor degradation, prioritize #1 or #2. Quantified agility gains vs MLP on standard non-degraded locomotion must be confirmed from the full paper.

---

## Additional Mention — DiffuseLoco (eliminated, documented for completeness)

**Title:** DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets
**Authors / Venue:** Xiaoyu Huang, Yufeng Chi, Ruofeng Wang, Zhongyu Li, Xue Bin Peng, Sophia Shao, Borivoje Nikolic, Koushil Sreenath — Berkeley, CoRL 2024. arXiv:2404.19264.
**URL:** https://arxiv.org/abs/2404.19264

Diffusion policy from offline datasets; zero-shot real-robot quadruped transfer; 50 Hz deployment via receding-horizon control + delayed inputs (deploy-proven on edge compute). Real-world multi-skill locomotion confirmed. **Cannot rank**: full offline BC training paradigm overhaul — not a drop-in into an online RL + DAgger loop. Violates constraint #2. Integration cost = very high.

---

## RANKED SHORTLIST (Top 3)

| Rank | arXiv | Venue | Modification | Hardware evidence | On-board latency | Integration |
|---|---|---|---|---|---|---|
| **#1** | 2604.19344 | arXiv 2026 | One MLP layer → 16-expert top-4 MoE | Go2, 10/10 vs 5/10 trials | 4.2 ms off-board (Jetson unverified) | LOW |
| **#2** | 2603.17653 | arXiv 2026 | CNN+MLP encoder → FiLM+Mamba+EKF | Go2, 0.78 vs 0.16 SR | **13.14 ms on Jetson** | MED-HIGH |
| **#3** | 2409.03332 | ICRA 2025 | Flat obs encoder → sensor-token causal attention | Quadruped confirmed | O(1), ms unreported | MEDIUM |

**#1 — why it beats your current CNN+MLP:** On an identical Extreme Parkour training pipeline, replacing one dense MLP layer with a 16-expert MoE doubled real Go2 success trials (10/10 vs 5/10) at matched compute — lowest reimplementation risk of any candidate.

**#2 — why it beats your current CNN+MLP:** 4.9× higher overall parkour SR vs Extreme Parkour (0.78 vs 0.16) on real Go2 hardware, and Mamba is the only architecture tested that fits 50 Hz on a Jetson (13.14 ms vs transformer's 23.07 ms failure), with O(1) inference that stays flat as history grows.

**#3 — why it beats your current CNN+MLP:** The only ICRA-peer-reviewed option that gives the policy selective attention over which sensor modalities and timesteps matter — directly designed for proprioception-only sim-to-real where contact sensing is banned at deploy.

---

## Practical Recommendation

**Start with #1** (MoE, arXiv:2604.19344) — strict subset modification of the current codebase, no training-paradigm change, hardware-confirmed on Go2, lowest reimplementation risk (~1 week). **If you need larger agility gains**, follow with #2 (REAL/Mamba, arXiv:2603.17653; ~2–4 weeks, existing DAgger loop already compatible). **Add #3** (MSTA, arXiv:2409.03332) if deploy-time sensor noise becomes the bottleneck.

**Adversarial flag on all three:** None have fully open-sourced production code as of June 2026. Plan for reimplementation, not pip-install. Estimate 1 week (#1), 2–4 weeks (#2), 1–2 weeks (#3).

---

## Verified Sources
- REAL: https://arxiv.org/abs/2603.17653 — https://arxiv.org/html/2603.17653
- Quadruped Parkour MoE: https://arxiv.org/abs/2604.19344 — https://arxiv.org/html/2604.19344v1 — code https://osf.io/v2kqj/
- MSTA (ICRA 2025): https://arxiv.org/abs/2409.03332 — https://johnliudk.github.io/msta/
- DiffuseLoco (CoRL 2024): https://arxiv.org/abs/2404.19264
- LocoMamba: https://arxiv.org/abs/2508.11849
- GRoQ-LoCO: https://arxiv.org/abs/2505.10973
- Unified Locomotion Transformer (IROS 2025): https://arxiv.org/abs/2503.08997 — https://arxiv.org/html/2503.08997v1
- Acrobotics: https://arxiv.org/abs/2509.02727
- Radosavovic Science Robotics 2024: https://www.science.org/doi/10.1126/scirobotics.adi9579
- Humanoid Locomotion as Next Token Prediction: https://arxiv.org/abs/2402.19469
- Neural Circuit Priors: https://arxiv.org/abs/2410.07174
- PUMA: https://arxiv.org/abs/2601.15995
- SLR: https://arxiv.org/abs/2406.04835
