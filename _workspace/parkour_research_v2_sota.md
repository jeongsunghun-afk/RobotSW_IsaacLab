# SOTA Agile Quadruped Locomotion / Parkour Survey — 2024–2026

**Scope:** Newest state-of-the-art agile quadruped locomotion / parkour works (2024–2026) that a prior survey MISSED. Baseline = Extreme Parkour derivative on Unitree Go2 (IsaacLab). Goal = fresh algorithm/architecture ideas with deploy code that layer onto teacher-student PPO without env rewrite or deploy-obs bloat.

**Excluded (already covered baseline):** Robot Parkour Learning (CoRL23), Extreme Parkour (ICRA24), ANYmal Parkour (Sci.Rob.24), DTC (Sci.Rob.24), SoloParkour (CoRL24), PIE (RA-L24), Walk These Ways, RMA, HIM, DreamWaQ, legged_gym, Wild ANYmal, APEX, CPG-RL/AllGaits.

**11 verified works found.** All arXiv IDs and code URLs verified via WebSearch/WebFetch.

---

## 1. REAL: Robust Extreme Agility via Spatio-Temporal Policy Learning and Physics-Guided Filtering

- **arXiv:** 2603.17653 — https://arxiv.org/abs/2603.17653
- **Authors:** Jialong Liu, Dehan Shen, Yanbo Wen, Zeyu Jiang, Changhao Chen
- **Lab/Venue:** PEAK-Lab, HKUST (Guangzhou) — **Preprint, March 2026 (NOT peer-reviewed)**

**Key NEW algorithm/architecture idea (over Extreme Parkour):**
Three stacked upgrades to the teacher-student distillation pipeline:
1. **FiLM-modulated Mamba backbone** for the student — proprioceptive signals dynamically modulate visual features via Feature-wise Linear Modulation (FiLM); a Mamba state-space model provides linear-time temporal memory, enabling robust short-term terrain recall even under 1-meter visual blind zones (replaces the LSTM in Extreme Parkour-style students).
2. **Physics-guided Extended Kalman Filter (EKF)** — fuses uncertainty-aware neural velocity predictions with rigid-body dynamics to enforce physically consistent, noise-robust state estimation.
3. **Consistency-aware loss gating** — dynamically rebalances behavioral-cloning vs. RL loss based on the instantaneous teacher-student action discrepancy.
Teacher uses a cross-modal attention mechanism: proprioceptive states are queries, terrain scan points are keys/values (state-conditioned terrain reasoning).

**Constraint fit:**
- No deploy-obs bloat: student uses depth + proprioception only.
- No full env rewrite: teacher-student structure compatible with existing parkour codebase.
- Layers onto teacher-student PPO: yes — direct architectural swap.

**Code + framework:** Planned release at https://jialonglong.github.io/REAL_wb/ (NOT yet live as of June 2026). Framework: **Isaac Gym** (~30 hr training, single RTX 4080).

**Real-robot deploy evidence:** Unitree Go2, onboard Jetson, 13.1 ms inference; traverses extreme obstacles with 1-meter visual blind zones. Direct ablation vs. Extreme Parkour, Robot Parkour Learning (RPL), and SoloParkour.

**Integration relevance:** Exact Go2 + Isaac Gym match. Replace LSTM student encoder with FiLM-Mamba; add EKF as a parallel velocity head; wrap existing PPO+BC loss with consistency gating. No env/reward changes.

**Critical caveat:** Code not yet released. Mamba inference kernel may add dependency overhead on Jetson. Unreviewed preprint.

---

## 2. PUMA: Perception-Driven Unified Foothold Prior for Mobility Augmented Quadruped Parkour

- **arXiv:** 2601.15995 — https://arxiv.org/abs/2601.15995
- **Authors:** Liang Wang, Kanzhong Yao, Yang Liu, Weikai Qin, Jun Wu, Zhe Sun, Qiuguo Zhu
- **Lab/Venue:** Institute of Cyber-Systems and Control, Zhejiang University + China Telecom TeleAI — **Preprint, January 2026**

**Key NEW algorithm/architecture idea (over Extreme Parkour):**
A **learned egocentric polar foothold prior** fed directly into the actor network. Four values: left/right forefoot distance to the current expected foothold + heading error between current orientation and target direction. A regression estimator predicts these from depth + proprioception. **Probability Annealing Selection (PAS)** gradually transitions from ground-truth to predicted footholds during training (curriculum to close the estimation gap). Custom NVIDIA Warp ray-tracing depth renderer for efficient sim depth. Gives the policy explicit geometric foothold reasoning without a separate hierarchical planner.

**Constraint fit:**
- No deploy-obs bloat: only 4 floats added to obs.
- No full env rewrite: yes (prior is an auxiliary obs + training-time curriculum).
- Layers onto teacher-student PPO: yes.

**Code + framework:** No code released. Framework: **Isaac Gym**, 2048 envs, RTX 4090.

**Real-robot deploy evidence:** DeepRobotics Lite3 (12-DoF), RK3588 compute, Intel RealSense D435i. Real-world deploy confirmed across discrete complex terrains. Wall-kick-assisted 1.2 m gap jumps, 0.7 m platform vaulting, emergent galloping. **Beats Extreme Parkour: 96.9% vs. 76.5% on wall-assisted gaps at 80° inclination.**

**Integration relevance:** Add a foothold prior regression head to the teacher; append 4 floats to student obs; wrap training loop with PAS annealing. Hardware-agnostic idea.

**Critical caveat:** Hardware is DeepRobotics Lite3, NOT Go2 — joint limits and leg geometry differ; foothold polar coordinates need recalibration for Go2. No code.

---

## 3. MoE-Loco: Mixture of Experts for Multitask Locomotion

- **arXiv:** 2503.08564 — https://arxiv.org/abs/2503.08564
- **Authors:** Runhan Huang, Shaoting Zhu, Yilun Du, Hang Zhao
- **Venue:** **IROS 2025 (peer-reviewed)**

**Key NEW algorithm/architecture idea (over Extreme Parkour):**
**Sparsely gated Mixture-of-Experts (MoE) policy backbone.** Routes inputs to specialized expert sub-networks; only a subset of parameters activate at inference. Mitigates gradient conflicts that arise in multi-task RL across diverse terrains. Different experts naturally specialize in distinct locomotion behaviors. Demonstrates emergent **skill composition** — combining expert activations yields novel gaits (dribbling = balancing + crossing experts; three-legged locomotion) WITHOUT retraining.

**Constraint fit:**
- No deploy-obs bloat: MoE is a policy-internal change.
- No full env rewrite: yes — direct MLP→MoE swap, env/reward/PPO unchanged.
- Layers onto teacher-student PPO: yes.

**Code + framework:** **https://github.com/hrh6666/MoE-Loco** (PUBLIC, verified). Framework: **Isaac Gym**, 4096 envs, RTX 3090. Training: 40k iters plane walking (both gaits) → 80k iters challenging terrain → 10k iters PAS to adapt to pure proprioception.

**Real-robot deploy evidence:** Unitree Go2 + NVIDIA Jetson Orin. Zero-shot real-world deploy across 9 terrain types (bars, pits, baffles, stairs, slopes, bipedal standing/walking).

**Integration relevance:** Exact Go2 + Jetson hardware. Replace policy MLP with sparse MoE; no env changes. Lowest integration friction of all candidates.

**Critical caveat:** Base version is proprioception-only (no depth/visual input) — Extreme Parkour uses depth-gated terrain perception, so combining MoE with the depth backbone needs additional engineering (see entry #4, which is the visual extension of this idea).

---

## 4. Quadruped Parkour Learning: Sparsely Gated Mixture of Experts with Visual Input

- **arXiv:** 2604.19344 — https://arxiv.org/abs/2604.19344
- **Authors:** Michael Ziegltrum, Jianhao Jiao, Tianhu Peng, Chengxu Zhou, Dimitrios Kanoulas
- **Lab/Venue:** Robot Perception and Learning Lab, University College London (UCL) — **Preprint, April 2026**

**Key NEW algorithm/architecture idea (over Extreme Parkour):**
Extends sparsely gated MoE specifically to **vision-based parkour** — directly addresses the missing visual input in MoE-Loco. Activates only a subset of parameters at inference while achieving **double the successful trials** on large obstacle traversal vs. an MLP baseline. Favorable performance/compute trade-off for scaling vision-based parkour policies.

**Constraint fit:**
- No deploy-obs bloat: same depth pipeline as Extreme Parkour.
- No full env rewrite: yes — MLP→MoE swap on an existing visual parkour pipeline.
- Layers onto teacher-student PPO: yes.

**Code + framework:** OSF anonymized (review code only): https://osf.io/v2kqj/files/github?view_only=7977dee10c0a44769184498eaba72e44 — full public repo not yet released. Framework: not specified in paper (likely IsaacGym/legged_gym based on Go2 baselines used).

**Real-robot deploy evidence:** Unitree Go2; real-world deploy confirmed (2x success vs. MLP on large obstacles).

**Integration relevance:** Most direct drop-in extension for a Go2 Extreme Parkour codebase — same depth visual pipeline, same Go2 hardware; just replace the MLP policy with sparse MoE.

**Critical caveat:** Preprint; anonymized code only; no formal release; framework unstated.

---

## 5. Parkour in the Wild: Learning a General and Extensible Agile Locomotion Policy Using Multi-Expert Distillation and RL Fine-tuning

- **arXiv:** 2505.11164 — https://arxiv.org/abs/2505.11164
- **Authors:** Nikita Rudin, Junzhe He, Joshua Aurand, Marco Hutter
- **Lab/Venue:** ETH RSL + NVIDIA Switzerland — **IJRR 2025 (peer-reviewed)**

**Key NEW algorithm/architecture idea (over Extreme Parkour / ANYmal Parkour / DTC):**
Addresses the skill-combination bottleneck of single-policy parkour:
1. Train 9 terrain-specific expert policies via RL (experts use elevation maps).
2. **Distill into a foundation policy via DAgger** (student uses depth images), with zero-mean Gaussian action noise during distillation to prevent overfitting.
3. **RL fine-tune** on a broader terrain set including real-world 3D scans; repeated fine-tuning adapts to new terrains without restart.
Enables CONTINUOUS skill blending (vs. discrete skill selection in ANYmal Parkour). Stabilization tricks: action noise robustness, conservative RL hyperparams, critic pre-training before policy update. Architecture: CNN (3 conv layers, per-image) + LSTM + MLP at 50 Hz on onboard CPU. State-of-the-art depth noise model for sim-to-real: edge pixel shuffling/removal + Perlin-noise holes + Gaussian blur.

**Constraint fit:**
- No deploy-obs bloat: depth + proprioception + commands.
- No full env rewrite: distillation curriculum + noise model are portable standalone ideas; the 9-expert stage is high overhead.
- Layers onto teacher-student PPO: partially (it IS a multi-expert teacher-student framework).

**Code + framework:** No code released. Framework: parallel sim, RSL-RL/legged_gym style; symmetry data augmentation; position-based task description.

**Real-robot deploy evidence:** ANYmal D, four RealSense D435i cameras, 50 Hz onboard CPU; navigates complex environments with agility/robustness.

**Integration relevance:** The DAgger distillation curriculum, depth noise model, and RL fine-tuning protocol are directly portable to Go2. The 9-expert stage is the high-overhead part.

**Critical caveat:** ANYmal D hardware (heavier, different morphology), NOT Go2. No code. 3-stage pipeline (9 experts + distillation + RL fine-tune) is high training overhead.

---

## 6. AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding

- **arXiv:** 2601.08485 — https://arxiv.org/abs/2601.08485
- **Authors:** Chong Zhang, Victor Klemm, Fan Yang, Marco Hutter
- **Lab/Venue:** ETH RSL — **Under journal review, January 2026**

**Key NEW algorithm/architecture idea (over Extreme Parkour / DTC):**
1. **Attention-based neural map encoder** replaces the standard height-scan encoder — extracts local AND global terrain features with attention-based saliency focus, producing an interpretable, generalized embedding for RL control.
2. **Learning-based uncertainty-aware mapping pipeline** — neural networks convert depth observations into local elevation maps WITH per-cell uncertainty, fused with odometry; robust to noise and occlusions.
3. Mapping pipeline integrates with parallel simulation so controllers train with online mapping (aids sim-to-real).

**Constraint fit:**
- No deploy-obs bloat: replaces existing terrain encoder.
- No full env rewrite: architectural swap in the perception stack.
- Layers onto teacher-student PPO: yes — teacher uses map encoder, student distills.

**Code + framework:** Project page https://sites.google.com/leggedrobotics.com/ame-2 — no public repo found. Framework: ETH RSL (legged_gym/RSL-RL based).

**Real-robot deploy evidence:** ANYmal-D quadruped + LimX TRON1 biped; real-world confirmed. ~95% success on training terrains, ~82% on unseen.

**Integration relevance:** Attention map encoder is an architectural swap in the perception stack; uncertainty-aware mapping directly relevant to Go2 depth robustness.

**Critical caveat:** ANYmal hardware, not Go2. No code released. Paper still under review.

---

## 7. Agile But Safe (ABS) + Successor BAS

### 7a. Agile But Safe (ABS)
- **arXiv:** 2401.17583 — https://arxiv.org/abs/2401.17583
- **Authors:** Tairan He, Chong Zhang, et al.
- **Lab/Venue:** LeCAR Lab, CMU + ETH Zürich — **RSS 2024 (peer-reviewed)**

### 7b. BAS: Bridging Adaptivity and Safety
- **arXiv:** 2501.04276 — https://arxiv.org/abs/2501.04276
- **Authors:** Yichao Zhong, Chong Zhang, Tairan He, Guanya Shi
- **Venue:** **L4DC 2025 (peer-reviewed)** — project page https://adaptive-safe-locomotion.github.io/

**Key NEW algorithm/architecture idea:**
- ABS: a **learned control-theoretic reach-avoid (RA) value network** governs switching between an agile policy and a recovery policy, achieving collision-free high-speed navigation (3.1 m/s). Trains agile policy, RA network, recovery policy, and exteroception representation network jointly in sim.
- BAS: adds a **physical parameter estimator** concurrently trained with the agile policy; agile policy and RA network are both conditioned on physical params (adaptive); on-policy fine-tuning phase for the estimator. Result: +19.8% speed, 2.36x lower collision rate vs. ABS in the real world.

**Constraint fit:**
- The RA safety layer is architecturally separable and portable.
- Not a terrain-traversal/parkour paper — obstacle avoidance at speed.

**Code + framework:** ABS: **https://github.com/LeCAR-Lab/ABS** (PUBLIC). Framework: Isaac Gym + legged_gym + RSL-RL. BAS: no public code found.

**Real-robot deploy evidence:** ABS on Unitree Go1 EDU (authors explicitly state code is "only for our hardware system" — NO Go2 support). BAS: real-world Go1.

**Integration relevance:** The reach-avoid value network concept is portable to Go2 as a separable safety layer; full ABS code transfer is non-trivial (Go1-specific).

**Critical caveat:** ABS code is Go1-only with no Go2 porting path provided. Not a parkour paper (collision-free navigation, not terrain traversal). BAS has no public code.

---

## 8. Agile Continuous Jumping in Discontinuous Terrains

- **arXiv:** 2409.10923 — https://arxiv.org/abs/2409.10923
- **Authors:** Yuxiang Yang, Guanya Shi, Changyi Lin, Xiangyun Meng, Rosario Scalise, Mateo Guaman Castro, Wenhao Yu, Tingnan Zhang, Ding Zhao, Jie Tan, Byron Boots
- **Lab/Venue:** Google DeepMind / CMU / UW — **ICRA 2025 (peer-reviewed)**

**Key NEW algorithm/architecture idea (over Extreme Parkour):**
**Hierarchical RL + model-based hybrid** for sequential continuous jumping:
1. Learned heightmap predictor for robust terrain perception.
2. RL-based centroidal-level motion policy for versatile, terrain-adaptive planning.
3. Low-level model-based leg controller for accurate motion tracking (with hardware-specific modeling to minimize sim-to-real gap).
Enables AGILE CONTINUOUS sequential jumps (vs. individual vaults) on human-sized stairs and sparse stepping stones.

**Constraint fit:**
- Hybrid RL + model-based does NOT plug into pure end-to-end PPO easily.
- The centroidal RL planner + heightmap predictor could augment as a high-level planning layer.

**Code + framework:** Project page https://yxyang.github.io/jumping_cod/ — code not confirmed public from search. Framework: RL (not specified).

**Real-robot deploy evidence:** Unitree Go1; continuous jumps on stairs and stepping stones.

**Integration relevance:** The centroidal RL planner + heightmap predictor could be added as a high-level planning layer for sequential stair/gap jumping skills.

**Critical caveat:** Go1 hardware; hybrid RL+model-based complicates the pure end-to-end PPO paradigm; code not confirmed public.

---

## 9. High-Speed Control and Navigation for Quadrupedal Robots on Complex and Discrete Terrain

- **arXiv:** 2506.02835 — https://arxiv.org/abs/2506.02835
- **Authors:** Hyeongjun Kim, Hyunsik Oh, Jeongsoo Park, Yunho Kim, Donghoon Youm, Moonkyu Jung, Minho Lee, Jemin Hwangbo
- **Lab/Venue:** KAIST (Hwangbo group) — **Science Robotics 2025 (DOI 10.1126/scirobotics.ads6192, peer-reviewed)**

**Key NEW algorithm/architecture idea:**
**Hierarchical planner + tracker** pipeline. Planner finds physically feasible foothold plans via sampling-based optimization with fast sequential filtering using heuristics + a neural network. Tracker trained with competitively-generated foothold distributions. Science Robotics-level validation.

**Constraint fit:**
- Hierarchical MPC/sampling + RL is NOT a drop-in for pure PPO.
- Algorithmic ideas (foothold sampling + learned filter) portable but require significant engineering.

**Code + framework:** Not released (dataset on Dryad). Framework: custom KAIST (not specified as IsaacGym/IsaacLab).

**Real-robot deploy evidence:** Raibo (KAIST in-house robot); runs on vertical walls, 1.3 m gap jumps, stepping stones at 4 m/s, autonomous nav over 30° ramps/stairs/boxes.

**Integration relevance:** Foothold-sampling + learned-filter ideas are portable but require substantial engineering; not directly layerable onto teacher-student PPO.

**Critical caveat:** Proprietary Raibo robot (not commercially available, not Go2). No code. Hierarchical pipeline not compatible with PPO-based codebase.

---

## 10. Helpful DoggyBot: Open-World Object Fetching using Legged Robots and Vision-Language Models

- **arXiv:** 2410.00231 — https://arxiv.org/abs/2410.00231
- **Lab/Venue:** CMU (Pathak-adjacent) — October 2024
- **Related:** Playful DoggyBot (arXiv 2409.19920, Xin Duan, Ziwen Zhuang, Hang Zhao, Soeren Schwertfeger; code https://github.com/playful-doggybot/playful-doggybot)

**Key NEW algorithm/architecture idea:**
Whole-body loco + manipulation: front-mounted gripper, low-level controller trained in sim with egocentric depth for agile skills (climbing, whole-body tilting), pre-trained VLM + third-person fisheye + egocentric RGB for semantic command generation; SAM2 integration for vision.

**Constraint fit:** Locomotion component is an adaptation of prior work; novelty is VLM integration + manipulation, not parkour.

**Code + framework:** **https://github.com/WooQi57/Helpful-Doggybot** (PUBLIC). Framework: Isaac Gym + legged_gym + rsl_rl (Python 3.8, requires Isaac Gym binaries).

**Real-robot deploy evidence:** Unitree Go2 + gripper (15 kg), Jetson Orin NX; 60% zero-shot success on open-world fetch tasks (e.g., fetch toy after climbing a queen-sized bed) in 2 unseen environments with no real-world training.

**Integration relevance:** Public code on exact Go2 hardware, but locomotion novelty is modest; main value is VLM + whole-body manipulation integration.

**Critical caveat:** Locomotion is adapted prior controller; contribution is VLM/manipulation; Go2+gripper-specific.

---

## 11. Acrobotics: A Generalist Approach to Quadrupedal Robots' Parkour

- **arXiv:** 2509.02727 — https://arxiv.org/abs/2509.02727
- **Authors:** Guillaume Gagné-Labelle, Vassil Atanassov, Ioannis Havoutis
- **Lab/Venue:** Oxford Robotics Institute — **TAROS 2025 (peer-reviewed)**

**Key NEW algorithm/architecture idea:**
A single **generalist RL policy** that rivals state-of-the-art MoE specialist policies while using only **25% as many agents** during training. Simple, easy-to-implement pipeline; focuses on essential design choices for robust, convergent generalist RL agents. Challenges the "MoE is necessary" assumption.

**Constraint fit:** Sim-only — no real-robot deploy reported.

**Code + framework:** Google Drive supplementary only; no public GitHub found. Framework: not specified.

**Real-robot deploy evidence:** NONE reported — simulation only.

**Integration relevance:** Useful as a counterpoint to MoE approaches (#3, #4) — suggests a well-tuned generalist policy may suffice; but no deploy evidence or public code.

**Critical caveat:** Sim-only; no real-robot deploy; code not on GitHub (Google Drive only).

---

# Ranked Top-3 Freshest Works Most Worth Studying/Adopting

Prioritization: (a) genuine novelty over Extreme Parkour, (b) open deploy code, (c) Go2 or near-identical hardware.

### #1 — REAL (arXiv 2603.17653, HKUST Guangzhou, March 2026)
**Justification:** Exact Go2 + Isaac Gym match, direct ablation vs. Extreme Parkour, and introduces FiLM-Mamba student backbone + physics-guided EKF + consistency loss-gating as a drop-in architectural upgrade to the existing teacher-student PPO pipeline — highest novelty-to-integration ratio found.
**Code availability:** Planned release (https://jialonglong.github.io/REAL_wb/) — NOT yet live; preprint.

### #2 — MoE-Loco (arXiv 2503.08564, IROS 2025)
**Justification:** Only candidate with verified OPEN code AND exact Go2 + Jetson hardware AND peer review; sparse MoE backbone is a clean MLP swap with zero env changes; emergent skill composition directly useful for multi-terrain parkour.
**Code availability:** PUBLIC — https://github.com/hrh6666/MoE-Loco (Isaac Gym).

### #3 — PUMA (arXiv 2601.15995, ZJU, January 2026)
**Justification:** The egocentric polar foothold prior (4 floats appended to obs + PAS curriculum) is the most practically integrable geometric idea — adds structured foothold reasoning to PPO with minimal architecture change, and directly beats Extreme Parkour on wall-assisted gap traversal (96.9% vs. 76.5%). Hardware-agnostic idea despite Lite3 (not Go2) testbed.
**Code availability:** Not released; preprint.

---

## Honorable Mentions (high value, secondary)
- **Quadruped Parkour MoE + Vision** (arXiv 2604.19344, UCL, April 2026): Direct Go2 visual MoE parkour, 2x success vs. MLP; anonymized OSF code only.
- **Parkour in the Wild** (arXiv 2505.11164, ETH RSL/IJRR 2025): Best DAgger distillation curriculum + depth noise model; ANYmal only, no code, but ideas portable.
- **Agile Continuous Jumping** (arXiv 2409.10923, Google/CMU, ICRA 2025): Best sequential jumping architecture for adding stair/gap-sequence skills; Go1, code unconfirmed.

## Recommended Study Order for Go2 IsaacLab Extreme-Parkour Codebase
1. **MoE-Loco first** — open code, Go2+Jetson exact match, MLP→MoE swap with zero env changes; run a baseline experiment immediately.
2. **REAL second** — highest architectural novelty for your teacher-student PPO; plan the FiLM-Mamba student swap + EKF now, integrate when code drops.
3. **PUMA third** — implement the 4-float foothold prior + PAS curriculum manually; smallest obs change with the largest documented win over Extreme Parkour.

---

## Verified Source Links
- REAL: https://arxiv.org/abs/2603.17653 | code (planned): https://jialonglong.github.io/REAL_wb/
- PUMA: https://arxiv.org/abs/2601.15995
- MoE-Loco: https://arxiv.org/abs/2503.08564 | code: https://github.com/hrh6666/MoE-Loco | page: https://moe-loco.github.io/
- Quadruped Parkour MoE+Vision: https://arxiv.org/abs/2604.19344 | OSF: https://osf.io/v2kqj/files/github?view_only=7977dee10c0a44769184498eaba72e44
- Parkour in the Wild: https://arxiv.org/abs/2505.11164
- AME-2: https://arxiv.org/abs/2601.08485 | page: https://sites.google.com/leggedrobotics.com/ame-2
- ABS: https://arxiv.org/abs/2401.17583 | code: https://github.com/LeCAR-Lab/ABS
- BAS: https://arxiv.org/abs/2501.04276 | page: https://adaptive-safe-locomotion.github.io/
- Agile Continuous Jumping: https://arxiv.org/abs/2409.10923 | page: https://yxyang.github.io/jumping_cod/
- KAIST High-Speed Discrete Terrain: https://arxiv.org/abs/2506.02835
- Helpful DoggyBot: https://arxiv.org/abs/2410.00231 | code: https://github.com/WooQi57/Helpful-Doggybot
- Playful DoggyBot: https://arxiv.org/abs/2409.19920 | code: https://github.com/playful-doggybot/playful-doggybot
- Acrobotics: https://arxiv.org/abs/2509.02727
