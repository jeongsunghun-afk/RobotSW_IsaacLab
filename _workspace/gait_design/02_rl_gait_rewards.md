# 02 — RL Gait-Reward Design for Quadrupeds (Literature & Implementation Survey)

> Task #2 deliverable. Scope: how RL reward design *induces / prescribes* quadruped gait, with exact reward forms where recoverable, the gait each induces, sources, and **applicability to our Go2 parkour env**.
>
> **Our env constraints (applied to every "applicability" verdict below):**
> - **Forward-only** command (vx>0, vy=0, yaw≈0); no omni-directional gait library needed.
> - **Obs has NO contact sensor** (sim-to-real). Contact may be used *inside reward only* (privileged at train time).
> - **height-scan obs** present (terrain geometry visible to policy).
> - **`total_reward` clip(min=0)**: oversized penalties zero out the whole step's reward → kill gradient/learning. Penalty terms must be *small / bounded*, never able to dominate the positive sum.
> - Weights are **episode-normalized** (per-step contributions already ×dt-scaled in logging; don't double-count).
> - **Go2 deployable** (12-DOF, standard actuators).
>
> Measured pathologies this survey must address: bounding on flat, 4-foot pronk on obstacles (stair 27.7% all-airborne), split-jumps (front land → re-takeoff → rear land; **fore-hind coordination absent**). Existing `feet_gait_pairing` term is weight=0, is_flat-gated, gradient-saturated (clamp 0.3 s).

---

## 0. Executive summary — what to steal

| Need | Best-fit technique | Why | Contact-obs needed? |
|---|---|---|---|
| **Fore-hind coordination** | **Phase-clock periodic reward** (Siekmann / Walk-These-Ways) with a **trot/diagonal offset** between front & hind legs | Directly prescribes *when* each foot is in stance vs swing → forces front and hind into a fixed phase relationship instead of emergent split-jumps | **No** — clock is a time signal, contact used in reward only |
| **Suppress unnecessary pronk** | (a) **anti-synchrony / no-fly term** (penalize all-feet-airborne or all-feet-synchronized), and/or (b) **time-reversal + temporal symmetry reward** (Symmetry-Guided RL), and/or (c) phase-clock with non-pronk offset | Pronk = all four legs in phase (offset 0,0,0). Any reward that *rewards a non-zero phase offset* or *penalizes simultaneous lift-off* removes the pronk equilibrium | **No** for clock/symmetry; contact-in-reward for no-fly |
| **Longer, deliberate steps** | **`feet_air_time` reward** (Rudin/legged_gym) | Rewards step duration on first contact; discourages tiny micro-taps | contact-in-reward only |

**Single most important design lever for us:** add a **deterministic per-leg phase clock** (a function of episode time, *not* of sensed contact) as both (i) a small obs signal and (ii) the schedule that a swing/stance reward tracks. This is exactly how Siekmann (Cassie) and Walk-These-Ways (Go1) get *prescribed* gaits **without contact sensing in observation** — the policy sees a clock, the reward uses privileged contact. That satisfies our no-contact-obs constraint while giving us explicit fore-hind timing control.

---

## 1. Periodic / phase-based gait reward

### 1.1 Siekmann et al. — *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition* (ICRA 2021)
- Source: Siekmann, Godse, Fern, Hurst. arXiv:2011.01387. https://arxiv.org/abs/2011.01387 (robot: Cassie)
- **Idea:** each leg has a **phase clock** `φ ∈ [0,1)` that cycles over the gait period. The period is split into a **swing region** and a **stance region** by a (probabilistic / von-Mises-smoothed) indicator. Two coefficient functions, `I_swing(φ)` and `I_stance(φ)`, gate two costs:
  - **Swing phase:** penalize **foot ground-reaction force** → reward zero force ⇒ foot should be in the air.
  - **Stance phase:** penalize **foot velocity** → reward zero velocity ⇒ foot should be planted.
  - Composite reward (schematically):
    `r_periodic = Σ_foot [ I_swing(φ_foot)·exp(−‖F_foot‖²/σ_f) + I_stance(φ_foot)·exp(−‖v_foot‖²/σ_v) ]`
  - **Gait is selected purely by the per-leg phase offsets** (which clock each foot is on). Offsetting left/right (or, for quadrupeds, front/hind) clocks by half a period gives an alternating gait; zero offset gives a hopping/pronk-like synchronous gait.
- **Induces:** *any* prescribed gait — standing, walking, hopping, running, skipping — by choice of offsets + swing/stance ratio (duty factor).
- **Key sim-to-real fact:** the policy's *observation* includes a **clock input** (e.g. `sin φ, cos φ`), **not** a force/contact sensor. Contact/force is used **only in the reward** at train time. ✔ matches our constraint.
- **Applicability to our parkour env — HIGH (top candidate):**
  - Port the swing-force / stance-velocity pair, driven by a **front-vs-hind half-period offset** to enforce trot-like fore-hind coordination.
  - Add `sin φ, cos φ` (1 shared clock, or 2 for front/hind) to obs — this is a time signal, **sim-to-real safe, not contact**.
  - **clip(min=0) caution:** both terms are bounded in `[0,1]` per foot (exp form) and are *positive rewards* (reward low force in swing / low velocity in stance), so they *add* to the positive sum rather than risk zeroing it. Good fit. Prefer the **positive-reward (exp)** formulation over a raw negative penalty exactly because of clip(min=0).

### 1.2 Margolis & Agrawal — *Walk These Ways* (CoRL 2022)
- Source: Margolis, Agrawal. arXiv:2212.03238. https://arxiv.org/abs/2212.03238 ; code https://github.com/Improbable-AI/walk-these-ways (robot: Unitree Go1). **Direct quadruped instance of Siekmann's scheme.**
- **Desired contact state** from phases via cumulative-normal (von-Mises-like) smoothing:
  `C_foot^cmd = Φ(t_foot,σ)·(1−Φ(t_foot−0.5,σ)) + Φ(t_foot−1,σ)·(1−Φ(t_foot−1.5,σ))`
  where `Φ` = Gaussian CDF; `t_foot` = that foot's phase. `C^cmd∈[0,1]` = "should this foot be planted now."
- **Augmented auxiliary rewards (Table 1):**

  | Term | Formula | Weight |
  |---|---|---|
  | Swing (force) | `Σ_foot [1−C_foot^cmd]·exp(−‖f^foot‖²/σ_cf)` | −0.08 |
  | Stance (velocity) | `Σ_foot [C_foot^cmd]·exp(−‖v_xy^foot‖²/σ_cv)` | −0.08 |

  (These are the quadruped form of Siekmann's two terms.)
- **Gait library via three timing offsets `(θ₁,θ₂,θ₃)`** between foot pairs:
  - **Pronk** `(0,0,0)` — all four legs in phase (**= our pathology**)
  - **Trot** `(0.5,0,0)` — diagonal pairs alternate
  - **Pace** `(0,0,0.5)` — lateral pairs alternate
  - **Bound** `(0,0.5,0)` — **front pair vs rear pair alternate**
  - Also conditioned on **duty factor**, footswing height, body height/pitch, frequency.
- **Induces:** continuously-interpolable gaits; the operator *picks* the gait by `(θ,duty,freq)`.
- **Applicability — HIGH:**
  - We don't need the full MoB *conditioning* (forward-only). We **fix** the offsets to a single target gait (a **trot** `θ₁=0.5`, or a mild **bound** with a *non-trivial* fore-hind offset) and bake it into the reward. The point for us is not gait diversity but **removing the pronk fixed point** by making `(0,0,0)` un-rewarded.
  - **Anti-pronk mechanism:** `C^cmd` for a trot/bound schedule is *never* "all four planted/all four lifted simultaneously," so the swing-force + stance-velocity rewards actively *penalize* the synchronized pronk/split-jump — exactly what we want.
  - **Fore-hind coordination mechanism:** choose offsets so front and hind legs have a fixed phase difference (trot diagonal, or bound half-period) → the policy can no longer get reward from "front land, re-takeoff, rear land."
  - clip(min=0): same note as 1.1 — keep the exp/positive form, modest weight (≈0.08-scale), bounded.
  - **Caveat / gap:** WTW conditions on contact-target inputs but the *clock/phase* fed to the policy is a time signal; contact itself stays out of obs. We must verify our implementation feeds **phase**, not measured contact, to obs. (Confirmed in paper that gait timing is a commanded/clock quantity.)

### 1.3 Decentralized phase oscillators / CPG-style (context)
- Source: "Learning Emergent Gaits with Decentralized Phase Oscillators…" arXiv:2402.08662. https://arxiv.org/pdf/2402.08662
- **Idea:** couple per-leg oscillators; reward shapes the coupling so gaits *emerge* rather than being hard-prescribed. Observation = oscillator phase (clock), not contact.
- **Applicability — MEDIUM:** more machinery than we need; but confirms the *phase-as-observation, contact-in-reward* pattern is standard and sim-to-real safe. Useful as a fallback if a hard-prescribed schedule proves too rigid over varied terrain.

---

## 2. Foot-contact schedule / air-time reward (legged_gym lineage)

### 2.1 Rudin et al. — *Learning to Walk in Minutes* / legged_gym (CoRL 2021)
- Source: Rudin, Hoeller, Reist, Hutter. arXiv:2109.11978. https://arxiv.org/abs/2109.11978 ; code https://github.com/leggedrobotics/legged_gym (robot: ANYmal). **This is our reward lineage (Extreme Parkour is a legged_gym fork).**
- **`_reward_feet_air_time` (exact, from source):**
  ```python
  contact = self.contact_forces[:, self.feet_indices, 2] > 1.
  contact_filt = torch.logical_or(contact, self.last_contacts)
  self.last_contacts = contact
  first_contact = (self.feet_air_time > 0.) * contact_filt
  self.feet_air_time += self.dt
  rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward per foot at touchdown
  rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1               # only when commanded to move
  self.feet_air_time *= ~contact_filt                                        # reset on contact
  return rew_airTime
  ```
  - **Reward = (air_time − 0.5 s) summed over feet, paid only at the touchdown step**, only if commanded speed > 0.1. Encourages each step to last ≈≥0.5 s (longer, deliberate steps; discourages micro-stutter/scuffing).
- Other relevant terms in same file:
  - `_reward_collision`: `Σ 1·(‖F_contact‖ on penalised bodies > 0.1)` — penalize knee/base contact.
  - `_reward_stumble`: penalize horizontal-dominant foot force (`F_xy > 5·|F_z|`) — toe-stub on edges.
  - `_reward_feet_contact_forces`: `Σ (‖F_foot‖ − F_max).clip(min=0)` — cap impact force (relevant to landing-shock; see doc 03).
  - **There is no `_reward_no_fly` in upstream legged_gym** (verified by reading source). It exists in some forks/community variants.
- **Applicability of `feet_air_time` — MEDIUM/HIGH but with a sharp caveat:**
  - ✔ contact used in reward only (no obs), Go2-deployable, already in our lineage.
  - ⚠ **air_time *rewards* long airborne phases per foot — this can REWARD pronk/jumps** (all feet long-airborne → big air-time). On its own it does **not** fix, and can *worsen*, our pronk pathology. Use it **only paired** with a phase/anti-synchrony term (§1, §2.2) that forbids simultaneous lift-off. By itself: **not** an anti-pronk tool.
  - ⚠ clip(min=0): `air_time − 0.5` can be negative for short steps; summed it's usually small, but verify it can't dominate. Many forks clamp the per-foot reward at first contact to be ≥0 to avoid punishing — consider `(air_time−0.5).clip(max=…)` and a modest weight.

### 2.2 "No-fly" / contact-count / anti-synchrony terms (community + cited works)
- Sources (descriptive, formulas vary by impl — flagged as such):
  - SayTap (Tang et al., 2023, arXiv:2306.07580, https://arxiv.org/pdf/2306.07580): desired **foot-contact pattern** is the control input; reward = **match between realized contact and a desired binary contact schedule**. The schedule (a "0/1 per foot over time" template) is generated externally and the policy is *conditioned on the pattern*, with a contact-matching reward. Conceptually identical to a phase clock + swing/stance reward, just expressed as a binary template.
  - Hierarchical RL for quadruped locomotion (arXiv:2506.20036) and others note: *"quadrupeds can reproduce bounding while keeping all feet on the ground… reward terms are needed to ensure flight phases in between"* and conversely terms to **penalize flying gaits**.
- **Canonical "no-fly" form (community pattern, not from one canonical paper — stated as such):**
  `r_nofly = −1 · ( Σ_foot contact_foot == 0 )` i.e. penalize the step when **zero feet** are in contact (full flight). Or symmetric anti-pronk: penalize when **all four** lift simultaneously.
- **Anti-pronk reward candidate (our design, grounded in the above):**
  - Let `n_contact = #feet with F_z > threshold`. Penalize the all-airborne state: `r = −w·𝟙[n_contact == 0]` (small `w`, bounded). This directly removes the 27.7%-all-airborne stair pronk.
  - **Better (gradient-friendly, clip-safe):** *reward* having ≥1 (or ≥2 for trot) feet planted: `r = +w·min(n_contact, 2)/2`, bounded in [0,1], positive → safe under clip(min=0).
- **Applicability — HIGH for anti-pronk:** contact-in-reward only ✔, cheap, Go2-safe. **This is the most direct, lowest-risk "suppress pronk" lever.** Pair with a phase term for fore-hind timing.
- **Gap:** there is no single authoritative "no_fly" paper with a fixed formula; it is a widely-used community term. Stated honestly rather than fabricating a citation.

---

## 3. Symmetry / coordination reward (prescribe trot / fore-hind sync)

### 3.1 Symmetry-Guided RL for quadruped gaits (Ding et al., 2024/2025)
- Source: *Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy…* arXiv:2403.10723 (v1 title *Leveraging Symmetries in Gaits for RL*) and updated arXiv:2510.10455. https://arxiv.org/abs/2403.10723 (robots: Petoi Bittle, Unitree Go2).
- **Three symmetries baked into the reward (no reference trajectory needed):**
  1. **Temporal symmetry** — encourage **periodicity** of motion (state at `t` ≈ state at `t+T`). Reward form: penalize difference between a state/feature now and one period ago.
  2. **Morphological symmetry** — left/right (and our front/back) limbs play symmetric roles (cf. our mirror augmentation, but expressed as reward).
  3. **Time-reversal symmetry** — **"front and back legs move in similar fashion"**; generates solutions where fore and hind limbs mirror through a half-period shift. **This is the formal statement of fore-hind coordination.**
- **Reward = command-tracking + smoothness + symmetry terms.** Enables trot/bound/half-bound/gallop and smooth transitions *without* predefined trajectories.
- **Induces:** clean, periodic, fore-hind-coordinated gaits (trot etc.) via the symmetry terms.
- **Applicability — HIGH for fore-hind coordination (the "transfer" candidate):**
  - **Time-reversal / front-back symmetry term is exactly "fore-hind coordination as a reward."** It says: the rear-leg trajectory should equal the front-leg trajectory shifted by a half-period. That kills split-jumps (front and rear *must* be coupled), and it needs **only proprioception + a phase/time variable — no contact obs.** ✔✔
  - **Temporal-symmetry (periodicity) term** discourages the irregular, aperiodic split-jump/pronk bursts by rewarding a steady cycle.
  - clip(min=0): implement as **bounded exp-similarity rewards** (positive) rather than raw L2 penalties so they add to the positive sum.
  - **Caveat:** their hierarchy/gait-transition machinery is overkill for forward-only; take just the **time-reversal + temporal symmetry reward terms**, not the hierarchy.

### 3.2 Spatio-temporal symmetry via half-period time-shift — does it exist? **Yes (above).**
- The task asked specifically whether "spatio-temporal symmetry for trot via half-period time-shift" is a real technique. **Confirmed:** §3.1's *time-reversal symmetry* and *temporal symmetry* are exactly this — a half-period (T/2) time-shifted comparison between front and hind (and between a leg and itself one period later). It is the cleanest literature basis for a **fore-hind coordination reward with no contact obs.**

### 3.3 Our existing `feet_gait_pairing` (Spot-style sync/async) — origin note
- Our env's `feet_gait_pairing` term resembles Boston-Dynamics-Spot-style **diagonal sync/async** gait rewards (reward FL↔RR and FR↔RL to be *in phase*, lateral pairs *out of phase* → trot). I could **not** locate a single canonical public paper that is the definitive origin of a "Spot GaitReward" with this exact name — flagged as a **gap** (likely internal/community provenance). The *mechanism* (diagonal-pair phase-locking reward) is well attested (WTW trot offset §1.2, symmetry §3.1). Our term being **weight=0 + is_flat-gated + clamp 0.3 s saturated** means it currently contributes nothing on obstacles — re-enabling/redesigning it (ungated, with a clip-safe positive form) is a concrete near-term action that doc 04 should weigh.

---

## 4. Parkour / jump context — is gait prescribed or terrain-forced?

### 4.1 Extreme Parkour (Cheng, Kumar, Shi, Pathak, 2023) — **our baseline lineage**
- Source: Cheng, Shi, Agarwal, Pathak. *Extreme Parkour with Legged Robots.* arXiv:2309.14341. https://ar5iv.labs.arxiv.org/html/2309.14341 (robot: Unitree A1/Go-class).
- **Reward design (from paper):**
  - **Inner-product velocity/heading reward:** `r_tracking = min(⟨v, d̂_w⟩, v_cmd)` — reward velocity *projected onto the world-frame waypoint direction*, capped at command. World-frame (not body-frame) specifically so the robot can't "turn around the obstacle" to farm reward.
  - **Feet-clearance / edge penalty:** `r_clearance = −Σ (c_i · M[p_i])` — penalize stepping within ~5 cm of terrain edges.
  - **Stylized orientation (optional):** `r_stylized = W·[0.5·⟨v̂_fwd, ĉ⟩ + 0.5]²` (used to toggle handstand).
- **CRITICAL finding for our project:** **there is NO explicit gait, footfall-pattern, or jump reward.** The paper states the reward terms *"typically lead to a gait that uses all four legs,"* and all skills (high/long jumps, handstand, foot placement) **emerge from terrain geometry + the inner-product velocity reward + regularization.** Jumping is **terrain-forced, not rewarded.**
- **Implication:** our pronk/split-jump pathology is the *predicted failure mode* of this design — with **nothing in the reward constraining footfall pattern**, the policy is free to settle into whatever clears the terrain, including pronk and uncoordinated split-jumps. **The fix is therefore additive: introduce the footfall/coordination signal that Extreme Parkour deliberately omits** (consistent with our project's "additive-only, no env rewrite" constraint). §1–§3 are exactly such additions.

### 4.2 Robot Parkour Learning (Zhuang et al., CoRL 2023)
- Source: Zhuang, Fu, Wang, Atkeson, Schwertfeger, Finn, Zhao. *Robot Parkour Learning.* arXiv:2309.05665 (project robot-parkour.github.io). (robot: A1.)
- **Approach:** per-skill specialist policies (climb, leap, crawl, tilt) each trained with **soft→hard dynamics constraints** and skill-specific rewards, then **distilled** into one vision policy. Skills are shaped by **terrain + a forward/penetration reward**, again **without a prescribed footfall schedule** — the jump/leap emerges from the gap terrain and a velocity/penetration objective. Reference-free (no motion capture).
- **Applicability:** confirms the field norm — parkour jumps are **terrain-and-velocity-driven, not gait-prescribed**. Reinforces that adding a footfall/coordination reward is novel-but-safe additive shaping, not contradicting baseline practice.

### 4.3 Adjacent jump-reward works (for completeness)
- *Curriculum-Based RL for Quadrupedal Jumping: A Reference-free Design* (arXiv:2401.16337) — explicit **jump** reward via curriculum (reward takeoff/landing states). Relevant if we ever want to *encourage* a clean coordinated jump rather than only suppress bad ones, but it **rewards** jumping (opposite of our "suppress unnecessary pronk" goal) → use only its *landing-coordination* ideas, not its takeoff bonus. (Landing-shock detail → doc 03.)
- *SF-TIM* (arXiv:2408.00486), *PIE* (arXiv:2408.13740), *SoloParkour* (arXiv:2409.13678) — parkour variants; none introduces a footfall-pattern reward that suppresses pronk. (Scanned titles/abstracts; no per-leg coordination reward found — stated as gap, not exhaustively read.)

---

## 5. Concrete reward-term candidates for our env (synthesis → hand to doc 04)

All written in **clip(min=0)-safe, contact-obs-free** form (contact used in reward only; policy sees at most a phase clock = time signal).

### 5.1 Fore-hind coordination (pick ≥1)
- **(A) Phase-clock swing/stance (Siekmann/WTW), trot offset.**
  - Add per-leg phase `φ_i` from a shared clock; front and hind offset by half-period (trot diagonal `θ₁=0.5`).
  - `r = w·Σ_foot [ (1−C_i^cmd)·exp(−‖F_i‖²/σ_f) + C_i^cmd·exp(−‖v_i‖²/σ_v) ]`, `w≈0.05–0.1`, bounded [0, ~1], **positive** → clip-safe.
  - Obs add: `sin φ, cos φ` (1–2 dims). **Sim-to-real safe (clock, not contact).**
- **(B) Time-reversal / front-back symmetry (Symmetry-Guided RL).**
  - `r = w·exp(−‖q_hind(t) − q_front(t−T/2)‖²/σ)` (joint-angle or foot-traj similarity, half-period shifted). Positive, bounded → clip-safe. **No clock in obs strictly required** (uses history), but a phase var helps.

### 5.2 Suppress unnecessary pronk / split-jump (pick ≥1)
- **(C) Reward feet-on-ground count (anti-flight), clip-safe positive form.**
  - `r = w·min(n_contact, k)/k` with `k=2` (trot) — rewards ≥2 feet planted; pronk (n=0) gets 0. `w` small.
- **(D) Penalize simultaneous lift-off (anti-pronk), bounded.**
  - `r = −w·𝟙[n_contact == 0]`, tiny `w` so it can't break clip(min=0). (Prefer (C) over (D) for gradient smoothness.)
- **(E) Periodicity / temporal-symmetry reward** to kill aperiodic split-jump bursts: `r = w·exp(−‖s(t) − s(t−T)‖²/σ)` on a gait-relevant feature.
- ⚠ **Do NOT use bare `feet_air_time` (§2.1) as the anti-pronk tool** — it can reward the very airborne phases we want to suppress. Only include it *with* (C)/(A) and a capped magnitude.

### 5.3 Re-examine existing `feet_gait_pairing`
- Currently weight=0, is_flat-gated, clamp-0.3-saturated → contributes nothing on obstacles (the exact place pronk appears). Cheapest first experiment: **ungate it from is_flat, give it a small positive clip-safe weight, and remove/raise the 0.3 s clamp** so its gradient is non-zero on stairs/gaps. This is the minimal additive change before adding (A)/(B).

---

## 6. Gaps / not-found (no fabrication)
- **Walk-These-Ways PDF** rendered as corrupted binary; formulas in §1.2 were recovered from the ar5iv HTML version and are reproduced as the paper states. von-Mises σ values and full auxiliary-reward weight table beyond the two swing/stance terms were **not** fully extracted — verify against the paper/code before tuning.
- **"Spot GaitReward" canonical origin:** no single definitive public paper located for the exact-named sync/async term; provenance likely internal/community. Mechanism is well-attested elsewhere (§1.2, §3.1). Flagged, not fabricated.
- **`_reward_no_fly`:** confirmed **absent** from upstream legged_gym source (I read it); it is a community/fork pattern. The anti-pronk forms in §5.2 are stated as *our* design grounded in attested principles, not quoted from one canonical paper.
- Parkour variants in §4.3 were scanned at abstract/title level, **not** fully read; "no footfall-pattern reward found" is a best-effort claim over what was read, not an exhaustive proof of absence.
- Exact numeric reward *scales* for most terms are impl-specific; values above (`w≈0.05–0.1`) are **starting suggestions**, to be tuned under our clip(min=0) budget — not taken from the papers verbatim.

---

## Sources
- Siekmann et al., *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition*, ICRA 2021 — https://arxiv.org/abs/2011.01387
- Margolis & Agrawal, *Walk These Ways*, CoRL 2022 — https://arxiv.org/abs/2212.03238 ; code https://github.com/Improbable-AI/walk-these-ways ; ar5iv https://ar5iv.labs.arxiv.org/html/2212.03238
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, CoRL 2021 — https://arxiv.org/abs/2109.11978 ; code https://github.com/leggedrobotics/legged_gym
- Ding, Chen et al. (Jiayu Ding & Xulin Chen, equal contrib., Syracuse Univ.), *Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy* (v1 *Leveraging Symmetries in Gaits for RL*) — https://arxiv.org/abs/2403.10723 ; updated https://arxiv.org/abs/2510.10455
- Tang et al., *SayTap: Language to Quadrupedal Locomotion*, 2023 — https://arxiv.org/pdf/2306.07580
- "Learning Emergent Gaits with Decentralized Phase Oscillators" — https://arxiv.org/pdf/2402.08662
- "Hierarchical RL and Value Optimization for Challenging Quadruped Locomotion" — https://arxiv.org/pdf/2506.20036
- Cheng et al., *Extreme Parkour with Legged Robots*, 2023 — https://ar5iv.labs.arxiv.org/html/2309.14341
- Zhuang et al., *Robot Parkour Learning*, CoRL 2023 — https://arxiv.org/abs/2309.05665
- *Curriculum-Based RL for Quadrupedal Jumping: A Reference-free Design* — https://arxiv.org/pdf/2401.16337
- *SF-TIM* https://arxiv.org/html/2408.00486v1 ; *PIE* https://arxiv.org/html/2408.13740v1 ; *SoloParkour* https://arxiv.org/pdf/2409.13678 (parkour variants, scanned)
