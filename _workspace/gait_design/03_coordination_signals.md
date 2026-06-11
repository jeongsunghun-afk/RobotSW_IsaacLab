# 03 — Fore-Hind Coordination · Landing-Impact · CoT Reward Signals + Sim-to-Real

**Task #3 deliverable** — reward signals that *directly target our measured pathologies* (fore-hind
coordination loss → split-jump, 4-foot simultaneous landing impact, inefficiency), plus sim-to-real
transferability considerations. All claims sourced; gaps stated explicitly. No fabrication.

> **Our pathologies (from project context / sibling tasks):** bounding on flat, 4-foot pronk on
> obstacles (stair 27.7 % all-airborne), **split-jumps** (front feet land → re-takeoff → rear feet
> land; fore-hind coordination absent; re-takeoff/flight stair 0.060 vs flat 0.002). Safety penalties
> are a **NO-GO** (measured impact/saturation below thresholds), but **inefficiency is confirmed**
> (CoT stair 1.49× flat). Goal = **quality** (efficiency + fore-hind coordination), not safety.

> **Hard constraints carried into every recommendation below:**
> (a) **NO contact-sensor in observations** (sim-to-real). Contact/force may be used *inside the
> reward only*.
> (b) `total_reward` is **clip(min=0)** — oversized penalties drive the sum negative, get clipped, and
> kill the learning gradient. Penalties must be small relative to positive headroom.
> (c) Reward weights are **episode-normalized (step_dt already applied)** — do not re-multiply by dt.
> (d) Must remain **Go2-deployable** (proprioceptive-only policy, real actuators).

---

## 0. Key framing fact: our baseline (Extreme Parkour) already uses contact-derived reward terms

This matters for constraint (a). Extreme Parkour (our code's ancestor) already computes several
reward terms *from contact forces and contact state* — entirely inside the reward, never as policy
observation:

- **absolute work penalty** (energy) on joint torques,
- **foot-jerk penalty** = large changes in **foot contact forces** (prevents motor backlash),
- **feet-drag penalty** = horizontal foot velocity *while in contact*,
- **collision penalty** = thigh/calf contacts,
- **feet-air-time** reward,
- **terrain-edge** penalty (foot contact within 5 cm of an edge).

Source: Cheng, Shi, Agarwal, Pathak, *Extreme Parkour with Legged Robots*, CoRL 2023 —
https://arxiv.org/abs/2309.14341 , https://extreme-parkour.github.io/ .

**Implication:** every contact/force-based reward proposed below is constraint-(a)-compatible *by the
same mechanism our baseline already uses* — the force/contact tensor is read in `_get_rewards()` and
never enters `_get_observations()`. The only things forbidden are (i) feeding contact into obs and
(ii) penalties large enough to violate constraint (b).

---

## 1. Energy / CoT penalty → efficient, naturally-coordinated gait

### Signal form
- **Negative motor power** (most common): `r_energy = -w · Σ_i |τ_i · q̇_i|` summed over joints, per
  step. Penalizes total mechanical power magnitude.
- **Positive mechanical work only** (motor-realistic, ignores regenerative/negative power):
  `P_i⁺(t) = max(0, τ_i · q̇_i)`, then `r_energy = -w · Σ_i P_i⁺`.
- **Cost of Transport (CoT)** as a *metric* (not usually a per-step reward because it needs distance):
  `CoT = Σ_i P_i⁺ / (m·g·v)` — useful for evaluation/curriculum gating, harder to use raw as a dense
  reward (division by small v is unstable).

### Effect
Fu, Kumar, Malik, Pathak, *Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged
Robots*, CoRL 2021 — https://arxiv.org/abs/2111.01674 , https://energy-locomotion.github.io/ — is the
canonical result: with **energy minimization as the dominant shaping term**, structured, animal-like
gaits *emerge* on flat/ideal terrain (walk ≈0.375 m/s, trot ≈0.9 m/s, bounce/gallop ≈1.5 m/s, Froude-
number-matched to horses/sheep) **without any hand-specified gait**. On rough terrain the same
principle yields *unstructured* gaits — consistent with animal motor control. Key mechanism for **us**:
a split-jump (front push-off → land → **second** rear push-off) does *two* costly positive-work
takeoffs per stride; a single coordinated push-off does one. An energy/work penalty therefore makes
split-jumps strictly more expensive than coordinated jumps and biases the optimum toward coordination.

### Side-effects / risks
- Too-strong energy weight → robot under-actuates, drags feet, or refuses to clear obstacles (energy-
  min favors *minimal* motion). Must be balanced against the parkour task/clearance reward.
- On rough terrain energy-min alone does **not** produce clean periodicity (Fu et al.) — it needs a
  coordination term too (§3). Energy is necessary-but-not-sufficient for our case.
- Naive `-|τq̇|` can fight velocity-tracking near obstacles where transient high power is required.

### Constraint check
- (a) ✅ uses τ, q̇ — proprioceptive internal quantities, not contact, never in obs.
- (b) ⚠️ **headroom-sensitive.** This is a magnitude penalty that scales with power; on obstacle
  push-offs it spikes. Must be scaled so the per-step sum cannot drive `total_reward` below 0. Our
  baseline already carries an absolute-work penalty, so the safe move is to **re-tune the existing
  weight up modestly**, not add a second uncapped power term.
- (c) ✅ per-step term, weight is episode-normalized — do not re-multiply by dt.
- (d) ✅ deploy-safe and standard; energy-shaped Go2 policies are widely deployed.

---

## 2. Landing-impact / contact-force-mitigation reward (reward-only, never obs)

### Signal form (menu)
- **Peak-GRF / contact-force penalty:** `r = -w · Σ_foot max(0, |F_foot| - F_thresh)²` — penalizes
  contact force above a threshold (soft-landing).
- **Foot-jerk / force-rate penalty:** `r = -w · Σ_foot |F_foot(t) - F_foot(t-1)|` — penalizes *rate of
  change* of contact force (this is exactly Extreme Parkour's existing foot-jerk term).
- **Contact-power penalty:** `r = -w · Σ_foot |F_foot · v_foot|` — penalizes the product of contact
  force and foot velocity (a foot slamming down at high velocity is penalized).
- **Foot-velocity-on-impact penalty:** limit vertical foot velocity at touchdown.

### Effect
Without an impact/force term, feet "hit the surface hard on each landing"; policies that exploit hard
impacts also **transfer worse sim-to-real** because they exploit the simulator's contact model. Soft-
contact incentives improve real transfer. General survey support: Mock & Muknahallipatna, *Hierarchical
RL and Value Optimization for Challenging Quadruped Locomotion*,
https://arxiv.org/abs/2506.20036 (explicitly: contact-force penalty above threshold prevents hard
landings and improves sim-to-real); contact-power and foot-velocity-variation terms in
*Behavior-evolution-inspired walking-gait RL*, https://arxiv.org/abs/2409.16862 . Foot-jerk/impact
rationale also in Extreme Parkour (https://arxiv.org/abs/2309.14341).

For **our 4-foot pronk landing**, a contact-force/contact-power penalty raises the cost of all four
feet slamming simultaneously vs. a staggered touchdown — secondary pressure toward coordination.

### Side-effects / risks
- **Directly collides with our project finding:** measured impact/saturation are *below thresholds*, so
  the team flagged safety penalties **NO-GO** as a primary lever. An impact penalty here is therefore
  **not** justified on safety grounds — it can only be justified as a *transfer-quality / coordination*
  nudge, and must be kept tiny.
- Strong force penalties make the policy timid (avoids ground contact → more airtime), which would
  *worsen* our flight-phase problem. **Counterproductive if over-weighted.**
- (b) headroom: impact spikes are large and bursty; an uncapped squared-force penalty is the single
  most likely term to punch `total_reward` below the clip floor. If used, prefer the **jerk / force-
  rate** form (already in baseline, bounded) over raw squared-GRF.

### Constraint check
- (a) ✅ force read in reward only (same as baseline foot-jerk). **Never add contact/force to obs.**
- (b) ❌-risk for raw squared-GRF; ✅ for the bounded jerk form already present.
- (c) ✅ per-step.
- (d) ✅ soft-landing *improves* deploy; but see "timid policy" risk.

**Verdict:** lowest-priority for *our* goal — impact is already below safety thresholds, and the goal
is quality/coordination, not safety. Keep the existing bounded jerk term; do **not** add a large new
GRF penalty.

---

## 3. Fore-hind / limb-coordination & flight-phase suppression — the direct lever

This axis is the one that *names our pathology*. Three sub-families:

### 3a. Symmetry-based coordination reward (most directly on-target)
Su, Huang et al., *Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy* —
https://arxiv.org/abs/2403.10723 (also v4 html). They embed three symmetries directly in the reward:
- **temporal symmetry** → encourages *periodicity* of motion (a stride looks the same each cycle —
  the opposite of an erratic split-jump);
- **time-reversal symmetry** → "ensures the **front and back legs move in a similar fashion**" (this is
  literally fore-hind coordination);
- **morphological symmetry** → legs sharing the same phase move *simultaneously*.

Result: one policy reproduced trot/bound/half-bound/gallop on **Unitree Go2** with "coordinated
footfall patterns and improved energetic efficiency." This is the strongest literature match to
"reward coordinated fore-hind timing." For a *jump*, the time-reversal/morphological pair is exactly
what says "front pair and rear pair should act as coordinated units, not independent takeoffs."

### 3b. Periodic / contact-schedule reward (gait regularity)
Reward feet for matching a **periodic stance/swing schedule** (duty factor + phase offsets). Variants:
- diagonal-pair contact/air-duration matching (trot prior),
- von-Mises / smooth periodic contact-indicator rewards (Margolis & Agrawal, *Walk These Ways*,
  https://arxiv.org/abs/2212.03238 — periodic gait-timing reward).
  ⚠️ **MEMORY CONSTRAINT:** Walk-These-Ways' *commanded-gait-as-policy-input* (input expansion) is a
  **dealbreaker** for us (memory: input팽창 거부). The **reward-shaping concept** (rewarding a periodic
  contact schedule) is still citable and usable *as long as no gait command is added to obs*. Use the
  schedule as a fixed internal target inside the reward, not a policy input.

### 3c. Simultaneous-flight / minimum-stance penalty (kills pronk *and* split-jump airtime)
- **Simultaneous-flight penalty:** penalize timesteps where **all (or > N) feet are airborne**
  simultaneously. This is the most literal counter to our "stair 27.7 % all-airborne" pronk and to the
  flight phase of split-jumps. Bipedal precedent: a "non-touching-ground penalty" that fires when both
  legs are in flight simultaneously (survey: *Learning-based legged locomotion: state of the art*,
  https://arxiv.org/abs/2406.01152). Quadruped generalization: penalize `num_feet_airborne == 4` (or
  reward `num_feet_in_contact ≥ 1`).
- **Minimum-stance / contact-duty reward:** reward maintaining at least one (or one fore + one hind)
  support foot — suppresses long flight phases.
- **feet-air-time** is the *opposite-signed* knob: it *encourages* lift (avoids foot-dragging) and is
  already in our baseline. For our problem we likely want to **cap/re-tune air-time downward** or pair
  it with a simultaneous-flight penalty so it stops rewarding the pronk's huge airtime.

### Effect
Directly shapes *which feet are on the ground when*. (3a/3c) attack the split-jump at its root: a
split-jump is, by definition, an *aperiodic, fore-hind-decoupled* contact sequence with an extra
flight phase. Rewarding periodicity + fore-hind symmetry + penalizing all-airborne makes the
coordinated single-jump (fore and hind push off/land together) the reward optimum.

### Side-effects / risks
- A simultaneous-flight penalty can forbid *legitimate* flight needed to clear gaps/large obstacles —
  must allow a brief grace window or threshold so real parkour jumps aren't killed. Tune the airborne-
  fraction threshold, don't hard-ban flight.
- Symmetry rewards can over-regularize and suppress the asymmetric gaits sometimes needed on rough
  terrain (Fu et al. note rough terrain wants *unstructured* gaits). Weight modestly.
- Periodic-schedule rewards bias toward one nominal gait (e.g., trot) — may conflict with terrain-
  appropriate gait switching. Use loose phase tolerance.

### Constraint check
- (a) ✅ all use contact *state* inside reward, not obs. **Critical:** do NOT expose foot-contact
  booleans or a gait-phase command to the policy — keep the schedule/contact read inside
  `_get_rewards()`. (3b note: no gait command in obs.)
- (b) ✅-friendly: contact-state penalties are *bounded* (per-foot 0/1 indicators × small weight),
  unlike force magnitudes — easy to keep inside the clip(min=0) headroom. This is a structural
  advantage of 3a/3c over §1/§2.
- (c) ✅ per-step contact accounting; weight episode-normalized.
- (d) ✅ Go2-validated (3a is literally on Go2). Proprioceptive-only compatible.

---

## 4. Sim-to-real transferability of gaits

### What transfers poorly (directly relevant to us)
- **Hard impacts / contact-model exploitation:** policies that slam feet exploit the simulator's
  contact model and transfer worse; smooth contact transitions transfer better
  (https://arxiv.org/abs/2506.20036, https://arxiv.org/abs/2309.14341). → our 4-foot slam is a
  transfer risk *independent* of whether sim impact is below a damage threshold.
- **Long flight phases / simultaneous 4-foot landing:** high airborne fraction means the policy relies
  on precise ballistic timing and a single hard synchronized touchdown — fragile to real actuator
  latency, mass/inertia error, and ground friction/restitution mismatch. Cross-simulator studies show
  contact modeling is the dominant sensitivity (Isaac Gym oscillatory artifacts vs MuJoCo smoother but
  foot-drift): *Sim-to-Real via Terrain Transformer*, https://arxiv.org/abs/2212.07740. A coordinated,
  more-grounded gait has more support phases to correct error → better transfer.
- **Reward overfitting on complex terrain** can yield non-transferring policies (general sim-to-real
  literature).

### What helps transfer
- **Proprioceptive-only policies** (no contact/vision in obs) are a known robust-deploy recipe
  (proprioception-only agile locomotion, gated specialist experts) — exactly our constraint (a). The
  *gait quality* must therefore come from **reward shaping + domain randomization**, not from feeding
  contact to the policy.
- **Domain randomization ↔ gait:** randomizing mass, friction, motor strength, latency pushes the
  policy away from brittle timing-exploit gaits toward robust, more-grounded ones — DR and the
  coordination rewards in §3 are complementary (DR makes the long-flight exploit unreliable; §3 makes
  it low-reward).
- **Energy/CoT shaping** (§1) tends to produce animal-like gaits that are inherently more transferable
  (Fu et al.).

### Constraint check
- (a) ✅ proprioceptive-only is the *recommendation*, fully aligned. The whole sim-to-real story
  reinforces "keep contact out of obs."
- (d) ✅ this section *is* the deploy argument.

---

## 5. Constraint-compatibility summary table

| Signal | (a) No contact in obs | (b) clip(min=0) headroom | (c) step_dt weight | (d) Go2 deploy | Net fit for split-jump |
|---|---|---|---|---|---|
| §1 Energy/work penalty (re-tune existing) | ✅ τ,q̇ only | ⚠️ magnitude, keep modest | ✅ | ✅ | **High** (double-takeoff costs 2×) |
| §1 CoT as dense reward | ✅ | ❌ unstable (÷v) | n/a | ✅ | Low (use as metric only) |
| §2 raw squared-GRF impact penalty | ✅ reward-only | ❌ bursty, likely clips | ✅ | ✅ but timid-risk | Low (impact already sub-threshold) |
| §2 foot-jerk/force-rate (baseline) | ✅ | ✅ bounded | ✅ | ✅ | Low–Med (keep as-is) |
| §3a symmetry (temporal+time-reversal) | ✅ contact-state in reward | ✅ bounded | ✅ | ✅ (Go2-proven) | **Highest** |
| §3c simultaneous-flight penalty | ✅ contact-state in reward | ✅ bounded indicator | ✅ | ✅ | **Highest** |
| §3b periodic schedule (reward-only) | ✅ *iff no gait cmd in obs* | ✅ | ✅ | ✅ | High (but gait-cmd-in-obs forbidden) |

---

## 6. Top candidates to suppress split-jump (fore-hind coordination absent)

**#1 — Simultaneous-flight / minimum-stance penalty (§3c).** Most *direct* and most *constraint-safe*.
Penalize all-four-feet-airborne beyond a small grace fraction, and/or reward "≥1 fore-support AND ≥1
hind-support." A split-jump *requires* a flight phase between the front-land and rear-takeoff; bounding-
on-flat and 4-foot pronk both have all-airborne windows. A bounded contact-state indicator penalty sits
comfortably under the clip(min=0) floor (constraint b), needs no obs change (a), and is Go2-deployable
(d). **Tune the airborne threshold so legitimate gap-clearing jumps survive** — do not hard-ban flight.

**#2 — Fore-hind symmetry / periodicity reward (§3a).** The literature's most on-point match
(2403.10723, on Go2): time-reversal symmetry explicitly rewards "front and back legs move in a similar
fashion," morphological symmetry rewards same-phase legs moving together, temporal symmetry rewards
periodicity. This converts "one coordinated jump" into the reward optimum and "split-jump" into a
penalized aperiodic, fore-hind-decoupled sequence. Bounded → clip-safe; contact-state in reward only →
obs-safe. Pairs naturally with our existing Go2-Parkour-Symmetry L/R mirror data-aug.

**#3 — Re-tune the existing energy/work penalty upward (§1).** Lowest-risk *additive* change (the term
already exists — no new term, no new buffer). A split-jump performs two positive-work takeoffs per
stride vs. one for a coordinated jump, so a modestly stronger work penalty makes the coordinated jump
cheaper. Fu et al. shows energy-min alone yields coordinated gaits on benign terrain; on our obstacle
terrain it is *necessary but not sufficient*, so use it as a **supporting** term under #1–#2, watching
the clip(min=0) headroom (don't let obstacle push-off power spikes drive the sum negative).

> **Recommended stack:** #1 (simultaneous-flight penalty, small bounded weight) **+** #2 (fore-hind
> symmetry/periodicity reward) as the primary coordination levers, with #3 (energy re-tune) as the
> efficiency support. All three are additive, obs-free, clip-safe, and Go2-deployable. **Avoid** a new
> large GRF/impact penalty (§2) — impact is already sub-threshold and a big force penalty both risks
> the clip floor and can make the policy *more* airborne.

---

## 7. Gaps / not-found (no fabrication)

- **No paper found that rewards "one coordinated quadruped jump vs. a split/staggered jump" by name.**
  The split-jump vs. single-jump distinction is *our* framing; the literature supplies the building
  blocks (simultaneous-flight penalty, fore-hind symmetry, periodicity, energy) but not a turnkey
  "split-jump penalty." Composition required.
- **Exact energy-reward coefficient / formula in Fu et al.** not extracted from the project page; the
  page gives emergent-gait speeds but not the raw reward equation. The general `-|τq̇|` / `max(0,τq̇)`
  forms above are from corroborating sources, not verified to be Fu et al.'s exact weights.
- **Exact Extreme-Parkour reward weights** not pulled term-by-term here; the *list* of terms is
  confirmed from the paper, the numeric scales are not quoted (avoid mis-citing — read the repo cfg if
  exact values are needed).
- **clip(min=0) interaction with each specific penalty magnitude** is reasoned from the constraint, not
  measured. The §6 stack needs an empirical headroom check (e.g., log per-step penalty sums vs. the
  positive reward floor) before trusting the magnitudes — flagged for task #4 / training.
- **Constraint-vs-penalty alternative:** Kim, Oh et al., *Not Only Rewards But Also Constraints:
  Applications on Legged Robot Locomotion* (https://arxiv.org/abs/2308.12517) formulates these as
  *constraints* rather than reward penalties, which structurally sidesteps the clip(min=0) headroom
  problem. Noted as a heavier architectural option (changes the optimizer/algorithm, not just reward),
  out of scope for an additive reward change but relevant if §1/§2 magnitudes prove un-tunable.

---

## Sources

- Fu, Kumar, Malik, Pathak — *Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots*, CoRL 2021 — https://arxiv.org/abs/2111.01674 , https://energy-locomotion.github.io/
- Cheng, Shi, Agarwal, Pathak — *Extreme Parkour with Legged Robots*, CoRL 2023 — https://arxiv.org/abs/2309.14341 , https://extreme-parkour.github.io/
- Su, Huang, et al. — *Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy* (Go2) — https://arxiv.org/abs/2403.10723
- Margolis, Agrawal — *Walk These Ways* (periodic gait-timing reward; input-expansion form forbidden for us) — https://arxiv.org/abs/2212.03238
- Mock, Muknahallipatna — *Hierarchical RL and Value Optimization for Challenging Quadruped Locomotion* (contact-force penalty, soft landing, sim-to-real) — https://arxiv.org/abs/2506.20036
- *Behavior-evolution-inspired walking-gait RL for quadruped robots* (contact-power, foot-velocity-variation rewards) — https://arxiv.org/abs/2409.16862
- *Sim-to-Real Transfer for Quadrupedal Locomotion via Terrain Transformer* (contact-model sensitivity) — https://arxiv.org/abs/2212.07740
- *Learning-based legged locomotion: state of the art and future perspectives* (simultaneous-flight penalty, reward-design survey) — https://arxiv.org/abs/2406.01152
- Kim, Oh, et al. — *Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion* — https://arxiv.org/abs/2308.12517
