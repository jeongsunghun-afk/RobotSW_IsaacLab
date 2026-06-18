# Pronk-cost raw measurement (pronk_v1)

- checkpoint: `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_12-13-02_positive_work/model_49999.pt`
- task: `Go2-Parkour-Symmetry`
- run time: 2026-06-15 10:10:05
- num_steps=3000  num_envs=256  seed=42  dt=0.02
- force_thresh=2.0 N  sat_frac=0.95
- ACTUATOR (read at runtime from _robot.actuators['base_legs']):
  - effort_limit (per actuated joint) = [23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5]
  - velocity_limit = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
  - saturation_effort = 23.5
- nominal total mass = 15.019 kg  -> bodyweight = 147.34 N
- episodes: total=958 failure=7 goal=0
- valid (non-failure-episode) sample fraction = 0.993
- terrain_level first5%=2.48 last5%=3.80

```
================================================================
(1) ALL-AIRBORNE (pronk) FREQUENCY per terrain
================================================================
terrain              n_valid   airborne_frac   contact_mean_feet
flat                  156000          0.0128               2.770
hurdle                153000          0.0786               2.439
step                  152626          0.0990               2.449
gap                   153000          0.0589               2.465
stair                 147831          0.2373               2.100

  interpretation: airborne_frac = fraction of steps with ALL 4 feet off ground.
  contact_mean_feet ~4 = stance/walk; low values + high airborne_frac = pronk-like flight.

================================================================
(2a) ACTUATOR SATURATION vs DYNAMIC DCMotor envelope
================================================================
  saturation flag: |applied_torque| >= 0.95 * dynamic_bound(joint_vel)
  static effort_limit (per actuated joint): [23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5]
  velocity_limit: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]   saturation_effort: 23.5

terrain            all_steps_satfrac   landing_satfrac   mean_sat_joints
flat                          0.0085            0.0000             0.011
hurdle                        0.0584            0.0125             0.087
step                          0.0686            0.0395             0.119
gap                           0.0448            0.0011             0.065
stair                         0.0877            0.0329             0.242

================================================================
(2b) LANDING IMPACT: peak vertical contact force at airborne->contact transition
================================================================
  normalized by bodyweight = 147.3 N (nominal mass 15.02 kg)
  peak vertical force = max over contact-history dim of |Fz| summed across 4 feet at landing

terrain           n_landings     mean_BW    p50_BW    p95_BW    max_BW
flat                     938       2.452     2.452     3.947     8.614
hurdle                  4712       3.341     3.190     6.403     9.786
step                    5086       3.133     2.726     7.294    15.627
gap                     4641       2.743     2.683     4.753     9.805
stair                   7654       3.557     3.118     8.202    17.224

================================================================
(3) RELATIVE Cost-of-Transport (CoT) proxy per terrain vs FLAT
================================================================
  CoT = Σ(positive τ·q̇)·dt / (m·g·distance).  Distance = path length of base xy over valid steps.
  Reported as ratio to FLAT (defensible relative claim). NOT a claim about alternating gait.

terrain               energy_J      dist_m         CoT   CoT_vs_flat
flat                  174182.0      3080.3      0.3838
hurdle                183481.5      3195.3      0.3897
step                  202396.0      3141.7      0.4372
gap                   178517.9      3240.8      0.3739
stair                 266148.5      3134.3      0.5763

  flat CoT reference = 0.3838
    flat           CoT=0.3838  (1.00x flat)
    hurdle         CoT=0.3897  (1.02x flat)
    step           CoT=0.4372  (1.14x flat)
    gap            CoT=0.3739  (0.97x flat)
    stair          CoT=0.5763  (1.50x flat)

================================================================
(4) SPLIT-JUMP / FRONT-REAR COORDINATION (per terrain)
================================================================
  flight segment = maximal run of all-airborne steps within a valid episode.
  split-jump (re-takeoff) = airborne -> PARTIAL contact (front-only/rear-only) -> airborne
    with NO full(>=3-foot) settle in between. Signals front/rear NOT used as one coordinated leap.

terrain         flights  flt_steps  mean_len  retakeoff  retk/flight  frontfirst  rearfirst
flat                944       1996      2.11          1        0.001           7         11
hurdle             4722      12027      2.55        279        0.059          34         48
step               5099      15117      2.96        278        0.055         180         49
gap                4651       9019      1.94        106        0.023           9        183
stair              7714      35084      4.55        602        0.078          67         45

  retk/flight high => obstacle crossings frequently split into multiple hops (coordination gap).
  frontfirst >> rearfirst => front pair consistently lands before rear (expected for forward
    motion); the concern is re-takeoff BETWEEN them (retakeoff column), not front-first itself.

================================================================
GO / NO-GO VERDICT
================================================================
  Pre-registered thresholds: X_sat=0.20 (landing saturation frac), Y_impact=4.0 bodyweights (peak landing force).
  Rule: if stair/step has (landing_satfrac > X) OR (median landing impact > Y),
        AND is notably higher than gap (legitimate-jump reference) => B+C penalty JUSTIFIED.
        Otherwise => HOLD (do not add penalty on this evidence).

  [stair] landing_satfrac=0.0329  median_impact=3.118 BW   (gap ref: sat=0.0011 imp=2.683 BW)
  [step] landing_satfrac=0.0395  median_impact=2.726 BW   (gap ref: sat=0.0011 imp=2.683 BW)

  NOTE (gap is NOT a clean control): user observed gap crossings are SPLIT jumps
  (front lands, re-takeoff before rear lands). Compare split-jump rate across terrains:
    flat       retakeoff/flight = 0.001
    gap        retakeoff/flight = 0.023
    stair      retakeoff/flight = 0.078
    step       retakeoff/flight = 0.055
    hurdle     retakeoff/flight = 0.059
  If gap/stair/step all show high retakeoff/flight, split-jumping is a pervasive coordination
  inefficiency (front/rear not used as one leap) — a SEPARATE signal from saturation/impact.

```
