# Pronk-cost raw measurement (pronk_v1)

- checkpoint: `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-12_09-53-06_positive_work_0.001/model_49999.pt`
- task: `Go2-Parkour-Symmetry`
- run time: 2026-06-15 10:08:02
- num_steps=3000  num_envs=256  seed=42  dt=0.02
- force_thresh=2.0 N  sat_frac=0.95
- ACTUATOR (read at runtime from _robot.actuators['base_legs']):
  - effort_limit (per actuated joint) = [23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5]
  - velocity_limit = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
  - saturation_effort = 23.5
- nominal total mass = 15.019 kg  -> bodyweight = 147.34 N
- episodes: total=965 failure=30 goal=0
- valid (non-failure-episode) sample fraction = 0.972
- terrain_level first5%=2.48 last5%=3.84

```
================================================================
(1) ALL-AIRBORNE (pronk) FREQUENCY per terrain
================================================================
terrain              n_valid   airborne_frac   contact_mean_feet
flat                  156000          0.0077               2.238
hurdle                147985          0.0672               2.204
step                  150142          0.0707               2.170
gap                   153000          0.0494               2.202
stair                 139411          0.1316               1.956

  interpretation: airborne_frac = fraction of steps with ALL 4 feet off ground.
  contact_mean_feet ~4 = stance/walk; low values + high airborne_frac = pronk-like flight.

================================================================
(2a) ACTUATOR SATURATION vs DYNAMIC DCMotor envelope
================================================================
  saturation flag: |applied_torque| >= 0.95 * dynamic_bound(joint_vel)
  static effort_limit (per actuated joint): [23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5]
  velocity_limit: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]   saturation_effort: 23.5

terrain            all_steps_satfrac   landing_satfrac   mean_sat_joints
flat                          0.9602            0.3438             1.918
hurdle                        0.7099            0.5022             1.467
step                          0.5315            0.7054             1.227
gap                           0.2294            0.1860             0.446
stair                         0.5775            0.6246             1.433

================================================================
(2b) LANDING IMPACT: peak vertical contact force at airborne->contact transition
================================================================
  normalized by bodyweight = 147.3 N (nominal mass 15.02 kg)
  peak vertical force = max over contact-history dim of |Fz| summed across 4 feet at landing

terrain           n_landings     mean_BW    p50_BW    p95_BW    max_BW
flat                     256       2.319     2.050     3.690     6.480
hurdle                  2489       2.297     2.081     5.244    10.674
step                    3714       2.185     1.880     5.310    11.804
gap                     2344       2.483     2.281     5.121     7.407
stair                   4862       1.854     1.537     5.024     9.248

================================================================
(3) RELATIVE Cost-of-Transport (CoT) proxy per terrain vs FLAT
================================================================
  CoT = Σ(positive τ·q̇)·dt / (m·g·distance).  Distance = path length of base xy over valid steps.
  Reported as ratio to FLAT (defensible relative claim). NOT a claim about alternating gait.

terrain               energy_J      dist_m         CoT   CoT_vs_flat
flat                  218378.1      3050.3      0.4859
hurdle                371648.7      2899.9      0.8698
step                  490477.0      3070.6      1.0841
gap                   289029.1      3372.8      0.5816
stair                 596382.4      3160.7      1.2807

  flat CoT reference = 0.4859
    flat           CoT=0.4859  (1.00x flat)
    hurdle         CoT=0.8698  (1.79x flat)
    step           CoT=1.0841  (2.23x flat)
    gap            CoT=0.5816  (1.20x flat)
    stair          CoT=1.2807  (2.64x flat)

================================================================
(4) SPLIT-JUMP / FRONT-REAR COORDINATION (per terrain)
================================================================
  flight segment = maximal run of all-airborne steps within a valid episode.
  split-jump (re-takeoff) = airborne -> PARTIAL contact (front-only/rear-only) -> airborne
    with NO full(>=3-foot) settle in between. Signals front/rear NOT used as one coordinated leap.

terrain         flights  flt_steps  mean_len  retakeoff  retk/flight  frontfirst  rearfirst
flat                257       1207      4.70          0        0.000          16          1
hurdle             2495       9941      3.98        531        0.213         269        171
step               3722      10614      2.85       1098        0.295         333         97
gap                2351       7565      3.22        209        0.089         373         11
stair              4896      18343      3.75       1418        0.290         472        244

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

  [stair] landing_satfrac=0.6246  median_impact=1.537 BW   (gap ref: sat=0.1860 imp=2.281 BW)
  [step] landing_satfrac=0.7054  median_impact=1.880 BW   (gap ref: sat=0.1860 imp=2.281 BW)

  NOTE (gap is NOT a clean control): user observed gap crossings are SPLIT jumps
  (front lands, re-takeoff before rear lands). Compare split-jump rate across terrains:
    flat       retakeoff/flight = 0.000
    gap        retakeoff/flight = 0.089
    stair      retakeoff/flight = 0.290
    step       retakeoff/flight = 0.295
    hurdle     retakeoff/flight = 0.213
  If gap/stair/step all show high retakeoff/flight, split-jumping is a pervasive coordination
  inefficiency (front/rear not used as one leap) — a SEPARATE signal from saturation/impact.

```
