# Pronk-cost raw measurement (pronk_v1)

- checkpoint: `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-09_17-53-09/model_16700.pt`
- task: `Go2-Parkour-Symmetry`
- run time: 2026-06-10 16:46:40
- num_steps=3000  num_envs=256  seed=42  dt=0.02
- force_thresh=2.0 N  sat_frac=0.95
- ACTUATOR (read at runtime from _robot.actuators['base_legs']):
  - effort_limit (per actuated joint) = [23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5]
  - velocity_limit = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
  - saturation_effort = 23.5
- nominal total mass = 15.019 kg  -> bodyweight = 147.34 N
- episodes: total=1000 failure=0 goal=0
- valid (non-failure-episode) sample fraction = 1.000
- terrain_level first5%=2.48 last5%=3.83

```
================================================================
(1) ALL-AIRBORNE (pronk) FREQUENCY per terrain
================================================================
terrain              n_valid   airborne_frac   contact_mean_feet
flat                  156000          0.0216               2.811
hurdle                153000          0.1255               2.450
step                  153000          0.1428               2.441
gap                   153000          0.1411               2.383
stair                 153000          0.2770               2.106

  interpretation: airborne_frac = fraction of steps with ALL 4 feet off ground.
  contact_mean_feet ~4 = stance/walk; low values + high airborne_frac = pronk-like flight.

================================================================
(2a) ACTUATOR SATURATION vs DYNAMIC DCMotor envelope
================================================================
  saturation flag: |applied_torque| >= 0.95 * dynamic_bound(joint_vel)
  static effort_limit (per actuated joint): [23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5, 23.5]
  velocity_limit: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]   saturation_effort: 23.5

terrain            all_steps_satfrac   landing_satfrac   mean_sat_joints
flat                          0.0074            0.0009             0.011
hurdle                        0.0438            0.0008             0.063
step                          0.0836            0.0183             0.136
gap                           0.0653            0.0250             0.105
stair                         0.0918            0.0584             0.202

================================================================
(2b) LANDING IMPACT: peak vertical contact force at airborne->contact transition
================================================================
  normalized by bodyweight = 147.3 N (nominal mass 15.02 kg)
  peak vertical force = max over contact-history dim of |Fz| summed across 4 feet at landing

terrain           n_landings     mean_BW    p50_BW    p95_BW    max_BW
flat                    2285       2.594     2.691     3.823     6.151
hurdle                  6167       3.213     3.085     5.883    10.382
step                    5969       3.427     3.020     7.977    17.438
gap                     7084       3.025     2.946     5.495    14.263
stair                   8607       3.517     3.124     7.821    21.508

================================================================
(3) RELATIVE Cost-of-Transport (CoT) proxy per terrain vs FLAT
================================================================
  CoT = Σ(positive τ·q̇)·dt / (m·g·distance).  Distance = path length of base xy over valid steps.
  Reported as ratio to FLAT (defensible relative claim). NOT a claim about alternating gait.

terrain               energy_J      dist_m         CoT   CoT_vs_flat
flat                  194671.7      3163.0      0.4177              
hurdle                219054.6      3338.6      0.4453              
step                  259518.6      3290.2      0.5354              
gap                   236183.8      3493.0      0.4589              
stair                 307585.8      3359.7      0.6214              

  flat CoT reference = 0.4177
    flat           CoT=0.4177  (1.00x flat)
    hurdle         CoT=0.4453  (1.07x flat)
    step           CoT=0.5354  (1.28x flat)
    gap            CoT=0.4589  (1.10x flat)
    stair          CoT=0.6214  (1.49x flat)

================================================================
(4) SPLIT-JUMP / FRONT-REAR COORDINATION (per terrain)
================================================================
  flight segment = maximal run of all-airborne steps within a valid episode.
  split-jump (re-takeoff) = airborne -> PARTIAL contact (front-only/rear-only) -> airborne
    with NO full(>=3-foot) settle in between. Signals front/rear NOT used as one coordinated leap.

terrain         flights  flt_steps  mean_len  retakeoff  retk/flight  frontfirst  rearfirst
flat               2293       3377      1.47          4        0.002           8          4
hurdle             6185      19195      3.10        109        0.018          49          2
step               5988      21844      3.65        214        0.036         499          7
gap                7102      21589      3.04        108        0.015         160         36
stair              8662      42385      4.89        518        0.060         145        119

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

  [stair] landing_satfrac=0.0584  median_impact=3.124 BW   (gap ref: sat=0.0250 imp=2.946 BW)
  [step] landing_satfrac=0.0183  median_impact=3.020 BW   (gap ref: sat=0.0250 imp=2.946 BW)

  NOTE (gap is NOT a clean control): user observed gap crossings are SPLIT jumps
  (front lands, re-takeoff before rear lands). Compare split-jump rate across terrains:
    flat       retakeoff/flight = 0.002
    gap        retakeoff/flight = 0.015
    stair      retakeoff/flight = 0.060
    step       retakeoff/flight = 0.036
    hurdle     retakeoff/flight = 0.018
  If gap/stair/step all show high retakeoff/flight, split-jumping is a pervasive coordination
  inefficiency (front/rear not used as one leap) — a SEPARATE signal from saturation/impact.

```
