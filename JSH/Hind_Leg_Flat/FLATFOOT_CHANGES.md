# Flat-foot (true 2-point contact) fix — 2026-07-24

Goal: make the flat foot rest FLAT (heel+toe spheres both on the floor) at standing,
matching the validated MuJoCo model `simulation/biped/biped_flatfoot.mjcf`. No retraining.

## 1. Collision-sphere geometry (edited: `configuration/hind_leg_physics.usd`)
Prior agent had put BOTH heel+toe on `foot_contact_link` (robot balanced on toe only). Fixed to
mirror MuJoCo body-origin placement:

| sphere | body                 | translate (local, m)    | radius |
|--------|----------------------|-------------------------|--------|
| heel   | `*_foot_link`        | (0, 0, 0)  = ankle      | 0.036  |
| toe    | `*_foot_contact_link`| (0.0168, 0, -0.0461)    | 0.036  |

Rationale: MuJoCo has heel `*_sphere2` at `foot_link` origin (the ankle) and toe `*_sphere` at
`foot_contact_link` origin, both r=0.036, ~0.16 m apart. In this USD the physical toe tip is the
existing mesh sphere at (0.0168,0,-0.0461) from `foot_contact_link` (0.049 m beyond its origin),
giving ankle->toe ≈ 0.159 m — matches MuJoCo's ~0.16 m. Prior wrong `heel_collision` on
`foot_contact_link` removed for both legs. Backup: `configuration/hind_leg_physics.usd.bak_preflat` (on server).

## 2. Standing pose (edited: `source/isaaclab_assets/isaaclab_assets/robots/rga.py`, `FLAT_HIND_LEG_CFG.init_state`)
Backup on server: `rga.py.bak_flatfoot`.

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.42),
    joint_pos={
        "HL_hip_joint": 0.0,  "HL_thigh_joint": 0.25, "HL_calf_joint": -0.40, "HL_foot_joint": 1.4386,
        "HR_hip_joint": 0.0,  "HR_thigh_joint": -0.25,"HR_calf_joint": -0.40, "HR_foot_joint": -1.4387,
    },
),
```

Found empirically in Isaac (FK sweep of ankle for coplanar heel/toe). Ankle-joint saturation
(+1.571) caps a flat-foot base at ~0.42 m (shallower crouch can't flatten the sole). HR = HL with
thigh & foot signs flipped (mirrored-URDF joint frames: HL/HR thigh & foot joints have opposite
`localRot0`; calf same) — calf keeps the same sign.

## 3. Verification (Isaac, headless), base pinned level
- (A) at rest height 0.4178 m: all four sphere bottoms z = 0.0000, heel-toe center gap 0.01 mm.
- (B) 5 mm press: contact Fz — HL heel 182.7 N / toe 35.5 N, HR heel 182.2 N / toe 36.4 N
  (purely vertical, no shear). Both spheres of each foot load => TRUE 2-point flat contact.

Recommended standing base height = 0.4178 m (spawn uses 0.42). Note: a free settle under the
default implicit PD folds the legs (control/balance, not geometry) — the pinned-base contact test
isolates and confirms the geometry.
