# pace_sim2real — vendoring note

Vendored from **https://github.com/leggedrobotics/pace-sim2real** (ETH Zurich RSL, Apache-2.0).

| | |
|---|---|
| Upstream commit | `9de8094bf50942516c89b66f48555058a7d3b632` |
| Upstream date | 2026-06-18 |
| Vendored on | 2026-07-13 |
| Upstream target | Isaac Sim 5.0 / IsaacLab `main` |
| This tree | Isaac Sim 6.0.1 / IsaacLab 6.1.11 |

Paper: F. Bjelonic, F. Tischhauser, M. Hutter, *Towards Bridging the Gap: Systematic Sim-to-Real
Transfer for Diverse Legged Robots*, arXiv:2509.06342 (IJRR).

## Why vendored rather than installed from upstream

The Go2 sim2real work couples PACE to the `r2s_go2` rig (real chirp capture, sim↔real overlay
validation). Keeping PACE in this tree lets the Go2 asset config, joint order and actuator object
stay a **single source of truth** shared by the live rig and the identification run, and lets the
IsaacLab 6.0 port live next to the code it patches.

## IsaacLab 6.0 port — what was changed and why

| # | Change | Reason |
|---|---|---|
| B1 | `pip install cmaes` (0.13.0) | Not present in the `isaac-6.0` conda env. |
| B2 | `cma_es.py`, `data_collection.py`: three friction writes → one `write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=, joint_dynamic_friction_coeff=, joint_viscous_friction_coeff=, ...)` | `write_joint_viscous_friction_coefficient_to_sim` and `write_joint_dynamic_friction_coefficient_to_sim` **do not exist** in IsaacLab 6.0. The single call also removes the upstream "write dynamic=0 first" ordering workaround. |
| B3 | `write_joint_{armature,position,velocity}_to_sim` → `*_to_sim_index` (keyword args) | The bare names are deprecated aliases in 6.0. |
| B4 | `articulation.data.default_joint_*` → `.torch[...]` | These are `ProxyArray` (warp-backed) in 6.0; implicit tensor use emits a `DeprecationWarning`. |
| B5 | `from isaaclab.utils import DelayBuffer` → `from isaaclab.utils.buffers import DelayBuffer` | `isaaclab.utils` is a lazy-export package in 6.0 and does not re-export `DelayBuffer`. |
| B6 | `utils/backend.py::require_physx_backend()`, called from both scripts | The Newton backend supports **static joint friction only**. A fit on Newton would silently optimize 24 of the 49 parameters into a no-op. |
| B7 | `from isaaclab.utils import configclass` → `from isaaclab.utils.configclass import configclass` | The former binds the submodule in this tree and raises `TypeError: 'module' object is not callable` (same landmine documented in `direct/r2s_go2/CLAUDE.md`). |
| B8 | Dropped `articulation.data.default_joint_dynamic_friction_coeff` writes | The property **does not exist** in 6.0. The dynamic coefficient is written to sim explicitly every generation, so no default cache is needed. |
| B9 | `PaceDCMotorCfg.max_delay: torch.int \| None` → `int` | `torch.int` is a dtype, not a type; using it as an annotation is invalid for `configclass`. |
| B10 | `data_collection.py`: matplotlib falls back to the `Agg` backend and writes PNGs when headless | `plt.show()` blocks/fails without a display. |
| B11 | `joint_ids` / `env_ids` index tensors built with `dtype=torch.int32` | The 6.0 joint-property writers dispatch warp kernels typed `int32`. A default `int64` tensor raises `RuntimeError: Could not convert array interface with typestr='<i8' to Warp array with dtype=int32`. |

## Data and log locations

`project_root()` walks up to the nearest `.git`, which resolves to the IsaacLab-6.0 repo root:

- data: `<repo>/data/<robot_name>/chirp_data.pt` (gitignored)
- logs: `<repo>/logs/pace/<robot_name>/<timestamp>/` (gitignored)

Set `PACE_ROOT` to override.

## Upgrading

Diff against upstream `9de8094` before pulling new commits; the port touches
`optim/cma_es.py`, `utils/pace_actuator*.py`, `utils/__init__.py`, `tasks/.../pace_sim2real_env_cfg.py`,
`tasks/.../anymal_pace_env_cfg.py` and both scripts under `scripts/pace/`.
