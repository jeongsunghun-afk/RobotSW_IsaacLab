# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left/Right (L<->R) symmetry augmentation for the Go2 Parkour direct environment.

This module provides a single ``data_augmentation_func`` consumed by ``PPOParkour`` (rsl_rl)
to softly enforce policy L/R equivariance via mirror data-augmentation.

Unlike the ANYmal reference (``manager_based/locomotion/velocity/mdp/symmetry/anymal.py``):
- parkour applies **only** the single left/right symmetry (``num_aug = 2``). The task is
  forward-only (command = forward velocity, scan offset +0.375m forward), so front-back /
  diagonal symmetries do NOT hold and must not be used.
- parkour observations are a **TensorDict with 5 groups** (``policy / scan / priv_explicit /
  priv_latent / history``), not a single flat tensor. All five groups are mirrored.
- joint swap and foot swap are constructed **dynamically from runtime names** to avoid
  hard-coded index ordering ambiguity (leg-grouped vs joint-type-grouped) and to keep the
  contact-sensor foot order and articulation foot order independently correct.

Design reference: ``_workspace/plans/parkour_symmetry_impl_design.md`` (sections B/C).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

# specify the functions that are available for import
__all__ = ["compute_parkour_symmetric_states"]

# --- proprio (policy group) layout, 46-dim (parkour_env.py:905-917) ---
#   [0]    yaw_diff               (target - heading, wrapped)
#   [1]    next_yaw_diff
#   [2:5]  projected_gravity_b
#   [5]    commands[:, 0]         (forward-only)
#   [6:18]  joint_pos - default
#   [18:30] joint_vel * 0.05
#   [30:42] actions
#   [42:46] contact_filt - 0.5    (per-foot)
_PROPRIO_DIM = 46
_SCAN_DIM = 187
_SCAN_DIM_3D = 294  # clearance_3d: 21 az × 14 el = 294
_PRIV_LATENT_DIM = 33
# scan grid: ordering "xy", size (1.6, 1.0) -> (y=11, x=17). Lateral axis = dim 1 (size 11).
_SCAN_LAT = 11
_SCAN_LON = 17
# clearance_3d grid: az-outer/el-inner, 21 az × 14 el.
# Azimuth spans [-100, +100]° symmetrically about forward (idx 10 = 0°).
# L/R mirror = reverse az index: mirror_idx = (20 - az_idx) * 14 + el_idx.
# Values are distances (no sign flip needed — only permutation).
_CLEAR_AZ = 21
_CLEAR_EL = 14
# voxel flat layout: (nx=27, ny=21, nz=13), C-order → flat_index = ix*(ny*nz)+iy*nz+iz.
# Mirror = y-flip (iy → ny-1-iy). ny=21 odd → center slice (iy=10) self-maps.
_VOXEL_NX = 27
_VOXEL_NY = 21
_VOXEL_NZ = 13
_VOXEL_DIM = 7371  # 27 * 21 * 13


def _flip_side(prefix: str) -> str:
    """Flip the left/right token in a Go2 leg prefix ("FL"<->"FR", "RL"<->"RR")."""
    if prefix.endswith("L"):
        return prefix[:-1] + "R"
    if prefix.endswith("R"):
        return prefix[:-1] + "L"
    return prefix


def _mirror_name(name: str) -> str:
    """Return the L<->R mirrored body/joint name (e.g. "FL_hip_joint" -> "FR_hip_joint")."""
    prefix, _, rest = name.partition("_")
    return f"{_flip_side(prefix)}_{rest}" if rest else _flip_side(prefix)


def _build_perm(names: list[str], device: torch.device) -> torch.Tensor:
    """Build the L<->R swap permutation for an ordered list of names.

    For each entry ``names[i]``, find the index of its mirrored name. Asserts the result is a
    valid permutation (no duplicates / no missing entries) to hard-fail on any name-match drift.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    perm: list[int] = []
    for n in names:
        mirrored = _mirror_name(n)
        if mirrored not in name_to_idx:
            raise ValueError(f"[parkour symmetry] no L/R mirror match for '{n}' (expected '{mirrored}'). names={names}")
        perm.append(name_to_idx[mirrored])
    perm_t = torch.tensor(perm, dtype=torch.long, device=device)
    # validate permutation integrity (catches silent index-shift / duplicate matches)
    if torch.unique(perm_t).numel() != len(names):
        raise ValueError(f"[parkour symmetry] swap perm is not a valid permutation: {perm} (names={names})")
    return perm_t


def _build_cache(env) -> dict:
    """Build (and cache on the env) the dynamic swap permutations + hip mask.

    Cached on ``env.unwrapped`` so name-matching + validation run once. Does not modify
    ``parkour_env.py`` — only attaches a runtime attribute.
    """
    u = env.unwrapped
    cache = getattr(u, "_parkour_symmetry_cache", None)
    if cache is not None:
        return cache

    device = u.device
    joint_names = list(u._robot.data.joint_names)
    # joint swap permutation (12 DOF) + hip mask (abduction joints: L/R sign-reversed)
    joint_swap = _build_perm(joint_names, device)
    hip_idx = torch.tensor([i for i, n in enumerate(joint_names) if "hip" in n], dtype=torch.long, device=device)
    if hip_idx.numel() == 0:
        raise ValueError(f"[parkour symmetry] no hip joints found in {joint_names}")

    # contact-sensor foot order (drives proprio[42:46] contact_filt). _feet_ids order asserted
    # [FL, FR, RL, RR] in parkour_env.py:207-210.
    contact_foot_names = [u._contact_sensor.body_names[i] for i in u._feet_ids]
    contact_foot_swap = _build_perm(contact_foot_names, device)

    # articulation foot order (drives priv_latent foot_friction via _foot_shape_indices).
    _, friction_foot_names = u._robot.find_bodies(".*foot")
    friction_foot_swap = _build_perm(list(friction_foot_names), device)

    # Document the (expected) equality of the two foot orderings; warn rather than crash if they
    # ever diverge so each consumer still uses its own correct permutation.
    if not torch.equal(contact_foot_swap, friction_foot_swap):
        import warnings

        warnings.warn(
            "[parkour symmetry] contact-sensor foot order != articulation foot order; "
            f"contact={contact_foot_names} ({contact_foot_swap.tolist()}), "
            f"friction={list(friction_foot_names)} ({friction_foot_swap.tolist()}). "
            "Each consumer uses its own permutation (correct), but verify foot mapping.",
            stacklevel=2,
        )

    cache = {
        "joint_swap": joint_swap,
        "hip_idx": hip_idx,
        "contact_foot_swap": contact_foot_swap,
        "friction_foot_swap": friction_foot_swap,
    }
    u._parkour_symmetry_cache = cache
    return cache


def _swap_joints(j: torch.Tensor, joint_swap: torch.Tensor, hip_idx: torch.Tensor) -> torch.Tensor:
    """L<->R joint swap + hip (abduction) sign reversal. Operates on the last dim (12 DOF).

    Works for arbitrary leading dims (used for both [B, 12] and history [B, T, 12]).
    The hip index set is invariant under the L<->R swap (hip->hip), so negating at ``hip_idx``
    after the swap is equivalent to negating the post-swap hip columns.
    """
    out = j[..., joint_swap].clone()
    out[..., hip_idx] = -out[..., hip_idx]
    return out


def _swap_joints_no_flip(j: torch.Tensor, joint_swap: torch.Tensor) -> torch.Tensor:
    """L<->R joint swap WITHOUT sign reversal — for magnitude quantities (stiffness/damping)."""
    return j[..., joint_swap].clone()


def _mirror_proprio(
    p: torch.Tensor, joint_swap: torch.Tensor, hip_idx: torch.Tensor, contact_foot_swap: torch.Tensor
) -> torch.Tensor:
    """Mirror a proprio tensor (..., 46). Last dim layout per module docstring."""
    m = p.clone()
    sign_g = torch.tensor([1.0, -1.0, 1.0], device=p.device)
    m[..., 0] = -m[..., 0]  # yaw_diff [추정]
    m[..., 1] = -m[..., 1]  # next_yaw_diff [추정]
    m[..., 2:5] = m[..., 2:5] * sign_g  # projected_gravity_b: y reversed
    # m[..., 5] commands (forward-only): unchanged
    m[..., 6:18] = _swap_joints(p[..., 6:18], joint_swap, hip_idx)  # joint_pos - default
    m[..., 18:30] = _swap_joints(p[..., 18:30], joint_swap, hip_idx)  # joint_vel * 0.05
    m[..., 30:42] = _swap_joints(p[..., 30:42], joint_swap, hip_idx)  # actions
    m[..., 42:46] = p[..., 42:46][..., contact_foot_swap]  # contact_filt (foot swap, no sign)
    return m


def _build_clearance_mirror_perm(num_azimuth: int, num_elevation: int, device: torch.device) -> torch.Tensor:
    """Build the L/R az-reversal permutation for the clearance_3d scan layout.

    Layout: az-outer / el-inner → flat index = az_idx * num_elevation + el_idx.
    Mirror: reverse az index (left↔right), keep el unchanged.
    Values are distances (positive scalars) — no sign flip required.

    Returns:
        perm: (num_azimuth * num_elevation,) LongTensor such that perm[i] gives the
              source index for destination i in the mirrored output.
    """
    perm = torch.zeros(num_azimuth * num_elevation, dtype=torch.long, device=device)
    for az_idx in range(num_azimuth):
        mirror_az = (num_azimuth - 1 - az_idx)
        for el_idx in range(num_elevation):
            src = mirror_az * num_elevation + el_idx
            dst = az_idx * num_elevation + el_idx
            perm[dst] = src
    return perm


# Module-level cache for the clearance perm (built once on first use, device-agnostic key).
_clearance_perm_cache: dict[str, torch.Tensor] = {}

# Module-level cache for the voxel y-flip perm.
_voxel_perm_cache: dict[str, torch.Tensor] = {}


def _build_voxel_mirror_perm(nx: int, ny: int, nz: int, device: torch.device) -> torch.Tensor:
    """Build the L/R y-flip permutation index for the flat voxel occupancy layout.

    Flat layout (C-order, matches voxel_occupancy.py flat_idx and parkour_env.py .reshape):
        flat_index = ix*(ny*nz) + iy*nz + iz

    Mirror rule: iy → ny-1-iy (y-axis flip = left/right swap).
    Occupancy is a scalar categorical → permutation only, no sign flip required.
    ny=21 is odd → center slice (iy=10) self-maps (involution centre).
    Involution property: mirror(mirror(x)) == x by construction.

    Returns:
        perm: (nx*ny*nz,) LongTensor where ``output[dst] = input[perm[dst]]``
              gives the y-mirrored voxel grid.
    """
    assert nx * ny * nz == _VOXEL_DIM, (
        f"[parkour symmetry] voxel grid {nx}×{ny}×{nz} = {nx * ny * nz} ≠ {_VOXEL_DIM}"
    )
    # Vectorised construction — no Python triple-loop needed.
    gx = torch.arange(nx, device=device)
    gy = torch.arange(ny, device=device)
    gz = torch.arange(nz, device=device)
    gx3, gy3, gz3 = torch.meshgrid(gx, gy, gz, indexing="ij")  # (nx, ny, nz) each
    mirror_gy = ny - 1 - gy3                                    # y-flip
    # For each destination (ix, iy, iz), the source is (ix, ny-1-iy, iz).
    src = gx3 * (ny * nz) + mirror_gy * nz + gz3               # (nx, ny, nz) source flat indices
    # C-order flatten: perm[dst_flat] = src[ix, iy, iz] where dst_flat = ix*(ny*nz)+iy*nz+iz
    return src.reshape(-1)  # (nx*ny*nz,)


def _mirror_scan(scan: torch.Tensor) -> torch.Tensor:
    """Mirror scan observations along the left/right axis.

    Dispatches based on the last dimension:
    - 187: height-scan 2D grid (11 lat × 17 lon), flip lateral dim.
    - 294: clearance_3d (21 az × 14 el), az-reversal permutation.

    Args:
        scan: (..., D) tensor where D is 187 or 294.

    Returns:
        Mirrored scan tensor with the same shape.
    """
    dim = scan.shape[-1]
    if dim == _SCAN_DIM:
        # 2D height-scan: flip lateral (y) axis.
        return scan.view(-1, _SCAN_LAT, _SCAN_LON).flip(dims=[1]).reshape(-1, _SCAN_DIM)
    elif dim == _SCAN_DIM_3D:
        # Clearance 3D: az-reversal permutation (L/R flip, values are distances — no sign change).
        device_key = str(scan.device)
        if device_key not in _clearance_perm_cache:
            _clearance_perm_cache[device_key] = _build_clearance_mirror_perm(
                _CLEAR_AZ, _CLEAR_EL, scan.device
            )
        perm = _clearance_perm_cache[device_key]
        return scan[..., perm]
    else:
        raise ValueError(
            f"[parkour symmetry] _mirror_scan: unsupported scan dim {dim}. "
            f"Expected {_SCAN_DIM} (height_scan) or {_SCAN_DIM_3D} (clearance_3d)."
        )


def _mirror_voxel(voxel: torch.Tensor) -> torch.Tensor:
    """Mirror flat voxel occupancy obs (..., 7371) along the y-axis (L/R flip).

    Applies the cached ``_build_voxel_mirror_perm`` permutation. Occupancy values
    are scalar categoricals — no sign flip, permutation only.

    Args:
        voxel: (..., 7371) float tensor (already flattened from (N,27,21,13) by parkour_env).

    Returns:
        Mirrored tensor with the same shape.
    """
    device_key = str(voxel.device)
    if device_key not in _voxel_perm_cache:
        _voxel_perm_cache[device_key] = _build_voxel_mirror_perm(
            _VOXEL_NX, _VOXEL_NY, _VOXEL_NZ, voxel.device
        )
    perm = _voxel_perm_cache[device_key]
    return voxel[..., perm]


def _mirror_priv_explicit(pe: torch.Tensor) -> torch.Tensor:
    """Mirror priv_explicit (B, 6): root_lin_vel_b[0:3] *[1,-1,1], root_ang_vel_b[3:6] *[-1,1,-1]."""
    m = pe.clone()
    m[..., 0:3] = m[..., 0:3] * torch.tensor([1.0, -1.0, 1.0], device=pe.device)
    m[..., 3:6] = m[..., 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=pe.device)
    return m


def _mirror_priv_latent(pl: torch.Tensor, joint_swap: torch.Tensor, friction_foot_swap: torch.Tensor) -> torch.Tensor:
    """Mirror priv_latent (B, 33).

    Layout: [0] base_friction | [1:5] foot_friction(4) | [5] base_mass | [6:9] base_com |
            [9:21] joint_stiffness_ratio(12) | [21:33] joint_damping_ratio(12).
    NOTE: stiffness/damping are MAGNITUDES -> joint swap ONLY, hip sign-flip is forbidden.
    """
    m = pl.clone()
    # [0] base_friction: scalar, unchanged
    m[..., 1:5] = pl[..., 1:5][..., friction_foot_swap]  # foot_friction: foot swap, no sign
    # [5] base_mass: unchanged
    m[..., 6:9] = m[..., 6:9] * torch.tensor([1.0, -1.0, 1.0], device=pl.device)  # base_com: y reversed [추정]
    m[..., 9:21] = _swap_joints_no_flip(pl[..., 9:21], joint_swap)  # stiffness ratio (no hip flip ★)
    m[..., 21:33] = _swap_joints_no_flip(pl[..., 21:33], joint_swap)  # damping ratio (no hip flip ★)
    return m


@torch.no_grad()
def compute_parkour_symmetric_states(*, env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Augment parkour observations/actions with the left/right mirror (num_aug = 2).

    Returns a batch of ``[original B; mirrored B]`` for each of ``obs`` and ``actions``. The first
    ``B`` rows are always the originals (rsl_rl ``PPOParkour`` relies on this ordering).

    Args:
        env: the (wrapped) VecEnv instance; ``env.unwrapped`` exposes the parkour env.
        obs: TensorDict with groups ``policy(46) / scan(187) / priv_explicit(6) /
            priv_latent(33) / history(N, 10, 46)``. ``None`` -> returns ``None`` for obs.
        actions: action tensor (B, 12). ``None`` -> returns ``None`` for actions.

    Returns:
        ``(obs_aug | None, actions_aug | None)``.
    """
    cache = _build_cache(env)
    joint_swap = cache["joint_swap"]
    hip_idx = cache["hip_idx"]
    contact_foot_swap = cache["contact_foot_swap"]
    friction_foot_swap = cache["friction_foot_swap"]

    # observations
    if obs is not None:
        b = obs.batch_size[0]
        # --- per-call shape gate (load-bearing: file comments say priv_latent=37/43, all STALE) ---
        if obs["policy"].shape[-1] != _PROPRIO_DIM:
            raise AssertionError(
                f"[parkour symmetry] policy dim {obs['policy'].shape[-1]} != {_PROPRIO_DIM}; "
                "proprio index map is stale."
            )
        if obs["scan"].shape[-1] not in (_SCAN_DIM, _SCAN_DIM_3D):
            raise AssertionError(
                f"[parkour symmetry] scan dim {obs['scan'].shape[-1]} not in "
                f"({_SCAN_DIM}, {_SCAN_DIM_3D}). "
                "Add new dimension or check env scan output."
            )
        if "voxel" in obs.keys() and obs["voxel"].shape[-1] != _VOXEL_DIM:
            raise AssertionError(
                f"[parkour symmetry] voxel dim {obs['voxel'].shape[-1]} != {_VOXEL_DIM}; "
                "check enable_voxel_scanner and parkour_env flatten order (C-order expected)."
            )
        if obs["priv_latent"].shape[-1] != _PRIV_LATENT_DIM:
            raise AssertionError(
                f"[parkour symmetry] priv_latent dim {obs['priv_latent'].shape[-1]} != {_PRIV_LATENT_DIM}; "
                "re-derive priv_latent index map from measured value."
            )

        out = obs.repeat(2)  # [original B; (to-be-filled) mirror B]
        out["policy"][b:] = _mirror_proprio(obs["policy"], joint_swap, hip_idx, contact_foot_swap)
        out["scan"][b:] = _mirror_scan(obs["scan"])
        # voxel (voxel-arm only — baseline obs do not contain this key; skip silently)
        if "voxel" in obs.keys():
            out["voxel"][b:] = _mirror_voxel(obs["voxel"])
        out["priv_explicit"][b:] = _mirror_priv_explicit(obs["priv_explicit"])
        out["priv_latent"][b:] = _mirror_priv_latent(obs["priv_latent"], joint_swap, friction_foot_swap)
        # history (N, T, 46): apply proprio mirror per time-frame (last dim = proprio).
        out["history"][b:] = _mirror_proprio(obs["history"], joint_swap, hip_idx, contact_foot_swap)
    else:
        out = None

    # actions (B, 12): same transform as proprio actions block -> joint swap + hip sign flip.
    if actions is not None:
        a = actions.repeat(2, 1)
        b_a = actions.shape[0]
        a[b_a:] = _swap_joints(actions, joint_swap, hip_idx)
    else:
        a = None

    return out, a
