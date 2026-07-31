# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left/Right (L<->R) symmetry augmentation for the Leg(17-DOF) ImitationTracking-RMA env.

Provides a single ``data_augmentation_func`` consumed by ``PPOAMP`` (rsl_rl) to softly enforce
policy L/R equivariance via mirror data-augmentation (``num_aug = 2``). The task is forward-only
(command = body-frame vx, vy≡0), so only the single sagittal-plane (L/R) mirror holds.

Observation groups (``leg_imitation_tracking_rma_env.py:_get_observations``):

  - policy(57)        : gravity(3) + lin_vel_cmd(2) + yaw_vel_cmd(1) +
                        joint_pos_off(17) + joint_vel(17) + actions(17)
  - priv_explicit(6)  : root_link_lin_vel_b(3) + root_link_ang_vel_b(3)
  - priv(38)          : base_mass(1) + base_com(3) + joint_stiffness_ratio(17) + joint_damping_ratio(17)
  - history(H, 57)    : proprio ring buffer (same layout as policy)

Joint mirror (built dynamically from runtime ``_robot.data.joint_names``):
    - L<->R swap: HL<->HR, FL<->FR, FB_waist -> itself (name-based, no hard-coded indices).
    - sign flip after swap (empirically verified against the roughly-symmetric SMR reference,
      and consistent with the Leg URDF joint axes):
        * all ``*_hip_joint``  (abduction, axis ≈ x)      -> FLIP
        * ``HL_foot_joint`` / ``HR_foot_joint`` (hind feet) -> FLIP  (URDF axis (0,±1,0) is
          sign-inconsistent between HL/HR, so the raw joint values are negated across the pair)
        * ``FB_waist_joint`` (yaw, axis z)                -> FLIP
        * all ``*_thigh_joint`` / ``*_calf_joint`` and the FRONT feet -> NO flip (sagittal pitch)
      The flip index set is invariant under the L<->R swap, so negating at ``flip_idx`` after the
      swap is well defined (same construction as the parkour hip-only case, generalized).

AMP note:
    AMP observations are delivered exclusively via ``env.extras["amp_obs"]`` to the discriminator
    pipeline, which is separate from the PPO data-augmentation path. They are NOT part of the obs
    TensorDict passed here, so no amp_obs mirroring is needed and must NOT be added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

__all__ = ["compute_leg_symmetric_states"]

# --- obs group dims (leg_imitation_tracking_rma_env.py) ---
_PROPRIO_DIM = 57
_PRIV_EXPLICIT_DIM = 6
_PRIV_DIM = 38
_NUM_JOINTS = 17

# proprio (policy / history) block offsets
_G = slice(0, 3)  # projected_gravity_b
_LVC = slice(3, 5)  # lin_vel_cmd (vx, vy)
_YVC = 5  # yaw_vel_cmd
_JP = slice(6, 23)  # joint_pos - default
_JV = slice(23, 40)  # joint_vel
_ACT = slice(40, 57)  # actions

# priv block offsets ([0] base_mass is a magnitude scalar -> unchanged, no slice needed)
_COM = slice(1, 4)
_STIFF = slice(4, 21)
_DAMP = slice(21, 38)


def _flip_side(prefix: str) -> str:
    """Flip the left/right token in a Leg limb prefix ("HL"<->"HR", "FL"<->"FR")."""
    if prefix.endswith("L"):
        return prefix[:-1] + "R"
    if prefix.endswith("R"):
        return prefix[:-1] + "L"
    return prefix


def _mirror_name(name: str) -> str:
    """Return the L<->R mirrored joint name (e.g. "FL_hip_joint" -> "FR_hip_joint")."""
    prefix, _, rest = name.partition("_")
    return f"{_flip_side(prefix)}_{rest}" if rest else _flip_side(prefix)


def _needs_flip(name: str) -> bool:
    """Whether a joint's value is sign-flipped after the L<->R swap.

    Verified against the (roughly L/R-symmetric) SMR reference distribution:
        hip -> FLIP, thigh/calf -> NO, hind foot -> FLIP, front foot -> NO, waist(yaw) -> FLIP.
    """
    if "hip" in name:
        return True
    if "waist" in name:
        return True
    if "foot" in name and (name.startswith("HL") or name.startswith("HR")):
        return True
    return False


def _build_cache(env) -> dict:
    """Build (and cache on ``env.unwrapped``) the joint swap permutation + flip index set."""
    u = env.unwrapped
    cache = getattr(u, "_leg_symmetry_cache", None)
    if cache is not None:
        return cache

    device = u.device
    joint_names = list(u._robot.data.joint_names)
    if len(joint_names) != _NUM_JOINTS:
        raise ValueError(f"[leg symmetry] expected {_NUM_JOINTS} joints, got {len(joint_names)}: {joint_names}")

    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    perm = []
    for n in joint_names:
        mirrored = _mirror_name(n)
        if mirrored not in name_to_idx:
            raise ValueError(
                f"[leg symmetry] no L/R mirror match for '{n}' (expected '{mirrored}'). names={joint_names}"
            )
        perm.append(name_to_idx[mirrored])
    joint_swap = torch.tensor(perm, dtype=torch.long, device=device)
    if torch.unique(joint_swap).numel() != _NUM_JOINTS:
        raise ValueError(f"[leg symmetry] swap perm is not a valid permutation: {perm} (names={joint_names})")

    flip_idx = torch.tensor([i for i, n in enumerate(joint_names) if _needs_flip(n)], dtype=torch.long, device=device)
    # flip set must be invariant under the swap (so post-swap negation at flip_idx is well defined)
    swapped_flip = torch.sort(joint_swap[flip_idx]).values
    if not torch.equal(swapped_flip, torch.sort(flip_idx).values):
        raise ValueError(
            f"[leg symmetry] flip index set is not swap-invariant. flip={[joint_names[i] for i in flip_idx.tolist()]}"
        )

    cache = {"joint_swap": joint_swap, "flip_idx": flip_idx}
    u._leg_symmetry_cache = cache
    return cache


def _swap_joints(j: torch.Tensor, joint_swap: torch.Tensor, flip_idx: torch.Tensor) -> torch.Tensor:
    """L<->R joint swap + per-joint sign flip. Operates on the last dim (17 DOF).

    Works for arbitrary leading dims (used for both [B, 17] and history [B, T, 17]).
    """
    out = j[..., joint_swap].clone()
    out[..., flip_idx] = -out[..., flip_idx]
    return out


def _swap_joints_no_flip(j: torch.Tensor, joint_swap: torch.Tensor) -> torch.Tensor:
    """L<->R joint swap WITHOUT sign flip — for magnitude quantities (stiffness/damping ratios)."""
    return j[..., joint_swap].clone()


def _mirror_proprio(p: torch.Tensor, joint_swap: torch.Tensor, flip_idx: torch.Tensor) -> torch.Tensor:
    """Mirror a proprio tensor (..., 57). Last-dim layout per module docstring."""
    m = p.clone()
    sign_g = torch.tensor([1.0, -1.0, 1.0], device=p.device)
    m[..., _G] = p[..., _G] * sign_g  # projected_gravity_b: y reversed
    m[..., _LVC] = p[..., _LVC] * torch.tensor([1.0, -1.0], device=p.device)  # lin_vel_cmd: vy reversed
    m[..., _YVC] = -p[..., _YVC]  # yaw_vel_cmd: reversed
    m[..., _JP] = _swap_joints(p[..., _JP], joint_swap, flip_idx)  # joint_pos - default
    m[..., _JV] = _swap_joints(p[..., _JV], joint_swap, flip_idx)  # joint_vel
    m[..., _ACT] = _swap_joints(p[..., _ACT], joint_swap, flip_idx)  # actions
    return m


def _mirror_priv_explicit(pe: torch.Tensor) -> torch.Tensor:
    """Mirror priv_explicit (..., 6): lin_vel_b[0:3]*[1,-1,1], ang_vel_b[3:6]*[-1,1,-1]."""
    m = pe.clone()
    m[..., 0:3] = pe[..., 0:3] * torch.tensor([1.0, -1.0, 1.0], device=pe.device)
    m[..., 3:6] = pe[..., 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=pe.device)
    return m


def _mirror_priv(pl: torch.Tensor, joint_swap: torch.Tensor) -> torch.Tensor:
    """Mirror priv (..., 38): base_mass unchanged, base_com y-reversed, stiffness/damping = swap only.

    stiffness/damping ratios are MAGNITUDES -> joint swap ONLY, sign flip is forbidden.
    """
    m = pl.clone()
    # base_mass [0]: scalar, unchanged
    m[..., _COM] = pl[..., _COM] * torch.tensor([1.0, -1.0, 1.0], device=pl.device)  # base_com: y reversed
    m[..., _STIFF] = _swap_joints_no_flip(pl[..., _STIFF], joint_swap)  # stiffness ratio (no flip)
    m[..., _DAMP] = _swap_joints_no_flip(pl[..., _DAMP], joint_swap)  # damping ratio (no flip)
    return m


@torch.no_grad()
def compute_leg_symmetric_states(*, env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Augment Leg-ImitationTracking-RMA observations/actions with the L/R mirror (num_aug = 2).

    Returns a batch of ``[original B; mirrored B]`` for each of ``obs`` and ``actions``. The first
    ``B`` rows are always the originals (rsl_rl relies on this ordering).

    Args:
        env: the (wrapped) VecEnv instance; ``env.unwrapped`` exposes the leg RMA env.
        obs: TensorDict with groups ``policy(57) / priv_explicit(6) / priv(38) / history(N, H, 57)``.
            ``None`` -> returns ``None`` for obs.
        actions: action tensor (B, 17). ``None`` -> returns ``None`` for actions.

    Returns:
        ``(obs_aug | None, actions_aug | None)``.
    """
    cache = _build_cache(env)
    joint_swap = cache["joint_swap"]
    flip_idx = cache["flip_idx"]

    # observations
    if obs is not None:
        b = obs.batch_size[0]
        # per-call shape gate (hard-fail on any obs layout drift)
        if obs["policy"].shape[-1] != _PROPRIO_DIM:
            raise AssertionError(f"[leg symmetry] policy dim {obs['policy'].shape[-1]} != {_PROPRIO_DIM}")
        if obs["priv_explicit"].shape[-1] != _PRIV_EXPLICIT_DIM:
            raise AssertionError(
                f"[leg symmetry] priv_explicit dim {obs['priv_explicit'].shape[-1]} != {_PRIV_EXPLICIT_DIM}"
            )
        if obs["priv"].shape[-1] != _PRIV_DIM:
            raise AssertionError(f"[leg symmetry] priv dim {obs['priv'].shape[-1]} != {_PRIV_DIM}")

        out = obs.repeat(2)  # [original B; (to-be-filled) mirror B]
        out["policy"][b:] = _mirror_proprio(obs["policy"], joint_swap, flip_idx)
        out["priv_explicit"][b:] = _mirror_priv_explicit(obs["priv_explicit"])
        out["priv"][b:] = _mirror_priv(obs["priv"], joint_swap)
        # history (N, H, 57): apply proprio mirror per time-frame (last dim = proprio).
        out["history"][b:] = _mirror_proprio(obs["history"], joint_swap, flip_idx)
    else:
        out = None

    # actions (B, 17): same transform as the proprio actions block -> joint swap + sign flip.
    if actions is not None:
        a = actions.repeat(2, 1)
        b_a = actions.shape[0]
        a[b_a:] = _swap_joints(actions, joint_swap, flip_idx)
    else:
        a = None

    return out, a
