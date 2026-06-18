# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left/Right symmetry augmentation wrapper for the Go2 ParkourImitation direct environment.

This module provides a thin ``data_augmentation_func`` consumed by ``PPOParkourAMP`` (rsl_rl)
to softly enforce policy L/R equivariance via mirror data-augmentation.

The ParkourImitation environment inherits the same observation structure as the base Parkour
environment (policy/scan/priv_explicit/priv_latent/history groups are identical), so this
module simply delegates to ``compute_parkour_symmetric_states`` from the parent task.

AMP note:
    AMP observations are NOT part of the obs TensorDict passed to the data-augmentation
    function.  They are delivered exclusively via ``env.extras["amp_obs"]`` to the PPOAMP
    discriminator pipeline, which is entirely separate from the PPO data-augmentation path.
    Therefore no amp_obs mirroring is needed here and must NOT be added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export the parent task's symmetric-state function verbatim.
# obs group layout (policy/scan/priv_explicit/priv_latent/history) is identical to parkour.
from isaaclab_tasks.direct.parkour.mdp.symmetry import compute_parkour_symmetric_states

if TYPE_CHECKING:
    import torch
    from tensordict import TensorDict

__all__ = ["compute_parkour_imitation_symmetric_states"]


def compute_parkour_imitation_symmetric_states(
    *, env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
):
    """Augment ParkourImitation observations/actions with the left/right mirror (num_aug=2).

    Thin wrapper around ``compute_parkour_symmetric_states``.  The observation groups
    (policy/scan/priv_explicit/priv_latent/history) are structurally identical between
    parkour and parkour_imitation, so no additional transformation is required.

    Args:
        env: the (wrapped) VecEnv instance.
        obs: TensorDict with groups ``policy(46) / scan(187) / priv_explicit(6) /
            priv_latent(37) / history(N, 10, 46)``.  ``None`` -> returns ``None`` for obs.
        actions: action tensor (B, 12).  ``None`` -> returns ``None`` for actions.

    Returns:
        ``(obs_aug | None, actions_aug | None)`` — same contract as the parent function.
    """
    return compute_parkour_symmetric_states(env=env, obs=obs, actions=actions)
