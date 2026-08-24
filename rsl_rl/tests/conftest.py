# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test fixtures for rsl_rl tests."""

import torch
from tensordict import TensorDict


def make_obs(num_envs: int, obs_dim: int = 8, device: str = "cpu") -> TensorDict:
    """Create an observation TensorDict with a single 'policy' key."""
    return TensorDict(
        {"policy": torch.randn(num_envs, obs_dim, device=device)},
        batch_size=[num_envs],
        device=device,
    )
