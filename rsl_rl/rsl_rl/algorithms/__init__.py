# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .distillation import Distillation
from .ppo import PPO
from .ppo_amp import PPOAMP, PPOAMPBase
from .ppo_parkour import PPOParkour

__all__ = ["PPO", "PPOAMP", "Distillation", "PPOAMPBase", "PPOParkour"]
