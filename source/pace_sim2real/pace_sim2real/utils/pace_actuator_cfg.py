# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from isaaclab.actuators import DCMotorCfg
from isaaclab.utils.configclass import configclass

from pace_sim2real.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model.

    This class extends the base DCMotorCfg with Pace-specific parameters.
    """

    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: dict[str, float] | list[float] | float | None = 0.0
    max_delay: int = 0
