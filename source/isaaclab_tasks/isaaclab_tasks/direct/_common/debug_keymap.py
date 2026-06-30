# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Key-binding and viewer debug configuration dataclasses."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class DebugKeyBindingCfg:
    """Per-feature key bindings. None = disabled.

    Key names must match ``carb.input.KeyboardInput.<NAME>`` attribute names.
    """

    toggle_free_fly: str | None = "F"
    free_fly_forward: str | None = "W"
    free_fly_back: str | None = "S"
    free_fly_left: str | None = "A"
    free_fly_right: str | None = "D"
    free_fly_forward_alt: str | None = "UP"
    free_fly_back_alt: str | None = "DOWN"
    free_fly_left_alt: str | None = "LEFT"
    free_fly_right_alt: str | None = "RIGHT"
    free_fly_up: str | None = "E"
    free_fly_down: str | None = "Q"
    free_fly_speed_up: str | None = "LEFT_SHIFT"
    env_index_prev: str | None = "LEFT_BRACKET"
    env_index_next: str | None = "RIGHT_BRACKET"


@configclass
class DebugViewerCfg:
    """Common viewer / debug-infrastructure configuration."""

    enabled: bool = True
    keys: DebugKeyBindingCfg = DebugKeyBindingCfg()
    free_fly_speed_mps: float = 2.0
    free_fly_speed_boost_mps: float = 6.0
    # When True, update() will invoke viewport auto-tracking each step.
    # Set to False when the env registers its own camera-follow callback
    # (e.g. parkour's robot-yaw-based tracker).
    auto_follow_asset: bool = False
    auto_follow_asset_name: str | None = None  # cfg.viewer.asset_name takes precedence
