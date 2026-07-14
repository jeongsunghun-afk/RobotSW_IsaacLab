# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0
#
# IsaacLab 6.0 port: physics-backend guard (added by the IsaacLab-6.0 vendoring).

from __future__ import annotations


def require_physx_backend(env) -> None:
    """Raise if the environment does not run on the PhysX physics backend.

    PACE identifies joint dynamic and viscous friction. In IsaacLab 6.0 only the PhysX backend
    accepts those coefficients; the Newton backend exposes static joint friction alone, so a
    fit launched on Newton would silently optimize two thirds of its parameters into a no-op.

    Args:
        env: The Gym environment wrapping a :class:`~isaaclab.envs.ManagerBasedRLEnv`.

    Raises:
        RuntimeError: If the active physics manager is not PhysX.
    """
    manager_name = env.unwrapped.sim.physics_manager.__name__.lower()
    if "physx" not in manager_name:
        raise RuntimeError(
            f"PACE requires the PhysX physics backend, but the environment runs on '{manager_name}'."
            " The Newton backend only supports static joint friction, so joint dynamic and viscous"
            " friction (24 of the optimized parameters) would never reach the simulation."
        )
    print(f"[INFO]: Physics backend check passed: {manager_name}")
