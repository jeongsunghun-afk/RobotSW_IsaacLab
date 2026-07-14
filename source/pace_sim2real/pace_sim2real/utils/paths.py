# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import os
from pathlib import Path

# optional environment-variable override
PACE_ROOT_ENV = "PACE_ROOT"


def project_root() -> Path:
    """
    Returns the top-level project directory.

    Priority:
      1) $PACE_ROOT if set
      2) walk up from this file until we find a marker (pyproject.toml / .git)
      3) fallback: 3 levels above this file
    """
    if PACE_ROOT_ENV in os.environ:
        return Path(os.environ[PACE_ROOT_ENV]).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / ".project-root").exists() or (parent / ".git").exists():
            return parent

    # fallback: 3 levels up from utils/
    return here.parents[3]
