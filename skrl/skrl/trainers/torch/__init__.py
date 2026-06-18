# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from skrl.trainers.torch.base import Trainer, TrainerCfg, generate_equally_spaced_scopes  # isort:skip

from skrl.trainers.torch.parallel import ParallelTrainer, ParallelTrainerCfg
from skrl.trainers.torch.sequential import SequentialTrainer, SequentialTrainerCfg
from skrl.trainers.torch.step import StepTrainer, StepTrainerCfg
