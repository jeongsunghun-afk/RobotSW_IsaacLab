# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from skrl.utils.tensorboard import SummaryWriter

log_dir = "test_tb_logs"
writer = SummaryWriter(log_dir=log_dir)
print("Writer created")
writer.add_scalar(tag="test/loss", value=0.5, timestep=0)
writer.flush()
print("Wrote scalar")
writer.close()
