from skrl.utils.tensorboard import SummaryWriter
import os, time

log_dir = "test_tb_logs"
writer = SummaryWriter(log_dir=log_dir)
print("Writer created")
writer.add_scalar(tag="test/loss", value=0.5, timestep=0)
writer.flush()
print("Wrote scalar")
writer.close()
