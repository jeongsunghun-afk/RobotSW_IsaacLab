import sys
import torch

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})

import traceback
try:
    from isaaclab_tasks.direct.R_Skeleton.skeleton_amp_env import SkeletonAmpEnv
    from isaaclab_tasks.direct.R_Skeleton.skeleton_env_cfg import SkeletonEnvCfg

    env_cfg = SkeletonEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.motion_file = "/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton/imitation/txt_datasets_sample"
    env = SkeletonAmpEnv(env_cfg)

    print(f"ref_idx: {env.motion_ref_body_index}")
    print(f"key_idx: {env.motion_key_body_indexes}")
    print(f"body_shape: {env._motion_loader.body_positions.shape}")
    print("TEST SUCCESS")
except Exception as e:
    traceback.print_exc()

app_launcher.app.close()
