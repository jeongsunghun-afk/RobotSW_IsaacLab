import os
import torch
import sys

# Adjust the path
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.isaaclab_tasks.direct.R_Skeleton.skeleton_amp_env import SkeletonAmpEnv
from isaaclab_tasks.isaaclab_tasks.direct.R_Skeleton.skeleton_amp_env_cfg import SkeletonAmpEnvCfg

def main():
    cfg = SkeletonAmpEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = "cpu"
    
    env = SkeletonAmpEnv(cfg=cfg)
    
    print("Environment created successfully.")
    obs, info = env.reset()
    print("Environment reset successfully.")
    
    # Run a few steps
    for _ in range(5):
        actions = torch.zeros(2, cfg.action_space)
        obs, rew, terminated, truncated, info = env.step(actions)
        
    print("Environment step successfully completed.")

if __name__ == "__main__":
    main()
