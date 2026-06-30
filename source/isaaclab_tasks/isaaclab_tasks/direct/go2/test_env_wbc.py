# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test WBC flag")
parser.add_argument("--env", type=str, choices=["neck", "interaction"], required=True)
parser.add_argument("--wbc", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

# IsaacLab imports
from isaaclab_tasks.direct.go2.go2_env_cfg import Go2NeckFlatEnvCfg
from isaaclab_tasks.direct.go2.go2_neck_env import Go2NeckEnv
from isaaclab_tasks.direct.go2.go2_neck_interaction_cfg import Go2NeckInteractionCfg
from isaaclab_tasks.direct.go2.go2_neck_interaction_env import Go2NeckInteractionEnv


def test_wbc_flag(cfg_class, env_class, enable_wbc: bool):
    print(f"\n{'=' * 50}")
    print(f"Testing {env_class.__name__} with WBC = {enable_wbc}")
    print(f"{'=' * 50}")

    # 1. Configuration Init
    cfg = cfg_class()
    cfg.whole_body_control = enable_wbc
    cfg.scene.num_envs = 2  # small number for testing
    cfg.__post_init__()

    print(f"[Config] action_space      : {cfg.action_space}")
    print(f"[Config] num_prio_obs      : {cfg.num_prio_obs}")
    print(f"[Config] observation_space : {cfg.observation_space}")

    try:
        # 2. Environment Init
        print("Initializing environment...")
        env = env_class(cfg=cfg, render_mode=None)

        # 3. Check property dimensions directly from Env
        print(f"[Env] single_action_space : {env.single_action_space}")
        print(f"[Env] single_observation_space : {env.single_observation_space}")

        # 4. Try one step with zeros
        actions = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
        obs, _ = env.step(actions)

        print("\nStep Successful!")
        print(f"obs['policy'] shape : {obs['policy'].shape}")

        if not enable_wbc:
            assert obs["policy"].shape[1] == cfg.observation_space, "Observation space mismatch!"

        env.close()

    except Exception as e:
        print(f"Error initializing or stepping environment: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if args_cli.env == "neck":
        env_cls = Go2NeckEnv
        cfg_cls = Go2NeckFlatEnvCfg
    elif args_cli.env == "interaction":
        env_cls = Go2NeckInteractionEnv
        cfg_cls = Go2NeckInteractionCfg
    else:
        raise ValueError("Invalid env type")

    test_wbc_flag(cfg_cls, env_cls, enable_wbc=args_cli.wbc)
