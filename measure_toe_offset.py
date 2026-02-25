"""
ankle_r에서 toe 링크까지의 로컬 오프셋을 측정하는 스크립트.
원본 R_skeleton.usd (toe joint 있는 버전)에서 시뮬레이션 초기에
ankle_r 위치와 toe 위치의 차이를 계산합니다.
"""
import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 원본 USD (toe joint 있는 버전)를 사용합니다
from isaaclab_tasks.direct.R_Skeleton.skeleton_env_cfg import SkeletonHistoryEnvCfg
from isaaclab_tasks.direct.R_Skeleton.skeleton_env import SkeletonEnv

def main():
    cfg = SkeletonHistoryEnvCfg()
    cfg.scene.num_envs = 1
    cfg.action_noise_model = None
    cfg.observation_noise_model = None

    env = SkeletonEnv(cfg=cfg)
    env.reset()

    body_names = env._robot.body_names
    body_pos = env._robot.data.body_pos_w[0].cpu().numpy()

    # ankle_r과 toe 링크를 찾아 오프셋 계산
    print("=" * 60)
    print("ankle_r 및 toe 링크 위치 분석")
    print("=" * 60)

    for leg in ["FL", "FR", "HL", "HR"]:
        # 앞다리(FL/FR)는 wrist_r, 뒷다리(HL/HR)는 ankle_r
        if leg in ("FL", "FR"):
            parent_name = f"{leg}_link6_wrist_r"
        else:
            parent_name = f"{leg}_link6_ankle_r"
        toe_name = f"{leg}_link7_toe"

        parent_idx = next((i for i, n in enumerate(body_names) if n == parent_name), None)
        toe_idx = next((i for i, n in enumerate(body_names) if n == toe_name), None)

        if parent_idx is not None and toe_idx is not None:
            parent_pos = body_pos[parent_idx]
            toe_pos = body_pos[toe_idx]
            offset = toe_pos - parent_pos
            print(f"\n[{leg}]")
            print(f"  parent ({parent_name}) pos (world): {parent_pos}")
            print(f"  toe pos (world): {toe_pos}")
            print(f"  offset (toe - parent): {offset}")
            print(f"  >> FrameTransformerSensor offset: ({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})")
        else:
            if parent_idx is None:
                print(f"[{leg}] parent NOT found. 사용 가능한 링크:")
                for n in body_names:
                    if leg in n:
                        print(f"  {n}")

    env.close()

if __name__ == "__main__":
    main()
