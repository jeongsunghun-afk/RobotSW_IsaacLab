import os
import torch
import numpy as np
from isaaclab.app import AppLauncher

# AppLauncher 초기화
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation
from isaaclab_assets.robots.rga import R_SKELETON_CFG
import isaaclab.sim as sim_utils

def check_joint_axes():
    # 씬 구성
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 로봇 스폰
    robot_cfg = R_SKELETON_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    
    sim.reset()
    robot.update(0.01)
    
    print("\n" + "="*60)
    print("R_Skeleton Joint Sign & Direction Check")
    print("="*60)
    
    joint_names = robot.joint_names
    num_joints = len(joint_names)
    
    # 1. Default Position 확인
    print(f"\nDefault Joint Positions (for env 0):")
    for i, name in enumerate(joint_names):
        print(f"  [{i:2d}] {name:30} : {robot.data.default_joint_pos[0, i]:.4f}")
    
    # 2. 각 관절을 +0.1만큼 움직였을 때 Toe의 변화 확인 (Axis Sign 추정)
    print(f"\nChecking Joint Axis Signs (Moving +0.1 rad):")
    toe_indices = [i for i, name in enumerate(robot.data.body_names) if "toe" in name]
    toe_names = [robot.data.body_names[i] for i in toe_indices]
    
    for j_idx in range(num_joints):
        # 원본 위치 저장
        initial_toe_pos = robot.data.body_pos_w[0, toe_indices].clone()
        
        # J_idx 관절만 +0.1 이동
        target_pos = robot.data.default_joint_pos.clone()
        target_pos[0, j_idx] += 0.5 # 눈에 띄게 0.5rad(약 28도) 이동
        
        # 시뮬레이션에 적용 (즉시 반영을 위해 write_joint_state 사용)
        robot.write_joint_state_to_sim(target_pos, torch.zeros_like(target_pos))
        sim.step()
        robot.update(0.01)
        
        final_toe_pos = robot.data.body_pos_w[0, toe_indices]
        diff = final_toe_pos - initial_toe_pos
        
        # 가장 많이 움직인 발가락 찾기
        max_diff_val, toe_idx = torch.max(torch.norm(diff, dim=-1), dim=0)
        if max_diff_val > 0.001:
            move_dir = diff[toe_idx]
            print(f"  Joint [{j_idx:2d}] {joint_names[j_idx]:30} -> Moves {toe_names[toe_idx]} in {move_dir.tolist()}")
            
    simulation_app.close()

if __name__ == "__main__":
    check_joint_axes()
