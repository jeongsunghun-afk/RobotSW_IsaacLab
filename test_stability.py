import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test R_Skeleton Stability")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.R_Skeleton.skeleton_env_cfg import SkeletonHistoryEnvCfg, SkeletonHistoryFixedEnvCfg
from isaaclab_tasks.direct.R_Skeleton.skeleton_env import SkeletonEnv
import numpy as np

def main():
    cfg = SkeletonHistoryFixedEnvCfg()
    cfg.scene.num_envs = 4
    cfg.action_noise_model = None
    cfg.observation_noise_model = None
    # cfg.sim.disable_gravity = True
    
    env = SkeletonEnv(cfg=cfg)
    obs, info = env.reset()

    print(f"DEBUG: env.num_envs={env.num_envs}, env.action_space.shape={env.action_space.shape}")
    print(f"DEBUG: Joint Names: {env._robot.joint_names}")
    print(f"DEBUG: env._robot.data.default_joint_pos.shape={env._robot.data.default_joint_pos.shape}")
    print(f"DEBUG: env._robot.data.default_joint_pos[0]: {env._robot.data.default_joint_pos[0].cpu().numpy()}")
    
    # Check initial heights
    body_pos_w = env._robot.data.body_pos_w[0].cpu().numpy()
    body_names = env._robot.body_names
    min_z = np.min(body_pos_w[:, 2])
    print(f"DEBUG: Initial Minimum Z across all bodies: {min_z:.3f} m")
    for name, pos in zip(body_names, body_pos_w):
        if pos[2] < min_z + 0.05:
            print(f"  Low link: {name} at Z={pos[2]:.3f}")
    
    masses = env._robot.root_physx_view.get_masses()[0].cpu().tolist()
    body_names = env._robot.body_names
    print(f"DEBUG: Body Masses:")
    for name, m in zip(body_names, masses):
        print(f"  {name}: {m:.3f} kg")
    
    first_joint_pos = env._robot.data.joint_pos[0].clone()
    print(f"DEBUG: Using initial joint pos as targets: {first_joint_pos.cpu().numpy()}")

    # FrameTransformerSensor 발끝 위치 검증
    print("\n=== foot_pos_w 검증 (FrameTransformerSensor) ===")
    foot_names = ["FL_toe", "FR_toe", "HL_toe", "HR_toe"]
    foot_pos = env.foot_pos_w[0].cpu().numpy()  # env 0만 확인
    for i, (name, pos) in enumerate(zip(foot_names, foot_pos)):
        print(f"  {name}: world pos = {pos}, Z = {pos[2]:.4f} m")
    
    count = 0
    total_deaths = 0
    with open("stability_log.txt", "w") as f:
        while simulation_app.is_running() and count < 1000:
            actions = torch.zeros(env.action_space.shape, device=env.device)
            obs, reward, terminated, truncated, info = env.step(actions)
            # direct_rl_env resets internally. We count how many died this step:
            deaths_this_step = env.reset_terminated.sum().item()
            total_deaths += deaths_this_step
            
            if count == 0:
                print(f"DEBUG: Initial Joint Pos (Step 0): {env._robot.data.joint_pos[0].cpu().numpy()}")
            
            count += 1
            if count % 10 == 0:
                pos = env._robot.data.root_pos_w[0].cpu().numpy()
                grav = env._robot.data.projected_gravity_b[0].cpu().numpy()
                vel = env._robot.data.root_lin_vel_w[0].cpu().numpy()
                print(f"Step {count}: pos={pos}, grav={grav}, vel={vel}")
                # Print specific joint tracking error and efforts
                joint_poses = env._robot.data.joint_pos[0].cpu().numpy()
                joint_targets = env._robot.data.joint_pos_target[0].cpu().numpy()
                joint_efforts = env._robot.data.applied_torque[0].cpu().numpy()
                
                max_diff = np.max(np.abs(joint_poses - joint_targets))
                max_diff_idx = np.argmax(np.abs(joint_poses - joint_targets))
                max_joint_name = env._robot.joint_names[max_diff_idx]
                max_effort = np.max(np.abs(joint_efforts))
                print(f"Tracking diff max: {max_diff:.3f} on {max_joint_name}, Max Effort: {max_effort:.3f}")
                
                # Check contact forces for env 0
                net_contact_forces = env._contact_sensor.data.net_forces_w[0].cpu().numpy()
                # Find feet/toe links
                toe_indices = [i for i, name in enumerate(env._robot.body_names) if "toe" in name]
                for idx in toe_indices:
                    force = net_contact_forces[idx]
                    print(f"  Foot Force ({env._robot.body_names[idx]}): {np.linalg.norm(force):.1f} N - {force}")
                
            if count % 100 == 0:
                f.write(f"Step: {count} - Total Deaths: {total_deaths}\n")
                f.flush()
                print(f"Step: {count} - Total Deaths: {total_deaths}")
    env.close()

if __name__ == "__main__":
    main()
