import argparse
import os
import sys
import pickle
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from interaction_env import InteractionEnv
from rsl_rl.runners import OnPolicyRunnerParkour

import genesis as gs
import torch

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

def get_train_cfg(exp_name, max_iterations, obs_cfg):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1.e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            # dagger parameters
            "dagger_update_freq": 20,
            "priv_reg_coef_schedual": [0, 0.1, 2000, 3000],
            "priv_reg_coef_schedual_resume": [0, 0.1, 0, 1]
        },
        "init_member_classes": {},
        "policy": {
            "scan_encoder_dims": [128, 64, 32],
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "priv_encoder_dims": [64, 20],
            "tanh_encoder_output": False
        },
        "runner": {
            "algorithm_class_name": "PPOParkour",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCriticRMA",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "estimator": {
            'train_with_estimated_states': True,
            'learning_rate': 1.e-4,
            'hidden_dims': [128, 64],
            'priv_states_dim': obs_cfg["num_priv"],
            'num_prop': obs_cfg["num_prio_obs"],
            'num_scan': obs_cfg["num_heights"]
        },
        'depth_encoder':{
            "if_depth": False,
            "depth_shape": None,
            "buffer_len": None,
            "hidden_dims": 512,
            "learning_rate": 1.e-3,
            "num_steps_per_env": None
        },
        "runner_class_name": "OnPolicyRunnerParkour",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "default_sitting_joint_angles": {
        'FL_hip_joint': 0.1,  
        'RL_hip_joint': 0.1,  
        'FR_hip_joint': -0.1,  
        'RR_hip_joint': -0.1,  

        'FL_thigh_joint': 1.57,  
        'RL_thigh_joint': 1.57,  
        'FR_thigh_joint': 1.57,  
        'RR_thigh_joint': 1.57,  

        'FL_calf_joint': -2.67,  
        'RL_calf_joint': -2.67,  
        'FR_calf_joint': -2.67,  
        'RR_calf_joint': -2.67  
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "left_dof_names": [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "right_dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        "feet_names":[
            "FR_calf",
            "FL_calf",
            "RR_calf",
            "RL_calf"
        ],
        "hip_names":[
            "FR_hip_joint", 
            "FL_hip_joint",
            "RR_hip_joint",
            "RL_hip_joint"
        ],
        # "penalize_contacts_name": [
        #     "base",
        #     "FR_thigh",
        #     "FR_calf",
        #     "FL_thigh",
        #     "FL_calf",
        #     "RR_thigh",
        #     "RR_calf",
        #     "RL_thigh",
        #     "RL_calf",
        # ],
        "penalize_contacts_name": [
            "base",
            "FR_thigh",
            "FL_thigh",
            "RR_thigh",
            "RL_thigh",
        ],
        "terminate_after_contacts_on": [
            "base"
        ],
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'thigh', 'trunk'],
        'feet_link_names': ['foot'],
        'hip_link_names': ['shoulder'],
        'base_link_name': ['base'],
        # PD
        "kp": 20.0,
        "kd": 1.0,
        # termination
        "termination_if_roll_greater_than": 1.484,  # rad (85 degrees)
        "termination_if_pitch_greater_than": 1.484,  # rad (85 degrees)
        # base pose
        "base_init_pos": [0.0, 0.0, 0.34],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 4.0,
        "resampling_time_s": 5.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100, # 10
        "next_goal_threshold": 0.2,
        "reach_goal_delay": 0.1,
        "domain_rand":{
            "randomize_friction": True,
            "friction_range": [0.5, 1.25],
            "randomize_base_mass": True,
            "added_mass_range": [-1., 3.],
            "randomize_com_displacement": True,
            "com_displacement_range": [-0.1, 0.1],
            "push_robots": True,
            "push_interval_s": 15,
            "max_push_vel_xy": 1.,
            "simulate_action_latency": True,
            "motor_strength_range": [0.8, 1.2]
        }
    }
    
    # dof pos limits
    # action smoothness 1,2
    reward_cfg = {
        "sigma": 0.2,
        # "base_height_target": 0.34,
        # "feet_height_target": 0.075,
        "reward_scales": {
            # Tracking Rewards
            "hip_positions": 0.5,
            "foot_positions": 0.5, 
            "base_height": 1.5,
            "base_pitch": 1.5,
            "feet_contact": 0.5,
            "foot_parallel": 0.0,
            # Regularization
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "delta_torques": -1.0e-5,
            "torques": -0.0001,
            "similar_to_default": -0.0,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.5,
            "torques_balance": -0.01,
        },
    }
    # reward_cfg = {
    #     "tracking_sigma": 0.25,
    #     "base_height_target": 0.3,
    #     "feet_height_target": 0.075,
    #     "reward_scales": {
    #         "tracking_lin_vel": 1.0,
    #         "tracking_ang_vel": 0.2,
    #         "lin_vel_z": -1.0,
    #         "base_height": -50.0,
    #         "action_rate": -0.005,
    #         "similar_to_default": -0.1,
    #     },
    # }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.3, 0.3],
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [-0.3, 0.3],
    }

    terrain_dict = {
        "flat_terrain" : 0.0,
        "fractal_terrain": 0.0,
        "random_uniform_terrain": 0.0,
        "sloped_terrain": 0.0,
        "pyramid_sloped_terrain": 0.0,
        "discrete_obstacles_terrain": 0.0,
        "wave_terrain": 0.0,
        "stairs_terrain": 0.0,
        "pyramid_stairs_terrain": 0.0,
        "stepping_stones_terrain": 0.0,
        "parkour_terrain": 0.0,
        "parkour_hurdle_terrain": 0.0,
        "parkour_flat_terrain": 1.0,
        "parkour_step_terrain": 0.0,
        "parkour_gap_terrain": 0.0,
        "parkour_stair_terrain": 0.0,
        "demo_terrain": 0.0
    }
    terrain_cfg = {
        "mesh_type": "trimesh",
        "hf2mesh_method": "grid",
        "max_error": 0.1,
        "max_error_camera": 2,
        "y_range": [-0.4, 0.4],
        "edge_width_thresh": 0.05,
        "horizontal_scale": 0.1,
        "horizontal_scale_camera": 0.1,
        "vertical_scale": 0.005,
        "border_size": 5,
        "height": [0.02, 0.06],
        "simplify_grid": False,
        "gap_size": [0.02, 0.1],
        "stepping_stone_distance": [0.02, 0.08],
        "downsampled_scale": 0.075,
        "curriculum": True,
        "all_vertical": False,
        "no_flat": True,
        "static_friction": 1.0,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
        "measure_heights": False,
        "measured_points_x": [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2],
        "measured_points_y": [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75],
        "measure_horizontal_noise": 0.0,
        "selected": False,
        "terrain_kwargs": None,
        "max_init_terrain_level": 5,
        "terrain_length": 10.,
        "terrain_width": 10.,
        "num_rows": 1,
        "num_cols": 1,
        "terrain_proportions": list(terrain_dict.values()),
        "terrain_dict": terrain_dict,
        "slope_treshold": 1.5,
        "origin_zero_z": True,
        "num_goals": 8,
        "max_platform_height": 0.2,
        "parkour": False,
        "x_init_range": 0.,
        "y_init_range": 0.,
        "update_interval" : 5
    }

    priv_explicit = True
    priv_latent = True
    ang_vel = False

    num_prio_obs = 42  # 3(gravity) + 1(interaction_cmd) + 12(dof_pos) + 12(dof_vel) + 12(actions) + 2(roll,pitch)
    if terrain_cfg['measure_heights']:
        num_heights = 132
    else:
        num_heights = 0
    
    if priv_explicit:
        num_priv = 3 
    else:
        num_priv = 0

    if priv_latent:
        num_priv_latent = 4 + 18
    else:
        num_priv_latent = 0
    
    history_len = 10

    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0
        },
        "history_encoding": True,
        "priv_explicit": priv_explicit,
        "priv_latent": priv_latent,
        "ang_vel": ang_vel,
        "num_future_goal_obs": 2,
        "history_len": history_len,
        "num_heights": num_heights,
        "num_priv": num_priv,
        "num_priv_latent": num_priv_latent,
        "num_prio_obs": num_prio_obs,
        "num_obs": num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len,
        "num_privileged_obs": None,  # Set to None to use obs as critic_obs
        "add_noise": True,
        "noise_scales": {
            'rotation': 0.01,
            'dof_pos': 0.01,
            'dof_vel': 0.05,
            'lin_vel': 0.05,
            'ang_vel': 0.05,
            'gravity': 0.02,
            'height_measurements': 0.02
        }
    }
    depth_cfg = {
        "use_camera": False,
        "camera_num_envs": 192,
        "camera_terrain_num_rows": 10,
        "camera_terrain_num_cols": 20,
        "position": [0.27, 0, 0.03],  # front camera
        "angle": [-5, 5],  # positive pitch down
        "update_interval": 5,  # 5 works without retraining, 8 worse
        "original": (106, 60),
        "resized": (87, 58),
        "horizontal_fov": 87,
        "buffer_len": 2,
        "near_clip": 0,
        "far_clip": 2,
        "dis_noise": 0.0,
        "scale": 1,
        "invert": True
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, depth_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="no_name")
    parser.add_argument("-B", "--num_envs", type=int, default=6144)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--depth", action='store_true')
    parser.add_argument("-r", "--resume_name", type=str, default="go1_parkour")
    parser.add_argument("--ckpt", type=int, default=1000)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = 'cuda:0' 
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    gs.init(logging_level="warning")

    log_dir = f"logs/interaction/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, depth_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, obs_cfg)

    if args.depth:
        args.num_envs = depth_cfg["camera_num_envs"]
        terrain_cfg["num_rows"] = depth_cfg["camera_terrain_num_rows"]
        terrain_cfg["num_cols"] = depth_cfg["camera_terrain_num_cols"]
        depth_cfg["use_camera"] = True

        train_cfg["depth_encoder"]["if_depth"] = True
        train_cfg["depth_encoder"]["depth_shape"] = depth_cfg["resized"]
        train_cfg["depth_encoder"]["buffer_len"] = depth_cfg["buffer_len"]
        train_cfg["depth_encoder"]["num_steps_per_env"] = depth_cfg["update_interval"] * 24



    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = InteractionEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, terrain_cfg=terrain_cfg, device=device, depth_cfg=depth_cfg, show_viewer=False
    )
    
    # rgb, _, _, _ = env.view_cam.render()

    # runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner = OnPolicyRunnerParkour(env, train_cfg, log_dir, device=device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    if args.depth:
        log_dir = f"logs/{args.resume_name}"
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        runner.load(resume_path)
        runner.learn_depth(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    else:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        

if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
