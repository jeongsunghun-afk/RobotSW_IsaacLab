# Go2 Parkour Environment

## Overview
Go2 robot parkour locomotion over challenging terrain (stairs, gaps, stepping stones, slopes).

## Training
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Parkour-Direct-v0 --num_envs 4096
```

## Evaluation
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Go2-Parkour-Direct-v0 --num_envs 32
```

## Key Files
- `parkour_env_cfg.py`: Environment configuration (terrain, rewards, sensors) → `ParkourEnvCfg`
- `parkour_env.py`: Environment implementation → `Go2ParkourEnv`
- `agents/rsl_rl_ppo_cfg.py`: RSL-RL PPO training config → `Go2ParkourPPORunnerCfg`

## Architecture
- Terrain: Procedural generation with curriculum (10 difficulty rows × 20 type cols)
  - Sub-terrains: pyramid_stairs, pyramid_stairs_inv, boxes, random_rough, hf_pyramid_slope, hf_pyramid_slope_inv, stepping_stones, gaps
- Observations: projected_gravity(3) + commands(3) + joint_pos(12) + joint_vel(12) + prev_actions(12) + height_scan(187) = 229
- Rewards: velocity tracking + regularization penalties (collision, stumble, edge, torque, etc.)
- Algorithm: PPO with [512, 256, 128] actor-critic networks, lr=2e-4

## Env Specs (Go2)
- DOF: 12 (action_space = 12)
- episode_length_s: 20.0
- decimation: 4 (200Hz physics / 50Hz policy)
- action_scale: 0.25
- termination_height: 0.1m

## Terrain Curriculum
- Game-inspired: advance level if traveled > 80% expected distance, regress if < 40%
- max_init_terrain_level: 9
