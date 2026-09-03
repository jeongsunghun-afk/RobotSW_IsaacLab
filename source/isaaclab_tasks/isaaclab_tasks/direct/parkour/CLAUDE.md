# Go2 Parkour Environment

## Overview
Go2 robot parkour locomotion over Genesis-style procedural terrain (flat/hurdle/step/gap/stair).
Task ID·러너 변종·파일 구조는 `README.md` 참조.

## Key Files
- `parkour_env_cfg.py`: `ParkourEnvCfg` (env/terrain/reward/sensor 설정)
- `parkour_env.py`: `Go2ParkourEnv`
- `parkour_lidar_env_cfg.py`/`parkour_lidar_env.py`: `ParkourLidarEnvCfg`/`Go2ParkourLidarEnv` (Task ID `Go2-Parkour-Lidar-v0`, `ParkourEnvCfg` 상속, 별도 3D lidar 관측)
- `mdp/symmetry.py`: L/R 대칭 데이터 증강
- `agents/rsl_rl_ppo_cfg.py`: `Go2ParkourPPORunnerCfg`

## Observations (`parkour_env_cfg.py:459-473`, dict 기반, deprecated 스칼라 필드 아님)
```
policy (proprio):        num_proprio = 42 + 4 = 46   (주석 기준 구성 3+1+1+1+12+12+12=42, +4는 미확인)
scan (height_scan):      187                          (num_scan_obs)
priv_explicit:           lin_vel_b(3)+ang_vel_b(3) = 6
priv_latent:             base_friction+foot_friction+base_mass+base_com+joint_stiffness/damping_ratio = 37
history:                 history_len(10) * num_proprio(46) = 460  (코드 주석은 10*42=420으로 되어 있어 코드 내 주석-필드값 불일치, 실제 필드 기준 460)
```
`obs_groups`(agents/rsl_rl_ppo_cfg.py:49-54): `policy=["policy"]`, `critic=["policy","scan","priv_explicit","priv_latent"]`, `scan=["scan"]`, `history=["history"]`, `priv=["priv_latent"]`. `ActorCriticRMA`가 이 dict를 소비함. `observation_space` 정수 필드(=46)는 하위호환용이며 실제 obs는 dict.

## Terrain (`PARKOUR_TERRAINS_CFG`, `parkour_env_cfg.py`)
- `num_rows=11`, `num_cols=40`, `size=(25.0,4.0)`, `curriculum=True`
- 활성 sub-terrain(각 proportion=0.2, 합 1.0): `parkour_flat`, `parkour_hurdle`, `parkour_step`, `parkour_gap`, `parkour_stair`
- 비활성(proportion=0.0, 코드에 등록만 됨): `parkour_stepping_stones`, `parkour_balance_beam`, `parkour_crawl`, `parkour_slope`, `parkour_zigzag_hurdles`, `parkour_rough_blocks`
- 코드 주석엔 Genesis 기준 비율(flat 0.1/hurdle 0.2/step 0.2/gap 0.3/stair 0.2)이 적혀 있으나 실제 할당값은 5종 모두 0.2로 균등함

## Env Specs (Go2)
- DOF: 12 (action_space = 12)
- episode_length_s: 20.0, decimation: 4 (200Hz physics / 50Hz policy)
- action_scale: 0.25
- termination_height: -0.2 (기준 대비 상대값, 절대 0.1m 아님)
- max_tilt: 1.309 rad

## Terrain Curriculum
- `move_up`: 이동거리 > 0.8 × expected_dist, `move_down`: < 0.4 × expected_dist (`parkour_env.py:1859-1860`)
- `max_init_terrain_level: 3`

## Algorithm
- PPO, actor/critic `[512, 256, 128]`, `learning_rate=2.0e-4` (`agents/rsl_rl_ppo_cfg.py`)

## 학습·평가
학습·렌더 실행 방법은 `.claude/rules/training.md` 참조.
