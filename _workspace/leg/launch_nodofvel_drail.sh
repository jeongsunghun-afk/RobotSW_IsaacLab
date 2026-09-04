#!/usr/bin/env bash
# leg 조건부 AMP — DRAIL 판별기 + dof_vel ablation (env.amp_drop_dof_vel=true).
# disc per-step 59 → 42, history 10 → 420, +cond 2 = 422.
# 나머지 override 는 condmatch cmdchg4s baseline 과 동일하게 유지한다.
set -euo pipefail
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate isaac-6.0
cd /home/lgb/IsaacLab-6.0

RUN_NAME=condmatch_drail_nodofvel_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32
LOG=logs/launch/train_${RUN_NAME}_gpu3.log
mkdir -p logs/launch

CUDA_VISIBLE_DEVICES=3 env -u DISPLAY ./isaaclab.sh -p scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Leg-Imitation-Tracking-RMA-v0 --headless \
  env.motion_file=source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/imitation/new_smr_leg_pkl \
  env.motion_weight_mode=command_uniform env.lin_vel_x_max=3.2 env.vel_err_scale=1.5 \
  env.resample_command_in_episode=true env.episode_length_s=20.0 \
  env.tar_change_time_min=4.0 env.tar_change_time_max=4.0 \
  env.amp_cond_mode=speed env.amp_drop_dof_vel=true \
  agent.amp.disc_arch=drail \
  --run_name "${RUN_NAME}" > "${LOG}" 2>&1
