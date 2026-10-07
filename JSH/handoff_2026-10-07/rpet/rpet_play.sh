#!/bin/bash
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac-5.1
cd /mnt/ssd1/jsh/RobotSW_IsaacLab
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export LEG_ASSET_ROOT=$R/assets/usd
export RPET_USD_PATH=$R/assets/usd_51/Leg_51.usd
export PYTHONPATH=$R/env/rsl_rl:/mnt/ssd1/jsh/np126_shim:${PYTHONPATH:-}
./isaaclab.sh -p /mnt/ssd1/jsh/rpet_play_2x.py --orig $R/examples/play_in_isaaclab.py \
  --checkpoint $R/model/checkpoint/model_30000.pt \
  --num_envs 4 --steps 200 --vx 1.0 --headless
echo "PLAY_EXIT=$?"
