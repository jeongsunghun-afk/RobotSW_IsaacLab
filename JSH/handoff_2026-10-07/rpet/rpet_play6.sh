#!/bin/bash
# R.pet 원본 play — isaac-6.0 라인(IsaacLab 3.0.0 / isaacsim 6.0.1.0 / py3.12). 어댑터 없음.
W=/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac6
cd $W
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export LEG_ASSET_ROOT=$R/assets/usd
python $R/examples/play_in_isaaclab.py \
  --checkpoint $R/model/checkpoint/model_30000.pt \
  --num_envs 4 --steps 200 --vx 1.0 --headless
echo "PLAY6_EXIT=$?"
