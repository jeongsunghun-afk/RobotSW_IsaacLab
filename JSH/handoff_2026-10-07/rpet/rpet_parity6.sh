#!/bin/bash
# standalone vs IsaacLab env 대조 — 이식/환경 정합의 수치 판정.
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
W=/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac6
cd $W
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export LEG_ASSET_ROOT=$R/assets/usd
python $R/standalone/test_parity.py --with_env --num_envs 4 --steps 64 \
  --task Leg-Imitation-Tracking-RMA-v0
echo "PARITY_EXIT=$?"
