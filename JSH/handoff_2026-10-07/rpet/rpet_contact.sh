#!/bin/bash
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac6
cd /mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y LEG_ASSET_ROOT=$R/assets/usd
python /mnt/ssd1/jsh/rpet_contact_check.py --num_envs 4 --steps 120 --headless
echo "CONTACT_EXIT=$?"
