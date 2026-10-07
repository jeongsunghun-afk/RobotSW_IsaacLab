#!/bin/bash
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac6
cd /mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y LEG_ASSET_ROOT=$R/assets/usd
g(){ echo "GCELL $1 L$2 vx$3"
  if [ "$1" = flat ]; then unset RPET_TERRAIN; else export RPET_TERRAIN=$1 RPET_TERRAIN_LEVEL=$2; fi
  python /mnt/ssd1/jsh/rpet_gait.py --checkpoint $R/model/checkpoint/model_30000.pt \
    --num_envs 64 --steps 500 --vx $3 2>&1 | grep -aE 'GAIT_JSON|\[gait\]|Error:|Traceback'; }
# ★검증: 평지에서 문서값 재현되나 (저속 pace 70~71% / 고속 gallop 81~87%)
g flat - 1.0
g flat - 3.0
