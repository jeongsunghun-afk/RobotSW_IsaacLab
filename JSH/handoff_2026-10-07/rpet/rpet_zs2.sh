#!/bin/bash
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac6
cd /mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y LEG_ASSET_ROOT=$R/assets/usd
run(){ echo "CELL $1 L$2 vx$3"
  RPET_TERRAIN=$1 RPET_TERRAIN_LEVEL=$2 python /mnt/ssd1/jsh/rpet_zeroshot2.py \
    --checkpoint $R/model/checkpoint/model_30000.pt --num_envs 16 --steps 400 --vx $3 2>&1 \
    | grep -aE 'RESULT_JSON|RPET-TERRAIN|Error|Traceback'; }
# 기준선(평지) 먼저
echo 'CELL flat L- vx1.0'
python /mnt/ssd1/jsh/rpet_zeroshot2.py --checkpoint $R/model/checkpoint/model_30000.pt \
  --num_envs 16 --steps 400 --vx 1.0 2>&1 | grep -aE 'RESULT_JSON|Error|Traceback'
for K in rough gap stepping; do for L in 2 5 8; do run $K $L 1.0; done; done
echo "ZS2_DONE $(date '+%F %T')"
