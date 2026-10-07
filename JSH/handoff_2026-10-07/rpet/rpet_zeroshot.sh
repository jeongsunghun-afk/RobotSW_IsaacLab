#!/bin/bash
# R.pet 험지 zero-shot — 재학습 0. 인계 체크포인트를 그대로 험지에 올린다.
# 세 갈래 동시 판정: ①게이트가 험지에서 유지되나 ②지각 없이 완만 험지를 통과하나 ③징검돌에서 낙상하나
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac6
cd /mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y LEG_ASSET_ROOT=$R/assets/usd
for KIND in rough gap stepping; do
  for LVL in 2 5 8; do
    for VX in 0.5 1.0 2.0; do
      echo "########## $KIND L$LVL vx$VX"
      RPET_TERRAIN=$KIND RPET_TERRAIN_LEVEL=$LVL python $R/examples/play_in_isaaclab.py \
        --checkpoint $R/model/checkpoint/model_30000.pt \
        --num_envs 16 --steps 300 --vx $VX 2>&1 \
        | grep -aE 'RPET-TERRAIN|요약|vx \[m/s\]|vy \[m/s\]|추종 오차|추정 오차|termination|Error|Traceback'
    done
  done
done
echo "ZEROSHOT_DONE $(date '+%F %T')"
