#!/bin/bash
# 험지 걸음 분류 — ★vx 1.0 만. 고속은 분류기 검증 실패(2s 창 FFT 분해능 0.5Hz 한계)라 쓰지 않는다.
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac6
cd /mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y LEG_ASSET_ROOT=$R/assets/usd
g(){ echo "GCELL $1 L$2"
  if [ "$1" = flat ]; then unset RPET_TERRAIN; else export RPET_TERRAIN=$1 RPET_TERRAIN_LEVEL=$2; fi
  python /mnt/ssd1/jsh/rpet_gait.py --checkpoint $R/model/checkpoint/model_30000.pt \
    --num_envs 64 --steps 500 --vx 1.0 2>&1 | grep -aE 'GAIT_JSON|Error:|Traceback'; }
g flat -
for K in rough gap stepping; do for L in 2 5 8; do g $K $L; done; done
echo "GAIT_TERRAIN_DONE $(date '+%F %T')"
