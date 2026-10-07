#!/bin/bash
# 사용: train_arm_generic.sh <G|N> <seed>
# 팔 G = 기하 발판 + TAMOLS 몸통참조 / 팔 N = 기하 발판, 몸통참조 없음.
ARM=$1; SD=$2
source /mnt/ssd1/jsh/train_arm_common.inc
unset GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP GO2_SREF_BASE_ANALYTIC
if [ "$ARM" = "N" ]; then unset GO2_STEP_TAMOLS GO2_STEP_TAMOLS_BASE; SUF=notamols; else SUF=geom; fi
RUNNAME=stepH_dtc_arm${ARM}s${SD}_${SUF}
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs 4096 --headless --seed $SD --max_iterations 3000 \
  --run_name $RUNNAME
echo "TRAIN_${RUNNAME}_DONE exit=$?"
