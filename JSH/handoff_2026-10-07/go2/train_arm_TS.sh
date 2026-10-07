#!/bin/bash
# 팔 TS = TAMOLS 발판 + 기하경로와 동일한 최근접-돌 스냅.  팔G/팔T 와 그 외 전부 동일.
source /mnt/ssd1/jsh/train_arm_common.inc
export GO2_STEP_TAMOLS_FH=1
export GO2_STEP_TAMOLS_FH_SNAP=1
RUNNAME=stepH_dtc_armTS_tamols_snap
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs 4096 --headless --seed 42 --max_iterations 3000 \
  --run_name $RUNNAME
echo "TRAIN_${RUNNAME}_DONE exit=$?"
