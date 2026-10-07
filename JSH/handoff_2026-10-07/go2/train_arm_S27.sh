#!/bin/bash
# 학습 seed 재현 확인 (seed 7).  헤드라인(지형 조건화 유의미)이 단일 학습seed 산물인지 검증.
source /mnt/ssd1/jsh/train_arm_common.inc
unset GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP
export GO2_SREF_BASE_ANALYTIC=1
RUNNAME=stepH_dtc_armS27_cyclemean
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs 4096 --headless --seed 7 --max_iterations 3000 \
  --run_name $RUNNAME
echo "TRAIN_${RUNNAME}_DONE exit=$?"
