#!/bin/bash
# 팔 S' = 발판 기하 스냅(팔G 동일) + 몸통참조를 **캐시 위상평균**으로.
# 팔G 와의 단일변수 = 몸통계획의 **지형 조건화** 유무.
source /mnt/ssd1/jsh/train_arm_common.inc
unset GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP
export GO2_SREF_BASE_ANALYTIC=1
RUNNAME=stepH_dtc_armS2_cyclemean
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs 4096 --headless --seed 42 --max_iterations 3000 \
  --run_name $RUNNAME
echo "TRAIN_${RUNNAME}_DONE exit=$?"
