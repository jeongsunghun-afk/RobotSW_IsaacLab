#!/bin/bash
# 팔 N = TAMOLS 완전 제거.  발판=기하 스냅(팔G와 동일) · base 참조=없음.
# 팔G 와의 단일변수 = **TAMOLS 몸통참조(critic priv +6)** 하나.
# 이게 팔G에 못 미치면 base 참조가 기여하고 있다는 뜻이고, 동률이면 TAMOLS 기여가 0이다.
source /mnt/ssd1/jsh/train_arm_common.inc
unset GO2_STEP_TAMOLS GO2_STEP_TAMOLS_BASE GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP
RUNNAME=stepH_dtc_armN_notamols
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs 4096 --headless --seed 42 --max_iterations 3000 \
  --run_name $RUNNAME
echo "TRAIN_${RUNNAME}_DONE exit=$?"
