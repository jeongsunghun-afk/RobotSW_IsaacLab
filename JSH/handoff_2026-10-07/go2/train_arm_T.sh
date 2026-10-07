#!/bin/bash
# =============================================================================
# [DTC 2309.15462v2 충실화] stepping 발판 참조원 A/B — 팔 T
#
#   G = 기하(최근접 돌 클램프)   ·   T = TAMOLS stepping 캐시
#   **단일 변수 = GO2_STEP_TAMOLS_FH 한 줄.** 나머지는 train_arm_common.inc 로 공유한다.
#
# 왜 셀 B 가 기준선이 아닌가: 발판 채점을 논문 형태(eps 1e-5 + 사이클당 1회 + 바닥 제거)로
# 바꿨기 때문에 셀 B(구 채점으로 학습됨)와의 비교는 교락된다. 그래서 **두 팔 다 새로 학습**한다.
# 셀 B 는 "채점 수정이 무엇을 바꿨나"의 참고점으로만 쓰고, 교락돼 있음을 리포트에 명시한다.
# =============================================================================
source /mnt/ssd1/jsh/train_arm_common.inc
# ── 팔 T: 발판 참조원 = TAMOLS stepping 캐시.
export GO2_STEP_TAMOLS_FH=1
RUNNAME=stepH_dtc_armT_tamols
ITERS=${ITERS:-3000}
ENVS=${ENVS:-4096}
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs $ENVS --headless --seed 42 --max_iterations $ITERS \
  --run_name $RUNNAME
echo "TRAIN_${RUNNAME}_DONE exit=$?"
