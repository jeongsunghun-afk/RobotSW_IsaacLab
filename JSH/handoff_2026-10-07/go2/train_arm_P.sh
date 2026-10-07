#!/bin/bash
# 팔 P = 강한 채널 상한 탐침.  GO2_STEP_TAMOLS_BASE_POLICY=1
#   = 몸통참조를 **정책 관측 +6 AND 보상**(base_ref_track)으로 준다.
# 팔 N(참조 없음)과의 단일변수 = 몸통참조의 **최대 가용 신호**.
#
# ★규정: 배포 후보가 아니라 상한 탐침이다. 정책 관측이므로 배포 시에도 TAMOLS 참조가 필요하고,
#   온라인 TO 는 정밀지형 4.5초/회로 실시간 불가다(구조화 지형 캐시 조회만 가능).
#   여기서 신호가 없으면 약한 채널(critic 전용)은 볼 필요가 없다 = 싼 기각 경로.
# ★scale/σ 는 기본값(0.0125 / 0.05)을 쓴다 — 코드 주석의 실현-sum 균형값이고
#   D13(양버킷 비중 왜곡) 회피용으로 이미 산출돼 있다. 건드리지 않는다.
ARM=P; SD=$1
source /mnt/ssd1/jsh/train_arm_common.inc
unset GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP GO2_SREF_BASE_ANALYTIC
export GO2_STEP_TAMOLS_BASE_POLICY=1
RUNNAME=stepH_dtc_armPs${SD}_basepolicy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs 4096 --headless --seed $SD --max_iterations 3000 \
  --run_name $RUNNAME
echo "TRAIN_${RUNNAME}_DONE exit=$?"
