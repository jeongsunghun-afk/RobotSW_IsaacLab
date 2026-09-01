#!/usr/bin/env bash
# velscale 50k 판정 보강 — estimator(배포) 조건 재측정 + 대표 롤아웃 영상.
#
# 다른 세션이 GT(use_estimator=False) 조건으로 3 arm × 8회를 이미 쟀다
# (`_workspace/leg/{base,vs10,vs15}_it49999_*`). 여기서는 그 램프를 재실행하지 않고
# 빠진 두 가지만 채운다.
#   1) --use_estimator : 실기는 base 속도를 못 재므로 배포 조건. cmd 0.5 는 GT 대비 최대 -24pp 벌어진다.
#   2) 영상            : 기존 램프는 --no_video 라 mp4 가 없다. GT 조건으로 렌더해 판정 표와 짝을 맞춘다.
#
# 플래그는 기존 GT 램프의 npz 메타데이터와 동일하게 맞췄다
# (hold 3.0 / ramp 2.0 / vx_max 4.0 / heading_hold / DR·push ON / force_stand / run_params).
set -u
cd /home/lgb/IsaacLab-6.0
PY=/home/user/miniconda3/envs/isaac-6.0/bin/python
R=logs/rsl_rl/leg_imitation_tracking_rma

declare -A RUN=(
  [base]="$R/2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd"
  [vs10]="$R/2026-08-26_17-19-51_velscale10_ds14_wcmd_vmax32"
  [vs15]="$R/2026-08-26_16-51-43_velscale15_ds14_wcmd_vmax32"
)

ramp () {  # $1=arm  $2=out_dir  $3...=추가 플래그
  local arm=$1 out=$2; shift 2
  local d=${RUN[$arm]}
  echo "### $(date +%H:%M:%S)  $out"
  $PY -u _workspace/leg/speed_ramp_record_rma.py \
    --checkpoint "$d/model_49999.pt" \
    --run_params "$d/params/env.yaml" \
    --force_stand --heading_hold \
    --out_dir "_workspace/leg/$out" "$@" 2>&1 | grep -E "저장|평균|Traceback|Error" | head -5
}

echo "===== 1/2  estimator(배포) 조건 — 3 arm x 4회 ====="
for i in 1 2 3 4; do
  for arm in base vs10 vs15; do
    ramp "$arm" "vsest_${arm}_${i}" --use_estimator --no_video
  done
done

echo "===== 2/2  영상 (GT 조건 — 판정 표와 동일) ====="
# vs10 은 GT 에서 8회 중 4회가 cmd 4.0 에서 넘어졌으므로 2회 렌더해 붕괴를 잡을 확률을 높인다.
ramp base vsvid_base_1 --cam_view side
ramp vs15 vsvid_vs15_1 --cam_view side
ramp vs10 vsvid_vs10_1 --cam_view side
ramp vs10 vsvid_vs10_2 --cam_view side

echo "===== 완료 $(date +%H:%M:%S) ====="
