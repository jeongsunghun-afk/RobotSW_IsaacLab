#!/usr/bin/env bash
# estimator 조건 표본을 GT 와 같은 n=8 로 맞춘다.
#
# n=4 에서 붕괴가 baseline 0/4 · s=1.0 4/4 · s=1.5 3/4 로 나왔다. GT(0/8 · 4/8 · 1/8) 대비
# s=1.5 의 붕괴율이 12.5% → 75% 로 뛴 것이 판정을 뒤집는 수치인데, 4개로는 우연과 구분이 안 된다.
# 앞선 실행이 GPU 를 놓은 뒤 이어서 돈다.
set -u
cd /home/lgb/IsaacLab-6.0
PY=/home/user/miniconda3/envs/isaac-6.0/bin/python
R=logs/rsl_rl/leg_imitation_tracking_rma

while pgrep -f "speed_ramp_record_rma.py" >/dev/null; do sleep 30; done
sleep 5

declare -A RUN=(
  [base]="$R/2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd"
  [vs10]="$R/2026-08-26_17-19-51_velscale10_ds14_wcmd_vmax32"
  [vs15]="$R/2026-08-26_16-51-43_velscale15_ds14_wcmd_vmax32"
)

for i in 5 6 7 8; do
  for arm in base vs10 vs15; do
    d=${RUN[$arm]}
    echo "### $(date +%H:%M:%S)  vsest_${arm}_${i}"
    $PY -u _workspace/leg/speed_ramp_record_rma.py \
      --checkpoint "$d/model_49999.pt" \
      --run_params "$d/params/env.yaml" \
      --force_stand --heading_hold --use_estimator --no_video \
      --out_dir "_workspace/leg/vsest_${arm}_${i}" 2>&1 | grep -E "데이터 저장|Traceback|Error" | head -3
  done
done
echo "===== n=8 완료 $(date +%H:%M:%S) ====="
