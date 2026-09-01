#!/usr/bin/env bash
# mirror arm 의 '완주 회차' 영상을 다시 뽑는다.
#
# 첫 시도(vsvid_mir_1)는 folded 0.12 라 접힘 기준을 통과했지만 변위가 69.2 m 였다 — 실제로는
# 붕괴다. mirror arm 은 덜 접힌 채로 주저앉아서 접힘 기준이 놓친다. 변위(>=110 m)로 판정한다.
set -u
cd /home/lgb/IsaacLab-6.0
PY=/home/user/miniconda3/envs/isaac-6.0/bin/python
D=logs/rsl_rl/leg_imitation_tracking_rma/2026-08-27_16-23-47_velscale10_ds14_cumirror_vmax32

for k in 2 3 4 5; do
  out=_workspace/leg/vsvid_mir_$k
  echo "### $(date +%H:%M:%S)  $out"
  $PY -u _workspace/leg/speed_ramp_record_rma.py \
    --checkpoint "$D/model_49999.pt" --run_params "$D/params/env.yaml" \
    --force_stand --heading_hold --cam_view side --out_dir "$out" 2>&1 \
    | grep -E "데이터 저장|Traceback" | head -2
  if $PY - "$out/ramp_data.npz" <<'EOF'
import sys, numpy as np
d = np.load(sys.argv[1])
adv = float(np.hypot(d["px"][-1] - d["px"][0], d["py"][-1] - d["py"][0]))
print(f"adv={adv:.1f}m -> {'OK' if adv >= 110.0 else 'FALL'}")
sys.exit(0 if adv >= 110.0 else 1)
EOF
  then echo "===== 완주 회차 확보: $out ====="; exit 0; fi
  echo "  (붕괴 — 재시도)"
done
echo "===== 4회 모두 붕괴 — 수동 확인 필요 ====="
