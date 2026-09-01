#!/usr/bin/env bash
# command_uniform_mirror arm(50k 완주)을 기존 3 arm 과 같은 조건으로 잰다.
#
# 플래그는 base/vs10/vs15 램프와 동일해야 같은 표에 넣을 수 있다
# (hold 3.0 / ramp 2.0 / vx_max 4.0 / heading_hold / DR·push ON / force_stand / run_params).
# GT n=8 + estimator n=8 + 영상 1편.
set -u
cd /home/lgb/IsaacLab-6.0
PY=/home/user/miniconda3/envs/isaac-6.0/bin/python
D=logs/rsl_rl/leg_imitation_tracking_rma/2026-08-27_16-23-47_velscale10_ds14_cumirror_vmax32

ramp () {  # $1=out_dir  $2...=추가 플래그
  local out=$1; shift
  echo "### $(date +%H:%M:%S)  $out"
  $PY -u _workspace/leg/speed_ramp_record_rma.py \
    --checkpoint "$D/model_49999.pt" --run_params "$D/params/env.yaml" \
    --force_stand --heading_hold \
    --out_dir "_workspace/leg/$out" "$@" 2>&1 | grep -E "데이터 저장|Traceback|Error" | head -3
}

echo "===== 1/3  GT n=8 ====="
for i in 1 2 3 4 5 6 7 8; do ramp "mir_it49999_${i}" --no_video; done

echo "===== 2/3  estimator n=8 ====="
for i in 1 2 3 4 5 6 7 8; do ramp "vsest_mir_${i}" --use_estimator --no_video; done

echo "===== 3/3  영상 (GT, 넘어지지 않은 회차가 나올 때까지 최대 3회) ====="
for k in 1 2 3; do
  ramp "vsvid_mir_${k}" --cam_view side
  if $PY - "_workspace/leg/vsvid_mir_${k}/ramp_data.npz" <<'EOF'
import sys, numpy as np
d = np.load(sys.argv[1]); names = [str(x) for x in d["joint_names"]]
idx = [i for i, n in enumerate(names) if "thigh" in n or "calf" in n]
folded = float((d["jpos"][-1, idx] < -0.6).mean())
adv = float(np.hypot(d["px"][-1] - d["px"][0], d["py"][-1] - d["py"][0]))
print(f"folded={folded:.2f} adv={adv:.1f}m -> {'FALL' if folded >= 0.25 else 'OK'}")
sys.exit(1 if folded >= 0.25 else 0)
EOF
  then echo "===== 성공 회차: vsvid_mir_${k} ====="; break; fi
  echo "  (붕괴 — 재시도)"
done
echo "===== 완료 $(date +%H:%M:%S) ====="
