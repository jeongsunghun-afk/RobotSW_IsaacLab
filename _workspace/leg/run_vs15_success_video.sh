#!/usr/bin/env bash
# s=1.5 의 '성공 회차' 영상을 얻는다.
#
# 첫 렌더(vsvid_vs15_1)가 하필 붕괴 회차였다. GT 붕괴율이 1/8 이므로 그 한 회차를 대표로 쓰면
# 7/8 인 실제 거동을 잘못 보여준다. 넘어지지 않은 회차가 나올 때까지 최대 3회 시도한다.
# n=8 보강 배치가 끝난 뒤에 돈다(같은 GPU 를 두고 경합하지 않도록).
set -u
cd /home/lgb/IsaacLab-6.0
PY=/home/user/miniconda3/envs/isaac-6.0/bin/python
D=logs/rsl_rl/leg_imitation_tracking_rma/2026-08-26_16-51-43_velscale15_ds14_wcmd_vmax32
LOG=_workspace/leg/train_logs/velscale_est_more.log

until grep -q "n=8 완료" "$LOG" 2>/dev/null; do sleep 30; done
while pgrep -f "speed_ramp_record_rma.py" >/dev/null; do sleep 20; done
sleep 5

for k in 2 3 4; do
  out=_workspace/leg/vsvid_vs15_$k
  echo "### $(date +%H:%M:%S)  $out"
  $PY -u _workspace/leg/speed_ramp_record_rma.py \
    --checkpoint "$D/model_49999.pt" --run_params "$D/params/env.yaml" \
    --force_stand --heading_hold --cam_view side --out_dir "$out" 2>&1 \
    | grep -E "데이터 저장|Traceback" | head -2
  # 넘어지지 않았으면 채택하고 종료
  if $PY - "$out/ramp_data.npz" <<'EOF'
import sys, numpy as np
d = np.load(sys.argv[1]); names = [str(x) for x in d["joint_names"]]
idx = [i for i, n in enumerate(names) if "thigh" in n or "calf" in n]
folded = float((d["jpos"][-1, idx] < -0.6).mean())
adv = float(np.hypot(d["px"][-1] - d["px"][0], d["py"][-1] - d["py"][0]))
print(f"folded={folded:.2f} adv={adv:.1f}m -> {'FALL' if folded >= 0.25 else 'OK'}")
sys.exit(1 if folded >= 0.25 else 0)
EOF
  then
    echo "===== 성공 회차 확보: $out ====="
    exit 0
  fi
  echo "  (붕괴 — 재시도)"
done
echo "===== 3회 모두 붕괴 — 수동 확인 필요 ====="
