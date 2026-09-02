#!/usr/bin/env bash
# 처짐 기동 프로브 재실행 (inference_mode 버그 수정 후). 트립 게이트 2건은 1차에서 완주했으므로 제외.
set -u
cd /home/lgb/IsaacLab-6.0
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate isaac-6.0
OUT=reports/hindleg_locomotion/hindLeg_history_direct/reset_noise_ab/metrics
DEV=cuda:1
BASE=logs/rsl_rl/hindLeg_history_direct/2026-08-28_09-39-20_gainclamp_ft/model_39799.pt
NEW=logs/rsl_rl/hindLeg_history_direct/2026-09-02_09-07-14_resetnoise_ft/model_42000.pt
P=_workspace/hindleg_droop_start_probe.py
COMMON="--num_envs 512 --steps 1000 --cmd_x 0.3 0.5 1.0 --device $DEV"
run() { local f="$OUT/$1"; shift; echo "[ab2] → $f ($(date +%H:%M:%S))"; ./isaaclab.sh -p "$@" > "$f" 2>&1; echo "[ab2]   rc=$? · cmd 블록 $(grep -c 'cmd vx =' "$f")"; }
run droop_base_realhip.txt  $P --checkpoint "$BASE" --start_pose real_hip $COMMON
run droop_new_realhip.txt   $P --checkpoint "$NEW"  --start_pose real_hip $COMMON
run droop_base_default.txt  $P --checkpoint "$BASE" --start_pose default  $COMMON
run droop_new_default.txt   $P --checkpoint "$NEW"  --start_pose default  $COMMON
echo "[ab2] 완료 $(date -Is)"
touch "$OUT/.ab2_done"
