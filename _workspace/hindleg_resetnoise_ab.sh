#!/usr/bin/env bash
# 리셋 노이즈 fine-tune A/B — 학습 종료를 기다렸다가 프로브 6개를 순서대로 돌린다.
#   ① 처짐 기동 프로브 4 = {baseline, resetnoise} × {real_hip, default}
#   ② 토크 트립 게이트 2 = 두 체크포인트 (정상상태가 안 나빠졌는지 대조)
# 결과는 reports/ 규약대로 metrics/ 에 원문 그대로 남긴다.
set -u
cd /home/lgb/IsaacLab-6.0
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate isaac-6.0

OUT=reports/hindleg_locomotion/hindLeg_history_direct/reset_noise_ab/metrics
DEV=cuda:1
BASE=logs/rsl_rl/hindLeg_history_direct/2026-08-28_09-39-20_gainclamp_ft/model_39799.pt
NEWRUN=logs/rsl_rl/hindLeg_history_direct/2026-09-02_09-07-14_resetnoise_ft
TRAIN_PID="${1:-}"

# ── 학습 종료 대기 ────────────────────────────────────────────────────────────
if [ -n "$TRAIN_PID" ]; then
  echo "[ab] 학습 pid $TRAIN_PID 종료 대기…"
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
  echo "[ab] 학습 종료 감지 $(date -Is)"
fi
sleep 30  # 마지막 체크포인트 flush 여유

NEW=$(ls -1 "$NEWRUN"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
if [ -z "$NEW" ]; then echo "[ab] ✗ 새 체크포인트를 못 찾았다: $NEWRUN"; exit 1; fi
echo "[ab] baseline = $BASE"
echo "[ab] new      = $NEW"

run() {  # run <출력파일> <스크립트> <인자...>
  local f="$OUT/$1"; shift
  echo "[ab] → $f  ($(date +%H:%M:%S))"
  ./isaaclab.sh -p "$@" > "$f" 2>&1
  echo "[ab]   rc=$?  $(grep -c . "$f") 줄"
}

P=_workspace/hindleg_droop_start_probe.py
T=_workspace/hindleg_trip_probe.py
COMMON="--num_envs 512 --steps 1000 --cmd_x 0.3 0.5 1.0 --device $DEV"

run droop_base_realhip.txt  $P --checkpoint "$BASE" --start_pose real_hip $COMMON
run droop_new_realhip.txt   $P --checkpoint "$NEW"  --start_pose real_hip $COMMON
run droop_base_default.txt  $P --checkpoint "$BASE" --start_pose default  $COMMON
run droop_new_default.txt   $P --checkpoint "$NEW"  --start_pose default  $COMMON
run trip_base.txt  $T --checkpoint "$BASE" --trip_nm 25 35 45 --cmd_x 0.3 0.5 1.0 --num_envs 512 --device $DEV
run trip_new.txt   $T --checkpoint "$NEW"  --trip_nm 25 35 45 --cmd_x 0.3 0.5 1.0 --num_envs 512 --device $DEV

echo "[ab] 완료 $(date -Is)  new=$NEW"
touch "$OUT/.ab_done"
