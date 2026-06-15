#!/bin/bash
set -x
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate isaac-5.1
echo "PY=$(which python)  CONDA_PREFIX=$CONDA_PREFIX"
cd /home/lgb/IsaacLab
WS=/home/lgb/IsaacLab/_workspace
SD=$WS/symmetry_direction

run_one () {
  local TAG=$1; local CKPT=$2
  echo "=== MEASURE $TAG : $CKPT ==="
  rm -f "$WS/pronk_cost_result.npz" "$WS/pronk_cost_raw.md"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py \
    --task Go2-Parkour-Symmetry --num_envs 256 --num_steps 3000 --headless \
    --checkpoint "$CKPT" 2>&1
  local rc=$?
  echo "=== EXIT $TAG rc=$rc ==="
  if [ -f "$WS/pronk_cost_result.npz" ]; then
    cp "$WS/pronk_cost_result.npz" "$SD/pronk_${TAG}.npz"; echo "renamed npz -> $SD/pronk_${TAG}.npz"
  else echo "NO NPZ for $TAG"; fi
  if [ -f "$WS/pronk_cost_raw.md" ]; then
    cp "$WS/pronk_cost_raw.md" "$SD/pronk_${TAG}.md"; echo "renamed md -> $SD/pronk_${TAG}.md"
  else echo "NO MD for $TAG"; fi
  return $rc
}

run_one "run8_pw1e-3" "logs/rsl_rl/go2_parkour_symmetry/2026-06-12_09-53-06_positive_work_0.001/model_49999.pt"
run_one "run5_pw3e-4" "logs/rsl_rl/go2_parkour_symmetry/2026-06-11_12-13-02_positive_work/model_49999.pt"
echo "=== ALL DONE ==="
