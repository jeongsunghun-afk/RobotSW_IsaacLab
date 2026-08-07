#!/usr/bin/env bash
# Render per-terrain gait videos for BOTH policies of the GT-voxel vs SL-Grid+crawl comparison.
#
# Each policy gets its own task + sensor overlay (the overlay shows that policy's ACTUAL input):
#   GT voxel teacher   -> Teacher3DVoxelGT task, --sensor voxel      (privileged, not deployable)
#   SL-Grid + crawl    -> Lidar-SL-Grid-Crawl task, --sensor lidar_grid (deployable Mid-360)
#
# NOTE: play_per_terrain.py builds the env from the CURRENT source cfg, not from each run's
# params/env.yaml. The source cfg now has num_crawls=8, so the GT voxel teacher (trained on
# crawl-3) is rendered OUT OF DISTRIBUTION on crawl. The other five terrains are unaffected.
set -u

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate isaac-6.0

cd /home/lgb/IsaacLab-6.0

OUT_DIR="${1:-/home/lgb/IsaacLab-6.0/_workspace/gt_vs_slgrid_crawl/videos}"
GPU="${2:-0}"
LEN="${3:-400}"

GT_CKPT=/home/lgb/IsaacLab-6.0/logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k/model_49999.pt
GT_TASK=Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0
GT_SENSOR=voxel

SL_CKPT=/home/lgb/IsaacLab-6.0/logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl/2026-08-05_12-57-11_slgrid_crawl_scratch_50k/model_49999.pt
SL_TASK=Go2-ParkourImitation-Lidar-SL-Grid-Crawl-EasyEntry-v0
SL_SENSOR=lidar_grid

mkdir -p "$OUT_DIR"

render() {  # render <tag> <task> <ckpt> <sensor> <terrain>
  local tag="$1" task="$2" ckpt="$3" sensor="$4" ter="$5"
  local out="$OUT_DIR/${ter}_${tag}.mp4"
  echo "=== [$tag/$ter] start $(date +%H:%M:%S) ==="
  env -u DISPLAY CUDA_VISIBLE_DEVICES="$GPU" ./isaaclab.sh -p scripts/demos/play_per_terrain.py \
    --task "$task" --num_envs 12 --headless \
    --checkpoint "$ckpt" \
    --terrain "$ter" --sensor "$sensor" --video_length "$LEN" \
    --out_path "$out" \
    > "$OUT_DIR/log_${ter}_${tag}.log" 2>&1
  local rc=$?
  if [ -f "$out" ]; then
    echo "=== [$tag/$ter] OK rc=$rc size=$(stat -c%s "$out") ==="
  else
    echo "=== [$tag/$ter] FAILED rc=$rc (no mp4) ==="
  fi
}

for TER in flat hurdle step gap stair crawl; do
  render gtvoxel "$GT_TASK" "$GT_CKPT" "$GT_SENSOR" "$TER"
  render slgrid  "$SL_TASK" "$SL_CKPT" "$SL_SENSOR" "$TER"
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
