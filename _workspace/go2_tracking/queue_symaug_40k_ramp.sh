#!/usr/bin/env bash
# symaug arm 40k **판정** 램프. 학습은 이미 끝났으므로(최종 `model_39999.pt`) 대기 없이 잰다.
#
# ★ 체크포인트 번호가 arm 마다 다르다 — symaug 는 `max_iterations=40000` 으로 정확히 끝나
#   최종이 `model_39999.pt` 이고, 부모는 60k 목표로 돌다 40k 에서 끊겨 `model_40000.pt` 다.
#   1 iter 차이라 비교에는 영향이 없지만 파일 이름은 사실대로 둔다.
#
# ★ 부모 seed0 은 이미 있다(`ramp_cmdlive_uniformw_40000_nopush/seed0`, hold_s 3.0 · ramp_s 1.5 ·
#   64 env · 같은 프로파일로 조건 일치 확인함). **덮어쓰지 않는다** — §30·§33 이 인용하는 값이다.
#   빠진 seed1 만 채운다.
#
# ★ GPU3 은 다른 세션의 parkour 학습이 점유 중이라 제외한다.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

L=logs/rsl_rl/go2_imitation_tracking
M=reports/go2_imitation/_comparisons/mimickit_vs_60_actuator_limit/metrics
SYM=$L/2026-09-04_08-59-04_symaug_uniformw_stock/model_39999.pt
PAR=$L/2026-09-02_09-46-47_cmdlive_uniformw_stock/model_40000.pt
SKIP_GPU=3
LOG=logs/queue_symaug_40k_ramp.log
export CONDA_PREFIX=/home/user/miniconda3/envs/isaac-6.0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
pick_gpu() {
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F', *' -v skip="$SKIP_GPU" '$1 != skip {print $1, $2-$3}' \
      | sort -k2 -nr | head -1 | cut -d' ' -f1
}

for f in "$SYM" "$PAR"; do
    [ -f "$f" ] || { log "중단: 체크포인트 없음 $f"; exit 1; }
done

# (출력디렉토리 : 체크포인트 : 시드)
JOBS=(
    "$M/ramp_symaug_39999_nopush/seed0:$SYM:0"
    "$M/ramp_symaug_39999_nopush/seed1:$SYM:1"
    "$M/ramp_cmdlive_uniformw_40000_nopush/seed1:$PAR:1"
)

for job in "${JOBS[@]}"; do
    OUT=${job%%:*}; rest=${job#*:}; CKPT=${rest%%:*}; S=${rest##*:}
    if [ -f "$OUT/ramp_data.npz" ]; then log "[$(basename "$(dirname "$OUT")")/$(basename "$OUT")] 이미 있음 — 건너뜀"; continue; fi
    g=$(pick_gpu)
    log "[$(basename "$(dirname "$OUT")")/seed$S] 시작 (GPU$g)"
    CUDA_VISIBLE_DEVICES=$g ./isaaclab.sh -p _workspace/go2_tracking/speed_ramp_record.py \
        --checkpoint "$CKPT" --num_envs 64 \
        --no_pace --no_push --no_video --seed "$S" \
        --out_dir "$OUT" >> "$LOG" 2>&1
    if [ -f "$OUT/ramp_data.npz" ]; then log "[$(basename "$(dirname "$OUT")")/seed$S] 완료"
    else log "[$(basename "$(dirname "$OUT")")/seed$S] ★실패"; fi
done

log "ALL_DONE"
