#!/usr/bin/env bash
# symaug arm 이 20k 에 닿으면 **학습은 끊지 않고** 시드 고정 × 2 회 램프를 잰다.
#
# 왜 20k 인가: 판정 대상인 좌우 비대칭은 **고속 직진(cmd 3.5/4.0)에서만** 나타난다
#   (선회 명령에서는 0.1~3.5 % 로 사라진다 — README §30). 그 구간이 성립하지 않는
#   6~11k 에서 재면 현상이 없는 자리를 재는 것이다. 20k 가 고속 직진이 서는 첫 지점이다.
#
# 왜 끊지 않는가: 이 arm 의 판정점은 여전히 40k 다(§24). 20k 는 중간 확인이며,
#   여기서 끊으면 40k 비교 자체가 사라진다.
#
# 왜 `--no_push --no_pace` 인가: 학습 cfg 와 램프 `dr.*` 를 맞추는 것이 규약이다(§26).
#   두 arm 모두 `push_robot=false` · `use_pace_params=false` 로 학습했다. 맞추지 않으면
#   학습에 없던 외란을 재게 되어 낙상이 3~5 % → 27~44 % 로 부풀었던 사고가 재현된다.
#
# 왜 시드 × 2 인가: 무시드 램프는 같은 체크포인트에서 `cmd 4.0` median 이 1.84 m/s 진동한다.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

L=logs/rsl_rl/go2_imitation_tracking
RUN_SYM=$L/2026-09-04_08-59-04_symaug_uniformw_stock       # symmetry data-aug ON
RUN_PAR=$L/2026-09-02_09-46-47_cmdlive_uniformw_stock      # 부모: 같은 조건, symmetry 없음
M=reports/go2_imitation/_comparisons/mimickit_vs_60_actuator_limit/metrics
IT=20000
SEEDS=(0 1)
TRAIN_GPU=3          # symaug 학습이 점유 중 — 램프는 여기를 피한다
LOG=logs/queue_symaug_20k_ramp.log
export CONDA_PREFIX=/home/user/miniconda3/envs/isaac-6.0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

train_alive() {  # ★ `run_name=...` 만으로 찾으면 그 문자열을 담은 셸까지 매칭된다. train.py 로 좁힌다.
    pgrep -f "train\.py.*run_name=$1\b" >/dev/null
}

pick_gpu() {  # 학습 GPU 를 뺀 나머지 중 여유 메모리 최대
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F', *' -v skip="$TRAIN_GPU" '$1 != skip {print $1, $2-$3}' \
      | sort -k2 -nr | head -1 | cut -d' ' -f1
}

log "대기 시작 — symaug 가 model_${IT}.pt 를 낼 때까지 (현재 $(ls $RUN_SYM/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1))"
while [ ! -f "$RUN_SYM/model_${IT}.pt" ]; do
    if ! train_alive symaug_uniformw_stock; then
        log "중단: symaug_uniformw_stock 이 ${IT} 전에 종료됨"; exit 1
    fi
    sleep 120
done
sleep 30   # 체크포인트 쓰기가 끝나도록 한 박자 — 부분 기록 파일을 재는 사고 방지

log "symaug ${IT} 도달. 학습은 계속 둔다(목표 40k)."
[ -f "$RUN_PAR/model_${IT}.pt" ] || { log "중단: 부모 model_${IT}.pt 없음"; exit 1; }

for spec in "symaug:$RUN_SYM" "cmdlive_uniformw:$RUN_PAR"; do
    n=${spec%%:*}; d=${spec#*:}
    for s in "${SEEDS[@]}"; do
        OUT=$M/ramp_${n}_${IT}_nopush/seed${s}
        g=$(pick_gpu)
        log "[$n seed $s] 시작 (GPU$g)"
        CUDA_VISIBLE_DEVICES=$g ./isaaclab.sh -p _workspace/go2_tracking/speed_ramp_record.py \
            --checkpoint "$d/model_${IT}.pt" --num_envs 64 \
            --no_pace --no_push --no_video --seed "$s" \
            --out_dir "$OUT" >> "$LOG" 2>&1
        if [ -f "$OUT/ramp_data.npz" ]; then log "[$n seed $s] 완료"; else log "[$n seed $s] ★실패"; fi
    done
done

log "ALL_DONE — npz: $M/ramp_{symaug,cmdlive_uniformw}_${IT}_nopush/seed{0,1}/"
