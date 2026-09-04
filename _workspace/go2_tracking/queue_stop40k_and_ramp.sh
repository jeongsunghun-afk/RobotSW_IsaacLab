#!/usr/bin/env bash
# 두 cmdlive run 이 40k 에 닿으면 **학습을 끊고** 시드 고정 × 2 회 램프를 재고 요약표까지 낸다.
#
# 왜 40k 에서 끊는가 (2026-09-03 실측 근거):
#  · 잘 수렴한 arm(lerp06/08/08_pace)은 40k → 60k 이득이 **+0.01~0.03 m/s** 뿐이다.
#  · 실패 arm들은 오르지 않고 **비단조로 배회**한다(48k 정점 후 하락). 그 폭이 램프 자체의
#    재현 산포와 같은 크기라, "40k 이후 이득"의 증거가 사실상 없다.
#  · 현재 두 run 의 `Episode_Reward/lin_vel_reward` 는 이미 평평하다(대조 −1.08/+0.63,
#    처치 +0.66/+0.15 — 5k 구간 평균 기준).
#  · 더 돌리는 게 공짜도 아니다 — 과거 `newpace` run 은 52k 에서 붕괴했고 학습지표 넷 다 놓쳤다.
#
# 왜 시드 × 2 인가: 램프가 실행마다 달라진다(시드 부재 시 `cmd 4.0` median 이 1.84 m/s 진폭).
# 시드를 고정하면 재현은 되지만 한 시드는 표본 하나라 arm 차이를 산포와 견줄 수 없다.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

L=logs/rsl_rl/go2_imitation_tracking
RUN_CTL=$L/2026-09-02_09-46-47_cmdlive_stock
RUN_TRT=$L/2026-09-02_09-46-47_cmdlive_uniformw_stock
M=reports/go2_imitation/_comparisons/mimickit_vs_60_actuator_limit/metrics
IT=40000
SEEDS=(0 1)
LOG=logs/queue_stop40k_and_ramp.log
export CONDA_PREFIX=/home/user/miniconda3/envs/isaac-6.0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

train_alive() {  # $1 = run_name. 학습 프로세스만 센다.
    # ★ `run_name=...` 만으로 찾으면 **그 문자열을 담은 셸 명령줄까지 매칭된다**
    #   (2026-09-03 에 실제로 종료 여부를 두 번 오판했다). `train.py` 로 좁힌다.
    pgrep -f "train\.py.*run_name=$1\b" >/dev/null
}

train_kill() {  # $1 = run_name, $2 = 시그널(기본 TERM)
    pkill -${2:-TERM} -f "train\.py.*run_name=$1\b"
}

pick_gpu() {
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F', *' '{print $1, $2-$3}' | sort -k2 -nr | head -1 | cut -d' ' -f1
}

log "대기 시작 — 두 run 이 model_${IT}.pt 를 낼 때까지"
while true; do
    [ -f "$RUN_CTL/model_${IT}.pt" ] && [ -f "$RUN_TRT/model_${IT}.pt" ] && break
    if ! train_alive cmdlive_stock && [ ! -f "$RUN_CTL/model_${IT}.pt" ]; then
        log "중단: cmdlive_stock 이 ${IT} 전에 종료됨"; exit 1
    fi
    if ! train_alive cmdlive_uniformw_stock && [ ! -f "$RUN_TRT/model_${IT}.pt" ]; then
        log "중단: cmdlive_uniformw_stock 이 ${IT} 전에 종료됨"; exit 1
    fi
    sleep 60
done

# 체크포인트 쓰기가 끝나도록 한 박자 둔다(부분 기록 파일을 재는 사고 방지).
sleep 30
log "두 run 모두 ${IT} 도달 — 학습 종료"
train_kill cmdlive_stock
train_kill cmdlive_uniformw_stock
sleep 20
for n in cmdlive_stock cmdlive_uniformw_stock; do
    if train_alive "$n"; then
        train_kill "$n" KILL; log "$n 강제 종료"
    fi
done
for d in "$RUN_CTL" "$RUN_TRT"; do
    log "$(basename "$d") 최종 model_$(ls "$d"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)"
done

for spec in "cmdlive:$RUN_CTL" "cmdlive_uniformw:$RUN_TRT"; do
    n=${spec%%:*}; d=${spec#*:}
    for s in "${SEEDS[@]}"; do
        OUT=$M/ramp_${n}_${IT}/seed${s}
        g=$(pick_gpu)
        log "[$n seed $s] 시작 (GPU$g)"
        CUDA_VISIBLE_DEVICES=$g ./isaaclab.sh -p _workspace/go2_tracking/speed_ramp_record.py \
            --checkpoint "$d/model_${IT}.pt" --num_envs 64 --no_pace --seed "$s" --no_video \
            --out_dir "$OUT" >> "$LOG" 2>&1
        if [ -f "$OUT/ramp_data.npz" ]; then log "[$n seed $s] 완료"; else log "[$n seed $s] ★실패"; fi
    done
done

log "요약표 생성"
./isaaclab.sh -p reports/go2_imitation/_comparisons/mimickit_vs_60_actuator_limit/logs/summarize_seeded_ramps.py \
    "$IT" "${SEEDS[@]}" >> "$LOG" 2>&1
log "ALL_DONE — 요약: $M/seeded_ramps_${IT}.md"
