#!/usr/bin/env bash
# push 외란 arm 을 **scratch 가 아니라 fine-tuning** 으로, 그리고 **순차로** 돈다.
#
# 발사 조건 (셋 다 만족해야 한다):
#   ① 부모 체크포인트 `model_40000.pt` 가 있고
#   ② 두 cmdlive 학습이 모두 끝났고 (대조 + 처치)
#   ③ 40k 시드 램프 4 회가 끝났다 (`queue_stop40k_and_ramp.sh` 의 ALL_DONE)
# ③ 까지 기다리는 이유는 램프와 GPU 를 다투지 않게 하기 위해서다. 램프는 20 분이면 끝난다.
# (40k 에서 끊는 근거는 `queue_stop40k_and_ramp.sh` 머리말 참조.)
#
# ★ resume 은 **argparse 플래그로만** 먹는다 — hydra `agent.resume=true` 는 조용히 무시된다
#   (`cli_args.py:76-81` 이 argparse 값을 agent_cfg 에 넣는 유일한 경로다).
#
# ★ `--max_iterations` 는 resume 시 **추가 iter** 다(`on_policy_runner_amp.py:119`
#   `total_it = start_it + num_learning_iterations`). 20000 을 주면 40000 → 60000 이 된다.
#
# ★ lerp anneal 은 이 cfg 에서 **무해하다** — `task_reward_lerp_start` 와 `task_reward_lerp` 가
#   둘 다 0.5 라 스케줄이 항등이다. 둘이 다른 cfg 에 재사용하면 resume 이 anneal 시계를 되감아
#   보상 혼합이 달라진다. 반드시 다시 확인할 것.
#
# ★ pgrep 패턴에 `train.py` 를 붙인 이유: `run_name=...` 만으로 찾으면 **그 문자열을 포함한
#   셸 명령줄까지 매칭된다**(2026-09-03 에 실제로 종료 여부를 오판했다). 학습만 골라야 한다.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

L=logs/rsl_rl/go2_imitation_tracking
PARENT=2026-09-02_09-46-47_cmdlive_stock
CKPT=model_40000.pt
ADD_ITERS=20000
RAMP_LOG=logs/queue_stop40k_and_ramp.log
LOG=logs/queue_push_finetune.log
export CONDA_PREFIX=/home/user/miniconda3/envs/isaac-6.0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

train_alive() {  # $1 = run_name. 학습 프로세스만 센다(셸 자기매칭 방지).
    pgrep -f "train\.py.*run_name=$1\b" >/dev/null
}

pick_gpu() {
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F', *' '{print $1, $2-$3}' | sort -k2 -nr | head -1 | cut -d' ' -f1
}

log "대기 시작 — 두 cmdlive 학습 종료 + 40k 램프 완료 후에 투입한다"
while true; do
    if [ ! -f "$L/$PARENT/$CKPT" ]; then
        if ! train_alive cmdlive_stock; then
            last=$(ls "$L/$PARENT"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
            log "중단: 부모가 40k 전에 종료됨 (최신 model_${last:-없음})"; exit 1
        fi
    elif ! train_alive cmdlive_stock && ! train_alive cmdlive_uniformw_stock; then
        # 램프 큐가 살아 있으면 ALL_DONE 을 기다리고, 죽었으면 더 기다릴 이유가 없다.
        if grep -q "ALL_DONE" "$RAMP_LOG" 2>/dev/null; then
            log "두 학습 종료 + 램프 완료 확인 — 투입한다"; break
        fi
        if ! pgrep -f "bash .*queue_stop40k_and_ramp\.sh" >/dev/null; then
            log "램프 큐가 종료됨(ALL_DONE 없음) — 더 기다리지 않고 투입한다"; break
        fi
    fi
    sleep 120
done

g=$(pick_gpu)
log "GPU$g 에서 push fine-tune 시작 (+${ADD_ITERS} iter → 60000)"
setsid nohup env CUDA_VISIBLE_DEVICES=$g ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless --max_iterations $ADD_ITERS \
    --resume --load_run "$PARENT" --checkpoint "$CKPT" \
    env.use_pace_params=false env.motion_uniform_weights=false env.dr.push_robot=true \
    agent.run_name=push_ft_from_cmdlive \
    > "$L/train_push_ft_from_cmdlive.out" 2>&1 < /dev/null &
log "투입 완료 pid=$! 로그=$L/train_push_ft_from_cmdlive.out"
