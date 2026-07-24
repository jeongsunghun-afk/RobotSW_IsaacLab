#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-BipedLeg policy_runner 런처 (터미널 2, conda torch — Isaac 앱 없음).
#   - 학습된 hind_leg history 정책을 로드해 UDP 폐루프로 구동, action을 sim + real 양쪽에 전송.
#
# 사용:  bash scripts/real2sim/r2s_biped_leg/run_policy_runner.sh [추가 인자]
# 조정(환경변수):
#   SIM_ENV=isaac-6.0 GPU=0
#   RUN_DIR=<학습 run 디렉토리>  CKPT=model_23100.pt
#   REAL_HOST=192.168.x.y        # real 엔드포인트가 있으면 지정 (없으면 real fan-out no-op)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-0}"
RUN_DIR="${RUN_DIR:-logs/rsl_rl/hindLeg_history_direct/2026-07-22_18-05-50_history_60_baseline}"
CKPT="${CKPT:-model_23100.pt}"
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

# conda 활성화 (isaac-6.0). policy_runner 는 torch+rsl_rl 만 쓰고 SimulationApp 은 띄우지 않는다.
# shellcheck disable=SC1090
source "$CONDA_HOOK"
conda activate "$SIM_ENV"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"

_args=(--run_dir "$RUN_DIR" --checkpoint "$CKPT")
[ -n "$REAL_HOST" ] && _args+=(--real_host "$REAL_HOST")

echo "[run_policy_runner_bl] env=$SIM_ENV python=$(command -v python) GPU=$GPU"
echo "[run_policy_runner_bl] run_dir=$RUN_DIR ckpt=$CKPT real_host=${REAL_HOST:-<none>}"
echo "[run_policy_runner_bl] args: ${_args[*]} $*"

exec python scripts/real2sim/policy_runner_bipedleg.py "${_args[@]}" "$@"
