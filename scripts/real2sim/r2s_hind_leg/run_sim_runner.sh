#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-HindLeg sim_runner 원커맨드 런처 (터미널 1, Isaac conda).
#   - isaac-6.0 conda env를 활성화한 뒤 sim_runner_hindleg 실행.
#
# 사용:  bash scripts/real2sim/r2s_hind_leg/run_sim_runner.sh [추가 인자]
# 조정(환경변수): SIM_ENV=isaac-6.0 GPU=2 LIVESTREAM=2 VIZ=kit

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-2}"
LIVESTREAM="${LIVESTREAM:-2}"
VIZ="${VIZ-kit}"           # 라이브스트림엔 kit 필요. 헤드리스로 끄려면 VIZ= (콜론 없는 기본값이라 빈 값 허용)
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

# conda 활성화 (isaac-6.0). ./isaaclab.sh -p 는 비대화셸서 base python으로 새므로 직접 activate.
# shellcheck disable=SC1090
source "$CONDA_HOOK"
conda activate "$SIM_ENV"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU" LIVESTREAM="$LIVESTREAM"

# 인자 조립: VIZ가 비어있지 않으면 --viz 전달(라이브스트림 필수), 비우면 헤드리스
_sim_args=()
[ -n "$VIZ" ] && _sim_args+=(--viz "$VIZ")

echo "[run_sim_runner_hl] env=$SIM_ENV python=$(command -v python) GPU=$GPU LIVESTREAM=$LIVESTREAM"
echo "[run_sim_runner_hl] args: ${_sim_args[*]} $*"

exec python scripts/real2sim/sim_runner_hindleg.py "${_sim_args[@]}" "$@"
