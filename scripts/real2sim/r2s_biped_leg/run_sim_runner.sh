#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-BipedLeg sim_runner 원커맨드 런처 (터미널 1, Isaac conda).
#   - isaac-6.0 conda env를 활성화한 뒤 sim_runner_bipedleg 실행.
#
# 사용:  bash scripts/real2sim/r2s_biped_leg/run_sim_runner.sh [추가 인자]
# 조정(환경변수): SIM_ENV=isaac-6.0 GPU=2 LIVESTREAM=2 VIZ=kit FIX_BASE=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-2}"
LIVESTREAM="${LIVESTREAM:-2}"
VIZ="${VIZ-kit}"           # 라이브스트림엔 kit 필요. 헤드리스로 끄려면 VIZ= (콜론 없는 기본값이라 빈 값 허용)
FIX_BASE="${FIX_BASE:-1}"  # 1=base 공중고정(--fix_base), 0=자유베이스
                           # 기본 ON: 자유베이스 2족은 GUI로 관절을 스텝하는 순간 즉시 넘어진다.
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

# conda 활성화 (isaac-6.0). ./isaaclab.sh -p 는 비대화셸서 base python으로 새므로 직접 activate.
# shellcheck disable=SC1090
source "$CONDA_HOOK"
conda activate "$SIM_ENV"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU" LIVESTREAM="$LIVESTREAM"

# 인자 조립: FIX_BASE가 0이 아니면 --fix_base, VIZ가 비어있지 않으면 --viz 전달(라이브스트림 필수).
# --viz 를 안 붙이면 헤드리스 (--headless 는 6.0에서 deprecated).
_sim_args=()
[ "$FIX_BASE" != "0" ] && _sim_args+=(--fix_base)
[ -n "$VIZ" ] && _sim_args+=(--viz "$VIZ")

echo "[run_sim_runner_bl] env=$SIM_ENV python=$(command -v python) GPU=$GPU LIVESTREAM=$LIVESTREAM FIX_BASE=$FIX_BASE"
echo "[run_sim_runner_bl] args: ${_sim_args[*]} $*"

exec python scripts/real2sim/sim_runner_bipedleg.py "${_sim_args[@]}" "$@"
