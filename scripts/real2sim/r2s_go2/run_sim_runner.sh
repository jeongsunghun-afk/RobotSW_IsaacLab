#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 sim_runner 원커맨드 런처 (터미널 1, Isaac conda).
#   - isaac-6.0 conda env를 활성화하고 라이브스트림 env를 설정한 뒤 sim_runner 실행.
#
# 사용:  bash scripts/real2sim/r2s_go2/run_sim_runner.sh [추가 인자]
# 조정(환경변수): SIM_ENV=isaac-6.0 GPU=2 LIVESTREAM=2 ENABLE_CAMERAS=1 SIM_EXTRA="--fix_base --viz kit"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-2}"
LIVESTREAM="${LIVESTREAM:-2}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
FIX_BASE="${FIX_BASE:-1}"  # 1=base 공중고정(--fix_base), 0=지면 자유낙하
VIZ="${VIZ-kit}"           # 라이브스트림엔 kit 필요. 끄려면 VIZ= (콜론 없는 기본값이라 빈 값 허용)
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

# conda 활성화 (isaac-6.0). ./isaaclab.sh -p 는 비대화셸서 base python으로 새므로 직접 activate 한다.
# shellcheck disable=SC1090
source "$CONDA_HOOK"
conda activate "$SIM_ENV"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU" LIVESTREAM="$LIVESTREAM" ENABLE_CAMERAS="$ENABLE_CAMERAS"

# 인자 조립: SIM_EXTRA가 설정되면 그것으로 완전히 대체, 아니면 FIX_BASE/VIZ 토글로 구성
if [ -n "${SIM_EXTRA:-}" ]; then
    read -ra _sim_args <<< "$SIM_EXTRA"
else
    _sim_args=()
    [ "$FIX_BASE" != "0" ] && _sim_args+=(--fix_base)
    [ -n "$VIZ" ] && _sim_args+=(--viz "$VIZ")
fi

echo "[run_sim_runner] env=$SIM_ENV python=$(command -v python) GPU=$GPU LIVESTREAM=$LIVESTREAM ENABLE_CAMERAS=$ENABLE_CAMERAS"
echo "[run_sim_runner] args: ${_sim_args[*]} $*"

exec python scripts/real2sim/sim_runner_go2.py "${_sim_args[@]}" "$@"
