#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 tuner sim 런처 (터미널 1, Isaac conda) — Phase 2.5.
#   - isaac-6.0 conda env를 활성화하고 라이브스트림을 켠 뒤 sim_runner_tuner_go2.py 실행.
#   - tuner_gui(:9875)가 보낸 물성을 실시간 반영하며 chirp/replay를 재생, telem(:9876)을 흘려보낸다.
#
# 사용:  bash scripts/real2sim/r2s_go2/run_tuner_sim.sh [--replay data/go2_real/chirp_kp25.pt] [추가 인자]
# 조정(환경변수): SIM_ENV=isaac-6.0 GPU=2 LIVESTREAM=2 ENABLE_CAMERAS=1 VIZ=kit
#   VIZ= (빈 값)으로 두면 헤드리스 — 3D 씬 없이 tuner_monitor 플롯만으로 튜닝.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-2}"
LIVESTREAM="${LIVESTREAM:-2}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
VIZ="${VIZ-kit}"  # 라이브스트림엔 kit 필요. 끄려면 VIZ= (콜론 없는 기본값이라 빈 값 허용)
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

# conda 활성화 (isaac-6.0). ./isaaclab.sh -p 는 비대화셸서 base python으로 새므로 직접 activate 한다.
# shellcheck disable=SC1090
source "$CONDA_HOOK"
conda activate "$SIM_ENV"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU" LIVESTREAM="$LIVESTREAM" ENABLE_CAMERAS="$ENABLE_CAMERAS"

_sim_args=()
[ -n "$VIZ" ] && _sim_args+=(--viz "$VIZ")

echo "[run_tuner_sim] env=$SIM_ENV python=$(command -v python) GPU=$GPU LIVESTREAM=$LIVESTREAM"
echo "[run_tuner_sim] args: ${_sim_args[*]} $*"

exec python scripts/real2sim/sim_runner_tuner_go2.py "${_sim_args[@]}" "$@"
