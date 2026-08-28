#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-BipedLeg policy 모드 sim_runner 런처 (터미널 1, Isaac conda).
#   - sim_runner_bipedleg 를 --policy_mode 로 실행 (POLICY_ACT 수신 → 1 step → POLICY_STATE 회신, lockstep).
#   - policy 모드는 자유베이스(정책이 균형)라 fix_base 개념이 없다.
#
# 사용:  bash scripts/real2sim/r2s_biped_leg/run_policy_sim.sh [추가 인자]
# 조정(환경변수): SIM_ENV=isaac-6.0 GPU=0 LIVESTREAM=2 VIZ=kit FIX_BASE=0
#   라이브스트림으로 로봇을 보려면 VIZ=kit(기본). 헤드리스로 끄려면 VIZ= 로 실행.
#   FIX_BASE=1 이면 base 를 공중에 고정한다 — 정책을 넘어뜨리지 않고 관절 거동·부호·토크를
#   먼저 확인하는 안전 점검용이다. ⚠ 접촉이 없고 base 상태가 학습 분포 밖이라
#   보행 성능·트립 판정은 자유베이스에서 다시 재야 한다.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-0}"
LIVESTREAM="${LIVESTREAM:-2}"
VIZ="${VIZ-kit}"           # 라이브스트림엔 kit 필요. 헤드리스로 끄려면 VIZ= (콜론 없는 기본값이라 빈 값 허용)
FIX_BASE="${FIX_BASE:-0}"  # 1=base 공중고정(--fix_base). 넘어질 수 없는 상태로 정책을 먼저 보는 안전 점검용.
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

# conda 활성화 (isaac-6.0). ./isaaclab.sh -p 는 비대화셸서 base python으로 새므로 직접 activate.
# shellcheck disable=SC1090
source "$CONDA_HOOK"
conda activate "$SIM_ENV"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU" LIVESTREAM="$LIVESTREAM"

_sim_args=(--unified)
[ "$FIX_BASE" != "0" ] && _sim_args+=(--fix_base)
[ -n "$VIZ" ] && _sim_args+=(--viz "$VIZ")

echo "[run_unified_sim_bl] env=$SIM_ENV python=$(command -v python) GPU=$GPU LIVESTREAM=$LIVESTREAM FIX_BASE=$FIX_BASE"
echo "[run_unified_sim_bl] args: ${_sim_args[*]} $*"

exec python scripts/real2sim/sim_runner_bipedleg.py "${_sim_args[@]}" "$@"
