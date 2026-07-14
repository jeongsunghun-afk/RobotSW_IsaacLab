#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# PACE 시스템 식별 파이프라인 원커맨드 런처 (Isaac conda 쪽, CONTRACT §12).
#   - isaac-6.0 conda env를 활성화하고 헤드리스로 실행한다.
#   - 표시 장치가 없어야 EGL 오프스크린으로 뜨므로 DISPLAY를 지운다.
#
# 사용:  bash run_pace.sh <subcommand> [추가 인자]
#
#   collect   합성 chirp 생성 (GT 주입, 실로봇 불필요)   → data/<robot>/chirp_*.pt
#   fit       다중 시퀀스 CMA-ES 적합                     → logs/pace/<robot>/
#   validate  hold-out 검증 (식별 vs nominal 비교)
#   convert   실기 캡처(.npz) → PACE 데이터셋(.pt)        (Isaac Sim 미기동)
#   tb        텐서보드
#
# 조정(환경변수): SIM_ENV=isaac-6.0  GPU=2  NUM_ENVS=4096  ROBOT=go2_sim

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-2}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROBOT="${ROBOT:-go2_sim}"
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

SUB="$1"
shift || true

# shellcheck disable=SC1090
source "$CONDA_HOOK"
conda activate "$SIM_ENV"
cd "$REPO" || exit 1

echo "[run_pace] env=$SIM_ENV python=$(command -v python) GPU=$GPU sub=${SUB:-?}"

case "$SUB" in
  collect)
    # 합성 chirp (Phase 1 자기복원용). --kp/--kd/--amplitude_scale/--out 로 게인 세트를 만든다.
    exec env -u DISPLAY python scripts/real2sim/collect_chirp_sim_go2.py \
      --headless --device "cuda:$GPU" "$@"
    ;;
  fit)
    # 다중 시퀀스 CMA-ES. 데이터셋 목록은 Go2PaceCfg.datasets 가 정한다.
    exec env -u DISPLAY python scripts/real2sim/fit_go2.py \
      --headless --device "cuda:$GPU" --num_envs "$NUM_ENVS" "$@"
    ;;
  validate)
    # hold-out 포함 검증. --params 로 특정 체크포인트 지정 가능(기본=최신 run의 마지막).
    exec env -u DISPLAY python scripts/real2sim/validate_go2.py \
      --headless --device "cuda:$GPU" "$@"
    ;;
  convert)
    # 실기 캡처 변환. Isaac Sim을 띄우지 않으므로 --device 불필요.
    exec python scripts/real2sim/convert_capture_to_pt.py "$@"
    ;;
  tb)
    exec tensorboard --logdir "logs/pace/$ROBOT" "$@"
    ;;
  *)
    echo "[run_pace] 사용법: bash run_pace.sh {collect|fit|validate|convert|tb} [인자...]" >&2
    echo "[run_pace]   예) bash run_pace.sh collect --kp 40 --kd 1.0 --amplitude_scale 0.8 --out chirp_kp40.pt" >&2
    echo "[run_pace]   예) bash run_pace.sh fit" >&2
    echo "[run_pace]   예) bash run_pace.sh convert --capture data/go2_real/chirp_kp25.npz" >&2
    exit 2
    ;;
esac
