# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# unitree_ros2 example 를 실기 GO2 로 실행하기 위한 환경 소싱 (M2). **source 해서 사용.**
# robot_env.sh(humble+cyclonedds overlay+RMW=cyclonedds+iface) 위에 example install 을 얹는다.
#
# 사용(대화형 셸에서):
#   source scripts/real2sim/r2s_go2/example_env.sh [iface]
#
# 그 후 바이너리 직접 실행(경로는 bin/ — ros2 run 아님):
#   BIN=/home/lgb/unitree_ros2/example/install/unitree_ros2_example/bin
#   "$BIN/read_low_state"          # 읽기전용(모션 없음) — 먼저 이걸로 확인
#   "$BIN/low_level_ctrl"          # ⚠ RL 다리 모션 — 안전조치(로봇 매달기+sport off) 후에만
#
# 주의: 이 스크립트는 환경만 세팅한다. 예제 실행은 사용자가 직접 한다.

_R2S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) 로봇 통신 env (conda 비활성화 + humble + cyclonedds_ws overlay + RMW=cyclonedds + iface).
# shellcheck disable=SC1091
source "$_R2S_DIR/robot_env.sh" "$@"

# 2) example install overlay (unitree_ros2_example 바이너리 + 런타임 라이브러리 경로).
_R2S_EXAMPLE_INSTALL="/home/lgb/unitree_ros2/example/install/setup.bash"
if [ -f "$_R2S_EXAMPLE_INSTALL" ]; then
    # shellcheck disable=SC1090
    source "$_R2S_EXAMPLE_INSTALL"
    echo "[example_env] ✅ 준비 완료. 바이너리: /home/lgb/unitree_ros2/example/install/unitree_ros2_example/bin/"
    echo "[example_env]   읽기전용:  \$BIN/read_low_state   (모션 없음)"
    echo "[example_env]   모션(주의): \$BIN/low_level_ctrl   ← 로봇 매달기 + sport off 후에만"
else
    echo "[example_env] ⚠ example install 없음($_R2S_EXAMPLE_INSTALL). colcon build 먼저."
fi
unset _R2S_DIR _R2S_EXAMPLE_INSTALL
