# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 편의 셸 명령: r2s_sim / r2s_udp / r2s_gui.
# 이 파일을 ~/.bashrc 에서 source 하면 터미널에서 cd/ls 처럼 바로 쓸 수 있다:
#
#   source /home/lgb/IsaacLab-6.0/scripts/real2sim/r2s_go2/r2s_commands.sh
#
# 그 후 각 터미널에서:
#   r2s_sim   # 터미널1: Isaac sim_runner (conda isaac-6.0, 라이브스트림)
#   r2s_udp   # 터미널2: ROS2 <-> UDP 브릿지 (sim_bridge)
#   r2s_gui   # 터미널3: PyQt GUI 컨트롤러 (gui_controller)
#
# 각 명령은 추가 인자를 그대로 전달한다. 예: r2s_sim --headless / GPU=0 r2s_sim

# 이 파일이 위치한 디렉토리 (source 시 BASH_SOURCE[0] = 이 파일 경로)
_R2S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

r2s_sim() { bash "$_R2S_DIR/run_sim_runner.sh" "$@"; }      # 터미널1: Isaac sim_runner
r2s_udp() { bash "$_R2S_DIR/run_sim_bridge.sh" "$@"; }      # 터미널2: ROS2<->UDP 브릿지
r2s_gui() { bash "$_R2S_DIR/run_gui_controller.sh" "$@"; }  # 터미널3: PyQt GUI
r2s_all() { bash "$_R2S_DIR/run_all.sh" "$@"; }             # (선택) tmux 3-pane 동시 기동
