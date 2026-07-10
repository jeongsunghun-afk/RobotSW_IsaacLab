#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 원커맨드 tmux 런처 — sim_runner + sim_bridge + gui_controller 를
# 하나의 tmux 세션(3 pane)에 동시에 띄운다. 각 pane이 자기 로그를 유지한다.
#
# 사용:
#   bash scripts/real2sim/r2s_go2/run_all.sh          # 세 프로세스 기동 후 세션에 attach
#   bash scripts/real2sim/r2s_go2/run_all.sh stop     # 세 프로세스 전체 종료(세션 kill)
#
# 조정(환경변수로 오버라이드 가능):
#   SIM_ENV=isaac-6.0  GPU=2  LIVESTREAM=2  ENABLE_CAMERAS=1  SIM_EXTRA="--fix_base --viz kit"
#
# 주의: gui_controller는 X 창을 띄우므로 디스플레이가 있는 터미널에서 실행할 것.

set -u

SESSION="${R2S_SESSION:-r2s_go2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# --- stop 서브커맨드 ---
if [ "${1:-}" = "stop" ]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "[run_all] 세션 '$SESSION' 종료됨" || echo "[run_all] 실행 중인 '$SESSION' 세션 없음"
  exit 0
fi

# --- 조정 가능한 설정 ---
SIM_ENV="${SIM_ENV:-isaac-6.0}"
GPU="${GPU:-2}"
LIVESTREAM="${LIVESTREAM:-2}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
SIM_EXTRA="${SIM_EXTRA:---fix_base --viz kit}"
CONDA_HOOK="${CONDA_HOOK:-/home/user/miniconda3/etc/profile.d/conda.sh}"

command -v tmux >/dev/null || { echo "[run_all] tmux가 필요합니다 (sudo apt install tmux)"; exit 1; }

# 기존 세션 정리
tmux kill-session -t "$SESSION" 2>/dev/null

SIM_CMD="cd '$REPO' && source '$CONDA_HOOK' && conda activate '$SIM_ENV' && CUDA_VISIBLE_DEVICES=$GPU LIVESTREAM=$LIVESTREAM ENABLE_CAMERAS=$ENABLE_CAMERAS python scripts/real2sim/sim_runner_go2.py $SIM_EXTRA"
BRIDGE_CMD="cd '$REPO' && bash scripts/real2sim/r2s_go2/run_sim_bridge.sh"
GUI_CMD="cd '$REPO' && bash scripts/real2sim/r2s_go2/run_gui_controller.sh"

# pane 0: sim_runner (Isaac)
tmux new-session -d -s "$SESSION" -n r2s
tmux send-keys -t "$SESSION:0.0" "$SIM_CMD" C-m

# pane 1: sim_bridge (ROS2)
tmux split-window -v -t "$SESSION:0"
tmux send-keys -t "$SESSION:0.1" "$BRIDGE_CMD" C-m

# pane 2: gui_controller (ROS2)
tmux split-window -v -t "$SESSION:0"
tmux send-keys -t "$SESSION:0.2" "$GUI_CMD" C-m

tmux select-layout -t "$SESSION:0" even-vertical
tmux select-pane -t "$SESSION:0.0"

echo "[run_all] tmux 세션 '$SESSION' 기동 (pane0=sim, pane1=bridge, pane2=gui)"
echo "[run_all] 종료: 'bash $0 stop'  또는 tmux에서 'Ctrl-b &'"
echo "[run_all] pane 이동: 'Ctrl-b ↑/↓', detach: 'Ctrl-b d'"

# 대화형 터미널이면 attach, 아니면 안내만
if [ -t 1 ]; then
  tmux attach -t "$SESSION"
else
  echo "[run_all] 비대화형 실행 — 'tmux attach -t $SESSION'로 접속하세요."
fi
