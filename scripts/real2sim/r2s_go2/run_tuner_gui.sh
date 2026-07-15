#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 tuner_gui 런처 (터미널 2) — Phase 2.5.
#   - 순수 UDP + PyQt5. rclpy/ROS2 불필요. "Launch Plot"이 tuner_monitor.py(matplotlib)를 spawn.
#   - conda가 활성화돼 있으면 비활성화 — 시스템 python3(PyQt5/matplotlib 보유)로 실행.
#   - PyQt5 창을 띄우므로 디스플레이(X11/Wayland)가 있는 환경에서 실행할 것.
#
# 사용:  bash scripts/real2sim/r2s_go2/run_tuner_gui.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# conda 완전 비활성화 (활성화된 경우에만) — conda python이 시스템 PyQt5를 shadow하지 않도록.
if [ "${CONDA_SHLVL:-0}" -gt 0 ]; then
  _base="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
  [ -z "$_base" ] && _base="$(conda info --base 2>/dev/null)"
  if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$_base/etc/profile.d/conda.sh"
  fi
  while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate 2>/dev/null || break; done
fi
unset PYTHONPATH PYTHONHOME

echo "[run_tuner_gui] python=$(command -v python3)  (순수 UDP, ROS2 불필요)"

exec /usr/bin/python3 "$HERE/tuner_gui.py" "$@"
