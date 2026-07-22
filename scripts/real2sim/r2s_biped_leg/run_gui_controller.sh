#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-BipedLeg gui_controller 원커맨드 런처 (터미널 2).
#   - ROS 비의존(순수 UDP)이라 humble 소싱 불필요. 시스템 python3(PyQt5)로 실행한다.
#     conda의 python3가 시스템 python을 shadow하지 않도록 conda를 완전히 비활성화한다
#     (spawn되는 monitor.py가 matplotlib을 가진 시스템 python을 상속하도록).
#   - PyQt5 창을 띄우므로 디스플레이(X11/Wayland)가 있는 환경에서 실행할 것.
#
# 사용:  bash scripts/real2sim/r2s_biped_leg/run_gui_controller.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
echo "[run_gui_controller_bl] python=$(command -v python3)  (pure UDP, no ROS)"

exec /usr/bin/python3 "$HERE/gui_controller.py" "$@"
