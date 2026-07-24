#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-BipedLeg gui_controller 원커맨드 런처 (터미널 2).
#   - GUI는 추후 ROS2(rclpy)로 확장되므로 **시스템 python3(/usr/bin/python3, py3.10) + /opt/ros/humble**에서
#     돌아야 한다(conda를 비활성화해야 rclpy가 안 깨짐 — r2s_go2 패턴). 따라서 conda를 완전히 비활성화하고
#     시스템 python3로 실행한다. spawn되는 monitor.py(matplotlib)도 시스템 python3를 상속한다.
#   - Position mode는 순수 UDP라 torch 불필요. Policy mode(deployable jit를 GUI 내부에서 추론)는 torch가
#     필요하며, torch는 시스템 python3에 설치돼 있어야 한다(없으면 Policy mode 자동 비활성, Position만 동작).
#   - PyQt5 창을 띄우므로 디스플레이(X11/Wayland)가 있는 환경에서 실행할 것.
#
# 사용:  bash scripts/real2sim/r2s_biped_leg/run_gui_controller.sh [--model <jit>] [--device cpu|cuda:0]

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
echo "[run_gui_controller_bl] python=$(command -v python3)  (system python3 — PyQt5; policy는 torch 있으면 활성)"

exec /usr/bin/python3 "$HERE/gui_controller.py" "$@"
