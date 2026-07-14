#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 실기 GO2와 대화하는 파이썬 스크립트용 공통 런처 (내부 헬퍼).
#   - conda를 완전히 비활성화한다 (conda의 python3(3.12)가 시스템 python3(3.10)를 shadow하면
#     rclpy 임포트가 깨진다 — CONTRACT §1).
#   - ROS2 humble + unitree_go overlay + CycloneDDS(실기 RMW) + iface를 세팅한다.
#   - /usr/bin/python3 로 대상 스크립트를 실행한다.
#
# 사용:  bash run_real_py.sh <script.py> [인자...]
# 조정:  R2S_ROBOT_IFACE=ens10f1  ROS_DOMAIN_ID=0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$1"
shift || true

if [ -z "$TARGET" ]; then
  echo "[run_real_py] 사용법: bash run_real_py.sh <script.py> [인자...]" >&2
  exit 2
fi

# 1) conda 완전 비활성화 (활성화된 경우에만)
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

# 2) ROS2 humble + unitree_go overlay
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# 3) 실기 RMW = CycloneDDS + GO2 네트워크 인터페이스 (192.168.123.x 보유 iface 자동탐지)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
_iface="${R2S_ROBOT_IFACE:-$(ip -o -4 addr show 2>/dev/null | awk '/192\.168\.123\./{print $2; exit}')}"
if [ -z "$_iface" ]; then
  echo "[run_real_py] ⚠ GO2 인터페이스 미확정 (192.168.123.x 보유 iface 없음)."
  echo "[run_real_py]   랜 연결 + static IP(192.168.123.99/24) 확인, 또는 R2S_ROBOT_IFACE=<iface> 지정."
else
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$_iface\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
fi

echo "[run_real_py] python=$(command -v python3)  ROS_DISTRO=${ROS_DISTRO:-?}  RMW=$RMW_IMPLEMENTATION  iface=${_iface:-미확정}"
echo "[run_real_py] exec: $(basename "$TARGET") $*"

exec /usr/bin/python3 "$HERE/$(basename "$TARGET")" "$@"
