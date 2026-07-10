#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 sim_bridge 원커맨드 런처.
#   - conda가 활성화돼 있으면 (스택 포함) 완전히 비활성화 — conda의 python3가 시스템
#     python3(3.10)를 shadow해 rclpy 임포트를 깨뜨리는 것을 방지.
#   - ROS2 humble + unitree_go overlay 소싱 후 /usr/bin/python3로 sim_bridge 실행.
#
# 사용:  bash scripts/real2sim/r2s_go2/run_sim_bridge.sh   (또는 ./run_sim_bridge.sh)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# conda 잔여 누수 제거 (시스템 python이 conda 패키지를 집지 않도록)
unset PYTHONPATH PYTHONHOME
echo "[run_sim_bridge] conda deactivated (CONDA_SHLVL=${CONDA_SHLVL:-0})"

# 2) ROS2 humble + unitree_go overlay 소싱 (ROS setup.bash는 set -u에서 죽으므로 미설정 상태 유지)
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# 2b) 동시(sim+real) 구동 모드: R2S_SHARED=1 이면 실로봇과 같은 RMW(CycloneDDS)로 전환하고
#     sim 상태를 /sim/lowstate 로 remap(실로봇 /lowstate 와 충돌 회피). 기본(미설정)=fastdds sim전용.
if [ "${R2S_SHARED:-0}" = "1" ]; then
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  _iface="${R2S_ROBOT_IFACE:-$(ip -o -4 addr show 2>/dev/null | awk '/192\.168\.123\./{print $2; exit}')}"
  _iface="${_iface:-ens10f1}"
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$_iface\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
  export R2S_STATE_TOPIC="${R2S_STATE_TOPIC:-/sim/lowstate}"
  echo "[run_sim_bridge] SHARED mode: RMW=cyclonedds iface=$_iface state_topic=$R2S_STATE_TOPIC"
fi

echo "[run_sim_bridge] python=$(command -v python3)  ROS_DISTRO=${ROS_DISTRO:-?}  ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW=${RMW_IMPLEMENTATION:-fastdds(default)}"

# 3) 실행 (인자는 그대로 전달)
exec /usr/bin/python3 "$HERE/sim_bridge.py" "$@"
