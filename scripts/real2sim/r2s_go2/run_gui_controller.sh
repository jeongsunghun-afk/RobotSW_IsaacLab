#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 gui_controller 원커맨드 런처.
#   - conda가 활성화돼 있으면 (스택 포함) 완전히 비활성화 — conda의 python3가 시스템
#     python3(3.10)를 shadow해 rclpy 임포트를 깨뜨리는 것을 방지.
#   - ROS2 humble + unitree_go overlay 소싱 후 /usr/bin/python3로 gui_controller 실행.
#   - PyQt5 창을 띄우므로 디스플레이(X11/Wayland)가 있는 환경에서 실행할 것.
#
# 사용:  bash scripts/real2sim/r2s_go2/run_gui_controller.sh   (또는 ./run_gui_controller.sh)

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
echo "[run_gui_controller] conda deactivated (CONDA_SHLVL=${CONDA_SHLVL:-0})"

# 2) ROS2 humble + unitree_go overlay 소싱
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# 2b) 동시(sim+real) 구동 모드: R2S_SHARED=1 이면 실로봇과 같은 RMW(CycloneDDS)로 전환.
#     ⚠ 이 모드에서 gui는 /lowcmd 를 CycloneDDS로 발행 → 같은 도메인의 실로봇도 즉시 명령을 받는다.
#     기본(미설정)=fastdds sim전용(실로봇 안 건드림).
if [ "${R2S_SHARED:-0}" = "1" ]; then
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  _iface="${R2S_ROBOT_IFACE:-$(ip -o -4 addr show 2>/dev/null | awk '/192\.168\.123\./{print $2; exit}')}"
  _iface="${_iface:-ens10f1}"
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$_iface\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
  # Monitor 창의 sim 소스: shared 에서 sim_bridge 는 /sim/lowstate 로 remap 발행(실로봇 /lowstate 와 분리).
  export R2S_SIM_STATE_TOPIC="${R2S_SIM_STATE_TOPIC:-/sim/lowstate}"
  echo "[run_gui_controller] SHARED mode: RMW=cyclonedds iface=$_iface  ⚠ 실로봇도 /lowcmd 수신"
else
  # 비-shared(sim 전용): sim_bridge 는 /lowstate 로 발행. Monitor 의 sim 소스를 여기에 맞춘다
  # (실로봇 부재 → robot 토픽과 같아져 monitor 가 robot 구독을 dedup, "sim only" 로 표시).
  export R2S_SIM_STATE_TOPIC="${R2S_SIM_STATE_TOPIC:-/lowstate}"
fi

echo "[run_gui_controller] python=$(command -v python3)  ROS_DISTRO=${ROS_DISTRO:-?}  ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  RMW=${RMW_IMPLEMENTATION:-fastdds(default)}"

# 3) 실행 (인자는 그대로 전달)
exec /usr/bin/python3 "$HERE/gui_controller.py" "$@"
