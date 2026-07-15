# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 편의 셸 명령. 이 파일을 ~/.bashrc 에서 source 하면 터미널에서 cd/ls 처럼 바로 쓸 수 있다:
#
#   source /home/lgb/IsaacLab-6.0/scripts/real2sim/r2s_go2/r2s_commands.sh
#
# 각 명령은 추가 인자를 그대로 전달한다. 예: r2s_sim --headless / GPU=0 r2s_sim
#
# ── Real2Sim 실시간 구동 (M1/M2) ──────────────────────────────────────────
#   r2s_sim   # 터미널1: Isaac sim_runner (conda isaac-6.0, 라이브스트림)
#   r2s_udp   # 터미널2: ROS2 <-> UDP 브릿지 (sim_bridge)
#   r2s_gui   # 터미널3: PyQt GUI 컨트롤러 (gui_controller)
#   r2s_all   # (선택) tmux 3-pane 동시 기동
#
# ── tuner 모드: 물성 실시간 튜닝 (Phase 2.5) ──────────────────────────────
#   r2s_tuner_sim   # 터미널1: Isaac sim_runner_tuner (chirp/replay 재생 + 실시간 물성 반영)
#                   #          replay 겹쳐보기: r2s_tuner_sim --replay data/go2_real/chirp_kp25.pt
#   r2s_tuner_gui   # 터미널2: PyQt 슬라이더(armature/viscous/coulomb/kp/kd/delay). "Launch Plot"로 오버레이.
#
# ── PACE 시스템 식별 (CONTRACT §12) ───────────────────────────────────────
#   실기 (ROS2 py3.10, 로봇을 매달고 sport 해제 후):
#     r2s_gain   --selftest                        # 추정기만 검증(로봇 불필요)
#     r2s_gain   --suspended --joint 1 --kp 25     # Unitree kp 단위 규약 실측
#     r2s_chirp  --dry-run                         # 무모션 계획 확인
#     r2s_chirp  --suspended --amplitude_scale 0.3 --out sweep_check.npz   # 리그 공진 확인
#     r2s_chirp  --suspended --kp 25 --kd 0.5 --out chirp_kp25.npz         # 본 수집
#   Isaac (conda isaac-6.0):
#     r2s_convert  --capture data/go2_real/chirp_kp25.npz   # .npz -> .pt (+품질/공진 판정)
#     r2s_collect                                            # 합성 chirp(GT 주입, 실로봇 불필요)
#     r2s_fit                                                # 다중 시퀀스 CMA-ES 적합
#     r2s_validate                                           # hold-out 검증(식별 vs nominal)
#     r2s_tb                                                 # 텐서보드
#
#   ⚠ 실기 수집 전 필수 2건: ① r2s_gain 으로 kp 규약 확인  ② r2s_chirp 저진폭 스윕으로 리그 공진 확인.

# 이 파일이 위치한 디렉토리 (source 시 BASH_SOURCE[0] = 이 파일 경로)
_R2S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 실시간 구동 ──
r2s_sim() { bash "$_R2S_DIR/run_sim_runner.sh" "$@"; }      # 터미널1: Isaac sim_runner
r2s_udp() { bash "$_R2S_DIR/run_sim_bridge.sh" "$@"; }      # 터미널2: ROS2<->UDP 브릿지
r2s_gui() { bash "$_R2S_DIR/run_gui_controller.sh" "$@"; }  # 터미널3: PyQt GUI
r2s_all() { bash "$_R2S_DIR/run_all.sh" "$@"; }             # (선택) tmux 3-pane 동시 기동

# ── tuner 모드: 물성 실시간 튜닝 (Phase 2.5) ──
r2s_tuner_sim() { bash "$_R2S_DIR/run_tuner_sim.sh" "$@"; }  # 터미널1: Isaac sim_runner_tuner
r2s_tuner_gui() { bash "$_R2S_DIR/run_tuner_gui.sh" "$@"; }  # 터미널2: 물성 슬라이더 GUI

# ── PACE 식별: 실기 (ROS2 py3.10 + CycloneDDS) ──
r2s_gain()  { bash "$_R2S_DIR/run_gain_check.sh" "$@"; }        # kp 단위 규약 실측
r2s_chirp() { bash "$_R2S_DIR/run_chirp_collector.sh" "$@"; }   # 매달린 GO2에서 chirp 수집

# ── PACE 식별: Isaac (conda isaac-6.0) ──
r2s_collect()  { bash "$_R2S_DIR/run_pace.sh" collect "$@"; }   # 합성 chirp 생성(GT 주입)
r2s_fit()      { bash "$_R2S_DIR/run_pace.sh" fit "$@"; }       # 다중 시퀀스 CMA-ES
r2s_validate() { bash "$_R2S_DIR/run_pace.sh" validate "$@"; }  # hold-out 검증
r2s_convert()  { bash "$_R2S_DIR/run_pace.sh" convert "$@"; }   # 실기 .npz -> .pt
r2s_tb()       { bash "$_R2S_DIR/run_pace.sh" tb "$@"; }        # 텐서보드
