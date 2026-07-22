#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg gui_controller — PyQt5 GUI가 순수 UDP로 sim_runner를 구동.

전송은 순수 UDP다 (ROS2는 향후, 메시지 형식 미정):

    gui ──cmd(9881)──▶ sim_runner ──state(9882)──▶ gui ──relay(9883)──▶ monitor

50Hz로 현재 목표각 + 관절별 kp/kd를 연속 발행하고, sim 상태를 수신해 monitor로 중계한다.
faithful PD가 켜져 있으므로 Gains 그룹의 kp/kd가 실제 sim drive 게인에 반영된다.

8-DOF 2족(HL 4관절 + HR 4관절)이라 관절 콤보에는 8개 레이블이 모두 나온다.

버튼:
    - Home (default): 중립(0) 자세로 보간 이동.
    - Joint Step: 선택 관절만 delta 스텝(soft limit 클램프).
    - Sine Sweep: 선택 관절에 사인 궤적(soft limit 클램프).
    - Gains: 선택 관절 kp/kd 실시간 변경(faithful PD).
    - Monitor: action/sim 실시간 plot 창(별도 프로세스).

시스템 python3 또는 conda python 어디서 실행해도 된다(ROS/torch 비의존).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import motions  # noqa: E402
import r2s_udp  # noqa: E402
from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QFont  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

FRAME_HZ: float = 50.0
FRAME_PERIOD_S: float = 1.0 / FRAME_HZ
PUBLISH_HZ: float = 50.0
PUBLISH_PERIOD_S: float = 1.0 / PUBLISH_HZ
SEQUENCE_DURATION_S: float = 1.5
STEP_DURATION_S: float = 0.5
HOST: str = "127.0.0.1"

# 게인 슬라이더 상한 — 이 리그의 실측 게인은 kp 12~65 / kd 1.0~6.0 으로, R_Skeleton 5-DOF 리그
# (kp 300 / kd 5)보다 한 자릿수 낮다. 상한을 1000/100 그대로 두면 기본값이 눈금 바닥에 붙어
# 사실상 조작이 불가능하므로, 기본값이 대략 중간에 오도록 kp 0~150 / kd 0~15 로 재조정한다.
# step도 이 스케일에 맞춰 잘게 잡는다(기존 10.0/0.5는 kp=12 관절에서 너무 거칠다).
KP_RANGE: tuple[float, float] = (0.0, 150.0)
KP_STEP: float = 1.0
KD_RANGE: tuple[float, float] = (0.0, 15.0)
KD_STEP: float = 0.1

_STYLESHEET: str = """
QWidget { background-color: #1c1e26; color: #e6e8ef;
    font-family: "Segoe UI", "Ubuntu", "Roboto", "DejaVu Sans", sans-serif; font-size: 13px; }
QMainWindow { background-color: #1c1e26; }
#titleLabel { font-size: 19px; font-weight: 600; color: #f4f6fb; }
#subtitleLabel { font-size: 11px; color: #8a8fa3; }
#divider { background-color: #33364688; max-height: 1px; border: none; }
QGroupBox { background-color: #242733; border: 1px solid #33364a; border-radius: 8px;
    margin-top: 14px; padding: 14px 10px 10px 10px; font-weight: 600; color: #c7cbe0; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px;
    padding: 0 6px; color: #9fb4ff; }
QLabel { color: #c7cbe0; background: transparent; }
QPushButton { background-color: #2c2f3d; color: #e6e8ef; border: 1px solid #3d4157;
    border-radius: 6px; padding: 7px 16px; font-weight: 500; }
QPushButton:hover { background-color: #363a4c; border-color: #4a4f6b; }
QPushButton:pressed { background-color: #23252f; }
QPushButton:disabled { background-color: #23252f; color: #5c6079; border-color: #2c2f3d; }
QPushButton#primaryButton { background-color: #4f7dfb; border-color: #4f7dfb; color: #ffffff; }
QPushButton#primaryButton:hover { background-color: #6b93fc; border-color: #6b93fc; }
QPushButton#dangerButton { background-color: #d9534f; border-color: #d9534f; color: #ffffff; }
QPushButton#dangerButton:hover { background-color: #e46864; }
QComboBox, QDoubleSpinBox { background-color: #1c1e26; border: 1px solid #3d4157;
    border-radius: 5px; padding: 5px 8px; min-height: 20px; }
QComboBox QAbstractItemView { background-color: #242733; border: 1px solid #3d4157;
    selection-background-color: #4f7dfb; outline: none; }
#statusBar { background-color: #242733; border: 1px solid #33364a; border-radius: 8px; }
#statusLabel { color: #9fb4ff; font-weight: 600; padding: 8px 14px; }
"""


class UdpLink:
    """순수 UDP 링크 — cmd 발신 / state 수신 / monitor 중계."""

    def __init__(self) -> None:
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_addr = (HOST, r2s_udp.CMD_PORT)
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind((HOST, r2s_udp.STATE_PORT))
        self._state_sock.setblocking(False)
        self._mon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._mon_addr = (HOST, r2s_udp.MONITOR_PORT)
        self._seq = 0

    def send_cmd(self, q: list[float], kp: list[float], kd: list[float]) -> None:
        """목표각 q + 관절별 kp/kd 발행(dq=0, tau=0)."""
        self._seq += 1
        zeros = [0.0] * r2s_udp.NUM_JOINTS
        self._cmd_sock.sendto(r2s_udp.pack_cmd(self._seq, q, zeros, kp, kd, zeros), self._cmd_addr)

    def recv_state_latest(self) -> dict | None:
        """수신 큐를 비우고 최신 상태만 반환(latest-wins). 없으면 None."""
        latest = None
        while True:
            try:
                data, _ = self._state_sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            parsed = r2s_udp.unpack_state(data)
            if parsed is not None:
                latest = parsed
        return latest

    def send_monitor(self, action_q: list[float], sim_q, sim_dq, sim_tau) -> None:
        self._mon_sock.sendto(r2s_udp.pack_monitor(self._seq, action_q, sim_q, sim_dq, sim_tau), self._mon_addr)

    def close(self) -> None:
        self._cmd_sock.close()
        self._state_sock.close()
        self._mon_sock.close()


class MainWindow(QMainWindow):
    """R2S-BipedLeg GUI main window."""

    def __init__(self, link: UdpLink) -> None:
        super().__init__()
        self._link = link
        # 중립(0) 자세에서 시작 — sim default_joint_pos 와 일치(스트림 점프 없음).
        self._current_pose: list[float] = list(motions.DEFAULT_POSE)
        self._latest_pose: list[float] = list(motions.DEFAULT_POSE)
        self._kp: list[float] = list(motions.DEFAULT_KP)
        self._kd: list[float] = list(motions.DEFAULT_KD)
        self._latest_sim: dict | None = None

        self._sequence: list[list[float]] = []
        self._sequence_idx: int = 0
        self._sequence_timer = QTimer(self)
        self._sequence_timer.timeout.connect(self._on_sequence_tick)

        self._sine_timer = QTimer(self)
        self._sine_timer.timeout.connect(self._on_sine_tick)
        self._sine_base_pose: list[float] = list(motions.DEFAULT_POSE)
        self._sine_start_time: float = 0.0

        self._monitor_proc: subprocess.Popen | None = None

        self.setWindowTitle("R2S-BipedLeg Controller")
        self._build_ui()

        self._publish_timer = QTimer(self)
        self._publish_timer.timeout.connect(self._on_publish_tick)
        self._publish_timer.start(int(PUBLISH_PERIOD_S * 1000))

    # -- UI --

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        title_label = QLabel("R2S-BipedLeg Controller")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("8-DOF biped leg (HL/HR) — real2sim UDP position control")
        subtitle_label.setObjectName("subtitleLabel")
        header_title = QVBoxLayout()
        header_title.setSpacing(2)
        header_title.addWidget(title_label)
        header_title.addWidget(subtitle_label)
        header_layout = QHBoxLayout()
        header_layout.addLayout(header_title)
        header_layout.addStretch(1)
        monitor_btn = QPushButton("Monitor")
        monitor_btn.clicked.connect(self._on_monitor_clicked)
        header_layout.addWidget(monitor_btn, alignment=Qt.AlignTop)
        layout.addLayout(header_layout)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        layout.addWidget(divider)

        # Pose
        pose_group = QGroupBox("Pose")
        pose_layout = QHBoxLayout(pose_group)
        home_btn = QPushButton("Home (default)")
        home_btn.setObjectName("primaryButton")
        home_btn.clicked.connect(self._on_home_clicked)
        pose_layout.addWidget(home_btn)
        pose_layout.addStretch(1)
        layout.addWidget(pose_group)

        # Joint Step (콤보는 HL 4개 + HR 4개 = 8 레이블 전부)
        step_group = QGroupBox("Joint Step")
        step_layout = QHBoxLayout(step_group)
        step_layout.addWidget(QLabel("Joint:"))
        self._step_joint_combo = QComboBox()
        self._step_joint_combo.addItems(motions.JOINT_NAMES)
        step_layout.addWidget(self._step_joint_combo)
        step_layout.addWidget(QLabel("delta [rad]:"))
        self._step_delta_spin = QDoubleSpinBox()
        self._step_delta_spin.setRange(-3.14, 3.14)
        self._step_delta_spin.setSingleStep(0.05)
        self._step_delta_spin.setValue(0.2)
        step_layout.addWidget(self._step_delta_spin)
        step_btn = QPushButton("Apply Step")
        step_btn.setObjectName("primaryButton")
        step_btn.clicked.connect(self._on_step_clicked)
        step_layout.addWidget(step_btn)
        step_layout.addStretch(1)
        layout.addWidget(step_group)

        # Sine Sweep
        sine_group = QGroupBox("Sine Sweep")
        sine_layout = QHBoxLayout(sine_group)
        sine_layout.addWidget(QLabel("Joint:"))
        self._sine_joint_combo = QComboBox()
        self._sine_joint_combo.addItems(motions.JOINT_NAMES)
        sine_layout.addWidget(self._sine_joint_combo)
        sine_layout.addWidget(QLabel("Amp [rad]:"))
        self._sine_amp_spin = QDoubleSpinBox()
        self._sine_amp_spin.setRange(0.0, 1.5)
        self._sine_amp_spin.setSingleStep(0.05)
        self._sine_amp_spin.setValue(0.2)
        sine_layout.addWidget(self._sine_amp_spin)
        sine_layout.addWidget(QLabel("Freq [Hz]:"))
        self._sine_freq_spin = QDoubleSpinBox()
        self._sine_freq_spin.setRange(0.05, 5.0)
        self._sine_freq_spin.setSingleStep(0.1)
        self._sine_freq_spin.setValue(0.5)
        sine_layout.addWidget(self._sine_freq_spin)
        self._sine_start_btn = QPushButton("Start")
        self._sine_start_btn.setObjectName("primaryButton")
        self._sine_start_btn.clicked.connect(self._on_sine_start_clicked)
        sine_layout.addWidget(self._sine_start_btn)
        self._sine_stop_btn = QPushButton("Stop")
        self._sine_stop_btn.setObjectName("dangerButton")
        self._sine_stop_btn.clicked.connect(self._on_sine_stop_clicked)
        self._sine_stop_btn.setEnabled(False)
        sine_layout.addWidget(self._sine_stop_btn)
        layout.addWidget(sine_group)

        # Gains (faithful PD) — 슬라이더 상한은 이 리그의 낮은 실측 게인에 맞춰 재조정(KP_RANGE/KD_RANGE).
        gain_group = QGroupBox("Gains (faithful PD)")
        gain_layout = QHBoxLayout(gain_group)
        gain_layout.addWidget(QLabel("Joint:"))
        self._gain_joint_combo = QComboBox()
        self._gain_joint_combo.addItems(motions.JOINT_NAMES)
        self._gain_joint_combo.currentIndexChanged.connect(self._on_gain_joint_changed)
        gain_layout.addWidget(self._gain_joint_combo)
        gain_layout.addWidget(QLabel("kp:"))
        self._kp_spin = QDoubleSpinBox()
        self._kp_spin.setRange(*KP_RANGE)
        self._kp_spin.setSingleStep(KP_STEP)
        self._kp_spin.setValue(self._kp[0])
        gain_layout.addWidget(self._kp_spin)
        gain_layout.addWidget(QLabel("kd:"))
        self._kd_spin = QDoubleSpinBox()
        self._kd_spin.setRange(*KD_RANGE)
        self._kd_spin.setSingleStep(KD_STEP)
        self._kd_spin.setValue(self._kd[0])
        gain_layout.addWidget(self._kd_spin)
        gain_btn = QPushButton("Apply Gains")
        gain_btn.setObjectName("primaryButton")
        gain_btn.clicked.connect(self._on_gain_apply_clicked)
        gain_layout.addWidget(gain_btn)
        gain_layout.addStretch(1)
        layout.addWidget(gain_group)

        layout.addStretch(1)

        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(4, 2, 4, 2)
        self._status_label = QLabel("Idle")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(self._status_label)
        layout.addWidget(status_bar)

        self.setCentralWidget(central)

    # -- sequence playback --

    def _play_poses(self, seq: list[list[float]]) -> None:
        self._stop_sine()
        self._sequence_timer.stop()
        self._sequence = seq
        self._sequence_idx = 0
        self._sequence_timer.start(int(FRAME_PERIOD_S * 1000))

    def _play_sequence_to(self, goal_pose: list[float], duration_s: float) -> None:
        num_steps = max(1, int(duration_s * FRAME_HZ))
        self._play_poses(motions.interpolate_sequence(self._current_pose, goal_pose, num_steps))

    def _on_sequence_tick(self) -> None:
        if self._sequence_idx >= len(self._sequence):
            self._sequence_timer.stop()
            return
        pose = self._sequence[self._sequence_idx]
        self._latest_pose = pose
        self._current_pose = pose
        self._sequence_idx += 1

    # -- pose buttons --

    def _on_home_clicked(self) -> None:
        self._status_label.setText("Moving to home (default) pose...")
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)

    def _on_step_clicked(self) -> None:
        joint_idx = self._step_joint_combo.currentIndex()
        delta = self._step_delta_spin.value()
        goal_pose = motions.step_pose(self._current_pose, joint_idx, delta)
        self._status_label.setText(f"Stepping {motions.JOINT_NAMES[joint_idx]} ({delta:+.2f} rad)...")
        self._play_sequence_to(goal_pose, STEP_DURATION_S)

    # -- sine --

    def _on_sine_start_clicked(self) -> None:
        self._sequence_timer.stop()
        self._sine_base_pose = list(self._current_pose)
        self._sine_start_time = time.monotonic()
        self._sine_start_btn.setEnabled(False)
        self._sine_stop_btn.setEnabled(True)
        self._status_label.setText(f"Sine sweep on {self._sine_joint_combo.currentText()}...")
        self._sine_timer.start(int(FRAME_PERIOD_S * 1000))

    def _on_sine_tick(self) -> None:
        joint_idx = self._sine_joint_combo.currentIndex()
        amplitude = self._sine_amp_spin.value()
        frequency = self._sine_freq_spin.value()
        t = time.monotonic() - self._sine_start_time
        pose = motions.sine_offset(self._sine_base_pose, joint_idx, amplitude, frequency, t)
        self._latest_pose = pose
        self._current_pose = pose

    def _on_sine_stop_clicked(self) -> None:
        self._stop_sine()
        self._status_label.setText("Sine sweep stopped")

    def _stop_sine(self) -> None:
        self._sine_timer.stop()
        self._sine_start_btn.setEnabled(True)
        self._sine_stop_btn.setEnabled(False)

    # -- gains (faithful PD) --

    def _on_gain_joint_changed(self, idx: int) -> None:
        self._kp_spin.setValue(self._kp[idx])
        self._kd_spin.setValue(self._kd[idx])

    def _on_gain_apply_clicked(self) -> None:
        idx = self._gain_joint_combo.currentIndex()
        self._kp[idx] = self._kp_spin.value()
        self._kd[idx] = self._kd_spin.value()
        # 게인이 한 자릿수라 소수 1자리까지 표시(5-DOF 리그의 %.0f 로는 12.5/1.1 구분 불가).
        self._status_label.setText(f"Gains {motions.JOINT_NAMES[idx]}: kp={self._kp[idx]:.1f} kd={self._kd[idx]:.2f}")

    # -- monitor --

    def _on_monitor_clicked(self) -> None:
        # 별도 프로세스로 monitor.py spawn — 무거운 matplotlib 렌더를 이 GUI의 50Hz UDP 발행
        # 루프에서 격리한다(in-process로 되돌리면 발행이 굶는다). 이미 실행 중이면 무시.
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._status_label.setText("Monitor already running")
            return
        monitor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor.py")
        self._monitor_proc = subprocess.Popen([sys.executable, monitor_py])
        self._status_label.setText("Monitor launched (separate process)")

    # -- continuous publisher + relay --

    def _on_publish_tick(self) -> None:
        # 1) 목표각 + kp/kd 발행
        self._link.send_cmd(self._latest_pose, self._kp, self._kd)
        # 2) sim 상태 수신(latest-wins)
        state = self._link.recv_state_latest()
        if state is not None:
            self._latest_sim = state
        # 3) monitor로 중계 (action + sim, time-aligned)
        if self._latest_sim is not None:
            self._link.send_monitor(
                self._latest_pose, self._latest_sim["q"], self._latest_sim["dq"], self._latest_sim["tau_est"]
            )
        else:
            zeros = [0.0] * r2s_udp.NUM_JOINTS
            self._link.send_monitor(self._latest_pose, zeros, zeros, zeros)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._publish_timer.stop()
        self._sequence_timer.stop()
        self._sine_timer.stop()
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._monitor_proc.terminate()
        super().closeEvent(event)


def main() -> None:
    link = UdpLink()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow(link)
    window.show()
    exit_code = app.exec_()
    link.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
