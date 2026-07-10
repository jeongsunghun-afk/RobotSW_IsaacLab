#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 gui_controller — PyQt5 GUI that publishes `/lowcmd` (LowCmd).

Run with system Python 3.10 + ROS2 humble + unitree_go overlay (CONTRACT.md §1):

    source /opt/ros/humble/setup.bash
    source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
    /usr/bin/python3 scripts/real2sim/r2s_go2/gui_controller.py

Buttons:
    - Default (stand): interpolate from the current pose to DEFAULT_POSE.
    - Joint step: step only the selected joint by current + delta (interpolated).
    - Sine sweep: inject a sine wave on the selected joint around DEFAULT_POSE
      (continuous, stopped with the Stop button).
    - Sit: interpolate from the current pose to SIT_POSE.

Every command is published with kp=25, kd=0.5 (all joints), mode=0x01, dq=0, tau=0 (CONTRACT.md §6).

The current target is streamed continuously at PUBLISH_HZ (not one-shot per button): the real
GO2 firmware faults on a gap in the command stream. Each LowCmd carries the deploy header
(head=0xFE 0xEF, level_flag=0xFF) and a valid CRC (see lowcmd_crc). Both are inert to the sim
path (sim_bridge strips them) but required for real deployment (CONTRACT.md §8).

Contract: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md

Visual styling is a self-contained QSS stylesheet applied in :func:`main` (see
``_STYLESHEET`` below); it is purely cosmetic and does not affect any publish/timer logic.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import lowcmd_crc  # noqa: E402
import motions  # noqa: E402
import r2s_udp  # noqa: E402
import rclpy  # noqa: E402
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
from rclpy.node import Node  # noqa: E402
from unitree_go.msg import LowCmd  # noqa: E402

FRAME_HZ: float = 50.0  # motion generation rate (sequence/sine frames)
FRAME_PERIOD_S: float = 1.0 / FRAME_HZ
# Continuous /lowcmd rate. 50Hz matches the standard RL cadence (policy 50Hz, sim
# decimation=4 → 200Hz physics / 50Hz control) and the real GO2 RL-deploy convention:
# the policy streams /lowcmd at 50Hz while the robot firmware runs its PD loop at a much
# higher internal rate. A steady 50Hz stream also satisfies the low-level watchdog.
PUBLISH_HZ: float = 50.0
PUBLISH_PERIOD_S: float = 1.0 / PUBLISH_HZ
SEQUENCE_DURATION_S: float = 1.5  # interpolation time for default/sit
STEP_DURATION_S: float = 0.5  # interpolation time for joint step
SPIN_PERIOD_S: float = 0.01  # rclpy spin_once polling period

# Self-contained dark-theme QSS. Purely cosmetic (colors/spacing/typography) — applied once
# in main() via app.setStyleSheet(); no widget behavior, signal wiring, or timer logic reads
# from it. Font family list relies on Qt's built-in fallback (no bundled font file needed).
_STYLESHEET: str = """
QWidget {
    background-color: #1c1e26;
    color: #e6e8ef;
    font-family: "Segoe UI", "Ubuntu", "Roboto", "DejaVu Sans", sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #1c1e26;
}

#titleLabel {
    font-size: 19px;
    font-weight: 600;
    color: #f4f6fb;
}

#subtitleLabel {
    font-size: 11px;
    color: #8a8fa3;
}

#divider {
    background-color: #33364688;
    max-height: 1px;
    border: none;
}

QGroupBox {
    background-color: #242733;
    border: 1px solid #33364a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    color: #c7cbe0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #9fb4ff;
}

QLabel {
    color: #c7cbe0;
    background: transparent;
}

QPushButton {
    background-color: #2c2f3d;
    color: #e6e8ef;
    border: 1px solid #3d4157;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #363a4c;
    border-color: #4a4f6b;
}

QPushButton:pressed {
    background-color: #23252f;
}

QPushButton:disabled {
    background-color: #23252f;
    color: #5c6079;
    border-color: #2c2f3d;
}

QPushButton#primaryButton {
    background-color: #4f7dfb;
    border-color: #4f7dfb;
    color: #ffffff;
}

QPushButton#primaryButton:hover {
    background-color: #6b93fc;
    border-color: #6b93fc;
}

QPushButton#primaryButton:pressed {
    background-color: #3d67e0;
}

QPushButton#dangerButton {
    background-color: #d9534f;
    border-color: #d9534f;
    color: #ffffff;
}

QPushButton#dangerButton:hover {
    background-color: #e46864;
}

QPushButton#dangerButton:pressed:enabled {
    background-color: #c14743;
}

QPushButton#dangerButton:disabled {
    background-color: #23252f;
    color: #5c6079;
    border-color: #2c2f3d;
}

QComboBox, QDoubleSpinBox {
    background-color: #1c1e26;
    border: 1px solid #3d4157;
    border-radius: 5px;
    padding: 5px 8px;
    min-height: 20px;
}

QComboBox:hover, QDoubleSpinBox:hover {
    border-color: #4a4f6b;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #242733;
    border: 1px solid #3d4157;
    selection-background-color: #4f7dfb;
    outline: none;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 16px;
    border: none;
}

#statusBar {
    background-color: #242733;
    border: 1px solid #33364a;
    border-radius: 8px;
}

#statusLabel {
    color: #9fb4ff;
    font-weight: 600;
    padding: 8px 14px;
}
"""


class GuiNode(Node):
    """Minimal rclpy node holding only the `/lowcmd` publisher."""

    def __init__(self) -> None:
        super().__init__("r2s_gui_controller")
        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)

    def publish_pose(self, pose: list[float]) -> None:
        """Publish 12 joint targets as a LowCmd (mode=0x01, dq=0, tau=0, default kp/kd).

        Sets the real-robot deploy header (head=0xFE 0xEF, level_flag=0xFF) and computes
        the LowCmd CRC via :mod:`lowcmd_crc`. The real GO2 firmware discards commands with
        an invalid CRC; the sim path (sim_bridge → UDP) ignores head/crc, so this is inert
        for sim but required for real deployment (CONTRACT.md §8).
        """
        msg = LowCmd()
        msg.head[0] = lowcmd_crc.HEAD[0]
        msg.head[1] = lowcmd_crc.HEAD[1]
        msg.level_flag = lowcmd_crc.LEVEL_FLAG_LOWLEVEL
        msg.gpio = 0
        for i in range(r2s_udp.NUM_MOTORS):
            motor = msg.motor_cmd[i]
            motor.mode = lowcmd_crc.MOTOR_MODE_SERVO
            motor.q = pose[i]
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = motions.DEFAULT_KP
            motor.kd = motions.DEFAULT_KD
        lowcmd_crc.set_crc(msg)
        self.lowcmd_pub.publish(msg)


class MainWindow(QMainWindow):
    """R2S-GO2 GUI main window."""

    def __init__(self, node: GuiNode) -> None:
        super().__init__()
        self._node = node
        # Start from the prone/folded pose (STAND_FOLDED), matching the sim's prone spawn
        # and the real GO2's lying start. GuiNode is open-loop (no /lowstate), so the
        # continuous publisher streams this from t=0 — it MUST be prone, otherwise the
        # robot would stand up the instant gui+bridge connect, before any button press.
        self._current_pose: list[float] = list(motions.STAND_FOLDED)
        # Command target streamed continuously to /lowcmd (real GO2 watchdog needs a
        # steady heartbeat, not one-shot publishes). Motion ticks update this; the
        # publish timer re-sends it at PUBLISH_HZ regardless of activity.
        self._latest_pose: list[float] = list(motions.STAND_FOLDED)

        # Sequence playback state (shared by default/step/sit)
        self._sequence: list[list[float]] = []
        self._sequence_idx: int = 0
        self._sequence_timer = QTimer(self)
        self._sequence_timer.timeout.connect(self._on_sequence_tick)

        # Sine sweep state
        self._sine_timer = QTimer(self)
        self._sine_timer.timeout.connect(self._on_sine_tick)
        self._sine_base_pose: list[float] = list(motions.DEFAULT_POSE)
        self._sine_start_time: float = 0.0

        # Monitor 는 **별도 프로세스**로 spawn 한다 (렌더를 gui heartbeat 에서 격리; CONTRACT §11).
        # None=미실행. subprocess.Popen 핸들.
        self._monitor_proc: subprocess.Popen | None = None

        self.setWindowTitle("R2S-GO2 Controller")
        self._build_ui()

        # Continuous /lowcmd publisher — always streams self._latest_pose (heartbeat).
        self._publish_timer = QTimer(self)
        self._publish_timer.timeout.connect(self._on_publish_tick)
        self._publish_timer.start(int(PUBLISH_PERIOD_S * 1000))

        # rclpy spin_once polling
        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._on_spin_tick)
        self._spin_timer.start(int(SPIN_PERIOD_S * 1000))

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        # -- Header --
        title_label = QLabel("R2S-GO2 Controller")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("Unitree GO2 — real2sim /lowcmd publisher")
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

        # -- Default / Sit --
        pose_group = QGroupBox("Pose")
        pose_layout = QHBoxLayout(pose_group)
        pose_layout.setSpacing(10)
        standup_btn = QPushButton("Stand Up")
        standup_btn.setObjectName("primaryButton")
        standup_btn.clicked.connect(self._on_standup_clicked)
        default_btn = QPushButton("Default (stand)")
        default_btn.clicked.connect(self._on_default_clicked)
        sit_btn = QPushButton("Sit")
        sit_btn.clicked.connect(self._on_sit_clicked)
        pose_layout.addWidget(standup_btn)
        pose_layout.addWidget(default_btn)
        pose_layout.addWidget(sit_btn)
        pose_layout.addStretch(1)
        layout.addWidget(pose_group)

        # -- Joint step --
        step_group = QGroupBox("Joint Step")
        step_layout = QHBoxLayout(step_group)
        step_layout.setSpacing(10)
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

        # -- Sine sweep --
        sine_group = QGroupBox("Sine Sweep")
        sine_layout = QHBoxLayout(sine_group)
        sine_layout.setSpacing(10)
        sine_layout.addWidget(QLabel("Joint:"))
        self._sine_joint_combo = QComboBox()
        self._sine_joint_combo.addItems(motions.JOINT_NAMES)
        sine_layout.addWidget(self._sine_joint_combo)
        sine_layout.addWidget(QLabel("Amplitude [rad]:"))
        self._sine_amp_spin = QDoubleSpinBox()
        self._sine_amp_spin.setRange(0.0, 1.5)
        self._sine_amp_spin.setSingleStep(0.05)
        self._sine_amp_spin.setValue(0.2)
        sine_layout.addWidget(self._sine_amp_spin)
        sine_layout.addWidget(QLabel("Frequency [Hz]:"))
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

        layout.addStretch(1)

        # -- Status bar --
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

    # -- Shared sequence playback --

    def _play_poses(self, seq: list[list[float]]) -> None:
        """Play a pre-built pose-frame sequence via the streaming publisher (also stops sine)."""
        self._stop_sine()
        self._sequence_timer.stop()
        self._sequence = seq
        self._sequence_idx = 0
        self._sequence_timer.start(int(FRAME_PERIOD_S * 1000))

    def _play_sequence_to(self, goal_pose: list[float], duration_s: float) -> None:
        """Interpolate from the current pose to goal_pose over duration_s (also stops sine)."""
        num_steps = max(1, int(duration_s * FRAME_HZ))
        self._play_poses(motions.interpolate_sequence(self._current_pose, goal_pose, num_steps))

    def _on_sequence_tick(self) -> None:
        if self._sequence_idx >= len(self._sequence):
            self._sequence_timer.stop()
            return
        pose = self._sequence[self._sequence_idx]
        # Update the streamed target; the publish timer sends it (do not publish here).
        self._latest_pose = pose
        self._current_pose = pose
        self._sequence_idx += 1

    # -- Pose buttons --

    def _on_standup_clicked(self) -> None:
        # go2_stand_example trajectory: current -> folded -> stand (hold). Same staged
        # method as the C++ example; ends standing (the example's final splay is omitted).
        self._status_label.setText("Standing up (go2_stand_example trajectory)...")
        self._play_poses(motions.stand_up_sequence(self._current_pose, FRAME_HZ))

    def _on_monitor_clicked(self) -> None:
        # 별도 프로세스로 monitor.py 를 spawn — 무거운 matplotlib 렌더를 gui 의 50Hz /lowcmd
        # heartbeat event loop 에서 완전히 격리한다 (CONTRACT §11). env(RMW/도메인/CYCLONEDDS_URI)
        # 는 자동 상속되어 같은 토픽을 관측한다. 이미 실행 중이면(살아있으면) 무시.
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._status_label.setText("Monitor already running")
            return
        monitor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor.py")
        self._monitor_proc = subprocess.Popen([sys.executable, monitor_py])
        self._status_label.setText("Monitor launched (separate process)")

    def _on_default_clicked(self) -> None:
        self._status_label.setText("Moving to default pose...")
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)

    def _on_sit_clicked(self) -> None:
        self._status_label.setText("Moving to sit pose...")
        self._play_sequence_to(motions.SIT_POSE, SEQUENCE_DURATION_S)

    def _on_step_clicked(self) -> None:
        joint_idx = self._step_joint_combo.currentIndex()
        delta = self._step_delta_spin.value()
        goal_pose = motions.step_pose(self._current_pose, joint_idx, delta)
        self._status_label.setText(f"Stepping {motions.JOINT_NAMES[joint_idx]} ({delta:+.2f} rad)...")
        self._play_sequence_to(goal_pose, STEP_DURATION_S)

    # -- Sine sweep --

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
        # Update the streamed target; the publish timer sends it (do not publish here).
        self._latest_pose = pose
        self._current_pose = pose

    def _on_sine_stop_clicked(self) -> None:
        self._stop_sine()
        self._status_label.setText("Sine sweep stopped")

    def _stop_sine(self) -> None:
        self._sine_timer.stop()
        self._sine_start_btn.setEnabled(True)
        self._sine_stop_btn.setEnabled(False)

    # -- continuous publisher --

    def _on_publish_tick(self) -> None:
        # Stream the current command target at PUBLISH_HZ. The real GO2 firmware treats a
        # gap in the command stream as a fault and drops to a protection mode, so we send
        # even when idle (holding the last pose). Inert to the sim path (bridge strips crc).
        if not rclpy.ok():
            self._publish_timer.stop()
            return
        self._node.publish_pose(self._latest_pose)

    # -- rclpy --

    def _on_spin_tick(self) -> None:
        # When the rclpy context is shut down (Ctrl+C/SIGTERM, etc.), spin_once keeps
        # raising on the invalid context, so guard and close the window to exit the Qt loop.
        if not rclpy.ok():
            self._spin_timer.stop()
            self.close()
            return
        try:
            # GuiNode has only the /lowcmd publisher (no subscriptions) — the monitor runs in
            # its own process, so nothing here competes with the publish heartbeat.
            rclpy.spin_once(self._node, timeout_sec=0.0)
        except KeyboardInterrupt:
            # Ctrl+C landed inside spin_once: close the window to exit the Qt loop cleanly.
            self._spin_timer.stop()
            self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._publish_timer.stop()
        self._spin_timer.stop()
        self._sequence_timer.stop()
        self._sine_timer.stop()
        # monitor 자식 프로세스가 살아있으면 정리 (orphan 방지).
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._monitor_proc.terminate()
        super().closeEvent(event)


def main() -> None:
    rclpy.init()
    node = GuiNode()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow(node)
    window.show()

    exit_code = app.exec_()

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
