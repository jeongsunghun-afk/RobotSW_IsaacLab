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

Process architecture — /lowcmd publishing AND target generation (interpolation/sine) run in a
**separate process** (``publisher_process_main``), NOT in the Qt UI process. The real GO2 firmware
faults on a gap in the command stream, and Qt QTimers slip whenever the UI event loop is busy
(dragging a slider, holding a button). The UI writes only a *motion spec* (mode + start time +
goal/frame-buffer/sine params) to a shared-memory ``mp.Array``; the publisher process computes the
target from elapsed wall-clock time at a steady 50 Hz, completely decoupled from the UI event loop.
So user interaction can starve neither the heartbeat nor the target updates (an early version moved
only publishing out — hz stayed 50 but motion still stuttered because the UI motion timers that
*generate* the target were starved). Threading would not work — the GIL lets a busy UI callback
starve a publisher thread (same reason the monitor is a separate process, CONTRACT §11). Each
LowCmd carries the deploy header (head=0xFE 0xEF, level_flag=0xFF) and a valid CRC (see lowcmd_crc);
both are inert to the sim path (sim_bridge strips them) but required for real deployment (§8).

Contract: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md

Visual styling is a self-contained QSS stylesheet applied in :func:`main` (see
``_STYLESHEET`` below); it is purely cosmetic and does not affect any publish/timer logic.
"""

from __future__ import annotations

import math
import multiprocessing as mp
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
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from unitree_go.msg import LowCmd, LowState  # noqa: E402

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

# 시작 자세 접근(startup approach) — gui 기동 시 로봇의 실측 현재 자세를 읽어 거기서부터 시작
# 자세로 부드럽게 보간한다. 예전엔 STAND_FOLDED 를 t=0 부터 그대로 발행해 로봇이 현재 자세에서
# 툭 스냅했다(kp=25 로 강하게). 이제 첫 /lowstate 를 받을 때까지 발행을 보류하고, 받으면 보간한다.
STARTUP_DURATION_S: float = 2.5  # 현재 자세 → 시작 자세 보간 시간 [s] (느리게 = 안전)
STARTUP_ACQUIRE_TIMEOUT_S: float = 2.0  # 첫 /lowstate 대기 상한 [s]. 넘으면 실측 없이 진행
# 시작 자세 획득용 상태 토픽. shared(실기) 모드면 실로봇 /lowstate, sim 전용이면 sim_bridge /lowstate.
STARTUP_STATE_TOPIC: str = os.environ.get("R2S_STARTUP_STATE_TOPIC", "/lowstate")
CMD_TOPIC: str = os.environ.get("R2S_CMD_TOPIC", "/lowcmd")  # publisher 프로세스가 발행하는 명령 토픽

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


_NUM = r2s_udp.NUM_MOTORS

# ---------------------------------------------------------------------------
# 공유 메모리 레이아웃 (multiprocessing.Array('d')). UI는 **모션 스펙**(어디로 몇 초에 걸쳐
# 가라)만 쓰고, publisher 프로세스가 50Hz 루프에서 **경과 시간 기반으로 목표를 계산**해 발행한다.
# 이렇게 하면 UI event loop가 조작으로 밀려도 목표가 매끄럽게(50Hz 균일) 갱신된다 — 발행뿐 아니라
# 목표 생성까지 UI에서 격리 (CONTRACT §11 동일 논리). time.monotonic()은 리눅스에서 프로세스 간
# 공통(CLOCK_MONOTONIC)이라 UI가 찍은 start_time을 publisher가 그대로 쓸 수 있다.
# ---------------------------------------------------------------------------
_MODE_HOLD = 0  # 고정 자세 유지 (BASE_Q)
_MODE_SEQUENCE = 1  # 프레임 버퍼를 경과 시간으로 인덱싱 (보간/Stand/Sit/Step/StandUp)
_MODE_SINE = 2  # BASE_Q + 선택 관절에 사인 주입 (경과 시간 기반)

_SM_MODE = 0
_SM_KP = 1
_SM_KD = 2
_SM_CMD_VALID = 3  # 0/1: 유효한 명령이 있는가. 시작 자세 획득 전엔 0 → 발행 보류(스냅 방지)
_SM_START_TIME = 4  # 모션 시작 시각 (time.monotonic)
_SM_FRAME_HZ = 5  # 시퀀스 프레임 생성 주파수 (인덱싱용)
_SM_FRAME_COUNT = 6  # 시퀀스 프레임 수
_SM_SINE_JOINT = 7
_SM_SINE_AMP = 8
_SM_SINE_FREQ = 9
_SM_MEASURED_VALID = 10  # 0/1: publisher가 /lowstate를 받았는가
_SM_BASE_Q = 11  # 11..22: hold 자세 / sine 기준 자세
_SM_MEASURED_Q = 23  # 23..34: 로봇 실측 관절각 (publisher가 /lowstate에서 씀)
_SM_CURRENT_Q = 35  # 35..46: publisher의 현재 출력 목표 (publisher가 매 cycle 씀, UI가 읽어 보간 시작점으로)
_SM_FRAMES = 47  # 47..: 시퀀스 프레임 버퍼 (MAX_FRAMES × 12)
MAX_FRAMES = 512  # 512/50 = 10.24s 최대 시퀀스
_SM_LEN = _SM_FRAMES + MAX_FRAMES * _NUM


def _build_lowcmd(pose: list[float], kp: float, kd: float) -> LowCmd:
    """12 관절 목표를 LowCmd로 직렬화 (mode=0x01, dq=0, tau=0, deploy 헤더 + CRC).

    실기 GO2 펌웨어는 CRC가 틀린 명령을 폐기한다. sim 경로(sim_bridge → UDP)는 head/crc를
    무시하므로 sim엔 무해하지만 실기 배포엔 필수다 (CONTRACT.md §8).
    """
    msg = LowCmd()
    msg.head[0] = lowcmd_crc.HEAD[0]
    msg.head[1] = lowcmd_crc.HEAD[1]
    msg.level_flag = lowcmd_crc.LEVEL_FLAG_LOWLEVEL
    msg.gpio = 0
    for i in range(_NUM):
        motor = msg.motor_cmd[i]
        motor.mode = lowcmd_crc.MOTOR_MODE_SERVO
        motor.q = pose[i]
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = kp
        motor.kd = kd
    lowcmd_crc.set_crc(msg)
    return msg


def _compute_target(shared, now: float) -> tuple[list[float], float, float] | None:
    """공유 메모리의 모션 스펙 + 경과 시간으로 현재 목표 자세를 계산한다 (publisher 프로세스에서 호출).

    Returns:
        ``(pose, kp, kd)`` 또는 명령이 없으면(``CMD_VALID=0``) None.
    """
    with shared.get_lock():
        if shared[_SM_CMD_VALID] < 0.5:
            return None
        mode = int(shared[_SM_MODE])
        kp = shared[_SM_KP]
        kd = shared[_SM_KD]
        base = [shared[_SM_BASE_Q + i] for i in range(_NUM)]
        if mode == _MODE_SEQUENCE:
            start = shared[_SM_START_TIME]
            frame_hz = shared[_SM_FRAME_HZ]
            count = int(shared[_SM_FRAME_COUNT])
            idx = int((now - start) * frame_hz)
            idx = 0 if idx < 0 else (count - 1 if idx >= count else idx)  # 끝 프레임에서 홀드
            off = _SM_FRAMES + idx * _NUM
            pose = [shared[off + i] for i in range(_NUM)]
        elif mode == _MODE_SINE:
            start = shared[_SM_START_TIME]
            joint = int(shared[_SM_SINE_JOINT])
            amp = shared[_SM_SINE_AMP]
            freq = shared[_SM_SINE_FREQ]
            pose = list(base)
            pose[joint] = base[joint] + amp * math.sin(2.0 * math.pi * freq * (now - start))
        else:  # _MODE_HOLD
            pose = base
    return pose, kp, kd


def publisher_process_main(shared, stop_flag) -> None:
    """**별도 프로세스**: 모션 스펙에서 목표를 계산해 /lowcmd 50Hz 발행 + /lowstate 구독. Qt와 완전 독립.

    UI event loop(위젯 조작·드래그)가 아무리 바빠도 이 프로세스는 영향받지 않는다. UI는 모션 스펙만
    쓰고, 목표 보간/사인 생성은 여기서 50Hz 균일하게 하므로 **목표값도 매끄럽게** 갱신된다(발행뿐 아니라
    목표 생성까지 UI에서 격리). 스레드로는 GIL 때문에 안 된다(monitor 격리와 동일 — CONTRACT §11).
    /lowstate의 실측 관절각과 현재 출력 목표를 공유 메모리에 되써 UI가 읽을 수 있게 한다.

    Args:
        shared: ``multiprocessing.Array('d', _SM_LEN)`` — 위 레이아웃 상수 참고.
        stop_flag: ``multiprocessing.Value('i')`` — 1이면 루프 종료.
    """
    rclpy.init()
    node = rclpy.create_node("r2s_gui_publisher")
    pub = node.create_publisher(LowCmd, CMD_TOPIC, 10)

    def on_lowstate(msg: LowState) -> None:
        with shared.get_lock():
            for i in range(_NUM):
                shared[_SM_MEASURED_Q + i] = float(msg.motor_state[i].q)
            shared[_SM_MEASURED_VALID] = 1.0

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(LowState, STARTUP_STATE_TOPIC, on_lowstate, qos)

    period = PUBLISH_PERIOD_S
    next_t = time.monotonic()
    try:
        while not stop_flag.value and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            result = _compute_target(shared, now)
            if result is not None:  # 시작 자세 획득 전(CMD_VALID=0)엔 발행 보류
                pose, kp, kd = result
                pub.publish(_build_lowcmd(pose, kp, kd))
                with shared.get_lock():  # UI가 보간 시작점으로 읽도록 현재 목표를 되쓴다
                    for i in range(_NUM):
                        shared[_SM_CURRENT_Q + i] = pose[i]
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # 밀렸으면 리싱크(누적 드리프트 방지)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class MainWindow(QMainWindow):
    """R2S-GO2 GUI main window."""

    def __init__(self, shared) -> None:
        super().__init__()
        # 공유 메모리(mp.Array) — 목표를 여기 쓰면 publisher 프로세스가 50Hz로 발행한다.
        # UI가 잠깐 멈춰도(사용자 조작) publisher가 마지막 목표를 계속 스트림해 heartbeat 유지.
        self._shared = shared
        # 현재 자세 — 버튼(Stand/Sit/Step)의 보간 시작점. 실측 획득 전엔 STAND_FOLDED 를 기본값으로
        # 두어(버튼을 눌러도 안전) 이후 첫 /lowstate 로 실측 자세로 덮어쓴다.
        self._current_pose: list[float] = list(motions.STAND_FOLDED)
        # 명령을 한 번이라도 공유 메모리에 썼는가. **시작 자세 획득 전엔 None**이라 아직 아무것도 쓰지
        # 않는다(CMD_VALID=0) — 로봇이 켜지자마자 시작 자세로 툭 스냅하는 걸 막는다(_on_startup_tick).
        self._latest_pose: list[float] | None = None

        # Sine sweep 상태 — 모션 생성은 publisher가 하므로 여기선 실행 여부/기준 자세만 추적한다.
        self._sine_active: bool = False
        self._sine_base_pose: list[float] = list(motions.DEFAULT_POSE)
        self._sine_start_time: float = 0.0

        # Monitor 는 **별도 프로세스**로 spawn 한다 (렌더를 gui heartbeat 에서 격리; CONTRACT §11).
        # None=미실행. subprocess.Popen 핸들.
        self._monitor_proc: subprocess.Popen | None = None

        self.setWindowTitle("R2S-GO2 Controller")
        self._build_ui()

        # sine 실행 중 스핀박스/콤보를 바꾸면 publisher에 파라미터만 갱신(모션은 안 끊김).
        self._sine_joint_combo.currentIndexChanged.connect(self._on_sine_param_changed)
        self._sine_amp_spin.valueChanged.connect(self._on_sine_param_changed)
        self._sine_freq_spin.valueChanged.connect(self._on_sine_param_changed)

        # /lowcmd 발행 + 목표 생성(보간/사인)은 별도 publisher 프로세스가 담당한다(이 UI 프로세스엔
        # 발행/모션 타이머·rclpy spin 없음) — UI 조작이 heartbeat·목표 갱신을 굶기지 못하게 하는 핵심.
        # UI는 공유 메모리에 "모션 스펙"만 쓴다.

        # 시작 자세 접근 상태머신 — 첫 /lowstate 로 로봇 현재 자세를 잡아 시작 자세로 보간(스냅 방지).
        self._startup_done: bool = False
        self._startup_deadline: float = time.monotonic() + STARTUP_ACQUIRE_TIMEOUT_S
        self._startup_timer = QTimer(self)
        self._startup_timer.timeout.connect(self._on_startup_tick)
        self._startup_timer.start(int(SPIN_PERIOD_S * 1000))

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

    # -- 모션 스펙 쓰기 (공유 메모리 → publisher 프로세스) --
    #
    # UI는 "무엇을 하라"는 스펙만 쓰고, 목표 보간/사인은 publisher가 50Hz 균일하게 계산한다.
    # 그래서 UI 조작으로 이 UI 프로세스가 잠깐 밀려도 목표값은 매끄럽게(끊김 없이) 갱신된다.

    def _read_current_target(self) -> list[float]:
        """publisher가 현재 발행 중인 목표를 읽는다 — 새 모션의 보간 시작점(모션 중 눌러도 정확)."""
        with self._shared.get_lock():
            if self._shared[_SM_CMD_VALID] < 0.5:
                return list(self._current_pose)
            return [self._shared[_SM_CURRENT_Q + i] for i in range(_NUM)]

    def _write_hold(self, pose: list[float]) -> None:
        """고정 자세 유지 명령."""
        self._sine_active = False
        self._current_pose = list(pose)
        self._latest_pose = list(pose)
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_HOLD
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = motions.DEFAULT_KD
            for i in range(_NUM):
                self._shared[_SM_BASE_Q + i] = float(pose[i])
            self._shared[_SM_CMD_VALID] = 1.0

    def _write_sequence(self, seq: list[list[float]]) -> None:
        """프레임 시퀀스를 publisher가 경과 시간으로 재생하도록 공유 메모리에 쓴다(sine 정지)."""
        self._sine_active = False
        if len(seq) > MAX_FRAMES:
            seq = seq[:MAX_FRAMES]  # 상한 초과분은 잘라낸다(10s 이상 시퀀스는 없음)
        self._current_pose = list(seq[-1])  # 시퀀스 종료 자세 (다음 버튼 보간 시작점 근사)
        self._latest_pose = list(seq[-1])
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_SEQUENCE
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = motions.DEFAULT_KD
            self._shared[_SM_FRAME_HZ] = FRAME_HZ
            self._shared[_SM_FRAME_COUNT] = float(len(seq))
            self._shared[_SM_START_TIME] = time.monotonic()
            for f, frame in enumerate(seq):
                off = _SM_FRAMES + f * _NUM
                for i in range(_NUM):
                    self._shared[off + i] = float(frame[i])
            self._shared[_SM_CMD_VALID] = 1.0

    def _write_sine(self, base: list[float], joint: int, amp: float, freq: float) -> None:
        """선택 관절에 사인을 주입하는 모션을 publisher가 생성하도록 공유 메모리에 쓴다."""
        self._sine_active = True
        self._sine_base_pose = list(base)
        self._sine_start_time = time.monotonic()
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_SINE
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = motions.DEFAULT_KD
            for i in range(_NUM):
                self._shared[_SM_BASE_Q + i] = float(base[i])
            self._shared[_SM_SINE_JOINT] = float(joint)
            self._shared[_SM_SINE_AMP] = float(amp)
            self._shared[_SM_SINE_FREQ] = float(freq)
            self._shared[_SM_START_TIME] = self._sine_start_time
            self._shared[_SM_CMD_VALID] = 1.0

    def _update_sine_params(self, joint: int, amp: float, freq: float) -> None:
        """sine 실행 중 파라미터만 갱신(관절/진폭/주파수). UI가 바빠도 publisher는 이전 값으로 계속 돈다."""
        if not self._sine_active:
            return
        with self._shared.get_lock():
            self._shared[_SM_SINE_JOINT] = float(joint)
            self._shared[_SM_SINE_AMP] = float(amp)
            self._shared[_SM_SINE_FREQ] = float(freq)

    # -- Shared sequence playback --

    def _play_poses(self, seq: list[list[float]]) -> None:
        """Play a pre-built pose-frame sequence via the publisher process (also stops sine)."""
        self._write_sequence(seq)

    def _play_sequence_to(self, goal_pose: list[float], duration_s: float) -> None:
        """Interpolate from the current published target to goal_pose over duration_s (also stops sine)."""
        start = self._read_current_target()  # publisher의 실제 현재 목표 (모션 중 눌러도 정확)
        num_steps = max(1, int(duration_s * FRAME_HZ))
        self._play_poses(motions.interpolate_sequence(start, goal_pose, num_steps))

    # -- Pose buttons --

    def _on_standup_clicked(self) -> None:
        # go2_stand_example trajectory: current -> folded -> stand (hold). Same staged
        # method as the C++ example; ends standing (the example's final splay is omitted).
        self._status_label.setText("Standing up (go2_stand_example trajectory)...")
        self._play_poses(motions.stand_up_sequence(self._read_current_target(), FRAME_HZ))

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
        goal_pose = motions.step_pose(self._read_current_target(), joint_idx, delta)
        self._status_label.setText(f"Stepping {motions.JOINT_NAMES[joint_idx]} ({delta:+.2f} rad)...")
        self._play_sequence_to(goal_pose, STEP_DURATION_S)

    # -- Sine sweep (모션 생성은 publisher 프로세스가 함) --

    def _on_sine_start_clicked(self) -> None:
        base = self._read_current_target()  # 현재 목표를 사인 기준으로
        joint = self._sine_joint_combo.currentIndex()
        amp = self._sine_amp_spin.value()
        freq = self._sine_freq_spin.value()
        self._sine_start_btn.setEnabled(False)
        self._sine_stop_btn.setEnabled(True)
        self._status_label.setText(f"Sine sweep on {self._sine_joint_combo.currentText()}...")
        self._write_sine(base, joint, amp, freq)

    def _on_sine_param_changed(self) -> None:
        # sine 실행 중 스핀박스/콤보 변경 → publisher에 파라미터만 갱신(모션은 안 끊김).
        self._update_sine_params(
            self._sine_joint_combo.currentIndex(),
            self._sine_amp_spin.value(),
            self._sine_freq_spin.value(),
        )

    def _on_sine_stop_clicked(self) -> None:
        # 현재 사인 위치를 계산해 그 자세로 홀드(정지 시 base로 튀지 않게).
        joint = self._sine_joint_combo.currentIndex()
        amp = self._sine_amp_spin.value()
        freq = self._sine_freq_spin.value()
        t = time.monotonic() - self._sine_start_time
        pose = motions.sine_offset(self._sine_base_pose, joint, amp, freq, t)
        self._stop_sine()
        self._write_hold(pose)
        self._status_label.setText("Sine sweep stopped")

    def _stop_sine(self) -> None:
        self._sine_active = False
        self._sine_start_btn.setEnabled(True)
        self._sine_stop_btn.setEnabled(False)

    # -- continuous publisher --

    def _on_startup_tick(self) -> None:
        """gui 기동 시 로봇 현재 자세를 획득해 시작 자세로 보간(스냅 방지). 1회만 수행.

        첫 /lowstate 로 실측 관절각을 잡으면 그 자세를 먼저 홀드(무동작)한 뒤 STAND_FOLDED 로
        느리게 보간한다. 지정 시간 안에 상태를 못 받으면(sim/robot 미기동 등) 실측 없이 진행한다.
        사용자가 그 사이 버튼을 눌러 이미 발행이 시작됐으면(_latest_pose 존재) 관여하지 않는다.
        """
        if self._startup_done:
            self._startup_timer.stop()
            return
        if self._latest_pose is not None:  # 버튼이 이미 모션을 시작함 — startup 개입 안 함
            self._startup_done = True
            self._startup_timer.stop()
            return
        # publisher 프로세스가 /lowstate에서 써준 실측 관절각을 공유 메모리에서 읽는다.
        measured = None
        with self._shared.get_lock():
            if self._shared[_SM_MEASURED_VALID] >= 0.5:
                measured = [self._shared[_SM_MEASURED_Q + i] for i in range(_NUM)]
        if measured is not None:
            start = list(measured)
            status = "시작: 로봇 현재 자세에서 시작 자세로 천천히 이동 중..."
        elif time.monotonic() >= self._startup_deadline:
            start = list(motions.STAND_FOLDED)  # /lowstate 미수신 — 실측 없이 시작 자세로
            status = "시작 자세로 홀드 (현재 자세 미획득 — sim/robot 상태 확인)"
        else:
            return  # 아직 첫 상태 대기
        # 현재 자세(실측)를 보간 시작점으로 두고 STAND_FOLDED 로 시퀀스 재생 — 첫 프레임이 실측이라 스냅 없음.
        self._current_pose = list(start)
        self._startup_done = True
        self._startup_timer.stop()
        self._status_label.setText(status)
        self._play_sequence_to(motions.STAND_FOLDED, STARTUP_DURATION_S)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._startup_timer.stop()
        # monitor 자식 프로세스가 살아있으면 정리 (orphan 방지). publisher 프로세스는 main()이 정리한다.
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._monitor_proc.terminate()
        super().closeEvent(event)


def main() -> None:
    # 공유 메모리 + 발행 stop 플래그를 만들고, **QApplication 생성 전에** publisher 프로세스를
    # fork 한다 — 이 시점엔 rclpy.init()도 QApplication도 없어 자식이 깨끗한 상태에서 시작한다
    # (자식만 rclpy.init(); 부모=UI는 rclpy를 init하지 않는다). fork 후 자식은 Qt를 전혀 만지지 않는다.
    mp.set_start_method("fork", force=True)
    shared = mp.Array("d", _SM_LEN)  # 0으로 초기화 → CMD_VALID=0(발행 보류), MEASURED_VALID=0
    stop_flag = mp.Value("i", 0)
    publisher = mp.Process(target=publisher_process_main, args=(shared, stop_flag), daemon=True)
    publisher.start()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow(shared)
    window.show()

    exit_code = app.exec_()

    # publisher 프로세스 정리 — stop 플래그 → join → 안 죽으면 terminate.
    stop_flag.value = 1
    publisher.join(timeout=2.0)
    if publisher.is_alive():
        publisher.terminate()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
