#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 tuner_gui — 물성 슬라이더를 UDP(:9875)로 sim_runner_tuner에 흘려보내는 PyQt5 GUI (Phase 2.5).

전 12관절 공통 스칼라 6개(armature/viscous/coulomb/kp/kd/delay)를 실시간으로 sim에 반영해
녹화 실기 궤적과 겹쳐 보며 **bounds와 CMA-ES 초기 분포**를 잡는 도구다. per-joint 49개는 CMA-ES 몫.

순수 UDP(rclpy 불필요) — 시스템 python3.10 또는 Isaac conda 어느 쪽에서도 PyQt5만 있으면 실행된다::

    python scripts/real2sim/r2s_go2/tuner_gui.py

"Launch Plot" 버튼은 tuner_monitor.py를 **별도 프로세스**로 띄운다(무거운 렌더를 이 event loop에서
격리 — monitor.py와 동일 패턴, CONTRACT §11).

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md §4.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import r2s_udp  # noqa: E402
from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

HOST: str = "127.0.0.1"
PUBLISH_HZ: float = 30.0  # 현재 슬라이더 값을 주기적으로 재전송(latest-wins라 runner가 뒤늦게 떠도 따라잡음).
_SLIDER_STEPS: int = 1000  # QSlider(정수)를 실수 범위로 매핑하는 해상도.

# (키, 라벨, 최소, 최대, 기본, 소수자리) — 범위는 r2s_go2_sysid_cfg의 bounds와 정합.
_PARAM_SPECS: list[tuple[str, str, float, float, float, int]] = [
    ("armature", "Armature [kg·m²]", 1e-5, 0.5, 0.01, 5),
    ("viscous", "Viscous [N·m·s/rad]", 0.0, 2.0, 0.0, 4),
    ("coulomb", "Coulomb [N·m]", 0.0, 1.5, 0.0, 4),
    ("kp", "Kp [N·m/rad]", 0.0, 100.0, 25.0, 2),
    ("kd", "Kd [N·m·s/rad]", 0.0, 5.0, 0.5, 3),
    ("delay", "Delay [sim step]", 0.0, 10.0, 0.0, 0),
]

_STYLESHEET: str = """
QWidget { background-color: #1c1e26; color: #e6e8ef; font-family: "Segoe UI", "Ubuntu", sans-serif; font-size: 13px; }
QMainWindow { background-color: #1c1e26; }
#titleLabel { font-size: 18px; font-weight: 600; color: #f4f6fb; }
#subtitleLabel { font-size: 11px; color: #8a8fa3; }
QGroupBox { background-color: #242733; border: 1px solid #33364a; border-radius: 8px; margin-top: 14px;
            padding: 14px 10px 10px 10px; font-weight: 600; color: #c7cbe0; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px;
                   color: #9fb4ff; }
QLabel { color: #c7cbe0; background: transparent; }
QPushButton { background-color: #2c2f3d; color: #e6e8ef; border: 1px solid #3d4157; border-radius: 6px;
              padding: 7px 16px; font-weight: 500; }
QPushButton:hover { background-color: #363a4c; border-color: #4a4f6b; }
QPushButton#primaryButton { background-color: #4f7dfb; border-color: #4f7dfb; color: #ffffff; }
QPushButton#primaryButton:hover { background-color: #6b93fc; }
QDoubleSpinBox { background-color: #1c1e26; border: 1px solid #3d4157; border-radius: 5px; padding: 4px 8px;
                 min-height: 20px; min-width: 90px; }
QSlider::groove:horizontal { height: 5px; background: #33364a; border-radius: 2px; }
QSlider::handle:horizontal { background: #4f7dfb; width: 15px; margin: -6px 0; border-radius: 7px; }
#statusBar { background-color: #242733; border: 1px solid #33364a; border-radius: 8px; }
#statusLabel { color: #9fb4ff; font-weight: 600; padding: 8px 14px; }
"""


class ParamRow:
    """슬라이더 + 스핀박스 한 쌍을 양방향 동기화하는 파라미터 컨트롤."""

    def __init__(self, key: str, label: str, lo: float, hi: float, default: float, decimals: int) -> None:
        self.key = key
        self._lo = lo
        self._hi = hi
        self._syncing = False  # 슬라이더<->스핀박스 상호 갱신 중 무한루프 방지.

        self.label = QLabel(label)
        self.label.setMinimumWidth(150)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(max((hi - lo) / 100.0, 10.0 ** (-decimals)))
        self.spin.setValue(default)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, _SLIDER_STEPS)
        self.slider.setValue(self._to_slider(default))

        self.spin.valueChanged.connect(self._on_spin)
        self.slider.valueChanged.connect(self._on_slider)

    def _to_slider(self, value: float) -> int:
        frac = 0.0 if self._hi == self._lo else (value - self._lo) / (self._hi - self._lo)
        return int(round(frac * _SLIDER_STEPS))

    def _to_value(self, tick: int) -> float:
        return self._lo + (tick / _SLIDER_STEPS) * (self._hi - self._lo)

    def _on_spin(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(self._to_slider(value))
        self._syncing = False

    def _on_slider(self, tick: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(self._to_value(tick))
        self._syncing = False

    def value(self) -> float:
        return self.spin.value()

    def add_to(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.label)
        row.addWidget(self.slider, stretch=1)
        row.addWidget(self.spin)
        layout.addLayout(row)


class MainWindow(QMainWindow):
    """물성 슬라이더 -> UDP param 발행 GUI."""

    def __init__(self) -> None:
        super().__init__()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (HOST, r2s_udp.TUNER_PARAM_PORT)
        self._seq = 0
        self._monitor_proc: subprocess.Popen | None = None

        self.setWindowTitle("R2S-GO2 Tuner")
        self._rows: dict[str, ParamRow] = {}
        self._build_ui()

        # 현재 값을 PUBLISH_HZ로 계속 재전송 — runner가 나중에 떠도 최신 값을 받는다(latest-wins).
        self._publish_timer = QTimer(self)
        self._publish_timer.timeout.connect(self._on_publish_tick)
        self._publish_timer.start(int(1000.0 / PUBLISH_HZ))

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        title = QLabel("R2S-GO2 Tuner")
        title.setObjectName("titleLabel")
        subtitle = QLabel("전역 물성 실시간 주입 — bounds/초기분포 확정용 (per-joint는 CMA-ES)")
        subtitle.setObjectName("subtitleLabel")
        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(title)
        header.addWidget(subtitle)
        header_row = QHBoxLayout()
        header_row.addLayout(header)
        header_row.addStretch(1)
        plot_btn = QPushButton("Launch Plot")
        plot_btn.setObjectName("primaryButton")
        plot_btn.clicked.connect(self._on_plot_clicked)
        header_row.addWidget(plot_btn, alignment=Qt.AlignTop)
        layout.addLayout(header_row)

        param_group = QGroupBox("Physical Parameters (all 12 joints)")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(10)
        for key, label, lo, hi, default, decimals in _PARAM_SPECS:
            row = ParamRow(key, label, lo, hi, default, decimals)
            row.add_to(param_layout)
            self._rows[key] = row
        layout.addWidget(param_group)

        reset_btn = QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._on_reset_clicked)
        layout.addWidget(reset_btn, alignment=Qt.AlignLeft)

        layout.addStretch(1)

        status_bar = QWidget()
        status_bar.setObjectName("statusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(4, 2, 4, 2)
        self._status_label = QLabel(f"Publishing to :{r2s_udp.TUNER_PARAM_PORT} @ {PUBLISH_HZ:.0f} Hz")
        self._status_label.setObjectName("statusLabel")
        status_layout.addWidget(self._status_label)
        layout.addWidget(status_bar)

        self.setCentralWidget(central)

    def _on_publish_tick(self) -> None:
        self._seq += 1
        packet = r2s_udp.pack_tuner_params(
            self._seq,
            self._rows["armature"].value(),
            self._rows["viscous"].value(),
            self._rows["coulomb"].value(),
            self._rows["kp"].value(),
            self._rows["kd"].value(),
            self._rows["delay"].value(),
        )
        self._sock.sendto(packet, self._addr)

    def _on_reset_clicked(self) -> None:
        for key, _label, _lo, _hi, default, _decimals in _PARAM_SPECS:
            self._rows[key].spin.setValue(default)
        self._status_label.setText("Reset to defaults")

    def _on_plot_clicked(self) -> None:
        # tuner_monitor.py를 별도 프로세스로 spawn — 무거운 matplotlib 렌더 격리(monitor.py 패턴).
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._status_label.setText("Plot already running")
            return
        monitor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuner_monitor.py")
        self._monitor_proc = subprocess.Popen([sys.executable, monitor_py])
        self._status_label.setText("Plot launched (separate process)")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._publish_timer.stop()
        self._sock.close()
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._monitor_proc.terminate()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow()
    window.resize(560, 420)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
