#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 tuner_monitor — tuner 텔레메트리(:9876)를 받아 q를 오버레이하고 RMSE를 보여주는 독립 프로세스.

tuner_gui의 "Launch Plot" 버튼이 이 스크립트를 별도 프로세스로 spawn 한다(무거운 렌더 격리 —
monitor.py와 동일 패턴, CONTRACT §11). 데이터 소스는 sim_runner_tuner가 UDP로 보내는 tuner 텔레메트리:

    - cmd  : 재생 중인 명령 관절각 q_cmd (점선).
    - sim  : 현재 sim 관절각 q_sim (실선 파랑) — 슬라이더로 물성을 바꾸면 이게 실시간으로 변한다.
    - real : 녹화된 실기 관절각 q_real (실선 빨강) — ``--replay`` 없으면 표시 안 함.

두 개의 plot(위→아래): 관절각 q 오버레이 / 추종오차(sim−real 또는 sim−cmd). 하단에 롤링 RMSE 표시.

rclpy는 쓰지 않는다(순수 UDP). 계약: source/isaaclab_tasks/.../r2s_go2/CONTRACT.md §4.
"""

from __future__ import annotations

import collections
import math
import os
import socket
import sys
import time

import matplotlib

matplotlib.use("Qt5Agg")  # pyplot 임포트 금지 (Qt 임베드, 전역 상태 회피).
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2s_udp  # noqa: E402

# CONTRACT §2 JOINT_ORDER 순서 (FR/FL/RR/RL × hip,thigh,calf). 태스크 cfg를 임포트하지 않기 위해 로컬 정의.
JOINT_NAMES: list[str] = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]  # fmt: skip

HOST: str = "127.0.0.1"
WINDOW_S: float = 10.0  # 롤링 표시 창 [s]
RENDER_HZ: float = 15.0
RENDER_PERIOD_S: float = 1.0 / RENDER_HZ
RESCALE_PERIOD_S: float = 1.0
DECIM_HZ: float = 60.0  # 500Hz 텔레메트리를 plot용으로 다운샘플
DECIM_PERIOD_S: float = 1.0 / DECIM_HZ
RECV_PERIOD_S: float = 0.005
BUF_MAXLEN: int = 2000

_C_CMD = "#f0a020"
_C_SIM = "#4f9dfb"
_C_REAL = "#e0554f"

_STYLESHEET = """
QWidget { background-color: #1c1e26; color: #e6e8ef; font-size: 13px; }
QLabel { color: #c7cbe0; background: transparent; }
QPushButton { background-color: #2c2f3d; color: #e6e8ef; border: 1px solid #3d4157;
    border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { background-color: #363a4c; }
QComboBox { background-color: #242733; border: 1px solid #3d4157; border-radius: 5px; padding: 4px 8px; }
QComboBox QAbstractItemView { background-color: #242733; selection-background-color: #4f7dfb; }
"""


class TunerMonitorWindow(QMainWindow):
    """선택 관절의 q_cmd/q_sim/q_real 오버레이 + 추종오차 + 롤링 RMSE 창."""

    def __init__(self, param_port: int) -> None:
        super().__init__()
        self._sel_idx: int = 0
        self._paused: bool = False
        self._has_real: bool = False

        # ring buffer: (wall_t, q_cmd, q_sim, q_real) — 선택 관절만 보관.
        self._buf: collections.deque = collections.deque(maxlen=BUF_MAXLEN)
        self._last_accept: float = 0.0

        # UDP telem 수신 소켓 (non-blocking).
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((HOST, param_port))
        self._sock.setblocking(False)

        self.setWindowTitle("R2S-GO2 Tuner Monitor")
        self.resize(720, 620)
        self._build_ui()
        self._build_plot()

        self._recv_timer = QTimer(self)
        self._recv_timer.timeout.connect(self._on_recv_tick)
        self._recv_timer.start(int(RECV_PERIOD_S * 1000))

        self._last_rescale: float = 0.0
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._on_render_tick)
        self._render_timer.start(int(RENDER_PERIOD_S * 1000))

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)
        ctrl.addWidget(QLabel("Joint:"))
        self._joint_combo = QComboBox()
        self._joint_combo.addItems(JOINT_NAMES)
        self._joint_combo.currentIndexChanged.connect(self._on_joint_changed)
        ctrl.addWidget(self._joint_combo)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        ctrl.addWidget(self._pause_btn)
        ctrl.addStretch(1)
        self._rmse_label = QLabel("RMSE —")
        ctrl.addWidget(self._rmse_label)
        root.addLayout(ctrl)

        self._canvas_holder = QVBoxLayout()
        root.addLayout(self._canvas_holder, 1)

        self._value_label = QLabel("—")
        root.addWidget(self._value_label)

        self.setCentralWidget(central)

    def _build_plot(self) -> None:
        self._fig = Figure(figsize=(7, 5), facecolor="#1c1e26")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas_holder.addWidget(self._canvas)

        specs = [("Joint position q [rad]", (-3.5, 3.5)), ("Tracking error [rad]", (-0.5, 0.5))]
        self._axes = []
        self._lines: list[dict] = []
        for row, (ylabel, ylim) in enumerate(specs):
            ax = self._fig.add_subplot(2, 1, row + 1)
            ax.set_facecolor("#242733")
            ax.set_xlim(-WINDOW_S, 0.0)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel, color="#c7cbe0", fontsize=9)
            ax.tick_params(colors="#8a8fa3", labelsize=8)
            ax.grid(True, color="#33364a", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color("#33364a")
            lines: dict = {}
            if row == 0:
                (lines["cmd"],) = ax.plot([], [], "--", color=_C_CMD, lw=1.4, label="cmd")
                (lines["sim"],) = ax.plot([], [], "-", color=_C_SIM, lw=1.4, label="sim")
                (lines["real"],) = ax.plot([], [], "-", color=_C_REAL, lw=1.4, label="real")
            else:
                (lines["err"],) = ax.plot([], [], "-", color=_C_SIM, lw=1.4, label="sim−real")
            ax.legend(loc="upper left", fontsize=8, facecolor="#242733", edgecolor="#33364a", labelcolor="#c7cbe0")
            self._axes.append(ax)
            self._lines.append(lines)
        self._axes[-1].set_xlabel("time [s] (0 = now)", color="#c7cbe0", fontsize=9)
        self._fig.tight_layout()

    # -- UDP 수신 --

    def _on_recv_tick(self) -> None:
        # 큐를 배수(drain)하고 decimate. 선택 관절 성분만 ring buffer에 append.
        latest = None
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
            except BlockingIOError:
                break
            parsed = r2s_udp.unpack_tuner_telem(data)
            if parsed is not None:
                latest = parsed
        if latest is None or self._paused:
            return
        self._has_real = latest["has_real"]
        now = time.monotonic()
        if now - self._last_accept < DECIM_PERIOD_S:
            return
        self._last_accept = now
        i = self._sel_idx
        self._buf.append((now, latest["q_cmd"][i], latest["q_sim"][i], latest["q_real"][i]))

    # -- UI 콜백 --

    def _on_joint_changed(self, idx: int) -> None:
        self._sel_idx = idx
        self._buf.clear()

    def _on_pause_clicked(self) -> None:
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")

    # -- 렌더 (full redraw, monitor.py와 동일 근거) --

    def _on_render_tick(self) -> None:
        now = time.monotonic()

        # q 오버레이 (col 1=cmd, 2=sim, 3=real)
        self._set_line(self._lines[0]["cmd"], 1, now)
        self._set_line(self._lines[0]["sim"], 2, now)
        if self._has_real:
            self._set_line(self._lines[0]["real"], 3, now)
        else:
            self._lines[0]["real"].set_data([], [])
        # 추종오차 = sim − real(replay) 또는 sim − cmd
        self._set_error_line(self._lines[1]["err"], now)

        if (now - self._last_rescale) >= RESCALE_PERIOD_S:
            self._last_rescale = now
            self._rescale_axes(now)

        self._canvas.draw_idle()
        self._update_labels(now)

    def _set_line(self, line, col: int, now: float) -> None:
        xs, ys = [], []
        for row in self._buf:
            dt = row[0] - now
            if dt < -WINDOW_S:
                continue
            xs.append(dt)
            ys.append(row[col])
        line.set_data(xs, ys)

    def _set_error_line(self, line, now: float) -> None:
        ref_col = 3 if self._has_real else 1  # real 있으면 sim−real, 없으면 sim−cmd
        xs, ys = [], []
        for row in self._buf:
            dt = row[0] - now
            if dt < -WINDOW_S:
                continue
            xs.append(dt)
            ys.append(row[2] - row[ref_col])
        line.set_data(xs, ys)

    def _window_errors(self, now: float) -> list[float]:
        ref_col = 3 if self._has_real else 1
        return [row[2] - row[ref_col] for row in self._buf if (row[0] - now) >= -WINDOW_S]

    def _rescale_axes(self, now: float) -> None:
        qvals = []
        for row in self._buf:
            if (row[0] - now) >= -WINDOW_S:
                qvals.extend((row[1], row[2]))
                if self._has_real:
                    qvals.append(row[3])
        if qvals:
            lo, hi = min(qvals), max(qvals)
            if hi - lo < 1e-3:
                lo, hi = lo - 0.5, hi + 0.5
            margin = 0.1 * (hi - lo)
            self._axes[0].set_ylim(lo - margin, hi + margin)
        errs = self._window_errors(now)
        if errs:
            amax = max(abs(min(errs)), abs(max(errs)), 1e-3)
            self._axes[1].set_ylim(-1.2 * amax, 1.2 * amax)

    def _update_labels(self, now: float) -> None:
        name = JOINT_NAMES[self._sel_idx]
        errs = self._window_errors(now)
        if errs:
            rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
            ref = "sim−real" if self._has_real else "sim−cmd"
            self._rmse_label.setText(f"RMSE({ref}) {rmse:.4f} rad")
        else:
            self._rmse_label.setText("RMSE —")
        if self._buf:
            _, qc, qs, qr = self._buf[-1]
            parts = [f"[{name}]", f"cmd={qc:+.3f}", f"sim={qs:+.3f}"]
            if self._has_real:
                parts.append(f"real={qr:+.3f}")
            self._value_label.setText("   ".join(parts))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._render_timer.stop()
        self._recv_timer.stop()
        self._sock.close()
        super().closeEvent(event)


def main() -> None:
    """독립 프로세스 진입점. tuner_gui의 "Launch Plot" 버튼이 spawn 한다."""
    app = QApplication(sys.argv)
    app.setStyleSheet(_STYLESHEET)
    win = TunerMonitorWindow(r2s_udp.TUNER_TELEM_PORT)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
