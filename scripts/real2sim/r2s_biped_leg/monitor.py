#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg monitor — 선택 관절의 action/sim 을 실시간 plot 하는 **별도 프로세스**.

gui_controller의 "Monitor" 버튼이 이 스크립트를 별도 프로세스로 spawn한다. GUI가 중계하는
UDP 패킷(MONITOR_PORT, action+sim time-aligned)을 받아 관절별 3 plot을 그린다:

    - q  : action(명령, 점선) / sim(엔코더)
    - tau_est : sim
    - dq : sim

관절 콤보에는 HL 4관절 + HR 4관절이 모두 나오며, 한 번에 한 관절만 그린다.
(robot 시리즈는 ROS2 실로봇 연동 시 추가.)

설계 — 별도 프로세스 + full redraw (r2s_go2에서 확립한 교훈):
    matplotlib 렌더를 GUI event loop에 두면 50Hz UDP 발행을 굶긴다. 별도 프로세스로 격리하고,
    blit 대신 draw_idle() full redraw를 쓴다(blit on-screen 합성이 일부 X 환경에서 창을 까맣게
    남기는 문제가 있었음).
"""

from __future__ import annotations

import collections
import contextlib
import os
import socket
import sys
import time

import matplotlib

matplotlib.use("Qt5Agg")  # pyplot 미임포트 (Qt 임베드)
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

sys.path.insert(0, os.path.dirname(__file__))
import motions  # noqa: E402
import r2s_udp  # noqa: E402

HOST: str = "127.0.0.1"
WINDOW_S: float = 10.0
RENDER_HZ: float = 15.0
RENDER_PERIOD_S: float = 1.0 / RENDER_HZ
RECV_PERIOD_S: float = 0.005  # UDP drain 주기
RESCALE_PERIOD_S: float = 1.0
Y_MIN_SPAN: float = 0.2  # y축 최소 폭 — 정지 중 노이즈가 화면 가득 확대되는 것을 막는다
BUF_MAXLEN: int = 1200  # WINDOW_S × 발행율(50Hz) 여유

_C_ACTION = "#f0a020"
_C_SIM = "#4f9dfb"


class MonitorWindow(QMainWindow):
    """선택 관절의 action/sim 실시간 plot 창."""

    def __init__(self, joint_names: list[str]) -> None:
        super().__init__()
        self._joint_names = joint_names
        self._sel_idx: int = 0
        self._paused: bool = False
        # (t, action_q, sim_q, sim_dq, sim_tau) — 선택 관절만.
        self._buf: collections.deque = collections.deque(maxlen=BUF_MAXLEN)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((HOST, r2s_udp.MONITOR_PORT))
        self._sock.setblocking(False)

        self.setWindowTitle("R2S-BipedLeg Monitor")
        self.resize(720, 720)
        self._build_ui()
        self._build_plot()

        self._recv_timer = QTimer(self)
        self._recv_timer.timeout.connect(self._on_recv_tick)
        self._recv_timer.start(int(RECV_PERIOD_S * 1000))

        self._last_rescale: float = 0.0
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._on_render_tick)
        self._render_timer.start(int(RENDER_PERIOD_S * 1000))

    # -- UI --

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Joint:"))
        self._joint_combo = QComboBox()
        self._joint_combo.addItems(self._joint_names)  # HL 4개 + HR 4개
        self._joint_combo.currentIndexChanged.connect(self._on_joint_changed)
        ctrl.addWidget(self._joint_combo)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        ctrl.addWidget(self._pause_btn)
        ctrl.addStretch(1)
        ctrl.addWidget(QLabel(f"UDP :{r2s_udp.MONITOR_PORT}  (action + sim)"))
        root.addLayout(ctrl)

        self._canvas_holder = QVBoxLayout()
        root.addLayout(self._canvas_holder, 1)

        self._value_label = QLabel("—")
        root.addWidget(self._value_label)
        self.setCentralWidget(central)

    def _build_plot(self) -> None:
        self._fig = Figure(figsize=(7, 6), facecolor="#1c1e26")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas_holder.addWidget(self._canvas)

        # 초기 ylim은 실측 envelope 기준(soft limit |q|<1.91, tau_max 56, v_max 29.6).
        # 1초마다 데이터 기반으로 재스케일되므로 대략만 맞으면 된다.
        specs = [
            ("Joint position q [rad]", (-2.0, 2.0)),
            ("Torque tau_est [Nm]", (-60.0, 60.0)),
            ("Joint velocity dq [rad/s]", (-30.0, 30.0)),
        ]
        self._axes = []
        self._lines: list[dict] = []
        for row, (ylabel, ylim) in enumerate(specs):
            ax = self._fig.add_subplot(3, 1, row + 1)
            ax.set_facecolor("#242733")
            ax.set_xlim(-WINDOW_S, 0.0)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel, color="#c7cbe0", fontsize=9)
            ax.tick_params(colors="#8a8fa3", labelsize=8)
            ax.grid(True, color="#33364a", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color("#33364a")
            lines = {}
            if row == 0:
                (lines["action"],) = ax.plot([], [], "--", color=_C_ACTION, lw=1.4, label="action")
            (lines["sim"],) = ax.plot([], [], "-", color=_C_SIM, lw=1.4, label="sim")
            ax.legend(loc="upper left", fontsize=8, facecolor="#242733", edgecolor="#33364a", labelcolor="#c7cbe0")
            self._axes.append(ax)
            self._lines.append(lines)
        self._axes[-1].set_xlabel("time [s] (0 = now)", color="#c7cbe0", fontsize=9)
        self._fig.tight_layout()

    # -- UDP 수신 --

    def _on_recv_tick(self) -> None:
        if self._paused:
            # 큐는 계속 비워 backlog 방지
            while True:
                try:
                    self._sock.recvfrom(4096)
                except (BlockingIOError, OSError):
                    break
            return
        now = time.monotonic()
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
            except (BlockingIOError, OSError):
                break
            m = r2s_udp.unpack_monitor(data)
            if m is None:
                continue
            i = self._sel_idx
            self._buf.append((now, m["action_q"][i], m["sim_q"][i], m["sim_dq"][i], m["sim_tau"][i]))

    # -- UI 콜백 --

    def _on_joint_changed(self, idx: int) -> None:
        self._sel_idx = idx
        self._buf.clear()

    def _on_pause_clicked(self) -> None:
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")

    # -- 렌더 (full redraw) --

    def _rescale_axes(self) -> None:
        now = time.monotonic()

        def _yrange(*cols):
            """창 안 값의 p1~p99 로 y 범위를 잡는다 (r2s_go2 monitor에서 확립한 교훈).

            min/max 를 쓰면 한 번의 스파이크가 10초 내내 축을 붙잡는다 — 특히 dq 는 스텝 전환
            순간의 바늘이 축을 넓혀 정작 보려는 신호 본체가 바닥에 붙은 직선처럼 보인다.
            백분위로 자르면 스파이크는 축 밖으로 나가고 신호 본체가 화면을 채운다.
            """
            vals = []
            for row in self._buf:
                if now - row[0] <= WINDOW_S:
                    for c in cols:
                        vals.append(row[c])
            if not vals:
                return None
            vals.sort()
            n = len(vals)
            lo, hi = vals[int(0.01 * (n - 1))], vals[int(0.99 * (n - 1))]
            if hi - lo < Y_MIN_SPAN:  # 정지 중 — 0 주변에서 노이즈가 확대되지 않게 바닥을 깐다
                mid = 0.5 * (lo + hi)
                lo, hi = mid - 0.5 * Y_MIN_SPAN, mid + 0.5 * Y_MIN_SPAN
            m = 0.1 * (hi - lo)
            return lo - m, hi + m

        r = _yrange(1, 2)  # q: action + sim
        if r:
            self._axes[0].set_ylim(*r)
        r = _yrange(4)  # tau
        if r:
            self._axes[1].set_ylim(*r)
        r = _yrange(3)  # dq
        if r:
            self._axes[2].set_ylim(*r)

    def _set_line(self, line, col: int, now: float) -> None:
        if line is None:
            return
        xs, ys = [], []
        for row in self._buf:
            dt = row[0] - now
            if dt < -WINDOW_S:
                continue
            xs.append(dt)
            ys.append(row[col])
        line.set_data(xs, ys)

    def _on_render_tick(self) -> None:
        now = time.monotonic()
        self._set_line(self._lines[0].get("action"), 1, now)
        self._set_line(self._lines[0].get("sim"), 2, now)
        self._set_line(self._lines[1].get("sim"), 4, now)
        self._set_line(self._lines[2].get("sim"), 3, now)

        if (now - self._last_rescale) >= RESCALE_PERIOD_S:
            self._last_rescale = now
            self._rescale_axes()

        self._canvas.draw_idle()
        self._update_value_label()

    def _update_value_label(self) -> None:
        name = self._joint_names[self._sel_idx]
        if not self._buf:
            self._value_label.setText(f"[{name}]  (no data)")
            return
        _, aq, sq, sdq, stau = self._buf[-1]
        self._value_label.setText(f"[{name}]   action q={aq:+.3f}   sim q={sq:+.3f} dq={sdq:+.2f} τ={stau:+.2f}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._recv_timer.stop()
        self._render_timer.stop()
        with contextlib.suppress(Exception):
            self._sock.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QWidget{background-color:#1c1e26;color:#e6e8ef;font-size:13px;}"
        "QLabel{color:#c7cbe0;}"
        "QPushButton{background-color:#2c2f3d;color:#e6e8ef;border:1px solid #3d4157;"
        "border-radius:6px;padding:6px 14px;}QPushButton:hover{background-color:#363a4c;}"
        "QComboBox{background-color:#242733;border:1px solid #3d4157;border-radius:5px;padding:4px 8px;}"
    )
    win = MonitorWindow(motions.JOINT_NAMES)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
