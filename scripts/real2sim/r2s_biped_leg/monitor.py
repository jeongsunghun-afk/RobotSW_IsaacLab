#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg monitor — 선택 관절의 action/sim/real 을 실시간 plot 하는 **별도 프로세스**.

gui_controller의 "Monitor" 버튼이 이 스크립트를 별도 프로세스로 spawn한다. GUI가 중계하는
UDP 패킷(MONITOR_PORT, action+sim time-aligned)을 받아 관절별 3 plot을 그린다:

    - q  : action(명령, 점선) / sim(엔코더) / real(실기, 있으면)
    - tau_est : sim / real
    - dq : sim / real

주 소켓(``--port``, 기본 MONITOR_PORT)과 별개로 REAL_MON_PORT(9890, gui_controller의
RealMonitorThread가 중계하는 실기 텔레메트리)을 항상 추가로 수신해 real 시리즈로 겹쳐 그린다
(``--port``가 이미 REAL_MON_PORT면 구 단독 실행 호환을 위해 두 번째 소켓은 만들지 않는다).
q/dq는 컨트롤 행의 단위 토글로 rad↔deg 표시를 전환할 수 있다(내부 버퍼는 항상 rad).

관절 콤보에는 HL 4관절 + HR 4관절이 모두 나오며, 한 번에 한 관절만 그린다.

설계 — 별도 프로세스 + pyqtgraph (2026-08-12 matplotlib에서 교체):
    렌더를 GUI event loop에 두면 50Hz UDP 발행을 굶기므로 별도 프로세스로 격리한다(r2s_go2 교훈).
    plot 백엔드는 matplotlib full redraw가 X11에서 39ms/frame(실측)이라 15Hz 표시 지연 ~100ms의
    주범이었고, pyqtgraph 는 setData 증분 렌더라 수 ms/frame — 30Hz 로 표시 지연을 절반 이하로
    줄인다. (구 matplotlib 시절 blit 금지 사유였던 "일부 X 환경에서 창이 까맣게 남는" 문제는
    pyqtgraph 의 QGraphicsView 파이프라인과 무관하다.)
"""

from __future__ import annotations

import collections
import contextlib
import os
import socket
import sys
import time

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
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

pg.setConfigOptions(antialias=True, background="#1c1e26", foreground="#c7cbe0")

HOST: str = "127.0.0.1"
WINDOW_S: float = 10.0
RENDER_HZ: float = 30.0  # pyqtgraph 증분 렌더 기준 (matplotlib 시절 15Hz)
RENDER_PERIOD_S: float = 1.0 / RENDER_HZ
RECV_PERIOD_S: float = 0.005  # UDP drain 주기
RESCALE_PERIOD_S: float = 1.0
Y_MIN_SPAN: float = 0.2  # y축 최소 폭 — 정지 중 노이즈가 화면 가득 확대되는 것을 막는다
BUF_MAXLEN: int = 1200  # WINDOW_S × 발행율(50Hz) 여유

_C_ACTION = "#f0a020"
_C_SIM = "#4f9dfb"
_C_REAL = "#4fc26b"
RAD2DEG: float = 57.29578


class MonitorWindow(QMainWindow):
    """선택 관절의 action/sim/real 실시간 plot 창.

    ``port``/``label``/``show_action`` 으로 주 데이터 소스를 바꿔 재사용한다 — 기본은 sim 중계(9883,
    action 점선 포함), gui_controller 의 Real Robot Monitor 는 TELEM 중계(9890, action 없음)를 단독
    띄운다. 주 소켓이 9890이 아닌 한, REAL_MON_PORT(9890)을 항상 추가로 수신해 real 시리즈를 sim 위에
    겹쳐 그린다(주 소켓이 이미 9890이면 구 단독 실행 호환을 위해 두 번째 소켓을 만들지 않는다).
    """

    def __init__(
        self,
        joint_names: list[str],
        port: int | None = None,
        label: str = "sim",
        show_action: bool = True,
        title: str = "R2S-BipedLeg Monitor",
        joint: str | None = None,
    ) -> None:
        super().__init__()
        self._joint_names = joint_names
        self._init_joint = joint
        self._port = port if port is not None else r2s_udp.MONITOR_PORT
        self._label = label
        self._show_action = show_action
        self._sel_idx: int = 0
        self._paused: bool = False
        self._deg_units: bool = False
        # (t, action_q, sim_q, sim_dq, sim_tau) — 선택 관절만.
        self._buf: collections.deque = collections.deque(maxlen=BUF_MAXLEN)
        # (t, real_q, real_dq, real_tau) — REAL_MON_PORT 중계, action 없음.
        self._real_buf: collections.deque = collections.deque(maxlen=BUF_MAXLEN)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((HOST, self._port))
        self._sock.setblocking(False)

        # real 시리즈용 추가 소켓 — 주 소켓이 이미 REAL_MON_PORT면(구 단독 실행 호환) 만들지 않는다.
        self._real_sock: socket.socket | None = None
        if self._port != r2s_udp.REAL_MON_PORT:
            try:
                real_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                real_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                real_sock.bind((HOST, r2s_udp.REAL_MON_PORT))
                real_sock.setblocking(False)
                self._real_sock = real_sock
            except OSError as exc:
                print(f"[monitor] real socket bind failed on port {r2s_udp.REAL_MON_PORT}: {exc}", file=sys.stderr)
        self._show_real: bool = self._real_sock is not None

        self.setWindowTitle(title)
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
        if self._init_joint in self._joint_names:
            self._sel_idx = self._joint_names.index(self._init_joint)
            self._joint_combo.setCurrentIndex(self._sel_idx)
        self._joint_combo.currentIndexChanged.connect(self._on_joint_changed)
        ctrl.addWidget(self._joint_combo)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        ctrl.addWidget(self._pause_btn)
        self._units_btn = QPushButton("Units: rad")
        self._units_btn.clicked.connect(self._on_units_clicked)
        ctrl.addWidget(self._units_btn)
        ctrl.addStretch(1)
        src = f"action + {self._label}" if self._show_action else self._label
        udp_txt = f"UDP :{self._port}  ({src})"
        if self._show_real:
            udp_txt += f"  + real :{r2s_udp.REAL_MON_PORT}"
        ctrl.addWidget(QLabel(udp_txt))
        root.addLayout(ctrl)

        self._canvas_holder = QVBoxLayout()
        root.addLayout(self._canvas_holder, 1)

        self._value_label = QLabel("—")
        root.addWidget(self._value_label)
        self.setCentralWidget(central)

    def _build_plot(self) -> None:
        self._glw = pg.GraphicsLayoutWidget()
        self._canvas_holder.addWidget(self._glw)

        # 초기 y범위는 실측 envelope 기준(soft limit |q|<1.91, tau_max 56, v_max 29.6).
        # 1초마다 데이터 기반으로 재스케일되므로 대략만 맞으면 된다. q/dq 라벨은 단위 토글에 따라
        # _update_axis_labels()가 다시 채운다.
        specs = [
            ("Joint position q", (-2.0, 2.0)),
            ("Torque tau_est [Nm]", (-60.0, 60.0)),
            ("Joint velocity dq", (-30.0, 30.0)),
        ]
        self._axes: list[pg.PlotItem] = []
        self._lines: list[dict] = []
        for row, (ylabel, ylim) in enumerate(specs):
            plot = self._glw.addPlot(row=row, col=0)
            plot.setXRange(-WINDOW_S, 0.0, padding=0)
            plot.setYRange(*ylim, padding=0)
            plot.setLabel("left", ylabel)
            # SI 접두 자동 표기 OFF — 켜두면 0.5 rad 가 "500 (x0.001)" 로 표시된다 (실측 확인).
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=0.25)
            # 축은 재스케일 로직(p1~p99)이 소유 — 마우스 줌/자동범위로 어긋나지 않게 잠근다.
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.setMenuEnabled(False)
            if row > 0:
                plot.setXLink(self._axes[0])
            legend = plot.addLegend(offset=(6, 4), labelTextSize="8pt")
            legend.setBrush(pg.mkBrush("#242733"))
            lines = {}
            if row == 0 and self._show_action:
                lines["action"] = plot.plot(
                    [], [], pen=pg.mkPen(_C_ACTION, width=1.4, style=Qt.DashLine), name="action"
                )
            lines["sim"] = plot.plot([], [], pen=pg.mkPen(_C_SIM, width=1.4), name=self._label)
            if self._show_real:
                lines["real"] = plot.plot([], [], pen=pg.mkPen(_C_REAL, width=1.4), name="real")
            self._axes.append(plot)
            self._lines.append(lines)
        self._axes[-1].setLabel("bottom", "time [s] (0 = now)")
        self._update_axis_labels()

    def _update_axis_labels(self) -> None:
        """q/dq y축 라벨을 현재 단위 토글(rad↔deg)에 맞춰 갱신한다."""
        uq, udq = ("deg", "deg/s") if self._deg_units else ("rad", "rad/s")
        self._axes[0].setLabel("left", f"Joint position q [{uq}]")
        self._axes[2].setLabel("left", f"Joint velocity dq [{udq}]")

    # -- UDP 수신 --

    def _on_recv_tick(self) -> None:
        if self._paused:
            # 큐는 계속 비워 backlog 방지
            for sock in (self._sock, self._real_sock):
                if sock is None:
                    continue
                while True:
                    try:
                        sock.recvfrom(4096)
                    except (BlockingIOError, OSError):
                        break
            return
        now = time.monotonic()
        i = self._sel_idx
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
            except (BlockingIOError, OSError):
                break
            m = r2s_udp.unpack_monitor(data)
            if m is None:
                continue
            self._buf.append((now, m["action_q"][i], m["sim_q"][i], m["sim_dq"][i], m["sim_tau"][i]))
        if self._real_sock is not None:
            while True:
                try:
                    data, _ = self._real_sock.recvfrom(4096)
                except (BlockingIOError, OSError):
                    break
                m = r2s_udp.unpack_monitor(data)
                if m is None:
                    continue
                # real 중계는 action_q=0(무의미) — sim_q/sim_dq/sim_tau 필드에 실기 q/dq/tau가 담긴다.
                self._real_buf.append((now, m["sim_q"][i], m["sim_dq"][i], m["sim_tau"][i]))

    # -- UI 콜백 --

    def _on_joint_changed(self, idx: int) -> None:
        self._sel_idx = idx
        self._buf.clear()
        self._real_buf.clear()

    def _on_pause_clicked(self) -> None:
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")

    def _on_units_clicked(self) -> None:
        self._deg_units = not self._deg_units
        self._units_btn.setText("Units: deg" if self._deg_units else "Units: rad")
        self._update_axis_labels()
        self._rescale_axes()

    def _unit_scale(self) -> float:
        """q/dq 표시 배율 — 내부 버퍼는 항상 rad, 화면 표시만 deg 로 바꾼다."""
        return RAD2DEG if self._deg_units else 1.0

    def _unit_suffix(self) -> tuple[str, str]:
        return (" deg", " deg/s") if self._deg_units else (" rad", " rad/s")

    # -- 렌더 (full redraw) --

    def _rescale_axes(self) -> None:
        now = time.monotonic()
        scale = self._unit_scale()

        def _yrange(*buf_cols):
            """창 안 값의 p1~p99 로 y 범위를 잡는다 (r2s_go2 monitor에서 확립한 교훈).

            min/max 를 쓰면 한 번의 스파이크가 10초 내내 축을 붙잡는다 — 특히 dq 는 스텝 전환
            순간의 바늘이 축을 넓혀 정작 보려는 신호 본체가 바닥에 붙은 직선처럼 보인다.
            백분위로 자르면 스파이크는 축 밖으로 나가고 신호 본체가 화면을 채운다.
            ``buf_cols``: (buffer, col) 쌍 목록 — sim/real 버퍼를 함께 넣어 합산 범위를 잡는다.
            """
            vals = []
            for buf, c in buf_cols:
                for row in buf:
                    if now - row[0] <= WINDOW_S:
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

        q_cols = [(self._buf, 2)]  # sim q
        if self._show_action:
            q_cols.append((self._buf, 1))  # action q
        if self._show_real:
            q_cols.append((self._real_buf, 1))  # real q
        r = _yrange(*q_cols)
        if r:
            self._axes[0].setYRange(r[0] * scale, r[1] * scale, padding=0)

        tau_cols = [(self._buf, 4)]  # sim tau
        if self._show_real:
            tau_cols.append((self._real_buf, 3))  # real tau
        r = _yrange(*tau_cols)
        if r:
            self._axes[1].setYRange(r[0], r[1], padding=0)  # tau 는 단위 토글 영향 없음

        dq_cols = [(self._buf, 3)]  # sim dq
        if self._show_real:
            dq_cols.append((self._real_buf, 2))  # real dq
        r = _yrange(*dq_cols)
        if r:
            self._axes[2].setYRange(r[0] * scale, r[1] * scale, padding=0)

    def _set_line(self, line, buf, col: int, now: float, scale: float = 1.0) -> None:
        if line is None:
            return
        xs, ys = [], []
        for row in buf:
            dt = row[0] - now
            if dt < -WINDOW_S:
                continue
            xs.append(dt)
            ys.append(row[col] * scale)
        line.setData(xs, ys)

    def _on_render_tick(self) -> None:
        now = time.monotonic()
        scale = self._unit_scale()
        self._set_line(self._lines[0].get("action"), self._buf, 1, now, scale)
        self._set_line(self._lines[0].get("sim"), self._buf, 2, now, scale)
        self._set_line(self._lines[1].get("sim"), self._buf, 4, now)
        self._set_line(self._lines[2].get("sim"), self._buf, 3, now, scale)
        if self._show_real:
            self._set_line(self._lines[0].get("real"), self._real_buf, 1, now, scale)
            self._set_line(self._lines[1].get("real"), self._real_buf, 3, now)
            self._set_line(self._lines[2].get("real"), self._real_buf, 2, now, scale)

        if (now - self._last_rescale) >= RESCALE_PERIOD_S:
            self._last_rescale = now
            self._rescale_axes()

        # pyqtgraph 는 setData/setYRange 가 곧 갱신 트리거 — 명시적 draw 호출 불필요.
        self._update_value_label()

    def _update_value_label(self) -> None:
        name = self._joint_names[self._sel_idx]
        has_real = self._show_real and self._real_buf
        if not self._buf and not has_real:
            self._value_label.setText(f"[{name}]  (no data)")
            return
        scale = self._unit_scale()
        uq, udq = self._unit_suffix()
        parts = [f"[{name}]"]
        if self._buf:
            _, aq, sq, sdq, stau = self._buf[-1]
            act = f"action q={aq * scale:+.3f}{uq}   " if self._show_action else ""
            parts.append(f"{act}{self._label} q={sq * scale:+.3f}{uq} dq={sdq * scale:+.2f}{udq} τ={stau:+.2f}")
        if has_real:
            _, rq, rdq, rtau = self._real_buf[-1]
            parts.append(f"real q={rq * scale:+.3f}{uq} dq={rdq * scale:+.2f}{udq} τ={rtau:+.2f}")
        self._value_label.setText("   ".join(parts))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._recv_timer.stop()
        self._render_timer.stop()
        with contextlib.suppress(Exception):
            self._sock.close()
        if self._real_sock is not None:
            with contextlib.suppress(Exception):
                self._real_sock.close()
        super().closeEvent(event)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="R2S-BipedLeg 실시간 plot (별도 프로세스)")
    parser.add_argument("--port", type=int, default=r2s_udp.MONITOR_PORT, help="MON 패킷 수신 포트")
    parser.add_argument("--label", default="sim", help="실측 시리즈 레이블 (예: real)")
    parser.add_argument("--no_action", action="store_true", help="action 점선 숨김 (real 텔레메트리용)")
    parser.add_argument("--title", default="R2S-BipedLeg Monitor", help="창 제목")
    parser.add_argument("--joint", default=None, help="초기 선택 관절 이름 (예: HL_thigh)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QWidget{background-color:#1c1e26;color:#e6e8ef;font-size:13px;}"
        "QLabel{color:#c7cbe0;}"
        "QPushButton{background-color:#2c2f3d;color:#e6e8ef;border:1px solid #3d4157;"
        "border-radius:6px;padding:6px 14px;}QPushButton:hover{background-color:#363a4c;}"
        "QComboBox{background-color:#242733;border:1px solid #3d4157;border-radius:5px;padding:4px 8px;}"
    )
    win = MonitorWindow(
        motions.JOINT_NAMES,
        port=args.port,
        label=args.label,
        show_action=not args.no_action,
        title=args.title,
        joint=args.joint,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
