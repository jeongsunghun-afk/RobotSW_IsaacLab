#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 monitor — 선택 모터의 action/sim/robot 를 실시간 plot 하는 **독립 프로세스**.

gui_controller 의 "Monitor" 버튼이 이 스크립트를 별도 프로세스로 spawn 한다. 데이터 소스(모터 i):

    - action : ``/lowcmd`` (LowCmd.motor_cmd[i].q) — GUI가 현재 발행 중인 목표각.
    - sim    : ``/sim/lowstate`` (LowState.motor_state[i]) — 시뮬레이터 엔코더.
    - robot  : ``/lowstate`` (LowState.motor_state[i]) — 실로봇 엔코더.

세 개의 plot(위→아래): 관절각 q / 추정 토크 tau_est / 관절 각속도 dq.

설계 — 왜 **별도 프로세스**인가 (``/lowcmd`` 50Hz heartbeat 보호, CONTRACT.md §8·§11):
    gui 의 발행 QTimer(50Hz)는 실로봇 firmware watchdog heartbeat 다. 스트림에 gap 이 생기면
    로봇이 protection fault 로 빠진다. monitor 를 gui 와 **같은 Qt event loop** 에 두면, 실제
    디스플레이의 무거운 matplotlib draw 가 발행 타이머를 막아 heartbeat 에 gap(측정: maxgap
    ~55ms)이 생긴다. matplotlib/Qt 가 GIL 을 쥐므로 background thread 로도 못 푼다. 그래서
    monitor 를 **완전히 다른 프로세스**로 격리한다 — gui 프로세스는 발행만 하고 렌더를 전혀 하지
    않는다. (env 상속으로 RMW/도메인/CYCLONEDDS_URI 를 물려받아 같은 토픽을 본다.)

프로세스 내부 렌더 비용도 최소화:
    1. 수집/렌더 분리 — ring buffer append 는 rclpy 콜백에서, canvas 갱신은 별도 QTimer.
    2. blitting — line artist 만 갱신, full ``canvas.draw()`` 는 리스케일/리사이즈 시(~1Hz)에만.
    3. x축 고정 — 각 점을 ``t_sample - t_now`` 로 그려 xlim 을 [-WINDOW, 0] 로 고정 → blit 성립.
    4. decimate — 실로봇 500Hz 는 plot 에 무의미하므로 :data:`DECIM_HZ` 로 다운샘플 + QoS depth=1
       latest-wins 로 큐 backlog 방지.

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md
"""

from __future__ import annotations

import collections
import contextlib
import os
import sys
import time

import matplotlib

matplotlib.use("Qt5Agg")  # pyplot 은 임포트하지 않는다 (Qt 임베드, 전역 상태 회피).
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
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from unitree_go.msg import LowCmd, LowState  # noqa: E402

NUM_MOTORS: int = 12
WINDOW_S: float = 10.0  # 롤링 표시 창 [s]
RENDER_HZ: float = 15.0  # canvas 갱신율 (낮게 유지)
RENDER_PERIOD_S: float = 1.0 / RENDER_HZ
RESCALE_PERIOD_S: float = 1.0  # y축 리스케일 + 배경 재캡처 주기 [s]
DECIM_HZ: float = 60.0  # 상태 토픽 다운샘플 상한 (500Hz 실로봇 → plot 부하 감소)
DECIM_PERIOD_S: float = 1.0 / DECIM_HZ
SPIN_PERIOD_S: float = 0.005  # rclpy spin 주기 (독립 프로세스라 heartbeat 무관)
BUF_MAXLEN: int = 2000  # WINDOW_S × DECIM_HZ 여유

# 시리즈 색 (action=점선, sim/robot=실선).
_C_ACTION = "#f0a020"
_C_SIM = "#4f9dfb"
_C_ROBOT = "#e0554f"

# 다크 테마 QSS (gui 와 일관). 위젯 chrome 만 — plot 색은 Figure/Axes facecolor 로 별도 지정.
_STYLESHEET = """
QWidget { background-color: #1c1e26; color: #e6e8ef; font-size: 13px; }
QLabel { color: #c7cbe0; background: transparent; }
QPushButton { background-color: #2c2f3d; color: #e6e8ef; border: 1px solid #3d4157;
    border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { background-color: #363a4c; }
QComboBox { background-color: #242733; border: 1px solid #3d4157; border-radius: 5px;
    padding: 4px 8px; }
QComboBox QAbstractItemView { background-color: #242733; selection-background-color: #4f7dfb; }
"""


def _resolve_topics() -> tuple[str, str, str]:
    """(cmd, sim, robot) 토픽명을 환경변수에서 해석.

    비-shared(sim 전용)에서는 sim_bridge 가 ``/lowstate`` 로 발행하므로 run_gui_controller.sh
    가 ``R2S_SIM_STATE_TOPIC=/lowstate`` 를 내보낸다. shared(sim+real)에서는
    sim=``/sim/lowstate``, robot=``/lowstate`` 로 분리된다. sim==robot 이면 실로봇이 없는
    것이므로 robot 구독을 dedup(생략)한다.
    """
    cmd = os.environ.get("R2S_CMD_TOPIC", "/lowcmd")
    sim = os.environ.get("R2S_SIM_STATE_TOPIC", "/sim/lowstate")
    robot = os.environ.get("R2S_ROBOT_STATE_TOPIC", "/lowstate")
    return cmd, sim, robot


class MonitorWindow(QMainWindow):
    """선택 모터의 action/sim/robot 실시간 plot 창."""

    def __init__(self, node, joint_names: list[str], parent=None) -> None:
        super().__init__(parent)
        self._node = node
        self._joint_names = joint_names
        self._sel_idx: int = 0
        self._paused: bool = False

        # 시리즈별 ring buffer. cmd=(t,q), state=(t,q,dq,tau). 콜백/렌더 모두 이 프로세스의
        # 단일 Qt 스레드에서 도므로 락 불필요.
        self._buf_cmd: collections.deque = collections.deque(maxlen=BUF_MAXLEN)
        self._buf_sim: collections.deque = collections.deque(maxlen=BUF_MAXLEN)
        self._buf_robot: collections.deque = collections.deque(maxlen=BUF_MAXLEN)
        # 상태 토픽 다운샘플용 마지막-수용 시각 (500Hz 실로봇 → DECIM_HZ 로 제한).
        self._last_sim_accept: float = 0.0
        self._last_robot_accept: float = 0.0

        self._cmd_topic, self._sim_topic, self._robot_topic = _resolve_topics()
        # sim==robot 이면 실로봇 부재 → robot 구독/plot 생략(라벨 뒤집힘 방지). _build_* 이 참조하므로
        # 구독 생성 전에 미리 결정한다.
        self._has_robot = self._robot_topic != self._sim_topic

        self.setWindowTitle("R2S-GO2 Monitor")
        self.resize(720, 720)
        self._build_ui()
        self._build_plot()

        # -- 구독 생성 (node 에 부착) — depth=1 latest-wins 로 큐 backlog 방지 --
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._subs = []
        self._subs.append(node.create_subscription(LowCmd, self._cmd_topic, self._on_cmd, qos))
        self._subs.append(node.create_subscription(LowState, self._sim_topic, self._on_sim, qos))
        if self._has_robot:
            self._subs.append(node.create_subscription(LowState, self._robot_topic, self._on_robot, qos))

        # -- rclpy spin 타이머 (이 프로세스 전용; gui heartbeat 와 무관) --
        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._on_spin_tick)
        self._spin_timer.start(int(SPIN_PERIOD_S * 1000))

        # -- 렌더 타이머 (수집과 분리) --
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
        ctrl.setSpacing(10)
        ctrl.addWidget(QLabel("Motor:"))
        self._motor_combo = QComboBox()
        self._motor_combo.addItems(self._joint_names)
        self._motor_combo.currentIndexChanged.connect(self._on_motor_changed)
        ctrl.addWidget(self._motor_combo)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        ctrl.addWidget(self._pause_btn)
        ctrl.addStretch(1)
        src = "sim + robot" if self._has_robot else "sim only"
        self._src_label = QLabel(f"cmd={self._cmd_topic}  sim={self._sim_topic}  ({src})")
        ctrl.addWidget(self._src_label)
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

        specs = [
            ("Joint position q [rad]", (-3.5, 3.5)),
            ("Torque tau_est [Nm]", (-40.0, 40.0)),
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
            if row == 0:  # q plot 에만 action(cmd) 오버레이
                (lines["action"],) = ax.plot([], [], "--", color=_C_ACTION, lw=1.4, label="action")
            (lines["sim"],) = ax.plot([], [], "-", color=_C_SIM, lw=1.4, label="sim")
            if self._has_robot:
                (lines["robot"],) = ax.plot([], [], "-", color=_C_ROBOT, lw=1.4, label="robot")
            ax.legend(loc="upper left", fontsize=8, facecolor="#242733", edgecolor="#33364a", labelcolor="#c7cbe0")
            self._axes.append(ax)
            self._lines.append(lines)
        self._axes[-1].set_xlabel("time [s] (0 = now)", color="#c7cbe0", fontsize=9)
        self._fig.tight_layout()

    # -- rclpy --

    def _on_spin_tick(self) -> None:
        # 이 프로세스의 구독을 배수(drain). heartbeat 와 무관하므로 여러 콜백을 몰아 처리.
        import rclpy

        if not rclpy.ok():
            self._spin_timer.stop()
            self.close()
            return
        for _ in range(16):
            rclpy.spin_once(self._node, timeout_sec=0.0)

    # -- 구독 콜백 (이 프로세스 spin tick 에서 실행, 단일 스레드) --

    def _on_cmd(self, msg: LowCmd) -> None:
        if self._paused:
            return
        self._buf_cmd.append((time.monotonic(), msg.motor_cmd[self._sel_idx].q))

    def _on_sim(self, msg: LowState) -> None:
        if self._paused:
            return
        now = time.monotonic()
        if now - self._last_sim_accept < DECIM_PERIOD_S:  # decimate to DECIM_HZ
            return
        self._last_sim_accept = now
        m = msg.motor_state[self._sel_idx]
        self._buf_sim.append((now, m.q, m.dq, m.tau_est))

    def _on_robot(self, msg: LowState) -> None:
        if self._paused:
            return
        now = time.monotonic()
        if now - self._last_robot_accept < DECIM_PERIOD_S:  # decimate to DECIM_HZ
            return
        self._last_robot_accept = now
        m = msg.motor_state[self._sel_idx]
        self._buf_robot.append((now, m.q, m.dq, m.tau_est))

    # -- UI 콜백 --

    def _on_motor_changed(self, idx: int) -> None:
        # 모터 전환 시 히스토리는 이전 모터 것이므로 초기화.
        self._sel_idx = idx
        self._buf_cmd.clear()
        self._buf_sim.clear()
        self._buf_robot.clear()

    def _on_pause_clicked(self) -> None:
        self._paused = not self._paused
        self._pause_btn.setText("Resume" if self._paused else "Pause")

    # -- 렌더 (full redraw) --
    #
    # blit 을 쓰지 않는다: monitor 가 별도 프로세스이므로 gui heartbeat 를 굶길 일이 없고,
    # blit on-screen 합성이 일부 X 환경에서 창을 까맣게 남기는 문제가 있었다(full repaint 전까지
    # 안 보임). draw_idle() 로 매 프레임 전체를 다시 그리는 편이 단순하고 견고하다.

    def _rescale_axes(self) -> None:
        now = time.monotonic()

        def _yrange(*series):
            vals = []
            for buf, cols in series:
                for row in buf:
                    if now - row[0] <= WINDOW_S:
                        for c in cols:
                            vals.append(row[c])
            if not vals:
                return None
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-3:
                lo, hi = lo - 0.5, hi + 0.5
            margin = 0.1 * (hi - lo)
            return lo - margin, hi + margin

        # q: cmd[1], sim[1], robot[1]
        r = _yrange((self._buf_cmd, (1,)), (self._buf_sim, (1,)), (self._buf_robot, (1,)))
        if r:
            self._axes[0].set_ylim(*r)
        # tau: sim[3], robot[3]
        r = _yrange((self._buf_sim, (3,)), (self._buf_robot, (3,)))
        if r:
            self._axes[1].set_ylim(*r)
        # dq: sim[2], robot[2]
        r = _yrange((self._buf_sim, (2,)), (self._buf_robot, (2,)))
        if r:
            self._axes[2].set_ylim(*r)

    def _on_render_tick(self) -> None:
        now = time.monotonic()

        # 라인 데이터 갱신
        self._set_line(self._lines[0].get("action"), self._buf_cmd, 1, now)  # q: action
        self._set_line(self._lines[0].get("sim"), self._buf_sim, 1, now)
        self._set_line(self._lines[0].get("robot"), self._buf_robot, 1, now)
        self._set_line(self._lines[1].get("sim"), self._buf_sim, 3, now)  # tau: col 3
        self._set_line(self._lines[1].get("robot"), self._buf_robot, 3, now)
        self._set_line(self._lines[2].get("sim"), self._buf_sim, 2, now)  # dq: col 2
        self._set_line(self._lines[2].get("robot"), self._buf_robot, 2, now)

        # y축 리스케일은 안정성을 위해 ~RESCALE_PERIOD_S 마다만 (매 프레임이면 지터).
        if (now - self._last_rescale) >= RESCALE_PERIOD_S:
            self._last_rescale = now
            self._rescale_axes()

        self._canvas.draw_idle()
        self._update_value_label()

    def _set_line(self, line, buf: collections.deque, col: int, now: float) -> None:
        if line is None:
            return
        xs, ys = [], []
        for row in buf:
            dt = row[0] - now  # 음수 (초 전)
            if dt < -WINDOW_S:
                continue
            xs.append(dt)
            ys.append(row[col])
        line.set_data(xs, ys)

    def _update_value_label(self) -> None:
        name = self._joint_names[self._sel_idx]
        parts = [f"[{name}]"]
        if self._buf_cmd:
            parts.append(f"action q={self._buf_cmd[-1][1]:+.3f}")
        if self._buf_sim:
            _, q, dq, tau = self._buf_sim[-1]
            parts.append(f"sim q={q:+.3f} dq={dq:+.2f} τ={tau:+.2f}")
        if self._has_robot and self._buf_robot:
            _, q, dq, tau = self._buf_robot[-1]
            parts.append(f"robot q={q:+.3f} dq={dq:+.2f} τ={tau:+.2f}")
        self._value_label.setText("   ".join(parts))

    # -- lifecycle --

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._render_timer.stop()
        self._spin_timer.stop()
        for sub in self._subs:
            with contextlib.suppress(Exception):
                self._node.destroy_subscription(sub)
        self._subs = []
        super().closeEvent(event)


def main() -> None:
    """독립 프로세스 진입점. gui 의 "Monitor" 버튼이 이 스크립트를 spawn 한다.

    RMW/ROS_DOMAIN_ID/CYCLONEDDS_URI 는 부모(gui) 프로세스에서 상속받는다 (같은 토픽 관측).
    """
    import rclpy
    from PyQt5.QtCore import Qt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import motions  # noqa: PLC0415

    rclpy.init()
    node = rclpy.create_node("r2s_monitor")

    app = QApplication(sys.argv)
    app.setStyleSheet(_STYLESHEET)
    win = MonitorWindow(node, motions.JOINT_NAMES)
    win.setAttribute(Qt.WA_DeleteOnClose, True)
    win.show()

    exit_code = app.exec_()

    with contextlib.suppress(Exception):
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
