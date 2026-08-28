"""monitor.py 가 실제로 곡선을 그리는지 / 빈 화면에 이유를 적는지 창을 띄워 확인한다.

Xvfb 로 돌리므로 **앱 쪽 렌더 경로만** 검정한다 — 원격 X11 전송 문제는 여기서 재현되지 않는다.
곡선이 여기서 보이면 "까만 화면"의 원인은 앱이 아니라 데이터 또는 X11 쪽이다.

    xvfb-run -a python3 _workspace/r2s_monitor_render_check.py <out_dir>
"""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, "scripts/real2sim/r2s_biped_leg")

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mon_render"
os.makedirs(OUT, exist_ok=True)

import monitor as M  # noqa: E402
import r2s_udp  # noqa: E402
from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
win = M.MonitorWindow(joint_names=list(M.motions.JOINT_NAMES), label="sim")
win.resize(1100, 800)
win.show()

SIM_PORT, REAL_PORT = r2s_udp.MONITOR_PORT, r2s_udp.REAL_MON_PORT
import socket  # noqa: E402

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state = {"n": 0, "shots": 0}


def feed():
    """250 개(=5 s @50Hz) 를 넣고 멈춘다 — 그 뒤 배너가 뜨는지 본다."""
    if state["n"] >= 250:
        return
    i = state["n"]
    q = [0.4 * math.sin(2 * math.pi * (i / 50.0) * 1.0 + j) for j in range(8)]
    dq = [2.0 * math.cos(2 * math.pi * (i / 50.0) * 1.0 + j) for j in range(8)]
    tau = [5.0 * math.sin(2 * math.pi * (i / 50.0) * 0.5 + j) for j in range(8)]
    tgt = [v * 1.1 for v in q]
    tx.sendto(r2s_udp.pack_monitor(i, tgt, q, dq, tau), ("127.0.0.1", SIM_PORT))
    tx.sendto(r2s_udp.pack_monitor(i, tgt, q, dq, tau), ("127.0.0.1", REAL_PORT))
    state["n"] += 1


feeder = QTimer()
feeder.timeout.connect(feed)
feeder.start(4)  # 250 Hz 로 밀어 넣어 5 s 분량을 1 s 만에 채운다


def grab(name):
    win.grab().save(os.path.join(OUT, name))
    print("wrote", os.path.join(OUT, name), flush=True)


def shot_with_data():
    grab("with_data.png")
    print("empty_note(데이터 있음) =", repr(win._empty_note.toPlainText()), flush=True)


def shot_stale():
    # 발행을 멈춘 지 WINDOW_S 초과 → 창 안에 점이 없어야 하고 배너가 떠야 한다
    win._anchor = (win._anchor[0], time.monotonic() - (M.WINDOW_S + 5)) if win._anchor else None
    if win._real_anchor:
        win._real_anchor = (win._real_anchor[0], time.monotonic() - (M.WINDOW_S + 5))
    win._on_render_tick()
    grab("stale.png")
    txt = win._empty_note.toPlainText()
    print("empty_note(정지) =", repr(txt), flush=True)
    ok = bool(txt.strip())
    print("★ 빈 화면에 이유 표기:", "PASS" if ok else "FAIL", flush=True)
    app.exit(0 if ok else 1)


QTimer.singleShot(1800, shot_with_data)
QTimer.singleShot(2400, shot_stale)
sys.exit(app.exec_())
