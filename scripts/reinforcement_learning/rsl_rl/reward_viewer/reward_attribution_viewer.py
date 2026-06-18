#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
reward_attribution_viewer.py — Standalone PyQt6 + pyqtgraph live viewer.

Subscribes to a ZMQ PUB socket published by play_reward_attribution_publisher.py
and renders per-env reward breakdowns + foot contact in real time.

Usage:
    python reward_attribution_viewer.py
    python reward_attribution_viewer.py --endpoint ipc:///tmp/parkour_reward.sock
    python reward_attribution_viewer.py --fps 30 --capacity 500
    python reward_attribution_viewer.py --selftest   # no display required

Requirements (install into isaac-parkour conda env):
    pip install PyQt6 pyqtgraph pyzmq msgpack

Design:  PYQT_IPC_SPEC.md §4-7
Protocol: reward_pub_protocol.py (imported from scripts/reinforcement_learning/rsl_rl/).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import cast

import numpy as np
import pyqtgraph as pg
import zmq
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# ─────────────────────────────────────────────────────────────────────────────
# Protocol import  (L1 module: reward_pub_protocol.py)
# ─────────────────────────────────────────────────────────────────────────────
# L3 drift resolution (Option A): import directly from L1's protocol module.
# The module lives at scripts/reinforcement_learning/rsl_rl/ relative to the
# repo root; compute path relative to this file so no hardcoded absolute path
# is needed.
_L1_PROTO_DIR = os.path.dirname(os.path.abspath(__file__))
if _L1_PROTO_DIR not in sys.path:
    sys.path.insert(0, _L1_PROTO_DIR)

from reward_pub_protocol import (
    PROTOCOL_VERSION,  # noqa: F401 — re-exported for downstream use
    encode_step,
    make_sub_socket,  # noqa: F401 — available; viewer keeps configurable endpoint
)
from reward_pub_protocol import (  # noqa: E402
    ZMQ_ENDPOINT as ENDPOINT_DEFAULT,
)
from reward_pub_protocol import (
    decode_step as decode_message,
)

# Presentation-side constants (viewer-only — not in the wire protocol)
REWARD_TERM_NAMES: list[str] = [
    "tracking_goal_vel",
    "tracking_yaw",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "orientation_l2",
    "dof_acc_l2",
    "collision",
    "action_rate_l2",
    "delta_torques",
    "torques_l2",
    "hip_pos",
    "dof_error_l2",
    "feet_stumble",
    "feet_edge",
    "feet_dragging",
    "feet_gait_pairing",
]
FOOT_NAMES: list[str] = ["FL", "FR", "RL", "RR"]


# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_TERMS: int = 16
NUM_FEET: int = 4
LANE_HEIGHT: float = 1.0  # vertical units per foot lane in contact subplot

# 16 reward-term colors (tab20 palette, as (R, G, B) tuples)
_TERM_COLORS: list[tuple[int, int, int]] = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (188, 189, 220),
]

# Foot lane colors: FL=blue, FR=green, RL=orange, RR=red
_FOOT_COLORS: list[tuple[int, int, int]] = [
    (76, 155, 232),  # FL — blue
    (92, 184, 92),  # FR — green
    (240, 173, 78),  # RL — orange
    (217, 83, 79),  # RR — red
]

# Gait-phase shading brushes (spec §S3: magenta 30%, cyan 20%)
_BRUSH_3LEG = pg.mkBrush(255, 0, 255, 76)  # 1 foot airborne — ~30 %
_BRUSH_2LEG = pg.mkBrush(0, 255, 255, 51)  # 2 feet airborne — ~20 %


# ─────────────────────────────────────────────────────────────────────────────
# Ring buffer  (spec §6)
# ─────────────────────────────────────────────────────────────────────────────
class RingBuffer:
    """
    Fixed-capacity ring buffer for per-env reward/contact time series.

    rewards   : [capacity, num_envs, NUM_TERMS]  float32
    contact   : [capacity, num_envs, NUM_FEET]   bool
    timestamps: [capacity]                        float64
    """

    def __init__(self, capacity: int = 500, num_envs: int = 5) -> None:
        self.capacity = capacity
        self.num_envs = num_envs
        self.rewards = np.zeros((capacity, num_envs, NUM_TERMS), dtype=np.float32)
        self.contact = np.zeros((capacity, num_envs, NUM_FEET), dtype=np.bool_)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self._write: int = 0
        self.size: int = 0

    def push(self, msg: dict) -> None:
        """Append one decoded step message; silently overwrites oldest when full."""
        idx = self._write
        self.timestamps[idx] = float(msg.get("t", time.time()))
        for env_data in msg.get("envs", []):
            eid = int(env_data.get("env_id", 0))
            if 0 <= eid < self.num_envs:
                rw = env_data.get("rewards", [])
                ct = env_data.get("contact", [])
                n_rw = min(len(rw), NUM_TERMS)
                n_ct = min(len(ct), NUM_FEET)
                if n_rw:
                    self.rewards[idx, eid, :n_rw] = rw[:n_rw]
                if n_ct:
                    self.contact[idx, eid, :n_ct] = ct[:n_ct]
        self._write = (self._write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_slice(self, env_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (rewards, contact, timestamps) for *env_id* in chronological order.

        rewards   : [T, NUM_TERMS]  float32
        contact   : [T, NUM_FEET]   bool
        timestamps: [T]             float64
        """
        n = self.size
        if n == 0:
            return (
                np.zeros((0, NUM_TERMS), dtype=np.float32),
                np.zeros((0, NUM_FEET), dtype=np.bool_),
                np.zeros(0, dtype=np.float64),
            )
        if n < self.capacity:
            return (
                self.rewards[:n, env_id, :].copy(),
                self.contact[:n, env_id, :].copy(),
                self.timestamps[:n].copy(),
            )
        # Full buffer — chronological order from write pointer
        w = self._write
        idx = (w + np.arange(n)) % self.capacity
        return (
            self.rewards[idx, env_id, :],
            self.contact[idx, env_id, :],
            self.timestamps[idx],
        )


# ─────────────────────────────────────────────────────────────────────────────
# ZMQ receiver thread  (spec §2 + task rule: QThread + poll(100 ms))
# ─────────────────────────────────────────────────────────────────────────────
class ZmqReceiver(QThread):
    """
    Background QThread that polls a ZMQ SUB socket with a 100 ms timeout.
    Emits *message_received* for every decoded message.
    Never tight-loops; poll() keeps CPU near zero when idle.
    """

    message_received = pyqtSignal(object)  # decoded dict
    status_changed = pyqtSignal(str)

    def __init__(self, endpoint: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.endpoint = endpoint
        self._stop_flag = False

    def stop(self) -> None:
        """Signal the run() loop to exit cleanly."""
        self._stop_flag = True

    def run(self) -> None:  # ← worker thread
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 100)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(self.endpoint)
        self.status_changed.emit(f"Subscribed → {self.endpoint}")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while not self._stop_flag:
            try:
                ready = dict(poller.poll(timeout=100))  # 100 ms — spec §2
            except zmq.ZMQError:
                break
            if ready.get(sock) == zmq.POLLIN:
                try:
                    raw = sock.recv(flags=zmq.NOBLOCK)
                    self.message_received.emit(decode_message(raw))
                except (zmq.Again, Exception):
                    pass  # transient; safe to skip

        sock.close()
        self.status_changed.emit("Receiver stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Per-environment tab  (4×4 reward grid + contact strip + active-feet strip)
# ─────────────────────────────────────────────────────────────────────────────
class EnvTab(QWidget):
    """
    Layout for one environment tab:

      Rows 0-3, Cols 0-3 : 4×4 grid of 16 individual reward-term subplots
      Row 4,    Col 0     : Foot contact strip (spans all 4 columns)
      Row 5,    Col 0     : Active-feet count + 3-leg / 2-leg gait shading

    All 18 subplots share the same x-axis (linked to the top-left cell p_reward[0]).
    """

    def __init__(self, env_id: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.env_id = env_id
        self._gait_3leg_pool: list[pg.LinearRegionItem] = []
        self._gait_2leg_pool: list[pg.LinearRegionItem] = []
        self._build_plots()

    # ──────────────────────────────────────────────────────────────────────
    def _build_plots(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(1)
        layout.setContentsMargins(2, 2, 2, 2)

        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw)

        # ── 4×4 reward grid (rows 0-3, cols 0-3) ─────────────────────────
        # p_reward[0] is the x-axis anchor; all other plots link to it.
        self._reward_plots: list[pg.PlotItem] = []
        self._reward_curves: list[pg.PlotDataItem] = []

        for idx in range(NUM_TERMS):
            row = idx // 4
            col = idx % 4
            name = REWARD_TERM_NAMES[idx] if idx < len(REWARD_TERM_NAMES) else f"term_{idx:02d}"
            rgb = _TERM_COLORS[idx % len(_TERM_COLORS)]

            p = glw.addPlot(row=row, col=col)
            p.setTitle(name, size="7pt")
            p.showGrid(x=False, y=True, alpha=0.25)
            p.setDownsampling(mode="peak")
            p.setClipToView(True)
            # Per-subplot y auto-range (terms span many orders of magnitude)
            p.enableAutoRange("y", True)
            # Hide x-axis labels on all but bottom reward row (row 3)
            if row < 3:
                p.hideAxis("bottom")
            # Hide y-axis labels on non-leftmost columns to save space
            if col > 0:
                p.hideAxis("left")

            if idx == 0:
                # Anchor plot — others link to this one
                p_anchor = p
            else:
                p.setXLink(p_anchor)

            # Single curve + legend entry
            legend = p.addLegend(offset=(2, 2), labelTextSize="6pt")
            curve = p.plot(pen=pg.mkPen(color=rgb, width=1.2), name=name)
            self._reward_plots.append(p)
            self._reward_curves.append(curve)

        # ── [contact] Foot contact strip (row 4, spanning col 0 only) ─────
        # GraphicsLayoutWidget doesn't support true colspan, so we place it
        # at row=4, col=0 and let the layout stretch via row weight.
        p_contact = glw.addPlot(row=4, col=0, colspan=4, title="Foot contact")
        p_contact.setXLink(p_anchor)
        p_contact.setLabel("left", "foot")
        p_contact.setYRange(-0.2, NUM_FEET * LANE_HEIGHT + 0.2)
        p_contact.getAxis("left").setTicks([[(i + LANE_HEIGHT * 0.5, FOOT_NAMES[i]) for i in range(NUM_FEET)]])
        p_contact.showGrid(x=True, y=False, alpha=0.3)
        p_contact.hideAxis("bottom")

        self._contact_curves: list[pg.PlotDataItem] = []
        for i, rgb in enumerate(_FOOT_COLORS):
            lane_base = float(i)
            c = p_contact.plot(
                pen=pg.mkPen(color=rgb, width=1),
                fillLevel=lane_base,
                brush=pg.mkBrush(*rgb, 140),
            )
            self._contact_curves.append(c)

        # ── [active feet] Count + gait shading (row 5, spanning col 0) ───
        p_feet = glw.addPlot(row=5, col=0, colspan=4, title="Active feet")
        p_feet.setXLink(p_anchor)
        p_feet.setLabel("left", "n_contact")
        p_feet.setLabel("bottom", "step")
        p_feet.setYRange(-0.2, 4.5)
        p_feet.getAxis("left").setTicks([[(i, str(i)) for i in range(5)]])
        p_feet.showGrid(x=True, y=True, alpha=0.3)

        self._active_curve = p_feet.plot(pen=pg.mkPen(color=(255, 255, 255), width=2))

        # Row stretch: reward rows (0-3) get equal share; contact & feet get less
        for row in range(4):
            glw.ci.layout.setRowStretchFactor(row, 3)
        glw.ci.layout.setRowStretchFactor(4, 2)
        glw.ci.layout.setRowStretchFactor(5, 2)

        self._p_anchor = p_anchor
        self._p_contact = p_contact
        self._p_feet = p_feet

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _step_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert (x, y) arrays to right-hold step-function coordinates.
        Doubles interior x values so each sample is held until the next one.
        """
        if len(x) < 2:
            return x, y
        n = len(x)
        x_s = np.empty(2 * n - 1, dtype=x.dtype)
        y_s = np.empty(2 * n - 1, dtype=y.dtype)
        x_s[0::2] = x  # original x positions
        x_s[1::2] = x[1:]  # transition x = next x
        y_s[0::2] = y  # value at x[i]
        y_s[1::2] = y[:-1]  # pre-transition value (hold current until jump)
        return x_s, y_s

    # ──────────────────────────────────────────────────────────────────────
    def update_plots(
        self,
        rewards: np.ndarray,  # [T, 16] float32
        contact: np.ndarray,  # [T, 4]  bool
        ts: np.ndarray,  # [T]     float64
    ) -> None:
        """Redraw all 18 subplots from the latest ring-buffer slice."""
        T = len(ts)
        if T == 0:
            return

        x = np.arange(T, dtype=np.float32)

        # ── 16 individual reward curves ───────────────────────────────────
        for i, curve in enumerate(self._reward_curves):
            curve.setData(x=x, y=rewards[:, i])

        # ── Contact curves (step fill, binary per foot) ───────────────────
        for i, curve in enumerate(self._contact_curves):
            lane_base = float(i)
            y_raw = contact[:, i].astype(np.float32) * LANE_HEIGHT * 0.9 + lane_base
            xs, ys = self._step_xy(x, y_raw)
            curve.setData(x=xs, y=ys)

        # ── Active feet curve ─────────────────────────────────────────────
        n_contact = contact.sum(axis=-1).astype(np.float32)
        xs, ys = self._step_xy(x, n_contact)
        self._active_curve.setData(x=xs, y=ys)

        # ── Gait phase shading ────────────────────────────────────────────
        self._update_gait_shading(x, contact)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _find_runs(x: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        """Return [(lo, hi), ...] for each contiguous True run in *mask*."""
        runs: list[tuple[float, float]] = []
        T = len(mask)
        i = 0
        while i < T:
            if mask[i]:
                j = i + 1
                while j < T and mask[j]:
                    j += 1
                runs.append((float(x[i]), float(x[j - 1]) + 1.0))
                i = j
            else:
                i += 1
        return runs

    def _sync_region_pool(
        self,
        pool: list[pg.LinearRegionItem],
        runs: list[tuple[float, float]],
        brush: pg.QtGui.QBrush,
    ) -> None:
        """Grow / reuse the pool of LinearRegionItems; hide unused items."""
        while len(pool) < len(runs):
            region = pg.LinearRegionItem(
                movable=False,
                brush=brush,
                pen=pg.mkPen(None),
            )
            self._p_feet.addItem(region)
            pool.append(region)
        for j, (lo, hi) in enumerate(runs):
            pool[j].setRegion((lo, hi))
            pool[j].setVisible(True)
        for j in range(len(runs), len(pool)):
            pool[j].setVisible(False)

    def _update_gait_shading(self, x: np.ndarray, contact: np.ndarray) -> None:
        n_air = NUM_FEET - contact.sum(axis=-1)
        self._sync_region_pool(self._gait_3leg_pool, self._find_runs(x, n_air == 1), _BRUSH_3LEG)
        self._sync_region_pool(self._gait_2leg_pool, self._find_runs(x, n_air == 2), _BRUSH_2LEG)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.setWindowTitle("Parkour Reward Attribution — Live Viewer")
        self.resize(1800, 1020)

        self._buf = RingBuffer(capacity=args.capacity, num_envs=args.num_envs)

        # Message stats
        self._msg_count: int = 0
        self._drop_count: int = 0
        self._last_step: int = -1
        self._last_msg_t: float = time.time()
        self._fps_frames: int = 0
        self._fps_last_t: float = time.time()
        self._fps: float = 0.0

        # Central widget — tab per env
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        vbox.addWidget(self._tabs)

        self._env_tabs: list[EnvTab] = []
        for i in range(args.num_envs):
            tab = EnvTab(env_id=i)
            self._tabs.addTab(tab, f"Env {i}")
            self._env_tabs.append(tab)

        # Status bar
        self._status_lbl = QLabel("Waiting for publisher…")
        self.statusBar().addWidget(self._status_lbl, 1)
        self.statusBar().addPermanentWidget(QLabel("  proto: L1"))

        # ZMQ receiver (background QThread)
        self._recv = ZmqReceiver(endpoint=args.endpoint)
        self._recv.message_received.connect(self._on_msg)
        self._recv.status_changed.connect(self._on_status)
        self._recv.start()

        # Render timer @ --fps Hz (only redraws the active tab)
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(max(1, int(1000 / args.fps)))
        self._render_timer.timeout.connect(self._render_tick)
        self._render_timer.start()

        # Stats refresh @ 1 Hz
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(1000)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start()

    # ──────────────────────────────────────────────────────────────────────
    def _on_msg(self, msg: dict) -> None:
        step = int(msg.get("step_idx", 0))
        if self._last_step >= 0 and step > self._last_step + 1:
            self._drop_count += step - self._last_step - 1
        self._last_step = step
        self._last_msg_t = time.time()
        self._msg_count += 1
        self._buf.push(msg)

    def _on_status(self, info: str) -> None:
        self._status_lbl.setText(info)

    def _render_tick(self) -> None:
        self._fps_frames += 1
        idx = self._tabs.currentIndex()
        if 0 <= idx < len(self._env_tabs):
            env_tab = cast(EnvTab, self._env_tabs[idx])
            rewards, contact, ts = self._buf.get_slice(idx)
            env_tab.update_plots(rewards, contact, ts)

    def _refresh_stats(self) -> None:
        now = time.time()
        elapsed = now - self._fps_last_t
        if elapsed > 0:
            self._fps = self._fps_frames / elapsed
        self._fps_frames = 0
        self._fps_last_t = now
        age = now - self._last_msg_t
        self._status_lbl.setText(
            f"FPS: {self._fps:.1f}  │  msgs: {self._msg_count}"
            f"  │  drops: {self._drop_count}"
            f"  │  last msg: {age:.1f}s ago"
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._render_timer.stop()
        self._stats_timer.stop()
        self._recv.stop()
        self._recv.wait(2000)  # 2 s grace period for thread exit
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Stub-driven self-test  (--selftest; no display required)
# ─────────────────────────────────────────────────────────────────────────────
def _run_selftest(args: argparse.Namespace) -> int:
    """
    Publish N stub messages over a loopback IPC socket and verify:
      - ≥90 % of messages received without blocking
      - Ring buffer populated for all 5 envs
      - get_slice returns correct shapes
      - No zombie ZMQ context after close

    Returns 0 on success, 1 on failure.
    """
    ENDPOINT = "ipc:///tmp/parkour_reward_selftest.sock"
    N = 60
    errors: list[str] = []

    # ── Publisher stub (background thread) ───────────────────────────────
    def _publish() -> None:
        pctx = zmq.Context()
        pub = pctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 100)
        pub.bind(ENDPOINT)
        time.sleep(0.35)  # let subscriber connect
        for step in range(N):
            env_payloads = [
                {
                    "env_id": i,
                    "rewards": [float(step * 0.01 + i * 0.1 + k * 0.001) for k in range(16)],
                    "contact": [bool((step + k) % 2 == 0) for k in range(4)],
                    "commands": [1.0, 0.0, 0.0],
                    "done": False,
                    "terrain_id": i,
                }
                for i in range(5)
            ]
            pub.send(encode_step(step_idx=step, env_payloads=env_payloads))
            time.sleep(0.005)  # ~200 Hz — faster than production 50 Hz
        time.sleep(0.3)
        pub.close()
        pctx.term()

    # ── Subscriber ────────────────────────────────────────────────────────
    sctx = zmq.Context()
    sub = sctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(ENDPOINT)

    pub_thread = threading.Thread(target=_publish, daemon=True)
    pub_thread.start()

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    buf = RingBuffer(capacity=args.capacity, num_envs=args.num_envs)
    received = 0
    deadline = time.time() + 15.0

    while received < N and time.time() < deadline:
        ready = dict(poller.poll(200))
        if ready.get(sub) == zmq.POLLIN:
            raw = sub.recv(flags=zmq.NOBLOCK)
            buf.push(decode_message(raw))
            received += 1

    sub.close()
    sctx.term()
    pub_thread.join(timeout=5.0)

    # ── Assertions ────────────────────────────────────────────────────────
    min_ok = int(N * 0.90)
    if received < min_ok:
        errors.append(f"RECV: {received}/{N} messages (need ≥{min_ok})")

    if buf.size == 0:
        errors.append("BUFFER: empty after receiving messages")
    else:
        for eid in range(5):
            rw, ct, ts = buf.get_slice(eid)
            if len(rw) == 0:
                errors.append(f"ENV{eid}: empty reward slice")
                continue
            if rw.shape[1] != NUM_TERMS:
                errors.append(f"ENV{eid}: rewards.shape={rw.shape}, expected (*,{NUM_TERMS})")
            if ct.shape[1] != NUM_FEET:
                errors.append(f"ENV{eid}: contact.shape={ct.shape}, expected (*,{NUM_FEET})")
            if len(ts) != len(rw):
                errors.append(f"ENV{eid}: timestamps len={len(ts)} ≠ rewards len={len(rw)}")
            if not np.isfinite(rw).all():
                errors.append(f"ENV{eid}: non-finite values in rewards")

    # ── Ring-buffer wraparound sanity check ───────────────────────────────
    small_buf = RingBuffer(capacity=10, num_envs=1)
    for s in range(25):
        small_buf.push(
            {
                "t": float(s),
                "step_idx": s,
                "envs": [
                    {
                        "env_id": 0,
                        "rewards": [float(s)] * 16,
                        "contact": [True, False, True, False],
                        "commands": [1.0, 0.0, 0.0],
                        "done": False,
                        "terrain_id": 0,
                    }
                ],
            }
        )
    rw2, _, _ = small_buf.get_slice(0)
    if len(rw2) != 10:
        errors.append(f"WRAPAROUND: expected 10 entries, got {len(rw2)}")
    elif abs(float(rw2[-1, 0]) - 24.0) > 1e-4:
        errors.append(f"WRAPAROUND: last reward={rw2[-1, 0]:.4f}, expected 24.0")

    # ── Report ────────────────────────────────────────────────────────────
    if errors:
        print("SELFTEST FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print(f"SELFTEST PASSED  received={received}/{N}  buf_size={buf.size}  wraparound=OK  envs=5  proto=L1")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--endpoint",
        default=ENDPOINT_DEFAULT,
        help=f"ZMQ PUB endpoint to subscribe to  (default: {ENDPOINT_DEFAULT})",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        metavar="HZ",
        help="Render tick rate in Hz  (default: 30)",
    )
    p.add_argument(
        "--capacity",
        type=int,
        default=500,
        help="Ring buffer capacity in steps  (default: 500)",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=5,
        dest="num_envs",
        help="Number of parallel environments  (default: 5)",
    )
    p.add_argument(
        "--selftest",
        action="store_true",
        help="Run stub-driven smoke test without GUI, then exit",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.selftest:
        sys.exit(_run_selftest(args))

    # Apply global pyqtgraph settings before QApplication
    pg.setConfigOptions(antialias=True, background="k", foreground="w")

    app = QApplication(sys.argv)
    app.setApplicationName("Parkour Reward Viewer")
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
