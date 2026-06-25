#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour reward live viewer — UDP subscriber.

Usage:
    python reward_viewer.py [--host 127.0.0.1] [--port 9876] [--window 500] [--refresh-ms 200]

Run this in a separate terminal while play.py is running.
Requires only matplotlib (no IsaacSim / isaaclab / torch dependency).
"""

import argparse
import contextlib
import json
import socket
import time
from collections import deque

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main() -> None:
    parser = argparse.ArgumentParser(description="Parkour reward live viewer (UDP subscriber)")
    parser.add_argument("--host", default="127.0.0.1", help="UDP bind address")
    parser.add_argument("--port", type=int, default=9876, help="UDP bind port")
    parser.add_argument("--window", type=int, default=500, help="Rolling window length (steps)")
    parser.add_argument("--refresh-ms", type=int, default=200, help="Plot refresh interval in milliseconds")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.setblocking(False)
    print(f"[reward-viewer] Listening on UDP {args.host}:{args.port}. Waiting for first message...")

    plt.ion()
    fig = None
    axes_flat = None
    lines: dict[str, Line2D] = {}
    active_keys: list = []
    buffers: dict = {}
    step_counter = 0
    prev_sig: tuple | None = None
    last_redraw_t = time.monotonic()
    refresh_s = args.refresh_ms / 1000.0
    initialized = False

    try:
        while True:
            # 1. Drain UDP queue — up to 64 packets per loop tick
            drained = 0
            while drained < 64:
                try:
                    data, _ = sock.recvfrom(65536)
                except BlockingIOError:
                    break
                try:
                    msg = json.loads(data.decode("utf-8"))
                except Exception:
                    continue

                # Initialize matplotlib figure on the first valid message
                if not initialized:
                    scales = msg.get("scales", {})
                    active_keys = [k for k, v in scales.items() if v != 0.0]
                    if len(active_keys) > 25:
                        print(
                            f"[reward-viewer] WARN: {len(active_keys)} active keys > 25 grid slots. Truncating."
                        )
                        active_keys = active_keys[:25]

                    fig, axes = plt.subplots(5, 5, figsize=(18, 12))
                    axes_flat = axes.flatten()
                    for i, key in enumerate(active_keys):
                        ax = axes_flat[i]
                        ax.set_title(f"{key} (scale={scales[key]:.3g})", fontsize=8)
                        ax.grid(True, alpha=0.3)
                        lines[key] = ax.plot([], [])[0]
                    for i in range(len(active_keys), 25):
                        axes_flat[i].axis("off")
                    fig.suptitle("Parkour reward live debug (env 0) — UDP viewer")
                    fig.tight_layout()
                    plt.show(block=False)
                    buffers = {k: deque(maxlen=args.window) for k in active_keys}
                    initialized = True
                    print(f"[reward-viewer] initialized with {len(active_keys)} active keys.")

                sig = tuple(msg.get("terrain", [0, 0]))
                rewards = msg.get("rewards", {})

                # Detect terrain change and reset buffers
                if prev_sig is not None and sig != prev_sig:
                    print(
                        f"[reward-viewer] terrain change: L{prev_sig[0]}->L{sig[0]}, "
                        f"C{prev_sig[1]}->C{sig[1]} at viewer_step {step_counter}"
                    )
                    for buf in buffers.values():
                        buf.clear()
                    step_counter = 0
                prev_sig = sig

                for k in active_keys:
                    buffers[k].append(float(rewards.get(k, 0.0)))
                step_counter += 1
                drained += 1

            # 2. Redraw if refresh interval has elapsed
            if initialized:
                assert fig is not None  # narrowed: fig is set during initialization
                now = time.monotonic()
                if now - last_redraw_t >= refresh_s:
                    for k, line in lines.items():
                        y = list(buffers[k])
                        if not y:
                            continue
                        x = list(range(max(0, step_counter - len(y)), step_counter))
                        line.set_data(x, y)
                        ax = line.axes
                        ax.set_xlim(max(0, step_counter - args.window), max(args.window, step_counter))
                        ymin, ymax = min(y), max(y)
                        pad = max(1e-6, (ymax - ymin) * 0.1)
                        ax.set_ylim(ymin - pad, ymax + pad)
                    try:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    except Exception:
                        print("[reward-viewer] window closed; exiting.")
                        return
                    last_redraw_t = now
                else:
                    try:
                        fig.canvas.flush_events()
                    except Exception:
                        return

            if drained == 0:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[reward-viewer] interrupted by user.")
    finally:
        with contextlib.suppress(Exception):
            sock.close()
        with contextlib.suppress(Exception):
            plt.close("all")


if __name__ == "__main__":
    main()
