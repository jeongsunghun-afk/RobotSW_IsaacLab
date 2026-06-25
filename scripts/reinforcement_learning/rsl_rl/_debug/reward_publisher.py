# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UDP publisher that broadcasts per-step reward breakdowns from the Parkour environment.

This module is intentionally dependency-free (stdlib only: socket, json).
It is imported by play.py and sends non-blocking UDP datagrams to a local viewer process.
"""

import contextlib
import json
import socket


class RewardUDPPublisher:
    """Non-blocking UDP publisher for live reward debugging.

    Reads ``env._last_reward_breakdown_env0`` every step and sends a JSON
    datagram to the viewer.  All send errors are silently swallowed so that
    the main simulation loop is never blocked or interrupted.

    Args:
        env: The unwrapped ParkourEnv (``DirectRLEnv.unwrapped``).
        target_env_id: Index of the environment instance to monitor.
        host: UDP destination address.
        port: UDP destination port.
    """

    def __init__(
        self,
        env,
        target_env_id: int = 0,
        host: str = "127.0.0.1",
        port: int = 9876,
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._addr = (host, port)
        self._target_env_id = target_env_id
        self._closed = False
        self._step_counter = 0

        # Read reward_scales from env cfg once at init
        scales_obj = env.cfg.reward_scales
        if hasattr(scales_obj, "__dict__"):
            self._scales = {k: float(v) for k, v in vars(scales_obj).items() if not k.startswith("_")}
        else:
            self._scales = {k: float(v) for k, v in dict(scales_obj).items()}

        print(
            f"[reward-publisher] UDP {host}:{port}, {len(self._scales)} reward terms tracked. "
            "Run `python reward_viewer.py` in another terminal to visualize."
        )

    def step(self, env) -> None:
        """Send reward breakdown for the current step via UDP (non-blocking).

        Call this immediately after ``env.step()``.  Safe to call even if the
        viewer is not running — packets are silently dropped.

        Args:
            env: The unwrapped environment (``DirectRLEnv.unwrapped``).
        """
        if self._closed:
            return
        if not hasattr(env, "_last_reward_breakdown_env0"):
            return
        if not hasattr(env, "_terrain_levels") or not hasattr(env, "_env_class"):
            return

        try:
            level = int(env._terrain_levels[self._target_env_id].item())
            cls = int(env._env_class[self._target_env_id].item())
        except Exception:
            return

        rewards = {k: float(v) for k, v in env._last_reward_breakdown_env0.items()}

        payload = {
            "step": self._step_counter,
            "terrain": [level, cls],
            "rewards": rewards,
            "scales": self._scales,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            self._sock.sendto(data, self._addr)
        except (BlockingIOError, OSError):
            pass  # viewer not running or send buffer full — silently drop
        except Exception:
            pass  # any other failure — silently ignore

        self._step_counter += 1

    def close(self) -> None:
        """Close the UDP socket."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._sock.close()
