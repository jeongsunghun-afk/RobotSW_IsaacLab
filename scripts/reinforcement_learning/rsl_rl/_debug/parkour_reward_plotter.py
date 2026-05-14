# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live reward debug plotter for Parkour env 0."""

import collections
import time

_MATPLOTLIB_AVAILABLE = False
_IMPORT_ERROR: str | None = None

try:
    import matplotlib

    try:
        matplotlib.use("TkAgg")
    except Exception as _backend_exc:
        print(f"[reward-plotter] TkAgg backend unavailable ({_backend_exc}); falling back to default backend.")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _MATPLOTLIB_AVAILABLE = True
except ImportError as _e:
    _IMPORT_ERROR = str(_e)


class _NoOpPlotter:
    """Stub plotter used when matplotlib is not available."""

    def step(self, env) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass


class LiveRewardPlotter:
    """Live 5×5 subplot reward visualizer for Parkour env 0.

    Reads ``env._last_reward_breakdown_env0`` (dict[str, float]) each step,
    accumulates values in rolling deques, and redraws the figure at most once
    per ``refresh_ms`` milliseconds.

    Terrain change is detected by comparing
    ``(env._terrain_levels[target_env_id], env._env_class[target_env_id])``
    across consecutive steps; on change all buffers are cleared and the step
    counter resets to 0.
    """

    def __new__(cls, env, target_env_id: int = 0, window: int = 500, refresh_ms: int = 200):
        if not _MATPLOTLIB_AVAILABLE:
            print(
                f"[reward-plotter] matplotlib not available ({_IMPORT_ERROR}); "
                "running without live plot (no-op stub)."
            )
            return _NoOpPlotter()
        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        env,
        target_env_id: int = 0,
        window: int = 500,
        refresh_ms: int = 200,
    ) -> None:
        # Guard: __new__ may have returned a _NoOpPlotter instead of self.
        if not isinstance(self, LiveRewardPlotter):
            return

        self._target_env_id = target_env_id
        self._window = window
        self._refresh_s = refresh_ms / 1000.0

        # --- resolve active reward keys from cfg.reward_scales ---
        scales_obj = env.cfg.reward_scales
        if hasattr(scales_obj, "__dict__"):
            scales_dict: dict = {
                k: float(v)
                for k, v in vars(scales_obj).items()
                if not k.startswith("_") and isinstance(v, (int, float))
            }
        elif isinstance(scales_obj, dict):
            scales_dict = {k: float(v) for k, v in scales_obj.items()}
        else:
            scales_dict = {}

        self._scales: dict[str, float] = scales_dict
        self._active_keys: list[str] = [k for k, v in self._scales.items() if v != 0.0]

        if len(self._active_keys) > 25:
            raise ValueError(
                f"[reward-plotter] Too many active reward terms ({len(self._active_keys)} > 25). "
                "Cannot fit in 5×5 grid."
            )

        # --- rolling buffers ---
        self._buffers: dict[str, collections.deque] = {
            k: collections.deque(maxlen=window) for k in self._active_keys
        }
        self._step_counter: int = 0
        self._prev_terrain_sig: tuple[int, int] | None = None
        self._last_redraw_t: float = time.monotonic()

        # --- build figure ---
        self._fig, axes = plt.subplots(5, 5, figsize=(18, 12), sharex=False)
        axes_flat = axes.flatten()  # 25 axes

        # Turn off unused subplots
        for idx in range(len(self._active_keys), 25):
            axes_flat[idx].axis("off")

        # Create line objects for active subplots
        self._lines: dict[str, Line2D] = {}
        for idx, key in enumerate(self._active_keys):
            ax = axes_flat[idx]
            scale = self._scales[key]
            ax.set_title(f"{key} (scale={scale:.3g})", fontsize=8)
            ax.grid(True, alpha=0.3)
            (line,) = ax.plot([], [], linewidth=0.8)
            self._lines[key] = line

        self._fig.suptitle("Parkour reward live debug (env 0)", fontsize=11)
        self._fig.tight_layout()
        plt.show(block=False)

    def step(self, env) -> None:
        """Call after every ``env.step()``. Fills buffers and redraws when due."""
        # Safety guards for non-parkour envs or missing attrs
        if not hasattr(env, "_last_reward_breakdown_env0"):
            return
        if not hasattr(env, "_terrain_levels") or not hasattr(env, "_env_class"):
            return

        # --- terrain change detection ---
        try:
            level = int(env._terrain_levels[self._target_env_id].item())
            cls = int(env._env_class[self._target_env_id].item())
        except Exception:
            return

        sig = (level, cls)
        if self._prev_terrain_sig is not None and sig != self._prev_terrain_sig:
            old = self._prev_terrain_sig
            print(
                f"[reward-plotter] terrain change: L{old[0]}->L{sig[0]}, "
                f"C{old[1]}->C{sig[1]} at sim_step {self._step_counter}"
            )
            for buf in self._buffers.values():
                buf.clear()
            self._step_counter = 0
        self._prev_terrain_sig = sig

        # --- fill buffers ---
        breakdown: dict = env._last_reward_breakdown_env0
        for k in self._active_keys:
            val = breakdown.get(k, 0.0)
            self._buffers[k].append(float(val))
        self._step_counter += 1

        # --- redraw if refresh interval elapsed ---
        now = time.monotonic()
        if now - self._last_redraw_t < self._refresh_s:
            return

        for key, line in self._lines.items():
            y = list(self._buffers[key])
            if not y:
                continue
            n = len(y)
            x = list(range(max(0, self._step_counter - n), self._step_counter))
            line.set_data(x, y)
            ax = line.axes
            x_left = max(0, self._step_counter - self._window)
            x_right = max(self._window, self._step_counter)
            ax.set_xlim(x_left, x_right)
            ymin, ymax = min(y), max(y)
            pad = max(1e-6, (ymax - ymin) * 0.1)
            ax.set_ylim(ymin - pad, ymax + pad)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)
        self._last_redraw_t = now

    def close(self) -> None:
        """Close the matplotlib figure."""
        try:
            plt.close(self._fig)
        except Exception:
            pass
