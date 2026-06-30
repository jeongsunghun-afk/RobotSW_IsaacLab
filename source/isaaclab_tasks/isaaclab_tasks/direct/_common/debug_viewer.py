# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DebugViewer — composition helper for IsaacLab DirectRLEnv.

Owns:
  - keyboard subscription (carb.input)
  - free-fly camera state machine (toggle + WASD + Q/E + Shift)
  - env_index switching ([ / ])
  - generic register_key(key, on_press, on_release, on_repeat)
  - generic register_debug_vis(name, callback, default_on=False) — sensor-style

Lifecycle::

    env.__init__():    self._debug_viewer = DebugViewer(self, cfg=...)
    env.step() end:    self._debug_viewer.update(self.step_dt)
    env.close():       self._debug_viewer.close()

Headless safety: when ``env.viewport_camera_controller`` is None all public methods
are no-ops.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

from .debug_keymap import DebugViewerCfg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DebugVisHandle
# ---------------------------------------------------------------------------


class DebugVisHandle:
    """Returned by :meth:`DebugViewer.register_debug_vis`. Controls on/off and removal."""

    def __init__(self, viewer: DebugViewer, name: str) -> None:
        self._viewer = viewer
        self._name = name

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this debug visualisation."""
        self._viewer.set_debug_vis(self._name, enabled)

    def unregister(self) -> None:
        """Permanently remove this debug visualisation from the viewer."""
        self._viewer.vis_unregister(self._name)

    @property
    def enabled(self) -> bool:
        """Whether the callback is currently active."""
        return self._viewer.vis_is_enabled(self._name)


# ---------------------------------------------------------------------------
# DebugViewer
# ---------------------------------------------------------------------------


class DebugViewer:
    """Composition helper that adds viewer/debug infrastructure to a DirectRLEnv.

    Parameters
    ----------
    env:
        The owning ``DirectRLEnv`` instance.
    cfg:
        Optional configuration.  Defaults to :class:`DebugViewerCfg` with all
        fields at their default values.

    Raises
    ------
    ValueError
        If *env* is ``None``.
    RuntimeError
        If a ``DebugViewer`` instance already exists for this env object.
    """

    # Weak registry: env id → DebugViewer, to detect duplicate instantiation.
    _registry: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def __init__(self, env: DirectRLEnv, cfg: DebugViewerCfg | None = None) -> None:
        if env is None:
            raise ValueError("DebugViewer: env must not be None.")

        env_id = id(env)
        if env_id in DebugViewer._registry:
            raise RuntimeError(
                f"DebugViewer: an instance already exists for env {env!r}. "
                "Call close() on the existing instance first."
            )

        self._cfg = cfg if cfg is not None else DebugViewerCfg()
        self._env_ref: weakref.ref[DirectRLEnv] = weakref.ref(env)
        self._closed = False

        # Active only when GUI is available and cfg.enabled is True.
        self._enabled: bool = self._cfg.enabled and (getattr(env, "viewport_camera_controller", None) is not None)

        # ---------- free-fly state ----------
        self._free_fly_camera: bool = False
        self._cam_keys_pressed: set = set()

        # ---------- user-registered key callbacks ----------
        # name → {"on_press": ..., "on_release": ..., "on_repeat": ...}
        self._user_key_callbacks: dict[str, dict] = {}

        # ---------- debug-vis registry ----------
        # name → {"callback": Callable[[float], None], "enabled": bool}
        self._debug_vis_registry: dict[str, dict] = {}

        # ---------- subscription handles ----------
        self._keyboard_sub = None
        self._input = None
        self._keyboard = None
        self._post_update_sub = None

        if self._enabled:
            self._setup_keyboard()
            self._setup_post_update_callback()

        DebugViewer._registry[env_id] = self

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release all subscriptions and registered callbacks. Idempotent."""
        if self._closed:
            return
        self._closed = True

        self._teardown_keyboard()
        self._teardown_post_update_callback()
        self._debug_vis_registry.clear()
        self._user_key_callbacks.clear()

        # Remove from registry
        env = self._env_ref()
        if env is not None:
            DebugViewer._registry.pop(id(env), None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_gui(self) -> bool:
        """True when the viewer is active (GUI available and cfg.enabled=True)."""
        return self._enabled and not self._closed

    @property
    def is_free_fly_camera(self) -> bool:
        """True when free-fly mode is currently active.

        Envs that own a separate camera-tracking callback can use this as a
        guard to skip their tracking logic while the user is flying freely.
        """
        return self._enabled and self._free_fly_camera

    def register_key(
        self,
        key: str,
        on_press: Callable[[], None] | None = None,
        on_release: Callable[[], None] | None = None,
        on_repeat: Callable[[], None] | None = None,
    ) -> None:
        """Register user callbacks for a keyboard key.

        Parameters
        ----------
        key:
            Attribute name on ``carb.input.KeyboardInput`` (e.g. ``"P"``).
        on_press, on_release, on_repeat:
            Callables invoked on the corresponding keyboard event type.  Any
            that are ``None`` are ignored.

        Notes
        -----
        - If the key is already registered, the existing callbacks are replaced
          and a warning is emitted.
        - Built-in keys (F / WASD / arrows / Q / E / Shift / [ / ]) can be
          overridden by user callbacks; the user callback takes precedence.
        - No-op when :attr:`has_gui` is ``False``.
        """
        if not self.has_gui:
            return
        if key in self._user_key_callbacks:
            log.warning("[debug_viewer] register_key: key '%s' already registered — overwriting.", key)
        self._user_key_callbacks[key] = {
            "on_press": on_press,
            "on_release": on_release,
            "on_repeat": on_repeat,
        }

    def register_debug_vis(
        self,
        name: str,
        callback: Callable[[float], None],
        default_on: bool = False,
    ) -> DebugVisHandle:
        """Register a per-frame debug-visualisation callback.

        Follows the ``sensor_base.set_debug_vis`` convention: the callback is
        *not* invoked at registration time; it starts firing only when enabled.

        Parameters
        ----------
        name:
            Unique identifier for this visualisation.
        callback:
            Called with ``dt`` (seconds since last post-update) each frame
            while enabled.
        default_on:
            If ``True``, the callback is enabled immediately after registration.

        Returns
        -------
        DebugVisHandle
            Handle to enable/disable or unregister the callback.
        """
        self._debug_vis_registry[name] = {
            "callback": callback,
            "enabled": default_on,
        }
        return DebugVisHandle(self, name)

    def set_debug_vis(self, name: str, enabled: bool) -> bool:
        """Enable or disable a registered debug visualisation.

        Returns
        -------
        bool
            ``True`` if the name was found and updated, ``False`` otherwise.
        """
        entry = self._debug_vis_registry.get(name)
        if entry is None:
            return False
        entry["enabled"] = enabled
        return True

    # Internal-API for DebugVisHandle (same module — no protected-access warning).
    def vis_unregister(self, name: str) -> None:
        """Remove a debug-vis entry by name. Called by :class:`DebugVisHandle`."""
        self._debug_vis_registry.pop(name, None)

    def vis_is_enabled(self, name: str) -> bool:
        """Return whether a debug-vis entry is currently enabled. Called by :class:`DebugVisHandle`."""
        entry = self._debug_vis_registry.get(name)
        if entry is None:
            return False
        return entry["enabled"]

    def update(self, dt: float) -> None:
        """Called from ``env.step()`` at the end of each policy step.

        Handles free-fly camera translation when free-fly mode is active.
        Debug-vis callbacks are driven by the post-update event stream
        subscription, not by this method.
        """
        if not self.has_gui:
            return
        if self._free_fly_camera:
            self._update_free_fly_camera(dt)

    # ------------------------------------------------------------------
    # Internal — keyboard
    # ------------------------------------------------------------------

    def _setup_keyboard(self) -> None:
        """Subscribe to carb keyboard events."""
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                log.warning("[debug_viewer] No app window found; keyboard subscription skipped.")
                return
            self._keyboard = app_window.get_keyboard()
            self._input = carb.input.acquire_input_interface()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
        except Exception as exc:
            log.warning("[debug_viewer] Keyboard setup failed: %s", exc)

    def _teardown_keyboard(self) -> None:
        """Unsubscribe from carb keyboard events."""
        try:
            if self._input is not None and self._keyboard is not None and self._keyboard_sub is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception:
            pass
        finally:
            self._keyboard_sub = None
            self._input = None
            self._keyboard = None

    def _on_keyboard_event(self, event, *_args) -> bool:
        """Dispatch carb keyboard events.

        Processing order:

        1. Free-fly movement keys (W/A/S/D/Q/E/arrows/LEFT_SHIFT):
           maintain pressed-set across PRESS / REPEAT / RELEASE; return early.
        2. F key (KEY_PRESS only): toggle free-fly mode.
        3. [ / ] keys (KEY_PRESS only): cycle env_index.
        4. User-registered key callbacks.

        Returns
        -------
        bool
            Always returns ``True`` (event consumed).
        """
        del _args
        try:
            import carb.input as ci

            KT = ci.KeyboardEventType
        except Exception:
            return True

        # Resolve all movement keys (lazily — these are builtin so always populated).
        _move_keys = set(
            filter(
                None,
                [
                    self._resolve_key(self._cfg.keys.free_fly_forward) if self._cfg.keys.free_fly_forward else None,
                    self._resolve_key(self._cfg.keys.free_fly_back) if self._cfg.keys.free_fly_back else None,
                    self._resolve_key(self._cfg.keys.free_fly_left) if self._cfg.keys.free_fly_left else None,
                    self._resolve_key(self._cfg.keys.free_fly_right) if self._cfg.keys.free_fly_right else None,
                    self._resolve_key(self._cfg.keys.free_fly_forward_alt)
                    if self._cfg.keys.free_fly_forward_alt
                    else None,
                    self._resolve_key(self._cfg.keys.free_fly_back_alt)
                    if self._cfg.keys.free_fly_back_alt
                    else None,
                    self._resolve_key(self._cfg.keys.free_fly_left_alt)
                    if self._cfg.keys.free_fly_left_alt
                    else None,
                    self._resolve_key(self._cfg.keys.free_fly_right_alt)
                    if self._cfg.keys.free_fly_right_alt
                    else None,
                    self._resolve_key(self._cfg.keys.free_fly_up) if self._cfg.keys.free_fly_up else None,
                    self._resolve_key(self._cfg.keys.free_fly_down) if self._cfg.keys.free_fly_down else None,
                    self._resolve_key(self._cfg.keys.free_fly_speed_up)
                    if self._cfg.keys.free_fly_speed_up
                    else None,
                ],
            )
        )

        # Step 1: movement keys — track pressed state.
        if event.input in _move_keys:
            if event.type in (KT.KEY_PRESS, KT.KEY_REPEAT):
                self._cam_keys_pressed.add(event.input)
            elif event.type == KT.KEY_RELEASE:
                self._cam_keys_pressed.discard(event.input)
            return True

        # Remaining bindings: KEY_PRESS only (unless user callback overrides).

        # Step 2: F toggle — check user override first, then builtin.
        _f_key = self._resolve_key(self._cfg.keys.toggle_free_fly) if self._cfg.keys.toggle_free_fly else None
        if _f_key is not None and event.input == _f_key:
            # User-registered override takes precedence.
            user_cb = self._user_key_callbacks.get(self._cfg.keys.toggle_free_fly or "")
            if user_cb is not None:
                if event.type == KT.KEY_PRESS and user_cb.get("on_press"):
                    user_cb["on_press"]()
                elif event.type == KT.KEY_RELEASE and user_cb.get("on_release"):
                    user_cb["on_release"]()
                elif event.type == KT.KEY_REPEAT and user_cb.get("on_repeat"):
                    user_cb["on_repeat"]()
                return True
            # Builtin free-fly toggle.
            if event.type == KT.KEY_PRESS:
                self._free_fly_camera = not self._free_fly_camera
                if self._free_fly_camera:
                    print("[debug_viewer] Camera mode: FREE-FLY (WASD/arrows to move, mouse to rotate)")
                else:
                    self._cam_keys_pressed.clear()
                    print("[debug_viewer] Camera mode: TRACKING (following robot)")
            return True

        # Only process PRESS for the remaining builtins.
        if event.type != KT.KEY_PRESS:
            # Still check user callbacks for non-press events.
            for key_name, entry in self._user_key_callbacks.items():
                resolved = self._resolve_key(key_name)
                if resolved is not None and event.input == resolved:
                    if event.type == KT.KEY_RELEASE and entry.get("on_release"):
                        entry["on_release"]()
                    elif event.type == KT.KEY_REPEAT and entry.get("on_repeat"):
                        entry["on_repeat"]()
                    return True
            return True

        # Step 3: env_index switching.
        _prev_key = self._resolve_key(self._cfg.keys.env_index_prev) if self._cfg.keys.env_index_prev else None
        _next_key = self._resolve_key(self._cfg.keys.env_index_next) if self._cfg.keys.env_index_next else None

        env = self._env_ref()
        if env is not None and (_prev_key is not None or _next_key is not None):
            current_idx = env.cfg.viewer.env_index
            new_idx = current_idx
            if _next_key is not None and event.input == _next_key:
                new_idx = current_idx + 1
            elif _prev_key is not None and event.input == _prev_key:
                new_idx = current_idx - 1
            else:
                new_idx = current_idx  # no match yet — fall through to user callbacks

            if new_idx != current_idx:
                new_idx = max(0, min(new_idx, env.num_envs - 1))
                if new_idx != current_idx:
                    env.cfg.viewer.env_index = new_idx
                    if env.viewport_camera_controller is not None:
                        env.viewport_camera_controller.set_view_env_index(new_idx)
                    print(f"[debug_viewer] Switched to env {new_idx}/{env.num_envs - 1}")
                return True

        # Step 4: user-registered key callbacks (KEY_PRESS).
        for key_name, entry in self._user_key_callbacks.items():
            resolved = self._resolve_key(key_name)
            if resolved is not None and event.input == resolved:
                if entry.get("on_press"):
                    entry["on_press"]()
                return True

        return True

    # ------------------------------------------------------------------
    # Internal — post-update event stream
    # ------------------------------------------------------------------

    def _setup_post_update_callback(self) -> None:
        """Subscribe to the Omniverse post-update event stream."""
        try:
            import omni.kit.app

            self._post_update_sub = (
                omni.kit.app.get_app_interface()
                .get_post_update_event_stream()
                .create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._on_post_update(event)
                )
            )
        except Exception as exc:
            log.warning("[debug_viewer] Post-update subscription failed: %s", exc)

    def _teardown_post_update_callback(self) -> None:
        """Release the post-update event stream subscription."""
        try:
            if self._post_update_sub is not None:
                self._post_update_sub = None
        except Exception:
            pass

    def _on_post_update(self, _event) -> None:
        """Fire all enabled debug-vis callbacks. Called every render frame."""
        del _event
        for entry in list(self._debug_vis_registry.values()):
            if entry["enabled"]:
                try:
                    entry["callback"](0.0)
                except Exception as exc:
                    log.warning("[debug_viewer] debug_vis callback raised: %s", exc)

    # ------------------------------------------------------------------
    # Internal — free-fly camera
    # ------------------------------------------------------------------

    def _update_free_fly_camera(self, dt: float) -> None:
        """Translate the viewport camera based on currently-pressed movement keys.

        Called from :meth:`update` when free-fly mode is active.  Mouse
        rotation is handled natively by the Omniverse viewport; this method
        handles translation only so both work together.

        Camera axes are derived from the USD world transform of the active
        perspective prim each frame, so mouse rotation is automatically
        incorporated.

        Parameters
        ----------
        dt:
            Elapsed time in seconds since the last policy step.  Passed from
            ``env.step()`` so no internal wall-clock tracking is needed.
        """
        if not self._cam_keys_pressed:
            return

        dt_clamped = max(0.0, min(dt, 0.1))
        if dt_clamped < 1e-9:
            return

        try:
            import omni.usd
            from pxr import Gf, UsdGeom
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        cam_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if not cam_prim or not cam_prim.IsValid():
            return

        xform_cache = UsdGeom.XformCache()
        m = xform_cache.GetLocalToWorldTransform(cam_prim)  # Gf.Matrix4d

        # USD camera: looks down local -Z, right is +X, up is +Y.
        eye = m.ExtractTranslation()
        forward = Gf.Vec3d(-m[2][0], -m[2][1], -m[2][2]).GetNormalized()
        right = Gf.Vec3d(m[0][0], m[0][1], m[0][2]).GetNormalized()
        world_up = Gf.Vec3d(0.0, 0.0, 1.0)

        try:
            import carb.input as ci

            K = ci.KeyboardInput
        except Exception:
            return

        pressed = self._cam_keys_pressed
        speed = (
            self._cfg.free_fly_speed_boost_mps
            if (self._resolve_key(self._cfg.keys.free_fly_speed_up) or K.LEFT_SHIFT) in pressed
            else self._cfg.free_fly_speed_mps
        )

        move = Gf.Vec3d(0.0, 0.0, 0.0)

        fwd_key = self._resolve_key(self._cfg.keys.free_fly_forward) if self._cfg.keys.free_fly_forward else None
        fwd_alt = (
            self._resolve_key(self._cfg.keys.free_fly_forward_alt) if self._cfg.keys.free_fly_forward_alt else None
        )
        back_key = self._resolve_key(self._cfg.keys.free_fly_back) if self._cfg.keys.free_fly_back else None
        back_alt = self._resolve_key(self._cfg.keys.free_fly_back_alt) if self._cfg.keys.free_fly_back_alt else None
        left_key = self._resolve_key(self._cfg.keys.free_fly_left) if self._cfg.keys.free_fly_left else None
        left_alt = self._resolve_key(self._cfg.keys.free_fly_left_alt) if self._cfg.keys.free_fly_left_alt else None
        right_key = self._resolve_key(self._cfg.keys.free_fly_right) if self._cfg.keys.free_fly_right else None
        right_alt = self._resolve_key(self._cfg.keys.free_fly_right_alt) if self._cfg.keys.free_fly_right_alt else None
        up_key = self._resolve_key(self._cfg.keys.free_fly_up) if self._cfg.keys.free_fly_up else None
        down_key = self._resolve_key(self._cfg.keys.free_fly_down) if self._cfg.keys.free_fly_down else None

        if (fwd_key and fwd_key in pressed) or (fwd_alt and fwd_alt in pressed):
            move += forward
        if (back_key and back_key in pressed) or (back_alt and back_alt in pressed):
            move -= forward
        if (right_key and right_key in pressed) or (right_alt and right_alt in pressed):
            move += right
        if (left_key and left_key in pressed) or (left_alt and left_alt in pressed):
            move -= right
        if up_key and up_key in pressed:
            move += world_up
        if down_key and down_key in pressed:
            move -= world_up

        length = move.GetLength()
        if length < 1e-9:
            return

        move = move.GetNormalized() * (speed * dt_clamped)

        env = self._env_ref()
        if env is None or env.viewport_camera_controller is None:
            return

        eye_new = (eye[0] + move[0], eye[1] + move[1], eye[2] + move[2])
        lookat_new = (
            eye_new[0] + forward[0],
            eye_new[1] + forward[1],
            eye_new[2] + forward[2],
        )
        env.viewport_camera_controller.update_view_location(eye=eye_new, lookat=lookat_new)

    @staticmethod
    def _resolve_key(name: str | None):
        """Resolve a key name string to the corresponding ``carb.input.KeyboardInput`` enum value.

        Parameters
        ----------
        name:
            Attribute name on ``carb.input.KeyboardInput`` (e.g. ``"LEFT_BRACKET"``), or
            ``None`` to disable the binding (returns ``None`` immediately).

        Returns
        -------
        carb.input.KeyboardInput or None
            The resolved enum value, or ``None`` if *name* is ``None`` or the
            attribute does not exist (a warning is emitted in the latter case).
        """
        if name is None:
            return None
        try:
            import carb.input as ci

            val = getattr(ci.KeyboardInput, name, None)
            if val is None:
                log.warning("[debug_viewer] Unknown key name '%s' — no carb.input.KeyboardInput.%s", name, name)
            return val
        except Exception:
            return None
