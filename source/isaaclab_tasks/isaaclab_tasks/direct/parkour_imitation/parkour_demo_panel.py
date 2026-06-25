# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""omni.ui panel helper for Go2ParkourDemoEnv interactive demo.

Usage (injected by demo launcher, NOT called from env.__init__)::

    from parkour_demo_panel import ParkourDemoPanel

    panel = ParkourDemoPanel(env)  # no-op in headless
    # ... run loop ...
    panel.close()

GUI guard: the panel is created only when a viewport camera controller is
present (i.e. GUI mode). In headless the constructor returns a no-op sentinel
so callers need not branch.

Widgets
-------
Mode combo       : "test" | "playground" → env._demo_mode
Test section     : terrain-type dropdown (flat/hurdle/step/gap/stair) + difficulty
                   slider 0-10 → env.set_demo_tile(class_id, level)
Playground section: vx slider (training range) → env._demo_vx
                    yaw dial [-π, π] → env._demo_yaw
HUD labels       : vx / yaw / env_class / terrain_level / term counters
                   (reads existing env buffers only — no new obs added)
Reset button     : env._reset_idx(torch.tensor([0], device=env.device))
Respawn helper   : auto-respawn on playground termination
Follow/free-fly toggle: wraps the "G" shortcut already registered in DebugViewer
Teleport button  : alias for reset (env index 0 → spawn origin)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

# ---------------------------------------------------------------------------
# Virtual joystick layout constants
# ---------------------------------------------------------------------------
_JOY_SIZE: int = 180  # px — square pad side length
_JOY_KNOB_R: int = 18  # px — knob radius (visual)
_JOY_CLAMP_R: float = 70.0  # px — max displacement from centre

if TYPE_CHECKING:
    from .parkour_demo_env import Go2ParkourDemoEnv

# ---------------------------------------------------------------------------
# Terrain class catalogue (T1 interface — must match _col_to_class LUT order)
# ---------------------------------------------------------------------------
_TERRAIN_CLASSES: list[tuple[str, int]] = [
    ("flat", 0),
    ("hurdle", 1),
    ("step", 2),
    ("gap", 3),
    ("stair", 4),
]
_CLASS_NAMES: list[str] = [name for name, _ in _TERRAIN_CLASSES]
_CLASS_IDS: list[int] = [cid for _, cid in _TERRAIN_CLASSES]

# Difficulty slider integer range
_LEVEL_MIN: int = 0
_LEVEL_MAX: int = 10


def _has_gui(env: Go2ParkourDemoEnv) -> bool:
    """Return True only when an interactive viewport is available.

    Priority:
      1. livestream (WebRTC) — headless locally but rendered remotely → treat as GUI.
         carb settings key: /app/livestream/enabled  (fallback: /app/livestream/proto)
      2. SimulationContext.has_gui() (권위 있는 판정)
      3. viewport_camera_controller fallback
    """
    # 1. Livestream detection — app is headless but stream is rendered on a remote client.
    try:
        import carb  # noqa: PLC0415

        settings = carb.settings.get_settings()
        if settings.get("/app/livestream/enabled"):
            return True
        # Fallback key used in some Isaac Sim builds (value is the protocol string when active).
        proto = settings.get("/app/livestream/proto")
        if proto:
            return True
    except Exception:
        pass

    # 2. Authoritative SimulationContext query.
    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim, "has_gui"):
        try:
            if sim.has_gui():
                return True
        except Exception:
            pass

    # 3. viewport_camera_controller presence as last resort.
    return getattr(env, "viewport_camera_controller", None) is not None


class ParkourDemoPanel:
    """omni.ui panel attached to a Go2ParkourDemoEnv instance.

    Instantiation is always safe; if no GUI is present the object becomes a
    no-op sentinel and all public methods are harmless.
    """

    def __init__(self, env: Go2ParkourDemoEnv) -> None:
        self._env = env
        self._window = None
        self._hud_labels: dict[str, object] = {}
        self._respawn_enabled: bool = False

        # Virtual joystick state
        self._joy_dragging: bool = False
        self._joy_base_screen_x: float = 0.0  # screen-space left edge of pad
        self._joy_base_screen_y: float = 0.0  # screen-space top edge of pad
        self._joy_knob_placer = None  # ui.Placer that positions the knob

        if not _has_gui(env):
            _sim = getattr(env, "sim", None)
            _has_gui_val = _sim.has_gui() if (_sim is not None and hasattr(_sim, "has_gui")) else "?"
            _vcc = getattr(env, "viewport_camera_controller", "MISSING")
            print(
                f"[panel] GUI not detected → panel skipped."
                f" sim.has_gui={_has_gui_val},"
                f" viewport_camera_controller={_vcc}"
            )
            return  # headless — remain no-op

        # Deferred import: only available when the GUI kit is loaded.
        try:
            import omni.ui as ui  # noqa: PLC0415
        except ImportError:
            # omni.ui not available even though viewport exists — be safe.
            print("[panel] omni.ui import failed → panel skipped")
            return  # # 확인 필요: isaac-sim 버전에 따라 경로 다를 수 있음

        self._ui = ui
        self._build_window(ui)
        print("[panel] Parkour Demo Panel window built and shown.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Refresh HUD labels. Call once per env step from the demo loop."""
        if self._window is None:
            return
        env = self._env
        ui = self._ui

        def _safe_scalar(t: torch.Tensor | None) -> float:
            if t is None:
                return float("nan")
            try:
                return float(t[0].item())
            except Exception:
                return float("nan")

        # vx comes from _commands[:,0]
        vx_val = _safe_scalar(getattr(env, "_commands", None))
        # _demo_yaw is a plain float
        yaw_val = float(getattr(env, "_demo_yaw", 0.0))

        # env class — _env_class is (num_envs,) int tensor
        env_class_t = getattr(env, "_env_class", None)
        env_class_val = int(env_class_t[0].item()) if env_class_t is not None else -1

        # terrain level — _terrain_levels is (num_envs,) int tensor
        terrain_lvl_t = getattr(env, "_terrain_levels", None)
        terrain_lvl_val = int(terrain_lvl_t[0].item()) if terrain_lvl_t is not None else -1

        # termination counters — plain int or tensor
        term_timeout = getattr(env, "_term_timeout", None)
        term_fall = getattr(env, "_term_fall", None)
        term_goal = getattr(env, "_term_goal_reached", None)

        def _term_str(t) -> str:
            if t is None:
                return "?"
            if isinstance(t, torch.Tensor):
                return str(int(t[0].item()))
            return str(int(bool(t)))

        updates = {
            "vx": f"vx: {vx_val:.3f} m/s",
            "yaw": f"yaw: {yaw_val:.3f} rad",
            "class": f"env_class: {env_class_val}",
            "level": f"terrain_level: {terrain_lvl_val}",
            "term_timeout": f"term_timeout: {_term_str(term_timeout)}",
            "term_fall": f"term_fall: {_term_str(term_fall)}",
            "term_goal": f"term_goal: {_term_str(term_goal)}",
        }
        for key, text in updates.items():
            lbl = self._hud_labels.get(key)
            if lbl is not None:
                try:
                    lbl.text = text
                except Exception:
                    pass

        # Respawn helper: if playground mode and any terminal condition fired, reset.
        if self._respawn_enabled and env._demo_mode == "playground":
            any_term = False
            for t in (term_timeout, term_fall, term_goal):
                if t is None:
                    continue
                if isinstance(t, torch.Tensor):
                    if bool(t[0].item()):
                        any_term = True
                        break
                elif bool(t):
                    any_term = True
                    break
            if any_term:
                self._do_reset()

    def close(self) -> None:
        """Destroy the window cleanly."""
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self, ui) -> None:
        env = self._env

        # Clamp helper — read training vx range from cfg
        vx_range = getattr(env.cfg, "command_cfg", {})
        if isinstance(vx_range, dict):
            vx_lo = float(vx_range.get("lin_vel_x_range", [0.0, 1.5])[0])
            vx_hi = float(vx_range.get("lin_vel_x_range", [0.0, 1.5])[1])
        else:
            vx_lo, vx_hi = 0.0, 1.5  # # 확인 필요: cfg 구조 변경 시 업데이트

        self._window = ui.Window(
            "Parkour Demo Panel",
            width=320,
            height=560,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,  # # 확인 필요: 버전 별 상수
        )
        self._window.visible = True
        # Viewport 우측에 dock 시도 — 실패해도 floating window로 표시됨.
        # 확인 필요: dock API는 omni.ui 버전에 따라 다름.
        try:
            self._window.deferred_dock_in("Viewport", ui.DockPolicy.DO_NOTHING)
        except Exception:
            pass  # dock 실패 시 floating window 유지

        with self._window.frame:
            with ui.VStack(spacing=6):
                # ---- Mode selector ----
                ui.Label("Mode", height=18)
                _init_mode_idx = 0 if getattr(env, "_demo_mode", "test") == "test" else 1
                mode_combo = ui.ComboBox(_init_mode_idx, *["test", "playground"])

                def _on_mode_changed(model, _):
                    idx = model.get_item_value_model().as_int
                    env._demo_mode = "test" if idx == 0 else "playground"
                    _is_pg = env._demo_mode == "playground"
                    # Synchronise section visibility and collapsed state
                    test_stack.visible = not _is_pg
                    play_stack.visible = _is_pg
                    try:
                        test_stack.collapsed = _is_pg
                        play_stack.collapsed = not _is_pg
                    except Exception:
                        pass  # # 확인 필요: collapsed setter 버전 의존적

                mode_combo.model.add_item_changed_fn(_on_mode_changed)

                ui.Spacer(height=4)

                # ---- TEST section ----
                with ui.CollapsableFrame("Test Controls", collapsed=False) as test_stack:
                    with ui.VStack(spacing=4):
                        ui.Label("Terrain type", height=16)
                        terrain_combo = ui.ComboBox(0, *_CLASS_NAMES)
                        self._terrain_class_idx: int = 0  # local shadow

                        def _on_terrain_type(model, _):
                            self._terrain_class_idx = model.get_item_value_model().as_int

                        terrain_combo.model.add_item_changed_fn(_on_terrain_type)

                        ui.Label("Difficulty (0-10)", height=16)
                        level_slider = ui.IntSlider(min=_LEVEL_MIN, max=_LEVEL_MAX)
                        self._terrain_level: int = 0

                        def _on_level(model):
                            self._terrain_level = model.as_int

                        level_slider.model.add_value_changed_fn(_on_level)

                        def _apply_tile():
                            class_id = _CLASS_IDS[self._terrain_class_idx]
                            level = self._terrain_level
                            try:
                                env.set_demo_tile(class_id, level)
                            except Exception as exc:
                                print(f"[ParkourDemoPanel] set_demo_tile error: {exc}")

                        ui.Button("Apply Tile", clicked_fn=_apply_tile, height=24)

                # ---- PLAYGROUND section ----
                with ui.CollapsableFrame("Playground Controls", collapsed=True) as play_stack:
                    with ui.VStack(spacing=4):
                        ui.Label(f"vx [{vx_lo:.2f}, {vx_hi:.2f}] m/s", height=16)
                        vx_slider = ui.FloatSlider(min=vx_lo, max=vx_hi)
                        vx_slider.model.set_value(float(getattr(env, "_demo_vx", 0.5)))

                        def _on_vx(model):
                            env._demo_vx = float(model.as_float)

                        vx_slider.model.add_value_changed_fn(_on_vx)

                        ui.Label("yaw_diff [-π, π] rad", height=16)
                        yaw_slider = ui.FloatSlider(min=-math.pi, max=math.pi)
                        yaw_slider.model.set_value(0.0)

                        def _on_yaw(model):
                            raw = float(model.as_float)
                            # wrap to [-π, π]
                            env._demo_yaw = math.atan2(math.sin(raw), math.cos(raw))

                        yaw_slider.model.add_value_changed_fn(_on_yaw)

                        # ---- Virtual Joystick ----
                        ui.Spacer(height=6)
                        ui.Label("Joystick (drag: ↑=vx, ←→=yaw)", height=16)
                        ui.Label("Release returns to centre / stops", height=14)
                        ui.Spacer(height=2)

                        # yaw_max: use π/2 as a comfortable steering ceiling.
                        # Positive nx → right stick → positive _demo_yaw.
                        # NOTE: yaw_diff convention = target_yaw - heading; +  = turn left.
                        # So right-stick should give negative yaw_diff to turn right.
                        # 확인 필요: 좌우 부호 시각확인 후 yaw_sign을 +1.0 ↔ -1.0 전환.
                        yaw_max: float = math.pi / 2.0
                        yaw_sign: float = -1.0  # nx>0(right) → negative yaw_diff → 우회전

                        # Outer fixed-size frame acts as the pad background.
                        # ZStack lets the knob Placer overlay the background rectangle.
                        with ui.ZStack(width=_JOY_SIZE, height=_JOY_SIZE):
                            # Background pad
                            pad_rect = ui.Rectangle(
                                width=_JOY_SIZE,
                                height=_JOY_SIZE,
                                style={
                                    "background_color": 0xFF2A2A2A,
                                    "border_color": 0xFF666666,
                                    "border_width": 2,
                                    "border_radius": 8,
                                },
                            )

                            # Cross-hair lines (purely decorative — InvisibleButton overlay captures mouse)
                            with ui.VStack():
                                ui.Spacer()
                                ui.Line(
                                    width=_JOY_SIZE,
                                    height=1,
                                    style={"color": 0xFF444444},
                                )
                                ui.Spacer()

                            # Knob positioned via Placer (offset from top-left of ZStack)
                            knob_center: int = _JOY_SIZE // 2 - _JOY_KNOB_R
                            knob_placer = ui.Placer(
                                offset_x=knob_center,
                                offset_y=knob_center,
                                draggable=False,  # we drive position manually
                            )
                            with knob_placer:
                                ui.Circle(
                                    width=_JOY_KNOB_R * 2,
                                    height=_JOY_KNOB_R * 2,
                                    style={
                                        "background_color": 0xFFDDDDDD,
                                        "border_color": 0xFFFFFFFF,
                                        "border_width": 1,
                                    },
                                )
                            self._joy_knob_placer = knob_placer

                            # Invisible hit-test overlay on top to capture mouse events.
                            # 확인 필요: omni.ui.Rectangle이 mouse callbacks를 지원하는지
                            # 버전마다 다를 수 있음. InvisibleButton 대신 Rectangle 사용.
                            hit_rect = ui.Rectangle(
                                width=_JOY_SIZE,
                                height=_JOY_SIZE,
                                style={"background_color": 0x00000000},  # fully transparent
                            )

                        # --- Mouse handler helpers ---
                        def _joy_update_from_screen(screen_x: float, screen_y: float) -> None:
                            """Compute normalised (nx, ny) from absolute screen coords and update env."""
                            cx = self._joy_base_screen_x + _JOY_SIZE / 2.0
                            cy = self._joy_base_screen_y + _JOY_SIZE / 2.0
                            dx = screen_x - cx
                            dy = screen_y - cy
                            # clamp within circle of radius _JOY_CLAMP_R
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist > _JOY_CLAMP_R:
                                scale = _JOY_CLAMP_R / dist
                                dx *= scale
                                dy *= scale
                            nx = dx / _JOY_CLAMP_R  # [-1, 1]  right → +1
                            ny = dy / _JOY_CLAMP_R  # [-1, 1]  down  → +1

                            # vx: forward = up = -ny (screen y down is positive)
                            forward = max(0.0, -ny)
                            env._demo_vx = forward * vx_hi

                            # yaw: left/right = nx (sign convention: see comment above)
                            env._demo_yaw = yaw_sign * nx * yaw_max

                            # Move knob visually
                            knob_offset_x = int(knob_center + dx)
                            knob_offset_y = int(knob_center + dy)
                            try:
                                knob_placer.offset_x = knob_offset_x
                                knob_placer.offset_y = knob_offset_y
                            except Exception:
                                pass  # 확인 필요: offset_x setter 지원 여부

                        def _joy_reset() -> None:
                            """Return knob to centre and zero velocities."""
                            env._demo_vx = 0.0
                            env._demo_yaw = 0.0
                            try:
                                knob_placer.offset_x = knob_center
                                knob_placer.offset_y = knob_center
                            except Exception:
                                pass

                        # Mouse pressed: record fallback base position and start drag.
                        # omni.ui mouse callback (x, y) are SCREEN coords, not widget-local.
                        # We derive the pad centre from hit_rect.screen_position_x/y + _JOY_SIZE/2.
                        # _joy_base_screen_x/y serves as fallback when screen_position is unavailable
                        # (e.g. layout not yet committed): store "pressed point - half pad" so that
                        # the initial touch lands exactly at the knob centre.
                        def _on_joy_pressed(x: float, y: float, button: int, modifier: int) -> None:
                            if button != 0:
                                return
                            self._joy_dragging = True
                            # Fallback centre: treat the pressed point as the pad centre.
                            self._joy_base_screen_x = x - _JOY_SIZE / 2.0
                            self._joy_base_screen_y = y - _JOY_SIZE / 2.0
                            _joy_update_local(x, y)

                        def _on_joy_moved(x: float, y: float, button: int, modifier: int) -> None:
                            if not self._joy_dragging:
                                return
                            _joy_update_local(x, y)

                        def _on_joy_released(x: float, y: float, button: int, modifier: int) -> None:
                            self._joy_dragging = False
                            _joy_reset()

                        def _joy_update_local(lx: float, ly: float) -> None:
                            """Compute joystick displacement from SCREEN coords (lx, ly).

                            Pad centre is derived from hit_rect.screen_position_x/y when available
                            (the widget has already been laid out by the time any mouse event fires).
                            Falls back to self._joy_base_screen_x/y, which _on_joy_pressed sets to
                            (pressed_x - _JOY_SIZE/2, pressed_y - _JOY_SIZE/2) so the initial touch
                            maps to zero displacement even when screen_position is unavailable.
                            """
                            base_x = getattr(hit_rect, "screen_position_x", None)
                            base_y = getattr(hit_rect, "screen_position_y", None)
                            if base_x is None or base_y is None:
                                base_x = self._joy_base_screen_x
                                base_y = self._joy_base_screen_y
                            cx = float(base_x) + _JOY_SIZE / 2.0
                            cy = float(base_y) + _JOY_SIZE / 2.0
                            dx = lx - cx
                            dy = ly - cy
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist > _JOY_CLAMP_R:
                                s = _JOY_CLAMP_R / dist
                                dx *= s
                                dy *= s
                            nx = dx / _JOY_CLAMP_R
                            ny = dy / _JOY_CLAMP_R
                            env._demo_vx = max(0.0, -ny) * vx_hi
                            env._demo_yaw = yaw_sign * nx * yaw_max
                            try:
                                knob_placer.offset_x = int(knob_center + dx)
                                knob_placer.offset_y = int(knob_center + dy)
                            except Exception:
                                pass

                        # Attach mouse handlers to the transparent hit rectangle.
                        # 확인 필요: set_mouse_pressed_fn / set_mouse_moved_fn /
                        # set_mouse_released_fn 이 omni.ui.Rectangle에서 지원되는지
                        # 버전 의존적. 미지원시 ui.InvisibleButton + clicked_fn 또는
                        # omni.kit.app.get_app().get_post_update_event_stream() 활용 필요.
                        try:
                            hit_rect.set_mouse_pressed_fn(_on_joy_pressed)
                            hit_rect.set_mouse_moved_fn(_on_joy_moved)
                            hit_rect.set_mouse_released_fn(_on_joy_released)
                        except AttributeError:
                            # Fallback: attach to pad_rect directly.
                            # 확인 필요: pad_rect도 미지원이면 InvisibleButton으로 교체 필요.
                            try:
                                pad_rect.set_mouse_pressed_fn(_on_joy_pressed)
                                pad_rect.set_mouse_moved_fn(_on_joy_moved)
                                pad_rect.set_mouse_released_fn(_on_joy_released)
                            except AttributeError:
                                pass  # GUI without mouse callback support — joystick non-functional

                        ui.Spacer(height=4)

                        # Respawn toggle
                        ui.Label("Auto-respawn on termination", height=16)
                        respawn_check = ui.CheckBox()
                        respawn_check.model.set_value(False)

                        def _on_respawn(model):
                            self._respawn_enabled = bool(model.as_bool)

                        respawn_check.model.add_value_changed_fn(_on_respawn)

                ui.Spacer(height=4)

                # ---- Camera controls ----
                with ui.CollapsableFrame("Camera", collapsed=False):
                    with ui.VStack(spacing=4):
                        ui.Label("Follow / Free-fly (G key)", height=16)

                        def _toggle_free_fly():
                            dv = getattr(env, "_debug_viewer", None)
                            if dv is not None and hasattr(dv, "_toggle_free_fly"):
                                dv._toggle_free_fly()
                            else:
                                # fallback: synthesise keyboard event via DebugViewer internal
                                pass  # # 확인 필요: _toggle_free_fly API가 없으면 G 키 이벤트 주입 필요

                        ui.Button("Toggle Free-fly (G)", clicked_fn=_toggle_free_fly, height=24)

                ui.Spacer(height=4)

                # ---- Utility buttons ----
                with ui.CollapsableFrame("Utilities", collapsed=False):
                    with ui.VStack(spacing=4):

                        def _do_reset_btn():
                            self._do_reset()

                        ui.Button("Reset (env 0)", clicked_fn=_do_reset_btn, height=24)
                        ui.Button("Teleport to Origin", clicked_fn=_do_reset_btn, height=24)

                ui.Spacer(height=4)

                # ---- HUD ----
                with ui.CollapsableFrame("HUD", collapsed=False):
                    with ui.VStack(spacing=2):
                        for key in ("vx", "yaw", "class", "level", "term_timeout", "term_fall", "term_goal"):
                            lbl = ui.Label("—", height=16)
                            self._hud_labels[key] = lbl

        # Initial visibility/collapsed sync — driven by env._demo_mode at build time
        _is_playground = getattr(env, "_demo_mode", "test") == "playground"
        test_stack.visible = not _is_playground
        play_stack.visible = _is_playground
        try:
            test_stack.collapsed = _is_playground
            play_stack.collapsed = not _is_playground
        except Exception:
            pass  # # 확인 필요: collapsed setter는 omni.ui 버전에 따라 미지원일 수 있음

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_reset(self) -> None:
        """Reset env index 0 via _reset_idx."""
        env = self._env
        try:
            idx = torch.tensor([0], dtype=torch.long, device=env.device)
            env._reset_idx(idx)
        except Exception as exc:
            print(f"[ParkourDemoPanel] _reset_idx error: {exc}")
