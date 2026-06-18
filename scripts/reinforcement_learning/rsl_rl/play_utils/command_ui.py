# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
command_ui.py
=============
tkinter 기반 커맨드 컨트롤 GUI.

- 환경의 command_cfg(키, 범위)를 읽어 슬라이더 + 숫자 입력을 동적으로 생성합니다.
- Apply & Reset 버튼: 현재 슬라이더 값을 command_state에 반영 + 환경 리셋 요청.
- Reset Only  버튼: 커맨드 값 변경 없이 환경 리셋만 요청.
- 별도 데몬 스레드에서 실행되므로 메인 시뮬레이션 루프를 블로킹하지 않습니다.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk


class CommandControlUI:
    """시뮬레이션 커맨드 컨트롤 GUI.

    Parameters
    ----------
    command_state : dict
        메인 루프와 공유하는 상태 딕셔너리.
        형식: ``{"values": list[float], "reset_requested": bool}``
    labels : list[str]
        각 커맨드의 레이블 (ex. "lin_vel_x", "ang_vel" …).
    ranges : list[tuple[float, float]]
        각 커맨드의 (min, max) 범위.
    initial_values : list[float] | None
        슬라이더 초기값. None이면 각 범위의 중간값 사용.
    """

    def __init__(
        self,
        command_state: dict,
        labels: list[str],
        ranges: list[tuple[float, float]],
        initial_values: list[float] | None = None,
    ) -> None:
        self.command_state = command_state
        self.labels = labels
        self.ranges = ranges
        self.num_commands = len(labels)

        # 초기값 설정
        if initial_values is None:
            self.initial_values = [round((lo + hi) / 2.0, 4) for lo, hi in ranges]
        else:
            self.initial_values = list(initial_values)

        # command_state 초기화
        self.command_state["values"] = list(self.initial_values)
        self.command_state["reset_requested"] = False

        # 별도 데몬 스레드에서 GUI 실행
        self._thread = threading.Thread(target=self._run_ui, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _run_ui(self) -> None:
        """tkinter 메인루프. 데몬 스레드에서 실행됩니다."""
        self.root = tk.Tk()
        self.root.title("Command Control")
        self.root.resizable(False, False)

        # 색상 팔레트
        BG = "#1e1e2e"
        FG = "#cdd6f4"
        ACCENT = "#89b4fa"
        BTN_APPLY = "#a6e3a1"
        BTN_RESET = "#f38ba8"
        BTN_FG = "#1e1e2e"
        SLIDER_BG = "#313244"
        ENTRY_BG = "#45475a"

        self.root.configure(bg=BG)

        # ── 타이틀 ──────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=BG)
        title_frame.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(
            title_frame,
            text="🎮  Command Control",
            font=("Segoe UI", 14, "bold"),
            bg=BG,
            fg=ACCENT,
        ).pack(side="left")

        # ── 구분선 ──────────────────────────────────────────────────
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # ── 스크롤 가능한 슬라이더 영역 ────────────────────────────
        canvas_frame = tk.Frame(self.root, bg=BG)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=4)

        canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0, width=480)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        self._sliders_frame = tk.Frame(canvas, bg=BG)

        self._sliders_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._sliders_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 마우스 휠 스크롤
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-int(e.delta / 120), "units"))

        canvas.pack(side="left", fill="both", expand=True)
        if self.num_commands > 8:
            scrollbar.pack(side="right", fill="y")
            canvas.configure(height=500)
        else:
            canvas.configure(height=max(60 * self.num_commands, 120))

        # ── 슬라이더 + Entry 동적 생성 ─────────────────────────────
        self._slider_vars: list[tk.DoubleVar] = []
        self._entry_vars: list[tk.StringVar] = []

        for i, (label, (lo, hi)) in enumerate(zip(self.labels, self.ranges)):
            row = tk.Frame(self._sliders_frame, bg=BG, pady=4)
            row.pack(fill="x", padx=10)

            # 레이블
            tk.Label(
                row,
                text=f"{i:02d}. {label}",
                font=("Consolas", 10),
                bg=BG,
                fg=FG,
                width=26,
                anchor="w",
            ).pack(side="left")

            init_val = self.initial_values[i]
            slider_var = tk.DoubleVar(value=init_val)
            entry_var = tk.StringVar(value=f"{init_val:.4f}")

            def _on_slider(val, idx=i, evar=entry_var, svar=slider_var):
                """슬라이더 이동 시 Entry 수치 갱신."""
                evar.set(f"{svar.get():.4f}")

            resolution = max((hi - lo) / 1000.0, 1e-5)
            slider = tk.Scale(
                row,
                variable=slider_var,
                from_=lo,
                to=hi,
                orient="horizontal",
                resolution=resolution,
                length=240,
                showvalue=False,
                bg=SLIDER_BG,
                fg=FG,
                troughcolor="#585b70",
                activebackground=ACCENT,
                highlightthickness=0,
                command=_on_slider,
            )
            slider.pack(side="left", padx=6)

            entry = tk.Entry(
                row,
                textvariable=entry_var,
                width=10,
                bg=ENTRY_BG,
                fg=FG,
                insertbackground=FG,
                font=("Consolas", 10),
                relief="flat",
            )
            entry.pack(side="left", padx=4)

            def _on_entry_return(event, idx=i, svar=slider_var, evar=entry_var, lo=lo, hi=hi):
                """엔트리에서 Enter 입력 시 슬라이더 값 갱신."""
                try:
                    val = float(evar.get())
                    val = max(lo, min(hi, val))
                    svar.set(val)
                    evar.set(f"{val:.4f}")
                except ValueError:
                    evar.set(f"{svar.get():.4f}")

            entry.bind("<Return>", _on_entry_return)
            entry.bind("<FocusOut>", _on_entry_return)

            self._slider_vars.append(slider_var)
            self._entry_vars.append(entry_var)

        # ── 구분선 ──────────────────────────────────────────────────
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── 상태 레이블 ─────────────────────────────────────────────
        self._status_var = tk.StringVar(value="대기 중…")
        tk.Label(
            self.root,
            textvariable=self._status_var,
            font=("Segoe UI", 9),
            bg=BG,
            fg="#a6adc8",
            anchor="center",
        ).pack(fill="x", padx=16, pady=(0, 4))

        # ── 버튼 영역 ───────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(fill="x", padx=16, pady=(4, 14))

        btn_style = dict(
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            cursor="hand2",
            bd=0,
            padx=12,
            pady=8,
        )

        tk.Button(
            btn_frame,
            text="✅  Apply & Reset",
            bg=BTN_APPLY,
            fg=BTN_FG,
            command=self._on_apply_reset,
            **btn_style,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        tk.Button(
            btn_frame,
            text="🔄  Reset Only",
            bg=BTN_RESET,
            fg=BTN_FG,
            command=self._on_reset_only,
            **btn_style,
        ).pack(side="left", fill="x", expand=True)

        self.root.mainloop()

    def _collect_values(self) -> list[float]:
        """현재 슬라이더 값 수집."""
        return [round(var.get(), 6) for var in self._slider_vars]

    def _on_apply_reset(self) -> None:
        """커맨드 값 적용 후 환경 리셋 요청."""
        values = self._collect_values()
        self.command_state["values"] = values
        self.command_state["reset_requested"] = True
        val_str = ", ".join(f"{v:.3f}" for v in values)
        self._status_var.set(f"✅ 적용됨: [{val_str}]")

    def _on_reset_only(self) -> None:
        """커맨드 값 변경 없이 환경 리셋만 요청."""
        self.command_state["reset_requested"] = True
        self._status_var.set("🔄 리셋 요청됨 (커맨드 유지)")
