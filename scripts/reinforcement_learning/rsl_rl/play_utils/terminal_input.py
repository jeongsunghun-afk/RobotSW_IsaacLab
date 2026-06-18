# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
terminal_input.py
=================
threading 기반 터미널 커맨드 입력 핸들러.

지원 모드:
1. **일반 모드** (Go2WTW: 14개 커맨드)
       형식: <gait> <x_vel> <y_vel> <yaw_vel>
       예시: trotting 1.0 0.0 0.0
2. **단순 모드** (R_Skeleton 등: 3개 이하 커맨드)
       형식: <x_vel> <y_vel> <yaw_vel>
       예시: 1.0 0.0 0.0
3. **Interaction 모드** (Go2Interaction: _interaction_command)
       형식: <motion_id>
       예시: 0  (default)  1 (sit)  2 (lie)  3 (stand_up)

NOTE: gait_commands 모듈을 생성자 파라미터로 주입받습니다.
      (importlib 단독 로드 시 상대 임포트 불가 문제 회피)
"""

from __future__ import annotations

import threading
from collections.abc import Callable


def _print_help(
    gait_names: list[str],
    num_commands: int,
    interaction_mode: bool = False,
    motion_labels: list[str] | None = None,
) -> None:
    """도움말 출력."""
    print("\n" + "=" * 60)
    print("  [커맨드 입력 가이드]")
    print("=" * 60)

    if interaction_mode:
        print("  형식: <motion_id>")
        print("  예시: 0  (기본 자세)")
        if motion_labels:
            for i, label in enumerate(motion_labels):
                print(f"         {i}  ({label})")
    elif num_commands >= 14:
        print("  형식: <gait> <x_vel> <y_vel> <yaw_vel>")
        print("  예시: trotting 1.0 0.0 0.0")
        print("        walking  0.5 0.0 0.3")
        print(f"  보행 목록: {', '.join(gait_names)}")
    else:
        print("  형식: <x_vel> <y_vel> <yaw_vel>")
        print("  예시: 1.0 0.0 0.0")

    print("  reset  — 환경 리셋 (커맨드 유지)")
    print("  help   — 이 도움말 출력")
    print("  quit   — 시뮬레이션 종료")
    print("=" * 60 + "\n")


class TerminalCommandInput:
    """시뮬레이션 루프와 병렬로 터미널 입력을 받는 핸들러.

    Parameters
    ----------
    command_state : dict
        메인 루프와 공유하는 상태 딕셔너리.
        - ``values``: list[float] — 적용할 커맨드 벡터 (일반/단순 모드)
        - ``interaction_cmd``: int — 모션 ID (interaction 모드)
        - ``reset_requested``: bool — True이면 환경 리셋 요청
        - ``quit_requested``: bool — True이면 시뮬레이션 종료 요청
    num_commands : int
        환경의 커맨드 차원 수. 14 이상이면 Go2WTW 보행 모드 사용.
    gait_mod : module
        gait_commands 모듈 (importlib로 로드된 객체).
        GAIT_NAMES, build_go2wtw_command, build_simple_command 속성 필요.
    interaction_mode : bool
        True이면 모션 ID 입력 모드 사용 (Go2Interaction 환경).
    num_motions : int
        interaction_mode에서 유효한 모션 ID 범위 [0, num_motions).
    motion_labels : list[str]
        interaction_mode에서 각 모션 ID에 대한 설명 레이블.
    on_quit : Callable | None
        quit 명령 시 호출할 콜백.
    """

    def __init__(
        self,
        command_state: dict,
        num_commands: int,
        gait_mod,
        interaction_mode: bool = False,
        num_motions: int = 4,
        motion_labels: list[str] | None = None,
        on_quit: Callable | None = None,
    ) -> None:
        self.command_state = command_state
        self.num_commands = num_commands
        self.on_quit = on_quit
        self._interaction_mode = interaction_mode
        self._num_motions = num_motions
        self._motion_labels = motion_labels or []

        # gait_commands 함수를 멤버로 저장
        self._gait_names: list[str] = gait_mod.GAIT_NAMES
        self._build_go2wtw = gait_mod.build_go2wtw_command
        self._build_simple = gait_mod.build_simple_command
        self._use_gait = (not interaction_mode) and (num_commands >= 14)

        # 스레드 시작
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()
        _print_help(self._gait_names, num_commands, interaction_mode, self._motion_labels)

    # ------------------------------------------------------------------

    def _input_loop(self) -> None:
        """별도 스레드에서 실행: 무한 입력 루프."""
        print("[터미널 입력] 준비됨. 커맨드를 입력하세요 (help for 도움말):")
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                continue

            tokens = raw.split()
            cmd = tokens[0].lower()

            # ── 특수 명령 ────────────────────────────────────────────
            if cmd == "help":
                _print_help(
                    self._gait_names,
                    self.num_commands,
                    self._interaction_mode,
                    self._motion_labels,
                )
                continue

            if cmd == "reset":
                self.command_state["reset_requested"] = True
                print("[INFO] 환경 리셋 요청됨 (커맨드 유지)")
                continue

            if cmd == "quit":
                print("[INFO] 종료 요청됨.")
                self.command_state["quit_requested"] = True
                if self.on_quit:
                    self.on_quit()
                break

            # ── 커맨드 파싱 ──────────────────────────────────────────
            if self._interaction_mode:
                self._parse_interaction_command(tokens)
            elif self._use_gait:
                self._parse_gait_command(tokens)
            else:
                self._parse_simple_command(tokens)

    # ------------------------------------------------------------------
    # 파서
    # ------------------------------------------------------------------

    def _parse_interaction_command(self, tokens: list[str]) -> None:
        """형식: <motion_id>
        Go2Interaction 환경의 _interaction_command를 설정합니다.
        """
        if len(tokens) < 1:
            print("[ERROR] 형식 오류. 필요: <motion_id>")
            return

        try:
            motion_id = int(tokens[0])
        except ValueError:
            print(f"[ERROR] motion_id는 정수여야 합니다. 입력됨: {tokens[0]}")
            return

        if not (0 <= motion_id < self._num_motions):
            print(f"[ERROR] 유효하지 않은 motion_id: {motion_id}. 범위: 0 ~ {self._num_motions - 1}")
            if self._motion_labels:
                for i, label in enumerate(self._motion_labels):
                    print(f"  {i}: {label}")
            return

        self.command_state["interaction_cmd"] = motion_id
        label = self._motion_labels[motion_id] if motion_id < len(self._motion_labels) else f"motion_{motion_id}"
        self.command_state["cmd_str"] = label
        self.command_state["reset_requested"] = True
        print(f"[INFO] Interaction 커맨드 적용: {motion_id} ({label})")

    def _parse_gait_command(self, tokens: list[str]) -> None:
        """형식: <gait> <x_vel> <y_vel> <yaw_vel>"""
        if len(tokens) < 4:
            print(f"[ERROR] 형식 오류. 필요: <gait> <x> <y> <yaw>. 입력됨: {' '.join(tokens)}")
            print(f"        보행 목록: {', '.join(self._gait_names)}")
            return

        gait = tokens[0].lower()
        try:
            x_vel = float(tokens[1])
            y_vel = float(tokens[2])
            yaw_vel = float(tokens[3])
        except ValueError:
            print("[ERROR] 속도 값은 float이어야 합니다. 예: trotting 1.0 0.0 0.0")
            return

        try:
            values = self._build_go2wtw(gait, x_vel, y_vel, yaw_vel)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return

        self.command_state["values"] = values
        self.command_state["cmd_str"] = f"{gait}_{x_vel:.2f}_{y_vel:.2f}_{yaw_vel:.2f}"
        self.command_state["reset_requested"] = True
        print(f"[INFO] 커맨드 적용: gait={gait}, x={x_vel:.2f}, y={y_vel:.2f}, yaw={yaw_vel:.2f}")

    def _parse_simple_command(self, tokens: list[str]) -> None:
        """형식: <x_vel> <y_vel> <yaw_vel>"""
        if len(tokens) < 3:
            print(f"[ERROR] 형식 오류. 필요: <x> <y> <yaw>. 입력됨: {' '.join(tokens)}")
            return

        try:
            x_vel = float(tokens[0])
            y_vel = float(tokens[1])
            yaw_vel = float(tokens[2])
        except ValueError:
            print("[ERROR] 속도 값은 float이어야 합니다. 예: 1.0 0.0 0.0")
            return

        values = self._build_simple(x_vel, y_vel, yaw_vel)
        if self.num_commands > 3:
            existing = self.command_state.get("values", [0.0] * self.num_commands)
            values = values + existing[3:]

        self.command_state["values"] = values
        self.command_state["cmd_str"] = f"vel_{x_vel:.2f}_{y_vel:.2f}_{yaw_vel:.2f}"
        self.command_state["reset_requested"] = True
        print(f"[INFO] 커맨드 적용: x={x_vel:.2f}, y={y_vel:.2f}, yaw={yaw_vel:.2f}")
