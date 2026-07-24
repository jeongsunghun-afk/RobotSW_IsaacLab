#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg gui_controller — PyQt5 GUI가 순수 UDP로 sim_runner를 구동.

Position mode (순수 UDP):

    gui ──cmd(9881)──▶ sim_runner ──state(9882)──▶ gui ──relay(9883)──▶ monitor

Policy mode (deployable jit를 **GUI 프로세스 내부에서** 직접 추론):

    gui(PolicyInferenceThread) ──POLICY_ACT(9886)──▶ sim_runner ──POLICY_STATE(9885)──▶ gui
                                ──REAL_ACT(9887)──▶ real ──REAL_STATE(9888)──▶ gui  (seam)

50Hz로 현재 목표각 + 관절별 kp/kd를 연속 발행하고(position), sim 상태를 수신해 monitor로 중계한다.
faithful PD가 켜져 있으므로 Gains 그룹의 kp/kd가 실제 sim drive 게인에 반영된다.
Policy mode에서는 실제 로봇 배포와 동일한 self-contained jit(``deployable_policy.pt``)를 로드해
policy_runner_bipedleg.py 와 bit-parity인 obs를 구성하고 요청-응답 lockstep으로 추론한다.

8-DOF 2족(HL 4관절 + HR 4관절)이라 관절 콤보에는 8개 레이블이 모두 나온다.

버튼:
    - Home (default): 중립(0) 자세로 보간 이동.
    - Joint Step: 선택 관절만 delta 스텝(soft limit 클램프).
    - Sine Sweep: 선택 관절에 사인 궤적(soft limit 클램프).
    - Gains: 선택 관절 kp/kd 실시간 변경(faithful PD).
    - Monitor: action/sim 실시간 plot 창(별도 프로세스).

GUI는 추후 ROS2 확장을 위해 **시스템 python3(/usr/bin/python3, py3.10)**로 실행한다(run_gui_controller.sh 가
conda를 비활성화하고 시스템 python3로 기동 — rclpy 호환). Position mode는 torch 없이 동작하고, Policy mode는
시스템 python3에 설치된 torch로 GUI 내부에서 추론한다. torch가 없거나 모델 파일이 없으면 Policy mode는
자동 비활성화되고 Position mode만 사용 가능하다.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.dirname(__file__))
import motions  # noqa: E402
import r2s_udp  # noqa: E402
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QFont  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# torch 는 policy mode 전용 — 없으면(시스템 python 등) policy UI를 비활성화하고 position mode만 제공한다.
# 실제 import 실패가 GUI 기동을 막지 않도록 top-level try/except로 감싼다(PolicyState/스레드만 torch를 dereference).
# TYPE_CHECKING 하에서는 torch 를 모듈로 import 해 타입체커가 `torch.Tensor` 어노테이션을 해석할 수 있게 하고,
# 런타임에서는 try/except 로 미설치 시 torch=None + _TORCH_OK=False (position mode는 torch 미참조로 동작).
if TYPE_CHECKING:
    import torch

    _TORCH_OK = True
else:
    try:
        import torch  # noqa: E402

        _TORCH_OK = True
    except ImportError:  # pragma: no cover - 시스템 python(torch 없음) 경로
        torch = None
        _TORCH_OK = False

# 기본 deployable 모델 경로 (export_deployable_bipedleg.py 산출물).
_DEFAULT_MODEL_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..",
    "logs",
    "rsl_rl",
    "hindLeg_history_direct",
    "2026-07-22_18-05-50_history_60_baseline",
    "exported",
    "deployable_policy.pt",
)

# --- policy obs/target 계약 (policy_runner_bipedleg.py 와 bit-parity) ---
NUM_JOINTS: int = r2s_udp.NUM_JOINTS  # 8
POLICY_DIM: int = 3 + 3 + NUM_JOINTS * 3 + 4  # gravity + cmd + (jpos,jvel,action) + clock = 34
HISTORY_LEN: int = 10
ACTION_SCALE: float = 0.25
GAIT_PERIOD: float = 0.6  # s
STEP_DT: float = 0.02  # s (decimation 4 / 200 Hz) → 50 Hz control
DEFAULT_JOINT_POS: list[float] = [0.0] * NUM_JOINTS  # hind_leg USD default 전부 0
X_VEL_RANGE: tuple[float, float] = (-0.5, 2.0)
YAW_RANGE: tuple[float, float] = (-0.5, 0.5)

FRAME_HZ: float = 50.0
FRAME_PERIOD_S: float = 1.0 / FRAME_HZ
PUBLISH_HZ: float = 50.0
PUBLISH_PERIOD_S: float = 1.0 / PUBLISH_HZ
SEQUENCE_DURATION_S: float = 1.5
STEP_DURATION_S: float = 0.5
HOST: str = "127.0.0.1"

# 게인 슬라이더 상한 — 이 리그의 실측 게인은 kp 12~65 / kd 1.0~6.0 으로, R_Skeleton 5-DOF 리그
# (kp 300 / kd 5)보다 한 자릿수 낮다. 상한을 1000/100 그대로 두면 기본값이 눈금 바닥에 붙어
# 사실상 조작이 불가능하므로, 기본값이 대략 중간에 오도록 kp 0~150 / kd 0~15 로 재조정한다.
# step도 이 스케일에 맞춰 잘게 잡는다(기존 10.0/0.5는 kp=12 관절에서 너무 거칠다).
KP_RANGE: tuple[float, float] = (0.0, 150.0)
KP_STEP: float = 1.0
KD_RANGE: tuple[float, float] = (0.0, 15.0)
KD_STEP: float = 0.1

_STYLESHEET: str = """
QWidget { background-color: #1c1e26; color: #e6e8ef;
    font-family: "Segoe UI", "Ubuntu", "Roboto", "DejaVu Sans", sans-serif; font-size: 13px; }
QMainWindow { background-color: #1c1e26; }
#titleLabel { font-size: 19px; font-weight: 600; color: #f4f6fb; }
#subtitleLabel { font-size: 11px; color: #8a8fa3; }
#divider { background-color: #33364688; max-height: 1px; border: none; }
QGroupBox { background-color: #242733; border: 1px solid #33364a; border-radius: 8px;
    margin-top: 14px; padding: 14px 10px 10px 10px; font-weight: 600; color: #c7cbe0; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px;
    padding: 0 6px; color: #9fb4ff; }
QLabel { color: #c7cbe0; background: transparent; }
QPushButton { background-color: #2c2f3d; color: #e6e8ef; border: 1px solid #3d4157;
    border-radius: 6px; padding: 7px 16px; font-weight: 500; }
QPushButton:hover { background-color: #363a4c; border-color: #4a4f6b; }
QPushButton:pressed { background-color: #23252f; }
QPushButton:disabled { background-color: #23252f; color: #5c6079; border-color: #2c2f3d; }
QPushButton#primaryButton { background-color: #4f7dfb; border-color: #4f7dfb; color: #ffffff; }
QPushButton#primaryButton:hover { background-color: #6b93fc; border-color: #6b93fc; }
QPushButton#dangerButton { background-color: #d9534f; border-color: #d9534f; color: #ffffff; }
QPushButton#dangerButton:hover { background-color: #e46864; }
QComboBox, QDoubleSpinBox { background-color: #1c1e26; border: 1px solid #3d4157;
    border-radius: 5px; padding: 5px 8px; min-height: 20px; }
QComboBox QAbstractItemView { background-color: #242733; border: 1px solid #3d4157;
    selection-background-color: #4f7dfb; outline: none; }
#statusBar { background-color: #242733; border: 1px solid #33364a; border-radius: 8px; }
#statusLabel { color: #9fb4ff; font-weight: 600; padding: 8px 14px; }
"""


class PolicyState:
    """clock / history / prev_action 상태를 관리하며 매 step obs(34)를 만든다.

    **policy_runner_bipedleg.py 의 PolicyState 와 bit-parity** (한 글자도 바꾸지 말 것). obs 순서는
    학습 hind_leg_env._get_observations 와 동일: gravity(3), cmd(3), q-default(8), dq(8),
    prev_action(8), clock(4). 전부 articulation 순서.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.default = torch.tensor(DEFAULT_JOINT_POS, device=device)
        self.reset()

    def reset(self) -> None:
        self.phase = 0.0  # gait clock ∈ [0,1)
        self.prev_action = torch.zeros(NUM_JOINTS, device=self.device)
        self.history = None  # 첫 obs에서 10× 복제로 초기화

    def _clock_obs(self) -> torch.Tensor:
        phi_hl = self.phase
        phi_hr = (self.phase + 0.5) % 1.0
        two_pi = 2.0 * math.pi
        return torch.tensor(
            [
                math.sin(two_pi * phi_hl),
                math.cos(two_pi * phi_hl),
                math.sin(two_pi * phi_hr),
                math.cos(two_pi * phi_hr),
            ],
            device=self.device,
        )

    def build_obs(self, q: torch.Tensor, dq: torch.Tensor, gravity: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """policy obs(34) 구성. q/dq/gravity/prev_action 모두 articulation 순서."""
        obs = torch.cat([gravity, cmd, q - self.default, dq, self.prev_action, self._clock_obs()], dim=0)
        return obs  # (34,)

    def push_history(self, obs: torch.Tensor) -> torch.Tensor:
        """history 버퍼 갱신 후 반환. 첫 스텝은 10× 복제(학습 episode_length_buf<=1 분기), 이후 roll."""
        if self.history is None:
            self.history = obs.unsqueeze(0).repeat(HISTORY_LEN, 1)  # (10, 34)
        else:
            self.history = torch.cat([self.history[1:], obs.unsqueeze(0)], dim=0)
        return self.history

    def advance(self, raw_action: torch.Tensor) -> None:
        """clock 진행 + prev_action 갱신 (obs 구성 후 호출)."""
        self.phase = (self.phase + STEP_DT / GAIT_PERIOD) % 1.0
        self.prev_action = raw_action.clone()


class PolicyInferenceThread(QThread):
    """Deployable jit 정책을 **GUI 프로세스 내부에서** lockstep으로 구동하는 스레드.

    policy_runner_bipedleg.py 의 obs 구성(PolicyState)과 요청-응답 lockstep을 그대로 이식한다.
    Qt 이벤트 루프를 막지 않도록 blocking recv 를 이 스레드에서 수행하고, 자세(grav_z)는 signal 로
    메인 스레드에 전달한다. 스레드는 GUI 수명 동안 살아 있고, 공유 command(mode/source/x_vel/yaw)를
    매 루프 읽는다: mode=0(idle)이면 action 미송신 → sim 은 step 하지 않고 직립 유지, mode=1(run)이면 폐루프.

    lockstep 불변식(**중요**): action 송신 → sim state blocking 대기 → 추론 → 다음 action.
    1 action = 1 sim step. 자유 타이머로 free-running 하면 자유베이스 2족이 넘어진다.
    """

    # grav_z, phase, x_vel, yaw — 1초(50 step)마다 방출 (상태바 갱신용).
    status = pyqtSignal(float, float, float, float)
    failed = pyqtSignal(str)

    def __init__(self, model_path: str, device: str, real_host: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._device = device
        self._real_host = real_host
        self._lock = threading.Lock()
        self._mode = 0  # 0=idle, 1=run
        self._source = 0  # 0=sim, 1=real
        self._x_vel = 0.0
        self._yaw = 0.0
        self._stop = False

    def set_command(self, mode: int, source: int, x_vel: float, yaw: float) -> None:
        """메인 스레드에서 공유 command 갱신 (학습 범위로 클램프)."""
        with self._lock:
            self._mode = int(mode)
            self._source = int(source)
            self._x_vel = max(X_VEL_RANGE[0], min(X_VEL_RANGE[1], float(x_vel)))
            self._yaw = max(YAW_RANGE[0], min(YAW_RANGE[1], float(yaw)))

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: C901 - 이식된 lockstep 루프(policy_runner 와 1:1)
        try:
            model = torch.jit.load(self._model_path, map_location=self._device).eval()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"모델 로드 실패: {exc}")
            return
        device = self._device
        ps = PolicyState(device)

        # --- 소켓 (policy_runner_bipedleg.py 와 동일 대역) ---
        sim_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # sim rich state 수신
        sim_state_sock.bind((HOST, r2s_udp.POLICY_STATE_PORT))
        sim_state_sock.setblocking(False)
        real_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # real state 수신 (seam)
        real_state_sock.bind((HOST, r2s_udp.REAL_STATE_PORT))
        real_state_sock.setblocking(False)
        act_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # action 송신 (sim + real)
        mon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # monitor 중계 (action vs sim)
        sim_act_addr = (HOST, r2s_udp.POLICY_ACT_PORT)
        real_act_addr = (self._real_host, r2s_udp.REAL_ACT_PORT) if self._real_host else None
        mon_addr = (HOST, r2s_udp.MONITOR_PORT)

        default_np = ps.default.cpu().numpy()
        seq = 0
        prev_mode = 0

        def recv_state_blocking(sock, timeout: float = 0.2):
            """선택된 source 소켓에서 state 하나 blocking 대기 후 latest-wins drain. 없으면 None."""
            sock.settimeout(timeout)
            try:
                data, _ = sock.recvfrom(4096)
            except (TimeoutError, OSError):
                return None
            latest = r2s_udp.unpack_policy_state(data)
            sock.setblocking(False)
            while True:
                try:
                    d, _ = sock.recvfrom(4096)
                except (BlockingIOError, OSError):
                    break
                p = r2s_udp.unpack_policy_state(d)
                if p is not None:
                    latest = p
            return latest

        def send_action(target) -> None:
            packet = r2s_udp.pack_policy_act(seq, target)
            act_sock.sendto(packet, sim_act_addr)  # sim
            if real_act_addr is not None:
                act_sock.sendto(packet, real_act_addr)  # real (seam)

        try:
            while not self._stop:
                with self._lock:
                    mode, source, x_vel, yaw = self._mode, self._source, self._x_vel, self._yaw

                # run 진입(idle→run): clock/history/prev_action 초기화 + 시동 action(sim 회신 주소 학습).
                if prev_mode == 0 and mode == 1:
                    ps.reset()
                    send_action(default_np)
                    seq += 1
                prev_mode = mode

                if mode == 0:
                    self.msleep(10)  # idle: action 미송신 → sim step 안 함 → upright 유지
                    continue

                # 방금 보낸 action에 대한 sim/real state 대기 (요청-응답 lockstep).
                src_sock = real_state_sock if source == 1 else sim_state_sock
                state = recv_state_blocking(src_sock)
                if state is None:
                    send_action(default_np)  # 응답 없음 → 재시동
                    seq += 1
                    continue

                # 추론 → 다음 action (articulation 순서). deployable jit 는 estimator/history_encoder 내장이라
                # 호출은 model(proprio, history) 뿐 (priv_explicit 불필요).
                q = torch.tensor(state["q"], device=device)
                dq = torch.tensor(state["dq"], device=device)
                gravity = torch.tensor(state["gravity"], device=device)
                cmd_vec = torch.tensor([x_vel, 0.0, yaw], device=device)  # y_vel≡0

                obs_policy = ps.build_obs(q, dq, gravity, cmd_vec)  # (34,)
                hist = ps.push_history(obs_policy)  # (10,34)
                with torch.inference_mode():
                    raw_action = model(obs_policy.unsqueeze(0), hist.unsqueeze(0))[0]  # (8,)
                ps.advance(raw_action)
                target = (ACTION_SCALE * raw_action + ps.default).cpu().numpy()
                send_action(target)

                # monitor 중계 (action_q=target vs sim q/dq; tau는 rich state에 없어 0).
                mon_sock.sendto(
                    r2s_udp.pack_monitor(seq, target, state["q"], state["dq"], [0.0] * NUM_JOINTS), mon_addr
                )
                seq += 1

                # 1초(50 step)마다 자세 로깅 — grav_z≈-1이면 직립, 0/양수면 기울어짐/전도.
                if seq % 50 == 0:
                    gz = float(gravity[2])
                    print(
                        f"[gui_infer] t={seq * STEP_DT:5.1f}s  grav_z={gz:+.2f}  phase={ps.phase:.2f}  "
                        f"x_vel={x_vel:+.2f} yaw={yaw:+.2f}  src={'real' if source == 1 else 'sim'}",
                        flush=True,
                    )
                    self.status.emit(gz, float(ps.phase), float(x_vel), float(yaw))
        finally:
            for s in (sim_state_sock, real_state_sock, act_sock, mon_sock):
                s.close()


class UdpLink:
    """순수 UDP 링크 — cmd 발신 / state 수신 / monitor 중계."""

    def __init__(self) -> None:
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_addr = (HOST, r2s_udp.CMD_PORT)
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind((HOST, r2s_udp.STATE_PORT))
        self._state_sock.setblocking(False)
        self._mon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._mon_addr = (HOST, r2s_udp.MONITOR_PORT)
        self._seq = 0

    def send_cmd(self, q: list[float], kp: list[float], kd: list[float]) -> None:
        """목표각 q + 관절별 kp/kd 발행(dq=0, tau=0)."""
        self._seq += 1
        zeros = [0.0] * r2s_udp.NUM_JOINTS
        self._cmd_sock.sendto(r2s_udp.pack_cmd(self._seq, q, zeros, kp, kd, zeros), self._cmd_addr)

    def recv_state_latest(self) -> dict | None:
        """수신 큐를 비우고 최신 상태만 반환(latest-wins). 없으면 None."""
        latest = None
        while True:
            try:
                data, _ = self._state_sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            parsed = r2s_udp.unpack_state(data)
            if parsed is not None:
                latest = parsed
        return latest

    def send_monitor(self, action_q: list[float], sim_q, sim_dq, sim_tau) -> None:
        self._mon_sock.sendto(r2s_udp.pack_monitor(self._seq, action_q, sim_q, sim_dq, sim_tau), self._mon_addr)

    def close(self) -> None:
        self._cmd_sock.close()
        self._state_sock.close()
        self._mon_sock.close()


class MainWindow(QMainWindow):
    """R2S-BipedLeg GUI main window."""

    def __init__(
        self, link: UdpLink, model_path: str | None = None, device: str = "cpu", real_host: str | None = None
    ) -> None:
        super().__init__()
        self._link = link
        # policy mode: deployable jit 경로/디바이스. 모델이 있고 torch가 있어야 policy UI 활성.
        self._model_path = model_path
        self._device = device
        self._real_host = real_host
        # `model_path is not None` 로 좁혀야 os.path.isfile(model_path) 가 None 경고 없이 통과.
        self._policy_available = model_path is not None and _TORCH_OK and os.path.isfile(model_path)
        self._policy_thread: PolicyInferenceThread | None = None
        # 중립(0) 자세에서 시작 — sim default_joint_pos 와 일치(스트림 점프 없음).
        self._current_pose: list[float] = list(motions.DEFAULT_POSE)
        self._latest_pose: list[float] = list(motions.DEFAULT_POSE)
        self._kp: list[float] = list(motions.DEFAULT_KP)
        self._kd: list[float] = list(motions.DEFAULT_KD)
        self._latest_sim: dict | None = None

        self._sequence: list[list[float]] = []
        self._sequence_idx: int = 0
        self._sequence_timer = QTimer(self)
        self._sequence_timer.timeout.connect(self._on_sequence_tick)

        self._sine_timer = QTimer(self)
        self._sine_timer.timeout.connect(self._on_sine_tick)
        self._sine_base_pose: list[float] = list(motions.DEFAULT_POSE)
        self._sine_start_time: float = 0.0

        self._monitor_proc: subprocess.Popen | None = None

        # policy mode 상태
        self._mode: str = "position"  # "position" | "policy"
        self._policy_run: int = 0  # 0=idle, 1=run
        self._policy_source: int = 0  # 0=sim, 1=real
        self._x_vel: float = 0.0
        self._yaw: float = 0.0

        self.setWindowTitle("R2S-BipedLeg Controller")
        self._build_ui()

        self._publish_timer = QTimer(self)
        self._publish_timer.timeout.connect(self._on_publish_tick)
        self._publish_timer.start(int(PUBLISH_PERIOD_S * 1000))

    # -- UI --

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        title_label = QLabel("R2S-BipedLeg Controller")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("8-DOF biped leg (HL/HR) — real2sim UDP position control")
        subtitle_label.setObjectName("subtitleLabel")
        header_title = QVBoxLayout()
        header_title.setSpacing(2)
        header_title.addWidget(title_label)
        header_title.addWidget(subtitle_label)
        header_layout = QHBoxLayout()
        header_layout.addLayout(header_title)
        header_layout.addStretch(1)
        header_layout.addWidget(QLabel("Mode:"), alignment=Qt.AlignTop)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Position", "Policy"])
        if not self._policy_available:
            # torch 없음 or 모델 파일 없음 → Policy 항목 비활성(선택 불가). Position mode만 사용.
            self._mode_combo.model().item(1).setEnabled(False)
            reason = "torch 없음" if not _TORCH_OK else "모델 파일 없음"
            self._mode_combo.setItemData(1, f"Policy 비활성 ({reason})", Qt.ToolTipRole)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        header_layout.addWidget(self._mode_combo, alignment=Qt.AlignTop)
        monitor_btn = QPushButton("Monitor")
        monitor_btn.clicked.connect(self._on_monitor_clicked)
        header_layout.addWidget(monitor_btn, alignment=Qt.AlignTop)
        layout.addLayout(header_layout)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        layout.addWidget(divider)

        # Pose
        pose_group = QGroupBox("Pose")
        pose_layout = QHBoxLayout(pose_group)
        home_btn = QPushButton("Home (default)")
        home_btn.setObjectName("primaryButton")
        home_btn.clicked.connect(self._on_home_clicked)
        pose_layout.addWidget(home_btn)
        pose_layout.addStretch(1)
        layout.addWidget(pose_group)

        # Joint Step (콤보는 HL 4개 + HR 4개 = 8 레이블 전부)
        step_group = QGroupBox("Joint Step")
        step_layout = QHBoxLayout(step_group)
        step_layout.addWidget(QLabel("Joint:"))
        self._step_joint_combo = QComboBox()
        self._step_joint_combo.addItems(motions.JOINT_NAMES)
        step_layout.addWidget(self._step_joint_combo)
        step_layout.addWidget(QLabel("delta [rad]:"))
        self._step_delta_spin = QDoubleSpinBox()
        self._step_delta_spin.setRange(-3.14, 3.14)
        self._step_delta_spin.setSingleStep(0.05)
        self._step_delta_spin.setValue(0.2)
        step_layout.addWidget(self._step_delta_spin)
        step_btn = QPushButton("Apply Step")
        step_btn.setObjectName("primaryButton")
        step_btn.clicked.connect(self._on_step_clicked)
        step_layout.addWidget(step_btn)
        step_layout.addStretch(1)
        layout.addWidget(step_group)

        # Sine Sweep
        sine_group = QGroupBox("Sine Sweep")
        sine_layout = QHBoxLayout(sine_group)
        sine_layout.addWidget(QLabel("Joint:"))
        self._sine_joint_combo = QComboBox()
        self._sine_joint_combo.addItems(motions.JOINT_NAMES)
        sine_layout.addWidget(self._sine_joint_combo)
        sine_layout.addWidget(QLabel("Amp [rad]:"))
        self._sine_amp_spin = QDoubleSpinBox()
        self._sine_amp_spin.setRange(0.0, 1.5)
        self._sine_amp_spin.setSingleStep(0.05)
        self._sine_amp_spin.setValue(0.2)
        sine_layout.addWidget(self._sine_amp_spin)
        sine_layout.addWidget(QLabel("Freq [Hz]:"))
        self._sine_freq_spin = QDoubleSpinBox()
        self._sine_freq_spin.setRange(0.05, 5.0)
        self._sine_freq_spin.setSingleStep(0.1)
        self._sine_freq_spin.setValue(0.5)
        sine_layout.addWidget(self._sine_freq_spin)
        self._sine_start_btn = QPushButton("Start")
        self._sine_start_btn.setObjectName("primaryButton")
        self._sine_start_btn.clicked.connect(self._on_sine_start_clicked)
        sine_layout.addWidget(self._sine_start_btn)
        self._sine_stop_btn = QPushButton("Stop")
        self._sine_stop_btn.setObjectName("dangerButton")
        self._sine_stop_btn.clicked.connect(self._on_sine_stop_clicked)
        self._sine_stop_btn.setEnabled(False)
        sine_layout.addWidget(self._sine_stop_btn)
        layout.addWidget(sine_group)

        # Gains (faithful PD) — 슬라이더 상한은 이 리그의 낮은 실측 게인에 맞춰 재조정(KP_RANGE/KD_RANGE).
        gain_group = QGroupBox("Gains (faithful PD)")
        gain_layout = QHBoxLayout(gain_group)
        gain_layout.addWidget(QLabel("Joint:"))
        self._gain_joint_combo = QComboBox()
        self._gain_joint_combo.addItems(motions.JOINT_NAMES)
        self._gain_joint_combo.currentIndexChanged.connect(self._on_gain_joint_changed)
        gain_layout.addWidget(self._gain_joint_combo)
        gain_layout.addWidget(QLabel("kp:"))
        self._kp_spin = QDoubleSpinBox()
        self._kp_spin.setRange(*KP_RANGE)
        self._kp_spin.setSingleStep(KP_STEP)
        self._kp_spin.setValue(self._kp[0])
        gain_layout.addWidget(self._kp_spin)
        gain_layout.addWidget(QLabel("kd:"))
        self._kd_spin = QDoubleSpinBox()
        self._kd_spin.setRange(*KD_RANGE)
        self._kd_spin.setSingleStep(KD_STEP)
        self._kd_spin.setValue(self._kd[0])
        gain_layout.addWidget(self._kd_spin)
        gain_btn = QPushButton("Apply Gains")
        gain_btn.setObjectName("primaryButton")
        gain_btn.clicked.connect(self._on_gain_apply_clicked)
        gain_layout.addWidget(gain_btn)
        gain_layout.addStretch(1)
        layout.addWidget(gain_group)

        # position-mode 패널 묶음 (mode 전환 시 일괄 show/hide)
        self._position_groups = [pose_group, step_group, sine_group, gain_group]

        # policy-mode 패널 (초기 숨김)
        self._policy_group = self._build_policy_group()
        layout.addWidget(self._policy_group)
        self._policy_group.setVisible(False)

        layout.addStretch(1)

        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(4, 2, 4, 2)
        self._status_label = QLabel("Idle")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(self._status_label)
        layout.addWidget(status_bar)

        self.setCentralWidget(central)

    # -- policy mode UI --

    def _build_policy_group(self) -> QGroupBox:
        """학습 정책 컨트롤러 패널: Run/Stop, input source(Sim/Real), x_vel/yaw."""
        group = QGroupBox("Policy Controller (deployable jit — GUI 내부 추론)")
        v = QVBoxLayout(group)

        row1 = QHBoxLayout()
        self._policy_run_btn = QPushButton("Run")
        self._policy_run_btn.setObjectName("primaryButton")
        self._policy_run_btn.clicked.connect(self._on_policy_run_clicked)
        row1.addWidget(self._policy_run_btn)
        self._policy_stop_btn = QPushButton("Stop")
        self._policy_stop_btn.setObjectName("dangerButton")
        self._policy_stop_btn.clicked.connect(self._on_policy_stop_clicked)
        self._policy_stop_btn.setEnabled(False)
        row1.addWidget(self._policy_stop_btn)
        row1.addSpacing(20)
        row1.addWidget(QLabel("Input source:"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(["Sim", "Real"])
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        row1.addWidget(self._source_combo)
        row1.addStretch(1)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("x_vel [m/s] 앞뒤:"))
        self._x_vel_spin = QDoubleSpinBox()
        self._x_vel_spin.setRange(-0.5, 2.0)
        self._x_vel_spin.setSingleStep(0.1)
        self._x_vel_spin.setValue(0.0)
        self._x_vel_spin.valueChanged.connect(self._on_x_vel_changed)
        row2.addWidget(self._x_vel_spin)
        row2.addSpacing(24)
        row2.addWidget(QLabel("yaw [rad/s] 좌우:"))
        self._yaw_spin = QDoubleSpinBox()
        self._yaw_spin.setRange(-0.5, 0.5)
        self._yaw_spin.setSingleStep(0.05)
        self._yaw_spin.setValue(0.0)
        self._yaw_spin.valueChanged.connect(self._on_yaw_changed)
        row2.addWidget(self._yaw_spin)
        row2.addStretch(1)
        v.addLayout(row2)

        hint = QLabel(
            "x_vel∈[-0.5,2.0], yaw∈[-0.5,0.5] (학습 범위). yaw 추종은 약함. Stop=정지(idle). "
            "GUI가 직접 추론해 sim으로 lockstep 전송(real은 seam)."
        )
        hint.setObjectName("subtitleLabel")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return group

    def _ensure_policy_thread(self) -> PolicyInferenceThread | None:
        """PolicyInferenceThread 를 GUI 수명 동안 1회만 생성해 idle 로 기동한다(소켓 rebind 회피).

        policy mode 진입 시 호출. 이후 mode/source/x_vel/yaw 는 set_command 로만 갱신하고, 스레드는
        closeEvent 에서만 정지한다. → Position↔Policy 토글을 반복해도 9885 포트를 재바인드하지 않는다.
        """
        if not self._policy_available or self._model_path is None:
            return None
        if self._policy_thread is None:
            self._policy_thread = PolicyInferenceThread(self._model_path, self._device, self._real_host)
            self._policy_thread.status.connect(self._on_policy_status)
            self._policy_thread.failed.connect(self._on_policy_failed)
            self._policy_thread.start()  # run() 진입: 모델 로드 후 idle 루프(action 미송신)
        return self._policy_thread

    def _push_policy_command(self) -> None:
        """현재 UI 상태(run/source/x_vel/yaw)를 추론 스레드로 전달."""
        if self._policy_thread is not None:
            self._policy_thread.set_command(self._policy_run, self._policy_source, self._x_vel, self._yaw)

    def _on_mode_changed(self, idx: int) -> None:
        self._mode = "policy" if idx == 1 else "position"
        is_policy = self._mode == "policy"
        for g in self._position_groups:
            g.setVisible(not is_policy)
        self._policy_group.setVisible(is_policy)
        if is_policy:
            # policy 진입: 스레드 기동(1회). 시작은 idle(Run 눌러야 폐루프). 현재 command를 반영.
            self._ensure_policy_thread()
            self._push_policy_command()
            self._status_label.setText("Mode: policy (idle — Run 을 눌러 시작)")
        else:
            # position 복귀: 스레드는 살려두고 idle 명령만(로봇이 마지막 target에 멈춰있지 않도록).
            self._policy_run = 0
            self._policy_run_btn.setEnabled(True)
            self._policy_stop_btn.setEnabled(False)
            self._push_policy_command()
            self._status_label.setText("Mode: position")

    def _on_policy_run_clicked(self) -> None:
        self._policy_run = 1
        self._policy_run_btn.setEnabled(False)
        self._policy_stop_btn.setEnabled(True)
        self._push_policy_command()
        src = "Real" if self._policy_source == 1 else "Sim"
        self._status_label.setText(f"Policy running (source={src})")

    def _on_policy_stop_clicked(self) -> None:
        self._policy_run = 0
        self._x_vel = 0.0
        self._yaw = 0.0
        self._x_vel_spin.setValue(0.0)
        self._yaw_spin.setValue(0.0)
        self._policy_run_btn.setEnabled(True)
        self._policy_stop_btn.setEnabled(False)
        self._push_policy_command()
        self._status_label.setText("Policy stopped (idle)")

    def _on_source_changed(self, idx: int) -> None:
        self._policy_source = idx  # 0=sim, 1=real
        self._push_policy_command()
        self._status_label.setText(f"Input source: {'Real' if idx == 1 else 'Sim'}")

    def _on_x_vel_changed(self, val: float) -> None:
        self._x_vel = val
        self._push_policy_command()

    def _on_yaw_changed(self, val: float) -> None:
        self._yaw = val
        self._push_policy_command()

    def _on_policy_status(self, grav_z: float, phase: float, x_vel: float, yaw: float) -> None:
        """추론 스레드가 1초마다 방출하는 자세를 상태바에 표시."""
        if self._policy_run == 1:
            self._status_label.setText(
                f"Policy running — grav_z={grav_z:+.2f} (≈-1 직립)  phase={phase:.2f}  "
                f"x_vel={x_vel:+.2f} yaw={yaw:+.2f}"
            )

    def _on_policy_failed(self, msg: str) -> None:
        """추론 스레드 모델 로드 실패 → policy UI 비활성."""
        self._policy_available = False
        self._status_label.setText(f"Policy 오류: {msg}")

    # -- sequence playback --

    def _play_poses(self, seq: list[list[float]]) -> None:
        self._stop_sine()
        self._sequence_timer.stop()
        self._sequence = seq
        self._sequence_idx = 0
        self._sequence_timer.start(int(FRAME_PERIOD_S * 1000))

    def _play_sequence_to(self, goal_pose: list[float], duration_s: float) -> None:
        num_steps = max(1, int(duration_s * FRAME_HZ))
        self._play_poses(motions.interpolate_sequence(self._current_pose, goal_pose, num_steps))

    def _on_sequence_tick(self) -> None:
        if self._sequence_idx >= len(self._sequence):
            self._sequence_timer.stop()
            return
        pose = self._sequence[self._sequence_idx]
        self._latest_pose = pose
        self._current_pose = pose
        self._sequence_idx += 1

    # -- pose buttons --

    def _on_home_clicked(self) -> None:
        self._status_label.setText("Moving to home (default) pose...")
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)

    def _on_step_clicked(self) -> None:
        joint_idx = self._step_joint_combo.currentIndex()
        delta = self._step_delta_spin.value()
        goal_pose = motions.step_pose(self._current_pose, joint_idx, delta)
        self._status_label.setText(f"Stepping {motions.JOINT_NAMES[joint_idx]} ({delta:+.2f} rad)...")
        self._play_sequence_to(goal_pose, STEP_DURATION_S)

    # -- sine --

    def _on_sine_start_clicked(self) -> None:
        self._sequence_timer.stop()
        self._sine_base_pose = list(self._current_pose)
        self._sine_start_time = time.monotonic()
        self._sine_start_btn.setEnabled(False)
        self._sine_stop_btn.setEnabled(True)
        self._status_label.setText(f"Sine sweep on {self._sine_joint_combo.currentText()}...")
        self._sine_timer.start(int(FRAME_PERIOD_S * 1000))

    def _on_sine_tick(self) -> None:
        joint_idx = self._sine_joint_combo.currentIndex()
        amplitude = self._sine_amp_spin.value()
        frequency = self._sine_freq_spin.value()
        t = time.monotonic() - self._sine_start_time
        pose = motions.sine_offset(self._sine_base_pose, joint_idx, amplitude, frequency, t)
        self._latest_pose = pose
        self._current_pose = pose

    def _on_sine_stop_clicked(self) -> None:
        self._stop_sine()
        self._status_label.setText("Sine sweep stopped")

    def _stop_sine(self) -> None:
        self._sine_timer.stop()
        self._sine_start_btn.setEnabled(True)
        self._sine_stop_btn.setEnabled(False)

    # -- gains (faithful PD) --

    def _on_gain_joint_changed(self, idx: int) -> None:
        self._kp_spin.setValue(self._kp[idx])
        self._kd_spin.setValue(self._kd[idx])

    def _on_gain_apply_clicked(self) -> None:
        idx = self._gain_joint_combo.currentIndex()
        self._kp[idx] = self._kp_spin.value()
        self._kd[idx] = self._kd_spin.value()
        # 게인이 한 자릿수라 소수 1자리까지 표시(5-DOF 리그의 %.0f 로는 12.5/1.1 구분 불가).
        self._status_label.setText(f"Gains {motions.JOINT_NAMES[idx]}: kp={self._kp[idx]:.1f} kd={self._kd[idx]:.2f}")

    # -- monitor --

    def _on_monitor_clicked(self) -> None:
        # 별도 프로세스로 monitor.py spawn — 무거운 matplotlib 렌더를 이 GUI의 50Hz UDP 발행
        # 루프에서 격리한다(in-process로 되돌리면 발행이 굶는다). 이미 실행 중이면 무시.
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._status_label.setText("Monitor already running")
            return
        monitor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor.py")
        self._monitor_proc = subprocess.Popen([sys.executable, monitor_py])
        self._status_label.setText("Monitor launched (separate process)")

    # -- continuous publisher + relay --

    def _on_publish_tick(self) -> None:
        # policy 모드: 추론은 PolicyInferenceThread(내부 lockstep)가 전담 — 이 50Hz 타이머는 관여하지 않는다.
        if self._mode == "policy":
            return
        # 1) 목표각 + kp/kd 발행
        self._link.send_cmd(self._latest_pose, self._kp, self._kd)
        # 2) sim 상태 수신(latest-wins)
        state = self._link.recv_state_latest()
        if state is not None:
            self._latest_sim = state
        # 3) monitor로 중계 (action + sim, time-aligned)
        if self._latest_sim is not None:
            self._link.send_monitor(
                self._latest_pose, self._latest_sim["q"], self._latest_sim["dq"], self._latest_sim["tau_est"]
            )
        else:
            zeros = [0.0] * r2s_udp.NUM_JOINTS
            self._link.send_monitor(self._latest_pose, zeros, zeros, zeros)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._publish_timer.stop()
        self._sequence_timer.stop()
        self._sine_timer.stop()
        # 추론 스레드 정지(소켓은 run() finally에서 닫힘). 여기서만 종료 → 토글 중 rebind 없음.
        if self._policy_thread is not None:
            self._policy_thread.request_stop()
            self._policy_thread.wait(2000)
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._monitor_proc.terminate()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="R2S-BipedLeg GUI controller (position + GUI 내부 policy 추론).")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL_PATH,
        help="deployable jit 경로 (기본: history_60_baseline/exported/deployable_policy.pt). "
        "없으면 Policy mode 비활성.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="정책 추론 디바이스 (기본 cpu — 273KB MLP라 CPU로 충분하고 디스플레이 GPU 경합 회피).",
    )
    parser.add_argument("--real_host", default=None, help="real 엔드포인트 IP (미지정 시 real fan-out no-op).")
    args = parser.parse_args()

    model_path = os.path.normpath(args.model) if args.model else None
    if model_path and not os.path.isfile(model_path):
        print(f"[gui_controller] 경고: 모델 파일 없음 → Policy mode 비활성: {model_path}", flush=True)

    link = UdpLink()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow(link, model_path=model_path, device=args.device, real_host=args.real_host)
    window.show()
    exit_code = app.exec_()
    link.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
