#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg gui_controller — PyQt5 GUI가 순수 UDP로 sim_runner를 구동.

Position mode (순수 UDP, **publisher 별도 프로세스** — r2s_go2 패턴):

    gui(UI, 모션 스펙만 씀) ──mp.Array──▶ publisher 프로세스 ──cmd(9881)──▶ sim_runner
                                          publisher ◀──state(9882)── sim_runner
                                          publisher ──relay(9883)──▶ monitor

Policy mode (deployable jit를 **GUI 프로세스 내부에서** 직접 추론):

    gui(PolicyInferenceThread) ──POLICY_ACT(9886)──▶ sim_runner ──POLICY_STATE(9885)──▶ gui
                                ──REAL_ACT(9887)──▶ real ──REAL_STATE(9888)──▶ gui  (seam)

프로세스 구조 — cmd 발행 AND 목표 생성(보간/사인)은 **별도 프로세스**(``publisher_process_main``)가
담당한다 (r2s_go2 gui 에서 확립한 교훈). Qt QTimer 는 UI event loop 가 바쁘면(스핀박스 드래그,
버튼 홀드) 슬립해 발행에 gap 이 생기고, 스레드는 GIL 때문에 못 푼다. UI 는 공유메모리(mp.Array)에
**모션 스펙**(어디로 몇 초에 걸쳐 가라)만 쓰고, publisher 가 경과 시간 기반으로 목표를 50Hz 균일하게
계산·발행한다. sim 상태 수신·monitor 중계·I_eff 수신도 publisher 가 맡고 실측 관절각을 공유메모리에
되써 UI(startup latch·보간 시작점)가 읽는다.

Startup: 첫 sim state 를 받을 때까지 발행을 보류하고(CMD_VALID=0), 받으면 실측 자세를 hold latch
한다 — 로봇/sim 이 다른 자세일 때 기동 즉시 목표로 스냅하는 것을 막는다(2s 타임아웃 시 default).

faithful PD가 켜져 있으므로 Gains 그룹의 kp/kd가 실제 sim drive 게인에 반영된다.
Policy mode에서는 실제 로봇 배포와 동일한 self-contained jit(``deployable_policy.pt``)를 로드해
policy_runner_bipedleg.py 와 bit-parity인 obs를 구성하고 요청-응답 lockstep으로 추론한다.

8-DOF 2족(HL 4관절 + HR 4관절)이라 관절 콤보에는 8개 레이블이 모두 나온다.

버튼:
    - Home (default): 중립(0) 자세로 보간 이동.
    - Joint Step: 선택 관절만 delta 스텝(soft limit 클램프).
    - Sine Sweep: 선택 관절에 사인 궤적(soft limit 클램프, 실행 중 파라미터 라이브 반영).
    - Gains: 선택 관절 kp/kd 실시간 변경(faithful PD).
    - Computed Gains: sim이 보낸 관절별 유효 관성 I_eff로 kp=I·ωn², kd=2ζ·I·ωn 을 계산해
      전 관절에 **임시 적용**(motions.py 기본값은 불변, Restore defaults로 복귀).
    - Monitor: action/sim 실시간 plot 창(별도 프로세스).

GUI는 추후 ROS2 확장을 위해 **시스템 python3(/usr/bin/python3, py3.10)**로 실행한다(run_gui_controller.sh 가
conda를 비활성화하고 시스템 python3로 기동 — rclpy 호환). Position mode는 torch 없이 동작하고, Policy mode는
시스템 python3에 설치된 torch로 GUI 내부에서 추론한다. torch가 없거나 모델 파일이 없으면 Policy mode는
자동 비활성화되고 Position mode만 사용 가능하다.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
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

FRAME_HZ: float = 50.0  # 시퀀스 프레임 생성 주파수 (publisher가 경과 시간으로 인덱싱)
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

# 계산 게인(I_eff 기반) 기본 파라미터 — kp = I·(2πf_n)², kd = 2ζ·I·(2πf_n).
# f_n=2Hz/ζ=0.7 은 실측 게인(hip kp 65 / kd 6)과 같은 자릿수가 나오는 보수적 시작점이다.
GAIN_FN_RANGE: tuple[float, float] = (0.5, 8.0)
GAIN_FN_DEFAULT: float = 2.0
GAIN_ZETA_RANGE: tuple[float, float] = (0.1, 2.0)
GAIN_ZETA_DEFAULT: float = 0.7

# 시작 자세 접근(startup latch) — 첫 sim state 로 실측 자세를 잡을 때까지 발행 보류(스냅 방지).
STARTUP_ACQUIRE_TIMEOUT_S: float = 2.0  # 첫 state 대기 상한 [s]. 넘으면 default 자세로 진행

# ---------------------------------------------------------------------------
# 공유 메모리 레이아웃 (multiprocessing.Array('d')). UI는 **모션 스펙**만 쓰고, publisher
# 프로세스가 50Hz 루프에서 경과 시간 기반으로 목표를 계산해 발행한다 (r2s_go2 gui 패턴).
# time.monotonic()은 리눅스에서 프로세스 간 공통(CLOCK_MONOTONIC)이라 UI가 찍은
# START_TIME을 publisher가 그대로 쓸 수 있다.
# ---------------------------------------------------------------------------
_MODE_HOLD = 0  # BASE_Q 고정 유지
_MODE_SEQUENCE = 1  # 프레임 버퍼를 경과 시간으로 인덱싱 (Home/Step 보간)
_MODE_SINE = 2  # BASE_Q + 선택 관절에 사인 주입 (경과 시간 기반)

_SM_MODE = 0
_SM_CMD_VALID = 1  # 0/1: 유효한 명령이 있는가. startup latch 전·policy mode 중엔 0 → 발행 보류
_SM_START_TIME = 2  # 모션 시작 시각 (time.monotonic)
_SM_FRAME_HZ = 3  # 시퀀스 프레임 생성 주파수 (인덱싱용)
_SM_FRAME_COUNT = 4  # 시퀀스 프레임 수
_SM_SINE_JOINT = 5
_SM_SINE_AMP = 6
_SM_SINE_FREQ = 7
_SM_MEASURED_VALID = 8  # 0/1: publisher가 sim state를 받았는가
_SM_IEFF_VALID = 9  # 0/1: publisher가 I_eff 패킷을 받았는가
_SM_KP = 10  # 10..17: 관절별 kp (r2s_go2와 달리 8관절 개별 게인)
_SM_KD = 18  # 18..25: 관절별 kd
_SM_BASE_Q = 26  # 26..33: hold 자세 / sine 기준 자세
_SM_MEASURED_Q = 34  # 34..41: sim 실측 관절각 (publisher가 state에서 씀)
_SM_CURRENT_Q = 42  # 42..49: publisher의 현재 출력 목표 (UI가 읽어 보간 시작점으로)
_SM_IEFF = 50  # 50..57: 관절별 유효 관성 [kg·m²] (publisher가 IEFF 패킷에서 씀)
_SM_FRAMES = 58  # 58..: 시퀀스 프레임 버퍼 (MAX_FRAMES × 8)
MAX_FRAMES = 512  # 512/50 = 10.24s 최대 시퀀스
_SM_LEN = _SM_FRAMES + MAX_FRAMES * NUM_JOINTS

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
            self.failed.emit(f"model load failed: {exc}")
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


def _compute_target(shared, now: float) -> tuple[list[float], list[float], list[float]] | None:
    """공유 메모리의 모션 스펙 + 경과 시간으로 현재 목표 자세를 계산한다 (publisher 프로세스에서 호출).

    Returns:
        ``(pose, kp, kd)`` 또는 명령이 없으면(``CMD_VALID=0``, startup latch 전/policy mode 중) None.
    """
    with shared.get_lock():
        if shared[_SM_CMD_VALID] < 0.5:
            return None
        mode = int(shared[_SM_MODE])
        kp = [shared[_SM_KP + i] for i in range(NUM_JOINTS)]
        kd = [shared[_SM_KD + i] for i in range(NUM_JOINTS)]
        base = [shared[_SM_BASE_Q + i] for i in range(NUM_JOINTS)]
        if mode == _MODE_SEQUENCE:
            start = shared[_SM_START_TIME]
            frame_hz = shared[_SM_FRAME_HZ]
            count = int(shared[_SM_FRAME_COUNT])
            idx = int((now - start) * frame_hz)
            idx = 0 if idx < 0 else (count - 1 if idx >= count else idx)  # 끝 프레임에서 홀드
            off = _SM_FRAMES + idx * NUM_JOINTS
            pose = [shared[off + i] for i in range(NUM_JOINTS)]
        elif mode == _MODE_SINE:
            start = shared[_SM_START_TIME]
            joint = int(shared[_SM_SINE_JOINT])
            amp = shared[_SM_SINE_AMP]
            freq = shared[_SM_SINE_FREQ]
            pose = list(base)
            pose[joint] = base[joint] + amp * math.sin(2.0 * math.pi * freq * (now - start))
        else:  # _MODE_HOLD
            pose = list(base)
    # soft limit 최종 클램프 — sine이 base+amp로 한계를 넘거나 base 자체(실측 latch)가
    # 한계 밖일 수 있다. 시퀀스 프레임은 이미 클램프된 끝점 사이 보간이지만 한 번 더는 무해.
    return motions.clamp_to_soft(pose), kp, kd


def publisher_process_main(shared, stop_flag) -> None:
    """**별도 프로세스**: 모션 스펙에서 목표를 계산해 cmd 50Hz 발행 + state/I_eff 수신. Qt와 완전 독립.

    UI event loop(위젯 조작·드래그)가 아무리 바빠도 이 프로세스는 영향받지 않는다 — 발행뿐 아니라
    목표 생성(보간/사인)까지 UI에서 격리한다(스레드는 GIL 때문에 안 됨, r2s_go2 gui 패턴).
    sim state의 실측 관절각을 공유 메모리에 되써 UI(startup latch·보간 시작점)가 읽을 수 있게 하고,
    monitor로 action+sim time-aligned 패킷을 중계한다. I_eff 패킷(sim_runner 1Hz)도 여기서 받는다.

    Args:
        shared: ``multiprocessing.Array('d', _SM_LEN)`` — 위 레이아웃 상수 참고.
        stop_flag: ``multiprocessing.Value('i')`` — 1이면 루프 종료.
    """
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_addr = (HOST, r2s_udp.CMD_PORT)
    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_sock.bind((HOST, r2s_udp.STATE_PORT))
    state_sock.setblocking(False)
    mon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mon_addr = (HOST, r2s_udp.MONITOR_PORT)

    latest_state: dict | None = None
    seq = 0
    zeros = [0.0] * NUM_JOINTS
    period = PUBLISH_PERIOD_S
    next_t = time.monotonic()
    try:
        while not stop_flag.value:
            # sim state / I_eff drain (latest-wins). 같은 포트에 두 패킷이 흘러 크기+magic으로 갈린다.
            while True:
                try:
                    data, _ = state_sock.recvfrom(4096)
                except (BlockingIOError, OSError):
                    break
                st = r2s_udp.unpack_state(data)
                if st is not None:
                    latest_state = st
                    with shared.get_lock():
                        for i in range(NUM_JOINTS):
                            shared[_SM_MEASURED_Q + i] = st["q"][i]
                        shared[_SM_MEASURED_VALID] = 1.0
                    continue
                ie = r2s_udp.unpack_ieff(data)
                if ie is not None:
                    with shared.get_lock():
                        for i in range(NUM_JOINTS):
                            shared[_SM_IEFF + i] = ie["ieff"][i]
                        shared[_SM_IEFF_VALID] = 1.0

            now = time.monotonic()
            result = _compute_target(shared, now)
            if result is not None:  # startup latch 전/policy mode 중(CMD_VALID=0)엔 발행 보류
                pose, kp, kd = result
                seq += 1
                cmd_sock.sendto(r2s_udp.pack_cmd(seq, pose, zeros, kp, kd, zeros), cmd_addr)
                with shared.get_lock():  # UI가 보간 시작점으로 읽도록 현재 목표를 되쓴다
                    for i in range(NUM_JOINTS):
                        shared[_SM_CURRENT_Q + i] = pose[i]
                # monitor 중계 (action + sim, time-aligned)
                if latest_state is not None:
                    mon_sock.sendto(
                        r2s_udp.pack_monitor(seq, pose, latest_state["q"], latest_state["dq"], latest_state["tau_est"]),
                        mon_addr,
                    )
                else:
                    mon_sock.sendto(r2s_udp.pack_monitor(seq, pose, zeros, zeros, zeros), mon_addr)

            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # 밀렸으면 리싱크(누적 드리프트 방지)
    except KeyboardInterrupt:
        pass
    finally:
        cmd_sock.close()
        state_sock.close()
        mon_sock.close()


class MainWindow(QMainWindow):
    """R2S-BipedLeg GUI main window."""

    def __init__(
        self, shared, model_path: str | None = None, device: str = "cpu", real_host: str | None = None
    ) -> None:
        super().__init__()
        # 공유 메모리(mp.Array) — 모션 스펙을 여기 쓰면 publisher 프로세스가 50Hz로 발행한다.
        # UI가 잠깐 멈춰도(사용자 조작) publisher가 목표 생성·발행을 계속한다.
        self._shared = shared
        # policy mode: deployable jit 경로/디바이스. 모델이 있고 torch가 있어야 policy UI 활성.
        self._model_path = model_path
        self._device = device
        self._real_host = real_host
        # `model_path is not None` 로 좁혀야 os.path.isfile(model_path) 가 None 경고 없이 통과.
        self._policy_available = model_path is not None and _TORCH_OK and os.path.isfile(model_path)
        self._policy_thread: PolicyInferenceThread | None = None
        self._kp: list[float] = list(motions.DEFAULT_KP)
        self._kd: list[float] = list(motions.DEFAULT_KD)

        self._sine_active: bool = False
        self._monitor_proc: subprocess.Popen | None = None

        # policy mode 상태
        self._mode: str = "position"  # "position" | "policy"
        self._policy_run: int = 0  # 0=idle, 1=run
        self._policy_source: int = 0  # 0=sim, 1=real
        self._x_vel: float = 0.0
        self._yaw: float = 0.0

        self.setWindowTitle("R2S-BipedLeg Controller")
        self._build_ui()

        # sine 실행 중 스핀박스/콤보를 바꾸면 publisher에 파라미터만 갱신(모션은 안 끊김).
        self._sine_joint_combo.currentIndexChanged.connect(self._on_sine_param_changed)
        self._sine_amp_spin.valueChanged.connect(self._on_sine_param_changed)
        self._sine_freq_spin.valueChanged.connect(self._on_sine_param_changed)

        # I_eff 수신 상태 표시 (1Hz면 충분 — 값은 상수).
        self._ieff_timer = QTimer(self)
        self._ieff_timer.timeout.connect(self._on_ieff_tick)
        self._ieff_timer.start(1000)

        # 시작 자세 접근 상태머신 — 첫 sim state 로 실측 자세를 latch 한 뒤에야 발행 시작(스냅 방지).
        self._startup_done: bool = False
        self._startup_deadline: float = time.monotonic() + STARTUP_ACQUIRE_TIMEOUT_S
        self._startup_timer = QTimer(self)
        self._startup_timer.timeout.connect(self._on_startup_tick)
        self._startup_timer.start(10)

    # -- 공유 메모리 헬퍼 --

    def _output_pose(self) -> list[float]:
        """publisher의 현재 출력 목표(보간/sine 시작점). 발행 전엔 default."""
        with self._shared.get_lock():
            if self._shared[_SM_CMD_VALID] >= 0.5:
                return [self._shared[_SM_CURRENT_Q + i] for i in range(NUM_JOINTS)]
        return list(motions.DEFAULT_POSE)

    def _write_gains(self) -> None:
        with self._shared.get_lock():
            for i in range(NUM_JOINTS):
                self._shared[_SM_KP + i] = self._kp[i]
                self._shared[_SM_KD + i] = self._kd[i]

    def _write_hold(self, base: list[float]) -> None:
        """base 자세 고정 유지 스펙을 쓴다 (CMD_VALID=1)."""
        with self._shared.get_lock():
            for i in range(NUM_JOINTS):
                self._shared[_SM_BASE_Q + i] = base[i]
            self._shared[_SM_MODE] = _MODE_HOLD
            self._shared[_SM_CMD_VALID] = 1.0

    # -- startup latch --

    def _on_startup_tick(self) -> None:
        """첫 sim state 수신 → 실측 자세 hold latch. 타임아웃 → default 자세로 진행."""
        if self._startup_done:
            self._startup_timer.stop()
            return
        with self._shared.get_lock():
            measured_valid = self._shared[_SM_MEASURED_VALID] >= 0.5
            measured = [self._shared[_SM_MEASURED_Q + i] for i in range(NUM_JOINTS)]
        if measured_valid:
            self._write_gains()
            self._write_hold(measured)
            self._status_label.setText("Startup: latched measured pose - publishing (use Home for neutral)")
        elif time.monotonic() >= self._startup_deadline:
            self._write_gains()
            self._write_hold(list(motions.DEFAULT_POSE))
            self._status_label.setText("Startup: no sim state in 2s - publishing default pose")
        else:
            return
        self._startup_done = True
        self._startup_timer.stop()

    # -- UI --

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        title_label = QLabel("R2S-BipedLeg Controller")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("8-DOF biped leg (HL/HR) - real2sim UDP position control")
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
            reason = "torch not installed" if not _TORCH_OK else "model file not found"
            self._mode_combo.setItemData(1, f"Policy disabled ({reason})", Qt.ToolTipRole)
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

        # Computed Gains — sim이 1Hz로 흘리는 관절별 유효 관성(I_eff)으로 kp/kd를 계산해
        # 전 관절에 임시 적용한다. motions.py 기본값은 건드리지 않는다(Restore로 복귀).
        cgain_group = QGroupBox("Computed Gains (I_eff based - temporary)")
        cgain_v = QVBoxLayout(cgain_group)
        self._ieff_label = QLabel("I_eff not received - start sim_runner (position mode) first")
        self._ieff_label.setObjectName("subtitleLabel")
        self._ieff_label.setWordWrap(True)
        cgain_v.addWidget(self._ieff_label)
        cgain_row = QHBoxLayout()
        cgain_row.addWidget(QLabel("f_n [Hz]:"))
        self._gain_fn_spin = QDoubleSpinBox()
        self._gain_fn_spin.setRange(*GAIN_FN_RANGE)
        self._gain_fn_spin.setSingleStep(0.1)
        self._gain_fn_spin.setValue(GAIN_FN_DEFAULT)
        cgain_row.addWidget(self._gain_fn_spin)
        cgain_row.addWidget(QLabel("zeta:"))
        self._gain_zeta_spin = QDoubleSpinBox()
        self._gain_zeta_spin.setRange(*GAIN_ZETA_RANGE)
        self._gain_zeta_spin.setSingleStep(0.05)
        self._gain_zeta_spin.setValue(GAIN_ZETA_DEFAULT)
        cgain_row.addWidget(self._gain_zeta_spin)
        cgain_compute_btn = QPushButton("Compute && Apply (all joints)")
        cgain_compute_btn.setObjectName("primaryButton")
        cgain_compute_btn.clicked.connect(self._on_gain_compute_clicked)
        cgain_row.addWidget(cgain_compute_btn)
        cgain_restore_btn = QPushButton("Restore defaults")
        cgain_restore_btn.clicked.connect(self._on_gain_restore_clicked)
        cgain_row.addWidget(cgain_restore_btn)
        cgain_row.addStretch(1)
        cgain_v.addLayout(cgain_row)
        layout.addWidget(cgain_group)

        # position-mode 패널 묶음 (mode 전환 시 일괄 show/hide)
        self._position_groups = [pose_group, step_group, sine_group, gain_group, cgain_group]

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
        group = QGroupBox("Policy Controller (deployable jit - in-GUI inference)")
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
        row2.addWidget(QLabel("x_vel [m/s] fwd/back:"))
        self._x_vel_spin = QDoubleSpinBox()
        self._x_vel_spin.setRange(-0.5, 2.0)
        self._x_vel_spin.setSingleStep(0.1)
        self._x_vel_spin.setValue(0.0)
        self._x_vel_spin.valueChanged.connect(self._on_x_vel_changed)
        row2.addWidget(self._x_vel_spin)
        row2.addSpacing(24)
        row2.addWidget(QLabel("yaw [rad/s] turn:"))
        self._yaw_spin = QDoubleSpinBox()
        self._yaw_spin.setRange(-0.5, 0.5)
        self._yaw_spin.setSingleStep(0.05)
        self._yaw_spin.setValue(0.0)
        self._yaw_spin.valueChanged.connect(self._on_yaw_changed)
        row2.addWidget(self._yaw_spin)
        row2.addStretch(1)
        v.addLayout(row2)

        hint = QLabel(
            "x_vel [-0.5, 2.0], yaw [-0.5, 0.5] (training range). Yaw tracking is weak. "
            "Stop = idle. GUI runs inference and streams lockstep to sim (real via seam)."
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
            # policy 진입: position publisher를 멈추고(CMD_VALID=0 — policy sim은 CMD 포트를
            # 안 듣지만 발행 계속은 무의미), 스레드 기동(1회). 시작은 idle(Run 눌러야 폐루프).
            self._sine_deactivate()
            with self._shared.get_lock():
                self._shared[_SM_CMD_VALID] = 0.0
            self._ensure_policy_thread()
            self._push_policy_command()
            self._status_label.setText("Mode: policy (idle - press Run to start)")
        else:
            # position 복귀: 스레드는 살려두고 idle 명령만(로봇이 마지막 target에 멈춰있지 않도록).
            self._policy_run = 0
            self._policy_run_btn.setEnabled(True)
            self._policy_stop_btn.setEnabled(False)
            self._push_policy_command()
            # publisher 재개 — 실측이 있으면 실측 자세, 없으면 마지막 출력 목표로 hold(스냅 방지).
            if self._startup_done:
                with self._shared.get_lock():
                    measured_valid = self._shared[_SM_MEASURED_VALID] >= 0.5
                    measured = [self._shared[_SM_MEASURED_Q + i] for i in range(NUM_JOINTS)]
                    current = [self._shared[_SM_CURRENT_Q + i] for i in range(NUM_JOINTS)]
                self._write_hold(measured if measured_valid else current)
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
                f"Policy running - grav_z={grav_z:+.2f} (-1 = upright)  phase={phase:.2f}  "
                f"x_vel={x_vel:+.2f} yaw={yaw:+.2f}"
            )

    def _on_policy_failed(self, msg: str) -> None:
        """추론 스레드 모델 로드 실패 → policy UI 비활성."""
        self._policy_available = False
        self._status_label.setText(f"Policy error: {msg}")

    # -- sequence playback (publisher가 경과 시간으로 인덱싱 — UI 타이머 없음) --

    def _play_sequence_to(self, goal_pose: list[float], duration_s: float) -> None:
        num_steps = max(1, min(MAX_FRAMES, int(duration_s * FRAME_HZ)))
        frames = motions.interpolate_sequence(self._output_pose(), goal_pose, num_steps)
        self._sine_deactivate()
        with self._shared.get_lock():
            for f_idx, frame in enumerate(frames):
                off = _SM_FRAMES + f_idx * NUM_JOINTS
                for i in range(NUM_JOINTS):
                    self._shared[off + i] = frame[i]
            self._shared[_SM_FRAME_COUNT] = float(len(frames))
            self._shared[_SM_FRAME_HZ] = FRAME_HZ
            self._shared[_SM_START_TIME] = time.monotonic()
            # 시퀀스가 끝나면 publisher가 마지막 프레임에서 홀드하지만, BASE_Q도 goal로 맞춰
            # 이후 HOLD 전환(sine stop 등)이 자연스럽게 이어지도록 한다.
            for i in range(NUM_JOINTS):
                self._shared[_SM_BASE_Q + i] = frames[-1][i]
            self._shared[_SM_MODE] = _MODE_SEQUENCE
            self._shared[_SM_CMD_VALID] = 1.0

    # -- pose buttons --

    def _on_home_clicked(self) -> None:
        self._status_label.setText("Moving to home (default) pose...")
        self._play_sequence_to(list(motions.DEFAULT_POSE), SEQUENCE_DURATION_S)

    def _on_step_clicked(self) -> None:
        joint_idx = self._step_joint_combo.currentIndex()
        delta = self._step_delta_spin.value()
        goal_pose = motions.step_pose(self._output_pose(), joint_idx, delta)
        self._status_label.setText(f"Stepping {motions.JOINT_NAMES[joint_idx]} ({delta:+.2f} rad)...")
        self._play_sequence_to(goal_pose, STEP_DURATION_S)

    # -- sine (publisher가 경과 시간 기반으로 생성 — UI 타이머 없음) --

    def _on_sine_start_clicked(self) -> None:
        base = self._output_pose()
        with self._shared.get_lock():
            for i in range(NUM_JOINTS):
                self._shared[_SM_BASE_Q + i] = base[i]
            self._shared[_SM_SINE_JOINT] = float(self._sine_joint_combo.currentIndex())
            self._shared[_SM_SINE_AMP] = self._sine_amp_spin.value()
            self._shared[_SM_SINE_FREQ] = self._sine_freq_spin.value()
            self._shared[_SM_START_TIME] = time.monotonic()
            self._shared[_SM_MODE] = _MODE_SINE
            self._shared[_SM_CMD_VALID] = 1.0
        self._sine_active = True
        self._sine_start_btn.setEnabled(False)
        self._sine_stop_btn.setEnabled(True)
        self._status_label.setText(f"Sine sweep on {self._sine_joint_combo.currentText()}...")

    def _on_sine_param_changed(self, _val=None) -> None:
        """sine 실행 중 관절/amp/freq 라이브 반영 (r2s_go2 패턴 — 모션 안 끊김)."""
        if not self._sine_active:
            return
        with self._shared.get_lock():
            self._shared[_SM_SINE_JOINT] = float(self._sine_joint_combo.currentIndex())
            self._shared[_SM_SINE_AMP] = self._sine_amp_spin.value()
            self._shared[_SM_SINE_FREQ] = self._sine_freq_spin.value()

    def _on_sine_stop_clicked(self) -> None:
        # 현재 출력에서 정지(hold) — base로 스냅백하지 않는다.
        self._write_hold(self._output_pose())
        self._sine_deactivate()
        self._status_label.setText("Sine sweep stopped")

    def _sine_deactivate(self) -> None:
        self._sine_active = False
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
        self._write_gains()
        # 게인이 한 자릿수라 소수 1자리까지 표시(5-DOF 리그의 %.0f 로는 12.5/1.1 구분 불가).
        self._status_label.setText(f"Gains {motions.JOINT_NAMES[idx]}: kp={self._kp[idx]:.1f} kd={self._kd[idx]:.2f}")

    # -- computed gains (I_eff 기반, 임시) --

    def _read_ieff(self) -> list[float] | None:
        with self._shared.get_lock():
            if self._shared[_SM_IEFF_VALID] < 0.5:
                return None
            return [self._shared[_SM_IEFF + i] for i in range(NUM_JOINTS)]

    def _on_ieff_tick(self) -> None:
        ieff = self._read_ieff()
        if ieff is None:
            self._ieff_label.setText("I_eff not received - start sim_runner (position mode) first")
        else:
            self._ieff_label.setText("I_eff [kg·m²]: " + "  ".join(f"{v:.3f}" for v in ieff))

    def _on_gain_compute_clicked(self) -> None:
        """kp = I·(2πf_n)², kd = 2ζ·I·(2πf_n) 을 전 관절에 **임시** 적용 (motions.py 기본값 불변)."""
        ieff = self._read_ieff()
        if ieff is None:
            self._status_label.setText("Computed gains unavailable - no I_eff (is sim_runner running?)")
            return
        wn = 2.0 * math.pi * self._gain_fn_spin.value()
        zeta = self._gain_zeta_spin.value()
        for i in range(NUM_JOINTS):
            self._kp[i] = min(KP_RANGE[1], max(KP_RANGE[0], ieff[i] * wn * wn))
            self._kd[i] = min(KD_RANGE[1], max(KD_RANGE[0], 2.0 * zeta * ieff[i] * wn))
        self._write_gains()
        sel = self._gain_joint_combo.currentIndex()
        self._kp_spin.setValue(self._kp[sel])
        self._kd_spin.setValue(self._kd[sel])
        table = "  ".join(f"{motions.JOINT_NAMES[i]}: {self._kp[i]:.1f}/{self._kd[i]:.2f}" for i in range(NUM_JOINTS))
        print(f"[gui] computed gains (f_n={wn / (2 * math.pi):.2f}Hz, ζ={zeta:.2f})  kp/kd — {table}", flush=True)
        self._status_label.setText(
            f"Computed gains applied (temp): f_n={wn / (2 * math.pi):.2f}Hz zeta={zeta:.2f} - full table in console"
        )

    def _on_gain_restore_clicked(self) -> None:
        self._kp = list(motions.DEFAULT_KP)
        self._kd = list(motions.DEFAULT_KD)
        self._write_gains()
        sel = self._gain_joint_combo.currentIndex()
        self._kp_spin.setValue(self._kp[sel])
        self._kd_spin.setValue(self._kd[sel])
        self._status_label.setText("Gains restored to defaults (measured, motions.py)")

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

    # (position 발행/monitor 중계는 publisher 프로세스 담당 — 이 UI 프로세스엔 발행 타이머가 없다.)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._startup_timer.stop()
        self._ieff_timer.stop()
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

    # 공유 메모리 + publisher 프로세스 — QApplication 생성 **전에** fork 한다 (Qt 상태를 자식이
    # 물려받지 않도록, r2s_go2 gui와 동일한 기동 순서).
    shared = mp.Array("d", _SM_LEN)
    stop_flag = mp.Value("i", 0)
    with shared.get_lock():
        shared[_SM_CMD_VALID] = 0.0  # startup latch 전 발행 보류
        for i in range(NUM_JOINTS):
            shared[_SM_KP + i] = motions.DEFAULT_KP[i]
            shared[_SM_KD + i] = motions.DEFAULT_KD[i]
            shared[_SM_BASE_Q + i] = motions.DEFAULT_POSE[i]
    publisher = mp.Process(target=publisher_process_main, args=(shared, stop_flag), daemon=True)
    publisher.start()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow(shared, model_path=model_path, device=args.device, real_host=args.real_host)
    window.show()
    exit_code = app.exec_()

    stop_flag.value = 1
    publisher.join(timeout=2.0)
    if publisher.is_alive():
        publisher.terminate()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
