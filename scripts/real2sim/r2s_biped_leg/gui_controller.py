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

Gains 그룹에서 적용한 kp/kd는 publisher 프로세스가 REAL_ACT_PORT(9887)로 GAIN(R2PK) 패킷을 통해
실기에도 갱신한다 — ACT(POLICY_ACT) 패킷은 target_q만 실어 나르므로(kp/kd 없음), 실기 쪽 게인을
바꾸는 유일한 경로다. 값이 바뀌면 즉시 + 그 외엔 1초 주기로 재송신한다. relax 중엔 보내지 않는다
(relax 동안 실제 kp/kd 대신 0을 보내면 파이 드라이버에 게인 0이 영구 잔류해, GUI가 relax 상태로
죽은 뒤 다른 peer가 engage해도 무게인 추종 불능이 된다) — relax 해제 즉시 최신 게인이 변경 감지로
다시 나간다.

버튼/컨트롤:
    - Home (default): 중립(0) 자세로 보간 이동.
    - Relax (zero torque): 무토크(limp) — sim은 kp=kd=0 CMD, 실기는 RELAX 패킷. **기동 기본 상태**라
      GUI를 켜는 것만으로는 어떤 목표도 구동하지 않는다(슬라이더/Home 조작 시 engage).
    - Joint Sliders: 8관절(leg-major) 슬라이더로 직접 자세 조작(soft limit 범위, rad/deg 표시 토글),
      "Send to robot" 체크박스로 목표를 REAL_ACT_PORT(9887)에도 fan-out(--real_host 지정 시 기본 ON).
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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
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
HOST: str = "127.0.0.1"
GAIN_RESEND_PERIOD_S: float = 1.0  # 실기 GAIN(R2PK) 주기 재송신 간격 — UDP 유실/real_runner 재시작 대비

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
_MODE_RELAX = 3  # 무토크(limp): sim엔 kp=kd=0 CMD, 실기엔 RELAX 패킷 — GUI 기동 기본 상태

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
_SM_REAL_ENABLE = 58  # 0/1: Joint Sliders가 목표를 실기(UDP 9887, REAL_ACT_PORT)로도 발행할지
_SM_FRAMES = 59  # 59..: 시퀀스 프레임 버퍼 (MAX_FRAMES × 8)
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
                # target/q/dq 는 articulation 순서라, monitor.py 가 기대하는 leg-major(motions.JOINT_NAMES)로
                # 재배열해야 한다 — 이전엔 articulation 순서 그대로 보내 monitor의 관절 레이블과 어긋났었다.
                target_lm = [target[a] for a in _ART_FOR_LEGMAJOR]
                q_lm = [state["q"][a] for a in _ART_FOR_LEGMAJOR]
                dq_lm = [state["dq"][a] for a in _ART_FOR_LEGMAJOR]
                mon_sock.sendto(r2s_udp.pack_monitor(seq, target_lm, q_lm, dq_lm, [0.0] * NUM_JOINTS), mon_addr)
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


# leg-major 인덱스 i(motions.JOINT_NAMES) ← articulation 인덱스 — monitor.py plot 중계용 재배열.
_ART_FOR_LEGMAJOR: list[int] = [0, 2, 4, 6, 1, 3, 5, 7]
# articulation 인덱스 p ← leg-major 인덱스 — pack_policy_act(articulation 순서 필요)로 실기 fan-out할 때
# Joint Sliders(leg-major)의 pose를 재배열하는 용도. _ART_FOR_LEGMAJOR의 역순열.
_LM_FOR_ART: list[int] = [0, 4, 1, 5, 2, 6, 3, 7]

_SLIDER_STEPS: int = 1000  # Joint Sliders 내부 int 해상도 (soft limit 범위를 이 스텝 수로 양자화)


def _slider_to_rad(joint_idx: int, value: int) -> float:
    """슬라이더 int 값(0..``_SLIDER_STEPS``) → soft limit 범위 내 rad."""
    lo, hi = motions.SOFT_LIMITS_RAD[joint_idx]
    return lo + (hi - lo) * value / _SLIDER_STEPS


def _rad_to_slider(joint_idx: int, rad: float) -> int:
    """rad(soft limit로 클램프) → 슬라이더 int 값(0..``_SLIDER_STEPS``)."""
    lo, hi = motions.SOFT_LIMITS_RAD[joint_idx]
    if hi <= lo:
        return 0
    frac = (rad - lo) / (hi - lo)
    return int(round(min(1.0, max(0.0, frac)) * _SLIDER_STEPS))


class RealMonitorThread(QThread):
    """real_runner TELEM(9889) 수신 스레드 — **관측 전용**, 로봇 명령에 관여하지 않는다.

    real_runner 는 peer(마지막 수신 주소)로만 회신하므로, 링크가 조용할 때(최근 1 s 내 TELEM 없음)만
    PING(8 B, 목표 없음 — real_runner 는 peer 등록만 하고 상태머신 불변)을 1 Hz 로 보낸다.
    policy 스레드가 ACT 를 흘리고 있으면 그것이 peer 를 유지하므로 이 스레드는 침묵한다.
    TELEM 포트는 STATE(9888)와 분리돼 있어 policy lockstep 수신과 충돌하지 않는다 — 보행 중 동시 사용 가능.
    """

    telem = pyqtSignal(object)  # dict: q/dq/tau/rpy/seq/hz — 10 Hz 로 최신값 방출
    failed = pyqtSignal(str)

    def __init__(self, host: str, parent=None) -> None:
        super().__init__(parent)
        self._host = host
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            rx.bind(("", r2s_udp.REAL_TELEM_PORT))
        except OSError as exc:
            self.failed.emit(f"bind {r2s_udp.REAL_TELEM_PORT} failed: {exc}")
            return
        rx.settimeout(0.05)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ping_addr = (self._host, r2s_udp.REAL_ACT_PORT)
        mon_addr = ("127.0.0.1", r2s_udp.REAL_MON_PORT)  # monitor.py plot 중계 (leg-major 재배열)

        latest: dict | None = None
        seq = 0
        last_rx = 0.0
        last_ping = 0.0
        last_emit = 0.0
        rx_stamps: list[float] = []  # 최근 1 s 수신 시각 (rate 표시용)
        try:
            while not self._stop:
                now = time.monotonic()
                if now - last_rx > 1.0 and now - last_ping > 1.0:
                    tx.sendto(r2s_udp.pack_policy_ping(seq), ping_addr)
                    seq += 1
                    last_ping = now
                try:
                    data, _ = rx.recvfrom(2048)
                except (TimeoutError, OSError):
                    data = None
                if data is not None:
                    t = r2s_udp.unpack_policy_telem(data)
                    if t is not None:
                        latest = t
                        last_rx = now
                        rx_stamps.append(now)
                        # plot 중계 — monitor.py 는 leg-major(motions.JOINT_NAMES) 순서를 기대한다.
                        q_lm = [t["q"][a] for a in _ART_FOR_LEGMAJOR]
                        dq_lm = [t["dq"][a] for a in _ART_FOR_LEGMAJOR]
                        tau_lm = [t["tau"][a] for a in _ART_FOR_LEGMAJOR]
                        tx.sendto(r2s_udp.pack_monitor(t["seq"], [0.0] * NUM_JOINTS, q_lm, dq_lm, tau_lm), mon_addr)
                if rx_stamps and now - rx_stamps[0] > 1.0:
                    rx_stamps = [s for s in rx_stamps if now - s <= 1.0]
                if now - last_emit >= 0.1:
                    last_emit = now
                    if latest is not None:
                        payload = dict(latest)
                        payload["hz"] = float(len(rx_stamps))
                        payload["stale"] = now - last_rx > 0.5
                        self.telem.emit(payload)
                    else:
                        self.telem.emit({"stale": True, "hz": 0.0})
        finally:
            rx.close()
            tx.close()


def _compute_target(shared, now: float) -> tuple[list[float], list[float], list[float], bool, bool] | None:
    """공유 메모리의 모션 스펙 + 경과 시간으로 현재 목표 자세를 계산한다 (publisher 프로세스에서 호출).

    Returns:
        ``(pose, kp, kd, real_enable, relax)`` 또는 명령이 없으면(``CMD_VALID=0``, startup latch
        전/policy mode 중) None. ``real_enable``은 Joint Sliders의 "Send to robot" 체크박스 상태,
        ``relax``면 sim엔 kp=kd=0 CMD, 실기엔 ACT 대신 RELAX 패킷이 나간다.
    """
    with shared.get_lock():
        if shared[_SM_CMD_VALID] < 0.5:
            return None
        mode = int(shared[_SM_MODE])
        kp = [shared[_SM_KP + i] for i in range(NUM_JOINTS)]
        kd = [shared[_SM_KD + i] for i in range(NUM_JOINTS)]
        base = [shared[_SM_BASE_Q + i] for i in range(NUM_JOINTS)]
        real_enable = shared[_SM_REAL_ENABLE] >= 0.5
        if mode == _MODE_RELAX:
            # 무토크: 게인 0. 목표는 실측 자세(모니터 action 라인이 현실을 따르게) — 게인 0이라 힘엔 무영향.
            measured_valid = shared[_SM_MEASURED_VALID] >= 0.5
            pose = [shared[(_SM_MEASURED_Q if measured_valid else _SM_BASE_Q) + i] for i in range(NUM_JOINTS)]
            return motions.clamp_to_soft(pose), [0.0] * NUM_JOINTS, [0.0] * NUM_JOINTS, real_enable, True
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
    return motions.clamp_to_soft(pose), kp, kd, real_enable, False


def publisher_process_main(shared, stop_flag, real_host: str | None = None) -> None:
    """**별도 프로세스**: 모션 스펙에서 목표를 계산해 cmd 50Hz 발행 + state/I_eff 수신. Qt와 완전 독립.

    UI event loop(위젯 조작·드래그)가 아무리 바빠도 이 프로세스는 영향받지 않는다 — 발행뿐 아니라
    목표 생성(보간/사인)까지 UI에서 격리한다(스레드는 GIL 때문에 안 됨, r2s_go2 gui 패턴).
    sim state의 실측 관절각을 공유 메모리에 되써 UI(startup latch·보간 시작점)가 읽을 수 있게 하고,
    monitor로 action+sim time-aligned 패킷을 중계한다. I_eff 패킷(sim_runner 1Hz)도 여기서 받는다.

    Joint Sliders의 "Send to robot" 체크박스(``_SM_REAL_ENABLE``)가 켜져 있고 ``real_host``가
    주어졌으면, 매 틱 발행 중인 목표(leg-major)를 articulation 순서로 재배열해
    ``pack_policy_act``로 실기(REAL_ACT_PORT=9887)에도 흘린다. 같은 조건에서 Gains 그룹의 kp/kd도
    articulation 순서로 재배열해 ``pack_policy_gain``(R2PK)으로 실기에 갱신한다 — 값이 바뀐 즉시,
    그 외엔 1초 주기로 재송신한다. relax 중엔 보내지 않는다(relax의 kp=kd=0을 그대로 GAIN으로 보내면
    파이 드라이버에 게인 0이 영구 잔류해 이후 engage 시 무게인 추종 불능이 되므로) — relax 해제
    즉시 최신 게인이 변경 감지로 다시 나간다.

    Args:
        shared: ``multiprocessing.Array('d', _SM_LEN)`` — 위 레이아웃 상수 참고.
        stop_flag: ``multiprocessing.Value('i')`` — 1이면 루프 종료.
        real_host: 실기 IP. None이면 실기 fan-out은 항상 no-op(체크박스도 GUI에서 비활성화됨).
    """
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_addr = (HOST, r2s_udp.CMD_PORT)
    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        state_sock.bind((HOST, r2s_udp.STATE_PORT))
    except OSError as exc:
        # bind 실패 = 대부분 gui_controller 중복 실행(먼저 뜬 인스턴스의 publisher가 9882 점유).
        # 별도 프로세스라 조용히 죽으면 "슬라이더/sine이 sim에 안 먹힘"으로만 보인다 — 크게 알린다.
        print(
            f"[publisher] FATAL: state port {r2s_udp.STATE_PORT} bind failed ({exc}) — "
            "another gui_controller instance running? Close it and restart this GUI.",
            flush=True,
        )
        return
    state_sock.setblocking(False)
    mon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mon_addr = (HOST, r2s_udp.MONITOR_PORT)
    real_act_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    real_act_addr = (real_host, r2s_udp.REAL_ACT_PORT) if real_host else None

    latest_state: dict | None = None
    seq = 0
    zeros = [0.0] * NUM_JOINTS
    period = PUBLISH_PERIOD_S
    next_t = time.monotonic()
    # GAIN(R2PK) 실기 송신 상태 — 값 변경 시 즉시 + 그 외엔 GAIN_RESEND_PERIOD_S 주기로 재송신.
    last_gain_kp: list[float] | None = None
    last_gain_kd: list[float] | None = None
    last_gain_time: float = 0.0
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
                pose, kp, kd, real_enable, relax = result
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
                # 실기 fan-out — pose는 leg-major(motions.JOINT_NAMES), pack_policy_act는 articulation 순서.
                # relax 중엔 목표 대신 RELAX 패킷(무토크 요청) — real_runner가 kp=kd=tau=0 능동 송신.
                if real_enable and real_act_addr is not None:
                    if relax:
                        real_act_sock.sendto(r2s_udp.pack_policy_relax(seq), real_act_addr)
                    else:
                        pose_art = [pose[_LM_FOR_ART[p]] for p in range(NUM_JOINTS)]
                        real_act_sock.sendto(r2s_udp.pack_policy_act(seq, pose_art), real_act_addr)
                    # GAIN(R2PK) — ACT는 target_q만 실어 나르므로(kp/kd 없음) 실기 게인을 갱신하는
                    # 유일한 경로다. ⚠relax 중엔 보내지 않는다: relax의 kp=kd=0을 GAIN으로 보내면
                    # 파이 kp_cmd에 0이 영구 잔류해(real은 relax 동안 어차피 송신단에서 0 강제라
                    # 중복), GUI가 relax 상태로 죽은 뒤 다른 peer가 ENGAGE하면 무게인 추종 불능이
                    # 된다. relax 해제 시 첫 ACT와 같은 틱에 실게인이 변경 감지로 즉시 나간다.
                    if not relax:
                        kp_art = [kp[_LM_FOR_ART[p]] for p in range(NUM_JOINTS)]
                        kd_art = [kd[_LM_FOR_ART[p]] for p in range(NUM_JOINTS)]
                        if (
                            last_gain_kp is None
                            or kp_art != last_gain_kp
                            or kd_art != last_gain_kd
                            or (now - last_gain_time) >= GAIN_RESEND_PERIOD_S
                        ):
                            real_act_sock.sendto(r2s_udp.pack_policy_gain(seq, kp_art, kd_art), real_act_addr)
                            last_gain_kp = kp_art
                            last_gain_kd = kd_art
                            last_gain_time = now

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
        real_act_sock.close()


class MainWindow(QMainWindow):
    """R2S-BipedLeg GUI main window."""

    def __init__(
        self,
        shared,
        model_path: str | None = None,
        device: str = "cpu",
        real_host: str | None = None,
        publisher_proc=None,
    ) -> None:
        super().__init__()
        # publisher 프로세스 감시 — 9882 bind 충돌(GUI 중복 실행) 등으로 조용히 죽으면
        # 슬라이더/sine이 sim에 안 먹히는데 원인이 안 보인다. 2초마다 생존 확인해 상태바에 알린다.
        self._publisher_proc = publisher_proc
        if publisher_proc is not None:
            self._pub_watch_timer = QTimer(self)
            self._pub_watch_timer.timeout.connect(self._on_pub_watch_tick)
            self._pub_watch_timer.start(2000)
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
        self._rm_thread: RealMonitorThread | None = None

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

        # RELAX 중엔 로봇/sim이 중력에 처지므로 슬라이더를 실측 자세에 계속 동기화 —
        # 사용자가 슬라이더를 잡는 순간(HOLD 전환) 그 관절만 움직이고 나머지는 현 자세 유지.
        self._relax_sync_timer = QTimer(self)
        self._relax_sync_timer.timeout.connect(self._on_relax_sync_tick)
        self._relax_sync_timer.start(500)

    def _on_pub_watch_tick(self) -> None:
        if self._publisher_proc is not None and not self._publisher_proc.is_alive():
            self._pub_watch_timer.stop()
            self._status_label.setText(
                "FATAL: publisher process died - another gui_controller running? Close it and restart."
            )
            self._status_label.setStyleSheet("color: #ff6b6b; font-weight: 600;")

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

    def _write_relax(self) -> None:
        """무토크(limp) 스펙을 쓴다 — sim 게인 0, 실기엔 RELAX 패킷. 기동 기본 상태이기도 하다."""
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_RELAX
            self._shared[_SM_CMD_VALID] = 1.0

    # -- startup latch --

    def _on_startup_tick(self) -> None:
        """기동 기본 상태 = RELAX(무토크) — 어떤 목표도 능동 구동하지 않는다 (2026-08-12 정책 변경).

        gains만 미리 써두고 relax 스펙을 쓴다. 슬라이더/Home을 조작하는 순간 hold/sequence로
        전환되며 그때부터 구동이 시작된다. 실측 자세 latch는 relax 중 슬라이더 동기화 타이머
        (:meth:`_on_relax_sync_tick`)가 담당한다.
        """
        if self._startup_done:
            self._startup_timer.stop()
            return
        self._write_gains()
        self._write_relax()
        self._status_label.setText("Startup: RELAX (zero torque) - move a slider or press Home to engage")
        self._startup_done = True
        self._startup_timer.stop()

    def _on_relax_sync_tick(self) -> None:
        """RELAX 모드 동안 슬라이더/값 라벨을 실측 자세로 따라가게 한다 (0.5 s 주기)."""
        with self._shared.get_lock():
            if self._shared[_SM_MODE] != _MODE_RELAX or self._shared[_SM_CMD_VALID] < 0.5:
                return
            if self._shared[_SM_MEASURED_VALID] < 0.5:
                return
            measured = [self._shared[_SM_MEASURED_Q + i] for i in range(NUM_JOINTS)]
        self._sync_sliders_to_pose(measured)

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
        relax_btn = QPushButton("Relax (zero torque)")
        relax_btn.setObjectName("dangerButton")
        relax_btn.clicked.connect(self._on_relax_clicked)
        pose_layout.addWidget(relax_btn)
        relax_hint = QLabel("Motors go limp - robot will droop under gravity")
        relax_hint.setObjectName("subtitleLabel")
        pose_layout.addWidget(relax_hint)
        pose_layout.addStretch(1)
        layout.addWidget(pose_group)

        # Joint Sliders (leg-major, motions.JOINT_NAMES) — 직접 자세 조작 + 선택적 실기 fan-out.
        slider_group = self._build_joint_sliders_group()
        layout.addWidget(slider_group)

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
        self._position_groups = [pose_group, slider_group, sine_group, gain_group, cgain_group]

        # policy-mode 패널 (초기 숨김)
        self._policy_group = self._build_policy_group()
        layout.addWidget(self._policy_group)
        self._policy_group.setVisible(False)

        # real 모니터 패널 — 모드와 무관하게 항상 표시 (관측 전용, 9889)
        layout.addWidget(self._build_real_monitor_group())

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

    # -- joint sliders UI --

    def _build_joint_sliders_group(self) -> QGroupBox:
        """직접 자세 조작 슬라이더(leg-major, motions.JOINT_NAMES) + 단위 토글 + 실기 fan-out 체크박스."""
        group = QGroupBox("Joint Sliders")
        v = QVBoxLayout(group)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Units:"))
        self._units_combo = QComboBox()
        self._units_combo.addItems(["rad", "deg"])
        self._units_combo.currentIndexChanged.connect(self._on_units_changed)
        top_row.addWidget(self._units_combo)
        top_row.addStretch(1)
        v.addLayout(top_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self._joint_sliders: list[QSlider] = []
        self._joint_value_labels: list[QLabel] = []
        for i, name in enumerate(motions.JOINT_NAMES):
            name_label = QLabel(name)
            name_label.setMinimumWidth(60)
            grid.addWidget(name_label, i, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(_SLIDER_STEPS)
            slider.setValue(_rad_to_slider(i, motions.DEFAULT_POSE[i]))
            slider.valueChanged.connect(lambda value, idx=i: self._on_joint_slider_changed(idx, value))
            grid.addWidget(slider, i, 1)
            value_label = QLabel(self._format_joint_value(motions.DEFAULT_POSE[i]))
            value_label.setMinimumWidth(90)
            grid.addWidget(value_label, i, 2)
            self._joint_sliders.append(slider)
            self._joint_value_labels.append(value_label)
        v.addLayout(grid)

        real_row = QHBoxLayout()
        self._real_enable_check = QCheckBox("Send to robot (UDP 9887)")
        self._real_enable_check.setEnabled(self._real_host is not None)
        if self._real_host is None:
            self._real_enable_check.setToolTip("launch with --real_host")
        self._real_enable_check.toggled.connect(self._on_real_enable_toggled)
        # --real_host 지정 시 기본 ON — 슬라이더가 sim+실기 동시 publish (사용자 요청 2026-08-11).
        # connect 이후 setChecked라 핸들러가 _SM_REAL_ENABLE까지 세팅한다. 끊고 싶으면 체크 해제.
        if self._real_host is not None:
            self._real_enable_check.setChecked(True)
        real_row.addWidget(self._real_enable_check)
        real_warn = QLabel("Streams targets to the real robot - calibrated robots only")
        real_warn.setObjectName("subtitleLabel")
        real_row.addWidget(real_warn)
        real_row.addStretch(1)
        v.addLayout(real_row)
        return group

    def _format_joint_value(self, rad: float) -> str:
        if self._units_combo.currentIndex() == 1:  # deg
            return f"{math.degrees(rad):+7.2f} deg"
        return f"{rad:+7.3f} rad"

    def _on_joint_slider_changed(self, joint_idx: int, value: int) -> None:
        rad = _slider_to_rad(joint_idx, value)
        self._joint_value_labels[joint_idx].setText(self._format_joint_value(rad))
        self._sine_deactivate()
        pose = [_slider_to_rad(i, self._joint_sliders[i].value()) for i in range(NUM_JOINTS)]
        self._write_hold(pose)

    def _on_units_changed(self, _idx: int) -> None:
        """단위 토글 — 내부는 항상 rad, 값 라벨 표시만 변환한다."""
        for i in range(NUM_JOINTS):
            rad = _slider_to_rad(i, self._joint_sliders[i].value())
            self._joint_value_labels[i].setText(self._format_joint_value(rad))

    def _sync_sliders_to_pose(self, pose: list[float]) -> None:
        """슬라이더를 pose로 동기화(신호 차단 — publisher 재발행 루프 방지). Home/startup latch 후 호출."""
        for i in range(NUM_JOINTS):
            self._joint_sliders[i].blockSignals(True)
            self._joint_sliders[i].setValue(_rad_to_slider(i, pose[i]))
            self._joint_sliders[i].blockSignals(False)
            self._joint_value_labels[i].setText(self._format_joint_value(pose[i]))

    def _on_real_enable_toggled(self, checked: bool) -> None:
        with self._shared.get_lock():
            self._shared[_SM_REAL_ENABLE] = 1.0 if checked else 0.0
        # _build_ui 중 기본 ON setChecked가 상태바 생성 전에 발화할 수 있어 가드.
        if hasattr(self, "_status_label"):
            self._status_label.setText(f"Real robot streaming: {'ON' if checked else 'off'}")

    # -- real monitor UI --

    def _build_real_monitor_group(self) -> QGroupBox:
        """실기 연결 상태 패널: Host + Start/Stop + 상태 요약 1줄. 관측 전용(9889 수신, 9890 중계는 유지)."""
        group = QGroupBox("Real Robot Monitor (read-only telemetry, port 9889)")
        v = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Host:"))
        self._rm_host_edit = QLineEdit(self._real_host or "192.168.60.5")
        self._rm_host_edit.setMaximumWidth(140)
        row.addWidget(self._rm_host_edit)
        self._rm_start_btn = QPushButton("Start")
        self._rm_start_btn.clicked.connect(self._on_rm_start_clicked)
        row.addWidget(self._rm_start_btn)
        self._rm_stop_btn = QPushButton("Stop")
        self._rm_stop_btn.setEnabled(False)
        self._rm_stop_btn.clicked.connect(self._on_rm_stop_clicked)
        row.addWidget(self._rm_stop_btn)
        row.addSpacing(12)
        self._rm_link_label = QLabel("Disconnected")
        row.addWidget(self._rm_link_label)
        row.addStretch(1)
        v.addLayout(row)
        return group

    def start_real_monitor(self) -> None:
        """Real Robot Monitor 를 프로그램적으로 시작 (``--monitor`` 플래그용)."""
        self._on_rm_start_clicked()

    def _on_rm_start_clicked(self) -> None:
        host = self._rm_host_edit.text().strip()
        if not host:
            self._rm_link_label.setText("enter host IP first")
            return
        self._rm_thread = RealMonitorThread(host)
        self._rm_thread.telem.connect(self._on_rm_telem)
        self._rm_thread.failed.connect(self._on_rm_failed)
        self._rm_thread.start()
        self._rm_start_btn.setEnabled(False)
        self._rm_stop_btn.setEnabled(True)
        self._rm_host_edit.setEnabled(False)
        self._rm_link_label.setText(f"Connecting to {host}...")

    def _on_rm_stop_clicked(self) -> None:
        if self._rm_thread is not None:
            self._rm_thread.request_stop()
            self._rm_thread.wait(1000)
            self._rm_thread = None
        self._rm_start_btn.setEnabled(True)
        self._rm_stop_btn.setEnabled(False)
        self._rm_host_edit.setEnabled(True)
        self._rm_link_label.setText("Disconnected")

    def _on_rm_telem(self, d: dict) -> None:
        if "q" not in d or d.get("stale"):
            self._rm_link_label.setText("Disconnected")
            return
        mask = d.get("valid_mask", 0x1FF)
        n_valid = sum(1 for j in range(NUM_JOINTS) if mask & (1 << j))
        imu_ok = bool(mask & (1 << 8))
        if n_valid < NUM_JOINTS:
            self._rm_link_label.setText(f"Connected (motors {n_valid}/8 - check RobotEmbedded)")
        else:
            self._rm_link_label.setText(
                f"Connected (motors {n_valid}/8, {d['hz']:.0f} Hz, IMU {'ok' if imu_ok else 'stale'})"
            )

    def _on_rm_failed(self, msg: str) -> None:
        self._rm_link_label.setText(f"error: {msg}")
        self._rm_start_btn.setEnabled(True)
        self._rm_stop_btn.setEnabled(False)
        self._rm_host_edit.setEnabled(True)

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
                hold_pose = measured if measured_valid else current
                self._write_hold(hold_pose)
                self._sync_sliders_to_pose(hold_pose)
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
        self._sync_sliders_to_pose(list(motions.DEFAULT_POSE))

    def _on_relax_clicked(self) -> None:
        """무토크(limp): sim은 kp=kd=0, 실기는 RELAX 패킷 — 기동 기본 상태와 동일."""
        self._sine_deactivate()
        self._write_relax()
        self._status_label.setText("RELAX (zero torque) - move a slider or press Home to engage")

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
        pose = self._output_pose()
        self._write_hold(pose)
        self._sync_sliders_to_pose(pose)
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
        if self._rm_thread is not None:
            self._rm_thread.request_stop()
            self._rm_thread.wait(1000)
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
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="기동 즉시 Real Robot Monitor 시작 (호스트는 --real_host 또는 패널 기본값).",
    )
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
        shared[_SM_REAL_ENABLE] = 0.0  # 실기 전달 기본 OFF
        for i in range(NUM_JOINTS):
            shared[_SM_KP + i] = motions.DEFAULT_KP[i]
            shared[_SM_KD + i] = motions.DEFAULT_KD[i]
            shared[_SM_BASE_Q + i] = motions.DEFAULT_POSE[i]
    publisher = mp.Process(target=publisher_process_main, args=(shared, stop_flag, args.real_host), daemon=True)
    publisher.start()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow(
        shared, model_path=model_path, device=args.device, real_host=args.real_host, publisher_proc=publisher
    )
    if args.monitor:
        window.start_real_monitor()
    window.show()
    exit_code = app.exec_()

    stop_flag.value = 1
    publisher.join(timeout=2.0)
    if publisher.is_alive():
        publisher.terminate()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
