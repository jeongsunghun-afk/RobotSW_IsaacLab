#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 gui_controller — PyQt5 GUI that publishes `/lowcmd` (LowCmd).

Run with system Python 3.10 + ROS2 humble + unitree_go overlay (CONTRACT.md §1):

    source /opt/ros/humble/setup.bash
    source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
    /usr/bin/python3 scripts/real2sim/r2s_go2/gui_controller.py

Buttons:
    - Default (stand): interpolate from the current pose to DEFAULT_POSE.
    - Joint step: step only the selected joint by current + delta (interpolated).
    - Sine sweep: inject a sine wave on the selected joint around DEFAULT_POSE
      (continuous, stopped with the Stop button).
    - Sit: interpolate from the current pose to SIT_POSE.
    - Policy: drive the exported go2_imitation_tracking deployable policy (RMA+estimator,
      proprio(45)+history(10x45) -> action(12)). Start interpolates to DEFAULT_POSE first
      (safety handover, real-hardware fall risk), then engages closed-loop control. Inference
      runs inside the publisher process, not the Qt UI process — see ``policy_runtime.py`` and
      ``_compute_target``'s ``_MODE_POLICY`` branch. Input source (Sim/Real ``/lowstate``) and
      x_vel/yaw_vel command are live-adjustable without breaking the control loop.
    - Recovery: drive the exported go2_recovery_flip_vel fall-recovery policy (plain PPO,
      obs(42) -> action(12)) to stand the robot back up after it has fallen. Unlike Policy,
      Start engages **immediately from the current pose** — the robot is on the ground and
      interpolating to DEFAULT_POSE first would be both unsafe and out of distribution (the
      training env's settle phase holds the measured pose, and its length is uniform on
      [0, 100] steps, so acting from tick 0 in a fallen pose is in-distribution). This mode
      also publishes kd=1.0, not the usual 0.5 — see ``recovery_runtime.RECOVERY_KD``.

Every command is published with kp=25, kd=0.5 (all joints), mode=0x01, dq=0, tau=0 (CONTRACT.md §6).
The one exception is Recovery mode, which publishes kd=1.0 to match the gains its policy was
trained with (``go2_recovery_env_cfg.py`` actuator ``damping=1.0``).

Process architecture — /lowcmd publishing AND target generation (interpolation/sine) run in a
**separate process** (``publisher_process_main``), NOT in the Qt UI process. The real GO2 firmware
faults on a gap in the command stream, and Qt QTimers slip whenever the UI event loop is busy
(dragging a slider, holding a button). The UI writes only a *motion spec* (mode + start time +
goal/frame-buffer/sine params) to a shared-memory ``mp.Array``; the publisher process computes the
target from elapsed wall-clock time at a steady 50 Hz, completely decoupled from the UI event loop.
So user interaction can starve neither the heartbeat nor the target updates (an early version moved
only publishing out — hz stayed 50 but motion still stuttered because the UI motion timers that
*generate* the target were starved). Threading would not work — the GIL lets a busy UI callback
starve a publisher thread (same reason the monitor is a separate process, CONTRACT §11). Each
LowCmd carries the deploy header (head=0xFE 0xEF, level_flag=0xFF) and a valid CRC (see lowcmd_crc);
both are inert to the sim path (sim_bridge strips them) but required for real deployment (§8).

Contract: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md

Visual styling is a self-contained QSS stylesheet applied in :func:`main` (see
``_STYLESHEET`` below); it is purely cosmetic and does not affect any publish/timer logic.
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import lowcmd_crc  # noqa: E402
import motions  # noqa: E402
import pedipulation_runtime  # noqa: E402
import policy_runtime  # noqa: E402
import r2s_udp  # noqa: E402
import rclpy  # noqa: E402
import recovery_runtime  # noqa: E402
from PyQt5.QtCore import Qt, QTimer  # noqa: E402
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
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from unitree_go.msg import LowCmd, LowState  # noqa: E402

FRAME_HZ: float = 50.0  # motion generation rate (sequence/sine frames)
FRAME_PERIOD_S: float = 1.0 / FRAME_HZ
# Continuous /lowcmd rate. 50Hz matches the standard RL cadence (policy 50Hz, sim
# decimation=4 → 200Hz physics / 50Hz control) and the real GO2 RL-deploy convention:
# the policy streams /lowcmd at 50Hz while the robot firmware runs its PD loop at a much
# higher internal rate. A steady 50Hz stream also satisfies the low-level watchdog.
PUBLISH_HZ: float = 50.0
PUBLISH_PERIOD_S: float = 1.0 / PUBLISH_HZ
SEQUENCE_DURATION_S: float = 1.5  # interpolation time for default/sit
STEP_DURATION_S: float = 0.5  # interpolation time for joint step
SPIN_PERIOD_S: float = 0.01  # rclpy spin_once polling period

# 시작 자세 접근(startup approach) — gui 기동 시 로봇의 실측 현재 자세를 읽어 거기서부터 시작
# 자세로 부드럽게 보간한다. 예전엔 STAND_FOLDED 를 t=0 부터 그대로 발행해 로봇이 현재 자세에서
# 툭 스냅했다(kp=25 로 강하게). 이제 첫 /lowstate 를 받을 때까지 발행을 보류하고, 받으면 보간한다.
STARTUP_DURATION_S: float = 2.5  # 현재 자세 → 시작 자세 보간 시간 [s] (느리게 = 안전)
STARTUP_ACQUIRE_TIMEOUT_S: float = 2.0  # 첫 /lowstate 대기 상한 [s]. 넘으면 실측 없이 진행
# 시작 자세 획득용 상태 토픽. shared(실기) 모드면 실로봇 /lowstate, sim 전용이면 sim_bridge /lowstate.
STARTUP_STATE_TOPIC: str = os.environ.get("R2S_STARTUP_STATE_TOPIC", "/lowstate")
CMD_TOPIC: str = os.environ.get("R2S_CMD_TOPIC", "/lowcmd")  # publisher 프로세스가 발행하는 명령 토픽

# Sim 제어 채널 (카메라 / 관절 물성) — sim_runner_go2.py 로 **직접** UDP. sim_bridge 를 거치지
# 않는다(ROS 명령 경로는 50Hz 실시간 스트림이고 이건 버튼 one-shot). 기본은 같은 머신.
SIM_CTRL_HOST: str = os.environ.get("R2S_SIM_HOST", "127.0.0.1")
SIM_CTRL_PORT: int = int(os.environ.get("R2S_CTRL_PORT", r2s_udp.CTRL_PORT))
# Plant 콤보 항목 순서 → ctrl 패킷 `plant` 값. 콤보 `addItems` 순서와 반드시 일치시킬 것.
# 학습 플랜트(set2)를 첫 항목에 두어 GUI 기본 선택이 정책과 맞게 한다.
PLANT_COMBO_ORDER: tuple[int, ...] = (r2s_udp.PLANT_SET2, r2s_udp.PLANT_SET3, r2s_udp.PLANT_NOMINAL)

# Policy 모드 상태 소스 — 실로봇은 항상 고정 "/lowstate"(remap 안 함, CONTRACT §3). sim은
# monitor.py와 동일한 R2S_SIM_STATE_TOPIC 관례를 재사용(run_gui_controller.sh가 shared/sim-only
# 모드에 맞춰 이미 export함 — sim-only일 땐 "/lowstate"로 겹쳐도 rclpy가 다중구독을 허용하므로
# 무해하다). STARTUP_STATE_TOPIC(위)은 기동 시 자세 획득 전용이라 건드리지 않는다 — 두 목적을
# 한 구독으로 겸용하면 STARTUP_STATE_TOPIC이 sim으로 오버라이드된 경우 policy의 "real" 소스
# 의미가 깨지므로, 약간의 구독 중복을 감수하고 policy 전용 구독을 별도로 둔다.
POLICY_REAL_STATE_TOPIC: str = "/lowstate"
POLICY_SIM_STATE_TOPIC: str = os.environ.get("R2S_SIM_STATE_TOPIC", "/sim/lowstate")
DEPLOYABLE_POLICY_PATH: str = os.environ.get(
    "R2S_DEPLOYABLE_POLICY",
    os.path.join(
        # scripts/real2sim/r2s_go2/gui_controller.py -> repo root (4 levels up from this file)
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        # obs 42 / priv_explicit 6 런. 구 런(2026-07-24_13-03-00)은 obs 45 / priv_explicit 3 이라
        # `policy_runtime.POLICY_DIM`(=42)과 맞지 않아 로드 시 shape mismatch 로 죽는다.
        # 다른 체크포인트를 쓰려면 `R2S_DEPLOYABLE_POLICY` 로 덮어쓰되, 그 런의 env.yaml
        # `observation_space` 가 42 인지 먼저 확인할 것.
        "logs/rsl_rl/go2_imitation_tracking/2026-07-30_10-35-28_obs42_novideo/exported/deployable_policy.pt",
    ),
)
# Recovery 모드 정책 — go2_recovery_flip_vel 런의 export 산출물(`export_recovery_go2.py`).
# 기본값은 success_rate 0.679(넘어진 env 기준 0.958)로 가장 좋은 런이다. 다른 체크포인트를 쓰려면
# `R2S_RECOVERY_POLICY`로 덮어쓴다. ⚠ tracking 용 deployable_policy.pt 와 obs 가 똑같이 42-dim
# 이지만 레이아웃이 달라 shape 로는 구분되지 않는다 — `recovery_runtime.RecoveryModel._warmup`이
# 입력 인자 개수로 잘못된 파일을 걸러낸다.
RECOVERY_POLICY_PATH: str = os.environ.get(
    "R2S_RECOVERY_POLICY",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "logs/rsl_rl/go2_recovery_flip_vel/2026-07-14_09-54-05_floor03_ent005/exported/recovery_policy.pt",
    ),
)

# Pedipulation 모드 정책 — go2_pedipulation 채택본의 export 산출물
# (`export_pedipulation_go2.py`). 다른 체크포인트를 쓰려면 `R2S_PEDIPULATION_POLICY` 로 덮어쓴다.
# 앞의 둘과 달리 obs 83 / action 28 이라 shape 만으로 구분된다
# (`pedipulation_runtime.PedipulationModel._warmup`).
PEDIPULATION_POLICY_PATH: str = os.environ.get(
    "R2S_PEDIPULATION_POLICY",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "logs/rsl_rl/go2_pedipulation/2026-08-03_15-53-29_hipscale_scratch_s10x/exported/pedipulation_policy.pt",
    ),
)

# Self-contained dark-theme QSS. Purely cosmetic (colors/spacing/typography) — applied once
# in main() via app.setStyleSheet(); no widget behavior, signal wiring, or timer logic reads
# from it. Font family list relies on Qt's built-in fallback (no bundled font file needed).
_STYLESHEET: str = """
QWidget {
    background-color: #1c1e26;
    color: #e6e8ef;
    font-family: "Segoe UI", "Ubuntu", "Roboto", "DejaVu Sans", sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #1c1e26;
}

#titleLabel {
    font-size: 19px;
    font-weight: 600;
    color: #f4f6fb;
}

#subtitleLabel {
    font-size: 11px;
    color: #8a8fa3;
}

#divider {
    background-color: #33364688;
    max-height: 1px;
    border: none;
}

QGroupBox {
    background-color: #242733;
    border: 1px solid #33364a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    color: #c7cbe0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #9fb4ff;
}

QLabel {
    color: #c7cbe0;
    background: transparent;
}

QPushButton {
    background-color: #2c2f3d;
    color: #e6e8ef;
    border: 1px solid #3d4157;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #363a4c;
    border-color: #4a4f6b;
}

QPushButton:pressed {
    background-color: #23252f;
}

QPushButton:disabled {
    background-color: #23252f;
    color: #5c6079;
    border-color: #2c2f3d;
}

QPushButton#primaryButton {
    background-color: #4f7dfb;
    border-color: #4f7dfb;
    color: #ffffff;
}

QPushButton#primaryButton:hover {
    background-color: #6b93fc;
    border-color: #6b93fc;
}

QPushButton#primaryButton:pressed {
    background-color: #3d67e0;
}

QPushButton#dangerButton {
    background-color: #d9534f;
    border-color: #d9534f;
    color: #ffffff;
}

QPushButton#dangerButton:hover {
    background-color: #e46864;
}

QPushButton#dangerButton:pressed:enabled {
    background-color: #c14743;
}

QPushButton#dangerButton:disabled {
    background-color: #23252f;
    color: #5c6079;
    border-color: #2c2f3d;
}

QComboBox, QDoubleSpinBox {
    background-color: #1c1e26;
    border: 1px solid #3d4157;
    border-radius: 5px;
    padding: 5px 8px;
    min-height: 20px;
}

QComboBox:hover, QDoubleSpinBox:hover {
    border-color: #4a4f6b;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #242733;
    border: 1px solid #3d4157;
    selection-background-color: #4f7dfb;
    outline: none;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 16px;
    border: none;
}

#statusBar {
    background-color: #242733;
    border: 1px solid #33364a;
    border-radius: 8px;
}

#statusLabel {
    color: #9fb4ff;
    font-weight: 600;
    padding: 8px 14px;
}
"""


_NUM = r2s_udp.NUM_MOTORS

# ---------------------------------------------------------------------------
# 공유 메모리 레이아웃 (multiprocessing.Array('d')). UI는 **모션 스펙**(어디로 몇 초에 걸쳐
# 가라)만 쓰고, publisher 프로세스가 50Hz 루프에서 **경과 시간 기반으로 목표를 계산**해 발행한다.
# 이렇게 하면 UI event loop가 조작으로 밀려도 목표가 매끄럽게(50Hz 균일) 갱신된다 — 발행뿐 아니라
# 목표 생성까지 UI에서 격리 (CONTRACT §11 동일 논리). time.monotonic()은 리눅스에서 프로세스 간
# 공통(CLOCK_MONOTONIC)이라 UI가 찍은 start_time을 publisher가 그대로 쓸 수 있다.
# ---------------------------------------------------------------------------
_MODE_HOLD = 0  # 고정 자세 유지 (BASE_Q)
_MODE_SEQUENCE = 1  # 프레임 버퍼를 경과 시간으로 인덱싱 (보간/Stand/Sit/Step/StandUp)
_MODE_SINE = 2  # BASE_Q + 선택 관절에 사인 주입 (경과 시간 기반)
_MODE_POLICY = 3  # 실측 proprio + policy 추론으로 매 tick 목표 계산 (base/start_time 미사용)
_MODE_RECOVERY = 4  # 실측 obs + fall-recovery 정책 추론으로 매 tick 목표 계산 (명령 없음)
_MODE_PEDIPULATION = 5  # 실측 obs + pedipulation 정책 추론 (명령 = 조작 다리 + 발 목표 offset)

_SM_MODE = 0
_SM_KP = 1
_SM_KD = 2
_SM_CMD_VALID = 3  # 0/1: 유효한 명령이 있는가. 시작 자세 획득 전엔 0 → 발행 보류(스냅 방지)
_SM_START_TIME = 4  # 모션 시작 시각 (time.monotonic)
_SM_FRAME_HZ = 5  # 시퀀스 프레임 생성 주파수 (인덱싱용)
_SM_FRAME_COUNT = 6  # 시퀀스 프레임 수
_SM_SINE_JOINT = 7
_SM_SINE_AMP = 8
_SM_SINE_FREQ = 9
_SM_MEASURED_VALID = 10  # 0/1: publisher가 /lowstate를 받았는가
_SM_BASE_Q = 11  # 11..22: hold 자세 / sine 기준 자세 / policy 모드 fallback(default) 자세
_SM_MEASURED_Q = 23  # 23..34: 로봇 실측 관절각 (publisher가 /lowstate에서 씀)
_SM_CURRENT_Q = 35  # 35..46: publisher의 현재 출력 목표 (publisher가 매 cycle 씀, UI가 읽어 보간 시작점으로)
# Policy 모드 — cmd(UI 슬라이더가 씀) + 실측 proprio 소스(publisher가 dual subscription으로 씀,
# real/sim 각각 별도 슬롯. 값은 전부 DDS 순서 그대로 저장, articulation 순서 변환은 추론 직전에
# policy_runtime에서 함). history 링버퍼(10×45)는 여기 없음 — publisher 프로세스 로컬 상태(§3).
_SM_POLICY_XVEL = 47  # lin_vel_cmd x [m/s]
_SM_POLICY_YAWVEL = 48  # yaw_vel_cmd [rad/s]
_SM_POLICY_INPUT_SOURCE = 49  # 0=sim, 1=real
_SM_POLICY_Q_REAL = 50  # 50..61 (DDS 순서)
_SM_POLICY_DQ_REAL = 62  # 62..73
_SM_POLICY_GYRO_REAL = 74  # 74..76
_SM_POLICY_QUAT_REAL = 77  # 77..80 (wxyz)
_SM_POLICY_VALID_REAL = 81
_SM_POLICY_Q_SIM = 82  # 82..93 (DDS 순서)
_SM_POLICY_DQ_SIM = 94  # 94..105
_SM_POLICY_GYRO_SIM = 106  # 106..108
_SM_POLICY_QUAT_SIM = 109  # 109..112 (wxyz)
_SM_POLICY_VALID_SIM = 113
# Recovery 모드 — 명령이 없어(목표=기립 고정) 슬라이더 슬롯도 없다. 실측 상태는 위 _SM_POLICY_*
# 슬롯(가공 없는 /lowstate 원본이라 모드와 무관)을 그대로 재사용하고, 입력 소스 선택만 따로 둔다:
# 한 슬롯을 공유하면 한쪽 콤보를 바꿀 때 다른 모드의 소스까지 끌려가기 때문이다.
_SM_RECOVERY_INPUT_SOURCE = 114  # 0=sim, 1=real (_POLICY_SOURCE_* 값 재사용)
# Pedipulation 모드 — 실측 상태는 위 _SM_POLICY_* 슬롯을 재사용하고(가공 없는 /lowstate 원본),
# 명령만 따로 둔다. 명령은 "어느 다리를 조작할지" + "nominal 발 위치 대비 offset[m]" 이다.
# 적분형 목표·nominal latch 는 publisher 프로세스 로컬 상태다(§3, `_PedipulationContext`).
_SM_PEDI_INPUT_SOURCE = 115  # 0=sim, 1=real
_SM_PEDI_MANIP_LEG = 116  # 0=FL, 1=FR, 2=RL, 3=RR (pedipulation_runtime.LEG_NAMES)
_SM_PEDI_OFF_X = 117  # nominal 대비 목표 offset [m]
_SM_PEDI_OFF_Y = 118
_SM_PEDI_OFF_Z = 119
_SM_FRAMES = 120  # 120..: 시퀀스 프레임 버퍼 (MAX_FRAMES × 12)
MAX_FRAMES = 512  # 512/50 = 10.24s 최대 시퀀스
_SM_LEN = _SM_FRAMES + MAX_FRAMES * _NUM

_POLICY_SOURCE_SIM = 0
_POLICY_SOURCE_REAL = 1


def _build_lowcmd(pose: list[float], kp: float, kd: float) -> LowCmd:
    """12 관절 목표를 LowCmd로 직렬화 (mode=0x01, dq=0, tau=0, deploy 헤더 + CRC).

    실기 GO2 펌웨어는 CRC가 틀린 명령을 폐기한다. sim 경로(sim_bridge → UDP)는 head/crc를
    무시하므로 sim엔 무해하지만 실기 배포엔 필수다 (CONTRACT.md §8).
    """
    msg = LowCmd()
    msg.head[0] = lowcmd_crc.HEAD[0]
    msg.head[1] = lowcmd_crc.HEAD[1]
    msg.level_flag = lowcmd_crc.LEVEL_FLAG_LOWLEVEL
    msg.gpio = 0
    for i in range(_NUM):
        motor = msg.motor_cmd[i]
        motor.mode = lowcmd_crc.MOTOR_MODE_SERVO
        motor.q = pose[i]
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = kp
        motor.kd = kd
    lowcmd_crc.set_crc(msg)
    return msg


class _PolicyContext:
    """publisher 프로세스 로컬 policy 상태(모델 + history). **공유메모리에 절대 안 올림**(§3
    결정 — history/모델 객체는 publisher만 읽고 쓰므로 굳이 mp.Array로 왕복시킬 이유가 없음)."""

    def __init__(self) -> None:
        self.model: policy_runtime.PolicyModel | None = None
        self.history = policy_runtime.ProprioHistory()
        self._load_attempted = False

    def ensure_loaded(self, node) -> None:
        """모델을 1회만 로드 시도(실패해도 재시도 안 함 — policy 모드는 그냥 fallback-hold로 동작)."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            self.model = policy_runtime.PolicyModel(DEPLOYABLE_POLICY_PATH)
            node.get_logger().info(f"[policy] deployable 모델 로드 완료: {DEPLOYABLE_POLICY_PATH}")
        except Exception as exc:  # noqa: BLE001 — 로드 실패는 fallback-hold로 안전하게 흡수
            node.get_logger().warning(f"[policy] 모델 로드 실패({exc}) — Policy 모드는 fallback-hold로 동작")


class _RecoveryContext:
    """publisher 프로세스 로컬 recovery 상태(모델 + previous_actions 버퍼). `_PolicyContext`와 동일 원칙."""

    def __init__(self) -> None:
        self.model: recovery_runtime.RecoveryModel | None = None
        self.prev_action = recovery_runtime.PrevActionBuffer()
        self._load_attempted = False

    def ensure_loaded(self, node) -> None:
        """모델을 1회만 로드 시도(실패해도 재시도 안 함 — Recovery 모드는 fallback-hold로 동작)."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            self.model = recovery_runtime.RecoveryModel(RECOVERY_POLICY_PATH)
            node.get_logger().info(f"[recovery] 모델 로드 완료: {RECOVERY_POLICY_PATH}")
        except Exception as exc:  # noqa: BLE001 — 로드 실패는 fallback-hold로 안전하게 흡수
            node.get_logger().warning(f"[recovery] 모델 로드 실패({exc}) — Recovery 모드는 fallback-hold로 동작")


class _PedipulationContext:
    """publisher 프로세스 로컬 pedipulation 상태. `_PolicyContext`·`_RecoveryContext`와 같은 원칙이되
    들고 가는 상태가 더 많다 — 조작 다리 목표가 **적분형**이고 nominal 발 위치를 latch 하기 때문이다.
    """

    def __init__(self) -> None:
        self.model: pedipulation_runtime.PedipulationModel | None = None
        self.state = pedipulation_runtime.PedipulationState()
        self._load_attempted = False
        # sim 뷰포트 마커용 최신 상태 (조작 다리, base frame 목표). publisher 루프가 읽어
        # sim_runner 로 보낸다. 모드가 아니거나 nominal 미확보면 leg 가 OFF 라 마커가 숨는다.
        self.marker_leg: int = r2s_udp.MARKER_LEG_OFF
        self.marker_target_b: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def ensure_loaded(self, node) -> None:
        """모델을 1회만 로드 시도(실패해도 재시도 안 함 — Pedipulation 모드는 fallback-hold로 동작)."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            self.model = pedipulation_runtime.PedipulationModel(PEDIPULATION_POLICY_PATH)
            node.get_logger().info(f"[pedipulation] 모델 로드 완료: {PEDIPULATION_POLICY_PATH}")
        except Exception as exc:  # noqa: BLE001 — 로드 실패는 fallback-hold로 안전하게 흡수
            node.get_logger().warning(f"[pedipulation] 모델 로드 실패({exc}) — Pedipulation 모드는 fallback-hold")


def _compute_target(
    shared,
    now: float,
    policy_ctx: _PolicyContext,
    recovery_ctx: _RecoveryContext,
    pedi_ctx: _PedipulationContext,
) -> tuple[list[float], float, float] | None:
    """공유 메모리의 모션 스펙 + 경과 시간으로 현재 목표 자세를 계산한다 (publisher 프로세스에서 호출).

    Returns:
        ``(pose, kp, kd)`` 또는 명령이 없으면(``CMD_VALID=0``) None.
    """
    policy_inputs = None  # _MODE_POLICY일 때만 채움 — lock 밖에서 추론하기 위해 값만 빼온다
    recovery_inputs = None  # _MODE_RECOVERY일 때만 채움 (동일 이유)
    pedi_inputs = None  # _MODE_PEDIPULATION일 때만 채움 (동일 이유)
    with shared.get_lock():
        if shared[_SM_CMD_VALID] < 0.5:
            return None
        mode = int(shared[_SM_MODE])
        kp = shared[_SM_KP]
        kd = shared[_SM_KD]
        base = [shared[_SM_BASE_Q + i] for i in range(_NUM)]
        if mode == _MODE_SEQUENCE:
            start = shared[_SM_START_TIME]
            frame_hz = shared[_SM_FRAME_HZ]
            count = int(shared[_SM_FRAME_COUNT])
            idx = int((now - start) * frame_hz)
            idx = 0 if idx < 0 else (count - 1 if idx >= count else idx)  # 끝 프레임에서 홀드
            off = _SM_FRAMES + idx * _NUM
            pose = [shared[off + i] for i in range(_NUM)]
        elif mode == _MODE_SINE:
            start = shared[_SM_START_TIME]
            joint = int(shared[_SM_SINE_JOINT])
            amp = shared[_SM_SINE_AMP]
            freq = shared[_SM_SINE_FREQ]
            pose = list(base)
            pose[joint] = base[joint] + amp * math.sin(2.0 * math.pi * freq * (now - start))
        elif mode == _MODE_POLICY:
            pose = base  # 기본값(실측 미수신/모델 미로드 시 fallback) — 아래서 조건부로 덮어씀
            source = int(shared[_SM_POLICY_INPUT_SOURCE])
            # 자이로 오프셋은 읽지 않는다 — proprio가 42-dim이 되면서 각속도가 정책 입력에서
            # 빠졌다(priv_explicit로 이동, estimator가 추정). 공유메모리 레이아웃 자체는
            # 유지되므로 _SM_POLICY_GYRO_* 상수는 그대로 둔다.
            if source == _POLICY_SOURCE_REAL:
                valid = shared[_SM_POLICY_VALID_REAL] >= 0.5
                q_off, dq_off, quat_off = (_SM_POLICY_Q_REAL, _SM_POLICY_DQ_REAL, _SM_POLICY_QUAT_REAL)
            else:
                valid = shared[_SM_POLICY_VALID_SIM] >= 0.5
                q_off, dq_off, quat_off = (_SM_POLICY_Q_SIM, _SM_POLICY_DQ_SIM, _SM_POLICY_QUAT_SIM)
            if valid and policy_ctx.model is not None:
                policy_inputs = (
                    [shared[q_off + i] for i in range(_NUM)],  # q_dds
                    [shared[dq_off + i] for i in range(_NUM)],  # dq_dds
                    tuple(shared[quat_off + i] for i in range(4)),  # quat_wxyz
                    shared[_SM_POLICY_XVEL],
                    shared[_SM_POLICY_YAWVEL],
                )
        elif mode == _MODE_RECOVERY:
            pose = base  # 기본값(실측 미수신/모델 미로드 시 fallback) — 아래서 조건부로 덮어씀
            source = int(shared[_SM_RECOVERY_INPUT_SOURCE])
            # policy 모드와 달리 자이로를 읽는다 — recovery obs 는 각속도를 입력으로 쓴다
            # (estimator 가 없어 자세 변화율을 직접 봐야 한다).
            if source == _POLICY_SOURCE_REAL:
                valid = shared[_SM_POLICY_VALID_REAL] >= 0.5
                q_off, dq_off, gyro_off, quat_off = (
                    _SM_POLICY_Q_REAL,
                    _SM_POLICY_DQ_REAL,
                    _SM_POLICY_GYRO_REAL,
                    _SM_POLICY_QUAT_REAL,
                )
            else:
                valid = shared[_SM_POLICY_VALID_SIM] >= 0.5
                q_off, dq_off, gyro_off, quat_off = (
                    _SM_POLICY_Q_SIM,
                    _SM_POLICY_DQ_SIM,
                    _SM_POLICY_GYRO_SIM,
                    _SM_POLICY_QUAT_SIM,
                )
            if valid and recovery_ctx.model is not None:
                recovery_inputs = (
                    [shared[q_off + i] for i in range(_NUM)],  # q_dds
                    [shared[dq_off + i] for i in range(_NUM)],  # dq_dds
                    tuple(shared[gyro_off + i] for i in range(3)),  # ang_vel_b
                    tuple(shared[quat_off + i] for i in range(4)),  # quat_wxyz
                )
        elif mode == _MODE_PEDIPULATION:
            pose = base  # 기본값(실측 미수신/모델 미로드/nominal 미확보 시 fallback)
            source = int(shared[_SM_PEDI_INPUT_SOURCE])
            # recovery 와 달리 자이로는 안 읽는다 — pedipulation obs 는 각속도를 쓰지 않는다
            # (base 각속도는 critic 전용 priv 로 빠져 있다, cfg D8).
            if source == _POLICY_SOURCE_REAL:
                valid = shared[_SM_POLICY_VALID_REAL] >= 0.5
                q_off, dq_off, quat_off = _SM_POLICY_Q_REAL, _SM_POLICY_DQ_REAL, _SM_POLICY_QUAT_REAL
            else:
                valid = shared[_SM_POLICY_VALID_SIM] >= 0.5
                q_off, dq_off, quat_off = _SM_POLICY_Q_SIM, _SM_POLICY_DQ_SIM, _SM_POLICY_QUAT_SIM
            if valid and pedi_ctx.model is not None:
                pedi_inputs = (
                    [shared[q_off + i] for i in range(_NUM)],  # q_dds
                    [shared[dq_off + i] for i in range(_NUM)],  # dq_dds
                    tuple(shared[quat_off + i] for i in range(4)),  # quat_wxyz
                    int(shared[_SM_PEDI_MANIP_LEG]),
                    (shared[_SM_PEDI_OFF_X], shared[_SM_PEDI_OFF_Y], shared[_SM_PEDI_OFF_Z]),
                )
        else:  # _MODE_HOLD
            pose = base

    # lock 밖: policy 추론(torch forward, GPU일 수 있음)은 UI의 공유메모리 접근(슬라이더/버튼)을
    # 블록하지 않도록 lock을 놓은 뒤 수행한다(§3 결정). policy_inputs가 없는 tick(다른 모드이거나,
    # 실측/모델 미준비인 fallback)은 history를 무효화해 다음 policy 진입 때 새로 seed되게 한다.
    if policy_inputs is None:
        policy_ctx.history.clear()
    else:
        q_dds, dq_dds, quat_wxyz, xvel, yaw_vel = policy_inputs
        q_art = policy_runtime.dds_to_art(q_dds)
        dq_art = policy_runtime.dds_to_art(dq_dds)
        proprio = policy_runtime.build_proprio(quat_wxyz, (xvel, 0.0), yaw_vel, q_art, dq_art)
        history = policy_ctx.history.push(proprio)
        action = policy_ctx.model.infer(proprio, history)
        target_art = policy_runtime.action_to_target_art(action)
        pose = policy_runtime.art_to_dds(target_art)

    # recovery 도 같은 원칙: 추론은 lock 밖에서. previous_actions 버퍼는 policy 의 history 와
    # 같은 역할이라 비활성 tick 에 무효화해 다음 진입 때 학습 reset(0)과 같은 상태로 시작한다.
    if recovery_inputs is None:
        recovery_ctx.prev_action.clear()
    else:
        q_dds, dq_dds, ang_vel_b, quat_wxyz = recovery_inputs
        obs = recovery_runtime.build_obs(
            ang_vel_b,
            quat_wxyz,
            recovery_runtime.dds_to_art(q_dds),
            recovery_runtime.dds_to_art(dq_dds),
            recovery_ctx.prev_action.get(),
        )
        clipped = recovery_runtime.clip_action(recovery_ctx.model.infer(obs))
        recovery_ctx.prev_action.commit(clipped)  # 다음 tick obs 의 previous_actions
        pose = recovery_runtime.art_to_dds(recovery_runtime.action_to_target_art(clipped))

    # pedipulation 도 같은 원칙: 추론은 lock 밖에서.
    #
    # ⚠ 리셋 조건이 policy/recovery 와 다르다. 저 둘은 `*_inputs is None` 이면 무조건 비우지만,
    #   여기서는 **모드를 벗어났을 때만** 비운다. `pedi_inputs` 는 모드가 맞는데도 None 이 될 수
    #   있기 때문이다(/lowstate 한 프레임 유실, 모델 미로드). 그때 `reset()` 을 부르면
    #   `nominal_foot_pos_b` 까지 날아가고, 다음 정상 프레임에서 **다리를 든 3족 자세**를 기준으로
    #   다시 latch 해버린다 — 깊이 게이트(0.20~0.40 m)는 지지 발 3개로 통과하므로 못 막는다.
    #   결과는 명령 기준계가 런 도중 조용히 어긋나는 것이다. nominal 은 학습 상태가 아니라
    #   **latch 된 캘리브레이션**이라 recovery 의 `prev_action.clear()` 와 성격이 다르다.
    if mode != _MODE_PEDIPULATION:
        pedi_ctx.state.reset()
        pedi_ctx.marker_leg = r2s_udp.MARKER_LEG_OFF  # 모드를 나가면 sim 마커도 숨긴다
    elif pedi_inputs is not None:
        q_dds, dq_dds, quat_wxyz, manip_leg, offset = pedi_inputs
        q_art = pedipulation_runtime.dds_to_art(q_dds)
        dq_art = pedipulation_runtime.dds_to_art(dq_dds)
        # nominal 은 4족으로 선 자세에서만 잡힌다(발 깊이 0.20~0.40 m). 못 잡으면 명령의 기준이
        # 없으므로 정책을 구동하지 않고 fallback-hold 를 유지한다 — 넘어진 채 시작하는 사고 방지.
        had_nominal = pedi_ctx.state.nominal_foot_pos_b is not None
        nominal = pedi_ctx.state.nominal_foot_pos_b if pedi_ctx.state.try_latch_nominal(q_art) else None
        if nominal is not None and not had_nominal:
            # latch 는 캘리브레이션이라 값 자체가 진단 정보다 — 명령이 어긋나면 여기부터 본다.
            _n = " ".join(
                f"{nm}=({f[0]:+.3f},{f[1]:+.3f},{f[2]:+.3f})" for nm, f in zip(pedipulation_runtime.LEG_NAMES, nominal)
            )
            print(f"[pedipulation] nominal latch: {_n}", flush=True)
        if nominal is not None:
            leg_role = pedipulation_runtime.leg_role_from_manip(manip_leg)
            target_b = pedipulation_runtime.foot_target_b(nominal, manip_leg, offset)
            obs = pedipulation_runtime.build_obs(quat_wxyz, q_art, dq_art, leg_role, target_b, pedi_ctx.state)
            clipped = pedipulation_runtime.clip_action(pedi_ctx.model.infer(obs))
            target_art = pedipulation_runtime.action_to_target_art(clipped, leg_role, pedi_ctx.state)
            pedi_ctx.state.prev_actions = clipped  # 다음 tick obs 의 prev_actions (clip 후·scale 전)
            pose = pedipulation_runtime.art_to_dds(target_art)
            # sim 뷰포트 마커 — 여기서만 갱신한다(정책이 실제로 돈 tick). nominal 을 못 잡아
            # fallback-hold 중이면 이 줄에 도달하지 않으므로 마커도 뜨지 않는다.
            pedi_ctx.marker_leg = manip_leg
            _tb = target_b[manip_leg]
            pedi_ctx.marker_target_b = (_tb[0], _tb[1], _tb[2])

    return pose, kp, kd


def publisher_process_main(shared, stop_flag) -> None:
    """**별도 프로세스**: 모션 스펙에서 목표를 계산해 /lowcmd 50Hz 발행 + /lowstate 구독. Qt와 완전 독립.

    UI event loop(위젯 조작·드래그)가 아무리 바빠도 이 프로세스는 영향받지 않는다. UI는 모션 스펙만
    쓰고, 목표 보간/사인 생성은 여기서 50Hz 균일하게 하므로 **목표값도 매끄럽게** 갱신된다(발행뿐 아니라
    목표 생성까지 UI에서 격리). 스레드로는 GIL 때문에 안 된다(monitor 격리와 동일 — CONTRACT §11).
    /lowstate의 실측 관절각과 현재 출력 목표를 공유 메모리에 되써 UI가 읽을 수 있게 한다.

    Args:
        shared: ``multiprocessing.Array('d', _SM_LEN)`` — 위 레이아웃 상수 참고.
        stop_flag: ``multiprocessing.Value('i')`` — 1이면 루프 종료.
    """
    rclpy.init()
    node = rclpy.create_node("r2s_gui_publisher")
    pub = node.create_publisher(LowCmd, CMD_TOPIC, 10)

    def on_lowstate(msg: LowState) -> None:
        with shared.get_lock():
            for i in range(_NUM):
                shared[_SM_MEASURED_Q + i] = float(msg.motor_state[i].q)
            shared[_SM_MEASURED_VALID] = 1.0

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(LowState, STARTUP_STATE_TOPIC, on_lowstate, qos)

    # Policy 모드 전용 구독 — STARTUP_STATE_TOPIC(기동 자세 획득용, 위)과 목적이 달라 별도로 둔다
    # (STARTUP_STATE_TOPIC이 env로 sim에 오버라이드된 구성이면 "real"의 의미가 깨지므로). q/dq/IMU
    # 전부 필요해 on_lowstate보다 더 많이 캡처한다. 둘 다 DDS 순서 그대로 저장.
    def on_lowstate_policy_real(msg: LowState) -> None:
        with shared.get_lock():
            for i in range(_NUM):
                shared[_SM_POLICY_Q_REAL + i] = float(msg.motor_state[i].q)
                shared[_SM_POLICY_DQ_REAL + i] = float(msg.motor_state[i].dq)
            for i in range(3):
                shared[_SM_POLICY_GYRO_REAL + i] = float(msg.imu_state.gyroscope[i])
            for i in range(4):
                shared[_SM_POLICY_QUAT_REAL + i] = float(msg.imu_state.quaternion[i])
            shared[_SM_POLICY_VALID_REAL] = 1.0

    def on_lowstate_policy_sim(msg: LowState) -> None:
        with shared.get_lock():
            for i in range(_NUM):
                shared[_SM_POLICY_Q_SIM + i] = float(msg.motor_state[i].q)
                shared[_SM_POLICY_DQ_SIM + i] = float(msg.motor_state[i].dq)
            for i in range(3):
                shared[_SM_POLICY_GYRO_SIM + i] = float(msg.imu_state.gyroscope[i])
            for i in range(4):
                shared[_SM_POLICY_QUAT_SIM + i] = float(msg.imu_state.quaternion[i])
            shared[_SM_POLICY_VALID_SIM] = 1.0

    node.create_subscription(LowState, POLICY_REAL_STATE_TOPIC, on_lowstate_policy_real, qos)
    node.create_subscription(LowState, POLICY_SIM_STATE_TOPIC, on_lowstate_policy_sim, qos)

    # deployable 모델 로드 + 워밍업(§3) — 여기서 1회, 라이브 50Hz 루프 진입 전에 끝낸다. torch는
    # PolicyModel 내부에서 fork 이후(=지금, 이 프로세스 안)에만 import 된다(CUDA-after-fork 회피).
    policy_ctx = _PolicyContext()
    policy_ctx.ensure_loaded(node)
    recovery_ctx = _RecoveryContext()
    recovery_ctx.ensure_loaded(node)
    pedi_ctx = _PedipulationContext()
    pedi_ctx.ensure_loaded(node)

    # sim 뷰포트 마커 송신용 (fire-and-forget). UI 프로세스의 `_ctrl_sock` 과 별개다 —
    # 마커 상태는 publisher 로컬이라 UI 로 왕복시킬 이유가 없다.
    marker_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    marker_seq = 0
    marker_last_off = True  # OFF 를 매 tick 반복해 쏘지 않기 위한 상태

    period = PUBLISH_PERIOD_S
    next_t = time.monotonic()
    try:
        while not stop_flag.value and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            result = _compute_target(shared, now, policy_ctx, recovery_ctx, pedi_ctx)
            if result is not None:  # 시작 자세 획득 전(CMD_VALID=0)엔 발행 보류
                pose, kp, kd = result
                pub.publish(_build_lowcmd(pose, kp, kd))
                with shared.get_lock():  # UI가 보간 시작점으로 읽도록 현재 목표를 되쓴다
                    for i in range(_NUM):
                        shared[_SM_CURRENT_Q + i] = pose[i]

            # pedipulation 마커. 켜져 있으면 매 tick 보낸다(발·base 가 움직이므로 갱신이 필요).
            # 꺼져 있으면 **전환 순간 한 번만** 보내 숨긴다 — 계속 쏠 이유가 없다.
            leg = pedi_ctx.marker_leg
            if leg >= 0 or not marker_last_off:
                marker_seq += 1
                # sim 미기동이면 조용히 넘어간다 — 마커는 표시 전용이라 실패해도 제어에 영향 없다.
                with contextlib.suppress(OSError):
                    marker_sock.sendto(
                        r2s_udp.pack_marker(marker_seq, leg, pedi_ctx.marker_target_b),
                        (SIM_CTRL_HOST, SIM_CTRL_PORT),
                    )
                marker_last_off = leg < 0
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # 밀렸으면 리싱크(누적 드리프트 방지)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class MainWindow(QMainWindow):
    """R2S-GO2 GUI main window."""

    def __init__(self, shared) -> None:
        super().__init__()
        # 공유 메모리(mp.Array) — 목표를 여기 쓰면 publisher 프로세스가 50Hz로 발행한다.
        # UI가 잠깐 멈춰도(사용자 조작) publisher가 마지막 목표를 계속 스트림해 heartbeat 유지.
        self._shared = shared
        # 현재 자세 — 버튼(Stand/Sit/Step)의 보간 시작점. 실측 획득 전엔 STAND_FOLDED 를 기본값으로
        # 두어(버튼을 눌러도 안전) 이후 첫 /lowstate 로 실측 자세로 덮어쓴다.
        self._current_pose: list[float] = list(motions.STAND_FOLDED)
        # 명령을 한 번이라도 공유 메모리에 썼는가. **시작 자세 획득 전엔 None**이라 아직 아무것도 쓰지
        # 않는다(CMD_VALID=0) — 로봇이 켜지자마자 시작 자세로 툭 스냅하는 걸 막는다(_on_startup_tick).
        self._latest_pose: list[float] | None = None

        # Sine sweep 상태 — 모션 생성은 publisher가 하므로 여기선 실행 여부/기준 자세만 추적한다.
        self._sine_active: bool = False
        self._sine_base_pose: list[float] = list(motions.DEFAULT_POSE)
        self._sine_start_time: float = 0.0

        # Policy 모드 상태 — 추론은 publisher 프로세스가 하므로(§3) 여기선 UI 버튼 상태 + engage
        # 대기 타이머만 추적한다. Start 클릭 시 안전 handover(DEFAULT_POSE로 보간 후 engage, 팀 결정)
        # 를 위해 일회성 QTimer로 체이닝한다(50Hz 연속 발행과 무관한 1회성 이벤트라 UI 타이머로 충분
        # — 이 파일 상단 docstring이 배제하는 건 "연속 목표 생성"이지 1회성 지연 실행이 아니다).
        self._policy_active: bool = False
        self._policy_engage_timer: QTimer | None = None

        # Recovery 모드 상태 — Policy와 달리 engage 타이머가 없다(즉시 engage, `_on_recovery_start_clicked`).
        self._recovery_active: bool = False
        # Pedipulation 모드 상태 — recovery 와 같이 즉시 engage 한다. 다만 "4족으로 서 있을 때"가
        # 전제라 publisher 쪽에서 nominal latch 게이트(발 깊이 0.20~0.40 m)가 한 번 더 막는다.
        self._pedi_active: bool = False

        # Sim 제어 채널 (카메라/플랜트). publisher 프로세스를 거치지 않고 UI 가 직접 UDP 로 보낸다 —
        # 버튼을 누를 때만 나가는 one-shot 이라 50Hz heartbeat 와 경합하지 않는다(tuner_gui 와 같은 방식).
        self._ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ctrl_seq: int = 0

        # Monitor 는 **별도 프로세스**로 spawn 한다 (렌더를 gui heartbeat 에서 격리; CONTRACT §11).
        # None=미실행. subprocess.Popen 핸들.
        self._monitor_proc: subprocess.Popen | None = None

        self.setWindowTitle("R2S-GO2 Controller")
        self._build_ui()

        # sine 실행 중 스핀박스/콤보를 바꾸면 publisher에 파라미터만 갱신(모션은 안 끊김).
        self._sine_joint_combo.currentIndexChanged.connect(self._on_sine_param_changed)
        self._sine_amp_spin.valueChanged.connect(self._on_sine_param_changed)
        self._sine_freq_spin.valueChanged.connect(self._on_sine_param_changed)

        # policy 실행 중 슬라이더/소스를 바꾸면 publisher에 cmd만 갱신(engage는 안 끊김) — sine과 동일 패턴.
        self._policy_xvel_spin.valueChanged.connect(self._on_policy_param_changed)
        self._policy_yaw_spin.valueChanged.connect(self._on_policy_param_changed)
        self._policy_source_combo.currentIndexChanged.connect(self._on_policy_param_changed)

        # recovery 는 조정할 명령이 없어 소스 콤보만 라이브 반영한다(policy 의 cmd 갱신과 동일 패턴).
        self._recovery_source_combo.currentIndexChanged.connect(self._on_recovery_param_changed)

        # pedipulation 은 실행 중에도 조작 다리·목표 offset 을 바꿀 수 있다(정책이 명령 변화를
        # 겪도록 학습됐다 — 에피소드 중 재샘플). policy 의 cmd 갱신과 같은 라이브 반영 패턴.
        self._pedi_source_combo.currentIndexChanged.connect(self._on_pedi_param_changed)
        self._pedi_leg_combo.currentIndexChanged.connect(self._on_pedi_param_changed)
        for _spin in self._pedi_off_spins:
            _spin.valueChanged.connect(self._on_pedi_param_changed)

        # Sim 설정은 값이 바뀔 때만 sim_runner 로 one-shot 전송한다(50Hz 스트림과 무관).
        self._sim_camera_combo.currentIndexChanged.connect(self._on_sim_ctrl_changed)
        self._sim_plant_combo.currentIndexChanged.connect(self._on_sim_ctrl_changed)

        # /lowcmd 발행 + 목표 생성(보간/사인)은 별도 publisher 프로세스가 담당한다(이 UI 프로세스엔
        # 발행/모션 타이머·rclpy spin 없음) — UI 조작이 heartbeat·목표 갱신을 굶기지 못하게 하는 핵심.
        # UI는 공유 메모리에 "모션 스펙"만 쓴다.

        # 시작 자세 접근 상태머신 — 첫 /lowstate 로 로봇 현재 자세를 잡아 시작 자세로 보간(스냅 방지).
        self._startup_done: bool = False
        self._startup_deadline: float = time.monotonic() + STARTUP_ACQUIRE_TIMEOUT_S
        self._startup_timer = QTimer(self)
        self._startup_timer.timeout.connect(self._on_startup_tick)
        self._startup_timer.start(int(SPIN_PERIOD_S * 1000))

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        # -- Header --
        title_label = QLabel("R2S-GO2 Controller")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("Unitree GO2 — real2sim /lowcmd publisher")
        subtitle_label.setObjectName("subtitleLabel")
        header_title = QVBoxLayout()
        header_title.setSpacing(2)
        header_title.addWidget(title_label)
        header_title.addWidget(subtitle_label)
        header_layout = QHBoxLayout()
        header_layout.addLayout(header_title)
        header_layout.addStretch(1)
        monitor_btn = QPushButton("Monitor")
        monitor_btn.clicked.connect(self._on_monitor_clicked)
        header_layout.addWidget(monitor_btn, alignment=Qt.AlignTop)
        layout.addLayout(header_layout)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        layout.addWidget(divider)

        # -- Default / Sit --
        pose_group = QGroupBox("Pose")
        pose_layout = QHBoxLayout(pose_group)
        pose_layout.setSpacing(10)
        standup_btn = QPushButton("Stand Up")
        standup_btn.setObjectName("primaryButton")
        standup_btn.clicked.connect(self._on_standup_clicked)
        default_btn = QPushButton("Default (stand)")
        default_btn.clicked.connect(self._on_default_clicked)
        sit_btn = QPushButton("Sit")
        sit_btn.clicked.connect(self._on_sit_clicked)
        pose_layout.addWidget(standup_btn)
        pose_layout.addWidget(default_btn)
        pose_layout.addWidget(sit_btn)
        pose_layout.addStretch(1)
        layout.addWidget(pose_group)

        # -- Joint step --
        step_group = QGroupBox("Joint Step")
        step_layout = QHBoxLayout(step_group)
        step_layout.setSpacing(10)
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

        # -- Sine sweep --
        sine_group = QGroupBox("Sine Sweep")
        sine_layout = QHBoxLayout(sine_group)
        sine_layout.setSpacing(10)
        sine_layout.addWidget(QLabel("Joint:"))
        self._sine_joint_combo = QComboBox()
        self._sine_joint_combo.addItems(motions.JOINT_NAMES)
        sine_layout.addWidget(self._sine_joint_combo)
        sine_layout.addWidget(QLabel("Amplitude [rad]:"))
        self._sine_amp_spin = QDoubleSpinBox()
        self._sine_amp_spin.setRange(0.0, 1.5)
        self._sine_amp_spin.setSingleStep(0.05)
        self._sine_amp_spin.setValue(0.2)
        sine_layout.addWidget(self._sine_amp_spin)
        sine_layout.addWidget(QLabel("Frequency [Hz]:"))
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

        # -- Policy (deployable RMA+estimator 정책, publisher 프로세스에서 50Hz 추론 — §3) --
        policy_group = QGroupBox("Policy")
        policy_layout = QHBoxLayout(policy_group)
        policy_layout.setSpacing(10)
        policy_layout.addWidget(QLabel("x_vel [m/s]:"))
        self._policy_xvel_spin = QDoubleSpinBox()
        self._policy_xvel_spin.setRange(*policy_runtime.LIN_VEL_X_RANGE)
        self._policy_xvel_spin.setSingleStep(0.1)
        self._policy_xvel_spin.setValue(0.0)
        policy_layout.addWidget(self._policy_xvel_spin)
        policy_layout.addWidget(QLabel("yaw_vel [rad/s]:"))
        self._policy_yaw_spin = QDoubleSpinBox()
        self._policy_yaw_spin.setRange(*policy_runtime.YAW_VEL_RANGE)
        self._policy_yaw_spin.setSingleStep(0.1)
        self._policy_yaw_spin.setValue(0.0)
        policy_layout.addWidget(self._policy_yaw_spin)
        policy_layout.addWidget(QLabel("Source:"))
        self._policy_source_combo = QComboBox()
        self._policy_source_combo.addItems(["Sim", "Real"])  # index 0=Sim, 1=Real (_POLICY_SOURCE_*)
        policy_layout.addWidget(self._policy_source_combo)
        self._policy_start_btn = QPushButton("Start")
        self._policy_start_btn.setObjectName("primaryButton")
        self._policy_start_btn.clicked.connect(self._on_policy_start_clicked)
        policy_layout.addWidget(self._policy_start_btn)
        self._policy_stop_btn = QPushButton("Stop")
        self._policy_stop_btn.setObjectName("dangerButton")
        self._policy_stop_btn.clicked.connect(self._on_policy_stop_clicked)
        self._policy_stop_btn.setEnabled(False)
        policy_layout.addWidget(self._policy_stop_btn)
        policy_layout.addStretch(1)
        layout.addWidget(policy_group)

        # -- Recovery (넘어진 상태에서 기립하는 fall-recovery 정책) --
        recovery_group = QGroupBox("Recovery (fall → stand)")
        recovery_layout = QHBoxLayout(recovery_group)
        recovery_layout.setSpacing(10)
        # 명령 위젯 없음 — 이 정책은 목표가 "default 자세로 일어서라" 하나로 고정이다.
        recovery_layout.addWidget(QLabel("Source:"))
        self._recovery_source_combo = QComboBox()
        self._recovery_source_combo.addItems(["Sim", "Real"])  # index 0=Sim, 1=Real (_POLICY_SOURCE_*)
        recovery_layout.addWidget(self._recovery_source_combo)
        self._recovery_start_btn = QPushButton("Start")
        self._recovery_start_btn.setObjectName("primaryButton")
        self._recovery_start_btn.clicked.connect(self._on_recovery_start_clicked)
        recovery_layout.addWidget(self._recovery_start_btn)
        self._recovery_stop_btn = QPushButton("Stop")
        self._recovery_stop_btn.setObjectName("dangerButton")
        self._recovery_stop_btn.clicked.connect(self._on_recovery_stop_clicked)
        self._recovery_stop_btn.setEnabled(False)
        recovery_layout.addWidget(self._recovery_stop_btn)
        recovery_hint = QLabel("For a fallen robot only (kd=1.0). Do not press while it is standing")
        recovery_hint.setObjectName("subtitleLabel")
        recovery_layout.addWidget(recovery_hint)
        recovery_layout.addStretch(1)
        layout.addWidget(recovery_group)

        # -- Pedipulation (다리 하나를 조작기로: 세 다리로 균형, 한 발을 목표 위치로) --
        pedi_group = QGroupBox("Pedipulation (3-leg balance + 1 foot as manipulator)")
        pedi_layout = QHBoxLayout(pedi_group)
        pedi_layout.setSpacing(10)

        pedi_layout.addWidget(QLabel("Source:"))
        self._pedi_source_combo = QComboBox()
        self._pedi_source_combo.addItems(["Sim", "Real"])  # index 0=Sim, 1=Real (_POLICY_SOURCE_*)
        pedi_layout.addWidget(self._pedi_source_combo)

        pedi_layout.addWidget(QLabel("Leg:"))
        self._pedi_leg_combo = QComboBox()
        self._pedi_leg_combo.addItems(pedipulation_runtime.LEG_NAMES)  # FL, FR, RL, RR
        pedi_layout.addWidget(self._pedi_leg_combo)

        # 목표 offset — 범위는 학습 명령 박스 그대로다. 넘어가면 분포 밖이라 스핀박스가 막는다.
        self._pedi_off_spins: list[QDoubleSpinBox] = []
        for label, (lo, hi) in zip(
            ("dx", "dy", "dz"),
            (pedipulation_runtime.CMD_BOX_X, pedipulation_runtime.CMD_BOX_Y, pedipulation_runtime.CMD_BOX_Z),
        ):
            pedi_layout.addWidget(QLabel(f"{label}:"))
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setValue(0.0)
            spin.setSuffix(" m")
            pedi_layout.addWidget(spin)
            self._pedi_off_spins.append(spin)

        self._pedi_start_btn = QPushButton("Start")
        self._pedi_start_btn.setObjectName("primaryButton")
        self._pedi_start_btn.clicked.connect(self._on_pedi_start_clicked)
        pedi_layout.addWidget(self._pedi_start_btn)
        self._pedi_stop_btn = QPushButton("Stop")
        self._pedi_stop_btn.setObjectName("dangerButton")
        self._pedi_stop_btn.clicked.connect(self._on_pedi_stop_clicked)
        self._pedi_stop_btn.setEnabled(False)
        pedi_layout.addWidget(self._pedi_stop_btn)

        pedi_hint = QLabel("Press while standing on all four — the nominal pose is latched at that moment")
        pedi_hint.setObjectName("subtitleLabel")
        pedi_layout.addWidget(pedi_hint)
        pedi_layout.addStretch(1)
        layout.addWidget(pedi_group)

        # -- Sim (Isaac 쪽 설정 — UDP 제어 채널로 sim_runner 에 직접 보낸다) --
        sim_group = QGroupBox("Sim")
        sim_layout = QHBoxLayout(sim_group)
        sim_layout.setSpacing(10)
        sim_layout.addWidget(QLabel("Camera:"))
        self._sim_camera_combo = QComboBox()
        self._sim_camera_combo.addItems(["Follow robot", "Free"])  # index 0=Follow, 1=Free
        self._sim_camera_combo.setToolTip("Follow = camera tracks the robot base / Free = Isaac viewport is yours")
        sim_layout.addWidget(self._sim_camera_combo)
        sim_layout.addWidget(QLabel("Plant:"))
        self._sim_plant_combo = QComboBox()
        # 콤보 순서 = PLANT_COMBO_ORDER 인덱스. 첫 항목이 기본 선택이므로 학습 플랜트인 set2 가 먼저다.
        self._sim_plant_combo.addItems(
            [
                "set2 — PACE live (training plant)",
                "set3 — 2026-08-04 recapture (not adopted)",
                "Default — nominal (no friction)",
            ]
        )
        self._sim_plant_combo.setToolTip(
            "Joint-property presets, for eyeballing how closely sim tracks the real GO2.\n\n"
            "set2  armature 0.001~0.016 / viscous ~0 / coulomb 0.15~0.67  <- plant the current policy trained on\n"
            "set3  armature 0.005~0.021 / viscous 0.13~0.18 / coulomb 0.10~0.58\n"
            "        Fits the new 5 Hz capture. coulomb and calf armature passed independent checks; viscous did not.\n"
            "Default  armature 0.01 / no friction — UNITREE_GO2_CFG defaults\n\n"
            "Warning: the Policy (section 11) net trained on the set2 plant — on Default it cannot walk.\n"
            "Warning: do not mix presets. The three parameters were identified jointly, "
            "so partial combinations have no basis."
        )
        sim_layout.addWidget(self._sim_plant_combo)
        sim_hint = QLabel("Sent straight to sim_runner (ignored when sim is not running)")
        sim_hint.setObjectName("subtitleLabel")
        sim_layout.addWidget(sim_hint)
        sim_layout.addStretch(1)
        layout.addWidget(sim_group)

        layout.addStretch(1)

        # -- Status bar --
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

    # -- 모션 스펙 쓰기 (공유 메모리 → publisher 프로세스) --
    #
    # UI는 "무엇을 하라"는 스펙만 쓰고, 목표 보간/사인은 publisher가 50Hz 균일하게 계산한다.
    # 그래서 UI 조작으로 이 UI 프로세스가 잠깐 밀려도 목표값은 매끄럽게(끊김 없이) 갱신된다.

    def _read_current_target(self) -> list[float]:
        """publisher가 현재 발행 중인 목표를 읽는다 — 새 모션의 보간 시작점(모션 중 눌러도 정확)."""
        with self._shared.get_lock():
            if self._shared[_SM_CMD_VALID] < 0.5:
                return list(self._current_pose)
            return [self._shared[_SM_CURRENT_Q + i] for i in range(_NUM)]

    def _write_hold(self, pose: list[float], kd: float = motions.DEFAULT_KD) -> None:
        """고정 자세 유지 명령.

        Args:
            pose: 유지할 관절각(DDS 순서).
            kd: 발행할 damping 게인. Recovery Stop 은 :data:`recovery_runtime.RECOVERY_KD`(1.0)를
                넘긴다 — 방금 기립을 마친 로봇의 kd 를 그 순간 절반으로 떨어뜨리지 않기 위해서다.
        """
        self._sine_active = False
        self._policy_active = False
        self._recovery_active = False
        self._pedi_active = False
        self._cancel_policy_engage()  # 대기 중인 engage 타이머가 나중에 엉뚱하게 발화하는 것 방지(안전)
        self._current_pose = list(pose)
        self._latest_pose = list(pose)
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_HOLD
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = kd
            for i in range(_NUM):
                self._shared[_SM_BASE_Q + i] = float(pose[i])
            self._shared[_SM_CMD_VALID] = 1.0

    def _write_sequence(self, seq: list[list[float]]) -> None:
        """프레임 시퀀스를 publisher가 경과 시간으로 재생하도록 공유 메모리에 쓴다(sine/policy 정지).

        ⚠ Policy Start 버튼도 안전 handover(팀 결정, §5)로 이 경로(→ `_play_sequence_to`)를 거친다 —
        그 시점엔 아직 engage 타이머가 없어(아래서 만들기 전) `_cancel_policy_engage()`가 no-op이라
        문제없다(호출 순서: 보간 시작 → 여기 도달 → 리턴 후에야 engage 타이머 생성).
        """
        self._sine_active = False
        self._policy_active = False
        self._recovery_active = False
        self._pedi_active = False
        self._cancel_policy_engage()
        if len(seq) > MAX_FRAMES:
            seq = seq[:MAX_FRAMES]  # 상한 초과분은 잘라낸다(10s 이상 시퀀스는 없음)
        self._current_pose = list(seq[-1])  # 시퀀스 종료 자세 (다음 버튼 보간 시작점 근사)
        self._latest_pose = list(seq[-1])
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_SEQUENCE
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = motions.DEFAULT_KD
            self._shared[_SM_FRAME_HZ] = FRAME_HZ
            self._shared[_SM_FRAME_COUNT] = float(len(seq))
            self._shared[_SM_START_TIME] = time.monotonic()
            for f, frame in enumerate(seq):
                off = _SM_FRAMES + f * _NUM
                for i in range(_NUM):
                    self._shared[off + i] = float(frame[i])
            self._shared[_SM_CMD_VALID] = 1.0

    def _write_sine(self, base: list[float], joint: int, amp: float, freq: float) -> None:
        """선택 관절에 사인을 주입하는 모션을 publisher가 생성하도록 공유 메모리에 쓴다."""
        self._sine_active = True
        self._policy_active = False
        self._recovery_active = False
        self._pedi_active = False
        self._cancel_policy_engage()
        self._sine_base_pose = list(base)
        self._sine_start_time = time.monotonic()
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_SINE
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = motions.DEFAULT_KD
            for i in range(_NUM):
                self._shared[_SM_BASE_Q + i] = float(base[i])
            self._shared[_SM_SINE_JOINT] = float(joint)
            self._shared[_SM_SINE_AMP] = float(amp)
            self._shared[_SM_SINE_FREQ] = float(freq)
            self._shared[_SM_START_TIME] = self._sine_start_time
            self._shared[_SM_CMD_VALID] = 1.0

    def _update_sine_params(self, joint: int, amp: float, freq: float) -> None:
        """sine 실행 중 파라미터만 갱신(관절/진폭/주파수). UI가 바빠도 publisher는 이전 값으로 계속 돈다."""
        if not self._sine_active:
            return
        with self._shared.get_lock():
            self._shared[_SM_SINE_JOINT] = float(joint)
            self._shared[_SM_SINE_AMP] = float(amp)
            self._shared[_SM_SINE_FREQ] = float(freq)

    def _write_policy(self, xvel: float, yaw_vel: float, source: int) -> None:
        """Policy 모드 진입 명령. sine/sequence와 달리 목표를 여기서 미리 굽지 않는다 — 매 tick
        publisher가 실측 proprio + 정책 추론으로 새로 계산하므로(§3), 여기선 모드/cmd만 쓴다.
        `_SM_BASE_Q`는 실측 미수신/모델 미로드 시 publisher의 fallback-hold 자세로 쓰인다 — 방금
        보간해 도착한 DEFAULT_POSE와 같은 값을 넣어(팀 결정 §5 handover) fallback도 안전하게 만든다.
        """
        self._sine_active = False
        self._policy_active = True
        self._recovery_active = False
        self._pedi_active = False
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_POLICY
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = motions.DEFAULT_KD
            for i in range(_NUM):
                self._shared[_SM_BASE_Q + i] = float(motions.DEFAULT_POSE[i])
            self._shared[_SM_POLICY_XVEL] = float(xvel)
            self._shared[_SM_POLICY_YAWVEL] = float(yaw_vel)
            self._shared[_SM_POLICY_INPUT_SOURCE] = float(source)
            self._shared[_SM_CMD_VALID] = 1.0
        self._current_pose = list(motions.DEFAULT_POSE)
        self._latest_pose = list(motions.DEFAULT_POSE)

    def _update_policy_cmd(self, xvel: float, yaw_vel: float, source: int) -> None:
        """policy 실행 중 cmd/입력소스만 갱신 — `_update_sine_params`와 동일 패턴."""
        if not self._policy_active:
            return
        with self._shared.get_lock():
            self._shared[_SM_POLICY_XVEL] = float(xvel)
            self._shared[_SM_POLICY_YAWVEL] = float(yaw_vel)
            self._shared[_SM_POLICY_INPUT_SOURCE] = float(source)

    def _write_recovery(self, source: int) -> None:
        """Recovery 모드 진입 명령. policy 모드와 같이 목표를 미리 굽지 않고 매 tick publisher 가
        실측 obs + 정책 추론으로 계산한다. 차이는 둘:

        - **kd 를 1.0 으로 발행한다** — 이 정책의 학습 액추에이터가 damping 1.0 이다
          (`go2_recovery_env_cfg.py`). GUI 기본값 0.5 로 돌리면 학습과 다른 플랜트가 된다.
        - `_SM_BASE_Q` fallback 을 **현재 발행 중인 목표**로 둔다. policy 모드는 방금 보간해 도착한
          DEFAULT_POSE 를 넣지만, recovery 는 넘어진 상태에서 시작하므로 default 로 fallback 하면
          모델 로드 실패·실측 미수신 시 오히려 바닥에서 다리를 뻗어버린다.
        """
        self._sine_active = False
        self._policy_active = False
        self._recovery_active = True
        self._cancel_policy_engage()
        hold = self._read_current_target()
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_RECOVERY
            self._shared[_SM_KP] = recovery_runtime.RECOVERY_KP
            self._shared[_SM_KD] = recovery_runtime.RECOVERY_KD
            for i in range(_NUM):
                self._shared[_SM_BASE_Q + i] = float(hold[i])
            self._shared[_SM_RECOVERY_INPUT_SOURCE] = float(source)
            self._shared[_SM_CMD_VALID] = 1.0
        self._current_pose = list(hold)
        self._latest_pose = list(hold)

    def _update_recovery_source(self, source: int) -> None:
        """recovery 실행 중 입력 소스만 갱신 — `_update_policy_cmd`와 동일 패턴."""
        if not self._recovery_active:
            return
        with self._shared.get_lock():
            self._shared[_SM_RECOVERY_INPUT_SOURCE] = float(source)

    def _stop_recovery(self) -> None:
        """recovery 비활성화 + 버튼 상태 복원 — `_stop_policy`와 동일 역할."""
        self._recovery_active = False
        self._recovery_start_btn.setEnabled(True)
        self._recovery_stop_btn.setEnabled(False)

    def _write_pedipulation(self, source: int, manip_leg: int, offset: tuple[float, float, float]) -> None:
        """Pedipulation 모드 진입 명령. 매 tick publisher 가 실측 obs + 정책 추론으로 목표를 만든다.

        kp/kd 는 **기본값 그대로**다 — 이 정책의 학습 액추에이터가 stiffness 25 / damping 0.5 로
        `motions.DEFAULT_KP/KD` 와 같다(recovery 처럼 따로 실을 이유가 없다).

        `_SM_BASE_Q` fallback 은 recovery 와 같이 **현재 발행 중인 목표**로 둔다. 모델 로드 실패나
        nominal 미확보(4족이 아닐 때) 시 default 로 끌고 가면 서 있던 자세가 튀기 때문이다.
        """
        self._sine_active = False
        self._policy_active = False
        self._recovery_active = False
        self._pedi_active = True
        self._cancel_policy_engage()
        hold = self._read_current_target()
        with self._shared.get_lock():
            self._shared[_SM_MODE] = _MODE_PEDIPULATION
            self._shared[_SM_KP] = motions.DEFAULT_KP
            self._shared[_SM_KD] = motions.DEFAULT_KD
            for i in range(_NUM):
                self._shared[_SM_BASE_Q + i] = float(hold[i])
            self._shared[_SM_PEDI_INPUT_SOURCE] = float(source)
            self._shared[_SM_PEDI_MANIP_LEG] = float(manip_leg)
            self._shared[_SM_PEDI_OFF_X] = float(offset[0])
            self._shared[_SM_PEDI_OFF_Y] = float(offset[1])
            self._shared[_SM_PEDI_OFF_Z] = float(offset[2])
            self._shared[_SM_CMD_VALID] = 1.0
        self._current_pose = list(hold)
        self._latest_pose = list(hold)

    def _update_pedi_cmd(self, source: int, manip_leg: int, offset: tuple[float, float, float]) -> None:
        """pedipulation 실행 중 명령만 갱신 — `_update_policy_cmd`와 동일 패턴.

        조작 다리를 바꾸면 정책이 그 자리에서 역할을 전환한다(S3 발 교체와 같은 조작). 적분형
        목표는 publisher 가 이어서 들고 가므로 여기서 건드리지 않는다.
        """
        if not self._pedi_active:
            return
        with self._shared.get_lock():
            self._shared[_SM_PEDI_INPUT_SOURCE] = float(source)
            self._shared[_SM_PEDI_MANIP_LEG] = float(manip_leg)
            self._shared[_SM_PEDI_OFF_X] = float(offset[0])
            self._shared[_SM_PEDI_OFF_Y] = float(offset[1])
            self._shared[_SM_PEDI_OFF_Z] = float(offset[2])

    def _stop_pedi(self) -> None:
        """pedipulation 비활성화 + 버튼 상태 복원 — `_stop_recovery`와 동일 역할."""
        self._pedi_active = False
        self._pedi_start_btn.setEnabled(True)
        self._pedi_stop_btn.setEnabled(False)

    def _pedi_offset(self) -> tuple[float, float, float]:
        """스핀박스 3개에서 목표 offset 을 읽는다 [m]."""
        return (
            self._pedi_off_spins[0].value(),
            self._pedi_off_spins[1].value(),
            self._pedi_off_spins[2].value(),
        )

    def _cancel_policy_engage(self) -> None:
        """대기 중인 policy engage 타이머(있으면) 취소. 다른 모션 명령이 boarding 도중 끼어들었는데
        타이머가 그대로 살아있으면 몇 초 뒤 엉뚱하게 policy로 전환돼버리므로 — sine의 저위험
        파라미터 갱신과 달리 이건 안전 문제라 `_write_hold`/`_write_sequence`/`_write_sine`에서
        항상 호출한다."""
        if self._policy_engage_timer is not None:
            self._policy_engage_timer.stop()
            self._policy_engage_timer = None

    def _stop_policy(self) -> None:
        """policy 비활성화 + 버튼 상태 복원(Stop 클릭 시 호출) — `_stop_sine`과 동일 역할."""
        self._policy_active = False
        self._policy_start_btn.setEnabled(True)
        self._policy_stop_btn.setEnabled(False)

    # -- Shared sequence playback --

    def _play_poses(self, seq: list[list[float]]) -> None:
        """Play a pre-built pose-frame sequence via the publisher process (also stops sine).

        Stand Up/Default/Sit/Step 이 전부 이 경로를 지난다. `_write_sequence` 는 `_recovery_active`
        플래그만 끄므로 버튼 복원은 여기서 한다 — 안 하면 Recovery 중에 Stand Up 을 누른 사용자가
        Start 가 비활성인 채로 남아 GUI 재시작 없이는 Recovery 를 다시 못 켠다.
        """
        self._stop_recovery()
        self._stop_pedi()
        self._write_sequence(seq)

    def _play_sequence_to(self, goal_pose: list[float], duration_s: float) -> None:
        """Interpolate from the current published target to goal_pose over duration_s (also stops sine)."""
        start = self._read_current_target()  # publisher의 실제 현재 목표 (모션 중 눌러도 정확)
        num_steps = max(1, int(duration_s * FRAME_HZ))
        self._play_poses(motions.interpolate_sequence(start, goal_pose, num_steps))

    # -- Pose buttons --

    def _on_standup_clicked(self) -> None:
        # go2_stand_example trajectory: current -> folded -> stand (hold). Same staged
        # method as the C++ example; ends standing (the example's final splay is omitted).
        self._status_label.setText("Standing up (go2_stand_example trajectory)...")
        self._play_poses(motions.stand_up_sequence(self._read_current_target(), FRAME_HZ))

    def _on_monitor_clicked(self) -> None:
        # 별도 프로세스로 monitor.py 를 spawn — 무거운 matplotlib 렌더를 gui 의 50Hz /lowcmd
        # heartbeat event loop 에서 완전히 격리한다 (CONTRACT §11). env(RMW/도메인/CYCLONEDDS_URI)
        # 는 자동 상속되어 같은 토픽을 관측한다. 이미 실행 중이면(살아있으면) 무시.
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._status_label.setText("Monitor already running")
            return
        monitor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor.py")
        self._monitor_proc = subprocess.Popen([sys.executable, monitor_py])
        self._status_label.setText("Monitor launched (separate process)")

    def _on_default_clicked(self) -> None:
        self._status_label.setText("Moving to default pose...")
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)

    def _on_sit_clicked(self) -> None:
        self._status_label.setText("Moving to sit pose...")
        self._play_sequence_to(motions.SIT_POSE, SEQUENCE_DURATION_S)

    def _on_step_clicked(self) -> None:
        joint_idx = self._step_joint_combo.currentIndex()
        delta = self._step_delta_spin.value()
        goal_pose = motions.step_pose(self._read_current_target(), joint_idx, delta)
        self._status_label.setText(f"Stepping {motions.JOINT_NAMES[joint_idx]} ({delta:+.2f} rad)...")
        self._play_sequence_to(goal_pose, STEP_DURATION_S)

    # -- Sine sweep (모션 생성은 publisher 프로세스가 함) --

    def _on_sine_start_clicked(self) -> None:
        base = self._read_current_target()  # 현재 목표를 사인 기준으로
        joint = self._sine_joint_combo.currentIndex()
        amp = self._sine_amp_spin.value()
        freq = self._sine_freq_spin.value()
        self._stop_recovery()  # 다른 모드가 켜져 있었으면 버튼 상태까지 정리(_write_sine 은 플래그만 끈다)
        self._stop_pedi()
        self._sine_start_btn.setEnabled(False)
        self._sine_stop_btn.setEnabled(True)
        self._status_label.setText(f"Sine sweep on {self._sine_joint_combo.currentText()}...")
        self._write_sine(base, joint, amp, freq)

    def _on_sine_param_changed(self) -> None:
        # sine 실행 중 스핀박스/콤보 변경 → publisher에 파라미터만 갱신(모션은 안 끊김).
        self._update_sine_params(
            self._sine_joint_combo.currentIndex(),
            self._sine_amp_spin.value(),
            self._sine_freq_spin.value(),
        )

    def _on_sine_stop_clicked(self) -> None:
        # **stand 자세로 보간**한다. 사인 도중 어느 위상에서 멈췄든 `_play_sequence_to` 가
        # publisher 의 현재 목표에서 출발하므로 정지 순간 target 이 튀지 않는다
        # (예전에는 사인 위상을 되계산해 그 자세로 홀드했다).
        self._stop_sine()
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)
        self._status_label.setText("Sine sweep stopped — returning to the stand pose")

    def _stop_sine(self) -> None:
        self._sine_active = False
        self._sine_start_btn.setEnabled(True)
        self._sine_stop_btn.setEnabled(False)

    # -- Policy (deployable RMA+estimator 정책) --

    def _on_policy_start_clicked(self) -> None:
        # 안전 handover(팀 결정, §5): 현재 자세 → DEFAULT_POSE로 먼저 보간한 뒤에만 engage한다.
        # 정책이 default 근처 자세를 가정하고 학습됐고, 실하드웨어에서 첫 액션이 임의 자세 기준으로
        # 튀는 것(낙상 위험)보다 안전이 우선이라 Sine/Step의 무보호 즉시-시작과는 다르게 간다.
        self._status_label.setText("Policy: moving to the default pose (waiting to engage)...")
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)  # _play_poses 가 recovery 도 정리
        # 위 호출(→ _write_sequence)이 정책 버튼 상태를 건드리지 않으므로 여기서 명시적으로 설정
        # (sine과 동일 관례 — write_* 는 *_active 플래그만, 버튼 복원/설정은 핸들러가 담당).
        self._policy_start_btn.setEnabled(False)
        self._policy_stop_btn.setEnabled(True)

        xvel = self._policy_xvel_spin.value()
        yaw_vel = self._policy_yaw_spin.value()
        source = self._policy_source_combo.currentIndex()  # 0=Sim, 1=Real (_POLICY_SOURCE_*)
        self._policy_engage_timer = QTimer(self)
        self._policy_engage_timer.setSingleShot(True)
        self._policy_engage_timer.timeout.connect(lambda: self._engage_policy(xvel, yaw_vel, source))
        self._policy_engage_timer.start(int(SEQUENCE_DURATION_S * 1000))

    def _engage_policy(self, xvel: float, yaw_vel: float, source: int) -> None:
        self._policy_engage_timer = None  # 이미 발화 — 더 취소할 대상 없음
        self._write_policy(xvel, yaw_vel, source)
        src_label = "Real" if source == _POLICY_SOURCE_REAL else "Sim"
        self._status_label.setText(f"Policy: engaged (source={src_label}, x_vel={xvel:.2f}, yaw_vel={yaw_vel:.2f})")

    def _on_policy_param_changed(self) -> None:
        # policy 실행 중 슬라이더/소스 변경 → publisher에 cmd만 갱신(engage는 안 끊김).
        self._update_policy_cmd(
            self._policy_xvel_spin.value(),
            self._policy_yaw_spin.value(),
            self._policy_source_combo.currentIndex(),
        )

    def _on_policy_stop_clicked(self) -> None:
        # engage 대기 중이었으면 그 타이머부터 취소한다(안 그러면 뒤늦게 engage 되어버린다).
        # 이후 **stand 자세로 보간**한다 — 정책이 남긴 임의 자세로 굳어 있으면 다음 명령을 주기
        # 전에 사람이 손으로 세워야 하기 때문이다. 현재 목표에서 출발하므로 target 은 안 튄다.
        self._cancel_policy_engage()
        self._stop_policy()
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)
        self._status_label.setText("Policy stopped — returning to the stand pose")

    # -- Sim 설정 (카메라 / 관절 물성) — sim_runner 로 UDP one-shot --

    def _on_sim_ctrl_changed(self) -> None:
        """Camera/Plant 콤보가 바뀌면 현재 두 값을 그대로 sim_runner 에 보낸다.

        전송은 fire-and-forget 이다 — sim 이 안 떠 있으면 아무 일도 일어나지 않는다(로컬 UDP 라
        보통 조용히 버려지고, 포트가 닫혀 ICMP 가 오면 다음 send 에서 예외가 나므로 흡수한다).
        sim_runner 는 **값이 실제로 바뀔 때만** 적용하므로 중복 전송도 무해하다.
        """
        camera = r2s_udp.CAMERA_FOLLOW if self._sim_camera_combo.currentIndex() == 0 else r2s_udp.CAMERA_FREE
        plant = PLANT_COMBO_ORDER[self._sim_plant_combo.currentIndex()]
        self._ctrl_seq += 1
        try:
            self._ctrl_sock.sendto(r2s_udp.pack_ctrl(self._ctrl_seq, camera, plant), (SIM_CTRL_HOST, SIM_CTRL_PORT))
        except OSError as exc:  # sim 미기동 등 — 상태줄에만 알리고 넘어간다
            self._status_label.setText(f"Failed to send sim setting ({exc}) — check that sim_runner is up")
            return
        cam_label = self._sim_camera_combo.currentText()
        plant_label = self._sim_plant_combo.currentText()
        self._status_label.setText(f"Sim: camera={cam_label}, plant={plant_label}")

    # -- Recovery (fall-recovery 정책) --

    def _on_recovery_start_clicked(self) -> None:
        # Policy Start 와 달리 **DEFAULT_POSE 로 보간하지 않고 즉시 engage** 한다. 로봇이 넘어져
        # 있는 상태이므로 default 로 끌고 가는 것은 위험할 뿐 아니라 학습 분포에서도 벗어난다:
        # 학습 env 의 settle 구간은 측정 자세를 그대로 유지하고(settle_mode="passive"), 그 길이가
        # env 마다 U[0, 100] step 이라 **0 step**(=첫 tick 부터 정책이 구동)도 학습에 포함돼 있다.
        self._stop_sine()
        self._stop_policy()  # 다른 모드가 켜져 있었으면 버튼 상태까지 정리
        source = self._recovery_source_combo.currentIndex()  # 0=Sim, 1=Real (_POLICY_SOURCE_*)
        self._recovery_start_btn.setEnabled(False)
        self._recovery_stop_btn.setEnabled(True)
        self._write_recovery(source)
        src_label = "Real" if source == _POLICY_SOURCE_REAL else "Sim"
        self._status_label.setText(f"Recovery: engaged (source={src_label}, kd={recovery_runtime.RECOVERY_KD})")

    def _on_recovery_param_changed(self) -> None:
        self._update_recovery_source(self._recovery_source_combo.currentIndex())

    def _on_recovery_stop_clicked(self) -> None:
        # 현재 목표 자세로 홀드(정지 시 target 이 튀지 않게) — sine/policy stop 과 동일 관례.
        # kd 는 recovery 값(1.0)을 유지한다: 기립 직후 게인을 절반으로 떨어뜨리면 그대로 주저앉는다.
        # 이후 Default/Stand Up/Policy 중 아무 버튼이나 누르면 기본 kd(0.5)로 돌아간다.
        pose = self._read_current_target()
        self._stop_recovery()
        self._stop_pedi()
        self._write_hold(pose, kd=recovery_runtime.RECOVERY_KD)
        self._status_label.setText("Recovery stopped (kd stays 1.0 — any other button returns it to 0.5)")

    def _on_pedi_start_clicked(self) -> None:
        # Recovery 와 같이 즉시 engage 한다. Policy 처럼 DEFAULT_POSE 로 먼저 보간하지 않는 이유는
        # 이 정책의 명령 기준(nominal)이 **시작 시점의 실제 4족 자세**라서다 — 보간 도중 잡으면
        # 엉뚱한 자세가 기준이 된다. 대신 publisher 가 발 깊이 0.20~0.40 m 게이트로 4족 여부를
        # 확인하고, 아니면 정책을 구동하지 않고 fallback-hold 를 유지한다.
        self._stop_sine()
        self._stop_policy()
        self._stop_recovery()
        source = self._pedi_source_combo.currentIndex()  # 0=Sim, 1=Real (_POLICY_SOURCE_*)
        manip_leg = self._pedi_leg_combo.currentIndex()
        offset = self._pedi_offset()
        self._pedi_start_btn.setEnabled(False)
        self._pedi_stop_btn.setEnabled(True)
        self._write_pedipulation(source, manip_leg, offset)
        src_label = "Real" if source == _POLICY_SOURCE_REAL else "Sim"
        leg_label = pedipulation_runtime.LEG_NAMES[manip_leg]
        self._status_label.setText(
            f"Pedipulation: engaged (source={src_label}, leg={leg_label}, "
            f"offset=({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}) m)"
        )

    def _on_pedi_param_changed(self) -> None:
        self._update_pedi_cmd(
            self._pedi_source_combo.currentIndex(),
            self._pedi_leg_combo.currentIndex(),
            self._pedi_offset(),
        )

    def _on_pedi_stop_clicked(self) -> None:
        # **stand 자세로 보간**한다 — sine/policy stop 과 같은 관례다. 조작 다리를 든 채로
        # 멈추면 3족으로 굳어 다음 명령 전에 사람이 세워야 한다. kd 는 기본값(0.5) 그대로라
        # recovery 처럼 되돌릴 것이 없다.
        self._stop_pedi()
        self._play_sequence_to(motions.DEFAULT_POSE, SEQUENCE_DURATION_S)
        self._status_label.setText("Pedipulation stopped — returning to the stand pose")

    # -- continuous publisher --

    def _on_startup_tick(self) -> None:
        """gui 기동 시 로봇 현재 자세를 획득해 시작 자세로 보간(스냅 방지). 1회만 수행.

        첫 /lowstate 로 실측 관절각을 잡으면 그 자세를 먼저 홀드(무동작)한 뒤 STAND_FOLDED 로
        느리게 보간한다. 지정 시간 안에 상태를 못 받으면(sim/robot 미기동 등) 실측 없이 진행한다.
        사용자가 그 사이 버튼을 눌러 이미 발행이 시작됐으면(_latest_pose 존재) 관여하지 않는다.
        """
        if self._startup_done:
            self._startup_timer.stop()
            return
        if self._latest_pose is not None:  # 버튼이 이미 모션을 시작함 — startup 개입 안 함
            self._startup_done = True
            self._startup_timer.stop()
            return
        # publisher 프로세스가 /lowstate에서 써준 실측 관절각을 공유 메모리에서 읽는다.
        measured = None
        with self._shared.get_lock():
            if self._shared[_SM_MEASURED_VALID] >= 0.5:
                measured = [self._shared[_SM_MEASURED_Q + i] for i in range(_NUM)]
        if measured is not None:
            start = list(measured)
            status = "Startup: easing from the robot's current pose to the start pose..."
        elif time.monotonic() >= self._startup_deadline:
            start = list(motions.STAND_FOLDED)  # /lowstate 미수신 — 실측 없이 시작 자세로
            status = "Holding the start pose (current pose not acquired — check sim/robot state)"
        else:
            return  # 아직 첫 상태 대기
        # 현재 자세(실측)를 보간 시작점으로 두고 STAND_FOLDED 로 시퀀스 재생 — 첫 프레임이 실측이라 스냅 없음.
        self._current_pose = list(start)
        self._startup_done = True
        self._startup_timer.stop()
        self._status_label.setText(status)
        self._play_sequence_to(motions.STAND_FOLDED, STARTUP_DURATION_S)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override signature)
        self._startup_timer.stop()
        self._cancel_policy_engage()
        self._ctrl_sock.close()
        # monitor 자식 프로세스가 살아있으면 정리 (orphan 방지). publisher 프로세스는 main()이 정리한다.
        if self._monitor_proc is not None and self._monitor_proc.poll() is None:
            self._monitor_proc.terminate()
        super().closeEvent(event)


def main() -> None:
    # 공유 메모리 + 발행 stop 플래그를 만들고, **QApplication 생성 전에** publisher 프로세스를
    # fork 한다 — 이 시점엔 rclpy.init()도 QApplication도 없어 자식이 깨끗한 상태에서 시작한다
    # (자식만 rclpy.init(); 부모=UI는 rclpy를 init하지 않는다). fork 후 자식은 Qt를 전혀 만지지 않는다.
    mp.set_start_method("fork", force=True)
    shared = mp.Array("d", _SM_LEN)  # 0으로 초기화 → CMD_VALID=0(발행 보류), MEASURED_VALID=0
    stop_flag = mp.Value("i", 0)
    publisher = mp.Process(target=publisher_process_main, args=(shared, stop_flag), daemon=True)
    publisher.start()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(_STYLESHEET)
    window = MainWindow(shared)
    window.show()

    exit_code = app.exec_()

    # publisher 프로세스 정리 — stop 플래그 → join → 안 죽으면 terminate.
    stop_flag.value = 1
    publisher.join(timeout=2.0)
    if publisher.is_alive():
        publisher.terminate()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
