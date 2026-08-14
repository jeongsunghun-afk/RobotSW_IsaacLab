# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg UDP 패킷 스키마 (gui <-> sim_runner <-> monitor 경계).

순수 stdlib(struct)만 사용 — Isaac conda Python 3.12 와 시스템 Python 3.10 양쪽에서
임포트 가능해야 한다. numpy/torch/ros 의존성 금지.

전송(현재, ROS2 미정): 순수 UDP.
    gui ──cmd(CMD_PORT)──▶ sim_runner ──state(STATE_PORT)──▶ gui
    gui ──monitor(MONITOR_PORT)──▶ monitor (별도 프로세스; action+sim 한 패킷, time-aligned)

8-DOF 2족 다리(HL 4관절 + HR 4관절, leg-major 순서)이며 IMU/base 상태는 다루지 않는다.
"""

from __future__ import annotations

import struct

NUM_JOINTS: int = 8  # leg-major: HL_{hip,thigh,calf,foot} → HR_{hip,thigh,calf,foot}

# 포트 — 기존 리그와 전부 겹치지 않는 별도 대역.
#   9871/9872 = r2s_go2 live, 9875/9876 = r2s_go2 PACE tuner, 9873/9874/9875 = r2s_hind_leg
CMD_PORT: int = 9881  # gui -> sim_runner
STATE_PORT: int = 9882  # sim_runner -> gui
MONITOR_PORT: int = 9883  # gui(relay) -> monitor

# --- policy mode 전용 포트 (position-control 대역과 겹치지 않음) ---
#   gui ──POLICY_CMD(9884)──▶ policy_runner ──POLICY_ACT(9886)──▶ sim_runner
#                                          ◀──POLICY_STATE(9885)──┘
#   policy_runner ──REAL_ACT(9887)──▶ real ──REAL_STATE(9888)──▶ policy_runner (seam)
POLICY_CMD_PORT: int = 9884  # gui -> policy_runner (mode/source/x_vel/yaw)
POLICY_STATE_PORT: int = 9885  # sim_runner -> policy_runner (rich state: q,dq,gravity)
POLICY_ACT_PORT: int = 9886  # policy_runner -> sim_runner (target q)
REAL_ACT_PORT: int = 9887  # policy_runner -> real endpoint (seam)
REAL_STATE_PORT: int = 9888  # real endpoint -> policy_runner (seam)
REAL_TELEM_PORT: int = 9889  # real endpoint -> gui monitor (q/dq/tau/rpy, 관측 전용)
REAL_MON_PORT: int = 9890  # gui monitor(relay) -> monitor.py plot (MON 패킷, leg-major 재배열)

# magic — r2s_hind_leg("R2HC"/"R2HS"/"R2MN")와 반드시 다른 값을 쓴다. 두 리그를 동시에 띄웠을 때
# 다른 리그로 잘못 보낸 패킷이 (관절 수가 달라 크기로도 걸러지지만) magic 단계에서 확실히 거부되어
# 절대 오파싱될 수 없게 하기 위함이다.
CMD_MAGIC: int = 0x52324243  # "R2BC"
STATE_MAGIC: int = 0x52324253  # "R2BS"
MONITOR_MAGIC: int = 0x5232424D  # "R2BM"
IEFF_MAGIC: int = 0x52324249  # "R2BI" — sim_runner -> gui, 관절별 유효 관성(STATE_PORT 공유, 저빈도)

# policy mode magic — 위 3개와 반드시 다른 값.
POLICY_CMD_MAGIC: int = 0x52325043  # "R2PC" gui -> policy_runner
POLICY_STATE_MAGIC: int = 0x52325053  # "R2PS" sim_runner -> policy_runner
POLICY_ACT_MAGIC: int = 0x52325041  # "R2PA" policy_runner -> sim_runner (and real)
POLICY_TELEM_MAGIC: int = 0x52325054  # "R2PT" real -> gui monitor (REAL_TELEM_PORT)
POLICY_PING_MAGIC: int = 0x52325047  # "R2PG" monitor -> real: peer 등록만 (목표 없음, 상태 불변)
POLICY_RELAX_MAGIC: int = 0x5232504C  # "R2PL" gui -> real: 무토크(limp) 요청 — kp=kd=tau=0 능동 송신
POLICY_GAIN_MAGIC: int = 0x5232504B  # "R2PK" gui -> real: kp/kd 런타임 갱신 (실기팀 2026-08-12 추가)

# 명령: magic(I) seq(I) + NUM_JOINTS x (q,dq,kp,kd,tau) f
_CMD_FMT: str = "<II" + "5f" * NUM_JOINTS
CMD_SIZE: int = struct.calcsize(_CMD_FMT)

# 상태: magic(I) seq(I) sim_time(f) + NUM_JOINTS x (q,dq,ddq,tau_est) f
_STATE_FMT: str = "<IIf" + "4f" * NUM_JOINTS  # codespell:ignore
STATE_SIZE: int = struct.calcsize(_STATE_FMT)

# 모니터: magic(I) seq(I) + NUM_JOINTS x action_q + NUM_JOINTS x (sim_q, sim_dq, sim_tau)
_MON_FMT: str = "<II" + "f" * NUM_JOINTS + "3f" * NUM_JOINTS
MON_SIZE: int = struct.calcsize(_MON_FMT)

# I_eff: magic(I) seq(I) + NUM_JOINTS x ieff — sim_runner가 STATE_PORT로 1Hz 전송.
# STATE 패킷(140B)과 크기가 달라 크기+magic 이중으로 갈린다.
_IEFF_FMT: str = "<II" + "f" * NUM_JOINTS
IEFF_SIZE: int = struct.calcsize(_IEFF_FMT)


# ---------------------------------------------------------------------------
# 명령 (gui -> sim_runner)
# ---------------------------------------------------------------------------


def pack_cmd(seq: int, q, dq, kp, kd, tau) -> bytes:
    """8-관절 PD 명령을 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        q: 관절 목표각 [rad], 길이 8.
        dq: 관절 목표각속도 [rad/s], 길이 8.
        kp: 위치 게인, 길이 8.
        kd: 속도 게인, 길이 8.
        tau: 피드포워드 토크 [Nm], 길이 8.

    Returns:
        CMD_SIZE 바이트 패킷.
    """
    vals: list[float] = []
    for i in range(NUM_JOINTS):
        vals.extend((float(q[i]), float(dq[i]), float(kp[i]), float(kd[i]), float(tau[i])))
    return struct.pack(_CMD_FMT, CMD_MAGIC, seq & 0xFFFFFFFF, *vals)


def unpack_cmd(data: bytes) -> dict | None:
    """명령 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``q``, ``dq``, ``kp``, ``kd``, ``tau`` (각 리스트 길이 8).
    """
    if len(data) != CMD_SIZE:
        return None
    fields = struct.unpack(_CMD_FMT, data)
    if fields[0] != CMD_MAGIC:
        return None
    seq = fields[1]
    body = fields[2:]
    q, dq, kp, kd, tau = [], [], [], [], []
    for i in range(NUM_JOINTS):
        base = i * 5
        q.append(body[base + 0])
        dq.append(body[base + 1])
        kp.append(body[base + 2])
        kd.append(body[base + 3])
        tau.append(body[base + 4])
    return {"seq": seq, "q": q, "dq": dq, "kp": kp, "kd": kd, "tau": tau}


# ---------------------------------------------------------------------------
# 상태 (sim_runner -> gui)
# ---------------------------------------------------------------------------


def pack_state(seq: int, sim_time: float, q, dq, ddq, tau_est) -> bytes:
    """8-관절 상태를 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        sim_time: 시뮬레이션 시간 [s].
        q: 관절각 [rad], 길이 8.
        dq: 관절각속도 [rad/s], 길이 8.
        ddq: 관절각가속도 [rad/s^2], 길이 8.
        tau_est: 추정 토크 [Nm], 길이 8.

    Returns:
        STATE_SIZE 바이트 패킷.
    """
    vals: list[float] = []
    for i in range(NUM_JOINTS):
        vals.extend((float(q[i]), float(dq[i]), float(ddq[i]), float(tau_est[i])))
    return struct.pack(_STATE_FMT, STATE_MAGIC, seq & 0xFFFFFFFF, float(sim_time), *vals)


def unpack_state(data: bytes) -> dict | None:
    """상태 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``sim_time``, ``q``, ``dq``, ``ddq``, ``tau_est`` (각 길이 8).
    """
    if len(data) != STATE_SIZE:
        return None
    fields = struct.unpack(_STATE_FMT, data)
    if fields[0] != STATE_MAGIC:
        return None
    seq = fields[1]
    sim_time = fields[2]
    body = fields[3:]
    q, dq, ddq, tau_est = [], [], [], []
    for i in range(NUM_JOINTS):
        base = i * 4
        q.append(body[base + 0])
        dq.append(body[base + 1])
        ddq.append(body[base + 2])
        tau_est.append(body[base + 3])
    return {"seq": seq, "sim_time": sim_time, "q": q, "dq": dq, "ddq": ddq, "tau_est": tau_est}


# ---------------------------------------------------------------------------
# 모니터 relay (gui -> monitor): action + sim 한 패킷 (time-aligned)
# ---------------------------------------------------------------------------


def pack_monitor(seq: int, action_q, sim_q, sim_dq, sim_tau) -> bytes:
    """monitor용 action+sim 결합 패킷 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        action_q: 명령 관절각 [rad], 길이 8.
        sim_q: sim 관절각 [rad], 길이 8.
        sim_dq: sim 관절각속도 [rad/s], 길이 8.
        sim_tau: sim 추정 토크 [Nm], 길이 8.

    Returns:
        MON_SIZE 바이트 패킷.
    """
    vals = [float(action_q[i]) for i in range(NUM_JOINTS)]
    for i in range(NUM_JOINTS):
        vals.extend((float(sim_q[i]), float(sim_dq[i]), float(sim_tau[i])))
    return struct.pack(_MON_FMT, MONITOR_MAGIC, seq & 0xFFFFFFFF, *vals)


def unpack_monitor(data: bytes) -> dict | None:
    """monitor 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``action_q``, ``sim_q``, ``sim_dq``, ``sim_tau`` (각 길이 8).
    """
    if len(data) != MON_SIZE:
        return None
    fields = struct.unpack(_MON_FMT, data)
    if fields[0] != MONITOR_MAGIC:
        return None
    seq = fields[1]
    body = fields[2:]
    action_q = list(body[:NUM_JOINTS])
    rest = body[NUM_JOINTS:]
    sim_q, sim_dq, sim_tau = [], [], []
    for i in range(NUM_JOINTS):
        base = i * 3
        sim_q.append(rest[base + 0])
        sim_dq.append(rest[base + 1])
        sim_tau.append(rest[base + 2])
    return {"seq": seq, "action_q": action_q, "sim_q": sim_q, "sim_dq": sim_dq, "sim_tau": sim_tau}


# ---------------------------------------------------------------------------
# I_eff (sim_runner -> gui): 관절별 유효 관성 — GUI의 계산 게인(kp=I·ωn², kd=2ζ·I·ωn)용
# ---------------------------------------------------------------------------


def pack_ieff(seq: int, ieff) -> bytes:
    """관절별 유효 관성(generalized mass matrix 대각)을 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        ieff: 관절별 유효 관성 [kg·m²], 길이 8, leg-major 순서.

    Returns:
        IEFF_SIZE 바이트 패킷.
    """
    vals = [float(ieff[i]) for i in range(NUM_JOINTS)]
    return struct.pack(_IEFF_FMT, IEFF_MAGIC, seq & 0xFFFFFFFF, *vals)


def unpack_ieff(data: bytes) -> dict | None:
    """I_eff 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``ieff`` (리스트 길이 8, leg-major 순서).
    """
    if len(data) != IEFF_SIZE:
        return None
    fields = struct.unpack(_IEFF_FMT, data)
    if fields[0] != IEFF_MAGIC:
        return None
    return {"seq": fields[1], "ieff": list(fields[2:])}


# ---------------------------------------------------------------------------
# policy mode: command (gui -> policy_runner)
# ---------------------------------------------------------------------------

# magic(I) seq(I) mode(i) source(i) x_vel(f) yaw(f)
#   mode:   0=idle(정책 정지), 1=run
#   source: 0=sim(sim state로 폐루프), 1=real(real state로 폐루프)
_POLICY_CMD_FMT: str = "<IIiiff"
POLICY_CMD_SIZE: int = struct.calcsize(_POLICY_CMD_FMT)


def pack_policy_cmd(seq: int, mode: int, source: int, x_vel: float, yaw: float) -> bytes:
    """gui -> policy_runner 제어 명령 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        mode: 0=idle(정책 정지·중립 유지), 1=run.
        source: 폐루프 input source. 0=sim, 1=real.
        x_vel: 전진 속도 명령 [m/s]. 학습 범위 [-0.5, 2.0].
        yaw: 요 각속도 명령 [rad/s]. 학습 범위 [-0.5, 0.5].

    Returns:
        POLICY_CMD_SIZE 바이트 패킷.
    """
    return struct.pack(
        _POLICY_CMD_FMT, POLICY_CMD_MAGIC, seq & 0xFFFFFFFF, int(mode), int(source), float(x_vel), float(yaw)
    )


def unpack_policy_cmd(data: bytes) -> dict | None:
    """policy 명령 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``mode``, ``source``, ``x_vel``, ``yaw``.
    """
    if len(data) != POLICY_CMD_SIZE:
        return None
    magic, seq, mode, source, x_vel, yaw = struct.unpack(_POLICY_CMD_FMT, data)
    if magic != POLICY_CMD_MAGIC:
        return None
    return {"seq": seq, "mode": mode, "source": source, "x_vel": x_vel, "yaw": yaw}


# ---------------------------------------------------------------------------
# policy mode: rich state (sim_runner/real -> policy_runner)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ★좌표 규약 버전 — 송신자가 **자기 규약을 스스로 알린다** (C++ 원본: real_runner/r2s_packets.hpp:40)
# ---------------------------------------------------------------------------
#
#   0 = (구) gear 미적용 + foot **raw**각(q_foot + q_calf) 보고/수신.
#       이 값을 명시적으로 보내는 송신자는 없다 — 필드가 없는 84 B STATE / 120 B TELEM 이 곧 버전 0.
#   1 = gear 적용(위치·속도) + foot **관절**각 보고/수신. (2026-08-14)
#       ⚠ TELEM 의 ``tau`` 만은 변환 없이 통과한다 — **채널기준**이라 소비자가 ``× gear`` 하면
#       관절토크다(calf 1.5 · foot 1.2). 브리지가 그 변환까지 하게 되면 버전 2.
#
# 즉 버전은 "무엇이 판정됐나"가 아니라 **"송신자가 무엇을 하는가"** 를 가리킨다.
# ⚠ 이 상수는 C++ ``R2S_CONVENTION_VERSION`` 과 값 일치 필수 (다른 언어라 중복 정의).
R2S_CONVENTION_VERSION: int = 1

# magic(I) seq(I) + q(8f) + dq(8f) + gravity(3f) + convention_version(B)
#   — **articulation 순서** (재매핑 없음). 84 → 85 B, offset 0~83 은 불변.
# ⚠ STATE 에는 **legacy(84 B) 분기를 두지 않는다** — 하드 컷오버다. TELEM 과 다르게 가는 이유:
#   TELEM 은 관측 전용이라 구버전을 읽어도 사람이 눈으로 보고 끝이지만, STATE 는 **정책 입력**이라
#   규약 미상 패킷이 조용히 흘러들면 프레임 불일치가 그대로 로봇 명령이 된다. 84 B 를 살려두면
#   "규약을 모르는 상태"가 계속 유지되므로, 크기 불일치로 **거부**해 문제를 즉시 드러낸다.
_POLICY_STATE_FMT: str = "<II" + "f" * NUM_JOINTS + "f" * NUM_JOINTS + "3f" + "B"
POLICY_STATE_SIZE: int = struct.calcsize(_POLICY_STATE_FMT)


def pack_policy_state(seq: int, q, dq, gravity, convention_version: int = R2S_CONVENTION_VERSION) -> bytes:
    """policy_runner용 rich state 직렬화 (articulation 순서).

    Args:
        seq: 증가 시퀀스 번호.
        q: 관절각 [rad], 길이 8, articulation 순서.
        dq: 관절각속도 [rad/s], 길이 8, articulation 순서.
        gravity: projected_gravity_b (base frame 중력 단위벡터), 길이 3.
        convention_version: 송신자가 신고하는 좌표 규약 (:data:`R2S_CONVENTION_VERSION` 참조).
            **자기가 실제로 하는 일**을 실을 것 — 기본값을 그대로 쓰면 거짓 신고가 될 수 있다.

    Returns:
        POLICY_STATE_SIZE 바이트 패킷.
    """
    vals = [float(q[i]) for i in range(NUM_JOINTS)]
    vals += [float(dq[i]) for i in range(NUM_JOINTS)]
    vals += [float(gravity[i]) for i in range(3)]
    return struct.pack(_POLICY_STATE_FMT, POLICY_STATE_MAGIC, seq & 0xFFFFFFFF, *vals, int(convention_version) & 0xFF)


def unpack_policy_state(data: bytes) -> dict | None:
    """policy rich state 역직렬화. magic 불일치/크기 오류 시 None.

    ⚠ **legacy(84 B) 분기 없음** — 구버전 STATE 는 크기 불일치로 ``None`` 이 된다(의도).
    사유는 :data:`R2S_CONVENTION_VERSION` 위 주석 참조.

    Returns:
        키: ``seq``, ``q`` (8), ``dq`` (8), ``gravity`` (3), ``convention_version``.
        각도/속도는 전부 articulation 순서.
    """
    if len(data) != POLICY_STATE_SIZE:
        return None
    fields = struct.unpack(_POLICY_STATE_FMT, data)
    if fields[0] != POLICY_STATE_MAGIC:
        return None
    seq = fields[1]
    body = fields[2:]
    q = list(body[:NUM_JOINTS])
    dq = list(body[NUM_JOINTS : 2 * NUM_JOINTS])
    gravity = list(body[2 * NUM_JOINTS : 2 * NUM_JOINTS + 3])
    return {"seq": seq, "q": q, "dq": dq, "gravity": gravity, "convention_version": int(body[-1])}


# ---------------------------------------------------------------------------
# policy mode: action (policy_runner -> sim_runner / real)
# ---------------------------------------------------------------------------

# magic(I) seq(I) + target_q(8f)  — **articulation 순서**
_POLICY_ACT_FMT: str = "<II" + "f" * NUM_JOINTS
POLICY_ACT_SIZE: int = struct.calcsize(_POLICY_ACT_FMT)


def pack_policy_act(seq: int, target_q) -> bytes:
    """policy_runner -> sim_runner/real 관절 목표각 직렬화 (articulation 순서).

    Args:
        seq: 증가 시퀀스 번호.
        target_q: 관절 목표각 [rad], 길이 8, articulation 순서.
            (= action_scale * raw_action + default_joint_pos)

    Returns:
        POLICY_ACT_SIZE 바이트 패킷.
    """
    vals = [float(target_q[i]) for i in range(NUM_JOINTS)]
    return struct.pack(_POLICY_ACT_FMT, POLICY_ACT_MAGIC, seq & 0xFFFFFFFF, *vals)


def unpack_policy_act(data: bytes) -> dict | None:
    """policy action 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``target_q`` (8, articulation 순서).
    """
    if len(data) != POLICY_ACT_SIZE:
        return None
    fields = struct.unpack(_POLICY_ACT_FMT, data)
    if fields[0] != POLICY_ACT_MAGIC:
        return None
    return {"seq": fields[1], "target_q": list(fields[2:])}


# ---------------------------------------------------------------------------
# policy mode: telemetry (real -> gui monitor) / ping (monitor -> real)
# ---------------------------------------------------------------------------

# magic(I) seq(I) valid_mask(I) + q(8f) + dq(8f) + tau(8f) + rpy(3f)  — **articulation 순서**, sim 좌표
# valid_mask: bit p(0~7)=관절 p 상태 유효, bit 8=IMU 수신됨. warmup 전에도 송신되므로 mask 로 구분.
_POLICY_TELEM_FMT: str = "<III" + "f" * NUM_JOINTS * 3 + "3f" + "B"
POLICY_TELEM_SIZE: int = struct.calcsize(_POLICY_TELEM_FMT)
# legacy 하위호환 — TELEM 은 **관측 전용**이라 구버전을 읽어도 사람이 눈으로 보고 끝이므로
# 살려 둔다(STATE 와 다르게 가는 이유는 R2S_CONVENTION_VERSION 위 주석 참조). 둘 다 규약 버전 0.
#   v2(2026-08-13, mask 있음 120 B)
_POLICY_TELEM_V2_FMT: str = "<III" + "f" * NUM_JOINTS * 3 + "3f"
_POLICY_TELEM_V2_SIZE: int = struct.calcsize(_POLICY_TELEM_V2_FMT)
#   v1(2026-08-10, mask 없음 116 B)
_POLICY_TELEM_V1_FMT: str = "<II" + "f" * NUM_JOINTS * 3 + "3f"
_POLICY_TELEM_V1_SIZE: int = struct.calcsize(_POLICY_TELEM_V1_FMT)

# magic(I) seq(I) — 페이로드 없음
_POLICY_PING_FMT: str = "<II"
POLICY_PING_SIZE: int = struct.calcsize(_POLICY_PING_FMT)


def pack_policy_telem(
    seq: int, q, dq, tau, rpy, valid_mask: int = 0x1FF, convention_version: int = R2S_CONVENTION_VERSION
) -> bytes:
    """real 엔드포인트 -> gui monitor 텔레메트리 직렬화 (articulation 순서).

    Args:
        seq: 마지막 ACT seq (peer 없이 자생 seq여도 무방 — 관측 전용).
        q: 관절각 [rad], 길이 8.
        dq: 관절각속도 [rad/s], 길이 8.
        tau: 관절 토크 [N·m], 길이 8 (모터 fTorque에 sign 적용).
        rpy: IMU roll/pitch/yaw 원값 [deg], 길이 3.
        valid_mask: bit p(0~7)=관절 p 유효, bit 8=IMU 수신 (기본 전부 유효).
            ⚠ 버전 1부터 foot(p=6,7) 비트는 "foot **과 그 calf 가 둘 다** 유효"를 뜻한다 —
            foot 관절각을 내려면 같은 다리 calf 가 필요하므로.
        convention_version: 송신자가 신고하는 좌표 규약 (:data:`R2S_CONVENTION_VERSION` 참조).

    Returns:
        POLICY_TELEM_SIZE 바이트 패킷.
    """
    vals = [float(q[i]) for i in range(NUM_JOINTS)]
    vals += [float(dq[i]) for i in range(NUM_JOINTS)]
    vals += [float(tau[i]) for i in range(NUM_JOINTS)]
    vals += [float(rpy[i]) for i in range(3)]
    return struct.pack(
        _POLICY_TELEM_FMT,
        POLICY_TELEM_MAGIC,
        seq & 0xFFFFFFFF,
        valid_mask & 0xFFFFFFFF,
        *vals,
        int(convention_version) & 0xFF,
    )


def unpack_policy_telem(data: bytes) -> dict | None:
    """텔레메트리 역직렬화. magic 불일치/크기 오류 시 None.

    legacy 도 수용한다 — v2(120 B, 규약 필드 없음) · v1(116 B, mask 도 없음). 둘 다 **규약 버전 0**
    으로 채워진다: 그 시절 브리지는 gear 미적용 + foot raw 였고, 필드가 없다는 것 자체가 버전 0 의
    서명이다.

    Returns:
        키: ``seq``, ``valid_mask``, ``q`` (8), ``dq`` (8), ``tau`` (8), ``rpy`` (3),
        ``convention_version``. v1 패킷은 ``valid_mask=0x1FF`` (전부 유효 가정)로 채워진다.
    """
    if len(data) == POLICY_TELEM_SIZE:
        fields = struct.unpack(_POLICY_TELEM_FMT, data)
        mask = fields[2]
        body = fields[3:-1]
        version = int(fields[-1])
    elif len(data) == _POLICY_TELEM_V2_SIZE:
        fields = struct.unpack(_POLICY_TELEM_V2_FMT, data)
        mask = fields[2]
        body = fields[3:]
        version = 0
    elif len(data) == _POLICY_TELEM_V1_SIZE:
        fields = struct.unpack(_POLICY_TELEM_V1_FMT, data)
        mask = 0x1FF
        body = fields[2:]
        version = 0
    else:
        return None
    if fields[0] != POLICY_TELEM_MAGIC:
        return None
    return {
        "seq": fields[1],
        "valid_mask": mask,
        "q": list(body[:NUM_JOINTS]),
        "dq": list(body[NUM_JOINTS : 2 * NUM_JOINTS]),
        "tau": list(body[2 * NUM_JOINTS : 3 * NUM_JOINTS]),
        "rpy": list(body[3 * NUM_JOINTS : 3 * NUM_JOINTS + 3]),
        "convention_version": version,
    }


def pack_policy_ping(seq: int) -> bytes:
    """monitor -> real keepalive 직렬화. real_runner는 peer 등록만 하고 상태를 바꾸지 않는다."""
    return struct.pack(_POLICY_PING_FMT, POLICY_PING_MAGIC, seq & 0xFFFFFFFF)


def pack_policy_relax(seq: int) -> bytes:
    """gui -> real 무토크(limp) 요청 직렬화 (PING과 같은 8B, magic만 다름).

    real_runner(bridge 모드)는 RELAX 상태로 전환해 kp=kd=tau=0 명령을 능동 송신한다 — 명령을
    끊는 게 아니라 zero-torque를 계속 보내는 것이 확실한 limp다. 이후 ACT를 받으면 현재 자세를
    재래치하고 ENGAGE 램프로 복귀한다. probe/hold 모드에서는 무시된다.
    """
    return struct.pack(_POLICY_PING_FMT, POLICY_RELAX_MAGIC, seq & 0xFFFFFFFF)


def unpack_policy_relax(data: bytes) -> dict | None:
    """relax 역직렬화. magic 불일치/크기 오류 시 None. 키: ``seq``."""
    if len(data) != POLICY_PING_SIZE:
        return None
    fields = struct.unpack(_POLICY_PING_FMT, data)
    if fields[0] != POLICY_RELAX_MAGIC:
        return None
    return {"seq": fields[1]}


# magic(I) seq(I) + kp(8f) + kd(8f) — **articulation 순서** (real_runner가 모터 순서로 매핑·클램프)
_POLICY_GAIN_FMT: str = "<II" + "f" * NUM_JOINTS * 2
POLICY_GAIN_SIZE: int = struct.calcsize(_POLICY_GAIN_FMT)


def pack_policy_gain(seq: int, kp, kd) -> bytes:
    """gui -> real kp/kd 런타임 갱신 직렬화 (articulation 순서).

    real_runner는 수신 시 모터 순서로 매핑하고 드라이버 상한(kp≤500, kd≤5)으로 클램프해
    다음 틱부터 반영한다. 목표각·상태머신은 불변이며 relax 중엔 여전히 0이 강제된다.
    """
    vals = [float(kp[i]) for i in range(NUM_JOINTS)]
    vals += [float(kd[i]) for i in range(NUM_JOINTS)]
    return struct.pack(_POLICY_GAIN_FMT, POLICY_GAIN_MAGIC, seq & 0xFFFFFFFF, *vals)


def unpack_policy_gain(data: bytes) -> dict | None:
    """gain 역직렬화. magic 불일치/크기 오류 시 None. 키: ``seq``, ``kp`` (8), ``kd`` (8)."""
    if len(data) != POLICY_GAIN_SIZE:
        return None
    fields = struct.unpack(_POLICY_GAIN_FMT, data)
    if fields[0] != POLICY_GAIN_MAGIC:
        return None
    return {"seq": fields[1], "kp": list(fields[2 : 2 + NUM_JOINTS]), "kd": list(fields[2 + NUM_JOINTS :])}


def unpack_policy_ping(data: bytes) -> dict | None:
    """ping 역직렬화. magic 불일치/크기 오류 시 None. 키: ``seq``."""
    if len(data) != POLICY_PING_SIZE:
        return None
    fields = struct.unpack(_POLICY_PING_FMT, data)
    if fields[0] != POLICY_PING_MAGIC:
        return None
    return {"seq": fields[1]}


# ---------------------------------------------------------------------------
# PLANT — gui -> sim_runner: sim 플랜트 파라미터 선택 적용 (PACE 식별값 vs 스톡 cfg)
# ---------------------------------------------------------------------------

PLANT_MAGIC: int = 0x52325050  # "R2PP" — gui -> sim (CMD_PORT 공유, one-shot)
# magic(I) seq(I) mode(I) + armature(8f)+viscous(8f)+coulomb(8f) — **leg-major** 순서.
# mode 0 = stock cfg 복원(배열 무시), 1 = 실린 값 적용.
_PLANT_FMT: str = "<III" + "f" * NUM_JOINTS * 3
PLANT_SIZE: int = struct.calcsize(_PLANT_FMT)


def pack_plant(seq: int, mode: int, armature, viscous, coulomb) -> bytes:
    """gui -> sim 플랜트 파라미터 직렬화 (leg-major 순서).

    Args:
        seq: 시퀀스 번호.
        mode: 0=stock cfg 복원(배열 무시), 1=armature/viscous/coulomb 적용.
        armature: 관절별 armature [kg·m²], 길이 8.
        viscous: 관절별 점성 마찰 [N·m·s/rad], 길이 8.
        coulomb: 관절별 Coulomb 마찰 [N·m], 길이 8.
    """
    vals = [float(armature[i]) for i in range(NUM_JOINTS)]
    vals += [float(viscous[i]) for i in range(NUM_JOINTS)]
    vals += [float(coulomb[i]) for i in range(NUM_JOINTS)]
    return struct.pack(_PLANT_FMT, PLANT_MAGIC, seq & 0xFFFFFFFF, int(mode), *vals)


def unpack_plant(data: bytes) -> dict | None:
    """plant 역직렬화. magic 불일치/크기 오류 시 None.

    키: ``seq``, ``mode``, ``armature`` (8), ``viscous`` (8), ``coulomb`` (8).
    """
    if len(data) != PLANT_SIZE:
        return None
    fields = struct.unpack(_PLANT_FMT, data)
    if fields[0] != PLANT_MAGIC:
        return None
    n = NUM_JOINTS
    return {
        "seq": fields[1],
        "mode": fields[2],
        "armature": list(fields[3 : 3 + n]),
        "viscous": list(fields[3 + n : 3 + 2 * n]),
        "coulomb": list(fields[3 + 2 * n : 3 + 3 * n]),
    }


if __name__ == "__main__":
    z = [0.0] * NUM_JOINTS
    c = pack_cmd(1, z, z, [65.0] * NUM_JOINTS, [6.0] * NUM_JOINTS, z)
    assert len(c) == CMD_SIZE, (len(c), CMD_SIZE)
    assert unpack_cmd(c)["kp"][0] == 65.0
    assert len(unpack_cmd(c)["q"]) == NUM_JOINTS
    s = pack_state(2, 1.5, z, z, z, z)
    assert len(s) == STATE_SIZE, (len(s), STATE_SIZE)
    assert unpack_state(s)["sim_time"] == 1.5
    assert len(unpack_state(s)["tau_est"]) == NUM_JOINTS
    m = pack_monitor(3, [0.1] * NUM_JOINTS, z, z, z)
    assert len(m) == MON_SIZE, (len(m), MON_SIZE)
    assert abs(unpack_monitor(m)["action_q"][0] - 0.1) < 1e-6
    assert len(unpack_monitor(m)["sim_q"]) == NUM_JOINTS
    # 다른 리그(hind_leg) magic이 섞여 들어와도 절대 파싱되지 않는지 확인.
    assert unpack_cmd(struct.pack(_CMD_FMT, 0x52324843, 0, *([0.0] * (5 * NUM_JOINTS)))) is None

    # I_eff 패킷 왕복 + STATE 포트 공유 시 오파싱 없는지 확인.
    ie = pack_ieff(7, [0.31, 0.22, 0.05, 0.01] * 2)
    assert len(ie) == IEFF_SIZE, (len(ie), IEFF_SIZE)
    die = unpack_ieff(ie)
    assert die is not None and abs(die["ieff"][0] - 0.31) < 1e-6 and len(die["ieff"]) == NUM_JOINTS
    assert unpack_state(ie) is None and unpack_ieff(s) is None
    # policy ACT(40B)와 크기가 같지만 magic으로 갈린다 (포트도 다름 — 방어선 2중).
    assert unpack_ieff(pack_policy_act(1, z)) is None

    # policy mode 패킷 왕복.
    pc = pack_policy_cmd(4, 1, 0, 1.5, -0.3)
    assert len(pc) == POLICY_CMD_SIZE, (len(pc), POLICY_CMD_SIZE)
    dpc = unpack_policy_cmd(pc)
    assert dpc["mode"] == 1 and dpc["source"] == 0 and abs(dpc["x_vel"] - 1.5) < 1e-6 and abs(dpc["yaw"] + 0.3) < 1e-6
    g = [0.0, 0.0, -1.0]
    ps = pack_policy_state(5, z, z, g)
    assert len(ps) == POLICY_STATE_SIZE == 85, (len(ps), POLICY_STATE_SIZE)
    dps = unpack_policy_state(ps)
    assert len(dps["q"]) == NUM_JOINTS and len(dps["gravity"]) == 3 and abs(dps["gravity"][2] + 1.0) < 1e-6
    # 규약 버전 왕복 — 기본값과 명시값 양쪽.
    assert dps["convention_version"] == R2S_CONVENTION_VERSION == 1
    assert unpack_policy_state(pack_policy_state(5, z, z, g, convention_version=0))["convention_version"] == 0
    # ★STATE 는 legacy(84 B) 분기가 **없다** — 하드 컷오버(위 주석 참조). 구버전은 거부돼야 한다.
    _legacy_state = struct.pack(
        "<II" + "f" * NUM_JOINTS * 2 + "3f", POLICY_STATE_MAGIC, 5, *([0.0] * (2 * NUM_JOINTS + 3))
    )
    assert len(_legacy_state) == 84
    assert unpack_policy_state(_legacy_state) is None, "84 B STATE 는 거부돼야 한다"
    pa = pack_policy_act(6, [0.2] * NUM_JOINTS)
    assert len(pa) == POLICY_ACT_SIZE, (len(pa), POLICY_ACT_SIZE)
    assert abs(unpack_policy_act(pa)["target_q"][0] - 0.2) < 1e-6
    # policy 패킷에 position-control magic이 섞여도 거부.
    assert unpack_policy_cmd(pack_cmd(1, z, z, z, z, z)) is None
    assert unpack_cmd(pack_policy_act(1, z)) is None

    # telemetry/ping 왕복.
    pt = pack_policy_telem(7, z, z, [1.5] * NUM_JOINTS, [0.5, -0.2, 10.0], valid_mask=0x103)
    assert len(pt) == POLICY_TELEM_SIZE == 121, (len(pt), POLICY_TELEM_SIZE)
    dpt = unpack_policy_telem(pt)
    assert abs(dpt["tau"][0] - 1.5) < 1e-6 and abs(dpt["rpy"][2] - 10.0) < 1e-6 and len(dpt["q"]) == NUM_JOINTS
    assert dpt["valid_mask"] == 0x103
    assert dpt["convention_version"] == R2S_CONVENTION_VERSION == 1
    # legacy v2(120B, 규약 필드 없음) — **버전 0** 으로 채워진다(필드 부재가 곧 버전 0).
    v2 = struct.pack(_POLICY_TELEM_V2_FMT, POLICY_TELEM_MAGIC, 9, 0x1FF, *([0.25] * (3 * NUM_JOINTS)), 0.0, 0.0, 0.0)
    assert len(v2) == 120
    dv2 = unpack_policy_telem(v2)
    assert dv2 is not None and dv2["convention_version"] == 0 and abs(dv2["q"][0] - 0.25) < 1e-6
    # legacy v1(116B, mask 도 없음) — 전부 유효 + 버전 0.
    v1 = struct.pack(_POLICY_TELEM_V1_FMT, POLICY_TELEM_MAGIC, 9, *([0.25] * (3 * NUM_JOINTS)), 0.0, 0.0, 0.0)
    dv1 = unpack_policy_telem(v1)
    assert dv1 is not None and dv1["valid_mask"] == 0x1FF and dv1["convention_version"] == 0
    # STATE(85B)/ACT(40B)와 크기가 달라 크기 단계에서 갈리고, magic으로도 거부.
    assert unpack_policy_state(pt) is None and unpack_policy_telem(ps) is None
    # ★크기 충돌 금지 — 이 파서들은 길이로 분기하므로 두 패킷이 같은 크기면 조용히 오파싱된다.
    #   (84→85, 120→121 로 옮겼으니 새로 겹친 게 없는지 매번 확인한다.)
    _sizes = {
        "CMD": CMD_SIZE,
        "STATE": STATE_SIZE,
        "MON": MON_SIZE,
        "IEFF": IEFF_SIZE,
        "PLANT": PLANT_SIZE,
        "POLICY_CMD": POLICY_CMD_SIZE,
        "POLICY_STATE": POLICY_STATE_SIZE,
        "POLICY_ACT": POLICY_ACT_SIZE,
        "POLICY_TELEM": POLICY_TELEM_SIZE,
        "POLICY_PING": POLICY_PING_SIZE,
        "POLICY_GAIN": POLICY_GAIN_SIZE,
        "TELEM_v2(legacy)": _POLICY_TELEM_V2_SIZE,
        "TELEM_v1(legacy)": _POLICY_TELEM_V1_SIZE,
    }
    # IEFF(40)와 POLICY_ACT(40)는 **의도된** 동일 크기 — magic + 포트로 갈린다(위에서 확인함).
    _dupes = [
        (a, b)
        for i, (a, sa) in enumerate(_sizes.items())
        for b, sb in list(_sizes.items())[i + 1 :]
        if sa == sb and {a, b} != {"IEFF", "POLICY_ACT"}
    ]
    assert not _dupes, f"패킷 크기 충돌: {_dupes} — 길이 분기가 오파싱된다"
    pg = pack_policy_ping(8)
    assert len(pg) == POLICY_PING_SIZE == 8
    assert unpack_policy_ping(pg)["seq"] == 8
    assert unpack_policy_act(pg) is None and unpack_policy_ping(pa) is None
    # relax: PING과 크기 동일, magic으로만 구분 — 상호 오파싱 불가 확인.
    pr = pack_policy_relax(9)
    assert unpack_policy_relax(pr)["seq"] == 9
    assert unpack_policy_ping(pr) is None and unpack_policy_relax(pg) is None
    # gain: 72B, r2s_bridge(실기팀) PolicyGainPacket과 바이트 일치.
    pk = pack_policy_gain(10, [100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 20.0, 20.0], [5.0] * NUM_JOINTS)
    assert len(pk) == POLICY_GAIN_SIZE == 72, (len(pk), POLICY_GAIN_SIZE)
    dpk = unpack_policy_gain(pk)
    assert dpk["kp"][2] == 50.0 and dpk["kd"][7] == 5.0 and dpk["seq"] == 10

    # plant: 108B, CMD 포트 공유 — CMD(168B)와 크기가 달라 크기 단계에서 갈리고 magic으로도 거부.
    pp = pack_plant(11, 1, [0.03] * NUM_JOINTS, [0.5] * NUM_JOINTS, [0.2] * NUM_JOINTS)
    assert len(pp) == PLANT_SIZE == 108, (len(pp), PLANT_SIZE)
    dpp = unpack_plant(pp)
    assert dpp["mode"] == 1 and abs(dpp["armature"][0] - 0.03) < 1e-6 and abs(dpp["coulomb"][7] - 0.2) < 1e-6
    assert unpack_plant(c) is None and unpack_cmd(pp) is None and unpack_ieff(pp) is None
    assert unpack_plant(pack_plant(12, 0, z, z, z))["mode"] == 0
    assert unpack_policy_gain(pa) is None and unpack_policy_act(pk) is None

    print(
        f"r2s_udp(bipedleg) OK  CMD={CMD_SIZE} STATE={STATE_SIZE} MON={MON_SIZE} "
        f"POLICY_CMD={POLICY_CMD_SIZE} POLICY_STATE={POLICY_STATE_SIZE} POLICY_ACT={POLICY_ACT_SIZE} "
        f"POLICY_TELEM={POLICY_TELEM_SIZE} POLICY_PING={POLICY_PING_SIZE}"
    )
