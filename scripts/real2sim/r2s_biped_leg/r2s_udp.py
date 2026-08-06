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

# magic(I) seq(I) + q(8f) + dq(8f) + gravity(3f)  — **articulation 순서** (재매핑 없음)
_POLICY_STATE_FMT: str = "<II" + "f" * NUM_JOINTS + "f" * NUM_JOINTS + "3f"
POLICY_STATE_SIZE: int = struct.calcsize(_POLICY_STATE_FMT)


def pack_policy_state(seq: int, q, dq, gravity) -> bytes:
    """policy_runner용 rich state 직렬화 (articulation 순서).

    Args:
        seq: 증가 시퀀스 번호.
        q: 관절각 [rad], 길이 8, articulation 순서.
        dq: 관절각속도 [rad/s], 길이 8, articulation 순서.
        gravity: projected_gravity_b (base frame 중력 단위벡터), 길이 3.

    Returns:
        POLICY_STATE_SIZE 바이트 패킷.
    """
    vals = [float(q[i]) for i in range(NUM_JOINTS)]
    vals += [float(dq[i]) for i in range(NUM_JOINTS)]
    vals += [float(gravity[i]) for i in range(3)]
    return struct.pack(_POLICY_STATE_FMT, POLICY_STATE_MAGIC, seq & 0xFFFFFFFF, *vals)


def unpack_policy_state(data: bytes) -> dict | None:
    """policy rich state 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``q`` (8), ``dq`` (8), ``gravity`` (3). 전부 articulation 순서.
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
    return {"seq": seq, "q": q, "dq": dq, "gravity": gravity}


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
    assert len(ps) == POLICY_STATE_SIZE, (len(ps), POLICY_STATE_SIZE)
    dps = unpack_policy_state(ps)
    assert len(dps["q"]) == NUM_JOINTS and len(dps["gravity"]) == 3 and abs(dps["gravity"][2] + 1.0) < 1e-6
    pa = pack_policy_act(6, [0.2] * NUM_JOINTS)
    assert len(pa) == POLICY_ACT_SIZE, (len(pa), POLICY_ACT_SIZE)
    assert abs(unpack_policy_act(pa)["target_q"][0] - 0.2) < 1e-6
    # policy 패킷에 position-control magic이 섞여도 거부.
    assert unpack_policy_cmd(pack_cmd(1, z, z, z, z, z)) is None
    assert unpack_cmd(pack_policy_act(1, z)) is None

    print(
        f"r2s_udp(bipedleg) OK  CMD={CMD_SIZE} STATE={STATE_SIZE} MON={MON_SIZE} "
        f"POLICY_CMD={POLICY_CMD_SIZE} POLICY_STATE={POLICY_STATE_SIZE} POLICY_ACT={POLICY_ACT_SIZE}"
    )
