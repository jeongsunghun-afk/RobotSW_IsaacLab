# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-HindLeg UDP 패킷 스키마 (gui <-> sim_runner <-> monitor 경계).

순수 stdlib(struct)만 사용 — Isaac conda Python 3.12 와 시스템 Python 3.10 양쪽에서
임포트 가능해야 한다. numpy/torch/ros 의존성 금지.

전송(현재, ROS2 미정): 순수 UDP.
    gui ──cmd(CMD_PORT)──▶ sim_runner ──state(STATE_PORT)──▶ gui
    gui ──monitor(MONITOR_PORT)──▶ monitor (별도 프로세스; action+sim 한 패킷, time-aligned)

고정베이스 단일 뒷다리(5-DOF)이므로 IMU/base 상태는 없다.

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_hind_leg/CONTRACT.md §4
"""

from __future__ import annotations

import struct

NUM_JOINTS: int = 5  # R_Skeleton hind leg: thigh_r, thigh_p, knee_p, ankle_p, toe_p

# 포트 (go2의 9871/9872 와 충돌 회피 위해 별도 대역)
CMD_PORT: int = 9873  # gui -> sim_runner
STATE_PORT: int = 9874  # sim_runner -> gui
MONITOR_PORT: int = 9875  # gui(relay) -> monitor

CMD_MAGIC: int = 0x52324843  # "R2HC"
STATE_MAGIC: int = 0x52324853  # "R2HS"
MONITOR_MAGIC: int = 0x52324D4E  # "R2MN"

# 명령: magic(I) seq(I) + 5 x (q,dq,kp,kd,tau) f
_CMD_FMT: str = "<II" + "5f" * NUM_JOINTS
CMD_SIZE: int = struct.calcsize(_CMD_FMT)

# 상태: magic(I) seq(I) sim_time(f) + 5 x (q,dq,ddq,tau_est) f
_STATE_FMT: str = "<IIf" + "4f" * NUM_JOINTS  # codespell:ignore
STATE_SIZE: int = struct.calcsize(_STATE_FMT)

# 모니터: magic(I) seq(I) + 5 x action_q + 5 x (sim_q, sim_dq, sim_tau)
_MON_FMT: str = "<II" + "f" * NUM_JOINTS + "3f" * NUM_JOINTS
MON_SIZE: int = struct.calcsize(_MON_FMT)


# ---------------------------------------------------------------------------
# 명령 (gui -> sim_runner)
# ---------------------------------------------------------------------------


def pack_cmd(seq: int, q, dq, kp, kd, tau) -> bytes:
    """5-관절 PD 명령을 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        q: 관절 목표각 [rad], 길이 5.
        dq: 관절 목표각속도 [rad/s], 길이 5.
        kp: 위치 게인, 길이 5.
        kd: 속도 게인, 길이 5.
        tau: 피드포워드 토크 [Nm], 길이 5.

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
        키: ``seq``, ``q``, ``dq``, ``kp``, ``kd``, ``tau`` (각 리스트 길이 5).
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
    """5-관절 상태를 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        sim_time: 시뮬레이션 시간 [s].
        q: 관절각 [rad], 길이 5.
        dq: 관절각속도 [rad/s], 길이 5.
        ddq: 관절각가속도 [rad/s^2], 길이 5.
        tau_est: 추정 토크 [Nm], 길이 5.

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
        키: ``seq``, ``sim_time``, ``q``, ``dq``, ``ddq``, ``tau_est`` (각 길이 5).
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
        action_q: 명령 관절각 [rad], 길이 5.
        sim_q: sim 관절각 [rad], 길이 5.
        sim_dq: sim 관절각속도 [rad/s], 길이 5.
        sim_tau: sim 추정 토크 [Nm], 길이 5.

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
        키: ``seq``, ``action_q``, ``sim_q``, ``sim_dq``, ``sim_tau`` (각 길이 5).
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


if __name__ == "__main__":
    z = [0.0] * NUM_JOINTS
    c = pack_cmd(1, z, z, [300.0] * 5, [5.0] * 5, z)
    assert len(c) == CMD_SIZE, (len(c), CMD_SIZE)
    assert unpack_cmd(c)["kp"][0] == 300.0
    s = pack_state(2, 1.5, z, z, z, z)
    assert len(s) == STATE_SIZE, (len(s), STATE_SIZE)
    assert unpack_state(s)["sim_time"] == 1.5
    m = pack_monitor(3, [0.1] * 5, z, z, z)
    assert len(m) == MON_SIZE, (len(m), MON_SIZE)
    assert abs(unpack_monitor(m)["action_q"][0] - 0.1) < 1e-6
    print(f"r2s_udp(hindleg) OK  CMD={CMD_SIZE} STATE={STATE_SIZE} MON={MON_SIZE}")
