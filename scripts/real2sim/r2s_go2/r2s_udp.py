# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 UDP 패킷 스키마 (sim_bridge <-> sim_runner 경계).

순수 stdlib(struct)만 사용 — Isaac conda Python 3.12 와 시스템 Python 3.10(ROS2 humble)
양쪽에서 임포트 가능해야 한다. numpy/torch/ros 의존성 금지.

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md §4
"""

from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# 상수 (CONTRACT.md §4)
# ---------------------------------------------------------------------------

NUM_MOTORS: int = 12  # GO2 다리 관절 (Unitree MotorCmd[20] 중 0..11)

CMD_PORT: int = 9871  # sim_bridge -> sim_runner
STATE_PORT: int = 9872  # sim_runner -> sim_bridge

CMD_MAGIC: int = 0x52324743  # "R2GC"
STATE_MAGIC: int = 0x52324753  # "R2GS"

# 명령 패킷: magic(I) seq(I) + 12 x (q,dq,kp,kd,tau) f
_CMD_FMT: str = "<II" + "5f" * NUM_MOTORS
CMD_SIZE: int = struct.calcsize(_CMD_FMT)  # 248

# 상태 패킷: magic(I) seq(I) sim_time(f) + 12 x (q,dq,ddq,tau_est) f + imu 10f
_STATE_FMT: str = "<IIf" + "4f" * NUM_MOTORS + "10f"  # codespell:ignore
STATE_SIZE: int = struct.calcsize(_STATE_FMT)  # 244


# ---------------------------------------------------------------------------
# 명령 패킷 (sim_bridge -> sim_runner)
# ---------------------------------------------------------------------------


def pack_cmd(seq: int, q, dq, kp, kd, tau) -> bytes:
    """12-관절 명령을 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        q: 관절 목표각 [rad], 길이 12.
        dq: 관절 목표각속도 [rad/s], 길이 12.
        kp: 위치 게인, 길이 12.
        kd: 속도 게인, 길이 12.
        tau: 피드포워드 토크 [Nm], 길이 12.

    Returns:
        248-byte 패킷.
    """
    vals: list[float] = []
    for i in range(NUM_MOTORS):
        vals.extend((float(q[i]), float(dq[i]), float(kp[i]), float(kd[i]), float(tau[i])))
    return struct.pack(_CMD_FMT, CMD_MAGIC, seq & 0xFFFFFFFF, *vals)


def unpack_cmd(data: bytes) -> dict | None:
    """명령 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``q``, ``dq``, ``kp``, ``kd``, ``tau`` (각 리스트 길이 12).
    """
    if len(data) != CMD_SIZE:
        return None
    fields = struct.unpack(_CMD_FMT, data)
    if fields[0] != CMD_MAGIC:
        return None
    seq = fields[1]
    body = fields[2:]
    q, dq, kp, kd, tau = [], [], [], [], []
    for i in range(NUM_MOTORS):
        base = i * 5
        q.append(body[base + 0])
        dq.append(body[base + 1])
        kp.append(body[base + 2])
        kd.append(body[base + 3])
        tau.append(body[base + 4])
    return {"seq": seq, "q": q, "dq": dq, "kp": kp, "kd": kd, "tau": tau}


# ---------------------------------------------------------------------------
# 상태 패킷 (sim_runner -> sim_bridge)
# ---------------------------------------------------------------------------


def pack_state(seq: int, sim_time: float, q, dq, ddq, tau_est, imu10) -> bytes:
    """12-관절 상태 + IMU를 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        sim_time: 시뮬레이션 시간 [s].
        q: 관절각 [rad], 길이 12.
        dq: 관절각속도 [rad/s], 길이 12.
        ddq: 관절각가속도 [rad/s^2], 길이 12.
        tau_est: 추정 토크 [Nm], 길이 12.
        imu10: [quat_w, quat_x, quat_y, quat_z, gyro_xyz, acc_xyz], 길이 10.

    Returns:
        244-byte 패킷.
    """
    vals: list[float] = []
    for i in range(NUM_MOTORS):
        vals.extend((float(q[i]), float(dq[i]), float(ddq[i]), float(tau_est[i])))
    vals.extend(float(x) for x in imu10)
    return struct.pack(_STATE_FMT, STATE_MAGIC, seq & 0xFFFFFFFF, float(sim_time), *vals)


def unpack_state(data: bytes) -> dict | None:
    """상태 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``sim_time``, ``q``, ``dq``, ``ddq``, ``tau_est`` (길이 12), ``imu`` (길이 10).
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
    for i in range(NUM_MOTORS):
        base = i * 4
        q.append(body[base + 0])
        dq.append(body[base + 1])
        ddq.append(body[base + 2])
        tau_est.append(body[base + 3])
    imu = list(body[NUM_MOTORS * 4 : NUM_MOTORS * 4 + 10])
    return {"seq": seq, "sim_time": sim_time, "q": q, "dq": dq, "ddq": ddq, "tau_est": tau_est, "imu": imu}


if __name__ == "__main__":
    # 자체 라운드트립 검증
    z = [0.0] * NUM_MOTORS
    c = pack_cmd(1, z, z, [25.0] * 12, [0.5] * 12, z)
    assert len(c) == CMD_SIZE == 248, (len(c), CMD_SIZE)
    assert unpack_cmd(c)["kp"][0] == 25.0
    s = pack_state(2, 1.5, z, z, z, z, [1.0] + [0.0] * 9)
    assert len(s) == STATE_SIZE == 244, (len(s), STATE_SIZE)
    assert unpack_state(s)["imu"][0] == 1.0
    print(f"r2s_udp roundtrip OK  CMD_SIZE={CMD_SIZE} STATE_SIZE={STATE_SIZE}")
