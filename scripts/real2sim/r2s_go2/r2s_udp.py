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

# tuner 모드 채널 (Phase 2.5) — live(9871/9872) 경로와 독립. 물성 슬라이더 <-> sim <-> 플롯.
TUNER_PARAM_PORT: int = 9875  # tuner_gui -> sim_runner_tuner (물성 파라미터)
TUNER_TELEM_PORT: int = 9876  # sim_runner_tuner -> tuner_monitor (q_sim/q_cmd/q_real)

CMD_MAGIC: int = 0x52324743  # "R2GC"
STATE_MAGIC: int = 0x52324753  # "R2GS"
TUNER_PARAM_MAGIC: int = 0x52325450  # "R2TP"
TUNER_TELEM_MAGIC: int = 0x52325454  # "R2TT"

# 명령 패킷: magic(I) seq(I) + 12 x (q,dq,kp,kd,tau) f
_CMD_FMT: str = "<II" + "5f" * NUM_MOTORS
CMD_SIZE: int = struct.calcsize(_CMD_FMT)  # 248

# 상태 패킷: magic(I) seq(I) sim_time(f) + 12 x (q,dq,ddq,tau_est) f + imu 10f
_STATE_FMT: str = "<IIf" + "4f" * NUM_MOTORS + "10f"  # codespell:ignore
STATE_SIZE: int = struct.calcsize(_STATE_FMT)  # 244

# tuner 파라미터 패킷: magic(I) seq(I) + 6 전역 스칼라 f
#   armature[kg·m²], viscous[N·m·s/rad], coulomb[N·m], kp[N·m/rad], kd[N·m·s/rad], delay[sim step]
# 전 12관절 공통 스칼라(사람이 눈으로 bounds를 잡는 용도 — per-joint 49개는 CMA-ES 몫).
_TUNER_PARAM_FMT: str = "<II6f"
TUNER_PARAM_SIZE: int = struct.calcsize(_TUNER_PARAM_FMT)  # 32

# tuner 텔레메트리 패킷: magic(I) seq(I) sim_time(f) has_real(I) + q_sim(12) q_cmd(12) q_real(12) f
#   has_real=0이면 q_real은 의미 없음(replay 없이 chirp만 구동 중).
_TUNER_TELEM_FMT: str = "<IIfI" + "f" * (3 * NUM_MOTORS)
TUNER_TELEM_SIZE: int = struct.calcsize(_TUNER_TELEM_FMT)  # 160


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


# ---------------------------------------------------------------------------
# tuner 파라미터 패킷 (tuner_gui -> sim_runner_tuner) — Phase 2.5
# ---------------------------------------------------------------------------


def pack_tuner_params(seq: int, armature, viscous, coulomb, kp, kd, delay) -> bytes:
    """전역 물성 스칼라 6개를 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        armature: 관절 armature [kg·m²], 전 관절 공통.
        viscous: 점성 마찰 [N·m·s/rad], 전 관절 공통.
        coulomb: 쿨롱 마찰 [N·m], 전 관절 공통.
        kp: 위치 게인 [N·m/rad], 전 관절 공통.
        kd: 속도 게인 [N·m·s/rad], 전 관절 공통.
        delay: 토크 지연 [sim step], 전 관절 공통(적용 시 정수로 반올림).

    Returns:
        32-byte 패킷.
    """
    return struct.pack(
        _TUNER_PARAM_FMT,
        TUNER_PARAM_MAGIC,
        seq & 0xFFFFFFFF,
        float(armature),
        float(viscous),
        float(coulomb),
        float(kp),
        float(kd),
        float(delay),
    )


def unpack_tuner_params(data: bytes) -> dict | None:
    """tuner 파라미터 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``armature``, ``viscous``, ``coulomb``, ``kp``, ``kd``, ``delay``.
    """
    if len(data) != TUNER_PARAM_SIZE:
        return None
    fields = struct.unpack(_TUNER_PARAM_FMT, data)
    if fields[0] != TUNER_PARAM_MAGIC:
        return None
    return {
        "seq": fields[1],
        "armature": fields[2],
        "viscous": fields[3],
        "coulomb": fields[4],
        "kp": fields[5],
        "kd": fields[6],
        "delay": fields[7],
    }


# ---------------------------------------------------------------------------
# tuner 텔레메트리 패킷 (sim_runner_tuner -> tuner_monitor) — Phase 2.5
# ---------------------------------------------------------------------------


def pack_tuner_telem(seq: int, sim_time: float, has_real: bool, q_sim, q_cmd, q_real) -> bytes:
    """sim/명령/실기 관절각을 UDP 바이트로 직렬화.

    Args:
        seq: 증가 시퀀스 번호.
        sim_time: 시뮬레이션 시간 [s].
        has_real: replay 데이터가 있어 q_real이 유효한지.
        q_sim: 현재 sim 관절각 [rad], 길이 12, JOINT_ORDER 순서.
        q_cmd: 재생 중인 명령 관절각 [rad], 길이 12.
        q_real: 녹화된 실기 관절각 [rad], 길이 12(has_real=False면 무의미).

    Returns:
        160-byte 패킷.
    """
    vals: list[float] = []
    vals.extend(float(x) for x in q_sim)
    vals.extend(float(x) for x in q_cmd)
    vals.extend(float(x) for x in q_real)
    return struct.pack(
        _TUNER_TELEM_FMT, TUNER_TELEM_MAGIC, seq & 0xFFFFFFFF, float(sim_time), 1 if has_real else 0, *vals
    )


def unpack_tuner_telem(data: bytes) -> dict | None:
    """tuner 텔레메트리 패킷 역직렬화. magic 불일치/크기 오류 시 None.

    Returns:
        키: ``seq``, ``sim_time``, ``has_real``, ``q_sim``, ``q_cmd``, ``q_real`` (각 길이 12).
    """
    if len(data) != TUNER_TELEM_SIZE:
        return None
    fields = struct.unpack(_TUNER_TELEM_FMT, data)
    if fields[0] != TUNER_TELEM_MAGIC:
        return None
    body = fields[4:]
    q_sim = list(body[0:NUM_MOTORS])
    q_cmd = list(body[NUM_MOTORS : 2 * NUM_MOTORS])
    q_real = list(body[2 * NUM_MOTORS : 3 * NUM_MOTORS])
    return {
        "seq": fields[1],
        "sim_time": fields[2],
        "has_real": bool(fields[3]),
        "q_sim": q_sim,
        "q_cmd": q_cmd,
        "q_real": q_real,
    }


if __name__ == "__main__":
    # 자체 라운드트립 검증
    z = [0.0] * NUM_MOTORS
    c = pack_cmd(1, z, z, [25.0] * 12, [0.5] * 12, z)
    assert len(c) == CMD_SIZE == 248, (len(c), CMD_SIZE)
    cmd = unpack_cmd(c)
    assert cmd is not None and cmd["kp"][0] == 25.0
    s = pack_state(2, 1.5, z, z, z, z, [1.0] + [0.0] * 9)
    assert len(s) == STATE_SIZE == 244, (len(s), STATE_SIZE)
    state = unpack_state(s)
    assert state is not None and state["imu"][0] == 1.0
    p = pack_tuner_params(3, 0.01, 0.1, 0.15, 25.0, 0.5, 4)
    assert len(p) == TUNER_PARAM_SIZE == 32, (len(p), TUNER_PARAM_SIZE)
    params = unpack_tuner_params(p)
    assert params is not None and params["kp"] == 25.0 and params["delay"] == 4.0
    one_to_twelve = [float(i) for i in range(NUM_MOTORS)]
    t = pack_tuner_telem(4, 2.5, True, one_to_twelve, z, z)
    assert len(t) == TUNER_TELEM_SIZE == 160, (len(t), TUNER_TELEM_SIZE)
    telem = unpack_tuner_telem(t)
    assert telem is not None and telem["has_real"] and telem["q_sim"][11] == 11.0
    print(
        f"r2s_udp roundtrip OK  CMD={CMD_SIZE} STATE={STATE_SIZE} "
        f"TUNER_PARAM={TUNER_PARAM_SIZE} TUNER_TELEM={TUNER_TELEM_SIZE}"
    )
