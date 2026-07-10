#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 LowCmd CRC — Unitree GO2 실기가 요구하는 LowCmd.crc 계산 (순수 stdlib).

실기 GO2 펌웨어는 패킹된 ``LowCmd`` C 구조체에 대한 CRC32를 검증하고, 값이 틀리면
명령을 조용히 폐기한다. 이 모듈은 Unitree 공식 ``motor_crc.cpp`` / ``motor_crc.h``
(unitree_ros2 example)의 구현을 **바이트 단위로** 재현한다.

핵심(공식 코드와 동일):
    - CRC32: polynomial 0x04C11DB7, init 0xFFFFFFFF, MSB-first, 반사/최종XOR 없음.
    - 대상: ROS 메시지 직렬화가 아니라 **C 구조체 메모리 레이아웃**(패딩 포함, 812바이트).
    - 범위: ``(sizeof(LowCmd) >> 2) - 1`` = 202 워드 (crc 필드 자신은 제외).

검증: 이 파일을 직접 실행하면 공식 C 코드를 컴파일해 뽑은 정답 벡터(V0/V1/V2)와
바이트 단위로 대조하는 self-test가 돈다 (하드웨어 불필요).

    python3 scripts/real2sim/r2s_go2/lowcmd_crc.py

구조체 레이아웃 (gcc x86-64/ARM 자연 정렬, sizeof=812):
    head[2]        @0    | levelFlag @2 | frameReserve @3
    SN[2]          @4    | version[2] @12 | bandWidth(u16) @20 | pad(2) @22
    motorCmd[20]   @24   (각 36바이트: mode@+0, pad(3), q/dq/tau/Kp/Kd @+4, reserve[3]@+24)
    bms(4)         @744  | wirelessRemote[40] @748 | led[12] @788 | fan[2] @800
    gpio           @802  | pad(1) @803 | reserve(u32) @804 | crc(u32) @808
"""

from __future__ import annotations

import struct
from typing import Any

LOWCMD_STRUCT_SIZE: int = 812  # sizeof(LowCmd), 공식 C 구조체 (self-test로 검증)
_NUM_WORDS: int = LOWCMD_STRUCT_SIZE >> 2  # 203
_CRC_WORDS: int = _NUM_WORDS - 1  # 202 (crc 필드 제외)

# LowCmd 표준 헤더 값 (go2_stand_example.cpp InitLowCmd 참조).
HEAD: tuple[int, int] = (0xFE, 0xEF)
LEVEL_FLAG_LOWLEVEL: int = 0xFF
MOTOR_MODE_SERVO: int = 0x01

_MOTOR_STRIDE: int = 36  # sizeof(MotorCmd)
_MOTOR_BASE: int = 24  # motorCmd[0] 시작 오프셋


def crc32_core(words) -> int:
    """Unitree crc32_core 포팅 — uint32 워드열에 대한 비트연산 CRC32.

    Args:
        words: uint32 값들의 시퀀스(리틀엔디안 호스트가 구조체를 uint32*로 읽은 순서).

    Returns:
        32비트 CRC 값.
    """
    crc = 0xFFFFFFFF
    poly = 0x04C11DB7
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= poly
            xbit >>= 1
    return crc & 0xFFFFFFFF


def _compute_crc_from_raw(
    head: tuple[int, int],
    level_flag: int,
    frame_reserve: int,
    sn: tuple[int, int],
    version: tuple[int, int],
    bandwidth: int,
    motor_cmd: list[tuple[int, float, float, float, float, float, tuple[int, int, int]]],
    bms_off: int,
    bms_reserve: tuple[int, int, int],
    wireless_remote,
    led,
    fan: tuple[int, int],
    gpio: int,
    reserve: int,
) -> int:
    """812바이트 LowCmd 이미지를 만들고 CRC를 계산한다 (모든 20개 motor_cmd 필요).

    packing recipe: 명시적 zero-padded 812바이트 이미지를 만든 뒤
    ``struct.unpack("<203I", buf)``로 워드화하고 crc32_core를 앞 202워드에 적용.
    이는 리틀엔디안 호스트에서 ``crc32_core((uint32_t*)&raw, ...)``가 읽는 순서와 동일하다.
    """
    buf = bytearray(LOWCMD_STRUCT_SIZE)  # 패딩 바이트는 0으로 초기화됨

    struct.pack_into("<2B", buf, 0, head[0] & 0xFF, head[1] & 0xFF)
    buf[2] = level_flag & 0xFF
    buf[3] = frame_reserve & 0xFF
    struct.pack_into("<2I", buf, 4, sn[0] & 0xFFFFFFFF, sn[1] & 0xFFFFFFFF)
    struct.pack_into("<2I", buf, 12, version[0] & 0xFFFFFFFF, version[1] & 0xFFFFFFFF)
    struct.pack_into("<H", buf, 20, bandwidth & 0xFFFF)
    # @22..24: padding (0)

    for i in range(20):
        mode, q, dq, tau, kp, kd, res3 = motor_cmd[i]
        off = _MOTOR_BASE + i * _MOTOR_STRIDE
        buf[off] = mode & 0xFF
        # off+1..off+4: padding (0)
        struct.pack_into("<5f", buf, off + 4, q, dq, tau, kp, kd)
        struct.pack_into("<3I", buf, off + 24, res3[0] & 0xFFFFFFFF, res3[1] & 0xFFFFFFFF, res3[2] & 0xFFFFFFFF)

    # bms @744 (off + 3 reserve)
    buf[744] = bms_off & 0xFF
    for k in range(3):
        buf[745 + k] = bms_reserve[k] & 0xFF
    # wirelessRemote[40] @748
    for k in range(40):
        buf[748 + k] = wireless_remote[k] & 0xFF
    # led[12] @788
    for k in range(12):
        buf[788 + k] = led[k] & 0xFF
    # fan[2] @800
    buf[800] = fan[0] & 0xFF
    buf[801] = fan[1] & 0xFF
    # gpio @802
    buf[802] = gpio & 0xFF
    # @803: padding (0)
    struct.pack_into("<I", buf, 804, reserve & 0xFFFFFFFF)
    # crc @808: 0으로 남김 (CRC 범위에서 제외됨)

    words = struct.unpack(f"<{_NUM_WORDS}I", buf)
    return crc32_core(words[:_CRC_WORDS])


def set_crc(msg) -> int:
    """unitree_go LowCmd ROS 메시지의 crc 필드를 계산·설정한다.

    Args:
        msg: ``unitree_go.msg.LowCmd`` 인스턴스. head/level_flag/motor_cmd 등이 이미
            채워져 있어야 한다. 이 함수가 ``msg.crc``를 in-place로 세팅한다.

    Returns:
        계산된 32비트 CRC 값(``msg.crc``에도 저장됨).
    """
    motor = []
    for i in range(20):
        m = msg.motor_cmd[i]
        motor.append((m.mode, m.q, m.dq, m.tau, m.kp, m.kd, (m.reserve[0], m.reserve[1], m.reserve[2])))
    crc = _compute_crc_from_raw(
        head=(msg.head[0], msg.head[1]),
        level_flag=msg.level_flag,
        frame_reserve=msg.frame_reserve,
        sn=(msg.sn[0], msg.sn[1]),
        version=(msg.version[0], msg.version[1]),
        bandwidth=msg.bandwidth,
        motor_cmd=motor,
        bms_off=msg.bms_cmd.off,
        bms_reserve=(msg.bms_cmd.reserve[0], msg.bms_cmd.reserve[1], msg.bms_cmd.reserve[2]),
        wireless_remote=[msg.wireless_remote[k] for k in range(40)],
        led=[msg.led[k] for k in range(12)],
        fan=(msg.fan[0], msg.fan[1]),
        gpio=msg.gpio,
        reserve=msg.reserve,
    )
    msg.crc = crc
    return crc


# ----------------------------------------------------------------------------
# self-test — 공식 C 코드(crc_oracle)에서 뽑은 정답 벡터와 바이트 단위 대조.
# ----------------------------------------------------------------------------
_MotorTuple = tuple[int, float, float, float, float, float, tuple[int, int, int]]
_ZERO_MOTOR: _MotorTuple = (0, 0.0, 0.0, 0.0, 0.0, 0.0, (0, 0, 0))


def _raw_kwargs(motor_cmd, head=(0, 0), level_flag=0, gpio=0) -> dict[str, Any]:
    return dict(
        head=head,
        level_flag=level_flag,
        frame_reserve=0,
        sn=(0, 0),
        version=(0, 0),
        bandwidth=0,
        motor_cmd=motor_cmd,
        bms_off=0,
        bms_reserve=(0, 0, 0),
        wireless_remote=[0] * 40,
        led=[0] * 12,
        fan=(0, 0),
        gpio=gpio,
        reserve=0,
    )


def _self_test() -> None:
    # V0: 전부 0
    v0 = _compute_crc_from_raw(**_raw_kwargs([_ZERO_MOTOR] * 20))
    assert v0 == 0x4B5A4880, f"V0 mismatch: {v0:08x} != 4b5a4880"

    # V1: head/level_flag/mode + 몇몇 q, kp=25/kd=0.5 (motor 0..11만)
    m1: list[_MotorTuple] = [_ZERO_MOTOR] * 20
    for i in range(12):
        m1[i] = (MOTOR_MODE_SERVO, 0.0, 0.0, 0.0, 25.0, 0.5, (0, 0, 0))
    m1[0] = (MOTOR_MODE_SERVO, 0.1, 0.0, 0.0, 25.0, 0.5, (0, 0, 0))
    m1[4] = (MOTOR_MODE_SERVO, 0.8, 0.0, 0.0, 25.0, 0.5, (0, 0, 0))
    m1[11] = (MOTOR_MODE_SERVO, -1.5, 0.0, 0.0, 25.0, 0.5, (0, 0, 0))
    v1 = _compute_crc_from_raw(**_raw_kwargs(m1, head=HEAD, level_flag=LEVEL_FLAG_LOWLEVEL))
    assert v1 == 0xB7628F5E, f"V1 mismatch: {v1:08x} != b7628f5e"

    # V2: DEFAULT_POSE 전관절 + kp25/kd0.5
    pose = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
    m2: list[_MotorTuple] = [_ZERO_MOTOR] * 20
    for i in range(12):
        m2[i] = (MOTOR_MODE_SERVO, pose[i], 0.0, 0.0, 25.0, 0.5, (0, 0, 0))
    v2 = _compute_crc_from_raw(**_raw_kwargs(m2, head=HEAD, level_flag=LEVEL_FLAG_LOWLEVEL))
    assert v2 == 0x9C9F3296, f"V2 mismatch: {v2:08x} != 9c9f3296"

    print(f"[lowcmd_crc] self-test PASS — sizeof={LOWCMD_STRUCT_SIZE}, words={_NUM_WORDS}, crc_words={_CRC_WORDS}")
    print(f"[lowcmd_crc]   V0={v0:08x}  V1={v1:08x}  V2={v2:08x}  (공식 C oracle과 일치)")


if __name__ == "__main__":
    _self_test()
