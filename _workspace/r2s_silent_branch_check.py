# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""침묵한 state 출처가 살아 있는 출처를 붙잡지 않는지 검정한다 (Isaac 불필요).

REAL 은 영원히 무응답, SIM 은 정상. 고치기 전에는 매 틱 REAL 대기 0.2 s 를 통째로 버렸다.
"""

import sys
import time

sys.path.insert(0, "scripts/real2sim")
sys.path.insert(0, "scripts/real2sim/r2s_biped_leg")

import policy_runner_bipedleg as R  # noqa: E402

SIM, REAL = R.POLICY_SRC_SIM, R.POLICY_SRC_REAL


class FakeSock:
    """needed 대상이면 recv_blocking 이, 아니면 recvfrom 이 호출된다."""

    def __init__(self, alive):
        self.alive = alive

    def setblocking(self, _):
        pass

    def recvfrom(self, _n):
        raise BlockingIOError


def recv_blocking(sock):
    if sock.alive:
        return {"convention_version": 1, "q": [0.0] * 8, "dq": [0.0] * 8, "gravity": [0, 0, -1]}
    time.sleep(0.2)  # 실제 recv_state_blocking 의 타임아웃과 동일
    return None


socks = {SIM: FakeSock(True), REAL: FakeSock(False)}
last_state = {SIM: None, REAL: None}
needed = {SIM, REAL}

R._silent_ticks.clear()
per_tick = []
for i in range(8):
    t0 = time.monotonic()
    R._collect_states(socks, needed, last_state, recv_blocking)
    per_tick.append((time.monotonic() - t0) * 1e3)

print("틱별 소요 [ms]: " + "  ".join(f"{v:5.0f}" for v in per_tick))
print(f"초기 3틱 평균 {sum(per_tick[:3]) / 3:6.1f} ms   이후 평균 {sum(per_tick[3:]) / len(per_tick[3:]):6.1f} ms")
ok = sum(per_tick[3:]) / len(per_tick[3:]) < 20.0
print("★ 침묵 제외 동작함 — REAL 이 죽어도 SIM 이 50 Hz 를 낼 수 있다" if ok else "⚠ 여전히 붙잡힌다")

# --- 복귀 검정: REAL 이 되살아나면 다시 쓰는가 ---
print("\n[복귀 검정] REAL 소켓이 다시 패킷을 낸다")
pkt = R.pack_policy_state(0, [0.0] * 8, [0.0] * 8, [0, 0, -1], convention_version=1) \
    if hasattr(R, "pack_policy_state") else None


class RevivedSock(FakeSock):
    def __init__(self):
        super().__init__(False)
        self._sent = False

    def recvfrom(self, _n=4096):
        if not self._sent:
            self._sent = True
            import r2s_udp as U
            return U.pack_policy_state(1, [0.0] * 8, [0.0] * 8, [0.0, 0.0, -1.0], convention_version=1), None
        raise BlockingIOError


socks[REAL] = RevivedSock()
t0 = time.monotonic()
R._collect_states(socks, needed, last_state, recv_blocking)
print(f"복귀 틱 소요 {(time.monotonic() - t0) * 1e3:.0f} ms · silent_ticks={dict(R._silent_ticks)}")
print("★ 복귀 후 silent 가 0 이면 다시 blocking 대기 대상" if R._silent_ticks.get(REAL) == 0 else "⚠ 복귀 실패")
