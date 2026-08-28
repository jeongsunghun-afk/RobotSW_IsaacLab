# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""unified 루프가 POLICY_ACT 1개당 정확히 1 step 하는지 확인한다.

50 Hz 로 N 개 보내고, 회신된 POLICY_STATE 의 seq 증가분이 N 과 같은지 본다.
슬라이더 경로가 끼어들면 sim 은 더 많이 step 하지만 POLICY_STATE 는 정책 스텝에서만
나가므로, **회신 수 = 보낸 수** 이면 정책 쪽은 1:1 이다. 추가로 q 변화량으로 과도 진행을 본다.
"""
import socket, struct, sys, time
sys.path.insert(0, "scripts/real2sim/r2s_biped_leg")
import r2s_udp as U

N = 100
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(("127.0.0.1", U.POLICY_STATE_PORT))
rx.settimeout(2.0)
target = [0.0] * 8
got, seqs = 0, []
for i in range(N):
    tx.sendto(U.pack_policy_act(i, target), ("127.0.0.1", U.POLICY_ACT_PORT))
    try:
        d, _ = rx.recvfrom(4096)
        st = U.unpack_policy_state(d)
        if st is not None:
            got += 1
            seqs.append(st["seq"])
    except TimeoutError:
        pass
    time.sleep(0.02)   # 50 Hz
print(f"보낸 POLICY_ACT {N} · 받은 POLICY_STATE {got}")
if seqs:
    print(f"seq {seqs[0]} → {seqs[-1]}  (증가 {seqs[-1]-seqs[0]})")
    print("★ 정책 회신이 1:1 이면 lockstep 유지" if got >= N * 0.9 else "⚠ 회신 누락 — 확인 필요")
