# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""r2s sim policy 경로가 목표를 soft limit 으로 자르는지 확인한다.

한계 밖 목표(±9 rad)를 보내고 회신 q 가 한계 안에 머무는지 본다.
자르기 전에는 PD 가 한계에 계속 밀어붙이며 포화했다.
"""
import os, socket, sys, time
sys.path.insert(0, "scripts/real2sim/r2s_biped_leg")
import r2s_udp as U, motions as M

PA, PS = int(os.environ["PA_PORT"]), int(os.environ["PS_PORT"])
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rx.bind(("127.0.0.1", PS)); rx.settimeout(1.0)

# ⚠ 하드코딩 금지 — POLICY_TO_MOTOR([0,4,1,5,2,6,3,7])와 **다른 순열**이다. GUI 것을 그대로 쓴다.
sys.path.insert(0, "scripts/real2sim/r2s_biped_leg")
import gui_controller as G
ART = G._ART_FOR_LEGMAJOR                 # leg-major i ← articulation
last = None
for i in range(150):                      # 3초: 한계 밖 목표로 계속 밀기
    tx.sendto(U.pack_policy_act(i, [9.0]*8), ("127.0.0.1", PA))
    try: d, _ = rx.recvfrom(4096); last = U.unpack_policy_state(d)
    except TimeoutError: pass
    time.sleep(0.02)

q_art = last["q"]
q_lm  = [q_art[a] for a in ART]
print("관절(leg-major)  q [rad]        soft hi        초과")
bad = 0
for i, (q, (lo, hi)) in enumerate(zip(q_lm, M.SOFT_LIMITS_RAD)):
    over = max(0.0, q - hi) + max(0.0, lo - q)
    if over > 1e-3: bad += 1
    print(f"  {M.JOINT_NAMES[i]:12s} {q:+8.4f}   [{lo:+.3f},{hi:+.3f}]   {over:+.4f}")
print(f"\n★ 한계 밖 관절 {bad} 개 — 0 이면 sim 이 목표를 자르고 있다")
