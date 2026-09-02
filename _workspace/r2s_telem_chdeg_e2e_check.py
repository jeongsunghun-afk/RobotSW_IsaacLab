# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""stub real_runner 를 띄워 TELEM 의 ch_deg 가 실제로 선로를 타는지 종단 검정."""
import importlib.util, pathlib, socket, subprocess, sys, time
p = pathlib.Path('/home/lgb/IsaacLab-6.0/scripts/real2sim/r2s_biped_leg/r2s_udp.py')
spec = importlib.util.spec_from_file_location('r2s_udp', p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

proc = subprocess.Popen(['/tmp/rrbuild/real_runner_bipedleg'], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True)
try:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", m.REAL_TELEM_PORT)); rx.settimeout(6.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    time.sleep(1.5)
    got = None
    for _ in range(60):
        tx.sendto(m.pack_policy_ping(1), ("127.0.0.1", m.REAL_ACT_PORT))
        try:
            data, _a = rx.recvfrom(2048)
        except socket.timeout:
            continue
        d = m.unpack_policy_telem(data)
        if d is not None:
            got = (len(data), d); break
        time.sleep(0.05)
    assert got, "TELEM 을 못 받았다"
    size, d = got
    print(f"[PASS] TELEM 수신 {size} B (기대 {m.POLICY_TELEM_SIZE})")
    assert size == m.POLICY_TELEM_SIZE
    ch = d["ch_deg_lm"]
    assert ch is not None and len(ch) == 8, f"ch_deg 없음: {ch}"
    print(f"[PASS] ch_deg_lm 수신 (bit25 유효): {[round(v,2) for v in ch]}")
    print(f"       q(관절, articulation): {[round(v,3) for v in d['q']]}")
    print(f"       tau(채널): {[round(v,2) for v in d['tau']]}")
    print("\n전 항목 통과")
finally:
    proc.terminate(); proc.wait(timeout=5)
