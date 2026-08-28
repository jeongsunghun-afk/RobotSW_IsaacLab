"""monitor.py `_append_seq` 의 발행측 재시작 처리를 검정한다 (디스플레이 불필요).

증상: 발행측이 짧게 돌다 멈췄다가 다시 켜지면 새 seq 가 0 근처로 돌아오는데, 종전 코드는
seq 크기만 보고 "UDP 재정렬"로 판정해 **새 패킷을 전부 버렸다**. 버퍼는 낡은 샘플만 쥔 채
창 밖으로 흘러가므로 화면이 한동안 빈다(= 까만 화면).

수정: 마지막 수신 이후 `STALE_RESET_S` 넘게 조용했다면 재정렬이 아니라 **재시작**으로 본다.
"""

from __future__ import annotations

import collections
import sys

sys.path.insert(0, "scripts/real2sim/r2s_biped_leg")

import monitor as M  # noqa: E402

append = M.MonitorWindow._append_seq
fail = 0


def check(label, ok, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        fail += 1


def feed(buf, anchor, seqs, t0, dt=0.02):
    t = t0
    for s in seqs:
        anchor = append(buf, anchor, s, (0.0, 0.0, 0.0, 0.0), t)
        t += dt
    return anchor, t


print("[1] 짧은 세션(seq 0~99) 뒤 재시작 — 종전엔 새 패킷이 전부 버려졌다")
buf = collections.deque(maxlen=M.BUF_MAXLEN)
anchor, t = feed(buf, None, range(100), 1000.0)
check("1차 세션 수신", len(buf) == 100 and anchor[0] == 99, f"n={len(buf)} last_seq={anchor[0]}")

# 발행측 정지 후 재시작: 조용한 시간을 두고 seq 가 0 부터 다시 온다
t_restart = t + M.STALE_RESET_S + 2.0
anchor, _ = feed(buf, anchor, range(30), t_restart)
check("재시작 패킷이 버려지지 않음", len(buf) == 30, f"버퍼 {len(buf)}개 (30이어야 함 — 0이면 버려진 것)")
check("앵커가 새 세션을 가리킴", anchor[0] == 29, f"last_seq={anchor[0]}")

print("\n[2] 진짜 UDP 재정렬은 여전히 버린다 (조용한 시간 없이 낮은 seq)")
buf2 = collections.deque(maxlen=M.BUF_MAXLEN)
anchor2, t2 = feed(buf2, None, range(50), 2000.0)
n_before = len(buf2)
anchor2 = append(buf2, anchor2, 40, (0.0, 0.0, 0.0, 0.0), t2)  # 바로 직후 도착한 옛 seq
check("재정렬 패킷은 무시", len(buf2) == n_before and anchor2[0] == 49, f"n={len(buf2)} last_seq={anchor2[0]}")

print("\n[3] seq 가 크게 뒤로 뛰면(BUF_MAXLEN 초과) 조용한 시간 없이도 재시작 처리")
buf3 = collections.deque(maxlen=M.BUF_MAXLEN)
anchor3, t3 = feed(buf3, None, range(3000, 3000 + 200), 3000.0)
anchor3 = append(buf3, anchor3, 5, (0.0, 0.0, 0.0, 0.0), t3)
check("버퍼 clear 후 새 세션 시작", len(buf3) == 1 and anchor3[0] == 5, f"n={len(buf3)} last_seq={anchor3[0]}")

print("\n[4] 정상 스트리밍은 영향 없음")
buf4 = collections.deque(maxlen=M.BUF_MAXLEN)
anchor4, _ = feed(buf4, None, range(500), 4000.0)
check("500개 전부 축적", len(buf4) == 500 and anchor4[0] == 499, f"n={len(buf4)}")

print()
if fail:
    print(f"★ {fail}건 실패")
    sys.exit(1)
print("★ 전 항목 통과")
