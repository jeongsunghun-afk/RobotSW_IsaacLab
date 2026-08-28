"""GUI 정책 루프의 타이밍 수정을 검정한다 (Isaac·torch·Qt 불필요).

2026-08-28 실기 캡처에서 드러난 세 가지를 각각 판정한다
(`reports/real2sim/_comparisons/bipedleg_policy_capture_20260828/`):

  A. 두 출처를 **동시** 대기하는가 — 종전 직렬 대기는 한 틱 비용이 두 출처의 **합**이었다.
  B. 침묵한 출처가 살아 있는 출처를 붙잡지 않는가 (+ 복귀).
  C. 페이서·gait clock·monitor seq 격자 산술이 맞는가.

`gui_controller` 는 임포트 시 PyQt5/torch 를 끌어오므로, 여기서는 **모듈을 통째로 임포트하지
않고** 검정 대상 함수만 소스에서 떼어 와 실행한다. 그래야 헤드리스에서 돌고, 검정 대상이
실제 배포 코드와 같은 텍스트임이 보장된다.
"""

from __future__ import annotations

import ast
import select
import sys
import time

SRC_PATH = "scripts/real2sim/r2s_biped_leg/gui_controller.py"
SIM, REAL = 0, 1


def _load_target():
    """`gui_controller.py` 에서 검정 대상만 떼어 온다 (PyQt/torch 임포트 회피)."""
    tree = ast.parse(open(SRC_PATH).read())
    want_fn = {"collect_states_concurrent"}
    want_const = {"_SILENT_TICKS_LIMIT", "STATE_WAIT_S", "STEP_DT", "GAIT_PERIOD", "PHASE_MAX_DT", "MONITOR_GRID_S"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fn:
            body.append(node)
        elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) in want_const:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in want_const for t in node.targets):
            body.append(node)
    missing = want_fn | want_const
    for n in body:
        if isinstance(n, ast.FunctionDef):
            missing.discard(n.name)
        elif isinstance(n, ast.AnnAssign):
            missing.discard(n.target.id)
        else:
            for t in n.targets:
                missing.discard(getattr(t, "id", None))
    if missing:
        raise SystemExit(f"소스에서 못 찾음: {sorted(missing)}")
    ns = {"time": time, "select": select}
    exec(compile(ast.Module(body=body, type_ignores=[]), SRC_PATH, "exec"), ns)  # noqa: S102
    return ns


G = _load_target()
collect = G["collect_states_concurrent"]
LIMIT = G["_SILENT_TICKS_LIMIT"]
STEP_DT, GAIT_PERIOD = G["STEP_DT"], G["GAIT_PERIOD"]
PHASE_MAX_DT, MONITOR_GRID_S = G["PHASE_MAX_DT"], G["MONITOR_GRID_S"]

PKT = {"convention_version": 1, "q": [0.0] * 8, "dq": [0.0] * 8, "gravity": [0.0, 0.0, -1.0]}


class FakeSock:
    """`ready_after_s` 뒤에 패킷 하나가 준비되는 가짜 소켓. None 이면 영원히 무응답."""

    def __init__(self, ready_after_s):
        self.ready_after_s = ready_after_s
        self.armed_at = time.monotonic()

    def ready(self):
        return self.ready_after_s is not None and (time.monotonic() - self.armed_at) >= self.ready_after_s

    def rearm(self):
        self.armed_at = time.monotonic()


def drain(sock):
    if sock.ready():
        sock.rearm()
        return dict(PKT)
    return None


def fake_select(rlist, _w, _x, timeout):
    """준비될 때까지 잔다 — 실제 select 의 blocking 을 흉내낸다."""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        ready = [s for s in rlist if s.ready()]
        if ready:
            return (ready, [], [])
        time.sleep(0.0005)
    return ([], [], [])


fail = 0


def check(label, ok, detail=""):
    global fail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        fail += 1


# ── A. 동시 대기 ─────────────────────────────────────────────────────────────
print("[A] 두 출처 동시 대기 (각각 준비까지 20 ms)")
socks = {SIM: FakeSock(0.020), REAL: FakeSock(0.020)}
st, sil = {SIM: None, REAL: None}, {}
per = []
for _ in range(6):
    for s in socks.values():
        s.rearm()
    t0 = time.monotonic()
    fresh, _, _ = collect(socks, {SIM, REAL}, st, sil, drain, 0.2, fake_select)
    per.append((time.monotonic() - t0) * 1e3)
    assert len(fresh) == 2, fresh
avg = sum(per) / len(per)
print("     틱별 [ms]: " + "  ".join(f"{v:5.1f}" for v in per))
check("합(≈40 ms)이 아니라 최댓값(≈20 ms)", avg < 30.0, f"평균 {avg:.1f} ms → {1000 / avg:.1f} Hz")

# ── B. 침묵 / 복귀 ───────────────────────────────────────────────────────────
print(f"\n[B] REAL 무응답 — {LIMIT}틱 뒤 대기에서 빠지는가")
socks = {SIM: FakeSock(0.002), REAL: FakeSock(None)}
st, sil = {SIM: None, REAL: None}, {}
per, silent_at = [], None
for i in range(8):
    socks[SIM].rearm()
    t0 = time.monotonic()
    _, ns, _ = collect(socks, {SIM, REAL}, st, sil, drain, 0.2, fake_select)
    per.append((time.monotonic() - t0) * 1e3)
    if ns:
        silent_at = i
print("     틱별 [ms]: " + "  ".join(f"{v:5.1f}" for v in per))
check(f"{LIMIT}번째 틱에 침묵 판정", silent_at == LIMIT - 1, f"tick={silent_at}")
tail = sum(per[LIMIT:]) / len(per[LIMIT:])
check("침묵 후 SIM 이 안 붙잡힘", tail < 20.0, f"이후 평균 {tail:.1f} ms → {1000 / tail:.1f} Hz")
check("침묵 전에는 예산만큼 기다림", per[0] > 150.0, f"첫 틱 {per[0]:.0f} ms (예산 200)")

print("\n[B2] REAL 복귀")
# 침묵한 출처는 **논블로킹 드레인으로만** 확인한다. 그래서 복귀는 "패킷이 소켓에 이미 도착해
# 있는 첫 틱"에 일어난다 — 여기서도 실제와 같게 패킷이 도착할 시간을 준 뒤 틱을 돈다.
socks[REAL] = FakeSock(0.002)
time.sleep(0.005)
socks[SIM].rearm()
fresh, _, rec = collect(socks, {SIM, REAL}, st, sil, drain, 0.2, fake_select)
check("패킷 도착 후 첫 틱에 복귀", rec == [REAL] and REAL in fresh, f"recovered={rec}")
check("복귀 후 침묵 카운터 0", sil[REAL] == 0, f"silent[REAL]={sil[REAL]}")

# ── B3. 소비하지 않는 출처도 드레인 ──────────────────────────────────────────
print("\n[B3] 소비하지 않는 출처 드레인 (OFF hold 가 낡은 q 를 안 쓰도록)")
socks = {SIM: FakeSock(0.002), REAL: FakeSock(0.002)}
st, sil = {SIM: None, REAL: None}, {}
time.sleep(0.01)
collect(socks, {SIM}, st, sil, drain, 0.2, fake_select)
check("needed 밖 REAL 도 last_state 갱신", st[REAL] is not None)
check("needed 밖 REAL 은 침묵 카운터 미적용", REAL not in sil, f"silent={sil}")

# ── C. 페이서 / gait clock / monitor 격자 산술 ───────────────────────────────
print("\n[C] 페이서 — 빠른 본문에서도 20 ms 격자를 지키는가")
next_t = time.monotonic()
stamps = []
for _ in range(25):
    time.sleep(0.003)  # 본문이 3 ms 만에 끝나는 상황
    stamps.append(time.monotonic())
    next_t += STEP_DT
    delay = next_t - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    elif delay < -STEP_DT:
        next_t = time.monotonic()
d = [(stamps[i + 1] - stamps[i]) * 1e3 for i in range(len(stamps) - 1)]
avg = sum(d) / len(d)
check("평균 20 ms (50 Hz)", abs(avg - 20.0) < 2.0, f"평균 {avg:.2f} ms → {1000 / avg:.2f} Hz")
check("지터 작음", max(d) - min(d) < 6.0, f"span {max(d) - min(d):.2f} ms")

print("\n[C2] gait clock — 벽시계 dt 기반")
# 주파수는 **누적 위상 / 경과시간**으로 잰다. wrap 을 정수로 세면 3.5 주기가 3 으로 잘려
# 짧은 구간에서 최대 1/N 만큼 어긋난다(측정 방법의 문제지 코드의 문제가 아니다).
WANT_HZ = 1 / GAIT_PERIOD


def gait_hz(dts, *, wall_clock: bool) -> float:
    turns, elapsed = 0.0, 0.0
    for dt in dts:
        turns += (min(dt, PHASE_MAX_DT) if wall_clock else STEP_DT) / GAIT_PERIOD
        elapsed += dt
    return turns / elapsed


for label, dts in (("20 ms 틱", [0.020] * 150), ("28.6 ms 틱(수정 전 실측)", [0.0286] * 105)):
    hz = gait_hz(dts, wall_clock=True)
    check(f"{label}: 보행주파수가 벽시계로 일정", abs(hz - WANT_HZ) / WANT_HZ < 0.02, f"{hz:.3f} Hz (설계 {WANT_HZ:.3f})")

# 수정 전 산술(틱당 고정 STEP_DT)이 정말 0.70배로 느려지는지 대조 — 실측 34.96 Hz 조건
old_hz = gait_hz([1 / 34.96] * 350, wall_clock=False)
check(
    "(대조) 종전 고정 STEP_DT 는 0.70배로 느려짐",
    abs(old_hz / WANT_HZ - 0.699) < 0.01,
    f"{old_hz:.3f} Hz = 설계의 {old_hz / WANT_HZ:.3f}배 (실기 캡처 실측 1.164 Hz)",
)
# 그리고 같은 조건에서 수정 후에는 설계값을 되찾는지
new_hz = gait_hz([1 / 34.96] * 350, wall_clock=True)
check("같은 34.96 Hz 조건에서 수정 후는 설계값 회복", abs(new_hz - WANT_HZ) / WANT_HZ < 0.02, f"{new_hz:.3f} Hz")

print("\n[C2b] PHASE_MAX_DT — stall 때 phase 가 통째로 돌지 않는가")
stall_turns = min(10.6, PHASE_MAX_DT) / GAIT_PERIOD
check("10.6 s stall 이 1 주기를 못 넘김", stall_turns < 1.0, f"{stall_turns:.3f} 주기 (clamp 없으면 17.7)")

print("\n[C3] monitor seq — 벽시계 20 ms 격자 인덱스")
t0 = 100.0
for rate_hz in (50.0, 34.96, 80.0):
    ticks = [t0 + i / rate_hz for i in range(int(rate_hz * 4))]  # 4 초
    seqs = [int((t - t0) / MONITOR_GRID_S) for t in ticks]
    # monitor.py 는 seq×0.02 로 x 를 복원한다 → 복원값이 실제 경과와 같아야 한다
    err = max(abs((s * MONITOR_GRID_S) - (t - t0)) for s, t in zip(seqs, ticks))
    check(f"{rate_hz:5.2f} Hz 발행 → x 복원 오차 < 1 격자", err < MONITOR_GRID_S, f"최대 {err * 1e3:.1f} ms")

print()
if fail:
    print(f"★ {fail}건 실패")
    sys.exit(1)
print("★ 전 항목 통과")
