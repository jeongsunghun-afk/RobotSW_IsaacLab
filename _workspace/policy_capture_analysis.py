"""GUI 정책 실행 기록(`data/bipedleg_gui/policy_*.npz`)의 타이밍·클램프·sim2real 갭 분석.

이 세션의 실기 실험에서 사용자가 보고한 두 증상을 데이터로 판정하기 위해 만들었다.
  ① monitor 에서 real 이 sim 보다 "빨리 지나간다"
  ② X11 렉

판정에 필요한 축만 본다 — 틱 주기(정책 루프가 정말 50 Hz 인가), monitor x축 규약이 가정하는
50 Hz 와의 차이, 클램프 발동, 그리고 두 갈래(sim/real)의 상태·목표 발산.

사용:
    python3 _workspace/policy_capture_analysis.py data/bipedleg_gui/policy_*.npz
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts/real2sim/r2s_biped_leg")
import motions as M  # noqa: E402

# monitor.py 가 x축 재구성에 쓰는 공칭 주기. 두 스트림 모두 이 상수로 그린다.
MONITOR_SAMPLE_PERIOD_S = 0.02
MONITOR_WINDOW_S = 10.0
# 학습 env 의 제어 주기 (`hind_leg_env_cfg` decimation × sim dt).
TRAIN_STEP_DT = 0.02

_ART_FOR_LEGMAJOR = [0, 2, 4, 6, 1, 3, 5, 7]


def _fmt(a: np.ndarray, w: int = 7, p: int = 3) -> str:
    return " ".join(f"{v:{w}.{p}f}" for v in a)


def analyze(path: str) -> None:
    d = np.load(path, allow_pickle=True)
    t = d["t"]
    seq = d["seq"]
    n = len(t)
    dur = float(t[-1] - t[0])
    dt = np.diff(t)
    rate = (n - 1) / dur

    print("=" * 78)
    print(path)
    print("=" * 78)

    # ── 1. 정책 루프 주기 ────────────────────────────────────────────────────
    print("\n[1] 정책 루프 주기")
    print(f"  샘플 {n}  구간 {dur:.1f}s  평균 {rate:.2f} Hz  (학습/설계 = 50 Hz)")
    print(
        f"  dt: mean {dt.mean() * 1e3:.2f} ms · med {np.median(dt) * 1e3:.2f} · "
        f"p05 {np.percentile(dt, 5) * 1e3:.2f} · p95 {np.percentile(dt, 95) * 1e3:.2f} · max {dt.max() * 1e3:.1f}"
    )
    # 큰 stall 은 평균을 오염시키므로 따로 센다.
    stall = dt > 0.1
    if stall.any():
        print(f"  ⚠ 100 ms 초과 stall {stall.sum()}회 (합 {dt[stall].sum():.2f}s = 구간의 {100 * dt[stall].sum() / dur:.1f}%)")
        clean = dt[~stall]
        print(f"    stall 제외 평균 {1 / clean.mean():.2f} Hz")
    # seq 는 매 틱 +1 이므로 누락 여부 확인
    gaps = np.diff(seq)
    if (gaps != 1).any():
        print(f"  ⚠ seq 불연속 {(gaps != 1).sum()}회 (최대 점프 {gaps.max()})")

    # ── 2. monitor x축 규약과의 어긋남 ───────────────────────────────────────
    print("\n[2] monitor x축 (sim 스트림은 이 seq 로 그려진다)")
    r = rate / (1.0 / MONITOR_SAMPLE_PERIOD_S)
    print(f"  monitor 가정 50 Hz vs 실제 {rate:.2f} Hz  →  배율 {r:.3f}")
    print(f"  sim 곡선은 실시간의 {r:.2f}배 폭으로 그려진다 (real 중계는 벽시계 50 Hz 게이트라 1.00배)")
    print(
        f"  10 s 창 왼쪽 끝에서 두 곡선의 어긋남 ≈ {MONITOR_WINDOW_S * (1 - r):.2f} s "
        f"(sim 이 {MONITOR_WINDOW_S / r:.1f}s 분량을 {MONITOR_WINDOW_S:.0f}s 폭에 그림)"
    )

    # ── 3. gait clock (phase 는 틱당 고정량이라 벽시계와 어긋난다) ────────────
    print("\n[3] gait clock")
    ph = d["phase"].astype(float)
    wraps = int(np.sum(np.diff(ph) < -0.5))
    ph_hz_wall = wraps / dur if dur > 0 else 0.0
    print(f"  phase wrap {wraps}회 / {dur:.1f}s → 실제 보행주파수 {ph_hz_wall:.3f} Hz")
    print(f"  설계값 = 1/GAIT_PERIOD, 틱당 STEP_DT({TRAIN_STEP_DT}s) 전진 가정")
    print(f"  → 벽시계 기준 gait 가 설계 대비 {r:.2f}배 느리게 흐른다")

    # ── 4. 클램프 ────────────────────────────────────────────────────────────
    print("\n[4] soft limit 클램프 (articulation order)")
    for nm in ("clamped_sim", "clamped_real"):
        c = d[nm]
        per = c.mean(0) * 100
        print(f"  {nm}: 임의관절 {c.any(1).mean() * 100:6.2f}% of ticks")
        for a in range(8):
            if per[a] > 0.5:
                jn = M.JOINT_NAMES[_ART_FOR_LEGMAJOR.index(a)]
                print(f"      {jn:10s} {per[a]:6.2f}%")

    # 얼마나 넘겼는가 — 클램프 후 값이 한계에 붙어 있는 정도로는 초과량을 모르므로
    # 목표와 한계의 관계만 본다(원 목표는 기록에 없다: 잘린 값이 저장된다).
    print("  ※ 기록된 target 은 **잘린 뒤** 값이라 초과량은 이 파일로 못 잰다 (마스크만 유효)")

    # ── 5. sim vs real 갈래 ──────────────────────────────────────────────────
    print("\n[5] sim / real 갈래 비교 (articulation order)")
    qs, qr = d["q_sim"].astype(float), d["q_real"].astype(float)
    ts_, tr_ = d["target_sim"].astype(float), d["target_real"].astype(float)
    ok = ~(np.isnan(qs).any(1) | np.isnan(qr).any(1))
    print(f"  두 갈래 모두 유효한 틱 {ok.sum()}/{n} ({100 * ok.mean():.1f}%)")
    if ok.sum() > 10:
        dq_ = np.abs(qs[ok] - qr[ok])
        dt_ = np.abs(ts_[ok] - tr_[ok])
        print(f"  |q_sim − q_real|   mean {_fmt(dq_.mean(0))}")
        print(f"                     p95  {_fmt(np.percentile(dq_, 95, axis=0))}")
        print(f"  |tgt_sim − tgt_real| mean {_fmt(dt_.mean(0))}")
        print(f"                     max  {_fmt(dt_.max(0))}")

    # 추종 오차 — 실기가 목표를 얼마나 따라가는가
    okr = ~np.isnan(qr).any(1)
    if okr.sum() > 10:
        err = qr[okr] - tr_[okr]
        print(f"  real 추종오차 (q−target) mean {_fmt(err.mean(0))}")
        print(f"                           std  {_fmt(err.std(0))}")

    # ── 6. TELEM (실기 200 Hz 기록) ──────────────────────────────────────────
    print("\n[6] TELEM (leg-major, tau 는 채널 좌표)")
    tt = d["telem_t"]
    tau = d["telem_tau_lm_channel"].astype(float)
    print(f"  샘플 {len(tt)}  구간 {tt[-1] - tt[0]:.1f}s  {(len(tt) - 1) / (tt[-1] - tt[0]):.2f} Hz")
    print(f"  기록 구간이 정책({dur:.1f}s)보다 {tt[-1] - tt[0] - dur:+.1f}s")
    print("  |tau| 채널 [N·m]   (펌웨어 트립 임계 = 15.0, 채널 상한 84)")
    print(f"    joint      mean    p95    p99    max   >15 비율   최장연속(ms)")
    for j in range(8):
        a = np.abs(tau[:, j])
        over = a > 15.0
        # 최장 연속 초과 구간 (TELEM 5 ms 격자)
        run = mx = 0
        for v in over:
            run = run + 1 if v else 0
            mx = max(mx, run)
        print(
            f"    {M.JOINT_NAMES[j]:10s} {a.mean():6.2f} {np.percentile(a, 95):6.2f} "
            f"{np.percentile(a, 99):6.2f} {a.max():6.2f}  {100 * over.mean():6.2f}%  {mx * 5:8.0f}"
        )
    print("  ※ 50 ms(=10 샘플) 이상 연속 초과가 펌웨어 limp 래치 조건이다")

    # ── 7b. hip 처짐 진단 — 영점 오차인가 중력 처짐인가 ──────────────────────
    #   같은 목표에 대해 sim 과 real 이 각각 얼마나 처지는지 비교한다.
    #   · 오차가 **자세와 무관하게 일정** → 영점(캘리브레이션) 오차 ⇒ offset 이 옳은 처방
    #   · 오차가 **부하(중력토크)에 비례** → PD 처짐/게인 부족 ⇒ offset 은 가동폭만 갉아먹는 미봉
    print("\n[7b] 관절별 처짐 진단 (real2sim 갭)")
    if ok.sum() > 50:
        e_s = qs[ok] - ts_[ok]  # sim 추종오차
        e_r = qr[ok] - tr_[ok]  # real 추종오차
        gap = e_r - e_s  # real 이 sim 보다 더 처진 양
        tau_art_ch = None
        print("    joint      sim오차   real오차   갭(real−sim)  갭std   갭CV   |판정")
        for a in range(8):
            jn = M.JOINT_NAMES[_ART_FOR_LEGMAJOR.index(a)]
            g = gap[:, a]
            cv = abs(g.std() / g.mean()) if abs(g.mean()) > 1e-6 else float("inf")
            # CV(변동계수)가 작으면 상수 바이어스(영점), 크면 부하 의존(중력 처짐)
            verdict = "상수 바이어스→offset 유효" if cv < 0.35 else ("혼합" if cv < 1.0 else "부하의존→offset은 미봉")
            print(
                f"    {jn:10s} {e_s[:, a].mean():+8.4f} {e_r[:, a].mean():+9.4f} "
                f"{g.mean():+11.4f} {g.std():7.4f} {cv:6.2f}  |{verdict}"
            )
        print("    (rad. 음수 = 목표보다 아래. CV = |std/mean|)")
        # hip 은 가동폭이 좁아 offset 여유를 따로 낸다
        for a in (0, 1):
            jn = M.JOINT_NAMES[_ART_FOR_LEGMAJOR.index(a)]
            lo, hi = M.SOFT_LIMITS_RAD[_ART_FOR_LEGMAJOR.index(a)]
            need = -gap[:, a].mean()
            print(
                f"    {jn}: soft limit [{lo:+.3f}, {hi:+.3f}] (폭 {hi - lo:.3f} rad) · "
                f"갭 보정에 필요한 offset {np.degrees(need):+.2f}° → 가동폭의 {100 * abs(need) / (hi - lo):.1f}% 소모"
            )
        _ = tau_art_ch

    # ── 7. 명령 ──────────────────────────────────────────────────────────────
    print("\n[7] 명령/권한")
    for k in ("x_vel", "yaw", "act_scale"):
        v = d[k].astype(float)
        u = np.unique(np.round(v, 3))
        print(f"  {k:10s} min {v.min():6.3f} max {v.max():6.3f}  고유값 {len(u)}개 {u[:8]}")
    po = d["pose_offset"].astype(float)
    print(f"  pose_offset (art) 최종 {_fmt(po[-1])}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
