# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""chirp 캡처가 쓸 수 있는 데이터인지 검증한다 — **적합 전에 반드시 돌릴 것**.

왜 필요한가
-----------
캡처는 GUI 가 **발행한** 목표를 기록한다. 그 뒤 `real_runner` 가 클램프하고 좌표를 변환하는데
**그 결과는 기록되지 않는다.** 2026-08-12 캡처 18 개 중 7 개가 이 구간에서 오염됐고, 그걸
모른 채 PACE 가 적합했다 (근거: reports/_comparisons/pace_bipedleg_foot_coupling_probe/README §15~19).

무엇을 하는가
-------------
드라이버의 PD 법칙을 뒤집어 **드라이버가 실제로 쫓던 목표**를 토크에서 복원하고, 기록된 명령과
비교한다::

    des_eff = q + (tau_meas + kd_ch·k·q̇) / (kp_ch·k)

깨끗한 캡처는 **4 mrad** 수준으로 맞는다. 크게 벌어지면 기록 ≠ 적용이다.

같이 재는 것
------------
* **유효 PD 게인** — `gear_k = 1` 인 hip/thigh 는 계통 보정 C 의 대조군이고, calf/foot 이
  `gear_k^n` 만큼 더 커야 한다. 게인이 규격에서 크게 벗어나면 그 캡처는 적합에 쓰면 안 된다.
* **명령 지연** — meta 의 `cmd_lag_estimated_ms` 가 적용 안 된 채 저장되므로 여기서 스캔한다.

실행::

    python scripts/real2sim/verify_chirp_capture.py data/bipedleg_g2/*.pt
    python scripts/real2sim/verify_chirp_capture.py --exponent 2.0 data/bipedleg_g2/chirp_*.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

#: 드라이버 감속비 오설정 배율 (leg-major). RL_INTERFACE.md §4.
GEAR_K = np.array([1.0, 1.0, 1.5, 1.2] * 2)
#: 실기 드라이버 채널 게인 (실기팀 2026-08-12 지정). `motions.DEFAULT_KP/KD` 와 같은 값.
KP_CH = np.array([100.0, 50.0, 50.0, 20.0] * 2)
KD_CH = np.full(8, 5.0)
NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot", "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]

#: 기록 명령과 복원 목표의 rms 차 [rad] 허용치. 깨끗한 캡처는 0.005 수준이다.
CMD_TOL_RAD = 0.05
#: 유효 게인이 규격 대비 이 범위를 벗어나면 경고 (계통 보정 C 를 감안한 폭).
GAIN_LO, GAIN_HI = 0.70, 1.30


def _load(path: Path) -> dict | None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "tau_meas" not in d:
        print(f"{path.name}: tau_meas 없음 — 검증 불가 (변환 전 데이터셋?)")
        return None
    return d


def _best_lag(q, des, tau, dq, span: int = 25) -> int:
    """명령 지연을 스캔한다 — meta 의 추정치는 적용돼 있지 않다."""
    best, best_lag = -np.inf, 0
    sl = slice(80, -80)
    for lag in range(-span, span + 1):
        dd = np.roll(des, lag, axis=0)
        score = 0.0
        for i in range(8):
            pred = KP_CH[i] * GEAR_K[i] * (dd[:, i] - q[:, i]) + KD_CH[i] * GEAR_K[i] * (-dq[:, i])
            y = tau[sl, i]
            p = pred[sl]
            if p.std() < 1e-9:
                continue
            score += 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if score > best:
            best, best_lag = score, lag
    return best_lag


def verify(path: Path, exponent: float) -> tuple[bool, bool]:
    """Returns:
    ``(cmd_ok, gain_ok)`` — 명령 무결성, 유효게인 규격 부합.
    """
    d = _load(path)
    if d is None:
        return False, False
    q = np.asarray(d["dof_pos"])
    des0 = np.asarray(d["des_dof_pos"])
    tau = np.asarray(d["tau_meas"])
    t = np.asarray(d["time"])
    dq = np.gradient(q, t, axis=0)
    meta = d["meta"]
    lag = _best_lag(q, des0, tau, dq)
    des = np.roll(des0, lag, axis=0)
    sl = slice(80, -80)

    print(f"\n{'=' * 82}")
    print(
        f"{path.name}   amp={meta.get('amplitude_scale', '?')}  "
        f"{meta.get('f0_hz', '?')}→{meta.get('f1_hz', '?')} Hz   최적 지연 {lag} step"
    )
    print(f"{'=' * 82}")
    print(f"  {'joint':10s} {'명령 불일치':>12s} {'유효 kp':>9s} {'기대':>8s} {'비':>7s} {'R2':>7s}")

    cmd_ok = gain_ok = True
    for i in range(8):
        kp_ch_eff = KP_CH[i] * GEAR_K[i]
        des_eff = q[:, i] + (tau[:, i] + KD_CH[i] * GEAR_K[i] * dq[:, i]) / kp_ch_eff
        mism = (des_eff - des[:, i])[sl].std()
        X = np.column_stack([(des[:, i] - q[:, i])[sl], -dq[sl, i], np.ones(len(t))[sl]])
        c, *_ = np.linalg.lstsq(X, tau[sl, i], rcond=None)
        r2 = 1 - ((tau[sl, i] - X @ c) ** 2).sum() / max(((tau[sl, i] - tau[sl, i].mean()) ** 2).sum(), 1e-12)
        kp_eff = c[0] * GEAR_K[i]
        # 기대 유효게인 = kp_ch·gear_k^n (README §19). gear_k=1 인 hip/thigh 는 n 과 무관해
        # 계통 보정 C 의 대조군이 된다 — 그 둘의 비가 0.9 근처면 정상이다.
        expect = KP_CH[i] * GEAR_K[i] ** exponent
        ratio = kp_eff / expect
        flag = ""
        if mism > CMD_TOL_RAD:
            flag += " ← 명령 오염"
            cmd_ok = False
        if not (GAIN_LO <= ratio <= GAIN_HI):
            flag += " ← 게인 이탈"
            gain_ok = False
        print(f"  {NAMES[i]:10s} {mism:12.4f} {kp_eff:9.1f} {expect:8.1f} {ratio:7.2f} {r2:7.3f}{flag}")

    verdict = (
        "PASS — 적합에 사용 가능"
        if (cmd_ok and gain_ok)
        else (
            "*** FAIL — 명령이 기록과 다르다 (복원 필요) ***"
            if not cmd_ok
            else "△ 명령은 정상이나 게인이 규격 이탈 — README §19 의 2차 감쇠"
        )
    )
    print(f"  → {verdict}")
    return cmd_ok, gain_ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="검증할 .pt 캡처 (변환 후, tau_meas 포함)")
    p.add_argument("--exponent", type=float, default=2.0, help="게인 지수 n (기본 2.0, README §19)")
    a = p.parse_args()
    results = [(pp, *verify(pp, a.exponent)) for pp in a.paths]
    cmd_good = [pp for pp, c, _ in results if c]
    both = [pp for pp, c, g in results if c and g]
    print(f"\n{'=' * 82}")
    print(f"명령 무결 {len(cmd_good)}/{len(results)}   ·   명령+게인 모두 정상 {len(both)}/{len(results)}")
    print("\n명령 무결 (복원 없이 쓸 수 있음):")
    for pp in cmd_good:
        print(f"   {'★' if pp in both else ' '} {pp}")
    print("   ★ = 게인도 규격 안")
    return 0 if cmd_good else 1


if __name__ == "__main__":
    sys.exit(main())
