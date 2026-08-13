# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""biped_leg GUI chirp 캡처(.npz) → PACE 데이터셋(.pt) 변환 + 품질/지연 판정.

``gui_controller.py``의 "Chirp (sysid capture)"가 남긴 50 Hz 캡처를 ``fit_bipedleg.py``가 먹는
step-동기 포맷(기본 200 Hz = 학습 플랜트 물리 그리드, ``--rate``로 변경)으로 바꾼다.
Isaac Sim은 띄우지 않는다 (torch만 필요).

하는 일 (``convert_capture_to_pt.py``(go2)의 biped_leg 판):

1. **균일 그리드 정렬** — 명령(50 Hz 발행시각)과 실기 TELEM(≈50 Hz 도착시각)이 비동기이므로,
   물리 그리드(기본 200 Hz)에 명령은 ZOH(브리지가 마지막 ACT를 유지, slew 기본 0), 실측은 선형보간으로 얹는다.
2. **foot↔calf 커플링 해제** — 캡처의 foot 항은 raw각(q_foot+q_calf)이다(명령·실측 모두).
   sysid env(``foot_coupling=False``)는 관절각 공간이므로 관절각으로 변환한다:
   ``q_foot_joint = q_raw − q_calf(실측)``, ``des_foot_joint = raw_cmd − q_calf(실측)``.
   ⚠ 남는 모델 갭(1차 적합에서 감수): 실기 foot PD의 kd는 (q̇_f+q̇_c)에 걸리는데 재생은 q̇_f만
   보고, foot 모터 토크의 calf 전치(τ_c += τ_f)도 재생에 없다 — calf·foot 식별에 오차 유입 가능.
3. **명령/상태 시간축 지연 판정** — go2에서 104 ms 어긋남이 관성·점성을 ~100× 오염시킨 전례
   (``convert_capture_to_pt.py`` 참조). PD 법칙 ``tau = kp(des−q) − kd·q̇``이 tau_real과 가장 잘
   맞는 시간이동으로 잰다(커플링 오염이 없는 hip/thigh 4관절만 사용). ``--cmd_lag_ms auto``로
   자동 보정, 기본 0.0 = 보정 없음(판정만).

실행::

    # 판정만
    /home/user/miniconda3/envs/isaac-6.0/bin/python3.12 scripts/real2sim/convert_gui_chirp_bipedleg.py \\
        --captures 'data/bipedleg_gui/chirp_gui_2026*.npz'
    # 변환 저장
    ... --out_dir data/bipedleg_real
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

GRID_HZ = 200.0  # 기본 = 학습 플랜트 물리 그리드(200 Hz). sysid env SYSID_RATE_HZ와 일치 필수.
# leg-major (calf, foot) 커플링 쌍 — gui_controller._COUPLED_CALF_FOOT와 동일.
COUPLED_CALF_FOOT = ((2, 3), (6, 7))
# 지연 판정에 쓰는 관절 (커플링 전치 오염이 없는 hip/thigh)
LAG_JOINTS = (0, 1, 4, 5)
JOINT_ORDER_FULL = [
    "HL_hip_joint",
    "HL_thigh_joint",
    "HL_calf_joint",
    "HL_foot_joint",
    "HR_hip_joint",
    "HR_thigh_joint",
    "HR_calf_joint",
    "HR_foot_joint",
]


def zoh(t_src: np.ndarray, v_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """ZOH 샘플링 — 각 grid 시각에 대해 '마지막으로 발행된' 값을 취한다."""
    idx = np.searchsorted(t_src, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 1)
    return v_src[idx]


def lininterp(t_src: np.ndarray, v_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_grid, t_src, v_src[:, j]) for j in range(v_src.shape[1])], axis=1)


def estimate_cmd_lag_ms(
    t_cmd, q_cmd, t_grid, q_meas, dq_meas, tau_meas, kp, kd, scan_ms: float = 100.0
) -> tuple[float, float]:
    """PD 정합성으로 명령 시간축 지연을 추정한다 (양수 = 명령 타임스탬프가 응답보다 앞섬).

    Returns:
        (best_lag_ms, best_corr) — hip/thigh 4관절 합산 상관이 최대가 되는 시프트.
    """
    best = (0.0, -2.0)
    for lag_ms in np.arange(-scan_ms, scan_ms + 1e-6, 2.0):
        des = zoh(t_cmd + lag_ms * 1e-3, q_cmd, t_grid)
        num = den_p = den_m = 0.0
        for j in LAG_JOINTS:
            pred = kp[j] * (des[:, j] - q_meas[:, j]) - kd[j] * dq_meas[:, j]
            meas = tau_meas[:, j]
            p = pred - pred.mean()
            m = meas - meas.mean()
            num += float((p * m).sum())
            den_p += float((p * p).sum())
            den_m += float((m * m).sum())
        if den_p > 0 and den_m > 0:
            c = num / np.sqrt(den_p * den_m)
            if c > best[1]:
                best = (float(lag_ms), c)
    return best


def convert(
    path: Path, out_dir: Path | None, cmd_lag_ms: str, rate: float = GRID_HZ, keep_raw_foot: bool = False
) -> None:
    d = np.load(path, allow_pickle=True)
    if bool(d["aborted"]):
        print(f"[skip] {path.name}: aborted 캡처")
        return
    if "t_real" not in d.files or len(d["t_real"]) < 100:
        print(f"[skip] {path.name}: 실기 TELEM 스트림 없음/부족")
        return

    t_cmd = d["t_cmd"].astype(np.float64)
    q_cmd = d["q_cmd"].astype(np.float64)
    t_real = d["t_real"].astype(np.float64)
    q_real = d["q_real"].astype(np.float64)
    dq_real = d["dq_real"].astype(np.float64)
    tau_real = d["tau_real"].astype(np.float64)
    kp = d["kp"].astype(np.float64)
    kd = d["kd"].astype(np.float64)

    # 겹치는 구간만 (real 스트림은 램프 구간을 포함하므로 cmd(스윕 전용) 창으로 잘린다)
    t0 = max(t_cmd[0], t_real[0])
    t1 = min(t_cmd[-1], t_real[-1])
    n = int((t1 - t0) * rate)
    t_grid = t0 + np.arange(n) / rate

    q_meas = lininterp(t_real, q_real, t_grid)
    dq_meas = lininterp(t_real, dq_real, t_grid)
    tau_meas = lininterp(t_real, tau_real, t_grid)

    # 품질: 수신률/최대 공백/추종 (⚠ 그리드 rate 파라미터와 이름 충돌 금지)
    real_in = (t_real >= t0) & (t_real <= t1)
    real_rate_hz = real_in.sum() / (t1 - t0)
    max_gap_ms = float(np.diff(t_real[real_in]).max() * 1e3) if real_in.sum() > 2 else float("nan")

    # 지연 판정 (관절각 공간 필요 없음 — hip/thigh는 커플링 무관)
    lag_ms, corr = estimate_cmd_lag_ms(t_cmd, q_cmd, t_grid, q_meas, dq_meas, tau_meas, kp, kd)
    applied = 0.0
    if cmd_lag_ms == "auto":
        applied = lag_ms
    elif float(cmd_lag_ms) != 0.0:
        applied = float(cmd_lag_ms)
    des = zoh(t_cmd + applied * 1e-3, q_cmd, t_grid)

    # foot 좌표 처리 — 두 규약:
    #  keep_raw_foot=True (커플링 재생 적합용, 권장): foot 명령·실측을 raw(엔코더) 그대로 둔다.
    #    sysid env(foot_coupling=True)가 raw 구동+전치를 재생하고 채점도 raw끼리 한다.
    #  False (구 관절각 규약): foot을 관절각으로 환산 — 커플링 없는 재생용(모델 갭 있음).
    if keep_raw_foot:
        q_joint = q_meas.copy()
        des_joint = des.copy()
    else:
        q_joint = q_meas.copy()
        des_joint = des.copy()
        for c, f in COUPLED_CALF_FOOT:
            des_joint[:, f] = des[:, f] - q_meas[:, c]
            q_joint[:, f] = q_meas[:, f] - q_meas[:, c]

    track_rms = np.sqrt(np.mean((des_joint - q_joint) ** 2, axis=0))
    pp = q_joint.max(axis=0) - q_joint.min(axis=0)
    print(
        f"[{path.name}] {t1 - t0:.1f}s  real {real_rate_hz:.1f}Hz(max gap {max_gap_ms:.0f}ms)  "
        f"lag={lag_ms:+.0f}ms(corr {corr:.3f}, 적용 {applied:+.0f}ms)\n"
        f"    관절 p-p [rad]: " + " ".join(f"{v:.2f}" for v in pp) + "\n"
        "    추종 RMS [rad]: " + " ".join(f"{v:.3f}" for v in track_rms)
    )
    if abs(lag_ms) > 30.0 and applied == 0.0:
        print(f"    ⚠ 지연 {lag_ms:+.0f}ms > 적합기 delay 상한(20ms) — --cmd_lag_ms auto 권장 (go2 104ms 전례)")

    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + ".pt")
    torch.save(
        {
            "time": torch.arange(n, dtype=torch.float32) / rate,
            "dof_pos": torch.from_numpy(q_joint).float(),
            "des_dof_pos": torch.from_numpy(des_joint).float(),
            "kp": torch.from_numpy(kp).float(),
            "kd": torch.from_numpy(kd).float(),
            "joint_order": JOINT_ORDER_FULL,
            "meta": {
                "source": "real_gui",
                "capture": str(path),
                "rate_hz": rate,
                "f0_hz": float(d["f0_hz"]),
                "f1_hz": float(d["f1_hz"]),
                "duration_s": float(d["duration_s"]),
                "amplitude_scale": float(d["amplitude_scale"]),
                "cmd_lag_applied_ms": applied,
                "cmd_lag_estimated_ms": lag_ms,
                "coupling_converted": not keep_raw_foot,
            },
        },
        out,
    )
    print(f"    → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="biped_leg GUI chirp 캡처 → PACE .pt 변환")
    parser.add_argument("--captures", required=True, help="npz 경로 glob (따옴표로 감쌀 것)")
    parser.add_argument("--out_dir", default=None, help="저장 디렉토리. 생략 시 판정만")
    parser.add_argument("--cmd_lag_ms", default="0.0", help="명령 시간축 보정 [ms]. 'auto'=추정값 적용")
    parser.add_argument(
        "--rate", type=float, default=GRID_HZ, help="출력 그리드 [Hz] — sysid env SYSID_RATE_HZ와 일치 필수"
    )
    parser.add_argument(
        "--keep_raw_foot", action="store_true", help="foot을 raw(엔코더) 그대로 저장 — 커플링 재생 적합용"
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.captures))
    if not files:
        raise SystemExit(f"매칭되는 캡처가 없다: {args.captures}")
    out_dir = Path(args.out_dir) if args.out_dir else None
    for f in files:
        convert(Path(f), out_dir, args.cmd_lag_ms, rate=args.rate, keep_raw_foot=args.keep_raw_foot)


if __name__ == "__main__":
    main()
