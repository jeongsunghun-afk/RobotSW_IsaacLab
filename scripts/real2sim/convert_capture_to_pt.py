# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""실기 chirp 캡처(.npz) → PACE 데이터셋(chirp_data.pt) 변환 + 품질 판정.

``r2s_go2/chirp_collector.py``(시스템 python3.10, torch 없음)가 남긴 원시 스트림을 읽어
Isaac conda(torch)에서 PACE가 먹는 포맷으로 바꾼다. Isaac Sim은 띄우지 않는다.

하는 일:

1. **균일 그리드 정렬** — 명령/상태가 서로 다른 시각에 비동기로 찍히므로, 500 Hz 균일 그리드에
   명령은 ZOH(실기 모터가 마지막 명령을 유지하는 것과 동일), 실측은 선형보간으로 얹는다.
2. **품질 판정** — 실효 발행률/수신률, 드롭, 추종오차.
3. **리그 공진 판정** — chirp의 순시 주파수별로 base 각속도를 묶어 본다. 특정 주파수에서 base가
   크게 흔들리면 **그 구간은 관절이 아니라 매다는 리그를 측정한 것**이고, PACE의 고정 base 전제가
   깨져 식별이 오염된다. 논문이 매단 ANYmal의 chirp를 2 Hz로 제한한 이유다.

실행::

    ./isaaclab.sh -p scripts/real2sim/convert_capture_to_pt.py --capture data/go2_real/chirp_kp25.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pace_sim2real.utils import project_root

# base 각속도가 이 값을 넘는 주파수 대역은 리그 공진으로 본다 [rad/s].
# 매단 로봇은 이상적으로 base가 정지해 있어야 한다(PACE의 fix_root_link 전제).
RIG_RESONANCE_GYRO_WARN = 0.5


def zoh(t_grid: np.ndarray, t_src: np.ndarray, values: np.ndarray) -> np.ndarray:
    """영차 홀드(ZOH) 재샘플 — 각 격자 시각에 대해 그 이전 마지막 값을 취한다.

    명령 신호에 쓴다. 실기 모터는 다음 명령이 올 때까지 마지막 목표각을 유지하므로 ZOH가 물리적으로 맞다.

    Args:
        t_grid: 균일 격자 시각 [s], shape (T,).
        t_src: 원본 시각 [s], 오름차순, shape (N,).
        values: 원본 값, shape (N, D).

    Returns:
        격자에 얹은 값, shape (T, D).
    """
    idx = np.searchsorted(t_src, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 1)
    return values[idx]


def lerp(t_grid: np.ndarray, t_src: np.ndarray, values: np.ndarray) -> np.ndarray:
    """선형보간 재샘플 — 연속 물리량(엔코더 각도)에 쓴다.

    Args:
        t_grid: 균일 격자 시각 [s], shape (T,).
        t_src: 원본 시각 [s], 오름차순, shape (N,).
        values: 원본 값, shape (N, D).

    Returns:
        격자에 얹은 값, shape (T, D).
    """
    return np.stack([np.interp(t_grid, t_src, values[:, d]) for d in range(values.shape[1])], axis=1)


def report_rig_resonance(t: np.ndarray, gyro: np.ndarray, f0: float, f1: float, duration: float) -> float:
    """chirp 순시 주파수별 base 각속도를 표로 뽑아 리그 공진을 판정한다.

    선형 chirp의 순시 주파수는 ``f(t) = f0 + (f1 - f0) * t / duration`` 이다. 시간축을 그대로
    주파수축으로 읽을 수 있다.

    Args:
        t: 상태 수신 시각 [s].
        gyro: base 각속도 [rad/s], shape (N, 3).
        f0: chirp 시작 주파수 [Hz].
        f1: chirp 종료 주파수 [Hz].
        duration: chirp 길이 [s].

    Returns:
        공진으로 의심되는 최저 주파수 [Hz]. 없으면 ``f1``.
    """
    freq = f0 + (f1 - f0) * np.clip(t / duration, 0.0, 1.0)
    mag = np.linalg.norm(gyro, axis=1)

    edges = np.linspace(f0, f1, 21)
    print("\n=== 리그 공진 점검 (chirp 순시 주파수 대역별 base 각속도) ===")
    print(f"{'대역 [Hz]':>16s} {'|w| mean':>10s} {'|w| max':>10s}")
    first_bad = f1
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freq >= lo) & (freq < hi)
        if not sel.any():
            continue
        m_mean, m_max = float(mag[sel].mean()), float(mag[sel].max())
        flag = ""
        if m_max > RIG_RESONANCE_GYRO_WARN:
            flag = "  ← 리그 흔들림"
            first_bad = min(first_bad, float(lo))
        print(f"{lo:7.2f}–{hi:6.2f} {m_mean:10.3f} {m_max:10.3f}{flag}")

    if first_bad < f1:
        print(
            f"\n⚠ {first_bad:.2f} Hz 부터 base가 크게 움직인다(|w| > {RIG_RESONANCE_GYRO_WARN} rad/s).\n"
            "  그 위 구간은 관절이 아니라 **매다는 리그**를 측정한 것이다 — PACE의 고정 base 전제가 깨진다.\n"
            f"  --max_frequency 를 {first_bad:.1f} Hz 아래로 낮춰 다시 수집할 것.\n"
            "  (논문도 같은 이유로 매단 ANYmal의 chirp를 2 Hz로 제한했다.)"
        )
    else:
        print(f"\n✅ 전 대역에서 base 각속도 < {RIG_RESONANCE_GYRO_WARN} rad/s — 고정 base 전제가 유지된다.")
    return first_bad


def main() -> None:
    p = argparse.ArgumentParser(description="실기 chirp 캡처(.npz)를 PACE 데이터셋(.pt)으로 변환한다.")
    p.add_argument("--capture", type=str, required=True, help="chirp_collector.py가 만든 .npz 경로.")
    p.add_argument("--out", type=str, default=None, help="출력 .pt 경로. 기본은 캡처와 같은 이름의 .pt.")
    p.add_argument("--rate", type=float, default=None, help="재샘플 격자 주파수 [Hz]. 기본은 캡처의 발행률.")
    p.add_argument("--trim", type=float, default=0.0, help="앞뒤로 잘라낼 시간 [s] (과도구간 제거).")
    args = p.parse_args()

    capture_path = Path(args.capture)
    if not capture_path.is_absolute():
        capture_path = project_root() / capture_path
    raw = np.load(capture_path, allow_pickle=False)

    if bool(raw["aborted"]):
        print("⚠ 이 캡처는 중단(abort)된 부분 데이터다. 식별에 쓰기 전에 원인을 확인할 것.")

    rate = float(args.rate if args.rate else raw["rate_hz"])
    f0, f1 = float(raw["f0_hz"]), float(raw["f1_hz"])
    duration = float(raw["duration_s"])

    t_cmd, q_cmd = raw["t_cmd"], raw["q_cmd"]
    t_state, q_state = raw["t_state"], raw["q"]

    # ---- 원시 스트림 품질 ----
    cmd_dt, st_dt = np.diff(t_cmd), np.diff(t_state)
    print("=== 원시 스트림 ===")
    print(f"명령 {len(t_cmd):6d} 샘플  실효 {1 / cmd_dt.mean():7.1f} Hz  max gap {cmd_dt.max() * 1e3:6.1f} ms")
    print(f"상태 {len(t_state):6d} 샘플  실효 {1 / st_dt.mean():7.1f} Hz  max gap {st_dt.max() * 1e3:6.1f} ms")
    if 1 / cmd_dt.mean() < 0.9 * rate:
        print(f"⚠ 발행률이 목표({rate:.0f} Hz)의 90% 미만이다 — 루프가 굶었다. 데이터 신뢰도가 떨어진다.")

    # ---- 리그 공진 (상태 시각 기준) ----
    report_rig_resonance(t_state - t_state[0], raw["imu_gyro"], f0, f1, duration)

    # ---- 균일 격자 정렬 ----
    # 두 스트림이 모두 존재하는 구간에서만 격자를 만든다.
    t_start = max(t_cmd[0], t_state[0]) + args.trim
    t_end = min(t_cmd[-1], t_state[-1]) - args.trim
    num_steps = int((t_end - t_start) * rate)
    t_grid = t_start + np.arange(num_steps) / rate

    des_dof_pos = zoh(t_grid, t_cmd, q_cmd)  # 명령 = ZOH (모터가 마지막 목표를 유지)
    dof_pos = lerp(t_grid, t_state, q_state)  # 엔코더 = 연속량 → 선형보간

    err = np.abs(dof_pos - des_dof_pos)
    print("\n=== 정렬 결과 ===")
    print(f"격자 {num_steps} 스텝 @ {rate:.0f} Hz  ({t_end - t_start:.2f} s)")
    print(f"|q - q_des| mean={err.mean():.4f} max={err.max():.4f} rad")
    if err.mean() < 0.02:
        print("⚠ 추종오차가 거의 없다 — 여기신호가 동역학을 자극하지 못했다. 진폭/주파수를 올릴 것.")

    # ---- 저장 ----
    out_path = Path(args.out) if args.out else capture_path.with_suffix(".pt")
    if not out_path.is_absolute():
        out_path = project_root() / out_path
    torch.save(
        {
            "time": torch.tensor(t_grid - t_grid[0], dtype=torch.float32),
            "dof_pos": torch.tensor(dof_pos, dtype=torch.float32),  # 실기 엔코더 읽음값
            "des_dof_pos": torch.tensor(des_dof_pos, dtype=torch.float32),
            # 재생 시 복원할 게인 (CONTRACT §12.4) — 틀린 게인으로 재생하면 플랜트 파라미터가 편향된다.
            "kp": torch.tensor(raw["kp"], dtype=torch.float32),
            "kd": torch.tensor(raw["kd"], dtype=torch.float32),
            "joint_order": [str(x) for x in raw["joint_order"]],
            "meta": {
                "source": "real",
                "capture": capture_path.name,
                "rate_hz": rate,
                "f0_hz": f0,
                "f1_hz": f1,
                "duration_s": duration,
                "amplitude_scale": float(raw["amplitude_scale"]),
                "aborted": bool(raw["aborted"]),
            },
        },
        out_path,
    )
    print(f"\n저장: {out_path}")
    print("다음: Go2PaceCfg.robot_name='go2_real', datasets/holdout에 이 파일을 등록한 뒤")
    print("      python scripts/real2sim/fit_go2.py --headless --num_envs 4096")


if __name__ == "__main__":
    main()
