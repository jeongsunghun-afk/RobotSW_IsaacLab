# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""`gait_classify._classify` 를 **정답을 아는 합성 신호**로 검증한다.

trot 과 pace 는 네 다리 위상차가 둘 다 (0.5, 0.5) 로 같고, 갈리는 것은 **어느 쌍이 함께
움직이는가** 뿐이다(trot=대각, pace=동측). 이 구분이 틀리면 "정책이 pace 로 붕괴했다" 같은
결론이 통째로 뒤집히므로, 실측 데이터로만 확인하지 말고 정답이 있는 신호로 검증한다.

위상 규약: FL 을 기준(0)으로 한 사이클 분율.

    보행     phi_FR  phi_HL  phi_HR   함께 움직이는 쌍
    trot      0.5     0.5     0.0     대각 (FL+HR, FR+HL)
    pace      0.5     0.0     0.5     동측 (FL+HL, FR+HR)
    bound     0.0     0.5     0.5     앞쌍, 뒷쌍
    pronk     0.0     0.0     0.0     넷 다
    gallop    0.15    0.5     0.65    앞쌍·뒷쌍이 조금 어긋난 채 함께

실행:  python _workspace/leg/test_gait_classify.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gait_classify import LEGS, _circ_mean_frac, _classify, _inst_phase  # noqa: E402

FS = 50.0  # policy rate [Hz] — 실제 npz 와 같다
DUR_S = 4.0
FREQ_HZ = 2.2  # 실측 보행 주파수대

GAITS = {
    #            phi_FR  phi_HL  phi_HR
    "trot": (0.5, 0.5, 0.0),
    "pace": (0.5, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.5),
    "pronk": (0.0, 0.0, 0.0),
    "gallop": (0.15, 0.5, 0.65),
}


def _synth(phis: tuple[float, float, float], noise: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """네 다리 thigh 신호를 만든다. 실제 관절각처럼 2 고조파를 섞고 잡음을 얹는다."""
    t = np.arange(int(DUR_S * FS)) / FS
    out = {}
    for lg, ph in zip(LEGS, (0.0,) + phis):
        a = 2 * np.pi * (FREQ_HZ * t + ph)
        out[lg] = 0.4 * np.sin(a) + 0.08 * np.sin(2 * a + 0.7) + noise * rng.standard_normal(t.size)
    return out


def _label(sig: dict[str, np.ndarray]) -> tuple[str, dict[str, float]]:
    ph = {lg: _inst_phase(sig[lg], FS) for lg in LEGS}
    phi = {lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS}
    return _classify(phi), phi


def main() -> int:
    rng = np.random.default_rng(0)
    n_fail = 0

    print(f"합성 신호 {FREQ_HZ} Hz · {DUR_S} s · {FS} Hz 표집\n")
    print(f"{'정답':8s}{'잡음':>7s}{'판정':>9s}{'phi_FR':>9s}{'phi_HL':>9s}{'phi_HR':>9s}   결과")
    print("-" * 62)
    for noise in (0.0, 0.02, 0.05):
        for want, phis in GAITS.items():
            got, phi = _label(_synth(phis, noise, rng))
            ok = got == want
            n_fail += not ok
            print(f"{want:8s}{noise:7.2f}{got:>9s}{phi['FR']:9.3f}{phi['HL']:9.3f}{phi['HR']:9.3f}   "
                  f"{'OK' if ok else '**FAIL**'}")

    # ★ trot 과 pace 를 정말 가르는지 — 두 보행은 front/hind 위상차가 (0.5, 0.5) 로 **같다**.
    # 갈리는 것은 대각(diag) vs 동측(ipsi) 뿐이라, 여기서 헷갈리면 표가 통째로 거짓말을 한다.
    print("\n[trot ↔ pace 경계] 대각 위상을 0 에서 0.5 로 밀며 라벨이 언제 넘어가는지")
    for d in np.arange(0.0, 0.51, 0.1):
        # phi_HR = d (대각쌍 어긋남), phi_HL = 0.5 - d 로 두면 d=0 이 trot, d=0.5 가 pace
        got, phi = _label(_synth((0.5, 0.5 - d, d), 0.02, rng))
        print(f"  diag 어긋남 {d:.1f} → {got:8s} (phi_FR {phi['FR']:.2f} phi_HL {phi['HL']:.2f} phi_HR {phi['HR']:.2f})")

    print(f"\n{'전부 통과' if n_fail == 0 else f'{n_fail}건 실패'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
