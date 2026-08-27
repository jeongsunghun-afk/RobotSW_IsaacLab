# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PACE fit 결과(mean_*.pt) → GUI 적용용 플랜트 파라미터 json 내보내기.

``fit_bipedleg.py``가 저장하는 ``mean_*.pt``는 **이미 물리 단위**다
(``save_checkpoint``가 ``_params_to_sim_params`` 변환 후 저장 — 이중 변환 금지).
33개 = armature(8) + viscous(8) + coulomb(8) + bias(8) + delay(1), **leg-major** 순서.

sim 플랜트에 적용 가능한 armature/viscous/coulomb 24개만 내보낸다:

* encoder bias — PACE 채점용 측정 오프셋이라 플랜트 물성이 아님 (참고값으로만 기록).
* delay — live 모드 DCMotor에는 지연 버퍼가 없어 적용 불가 (참고값으로만 기록).

실행::

    /home/user/miniconda3/envs/isaac-6.0/bin/python3.12 scripts/real2sim/export_pace_params.py \\
        --mean logs/pace/bipedleg_real/26_08_12_22-14-44/mean_182.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

JOINT_NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot", "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
NUM_JOINTS = len(JOINT_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE mean_*.pt → 플랜트 파라미터 json")
    parser.add_argument("--mean", required=True, help="fit_bipedleg.py 산출 mean_*.pt 경로 (물리 단위 33개)")
    parser.add_argument("--out", default="data/bipedleg_pace_params.json", help="출력 json 경로")
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="좌우(HL/HR)를 평균해 대칭화한다. **`rga.py` 에 반영한 값과 맞추려면 필수** — "
        "0819 적합의 viscous 좌우비(hip 3.24 · foot 3.54)는 캡처들이 kd 를 공유해 생긴 축퇴 "
        "아티팩트이고(README §37-e), `rga.py` 는 좌우평균을 싣고 있다(§38-c). 이 플래그 없이 "
        "내보내면 GUI 오버라이드가 학습 플랜트와 **다른 로봇**을 sim 에 적용한다.",
    )
    args = parser.parse_args()

    mean_path = Path(args.mean)
    sim = torch.load(mean_path, map_location="cpu").double()
    if sim.numel() != 4 * NUM_JOINTS + 1:
        raise SystemExit(f"기대 33개 파라미터, 실제 {sim.numel()}개: {mean_path}")
    if args.symmetrize:
        half = NUM_JOINTS // 2
        for blk in range(4):  # armature / viscous / coulomb / bias — delay 는 스칼라라 제외
            lo = blk * NUM_JOINTS
            mean_lr = 0.5 * (sim[lo : lo + half] + sim[lo + half : lo + NUM_JOINTS])
            sim[lo : lo + half] = mean_lr
            sim[lo + half : lo + NUM_JOINTS] = mean_lr
        print("[export_pace_params] 좌우 평균으로 대칭화 (rga.py 반영값과 동일 규칙)")

    n = NUM_JOINTS
    payload = {
        "joint_order": JOINT_NAMES,
        "armature": {nm: round(float(sim[i]), 6) for i, nm in enumerate(JOINT_NAMES)},
        "viscous": {nm: round(float(sim[n + i]), 6) for i, nm in enumerate(JOINT_NAMES)},
        "coulomb": {nm: round(float(sim[2 * n + i]), 6) for i, nm in enumerate(JOINT_NAMES)},
        "source_run": str(mean_path),
        "bias_unapplied": [round(float(sim[3 * n + i]), 6) for i in range(n)],
        "delay_ms_unapplied": round(float(sim[4 * n]) * 5.0, 2),  # 1 step = 5 ms @200 Hz
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[export_pace_params] {mean_path} → {out}")
    for key in ("armature", "viscous", "coulomb"):
        print(f"  {key}: " + " ".join(f"{payload[key][nm]:.4f}" for nm in JOINT_NAMES))


if __name__ == "__main__":
    main()
