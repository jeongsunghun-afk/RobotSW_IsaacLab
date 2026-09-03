# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""참조 모션 클립 자체의 보행 종류를 분류한다.

정책이 참조 상태에서 리셋돼 그 걸음을 이어받는 것이라면, 데이터셋에 gallop 계열이 있어야 한다.
없다면 gallop 은 정책이 스스로 만든 것이다.
"""

import glob
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, "/home/lgb/IsaacLab-6.0/_workspace/leg")
from gait_classify import LEGS, _circ_mean_frac, _classify, _inst_phase  # noqa: E402

D = "/home/lgb/IsaacLab-6.0/source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/imitation/new_smr_leg_pkl"


def main():
    for p in sorted(glob.glob(f"{D}/*.pkl")):
        with open(p, "rb") as f:
            m = pickle.load(f)
        if isinstance(m, dict):
            keys = list(m.keys())
            arr = None
            for k in ("frames", "joint_pos", "dof_pos", "qpos", "jpos"):
                if k in m:
                    arr = np.asarray(m[k], dtype=float)
                    break
            fps = float(m.get("fps", m.get("FPS", 50.0)))
        else:
            arr, fps, keys = np.asarray(m, dtype=float), 50.0, "(array)"
        name = pathlib.Path(p).stem
        if arr is None:
            print(f"{name:28s} keys={keys}")
            continue
        # 관절 이름을 못 찾으면 USD 순서를 가정할 수 없으므로 건너뛴다
        names = (m.get("dof_names") or m.get("joint_names")) if isinstance(m, dict) else None
        if names is None:
            print(f"{name:28s} shape={arr.shape} fps={fps}  (joint_names 없음 — 매핑 불가)")
            continue
        names = [str(s) for s in names]
        try:
            idx = {lg: names.index(f"{lg}_thigh_joint") for lg in LEGS}
        except ValueError:
            print(f"{name:28s} (thigh 관절명 불일치: {names[:4]})")
            continue
        OFF = arr.shape[1] - len(names)  # ★ frames 앞에 root 열이 붙어 있다
        sig = {lg: arr[:, OFF + idx[lg]] for lg in LEGS}
        ph = {lg: _inst_phase(sig[lg], fps) for lg in LEGS}
        freq = float(np.polyfit(np.arange(len(arr)) / fps, ph["FL"], 1)[0]) / (2 * np.pi)
        phi = {lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS}
        print(
            f"{name:28s} {len(arr)/fps:5.2f}s {freq:5.2f}Hz  "
            f"FR {phi['FR']:.2f} HL {phi['HL']:.2f} HR {phi['HR']:.2f}  -> {_classify(phi)}"
        )


if __name__ == "__main__":
    main()
