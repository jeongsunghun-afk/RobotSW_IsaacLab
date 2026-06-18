# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CSV 데이터 로거 — Real2Sim 관절 상태 기록."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import IO

import numpy as np

JOINT_LABELS = ["thigh_r", "thigh_p", "knee_p", "ankle_p", "toe_p"]

# CSV 헤더 (17컬럼)
COLUMNS: list[str] = (
    ["timestamp"] + [f"{j}_{k}" for j in JOINT_LABELS for k in ["pos", "vel", "torque"]] + ["setpoint_applied"]
)

_FLUSH_INTERVAL = 200  # 200행마다 disk flush


class DataLogger:
    """관절 state를 CSV로 저장하는 버퍼드 로거."""

    def __init__(self, log_dir: str = "logs/real2sim") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[list[str]] = []
        self._path: str = ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = str(self._log_dir / f"{ts}.csv")
        self._file: IO[str] | None = open(self._path, "w", newline="")  # noqa: SIM115
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

    def log(self, state: dict, setpoint: np.ndarray | None = None) -> None:
        """state dict에서 한 행을 버퍼에 추가."""
        pos = np.asarray(state.get("pos", np.zeros(5)), dtype=np.float32)
        vel = np.asarray(state.get("vel", np.zeros(5)), dtype=np.float32)
        torque = np.asarray(state.get("torque", np.zeros(5)), dtype=np.float32)
        timestamp = float(state.get("timestamp", 0.0))

        row: list[str] = [f"{timestamp:.6f}"]
        for i in range(5):
            row += [f"{pos[i]:.6f}", f"{vel[i]:.6f}", f"{torque[i]:.6f}"]
        row.append(str(setpoint.tolist()) if setpoint is not None else "[]")

        self._buffer.append(row)
        if len(self._buffer) >= _FLUSH_INTERVAL:
            self._flush()

    def _flush(self) -> None:
        if self._writer is not None and self._buffer:
            self._writer.writerows(self._buffer)
            self._buffer.clear()
            if self._file is not None:
                self._file.flush()

    def save(self) -> str:
        """버퍼를 비우고 파일 경로 반환."""
        self._flush()
        if self._file is not None:
            self._file.close()
            self._file = None
        return self._path

    def get_path(self) -> str:
        return self._path

    def __del__(self) -> None:
        try:
            self.save()
        except Exception:
            pass
