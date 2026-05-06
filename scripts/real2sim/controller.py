"""Real2Sim 컨트롤러 GUI.

PyQt5 슬라이더로 R_Skeleton Hind Leg 관절을 실시간 제어.
시뮬레이션과 ZMQ로 통신.

실행 (별도 터미널):
  python scripts/real2sim/controller.py           # ZMQ (기본값)
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
from collections import deque
from pathlib import Path
from typing import cast

import numpy as np

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as e:
    raise ImportError("PyQt5가 설치되지 않았습니다. pip install PyQt5") from e

try:
    import pyqtgraph as pg
except ImportError as e:
    raise ImportError("pyqtgraph가 설치되지 않았습니다. pip install pyqtgraph>=0.13") from e

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from utils.data_logger import DataLogger  # noqa: E402

from utils.zmq_bridge import ControllerTransport, ZMQControllerBridge  # noqa: E402

# ---------------------------------------------------------------------------
# 관절 파라미터 (r2s_hind_leg_env_cfg.py와 동기화)
# ---------------------------------------------------------------------------
JOINT_LABELS = ["thigh_r", "thigh_p", "knee_p", "ankle_p", "toe_p"]
JOINT_LIMITS_DEG: list[tuple[float, float]] = [
    (-60.0, 60.0),
    (-90.0, 90.0),
    (0.0, 120.0),
    (-45.0, 45.0),
    (-30.0, 30.0),
]
JOINT_LIMITS_RAD: list[tuple[float, float]] = [
    (math.radians(lo), math.radians(hi)) for lo, hi in JOINT_LIMITS_DEG
]
NUM_JOINTS = 5

# ---------------------------------------------------------------------------
# 모션 데이터셋 파라미터
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).parent / "smr_hind_leg" / "new_dataset"
FRAME_JOINT_POS_SLICE = slice(7, 12)   # 29차원 프레임에서 joint_pos (rad)
FRAME_JOINT_VEL_SLICE = slice(21, 26)  # 29차원 프레임에서 joint_vel (rad/s)

# 그래프 설정
HISTORY_LEN = 150   # 3초 × 50Hz
RENDER_HZ = 30      # GUI 갱신 주파수
COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]


# ---------------------------------------------------------------------------
# 모션 클립 로더
# ---------------------------------------------------------------------------
def load_motion_clip(path: Path) -> dict:
    """모션 데이터셋 txt(JSON) 파일을 로드하여 딕셔너리로 반환.

    반환 형식:
      {
        "name": str,
        "frame_duration": float,
        "loop_mode": str,
        "frames": np.ndarray (T, 29),
        "joint_pos": np.ndarray (T, 5),   # rad
        "joint_vel": np.ndarray (T, 5),   # rad/s
      }
    """
    with path.open("r") as f:
        raw = json.load(f)

    frames_list: list[list[float]] = raw["Frames"]
    frames = np.array(frames_list, dtype=np.float32)

    return {
        "name": path.stem,
        "frame_duration": float(raw.get("FrameDuration", 1.0 / 60.0)),
        "loop_mode": str(raw.get("LoopMode", "Wrap")),
        "frames": frames,
        "joint_pos": frames[:, FRAME_JOINT_POS_SLICE].copy(),
        "joint_vel": frames[:, FRAME_JOINT_VEL_SLICE].copy(),
    }


# ---------------------------------------------------------------------------
# ZMQ 수신 스레드
# ---------------------------------------------------------------------------
class StateReceiver(QtCore.QThread):
    state_received = QtCore.pyqtSignal(dict)

    def __init__(self, bridge: ControllerTransport) -> None:
        super().__init__()
        self._bridge = bridge
        self._running = True

    def run(self) -> None:
        while self._running:
            state = self._bridge.recv_state(timeout_ms=100)
            if state is not None:
                self.state_received.emit(state)

    def stop(self) -> None:
        self._running = False
        self.wait(500)


# ---------------------------------------------------------------------------
# 메인 윈도우
# ---------------------------------------------------------------------------
class R2SController(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("R2S Controller — R_Skeleton Hind Leg")
        self.resize(1000, 650)

        # Transport 브릿지 (ZMQ)
        self._bridge: ControllerTransport = cast(ControllerTransport, ZMQControllerBridge())

        # 데이터 로거
        self._logger = DataLogger()

        # 링 버퍼 (pos / vel / torque × 5관절)
        self._history: dict[str, deque] = {
            f"{j}_{k}": deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
            for j in JOINT_LABELS
            for k in ["pos", "vel", "torque"]
        }
        self._timestamps: deque = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

        # 현재 setpoint (rad)
        self._current_setpoint = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._connected = False

        # 모션 재생 상태
        self._motion_clip: dict | None = None
        self._motion_frame_idx: int = 0
        self._motion_playing: bool = False

        self._build_ui()
        self._start_receiver()

        # GUI 갱신 타이머 (30Hz)
        self._render_timer = QtCore.QTimer()
        self._render_timer.timeout.connect(self._update_plots)
        self._render_timer.start(1000 // RENDER_HZ)

        # 모션 재생 타이머 (frame_duration에 맞게 동적 설정)
        self._motion_timer = QtCore.QTimer()
        self._motion_timer.timeout.connect(self._advance_motion_frame)

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # 왼쪽: 슬라이더 패널 + 모션 재생 패널을 수직으로 배치
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(300)
        left_vbox = QtWidgets.QVBoxLayout(left_panel)
        left_vbox.setContentsMargins(0, 0, 0, 0)

        # --- 슬라이더 패널 ---
        slider_panel = QtWidgets.QGroupBox("Joint Controller (deg)")
        slider_layout = QtWidgets.QVBoxLayout(slider_panel)

        self._sliders: list[QtWidgets.QSlider] = []
        self._value_labels: list[QtWidgets.QLabel] = []

        for i, (label, (lo, hi)) in enumerate(zip(JOINT_LABELS, JOINT_LIMITS_DEG)):
            row = QtWidgets.QHBoxLayout()
            name_lbl = QtWidgets.QLabel(f"{label}:")
            name_lbl.setFixedWidth(70)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(int(lo * 10))
            slider.setMaximum(int(hi * 10))
            slider.setValue(0)
            slider.setTickInterval(int((hi - lo) * 10 / 4))
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            val_lbl = QtWidgets.QLabel("0.0°")
            val_lbl.setFixedWidth(50)
            slider.valueChanged.connect(lambda v, idx=i, lbl=val_lbl: self._on_slider(idx, v, lbl))
            row.addWidget(name_lbl)
            row.addWidget(slider)
            row.addWidget(val_lbl)
            slider_layout.addLayout(row)
            self._sliders.append(slider)
            self._value_labels.append(val_lbl)

        # 리셋 버튼
        reset_btn = QtWidgets.QPushButton("Reset (All Joints to 0°)")
        reset_btn.clicked.connect(self._reset_sliders)
        slider_layout.addWidget(reset_btn)

        # 저장 버튼
        save_btn = QtWidgets.QPushButton("Save Data")
        save_btn.clicked.connect(self._save_data)
        slider_layout.addWidget(save_btn)

        # 연결 상태
        self._status_lbl = QtWidgets.QLabel("● Wait...")
        self._status_lbl.setStyleSheet("color: orange; font-weight: bold;")
        slider_layout.addWidget(self._status_lbl)

        left_vbox.addWidget(slider_panel)

        # --- 모션 재생 패널 ---
        motion_panel = QtWidgets.QGroupBox("Motion Playback")
        motion_layout = QtWidgets.QVBoxLayout(motion_panel)

        # ComboBox: 데이터셋 파일 목록
        combo_row = QtWidgets.QHBoxLayout()
        combo_lbl = QtWidgets.QLabel("Motion:")
        combo_lbl.setFixedWidth(55)
        self._motion_combo = QtWidgets.QComboBox()
        self._motion_combo.addItem("(none)")
        self._populate_motion_combo()
        combo_row.addWidget(combo_lbl)
        combo_row.addWidget(self._motion_combo)
        motion_layout.addLayout(combo_row)

        # Play / Stop 버튼
        btn_row = QtWidgets.QHBoxLayout()
        self._play_btn = QtWidgets.QPushButton("Play")
        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._on_play_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(self._stop_btn)
        motion_layout.addLayout(btn_row)

        # 상태 라벨
        self._motion_status_lbl = QtWidgets.QLabel("Frame 0/0 (0.00s)")
        motion_layout.addWidget(self._motion_status_lbl)

        left_vbox.addWidget(motion_panel)
        left_vbox.addStretch()

        main_layout.addWidget(left_panel)

        # 오른쪽: 그래프 패널 (pyqtgraph)
        graph_panel = QtWidgets.QGroupBox("Real-time Joint Status")
        graph_layout = QtWidgets.QVBoxLayout(graph_panel)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self._plots: dict[str, pg.PlotWidget] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}

        for metric, unit in [("pos", "rad"), ("vel", "rad/s"), ("torque", "Nm")]:
            pw = pg.PlotWidget(title=f"Joint {metric} ({unit})")
            pw.setMaximumHeight(160)
            pw.addLegend(offset=(5, 5))
            pw.showGrid(x=True, y=True, alpha=0.3)
            for i, j in enumerate(JOINT_LABELS):
                key = f"{j}_{metric}"
                curve = pw.plot(
                    np.zeros(HISTORY_LEN),
                    pen=pg.mkPen(color=COLORS[i], width=1.5),
                    name=j,
                )
                self._curves[key] = curve
            self._plots[metric] = pw
            graph_layout.addWidget(pw)

        main_layout.addWidget(graph_panel)

    def _populate_motion_combo(self) -> None:
        """DATASET_DIR의 *.txt 파일을 ComboBox에 추가."""
        if not DATASET_DIR.exists():
            print(f"[R2S Controller] 경고: 데이터셋 디렉토리가 존재하지 않습니다: {DATASET_DIR}")
            return
        txt_files = sorted(DATASET_DIR.glob("*.txt"))
        for f in txt_files:
            self._motion_combo.addItem(f.stem, userData=f)

    # ------------------------------------------------------------------
    # 이벤트 핸들러
    # ------------------------------------------------------------------
    def _on_slider(self, idx: int, value_x10: int, lbl: QtWidgets.QLabel) -> None:
        deg = value_x10 / 10.0
        lbl.setText(f"{deg:.1f}°")
        self._current_setpoint[idx] = math.radians(deg)
        self._bridge.send_setpoint(self._current_setpoint.copy())

    def _reset_sliders(self) -> None:
        for slider in self._sliders:
            slider.setValue(0)

    def _save_data(self) -> None:
        path = self._logger.save()
        QtWidgets.QMessageBox.information(self, "저장 완료", f"저장: {path}")

    # ------------------------------------------------------------------
    # 모션 재생 핸들러
    # ------------------------------------------------------------------
    def _on_play_clicked(self) -> None:
        """Play 버튼 클릭: 선택된 모션 클립을 프레임 단위로 재생 시작."""
        combo_idx = self._motion_combo.currentIndex()
        if combo_idx == 0:
            # "(none)" 선택 시 무시
            return

        # 모션 클립 로드 (선택이 바뀐 경우 또는 처음 재생 시)
        path: Path = self._motion_combo.currentData()
        if self._motion_clip is None or self._motion_clip["name"] != path.stem:
            try:
                self._motion_clip = load_motion_clip(path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "로드 오류", f"모션 파일 로드 실패:\n{exc}")
                return

        self._motion_frame_idx = 0
        self._motion_playing = True

        # UI 상태 갱신
        self._play_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        for slider in self._sliders:
            slider.setEnabled(False)
        self._motion_status_lbl.setText("Settling...")

        # 타이머 간격 설정 후 시작 (frame_duration 기반, 최소 1ms)
        frame_duration = self._motion_clip["frame_duration"]
        interval_ms = max(1, int(round(frame_duration * 1000)))
        self._motion_timer.setInterval(interval_ms)
        self._motion_timer.start()

    def _on_stop_clicked(self) -> None:
        """Stop 버튼 클릭: 재생 중지 및 슬라이더 복원."""
        self._motion_playing = False
        self._motion_timer.stop()

        # UI 상태 복원
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        for slider in self._sliders:
            slider.setEnabled(True)

        total = len(self._motion_clip["joint_pos"]) if self._motion_clip else 0
        self._motion_status_lbl.setText(f"Stopped — Frame {self._motion_frame_idx}/{total}")

    def _advance_motion_frame(self) -> None:
        """모션 타이머 콜백: 현재 프레임의 joint_pos를 setpoint으로 전송."""
        if not self._motion_playing or self._motion_clip is None:
            return

        joint_pos: np.ndarray = self._motion_clip["joint_pos"]
        total_frames = len(joint_pos)
        idx = self._motion_frame_idx

        # 현재 프레임 joint_pos 가져와 JOINT_LIMITS_RAD로 clip
        frame_pos = joint_pos[idx].copy()
        for j, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            frame_pos[j] = float(np.clip(frame_pos[j], lo, hi))

        # setpoint 갱신
        self._current_setpoint[:] = frame_pos

        # 슬라이더 시각적 동기화 (blockSignals로 콜백 재진입 방지)
        for j, (slider, val_lbl) in enumerate(zip(self._sliders, self._value_labels)):
            deg = math.degrees(float(frame_pos[j]))
            slider.blockSignals(True)
            slider.setValue(int(round(deg * 10)))
            slider.blockSignals(False)
            val_lbl.setText(f"{deg:.1f}°")

        # setpoint 전송
        self._bridge.send_setpoint(self._current_setpoint.copy())

        # 상태 라벨 갱신
        elapsed_s = idx * self._motion_clip["frame_duration"]
        self._motion_status_lbl.setText(f"Frame {idx + 1}/{total_frames} ({elapsed_s:.2f}s)")

        # 다음 프레임 인덱스 결정
        loop_mode = self._motion_clip["loop_mode"]
        next_idx = idx + 1
        if next_idx >= total_frames:
            if loop_mode in ("Wrap", "Loop"):
                self._motion_frame_idx = 0
            else:
                # 단발 재생 종료
                self._motion_frame_idx = total_frames - 1
                self._on_stop_clicked()
        else:
            self._motion_frame_idx = next_idx

    # ------------------------------------------------------------------
    # ZMQ 수신 → 링 버퍼 업데이트
    # ------------------------------------------------------------------
    def _on_state(self, state: dict) -> None:
        if not self._connected:
            self._connected = True
            self._status_lbl.setText("● Connected")
            self._status_lbl.setStyleSheet("color: green; font-weight: bold;")

        pos = np.asarray(state.get("pos", np.zeros(5)), dtype=np.float32)
        vel = np.asarray(state.get("vel", np.zeros(5)), dtype=np.float32)
        torque = np.asarray(state.get("torque", np.zeros(5)), dtype=np.float32)
        ts = float(state.get("timestamp", 0.0))

        self._timestamps.append(ts)
        for i, j in enumerate(JOINT_LABELS):
            self._history[f"{j}_pos"].append(float(pos[i]))
            self._history[f"{j}_vel"].append(float(vel[i]))
            self._history[f"{j}_torque"].append(float(torque[i]))

        self._logger.log(state, setpoint=self._current_setpoint.copy())

    def _update_plots(self) -> None:
        """30Hz 타이머에서 pyqtgraph 갱신."""
        for metric in ("pos", "vel", "torque"):
            for j in JOINT_LABELS:
                key = f"{j}_{metric}"
                self._curves[key].setData(np.array(self._history[key]))

    # ------------------------------------------------------------------
    # ZMQ 수신 스레드
    # ------------------------------------------------------------------
    def _start_receiver(self) -> None:
        self._receiver = StateReceiver(self._bridge)
        self._receiver.state_received.connect(self._on_state)
        self._receiver.start()

    # ------------------------------------------------------------------
    # 종료 처리
    # ------------------------------------------------------------------
    def closeEvent(self, event: object) -> None:
        self._motion_timer.stop()
        self._render_timer.stop()
        self._receiver.stop()
        path = self._logger.save()
        if path:
            print(f"[R2S Controller] Save Data: {path}")
        self._bridge.close()
        super().closeEvent(event)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description="R2S Hind Leg 컨트롤러 GUI")
    _, qt_args = parser.parse_known_args()

    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    window = R2SController()
    window.show()
    app.aboutToQuit.connect(window.close)

    # Ctrl+C 정상 종료: Qt 이벤트 루프는 Python 시그널을 가로채므로,
    # (1) SIGINT 핸들러를 직접 등록해 app.quit()을 호출하고,
    # (2) 100ms 주기 noop 타이머로 Python 인터프리터에 제어권을 양보해 시그널이 처리되게 한다.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_timer = QtCore.QTimer()
    sigint_timer.start(100)
    sigint_timer.timeout.connect(lambda: None)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
