"""
data_recorder.py
================
IsaacLab 시뮬레이션 중 로봇 데이터를 500스텝 단위로 수집하고,
results 폴더에 CSV / Plot / 동영상을 저장합니다.

저장 데이터:
    - joint torque          (num_joints)
    - joint position        (num_joints)
    - joint velocity        (num_joints)
    - processed actions     (num_joints, env._processed_actions)
    - base linear velocity  (3, body frame)
    - base angular velocity (3, body frame)

저장 결과 폴더 구조:
    results/
    └── <load_run>/
        └── <command_label>/
            ├── joint_torques.png
            ├── joint_positions.png
            ├── joint_velocities.png
            ├── processed_actions.png
            ├── base_velocity.png
            ├── robot_data.csv
            └── video.mp4  (있으면 복사)

RMS 계산 시 IQR 기반 outlier를 제거합니다.
"""

from __future__ import annotations

import pathlib
import shutil

import matplotlib
matplotlib.use("Agg")          # 헤드리스 환경용, GUI 불필요
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────

def _rms_no_outlier(arr: np.ndarray) -> float:
    """IQR 방법으로 outlier를 제거한 뒤 RMS를 계산합니다."""
    if arr.size == 0:
        return float("nan")
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = arr[(arr >= lo) & (arr <= hi)]
    if filtered.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(filtered ** 2)))


def _plot_grid(
    data: np.ndarray,          # [steps, channels]
    labels: list[str],
    title: str,
    ylabel: str,
    save_path: pathlib.Path,
    ncols: int = 4,
) -> None:
    """채널별 subplot grid를 그리고 파일로 저장합니다."""
    n = len(labels)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.8))
    axes = np.array(axes).reshape(-1)

    steps = np.arange(data.shape[0])

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        y = data[:, i]
        rms = _rms_no_outlier(y)
        ax.plot(steps, y, linewidth=0.9, color="steelblue")
        ax.axhline(rms, color="tomato", linewidth=1.2, linestyle="--", label=f"+RMS={rms:.4f}")
        ax.axhline(-rms, color="tomato", linewidth=1.2, linestyle="--", label=f"-RMS={-rms:.4f}")
        ax.set_title(lbl, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlabel("step", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    # 빈 subplot 숨기기
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# DataRecorder 클래스
# ──────────────────────────────────────────────────────────────────────

class DataRecorder:
    """시뮬레이션 루프에서 스텝마다 데이터를 수집합니다.

    Parameters
    ----------
    results_root : pathlib.Path
        'results' 최상위 폴더 절대 경로.
    load_run_name : str
        실험 이름 (save 폴더 1단계).
    max_steps : int
        수집 완료 기준 스텝 수 (기본 500).
    """

    def __init__(
        self,
        results_root: pathlib.Path,
        task_name: str,
        load_run_name: str,
        max_steps: int = 500,
    ) -> None:
        self._results_root = pathlib.Path(results_root)
        self._task_name = task_name
        self._load_run_name = load_run_name
        self._max_steps = max_steps

        # 데이터 버퍼
        self._torques: list[np.ndarray] = []
        self._joint_pos: list[np.ndarray] = []
        self._joint_vel: list[np.ndarray] = []
        self._proc_actions: list[np.ndarray] = []
        self._lin_vel: list[np.ndarray] = []
        self._ang_vel: list[np.ndarray] = []

        self._step = 0
        self._saved = False      # 이번 세그먼트 저장 완료 여부

    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def is_full(self) -> bool:
        return self._step >= self._max_steps

    @property
    def is_saved(self) -> bool:
        return self._saved

    def reset(self) -> None:
        """새 커맨드가 입력됐을 때 버퍼를 초기화합니다."""
        self._torques.clear()
        self._joint_pos.clear()
        self._joint_vel.clear()
        self._proc_actions.clear()
        self._lin_vel.clear()
        self._ang_vel.clear()
        self._step = 0
        self._saved = False

    # ------------------------------------------------------------------

    def record(self, env) -> None:
        """env에서 현재 스텝 데이터를 수집합니다 (500스텝 이후 무시)."""
        if self._step >= self._max_steps:
            return

        base_env = env.unwrapped
        robot = getattr(base_env, "_robot", None)
        if robot is None:
            return

        try:
            self._torques.append(
                robot.data.applied_torque[0].detach().cpu().numpy().copy()
            )
            self._joint_pos.append(
                robot.data.joint_pos[0].detach().cpu().numpy().copy()
            )
            self._joint_vel.append(
                robot.data.joint_vel[0].detach().cpu().numpy().copy()
            )
            # processed actions (없으면 zeros 대체)
            proc = getattr(base_env, "_processed_actions", None)
            if proc is not None:
                self._proc_actions.append(
                    proc[0].detach().cpu().numpy().copy()
                )
            else:
                self._proc_actions.append(
                    np.zeros(robot.data.joint_pos.shape[1])
                )
            self._lin_vel.append(
                robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy()
            )
            self._ang_vel.append(
                robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy()
            )
        except Exception as e:
            print(f"[DataRecorder] 데이터 수집 오류 (스텝 {self._step}): {e}")
            return

        self._step += 1

    # ------------------------------------------------------------------

    def save(
        self,
        command_label: str,
        joint_names: list[str] | None = None,
        video_src: pathlib.Path | None = None,
    ) -> pathlib.Path | None:
        """500스텝 데이터를 results 폴더에 저장합니다.

        Returns
        -------
        save_dir : pathlib.Path | None
            저장된 폴더 경로. 스텝 부족이면 None 반환.
        """
        if self._step < self._max_steps:
            print(f"[DataRecorder] {self._step}/500 스텝 — 저장 조건 미달, 건너뜀.")
            return None

        # ── 저장 디렉토리 생성 ───────────────────────────────────────
        # command_label에 파일명으로 불가능한 문자 제거
        safe_label = (
            command_label
            .replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
        )
        # 폴더 구조: results / task_name / model_name / command_label
        save_dir = self._results_root / self._task_name / self._load_run_name / safe_label
        save_dir.mkdir(parents=True, exist_ok=True)

        # numpy 배열로 변환
        torques       = np.array(self._torques)        # [500, J]
        joint_pos     = np.array(self._joint_pos)
        joint_vel     = np.array(self._joint_vel)
        proc_actions  = np.array(self._proc_actions)
        lin_vel       = np.array(self._lin_vel)        # [500, 3]
        ang_vel       = np.array(self._ang_vel)

        n_joints = torques.shape[1]
        if joint_names is None or len(joint_names) != n_joints:
            joint_names = [f"joint_{i}" for i in range(n_joints)]

        print(f"[DataRecorder] 저장 시작 → {save_dir}")

        # ── CSV 저장 ─────────────────────────────────────────────────
        self._save_csv(
            save_dir, joint_names,
            torques, joint_pos, joint_vel, proc_actions,
            lin_vel, ang_vel,
        )

        # ── Plot 저장 ────────────────────────────────────────────────
        _plot_grid(torques,      joint_names, "Joint Torques",
                   "Torque [Nm]",  save_dir / "joint_torques.png")
        _plot_grid(joint_pos,    joint_names, "Joint Positions",
                   "Pos [rad]",    save_dir / "joint_positions.png")
        _plot_grid(joint_vel,    joint_names, "Joint Velocities",
                   "Vel [rad/s]",  save_dir / "joint_velocities.png")
        _plot_grid(proc_actions, joint_names, "Processed Actions",
                   "Pos cmd [rad]", save_dir / "processed_actions.png")

        self._plot_base_velocity(lin_vel, ang_vel, save_dir / "base_velocity.png")

        # ── 동영상 복사 ──────────────────────────────────────────────
        if video_src is not None and pathlib.Path(video_src).exists():
            dst = save_dir / "video.mp4"
            shutil.copy2(str(video_src), str(dst))
            print(f"[DataRecorder] 동영상 복사 완료: {dst}")

        print(f"[DataRecorder] 저장 완료: {save_dir}")
        self._saved = True
        return save_dir

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _save_csv(
        self,
        save_dir: pathlib.Path,
        joint_names: list[str],
        torques, joint_pos, joint_vel, proc_actions,
        lin_vel, ang_vel,
    ) -> None:
        """모든 데이터를 단일 CSV로 저장합니다."""
        import csv

        n_steps, n_joints = torques.shape
        headers = ["step"]
        # joint별 4개 항목
        for jn in joint_names:
            jn_s = jn.replace(",", "_")
            headers += [
                f"torque_{jn_s}",
                f"pos_{jn_s}",
                f"vel_{jn_s}",
                f"action_{jn_s}",
            ]
        # base velocity
        headers += ["lin_vel_x", "lin_vel_y", "lin_vel_z",
                    "ang_vel_x", "ang_vel_y", "ang_vel_z"]

        csv_path = save_dir / "robot_data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for t in range(n_steps):
                row = [t]
                for j in range(n_joints):
                    row += [
                        torques[t, j],
                        joint_pos[t, j],
                        joint_vel[t, j],
                        proc_actions[t, j],
                    ]
                row += list(lin_vel[t]) + list(ang_vel[t])
                writer.writerow(row)

        # RMS 요약 CSV
        rms_path = save_dir / "rms_summary.csv"
        with open(rms_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["joint", "torque_rms", "pos_rms", "vel_rms", "action_rms"])
            for j, jn in enumerate(joint_names):
                writer.writerow([
                    jn,
                    _rms_no_outlier(torques[:, j]),
                    _rms_no_outlier(joint_pos[:, j]),
                    _rms_no_outlier(joint_vel[:, j]),
                    _rms_no_outlier(proc_actions[:, j]),
                ])
            # base velocity RMS
            for ax_name, idx in [("lin_vel_x", 0), ("lin_vel_y", 1), ("lin_vel_z", 2),
                                  ("ang_vel_x", 0), ("ang_vel_y", 1), ("ang_vel_z", 2)]:
                arr = lin_vel[:, idx] if ax_name.startswith("lin") else ang_vel[:, idx]
                writer.writerow([ax_name, _rms_no_outlier(arr), "", "", ""])

        print(f"[DataRecorder] CSV 저장: {csv_path}, {rms_path}")

    def _plot_base_velocity(
        self,
        lin_vel: np.ndarray,   # [500, 3]
        ang_vel: np.ndarray,   # [500, 3]
        save_path: pathlib.Path,
    ) -> None:
        """Base 선속도 + 각속도를 2행 3열로 플롯합니다."""
        lin_labels = ["lin_vel_x", "lin_vel_y", "lin_vel_z"]
        ang_labels = ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
        all_data   = list(zip(lin_labels, [lin_vel[:, i] for i in range(3)])) + \
                     list(zip(ang_labels, [ang_vel[:, i] for i in range(3)]))

        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        axes = axes.reshape(-1)
        steps = np.arange(lin_vel.shape[0])

        for ax, (lbl, y) in zip(axes, all_data):
            rms = _rms_no_outlier(y)
            ax.plot(steps, y, linewidth=0.9, color="mediumseagreen")
            ax.axhline(rms, color="tomato", linewidth=1.2, linestyle="--", label=f"+RMS={rms:.4f}")
            ax.axhline(-rms, color="tomato", linewidth=1.2, linestyle="--", label=f"-RMS={-rms:.4f}")
            ax.set_title(lbl, fontsize=9)
            ax.set_ylabel("m/s or rad/s", fontsize=7)
            ax.set_xlabel("step", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)

        fig.suptitle("Base Velocity (Body Frame)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=130)
        plt.close(fig)
        print(f"[DataRecorder] Base velocity plot 저장: {save_path}")
