"""
report_data_recorder.py
================
IsaacLab 시뮬레이션 중 로봇 데이터를 500스텝 단위로 *모든 환경(env)에 대해 동시에* 수집하고,
results 폴더 내 개별 환경별 디렉토리에 CSV / 조인트 분할 Plot / 동영상을 저장합니다.

저장 데이터 (환경별):
    - joint torque          (num_joints)
    - joint position        (num_joints)
    - joint velocity        (num_joints)
    - processed actions     (num_joints)
    - base linear velocity  (3, body frame)
    - base angular velocity (3, body frame)

Plots 분할:
    1. 왼쪽 다리 (Left Leg)
    2. 오른쪽 다리 (Right Leg)
    3. 허리/목 등 (Waist/Neck/Other)
"""

from __future__ import annotations

import pathlib
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    if n == 0:
        return
    
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


class ReportMultiDataRecorder:
    """시뮬레이션 루프에서 스텝마다 모든 환경(로봇)의 데이터를 동시에 수집합니다.

    Parameters
    ----------
    results_root : pathlib.Path
        'results' 최상위 폴더 절대 경로.
    task_name : str
        태스크명.
    load_run_name : str
        실험 이름.
    num_envs : int
        기록할 환경의 수.
    max_steps : int
        수집 완료 기준 스텝 수 (기본 500).
    """

    def __init__(
        self,
        results_root: pathlib.Path,
        task_name: str,
        load_run_name: str,
        num_envs: int,
        max_steps: int = 500,
    ) -> None:
        self._results_root = pathlib.Path(results_root)
        self._task_name = task_name
        self._load_run_name = load_run_name
        self._num_envs = num_envs
        self._max_steps = max_steps

        # 데이터 버퍼 - env별 리스트
        # 구조: [env_id][step_idx] = np.ndarray
        self._torques: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._joint_pos: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._joint_vel: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._proc_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._lin_vel: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._ang_vel: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

        self._step = 0
        self._saved = False

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
        """버퍼 초기화"""
        for i in range(self._num_envs):
            self._torques[i].clear()
            self._joint_pos[i].clear()
            self._joint_vel[i].clear()
            self._proc_actions[i].clear()
            self._lin_vel[i].clear()
            self._ang_vel[i].clear()
        self._step = 0
        self._saved = False

    def record(self, env) -> None:
        """env에서 모든 로봇의 현재 스텝 데이터를 수집합니다."""
        if self._step >= self._max_steps:
            return

        base_env = env.unwrapped
        robot = getattr(base_env, "_robot", None)
        if robot is None:
            return

        try:
            # 모두 [num_envs, num_channels] 형태로 가져옴
            curr_torques = robot.data.applied_torque.detach().cpu().numpy()
            curr_joint_pos = robot.data.joint_pos.detach().cpu().numpy()
            curr_joint_vel = robot.data.joint_vel.detach().cpu().numpy()
            curr_lin_vel = robot.data.root_lin_vel_b.detach().cpu().numpy()
            curr_ang_vel = robot.data.root_ang_vel_b.detach().cpu().numpy()

            proc = getattr(base_env, "_processed_actions", None)
            if proc is not None:
                curr_proc_actions = proc.detach().cpu().numpy()
            else:
                curr_proc_actions = np.zeros_like(curr_joint_pos)

            for i in range(self._num_envs):
                self._torques[i].append(curr_torques[i].copy())
                self._joint_pos[i].append(curr_joint_pos[i].copy())
                self._joint_vel[i].append(curr_joint_vel[i].copy())
                self._proc_actions[i].append(curr_proc_actions[i].copy())
                self._lin_vel[i].append(curr_lin_vel[i].copy())
                self._ang_vel[i].append(curr_ang_vel[i].copy())

        except Exception as e:
            print(f"[ReportMultiDataRecorder] 데이터 수집 오류 (스텝 {self._step}): {e}", flush=True)
            return

        self._step += 1
        # print(f"Recorded step {self._step}", flush=True)

    def save(
        self,
        command_labels: list[str],
        timestamp_str: str,
        joint_names: list[str] | None = None,
        video_src: pathlib.Path | None = None,
    ) -> pathlib.Path | None:
        """수집된 데이터를 지정된 세부 구조에 저장합니다."""
        if self._step < self._max_steps:
            print(f"[ReportMultiDataRecorder] {self._step}/500 스텝 — 저장 조건 미달.")
            return None

        # Base Directory: results / task_name / load_run_name / report_{timestamp}
        base_save_dir = self._results_root / self._task_name / self._load_run_name / f"report_{timestamp_str}"
        base_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ReportMultiDataRecorder] 저장 시작 → {base_save_dir}")

        # 복사용 비디오 처리
        if video_src is not None and pathlib.Path(video_src).exists():
            dst = base_save_dir / "video.mp4"
            shutil.copy2(str(video_src), str(dst))
            print(f"[ReportMultiDataRecorder] 동영상 복사 완료: {dst}")

        n_joints = len(self._torques[0][0])
        if joint_names is None or len(joint_names) != n_joints:
            joint_names = [f"joint_{i}" for i in range(n_joints)]

        # --- 조인트 필터링 인덱스 분류 ---
        # 1: 왼쪽 다리 (Left)
        # 2: 오른쪽 다리 (Right)
        # 3: 허리/목 등 (기타)
        idx_left = []
        idx_right = []
        idx_other = []
        
        for i, name in enumerate(joint_names):
            n_lower = name.lower()
            if "l_" in n_lower or "left" in n_lower or "fl_" in n_lower or "rl_" in n_lower:
                idx_left.append(i)
            elif "r_" in n_lower or "right" in n_lower or "fr_" in n_lower or "rr_" in n_lower:
                idx_right.append(i)
            else:
                idx_other.append(i)
                
        # 각 환경별 데이터 저장 루프
        for env_id in range(self._num_envs):
            cmd_label = command_labels[env_id] if env_id < len(command_labels) else f"cmd_{env_id}"
            safe_label = cmd_label.replace(" ", "_").replace("/", "-").replace("\\", "-")
            env_dir = base_save_dir / f"env{env_id}_{safe_label}"
            env_dir.mkdir(parents=True, exist_ok=True)

            torques = np.array(self._torques[env_id])
            joint_pos = np.array(self._joint_pos[env_id]) * (180.0 / np.pi)
            joint_vel = np.array(self._joint_vel[env_id]) * (180.0 / np.pi)
            proc_actions = np.array(self._proc_actions[env_id]) * (180.0 / np.pi)
            lin_vel = np.array(self._lin_vel[env_id])
            ang_vel = np.array(self._ang_vel[env_id]) * (180.0 / np.pi)

            self._save_csv(
                env_dir, joint_names, torques, joint_pos, joint_vel, proc_actions, lin_vel, ang_vel
            )

            # 조인트 파트별 그래프 생성 헬퍼
            def save_split_plots(data, title_prefix, ylabel, file_prefix):
                # Left
                if len(idx_left) > 0:
                    _plot_grid(data[:, idx_left], [joint_names[i] for i in idx_left],
                               f"{title_prefix} (Left Leg)", ylabel, env_dir / f"{file_prefix}_left_leg.png")
                # Right
                if len(idx_right) > 0:
                    _plot_grid(data[:, idx_right], [joint_names[i] for i in idx_right],
                               f"{title_prefix} (Right Leg)", ylabel, env_dir / f"{file_prefix}_right_leg.png")
                # Other (Waist/Neck)
                if len(idx_other) > 0:
                    _plot_grid(data[:, idx_other], [joint_names[i] for i in idx_other],
                               f"{title_prefix} (Waist/Neck)", ylabel, env_dir / f"{file_prefix}_waist_neck.png")

            # Torques, Positions, Velocities, Actions 분할 저장
            save_split_plots(torques, "Joint Torques", "Torque [Nm]", "joint_torques")
            save_split_plots(joint_pos, "Joint Positions", "Pos [deg]", "joint_positions")
            save_split_plots(joint_vel, "Joint Velocities", "Vel [deg/s]", "joint_velocities")
            save_split_plots(proc_actions, "Processed Actions", "Pos cmd [deg]", "processed_actions")

            # Base Velocity Plot
            self._plot_base_velocity(lin_vel, ang_vel, env_dir / "base_velocity.png")

        print(f"[ReportMultiDataRecorder] 모든 환경 데이터 저장 완료: {base_save_dir}")
        self._saved = True
        return base_save_dir

    def _save_csv(
        self,
        save_dir: pathlib.Path,
        joint_names: list[str],
        torques, joint_pos, joint_vel, proc_actions,
        lin_vel, ang_vel,
    ) -> None:
        import csv

        n_steps, n_joints = torques.shape
        headers = ["step"]
        for jn in joint_names:
            jn_s = jn.replace(",", "_")
            headers += [
                f"torque_{jn_s}",
                f"pos_{jn_s}",
                f"vel_{jn_s}",
                f"action_{jn_s}",
            ]
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
            for ax_name, idx in [("lin_vel_x", 0), ("lin_vel_y", 1), ("lin_vel_z", 2),
                                  ("ang_vel_x", 0), ("ang_vel_y", 1), ("ang_vel_z", 2)]:
                arr = lin_vel[:, idx] if ax_name.startswith("lin") else ang_vel[:, idx]
                writer.writerow([ax_name, _rms_no_outlier(arr), "", "", ""])

    def _plot_base_velocity(
        self,
        lin_vel: np.ndarray,
        ang_vel: np.ndarray,
        save_path: pathlib.Path,
    ) -> None:
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
            if "ang" in lbl:
                ax.set_ylabel("deg/s", fontsize=7)
            else:
                ax.set_ylabel("m/s", fontsize=7)
            ax.set_xlabel("step", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)

        fig.suptitle("Base Velocity (Body Frame)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=130)
        plt.close(fig)
