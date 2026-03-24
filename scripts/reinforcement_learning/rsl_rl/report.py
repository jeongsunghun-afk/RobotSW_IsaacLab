# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
report.py
=========
여러 커맨드 설정을 YAML 파일에서 읽어와 그 개수만큼 병렬로 로봇을 띄운 후,
500 스텝 동안 각각의 동작을 수행하고 시뮬레이션 데이터를 분할/추출하는 리포트 스크립트.

주요 특징:
- `--commands_file` 인자로 입력 YAML 파일을 지정 (기본: commands.yaml).
- 입력 커맨드 개수만큼 `num_envs`를 강제 설정.
- 동영상 녹화 및 데이터 저장을 기본적으로(Default) 강제 적용.
- 500스텝 후 결과를 results 폴더에 개별 저장하고 스크립트 자동 종료.
"""

import argparse
import sys
import yaml
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="여러 커맨드를 일괄 평가하고 레포팅하는 스크립트.")
parser.add_argument(
    "--commands_file", type=str, default="scripts/reinforcement_learning/rsl_rl/commands.yaml",
    help="평가할 커맨드들을 정의한 YAML 파일의 위치."
)
# video, save_data 는 다른 arg parser에서 이미 추가되므로 내부에서 True로 강제합니다.
parser.add_argument("--video_length", type=int, default=500, help="녹화 길이 (스텝).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Fabric 비활성화 (USD I/O 사용).",
)
parser.add_argument("--num_envs", type=int, default=None, help="환경 수. (yaml 개수로 덮어씌워짐)")
parser.add_argument("--task", type=str, default=None, help="태스크 이름.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="RL 에이전트 설정 엔트리포인트 이름.",
)
parser.add_argument("--seed", type=int, default=None, help="환경 시드.")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true",
    help="Nucleus에서 사전학습 체크포인트 사용.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="실시간 평가 모드.")
parser.add_argument("--wbc", action="store_true", default=False, help="Enable whole body control (19 DoF instead of 12 DoF).")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Report.py 의 기본 속성 강제
args_cli.video = True
args_cli.save_data = True
args_cli.enable_cameras = True
args_cli.headless = True

# ── YAML 파싱 및 환경 수 결정 ───────────────────────────────────────────────
with open(args_cli.commands_file, "r", encoding="utf-8") as f:
    full_commands_dict = yaml.safe_load(f)

if not full_commands_dict:
    raise ValueError(f"입력 파일 {args_cli.commands_file} 이 비어 있거나 파싱할 수 없습니다.")

task_group = args_cli.task.split("-")[0]  # e.g., "Go2WTW-v1" -> "Go2WTW"
# Fallback if group is not explicitly in YAML
if task_group in full_commands_dict and isinstance(full_commands_dict[task_group], dict):
    commands_dict = full_commands_dict[task_group]
    print(f"[INFO] Task '{task_group}'에 해당하는 커맨드 그룹을 로드합니다.")
else:
    # If the file hasn't been migrated or task_group isn't found, try to use the root level
    print(f"[WARN] Task '{task_group}' 그룹을 찾을 수 없습니다. (fallback to root level or using first group)")
    if "env_0" in full_commands_dict:
        commands_dict = full_commands_dict
    else:
        # Default fallback to the first key if no env_0 exists
        first_key = list(full_commands_dict.keys())[0]
        commands_dict = full_commands_dict[first_key]
        print(f"[WARN] Fallback: Using group '{first_key}'")

command_keys = list(commands_dict.keys())
parsed_num_envs = len(command_keys)
print(f"[INFO] {args_cli.commands_file} (Group: {task_group}) 에서 {parsed_num_envs} 개의 커맨드를 읽었습니다.")
# args_cli 덮어쓰기
args_cli.num_envs = parsed_num_envs

sys.argv = [sys.argv[0]] + hydra_args

# Isaac Sim 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import time
import datetime

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerParkour
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    export_policy_as_jit_parkour,
    export_policy_as_onnx_parkour,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ── play_utils 임포트 (파일 절대 경로) ────────────────────────────────────────
import importlib.util as _ilu
import inspect as _inspect
import pathlib as _pl

_THIS_FILE = _pl.Path(
    __file__ if (__file__ is not None and _pl.Path(__file__).exists())
    else _inspect.getfile(_inspect.currentframe())
).resolve()
_PLAY_UTILS_DIR = _THIS_FILE.parent / "play_utils"

def _load_module(name: str, filename: str):
    abs_path = _PLAY_UTILS_DIR / filename
    spec = _ilu.spec_from_file_location(name, abs_path)
    if spec is None:
        raise ImportError(f"[report] 모듈 로드 실패: {abs_path}")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_env_mod               = _load_module("play_utils.env_utils",              "env_utils.py")
_gait_mod              = _load_module("play_utils.gait_commands",          "gait_commands.py")
# 새로 작성한 report_data_recorder 로드
_report_recorder_mod   = _load_module("play_utils.report_data_recorder",   "report_data_recorder.py")

print_action_joint_mapping   = _env_mod.print_action_joint_mapping
get_env_command_info         = _env_mod.get_env_command_info
get_env_interaction_info     = _env_mod.get_env_interaction_info
build_go2wtw_command         = _gait_mod.build_go2wtw_command
build_simple_command         = _gait_mod.build_simple_command
ReportMultiDataRecorder      = _report_recorder_mod.ReportMultiDataRecorder


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # ── 설정 ─────────────────────────────────────────────────────────
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(args_cli, "wbc") and hasattr(env_cfg, "whole_body_control"):
        env_cfg.whole_body_control = args_cli.wbc
        if hasattr(env_cfg, "__post_init__"):
            env_cfg.__post_init__()

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] 실험 디렉토리 로드: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # ── 환경 생성 ────────────────────────────────────────────────────
    env_cfg.debug_vis = True
    env_cfg.events = None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.set_debug_vis(getattr(env_cfg, "debug_vis", True))

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_state = {"record_video_now": True, "last_video_step": 0}

    def custom_step_trigger(step):
        if video_state.get("record_video_now", False):
            video_state["record_video_now"] = False
            video_state["last_video_step"] = step
            return True
        return False

    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "report"),
        "step_trigger": custom_step_trigger,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    # env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print_action_joint_mapping(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 모델 로드 ────────────────────────────────────────────────────
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAMP":
        runner = OnPolicyRunnerAMP(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkour":
        runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"지원하지 않는 runner 클래스: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── 환경 내부 커맨드 파악 ──────────────────────────────────────────
    labels, ranges = get_env_command_info(env)
    num_commands = len(labels)
    has_interaction, num_motions, motion_labels = get_env_interaction_info(env)

    # ── 텐서 커맨드 빌드 (YAML 결과 기반) ───────────────────────────────
    # 각 env가 수행해야할 command values 와 label 을 저장합니다.
    command_tensor_list = []
    interaction_tensor_list = []
    env_labels = []

    for idx, key in enumerate(command_keys):
        cfg = commands_dict[key]
        c_type = cfg.get("type", "simple")

        if c_type == "go2wtw":
            gait = cfg.get("gait", "trotting")
            vx = float(cfg.get("lin_vel_x", 0.0))
            vy = float(cfg.get("lin_vel_y", 0.0))
            wz = float(cfg.get("yaw_vel", 0.0))
            cmd_vals = build_go2wtw_command(gait, vx, vy, wz)
            command_tensor_list.append(cmd_vals)
            interaction_tensor_list.append(0)
            env_labels.append(f"{gait}_vx{vx}_vy{vy}_wz{wz}")

        elif c_type == "interaction":
            cmd_id = int(cfg.get("command_id", 0))
            # Go2Interaction 등인 경우
            command_tensor_list.append([0.0]*max(num_commands, 1))
            interaction_tensor_list.append(cmd_id)
            label = motion_labels[cmd_id] if cmd_id < len(motion_labels) else f"motion_{cmd_id}"
            env_labels.append(label)

        else:
            # simple
            vx = float(cfg.get("lin_vel_x", 0.0))
            vy = float(cfg.get("lin_vel_y", 0.0))
            wz = float(cfg.get("yaw_vel", 0.0))
            cmd_vals = build_simple_command(vx, vy, wz)
            command_tensor_list.append(cmd_vals)
            interaction_tensor_list.append(0)
            env_labels.append(f"vel_vx{vx}_vy{vy}_wz{wz}")

    # shape [num_envs, command_dim]
    if num_commands > 0 and not has_interaction:
        c_array = torch.tensor(command_tensor_list, dtype=torch.float32, device=env.unwrapped.device)
        # 만약 차원이 넘치면 자르고 모자라면 패딩처리하기 위한 로직
        n_dim = min(c_array.shape[1], env.unwrapped._commands.shape[1])
        c_array_aligned = c_array[:, :n_dim]
    else:
        c_array_aligned = None

    if has_interaction:
        i_array = torch.tensor(interaction_tensor_list, dtype=torch.long, device=env.unwrapped.device).unsqueeze(1)
    else:
        i_array = None

    # ── Data Recorder 생성 ──────────────────────────────────────────
    load_run_name = os.path.basename(os.path.normpath(log_dir))
    data_recorder = ReportMultiDataRecorder(
        results_root=os.path.join(os.getcwd(), "results"),
        task_name=args_cli.task,
        load_run_name=load_run_name,
        num_envs=args_cli.num_envs,
        max_steps=500,
    )



    # ── Collision Force 실시간 마커 초기화 ───────────────────────────
    # contact_sensor의 net_forces_w를 이용해 충돌 중인 링크를 실시간 시각화.
    # - External collision (발/지면 접촉): 빨간색 sphere
    # - Self-collision (발 아닌 링크 + 높이 충분): 노란색 sphere
    # force_matrix_w는 filter_prim_paths_expr 미설정으로 사용 불가.
    # 링크 쌍은 공간 근접성(두 self-collision 링크 간 거리) 휴리스틱으로 추론.
    # NOTE: 마커 객체만 여기서 생성. body_names/find_bodies는 센서 초기화가
    # 완료되는 env.reset() 이후 루프 첫 스텝에서 lazy-init으로 수행.
    _CONTACT_THRESHOLD = 1.0    # N: 이 값 이상의 force를 collision로 판단
    _GROUND_HEIGHT_THRESH = 0.12  # m: body z 위치가 이보다 낮으면 external 판단
    _PAIR_DIST_THRESH = 0.25    # m: 이 거리 이내의 두 링크는 서로 충돌 쌍으로 추론
    _COLLISION_LOG_INTERVAL = 50  # 스텝마다 터미널 로그 출력
    _ext_contact_marker = None
    _self_contact_marker = None
    _sensor_body_names: list[str] = []
    _foot_sensor_ids: list[int] = []
    _collision_sensor_ready = False   # reset() 후 lazy-init 완료 플래그
    _contact_sensor_ref = getattr(env.unwrapped, "contact_sensor", None)
    if _contact_sensor_ref is not None:
        try:
            # External collision 마커 — 빨간색 sphere
            _ext_cfg = copy.deepcopy(SPHERE_MARKER_CFG)
            _ext_cfg.markers["sphere"].radius = 0.07  # type: ignore[attr-defined]
            _ext_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.1, 0.1)  # type: ignore[attr-defined]
            _ext_cfg.prim_path = "/Visuals/ContactForce/external"
            _ext_contact_marker = VisualizationMarkers(_ext_cfg)
            print("[INFO] External collision 마커 생성 완료.")
        except Exception as _e:
            print(f"[WARN] External 마커 생성 실패 (skip): {_e}")
        try:
            # Self-collision 마커 — 노란색 sphere
            _self_cfg = copy.deepcopy(SPHERE_MARKER_CFG)
            _self_cfg.markers["sphere"].radius = 0.07  # type: ignore[attr-defined]
            _self_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 1.0, 0.0)  # type: ignore[attr-defined]
            _self_cfg.prim_path = "/Visuals/ContactForce/self"
            _self_contact_marker = VisualizationMarkers(_self_cfg)
            print("[INFO] Self-collision 마커 생성 완료.")
        except Exception as _e:
            print(f"[WARN] Self 마커 생성 실패 (skip): {_e}")

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────
    dt = env.unwrapped.step_dt
    
    # 커맨드가 환경 내부 리샘플링으로 인해 덮어씌워지지 않도록 비활성화
    def dummy_resample(env_ids):
        pass
    env.unwrapped._resample_commands = dummy_resample

    if c_array_aligned is not None:
        env.unwrapped._commands[:, :n_dim] = c_array_aligned
    if i_array is not None:
        env.unwrapped._interaction_command[:] = i_array

    # 강제 주입된 커맨드가 obs(관측치) 텐서에 즉시 반영되도록 명시적 reset 호출
    obs, _ = env.reset()
    timestep = 0

    print("[INFO] Report 루프 시작 (500 스텝).")
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_frames = []

    print(c_array_aligned)
    
    while simulation_app.is_running():
        start_time = time.time()
        if timestep % 50 == 0:
            print(f"Report Step: {timestep}/500", flush=True)

        with torch.inference_mode():
            # ── 강제 커맨드 지속 적용 ────────────────────────────────────────────
            # (reset 시 내부 변수가 꼬일 수 있으므로毎스텝 방어적으로 덮어씀)
            if c_array_aligned is not None:
                env.unwrapped._commands[:, :n_dim] = c_array_aligned
            if i_array is not None:
                env.unwrapped._interaction_command[:] = i_array

            # ── 추론 ────────────────────────────────────────────────────────
            actions = policy(obs)
            # actions = torch.zeros_like(actions, device=actions.device)

            # ── Collision 센서 lazy-init (reset() 이후 첫 스텝에서 수행) ────────
            if not _collision_sensor_ready and _contact_sensor_ref is not None:
                try:
                    if hasattr(_contact_sensor_ref, "body_names"):
                        _sensor_body_names = list(_contact_sensor_ref.body_names)
                    _foot_ids_t, _ = _contact_sensor_ref.find_bodies(".*foot")
                    _foot_sensor_ids = _foot_ids_t.tolist() if hasattr(_foot_ids_t, "tolist") else list(_foot_ids_t)
                    print(f"[INFO] Collision 센서 초기화 완료. 추적 bodies: {len(_sensor_body_names)}, 발 IDs: {_foot_sensor_ids}")
                except Exception as _lazy_e:
                    print(f"[WARN] Collision 센서 lazy-init 실패: {_lazy_e}")
                _collision_sensor_ready = True  # 실패해도 재시도 안 함

            # ── Collision Force 마커 업데이트 ────────────────────────────────
            # force_matrix_w 미사용(filter_prim_paths_expr 없음).
            # net_forces_w + 높이 휴리스틱으로 external/self 구분.
            # self-collision 쌍은 공간 근접성으로 추론 후 터미널 로그 출력.
            if (_ext_contact_marker is not None or _self_contact_marker is not None) and _contact_sensor_ref is not None:
                try:
                    forces_w = _contact_sensor_ref.data.net_forces_w  # (num_envs, num_sensor_bodies, 3)
                    _robot_vis = getattr(env.unwrapped, "_robot", None)
                    if forces_w is not None and forces_w.numel() > 0 and _robot_vis is not None:
                        force_mag = forces_w.norm(dim=-1)          # (num_envs, num_sensor_bodies)
                        contact_mask = force_mag > _CONTACT_THRESHOLD

                        # robot body_pos_w와 sensor 인덱스 범위 맞추기
                        body_pos_w = _robot_vis.data.body_pos_w    # (num_envs, num_robot_bodies, 3)
                        n_s = forces_w.shape[1]
                        n_r = body_pos_w.shape[1]
                        n_b = min(n_s, n_r)
                        contact_mask = contact_mask[:, :n_b]
                        body_pos = body_pos_w[:, :n_b, :]          # (num_envs, n_b, 3)

                        # 발 body 마스크 (1D, shape: n_b)
                        foot_mask = torch.zeros(n_b, dtype=torch.bool, device=forces_w.device)
                        for _fid in _foot_sensor_ids:
                            if _fid < n_b:
                                foot_mask[_fid] = True

                        body_z = body_pos[..., 2]                  # (num_envs, n_b)
                        low_mask = body_z < _GROUND_HEIGHT_THRESH  # 지면 근접 body

                        # External: 발 body이거나 지면 근접 body 중 contact 있는 것
                        ext_mask = contact_mask & (foot_mask.unsqueeze(0) | low_mask)
                        # Self-collision: 발 아닌 body, 지면 높이 이상, contact 있는 것
                        self_mask = contact_mask & ~foot_mask.unsqueeze(0) & ~low_mask

                        # 마커 업데이트
                        if _ext_contact_marker is not None:
                            ext_pos = body_pos[ext_mask]
                            if ext_pos.shape[0] > 0:
                                _ext_contact_marker.visualize(ext_pos)
                            else:
                                _ext_contact_marker.set_visibility(False)

                        if _self_contact_marker is not None:
                            self_pos = body_pos[self_mask]
                            if self_pos.shape[0] > 0:
                                _self_contact_marker.visualize(self_pos)
                            else:
                                _self_contact_marker.set_visibility(False)

                        # 터미널 로그: 충돌 링크 이름 + self-collision 쌍 추론 출력
                        if timestep % _COLLISION_LOG_INTERVAL == 0 and (ext_mask.any() or self_mask.any()):
                            print(f"[Collision @ step {timestep}]")
                            for _ei in range(min(forces_w.shape[0], 4)):
                                _ext_ids = ext_mask[_ei].nonzero(as_tuple=True)[0].tolist()
                                _self_ids = self_mask[_ei].nonzero(as_tuple=True)[0].tolist()
                                if _ext_ids:
                                    _ext_names = [_sensor_body_names[i] if i < len(_sensor_body_names) else f"body_{i}" for i in _ext_ids]
                                    print(f"  [Env{_ei}] External (빨강): {_ext_names}")
                                if _self_ids:
                                    _self_names = [_sensor_body_names[i] if i < len(_sensor_body_names) else f"body_{i}" for i in _self_ids]
                                    print(f"  [Env{_ei}] Self-Collision 링크 (노랑): {_self_names}")
                                    # 공간 근접성으로 충돌 쌍 추론
                                    if len(_self_ids) >= 2:
                                        _self_pos_ei = body_pos[_ei, _self_ids, :]  # (k, 3)
                                        _pairs_found = []
                                        for _pi in range(len(_self_ids)):
                                            for _pj in range(_pi + 1, len(_self_ids)):
                                                _d = (_self_pos_ei[_pi] - _self_pos_ei[_pj]).norm().item()
                                                if _d < _PAIR_DIST_THRESH:
                                                    _na = _self_names[_pi]
                                                    _nb = _self_names[_pj]
                                                    _pairs_found.append(f"{_na} <-> {_nb} (dist={_d:.3f}m)")
                                        if _pairs_found:
                                            print(f"  [Env{_ei}] 추정 충돌 쌍: {_pairs_found}")
                except Exception:
                    pass

            # print(actions)

            # ── 영상 프레임 캡처 ─────────────────────────────────────────────
            if getattr(args_cli, "video", False) and len(video_frames) < args_cli.video_length:
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, list):
                        video_frames.append(frame[0])
                    else:
                        video_frames.append(frame)

            # ── 데이터 수집 및 확인 ──────────────────────────────────────────
            if not data_recorder.is_full:
                data_recorder.record(env)
                if data_recorder.is_full and not data_recorder.is_saved:
                    joint_names = env.unwrapped._robot.data.joint_names
                    
                    video_src = os.path.join(log_dir, "videos", "report", f"rl-video-{timestamp_str}.mp4")
                    
                    if getattr(args_cli, "video", False) and len(video_frames) > 0:
                        print(f"[INFO] 동영상 저장 중... ({len(video_frames)} 프레임)", flush=True)
                        os.makedirs(os.path.dirname(video_src), exist_ok=True)
                        import imageio
                        imageio.mimsave(video_src, video_frames, fps=int(1.0/dt))
                    else:
                        video_src = None
                        
                    data_recorder.save(
                        command_labels=env_labels,
                        timestamp_str=timestamp_str,
                        joint_names=joint_names,
                        video_src=video_src,
                    )
                    
                    # 500 스텝, 저장 완료 시 깔끔하게 종료
                    print("[INFO] 500스텝 데이터 저장이 완료되어 종료합니다.", flush=True)
                    break

            obs, _, dones, _ = env.step(actions)
            
            p_reset = dones.any()
            if p_reset:
                try:
                    policy_nn = runner.alg.policy
                except AttributeError:
                    policy_nn = runner.alg.actor_critic
                policy_nn.reset(dones)
                
                # 강제 재적용
                if c_array_aligned is not None:
                    env.unwrapped._commands[:, :n_dim] = c_array_aligned
                if i_array is not None:
                    env.unwrapped._interaction_command[:] = i_array

        timestep += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
