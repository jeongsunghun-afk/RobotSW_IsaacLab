"""전문가 데이터(.txt)를 Isaac Lab 로봇에 직접 주입하여 시각적으로 디버깅하는 스크립트."""

import argparse
import os
import torch
import numpy as np

from isaaclab.app import AppLauncher

# argparse 설정
parser = argparse.ArgumentParser(description="전문가 데이터 재생 및 디버깅 도구")
parser.add_argument("--motion_file", type=str, default=None, help="재생할 .txt 모션 파일 경로")
parser.add_argument("--num_envs", type=int, default=1, help="환경(로봇) 개수")
parser.add_argument("--fix_sign", action="store_true", help="조인트 부호 보정 적용 여부")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# App 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaaclab_assets.robots.rga import R_SKELETON_CFG
from motion_loader import SkeletonMotionLoader

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply

@configclass
class PlaybackSceneCfg(InteractiveSceneCfg):
    """재생용 씬 설정."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )
    
    # 물리 설정을 표준으로 복구 및 경고 해결
    robot: ArticulationCfg = R_SKELETON_CFG.replace(
        prim_path="/World/Robot",
    )
    # 중력 끄기 및 에러 방지용 max_depenetration_velocity 재설정
    robot.spawn.rigid_props.disable_gravity = True
    robot.spawn.rigid_props.max_depenetration_velocity = 10.0

# 전문가 데이터 발끝(Toe) 위치 시각화용 마커 설정 (InteractiveScene 외부에서 정의)
toe_markers_cfg = VisualizationMarkersCfg(
    prim_path="/World/Visuals/ToeMarkers",
    markers={
        "fl": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))),
        "fr": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
        "hl": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))),
        "hr": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))),
    }
)

def main():
    print("[1/4] 시뮬레이션 컨텍스트 초기화 중...")
    sim_cfg = sim_utils.SimulationCfg(device=getattr(args_cli, "device", "cuda:0"))
    sim_context = sim_utils.SimulationContext(sim_cfg)
    
    print("[2/4] 씬 및 로봇 구성 중...")
    scene_cfg = PlaybackSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    robot = scene.articulations["robot"]
    
    # 시각화 마커 초기화
    toe_markers = VisualizationMarkers(toe_markers_cfg)

    print("[3/4] 시뮬레이션 리셋 중...")
    sim_context.reset()

    # 모션 로더 초기화
    print("[4/4] 모션 로더 초기화 중...")
    if args_cli.motion_file is None:
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        motion_dir = os.path.join(_THIS_DIR, "imitation", "txt_dataset_skeleton_stmr")
    else:
        motion_dir = args_cli.motion_file

    motion_loader = SkeletonMotionLoader(
        motion_files=motion_dir,
        device=scene.device
    )

    robot_joint_names = robot.data.joint_names
    motion_dof_indexes = motion_loader.get_dof_index(robot_joint_names)

    # 50Hz 제어 주기 (학습 환경과 동일)
    sim_dt = 1/50.0
    count = 0
    
    print(f"재생 시작: {motion_dir}")
    print(f"환경 원점: {scene.env_origins[0]}")

    while simulation_app.is_running():
        # 데이터 샘플링 (시간 기반 보간 적용)
        current_time = count * sim_dt
        times = np.ones(args_cli.num_envs) * current_time
        dof_pos, _, body_pos, body_rot, _, _ = motion_loader.sample(args_cli.num_envs, times=times)

        # Env Origin (첫 번째 환경 기준)
        env_origin = scene.env_origins[0]

        # Root Pose (body_pos index 4 is 'base')
        root_pos = body_pos[:, 4, :]
        root_quat = body_rot[:, 4, :]

        # 전문가 데이터 발끝 위치 추출 및 좌표 변환
        # body_pos[0, :4, :]는 로봇 베이스 기준 로컬 좌표이므로 월드로 변환합니다.
        toe_pos_local = body_pos[0, :4, :] # (4, 3)
        
        # R_root * P_local + P_root
        # root_quat[0]는 (4,), root_pos[0]는 (3,) 이므로 4개 발끝에 맞춰 확장합니다.
        root_quat_repeated = root_quat[0].repeat(4, 1) # (4, 4)
        toe_pos_world = root_pos[0] + quat_apply(root_quat_repeated, toe_pos_local)
        
        # 환경 원점 반영 (World Pose)
        toe_pos_final = toe_pos_world + env_origin
        
        # 발끝 마커 시각화 (인덱스 0~3 매핑)
        toe_markers.visualize(
            translations=toe_pos_final,
            marker_indices=[0, 1, 2, 3]
        )

        # 조인트 상태 주입
        target_dof_pos = dof_pos[:, motion_dof_indexes]

        # 시뮬레이션에 상태 강제 주입
        robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        robot.write_root_velocity_to_sim(torch.zeros(args_cli.num_envs, 6, device=scene.device))
        robot.write_joint_state_to_sim(target_dof_pos, torch.zeros_like(target_dof_pos))
        robot.write_joint_velocity_to_sim(torch.zeros_like(target_dof_pos))

        # 카메라 추적 (정면 뷰)
        if count % 1 == 0:
            eye = root_pos[0].cpu().numpy() + np.array([2.5, 0.0, 1.5])
            target = root_pos[0].cpu().numpy()
            sim_context.set_camera_view(eye=eye, target=target)

        # 렌더링 업데이트
        sim_context.render()
        
        count += 1
        if current_time > motion_loader.duration:
            count = 0
            print("루프 리셋")

    simulation_app.close()

if __name__ == "__main__":
    main()
