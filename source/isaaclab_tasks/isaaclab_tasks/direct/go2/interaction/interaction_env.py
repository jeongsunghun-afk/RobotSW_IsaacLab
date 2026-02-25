import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import os
import pandas as pd
import numpy as np


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class InteractionEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg=None, depth_cfg=None, show_viewer=False, task=6, device="cuda"):
        self.device = torch.device(device)
        self.task = task

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg.get("num_privileged_obs", None)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        
        # OnPolicyRunnerParkour requirements
        self.num_prio_obs = obs_cfg["num_prio_obs"]
        self.num_heights = obs_cfg["num_heights"]
        self.num_priv = obs_cfg["num_priv"]
        self.num_priv_latent = obs_cfg["num_priv_latent"]
        self.history_len = obs_cfg["history_len"]
        self.obs_cfg = obs_cfg

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # === Imitation-specific parameters ===
        self.size_prop = 0.9
        self.y_offset = 0.07
        self.z_offset = 0.05
        self.x_offset = 0.0  # X방향 오프셋 추가 (모든 hip, foot에 적용) - ref_data를 뒤로 25.7cm 이동
        
        # Joint indices definition (Genesis specific - 이 부분은 실제 로봇 구조에 맞게 수정 필요)
        self.ref_hip_indices = [15, 20, 4, 8]  # FL_hip, FR_hip, RL_hip, RR_hip
        self.ref_foot_indices = [18, 23, 7, 11]  # FL_foot, FR_foot, RL_foot, RR_foot
        self.left_indices = [15, 18, 4, 7]  # FL_hip, FL_foot, RL_hip, RL_foot
        self.right_indices = [20, 23, 8, 11]  # FR_hip, FR_foot, RR_hip, RR_foot
        # self.left_indices = [16, 18, 5, 7]  # FL_hip, FL_foot, RL_hip, RL_foot
        # self.right_indices = [21, 23, 9, 11]  # FR_hip, FR_foot, RR_hip, RR_foot
        
        # Motion files definition
        self.motion_files = {
            0: "default.csv",  # Default state
            1: "sit.csv",      # Sitting
            2: "lay.csv",      # Laying
            3: "stand_cap_v2.csv",    # Stand up
        }
        
        # Motion data initialization
        self.target_world_positions = {}
        self.all_ref_data = {}
        self.all_preprocessed_data = {}

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        
        file_name = "urdf/go1/urdf/go1.urdf"
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=file_name,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                merge_fixed_links=False
            ),
        )
        self.cam = None
        if not show_viewer:
            # Camera following the robots. For visualization
            self.view_cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (2.5, 2.5, 0.7),
                lookat = (0.0, 0.0, 0.7),
                up     = (0.0, 0.0, 1.0),
                fov    = 40,
                GUI    = True,
            )
            self.view_cam.follow_entity(self.robot, fixed_axis=(None, None, 0.5), smoothing=0.5, fix_orientation=True)
            # # For FOV of Go1 Robot
            self.cam = self.scene.add_camera(
                res    = (640, 480),
                pos    = (0.3, 0.0, 0.42),
                lookat = (1.3, 0.0, -0.157),
                fov    = 87,
                GUI    = False,
            )
            self.cam.follow_entity(self.robot, fixed_axis=(None, None, None), smoothing=0.5, fix_orientation=True)
        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_start for name in self.env_cfg["dof_names"]]
        # Precompute indices for slicing left and right torques from the main self.torques tensor
        # The columns of self.torques correspond to the order of dof_names in env_cfg
        all_controlled_dof_names = self.env_cfg["dof_names"]
        left_dof_names_from_cfg = self.env_cfg["left_dof_names"]
        right_dof_names_from_cfg = self.env_cfg["right_dof_names"]

        dof_name_to_ordered_idx = {name: i for i, name in enumerate(all_controlled_dof_names)}

        self.left_torque_indices = torch.tensor(
            [dof_name_to_ordered_idx[name] for name in left_dof_names_from_cfg if name in dof_name_to_ordered_idx],
            dtype=torch.long, device=self.device
        )
        self.right_torque_indices = torch.tensor(
            [dof_name_to_ordered_idx[name] for name in right_dof_names_from_cfg if name in dof_name_to_ordered_idx],
            dtype=torch.long, device=self.device
        )

        # PD control parameters with randomization
        self.base_kp = self.env_cfg["kp"]
        self.base_kd = self.env_cfg["kd"]
        
        # Initialize PD gains with randomization (0.9~1.1 range)
        self.kp_randomization_range = (0.8, 1.2)
        self.kd_randomization_range = (0.8, 1.2)
        
        # Store randomized gains for each environment
        self.randomized_kp = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.randomized_kd = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        
        # Initialize with randomized gains
        self._randomize_pd_gains(torch.arange(self.num_envs))

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        
        # Privileged observations buffer
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        else:
            self.privileged_obs_buf = None
        self.extra_commands = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float) 
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.robot.get_vel())
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        # === Motion-related buffers ===
        self.current_frames = torch.zeros(self.num_envs, 1, dtype=torch.int, device=self.device, requires_grad=False)
        self.interaction_command = torch.zeros(self.num_envs, 1, dtype=torch.int, device=self.device)
        
        # Initialize observation history buffer
        self.obs_history = None

        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices
        
        # def find_joint_indices(names):
        #     joint_indices = list()
        #     for joint in self.robot.joints:
        #         flag = False
        #         for name in names:
        #             if name in joint.name:
        #                 flag = True
        #         if flag:
        #             joint_indices.append(joint.idx - self.robot.joint_start)
        #     return joint_indices

        # 전체 링크 정보를 보기 위해 하나씩 출력
        print("=== Robot Links ===")
        for i, link in enumerate(self.robot.links):
            print(f"Link {i}: {link}, name: {link.name}")
        print("=================")
        
        # 전체 Joint 정보를 보기 위해 하나씩 출력
        # print("=== Robot Joints ===")
        # for i, joint in enumerate(self.robot.joints):
        #     print(f"Joint {i}: {joint}, name: {joint.name}, dof_idx_local: {joint.dof_idx_local}")
        # print("=================")
        
        self.termination_contact_link_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        )
        self.penalized_contact_link_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        )
        print("penalized_contact_link_indices : ", self.penalized_contact_link_indices)
        self.feet_link_indices = find_link_indices(
            self.env_cfg['feet_link_names']
        )
        print("feet_link_indices : ", self.feet_link_indices)
        self.hip_link_indices = find_link_indices(
            self.env_cfg['hip_link_names']
        )
        print("hip_link_indices : ", self.hip_link_indices)
        
        # Joint indices도 찾아서 출력
        # self.hip_joint_indices = find_joint_indices(
        #     self.env_cfg['hip_link_names']  # hip joint 이름이 hip link 이름과 비슷할 것으로 예상
        # )
        # print("hip_joint_indices : ", self.hip_joint_indices)
        
        # self.foot_joint_indices = find_joint_indices(
        #     self.env_cfg['feet_link_names']  # foot joint 이름이 foot link 이름과 비슷할 것으로 예상
        # )
        # print("foot_joint_indices : ", self.foot_joint_indices)
        
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.penalized_contact_link_indices) > 0
        assert len(self.feet_link_indices) > 0

        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        self.all_link_pos = torch.ones(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float,
        )

        self.foot_positions = torch.ones(
            (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float,
        )
        self.foot_quaternions = torch.ones(
            (self.num_envs, len(self.feet_link_indices), 4), device=self.device, dtype=gs.tc_float,
        )
        self.foot_velocities = torch.ones(
            (self.num_envs, len(self.feet_link_indices), 3), device=self.device, dtype=gs.tc_float,
        )

        self.hip_positions = torch.ones(
            (self.num_envs, len(self.hip_link_indices), 3), device=self.device, dtype=gs.tc_float,
        )
        self.hip_quaternions = torch.ones(
            (self.num_envs, len(self.hip_link_indices), 4), device=self.device, dtype=gs.tc_float,
        )
        self.hip_velocities = torch.ones(
            (self.num_envs, len(self.hip_link_indices), 3), device=self.device, dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        
        # === Motion data initialization ===
        # Preload all motion data
        self.preload_all_motions()
        
        # Set default motion (0: default)
        self.current_motion_id = 0

    # === Motion Data Loading Functions ===
    def _preprocess_motion_data(self, motion_data):
        """Process a single motion data (DataFrame)"""
        preprocessed_data = {}
        
        # Check max frame count
        max_frames = int(motion_data['frame'].max())
        
        # Process all frames
        for frame in range(max_frames + 1):
            frame_data = motion_data[motion_data['frame'] == frame]
            
            # Dictionary to store preprocessed position data for this frame
            preprocessed_data[frame] = {}
            
            # joint_idx=0인 행에서 base rotation 정보 추출
            base_row = frame_data[frame_data['joint_idx'] == 0]
            if not base_row.empty:
                # roll, pitch, yaw 값 저장
                preprocessed_data[frame]['roll'] = float(base_row['roll'].values[0])
                preprocessed_data[frame]['pitch'] = float(base_row['pitch'].values[0])
                preprocessed_data[frame]['yaw'] = float(base_row['yaw'].values[0])
                preprocessed_data[frame]['base_height'] = float(base_row['base_frame_z'].values[0])

            # Preprocess Hip positions
            for hip_idx in self.ref_hip_indices:
                hip_mask = (frame_data['joint_idx'] == hip_idx)
                hip_data = frame_data[hip_mask]
                
                if not hip_data.empty:
                    # Base-relative coordinates (with scaling and x_offset)
                    x_rel = float(hip_data['base_rel_x'].values[0]) * self.size_prop + self.x_offset
                    y_rel = float(hip_data['base_rel_y'].values[0]) * self.size_prop
                    
                    # Apply y-offset based on left/right distinction
                    if hip_idx in self.left_indices:
                        y_rel -= self.y_offset  # Left legs offset in negative direction
                    elif hip_idx in self.right_indices:
                        y_rel += self.y_offset  # Right legs offset in positive direction
                    
                    z_rel = float(hip_data['base_rel_z'].values[0]) * self.size_prop + self.z_offset
                    
                    # Store preprocessed relative coordinates
                    preprocessed_data[frame][hip_idx] = {
                        'x_rel': x_rel, 
                        'y_rel': y_rel, 
                        'z_rel': z_rel
                    }
            
            # Preprocess Foot positions
            for foot_idx in self.ref_foot_indices:
                foot_mask = (frame_data['joint_idx'] == foot_idx)
                foot_data = frame_data[foot_mask]
                
                if not foot_data.empty:
                    # Base-relative coordinates (with scaling and x_offset)
                    x_rel = float(foot_data['base_rel_x'].values[0]) * self.size_prop + self.x_offset
                    y_rel = float(foot_data['base_rel_y'].values[0]) * self.size_prop
                    
                    # Apply y-offset based on left/right distinction
                    if foot_idx in self.left_indices:
                        y_rel -= self.y_offset
                    elif foot_idx in self.right_indices:
                        y_rel += self.y_offset
                    
                    z_rel = float(foot_data['base_rel_z'].values[0]) * self.size_prop + self.z_offset
                    
                    # Store preprocessed relative coordinates
                    preprocessed_data[frame][foot_idx] = {
                        'x_rel': x_rel, 
                        'y_rel': y_rel, 
                        'z_rel': z_rel
                    }
        
        return preprocessed_data

    def preload_all_motions(self):
        """모든 모션 데이터를 미리 로드하고 전처리"""
        # 현재 파일의 디렉토리를 기준으로 절대 경로 생성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ref_motion_path = os.path.join(current_dir, 'ref_motion')
        
        for motion_id, filename in self.motion_files.items():
            motion_data = pd.read_csv(os.path.join(ref_motion_path, filename))
            self.all_preprocessed_data[motion_id] = self._preprocess_motion_data(motion_data)
        print("*"*50)
        print(f"Total {len(self.all_preprocessed_data)} motion data loaded")
        print("*"*50)
        # 각 모션의 프레임 길이를 리스트로 직접 생성
        motion_lengths = []
        for i in range(len(self.motion_files)):  # 0부터 순차적으로
            motion_lengths.append(len(self.all_preprocessed_data[i]))
        
        # 리스트를 텐서로 변환
        self.motion_lengths_tensor = torch.tensor(motion_lengths, device=self.device)
        print("motion_lengths : ", self.motion_lengths_tensor)

    def get_target_reference_data(self):
        """각 env별 현재 프레임과 모션에 따른 참조 데이터 반환 (벡터화 버전)"""
        motion_ids = self.interaction_command.long()
        frames = self.current_frames.long()
        
        # print(self.num_envs)
        # print(motion_ids.shape)
        # print("motion_ids : ", motion_ids)
        # print("frames : ", frames)
        # 모든 env의 현재 모션/프레임 데이터를 한번에 가져오기
        motion_data = []
        for mid, f in zip(motion_ids.squeeze(), frames.squeeze()):
            if isinstance(mid, torch.Tensor):
                mid = mid.item()
            if isinstance(f, torch.Tensor):
                # 텐서가 스칼라가 아닌 경우(4096개 요소가 있는 경우) 첫 번째 요소만 가져옴
                f = f[0].item() if f.numel() > 1 else f.item()
            motion_data.append(self.all_preprocessed_data[mid][f])
        
        # 한번에 모든 env의 데이터 준비
        hip_pos_local = torch.stack([
            torch.tensor([[motion[hip_idx]['x_rel'], 
                          motion[hip_idx]['y_rel'], 
                          motion[hip_idx]['z_rel']] 
                         for hip_idx in self.ref_hip_indices], 
                        device=self.device)
            for motion in motion_data
        ])  # [num_envs, num_hips, 3]
        
        foot_pos_local = torch.stack([
            torch.tensor([[motion[foot_idx]['x_rel'], 
                          motion[foot_idx]['y_rel'], 
                          motion[foot_idx]['z_rel']] 
                         for foot_idx in self.ref_foot_indices], 
                        device=self.device)
            for motion in motion_data
        ])  # [num_envs, num_feet, 3]
        
        # root position을 한번에 더하기
        root_pos = self.base_pos.unsqueeze(1)  # [num_envs, 1, 3]
        hip_pos_world = root_pos + hip_pos_local  # broadcasting
        foot_pos_world = root_pos + foot_pos_local  # broadcasting
        # base rotation & height 한번에 처리
        base_rot_world = torch.stack([
            torch.tensor([motion['roll'], -1 * motion['pitch'], motion['yaw']], 
                        device=self.device)
            for motion in motion_data
        ])  # [num_envs, 3]
        
        base_height = torch.tensor([
            motion['base_height'] #+ 0.05  # 로봇과 모션 데이터의 크기 차이를 보정하기 위해 0.1 offset 추가
            for motion in motion_data

        ], device=self.device)  # [num_envs, 1]
        return {
            'hip_positions': hip_pos_world,
            'foot_positions': foot_pos_world,
            'base_rotation': base_rot_world,
            'base_height': base_height
        }

    def calculate_world_positions(self):
        """
        calculate world positions
        """
        target_ref_data = self.get_target_reference_data()
        
        if 'hip_positions' in target_ref_data and 'foot_positions' in target_ref_data:
            self.target_world_positions = {
                'hip': target_ref_data['hip_positions'],
                'foot': target_ref_data['foot_positions']
            }
        else:
            self.target_world_positions = {}
        
        # base_rotation 정보 추가
        if 'base_rotation' in target_ref_data:
            self.target_world_positions['base_rotation'] = target_ref_data['base_rotation']
            
        # base_height 정보 추가
        if 'base_height' in target_ref_data:
            self.target_world_positions['base_height'] = target_ref_data['base_height'].squeeze(-1)  # [num_envs, 1] -> [num_envs]
            
        return self.target_world_positions

    def _draw_debug_vis(self):
        """디버깅을 위한 시각화 - Genesis용"""
        # Scene clear
        self.scene.clear_debug_objects()
        
        # 참조 데이터가 없으면 종료
        if not hasattr(self, 'base_pos') or not hasattr(self, 'base_quat'):
            return
        
        # 모든 환경에 대해 시각화 (첫 번째 환경만 하려면 range(1)로 변경)
        for i in range(1):
            # === 1. 로봇 base 위치 시각화 (녹색) ===
            base_pos_single = self.base_pos[i:i+1]  # [1, 3]
            # print(base_pos_single)
            self.scene.draw_debug_spheres(
                poss=base_pos_single, 
                radius=0.1, 
                color=(0, 1, 0, 0.8)  # 녹색
            )
            
            # # === 2. Hip 위치 시각화 (빨간색) ===
            if hasattr(self, 'hip_positions') and self.hip_positions is not None:
                # 로봇의 실제 hip 위치 - 이미 world coordinates
                current_hip_positions = self.hip_positions[i]
                self.scene.draw_debug_spheres(
                    poss=current_hip_positions[0], 
                    radius=0.05, 
                    color=(1, 0, 0, 0.5)  # 빨간색
                )
                self.scene.draw_debug_spheres(
                    poss=current_hip_positions[1], 
                    radius=0.05, 
                    color=(1, 0., 1., 0.5)  # 빨간색
                )
                self.scene.draw_debug_spheres(
                    poss=current_hip_positions[2], 
                    radius=0.05, 
                    color=(1, 1., 0., 0.5)  # 빨간색
                )
                self.scene.draw_debug_spheres(
                    poss=current_hip_positions[3], 
                    radius=0.05, 
                    color=(1, 1., 1., 0.5)  # 빨간색
                )
            
            # === 3. Foot 위치 시각화 (파란색) ===
            if hasattr(self, 'foot_positions') and self.foot_positions is not None:
                # 로봇의 실제 foot 위치 - 이미 world coordinates
                current_foot_positions = self.foot_positions[i]
                self.scene.draw_debug_spheres(
                    poss=current_foot_positions[0], 
                    radius=0.05, 
                    color=(0, 0, 1, 0.5)  # 파란색
                )
                self.scene.draw_debug_spheres(
                    poss=current_foot_positions[1], 
                    radius=0.05, 
                    color=(0., 1., 1, 0.5)  # 파란색
                )
                self.scene.draw_debug_spheres(
                    poss=current_foot_positions[2], 
                    radius=0.05, 
                    color=(0., 1., 0., 0.5)  # 파란색
                )
                self.scene.draw_debug_spheres(
                    poss=current_foot_positions[3], 
                    radius=0.05, 
                    color=(0., 0., 0., 0.5)  # 파란색
                )
            
            # === 4. Target 위치 시각화 (만약 있다면) ===
            if hasattr(self, 'target_world_positions') and self.target_world_positions:
                # Target hip 위치 (빨간색, 더 작은 크기)
                if 'hip' in self.target_world_positions:
                    target_hip_positions = self.target_world_positions['hip'][i]  # [num_hips, 3]
                    self.scene.draw_debug_spheres(
                        poss=target_hip_positions[0], 
                        radius=0.03, 
                        color=(1, 0., 0., 0.8)  # 연한 빨간색
                    )
                    self.scene.draw_debug_spheres(
                        poss=target_hip_positions[1], 
                        radius=0.03, 
                        color=(1, 0., 1.0, 0.8)  # 연한 빨간색
                    )
                    self.scene.draw_debug_spheres(
                        poss=target_hip_positions[2], 
                        radius=0.03, 
                        color=(1, 1., 0., 0.8)  # 연한 빨간색
                    )
                    self.scene.draw_debug_spheres(
                        poss=target_hip_positions[3], 
                        radius=0.03, 
                        color=(1, 1., 1., 0.8)  # 연한 빨간색
                    )
                
                # Target foot 위치 (파란색, 더 작은 크기)
                if 'foot' in self.target_world_positions:
                    target_foot_positions = self.target_world_positions['foot'][i]  # [num_feet, 3]
                    self.scene.draw_debug_spheres(
                        poss=target_foot_positions[0], 
                        radius=0.03, 
                        color=(0., 0., 1, 0.8)  # 연한 파란색
                    )
                    self.scene.draw_debug_spheres(
                        poss=target_foot_positions[1], 
                        radius=0.03, 
                        color=(0., 1., 1., 0.8)  # 연한 파란색
                    )
                    self.scene.draw_debug_spheres(
                        poss=target_foot_positions[2], 
                        radius=0.03, 
                        color=(0., 1., 0, 0.8)  # 연한 파란색
                    )
                    self.scene.draw_debug_spheres(
                        poss=target_foot_positions[3], 
                        radius=0.03, 
                        color=(0., 0., 0., 0.8)  # 연한 파란색
                    )
            # if hasattr(self, 'hip_positions') and self.hip_positions is not None:
            #     # 로봇의 실제 hip 위치 - 이미 world coordinates
            #     current_hip_positions = self.hip_positions[i]
            #     self.scene.draw_debug_spheres(
            #         poss=current_hip_positions, 
            #         radius=0.05, 
            #         color=(1, 0, 0, 0.8)  # 빨간색
            #     )
            
            # # === 3. Foot 위치 시각화 (파란색) ===
            # if hasattr(self, 'foot_positions') and self.foot_positions is not None:
            #     # 로봇의 실제 foot 위치 - 이미 world coordinates
            #     current_foot_positions = self.foot_positions[i]
            #     self.scene.draw_debug_spheres(
            #         poss=current_foot_positions, 
            #         radius=0.05, 
            #         color=(0, 0, 1, 0.8)  # 파란색
            #     )
            
            # # === 4. Target 위치 시각화 (만약 있다면) ===
            # if hasattr(self, 'target_world_positions') and self.target_world_positions:
            #     # Target hip 위치 (빨간색, 더 작은 크기)
            #     if 'hip' in self.target_world_positions:
            #         target_hip_positions = self.target_world_positions['hip'][i]  # [num_hips, 3]
            #         self.scene.draw_debug_spheres(
            #             poss=target_hip_positions, 
            #             radius=0.03, 
            #             color=(1, 0.5, 0.5, 0.6)  # 연한 빨간색
            #         )
                
            #     # Target foot 위치 (파란색, 더 작은 크기)
            #     if 'foot' in self.target_world_positions:
            #         target_foot_positions = self.target_world_positions['foot'][i]  # [num_feet, 3]
            #         self.scene.draw_debug_spheres(
            #             poss=target_foot_positions, 
            #             radius=0.03, 
            #             color=(0.5, 0.5, 1, 0.6)  # 연한 파란색
            #         )
            
            # self.scene.draw_debug_spheres(poss=self.robot.get_links_pos()[i, 7],radius=0.05, color=(0, 1, 0, 0.7))

    def _resample_commands(self, envs_idx):
        """각 env별 새로운 모션 command 샘플링"""
        # 기존 속도 command 샘플링
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device) + self.extra_commands[envs_idx]
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        
        # 모션 파일 개수만큼 랜덤 정수 생성 (모션 데이터가 있는 경우에만)
        if hasattr(self, 'all_preprocessed_data') and len(self.all_preprocessed_data) > 0:
            num_motions = len(self.all_preprocessed_data)

            self.interaction_command[envs_idx] = torch.randint(
                0,  # min value
                num_motions,  # max value (exclusive)
                size=(len(envs_idx), 1),  # shape: [num_reset_envs, 1]
                device=self.device,
                dtype=torch.int
            )

            self.current_frames[envs_idx] = 0

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # === Motion frame update ===
        # if hasattr(self, 'motion_lengths_tensor') and len(self.motion_lengths_tensor) > 0:
            # self.current_frames = (self.current_frames + 1) % self.motion_lengths_tensor[self.interaction_command.squeeze(-1)]
        self.current_frames = (self.current_frames + 1) % self.motion_lengths_tensor[self.interaction_command]
        # Fetch all motor torques once
        self.torques = self.robot.get_dofs_control_force(self.motor_dofs)
        
        # Apply motor strength noise (scale factor)
        if hasattr(self.env_cfg, 'domain_rand') and 'motor_strength_range' in self.env_cfg['domain_rand']:
            motor_noise_scale = gs_rand_float(
                self.env_cfg['domain_rand']['motor_strength_range'][0],
                self.env_cfg['domain_rand']['motor_strength_range'][1],
                (self.num_envs, self.num_actions),  # 각 env, 각 joint별로
                self.device
            )
            self.torques = self.torques * motor_noise_scale
        
        # Slice to get left and right torques
        self.left_torques = self.torques[:, self.left_torque_indices]
        self.right_torques = self.torques[:, self.right_torque_indices]

        # all link pos & quat
        self.all_link_pos = self.robot.get_links_pos()
        self.all_link_quat = self.robot.get_links_quat()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            degrees=False  # 라디안 단위로 변경
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        
        self.foot_positions = self.robot.get_links_pos(self.feet_link_indices)
        self.hip_positions = self.robot.get_links_pos(self.hip_link_indices)
        
        # self.foot_positions[:] = self.scene.rigid_solver.get_links_pos(self.feet_link_indices)
        # self.foot_quaternions[:] = self.scene.rigid_solver.get_links_quat(self.feet_link_indices)
        # self.foot_velocities[:] = self.scene.rigid_solver.get_links_vel(self.feet_link_indices)
        # self.hip_positions[:] = self.scene.rigid_solver.get_links_pos(self.hip_link_indices)
        # hip pos debugging
        # hip_names = ["FL_hip", "FR_hip", "HL_hip", "HR_hip"]
        # print("hip_positions : ", self.hip_positions[1])
        # for i, name in enumerate(hip_names):
        #     print(f"{name}: {self.hip_positions[1,i]}")
            
        self.hip_quaternions[:] = self.scene.rigid_solver.get_links_quat(self.hip_link_indices)
        self.hip_velocities[:] = self.scene.rigid_solver.get_links_vel(self.hip_link_indices)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.extras["command_scale"] = torch.mean(self.extra_commands)

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        # === Calculate world positions ===
        self.calculate_world_positions()
        
        # === Debug: Check x-axis offset ===
        if hasattr(self, 'target_world_positions') and self.target_world_positions and self.episode_length_buf[0] % 100 == 1:
            if 'hip' in self.target_world_positions:
                # 현재 로봇 base x 위치
                current_base_x = self.base_pos[0, 0].item()
                # 타겟 hip들의 평균 x 위치 (첫 번째 env)
                target_hip_x_mean = self.target_world_positions['hip'][0, :, 0].mean().item()
                # 현재 로봇 hip들의 평균 x 위치 (첫 번째 env)
                current_hip_x_mean = self.hip_positions[0, :, 0].mean().item()
                
                print(f"DEBUG X-offset - Base: {current_base_x:.3f}, Target Hip Mean: {target_hip_x_mean:.3f}, Current Hip Mean: {current_hip_x_mean:.3f}")
                print(f"  Hip offset from base: Target={target_hip_x_mean - current_base_x:.3f}, Current={current_hip_x_mean - current_base_x:.3f}")
                print(f"  Difference: {(target_hip_x_mean - current_base_x) - (current_hip_x_mean - current_base_x):.3f}")

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations for parkour runner compatibility
        # Base proprioceptive observations (prio_obs)
        try:
            prio_obs = torch.cat([
                self.projected_gravity,  # 3
                self.interaction_command,  # 1
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                self.base_euler[:, :2],  # roll, pitch (2) - required for parkour
            ], dim=-1)  # Total: 3+1+12+12+12+2 = 42 dimensions
        except Exception as e:
            print(f"ERROR in prio_obs creation: {e}")
            print(f"projected_gravity shape: {self.projected_gravity.shape}")
            print(f"interaction_command shape: {self.interaction_command.shape}")
            print(f"dof_pos shape: {self.dof_pos.shape}")
            print(f"dof_vel shape: {self.dof_vel.shape}")
            print(f"actions shape: {self.actions.shape}")
            print(f"base_euler shape: {self.base_euler.shape}")
            raise
        
        # Height measurements (empty for interaction tasks)
        heights = torch.zeros((self.num_envs, self.num_heights), device=self.device, dtype=gs.tc_float)
        
        # Privileged observations - based on go1_task_md_interaction.py
        if self.num_priv > 0:
            # Extract robot base angles (roll, pitch, yaw)
            priv_obs = torch.cat([
                self.base_euler,  # roll, pitch, yaw (3)
            ], dim=-1)
        else:
            priv_obs = torch.zeros((self.num_envs, self.num_priv), device=self.device, dtype=gs.tc_float)
        
        # Latent privileged observations - extended from go1_task_md_interaction.py
        if self.num_priv_latent > 0:
            # Include motion tracking information and robot state
            priv_latent = torch.cat([
                self.base_pos[:, 2:3],  # base height (1)
                self.base_lin_vel,      # base linear velocity (3)
                self.base_ang_vel,      # base angular velocity (3)
                self.dof_pos[:, :6],    # first 6 joint positions (6)
                self.dof_vel[:, :6],    # first 6 joint velocities (6)
                self.torques[:, :3],    # first 3 joint torques (3)
            ], dim=-1)  # Total: 1+3+3+6+6+3 = 22, pad to num_priv_latent if needed
            
            # Pad with zeros if needed to match expected size
            if priv_latent.shape[1] < self.num_priv_latent:
                padding_size = self.num_priv_latent - priv_latent.shape[1]
                padding = torch.zeros((self.num_envs, padding_size), device=self.device, dtype=gs.tc_float)
                priv_latent = torch.cat([priv_latent, padding], dim=-1)
            elif priv_latent.shape[1] > self.num_priv_latent:
                priv_latent = priv_latent[:, :self.num_priv_latent]
        else:
            priv_latent = torch.zeros((self.num_envs, self.num_priv_latent), device=self.device, dtype=gs.tc_float)
        
        # History buffer (for now, just repeat current prio_obs)
        if not hasattr(self, 'obs_history') or self.obs_history is None:
            self.obs_history = torch.zeros((self.num_envs, self.history_len, self.num_prio_obs), device=self.device, dtype=gs.tc_float)
        
        # Update history buffer
        self.obs_history = torch.roll(self.obs_history, 1, dims=1)
        self.obs_history[:, 0] = prio_obs
        history_obs = self.obs_history.flatten(start_dim=1)  # [num_envs, history_len * num_prio_obs]
        
        # Combine all observations
        self.obs_buf = torch.cat([
            prio_obs,  # num_prio_obs
            priv_obs,  # num_priv
            priv_latent,  # num_priv_latent
            history_obs,  # num_prio_obs * history_len
        ], dim=-1)
        
        # Debug: Print observation sizes
        # if self.episode_length_buf[0] == 1:  # Only print at the start
        #     print(f"DEBUG: prio_obs shape: {prio_obs.shape}")
        #     print(f"DEBUG: heights shape: {heights.shape}")
        #     print(f"DEBUG: priv_obs shape: {priv_obs.shape}")
        #     print(f"DEBUG: priv_latent shape: {priv_latent.shape}")
        #     print(f"DEBUG: history_obs shape: {history_obs.shape}")
        #     print(f"DEBUG: final obs_buf shape: {self.obs_buf.shape}")
        #     print(f"DEBUG: expected total: {self.num_prio_obs + self.num_heights + self.num_priv + self.num_priv_latent + self.num_prio_obs * self.history_len}")
        
        # Update privileged observations buffer if needed
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat([
                priv_obs,
                priv_latent,
            ], dim=-1)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # === Debug visualization (optional) ===
        # self._draw_debug_vis()

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        if self.privileged_obs_buf is not None:
            print(f"DEBUG get_privileged_observations: privileged_obs_buf shape: {self.privileged_obs_buf.shape}")
        else:
            print("DEBUG get_privileged_observations: privileged_obs_buf is None")
        return self.privileged_obs_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0.
        self.base_ang_vel[envs_idx] = 0.
        self.robot.zero_all_dofs_velocity(envs_idx)

        # === Reset motion-related buffers ===
        if hasattr(self, 'current_frames'):
            self.current_frames[envs_idx] = 0

        # reset buffers
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        
        # Reset observation history for reset environments
        if hasattr(self, 'obs_history') and self.obs_history is not None:
            self.obs_history[envs_idx] = 0.0

        # Randomize PD gains for reset environments
        self._randomize_pd_gains(envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        # Generate initial observations
        # Calculate base euler angles for reset
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            degrees=False
        )
        
        # Base proprioceptive observations (prio_obs)
        prio_obs = torch.cat([
            self.projected_gravity,  # 3
            self.interaction_command,  # 1
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
            self.base_euler[:, :2],  # roll, pitch (2) - required for parkour
        ], dim=-1)
        
        # Height measurements (empty for interaction tasks)
        heights = torch.zeros((self.num_envs, self.num_heights), device=self.device, dtype=gs.tc_float)
        
        # Privileged observations - based on go1_task_md_interaction.py
        if self.num_priv > 0:
            # Extract robot base angles (roll, pitch, yaw)
            priv_obs = torch.cat([
                self.base_euler,  # roll, pitch, yaw (3)
            ], dim=-1)
        else:
            priv_obs = torch.zeros((self.num_envs, self.num_priv), device=self.device, dtype=gs.tc_float)
        
        # Latent privileged observations - extended from go1_task_md_interaction.py
        if self.num_priv_latent > 0:
            # Include motion tracking information and robot state
            # For reset, use zero values for velocities and actions
            priv_latent = torch.cat([
                self.base_pos[:, 2:3],  # base height (1)
                torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float),  # base linear velocity (3)
                torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float),  # base angular velocity (3)
                self.dof_pos[:, :6],    # first 6 joint positions (6)
                torch.zeros((self.num_envs, 6), device=self.device, dtype=gs.tc_float),  # first 6 joint velocities (6)
                torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float),  # first 3 joint torques (3)
            ], dim=-1)  # Total: 1+3+3+6+6+3 = 22, pad to num_priv_latent if needed
            
            # Pad with zeros if needed to match expected size
            if priv_latent.shape[1] < self.num_priv_latent:
                padding_size = self.num_priv_latent - priv_latent.shape[1]
                padding = torch.zeros((self.num_envs, padding_size), device=self.device, dtype=gs.tc_float)
                priv_latent = torch.cat([priv_latent, padding], dim=-1)
            elif priv_latent.shape[1] > self.num_priv_latent:
                priv_latent = priv_latent[:, :self.num_priv_latent]
        else:
            priv_latent = torch.zeros((self.num_envs, self.num_priv_latent), device=self.device, dtype=gs.tc_float)
        
        # Initialize history buffer with zeros
        if not hasattr(self, 'obs_history') or self.obs_history is None:
            self.obs_history = torch.zeros((self.num_envs, self.history_len, self.num_prio_obs), device=self.device, dtype=gs.tc_float)
        
        # Fill history with current observation
        self.obs_history[:, :] = prio_obs.unsqueeze(1)
        history_obs = self.obs_history.flatten(start_dim=1)
        
        # Combine all observations
        self.obs_buf = torch.cat([
            prio_obs,  # num_prio_obs
            heights,   # num_heights
            priv_obs,  # num_priv
            priv_latent,  # num_priv_latent
            history_obs,  # num_prio_obs * history_len
        ], dim=-1)
        
        print(f"DEBUG RESET: obs_buf shape: {self.obs_buf.shape}")
        print(f"DEBUG RESET: expected size: {self.num_prio_obs + self.num_heights + self.num_priv + self.num_priv_latent + self.num_prio_obs * self.history_len}")
        
        return self.obs_buf, None


    def _randomize_friction(self, envs_idx=None):
        ''' Randomize friction of all links'''
        min_friction, max_friction = self.env_cfg["domain_rand"]["friction_range"]

        solver = self.scene.rigid_solver

        ratios = gs.rand((len(envs_idx), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_friction - min_friction) + min_friction

        self.friction_coeffs_tensor[envs_idx] = ratios
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), envs_idx)
    
    def _randomize_base_mass(self, envs_idx=None):
        ''' Randomize base mass'''
        min_mass, max_mass = self.env_cfg["domain_rand"]["added_mass_range"]
        base_link_id = 1
        added_mass = gs.rand((len(envs_idx), 1), dtype=float) * (max_mass - min_mass) + min_mass
        self.mass_params_tensor[envs_idx, 0] = added_mass.squeeze(-1)
        self.scene.rigid_solver.set_links_mass_shift(added_mass, [base_link_id, ], envs_idx)
    
    def _randomize_com_displacement(self, envs_idx):

        min_displacement, max_displacement = self.env_cfg["domain_rand"]["com_displacement_range"]

        base_link_id = 1
        com_displacement = gs.rand((len(envs_idx), 1, 3), dtype=float) \
                            * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02
        self.mass_params_tensor[envs_idx, 1:4] = com_displacement.squeeze(1)
        self.scene.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], envs_idx)
    
    def _randomize_pd_gains(self, envs_idx):
        """Randomize PD gains for specified environments"""
        if len(envs_idx) == 0:
            return
            
        # Generate random multipliers for Kp gains
        kp_multipliers = gs_rand_float(
            self.kp_randomization_range[0], 
            self.kp_randomization_range[1], 
            (len(envs_idx), self.num_actions), 
            self.device
        )
        
        # Generate random multipliers for Kd gains
        kd_multipliers = gs_rand_float(
            self.kd_randomization_range[0], 
            self.kd_randomization_range[1], 
            (len(envs_idx), self.num_actions), 
            self.device
        )
        
        # Apply randomization to base gains
        self.randomized_kp[envs_idx] = self.base_kp * kp_multipliers
        self.randomized_kd[envs_idx] = self.base_kd * kd_multipliers
        
        # Set the randomized gains to the robot
        # Genesis expects gains for each environment and each DOF separately
        for env_idx in envs_idx:
            env_kp = self.randomized_kp[env_idx].cpu().numpy()
            env_kd = self.randomized_kd[env_idx].cpu().numpy()
            
            self.robot.set_dofs_kp(env_kp, self.motor_dofs, envs_idx=[env_idx])
            self.robot.set_dofs_kv(env_kd, self.motor_dofs, envs_idx=[env_idx])
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.push_interval_s > 0:
            max_push_vel_xy = self.env_cfg["domain_rand"]["max_push_vel_xy"]
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)
    
    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose only when interaction command is 0
        rew = torch.zeros(self.num_envs, device=self.device)
        mask = (self.interaction_command == 0).squeeze(-1)
        rew[mask] = torch.sum(torch.abs(self.dof_pos[mask] - self.default_dof_pos), dim=1)
        return rew

    def _reward_base_height(self):
        """로봇의 base height와 target height의 일치 정도에 따른 보상"""
        if not self.target_world_positions or 'base_height' not in self.target_world_positions:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 현재 base height 계산
        current_heights = self.base_pos[:, 2]  # z 좌표
        
        # interaction_command가 2인 환경들을 마스킹
        mask_interaction_2 = (self.interaction_command == 2).squeeze(-1)
        mask_normal = ~mask_interaction_2
        
        # 타겟 height 초기화
        target_heights = torch.zeros(self.num_envs, device=self.device)
        
        # interaction_command가 2인 환경은 0.05로 고정
        target_heights[mask_interaction_2] = 0.05
        
        # 나머지 환경은 기존 target_world_positions 사용
        if torch.any(mask_normal):
            target_heights[mask_normal] = self.target_world_positions['base_height'][mask_normal]
        
        # height 차이 계산
        height_diff = torch.abs(target_heights - current_heights)
        reward = torch.exp(-1.0 * height_diff / self.reward_cfg["sigma"])  # 계수는 조정 가능
        
        return reward

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_vel[:, :2]
        # cur_vel = self.root_states[:, 7:9]
        # print(target_vec_norm, cur_vel, self.commands[:, 0])
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew

    def _reward_tracking_yaw(self):
        # print(self.target_yaw, self.yaw)
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        return rew
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        # rew[self.env_class != 12] *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew[self.env_class != 12] = 0.
        return rew

    def _reward_torques_balance(self):
        return torch.norm(self.left_torques - self.right_torques, dim=1)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        rew = torch.sum(torch.square(self.dof_vel), dim=1)
        return rew
    
    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
     
    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        # rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
        #      4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        # 1) feet_indices 중에서 geom_a에 포함된 index를 찾기
        if self.contact_forces.numel() == 0:  # 해당하는 접촉이 없으면 return 0
            return 0.0
        
        # 3) force의 x,y 성분의 norm 계산
        force_xy_norm = torch.norm(self.contact_forces[:, :, :2], dim=-1)  # L2 norm of (fx, fy)

        # 4) z 성분과 비교하여 페널티 적용
        force_z = self.contact_forces[:, :, 2].abs()
        rew = torch.any(force_xy_norm > force_z, dim=1) # x, y 힘이 z보다 크면 stumble 발생

        return rew.float()

    def _reward_hip_positions(self):
        """로봇의 hip 위치와 target hip 위치가 가까울수록 보상 부여"""
        if not self.target_world_positions or 'hip' not in self.target_world_positions:
            return torch.zeros(self.num_envs, device=self.device)
               
        # 로봇의 실제 hip 위치 가져오기
        current_hip_positions = self.hip_positions  # [:, N, xyz]
        # 타겟 hip 위치 가져오기
        target_hip_positions = self.target_world_positions['hip']  # [:, N, xyz]
        
        # 거리 계산 (유클리드 거리)
        hip_pos_error = torch.sum(torch.norm(current_hip_positions - target_hip_positions, dim=2), dim=1)
        # 거리가 작을수록 큰 보상 (지수함수 사용)
        reward = torch.exp(-hip_pos_error / self.reward_cfg["sigma"])

        return reward

    def _reward_foot_positions(self):
        """로봇의 foot 위치와 target foot 위치가 가까울수록 보상 부여"""
        if not self.target_world_positions or 'foot' not in self.target_world_positions:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 로봇의 실제 foot 위치 가져오기
        current_foot_positions = self.foot_positions  # [:, N, xyz]
        
        # 타겟 foot 위치 가져오기
        target_foot_positions = self.target_world_positions['foot']  # [:, N, xyz]
        
        # 거리 계산 (유클리드 거리)
        foot_pos_error = torch.sum(torch.norm(current_foot_positions - target_foot_positions, dim=2), dim=1)
        
        # 거리가 작을수록 큰 보상 (지수함수 사용)
        reward = torch.exp(-foot_pos_error / self.reward_cfg["sigma"])
        # print("foot_position reward : ", reward)

        return reward

    def _reward_base_pitch(self):
        """로봇의 base pitch와 target pitch의 일치 정도에 따른 보상"""
        if not self.target_world_positions or 'base_rotation' not in self.target_world_positions:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 이미 계산된 base_p 사용 (중복 계산 방지)
        current_pitch = self.base_euler[:, 1]
        # print("current_pitch : ", current_pitch)
        # 참조 데이터의 pitch 가져오기
        target_pitch = self.target_world_positions['base_rotation'][:, 1]  # pitch는 인덱스 1
        # print("target_pitch : ", target_pitch)
        # pitch 차이 계산 (각도 차이 최소화)
        pitch_diff = torch.abs(current_pitch - target_pitch)
        # # 각도 차이가 작을수록 큰 보상 (지수함수 사용)
        reward = torch.exp(-1.0 * pitch_diff / self.reward_cfg["sigma"])  # 5.0은 민감도 계수 (조정 가능)
        
        return reward
    
    def _reward_feet_contact(self):
        """설정된 발 접촉 패턴에 따라 reward 계산
        - True로 설정된 발: 접촉 시 reward
        - False로 설정된 발: 비접촉 시 reward
        """
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.  # [num_envs, 4]
        
        # interaction_command는 [num_envs, 1] 형태
        # squeeze(-1)로 [num_envs] 형태로 변환
        commands = self.interaction_command.squeeze(-1)  # [num_envs]
        
        # 각 env별로 독립적으로 조건 평가
        # commands == 3인 env만 False로 설정되고, 나머지는 모두 True
        FL_contact = torch.where(commands == 3, 
                           torch.zeros_like(commands, dtype=torch.bool),  # commands == 3: False
                           torch.ones_like(commands, dtype=torch.bool))   # commands != 3: True
        
        FR_contact = torch.where(commands == 3,
                           torch.zeros_like(commands, dtype=torch.bool),  # commands == 3: False  
                           torch.ones_like(commands, dtype=torch.bool))   # commands != 3: True
        
        # [num_envs, 4] 형태의 마스크 생성
        contact_mask = torch.stack([
            FL_contact,  # [num_envs]
            FR_contact,  # [num_envs]
            torch.ones_like(FL_contact, dtype=torch.bool),  # HL_contact: 항상 True
            torch.ones_like(FL_contact, dtype=torch.bool)   # HR_contact: 항상 True
        ], dim=1)  # -> [num_envs, 4]
        
        # 각 env별로 원하는 접촉 상태와 실제 접촉 상태 비교
        correct_contacts = torch.where(
            contact_mask,  # [num_envs, 4]
            contact,      # [num_envs, 4] True로 설정된 발: 접촉해야 함
            ~contact      # [num_envs, 4] False로 설정된 발: 떨어져있어야 함
        )
        
        # 모든 발이 원하는 상태일 때 reward
        reward = torch.all(correct_contacts, dim=1).float()
        
        return reward

    def _reward_foot_parallel(self):
        """로봇의 앞발끼리/뒷발끼리 base frame 기준으로 x 방향으로 평행한 위치에 있도록 보상"""
        if not hasattr(self, 'foot_positions') or self.foot_positions is None:
            return torch.zeros(self.num_envs, device=self.device)
        
        # foot_positions를 base frame으로 변환
        # 1. World coordinates에서 base position을 빼서 base-relative position 구하기
        base_pos_expanded = self.base_pos.unsqueeze(1)  # [num_envs, 1, 3]
        foot_positions_relative = self.foot_positions - base_pos_expanded  # [num_envs, 4, 3]
        
        # 2. Base quaternion의 inverse를 사용해서 base frame으로 회전 변환
        inv_base_quat = inv_quat(self.base_quat)  # [num_envs, 4]
        inv_base_quat_expanded = inv_base_quat.unsqueeze(1).expand(-1, 4, -1)  # [num_envs, 4, 4]
        
        # 각 foot position을 base frame으로 변환
        foot_positions_base_frame = transform_by_quat(
            foot_positions_relative.reshape(-1, 3),  # [num_envs*4, 3]
            inv_base_quat_expanded.reshape(-1, 4)    # [num_envs*4, 4]
        ).reshape(self.num_envs, 4, 3)  # [num_envs, 4, 3]
        
        # Base frame에서 x 좌표만 추출
        foot_x_coords = foot_positions_base_frame[:, :, 0]  # [num_envs, 4]
        
        # 앞발끼리 x 좌표 차이 (FL vs FR)
        front_feet_x_diff = torch.abs(foot_x_coords[:, 0] - foot_x_coords[:, 1])  # [num_envs]
        
        # 뒷발끼리 x 좌표 차이 (HL vs HR) 
        rear_feet_x_diff = torch.abs(foot_x_coords[:, 2] - foot_x_coords[:, 3])   # [num_envs]
        
        # 차이가 작을수록 높은 보상 (지수함수 사용)
        front_parallel_reward = torch.exp(-5.0 * front_feet_x_diff)  # 계수 5.0은 조정 가능
        rear_parallel_reward = torch.exp(-5.0 * rear_feet_x_diff)
        
        # 앞발과 뒷발 보상을 합산
        total_reward = (front_parallel_reward + rear_parallel_reward) / 2.0
        
        return total_reward
