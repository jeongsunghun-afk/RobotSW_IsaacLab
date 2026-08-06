# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 Real2Sim 환경 — 순수 포지션 제어 테스트."""

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .r2s_go2_env_cfg import (
    DEFAULT_KD,
    DEFAULT_KP,
    FIXED_BASE_HEIGHT_M,
    JOINT_ORDER,
    NUM_JOINTS,
    V_MAX_RAD,
    R2SGo2EnvCfg,
)

# PACE 식별 물성. 값 자체는 이 r2s 파이프라인(`scripts/real2sim/fit_go2.py` → `validate_go2.py`)의
# 산출물이지만 현재 유일한 정의 위치가 go2_imitation_tracking cfg 다. 중복 정의하면 한쪽만 갱신되는
# drift 가 나므로 거기서 가져온다 — 언젠가 정식 위치를 이 패키지로 옮기는 편이 맞다.
from isaaclab_tasks.direct.go2_imitation_tracking.go2_imitation_tracking_env_cfg import (  # isort: skip
    PACE_ARMATURE,
    PACE_ARMATURE_SET3,
    PACE_COULOMB,
    PACE_COULOMB_SET3,
    PACE_VISCOUS,
    PACE_VISCOUS_SET3,
)

# `set_joint_plant()` 프리셋 코드. `scripts/real2sim/r2s_go2/r2s_udp.py` 의 `PLANT_*` 와 같은 값이며,
# UDP ctrl 패킷이 이 정수를 그대로 나른다. 여기서 다시 정의하는 이유는 sim 쪽(Isaac python)이
# scripts/ 를 임포트하지 않기 때문 — 둘 중 하나를 바꾸면 반대쪽도 함께 바꿀 것.
PLANT_DEFAULT: int = 0
PLANT_SET2: int = 1
PLANT_SET3: int = 2


class R2SGo2Env(DirectRLEnv):
    """R2S-GO2 Real2Sim 환경.

    RL 없음. sim_runner_go2.py가 UDP로 받은 PD 목표를 set_setpoint()로 주입하면
    slew rate limiter를 통해 안전하게 적용하고, get_lowstate()로 현재 상태를
    Unitree LowState 스키마(CONTRACT §3, §4.2)와 맞춰 반환합니다.

    Slew rate: max_step[i] = V_MAX_RAD[i] / control_freq (1/50 s)
    """

    cfg: R2SGo2EnvCfg

    def __init__(self, cfg: R2SGo2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스 매핑 (CONTRACT §2 고정 순서, USD 로드 순서 독립)
        self._joint_ids, found_names = self.robot.find_joints(JOINT_ORDER, preserve_order=True)
        assert len(self._joint_ids) == NUM_JOINTS, (
            f"관절 {NUM_JOINTS}개 필요, {len(self._joint_ids)}개 발견.\n"
            f"찾은 이름: {found_names}\n"
            f"전체 관절: {list(self.robot.data.joint_names)}"
        )
        assert found_names == JOINT_ORDER, (
            f"관절 순서가 CONTRACT §2 고정 순서와 다름.\n찾은 순서: {found_names}\n기대 순서: {JOINT_ORDER}"
        )
        print(f"[R2SGo2Env] 관절 매핑: {list(zip(JOINT_ORDER, found_names))}")

        # base body 인덱스 (IMU 가속도 조회용)
        base_ids, base_names = self.robot.find_bodies("base")
        assert len(base_ids) == 1, f"base body 1개 필요, {len(base_ids)}개 발견: {base_names}"
        self._base_id = base_ids[0]

        # 발 body 인덱스 — pedipulation 마커용. 순서는 `pedipulation_runtime.LEG_NAMES`
        # (FL, FR, RL, RR) 와 맞춰야 GUI 가 보낸 다리 인덱스가 같은 발을 가리킨다.
        self._foot_ids, foot_names = self.robot.find_bodies(
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
        )
        assert len(self._foot_ids) == 4, f"발 4개 필요, {len(self._foot_ids)}개 발견: {foot_names}"
        print(f"[R2SGo2Env] 발 매핑(마커용): {list(zip(('FL', 'FR', 'RL', 'RR'), foot_names))}")
        self._pedi_markers: VisualizationMarkers | None = None  # 첫 표시 요청 때 만든다
        self._log_marker = os.environ.get("R2S_LOG_MARKER") == "1"
        self._marker_log_n = 0
        self._no_slew = os.environ.get("R2S_NO_SLEW") == "1"
        if self._no_slew:
            print("[R2SGo2Env] ⚠ slew rate limiter 우회 (R2S_NO_SLEW=1) — 진단 전용")
        self._log_slew = os.environ.get("R2S_LOG_SLEW") == "1"
        self._slew_log_n = 0
        self._slew_clipped_total = 0
        self._slew_max_raw = 0.0

        # slew rate 한계 (rad per control step = 1/50 s)
        control_freq = 1.0 / (self.cfg.sim.dt * self.cfg.decimation)
        self._max_step = torch.tensor([v / control_freq for v in V_MAX_RAD], device=self.device, dtype=torch.float32)

        # 새 버퍼 — _reset_idx에서 초기화 필수 (CLAUDE.md 전역 DO 규칙)
        self._setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._prev_setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        # dq/kp/kd/tau는 M1에서 시뮬레이션에 미적용(향후 faithful 모드 seam) — 버퍼 저장만 (CONTRACT §5).
        self._dq_setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._kp = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._kd = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._tau_ff = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

        # ── PACE 식별 관절 물성 (articulation 관절 순서) ────────────────────────
        # cfg 로는 못 넣는다 — viscous/coulomb 은 ArticulationCfg 필드가 아니라 PhysX 에 직접
        # write 해야 하는 관절 속성이라, go2_imitation_tracking_env 와 같이 reset 때 쓴다.
        # 미식별 관절이 생기면 nominal(armature 0.01, 마찰 0)로 남는다.
        names = list(self.robot.data.joint_names)
        nj = len(names)
        # nominal = cfg 액추에이터 기본값(일반적인 시뮬 값). PACE 미식별 관절도 이 값으로 남는다.
        # 지금 읽어두는 이유: 아직 PACE write 가 한 번도 없었으므로 이 값이 곧 nominal 이다.
        # (`default_joint_armature` 는 IsaacLab 4.0 에서 없어질 예정이라 쓰지 않는다.)
        self._nominal_armature = self.robot.data.joint_armature[0].clone()
        self._nominal_zero = torch.zeros(nj, device=self.device)

        def _spread(table: dict[str, float] | None) -> torch.Tensor:
            """관절 타입별 스칼라를 articulation 관절 순서 텐서로 펼친다. None 이면 0 텐서."""
            out = torch.zeros(nj, device=self.device)
            if table is not None:
                for key in ("hip", "thigh", "calf"):
                    ids = [i for i, n in enumerate(names) if n.endswith(f"_{key}_joint")]
                    out[ids] = table[key]
            return out

        # 프리셋 3종. armature 만 nominal 을 바닥값으로 깔아 **미식별 관절**(있다면)이 0 이 되지
        # 않게 한다 — viscous/coulomb 은 미식별이면 0 이 맞다.
        def _armature(table: dict[str, float]) -> torch.Tensor:
            out = self._nominal_armature.clone()
            for key in ("hip", "thigh", "calf"):
                ids = [i for i, n in enumerate(names) if n.endswith(f"_{key}_joint")]
                out[ids] = table[key]
            return out

        self._plant_presets: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            PLANT_DEFAULT: (self._nominal_armature, self._nominal_zero, self._nominal_zero),
            PLANT_SET2: (_armature(PACE_ARMATURE), _spread(PACE_VISCOUS), _spread(PACE_COULOMB)),
            PLANT_SET3: (_armature(PACE_ARMATURE_SET3), _spread(PACE_VISCOUS_SET3), _spread(PACE_COULOMB_SET3)),
        }
        # 하위호환: 기존 코드가 참조하던 이름 (= set2)
        self._pace_armature, self._pace_viscous, self._pace_coulomb = self._plant_presets[PLANT_SET2]

        self._all_joint_ids = torch.arange(nj, dtype=torch.int32, device=self.device)
        # 현재 프리셋. cfg 가 초기값이고, 이후 set_joint_plant() 가 갱신한다(GUI 런타임 전환).
        self._plant_mode = PLANT_SET2 if bool(self.cfg.use_pace_params) else PLANT_DEFAULT

    # ------------------------------------------------------------------
    # 외부 인터페이스 (sim_runner_go2.py에서 호출)
    # ------------------------------------------------------------------

    def _to_tensor(self, x: torch.Tensor | list[float]) -> torch.Tensor:
        """12-벡터를 (num_envs, 12) torch 텐서로 정규화."""
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        return t

    def set_setpoint(
        self,
        q: torch.Tensor | list[float],
        dq: torch.Tensor | list[float],
        kp: torch.Tensor | list[float],
        kd: torch.Tensor | list[float],
        tau: torch.Tensor | list[float],
    ) -> None:
        """PD 목표 주입 (CONTRACT §5). shape: (num_envs, 12) 또는 (12,), JOINT_ORDER 순서.

        M1 기본 동작: q만 slew rate limiter를 통과해 position target으로 실제 적용된다.
        dq/kp/kd/tau는 버퍼에 저장만 되고 시뮬레이션에는 반영되지 않는다(향후 faithful PD 모드 seam).

        Args:
            q: 목표 관절각 [rad].
            dq: 목표 관절각속도 [rad/s]. 현재 미적용.
            kp: 위치 게인. 현재 미적용(cfg 액추에이터 PD 고정).
            kd: 속도 게인. 현재 미적용.
            tau: 피드포워드 토크 [Nm]. 현재 미적용.
        """
        self._setpoint.copy_(self._to_tensor(q))
        self._dq_setpoint.copy_(self._to_tensor(dq))
        self._kp.copy_(self._to_tensor(kp))
        self._kd.copy_(self._to_tensor(kd))
        self._tau_ff.copy_(self._to_tensor(tau))

    def set_joint_plant(self, plant: int | bool, env_ids: torch.Tensor | None = None) -> None:
        """관절 물성 프리셋을 **런타임에** 전환한다.

        armature / viscous / coulomb 는 `ArticulationCfg` 필드가 아니라 PhysX 에 직접 write 해야
        하는 관절 속성이라 cfg 만으로는 바꿀 수 없다. 세 값을 한 번에 갈아끼우므로 프리셋 사이를
        오가도 상태가 섞이지 않는다.

        ⚠ kp/kd(25/0.5)는 건드리지 않는다 — PACE viscous 는 kd 오차를 흡수하도록 함께 식별된
        조합이라 게인을 따로 바꾸면 식별 결과가 깨진다.

        ⚠ 프리셋을 **섞지 말 것.** 세 파라미터는 결합 식별이라 한 세트의 coulomb 과 다른 세트의
        viscous 를 합치면 어떤 적합도 산출하지 않은 플랜트가 된다.

        Args:
            plant: :data:`PLANT_DEFAULT`(nominal, armature 0.01 마찰 0) / :data:`PLANT_SET2`(현재
                라이브 `PACE_*`) / :data:`PLANT_SET3`(2026-08-04 신규 캡처, 미채택 비교용).
                하위호환으로 ``bool`` 도 받는다 — True 는 :data:`PLANT_SET2` 로 해석한다.
            env_ids: 적용할 env. None 이면 전체.

        Raises:
            KeyError: 알 수 없는 프리셋 코드.
        """
        if isinstance(plant, bool):
            plant = PLANT_SET2 if plant else PLANT_DEFAULT
        armature_t, viscous_t, coulomb_t = self._plant_presets[int(plant)]
        self._plant_mode = int(plant)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        n = len(ids)
        env_ids_int = ids.to(torch.int32)
        armature = armature_t.unsqueeze(0).repeat(n, 1)
        viscous = viscous_t.unsqueeze(0).repeat(n, 1)
        coulomb = coulomb_t.unsqueeze(0).repeat(n, 1)
        self.robot.write_joint_armature_to_sim_index(
            armature=armature, joint_ids=self._all_joint_ids, env_ids=env_ids_int
        )
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=coulomb,
            joint_dynamic_friction_coeff=coulomb,
            joint_viscous_friction_coeff=viscous,
            joint_ids=self._all_joint_ids,
            env_ids=env_ids_int,
        )

    def set_camera_follow(self, follow: bool) -> None:
        """뷰포트 카메라를 로봇 추적 / 자유 조작 사이에서 전환한다.

        추적은 IsaacLab `ViewportCameraController` 의 `origin_type="asset_root"` 를 쓴다(매 렌더
        스텝마다 eye/lookat 을 로봇 base 기준으로 다시 잡아준다).

        자유 모드는 `update_view_to_world()` 를 **부르지 않는다** — 그러면 카메라가 cfg 의 기본
        위치로 튀어버려서, 보던 각도를 유지한 채 조작을 넘겨받고 싶은 의도와 어긋난다. 대신
        `origin_type` 만 "world" 로 바꿔 추적 콜백을 멈춘다(콜백이 이 값만 본다).

        headless 에서는 `viewport_camera_controller` 가 None 이라 조용히 no-op 이다.

        Args:
            follow: True 면 로봇 추적, False 면 자유 조작.
        """
        ctrl = self.viewport_camera_controller
        if ctrl is None:
            return
        if follow:
            ctrl.cfg.origin_type = "asset_root"
            ctrl.update_view_to_asset_root(self.cfg.viewer.asset_name or "robot")
        else:
            ctrl.cfg.origin_type = "world"

    def set_pedi_markers(self, leg: int, target_b: tuple[float, float, float]) -> None:
        """pedipulation 조작 발과 목표 위치를 구 두 개로 표시한다.

        **빨간 구 = 조작 발의 현재 위치**, **초록 구 = 명령받은 목표**. 둘이 붙어 있으면 추종이
        된 것이고, 벌어진 거리가 곧 추종 오차다.

        발 위치는 GUI 가 보내지 않는다 — 여기서 ``body_pos_w`` 로 직접 안다. 목표는 base frame
        으로 받아 현재 base pose 로 world 에 옮긴다(GUI 가 base pose 를 알 필요가 없게).

        Args:
            leg: 조작 다리 인덱스 0~3 (FL, FR, RL, RR). 음수면 마커를 숨긴다.
            target_b: 목표 위치, base frame [m].
        """
        if leg < 0:
            if self._pedi_markers is not None:
                self._pedi_markers.set_visibility(False)
            return

        markers = self._pedi_markers
        if markers is None:
            # 첫 요청 때 만든다 — pedipulation 을 안 쓰는 세션에 prim 을 얹지 않기 위해서다.
            markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/R2SPedipulation",
                    markers={
                        "current": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
                        ),
                        "target": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 1.0, 0.05)),
                        ),
                    },
                )
            )
            self._pedi_markers = markers
            # 화면을 못 보는 환경(headless·가상 디스플레이)에서 마커가 실제로 만들어졌는지
            # 확인할 유일한 단서라 한 번만 찍는다.
            print(
                f"[R2SGo2Env] pedipulation 마커 생성: {markers.cfg.prim_path} "
                f"(prototype {markers.num_prototypes}개 — 0=현재발/빨강, 1=목표/초록)",
                flush=True,
            )

        root_pos_w = self.robot.data.root_link_pos_w[0]
        root_quat_w = self.robot.data.root_link_quat_w[0]
        foot_w = self.robot.data.body_link_pos_w[0, self._foot_ids[leg]]
        # ⚠ `_to_tensor` 는 12-벡터 전용(num_envs 로 expand)이라 3-벡터에 쓰면 안 된다.
        target_t = torch.as_tensor(target_b, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_w = quat_apply(root_quat_w.unsqueeze(0), target_t)[0] + root_pos_w

        markers.visualize(
            translations=torch.stack([foot_w, target_w], dim=0),
            marker_indices=[0, 1],  # 0 = current(빨강), 1 = target(초록)
        )
        markers.set_visibility(True)

        # 진단(`R2S_LOG_MARKER=1`): 화면을 못 보는 환경에서 구가 **제 자리에** 있는지 확인하는
        # 유일한 수단이다. world 로 보낸 목표를 같은 base pose 로 되돌려(quat_apply_inverse)
        # 입력 `target_b` 와 비교한다 — 어긋나면 쿼터니언 규약이 틀린 것이다. 발 위치도 base
        # frame 으로 같이 찍어 leg 인덱스가 의도한 다리를 가리키는지 본다.
        if self._log_marker:
            self._marker_log_n += 1
            if self._marker_log_n % 50 == 1:  # 50 tick(=1 s)마다 한 줄
                # ⚠ 인자 순서는 (quat, vec) 이고 quat 은 (x, y, z, w) 다 — `root_link_quat_w` 와
                #   같은 규약이라 그대로 넣는다(실기 IMU 의 wxyz 와 헷갈리지 말 것).
                back = quat_apply_inverse(root_quat_w.unsqueeze(0), (target_w - root_pos_w).unsqueeze(0))
                foot_b = quat_apply_inverse(root_quat_w.unsqueeze(0), (foot_w - root_pos_w).unsqueeze(0))
                err = float(torch.norm(back[0] - torch.as_tensor(target_b, device=self.device)).item())
                print(
                    f"[marker] leg={leg} target_b_in=({target_b[0]:+.3f},{target_b[1]:+.3f},{target_b[2]:+.3f}) "
                    f"roundtrip=({back[0, 0]:+.3f},{back[0, 1]:+.3f},{back[0, 2]:+.3f}) err={err:.6f} | "
                    f"foot_b=({foot_b[0, 0]:+.3f},{foot_b[0, 1]:+.3f},{foot_b[0, 2]:+.3f})",
                    flush=True,
                )

    def close(self) -> None:
        """추적을 먼저 끄고 닫는다.

        `DirectRLEnv.close()` 는 `del self.scene` 을 `del self.viewport_camera_controller` **보다
        먼저** 한다. 그 사이에 Kit 의 post-update 콜백이 한 번 더 발화하면
        `update_view_to_asset_root` 가 `self._env.scene` 을 만지며 AttributeError 를 뱉는다
        (`viewport_camera_controller.py:164`). 추적을 쓰지 않던 시절엔 콜백이 즉시 반환해 드러나지
        않던 순서 문제라, 코어를 건드리지 않고 여기서 먼저 멈춘다.
        """
        self.set_camera_follow(False)
        super().close()

    def get_lowstate(self) -> dict:
        """현재 lowstate 반환 (numpy). num_envs==1 가정 (CONTRACT §5, §4.2).

        Returns:
            dict with keys:
                q, dq, ddq, tau_est: 각 길이 12, JOINT_ORDER 순서.
                imu: 길이 10 — quat_w,quat_x,quat_y,quat_z, gyro_xyz, acc_xyz.
        """
        q = self.robot.data.joint_pos[0, self._joint_ids]
        dq = self.robot.data.joint_vel[0, self._joint_ids]
        ddq = self.robot.data.joint_acc[0, self._joint_ids]
        tau_est = self.robot.data.applied_torque[0, self._joint_ids]

        # root_quat_w는 IsaacLab 6.0부터 (x,y,z,w) 순서. Unitree LowState는 (w,x,y,z) 순이라 재배열.
        quat_xyzw = self.robot.data.root_quat_w[0]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        gyro = self.robot.data.root_ang_vel_b[0]
        # base link 선가속도: body_lin_acc_w(world frame, shape (num_envs, num_bodies, 3))를
        # base id로 인덱싱한 뒤 root 회전의 역으로 body frame으로 변환(accelerometer 관례).
        acc_w = self.robot.data.body_lin_acc_w[0, self._base_id]
        acc_b = quat_apply_inverse(quat_xyzw.unsqueeze(0), acc_w.unsqueeze(0)).squeeze(0)
        imu = torch.cat([quat_wxyz, gyro, acc_b])

        return {
            "q": q.cpu().numpy(),
            "dq": dq.cpu().numpy(),
            "ddq": ddq.cpu().numpy(),
            "tau_est": tau_est.cpu().numpy(),
            "imu": imu.cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # DirectRLEnv 필수 메서드
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        # fix_base 적용 (CONTRACT §5): env 빌드 시점에 cfg.fix_base를 읽어야
        # sim_runner_go2.py가 --fix_base로 cfg 생성 후 덮어쓴 값도 반영된다.
        # configclass가 robot cfg를 인스턴스마다 deepcopy하므로 전역 UNITREE_GO2_CFG 오염 없음.
        if self.cfg.fix_base:
            self.cfg.robot.spawn.articulation_props.fix_root_link = True
            self.cfg.robot.init_state.pos = (0.0, 0.0, FIXED_BASE_HEIGHT_M)
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.sysid:
            # sysid 모드: actions = 절대 관절 목표각 [rad], **articulation 관절 순서**.
            # PACE(fit.py / collect_chirp_sim.py)가 관절 인덱스로 직접 채워 넣는 규약이며,
            # slew limiter는 chirp 고주파를 왜곡하므로 우회한다.
            self.robot.set_joint_position_target_index(target=actions)
            return
        del actions  # live 모드: setpoint은 self._setpoint에서 직접 주입
        # Slew rate limiter: 한 step에서 max_step 이상 이동 불가
        #
        # ⚠ **학습 env 에는 이 limiter 가 없다.** `go2_pedipulation_env` 는
        # `set_joint_position_target(...)` 로 목표를 그대로 넣는다. 정책이 매 step 크게 흔드는
        # 목표를 여기서만 자르면 r2s 는 학습과 다른 plant 가 되어 추종이 나빠진다.
        # `R2S_NO_SLEW=1` 은 그 영향을 재는 **진단용 우회**다(sysid 모드가 chirp 왜곡을 피해
        # 우회하는 것과 같은 이유). 실기에서는 켜 두는 것이 안전하다 — 목표 점프를 그대로
        # 내보내면 모터가 위험하다.
        raw_delta = self._setpoint - self._prev_setpoint
        delta = raw_delta if self._no_slew else raw_delta.clamp(-self._max_step, self._max_step)
        smoothed = self._prev_setpoint + delta
        self._prev_setpoint = smoothed.clone()
        self.robot.set_joint_position_target_index(target=smoothed, joint_ids=self._joint_ids)

        # 진단(`R2S_LOG_SLEW=1`): slew limiter 가 실제로 목표를 자르는지, 그리고 목표 추종
        # 오차와 토크가 얼마인지. **학습 env 에는 이 limiter 가 없다**(직접 target 주입)이므로,
        # 여기서 자르고 있다면 r2s 에서만 다른 plant 가 되어 추종이 나빠진다.
        if self._log_slew:
            self._slew_log_n += 1
            clipped = (raw_delta.abs() > self._max_step + 1e-9).sum().item()
            self._slew_clipped_total += int(clipped)
            self._slew_max_raw = max(self._slew_max_raw, float(raw_delta.abs().max().item()))
            if self._slew_log_n % 50 == 0:
                q_now = self.robot.data.joint_pos[:, self._joint_ids]
                track_err = float((self._setpoint - q_now).abs().max().item())
                tau = self.robot.data.applied_torque[:, self._joint_ids]
                print(
                    f"[slew] n={self._slew_log_n} clip누적={self._slew_clipped_total} "
                    f"max|Δ목표|={self._slew_max_raw:.4f} (한계 {float(self._max_step[0]):.3f}) "
                    f"max|목표-실제|={track_err:.4f} rad  max|τ|={float(tau.abs().max().item()):.2f} N·m",
                    flush=True,
                )
                self._slew_max_raw = 0.0

    def _apply_action(self) -> None:
        pass  # _pre_physics_step에서 처리

    def _get_observations(self) -> dict:
        pos = self.robot.data.joint_pos[:, self._joint_ids]
        vel = self.robot.data.joint_vel[:, self._joint_ids]
        torque = self.robot.data.applied_torque[:, self._joint_ids]
        return {"policy": torch.cat([pos, vel, torque], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # RL 없음 — stub
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """새 버퍼 초기화 — CLAUDE.md 전역 DO 규칙."""
        super()._reset_idx(env_ids)

        # 로봇을 cfg init_state(prone)로 되돌린다. `DirectRLEnv._reset_idx`는 articulation 상태를
        # 건드리지 않으므로, 이걸 하지 않으면 로봇이 USD rest(관절 ≈ 0, 다리 뻗은 자세)에 남고
        # PD 가 prone 목표(calf −2.6)까지 2.57 rad 를 끌어당기며 첫 스텝에 큰 킥이 생긴다.
        # nominal 물성에서는 흐물흐물 접히고 말아 무해했지만, PACE 물성에서는 관절 추종이 정확해져
        # 기체가 base_h 0.42 까지 튀어올랐다가 넘어진다.
        # sysid 모드는 제외한다 — PACE 적합 harness 가 상태를 직접 관리하고, 이미 검증된
        # (hold-out RMSE 0.017 rad) 적합 거동을 바꾸면 안 된다.
        #
        # **첫 reset 에서만** 쓴다. 이 env 는 RL 이 아니라 라이브 제어 harness 라 에피소드 경계에
        # 의미가 없는데, `episode_length_s=600` 때문에 10분마다 time_out reset 이 걸린다. 매번
        # write 하면 GUI 로 조작하던 로봇이 10분마다 spawn 자세로 순간이동한다 — 고치려던 것은
        # "sim 시작 시 초기자세가 적용되지 않는 것"이지 주기적 리셋이 아니다.
        if not self.cfg.sysid and not getattr(self, "_init_state_written", False):
            self._init_state_written = True
            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
            self.robot.write_joint_state_to_sim_index(
                position=self.robot.data.default_joint_pos[env_ids],
                velocity=self.robot.data.default_joint_vel[env_ids],
                env_ids=env_ids,
            )

        default_pos = self.robot.data.default_joint_pos[:, self._joint_ids]
        self._setpoint[env_ids] = default_pos[env_ids]
        self._prev_setpoint[env_ids] = default_pos[env_ids]
        self._dq_setpoint[env_ids] = 0.0
        self._kp[env_ids] = DEFAULT_KP
        self._kd[env_ids] = DEFAULT_KD
        self._tau_ff[env_ids] = 0.0

        # 관절 물성 write. reset 마다 다시 쓰는 이유는 go2_imitation_tracking_env 와 같다 —
        # 이 값들은 관절 속성이라 articulation 이 재초기화되면 되돌아간다.
        self.set_joint_plant(self._plant_mode, env_ids=env_ids)
