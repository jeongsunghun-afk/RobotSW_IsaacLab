# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Latent-space Imitation 환경 — body-frame 속도추종 command (vx, vy, yaw_rate).

`go2_imitation_tracking` 에서 **AMP discriminator 를 동결 motion-VAE 인코더의 latent
스타일 보상으로 교체**한 환경이다. 태스크 보상·리셋(RSI)·DR·RMA obs 구조는 그대로다.

  - Style reward: env 별 시간창 marginal KL (아래 §스타일 보상)
  - Task reward: lin_vel_reward(0.7) + yaw_vel_reward(0.3)
  - Reset: 항상 RSI (Reference State Initialization)
  - 알고리즘: RMA(ActorCriticRMA) + estimator + PPOLatent + OnPolicyRunnerLatent

스타일 보상
-----------

Phase 2 오프라인 검증에서 **스텝 로그밀도** ``-log N(z; mu_ref, Sigma_ref)`` 는 정지를
expert 보다 높게 채점하는 것이 실측됐다(클립 AUROC ``cat_stand`` 0.055). 그래서 보상은
**env 별 시간창의 marginal KL** 로 건다 — 정지하면 창 안 ``z`` 분산이 축퇴해
``var_e << var_ref`` 가 되어 KL 이 폭증한다::

    env e 의 최근 N 전이쌍에서   mu_e, var_e   (대각)
    D_e = KL( N(mu_e, diag var_e) || N(mu_ref, diag var_ref) )
    r_style_e = exp( -c_kl * (D_e - offset) )

배치 전체에 같은 상수를 주면 PPO advantage 에서 value baseline 에 흡수되어 정책 gradient
기여가 0 이다. 그래서 창을 **env 마다** 둔다.

``x_vae`` 49-D 레이아웃 (Phase 1 인코더가 학습된 순서 — 이쪽이 기준이다)::

    [0]      base height                       1
    [1:7]    heading-relative 6D 회전          6    (회전행렬 1·2 열)
    [7:10]   base 선속도 (body frame)          3
    [10:13]  base 각속도 (body frame)          3
    [13:25]  발 위치 (base-local, 4x3)        12
    [25:37]  관절각 (motion DOF_NAMES 순서)   12
    [37:49]  관절속도 (motion DOF_NAMES 순서) 12

★ AMP obs 의 ``root_rot_tan_norm`` 6D 는 이것과 **순열이 아니다** — AMP 는 회전행렬의
1·3 열(``R·x̂``, ``R·ẑ``)을, ``x_vae`` 는 1·2 열(``R·x̂``, ``R·ŷ``)을 쓴다. 인코더가
``x_vae`` 순서로 학습됐으므로 여기서는 ``x_vae`` 규약을 직접 계산한다.

★ 전이 쌍 간격은 인코더 학습 설정(stride 2 · predict_ahead 1 · 60 fps) 이 정한 **1/30 초**
이고 policy dt 는 1/50 초다. 그래서 ``x_vae`` 링버퍼에서 ``t - 1/30`` 지점을 **선형보간**해
``x_prev`` 를 만든다. 6D 회전 블록의 선형보간은 엄밀히 유효회전이 아니지만 0.02 초 구간
각변화가 작아 오차가 무시 가능하다.

Policy observation dict (실배포 가능 RMA 구조 — estimator가 policy(42)로부터 root 선속도·각속도를
추정하고, priv_explicit(6)는 학습 시 GT critic/estimator target 용):
  policy(42)        = projected_gravity_b(3) +
                       lin_vel_cmd(2) + yaw_vel_cmd(1) +
                       joint_pos - default(12) + joint_vel(12) + actions(12)
  priv_explicit(6)  = root_lin_vel_b * scale + root_ang_vel_b * scale
  priv_latent(19)   = armature/friction/mass/foot_fric/kp/kd/action_delay + encoder_bias(12)
  history(10, 42)   = policy proprio ring buffer (noised)
"""

from __future__ import annotations

import glob
import math
import os

import torch
import warp as wp  # IsaacLab 3.0: _ALL_INDICES is a warp array → wp.to_torch()
from rsl_rl.modules.motion_encoder import MotionEncoder

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_inverse

from .go2_imitation_latent_env_cfg import (
    PACE_ARMATURE,
    PACE_COULOMB,
    PACE_ENCODER_BIAS_MAG,
    PACE_VISCOUS,
    Go2ImitationLatentEnvCfg,
)
from .motion_lib import Go2MotionLib

#: ``x_vae`` 특징 차원. Phase 1 인코더가 이 폭으로 학습됐다.
X_VAE_DIM = 49

#: 저장소 루트 — cfg 의 상대 경로(체크포인트·참조 통계)를 푸는 기준.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 5))


class Go2ImitationLatentEnv(DirectRLEnv):
    """Go2 AMP + body-frame 속도추종 Imitation 환경 (RMA + estimator 아키텍처).

    Policy proprio (42-dim — priv_explicit로 분리한 root 속도 항은 제외):
        projected_gravity_b(3) +
        lin_vel_cmd(2) + yaw_vel_cmd(1) +
        joint_pos - default(12) + joint_vel(12) + actions(12)

    priv_explicit(6-dim, 노이즈 없는 GT — 학습 시 critic 입력 / estimator target):
        root_lin_vel_b * priv_explicit_lin_vel_scale
        root_ang_vel_b * priv_explicit_ang_vel_scale
        둘 다 policy obs에 없으므로 estimator는 **미관측 상태 추정**을 학습한다
        (각속도가 obs에 있던 이전 버전의 denoising 과제가 아니다).

    priv_latent(19-dim, quasi-static domain-rand 파라미터 — RMA priv encoder 입력):
        armature_scale(1) + joint_friction_scale(1) + base_mass_offset(1) +
        foot_friction_offset(1) + kp_scale(1) + kd_scale(1) +
        action_delay_norm(1) + encoder_bias_norm(12)

    history(10, 42): policy proprio ring buffer (노이즈 포함, 최신이 index 0).

    AMP disc observation (49-dim per step):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [R4: heading-relative 6D rotation]
    """

    cfg: Go2ImitationLatentEnvCfg

    # key body 이름 (발 4개) — motion_lib foot 순서와 일치
    KEY_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    #: 관절 회전축 — 이름 접미사로 고른다. Go2 는 hip 이 x, thigh/calf 가 y 축이다
    #: (MimicKit `data/assets/go2/go2.xml` 의 12 개 `<joint axis=...>` 로 확인).
    #: tan-norm 인코딩에서 축 선택은 **어느 성분이 상수가 되느냐**만 바꾼다 — 정보량은
    #: 어느 축을 쓰든 (cos θ, sin θ) 로 같고 첫 선형층이 고정 재배치를 흡수한다.
    JOINT_AXIS_BY_SUFFIX = {"hip": (1.0, 0.0, 0.0), "thigh": (0.0, 1.0, 0.0), "calf": (0.0, 1.0, 0.0)}

    def __init__(self, cfg: Go2ImitationLatentEnvCfg, render_mode: str | None = None, **kwargs):
        # policy obs 폭은 관절 인코딩에 달려 있다. `super().__init__` 이 `cfg.observation_space` 로
        # gym space 를 만들므로 **그 전에** 확정해야 하고, hydra 오버라이드는 env 생성 시점엔 이미
        # cfg 에 반영돼 있으므로 여기가 유일하게 맞는 자리다(cfg `__post_init__` 은 너무 이르다).
        # ⚠ 아래 두 키는 **파생값이다** — 여기서 무조건 덮어쓰므로 hydra 로 직접 오버라이드해도
        #   조용히 무시된다. 폭을 바꾸려면 `*_tan_norm` 플래그를 쓸 것.
        joint_pos_obs_dim = 12 * 6 if cfg.joint_pos_tan_norm else 12
        cfg.observation_space = 3 + 2 + 1 + joint_pos_obs_dim + 12 + 12

        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg

        # ── policy proprio 레이아웃 ──────────────────────────────
        # `_apply_obs_dr` 는 인덱스를 하드코딩하지 않고 이 값에서 유도한다. 예전에 각속도를
        # 뺐을 때 슬라이스만 밀린 채 남겨두면 σ 가 4 배로 잘못 주입되는 사고가 있었다.
        self._joint_pos_obs_dim = joint_pos_obs_dim
        self._obs_idx_joint_vel = 6 + joint_pos_obs_dim  # joint_vel 블록 시작
        self._obs_idx_actions = self._obs_idx_joint_vel + 12

        # ── 모션 라이브러리 ─────────────────────────────────────
        motion_file = self.cfg.motion_file
        if os.path.isdir(motion_file):
            pkl_files = sorted(glob.glob(os.path.join(motion_file, "*.pkl")))
            assert pkl_files, f"PKL 파일 없음: {motion_file}"
            motion_files = pkl_files
        else:
            motion_files = [motion_file]
        self._motion_lib = Go2MotionLib(motion_files=motion_files, device=self.device)

        # ── body / joint 인덱스 ──────────────────────────────────
        self.ref_body_index = self._robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self._robot.data.body_names.index(name) for name in self.KEY_BODY_NAMES]

        # hip(abduction) 관절 인덱스 — 액션 범위 축소용
        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(self._robot.data.joint_names) if "hip" in n], dtype=torch.long, device=self.device
        )

        # ── tan-norm 인코딩용 관절 회전축 [12, 3] ────────────────
        # 꺼져 있으면 **빈 텐서**다 — `_joint_tan_norm` 은 jit 함수라 Optional 을 못 받으므로
        # `numel() == 0` 을 off 신호로 쓴다.
        axes = []
        for name in self._robot.data.joint_names:
            match = [ax for suffix, ax in self.JOINT_AXIS_BY_SUFFIX.items() if suffix in name]
            if len(match) != 1:
                raise ValueError(
                    f"관절 '{name}' 의 회전축을 정할 수 없다 (매칭 {len(match)} 건). "
                    f"`JOINT_AXIS_BY_SUFFIX` 에 접미사를 추가할 것."
                )
            axes.append(match[0])
        _axis = torch.tensor(axes, dtype=torch.float32, device=self.device)  # [12, 3]
        _empty = torch.zeros(0, 3, dtype=torch.float32, device=self.device)
        self._joint_axis = _axis if self.cfg.joint_pos_tan_norm else _empty

        # ── motion_lib ↔ IsaacLab joint 순서 매핑 ────────────────
        # IsaacLab joint 순서(알파벳 등)와 PKL DOF_NAMES 순서가 다를 수 있음.
        # go2_amp_env.py의 motion_dof_indexes 패턴과 동일.
        robot_joint_names = list(self._robot.data.joint_names)
        try:
            self._motion_dof_indices = self._motion_lib.get_dof_index(robot_joint_names)
            print(f"[Go2ImitationLatentEnv] IsaacLab joint 순서: {robot_joint_names}")
            print(f"[Go2ImitationLatentEnv] motion_dof_indices: {self._motion_dof_indices}")
        except AssertionError:
            print("[Go2ImitationLatentEnv] DOF 이름 불일치 — 순서대로 1:1 매핑 사용")
            self._motion_dof_indices = list(range(12))

        # ── Velocity tracking command state ─────────────────────
        self._lin_vel_cmd = torch.zeros(self.num_envs, 2, device=self.device)  # (vx, vy) — vy 항상 0
        # ④ 정지 명령으로 강제된 env (`rel_standing_envs`). `_resample_steering` 이 갱신한다.
        self._standing_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._yaw_vel_cmd = torch.zeros(self.num_envs, device=self.device)  # yaw rate [rad/s]
        self._tar_timer = torch.zeros(self.num_envs, device=self.device)  # [N] seconds to change

        # ── x_vae ↔ motion DOF 순서 역순열 ───────────────────────
        # `_motion_dof_indices[i]` 는 IsaacLab 관절 i 의 motion 인덱스다. `x_vae` 는 motion
        # `DOF_NAMES` 순서이므로 **역순열**이 필요하다. 손으로 유도하지 않고 argsort 로 만들고
        # 참조 프레임 왕복으로 검증한다(보고서 V1a).
        _mdi = torch.tensor(self._motion_dof_indices, dtype=torch.long, device=self.device)
        self._x_vae_dof_indices = torch.argsort(_mdi)  # motion 순서 → IsaacLab 인덱스

        # ── 동결 motion-VAE 인코더 + 참조 통계 ────────────────────
        lat = self.cfg.latent
        ckpt_path = lat.encoder_ckpt if os.path.isabs(lat.encoder_ckpt) else os.path.join(_REPO_ROOT, lat.encoder_ckpt)
        ref_path = lat.ref_stats if os.path.isabs(lat.ref_stats) else os.path.join(_REPO_ROOT, lat.ref_stats)
        self._motion_encoder = MotionEncoder.from_checkpoint(ckpt_path, device=self.device, freeze=True)
        _ref = torch.load(ref_path, map_location=self.device, weights_only=False)
        if int(_ref["latent_dim"]) != self._motion_encoder.latent_dim or int(_ref["x_dim"]) != X_VAE_DIM:
            raise ValueError(
                f"참조 통계와 인코더가 어긋난다: ref latent {_ref['latent_dim']} / x {_ref['x_dim']} vs "
                f"encoder latent {self._motion_encoder.latent_dim} / x {self._motion_encoder.x_dim}"
            )
        self._ref_stats = _ref
        self._mu_ref = _ref["mu_ref"].to(self.device)
        self._var_ref = _ref["var_ref"].to(self.device)
        self._var_floor = float(_ref["var_floor"])
        # cfg 가 None 이면 참조 통계가 데이터에서 정한 값을 쓴다(권장).
        self._c_kl = float(lat.c_kl) if lat.c_kl is not None else float(_ref["c_kl"])
        self._kl_offset = float(lat.kl_offset) if lat.kl_offset is not None else float(_ref["offset"])
        self._window_n = int(lat.window_n)

        # 전이 쌍 간격 [s] — 인코더 학습 설정이 정한다. policy dt 의 정수배가 아니므로 보간한다.
        self._pair_dt = float(_ref["pair_dt"])
        _lag = self._pair_dt / self.step_dt
        self._x_lag_lo = int(math.floor(_lag))
        self._x_lag_frac = float(_lag - self._x_lag_lo)
        _depth = self._x_lag_lo + 2
        print(
            f"[Go2ImitationLatentEnv] 전이 간격 {self._pair_dt * 1000:.2f} ms = {_lag:.4f} × policy dt"
            f" → x_vae 링버퍼 깊이 {_depth} (보간 가중치 {self._x_lag_frac:.4f})"
        )
        print(
            f"[Go2ImitationLatentEnv] window_n {self._window_n} · c_kl {self._c_kl:.5f} ·"
            f" offset {self._kl_offset:.4f} nat · 참조 train 세션 {len(_ref['train_sessions'])}개"
        )

        # ── x_vae 링버퍼 (index 0 = 최신) ─────────────────────────
        self._x_vae_hist = torch.zeros(self.num_envs, _depth, X_VAE_DIM, device=self.device)
        self._x_vae_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── env 별 z 시간창 (index 0 = 최신) ──────────────────────
        self._z_window = torch.zeros(self.num_envs, self._window_n, self._motion_encoder.latent_dim, device=self.device)
        self._z_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 창이 안 찬 env 는 0 이 아니라 **중립값**(expert 중앙 창의 보상 = 1.0)을 받는다.
        # 0 을 주면 리셋 직후 창이 찰 때까지 모든 env 가 벌점을 받아 보상이 에피소드 나이와
        # 상관되고, 종료율과 스타일이 뒤섞인다.
        self._style_reward = torch.full((self.num_envs,), float(lat.neutral_reward), device=self.device)
        self._latent_kl = torch.zeros(self.num_envs, device=self.device)

        # ── PACE 식별 파라미터 + Domain Randomization 초기화 ─────
        self._init_domain_rand()

        # ── RMA proprio history 링버퍼 (policy obs 42-dim × history_len) ────
        self._proprio_history = torch.zeros(
            self.num_envs, self.cfg.history_len, self.cfg.observation_space, device=self.device
        )

    # ──────────────────────────────────────────────────────────
    # IsaacLab DirectRLEnv 필수 메서드
    # ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

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
        # Env isolation: filter cross-env collisions unconditionally (GPU too). In IsaacLab 3.0
        # the auto-filter path (interactive_scene:218) is skipped when the scene cfg declares no
        # entities (has_scene_cfg_entities=False), so the old cpu-only guard left GPU runs
        # unfiltered — robots from different envs physically collide. Ref: IsaacLab #1918.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self._episode_sums = {
            "lin_vel_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "yaw_vel_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        if self.cfg.standing_style_substitute:
            # ④ 진단용 — **보상이 아니다.** style 대체가 실제로 물리는지 학습 중에 보려면 필요하다
            # (pose 보상은 extras 경로라 `Episode_Reward/*` 에 안 잡힌다). 판정은
            #   standing_base_h / standing_frac = 정지 중 평균 base 높이
            # 로 하며, 0.22(웅크림) → 0.32(default 기립) 로 올라가야 기전이 작동한 것이다.
            for _k in ("standing_pose_reward", "standing_frac", "standing_base_h"):
                self._episode_sums[_k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        actions = self._actions.clone()
        if self.cfg.hip_scale_reduction:
            actions[:, self._hip_joint_ids] *= 0.5
        target = self.cfg.action_scale * actions + self._robot.data.default_joint_pos
        # DR: action(토크 명령) 지연 — per-env 지연 스텝만큼 과거 target 을 적용
        if self.cfg.domain_rand and self.cfg.dr.randomize_action_delay:
            self._action_delay_buf = torch.cat([target.unsqueeze(1), self._action_delay_buf[:, :-1]], dim=1)
            rows = torch.arange(self.num_envs, device=self.device)
            self._processed_actions = self._action_delay_buf[rows, self._action_delay_steps]
        else:
            self._processed_actions = target

    def _post_physics_step(self):
        # 목표 속도 타이머 업데이트
        self._tar_timer -= self.step_dt
        change_mask = self._tar_timer <= 0.0
        if change_mask.any():
            change_ids = change_mask.nonzero(as_tuple=False).flatten()
            self._resample_steering(change_ids)

        # DR: 주기적 외란 push (base 수평 속도 킥)
        if self.cfg.domain_rand and self.cfg.dr.push_robot:
            self._push_timer -= self.step_dt
            push_ids = (self._push_timer <= 0.0).nonzero(as_tuple=False).flatten()
            if push_ids.numel() > 0:
                self._push_robots(push_ids)
                self._push_timer[push_ids] = self.cfg.dr.push_interval_s

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        root_pos_w = self._robot.data.body_pos_w[:, self.ref_body_index]  # [N,3]
        # IsaacLab 3.0+: body_quat_w is (x,y,z,w). motion_lib quats (wxyz) are converted at
        # the consumption boundary inside `_reset_strategy_rsi` / `reference_x_vae`.
        root_quat_w = self._robot.data.body_quat_w[:, self.ref_body_index]  # [N,4] xyzw
        # IsaacLab 3.0: ArticulationData props return ProxyArray; .torch needed at
        # @torch.jit.script boundaries (`_compute_x_vae`). Indexed accessors above
        # (body_pos_w[:, i], body_quat_w[:, i]) already unwrap to torch via __getitem__.
        root_lin_vel_b = self._robot.data.root_lin_vel_b.torch  # [N,3]
        root_ang_vel_b = self._robot.data.root_ang_vel_b.torch  # [N,3]

        # ── 발 위치 (local frame) ─────────────────────────────
        rel_pos = self._robot.data.body_pos_w[:, self.key_body_indexes] - root_pos_w.unsqueeze(1)
        N, K = rel_pos.shape[:2]
        local_key_body_pos = quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
            rel_pos.reshape(-1, 3),
        ).view(N, K, 3)

        # ── x_vae 계산 → 동결 인코더 → env 별 z 창 → 스타일 보상 ──
        x_vae = _compute_x_vae(
            self._robot.data.joint_pos.torch[:, self._x_vae_dof_indices],
            self._robot.data.joint_vel.torch[:, self._x_vae_dof_indices],
            root_pos_w,
            root_quat_w,
            root_lin_vel_b,
            root_ang_vel_b,
            local_key_body_pos,
        )
        self._push_x_vae(x_vae)
        self._update_style_reward()

        self.extras = {
            # 러너가 AMP disc reward 자리에 그대로 넣는 env 별 스타일 보상 [N]
            "style_reward": self._style_reward.clone(),
            "latent_kl": self._latent_kl.clone(),
            # 창이 찬 env 마스크 — 로깅에서 warm-up env 를 빼려면 필요하다.
            "style_ready": (self._z_count >= self._window_n).clone(),
        }

        # ④ 정지 구간 style 대체 — 러너의 보상 융합에 쓰인다:
        #     total = lerp·task + (1−lerp)·(style_weight·style + (1−style_weight)·style_substitute)
        # 두 키가 없으면 러너는 style_weight=1 / substitute=0 으로 보므로 다른 AMP task 는 불변.
        if self.cfg.standing_style_substitute:
            _pose_r = self._compute_standing_pose_reward()
            self.extras["style_weight"] = (~self._standing_mask).float()
            self.extras["style_substitute"] = _pose_r
            # 진단 누적. 여기서 하는 이유: `_reset_idx`(누적 초기화)가 이 시점보다 먼저 돌므로
            # 리셋된 env 는 새 에피소드분부터 쌓인다. `standing_base_h / standing_frac` 가 정지 중
            # 평균 base 높이다(둘 다 같은 상수로 정규화되므로 비율은 그대로 [m]).
            _stand_f = self._standing_mask.float()
            self._episode_sums["standing_pose_reward"] += _pose_r
            self._episode_sums["standing_frac"] += _stand_f
            self._episode_sums["standing_base_h"] += _stand_f * root_pos_w[:, 2]

        # ── Policy proprio (42-dim) ────────────────────────────────────────────
        # priv_explicit로 분리된 신호는 policy obs에서 **제외**한다 — estimator가 추정하는
        # 대상을 actor에게 직접 보여주면 추정 구조가 무의미해지기 때문이다.
        # root_lin_vel_b(실측 불가)와 root_ang_vel_b(priv_explicit로 이동)가 모두 빠졌다.
        # 관절각 노이즈(`joint_pos_noise`)와 엔코더 오프셋(`encoder_bias`)은 **라디안 공간에서**
        # 더한 뒤 인코딩한다. tan-norm 출력에 라디안을 더하면 물리적 의미가 사라지기 때문이다.
        # raw 경로에서는 인코딩이 항등이라 예전처럼 블록에 더하는 것과 bit-identical 하다.
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos  # [N, 12] rad
        if self.cfg.domain_rand:
            if self.cfg.dr.obs_noise:
                joint_pos_rel = joint_pos_rel + torch.randn_like(joint_pos_rel) * self.cfg.dr.joint_pos_noise
            if self.cfg.dr.encoder_bias:
                joint_pos_rel = joint_pos_rel + self._encoder_bias

        proprio = torch.cat(
            [
                self._robot.data.projected_gravity_b,  # 3
                self._lin_vel_cmd,  # 2 (vx, vy)
                self._yaw_vel_cmd.unsqueeze(-1),  # 1
                self._encode_joint_pos(joint_pos_rel),  # 12 (raw) 또는 72 (tan-norm)
                self._robot.data.joint_vel,  # 12
                self.actions,  # 12
            ],
            dim=-1,
        )  # total = 42 (raw) 또는 102 (tan-norm)
        policy_obs = self._apply_obs_dr(proprio)

        # ── priv_explicit(6) — GT root 선속도/각속도 (critic/estimator target, 노이즈 없음) ──
        # 두 블록 모두 policy obs에 없다 → estimator는 proprio(42)+history로부터
        # **미관측 상태 추정**을 학습하고, critic은 GT를 본다(asymmetric actor-critic).
        priv_explicit = torch.cat(
            [
                root_lin_vel_b * self.cfg.priv_explicit_lin_vel_scale,  # 3
                root_ang_vel_b * self.cfg.priv_explicit_ang_vel_scale,  # 3
            ],
            dim=-1,
        )  # total = 6

        # ── priv_latent(19) — quasi-static domain-rand 파라미터 ──
        priv_latent = self._get_priv_latent()

        # ── proprio history 링버퍼 갱신 (노이즈 포함된 policy_obs 저장) ──
        self._proprio_history = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([policy_obs] * self.cfg.history_len, dim=1),
            torch.cat([self._proprio_history[:, 1:], policy_obs.unsqueeze(1)], dim=1),
        )

        return {
            "policy": policy_obs,
            "priv_explicit": priv_explicit,
            "priv_latent": priv_latent,
            "history": self._proprio_history,
        }

    def _compute_standing_pose_reward(self) -> torch.Tensor:
        """정지 명령 env 에서 style reward 를 대신할 **default pose 유지 보상**.

        ``r = w · exp(−scale · mean((q − q_default)²))``, 정지 env 만 값이 있고 나머지는 0 이다.

        왜 style 을 대체하는가: cmd 0 구간에서 discriminator 가 **웅크린 정지를 빠른 보행보다
        참조답다고 평가**해(0.904 vs 0.735) style 이 흡수 상태를 강화하고 있었다. 참조 18클립이
        전부 걷는 중이라 disc 를 고칠 데이터가 없으므로, 그 구간만 튜닝 가능한 신호로 바꾼다.

        기체 높이 목표를 따로 두지 않는다 — default 관절각이 서 있는 높이를 물리적으로 결정하고,
        관측된 실패("관절이 굽어 base_h 0.217")가 곧 관절각 오차이기 때문이다.
        """
        err = torch.mean((self._robot.data.joint_pos - self._robot.data.default_joint_pos) ** 2, dim=-1)
        r = self.cfg.standing_pose_reward_w * torch.exp(-self.cfg.standing_pose_err_scale * err)
        return torch.where(self._standing_mask, r, torch.zeros_like(r))

    def _get_rewards(self) -> torch.Tensor:
        # ── (1) lin_vel tracking — body frame 직접 비교 ──────────
        lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]  # (vx, vy) actual
        lin_vel_err = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
        lin_vel_reward = torch.exp(-self.cfg.vel_err_scale * lin_vel_err)

        # ── (2) yaw_vel tracking ─────────────────────────────────
        yaw_vel_b = self._robot.data.root_ang_vel_b[:, 2]
        yaw_vel_err = (self._yaw_vel_cmd - yaw_vel_b) ** 2
        yaw_vel_reward = torch.exp(-self.cfg.yaw_vel_err_scale * yaw_vel_err)

        reward = self.cfg.lin_vel_reward_w * lin_vel_reward + self.cfg.yaw_vel_reward_w * yaw_vel_reward

        self._episode_sums["lin_vel_reward"] += lin_vel_reward
        self._episode_sums["yaw_vel_reward"] += yaw_vel_reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            base_height = self._robot.data.body_pos_w[:, self.ref_body_index, 2]
            died = base_height < self.cfg.termination_height

            flipped = self._robot.data.projected_gravity_b[:, 2] > 0.0
            died = died | flipped

            grav_b = self._robot.data.projected_gravity_b
            roll = torch.atan2(grav_b[:, 1], -grav_b[:, 2])
            pitch = torch.atan2(-grav_b[:, 0], torch.sqrt(grav_b[:, 1] ** 2 + grav_b[:, 2] ** 2))
            roll_limit = self.cfg.roll_termination_deg * math.pi / 180.0
            pitch_limit = self.cfg.pitch_termination_deg * math.pi / 180.0
            bad_orientation = (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)
            died = died | bad_orientation

            if hasattr(self, "contact_sensor"):
                contact_forces = self.contact_sensor.data.net_forces_w
                if contact_forces is not None and contact_forces.numel() > 0:
                    bad_contacts = self._get_body_contact(
                        contact_forces,
                        self._robot.data.body_names,
                        "base",
                        threshold=self.cfg.contact_force_threshold,
                    )
                    died = died | bad_contacts

            # 첫 스텝 직후부터만 termination 적용 (physics 정착 시간)
            died = died & (self.episode_length_buf > 1)
        else:
            died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)  # IsaacLab 3.0: _ALL_INDICES is wp.array

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 항상 RSI
        root_state, joint_pos, joint_vel = self._reset_strategy_rsi(env_ids)  # type: ignore[arg-type]

        # IsaacLab 3.0: partial-env writes use the *_index API with keyword args.
        self._robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)

        # 속도 명령 재샘플링 (버퍼 초기화 — 불변규칙 §3)
        self._resample_steering(env_ids)  # type: ignore[arg-type]

        # RMA proprio history 리셋 (실질적인 재구성은 다음 _get_observations의
        # episode_length_buf<=1 분기에서 현재 proprio로 복제되지만, 방어적으로 0 초기화)
        self._proprio_history[env_ids] = 0.0

        # ── latent 스타일 상태 리셋 ──────────────────────────────
        # ★ 참조 모션으로 채우지 않는다. RSI 가 amp 버퍼를 참조로 채웠던 것처럼 하면 리셋
        #   직후 expert 품질 z 가 창에 들어가 `r_style` 이 부풀려진다. 유효 개수를 0 으로
        #   되돌리고 그동안은 중립값을 준다.
        self._x_vae_hist[env_ids] = 0.0
        self._x_vae_count[env_ids] = 0
        self._z_window[env_ids] = 0.0
        self._z_count[env_ids] = 0
        self._style_reward[env_ids] = self.cfg.latent.neutral_reward
        self._latent_kl[env_ids] = 0.0

        # Episode 통계 리셋 및 로깅
        extras: dict = {}
        for key in self._episode_sums:
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])  # type: ignore[index]
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0  # type: ignore[index]

        if "log" not in self.extras:
            self.extras["log"] = {}  # type: ignore[assignment]
        self.extras["log"].update(extras)  # type: ignore[union-attr]

        # PACE 플랜트 반영 + per-env Domain Randomization 재샘플
        self._apply_domain_rand(env_ids)  # type: ignore[arg-type]

    # ──────────────────────────────────────────────────────────
    # PACE 파라미터 + Domain Randomization
    # ──────────────────────────────────────────────────────────

    def _init_domain_rand(self):
        """PACE nominal 플랜트 텐서 + DR per-env 상태 버퍼 초기화.

        관절 타입(hip/thigh/calf)별 식별값을 robot joint order 로 펼쳐두고, per-env 스케일·
        encoder bias·action delay·push 타이머 버퍼를 만든다. 실제 sim write 는 리셋마다
        :meth:`_apply_domain_rand` 에서 수행한다 (각 env = 서로 다른 로봇).
        """
        device = self.device
        names = list(self._robot.data.joint_names)
        nj = self._robot.num_joints

        # PACE nominal (robot joint order). 미식별 관절은 nominal(armature 0.01, 마찰 0) 유지.
        self._pace_armature = torch.full((nj,), 0.01, device=device)
        self._pace_viscous = torch.zeros(nj, device=device)
        self._pace_coulomb = torch.zeros(nj, device=device)
        for key in ("hip", "thigh", "calf"):
            ids = [i for i, n in enumerate(names) if n.endswith(f"_{key}_joint")]
            self._pace_armature[ids] = PACE_ARMATURE[key]
            self._pace_viscous[ids] = PACE_VISCOUS[key]
            self._pace_coulomb[ids] = PACE_COULOMB[key]
        self._all_joint_ids = torch.arange(nj, dtype=torch.int32, device=device)

        # 게인 원본 (DCMotor 는 actuator.stiffness/damping 텐서로 토크를 계산 → 이 텐서를 직접 스케일)
        # calf 는 effort_limit 이 달라 별도 actuator 그룹이므로 **전 그룹**을 돌아야 한다.
        # 한 그룹만 잡으면 게인 DR 이 calf 에 걸리지 않는다.
        self._acts = list(self._robot.actuators.values())
        self._base_kp = [a.stiffness.clone() for a in self._acts]
        self._base_kd = [a.damping.clone() for a in self._acts]
        self._act = self._acts[0]  # 하위호환: 단일 그룹을 가정하던 외부 도구용

        # per-env DR 상태
        self._encoder_bias = torch.zeros(self.num_envs, 12, device=device)
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        d_cap = max(1, int(self.cfg.dr.max_action_delay_steps))
        self._action_delay_buf = self._robot.data.default_joint_pos.unsqueeze(1).repeat(1, d_cap + 1, 1).clone()
        self._push_timer = torch.full((self.num_envs,), self.cfg.dr.push_interval_s, device=device)

        # mass DR 기준값 (base 링크 원본 질량)
        self._base_body_id = self._robot.data.body_names.index("base")
        _bid = self._base_body_id
        self._default_base_mass = self._robot.data.body_mass.torch[:, _bid : _bid + 1].clone()

        # ── RMA priv_latent 버퍼 (per-env, quasi-static DR 파라미터) ────────────
        # nominal 값(해당 DR 항목이 off 이거나 아직 샘플되지 않은 env)은 항상 이 초기값 유지:
        # scale류 = 1.0(무작위화 없음), offset류 = 0.0(nominal 대비 편차 없음).
        # ActorCriticRMA는 priv_latent 키를 무조건 요구하므로 DR off 여도 항상 내보내야 함.
        self._priv_armature_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_joint_friction_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_base_mass_offset = torch.zeros(self.num_envs, 1, device=device)  # [kg]
        self._priv_foot_friction_offset = torch.zeros(self.num_envs, 1, device=device)  # nominal(1.0) 대비 오프셋
        self._priv_kp_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_kd_scale = torch.ones(self.num_envs, 1, device=device)

        self._dr_ready = False  # 첫 전체 reset 에서 startup material DR 적용 후 True

    def _sample(self, rng: tuple[float, float], n: int, d: int) -> torch.Tensor:
        """[n, d] uniform 샘플 (rng=(lo, hi))."""
        lo, hi = rng
        return torch.rand(n, d, device=self.device) * (hi - lo) + lo

    def _apply_domain_rand(self, env_ids: torch.Tensor):
        """PACE 플랜트 반영 + per-env DR 재샘플 (리셋된 env_ids 대상)."""
        env_ids_long = env_ids.to(torch.long)
        env_ids_int = env_ids.to(torch.int32)
        n = int(env_ids_long.numel())
        dr = self.cfg.dr

        # ── PACE armature / viscous / Coulomb (+ per-env 스케일) ──
        if self.cfg.use_pace_params:
            arm_s = (
                self._sample(dr.armature_scale_range, n, 1) if (self.cfg.domain_rand and dr.randomize_armature) else 1.0
            )
            jf_s = (
                self._sample(dr.joint_friction_scale_range, n, 1)
                if (self.cfg.domain_rand and dr.randomize_joint_friction)
                else 1.0
            )
            armature = self._pace_armature.unsqueeze(0).repeat(n, 1) * arm_s
            viscous = self._pace_viscous.unsqueeze(0).repeat(n, 1) * jf_s
            coulomb = self._pace_coulomb.unsqueeze(0).repeat(n, 1) * jf_s
            self._robot.write_joint_armature_to_sim_index(
                armature=armature, joint_ids=self._all_joint_ids, env_ids=env_ids_int
            )
            self._robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=coulomb,
                joint_dynamic_friction_coeff=coulomb,
                joint_viscous_friction_coeff=viscous,
                joint_ids=self._all_joint_ids,
                env_ids=env_ids_int,
            )
            # priv_latent 버퍼 갱신: 실제로 랜덤화된 env_ids만 (스칼라 nominal=1.0 인 경우는 미기록,
            # 초기값이 이미 nominal 이므로 무의미한 write 생략)
            if self.cfg.domain_rand and dr.randomize_armature:
                self._priv_armature_scale[env_ids_long] = arm_s
            if self.cfg.domain_rand and dr.randomize_joint_friction:
                self._priv_joint_friction_scale[env_ids_long] = jf_s

        if not self.cfg.domain_rand:
            return

        # ── startup-only: 마찰(material) 랜덤화 (첫 전체 reset 에서 모든 env) ──
        if not self._dr_ready:
            self._randomize_material_startup()
            self._dr_ready = True

        # ── base payload 질량 ──
        if dr.randomize_mass:
            add = self._sample(dr.added_base_mass_range, n, 1)
            new_mass = torch.clamp(self._default_base_mass[env_ids_long] + add, min=1e-6)
            body_ids = torch.tensor([self._base_body_id], dtype=torch.int32, device=self.device)
            self._robot.set_masses_index(masses=new_mass, body_ids=body_ids, env_ids=env_ids_int)
            # priv_latent: 실제 적용된(clamp 반영) nominal 대비 오프셋 [kg]
            self._priv_base_mass_offset[env_ids_long] = new_mass - self._default_base_mass[env_ids_long]

        # ── PD 게인 스케일 (DCMotor actuator 텐서 직접 수정) ──
        if dr.randomize_gains:
            kp_scale = self._sample(dr.kp_scale_range, n, 1)
            kd_scale = self._sample(dr.kd_scale_range, n, 1)
            for _a, _kp0, _kd0 in zip(self._acts, self._base_kp, self._base_kd):
                _a.stiffness[env_ids_long] = _kp0[env_ids_long] * kp_scale
                _a.damping[env_ids_long] = _kd0[env_ids_long] * kd_scale
            self._priv_kp_scale[env_ids_long] = kp_scale
            self._priv_kd_scale[env_ids_long] = kd_scale

        # ── encoder bias / action delay / push 타이머 ──
        if dr.encoder_bias:
            rand_bias = torch.rand(n, 12, device=self.device) * 2.0 - 1.0
            self._encoder_bias[env_ids_long] = rand_bias * PACE_ENCODER_BIAS_MAG
        if dr.randomize_action_delay:
            self._action_delay_steps[env_ids_long] = torch.randint(
                0, dr.max_action_delay_steps + 1, (n,), device=self.device
            )
            self._action_delay_buf[env_ids_long] = self._robot.data.default_joint_pos[env_ids_long].unsqueeze(1)
        self._push_timer[env_ids_long] = dr.push_interval_s

    def _randomize_material_startup(self):
        """모든 env 의 강체 마찰(static·dynamic)을 per-env 값으로 랜덤화 (startup 1회).

        코어 ``events.randomize_rigid_body_material`` 의 PhysX 경로와 동일한 API 사용
        (root_view + warp). 전체 shape 에 동일 계수를 적용해 "로봇별 접지 마찰"을 근사한다.
        """
        dr = self.cfg.dr
        materials = wp.to_torch(self._robot.root_view.get_material_properties())  # [num_envs, num_shapes, 3] CPU
        lo, hi = dr.foot_friction_range
        fr = torch.rand(materials.shape[0], 1, device="cpu") * (hi - lo) + lo  # per-env 마찰
        materials[..., 0] = fr  # static friction
        materials[..., 1] = fr  # dynamic friction
        env_ids = torch.arange(materials.shape[0], device="cpu", dtype=torch.int32)
        self._robot.root_view.set_material_properties(
            wp.from_torch(materials.contiguous(), dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
        )
        # priv_latent: nominal 마찰(env_cfg static_friction=1.0) 대비 오프셋으로 저장 (전 env, startup 1회뿐)
        self._priv_foot_friction_offset[:, 0] = fr.to(self.device).squeeze(-1) - 1.0

    def _push_robots(self, env_ids: torch.Tensor):
        """base 에 랜덤 수평 속도 킥을 주어 외란(impulse)을 흉내낸다."""
        n = int(env_ids.numel())
        lin = self._robot.data.root_lin_vel_w[env_ids].clone()  # [n,3] world
        ang = self._robot.data.root_ang_vel_w[env_ids].clone()  # [n,3] world
        kick = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self.cfg.dr.max_push_vel_xy
        lin[:, 0:2] += kick
        vel = torch.cat([lin, ang], dim=-1)  # [n,6]
        self._robot.write_root_com_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids.to(torch.int32))

    def _encode_joint_pos(self, joint_pos_rel: torch.Tensor) -> torch.Tensor:
        """관절각 [N, 12] rad → policy obs 블록. raw 는 항등, tan-norm 은 [N, 72].

        tan-norm 은 MimicKit ``torch_util.quat_to_tan_norm`` 과 같은 식이다: 관절 회전
        ``q = angle_axis(a_j, θ_j)`` 로 기준 tan(1,0,0)·norm(0,0,1) 을 돌려 6 차원을 만든다.

        Args:
            joint_pos_rel: 기본자세 상대 관절각 [rad], shape [num_envs, 12].

        Returns:
            shape [num_envs, 12] (raw) 또는 [num_envs, 72] (tan-norm).
        """
        if not self.cfg.joint_pos_tan_norm:
            return joint_pos_rel
        return _joint_tan_norm(joint_pos_rel, self._joint_axis)

    def _apply_obs_dr(self, obs: torch.Tensor) -> torch.Tensor:
        """policy proprio 에 **중력·관절속도** 관측 노이즈를 더한다. AMP obs/priv_explicit 는 불변.

        레이아웃 — root_lin_vel_b(실측 불가)와 root_ang_vel_b(priv_explicit로 이동)가 모두
        policy obs에서 빠져 있다. GT priv_explicit는 노이즈 없이 그대로 사용한다:
            projected_gravity_b[0:3] +
            lin_vel_cmd[3:5] + yaw_vel_cmd[5:6] +
            joint_pos 블록[6 : 6+D] + joint_vel[6+D : 18+D] + actions[18+D : 30+D]
        여기서 D 는 12(raw) 또는 72(tan-norm)다.

        ⚠ 관절각 노이즈와 ``encoder_bias`` 는 여기가 아니라 ``_get_observations`` 에서 **라디안
        공간**에 더한다 — tan-norm 인코딩 후에는 라디안을 더할 수 없기 때문이다.

        ⚠ 인덱스를 하드코딩하지 말 것. 예전에 각속도를 obs 에서 뺐을 때 슬라이스를 밀린 채
        남겨두면 σ=0.2 노이즈가 σ=0.05 인 projected_gravity_b 에 주입돼 4 배 증폭됐다.
        ``dr.ang_vel_noise`` 는 주입할 자리가 없어 dead config 다.
        """
        if not self.cfg.domain_rand:
            return obs
        dr = self.cfg.dr
        noise = torch.zeros_like(obs)
        if dr.obs_noise:
            n = self.num_envs
            jv = self._obs_idx_joint_vel
            noise[:, 0:3] = torch.randn(n, 3, device=self.device) * dr.gravity_noise  # projected_gravity_b
            noise[:, jv : jv + 12] = torch.randn(n, 12, device=self.device) * dr.joint_vel_noise  # joint_vel
        return obs + noise

    def _get_priv_latent(self) -> torch.Tensor:
        """RMA priv_latent(19-dim) 조립 — quasi-static domain-rand 파라미터, 매 env 항상 반환.

        DR 항목이 off 이거나 아직 샘플되지 않은 env는 버퍼 nominal 값(scale=1.0, offset=0.0)을
        그대로 반환한다 (ActorCriticRMA가 priv_latent 키를 무조건 요구하므로).
        """
        return torch.cat(
            [
                self._priv_armature_scale,  # 1
                self._priv_joint_friction_scale,  # 1
                self._priv_base_mass_offset,  # 1  [kg]
                self._priv_foot_friction_offset,  # 1
                self._priv_kp_scale,  # 1
                self._priv_kd_scale,  # 1
                (self._action_delay_steps.float() / max(1, self.cfg.dr.max_action_delay_steps)).unsqueeze(-1),  # 1
                self._encoder_bias / PACE_ENCODER_BIAS_MAG,  # 12
            ],
            dim=-1,
        )  # (N, 19)

    # ──────────────────────────────────────────────────────────
    # 리셋 전략
    # ──────────────────────────────────────────────────────────

    def _reset_strategy_rsi(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference State Initialization — 모션 데이터로 초기화."""
        n = len(env_ids)
        motion_ids = self._motion_lib.sample_motions(n)
        start = "start" in self.cfg.reset_strategy
        if start:
            times = torch.zeros(n, device=self.device)
        else:
            times = self._motion_lib.sample_times(motion_ids)

        root_pos, root_quat, lin_vel_b, ang_vel_b, dof_pos, dof_vel, _ = self._motion_lib.calc_motion_frame(
            motion_ids, times
        )

        # root_state [N, 13]: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        # motion_lib returns wxyz; IsaacLab 6.0 sim and quat_apply expect xyzw.
        root_quat_xyzw = convert_quat(root_quat, to="xyzw")
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = root_pos + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_quat_xyzw  # xyzw

        # motion_lib 속도는 body frame → world frame 변환 필요
        root_state[:, 7:10] = quat_apply(root_quat_xyzw, lin_vel_b)
        root_state[:, 10:13] = quat_apply(root_quat_xyzw, ang_vel_b)

        joint_pos_out = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel_out = self._robot.data.default_joint_vel[env_ids].clone()
        # motion data → IsaacLab joint 순서로 재배열
        joint_pos_out[:, :12] = dof_pos[:, self._motion_dof_indices]
        joint_vel_out[:, :12] = dof_vel[:, self._motion_dof_indices]

        # 마지막 RSI 샘플 기록 — 검증 스크립트가 "이 프레임의 참조 x_vae" 를 다시 만들 수
        # 있어야 sim↔motion 채널 편차(V1b)를 잴 수 있다.
        self._last_rsi_motion_ids = motion_ids
        self._last_rsi_times = times
        self._last_rsi_env_ids = env_ids

        # ── rest-init: 일부 env 는 참조 프레임 대신 **기립 정지**로 시작한다 ──────────────
        # 참조 18 클립은 전부 "이미 걷고 있는" 상태라(가장 느린 go2_walk 도 평균 0.15 m/s),
        # RSI 만 쓰면 정지가 초기 조건이 된 적이 없다 → 정지→이동 전이를 배울 기회가 없다.
        if self.cfg.rel_rest_init > 0.0:
            rest = torch.rand(n, device=self.device) < self.cfg.rel_rest_init
            if rest.any():
                default_root = self._robot.data.default_root_state[env_ids][rest].clone()
                # default_root_state 는 env-local 좌표라 origin 을 더해야 한다(RSI 경로와 동일).
                default_root[:, 0:3] = default_root[:, 0:3] + self.scene.env_origins[env_ids][rest]
                default_root[:, 7:13] = 0.0  # lin/ang velocity = 0
                root_state[rest] = default_root
                joint_pos_out[rest] = self._robot.data.default_joint_pos[env_ids][rest]
                joint_vel_out[rest] = 0.0
                # latent 창은 리셋에서 비워지고 실측으로만 채워지므로, rest-init env 도
                # 참조 모션이 섞이지 않는다 (AMP 경로에 있던 ~0.1% 오염이 여기엔 없다).

        return root_state, joint_pos_out, joint_vel_out

    # ──────────────────────────────────────────────────────────
    # 속도 명령 관리 (재샘플링)
    # ──────────────────────────────────────────────────────────

    def _resample_steering(self, env_ids: torch.Tensor):
        """lin_vel_cmd(vx, vy), yaw_vel_cmd, 타이머를 재샘플링합니다."""
        n = len(env_ids)

        # vx: [lin_vel_x_min, lin_vel_x_max]
        self._lin_vel_cmd[env_ids, 0] = (
            torch.rand(n, device=self.device) * (self.cfg.lin_vel_x_max - self.cfg.lin_vel_x_min)
            + self.cfg.lin_vel_x_min
        )
        # vy: [lin_vel_y_min, lin_vel_y_max] — 기본 (0.0, 0.0) 이므로 항상 0
        self._lin_vel_cmd[env_ids, 1] = (
            torch.rand(n, device=self.device) * (self.cfg.lin_vel_y_max - self.cfg.lin_vel_y_min)
            + self.cfg.lin_vel_y_min
        )

        # yaw rate: [yaw_vel_min, yaw_vel_max]
        self._yaw_vel_cmd[env_ids] = (
            torch.rand(n, device=self.device) * (self.cfg.yaw_vel_max - self.cfg.yaw_vel_min) + self.cfg.yaw_vel_min
        )

        # 일부 env 는 **정확히 0**(정지) 명령으로 강제한다. uniform 샘플의 near-zero tail 만으로는
        # 정지가 뚜렷한 모드로 학습되지 않는다(cmd 0 에서 task reward 는 v=0 일 때 1.0 최대인데도
        # 정책이 그 상태에서 빠져나오지 못했다). IsaacLab 표준 velocity command 의
        # `rel_standing_envs` 와 같은 장치.
        if self.cfg.rel_standing_envs > 0.0:
            standing = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
            self._lin_vel_cmd[env_ids[standing]] = 0.0
            self._yaw_vel_cmd[env_ids[standing]] = 0.0
            # ④ style 대체용 마스크. **명령 기준**이라 정책이 스스로 멈춰서 pose 보상을 수확할 수
            # 없다(실측 속도로 게이트하면 그게 가능해진다). 다음 재샘플까지 유지된다.
            self._standing_mask[env_ids] = standing
        else:
            self._standing_mask[env_ids] = False

        # 다음 변경까지 남은 시간
        self._tar_timer[env_ids] = (
            torch.rand(n, device=self.device) * (self.cfg.tar_change_time_max - self.cfg.tar_change_time_min)
            + self.cfg.tar_change_time_min
        )

    # ──────────────────────────────────────────────────────────
    # AMP 인터페이스 (OnPolicyRunnerAMPBase 필수)
    # ──────────────────────────────────────────────────────────

    def reference_x_vae(self, motion_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """참조 모션 프레임의 ``x_vae`` [n, 49] — motion_lib 경로.

        ``_get_observations`` 의 sim 실측 경로와 **같은 레이아웃**을 낸다. 두 경로의 값을
        같은 프레임에서 대조하는 것이 게이트 V1b 다 (sim↔motion 채널 편차).

        Args:
            motion_ids: 모션 인덱스, shape [n].
            times: 모션 시각 [s], shape [n].
        """
        root_pos, root_quat_wxyz, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos = self._motion_lib.calc_motion_frame(
            motion_ids, times
        )
        root_quat_xyzw = convert_quat(root_quat_wxyz, to="xyzw")
        return _compute_x_vae(dof_pos, dof_vel, root_pos, root_quat_xyzw, lin_vel, ang_vel, foot_pos)

    # ──────────────────────────────────────────────────────────
    # latent 스타일 보상
    # ──────────────────────────────────────────────────────────

    def _push_x_vae(self, x_vae: torch.Tensor) -> None:
        """``x_vae`` 링버퍼를 한 칸 밀고 최신 프레임을 넣는다 (index 0 = 최신)."""
        self._x_vae_hist = torch.cat([x_vae.unsqueeze(1), self._x_vae_hist[:, :-1]], dim=1)
        self._x_vae_count = torch.clamp(self._x_vae_count + 1, max=self._x_vae_hist.shape[1])

    def _update_style_reward(self) -> None:
        """전이 쌍 → ``z`` → env 별 창 → ``D_e`` → ``r_style``.

        전이 간격이 policy dt 의 정수배가 아니므로 ``x_prev`` 는 링버퍼의 두 프레임을
        선형보간해 만든다. 링버퍼가 아직 안 찬 env 는 ``z`` 를 창에 밀지 않는다.
        """
        lo = self._x_lag_lo
        f = self._x_lag_frac
        x_prev = (1.0 - f) * self._x_vae_hist[:, lo] + f * self._x_vae_hist[:, lo + 1]
        x_curr = self._x_vae_hist[:, 0]
        z = self._motion_encoder.inference(x_prev, x_curr)  # [N, latent_dim]

        # 링버퍼가 찬 env 만 창에 밀어 넣는다 (리셋 직후 보간이 무효인 구간 배제)
        pushable = self._x_vae_count >= self._x_vae_hist.shape[1]  # [N]
        shifted = torch.cat([z.unsqueeze(1), self._z_window[:, :-1]], dim=1)
        self._z_window = torch.where(pushable[:, None, None], shifted, self._z_window)
        self._z_count = torch.where(pushable, torch.clamp(self._z_count + 1, max=self._window_n), self._z_count)

        # 창 marginal KL (대각 닫힌형)
        mu_e = self._z_window.mean(dim=1)
        var_e = self._z_window.var(dim=1, unbiased=False).clamp(min=self._var_floor)
        d_e = 0.5 * (
            torch.log(self._var_ref / var_e) + (var_e + (mu_e - self._mu_ref).pow(2)) / self._var_ref - 1.0
        ).sum(dim=-1)

        ready = self._z_count >= self._window_n
        self._latent_kl = torch.where(ready, d_e, torch.zeros_like(d_e))
        r = torch.exp(-self._c_kl * (d_e - self._kl_offset))
        self._style_reward = torch.where(ready, r, torch.full_like(r, self.cfg.latent.neutral_reward))

    # ──────────────────────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────────────────────

    def _get_body_contact(
        self,
        contact_forces: torch.Tensor,
        body_names: list[str],
        keyword: str,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        indexes = [i for i, name in enumerate(body_names) if keyword in name.lower()]
        if not indexes:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if contact_forces.shape[1] >= max(indexes) + 1:
            forces = contact_forces[:, indexes, :]
            return torch.norm(forces, dim=-1).max(dim=-1).values > threshold
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


# ──────────────────────────────────────────────────────────────
# JIT 컴파일 함수
# ──────────────────────────────────────────────────────────────


@torch.jit.script
def _joint_tan_norm(theta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """관절각 → 관절별 회전의 tan-norm (MimicKit ``quat_to_tan_norm`` 과 같은 값).

    회전축 a 둘레로 θ 만큼 돌린 회전을 기준 tan(1,0,0)·norm(0,0,1) 에 적용해 관절당 6 차원을
    만든다. 쿼터니언을 거치지 않고 로드리게스 식 ``v·cosθ + (a×v)·sinθ + a(a·v)(1−cosθ)`` 를
    직접 쓴다 — 결과는 같고(단위 테스트로 확인) jit 안에서 안전하다.

    **policy obs 와 disc obs 가 이 함수 하나를 공유한다.** 두 경로가 따로 구현되면 언젠가
    갈라지는데, disc 는 정책 obs 와 달리 expert(참조 모션) 쪽과도 대칭이어야 해서 치명적이다.

    Args:
        theta: 관절각 [rad], shape [batch, num_joints].
        axis: 관절별 단위 회전축, shape [num_joints, 3].

    Returns:
        shape [batch, num_joints * 6] — 관절마다 [tan(3), norm(3)].
    """
    n, j = theta.shape[0], theta.shape[1]
    c = torch.cos(theta).unsqueeze(-1)  # [n, j, 1]
    s = torch.sin(theta).unsqueeze(-1)
    a = axis.unsqueeze(0).expand(n, j, 3)  # [n, j, 3]
    out = torch.zeros(n, j, 6, dtype=theta.dtype, device=theta.device)
    for k in range(2):
        ref = torch.zeros(3, dtype=theta.dtype, device=theta.device)
        ref[0 if k == 0 else 2] = 1.0  # tan=(1,0,0), norm=(0,0,1)
        v = ref.view(1, 1, 3).expand(n, j, 3)
        rot = v * c + torch.cross(a, v, dim=-1) * s + a * (a * v).sum(-1, keepdim=True) * (1.0 - c)
        out[:, :, 3 * k : 3 * k + 3] = rot
    return out.reshape(n, j * 6)


@torch.jit.script
def _heading_relative_rot6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """heading(yaw) 을 제거한 6D 회전 표현 [N, 6] — 회전행렬의 1·2 열.

    ``train_go2_motion_vae.py:heading_relative_rot6d`` (numpy, wxyz) 과 같은 값을 낸다.
    yaw-only 쿼터니언의 켤레를 **왼쪽**에 곱해 heading 을 제거한 뒤 ``[R·x, R·y]`` 를 쌓는다.

    ★ AMP 의 ``root_rot_tan_norm`` 은 ``[R·x, R·z]`` 라 이것과 **순열이 아니다**.
    인코더가 이 규약으로 학습됐으므로 여기가 기준이다.

    Args:
        quat_xyzw: root 쿼터니언 (x, y, z, w), shape [N, 4].
    """
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    hw, hz = torch.cos(-0.5 * yaw), torch.sin(-0.5 * yaw)
    rw = hw * w - hz * z
    rx = hw * x - hz * y
    ry = hw * y + hz * x
    rz = hw * z + hz * w
    col0 = torch.stack([1 - 2 * (ry * ry + rz * rz), 2 * (rx * ry + rw * rz), 2 * (rx * rz - rw * ry)], dim=-1)
    col1 = torch.stack([2 * (rx * ry - rw * rz), 1 - 2 * (rx * rx + rz * rz), 2 * (ry * rz + rw * rx)], dim=-1)
    return torch.cat([col0, col1], dim=-1)


@torch.jit.script
def _compute_x_vae(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    foot_pos_local: torch.Tensor,
) -> torch.Tensor:
    """동결 인코더 입력 ``x_vae`` (49-D) 계산.

    ★ live(정책) · expert(참조 모션) **두 경로가 전부 이 함수를 지난다.** 스타일 채점은
    두 분포를 같은 자로 재야 의미가 있으므로 인코딩을 여기 한 곳에만 둔다.

    Args:
        dof_pos: 관절각 [rad], **motion ``DOF_NAMES`` 순서**, shape [N, 12].
        dof_vel: 관절속도 [rad/s], 같은 순서, shape [N, 12].
        root_pos: base 위치 [m], shape [N, 3] (z 성분만 쓴다).
        root_quat_xyzw: base 자세 (x, y, z, w), shape [N, 4].
        root_lin_vel_b: base 선속도 [m/s], body frame, shape [N, 3].
        root_ang_vel_b: base 각속도 [rad/s], body frame, shape [N, 3].
        foot_pos_local: 발 위치 [m], base-local, shape [N, 4, 3] (FL, FR, RL, RR).

    Returns:
        shape [N, 49] — ``[height 1][rot6d 6][lin_vel 3][ang_vel 3][foot 12][q 12][qd 12]``.
    """
    return torch.cat(
        [
            root_pos[:, 2:3],  # 1
            _heading_relative_rot6d(root_quat_xyzw),  # 6
            root_lin_vel_b,  # 3
            root_ang_vel_b,  # 3
            foot_pos_local.reshape(foot_pos_local.shape[0], -1),  # 12
            dof_pos,  # 12
            dof_vel,  # 12
        ],
        dim=-1,
    )
