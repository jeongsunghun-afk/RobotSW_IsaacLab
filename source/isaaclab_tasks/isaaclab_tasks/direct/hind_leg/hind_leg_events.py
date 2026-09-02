# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""hind_leg 전용 도메인 랜덤화 term — 게인·armature·관절마찰을 **한 인자로 연동** 스케일.

왜 코어 term 두 개(:class:`randomize_actuator_gains` + :class:`randomize_joint_parameters`)로
안 되는가: 두 term 은 서로 **독립적으로** 표본을 뽑으므로 "게인과 armature 가 같은 배수"라는
구속을 표현할 수 없다. 그런데 우리가 겨냥하는 불확실성은 정확히 그 구속 위에 있다.

배경 — 외부 기어단의 지수 문제 (2026-09-02 기전 정정: 드라이버 "오설정"이 아니라 모터 뒤에
붙은 실제 기어단이다. RL_INTERFACE.md §4):
    드라이버가 모터 내장 감속기(전 축 7:1)까지만 보고하므로 실효 관절강성이 명목 kp 의 **k^n** 배다
    (gear_k = 실제감속비/7 = hip·thigh 1.0 / calf 1.5 / foot 1.2). 지수 n 이 1 인지 2 인지는
    현 데이터로 **원리적으로 판정 불가**다 — 게인 오차가 armature·마찰의 재스케일로 흡수돼
    잔차에 남는 신호가 잡음 바닥의 1/3~1/21 이다.

    ★핵심: 진짜 ``n=1`` 은 ``(armature, viscous, coulomb, kp, kd)`` 가 **전부** ×1/k 인 점이다.
      **게인만** ×1/k 하면 다른 로봇이 된다:

      | 조작                    | ω_n            | ζ    |
      |-------------------------|----------------|------|
      | 전부 ×1/k (진짜 n=1)    | n=2 와 **같음** | 같음 |
      | 게인만 ×1/k             | √(1/k) 배 하락  | 같음 |

      따라서 게인만 흔드는 DR 은 일반 강건성은 주지만 **지수 불확실성을 겨냥하지 못한다**.
      이 term 은 다섯 물성에 **같은 인자**를 곱해 그 다양체 위에서만 움직인다.

또한 이 term 은 별개 갭도 함께 닫는다: 종전 ``EventCfg`` 에는 ``randomize_joint_parameters`` 가
없어서 **PACE 가 식별하는 주요 물성 두 개(armature·관절마찰)가 전혀 랜덤화되지 않고 있었다.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

# ⚠ `isaaclab.assets.Articulation` / `isaaclab.actuators.ImplicitActuator` 를 **모듈 최상단에서
#   import 하지 않는다.** 이 모듈은 `hind_leg_env_cfg` 가 최상단에서 import 하고, 학습 스크립트는
#   env cfg 를 `SimulationApp` 생성 **전에** 해석한다. 그 시점에 물리 백엔드/pxr 바인딩을 끌어오면
#   Kit 이 뒤늦게 같은 모듈을 다시 적재하며 `free(): invalid pointer` 로 프로세스가 죽는다
#   (2026-08-14 실측: HindLeg 학습만 100% abort, 같은 스크립트로 Cartpole 은 정상).
#   Articulation 은 타입 주석 전용이라 TYPE_CHECKING 으로 충분하고, ImplicitActuator 는
#   런타임 isinstance 검사에만 쓰이므로 호출 시점에 지역 import 한다.

# 관절 종류별 gear_k (RL_INTERFACE.md §4 = 실제감속비/7). 지수 하한은 1/gear_k 다.
#   hip/thigh 는 k=1 이라 지수와 무관 → s ≡ 1.
GEAR_K: dict[str, float] = {"hip": 1.0, "thigh": 1.0, "calf": 1.5, "foot": 1.2}

# 관절명에서 종류를 뽑는 순서 (부분문자열 매칭). 긴 것부터 볼 필요는 없다 — 서로 겹치지 않는다.
_JOINT_KINDS: tuple[str, ...] = ("hip", "thigh", "calf", "foot")


class randomize_coupled_plant_scale(ManagerTermBase):
    """관절별 인자 하나로 게인·armature·관절마찰을 **연동** 스케일하는 event term.

    관절 ``j``, 환경 ``e`` 에 대해::

        factor[e, j] = s_exponent[e, kind(j)] × u_general[e, j]

    * ``s_exponent`` — 지수 불확실성. **베르누이**다 (불확실성이 이진이므로 연속구간보다 정확):
      hip/thigh ``s=1`` 고정, calf ``s ∈ {1/1.5, 1}``, foot ``s ∈ {1/1.2, 1}``.
      ⚠ **좌우 다리는 같은 s** — 같은 로봇이니 지수가 다리마다 다를 수 없다. 관절 종류별로
      한 번 뽑아 HL/HR 에 공통 적용한다 (:paramref:`exponent_shared_across_kinds` 도 참고).
    * ``u_general`` — 기존 플랜트 DR. 관절마다 독립으로 ``log_uniform`` 표본.

    적용 대상은 ``actuator.stiffness`` / ``actuator.damping`` / ``armature`` /
    관절마찰(static·dynamic·viscous) 다섯이며 **모두 같은 factor** 를 받는다.

    코어 두 term 의 규약을 그대로 따른다 — **생성 시점에 캐시한 default 로 리셋한 뒤 scale**.
    이 규약을 어기면 다른 term 과 조합할 때 값이 복리로 누적되며 조용히 깨진다.

    ⚠ **`data.default_joint_*` 캐시도 함께 갱신한다** (코어 ``randomize_joint_parameters`` 는 sim
      에만 쓴다). ``hind_leg_env`` 의 raw 좌표 foot 마찰(``foot_raw_friction``)이 foot 의
      viscous/Coulomb 슬롯을 **그 캐시에서** b_raw/c_raw 로 읽기 때문이다 —
      ``_clear_foot_joint_friction()`` 이 foot 의 PhysX 마찰을 매 제어 스텝 0 으로 눌러 두므로
      sim 쪽에만 쓰면 **foot 마찰 DR 이 통째로 사라진다**. armature 도 같은 이유로 캐시를 갱신한다
      (raw 마찰의 안정 캡이 ``default_joint_armature`` 를 읽는다).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        num_joints = self.asset.num_joints
        device = self.asset.device

        # 이 term 은 전 관절을 대상으로 한다 (종류 분류는 관절명으로 내부에서 한다).
        # 부분집합을 주면 인덱싱만 복잡해지고 얻는 게 없다 — 명시적으로 막는다.
        if self.asset_cfg.joint_ids != slice(None) and len(self.asset_cfg.joint_ids) != num_joints:
            raise ValueError(
                f"{type(self).__name__} 은 전 관절 대상 term 이다 — asset_cfg.joint_names 는 '.*' 여야 한다. "
                f"(받은 대상 {len(self.asset_cfg.joint_ids)}개 / 전체 {num_joints}개)"
            )

        names = list(self.asset.data.joint_names)
        # 관절 종류별 articulation 열 인덱스 + 지수 하한.
        self._kind_cols: dict[str, list[int]] = {}
        for j, name in enumerate(names):
            kind = next((k for k in _JOINT_KINDS if k in name), None)
            if kind is None:
                raise ValueError(f"관절명에서 종류를 판별할 수 없다: {name!r} (기대 종류 {_JOINT_KINDS})")
            self._kind_cols.setdefault(kind, []).append(j)
        # 지수가 걸리는 종류만 남긴다 (gear_k > 1).
        self._exp_kinds: list[str] = [k for k in self._kind_cols if GEAR_K[k] > 1.0]

        # ---- default 캐시 (생성 시점 = 어떤 랜덤화보다 먼저) ----
        # ⚠ armature/마찰은 `data.default_*` 에서 읽는다 — raw 마찰 모델이 b_raw/c_raw 의 저장소로
        #   쓰는 바로 그 텐서라서 같은 대상을 스케일해야 정합이다. lazy clone 이므로 여기서 최초
        #   접근하면 현재 sim 값(= cfg 값)이 복제된다. hind_leg_env 가 foot 마찰을 0 으로 누르기
        #   **전**이어야 하며, event manager 생성이 첫 물리 스텝보다 앞이므로 순서는 보장된다.
        self._def_armature = self._t(self.asset.data.default_joint_armature).clone()
        self._def_fric_static = self._t(self.asset.data.default_joint_friction_coeff).clone()
        self._def_fric_visc = self._t(self.asset.data.default_joint_viscous_friction_coeff).clone()
        # dynamic 은 default 컨테이너가 없다(6.0) → 현재 sim 값을 기준으로 잡는다.
        self._def_fric_dyn = self._t(self.asset.data.joint_dynamic_friction_coeff).clone()

        # explicit actuator 의 진짜 게인은 actuator.stiffness/damping 텐서다 (PD 를 파이썬에서 계산).
        self._def_kp: dict[str, torch.Tensor] = {}
        self._def_kd: dict[str, torch.Tensor] = {}
        self._act_cols: dict[str, torch.Tensor] = {}
        for aname, actuator in self.asset.actuators.items():
            ids = actuator.joint_indices
            ids = list(range(num_joints)) if isinstance(ids, slice) else [int(i) for i in ids]
            self._act_cols[aname] = torch.as_tensor(ids, dtype=torch.long, device=device)
            self._def_kp[aname] = actuator.stiffness.clone()
            self._def_kd[aname] = actuator.damping.clone()

        self._all_joint_ids_i32 = torch.arange(num_joints, dtype=torch.int32, device=device)

    @staticmethod
    def _t(x) -> torch.Tensor:
        """6.0 data 컨테이너(warp 프론트엔드)의 torch 뷰 — torch 텐서면 그대로."""
        return x.torch if hasattr(x, "torch") else x

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        general_distribution_params: tuple[float, float] = (0.75, 1.5),
        distribution: Literal["log_uniform", "uniform"] = "log_uniform",
        exponent_shared_across_kinds: bool = False,
    ) -> None:
        """Args:
        env: 환경 인스턴스.
        env_ids: 랜덤화할 환경 인덱스. None 이면 전체.
        asset_cfg: 대상 asset (전 관절, ``joint_names='.*'``).
        general_distribution_params: ``u_general`` 표본 구간 (배율). 기존 플랜트 DR 값.
        distribution: ``u_general`` 분포. 배율이므로 ``log_uniform`` 이 기본.
        exponent_shared_across_kinds: True 면 calf·foot 이 **같은 베르누이 비트**를 쓴다.
            지수 n 은 드라이버 펌웨어 하나의 성질이므로 물리적으로는 이쪽이 맞다(좌우를
            공통으로 두는 것과 같은 논거). 기본 False = 종류별 독립 추첨(지정 스펙).
        """
        del asset_cfg  # __init__에서 이미 해석해 뒀다
        device = self.asset.device
        num_joints = self.asset.num_joints
        ids: torch.Tensor = (torch.arange(env.scene.num_envs, device=device) if env_ids is None else env_ids).to(
            device=device, dtype=torch.long
        )
        n = int(ids.numel())
        if n == 0:
            return

        # ---- u_general: 관절마다 독립 표본 ----
        lo, hi = float(general_distribution_params[0]), float(general_distribution_params[1])
        if lo <= 0.0 or hi < lo:
            raise ValueError(f"general_distribution_params 가 배율로 부적절하다: ({lo}, {hi})")
        if distribution == "log_uniform":
            u = torch.exp(
                torch.rand(n, num_joints, device=device) * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo)))
                + torch.log(torch.tensor(lo))
            )
        else:
            u = torch.rand(n, num_joints, device=device) * (hi - lo) + lo

        # ---- s_exponent: 관절 종류별 베르누이, 좌우 공통 ----
        # 종류마다 (n, 1) 비트를 뽑아 그 종류의 **모든** 열(HL·HR)에 같은 값을 흩뿌린다 →
        # 좌우가 자동으로 같은 s 를 받는다.
        s = torch.ones(n, num_joints, device=device)
        shared_bit = torch.randint(0, 2, (n, 1), device=device, dtype=torch.bool)
        for kind in self._exp_kinds:
            bit = (
                shared_bit
                if exponent_shared_across_kinds
                else torch.randint(0, 2, (n, 1), device=device, dtype=torch.bool)
            )
            s_low = 1.0 / GEAR_K[kind]
            s_kind = torch.where(
                bit, torch.ones_like(bit, dtype=torch.float), torch.full_like(bit, s_low, dtype=torch.float)
            )
            s[:, self._kind_cols[kind]] = s_kind  # (n,1) → (n, len(cols)) 브로드캐스트

        factor = s * u  # (n, num_joints)

        # ---- 적용: default × factor (코어 규약 — 캐시된 default 로 리셋 후 scale) ----
        env_ids_i32 = ids.to(torch.int32)
        armature = self._def_armature[ids] * factor
        fric_static = torch.clamp(self._def_fric_static[ids] * factor, min=0.0)
        fric_visc = torch.clamp(self._def_fric_visc[ids] * factor, min=0.0)
        fric_dyn = torch.clamp(self._def_fric_dyn[ids] * factor, min=0.0)
        # dynamic ≤ static (코어와 동일 보정). ⚠ 이 클램프가 물리면 fric_dyn 만 factor 를 못 받아
        # **다양체 밖으로 나간다** — "다섯 물성이 같은 배수"라는 이 term 의 전제가 깨진다.
        # 현 cfg 는 static=dynamic=0.38 이라 같은 배수를 곱한 뒤에도 계속 같아서 클램프가 no-op 다
        # (2026-08-14 실측: dynamic>static 인 관절 0개). 경계에 정확히 걸쳐 있으므로, dynamic 을
        # static 보다 크게 바꾸면 이 전제가 조용히 깨진다는 점만 기억할 것.
        fric_dyn = torch.minimum(fric_dyn, fric_static)

        self.asset.write_joint_armature_to_sim_index(
            armature=armature, joint_ids=self._all_joint_ids_i32, env_ids=env_ids_i32
        )
        self.asset.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=fric_static,
            joint_dynamic_friction_coeff=fric_dyn,
            joint_viscous_friction_coeff=fric_visc,
            joint_ids=self._all_joint_ids_i32,
            env_ids=env_ids_i32,
        )
        # ⚠ 캐시도 갱신 — 클래스 docstring 의 raw 마찰 항목 참고 (안 하면 foot 마찰 DR 이 사라진다).
        self._t(self.asset.data.default_joint_armature)[ids] = armature
        self._t(self.asset.data.default_joint_friction_coeff)[ids] = fric_static
        self._t(self.asset.data.default_joint_viscous_friction_coeff)[ids] = fric_visc

        # ---- 게인 ----
        from isaaclab.actuators import ImplicitActuator  # 지역 import — 모듈 상단 주석 참고

        for aname, actuator in self.asset.actuators.items():
            cols = self._act_cols[aname]
            f = factor[:, cols]
            kp = self._def_kp[aname][ids] * f
            kd = self._def_kd[aname][ids] * f
            actuator.stiffness[ids] = kp
            actuator.damping[ids] = kd
            if isinstance(actuator, ImplicitActuator):
                # implicit 은 PhysX 쪽이 진짜 게인이다.
                self.asset.write_joint_stiffness_to_sim_index(
                    stiffness=kp, joint_ids=cols.to(torch.int32), env_ids=env_ids_i32
                )
                self.asset.write_joint_damping_to_sim_index(
                    damping=kd, joint_ids=cols.to(torch.int32), env_ids=env_ids_i32
                )
