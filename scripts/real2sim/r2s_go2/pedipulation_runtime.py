# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 GUI pedipulation 모드 — 다리 하나를 조작기로 쓰는 정책의 순수 python 런타임.

`policy_runtime.py`(속도명령 locomotion) · `recovery_runtime.py`(기립)와 같은 역할이되 대상
정책이 다르다: `Go2-Pedipulation-v0` (`logs/rsl_rl/go2_pedipulation`). 세 다리로 균형을 잡고
남은 한 다리의 발을 명령받은 위치로 옮겨 유지한다.

**이 런타임이 앞의 둘과 근본적으로 다른 점 — 상태를 들고 간다.**
조작 다리 목표가 적분형이라(``man_target = _manip_joint_target + delta``) 매 tick 의 출력이
직전 tick 의 출력에 의존한다. 무상태 함수로 포팅하면 조용히 틀린 각도가 나간다. 그래서 모든
지속 상태를 :class:`PedipulationState` 하나에 모으고, 정책 시작마다 :meth:`reset` 을 부른다.

계약 (`go2_pedipulation_env.py` / `_env_cfg.py` / 채택본 체크포인트 기준)::

    model(obs(B,83)) -> action(B,28)

    obs(83) = projected_gravity_b(3)
            + (joint_pos - default_joint_pos)(12)
            + joint_vel(12)
            + prev_actions(28)
            + foot_pos_b(12)          # 발 4개 × xyz, base frame
            + leg_role(4)             # 1=지지, 0=조작
            + foot_err_b(12)          # (목표 - 현재), 지지 다리 슬롯은 0

★ **actor 는 history 를 보지 않는다 — `agent.yaml` 을 믿지 말 것.** 그 파일에는
``obs_groups.policy = [policy, history]`` 라고 적혀 있지만 채택본의 실제 가중치는
``actor.mlp.0.weight`` 가 **(512, 83)** 이다. history(830)·priv(30)를 다 보는 것은 critic
쪽(943)뿐이다. env 가 obs dict 에 ``history`` 를 담기는 하므로 yaml 만 읽으면 반대로
결론 내기 쉽다 — 그래서 `export_pedipulation_go2.py` 가 체크포인트에서 actor 입력 차원을
읽어 판정하고, 이 런타임은 그 결과(83)에 맞춘다. **가중치가 진실이고 yaml 은 참고다.**
history 를 쓰는 런을 나중에 배포하게 되면 export 로그의 ``actor_uses_history`` 가 True 로
나오고, 그때 이 파일에 링버퍼를 되살려야 한다.

    action(28) = a_loc(12) + a_man(12) + a_stiffness(4, **미사용** — 자리만 유지)
    clipped = clip(action, ±4.0)                    # agent.yaml `clip_actions`
    a_loc[hip] ×= 0.5                               # env.yaml `hip_scale_reduction`
    loc_target = 0.25 × a_loc + default_joint_pos   # 지지 다리: 절대 목표
    delta      = clamp(0.25 × a_man, ±0.15)         # cfg.manip_delta_clip
    man_target = _manip_joint_target + delta        # 조작 다리: **적분형**
    target     = role×loc_target + (1-role)×man_target
    target     = clamp(target, joint_pos_limits)
    _manip_joint_target = target                    # 12개 전부. 필터 전 값을 남긴다
    prev_actions(다음 tick obs) = **clipped** — hip 스케일·action_scale 이전 값

⚠ **hip 축소는 `a_loc` 에만 걸린다.** `a_man` 은 적분형이라 같은 계수를 곱하면 "범위 축소"가
아니라 "hip 이동 속도 감쇠"가 된다. 그래서 :func:`policy_runtime.action_to_target_art` 를
재사용할 수 없다 — 그쪽은 12-dim 단일 절대 매핑이고 hip 을 전 슬롯에 곱한다.

⚠ **`nominal_foot_pos_b` 는 상수가 아니다.** env 는 정착한 4족 자세에서 측정해 latch 하고
(`_try_capture_nominal_foot_pos`), 그 값은 실행마다 1~3 cm 씩 다르다. default 관절각의 FK 로
계산한 값과도 17~33 mm 어긋난다. 따라서 이 런타임도 **정책 시작 시점의 실측 관절각**으로
FK 를 돌려 latch 한다 (:meth:`PedipulationState.try_latch_nominal`). 명령은 이 값 기준의
offset 이므로, 상수를 박으면 모든 명령이 그만큼 어긋난다.

PD 게인은 학습 액추에이터(stiffness 25 / damping 0.5)가 `motions.DEFAULT_KP/KD` 와 같아
`recovery_runtime.RECOVERY_KD` 같은 별도 상수가 필요 없다.

관절 순서·쿼터니언·default 자세는 `policy_runtime` 에서 **import 해서 재사용**한다
(Go2 공통 상수라 중복 정의하면 drift 위험). torch 의존은 :class:`PedipulationModel` 내부로
지연 import 하는 것도 동일 — 반드시 **fork 이후**(publisher 프로세스)에서만 인스턴스화할 것.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import policy_runtime  # noqa: E402

# ── Go2 공통 상수 (policy_runtime 재사용) ────────────────────────────────────
NUM_JOINTS: int = policy_runtime.NUM_JOINTS
ART_ORDER: list[str] = policy_runtime.ART_ORDER
DEFAULT_JOINT_POS_ART: list[float] = policy_runtime.DEFAULT_JOINT_POS_ART
dds_to_art = policy_runtime.dds_to_art
art_to_dds = policy_runtime.art_to_dds
projected_gravity_b = policy_runtime.projected_gravity_b

# ── 정책 계약 ────────────────────────────────────────────────────────────────
NUM_LEGS: int = 4
PEDI_OBS_DIM: int = 83  # 3 + 12 + 12 + 28 + 12 + 4 + 12
PEDI_ACTION_DIM: int = 28  # a_loc(12) + a_man(12) + a_stiffness(4)
# jit 이 실제로 받는 차원. env cfg 의 `history_len=10` 은 critic 쪽 관측이라 여기 없다.
PEDI_INPUT_DIM: int = PEDI_OBS_DIM  # 83

ACTION_SCALE: float = 0.25  # cfg.action_scale
ACTION_CLIP: float = 4.0  # agent.yaml clip_actions (RslRlVecEnvWrapper 가 적용)
HIP_ACTION_SCALE: float = 0.5  # cfg.hip_scale_reduction=True 일 때 env 하드코딩 값
MANIP_DELTA_CLIP: float = 0.15  # cfg.manip_delta_clip [rad/step]

# 다리 인덱스 — leg_role / foot_pos_b 가 쓰는 순서 (env `_foot_body_ids` 와 동일)
LEG_NAMES: list[str] = ["FL", "FR", "RL", "RR"]

_HIP_IDS_ART: list[int] = [i for i, n in enumerate(ART_ORDER) if "_hip_" in n]
# articulation 순서는 type-major(hip×4, thigh×4, calf×4)라 관절 i 의 다리는 i % 4 가 아니라
# 이름으로 뽑는다. env `_leg_of_joint` 와 같은 값이어야 한다.
_LEG_OF_JOINT: list[int] = [LEG_NAMES.index(n.split("_")[0]) for n in ART_ORDER]

# 관절 한계 [rad] — 스폰되는 Go2 자산의 `robot.data.joint_pos_limits` 를 그대로 옮긴 값
# (articulation 순서). 적분형 목표가 한계 밖으로 누적되는 것을 막는 clamp 에 쓴다.
# ⚠ 자릿수를 줄이지 말 것. calf 상한을 -0.8378 로 반올림했더니 clamp 가 걸리는 자세에서
#   env 대비 4e-5 rad 오차가 났다 — 정확히 그 반올림 폭이다.
JOINT_POS_LIMITS_LOWER: list[float] = [
    -1.047198, -1.047198, -1.047198, -1.047198,  # hip
    -1.570796, -1.570796, -0.523599, -0.523599,  # thigh (front / rear 가 다르다)
    -2.722714, -2.722714, -2.722714, -2.722714,  # calf
]  # fmt: skip
JOINT_POS_LIMITS_UPPER: list[float] = [
    1.047198, 1.047198, 1.047198, 1.047198,
    3.490659, 3.490659, 4.537899, 4.537899,
    -0.837758, -0.837758, -0.837758, -0.837758,
]  # fmt: skip

# ── 명령 박스 (cfg CommandCfg 의 학습 최대 범위) ─────────────────────────────
# nominal 발 위치 대비 offset [m]. GUI 슬라이더는 이 범위를 넘지 못하게 한다 —
# 넘으면 학습 분포 밖이고, 기구학적으로 도달 못 하는 목표는 정책을 발산시킨다.
CMD_BOX_X: tuple[float, float] = (-0.20, 0.20)
CMD_BOX_Y: tuple[float, float] = (-0.14, 0.14)
CMD_BOX_Z: tuple[float, float] = (0.0, 0.26)

# ── 순기구학 ────────────────────────────────────────────────────────────────
# Go2 다리 지오메트리. 전 관절 0 자세에서 링크 원점을 base frame 으로 재 얻은 값이라
# 스펙 시트가 아니라 **실제로 스폰되는 자산**에서 나온 수치다. 검증: 무작위 64 자세 ×
# 4 다리에서 sim 의 `_compute_foot_pos_b()` 와 최대 오차 0.0004 mm.
HIP_OFFSET_X: float = 0.1934  # base → hip, 전후 [m]
HIP_OFFSET_Y: float = 0.0465  # base → hip, 좌우 [m]
THIGH_OFFSET_Y: float = 0.0955  # hip → thigh 축, 좌우 [m]
L_THIGH: float = 0.213  # thigh 링크 길이 [m]
L_CALF: float = 0.213  # calf 링크 길이 [m]

_SIGN_X: tuple[float, ...] = (1.0, 1.0, -1.0, -1.0)  # FL, FR = 앞(+x)
_SIGN_Y: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0)  # FL, RL = 왼쪽(+y)


def foot_pos_b(q_art: list[float]) -> list[tuple[float, float, float]]:
    """관절각으로부터 발 4개의 위치를 base frame 으로 계산한다 [m].

    sim 은 `body_pos_w` 를 직접 읽지만 실기에는 그런 값이 없어 관절각에서 FK 로 만든다.

    Args:
        q_art: 관절각 12개, articulation 순서 [rad].

    Returns:
        발 4개의 ``(x, y, z)``. 순서는 :data:`LEG_NAMES` (FL, FR, RL, RR).
    """
    out: list[tuple[float, float, float]] = []
    for leg in range(NUM_LEGS):
        q_hip, q_thigh, q_calf = q_art[leg], q_art[4 + leg], q_art[8 + leg]
        d = _SIGN_Y[leg] * THIGH_OFFSET_Y
        # 다리 평면(x-z) 내 2링크 체인 — thigh/calf 는 +y 축 회전
        x = -L_THIGH * math.sin(q_thigh) - L_CALF * math.sin(q_thigh + q_calf)
        z_leg = -L_THIGH * math.cos(q_thigh) - L_CALF * math.cos(q_thigh + q_calf)
        # abduction — +x 축 회전
        y = d * math.cos(q_hip) - z_leg * math.sin(q_hip)
        z = d * math.sin(q_hip) + z_leg * math.cos(q_hip)
        out.append((_SIGN_X[leg] * HIP_OFFSET_X + x, _SIGN_Y[leg] * HIP_OFFSET_Y + y, z))
    return out


def leg_role_from_manip(manip_leg: int) -> list[float]:
    """조작 다리 인덱스를 ``leg_role`` 벡터로 바꾼다.

    ⚠ **극성 주의**: 1 = 지지, 0 = 조작이다. 뒤집으면 세 다리를 들려 한다.

    Args:
        manip_leg: 조작할 다리 인덱스 (:data:`LEG_NAMES` 기준 0~3).

    Returns:
        길이 4 의 ``leg_role``. 조작 다리 슬롯만 0.0 이고 나머지는 1.0.
    """
    if not 0 <= manip_leg < NUM_LEGS:
        raise ValueError(f"manip_leg 는 0~{NUM_LEGS - 1} 여야 한다: {manip_leg}")
    role = [1.0] * NUM_LEGS
    role[manip_leg] = 0.0
    return role


def clip_action(action: list[float]) -> list[float]:
    """정책 출력을 학습과 같은 범위로 자른다 (``RslRlVecEnvWrapper(clip_actions=4.0)``)."""
    if len(action) != PEDI_ACTION_DIM:
        raise ValueError(f"action 은 {PEDI_ACTION_DIM}-dim 이어야 한다: {len(action)}")
    return [min(max(a, -ACTION_CLIP), ACTION_CLIP) for a in action]


class PedipulationState:
    """정책 tick 사이에 살아남아야 하는 상태 전부.

    적분형 조작 목표 때문에 이 런타임은 무상태 함수가 될 수 없다. 정책을 켤 때마다
    :meth:`reset` 을 불러 직전 세션의 적분값이 새 세션으로 새지 않게 한다.
    """

    def __init__(self) -> None:
        self.prev_actions: list[float] = [0.0] * PEDI_ACTION_DIM
        self.manip_joint_target: list[float] = list(DEFAULT_JOINT_POS_ART)
        self.nominal_foot_pos_b: list[tuple[float, float, float]] | None = None

    def reset(self, q_art: list[float] | None = None) -> None:
        """정책 시작 시점으로 되돌린다.

        Args:
            q_art: 현재 실측 관절각. 주면 적분 기준을 현재 자세로 잡아 첫 tick 에 목표가
                튀지 않는다. 없으면 default 자세를 기준으로 한다.
        """
        self.prev_actions = [0.0] * PEDI_ACTION_DIM
        self.manip_joint_target = list(q_art) if q_art is not None else list(DEFAULT_JOINT_POS_ART)
        self.nominal_foot_pos_b = None

    def try_latch_nominal(self, q_art: list[float]) -> bool:
        """4족으로 서 있는 현재 자세에서 nominal 발 위치를 못 박는다.

        env 의 `_try_capture_nominal_foot_pos` 와 같은 판정을 쓴다 — 발 깊이가 0.20~0.40 m
        범위일 때만 유효한 기립 자세로 보고 latch 한다. 넘어져 있거나 다리를 든 상태에서
        latch 하면 이후 모든 명령이 그 자세를 기준으로 어긋난다.

        Returns:
            이번 호출로 latch 됐거나 이미 latch 돼 있으면 True.
        """
        if self.nominal_foot_pos_b is not None:
            return True
        feet = foot_pos_b(q_art)
        if all(0.20 < -f[2] < 0.40 for f in feet):
            self.nominal_foot_pos_b = feet
            return True
        return False


def foot_target_b(
    nominal_foot_pos_b: list[tuple[float, float, float]],
    manip_leg: int,
    offset: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """명령 offset 을 절대 목표 위치로 바꾼다 (base frame).

    조작 다리만 ``nominal + offset`` 이고 지지 다리 슬롯은 nominal 그대로다 — 어차피
    :func:`build_obs` 에서 지지 슬롯 오차는 0 으로 마스킹된다.
    """
    out = list(nominal_foot_pos_b)
    nx, ny, nz = nominal_foot_pos_b[manip_leg]
    out[manip_leg] = (nx + offset[0], ny + offset[1], nz + offset[2])
    return out


def build_obs(
    quat_wxyz: tuple[float, float, float, float],
    q_art: list[float],
    dq_art: list[float],
    leg_role: list[float],
    target_b: list[tuple[float, float, float]],
    state: PedipulationState,
) -> list[float]:
    """정책 입력 ``obs(83)`` 을 만든다.

    Args:
        quat_wxyz: base 자세 쿼터니언 (w, x, y, z).
        q_art: 관절 위치 12개, articulation 순서 [rad].
        dq_art: 관절 속도 12개, articulation 순서 [rad/s].
        leg_role: 다리 역할 4개 (1=지지, 0=조작). :func:`leg_role_from_manip` 참조.
        target_b: 발 4개의 목표 위치, base frame [m]. :func:`foot_target_b` 참조.
        state: 지속 상태. ``prev_actions`` 를 읽는다.

    Returns:
        정책 관측 83개.
    """
    if len(q_art) != NUM_JOINTS or len(dq_art) != NUM_JOINTS:
        raise ValueError(f"관절 벡터는 {NUM_JOINTS}-dim 이어야 한다")
    if len(leg_role) != NUM_LEGS:
        raise ValueError(f"leg_role 은 {NUM_LEGS}-dim 이어야 한다")
    if sum(1 for r in leg_role if r == 0.0) != 1:
        # 학습은 조작 다리 1개만 겪었다. 0 이 여러 개면 세 다리를 들려 한다.
        raise ValueError(f"leg_role 에는 0(조작 다리)이 정확히 하나여야 한다: {leg_role}")

    feet = foot_pos_b(q_art)
    obs: list[float] = []
    obs.extend(projected_gravity_b(quat_wxyz))  # 3
    obs.extend(q - d for q, d in zip(q_art, DEFAULT_JOINT_POS_ART))  # 12
    obs.extend(dq_art)  # 12
    obs.extend(state.prev_actions)  # 28
    for f in feet:  # 12
        obs.extend(f)
    obs.extend(leg_role)  # 4
    for leg in range(NUM_LEGS):  # 12 — 지지 다리 슬롯은 0
        manip = 1.0 - leg_role[leg]
        obs.extend((t - c) * manip for t, c in zip(target_b[leg], feet[leg]))

    if len(obs) != PEDI_OBS_DIM:
        raise AssertionError(f"policy obs 가 {PEDI_OBS_DIM} 이 아니다: {len(obs)}")
    return obs


def action_to_target_art(clipped: list[float], leg_role: list[float], state: PedipulationState) -> list[float]:
    """clip 된 정책 출력을 관절 목표각으로 바꾸고 적분 상태를 갱신한다.

    :func:`clip_action` 결과를 받는다 — obs 에 실리는 ``prev_actions`` 가 clip 된 값이라
    호출부가 그 값을 그대로 보관해야 하기 때문이다(여기서 또 clip 하면 어느 쪽이 obs 로
    가는지 흐려진다).

    Args:
        clipped: :func:`clip_action` 을 거친 28-dim 액션.
        leg_role: 다리 역할 4개 (1=지지, 0=조작).
        state: 적분 상태. ``manip_joint_target`` 을 읽고 새 목표로 갱신한다.

    Returns:
        관절 목표각 12개, articulation 순서 [rad].
    """
    a_loc = list(clipped[0:12])
    a_man = clipped[12:24]
    # clipped[24:28] 은 a_stiffness — cfg.use_stiffness_action=False 라 쓰지 않는다.

    # hip 축소는 a_loc 에만. a_man 은 적분형이라 곱하면 이동 속도가 감쇠된다.
    for j in _HIP_IDS_ART:
        a_loc[j] *= HIP_ACTION_SCALE

    target: list[float] = []
    for j in range(NUM_JOINTS):
        role = leg_role[_LEG_OF_JOINT[j]]
        loc_target = a_loc[j] * ACTION_SCALE + DEFAULT_JOINT_POS_ART[j]
        delta = min(max(a_man[j] * ACTION_SCALE, -MANIP_DELTA_CLIP), MANIP_DELTA_CLIP)
        man_target = state.manip_joint_target[j] + delta
        t = role * loc_target + (1.0 - role) * man_target
        target.append(min(max(t, JOINT_POS_LIMITS_LOWER[j]), JOINT_POS_LIMITS_UPPER[j]))

    # 다음 스텝의 적분 기준 — 지지 슬롯도 함께 이어받아 stance→manip 전환에서 튀지 않는다.
    state.manip_joint_target = list(target)
    return target


class PedipulationModel:
    """torch.jit 로 export 된 pedipulation 정책. **fork 이후에만** 만들 것."""

    def __init__(self, path: str, device: str = "cuda:0") -> None:
        import torch

        self._torch = torch
        if not os.path.isfile(path):
            raise FileNotFoundError(f"pedipulation 정책이 없다: {path}")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self._warmup()

    def _warmup(self) -> None:
        """차원으로 다른 정책을 걸러낸다.

        tracking(42→12) · recovery(42→12) 와 달리 이 정책은 83 입력 / 28 출력이라 shape 만으로
        구분된다 — 파일을 바꿔 끼우면 여기서 죽는다. 실기로 나가는 각도라 조용히 통과시키지
        않는다(그 둘은 obs 가 42 로 같아 warmup 인자 개수로 갈라야 했다).
        """
        torch = self._torch
        with torch.inference_mode():
            out = self.model(torch.zeros(1, PEDI_INPUT_DIM, device=self.device))
        if out.shape[-1] != PEDI_ACTION_DIM:
            raise RuntimeError(f"정책 출력이 {PEDI_ACTION_DIM}-dim 이 아니다: {tuple(out.shape)}")

    def infer(self, obs: list[float]) -> list[float]:
        """관측 83개로 액션 28개를 낸다 (clip 전 raw)."""
        if len(obs) != PEDI_OBS_DIM:
            raise ValueError(f"입력이 {PEDI_OBS_DIM}-dim 이어야 한다: {len(obs)}")
        torch = self._torch
        with torch.inference_mode():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.model(x).squeeze(0).tolist()
