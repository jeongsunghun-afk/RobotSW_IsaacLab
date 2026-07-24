# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 GUI policy mode — deployable RMA+estimator 정책의 순수 python 런타임.

`motions.py`와 같은 원칙(순수 함수/상수, ros/rclpy 의존 없음)을 따르되, 여기엔 **articulation
순서**(학습 env `Go2-Imitation-Tracking-v0`)와 **DDS 순서**(motions.JOINT_NAMES, CONTRACT §2) 간
변환, proprio(45-dim) 조립, torch.jit deployable 모델 래퍼가 들어간다.

torch 의존은 :class:`PolicyModel` 내부로 지연 import 한다 — 이 모듈 자체(및 상수/remap 함수)는
gui_controller.py의 **UI 프로세스**에서도 import 되므로(모드 상수·kp/kd 등은 fork 전에 필요할 수
있음), 모듈 최상단에서 torch를 끌어오면 fork 전에 부모 프로세스가 CUDA 컨텍스트를 건드릴 위험이
있다(CUDA-after-fork 문제). `PolicyModel`은 반드시 **fork 이후**(publisher 프로세스)에서만
인스턴스화할 것.

계약 (배포 가능 정책, team-lead 확정 2026-07-24):
    model(proprio(B,45), history(B,10,45)) -> action(B,12)
    proprio = root_ang_vel_b(3) + projected_gravity_b(3) + lin_vel_cmd(2) + yaw_vel_cmd(1)
            + (joint_pos-default)(12) + joint_vel(12) + actions(12, **항상 0.0** — 학습 내내
              self.actions가 __init__ 1회 zero-fill 후 갱신되지 않은 dead channel이라 실제
              last-action을 넣으면 학습된 적 없는 랜덤 가중치에 신호를 넣는 셈이 되어 OOD)
    target = clip(action, -4, 4) * 0.25 + default_joint_pos  (articulation 순서)

관절 순서는 두 벌이다:
    - articulation 순서(학습 env `go2_imitation_tracking_env.py`, USD 파싱 순서라 코드 상수로
      존재하지 않음 — 15개 이상 학습 로그의 `[Go2ImitationTrackingEnv] IsaacLab joint 순서:`
      출력이 전부 일치해 확보): FL/FR/RL/RR × hip→thigh→calf.
    - DDS(Unitree per-leg) 순서(motions.JOINT_NAMES, CONTRACT §2): FR/FL/RR/RL × hip,thigh,calf.
둘 사이 변환은 **이름으로** permutation을 구축한다(수작업 인덱스 나열 금지 — 오타 위험).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motions  # noqa: E402

NUM_JOINTS: int = 12
POLICY_DIM: int = 45
HISTORY_LEN: int = 10
ACTION_SCALE: float = 0.25
ACTION_CLIP: float = 4.0

# ---------------------------------------------------------------------------
# 관절 순서 + remap (이름 기반 — motions.py 스타일)
# ---------------------------------------------------------------------------

# articulation 순서(학습 env). go2_imitation_tracking_env_cfg.py의 UNITREE_GO2_CFG(USD)가
# 바뀌면 재검증 필요 — 검증 방법: 학습 stdout의 "[Go2ImitationTrackingEnv] IsaacLab joint 순서:" 라인.
ART_ORDER: list[str] = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

DDS_ORDER: list[str] = motions.JOINT_NAMES  # Unitree per-leg 순서 (CONTRACT §2)

# vec_art[j] = vec_dds[_ART_FROM_DDS[j]]  (DDS 순서 -> articulation 순서로 읽을 인덱스)
_ART_FROM_DDS: list[int] = [DDS_ORDER.index(name) for name in ART_ORDER]
# vec_dds[k] = vec_art[_DDS_FROM_ART[k]]  (articulation 순서 -> DDS 순서로 읽을 인덱스)
_DDS_FROM_ART: list[int] = [ART_ORDER.index(name) for name in DDS_ORDER]


def dds_to_art(vec_dds: list[float]) -> list[float]:
    """DDS(Unitree per-leg) 순서 12-벡터를 articulation(FL/FR/RL/RR 그룹) 순서로 재배열."""
    return [vec_dds[_ART_FROM_DDS[j]] for j in range(NUM_JOINTS)]


def art_to_dds(vec_art: list[float]) -> list[float]:
    """articulation 순서 12-벡터를 DDS(Unitree per-leg) 순서로 재배열."""
    return [vec_art[_DDS_FROM_ART[k]] for k in range(NUM_JOINTS)]


# 학습 env의 default_joint_pos(articulation 순서) — motions.DEFAULT_POSE(DDS 순서, UNITREE_GO2_CFG
# 정규식 기본값과 수치 일치 확인됨)를 remap해서 얻는다. 별도 상수로 중복 정의하면 drift 위험이라
# 단일 source of truth(motions.DEFAULT_POSE)에서 파생시킨다.
DEFAULT_JOINT_POS_ART: list[float] = dds_to_art(motions.DEFAULT_POSE)

# ---------------------------------------------------------------------------
# command 슬라이더 범위 — go2_imitation_tracking_env_cfg.py:143-148 값을 그대로 복사(2026-07-24
# 시점). system python(gui_controller.py가 요구하는 인터프리터)은 isaaclab import 불가라 런타임
# 참조가 안 되므로 하드코딩한다 — ⚠ 학습 cfg가 이 범위를 바꾸면 여기도 갱신할 것.
# ---------------------------------------------------------------------------
LIN_VEL_X_RANGE: tuple[float, float] = (0.0, 4.0)  # [m/s]
LIN_VEL_Y_RANGE: tuple[float, float] = (0.0, 0.0)  # vy 항상 0 — 슬라이더 없음
YAW_VEL_RANGE: tuple[float, float] = (-1.5, 1.5)  # [rad/s]


# ---------------------------------------------------------------------------
# 쿼터니언 / projected_gravity_b — isaaclab.utils.math.quat_apply_inverse(xyzw)와 동일 공식을
# 순수 python으로 재구현(system python은 isaaclab/torch 의존 불가). DDS IMU quat은 wxyz
# (CONTRACT §3, sim_bridge.py `msg.imu_state.quaternion = [quat_w,quat_x,quat_y,quat_z]`)이고
# IsaacLab 6.0의 `root_quat_w`/quat_apply_inverse는 xyzw이므로 반드시 재배열해야 한다
# (r2s_go2/CLAUDE.md "get_lowstate() IMU quat 변환 주의" 절의 `[[3,0,1,2]]`와 정반대 방향).
# ---------------------------------------------------------------------------


def quat_wxyz_to_xyzw(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """(w,x,y,z) -> (x,y,z,w)."""
    w, x, y, z = quat_wxyz
    return (x, y, z, w)


def _quat_apply_inverse_xyzw(
    quat_xyzw: tuple[float, float, float, float], vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    """isaaclab.utils.math.quat_apply_inverse와 동일 공식(xyzw 컨벤션), 순수 python 재구현."""
    x, y, z, w = quat_xyzw
    vx, vy, vz = vec
    # t = 2 * cross(xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v - w*t + cross(xyz, t)
    cx = y * tz - z * ty
    cy = z * tx - x * tz
    cz = x * ty - y * tx
    return (vx - w * tx + cx, vy - w * ty + cy, vz - w * tz + cz)


def projected_gravity_b(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """실측 IMU quat(w,x,y,z)에서 projected_gravity_b(단위벡터, body frame)를 계산.

    `isaaclab` `ArticulationData.projected_gravity_b`("Projection of the gravity direction on
    base frame", `base_articulation_data.py:793`)와 동일 정의 — 9.81 스케일 아닌 단위벡터.
    """
    return _quat_apply_inverse_xyzw(quat_wxyz_to_xyzw(quat_wxyz), (0.0, 0.0, -1.0))


# ---------------------------------------------------------------------------
# proprio 조립 + action -> target
# ---------------------------------------------------------------------------


def build_proprio(
    gyro_xyz: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    lin_vel_cmd_xy: tuple[float, float],
    yaw_vel_cmd: float,
    joint_pos_art: list[float],
    joint_vel_art: list[float],
) -> list[float]:
    """policy proprio(45-dim, articulation 순서 joint 항)를 조립한다.

    순서: root_ang_vel_b(3) + projected_gravity_b(3) + lin_vel_cmd(2) + yaw_vel_cmd(1)
        + (joint_pos-default)(12) + joint_vel(12) + actions(12, **항상 0.0**, dead channel).

    `_apply_obs_dr`(env.py)는 domain_rand 학습 시에만 additive noise를 더하는 순수 훈련용
    로직이라(9.81 스케일 등 어떤 곱셈도 없음) 배포 시엔 원시 SI 단위 값을 그대로 사용한다 —
    정규화는 jit 내부 `actor_obs_normalizer`가 처리하므로 여기서 추가 스케일링 불필요.
    """
    grav = projected_gravity_b(quat_wxyz)
    pos_rel = [joint_pos_art[i] - DEFAULT_JOINT_POS_ART[i] for i in range(NUM_JOINTS)]
    proprio = [
        *gyro_xyz,
        *grav,
        *lin_vel_cmd_xy,
        yaw_vel_cmd,
        *pos_rel,
        *joint_vel_art,
        *([0.0] * NUM_JOINTS),  # actions — team-lead 확정(2026-07-24): 학습 내내 0, 반드시 0 고정
    ]
    assert len(proprio) == POLICY_DIM, f"proprio 차원 불일치: {len(proprio)} != {POLICY_DIM}"
    return proprio


def action_to_target_art(action: list[float]) -> list[float]:
    """정책 action(12) -> articulation 순서 목표 관절각. clip(±4) 후 action_scale(0.25) + default."""
    clipped = [max(-ACTION_CLIP, min(ACTION_CLIP, a)) for a in action]
    return [ACTION_SCALE * clipped[i] + DEFAULT_JOINT_POS_ART[i] for i in range(NUM_JOINTS)]


# ---------------------------------------------------------------------------
# proprio history 링버퍼 — publisher 프로세스 로컬 상태(공유메모리에 안 올림, §3 결정).
# ---------------------------------------------------------------------------


class ProprioHistory:
    """policy 모드의 10×45 proprio 링버퍼.

    시간순 유지(oldest-first, index -1이 최신) — `go2_imitation_tracking_env.py:337-340`의
    `torch.cat([hist[:, 1:], new.unsqueeze(1)])`(oldest 하나 버리고 newest를 끝에 append)와
    동일 순서. 최초 진입 시(reset)엔 학습 시 reset 동작과 동일하게 10칸 전부를 첫 proprio로 채운다.
    """

    def __init__(self) -> None:
        self._buf: list[list[float]] | None = None

    def clear(self) -> None:
        """다음 push()가 새 seed로 10칸을 다시 채우도록 무효화(모드 재진입 시 stale history 방지)."""
        self._buf = None

    def push(self, proprio: list[float]) -> list[list[float]]:
        if self._buf is None:
            self._buf = [list(proprio) for _ in range(HISTORY_LEN)]
        else:
            self._buf = self._buf[1:] + [list(proprio)]
        return self._buf


# ---------------------------------------------------------------------------
# torch.jit deployable 모델 래퍼 — **fork 이후에만 인스턴스화할 것** (모듈 docstring 참고).
# ---------------------------------------------------------------------------


class PolicyModel:
    """deployable_policy.pt(torch.jit.script) 래퍼. forward(proprio, history) -> action."""

    def __init__(self, path: str, device: str = "cuda:0") -> None:
        import torch  # 지연 import — CUDA-after-fork 회피(모듈 최상단 import 금지, docstring 참고)

        self._torch = torch
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = torch.jit.load(path, map_location=device).eval()
        self._warmup()

    def _warmup(self) -> None:
        """JIT/cudnn 워밍업 — 라이브 50Hz tick 첫 호출이 워밍업 비용을 지지 않게 미리 1회 실행."""
        torch = self._torch
        with torch.inference_mode():
            dummy_p = torch.zeros(1, POLICY_DIM, device=self.device)
            dummy_h = torch.zeros(1, HISTORY_LEN, POLICY_DIM, device=self.device)
            self.model(dummy_p, dummy_h)

    def infer(self, proprio: list[float], history: list[list[float]]) -> list[float]:
        torch = self._torch
        with torch.inference_mode():
            p = torch.tensor([proprio], dtype=torch.float32, device=self.device)
            h = torch.tensor([history], dtype=torch.float32, device=self.device)
            action = self.model(p, h)
        return action[0].tolist()


if __name__ == "__main__":
    # 자체 검증 (torch 불필요 — remap/proprio/history만). r2s_udp.py 스타일의 self-test.
    assert sorted(ART_ORDER) == sorted(DDS_ORDER), "articulation/DDS 관절 이름 집합 불일치"

    # round-trip: dds -> art -> dds 가 항등이어야 함
    probe = [float(i) for i in range(NUM_JOINTS)]
    rt = art_to_dds(dds_to_art(probe))
    assert rt == probe, f"round-trip 실패: {rt} != {probe}"

    # default pos remap이 UNITREE_GO2_CFG 정규식 기대값과 일치하는지(손으로 유도한 참값과 대조)
    expected = {
        "FL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }
    for name, val in expected.items():
        got = DEFAULT_JOINT_POS_ART[ART_ORDER.index(name)]
        assert abs(got - val) < 1e-9, f"{name}: {got} != {val}"

    # projected_gravity_b(identity quat) == (0,0,-1)
    g = projected_gravity_b((1.0, 0.0, 0.0, 0.0))
    assert all(abs(g[i] - (0.0, 0.0, -1.0)[i]) < 1e-9 for i in range(3)), g

    # proprio 차원 + actions 슬롯이 정확히 0
    proprio = build_proprio((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.0), 0.2, [0.0] * 12, [0.0] * 12)
    assert len(proprio) == POLICY_DIM
    assert proprio[33:45] == [0.0] * 12

    # history: reset은 10칸 동일, push는 oldest-first 유지
    hist = ProprioHistory()
    h0 = hist.push(proprio)
    assert len(h0) == HISTORY_LEN and all(row == proprio for row in h0)
    proprio2 = list(proprio)
    proprio2[0] = 99.0
    h1 = hist.push(proprio2)
    assert h1[-1] == proprio2 and h1[0] == proprio  # 최신이 끝, 최초 proprio가 아직 앞쪽에 남음

    # action -> target 클립 확인
    tgt = action_to_target_art([100.0] + [0.0] * 11)  # 클립되어 4.0으로 saturate
    assert abs(tgt[0] - (ACTION_SCALE * ACTION_CLIP + DEFAULT_JOINT_POS_ART[0])) < 1e-9

    print("policy_runtime self-test OK")
