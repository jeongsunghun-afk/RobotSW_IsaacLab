# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 GUI policy mode — deployable RMA+estimator 정책의 순수 python 런타임.

`motions.py`와 같은 원칙(순수 함수/상수, ros/rclpy 의존 없음)을 따르되, 여기엔 **articulation
순서**(학습 env `Go2-Imitation-Tracking-v0`)와 **DDS 순서**(motions.JOINT_NAMES, CONTRACT §2) 간
변환, proprio(42-dim) 조립, torch.jit deployable 모델 래퍼가 들어간다.

torch 의존은 :class:`PolicyModel` 내부로 지연 import 한다 — 이 모듈 자체(및 상수/remap 함수)는
gui_controller.py의 **UI 프로세스**에서도 import 되므로(모드 상수·kp/kd 등은 fork 전에 필요할 수
있음), 모듈 최상단에서 torch를 끌어오면 fork 전에 부모 프로세스가 CUDA 컨텍스트를 건드릴 위험이
있다(CUDA-after-fork 문제). `PolicyModel`은 반드시 **fork 이후**(publisher 프로세스)에서만
인스턴스화할 것.

계약 (배포 가능 정책, 2026-07-30 갱신 — obs 45→42):
    model(proprio(B,42), history(B,10,42)) -> action(B,12)
    proprio = projected_gravity_b(3) + lin_vel_cmd(2) + yaw_vel_cmd(1)
            + (joint_pos-default)(12) + joint_vel(12) + actions(12, **항상 0.0** — 학습 내내
              self.actions가 __init__ 1회 zero-fill 후 갱신되지 않은 dead channel이라 실제
              last-action을 넣으면 학습된 적 없는 랜덤 가중치에 신호를 넣는 셈이 되어 OOD)
    clipped = clip(action, ±ACTION_CLIP);  clipped[hip] *= HIP_ACTION_SCALE
    target  = ACTION_SCALE * clipped + default_joint_pos     (articulation 순서)

**IMU 자이로(root_ang_vel_b)는 proprio에 넣지 않는다.** 학습 env에서 각속도를 priv_explicit로
옮기면서 policy obs에서 제외했기 때문이다(estimator가 추정 대상을 직접 보면 추정 구조가
무의미해지므로). 실기에서 측정 가능한 값이지만 정책 입력으로 쓰지 않는 것이 학습과 일치한다.
→ 45-dim으로 학습된 구(舊) 체크포인트에는 이 런타임을 쓸 수 없다(차원 assert가 잡는다).

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
POLICY_DIM: int = 42  # 2026-07-30: root_ang_vel_b(3)가 priv_explicit로 이동해 45→42
HISTORY_LEN: int = 10
ACTION_SCALE: float = 0.25  # cfg.action_scale

# ★ 아래 둘은 **학습 cfg 와 반드시 짝**이다. 값이 어긋나면 조용히 틀린 각도가 나간다 — hip 이
#   2배로 벌어지거나, 학습에서 쓰던 액션 범위가 배포에서 잘린다.
#
#   그래서 상수로 박아 두지 않고 **정책이 학습된 run 의 cfg 에서 읽는다**
#   (:func:`configure_from_policy`, `PolicyModel` 로드 시 자동 호출). 확정 전에는 ``None`` 이고,
#   그 상태로 :func:`action_to_target_art` 를 부르면 **에러로 막는다.** 실기로 나가는 각도라
#   "적당한 기본값"으로 넘겨짚지 않는다 — 예전 기본값 (10.0, 0.5) 은 정작 기본
#   `R2S_DEPLOYABLE_POLICY` 인 obs42 런(clip 4.0 · hip 축소 없음)과 어긋나 있었다.
#
#   환경변수를 주면 그것이 이긴다 (cfg 를 못 찾는 정책이나 수동 실험용):
#       R2S_POLICY_ACTION_CLIP=4.0 R2S_POLICY_HIP_SCALE=1.0 r2s_gui
ACTION_CLIP: float | None = None  # = 학습 agent.yaml 의 `clip_actions`
HIP_ACTION_SCALE: float | None = None  # = 학습 env.yaml 의 `hip_scale_reduction`

# `hip_scale_reduction` 은 bool 이고 배율 0.5 는 env 쪽에 하드코딩돼 있다
# (`go2_imitation_tracking_env.py:243`). cfg 가 float 로 바뀌면 여기도 같이 고쳐야 한다.
HIP_SCALE_WHEN_ENABLED: float = 0.5

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

# hip(abduction) 관절의 articulation 인덱스 — 이름으로 뽑는다(수작업 나열 금지).
_HIP_IDS_ART: list[int] = [i for i, name in enumerate(ART_ORDER) if "_hip_" in name]

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
    quat_wxyz: tuple[float, float, float, float],
    lin_vel_cmd_xy: tuple[float, float],
    yaw_vel_cmd: float,
    joint_pos_art: list[float],
    joint_vel_art: list[float],
) -> list[float]:
    """policy proprio(42-dim, articulation 순서 joint 항)를 조립한다.

    순서: projected_gravity_b(3) + lin_vel_cmd(2) + yaw_vel_cmd(1)
        + (joint_pos-default)(12) + joint_vel(12) + actions(12, **항상 0.0**, dead channel).

    자이로(각속도)는 인자로 받지 않는다 — 학습 env가 각속도를 priv_explicit로 옮겨
    policy obs에서 제외했으므로 정책은 이 신호를 입력으로 본 적이 없다.

    `_apply_obs_dr`(env.py)는 domain_rand 학습 시에만 additive noise를 더하는 순수 훈련용
    로직이라(9.81 스케일 등 어떤 곱셈도 없음) 배포 시엔 원시 SI 단위 값을 그대로 사용한다 —
    정규화는 jit 내부 `actor_obs_normalizer`가 처리하므로 여기서 추가 스케일링 불필요.
    """
    grav = projected_gravity_b(quat_wxyz)
    pos_rel = [joint_pos_art[i] - DEFAULT_JOINT_POS_ART[i] for i in range(NUM_JOINTS)]
    proprio = [
        *grav,
        *lin_vel_cmd_xy,
        yaw_vel_cmd,
        *pos_rel,
        *joint_vel_art,
        *([0.0] * NUM_JOINTS),  # actions — team-lead 확정(2026-07-24): 학습 내내 0, 반드시 0 고정
    ]
    assert len(proprio) == POLICY_DIM, f"proprio 차원 불일치: {len(proprio)} != {POLICY_DIM}"
    return proprio


def _run_dir_of_policy(policy_path: str) -> str:
    """deployable jit 경로에서 학습 run 디렉터리를 유도한다.

    `run_export_policy.sh` 가 `<run>/exported/deployable_policy.pt` 로 내보내므로 두 단계 위다.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(policy_path)))


def _load_run_yaml(path: str) -> dict:
    """IsaacLab 이 덤프한 params yaml 을 읽는다.

    `env.yaml` 은 `!!python/tuple` 태그를 쓰므로 순수 SafeLoader 로는 못 읽는다. 임의 객체를
    만들지 않도록 tuple 생성자만 더한 **하위 로더**를 쓴다(전역 SafeLoader 를 건드리지 않는다).
    """
    import yaml

    class _TupleLoader(yaml.SafeLoader):
        pass

    _TupleLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda ldr, node: tuple(ldr.construct_sequence(node)),
    )
    with open(path) as fh:
        loaded = yaml.load(fh, Loader=_TupleLoader)
    return loaded if isinstance(loaded, dict) else {}


def resolve_run_scaling(policy_path: str) -> tuple[float | None, float | None, list[str]]:
    """정책이 학습된 run 의 cfg 에서 ``(action_clip, hip_action_scale)`` 을 읽는다.

    Args:
        policy_path: deployable jit 경로. 이 파일 기준으로 `../params/{agent,env}.yaml` 을 본다.

    Returns:
        ``(action_clip, hip_action_scale, notes)``. 읽지 못한 값은 ``None`` 이고, ``notes`` 는
        무엇을 어디서 정했는지 사람이 읽을 설명이다.

    hip 은 **키가 있고 참일 때만** 축소한다. 키가 아예 없으면 그 run 은 이 기능이 생기기 전에
    학습된 것이므로 축소 없음(1.0)이 맞다 — 없는 것을 "모름"으로 보고 기본값을 씌우면 구 정책의
    hip 이 절반으로 눌린다.
    """
    run_dir = _run_dir_of_policy(policy_path)
    notes: list[str] = []
    clip: float | None = None
    hip: float | None = None

    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.isfile(agent_yaml):
        val = _load_run_yaml(agent_yaml).get("clip_actions")
        if val is None:
            notes.append(f"agent.yaml 에 clip_actions 없음: {agent_yaml}")
        else:
            clip = float(val)
            notes.append(f"action_clip={clip} ← {agent_yaml}")
    else:
        notes.append(f"agent.yaml 없음: {agent_yaml}")

    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if os.path.isfile(env_yaml):
        cfg = _load_run_yaml(env_yaml)
        if "hip_scale_reduction" in cfg:
            enabled = bool(cfg["hip_scale_reduction"])
            hip = HIP_SCALE_WHEN_ENABLED if enabled else 1.0
            notes.append(f"hip_scale={hip} (hip_scale_reduction={enabled}) ← {env_yaml}")
        else:
            hip = 1.0
            notes.append(f"hip_scale=1.0 (env.yaml 에 hip_scale_reduction 키 없음 = 기능 이전 런): {env_yaml}")
    else:
        notes.append(f"env.yaml 없음: {env_yaml}")

    return clip, hip, notes


def configure_from_policy(policy_path: str) -> None:
    """``ACTION_CLIP`` / ``HIP_ACTION_SCALE`` 을 학습 cfg 로 확정한다.

    환경변수(`R2S_POLICY_ACTION_CLIP` / `R2S_POLICY_HIP_SCALE`)가 있으면 그것이 이긴다.
    둘 다 정하지 못하면 여기서 막는다 — 어떤 값이 맞는지 모르는 채로 실기에 각도를 내보내는
    것보다 뜨지 않는 편이 낫다.

    Args:
        policy_path: 로드할 deployable jit 경로.

    Raises:
        RuntimeError: 학습 cfg 를 찾지 못했고 환경변수로도 주지 않았을 때.
    """
    global ACTION_CLIP, HIP_ACTION_SCALE

    clip, hip, notes = resolve_run_scaling(policy_path)

    env_clip = os.environ.get("R2S_POLICY_ACTION_CLIP")
    if env_clip is not None:
        clip = float(env_clip)
        notes.append(f"action_clip={clip} ← R2S_POLICY_ACTION_CLIP (cfg 보다 우선)")
    env_hip = os.environ.get("R2S_POLICY_HIP_SCALE")
    if env_hip is not None:
        hip = float(env_hip)
        notes.append(f"hip_scale={hip} ← R2S_POLICY_HIP_SCALE (cfg 보다 우선)")

    missing = [n for n, v in (("action_clip", clip), ("hip_scale", hip)) if v is None]
    if missing:
        raise RuntimeError(
            f"정책 스케일링을 확정하지 못했다: {', '.join(missing)}\n"
            f"  정책: {policy_path}\n"
            + "".join(f"  - {n}\n" for n in notes)
            + "  학습 run 의 params/ 가 없는 정책이면 환경변수로 명시할 것:\n"
            "      R2S_POLICY_ACTION_CLIP=<학습 clip_actions> "
            "R2S_POLICY_HIP_SCALE=<hip 축소 시 0.5, 아니면 1.0> r2s_gui"
        )

    ACTION_CLIP, HIP_ACTION_SCALE = clip, hip
    print(
        f"[policy_runtime] action_clip={ACTION_CLIP} hip_scale={HIP_ACTION_SCALE} action_scale={ACTION_SCALE}",
        flush=True,
    )
    for note in notes:
        print(f"[policy_runtime]   {note}", flush=True)


def configure_manual(action_clip: float, hip_action_scale: float) -> None:
    """cfg 없이 값을 직접 지정한다 (self-test·단위 테스트용)."""
    global ACTION_CLIP, HIP_ACTION_SCALE
    ACTION_CLIP, HIP_ACTION_SCALE = float(action_clip), float(hip_action_scale)


def action_to_target_art(action: list[float]) -> list[float]:
    """정책 action(12) -> articulation 순서 목표 관절각.

    학습 env(`go2_imitation_tracking_env._pre_physics_step`)와 같은 순서로 적용한다:
    ``clip(±ACTION_CLIP)`` → hip 만 ``HIP_ACTION_SCALE`` 배 → ``ACTION_SCALE`` 배 → default 더하기.

    **이 변환은 정책 action 에만 쓴다.** GUI 의 Pose / Joint Step / Sine Sweep 은 절대 관절각을
    그대로 발행하므로 이 함수를 타지 않는다(스케일·클립 모두 미적용).

    Args:
        action: 정책이 낸 raw action 12개.

    Returns:
        articulation 순서 목표 관절각 12개 [rad].

    Raises:
        RuntimeError: :func:`configure_from_policy` 로 스케일링을 확정하기 전에 불렀을 때.
    """
    if ACTION_CLIP is None or HIP_ACTION_SCALE is None:
        raise RuntimeError(
            "정책 스케일링이 확정되지 않았다 — configure_from_policy(<jit 경로>) 를 먼저 부를 것."
            " (PolicyModel 을 쓰면 자동으로 불린다.)"
        )
    clip, hip_scale = ACTION_CLIP, HIP_ACTION_SCALE
    clipped = [max(-clip, min(clip, a)) for a in action]
    for i in _HIP_IDS_ART:
        clipped[i] *= hip_scale
    return [ACTION_SCALE * clipped[i] + DEFAULT_JOINT_POS_ART[i] for i in range(NUM_JOINTS)]


# ---------------------------------------------------------------------------
# proprio history 링버퍼 — publisher 프로세스 로컬 상태(공유메모리에 안 올림, §3 결정).
# ---------------------------------------------------------------------------


class ProprioHistory:
    """policy 모드의 10×42 proprio 링버퍼.

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
        self._path = path
        # ⚠ jit 로드보다 **먼저** 확정한다. 여기서 막히면 정책이 아예 안 뜨는데, 그것이 틀린
        #    스케일로 실기에 각도를 내보내는 것보다 낫다. 실제 사용 값도 stdout 에 찍힌다.
        configure_from_policy(path)
        self.model = torch.jit.load(path, map_location=device).eval()
        self._warmup()

    def _warmup(self) -> None:
        """JIT/cudnn 워밍업 — 라이브 50Hz tick 첫 호출이 워밍업 비용을 지지 않게 미리 1회 실행.

        차원 가드도 겸한다. obs 레이아웃이 다른 런(예: obs 45 / priv_explicit 3)의 jit 을
        실수로 지정하면 여기서 shape mismatch 로 죽는데, raw 에러는 원인을 알기 어려워 다시 던진다.
        """
        torch = self._torch
        with torch.inference_mode():
            dummy_p = torch.zeros(1, POLICY_DIM, device=self.device)
            dummy_h = torch.zeros(1, HISTORY_LEN, POLICY_DIM, device=self.device)
            try:
                self.model(dummy_p, dummy_h)
            except RuntimeError as e:
                raise RuntimeError(
                    f"deployable jit 이 proprio {POLICY_DIM}-dim 을 받지 않는다: {self._path}\n"
                    f"  이 런타임은 obs {POLICY_DIM} 레이아웃 전용이다. 다른 레이아웃(예: 45)으로 학습한 런의"
                    f" 체크포인트라면 export 한 run 의 params/env.yaml `observation_space` 를 확인할 것.\n"
                    f"  원본 오류: {e}"
                ) from e

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
    proprio = build_proprio((1.0, 0.0, 0.0, 0.0), (0.5, 0.0), 0.2, [0.0] * 12, [0.0] * 12)
    assert len(proprio) == POLICY_DIM
    assert proprio[30:42] == [0.0] * 12

    # history: reset은 10칸 동일, push는 oldest-first 유지
    hist = ProprioHistory()
    h0 = hist.push(proprio)
    assert len(h0) == HISTORY_LEN and all(row == proprio for row in h0)
    proprio2 = list(proprio)
    proprio2[0] = 99.0
    h1 = hist.push(proprio2)
    assert h1[-1] == proprio2 and h1[0] == proprio  # 최신이 끝, 최초 proprio가 아직 앞쪽에 남음

    # 스케일링을 확정하기 전에는 각도를 못 내보내야 한다 (실기 안전장치).
    assert ACTION_CLIP is None and HIP_ACTION_SCALE is None, "import 직후엔 미확정이어야 한다"
    try:
        action_to_target_art([0.0] * NUM_JOINTS)
        raise AssertionError("미확정 상태인데 action_to_target_art 가 통과했다")
    except RuntimeError:
        pass

    # action -> target: 클립 + hip 축소가 학습 env 와 같은 순서로 걸리는지 확인.
    # ART_ORDER[0] 은 hip 이므로 hip 스케일까지 곱해져야 한다.
    configure_manual(action_clip=10.0, hip_action_scale=HIP_SCALE_WHEN_ENABLED)
    assert 0 in _HIP_IDS_ART and len(_HIP_IDS_ART) == 4, _HIP_IDS_ART
    tgt = action_to_target_art([100.0] + [0.0] * 11)  # 클립되어 ACTION_CLIP 으로 saturate
    assert abs(tgt[0] - (ACTION_SCALE * 0.5 * 10.0 + DEFAULT_JOINT_POS_ART[0])) < 1e-9
    # thigh(=ART_ORDER[4], hip 아님)에는 hip 스케일이 걸리면 안 된다.
    a_thigh = [0.0] * NUM_JOINTS
    a_thigh[4] = 1.0
    tgt = action_to_target_art(a_thigh)
    assert abs(tgt[4] - (ACTION_SCALE * 1.0 + DEFAULT_JOINT_POS_ART[4])) < 1e-9
    # 클립 미만은 그대로 통과
    a_small = [0.0] * NUM_JOINTS
    a_small[4] = 0.5
    assert abs(action_to_target_art(a_small)[4] - (ACTION_SCALE * 0.5 + DEFAULT_JOINT_POS_ART[4])) < 1e-9

    # hip 축소를 끄면(=구 정책) hip 에도 배율이 걸리지 않아야 한다.
    configure_manual(action_clip=4.0, hip_action_scale=1.0)
    tgt = action_to_target_art([1.0] + [0.0] * 11)
    assert abs(tgt[0] - (ACTION_SCALE * 1.0 + DEFAULT_JOINT_POS_ART[0])) < 1e-9
    # 클립도 학습값을 따라가야 한다 (10.0 이 아니라 4.0 에서 saturate).
    tgt = action_to_target_art([100.0] + [0.0] * 11)
    assert abs(tgt[0] - (ACTION_SCALE * 4.0 + DEFAULT_JOINT_POS_ART[0])) < 1e-9

    # cfg 해석: 실제 학습 run 들로 확인한다. hip 키가 있는 런과 없는 런을 모두 본다.
    # scripts/real2sim/r2s_go2/policy_runtime.py -> repo root (4 단계 위)
    _repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    _cases = [
        # (run 이름, 기대 clip, 기대 hip_scale)
        ("2026-07-30_10-35-28_obs42_novideo", 4.0, 1.0),  # hip_scale_reduction 키가 없는 구 런
        ("2026-08-04_20-55-56_clip10_scratch", 10.0, 0.5),  # hip_scale_reduction: true
    ]
    for _run, _want_clip, _want_hip in _cases:
        _jit = os.path.join(_repo, "logs/rsl_rl/go2_imitation_tracking", _run, "exported/deployable_policy.pt")
        if not os.path.isfile(os.path.join(_run_dir_of_policy(_jit), "params", "env.yaml")):
            print(f"  (건너뜀 — 로컬에 run 없음: {_run})")
            continue
        _clip, _hip, _notes = resolve_run_scaling(_jit)
        assert _clip == _want_clip, f"{_run}: clip {_clip} != {_want_clip}\n" + "\n".join(_notes)
        assert _hip == _want_hip, f"{_run}: hip {_hip} != {_want_hip}\n" + "\n".join(_notes)
        print(f"  cfg 해석 OK: {_run} -> clip={_clip} hip_scale={_hip}")

    print("policy_runtime self-test OK")
