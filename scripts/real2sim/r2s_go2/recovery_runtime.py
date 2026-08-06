# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 GUI recovery 모드 — 넘어진 상태에서 기립하는 fall-recovery 정책의 순수 python 런타임.

`policy_runtime.py`와 같은 역할이되 대상 정책이 다르다: `Go2Recovery-FlipVel-v0`
(`logs/rsl_rl/go2_recovery_flip_vel`)로 학습한 **plain PPO ActorCritic**이다.
RMA/estimator/history가 없어 입력이 obs 한 벌뿐이고, 명령(command)도 없다 — 목표는
언제나 "default 자세로 일어서라" 하나로 고정이라 슬라이더가 필요 없다.

관절 순서 변환·쿼터니언·default 자세는 `policy_runtime`에서 **import 해서 재사용**한다
(Go2 공통 상수라 중복 정의하면 drift 위험). torch 의존은 :class:`RecoveryModel` 내부로
지연 import 하는 것도 동일 — 반드시 **fork 이후**(publisher 프로세스)에서만 인스턴스화할 것.

계약 (`go2_recovery_env.py`, `go2_recovery_env_cfg.py` 기준):
    model(obs(B,42)) -> action(B,12)
    obs = root_ang_vel_b × 0.25 (3) + projected_gravity_b (3)
        + (joint_pos - default) × 1.0 (12) + joint_vel × 0.05 (12)
        + previous_actions (12)
    clipped = clip(action, ±100)                       # cfg.action_clip
    scaled  = clipped;  scaled[hip] ×= 0.5             # cfg.hip_scale_reduction
    target  = 0.25 × scaled + default_joint_pos        # cfg.action_scale (articulation 순서)
    previous_actions(다음 tick obs) = **clipped** — hip 스케일 전, target 아님
        (`go2_recovery_env.py:215` `self._actions = clipped`, obs는 이 값을 그대로 싣는다)

**tracking 정책과 obs 차원이 42로 같지만 레이아웃은 전혀 다르다.** tracking은 각속도가 없고
command 3-dim이 있으며 joint_vel이 raw다. 파일을 바꿔 끼우는 실수를 shape 검사로는 못 잡으므로
:meth:`RecoveryModel._warmup`이 인자 개수(recovery=1, tracking=2)로 구분해 명시적으로 막는다.

⚠ **PD 게인이 다르다.** recovery 학습 액추에이터는 stiffness 25 / **damping 1.0**
(`go2_recovery_env_cfg.py:135`)인 반면 tracking 및 GUI 기본값은 damping 0.5
(`motions.DEFAULT_KD`)다. 그래서 recovery 모드는 :data:`RECOVERY_KD`로 발행한다.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motions  # noqa: E402
import policy_runtime  # noqa: E402

# Go2 공통 상수 — policy_runtime 이 단일 source of truth (재정의 금지).
NUM_JOINTS: int = policy_runtime.NUM_JOINTS
ART_ORDER: list[str] = policy_runtime.ART_ORDER
DEFAULT_JOINT_POS_ART: list[float] = policy_runtime.DEFAULT_JOINT_POS_ART
dds_to_art = policy_runtime.dds_to_art
art_to_dds = policy_runtime.art_to_dds
projected_gravity_b = policy_runtime.projected_gravity_b

RECOVERY_OBS_DIM: int = 42  # 3 + 3 + 12 + 12 + 12

# go2_recovery_env_cfg.py 값을 그대로 복사(2026-07-31). system python은 isaaclab import 불가라
# 런타임 참조가 안 되므로 하드코딩한다 — ⚠ 학습 cfg가 바뀌면 여기도 갱신할 것.
ANG_VEL_SCALE: float = 0.25  # cfg.ang_vel_scale
DOF_POS_SCALE: float = 1.0  # cfg.dof_pos_scale
DOF_VEL_SCALE: float = 0.05  # cfg.dof_vel_scale
ACTION_SCALE: float = 0.25  # cfg.action_scale
ACTION_CLIP: float = 100.0  # cfg.action_clip (soft joint limit이 최종 clamp라 사실상 무제한)
HIP_ACTION_SCALE: float = 0.5  # cfg.hip_scale_reduction=True 일 때 env가 곱하는 값 (env.py 하드코딩)

# 학습 액추에이터 PD (cfg.robot actuators "base_legs"). GUI 기본값(damping 0.5)과 달라 명시한다.
RECOVERY_KP: float = 25.0
RECOVERY_KD: float = 1.0

# hip 관절의 articulation 인덱스. env가 `"hip" in name`으로 뽑으므로(`go2_recovery_env.py:73-77`)
# 같은 방식으로 유도한다 — 인덱스 수작업 나열 금지.
HIP_IDS_ART: list[int] = [i for i, n in enumerate(ART_ORDER) if "hip" in n]


def build_obs(
    ang_vel_b: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    joint_pos_art: list[float],
    joint_vel_art: list[float],
    prev_action: list[float],
) -> list[float]:
    """recovery 정책 obs(42-dim, articulation 순서 joint 항)를 조립한다.

    순서: ang_vel×0.25(3) + projected_gravity_b(3) + (joint_pos-default)×1.0(12)
        + joint_vel×0.05(12) + previous_actions(12).

    tracking 정책과 달리 각속도를 **입력으로 쓴다** — recovery는 estimator가 없어 자세 변화율을
    직접 봐야 하고, IMU 자이로는 실기에서 그대로 측정 가능하다.

    Args:
        ang_vel_b: body frame 각속도 [rad/s] (IMU 자이로 그대로).
        quat_wxyz: IMU 쿼터니언 (w,x,y,z).
        joint_pos_art: articulation 순서 관절각 [rad].
        joint_vel_art: articulation 순서 관절 각속도 [rad/s].
        prev_action: 직전 tick의 clip 후 action(12). 최초 진입 시 0으로 채운다.
    """
    grav = projected_gravity_b(quat_wxyz)
    pos_rel = [(joint_pos_art[i] - DEFAULT_JOINT_POS_ART[i]) * DOF_POS_SCALE for i in range(NUM_JOINTS)]
    obs = [
        *(v * ANG_VEL_SCALE for v in ang_vel_b),
        *grav,
        *pos_rel,
        *(v * DOF_VEL_SCALE for v in joint_vel_art),
        *prev_action,
    ]
    assert len(obs) == RECOVERY_OBS_DIM, f"recovery obs 차원 불일치: {len(obs)} != {RECOVERY_OBS_DIM}"
    return obs


def clip_action(action: list[float]) -> list[float]:
    """raw action을 ±ACTION_CLIP으로 자른다. **이 값이 다음 tick obs의 previous_actions**가 된다."""
    return [max(-ACTION_CLIP, min(ACTION_CLIP, a)) for a in action]


def action_to_target_art(clipped: list[float]) -> list[float]:
    """clip 된 action(12) -> articulation 순서 목표 관절각. hip만 0.5× 후 action_scale + default.

    :func:`clip_action` 결과를 받는다 — clip을 여기서 또 하지 않는 이유는 obs에 실리는 값이
    "clip 후 / hip 스케일 전"이라 호출부가 그 중간값을 반드시 보관해야 하기 때문이다.
    """
    scaled = list(clipped)
    for i in HIP_IDS_ART:
        scaled[i] *= HIP_ACTION_SCALE
    return [ACTION_SCALE * scaled[i] + DEFAULT_JOINT_POS_ART[i] for i in range(NUM_JOINTS)]


# ---------------------------------------------------------------------------
# previous_actions 버퍼 — publisher 프로세스 로컬 상태 (policy_runtime.ProprioHistory와 동일 취급).
# ---------------------------------------------------------------------------


class PrevActionBuffer:
    """recovery obs의 previous_actions 슬롯(12).

    학습 env는 `_reset_idx`에서 action 버퍼를 0으로 두고 시작하므로(`go2_recovery_env.py:100`),
    모드 진입(=reset 대응)마다 :meth:`clear`로 0에서 다시 시작해야 학습 시작 조건과 일치한다.
    """

    def __init__(self) -> None:
        self._prev: list[float] = [0.0] * NUM_JOINTS

    def clear(self) -> None:
        """모드 재진입 시 stale action이 obs에 실리는 것을 막는다(학습 reset과 동일하게 0)."""
        self._prev = [0.0] * NUM_JOINTS

    def get(self) -> list[float]:
        return list(self._prev)

    def commit(self, clipped: list[float]) -> None:
        """이번 tick의 clip 된 action을 다음 tick obs용으로 저장."""
        self._prev = list(clipped)


# ---------------------------------------------------------------------------
# torch.jit 모델 래퍼 — **fork 이후에만 인스턴스화할 것** (모듈 docstring 참고).
# ---------------------------------------------------------------------------


class RecoveryModel:
    """recovery_policy.pt(torch.jit.script) 래퍼. forward(obs) -> action."""

    def __init__(self, path: str, device: str = "cuda:0") -> None:
        import torch  # 지연 import — CUDA-after-fork 회피 (policy_runtime.PolicyModel과 동일)

        self._torch = torch
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self._path = path
        self.model = torch.jit.load(path, map_location=device).eval()
        self._warmup()

    def _warmup(self) -> None:
        """JIT/cudnn 워밍업 + 파일 종류 가드.

        tracking deployable jit는 `forward(proprio, history)`로 인자가 2개라 여기서 TypeError로
        걸린다. 두 정책 모두 obs 42-dim이라 **shape 검사로는 구분되지 않으므로** 이 가드가
        잘못된 파일을 지정했을 때의 유일한 방어선이다.
        """
        torch = self._torch
        with torch.inference_mode():
            dummy = torch.zeros(1, RECOVERY_OBS_DIM, device=self.device)
            try:
                out = self.model(dummy)
            except (RuntimeError, TypeError) as e:
                raise RuntimeError(
                    f"recovery jit 이 obs {RECOVERY_OBS_DIM}-dim 단일 입력을 받지 않는다: {self._path}\n"
                    f"  tracking 용 deployable_policy.pt(입력 2개: proprio, history)를 잘못 지정했을 수 있다.\n"
                    f"  recovery 용은 export_recovery_go2.py 가 만드는 recovery_policy.pt 다.\n"
                    f"  원본 오류: {e}"
                ) from e
            if tuple(out.shape) != (1, NUM_JOINTS):
                raise RuntimeError(f"recovery jit 출력 shape 이상: {tuple(out.shape)} != (1, {NUM_JOINTS})")

    def infer(self, obs: list[float]) -> list[float]:
        torch = self._torch
        with torch.inference_mode():
            x = torch.tensor([obs], dtype=torch.float32, device=self.device)
            action = self.model(x)
        return action[0].tolist()


if __name__ == "__main__":
    # 자체 검증 (torch 불필요 — obs 조립/action 변환만). policy_runtime.py 와 같은 스타일.

    # hip 인덱스: articulation 순서에서 앞 4개가 hip (FL/FR/RL/RR)
    assert HIP_IDS_ART == [0, 1, 2, 3], HIP_IDS_ART
    assert all("hip" in ART_ORDER[i] for i in HIP_IDS_ART)

    # obs 차원 + 슬롯 배치
    prev = [float(i + 1) for i in range(NUM_JOINTS)]
    obs = build_obs((0.4, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), list(DEFAULT_JOINT_POS_ART), [2.0] * 12, prev)
    assert len(obs) == RECOVERY_OBS_DIM
    assert abs(obs[0] - 0.4 * ANG_VEL_SCALE) < 1e-9, obs[0]  # ang_vel 스케일
    assert all(abs(obs[3 + i] - (0.0, 0.0, -1.0)[i]) < 1e-9 for i in range(3)), obs[3:6]  # identity quat 중력
    assert obs[6:18] == [0.0] * 12, obs[6:18]  # default 자세면 joint_pos_error = 0
    assert all(abs(v - 2.0 * DOF_VEL_SCALE) < 1e-9 for v in obs[18:30]), obs[18:30]  # joint_vel 스케일
    assert obs[30:42] == prev, obs[30:42]  # previous_actions 는 스케일 없이 그대로

    # action -> target: hip 만 0.5×
    act = [1.0] * NUM_JOINTS
    clipped = clip_action(act)
    assert clipped == act
    tgt = action_to_target_art(clipped)
    for i in range(NUM_JOINTS):
        expect_scale = HIP_ACTION_SCALE if i in HIP_IDS_ART else 1.0
        expect = ACTION_SCALE * expect_scale + DEFAULT_JOINT_POS_ART[i]
        assert abs(tgt[i] - expect) < 1e-9, f"{ART_ORDER[i]}: {tgt[i]} != {expect}"

    # clip 경계
    assert clip_action([1e9] + [0.0] * 11)[0] == ACTION_CLIP

    # prev-action 버퍼: 진입 시 0, commit 후 그 값, clear 로 복귀
    buf = PrevActionBuffer()
    assert buf.get() == [0.0] * NUM_JOINTS
    buf.commit(clipped)
    assert buf.get() == clipped
    buf.clear()
    assert buf.get() == [0.0] * NUM_JOINTS

    # PD 게인이 GUI 기본값과 다르다는 사실 자체를 고정 (둘이 같아지면 이 파일의 전제가 깨진 것)
    assert RECOVERY_KP == motions.DEFAULT_KP, "kp 는 tracking 과 같아야 한다"
    assert RECOVERY_KD != motions.DEFAULT_KD, "recovery 는 damping 1.0 으로 학습됐다"

    print("recovery_runtime self-test OK")
