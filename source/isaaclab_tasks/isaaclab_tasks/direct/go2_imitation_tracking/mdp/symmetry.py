# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2-Imitation-Tracking 좌우(L/R) 미러 — rsl_rl symmetry augmentation / mirror loss 용.

왜 필요한가 (2026-09-03 실측)
-----------------------------
`cmd 3.5` 에서 thigh ROM 좌우차가 **36.5 %** 였다(대조군 7.2 %, MimicKit 16.5 %).
`|vy|`·`|yaw|` 는 셋이 비슷하므로 몸통이 흐르는 게 아니라 **다리 사용이 한쪽으로 쏠린다.**

★ 데이터셋에는 이미 `*_mirror.pkl` 이 있고 클립 균등으로 뽑히므로 **참조 분포는 좌우 균형**이다.
그런데도 정책이 쏠리는 이유: AMP 는 **0.2 s 창의 분포**를 맞출 뿐이고, expert 분포는
왼쪽 편향 창과 오른쪽 편향 창의 **합집합**이다. 한쪽으로 일관되게 쏠린 정책의 창도 그 합집합
안에 들어가므로 discriminator 가 벌할 근거가 없다. 즉 **미러 데이터는 목표 분포를 대칭으로
만들 뿐, 개별 롤아웃을 대칭으로 미는 기울기를 주지 않는다.**
leg 계열에서 같은 실험이 이미 실패했다 — 미러 클립을 추가해 데이터셋 대칭오차를 +0.024 →
+0.014 로 줄였는데도 정책 비대칭은 9.4k −0.024 → 50k **+0.646** 으로 단조 발산했다
(`project_leg_imitation_symmetry_fix`). 거기서 통한 것은 **mirror loss** 였다.

미러 맵 (참조 쌍으로 실측 검증, 오차 **정확히 0.0**)
--------------------------------------------------
`go2_run2` 와 `go2_run2_mirror` 를 직접 대조해 관절별로 판정했다::

    hip   (FL/FR/RL/RR)  swap + **부호 반전**
    thigh (FL/FR/RL/RR)  swap 만
    calf  (FL/FR/RL/RR)  swap 만

★ 관절 타입으로 일반화하지 말 것 — leg(17-DOF)에서는 뒷발·waist 도 FLIP 이었고, 그 원인이
URDF 축 표기 불일치였다. **로봇이 바뀌면 참조 쌍으로 다시 실측해야 한다.**

★ `default_joint_pos` 는 hip 이 ±0.1 로 **반대칭**이고 thigh/calf 는 좌우 동일하므로
(`unitree.py`), policy obs 의 ``joint_pos - default`` 오프셋에 같은 미러를 적용하면 정확히 맞는다.

관측 레이아웃 (`joint_pos_tan_norm=False` 기준)
---------------------------------------------
::

    policy(42)   [0:3]  projected_gravity_b   (gx, gy, gz) -> (gx, -gy, gz)
                 [3:5]  lin_vel_cmd (vx, vy)  -> (vx, -vy)
                 [5:6]  yaw_vel_cmd           -> -yaw
                 [6:18] joint_pos - default   swap + hip flip
                 [18:30] joint_vel            swap + hip flip
                 [30:42] actions              swap + hip flip
    priv_explicit(6)  [0:3] root_lin_vel_b -> (x, -y, z)
                      [3:6] root_ang_vel_b -> (-wx, wy, -wz)
    priv_latent(19)   [0:7]  스칼라 DR 파라미터 (미러 불변)
                      [7:19] encoder_bias    swap + hip flip
    history(N, H, 42) 프레임마다 policy 미러
    actions(12)       swap + hip flip

⚠️ AMP discriminator 관측은 **미러하지 않는다** — expert 배치와 짝이 맞아야 하고, 데이터셋에
이미 미러 클립이 있어 그쪽은 데이터 레벨에서 대칭이다.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

# ── 관측 블록 오프셋 (policy / history 공용) ──────────────────────────────────
_PROPRIO_DIM = 42
_PRIV_EXPLICIT_DIM = 6
_PRIV_LATENT_DIM = 19
_NUM_JOINTS = 12

_GRAV = slice(0, 3)
_LIN_CMD = slice(3, 5)
_YAW_CMD = slice(5, 6)
_JOINT_POS = slice(6, 18)
_JOINT_VEL = slice(18, 30)
_ACTIONS = slice(30, 42)

# priv_latent: 앞 7 개는 스칼라(armature/joint_friction/base_mass/foot_friction/kp/kd/action_delay),
# 뒤 12 개가 관절별 encoder_bias 다.
_PRIV_LATENT_SCALARS = 7
_ENCODER_BIAS = slice(_PRIV_LATENT_SCALARS, _PRIV_LATENT_DIM)


def _flip_side(prefix: str) -> str:
    """``FL`` <-> ``FR``, ``RL`` <-> ``RR``."""
    if prefix.endswith("L"):
        return prefix[:-1] + "R"
    if prefix.endswith("R"):
        return prefix[:-1] + "L"
    return prefix


def mirror_joint_name(name: str) -> str:
    """``FL_hip_joint`` -> ``FR_hip_joint``."""
    prefix, _, rest = name.partition("_")
    return f"{_flip_side(prefix)}_{rest}" if rest else _flip_side(prefix)


def mirror_joint_needs_flip(name: str) -> bool:
    """부호를 뒤집어야 하는 관절인가.

    Go2 는 **hip(abduction)만** 뒤집는다. 참조 쌍 `go2_run2` / `go2_run2_mirror` 로 실측 검증했고
    (swap+flip 오차 0.0, swap-only 오차 0.6~1.1 rad), thigh/calf 는 그 반대다.
    """
    return "hip" in name


def _build_cache(env) -> dict:
    """관절 swap 순열과 flip 인덱스를 런타임 이름에서 만들어 env 에 캐시한다.

    ★ 인덱스를 하드코딩하지 않는 이유: 관절 순서가 다리별 묶음(`FL_hip, FL_thigh, ...`)인지
    타입별 묶음(`hip x4, thigh x4, ...`)인지는 자산/버전에 따라 다르다. 이름으로 짜면 안전하다.
    """
    u = env.unwrapped
    cache = getattr(u, "_go2_symmetry_cache", None)
    if cache is not None:
        return cache

    if getattr(u.cfg, "joint_pos_tan_norm", False):
        raise NotImplementedError(
            "[go2 symmetry] joint_pos_tan_norm=True 는 아직 지원하지 않는다 — 관절 블록이 12 -> 72 "
            "(관절별 회전 tan-norm 6D)로 바뀌어 미러가 스칼라 부호 반전이 아니게 된다."
        )

    # ★ 인덱스 텐서를 **CPU 에 만든다.** `u.device` 로 만들면 rsl_rl 이 다른 디바이스의 텐서를
    #   넘길 때 "indices should be ... same device as the indexed tensor (cuda:0)" 로 터진다
    #   (leg symmetry 에도 있던 같은 버그 — `--device cuda:N` 학습에서 재현). 사용처에서
    #   대상 텐서의 디바이스로 옮긴다(12 원소라 비용 무시 가능).
    joint_names = list(u._robot.data.joint_names)
    if len(joint_names) != _NUM_JOINTS:
        raise AssertionError(f"[go2 symmetry] joint 수 {len(joint_names)} != {_NUM_JOINTS}: {joint_names}")

    perm = [joint_names.index(mirror_joint_name(n)) for n in joint_names]
    joint_swap = torch.tensor(perm, dtype=torch.long)
    # 순열이 아니면(이름 매칭 실패) 조용히 틀린 미러가 되므로 여기서 막는다.
    if sorted(perm) != list(range(_NUM_JOINTS)):
        raise ValueError(f"[go2 symmetry] swap 순열이 아니다: {perm} (names={joint_names})")

    flip_idx = torch.tensor([i for i, n in enumerate(joint_names) if mirror_joint_needs_flip(n)], dtype=torch.long)
    if flip_idx.numel() != 4:
        raise AssertionError(f"[go2 symmetry] flip 대상(hip)이 4 개가 아니다: {flip_idx.tolist()} ({joint_names})")

    cache = {"joint_swap": joint_swap, "flip_idx": flip_idx}
    u._go2_symmetry_cache = cache
    return cache


def _swap_joints(j: torch.Tensor, joint_swap: torch.Tensor, flip_idx: torch.Tensor) -> torch.Tensor:
    """관절 벡터를 좌우 교환하고 hip 부호를 뒤집는다. 앞쪽 차원은 임의(`[B,12]`/`[B,H,12]` 공용)."""
    # 인덱스는 CPU 캐시에 있으므로 대상 텐서 디바이스로 옮겨 쓴다(_build_cache 주석 참조).
    sw = joint_swap.to(j.device)
    fl = flip_idx.to(j.device)
    out = j[..., sw].clone()
    out[..., fl] = -out[..., fl]
    return out


def _mirror_proprio(p: torch.Tensor, joint_swap: torch.Tensor, flip_idx: torch.Tensor) -> torch.Tensor:
    """policy / history proprio(42) 미러. 마지막 차원이 proprio 여야 한다."""
    out = p.clone()
    out[..., _GRAV.start + 1] = -out[..., _GRAV.start + 1]  # projected_gravity_b 의 y
    # lin_vel_cmd 의 vy — 현재 항상 0 이지만 규약상 뒤집는다(채널이 살아나면 자동으로 맞는다)
    out[..., _LIN_CMD.start + 1] = -out[..., _LIN_CMD.start + 1]
    out[..., _YAW_CMD] = -out[..., _YAW_CMD]
    out[..., _JOINT_POS] = _swap_joints(p[..., _JOINT_POS], joint_swap, flip_idx)
    out[..., _JOINT_VEL] = _swap_joints(p[..., _JOINT_VEL], joint_swap, flip_idx)
    out[..., _ACTIONS] = _swap_joints(p[..., _ACTIONS], joint_swap, flip_idx)
    return out


def _mirror_priv_explicit(e: torch.Tensor) -> torch.Tensor:
    """root 선속도(y 반전) + 각속도(x, z 반전) — 시상면 미러의 표준 변환."""
    out = e.clone()
    out[..., 1] = -out[..., 1]  # lin_vel y
    out[..., 3] = -out[..., 3]  # ang_vel x (roll)
    out[..., 5] = -out[..., 5]  # ang_vel z (yaw)
    return out


def _mirror_priv_latent(v: torch.Tensor, joint_swap: torch.Tensor, flip_idx: torch.Tensor) -> torch.Tensor:
    """앞 7 개 스칼라는 그대로, 뒤 12 개 encoder_bias 만 관절 미러."""
    out = v.clone()
    out[..., _ENCODER_BIAS] = _swap_joints(v[..., _ENCODER_BIAS], joint_swap, flip_idx)
    return out


def compute_go2_symmetric_states(
    *, env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Go2-Imitation-Tracking 관측/행동에 좌우 미러를 덧붙인다 (num_aug = 2).

    ``[원본 B; 미러 B]`` 순서로 돌려준다 — rsl_rl 이 앞쪽 B 가 원본이라고 가정한다.

    Args:
        env: 래핑된 VecEnv. ``env.unwrapped`` 가 go2 env 를 노출한다.
        obs: ``policy(42) / priv_explicit(6) / priv_latent(19) / history(N, H, 42)`` TensorDict.
            ``None`` 이면 관측은 ``None`` 을 돌려준다.
        actions: 행동 텐서 ``(B, 12)``. ``None`` 이면 행동은 ``None`` 을 돌려준다.

    Returns:
        ``(obs_aug | None, actions_aug | None)``.
    """
    cache = _build_cache(env)
    joint_swap, flip_idx = cache["joint_swap"], cache["flip_idx"]

    if obs is not None:
        b = obs.batch_size[0]
        # 레이아웃이 바뀌면 조용히 틀린 미러가 되므로 호출마다 폭을 검사한다.
        for key, want in (
            ("policy", _PROPRIO_DIM),
            ("priv_explicit", _PRIV_EXPLICIT_DIM),
            ("priv_latent", _PRIV_LATENT_DIM),
        ):
            got = obs[key].shape[-1]
            if got != want:
                raise AssertionError(f"[go2 symmetry] {key} dim {got} != {want}")

        out = obs.repeat(2)  # [원본 B; (채울) 미러 B]
        out["policy"][b:] = _mirror_proprio(obs["policy"], joint_swap, flip_idx)
        out["priv_explicit"][b:] = _mirror_priv_explicit(obs["priv_explicit"])
        out["priv_latent"][b:] = _mirror_priv_latent(obs["priv_latent"], joint_swap, flip_idx)
        out["history"][b:] = _mirror_proprio(obs["history"], joint_swap, flip_idx)
    else:
        out = None

    if actions is not None:
        a = actions.repeat(2, 1)
        a[actions.shape[0] :] = _swap_joints(actions, joint_swap, flip_idx)
    else:
        a = None

    return out, a
