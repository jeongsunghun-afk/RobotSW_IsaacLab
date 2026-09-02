# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""명령 조건부 AMP discriminator 를 위한 env 측 조건 열 생성 (go2 / leg imitation_tracking 공용).

discriminator 는 무조건부라 "이 속도에서 어떤 걸음이어야 하는가"를 모른다. 여기서 AMP obs 끝에
조건 벡터를 붙여 준다.

.. code-block:: text

    disc 입력 = [ amp_obs (H × D) | cond (k) | valid (1) ]
    정책 샘플  cond = [ |v_cmd| / v_max ,  yaw_cmd / yaw_max ][:k]    valid = 1
    expert    cond = [ clip_mean_speed / v_max , clip_mean_yaw / yaw_max ][:k]   valid = 1
    조건 없음  cond = 0, valid = 0   (정지 명령 env, 알고리즘 측 조건 dropout)

expert 라벨을 창의 순간 속도가 아니라 **클립 평균**으로 두는 이유: 정책 쪽 조건이 명령(set-point)이라
순간 속도가 아니고, 참조 클립도 대체로 등속이라 클립 평균이 명령의 가장 가까운 대응이다.

러너는 ``env.amp_cond_dim`` 을 읽어 알고리즘에 넘기고, disc 는 그 열을 정규화·gradient penalty 에서
제외한다. ``amp_cond_mode="none"`` 이면 아무 열도 붙지 않아 기존 run 과 완전히 같다.
"""

from __future__ import annotations

import torch

AMP_COND_MODES: dict[str, int] = {"none": 0, "speed": 1, "speed_yaw": 2}


class AMPCommandConditionMixin:
    """env 에 섞어 쓰는 조건 열 헬퍼. 호스트 env 가 다음을 갖고 있어야 한다.

    ``cfg.amp_cond_mode / amp_cond_v_max / amp_cond_yaw_max / lin_vel_x_max / yaw_vel_min / yaw_vel_max``,
    ``self._motion_lib`` (``motion_mean_speeds``·``motion_mean_yaw_rates``·``motion_names``),
    ``self._lin_vel_cmd [N,2]``, ``self._yaw_vel_cmd [N]``, ``self.device``.
    정지 명령 마스크 ``self._standing_mask [N] bool`` 은 있으면 쓰고 없으면 무시한다.
    """

    amp_cond_dim: int = 0

    def _init_amp_condition(self) -> int:
        """조건 열 수를 정하고 expert 클립 라벨 표를 만든다. 반환값은 AMP obs 에 더할 열 수."""
        cfg = self.cfg  # type: ignore[attr-defined]
        mode = getattr(cfg, "amp_cond_mode", "none")
        if mode not in AMP_COND_MODES:
            raise ValueError(f"amp_cond_mode 는 {tuple(AMP_COND_MODES)} 중 하나여야 한다: {mode}")
        k = AMP_COND_MODES[mode]
        self._amp_cond_values_dim = k
        self.amp_cond_dim = k + 1 if k > 0 else 0
        if k == 0:
            return 0

        v_max = getattr(cfg, "amp_cond_v_max", None) or float(cfg.lin_vel_x_max)
        yaw_max = getattr(cfg, "amp_cond_yaw_max", None) or max(
            abs(float(cfg.yaw_vel_min)), abs(float(cfg.yaw_vel_max))
        )
        yaw_max = max(yaw_max, 1e-6)
        self._amp_cond_scale = torch.tensor([v_max, yaw_max][:k], dtype=torch.float32, device=self.device)  # type: ignore[attr-defined]

        lib = self._motion_lib  # type: ignore[attr-defined]
        speeds = lib.motion_mean_speeds.to(self.device)  # type: ignore[attr-defined]
        yaws = lib.motion_mean_yaw_rates.to(self.device)  # type: ignore[attr-defined]
        raw = torch.stack([speeds, yaws], dim=-1)[:, :k]  # [M, k]
        self._motion_cond = raw / self._amp_cond_scale  # [M, k]

        print(f"[AMPCommandCondition] mode={mode} cond_dim={self.amp_cond_dim} v_max={v_max:.2f} yaw_max={yaw_max:.2f}")
        for name, sp, yw in zip(lib.motion_names, speeds.tolist(), yaws.tolist()):
            print(f"  {name:32s} speed {sp:5.2f} m/s  yaw {yw:+5.2f} rad/s")
        return self.amp_cond_dim

    def _policy_amp_cond(self, env_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        """정책 샘플의 조건 열 ``[n, cond_dim]``. 조건 모드가 꺼져 있으면 None."""
        if self.amp_cond_dim == 0:
            return None
        lin = self._lin_vel_cmd if env_ids is None else self._lin_vel_cmd[env_ids]  # type: ignore[attr-defined]
        yaw = self._yaw_vel_cmd if env_ids is None else self._yaw_vel_cmd[env_ids]  # type: ignore[attr-defined]
        raw = torch.stack([torch.linalg.norm(lin, dim=-1), yaw], dim=-1)[:, : self._amp_cond_values_dim]
        cond = raw / self._amp_cond_scale
        valid = torch.ones(cond.shape[0], 1, dtype=torch.float32, device=cond.device)
        standing = getattr(self, "_standing_mask", None)
        if standing is not None:
            # 정지 명령 env 는 참조에 정지 클립이 없어 조건을 걸 대상이 없다 → 무조건부로 판별.
            st = standing if env_ids is None else standing[env_ids]
            cond = torch.where(st.unsqueeze(-1), torch.zeros_like(cond), cond)
            valid = torch.where(st.unsqueeze(-1), torch.zeros_like(valid), valid)
        return torch.cat([cond, valid], dim=-1)

    def _expert_amp_cond(self, motion_ids: torch.Tensor) -> torch.Tensor | None:
        """expert 샘플의 조건 열 ``[n, cond_dim]`` (클립 평균 라벨, valid=1)."""
        if self.amp_cond_dim == 0:
            return None
        cond = self._motion_cond[motion_ids.to(self._motion_cond.device)]
        valid = torch.ones(cond.shape[0], 1, dtype=torch.float32, device=cond.device)
        return torch.cat([cond, valid], dim=-1)

    @staticmethod
    def _append_amp_cond(amp_obs_flat: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """``[n, H·D]`` 뒤에 조건 열을 붙인다. cond 가 None 이면 그대로."""
        if cond is None:
            return amp_obs_flat
        return torch.cat([amp_obs_flat, cond.to(amp_obs_flat.device)], dim=-1)
