# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Parkour-Imitation hybrid environment.

Extends Go2ParkourEnv with:
  - AMP discriminator observation buffer  [N, history_length, 43] (base, ring-buffer)
  - AMP quat history buffer               [N, history_length, 4]  (for tan_norm computation)
  - Flat-env mask                         [N] bool — True when env is on flat terrain
  - AMP reward buffer                     [N] float — written by the AMP runner

AMP discriminator observation layout (49-dim per step, 490-dim flattened × 10 history):
    dof_pos(12) + dof_vel(12) + root_height(1) + root_lin_vel_b(3) +
    root_ang_vel_b(3) + foot_pos_local(12) + root_rot_tan_norm(6) = 49

root_rot_tan_norm(6): heading-relative 6D rotation feature (MimicKit compute_tar_obs 방식).
  Internal base buffer stores 43-dim; tan_norm 6D is appended at consumption time.
  Ring-buffer convention: oldest frame at index 0, newest at index -1.

The parkour policy observation dict is unchanged (keys: policy, scan, priv_explicit,
priv_latent, history).  A new key ``amp_obs`` [N, 490] is appended for the AMP runner.

Spawn distribution: parkour's default uniform terrain assignment is used unchanged.
AMP applies only to envs spawned on parkour_flat (flat_env_mask) — both reward and
discriminator gradient are masked.  See _flat_env_mask / _update_flat_env_mask.

Reward interface (for reward-worker):
    self._amp_reward_buf  [N]  — AMP runner writes per-env disc reward here each step
    self._flat_env_mask   [N]  — True = flat terrain, AMP reward applies to these envs

Network interface (for network-worker):
    amp_obs key shape: (N, cfg.amp_history_length * cfg.amp_obs_dim) = (N, 490)
"""

from __future__ import annotations

import pathlib

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from isaaclab_tasks.direct.parkour.parkour_env import Go2ParkourEnv
from isaaclab_tasks.direct.parkour.parkour_env_cfg import TERRAIN_CLASS_FLAT

from .motion_lib import Go2MotionLib
from .parkour_imitation_env_cfg import ParkourImitationEnvCfg


class Go2ParkourImitationEnv(Go2ParkourEnv):
    """Parkour + AMP imitation hybrid environment.

    Inherits all parkour mechanics from Go2ParkourEnv and adds:
    - AMP base history buffer for discriminator observations (43-dim × H steps).
    - AMP quat history buffer for root_rot_tan_norm computation (4-dim × H steps).
    - Flat-terrain mask for AMP reward/discriminator gating.
    - AMP reward buffer interface for the AMP runner.

    AMP obs per step = 49-dim:
        base 43-dim (dof_pos/vel + root_height + lin/ang_vel + foot_pos_local) +
        root_rot_tan_norm 6-dim (heading-relative 6D rotation, MimicKit 방식).
    Ring-buffer: oldest at index 0, newest at index -1.

    Spawn distribution uses parkour's default uniform terrain assignment unchanged.
    AMP reward and discriminator gradient are applied only to flat-terrain envs
    (_flat_env_mask).

    The DebugViewer is already managed by the base class; this subclass does NOT
    call ``_debug_viewer.update()`` again.
    """

    cfg: ParkourImitationEnvCfg

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, cfg: ParkourImitationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── AMP foot body indices (robot articulation, NOT contact sensor) ──────
        # _feet_ids are contact-sensor indices; we need robot articulation indices
        # for body_pos_w access in _update_amp_obs_buf.
        self._amp_foot_body_ids, _amp_foot_names = self._robot.find_bodies(".*foot")
        assert len(self._amp_foot_body_ids) == 4, (
            f"AMP: expected 4 foot bodies, got {len(self._amp_foot_body_ids)}: {_amp_foot_names}"
        )

        # ── AMP base history buffer [N, H, 43] ───────────────────────────────
        # Stores the 43-dim base AMP frame per step (ring-buffer, newest at index -1).
        # tan_norm 6D is NOT stored here; computed at consumption time from _amp_quat_buf.
        _AMP_BASE_DIM = 43  # fixed: dof_pos(12)+dof_vel(12)+root_height(1)+lin_vel(3)+ang_vel(3)+foot(12)
        self._amp_obs_buf = torch.zeros(
            self.num_envs,
            cfg.amp_history_length,
            _AMP_BASE_DIM,
            device=self.device,
        )

        # ── AMP quat history buffer [N, H, 4] (wxyz, identity-initialized) ──
        # Per-step root_quat history for heading-relative tan_norm computation.
        # Ring-buffer with same oldest/newest convention as _amp_obs_buf.
        self._amp_quat_buf = torch.zeros(
            self.num_envs,
            cfg.amp_history_length,
            4,
            device=self.device,
        )
        self._amp_quat_buf[..., 0] = 1.0  # identity quat wxyz: w=1

        # ── Per-env flat terrain mask [N] bool ───────────────────────────────
        self._flat_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── AMP reward buffer [N] — written by AMP runner each step ──────────
        self._amp_reward_buf = torch.zeros(self.num_envs, device=self.device)

        # ── Terminal AMP obs buffer [N, H*49] — captured in _reset_idx before super ──
        # Runner indexes terminal_amp_obs[dones.bool()] so must be full [N, H*49] every step.
        # Updated for reset envs in _reset_idx; cloned into extras each step in
        # _get_observations so the runner always sees a fresh [num_envs, 490] tensor.
        _amp_flat_dim = cfg.amp_history_length * cfg.amp_obs_dim  # 10 * 49 = 490
        self._terminal_amp_obs = torch.zeros(self.num_envs, _amp_flat_dim, device=self.device)

        # ── Resolve amp_motion_pkl to absolute path ───────────────────────────
        _pkl_path = pathlib.Path(cfg.amp_motion_pkl)
        if not _pkl_path.is_absolute():
            _pkl_path = pathlib.Path(__file__).parent / _pkl_path
        self._amp_motion_pkl_path = _pkl_path

        # ── Motion library (used by get_amp_observations for expert sampling) ─
        self._motion_lib = Go2MotionLib(
            motion_files=str(self._amp_motion_pkl_path),
            device=self.device,
        )

        # DOF order: motion_lib stores frames in DOF_NAMES order; IsaacLab may
        # use a different alphabetical order.  Build the permutation index once.
        robot_joint_names = list(self._robot.data.joint_names)
        try:
            self._motion_dof_indices = self._motion_lib.get_dof_index(robot_joint_names)
        except AssertionError:
            # Fallback: assume orders match 1:1
            self._motion_dof_indices = list(range(12))

        # ── Initialise flat env mask from super's terrain assignment ─────────
        self._update_flat_env_mask()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Extend parkour observations with AMP discriminator buffer.

        Calls super() which returns the standard parkour obs dict and also
        calls ``_debug_viewer.update()`` — we must NOT call it again here.

        Returns:
            dict with all parkour keys plus ``amp_obs`` key of shape [N, 490].
        """
        obs = super()._get_observations()
        self._update_amp_obs_buf()
        # Push AMP obs into extras so OnPolicyRunnerParkourAMP can read it.
        # _build_flat_amp_obs() applies tan_norm to produce [N, H, 49] then flattens to [N, 490].
        # terminal_amp_obs is a persistent [N, 490] buffer updated in _reset_idx;
        # cloned here every step so the runner can index [dones.bool()] on it.
        self.extras["amp_obs"] = self._build_flat_amp_obs(self._amp_obs_buf, self._amp_quat_buf)
        self.extras["terminal_amp_obs"] = self._terminal_amp_obs.clone()
        self._log_amp_metrics()
        return obs

    def _update_amp_obs_buf(self) -> None:
        """Compute the current 43-dim base AMP frame and push into the ring buffers.

        Base AMP obs layout (43-dim, stored in _amp_obs_buf):
            dof_pos       (12)  — joint positions (rad)
            dof_vel       (12)  — joint velocities (rad/s)
            root_height    (1)  — z position of root in world frame (m)
            root_lin_vel_b (3)  — root linear velocity in body frame (m/s)
            root_ang_vel_b (3)  — root angular velocity in body frame (rad/s)
            foot_pos_local(12)  — [FL,FR,RL,RR] × 3, expressed in root body frame (m)

        root_quat_w (4) is stored separately in _amp_quat_buf for tan_norm computation.

        Ring-buffer convention: oldest frame at index 0, newest at index -1.
        ``torch.roll(..., shifts=-1, dims=1)`` shifts left so index -1 is overwritten.
        """
        root_pos_w = self._robot.data.root_pos_w          # [N, 3]
        root_quat_w = self._robot.data.root_quat_w        # [N, 4]  wxyz convention
        root_lin_vel_b = self._robot.data.root_lin_vel_b  # [N, 3]
        root_ang_vel_b = self._robot.data.root_ang_vel_b  # [N, 3]

        # Foot positions in base-local frame
        foot_pos_w = self._robot.data.body_pos_w[:, self._amp_foot_body_ids, :]  # [N, 4, 3]
        rel_pos = foot_pos_w - root_pos_w.unsqueeze(1)                           # [N, 4, 3]
        N, K = rel_pos.shape[:2]
        local_foot_pos = quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
            rel_pos.reshape(-1, 3),
        ).view(N, K, 3)  # [N, 4, 3]

        # Build the 43-dim base AMP frame for the current step
        current_frame = torch.cat(
            [
                self._robot.data.joint_pos,          # 12  rad
                self._robot.data.joint_vel,          # 12  rad/s
                root_pos_w[:, 2:3],                  # 1   m
                root_lin_vel_b,                      # 3   m/s
                root_ang_vel_b,                      # 3   rad/s
                local_foot_pos.view(N, -1),          # 12  m
            ],
            dim=-1,
        )  # [N, 43]

        # Roll history left (discard oldest at index 0) and write newest at index -1
        self._amp_obs_buf = torch.roll(self._amp_obs_buf, shifts=-1, dims=1)
        self._amp_obs_buf[:, -1, :] = current_frame

        # Update quat history with the same roll convention
        self._amp_quat_buf = torch.roll(self._amp_quat_buf, shifts=-1, dims=1)
        self._amp_quat_buf[:, -1, :] = root_quat_w

    def _build_flat_amp_obs(
        self,
        base_buf: torch.Tensor,
        quat_buf: torch.Tensor,
    ) -> torch.Tensor:
        """Append root_rot_tan_norm(6) to base 43-dim buffer and flatten.

        Args:
            base_buf: [N, H, 43] — base AMP history (ring-buffer, newest at -1)
            quat_buf: [N, H, 4]  — root quat history (same ring-buffer convention)

        Returns:
            Tensor [N, H*49] — flattened 49-dim AMP obs for discriminator input.
        """
        N, H = base_buf.shape[:2]
        rot_tan_norm = _apply_root_rot_tan_norm(quat_buf, N, H)       # [N, H, 6]
        full_obs = torch.cat([base_buf, rot_tan_norm], dim=-1)         # [N, H, 49]
        return full_obs.flatten(start_dim=1)                           # [N, H*49]

    def _log_amp_metrics(self) -> None:
        """Write per-step AMP diagnostics to ``self.extras["log"]``.

        Called from ``_get_observations`` after ``_update_amp_obs_buf``.
        DirectRLEnv step order: ``_get_dones → _get_rewards → _reset_idx → _get_observations``.
        ``_reset_idx`` overwrites ``self.extras["log"]`` with a fresh dict, so keys written
        here always persist until the next reset batch.

        Note on timing: ``_amp_reward_buf`` is written by the AMP runner AFTER ``env.step()``
        returns (step t reads step t-1 disc reward, zeros at episode start).  This one-step
        lag is acceptable for monitoring purposes.

        Note on magnitude: ``total_reward_mean_*`` is summed from
        ``_last_reward_breakdown_per_env`` (sum of scaled per-term values, pre-clip).
        This reflects the raw accumulation before the ``torch.clip(total, min=0.)`` in
        ``_get_rewards`` — showing negative totals that the clip would suppress is more
        informative than a silently floored metric.

        Metric keys (WandB ``AMP/`` group):
            AMP/amp_reward_mean_flat    — mean disc reward on flat-terrain envs (AMP target)
            AMP/amp_reward_mean_terrain — mean disc reward on non-flat envs (integrity check: ~0)
            AMP/flat_env_fraction       — fraction of envs currently on flat terrain
            AMP/total_reward_mean_flat  — mean parkour reward on flat envs
            AMP/total_reward_mean_terrain — mean parkour reward on non-flat envs
        """
        flat_mask = self._flat_env_mask        # [N] bool
        non_flat_mask = ~flat_mask             # [N] bool

        # Guard against empty masks (small batches may be all-flat or all-non-flat)
        def _safe_masked_mean(buf: torch.Tensor, mask: torch.Tensor) -> float:
            return buf[mask].mean().item() if mask.any() else 0.0

        # Per-env parkour reward total (pre-clip, sum across all terms)
        total_per_env = self._last_reward_breakdown_per_env.sum(dim=1)  # [N]

        log = self.extras.setdefault("log", {})
        log["AMP/amp_reward_mean_flat"] = _safe_masked_mean(self._amp_reward_buf, flat_mask)
        log["AMP/amp_reward_mean_terrain"] = _safe_masked_mean(self._amp_reward_buf, non_flat_mask)
        log["AMP/flat_env_fraction"] = flat_mask.float().mean().item()
        log["AMP/total_reward_mean_flat"] = _safe_masked_mean(total_per_env, flat_mask)
        log["AMP/total_reward_mean_terrain"] = _safe_masked_mean(total_per_env, non_flat_mask)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset environment indices.

        Ordering is critical:
        0. Snapshot terminal AMP obs BEFORE super (robot state not yet reset).
        1. ``super()._reset_idx()`` — runs terrain curriculum, updates ``_env_class[env_ids]``
           and ``env_origins[env_ids]`` using parkour's default uniform assignment.
        2. Clear AMP buffer and AMP reward buffer for reset envs.
        3. Refresh flat_env_mask for reset envs (depends on fresh ``_env_class``).
        """
        # ── 0. Terminal AMP obs snapshot ────────────────────────────────────────
        # Robot data still holds the last physics step of the ending episode.
        # _terminal_amp_obs[env_ids] is updated here; the full [N, 490] buffer is
        # cloned into extras["terminal_amp_obs"] every step in _get_observations so
        # the runner can index terminal_amp_obs[dones.bool()] correctly.
        # _build_flat_amp_obs applies tan_norm using current _amp_quat_buf (pre-reset).
        if env_ids is not None and env_ids.numel() > 0:
            self._terminal_amp_obs[env_ids] = self._build_flat_amp_obs(
                self._amp_obs_buf[env_ids],
                self._amp_quat_buf[env_ids],
            )

        super()._reset_idx(env_ids)

        # Normalise env_ids (super may have expanded None → ALL_INDICES internally,
        # but we need a concrete tensor here for buffer indexing).
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Clear AMP history and reward for reset envs
        self._amp_obs_buf[env_ids] = 0.0
        # Reset quat buf to identity quaternion (w=1, x=y=z=0)
        self._amp_quat_buf[env_ids] = 0.0
        self._amp_quat_buf[env_ids, :, 0] = 1.0
        self._amp_reward_buf[env_ids] = 0.0

        # Refresh flat mask after super has updated _env_class via terrain curriculum
        self._update_flat_env_mask(env_ids)

    # ------------------------------------------------------------------
    # AMP runner interface
    # ------------------------------------------------------------------

    @property
    def amp_observation_space(self) -> gym.spaces.Box:
        """Gym space describing the flattened AMP observation vector.

        Shape: (amp_history_length * amp_obs_dim,) = (10 * 49,) = (490,).
        Matches the flatten of _build_flat_amp_obs() output [N, 10, 49] → [N, 490].
        """
        dim = self.cfg.amp_history_length * self.cfg.amp_obs_dim  # 10 * 49 = 490
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def get_amp_observations(self, num_samples: int) -> torch.Tensor:
        """Return expert AMP observation samples from the reference motion library.

        Builds a history window of H=cfg.amp_history_length consecutive frames,
        laid out oldest-first to match the live _amp_obs_buf ring-buffer
        (torch.roll shifts oldest out at index 0, newest lands at index -1):
            returned[:, :49]   = frame at t-(H-1)*step_dt  (oldest)
            returned[:, -49:]  = frame at t                (newest)

        49-dim per step layout (expert side matches live side exactly):
            dof_pos(12) + dof_vel(12) + root_height(1) + root_lin_vel_b(3) +
            root_ang_vel(3) + foot_pos_local(12) + root_rot_tan_norm(6) = 49

        root_rot_tan_norm is computed via _apply_root_rot_tan_norm() with the same
        heading-relative convention as the live _build_flat_amp_obs() path.

        Args:
            num_samples: Number of independent expert windows to sample.

        Returns:
            Tensor of shape (num_samples, amp_history_length * amp_obs_dim)
            = (num_samples, 490), dtype float32, on self.device.
        """
        H = self.cfg.amp_history_length
        motion_ids = self._motion_lib.sample_motions(num_samples)
        times = self._motion_lib.sample_times(motion_ids)

        # Build time offsets: index 0 → oldest (t-(H-1)*dt), index H-1 → newest (t).
        # arange(H-1, -1, -1) = [H-1, H-2, ..., 1, 0] → subtracted from t gives ascending time.
        offsets = torch.arange(H - 1, -1, -1, device=self.device).float() * self.step_dt  # [H]
        times_flat = (times.unsqueeze(-1) - offsets.unsqueeze(0)).clamp(min=0.0).reshape(-1)  # [N*H]
        ids_flat = motion_ids.unsqueeze(-1).expand(-1, H).reshape(-1)  # [N*H]

        rp, rq, lv, av, dp, dv, fp = self._motion_lib.calc_motion_frame(ids_flat, times_flat)

        # Re-order DOFs from motion_lib order to IsaacLab joint order (mirrors live side).
        dp = dp[:, self._motion_dof_indices]
        dv = dv[:, self._motion_dof_indices]

        # Build 43-dim base AMP frame (same layout as _update_amp_obs_buf):
        #   dof_pos(12) + dof_vel(12) + root_height(1) + root_lin_vel_b(3) +
        #   root_ang_vel(3) + foot_pos_local(12) = 43
        base_frame = torch.cat(
            [
                dp,                          # 12  rad
                dv,                          # 12  rad/s
                rp[:, 2:3],                  # 1   m  (root height)
                lv,                          # 3   m/s body-frame
                av,                          # 3   rad/s body-frame
                fp.view(fp.shape[0], -1),    # 12  m  (4 feet x 3, body-local)
            ],
            dim=-1,
        )  # [N*H, 43]

        # Reshape to [N, H, 43] and [N, H, 4] for _apply_root_rot_tan_norm
        base_buf = base_frame.view(num_samples, H, 43)       # [N, H, 43]
        quat_buf = rq.view(num_samples, H, 4)                # [N, H, 4]

        # Append tan_norm to produce [N, H, 49] then flatten to [N, H*49]
        return self._build_flat_amp_obs(base_buf, quat_buf)  # [N, 490]

    # ------------------------------------------------------------------
    # Flat-env mask
    # ------------------------------------------------------------------

    def _update_flat_env_mask(self, env_ids: torch.Tensor | None = None) -> None:
        """Update the per-env flat terrain boolean mask.

        Args:
            env_ids: If None, refresh the full mask.  Otherwise refresh only those envs.
        """
        if env_ids is None:
            self._flat_env_mask = self._env_class == TERRAIN_CLASS_FLAT
        else:
            self._flat_env_mask[env_ids] = self._env_class[env_ids] == TERRAIN_CLASS_FLAT


# ---------------------------------------------------------------------------
# Module-level helpers (ported from go2_imitation_env.py)
# ---------------------------------------------------------------------------


@torch.jit.script
def _calc_heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only quaternion inverse (heading 기준 로컬 변환). quat: [N,4] wxyz."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = -yaw * 0.5  # inverse = negative yaw
    heading_inv = torch.stack(
        [torch.cos(half_yaw), torch.zeros_like(half_yaw), torch.zeros_like(half_yaw), torch.sin(half_yaw)],
        dim=-1,
    )
    return heading_inv  # [N,4] wxyz


def _apply_root_rot_tan_norm(
    quat_buf: torch.Tensor,
    num_envs: int,
    n_hist: int,
) -> torch.Tensor:
    """Window 내 각 frame의 root_quat을 현재 frame heading-inv 기준 local로 변환 후 6D tan_norm 반환.

    MimicKit compute_tar_obs (deepmimic_env.py:733,742) 방식:
      - ref = quat_buf[:, -1, :] (window index -1 = 가장 최근 frame, ring-buffer 규약)
      - heading_inv_rot = _calc_heading_quat_inv(ref)
      - relative_quat[h] = quat_mul(heading_inv_expand, quat_buf[:, h, :])
      - tan_norm: [quat_rotate(q, [1,0,0]), quat_rotate(q, [0,0,1])]

    Note: parkour_imitation_env.py 의 ring-buffer 규약은 newest frame = index -1.
    (go2_imitation 의 shift-left 규약과 반대. newest=index 0 대신 index -1.)

    Args:
        quat_buf: [N, H, 4] wxyz — per-step root_quat history (index -1 = newest)
        num_envs: N
        n_hist:   H

    Returns:
        rot_tan_norm: [N, H, 6]
    """
    # ref heading-inv: yaw-only inverse of the newest frame (index -1 in ring-buffer)
    ref_quat = quat_buf[:, -1, :]  # [N, 4]
    heading_inv = _calc_heading_quat_inv(ref_quat)  # [N, 4]

    heading_inv_exp = heading_inv.unsqueeze(1).expand(-1, n_hist, -1).reshape(num_envs * n_hist, 4)  # [N*H, 4]
    quats_flat = quat_buf.reshape(num_envs * n_hist, 4)  # [N*H, 4]

    rel_quat = quat_mul(heading_inv_exp, quats_flat)  # [N*H, 4]

    tan_ref = torch.zeros(num_envs * n_hist, 3, dtype=quat_buf.dtype, device=quat_buf.device)
    tan_ref[:, 0] = 1.0  # [1, 0, 0]
    norm_ref = torch.zeros(num_envs * n_hist, 3, dtype=quat_buf.dtype, device=quat_buf.device)
    norm_ref[:, 2] = 1.0  # [0, 0, 1]

    tan = quat_apply(rel_quat, tan_ref)   # [N*H, 3]
    norm = quat_apply(rel_quat, norm_ref)  # [N*H, 3]

    tan_norm = torch.cat([tan, norm], dim=-1)  # [N*H, 6]
    return tan_norm.view(num_envs, n_hist, 6)

