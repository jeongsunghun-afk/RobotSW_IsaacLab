# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Parkour-Imitation TerrainStyle environment.

Subclass of Go2ParkourImitationEnv that narrows the AMP discriminator observation
to **terrain-invariant features only**, so a flat-ground demo can guide gait style
on all terrain types without OOD collapse.

AMP discriminator observation layout (37-dim per step, history 10 → 370-dim flat):

    Feature                   dim   removed reason
    ─────────────────────────────────────────────────────────
    dof_pos                    12   kept — gait phase/pattern
    dof_vel                    12   kept — gait velocity
    root_lin_vel_b  xy only     2   kept z removed (jump/fall vertical vel OOD)
    root_ang_vel_b              3   kept — turning behaviour
    foot_pos_local  xy only     8   kept z removed (foot clearance OOD on terrain)
    ─────────────────────────────────────────────────────────
    Total                      37

Removed vs parent (49-dim):
    root_height            (1)  — absolute height, terrain-dependent
    root_lin_vel_b z       (1)  — vertical velocity, jump/drop OOD
    foot_pos_local z×4     (4)  — foot clearance, terrain-dependent
    root_rot_tan_norm      (6)  — pitch/roll orientation, ramp/obstacle OOD
    ─────────────────────────────────────────────────────────
    Removed total         (12)   49 − 12 = 37 ✓

foot_pos_local layout: local_foot_pos is [N, 4, 3] with order [FL, FR, RL, RR].
When flattened it becomes [FL_x, FL_y, FL_z, FR_x, FR_y, FR_z, RL_x, RL_y, RR_x, RR_y].
z-only removal is done via reshape(-1, 4, 3)[..., :2].reshape(-1, 8).

Live path:  _update_amp_obs_buf_terrain_style → _amp_obs_buf [N, H, 37]
Expert path: get_amp_observations (override) → cat same 37-dim per frame
Both paths use _build_flat_amp_obs_terrain_style (no tan_norm, no quat_buf).

_amp_quat_buf is NOT allocated in this subclass (no tan_norm computation).

Parent class (Go2ParkourImitationEnv) is unchanged — Go2-ParkourImitation-v0 unaffected.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import quat_apply_inverse

from .parkour_imitation_env import Go2ParkourImitationEnv
from .parkour_imitation_env_cfg import ParkourImitationEnvCfg

# Per-frame AMP obs dim for terrain-invariant variant.
# dof_pos(12) + dof_vel(12) + root_lin_vel_b_xy(2) + root_ang_vel_b(3) + foot_pos_local_xy(8) = 37
_TERRAIN_STYLE_AMP_DIM = 37


class ParkourImitationTerrainStyleEnv(Go2ParkourImitationEnv):
    """Parkour-Imitation with terrain-invariant AMP discriminator observations.

    Inherits all parkour mechanics and AMP infrastructure from Go2ParkourImitationEnv
    (flat_env_mask, _amp_reward_buf, motion_lib, reset ordering, log metrics, etc.)
    and overrides only the amp_obs construction to use 37-dim terrain-invariant features.

    The parent's _amp_quat_buf (for tan_norm) is NOT needed here. __init__ zeros it
    out and replaces _amp_obs_buf with the correct [N, H, 37] shape. No new buffers
    are added beyond what the parent already allocates.

    cfg.amp_obs_dim must be set to 37 (done in ParkourImitationTerrainStyleEnvCfg).
    cfg.amp_history_length is inherited (default 10) → flat dim = 370.
    """

    cfg: ParkourImitationEnvCfg  # cfg-worker will provide a subclass with amp_obs_dim=37

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, cfg: ParkourImitationEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ allocates _amp_obs_buf [N, H, 43] and _amp_quat_buf [N, H, 4]
        # with parent's amp_obs_dim=49. We override both buffers below.
        super().__init__(cfg, render_mode, **kwargs)

        H = cfg.amp_history_length

        # Override _amp_obs_buf: parent allocated [N, H, 43]; we need [N, H, 37].
        # The parent's 43-dim base buffer is replaced with our 37-dim terrain-invariant buffer.
        self._amp_obs_buf = torch.zeros(
            self.num_envs,
            H,
            _TERRAIN_STYLE_AMP_DIM,
            device=self.device,
        )

        # _amp_quat_buf is NOT used in this subclass (tan_norm removed).
        # Set to a zero [N, H, 4] placeholder so any accidental parent call is benign.
        self._amp_quat_buf = torch.zeros(self.num_envs, H, 4, device=self.device)
        self._amp_quat_buf[..., 0] = 1.0  # identity — unused but safe

        # Override _terminal_amp_obs: parent allocated with parent's amp_obs_dim (49).
        # This subclass uses amp_obs_dim=37 (set in cfg), so reallocate to match.
        _amp_flat_dim = H * _TERRAIN_STYLE_AMP_DIM  # 10 * 37 = 370
        self._terminal_amp_obs = torch.zeros(self.num_envs, _amp_flat_dim, device=self.device)

        # Sanity-check: cfg.amp_obs_dim must match _TERRAIN_STYLE_AMP_DIM.
        assert cfg.amp_obs_dim == _TERRAIN_STYLE_AMP_DIM, (
            f"ParkourImitationTerrainStyleEnv requires cfg.amp_obs_dim={_TERRAIN_STYLE_AMP_DIM}, "
            f"got {cfg.amp_obs_dim}. Use ParkourImitationTerrainStyleEnvCfg."
        )

    # ------------------------------------------------------------------
    # Override: live AMP obs construction
    # ------------------------------------------------------------------

    def _update_amp_obs_buf(self) -> None:
        """Compute 37-dim terrain-invariant AMP frame and push into ring buffer.

        Terrain-invariant AMP obs layout (37-dim):
            dof_pos            (12)  joint positions (rad)
            dof_vel            (12)  joint velocities (rad/s)
            root_lin_vel_b_xy   (2)  body-frame linear vel, x and y only (m/s)
            root_ang_vel_b      (3)  body-frame angular velocity (rad/s)
            foot_pos_local_xy   (8)  [FL,FR,RL,RR]×xy in root body frame (m)

        Removed vs parent:
            root_height (1), root_lin_vel_b_z (1), foot_pos_local_z×4 (4), tan_norm (6)

        Ring-buffer: oldest at index 0, newest at index -1 (same as parent).
        """
        root_pos_w = self._robot.data.root_pos_w  # [N, 3]
        root_quat_w = self._robot.data.root_quat_w  # [N, 4]  wxyz
        root_lin_vel_b = self._robot.data.root_lin_vel_b  # [N, 3]
        root_ang_vel_b = self._robot.data.root_ang_vel_b  # [N, 3]

        # Foot positions in base-local frame: [N, 4, 3] → [FL,FR,RL,RR]×xyz
        foot_pos_w = self._robot.data.body_pos_w[:, self._amp_foot_body_ids, :]  # [N, 4, 3]
        rel_pos = foot_pos_w - root_pos_w.unsqueeze(1)  # [N, 4, 3]
        N, K = rel_pos.shape[:2]
        local_foot_pos = quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
            rel_pos.reshape(-1, 3),
        ).view(N, K, 3)  # [N, 4, 3]  layout: [FL, FR, RL, RR] × [x, y, z]

        # Extract xy only per foot: [N, 4, 3] → [N, 4, 2] → [N, 8]
        foot_pos_local_xy = local_foot_pos[..., :2].reshape(N, -1)  # [N, 8]

        # Build 37-dim terrain-invariant AMP frame
        current_frame = torch.cat(
            [
                self._robot.data.joint_pos,  # 12  rad
                self._robot.data.joint_vel,  # 12  rad/s
                root_lin_vel_b[:, :2],  #  2  m/s  (x, y only — z removed)
                root_ang_vel_b,  #  3  rad/s
                foot_pos_local_xy,  #  8  m    (xy only — z removed)
            ],
            dim=-1,
        )  # [N, 37]

        # Roll history left (discard oldest at index 0) and write newest at index -1
        self._amp_obs_buf = torch.roll(self._amp_obs_buf, shifts=-1, dims=1)
        self._amp_obs_buf[:, -1, :] = current_frame

        # _amp_quat_buf is not used in this subclass; no update needed.

    # ------------------------------------------------------------------
    # Override: flatten AMP obs (no tan_norm)
    # ------------------------------------------------------------------

    def _build_flat_amp_obs(
        self,
        base_buf: torch.Tensor,
        quat_buf: torch.Tensor,  # unused — tan_norm removed; kept for parent signature compatibility
    ) -> torch.Tensor:
        """Flatten [N, H, 37] AMP buffer — no tan_norm appended.

        Args:
            base_buf: [N, H, 37] — terrain-invariant AMP history (ring-buffer, newest at -1)
            quat_buf: [N, H, 4]  — ignored (tan_norm removed in this subclass)

        Returns:
            Tensor [N, H*37] — flattened 37-dim AMP obs for discriminator input.
        """
        return base_buf.flatten(start_dim=1)  # [N, H*37] = [N, 370]

    # ------------------------------------------------------------------
    # Override: amp_observation_space property
    # ------------------------------------------------------------------

    @property
    def amp_observation_space(self):
        """Gym space describing the flattened terrain-invariant AMP observation.

        Shape: (amp_history_length * 37,) = (10 * 37,) = (370,).
        """
        import gymnasium as gym

        dim = self.cfg.amp_history_length * _TERRAIN_STYLE_AMP_DIM  # 10 * 37 = 370
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Override: expert AMP obs from motion library
    # ------------------------------------------------------------------

    def get_amp_observations(self, num_samples: int) -> torch.Tensor:
        """Return expert AMP observation samples — terrain-invariant 37-dim per frame.

        Layout per frame (37-dim) matches _update_amp_obs_buf exactly:
            dof_pos(12) + dof_vel(12) + root_lin_vel_b_xy(2) + root_ang_vel_b(3) +
            foot_pos_local_xy(8) = 37

        History ordering (oldest-first) is identical to parent:
            returned[:, :37]   = frame at t-(H-1)*step_dt  (oldest)
            returned[:, -37:]  = frame at t                (newest)

        Args:
            num_samples: Number of independent expert windows to sample.

        Returns:
            Tensor of shape (num_samples, amp_history_length * 37) = (num_samples, 370),
            dtype float32, on self.device.
        """
        H = self.cfg.amp_history_length
        motion_ids = self._motion_lib.sample_motions(num_samples)
        times = self._motion_lib.sample_times(motion_ids)

        # Time offsets: index 0 → oldest (t-(H-1)*dt), index H-1 → newest (t)
        offsets = torch.arange(H - 1, -1, -1, device=self.device).float() * self.step_dt  # [H]
        times_flat = (times.unsqueeze(-1) - offsets.unsqueeze(0)).clamp(min=0.0).reshape(-1)  # [N*H]
        ids_flat = motion_ids.unsqueeze(-1).expand(-1, H).reshape(-1)  # [N*H]

        _, _, lv, av, dp, dv, fp = self._motion_lib.calc_motion_frame(ids_flat, times_flat)
        # return order: root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos_local
        # root_pos/root_quat unused here (no root_height, no tan_norm in terrain-invariant obs)
        # lv: [N*H, 3], av: [N*H, 3], dp: [N*H, 12], dv: [N*H, 12], fp: [N*H, 4, 3]

        # Re-order DOFs from motion_lib order to IsaacLab joint order (mirrors live side)
        dp = dp[:, self._motion_dof_indices]
        dv = dv[:, self._motion_dof_indices]

        # foot_pos_local layout: fp is [N*H, 4, 3] = [FL,FR,RL,RR] × [x,y,z]
        # Extract xy only: [N*H, 4, 3] → [N*H, 4, 2] → [N*H, 8]
        foot_pos_local_xy = fp[..., :2].reshape(fp.shape[0], -1)  # [N*H, 8]

        # Build 37-dim terrain-invariant frame (same layout as live _update_amp_obs_buf)
        frame = torch.cat(
            [
                dp,  # 12  rad
                dv,  # 12  rad/s
                lv[:, :2],  #  2  m/s  (x, y only — z removed)
                av,  #  3  rad/s
                foot_pos_local_xy,  #  8  m    (xy only — z removed)
            ],
            dim=-1,
        )  # [N*H, 37]

        # Reshape to [N, H, 37] then flatten to [N, H*37]
        base_buf = frame.view(num_samples, H, _TERRAIN_STYLE_AMP_DIM)  # [N, H, 37]
        return base_buf.flatten(start_dim=1)  # [N, 370]
