# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-repetitive (rolling-window) Livox Mid-360 LiDAR sensor.

Why this exists
---------------
The core :class:`~isaaclab.sensors.lidar_sensor.LidarSensor` casts a **static** ray pattern:
:func:`~isaaclab.sensors.ray_caster.patterns.patterns.livox_pattern` slices one contiguous
window out of the packaged ``mid360.npy`` file at initialisation and re-casts those same
directions on every sensor update, identically for every environment.

A real Mid-360 does not work that way. Rosbag capture (2026-09-03) shows a **non-repetitive**
scan: every 0.1 s frame is a different contiguous window of the time-ordered pattern, so the
beam pattern sweeps the field of view rather than sampling it at fixed angles. Averaged over a
second the coverage is much denser than any single frame, and a policy trained against a frozen
pattern can learn to rely on rays that will not be there at deployment time.

This subclass restores that behaviour without touching the core sensor: it keeps the whole
800k-row pattern resident on the device and, before each cast, writes the next ``num_rays``-row
window into the ray-direction buffer in place. Environments are given independent scan phases so
they are not all looking at the same part of the pattern at the same moment.

Mechanism
---------
:attr:`~isaaclab.sensors.ray_caster.base_ray_caster.BaseRayCaster.ray_directions` is a
:class:`~isaaclab.utils.warp.proxy_array.ProxyArray` wrapping a ``wp.vec3f`` array of shape
``(num_envs, num_rays)``. Its ``.torch`` view is zero-copy (``wp.to_torch``), and the ray-cast
kernel re-reads ``ray_directions.warp`` on every ``_update_buffers_impl``. Writing into the torch
view therefore changes what the very next cast uses, with no re-allocation and no core change.
:meth:`_initialize_rays_impl` asserts the zero-copy property rather than assuming it.

Note on the offset-quaternion convention:
    ``offset.rot`` is documented as ``(w, x, y, z)`` on
    :class:`~isaaclab.sensors.ray_caster.ray_caster_cfg.RayCasterCfg.OffsetCfg`, but
    ``base_ray_caster._initialize_rays_impl`` feeds it straight to
    :func:`isaaclab.utils.math.quat_apply`, which reads ``(x, y, z, w)``. This module reproduces
    the core call **exactly**, mismatch included, so the rolling window lands in the same frame
    as the static pattern it replaces. Do not "fix" it here — that would silently rotate every
    trained checkpoint's field of view.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.sensors.lidar_sensor import LidarSensor
from isaaclab.sensors.ray_caster.patterns import patterns as _patterns
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from .mid360_rolling_lidar_cfg import Mid360RollingLidarSensorCfg

# Packaged Mid-360 scan pattern: (800000, 2) float32, columns [theta, phi] in radians, stored in
# acquisition (time) order. This is the same file ``livox_pattern`` slices its static window from,
# located relative to the patterns module so the two can never drift apart.
DEFAULT_SCAN_PATTERN_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(_patterns.__file__)), "scan_patterns", "mid360.npy"
)


@wp.kernel(enable_backward=False)
def advance_scan_pos_kernel(
    env_mask: wp.array(dtype=wp.bool),
    num_rays: wp.int32,
    pattern_size: wp.int32,
    scan_pos: wp.array(dtype=wp.int32),
):
    """Advance each masked environment's scan position by one frame.

    Args:
        env_mask: Boolean mask of environments to advance. Shape is (num_envs,).
        num_rays: Rays per frame, i.e. the window stride in pattern rows.
        pattern_size: Total rows in the scan pattern.
        scan_pos: Per-env row index of the current window start. Modified in place.
    """
    env = wp.tid()
    if not env_mask[env]:
        return
    scan_pos[env] = (scan_pos[env] + num_rays) % pattern_size


@wp.kernel(enable_backward=False)
def write_scan_window_kernel(
    env_mask: wp.array(dtype=wp.bool),
    scan_pos: wp.array(dtype=wp.int32),
    pattern: wp.array(dtype=wp.vec3f),
    pattern_size: wp.int32,
    mount_quat: wp.array(dtype=wp.quatf),
    apply_mount_quat: wp.int32,
    ray_directions: wp.array2d(dtype=wp.vec3f),
):
    """Copy each masked environment's current pattern window into the ray-direction buffer.

    Masked in the same style as the core ray-caster kernels: an unmasked environment returns
    before touching memory, so a no-op update costs nothing. That matters because
    :meth:`~isaaclab.sensors.sensor_base.SensorBase._update_outdated_buffers` calls
    ``_update_buffers_impl`` on every access, usually with an all-false mask.

    Args:
        env_mask: Boolean mask of environments to write. Shape is (num_envs,).
        scan_pos: Per-env row index of the current window start. Shape is (num_envs,).
        pattern: Unit ray directions for the whole pattern. Pre-rotated by the mount offset when
            ``apply_mount_quat`` is 0, raw sensor-frame otherwise. Shape is (pattern_size,).
        pattern_size: Total rows in the scan pattern.
        mount_quat: Per-env mount rotation. Only read when ``apply_mount_quat`` is 1.
            Shape is (num_envs,).
        apply_mount_quat: 1 to rotate by ``mount_quat``, 0 to copy the pre-rotated pattern
            unchanged. The two are kept as separate paths rather than folding an identity
            quaternion into one, because ``wp.quat_rotate`` and the torch ``quat_apply`` that
            pre-rotates the pattern do not agree in the last bits — and the ``None`` case must
            stay bit-identical to the behaviour that predates the randomisation.
        ray_directions: Body-local ray directions. Shape is (num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    d = pattern[(scan_pos[env] + ray) % pattern_size]
    if apply_mount_quat == 1:
        d = wp.quat_rotate(mount_quat[env], d)
    ray_directions[env, ray] = d


class Mid360RollingLidarSensor(LidarSensor):
    """Mid-360 LiDAR whose ray directions advance through the scan pattern each update.

    Behaves exactly like :class:`~isaaclab.sensors.lidar_sensor.LidarSensor` when
    ``cfg.rolling_scan`` is ``False``; that is the reference path for ablations.

    Attributes:
        mount_pitch_deg: ``(num_envs,)`` boom pitch per env [deg]. All zeros when
            ``cfg.mount_pitch_range_deg`` is None, since the single ``offset.rot`` applies.

            This is the pitch that takes effect from the env's **next** sensor update, not
            necessarily the one baked into the current ``ray_directions``. A reset redraws it
            immediately while the directions are rewritten only on the next
            ``_update_buffers_impl`` — deliberately, so an env's cached ``ray_hits_w`` always
            stays consistent with the directions they were cast with. With ``update_period``
            set to the measurement period, the two disagree for up to one sensor period after
            a reset, so do not use this as a label for the directions currently in the buffer.
        _scan_dirs_full: ``(P, 3)`` float32 unit ray directions for the whole pattern. Rotated
            into the body frame by ``cfg.offset.rot`` when the mount pitch is not randomised,
            raw sensor-frame when it is (the per-env rotation happens in the write kernel).
        _scan_pos: ``(num_envs,)`` int32 row index at which each environment's next window
            starts. Zero-copy torch view of ``_scan_pos_wp``.
    """

    cfg: Mid360RollingLidarSensorCfg
    """The configuration parameters."""

    """
    Implementation.
    """

    def _initialize_rays_impl(self) -> None:
        """Create the core ray buffers, then load the full pattern and seed the first window."""
        super()._initialize_rays_impl()

        # ``ray_directions`` is sized by ``_view_count`` while ``env_mask`` is sized by
        # ``_num_envs``; the rolling write indexes one with the other, so they must agree.
        if self._view_count != self._num_envs:
            raise RuntimeError(
                "Mid360RollingLidarSensor requires one sensor view per environment, got"
                f" view_count={self._view_count} and num_envs={self._num_envs}."
            )

        pattern_file = self.cfg.scan_pattern_file or DEFAULT_SCAN_PATTERN_FILE
        if not os.path.exists(pattern_file):
            raise FileNotFoundError(f"Mid-360 scan pattern file not found: {pattern_file}")

        # (P, 2) float32 [theta, phi] in radians, time-ordered.
        pattern = np.load(pattern_file)
        if pattern.ndim != 2 or pattern.shape[1] != 2:
            raise ValueError(f"Expected a (P, 2) [theta, phi] pattern in {pattern_file}, got {pattern.shape}.")
        if getattr(self.cfg.pattern_cfg, "downsample", 1) != 1:
            # ``livox_pattern`` downsamples AFTER slicing, so num_rays would be samples/downsample
            # while the window below advances over contiguous un-downsampled rows.
            raise ValueError("Mid360RollingLidarSensor requires pattern_cfg.downsample == 1.")
        if pattern.shape[0] < self.num_rays:
            raise ValueError(
                f"Scan pattern {pattern_file} has {pattern.shape[0]} rows, fewer than the"
                f" {self.num_rays} rays per frame requested by the pattern cfg."
            )

        theta = torch.from_numpy(np.ascontiguousarray(pattern[:, 0])).to(self._device)
        phi = torch.from_numpy(np.ascontiguousarray(pattern[:, 1])).to(self._device)

        # Spherical -> Cartesian, matching ``patterns._livox_scan_pattern`` exactly
        # (x = forward, y = left, z = up).
        cos_phi = torch.cos(phi)
        dirs = torch.stack([torch.cos(theta) * cos_phi, torch.sin(theta) * cos_phi, torch.sin(phi)], dim=1)
        dirs = (dirs / torch.norm(dirs, dim=1, keepdim=True)).to(torch.float32)

        # Mount rotation. With ``mount_pitch_range_deg`` set, every env has its own pitch, so the
        # pattern must stay in the raw sensor frame and be rotated per env inside the write
        # kernel. Otherwise it is pre-rotated once here, exactly as the core does for the static
        # pattern — see the module docstring for why the (documented) wxyz tuple is passed to an
        # xyzw-reading ``quat_apply``.
        self._mount_pitch_range: tuple[float, float] | None = self.cfg.mount_pitch_range_deg
        if self._mount_pitch_range is None:
            offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device, dtype=torch.float32)
            dirs = math_utils.quat_apply(offset_quat.repeat(dirs.shape[0], 1), dirs)
        elif not self.cfg.rolling_scan:
            raise ValueError("Mid360RollingLidarSensorCfg.mount_pitch_range_deg requires rolling_scan=True.")

        self._scan_dirs_full: torch.Tensor = dirs.contiguous()
        self._scan_pattern_size: int = int(dirs.shape[0])
        self._scan_dirs_full_wp = wp.from_torch(self._scan_dirs_full, dtype=wp.vec3f)

        # Per-env scan phase, int32 because it indexes the pattern inside a warp kernel and the
        # pattern is 8e5 rows. ``_scan_pos`` is the zero-copy torch view of ``_scan_pos_wp``;
        # writing either is visible to the other. Random phases decorrelate the environments,
        # so a training batch covers the whole pattern instead of one shared window.
        self._scan_pos_wp = wp.zeros(self._num_envs, dtype=wp.int32, device=self._device)
        self._scan_pos: torch.Tensor = wp.to_torch(self._scan_pos_wp)
        if self.cfg.random_scan_phase:
            self._scan_pos.copy_(torch.randint(0, self._scan_pattern_size, (self._num_envs,), device=self._device))

        # All-true mask for the initial seeding launch, kept for reuse.
        self._all_env_mask_wp = wp.ones(self._num_envs, dtype=wp.bool, device=self._device)

        # Per-env mount rotation, as a ``wp.quatf`` whose torch view is (num_envs, 4) in the
        # (x, y, z, w) storage order warp uses. Allocated even when the range is None (one
        # 16-byte row per env) so the kernel signature never changes; the kernel ignores it.
        self._mount_quat_torch: torch.Tensor = torch.zeros(self._num_envs, 4, device=self._device)
        self._mount_quat_torch[:, 3] = 1.0
        self._mount_quat_wp = wp.from_torch(self._mount_quat_torch.contiguous(), dtype=wp.quatf)
        self.mount_pitch_deg: torch.Tensor = torch.zeros(self._num_envs, device=self._device)
        if self._mount_pitch_range is not None:
            self._draw_mount_pitch(torch.arange(self._num_envs, device=self._device))
            print(
                f"[Mid360Rolling] mount pitch randomised over {self._mount_pitch_range} deg:"
                f" min={float(self.mount_pitch_deg.min()):.2f} mean={float(self.mount_pitch_deg.mean()):.2f}"
                f" max={float(self.mount_pitch_deg.max()):.2f} deg ({self._num_envs} envs)"
            )

        # Zero-copy contract: writing into ``.torch`` must be visible to the warp kernel that
        # reads ``.warp``. Checked here rather than assumed, because a silent copy would leave
        # the sensor casting the initial static window forever and every downstream number
        # would still look plausible.
        dirs_view = self.ray_directions.torch
        if dirs_view.data_ptr() != self.ray_directions.warp.ptr:
            raise RuntimeError(
                "ray_directions.torch is not a zero-copy view of ray_directions.warp; the rolling"
                " scan cannot write into the buffer the ray-cast kernel reads."
            )

        # Seed each env's first window so the very first cast already uses a real window rather
        # than the static slice the core installed.
        if self.cfg.rolling_scan:
            self._write_scan_windows(self._all_env_mask_wp)

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        """Advance the scan window for the updating environments, then cast.

        Only environments selected by ``env_mask`` advance and get new directions written. An
        environment that is not updating keeps its cached ``ray_hits_w`` from an earlier cast,
        and those hits are only meaningful alongside the directions they were cast with — so
        rewriting its directions here would silently desynchronise the two.

        Args:
            env_mask: Boolean warp array of shape ``(num_envs,)`` marking the environments whose
                buffers are due for an update.
        """
        if self.cfg.rolling_scan:
            # Both steps run as masked warp kernels, for two reasons. Converting the mask to
            # indices (``nonzero``) would force a device-to-host synchronisation on every call,
            # and a torch gather would move the whole ``(num_envs, num_rays, 3)`` buffer even
            # when the mask is empty. This method is called on *every* sensor data access, and
            # with ``update_period`` set the mask is empty on most of them.
            wp.launch(
                advance_scan_pos_kernel,
                dim=self._num_envs,
                inputs=[env_mask, self.num_rays, self._scan_pattern_size, self._scan_pos_wp],
                device=self._device,
            )
            self._write_scan_windows(env_mask)

        super()._update_buffers_impl(env_mask)

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the sensor and re-randomise the scan phase of the reset environments.

        A real sensor's phase at the start of an episode is arbitrary, so holding the phase
        across resets would make it a fixed function of the step index — something a policy can
        latch onto in simulation and never find at deployment.

        Args:
            env_ids: Environment indices to reset. ``None`` resets all environments.
            env_mask: Boolean warp array of shape ``(num_envs,)``. Takes priority over
                ``env_ids`` when given, matching
                :meth:`~isaaclab.sensors.sensor_base.SensorBase.reset`.
        """
        super().reset(env_ids, env_mask)

        # Called once during sim start-up before the ray buffers exist.
        if not hasattr(self, "_scan_pos"):
            return

        if env_mask is not None:
            reset_ids = wp.to_torch(env_mask).to(torch.bool).view(-1).nonzero(as_tuple=False).squeeze(-1)
        elif env_ids is None:
            reset_ids = torch.arange(self._num_envs, device=self._device)
        else:
            reset_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.int64).view(-1)

        if reset_ids.numel() == 0:
            return

        # The two draws are independent: an ablation may randomise the mount pitch while holding
        # the scan phase fixed, or the reverse.
        if self.cfg.random_scan_phase:
            self._scan_pos[reset_ids] = torch.randint(
                0, self._scan_pattern_size, (reset_ids.numel(),), device=self._device, dtype=torch.int32
            )
        if self._mount_pitch_range is not None:
            self._draw_mount_pitch(reset_ids)

    """
    Helpers.
    """

    def _draw_mount_pitch(self, env_ids: torch.Tensor) -> None:
        """Draw a fresh mount pitch for the given envs and rebuild their mount quaternions.

        The quaternion must reproduce the rotation the core actually performs, not the one the
        cfg documents: ``base_ray_caster`` hands the ``(w, x, y, z)`` tuple
        ``(0, cos(p/2), 0, sin(p/2))`` to :func:`isaaclab.utils.math.quat_apply`, which reads it
        as ``(x, y, z, w)``. Warp stores ``wp.quatf`` in that same ``(x, y, z, w)`` order, so the
        components are written straight into those slots. A degenerate range reproduces the
        single-rotation path exactly, which the smoke test checks.

        The new directions are not written here. A reset env is marked outdated, so the next
        ``_update_buffers_impl`` rewrites its window with the new pitch; until then its ray
        directions still match the ``ray_hits_w`` they were cast with.

        Args:
            env_ids: Environment indices to redraw. Shape is (M,).
        """
        lo, hi = self._mount_pitch_range  # type: ignore[misc]
        pitch = torch.empty(env_ids.numel(), device=self._device).uniform_(float(lo), float(hi))
        self.mount_pitch_deg[env_ids] = pitch

        half = torch.deg2rad(pitch) * 0.5
        self._mount_quat_torch[env_ids, 0] = 0.0  # x
        self._mount_quat_torch[env_ids, 1] = torch.cos(half)  # y
        self._mount_quat_torch[env_ids, 2] = 0.0  # z
        self._mount_quat_torch[env_ids, 3] = torch.sin(half)  # w

    def _write_scan_windows(self, env_mask: wp.array) -> None:
        """Write the current pattern window of every masked environment into the ray buffer.

        Environments outside the mask keep the directions their cached ``ray_hits_w`` were cast
        with, and the kernel returns before touching their rows at all.

        Args:
            env_mask: Boolean warp array of shape ``(num_envs,)``, True where the window should
                be rewritten.
        """
        wp.launch(
            write_scan_window_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._scan_pos_wp,
                self._scan_dirs_full_wp,
                self._scan_pattern_size,
                self._mount_quat_wp,
                int(self._mount_pitch_range is not None),
                self.ray_directions.warp,
            ],
            device=self._device,
        )
