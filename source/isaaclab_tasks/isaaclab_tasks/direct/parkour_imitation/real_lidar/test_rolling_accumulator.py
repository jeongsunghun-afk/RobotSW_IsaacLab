# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the rolling world-anchored LiDAR occupancy accumulator.

The rolling accumulator (``cfg.lidar_grid_accumulate_mode = "rolling"``) exists because two
reports measured two separate defects in the warp accumulator it replaces:

* ``reports/go2_parkour/_comparisons/real_lidar_vs_sim/acc_diag/README.md`` — trilinear warping
  every 0.1 s dilutes the one-cell-thick ground sheet under sub-cell motion on every axis, so the
  0.8 m blind zone reads empty (near-zone lit fraction 0.16-0.27 measured against 0.99 predicted
  from decay alone above 0.6 m/s).
* ``reports/presentation/05_parkour_learning/assets/render_acc_fix/README.md`` — nearest transport
  with residual carry restores the blind zone (0.56-0.89 at |y| <= 0.3 m) but closes a 3-cell hole
  once per-tick yaw jitter reaches ~4 deg, and *every* variant including the one in production
  marks gap trenches as occupied at ground level (0.485 warp / 0.705 nearest, threshold 0.5)
  because decayed-max has no operation that clears a cell.

Sub-tests, selected with ``--test``:

``synth``
    Sim-free transport exactness, run against the env's real ``_scroll_store`` / ``_readout_store``
    through a duck-typed stub, so this is not a re-implementation of the arithmetic under test.
``warp_identity``
    The default ``"warp"`` mode still equals a frozen copy of the pre-change accumulator body,
    tick for tick, on real frames.
``rotation``
    Pure-yaw rotation invariance in sim: the terrain the readout reports must not move in world
    coordinates while the robot spins in place.
``carving``
    Sim-free assertions on the free-space rules: what a ground hit protects, what a trench hit
    clears, that a hit beats a carve in the same tick, that a discarded return carves nothing,
    and that the ray stride selects a deterministic subset.
``shadow``
    Blind-zone fill and gap-trench false occupancy in sim.  The env stays in its default warp
    mode so the policy is on-distribution, and every rolling variant runs as a shadow on the same
    frames.  A measurement rather than a rule: it asserts only that it produced samples.
``perf``
    ms per CONTROL step and peak GPU memory for warp / rolling / rolling+free.  A measurement;
    a configuration that raises for any reason other than running out of GPU memory fails.

.. code-block:: bash

    source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate isaac-6.0
    cd /home/lgb/IsaacLab-6.0
    CUDA_VISIBLE_DEVICES=1 env -u DISPLAY ./isaaclab.sh -p \\
      source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/real_lidar/test_rolling_accumulator.py \\
      --test synth,carving,warp_identity,rotation --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROLL = "Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-Roll-EasyEntry-v0"
_ROLL_NF = "Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-RollNoFree-EasyEntry-v0"
_REAL = "Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-EasyEntry-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--test", type=str, default="synth,carving,warp_identity,rotation", help="Comma-separated sub-tests."
)
parser.add_argument("--num_envs", type=int, default=16, help="Envs for the sim sub-tests.")
parser.add_argument("--steps", type=int, default=200, help="Recorded control steps for the shadow table.")
parser.add_argument("--warmup", type=int, default=40, help="Control steps before recording starts.")
parser.add_argument("--terrain", type=str, default="gap", help="Terrain to isolate for the shadow table.")
parser.add_argument("--pin_level", type=int, default=3, help="Terrain level to pin (curriculum neutralised).")
parser.add_argument(
    "--task",
    type=str,
    default=_ROLL,
    help="Task for the rotation sub-test (the shadow table always uses the RealSensor arm).",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint for the shadow table.")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--perf_envs", type=int, nargs="+", default=[1024, 1280], help="Env counts for the perf pass.")
parser.add_argument("--perf_steps", type=int, default=100, help="Control steps to time per configuration.")
parser.add_argument("--out_json", type=str, default=None, help="Write the collected numbers here.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_lidar_env import (  # noqa: E402
    Go2ParkourImitationRandomGoalLidarEnv as _Env,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

RESULTS: dict = {}
FAILURES: list[str] = []


def _check(name: str, ok: bool, detail: str) -> None:
    """Record one assertion and print it; failures are collected, not raised."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# ======================================================================
# (b) sim-free transport exactness
# ======================================================================


class _StubCfg:
    """The cfg fields the store methods read, defaulted to the shipped values."""

    lidar_grid_store_radius = 2.3
    lidar_grid_store_z_margin_cells = 1
    lidar_grid_chunk_size = 256
    lidar_grid_readout_mode = "bilinear"
    lidar_grid_free_space = True
    lidar_grid_free_space_rule = "above_hit"
    lidar_grid_free_space_above_hit_cells = 2.5
    lidar_grid_free_space_hit_margin_cells = 1
    lidar_grid_free_space_beta = 0.5
    lidar_grid_free_space_step = 0.1
    lidar_grid_free_space_min_rays = 1
    lidar_grid_free_space_ray_stride = 1
    lidar_grid_free_space_debug_count = False
    lidar_grid_free_space_chunk_size = 64

    def __init__(self, **over):
        for k, v in over.items():
            if not hasattr(type(self), k):
                raise AttributeError(f"unknown cfg field {k}")
            setattr(self, k, v)


class _StoreStub:
    """Duck-typed stand-in for the env, carrying only what the store methods read.

    The point of the stub is that the sim-free tests exercise
    :meth:`~...Go2ParkourImitationRandomGoalLidarEnv._scroll_store`, ``_readout_store``,
    ``_carve_free_space`` and ``_scatter_hits_into_store`` themselves rather than a copy of them,
    so a test cannot pass while the shipped arithmetic is wrong.  Even the store geometry comes
    from the shipped ``_setup_rolling_store``.
    """

    _OWN = (
        "_setup_rolling_store",
        "_snap_origin",
        "_scroll_store",
        "_scatter_hits_into_store",
        "_carve_free_space",
        "_readout_store",
    )

    def __init__(self, device: str, n_env: int = 1, mount_x: float = 0.333644, **cfg_over):
        from isaaclab_tasks.direct.parkour.voxel_occupancy import VoxelOccupancyCfg, voxel_grid_shape

        self.device = device
        self.num_envs = n_env
        self.cfg = _StubCfg(**cfg_over)
        self._lidar_grid_cfg = VoxelOccupancyCfg()
        self._lidar_grid_shape = voxel_grid_shape(self._lidar_grid_cfg)
        res = self._lidar_grid_cfg.resolution
        nx, ny, nz = self._lidar_grid_shape
        lo = torch.tensor(
            [self._lidar_grid_cfg.x_range[0], self._lidar_grid_cfg.y_range[0], self._lidar_grid_cfg.z_range[0]],
            device=device,
        )
        self._lidar_grid_lo = lo
        ii = torch.meshgrid(
            torch.arange(nx, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nz, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self._lidar_grid_centres = lo + torch.stack(ii, dim=-1) * res
        # Grid origin -> Mid-360 mount, body frame.  Read by the sample and reach arithmetic.
        self._lidar_grid_origin_shift = torch.tensor([mount_x, 0.0, 0.0], device=device)
        for meth in self._OWN:
            setattr(self, meth, types.MethodType(getattr(_Env, meth), self))
        self._setup_rolling_store(self.cfg)

    # bound wrappers so the calls below read like normal method calls
    def scroll(self, shift):
        return self._scroll_store(self._lidar_grid_store, shift)

    def readout(self, origin, yaw):
        return self._readout_store(origin, yaw)

    def snap(self, origin):
        return self._snap_origin(origin)

    def z_index(self, z_m: float) -> int:
        """Store z index of a metric height above the grid origin."""
        return int(round((z_m - float(self._lidar_store_lo[2])) / self._lidar_grid_cfg.resolution))


def test_synth(device: str) -> dict:
    """Transport a synthetic sheet with a 3-cell hole and check value, hole width and position.

    Mirrors ``render_acc_fix/warp_variants.py``'s ``test_transport`` / ``test_hole_transport``:
    forward motion 0.085 m per tick (the measured speed), dz 0.02 m per tick, yaw jitter at two
    amplitudes, ten ticks, and **no new observations** so only transport is under test.  The hole
    width is measured differentially against an identical run with no hole, because a sheet moving
    forward drags its own emptied rear edge into a fixed window and that edge would otherwise be
    counted as hole.
    """
    print("\n=== (b) sim-free transport exactness ===")
    alpha, dx, dz, n_ticks = 0.94, 0.085, 0.02, 10
    out: dict = {}
    for yaw_deg in (2.0, 5.0):
        runs = {}
        for holed in (True, False):
            st = _StoreStub(device)
            res = st._lidar_grid_cfg.resolution
            sxy, _, sz = st._lidar_store_shape
            # Ground sheet: one z layer, uniform in x and y.  The hole is 3 store columns wide in
            # x, spanning all y, placed 1.0 m ahead of the start so it stays in the readout box.
            zg = sz // 2
            # Striped, not uniform.  A uniform sheet is invariant under any x shift, so a
            # transport that moved the map the wrong way — or not at all — would still show the
            # right peak and the right hole width; only one assertion would catch it.  Stripes
            # every 4 cells in x and 3 in y make every one of the checks below sensitive to the
            # sign and size of the shift.
            ax = torch.arange(sxy, device=device).view(sxy, 1)
            ay = torch.arange(sxy, device=device).view(1, sxy)
            stripes = torch.where((ax % 4 == 0) | (ay % 3 == 0), 1.0, 0.6)
            st._lidar_grid_store[0, :, :, zg] = stripes
            hole_x0 = sxy // 2 + int(round(1.0 / res))
            if holed:
                st._lidar_grid_store[0, hole_x0 : hole_x0 + 3, :, zg] = 0.0
            hole_world_x = (hole_x0 + 1 - sxy // 2) * res  # centre column, world x, fixed forever

            origin = torch.zeros(1, 3, device=device)
            origin[0, 2] = 0.0
            st._lidar_store_snap = st.snap(origin)
            peaks, rows, errs = [], [], []
            for k in range(n_ticks):
                origin = origin.clone()
                origin[0, 0] += dx
                origin[0, 2] += dz
                yaw = torch.tensor([math.radians(yaw_deg) * (1.0 if k % 2 == 0 else -1.0)], device=device)
                snap_new = st.snap(origin)
                shift = torch.round((snap_new - st._lidar_store_snap) / res).long()
                st._lidar_grid_store = st.scroll(shift) * alpha
                st._lidar_store_snap = snap_new
                ro = st.readout(origin, yaw).view(*st._lidar_grid_shape)
                peaks.append(float(st._lidar_grid_store.max()))
                # Centre y row of the readout at the z index the sheet is nearest to.
                nx, ny, nz = st._lidar_grid_shape
                zr = int(torch.argmax(ro.amax(dim=(0, 1))))
                rows.append(ro[:, ny // 2, zr].clone())
                # Position: world x of the store column that still holds the sheet peak, minus
                # the true (unmoving) world x of the hole centre.
                errs.append(float(origin[0, 0]))
            runs[holed] = {"peaks": peaks, "rows": rows, "hole_world_x": hole_world_x, "ox": errs}

        # value: the store peak must be exactly alpha**n — no dilution at all
        ratio = [p / (alpha ** (k + 1)) for k, p in enumerate(runs[False]["peaks"])]
        _check(
            f"store peak == 0.94^n (yaw +-{yaw_deg} deg)",
            max(abs(r - 1.0) for r in ratio) < 1e-6,
            f"max relative deviation {max(abs(r - 1.0) for r in ratio):.3e} over {n_ticks} ticks",
        )

        # hole width, differential, on the readout centre row
        def hole_mask(k: int) -> torch.Tensor:
            """Columns the holed run lost relative to the identical un-holed run.

            Relative, not against a fixed fraction of the peak: the sheet is striped, so an
            absolute threshold would flag the dim stripes as hole.
            """
            r_h, r_n = runs[True]["rows"][k], runs[False]["rows"][k]
            return (r_n > 0.1 * alpha ** (k + 1)) & (r_h < 0.5 * r_n)

        widths = [int(hole_mask(k).sum().item()) for k in range(n_ticks)]
        _check(
            f"hole stays 3 columns (yaw +-{yaw_deg} deg)",
            all(abs(w - 3) <= 1 for w in widths),
            f"widths per tick {widths}",
        )

        # position: readout hole centre in world x against the true value
        pos_err = []
        for k in range(n_ticks):
            sel = hole_mask(k).nonzero().flatten()
            if sel.numel() == 0:
                pos_err.append(float("nan"))
                continue
            xs = st._lidar_grid_lo[0].item() + sel.float() * st._lidar_grid_cfg.resolution
            # readout x is body-frame; the grid origin sits at world x = origin_x
            world_x = float(xs.mean()) + runs[True]["ox"][k]
            pos_err.append(abs(world_x - runs[True]["hole_world_x"]))
        finite = [e for e in pos_err if np.isfinite(e)]
        _check(
            f"residual never accumulates (yaw +-{yaw_deg} deg)",
            len(finite) == n_ticks and max(finite) <= 0.5 * st._lidar_grid_cfg.resolution + 1e-6,
            f"max |world-x error| {max(finite):.4f} m = {max(finite) / 0.1:.2f} cell over {n_ticks} ticks",
        )

        # stripe phase: the bright stripes are fixed in the world, so their phase in world x must
        # not move.  This is what the stripes are for — a transport with the wrong sign, or one
        # that does not move at all, keeps the peak and the hole width but slides this phase by
        # the full accumulated displacement.
        period = 4 * res  # the store's x stripe period
        phases = []
        for k in range(n_ticks):
            r_n = runs[False]["rows"][k]
            bright = (r_n > 0.9 * r_n.max()).nonzero().flatten()
            xs = st._lidar_grid_lo[0].item() + bright.float() * res + runs[False]["ox"][k]
            ang = 2.0 * math.pi * xs / period
            phases.append(math.atan2(float(torch.sin(ang).mean()), float(torch.cos(ang).mean())))
        drift = [abs((p - phases[0] + math.pi) % (2 * math.pi) - math.pi) * period / (2 * math.pi) for p in phases]
        _check(
            f"stripe phase holds its world position (yaw +-{yaw_deg} deg)",
            max(drift) < 0.5 * res,
            f"max world-x phase drift {max(drift):.4f} m = {max(drift) / res:.2f} cell over "
            f"{n_ticks} ticks (accumulated displacement {n_ticks * dx:.2f} m)",
        )

        # readout blur: bounded and non-compounding, unlike the warp path's
        ro_peaks = [float(r.max()) for r in runs[False]["rows"]]
        blur = [ro_peaks[k] / (alpha ** (k + 1)) for k in range(n_ticks)]
        out[f"yaw{yaw_deg:g}"] = {
            "store_peak": runs[False]["peaks"],
            "readout_peak": ro_peaks,
            "readout_over_store": blur,
            "hole_widths": widths,
            "pos_err_m": pos_err,
            "stripe_phase_drift_m": drift,
        }
        print(
            f"  readout/store peak ratio (z blur from the sub-cell residual): "
            f"min {min(blur):.3f} max {max(blur):.3f} — bounded, does not compound"
        )
    return out


# ======================================================================
# (g) warp mode byte-identity against a frozen pre-change reference
# ======================================================================


def _ray_to(stub, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Body-frame direction and range of a ray from the Mid-360 mount to ``target``.

    Args:
        stub: A :class:`_StoreStub`.
        target: ``(3,)`` hit point in the store frame [m] (identity orientation, zero residual,
            so the store frame and the body frame coincide).

    Returns:
        ``(dir, dist)`` — unit direction ``(3,)`` and range ``()`` [m].
    """
    v = target - stub._lidar_grid_origin_shift
    d = torch.linalg.vector_norm(v)
    return v / d, d


def test_carving(device: str) -> dict:
    """Sim-free assertions on the shipped ``_carve_free_space``.

    Geometry is chosen so the store frame, the body frame and the world frame all coincide: the
    orientation is identity and the sub-cell residual is zero, so a point's store coordinates are
    just its metres from the grid origin.  The store starts uniformly at 1.0, so any cell that
    reads ``beta`` afterwards was carved and any cell still at 1.0 was not.

    The ground plane sits 0.4 m below the grid origin, which is where it sits in the real arm
    (base height ~0.35 m plus the 0.05 m clearance mount).
    """
    print("\n=== (f) free-space carving rules ===")
    out: dict = {}
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)  # xyzw identity
    off = torch.zeros(1, 3, device=device)
    res = 0.1
    z_ground = -0.4  # grid origin to ground
    z_deep = -0.7  # trench floor, 3 cells below the 2.5-cell guard
    z_shallow = -0.6  # trench floor only 2 cells down, inside the guard

    def fresh(**cfg_over):
        st = _StoreStub(device, **cfg_over)
        st._lidar_grid_store.fill_(1.0)
        return st

    def rays(st, targets, slots=None, n_rays=None):
        """Build ``(dirs, dists, valid)`` for one ray per target, optionally at chosen slots."""
        n = n_rays if n_rays is not None else len(targets)
        dir_t = torch.zeros(1, n, 3, device=device)
        dst_t = torch.full((1, n), 40.0, device=device)
        val_t = torch.zeros(1, n, dtype=torch.bool, device=device)
        for i, tgt in enumerate(targets):
            d, r = _ray_to(st, torch.tensor(tgt, device=device))
            slot = i if slots is None else slots[i]
            dir_t[0, slot] = d
            dst_t[0, slot] = r
            val_t[0, slot] = True
        return dir_t, dst_t, val_t

    def carve(st, targets, slots=None, n_rays=None):
        dir_t, dst_t, val_t = rays(st, targets, slots, n_rays)
        st._carve_free_space(st._lidar_grid_store, dst_t, val_t, off, ident, dir_t)
        return st._lidar_grid_store[0]

    # (a) a ray ending on the ground carves the air it crossed, never its own ground plane ----
    st = fresh()
    g = carve(st, [(1.6, 0.0, z_ground)])
    zg = st.z_index(z_ground)
    n_ground = int((g[:, :, zg] < 1.0).sum())
    n_above = int((g[:, :, zg + 1 :] < 1.0).sum())
    n_below = int((g[:, :, :zg] < 1.0).sum())
    _check(
        "ground hit carves the air it crossed, never its own ground plane",
        n_ground == 0 and n_above > 0 and n_below == 0,
        f"carved cells: ground plane {n_ground} (want 0), above it {n_above} (want > 0), below it {n_below} (want 0)",
    )
    out["ground_hit"] = {"ground": n_ground, "above": n_above, "below": n_below}

    # (b) a ray into a deep trench clears the ground-level cells bridging it, not its floor ---
    st = fresh()
    g = carve(st, [(1.6, 0.0, z_deep)])
    zt = st.z_index(z_deep)
    ix = int(round((1.6 - float(st._lidar_store_lo[0])) / res))
    iy = int(round((0.0 - float(st._lidar_store_lo[1])) / res))
    n_ground = int((g[:, :, st.z_index(z_ground)] < 1.0).sum())
    floor_cell = float(g[ix, iy, zt])
    _check(
        "deep trench hit clears the ground-level cells above it, not the floor",
        n_ground > 0 and floor_cell == 1.0,
        f"ground-level cells carved over the trench = {n_ground} (want > 0); "
        f"the hit's own floor cell = {floor_cell:.3f} (want 1.0)",
    )
    out["trench_deep"] = {"ground_carved": n_ground, "floor_cell": floor_cell}

    # A trench shallower than the guard is deliberately NOT cleared: the guard is what stops a
    # grazing ground ray eroding the blind zone, and it cannot tell a 2-cell dip from that.
    st = fresh()
    g = carve(st, [(1.6, 0.0, z_shallow)])
    n_shallow = int((g[:, :, st.z_index(z_ground)] < 1.0).sum())
    _check(
        "a trench shallower than the guard is left alone",
        n_shallow == 0,
        f"ground-level cells carved over a 2-cell dip = {n_shallow} (want 0, guard is 2.5 cells)",
    )
    out["trench_shallow"] = {"ground_carved": n_shallow}

    # (c) a cell that is carved AND hit in the same tick ends at 1.0 --------------------------
    # The target is taken from the carve itself rather than guessed: carve once with the trench
    # ray, pick a cell it cleared, then aim a second ray at that cell's centre and run the shipped
    # order (carve, decay, scatter).
    st = fresh()
    g = carve(st, [(1.6, 0.0, z_deep)])
    hit_idx = (g < 1.0).nonzero(as_tuple=False)[0]
    centre = st._lidar_store_lo + hit_idx.float() * res
    st = fresh()
    dir_t, dst_t, val_t = rays(st, [(1.6, 0.0, z_deep), tuple(float(v) for v in centre)])
    store = st._lidar_grid_store
    st._carve_free_space(store, dst_t, val_t, off, ident, dir_t)
    after_carve = float(store[0, hit_idx[0], hit_idx[1], hit_idx[2]])
    store.mul_(0.94)
    st._scatter_hits_into_store(store, dst_t, val_t, off, ident, dir_t)
    after_hit = float(store[0, hit_idx[0], hit_idx[1], hit_idx[2]])
    _check(
        "a hit overrides a carve in the same tick",
        after_carve < 1.0 and abs(after_hit - 1.0) < 1e-6,
        f"cell after carve {after_carve:.3f} (want < 1.0), after the scatter {after_hit:.3f} (want 1.0)",
    )
    out["hit_beats_carve"] = {"after_carve": after_carve, "after_hit": after_hit}

    # (d) a discarded return carves nothing ---------------------------------------------------
    st = fresh()
    dir_t, dst_t, _ = rays(st, [(1.6, 0.0, z_deep)])
    st._carve_free_space(
        st._lidar_grid_store, dst_t, torch.zeros(1, 1, dtype=torch.bool, device=device), off, ident, dir_t
    )
    n_changed = int((st._lidar_grid_store != 1.0).sum())
    _check("an invalid return carves nothing", n_changed == 0, f"cells changed = {n_changed} (want 0)")
    out["invalid_ray"] = {"changed": n_changed}

    # (e) stride selects rays 0, k, 2k, ... deterministically ---------------------------------
    stride, n_rays = 4, 16
    per_ray = {}
    for j in range(n_rays):
        st = fresh(lidar_grid_free_space_ray_stride=stride)
        g = carve(st, [(1.6, 0.0, z_deep)], slots=[j], n_rays=n_rays)
        per_ray[j] = int((g < 1.0).sum())
    used = sorted(j for j, n in per_ray.items() if n > 0)
    want = list(range(0, n_rays, stride))
    _check(
        f"stride {stride} uses exactly rays 0, {stride}, {2 * stride}, ...",
        used == want,
        f"rays that carved = {used}, expected {want} ({len(want)} of {n_rays})",
    )
    out["stride"] = {"stride": stride, "n_rays": n_rays, "rays_used": used}
    return out


def _frozen_warp_update(u, distances, hit_valid, state):
    """The accumulator body exactly as it stood before the rolling mode was added.

    Deliberately a verbatim copy rather than a call into the env, so that a regression in the
    shipped warp path shows up as a nonzero difference instead of being mirrored by the reference.
    ``state`` carries this reference's own accumulator and pose, independent of the env's.
    """
    alpha = float(u.cfg.lidar_grid_ema_alpha)
    origin, yaw = u._grid_frame_pose()
    if getattr(u.cfg, "lidar_grid_accumulate_at_sensor_rate", True) and state["ctr"] is not None:
        state["ctr"] += 1
        tick = state["ctr"] >= u._lidar_push_every
        if not bool(tick.any()):
            return state["acc"].reshape(u.num_envs, -1)
        state["ctr"][tick] = 0
    else:
        tick = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)

    frame = u._compute_lidar_occupancy_grid(distances, hit_valid).view(u.num_envs, *u._lidar_grid_shape)
    dpos = origin - state["prev_pos"]
    c, s = torch.cos(state["prev_yaw"]), torch.sin(state["prev_yaw"])
    d = torch.stack([c * dpos[:, 0] + s * dpos[:, 1], -s * dpos[:, 0] + c * dpos[:, 1], dpos[:, 2]], dim=-1)
    dpsi = yaw - state["prev_yaw"]
    warped = u._warp_occupancy_acc(state["acc"], d, dpsi)
    blended = torch.maximum(alpha * warped, frame)
    state["acc"] = torch.where(tick.view(-1, 1, 1, 1), blended, state["acc"])
    state["prev_pos"] = torch.where(tick.view(-1, 1), origin, state["prev_pos"])
    state["prev_yaw"] = torch.where(tick, yaw, state["prev_yaw"])
    return state["acc"].reshape(u.num_envs, -1)


def test_warp_identity(n_envs: int, steps: int = 60) -> dict:
    """Run the RealSensor arm and require 0.0 difference against the frozen reference."""
    print("\n=== (g) warp mode is byte-identical to the pre-change accumulator ===")
    env_cfg = parse_env_cfg(_REAL, device=args_cli.device, num_envs=n_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(_REAL, cfg=env_cfg)
    u = env.unwrapped
    if getattr(u.cfg, "lidar_grid_accumulate_mode", "warp") != "warp":
        raise RuntimeError("the RealSensor arm must default to warp mode")
    env.reset()

    state = {
        "acc": u._lidar_grid_acc.clone(),
        "prev_pos": u._lidar_grid_prev_pos.clone(),
        "prev_yaw": u._lidar_grid_prev_yaw.clone(),
        "ctr": u._lidar_push_ctr.clone() if hasattr(u, "_lidar_push_ctr") else None,
    }
    orig = u._update_occupancy_accumulator
    rec = {"max_err": 0.0, "calls": 0}

    def hook(distances, hit_valid):
        out = orig(distances, hit_valid)
        ref = _frozen_warp_update(u, distances, hit_valid, state)
        err = float((out - ref).abs().max().item())
        rec["max_err"] = max(rec["max_err"], err)
        rec["calls"] += 1
        return out

    def reset_hook(env_ids, _orig=u._reset_idx):
        _orig(env_ids)
        if state["acc"] is None:
            return
        if env_ids is None:
            state["acc"].zero_()
            state["prev_pos"].copy_(u._lidar_grid_prev_pos)
            state["prev_yaw"].copy_(u._lidar_grid_prev_yaw)
            if state["ctr"] is not None:
                state["ctr"].copy_(u._lidar_push_ctr)
        else:
            state["acc"][env_ids] = 0.0
            state["prev_pos"][env_ids] = u._lidar_grid_prev_pos[env_ids]
            state["prev_yaw"][env_ids] = u._lidar_grid_prev_yaw[env_ids]
            if state["ctr"] is not None:
                state["ctr"][env_ids] = u._lidar_push_ctr[env_ids]

    u._update_occupancy_accumulator = hook
    u._reset_idx = reset_hook

    with torch.inference_mode():
        for _ in range(steps):
            env.step(torch.zeros(u.num_envs, u.cfg.action_space, device=u.device))
    env.close()
    _check(
        "warp accumulator equals the frozen reference",
        rec["max_err"] == 0.0,
        f"max abs difference {rec['max_err']:.3e} over {rec['calls']} accumulator calls / {steps} steps",
    )
    return {"max_err": rec["max_err"], "calls": rec["calls"], "steps": steps}


# ======================================================================
# (a) rotation invariance
# ======================================================================


def test_rotation(task: str, n_envs: int) -> dict:
    """Spin the robot in place and require the readout's world-frame terrain to hold still.

    The robot is frozen (zero root velocity, root pose rewritten every control step) and only its
    yaw advances, 30 deg over the recorded ticks.  Each tick's readout occupancy is mapped into
    world cell indices and compared with the **previous** tick's.

    Three details decide whether this measures the invariant or something else:

    * **Consecutive ticks, not tick 0.** The map accumulates, so a later tick legitimately holds
      cells an earlier one had not observed yet.  Comparing against tick 0 measures map growth.
    * **Mutual coverage only.** A rotation swings the readout box's corners over new ground, so a
      cell can be absent from one tick simply because that tick could not see it.  Both sets are
      restricted to the region inside *both* readout boxes, with a 2-cell margin.
    * **Threshold 0.3, not 0.5.** At 0.5 the comparison is dominated by the readout's own
      interpolation at cell boundaries rather than by whether the terrain moved.
    """
    print(f"\n=== (a) rotation invariance — {task} ===")
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=n_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    nx, ny, nz = u._lidar_grid_shape
    res = float(u._lidar_grid_cfg.resolution)
    lo = u._lidar_grid_lo
    hi = lo + torch.tensor([(nx - 1) * res, (ny - 1) * res, (nz - 1) * res], device=u.device)
    # Margin in x and y only.  A z margin would cut the ground sheet itself — it sits ~0.4 m
    # below the grid origin, i.e. within two cells of the box's floor — and then a millimetre
    # of body-height difference between ticks would flip hundreds of cells in and out.
    margin = torch.tensor([2.0 * res, 2.0 * res, 0.0], device=u.device)
    c = u._lidar_grid_centres
    e = 1  # env 0 is the reserved flat column in these tasks

    pos0 = u._robot.data.root_pos_w.torch.clone()
    quat0 = u._robot.data.root_quat_w.torch.clone()
    yaw0 = torch.atan2(
        2.0 * (quat0[:, 3] * quat0[:, 2] + quat0[:, 0] * quat0[:, 1]),
        1.0 - 2.0 * (quat0[:, 1] * quat0[:, 1] + quat0[:, 2] * quat0[:, 2]),
    )
    push_every = int(getattr(u, "_lidar_push_every", 5))
    n_warm = 6 * push_every  # let the map fill before anything is compared
    n_rot = 8 * push_every
    d_yaw = math.radians(30.0) / n_rot
    frames: list[dict] = []
    drift: list[float] = []
    recall: list[float] = []

    def hold(yaw_k):
        q = torch.zeros_like(quat0)
        q[:, 2] = torch.sin(yaw_k * 0.5)
        q[:, 3] = torch.cos(yaw_k * 0.5)
        ids = torch.arange(u.num_envs, device=u.device)
        u._robot.write_root_pose_to_sim_index(root_pose=torch.cat([pos0, q], dim=-1), env_ids=ids)
        u._robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(u.num_envs, 6, device=u.device), env_ids=ids
        )

    with torch.inference_mode():
        for k in range(n_warm + n_rot + 1):
            yaw_k = yaw0 if k < n_warm else yaw0 + d_yaw * (k - n_warm)
            hold(yaw_k)
            env.step(torch.zeros(u.num_envs, u.cfg.action_space, device=u.device))
            if k < n_warm or (k - n_warm) % push_every != 0:
                continue
            origin, yaw_now = u._grid_frame_pose()
            drift.append(float((u._robot.data.root_pos_w.torch[e] - pos0[e]).norm()))
            # Scatter/readout consistency: the fresh scan must be readable where it was written.
            # A sign or axis error in either rotation leaves the accumulator plausible-looking but
            # puts the scan somewhere else, and nothing else in this test would notice.
            fresh = u._compute_lidar_occupancy_grid(u._mid360._data.distances, u._lidar_grid_last_hit_valid)
            fresh = fresh[e].view(nx, ny, nz) > 0.5
            if bool(fresh.any()):
                recall.append(float((u._lidar_grid_last[e].view(nx, ny, nz)[fresh] > 0.5).float().mean()))
            grid = u._lidar_grid_last[e].view(nx, ny, nz)
            occ = (grid > 0.3).nonzero(as_tuple=False)
            local = c[occ[:, 0], occ[:, 1], occ[:, 2]] if occ.shape[0] else torch.zeros(0, 3, device=u.device)
            cy, sy = torch.cos(yaw_now[e]), torch.sin(yaw_now[e])
            w = torch.stack(
                [
                    cy * local[:, 0] - sy * local[:, 1] + origin[e, 0],
                    sy * local[:, 0] + cy * local[:, 1] + origin[e, 1],
                    local[:, 2] + origin[e, 2],
                ],
                dim=-1,
            )
            frames.append(
                {
                    "w": w.clone(),
                    "origin": origin[e].clone(),
                    "yaw": float(yaw_now[e]),
                    # Value retention, the axis on which the two modes genuinely differ: the warp
                    # path dilutes what it transports, the rolling store does not.  Positional
                    # stability (the IoU above) is not that axis — trilinear warping is position
                    # accurate, it just loses amplitude.
                    "mean_val": float(grid.mean()),
                    "frac_gt_half": float((grid > 0.5).float().mean()),
                }
            )

    env.close()

    def inside(w: torch.Tensor, fr: dict) -> torch.Tensor:
        """Mask of world points that fall inside ``fr``'s readout box, minus a 2-cell margin."""
        rel = w - fr["origin"]
        cy, sy = math.cos(-fr["yaw"]), math.sin(-fr["yaw"])
        bx = cy * rel[:, 0] - sy * rel[:, 1]
        by = sy * rel[:, 0] + cy * rel[:, 1]
        bz = rel[:, 2]
        b = torch.stack([bx, by, bz], dim=-1)
        return ((b >= lo + margin) & (b <= hi - margin)).all(dim=-1)

    def key(w: torch.Tensor) -> set:
        idx = torch.round(w / res).to(torch.int32).cpu().numpy()
        return {(int(r[0]), int(r[1]), int(r[2])) for r in idx}

    _check(
        f"fresh scan is readable where it was written ({task.split('-RealSensor-')[-1]})",
        len(recall) > 0 and min(recall) > 0.9,
        f"min per-tick recall of the current frame in the readout "
        f"{min(recall) if recall else float('nan'):.3f} over {len(recall)} ticks",
    )

    _NB = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    def agree(sa: set, sb: set) -> float:
        """Fraction of cells that have a counterpart within one cell, symmetrised.

        A strict set intersection is not the right test here and would fail on a perfect
        implementation: the readout's cell centres are placed by the robot's own pose, which is
        not on the world lattice, so the same physical surface rounds to a different world index
        when the origin moves by a fraction of a cell.  One cell of tolerance removes that
        quantisation without hiding real motion — 30 deg of yaw would move terrain at 1.5 m by
        7.8 cells.
        """
        db = {(b[0] + o[0], b[1] + o[1], b[2] + o[2]) for b in sb for o in _NB}
        da = {(a[0] + o[0], a[1] + o[1], a[2] + o[2]) for a in sa for o in _NB}
        hit = len([a for a in sa if a in db]) + len([b for b in sb if b in da])
        return hit / max(1, len(sa) + len(sb))

    ious, tol, sizes = [], [], []
    for k in range(1, len(frames)):
        a, b = frames[k - 1], frames[k]
        sa = key(a["w"][inside(a["w"], b)])
        sb = key(b["w"][inside(b["w"], a)])
        if not (sa or sb):
            continue
        ious.append(len(sa & sb) / max(1, len(sa | sb)))
        tol.append(agree(sa, sb))
        sizes.append((len(sa), len(sb)))
    swept = math.degrees(frames[-1]["yaw"] - frames[0]["yaw"]) if frames else 0.0
    _check(
        f"world-frame terrain holds still under pure yaw ({task.split('-RealSensor-')[-1]})",
        len(tol) > 0 and min(tol) > 0.9,
        f"min agreement {min(tol) if tol else float('nan'):.3f} mean {np.mean(tol) if tol else float('nan'):.3f} "
        f"(strict IoU min {min(ious) if ious else float('nan'):.3f}) over {len(tol)} consecutive tick "
        f"pairs, yaw swept {swept:.1f} deg, root drift <= {max(drift) if drift else float('nan'):.3f} m",
    )
    return {
        "strict_iou": ious,
        "tolerant_agreement": tol,
        "min_tolerant": float(min(tol)) if tol else float("nan"),
        "mean_tolerant": float(np.mean(tol)) if tol else float("nan"),
        "min_strict_iou": float(min(ious)) if ious else float("nan"),
        "mean_strict_iou": float(np.mean(ious)) if ious else float("nan"),
        "yaw_swept_deg": swept,
        "max_root_drift_m": float(max(drift)) if drift else float("nan"),
        "min_fresh_scan_recall": float(min(recall)) if recall else float("nan"),
        "set_sizes": sizes,
        "mean_val_per_tick": [f["mean_val"] for f in frames],
        "frac_gt_half_per_tick": [f["frac_gt_half"] for f in frames],
    }


# ======================================================================
# (d) perf
# ======================================================================


def test_perf(n_envs_list: list[int], n_steps: int) -> dict:
    """Time the accumulator per CONTROL step, in a realistically desynced state.

    Reporting per sensor tick understates the cost.  ``_reset_idx`` re-phases each env's
    ``_lidar_push_ctr``, so after a few hundred steps the envs no longer tick together: some
    subset ticks on essentially every control step, and the accumulator therefore does work on
    every step even though each individual env only updates once per 0.1 s.  This runs the env
    long enough for that desync to develop, then times every call and divides by control steps.

    Args:
        n_envs_list: Environment counts to measure.
        n_steps: Timed control steps per configuration.

    Returns:
        Per-configuration ms per control step, the tick statistics that explain it, and peak
        allocated GPU memory.  This is a measurement, but a configuration that raises for any
        reason other than running out of GPU memory is recorded as a failure.
    """
    print("\n=== (d) perf: ms per CONTROL step ===")
    out: dict = {}
    warm = 320  # long enough for reset-driven tick desync to develop
    for n_envs in n_envs_list:
        for tag, task, fs_stride in (
            ("warp", _REAL, None),
            ("rolling", _ROLL_NF, None),
            ("rolling+free s1", _ROLL, 1),
            ("rolling+free s4", _ROLL, 4),
        ):
            key = f"{tag}@{n_envs}"
            env = None
            try:
                env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=n_envs)
                env_cfg.seed = args_cli.seed
                if fs_stride is not None:
                    env_cfg.lidar_grid_free_space_ray_stride = int(fs_stride)
                env = gym.make(task, cfg=env_cfg)
                u = env.unwrapped
                env.reset()
                push_every = int(getattr(u, "_lidar_push_every", 1))
                orig = u._update_occupancy_accumulator
                rec: dict = {"ms": [], "tick_frac": [], "steps": 0}

                def hook(distances, hit_valid, _o=orig, _r=rec, _u=u, _pe=push_every):
                    ctr = getattr(_u, "_lidar_push_ctr", None)
                    frac = float(((ctr + 1) >= _pe).float().mean()) if ctr is not None else 1.0
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    res = _o(distances, hit_valid)
                    torch.cuda.synchronize()
                    _r["ms"].append((time.perf_counter() - t0) * 1e3)
                    _r["tick_frac"].append(frac)
                    _r["steps"] += 1
                    return res

                u._update_occupancy_accumulator = hook
                act = torch.zeros(n_envs, u.cfg.action_space, device=u.device)
                with torch.inference_mode():
                    for _ in range(warm):
                        env.step(act)
                    rec["ms"].clear()
                    rec["tick_frac"].clear()
                    rec["steps"] = 0
                    base_mem = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    for _ in range(n_steps):
                        env.step(act)
                    peak = torch.cuda.max_memory_allocated()
                ms = sorted(rec["ms"])
                tf = np.asarray(rec["tick_frac"])
                out[key] = {
                    "ms_per_control_step_mean": float(np.mean(rec["ms"])),
                    "ms_per_control_step_median": ms[len(ms) // 2],
                    "ms_per_control_step_p90": ms[int(0.9 * (len(ms) - 1))],
                    "mean_frac_envs_ticking": float(tf.mean()),
                    "frac_steps_with_any_tick": float((tf > 0).mean()),
                    "n_steps": rec["steps"],
                    "peak_alloc_gib": peak / 2**30,
                    "peak_over_baseline_gib": (peak - base_mem) / 2**30,
                    "fs_stride": fs_stride,
                }
                print(
                    f"  {tag:16s} @{n_envs:5d}: {out[key]['ms_per_control_step_mean']:6.2f} ms/step "
                    f"(median {out[key]['ms_per_control_step_median']:6.2f}, p90 "
                    f"{out[key]['ms_per_control_step_p90']:6.2f}) | envs ticking per step "
                    f"{out[key]['mean_frac_envs_ticking']:.3f}, steps with a tick "
                    f"{out[key]['frac_steps_with_any_tick']:.3f} | peak +"
                    f"{out[key]['peak_over_baseline_gib']:.2f} GiB"
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                out[key] = {"error": msg}
                oom = "out of memory" in msg.lower() or "CUDA" in msg
                print(f"  {tag:16s} @{n_envs:5d}: FAILED {msg}")
                _check(
                    f"perf configuration {key} runs",
                    oom,
                    "skipped: the GPU ran out of memory, which is an environment limit" if oom else f"raised {msg}",
                )
            finally:
                if env is not None:
                    env.close()
                torch.cuda.empty_cache()
    return out


# ======================================================================
# (c) blind-zone fill and gap-trench false occupancy, in sim
# ======================================================================


def _isolate_terrain(env_cfg, terrain_name: str) -> None:
    """Give the terrain generator to one sub-terrain, keeping exactly one flat column.

    Identical to ``render_acc_fix/play_acc_fix.py``'s helper of the same name: the RandomGoal env
    family asserts at least one ``parkour_flat`` column exists, so flat is pinned to
    ``1 / num_cols`` rather than zero.
    """
    tg = env_cfg.terrain.terrain_generator
    key = terrain_name if terrain_name in tg.sub_terrains else f"parkour_{terrain_name}"
    if key not in tg.sub_terrains:
        raise ValueError(f"unknown terrain '{terrain_name}'; have {list(tg.sub_terrains)}")
    for k in list(tg.sub_terrains.keys()):
        tg.sub_terrains[k].proportion = 0.0
    if key == "parkour_flat":
        tg.sub_terrains["parkour_flat"].proportion = 1.0
    else:
        flat = 1.0 / float(tg.num_cols)
        tg.sub_terrains["parkour_flat"].proportion = flat
        tg.sub_terrains[key].proportion = 1.0 - flat


def _scanner_truth(u, hs, lo, res, nx, ny):
    """Per-env dense surface truth from the height scanner, in each env's own grid frame.

    Batched over every env rather than one tracked env.  Scoring a single env made the numbers a
    lottery: two runs of the same config landed 2.3x apart on trench false occupancy, and one
    hurdle run had the tracked robot standing still for all 200 frames.  Averaging over every
    non-flat env replaces that with a population the live accumulator and every shadow see
    identically.

    Args:
        u: The unwrapped env.
        hs: The height scanner sensor.
        lo: ``(3,)`` grid lower corner [m].
        res: Cell size [m].
        nx: Grid cells along x.
        ny: Grid cells along y.

    Returns:
        ``(surf, covered)`` — ``(N, nx, ny)`` fractional z index of the returned surface (NaN where
        no return) and the bool mask of columns that did return.
    """
    origin, yaw = u._grid_frame_pose()
    hw = hs.data.ray_hits_w
    hw = hw.torch if hasattr(hw, "torch") else hw  # (N, R, 3) world
    rel = hw - origin.unsqueeze(1)
    cy = torch.cos(-yaw).unsqueeze(1)
    sy = torch.sin(-yaw).unsqueeze(1)
    lx = cy * rel[..., 0] - sy * rel[..., 1]
    ly = sy * rel[..., 0] + cy * rel[..., 1]
    lz = rel[..., 2]
    fin = torch.isfinite(hw).all(dim=-1)
    xi = torch.round((lx - lo[0]) / res).long().clamp(0, nx - 1)
    yi = torch.round((ly - lo[1]) / res).long().clamp(0, ny - 1)
    flat = xi * ny + yi
    n_col = nx * ny
    # One spare column absorbs the rays that returned nothing, so no per-env masking is needed.
    surf = torch.full((u.num_envs, n_col + 1), float("nan"), device=u.device)
    surf.scatter_(1, torch.where(fin, flat, torch.full_like(flat, n_col)), (lz - lo[2]) / res)
    surf = surf[:, :n_col].view(u.num_envs, nx, ny)
    return surf, ~torch.isnan(surf)


def _grid_rows_batch(vol, win, zg, hole, covered) -> dict:
    """Score one accumulator for every env at once.

    Metric definitions are those of
    ``reports/presentation/05_parkour_learning/assets/render_acc_fix/play_acc_fix.py`` (its
    ``near`` closure and its scanner-truth block), re-derived here rather than imported: that
    script builds its parser and launches the app at module scope and hard-requires a warm RTX
    renderer it needs only for the PNG panels.  Kept verbatim from it: hole columns are
    ``footprint & ~covered`` with the footprint taken from the scanner's static ``ray_starts`` so
    an aimed-at column that returned nothing still counts; the ground index is the rounded median
    scanner surface over the whole footprint; near-zone statistics are the column max over ``x``
    in [0, 0.8] m at both ``|y|`` windows.

    Args:
        vol: ``(E, nx, ny, nz)`` occupancy.
        win: ``(ix0, ix1, iy0, iy1, jy0, jy1)`` near-zone index windows.
        zg: ``(E,)`` per-env ground z index.
        hole: ``(E, nx, ny)`` bool, aimed-at columns with no return.
        covered: ``(E, nx, ny)`` bool, columns that returned.

    Returns:
        Per-env tensors for each metric; hole metrics are NaN for envs with no hole in view.
    """
    ix0, ix1, iy0, iy1, jy0, jy1 = win
    e_n, nx, ny, nz = vol.shape
    cm = vol[:, ix0:ix1].amax(dim=3)  # (E, dx, ny)
    out = {
        "near_mean": cm[:, :, iy0:iy1].mean(dim=(1, 2)),
        "near_f5": (cm[:, :, iy0:iy1] > 0.5).float().mean(dim=(1, 2)),
        "near_mean_alt": cm[:, :, jy0:jy1].mean(dim=(1, 2)),
        "near_f5_alt": (cm[:, :, jy0:jy1] > 0.5).float().mean(dim=(1, 2)),
    }
    gslice = torch.gather(vol, 3, zg.view(e_n, 1, 1, 1).expand(e_n, nx, ny, 1)).squeeze(3)  # (E, nx, ny)
    n_h = hole.sum(dim=(1, 2)).float()
    ground = covered & ~hole
    n_g = ground.sum(dim=(1, 2)).float()
    nan = torch.tensor(float("nan"), device=vol.device)
    for thr in (0.5, 0.3):
        tg = f"{thr:g}".replace(".", "")
        lit = (gslice > thr).float()
        out[f"holebridge_{tg}"] = torch.where(n_h > 0, (lit * hole).sum(dim=(1, 2)) / n_h.clamp(min=1), nan)
        out[f"groundkeep_{tg}"] = torch.where(n_g > 0, (lit * ground).sum(dim=(1, 2)) / n_g.clamp(min=1), nan)
    return out


class _CfgShim:
    """Read-through view of an env cfg with a few fields overridden."""

    def __init__(self, base, over: dict):
        self._base = base
        for k, v in over.items():
            setattr(self, k, v)

    def __getattr__(self, key):
        return getattr(self._base, key)


class _RollShadow:
    """A second rolling store driven by the live env's frames, under its own config.

    This is the comparison the first round could not make.  Switching the env itself to a rolling
    accumulator changes the observation, so the 20k checkpoint acts differently and the
    trajectories diverge — on hurdle it stopped walking entirely.  Here the env stays in its
    default warp mode, so the policy is on-distribution, and every rolling variant is updated from
    the *same* scan, the same pose and the same tick schedule.  Differences between rows are then
    the accumulator and nothing else.

    The shipped code does the work: the store methods are rebound to this object, so this runs
    ``Go2ParkourImitationRandomGoalLidarEnv._update_occupancy_accumulator_rolling`` verbatim rather
    than a copy of it.  Everything the update reads but does not own — the robot, the sensor, the
    grid geometry, ``_grid_frame_pose`` — is delegated to the env.
    """

    _OWN = (
        "_setup_rolling_store",
        "_snap_origin",
        "_scroll_store",
        "_scatter_hits_into_store",
        "_carve_free_space",
        "_readout_store",
        "_update_occupancy_accumulator_rolling",
    )

    def __init__(self, env, name: str, over: dict):
        object.__setattr__(self, "_env", env)
        self.name = name
        self.cfg = _CfgShim(env.cfg, {"lidar_grid_accumulate_mode": "rolling", **over})
        for meth in self._OWN:
            setattr(self, meth, types.MethodType(getattr(_Env, meth), self))
        self._setup_rolling_store(self.cfg)
        self._lidar_push_ctr = env._lidar_push_ctr.clone()
        self._lidar_grid_prev_pos = env._lidar_grid_prev_pos.clone()
        self._lidar_grid_prev_yaw = env._lidar_grid_prev_yaw.clone()
        # Past every milestone, so the shipped update never takes its diagnostic branch.
        self._lidar_grid_diag_next = 1 << 30

    def __getattr__(self, key):
        return getattr(self._env, key)

    def sync_ctr(self, ctr_prev: torch.Tensor) -> None:
        """Adopt the env's pre-update tick counter so this shadow ticks on the same steps."""
        self._lidar_push_ctr.copy_(ctr_prev)

    def reset(self, env_ids) -> None:
        """Mirror ``_reset_idx``: drop the map, re-snap, clear the held readout."""
        origin, yaw = self._env._grid_frame_pose()
        snap = self._snap_origin(origin)
        if env_ids is None:
            self._lidar_grid_store.zero_()
            self._lidar_store_snap.copy_(snap)
            self._lidar_store_readout.zero_()
            self._lidar_grid_prev_pos.copy_(origin)
            self._lidar_grid_prev_yaw.copy_(yaw)
        else:
            self._lidar_grid_store[env_ids] = 0.0
            self._lidar_store_snap[env_ids] = snap[env_ids]
            self._lidar_store_readout[env_ids] = 0.0
            self._lidar_grid_prev_pos[env_ids] = origin[env_ids]
            self._lidar_grid_prev_yaw[env_ids] = yaw[env_ids]


_AH = {"lidar_grid_free_space": True, "lidar_grid_free_space_rule": "above_hit"}


def _ah(guard: float, min_rays: int = 1, stride: int = 1, debug: bool = False) -> dict:
    """Config overrides for an ``above_hit`` variant."""
    return _AH | {
        "lidar_grid_free_space_above_hit_cells": guard,
        "lidar_grid_free_space_min_rays": min_rays,
        "lidar_grid_free_space_ray_stride": stride,
        "lidar_grid_free_space_debug_count": debug,
    }


SHADOW_VARIANTS: list[tuple[str, dict]] = [
    ("rollnofree", {"lidar_grid_free_space": False}),
    (
        "roll_margin2",
        {
            "lidar_grid_free_space": True,
            "lidar_grid_free_space_rule": "margin",
            "lidar_grid_free_space_hit_margin_cells": 2,
        },
    ),
    ("ah_g0.5", _ah(0.5, debug=True)),
    ("ah_g1.5", _ah(1.5)),
    ("ah_g2.5", _ah(2.5)),
    ("ah_g3.5", _ah(3.5)),
    ("ah_g1.5_min4", _ah(1.5, min_rays=4)),
    ("ah_g2.5_min4", _ah(2.5, min_rays=4)),
    ("ah_g2.5_s2", _ah(2.5, stride=2)),
    ("ah_g2.5_s4", _ah(2.5, stride=4)),
]


def test_shadow_table(terrain: str, checkpoint: str, n_envs: int, warmup: int, steps: int) -> dict:
    """Score every rolling variant against the live warp accumulator on identical frames.

    The env runs the unmodified RealSensor arm, so the checkpoint sees the observation it was
    trained on and walks normally.  Each variant in :data:`SHADOW_VARIANTS` maintains its own store
    from the same scans, and every non-flat env is scored on every recorded frame.

    This is a measurement, not an assertion: it makes no ``_check`` call and cannot fail.

    Args:
        terrain: Sub-terrain to isolate (``gap``, ``hurdle``, ...).
        checkpoint: Policy checkpoint to roll out.
        n_envs: Environment count.
        warmup: Control steps before recording starts.
        steps: Recorded control steps.

    Returns:
        Per-variant means of the near-zone, hole-bridging and ground-retention metrics, plus the
        population's kinematics, the per-tick ray-traversal counts and the variant configs.
    """
    print(f"\n=== (c) same-frame shadow table — {terrain} ===")
    import inspect

    from rsl_rl.algorithms.ppo_parkour import PPOParkour
    from rsl_rl.runners.on_policy_runner_parkour_amp import OnPolicyRunnerParkourAMPVoxel

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry

    task = _REAL
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=n_envs)
    env_cfg.seed = args_cli.seed
    _isolate_terrain(env_cfg, terrain)
    if hasattr(env_cfg, "terrain_max_init_level"):
        env_cfg.terrain_max_init_level = args_cli.pin_level
    if hasattr(env_cfg.terrain, "max_init_terrain_level"):
        env_cfg.terrain.max_init_terrain_level = args_cli.pin_level
    if hasattr(env_cfg, "enable_random_goal"):
        env_cfg.enable_random_goal = False
    env_cfg.debug_vis = False
    env_cfg.log_dir = os.path.dirname(os.path.abspath(checkpoint))

    env = gym.make(task, cfg=env_cfg)
    u = env.unwrapped
    u.set_debug_vis(False)
    if getattr(u.cfg, "lidar_grid_accumulate_mode", "warp") != "warp":
        raise RuntimeError("the shadow table requires the env itself to stay in warp mode")

    pin = int(args_cli.pin_level)
    u._terrain_levels[:] = pin
    u._terrain.env_origins[:] = u._terrain.terrain_origins[u._terrain_levels, u._terrain_types]

    def _pinned(env_ids, _u=u, _l=pin):
        _u._skip_curriculum[env_ids] = False
        _u._gap_diag = {}
        _u._terrain_levels[env_ids] = _l
        _u._terrain.env_origins[env_ids] = _u._terrain.terrain_origins[_l, _u._terrain_types[env_ids]]

    u._update_terrain_curriculum = _pinned
    env.reset()

    nx, ny, nz = u._lidar_grid_shape
    res = float(u._lidar_grid_cfg.resolution)
    lo = u._lidar_grid_lo
    xs = lo[0].item() + np.arange(nx) * res
    ys = lo[1].item() + np.arange(ny) * res
    ixs = np.nonzero((xs >= -1e-6) & (xs <= 0.8 + 1e-6))[0]
    ix0, ix1 = int(ixs[0]), int(ixs[-1]) + 1

    def y_window(half):
        iy = np.nonzero(np.abs(ys) <= half + 1e-6)[0]
        return int(iy[0]), int(iy[-1]) + 1

    iy0, iy1 = y_window(0.5)
    jy0, jy1 = y_window(0.3)
    win = (ix0, ix1, iy0, iy1, jy0, jy1)

    hs = u.scene["height_scanner"]
    rs = hs.ray_starts
    rs = rs.torch if hasattr(rs, "torch") else rs
    rs = rs[0] if rs.dim() == 3 else rs
    fx = torch.round((rs[:, 0] - lo[0]) / res).long().clamp(0, nx - 1)
    fy = torch.round((rs[:, 1] - lo[1]) / res).long().clamp(0, ny - 1)
    footprint = torch.zeros(nx, ny, dtype=torch.bool, device=u.device)
    footprint[fx, fy] = True
    n_rays_hs = int(hs.data.ray_hits_w.shape[1])
    if int(footprint.sum()) != n_rays_hs:
        raise RuntimeError(
            f"footprint has {int(footprint.sum())} columns for {n_rays_hs} rays — a missing return "
            "could not be told from a column no ray was aimed at."
        )

    resume = os.path.abspath(checkpoint)
    if not os.path.isfile(resume):
        raise FileNotFoundError(resume)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = str(u.device)
    agent_cfg.seed = args_cli.seed
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    cfg_dict = agent_cfg.to_dict()
    accepted = set(inspect.signature(PPOParkour.__init__).parameters) - {"self"}
    cfg_dict["algorithm"] = {k: v for k, v in cfg_dict["algorithm"].items() if k in accepted or k == "class_name"}
    runner = OnPolicyRunnerParkourAMPVoxel(wrapped, cfg_dict, log_dir=None, device=str(u.device))
    runner.load(resume)
    policy = runner.get_inference_policy(device=u.device)
    policy_nn = runner.alg.policy

    # Allocated OUTSIDE inference mode on purpose.  The first observation is fetched outside it
    # (``get_observations`` before the loop) and that call already drives every shadow, so a store
    # allocated inside inference mode would be an inference tensor the outside call cannot update
    # in place.  From the loop onward every write happens inside inference mode, which is allowed.
    shadows = [_RollShadow(u, name, over) for name, over in SHADOW_VARIANTS]
    print(f"  shadows: {[s.name for s in shadows]}")
    print(f"  free-space samples per ray = {shadows[-1]._lidar_free_max_samples} (readout-box bound)")

    orig_update = u._update_occupancy_accumulator
    orig_reset = u._reset_idx
    timing: dict[str, list[float]] = {s.name: [] for s in shadows}

    def upd(distances, hit_valid):
        ctr_prev = u._lidar_push_ctr.clone() if hasattr(u, "_lidar_push_ctr") else None
        out = orig_update(distances, hit_valid)
        for sh in shadows:
            if ctr_prev is not None:
                sh.sync_ctr(ctr_prev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sh._update_occupancy_accumulator_rolling(distances, hit_valid)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if dt > 0.5:
                timing[sh.name].append(dt)
        return out

    def rst(env_ids, _o=orig_reset):
        _o(env_ids)
        for sh in shadows:
            sh.reset(env_ids)

    u._update_occupancy_accumulator = upd
    u._reset_idx = rst

    # Every env on the target sub-terrain, not one tracked env.
    keep = (u._env_class != int(u._col_to_class[0].item())).clone()
    if not bool(keep.any()):
        keep = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
    print(f"  scoring {int(keep.sum())}/{u.num_envs} envs (flat column excluded)")
    rows: dict[str, list[np.ndarray]] = {"env_warp": []} | {s.name: [] for s in shadows}
    kin: list[np.ndarray] = []
    traversals: list[dict] = []
    dbg = next((sh for sh in shadows if getattr(sh.cfg, "lidar_grid_free_space_debug_count", False)), None)
    obs = wrapped.get_observations()
    just_reset = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
    keys: list[str] = []

    with torch.inference_mode():
        for t in range(warmup + steps):
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            policy_nn.reset(dones)
            fresh = just_reset.clone()
            just_reset = dones.to(torch.bool).clone()
            if t < warmup:
                continue
            origin, yaw = u._grid_frame_pose()
            surf, covered = _scanner_truth(u, hs, lo, res, nx, ny)
            hole = footprint.unsqueeze(0) & ~covered
            med = torch.nanmedian(surf.view(u.num_envs, -1), dim=1).values
            zg = torch.nan_to_num(med, nan=0.0).round().long().clamp(0, nz - 1)
            # Drop envs that reset into this frame (an emptied accumulator is empty by definition)
            # and envs whose scanner returned nothing at all.
            use = keep & ~fresh & covered.any(dim=1).any(dim=1)
            if not bool(use.any()):
                continue
            ui = use.nonzero(as_tuple=False).flatten()
            _lb = u._robot.data.root_lin_vel_b
            _lb = _lb.torch if hasattr(_lb, "torch") else _lb
            kin.append(
                np.stack(
                    [
                        _lb[ui, 0].cpu().numpy(),
                        u._robot.data.root_pos_w.torch[ui, 2].cpu().numpy(),
                        hole[ui].sum(dim=(1, 2)).float().cpu().numpy(),
                    ],
                    axis=1,
                )
            )
            batch = {"env_warp": u._lidar_grid_last[ui].view(-1, nx, ny, nz)}
            batch.update({sh.name: sh._lidar_store_readout[ui].view(-1, nx, ny, nz) for sh in shadows})
            for tag, vol in batch.items():
                r = _grid_rows_batch(vol, win, zg[ui], hole[ui], covered[ui])
                if not keys:
                    keys = list(r)
                rows[tag].append(np.stack([r[k].cpu().numpy() for k in keys], axis=1))
            cgrid = getattr(dbg, "_lidar_free_last_count", None)
            if cgrid is not None:
                # How many rays actually cross a trench cell against a true-ground cell, in the
                # same frame and z slice.  This decides whether a traversal threshold can separate
                # the two cases at all.
                _saved = dbg._lidar_grid_store
                dbg._lidar_grid_store = cgrid
                cro = dbg._readout_store(origin, yaw).view(u.num_envs, nx, ny, nz)
                dbg._lidar_grid_store = _saved
                cs = torch.gather(cro, 3, zg.view(-1, 1, 1, 1).expand(-1, nx, ny, 1)).squeeze(3)[ui]
                h_u, g_u = hole[ui], (covered & ~hole)[ui]
                nh = h_u.sum(dim=(1, 2)).float()
                traversals.append(
                    {
                        "hole": float(((cs * h_u).sum(dim=(1, 2)) / nh.clamp(min=1))[nh > 0].mean())
                        if bool((nh > 0).any())
                        else float("nan"),
                        "ground": float(((cs * g_u).sum(dim=(1, 2)) / g_u.sum(dim=(1, 2)).clamp(min=1)).mean()),
                    }
                )

    env.close()
    out: dict = {}
    for tag, rs_ in rows.items():
        if not rs_:
            out[tag] = {"n_samples": 0}
            continue
        arr = np.concatenate(rs_, axis=0)  # (env-frames, metrics)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # envs with no hole in view
            out[tag] = {k: float(np.nanmean(arr[:, i])) for i, k in enumerate(keys)}
        out[tag]["n_samples"] = int(arr.shape[0])
        if tag in timing and timing[tag]:
            ts = sorted(timing[tag])
            out[tag][f"ms_per_tick_median_{n_envs}env"] = ts[len(ts) // 2]
    kin_a = np.concatenate(kin, axis=0) if kin else np.zeros((0, 3))
    out["_kinematics"] = {
        "mean_vx": float(kin_a[:, 0].mean()) if kin_a.size else float("nan"),
        "frac_vx_below_0p2": float((np.abs(kin_a[:, 0]) < 0.2).mean()) if kin_a.size else float("nan"),
        "mean_base_z": float(kin_a[:, 1].mean()) if kin_a.size else float("nan"),
        "mean_hole_cols": float(kin_a[:, 2].mean()) if kin_a.size else float("nan"),
        "frac_samples_with_hole": float((kin_a[:, 2] > 0).mean()) if kin_a.size else float("nan"),
        "n_env_frames": int(kin_a.shape[0]),
    }
    if traversals:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out["_traversals_per_tick"] = {k: float(np.nanmean([t[k] for t in traversals])) for k in ("hole", "ground")}
        print(f"  rays crossing a cell per tick (ground slice): {out['_traversals_per_tick']}")
    out["_variants"] = dict(SHADOW_VARIANTS)
    # The table is a measurement, not a rule, but an empty one must not read as a pass.
    n_per = {k: out[k].get("n_samples", 0) for k in list(out) if not k.startswith("_")}
    _check(
        f"shadow table on {terrain} produced samples for every variant",
        all(out[t].get("n_samples", 0) > 0 for t in ["env_warp"] + [sh.name for sh in shadows]),
        f"env-frames per variant = {n_per}",
    )
    print(f"  kinematics: {out['_kinematics']}")
    print(f"  {'variant':16s} {'near|y|.3':>10s} {'near>.5':>8s} {'hole>.5':>8s} {'hole>.3':>8s} {'grnd>.5':>8s}")
    for tag in ["env_warp"] + [s.name for s in shadows]:
        r = out[tag]
        if not r.get("n_samples"):
            continue
        print(
            f"  {tag:16s} {r['near_mean_alt']:10.3f} {r['near_f5_alt']:8.3f} "
            f"{r['holebridge_05']:8.3f} {r['holebridge_03']:8.3f} {r['groundkeep_05']:8.3f}"
        )
    return out


# ======================================================================


_KNOWN_TESTS = ("synth", "carving", "warp_identity", "rotation", "perf", "shadow")


def main() -> None:
    tests = [t.strip() for t in args_cli.test.split(",") if t.strip()]
    unknown = [t for t in tests if t not in _KNOWN_TESTS]
    if unknown:
        raise SystemExit(f"unknown --test name(s) {unknown}; known: {list(_KNOWN_TESTS)}")
    if "synth" in tests:
        RESULTS["synth"] = test_synth(args_cli.device or "cuda:0")
    if "carving" in tests:
        RESULTS["carving"] = test_carving(args_cli.device or "cuda:0")
    if "warp_identity" in tests:
        RESULTS["warp_identity"] = test_warp_identity(args_cli.num_envs)
    if "rotation" in tests:
        RESULTS["rotation"] = {t: test_rotation(t, args_cli.num_envs) for t in (args_cli.task.split(",")[0], _REAL)}
    if "perf" in tests:
        RESULTS["perf"] = test_perf(args_cli.perf_envs, args_cli.perf_steps)
    if "shadow" in tests:
        if not args_cli.checkpoint:
            raise SystemExit("--checkpoint is required for the shadow sub-test")
        RESULTS["shadow"] = {}
        for tr in [t.strip() for t in args_cli.terrain.split(",") if t.strip()]:
            RESULTS["shadow"][tr] = test_shadow_table(
                tr, args_cli.checkpoint, args_cli.num_envs, args_cli.warmup, args_cli.steps
            )
            if args_cli.out_json:
                with open(args_cli.out_json, "w") as f:
                    json.dump(RESULTS, f, indent=2, default=float)
    if args_cli.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out_json)), exist_ok=True)
        with open(args_cli.out_json, "w") as f:
            json.dump(RESULTS, f, indent=2, default=float)
        print(f"\nwrote {args_cli.out_json}")
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all assertions passed")


if __name__ == "__main__":
    main()
    simulation_app.close()
