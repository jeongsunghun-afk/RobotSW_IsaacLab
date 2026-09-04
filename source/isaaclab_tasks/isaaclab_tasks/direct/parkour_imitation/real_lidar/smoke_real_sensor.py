# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the real-sensor Mid-360 arm.

Checks, per sensor frame, that the rolling scan actually rolls, that the blind range and rear
crop actually remove rays, and that the observation contract is unchanged.  Run the baseline
task through the same script to confirm it is untouched.

.. code-block:: bash

    source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate isaac-6.0 && \\
    cd /home/lgb/IsaacLab-6.0 && CUDA_VISIBLE_DEVICES=2 env -u DISPLAY ./isaaclab.sh -p \\
      source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/real_lidar/smoke_real_sensor.py \\
      --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_REAL = "Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-EasyEntry-v0"
_BASE = "Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=_REAL, help="Gym task id to smoke-test.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=15, help="Control steps to run.")
parser.add_argument("--timing", action="store_true", help="Also time sensor updates at --timing_envs.")
parser.add_argument("--timing_envs", type=int, default=1024, help="Env count for the timing pass.")
parser.add_argument(
    "--pitch_range_override",
    type=float,
    nargs=2,
    default=None,
    help="Set lidar_mount_pitch_range_deg after cfg construction; use 30 30 for the degenerate check.",
)
parser.add_argument(
    "--check_static_guard",
    action="store_true",
    help="Only check that a pitch range on a non-rolling sensor cfg raises, then exit.",
)
parser.add_argument(
    "--no_pitch_range",
    action="store_true",
    help="Force lidar_mount_pitch_range_deg=None, i.e. the single-rotation reference path.",
)
parser.add_argument(
    "--pitch_override",
    type=float,
    default=None,
    help="Set lidar_mount_pitch_deg AFTER cfg construction, the way hydra applies env.* overrides.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.parkour_imitation.mid360_rolling_lidar import (  # noqa: E402
    DEFAULT_SCAN_PATTERN_FILE,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def quat_apply_np(q_xyzw_order: tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` the way ``base_ray_caster`` does (the tuple is fed to an xyzw ``quat_apply``)."""
    q = np.asarray(q_xyzw_order, dtype=np.float32)
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(np.broadcast_to(xyz, v.shape), v)
    return v + w * t + np.cross(np.broadcast_to(xyz, v.shape), t)


def report_cadence_and_geometry(
    env,
    sensor,
    cfg,
    updates: list[dict],
    step_dirs0: list[torch.Tensor],
    push_steps: dict[int, list[int]],
    rolling: bool,
    max_dist: float,
    failures: list[str],
) -> None:
    """Report the sensor-update cadence and check the per-env ray geometry.

    Args:
        env: The instantiated environment.
        sensor: The Mid-360 sensor under test.
        cfg: The environment cfg.
        updates: One record per ``_update_buffers_impl`` call, from the recorder.
        step_dirs0: Env 0's ray directions snapshotted at the end of each control step.
        push_steps: Control steps at which each traced env pushed an accumulator frame.
        rolling: Whether the rolling scan is active.
        max_dist: Sensor ``max_distance`` [m].
        failures: Accumulated failure messages, appended to in place.
    """
    # ------------------------------------------------------------------
    # Sensor cadence and the per-env partial-mask path
    # ------------------------------------------------------------------
    print("-" * 78)
    N = env.num_envs
    upd_steps = [u["step"] for u in updates]
    print(f"sensor updates at control steps : {upd_steps}")
    print(
        f"  -> {len(updates)} updates over {args_cli.steps} control steps"
        f" (update_period={cfg.mid360_lidar.update_period} s, step_dt={env.step_dt} s)"
    )
    for e in (0, 2):
        in_mask = [u["step"] for u in updates if bool(u["mask"][e])]
        print(f"  env {e}: sensor-update steps {in_mask}")
        print(f"  env {e}: accumulator push steps {push_steps[e]}")

    if rolling:
        partial = [u for u in updates if 0 < int(u["mask"].sum()) < N]
        full = [u for u in updates if int(u["mask"].sum()) == N]
        print(
            f"full-mask updates: {len(full)} | partial-mask updates: {len(partial)}"
            f" {[(u['step'], int(u['mask'].sum())) for u in partial]}"
        )

        # (a) every update advances the masked envs by exactly R and leaves the rest alone;
        #     ray_directions changes for exactly the masked envs.
        for u in updates:
            adv = (u["pos_after"] - u["pos_before"]) % sensor._scan_pattern_size
            want = u["mask"].long() * sensor.num_rays
            if not bool((adv == want).all()):
                failures.append(f"update at step {u['step']}: scan advance does not match the mask")
            if not bool(torch.equal(u["changed"], u["mask"])):
                bad = (u["changed"] != u["mask"]).nonzero().flatten().tolist()
                failures.append(f"update at step {u['step']}: dirs changed != mask for envs {bad}")

        # (a, cont.) between sensor updates the buffer must be completely still.
        for i in range(1, len(step_dirs0)):
            moved = not torch.equal(step_dirs0[i], step_dirs0[i - 1])
            was_update = any(u["step"] == i and bool(u["mask"][0]) for u in updates)
            if moved != was_update:
                failures.append(f"step {i}: env0 directions moved={moved} but env0 sensor update={was_update}")

        if not partial:
            failures.append("no partial-mask update was ever observed; the per-env path is untested")
        else:
            print(
                f"partial-mask path exercised: {len(partial)} updates, "
                f"masked envs e.g. {partial[0]['mask'].nonzero().flatten().tolist()}"
            )

    # (b) geometry: every env's hits must lie along that env's OWN ray directions.
    #     NOTE: ``data.distances`` carries the injected range noise and the dropout
    #     substitution, so it is NOT the geometric distance.  The identity is checked against
    #     the raw distance recovered from the ray buffers, and the noise is reported separately.
    starts_w = wp.to_torch(sensor._ray_starts_w)  # (N, R, 3)
    dirs_w = wp.to_torch(sensor._ray_directions_w)  # (N, R, 3)
    hits_w = sensor.data.ray_hits_w  # (N, R, 3), inf where the ray missed
    finite = torch.isfinite(hits_w).all(dim=-1)
    raw_d = (hits_w - starts_w).norm(dim=-1)
    recon = starts_w + raw_d.unsqueeze(-1) * dirs_w
    geo_err = float((recon - hits_w)[finite].abs().max())
    print(f"hit reconstruction  |ray_starts_w + raw_d * dir_w - ray_hits_w|max = {geo_err:.3e} m")
    if geo_err > 1e-3:
        failures.append(f"hit reconstruction error {geo_err:.3e} m exceeds 1e-3 m")

    # Body -> world link: the world dirs the kernel produced must be this env's own body dirs
    # rotated by this env's own orientation.  A cross-env mix-up shows up here and nowhere else.
    body = sensor.ray_directions.torch
    body = body if body.dim() == 3 else body.unsqueeze(0).expand(N, -1, -1)
    q = sensor.data.quat_w.torch if hasattr(sensor.data.quat_w, "torch") else sensor.data.quat_w
    qe = q.unsqueeze(1).expand(N, sensor.num_rays, 4)
    # The warp kernel stores quat_w as wp.quatf, whose torch view is (x, y, z, w); both orders
    # are tried so the check reports the convention it found instead of assuming one.
    errs = {
        "xyzw": float((quat_apply(qe, body) - dirs_w).abs().max()),
        "wxyz": float((quat_apply(qe[..., [1, 2, 3, 0]], body) - dirs_w).abs().max()),
    }
    conv = min(errs, key=lambda k: errs[k])
    print(
        f"body->world link    |R(quat_w[e]) @ dirs[e] - ray_directions_w[e]|max = {errs[conv]:.3e}"
        f" (quat_w read as {conv}; other order gives {errs['wxyz' if conv == 'xyzw' else 'xyzw']:.3e})"
    )
    if errs[conv] > 1e-4:
        failures.append(f"per-env body->world direction mismatch {errs[conv]:.3e}")

    # data.distances carries injected noise AND the dropout substitution (dropout rays are set
    # to max_distance regardless of what they hit), so the mean is meaningless. The median of
    # the absolute difference isolates the noise; the tail is reported as a separate fraction.
    # ``LidarSensor._update_buffers_impl`` re-applies noise AND re-draws the dropout mask on
    # every ``.data`` access, so two reads in the same step are two different scans. Both the
    # validity mask and the difference must come from one snapshot, or ~pixel_dropout_prob of
    # the rays look like huge errors purely because they were dropped in only one of the draws.
    dist_snapshot = sensor.data.distances.clone()
    valid = finite & (dist_snapshot < max_dist)
    diff = (dist_snapshot - raw_d)[valid].abs()
    print(
        f"data.distances vs raw geometric distance (single snapshot): median |diff|"
        f" {float(diff.median()):.4f} m, {float((diff > 0.1).float().mean()):.3%} of rays beyond 0.1 m"
        f" (injected random_distance_noise = {cfg.mid360_lidar.random_distance_noise} m)"
    )
    if float(diff.median()) > 5.0 * cfg.mid360_lidar.random_distance_noise:
        failures.append(f"median distance noise {float(diff.median()):.4f} m is far above the configured sigma")


def check_mount_pitch(env, sensor, cfg, ref_dirs: torch.Tensor, rot, failures: list[str]) -> None:
    """Verify the per-env mount-pitch randomisation against an independent reconstruction.

    The gating comparison is exact: env ``e``'s written directions must equal its own pattern
    window rotated by its own drawn pitch, recomputed here in numpy. The elevation-span formula
    is reported alongside as an observed deviation rather than asserted on, because a 20k-row
    window covers the pattern's phi range only to ~0.3 deg and that eats most of a 0.5 deg budget.

    Args:
        env: The instantiated environment.
        sensor: The rolling Mid-360 sensor.
        cfg: The environment cfg.
        ref_dirs: The sensor's own pattern table, as written to the device.
        rot: ``cfg.mid360_lidar.offset.rot``, the single-rotation mount quaternion.
        failures: Accumulated failure messages, appended to in place.
    """
    rng = getattr(cfg.mid360_lidar, "mount_pitch_range_deg", None)
    pitches = sensor.mount_pitch_deg
    dirs = sensor.ray_directions.torch
    pos = sensor._scan_pos
    P = sensor._scan_pattern_size
    R = sensor.num_rays

    if rng is None:
        print("mount pitch randomisation: OFF (single offset.rot path)")
        if float(pitches.abs().max()) != 0.0:
            failures.append("mount_pitch_deg is nonzero although mount_pitch_range_deg is None")
        return

    lo, hi = rng
    print(
        f"mount pitch drawn: min {float(pitches.min()):.3f} / mean {float(pitches.mean()):.3f}"
        f" / max {float(pitches.max()):.3f} deg over {env.num_envs} envs (range {lo}-{hi})"
    )
    if float(pitches.min()) < lo - 1e-4 or float(pitches.max()) > hi + 1e-4:
        failures.append(f"drawn mount pitch [{float(pitches.min())}, {float(pitches.max())}] escapes {rng}")
    n_distinct = int(torch.unique(pitches).numel())
    if hi > lo and n_distinct < env.num_envs:
        failures.append(f"only {n_distinct} distinct pitches across {env.num_envs} envs")
    print(f"  distinct pitches: {n_distinct} / {env.num_envs}")

    # Exact per-env reconstruction: raw pattern window -> that env's quaternion.
    raw = np.load(cfg.mid360_lidar.scan_pattern_file or DEFAULT_SCAN_PATTERN_FILE)
    th, ph = raw[:, 0], raw[:, 1]
    base = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], 1).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    if rng[0] == rng[1]:
        # Degenerate range: the written directions must equal the pre-rotated single-rotation
        # path, i.e. the whole randomisation collapses to current behaviour.
        single = quat_apply_np(rot, base)
        worst = 0.0
        for e in range(min(4, env.num_envs)):
            idx = (int(pos[e]) + np.arange(R)) % P
            worst = max(worst, float(np.abs(dirs[e].cpu().numpy() - single[idx]).max()))
        print(f"degenerate range {rng}: max |written - single offset.rot path| = {worst:.3e}")
        if worst > 1e-6:
            failures.append(f"degenerate pitch range differs from the single-rotation path by {worst:.2e}")

    print(
        f"  {'env':>4} {'pitch':>7} {'el_min':>9} {'el_max':>9} {'|dir err|':>10}"
        f" {'d(el_max-(p+4.49))':>19} {'d(el_min+(p+47.77))':>20}"
    )
    for e in range(min(3, env.num_envs)):
        p_e = float(pitches[e])
        half = math.radians(p_e) * 0.5
        want = quat_apply_np((0.0, math.cos(half), 0.0, math.sin(half)), base)
        idx = (int(pos[e]) + np.arange(R)) % P
        err = float(np.abs(dirs[e].cpu().numpy() - want[idx]).max())
        el = torch.rad2deg(torch.atan2(dirs[e][:, 2], dirs[e][:, :2].norm(dim=-1)))
        el_min, el_max = float(el.min()), float(el.max())
        print(
            f"  {e:>4} {p_e:7.3f} {el_min:9.3f} {el_max:9.3f} {err:10.3e}"
            f" {el_max - (p_e + 4.49):19.3f} {el_min + (p_e + 47.77):20.3f}"
        )
        if err > 1e-6:
            failures.append(f"env {e}: written directions differ from its own pitch+window by {err:.2e}")


def check_reset_redraws(env, sensor, cfg, ref_dirs: torch.Tensor, rot, rolling: bool, failures: list[str]) -> None:
    """Check the mount-pitch randomisation and the per-env redraw on reset.

    Args:
        env: The instantiated environment.
        sensor: The Mid-360 sensor under test.
        cfg: The environment cfg.
        ref_dirs: The sensor's own pattern table, as written to the device.
        rot: ``cfg.mid360_lidar.offset.rot``.
        rolling: Whether the rolling scan is active.
        failures: Accumulated failure messages, appended to in place.
    """
    # Reset path: scene.reset -> sensor.reset(env_ids) re-randomises the scan phase of exactly
    # the reset envs.  Nothing in a 15-step zero-action rollout terminates, so this is driven
    # directly; it also exercises the env_ids-as-tensor branch of the index resolution.
    if rolling:
        check_mount_pitch(env, sensor, cfg, ref_dirs, rot, failures)

        pitch_before = sensor.mount_pitch_deg.clone()
        before = sensor._scan_pos.clone()
        sensor.reset(torch.tensor([0, 1], device=sensor._scan_pos.device))
        after = sensor._scan_pos
        moved = [int(i) for i in range(env.num_envs) if int(before[i]) != int(after[i])]
        print(f"reset([0, 1]) changed scan phase of envs {moved} (expected [0, 1])")
        if moved != [0, 1]:
            failures.append(f"reset([0, 1]) changed envs {moved}, expected exactly [0, 1]")

        rng = getattr(cfg.mid360_lidar, "mount_pitch_range_deg", None)
        if rng is not None:
            pa = sensor.mount_pitch_deg
            p_moved = [int(i) for i in range(env.num_envs) if float(pitch_before[i]) != float(pa[i])]
            print(
                f"reset([0, 1]) redrew mount pitch of envs {p_moved};"
                f" env0 {float(pitch_before[0]):.3f} -> {float(pa[0]):.3f} deg"
            )
            if rng[0] == rng[1]:
                # A zero-width range redraws the same number, so "changed" cannot be the test.
                # What must hold is that every pitch is still the single value in the range.
                if float(pa.min()) != rng[0] or float(pa.max()) != rng[0]:
                    failures.append(f"degenerate range: pitches left {rng}")
            elif p_moved != [0, 1]:
                failures.append(f"reset([0, 1]) redrew mount pitch for envs {p_moved}, expected [0, 1]")
            others = [int(i) for i in range(2, env.num_envs) if float(pitch_before[i]) != float(pa[i])]
            if others:
                failures.append(f"reset([0, 1]) also changed the mount pitch of envs {others}")


def build_env_cfg():
    """Build the task cfg and apply the CLI mount overrides the way hydra would.

    Hydra applies ``env.*`` overrides to the already-constructed cfg, after ``__post_init__``
    has run, so setting them here reproduces the path a command-line override actually takes.

    Returns:
        The environment cfg with any requested mount overrides applied.
    """
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.pitch_override is not None:
        env_cfg.lidar_mount_pitch_deg = args_cli.pitch_override
    if args_cli.pitch_range_override is not None:
        env_cfg.lidar_mount_pitch_range_deg = tuple(args_cli.pitch_range_override)
    if args_cli.no_pitch_range:
        env_cfg.lidar_mount_pitch_range_deg = None
    return env_cfg


def main() -> None:
    env_cfg = build_env_cfg()

    if args_cli.check_static_guard:
        # lidar_mount_pitch_range_deg on a plain (non-rolling) LidarSensorCfg must be rejected,
        # not silently ignored: the static sensor never writes a scan window, so there is nowhere
        # for a per-env rotation to be applied.
        from isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_lidar_env_cfg import (
            resolve_lidar_mount,
        )

        env_cfg.lidar_mount_pitch_range_deg = (22.0, 30.0)
        try:
            resolve_lidar_mount(env_cfg)
        except TypeError as exc:
            print(f"static-cfg guard raised as expected: {exc}")
            return
        raise SystemExit("FAIL: lidar_mount_pitch_range_deg on a static LidarSensorCfg did not raise")
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    cfg = env.cfg
    sensor = env._mid360
    rot = tuple(cfg.mid360_lidar.offset.rot)
    blind = float(getattr(cfg, "lidar_blind_range", 0.0))
    crop = float(getattr(cfg, "lidar_rear_crop_deg", 0.0))
    rolling = bool(getattr(cfg.mid360_lidar, "rolling_scan", False))

    print("\n" + "=" * 78)
    print(f"TASK                : {args_cli.task}")
    print(f"sensor class        : {type(sensor).__name__}")
    print(f"num_rays R          : {sensor.num_rays}")
    print(f"offset.pos          : {tuple(round(v, 6) for v in cfg.mid360_lidar.offset.pos)}")
    print(f"offset.rot (w,x,y,z): {tuple(round(v, 6) for v in rot)}")
    # (e) mount pitch: quat_mul(Rx(180), Ry(p)) = (0, cos(p/2), 0, sin(p/2)) -> p = 2*atan2(z, x).
    pitch_deg = math.degrees(2.0 * math.atan2(rot[3], rot[1]))
    print(f"  -> implied mount pitch: {pitch_deg:.4f} deg")
    print(f"lidar_use_body_occ_mask : {getattr(cfg, 'lidar_use_body_occ_mask', True)}")
    print(f"  -> _body_occ_grid is None: {env._body_occ_grid is None}")
    print(f"lidar_blind_range   : {blind} m")
    print(f"lidar_rear_crop_deg : {crop} deg")
    print(f"rolling_scan        : {rolling}")
    print(f"mount_pitch_range   : {getattr(cfg.mid360_lidar, 'mount_pitch_range_deg', None)} deg")
    # cfg.offset.rot is baked into ray_directions once, at init. The kernel's separate
    # offset quaternion comes from the USD prim hierarchy (sensor prim relative to its
    # rigid-body ancestor), so with prim_path on /Robot/base it must be identity — otherwise
    # the mount rotation would be applied twice and mount_pitch_deg would not describe the
    # body-local pattern.
    _oq = sensor._offset_quat_contiguous
    _ident = torch.zeros_like(_oq)
    _ident[:, 3] = 1.0
    print(
        f"prim-hierarchy offset quat is identity: {bool(torch.allclose(_oq, _ident, atol=1e-6))}"
        f" (row 0 = {[round(float(v), 6) for v in _oq[0]]})"
    )
    print(f"pattern samples     : {cfg.mid360_lidar.pattern_cfg.samples}")
    print(f"pixel_dropout_prob  : {cfg.mid360_lidar.pixel_dropout_prob}")
    print("=" * 78)

    # Reference pattern for check (b): the same rows the sensor is supposed to be walking.
    # Typed as a real tensor rather than an Optional so the later indexing needs no narrowing.
    ref_dirs: torch.Tensor = torch.empty(0, 3, device=env.device)
    if rolling:
        ref_dirs = sensor._scan_dirs_full  # (P, 3), already offset-rotated
        print(f"pattern rows P      : {sensor._scan_pattern_size}")
        print(f"update_period       : {cfg.mid360_lidar.update_period} s")
        # Independent reconstruction from the .npy, to prove the on-device table is right.
        raw = np.load(cfg.mid360_lidar.scan_pattern_file or DEFAULT_SCAN_PATTERN_FILE)
        th, ph = raw[:, 0], raw[:, 1]
        chk = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], 1).astype(np.float32)
        chk /= np.linalg.norm(chk, axis=1, keepdims=True)
        chk = quat_apply_np(rot, chk)
        err = float(np.abs(chk - ref_dirs.cpu().numpy()).max())
        print(f"on-device pattern table vs .npy reconstruction: max abs err = {err:.3e}")

    max_dist = float(cfg.mid360_lidar.max_distance)
    failures: list[str] = []

    # Record every ``_update_buffers_impl`` call: the mask it was handed, which envs' ray
    # directions actually changed, and how far each env's scan position moved.  Instrumenting
    # here rather than in the sensor keeps the test scaffolding out of the training path.
    updates: list[dict] = []
    cur_step = [-1]
    _orig_impl = sensor._update_buffers_impl

    def _recording_impl(env_mask, _orig=_orig_impl):
        d_before = sensor.ray_directions.torch.clone()
        pos_before = sensor._scan_pos.clone() if hasattr(sensor, "_scan_pos") else None
        _orig(env_mask)
        d_after = sensor.ray_directions.torch
        updates.append(
            {
                "step": cur_step[0],
                "mask": wp.to_torch(env_mask).to(torch.bool).view(-1).clone(),
                "changed": (d_after != d_before).flatten(1).any(dim=1).clone(),
                "pos_before": pos_before,
                "pos_after": sensor._scan_pos.clone() if hasattr(sensor, "_scan_pos") else None,
            }
        )

    sensor._update_buffers_impl = _recording_impl

    # Per-control-step traces, for the cadence report.
    step_dirs0: list[torch.Tensor] = []
    push_steps: dict[int, list[int]] = {0: [], 2: []}
    reset_step = max(0, args_cli.steps // 3 + 2)  # ~12 at the default 40 steps: mid-window
    reset_envs = [2, 5]
    reset_done = False

    env.reset()
    for step in range(args_cli.steps):
        cur_step[0] = step
        obs, _, _, _, _ = env.step(torch.zeros(env.num_envs, env.cfg.action_space, device=env.device))

        # Which envs pushed a new accumulator frame this step (counter is zeroed on a push).
        ctr = getattr(env, "_lidar_push_ctr", None)
        if ctr is not None:
            for e in push_steps:
                if int(ctr[e]) == 0:
                    push_steps[e].append(step)

        dirs = sensor.ray_directions.torch  # (N, R, 3) rolling, (R, 3) static
        dist = sensor.data.distances  # (N, R)

        # Reproduce the env's own hit_valid, gate by gate, to attribute the removals.
        base_valid = dist < max_dist
        if env._body_occ_grid is not None:
            az_b = (
                (torch.atan2(dirs[..., 1], dirs[..., 0]) * (180.0 / math.pi) % 360.0 / env._body_occ_res)
                .long()
                .clamp(0, env._body_occ_n_az - 1)
            )
            el_b = (
                ((torch.asin(dirs[..., 2].clamp(-1, 1)) * (180.0 / math.pi) + 90.0) / env._body_occ_res)
                .long()
                .clamp(0, env._body_occ_n_el - 1)
            )
            base_valid = base_valid & ~env._body_occ_grid[el_b, az_b]
        after_blind = base_valid & (dist >= blind) if blind > 0 else base_valid
        az_deg = torch.atan2(dirs[..., 1], dirs[..., 0]) * (180.0 / math.pi)
        rear = az_deg.abs() >= 180.0 - 0.5 * crop if crop > 0 else torch.zeros_like(base_valid)
        final = after_blind & ~rear

        n_base = float(base_valid.sum(-1).float().mean())
        n_final = float(final.sum(-1).float().mean())
        f_blind = float((base_valid & ~after_blind).sum(-1).float().mean()) / max(n_base, 1.0)
        f_crop = float((after_blind & rear).sum(-1).float().mean()) / max(n_base, 1.0)

        line = (
            f"step {step:3d} | dirs {tuple(dirs.shape)} | valid_before_gates {n_base:8.1f}"
            f" | valid_final {n_final:8.1f} | removed_blind {f_blind:6.3%} | removed_crop {f_crop:6.3%}"
        )

        if rolling:
            step_dirs0.append(dirs[0].clone())
            same_env = bool(torch.equal(dirs[0], dirs[1]))
            line += f" | env0==env1: {same_env}"
            if same_env:
                failures.append(f"step {step}: env0 and env1 have identical ray_directions")
            # Every env's live window must be exactly the R pattern rows starting at its own
            # scan position — checked on every control step, update or not.
            pos = sensor._scan_pos
            offs = torch.arange(sensor.num_rays, device=dirs.device)
            # With per-env mount pitch the pattern table is unrotated, so the expected window is
            # the raw rows rotated by that env's own quaternion (xyzw, as warp stores it).
            randomised = getattr(cfg.mid360_lidar, "mount_pitch_range_deg", None) is not None
            werr = 0.0
            for e in (0, 1, 2, 5):
                idx = (int(pos[e]) + offs) % sensor._scan_pattern_size
                want = ref_dirs[idx]
                if randomised:
                    want = quat_apply(sensor._mount_quat_torch[e].expand(want.shape[0], 4), want)
                werr = max(werr, float((dirs[e] - want).abs().max()))
            line += f" | window err {werr:.1e}"
            if werr > 1e-5:
                failures.append(f"step {step}: live directions do not match the pattern rows (err {werr:.2e})")

        print(line)

        # Partial-mask trigger: reset two envs mid-window so their sensor timers desynchronise
        # from the rest.  ``_reset_idx`` is the env's own reset path and reaches
        # ``scene.reset -> sensor.reset``, which zeroes those envs' timestamps.
        if rolling and not reset_done and step == reset_step:
            print(f"    >>> resetting envs {reset_envs} at control step {step} (mid-window)")
            env._reset_idx(torch.tensor(reset_envs, device=env.device))
            reset_done = True

        # (c) no valid hit closer than the blind range
        if blind > 0:
            mn = float(dist[final].min()) if bool(final.any()) else float("nan")
            if not math.isnan(mn) and mn < blind - 1e-6:
                failures.append(f"step {step}: valid hit at {mn:.4f} m < blind {blind} m")
        # (d) no valid hit inside the rear wedge
        if crop > 0 and bool((final & rear).any()):
            failures.append(f"step {step}: {int((final & rear).sum())} valid hits inside the rear wedge")

    report_cadence_and_geometry(env, sensor, cfg, updates, step_dirs0, push_steps, rolling, max_dist, failures)

    # (f) observation contract
    print("-" * 78)
    for k, v in obs.items():
        print(f"obs[{k!r}].shape = {tuple(v.shape)}")
    lid = tuple(obs["lidar"].shape)
    print(f"obs['lidar'] shape == ({args_cli.num_envs}, 7371): {lid == (args_cli.num_envs, 7371)}")
    if lid != (args_cli.num_envs, 7371):
        failures.append(f"obs['lidar'] shape {lid} != ({args_cli.num_envs}, 7371)")

    # The grid arms return early from _get_observations, so the range-image path never runs in
    # normal operation.  Exercise it once directly while per-env directions are live.
    img = env._compute_range_image(sensor.data.distances, final)
    want = (env.num_envs, 2, cfg.lidar_image_h, cfg.lidar_image_w)
    print(
        f"_compute_range_image -> {tuple(img.shape)} (expected {want}); el span "
        f"[{env._lidar_el_min_deg:.2f}, {env._lidar_el_max_deg:.2f}] deg"
    )
    if tuple(img.shape) != want:
        failures.append(f"_compute_range_image shape {tuple(img.shape)} != {want}")

    # Regression guard for the per-chunk ``dirs[beg:end]`` refactor in
    # ``_compute_lidar_occupancy_grid``.  The static sensor already returns a 3-D buffer whose
    # rows are identical, so the old ``dirs[0]`` broadcast and the new per-chunk slice must give
    # bit-identical grids.  Comparing across runs cannot show this (terrain, initial pose and
    # dropout are all unseeded), so it is checked in-process against a replayed dirs[0] path.
    if not rolling:
        d = sensor.ray_directions.torch
        rows_identical = bool(torch.equal(d, d[0].unsqueeze(0).expand_as(d)))
        print(f"static sensor: all env ray_direction rows identical: {rows_identical}")
        if not rows_identical:
            failures.append("static sensor ray_directions differ between envs")
        # Snapshot first: reading ``sensor.data`` can trigger a buffer update, which reads
        # ``ray_directions.warp`` — not something the stand-in below provides.
        dist_snap = sensor.data.distances.clone()
        g_new = env._compute_lidar_occupancy_grid(dist_snap, final)
        # Replay the pre-refactor path by handing the method a genuine (R, 3) buffer, which it
        # still supports via the 2-D broadcast branch.
        saved = sensor.ray_directions
        sensor.ray_directions = type("_Dirs2D", (), {"torch": d[0]})()
        try:
            g_old = env._compute_lidar_occupancy_grid(dist_snap, final)
        finally:
            sensor.ray_directions = saved
        same = bool(torch.equal(g_new, g_old))
        print(f"occupancy grid: per-chunk 3-D slice == legacy dirs[0] broadcast: {same}")
        if not same:
            failures.append(f"grid refactor changed the baseline result ({int((g_new != g_old).sum())} cells)")

    check_reset_redraws(env, sensor, cfg, ref_dirs, rot, rolling, failures)

    print("-" * 78)
    print("SMOKE TEST: PASS" if not failures else "SMOKE TEST: FAIL")
    for f in failures:
        print("  FAIL:", f)

    env.close()

    if args_cli.timing:
        del env
        torch.cuda.empty_cache()
        for task in (_REAL, _BASE):
            c = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.timing_envs)
            e = gym.make(task, cfg=c).unwrapped
            s = e._mid360
            act = torch.zeros(e.num_envs, e.cfg.action_space, device=e.device)
            e.reset()
            for _ in range(3):
                e.step(act)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n = 20
            for _ in range(n):
                wp.to_torch(s._is_outdated).fill_(True)
                s._update_outdated_buffers()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / n * 1e3
            print(
                f"[timing] {task}\n         {type(s).__name__} R={s.num_rays} "
                f"envs={args_cli.timing_envs}: {ms:.3f} ms / sensor update"
            )
            e.close()
            del e
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    simulation_app.close()
