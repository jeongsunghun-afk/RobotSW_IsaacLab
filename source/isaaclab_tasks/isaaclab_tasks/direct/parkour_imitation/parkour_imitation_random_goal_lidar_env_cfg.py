# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Go2 ParkourImitation-Symmetry-RandomGoal + Mid-360 LiDAR environment.

Policy obs (policy/scan/priv_explicit/priv_latent/history) and amp_obs are UNCHANGED
relative to ParkourImitationRandomGoalEnvCfg.  The Mid-360 LiDAR is attached as an
additional sensor but is NEVER concatenated into any policy obs group.  Its range-image
projection is exposed only as ``obs["lidar"]`` (R2 side-channel for the SL runner arm).

height_scan (obs["scan"]) is kept intact as the critic supervision target.

Sim 6.0 migration notes
-----------------------
* Uses the core ``LidarSensorCfg`` (single-mesh native ray-caster) with the core
  ``LivoxPatternCfg(sensor_type="mid360")``.
* The Sim 5.1 fork-only field ``dynamic_env_mesh_prim_paths`` (multi-mesh body
  self-occlusion) is dropped — the native ray-caster is single-mesh.  Body
  self-occlusion is handled at the env level via the precomputed static az/el grid
  (``body_occ_azel_grid.npy``); see ``parkour_imitation_random_goal_lidar_env.py``.
* Mount ``offset`` (pos/rot) matches the already-ported 6.0 ``parkour_lidar_env_cfg.py``
  reference for the identical Mid-360 boom mount.
"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.lidar_sensor_cfg import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LivoxPatternCfg
from isaaclab.utils.configclass import configclass

from .parkour_imitation_random_goal_env_cfg import (
    ParkourImitationRandomGoalEnvCfg,
    apply_easy_entry_terrain_lowering,
)


@configclass
class ParkourImitationRandomGoalLidarEnvCfg(ParkourImitationRandomGoalEnvCfg):
    """Go2 ParkourImitation-RandomGoal cfg with an additional Livox Mid-360 LiDAR.

    All policy-facing fields (observation_space, height_scanner, obs_groups, amp_obs, etc.)
    are inherited UNCHANGED from ParkourImitationRandomGoalEnvCfg.  Only ``mid360_lidar``
    and the range-image projection parameters (lidar_image_h/w, lidar_max_range,
    lidar_frame_stack) are added.

    Mount parameters (pos / rot) and noise settings are identical to those in
    ``parkour_lidar_env_cfg.ParkourLidarEnvCfg`` (6.0 reference), which was verified with
    the Go2 SLAM local-window reference implementation (2026-06-25).

    ``debug_vis=False`` is intentional — this cfg is intended for large-scale training
    where viewport point-cloud rendering would add unnecessary overhead.

    Range-image contract (R2)
    -------------------------
    ``obs["lidar"]`` shape = ``(N, lidar_frame_stack * 2 * lidar_image_h * lidar_image_w)``
    produced by flattening a ``(N, K, C=2, H, W)`` buffer via ``.reshape(N, -1)``.

    - ch0 = range_norm: ``clamp(d, 0, lidar_max_range) / lidar_max_range`` for a hit;
      ``1.0`` for miss/occluded.
    - ch1 = hit_mask: ``1.0`` for valid hit, ``0.0`` for miss/occluded/dropout.
    - Frame stack: newest frame at index k=0, older frames at k=1..K-1.
    """

    # ------------------------------------------------------------------
    # Range-image projection parameters (R2 contract)
    # ------------------------------------------------------------------
    # Elevation bins for the range image (H).
    lidar_image_h: int = 24
    # Azimuth bins for the range image (W).  Full 360° coverage.
    lidar_image_w: int = 96
    # Maximum range for range normalisation (m).  Hits beyond this clamp to 1.0
    # in range_norm but still carry hit_mask=1 (distinct from empty bins).
    lidar_max_range: float = 20.0
    # Number of temporal frames stacked in the obs["lidar"] buffer (K).
    lidar_frame_stack: int = 3

    # Temporal ring-buffer push cadence.
    #
    # False (default): push a new range-image frame into the ring buffer on EVERY env
    #     control step.  With step_dt=0.02 s and K=3 this spans only 0.06 s — the three
    #     slots carry near-identical content (near-duplicate frames).
    #
    # True: push to the ring buffer only once every
    #     push_every = round(1 / (update_frequency * step_dt))
    #   env steps — matching the Mid-360 measurement rate (10 Hz).  At step_dt=0.02 s
    #   this gives push_every=5, so K=3 slots span 3x0.1 s = 0.3 s of real temporal
    #   context.  Non-push steps hold the previous ring-buffer content unchanged
    #   (obs["lidar"] is constant between pushes), matching real 10 Hz deployment.
    #
    # Note: LidarSensorCfg inherits update_period=0.0 from SensorBaseCfg, so the
    #   sensor's _is_outdated gate fires on every physics step regardless of
    #   update_frequency; push_every is therefore derived from update_frequency
    #   directly, not from update_period.
    lidar_stack_at_sensor_rate: bool = False

    # Ring-buffer contents for the not-yet-filled slots after a reset.
    #
    # False (default, legacy): ``_reset_idx`` zeroes the whole buffer.  This contradicts the
    #     range-image convention, where an *empty* bin is ``ch0 = 1.0`` (see
    #     ``_compute_range_image``: ``range_grid = torch.ones(...)``).  ``ch0 = 0.0`` decodes as
    #     ``range_norm = 0`` -> distance 0 -> geometry touching the sensor, so the stale slots
    #     assert a false obstacle rather than "no data".  The state persists for
    #     ``lidar_frame_stack * push_every`` control steps after every reset, i.e. it scales with
    #     K: 15 steps at K=3 up to 75 at K=15, which is 1.9% to 9.7% of a ~775-step episode.
    #     Any comparison across K is therefore confounded by a K-proportional penalty.
    #
    # True: on the first push after a reset, the newly measured frame is replicated into all K
    #     slots, so the stack starts self-consistent and no window of degraded observations
    #     exists.  This also matches deployment, where a policy starting up holds the current
    #     scan rather than a zero frame.
    #
    # Left False by default so runs launched before this option remain reproducible; the arms
    # that need a clean cross-K comparison set it explicitly.
    lidar_fill_stack_on_reset: bool = False

    # ------------------------------------------------------------------
    # Metric occupancy grid (rung A1-0) — alternative to the angular range image
    # ------------------------------------------------------------------
    # When True, ``obs["lidar"]`` carries a binary occupancy grid scattered from the Mid-360
    # hits into the teacher's voxel frame (yaw-aligned, 0.1 m, 27x21x13 = 7371) instead of the
    # stacked range image.  The contents are REPLACED rather than added, because the question the
    # rung asks is "does the metric grid beat the angular stack", not "does adding it help" — and
    # because a new observation key would be dropped by the distillation runner's storage
    # whitelist and never reach the update's minibatches.
    lidar_obs_as_occupancy_grid: bool = False

    # Ceiling control.  Replaces ``obs["lidar"]`` with the privileged voxel grid the teacher
    # itself reads, so the student's terrain input carries zero reconstruction error while every
    # other wiring (encoder, obs key, storage whitelist) stays byte-identical to the A1-0 arm.
    # This bounds the entire "reconstruct the GT voxel from LiDAR" programme: no reconstruction
    # module can do better than being handed the target outright.  Requires
    # ``lidar_obs_as_occupancy_grid`` (it reuses that branch) and ``enable_voxel_scanner``.
    lidar_obs_use_privileged_voxel: bool = False

    # Base-frame z offset of the grid origin.  MUST match the clearance scanner's mount
    # (``RayCasterCfg.OffsetCfg(pos=(0, 0, 0.05))`` in ``parkour_env``) or the student's grid is
    # shifted relative to the teacher's voxel grid and the two stop being comparable.
    lidar_grid_origin_z: float = 0.05

    # Envs processed per chunk while scattering.  The intermediate is ``(chunk, R, 3)`` with
    # R ~= 24k rays, so a full 1024-env batch would allocate ~295 MB before counting the
    # rotation's own temporaries.
    lidar_grid_chunk_size: int = 256

    # ------------------------------------------------------------------
    # Pose-registered accumulation over the metric grid (rung A1-1)
    # ------------------------------------------------------------------
    # Requires ``lidar_obs_as_occupancy_grid``.  Each sensor tick the accumulator is transported
    # into the new yaw-aligned frame and the fresh scan is blended in, so cells that went blind
    # keep the value they had when they were still visible.  This is the mechanism the blind-band
    # hypothesis actually calls for: 74-78% of the cells the robot steps on (gap terrain) were
    # observed earlier at >= 1.2 m range.
    lidar_grid_accumulate: bool = False

    # EMA retention per sensor tick.  Half-life is ln(0.5)/ln(alpha) ticks at 0.1 s each:
    # 0.90 -> 0.66 s, 0.933 -> 1.0 s, 0.955 -> 1.5 s.  The measured lookback requirement is
    # 1.0-1.5 s, so the useful band is ~0.93-0.955.  Note Occupancy Anticipation's 0.9 is tuned
    # for a 5 Hz mapping loop and does not transfer directly.
    lidar_grid_ema_alpha: float = 0.94

    # Accumulate at the sensor rate rather than the control rate.  The Mid-360 only produces new
    # data at ``update_frequency``, so warping every control step would add five resampling
    # passes per new measurement — blur without information.
    lidar_grid_accumulate_at_sensor_rate: bool = True

    # Mid-360 LiDAR — range-image side-channel only, NEVER concatenated into policy obs.
    # Exposed as: obs["lidar"] (N, K*2*H*W).
    mid360_lidar: LidarSensorCfg = LidarSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(
            # Mount position confirmed from SLAM_local_window mount_transform.py
            # (MOUNT_OFFSET_BASE_LINK_M): x=0.3336 uses the runtime-faithful boom mount.
            pos=(0.333644, -0.000485, 0.050079),
            # dome-down 180° flip ⊗ pitch-up 30° — matches 6.0 parkour_lidar_env_cfg reference.
            rot=(0.0, 0.96593, 0.0, 0.25882),
        ),
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            samples=24000,
            use_simple_grid=False,
            downsample=1,
        ),
        ray_alignment="base",
        mesh_prim_paths=["/World/ground"],
        max_distance=40.0,
        min_range=0.2,
        debug_vis=False,  # disabled for training — no viewport overhead
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        update_frequency=10.0,
        # Noise: Mid-360 datasheet range accuracy ±2 cm (1σ); 10% dropout for sim-to-real DR.
        enable_sensor_noise=True,
        random_distance_noise=0.02,
        pixel_dropout_prob=0.10,
        random_angle_noise=0.0,
        pixel_std_dev_multiplier=0.0,
    )


@configclass
class ParkourImitationRandomGoalLidarSLEnvCfg(ParkourImitationRandomGoalLidarEnvCfg):
    """SL (R2 student-learning) arm cfg for ``Go2-ParkourImitation-Symmetry-RandomGoal-Lidar-SL-v0``.

    Identical to :class:`ParkourImitationRandomGoalLidarEnvCfg` (policy obs, amp_obs, LiDAR
    range-image side-channel — all unchanged) except the LiDAR temporal frame-stack is pushed
    at the sensor rate instead of the control rate.

    With the default control-rate cadence, K=3 frames span only step_dt*3 = 0.06 s of
    near-duplicate content.  Setting ``lidar_stack_at_sensor_rate = True`` pushes frames every
    ``push_every`` control steps so K=3 spans ~0.3 s (matching the 10 Hz sensor cadence).
    Obs shape is unchanged (``obs["lidar"]`` stays (N, 3*2*24*96=13824)).
    """

    lidar_stack_at_sensor_rate: bool = True


@configclass
class ParkourImitationRandomGoalLidarSLEasyEntryEnvCfg(ParkourImitationRandomGoalLidarSLEnvCfg):
    """LiDAR SL (student-learning) arm on the EasyEntry lowered-obstacle terrain floor.

    Identical to :class:`ParkourImitationRandomGoalLidarSLEnvCfg` (Mid-360 LiDAR range-image
    side-channel, sensor-rate frame stacking, policy obs, amp_obs — all unchanged) except the
    difficulty-0 end of each obstacle sub-terrain is lowered so the 6.0 curriculum can
    re-enter obstacle levels (see
    :class:`~isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_env_cfg.ParkourImitationRandomGoalEasyEntryEnvCfg`).

    LiDAR keeps the standard 5-terrain mix (no ``parkour_crawl`` rebalance), so the lowering
    applies to the same four obstacle sub-terrains as the base EasyEntry variant.
    """

    def __post_init__(self):
        super().__post_init__()
        apply_easy_entry_terrain_lowering(self)


@configclass
class ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg(ParkourImitationRandomGoalLidarSLEasyEntryEnvCfg):
    """Distillation arm: emits the voxel-teacher obs and the LiDAR-student obs in the same step.

    Used by ``Go2-ParkourImitation-Lidar-Distill-EasyEntry-v0``, where a frozen voxel teacher
    labels actions that a raw-LiDAR student imitates.  Both policies must see the state they
    were built for **within one env**, so this cfg turns on every teacher-side producer on top
    of the LiDAR SL arm:

    * ``enable_clearance_scanner`` / ``clearance_as_scan`` — routes clearance-294 into the
      ``"scan"`` slot.  The teacher's critic consumes it, and :class:`ActorCriticRMALidar`
      asserts the group exists (it sizes the actor MLP from it).
    * ``enable_voxel_scanner`` — adds ``obs["voxel"]`` (N, 7371).  The voxel fill reuses the
      clearance ray hits, so the clearance scanner above is a prerequisite, not an extra.

    ``obs["lidar"]`` (N, 13824) is still produced by
    :class:`~isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_lidar_env.Go2ParkourImitationRandomGoalLidarEnv`,
    so no new env class is needed — only this cfg.

    **Terrain must match the teacher's.** The teacher was trained on the Teacher3D mix that
    adds ``parkour_crawl`` at 0.25, whereas the LiDAR arms inherit the standard 5-terrain mix.
    Rolling the student out on a different mix would generate teacher labels on states the
    teacher never trained for, so ``__post_init__`` rewrites the proportions to the teacher's
    allocation before re-applying the EasyEntry lowering.
    """

    # Teacher-side obs producers (the student only reads obs["lidar"]).
    enable_clearance_scanner: bool = True
    clearance_as_scan: bool = True
    enable_voxel_scanner: bool = True


@configclass
class ParkourImitationRandomGoalLidarDistillK10EasyEntryEnvCfg(ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg):
    """Distillation arm with the temporal window widened from 0.3 s to 1.0 s (K=3 -> 10).

    Rung A0 of the voxel-reconstruction ladder (see
    ``_workspace/distill_videos/PLAN_voxel_recon_module.md``).  This is the control arm: it
    changes the window length and **nothing else** — no metric grid, no ego-motion
    registration, no auxiliary perception loss.  Its purpose is to make the later rungs
    attributable: without it, any gain from registration cannot be separated from the gain
    of simply looking further back.

    Why 1.0 s.  With ``lidar_stack_at_sensor_rate=True`` (inherited), frames are pushed every
    ``push_every = round(1 / (10 Hz * 0.02 s)) = 5`` control steps, so K frames span
    ``K * 0.1 s``.  Measurement showed 74-78% of the cells the robot steps on (gap terrain)
    were observed earlier at >= 1.2 m range, needing a 1.0-1.5 s lookback that the previous
    0.3 s window could not reach.  K=10 matches Omni-Perception's ``Nhist=10`` scans;
    DreamWaQ++ uses K=5 (0.5 s).  Raising the sensor rate instead would not help — the
    Mid-360 emits a fixed ~200k points/s on a non-repetitive scan pattern, so a higher
    publish rate slices the same point budget into sparser frames.

    Consequences of K=10:

    * ``obs["lidar"]`` grows from ``3*2*24*96 = 13824`` to ``10*2*24*96 = 46080`` floats.
    * ``ActorCriticRMALidar`` needs no cfg change — it infers K from the observation width.
    * Rollout storage grows accordingly; see
      :class:`~isaaclab_tasks.direct.parkour_imitation.agents.rsl_rl_amp_cfg.Go2ParkourImitationLidarDistillK10RunnerCfg`.

    Everything the teacher needs (clearance scanner, ``clearance_as_scan``, voxel scanner,
    the teacher's terrain mix) is inherited unchanged, so teacher labels stay on-distribution.
    """

    lidar_frame_stack: int = 10


@configclass
class ParkourImitationRandomGoalLidarDistillK5EasyEntryEnvCfg(ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg):
    """Distillation arm with a 0.5 s temporal window (K=5) — the lower bracket of rung A0.

    Run alongside :class:`ParkourImitationRandomGoalLidarDistillK10EasyEntryEnvCfg` (1.0 s) and
    the K=3 baseline (0.3 s) to turn rung A0 into a **dose-response curve** over window length
    rather than a single point.  A monotonic trend across 0.3 / 0.5 / 1.0 s is much stronger
    evidence that the window is the lever than one arm beating one baseline; a flat trend beyond
    0.5 s says the extra history is not being used.

    K=5 is also the value DreamWaQ++ uses (it concatenates the last 5 point-cloud measurements),
    so this arm doubles as a literature-matched reference point.  Note DreamWaQ++ additionally
    SE(3)-registers those frames, which this arm deliberately does not — registration is rung A1.

    ``obs["lidar"]`` = ``5*2*24*96 = 23040`` floats.  ``ActorCriticRMALidar`` infers K from that
    width, so no policy-side change is needed.
    """

    lidar_frame_stack: int = 5


@configclass
class ParkourImitationRandomGoalLidarDistillK15EasyEntryEnvCfg(ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg):
    """Distillation arm with a 1.5 s temporal window (K=15) — upper bracket of rung A0.

    Completes the window-length dose-response curve (0.3 / 0.5 / 1.0 / 1.5 s).  1.5 s is the
    upper end of the measured requirement: 74-78% of the cells the robot steps on (gap terrain)
    were observed earlier at >= 1.2 m range, which puts the needed lookback at 1.0-1.5 s.

    This arm answers whether 1.0 s was still short.  If K=15 keeps improving over K=10 the
    window is not yet saturated; if it flattens or regresses, 1.0 s already covers the useful
    history and the remaining deficit is not a matter of window length.

    Note the cost is not only memory: at K=15 the CNN sees 30 input channels, so more of the
    encoder's capacity goes to stacking frames that a registered representation would have
    merged.  A regression here would itself be evidence for rung A1.

    ``obs["lidar"]`` = ``15*2*24*96 = 69120`` floats — 5x the K=3 baseline.  Rollout storage for
    the lidar group is ~4.53 GB at ``num_steps_per_env=16`` and 1024 envs, putting the process
    at roughly 16.1 GB; it needs a card with that much free.
    """

    lidar_frame_stack: int = 15


@configclass
class ParkourImitationRandomGoalLidarDistillK10FixEasyEntryEnvCfg(
    ParkourImitationRandomGoalLidarDistillK10EasyEntryEnvCfg
):
    """K=10 with the post-reset stack warm-up removed — discriminator for the A0 trend.

    Rung A0 measured a penalty that grows monotonically with K.  Two explanations fit:

    1. **Encoder capacity**: without ego-motion registration, a larger K only widens the CNN's
       input channels (6 -> 20 -> 30) and spends capacity stacking frames a registered
       representation would have merged.
    2. **Post-reset warm-up**: the ring buffer is zeroed on reset, and ``ch0 = 0.0`` decodes as
       geometry touching the sensor rather than as an empty bin (``ch0 = 1.0``).  That false
       observation lasts ``K * push_every`` control steps after every reset — 15 steps at K=3 up
       to 75 at K=15, i.e. 1.9% to 9.7% of a ~775-step episode — so the penalty is
       K-proportional by construction.

    Both predict "K=15 worst", so extending the window sweep cannot separate them.  This arm
    can: it is K=10 with explanation 2 removed (``lidar_fill_stack_on_reset``), and nothing else
    changed.  If the penalty versus K=3 disappears it was warm-up; if it survives, capacity (or
    something else) remains in play.

    The distinction matters beyond bookkeeping: recording the wrong mechanism in the plan would
    steer the later rungs, and warm-up is a defect worth fixing on its own merits regardless of
    which explanation wins.
    """

    lidar_fill_stack_on_reset: bool = True


@configclass
class ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg(ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg):
    """Rung A1-0: single-frame metric occupancy grid instead of the stacked range image.

    Isolates **representation** from **accumulation**.  The original A1 plan bundled both (metric
    grid *and* a pose-registered accumulator), which violates the ladder's own one-knob-per-rung
    rule; A1-0 changes only how a single scan is encoded, and A1-1 adds the accumulator on top.

    What changes: ``obs["lidar"]`` becomes a binary 27x21x13 = 7371 occupancy grid in the
    teacher's exact voxel frame (yaw-aligned, 0.1 m, origin at the clearance mount) instead of
    the ``3*2*24*96 = 13824`` range-image stack.  Note the observation is **smaller** while
    carrying more of the scan: the angular projection min-reduces ~13.8k forward hits into ~1.3k
    bins, and its row spacing degrades from 0.11 m at 0.74 m to 1.69 m at 4.05 m.

    Two effects arrive together and cannot be separated within this rung:

    * the ~90% of hits the min-reduce discarded are kept, and
    * the base-vs-yaw frame mismatch disappears (the Mid-360 is ``ray_alignment="base"`` so its
      rays pitch with the body, while the teacher's grid is yaw-aligned).  A pitch-gated deficit
      was measured on step terrain — raised tops first became visible 2.6-2.7 deg more nose-up —
      so some step improvement may come from the frame alone rather than from the extra hits.

    The paired runner cfg switches the student's encoder to the voxel CNN, making student and
    teacher **architecturally identical** and differing only in where the grid comes from:
    privileged clearance rays for the teacher, the student's own Mid-360 here.  That is the
    cleanest available test of "is the deficit the sensor or the representation?".

    The frame stack is inherited but unused — the range-image path is skipped entirely.
    """

    lidar_obs_as_occupancy_grid: bool = True


@configclass
class ParkourImitationRandomGoalLidarDistillCeilingEasyEntryEnvCfg(
    ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg
):
    """Ceiling control: the student is handed the teacher's own grid instead of its LiDAR's.

    Everything is identical to the A1-0 grid arm except the *provenance* of ``obs["lidar"]`` —
    here it is a copy of ``obs["voxel"]``, the privileged clearance-ray grid the teacher reads.
    Reconstruction error is therefore exactly zero.

    Why this is the right gate before building any LiDAR history/reconstruction module: such a
    module can at best recover the teacher's grid, so whatever this arm reaches is an **upper
    bound** on the entire programme.  Two outcomes, both decisive:

    * reaches the teacher's pinned completion (0.260) — reconstruction *is* the binding
      constraint, and the headroom a module could buy is quantified (A1-0 sits at 0.238).
    * stalls near A1-0 — no reconstruction module can help, because a perfect one is already
      being simulated here.  The residual then lives in the objective or the control axis, not
      in perception.

    Note the student's own encoder and latent path still differ from the teacher's
    (``hist_encoding=True`` versus the teacher's path), so the loss floor is not expected to be
    zero.  That residual is the latent-path difference, which is exactly what this control holds
    fixed while removing grid error.
    """

    lidar_obs_use_privileged_voxel: bool = True


@configclass
class ParkourImitationRandomGoalLidarDistillAccEasyEntryEnvCfg(
    ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg
):
    """Rung A1-1: pose-registered accumulation on top of the metric grid.

    Adds exactly one thing to A1-0 — memory.  Each sensor tick the accumulator is transported into
    the new yaw-aligned frame using the frame's own pose change and the fresh scan is merged with
    ``max(alpha * warped, frame)``, so a cell that has gone blind keeps the value it had while it
    was still visible.

    **This is the first rung that actually addresses the blind band.** A1-0 changed only how a
    single scan is encoded, so the frontal corridor at 0.45-1.0 m stayed empty; here it can be
    filled from the 74-78% of stepped-on cells (gap terrain) that were observed earlier at
    >= 1.2 m range.

    ``alpha`` sets the memory half-life in sensor ticks (0.1 s each): 0.933 -> 1.0 s,
    0.955 -> 1.5 s, bracketing the measured requirement.  The default 0.94 sits just inside that
    band; it is the obvious first thing to sweep if the rung is inconclusive.

    Deliberately a decayed max rather than a lerp: a lerp would report a currently observed cell
    as ``1 - alpha`` = 0.06 on first sighting, which inverts the intent that present observations
    dominate and only memory fades.

    Registration uses the simulator's exact pose.  Deployment cannot, so the paired A1-1b arm
    (once added) integrates the ``Estimator``'s predicted body velocity instead; the difference
    between the two is the cost the estimator imposes, which nothing in this repo has measured.
    """

    lidar_grid_accumulate: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Match the teacher's terrain allocation (Teacher3D mix, crawl included).
        for key in self.terrain.terrain_generator.sub_terrains:
            self.terrain.terrain_generator.sub_terrains[key].proportion = 0.0
        self.terrain.terrain_generator.sub_terrains["parkour_flat"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_hurdle"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_step"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_gap"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_stair"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_crawl"].proportion = 0.25
        # Re-apply the re-entry rung: the lowering rewrites only the *lower* range bounds of
        # the four obstacle terrains, so running it after the proportion rewrite is idempotent.
        apply_easy_entry_terrain_lowering(self)


@configclass
class ParkourImitationRandomGoalLidarDistillR4EasyEntryEnvCfg(ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg):
    """Distillation arm with the range channel normalised over 4 m instead of 20 m.

    ``ch0 = clamp(d, 0, lidar_max_range) / lidar_max_range``.  At 20 m the band that actually
    matters for foot placement (0-2 m) occupies only the first 10% of the channel's numeric
    range, so near-field depth differences are compressed into a sliver.  4 m matches the
    teacher's clearance ``max_distance``, giving the useful band 5x the dynamic range.

    Measured justification (see ``_workspace/distill_videos/ROADMAP.md``):

    * Only 3.3% of forward ground hits lie beyond 5 m, and the rows that carry them already have
      1.7-7.1 m of metric quantisation per row — so clamping loses almost nothing.
    * The student's advantage over the teacher comes from the 1.2-2.0 m band (hurdle +0.76), which
      4 m preserves; cutting at 2.0 m would have destroyed it.

    ``ch1`` (hit_mask) is untouched, so a genuine hit beyond 4 m is still distinguishable from a
    miss even though both saturate ch0 to 1.0.

    NOTE: this changes the observation scaling, so a policy trained with 20 m cannot be evaluated
    under this cfg — it needs its own training run.
    """

    lidar_max_range: float = 4.0


@configclass
class ParkourImitationRandomGoalLidarSLGridCrawlEasyEntryEnvCfg(
    ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg
):
    """From-scratch LiDAR-grid arm trained on a mix that **includes** ``parkour_crawl``.

    The SL-Grid arm this inherits from trains on the five-terrain mix at 0.20 each and never
    sees a crawl tunnel, while the privileged voxel teacher allocates 0.25 to crawl. Probing
    both on crawl therefore measured zero-shot transfer, not capability: SL-Grid completed
    **0 of 542 pinned episodes** and toppled in 43% of them, against 0.430 for the teacher —
    a result that says nothing about whether a Mid-360 grid can support crawling.

    Proportions here are ``parkour_flat`` 0.20 with the other five at 0.16 (sum 1.0), so crawl
    carries the same weight as every other obstacle rather than the teacher's 0.25. The flat
    share stays highest because the AMP flat-column mask needs it and because a flat walker is
    the curriculum's entry point.

    ``parkour_crawl`` is deliberately left out of
    :func:`~isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_env_cfg.apply_easy_entry_terrain_lowering`,
    matching the teacher: the helper only lowers hurdle / gap / step / stair, and a crawl
    ceiling has no "lower bound" that makes the tunnel easier without removing it.

    **Not comparable to the existing SL-Grid checkpoint.** The terrain distribution differs, so
    this starts its own baseline family; the two cannot share a results table.
    """

    def __post_init__(self):
        super().__post_init__()
        for key in self.terrain.terrain_generator.sub_terrains:
            self.terrain.terrain_generator.sub_terrains[key].proportion = 0.0
        self.terrain.terrain_generator.sub_terrains["parkour_flat"].proportion = 0.20
        for key in ("parkour_hurdle", "parkour_step", "parkour_gap", "parkour_stair", "parkour_crawl"):
            self.terrain.terrain_generator.sub_terrains[key].proportion = 0.16
        # Idempotent after the proportion rewrite: the lowering only touches range bounds.
        apply_easy_entry_terrain_lowering(self)
