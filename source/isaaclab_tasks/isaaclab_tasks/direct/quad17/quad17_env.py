# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DTC Phase-1 17-DOF quadruped velocity-tracking env (flat terrain, ActorCriticRMA).

Adapted from ``direct/hind_leg`` (2-leg). Differences: root body is "Base", there are 4 feet
(``.*_foot_contact_link``), and the gait clock is a trot (diagonal pairs anti-phase) instead of
the biped's left/right anti-phase. The observation/reward/RMA-group machinery is otherwise reused
verbatim, which is why per-leg DOF (hip/thigh/calf/foot) matching the hind_leg base mattered.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor

from .quad17_env_cfg import Quad17VelocityEnvCfg

# Trot gait: diagonal leg pairs move together. (HL, FR) share phase 0.0; (HR, FL) share phase 0.5.
_TROT_PHASE_OFFSET = {"HL": 0.0, "FR": 0.0, "HR": 0.5, "FL": 0.5}

# Nominal foot offset in the base frame (x forward, y left), used ONLY to seed obs-visible foothold
# targets at reset (before body_pos_w / the lazily-captured _foot_offset_b are available). Rough
# geometry (~0.61 m wheelbase); refined to the true sole position on the first post-reset reward step.
_NOMINAL_FOOT_OFFSET_B = {
    "FL": (0.30, 0.15),
    "FR": (0.30, -0.15),
    "HL": (-0.30, 0.15),
    "HR": (-0.30, -0.15),
}


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Quad17Env(DirectRLEnv):
    cfg: Quad17VelocityEnvCfg

    def __init__(self, cfg: Quad17VelocityEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.command_curriculum = self.cfg.command_curriculum
        self.curriculum_rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

        if self.cfg.history_observation:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.history_len, self.cfg.num_prio_obs, device=self.device, dtype=torch.float
            )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "similar_to_default",
                "base_height",
                "termination",
                "gait_stance",
                "gait_swing",
                "foot_slip",
                "foothold_track",  # P2: touchdown foothold-tracking bonus (positive, sparse)
                "foothold_place_err",  # P2: diagnostic — clamped XY placement err at touchdowns (m^2)
                "foot_in_gap",  # P2 gap-test: mild penalty for a stance foot dropped into a trench
                "base_pose_track",  # DTC Eq1: advancing base-position tracking (anti-hesitation)
                "progress",  # DTC linear term: clipped forward base velocity (constant fwd gradient)
            ]
        }
        # Base body ("Base" — capital, MJCF root).
        self._base_id, _ = self._contact_sensor.find_bodies("Base")
        # Contact-sensor foot bodies (4) for swing/stance gating.
        self._feet_ids, feet_contact_names = self._contact_sensor.find_bodies(".*_foot_contact_link")
        # Robot-articulation foot bodies (different index space) for sole position/velocity.
        self._sole_body_ids, sole_body_names = self._robot.find_bodies(".*_foot_contact_link")
        assert len(self._sole_body_ids) == 4, (
            f"Expected exactly 4 sole bodies matching '.*_foot_contact_link', got {len(self._sole_body_ids)}: "
            f"{sole_body_names}"
        )
        # Foot ordering must match between contact sensor and robot bodies, else swing/stance gating swaps.
        assert [n.split("/")[-1] for n in feet_contact_names] == [n.split("/")[-1] for n in sole_body_names], (
            f"Foot name order mismatch — contact sensor: {feet_contact_names}, sole bodies: {sole_body_names}"
        )

        # Per-foot trot phase offset (aligned to _sole_body_ids order), parsed from the leg prefix.
        offsets = []
        for n in sole_body_names:
            leg = n.split("/")[-1].split("_")[0]  # e.g. "HL_foot_contact_link" -> "HL"
            assert leg in _TROT_PHASE_OFFSET, f"Unknown leg prefix '{leg}' in foot body '{n}'"
            offsets.append(_TROT_PHASE_OFFSET[leg])
        self._foot_phase_offset = torch.tensor(offsets, device=self.device)  # (4,)

        # Nominal base-frame foot offset (4, 2), _sole_body_ids order — reset-seed only (see constant).
        nominal = []
        for n in sole_body_names:
            leg = n.split("/")[-1].split("_")[0]
            nominal.append(_NOMINAL_FOOT_OFFSET_B[leg])
        self._nominal_foot_offset = torch.tensor(nominal, device=self.device)  # (4, 2)

        # -------- IK desired-joint obs static index maps (DTC Fig 7D enabler) --------
        # The +12 IK block encodes the current foothold target in JOINT space via a first-order
        # Jacobian inverse (see _ik_desired_joint). Build the static maps once: for each of the 4
        # sole bodies (in _sole_body_ids order) its row in the articulation Jacobian, and that leg's
        # hip/thigh/calf columns. Floating base -> Jacobian body index == articulation body index and
        # joint columns are shifted by the 6 base-DOF columns; fixed base -> body-1 / no shift (see
        # isaaclab.envs.mdp.actions.task_space_actions for this convention).
        self._ik_joints_per_leg = 3  # hip/thigh/calf -> 4 legs x 3 = +12 obs dims
        _fixed = self._robot.is_fixed_base
        _col_offset = 0 if _fixed else 6
        _jac_body_ids, _jac_col_ids = [], []
        for _bi, _n in zip(self._sole_body_ids, sole_body_names):
            _leg = _n.split("/")[-1].split("_")[0]  # e.g. "HL_foot_contact_link" -> "HL"
            _jac_body_ids.append(_bi - 1 if _fixed else _bi)
            _cols = []
            for _j in ("hip", "thigh", "calf"):
                _jid, _ = self._robot.find_joints(f"{_leg}_{_j}_joint")
                assert len(_jid) == 1, f"expected exactly 1 joint for '{_leg}_{_j}_joint', got {_jid}"
                _cols.append(_jid[0] + _col_offset)
            _jac_col_ids.append(_cols)
        self._ik_jac_body_ids = torch.tensor(_jac_body_ids, device=self.device, dtype=torch.long)  # (4,)
        self._ik_jac_col_ids = torch.tensor(_jac_col_ids, device=self.device, dtype=torch.long)  # (4, 3)

        # Sole rest-z captured lazily on first reward step (body_pos_w invalid in __init__).
        self._sole_rest_z: torch.Tensor | None = None

        # -------- DTC Phase-2 foothold buffers (all in _sole_body_ids order) --------
        n_env = self.num_envs
        # World-frame reference footholds: current target + one-step look-ahead, and the actual
        # landing position captured at each touchdown.
        self._foothold_target = torch.zeros(n_env, 4, 3, device=self.device)
        self._foothold_target2 = torch.zeros(n_env, 4, 3, device=self.device)
        self._stance_pos = torch.zeros(n_env, 4, 3, device=self.device)
        # Time (s) since each foot's target was last (re)anchored; obs channel + age fallback.
        self._target_age = torch.zeros(n_env, 4, device=self.device)
        # Per-env flag: seed was pose-relative approximate; refine to true sole pos on first reward.
        self._foothold_needs_refine = torch.zeros(n_env, dtype=torch.bool, device=self.device)
        # Accurate base-frame foot offset (4, 2), captured lazily in _get_rewards (needs body_pos_w).
        self._foot_offset_b: torch.Tensor | None = None

        # Eval-only foothold diagnostics (NOT zeroed on episode reset) for the ablation gate (Step 8).
        # Purely diagnostic — never added to the reward, never in obs — so they do not affect training
        # or checkpoint compatibility. The ablation eval script zeroes these after a settle window.
        self._eval_td_count = torch.zeros((), device=self.device)  # total touchdown events
        self._eval_track_sum = torch.zeros((), device=self.device)  # sum of -log(err2+eps) over touchdowns
        self._eval_err_sum = torch.zeros((), device=self.device)  # sum of clamped XY err2 over touchdowns

        # Gait phase clock in [0, 1); randomized per-episode in _reset_idx.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)

        # -------- DTC Eq1 base-pose (position) tracking: anti-hesitation reference (v3) --------
        # Per-env anchor for an advancing base reference x_ref = x0 + cmd_vx*t_ref, y_ref = y0, where
        # x0/y0 = the robot's ACTUAL base xy at reset and t_ref = (episode_length_buf - t0_buf)*dt so
        # t_ref = 0 at EVERY reset (robust to IsaacLab's staggered-start randint on the first full
        # reset). Standing still lets base_x fall behind the advancing x_ref -> exp tracking error +
        # linear progress penalty + progress termination all push forward into the trenches, where the
        # foothold obs + gap physical drop teach tracking. All three MUST be (re)set in _reset_idx.
        self._base_ref_x0 = torch.zeros(self.num_envs, device=self.device)
        self._base_ref_y0 = torch.zeros(self.num_envs, device=self.device)
        # episode_length_buf value captured at reset -> subtract so the reference clock starts at 0.
        self._base_ref_t0_buf = torch.zeros(self.num_envs, device=self.device)
        # previous-step base x for the linear forward-progress (velocity) reward.
        self._prev_base_x = torch.zeros(self.num_envs, device=self.device)

        # -------- DTC Phase-2 gap-test terrain wiring --------
        # gap_terrain toggle may be overridden by env-var (QUAD17_GAP_TERRAIN=0/1); default = cfg value.
        import os as _os

        _gap_env = _os.environ.get("QUAD17_GAP_TERRAIN", "").strip()
        if _gap_env != "":
            self.cfg.gap_terrain = _gap_env not in ("0", "false", "False", "no")
        self._gap = bool(self.cfg.gap_terrain)
        # Per-episode forward spawn offset (m): shifts the robot's phase relative to the static trench
        # pattern each reset so it cannot memorize a fixed body-relative gap schedule. Re-randomized in
        # _reset_idx; bounded to one nominal period so the feet stay on the spawn platform.
        self._spawn_dx = torch.zeros(self.num_envs, device=self.device)
        self._spawn_dx_max = float(self.cfg.gap_strip_width + self.cfg.gap_width)
        if self._gap:
            # Trench field geometry was built in _setup_scene; _gap_surface_z (strip-top world z) and
            # _strip_centers_x (sorted world-x of every strip center, for the foothold snap) are set there.
            # Raise the base-height target and the command distribution for the crossing test.
            self.cfg.base_height_target = self.cfg.base_height_target + self._gap_surface_z
            self.cfg.command_cfg = {
                "lin_vel_x_range": list(self.cfg.gap_lin_vel_x_range),  # forward vx only
                "lin_vel_y_range": [0.0, 0.0],
                "ang_vel_range": [0.0, 0.0],
            }
            self.cfg.rel_standing_envs = 0.0  # no standing envs: every robot must cross trenches
            print(
                f"[quad17][P2][gap] transverse trenches ON: strip_w={self.cfg.gap_strip_width} "
                f"gap_w PER-BAND RAMP {self.cfg.gap_width_min}->{self.cfg.gap_width_max} m "
                f"+/-{self.cfg.gap_spacing_jitter} (start={self.cfg.curric_x_start}m ramp={self._gap_curric_ramp:.2f}m "
                f"corridor={self._gap_corridor_len:.2f}m x{self._gap_n_corridors}) depth={self.cfg.gap_depth} "
                f"surface_z={self._gap_surface_z:.3f} n_strips={self._strip_centers_x.numel()} "
                f"base_h_target={self.cfg.base_height_target:.3f} cmd_vx={self.cfg.gap_lin_vel_x_range}"
            )

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalzied_body_names)

        # Ablation gate (Step 8): QUAD17_FOOTHOLD_ABLATE = "" | "zero" | "scramble". Set at eval time
        # to blind/scramble ONLY the 28-dim foothold obs block; the reward still scores against the
        # true target buffer, so a drop in foothold_track under ablation proves the channel is used.
        import os

        self._foothold_ablate = os.environ.get("QUAD17_FOOTHOLD_ABLATE", "").strip().lower()

        # Optional env-var overrides for foothold hyperparams (Step-8 remedy sweeps: increase jitter /
        # lower eps to lift the jitter signal above the reward floor). Committed cfg defaults stay at
        # the mandated spec values; overrides apply only when the var is set, and MUST be set identically
        # for a retrain and its ablation eval (they change both the obs targets and the reward).
        for _var, _attr in (
            ("FOOTHOLD_JITTER_XY", "foothold_jitter_xy"),
            ("FOOTHOLD_JITTER_Z", "foothold_jitter_z"),
            ("FOOTHOLD_EPS", "foothold_track_eps"),
            ("FOOTHOLD_ERR2_MAX", "foothold_err2_max"),
            ("FOOTHOLD_TRACK_SCALE", "foothold_track_reward_scale"),
            # DTC Eq1 base-pose reference sweep knobs (change reward only; safe to vary per run).
            ("BASE_POSE_SCALE", "base_pose_track_reward_scale"),
            ("BASE_POSE_SIGMA", "base_pose_track_sigma"),
            # v3 anti-hesitation sweep knobs: linear progress reward + progress-behind termination.
            ("PROGRESS_SCALE", "progress_reward_scale"),
            ("PROGRESS_FAIL_DIST", "progress_fail_dist"),
        ):
            _val = os.environ.get(_var, "").strip()
            if _val:
                setattr(self.cfg, _attr, float(_val))
                print(f"[quad17][P2] override {_attr} = {_val}")

        foothold_dim = self.cfg.num_foothold_targets * 4 * 3 + 4 if self.cfg.foothold_obs else 0
        ik_dim = 4 * self._ik_joints_per_leg if (self.cfg.foothold_obs and self.cfg.foothold_ik_obs) else 0
        print(
            f"[quad17][P2] observation_space={self.cfg.observation_space} num_prio_obs={self.cfg.num_prio_obs} "
            f"foothold_obs={self.cfg.foothold_obs} foothold_dim(+{foothold_dim}) "
            f"ik_obs={self.cfg.foothold_ik_obs} ik_dim(+{ik_dim}) "
            f"ablate='{self._foothold_ablate or 'none'}' lin_vel_scale={self.cfg.lin_vel_reward_scale}"
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # DTC Phase-2 gap test: add randomized transverse trenches on top of the flat ground plane
        # (the plane at z=0 becomes the trench floor; solid strips are raised gap_depth above it).
        # Built as ONE shared static mesh under /World/ground BEFORE cloning (terrain is not cloned).
        if bool(self.cfg.gap_terrain):
            self._build_gap_terrain()
        # Remove the stray ground plane the MJCF->USD conversion baked *inside* the robot. Must run on
        # the source env BEFORE cloning/physics replication so every env inherits it disabled.
        self._disable_baked_floor_collision()
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _disable_baked_floor_collision(self):
        """Disable the stray ground ``CollisionPlane`` baked into the robot USD.

        The MJCF->USD conversion imported the MJCF ``worldBody`` ground plane as a rigid body *inside*
        the robot articulation (``/Robot/worldBody/floor`` with a ``CollisionPlane``). Left enabled it
        is an infinite collider fixed at the robot's local z=0 that collides pathologically with the
        robot's own feet/base on the first physics step after every reset — producing a spurious
        ~90-270 N contact force reported on the ``Base`` body (which trips the base-contact termination
        on step 1, so every episode dies immediately and mean episode length is pinned at 1.0) and a
        violent depenetration launch. Disabling its collision on the source env (before cloning) lets
        the robot rest cleanly on the real ``/World/ground`` plane instead.
        """
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        count = 0
        for prim in stage.Traverse():
            if "/Robot/worldBody/floor" in prim.GetPath().pathString and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                count += 1
        print(f"[quad17] disabled baked-in floor collision on {count} prim(s)")

    def _build_gap_terrain(self):
        """Build the shared static trench field (transverse gaps) for the DTC Phase-2 gap test.

        The flat ground plane (``/World/ground/terrain`` at z=0) stays and becomes the TRENCH FLOOR.
        On top of it we lay periodic solid strips (full-width in y, thin in x) whose tops sit at
        ``z = gap_depth`` — so a foot that lands on a strip stands at the walking surface (z=gap_depth)
        and a foot that misses drops ``gap_depth`` (>=0.25 m) down onto the plane: a real physical
        drop, not a virtual penalty. Strip spacing is JITTERED per strip so there is no memorizable
        period; combined with the position-free policy obs this forces the policy to read the
        (strip-snapped) foothold obs to know where solid ground is.

        Geometry is ONE concatenated trimesh imported under ``/World/ground`` (shared, not cloned),
        so a single field serves every env (env origins share x per grid row, and strips span full y).
        A wide solid PLATFORM at each spawn row guarantees all four feet start on solid ground; the
        trenches begin forward of it. ``_strip_centers_x`` (sorted world-x of every strip center) is
        stored for the foothold snap so the reference targets EXACTLY match the physical strips.
        """
        import os as _os

        import trimesh

        strip_w = float(self.cfg.gap_strip_width)
        jitter = float(self.cfg.gap_spacing_jitter)
        depth = float(self.cfg.gap_depth)
        corridor = float(self.cfg.gap_forward_corridor)
        plat_len = 2.0  # wide spawn platform so all 4 feet (~0.6 m wheelbase) start on solid ground
        # -------- PER-BAND spatial gap-difficulty curriculum --------
        # IsaacLab spreads envs on a grid, so there is ONE spawn platform per grid row. The gap width
        # ramps easy->hard within EACH corridor, RESET at every platform's leading edge, so every robot —
        # whatever row it spawns in — starts in a trivial-gap corridor right past its own platform (a
        # global ramp only kept the back row easy, diluting the average). ONLY the gap ramps; strip_w is
        # fixed and _strip_centers_x is stored from the same accumulated centers, so the foothold x-snap
        # stays exact at every (non-uniform) difficulty. Build-time env-var overrides for sweeps.
        gap_min = float(_os.environ.get("GAP_WIDTH_MIN", "") or self.cfg.gap_width_min)
        gap_max = float(_os.environ.get("GAP_WIDTH_MAX", "") or self.cfg.gap_width_max)
        curric_start = float(_os.environ.get("CURRIC_X_START", "") or self.cfg.curric_x_start)

        origins = self._terrain.env_origins.detach().cpu().numpy()  # (N,3)
        ox_min, ox_max = float(origins[:, 0].min()), float(origins[:, 0].max())
        oy_min, oy_max = float(origins[:, 1].min()), float(origins[:, 1].max())
        # trench field bounds: behind spawn a little, forward across the walking corridor, full y width
        x_lo, x_hi = ox_min - 1.5, ox_max + corridor + 1.5
        y_lo, y_hi = oy_min - 1.5, oy_max + 1.5
        y_mid, y_len = 0.5 * (y_lo + y_hi), (y_hi - y_lo)

        # Per-band ramp anchors: one platform band per grid row, leading (forward, +x) edge = ox+0.25+len/2.
        # A strip's corridor = the largest platform edge <= its x (searchsorted); the ramp measures the
        # strip's distance past THAT edge, so difficulty resets at every band.
        plat_ox = np.unique(np.round(origins[:, 0], 3))  # sorted unique grid-row x
        plat_edges = np.sort(plat_ox + 0.25 + 0.5 * plat_len)  # platform leading edges
        if plat_ox.size >= 2:
            row_spacing = float(np.median(np.diff(np.sort(plat_ox))))
            corridor_len = max(0.5, row_spacing - plat_len)  # trench length between consecutive platforms
        else:
            corridor_len = float(corridor)  # single row: the whole forward corridor
        # size the ramp to complete WITHIN a corridor (short corridor -> lower reachable max, still easy
        # near spawn for all rows). Priority: env-var CURRIC_X_RAMP > cfg.curric_x_ramp(>0) > auto.
        _ramp_env = _os.environ.get("CURRIC_X_RAMP", "").strip()
        if _ramp_env:
            curric_ramp = max(1e-6, float(_ramp_env))
        elif float(self.cfg.curric_x_ramp) > 0.0:
            curric_ramp = float(self.cfg.curric_x_ramp)
        else:
            curric_ramp = max(2.0, 0.7 * corridor_len)  # auto: complete ~within one corridor

        # jittered strip centers (deterministic seed -> _strip_centers_x matches the built mesh exactly).
        # The gap AFTER each strip follows ITS corridor's ramp (distance past the platform edge just behind
        # it); jitter is added on top (non-periodic).
        rng = np.random.RandomState(0)
        centers, gaps_used = [], []
        x = x_lo
        while x <= x_hi:
            centers.append(x)
            ei = int(np.searchsorted(plat_edges, x, side="right")) - 1  # nearest platform edge behind x
            edge = plat_edges[ei] if ei >= 0 else plat_edges[0]  # before the first edge -> ramp 0 (easy)
            ramp = min(1.0, max(0.0, (x - edge - curric_start) / curric_ramp))  # 0 just past platform -> 1 far
            gw = gap_min + (gap_max - gap_min) * ramp
            gap = max(0.05, gw + float(rng.uniform(-jitter, jitter)))
            gaps_used.append(gap)
            x += strip_w + gap
        centers_np = np.asarray(centers, dtype=np.float64)
        gaps_np = np.asarray(gaps_used, dtype=np.float64)
        # store for the __init__ banner / diagnostics
        self._gap_corridor_len = corridor_len
        self._gap_curric_ramp = curric_ramp
        self._gap_n_corridors = len(plat_edges)

        boxes = []
        for c in centers_np:  # solid strips (top at z=depth, resting on the plane at z=0)
            b = trimesh.creation.box(extents=(strip_w, y_len, depth))
            b.apply_translation((float(c), y_mid, 0.5 * depth))
            boxes.append(b)
        for ox in np.unique(np.round(origins[:, 0], 3)):  # one platform band per grid row (shared y)
            b = trimesh.creation.box(extents=(plat_len, y_len, depth))
            b.apply_translation((float(ox) + 0.25, y_mid, 0.5 * depth))
            boxes.append(b)
        mesh = trimesh.util.concatenate(boxes)
        self._terrain.import_mesh("gap_strips", mesh)

        self._gap_surface_z = depth  # world z of the strip tops = walking surface for the gap test
        self._strip_centers_x = torch.tensor(
            np.sort(centers_np), dtype=torch.float, device=self.device
        )  # (K,) sorted world-x of solid-strip centers (foothold snap targets); non-uniform under the ramp
        print(
            f"[quad17][gap] built PER-BAND RAMPED trench field: {len(centers_np)} strips over "
            f"x[{x_lo:.1f},{x_hi:.1f}] y[{y_lo:.1f},{y_hi:.1f}] | {len(plat_edges)} platforms/corridors "
            f"corridor_len={corridor_len:.2f}m curric_ramp={curric_ramp:.2f}m start={curric_start:.2f}m | "
            f"gap_w ramp {gap_min:.2f}->{gap_max:.2f} m (actual min={gaps_np.min():.3f} max={gaps_np.max():.3f})"
        )

    def _snap_x_to_strip(self, x: torch.Tensor) -> torch.Tensor:
        """Snap x (any shape) to the nearest solid-strip center (from ``_strip_centers_x``).

        Transverse trenches alternate solid/gap only along world-x, so only x is snapped; y (which runs
        along the full-width strips) is left as the Raibert prediction. Nearest-neighbour via
        ``searchsorted`` on the sorted centers tensor — guarantees the reference foothold lands on a
        physically solid strip, so a foot tracking it will not drop into a trench.
        """
        c = self._strip_centers_x
        flat = x.reshape(-1)
        idx = torch.searchsorted(c, flat).clamp(1, c.numel() - 1)
        left, right = c[idx - 1], c[idx]
        nearest = torch.where((flat - left) <= (right - flat), left, right)
        return nearest.reshape(x.shape)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._gait_phase = (self._gait_phase + self.step_dt / self.cfg.gait_period) % 1.0
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _clock_obs(self) -> torch.Tensor:
        """4-dim trot clock: (sin, cos) for the two diagonal phase groups (offset 0.0 and 0.5)."""
        phi_a = self._gait_phase  # group (HL, FR)
        phi_b = (self._gait_phase + 0.5) % 1.0  # group (HR, FL)
        return torch.stack(
            [
                torch.sin(2.0 * torch.pi * phi_a),
                torch.cos(2.0 * torch.pi * phi_a),
                torch.sin(2.0 * torch.pi * phi_b),
                torch.cos(2.0 * torch.pi * phi_b),
            ],
            dim=1,
        )

    def _foothold_cartesian(self) -> torch.Tensor:
        """28-dim Cartesian foothold reference block (pre-ablation).

        Two world-frame reference footholds (current target + one-step look-ahead) are each expressed
        as a base-frame offset from the root (4 feet x 3 = 12 dims each = 24), followed by the 4
        per-foot target ages (s). Reads directly from ``_foothold_target`` which is why the reset seed
        MUST be a bounded pose-relative value (see ``_reset_idx``) — a stale world target here would be
        stacked into every history slot and poison the adaptation module.
        """
        n = self.num_envs
        base = self._robot.data.root_link_pos_w  # (N,3)
        q = self._robot.data.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)  # (N*4,4)
        rels = []
        for tgt in (self._foothold_target, self._foothold_target2):
            v = (tgt - base[:, None, :]).reshape(-1, 3)  # world offset (N*4,3)
            rels.append(math_utils.quat_rotate_inverse(q, v).reshape(n, 12))  # body frame
        return torch.cat([*rels, self._target_age], dim=-1)  # (N,28)

    def _ik_desired_joint(self) -> torch.Tensor:
        """+12 IK block (pre-ablation): per-leg first-order Jacobian-inverse joint delta toward target.

        For each leg: ``delta_q = J_leg^+ (p_target - p_footcurrent)`` — position-only 3D world error,
        where ``J_leg`` is the 3x3 linear-velocity Jacobian of that leg's sole body w.r.t. its
        hip/thigh/calf joints (world frame). This is the DTC Fig-7D "desired joint position (IK)"
        signal: without it the policy cannot map a Cartesian foothold target to joint actions. Robust
        (SVD pseudo-inverse, no hardcoded link lengths). Computed on the fly from the live articulation
        Jacobian + ``_foothold_target`` (already seeded in ``_reset_idx``), so no new persistent buffer.
        """
        n = self.num_envs
        jac = self._robot.root_physx_view.get_jacobians()  # (N, num_bodies, 6, 6+num_joints) floating base
        jac_lin = jac[:, self._ik_jac_body_ids, 0:3, :]  # 3 linear rows of the 4 sole bodies -> (N,4,3,C)
        idx = self._ik_jac_col_ids[None, :, None, :].expand(n, 4, 3, self._ik_joints_per_leg)  # (N,4,3,3)
        j_leg = torch.gather(jac_lin, dim=3, index=idx)  # per-leg 3x3 linear Jacobian -> (N,4,3rows,3cols)
        p_cur = self._robot.data.body_pos_w[:, self._sole_body_ids, :]  # (N,4,3)
        err = (self._foothold_target - p_cur).unsqueeze(-1)  # world position error (N,4,3,1)
        dq = torch.matmul(torch.linalg.pinv(j_leg), err).squeeze(-1)  # (N,4,3)
        dq = torch.clamp(dq, -torch.pi, torch.pi)  # bound against near-singular pinv blow-ups
        return dq.reshape(n, 4 * self._ik_joints_per_leg)  # (N,12)

    def _foothold_obs(self) -> torch.Tensor:
        """Foothold reference block appended to the policy obs (DTC Phase-2): Cartesian (28) [+ IK (12)].

        The IK block is DERIVED from the same ``_foothold_target`` as the Cartesian block, so the Step-8
        ablation gate MUST blind BOTH consistently or a scrambled policy could recover the true target
        from the IK channel. Compute one across-env permutation / zero and apply it to the concatenated
        block: ``zero`` zeros both, ``scramble`` applies the SAME row permutation to both.
        """
        n = self.num_envs
        blocks = [self._foothold_cartesian()]  # (N,28)
        if self.cfg.foothold_ik_obs:
            blocks.append(self._ik_desired_joint())  # (N,12)
        out = torch.cat(blocks, dim=-1)  # (N,28) or (N,40)
        # Ablation gate (Step 8): blind/scramble the WHOLE block (Cartesian + IK); reward still uses
        # the true buffer. One shared permutation keeps the two target-derived channels consistent.
        if self._foothold_ablate == "zero":
            out = torch.zeros_like(out)
        elif self._foothold_ablate == "scramble":
            out = out[torch.randperm(n, device=self.device)]
        return out

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        clock_obs = self._clock_obs()

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.projected_gravity_b,  # 3
                    self._commands,  # 3
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 17
                    self._robot.data.joint_vel,  # 17
                    self._actions,  # 17
                    clock_obs if self.cfg.clock_inputs else None,  # 4
                    self._foothold_obs() if self.cfg.foothold_obs else None,  # 28 (+12 IK if foothold_ik_obs)
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        if self.cfg.history_observation:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([obs] * self.cfg.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], obs.unsqueeze(1)], dim=1),
            )
            observations["history"] = self.obs_history_buf

        if self.cfg.priv_explicit:
            priv_explicit = torch.cat(
                [
                    self._robot.data.root_lin_vel_b * 2.0,  # 3
                    self._robot.data.root_ang_vel_b * 0.25,  # 3
                ],
                dim=-1,
            )
            observations["priv_explicit"] = priv_explicit
        if self.cfg.priv_latent:
            priv_obs = torch.cat(
                [
                    torch.tensor(self._robot.root_physx_view.get_masses(), device=self.device),
                    torch.tensor(
                        self._robot.root_physx_view.get_material_properties().reshape(self.num_envs, -1),
                        device=self.device,
                    ),
                ],
                dim=-1,
            )
            observations["priv_latent"] = priv_obs

        return observations

    def _regen_footholds(self, mask: torch.Tensor):
        """Re-anchor foothold targets for the masked (env, foot) entries from the CURRENT base pose.

        ``mask``: (N, 4) bool. For each masked foot: capture the current sole pos as the realized
        stance, generate a fresh target (per-foot Raibert placement + decorrelated bounded jitter)
        plus a one-step-ahead ``target2`` (one more Raibert step), and reset that foot's age to 0.

        Per-foot Raibert means no foot reads another foot's just-mutated buffer, so this is a safe
        vectorized where-update (avoids the intra-call ordering hazard). The jitter — decorrelated
        from the command — is MANDATORY: without it target == f(cmd, clock, pose), which the policy
        already observes, so the reward becomes a shortcut the foothold obs is free to ignore.
        """
        if not bool(torch.any(mask)):
            return
        n = self.num_envs
        sole_pos_w = self._robot.data.body_pos_w[:, self._sole_body_ids, :]  # (N,4,3)
        base_pos = self._robot.data.root_link_pos_w  # (N,3)
        yaw_q = math_utils.yaw_quat(self._robot.data.root_quat_w)  # (N,4)

        # base-frame foot offset -> world (yaw only)
        off_b3 = torch.cat([self._foot_offset_b, torch.zeros(4, 1, device=self.device)], dim=1)  # (4,3)
        yaw_q_e = yaw_q[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
        off_w = math_utils.quat_apply(yaw_q_e, off_b3[None].expand(n, -1, -1).reshape(-1, 3)).reshape(n, 4, 3)

        # command velocity (base-frame vx, vy) -> world; Raibert placement offset (0.5 * T_stance * v)
        v_cmd_b = torch.zeros(n, 3, device=self.device)
        v_cmd_b[:, :2] = self._commands[:, :2]
        v_cmd_w = math_utils.quat_apply(yaw_q, v_cmd_b)  # (N,3)
        raibert = (0.5 * self.cfg.foothold_T_stance * v_cmd_w)[:, None, :]  # (N,1,3), z~0

        # Gap test provides the "must read obs" pressure via the terrain, so the decorrelation jitter is
        # disabled (the strip snap below is itself a command-decorrelated nonlinear function of pose).
        jxy, jz = (0.0, 0.0) if self._gap else (self.cfg.foothold_jitter_xy, self.cfg.foothold_jitter_z)
        j1 = torch.empty(n, 4, 3, device=self.device)
        j1[..., 0].uniform_(-jxy, jxy)
        j1[..., 1].uniform_(-jxy, jxy)
        j1[..., 2].uniform_(-jz, jz)
        j2 = torch.empty(n, 4, 3, device=self.device)
        j2[..., 0].uniform_(-jxy, jxy)
        j2[..., 1].uniform_(-jxy, jxy)
        j2[..., 2].uniform_(-jz, jz)

        tgt1 = base_pos[:, None, :] + off_w + raibert + j1  # (N,4,3)
        tgt1[..., 2] = sole_pos_w[..., 2] + j1[..., 2]  # foot target z = ground (sole z), not base z
        tgt2 = tgt1 + raibert + j2  # one more Raibert step ahead

        if self._gap:
            # SNAP each target x to the nearest solid-strip center so the reference foothold is always
            # on physically solid ground; a foot that tracks it stays out of the trenches. z -> strip top.
            tgt1[..., 0] = self._snap_x_to_strip(tgt1[..., 0])
            tgt2[..., 0] = self._snap_x_to_strip(tgt2[..., 0])
            tgt1[..., 2] = self._gap_surface_z
            tgt2[..., 2] = self._gap_surface_z

        m3 = mask.unsqueeze(-1)
        self._stance_pos = torch.where(m3, sole_pos_w, self._stance_pos)
        self._foothold_target = torch.where(m3, tgt1, self._foothold_target)
        self._foothold_target2 = torch.where(m3, tgt2, self._foothold_target2)
        self._target_age = torch.where(mask, torch.zeros_like(self._target_age), self._target_age)

    def _base_ref_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advancing base-pose reference (DTC Eq1), anchored so error = 0 at every episode reset.

        ``t_ref = (episode_length_buf - _base_ref_t0_buf) * step_dt`` restarts at 0 each reset (robust to
        the staggered-start randint on the first full reset), and ``x0/y0`` are the robot's actual base
        xy at reset, so a freshly reset robot starts AT the reference (no spurious step-0 error / death).
        Shared by ``_get_dones`` (progress termination) and ``_get_rewards`` (exp tracking) so both use
        the identical reference within a step (episode_length_buf is not re-incremented between them).
        """
        t_ref = (self.episode_length_buf.float() - self._base_ref_t0_buf) * self.step_dt  # (N,), 0 at reset
        ref_x = self._base_ref_x0 + self._commands[:, 0] * t_ref  # forward-only advancing x reference
        ref_y = self._base_ref_y0  # forward-only: y reference held at spawn y
        return ref_x, ref_y

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / self.cfg.lin_vel_tracking_sigma)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.1)
        # z velocity
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques / accel / action rate
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time (cmd-gated)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.2) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # similar to default
        similar_to_default = torch.sum(
            torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1
        )
        # base height (against explicit target if provided, else default root state)
        target_h = getattr(self.cfg, "base_height_target", None)
        if target_h is None:
            target_h = self._robot.data.default_root_state[:, 2]
        base_height = torch.square(self._robot.data.root_link_pos_w[:, 2] - target_h)
        # termination = base contact
        termination = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        ).float()

        # -------- trot gait phase terms (4 feet) --------
        # per-foot phase = gait_phase + trot offset (0.0 for HL/FR, 0.5 for HR/FL)
        phi = (self._gait_phase.unsqueeze(1) + self._foot_phase_offset.unsqueeze(0)) % 1.0  # (N, 4)
        theta = 2.0 * torch.pi * phi
        E_swing = 0.5 * (1.0 + torch.tanh(self.cfg.gait_phase_sharpness * torch.sin(theta)))  # (N, 4)

        # standing override: near-zero command -> no swing (stand still, don't march in place)
        cmd_xy = torch.norm(self._commands[:, :2], dim=1)
        standing = (cmd_xy < self.cfg.standing_vel_threshold) & (
            self._commands[:, 2].abs() < self.cfg.standing_yaw_threshold
        )
        E_swing = torch.where(standing.unsqueeze(1), torch.zeros_like(E_swing), E_swing)
        E_stance = 1.0 - E_swing

        if self._sole_rest_z is None:
            self._sole_rest_z = self._robot.data.body_pos_w[:, self._sole_body_ids, 2].mean(dim=0).detach()
        # P2: lazily capture the accurate base-frame foot offset (needs body_pos_w), then refine any
        # reset-seeded targets to the true sole pos BEFORE the touchdown reward is scored this step.
        if self._foot_offset_b is None:
            sole0 = self._robot.data.body_pos_w[:, self._sole_body_ids, :]  # (N,4,3)
            yq = math_utils.yaw_quat(self._robot.data.root_quat_w)
            yq_e = yq[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
            rel0 = math_utils.quat_rotate_inverse(
                yq_e, (sole0 - self._robot.data.root_link_pos_w[:, None, :]).reshape(-1, 3)
            ).reshape(self.num_envs, 4, 3)
            self._foot_offset_b = rel0[..., :2].mean(dim=0).detach()  # (4,2)
        if bool(torch.any(self._foothold_needs_refine)):
            self._regen_footholds(self._foothold_needs_refine[:, None].expand(-1, 4))
            self._foothold_needs_refine[:] = False
        contact_filt = torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1.0
        sole_z = self._robot.data.body_pos_w[:, self._sole_body_ids, 2]
        lift = sole_z - self._sole_rest_z

        # (A) stance bonus, (B) swing clearance bonus, (C) anti-slip penalty
        gait_stance = torch.sum(E_stance * contact_filt.float(), dim=1)  # (N,) in [0, 4]
        clearance = torch.clamp(lift / self.cfg.gait_swing_height, 0.0, 1.0)
        gait_swing = torch.sum(E_swing * clearance, dim=1)
        sole_vel_xy = torch.norm(self._robot.data.body_lin_vel_w[:, self._sole_body_ids, :2], dim=-1)
        slip_pen = torch.sum(contact_filt.float() * sole_vel_xy**2, dim=1)

        # (P2 gap-test) foot-in-trench penalty: a stance foot whose sole dropped well below the strip
        # surface has physically fallen into a gap. Mild, physically-grounded (it measures the real
        # drop); termination still comes from base contact when the drop destabilizes the robot.
        if self._gap:
            drop = torch.clamp(self._gap_surface_z - sole_z, min=0.0)  # (N,4) depth below strip top
            foot_in_gap = torch.sum(contact_filt.float() * (drop > 0.5 * self.cfg.gap_depth).float() * drop, dim=1)
        else:
            foot_in_gap = torch.zeros(self.num_envs, device=self.device)

        # -------- (P2) touchdown foothold-tracking reward --------
        # Gate on a first-contact event during the foot's scheduled stance phase (a touchdown). The
        # bonus is -log(err2 + eps) on the XY distance from the sole to its target; XY-only removes the
        # sole_rest_z bias, and the err2 clamp stops a single far-miss dominating the unbounded log.
        foothold_gate = first_contact.float() * (E_stance > 0.5).float()  # (N,4)
        sole_xy = self._robot.data.body_pos_w[:, self._sole_body_ids, :2]  # (N,4,2)
        foot_err2 = torch.clamp(
            torch.sum((sole_xy - self._foothold_target[..., :2]) ** 2, dim=-1), max=self.cfg.foothold_err2_max
        )  # (N,4)
        foothold_track = torch.sum(foothold_gate * (-torch.log(foot_err2 + self.cfg.foothold_track_eps)), dim=1)
        foothold_place_err = torch.sum(foothold_gate * foot_err2, dim=1)  # diagnostic (clamped m^2)
        # eval-only running totals (ablation gate); do not feed back into the reward
        self._eval_td_count += foothold_gate.sum()
        self._eval_track_sum += foothold_track.sum()
        self._eval_err_sum += foothold_place_err.sum()

        # -------- (DTC Eq1) advancing base-pose (position) tracking reward --------
        # Reference base trajectory advances FORWARD at the commanded velocity (see _base_ref_xy):
        #   x_ref = x0 + cmd_vx * t_ref ,  y_ref = y0   (forward-only gap test: vy≈yaw≈0)
        # Reward = exp(-||base_xy - ref_xy||^2 / sigma) (DTC Eq 1 form, position order n=0). This handles
        # FINE tracking near the reference; far from it the exp gradient vanishes, so the linear progress
        # reward below (constant forward gradient) + progress termination in _get_dones do the pulling.
        base_x = self._robot.data.root_link_pos_w[:, 0]  # (N,)
        ref_x, ref_y = self._base_ref_xy()
        base_pose_err2 = (base_x - ref_x) ** 2 + (self._robot.data.root_link_pos_w[:, 1] - ref_y) ** 2  # (N,)
        base_pose_track = torch.exp(-base_pose_err2 / self.cfg.base_pose_track_sigma)  # (N,) in (0,1]

        # -------- (DTC linear term) forward-progress reward: constant forward gradient --------
        # exp() has ~zero gradient once the robot has fallen behind, so it cannot pull a hesitant robot
        # forward. Add a LINEAR reward on per-step forward base displacement (i.e. forward velocity),
        # clipped to [0, 1.5*cmd_vx] so it cannot be gamed by lunging and standing still earns nothing.
        # _prev_base_x is set to the spawn x in _reset_idx and advanced at the end of this method.
        fwd_dx = base_x - self._prev_base_x  # (N,) per-step forward displacement
        fwd_cap = torch.clamp(self._commands[:, 0], min=0.0) * self.step_dt * 1.5  # per-env clip ceiling
        progress = torch.minimum(torch.clamp(fwd_dx, min=0.0), fwd_cap) / self.step_dt  # (N,) fwd vel [0,1.5cmd]

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "similar_to_default": similar_to_default * self.cfg.similar_to_default_reward_scale * self.step_dt,
            "base_height": base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            "gait_stance": gait_stance * self.cfg.gait_stance_reward_scale * self.step_dt,
            "gait_swing": gait_swing * self.cfg.gait_swing_reward_scale * self.step_dt,
            "foot_slip": slip_pen * self.cfg.foot_slip_reward_scale * self.step_dt,
            "foothold_track": foothold_track * self.cfg.foothold_track_reward_scale * self.step_dt,
            "foot_in_gap": foot_in_gap * self.cfg.foot_in_gap_reward_scale * self.step_dt,
            "base_pose_track": base_pose_track * self.cfg.base_pose_track_reward_scale * self.step_dt,
            "progress": progress * self.cfg.progress_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        self.curriculum_rew_buf += reward
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["foothold_place_err"] += foothold_place_err  # diagnostic, not summed into reward

        # -------- (P2) advance foothold targets AFTER the reward is scored --------
        # New touchdown OR age fallback (a scheduled contact that never fired can't stall the target).
        force_advance = self._target_age > self.cfg.foothold_max_age  # (N,4)
        new_touchdown = first_contact.bool() & (E_stance > 0.5)  # (N,4)
        self._regen_footholds(new_touchdown | force_advance)
        self._target_age += self.step_dt
        # advance the progress-reward reference (envs that reset this step get overwritten in _reset_idx)
        self._prev_base_x = base_x.detach().clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        if self._gap:
            # safety net: base sank below the strip surface (fell into a trench) even without a base
            # contact force spike — treat as a fall so the episode ends and the drop is penalized.
            died = died | (self._robot.data.root_link_pos_w[:, 2] < self._gap_surface_z)
        # (v3) progress termination: removes the "stand still is safe" harbor. If the base falls more
        # than progress_fail_dist behind the advancing reference, kill the episode -> standing still
        # (falling behind) becomes death, forcing forward motion. Because the reference is anchored so
        # t_ref=0 at reset (see _base_ref_xy), a fresh robot has err≈0 and this does NOT fire at step 0.
        ref_x, _ = self._base_ref_xy()
        died = died | ((ref_x - self._robot.data.root_link_pos_w[:, 0]) > self.cfg.progress_fail_dist)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        if self.cfg.history_observation:
            self.obs_history_buf[env_ids, :, :] = 0.0
        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        if self._gap:
            # Gap test: spawn on the solid strip surface (raise by strip height) and add a per-episode
            # forward offset (re-randomized here -> different phase vs. the static trench pattern each
            # reset). The wide spawn platform keeps all four feet on solid ground for this offset range.
            self._spawn_dx[env_ids] = torch.rand(len(env_ids), device=self.device) * self._spawn_dx_max
            default_root_state[:, 2] += self._gap_surface_z
            default_root_state[:, 0] += self._spawn_dx[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -------- (P2) seed obs-visible foothold targets --------
        # DirectRLEnv order is _get_rewards -> _reset_idx (no sim step) -> _get_observations, so the
        # FIRST post-reset obs reads these targets. Seed a BOUNDED pose-relative value computable
        # WITHOUT body_pos_w (nominal offset rotated by spawn yaw); a stale world target would be
        # stacked into every history slot and poison the RMA adaptation module. _foothold_needs_refine
        # makes the first post-reset _get_rewards replace it with the true sole pos.
        n_ids = len(env_ids)
        seed_base = default_root_state[:, :3]  # world spawn xyz (env origins already added above)
        # DTC Eq1 base-pose reference anchor = the ACTUAL spawn xy (env origin + gap forward offset).
        # MUST be set here so x_ref/y_ref restart from the robot's true position every episode.
        self._base_ref_x0[env_ids] = default_root_state[:, 0]
        self._base_ref_y0[env_ids] = default_root_state[:, 1]
        # Capture episode_length_buf AFTER the staggered-start randint above so t_ref = (buf - t0)*dt = 0
        # at this reset -> the advancing reference starts AT the robot (err=0), never ahead of it.
        self._base_ref_t0_buf[env_ids] = self.episode_length_buf[env_ids].float()
        # Seed the linear progress reference to the spawn x (first-step forward displacement ≈ 0).
        self._prev_base_x[env_ids] = default_root_state[:, 0]
        yaw_q0 = math_utils.yaw_quat(default_root_state[:, 3:7])  # (n,4)
        off_b3 = torch.cat([self._nominal_foot_offset, torch.zeros(4, 1, device=self.device)], dim=1)  # (4,3)
        yaw_e = yaw_q0[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
        off_w = math_utils.quat_apply(yaw_e, off_b3[None].expand(n_ids, -1, -1).reshape(-1, 3)).reshape(n_ids, 4, 3)
        seed = seed_base[:, None, :] + off_w  # (n,4,3)
        ground_z = self._terrain.env_origins[env_ids][:, 2]
        if self._gap:
            ground_z = ground_z + self._gap_surface_z  # strip-top surface, not the trench floor
        seed[..., 2] = ground_z[:, None].expand(-1, 4)  # ground z, not base z
        self._foothold_target[env_ids] = seed
        self._foothold_target2[env_ids] = seed
        self._stance_pos[env_ids] = seed
        self._target_age[env_ids] = 0.0
        self._foothold_needs_refine[env_ids] = True
        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        self._commands[env_ids, 0] = torch_rand_float(
            *self.cfg.command_cfg["lin_vel_x_range"], (len(env_ids),), self.device
        )
        self._commands[env_ids, 1] = torch_rand_float(
            *self.cfg.command_cfg["lin_vel_y_range"], (len(env_ids),), self.device
        )
        self._commands[env_ids, 2] = torch_rand_float(
            *self.cfg.command_cfg["ang_vel_range"], (len(env_ids),), self.device
        )
        # force a fraction of envs to standing (cmd=0) so the policy learns the cmd=0 equilibrium
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_standing_envs
        self._commands[env_ids[standing], :] = 0.0

    # -- Debug visualization (command / measured velocity arrows) --
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(
                    GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
                )
                self.current_vel_visualizer = VisualizationMarkers(
                    BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
                )
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity):
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
