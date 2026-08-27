# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 ParkourImitation-RandomGoal environment.

Extends Go2ParkourImitationEnv with a per-episode "360° random goal" mode that
activates for a subset of curriculum-graduated environments.  Goal positions are
sampled uniformly at a random yaw and distance around the robot's current base XY,
with rejection sampling against the global height field to avoid placing goals in
voids, gaps, or large elevation changes.

Inheritance chain:
    Go2ParkourImitationRandomGoalEnv
        → Go2ParkourImitationEnv      (AMP buffers, flat-env mask)
            → Go2ParkourEnv           (terrain curriculum, goal system, obs)

New buffers (all reset in _reset_idx):
    _graduated           [N] bool  — True once env has graduated to max terrain level.
    _random_goal_mode    [N] bool  — True in the current episode if random-goal active.
    _random_goals_reached [N] long — Number of random goals reached this episode.

Contract (obs / AMP / priv_latent):
    - obs dimensions unchanged: policy(235) + scan(187) + priv_explicit(3) + priv_latent(33) + history
    - amp_obs shape (N, 490) unchanged — AMP path is fully delegated to parent.
    - priv_latent = 33 (immutable, verified in parent).
    - No new obs keys added.
"""

from __future__ import annotations

import math

import torch

from isaaclab_tasks.direct.parkour.parkour_env_cfg import TERRAIN_CLASS_FLAT

from .parkour_imitation_env import Go2ParkourImitationEnv
from .parkour_imitation_random_goal_env_cfg import ParkourImitationRandomGoalEnvCfg


class Go2ParkourImitationRandomGoalEnv(Go2ParkourImitationEnv):
    """Parkour-Imitation environment with curriculum-gated 360° random goal mode.

    Random-goal mode is activated per-episode for curriculum-graduated envs at rate
    ``cfg.random_goal_graduated_ratio``.  In random-goal mode the fixed terrain
    waypoints are replaced by successive randomly-placed goals sampled around the
    robot's current position.  The episode succeeds when
    ``cfg.num_random_goals`` consecutive goals have been reached.

    When ``cfg.random_goal_force_flat`` is True, random-goal envs are forced onto
    flat terrain columns so yaw exploration is unconfounded by obstacles.

    All AMP mechanics (terminal snapshot, ring-buffer, flat-env mask, runner interface)
    are inherited unchanged from Go2ParkourImitationEnv.  This class only overrides
    ``_reset_idx``, ``_update_terrain_curriculum``, ``_update_goals``, and adds the
    ``_sample_random_goal`` helper.
    """

    cfg: ParkourImitationRandomGoalEnvCfg

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, cfg: ParkourImitationRandomGoalEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ creates all parent buffers including _env_goals, _current_goal_idx,
        # _terrain_levels, _terrain_types, _col_to_class, AMP bufs, etc.
        super().__init__(cfg, render_mode, **kwargs)

        # ── New per-env buffers ───────────────────────────────────────────────
        # _graduated: True once this env has reached the max terrain level.
        #   Permanent — never reset to False; graduation is irreversible.
        self._graduated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # _random_goal_mode: True in the current episode for this env.
        #   Re-drawn each episode reset for graduated envs.
        self._random_goal_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # _random_goals_reached: count of random goals reached this episode.
        self._random_goals_reached = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── Precompute flat-terrain column set ───────────────────────────────
        # _col_to_class is built by the grandparent using proportion cumsum.
        # We need the set of columns that map to TERRAIN_CLASS_FLAT for the
        # forced-flat override in _update_terrain_curriculum.
        self._flat_cols = (self._col_to_class == TERRAIN_CLASS_FLAT).nonzero(as_tuple=False).squeeze(-1)
        # Sanity: there must be at least one flat column.
        assert self._flat_cols.numel() > 0, (
            "No flat-terrain columns found in _col_to_class.  "
            "Ensure parkour_flat has proportion > 0 in the terrain cfg."
        )

    # ------------------------------------------------------------------
    # Terrain curriculum override — graduation detection + flat override
    # ------------------------------------------------------------------

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        """Run parent curriculum then detect graduations and optionally force flat terrain.

        Graduation detection:
            The parent wraps ``level`` to a random value in [0, max_level] whenever
            ``level >= max_level`` after ``level += move_up - move_down``.  We cannot
            inspect the per-env move_up/down tensors directly (they are local to the
            parent), but we CAN compare levels before and after the parent call:
            - If level DECREASED (wrap happened) and the env was at max_level before the
              call, it graduated.
            - Alternatively: after the parent call, any env whose level was boosted past
              max_level and got wrapped has ``level < pre_level`` (wrap-down signal).
            The safest criterion that mirrors the parent condition exactly:
              pre_level >= max_level  →  the env was at or above ceiling before the call,
              meaning move_up must have triggered the wrap.  That IS graduation.

            NOTE: ``init_done`` guard is inherited from parent — this method is a no-op
            on the very first reset (parent returns early).

        Flat override:
            If ``cfg.random_goal_force_flat`` and the env is in random-goal mode
            (which is set before this method is called from _reset_idx), override
            ``_terrain_types`` to a randomly chosen flat column and refresh
            ``env_origins`` so the robot spawns on flat ground.
            _init_env_goals runs immediately after in the grandparent's _reset_idx
            and will use the updated _terrain_types.
        """
        if not getattr(self, "init_done", False):
            super()._update_terrain_curriculum(env_ids)
            return

        max_level = self.cfg.terrain.terrain_generator.num_rows - 1

        # Capture levels BEFORE parent call.
        pre_levels = self._terrain_levels[env_ids].clone()

        # Run parent curriculum (modifies _terrain_levels[env_ids] and env_origins in-place).
        super()._update_terrain_curriculum(env_ids)

        # ── Graduation detection ─────────────────────────────────────────────
        # An env graduated if its pre-call level was >= max_level (the only way to
        # trigger the wrap branch inside the parent).  Once graduated, the flag is
        # permanent (OR-assign, never cleared).
        graduated_mask = pre_levels >= max_level  # [n] bool
        if graduated_mask.any():
            self._graduated[env_ids[graduated_mask]] = True

        # ── Flat terrain override for random-goal envs ───────────────────────
        if self.cfg.enable_random_goal and self.cfg.random_goal_force_flat:
            rg_mask = self._random_goal_mode[env_ids]  # [n] bool
            rg_env_ids = env_ids[rg_mask]
            if rg_env_ids.numel() > 0:
                # Sample a random flat column for each random-goal env.
                n_rg = rg_env_ids.numel()
                rand_col_indices = torch.randint(0, self._flat_cols.numel(), (n_rg,), device=self.device)
                chosen_flat_cols = self._flat_cols[rand_col_indices]
                self._terrain_types[rg_env_ids] = chosen_flat_cols
                # Refresh env_origins so _init_env_goals and robot spawn use the flat patch.
                self._terrain.env_origins[rg_env_ids] = self._terrain.terrain_origins[
                    self._terrain_levels[rg_env_ids], self._terrain_types[rg_env_ids]
                ]

    # ------------------------------------------------------------------
    # Reset override
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset with random-goal mode assignment and new buffer initialisation.

        Call ordering (critical — must preserve AMP and parkour invariants):

        [Parent imitation env ordering]
        0.  Terminal AMP snapshot (pre-super, robot state intact).

        [This override, PRE-super]
        A.  Per-episode random-goal mode assignment for graduated envs.
            Must happen BEFORE super() so that _update_terrain_curriculum (called
            inside the grandparent's _reset_idx via super chain) can see
            _random_goal_mode and apply the flat override.

        [super()._reset_idx — runs imitation parent which runs grandparent]
        1.  AMP terminal snapshot (parent step 0).
        2.  Grandparent _reset_idx:
              buffers → _update_terrain_curriculum (where flat override occurs)
              → _env_class refresh → _init_env_goals → robot write → _resample_commands.
        3.  AMP buf clear + _update_flat_env_mask.

        [Post-super]
        B.  For random-goal envs: replace _init_env_goals result with first random goal.
        C.  Initialise new buffers (_graduated permanent, _random_goals_reached zero).
        """
        # Normalise env_ids early — needed for pre-super assignment.
        # Assign to a local non-Optional variable so Pyright narrows the type.
        if env_ids is None:
            ids: torch.Tensor = torch.arange(self.num_envs, device=self.device)
        else:
            ids = env_ids

        # ── A. Per-episode random-goal mode assignment ────────────────────────
        # Assign BEFORE super() so _update_terrain_curriculum sees the mode.
        # Only activated when master switch is on.
        if self.cfg.enable_random_goal:
            # Always clear mode for all reset envs first.
            self._random_goal_mode[ids] = False
            # Graduated envs get a per-episode coin-flip.
            # Viewer-forced envs (_skip_curriculum[ids] == True) are excluded so
            # _change_terrain_for_viewer's forced terrain type/level is not overwritten
            # by the flat-override block in _update_terrain_curriculum.
            #
            # Restrict random-goal mode to graduated envs that are ALREADY on flat terrain
            # (self._env_class == TERRAIN_CLASS_FLAT). This prevents draining graduated envs
            # off the challenging terrains (hurdle/gap/stair/step) onto flat — only flat-resident
            # graduated envs do random-goal (yaw) training, so those hard terrains keep their
            # robots instead of visibly emptying into the flat column.
            grad_mask = (
                self._graduated[ids] & ~self._skip_curriculum[ids] & (self._env_class[ids] == TERRAIN_CLASS_FLAT)
            )  # [n] bool
            grad_env_ids = ids[grad_mask]
            if grad_env_ids.numel() > 0:
                coin = torch.rand(grad_env_ids.numel(), device=self.device)
                self._random_goal_mode[grad_env_ids] = coin < self.cfg.random_goal_graduated_ratio
        else:
            self._random_goal_mode[ids] = False

        # ── 0–3. Delegate to parent imitation env (AMP + grandparent parkour) ─
        # Pass the concrete tensor (parent also handles None internally but we
        # already normalised above; passing None would re-normalise harmlessly,
        # but passing the tensor avoids the double normalisation cost).
        super()._reset_idx(ids)

        # ── B. Random-goal env: replace terrain goals with first random goal ──
        if self.cfg.enable_random_goal:
            rg_mask = self._random_goal_mode[ids]
            rg_env_ids = ids[rg_mask]
            if rg_env_ids.numel() > 0:
                # Reset goal index to 0 so _gather_cur_goals points to slot 0.
                self._current_goal_idx[rg_env_ids] = 0
                # Sample the first random goal and write into _env_goals.
                self._sample_random_goal(rg_env_ids)

        # ── C. Reset new buffers ──────────────────────────────────────────────
        # _graduated is PERMANENT — never cleared on reset.
        self._random_goals_reached[ids] = 0
        # _random_goal_mode already set in step A above (no re-clear needed).

    # ------------------------------------------------------------------
    # Random goal sampling
    # ------------------------------------------------------------------

    def _sample_random_goal(self, env_ids: torch.Tensor) -> None:
        """Sample a random goal around each env's current robot base XY.

        Samples angle within the robot's forward cone and distance ~ U(dist_min, dist_max)
        to generate candidate XY positions in world frame.

        Direction sampling:
            angle = heading_w[env] + U(-half_cone, +half_cone)
            where half_cone = radians(cfg.random_goal_forward_cone_deg) * 0.5.
            heading_w is the world-frame yaw scalar [N] from RigidBodyData.
            When cfg.random_goal_forward_cone_deg == 360, this reduces to omnidirectional.

        Each candidate is validated against the global height field:
            1. Must lie within the height field grid bounds.
            2. |terrain_height(candidate_xy) - terrain_height(robot_xy)| ≤ max_height_diff.

        The offset is re-sampled inside the rejection loop on every attempt, so
        rejected candidates cannot drift outside the forward cone on retry.

        On fallback (all attempts exhausted): a fresh cone-constrained sample is taken
        ignoring the height-diff constraint — the direction stays within the forward cone.

        Writes result directly into:
            self._env_goals[env_ids, 0]   ← goal world position [x, y, terrain_h + z_offset]
            self._env_goals[env_ids, 1:]  ← repeated for future-goal obs consistency.

        Args:
            env_ids: 1-D tensor of environment indices to sample goals for.
        """
        n = env_ids.numel()
        if n == 0:
            return

        dist_min, dist_max = self.cfg.random_goal_dist_range
        z_offset = self.cfg.random_goal_z_offset
        max_h_diff = self.cfg.random_goal_max_height_diff
        max_tries = self.cfg.random_goal_max_sample_tries

        # Forward-cone half-angle [rad].  360° → full circle (backward compatible).
        half_cone = math.radians(self.cfg.random_goal_forward_cone_deg) * 0.5

        # Robot base XY in world frame — use current positions (post-physics, pre-reset robot write).
        # After super()._reset_idx(), robot has been repositioned to env_origins.
        # We use env_origins as the robot's known spawn position.
        robot_xy = self._terrain.env_origins[env_ids, :2]  # [n, 2] world frame

        # Robot world-frame yaw (scalar, radians) — same source as _update_goals/_target_yaw.
        robot_heading = self._robot.data.heading_w[env_ids]  # [n]

        # Height field metadata (built during __init__ by grandparent).
        origin = self._edge_mask_origin  # [2] (x, y) world coords of grid cell (0,0)
        inv_scale = self._edge_mask_inv_scale  # cells per metre
        hf = self._edge_mask_height_field  # [H, W] float32 height in world metres
        hf_rows, hf_cols = hf.shape

        # Robot terrain height (for height-diff check).
        robot_ix = ((robot_xy[:, 0] - origin[0]) * inv_scale).long().clamp(0, hf_rows - 1)
        robot_iy = ((robot_xy[:, 1] - origin[1]) * inv_scale).long().clamp(0, hf_cols - 1)
        robot_h = hf[robot_ix, robot_iy]  # [n]

        # Acceptance tracking.
        accepted = torch.zeros(n, dtype=torch.bool, device=self.device)
        goal_xy = robot_xy.clone()  # fallback = robot position (overwritten when accepted)
        goal_z = robot_h + z_offset  # fallback height

        # Vectorised rejection sampling over max_tries rounds.
        for _attempt in range(max_tries):
            # Only re-sample envs not yet accepted.
            pending = (~accepted).nonzero(as_tuple=False).squeeze(-1)
            if pending.numel() == 0:
                break

            # Sub-batch for pending envs only.
            n_pending = pending.numel()
            # Re-use robot_xy / heading for pending subset.
            pend_robot_xy = robot_xy[pending]  # [n_p, 2]
            pend_robot_h = robot_h[pending]  # [n_p]
            pend_heading = robot_heading[pending]  # [n_p]

            # Sample forward-cone direction for pending batch.
            # offset is re-sampled every attempt so rejected envs stay within the cone.
            offsets = (torch.rand(n_pending, device=self.device) * 2.0 - 1.0) * half_cone
            angles = pend_heading + offsets  # world-frame angle [n_p]
            dists = dist_min + torch.rand(n_pending, device=self.device) * (dist_max - dist_min)
            dx = dists * torch.cos(angles)
            dy = dists * torch.sin(angles)
            cand_xy = pend_robot_xy + torch.stack([dx, dy], dim=-1)  # [n_p, 2]

            # Grid indices.
            ix = ((cand_xy[:, 0] - origin[0]) * inv_scale).long()
            iy = ((cand_xy[:, 1] - origin[1]) * inv_scale).long()

            # Bounds check.
            in_bounds = (ix >= 0) & (ix < hf_rows) & (iy >= 0) & (iy < hf_cols)

            # Height field lookup for in-bounds candidates.
            cand_h = torch.zeros(n_pending, device=self.device)
            if in_bounds.any():
                ix_v = ix[in_bounds].clamp(0, hf_rows - 1)
                iy_v = iy[in_bounds].clamp(0, hf_cols - 1)
                cand_h[in_bounds] = hf[ix_v, iy_v]

            # Height diff check.
            h_ok = (cand_h - pend_robot_h).abs() <= max_h_diff

            # Accept if in-bounds AND height-diff OK.
            ok = in_bounds & h_ok

            # Write accepted results back into full-batch tensors.
            newly_accepted = pending[ok]
            if newly_accepted.numel() > 0:
                goal_xy[newly_accepted] = cand_xy[ok]
                goal_z[newly_accepted] = cand_h[ok] + z_offset
                accepted[newly_accepted] = True

            # On last attempt: fallback — cone-constrained sample, height-diff relaxed.
            if _attempt == max_tries - 1:
                still_pending = (~accepted).nonzero(as_tuple=False).squeeze(-1)
                if still_pending.numel() > 0:
                    n_fb = still_pending.numel()
                    fb_robot_xy = robot_xy[still_pending]
                    fb_robot_h = robot_h[still_pending]
                    fb_heading = robot_heading[still_pending]
                    # Direction stays within the forward cone — no backward fallback.
                    fb_offsets = (torch.rand(n_fb, device=self.device) * 2.0 - 1.0) * half_cone
                    fb_angles = fb_heading + fb_offsets
                    fb_dists = dist_min + torch.rand(n_fb, device=self.device) * (dist_max - dist_min)
                    fb_dx = fb_dists * torch.cos(fb_angles)
                    fb_dy = fb_dists * torch.sin(fb_angles)
                    fb_cand = fb_robot_xy + torch.stack([fb_dx, fb_dy], dim=-1)
                    fb_ix = ((fb_cand[:, 0] - origin[0]) * inv_scale).long()
                    fb_iy = ((fb_cand[:, 1] - origin[1]) * inv_scale).long()
                    fb_ib = (fb_ix >= 0) & (fb_ix < hf_rows) & (fb_iy >= 0) & (fb_iy < hf_cols)
                    fb_h = fb_robot_h.clone()
                    if fb_ib.any():
                        fb_h[fb_ib] = hf[fb_ix[fb_ib].clamp(0, hf_rows - 1), fb_iy[fb_ib].clamp(0, hf_cols - 1)]
                    goal_xy[still_pending] = fb_cand
                    goal_z[still_pending] = fb_h + z_offset

        # ── Write goal into _env_goals ────────────────────────────────────────
        # Slot 0 = current goal (pointed at by _current_goal_idx == 0).
        goal_pos = torch.cat([goal_xy, goal_z.unsqueeze(-1)], dim=-1)  # [n, 3]
        self._env_goals[env_ids, 0] = goal_pos

        # Fill all remaining slots (future-goal obs lookahead) with the same goal.
        # This keeps _gather_cur_goals(future=1...) consistent and prevents stale
        # terrain-waypoint values from leaking into obs for random-goal envs.
        num_total_slots = self._env_goals.shape[1]  # num_goals + num_future_goal_obs
        for slot in range(1, num_total_slots):
            self._env_goals[env_ids, slot] = goal_pos

    # ------------------------------------------------------------------
    # Goal update override — random-goal succession and success detection
    # ------------------------------------------------------------------

    def _update_goals(self) -> None:
        """Goal update with random-goal succession for active random-goal envs.

        For non-random-goal envs: delegates entirely to parent (terrain waypoints).
        For random-goal envs:
            - Checks distance to current goal (slot 0 in _env_goals).
            - On reach + hold timer: increments _random_goals_reached, samples new goal.
            - Success: _random_goals_reached >= num_random_goals → sets _term_goal_reached.
            - _term_goal_reached is OR-combined so parent's terrain-waypoint success
              (for non-random envs) is preserved correctly.

        Notes:
            - _target_pos_rel, _target_yaw, _cur_goals are set by the parent call for
              non-random envs and must be correctly set for random envs too.
              We call super() unconditionally (it sets them for ALL envs) then correct
              _term_goal_reached for random envs.  This avoids duplicating the yaw/rel
              computation and maintains correctness for mixed batches.
        """
        # Disable master switch path.
        if not self.cfg.enable_random_goal or not self._random_goal_mode.any():
            super()._update_goals()
            return

        # Run the full parent goal update for ALL envs.
        # This correctly handles non-random-goal envs and sets _cur_goals /
        # _target_pos_rel / _target_yaw for all envs (random-goal envs included,
        # using whatever is in _env_goals[env, 0]).
        super()._update_goals()

        # ── Random-goal-specific post-processing ─────────────────────────────
        rg_mask = self._random_goal_mode  # [N] bool — active random-goal envs
        if not rg_mask.any():
            return

        hold_time_steps = int(self.cfg.reach_goal_delay / self.step_dt)
        base_xy = self._robot.data.root_link_pos_w[:, :2]  # [N, 2]

        # Current goal XY for random-goal envs (slot 0, written by _sample_random_goal).
        cur_goal_xy = self._env_goals[:, 0, :2]  # [N, 2]
        dist = torch.norm(base_xy - cur_goal_xy, dim=1)  # [N]
        within_threshold = dist < self.cfg.next_goal_threshold  # [N]

        # Hold timer fires when timer > hold_time_steps (matches parent convention).
        hold_fired = self._reach_goal_timer > hold_time_steps  # [N]

        # Envs that just completed a random goal this step.
        just_reached = rg_mask & within_threshold & hold_fired  # [N]

        if just_reached.any():
            reached_ids = just_reached.nonzero(as_tuple=False).squeeze(-1)

            # Increment consecutive goal counter.
            self._random_goals_reached[reached_ids] += 1

            # Reset hold timer for these envs (parent may have already advanced it for
            # non-random envs; we correct here for the random-goal subset only).
            self._reach_goal_timer[reached_ids] = 0

            # Check success condition.
            success = self._random_goals_reached[reached_ids] >= self.cfg.num_random_goals
            success_ids = reached_ids[success]
            if success_ids.numel() > 0:
                # Mark as terminal — bootstrapped by DirectRLEnv / PPO as time_out.
                self._term_goal_reached[success_ids] = True

            # For envs that reached a goal but haven't finished the sequence yet:
            # sample a new random goal.
            not_done_ids = reached_ids[~success]
            if not_done_ids.numel() > 0:
                self._sample_random_goal(not_done_ids)
                # Keep _current_goal_idx at 0 (ring-buffer convention for random-goal mode).
                self._current_goal_idx[not_done_ids] = 0

        # Suppress the parent's _term_goal_reached signal for random-goal envs that
        # have NOT yet completed the full sequence.  The parent sets _term_goal_reached
        # when _current_goal_idx >= num_goals-1 AND hold fires — this fires for random
        # envs because we keep _current_goal_idx at 0 and num_goals may be small.
        # Override: for random envs, _term_goal_reached is managed exclusively above.
        rg_not_done = rg_mask & ~self._term_goal_reached
        if rg_not_done.any():
            self._term_goal_reached[rg_not_done] = False
