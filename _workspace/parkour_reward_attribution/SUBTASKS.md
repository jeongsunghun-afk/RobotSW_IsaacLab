# Reward Attribution Tool — Stage-2 Subtasks

Source of truth for design: `/home/lgb/IsaacLab/_workspace/parkour_reward_attribution/DESIGN.md`.
Read it first; this file is the dispatch table for the next stage's workers.

Conventions:
- Use absolute paths.
- Cite file:line for every claim.
- Don't touch `source/isaaclab/` core.
- Don't add anything to the policy observation space.
- All commands launch via `./isaaclab.sh -p`, not bare `python`.

---

## Subtask S1 — Add per-env reward-breakdown buffer to ParkourEnv

**Owner worker type:** `obs-worker` (env-side code change to a `*_env.py`).

**Why obs-worker (not reward-worker):** the change is a *read-only logging buffer*, not a reward
function modification. The reward computation, scales, and total reward all remain bit-identical.
obs-worker owns the env-side data buffers (per the registry in `/home/lgb/IsaacLab/CLAUDE.md`).

**Files to modify:**
- `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`
  - In `__init__()` near line 187-189 (next to `_last_reward_breakdown_env0`), add:
    - `self._last_reward_breakdown_per_env: torch.Tensor(num_envs, K)` (zeros, on `self.device`).
    - `self._reward_term_names: list[str] = list(self.cfg.reward_scales.keys())` (canonical order).
  - In `_get_rewards()` for-loop at lines 1157-1161, switch to enumerate and write:
    - `self._last_reward_breakdown_per_env[:, i] = scaled`.
  - Do **not** add to `_reset_idx`; the buffer is fully overwritten each step (same convention as
    `_last_reward_breakdown_env0`).

**Files to NOT modify:**
- `parkour_env_cfg.py` — no cfg changes; this is purely an internal env buffer.
- The reward computation itself (lines 1136-1153 dict, line 1158 scaling formula).
- The contact-sensor cfg.
- Any policy obs assembly (sim-to-real constraint).

**Acceptance criteria:**
1. `assert env._last_reward_breakdown_per_env.shape == (env.num_envs, 16)` after one step.
2. For env 0 and every term key, `env._last_reward_breakdown_per_env[0, idx_of(key)]` equals
   `env._last_reward_breakdown_env0[key]` (consistency with existing env0 dict).
3. Training run via `train.py --task Go2-Parkour-Direct-v0 --max_iterations 2` completes without
   error and the WandB / TF episodic-reward curves are bit-identical to the same seed before the
   change (use `Episode_Reward/*` keys for the comparison).
4. `validate-code` worker confirms no observation/cfg/reward shape regressions.

**Dependencies:** none. This is the foundational change; S2 depends on it.

---

## Subtask S2 — Create `play_reward_attribution.py` MVP script

**Owner worker type:** `oh-my-claudecode:executor` (Sonnet; standard scripting work). The script
is a play-script fork — not a worker-defined env/algo change, so no `*-worker` is the natural fit.

**Files to create:**
- `/home/lgb/IsaacLab/scripts/reinforcement_learning/rsl_rl/play_reward_attribution.py`

**Files NOT to modify:**
- `play.py` itself (sibling script approach per DESIGN.md §7).
- `parkour_env.py` / `parkour_env_cfg.py` (env-side hooks are S1's job).

**Script must implement (per DESIGN.md §3–§7):**
1. CLI: `--task Go2-Parkour-Direct-v0` (only this task supported; error out for others);
   `--num_envs` (default 5, validated against `num_active_terrains`); `--difficulty` (default 5);
   inherits `--load_run`, `--checkpoint`, `--device` from `cli_args.add_rsl_rl_args` (cf.
   play.py:44).
2. Hydra-wrapped `main(env_cfg, agent_cfg)` (cf. play.py:202-203).
3. Cfg overrides BEFORE `gym.make`:
   - `env_cfg.scene.num_envs = num_active_terrains`
   - `env_cfg.terrain_curriculum = False`
   - `env_cfg.terrain.max_init_terrain_level = args.difficulty`
   - `env_cfg.debug_vis = False`
4. Standard env construction + `RslRlVecEnvWrapper` + `OnPolicyRunnerParkour` load
   (copied from play.py:247-296). Reject other runner classes with a clear error.
5. Post-construction terrain pinning helper (DESIGN.md §4 code block — apply *exactly*).
6. Pre-allocate buffers (DESIGN.md §2 sizes).
7. Step loop with per-env freeze on first `done=True` (DESIGN.md §7 pseudocode).
8. Fixed commands: `_commands[:, 0] = 1.0`, `[:, 1] = 0.0`, `[:, 2] = 0.0` every step (matches
   play.py:384-386 for parkour task).
9. On exit: save buffers + invoke plot helper (S3).

**Output paths (must match):**
- `{log_dir}/reward_attribution/reward_attribution_all.npz` — full buffer + metadata.
- `{log_dir}/reward_attribution/reward_attribution_{terrain_name}.npz` — per-env slice.
- `{log_dir}/reward_attribution/reward_attribution_{terrain_name}.png` — per-env plot.
- `{log_dir}/reward_attribution/run_meta.json` — checkpoint, command, term names, foot names, dt.

**Acceptance criteria:**
1. End-to-end run completes:
   `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reward_attribution.py
    --task Go2-Parkour-Direct-v0 --num_envs 5 --difficulty 5
    --load_run <latest> --checkpoint <latest model_*.pt>`
2. After exit, 5 PNGs, 5 per-terrain npzs, 1 all-buffer npz, 1 run_meta.json exist.
3. Each env spawned on a *different* terrain class — verifiable via `run_meta.json["terrain_ids"]`
   matching `[0,1,2,3,4]` in some order.
4. All envs at level=5 — verifiable via `run_meta.json["terrain_levels"] == [5,5,5,5,5]`.
5. `episode_len[i] >= 1` for every env (no env crashes at t=0).
6. Script does NOT print spurious training-time logs from `OnPolicyRunnerParkour` (set log_dir=None
   matching play.py:276-282 pattern).

**Dependencies:** S1 must be merged. S3 is a separate optional polish task; S2 should ship with a
*minimal inlined* plotter (10–20 LOC) so the user can iterate even without S3.

---

## Subtask S3 — Offline plotting helper (matplotlib)

**Owner worker type:** `oh-my-claudecode:executor`. Pure Python, no Isaac coupling.

**Files to create:**
- `/home/lgb/IsaacLab/scripts/reinforcement_learning/rsl_rl/play_utils/reward_attribution_plot.py`
  (sibling of `play_utils/` which already exists per file listing at top of investigation).

**Function signature:**
```python
def plot_terrain_attribution(
    npz_path: str,
    out_png_path: str,
    figsize: tuple[float, float] = (14, 8),
) -> None:
    """Read a per-terrain npz, render the 2-subplot figure described in DESIGN.md §8."""
```

**Plot requirements (DESIGN.md §8):**
- 2 subplots stacked, sharex=True. Top = reward time series (16 lines, color-coded by term family);
  bottom = foot contact (4 lanes, black=contact, white=air).
- A shaded band on the bottom subplot marking "3-leg phases" (exactly 1 foot airborne) in a
  distinct color (e.g. magenta @ 30% alpha) and "2-leg phases" in a fainter color (e.g.
  cyan @ 20%). Compute via `n_airborne = 4 - contact.sum(dim=-1)`.
- X-axis: bottom labeled in seconds (`t * step_dt`); top labeled in step index. Use twin axis or
  simply two-line label.
- Title: `f"Terrain: {terrain_name}  (level=5, ep_len={T} steps)"`.
- Legend on right, outside axes. Group reward terms with `bbox_to_anchor=(1.02, 1)`.
- Optional dotted black line `total = rewards.sum(dim=-1).clip(min=0)` (matches env's
  total_reward clip at parkour_env.py:1164) overlaid on top subplot.

**Acceptance criteria:**
1. Calling `plot_terrain_attribution(npz, out_png)` on a saved buffer produces a valid PNG.
2. The bottom subplot visually shows distinct 3-leg / 2-leg shaded bands when the underlying data
   contains such phases.
3. Plot regenerates from npz alone — no Isaac Sim runtime dependency. (Verify by running
   `python -c "from play_utils.reward_attribution_plot import plot_terrain_attribution; ..."` outside
   the isaac-parkour env, e.g. with a pure pip env that has numpy + matplotlib only.)
4. File is importable: `from play_utils.reward_attribution_plot import plot_terrain_attribution`.

**Dependencies:** S1 + S2 produce the npz format; S3 consumes it.

---

## Subtask S4 — (DESIGN ONLY) PyQt + IPC live viewer spec

**Owner worker type:** `oh-my-claudecode:writer` (Haiku — pure documentation).

**Files to create:**
- `/home/lgb/IsaacLab/_workspace/parkour_reward_attribution/PYQT_IPC_SPEC.md`

**Scope:** flesh out DESIGN.md §9 into an implementation-ready spec WITHOUT writing any code.
Include:
1. ZMQ socket setup (PUB side: where in `play_reward_attribution.py` to inject the socket; SUB side:
   `reward_attribution_viewer.py` startup).
2. msgpack schema definition (field names, dtypes, version field).
3. PyQt6 vs PySide6 dependency tradeoff for the isaac-parkour conda env.
4. pyqtgraph PlotWidget configuration (ring buffer size, axis lock, downsample policy).
5. Drop-frame policy (HWM, what the viewer does on backpressure).
6. Launch script: `./scripts/.../play_reward_attribution.py --live` flag and a separate
   `python reward_attribution_viewer.py` consumer.
7. Failure modes: viewer not running (sim must continue), viewer crashes mid-run, ZMQ socket
   busy.

**Acceptance criteria:**
1. PYQT_IPC_SPEC.md exists and references DESIGN.md §9 as parent.
2. No code is committed — the deliverable is the markdown only.
3. Spec answers the "what library + what schema + how to launch + how to fail" questions
   completely enough that a stage-3 implementation worker could pick it up cold.

**Dependencies:** none on S1/S2/S3 strictly, but for context the spec should assume S1's
`_last_reward_breakdown_per_env` exists as the producer source.

**Status:** SKIP if user signals MVP is sufficient. This subtask is the lowest priority of the four.

---

## Dependency graph

```
            S1 (env buffer, obs-worker)
             │
             ▼
            S2 (play script, executor) ──► S3 (plot helper, executor)

            S4 (PyQt+IPC spec, writer) — independent, lowest priority
```

## Recommended dispatch order

1. **S1 first** (sequential, blocking) — env hook is the prerequisite for S2.
2. **S2 + S3 in parallel** once S1 is verified (S2 can ship with a minimal inlined plot; S3 polishes
   it as a reusable module). If parallel risks merge conflict on the play_utils dir, run S2 first
   and S3 immediately after.
3. **S4 only on user demand** — pure design doc, no urgency, MVP works without it.

## Cross-cutting validation

After S1+S2 land, run `validate-code` worker on the changed files to confirm:
- No observation-shape regression (dict obs keys + critic dim still match runner cfg).
- No `_reset_idx` buffer that wasn't initialized (S1's buffer is overwritten each step so it's
  exempt).
- IL / AMP / discriminator dims untouched (S1 doesn't touch any of those).
