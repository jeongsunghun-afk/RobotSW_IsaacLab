# Reward Attribution Analysis Tool — DESIGN

**Status:** Design only. Do not implement from this doc; stage-2 workers (see `SUBTASKS.md`) will own
implementation under separate task IDs.

**Owner of this design:** `design-architect` (team `parkour-reward-attribution`).

**Date framing:** 2026-05-27. Replaces nothing; this is a *new* analysis tool, not a fix to the parkour
training env.

---

## 1. Problem framing

The user-observed failure mode is a **3-LEG GAIT**: when the Go2 policy is faced with parkour
terrain (step, gap, stair, hurdle), it adopts a strategy of carrying or "skipping" one leg over the
obstacle while the other three plant. This is a *learned behavior*, not a joint-limit symptom. The
prior `_workspace/calf_divergence/00_SYNTHESIS.md` investigation focused on joint divergence and is
**not load-bearing for this work** (per project memory note `project_parkour_3leg_gait_reframe.md`).

The diagnostic question the tool must answer:

> When the robot is on terrain X (step, gap, stair, hurdle, flat) at difficulty 5, **which of the
> 16 reward terms is most strongly correlated, in time, with phases where only 3 of 4 feet are in
> contact with the ground?**

The data product is therefore *paired*: a per-step reward-breakdown time series + a per-step
foot-contact bitmask, on the same `t` axis, per-terrain. The user does the visual attribution.

Out of scope for this tool:
- Suggesting reward changes ("rebalance feet_gait_pairing").
- Re-training, fine-tuning, or modifying the policy.
- Investigating actuator saturation / calf divergence.
- Touching the policy observation space (sim-to-real lock; project memory
  `project_parkour_analysis_constraints.md`).

---

## 2. Buffer schema

The tool produces one ND-buffer per run, materialized to `.npz` per terrain on episode termination.

| Field          | Shape                                    | Dtype       | Source                                                                                 |
|----------------|------------------------------------------|-------------|----------------------------------------------------------------------------------------|
| `rewards`      | `[T_max, N_envs=5, 16]`                  | `float32`   | New env-side buffer `_last_reward_breakdown_per_env` (see §5)                          |
| `foot_contact` | `[T_max, N_envs=5, 4]`                   | `bool`      | Derived in play script from `ContactSensor.data.net_forces_w_history` (see §6)         |
| `episode_len`  | `[N_envs=5]`                             | `int32`     | Step index at which env first returned `done=True`                                     |
| `terrain_id`   | `[N_envs=5]`                             | `int32`     | `_env_class` values (= `TERRAIN_CLASS_*` constants)                                    |
| `term_names`   | `list[str]` len 16                       | (metadata)  | `list(env.cfg.reward_scales.keys())`                                                   |
| `foot_names`   | `["FL", "FR", "RL", "RR"]`               | (metadata)  | URDF order, per `parkour_env.py:1076`                                                  |
| `reward_scales`| `dict[str,float]` len 16                 | (metadata)  | `env.cfg.reward_scales` (so raw vs scaled can be derived)                              |
| `commands`     | `[T_max, N_envs=5, 3]` (optional)        | `float32`   | `env._commands[:, :3]` per step — useful for visual context                            |

**Notes on the buffer:**
- `T_max` = `cfg.episode_length_s / step_dt` = `20.0 / 0.02` = **1000** steps. Pre-allocate
  `[1000, 5, 16]` float32 (~313 KB) and `[1000, 5, 4]` bool (~20 KB). Trivial memory.
- After the first `done=True` for env `i`, the recorder *freezes* env `i`'s slice and stops writing.
  When all envs are frozen, the script saves and exits.
- The MVP stores values **after scaling** (i.e. `cfg.reward_scales[k] * step_dt * raw_value`), since
  this is what the env already computes per `parkour_env.py:1158`. Storing `reward_scales` alongside
  lets a notebook recover raw values if needed.

---

## 3. Terrain → env mapping

The terrain generator declares **11 sub-terrain entries** at
`/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py:54-249`,
of which **5 are ACTIVE** (proportion > 0) and 6 are REGISTERED-INACTIVE (proportion = 0).

| Class ID | Key                       | Active? | Proportion | Source (cfg)                                     |
|---------:|---------------------------|---------|-----------:|--------------------------------------------------|
| 0        | `parkour_flat`            | ✅      | 0.2        | parkour_env_cfg.py:61-76                         |
| 1        | `parkour_hurdle`          | ✅      | 0.2        | parkour_env_cfg.py:77-91                         |
| 2        | `parkour_step`            | ✅      | 0.2        | parkour_env_cfg.py:92-108                        |
| 3        | `parkour_gap`             | ✅      | 0.2        | parkour_env_cfg.py:109-122                       |
| 4        | `parkour_stair`           | ✅      | 0.2        | parkour_env_cfg.py:123-136                       |
| 5        | `parkour_stepping_stones` | ❌      | 0.0        | parkour_env_cfg.py:147-163                       |
| 6        | `parkour_balance_beam`    | ❌      | 0.0        | parkour_env_cfg.py:164-180                       |
| 7        | `parkour_crawl`           | ❌      | 0.0        | parkour_env_cfg.py:181-201                       |
| 8        | `parkour_slope`           | ❌      | 0.0        | parkour_env_cfg.py:202-216                       |
| 9        | `parkour_zigzag_hurdles`  | ❌      | 0.0        | parkour_env_cfg.py:217-233                       |
| 10       | `parkour_rough_blocks`    | ❌      | 0.0        | parkour_env_cfg.py:234-248                       |

**MVP:** `num_envs = 5` (one env per *active* terrain). Inactive terrains have zero columns in the
generator grid and cannot be sampled (`parkour_env.py:155-161` LUT — they would yield
empty masks). The script must NOT attempt to spawn an env on an inactive class.

If, later, the user activates one of the inactive terrains by raising its proportion (and
rebalancing the active set so proportions sum to 1.0, per the assert at
`parkour_env.py:151-154`), the script auto-detects new active terrains via
`[t for t in TERRAIN_CLASS_* if cfg.terrain.terrain_generator.sub_terrains[name].proportion > 0]`,
and `num_envs` updates accordingly.

**Column assignment** (for difficulty pinning): with `num_cols=40` (cfg:48) and 5 active terrains
of proportion 0.2 each, the proportion-based LUT (`parkour_env.py:155-161`) assigns columns:

| Class           | Column range |
|-----------------|--------------|
| `flat` (0)      | 0  – 7       |
| `hurdle` (1)    | 8  – 15      |
| `step` (2)      | 16 – 23      |
| `gap` (3)       | 24 – 31      |
| `stair` (4)     | 32 – 39      |

The script picks the *first* column of each class deterministically:
`col_of[k] = int((env._col_to_class == k).nonzero()[0].item())`. Reading from `_col_to_class`
directly (computed at env init, parkour_env.py:160) ensures the script tracks the live LUT and
does not hardcode column ranges.

---

## 4. Difficulty pinning approach

**Goal:** every env at level=5 (mid difficulty; `num_rows=11`, max level=10), one per terrain class.

**Approach** (no env-code changes required; uses existing public surfaces):

1. In the play script, before constructing the env, set `env_cfg.terrain_curriculum = False`
   (defined `parkour_env_cfg.py:615`). This causes `_reset_idx` to **skip** the curriculum block
   (`parkour_env.py:1238-1241`), so `_terrain_levels` and `_terrain_types` are not mutated after
   reset.
2. (Optional) set `env_cfg.terrain.max_init_terrain_level = 5` so the very first random init
   places envs around the target row. This is cosmetic — step 3 overrides anyway.
3. After `env = gym.make(...)`, override per-env terrain pinning. This mirrors the existing
   `_change_terrain_for_viewer` keyboard hook (`parkour_env.py:1681-1738`) but applied at script
   level:

   ```python
   raw = env.unwrapped
   active_classes = [TERRAIN_CLASS_FLAT, TERRAIN_CLASS_HURDLE, TERRAIN_CLASS_STEP,
                     TERRAIN_CLASS_GAP, TERRAIN_CLASS_STAIR]
   col_per_env = torch.tensor(
       [int((raw._col_to_class == k).nonzero()[0].item()) for k in active_classes],
       dtype=torch.long, device=raw.device,
   )
   raw._terrain_levels[:] = 5
   raw._terrain_types[:]  = col_per_env
   raw._env_class[:]      = raw._col_to_class[raw._terrain_types]
   raw._terrain.env_origins[:] = raw._terrain.terrain_origins[
       raw._terrain_levels, raw._terrain_types
   ]
   raw._skip_curriculum[:] = True  # belt-and-suspenders: even if curriculum=True, skip
   raw._reset_idx(None)            # full reset spawns each env on its assigned tile
   raw.episode_length_buf[:] = 0   # cancel _reset_idx's randomized spread (parkour_env.py:1205-1207)
   ```

   `_terrain_levels` and `_terrain_types` are aliases of the `TerrainImporter`'s tensors
   (`parkour_env.py:142-143`), so writing into them updates the canonical grid coords. The
   subsequent `_terrain.env_origins[:] = ...` propagates the new spawn position. This pattern is
   *identical* to what the F/L keyboard handler already does and is therefore on a validated
   code path.

4. With `terrain_curriculum=False`, the curriculum advance/regress logic at
   `parkour_env.py:1304-1351` is skipped on every reset, so level=5 sticks for the whole run.

**Why this is safe (sim-to-real / training safety):** this is the existing `_change_terrain_for_viewer`
behavior generalized from "one env" to "all 5 envs", executed once at script start. No new
code paths are exercised; the env is unmodified.

---

## 5. Per-term reward read path

**Current state of the env:**
- The per-step *scaled* reward breakdown is materialized inside `_get_rewards()` at
  `parkour_env.py:1136-1162`. The dict `reward_values` (lines 1136-1153) holds the 16 raw terms.
- The for-loop at lines 1157-1161 scales each term by `cfg.reward_scales[k] * step_dt`, accumulates
  into `_episode_sums`, and (only for env 0) stores into `_last_reward_breakdown_env0`
  (parkour_env.py:1161). A pre-existing UDP debug publisher
  (`/home/lgb/IsaacLab/scripts/reinforcement_learning/rsl_rl/_debug/reward_publisher.py:74-88`)
  consumes this dict.

**Gap:** `_last_reward_breakdown_env0` is **env-0 only**. For per-env-per-step recording we need a
`[N, 16]` tensor. Re-implementing the 16 reward functions externally is brittle (term semantics
change across edits) — so the cleanest path is a single-line env addition.

**Proposed minimal env-side hook** (one buffer + 2 lines in `_get_rewards()`):

In `__init__()` (next to `_last_reward_breakdown_env0` at line 187-189), add:
```python
# Per-env per-term scaled reward contribution — read by external analysis tools.
# Updated every _get_rewards() call; never affects reward computation.
self._last_reward_breakdown_per_env = torch.zeros(
    self.num_envs, len(self.cfg.reward_scales), device=self.device
)
self._reward_term_names: list[str] = list(self.cfg.reward_scales.keys())  # canonical ordering
```

In the for-loop at parkour_env.py:1157-1161, after `scaled = cfg.reward_scales[key] * step_dt * value`:
```python
self._last_reward_breakdown_per_env[:, i] = scaled    # i = enumerate index
```
(Loop becomes `for i, (key, value) in enumerate(reward_values.items()):`.)

**Why this is safe:**
- The buffer is *write-only* from the env's perspective (no gradient touches it).
- Order is the dict-insertion order of `cfg.reward_scales`, which is stable in Python 3.7+ and
  is already the same ordering used by `_episode_sums` and `_last_reward_breakdown_env0`. So this
  buffer aligns 1:1 with the existing logging contract.
- `_reset_idx` does NOT need to zero this buffer — it's overwritten every `_get_rewards()` call.
  (Following the same no-reset pattern as `_last_reward_breakdown_env0`.)
- Cost: `5 envs × 16 floats = 80 floats per step`. Negligible.

**Alternative considered & rejected:** monkey-patching `_get_rewards()` from the play script to
intercept the local dict. Rejected because (a) Hydra-wrapped construction makes pre-init patching
fragile, (b) future env edits could silently break the patch, (c) the env owner is the right place
for this attribute.

**Term names ordering** (the 16 keys, locked at `parkour_env_cfg.py:566-583`):
```
['tracking_goal_vel', 'tracking_yaw', 'lin_vel_z_l2', 'ang_vel_xy_l2',
 'orientation_l2', 'dof_acc_l2', 'collision', 'action_rate_l2',
 'delta_torques', 'torques_l2', 'hip_pos', 'dof_error_l2',
 'feet_stumble', 'feet_edge', 'feet_dragging', 'feet_gait_pairing']
```

---

## 6. Per-foot contact read path

**Zero env modification required.** The contact sensor is already configured at
`parkour_env_cfg.py:537-542`, and `_feet_ids` is computed at `parkour_env.py:193` from
`.find_bodies(".*foot")` in URDF order `[FL, FR, RL, RR]` (verified by the hind-foot assertion at
parkour_env.py:194-198).

**Read pattern in the play script** (executed *after* `env.step()`):

```python
raw = env.unwrapped
forces = raw._contact_sensor.data.net_forces_w_history  # (N, hist, B, 3)
contact = torch.norm(forces[:, 0, raw._feet_ids], dim=-1) > 2.0  # (N, 4) bool
```

This is the **same threshold + indexing** the env's own `feet_*` rewards use at
parkour_env.py:1069, so the contact bitmask we record is the *raw current-step* contact (debounced
`contact_filt` is also available via `torch.logical_or(contact, raw._last_contacts)`, but raw is
preferred for unambiguous per-step semantics in the plot).

**Foot index order** (per parkour_env.py:1076 comment): `[FL(0), FR(1), RL(2), RR(3)]`.

**Pitfall guard:** `_feet_ids` is a `list[int]` (returned by `ContactSensor.find_bodies`). When
fancy-indexing a tensor with it, cast to a `torch.LongTensor` or use a tuple — current pattern at
parkour_env.py:1069 (`net_contact_forces[:, 0, self._feet_ids]`) works because PyTorch accepts
list-of-ints as advanced indexing. The script should copy the same pattern verbatim to avoid
divergence.

**Optional extra signals** (cheap, no env change):
- Per-foot z-velocity: `raw._robot.data.body_lin_vel_w[:, raw._feet_ids, 2]` — would let the
  user disambiguate "foot held in air" vs "foot dragging on ground while moving".
- Per-foot air-time: `raw._contact_sensor.data.current_air_time[:, raw._feet_ids]`
  (parkour_env.py:1085) — a smoother contact signal than the boolean.

Both are listed as **optional add-ons**; not in MVP.

---

## 7. Modified play script

**Decision: create a NEW sibling script.** Path:
`/home/lgb/IsaacLab/scripts/reinforcement_learning/rsl_rl/play_reward_attribution.py`

**Rationale for sibling (not fork-in-place):**
- `play.py` is a multi-task script (Go2WTW, Go2Neck, R_Skeleton, Parkour, MotionJig). Adding a
  reward-attribution path inside its already-branchy main loop would be invasive.
- The reward-attribution loop has fundamentally different control flow (per-env freeze + stop on
  all-done), unlike the open-ended replay loop of `play.py`.
- Keeps git history of the existing play workflow clean; reduces blast radius of any bug.

**Inheritance:** the new script reuses `play.py`'s established patterns 1:1:
- AppLauncher / CLI arg parsing (play.py:11-58)
- Hydra task-config decorator and runner-class branching (play.py:202-282) — restrict to the
  parkour runner (`OnPolicyRunnerParkour`); error out for other tasks since terrain pinning is
  parkour-specific.
- Checkpoint resolution via `get_checkpoint_path` (play.py:222-232).
- `RslRlVecEnvWrapper` and `policy = runner.get_inference_policy(...)` (play.py:270-286).

**Run command** (per `isaaclab.sh` launcher convention):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reward_attribution.py \
    --task Go2-Parkour-Direct-v0 --num_envs 5 --difficulty 5 \
    --load_run <run> --checkpoint <model_*.pt>
```

(`--difficulty` is a new CLI flag, default 5; allows user to sweep later.)

**Loop skeleton (pseudocode):**

```python
@hydra_task_config("Go2-Parkour-Direct-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 1. Force fixed-difficulty + curriculum off
    env_cfg.scene.num_envs = 5
    env_cfg.terrain_curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    # 2. Disable video, edge-mask debug (optional but cleaner)
    env_cfg.debug_vis = False
    # 3. Standard env creation + policy load (copied from play.py:247-296)
    env = gym.make(...); env = RslRlVecEnvWrapper(env, clip_actions=...)
    runner = OnPolicyRunnerParkour(env, ...); runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    # 4. Pin terrain (see §4)
    pin_terrain_per_env(env.unwrapped)  # helper
    obs = env.get_observations()
    # 5. Pre-allocate buffers
    T_max = int(env.unwrapped.cfg.episode_length_s / env.unwrapped.step_dt)  # 1000
    N = env.unwrapped.num_envs                                                # 5
    K = len(env.unwrapped.cfg.reward_scales)                                  # 16
    rewards   = torch.zeros(T_max, N, K, dtype=torch.float32, device="cpu")
    contacts  = torch.zeros(T_max, N, 4, dtype=torch.bool,    device="cpu")
    commands  = torch.zeros(T_max, N, 3, dtype=torch.float32, device="cpu")
    ep_len    = torch.full((N,), -1, dtype=torch.int32)
    env_done  = torch.zeros(N, dtype=torch.bool)
    # 6. Main loop
    t = 0
    while simulation_app.is_running() and t < T_max and not env_done.all():
        with torch.inference_mode():
            env.unwrapped._commands[:, 0] = 1.0  # forward vel = 1 m/s (deterministic)
            env.unwrapped._commands[:, 1] = 0.0
            env.unwrapped._commands[:, 2] = 0.0
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # 6a. Capture per-env per-term rewards
            r_step = env.unwrapped._last_reward_breakdown_per_env.cpu()  # (N, K)
            # 6b. Capture per-foot contact (raw)
            raw = env.unwrapped
            forces = raw._contact_sensor.data.net_forces_w_history
            c_step = (torch.norm(forces[:, 0, raw._feet_ids], dim=-1) > 2.0).cpu()  # (N, 4)
            # 6c. Capture commands
            cmd_step = raw._commands[:, :3].cpu()
            # 6d. Write into buffers only for envs that haven't terminated yet
            keep = ~env_done
            rewards[t, keep]  = r_step[keep]
            contacts[t, keep] = c_step[keep]
            commands[t, keep] = cmd_step[keep]
            # 6e. Mark newly-terminated envs
            done_now = dones.cpu().bool() & ~env_done
            for i in done_now.nonzero(as_tuple=True)[0].tolist():
                ep_len[i] = t + 1
                env_done[i] = True
            policy_nn.reset(dones)
            t += 1
    # 7. For envs that hit T_max without terminating, mark ep_len = T_max
    ep_len[~env_done] = t
    # 8. Save buffer + run plot
    save_npz_and_plot(log_dir, rewards, contacts, commands, ep_len, env, T_used=t)
```

**Episode-stop gating semantics:**
- The env's `_reset_idx` auto-respawns on every `done=True` (DirectRLEnv standard).
- We do NOT prevent the env from respawning — that would require monkey-patching `_reset_idx`,
  which is invasive. Instead, we *freeze the recording* per-env once `env_done[i] == True`,
  which is equivalent for analysis purposes (we only ever look at `rewards[:ep_len[i], i, :]`).
- The auto-respawn for finished envs keeps the simulator running cleanly; we just ignore their
  data after termination.
- Loop exits when `env_done.all()` OR `t >= T_max` (defensive).

---

## 8. Offline plot design (matplotlib)

**Path convention:**
- Output dir: `{log_dir}/reward_attribution/` (under the checkpoint's run dir, so artifacts are
  versioned with the model).
- Per-terrain file: `reward_attribution_{terrain_name}.png` and `reward_attribution_{terrain_name}.npz`.
- One combined buffer dump: `reward_attribution_all.npz` (contains everything from §2).
- Run metadata: `run_meta.json` (checkpoint path, difficulty, command, term names, foot names).

**Per-terrain figure layout** (matplotlib `subplots(2, 1, sharex=True)`, figsize=(14, 8)):

```
┌──────────────────────────────────────────────────────────┐
│  Terrain: parkour_step  (env_idx=2, level=5, T=872 steps)│
├──────────────────────────────────────────────────────────┤
│                                                          │
│ Reward time series                                       │  ← top subplot
│ Y: scaled reward contribution per step                   │
│ 16 lines, one per term, colored by term family:          │
│   - tracking_*: blues                                    │
│   - lin/ang/orient/dof_acc: reds (penalties)             │
│   - collision/torques/action_rate: oranges               │
│   - feet_*: greens (THESE are the gait-related ones)     │
│ Legend on right; sum line overlaid black dotted (total)  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ Foot contact (4 channels)                                │  ← bottom subplot
│ Y: 4 stacked binary lanes — FL, FR, RL, RR (top to btm)  │
│ Color: black = in contact, white = airborne              │
│ Shading: a colored bar across the bottom marking         │
│  "3-leg phases" (where exactly 1 foot is airborne)       │
│  and "2-leg phases" (where 2 feet are airborne)          │
│                                                          │
│ X axis: time step (also seconds in twin axis: t·step_dt) │
└──────────────────────────────────────────────────────────┘
```

The "3-leg phase" shading on the bottom is *the* visual cue the user wants. With both subplots
sharing the time axis, the user can read off "during this 3-leg shaded band, term X spiked / dipped".

**Plot generation function signature (pseudocode):**

```python
def make_plot(npz_path, out_png_path):
    data = np.load(npz_path)
    rewards   = data["rewards"]       # (T, K) — sliced per env already
    contacts  = data["foot_contact"]  # (T, 4)
    term_names = data["term_names"]
    foot_names = data["foot_names"]
    T = data["episode_len"]
    # ... matplotlib code ...
```

**Why offline + matplotlib (not live):** the user has tried live matplotlib and reported it
"didn't work" (likely GIL contention with Isaac's render loop, or QtAgg backend conflict). Offline
matplotlib **after** the episode is robust because the sim app has exited by then. PyQt+IPC is the
*live* version (§9) and is design-only.

---

## 9. PyQt + IPC live viewer — DESIGN ONLY

**Status:** No implementation in this MVP. Documented here as the recommended path if live
visualization becomes a requirement in stage-3 or beyond.

**Why live matplotlib previously failed (working hypothesis):**
- Matplotlib `plt.pause()` / `Figure.canvas.draw_idle()` in the same process as Isaac Sim shares
  the Python GIL with the sim's Python-side update callbacks and Omniverse Kit's Qt event loop.
- Kit already owns a Qt main loop (PySide). Embedding a second matplotlib `TkAgg` / `QtAgg`
  backend in the same process collides — typical symptom is window freeze or render glitch.
- Even with a non-interactive backend, `plt.draw()` blocking for >5 ms per step at 50 Hz costs >25%
  of the per-step budget.

**Process isolation fixes all of the above.** The sim and the viewer become separate processes;
the only coupling is a small IPC channel.

**Recommended stack:**
- **Transport:** ZeroMQ `PUB/SUB` over `ipc:///tmp/parkour_reward.sock` (Unix domain socket).
  - Non-blocking PUB on sim side; slow consumers drop frames automatically (high-water-mark).
  - No broker needed; one publisher, one subscriber.
  - msgpack-encoded payloads (~200 bytes/step uncompressed).
- **Viewer framework:** PyQt6 (or PySide6) main window with **pyqtgraph** for plot widgets.
  - pyqtgraph is the de-facto choice for real-time scientific plotting in Python; ring-buffer
    updates at >1 kHz are routine.
  - PyQt's `QSocketNotifier` integrates ZMQ directly into the Qt event loop, so there is no
    polling thread.
- **Plot layout:** mirror §8 (top = 16-line reward strip, bottom = 4-lane contact strip), but
  scrolling left-to-right with a rolling time window (e.g. last 500 steps = 10 s).
- **Per-env selector:** dropdown to pick which of the 5 envs to view; or 5 side-by-side plots
  if the user has screen real estate.

**Data flow diagram (ASCII):**

```
   ┌─────────────────────────────────────────────────────────────────┐
   │ Process A — sim                                                 │
   │  ./isaaclab.sh -p play_reward_attribution.py                    │
   │                                                                 │
   │  ┌────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
   │  │ Go2Parkour │ →  │ _get_rewards()   │ →  │ ZMQ PUB socket   │ │
   │  │ DirectRL   │    │ + contact read   │    │ (non-blocking)   │ │
   │  │ env        │    │ + msgpack encode │    │ ipc:///tmp/...   │ │
   │  └────────────┘    └──────────────────┘    └────────┬─────────┘ │
   │                                                     │            │
   └─────────────────────────────────────────────────────┼────────────┘
                                                         │ IPC
   ┌─────────────────────────────────────────────────────┼────────────┐
   │ Process B — viewer                                  ↓            │
   │  python reward_attribution_viewer.py                              │
   │                                                                   │
   │  ┌──────────────────┐   ┌──────────────────┐   ┌───────────────┐ │
   │  │ ZMQ SUB socket   │ → │ QSocketNotifier  │ → │ Ring buffers  │ │
   │  │ (HWM=10, drops)  │   │ → Qt event loop  │   │ (T, K), (T,4) │ │
   │  └──────────────────┘   └──────────────────┘   └──────┬────────┘ │
   │                                                       │           │
   │  ┌──────────────────────────────────────────┐  ┌──────↓────────┐ │
   │  │ pyqtgraph PlotWidgets                    │ ←│ update tick   │ │
   │  │  - reward strip (16 PlotCurveItem)       │  │ (Qt 30 Hz)    │ │
   │  │  - contact strip (4 BarGraphItem)        │  └───────────────┘ │
   │  │  - env selector (QComboBox)              │                    │
   │  └──────────────────────────────────────────┘                    │
   └───────────────────────────────────────────────────────────────────┘
```

**Update rate budget:**
- Sim policy step rate: 50 Hz (decimation=4, physics=200 Hz; per cfg lines 388 + 414).
- Per step payload: `5 envs × (16 rewards × 4B + 4 contacts × 1B + 3 commands × 4B) ≈ 480 B` raw,
  plus msgpack overhead → ~700 B. At 50 Hz that's 35 KB/s. ZMQ over IPC handles this 100×+.
- Viewer redraws at 30 Hz (Qt tick). Ring buffer of 1000 steps × 16 lines × float32 = 64 KB —
  trivial. Pyqtgraph throughput is well in excess of this.

**Pros vs the §8 offline matplotlib MVP:**
| Aspect                    | Offline matplotlib (MVP)            | PyQt + IPC (design)             |
|---------------------------|-------------------------------------|---------------------------------|
| Latency                   | Episode-end, batch                  | <33 ms (Qt tick rate)           |
| Process isolation         | N/A (no shared state with sim)      | Full (separate Python)          |
| Implementation effort     | ~150 LOC + a plot fn                | ~600–800 LOC across 2 scripts   |
| Risk of breaking sim loop | None (post-run)                     | Low (PUB is non-blocking)       |
| Debuggability of viewer   | Re-run on saved npz                 | Independent restart, no re-sim  |
| Step-by-step pause/replay | Not supported                       | Trivial to add                  |
| Multi-env comparison      | Per-terrain PNG only                | Selectable / side-by-side       |

**Cons of PyQt + IPC:**
- Two processes to launch + tear down. Need a smoke test that the SUB socket actually receives.
- pyqtgraph dependency adds a wheel that's not currently in the isaac-parkour conda env.
- Color scheme & widget polish takes nontrivial UI design effort.

**Recommendation:** ship MVP (offline matplotlib, this design) first. Promote to PyQt+IPC if and
only if the user finds the offline plot insufficient for cause-finding — at which point the npz
buffer format from §2 is reusable verbatim by the viewer (it just becomes a live source rather
than a saved file).

---

## 10. Live PyQt Viewer Usage

**Status:** Implemented (L1 + L2 + L3 integration, 2026-05-27).

### Install

`pyzmq` and `msgpack` are already in the `isaac-parkour` conda env.
Install the two viewer-side packages:

```bash
conda activate isaac-parkour
pip install PyQt6 pyqtgraph
```

### Launch order: viewer first, then sim

```bash
# Terminal 1 — start viewer (binds nothing; it connects to the publisher)
conda activate isaac-parkour
python scripts/reinforcement_learning/rsl_rl/reward_viewer/reward_attribution_viewer.py
# Optional: override endpoint, ring-buffer size, or render rate
python scripts/reinforcement_learning/rsl_rl/reward_viewer/reward_attribution_viewer.py \
    --endpoint ipc:///tmp/parkour_reward.sock \
    --capacity 500 \
    --fps 30 \
    --num-envs 5

# Terminal 2 — start sim with live-viz enabled
conda activate isaac-parkour
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/reward_viewer/play_reward_attribution.py \
    --task Go2-Parkour-Direct-v0 --num_envs 5 --difficulty 5 \
    --load_run <run_name> --checkpoint model_<N>.pt \
    --live-viz
# TCP override for smoke-testing (avoids IPC socket cleanup issues):
# add --zmq-endpoint tcp://127.0.0.1:5557
```

The viewer connects to the endpoint and waits; frames start arriving once the sim
enters its step loop (typically 30–60 s after launch, after terrain/asset load).

### Default endpoint

`ipc:///tmp/parkour_reward.sock`  (defined in `reward_pub_protocol.ZMQ_ENDPOINT`)

Override on both sides:
- Viewer: `--endpoint <ENDPOINT>`
- Sim:    `--zmq-endpoint <ENDPOINT>`

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Address already in use` on IPC | `rm /tmp/parkour_reward.sock` then restart sim |
| No frames received (viewer shows "Waiting…") | Ensure viewer is started **before** the sim; the PUB socket is only bound once sim enters the step loop |
| `ValueError: Unsupported protocol version` | Viewer and sim are using different versions of `reward_pub_protocol.py` — ensure both pull from the same file at `scripts/reinforcement_learning/rsl_rl/reward_viewer/reward_pub_protocol.py` |
| Viewer crashes on init (`QApplication` error) | Ensure `DISPLAY` is set; for remote sessions use `export DISPLAY=:0` or an X11 tunnel |
| HWM drops (viewer shows `drops > 0`) | Normal under heavy sim load; viewer auto-recovers. Reduce `--fps` if drops are excessive |

### Smoke-test (no sim required)

```bash
conda activate isaac-parkour
python scripts/reinforcement_learning/rsl_rl/reward_viewer/reward_attribution_viewer.py --selftest
# Expected: SELFTEST PASSED  received=60/60  buf_size=60  wraparound=OK  envs=5  proto=L1
```

---

## 11. Acceptance gate for the design

The stage-2 implementation can be considered complete when:

1. `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_reward_attribution.py
    --task Go2-Parkour-Direct-v0 --num_envs 5 --difficulty 5 --load_run <latest> --checkpoint
    <latest model_*.pt>` runs to completion without crashing.
2. Output dir `{log_dir}/reward_attribution/` contains 5 PNGs + 5 npzs + 1 all-buffer npz + 1
   run_meta.json.
3. Each PNG shows both subplots populated; foot-contact subplot reveals visible 3-leg/2-leg phases
   on terrain types where the policy struggles (step, stair, gap).
4. The env-side change (§5) is a *single buffer + a write inside the existing for-loop*, with no
   change to the reward computation itself. Existing training runs unaffected (same `_episode_sums`,
   same total reward, same gradient).
5. No change touches `source/isaaclab/`, the policy observation space, or the contact sensor cfg.

