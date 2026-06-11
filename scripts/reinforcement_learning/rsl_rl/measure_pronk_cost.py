# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pronk-cost diagnostic for Go2-Parkour-Symmetry.

Quantifies whether the "4-foot simultaneous jump" (pronk) behaviour is actually
inefficient / dangerous on stair/step terrain, vs the legitimate gap-jump and the
flat-walk references.  Collects per-step signals under the NATIVE policy (no command
forcing — the env drives traversal via its goal/command system, matching training).

Signals collected per step (T, N, ...):
  - applied_torque            : robot.data.applied_torque                    (N, 12)
  - feet contact force        : contact_sensor.net_forces_w_history          (N, 4, 3) via history-max
  - all-airborne (pronk)      : all 4 feet below force threshold             (N,)
  - env terrain class         : env._env_class                              (N,)
  - air_time                  : contact_sensor.current_air_time[feet]        (N, 4)
  - joint_vel                 : robot.data.joint_vel                         (N, 12)  (CoT + DCMotor envelope)
  - terrain_level             : env._terrain_levels                          (N,)
  - root_pos_w (xy)           : robot.data.root_pos_w[:, :2]                 (N, 2)   (distance for CoT)
  - dones + 4 termination causes (tilt / low_height / base_contact / goal_reached)

Failed-episode exclusion is done POST-HOC by segmenting each env timeline at done
boundaries and dropping every sample belonging to a failure-terminated episode.

Actuator limits (effort_limit / saturation_effort / velocity_limit) are READ FROM THE
RUNTIME actuator object — never hard-coded.  Saturation is flagged against the DYNAMIC
DCMotor torque-speed envelope, not a static scalar.

Run (headless):
  cd /home/lgb/IsaacLab && ./isaaclab.sh -p \
      scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py \
      --task Go2-Parkour-Symmetry --num_envs 256 --num_steps 3000 --headless \
      --checkpoint logs/rsl_rl/go2_parkour_symmetry/2026-06-09_17-53-09/model_16700.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pronk-cost diagnostic for Go2 Parkour Symmetry.")
parser.add_argument("--task", type=str, default="Go2-Parkour-Symmetry")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=3000)
# --checkpoint provided by cli_args.add_rsl_rl_args()
parser.add_argument(
    "--force_thresh", type=float, default=2.0, help="Contact force threshold [N] (default 2.0, matches env reward)"
)
parser.add_argument("--sat_frac", type=float, default=0.95, help="|tau| >= sat_frac * dynamic_bound => saturated")
parser.add_argument("--seed", type=int, default=42)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunnerParkour

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

SCRIPT_VERSION = "pronk_v1"
OUT_DIR = "/home/lgb/IsaacLab/_workspace"


# ── main ──────────────────────────────────────────────────────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 64}")
    print(f"=== measure_pronk_cost {SCRIPT_VERSION} ===")
    print(f"=== started {ts} ===")
    print(f"{'=' * 64}\n")

    # ── Environment setup ──────────────────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.debug_vis = False
    for attr in ("debug_vis_edge_mask", "enable_keyboard_view_switch"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    # Checkpoint
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[{SCRIPT_VERSION}] Task       : {args_cli.task}")
    print(f"[{SCRIPT_VERSION}] Checkpoint : {resume_path}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    assert agent_cfg.class_name == "OnPolicyRunnerParkour", agent_cfg.class_name
    runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    base_env = env.unwrapped
    device = base_env.device
    N = args_cli.num_envs
    T = args_cli.num_steps

    # Foot ids and names (runtime, no hard-coding)  — order [FL, FR, RL, RR]
    feet_ids = list(base_env._feet_ids)
    foot_names = [base_env._contact_sensor.body_names[i] for i in feet_ids]
    print(f"[{SCRIPT_VERSION}] Foot body names : {foot_names}")
    print(f"[{SCRIPT_VERSION}] Foot body ids   : {feet_ids}")

    from isaaclab_tasks.direct.parkour.parkour_env import _TERRAIN_CLASS_NAMES

    # ── Read ACTUAL actuator limits from the runtime actuator object ───────────
    act = base_env._robot.actuators["base_legs"]
    # effort_limit / velocity_limit are resolved to per-joint tensors (num_envs, num_act_joints)
    eff_lim = act.effort_limit.to(device).float()  # (N, J) or scalar -> broadcast
    if eff_lim.dim() == 0:
        eff_lim = eff_lim.expand(N, 12).contiguous()
    vel_lim = act.velocity_limit.to(device).float()
    if vel_lim.dim() == 0:
        vel_lim = vel_lim.expand(N, 12).contiguous()
    sat_eff = float(act._saturation_effort) if hasattr(act, "_saturation_effort") else float(act.cfg.saturation_effort)
    # The DCMotor acts only on the 12 leg joints. applied_torque[:, act.joint_indices] is aligned
    # with act.effort_limit / act.velocity_limit (articulation.py:1901). joint_indices may be a
    # slice OR a tensor; we index applied_torque/joint_vel with it directly to stay aligned.
    act_joint_ids = act.joint_indices  # slice | tensor — indices into robot joints (articulation order)
    print(f"[{SCRIPT_VERSION}] actuator effort_limit (per-joint, env0): "
          f"{eff_lim[0].detach().cpu().numpy().round(3).tolist()}")
    print(f"[{SCRIPT_VERSION}] actuator velocity_limit (per-joint, env0): "
          f"{vel_lim[0].detach().cpu().numpy().round(3).tolist()}")
    print(f"[{SCRIPT_VERSION}] actuator saturation_effort (scalar): {sat_eff}")
    print(f"[{SCRIPT_VERSION}] actuator controls joint ids: {act_joint_ids.tolist() if torch.is_tensor(act_joint_ids) else act_joint_ids}")

    # ── Nominal robot mass (for bodyweight normalization) ───────────────────────
    default_mass = base_env._robot.data.default_mass  # (N, num_bodies)
    total_mass = float(default_mass[0].sum().item())
    bodyweight_N = total_mass * 9.81
    print(f"[{SCRIPT_VERSION}] nominal total mass = {total_mass:.3f} kg  -> bodyweight = {bodyweight_N:.2f} N")
    print(f"[{SCRIPT_VERSION}]   (note: env randomizes base mass +[-1,3] kg; bodyweight uses nominal default_mass)\n")

    # ── Allocate collection buffers (CPU numpy, streamed) ──────────────────────
    arr_class = np.zeros((T, N), dtype=np.int64)
    arr_level = np.zeros((T, N), dtype=np.int16)
    arr_contact = np.zeros((T, N, 4), dtype=bool)            # per-foot contact (history-max norm > thresh)
    arr_airborne = np.zeros((T, N), dtype=bool)              # all 4 feet airborne (pronk indicator)
    arr_peakfz = np.zeros((T, N, 4), dtype=np.float32)       # per-foot peak vertical force over history dim
    arr_sat = np.zeros((T, N), dtype=bool)                   # ANY leg joint saturated this step
    arr_satcount = np.zeros((T, N), dtype=np.int8)           # how many of 12 joints saturated
    arr_mech_pow = np.zeros((T, N), dtype=np.float32)        # sum of positive tau*qdot over 12 joints [W]
    arr_rootxy = np.zeros((T, N, 2), dtype=np.float32)       # base xy (distance for CoT)
    arr_dones = np.zeros((T, N), dtype=bool)
    arr_c_tilt = np.zeros((T, N), dtype=bool)
    arr_c_low = np.zeros((T, N), dtype=bool)
    arr_c_base = np.zeros((T, N), dtype=bool)
    arr_c_goal = np.zeros((T, N), dtype=bool)

    obs = env.get_observations()

    # jids indexes applied_torque/joint_vel to match the actuator's joint order, so the columns
    # align 1:1 with eff_lim / vel_lim (both (num_envs, num_act_joints)). joint_indices is a slice
    # or tensor; either indexes a (N, num_joints) tensor correctly along dim=1.
    jids = act_joint_ids
    if torch.is_tensor(jids):
        jids = jids.to(device).long()

    print(f"[{SCRIPT_VERSION}] Collecting {T} steps x {N} envs (NATIVE policy, no command forcing) ...")
    t_start = time.time()
    for step in range(T):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        # ── contact forces: history-max over the history dim (catches sub-step transients) ──
        net_cf = base_env._contact_sensor.data.net_forces_w_history  # (N, hist, bodies, 3)
        feet_cf_hist = net_cf[:, :, feet_ids, :]                      # (N, hist, 4, 3)
        # per-foot peak vertical (z) force over history
        peak_fz = feet_cf_hist[..., 2].abs().max(dim=1).values        # (N, 4)
        # contact = current (index 0) norm > thresh (matches env reward semantics)
        contact = torch.norm(net_cf[:, 0, feet_ids], dim=-1) > args_cli.force_thresh  # (N, 4)
        airborne = torch.all(~contact, dim=1)                         # (N,)

        # ── torque / saturation (dynamic DCMotor envelope) ──
        tau = base_env._robot.data.applied_torque[:, jids]            # (N, 12) actuator-joint order
        qd = base_env._robot.data.joint_vel[:, jids]                  # (N, 12)
        # DCMotor _clip_effort envelope magnitude at this joint speed:
        #   top    = sat*(1 - qd/vel_lim) clipped at +eff_lim
        #   bottom = sat*(-1 - qd/vel_lim) clipped at -eff_lim
        # The reachable |tau| upper bound is min(|top|, |bottom| region) -> we use the side
        # matching torque sign. Equivalent compact bound on |tau|:
        top = torch.clamp(sat_eff * (1.0 - qd / vel_lim), max=eff_lim)
        bottom = torch.clamp(sat_eff * (-1.0 - qd / vel_lim), min=-eff_lim)
        # bound for a given torque sign: positive tau bounded by top, negative by |bottom|
        pos_bound = torch.clamp(top, min=0.0)
        neg_bound = torch.clamp(-bottom, min=0.0)
        dyn_bound = torch.where(tau >= 0, pos_bound, neg_bound)       # (N, 12) >=0
        dyn_bound = torch.clamp(dyn_bound, min=1e-3)
        sat_mask = tau.abs() >= (args_cli.sat_frac * dyn_bound)       # (N, 12)
        sat_count = sat_mask.sum(dim=1).to(torch.int8)                # (N,)
        any_sat = sat_count > 0                                        # (N,)

        # ── mechanical power: sum positive tau*qdot (CoT numerator) ──
        mech_pow = torch.clamp(tau * qd, min=0.0).sum(dim=1)         # (N,) [W]

        # ── terrain / level / pose ──
        env_class = base_env._env_class
        level = base_env._terrain_levels
        rootxy = base_env._robot.data.root_pos_w[:, :2]

        # ── termination causes (set this step by _get_dones) ──
        # Skip spawn transient (episode_length<=1): mark as airborne=False/contact=False handled below.
        ep_len = base_env.episode_length_buf

        # store
        arr_class[step] = env_class.cpu().numpy()
        arr_level[step] = level.cpu().numpy().astype(np.int16)
        arr_contact[step] = contact.cpu().numpy()
        arr_airborne[step] = airborne.cpu().numpy()
        arr_peakfz[step] = peak_fz.cpu().numpy()
        arr_sat[step] = any_sat.cpu().numpy()
        arr_satcount[step] = sat_count.cpu().numpy()
        arr_mech_pow[step] = mech_pow.cpu().numpy()
        arr_rootxy[step] = rootxy.cpu().numpy()
        arr_dones[step] = dones.cpu().numpy().astype(bool)
        arr_c_tilt[step] = base_env._term_tilt.cpu().numpy()
        arr_c_low[step] = base_env._term_low_height.cpu().numpy()
        arr_c_base[step] = base_env._term_base_contact.cpu().numpy()
        arr_c_goal[step] = base_env._term_goal_reached.cpu().numpy()

        # invalidate spawn-transient samples (ep_len<=1)
        invalid = (ep_len <= 1).cpu().numpy()
        arr_airborne[step][invalid] = False
        arr_contact[step][invalid] = False

        if (step + 1) % 250 == 0:
            goals_so_far = int(arr_c_goal[: step + 1].sum())
            print(f"  step {step + 1}/{T}  ({time.time() - t_start:.1f}s)  goal_reached_events={goals_so_far}")

    total_time = time.time() - t_start
    print(f"[{SCRIPT_VERSION}] Collection done in {total_time:.1f}s\n")

    # ════════════════════════════════════════════════════════════════════════════
    #  POST-PROCESSING: segment episodes, drop failure episodes, per-terrain stats
    # ════════════════════════════════════════════════════════════════════════════
    dt = float(base_env.step_dt)
    eff_lim_np = eff_lim[0].detach().cpu().numpy()
    vel_lim_np = vel_lim[0].detach().cpu().numpy()

    # Build a per-sample "valid" mask: a sample (t, n) is VALID if the episode it belongs to
    # terminated by goal_reached or timeout (success/neutral), NOT by tilt/low_height/base_contact.
    # Segment each env-column at done==True boundaries.
    valid = np.ones((T, N), dtype=bool)
    # also a "progressing" diagnostic: did the env ever reach a goal during the run
    n_fail_episodes = 0
    n_total_episodes = 0
    n_goal_episodes = 0
    for n in range(N):
        done_steps = np.nonzero(arr_dones[:, n])[0]
        seg_start = 0
        for ds in done_steps:
            n_total_episodes += 1
            # failure if any failure cause flagged on the done step
            is_fail = bool(arr_c_tilt[ds, n] or arr_c_low[ds, n] or arr_c_base[ds, n])
            is_goal = bool(arr_c_goal[ds, n])
            if is_goal:
                n_goal_episodes += 1
            if is_fail and not is_goal:
                n_fail_episodes += 1
                valid[seg_start : ds + 1, n] = False
            seg_start = ds + 1
        # trailing (unterminated) segment kept as valid (in-progress)

    print(f"[{SCRIPT_VERSION}] Episodes: total={n_total_episodes}  failure={n_fail_episodes}  "
          f"goal_reached={n_goal_episodes}")
    if n_total_episodes > 0:
        print(f"[{SCRIPT_VERSION}]   failure rate = {100.0 * n_fail_episodes / n_total_episodes:.1f}%  "
              f"goal rate = {100.0 * n_goal_episodes / n_total_episodes:.1f}%")
    valid_frac = valid.mean()
    print(f"[{SCRIPT_VERSION}] valid (non-failure-episode) sample fraction = {valid_frac:.3f}\n")

    # PROGRESSION sanity: terrain level distribution start vs end
    lvl_start = arr_level[: max(1, T // 20)].mean()
    lvl_end = arr_level[-max(1, T // 20) :].mean()
    print(f"[{SCRIPT_VERSION}] mean terrain_level: first 5% steps={lvl_start:.2f}  last 5% steps={lvl_end:.2f}")
    if n_goal_episodes == 0:
        print(f"[{SCRIPT_VERSION}] *** WARNING: ZERO goal-reached episodes — robot may not be traversing. "
              "Per-terrain numbers may be contaminated. ***")
    print()

    class_ids = sorted({int(c) for c in np.unique(arr_class)})

    # ── helper: per-terrain metric over valid samples ──
    def terrain_mask(cls):
        return (arr_class == cls) & valid

    report_lines: list[str] = []

    def emit(s=""):
        print(s)
        report_lines.append(s)

    # ── (1) PRONK FREQUENCY per terrain (all-airborne fraction over valid steps) ──
    emit("=" * 64)
    emit("(1) ALL-AIRBORNE (pronk) FREQUENCY per terrain")
    emit("=" * 64)
    emit(f"{'terrain':<16}{'n_valid':>12}{'airborne_frac':>16}{'contact_mean_feet':>20}")
    pronk_by_terrain = {}
    for cls in class_ids:
        m = terrain_mask(cls)
        nv = int(m.sum())
        if nv == 0:
            continue
        af = float(arr_airborne[m].mean())
        cmf = float(arr_contact[m].sum(axis=-1).mean())  # mean #feet in contact
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"class_{cls}")
        pronk_by_terrain[cname] = af
        emit(f"{cname:<16}{nv:>12}{af:>16.4f}{cmf:>20.3f}")
    emit("")
    emit("  interpretation: airborne_frac = fraction of steps with ALL 4 feet off ground.")
    emit("  contact_mean_feet ~4 = stance/walk; low values + high airborne_frac = pronk-like flight.")
    emit("")

    # ── (2a) SATURATION per terrain (landing steps) ──
    # 'landing step' = step where a foot just made contact after airborne (airborne[t-1] & contact[t]).
    # Build landing mask: previous step all-airborne, this step >=1 foot contact.
    prev_airborne = np.zeros_like(arr_airborne)
    prev_airborne[1:] = arr_airborne[:-1]
    any_contact = arr_contact.any(axis=-1)
    landing_step = prev_airborne & any_contact & valid  # (T, N)

    emit("=" * 64)
    emit("(2a) ACTUATOR SATURATION vs DYNAMIC DCMotor envelope")
    emit("=" * 64)
    emit(f"  saturation flag: |applied_torque| >= {args_cli.sat_frac} * dynamic_bound(joint_vel)")
    emit(f"  static effort_limit (per actuated joint): {eff_lim_np.round(2).tolist()}")
    emit(f"  velocity_limit: {vel_lim_np.round(2).tolist()}   saturation_effort: {sat_eff}")
    emit("")
    emit(f"{'terrain':<16}{'all_steps_satfrac':>20}{'landing_satfrac':>18}{'mean_sat_joints':>18}")
    sat_by_terrain = {}
    for cls in class_ids:
        m_all = terrain_mask(cls)
        m_land = landing_step & (arr_class == cls)
        if int(m_all.sum()) == 0:
            continue
        sat_all = float(arr_sat[m_all].mean())
        sat_land = float(arr_sat[m_land].mean()) if int(m_land.sum()) > 0 else float("nan")
        msj = float(arr_satcount[m_all].mean())
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"class_{cls}")
        sat_by_terrain[cname] = sat_land
        emit(f"{cname:<16}{sat_all:>20.4f}{sat_land:>18.4f}{msj:>18.3f}")
    emit("")

    # ── (2b) LANDING IMPACT per terrain (peak vertical force at airborne->contact, bodyweight units) ──
    emit("=" * 64)
    emit("(2b) LANDING IMPACT: peak vertical contact force at airborne->contact transition")
    emit("=" * 64)
    emit(f"  normalized by bodyweight = {bodyweight_N:.1f} N (nominal mass {total_mass:.2f} kg)")
    emit(f"  peak vertical force = max over contact-history dim of |Fz| summed across 4 feet at landing")
    emit("")
    emit(f"{'terrain':<16}{'n_landings':>12}{'mean_BW':>12}{'p50_BW':>10}{'p95_BW':>10}{'max_BW':>10}")
    impact_by_terrain = {}
    for cls in class_ids:
        m_land = landing_step & (arr_class == cls)
        nl = int(m_land.sum())
        if nl == 0:
            continue
        # total vertical impact = sum of per-foot peak |Fz| across 4 feet at the landing step
        impact_N = arr_peakfz[m_land].sum(axis=-1)  # (n_landings,) N
        impact_bw = impact_N / bodyweight_N
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"class_{cls}")
        impact_by_terrain[cname] = float(np.median(impact_bw))
        emit(f"{cname:<16}{nl:>12}{impact_bw.mean():>12.3f}{np.percentile(impact_bw, 50):>10.3f}"
             f"{np.percentile(impact_bw, 95):>10.3f}{impact_bw.max():>10.3f}")
    emit("")

    # ── (3) RELATIVE CoT per terrain vs flat ──
    # CoT proxy = sum_t(positive mech power)*dt / (mass*g*distance_traveled_in_terrain)
    emit("=" * 64)
    emit("(3) RELATIVE Cost-of-Transport (CoT) proxy per terrain vs FLAT")
    emit("=" * 64)
    emit("  CoT = Σ(positive τ·q̇)·dt / (m·g·distance).  Distance = path length of base xy over valid steps.")
    emit("  Reported as ratio to FLAT (defensible relative claim). NOT a claim about alternating gait.")
    emit("")
    cot_by_terrain = {}
    emit(f"{'terrain':<16}{'energy_J':>14}{'dist_m':>12}{'CoT':>12}{'CoT_vs_flat':>14}")
    # compute energy and distance per terrain over valid samples.
    # distance: sum of per-step displacement magnitudes where consecutive steps are valid & same env.
    for cls in class_ids:
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"class_{cls}")
        m = terrain_mask(cls)
        if int(m.sum()) == 0:
            continue
        energy = float((arr_mech_pow[m]).sum() * dt)  # J
        # distance: per env, sum |Δxy| between consecutive valid same-terrain steps
        dist = 0.0
        for n in range(N):
            col_valid = m[:, n]
            if not col_valid.any():
                continue
            xy = arr_rootxy[:, n, :]
            d = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # (T-1,)
            step_pair_valid = col_valid[1:] & col_valid[:-1]
            # cap teleport jumps (resets move robot to spawn): drop > 0.5 m single-step jumps
            d = np.where(d > 0.5, 0.0, d)
            dist += float(d[step_pair_valid].sum())
        cot = energy / (total_mass * 9.81 * dist) if dist > 1e-3 else float("nan")
        cot_by_terrain[cname] = cot
        emit(f"{cname:<16}{energy:>14.1f}{dist:>12.1f}{cot:>12.4f}{'':>14}")
    # fill in ratio column
    flat_cot = cot_by_terrain.get("flat", float("nan"))
    emit("")
    emit(f"  flat CoT reference = {flat_cot:.4f}")
    for cname, cot in cot_by_terrain.items():
        ratio = cot / flat_cot if (flat_cot == flat_cot and flat_cot > 1e-6) else float("nan")
        emit(f"    {cname:<14} CoT={cot:.4f}  ({ratio:.2f}x flat)")
    emit("")

    # ════════════════════════════════════════════════════════════════════════════
    # (4) SPLIT-JUMP / FRONT-REAR COORDINATION ANALYSIS
    #   Front pair = feet 0,1 (FL,FR); Rear pair = feet 2,3 (RL,RR)  [order verified at runtime].
    #   (a) flight-segment count per traversal: number of maximal all-airborne runs, and how often a
    #       single obstacle crossing is broken into >1 flight (split jump).
    #   (b) re-takeoff signature: all-airborne -> PARTIAL contact (front-only OR rear-only, not all 4)
    #       -> all-airborne again WITHOUT a full 4-foot settle in between. This is the visual
    #       "front lands, takes off again before rear lands" = front/rear coordination failure.
    # ════════════════════════════════════════════════════════════════════════════
    front_ids = [0, 1]  # FL, FR (foot order [FL,FR,RL,RR] verified above)
    rear_ids = [2, 3]   # RL, RR
    front_contact = arr_contact[:, :, front_ids].any(axis=-1)  # (T,N) any front foot down
    rear_contact = arr_contact[:, :, rear_ids].any(axis=-1)    # (T,N) any rear foot down
    n_contact = arr_contact.sum(axis=-1)                       # (T,N) number of feet down
    full_settle = n_contact >= 3                               # 3-4 feet = settled stance

    emit("=" * 64)
    emit("(4) SPLIT-JUMP / FRONT-REAR COORDINATION (per terrain)")
    emit("=" * 64)
    emit("  flight segment = maximal run of all-airborne steps within a valid episode.")
    emit("  split-jump (re-takeoff) = airborne -> PARTIAL contact (front-only/rear-only) -> airborne")
    emit("    with NO full(>=3-foot) settle in between. Signals front/rear NOT used as one coordinated leap.")
    emit("")
    emit(f"{'terrain':<14}{'flights':>9}{'flt_steps':>11}{'mean_len':>10}{'retakeoff':>11}"
         f"{'retk/flight':>13}{'frontfirst':>12}{'rearfirst':>11}")
    split_by_terrain = {}
    for cls in class_ids:
        cname = _TERRAIN_CLASS_NAMES.get(cls, f"class_{cls}")
        cls_mask = arr_class == cls
        n_flights = 0
        flight_steps = 0
        flight_lengths = []
        n_retakeoff = 0
        n_front_first = 0  # landing transitions where front contacts before rear
        n_rear_first = 0
        for n in range(N):
            air = arr_airborne[:, n] & valid[:, n] & cls_mask[:, n]
            fc = front_contact[:, n]
            rc = rear_contact[:, n]
            fs = full_settle[:, n] & valid[:, n]
            v = valid[:, n] & cls_mask[:, n]
            t = 0
            while t < T:
                if air[t]:
                    # start of a flight segment
                    seg0 = t
                    while t < T and arr_airborne[t, n] and valid[t, n]:
                        t += 1
                    seg_len = t - seg0
                    if seg_len > 0:
                        n_flights += 1
                        flight_steps += seg_len
                        flight_lengths.append(seg_len)
                    # examine the landing window after this flight (next up to 8 steps)
                    w_end = min(T, t + 8)
                    saw_partial = False
                    saw_full = False
                    front_land_t = -1
                    rear_land_t = -1
                    for u in range(t, w_end):
                        if not v[u]:
                            break
                        if fs[u]:
                            saw_full = True
                            break  # settled — clean landing, stop window
                        partial = (fc[u] or rc[u]) and (n_contact[u, n] < 3)
                        if partial:
                            saw_partial = True
                            if fc[u] and front_land_t < 0:
                                front_land_t = u
                            if rc[u] and rear_land_t < 0:
                                rear_land_t = u
                        # re-takeoff: after a partial contact we go airborne again before full settle
                        if saw_partial and arr_airborne[u, n] and u > t:
                            n_retakeoff += 1
                            break
                    if front_land_t >= 0 and rear_land_t >= 0:
                        if front_land_t < rear_land_t:
                            n_front_first += 1
                        elif rear_land_t < front_land_t:
                            n_rear_first += 1
                else:
                    t += 1
        mean_len = float(np.mean(flight_lengths)) if flight_lengths else float("nan")
        retk_per_flight = (n_retakeoff / n_flights) if n_flights > 0 else float("nan")
        split_by_terrain[cname] = retk_per_flight
        emit(f"{cname:<14}{n_flights:>9}{flight_steps:>11}{mean_len:>10.2f}{n_retakeoff:>11}"
             f"{retk_per_flight:>13.3f}{n_front_first:>12}{n_rear_first:>11}")
    emit("")
    emit("  retk/flight high => obstacle crossings frequently split into multiple hops (coordination gap).")
    emit("  frontfirst >> rearfirst => front pair consistently lands before rear (expected for forward")
    emit("    motion); the concern is re-takeoff BETWEEN them (retakeoff column), not front-first itself.")
    emit("")

    # ── GO / NO-GO VERDICT ──
    # Thresholds defined BEFORE measurement (see report). X = saturation, Y = landing impact.
    X_SAT = 0.20   # >20% of landing steps saturated
    Y_IMPACT = 4.0  # >4 bodyweights peak landing force
    emit("=" * 64)
    emit("GO / NO-GO VERDICT")
    emit("=" * 64)
    emit(f"  Pre-registered thresholds: X_sat={X_SAT:.2f} (landing saturation frac), "
         f"Y_impact={Y_IMPACT:.1f} bodyweights (peak landing force).")
    emit("  Rule: if stair/step has (landing_satfrac > X) OR (median landing impact > Y),")
    emit("        AND is notably higher than gap (legitimate-jump reference) => B+C penalty JUSTIFIED.")
    emit("        Otherwise => HOLD (do not add penalty on this evidence).")
    emit("")
    gap_sat = sat_by_terrain.get("gap", float("nan"))
    gap_imp = impact_by_terrain.get("gap", float("nan"))
    gap_split = split_by_terrain.get("gap", float("nan"))
    for tname in ("stair", "step"):
        s = sat_by_terrain.get(tname, float("nan"))
        imp = impact_by_terrain.get(tname, float("nan"))
        emit(f"  [{tname}] landing_satfrac={s:.4f}  median_impact={imp:.3f} BW   "
             f"(gap ref: sat={gap_sat:.4f} imp={gap_imp:.3f} BW)")
    emit("")
    emit("  NOTE (gap is NOT a clean control): user observed gap crossings are SPLIT jumps")
    emit("  (front lands, re-takeoff before rear lands). Compare split-jump rate across terrains:")
    for tname in ("flat", "gap", "stair", "step", "hurdle"):
        sp = split_by_terrain.get(tname, float("nan"))
        if sp == sp:  # not nan
            emit(f"    {tname:<10} retakeoff/flight = {sp:.3f}")
    emit("  If gap/stair/step all show high retakeoff/flight, split-jumping is a pervasive coordination")
    emit("  inefficiency (front/rear not used as one leap) — a SEPARATE signal from saturation/impact.")
    emit("")

    # ── Save NPZ ──
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, "pronk_cost_result.npz")
    np.savez_compressed(
        npz_path,
        version=np.array([SCRIPT_VERSION]),
        checkpoint=np.array([resume_path]),
        task=np.array([args_cli.task]),
        foot_names=np.array(foot_names),
        arr_class=arr_class,
        arr_level=arr_level,
        arr_contact=arr_contact.astype(np.uint8),
        arr_airborne=arr_airborne.astype(np.uint8),
        arr_peakfz=arr_peakfz,
        arr_sat=arr_sat.astype(np.uint8),
        arr_satcount=arr_satcount,
        arr_mech_pow=arr_mech_pow,
        arr_rootxy=arr_rootxy,
        arr_dones=arr_dones.astype(np.uint8),
        arr_c_tilt=arr_c_tilt.astype(np.uint8),
        arr_c_low=arr_c_low.astype(np.uint8),
        arr_c_base=arr_c_base.astype(np.uint8),
        arr_c_goal=arr_c_goal.astype(np.uint8),
        valid=valid.astype(np.uint8),
        eff_lim=eff_lim_np,
        vel_lim=vel_lim_np,
        sat_eff=np.float32(sat_eff),
        total_mass=np.float32(total_mass),
        bodyweight_N=np.float32(bodyweight_N),
        dt=np.float32(dt),
        force_thresh=np.float32(args_cli.force_thresh),
        sat_frac=np.float32(args_cli.sat_frac),
        num_steps=np.int32(T),
        num_envs=np.int32(N),
    )
    print(f"\n[{SCRIPT_VERSION}] NPZ saved : {npz_path}")

    # ── Save MD (raw measured tables; full analysis report written separately) ──
    md_path = os.path.join(OUT_DIR, "pronk_cost_raw.md")
    header = [
        f"# Pronk-cost raw measurement ({SCRIPT_VERSION})",
        "",
        f"- checkpoint: `{resume_path}`",
        f"- task: `{args_cli.task}`",
        f"- run time: {ts}",
        f"- num_steps={T}  num_envs={N}  seed={args_cli.seed}  dt={dt}",
        f"- force_thresh={args_cli.force_thresh} N  sat_frac={args_cli.sat_frac}",
        f"- ACTUATOR (read at runtime from _robot.actuators['base_legs']):",
        f"  - effort_limit (per actuated joint) = {eff_lim_np.round(3).tolist()}",
        f"  - velocity_limit = {vel_lim_np.round(3).tolist()}",
        f"  - saturation_effort = {sat_eff}",
        f"- nominal total mass = {total_mass:.3f} kg  -> bodyweight = {bodyweight_N:.2f} N",
        f"- episodes: total={n_total_episodes} failure={n_fail_episodes} goal={n_goal_episodes}",
        f"- valid (non-failure-episode) sample fraction = {valid_frac:.3f}",
        f"- terrain_level first5%={lvl_start:.2f} last5%={lvl_end:.2f}",
        "",
        "```",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(report_lines) + "\n")
        f.write("```\n")
    print(f"[{SCRIPT_VERSION}] MD  saved : {md_path}")

    ts_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 64}")
    print(f"=== measure_pronk_cost {SCRIPT_VERSION} DONE @ {ts_end} ===")
    print(f"{'=' * 64}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
