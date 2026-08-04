# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit the ground-truth voxel column scanner on one pinned sub-terrain.

Why this exists
---------------
``fill_voxel_grid_gt`` models each grid column as at most two solid intervals: everything at or
below the downward ray's hit, and everything at or above the upward ray's hit. The upward hit is
*assumed* to be a ceiling. When a column's origin sits inside solid geometry — the robot walking
beside a step block or a stair riser taller than the sensor mount — the upward ray instead exits
through that block's **top** face, and the fill marks the whole volume above it as solid. The
observation then carries a phantom roof over terrain that has none.

The discriminating fact is structural: of every generator in ``parkour_terrains.py``, only
``parkour_crawl_terrain`` emits a ceiling. So on flat / hurdle / step / gap / stair the number of
columns reporting a ceiling must be **exactly zero**, and any nonzero count is a phantom. That is
the assertion this script makes, per terrain.

What it reports
---------------
Per pinned terrain, aggregated over the rollout and over **all** envs (never a sample — env 0 is
the reserved flat column and low-index envs cluster on flat, so reading a few envs makes a broken
fill look fine and a working one look broken):

* ``ceiling_cols`` — columns whose upward ray hit anything. Expected 0 off crawl.
* ``ground_missing_cols`` — columns whose downward ray missed, i.e. no ground within range.
* ``occ_cells`` — occupied cells in the built grid, the volume-fill sanity check.
* when the scanner emits a third (top-down) ray set, the pillar/ceiling split and the
  ``top - up`` margin distribution that the discriminator's epsilon is chosen from.

The policy is only a **pose generator** here: it walks the robot onto obstacles so columns
actually straddle geometry. Nothing about the policy's quality is measured, so a checkpoint
trained against a different observation is still a valid driver.

Usage::

    ./isaaclab.sh -p scripts/demos/verify_voxel_gt.py \
        --task Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0 \
        --checkpoint /abs/path/model_49999.pt \
        --terrain step --level 6 --num_envs 64 --steps 600 --headless \
        --out_path reports/.../metrics/gt_audit_step.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# local imports — reuse the RSL-RL CLI arg helpers from the play/train scripts.
import os as _os_boot
import sys

from isaaclab.app import AppLauncher

sys.path.insert(
    0,
    _os_boot.path.join(
        _os_boot.path.dirname(_os_boot.path.abspath(__file__)), "..", "reinforcement_learning", "rsl_rl"
    ),
)
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser(description="Audit the GT voxel column scanner on a pinned sub-terrain.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--terrain",
    type=str,
    required=True,
    choices=["hurdle", "step", "gap", "stair", "flat", "crawl"],
    help="Which single sub-terrain to isolate (proportion 1.0, all others 0.0).",
)
parser.add_argument("--out_path", type=str, required=True, help="Output JSON path for the audit statistics.")
parser.add_argument("--level", type=int, required=True, help="Terrain level to pin ALL envs to (0..num_rows-1).")
parser.add_argument("--steps", type=int, default=600, help="Rollout steps.")
parser.add_argument("--warmup", type=int, default=100, help="Steps to skip before accumulating (spawn pads are flat).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerParkour
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP, OnPolicyRunnerAMPBase
from rsl_rl.runners.on_policy_runner_parkour_amp import (
    OnPolicyRunnerParkourAMP,
    OnPolicyRunnerParkourAMPLidar,
    OnPolicyRunnerParkourAMPVoxel,
)

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.parkour.parkour_env_cfg import (
    TERRAIN_CLASS_CRAWL,
    TERRAIN_CLASS_FLAT,
    TERRAIN_CLASS_GAP,
    TERRAIN_CLASS_HURDLE,
    TERRAIN_CLASS_STAIR,
    TERRAIN_CLASS_STEP,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

TERRAIN_KEY = {
    "flat": "parkour_flat",
    "hurdle": "parkour_hurdle",
    "step": "parkour_step",
    "gap": "parkour_gap",
    "stair": "parkour_stair",
    "crawl": "parkour_crawl",
}

_TERRAIN_CLASS_BY_NAME = {
    "flat": TERRAIN_CLASS_FLAT,
    "hurdle": TERRAIN_CLASS_HURDLE,
    "step": TERRAIN_CLASS_STEP,
    "gap": TERRAIN_CLASS_GAP,
    "stair": TERRAIN_CLASS_STAIR,
    "crawl": TERRAIN_CLASS_CRAWL,
}

# Only ``parkour_crawl_terrain`` builds a ceiling; every other generator is solid-from-the-ground-up.
CEILING_TERRAINS = {"crawl"}

# Must match ``fill_voxel_grid_gt``'s default, or the audit grades a different rule than the env
# applies. Pillars agree to float error; crawl's ceiling and its top-blocker are 0.60 m apart.
PILLAR_EPS = 0.05

# Fraction of (env, frame) samples that may show an isolated single-column ceiling on a terrain
# that has none. Set an order of magnitude above the grazing rate measured on pinned stair
# (5.3e-4) and three orders below the systematic phantom roof it replaced (0.989).
GRAZING_FRAC = 5e-3

_RUNNER_CLASSES = {
    "OnPolicyRunner": OnPolicyRunner,
    "OnPolicyRunnerParkour": OnPolicyRunnerParkour,
    "OnPolicyRunnerAMP": OnPolicyRunnerAMP,
    "OnPolicyRunnerAMPBase": OnPolicyRunnerAMPBase,
    "OnPolicyRunnerParkourAMP": OnPolicyRunnerParkourAMP,
    "OnPolicyRunnerParkourAMPVoxel": OnPolicyRunnerParkourAMPVoxel,
    "OnPolicyRunnerParkourAMPLidar": OnPolicyRunnerParkourAMPLidar,
}


def _isolate_terrain(env_cfg, terrain_name: str) -> None:
    """Set the target sub-terrain to (almost) the whole grid, in place.

    One flat column stays reserved because the RandomGoal family asserts a flat column exists for
    AMP flat-masking; ``_col_to_class`` maps column 0 to flat when ``flat`` holds ``1/num_cols``.
    """
    target_key = TERRAIN_KEY[terrain_name]
    tg = env_cfg.terrain.terrain_generator
    if target_key not in tg.sub_terrains:
        raise KeyError(f"Terrain '{terrain_name}' -> key '{target_key}' not in {list(tg.sub_terrains.keys())}")
    for key in list(tg.sub_terrains.keys()):
        tg.sub_terrains[key].proportion = 0.0
    if target_key == "parkour_flat":
        tg.sub_terrains["parkour_flat"].proportion = 1.0
    else:
        flat_prop = 1.0 / float(tg.num_cols)
        tg.sub_terrains["parkour_flat"].proportion = flat_prop
        tg.sub_terrains[target_key].proportion = 1.0 - flat_prop
    active = {k: v.proportion for k, v in tg.sub_terrains.items() if v.proportion > 0.0}
    print(f"[gt-audit] isolated terrain='{terrain_name}' key='{target_key}'. active proportions: {active}")


def _column_rays(scanner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Split the column scanner's hits into signed down / up / top-down surface heights.

    Heights are returned in the grid frame (z = 0 at the sensor origin), so ``down`` is negative
    below the origin and ``up`` positive above it. Misses are pushed out of the volume in the
    direction that marks nothing: ``-inf`` for surfaces below, ``+inf`` for surfaces above.

    Returns:
        ``(ground_z, ceil_z, top_z)`` each ``(N, n_col)``; ``top_z`` is None for a 2-ray pattern.
    """
    hits_z = scanner.data.ray_hits_w[..., 2]
    org_z = scanner.data.pos_w[:, 2].unsqueeze(1)
    # The pattern cfg owns the column count, so the number of ray sets follows from the ray count
    # instead of being guessed — a 2-ray and a 3-ray pattern must not be told apart by divisibility.
    pc = scanner.cfg.pattern_cfg
    n_col = pc.nx * pc.ny
    n_sets = hits_z.shape[1] // n_col

    rel = hits_z - org_z  # signed height relative to the origin
    neg_inf, pos_inf = -float("inf"), float("inf")
    ground = torch.nan_to_num(rel[:, :n_col], nan=neg_inf, posinf=neg_inf, neginf=neg_inf)
    ceil = torch.nan_to_num(rel[:, n_col : 2 * n_col], nan=pos_inf, posinf=pos_inf, neginf=pos_inf)
    top = None
    if n_sets >= 3:
        top = torch.nan_to_num(rel[:, 2 * n_col : 3 * n_col], nan=neg_inf, posinf=neg_inf, neginf=neg_inf)
    return ground, ceil, top


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out on one pinned terrain and audit the column scanner every step."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if not (args_cli.checkpoint and os.path.isabs(args_cli.checkpoint) and os.path.isfile(args_cli.checkpoint)):
        raise FileNotFoundError(f"--checkpoint must be an existing absolute path (got {args_cli.checkpoint!r})")
    resume_path = args_cli.checkpoint
    env_cfg.log_dir = os.path.dirname(resume_path)

    _isolate_terrain(env_cfg, args_cli.terrain)
    if hasattr(env_cfg, "terrain_max_init_level"):
        env_cfg.terrain_max_init_level = 8
    if hasattr(env_cfg.terrain, "max_init_terrain_level"):
        env_cfg.terrain.max_init_terrain_level = 8
    if hasattr(env_cfg, "enable_random_goal"):
        env_cfg.enable_random_goal = False
    env_cfg.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.unwrapped.set_debug_vis(False)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise RuntimeError("MARL envs are not supported by this audit")

    out_path = os.path.abspath(args_cli.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    import inspect as _inspect

    from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPOParkour

    _accepted = set(_inspect.signature(_VendoredPPOParkour.__init__).parameters.keys()) - {"self"}
    _cfg_dict = agent_cfg.to_dict()
    _cfg_dict["algorithm"] = {k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted or k == "class_name"}
    if agent_cfg.class_name not in _RUNNER_CLASSES:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    cfg_for_runner = agent_cfg.to_dict() if agent_cfg.class_name == "OnPolicyRunner" else _cfg_dict
    runner = _RUNNER_CLASSES[agent_cfg.class_name](env, cfg_for_runner, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    print(f"[gt-audit] loaded {resume_path} (runner={agent_cfg.class_name})")

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    u = env.unwrapped
    scanner = getattr(u, "_voxel_col_scanner", None)
    if scanner is None:
        raise RuntimeError(
            "env has no _voxel_col_scanner — the task must set voxel_gt_columns=True "
            "(e.g. Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0)"
        )

    origins = u._terrain.terrain_origins
    num_rows = int(origins.shape[0])
    level = max(0, min(args_cli.level, num_rows - 1))
    target_class = _TERRAIN_CLASS_BY_NAME[args_cli.terrain]
    cols = (u._col_to_class == target_class).nonzero(as_tuple=False).flatten()
    if cols.numel() == 0:
        raise RuntimeError(f"no column of class {target_class} ({args_cli.terrain}) exists")
    for i in range(u.num_envs):
        col = int(cols[i % cols.numel()].item())
        u._terrain_levels[i] = level
        u._terrain_types[i] = col
        u._env_class[i] = u._col_to_class[col]
        u._terrain.env_origins[i] = origins[level, col]
        u._skip_curriculum[i] = True
    u._reset_idx(torch.arange(u.num_envs, device=u.device))
    print(f"[gt-audit] pinned {u.num_envs} envs to terrain='{args_cli.terrain}' level={level}")

    obs = env.get_observations()

    acc = {
        "ceiling_cols_sum": 0.0,
        "ceiling_cols_max": 0,
        "envs_with_ceiling_sum": 0.0,
        "ground_missing_sum": 0.0,
        "ground_missing_max": 0,
        "occ_sum": 0.0,
        "occ_min": None,
        "occ_max": 0,
        "relief_max": 0.0,
        "pillar_cols_sum": 0.0,
        "true_ceiling_cols_sum": 0.0,
        "hidden_block_cols_sum": 0.0,
        "hidden_block_cols_max": 0,
        "full_cols_sum": 0.0,
        "full_cols_max": 0,
        "ground_near_origin_sum": 0.0,
        "ground_near_origin_max": 0,
        "ground_samples": [],
        "empty_grid_sum": 0.0,
        "base_z_sum": 0.0,
        "base_z_max": 0.0,
        "base_z_when_empty_sum": 0.0,
        "base_z_when_empty_n": 0,
        "margin_min": None,
        "margin_samples": [],
        "n_frames": 0,
    }
    has_top = None

    for t in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        u._skip_curriculum[:] = True
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            if t < args_cli.warmup:
                continue
            ground_raw, ceil_raw_ray, top = _column_rays(scanner)
            has_top = top is not None
            # Grade the classification the env applied, not a re-derivation from raw ray hits.
            # The two differ exactly where the normals overrule a hit, which is the whole fix.
            diag = getattr(u, "_voxel_gt_diag", None)
            if diag is None:
                raise RuntimeError("env did not publish _voxel_gt_diag; audit cannot grade the shipped grid")
            ground, ceil = diag["ground_z"], diag["ceil_z"]

            ceil_raw = ceil_raw_ray
            # ``ceil_z`` finite means the env put a ceiling there — pillars were already
            # reclassified to +inf inside the fill.
            ceil_hit = torch.isfinite(ceil)
            pillar_mask = diag["pillar"]
            per_env_ceiling = ceil_hit.sum(dim=1)
            acc["ceiling_cols_sum"] += float(per_env_ceiling.float().mean().item())
            acc["ceiling_cols_max"] = max(acc["ceiling_cols_max"], int(per_env_ceiling.max().item()))
            acc["envs_with_ceiling_sum"] += float((per_env_ceiling > 0).float().mean().item())

            miss = ~torch.isfinite(ground) & ~diag["slab"]
            acc["ground_missing_sum"] += float(miss.sum(dim=1).float().mean().item())
            acc["ground_missing_max"] = max(acc["ground_missing_max"], int(miss.sum(dim=1).max().item()))

            # A "ground" at the sensor's own height is impossible for a standing robot: it means
            # the downward ray exited a solid the origin was inside (a crawl ceiling slab) and the
            # fill then plugs the corridor below it. Mirror of the phantom-ceiling case.
            # Applied ground at the sensor's own height is impossible for a standing robot; it is
            # the signature of a downward ray that exited a solid rather than entering the floor.
            # Restricted to columns whose ground was accepted as *real* ground: a pillar's ground
            # is its block's top face, which legitimately sits at mount height when the robot walks
            # beside a block its own size, so including pillars counts correct fills as failures.
            near = diag["real_ground"] & torch.isfinite(ground) & (ground.abs() < 0.15)
            acc["ground_near_origin_sum"] += float(near.sum(dim=1).float().mean().item())
            acc["ground_near_origin_max"] = max(acc["ground_near_origin_max"], int(near.sum(dim=1).max().item()))
            g = ground[torch.isfinite(ground)]
            if g.numel() > 0 and len(acc["ground_samples"]) < 20000:
                acc["ground_samples"].extend(g.flatten().tolist()[:2000])

            fin = torch.where(torch.isfinite(ground_raw), ground_raw, torch.zeros_like(ground_raw))
            acc["relief_max"] = max(acc["relief_max"], float((fin.amax(1) - fin.amin(1)).max().item()))

            grid = u._voxel_grid
            occ = (grid == 1).flatten(1).sum(1).float()
            # An all-zero grid is not "flat ground" — it is the terrain having left the volume
            # entirely, which the policy cannot tell from open space. The grid floor sits only
            # ``|z_range[0]| - mount_height`` below the surface, so a leap can drop the ground out
            # of it. Tracked separately because occ_min alone hides it: one env keeping ground is
            # enough to keep the minimum positive.
            acc["empty_grid_sum"] += float((occ == 0).float().mean().item())
            # A column with every cell solid has no free space for the robot anywhere in it. On a
            # crawl corridor that is the plugged-tunnel signature; a pillar taller than the grid
            # top is the only legitimate source, so this is read per terrain, not as a global gate.
            full_col = (grid == 1).all(dim=-1).flatten(1).sum(1).float()
            acc["full_cols_sum"] += float(full_col.mean().item())
            acc["full_cols_max"] = max(acc["full_cols_max"], int(full_col.max().item()))
            base_z = u._robot.data.root_pos_w[:, 2] - u._terrain.env_origins[:, 2]
            acc["base_z_sum"] += float(base_z.mean().item())
            acc["base_z_max"] = max(acc["base_z_max"], float(base_z.max().item()))
            if bool((occ == 0).any()):
                acc["base_z_when_empty_sum"] += float(base_z[occ == 0].mean().item())
                acc["base_z_when_empty_n"] += 1
            acc["occ_sum"] += float(occ.mean().item())
            acc["occ_min"] = int(occ.min().item()) if acc["occ_min"] is None else min(acc["occ_min"], int(occ.min()))
            acc["occ_max"] = max(acc["occ_max"], int(occ.max().item()))

            if top is not None:
                # Margin between the upward hit and the topmost surface, over every column that
                # produced both. Its lower tail is what PILLAR_EPS must sit below.
                both = torch.isfinite(ceil_raw) & torch.isfinite(top)
                m = (top - ceil_raw)[both]
                if m.numel() > 0:
                    mn = float(m.min().item())
                    acc["margin_min"] = mn if acc["margin_min"] is None else min(acc["margin_min"], mn)
                    if len(acc["margin_samples"]) < 20000:
                        acc["margin_samples"].extend(m.flatten().tolist()[:2000])
                acc["pillar_cols_sum"] += float(pillar_mask.sum(dim=1).float().mean().item())
                acc["true_ceiling_cols_sum"] += float(per_env_ceiling.float().mean().item())
                # Geometry the origin-down ray cannot see: the topmost surface sits a full cell
                # above the ground surface while no ceiling was classified. The two-interval model
                # would drop it. Counted before deciding whether it needs handling at all.
                hidden = (
                    torch.isfinite(top)
                    & torch.isfinite(ground)
                    & (top > ground + float(u._voxel_cfg.resolution))
                    & ~pillar_mask
                    & ~ceil_hit
                    & ~diag["slab"]
                )
                acc["hidden_block_cols_sum"] += float(hidden.sum(dim=1).float().mean().item())
                acc["hidden_block_cols_max"] = max(acc["hidden_block_cols_max"], int(hidden.sum(dim=1).max().item()))
            acc["n_frames"] += 1

    n = max(1, acc["n_frames"])
    expects_ceiling = args_cli.terrain in CEILING_TERRAINS
    ceiling_mean = acc["ceiling_cols_sum"] / n
    frac_env_frames = acc["envs_with_ceiling_sum"] / n
    if expects_ceiling:
        verdict_ceiling = "PASS" if ceiling_mean > 0.0 else "FAIL_NO_CEILING"
    elif acc["ceiling_cols_max"] == 0:
        verdict_ceiling = "PASS"
    elif acc["ceiling_cols_max"] <= 1 and frac_env_frames < GRAZING_FRAC:
        # A single column in isolated frames is a ray grazing a mesh edge, not the systematic roof
        # the pillar test exists to remove — that one covered hundreds of columns in ~99% of frames.
        # Reported separately rather than folded into PASS so a regression cannot hide behind it.
        verdict_ceiling = "PASS_WITH_GRAZING"
    else:
        verdict_ceiling = "FAIL_PHANTOM_CEILING"

    ms = sorted(acc.pop("margin_samples"))
    gs = sorted(acc.pop("ground_samples"))
    result = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "terrain": args_cli.terrain,
        "level": level,
        "num_envs": int(u.num_envs),
        "steps": args_cli.steps,
        "warmup": args_cli.warmup,
        "frames_accumulated": acc["n_frames"],
        "ray_sets": 3 if has_top else 2,
        "terrain_has_ceiling_by_construction": expects_ceiling,
        "ceiling_cols_mean_per_env": ceiling_mean,
        "ceiling_cols_max": acc["ceiling_cols_max"],
        "frac_envs_with_any_ceiling": acc["envs_with_ceiling_sum"] / n,
        "ground_missing_cols_mean": acc["ground_missing_sum"] / n,
        "ground_missing_cols_max": acc["ground_missing_max"],
        "occ_cells_mean": acc["occ_sum"] / n,
        "occ_cells_min": acc["occ_min"],
        "occ_cells_max": acc["occ_max"],
        "ground_relief_max_m": acc["relief_max"],
        "empty_grid_frac": acc["empty_grid_sum"] / n,
        "full_cols_mean": acc["full_cols_sum"] / n,
        "full_cols_max": acc["full_cols_max"],
        "verdict_ground": ("PASS" if acc["ground_near_origin_max"] == 0 else "FAIL_PHANTOM_GROUND"),
        "ground_near_origin_cols_mean": acc["ground_near_origin_sum"] / n,
        "ground_near_origin_cols_max": acc["ground_near_origin_max"],
        "ground_z_p05_m": (gs[int(0.05 * len(gs))] if gs else None),
        "ground_z_p50_m": (gs[len(gs) // 2] if gs else None),
        "ground_z_p95_m": (gs[int(0.95 * len(gs))] if gs else None),
        "base_z_mean_m": acc["base_z_sum"] / n,
        "base_z_max_m": acc["base_z_max"],
        "base_z_when_empty_mean_m": (
            acc["base_z_when_empty_sum"] / acc["base_z_when_empty_n"] if acc["base_z_when_empty_n"] else None
        ),
        "verdict_ceiling": verdict_ceiling,
    }
    if has_top:
        result.update(
            {
                "pillar_eps_m": PILLAR_EPS,
                "pillar_cols_mean_per_env": acc["pillar_cols_sum"] / n,
                "true_ceiling_cols_mean_per_env": acc["true_ceiling_cols_sum"] / n,
                "hidden_block_cols_mean_per_env": acc["hidden_block_cols_sum"] / n,
                "hidden_block_cols_max": acc["hidden_block_cols_max"],
                "top_minus_up_margin_min_m": acc["margin_min"],
                "top_minus_up_margin_p01_m": (ms[int(0.01 * len(ms))] if ms else None),
                "top_minus_up_margin_p50_m": (ms[len(ms) // 2] if ms else None),
            }
        )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[gt-audit] {args_cli.terrain}: ceiling={verdict_ceiling} ground={result['verdict_ground']} | "
        f"ceil_cols={ceiling_mean:.1f}/max{acc['ceiling_cols_max']} "
        f"near_origin={result['ground_near_origin_cols_mean']:.1f}/max{acc['ground_near_origin_max']} "
        f"full_cols={result['full_cols_mean']:.1f} occ={result['occ_cells_mean']:.0f} -> {out_path}"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
