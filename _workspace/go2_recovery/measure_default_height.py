"""Measure Go2 default-pose standing base height via gym.make (Go2Recovery-v0).

목적:
    Go2가 default joint pose로 완전히 섰을 때 실제 base height를 측정,
    recovery reward의 target_height=0.27m와 비교하여
    "target이 낮아 앉음을 유인하는가"를 검증한다.

방식 A — joint 직접 강제 write (새로운 방식):
    - env.reset() 후 매 step마다 write_joint_state_to_sim()으로 joint_pos를
      default_joint_pos에 직접 고정(PD 아님). root는 자유(중력+발 접촉).
    - 초기 root: z=0.40 공중, orientation=[1,0,0,0](wxyz, 직립). 자유낙하→발 접지.
    - 300 step 안정화 후 tail 평균으로 base height 측정.
    - 강제 검증: 측정된 실제 joint_pos vs default_joint_pos 차이(≈0이어야 함).

방식 B — PD hold / settle override (기존 방식):
    - env_cfg settle_max_steps 경로로 default_joint_pos를 PD target으로 유지.
    - 실제 joint_pos vs default 차이(PD 오차)도 함께 출력.

두 방식의 height 차이 → "PD hold가 부정확했는가" 드러남.
target_height=0.27과 비교.

실행:
    ./isaaclab.sh -p _workspace/go2_recovery/measure_default_height.py --headless

    # step 수 조정:
    ./isaaclab.sh -p _workspace/go2_recovery/measure_default_height.py --headless \\
        --settle_steps 500 --tail_steps 80 --num_envs 64

제약:
    - articulation 직접 spawn 금지 (hang 원인) — gym.make env 경로만 사용.
    - python 직접 실행 금지 — ./isaaclab.sh -p 사용.
"""

from __future__ import annotations

import argparse
import sys

# ── 1. Argparse before AppLauncher ────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Go2 default-pose standing height measurement.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs.")
parser.add_argument(
    "--settle_steps",
    type=int,
    default=400,
    help="Steps to stabilize (both methods share this count). Default: 400.",
)
parser.add_argument(
    "--tail_steps",
    type=int,
    default=80,
    help="Last N steps used for height averaging. Default: 80.",
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Post-launch imports ────────────────────────────────────────────────────

import torch

import isaaclab_tasks  # noqa: F401  (registers all tasks)
from isaaclab_tasks.utils.hydra import hydra_task_config

import gymnasium as gym  # noqa: E402 (after AppLauncher)

SETTLE_STEPS: int = args_cli.settle_steps
TAIL_STEPS: int = args_cli.tail_steps
NUM_ENVS: int = args_cli.num_envs

TARGET_HEIGHT: float = 0.27
DROP_HEIGHT: float = 0.40       # 초기 root z (발이 착지하도록 충분히 낮춤)

# ── 3. Helper: reset all envs to aerial-upright + default joints ───────────────

def _force_reset_aerial(raw_env, device: str) -> None:
    """모든 env를 직립 공중(z=DROP_HEIGHT) + default joint 상태로 직접 write."""
    num_envs: int = raw_env.num_envs
    robot = raw_env._robot

    all_ids = torch.arange(num_envs, dtype=torch.long, device=device)

    # root pose: env origin + z 높이 오프셋, quat=[1,0,0,0](wxyz=직립)
    root_pos = robot.data.default_root_state[:, :3].clone()
    root_pos += raw_env._terrain.env_origins  # env 간격 반영
    root_pos[:, 2] = raw_env._terrain.env_origins[:, 2] + DROP_HEIGHT

    quat_upright = torch.zeros(num_envs, 4, device=device)
    quat_upright[:, 0] = 1.0          # w=1 (unit quaternion, 회전 없음)

    root_pose_7 = torch.cat([root_pos, quat_upright], dim=-1)  # (num_envs, 7)
    root_vel_6 = torch.zeros(num_envs, 6, device=device)

    # joint: 정확히 default
    joint_pos = robot.data.default_joint_pos.clone()  # (num_envs, 12)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_root_pose_to_sim(root_pose_7, all_ids)
    robot.write_root_velocity_to_sim(root_vel_6, all_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, all_ids)


def _force_joint_default(raw_env, device: str) -> None:
    """매 step 호출: joint_pos를 default에 직접 고정, root는 건드리지 않음."""
    robot = raw_env._robot
    num_envs: int = raw_env.num_envs
    all_ids = torch.arange(num_envs, dtype=torch.long, device=device)

    joint_pos = robot.data.default_joint_pos.clone()  # (num_envs, 12)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, all_ids)


# ── 4. Main ───────────────────────────────────────────────────────────────────


@hydra_task_config("Go2Recovery-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Measure base height: method A (joint force-write) and method B (PD/settle)."""

    device = agent_cfg.device if hasattr(agent_cfg, "device") else "cuda:0"

    # ══════════════════════════════════════════════════════════════════════════
    # 방식 A — joint 직접 강제 write
    # ══════════════════════════════════════════════════════════════════════════

    print()
    print("=" * 65)
    print("  [방식 A] Joint 직접 강제 write (write_joint_state_to_sim)")
    print("=" * 65)

    # cfg A: settle 비활성화 (PD 경로 사용 안 함), standing-start 100%
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.fall_standing_ratio = 1.0
    env_cfg.fall_sitting_ratio = 0.0
    env_cfg.settle_max_steps = 0          # settle override 비활성 (방식 A는 직접 write)
    env_cfg.terminate_on_success = False
    env_cfg.sim.device = device

    env_a = gym.make("Go2Recovery-v0", cfg=env_cfg)
    raw_a = env_a.unwrapped  # type: ignore[attr-defined]

    num_envs: int = raw_a.num_envs
    num_actions: int = raw_a.cfg.action_space
    zero_actions = torch.zeros(num_envs, num_actions, device=device)

    # reset 후 즉시 aerial-upright + default joint으로 덮어씀
    env_a.reset()
    _force_reset_aerial(raw_a, device)

    height_buf_a: list[torch.Tensor] = []
    joint_diff_buf_a: list[torch.Tensor] = []

    print(f"  num_envs    : {num_envs}")
    print(f"  settle_steps: {SETTLE_STEPS}")
    print(f"  tail_steps  : {TAIL_STEPS}")
    print(f"  drop_height : {DROP_HEIGHT} m (initial root z above terrain)")
    print()
    print(f"  Running {SETTLE_STEPS} steps (joint forced to default every step) ...")

    for step_idx in range(SETTLE_STEPS):
        # joint를 default로 강제 (매 step, sim에 직접 write)
        _force_joint_default(raw_a, device)

        env_a.step(zero_actions)

        h = raw_a._robot.data.root_pos_w[:, 2].clone()  # (num_envs,)

        # 실제 joint_pos - default_joint_pos 차이 (강제가 유지됐는지 검증)
        actual_jp = raw_a._robot.data.joint_pos.clone()            # (num_envs, 12)
        default_jp = raw_a._robot.data.default_joint_pos.clone()   # (num_envs, 12)
        diff = (actual_jp - default_jp).abs()                      # (num_envs, 12)

        if step_idx >= SETTLE_STEPS - TAIL_STEPS:
            height_buf_a.append(h)
            joint_diff_buf_a.append(diff)

        if (step_idx + 1) % 100 == 0:
            print(
                f"  step {step_idx + 1:4d}/{SETTLE_STEPS}  "
                f"height mean={h.mean().item():.4f} m  "
                f"std={h.std().item():.5f} m  "
                f"joint_diff_max={diff.max().item():.5f} rad"
            )

    env_a.close()

    # 통계 A
    ht_a = torch.stack(height_buf_a, dim=0)          # (TAIL, num_envs)
    jd_a = torch.stack(joint_diff_buf_a, dim=0)      # (TAIL, num_envs, 12)

    ha_mean = ht_a.flatten().mean().item()
    ha_std  = ht_a.flatten().std().item()
    ha_temporal_std = ht_a.mean(dim=1).std().item()

    jda_max  = jd_a.max().item()
    jda_mean = jd_a.mean().item()

    # ══════════════════════════════════════════════════════════════════════════
    # 방식 B — PD hold via settle override (기존 방식)
    # ══════════════════════════════════════════════════════════════════════════

    print()
    print("=" * 65)
    print("  [방식 B] PD hold via settle override (set_joint_position_target)")
    print("=" * 65)

    # cfg B: settle 활성화, standing-start 100%
    # env_cfg 재활용이 안 되므로 새 hydra cfg 없이 동일 객체 재사용
    # → 이미 수정된 env_cfg에 settle만 켬
    env_cfg.settle_max_steps = SETTLE_STEPS
    env_cfg.sim.device = device

    env_b = gym.make("Go2Recovery-v0", cfg=env_cfg)
    raw_b = env_b.unwrapped  # type: ignore[attr-defined]

    num_actions_b: int = raw_b.cfg.action_space
    zero_actions_b = torch.zeros(num_envs, num_actions_b, device=device)

    env_b.reset()

    # settle 카운터를 0으로 강제 → 전체 SETTLE_STEPS 동안 settle 유지
    raw_b._settle_steps[:] = SETTLE_STEPS
    raw_b._settle_counter[:] = 0

    height_buf_b: list[torch.Tensor] = []
    joint_diff_buf_b: list[torch.Tensor] = []

    print(f"  settle_steps: {SETTLE_STEPS}")
    print()
    print(f"  Running {SETTLE_STEPS} steps (PD target = default_joint_pos) ...")

    for step_idx in range(SETTLE_STEPS):
        env_b.step(zero_actions_b)

        h = raw_b._robot.data.root_pos_w[:, 2].clone()

        actual_jp = raw_b._robot.data.joint_pos.clone()
        default_jp = raw_b._robot.data.default_joint_pos.clone()
        diff = (actual_jp - default_jp).abs()

        if step_idx >= SETTLE_STEPS - TAIL_STEPS:
            height_buf_b.append(h)
            joint_diff_buf_b.append(diff)

        if (step_idx + 1) % 100 == 0:
            print(
                f"  step {step_idx + 1:4d}/{SETTLE_STEPS}  "
                f"height mean={h.mean().item():.4f} m  "
                f"std={h.std().item():.5f} m  "
                f"joint_diff_max={diff.max().item():.5f} rad"
            )

    env_b.close()

    # 통계 B
    ht_b = torch.stack(height_buf_b, dim=0)
    jd_b = torch.stack(joint_diff_buf_b, dim=0)

    hb_mean = ht_b.flatten().mean().item()
    hb_std  = ht_b.flatten().std().item()
    hb_temporal_std = ht_b.mean(dim=1).std().item()

    jdb_max  = jd_b.max().item()
    jdb_mean = jd_b.mean().item()

    # ══════════════════════════════════════════════════════════════════════════
    # 최종 보고
    # ══════════════════════════════════════════════════════════════════════════

    diff_ab = ha_mean - hb_mean
    diff_a_target = ha_mean - TARGET_HEIGHT
    diff_b_target = hb_mean - TARGET_HEIGHT

    print()
    print("=" * 65)
    print("  Go2 Default-Pose Standing Base Height — Final Report")
    print("=" * 65)
    print()
    print("  [방식 A] Joint 직접 강제 write (write_joint_state_to_sim)")
    print(f"    base height (tail-{TAIL_STEPS} mean) : {ha_mean:.4f} m  (std={ha_std:.5f} m)")
    print(f"    수렴 (step-간 std)               : {ha_temporal_std:.6f} m", end="")
    print("  ← CONVERGED" if ha_temporal_std < 1e-3 else "  ← NOT YET CONVERGED")
    print(f"    실제 joint_pos vs default")
    print(f"      max  abs diff : {jda_max:.6f} rad  (강제 검증: ≈0 이어야)")
    print(f"      mean abs diff : {jda_mean:.6f} rad")
    print()
    print("  [방식 B] PD hold (settle override)")
    print(f"    base height (tail-{TAIL_STEPS} mean) : {hb_mean:.4f} m  (std={hb_std:.5f} m)")
    print(f"    수렴 (step-간 std)               : {hb_temporal_std:.6f} m", end="")
    print("  ← CONVERGED" if hb_temporal_std < 1e-3 else "  ← NOT YET CONVERGED")
    print(f"    실제 joint_pos vs default (PD 오차)")
    print(f"      max  abs diff : {jdb_max:.6f} rad")
    print(f"      mean abs diff : {jdb_mean:.6f} rad")
    print()
    print("  ── 비교 ─────────────────────────────────────────────────────")
    print(f"    방식 A − 방식 B height 차이 : {diff_ab:+.4f} m", end="")
    if abs(diff_ab) < 2e-3:
        print("  (두 방식 일치 — PD hold 오차 무시 가능)")
    elif diff_ab > 0:
        print("  (A > B: PD hold 시 실제 joint이 굽어 height가 낮았음)")
    else:
        print("  (A < B: PD hold 시 오버슈트로 height가 높았음)")
    print()
    print(f"    target_height (cfg)  : {TARGET_HEIGHT:.4f} m")
    print(f"    방식 A − target      : {diff_a_target:+.4f} m")
    print(f"    방식 B − target      : {diff_b_target:+.4f} m")
    print()

    # 해석
    ref = ha_mean  # 방식 A를 정답 기준으로 사용
    if ref - TARGET_HEIGHT > 0.01:
        print(f"  RESULT: 실제 default-pose height ({ref:.4f} m) > target ({TARGET_HEIGHT:.3f} m)")
        print(f"    r_height 수식에서 root_h > target → h_err ≤ 0 → r_height = 1 (만점).")
        print(f"    target_height 자체는 앉음을 유인하지 않음.")
        print(f"    앉음 원인은 다른 reward 항(r_pose/r_vel/stand_cos_threshold) 확인 필요.")
    elif ref - TARGET_HEIGHT < -0.01:
        print(f"  RESULT: 실제 default-pose height ({ref:.4f} m) < target ({TARGET_HEIGHT:.3f} m)")
        print(f"    완전히 서도 target 미달 → r_height < 1 (달성 불가).")
        print(f"    권고: target_height를 {ref:.3f} m 이하로 낮출 것.")
    else:
        print(f"  RESULT: 실제 height ({ref:.4f} m) ≈ target ({TARGET_HEIGHT:.3f} m) (±0.01 m 이내).")
        print(f"    target_height 설정 적절. height 단독으로는 앉음 유인 없음.")

    print("=" * 65)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]  # hydra_task_config injects env_cfg/agent_cfg at runtime
    simulation_app.close()
