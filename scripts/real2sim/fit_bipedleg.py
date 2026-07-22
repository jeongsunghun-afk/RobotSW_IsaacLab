# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg PACE 적합 — 여러 녹화 시퀀스를 동시에 맞춘다.

8-DOF 2족 다리(``HIND_LEG_CFG``)용. ``fit_go2.py``의 다중 궤적 CMA-ES 루프를 그대로 이식했고,
파라미터 수만 49개(12관절)에서 **33개**(armature 8 + viscous 8 + coulomb 8 + bias 8 + delay 1)로 바뀐다.

upstream ``scripts/pace/fit.py``는 궤적 하나만 재생한다. 이 스크립트는 ``BipedLegPaceCfg.datasets``의
모든 시퀀스를 한 세대 안에서 차례로 재생하고 점수를 합산해, **하나의 파라미터 집합이 모든 시퀀스를
동시에 설명**하도록 만든다. 시퀀스마다 **녹화 당시의 PD 게인을 복원**해서 재생한다.

PACE는 PD 게인을 식별하지 않는다 — 논문이 밝히듯 ``{armature, damping, kp, kd}``의 공통 스케일이
폐루프 거동을 보존하므로 최적해가 축퇴한다. 따라서 게인은 **알려진 입력**이고, 잘못된 게인으로
재생하면 그 불일치가 플랜트 파라미터를 편향시킨다. go2와 달리 이 다리는 게인이 스칼라가 아니라
**관절별 벡터**(``NOMINAL_KP``/``NOMINAL_KD``)이며, 데이터셋마다 배율(``gain_scale``)만 다르다.

hold-out 검증(``BipedLegPaceCfg.holdout``)은 적합에 쓰지 않는다. 논문의 전신 단계 검증도 **보지 않은
PD 게인**에서 재현되는지를 본다.

⚠ 모델 갭: PACE 액추에이터는 :class:`PaceDCMotor`(토크-속도 포화 곡선이 있는 DC 모터)다. 반면
``HIND_LEG_CFG``와 ``direct/hind_leg`` RL env는 :class:`ImplicitActuator`를 쓴다. 여기서 식별된
파라미터는 **DC 모터 플랜트 기준**이므로, RL env로 옮기려면 그 env도 DC 모터로 바꾸거나 갭을 감수해야 한다.

실행::

    python scripts/real2sim/fit_bipedleg.py --headless --num_envs 4096
    tensorboard --logdir logs/pace/bipedleg_sim
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="R2S-BipedLeg multi-trajectory PACE system identification.")
parser.add_argument("--num_envs", type=int, default=4096, help="CMA-ES population (one candidate per env).")
parser.add_argument("--task", type=str, default="Isaac-R2S-BipedLeg-Sysid-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from pace_sim2real.optim import MultiTrajectoryCMAES  # isort: skip
from pace_sim2real.utils import project_root, require_physx_backend  # isort: skip


def load_dataset(path, joint_order: list[str], device: str) -> dict:
    """녹화 시퀀스를 읽고 관절 순서 / 게인 존재를 검증한다.

    Args:
        path: ``chirp_*.pt`` 경로.
        joint_order: 기대하는 관절 순서 (8개, leg-major).
        device: 텐서를 올릴 디바이스.

    Returns:
        ``time``, ``dof_pos``, ``des_dof_pos``, ``kp``, ``kd``, ``name``을 담은 dict.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        RuntimeError: 저장된 관절 순서가 다르거나 kp/kd가 없을 때.
    """
    if not path.exists():
        raise FileNotFoundError(f"데이터셋이 없다: {path}")
    raw = torch.load(path)

    stored_order = raw.get("joint_order")
    if stored_order is not None and list(stored_order) != list(joint_order):
        raise RuntimeError(
            f"{path.name}의 관절 순서가 cfg의 joint_order와 다르다.\n  저장: {stored_order}\n  기대: {joint_order}"
        )

    num_joints = len(joint_order)
    if "kp" not in raw or "kd" not in raw:
        raise RuntimeError(
            f"{path.name}에 kp/kd가 없다. 녹화 당시의 게인을 모르면 재생이 성립하지 않는다. "
            "PACE는 게인을 식별하지 않으므로 게인은 알려진 입력이어야 한다 — "
            "collector를 최신 스키마로 다시 돌릴 것."
        )

    return {
        "name": path.name,
        "time": raw["time"].to(device),
        "dof_pos": raw["dof_pos"].to(device),
        "des_dof_pos": raw["des_dof_pos"].to(device),
        "kp": raw["kp"].to(device).reshape(num_joints),
        "kd": raw["kd"].to(device).reshape(num_joints),
    }


def gain_summary(kp: torch.Tensor, kd: torch.Tensor) -> str:
    """관절별 게인 벡터를 한 줄로 요약한다 (HL 다리 4개만, HR은 미러라 동일).

    Args:
        kp: 위치 게인 [N·m/rad], shape (8,).
        kd: 속도 게인 [N·m·s/rad], shape (8,).

    Returns:
        표에 넣을 요약 문자열.
    """
    kp_s = " ".join(f"{float(v):.1f}" for v in kp[:4])
    kd_s = " ".join(f"{float(v):.2f}" for v in kd[:4])
    return f"kp=[{kp_s}] kd=[{kd_s}]"


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Newton 백엔드는 static 마찰만 지원해 Coulomb/viscous 식별이 성립하지 않는다.
    require_physx_backend(env)

    device = env.unwrapped.device
    articulation = env.unwrapped.scene["robot"]
    sim2real = env_cfg.sim2real
    joint_order = sim2real.joint_order

    # warp 커널은 관절 인덱스를 int32로 요구한다. 텐서 인덱싱에는 long 버전을 쓴다.
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order], dtype=torch.int32, device=device
    )
    joint_ids_long = joint_ids.to(torch.long)

    data_root = project_root() / "data"
    datasets = [load_dataset(data_root / rel, joint_order, device) for rel in sim2real.datasets]
    if not datasets:
        raise RuntimeError("BipedLegPaceCfg.datasets가 비어 있다.")

    print("[INFO]: 파라미터 33개 = armature 8 + viscous 8 + coulomb 8 + bias 8 + delay 1")
    print(f"[INFO]: {len(datasets)}개 시퀀스로 적합한다:")
    for d in datasets:
        print(f"         {d['name']:28s} {d['dof_pos'].shape[0]:6d} steps  {gain_summary(d['kp'], d['kd'])}")
    if sim2real.holdout:
        print(f"[INFO]: hold-out (적합에 미사용): {sim2real.holdout}")

    log_dir = project_root() / "logs" / "pace" / sim2real.robot_name
    opt = MultiTrajectoryCMAES(
        bounds=sim2real.bounds_params.to(device),
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=sim2real.cmaes.max_iteration,
        # 상위 클래스는 config.pt 저장과 궤적 버퍼 크기 산정에 첫 시퀀스를 쓴다.
        data=datasets[0],
        device=device,
        epsilon=sim2real.cmaes.epsilon,
        sigma=sim2real.cmaes.sigma,
        save_interval=sim2real.cmaes.save_interval,
        save_optimization_process=sim2real.cmaes.save_optimization_process,
    )

    actions = torch.zeros(env.unwrapped.num_envs, articulation.num_joints, device=device)

    # 루프 전체를 inference_mode로 감싼다. env.step이 만든 버퍼는 inference tensor가 되므로,
    # 이후의 env.reset()/actuator write를 inference_mode 밖에서 하면 in-place 갱신이 거부된다
    # (upstream fit.py도 같은 이유로 전체를 감싼다).
    with torch.inference_mode():
        while simulation_app.is_running():
            for idx, dataset in enumerate(datasets):
                measured = dataset["dof_pos"]
                target = dataset["des_dof_pos"]
                num_steps = measured.shape[0]
                initial_pos = measured[0].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

                # 이 시퀀스가 녹화될 때의 게인을 복원한 뒤, 후보 파라미터를 다시 써 넣는다.
                # explicit actuator는 PD를 파이썬에서 계산하므로 진짜 게인은 actuator.stiffness /
                # actuator.damping 텐서다 — write_joint_stiffness_to_sim*은 여기서 no-op이다.
                env.reset()
                MultiTrajectoryCMAES.apply_gains(articulation, joint_ids, dataset["kp"], dataset["kd"])
                opt.update_simulator(articulation, joint_ids, initial_pos)
                opt.begin_trajectory(idx)

                for counter in range(num_steps):
                    opt.tell(
                        articulation.data.joint_pos[:, joint_ids_long],
                        measured[counter].unsqueeze(0).repeat(env.unwrapped.num_envs, 1),
                    )
                    actions[:, joint_ids_long] = target[counter].unsqueeze(0)
                    env.step(actions)

            opt.evolve()
            if opt.finished():
                break

    opt.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
