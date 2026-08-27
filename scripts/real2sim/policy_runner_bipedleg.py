# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg policy_runner — 학습된 hind_leg history 정책을 UDP 브릿지로 구동.

Isaac Sim(SimulationApp)을 띄우지 않는다 — 순수 torch + rsl_rl 로 정책만 로드/추론한다
(mock env 로 OnPolicyRunnerParkour 초기화; verify_policy_load.py 로 검증됨). conda isaac-6.0 python.

데이터 흐름 (POLICY_MODE_SPEC.md):

    gui ──POLICY_CMD(9884)──▶ policy_runner ──POLICY_ACT(9886)──▶ sim_runner_bipedleg
                                            ◀──POLICY_STATE(9885)──┘
    policy_runner ──REAL_ACT(9887)──▶ real ──REAL_STATE(9888)──▶ policy_runner  (seam)

50Hz 루프:
    1. gui POLICY_CMD drain → mode / input_source / x_vel / yaw
    2. input_source(sim/real) state drain → q, dq, gravity  (articulation 순서)
    3. mode=run & state 有 → obs 구성(bit-parity) → estimator priv_explicit → act_inference
       → target = action_scale·action + default → **sim + real 양쪽 fan-out**
    4. clock 진행 / prev_action 갱신

joint 순서는 **articulation 순서**로 통일(재매핑 없음). sim_runner get_policy_state 와 정책이 동일 순서.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/real2sim/policy_runner_bipedleg.py \
        --run_dir logs/rsl_rl/hindLeg_history_direct/2026-08-26_13-41-55_pace0819sym \
        --checkpoint model_50000.pt
    # real 엔드포인트가 있으면:  --real_host 192.168.x.y --real_port 9887
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import socket
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_biped_leg"))
from r2s_udp import (  # isort: skip
    POLICY_ACT_PORT,
    POLICY_CMD_PORT,
    POLICY_STATE_PORT,
    REAL_ACT_PORT,
    REAL_STATE_PORT,
    pack_policy_act,
    unpack_policy_cmd,
    unpack_policy_state,
)

# ---------------------------------------------------------------------------
# 학습 하이퍼파라미터 (hind_leg HindLegHistoryEnvCfg 와 bit-parity) — 정책 obs/target 계약
# ---------------------------------------------------------------------------
NUM_JOINTS = 8
POLICY_DIM = 3 + 3 + NUM_JOINTS * 3 + 4  # gravity + cmd + (jpos,jvel,action) + clock = 34
HISTORY_LEN = 10
ACTION_SCALE = 0.25
GAIT_PERIOD = 0.6  # s
STEP_DT = 0.02  # s (decimation 4 / 200 Hz) → 50 Hz control
CONTROL_HZ = 1.0 / STEP_DT
# default_joint_pos: hind_leg USD 전부 0 (CONTRACT.md §2, 같은 USD). obs=joint_pos-default, target=scale·a+default.
DEFAULT_JOINT_POS = [0.0] * NUM_JOINTS
# 명령 학습 범위 — 클램프용 (x_vel∈[-0.5,2.0], yaw∈[-0.5,0.5], y_vel≡0 미학습).
X_VEL_RANGE = (-0.5, 2.0)
YAW_RANGE = (-0.5, 0.5)

HOST = "127.0.0.1"
# 이 러너가 받아들이는 좌표 규약 (r2s_udp.R2S_CONVENTION_VERSION). 정책 obs/action 은 **관절 좌표**
# 이고 raw↔관절 변환은 브리지(real_runner)가 전담하므로, 관절 좌표를 신고하는 버전만 받는다.
_REQUIRED_CONVENTION_VERSION: int = 1


# ---------------------------------------------------------------------------
# 정책 로드 (mock env; Isaac 앱 없음) — verify_policy_load.py 로 검증된 경로
# ---------------------------------------------------------------------------


def _load_policy(run_dir: str, checkpoint: str, device: str):
    """OnPolicyRunnerParkour 를 mock env 로 초기화·로드하고 (policy, estimator) 를 반환."""
    from rsl_rl.algorithms.ppo_parkour import PPOParkour
    from rsl_rl.runners import OnPolicyRunnerParkour
    from tensordict import TensorDict

    ckpt_path = os.path.join(run_dir, checkpoint)
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    with open(agent_yaml) as f:
        train_cfg = yaml.safe_load(f)
    train_cfg["algorithm"].setdefault("rnd_cfg", None)
    train_cfg["algorithm"].setdefault("symmetry_cfg", None)
    _accepted = set(inspect.signature(PPOParkour.__init__).parameters)
    train_cfg["algorithm"] = {k: v for k, v in train_cfg["algorithm"].items() if k in _accepted or k == "class_name"}

    # priv_latent 차원은 ckpt priv_encoder 첫 레이어에서 역산 (mock obs 사이징용).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    priv_latent_dim = None
    for k, v in ckpt["model_state_dict"].items():
        if "priv_encoder" in k and k.endswith("0.weight") and v.dim() == 2:
            priv_latent_dim = int(v.shape[1])
            break
    if priv_latent_dim is None:
        raise RuntimeError("priv_encoder 첫 레이어를 찾지 못했습니다 — 체크포인트 구조 확인 필요.")

    def make_obs():
        return TensorDict(
            {
                "policy": torch.zeros(1, POLICY_DIM, device=device),
                "priv_explicit": torch.zeros(1, 6, device=device),
                "priv_latent": torch.zeros(1, priv_latent_dim, device=device),
                "history": torch.zeros(1, HISTORY_LEN, POLICY_DIM, device=device),
            },
            batch_size=[1],
        )

    class _MockRobotData:
        joint_names = [f"j{i}" for i in range(NUM_JOINTS)]

    class _MockRobot:
        data = _MockRobotData()

    class _MockEnv:
        def __init__(self):
            self.num_envs = 1
            self.num_actions = NUM_JOINTS
            self.device = device
            self.cfg = type("Cfg", (), {})()
            self._robot = _MockRobot()
            self.max_episode_length = 1000

        @property
        def unwrapped(self):
            return self

        def get_observations(self):
            return make_obs()

    runner = OnPolicyRunnerParkour(_MockEnv(), train_cfg, log_dir=None, device=device)
    runner.load(ckpt_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=device)
    estimator = runner.alg.estimator
    print(f"[policy_runner] loaded {ckpt_path}  (priv_latent={priv_latent_dim})", flush=True)
    return policy, estimator


# ---------------------------------------------------------------------------
# obs 구성 (hind_leg_env._get_observations 와 bit-parity, articulation 순서)
# ---------------------------------------------------------------------------


class PolicyState:
    """clock / history / prev_action 상태를 관리하며 매 step obs dict를 만든다."""

    def __init__(self, device: str):
        self.device = device
        self.default = torch.tensor(DEFAULT_JOINT_POS, device=device)
        self.reset()

    def reset(self):
        self.phase = 0.0  # gait clock ∈ [0,1)
        self.prev_action = torch.zeros(NUM_JOINTS, device=self.device)
        self.history = None  # 첫 obs에서 10× 복제로 초기화

    def _clock_obs(self) -> torch.Tensor:
        phi_hl = self.phase
        phi_hr = (self.phase + 0.5) % 1.0
        two_pi = 2.0 * math.pi
        return torch.tensor(
            [
                math.sin(two_pi * phi_hl),
                math.cos(two_pi * phi_hl),
                math.sin(two_pi * phi_hr),
                math.cos(two_pi * phi_hr),
            ],
            device=self.device,
        )

    def build_obs(self, q: torch.Tensor, dq: torch.Tensor, gravity: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        """policy obs(34) 구성. q/dq/gravity/prev_action 모두 articulation 순서.

        순서(학습 _get_observations): gravity(3), cmd(3), q-default(8), dq(8), prev_action(8), clock(4).
        """
        obs = torch.cat([gravity, cmd, q - self.default, dq, self.prev_action, self._clock_obs()], dim=0)
        return obs  # (34,)

    def push_history(self, obs: torch.Tensor) -> torch.Tensor:
        """history 버퍼 갱신 후 반환. 첫 스텝은 10× 복제(학습 episode_length_buf<=1 분기), 이후 roll."""
        if self.history is None:
            self.history = obs.unsqueeze(0).repeat(HISTORY_LEN, 1)  # (10, 34)
        else:
            self.history = torch.cat([self.history[1:], obs.unsqueeze(0)], dim=0)
        return self.history

    def advance(self, raw_action: torch.Tensor):
        """clock 진행 + prev_action 갱신 (obs 구성 후 호출)."""
        self.phase = (self.phase + STEP_DT / GAIT_PERIOD) % 1.0
        self.prev_action = raw_action.clone()


def main() -> None:
    parser = argparse.ArgumentParser(description="R2S-BipedLeg policy runner (UDP bridge, no Isaac app).")
    parser.add_argument("--run_dir", required=True, help="학습 run 디렉토리 (params/agent.yaml 포함).")
    parser.add_argument("--checkpoint", default="model_23100.pt", help="체크포인트 파일명.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--real_host", default=None, help="real 엔드포인트 IP (미지정 시 real fan-out no-op).")
    parser.add_argument("--real_act_port", type=int, default=REAL_ACT_PORT, help="real action 수신 포트.")
    args = parser.parse_args()

    device = args.device
    policy, estimator = _load_policy(args.run_dir, args.checkpoint, device)
    ps = PolicyState(device)

    # --- 소켓 ---
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # gui POLICY_CMD 수신
    cmd_sock.bind((HOST, POLICY_CMD_PORT))
    cmd_sock.setblocking(False)
    sim_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # sim rich state 수신
    sim_state_sock.bind((HOST, POLICY_STATE_PORT))
    sim_state_sock.setblocking(False)
    real_state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # real state 수신 (seam)
    real_state_sock.bind((HOST, REAL_STATE_PORT))
    real_state_sock.setblocking(False)
    act_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # action 송신 (sim + real)
    sim_act_addr = (HOST, POLICY_ACT_PORT)
    real_act_addr = (args.real_host, args.real_act_port) if args.real_host else None

    print(
        f"[policy_runner] cmd:{POLICY_CMD_PORT}  sim_state:{POLICY_STATE_PORT}  real_state:{REAL_STATE_PORT}  "
        f"→ act sim:{POLICY_ACT_PORT}" + (f" real:{real_act_addr}" if real_act_addr else "  (real: no-op)"),
        flush=True,
    )
    print(
        "[policy_runner] cmd 범위: x_vel∈[-0.5,2.0], yaw∈[-0.5,0.5] (y_vel≡0). yaw 추종은 약함(학습 0.145).", flush=True
    )

    mode = 0  # 0=idle, 1=run
    source = 0  # 0=sim, 1=real
    no_state = 0  # 연속 state 무응답 횟수 (구버전 STATE 거부를 조용히 넘기지 않기 위한 카운터)
    x_vel = 0.0
    yaw = 0.0
    seq = 0

    def drain_latest(sock, unpack):
        latest = None
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except (BlockingIOError, OSError):
                break
            parsed = unpack(data)
            if parsed is not None:
                latest = parsed
        return latest

    def recv_state_blocking(sock, timeout=0.2):
        """선택된 source 소켓에서 state 하나를 blocking 대기 후 latest-wins drain. 없으면 None."""
        sock.settimeout(timeout)
        try:
            data, _ = sock.recvfrom(4096)
        except (TimeoutError, OSError):
            return None
        latest = unpack_policy_state(data)
        sock.setblocking(False)
        while True:
            try:
                d, _ = sock.recvfrom(4096)
            except (BlockingIOError, OSError):
                break
            p = unpack_policy_state(d)
            if p is not None:
                latest = p
        return latest

    def send_action(target) -> None:
        packet = pack_policy_act(seq, target)
        act_sock.sendto(packet, sim_act_addr)  # sim
        if real_act_addr is not None:
            act_sock.sendto(packet, real_act_addr)  # real (seam)

    # 요청-응답 lockstep: policy_runner가 action을 보내면 sim_runner가 정확히 1 step 후 state를 회신하고,
    # policy_runner는 그 state로 다음 action을 계산한다 → policy 1 step = sim 1 step (학습 불변식 유지).
    # 50Hz 자유 타이머가 아니라 sim 응답에 맞춰 진행하므로 "state 대기" 교란/clock 비동기가 사라진다.
    try:
        while True:
            # 1) gui 명령 (비동기)
            cmd = drain_latest(cmd_sock, unpack_policy_cmd)
            if cmd is not None:
                prev_mode = mode
                mode, source = cmd["mode"], cmd["source"]
                x_vel = max(X_VEL_RANGE[0], min(X_VEL_RANGE[1], cmd["x_vel"]))
                yaw = max(YAW_RANGE[0], min(YAW_RANGE[1], cmd["yaw"]))
                if prev_mode == 0 and mode == 1:
                    ps.reset()  # run 진입 시 clock/history/prev_action 초기화
                    send_action(ps.default.cpu().numpy())  # 시동: sim_runner 회신 주소 학습 + 첫 step
                    seq += 1

            if mode == 0:
                time.sleep(0.01)  # idle: action 미송신 → sim은 step하지 않고 upright 유지
                continue

            # 2) 방금 보낸 action에 대한 sim/real state를 대기 (요청-응답)
            src_sock = real_state_sock if source == 1 else sim_state_sock
            state = recv_state_blocking(src_sock)
            if state is None:
                # ⚠ 구버전(84 B) STATE 도 여기로 온다 — unpack이 크기 불일치로 None을 주기 때문.
                #   그대로 두면 조용한 무한 재시동 루프가 되므로 일정 횟수마다 원인을 알린다.
                no_state += 1
                if no_state % 50 == 0:
                    print(
                        f"[policy_runner] ⚠ state 무응답 {no_state}회 — 송신자가 죽었거나, **구버전 84 B "
                        f"STATE**를 보내고 있다(규약 필드 없음 → 크기 불일치로 거부). "
                        f"src={'real' if source == 1 else 'sim'}",
                        flush=True,
                    )
                send_action(ps.default.cpu().numpy())  # 응답 없음 → 재시동
                seq += 1
                continue
            no_state = 0

            # 3) 좌표 규약 검사 — 불일치면 **즉시 종료**한다.
            #    변환은 브리지(real_runner)가 전담하고 여기서는 하지 않는다(2026-08-14 결정):
            #    정책 러너를 gui_controller에 통합할 계획이라, 여기에 변환을 두면 GUI 안에
            #    raw/관절 두 규약이 공존하게 된다. 그래서 "맞춰주기"가 아니라 "거부"가 맞다.
            ver = state.get("convention_version")
            if ver != _REQUIRED_CONVENTION_VERSION:
                raise SystemExit(
                    f"[policy_runner] 좌표 규약 불일치 — env 좌표 이관 전.\n"
                    f"  받은 convention_version={ver}, 필요={_REQUIRED_CONVENTION_VERSION} "
                    f"(source={'real' if source == 1 else 'sim'})\n"
                    f"  0 = gear 미적용 + foot raw각 / 1 = gear 적용 + foot 관절각.\n"
                    f"  정책은 관절 좌표만 안다 — 이 러너는 변환하지 않는다(변환은 real_runner 담당)."
                )

            # 3) 추론 → 다음 action (articulation 순서)
            q = torch.tensor(state["q"], device=device)
            dq = torch.tensor(state["dq"], device=device)
            gravity = torch.tensor(state["gravity"], device=device)
            cmd_vec = torch.tensor([x_vel, 0.0, yaw], device=device)  # y_vel≡0

            obs_policy = ps.build_obs(q, dq, gravity, cmd_vec)  # (34,)
            hist = ps.push_history(obs_policy)  # (10,34)
            with torch.inference_mode():
                obs_b = obs_policy.unsqueeze(0)  # (1,34)
                priv_explicit = estimator(obs_b)  # (1,6) — base vel 추정
                obs_dict = {"policy": obs_b, "priv_explicit": priv_explicit, "history": hist.unsqueeze(0)}
                raw_action = policy(obs_dict)[0]  # (8,) articulation 순서
            ps.advance(raw_action)
            target = (ACTION_SCALE * raw_action + ps.default).cpu().numpy()
            send_action(target)
            seq += 1
            # 1초(50 step)마다 자세 로깅 — grav_z≈-1이면 직립, 0/양수면 기울어짐/전도.
            if seq % 50 == 0:
                print(
                    f"[policy_runner] t={seq * STEP_DT:5.1f}s  grav_z={float(gravity[2]):+.2f}  "
                    f"phase={ps.phase:.2f}  x_vel={x_vel:+.2f} yaw={yaw:+.2f}  src={'real' if source == 1 else 'sim'}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[policy_runner] stopped.", flush=True)
    finally:
        for s in (cmd_sock, sim_state_sock, real_state_sock, act_sock):
            s.close()


if __name__ == "__main__":
    main()
