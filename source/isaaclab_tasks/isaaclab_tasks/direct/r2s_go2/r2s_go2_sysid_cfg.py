# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 시스템 식별(sysid) 모드 설정 — PACE CMA-ES용.

live 모드(`Isaac-R2S-Go2-v0`)와 **같은 GO2 asset / JOINT_ORDER / actuator 정의를 공유**하되,
CMA-ES 배치 적합에 맞게 다음만 바꾼다:

* ``num_envs`` = CMA-ES population (후보 파라미터 1개 = 환경 1개)
* 500 Hz 물리 = 500 Hz 제어 (``dt=1/500``, ``decimation=1``) — 실기 ``/lowcmd`` 발행률과 정합
* :class:`~pace_sim2real.utils.PaceDCMotorCfg` — 엔코더 바이어스 + 토크 지연 버퍼를 얹은 DC 모터
* slew rate limiter 우회 (chirp 고주파를 왜곡하므로) + action을 관절 목표각으로 직접 사용
* ``fix_base=True`` — 공중 고정. PACE의 ``fix_root_link``와 동일한 전제

UDP/ROS2는 쓰지 않는다. 적합은 오프라인 배치이며 입력은 ``data/<robot_name>/chirp_data.pt``다.

실행::

    python scripts/pace/fit.py --headless --num_envs 4096 --task Isaac-R2S-Go2-Sysid-v0
"""

from __future__ import annotations

import torch
from pace_sim2real import PaceCfg
from pace_sim2real.utils import PaceDCMotorCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .r2s_go2_env_cfg import JOINT_ORDER, NUM_JOINTS, PRONE_HEIGHT_M, PRONE_JOINT_POS, R2SGo2EnvCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# ---------------------------------------------------------------------------
# 제어율 / 게인
# ---------------------------------------------------------------------------

# 실기 /lowcmd 발행률과 sim 제어율을 일치시킨다. live 모드의 50 Hz는 RL 배포용이라
# 10 Hz chirp를 계단화해 위상을 왜곡한다 (CONTRACT §8의 rate seam).
SYSID_RATE_HZ: float = 500.0

# ⚠ PACE는 PD 게인을 식별하지 않는다. 아래 값은 **실기 chirp 수집에 사용한 kp/kd와 반드시 같아야 한다.**
# 다른 게인으로 수집했다면 여기도 같이 바꿔야 하며, 그러지 않으면 나머지 49개 파라미터가
# 게인 불일치를 흡수하려 들면서 물리적으로 무의미한 값으로 수렴한다.
SYSID_KP: float = 25.0
SYSID_KD: float = 0.5

# ---------------------------------------------------------------------------
# PACE 액추에이터 (UNITREE_GO2_CFG base_legs 물성 계승 + 바이어스/지연)
# ---------------------------------------------------------------------------

GO2_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=23.5,
    saturation_effort=23.5,
    velocity_limit=30.0,
    stiffness={".*": SYSID_KP},
    damping={".*": SYSID_KD},
    # 아래 5종은 CMA-ES가 매 세대 덮어쓴다 (여기 값은 초기치일 뿐).
    armature={".*": 0.01},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    encoder_bias={".*": 0.0},
    max_delay=10,  # DelayBuffer 용량 [sim step]. 1 step = 2 ms @500 Hz.
)

# ---------------------------------------------------------------------------
# 합성 데이터용 GT (Phase 1 자기복원 테스트 전용 — 실기 데이터에는 쓰지 않는다)
# ---------------------------------------------------------------------------

# CMA-ES가 이 값들을 되찾아오는지로 파이프라인 정확성을 검증한다.
# 관절 타입별로 값을 다르게 줘서 "12개가 전부 같은 값"이라는 자명한 해로 수렴하는 걸 막는다.
SYNTHETIC_GT_ARMATURE: list[float] = [0.010, 0.014, 0.018] * 4  # [kg·m²]
SYNTHETIC_GT_VISCOUS: list[float] = [0.10, 0.15, 0.20] * 4  # [N·m·s/rad]
SYNTHETIC_GT_COULOMB: list[float] = [0.15, 0.20, 0.25] * 4  # [N·m]
SYNTHETIC_GT_BIAS: list[float] = [0.02, -0.03, 0.04] * 4  # [rad]
SYNTHETIC_GT_DELAY: int = 4  # [sim step] = 8 ms @500 Hz


@configclass
class Go2PaceCfg(PaceCfg):
    """GO2용 PACE 설정 — 데이터 경로, 관절 순서, CMA-ES 탐색 범위."""

    robot_name: str = "go2_real"  # 로그 디렉터리 이름도 겸한다 (logs/pace/<robot_name>/)
    data_dir: str = "go2_sim/chirp_data.pt"  # upstream fit.py(단일 데이터셋) 호환용
    joint_order: list[str] = JOINT_ORDER  # CONTRACT §2 — 실기 LowState 인덱스 0..11과 동일

    # 적합용 데이터셋 (<repo>/data/ 기준). 여러 시퀀스를 동시에 맞출 수 있다 —
    # 논문도 진폭이 다른 여러 시퀀스를 쓰고, 단일 드라이브 단계에서는 게인 3종을 결합 적합한다.
    # 각 .pt는 녹화 당시의 kp/kd를 담고 있으며 재생 시 그 게인이 actuator에 복원된다.
    #
    # ★ 2026-08-04 재수집분을 쓴다 (`go2_real/`). 2026-07-15 캡처는 시간축이 104 ms 어긋나
    # 폐기했고, 그걸 사후 보정한 `go2_real_aligned/` 도 쓰지 않는다 — 보정 시 **진짜 구동 지연
    # (≈8 ms)까지 함께 빼버려** 적합기 delay 가 1 ms 로 수렴했기 때문이다.
    # 재수집분은 수집 시점에 정렬 6~8 ms(3~4 step), 유실 2~5% 로 확인됐다. 이 정도 지연은
    # 아래 `delay` 상한 10 step 안이므로 **보정하지 않고 적합기가 식별하게 둔다.**
    # 근거·측정: `reports/real2sim/_comparisons/pace_go2_sysid_excitation_audit/`.
    #
    # 배분: **배포 게인(kp=25)을 반드시 적합에 포함**한다. 학습·배포가 25/0.5 이므로 그 조건이
    # 적합에서 빠지면 안 된다. kp=15 는 게인이 낮아 플랜트가 가장 잘 보이는 데이터셋이다
    # (논문: "we therefore use small gains"). kp=35 를 hold-out 으로 남긴다.
    datasets: list[str] = [
        "go2_real/chirp_kp15.pt",  # kp=15 kd=0.5 — 플랜트가 가장 잘 보인다
        "go2_real/chirp_kp25.pt",  # kp=25 kd=0.5 — **배포 게인**
    ]

    # hold-out 검증용 (적합에 절대 쓰지 않는다). 논문의 전신 단계 검증 방식 = **보지 않은 PD 게인**
    # 및 보지 않은 궤적에서 재현되는지 확인 (Tytan: ID at kp=60/kd=2 → validation at kp=145/kd=5).
    # ⚠ 이 hold-out 은 **게인 일반화만** 검증한다. 세 캡처가 같은 세션·같은 대역·같은 리그를
    # 공유하므로, **수집 자체의 결함은 원리적으로 못 잡는다.** 실제로 104 ms 어긋남을 이
    # hold-out 은 통과시켰다(RMSE 0.017). 물리적 타당성은 별도 검사할 것
    # (에너지 수지 / 독립 회귀 / 링크 관성 / 가시성 — 위 reports 디렉터리의 logs/).
    #
    # ⚠ **리그 오염이 남아 있다.** 매단 base 의 pitch 각가속도가 관절 토크의 13~15%(중앙값,
    # 최대 34%)를 만든다. `--pattern pitch_cancel` 로 21.5% → 6.7%(저진폭 스윕 기준)까지
    # 줄였지만 0 이 아니다. base 고정이 불가능한 리그의 한계이며, 이 몫은 모델에 항이 없어
    # armature/viscous/coulomb 가 흡수한다. 결과 해석 시 이 편향을 명시할 것.
    holdout: list[str] = ["go2_real/chirp_kp35.pt"]  # kp=35 kd=0.5
    # 49 = armature(12) + viscous(12) + coulomb(12) + bias(12) + delay(1)
    bounds_params: torch.Tensor = torch.zeros((4 * NUM_JOINTS + 1, 2))

    def __post_init__(self):
        n = NUM_JOINTS
        # armature [kg·m²].
        #   ⚠ 로터 관성만 생각해 상한을 좁게 잡으면 안 된다. PACE 논문(§단일드라이브 vs 전신 대조)에서
        #   전신 식별 armature가 단일 드라이브 예측의 ~4배까지 나왔다(ANYmal LF-HFE 0.106 kg·m²).
        #   CAD 링크 관성 오차와 펌웨어 보상을 armature가 흡수하기 때문이다.
        #   GO2는 ANYmal보다 훨씬 가볍지만, bound가 구속조건이 되면 안 되므로 여유를 크게 둔다.
        self.bounds_params[0:n, 0] = 1e-5
        self.bounds_params[0:n, 1] = 0.5
        # viscous friction [N·m·s/rad].
        #   여기에는 실제 점성 마찰뿐 아니라 **kd 오차가 통째로 흡수**된다(둘 다 q̇에 곱해져 수학적으로
        #   구분 불가 — 논문의 게인 축퇴 논거와 동일). GO2 kd=0.5 자체가 이 항과 같은 크기라 여유가 필요하다.
        #   참고: ANYmal 식별값 ~5 N·m·s/rad.
        #   ⚠ 실기 kp15/25/35 1차 적합(26_07_15)에서 12관절 viscous가 전부 상한 2.0에 레일 포화 →
        #   진짜 최적점이 박스 밖. 상한을 5.0으로 올려 재적합한다. 재차 5.0에 붙으면 더 올릴 것.
        self.bounds_params[n : 2 * n, 0] = 0.0
        self.bounds_params[n : 2 * n, 1] = 5.0
        # Coulomb friction [N·m] — Isaac ≥5.0에서 계수가 아니라 effort. GO2 기어비 ~6.33 기어박스 마찰.
        self.bounds_params[2 * n : 3 * n, 0] = 0.0
        self.bounds_params[2 * n : 3 * n, 1] = 1.5
        # encoder bias [rad]
        self.bounds_params[3 * n : 4 * n, 0] = -0.1
        self.bounds_params[3 * n : 4 * n, 1] = 0.1
        # global delay [sim step] — 0~10 step = 0~20 ms @500 Hz. DDS 전송지연을 포함한 총지연.
        #   논문 식별값은 Tytan/ANYmal D 모두 ≈7.5 ms → 500 Hz에서 3.75 step으로 이 범위의 한가운데다.
        self.bounds_params[4 * n, 0] = 0.0
        self.bounds_params[4 * n, 1] = 10.0


@configclass
class R2SGo2SysidEnvCfg(R2SGo2EnvCfg):
    """CMA-ES 배치 적합용 GO2 환경 (live 모드와 asset/관절순서 공유)."""

    sysid: bool = True
    fix_base: bool = True  # 공중 고정 — 접촉력이 들어오면 식별이 오염된다.

    # ⚠ **반드시 False.** live 모드는 식별된 PACE 물성을 sim 에 써넣지만(기본 True), 여기는 그
    # 물성을 **찾는** 쪽이다. 켜두면 CMA-ES 가 이미 PACE 값이 들어간 플랜트 위에서 적합을 시작해
    # armature/마찰이 이중 적용되고 식별 결과가 조용히 망가진다.
    use_pace_params: bool = False

    decimation: int = 1
    sim: SimulationCfg = SimulationCfg(dt=1.0 / SYSID_RATE_HZ, render_interval=1)

    # population이 커도 씬이 과도하게 커지지 않도록 간격을 좁힌다 (PACE 기본값과 동일).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # live 모드와 같은 GO2 asset. actuator만 PACE 모델로 교체한다.
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=UNITREE_GO2_CFG.init_state.replace(pos=(0.0, 0.0, PRONE_HEIGHT_M), joint_pos=PRONE_JOINT_POS),
        actuators={"base_legs": GO2_PACE_ACTUATOR_CFG},
    )

    sim2real: PaceCfg = Go2PaceCfg()
