# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg 시스템 식별(sysid) 모드 설정 — PACE CMA-ES용.

live 모드(`Isaac-R2S-BipedLeg-v0`)와 **같은 8-DOF 2족 asset / 관절 순서를 공유**하되,
CMA-ES 배치 적합에 맞게 다음만 바꾼다:

* ``num_envs`` = CMA-ES population (후보 파라미터 1개 = 환경 1개)
* 200 Hz 물리 = 200 Hz 제어 (``dt=1/200``, ``decimation=1``) — 학습 플랜트의 물리 그리드와 동일
* :class:`~pace_sim2real.utils.PaceDCMotorCfg` — 엔코더 바이어스 + 토크 지연 버퍼를 얹은 DC 모터
* slew rate limiter 우회 (chirp 고주파를 왜곡하므로) + action을 관절 목표각으로 직접 사용
* ``fix_base=True`` — 공중 고정. PACE의 ``fix_root_link``와 동일한 전제 (접촉력이 식별을 오염시킨다)

UDP/GUI는 쓰지 않는다. 적합은 오프라인 배치이며 입력은 ``data/<robot_name>/*.pt``다.

⚠⚠ **모델 갭 (중요, 조용히 넘어가지 말 것)**
    ``pace_sim2real``이 제공하는 액추에이터는 :class:`PaceDCMotor`(DCMotor 상속) **하나뿐**이다.
    반면 :data:`HIND_LEG_CFG`와 이 로봇의 RL 환경(``direct/hind_leg``)은 **ImplicitActuator**를 쓴다.
    즉 여기서 식별되는 33개 파라미터는 *토크–속도 포화 곡선이 있는 DC 모터 플랜트* 기준값이다.
    ImplicitActuator에는 그 포화 곡선이 없으므로, 식별 결과를 RL 환경에 그대로 옮기면
    포화 영역(고속·고토크 구간)에서 체계적으로 어긋난다. 옮기려면 둘 중 하나를 선택해야 한다:
      1. ``direct/hind_leg`` 환경도 DCMotor 계열로 바꾼다 (권장 — 플랜트가 일치).
      2. 갭을 감수하고 armature/마찰/바이어스만 이식한다 (포화 미고려를 명시적으로 기록할 것).

실행::

    python scripts/real2sim/fit_bipedleg.py --headless --num_envs 4096 \\
        --task Isaac-R2S-BipedLeg-Sysid-v0
"""

from __future__ import annotations

import torch
from pace_sim2real import PaceCfg
from pace_sim2real.utils import PaceDCMotorCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from .r2s_biped_leg_env_cfg import JOINT_NAME_PATTERNS, NUM_JOINTS, R2SBipedLegEnvCfg

from isaaclab_assets.robots.rga import HIND_LEG_CFG  # isort: skip

# ---------------------------------------------------------------------------
# 제어율 / 게인
# ---------------------------------------------------------------------------

# 2026-08-12: 500→200 Hz. 식별 파라미터가 이식될 **학습 플랜트의 물리 적분 그리드(200 Hz,
# decimation 4 = 50 Hz 제어)와 동일한 그리드**에서 식별해야 적분기 의존 편향이 전이되지 않는다
# (참고: upstream PACE 예제는 400 Hz — 500은 구 sim 수집기 발행률과 맞춘 포팅 선택이었다).
# 실기 GUI 캡처(50 Hz 발행)는 convert_gui_chirp_bipedleg.py가 ZOH로 이 그리드에 얹는다
# (명령 1개 = 4 물리스텝 유지 — 브리지가 마지막 ACT를 유지하는 것과 동일).
# delay 파라미터 해상도는 1 step = 5 ms가 된다 (max_delay=10 → 0~50 ms).
SYSID_RATE_HZ: float = 200.0

# ⚠ PACE는 PD 게인을 식별하지 않는다. 아래 값은 **chirp 수집에 사용한 kp/kd와 반드시 같아야 한다.**
# 다른 게인으로 수집했다면 여기도 같이 바꿔야 하며, 그러지 않으면 나머지 33개 파라미터가
# 게인 불일치를 흡수하려 들면서 물리적으로 무의미한 값으로 수렴한다.
#
# ⚠ go2(`r2s_go2_sysid_cfg.SYSID_KP/KD`)는 12관절 공통 **스칼라**였지만, 이 다리는 관절별로
# 게인이 다르므로 **길이 8의 관절별 리스트**다 (HIND_LEG_CFG legs 액추에이터 실측값과 동일).
# 데이터셋의 ``gain_scale``은 이 벡터 전체에 곱해지는 배율이다.
NOMINAL_KP: list[float] = [65.0, 53.0, 12.0, 20.0, 65.0, 53.0, 12.0, 20.0]
NOMINAL_KD: list[float] = [6.0, 4.8, 1.1, 1.0, 6.0, 4.8, 1.1, 1.0]

# ---------------------------------------------------------------------------
# PACE 액추에이터 (HIND_LEG_CFG legs 물성 계승 + 바이어스/지연)
# ---------------------------------------------------------------------------

# 관절별 실기 peak 토크 — 모터 12 N·m × 실제감속비 7/7/10.5/8.4 (RL_INTERFACE.md §6-c).
# 구값 28/28/42/56은 1/3 오인이었다. HIND_LEG_CFG(rga.py)와 값 일치 필수 (플랜트 페어링).
_EFFORT_LIMIT: dict[str, float] = {
    "HL_hip_joint": 84.0,
    "HL_thigh_joint": 84.0,
    "HL_calf_joint": 126.0,
    "HL_foot_joint": 100.8,
    "HR_hip_joint": 84.0,
    "HR_thigh_joint": 84.0,
    "HR_calf_joint": 126.0,
    "HR_foot_joint": 100.8,
}
# 실기 무부하 속도한계 — 모터 207.2 rad/s ÷ 실제감속비 (RL_INTERFACE.md §6-c). 구 foot 14.8 폐기.
_VELOCITY_LIMIT: dict[str, float] = {
    "HL_hip_joint": 29.6,
    "HL_thigh_joint": 29.6,
    "HL_calf_joint": 19.7,
    "HL_foot_joint": 24.6,
    "HR_hip_joint": 29.6,
    "HR_thigh_joint": 29.6,
    "HR_calf_joint": 19.7,
    "HR_foot_joint": 24.6,
}

# ⚠ ``saturation_effort``(토크–속도 곡선의 stall 토크)는 :class:`DCMotor`가 **스칼라로만** 받는다.
# ``effort_limit``/``velocity_limit``은 ``_parse_joint_parameter``로 관절별 텐서가 되지만
# ``saturation_effort``는 ``self.cfg.saturation_effort``를 그대로 쓰기 때문에(actuator_pd.py의
# ``_vel_at_effort_lim = velocity_limit * (1 + effort_limit / saturation_effort)``) dict를 주면
# 액추에이터 생성 시점에 TypeError로 죽는다.
#   → 관절별 τ_max(84/84/126/100.8) 중 **최댓값**을 쓴다. 이러면 어떤 관절도 자신의 effort_limit보다
#     낮게 인위적으로 잘리지 않는다(토크 상한은 관절별 ``effort_limit``이 그대로 강제한다).
#     대가로 hip/thigh/foot의 고속 구간 토크 감쇠가 실제보다 완만해진다.
#     관절별 곡선이 꼭 필요해지면 τ_max별로 액추에이터 그룹을 3개로 쪼개면 된다
#     (PACE의 ``update_simulator``/``apply_gains``는 ``articulation.actuators``를 순회하므로 다중 그룹 호환).
SATURATION_EFFORT_NM: float = 126.0

BIPEDLEG_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=list(JOINT_NAME_PATTERNS),  # 정규식이 아니라 정확한 관절명 8개 (CONTRACT §2)
    effort_limit=_EFFORT_LIMIT,
    saturation_effort=SATURATION_EFFORT_NM,
    velocity_limit=_VELOCITY_LIMIT,
    stiffness=dict(zip(JOINT_NAME_PATTERNS, NOMINAL_KP)),
    damping=dict(zip(JOINT_NAME_PATTERNS, NOMINAL_KD)),
    # 아래 5종은 CMA-ES가 매 세대 덮어쓴다 (여기 값은 초기치일 뿐).
    # armature 초기치 = 실측 ROTOR_I 7.4e-4 × 감속비² (RL_INTERFACE.md §6-a, rga.py와 동일).
    armature={
        ".*_hip_joint": 0.0363,
        ".*_thigh_joint": 0.0363,
        ".*_calf_joint": 0.0816,
        ".*_foot_joint": 0.0522,
    },
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    encoder_bias={".*": 0.0},
    max_delay=10,  # DelayBuffer 용량 [sim step]. 1 step = 5 ms @200 Hz.
)

# ---------------------------------------------------------------------------
# 합성 데이터용 GT (자기복원 게이트 전용 — 실기 데이터에는 쓰지 않는다)
# ---------------------------------------------------------------------------

# CMA-ES가 이 값들을 되찾아오는지로 파이프라인 정확성을 검증한다.
# 관절별로도 **좌우별로도** 전부 다른 값을 준다 — 그래야 "8개가 전부 같은 값"이라는 자명한 해도,
# 좌우 인덱스 스왑(HL↔HR 매핑 실수)도 게이트에서 잡힌다.
SYNTHETIC_GT_ARMATURE: list[float] = [0.010, 0.014, 0.018, 0.004, 0.012, 0.016, 0.020, 0.005]  # [kg·m²]
SYNTHETIC_GT_VISCOUS: list[float] = [0.10, 0.15, 0.20, 0.08, 0.12, 0.17, 0.22, 0.09]  # [N·m·s/rad]
SYNTHETIC_GT_COULOMB: list[float] = [0.15, 0.20, 0.25, 0.10, 0.17, 0.22, 0.27, 0.11]  # [N·m]
SYNTHETIC_GT_BIAS: list[float] = [0.02, -0.03, 0.04, -0.01, -0.02, 0.03, -0.04, 0.01]  # [rad]
SYNTHETIC_GT_DELAY: int = 4  # [sim step] = 20 ms @200 Hz


@configclass
class BipedLegPaceCfg(PaceCfg):
    """biped leg용 PACE 설정 — 데이터 경로, 관절 순서, CMA-ES 탐색 범위."""

    robot_name: str = "bipedleg_sim"  # 합성 데이터용. 실기 데이터는 "bipedleg_real"로 바꾼다.
    data_dir: str = "bipedleg_sim/chirp_data.pt"  # upstream fit.py(단일 데이터셋) 호환용
    joint_order: list[str] = list(JOINT_NAME_PATTERNS)  # CONTRACT §2 — leg-major(HL 4개 → HR 4개)

    # 적합용 데이터셋 (<repo>/data/ 기준). 여러 시퀀스를 동시에 맞출 수 있다 —
    # 논문도 진폭이 다른 여러 시퀀스를 쓰고, 단일 드라이브 단계에서는 게인 여러 종을 결합 적합한다.
    # 각 .pt는 녹화 당시의 kp/kd(= NOMINAL × gain_scale)를 담고 있으며 재생 시 그 게인이 복원된다.
    datasets: list[str] = [
        "bipedleg_sim/chirp_g060.pt",  # gain_scale=0.6, 진폭 100%
        "bipedleg_sim/chirp_g160.pt",  # gain_scale=1.6, 진폭  60% (게인이 높을수록 진폭을 줄여 토크 포화를 피한다 —
        #                                 포화 구간은 토크가 파라미터에 무감각해져 정보를 파괴한다)
    ]

    # hold-out 검증용 (적합에 절대 쓰지 않는다). 논문의 검증 방식 = **보지 않은 PD 게인**에서
    # 재현되는지 확인 (Tytan: ID at kp=60/kd=2 → validation at kp=145/kd=5).
    holdout: list[str] = ["bipedleg_sim/chirp_g100.pt"]  # gain_scale=1.0(미관측), 진폭 80%

    # 33 = armature(8) + viscous(8) + coulomb(8) + bias(8) + delay(1)
    bounds_params: torch.Tensor = torch.zeros((4 * NUM_JOINTS + 1, 2))

    def __post_init__(self):
        n = NUM_JOINTS
        # armature [kg·m²].
        #   ⚠ 로터 관성만 생각해 상한을 좁게 잡으면 안 된다. PACE 논문(§단일드라이브 vs 전신 대조)에서
        #   전신 식별 armature가 단일 드라이브 예측의 ~4배까지 나왔다(ANYmal LF-HFE 0.106 kg·m²).
        #   CAD 링크 관성 오차와 펌웨어 보상을 armature가 흡수하기 때문이다.
        #   bound가 구속조건이 되면 안 되므로 여유를 크게 둔다.
        self.bounds_params[0:n, 0] = 1e-5
        self.bounds_params[0:n, 1] = 0.5
        # viscous friction [N·m·s/rad].
        #   여기에는 실제 점성 마찰뿐 아니라 **kd 오차가 통째로 흡수**된다(둘 다 q̇에 곱해져 수학적으로
        #   구분 불가 — 논문의 게인 축퇴 논거와 동일). 이 다리 kd가 1.0~6.0이라 여유가 필요하다.
        #   참고: ANYmal 식별값 ~5 N·m·s/rad. go2 실기 1차 적합에서 상한 2.0에 전 관절 레일 포화가
        #   난 전례가 있어 처음부터 5.0으로 둔다. 5.0에 붙으면 더 올릴 것.
        self.bounds_params[n : 2 * n, 0] = 0.0
        self.bounds_params[n : 2 * n, 1] = 5.0
        # Coulomb friction [N·m] — Isaac ≥5.0에서 계수가 아니라 effort.
        #   상한을 go2(1.5)보다 높인다 — 이 다리 τ_max가 84~126 N·m로 go2 23.5보다 크다.
        self.bounds_params[2 * n : 3 * n, 0] = 0.0
        self.bounds_params[2 * n : 3 * n, 1] = 2.0
        # encoder bias [rad]
        self.bounds_params[3 * n : 4 * n, 0] = -0.1
        self.bounds_params[3 * n : 4 * n, 1] = 0.1
        # global delay [sim step] — 0~10 step = 0~50 ms @200 Hz. 전송지연을 포함한 총지연.
        #   논문 식별값은 Tytan/ANYmal D 모두 ≈7.5 ms(200 Hz에서 1.5 step). 실기 GUI 캡처는
        #   변환 단계에서 겉보기 지연(~18 ms)을 이미 보정하므로 잔여 지연은 0 근처가 정상이다.
        self.bounds_params[4 * n, 0] = 0.0
        self.bounds_params[4 * n, 1] = 10.0


@configclass
class R2SBipedLegSysidEnvCfg(R2SBipedLegEnvCfg):
    """CMA-ES 배치 적합용 biped leg 환경 (live 모드와 asset/관절순서 공유)."""

    sysid: bool = True
    fix_base: bool = True  # 공중 고정 — 접촉력이 들어오면 식별이 오염된다.
    # 2026-08-13: 커플링 재생 ON — foot 엔코더는 raw각(q_foot+q_calf)만 재므로(관절각은 실기 어디서도
    # 직접 측정 불가), 적합은 세 층을 전부 raw로 일관시킨다: ①데이터셋 foot=raw(convert
    # --keep_raw_foot) ②재생=raw 구동+전치(_pre_physics_step sysid 분기) ③채점=sim도 가상 엔코더
    # (q_f+q_c)로 환산해 비교(fit_bipedleg.py). ⚠collect_chirp_sim_bipedleg.py(sim 수집기)는 이
    # 규약 전환을 아직 모른다 — sim 수집 데이터를 다시 쓸 일이 생기면 foot 명령 의미부터 확인할 것.
    foot_coupling: bool = True

    decimation: int = 1
    sim: SimulationCfg = SimulationCfg(dt=1.0 / SYSID_RATE_HZ, render_interval=1)

    # population이 커도 씬이 과도하게 커지지 않도록 간격을 좁힌다 (PACE 기본값과 동일).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # live 모드와 같은 asset. actuator만 PACE 모델로 교체한다(그룹 키 "legs"는 HIND_LEG_CFG와 동일).
    robot: ArticulationCfg = HIND_LEG_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={"legs": BIPEDLEG_PACE_ACTUATOR_CFG},
    )

    sim2real: PaceCfg = BipedLegPaceCfg()
