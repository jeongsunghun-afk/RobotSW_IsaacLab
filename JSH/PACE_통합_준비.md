# PACE sim2real 통합 준비 (HindLeg RL 파이프라인)

> 목적: ETH RSL의 **PACE**(Precise Adaptation through Continuous Evolution) 액추에이터 식별
> 프레임워크를 우리 8-DOF biped `HindLeg-Direct-v0` 파이프라인에 **꽂을 수 있게** 미리 배선
> 계획을 확정한다. **PACE는 실물 하드웨어의 chirp 데이터가 있어야 실제로 돌릴 수 있으므로,
> 지금은 "하드웨어 오면 바로 실행" 상태로 만드는 준비 문서**이다. (학습/Isaac 실행 안 함.)
>
> 조사 대상 리포지토리: `https://github.com/leggedrobotics/pace-sim2real`
> (Apache-2.0, ETH Zurich RSL, author Filip Bjelonic)
> 조사 시점 클론 위치(임시): `scratchpad/pace-sim2real` — **우리 repo에는 복제하지 않음**.

---

## 1. PACE 개요 — 무엇을 식별하고, 어떻게

### 1.1 한 줄 요약
실물 관절을 **chirp(주파수 스윕) 신호**로 가진 → 실측 관절궤적(q, q_des)을 기록 →
동일 명령을 시뮬에서 4096개 후보 파라미터로 병렬 재생 → **CMA-ES**로
"시뮬 궤적이 실측 궤적과 가장 잘 맞는" 물리 파라미터를 찾는다.

### 1.2 식별하는 파라미터 (관절당 4개 + 전역 1개)
`optim/cma_es.py`의 파라미터 벡터 레이아웃 (`CMAESOptimizer.__init__`, L58-63):

| 슬라이스 | 파라미터 | 물리 의미 | 배선 대상 API |
|---|---|---|---|
| `armature_idx = [0:N]` | **armature (반사관성)** | 로터관성×기어비² [kg·m²] | `write_joint_armature_to_sim` |
| `damping_idx = [N:2N]` | **viscous friction (점성마찰)** | 속도비례 마찰 [Nm·s/rad] | `write_joint_viscous_friction_coefficient_to_sim` |
| `friction_idx = [2N:3N]` | **coulomb/dry friction (건마찰)** | static=dynamic으로 넣어 쿨롱마찰 [Nm] | `write_joint_friction_coefficient_to_sim` (+dynamic) |
| `bias_idx = [3N:4N]` | **encoder bias (엔코더 오프셋)** | 관절 영점 오차 [rad] | `PaceDCMotor.update_encoder_bias` |
| `delay_idx = 4N` (스칼라) | **actuation delay (구동 지연)** | 명령→토크 지연 [sim step, 정수] | `PaceDCMotor.update_time_lags` |

- N = 관절 수. 우리 biped는 **N=8 → 파라미터 33개** (`4·8+1`).
- **주의**: `damping`(= viscous friction, 물리)과 PD 제어게인 `Kd`는 **별개**다.
  Kp/Kd(PD 게인)는 **식별하지 않는다**(모터스펙/튜닝값 고정). CMA-ES는 순수 물리 다이내믹스만 맞춘다.
- **토크-속도 곡선**은 PACE가 식별하지 않고, 부모 클래스 `DCMotor`가
  `saturation_effort`·`effort_limit`·`velocity_limit`로 모델링(모터스펙에서 넣음).
- **backlash(백래시)는 모델링 안 함**. 백래시 대체로 encoder bias + friction만.

### 1.3 핵심 클래스/모듈 (실제 경로)

| 파일 | 클래스/함수 | 역할 |
|---|---|---|
| `source/pace_sim2real/pace_sim2real/utils/pace_actuator.py` | `PaceDCMotor(DCMotor)` | 액추에이터 모델. `DCMotor`에 **encoder bias + 토크 지연버퍼(`DelayBuffer`)** 추가 |
| `.../utils/pace_actuator_cfg.py` | `PaceDCMotorCfg(DCMotorCfg)` | 위 cfg. 필드: `class_type`, `encoder_bias`, `max_delay` (+ `DCMotorCfg`의 `saturation_effort`/`stiffness`/`damping`/`friction`/`dynamic_friction`/`viscous_friction`/`effort_limit`/`velocity_limit`) |
| `.../optim/cma_es.py` | `CMAESOptimizer` | CMA-ES 루프. `cmaes` 패키지 사용. loss = 위치궤적 MSE |
| `scripts/pace/data_collection.py` | (main) | **chirp 생성 + 궤적 기록** → `data/<robot>/chirp_data.pt` |
| `scripts/pace/fit.py` | (main) | chirp 데이터 로드 → CMA-ES 식별 → `logs/pace/<robot>/<ts>/mean_XXX.pt` |
| `scripts/pace/plot_trajectory.py` | (main) | 결과 궤적/스코어 시각화 |
| `.../tasks/manager_based/pace/pace_sim2real_env_cfg.py` | `PaceSim2realEnvCfg`, `PaceCfg`, `CMAESOptimizerCfg`, `PaceSim2realSceneCfg` | 식별용 매니저기반 env (fix_root_link=True, 400Hz, decimation=1) |
| `.../tasks/manager_based/pace/anymal_pace_env_cfg.py` | `AnymalDPaceEnvCfg`, `ANYDRIVE_PACE_ACTUATOR_CFG`, `AnymalDPaceCfg` | **로봇별 예시** — 우리가 복제할 템플릿 |

### 1.4 PaceDCMotor 실제 시그니처 (인용)
```python
class PaceDCMotor(DCMotor):
    cfg: PaceDCMotorCfg
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)
        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, ...))
    def compute(self, control_action, joint_pos, joint_vel):
        # PD 컨트롤러가 "엔코더 프레임" 위치(joint_pos - encoder_bias)로 동작
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        return control_action_sim

@configclass
class PaceDCMotorCfg(DCMotorCfg):
    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: dict[str, float] | float | None = 0.0
    max_delay: torch.int | None = 0
```

### 1.5 CMA-ES loss / 설정 (인용)
```python
# cma_es.py  tell(): 시뮬 위치 - 실측 위치 - bias 의 제곱합 (시간 누적)
self.scores += torch.sum(torch.square(sim_dof_pos - real_dof_pos - self.sim_params[:, self.bias_idx]), dim=1)
```
- **loss = 관절 위치궤적 MSE** (bias 보정 후). 토크/속도가 아니라 **위치**를 맞춘다.
- 후보 하나 = env 하나. **population_size = num_envs**(예: 4096). 모든 후보가 같은 명령궤적을 병렬 재생.
- `CMAESOptimizerCfg`: `max_iteration=200`, `epsilon=1e-2`(조기종료), `sigma=0.5`, `save_interval=10`.
- 파라미터는 내부적으로 `[-1,1]` 정규화 후 `bounds`로 매핑(`_params_to_sim_params`).
- 출력: `logs/pace/<robot_name>/<YY_MM_DD_hh-mm-ss>/mean_XXX.pt`(베스트 평균 파라미터),
  `best_trajectory.pt`, `config.pt`, TensorBoard 로그.

### 1.6 chirp 데이터 워크플로 (`data_collection.py`)
- **선형 chirp**: `phase = 2π(f0·t + (f1-f0)/(2T)·t²)`, `signal = sin(phase)`.
  기본 `f0=0.1Hz → f1=10Hz`, `T=20s`, sample_rate = 물리 dt⁻¹(400Hz).
- 관절별 **bias/scale/direction** 다르게 적용(예: anymal `trajectory_scale=[0.25,0.5,-2.0]×4`,
  `trajectory_bias=[0,0.4,0.8]×4`) → HAA/HFE/KFE가 서로 다른 진폭·중심으로 스윙.
- **저장 포맷** `chirp_data.pt` = dict:
  ```python
  {"time": Tensor[T], "dof_pos": Tensor[T,N], "des_dof_pos": Tensor[T,N]}
  ```
  `dof_pos` = 측정 관절위치, `des_dof_pos` = 명령(목표) 관절위치.
- **중요 사실**: repo의 `data_collection.py`는 **시뮬 스크립트**다(Isaac 띄워서 chirp를
  가짜 GT 파라미터 armature=0.1/damping=4.5/friction=0.05/bias=0.05/delay=5로 시뮬에 걸고
  기록). 즉 **sim-to-sim 리허설이 이미 내장**되어 있다(§5). 실물에서는 이 스크립트 대신
  **실제 로봇에서 같은 chirp를 걸고 q·q_des를 같은 dict 포맷으로 로깅**하면 된다.

### 1.7 로봇별 배선 예시 (anymal — 우리 템플릿)
```python
# anymal_pace_env_cfg.py
ANYDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=140.0, effort_limit=89.0, velocity_limit=8.5,
    stiffness={".*": 85.0},  # Kp
    damping={".*": 0.6},     # Kd (PD, 식별 대상 아님)
    encoder_bias={".*": 0.0},
    friction={".*": 0.0}, dynamic_friction={".*": 0.0}, viscous_friction={".*": 0.0},
    max_delay=10,
)
class AnymalDPaceCfg(PaceCfg):
    robot_name = "anymal_d_sim"
    data_dir = "anymal_d_sim/chirp_data.pt"
    bounds_params = torch.zeros((49, 2))  # 12+12+12+12+1
    joint_order = ["LF_HAA","LF_HFE","LF_KFE", ...]  # 12개
    def __post_init__(self):
        self.bounds_params[:12,0]=1e-5; self.bounds_params[:12,1]=1.0   # armature [1e-5,1.0] kg·m²
        self.bounds_params[12:24,1]=7.0                                  # viscous  [0,7] Nm·s/rad
        self.bounds_params[24:36,1]=0.5                                  # friction [0,0.5] Nm
        self.bounds_params[36:48,0]=-0.1; self.bounds_params[36:48,1]=0.1 # bias [-0.1,0.1] rad
        self.bounds_params[48,1]=10.0                                     # delay [0,10] step
# scene: ANYMAL_D_CFG.replace(actuators={"legs": ANYDRIVE_PACE_ACTUATOR_CFG})
```

---

## 2. 현재 상태 대비

### 2.1 우리 현재 액추에이터 (`source/isaaclab_assets/isaaclab_assets/robots/rga.py` `HIND_LEG_CFG`, L470-496)
단일 `ImplicitActuatorCfg` 그룹 `"legs"`가 `.*_hip/.*_thigh/.*_calf/.*_foot` 4관절×2다리를 커버:
- `stiffness` = {hip 65, thigh 53, calf 12, foot 20} (관성 스케일 손튜닝)
- `damping` = {hip 6, thigh 4.8, calf 1.1, foot 1.0}
- `effort_limit_sim` = {hip 28, thigh 28, calf 42, foot 56}
- `velocity_limit_sim` = {hip 29.6, thigh 29.6, calf 19.7, foot 14.8}
- **armature / friction / viscous / delay / encoder bias 전부 없음(미식별)**.
- `ImplicitActuatorCfg` = PhysX 내장 PD. 토크-속도 곡선/지연/엔코더 오프셋 모델 없음.

### 2.2 파라미터 비교표

| 항목 | 현재 `ImplicitActuatorCfg` "legs" | PACE `PaceDCMotor` (식별 후) | 지금 우리에게 없는가 |
|---|---|---|---|
| PD Kp/Kd | 손튜닝 (hip65/6 …) | `stiffness`/`damping`(모터스펙, 식별 X) | 있음(값만 재검토) |
| effort limit | `effort_limit_sim` 있음 | `effort_limit` + `saturation_effort` | saturation_effort 없음 |
| velocity limit | `velocity_limit_sim` 있음 | `velocity_limit` (토크-속도 derating) | 곡선효과 없음 |
| **armature(반사관성)** | **없음** | 관절당 식별 | **없음 ← 최우선** |
| **viscous friction** | **없음** | 관절당 식별 | **없음** |
| **coulomb friction** | **없음** | 관절당 식별 | **없음** |
| **encoder bias** | **없음** | 관절당 식별 | **없음** |
| **actuation delay** | **없음(0)** | 전역 식별 [step] | **없음** |
| 액추에이터 타입 | Implicit(PhysX PD) | Explicit `DCMotor` 파생 | 타입 자체가 바뀜 |

> **참고**: 02_Leg 쿼드 트랙에서 이미 "armature(반사관성) 부재 = 폐루프 발산 근본원인"을
> 여러 번 확인했다(GEARBOX, CI-MPC 교훈). biped RL에서도 **armature 식별이 sim2real 1순위**.

### 2.3 우리 IsaacLab에 PACE 필수 API 존재 확인 (드롭인 가능성 ✅)
우리 repo(Isaac Sim 5.1) 코어에서 아래 전부 확인됨:
- `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py:42` `DCMotorCfg(IdealPDActuatorCfg)`, `saturation_effort`(L47).
- `actuator_base_cfg.py:156/163` `dynamic_friction` / `viscous_friction` 필드 존재.
- `articulation.py`: `write_joint_armature_to_sim`(L840), `write_joint_friction_coefficient_to_sim`(L871),
  `write_joint_viscous_friction_coefficient_to_sim`(L973), `default_joint_viscous_friction_coeff`(data L223).
- `isaaclab/utils/buffers/delay_buffer.py` `DelayBuffer` 존재.
> 결론: PACE는 **우리 IsaacLab에 코어 수정 없이 드롭인 가능**. (PACE는 Isaac Sim ≥5.0 요구,
> 우리는 5.1이라 viscous friction 지원 OK.)

---

## 3. 하드웨어 데이터 수집 프로토콜 (실물 도착 시 실행)

목표: 각 관절 클래스(hip / thigh / calf / foot)에 chirp를 걸어 `chirp_data.pt` 포맷으로 로깅.
우리 관절 8개: `HL/HR × {hip, thigh, calf, foot}`. 모터스펙(정격/피크 Nm): **hip 28/84, thigh 28/84(hip급),
calf(knee) 42/126, foot(ankle) 56/168**, 감속비 존재.

### 3.1 공통 셋업 체크리스트
- [ ] 로봇 **고정(base clamp/행잉)** — PACE 식별 env는 `fix_root_link=True`. 실물도 몸통 고정하고
      다리만 자유롭게(발이 지면/장애물에 안 닿게). 접촉 없는 자유공진이 식별에 유리.
- [ ] 제어주기 = 400Hz(=PACE `sim.dt=0.0025`, decimation=1)로 로깅. 다르면 §4에서 `sim.dt` 맞춤.
- [ ] 명령은 **위치제어(목표 관절각)**. PD 게인은 배포에 쓸 값과 동일하게 고정하고 그 값을 기록.
- [ ] 로깅 채널(관절별, 400Hz 동기):
      - `time` [s], `dof_pos` = **측정 관절각 q** [rad], `des_dof_pos` = **명령 관절각 q_des** [rad].
      - (참고용 추가 로깅 권장: 측정 토크 τ, 관절속도 q̇ — PACE loss엔 안 쓰지만 검증/디버그용.)
- [ ] `joint_order` 리스트를 **시뮬 관절 이름 순서와 정확히** 맞춰 저장.

### 3.2 chirp 파형 (관절 클래스별)
- 파형: 선형 chirp `sin(2π(f0·t + (f1-f0)/(2T)·t²))`, **T=20s** 권장(길수록 저주파 잘 잡음).
- **주파수 범위 f0→f1**: `0.1Hz → 10Hz` 기본. 관절 대역폭이 낮으면(무거운 다리) f1을 5~8Hz로 낮춤.
- **진폭(관절 가동범위·속도한계 내)** — 각 관절의 `velocity_limit`가 상한. 피크속도 ≈ 2π·f1·A 이므로
  고주파에서 속도포화 안 나게 진폭 A를 제한. 권장 시작값(중심각 ± A):

| 관절 클래스 | vel_limit [rad/s] | 중심각(bias) | 진폭 A [rad] | f0→f1 | 안전 메모 |
|---|---|---|---|---|---|
| hip | 29.6 | 0(중립) | 0.25 | 0.1→10 | 좌우 충돌범위 확인 |
| thigh | 29.6 | -0.26/+0.26 | 0.4 | 0.1→10 | 무릎 self-collision 주의 |
| calf(knee) | 19.7 | ±0.87 | 0.5 | 0.1→8 | 관절 하드스톱 여유 확보 |
| foot(ankle) | 14.8 | ±0.87 | 0.35 | 0.1→8 | 가장 낮은 대역폭, f1 보수적 |

  > anymal 참고값: `trajectory_scale=[0.25,0.5,-2.0]`, `bias=[0,0.4,0.8]` (HAA/HFE/KFE). 우리는
  > 위 표를 시작점으로, 실물에서 **토크 피크가 정격(hip28/knee42/ankle56 Nm)을 넘지 않는지** 모니터.

### 3.3 안전 메모 (물리 한계 준수 — 비물리 클리핑 금지 방침)
- [ ] **토크 한계**: 실측 τ가 피크토크(hip84/knee126/ankle168) 근처면 즉시 진폭↓. 정격 초과 지속 금지.
- [ ] **속도 한계**: 고주파 구간에서 q̇가 vel_limit 근접 시 f1 또는 A 축소.
- [ ] **하드스톱**: chirp 중심±진폭이 관절 물리한계에 여유(≥5°) 남기게 설정.
- [ ] 한 번에 **한 관절씩** 가진(다관절 동시 가진은 커플링으로 식별 오염 가능) → PACE는 전관절
      동시 chirp도 하지만, 초기엔 관절별 개별 스윕 후 합치는 것이 안전.
- [ ] E-stop 준비, 첫 실행은 저진폭(표의 50%)으로 리허설.

### 3.4 산출물
- `data/hind_leg/chirp_data.pt` (dict: time/dof_pos/des_dof_pos). 우리 PACE 태스크가 이 경로를 읽음.

---

## 4. 식별 → 배선 절차 (하드웨어 데이터 확보 후)

### Step A — PACE를 우리 repo에 벤더링(§6) + 우리 로봇 태스크 cfg 작성
`anymal_pace_env_cfg.py`를 복제해 `hind_leg_pace_env_cfg.py` 작성:
```python
from isaaclab_assets.robots.rga import HIND_LEG_CFG
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

HINDLEG_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", ".*_foot_joint"],
    # 토크-속도 곡선: 모터스펙에서 (peak를 saturation, rated를 effort_limit 로)
    saturation_effort={".*_hip_joint":84, ".*_thigh_joint":84, ".*_calf_joint":126, ".*_foot_joint":168},
    effort_limit   ={".*_hip_joint":28, ".*_thigh_joint":28, ".*_calf_joint":42, ".*_foot_joint":56},
    velocity_limit ={".*_hip_joint":29.6, ".*_thigh_joint":29.6, ".*_calf_joint":19.7, ".*_foot_joint":14.8},
    stiffness={".*_hip_joint":65.,".*_thigh_joint":53.,".*_calf_joint":12.,".*_foot_joint":20.},  # 현 Kp
    damping  ={".*_hip_joint":6.,".*_thigh_joint":4.8,".*_calf_joint":1.1,".*_foot_joint":1.0},    # 현 Kd
    encoder_bias={".*":0.0}, friction={".*":0.0}, dynamic_friction={".*":0.0}, viscous_friction={".*":0.0},
    max_delay=10,
)

class HindLegPaceCfg(PaceCfg):
    robot_name = "hind_leg"
    data_dir = "hind_leg/chirp_data.pt"
    joint_order = ["HL_hip_joint","HL_thigh_joint","HL_calf_joint","HL_foot_joint",
                   "HR_hip_joint","HR_thigh_joint","HR_calf_joint","HR_foot_joint"]  # ← 실제 이름/순서 확인
    bounds_params = torch.zeros((4*8+1, 2))  # 33
    def __post_init__(self):
        N=8
        self.bounds_params[0:N,0]=1e-5;   self.bounds_params[0:N,1]=1.0   # armature
        self.bounds_params[N:2*N,1]=7.0                                    # viscous
        self.bounds_params[2*N:3*N,1]=0.5                                  # coulomb
        self.bounds_params[3*N:4*N,0]=-0.1; self.bounds_params[3*N:4*N,1]=0.1  # bias
        self.bounds_params[4*N,1]=10.0                                     # delay
# scene.robot = HIND_LEG_CFG.replace(actuators={"legs": HINDLEG_PACE_ACTUATOR_CFG}, ...fix_root_link)
# gym.register("Isaac-Pace-HindLeg-v0", ... entry_point AnymalDPaceEnvCfg 대응)
```
> 주의: armature 상한 1.0은 anymal(대형) 기준. 우리 유효관성(hip 0.163/thigh 0.133/calf 0.030/foot 0.0019
> kg·m² — rga.py 주석)이 훨씬 작으니 **상한을 0.3~0.5로 좁혀** 탐색효율↑ (실물 기어관성 감안 조정).

### Step B — 식별 실행 (CMA-ES)
```bash
# (우리 repo 루트에서, isaaclab.sh 사용 — CLAUDE.md 규칙)
./isaaclab.sh -p <PACE>/scripts/pace/fit.py --task Isaac-Pace-HindLeg-v0 --num_envs 4096 --headless
```
→ `logs/pace/hind_leg/<timestamp>/mean_XXX.pt` (33-벡터, [-1,1]가 아니라 이미 물리단위로 저장됨:
   `_params_to_sim_params` 적용된 mean). 콘솔에도 armature/viscous/friction/bias/delay 최적값 출력.
   `plot_trajectory.py`로 실측 vs 식별 궤적 오버레이 확인(수렴 검증).

### Step C — 식별값을 `HIND_LEG_CFG`에 배선
`ImplicitActuatorCfg "legs"` → **`PaceDCMotorCfg` 기반**으로 교체(관절별 식별값 하드코딩):
```python
# rga.py — HIND_LEG_CFG.actuators 교체 스켈레톤 (값은 mean_XXX.pt에서)
from pace_sim2real.utils import PaceDCMotorCfg  # 벤더링 후 경로
actuators={
  "legs": PaceDCMotorCfg(
     joint_names_expr=[".*_hip_joint",".*_thigh_joint",".*_calf_joint",".*_foot_joint"],
     saturation_effort={...84/84/126/168...}, effort_limit={...28/28/42/56...},
     velocity_limit   ={...29.6/29.6/19.7/14.8...},
     stiffness={...65/53/12/20...}, damping={...6/4.8/1.1/1.0...},   # PD 게인(유지)
     encoder_bias={ "HL_hip_joint": <bias0>, ... },                 # ← 식별
     viscous_friction={ "HL_hip_joint": <visc0>, ... },             # ← 식별
     friction={ "HL_hip_joint": <cou0>, ... },                      # ← 식별 (static)
     dynamic_friction={ ...동일값... },                              # ← 식별 (=static, 쿨롱)
     max_delay=<delay_int>,                                          # ← 식별(전역)
  ),
}
```
- **armature는 액추에이터 cfg 필드가 아니라 관절속성** → `ArticulationCfg`의 armature로 설정하거나
  env `startup` 이벤트에서 `write_joint_armature_to_sim`으로 주입(PACE `update_simulator`가 하던 방식).
  가장 깔끔한 배선: **startup EventTerm** 하나 추가해 armature/viscous/friction/delay/bias를 식별 mean으로
  써넣기(학습·평가 공통). (rga.py cfg에 armature 필드가 있으면 거기에 직접.)

### Step D — DR을 식별값 "주변 좁은 범위"로 재조정
현재 `hind_leg_env_cfg.py EventCfg`의 넓은 추측 DR을 **식별 중심 ± 좁은 밴드**로 교체:

| 현재(추측 광역 DR) | PACE 후 (식별 중심 tight DR) |
|---|---|
| `randomize_actuator_gains` stiffness/damping ×[0.75,1.5] log_uniform | Kp/Kd ×[0.9,1.1] 로 축소(식별로 불확실성↓) |
| armature: (없음) | **신규**: 식별 armature ×[0.8,1.2] |
| viscous/coulomb friction: (없음) | **신규**: 식별값 ×[0.7,1.3] (마찰 온도의존성 커버) |
| delay: (없음) | **신규**: 식별 delay ± 1~2 step |
| `randomize_rigid_body_material` μ (0.4~1.5) | 실측 접촉마찰 알면 좁힘, 모르면 유지 |
| mass/CoM DR | 유지(모델링 오차 커버) |

> 철학: **"넓은 추측 DR로 robust 억지"에서 "실측 중심 tight DR로 정밀 전이"로 전환**.
> 광역 DR은 정책을 보수적(느리고 뻣뻣)하게 만든다 — 식별 후엔 좁혀서 성능·자연스러움 회복.

### Step E — 재학습
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task HindLeg-Direct-v0 --num_envs 4096
# 평발 변형도:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task HindLeg-Flat-Direct-v0 --num_envs 4096
```
> `FLAT_HIND_LEG_CFG`도 동일 액추에이터 그룹을 쓰므로 같은 식별값 배선(같은 모터).

---

## 5. 하드웨어 없이 지금 가능한 것 — sim-to-sim 검증 (권장, 리스크 제거)

**PACE가 "알려진" 파라미터를 되찾는지" 시뮬만으로 리허설**할 수 있다. `data_collection.py`가 이미
GT 파라미터를 시뮬에 심고 chirp를 걸어 `chirp_data.pt`를 만들므로, 그걸 fit이 복원하는지 보면 된다.

### 5.1 절차 (하드웨어 0)
1. **우리 로봇 PACE 태스크**(§4 Step A) 등록 — `Isaac-Pace-HindLeg-v0`.
2. `data_collection.py`의 GT 파라미터(armature/damping/friction/bias/delay)와
   `trajectory_scale/bias/directions`를 **우리 8관절용으로 수정**(12→8, 우리 진폭표 §3.2).
3. 생성:
   ```bash
   ./isaaclab.sh -p <PACE>/scripts/pace/data_collection.py --task Isaac-Pace-HindLeg-v0 \
       --num_envs 1 --min_frequency 0.1 --max_frequency 10 --duration 20 --headless
   ```
   → `data/hind_leg/chirp_data.pt` (GT 파라미터로 생성된 "가짜 실물" 궤적).
4. 식별:
   ```bash
   ./isaaclab.sh -p <PACE>/scripts/pace/fit.py --task Isaac-Pace-HindLeg-v0 --num_envs 4096 --headless
   ```
5. **검증**: `mean_XXX.pt`의 armature/viscous/coulomb/bias/delay가 3번의 **GT값으로 수렴**하는지 확인
   (`plot_trajectory.py` 궤적 오버레이 + 콘솔 파라미터). 수렴하면 **파이프라인·배선·bounds·관절순서**가
   전부 옳다는 뜻 → 실물 데이터만 꽂으면 됨.

### 5.2 이 리허설이 잡아주는 것
- `joint_order`/관절이름 오매핑, bounds 범위(우리 관절엔 armature 상한 1.0이 너무 큰가), N=8 배선,
  `chirp_data.pt` 포맷 정합, 400Hz/decimation 정합, PaceDCMotor 드롭인 동작 — **전부 하드웨어 전에 검증**.

### 5.3 필요 벤더링/수정 코드 (하드웨어 없이도)
- PACE `source/pace_sim2real` (editable install) + `scripts/pace/*`.
- 신규 `hind_leg_pace_env_cfg.py`(§4 Step A) + gym.register.
- `data_collection.py`를 8관절용으로 fork(진폭/방향/GT). 나머지는 그대로.

---

## 6. 의존성 · 셋업

### 6.1 버전 호환
- PACE 공식 테스트: **Isaac Sim 5.0 / Isaac Lab 0.46.2 / Python 3.11.13**. `setup.py` classifiers에
  `Isaac Sim 4.5.0`·`5.0.0`. **우리 = Isaac Sim 5.1** → 5.0 상위이므로 viscous friction 등 요구기능 충족(OK).
- PACE 의존 pip: `cmaes`, `psutil` (`setup.py INSTALL_REQUIRES`). `cmaes`는 우리 venv에 추가 설치 필요.
- PACE 필수 IsaacLab API(§2.3) 우리 코어에 **전부 존재 확인** → 코어 수정 불필요.

### 6.2 설치 방식 (권장: 벤더링)
- PACE는 **IsaacLab 밖 별도 extension**으로 editable install 하는 구조:
  ```bash
  git clone https://github.com/leggedrobotics/pace-sim2real.git   # IsaacLab 밖
  ./isaaclab.sh -p -m pip install -e pace-sim2real/source/pace_sim2real
  ./isaaclab.sh -p -m pip install cmaes
  ```
- **우리 repo에 넣을 것 vs 외부 참조**:
  - **외부 참조(권장)**: `source/pace_sim2real`(PaceDCMotor/CMAESOptimizer/env cfg) — 그대로 pip -e.
    upstream(RGA) 오염 없이 유지. JSH 폴더에 clone 경로만 문서화.
  - **우리 repo에 벤더링(최소)**: (a) `hind_leg_pace_env_cfg.py`(우리 로봇 태스크), (b) 8관절용
    `data_collection` fork, (c) 최종 식별값이 박힌 `HIND_LEG_CFG` 액추에이터 교체 — 이 3개는 우리 자산이니
    **JSH/ 또는 fork(origin)** 에 커밋. (upstream push 금지 규칙 준수.)
  - `PaceDCMotorCfg` 자체를 우리 rga.py가 import하므로 **런타임에 pace_sim2real 설치 필수** →
    배포/서버 환경에도 pip -e 필요. (완전 독립 원하면 `pace_actuator.py`+`pace_actuator_cfg.py`
    2파일만 우리 repo로 카피해도 됨 — DelayBuffer/DCMotor는 우리 IsaacLab에 이미 있음.)

### 6.3 GPU 서버
- 식별(fit)은 4096 env GPU 병렬 → GPU 서버(jsh@192.168.1.205)에서 실행 권장.
  단 현재 서버접속 블로커(서브넷 불일치) 해결 후. sim-to-sim 리허설은 로컬 GPU로도 소규모 가능.

---

## 7. 미해결 / 리스크

1. **PaceDCMotor 드롭인 여부**: ✅ 사실상 드롭인. `PaceDCMotorCfg`는 `DCMotorCfg` 확장이고
   우리 IsaacLab에 필요한 필드/API 전부 존재(§2.3). **단 타입이 Implicit→Explicit(DCMotor)으로 바뀌므로**
   PhysX 내장 PD가 아니라 Python측 토크계산이 된다 — 학습 속도/수치거동이 미세하게 달라질 수 있음(재튜닝 여지).
2. **armature bounds**: anymal 상한 1.0 kg·m²는 우리 소형 관절(0.002~0.16)에 과대 → 탐색 낭비/국소최적 위험.
   **우리 유효관성 기준으로 상한 좁혀야**(§4 주석). foot(0.0019)은 하한 1e-5도 재검토.
3. **loss가 위치궤적만**: 토크/전류를 직접 안 맞춘다. 마찰/armature가 위치에 미치는 영향이 작은 저부하
   구간에선 식별력이 약할 수 있음 → chirp 진폭·주파수를 충분히 넣어 다이내믹 여기 확보 필요.
4. **delay가 전역 스칼라**: 관절별 지연차이는 못 잡음(1개 값). 우리 통신구조가 관절별로 다르면 근사.
5. **PD 게인 미식별**: PACE는 Kp/Kd를 고정으로 둠. 실물 PD 게인을 정확히 알아야(펌웨어값) 식별이 의미.
   모르면 Kp/Kd도 bounds에 넣도록 CMA-ES 확장 필요(현재 코드엔 없음 — 개조 리스크).
6. **fix_root_link 식별 vs 보행 배포 갭**: 식별은 몸통 고정 자유공진. 접촉/지지 상태의 마찰·backlash는
   다를 수 있음. 그래도 armature·모터마찰은 접촉무관이라 대부분 전이됨.
7. **문서 미완성**: repo `docs/concepts/actuators.md`·`real-world.md` = "Coming soon"/빈 파일.
   실물 로깅 상세(하드웨어측 스크립트)는 repo에 없음 → **우리가 로깅 코드를 자체 작성**해야(포맷은 §3.1로 확정).
8. **버전 드리프트**: PACE는 IsaacLab 0.46.2 대상. 우리 IsaacLab 마이너버전이 더 나가면
   `write_joint_*` 시그니처 변할 수 있음 — §5 sim-to-sim 리허설이 이 리스크를 사전 검출.
9. **cmaes 패키지**: 외부 의존(pip). 서버 오프라인 시 사전 설치 필요.

---

## sim-to-sim 검증 결과 (dry run 완료 — 2026-07-24)

**결론: 파이프라인 검증 완료.** GT를 주입한 chirp 데이터로 CMA-ES가 잘-관측되는 파라미터
(armature·delay)를 거의 완벽히 복구 → **우리 로봇(HindLeg 8-DOF)에서 PACE 식별 파이프라인이
작동함을 실물 데이터 없이 입증.** 실물 chirp 로그만 확보되면 그대로 식별 실행 가능.

### 서버/환경
- SSH `jsh@192.168.1.205`, conda `isaac-5.1`, `CUDA_VISIBLE_DEVICES=2`.
- PACE(벤더링): `/mnt/ssd1/jsh/pace-sim2real` (editable install, `cmaes` 포함).
- 실행 워크스페이스: `/mnt/ssd1/jsh/pace_run/` (스크립트·로그).
- 태스크 등록: `Isaac-Pace-HindLeg-v0` (`.../tasks/manager_based/pace/hindleg_pace_env_cfg.py`),
  `HIND_LEG_CFG` + `PaceDCMotor` 8관절, GT 주입·bounds 축소.
- **GPU 공유 주의**: GPU2에서 4096-env `HindLeg-Flat-Direct-v0` 학습(tmux `trainflat`) 동시 구동 →
  PACE는 `data_collection --num_envs 1`, `fit --num_envs 256`로 소형 유지, 별도 tmux 소켓 `-L pace` 사용.

### device 버그 수정 (근본 원인 + 수정)
- 증상: 100 iter CMA-ES는 정상 수렴했으나 **마지막 결과추출**에서
  `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`.
- 파일/라인: `source/pace_sim2real/pace_sim2real/optim/cma_es.py:157`
  `get_best_sim_params()` → `_params_to_sim_params()` (line 153).
- 근본 원인: `best_params = torch.tensor(self.optimizer._mean)` — cmaes 라이브러리의 `_mean`은
  numpy(CPU) → device 미지정으로 **CPU 텐서** 생성. `self.bounds`는 cuda:0라 연산 시 불일치.
  (동일 연산인 line 103의 finish-checkpoint 저장은 `device=self.device`가 있어 정상 → `mean_099.pt`가
  이미 저장돼 있었음 = `self.bounds`가 cuda임을 방증. CPU 텐서는 line 157 하나뿐이었음.)
- 수정(벤더 install만, `RobotSW_IsaacLab/source`는 미변경):
  - `cma_es.py:157` → `torch.tensor(self.optimizer._mean, device=self.device)` (근본 수정)
  - `cma_es.py:38`  → `self.bounds = bounds.to(device)` (방어적 하드닝)
- 검증: 재실행 시 crash 없이 `[RESULT]` 정상 덤프(아래).

### anymal Phase-1 결과
- `Isaac-Pace-Anymal-D-v0` fit(12관절, num_envs 256) **정상 기동·CMA-ES 수렴 진행**
  (min score ~0.0116로 안정화). **asset 블로커·API 불일치 없음** → 레퍼런스 로봇에서 파이프라인 유효.
- 단, 로그가 iter 35/60에서 끊김(중단, crash/finish 아님) → anymal 최종 `[RESULT]` 덤프는 없음.
  device 버그는 anymal에는 도달 전(끝단에서만 발생)이라 무관.

### HindLeg 복구 vs GT (8관절, 100-iter 수렴 `mean_099.pt`, min score 3.3e-5)
joint_order: `[HL_hip, HR_hip, HL_thigh, HR_thigh, HL_calf, HR_calf, HL_foot, HR_foot]`

| 파라미터 | GT | 복구(8관절) | 오차 | 판정 |
|---|---|---|---|---|
| **armature** [kg·m²] | hip .05 / thigh .04 / calf .02 / foot .01 | .0511 .0499 .0401 .0394 .0200 .0201 .0100 .0100 | **±2.1%** | ★거의 완벽 |
| **delay** [step] | 5 | 5.324 | +6.5% | ★양호 |
| **viscous** [Nm·s/rad] | 0.10 (전관절) | .265 .132 .092 .101 .092 .090 .098 .092 | hip제외 ±10% | thigh/calf/foot 양호, **hip 과대추정** |
| **coulomb** [Nm] | 0.03 (전관절) | .090 .050 .068 .068 .073 .068 .071 .083 | +67~+201% | 체계적 과대추정(관측성 낮음) |
| **bias** [rad] | 0.02 (전관절) | .020 .031 .008 .020 .020 .021 .019 .012 | 5/8 ±6%, 3관절 산포 | 대체로 양호 |

**해석**: armature·delay = sim2real 최우선 파라미터(반사관성·구동지연)를 거의 완벽 복구 = 파이프라인 유효.
viscous는 움직임이 큰 관절(thigh/calf/foot) 양호, hip은 고강성(Kp65)+작은 chirp진폭(0.15rad)이라
점성마찰 관측성이 낮아 과대. coulomb·bias는 소형관절·저부하에서 관측성이 약한 파라미터(마찰↔점성 trade-off,
PACE 문헌과 동일 경향). → **실물 적용 시 armature·delay는 신뢰, coulomb/bias는 chirp 진폭·주파수를 더 넣어
여기(excitation) 확보 필요**(§7-3 리스크와 일치).

### 재현 커맨드 (실물 chirp 데이터 도착 시 이 절차 그대로)
```bash
ssh -i ~/.ssh/id_rga jsh@192.168.1.205
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh && conda activate isaac-5.1
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
# (1) 데이터 수집: sim-to-sim은 GT주입 스크립트, 실물은 chirp_data.pt를 data/hindleg_sim/에 배치
python /mnt/ssd1/jsh/pace_run/data_collection_hindleg.py --task Isaac-Pace-HindLeg-v0 --num_envs 1 --headless
# (2) 식별(CMA-ES): GPU 공유 시 별도 소켓 + 소형 num_envs
tmux -L pace new-session -d -s pacefit \
  "bash /mnt/ssd1/jsh/pace_run/hindleg_fit.sh > /mnt/ssd1/jsh/pace_run/logs/hindleg_fit.log 2>&1"
#   hindleg_fit.sh = fit_run.py --task Isaac-Pace-HindLeg-v0 --num_envs 256 --max_iter 100 --headless
#   ★ PYTHONUNBUFFERED=1 / python -u 필수(로그 실시간 flush; 없으면 진행이 파일에 안 보임)
# (3) 결과: 로그의 [RESULT] armature/viscous/coulomb/bias/delay,
#     또는 logs/pace/hindleg_sim/<ts>/mean_<iter>.pt (분포평균 sim params) 로드
```
- 로컬 보존: `RobotSW_IsaacLab/JSH/pace_hindleg/` (`hindleg_pace_env_cfg.py`, `data_collection_hindleg.py`,
  `fit_run.py`, `cma_es.py`[수정본], `gt.pt`, `report.py`).
- 주의(하드경험): `pkill -f`에 자기 SSH 명령과 겹치는 패턴 금지(자기 셸이 죽음) → **PID로만 kill**.
  Python 로그 미출력은 대개 stdout 블록버퍼링 → `PYTHONUNBUFFERED=1`/`python -u`로 해결.

---

## 8. 다음 액션 요약
- [ ] (지금 가능) PACE clone + editable install + `cmaes` 설치.
- [ ] (지금 가능) `hind_leg_pace_env_cfg.py` 작성 + `Isaac-Pace-HindLeg-v0` 등록.
- [x] (완료 2026-07-24) **sim-to-sim 리허설**(§5) — 파이프라인·bounds·관절순서 검증 + device 버그 수정. 결과=§「sim-to-sim 검증 결과」.
- [ ] (하드웨어 도착) §3 프로토콜로 chirp 로깅 → `data/hind_leg/chirp_data.pt`.
- [ ] (하드웨어 도착) `fit.py` 실행 → 식별 → §4 Step C 배선 → tight DR → 재학습.

> 관련 메모리: `pace-sim2real-actuator-id`, `sim2real-checklist`, `no-unphysical-sim2real-hacks`,
> `biped-mpc-reimpl`(RL 피벗), `gpu-server-hindleg-rl`.
