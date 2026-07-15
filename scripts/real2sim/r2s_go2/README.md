# R2S-GO2 — Real2Sim GO2 실험 가이드

Unitree GO2를 Isaac Sim에 올리고, ROS2 GUI로 관절을 실시간 제어하며,
필요하면 **시뮬레이터와 실로봇을 동시에** 같은 명령으로 구동/비교하는 도구 모음.

```
GUI 버튼 ─▶ ROS2 /lowcmd ─▶ sim_bridge ─▶ UDP:9871 ─▶ sim_runner(Isaac GO2)
   ▲                                                        │
   │            ROS2 /lowstate ◀─ sim_bridge ◀─ UDP:9872 ◀──┘
 Monitor(별도 프로세스): /lowcmd(action) + /lowstate(sim/robot) 실시간 plot
```

그리고 **매달린 실기의 chirp 데이터로 GO2 물성(armature/마찰/지연/바이어스)을 식별**한다 → [§10 PACE](#10-pace-시스템-식별--go2-물성-실측-contract-12).

고정 계약(토픽/패킷/조인트 순서/게인)은
[`CONTRACT.md`](../../../source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md) 참고.

---

## 1. 실행 환경 (Python 두 세계, 반드시 구분)

| 프로세스 | Python | 역할 |
|---|---|---|
| `sim_runner_go2.py` | **Isaac conda `isaac-6.0` (3.12)** | Isaac Sim에서 GO2 구동, UDP로 명령/상태 교환. rclpy 미사용. |
| `sim_bridge.py` | **시스템 `/usr/bin/python3` (3.10) + ROS2 humble** | `/lowcmd`↔UDP↔`/lowstate` 브릿지 |
| `gui_controller.py` | 시스템 3.10 + humble | PyQt5 GUI, `/lowcmd` 50Hz 발행 |
| `monitor.py` | 시스템 3.10 + humble | (GUI가 spawn) 실시간 plot 창 |

> **함정**: conda가 켜진 셸에서 `python3`는 conda(3.12)를 가리켜 rclpy 임포트가 깨진다.
> `run_*.sh` 런처들은 conda를 **완전히 비활성화**한 뒤 humble + unitree_go overlay를 소싱하고
> `/usr/bin/python3`로 실행하므로, conda가 켜진 터미널에서 그냥 실행해도 된다.

---

## 2. 빠른 시작 (별칭 권장)

한 번만 별칭을 등록하면 세 터미널에서 `cd`처럼 바로 쓸 수 있다:

```bash
source scripts/real2sim/r2s_go2/r2s_commands.sh
```

**실시간 구동 (M1/M2)**

| 별칭 | 터미널 | 실제 스크립트 |
|---|---|---|
| `r2s_sim` | 1 (Isaac) | `run_sim_runner.sh` — conda `isaac-6.0` 활성화 + livestream + sim_runner |
| `r2s_udp` | 2 (ROS2) | `run_sim_bridge.sh` — 브릿지 |
| `r2s_gui` | 3 (ROS2) | `run_gui_controller.sh` — GUI |
| `r2s_all` | (선택) | `run_all.sh` — tmux 3-pane 동시 기동 (`r2s_all stop`으로 종료) |

**PACE 시스템 식별 (CONTRACT §12)** — GO2 물성(armature/마찰/지연/바이어스)을 실측으로 식별

| 별칭 | 세계 | 하는 일 |
|---|---|---|
| `r2s_gain` | ROS2 (실기) | Unitree kp가 IsaacLab과 같은 N·m/rad인지 실측 (**수집 전 필수**). `--selftest`는 로봇 불필요 |
| `r2s_chirp` | ROS2 (실기) | 매달린 GO2에서 500 Hz chirp 수집 → `.npz`. `--dry-run` / 저진폭 공진 스윕 지원 |
| `r2s_convert` | Isaac | `.npz` → PACE `.pt` 변환 + **품질·리그 공진 판정** |
| `r2s_collect` | Isaac | 합성 chirp 생성(GT 주입) — 실로봇 없이 파이프라인 검증용 |
| `r2s_fit` | Isaac | 다중 시퀀스 CMA-ES 적합 (49개 파라미터) |
| `r2s_validate` | Isaac | hold-out 검증 — 식별값 vs 현재 nominal cfg 비교 |
| `r2s_tb` | Isaac | 텐서보드 |
| `r2s_tuner_sim` | Isaac | tuner: chirp/replay 재생 + 슬라이더 물성 실시간 반영 (§10.6) |
| `r2s_tuner_gui` | (UDP) | tuner: 물성 슬라이더 GUI + 오버레이 플롯 (§10.6) |

> ⚠ 실기 수집 전 **필수 2건**: ① `r2s_gain`으로 kp 규약 확인, ② `r2s_chirp --amplitude_scale 0.3`
> 저진폭 스윕으로 **매다는 거치대의 공진 주파수** 확인. 거치대가 흔들린 대역의 데이터로는 관절을
> 식별할 수 없다(논문도 같은 이유로 매단 ANYmal의 chirp를 2 Hz로 제한했다).

각 별칭은 인자를 그대로 전달한다: `r2s_sim --headless`, `GPU=0 r2s_sim`, `FIX_BASE=0 r2s_sim`.

---

## 3. 실행 순서 (sim 전용)

```bash
# 터미널 1 — Isaac sim_runner
r2s_sim
#   또는 수동:
#   conda activate isaac-6.0
#   python scripts/real2sim/sim_runner_go2.py --num_envs 1 [--fix_base] [--headless]

# 터미널 2 — ROS2 <-> UDP 브릿지
r2s_udp                                    # = bash .../run_sim_bridge.sh

# 터미널 3 — PyQt GUI (디스플레이 필요)
r2s_gui                                     # = bash .../run_gui_controller.sh
```

GUI가 뜨면 로봇은 **엎드린(prone) 자세**로 시작한다(실기 GO2와 동일).

---

## 4. GUI 버튼

| 버튼 | 동작 |
|---|---|
| **Stand Up** | prone → folded → stand 궤적(go2_stand_example 방식)으로 기립 |
| **Default (stand)** | `DEFAULT_POSE`(선 자세)로 보간 이동 |
| **Sit** | `SIT_POSE`(웅크림)로 보간 이동 |
| **Joint Step** | 선택 관절만 delta[rad]만큼 스텝 |
| **Sine Sweep** | 선택 관절에 사인 궤적 주입(Start/Stop) |
| **Monitor** | 실시간 plot 창 열기(아래 §5) |

모든 명령은 `kp=25, kd=0.5`(전 관절), `mode=0x01`, `dq=0`, `tau=0`으로 50Hz 연속 발행된다
(실로봇 watchdog heartbeat + CRC/헤더 포함, `CONTRACT.md` §6·§8).

**시작 자세 접근(startup approach)**: gui를 켜면 `/lowstate`로 **로봇의 현재 자세를 먼저 읽어** 그
자세에서 시작 자세(STAND_FOLDED)로 약 2.5초에 걸쳐 **부드럽게 보간**한다(첫 상태를 받기 전엔 발행
보류). 예전처럼 켜자마자 고정 자세를 쏴서 툭 스냅하지 않는다. 보간 시간은 `STARTUP_DURATION_S`.

**발행 + 목표 생성이 별도 프로세스**: `/lowcmd` 50Hz 발행과 **목표 보간·사인 생성**을 UI가 아니라
**별도 publisher 프로세스**가 한다. UI(슬라이더·버튼)로 Qt event loop가 바빠도 목표가 매끄럽게(50Hz
균일) 갱신된다 — 예전엔 UI 조작 시 목표값이 띄엄띄엄 갱신돼 로봇이 툭툭 끊겼다(hz는 50이어도).
UI는 공유 메모리에 **"모션 스펙"(어디로 몇 초에 걸쳐 가라)만** 쓰고, publisher가 경과 시간으로 목표를
계산해 발행한다(monitor 격리와 동일 이유, `CONTRACT.md` §11.3).

---

## 5. Monitor — 실시간 plot

GUI의 **Monitor** 버튼 → 선택 모터의 3개 plot 창이 **별도 프로세스**로 뜬다:

- **q** (관절각): action(점선=명령) / sim / robot
- **tau_est** (토크): sim / robot
- **dq** (각속도): sim / robot

- **action** = `/lowcmd`의 목표각(절대값). residual 아님 — 이 시스템은 직접 위치 제어.
- **sim** = `/sim/lowstate`(shared) 또는 `/lowstate`(sim 전용)의 엔코더.
- **robot** = `/lowstate`(실로봇)의 엔코더.

> 왜 별도 프로세스인가: matplotlib 렌더가 GUI와 같은 event loop에 있으면 50Hz `/lowcmd`
> heartbeat를 굶겨 실로봇이 protection fault로 빠질 수 있다. 프로세스 격리로 GUI는 발행만,
> monitor는 렌더만 한다(`CONTRACT.md` §11).

모터 선택 콤보 + Pause 버튼. sim 전용 모드에서는 robot 라인이 sim과 겹치지 않도록 자동으로
"sim only"로 표시된다.

---

## 6. 동시 구동 (sim + 실로봇, ⚠ 실로봇이 움직인다)

GUI 버튼 하나의 `/lowcmd`를 **sim과 실로봇 양쪽**이 받게 하는 모드.

> ⚠ **안전**: `R2S_SHARED=1 r2s_gui`를 켜는 순간 실로봇도 즉시 `/lowcmd`를 받는다.
> **실로봇을 안전 위치(엎드림/매달기) + sport release 상태**로 둔 뒤 켤 것.
> 실로봇 이더넷 연결 + `192.168.123.x` static IP 세팅이 선행되어야 한다.

```bash
# 터미널 1 — sim (지면, 실로봇과 정합하려면 fix_base 끄기)
FIX_BASE=0 r2s_sim
# 터미널 2 — 브릿지 (shared: cyclonedds + /sim/lowstate remap)
R2S_SHARED=1 r2s_udp
# 터미널 3 — GUI (shared)
R2S_SHARED=1 r2s_gui
```

shared 모드에서 sim 상태는 `/sim/lowstate`로, 실로봇은 `/lowstate`로 분리 발행되어
Monitor가 sim vs robot을 나란히 비교한다.

---

## 7. 실로봇 통신 점검 (M2, sim 불필요)

로봇 이더넷 연결 후 서버↔GO2 ROS2 통신을 점검:

```bash
bash scripts/real2sim/r2s_go2/check_go2_comms.sh [iface]
#   iface 생략 시 192.168.123.x 보유 iface 자동탐지(예: ens10f1).
#   ping + 기대 토픽(/lowstate,/lowcmd,...) + read_lowstate.py 파싱까지 PASS/FAIL.
```

- `read_lowstate.py` — `/lowstate` 구독해 motor q/dq, imu quat, foot_force 출력(실측 ~500Hz).
- `safe_joint_test.py` — 엎드린 상태에서 한 관절만 조심스럽게 흔드는 안전 테스트(`--dry-run` 지원).

> daemon이 stale 토픽 캐시를 들면 로봇 토픽이 안 보인다 — `check_go2_comms.sh`는 discovery 전에
> `ros2 daemon`을 재시작해 이 함정을 피한다.

---

## 8. 옵션 / 환경변수

| 변수 / 플래그 | 기본 | 의미 |
|---|---|---|
| `--fix_base` / `FIX_BASE` | 지면(0) | base를 공중(z=0.5)에 고정 — 접촉/균형 없이 관절 추종만 볼 때 |
| `--headless` | off | 렌더 창 없이 실행(원격/자동화) |
| `--num_envs` | 1 | 시뮬 환경 수(브릿지는 env0만 사용) |
| `R2S_SHARED` | 0 | 1이면 브릿지/GUI를 cyclonedds로 전환 → 실로봇과 같은 도메인 |
| `R2S_ROBOT_IFACE` | 자동탐지 | 로봇 네트워크 인터페이스(예: `ens10f1`) |
| `R2S_SIM_STATE_TOPIC` | 모드별 자동 | Monitor의 sim 소스 토픽(shared=`/sim/lowstate`, sim전용=`/lowstate`) |
| `R2S_ROBOT_STATE_TOPIC` | `/lowstate` | Monitor의 robot 소스 토픽 |
| `ROS_DOMAIN_ID` | 0 | 모든 ROS2 터미널이 동일해야 discovery됨 |
| `GPU` | 2 | sim_runner / PACE CUDA 디바이스 |
| `R2S_LOG_BASEZ` | off | 1이면 sim_runner가 base_z + mean thigh 디버그 로그 출력 |

PACE 식별(§10) 전용:

| 변수 | 기본 | 의미 |
|---|---|---|
| `NUM_ENVS` | 4096 | `r2s_fit`의 CMA-ES population (= 시뮬 환경 수) |
| `ROBOT` | `go2_sim` | `r2s_tb`가 볼 로그 디렉터리(`logs/pace/<ROBOT>`). 실기는 `go2_real` |
| `SIM_ENV` | `isaac-6.0` | Isaac conda env 이름 |
| `R2S_ROBOT_IFACE` | 자동탐지 | `r2s_gain` / `r2s_chirp`가 쓸 로봇 인터페이스 |

---

## 9. 트러블슈팅

- **ros2 노드가 서로 안 보임 / `/lowstate` 안 옴**: 모든 터미널의 `ROS_DOMAIN_ID`와
  `RMW_IMPLEMENTATION`이 일치해야 한다. `echo $ROS_DOMAIN_ID $RMW_IMPLEMENTATION`로 확인.
  sim 전용은 기본 FastDDS, shared는 CycloneDDS — 섞이면 안 보인다.
- **sim_runner가 `ModuleNotFoundError`로 죽음**: `./isaaclab.sh -p`가 base conda를 가리키는 경우.
  `conda activate isaac-6.0` 후 실행하거나 `r2s_sim` 사용.
- **`/lowstate`는 오는데 로봇이 안 움직임**: 지면에 떨어졌거나 관절 순서 어긋남 → `--fix_base`로 격리 재확인.
- **Monitor 창이 안 뜸/에러**: 디스플레이가 있는 환경인지, 이미 실행 중인지 확인(중복 실행 방지됨).
- **엎드린 자세에서 sim≠robot, 목표 미도달**: 버그 아님 — kp=25가 약해 뒷 hip이 벌어지고 calf는
  기계 한계에 박힌다(접촉). Real2Sim gap의 실측 신호. 자세한 분석은 `CONTRACT.md` 참고.
- **실로봇이 "드르륵드륵" 떨며 명령을 제대로 안 따름 (shared 모드)**: 통신 속도 문제가 아니라
  `/lowcmd`에 **publisher가 2개**라서다 — gui(50Hz) + 로봇 sport 서비스(`_CREATED_BY_BARE_DDS_APP_`,
  ~450Hz)가 상충하는 명령을 쏘면 로봇이 두 목표 사이를 튕긴다. **sport를 내리면**(리모컨 L2+A→L2+B)
  해결된다. 확인: `ros2 topic info /lowcmd --verbose | grep "Publisher count"` → **1**이어야 정상.

---

## 10. PACE 시스템 식별 — GO2 물성 실측 (CONTRACT §12)

CONTRACT §6이 "튜닝 대상"으로 남겨둔 GO2 물성(armature / 마찰 / 지연 / 엔코더 바이어스)을 **매달린
실기의 chirp 데이터로부터 자동 식별**한다. 기반은 [PACE](https://github.com/leggedrobotics/pace-sim2real)
(ETH RSL, arXiv:2509.06342)이며 `source/pace_sim2real/`에 벤더링되어 있다.

식별하는 것은 **12관절 × 4 + 1 = 49개**: armature, viscous 마찰, Coulomb 마찰, 엔코더 바이어스, 전역 지연.
CMA-ES가 **후보 파라미터 1개 = 시뮬 환경 1개**로 4096개를 동시에 굴려, 실측 궤적을 가장 잘 재현하는
조합을 찾는다(`Isaac-R2S-Go2-Sysid-v0`, 500 Hz, 공중 고정).

### 10.1 파이프라인

```
[실기]  r2s_gain    kp 규약 실측 (필수 선행)
        r2s_chirp   매달린 GO2에 500Hz chirp → /lowstate 수집 → .npz
                       │
[Isaac] r2s_convert  .npz → .pt  (+ 품질·거치대 공진 판정)
        r2s_fit      다중 시퀀스 CMA-ES (49개 파라미터)
        r2s_validate hold-out 검증 — 식별값 vs 현재 nominal cfg
```

여기신호는 `chirp.py` **한 곳**에서 나온다(순수 stdlib → Isaac 3.12 / ROS2 3.10 양쪽에서 임포트).
sim 재생과 실기 수집이 같은 함수를 써야 식별이 성립한다.

### 10.2 ⚠ 실기 수집 전 필수 2건 — 절차

두 절차 모두 **로봇을 매달고 sport(고수준) 서비스를 내린 뒤**(리모컨 L2+A → L2+B) 실행한다.
`--suspended` 플래그가 없으면 도구는 `/lowcmd`를 전혀 발행하지 않는다(안전 가드).

#### A. kp 단위 규약 실측 (`r2s_gain`)

**왜 필요한가.** PACE는 PD 게인을 식별하지 않는다 — `{armature, damping, kp, kd}`의 **공통 스케일이
폐루프 거동을 보존**해서 최적해가 축퇴하기 때문이다(논문 명시). Unitree kp가 IsaacLab과 다른 단위면
CMA-ES는 armature/마찰을 그만큼 편향시켜 **위치는 완벽히 재현하면서 물리값은 틀린** 답을 낸다.
score는 훌륭하게 나온다. **이 오차는 식별 데이터로 잡을 수 없다** — 사전 측정이 유일한 방어다.
원리엔 모델이 필요 없다: PD 법칙이 `τ = kp·e`이므로, 목표각을 조금씩 옮겨 `(e, tau_est)`를 모아
**직선의 기울기**를 뽑으면 그게 실효 kp다.

**절차**

1. **추정기 자체검증**(로봇 불필요). 합성 데이터로 직선 적합기가 맞는지부터 확인한다.
   ```bash
   r2s_gain --selftest
   ```
   → `selftest OK — kp_true=25.0, 추정=24.96 … R²=0.9998` 가 나오면 도구는 정상.
2. **실측**. FR_thigh(관절 1)를 여러 오프셋으로 옮기며 정착 후 `(e, tau_est)`를 모아 기울기를 뽑는다.
   ```bash
   r2s_gain --suspended --joint 1 --kp 25
   ```
   `--settle`(오프셋마다 정착 대기, 기본 1.5 s) · `--window`(정착 후 평균 구간 0.5 s) · `--engage`(kp
   램프-인/아웃 2 s)로 조정한다.
3. **출력 읽기**. 표 아래 요약을 본다:
   ```
   실효 kp (기울기): 24.9  N·m/rad
   절편 b         : +0.02 N·m      (0에 가까워야 정상 — 크면 오프셋/마찰)
   R²             : 0.997          (1에 가까워야 PD 법칙대로 동작)
   ▶ α = 실효/명령 = 0.998
   ```
4. **판정.**
   - `α ≈ 1` 그리고 `R² > ~0.95` → Unitree kp가 IsaacLab과 **같은 N·m/rad 규약**이다. 그대로 진행.
   - `α`가 1에서 벗어남 → sysid/학습 cfg의 `stiffness`를 실효값으로 쓰거나, 원인(단위/기어비)을 먼저 규명.
   - `R²`가 낮음 → 정착이 덜 됐거나 마찰/포화가 지배. `--settle`을 늘리거나 오프셋(kp)을 키운다.
5. **관절 대표 3종 반복 권장** — hip/thigh/calf는 기어비·마찰이 달라 규약이 관절마다 다를 수 있다:
   `--joint 0`(hip) · `--joint 1`(thigh) · `--joint 2`(calf).

#### B. 리그(매다는 거치대) 공진 스윕 (`r2s_chirp --amplitude_scale 0.3` → `r2s_convert`)

**왜 필요한가.** 로봇을 매다는 구조물이 흔들리면 측정된 관절 움직임에 "몸통이 출렁여서 생긴 것"이
섞인다. PACE는 base 고정을 전제하므로 그 데이터로는 관절을 식별할 수 없다. 논문도 같은 이유로 매단
ANYmal의 chirp를 **2 Hz까지만** 올렸다("structural constraints"). 수집기는 **IMU를 함께 기록**하고,
`r2s_convert`가 chirp 순시 주파수 대역별 base 각속도를 표로 뽑아 공진 대역을 경고한다.

**절차**

1. **dry-run으로 계획 확인**(무모션, 발행 없음). 궤적·soft limit·리밋 캡을 검사한다.
   ```bash
   r2s_chirp --dry-run
   ```
2. **저진폭 스윕 수집**. 진폭 30%(`--amplitude_scale 0.3`)라 관절 정보는 거의 없지만, 넓은 주파수로
   거치대를 훑어 공진을 드러낸다. IMU가 함께 기록된다.
   ```bash
   r2s_chirp --suspended --amplitude_scale 0.3 --out sweep_check.npz
   ```
   (기본 chirp: 0.1→10 Hz, 20 s, 500 Hz 발행. `--min_frequency` / `--max_frequency`로 대역 조정.)
3. **공진 판정**. Isaac 쪽에서 변환기를 돌리면 "리그 공진 점검" 표가 나온다.
   ```bash
   r2s_convert --capture data/go2_real/sweep_check.npz
   ```
   ```
   === 리그 공진 점검 (chirp 순시 주파수 대역별 base 각속도) ===
           대역 [Hz]   |w| mean    |w| max
        0.10–  1.10      0.031      0.088
        …
        5.10–  6.10      0.402      0.713  ⚠
   ⚠ 5.54 Hz 부터 base가 크게 움직인다(|w| > 0.5 rad/s).
     --max_frequency 를 5.5 Hz 아래로 낮춰 다시 수집할 것.
   ```
   판정 임계는 `RIG_RESONANCE_GYRO_WARN = 0.5 rad/s`(`convert_capture_to_pt.py`).
4. **상한 결정.** 경고가 나온 **첫 공진 주파수보다 낮게** 본 수집의 `--max_frequency`를 잡는다.
   전 대역이 `< 0.5 rad/s`(✅)면 기본 10 Hz를 그대로 써도 된다.

### 10.3 실기 순서

```bash
r2s_gain  --selftest                                               # A-1: 추정기 검증 (로봇 불필요)
r2s_gain  --suspended --joint 1 --kp 25                            # A: kp 규약 실측 (관절 0/1/2 반복)
r2s_chirp --dry-run                                                # B-1: 계획/리밋 검사 (무모션)
r2s_chirp --suspended --amplitude_scale 0.3 --out sweep_check.npz  # B: 공진 스윕
r2s_convert --capture data/go2_real/sweep_check.npz                #    → 몇 Hz부터 흔들리는지
# ↑ 공진 첫 주파수 확인 (기본 캡은 2 Hz — chirp_collector.MAX_FREQUENCY)
r2s_chirp --interactive --kp 25 --kd 0.5 --out chirp_kp25.npz      # 본 수집 (게인 세트별 반복)
#   대화형: ① 현재 자세에서 천천히 stand-up → ② 로봇 매달고 Enter → ③ chirp 발행+기록
r2s_convert --capture data/go2_real/chirp_kp25.npz
r2s_fit ; r2s_validate                                             # 적합 → hold-out 검증
```

**본 수집 흐름(`--interactive`)**: kp 램프-인 → **천천히 stand-up**(`--standup_time`, 기본 3s) →
**"로봇을 매달고 Enter"** 프롬프트(대기 중에도 자세 홀드=heartbeat 유지, Ctrl+C 취소) → chirp 중심
이동 → chirp 발행+기록 → kp 램프-다운. `--suspended`는 stand-up/Enter 없이 이미 매달린 전제로 바로
간다(반복·자동 수집용). 주파수는 기본 **2 Hz**로 캡된다(`MAX_FREQUENCY`, 논문의 매단 ANYmal 값).

**게인은 여러 세트로 수집한다.** 데이터셋마다 kp/kd가 함께 저장되고 재생 시 복원된다 — 틀린 게인으로
재생하면 잡음이 아니라 **편향**이 생긴다. 논문 방식대로 여러 시퀀스로 적합하고, **보지 않은 게인**을
hold-out으로 빼서 검증한다. 게인이 높은 세트는 **진폭을 줄인다**(토크가 effort limit 23.5 N·m에
포화하면 그 구간은 파라미터에 무감각해져 정보를 파괴한다).

### 10.4 안전

- sport(고수준) 서비스를 먼저 내릴 것 (리모컨 L2+A → L2+B).
- **`--suspended` 없이는 발행하지 않는다.** 발이 지면에 닿으면 접촉력이 모델 밖 항으로 들어가 식별이 오염된다.
- 목표각은 chirp 중심 ±`max_dev` clamp + **발행 전 soft limit 검사**(calf가 가장 빡빡하다).
- 추종오차가 `abort_dev`를 넘으면 즉시 중단 → 부분 데이터 저장 → kp 램프-다운.
- `--dry-run`은 `/lowcmd`를 전혀 발행하지 않는다.

### 10.5 실로봇 없이 검증된 것

`r2s_collect`가 **알려진 GT를 주입한 합성 chirp**를 만들고 `r2s_fit`이 그걸 되찾아오는지로 파이프라인
정확성을 검증한다(Phase 0/1 게이트 통과: GT 소수 4자리 복원, hold-out RMSE 5e-6 rad).
단 이건 sim-to-sim 결정론이라 **파이프라인만** 증명한다 — 실데이터 합격선은 score < 0.005이고
수렴에 10–24시간이 걸린다.

### 10.6 tuner 모드 — 물성 실시간 튜닝 (bounds/초기분포 잡기)

CMA-ES가 자동으로 49개를 찾기 전에, **사람이 전역 스칼라 6개(armature/viscous/Coulomb/kp/kd/delay)를
슬라이더로 쓸어보며** 실기 궤적과 겹쳐 보고 `Go2PaceCfg`의 **bounds와 초기 분포**를 실측 기반으로
잡는 도구다. `Isaac-R2S-Go2-Sysid-v0`를 num_envs=1로 재사용한다(live 경로 불변, 별도 UDP 포트 9875/9876).

```bash
# 터미널 1 (Isaac): chirp 재생 + 슬라이더 값 실시간 반영
r2s_tuner_sim
#   녹화 실기 궤적과 겹쳐 보려면: r2s_tuner_sim --replay data/go2_real/chirp_kp25.pt
# 터미널 2: 물성 슬라이더 GUI (순수 UDP, ROS2 불필요). "Launch Plot"으로 오버레이 창.
r2s_tuner_gui
```

- 슬라이더를 움직이면 `write_joint_*_to_sim`으로 sim 물성이 **즉시** 바뀐다.
- **Launch Plot** → 선택 관절의 `q_cmd`(점선) / `q_sim`(파랑) / `q_real`(빨강, `--replay` 시) 오버레이 +
  롤링 RMSE. `--replay`가 없으면 chirp만 구동하며 `q_sim`이 물성에 반응하는지만 본다.
- 전역 스칼라만 다룬다(per-joint 49개는 CMA-ES 몫). 여기서 잡은 대략적 범위를 `r2s_go2_sysid_cfg.py`의
  `bounds_params`에 반영한다.

계획 전문: `.omc/plans/pace-go2-sim2real-plan.md` · 포팅 내역: `source/pace_sim2real/VENDORING.md`

---

## 11. 파일

**Real2Sim 실시간 구동 (M1/M2)**

| 파일 | 역할 |
|---|---|
| `sim_runner_go2.py` (상위 폴더) | Isaac 런처 + UDP 루프 |
| `sim_bridge.py` | `/lowcmd`↔UDP↔`/lowstate` 브릿지(rclpy) |
| `gui_controller.py` | PyQt5 GUI, `/lowcmd` 50Hz 발행 |
| `monitor.py` | 실시간 plot(별도 프로세스) |
| `motions.py` | 자세 상수 + 보간/스텝/사인 유틸(순수 함수) |
| `lowcmd_crc.py` | LowCmd CRC/헤더(실기 deploy) |
| `r2s_udp.py` | UDP 패킷 스키마(공유, stdlib만) — 재구현 금지 |
| `read_lowstate.py` / `safe_joint_test.py` | 실로봇 상태 읽기 / 안전 관절 테스트 |
| `check_go2_comms.sh` | 실로봇 통신 점검 |

**PACE 시스템 식별 (§10)**

| 파일 | 세계 | 역할 |
|---|---|---|
| `chirp.py` | **공유** (stdlib) | 여기신호 정의 + soft limit 검사. sim/실기가 **같은 함수**를 쓴다 |
| `gain_check.py` | ROS2 | Unitree kp 단위 규약 실측 (`--selftest`는 로봇 불필요) |
| `chirp_collector.py` | ROS2 | 매달린 GO2에서 500Hz chirp 수집 → `.npz` (IMU 포함) |
| `convert_capture_to_pt.py` (상위 폴더) | Isaac | `.npz` → PACE `.pt` (ZOH 정렬 + 품질·공진 판정) |
| `collect_chirp_sim_go2.py` (상위 폴더) | Isaac | 합성 chirp 생성(GT 주입) — 파이프라인 검증용 |
| `fit_go2.py` (상위 폴더) | Isaac | 다중 시퀀스 CMA-ES 적합 |
| `validate_go2.py` (상위 폴더) | Isaac | hold-out 검증 (식별값 vs nominal) |
| `sim_runner_tuner_go2.py` (상위 폴더) | Isaac | tuner: sysid env(num_envs=1) chirp/replay 재생 + 실시간 물성 write |
| `tuner_gui.py` | (UDP) | tuner: PyQt5 물성 슬라이더 6개 → UDP:9875 |
| `tuner_monitor.py` | (UDP) | tuner: telem(:9876) → q 오버레이 + 롤링 RMSE (별도 프로세스) |

**런처**

| 파일 | 역할 |
|---|---|
| `r2s_commands.sh` | 별칭 등록 (`r2s_sim`/`r2s_gui`/… + `r2s_gain`/`r2s_chirp`/`r2s_fit`/…) |
| `run_sim_runner.sh` / `run_sim_bridge.sh` / `run_gui_controller.sh` / `run_all.sh` | 실시간 구동 |
| `run_real_py.sh` | 실기 ROS2 공통 런처(conda off + humble + cyclonedds + iface) |
| `run_gain_check.sh` / `run_chirp_collector.sh` | 실기 PACE 도구 |
| `run_pace.sh` | Isaac PACE 파이프라인 (`collect`/`fit`/`validate`/`convert`/`tb`) |
| `run_tuner_sim.sh` / `run_tuner_gui.sh` | tuner 모드 (§10.6) |
| `robot_env.sh` / `example_env.sh` | ROS2/로봇 env 소싱 헬퍼 |
