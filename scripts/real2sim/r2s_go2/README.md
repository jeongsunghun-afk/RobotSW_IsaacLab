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
| **Policy** | `go2_imitation_tracking` 보행 정책 구동(아래 §11) |
| **Recovery** | `go2_recovery_flip_vel` 기립 정책 구동 — **넘어진 상태 전용**(아래 §12) |
| **Pedipulation** | `go2_pedipulation` 정책 구동 — 세 다리로 균형을 잡고 한 발을 목표로. **4족으로 선 상태 전용**(아래 §14) |
| **Sim → Camera** | Isaac 뷰포트 카메라 `Follow robot` / `Free` 전환(아래 §13.4) |
| **Sim → Plant** | 관절 물성 `set2`(학습 플랜트) / `set3`(재캡처, 미채택) / `Default`(nominal) 전환(아래 §13) |
| **Monitor** | 실시간 plot 창 열기(아래 §5) |

모든 명령은 `kp=25, kd=0.5`(전 관절), `mode=0x01`, `dq=0`, `tau=0`으로 50Hz 연속 발행된다
(실로봇 watchdog heartbeat + CRC/헤더 포함, `CONTRACT.md` §6·§8).
**단 하나의 예외가 Recovery 모드로, `kd=1.0`을 쓴다** — 그 정책의 학습 액추에이터 damping이
1.0이기 때문이다(§12.3).

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
| `R2S_DEPLOYABLE_POLICY` | obs42 런의 `exported/deployable_policy.pt` | Policy 모드가 로드할 jit 경로(§11) |
| `R2S_POLICY_ACTION_CLIP` | 학습 `agent.yaml` 의 `clip_actions` | 보통 **불필요** — 정책 run 의 cfg 에서 자동으로 읽는다(§11.5). 주면 cfg 를 이긴다 |
| `R2S_POLICY_HIP_SCALE` | 학습 `env.yaml` 의 `hip_scale_reduction` (true→`0.5`, 없거나 false→`1.0`) | 위와 같음. 주면 cfg 를 이긴다 |
| `R2S_RECOVERY_POLICY` | floor03_ent005 런의 `exported/recovery_policy.pt` | Recovery 모드가 로드할 jit 경로(§12) |
| `--camera follow\|free` | `follow` | sim_runner 기동 시 카메라 모드(§13.4). GUI 로 언제든 바꿀 수 있다 |
| `--plant default\|set2\|set3` | env cfg(`use_pace_params`=True → `set2`) | sim_runner 기동 시 관절 물성 프리셋(§13). GUI 로 언제든 바꿀 수 있다. `nominal`/`pace` 는 `default`/`set2` 의 구 별칭 |
| `--ctrl_port` | 9877 | GUI → sim_runner 제어 채널 포트 |
| `R2S_SIM_HOST` | `127.0.0.1` | GUI 가 제어 패킷을 보낼 sim_runner 호스트 |
| `R2S_CTRL_PORT` | 9877 | 위와 짝. sim_runner 의 `--ctrl_port` 와 맞출 것 |

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

### 10.3 실기 순서 (전체 런북)

> 전제: GO2 ↔ 서버 이더넷, 서버 static IP `192.168.123.99/24`, 리모컨 **L2+A → L2+B** 로
> sport 서비스 해제. 셸에 `source .../r2s_commands.sh` 로 `r2s_*` 함수 등록.

| # | 명령 | 로봇 | 무엇을 하나 |
|---|---|---|---|
| 0 | `bash check_go2_comms.sh` | 정지 | ROS2/DDS 디스커버리 점검 (ros2 daemon 재시작 포함) |
| 1 | `r2s_gain --selftest` | 불필요 | 기울기 추정기를 합성 데이터로 검증 |
| 2 | `r2s_chirp --dry-run` | 정지 | 궤적 생성 + soft limit 검사만. **발행 없음** |
| 3 | (물리) 로봇을 매단다 | — | 발이 지면·장애물에 안 닿게 |
| 4 | `r2s_gain --suspended --joint {0,1,2} --kp 25` | 매달림 | **kp 규약 α 측정** (관절 3개 반복) |
| 5 | `r2s_chirp --suspended --amplitude_scale 0.3 --out sweep_check.npz` | 매달림 | **리그 공진 스윕** (저진폭) |
| 6 | `r2s_convert --capture data/go2_real/sweep_check.npz` | — | 공진 첫 주파수 판정 |
| 7 | `r2s_chirp --suspended --kp {15,25,35} --kd 0.5 --out chirp_kp*.npz` | 매달림 | **본 수집** (게인 3종) |
| 8 | `r2s_convert --capture data/go2_real/chirp_kp*.npz` | — | `.npz` → `.pt` + 품질/정렬 판정 |
| 9 | `r2s_fit` → `r2s_validate` | — | CMA-ES 적합 → hold-out 검증 |

바닥에서 시작하려면 7번 첫 회를 `--interactive` 로 돌린다 — kp 램프-인 → 천천히 stand-up →
**"매달고 Enter"** 프롬프트(대기 중에도 heartbeat 유지) → chirp. 이후는 `--suspended` 로 반복.

```bash
# 0~2: 로봇 안 움직임
bash scripts/real2sim/r2s_go2/check_go2_comms.sh
r2s_gain  --selftest
r2s_chirp --dry-run
#   확인:  chirp 0.1->8.0 Hz / 진폭 테이퍼 1.5 Hz 부터 / abort ... > 1.00 rad / soft limit 통과

# 3: 로봇을 매단다

# 4: kp 규약 (관절 0=hip, 1=thigh, 2=calf)
r2s_gain --suspended --joint 0 --kp 25
r2s_gain --suspended --joint 1 --kp 25
r2s_gain --suspended --joint 2 --kp 25
#   합격: |α − 1| <= 0.1  그리고  R² >= 0.9

# 5~6: 리그 공진
r2s_chirp   --suspended --amplitude_scale 0.3 --out sweep_check.npz
r2s_convert --capture data/go2_real/sweep_check.npz
#   전 대역 |w| < 0.5 rad/s 면 8 Hz 그대로. 경고가 나오면 그 주파수 아래로 --max_frequency 를 낮춘다.

# 7: 본 수집 (공진 결과에 따라 --max_frequency 조정)
r2s_chirp --suspended --kp 15 --kd 0.5 --out chirp_kp15.npz
r2s_chirp --suspended --kp 25 --kd 0.5 --out chirp_kp25.npz
r2s_chirp --suspended --kp 35 --kd 0.5 --out chirp_kp35.npz
#   ★ 매 회 저장 직후 "상태 스트림 손실률" 과 "명령/상태 정렬 lag" 두 줄을 반드시 볼 것

# 8~9
r2s_convert --capture data/go2_real/chirp_kp15.npz
r2s_convert --capture data/go2_real/chirp_kp25.npz
r2s_convert --capture data/go2_real/chirp_kp35.npz
r2s_fit ; r2s_validate
```

**적합 전에 `r2s_go2_sysid_cfg.py` 의 `datasets`/`holdout` 을 새 캡처 경로로 바꿀 것.**
배포 게인(25)은 반드시 `datasets` 에 넣는다 — 권장 배분은 `datasets={15,25}`, `holdout={35}`.

#### 세 가지 사전 검사가 각각 막는 것

셋 다 **"적합은 잘 되는데 값이 틀린"** 결함이라 hold-out RMSE 로는 **하나도 못 잡는다.**

| 검사 | 놓치면 생기는 일 |
|---|---|
| 4. kp 규약 (`r2s_gain`) | 실토크가 `α·kp·e` 인데 `kp` 로 믿으면 armature/마찰이 **1/α 배 편향**. 위치 궤적은 완벽히 재현된다 |
| 5. 리그 공진 (`--amplitude_scale 0.3`) | 관절이 아니라 **거치대를 잰다** |
| 7. 정렬 lag (저장 시 자동) | 시간축 어긋남을 armature/viscous 가 흡수. 2026-07-15 캡처가 104 ms 로 이렇게 망가졌다 |

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

**학습 정책 구동 (§11 Policy / §12 Recovery)**

| 파일 | 역할 |
|---|---|
| `policy_runtime.py` | tracking obs(42) 조립 + 관절 순서 변환 + jit 래퍼 (Go2 공통 상수의 source of truth) |
| `recovery_runtime.py` | recovery obs(42) 조립 + `previous_actions` 버퍼 + jit 래퍼 (위 모듈의 상수 재사용) |
| `export_deployable_go2.py` (상위 폴더) | tracking: actor+estimator+history_encoder → 단일 jit |
| `export_recovery_go2.py` (상위 폴더) | recovery: rsl_rl 표준 export → `recovery_policy.pt` |
| `verify_recovery_deploy_go2.py` (상위 폴더) | recovery 배포 경로 정합성 검증 (§12.5) |
| `run_export_policy.sh` / `run_export_recovery.sh` | 위 두 export 의 원커맨드 래퍼 |

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

---

## 11. Policy 모드 — 학습 정책을 sim/실기에서 구동

GUI의 Policy 모드는 `go2_imitation_tracking` 학습 정책을 **단일 torch.jit**(actor +
estimator + history_encoder)로 묶은 `deployable_policy.pt`를 50 Hz로 돌린다.

### 11.1 체크포인트 → deployable jit

```bash
conda activate isaac-6.0
python scripts/real2sim/export_deployable_go2.py \
  --run_dir logs/rsl_rl/go2_imitation_tracking/<RUN> \
  --checkpoint model_<N>.pt --device cpu
# → <run_dir>/exported/deployable_policy.pt
```

`--device cpu`로 뽑아도 된다(체크포인트가 cuda 텐서여도 `map_location`으로 처리). GPU 없는
배포 머신을 상정한 것이다. export는 세 가지를 자체 검증한다 — wrapper가 `act_inference`와
일치하는지, jit 재로드가 일치하는지, 그리고 estimator가 실제로 빌드·복원됐는지.

**학습이 끝나면 최종 체크포인트로 다시 뽑아야 한다.** 현재 기본 경로가 가리키는 것은
학습 도중 체크포인트다.

### 11.2 ⚠ obs 레이아웃이 맞아야 한다

`policy_runtime.py`는 **obs 42 / priv_explicit 6** 레이아웃 전용이다(`POLICY_DIM = 42`).

| 런 | `observation_space` | `num_priv_explicit` | 이 런타임과 |
|---|---|---|---|
| `2026-07-30_10-35-28_obs42_novideo` 이후 | 42 | 6 | **호환** |
| `angvel_priv_s025` 계열 | 45 | 6 | 불가 |
| `2026-07-24_13-03-00` 등 초기 | 45 | 3 | 불가 |

다른 런을 쓰려면 그 런의 `params/env.yaml`에서 `observation_space`를 먼저 확인할 것.
맞지 않으면 `PolicyModel` 생성 시 어떤 레이아웃을 기대했는지 밝히며 즉시 실패한다(조용히
잘못 동작하지 않는다).

### 11.3 proprio 규약 (`policy_runtime.build_proprio`)

```
projected_gravity_b(3) + lin_vel_cmd(2) + yaw_vel_cmd(1)
  + (joint_pos − default)(12) + joint_vel(12) + actions(12, 항상 0)
```

- **자이로(각속도)를 넣지 않는다.** 학습 env가 각속도를 `priv_explicit`으로 옮겨 policy obs
  에서 뺐으므로 정책은 이 신호를 입력으로 본 적이 없다. estimator가 proprio에서 추정한다.
- **마지막 12칸(actions)은 항상 0이다.** 학습 env가 obs에 넣는 `self.actions`가 `DirectRLEnv.
  __init__`에서 0으로 초기화된 뒤 갱신되지 않기 때문이다(실제 action은 `self._actions`에
  들어간다). 배포에서 0을 넣는 것이 학습과 일치하는 동작이다.
- history는 10칸 oldest-first, index −1이 최신. 첫 진입 시 10칸을 현재 proprio로 채운다
  (env의 `episode_length_buf <= 1` 분기와 동일).
- 스케일링은 하지 않는다 — 정규화는 jit 내부 `actor_obs_normalizer`가 처리한다.

### 11.4 알려진 거동

램프 측정(`reports/rsl_rl/go2_imitation_tracking/2026-07-30_10-35-28_obs42_novideo/`)에서 확인된 것:

- 속도 천장 **~1.4 m/s** (명령 범위는 0~4 m/s).
- **정지 상태에서 재출발하지 못한다.** 한 번 서면 명령을 올려도 웅크린 채(`base_h` 0.34 → 0.22)
  머문다. 실기 투입 시 **정지에서 시작하지 말고** 낮은 속도로 이미 움직이는 상태에서
  정책으로 전환할 것.

  > **2026-08-01 갱신 — finetune(`2026-07-31_12-27-01_reststand_ft`, `model_59999`)으로 크게
  > 나아졌지만 이 지시는 그대로다.** 서 있는 로봇에서 engage 하는 시나리오(= GUI Policy Start)를
  > 64 env 로 재니 재출발이 **0/64 → 18/64 (28.1%)** 다. 개선은 확실하지만(z=4.57) 72% 는 여전히
  > 웅크린 채 그대로다. 달리다 멈춘 뒤 재가속하는 경우는 75.9% 로 잘 되므로, **운동량이 있는
  > 상태에서 넘기라**는 지시가 정확히 맞는 처방이다.
  > 근거: `reports/rsl_rl/go2_imitation_tracking/2026-07-31_12-27-01_reststand_ft/` "★ 정지 스폰".
- 정지 자세를 넣으면 action이 clip(±4)을 크게 넘는다(−20.9까지) — 위와 같은 원인이다.

> 이 제약은 **정책 자체의 성질**이지 배포 경로(평균 행동 / history_latent) 탓이 아니다.
> 행동 생성 2×2(평균·샘플 × history_latent·priv_latent)를 모두 시험했고, 학습이 실제로
> 굴리는 조합조차 64개 중 4개만 재출발했다. 원인 분석은
> `reports/rsl_rl/go2_imitation_tracking/2026-07-30_10-35-28_obs42_novideo/README.md`
> "읽히는 것 4 → 원인 규명".

### 11.5 action clip / hip 축소는 정책의 학습 cfg 에서 읽는다

`clip(±clip_actions)` → hip 만 `×0.5` → `×action_scale` → default 더하기 순서는 학습 env
(`go2_imitation_tracking_env.py:242-244`)와 같아야 한다. 앞의 두 값은 **정책마다 다르므로**
`policy_runtime.configure_from_policy()` 가 jit 옆의 학습 cfg 에서 읽는다
(`PolicyModel` 로드 시 자동 호출):

| 값 | 출처 | 규칙 |
|---|---|---|
| `ACTION_CLIP` | `<run>/params/agent.yaml` 의 `clip_actions` | 그대로 |
| `HIP_ACTION_SCALE` | `<run>/params/env.yaml` 의 `hip_scale_reduction` | `true` → **0.5**, `false` → 1.0, **키 없음 → 1.0** |

run 디렉터리는 jit 경로에서 유도한다 (`run_export_policy.sh` 가
`<run>/exported/deployable_policy.pt` 로 내보내므로 두 단계 위).

**키가 없으면 1.0 이다.** 그 run 은 이 기능이 생기기 전에 학습된 것이므로 축소가 없었다.
"모름"으로 보고 0.5 를 씌우면 구 정책의 hip 이 절반으로 눌린다.

cfg 를 못 찾으면 **로드를 막는다.** GUI 는 이를 흡수해 fallback-hold 로 떨어지고 경고에 이유가
찍힌다. 어떤 값이 맞는지 모르는 채 실기에 각도를 내보내지 않기 위해서다. run 디렉터리 없이
jit 만 가진 정책이라면 환경변수로 명시한다 (cfg 보다 우선):

```bash
R2S_POLICY_ACTION_CLIP=4.0 R2S_POLICY_HIP_SCALE=1.0 r2s_gui
```

> **2026-08-05 이전에는 상수 (10.0, 0.5) 가 박혀 있었다.** 그런데 기본
> `R2S_DEPLOYABLE_POLICY` 인 obs42 런은 `clip_actions=4.0` · `hip_scale_reduction` 키 없음
> 이라, 환경변수를 손으로 주지 않으면 **hip 이 절반으로 눌리고 액션 범위가 2.5배로 늘어난**
> 상태로 돌고 있었다. 지금은 자동으로 (4.0, 1.0) 로 맞는다.

---

## 12. Recovery 모드 — 넘어진 상태에서 기립

GUI의 Recovery 모드는 `go2_recovery_flip_vel` 학습 정책(`Go2Recovery-FlipVel-v0`)을 jit로 뽑아
50 Hz로 돌린다. Policy 모드와 완전히 별개의 정책이며 **명령(command)이 없다** — 목표가
"default 자세로 일어서라" 하나로 고정이라 슬라이더가 없고 입력 소스(Sim/Real)만 고른다.

### 12.1 체크포인트 → jit

```bash
bash scripts/real2sim/r2s_go2/run_export_recovery.sh          # 최신 ckpt 자동 선택
# 또는
conda activate isaac-6.0
python scripts/real2sim/export_recovery_go2.py \
  --run_dir logs/rsl_rl/go2_recovery_flip_vel/<RUN> \
  --checkpoint model_<N>.pt --device cpu
# → <run_dir>/exported/recovery_policy.pt
```

기본 런은 `2026-07-14_09-54-05_floor03_ent005`(`model_9999.pt`)다. 이 task 에서 가장 좋은 런으로,
`success_rate` 0.679 / **넘어진 env 기준 0.958**이다. 다른 파일을 쓰려면 `R2S_RECOVERY_POLICY`로
덮어쓴다.

정책이 estimator/history 없는 plain PPO라 rsl_rl 표준 export(`export_policy_to_jit`)를 그대로
쓴다. **직접 `actor(normalizer(obs))`로 wrapper를 짜면 안 된다** — rsl_rl 5.0.1의 `MLPModel`은
MLP 출력이 분포 파라미터라 결정론적 행동을 얻으려면 `distribution.deterministic_output`을 거쳐야
하고, 표준 export가 그 순서를 이미 담고 있다.

### 12.2 ⚠ tracking jit 과 바꿔 끼우면 안 된다

두 정책 모두 obs가 **42-dim이지만 레이아웃이 전혀 다르다.**

| | tracking (`deployable_policy.pt`) | recovery (`recovery_policy.pt`) |
|---|---|---|
| 각속도 | 없음(priv_explicit로 이동) | **있음**, ×0.25 |
| 명령 | `lin_vel(2) + yaw_vel(1)` | 없음 |
| `joint_vel` | raw | **×0.05** |
| 마지막 12칸 | actions, **항상 0** (dead channel) | **live** `previous_actions` |
| 입력 개수 | 2 (`proprio`, `history`) | 1 (`obs`) |

shape 검사로는 구분되지 않으므로 `RecoveryModel._warmup`이 **입력 인자 개수**로 잘못된 파일을
막는다(tracking jit을 넣으면 로드 시점에 무엇이 잘못됐는지 밝히며 실패한다).

### 12.3 obs 규약 (`recovery_runtime.build_obs`)

```
root_ang_vel_b × 0.25 (3) + projected_gravity_b (3)
  + (joint_pos − default) × 1.0 (12) + joint_vel × 0.05 (12) + previous_actions (12)
```

- **`previous_actions`는 clip(±100) 직후 / hip 0.5× 적용 전 값이다** (`go2_recovery_env.py`의
  `self._actions = clipped`). 관절 목표가 아니다. 모드 진입 시 학습 reset과 같이 0에서 시작한다.
- action → 목표: `clip(±100)` → **hip 관절만 ×0.5** → `×0.25 + default_joint_pos`.
- **kp=25, kd=1.0으로 발행한다.** 학습 액추에이터가 `stiffness 25 / damping 1.0`이다
  (tracking과 GUI 기본값은 damping 0.5). Recovery Stop은 이 kd를 **유지한 채** 홀드한다 —
  기립 직후 게인을 절반으로 떨어뜨리면 그대로 주저앉기 때문이다. 이후 Default/Stand Up/Policy
  중 아무 버튼이나 누르면 기본 0.5로 돌아간다.

### 12.4 Start가 즉시 engage 하는 이유

Policy Start는 안전을 위해 `DEFAULT_POSE`로 먼저 보간한 뒤 engage하지만, Recovery Start는
**현재 자세에서 곧바로 engage**한다. 로봇이 바닥에 있는 상태에서 default로 끌고 가는 것은
위험할 뿐 아니라 학습 분포에서도 벗어난다: 학습 env의 settle 구간은 측정 자세를 그대로
유지하고(`settle_mode="passive"`), 그 길이가 env마다 `U[0, 100]` step이라 **0 step**(첫 tick부터
정책이 구동)도 학습에 포함돼 있다.

### 12.5 검증된 것 / 안 된 것

**검증됨** (`scripts/real2sim/verify_recovery_deploy_go2.py`, 40 step × 4 env):

- articulation 관절 순서가 `policy_runtime.ART_ORDER`와 일치(env에서 직접 대조).
- 배포 경로가 조립한 obs가 학습 env의 obs와 일치 — 최대 오차 `6.6e-07`.
- 배포 경로의 action→목표 변환이 env `_processed_actions`와 일치 — 최대 오차 `1.2e-07`.
- 두 변환 모두 **DDS 순서 왕복을 거친 뒤** 대조했다(순서 오류가 상쇄되지 않게).

```bash
conda activate isaac-6.0
CUDA_VISIBLE_DEVICES=1 env -u DISPLAY python scripts/real2sim/verify_recovery_deploy_go2.py \
  --checkpoint logs/rsl_rl/go2_recovery_flip_vel/2026-07-14_09-54-05_floor03_ent005/exported/recovery_policy.pt
```

**r2s sim end-to-end 도 검증됨** (`verify_policy_deploy_go2.py --recovery`, `--flip` 포함).
GUI Recovery Start 와 같이 현재 자세에서 즉시 engage 하고, PACE 물성(§13) ON/OFF 양쪽으로 잰다:

| 시나리오 | 플랜트 | base_h 최종 | upright | jvel p99 | 판정 |
|---|---|---|---|---|---|
| 엎드림(prone spawn) | PACE ON | 0.323 | 0.998 | 2.089 | 기립 성공 |
| 엎드림 | PACE OFF | 0.316 | 0.997 | 2.453 | 기립 성공 |
| **belly-up (roll 180°)** | **PACE ON** | **0.318** | **1.000** | **3.824** | **기립 성공** |
| belly-up | PACE OFF | 0.317 | 0.999 | 8.115 | 기립 성공 |

`use_pace_params=True`(현재 기본값)가 recovery 를 막지 않는다. PACE viscous 의 관절속도 천장은
`23.5(1−q̇/30) = 2.41·q̇` → **7.36 rad/s** 인데, 이 정책은 flip 복구에서도 p99 3.8 rad/s 밖에 안
쓴다 — FlipVel 학습의 `r_joint_vel` 페널티가 whip(8~22 rad/s)을 이미 제거했기 때문이다.
PACE OFF 는 8.1 rad/s 까지 쓰므로, **PACE ON 이 오히려 더 부드럽게 복구한다**(RMS 0.786 vs 1.475).

```bash
conda activate isaac-6.0
CUDA_VISIBLE_DEVICES=1 env -u DISPLAY python scripts/real2sim/verify_policy_deploy_go2.py \
  --recovery --flip --checkpoint <...>/exported/recovery_policy.pt --hold_s 12 [--no_pace]
```

**아직 안 된 것 — 실기 투입 전에 반드시 볼 것:**

- **실기 미수행.** 위는 전부 sim 이다.
- **GUI 3-프로세스 실기동 미수행.** 검증 도구는 ROS/UDP 를 우회한다.
- **`r2s_go2_env.set_setpoint` 는 LowCmd 의 kp/kd 를 저장만 하고 적용하지 않는다**(cfg 액추에이터
  PD 고정). 즉 §12.3 의 kd=1.0 은 **sim 에서는 무효고 실기에서만 효과가 있다.**
- **서 있는 상태에서 누르면 어떻게 되는지 모른다.** 이 런의 `success_rate_non_fallen`은
  9999 iter 시점에도 **0.000**이다(넘어진 env는 0.958). GUI 힌트 라벨에도 적어두었지만,
  넘어진 상태 전용으로 쓸 것.

---

## 13. r2s sim 관절 물성 (PACE) — `use_pace_params`

`Isaac-R2S-Go2-v0` 는 **PACE 식별 물성으로 돈다** (`R2SGo2EnvCfg.use_pace_params`, 기본 `True`
= `set2`). GUI 의 `Plant:` 콤보 또는 `--plant` 로 **세 프리셋을 런타임에 전환**할 수 있다 —
실기 GO2 와 sim 의 명령 추종 정도를 눈으로 비교하기 위한 것이다.

| 프리셋 | armature [kg·m²] | viscous [N·m·s/rad] | coulomb [N·m] | 비고 |
|---|---|---|---|---|
| **set2** (기본) | 0.00101 / 0.00011 / 0.01610 | 0.0019 / 0.0162 / 0.0015 | 0.198 / 0.151 / 0.670 | **현재 정책이 학습된 플랜트** |
| **set3** | 0.00758 / 0.00531 / 0.02106 | 0.181 / 0.144 / 0.127 | 0.145 / 0.100 / 0.577 | 2026-08-04 재캡처. **미채택** — viscous 기각 |
| **default** | 0.01 / 0.01 / 0.01 | 0 | 0 | `UNITREE_GO2_CFG` nominal |

actuator kp / kd 는 25 / 0.5 고정 — **바꾸지 말 것.** viscous 가 kd 오차를 흡수하도록 함께
식별된 조합이라 게인만 따로 바꾸면 식별 결과가 깨진다.

> **⚠ 프리셋을 섞지 말 것.** armature/viscous/coulomb 은 **결합 식별**이라 set3 의 coulomb 과
> set2 의 viscous 를 합치면 어떤 적합도 산출하지 않은 플랜트가 된다.
>
> **set3 는 왜 미채택인가**: coulomb(독립 회귀 대비 1.10~1.45x)과 calf armature(1.4x)는 검사를
> 통과했지만 viscous 가 회귀 대비 **3.5~10.8x**, 에너지 수지 대비 **1.6~3.0x** 로 기각됐다.
> 리그 오염과는 무상관(corr −0.21~+0.09)이라 원인 미상이다.
> 상세: `reports/_comparisons/pace_go2_sysid_excitation_audit/`.
>
> **§11 Policy 모드는 set2 에서만 정상 동작한다** — default 로 두면 걷지 못한다(아래 A/B).
> set3 는 감쇠가 set2 대비 +25~36% 라 거동이 달라진다. 그 차이를 보는 것이 이 콤보의 목적이다.

**2026-07-31 이전에는 nominal(`UNITREE_GO2_CFG`: armature 0.01, 마찰 0)로 돌았다.** 학습 env 와
플랜트가 전혀 달라 §11 Policy 모드가 관절 초당 ~22회 진동으로 걷지 못했다. 원인 규명과 A/B 는
`reports/rsl_rl/go2_imitation_tracking/_comparisons/r2s_sim_plant_gap/`.

수정 후 (배포 경로 end-to-end, `cmd 0`, engage 1s 이후):

| | jvel RMS | 부호반전/step | base_h |
|---|---|---|---|
| PACE ON | **0.246** | **0.0010** | 0.228 m (기립 유지) |
| PACE OFF | 9.406 | 0.3820 | 0.110 m (주저앉음) |

### 13.1 함께 고친 것 — reset 이 로봇을 초기자세로 쓰지 않았다

`R2SGo2Env._reset_idx` 가 articulation 상태를 write 하지 않아, `env.reset()` 직후 로봇이 USD
rest(관절 ≈ 0, 다리 뻗은 자세)에 남고 prone setpoint 과 **2.57 rad** 어긋난 채 첫 스텝에 16~23 N·m
킥이 걸렸다. nominal 에서는 흐물흐물 접혀 무해했지만 PACE 에서는 기체가 base_h 0.42 까지 튀어올라
넘어졌다. `default_root_state`/`default_joint_pos` 를 write 하도록 고쳤다(오차 0.0000, 첫 스텝
토크 1.58 N·m).

**첫 reset 에서만** 쓴다 — 이 env 는 RL 이 아니라 라이브 제어 harness 인데 `episode_length_s=600`
때문에 10분마다 time_out reset 이 걸린다. 매번 write 하면 조작 중인 로봇이 10분마다 spawn 자세로
순간이동한다.

### 13.2 ⚠ sysid 모드는 반드시 OFF

`R2SGo2SysidEnvCfg.use_pace_params = False` 다. sysid 는 이 물성을 **찾는** 쪽이라, 켜두면 CMA-ES 가
이미 PACE 값이 들어간 플랜트 위에서 적합을 시작해 이중 적용되고 식별이 조용히 망가진다.
`_reset_idx` 의 초기자세 write 도 sysid 에서는 건너뛴다(PACE 적합 harness 가 상태를 직접 관리하고,
이미 hold-out RMSE 0.017 rad 로 검증된 거동이다). tuner 모드도 같은 sysid task 를 재사용하므로 함께
안전하다.

### 13.3 런타임 전환 — GUI **Sim** 그룹

GUI 의 **Sim** 그룹에서 `Plant` 를 바꾸면 sim 을 재시작하지 않고 물성이 갈아끼워진다.
`R2SGo2Env.set_joint_plant()` 이 armature/viscous/coulomb 세 값을 한 번에 쓰므로 두 프리셋을
오가도 상태가 섞이지 않는다(왕복 2회 후에도 값 동일함을 확인).

기동 시 기본값은 `sim_runner_go2.py --plant pace|nominal` 로도 줄 수 있다.

```
GUI Sim 콤보 ─UDP:9877─▶ sim_runner_go2 ─▶ env.set_joint_plant() / set_camera_follow()
```

명령(9871)과 분리한 채널이다 — 50 Hz `/lowcmd` 스트림은 실기 watchdog 이 걸린 실시간 경로라
필드를 늘리면 안 되고, 이쪽은 버튼을 누를 때만 나가는 one-shot 이다. `sim_bridge` 를 거치지 않고
GUI 가 직접 보낸다(`tuner_gui` → 9875 와 같은 방식). sim 이 안 떠 있으면 조용히 무시된다.

### 13.4 카메라 — Follow / Free

기본은 **로봇 추적**이다. `R2SGo2EnvCfg.viewer` 가 `origin_type="asset_root"`, `asset_name="robot"`
이라 IsaacLab 이 매 렌더 스텝마다 eye/lookat 을 로봇 base 기준 상대좌표로 다시 잡아준다
(`eye=(-2.2, -1.6, 0.9)`, `lookat=(0, 0, 0.15)` — 뒤 왼쪽 위에서 내려다봄).

GUI **Sim → Camera** 로 `Free` 를 고르면 추적만 멈추고 **카메라는 보던 자리에 그대로 있다**.
`update_view_to_world()` 를 부르지 않는 이유가 이것이다 — 그러면 카메라가 cfg 기본 위치로 튀어서,
각도를 유지한 채 조작을 넘겨받고 싶은 의도와 어긋난다. `Follow` 로 되돌리면 다시 붙는다.

검증(뷰포트 있는 실행, 로봇을 +3 m 이동):

| | 결과 |
|---|---|
| Follow | base Δx **+2.989** vs cam Δx **+2.990** — 따라감 |
| Free 전환 | 카메라 위치 변화 **0.000** — 스냅 없음 |
| Free 중 로봇 이동 | cam Δx **+0.000** — 멈춰 있음 |
| Follow 복귀 | cam − base = **−2.200** = cfg `eye` x 그대로 |

⚠ **headless 에서는 뷰포트 자체가 없어 전부 no-op 이다**(`viewport_camera_controller` 가 None).
`r2s_sim` 은 `--viz kit` + livestream 으로 뜨므로 컨트롤러가 생긴다 — `--viz` 를 끄면 카메라 기능도
같이 사라진다.

> `R2SGo2Env.close()` 가 추적을 먼저 끈다. `DirectRLEnv.close()` 는 `del self.scene` 을 뷰포트
> 컨트롤러 삭제보다 **먼저** 해서, 그 사이 콜백이 한 번 더 발화하면 AttributeError 를 뱉는다
> (추적을 안 쓰던 시절엔 콜백이 즉시 반환해 드러나지 않던 순서 문제).

### 13.5 되돌리기

`use_pace_params=False`(또는 GUI Plant 콤보에서 `Default — nominal`, 또는 `--plant default`)로
두면 이전 nominal 거동으로 정확히 돌아간다(PACE 이전에 수집한 데이터를 재현할 때).

---

## 14. Pedipulation 모드 — 3족 균형 + 발 1개 조작

`go2_pedipulation` 정책(`Go2-Pedipulation-v0`)을 GUI에서 구동한다. 세 다리로 균형을 잡고
남은 한 다리의 발을 명령받은 위치로 옮겨 **유지**한다. 학습·게이트 근거는
`reports/rsl_rl/go2_pedipulation/README.md`.

### 14.1 준비 — 정책 export

```bash
./isaaclab.sh -p scripts/real2sim/export_pedipulation_go2.py \
    --run_dir logs/rsl_rl/go2_pedipulation/2026-08-03_15-53-29_hipscale_scratch_s10x \
    --checkpoint model_19999.pt --device cpu
# → <run_dir>/exported/pedipulation_policy.pt
```

기본 경로가 위 채택본이라 다른 체크포인트를 쓸 때만 `R2S_PEDIPULATION_POLICY`로 덮어쓴다.

### 14.2 사용

sim 3프로세스를 §3 순서대로 띄운다. ⚠ **sim 전용 브릿지는 `/lowstate`로 발행하는데 GUI의
Sim 소스 기본값은 `/sim/lowstate`다**(그건 §6 동시 구동에서 remap 하는 이름). 맞춰주지 않으면
Source=Sim이 상태를 못 받아 `valid`가 서지 않고 **에러 없이 fallback-hold**에 머문다 —
로봇이 가만히 있는 것으로만 보여 원인을 찾기 어렵다.

```bash
# 터미널 1 — Isaac (학습 플랜트로)
./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py --num_envs 1 --plant set2

# 터미널 2 — 브릿지
bash scripts/real2sim/r2s_go2/run_sim_bridge.sh

# 터미널 3 — GUI (sim 전용이면 상태 토픽을 맞춰준다)
R2S_SIM_STATE_TOPIC=/lowstate bash scripts/real2sim/r2s_go2/run_gui_controller.sh
```

`Pedipulation` 그룹에서 **Source**(Sim/Real) · **Leg**(조작할 다리) · **dx/dy/dz**(nominal 발
위치 대비 목표 offset [m])를 고르고 **Start**. 실행 중에도 Leg·offset을 바꾸면 즉시 반영된다
(정책이 에피소드 중 명령 변화를 겪도록 학습됐다). **Stop**은 현재 목표 자세로 홀드한다.

offset 스핀박스의 범위는 **학습 명령 박스 그대로**다(x ±0.20, y ±0.14, z 0~0.26 m). 이 밖은
분포 밖이고 기구학적으로 도달 못 하는 목표라 UI가 애초에 막는다.

### 14.3 ⚠ 4족으로 선 상태에서 시작할 것

명령은 절대 좌표가 아니라 **nominal 발 위치 대비 offset**이고, 그 nominal을 **Start 시점의
실제 자세에서 FK로 잡는다**(`PedipulationState.try_latch_nominal`). 그래서:

- 발 깊이가 0.20~0.40 m 범위여야 latch 된다. 넘어져 있거나 다리를 든 상태면 latch 되지 않고
  정책을 구동하지 않는다(fallback-hold 유지) — 잘못된 기준으로 시작하는 사고를 막는 게이트다.
- Recovery처럼 즉시 engage 한다. Policy 모드처럼 `DEFAULT_POSE`로 먼저 보간하지 **않는** 이유가
  이것이다 — 보간 도중에 nominal을 잡으면 엉뚱한 자세가 명령 기준이 된다.

nominal은 상수가 아니다. env도 매 실행 측정해 latch 하며 그 값이 실행마다 1~3 cm 다르다
(default 관절각의 FK 값과도 17~33 mm 어긋난다). 그래서 런타임도 상수를 박지 않고 측정한다.

### 14.4 게인·플랜트

kp/kd는 **기본값(25/0.5) 그대로**다 — 이 정책의 학습 액추에이터가 stiffness 25 / damping 0.5로
`motions.DEFAULT_KP/KD`와 같다. Recovery처럼 따로 실을 이유가 없다.

⚠ sim에서 검증할 때 **Plant를 `set2`(학습 플랜트)로 둘 것**. 이 정책은 `use_pace_params=True`로
학습됐다(§13).

### 14.5 앞의 두 정책과 다른 점

| | Policy(tracking) | Recovery | **Pedipulation** |
|---|---|---|---|
| obs | 42 + history | 42 | **83** |
| action | 12 | 12 | **28** (a_loc 12 + a_man 12 + 강성 4 미사용) |
| 명령 | 속도(x, yaw) | 없음 | **조작 다리 + 발 목표 offset** |
| 상태 유지 | history 링버퍼 | prev_actions | **prev_actions + 적분형 목표 + nominal latch** |

**적분형 목표가 이 모드의 핵심 차이다.** 조작 다리는 절대 각도가 아니라
`man_target += clamp(a_man × 0.25, ±0.15)`로 직전 목표에 누적된다. 그래서 런타임이 무상태
함수일 수 없고, 모드를 벗어난 tick마다 상태를 통째로 리셋한다(안 하면 재진입 첫 tick에 목표가
튄다). hip 축소(×0.5)도 **지지 다리 슬롯에만** 걸린다 — 적분형에 같은 계수를 곱하면 "범위 축소"가
아니라 "이동 속도 감쇠"가 되고, 조작 다리의 abduction 권한은 이 task의 목적 자체다.

obs 83 / action 28이라 다른 두 정책과 **shape만으로 구분**된다(`PedipulationModel._warmup`).
tracking과 recovery는 obs가 42로 같아 인자 개수로 갈라야 했던 것과 대비된다.

### 14.6 검증 상태

- 순기구학: 무작위 64 자세 × 4 다리에서 sim의 `_compute_foot_pos_b()`와 **최대 오차 0.0004 mm**.
- 런타임 ↔ env 대조: DR을 끈 160 step에서 **obs 최대 5.6e-7 · 관절 목표 최대 1.9e-6 rad**
  (float32 정밀도 수준). obs 배치·prev_actions 시점·적분 목표·hip 축소 범위까지 포함한 대조이며,
  **네 다리를 모두 조작 다리로 돌려가며** 쟀다 — 한 다리만 쓰면 적분 경로와 hip 축소 회피가
  나머지 세 슬롯에서 검증되지 않는다(그쪽은 쉬운 분기인 `loc_target`만 탄다).
- jit export: 저장본과 결정론적 추론 경로가 **완전히 일치**(diff 0.0).
- **sim 통합 구동 (2026-08-06)**: `sim_runner(--plant set2)` + 브릿지 + GUI 3프로세스를 실제로
  띄워 GUI에서 조작했다. 아래는 **실측 `/lowstate`의 관절각을 FK로 되돌린** 조작 발 위치이므로
  (그 FK가 위의 0.0004 mm짜리다) 명령을 되읊은 값이 아니라 실제 위치 판독이다.

  | 조작 | 조작 발 위치 (base frame) |
  |---|---|
  | Start (FL, dz=0.15) | z **−0.287 → −0.125 m** (Δ +0.162) |
  | dx 0.12 로 변경 | x **0.216 → 0.334 m** (nominal 0.207 + 0.12 = 0.327) |
  | Leg FL → RR 교체 | RR 발이 z **−0.096 m** 로 들림 |

  영상: `videos/model_19999__r2s_gui_pedipulation__20260806-103100.mp4`,
  로그: `videos/r2s_gui_pedipulation_demo.log`.
- **실기 미검증** — 위는 전부 sim이다.
