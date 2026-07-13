# R2S-HindLeg — Real2Sim 뒷다리 실험 가이드

R_Skeleton 뒷다리(고정베이스 5-DOF)를 Isaac Sim에 올리고, PyQt5 GUI로 관절을 실시간 제어하며,
Monitor 창으로 명령(action) vs 시뮬(sim) 응답을 실시간 비교하는 도구 모음 (r2s_go2 패턴 기반).

```
GUI 버튼 ─cmd UDP(9873)─▶ sim_runner_hindleg(Isaac) ─state UDP(9874)─▶ GUI
                                        GUI ─relay UDP(9875)─▶ monitor(별도 프로세스)
```

전송은 **순수 UDP**다. ROS2는 향후 사용 예정이나 메시지 형식 미정이라 지금은 UDP로 두고
[`CONTRACT.md`](../../../source/isaaclab_tasks/isaaclab_tasks/direct/r2s_hind_leg/CONTRACT.md) §3의
seam으로 남겨둔다. (go2의 ROS2 unitree 계층은 GO2 전용이라 이 커스텀 다리에 재사용 불가.)

---

## 1. 실행 환경

| 프로세스 | Python | 역할 |
|---|---|---|
| `sim_runner_hindleg.py` | Isaac conda `isaac-6.0` | Isaac에서 다리 구동, UDP 명령/상태 교환 |
| `gui_controller.py` | 시스템 `/usr/bin/python3` (PyQt5) | UDP 발행 GUI (ROS 비의존) |
| `monitor.py` | 시스템 3.10 (matplotlib) | GUI가 spawn하는 실시간 plot 창 |

---

## 2. 빠른 시작 (별칭)

```bash
source scripts/real2sim/r2s_hind_leg/r2s_commands.sh
```

| 별칭 | 터미널 | 스크립트 |
|---|---|---|
| `r2s_hl_sim` | 1 (Isaac) | `run_sim_runner.sh` — conda `isaac-6.0` + sim_runner |
| `r2s_hl_gui` | 2 (시스템 py) | `run_gui_controller.sh` — GUI |

---

## 3. 실행 순서

```bash
# 터미널 1 — Isaac sim_runner
r2s_hl_sim
#   또는 수동: conda activate isaac-6.0
#             python scripts/real2sim/sim_runner_hindleg.py --num_envs 1 [--headless]

# 터미널 2 — PyQt GUI (디스플레이 필요)
r2s_hl_gui
```

다리는 중립(0) 자세로 시작한다(sim default_joint_pos와 일치).

---

## 4. GUI 버튼

| 그룹 | 동작 |
|---|---|
| **Home (default)** | 중립(0) 자세로 보간 이동 |
| **Joint Step** | 선택 관절만 delta[rad] 스텝 (soft limit 클램프) |
| **Sine Sweep** | 선택 관절에 사인 궤적 주입(Start/Stop, soft limit 클램프) |
| **Gains (faithful PD)** | 선택 관절 kp/kd 실시간 변경 → sim drive 게인에 즉시 반영 |
| **Monitor** | action/sim 실시간 plot 창(별도 프로세스) |

명령은 관절별 `kp/kd`(기본 300/300/300/100/100, kd=5)와 함께 50Hz로 연속 발행된다.
soft limit(thigh_r ±2.826, 나머지 ±1.413 rad)을 벗어나는 명령은 자동 클램프된다.

---

## 5. Monitor — action vs sim 실시간 비교

GUI **Monitor** 버튼 → 선택 관절의 3개 plot:

- **q**: action(점선=명령) / sim(엔코더)
- **tau_est**: sim
- **dq**: sim

별도 프로세스로 렌더를 GUI의 50Hz UDP 발행에서 격리한다(CONTRACT §11). robot 시리즈는
ROS2 실로봇 연동 시 추가된다.

---

## 6. 옵션 / 환경변수

| 변수 / 플래그 | 기본 | 의미 |
|---|---|---|
| `--headless` | off | 렌더 창 없이 실행 |
| `--num_envs` | 1 | 시뮬 환경 수(1만 지원) |
| `GPU` | 2 | sim_runner CUDA 디바이스 |
| `faithful_pd` (cfg) | True | GUI kp/kd를 실제 sim 게인에 반영 |

---

## 7. 트러블슈팅

- **sim_runner가 `ModuleNotFoundError`로 죽음**: `./isaaclab.sh -p`가 base conda를 가리키는 경우.
  `conda activate isaac-6.0` 후 실행하거나 `r2s_hl_sim` 사용.
- **GUI에서 명령해도 sim이 안 움직임**: sim_runner가 떠 있는지, 포트 9873/9874가 열려 있는지 확인
  (`ss -uln | grep 987`). 다른 R2S(go2, 9871/9872)와는 포트가 다르므로 동시 실행 가능.
- **Monitor 창이 비어있음**: GUI가 sim 상태를 받아야 sim 라인이 그려진다(sim_runner 먼저 기동).
- **관절이 명령보다 덜 감**: soft limit(±81°/±162°) 클램프. GUI가 자동 클램프하므로 명령값 자체가 제한됨.

---

## 8. 파일

| 파일 | 역할 |
|---|---|
| `sim_runner_hindleg.py` (상위 폴더) | Isaac 런처 + UDP 루프 |
| `gui_controller.py` | PyQt5 GUI, UDP 발행 + monitor 중계 |
| `monitor.py` | 실시간 plot(별도 프로세스) |
| `motions.py` | 자세 상수 + 보간/스텝/사인/클램프 유틸(순수 함수) |
| `r2s_udp.py` | UDP 패킷 스키마(공유, stdlib만) — 재구현 금지 |
| `run_*.sh` / `r2s_commands.sh` | 원커맨드 런처 + 별칭 |
