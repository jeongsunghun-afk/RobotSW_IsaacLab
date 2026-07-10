# R2S-GO2 — Milestone 1

GUI 버튼 -> ROS2(`/lowcmd`) -> `sim_bridge` -> UDP -> Isaac Sim GO2 -> UDP -> `sim_bridge` -> ROS2(`/lowstate`)
한 바퀴 통신 검증. 고정 계약은
[`CONTRACT.md`](../../../source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md) 참고.

## 실행 환경 (3개 프로세스, 반드시 구분해서 실행)

| 프로세스 | Python | 원커맨드 실행 |
|---|---|---|
| sim_runner | Isaac conda 3.12 | `conda activate isaac-6.0 && python scripts/real2sim/sim_runner_go2.py [--fix_base]` |
| sim_bridge | 시스템 3.10 | `bash scripts/real2sim/r2s_go2/run_sim_bridge.sh` |
| gui_controller | 시스템 3.10 | `bash scripts/real2sim/r2s_go2/run_gui_controller.sh` |

**함정**: conda가 활성화된 셸에서 `python3`는 conda(3.12)를 가리켜 rclpy 임포트가 깨진다.
`run_sim_bridge.sh`/`run_gui_controller.sh` 런처는 **conda를 (스택 포함) 완전히 비활성화한 뒤**
humble + unitree_go overlay를 소싱하고 `/usr/bin/python3`로 실행하므로, conda가 켜진 터미널에서
그냥 실행해도 된다.

수동으로 하고 싶으면(참고용):

```bash
source /opt/ros/humble/setup.bash
source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
export ROS_DOMAIN_ID=0
/usr/bin/python3 scripts/real2sim/r2s_go2/sim_bridge.py      # 또는 gui_controller.py
```

## 실행 순서

1. **sim_runner** 기동 (Isaac 3.12, GO2 로드, UDP 9871 수신 대기).
   ```bash
   conda activate isaac-6.0
   python scripts/real2sim/sim_runner_go2.py --fix_base
   ```
2. **sim_bridge** 기동 (원커맨드, conda 자동 비활성화). `/lowcmd` 구독, `/lowstate` 발행 시작.
   ```bash
   bash scripts/real2sim/r2s_go2/run_sim_bridge.sh
   ```
3. **gui_controller** 기동 (원커맨드, 디스플레이 필요). PyQt5 창이 뜬다.
   ```bash
   bash scripts/real2sim/r2s_go2/run_gui_controller.sh
   ```
4. (선택) 별도 터미널에서 상태 확인:
   ```bash
   source /opt/ros/humble/setup.bash
   source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash
   ros2 topic echo /lowcmd
   ros2 topic echo /lowstate
   ```
5. GUI "기본자세" 버튼 -> sim GO2가 default pose로 보간 이동.
6. GUI "관절 스텝" 버튼 -> 선택 관절만 delta만큼 스텝, `/lowstate`의 q가 추종하는지 확인.
7. GUI "사인 스윕" 시작/정지 -> 선택 관절이 연속 사인 궤적을 따라가는지 확인.
8. GUI "앉기" 버튼 -> 웅크린 자세로 보간 이동.

## 파일

- `sim_bridge.py` — rclpy 노드. `/lowcmd` -> UDP(9871, sim_runner로), UDP(9872, sim_runner로부터) -> `/lowstate`(50Hz).
- `gui_controller.py` — PyQt5 + rclpy. 자세/스텝/사인/앉기 버튼, 모두 50Hz로 `/lowcmd` 발행.
- `motions.py` — 목표각 상수 및 보간/스텝/사인 유틸(순수 함수, ROS 비의존).
- `r2s_udp.py` — UDP 패킷 스키마(공유, stdlib만 사용). 재구현하지 말 것.

## 조인트 순서

Unitree 표준 idx0..11 (`CONTRACT.md` §2). `motions.JOINT_NAMES`가 동일 순서를 그대로 반영한다.

## 게인

모든 GUI 명령은 `kp=25, kd=0.5`(전 관절), `mode=0x01`, `dq=0`, `tau=0`으로 발행한다
(`CONTRACT.md` §6, `UNITREE_GO2_CFG` base_legs 액추에이터와 일치).

## 옵션

- `sim_runner_go2.py --fix_base` — 로봇 base를 공중(z=0.5)에 고정 spawn(다리만 자유). 지면 접촉/균형 없이
  관절 추종·PD 응답만 깨끗하게 보려는 real2sim 테스트에 권장.
- `--headless` — 렌더 창 없이 실행(원격/자동화). 육안 확인이 필요하면 생략.

## 트러블슈팅 (통신이 안 될 때)

- **`ros2` 노드가 서로 안 보임 / `/lowstate` 데이터 안 옴**: `sim_bridge`와 `gui_controller`(및
  `ros2 topic echo`) 터미널의 **`ROS_DOMAIN_ID`와 RMW 구현이 일치**해야 discovery된다. 셸마다
  `echo $ROS_DOMAIN_ID $RMW_IMPLEMENTATION`로 확인하고, 다르면 모든 터미널에서 동일하게 맞춘다
  (예: `export ROS_DOMAIN_ID=0`). 기본 RMW(FastDDS)로 검증됨 — 한 터미널만 CycloneDDS로 소싱하면 안 보인다.
- **sim_runner가 `ModuleNotFoundError`(lazy_loader 등)로 죽음**: `./isaaclab.sh -p`가 base conda를
  가리키는 경우다. `conda activate isaac-6.0` 후 `python scripts/real2sim/sim_runner_go2.py ...`로 실행한다.
- **`/lowstate`는 오는데 로봇이 안 움직임**: `sim_runner`가 `--fix_base` 없이 지면에 떨어져 있거나,
  명령 관절 순서가 어긋난 경우. `--fix_base`로 격리해 재확인.

## 검증 상태 (Milestone 1)

- sim-side 왕복 E2E **PASS**: 실제 4-컴포넌트(sim_runner+sim_bridge) 기동 상태에서 `/lowcmd`(FL_hip
  0.1→0.5 스텝) → sim GO2 관절 추종(0.451) → `/lowstate` 300/300 수신.
- `gui_controller`는 빌드 + 오프스크린 단위검증 완료. 실제 창 클릭 육안 확인은 디스플레이가 있는
  환경에서 위 "실행 순서"대로 수행(명령 값은 `motions.DEFAULT_POSE`가 canonical과 일치함을 검증함).
