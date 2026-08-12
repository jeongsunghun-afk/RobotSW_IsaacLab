# real_runner_bipedleg — 실기(라즈베리파이) UDP↔SharedMem 브리지

`RobotTestGait` 예제(모터 제어 코드 + 통신 설명 PDF)를 참조해 만든, r2s_biped_leg policy 모드의
**real 엔드포인트** 구현이다. 라즈베리파이에서 돌며 워크스테이션의 `policy_runner_bipedleg.py` 와
UDP seam(9887/9888)으로 연결된다.

```
[워크스테이션]                                [라즈베리파이 유선 192.168.60.5 / WiFi 192.168.10.19]
policy_runner ──POLICY_ACT("R2PA",9887)──▶ real_runner ──SHM──▶ RobotEmbedded ──EtherCAT──▶ MCU ──CAN-FD──▶ MD80×8
              ◀─POLICY_STATE("R2PS",9888)──    │        ◀──SHM── 모터 상태(pos/vel) + IMU(RPY)
                                        1 kHz 루프 / 회신은 20 ms 페이싱
```

## 핵심 설계 (RobotTestGait 준수 사항)

| 항목 | 값 | 근거 |
|---|---|---|
| 제어 주기 | 1 ms 루프 | RobotTestGait 1 ms POSIX 타이머 |
| 상태 읽기 | `IsUpdatedMotorStatus16` → 비트 flag → `GetMotorStatus16` | main.cpp 예제 패턴 |
| 전송 게이트 | 모터 상태 **100회** 수신 후 명령 enable | main.cpp `m_ucGaitReceivedCnt_MotorStatus` |
| 명령 형식 | `MotGeneral_t` (ucMode=1, pos[**deg**], kp/kd, float16) → `SetMotorCommand16` | Data Format PDF: pos=deg, vel=deg/s, tau=Nm |
| 게인 상한 | kp∈[0,500], kd∈[0,**5**] 클램프 | `defineConfigMotor.h` `DEF_MOT_GAIN_*_MAX` |
| **STATE 회신 페이싱** | **20 ms 간격** | sim seam은 lockstep(시뮬이 시간 소유)이지만 실기는 실시간 → 즉시 회신하면 policy가 kHz로 자유질주해 gait clock이 실시간의 수십 배로 돈다. 브리지가 50 Hz로 묶는다 |
| **TELEM 송신** | peer 등록 즉시 50 Hz, peer:**9889** | 관측 전용 q/dq/**tau**/rpy + `valid_mask`(bit0~7=관절, bit8=IMU). **모터 warmup 전에도 송신** → 워크스테이션에서 "링크 죽음 vs 모터데이터 없음" 구분 가능. STATE는 반대로 all_stt 게이트 유지(정책 입력에 가짜 0 금지) |
| **PING 수신** | 8 B "R2PG" | 모니터 keepalive. **peer 등록만** 하고 목표/상태머신 불변 → bridge TRACK 중에도 무해 |
| **RELAX 수신** | 8 B "R2PL" | 무토크(limp) 요청 — bridge 모드에서 RELAX 상태로 전환해 **kp=kd=tau=0을 능동 송신**(명령 중단이 아님). 이후 ACT 수신 시 처진 **현재 자세를 재래치**하고 ENGAGE 램프로 복귀. probe/hold에선 무시. GUI 기동 기본 상태 + "Relax (zero torque)" 버튼이 보낸다 |
| **GAIN 수신** | 72 B "R2PK", peer:**9887**(ACT와 동일) | kp/kd 런타임 갱신 — articulation 순서로 받아 `POLICY_TO_MOTOR`로 모터 순서 매핑 후 드라이버 상한 kp∈[0,500]/kd∈[0,5]로 클램프해 `kp_cmd`/`kd_cmd`에 반영(다음 틱부터 적용). **목표각·상태머신은 불변** — RELAX 중엔 명령 전송부가 `kp_cmd`/`kd_cmd` 대신 0을 강제 송신해 무토크를 유지하므로 GAIN 갱신 자체는 안전. 값이 실제로 바뀔 때만 로그(1 Hz 갱신 송신 대비 콘솔 스팸 방지) |

관절 매핑: 정책 UDP는 articulation(type-major) `[HL_hip, HR_hip, HL_thigh, HR_thigh, HL_calf,
HR_calf, HL_foot, HR_foot]`, 실기 모터는 leg-major `[LtR, LtP, LkP, LaP, RtR, RtP, RkP, RaP]`
(0~7; SHM 채널 8/9=waist 미사용). 변환은 `calib_bipedleg.hpp` `POLICY_TO_MOTOR` + sign/zero_deg.

## 파이에서 빌드·실행

```bash
# 1) 소스 복사 (RobotTestGait 헤더를 상대경로로 참조하므로 상위 폴더째)
scp -r scripts/real2sim/r2s_biped_leg/{real_runner,RobotTestGait} rpetubt@192.168.60.5:~/ZSource/r2s_bridge/
# 접속 정보(예제 PDF): user rpetubt / rga2023!
# 유선 직결(ens10f1↔파이 enp3s0) = 192.168.60.5, WiFi(RGA1_5G) = 192.168.10.19
# 파이 eth0(192.168.50.3)은 EtherCAT용이므로 서버 연결에 사용 금지

# 2) 빌드 (파이에는 /usr/include/RobotSharedMem.h + libRobotSharedMem.so 설치돼 있어야 함
#    — 없으면 ~/ZSource/RobotSharedLib 에서 컴파일·설치, '기타 Shared Memory 라이브러리' PDF 참조)
cd ~/ZSource/r2s_bridge/real_runner && mkdir -p build && cd build && cmake .. && make

# 3) 실행 순서 (PDF의 RobotTestGait 실행 순서와 동일)
sudo ~/ZSource/RobotEmbedded/build/src/RobotEmbedded     # 터미널 A: EtherCAT master 먼저
sudo ./real_runner_bipedleg --probe                      # 터미널 B: 브리지 (아래 브링업 사다리)
```

워크스테이션 쪽은 기존 3-터미널에 `REAL_HOST`만 추가:

```bash
r2s_bl_psim                                  # sim (관찰용 미러)
REAL_HOST=192.168.60.5 r2s_bl_prun           # policy_runner — REAL_ACT를 파이로도 fan-out (유선; WiFi면 192.168.10.19)
r2s_bl_gui                                   # GUI → Mode: Policy, Source: Real
```

## 브링업 사다리 (반드시 순서대로)

1. **`--probe`** — 명령 전송 없음. 관절을 손으로 움직여 sign/zero/limit 실측 (0.5 s마다 출력).
2. **`--hold`** — 현재 자세 latch 후 위치유지만 전송. PD 게인·통신 체인의 액추에이션 안전 확인.
   ACT를 받아도 TRACK으로 넘어가지 않는다.
3. **(기본 브리지)** — HOLD로 시작, 첫 ACT 수신 시 500 ms engage ramp 후 TRACK.
   브링업 중엔 `--slew_dps 180` 권장(급격한 명령 제한), 검증 후 제거(학습 plant와 일치시키려면 off).

옵션: `--act_port N`(기본 9887) `--engage_ms N`(기본 500) `--slew_dps X`(기본 0=무제한)

## ⚠ 실기 투입 전 캘리브레이션 체크리스트 (`calib_bipedleg.hpp`)

기본값은 **placeholder(sign=+1, zero_deg=0)** 다. 미캘리브레이션 상태로 TRACK 금지.

- [ ] **sign** ×8: `--probe`로 각 관절을 sim 양(+)방향(URDF 축)으로 손으로 밀고 출력 부호 확인.
      hind_leg 사례처럼 좌우 부호가 URDF 표기와 뒤집힌 관절이 있을 수 있다 — 관절별 실측이 정답.
- [ ] **zero_deg** ×8: 로봇을 중립자세(sim q=0)에 물리적으로 정렬하고 `--probe` 출력값 기록.
      RobotTestGait 예제의 `fPosZero {-90,0,60,-90,30,0,90,90}`는 모터 원점≠로봇 중립을 시사한다.
- [ ] **soft limit 재확인**: sign/zero 확정 후 실기 가동범위가 `SOFT_LIMITS_RAD`를 실제로 커버하는지.
- [ ] **IMU**: RPY 단위(deg 가정), roll/pitch 부호 규약, 그리고 이 SHM 경로에서 IMU가 실제로
      갱신되는지(`--probe` 출력에 "IMU 미수신!" 표시). 미수신이면 gravity=(0,0,-1) 고정 —
      **기립 균형에 gravity가 필수이므로 미수신 상태로 실보행 금지.**
- [ ] **속도 단위**: 상태 fVelocity가 deg/s인지(Data Format PDF 기준) 사인 스윕으로 교차검증.
- [ ] **hip kd**: sim 6.0 vs 드라이버 상한 5.0 → 5.0으로 클램프됨(기동 시 WARN). plant 차이로
      남는 항목 — 필요 시 sim 쪽을 kd=5로 재학습/재검증.
- [ ] **SHM API**: 실헤더 `/usr/include/RobotSharedMem.h` 시그니처가 stub 가정과 다르면 컴파일
      에러로 드러난다 — 에러 나면 stub이 아니라 **호출부를 실헤더에 맞출 것**.

## 로컬 검증 (로봇 없이)

```bash
cd real_runner && mkdir -p build && cd build && cmake -DSTUB_SHM=ON .. && make
./real_runner_bipedleg &   # stub: 1차 지연 echo plant + 직립 IMU
# 별도 셸에서 r2s_udp.py로 ACT 송신 → STATE 회신 파리티/페이싱/매핑 왕복 검증
```

2026-08-06 검증: 회신 페이싱 20.6 ms, q 왕복 오차 0.0005 rad(float16 양자화), gravity (0,0,-1),
HOLD 모드 ACT 무시, 게인 클램프 WARN 정상.

GAIN 송신 검사(kp/kd 런타임 갱신, `r2s_udp.pack_policy_gain` 사용):

```bash
# 전 관절 broadcast: kp=50, kd=2
./comm_check.py --host 127.0.0.1 --kp 50 --kd 2 --sine_joint 2 --duration 4
# 관절별 지정(articulation 순서, 콤마 8개)
./comm_check.py --host 127.0.0.1 --kp 100,100,50,50,50,50,20,20 --kd 5 --duration 4
```

2026-08-12 stub 루프백 검증: `--sine_joint 2` 고정 조건에서 kp 50→100 시 TELEM max|tau[HL_thigh]|
2.22→4.67 N·m(비 2.10, 기대치 2.0 — stub 합성 tau = 0.02·kp·err 이므로 비례가 GAIN 반영의 직접
증거) — 오차는 두 실행 간 실시간 페이싱 지터(±수 ms)에서 기인. GAIN 패킷은 5회 반복 송신했지만
파이 로그의 "GAIN 수신" 줄은 값이 바뀔 때 **1회만** 출력됨을 확인(동일값 재송신 시 스팸 없음).

## 파일

| 파일 | 역할 |
|---|---|
| `real_runner_bipedleg.cpp` | 1 kHz 브리지 본체 (상태머신 WAIT_STATUS→HOLD→ENGAGE→TRACK) |
| `r2s_packets.hpp` | `r2s_udp.py` policy 패킷과 바이트 일치하는 C++ 정의 (r2s_udp.py가 단일 진실) |
| `calib_bipedleg.hpp` | 모터↔정책 축 매핑 + sign/zero/게인/soft limit 표 (**캘리브레이션 대상**) |
| `comm_check.py` | 워크스테이션용 UDP seam 검사기 — ACT 50 Hz 송신, STATE 페이싱/수신율/TELEM 판정. `--sine_joint`로 단관절 sine 액추에이션 테스트(⚠ bridge 모드에선 실제로 움직임) |
| `stub/` | 로컬 빌드·루프백 테스트용 가짜 SHM(echo plant, 합성 dq/tau) + float16 shim. 파이 빌드에선 미사용 |

## 실기 모니터링 (gui_controller "Real Robot Monitor")

워크스테이션에서 관절별 q/dq/tau + IMU rpy 를 라이브로 본다 (TELEM 9889, 관측 전용):

```bash
bash scripts/real2sim/r2s_biped_leg/run_gui_controller.sh --monitor            # 기동 즉시 수신 시작
# 또는 GUI 패널에서 Host(기본 192.168.60.5) 입력 후 Start
# 패널의 "Plot" 버튼 = 관절별 q/tau/dq 실시간 plot 창(monitor.py 별도 프로세스, 9890 중계).
#   단독 실행: python3 monitor.py --port 9890 --label real --no_action --joint HL_thigh
```

- 모니터는 링크가 조용할 때만 8 B PING 을 보낸다(목표 없음) — probe/hold/bridge 어느 모드에서도 안전.
- policy 모드(ACT 스트림)와 **동시 사용 가능** (STATE 9888과 포트 분리).
- 표의 관절 순서는 TELEM = articulation type-major (`motions.JOINT_NAMES` 의 leg-major와 다름).
- ⚠ TELEM 은 2026-08-10 추가 — 파이의 real_runner 가 그 전 빌드면 **re-scp + rebuild** 필요.
