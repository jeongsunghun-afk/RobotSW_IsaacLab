# real_runner_bipedleg — 실기(라즈베리파이) UDP↔SharedMem 브리지

`RobotTestGait` 예제(모터 제어 코드 + 통신 설명 PDF)를 참조해 만든, r2s_biped_leg policy 모드의
**real 엔드포인트** 구현이다. 라즈베리파이에서 돌며 워크스테이션의 `policy_runner_bipedleg.py` 와
UDP seam(9887/9888)으로 연결된다.

```
[워크스테이션]                                [라즈베리파이 유선 192.168.60.5 / WiFi 192.168.10.19]
policy_runner ──POLICY_ACT("R2PA",9887)──▶ real_runner ──SHM──▶ RobotEmbedded ──EtherCAT──▶ MCU ──CAN-FD──▶ MD80×8
              ◀─POLICY_STATE("R2PS",9888)──    │        ◀──SHM── 모터 상태(pos/vel) + IMU(RPY)
                                        1 kHz 루프 / STATE 20 ms 페이싱 · TELEM 5 ms(200 Hz)
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
| **TELEM 송신** | peer 등록 즉시 **200 Hz**(`kTelemDtSec`=5 ms), peer:**9889** | 관측 전용 q/dq/**tau**/rpy + `valid_mask`(bit0~7=관절, bit8=IMU). **모터 warmup 전에도 송신** → 워크스테이션에서 "링크 죽음 vs 모터데이터 없음" 구분 가능. STATE는 반대로 all_stt 게이트 유지(정책 입력에 가짜 0 금지). ★**STATE와 페이싱 완전 분리** — TELEM을 올려도 50 Hz 정책 클록은 불변 |
| **PING 수신** | 8 B "R2PG" | 모니터 keepalive. **peer 등록만** 하고 목표/상태머신 불변 → bridge TRACK 중에도 무해 |
| **RELAX 수신** | 8 B "R2PL" | 무토크(limp) 요청 — bridge 모드에서 RELAX 상태로 전환해 **kp=kd=tau=0을 능동 송신**(명령 중단이 아님). 이후 ACT 수신 시 처진 **현재 자세를 재래치**하고 ENGAGE 램프로 복귀. probe/hold에선 무시. GUI 기동 기본 상태 + "Relax (zero torque)" 버튼이 보낸다 |
| **GAIN 수신** | 72 B "R2PK", peer:**9887**(ACT와 동일) | kp/kd 런타임 갱신 — articulation 순서로 받아 `POLICY_TO_MOTOR`로 모터 순서 매핑 후 드라이버 상한 kp∈[0,500]/kd∈[0,5]로 클램프해 `kp_cmd`/`kd_cmd`에 반영(다음 틱부터 적용). **목표각·상태머신은 불변** — RELAX 중엔 명령 전송부가 `kp_cmd`/`kd_cmd` 대신 0을 강제 송신해 무토크를 유지하므로 GAIN 갱신 자체는 안전. 값이 실제로 바뀔 때만 로그(1 Hz 갱신 송신 대비 콘솔 스팸 방지) |

### ★TELEM 200 Hz (2026-08-19)

TELEM 페이싱이 **sysid 캡처의 시간 해상도를 정한다** — `gui_controller` 의 `RealMonitorThread`
가 수신하는 족족 기록하므로, `kTelemDtSec` 이 그대로 npz 의 `t_real` 레이트가 된다.

이전에는 TELEM 이 STATE 와 같은 20 ms 게이트 안에 있어 50 Hz 였고, 실제 캡처는 48.1~48.7 Hz 로
찍혔다. 그 해상도로는 chirp 을 5 Hz 로 올렸을 때 선형보간 오차가 진폭의 **5.2 %** 에 달해
(관측 잔차 RMS 전체와 맞먹는 크기) armature 를 식별할 수 없다. 200 Hz 면 **0.31 %** 다.
근거: `reports/_comparisons/pace_bipedleg_foot_coupling_probe/README.md` §23-j.

두 가지를 고쳤다:

1. **TELEM 을 STATE 게이트 밖으로 분리** (`kTelemDtSec` = 5 ms). 루프는 원래 1 kHz 라
   추가 비용이 없고, **20 ms STATE 페이싱은 그대로**다 — 그건 policy_runner 가 lockstep 이라
   50 Hz 정책 클록을 소유하는 값이므로 절대 건드리면 안 된다.
2. **페이싱을 위상 누적(`+= dt`)으로** — `= t` 리셋은 1 kHz 틱에서 매번 평균 반 틱씩 늦게
   걸린 지연이 누적돼 레이트를 떨어뜨린다. 이게 **구 50 Hz TELEM 이 48.5 Hz 로 찍힌 원인**이다.
   5 ms 목표에서는 −10 %(실측 180 Hz)까지 벌어졌고, 위상 누적으로 **정확히 200.0 Hz** 가 됐다.

stub 루프백 실측(`comm_check.py --duration 10`): TELEM **2000 개/10 s = 200.0 Hz**,
STATE 488 개 = 48.8 Hz(간격 mean 20.53 ms) — 즉 정책 클록 불변.

> ⚠ **미해결로 남긴 것**: STATE 도 같은 리셋 방식이라 실측 **48.8 Hz**(20.53 ms)로 돈다.
> sim 의 `STEP_DT = 0.02`(50 Hz) 대비 **2.4 % 느린 정책 클록**이고, 이건 그 자체로 sim2real
> 갭이다. 여기선 **고치지 않았다** — 보행 타이밍을 바꾸는 변경이라 별도 판단이 필요하다.

> ⚠ GUI 의 monitor.py **plot 중계는 50 Hz 로 솎는다**(`mon_relay_dt`). 기록(`_rec`)만 풀레이트다 —
> 그래프에 200 Hz 는 필요 없고 monitor.py 에 4 배 트래픽을 보낼 이유도 없다.

### ★서버 두 대가 한 로봇을 쓰는 법 (2026-08-19)

**브리지를 두 개 띄우면 안 된다.** `bind: Address already in use` 는 고쳐야 할 버그가 아니라
**안전장치**다 — 인스턴스가 둘이면 SHM writer 가 둘이 되어 같은 모터에 명령이 충돌한다.
(`SO_REUSEADDR` 를 일부러 걸지 않았다.)

대신 **한 인스턴스가 여러 워크스테이션에 TELEM 을 뿌린다**:

| | 대상 | 개수 | 등록 조건 |
|---|---|---|---|
| **STATE** (9888) | 명령 peer | **반드시 1** | **ACT 를 보낸 쪽**만 |
| **TELEM** (9889) | 관측 peer | 최대 `kMaxTelemPeers`=4 | 아무 패킷이나 보내면 등록 (IP 기준) |

즉 **관측은 여러 대가 동시에, 명령은 한 대만**이다. 관측 peer 는 `kPeerTimeoutSec`(10 s) 동안
아무 패킷도 없으면 내려간다. GUI 는 1 Hz 로 PING 하므로 살아 있는 한 만료되지 않는다.

> ⚠ 이 분리는 **잠재 버그도 함께 고쳤다**. 예전엔 peer 슬롯이 하나여서 모니터가 PING 만 보내도
> STATE 목적지가 그쪽으로 넘어갔다 — 정책 입력이 모니터로 새는 구조였는데, policy_runner 가
> 50 Hz 로 ACT 를 쏴서 즉시 되찾는 바람에 드러나지 않았을 뿐이다.

루프백 실측(소스 IP 127.0.0.1 / 127.0.0.2 동시): **양쪽 다 200.2 Hz** 수신.

`bind` 오류가 났다면 먼저 좀비를 의심할 것:

```bash
sudo ss -ulpn | grep 9887
ps -ef | grep real_runner
```

정말로 **로봇이 두 대**라면 그때는 인스턴스도 둘이 맞고, `--act_port` 로 분리하면 된다.

### probe 출력 — `zero_deg` 캘리브레이션용 채널각 (2026-08-19)

`--probe` 는 관절각 아래에 **채널각 원값**을 함께 찍는다:

```
[HOLD] q_joint[rad]: +0.031 -0.512 ...              ← 관절각, policy 순서
        ch_deg(모터순서, zero_deg 에 그대로 복사): +1.78 -29.33 ...
                                               HL_hip HL_thigh HL_calf HL_foot ...
```

`zero_deg` 는 **채널 deg · 모터 순서**로 정의되는데 위 `q_joint` 줄은 관절각 · policy 순서라
그대로 옮길 수 없다. 손으로 환산하려면 gear 를 곱하고 foot 은 calf 를 다시 더해야 하는데
(raw = foot + calf), 그게 이 프로젝트에서 반복해 사고를 낸 변환이다. 그래서 드라이버 보고값을
**변환 없이** 찍는다.

절차: 로봇을 sim 중립자세(전 관절 q=0)에 물리적으로 정렬 → `ch_deg` 줄을 `MOTOR_CALIB` 의
`zero_deg` 에 순서대로 복사 → 재빌드 → probe 재실행 → **`q_joint` 줄이 전부 0 근처인지 확인**.
이 마지막 확인 없이 넘어가지 말 것.

관절 매핑: 정책 UDP는 articulation(type-major) `[HL_hip, HR_hip, HL_thigh, HR_thigh, HL_calf,
HR_calf, HL_foot, HR_foot]`, 실기 모터는 leg-major `[LtR, LtP, LkP, LaP, RtR, RtP, RkP, RaP]`
(0~7; SHM 채널 8/9=waist 미사용). 변환은 `calib_bipedleg.hpp` `POLICY_TO_MOTOR` + sign/zero_deg/**gear**.

⚠ **gear** (2026-08-14 추가) — 드라이버가 전 축을 7:1 감속비로 가정해 각도를 보고/수신하므로
채널각 = 관절각 × `gear`(실제감속비/7 = hip 1.0 · thigh 1.0 · calf **1.5** · foot **1.2**).
`RL_INTERFACE.md` §4. **위치·속도에만** 적용했다. 게인(kp/kd)과 `tau` 는 드라이버 보고값 그대로다.

⚠ **게인·토크의 gear 관계는 2026-08-14 실측으로 확정됐다** (`calib_bipedleg.hpp` 의 `GAIN_GEAR`
블록). 준정적 구간에서 `|τ_보고| / |kp·e_관절 − kd·q̇_관절|` = **1.5008** (calf, gear 1.5 대비 오차
0.06%; hip/thigh 대조군 1.0006~1.0029) → 드라이버 PD 가 **채널각 오차**에 게인을 곱한다. §4 의
"실제토크 = 보고토크 × gear" 와 결합하면 **실효 관절강성 = kp·gear²**(calf 50→112.5 · foot
30→43.2), 감쇠비는 ζ×gear(calf 0.76→1.13 과감쇠). 목표 관절강성을 내려면 `kp_ch = kp_joint/gear²`.
- `tau` 는 **채널기준**이다 — 소비자가 `× gear` 하면 관절토크(calf 1.5 · foot 1.2). 토크 트립
  임계 15 Nm 도 채널기준이라 calf 는 실제 22.5 Nm다.
- **게인 변환 코드는 넣지 않았다.** 현재 kp/kd 는 실기팀이 실기에서 직접 고른 **채널 게인**이라
  변환해 넣을 sim 게인 원본이 없다. 변환은 정책 배포 시점에 "목표 관절강성 → 채널 게인"
  방향으로 정할 문제다. ⇒ 드라이버가 고쳐져 gear=1 이 되면 calf·foot 이 2.25·1.44배 약해진다.

⚠ **foot↔calf 커플링** (2026-08-14 추가) — 발목이 링키지로 무릎에 물려 있어 채널각은 관절각이
아니라 raw각(`q_raw_foot = q_foot + 1.0·q_calf`, `RL_INTERFACE.md` §2)이다. 브리지가 이것도
해제하므로 **UDP seam 양쪽은 관절(모델) 좌표만 주고받는다** — 워크스테이션에 raw 좌표가 남아
있으면 안 된다. 명령의 커플링 되먹임은 **목표값끼리** 합성한다(측정 calf 가 아니다 —
학습 env `hind_leg_env.py:267` 과 동일). 커플링 연산은 **policy(articulation) 순서**에서만
성립한다(calf p=4,5 / foot p=6,7 → 짝이 `p-2`); 모터 순서에서는 짝 규칙이 다르다.

⚠ **`convention_version`** — 브리지가 자기 좌표·단위 규약을 **STATE·TELEM 양쪽에** 실어 보낸다
(현재 **1** = gear 적용 + foot 관절 좌표). GUI 는 이 값을 **그대로** npz 에 기록할 것 — 예전처럼
GUI 가 상수로 찍으면 배포와 도장이 어긋난다. 버전은 "무엇이 판정됐나"가 아니라 **"브리지가
무엇을 하는가"** 를 가리킨다(그래서 tau 단위가 확정돼도 브리지가 변환을 안 하는 한 1 이다).

| 패킷 | 크기 | 버전 필드 offset |
|---|---|---|
| STATE (9888) | 84 → **85 B** | 84 |
| TELEM (9889) | 121 → **157 B** | 120(규약)·124(tick)·128~156(cmd_q) |

기존 offset 은 매번 **불변**이라 `r2s_udp.py` 는 포맷 **끝에만** 필드를 붙이면 된다
(STATE 는 `B`, TELEM 은 `B` → `BI8f`). 반영 전까지 **policy_runner·GUI·`comm_check.py` 가 크기
불일치로 패킷을 거부한다** — 조용한 오독보다 낫다는 판단으로 일부러 크기를 바꿨다.
`unpack_policy_telem` 은 구 크기(121/120/116 B)도 계속 읽되 `telem_tick`·`cmd_q` 를 **None** 으로
돌려준다 — 0 으로 채우면 "에코가 없다"가 "에코가 0이다"로 조용히 둔갑한다.
연산 순서는 `q_ch = q_raw·sign·gear + zero_deg` / `q_raw = (q_ch − zero_deg)/(sign·gear)` —
`zero_deg` 는 **채널각 단위**라 gear 로 나누기 전에 뺀다.

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
