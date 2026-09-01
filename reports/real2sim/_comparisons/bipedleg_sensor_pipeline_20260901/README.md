# [2026-09-01] real_runner_bipedleg 센서 변환 데이터 흐름 해설 — 검증된 변환 vs 미검증 가정 4건

> **요약(BLUF)**: `scripts/real2sim/r2s_biped_leg/real_runner/real_runner_bipedleg.cpp`(biped_leg 8-DOF 실기 브리지)가 raw 모터/IMU 센서를 정책 관측(q, dq, gravity)까지 어떻게 변환하는지 소스 전수 추적. 아핀 변환·foot-calf 커플링 해제·projected gravity 공식은 손 유도로 검증됐으나, IMU 단위·SHM 레이아웃·`zero_deg` 캘리브레이션·IMU 유효성 신호 부재 등 4건이 미검증 상태로 실사용(TRACK 모드 16,881틱) 중이다.

- STATE 패킷에 IMU 유효 비트가 없어 "IMU 사망"과 "로봇이 진짜 직립"을 정책이 구분할 수 없음 — 08-28 캡처 482.9초 IMU 무수신 사태가 안 드러난 기전
- `zero_deg` 8축 전부 0.0(미캘리브레이션 placeholder)인 채로 TRACK 모드가 이미 운용 중
- go2 쪽 브리지(`scripts/real2sim/r2s_go2/`)는 별도 프로그램 — 이 보고서 범위 밖
- 다음: 파이 벤더 `RobotSharedMem.h` 확인, STATE에 IMU valid bit 추가

## 1. 물리 구조

라즈베리파이에서 실행. RobotEmbedded(EtherCAT master)와 Shared Memory로 모터를 교환하고, 워크스테이션과는 UDP로 연결된다.

| 항목 | 값 | 근거 |
|------|-----|------|
| 제어 루프 주기 | 1 ms | `real_runner_bipedleg.cpp:63` (`kLoopDtSec = 0.001`) |
| 페이싱 방식 | `clock_nanosleep(TIMER_ABSTIME)` 절대시각 | `:772-778` |
| ACT 포트 9887 | 수신 (정책 → 실기) | — |
| STATE 포트 9888 | 송신, 20 ms 페이싱(=50 Hz) | `:64`, `:571` |
| TELEM 포트 9889 | 송신, 5 ms 페이싱(=200 Hz), 관측 전용 | `:74`, `:625` |

STATE와 TELEM은 페이싱·포트·수신자가 모두 분리돼 있다. STATE 페이싱이 정책 50 Hz 클록을 소유한다(policy_runner가 lockstep으로 따라간다).

## 2. 브리지가 읽는 원시 센서 (SHM)

### 2-a. 모터 상태 — `MotGeneral_t`, 8축

`RobotMemGait_IsUpdatedMotorStatus16()` → `GetUpdatedFlag_MotorStatus16()` 비트마스크 → `GetMotorStatus16()` 축별 수신 (`:313-329`). 100회 수신 게이트 후에만 명령 전송이 enable된다(`kStatusWarmupCnt = 100`, `:81`).

브리지가 쓰는 필드는 셋뿐이다: `fPosition`[deg, 채널], `fVelocity`[deg/s, 채널], `fTorque`[N·m, 채널].

이 필드들은 전부 `float16`이다(`RobotTestGait/inc/define/defineConfigMotor.h:151-164` — 그 아래 `float` 버전은 주석처리된 죽은 코드). 즉 q·dq·tau는 채널→관절 변환 이전에 이미 양자화돼 들어온다. 코드 자신이 60 deg 부근의 ULP를 0.0625 deg ≈ 1.1 mrad로 명시하고 있다(`:242-243`). 이 양자화 폭이 뒤에서 `gear`로 나뉘므로(§4-a), 관절 공간 분해능은 축마다 다르다 — hip/thigh는 `gear=1.0`이라 그대로, calf는 `÷1.5`, foot은 `÷1.2` 더 미세한 단계로 변환된다.

### 2-b. IMU

`RobotMemGait_IsUpdatedIMU()` → `GetIMU(buf, IDX_OF_IMU_ForeC_START, LEN_OF_IMU_DATA)` (`:332-338`). 버퍼 전체를 가져오지만 실제로 쓰는 것은 `imu_buf[IDX_OF_IMU_ARPY + 0..1]` = roll, pitch 둘뿐이다(`imu_buf` 참조는 `:592`, `:593`, `:679`, `:742`, `:743`이 전부).

## 3. 센서 예산 — 읽는 것 / 가져오지만 버리는 것 / 아예 안 읽는 것

| 구분 | 필드 | 처리 |
|------|------|------|
| 읽어서 정책까지 감 | 모터 `fPosition` → q, `fVelocity` → dq | §4 변환 거쳐 STATE |
| 읽어서 정책까지 감 | IMU roll/pitch | §5 projected gravity 거쳐 STATE |
| 가져오지만 정책엔 안 감 | IMU yaw | TELEM `rpy[2]`로만 중계 — gravity 계산에서 제외(yaw는 중력에 무영향이라 의도된 설계) |
| 가져오지만 정책엔 안 감 | 모터 `fTorque` | TELEM 전용, STATE에는 없음(§7) |
| 아예 안 읽음 | IMU 가속도 `IDX_OF_IMU_ACCL` | 스텁 헤더에 정의는 있으나 `.cpp` 어디서도 참조 없음(grep 확인) |
| 아예 안 읽음 | 모터 `fCurrent` / `fAccelrationOrTemperture` / `ucStatus` | — |
| 애초에 채널 없음 | base 선속도·각속도 | 정책 obs 34차원 = gravity(3)+cmd(3)+q(8)+dq(8)+prev_action(8)+clock(4)(`gui_controller.py:149`, `:524`, `:554`) — 자이로가 obs에 들어갈 자리 자체가 없음 |

## 4. 좌표 변환 — 하드웨어 경계에서 딱 한 번

설계 규약(`:10-16`): UDP seam 양쪽은 관절(모델) 좌표만 주고받는다. 변환 지점은 `motor_deg_to_joint` / `motor_dps_to_joint` / `joint_to_motor_deg` 셋뿐이고, 워크스테이션에 raw 좌표가 존재해서는 안 된다.

### 4-a. 채널 ↔ raw 아핀 변환 (`:94-105`)

기호 정의:

| 기호 | 의미 | 단위 |
|------|------|------|
| `q_ch` | 채널(모터 드라이버) 각도 | deg |
| `q_raw` | 관절(모델) 각도 | rad |
| `dq_ch` | 채널 각속도 | deg/s |
| `dq_raw` | 관절 각속도 | rad/s |
| `sign` | 방향 부호 | ±1 (무차원) |
| `gear` | 드라이버 감속비 오설정 보정 계수 = 실제감속비/7 | 무차원 |
| `zero_deg` | 채널각 기준 영점 오프셋 | deg |
| `rad2deg`, `deg2rad` | 단위 환산 상수 | — |

```text
쓰기: q_ch  = sign * q_raw * rad2deg * gear + zero_deg
읽기: q_raw = sign * (q_ch - zero_deg) * deg2rad / gear
읽기: dq_raw = sign * dq_ch * deg2rad / gear      (속도는 zero_deg 상수항이 안 붙음)
```

`gear`는 hip 1.0 · thigh 1.0 · calf **1.5** · foot **1.2**이다(`calib_bipedleg.hpp:80-89`, 모터순서 {1.0, 1.0, 1.5, 1.2, 1.0, 1.0, 1.5, 1.2}). `zero_deg`는 채널각 단위이므로 읽기 변환에서 **`gear`로 나누기 전에** 뺀다 — 순서를 바꾸면(먼저 나누고 빼면) 오프셋 크기가 `gear`배만큼 달라져 틀린 관절각이 나온다.

### 4-b. foot ↔ calf 기구 커플링 해제 (`:119-156`, `calib_bipedleg.hpp:37-54`)

발목이 링키지로 구동돼 무릎(calf) 각을 따라간다. `FOOT_CALF_COEF = 1.0`:

```text
읽기: q_joint_foot = q_raw_foot - coef * q_joint_calf
쓰기: q_raw_foot   = q_joint_foot + coef * q_joint_calf
```

함정 세 가지:

1. 이 커플링 연산은 **policy(articulation) 순서 벡터에서만** 성립한다. policy 순서 `[HL_hip, HR_hip, HL_thigh, HR_thigh, HL_calf, HR_calf, HL_foot, HR_foot]`에서는 짝이 정확히 `p-2`지만, 모터(leg-major) 순서에서는 규칙이 다르다(`calib_bipedleg.hpp:41-47`).
2. 쓰기 커플링 되먹임은 **목표값끼리** 합성한다(측정 calf가 아니라 명령 calf). 학습 env가 그렇게 구현돼 있기 때문이다(`hind_leg_env.py:267`). 측정값을 쓰면 추종오차·지연만큼 sim과 실기의 raw 목표가 갈린다(`:146-148`).
3. 이 변환 함수들(`motor_deg_to_joint` 등) 자체는 **클램프하지 않는다** — RELAX 휴지 자세가 공칭 soft limit 밖이라 여기서 자르면 자세가 왜곡된다. 클램프는 TRACK 목표 생성부에서만 별도로 한다(`:115-116`, `:492-499`).

### 4-c. 인덱스 재배열

`POLICY_TO_MOTOR = {0, 4, 1, 5, 2, 6, 3, 7}` — 실기 leg-major ↔ 정책 type-major 순서 변환.

## 5. projected gravity — 공식은 검증됨, 입력 단위는 미검증

`:591-606`:

기호 정의:

| 기호 | 의미 | 단위 |
|------|------|------|
| `roll`, `pitch` | IMU 롤·피치 | rad (변환 후) |
| `g[0..2]` | body-frame projected gravity 벡터 | 무차원 (단위 벡터 성분) |

```text
if IMU_RPY_IS_DEG: roll *= deg2rad; pitch *= deg2rad

g[0] = +sin(pitch)
g[1] = -sin(roll) * cos(pitch)
g[2] = -cos(roll) * cos(pitch)
```

이번 세션에 손으로 유도해 공식이 맞음을 확인했다: ZYX 오일러(R = Rz·Ry·Rx)에서 `g_body = R^T · (0,0,-1) = [sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)]`. yaw가 공식에서 빠지는 것도 정상이다(중력 방향은 yaw 회전에 불변).

단, **이 공식 정합성은 입력 단위 정합성과는 별개**다 — `roll`, `pitch`를 IMU에서 읽어오는 시점의 원시 단위가 정말 도(deg)인지는 §9-1에서 별도로 미검증 상태다. 공식이 맞다는 것이 gravity 출력값이 맞다는 것을 보장하지 않는다.

## 6. 구조적 비대칭 — STATE에는 IMU 유효 플래그가 없다

- TELEM(`PolicyTelemPacket`)은 IMU 수신 여부를 `valid_mask` bit 8로 내린다(`:675-677`). 관절 유효성도 bit 0..7로 내린다.
- STATE(`PolicyStatePacket`, `r2s_packets.hpp:61-70`)는 `q`/`dq`/`gravity`/`convention_version`뿐이다 — **IMU 유효 비트가 없다.**
- IMU 미수신 시 브리지는 `gravity = (0, 0, -1)` 직립 가정을 폴백으로 내보낸다(`:602-606`).

정책 입장에서 "IMU가 죽음"과 "로봇이 진짜 직립"이 **구분 불가능**하다. 이것이 2026-08-28 캡처에서 482.9초 내내 IMU가 한 번도 안 들어왔는데도 그 사실이 드러나지 않은 기전이다(판정 근거: 그 구간 `gravity`가 정확히 `(0,0,-1)`이고 std가 0). 관련 보고서: `reports/real2sim/_comparisons/bipedleg_policy_capture_20260828/README.md` §4.

## 7. tau 규약 — 변환하지 않고 통과

`:666-672`: `tm.tau[p] = sign * fTorque` — **sign만 적용**, `gear`·커플링 전치는 **미적용**. 즉 TELEM의 tau는 채널기준이고, 소비자가 `× gear`해야 관절토크가 된다(calf 1.5, foot 1.2). 브리지가 변환하지 않는 이유는 `convention_version`을 올리지 않고 기존 소비자를 깨지 않기 위해서다(변환하면 버전 2로 올려야 한다).

따름정리: 펌웨어 토크 트립 임계 15 N·m도 채널기준이라, calf의 실제 관절 트립은 22.5 N·m다(`calib_bipedleg.hpp:114-115`).

08-28 보고서가 인용한 τ 수치는 HL_hip/thigh 축(gear=1.0)에서 잰 것이었고, 그 두 축은 채널=관절이라 이 규약에 영향받지 않는다 — 즉 그 보고서의 τ 판정 자체는 이 사실로 뒤집히지 않는다.

커플링 전치 되먹임(`τ_raw_calf -= coef * τ_foot`)도 미구현이지만, MIT 모드로 `cmd.fTorque = 0`을 보내는 한 피드포워드 토크가 없어 지금은 불필요하다. `fTorque ≠ 0`을 쓰게 되면 그때 필요해진다.

## 8. TELEM 전용 진단 채널 (2026-08-19 추가)

- `telem_tick` 단조 카운터 — 소비자가 시간축을 구조적으로 복원.
- `valid_mask` bit 0..7 관절 유효 / bit 8 IMU / bit 16+p **목표가 soft limit으로 잘림** / bit 24 `cmd_q` 유효.
- `cmd_q` — 실제 전송한 float16 값을 되읽어(`applied_ch_deg`, `:558-559`) `q`와 **같은 변환**으로 관절 좌표에 올린 값. 소비자가 아무 변환 없이 `cmd_q - q`를 추종오차로 쓸 수 있다(`:684-689`).
- foot 유효성 규칙: foot 관절각은 같은 다리 calf 없이는 정의되지 않으므로, calf가 무효면 foot도 무효로 내린다(`:657-659`).
- TELEM 페이싱은 위상 **누적**(`+= dt`)이지 리셋(`= t`)이 아니다 — 리셋하면 1 kHz 루프의 반 틱 지연이 매 주기 누적돼 실측 레이트가 목표보다 낮아진다. 구 50 Hz TELEM이 캡처에서 48.1~48.7 Hz로 찍힌 원인이 그것이고, 5 ms 목표(200 Hz)에서는 이 효과가 상대적으로 더 커져 −10%(실측 180 Hz)로 나타난다(`:626-635`).

## 9. 미검증 가정 목록 (전부 소스 자신이 플래그를 달아 둔 것)

1. **`IMU_RPY_IS_DEG = true`** — `calib_bipedleg.hpp:152-153` 주석: "RobotTestGait는 `%6.1f`로 출력(도 단위로 추정). 실측으로 확정할 것." 실패 모드: 실제가 라디안이면 `× deg2rad`가 값을 57.3배 줄여 gravity가 틀렸지만 그럴싸한(작은 기울기처럼 보이는) 값을 낸다.
2. **IMU 버퍼 레이아웃**(`IDX_OF_IMU_ARPY=0`, `ACCL=3`, `LEN=6`) — 이 값들은 `real_runner/stub/RobotSharedMem.h:14-18`의 것이고, 헤더 자신이 "실헤더 값과 다를 수 있음(스텁 전용 가정값)"이라 적고 있다. 파이는 벤더 실헤더로 컴파일되며, 그 헤더는 이번 세션에 읽지 못했다(로컬에 없음 — `find /` 결과 스텁 하나뿐). 해소 경로: 파이의 `RobotSharedMem.h`를 한 번 읽는 것.
3. **`fVelocity` 단위 [deg/s]** — `:586` 주석이 "Data Format 문서"를 근거로 들지만 이번 세션에 그 문서를 확인하지 못했다.
4. **`zero_deg`가 8축 전부 0.0**(`calib_bipedleg.hpp:80-89`). 헤더 배너(`:12-15`)가 "sign/zero_deg는 미캘리브레이션 placeholder … 실기 probe로 확정하기 전에는 TRACK 모드 금지"라고 적고 있는데, 08-28 캡처는 TRACK으로 16,881틱을 돌았다. 즉 §4-a 아핀 변환의 상수항이 실측되지 않은 상태이고, 관절 영점이 사실상 "엔코더가 0이라 부르는 위치"로 정의돼 있다.

   세 가지를 섞지 말 것:
   - `sign`은 **해소됐다**. `calib_bipedleg.hpp:73-76`: 실기 방향을 실측한 뒤 sim 자산 쪽 축을 반전(`Hind_Leg_URDF3_SignFix`)해 맞췄으므로 전 축 `+1.0f`가 설계상 정답이고, 반전 관절은 클램프 범위만 `[lo,hi] → [-hi,-lo]`로 뒤집혔을 뿐이다. `sign`과 `zero_deg`를 한 덩어리 "placeholder"로 묶지 말 것.
   - `zero_deg`(하드웨어 경계의 기준각)와 GUI의 `_POSE_OFFSET_DEG`(`gui_controller.py:975`, 워크스테이션 쪽 목표 트림)는 서로 다른 프로세스의 서로 다른 메커니즘이다.
   - 08-28 보고서의 HL_hip 처짐(stiction)과 인과를 연결하지 않는다. 그 보고서는 τ가 0으로 수렴하지 않는다는 근거로 엔코더 영점 오차를 이미 배제했다. `zero_deg=0`은 그와 별개의 열린 캘리브레이션 갭이다.
   - 파생 오염 1건: `RELAX_REST_POSE_SIM`(`calib_bipedleg.hpp:124-142`)이 이 placeholder 프레임에서 캡처된 값이고, 주석이 "zero 캘리브레이션 후 재캡처할 것"이라 적고 있다.

## 10. 다음 액션

- [ ] 파이의 벤더 `RobotSharedMem.h`를 한 번 읽어 IMU 레이아웃·단위 확정 (가장 싸고, §9-1·9-2를 동시에 닫는다)
- [ ] STATE 패킷에 IMU 유효 비트 추가 — 지금은 정책이 IMU 사망을 알 방법이 없다. 규약 변경이므로 `convention_version` 상향 동반
- [ ] probe 모드로 `zero_deg` 실측 후 `RELAX_REST_POSE_SIM` 재캡처
- [ ] (기존 미해결) IMU 미수신 자체의 원인 규명 — 접지 보행 전 필수
