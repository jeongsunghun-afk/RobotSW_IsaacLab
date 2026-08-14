/**
 * calib_bipedleg.hpp — 실기 모터축(RobotTestGait ENUM_GAIT_JOINT_ID 순) ↔ 정책 축 캘리브레이션 표.
 *
 * 모터 인덱스(leg-major, RobotTestGait 와 동일):
 *   0 LtR(HL_hip)  1 LtP(HL_thigh)  2 LkP(HL_calf)  3 LaP(HL_foot)
 *   4 RtR(HR_hip)  5 RtP(HR_thigh)  6 RkP(HR_calf)  7 RaP(HR_foot)
 *   (SHM 채널 8/9 = waist roll/pitch, 이 로봇에선 미사용)
 *
 * 정책(articulation, type-major) 순서 — POLICY_MODE_SPEC.md:
 *   [HL_hip, HR_hip, HL_thigh, HR_thigh, HL_calf, HR_calf, HL_foot, HR_foot]
 *
 * ⚠⚠ sign / zero_deg 는 **미캘리브레이션 placeholder(+1 / 0)** 다.
 *    실기 probe(README §캘리브레이션)로 확정하기 전에는 TRACK 모드 금지.
 *    RobotTestGait 예제의 fPosZero {-90,0,60,-90,30,0,90,90} 는 모터 원점이
 *    로봇 중립자세와 다를 수 있음을 시사한다 — 반드시 실측할 것.
 *
 * ⚠⚠ 구조적 갭 (RL_INTERFACE.md, 2026-08-10 영점 인계 문서에서 확인):
 *    1. [2026-08-14 해소] 드라이버가 전 축 7:1 가정으로 보고하므로 채널각 = 관절각 × gear_k
 *       (hip/thigh 1.0 · calf 1.5 · foot 1.2). MotorCalib::gear 필드로 반영했다 (§4).
 *       각도·속도만 보정했고 **게인(kp/kd)은 미보정** — GAIN_GEAR 블록 참조(관계는 실측 확정,
 *       적용은 정책 배포 시점 문제).
 *    2. [2026-08-14 해소] foot 은 calf 와 기구 커플링 (q_raw_foot = q_foot + 1.0·q_calf, §2).
 *       COUPLED_CALF_POLICY / FOOT_CALF_COEF 로 반영했고, 브리지가 하드웨어 경계에서 한 번만
 *       변환하므로 **워크스테이션은 관절(모델) 좌표 하나만 본다**. 각도·속도만 해제했고
 *       **토크의 전치 되먹임(τ_raw_calf −= coef·τ_foot)은 미구현** — MIT 모드로 fTorque=0 을
 *       보내는 한 피드포워드 토크가 없어 불필요하다(GAIN_GEAR 블록 참조).
 *    3. 채널각은 ±180° 래핑된다(클램프 아님) — 송신 전 포화 필수 (§5).
 */
#ifndef __CALIB_BIPEDLEG_HPP__
#define __CALIB_BIPEDLEG_HPP__

constexpr int NUM_MOTORS = 8;

// 정책 인덱스 p → 실기 모터 인덱스 (articulation type-major → leg-major)
constexpr int POLICY_TO_MOTOR[NUM_MOTORS] = {0, 4, 1, 5, 2, 6, 3, 7};

// ★foot↔calf 기구 커플링 (RL_INTERFACE.md §2). 발목이 링키지로 구동돼 무릎을 따라 움직인다:
//     q_joint_foot = q_raw_foot − coef·q_joint_calf      (읽기)
//     q_raw_foot   = q_joint_foot + coef·q_joint_calf    (쓰기)
//
// ⚠⚠ **인덱스는 policy(articulation) 순서다 — 모터 순서가 아니다.**
//   policy 순서 = [HL_hip, HR_hip, HL_thigh, HR_thigh, HL_calf, HR_calf, HL_foot, HR_foot]
//   → calf 는 p=4,5 · foot 은 p=6,7 이라 같은 다리 짝이 정확히 **p-2** 다.
//   (교차 확인: POLICY_TO_MOTOR[6]=3=HL_foot, POLICY_TO_MOTOR[4]=2=HL_calf → 둘 다 HL.
//    POLICY_TO_MOTOR[7]=7=HR_foot, POLICY_TO_MOTOR[5]=6=HR_calf → 둘 다 HR.)
//   모터(leg-major) 순서로는 foot=3,7 / calf=2,6 이라 짝이 p-1 이 되어 **다른 규칙**이다.
//   ⇒ 커플링 연산은 반드시 policy 순서 벡터에서 하고, 모터 순서로는 절대 하지 말 것.
//
// ⚠ coef 는 raw각/모델각 공간(감속비 **이후**)의 계수다(§2-a). gear 를 푼 뒤에 적용해야 한다.
//   부호를 틀리면 상쇄가 아니라 **가중**된다(§2-b, 실측). 크기 1.0 은 육안 일치만 확인됐고
//   정량 실측은 없다(§9) — 도메인 랜덤화 후보.
constexpr float FOOT_CALF_COEF = 1.0f;
// policy 인덱스 p 의 커플링 소스(calf) policy 인덱스. -1 = 커플링 없음.
constexpr int COUPLED_CALF_POLICY[NUM_MOTORS] = {-1, -1, -1, -1, -1, -1, 4, 5};

struct MotorCalib {
    const char* name;  // "sim이름(실기이름)"
    float sign;        // 실기 회전 양방향 vs sim URDF 축 (+1/-1) — 실측 필요
    float zero_deg;    // sim q=0(중립) 일 때 실기 채널각 [deg] — 실측 필요
    float gear;        // 드라이버 감속비 오설정 보정 = 실제감속비/7, RL_INTERFACE.md §4.
                       // 채널각 = 관절각 × gear (hip/thigh 7:1 → 1.0, calf 10.5:1 → 1.5,
                       // foot 8.4:1 → 1.2). 위치·속도 모두에 걸린다.
    float kp;          // MIT 게인. 드라이버 범위 [0,500] (defineConfigMotor.h)
    float kd;          // MIT 게인. 드라이버 범위 [0,5]
    float min_rad;     // sim 좌표계 soft limit [rad] (motions.py SOFT_LIMITS_RAD)
    float max_rad;
};

// kp/kd = motions.py DEFAULT_KP/DEFAULT_KD 와 값 일치 필수 (중복 정의).
// 2026-08-12 실기팀 지정값: hip 100/5, thigh 50/5, calf 50/5, foot 20/5 — 전부 드라이버
// 상한(kp 500 / kd 5) 이내라 기동 시 클램프 WARN 이 나오지 않아야 정상.
// soft limit: 신규 CAD 리비전 Hind_Leg_URDF2 USD 기준 (2026-08-11). 좌우 동일 규약(미러 아님).
// 2026-08-12 SignFix: 실기 방향 실측에서 HL_hip/HL_calf/HL_foot/HR_hip/HR_thigh가 sim과
// 반대로 돌아 **sim 자산(Hind_Leg_URDF3_SignFix) 쪽 축을 반전**해 실기에 맞췄다. 따라서
// sign은 전 관절 +1 유지가 정답이고, 반전 관절의 sim 좌표 클램프(min/max)만 [lo,hi]→[-hi,-lo]로
// 뒤집혔다. zero_deg는 여전히 placeholder — probe 실측 후 채울 것.
// 2026-08-14 gear: 드라이버가 전 축 7:1 로 가정해 각도를 보고/수신하므로 채널각 = 관절각 × gear
// (실제감속비/7 — hip 7 · thigh 7 · calf 10.5 · foot 8.4). ⚠ 이 배열은 leg-major(모터 인덱스)
// 순서라 gear 는 {1.0, 1.0, 1.5, 1.2, 1.0, 1.0, 1.5, 1.2} 다. RL_INTERFACE.md §4.
constexpr MotorCalib MOTOR_CALIB[NUM_MOTORS] = {
    {"HL_hip(LtR)", +1.0f, 0.0f, 1.0f, 100.0f, 5.0f, -0.2340f, 0.2340f},   // 축 반전(대칭이라 값 동일)
    {"HL_thigh(LtP)", +1.0f, 0.0f, 1.0f, 50.0f, 5.0f, -0.9555f, 2.1855f},
    {"HL_calf(LkP)", +1.0f, 0.0f, 1.5f, 50.0f, 5.0f, -0.7745f, 0.9445f},   // 축 반전
    {"HL_foot(LaP)", +1.0f, 0.0f, 1.2f, 20.0f, 5.0f, -0.3440f, 1.3840f},   // 축 반전
    {"HR_hip(RtR)", +1.0f, 0.0f, 1.0f, 100.0f, 5.0f, -0.2340f, 0.2340f},   // 축 반전(대칭이라 값 동일)
    {"HR_thigh(RtP)", +1.0f, 0.0f, 1.0f, 50.0f, 5.0f, -2.1855f, 0.9555f},  // 축 반전 — thigh/calf/foot 좌우 미러
    {"HR_calf(RkP)", +1.0f, 0.0f, 1.5f, 50.0f, 5.0f, -0.9445f, 0.7745f},
    {"HR_foot(RaP)", +1.0f, 0.0f, 1.2f, 20.0f, 5.0f, -1.3840f, 0.3440f},
};

// ★GAIN_GEAR — 게인·토크의 gear 관계는 **2026-08-14 실측으로 확정**됐다(이전 "미판정" 서술 폐기).
//   단, 아래 kp/kd 에 변환을 **적용하지는 않았다** — 이유는 맨 아래.
//
// [실측] 준정적 구간에서 |τ_보고| / |kp·e_관절 − kd·q̇_관절| 을 재니 calf 에서 **1.5008** 이 나왔다
//   (gear 1.5 대비 오차 0.06%, gear² 2.25 대비 −33%). hip/thigh 대조군은 1.0006~1.0029.
//   ⇒ 드라이버 PD 가 **채널각 오차**에 게인을 곱한다: τ_drv = kp·(q_ch_des − q_ch) + kd·(Δdq_ch)
//                                                    = gear · (kp·e_관절 + kd·Δq̇_관절)
//   이 실측이 직접 말해주는 건 여기까지다 — τ_보고 가 gear 배로 세다는 것.
//
// [문서] RL_INTERFACE.md §4: 실제 관절토크 = **보고토크 × gear**(보고토크는 채널기준).
//
// [실측 + 문서 결합] 실효 관절강성 = kp · gear²  ⇒  calf 50 → 112.5(2.25배) · foot 30 → 43.2(1.44배).
//   감쇠비도 ζ×gear 로 변한다(calf 0.76 → 1.13 **과감쇠**). §4 서술과 일치한다.
//   목표 관절강성 kp_joint 를 내고 싶으면:  kp_ch = kp_joint / gear²   (kd 도 같은 식)
//   컨버터(sysid) 기본값이 이 결론을 따라 `GAIN_GEAR_SCALE = 2.0` 으로 확정됐고 gear¹ 대조군과
//   함께 재적합 중이다.
//
// ⚠ **그런데 아래 kp/kd 는 변환하지 않는다.** 이 값들은 2026-08-12 실기팀이 **실기에서 직접 고른
//   채널 게인**이라, 변환해 넣을 "sim 게인 원본"이 따로 없다. 즉 지금 값은 이미 실기 거동으로
//   고른 결과다. 변환은 학습 정책을 배포할 때 **"목표 관절강성 → 채널 게인"** 방향으로 정할
//   문제이고, 그때 위 `kp_ch = kp_joint / gear²` 를 쓴다.
//   ⇒ 드라이버가 고쳐져 gear 가 1 이 되면 calf·foot 이 갑자기 2.25·1.44배 약해진다(§4 경고).
//
// ⚠ tau 는 브리지가 **변환하지 않고 그대로 통과**시킨다(채널기준). 소비자가 `× gear` 하면
//   관절토크다(calf 1.5 · foot 1.2). 토크 트립 임계 15 Nm 도 채널기준이라 calf 는 실제 22.5 Nm다.
// ⚠ 커플링 **전치 되먹임**(τ_raw_calf −= coef·τ_foot, RL_INTERFACE §1)도 넣지 않았다.
//   MIT 모드로 `cmd.fTorque = 0` 을 보내는 한 피드포워드 토크가 없어 불필요하다.
//   **`fTorque ≠ 0` 을 쓰게 되면 그때 필요하다** — 각도처럼 목적축에 더하면 방향이 반대라 틀린다.

// 드라이버 게인 상한 (defineConfigMotor.h DEF_MOT_GAIN_*_MAX)
constexpr float DRV_GAIN_P_MAX = 500.0f;
constexpr float DRV_GAIN_D_MAX = 5.0f;

// RELAX 휴지(droop) 자세 [**관절각** rad, **policy(articulation) 순서**] — 2026-08-12 실기에서
// 무토크로 늘어뜨렸을 때의 자세를 [RELAX] 로그로 캡처한 값. staged relax(이 자세로 천천히 이동한
// 뒤 무토크)의 목표로 쓴다. ⚠ zero_deg placeholder 프레임의 실측값이라 MOTOR_CALIB min/max 클램프를
// **거치지 않고** 그대로 변환·명령한다(실측 droop 자세 = 물리적으로 도달 가능함이 정의상 보장 —
// 클램프하면 HL_calf +0.856 등이 잘려 자세가 왜곡된다). zero 캘리브레이션 후 재캡처할 것.
//
// ★이 상수는 **프레임이 두 번 바뀌었다.** 매번 "같은 물리 자세"를 유지하도록 재척도했고, 검증은
//   전부 "재척도값을 다시 채널각으로 환산하면 원래 실측 채널각이 나오는가" 로 했다:
//   (1) 2026-08-14 gear — 원 캡처 {…, +1.284, −0.940, +0.033, −0.032} 는 gear 보정 **이전**
//       프레임이라 calf ÷1.5 · foot ÷1.2 (hip/thigh 는 gear 1.0 이라 불변).
//   (2) 2026-08-14 커플링 — 위 (1) 결과의 foot 은 아직 **raw**각(q_f+q_c)이다. 브리지가 관절
//       좌표를 받게 되면서 foot 항도 **관절각**이어야 한다:
//         foot_joint = foot_raw − coef·calf_joint
//         HL: +0.0275 − (+0.856)   = −0.8285      HR: −0.0267 − (−0.6267) = +0.6000
// ⚠ 재척도된 foot 값은 MOTOR_CALIB 의 foot min/max([−0.344,+1.384] / [−1.384,+0.344]) **밖**이다.
//   이건 오류가 아니라 위 "클램프를 거치지 않는다" 규약이 왜 필요한지의 실증이다 — 무토크로
//   늘어뜨린 자세는 도달 가능하지만 공칭 soft limit 이 담지 못한다. **클램프 경로에 넣지 말 것.**
constexpr float RELAX_REST_POSE_SIM[NUM_MOTORS] = {+0.205f, -0.235f, +0.621f, -0.478f,
                                                   +0.856f, -0.6267f, -0.8285f, +0.6000f};
// staged relax 이동 속도 [rad/s] — 최대 관절 델타 / 이 값 = 램프 시간(최소 0.5 s).
// ⚠ 2026-08-14 gear 이후 델타가 **참 관절 rad** 로 계산된다(이전엔 calf 가 1.5배 부풀어 있었다).
//   값의 의미는 이제 정확히 관절 rad/s 지만, HOLD→droop 램프 시간이 calf 지배 구간에서
//   약 1.5배 짧아진다(예 2.57 s → 1.71 s). 낙하가 급하면 이 상수를 낮출 것.
constexpr float RELAX_RAMP_RADPS = 0.5f;
// staged relax 게인 페이드 시간 [s] — 휴지 자세 도달 후 kp/kd를 이 시간에 걸쳐 선형으로 0까지
// 내린다 (즉시 0으로 끊으면 잔여 중력 하중이 한 번에 풀리며 과도하게 처지는 실기 관찰, 2026-08-13).
constexpr float RELAX_FADE_SEC = 2.0f;

// IMU RPY 단위 — RobotTestGait 는 %6.1f 로 출력(도 단위로 추정). 실측으로 확정할 것.
constexpr bool IMU_RPY_IS_DEG = true;

#endif  // __CALIB_BIPEDLEG_HPP__
