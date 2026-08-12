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
 *    1. 드라이버가 전 축 7:1 가정으로 보고하므로 채널각 = 관절각 × gear_k
 *       (hip/thigh 1.0 · calf 1.5 · foot 1.2). 이 표의 sign/zero 선형변환만으로는
 *       calf·foot 배율을 표현할 수 없다 — gear 필드 추가 필요 (§4).
 *    2. foot 은 calf 와 기구 커플링 (q_raw_foot = q_foot + 1.0·q_calf, §2).
 *       sign/zero/gear 만으로는 표현 불가 — 커플링 항 추가 필요. 토크는 전치로 되먹임.
 *    3. 채널각은 ±180° 래핑된다(클램프 아님) — 송신 전 포화 필수 (§5).
 */
#ifndef __CALIB_BIPEDLEG_HPP__
#define __CALIB_BIPEDLEG_HPP__

constexpr int NUM_MOTORS = 8;

// 정책 인덱스 p → 실기 모터 인덱스 (articulation type-major → leg-major)
constexpr int POLICY_TO_MOTOR[NUM_MOTORS] = {0, 4, 1, 5, 2, 6, 3, 7};

struct MotorCalib {
    const char* name;  // "sim이름(실기이름)"
    float sign;        // 실기 회전 양방향 vs sim URDF 축 (+1/-1) — 실측 필요
    float zero_deg;    // sim q=0(중립) 일 때 실기 모터각 [deg] — 실측 필요
    float kp;          // MIT 게인. 드라이버 범위 [0,500] (defineConfigMotor.h)
    float kd;          // MIT 게인. 드라이버 범위 [0,5]
    float min_rad;     // sim 좌표계 soft limit [rad] (motions.py SOFT_LIMITS_RAD)
    float max_rad;
};

// kp/kd = motions.py DEFAULT_KP/DEFAULT_KD 와 값 일치 필수 (중복 정의).
// 2026-08-12 실기팀 지정값: hip 100/5, thigh 50/5, calf 50/5, foot 20/5 — 전부 드라이버
// 상한(kp 500 / kd 5) 이내라 기동 시 클램프 WARN 이 나오지 않아야 정상.
// soft limit: 신규 CAD 리비전 Hind_Leg_URDF2 USD 기준 (2026-08-11). 좌우 동일 규약(미러 아님).
constexpr MotorCalib MOTOR_CALIB[NUM_MOTORS] = {
    {"HL_hip(LtR)", +1.0f, 0.0f, 100.0f, 5.0f, -0.2340f, 0.2340f},
    {"HL_thigh(LtP)", +1.0f, 0.0f, 50.0f, 5.0f, -0.9555f, 2.1855f},
    {"HL_calf(LkP)", +1.0f, 0.0f, 50.0f, 5.0f, -0.9445f, 0.7745f},
    {"HL_foot(LaP)", +1.0f, 0.0f, 20.0f, 5.0f, -1.3840f, 0.3440f},
    {"HR_hip(RtR)", +1.0f, 0.0f, 100.0f, 5.0f, -0.2340f, 0.2340f},
    {"HR_thigh(RtP)", +1.0f, 0.0f, 50.0f, 5.0f, -0.9555f, 2.1855f},  // HL과 동일 (URDF2 비미러)
    {"HR_calf(RkP)", +1.0f, 0.0f, 50.0f, 5.0f, -0.9445f, 0.7745f},
    {"HR_foot(RaP)", +1.0f, 0.0f, 20.0f, 5.0f, -1.3840f, 0.3440f},  // HL과 동일 (URDF2 비미러)
};

// 드라이버 게인 상한 (defineConfigMotor.h DEF_MOT_GAIN_*_MAX)
constexpr float DRV_GAIN_P_MAX = 500.0f;
constexpr float DRV_GAIN_D_MAX = 5.0f;

// IMU RPY 단위 — RobotTestGait 는 %6.1f 로 출력(도 단위로 추정). 실측으로 확정할 것.
constexpr bool IMU_RPY_IS_DEG = true;

#endif  // __CALIB_BIPEDLEG_HPP__
