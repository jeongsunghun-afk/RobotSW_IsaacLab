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

// kp/kd = motions.py DEFAULT_KP/DEFAULT_KD (sim 학습 게인, Nm/rad · Nm·s/rad 가정).
// ⚠ hip kd 6.0 은 드라이버 상한(DEF_MOT_GAIN_D_MAX=5)을 넘는다 — 런타임에서 5.0 으로
//   클램프하고 경고를 출력한다. sim(6.0)과 실기(5.0)의 plant 차이로 남는 항목.
constexpr MotorCalib MOTOR_CALIB[NUM_MOTORS] = {
    {"HL_hip(LtR)", +1.0f, 0.0f, 65.0f, 6.0f, -0.5498f, 0.5498f},
    {"HL_thigh(LtP)", +1.0f, 0.0f, 53.0f, 4.8f, -1.9024f, 1.5533f},
    {"HL_calf(LkP)", +1.0f, 0.0f, 12.0f, 1.1f, -1.3836f, 0.4236f},
    {"HL_foot(LaP)", +1.0f, 0.0f, 20.0f, 1.0f, -0.4192f, 1.4662f},
    {"HR_hip(RtR)", +1.0f, 0.0f, 65.0f, 6.0f, -0.5498f, 0.5498f},
    {"HR_thigh(RtP)", +1.0f, 0.0f, 53.0f, 4.8f, -1.9024f, 1.5533f},
    {"HR_calf(RkP)", +1.0f, 0.0f, 12.0f, 1.1f, -1.3836f, 0.4236f},
    {"HR_foot(RaP)", +1.0f, 0.0f, 20.0f, 1.0f, -1.4662f, 0.4192f},  // HL_foot 미러(의도된 비대칭)
};

// 드라이버 게인 상한 (defineConfigMotor.h DEF_MOT_GAIN_*_MAX)
constexpr float DRV_GAIN_P_MAX = 500.0f;
constexpr float DRV_GAIN_D_MAX = 5.0f;

// IMU RPY 단위 — RobotTestGait 는 %6.1f 로 출력(도 단위로 추정). 실측으로 확정할 것.
constexpr bool IMU_RPY_IS_DEG = true;

#endif  // __CALIB_BIPEDLEG_HPP__
