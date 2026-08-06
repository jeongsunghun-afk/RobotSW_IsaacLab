/**
 * RobotSharedMem_stub.cpp — 로컬 루프백 테스트용 가짜 plant.
 *
 * 명령 위치를 1차 지연(시정수 ~30 ms)으로 따라가는 에코 plant. IMU 는 직립(RPY=0) 고정.
 * real_runner 의 UDP 파싱·인덱스 매핑·deg/rad 변환 왕복을 로봇 없이 검증하는 용도다.
 */
#include <cstring>

#include "RobotSharedMem.h"

#include "float16_compat.hpp"
#include "../../RobotTestGait/inc/define/defineGeneral.h"
#include "../../RobotTestGait/inc/define/defineConfigMotor.h"

namespace {
constexpr int kNumMotors = 8;
MotGeneral_t g_cmd[kNumMotors];
float g_pos_deg[kNumMotors] = {0};
bool g_has_cmd[kNumMotors] = {false};
}  // namespace

extern "C" {

unsigned int RobotMemGait_InitComm(void) {
    std::memset(g_cmd, 0, sizeof(g_cmd));
    return ENUM_RESULT_SUCCESS;
}

unsigned char RobotMemGait_IsUpdatedMotorStatus16(void) { return 1; }

unsigned long RobotMemGait_GetUpdatedFlag_MotorStatus16(void) { return 0xFFul; }

unsigned int RobotMemGait_GetMotorStatus16(MotorParam16_t* ptrStatus, unsigned int unMotorIdx) {
    if (unMotorIdx >= kNumMotors) {
        return ENUM_RESULT_FAILURE;
    }
    // 1차 지연 plant: pos += (target - pos) * alpha (호출은 1 kHz 가정)
    if (g_has_cmd[unMotorIdx]) {
        float target = static_cast<float>(g_cmd[unMotorIdx].fPosition);
        g_pos_deg[unMotorIdx] += (target - g_pos_deg[unMotorIdx]) * 0.03f;
    }
    MotGeneral_t stt;
    std::memset(&stt, 0, sizeof(stt));
    stt.ucDevID = static_cast<unsigned char>(unMotorIdx);
    stt.fPosition = static_cast<float16>(g_pos_deg[unMotorIdx]);
    stt.fVelocity = static_cast<float16>(0.0f);
    std::memcpy(ptrStatus, &stt, sizeof(MotGeneral_t));
    return ENUM_RESULT_SUCCESS;
}

unsigned int RobotMemGait_SetMotorCommand16(MotorParam16_t* ptrCommand, unsigned int unMotorIdx) {
    if (unMotorIdx >= kNumMotors) {
        return ENUM_RESULT_FAILURE;
    }
    std::memcpy(&g_cmd[unMotorIdx], ptrCommand, sizeof(MotGeneral_t));
    g_has_cmd[unMotorIdx] = true;
    return ENUM_RESULT_SUCCESS;
}

unsigned char RobotMemGait_IsUpdatedIMU(void) { return 1; }

unsigned int RobotMemGait_GetIMU(float* ptrIMU, unsigned int unStartIdx, unsigned int unLength) {
    (void)unStartIdx;
    for (unsigned int i = 0; i < unLength; i++) {
        ptrIMU[i] = 0.0f;  // RPY=0(직립), accel=0
    }
    return ENUM_RESULT_SUCCESS;
}

}  // extern "C"
