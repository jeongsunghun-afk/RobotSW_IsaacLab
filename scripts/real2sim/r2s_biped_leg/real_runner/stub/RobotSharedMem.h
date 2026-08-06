/**
 * RobotSharedMem.h (STUB) — 로컬(로봇 없는 환경) 빌드/루프백 테스트 전용.
 *
 * ⚠ 라즈베리파이 실빌드에서는 절대 이 파일이 include 되면 안 된다 —
 *   실기에는 /usr/include/RobotSharedMem.h (RobotSharedLib 설치본)가 있고,
 *   CMake 는 -DSTUB_SHM=ON 일 때만 이 디렉터리를 include path 에 추가한다.
 *
 * 함수 시그니처는 RobotTestGait/src/main.cpp 의 사용부에서 역추론했다.
 * 실헤더와 다르면 파이 빌드 시 컴파일 에러로 드러난다(런타임 오동작보다 안전).
 */
#ifndef __ROBOT_SHARED_MEM_STUB_H__
#define __ROBOT_SHARED_MEM_STUB_H__

// IMU 버퍼 레이아웃 — 실헤더 값과 다를 수 있음(스텁 전용 가정값).
#define IDX_OF_IMU_ForeC_START ((unsigned int)0)
#define IDX_OF_IMU_ARPY ((unsigned int)0)
#define IDX_OF_IMU_ACCL ((unsigned int)3)
#define LEN_OF_IMU_DATA ((unsigned int)6)

typedef struct STRUCT_MOTOR_PARAM16 {
    unsigned char ucRaw[20];  // MotGeneral_t(4×uchar + 8×float16) 와 동일 크기의 불투명 뷰
} MotorParam16_t;

#ifdef __cplusplus
extern "C" {
#endif

unsigned int RobotMemGait_InitComm(void);

unsigned char RobotMemGait_IsUpdatedMotorStatus16(void);
unsigned long RobotMemGait_GetUpdatedFlag_MotorStatus16(void);
unsigned int RobotMemGait_GetMotorStatus16(MotorParam16_t* ptrStatus, unsigned int unMotorIdx);
unsigned int RobotMemGait_SetMotorCommand16(MotorParam16_t* ptrCommand, unsigned int unMotorIdx);

unsigned char RobotMemGait_IsUpdatedIMU(void);
unsigned int RobotMemGait_GetIMU(float* ptrIMU, unsigned int unStartIdx, unsigned int unLength);

#ifdef __cplusplus
}
#endif

#endif  // __ROBOT_SHARED_MEM_STUB_H__
