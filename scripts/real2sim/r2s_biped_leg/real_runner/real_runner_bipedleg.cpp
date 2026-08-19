/**
 * real_runner_bipedleg — r2s_biped_leg 실기(real) 엔드포인트 브리지.
 *
 * 라즈베리파이에서 RobotEmbedded(EtherCAT master)와 Shared Memory 로 모터를 교환하고,
 * 워크스테이션의 policy_runner_bipedleg.py 와 UDP seam 으로 연결한다:
 *
 *   policy_runner ──POLICY_ACT("R2PA", 9887)──▶ real_runner ──SHM──▶ RobotEmbedded ──▶ 모터
 *   policy_runner ◀──POLICY_STATE("R2PS", 9888)── real_runner ◀──SHM── 모터상태 + IMU
 *
 * ★좌표 규약 (2026-08-14, TELEM convention_version=1):
 *   UDP seam 양쪽은 **관절(모델) 좌표**만 주고받는다. 채널각↔관절각 변환(감속비 오설정 보정
 *   `gear` + foot↔calf 커플링 해제)은 **이 브리지가 하드웨어 경계에서 한 번만** 한다.
 *   ⇒ 워크스테이션(policy_runner·GUI·env)에 raw 좌표가 존재해서는 안 된다.
 *   변환 지점은 motor_deg_to_joint / motor_dps_to_joint / joint_to_motor_deg 셋뿐이다.
 *   예외: `tau` 는 드라이버 보고값 그대로다(**채널기준** — 소비자가 ×gear 하면 관절토크.
 *   calib 의 GAIN_GEAR 블록 참조).
 *
 * 제어 구조 (RobotTestGait 예제 준수):
 *  - 1 ms 주기 루프. 모터 상태는 IsUpdatedMotorStatus16 → GetUpdatedFlag → GetMotorStatus16.
 *  - 모터 상태 100 회 수신 후에만 명령 전송 enable (RobotEmbedded 기동 확인 게이트).
 *  - 명령은 MotGeneral_t (ucMode=1, pos[deg], kp/kd) → SetMotorCommand16.
 *
 * 페이싱 (중요): sim seam 은 lockstep 이라 sim 이 시간을 소유하지만, 실기는 실시간이므로
 * **브리지가 STATE 회신을 20 ms(STEP_DT) 간격으로 페이싱**해서 policy 루프를 50 Hz 로 묶는다.
 * 회신을 즉시 보내면 policy 가 kHz 로 자유질주해 gait clock 이 실시간의 수십 배로 돈다.
 *
 * 모드:
 *   --probe : 명령 전송 없이 관절각/IMU 만 2 Hz 출력 (sign/zero 캘리브레이션용)
 *   --hold  : 현재 자세 latch 후 위치유지 명령만 전송 (액추에이션 안전 확인용)
 *   (기본)  : HOLD 로 시작, 첫 ACT 수신 시 engage ramp 후 TRACK
 *
 * 빌드: 파이에서 CMake (libRobotSharedMem 링크). 로컬 검증은 -DSTUB_SHM=ON (stub/ 참조).
 */

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#ifdef STUB_SHM
#include "stub/float16_compat.hpp"  // _Float16 미지원 로컬 컴파일러용 shim (파이에선 no-op)
#endif

#include "../RobotTestGait/inc/define/defineGeneral.h"
#include "../RobotTestGait/inc/define/defineConfigMotor.h"
#include <RobotSharedMem.h>

#include "calib_bipedleg.hpp"
#include "r2s_packets.hpp"

static_assert(NUM_MOTORS == R2S_NUM_JOINTS, "motor/packet joint count mismatch");

namespace {

constexpr double kLoopDtSec = 0.001;      // 1 ms — RobotTestGait 와 동일
constexpr double kStepDtSec = 0.020;      // STATE 회신 페이싱 = 학습 STEP_DT(50 Hz)
// TELEM 페이싱 — **STATE 와 독립**. sysid 캡처 해상도를 정하는 값이라 sim 물리 그리드에 맞춘다
// (`SYSID_RATE_HZ = GRID_HZ = 200 Hz`) → 컨버터의 리샘플링이 사실상 사라진다.
// ⚠ 2026-08-19 이전에는 TELEM 이 kStepDtSec 게이트 안에 있어 50 Hz 였고, 캡처의 t_real 이
//   48.1~48.7 Hz 로 기록됐다. 그 해상도로는 chirp 을 5 Hz 로 올렸을 때 선형보간 오차가
//   진폭의 5.2 % 에 달해(= 현재 잔차 RMS 전체와 맞먹음) armature 를 식별할 수 없다.
//   200 Hz 면 0.31 % 로 떨어진다. 근거: reports/_comparisons/pace_bipedleg_foot_coupling_probe
//   README §23-j.
// ★이 값을 kStepDtSec 과 엮지 말 것 — STATE 는 policy_runner 가 lockstep 이라 20 ms 가
//   50 Hz 정책 클록을 소유한다. 빠르게 만들면 gait clock 이 자유질주한다.
constexpr double kTelemDtSec = 0.005;     // TELEM 송신 페이싱 = 200 Hz (관측 전용, lockstep 무관)
constexpr double kActTimeoutSec = 0.5;    // TRACK 중 ACT 두절 → 마지막 목표로 hold
constexpr unsigned kStatusWarmupCnt = 100;  // RobotTestGait 게이트와 동일

enum class Mode { kProbe, kHold, kBridge };
enum class State { kWaitStatus, kHold, kEngage, kTrack, kRelaxRamp, kRelaxFade, kRelax };

double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

// sim 좌표(raw각) [rad] → 실기 채널각 [deg].  RL_INTERFACE.md §1: q_ch = q_raw·sign·gear + offset
// gear 는 드라이버 감속비 오설정 보정(§4) — offset(zero_deg)은 채널각 단위라 **곱한 뒤** 더한다.
float sim_to_motor_deg(int m, float q_rad) {
    return MOTOR_CALIB[m].sign * q_rad * DEF_RAD2DEG * MOTOR_CALIB[m].gear + MOTOR_CALIB[m].zero_deg;
}
// 실기 채널각 [deg] → sim 좌표(raw각) [rad].  §1: q_raw = (q_ch − offset)/(sign·gear)
// offset 이 채널각 단위이므로 gear 로 나누기 **전에** 뺀다 (sign 은 ±1 이라 곱=나눗셈).
float motor_deg_to_sim(int m, float pos_deg) {
    return MOTOR_CALIB[m].sign * (pos_deg - MOTOR_CALIB[m].zero_deg) * DEF_DEG2RAD / MOTOR_CALIB[m].gear;
}
// 실기 채널 각속도 [deg/s] → sim raw 각속도 [rad/s]. 위치와 같은 선형관계에서 상수항만 빠진다(§1).
float motor_dps_to_sim(int m, float vel_dps) {
    return MOTOR_CALIB[m].sign * vel_dps * DEF_DEG2RAD / MOTOR_CALIB[m].gear;
}

// ---- 커플링까지 포함한 벡터 변환 (하드웨어 경계에서 딱 한 번) ----
// 위 스칼라 함수들은 채널↔**raw** 까지만 한다. 아래 두 함수가 foot↔calf 커플링을 마저 풀어
// **관절(모델) 좌표**로 오간다. 워크스테이션은 관절 좌표만 보므로 이 경계 밖에 커플링 코드가
// 있어서는 안 된다.
//
// ⚠⚠ **커플링 연산은 policy(articulation) 순서 벡터에서만 한다.** 짝이 p-2 인 것은 policy
//   순서에서만 성립하고(calf 4,5 / foot 6,7), 모터 순서(calf 2,6 / foot 3,7)에서는 규칙이
//   다르다. calib 의 COUPLED_CALF_POLICY 주석 참조.
// ⚠ 클램프하지 않는다 — RELAX 휴지 자세가 공칭 soft limit 밖이라(calib 참조) 여기서 자르면
//   자세가 왜곡된다. 클램프는 TRACK 목표 생성부에서만 한다.

// 채널각 [deg, **모터 순서**] → 관절각 [rad, **policy 순서**]
void motor_deg_to_joint(const float ch_deg[NUM_MOTORS], float q_joint[R2S_NUM_JOINTS]) {
    float q_raw[R2S_NUM_JOINTS];
    for (int p = 0; p < R2S_NUM_JOINTS; p++) {
        int m = POLICY_TO_MOTOR[p];
        q_raw[p] = motor_deg_to_sim(m, ch_deg[m]);
    }
    for (int p = 0; p < R2S_NUM_JOINTS; p++) {
        int pc = COUPLED_CALF_POLICY[p];
        // calf 는 커플링이 없어 q_raw[pc] == q_joint[pc] 다 — 그래서 raw 배열에서 바로 뺄 수 있다.
        q_joint[p] = (pc < 0) ? q_raw[p] : q_raw[p] - FOOT_CALF_COEF * q_raw[pc];
    }
}

// 채널 각속도 [deg/s, **모터 순서**] → 관절 각속도 [rad/s, **policy 순서**] (위와 같은 선형관계)
void motor_dps_to_joint(const float ch_dps[NUM_MOTORS], float dq_joint[R2S_NUM_JOINTS]) {
    float dq_raw[R2S_NUM_JOINTS];
    for (int p = 0; p < R2S_NUM_JOINTS; p++) {
        int m = POLICY_TO_MOTOR[p];
        dq_raw[p] = motor_dps_to_sim(m, ch_dps[m]);
    }
    for (int p = 0; p < R2S_NUM_JOINTS; p++) {
        int pc = COUPLED_CALF_POLICY[p];
        dq_joint[p] = (pc < 0) ? dq_raw[p] : dq_raw[p] - FOOT_CALF_COEF * dq_raw[pc];
    }
}

// 관절각 [rad, **policy 순서**] → 채널각 [deg, **모터 순서**]
// ⚠ 커플링 되먹임은 **목표값끼리** 합성한다(측정 calf 가 아니다) — 학습 env 가 그렇게 한다:
//   hind_leg_env.py:267 `raw_t = processed[foot_ids] + processed[calf_ids]`.
//   측정값을 쓰면 추종 오차·지연만큼 sim 과 실기의 raw 목표가 갈린다.
void joint_to_motor_deg(const float q_joint[R2S_NUM_JOINTS], float ch_deg[NUM_MOTORS]) {
    for (int p = 0; p < R2S_NUM_JOINTS; p++) {
        int m = POLICY_TO_MOTOR[p];
        int pc = COUPLED_CALF_POLICY[p];
        float q_raw = (pc < 0) ? q_joint[p] : q_joint[p] + FOOT_CALF_COEF * q_joint[pc];
        ch_deg[m] = sim_to_motor_deg(m, q_raw);
    }
}

float clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // 리다이렉트/ssh 로그 유실 방지
    Mode mode = Mode::kBridge;
    int act_port = REAL_ACT_PORT;
    float slew_dps = 0.0f;      // 0 = slew 제한 없음 (학습 plant 와 일치). 브링업 시 예: 180
    double engage_sec = 0.5;    // HOLD→TRACK 진입 램프 길이

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--probe") == 0) {
            mode = Mode::kProbe;
        } else if (std::strcmp(argv[i], "--hold") == 0) {
            mode = Mode::kHold;
        } else if (std::strcmp(argv[i], "--act_port") == 0 && i + 1 < argc) {
            act_port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--slew_dps") == 0 && i + 1 < argc) {
            slew_dps = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--engage_ms") == 0 && i + 1 < argc) {
            engage_sec = std::atof(argv[++i]) * 1e-3;
        } else {
            std::printf("usage: %s [--probe|--hold] [--act_port N] [--slew_dps X] [--engage_ms N]\n", argv[0]);
            return 1;
        }
    }

    // ---- 게인 검증 (드라이버 상한 클램프) ----
    float kp_cmd[NUM_MOTORS];
    float kd_cmd[NUM_MOTORS];
    for (int m = 0; m < NUM_MOTORS; m++) {
        kp_cmd[m] = clamp(MOTOR_CALIB[m].kp, 0.0f, DRV_GAIN_P_MAX);
        kd_cmd[m] = clamp(MOTOR_CALIB[m].kd, 0.0f, DRV_GAIN_D_MAX);
        if (kp_cmd[m] != MOTOR_CALIB[m].kp || kd_cmd[m] != MOTOR_CALIB[m].kd) {
            std::printf("[real_runner] WARN %s: gain clamp kp %.1f->%.1f kd %.1f->%.1f (드라이버 상한)\n",
                        MOTOR_CALIB[m].name, MOTOR_CALIB[m].kp, kp_cmd[m], MOTOR_CALIB[m].kd, kd_cmd[m]);
        }
    }

    // ---- UDP ----
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::perror("socket");
        return 1;
    }
    sockaddr_in bind_addr{};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    bind_addr.sin_port = htons(static_cast<uint16_t>(act_port));
    if (bind(sock, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0) {
        std::perror("bind");
        return 1;
    }
    fcntl(sock, F_SETFL, O_NONBLOCK);

    // ---- Shared Memory ----
    unsigned int res = RobotMemGait_InitComm();
    std::printf("[real_runner] RobotMemGait_InitComm %x  mode=%s  act_port=%d\n", res,
                mode == Mode::kProbe ? "PROBE" : (mode == Mode::kHold ? "HOLD" : "BRIDGE"), act_port);
    if (res != ENUM_RESULT_SUCCESS) {
        std::printf("[real_runner] SHM init 실패 — RobotEmbedded 가 먼저 떠 있어야 한다 (sudo ./src/RobotEmbedded)\n");
        return 1;
    }

    // ---- 상태 변수 ----
    MotGeneral_t motor_stt[NUM_MOTORS];
    std::memset(motor_stt, 0, sizeof(motor_stt));
    bool stt_valid[NUM_MOTORS] = {false};
    unsigned status_rx_cnt = 0;
    bool transmit_ena = false;

    float imu_buf[LEN_OF_IMU_DATA] = {0};
    bool imu_seen = false;

    float target_pol[R2S_NUM_JOINTS] = {0};   // 최신 policy 목표 [rad], articulation 순서
    float latched_deg[NUM_MOTORS] = {0};      // HOLD/engage 기준 자세 [deg], 모터 순서
    float cmd_deg[NUM_MOTORS] = {0};          // 이번 틱 전송 명령 [deg]
    float prev_cmd_deg[NUM_MOTORS] = {0};
    bool have_prev_cmd = false;

    State state = State::kWaitStatus;
    double engage_t0 = 0.0;
    // staged relax: 휴지(droop) 자세로 램프 후 무토크 (RELAX_REST_POSE_SIM, calib 참조)
    float relax_start_deg[NUM_MOTORS] = {0};  // 램프 시작 자세 [deg]
    float relax_rest_deg[NUM_MOTORS] = {0};   // 휴지 자세 [deg] (클램프 없이 변환 — calib 주석 참조)
    double relax_t0 = 0.0;
    double relax_ramp_sec = 1.0;
    double relax_fade_t0 = 0.0;  // 게인 페이드 시작 시각
    double last_act_time = -1.0;
    double last_reply_time = -1.0;
    double last_telem_time = -1.0;  // TELEM 은 STATE 와 별도 페이싱 (kTelemDtSec)
    double last_print_time = 0.0;
    uint32_t last_act_seq = 0;
    sockaddr_in peer_addr{};
    bool have_peer = false;

    timespec next_tick;
    clock_gettime(CLOCK_MONOTONIC, &next_tick);

    std::printf("[real_runner] loop start (1 kHz, STATE pacing %.0f ms = %.0f Hz, TELEM:%d @ %.0f Hz, PING 지원)\n",
                kStepDtSec * 1e3, 1.0 / kStepDtSec, REAL_TELEM_PORT, 1.0 / kTelemDtSec);

    while (true) {
        // ---- 1) 모터 상태 읽기 (RobotTestGait 패턴) ----
        if (RobotMemGait_IsUpdatedMotorStatus16() == 1) {
            unsigned long flag = RobotMemGait_GetUpdatedFlag_MotorStatus16();
            MotGeneral_t stt;
            for (unsigned m = 0; m < NUM_MOTORS; m++) {
                if ((flag & (1ul << m)) != 0) {
                    if (RobotMemGait_GetMotorStatus16(reinterpret_cast<MotorParam16_t*>(&stt), m)
                        == ENUM_RESULT_SUCCESS) {
                        std::memcpy(&motor_stt[m], &stt, sizeof(MotGeneral_t));
                        stt_valid[m] = true;
                    }
                }
            }
            if (status_rx_cnt < kStatusWarmupCnt + 1) {
                status_rx_cnt++;
            }
            transmit_ena = status_rx_cnt > kStatusWarmupCnt;
        }

        // ---- 2) IMU 읽기 ----
        if (RobotMemGait_IsUpdatedIMU() == 1) {
            float buf[LEN_OF_IMU_DATA];
            if (RobotMemGait_GetIMU(&buf[0], IDX_OF_IMU_ForeC_START, LEN_OF_IMU_DATA) == ENUM_RESULT_SUCCESS) {
                std::memcpy(imu_buf, buf, sizeof(imu_buf));
                imu_seen = true;
            }
        }

        // ---- 3) UDP ACT drain (latest-wins) ----
        {
            uint8_t rx[256];
            sockaddr_in from{};
            socklen_t from_len = sizeof(from);
            ssize_t n;
            while ((n = recvfrom(sock, rx, sizeof(rx), 0, reinterpret_cast<sockaddr*>(&from), &from_len)) > 0) {
                PolicyActPacket act;
                PolicyPingPacket ping;
                PolicyGainPacket gain;
                if (unpack_policy_act(rx, static_cast<size_t>(n), act)) {
                    std::memcpy(target_pol, act.target_q, sizeof(target_pol));
                    last_act_seq = act.seq;
                    last_act_time = now_sec();
                    peer_addr = from;
                    have_peer = true;
                    if ((state == State::kHold || state == State::kRelax || state == State::kRelaxRamp
                         || state == State::kRelaxFade)
                        && mode == Mode::kBridge) {
                        // RELAX(램프 포함)에서 복귀 시엔 처져 있는 **현재** 자세에서 램프를 시작해야 한다.
                        if (state == State::kRelax || state == State::kRelaxRamp || state == State::kRelaxFade) {
                            for (int m = 0; m < NUM_MOTORS; m++) {
                                latched_deg[m] = static_cast<float>(motor_stt[m].fPosition);
                            }
                        }
                        state = State::kEngage;
                        engage_t0 = last_act_time;
                        std::printf("[real_runner] ACT 수신(seq=%u) → ENGAGE %.0f ms\n", act.seq, engage_sec * 1e3);
                    }
                } else if (unpack_policy_ping(rx, static_cast<size_t>(n), ping)) {
                    // 모니터 keepalive: peer 만 등록. 목표/상태머신 불변 — bridge TRACK 중에도 무해.
                    peer_addr = from;
                    have_peer = true;
                } else if (unpack_policy_relax(rx, static_cast<size_t>(n), ping)) {
                    // 무토크(limp) 요청 — bridge 모드에서만. kp=kd=tau=0 을 능동 송신한다(명령 중단이
                    // 아님 — 드라이버가 마지막 명령을 유지할 수 있으므로 zero-torque 를 계속 보낸다).
                    peer_addr = from;
                    have_peer = true;
                    // staged relax (2026-08-12): 즉시 무토크가 아니라 실측 휴지(droop) 자세로
                    // RELAX_RAMP_RADPS 속도로 이동한 뒤 무토크로 전환한다 — 높은 자세에서 바로
                    // 풀면 낙하 충격이 있기 때문. RELAX 패킷은 50Hz로 반복 수신되므로 램프/무토크
                    // 중에는 재트리거하지 않는다.
                    if (mode == Mode::kBridge && state != State::kRelax && state != State::kRelaxRamp
                        && state != State::kRelaxFade && state != State::kWaitStatus) {
                        // RELAX_REST_POSE_SIM 은 **관절각**(policy 순서)이므로 커플링 되먹임까지
                        // 하는 joint_to_motor_deg 로 변환한다. 클램프는 거치지 않는다 — 휴지
                        // 자세가 공칭 soft limit 밖이라 자르면 왜곡된다(calib 주석 참조).
                        for (int m = 0; m < NUM_MOTORS; m++) {
                            relax_start_deg[m] = static_cast<float>(motor_stt[m].fPosition);
                        }
                        joint_to_motor_deg(RELAX_REST_POSE_SIM, relax_rest_deg);
                        // 램프 길이는 **관절 좌표**에서 잰다 — 목표도 현재자세도 같은 프레임이어야 한다.
                        float cur_joint[R2S_NUM_JOINTS];
                        motor_deg_to_joint(relax_start_deg, cur_joint);
                        float max_delta_rad = 0.0f;
                        for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                            float d = std::fabs(RELAX_REST_POSE_SIM[p] - cur_joint[p]);
                            if (d > max_delta_rad) max_delta_rad = d;
                        }
                        relax_ramp_sec = max_delta_rad / RELAX_RAMP_RADPS;
                        if (relax_ramp_sec < 0.5) relax_ramp_sec = 0.5;
                        relax_t0 = now_sec();
                        state = State::kRelaxRamp;
                        std::printf("[real_runner] RELAX 수신 → 휴지 자세로 램프 %.1f s (max Δ %.2f rad) 후 무토크\n",
                                    relax_ramp_sec, max_delta_rad);
                    }
                } else if (unpack_policy_gain(rx, static_cast<size_t>(n), gain)) {
                    // kp/kd 런타임 갱신 — articulation 순서로 받아 모터 순서로 매핑, 드라이버 상한 클램프.
                    // 다음 틱부터 kp_cmd/kd_cmd 로 반영된다(relax 중엔 여전히 0 강제). 목표/상태머신 불변.
                    peer_addr = from;
                    have_peer = true;
                    bool changed = false;
                    for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                        int m = POLICY_TO_MOTOR[p];
                        float new_kp = clamp(gain.kp[p], 0.0f, DRV_GAIN_P_MAX);
                        float new_kd = clamp(gain.kd[p], 0.0f, DRV_GAIN_D_MAX);
                        if (new_kp != kp_cmd[m] || new_kd != kd_cmd[m]) {
                            changed = true;
                        }
                        kp_cmd[m] = new_kp;
                        kd_cmd[m] = new_kd;
                    }
                    // 값이 실제로 바뀌었을 때만 로그 — 워크스테이션이 1 Hz 로 갱신 송신하므로
                    // 매번 찍으면 콘솔이 초당 1줄씩 오염된다.
                    if (changed) {
                        std::printf("[real_runner] GAIN 수신(seq=%u) kp[0]=%.1f kd[0]=%.1f (모터순서, 클램프 후)\n",
                                    gain.seq, kp_cmd[0], kd_cmd[0]);
                    }
                }
                from_len = sizeof(from);
            }
        }

        const double t = now_sec();
        const bool all_stt = stt_valid[0] && stt_valid[1] && stt_valid[2] && stt_valid[3] && stt_valid[4]
                             && stt_valid[5] && stt_valid[6] && stt_valid[7];

        // ---- 4) 상태머신 / 명령 생성 ----
        if (state == State::kWaitStatus) {
            if (transmit_ena && all_stt) {
                for (int m = 0; m < NUM_MOTORS; m++) {
                    latched_deg[m] = static_cast<float>(motor_stt[m].fPosition);
                }
                state = State::kHold;
                std::printf("[real_runner] 모터 상태 warmup 완료 → HOLD (현재 자세 latch)\n");
            }
        }

        if (state != State::kWaitStatus && mode != Mode::kProbe) {
            const bool relax_now = (state == State::kRelax);
            if (relax_now) {
                // 무토크: 목표=현재 자세(게인 0이라 사실상 무의미), slew 무관하게 현재값 추종.
                for (int m = 0; m < NUM_MOTORS; m++) {
                    cmd_deg[m] = static_cast<float>(motor_stt[m].fPosition);
                }
            } else if (state == State::kRelaxRamp) {
                // staged relax: 실측 휴지 자세로 선형 램프 (게인은 정상 유지 — 낙하 아님, 이동).
                // soft limit 클램프 없음 — calib RELAX_REST_POSE_SIM 주석 참조.
                double t_now = now_sec();
                float a = static_cast<float>(clamp(static_cast<float>((t_now - relax_t0) / relax_ramp_sec), 0.0f, 1.0f));
                for (int m = 0; m < NUM_MOTORS; m++) {
                    cmd_deg[m] = relax_start_deg[m] + (relax_rest_deg[m] - relax_start_deg[m]) * a;
                }
                if (a >= 1.0f) {
                    // 즉시 무토크로 끊지 않는다 — 잔여 중력 하중이 한 번에 풀리며 과도하게 처지는
                    // 실기 관찰(2026-08-13). 게인을 RELAX_FADE_SEC에 걸쳐 선형으로 0까지 내린다.
                    state = State::kRelaxFade;
                    relax_fade_t0 = now_sec();
                    std::printf("[real_runner] 휴지 자세 도달 → 게인 페이드 %.1f s 후 무토크\n",
                                static_cast<double>(RELAX_FADE_SEC));
                }
            } else if (state == State::kRelaxFade) {
                // 게인 페이드: 휴지 자세를 목표로 유지한 채 kp/kd만 서서히 0으로 (아래 gain_scale).
                for (int m = 0; m < NUM_MOTORS; m++) {
                    cmd_deg[m] = relax_rest_deg[m];
                }
                if (now_sec() - relax_fade_t0 >= RELAX_FADE_SEC) {
                    state = State::kRelax;
                    std::printf("[real_runner] 게인 페이드 완료 → 무토크(limp)\n");
                }
            } else if (state == State::kHold) {
                for (int m = 0; m < NUM_MOTORS; m++) {
                    cmd_deg[m] = latched_deg[m];
                }
            } else {
                // TRACK 목표: 정책이 주는 **관절각**을 그대로 클램프한 뒤 채널각으로 변환한다.
                // 커플링 되먹임(raw = foot + calf)은 joint_to_motor_deg 가 **목표값끼리** 한다.
                // ⇒ 예전의 `lo += qc`(측정 calf 만큼 한계를 평행이동) 는 필요 없어졌다: 목표가
                //   이제 관절각이므로 MOTOR_CALIB 의 min/max 가 그대로 맞는 한계다.
                float track_deg[NUM_MOTORS];
                float q_cmd[R2S_NUM_JOINTS];
                for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                    int m = POLICY_TO_MOTOR[p];
                    q_cmd[p] = clamp(target_pol[p], MOTOR_CALIB[m].min_rad, MOTOR_CALIB[m].max_rad);
                }
                joint_to_motor_deg(q_cmd, track_deg);
                if (state == State::kEngage) {
                    float a = static_cast<float>(clamp(static_cast<float>((t - engage_t0) / engage_sec), 0.0f, 1.0f));
                    for (int m = 0; m < NUM_MOTORS; m++) {
                        cmd_deg[m] = latched_deg[m] + (track_deg[m] - latched_deg[m]) * a;
                    }
                    if (a >= 1.0f) {
                        state = State::kTrack;
                        std::printf("[real_runner] ENGAGE 완료 → TRACK\n");
                    }
                } else {  // kTrack
                    if (t - last_act_time > kActTimeoutSec) {
                        // ACT 두절: 마지막 명령 유지 (전송은 계속 — PD 위치 hold)
                    } else {
                        for (int m = 0; m < NUM_MOTORS; m++) {
                            cmd_deg[m] = track_deg[m];
                        }
                    }
                }
            }

            // 선택적 slew 제한 (기본 off — 학습 plant 와 일치시키려면 끈 상태로 검증).
            // relax 중엔 목표가 현재 자세라 제한이 무의미하고, 처지는 속도를 따라가지 못하면
            // 오히려 유해 → 스킵.
            // ⚠ slew_dps 는 **채널** deg/s 라 gear 가 곱해진 축에선 관절 기준으로 더 빡세다
            // (calf 180 dps → 관절 120 dps · foot 180 → 150). 기본 0(off)이라 현재 영향 없음.
            if (slew_dps > 0.0f && have_prev_cmd && !relax_now) {
                float max_step = slew_dps * static_cast<float>(kLoopDtSec);
                for (int m = 0; m < NUM_MOTORS; m++) {
                    cmd_deg[m] = clamp(cmd_deg[m], prev_cmd_deg[m] - max_step, prev_cmd_deg[m] + max_step);
                }
            }

            // ---- 5) 명령 전송 (RobotTestGait 패턴: MotGeneral_t, ucMode=1) ----
            if (transmit_ena) {
                for (unsigned m = 0; m < NUM_MOTORS; m++) {
                    MotGeneral_t cmd;
                    std::memset(&cmd, 0, sizeof(cmd));
                    cmd.ucDevID = static_cast<unsigned char>(m);
                    cmd.ucMode = 1;
                    cmd.ucCommand = 0;
                    cmd.fPosition = static_cast<float16>(cmd_deg[m]);
                    cmd.fVelocity = static_cast<float16>(0.0f);
                    cmd.fAccelrationOrTemperture = static_cast<float16>(0.0f);
                    cmd.fTorque = static_cast<float16>(0.0f);
                    // 게인 스케일: TRACK/HOLD/램프=1, 페이드=1→0 선형, 무토크=0.
                    float gain_scale = 1.0f;
                    if (relax_now) {
                        gain_scale = 0.0f;
                    } else if (state == State::kRelaxFade) {
                        gain_scale = clamp(
                            1.0f - static_cast<float>((now_sec() - relax_fade_t0) / RELAX_FADE_SEC), 0.0f, 1.0f);
                    }
                    cmd.fGainKp = static_cast<float16>(kp_cmd[m] * gain_scale);
                    cmd.fGainKd = static_cast<float16>(kd_cmd[m] * gain_scale);
                    cmd.fGainKi = static_cast<float16>(0.0f);
                    RobotMemGait_SetMotorCommand16(reinterpret_cast<MotorParam16_t*>(&cmd), m);
                    prev_cmd_deg[m] = cmd_deg[m];
                }
                have_prev_cmd = true;
            }
        }

        // ---- 6a) STATE 회신 (20 ms 페이싱 = policy 50 Hz) ----
        // STATE 는 정책 입력이므로 all_stt(모터 8개 유효) 게이트 유지 — 가짜 0 자세로 정책이
        // 돌면 위험. ★이 페이싱이 50 Hz 정책 클록을 소유한다(policy_runner 는 회신 즉시 다음
        // ACT 를 보내는 lockstep) — 빠르게 만들면 gait clock 이 자유질주한다.
        // TELEM 은 6b 에서 **독립 페이싱**으로 내보낸다.
        if (have_peer && (t - last_reply_time) >= kStepDtSec) {
            last_reply_time = t;
            sockaddr_in reply = peer_addr;
            if (all_stt) {
                PolicyStatePacket st;
                st.magic = POLICY_STATE_MAGIC;
                st.seq = last_act_seq;
                st.convention_version = R2S_CONVENTION_VERSION;  // 브리지가 자기 규약을 알린다
                // 관절 좌표로 보고 (gear 해제 + foot 커플링 해제). all_stt 게이트 안이라 8축 전부
                // 유효함이 보장돼 foot 이 참조하는 calf 도 항상 있다.
                {
                    float ch_deg[NUM_MOTORS];
                    float ch_dps[NUM_MOTORS];
                    for (int m = 0; m < NUM_MOTORS; m++) {
                        ch_deg[m] = static_cast<float>(motor_stt[m].fPosition);
                        ch_dps[m] = static_cast<float>(motor_stt[m].fVelocity);  // [deg/s] (Data Format 문서)
                    }
                    motor_deg_to_joint(ch_deg, st.q);
                    motor_dps_to_joint(ch_dps, st.dq);
                }
                // IMU RPY → projected gravity (ZYX, yaw 는 중력에 무영향)
                float roll = imu_buf[IDX_OF_IMU_ARPY + 0];
                float pitch = imu_buf[IDX_OF_IMU_ARPY + 1];
                if (IMU_RPY_IS_DEG) {
                    roll *= DEF_DEG2RAD;
                    pitch *= DEF_DEG2RAD;
                }
                if (imu_seen) {
                    st.gravity[0] = std::sin(pitch);
                    st.gravity[1] = -std::sin(roll) * std::cos(pitch);
                    st.gravity[2] = -std::cos(roll) * std::cos(pitch);
                } else {
                    st.gravity[0] = 0.0f;  // IMU 미수신 → 직립 가정 (경고는 주기 출력에서)
                    st.gravity[1] = 0.0f;
                    st.gravity[2] = -1.0f;
                }
                reply.sin_port = htons(static_cast<uint16_t>(REAL_STATE_PORT));
                sendto(sock, &st, sizeof(st), 0, reinterpret_cast<sockaddr*>(&reply), sizeof(reply));
            }
        }

        // ---- 6b) TELEM (관측 전용, kTelemDtSec = 200 Hz) ----
        // 이 페이싱이 **sysid 캡처의 시간 해상도**를 정한다 — gui_controller 가 수신하는 족족
        // 기록하므로(`RealMonitorThread._rec`), 여기 레이트가 그대로 npz 의 `t_real` 이 된다.
        // warmup 전에도 송신하고 무효 관절은 valid_mask 로 표시 → 워크스테이션에서
        // "링크 죽음 vs 모터데이터 없음" 구분. STATE(6a)와 포트·페이싱 모두 분리돼 있어
        // policy lockstep 과 간섭하지 않는다.
        if (have_peer && (t - last_telem_time) >= kTelemDtSec) {
            // ★위상 누적(`+= dt`)이지 리셋(`= t`)이 아니다. 1 kHz 루프에서 `>= dt` 를 검사하면
            // 매번 평균 반 틱(0.5 ms)씩 늦게 걸리는데, `= t` 로 리셋하면 그 지연이 **매 주기
            // 누적**돼 실측 레이트가 목표보다 낮아진다. 실제로 그게 구 50 Hz TELEM 이 캡처에서
            // `t_real` 48.1~48.7 Hz 로 찍힌 원인이고(README §23-j), 5 ms 목표에서는 −10 % 로
            // 더 심해진다(실측 180 Hz). 위상을 누적하면 평균 주기가 정확히 kTelemDtSec 이 된다.
            // 한 주기 넘게 밀렸으면(스톨) 따라잡기 폭주 대신 위상을 재동기한다.
            last_telem_time += kTelemDtSec;
            if (t - last_telem_time > kTelemDtSec) {
                last_telem_time = t;
            }
            sockaddr_in reply = peer_addr;
            PolicyTelemPacket tm;
            std::memset(&tm, 0, sizeof(tm));
            tm.magic = POLICY_TELEM_MAGIC;
            tm.seq = last_act_seq;
            tm.convention_version = R2S_CONVENTION_VERSION;  // 브리지가 자기 규약을 알린다
            {
                // 관절 좌표 변환은 8축 전부 채워 한 번에 한다. 무효 축의 fPosition 은 0 이지만,
                // 아래에서 valid_mask 를 내려 그 값을 쓰지 말라고 알린다.
                float ch_deg[NUM_MOTORS];
                float ch_dps[NUM_MOTORS];
                for (int m = 0; m < NUM_MOTORS; m++) {
                    ch_deg[m] = static_cast<float>(motor_stt[m].fPosition);
                    ch_dps[m] = static_cast<float>(motor_stt[m].fVelocity);
                }
                float q_joint[R2S_NUM_JOINTS];
                float dq_joint[R2S_NUM_JOINTS];
                motor_deg_to_joint(ch_deg, q_joint);
                motor_dps_to_joint(ch_dps, dq_joint);
                for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                    int m = POLICY_TO_MOTOR[p];
                    int pc = COUPLED_CALF_POLICY[p];
                    // ⚠ foot 관절각은 같은 다리 calf 없이는 **정의되지 않는다** — calf 가 무효면
                    // foot 도 무효로 내린다. raw 값을 관절각인 척 내보내지 않기 위한 것.
                    bool ok = stt_valid[m] && (pc < 0 || stt_valid[POLICY_TO_MOTOR[pc]]);
                    if (!ok) {
                        continue;  // q/dq/tau 는 memset 으로 0, valid_mask 비트도 0 유지
                    }
                    tm.valid_mask |= (1u << p);
                    tm.q[p] = q_joint[p];
                    tm.dq[p] = dq_joint[p];
                    // ⚠ tau 는 드라이버 보고값을 **그대로 통과**시킨다 — gear·커플링 둘 다 미적용.
                    // 단위는 **채널기준으로 실측 확정**됐다(2026-08-14, calib 의 GAIN_GEAR 블록):
                    // 소비자가 `× gear` 하면 관절토크다(calf 1.5 · foot 1.2). 여기서 변환하지 않는
                    // 이유는 규약 버전을 올리지 않고 기존 소비자를 깨지 않기 위해서다 — 브리지가
                    // 변환하게 되면 convention_version 을 2 로 올린다.
                    // 커플링 전치(τ_raw_calf −= coef·τ_foot, §1)는 fTorque=0 이라 불필요.
                    tm.tau[p] = MOTOR_CALIB[m].sign * static_cast<float>(motor_stt[m].fTorque);  // [N·m, 채널기준]
                }
            }
            if (imu_seen) {
                tm.valid_mask |= (1u << 8);
            }
            for (int k = 0; k < 3; k++) {
                tm.rpy[k] = imu_buf[IDX_OF_IMU_ARPY + k];  // 원값 [deg 추정] — 해석은 수신측
            }
            reply.sin_port = htons(static_cast<uint16_t>(REAL_TELEM_PORT));
            sendto(sock, &tm, sizeof(tm), 0, reinterpret_cast<sockaddr*>(&reply), sizeof(reply));
        }

        // ---- 7) 주기 로그 (probe 0.5 s / 그 외 2 s) ----
        double print_dt = (mode == Mode::kProbe) ? 0.5 : 2.0;
        if (t - last_print_time >= print_dt) {
            last_print_time = t;
            const char* sname = state == State::kWaitStatus  ? "WAIT_STATUS"
                                : state == State::kHold      ? "HOLD"
                                : state == State::kEngage    ? "ENGAGE"
                                : state == State::kRelaxRamp ? "RELAX_RAMP"
                                : state == State::kRelaxFade ? "RELAX_FADE"
                                : state == State::kRelax     ? "RELAX"
                                                             : "TRACK";
            // ★출력은 **관절각**이다(policy 순서). RELAX_REST_POSE_SIM 의 출처가 바로 이 줄이라
            // 프레임이 어긋나면 다음 재캡처 때 버그가 되살아난다 — 반드시 같은 프레임을 유지할 것.
            // 라벨도 q_sim → q_joint 로 바꿨다(로그만 떼어 봐도 프레임을 알 수 있게).
            float ch_deg_dbg[NUM_MOTORS];
            float q_joint_dbg[R2S_NUM_JOINTS];
            for (int m = 0; m < NUM_MOTORS; m++) {
                ch_deg_dbg[m] = static_cast<float>(motor_stt[m].fPosition);
            }
            motor_deg_to_joint(ch_deg_dbg, q_joint_dbg);
            std::printf("[%s] q_joint[rad]:", sname);
            for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                int m = POLICY_TO_MOTOR[p];
                int pc = COUPLED_CALF_POLICY[p];
                bool ok = stt_valid[m] && (pc < 0 || stt_valid[POLICY_TO_MOTOR[pc]]);
                std::printf(" %+.3f", ok ? q_joint_dbg[p] : 0.0f);
            }
            std::printf("  rpy:[%+.1f %+.1f %+.1f]%s  peer:%s\n", imu_buf[IDX_OF_IMU_ARPY + 0],
                        imu_buf[IDX_OF_IMU_ARPY + 1], imu_buf[IDX_OF_IMU_ARPY + 2], imu_seen ? "" : "(IMU 미수신!)",
                        have_peer ? inet_ntoa(peer_addr.sin_addr) : "-");
        }

        // ---- 8) 1 ms 절대시각 페이싱 ----
        next_tick.tv_nsec += 1000000L;
        if (next_tick.tv_nsec >= 1000000000L) {
            next_tick.tv_nsec -= 1000000000L;
            next_tick.tv_sec += 1;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_tick, nullptr);
    }

    close(sock);
    return 0;
}
