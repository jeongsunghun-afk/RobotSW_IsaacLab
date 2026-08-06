/**
 * real_runner_bipedleg — r2s_biped_leg 실기(real) 엔드포인트 브리지.
 *
 * 라즈베리파이에서 RobotEmbedded(EtherCAT master)와 Shared Memory 로 모터를 교환하고,
 * 워크스테이션의 policy_runner_bipedleg.py 와 UDP seam 으로 연결한다:
 *
 *   policy_runner ──POLICY_ACT("R2PA", 9887)──▶ real_runner ──SHM──▶ RobotEmbedded ──▶ 모터
 *   policy_runner ◀──POLICY_STATE("R2PS", 9888)── real_runner ◀──SHM── 모터상태 + IMU
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
constexpr double kActTimeoutSec = 0.5;    // TRACK 중 ACT 두절 → 마지막 목표로 hold
constexpr unsigned kStatusWarmupCnt = 100;  // RobotTestGait 게이트와 동일

enum class Mode { kProbe, kHold, kBridge };
enum class State { kWaitStatus, kHold, kEngage, kTrack };

double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

// sim 좌표 [rad] → 실기 모터각 [deg]
float sim_to_motor_deg(int m, float q_rad) {
    return MOTOR_CALIB[m].sign * q_rad * DEF_RAD2DEG + MOTOR_CALIB[m].zero_deg;
}
// 실기 모터각 [deg] → sim 좌표 [rad]
float motor_deg_to_sim(int m, float pos_deg) {
    return MOTOR_CALIB[m].sign * (pos_deg - MOTOR_CALIB[m].zero_deg) * DEF_DEG2RAD;
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
    double last_act_time = -1.0;
    double last_reply_time = -1.0;
    double last_print_time = 0.0;
    uint32_t last_act_seq = 0;
    sockaddr_in peer_addr{};
    bool have_peer = false;

    timespec next_tick;
    clock_gettime(CLOCK_MONOTONIC, &next_tick);

    std::printf("[real_runner] loop start (1 kHz, reply pacing %.0f ms)\n", kStepDtSec * 1e3);

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
                if (unpack_policy_act(rx, static_cast<size_t>(n), act)) {
                    std::memcpy(target_pol, act.target_q, sizeof(target_pol));
                    last_act_seq = act.seq;
                    last_act_time = now_sec();
                    peer_addr = from;
                    have_peer = true;
                    if (state == State::kHold && mode == Mode::kBridge) {
                        state = State::kEngage;
                        engage_t0 = last_act_time;
                        std::printf("[real_runner] ACT 수신(seq=%u) → ENGAGE %.0f ms\n", act.seq, engage_sec * 1e3);
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
            if (state == State::kHold) {
                for (int m = 0; m < NUM_MOTORS; m++) {
                    cmd_deg[m] = latched_deg[m];
                }
            } else {
                // TRACK 목표: sim rad → 모터 deg (soft limit 은 sim 좌표에서 클램프)
                float track_deg[NUM_MOTORS];
                for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                    int m = POLICY_TO_MOTOR[p];
                    float q = clamp(target_pol[p], MOTOR_CALIB[m].min_rad, MOTOR_CALIB[m].max_rad);
                    track_deg[m] = sim_to_motor_deg(m, q);
                }
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

            // 선택적 slew 제한 (기본 off — 학습 plant 와 일치시키려면 끈 상태로 검증)
            if (slew_dps > 0.0f && have_prev_cmd) {
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
                    cmd.fGainKp = static_cast<float16>(kp_cmd[m]);
                    cmd.fGainKd = static_cast<float16>(kd_cmd[m]);
                    cmd.fGainKi = static_cast<float16>(0.0f);
                    RobotMemGait_SetMotorCommand16(reinterpret_cast<MotorParam16_t*>(&cmd), m);
                    prev_cmd_deg[m] = cmd_deg[m];
                }
                have_prev_cmd = true;
            }
        }

        // ---- 6) STATE 회신 (20 ms 페이싱 = policy 50 Hz) ----
        if (have_peer && all_stt && (t - last_reply_time) >= kStepDtSec) {
            PolicyStatePacket st;
            st.magic = POLICY_STATE_MAGIC;
            st.seq = last_act_seq;
            for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                int m = POLICY_TO_MOTOR[p];
                st.q[p] = motor_deg_to_sim(m, static_cast<float>(motor_stt[m].fPosition));
                float vel = static_cast<float>(motor_stt[m].fVelocity);  // [deg/s] (Data Format 문서)
                st.dq[p] = MOTOR_CALIB[m].sign * vel * DEF_DEG2RAD;
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
            sockaddr_in reply = peer_addr;
            reply.sin_port = htons(static_cast<uint16_t>(REAL_STATE_PORT));
            sendto(sock, &st, sizeof(st), 0, reinterpret_cast<sockaddr*>(&reply), sizeof(reply));
            last_reply_time = t;
        }

        // ---- 7) 주기 로그 (probe 0.5 s / 그 외 2 s) ----
        double print_dt = (mode == Mode::kProbe) ? 0.5 : 2.0;
        if (t - last_print_time >= print_dt) {
            last_print_time = t;
            const char* sname = state == State::kWaitStatus ? "WAIT_STATUS"
                                : state == State::kHold     ? "HOLD"
                                : state == State::kEngage   ? "ENGAGE"
                                                            : "TRACK";
            std::printf("[%s] q_sim[rad]:", sname);
            for (int p = 0; p < R2S_NUM_JOINTS; p++) {
                int m = POLICY_TO_MOTOR[p];
                std::printf(" %+.3f", stt_valid[m] ? motor_deg_to_sim(m, (float)motor_stt[m].fPosition) : 0.0f);
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
