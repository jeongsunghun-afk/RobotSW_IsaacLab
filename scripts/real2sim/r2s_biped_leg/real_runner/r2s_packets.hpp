/**
 * r2s_packets.hpp — r2s_udp.py 의 policy mode 패킷과 바이트 단위로 일치하는 C++ 정의.
 *
 * ⚠ 이 파일은 scripts/real2sim/r2s_biped_leg/r2s_udp.py 가 단일 진실이다.
 *   _POLICY_ACT_FMT  = "<II8f"        (40 B)  policy_runner → real (REAL_ACT 9887)
 *   _POLICY_STATE_FMT= "<II8f8f3f"    (84 B)  real → policy_runner (REAL_STATE 9888)
 *   _POLICY_TELEM_FMT= "<III8f8f8f3f" (120 B) real → gui monitor (REAL_TELEM 9889, valid_mask 포함)
 *   _POLICY_PING_FMT = "<II"          (8 B)   monitor → real (REAL_ACT 9887, peer 등록만)
 *   _POLICY_GAIN_FMT = "<II8f8f"      (72 B)  policy_runner → real (REAL_ACT 9887, kp/kd 갱신)
 * 값을 바꾸면 양쪽을 함께 바꿔야 한다. little-endian 호스트(aarch64/x86-64) 가정.
 */
#ifndef __R2S_PACKETS_HPP__
#define __R2S_PACKETS_HPP__

#include <cstdint>
#include <cstring>

constexpr int R2S_NUM_JOINTS = 8;

constexpr uint32_t POLICY_ACT_MAGIC = 0x52325041u;    // "R2PA"
constexpr uint32_t POLICY_STATE_MAGIC = 0x52325053u;  // "R2PS"
constexpr uint32_t POLICY_TELEM_MAGIC = 0x52325054u;  // "R2PT"
constexpr uint32_t POLICY_PING_MAGIC = 0x52325047u;   // "R2PG"
constexpr uint32_t POLICY_RELAX_MAGIC = 0x5232504Cu;  // "R2PL" — 무토크(limp) 요청, PING과 동일 8B
constexpr uint32_t POLICY_GAIN_MAGIC = 0x5232504Bu;   // "R2PK" — kp/kd 런타임 갱신(72B, articulation 순서)

constexpr int REAL_ACT_PORT = 9887;    // policy_runner → real (이 프로그램이 bind; PING 도 이 포트)
constexpr int REAL_STATE_PORT = 9888;  // real → policy_runner (송신자 IP 로 회신)
constexpr int REAL_TELEM_PORT = 9889;  // real → gui monitor (송신자 IP 로 회신, 관측 전용)

#pragma pack(push, 1)
struct PolicyActPacket {
    uint32_t magic;
    uint32_t seq;
    float target_q[R2S_NUM_JOINTS];  // [rad], articulation(type-major) 순서
};

struct PolicyStatePacket {
    uint32_t magic;
    uint32_t seq;
    float q[R2S_NUM_JOINTS];   // [rad], articulation 순서
    float dq[R2S_NUM_JOINTS];  // [rad/s], articulation 순서
    float gravity[3];          // projected gravity (base frame 단위벡터)
};

struct PolicyTelemPacket {
    uint32_t magic;
    uint32_t seq;
    uint32_t valid_mask;        // bit p(0~7)=관절 p 상태 유효(articulation), bit 8=IMU 수신됨
    float q[R2S_NUM_JOINTS];    // [rad], articulation 순서 (무효 관절은 0)
    float dq[R2S_NUM_JOINTS];   // [rad/s], articulation 순서
    float tau[R2S_NUM_JOINTS];  // [N·m], articulation 순서 (fTorque 에 sign 적용)
    float rpy[3];               // IMU roll/pitch/yaw 원값 [deg]
};

struct PolicyPingPacket {
    uint32_t magic;
    uint32_t seq;
};

struct PolicyGainPacket {
    uint32_t magic;
    uint32_t seq;
    float kp[R2S_NUM_JOINTS];  // MIT P 게인, articulation(type-major) 순서. 드라이버 상한 [0,500]
    float kd[R2S_NUM_JOINTS];  // MIT D 게인, articulation(type-major) 순서. 드라이버 상한 [0,5]
};
#pragma pack(pop)

static_assert(sizeof(PolicyActPacket) == 40, "r2s_udp.py POLICY_ACT_SIZE(40) mismatch");
static_assert(sizeof(PolicyStatePacket) == 84, "r2s_udp.py POLICY_STATE_SIZE(84) mismatch");
static_assert(sizeof(PolicyTelemPacket) == 120, "r2s_udp.py POLICY_TELEM_SIZE(120) mismatch");
static_assert(sizeof(PolicyPingPacket) == 8, "r2s_udp.py POLICY_PING_SIZE(8) mismatch");
static_assert(sizeof(PolicyGainPacket) == 72, "r2s_udp.py POLICY_GAIN_SIZE(72) mismatch");

inline bool unpack_policy_act(const uint8_t* data, size_t len, PolicyActPacket& out) {
    if (len != sizeof(PolicyActPacket)) {
        return false;
    }
    std::memcpy(&out, data, sizeof(out));
    return out.magic == POLICY_ACT_MAGIC;
}

inline bool unpack_policy_ping(const uint8_t* data, size_t len, PolicyPingPacket& out) {
    if (len != sizeof(PolicyPingPacket)) {
        return false;
    }
    std::memcpy(&out, data, sizeof(out));
    return out.magic == POLICY_PING_MAGIC;
}

inline bool unpack_policy_relax(const uint8_t* data, size_t len, PolicyPingPacket& out) {
    if (len != sizeof(PolicyPingPacket)) {
        return false;
    }
    std::memcpy(&out, data, sizeof(out));
    return out.magic == POLICY_RELAX_MAGIC;
}

inline bool unpack_policy_gain(const uint8_t* data, size_t len, PolicyGainPacket& out) {
    if (len != sizeof(PolicyGainPacket)) {
        return false;
    }
    std::memcpy(&out, data, sizeof(out));
    return out.magic == POLICY_GAIN_MAGIC;
}

#endif  // __R2S_PACKETS_HPP__
