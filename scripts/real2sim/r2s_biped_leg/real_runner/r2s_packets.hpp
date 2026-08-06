/**
 * r2s_packets.hpp — r2s_udp.py 의 policy mode 패킷과 바이트 단위로 일치하는 C++ 정의.
 *
 * ⚠ 이 파일은 scripts/real2sim/r2s_biped_leg/r2s_udp.py 가 단일 진실이다.
 *   _POLICY_ACT_FMT  = "<II8f"      (40 B)  policy_runner → real (REAL_ACT 9887)
 *   _POLICY_STATE_FMT= "<II8f8f3f"  (84 B)  real → policy_runner (REAL_STATE 9888)
 * 값을 바꾸면 양쪽을 함께 바꿔야 한다. little-endian 호스트(aarch64/x86-64) 가정.
 */
#ifndef __R2S_PACKETS_HPP__
#define __R2S_PACKETS_HPP__

#include <cstdint>
#include <cstring>

constexpr int R2S_NUM_JOINTS = 8;

constexpr uint32_t POLICY_ACT_MAGIC = 0x52325041u;    // "R2PA"
constexpr uint32_t POLICY_STATE_MAGIC = 0x52325053u;  // "R2PS"

constexpr int REAL_ACT_PORT = 9887;    // policy_runner → real (이 프로그램이 bind)
constexpr int REAL_STATE_PORT = 9888;  // real → policy_runner (송신자 IP 로 회신)

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
#pragma pack(pop)

static_assert(sizeof(PolicyActPacket) == 40, "r2s_udp.py POLICY_ACT_SIZE(40) mismatch");
static_assert(sizeof(PolicyStatePacket) == 84, "r2s_udp.py POLICY_STATE_SIZE(84) mismatch");

inline bool unpack_policy_act(const uint8_t* data, size_t len, PolicyActPacket& out) {
    if (len != sizeof(PolicyActPacket)) {
        return false;
    }
    std::memcpy(&out, data, sizeof(out));
    return out.magic == POLICY_ACT_MAGIC;
}

#endif  // __R2S_PACKETS_HPP__
