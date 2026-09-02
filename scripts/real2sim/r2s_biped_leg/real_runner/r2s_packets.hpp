/**
 * r2s_packets.hpp — r2s_udp.py 의 policy mode 패킷과 바이트 단위로 일치하는 C++ 정의.
 *
 * ⚠ 이 파일은 scripts/real2sim/r2s_biped_leg/r2s_udp.py 가 단일 진실이다.
 *   _POLICY_ACT_FMT  = "<II8f"        (40 B)  policy_runner → real (REAL_ACT 9887)
 *   _POLICY_STATE_FMT= "<II8f8f3fB"   (85 B)  real → policy_runner (REAL_STATE 9888, 규약버전)
 *   _POLICY_TELEM_FMT= "<III8f8f8f3fBI8f8f" (189 B) real → gui monitor (REAL_TELEM 9889,
 *                       valid_mask + 규약버전 + **telem_tick + cmd_q**, 2026-08-19)
 *   _POLICY_PING_FMT = "<II"          (8 B)   monitor → real (REAL_ACT 9887, peer 등록만)
 *   _POLICY_GAIN_FMT = "<II8f8f"      (72 B)  policy_runner → real (REAL_ACT 9887, kp/kd 갱신)
 * 값을 바꾸면 양쪽을 함께 바꿔야 한다. little-endian 호스트(aarch64/x86-64) 가정.
 *
 * ⚠ 2026-08-14: STATE 84→85 B · TELEM 120→121 B 로 늘리며 **이 미러를 먼저 고쳤다**(위 "단일 진실"
 *   규약과 반대 방향). r2s_udp.py 반영 전까지 잠정 상태이고, 그 전에는 policy_runner 와 GUI 가
 *   크기 불일치로 패킷을 **거부한다** — 조용히 오독하는 것보다 낫다고 판단한 의도적 선택이다.
 */
#ifndef __R2S_PACKETS_HPP__
#define __R2S_PACKETS_HPP__

#include <cstdint>
#include <cstring>

constexpr int R2S_NUM_JOINTS = 8;

/**
 * ★규약 버전 — 브리지가 **자기 좌표·단위 규약을 스스로 알린다**.
 * STATE(offset 84) 와 TELEM(offset 120) 둘 다 `convention_version` 으로 실어 보낸다.
 *
 * 이전에는 GUI 가 상수로 `gear_applied=False` 를 npz 에 찍었는데, 그건 "브리지 소스가 고쳐졌나"가
 * 아니라 "**캡처 당시 파이에서 돌던 바이너리**가 뭘 했나"를 뜻해야 하는 값이라 사람이 두 곳을
 * 동기화해야 했다. 브리지가 실어 보내면 도장이 자기유지된다 — GUI 는 **받은 값을 그대로 기록**할 것.
 *
 *   0 = (구) gear 미적용 + foot **raw**각(q_foot+q_calf) 보고/수신.
 *       이 값을 명시적으로 보내는 브리지는 없다 — 필드가 없는 84 B STATE / 120 B TELEM 이 곧 버전 0.
 *   1 = gear 적용(위치·속도) + foot **관절**각 보고/수신. (2026-08-14)
 *       ⚠ `tau` 만은 브리지가 **변환하지 않고 그대로 통과**시킨다. 단위는 미판정이 아니라
 *       **채널기준으로 실측 확정**됐다(2026-08-14, calib 의 GAIN_GEAR 블록 참조) — 소비자가
 *       `× gear` 하면 관절토크다(calf 1.5 · foot 1.2). 브리지가 그 변환까지 하게 되면 버전 2.
 *       ⇒ 버전은 "무엇이 판정됐나"가 아니라 **"브리지가 무엇을 하는가"** 를 가리킨다.
 */
constexpr uint8_t R2S_CONVENTION_VERSION = 1;

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
    float q[R2S_NUM_JOINTS];   // [rad], articulation 순서. 버전 1: **관절각**
    float dq[R2S_NUM_JOINTS];  // [rad/s], articulation 순서. 버전 1: **관절 각속도**
    float gravity[3];          // projected gravity (base frame 단위벡터)
    // ★정책 **입력** 경로라 프레임이 조용히 어긋나면 가장 위험하다 — TELEM 과 같은 이유로 버전을
    //   싣는다. 버전 정의는 R2S_CONVENTION_VERSION 참조. offset 84, 패킷 85 B.
    uint8_t convention_version;
};

struct PolicyTelemPacket {
    uint32_t magic;
    uint32_t seq;
    // bit p(0~7)=관절 p 상태 유효(articulation), bit 8=IMU 수신됨.
    // bit 16+p(16~23)=관절 p 의 목표가 이번 틱에 **soft limit 으로 클램프됐다**(2026-08-19).
    //   → §18 류의 "명령이 조용히 잘려 기록엔 안 남는" 사고를 캡처가 스스로 신고하게 만든다.
    // bit 24 = `cmd_q` 가 유효하다(= 이번 틱에 실제로 명령을 전송했다). 0 이면 cmd_q 는 0 이다.
    // ⚠ 버전 1 부터 **foot(p=6,7) 비트의 의미가 달라졌다**: foot 을 관절각으로 내보내려면 같은 다리
    //   calf(p-2) 값이 필요하므로, calf 가 무효면 foot 관절각은 **정의되지 않는다**. 그래서
    //   foot 비트는 "foot **과 그 calf 가 둘 다** 유효"를 뜻한다. raw 값을 관절각인 척 내보내지
    //   않기 위한 것 — 이 프로젝트에서 반복된 조용한 프레임 불일치 사고를 막는다.
    uint32_t valid_mask;
    float q[R2S_NUM_JOINTS];    // [rad], articulation 순서 (무효 관절은 0). 버전 1: **관절각**
    float dq[R2S_NUM_JOINTS];   // [rad/s], articulation 순서. 버전 1: **관절 각속도**
    float tau[R2S_NUM_JOINTS];  // [N·m], articulation 순서 (fTorque 에 sign 적용). **채널기준** —
                                // 소비자가 × gear 하면 관절토크 (calf 1.5 · foot 1.2)
    float rpy[3];               // IMU roll/pitch/yaw 원값 [deg]
    uint8_t convention_version;  // = R2S_CONVENTION_VERSION. 위 정의 참조. offset 120

    // ── 2026-08-19 확장 (121 → 157 B). 규약 버전은 **1 그대로**다 — 기존 필드의 의미가 하나도
    //    안 바뀌었고, 새 필드는 **패킷 크기**로 구분한다(legacy 116/120/121 B 와 같은 방식).
    //    버전 2 는 여전히 "브리지가 tau 까지 변환한다"에 예약돼 있다.

    /** TELEM 송신 틱 카운터. 매 송신마다 +1, `kTelemDtSec`(5 ms) 간격.
     *
     * ★있는 이유: 소비자(gui)가 찍는 도착 시각은 "파이썬이 알아챈 시각"이라 GIL·Qt·네트워크
     *   지터가 그대로 섞인다(실측 std 0.50 ms — 5 ms 간격의 10 %). 틱을 실어 보내면
     *   `t = t0 + tick × 5 ms` 로 **간격이 구조적으로 정확**해지고, 미지수는 t0 하나만 남는다.
     *   그 하나는 PACE 의 delay 파라미터와 재생 shift 가 이미 흡수하는 값이다.
     *   유실도 tick 이 건너뛰는 것으로 드러나 "빠진 걸 모른 채 보간"하는 사고를 막는다.
     */
    uint32_t telem_tick;

    /** 이 틱에 **실제로 드라이버에 실린** 목표각 [rad], articulation 순서, 관절 좌표.
     *
     * ★있는 이유: 지금까지 캡처는 GUI 가 **발행한** 값만 담아서, 드라이버가 실제로 무엇을
     *   받았는지 알 수 없었다. 그 사이에 soft-limit 클램프 · ENGAGE 램프 · slew 제한 ·
     *   float16 양자화가 끼어든다. 그래서 "sim 이 느린 것"과 "명령 기록이 어긋난 것"을
     *   가를 수 없었다(§18 오염 사고의 근본 원인이기도 하다).
     *
     * 값은 실제 전송값 `float16(cmd_deg)` 를 **되돌려** `motor_deg_to_joint` 로 환산한 것이라
     * float16 양자화까지 포함한다. `q` 와 **같은 프레임·같은 변환**이므로 소비자는 아무 변환
     * 없이 `cmd_q − q` 를 추종오차로 바로 쓸 수 있다.
     *
     * 전송 중이 아닐 때(warmup·HOLD 전·relax 무토크)는 0 이고, `valid_mask` bit 24 로 알린다.
     */
    float cmd_q[R2S_NUM_JOINTS];
    /** 드라이버가 보고한 **채널각 원값** [deg], **모터(leg-major) 순서**. valid_mask bit 25.
     *
     * ⚠ 이 패킷에서 **유일하게 모터 순서**인 필드다 (q/dq/tau/cmd_q 는 articulation).
     *   `zero_deg` 가 MOTOR_CALIB 의 모터 순서로 정의돼 있어 그 순서를 그대로 실어야
     *   워크스테이션이 재배열 없이 복사할 수 있다 — 재배열 한 번이 이 프로젝트에서 반복해
     *   사고를 낸 지점이다.
     *
     * 왜 필요한가 — `zero_deg` 를 실측하려면 **변환 이전의 채널각**이 있어야 한다. q 는 이미
     * `(q_ch − zero_deg)/gear` 를 거친 값이라 zero_deg 가 틀린 상태에서는 역산해도 그 오차가
     * 그대로 남는다. 영점 캘리브레이션은 원값을 봐야 성립한다.
     */
    float ch_deg[R2S_NUM_JOINTS];
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
// ⚠ 84 → 85 B (convention_version 추가, offset 0~83 은 불변). r2s_udp.py 를 같이 고쳐야 하며,
//   그 전까지 policy_runner 는 크기 불일치로 STATE 를 **거부**한다(조용한 오독보다 낫다 — 의도).
static_assert(sizeof(PolicyStatePacket) == 85, "r2s_udp.py POLICY_STATE_SIZE(85) mismatch");
// ⚠ 157 → 189 B (ch_deg 추가, offset 0~156 은 불변). r2s_udp.py 를 같이 고쳐야 하며,
//   그 전까지 GUI/comm_check 는 크기 불일치로 TELEM 을 **거부**한다(조용한 오독보다 낫다 — 의도).
//   r2s_udp.py 는 크기로 버전을 가르므로(v4 157 · v3 121 · v2 120 · v1 116) **끝에만 붙일 것** —
//   중간에 넣으면 구 캡처가 조용히 오독된다.
//   (그 이전: 121 → 157 B telem_tick + cmd_q · 120 → 121 B convention_version.)
static_assert(sizeof(PolicyTelemPacket) == 189, "r2s_udp.py POLICY_TELEM_SIZE(189) mismatch");
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
