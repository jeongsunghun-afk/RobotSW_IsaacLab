/**
 * float16_compat.hpp — _Float16 미지원 컴파일러(x86-64 gcc<12 등)용 IEEE half shim.
 *
 * STUB 빌드 전용. 파이(aarch64 gcc)와 gcc>=12 는 네이티브 _Float16 이 있으므로
 * __FLT16_MANT_DIG__ 로 감지해 아무것도 하지 않는다. defineConfigMotor.h 의
 * `using float16 = _Float16;` 보다 먼저 include 되어야 한다.
 * 크기/정렬(2바이트)이 실기 _Float16 과 동일해 MotGeneral_t 레이아웃이 보존된다.
 */
#ifndef __FLOAT16_COMPAT_HPP__
#define __FLOAT16_COMPAT_HPP__

#if !defined(__FLT16_MANT_DIG__)

#include <cstdint>
#include <cstring>

struct RgaHalf {
    std::uint16_t bits = 0;

    RgaHalf() = default;
    RgaHalf(float f) { bits = from_float(f); }  // NOLINT(implicit) — _Float16 대체이므로 암시 변환 필요
    operator float() const { return to_float(bits); }

    static std::uint16_t from_float(float f) {
        std::uint32_t x;
        std::memcpy(&x, &f, 4);
        std::uint32_t sign = (x >> 16) & 0x8000u;
        std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        std::uint32_t man = x & 0x7FFFFFu;
        if (exp <= 0) {
            return static_cast<std::uint16_t>(sign);  // 언더플로 → ±0 (denormal 생략, 테스트 용도로 충분)
        }
        if (exp >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7C00u);  // 오버플로 → ±inf
        }
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (man >> 13));
    }

    static float to_float(std::uint16_t h) {
        std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
        std::uint32_t exp = (h >> 10) & 0x1Fu;
        std::uint32_t man = h & 0x3FFu;
        std::uint32_t x;
        if (exp == 0) {
            x = sign;  // ±0/denormal → ±0
        } else if (exp == 31) {
            x = sign | 0x7F800000u | (man << 13);  // inf/NaN
        } else {
            x = sign | ((exp - 15 + 127) << 23) | (man << 13);
        }
        float f;
        std::memcpy(&f, &x, 4);
        return f;
    }
};

static_assert(sizeof(RgaHalf) == 2, "half shim must be 2 bytes");

#define _Float16 RgaHalf

#endif  // !__FLT16_MANT_DIG__

#endif  // __FLOAT16_COMPAT_HPP__
