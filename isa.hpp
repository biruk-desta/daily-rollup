#pragma once

#include <string_view>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

inline bool supported_isa(std::string_view variant) {
#if defined(__x86_64__)
    unsigned a, b, c, d;
    constexpr unsigned basic = bit_SSE3 | bit_SSSE3 | bit_CMPXCHG16B | bit_SSE4_1 | bit_SSE4_2 |
        bit_POPCNT | bit_MOVBE | bit_FMA | bit_F16C | bit_OSXSAVE | bit_AVX;
    if (!__get_cpuid(1, &a, &b, &c, &d) || (c & basic) != basic) return false;
    unsigned family = (a >> 8) & 0xf;
    unsigned model = ((a >> 12) & 0xf0) | ((a >> 4) & 0xf);
    unsigned os, high;
    asm volatile("xgetbv" : "=a"(os), "=d"(high) : "c"(0));
    if ((os & 6) != 6) return false;
    if (!__get_cpuid(0x80000001, &a, &b, &c, &d) || (c & 0x21) != 0x21) return false;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    constexpr unsigned v3 = bit_AVX2 | bit_BMI | bit_BMI2;
    if ((b & v3) != v3) return false;
    if (variant == "v3") return true;

    constexpr unsigned avx512 = bit_AVX512F | bit_AVX512DQ | bit_AVX512CD | bit_AVX512BW | bit_AVX512VL;
    return variant == "cascadelake" && family == 6 && model == 0x55 &&
        (os & 0xe6) == 0xe6 && (b & avx512) == avx512 && (c & (1u << 11));
#else
    (void)variant;
    return false;
#endif
}
