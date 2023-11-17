#include <array>
#include <cstddef>
#include <cstdint>
//#include <stdfloat> // std::float32_t, std::float64_t

#include <gtest/gtest.h>
#include "simd.hpp"

#ifdef _WIN16
#define cpuid(info, x)    __cpuidex(info, x, 0)
#else
#include <cpuid.h>
void cpuid(int info[4], int InfoType){
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif

// Currently only checks for CPU support, not OS - therefore, this is not 100% accurate
void print_supported_intructions()
{
    //  Misc.
    bool HW_MMX = false;
    bool HW_x64 = false;
    bool HW_ABM = false;      // Advanced Bit Manipulation
    bool HW_RDRAND = false;
    bool HW_BMI1 = false;
    bool HW_BMI2 = false;
    bool HW_ADX = false;
    bool HW_PREFETCHWT1 = false;

    //  SIMD: 128-bit
    bool HW_SSE = false;
    bool HW_SSE2 = false;
    bool HW_SSE3 = false;
    bool HW_SSSE3 = false;
    bool HW_SSE4_1 = false;
    bool HW_SSE4_2 = false;
    bool HW_SSE4A = false;
    bool HW_AES = false;
    bool HW_SHA = false;

    //  SIMD: 256-bit
    bool HW_AVX = false;
    bool HW_XOP = false;
    bool HW_FMA3 = false;
    bool HW_FMA4 = false;
    bool HW_AVX2 = false;

    //  SIMD: 512-bit
    bool HW_AVX512F = false;    //  AVX512 Foundation
    bool HW_AVX512CD = false;   //  AVX512 Conflict Detection
    bool HW_AVX512PF = false;   //  AVX512 Prefetch
    bool HW_AVX512ER = false;   //  AVX512 Exponential + Reciprocal
    bool HW_AVX512VL = false;   //  AVX512 Vector Length Extensions
    bool HW_AVX512BW = false;   //  AVX512 Byte + Word
    bool HW_AVX512DQ = false;   //  AVX512 Doubleword + Quadword
    bool HW_AVX512IFMA = false; //  AVX512 Integer 52-bit Fused Multiply-Add
    bool HW_AVX512VBMI = false; //  AVX512 Vector Byte Manipulation Instructions

    int info[4];
    cpuid(info, 0);
    int nIds = info[0];

    cpuid(info, 0x80000000);
    unsigned nExIds = info[0];

    std::cout << "Supported intructions by CPU:\n";

    //  Detect Features
    if (nIds >= 0x00000001){
        cpuid(info,0x00000001);
        HW_MMX    = (info[3] & ((int)1 << 23)) != 0;
        HW_SSE    = (info[3] & ((int)1 << 25)) != 0;
        HW_SSE2   = (info[3] & ((int)1 << 26)) != 0;
        HW_SSE3   = (info[2] & ((int)1 <<  0)) != 0;

        HW_SSSE3  = (info[2] & ((int)1 <<  9)) != 0;
        HW_SSE4_1 = (info[2] & ((int)1 << 19)) != 0;
        HW_SSE4_2 = (info[2] & ((int)1 << 20)) != 0;
        HW_AES    = (info[2] & ((int)1 << 25)) != 0;

        HW_AVX    = (info[2] & ((int)1 << 28)) != 0;
        HW_FMA3   = (info[2] & ((int)1 << 12)) != 0;

        HW_RDRAND = (info[2] & ((int)1 << 30)) != 0;

        std::cout << (HW_MMX ? "MMX " : "") << (HW_SSE ? "SSE " : "") << (HW_SSE2 ? "SSE2 " : "") << (HW_SSE3 ? "SSE3\n" : "\n");
        std::cout << (HW_SSSE3 ? "SSSE3 " : "") << (HW_SSE4_1 ? "SSE4.1 " : "") << (HW_SSE4_2 ? "SSE4.2 " : "") << (HW_AES ? "AES\n" : "\n");
        std::cout << (HW_AVX ? "AVX " : "") << (HW_FMA3 ? "FMA3\n" : "\n");
        std::cout << (HW_RDRAND ? "RDRAND\n" : "\n");
    }
    if (nIds >= 0x00000007){
        cpuid(info,0x00000007);
        HW_AVX2   = (info[1] & ((int)1 <<  5)) != 0;

        HW_BMI1        = (info[1] & ((int)1 <<  3)) != 0;
        HW_BMI2        = (info[1] & ((int)1 <<  8)) != 0;
        HW_ADX         = (info[1] & ((int)1 << 19)) != 0;
        HW_SHA         = (info[1] & ((int)1 << 29)) != 0;
        HW_PREFETCHWT1 = (info[2] & ((int)1 <<  0)) != 0;

        HW_AVX512F     = (info[1] & ((int)1 << 16)) != 0;
        HW_AVX512CD    = (info[1] & ((int)1 << 28)) != 0;
        HW_AVX512PF    = (info[1] & ((int)1 << 26)) != 0;
        HW_AVX512ER    = (info[1] & ((int)1 << 27)) != 0;
        HW_AVX512VL    = (info[1] & ((int)1 << 31)) != 0;
        HW_AVX512BW    = (info[1] & ((int)1 << 30)) != 0;
        HW_AVX512DQ    = (info[1] & ((int)1 << 17)) != 0;
        HW_AVX512IFMA  = (info[1] & ((int)1 << 21)) != 0;
        HW_AVX512VBMI  = (info[2] & ((int)1 <<  1)) != 0;

        std::cout << (HW_AVX2 ? "AVX2\n" : "\n");
        std::cout << (HW_BMI1 ? "BMI1 " : "") << (HW_BMI2 ? "BMI2 " : "") << (HW_ADX ? "ADX " : "") << (HW_SHA ? "SHA " : "") << (HW_PREFETCHWT1 ? "PREFETCHWT1\n" : "\n");
        std::cout << (HW_AVX512F ? "AVX512F " : "") << (HW_AVX512CD ? "AVX512CD " : "") << (HW_AVX512PF ? "AVX512PF " : "") << (HW_AVX512ER ? "AVX512ER " : "") << (HW_AVX512VL ? "AVX512VL" : "");
        std::cout << (HW_AVX512BW ? "AVX512BW " : "") << (HW_AVX512DQ ? "AVX512DQ " : "") << (HW_AVX512IFMA ? "AVX512IFMA " : "") << (HW_AVX512VBMI ? "AVX512VBMI\n" : "\n");
    }
    if (nExIds >= 0x80000001){
        cpuid(info,0x80000001);
        HW_x64   = (info[3] & ((int)1 << 29)) != 0;
        HW_ABM   = (info[2] & ((int)1 <<  5)) != 0;
        HW_SSE4A = (info[2] & ((int)1 <<  6)) != 0;
        HW_FMA4  = (info[2] & ((int)1 << 16)) != 0;
        HW_XOP   = (info[2] & ((int)1 << 11)) != 0;

        std::cout << (HW_x64 ? "x86_64 " : "") << (HW_ABM ? "ABM " : "") << (HW_SSE4A ? "SSE4A " : "") << (HW_FMA4 ? "FMA4 " : "") << (HW_XOP ? "XOP\n" : "\n");
    }

    std::cout << "--- NOTE: This check is for CPU support, not OS support ---\n";
    std::cout << std::endl;
}

//-----------------------------------------------------------------------------
//  float32_t
//-----------------------------------------------------------------------------

TEST(float32_t, add)
{
    const std::size_t N = 20;

    simd::vector<float, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    float expected;

    simd::vector<float, N> result1 = simd_int_array + simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<float, N> result2 = simd_int_array + 10.0f;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + 10.0f;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<float, N> result3 = 20.0f + simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0f + simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(float32_t, sub)
{
    const std::size_t N = 20;

    simd::vector<float, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    float expected;

    simd::vector<float, N> result1 = simd_int_array - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<float, N> result2 = simd_int_array - 10.0f;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - 10.0f;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<float, N> result3 = 20.0f - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0f - simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(float32_t, mul)
{
    const std::size_t N = 20;

    simd::vector<float, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    float expected;

    simd::vector<float, N> result1 = simd_int_array * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector expected1 and result1 differ at index " << i;
    }

    simd::vector<float, N> result2 = simd_int_array * 10.0f;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * 10.0f;
        EXPECT_EQ(expected, result2[i]) << "Vector expected2 and result2 differ at index " << i;
    }

    simd::vector<float, N> result3 = 20.0f * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0f * simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector expected3 and result3 differ at index " << i;
    }
}

TEST(float32_t, div)
{
    const std::size_t N = 20;

    simd::vector<float, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    float expected;

    simd::vector<float, N> result1 = simd_int_array / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<float, N> result2 = simd_int_array / 10.0f;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / 10.0f;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<float, N> result3 = 20.0f / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0f / simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector result3 differ at index " << i;
    }
}

//-----------------------------------------------------------------------------
//  float64_t
//-----------------------------------------------------------------------------

TEST(float64_t, add)
{
    const std::size_t N = 20;

    simd::vector<double, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    double expected;

    simd::vector<double, N> result1 = simd_int_array + simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<double, N> result2 = simd_int_array + 10.0;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + 10.0;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<double, N> result3 = 20.0 + simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0 + simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(float64_t, sub)
{
    const std::size_t N = 20;

    simd::vector<double, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    double expected;

    simd::vector<double, N> result1 = simd_int_array - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<double, N> result2 = simd_int_array - 10.0;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - 10.0;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<double, N> result3 = 20.0 - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0 - simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(float64_t, mul)
{
    const std::size_t N = 20;

    simd::vector<double, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    double expected;

    simd::vector<double, N> result1 = simd_int_array * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector expected1 and result1 differ at index " << i;
    }

    simd::vector<double, N> result2 = simd_int_array * 10.0;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * 10.0;
        EXPECT_EQ(expected, result2[i]) << "Vector expected2 and result2 differ at index " << i;
    }

    simd::vector<double, N> result3 = 20.0 * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0 * simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector expected3 and result3 differ at index " << i;
    }
}

TEST(float64_t, div)
{
    const std::size_t N = 20;

    simd::vector<double, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    double expected;

    simd::vector<double, N> result1 = simd_int_array / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<double, N> result2 = simd_int_array / 10.0;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / 10.0;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<double, N> result3 = 20.0 / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20.0 / simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector result3 differ at index " << i;
    }
}

//-----------------------------------------------------------------------------
//  int8_t
//-----------------------------------------------------------------------------

TEST(int8_t, add)
{
    const std::size_t N = 20;

    simd::vector<std::int8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::int8_t expected;

    simd::vector<std::int8_t, N> result1 = simd_int_array + simd_int_array;

    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result2 = simd_int_array + 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + 10;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result3 = 20 + simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 + simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(int8_t, sub)
{
    const std::size_t N = 20;

    simd::vector<std::int8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::int8_t expected;

    simd::vector<std::int8_t, N> result1 = simd_int_array - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result2 = simd_int_array - 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - 10;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result3 = 20 - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 - simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(int8_t, mul)
{
    const std::size_t N = 20;

    simd::vector<std::int8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::int8_t expected;

    simd::vector<std::int8_t, N> result1 = simd_int_array * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector expected1 and result1 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result2 = simd_int_array * 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * 10;
        EXPECT_EQ(expected, result2[i]) << "Vector expected2 and result2 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result3 = 20 * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 * simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector expected3 and result3 differ at index " << i;
    }
}

TEST(int8_t, div)
{
    const std::size_t N = 20;

    simd::vector<std::int8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::int8_t expected;

    simd::vector<std::int8_t, N> result1 = simd_int_array / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result2 = simd_int_array / 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / 10;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<std::int8_t, N> result3 = 20 / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 / simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector result3 differ at index " << i;
    }
}

//-----------------------------------------------------------------------------
//  uint8_t
//-----------------------------------------------------------------------------

TEST(uint8_t, add)
{
    const std::size_t N = 20;

    simd::vector<std::uint8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::uint8_t expected;

    simd::vector<std::uint8_t, N> result1 = simd_int_array + simd_int_array;

    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result2 = simd_int_array + 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] + 10;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result3 = 20 + simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 + simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(uint8_t, sub)
{
    const std::size_t N = 20;

    simd::vector<std::uint8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::uint8_t expected;

    simd::vector<std::uint8_t, N> result1 = simd_int_array - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result2 = simd_int_array - 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] - 10;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result3 = 20 - simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 - simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vectors result3 differ at index " << i;
    }
}

TEST(uint8_t, mul)
{
    const std::size_t N = 20;

    simd::vector<std::uint8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::uint8_t expected;

    simd::vector<std::uint8_t, N> result1 = simd_int_array * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector expected1 and result1 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result2 = simd_int_array * 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] * 10;
        EXPECT_EQ(expected, result2[i]) << "Vector expected2 and result2 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result3 = 20 * simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 * simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector expected3 and result3 differ at index " << i;
    }
}

TEST(uint8_t, div)
{
    const std::size_t N = 20;

    simd::vector<std::uint8_t, N> simd_int_array = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    std::uint8_t expected;

    simd::vector<std::uint8_t, N> result1 = simd_int_array / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / simd_int_array[i];
        EXPECT_EQ(expected, result1[i]) << "Vector result1 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result2 = simd_int_array / 10;
    for (int i = 0; i < N; ++i) {
        expected = simd_int_array[i] / 10;
        EXPECT_EQ(expected, result2[i]) << "Vector result2 differ at index " << i;
    }

    simd::vector<std::uint8_t, N> result3 = 20 / simd_int_array;
    for (int i = 0; i < N; ++i) {
        expected = 20 / simd_int_array[i];
        EXPECT_EQ(expected, result3[i]) << "Vector result3 differ at index " << i;
    }
}

int main(int argc, char* argv[])
{
    print_supported_intructions();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
