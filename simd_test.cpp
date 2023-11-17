#include <array>
#include <cstdint>
//#include <stdfloat>

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

TEST(float32_t, add)
{
    //GTEST_ASSERT_EQ(add(10, 22), 32);

    const std::size_t N = 20;

    const std::array<float, N> expected1 = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40};
    const std::array<float, N> expected2 = {11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
    const std::array<float, N> expected3 = {21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};

    simd::vector<float, N> simd_int_array1 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    simd::vector<float, N> simd_int_array2 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};

    simd::vector<float, N> result1 = simd_int_array1 + simd_int_array2;

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(expected1[i], result1[i]) << "Vectors expected1 and result1 differ at index " << i;
    }

    //simd::vector<float, 20> result2 = simd_int_array1 - simd_int_array2;
    //simd::vector<float, 20> result3 = simd_int_array1 * simd_int_array2;
    //simd::vector<float, 20> result4 = simd_int_array1 / simd_int_array2;
    simd::vector<float, N> result2 = simd_int_array2 + 10;

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(expected2[i], result2[i]) << "Vectors expected2 and result2 differ at index " << i;
    }

    simd::vector<float, N> result3 = 20.0f + simd_int_array2;

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(expected3[i], result3[i]) << "Vectors expected3 and result3 differ at index " << i;
    }
    //simd::vector<float, 20> result7 = simd_int_array2 - 10;
    //simd::vector<float, 20> result8 = 20.0f - simd_int_array2;
    //simd::vector<float, 20> result9 = simd_int_array2 * 10;
    //simd::vector<float, 20> result10 = 20.0f * simd_int_array2;
    //simd::vector<float, 20> result11 = simd_int_array2 / 10;
    //simd::vector<float, 20> result12 = 20.0f / simd_int_array2;
}

int main(int argc, char* argv[])
{
    print_supported_intructions();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
