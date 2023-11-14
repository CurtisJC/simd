/*
 *  SIMD (single instruction multiple data) header library
 *
 *  Wrapper for intrinsic functions using templates where possible.
 *
 *  Alternatives have been implemented where intrinsic functions do not exist to provide a level of abstraction to the
 *  underlying operations
 */
#pragma once

#include <type_traits>
#include <immintrin.h>

namespace simd {

    //__m128i _mm_mul_epi8(__m128i a, __m128i b)
    //{
    //    __m128i mask = _mm_set1_epi16(0xff00);
    //    // mask higher bytes:
    //    __m128i a_hi = _mm_and_si128(a, mask);
    //    __m128i b_hi = _mm_and_si128(b, mask);
//
    //    __m128i r_hi = _mm_mulhi_epi16(a_hi, b_hi);
    //    // mask out garbage in lower half:
    //    r_hi = _mm_and_si128(r_hi, mask);
//
    //    // shift lower bytes to upper half
    //    __m128i a_lo = _mm_slli_epi16(a,8);
    //    __m128i b_lo = _mm_slli_epi16(b,8);
    //    __m128i r_lo = _mm_mulhi_epi16(a_lo, b_lo);
    //    // shift result to the lower half:
    //    r_lo = _mm_srli_epi16(r_lo,8);
//
    //    // join result and return:
    //    return _mm_or_si128(r_hi, r_lo);
    //}
//
    //__m128i _mm_mul_epu16(__m128i a, __m128i b)
    //{
    //    // Convert the input integers to unsigned 32-bit integers.
    //    __m128i a_u32 = _mm_cvtpu16_epi32(a);
    //    __m128i b_u32 = _mm_cvtpu16_epi32(b);
//
    //    // Multiply the two integers using the _mm_mul_epi32 intrinsic.
    //    __m128i product = _mm_mul_epi32(a_u32, b_u32);
//
    //    // Convert the product back to unsigned 16-bit integers.
    //    return _mm_cvtepi32_epi16(product);
    //}
//
    //__m128i _mm_mul_epi16(__m128i a, __m128i b)
    //{
    //    // Perform high-order and low-order multiplications.
    //    __m128i high = _mm_mulhi_epi16(a, b);
    //    __m128i low = _mm_mullo_epi16(a, b);
//
    //    // Combine the high-order and low-order products.
    //    return _mm_add_epi16(high, _mm_slli_epi16(low, 16));
    //}

    template<typename V>
    V load(void const* mem_addr)
    {
        if constexpr (std::is_same_v<V, __m128i>)      { return _mm_loadu_si128((__m128i_u*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_loadu_ps((float*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_loadu_pd((double*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m256i>) { return _mm256_loadu_si256((__m256i_u*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_loadu_ps((float*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_loadu_pd((double*)mem_addr); }
    }

    template<typename V>
    void store(void const* mem_addr, V a)
    {
        if constexpr (std::is_same_v<V, __m128i>)      { _mm_storeu_si128((__m128i_u*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m128>)  { _mm_storeu_ps((float*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m128d>) { _mm_storeu_pd((double*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m256i>) { _mm256_storeu_si256((__m256i_u*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m256>)  { _mm256_storeu_ps((float*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m256d>) { _mm256_storeu_pd((double*)mem_addr, a); }
    }

    template<typename T, typename V>
    V set(T const s)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_set1_epi8(s); }  // SSE2
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_set1_epi16(s); } // SSE2
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_set1_epi32(s); } // SSE2
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_set1_epi64(s); } // SSE2
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_set1_ps(s); } // SSE
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_set1_pd(s); } // SSE2
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_set1_epi8(s); }  // AVX2
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_set1_epi16(s); } // AVX2
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_set1_epi32(s); } // AVX2
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_set1_epi64(s); } // AVX2
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_set1_ps(s); } // AVX
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_set1_pd(s); } // AVX
    }

    template<typename T, typename V>
    V add(V const a, V const b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_add_epi8(a, b); }  // SSE2
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_add_epi16(a, b); } // SSE2
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_add_epi32(a, b); } // SSE2
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_add_epi64(a, b); } // SSE2
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_add_ps(a, b); } // SSE
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_add_pd(a, b); } // SSE2
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_add_epi8(a, b); }  // AVX2
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_add_epi16(a, b); } // AVX2
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_add_epi32(a, b); } // AVX2
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_add_epi64(a, b); } // AVX2
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_add_ps(a, b); } // AVX
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_add_pd(a, b); } // AVX
    }

    template<typename T, typename V>
    V sub(V const a, V const b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_sub_epi8(a, b); }  // SSE2
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_sub_epi16(a, b); } // SSE2
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_sub_epi32(a, b); } // SSE2
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_sub_epi64(a, b); } // SSE2
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_sub_ps(a, b); } // SSE
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_sub_pd(a, b); } // SSE2
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_sub_epi8(a, b); }  // AVX2
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_sub_epi16(a, b); } // AVX2
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_sub_epi32(a, b); } // AVX2
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_sub_epi64(a, b); } // AVX2
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_sub_ps(a, b); } // AVX
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_sub_pd(a, b); } // AVX
    }

    template<typename T, typename V>
    V mul(V const a, V const b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)       { return _mm_mul_epu8(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int8_t>)   { return _mm_mul_epi8(a, b); } // TODO
            else if constexpr (std::is_same_v<T, uint16_t>) { return _mm_mul_epu16(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int16_t>)  { return _mm_mullo_epi16(a, b); } // SSE2
            else if constexpr (std::is_same_v<T, uint32_t>) { return _mm_mul_epu32(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int32_t>)  { return _mm_mullo_epi32(a, b); } // SSE4.1
            else if constexpr (std::is_same_v<T, uint64_t>) { return _mm_mul_epu64(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int64_t>)  { return _mm_mullo_epi64(a, b); } // AVX512DQ + AVX512VL
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_mul_ps(a, b); } // SSE
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_mul_pd(a, b); } // SSE2
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)       { return _mm256_mul_epu8(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int8_t>)   { return _mm256_mul_epi8(a, b); } // TODO
            else if constexpr (std::is_same_v<T, uint16_t>) { return _mm256_mul_epu16(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int16_t>)  { return _mm256_mullo_epi16(a, b); } // AVX2
            else if constexpr (std::is_same_v<T, uint32_t>) { return _mm256_mul_epu32(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int32_t>)  { return _mm256_mullo_epi32(a, b); } // AVX2
            else if constexpr (std::is_same_v<T, uint64_t>) { return _mm256_mul_epu64(a, b); } // TODO
            else if constexpr (std::is_same_v<T, int64_t>)  { return _mm256_mullo_epi64(a, b); } // AVX512DQ + AVX512VL
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_mul_ps(a, b); } // AVX
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_mul_pd(a, b); } // AVX
    }

    template<typename T, typename V>
    V div(V const a, V const b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)       { return _mm_div_epu8(a, b); }  // SSE
            else if constexpr (std::is_same_v<T, int8_t>)   { return _mm_div_epi8(a, b); }  // SSE
            else if constexpr (std::is_same_v<T, uint16_t>) { return _mm_div_epu16(a, b); } // SSE
            else if constexpr (std::is_same_v<T, int16_t>)  { return _mm_div_epi16(a, b); } // SSE
            // else if constexpr (std::is_same_v<T, uint32_t>) { return _mm_div_epu32(a, b); } // SSE - FAIL - SVML (Intel only)
            // else if constexpr (std::is_same_v<T, int32_t>)  { return _mm_div_epi32(a, b); } // SSE - FAIL - SVML (Intel only)
            else if constexpr (std::is_same_v<T, uint64_t>) { return _mm_div_epu64(a, b); } // SSE
            else if constexpr (std::is_same_v<T, int64_t>)  { return _mm_div_epi64(a, b); } // SSE
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_div_ps(a, b); } // SSE
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_div_pd(a, b); } // SSE2
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)       { return _mm256_div_epu8(a, b); }  // AVX
            else if constexpr (std::is_same_v<T, int8_t>)   { return _mm256_div_epi8(a, b); }  // AVX
            else if constexpr (std::is_same_v<T, uint16_t>) { return _mm256_div_epu16(a, b); } // AVX
            else if constexpr (std::is_same_v<T, int16_t>)  { return _mm256_div_epi16(a, b); } // AVX
            // else if constexpr (std::is_same_v<T, uint32_t>) { return _mm256_div_epu32(a, b); } // AVX - FAIL - SVML (Intel only)
            // else if constexpr (std::is_same_v<T, int32_t>)  { return _mm256_div_epi32(a, b); } // AVX - FAIL - SVML (Intel only)
            else if constexpr (std::is_same_v<T, uint64_t>) { return _mm256_div_epu64(a, b); } // AVX
            else if constexpr (std::is_same_v<T, int64_t>)  { return _mm256_div_epi64(a, b); } // AVX
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_div_ps(a, b); } // AVX
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_div_pd(a, b); } // AVX
    }
}
