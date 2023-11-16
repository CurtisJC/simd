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

    static inline __m128i _mm_div_epi16(__m128i const& a, __m128i const& b) {
        // Setup the constants.
        const __m128  two     = _mm_set1_ps(2.00000051757f);
        const __m128i lo_mask = _mm_set1_epi32(0xFFFF);

        // Convert to two 32-bit integers
        const __m128i a_hi_epi32       = _mm_srai_epi32(a, 16);
        const __m128i a_lo_epi32       = _mm_srai_epi32(_mm_slli_epi32(a, 16), 16);
        const __m128i b_hi_epi32       = _mm_srai_epi32(b, 16);
        const __m128i b_lo_epi32       = _mm_srai_epi32(_mm_slli_epi32(b, 16), 16);

        // Convert to 32-bit floats
        const __m128 a_hi = _mm_cvtepi32_ps(a_hi_epi32);
        const __m128 a_lo = _mm_cvtepi32_ps(a_lo_epi32);
        const __m128 b_hi = _mm_cvtepi32_ps(b_hi_epi32);
        const __m128 b_lo = _mm_cvtepi32_ps(b_lo_epi32);

        // Calculate the reciprocal
        const __m128 b_hi_rcp = _mm_rcp_ps(b_hi);
        const __m128 b_lo_rcp = _mm_rcp_ps(b_lo);

        // Calculate the inverse
        #ifdef __FMA__
            const __m128 b_hi_inv_1 = _mm_fnmadd_ps(b_hi_rcp, b_hi, two);
            const __m128 b_lo_inv_1 = _mm_fnmadd_ps(b_lo_rcp, b_lo, two);
        #else
            const __m128 b_hi_inv_1 = _mm_sub_ps(two, _mm_mul_ps(b_hi_rcp, b_hi));
            const __m128 b_lo_inv_1 = _mm_sub_ps(two, _mm_mul_ps(b_lo_rcp, b_lo));
        #endif

        // Compensate for the loss
        const __m128 b_hi_rcp_1 = _mm_mul_ps(b_hi_rcp, b_hi_inv_1);
        const __m128 b_lo_rcp_1 = _mm_mul_ps(b_lo_rcp, b_lo_inv_1);

        // Perform the division by multiplication
        const __m128 hi = _mm_mul_ps(a_hi, b_hi_rcp_1);
        const __m128 lo = _mm_mul_ps(a_lo, b_lo_rcp_1);

        // Convert back to integers
        const __m128i hi_epi32 = _mm_cvttps_epi32(hi);
        const __m128i lo_epi32 = _mm_cvttps_epi32(lo);

        // Zero-out the unnecessary parts
        const __m128i hi_epi32_shift = _mm_slli_epi32(hi_epi32, 16);

        // Blend the bits, and return
        #ifdef __SSE4_1__
            return _mm_blend_epi16(lo_epi32, hi_epi32_shift, 0xAA);
        #else
            return _mm_or_si128(hi_epi32_shift, _mm_and_si128(lo_epi32, const_mm_div_epi16_lo_mask));
        #endif
    }

    static inline __m256i _mm256_div_epi16(__m256i const& a, __m256i const& b) {
        // Setup the constants.
        const __m256 two = _mm256_set1_ps(2.00000051757f);

        // Convert to two 32-bit integers
        const __m256i a_hi_epi32       = _mm256_srai_epi32(a, 16);
        const __m256i a_lo_epi32       = _mm256_srai_epi32(_mm256_slli_epi32(a, 16), 16);
        const __m256i b_hi_epi32       = _mm256_srai_epi32(b, 16);
        const __m256i b_lo_epi32       = _mm256_srai_epi32(_mm256_slli_epi32(b, 16), 16);

        // Convert to 32-bit floats
        const __m256 a_hi = _mm256_cvtepi32_ps(a_hi_epi32);
        const __m256 a_lo = _mm256_cvtepi32_ps(a_lo_epi32);
        const __m256 b_hi = _mm256_cvtepi32_ps(b_hi_epi32);
        const __m256 b_lo = _mm256_cvtepi32_ps(b_lo_epi32);

        // Calculate the reciprocal
        const __m256 b_hi_rcp = _mm256_rcp_ps(b_hi);
        const __m256 b_lo_rcp = _mm256_rcp_ps(b_lo);

        // Calculate the inverse
        // Compensate for the loss
        // Perform the division by multiplication
        const __m256 hi = _mm256_mul_ps(a_hi, _mm256_mul_ps(b_hi_rcp, _mm256_fnmadd_ps(b_hi_rcp, b_hi, two)));
        const __m256 lo = _mm256_mul_ps(a_lo, _mm256_mul_ps(b_lo_rcp, _mm256_fnmadd_ps(b_lo_rcp, b_lo, two)));

        // Convert back to integers
        // Blend the low and the high-parts
        const __m256i hi_epi32_shift = _mm256_slli_epi32(_mm256_cvttps_epi32(hi), 16);
        return _mm256_blend_epi16(_mm256_cvttps_epi32(lo), hi_epi32_shift, 0xAA);
    }

    template<typename V>
    requires (std::is_same_v<V, __m128i>)
    V load(void const* mem_addr) { return _mm_loadu_si128((__m128i_u*)mem_addr); }

    template<typename V>
    requires (std::is_same_v<V, __m128>)
    V load(void const* mem_addr) { return _mm_loadu_ps((float*)mem_addr); }

    template<typename V>
    requires (std::is_same_v<V, __m128d>)
    V load(void const* mem_addr) { return _mm_loadu_pd((double*)mem_addr); }

    template<typename V>
    requires (std::is_same_v<V, __m256i>)
    V load(void const* mem_addr) { return _mm256_loadu_si256((__m256i_u*)mem_addr); }

    template<typename V>
    requires (std::is_same_v<V, __m256>)
    V load(void const* mem_addr) { return _mm256_loadu_ps((float*)mem_addr); }

    template<typename V>
    requires (std::is_same_v<V, __m256d>)
    V load(void const* mem_addr) { return _mm256_loadu_pd((double*)mem_addr); }

    template<typename V>
    requires (std::is_same_v<V, __m128i>)
    void store(void const* mem_addr, V a) { _mm_storeu_si128((__m128i_u*)mem_addr, a); }

    template<typename V>
    requires (std::is_same_v<V, __m128>)
    void store(void const* mem_addr, V a) { _mm_storeu_ps((float*)mem_addr, a); }

    template<typename V>
    requires (std::is_same_v<V, __m128d>)
    void store(void const* mem_addr, V a) { _mm_storeu_pd((double*)mem_addr, a); }

    template<typename V>
    requires (std::is_same_v<V, __m256i>)
    void store(void const* mem_addr, V a) { _mm256_storeu_si256((__m256i_u*)mem_addr, a); }

    template<typename V>
    requires (std::is_same_v<V, __m256>)
    void store(void const* mem_addr, V a) { _mm256_storeu_ps((float*)mem_addr, a); }

    template<typename V>
    requires (std::is_same_v<V, __m256d>)
    void store(void const* mem_addr, V a) { _mm256_storeu_pd((double*)mem_addr, a); }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128i>)
    V set(T const s) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_set1_epi8(s); }  // SSE2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_set1_epi16(s); } // SSE2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_set1_epi32(s); } // SSE2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_set1_epi64(s); } // SSE2
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V set(T const s) { return _mm_set1_ps(s); } // SSE

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V set(T const s) { return _mm_set1_pd(s); } // SSE2

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256i>)
    V set(T const s) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_set1_epi8(s); }  // AVX2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_set1_epi16(s); } // AVX2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_set1_epi32(s); } // AVX2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_set1_epi64(s); } // AVX2
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V set(T const s) { return _mm256_set1_ps(s); } // AVX

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V set(T const s) { return _mm256_set1_pd(s); } // AVX

    // 128bit vector integer addition
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128i>)
    V add(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_add_epi8(a, b); }  // SSE2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_add_epi16(a, b); } // SSE2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_add_epi32(a, b); } // SSE2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_add_epi64(a, b); } // SSE2
    }

    // 128bit vector float addition
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V add(V const a, V const b) { return _mm_add_ps(a, b); } // SSE

    // 128bit vector double addition
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V add(V const a, V const b) { return _mm_add_pd(a, b); } // SSE2

    // 256bit vector integer addition
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256i>)
    V add(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_add_epi8(a, b); }  // AVX2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_add_epi16(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_add_epi32(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_add_epi64(a, b); } // AVX2
    }

    // 256bit vector float addition
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V add(V const a, V const b) { return _mm256_add_ps(a, b); } // AVX

    // 256bit vector double addition
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V add(V const a, V const b) { return _mm256_add_pd(a, b); } // AVX

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128i>)
    V sub(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_sub_epi8(a, b); }  // SSE2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_sub_epi16(a, b); } // SSE2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_sub_epi32(a, b); } // SSE2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_sub_epi64(a, b); } // SSE2
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V sub(V const a, V const b) { return _mm_sub_ps(a, b); } // SSE

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V sub(V const a, V const b) { return _mm_sub_pd(a, b); } // SSE2

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256i>)
    V sub(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_sub_epi8(a, b); }  // AVX2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_sub_epi16(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_sub_epi32(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_sub_epi64(a, b); } // AVX2
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V sub(V const a, V const b) { return _mm256_sub_ps(a, b); } // AVX

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V sub(V const a, V const b) { return _mm256_sub_pd(a, b); } // AVX

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128i>)
    V mul(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t>)       { return _mm_mul_epu8(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int8_t>)   { return _mm_mul_epi8(a, b); } // TODO
        else if constexpr (std::is_same_v<T, uint16_t>) { return _mm_mul_epu16(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int16_t>)  { return _mm_mullo_epi16(a, b); } // SSE2
        else if constexpr (std::is_same_v<T, uint32_t>) { return _mm_mul_epu32(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int32_t>)  { return _mm_mullo_epi32(a, b); } // SSE4.1
        else if constexpr (std::is_same_v<T, uint64_t>) { return _mm_mul_epu64(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int64_t>)  { return _mm_mullo_epi64(a, b); } // AVX512DQ + AVX512VL
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V mul(V const a, V const b) { return _mm_mul_ps(a, b); } // SSE

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V mul(V const a, V const b) { return _mm_mul_pd(a, b); } // SSE2

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256i>)
    V mul(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t>)       { return _mm256_mul_epu8(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int8_t>)   { return _mm256_mul_epi8(a, b); } // TODO
        else if constexpr (std::is_same_v<T, uint16_t>) { return _mm256_mul_epu16(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int16_t>)  { return _mm256_mullo_epi16(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint32_t>) { return _mm256_mul_epu32(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int32_t>)  { return _mm256_mullo_epi32(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint64_t>) { return _mm256_mul_epu64(a, b); } // TODO
        else if constexpr (std::is_same_v<T, int64_t>)  { return _mm256_mullo_epi64(a, b); } // AVX512DQ + AVX512VL
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V mul(V const a, V const b) { return _mm256_mul_ps(a, b); } // AVX

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V mul(V const a, V const b) { return _mm256_mul_pd(a, b); } // AVX

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128i>)
    V div(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t>)       { return _mm_div_epu8(a, b); }  // SSE - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int8_t>)   { return _mm_div_epi8(a, b); }  // SSE - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, uint16_t>) { return _mm_div_epu16(a, b); } // SSE - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int16_t>)  { return _mm_div_epi16(a, b); } // SSE - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, uint32_t>) { return _mm_div_epu32(a, b); } // SSE - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int32_t>)  { return _mm_div_epi32(a, b); } // SSE - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, uint64_t>) { return _mm_div_epu64(a, b); } // SSE - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int64_t>)  { return _mm_div_epi64(a, b); } // SSE - FAIL - SVML (Intel only)
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V div(V const a, V const b) { return _mm_div_ps(a, b); } // SSE

    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V div(V const a, V const b) { return _mm_div_pd(a, b); } // SSE2

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256i>)
    V div(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t>)       { return _mm256_div_epu8(a, b); }  // AVX - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int8_t>)   { return _mm256_div_epi8(a, b); }  // AVX - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, uint16_t>) { return _mm256_div_epu16(a, b); } // AVX - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int16_t>)  { return _mm256_div_epi16(a, b); } // AVX - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, uint32_t>) { return _mm256_div_epu32(a, b); } // AVX - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int32_t>)  { return _mm256_div_epi32(a, b); } // AVX - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, uint64_t>) { return _mm256_div_epu64(a, b); } // AVX - FAIL - SVML (Intel only)
        else if constexpr (std::is_same_v<T, int64_t>)  { return _mm256_div_epi64(a, b); } // AVX - FAIL - SVML (Intel only)
    }

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V div(V const a, V const b) { return _mm256_div_ps(a, b); } // AVX

    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V div(V const a, V const b) { return _mm256_div_pd(a, b); } // AVX
}
