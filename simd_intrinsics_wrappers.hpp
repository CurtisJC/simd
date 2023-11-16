/*
 *  SIMD (single instruction multiple data) header library
 *
 *  Wrapper for intrinsic functions using templates where possible.
 */
#pragma once

#include <type_traits>
#include <immintrin.h>

#include "simd_intrinsics_alternatives.hpp"

namespace simd {
//-----------------------------------------------------------------------------
//  load instructions
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
//  store instructions
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
//  set instructions
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
//  addition instructions
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
//  subtraction instructions
//-----------------------------------------------------------------------------
    // 128bit vector integer subtraction
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128i>)
    V sub(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_sub_epi8(a, b); }  // SSE2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_sub_epi16(a, b); } // SSE2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_sub_epi32(a, b); } // SSE2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_sub_epi64(a, b); } // SSE2
    }

    // 128bit vector float subtraction
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V sub(V const a, V const b) { return _mm_sub_ps(a, b); } // SSE

    // 128bit vector double subtraction
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V sub(V const a, V const b) { return _mm_sub_pd(a, b); } // SSE2

    // 256bit vector integer subtraction
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256i>)
    V sub(V const a, V const b) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_sub_epi8(a, b); }  // AVX2
        else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_sub_epi16(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_sub_epi32(a, b); } // AVX2
        else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_sub_epi64(a, b); } // AVX2
    }

    // 256bit vector float subtraction
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V sub(V const a, V const b) { return _mm256_sub_ps(a, b); } // AVX

    // 256bit vector double subtraction
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V sub(V const a, V const b) { return _mm256_sub_pd(a, b); } // AVX

//-----------------------------------------------------------------------------
//  multiplication instructions
//-----------------------------------------------------------------------------
    // 128bit vector integer multiplication
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

    // 128bit vector float multiplication
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V mul(V const a, V const b) { return _mm_mul_ps(a, b); } // SSE

    // 128bit vector double multiplication
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V mul(V const a, V const b) { return _mm_mul_pd(a, b); } // SSE2

    // 256bit vector integer multiplication
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

    // 256bit vector float multiplication
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V mul(V const a, V const b) { return _mm256_mul_ps(a, b); } // AVX

    // 256bit vector double multiplication
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V mul(V const a, V const b) { return _mm256_mul_pd(a, b); } // AVX

//-----------------------------------------------------------------------------
//  division instructions
//-----------------------------------------------------------------------------
    // 128bit vector integer division
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

    // 128bit vector float division
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128>)
    V div(V const a, V const b) { return _mm_div_ps(a, b); } // SSE

    // 128bit vector double division
    template<typename T, typename V>
    requires (std::is_same_v<V, __m128d>)
    V div(V const a, V const b) { return _mm_div_pd(a, b); } // SSE2

    // 256bit vector integer division
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

    // 256bit vector float division
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256>)
    V div(V const a, V const b) { return _mm256_div_ps(a, b); } // AVX

    // 256bit vector double division
    template<typename T, typename V>
    requires (std::is_same_v<V, __m256d>)
    V div(V const a, V const b) { return _mm256_div_pd(a, b); } // AVX
}
