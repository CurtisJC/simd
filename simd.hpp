/*
 *  SIMD (single instruction multiple data) header library
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <type_traits>
#include <immintrin.h>


namespace simd {

    __m128i __mm_mul_epi8(__m128i a, __m128i b)
    {
        __m128i mask = _mm_set1_epi16(0xff00);
        // mask higher bytes:
        __m128i a_hi = _mm_and_si128(a, mask);
        __m128i b_hi = _mm_and_si128(b, mask);

        __m128i r_hi = _mm_mulhi_epi16(a_hi, b_hi);
        // mask out garbage in lower half:
        r_hi = _mm_and_si128(r_hi, mask);

        // shift lower bytes to upper half
        __m128i a_lo = _mm_slli_epi16(a,8);
        __m128i b_lo = _mm_slli_epi16(b,8);
        __m128i r_lo = _mm_mulhi_epi16(a_lo, b_lo);
        // shift result to the lower half:
        r_lo = _mm_srli_epi16(r_lo,8);

        // join result and return:
        return _mm_or_si128(r_hi, r_lo);
    }

    __m128i __mm_mul_epu16(__m128i a, __m128i b)
    {
        // Convert the input integers to unsigned 32-bit integers.
        __m128i a_u32 = _mm_cvtpu16_epi32(a);
        __m128i b_u32 = _mm_cvtpu16_epi32(b);

        // Multiply the two integers using the _mm_mul_epi32 intrinsic.
        __m128i product = _mm_mul_epi32(a_u32, b_u32);

        // Convert the product back to unsigned 16-bit integers.
        return _mm_cvtepi32_epi16(product);
    }

    __m128i __mm_mul_epi16(__m128i a, __m128i b)
    {
        // Perform high-order and low-order multiplications.
        __m128i high = _mm_mulhi_epi16(a, b);
        __m128i low = _mm_mullo_epi16(a, b);

        // Combine the high-order and low-order products.
        return _mm_add_epi16(high, _mm_slli_epi16(low, 16));
    }

    template<typename V>
    V load(void *mem_addr)
    {
        if constexpr (std::is_same_v<V, __m128i>)      { return _mm_loadu_si128((__m128i_u*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_loadu_ps((float*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_loadu_pd((double*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m256i>) { return _mm256_loadu_si256((__m256i_u*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_loadu_ps((float*)mem_addr); }
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_loadu_pd((double*)mem_addr); }
    }

    template<typename V>
    void store(void *mem_addr, V a)
    {
        if constexpr (std::is_same_v<V, __m128i>)      { _mm_storeu_si128((__m128i_u*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m128>)  { _mm_storeu_ps((float*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m128d>) { _mm_storeu_pd((double*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m256i>) { _mm256_storeu_si256((__m256i_u*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m256>)  { _mm256_storeu_ps((float*)mem_addr, a); }
        else if constexpr (std::is_same_v<V, __m256d>) { _mm256_storeu_pd((double*)mem_addr, a); }
    }

    template<typename T, typename V>
    V add(V a, V b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_add_epi8(a, b); }
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_add_epi16(a, b); }
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_add_epi32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_add_epi64(a, b); }
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_add_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_add_pd(a, b); }
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_add_epi8(a, b); }
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_add_epi16(a, b); }
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_add_epi32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_add_epi64(a, b); }
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_add_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_add_pd(a, b); }
    }

    template<typename T, typename V>
    V sub(V a, V b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm_sub_epi8(a, b); }
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm_sub_epi16(a, b); }
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm_sub_epi32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm_sub_epi64(a, b); }
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_sub_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_sub_pd(a, b); }
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)        { return _mm256_sub_epi8(a, b); }
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>) { return _mm256_sub_epi16(a, b); }
            else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) { return _mm256_sub_epi32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) { return _mm256_sub_epi64(a, b); }
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_sub_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_sub_pd(a, b); }
    }

    template<typename T, typename V>
    V mul(V a, V b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)
            {
                // TODO: __mm_mul_epu8
            }
            else if constexpr (std::is_same_v<T, int8_t>)   { return __mm_mul_epi8(a, b); }
            else if constexpr (std::is_same_v<T, uint16_t>) { return _mm_mul_epu16(a, b); }
            else if constexpr (std::is_same_v<T, int16_t>)  { return _mm_mul_epi16(a, b); }
            else if constexpr (std::is_same_v<T, uint32_t>) { return _mm_mul_epi32(a, b); }
            else if constexpr (std::is_same_v<T, int32_t>)  { return _mm_mul_epu32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>)
            {
                // TODO: _mm_mul_epu64
                // TODO: _mm_mul_epi64
            }
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_mul_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_mul_pd(a, b); }
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
            {
                // TODO: _mm256_mul_epu8(a, b);
                // TODO: _mm256_mul_epi8(a, b);
            }
            else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t>)
            {
                // TODO: _mm256_mul_epu16
                // TODO: _mm256_mul_epi16
            }
            else if constexpr (std::is_same_v<T, uint32_t>)     { return _mm256_mul_epi32(a, b); }
            else if constexpr (std::is_same_v<T, int32_t>) { return _mm256_mul_epu32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>)
            {
                // TODO: _mm256_mul_epu64(a, b);
                // TODO: _mm256_mul_epi64(a, b);
            }
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_mul_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_mul_pd(a, b); }
    }

    template<typename T, typename V>
    V div(V a, V b)
    {
        if constexpr (std::is_same_v<V, __m128i>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)       { return _mm_div_epu8(a, b); }
            else if constexpr (std::is_same_v<T, int8_t>)   { return _mm_div_epi8(a, b); }
            else if constexpr (std::is_same_v<T, uint16_t>) { return _mm_div_epi16(a, b); }
            else if constexpr (std::is_same_v<T, int16_t>)  { return _mm_div_epu16(a, b); }
            else if constexpr (std::is_same_v<T, uint32_t>) { return _mm_div_epi32(a, b); }
            else if constexpr (std::is_same_v<T, int32_t>)  { return _mm_div_epu32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t>) { return _mm_div_epu64(a, b); }
            else if constexpr (std::is_same_v<T, int64_t>)  { return _mm_div_epi64(a, b); }
        }
        else if constexpr (std::is_same_v<V, __m128>)  { return _mm_div_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m128d>) { return _mm_div_pd(a, b); }
        else if constexpr (std::is_same_v<V, __m256i>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)       { return _mm256_div_epu8(a, b); }
            else if constexpr (std::is_same_v<T, int8_t>)   { return _mm256_div_epi8(a, b); }
            else if constexpr (std::is_same_v<T, uint16_t>) { return _mm256_div_epi16(a, b); }
            else if constexpr (std::is_same_v<T, int16_t>)  { return _mm256_div_epu16(a, b); }
            else if constexpr (std::is_same_v<T, uint32_t>) { return _mm256_div_epi32(a, b); }
            else if constexpr (std::is_same_v<T, int32_t>)  { return _mm256_div_epu32(a, b); }
            else if constexpr (std::is_same_v<T, uint64_t>) { return _mm256_div_epu64(a, b); }
            else if constexpr (std::is_same_v<T, int64_t>)  { return _mm256_div_epi64(a, b); }
        }
        else if constexpr (std::is_same_v<V, __m256>)  { return _mm256_div_ps(a, b); }
        else if constexpr (std::is_same_v<V, __m256d>) { return _mm256_div_pd(a, b); }
    }

    template<typename T, typename V, size_t N = sizeof(V)/sizeof(T)>
    class __SIMDVec
    {
    public:
        explicit __SIMDVec(T *mem_addr) noexcept : vec(load<V>(mem_addr)) {};
        explicit __SIMDVec(std::array<T, N> &local) noexcept : vec(load<V>(local.data())) {};
        explicit __SIMDVec(V other) noexcept : vec(other) {};

        std::array<T, N> to_local()
        {
            std::array<T, N> local;
            store<V>(local.data(), vec);
            return local;
        }

        __SIMDVec operator +(const __SIMDVec &other) const
        {
            return __SIMDVec(add<T, V>(vec, other.vec));
        }

        __SIMDVec& operator +=(const __SIMDVec &other) 
        {
            vec = __SIMDVec(add<T, V>(vec, other.vec))
            return *this;
        }

        __SIMDVec operator -(const __SIMDVec &other) const
        {
            return __SIMDVec(sub<T, V>(vec, other.vec));
        }

        __SIMDVec& operator -=(const __SIMDVec &other)
        {
            vec = __SIMDVec(sub<T, V>(vec, other.vec));
            return *this;
        }

        __SIMDVec operator *(const __SIMDVec &other) const
        {
            return __SIMDVec(mul<T, V>(vec, other.vec));
        }

        __SIMDVec& operator *=(const __SIMDVec &other)
        {
            vec = __SIMDVec(mul<T, V>(vec, other.vec))
            return *this;
        }

        __SIMDVec operator /(const __SIMDVec &other) const
        {
            return __SIMDVec(div<T, V>(vec, other.vec));
        }

        __SIMDVec& operator /=(const __SIMDVec &other)
        {
            vec = __SIMDVec(div<T, V>(vec, other.vec));
            return *this;
        }

    private:

        V vec;
    };

#ifdef __SSE2__
#ifdef __AVX__
    using vec16_int8_t = __SIMDVec<int8_t, __m128i>;
    using vec8_int16_t = __SIMDVec<int16_t, __m128i>;
#ifdef __SSE4_1__
    using vec4_int32_t = __SIMDVec<int32_t, __m128i>;
    using vec2_int64_t = __SIMDVec<int64_t, __m128i>;
#endif // __SSE4_1__

    using vec16_uint8_t = __SIMDVec<uint8_t, __m128i>;
    using vec8_uint16_t = __SIMDVec<uint16_t, __m128i>;
    using vec4_uint32_t = __SIMDVec<uint32_t, __m128i>;
    using vec2_uint64_t = __SIMDVec<uint64_t, __m128i>;

    using vec4_float32_t = __SIMDVec<float, __m128>;
    using vec2_float64_t = __SIMDVec<double, __m128>;

#ifdef __AVX2__
    using vec32_int8_t = __SIMDVec<int8_t, __m256i>;
    using vec16_int16_t = __SIMDVec<int16_t, __m256i>;
    using vec8_int32_t = __SIMDVec<int32_t, __m256i>;
    using vec4_int64_t = __SIMDVec<int64_t, __m256i>;

    using vec32_uint8_t = __SIMDVec<uint8_t, __m256i>;
    using vec16_uint16_t = __SIMDVec<uint16_t, __m256i>;
    using vec8_uint32_t = __SIMDVec<uint32_t, __m256i>;
    using vec4_uint64_t = __SIMDVec<uint64_t, __m256i>;
#endif // __AVX2__

    using vec8_float32_t = __SIMDVec<float, __m256>;
    using vec4_float64_t = __SIMDVec<double, __m256>;
#endif // __AVX__
#endif // __SSE2__
};