/*
 *  SIMD (single instruction multiple data) header library
 *
 *  Wrapper for __m* types found in the intrinsics.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
//#include <stdfloat> // std::float32_t, std::float64_t
#include <immintrin.h>

#include "simd_intrinsics_wrappers.hpp"

namespace simd {

    template<typename T, typename V, size_t N = sizeof(V)/sizeof(T)>
    class __SIMDVec
    {
    private:
        V vec;

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
            vec = __SIMDVec(add<T, V>(vec, other.vec));
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
            vec = __SIMDVec(mul<T, V>(vec, other.vec));
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
}