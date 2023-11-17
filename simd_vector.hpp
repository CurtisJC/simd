/*
 *  SIMD (single instruction multiple data) header library
 *
 *  Extension to std::array that supports vectorised operations using intrinsics.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
//#include <stdfloat> // std::float32_t, std::float64_t
#include <immintrin.h>

#include "simd_intrinsics_wrappers.hpp"

namespace simd {
    template <typename T, std::size_t N>
    class vector {
    public:
        std::array<T, N> data;

        T& operator[](std::size_t index) {
            return data[index];
        }

        const T& operator[](std::size_t index) const {
            return data[index];
        }

        vector<T, N> operator+(vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop<__m256i>(i, data, other.data, result.data, [](__m256i a, __m256i b){ return add<T, __m256i>(a, b); });
            #else
                simd_loop<__m128i>(i, data, other.data, result.data, [](__m128i a, __m128i b){ return add<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256>(i, data, other.data, result.data, [](__m256 a, __m256 b){ return add<T, __m256>(a, b); });
            #else
                simd_loop<__m128>(i, data, other.data, result.data, [](__m128 a, __m128 b){ return add<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256d>(i, data, other.data, result.data, [](__m256d a, __m256d b){ return add<T, __m256d>(a, b); });
            #else
                simd_loop<__m128d>(i, data, other.data, result.data, [](__m128d a, __m128d b){ return add<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] + other.data[i];
            }

            return result;
        }

        vector<T, N> operator+(T const& s)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, data, result.data, [](__m256i a, __m256i b){ return add<T, __m256i>(a, b); });
            #else
                simd_loop_scalar<__m128i>(i, s, data, result.data, [](__m128i a, __m128i b){ return add<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, data, result.data, [](__m256 a, __m256 b){ return add<T, __m256>(a, b); });
            #else
                simd_loop_scalar<__m128>(i, s, data, result.data, [](__m128 a, __m128 b){ return add<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, data, result.data, [](__m256d a, __m256d b){ return add<T, __m256d>(a, b); });
            #else
                simd_loop_scalar<__m128d>(i, s, data, result.data, [](__m128d a, __m128d b){ return add<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] + s;
            }
            return result;
        }

        friend vector<T, N> operator+(T const& s, const vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, other.data, result.data, [](__m256i a, __m256i b){ return add<T, __m256i>(a, b); });
            #else
                simd_loop_scalar<__m128i>(i, s, other.data, result.data, [](__m128i a, __m128i b){ return add<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, other.data, result.data, [](__m256 a, __m256 b){ return add<T, __m256>(a, b); });
            #else
                simd_loop_scalar<__m128>(i, s, other.data, result.data, [](__m128 a, __m128 b){ return add<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, other.data, result.data, [](__m256d a, __m256d b){ return add<T, __m256d>(a, b); });
            #else
                simd_loop_scalar<__m128d>(i, s, other.data, result.data, [](__m128d a, __m128d b){ return add<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = s + other.data[i];
            }
            return result;
        }

        vector<T, N> operator-(vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop<__m256i>(i, data, other.data, result.data, [](__m256i a, __m256i b){ return sub<T, __m256i>(a, b); });
            #else
                simd_loop<__m128i>(i, data, other.data, result.data, [](__m128i a, __m128i b){ return sub<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256>(i, data, other.data, result.data, [](__m256 a, __m256 b){ return sub<T, __m256>(a, b); });
            #else
                simd_loop<__m128>(i, data, other.data, result.data, [](__m128 a, __m128 b){ return sub<T, __m128>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256d>(i, data, other.data, result.data, [](__m256d a, __m256d b){ return sub<T, __m256d>(a, b); });
            #else
                simd_loop<__m128d>(i, data, other.data, result.data, [](__m128d a, __m128d b){ return sub<T, __m128d>(a, b); });
            #endif // __AVX2__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] - other.data[i];
            }
            return result;
        }

        vector<T, N> operator-(T const& s)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, data, result.data, [](__m256i a, __m256i b){ return sub<T, __m256i>(b, a); });
            #else
                simd_loop_scalar<__m128i>(i, s, data, result.data, [](__m128i a, __m128i b){ return sub<T, __m128i>(b, a); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, data, result.data, [](__m256 a, __m256 b){ return sub<T, __m256>(b, a); });
            #else
                simd_loop_scalar<__m128>(i, s, data, result.data, [](__m128 a, __m128 b){ return sub<T, __m128>(b, a); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, data, result.data, [](__m256d a, __m256d b){ return sub<T, __m256d>(b, a); });
            #else
                simd_loop_scalar<__m128d>(i, s, data, result.data, [](__m128d a, __m128d b){ return sub<T, __m128d>(b, a); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] - s;
            }
            return result;
        }

        friend vector<T, N> operator-(T const& s, const vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, other.data, result.data, [](__m256i a, __m256i b){ return sub<T, __m256i>(a, b); });
            #else
                simd_loop_scalar<__m128i>(i, s, other.data, result.data, [](__m128i a, __m128i b){ return sub<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, other.data, result.data, [](__m256 a, __m256 b){ return sub<T, __m256>(a, b); });
            #else
                simd_loop_scalar<__m128>(i, s, other.data, result.data, [](__m128 a, __m128 b){ return sub<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, other.data, result.data, [](__m256d a, __m256d b){ return sub<T, __m256d>(a, b); });
            #else
                simd_loop_scalar<__m128d>(i, s, other.data, result.data, [](__m128d a, __m128d b){ return sub<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = s - other.data[i];
            }
            return result;
        }

        vector<T, N> operator*(vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop<__m256i>(i, data, other.data, result.data, [](__m256i a, __m256i b){ return mul<T, __m256i>(a, b); });
            #else
                simd_loop<__m128i>(i, data, other.data, result.data, [](__m128i a, __m128i b){ return mul<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256>(i, data, other.data, result.data, [](__m256 a, __m256 b){ return mul<T, __m256>(a, b); });
            #else
                simd_loop<__m128>(i, data, other.data, result.data, [](__m128 a, __m128 b){ return mul<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256d>(i, data, other.data, result.data, [](__m256d a, __m256d b){ return mul<T, __m256d>(a, b); });
            #else
                simd_loop<__m128d>(i, data, other.data, result.data, [](__m128d a, __m128d b){ return mul<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] * other.data[i];
            }
            return result;
        }

        vector<T, N> operator*(T const& s)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, data, result.data, [](__m256i a, __m256i b){ return mul<T, __m256i>(a, b); });
            #else
                simd_loop_scalar<__m128i>(i, s, data, result.data, [](__m128i a, __m128i b){ return mul<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, data, result.data, [](__m256 a, __m256 b){ return mul<T, __m256>(a, b); });
            #else
                simd_loop_scalar<__m128>(i, s, data, result.data, [](__m128 a, __m128 b){ return mul<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, data, result.data, [](__m256d a, __m256d b){ return mul<T, __m256d>(a, b); });
            #else
                simd_loop_scalar<__m128d>(i, s, data, result.data, [](__m128d a, __m128d b){ return add<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] * s;
            }
            return result;
        }

        friend vector<T, N> operator*(T const& s, const vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, other.data, result.data, [](__m256i a, __m256i b){ return mul<T, __m256i>(a, b); });
            #else
                simd_loop_scalar<__m128i>(i, s, other.data, result.data, [](__m128i a, __m128i b){ return mul<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, other.data, result.data, [](__m256 a, __m256 b){ return mul<T, __m256>(a, b); });
            #else
                simd_loop_scalar<__m128>(i, s, other.data, result.data, [](__m128 a, __m128 b){ return mul<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, other.data, result.data, [](__m256d a, __m256d b){ return mul<T, __m256d>(a, b); });
            #else
                simd_loop_scalar<__m128d>(i, s, other.data, result.data, [](__m128d a, __m128d b){ return mul<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = s * other.data[i];
            }
            return result;
        }

        vector<T, N> operator/(vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop<__m256i>(i, data, other.data, result.data, [](__m256i a, __m256i b){ return div<T, __m256i>(a, b); });
            #else
                simd_loop<__m128i>(i, data, other.data, result.data, [](__m128i a, __m128i b){ return div<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256>(i, data, other.data, result.data, [](__m256 a, __m256 b){ return div<T, __m256>(a, b); });
            #else
                simd_loop<__m128>(i, data, other.data, result.data, [](__m128 a, __m128 b){ return div<T, __m128>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256d>(i, data, other.data, result.data, [](__m256d a, __m256d b){ return div<T, __m256d>(a, b); });
            #else
                simd_loop<__m128d>(i, data, other.data, result.data, [](__m128d a, __m128d b){ return div<T, __m128d>(a, b); });
            #endif // __AVX2__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] / other.data[i];
            }
            return result;
        }

        vector<T, N> operator/(T const& s)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, data, result.data, [](__m256i a, __m256i b){ return div<T, __m256i>(b, a); });
            #else
                simd_loop_scalar<__m128i>(i, s, data, result.data, [](__m128i a, __m128i b){ return div<T, __m128i>(b, a); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, data, result.data, [](__m256 a, __m256 b){ return div<T, __m256>(b, a); });
            #else
                simd_loop_scalar<__m128>(i, s, data, result.data, [](__m128 a, __m128 b){ return div<T, __m128>(b, a); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, data, result.data, [](__m256d a, __m256d b){ return div<T, __m256d>(b, a); });
            #else
                simd_loop_scalar<__m128d>(i, s, data, result.data, [](__m128d a, __m128d b){ return div<T, __m128d>(b, a); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] / s;
            }
            return result;
        }

        friend vector<T, N> operator/(T const& s, const vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop_scalar<__m256i>(i, s, other.data, result.data, [](__m256i a, __m256i b){ return div<T, __m256i>(a, b); });
            #else
                simd_loop_scalar<__m128i>(i, s, other.data, result.data, [](__m128i a, __m128i b){ return div<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256>(i, s, other.data, result.data, [](__m256 a, __m256 b){ return div<T, __m256>(a, b); });
            #else
                simd_loop_scalar<__m128>(i, s, other.data, result.data, [](__m128 a, __m128 b){ return div<T, __m128>(a, b); });
            #endif // __AVX__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop_scalar<__m256d>(i, s, other.data, result.data, [](__m256d a, __m256d b){ return div<T, __m256d>(a, b); });
            #else
                simd_loop_scalar<__m128d>(i, s, other.data, result.data, [](__m128d a, __m128d b){ return div<T, __m128d>(a, b); });
            #endif // __AVX__
            }

            for (; i < N; i++)
            {
                result[i] = s / other.data[i];
            }
            return result;
        }

        vector<T, N> operator==(vector& other)
        {
            vector<T, N> result;
            std::size_t i = 0;
            if constexpr (std::is_integral<T>::value)
            {
            #ifdef __AVX2__
                simd_loop<__m256i>(i, data, other.data, result.data, [](__m256i a, __m256i b){ return cmpeq<T, __m256i>(a, b); });
            #else
                simd_loop<__m128i>(i, data, other.data, result.data, [](__m128i a, __m128i b){ return cmpeq<T, __m128i>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, float>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256>(i, data, other.data, result.data, [](__m256 a, __m256 b){ return cmpeq<T, __m256>(a, b); });
            #else
                simd_loop<__m128>(i, data, other.data, result.data, [](__m128 a, __m128 b){ return cmpeq<T, __m128>(a, b); });
            #endif // __AVX2__
            }
            else if constexpr (std::is_same<T, double>::value)
            {
            #ifdef __AVX__
                simd_loop<__m256d>(i, data, other.data, result.data, [](__m256d a, __m256d b){ return cmpeq<T, __m256d>(a, b); });
            #else
                simd_loop<__m128d>(i, data, other.data, result.data, [](__m128d a, __m128d b){ return cmpeq<T, __m128d>(a, b); });
            #endif // __AVX2__
            }

            for (; i < N; i++)
            {
                result[i] = data[i] == other.data[i] ? ~T(0) : 0;
            }
            return result;
        }

    private:
        template<typename V>
        static void simd_loop(std::size_t &i, std::array<T, N> const& a, std::array<T, N> const& b, std::array<T, N>& c, auto lamba_op)
        {
            const std::size_t VN = sizeof(V)/sizeof(T);
            if constexpr (N >= VN) {
                for (; i < N - VN; i += VN) {
                    V va = load<V>(&a[i]);
                    V vb = load<V>(&b[i]);
                    store<V>(&c[i], lamba_op(va, vb));
                }
            }
        }

        template<typename V>
        static void simd_loop_scalar(std::size_t &i, T const a, std::array<T, N> const& b, std::array<T, N> &c, auto lamba_op)
        {
            const std::size_t VN = sizeof(V)/sizeof(T);
            if constexpr (N >= VN) {
                V va = set<T, V>(a);
                for (; i < N - VN; i += VN) {
                    V vb = load<V>(&b[i]);
                    store<V>(&c[i], lamba_op(va, vb));
                }
            }
        }
    };
}
