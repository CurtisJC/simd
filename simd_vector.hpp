/*
 *  SIMD (single instruction multiple data) header library
 *
 *  Staticically allocated array type that supports vectorised operations using intrinsics.
 */
#pragma once

#include <array>
#include <cstddef>
#include <concepts>
#include <immintrin.h>

#include "simd_intrinsics_wrappers.hpp"

namespace simd {
    template <typename T, std::size_t N, typename Cont = T[N ? N : 1]>
    requires ( std::is_arithmetic<T>::value == true )
    class vector {
    public:
        alignas(64) Cont data;

    #ifdef __AVX2__
        using vreg = std::conditional_t<std::is_integral_v<T>, __m256i, 
                     std::conditional_t<std::is_same_v<T, std::float32_t>, __m256, __m256d>>;
    #else
        using vreg = std::conditional_t<std::is_integral_v<T>, __m128i, 
                     std::conditional_t<std::is_same_v<T, std::float32_t>, __m128, __m128d>>;
    #endif // __AVX2__

        using iterator = T*;
        using const_iterator = T const*;

        inline iterator begin() noexcept { return data; }
        inline const_iterator cbegin() const noexcept { return data; }
        inline iterator end() noexcept { return data + N; }
        inline const_iterator cend() const noexcept { data + N; }

        T& operator[](std::size_t index) {
            return data[index];
        }

        const T& operator[](std::size_t index) const {
            return data[index];
        }

        vector<T, N> operator+(vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop<vreg>(i, data, other.data, result.data, [](vreg a, vreg b){ return add<T, vreg>(a, b); });
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

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, data, result.data, [](vreg a, vreg b){ return add<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = data[i] + s;
            }
            return result;
        }

        friend vector<T, N> operator+(T const& s, vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, other.data, result.data, [](vreg a, vreg b){ return add<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = s + other.data[i];
            }
            return result;
        }

        vector<T, N> operator-(vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop<vreg>(i, data, other.data, result.data, [](vreg a, vreg b){ return sub<T, vreg>(a, b); });
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

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, data, result.data, [](vreg a, vreg b){ return sub<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = data[i] - s;
            }
            return result;
        }

        friend vector<T, N> operator-(T const& s, vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, other.data, result.data, [](vreg a, vreg b){ return sub<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = s - other.data[i];
            }
            return result;
        }

        vector<T, N> operator*(vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop<vreg>(i, data, other.data, result.data, [](vreg a, vreg b){ return mul<T, vreg>(a, b); });
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

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, data, result.data, [](vreg a, vreg b){ return mul<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = data[i] * s;
            }
            return result;
        }

        friend vector<T, N> operator*(T const& s, vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, other.data, result.data, [](vreg a, vreg b){ return mul<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = s * other.data[i];
            }
            return result;
        }

        vector<T, N> operator/(vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop<vreg>(i, data, other.data, result.data, [](vreg a, vreg b){ return div<T, vreg>(a, b); });
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

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, data, result.data, [](vreg a, vreg b){ return div<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = data[i] / s;
            }
            return result;
        }

        friend vector<T, N> operator/(T const& s, vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop_scalar<vreg>(i, s, other.data, result.data, [](vreg a, vreg b){ return div<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = s / other.data[i];
            }
            return result;
        }

        vector<T, N> operator==(vector const& other)
        {
            vector<T, N> result;
            std::size_t i = 0;

            if constexpr ( (sizeof(vreg) / sizeof(T)) < N ) {
                simd_loop<vreg>(i, data, other.data, result.data, [](vreg a, vreg b){ return cmpeq<T, vreg>(a, b); });
            }

            for (; i < N; i++)
            {
                result[i] = data[i] == other.data[i] ? ~T(0) : 0;
            }
            return result;
        }

    private:
        template<typename V>
        static void simd_loop(std::size_t &i, T const a[N], T const b[N], T c[N], auto lamba_op)
        {
            const std::size_t VN = sizeof(V)/sizeof(T);
            if constexpr (N >= VN) {
                V va1, va2;
                V vb1, vb2;
                for (; i < N - (VN * 4); i += (VN * 4)) {
                    va1 = load<V>(&a[i]);
                    va2 = load<V>(&a[i+VN]);
                    vb1 = load<V>(&b[i]);
                    vb2 = load<V>(&b[i+VN]);
                    store<V>(&c[i], lamba_op(va1, vb1));
                    store<V>(&c[i+VN], lamba_op(va2, vb2));
                }
            }
        }

        template<typename V>
        static void simd_loop_scalar(std::size_t &i, T const a, T const b[N], T c[N], auto lamba_op)
        {
            const std::size_t VN = sizeof(V)/sizeof(T);
            if constexpr (N >= VN) {
                V va = set<T, V>(a);
                V vb1, vb2;
                for (; i < N - (VN * 2); i += (VN * 2)) {
                    vb1 = load<V>(&b[i]);
                    vb2 = load<V>(&b[i+VN]);
                    store<V>(&c[i], lamba_op(va, vb1));
                    store<V>(&c[i+VN], lamba_op(va, vb2));
                }
            }
        }
    };
}
