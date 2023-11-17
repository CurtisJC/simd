
/*
 *  SIMD (single instruction multiple data) header library
 *
 *  Alternatives have been implemented where intrinsic functions do not exist to provide a level of abstraction to the
 *  underlying operations
 */
#pragma once

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

//-----------------------------------------------------------------------------
//  multiplication instructions
//-----------------------------------------------------------------------------
    inline __m128i simd_mul_si32(__m128i const& a, __m128i const& b) {
    #ifdef __SSE4_1__
        return _mm_mullo_epi32(a, b);
    #else // SSE 2
        __m128i tmp1 = _mm_mul_epu32(a,b); // mul 2,0
        __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); // mul 3,1
        return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); // shuffle results to [63..0] and pack */
    #endif // __SSE4_1__
    }
    
    inline __m256i simd_mul_si32(__m256i const& a, __m256i const& b) {
        return _mm256_mullo_epi32(a, b);
    }

    inline __m128i simd_mul_ui32(__m128i const& a, __m128i const& b) {
        // STUB
        return a;
    }

    inline __m256i simd_mul_ui32(__m256i const& a, __m256i const& b) {
        // STUB
        return a;
    }

//-----------------------------------------------------------------------------
//  division instructions
//-----------------------------------------------------------------------------
    inline __m128i simd_div_si16(__m128i const& a, __m128i const& b) {
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

    inline __m256i simd_div_si16(__m256i const& a, __m256i const& b) {
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

    inline __m128i simd_div_ui32(__m128i const& a, __m128i const& b) {
        // Setup the constants.
        const __m128 two = _mm_set1_ps(2.00000051757f);

        // Convert to 32-bit floats
        const __m128 a_f = _mm_cvtepi32_ps(a);
        const __m128 b_f = _mm_cvtepi32_ps(b);

        // Calculate the reciprocal
        const __m128 b_rcp = _mm_rcp_ps(b_f);

        // Calculate the inverse
        // Compensate for the loss
        // Perform the division by multiplication
        const __m128 f = _mm_mul_ps(a_f, _mm_mul_ps(b_rcp, _mm_fnmadd_ps(b_rcp, b_f, two)));

        // Convert back to integers
        return _mm_cvttps_epi32(f);
    }

    inline __m256i simd_div_ui32(__m256i const& a, __m256i const& b) {
        // Setup the constants.
        const __m256 two = _mm256_set1_ps(2.00000051757f);

        // Convert to 32-bit floats
        const __m256 a_f = _mm256_cvtepi32_ps(a);
        const __m256 b_f = _mm256_cvtepi32_ps(b);

        // Calculate the reciprocal
        const __m256 b_rcp = _mm256_rcp_ps(b_f);

        // Calculate the inverse
        // Compensate for the loss
        // Perform the division by multiplication
        const __m256 f = _mm256_mul_ps(a_f, _mm256_mul_ps(b_rcp, _mm256_fnmadd_ps(b_rcp, b_f, two)));

        // Convert back to integers
        return _mm256_cvttps_epi32(f);
    }
}
