#ifndef __POOL_EPI8_H
#define __POOL_EPI8_H

#include "lattice.h"
#include "vec.h"
#include "UidHashTable.h"
#if defined(HAVE_CUDA)
#include "bgj_cuda.h"
#endif

#include "../dep/g6k/parallel_algorithms.hpp"
#include "../dep/g6k/thread_pool.hpp"

#include <omp.h>
#include <immintrin.h>



#define AVX2_EDP_4x4(__u, __S0, __S1, __S2, __S3, __len, __d0, __d1, __d2, __d3)\
                                                              do {            \
    __m256i __acc00 = _mm256_setzero_si256();                                 \
    __m256i __acc01 = _mm256_setzero_si256();                                 \
    __m256i __acc02 = _mm256_setzero_si256();                                 \
    __m256i __acc03 = _mm256_setzero_si256();                                 \
    __m256i __acc10 = _mm256_setzero_si256();                                 \
    __m256i __acc11 = _mm256_setzero_si256();                                 \
    __m256i __acc12 = _mm256_setzero_si256();                                 \
    __m256i __acc13 = _mm256_setzero_si256();                                 \
    __m256i __acc20 = _mm256_setzero_si256();                                 \
    __m256i __acc21 = _mm256_setzero_si256();                                 \
    __m256i __acc22 = _mm256_setzero_si256();                                 \
    __m256i __acc23 = _mm256_setzero_si256();                                 \
    __m256i __acc30 = _mm256_setzero_si256();                                 \
    __m256i __acc31 = _mm256_setzero_si256();                                 \
    __m256i __acc32 = _mm256_setzero_si256();                                 \
    __m256i __acc33 = _mm256_setzero_si256();                                 \
    for (long __i = 0; __i < __len; __i += 32) {                              \
        __m256i __u0 = _mm256_load_si256((__m256i *)(__u+__i+0*__len));       \
        __m256i __u1 = _mm256_load_si256((__m256i *)(__u+__i+1*__len));       \
        __m256i __u2 = _mm256_load_si256((__m256i *)(__u+__i+2*__len));       \
        __m256i __u3 = _mm256_load_si256((__m256i *)(__u+__i+3*__len));       \
        __m256i __s0 = _mm256_load_si256((__m256i *)(__S0+__i));              \
        __m256i __s1 = _mm256_load_si256((__m256i *)(__S1+__i));              \
        __m256i __s2 = _mm256_load_si256((__m256i *)(__S2+__i));              \
        __m256i __s3 = _mm256_load_si256((__m256i *)(__S3+__i));              \
        __acc00 = _mm256_dpbusd_epi32(__acc00, __u0, __s0);                   \
        __acc01 = _mm256_dpbusd_epi32(__acc01, __u0, __s1);                   \
        __acc02 = _mm256_dpbusd_epi32(__acc02, __u0, __s2);                   \
        __acc03 = _mm256_dpbusd_epi32(__acc03, __u0, __s3);                   \
        __acc10 = _mm256_dpbusd_epi32(__acc10, __u1, __s0);                   \
        __acc11 = _mm256_dpbusd_epi32(__acc11, __u1, __s1);                   \
        __acc12 = _mm256_dpbusd_epi32(__acc12, __u1, __s2);                   \
        __acc13 = _mm256_dpbusd_epi32(__acc13, __u1, __s3);                   \
        __acc20 = _mm256_dpbusd_epi32(__acc20, __u2, __s0);                   \
        __acc21 = _mm256_dpbusd_epi32(__acc21, __u2, __s1);                   \
        __acc22 = _mm256_dpbusd_epi32(__acc22, __u2, __s2);                   \
        __acc23 = _mm256_dpbusd_epi32(__acc23, __u2, __s3);                   \
        __acc30 = _mm256_dpbusd_epi32(__acc30, __u3, __s0);                   \
        __acc31 = _mm256_dpbusd_epi32(__acc31, __u3, __s1);                   \
        __acc32 = _mm256_dpbusd_epi32(__acc32, __u3, __s2);                   \
        __acc33 = _mm256_dpbusd_epi32(__acc33, __u3, __s3);                   \
    }                                                                         \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc01);                            \
    __acc02 = _mm256_hadd_epi32(__acc02, __acc03);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc11);                            \
    __acc12 = _mm256_hadd_epi32(__acc12, __acc13);                            \
    __acc20 = _mm256_hadd_epi32(__acc20, __acc21);                            \
    __acc22 = _mm256_hadd_epi32(__acc22, __acc23);                            \
    __acc30 = _mm256_hadd_epi32(__acc30, __acc31);                            \
    __acc32 = _mm256_hadd_epi32(__acc32, __acc33);                            \
    __d0 = _mm256_hadd_epi32(__acc00, __acc02);                               \
    __d1 = _mm256_hadd_epi32(__acc10, __acc12);                               \
    __d2 = _mm256_hadd_epi32(__acc20, __acc22);                               \
    __d3 = _mm256_hadd_epi32(__acc30, __acc32);                               \
} while(0)

#define AVX2_EDP_4x4V(__u, __S0, __S1, __S2, __S3, __len, __d0, __d1, __d2, __d3)\
                                                              do {            \
    __m256i __acc00 = _mm256_setzero_si256();                                 \
    __m256i __acc01 = _mm256_setzero_si256();                                 \
    __m256i __acc02 = _mm256_setzero_si256();                                 \
    __m256i __acc03 = _mm256_setzero_si256();                                 \
    __m256i __acc10 = _mm256_setzero_si256();                                 \
    __m256i __acc11 = _mm256_setzero_si256();                                 \
    __m256i __acc12 = _mm256_setzero_si256();                                 \
    __m256i __acc13 = _mm256_setzero_si256();                                 \
    __m256i __acc20 = _mm256_setzero_si256();                                 \
    __m256i __acc21 = _mm256_setzero_si256();                                 \
    __m256i __acc22 = _mm256_setzero_si256();                                 \
    __m256i __acc23 = _mm256_setzero_si256();                                 \
    __m256i __acc30 = _mm256_setzero_si256();                                 \
    __m256i __acc31 = _mm256_setzero_si256();                                 \
    __m256i __acc32 = _mm256_setzero_si256();                                 \
    __m256i __acc33 = _mm256_setzero_si256();                                 \
    for (long __i = 0; __i < __len; __i += 32) {                              \
        __m256i __u0 = _mm256_load_si256((__m256i *)(__u+__i+0*__len));       \
        __m256i __u1 = _mm256_load_si256((__m256i *)(__u+__i+1*__len));       \
        __m256i __u2 = _mm256_load_si256((__m256i *)(__u+__i+2*__len));       \
        __m256i __u3 = _mm256_load_si256((__m256i *)(__u+__i+3*__len));       \
        __m256i __s0 = _mm256_load_si256((__m256i *)(__S0+__i));              \
        __m256i __s1 = _mm256_load_si256((__m256i *)(__S1+__i));              \
        __m256i __s2 = _mm256_load_si256((__m256i *)(__S2+__i));              \
        __m256i __s3 = _mm256_load_si256((__m256i *)(__S3+__i));              \
        __acc00 = _mm256_dpbusd_epi32(__acc00, __u0, __s0);                   \
        __acc01 = _mm256_dpbusd_epi32(__acc01, __u1, __s0);                   \
        __acc02 = _mm256_dpbusd_epi32(__acc02, __u2, __s0);                   \
        __acc03 = _mm256_dpbusd_epi32(__acc03, __u3, __s0);                   \
        __acc10 = _mm256_dpbusd_epi32(__acc10, __u0, __s1);                   \
        __acc11 = _mm256_dpbusd_epi32(__acc11, __u1, __s1);                   \
        __acc12 = _mm256_dpbusd_epi32(__acc12, __u2, __s1);                   \
        __acc13 = _mm256_dpbusd_epi32(__acc13, __u3, __s1);                   \
        __acc20 = _mm256_dpbusd_epi32(__acc20, __u0, __s2);                   \
        __acc21 = _mm256_dpbusd_epi32(__acc21, __u1, __s2);                   \
        __acc22 = _mm256_dpbusd_epi32(__acc22, __u2, __s2);                   \
        __acc23 = _mm256_dpbusd_epi32(__acc23, __u3, __s2);                   \
        __acc30 = _mm256_dpbusd_epi32(__acc30, __u0, __s3);                   \
        __acc31 = _mm256_dpbusd_epi32(__acc31, __u1, __s3);                   \
        __acc32 = _mm256_dpbusd_epi32(__acc32, __u2, __s3);                   \
        __acc33 = _mm256_dpbusd_epi32(__acc33, __u3, __s3);                   \
    }                                                                         \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc01);                            \
    __acc02 = _mm256_hadd_epi32(__acc02, __acc03);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc11);                            \
    __acc12 = _mm256_hadd_epi32(__acc12, __acc13);                            \
    __acc20 = _mm256_hadd_epi32(__acc20, __acc21);                            \
    __acc22 = _mm256_hadd_epi32(__acc22, __acc23);                            \
    __acc30 = _mm256_hadd_epi32(__acc30, __acc31);                            \
    __acc32 = _mm256_hadd_epi32(__acc32, __acc33);                            \
    __d0 = _mm256_hadd_epi32(__acc00, __acc02);                               \
    __d1 = _mm256_hadd_epi32(__acc10, __acc12);                               \
    __d2 = _mm256_hadd_epi32(__acc20, __acc22);                               \
    __d3 = _mm256_hadd_epi32(__acc30, __acc32);                               \
} while(0)

#define AVX2_EDP_3x4(__u, __S0, __S1, __S2, __S3, __len, __d0, __d1, __d2)    \
                                                        do {                  \
    __m256i __acc00 = _mm256_setzero_si256();                                 \
    __m256i __acc01 = _mm256_setzero_si256();                                 \
    __m256i __acc02 = _mm256_setzero_si256();                                 \
    __m256i __acc03 = _mm256_setzero_si256();                                 \
    __m256i __acc10 = _mm256_setzero_si256();                                 \
    __m256i __acc11 = _mm256_setzero_si256();                                 \
    __m256i __acc12 = _mm256_setzero_si256();                                 \
    __m256i __acc13 = _mm256_setzero_si256();                                 \
    __m256i __acc20 = _mm256_setzero_si256();                                 \
    __m256i __acc21 = _mm256_setzero_si256();                                 \
    __m256i __acc22 = _mm256_setzero_si256();                                 \
    __m256i __acc23 = _mm256_setzero_si256();                                 \
    for (long __i = 0; __i < __len; __i += 32) {                              \
        __m256i __u0 = _mm256_load_si256((__m256i *)(__u+__i+0*__len));       \
        __m256i __u1 = _mm256_load_si256((__m256i *)(__u+__i+1*__len));       \
        __m256i __u2 = _mm256_load_si256((__m256i *)(__u+__i+2*__len));       \
        __m256i __s0 = _mm256_load_si256((__m256i *)(__S0+__i));              \
        __m256i __s1 = _mm256_load_si256((__m256i *)(__S1+__i));              \
        __m256i __s2 = _mm256_load_si256((__m256i *)(__S2+__i));              \
        __m256i __s3 = _mm256_load_si256((__m256i *)(__S3+__i));              \
        __acc00 = _mm256_dpbusd_epi32(__acc00, __u0, __s0);                   \
        __acc01 = _mm256_dpbusd_epi32(__acc01, __u0, __s1);                   \
        __acc02 = _mm256_dpbusd_epi32(__acc02, __u0, __s2);                   \
        __acc03 = _mm256_dpbusd_epi32(__acc03, __u0, __s3);                   \
        __acc10 = _mm256_dpbusd_epi32(__acc10, __u1, __s0);                   \
        __acc11 = _mm256_dpbusd_epi32(__acc11, __u1, __s1);                   \
        __acc12 = _mm256_dpbusd_epi32(__acc12, __u1, __s2);                   \
        __acc13 = _mm256_dpbusd_epi32(__acc13, __u1, __s3);                   \
        __acc20 = _mm256_dpbusd_epi32(__acc20, __u2, __s0);                   \
        __acc21 = _mm256_dpbusd_epi32(__acc21, __u2, __s1);                   \
        __acc22 = _mm256_dpbusd_epi32(__acc22, __u2, __s2);                   \
        __acc23 = _mm256_dpbusd_epi32(__acc23, __u2, __s3);                   \
    }                                                                         \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc01);                            \
    __acc02 = _mm256_hadd_epi32(__acc02, __acc03);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc11);                            \
    __acc12 = _mm256_hadd_epi32(__acc12, __acc13);                            \
    __acc20 = _mm256_hadd_epi32(__acc20, __acc21);                            \
    __acc22 = _mm256_hadd_epi32(__acc22, __acc23);                            \
    __d0 = _mm256_hadd_epi32(__acc00, __acc02);                               \
    __d1 = _mm256_hadd_epi32(__acc10, __acc12);                               \
    __d2 = _mm256_hadd_epi32(__acc20, __acc22);                               \
} while (0)

#define AVX2_EDP_3x4V(__u, __S0, __S1, __S2, __len, __d0, __d1, __d2)         \
                                                        do {                  \
    __m256i __acc00 = _mm256_setzero_si256();                                 \
    __m256i __acc01 = _mm256_setzero_si256();                                 \
    __m256i __acc02 = _mm256_setzero_si256();                                 \
    __m256i __acc03 = _mm256_setzero_si256();                                 \
    __m256i __acc10 = _mm256_setzero_si256();                                 \
    __m256i __acc11 = _mm256_setzero_si256();                                 \
    __m256i __acc12 = _mm256_setzero_si256();                                 \
    __m256i __acc13 = _mm256_setzero_si256();                                 \
    __m256i __acc20 = _mm256_setzero_si256();                                 \
    __m256i __acc21 = _mm256_setzero_si256();                                 \
    __m256i __acc22 = _mm256_setzero_si256();                                 \
    __m256i __acc23 = _mm256_setzero_si256();                                 \
    for (long __i = 0; __i < __len; __i += 32) {                              \
        __m256i __s0 = _mm256_load_si256((__m256i *)(__S0+__i));              \
        __m256i __s1 = _mm256_load_si256((__m256i *)(__S1+__i));              \
        __m256i __s2 = _mm256_load_si256((__m256i *)(__S2+__i));              \
        __m256i __u0 = _mm256_load_si256((__m256i *)(__u+__i+0*__len));       \
        __m256i __u1 = _mm256_load_si256((__m256i *)(__u+__i+1*__len));       \
        __m256i __u2 = _mm256_load_si256((__m256i *)(__u+__i+2*__len));       \
        __m256i __u3 = _mm256_load_si256((__m256i *)(__u+__i+3*__len));       \
        __acc00 = _mm256_dpbusd_epi32(__acc00, __u0, __s0);                   \
        __acc01 = _mm256_dpbusd_epi32(__acc01, __u1, __s0);                   \
        __acc02 = _mm256_dpbusd_epi32(__acc02, __u2, __s0);                   \
        __acc03 = _mm256_dpbusd_epi32(__acc03, __u3, __s0);                   \
        __acc10 = _mm256_dpbusd_epi32(__acc10, __u0, __s1);                   \
        __acc11 = _mm256_dpbusd_epi32(__acc11, __u1, __s1);                   \
        __acc12 = _mm256_dpbusd_epi32(__acc12, __u2, __s1);                   \
        __acc13 = _mm256_dpbusd_epi32(__acc13, __u3, __s1);                   \
        __acc20 = _mm256_dpbusd_epi32(__acc20, __u0, __s2);                   \
        __acc21 = _mm256_dpbusd_epi32(__acc21, __u1, __s2);                   \
        __acc22 = _mm256_dpbusd_epi32(__acc22, __u2, __s2);                   \
        __acc23 = _mm256_dpbusd_epi32(__acc23, __u3, __s2);                   \
    }                                                                         \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc01);                            \
    __acc02 = _mm256_hadd_epi32(__acc02, __acc03);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc11);                            \
    __acc12 = _mm256_hadd_epi32(__acc12, __acc13);                            \
    __acc20 = _mm256_hadd_epi32(__acc20, __acc21);                            \
    __acc22 = _mm256_hadd_epi32(__acc22, __acc23);                            \
    __d0 = _mm256_hadd_epi32(__acc00, __acc02);                               \
    __d1 = _mm256_hadd_epi32(__acc10, __acc12);                               \
    __d2 = _mm256_hadd_epi32(__acc20, __acc22);                               \
} while (0)

#define AVX2_EDP_2x8(__u, __S0, __S1, __S2, __S3, __S4, __S5, __S6, __S7, __len, __d0, __d1)\
                                                  do {                        \
    __m256i __acc00 = _mm256_setzero_si256();                                 \
    __m256i __acc01 = _mm256_setzero_si256();                                 \
    __m256i __acc02 = _mm256_setzero_si256();                                 \
    __m256i __acc03 = _mm256_setzero_si256();                                 \
    __m256i __acc04 = _mm256_setzero_si256();                                 \
    __m256i __acc05 = _mm256_setzero_si256();                                 \
    __m256i __acc06 = _mm256_setzero_si256();                                 \
    __m256i __acc07 = _mm256_setzero_si256();                                 \
    __m256i __acc10 = _mm256_setzero_si256();                                 \
    __m256i __acc11 = _mm256_setzero_si256();                                 \
    __m256i __acc12 = _mm256_setzero_si256();                                 \
    __m256i __acc13 = _mm256_setzero_si256();                                 \
    __m256i __acc14 = _mm256_setzero_si256();                                 \
    __m256i __acc15 = _mm256_setzero_si256();                                 \
    __m256i __acc16 = _mm256_setzero_si256();                                 \
    __m256i __acc17 = _mm256_setzero_si256();                                 \
    for (long __i = 0; __i < __len; __i += 32) {                              \
        __m256i __u0 = _mm256_load_si256((__m256i *)(__u+__i+0*__len));       \
        __m256i __u1 = _mm256_load_si256((__m256i *)(__u+__i+1*__len));       \
        __m256i __s0 = _mm256_load_si256((__m256i *)(__S0+__i));              \
        __m256i __s1 = _mm256_load_si256((__m256i *)(__S1+__i));              \
        __m256i __s2 = _mm256_load_si256((__m256i *)(__S2+__i));              \
        __m256i __s3 = _mm256_load_si256((__m256i *)(__S3+__i));              \
        __m256i __s4 = _mm256_load_si256((__m256i *)(__S4+__i));              \
        __m256i __s5 = _mm256_load_si256((__m256i *)(__S5+__i));              \
        __m256i __s6 = _mm256_load_si256((__m256i *)(__S6+__i));              \
        __m256i __s7 = _mm256_load_si256((__m256i *)(__S7+__i));              \
        __acc00 = _mm256_dpbusd_epi32(__acc00, __u0, __s0);                   \
        __acc01 = _mm256_dpbusd_epi32(__acc01, __u0, __s1);                   \
        __acc02 = _mm256_dpbusd_epi32(__acc02, __u0, __s2);                   \
        __acc03 = _mm256_dpbusd_epi32(__acc03, __u0, __s3);                   \
        __acc04 = _mm256_dpbusd_epi32(__acc04, __u0, __s4);                   \
        __acc05 = _mm256_dpbusd_epi32(__acc05, __u0, __s5);                   \
        __acc06 = _mm256_dpbusd_epi32(__acc06, __u0, __s6);                   \
        __acc07 = _mm256_dpbusd_epi32(__acc07, __u0, __s7);                   \
        __acc10 = _mm256_dpbusd_epi32(__acc10, __u1, __s0);                   \
        __acc11 = _mm256_dpbusd_epi32(__acc11, __u1, __s1);                   \
        __acc12 = _mm256_dpbusd_epi32(__acc12, __u1, __s2);                   \
        __acc13 = _mm256_dpbusd_epi32(__acc13, __u1, __s3);                   \
        __acc14 = _mm256_dpbusd_epi32(__acc14, __u1, __s4);                   \
        __acc15 = _mm256_dpbusd_epi32(__acc15, __u1, __s5);                   \
        __acc16 = _mm256_dpbusd_epi32(__acc16, __u1, __s6);                   \
        __acc17 = _mm256_dpbusd_epi32(__acc17, __u1, __s7);                   \
    }                                                                         \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc01);                            \
    __acc02 = _mm256_hadd_epi32(__acc02, __acc03);                            \
    __acc04 = _mm256_hadd_epi32(__acc04, __acc05);                            \
    __acc06 = _mm256_hadd_epi32(__acc06, __acc07);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc11);                            \
    __acc12 = _mm256_hadd_epi32(__acc12, __acc13);                            \
    __acc14 = _mm256_hadd_epi32(__acc14, __acc15);                            \
    __acc16 = _mm256_hadd_epi32(__acc16, __acc17);                            \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc02);                            \
    __acc04 = _mm256_hadd_epi32(__acc04, __acc06);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc12);                            \
    __acc14 = _mm256_hadd_epi32(__acc14, __acc16);                            \
    __m256i __r0lo = _mm256_permute2f128_ps(__acc00, __acc04, 48);            \
    __m256i __r0hi = _mm256_permute2f128_ps(__acc00, __acc04, 33);            \
    __m256i __r1lo = _mm256_permute2f128_ps(__acc10, __acc14, 48);            \
    __m256i __r1hi = _mm256_permute2f128_ps(__acc10, __acc14, 33);            \
    __d0 = _mm256_add_epi32(__r0lo, __r0hi);                                  \
    __d1 = _mm256_add_epi32(__r1lo, __r1hi);                                  \
} while (0)

#define AVX2_EDP_2x8V(__u, __S0, __S1, __len, __d0, __d1)                     \
                                                  do {                        \
    __m256i __acc00 = _mm256_setzero_si256();                                 \
    __m256i __acc01 = _mm256_setzero_si256();                                 \
    __m256i __acc02 = _mm256_setzero_si256();                                 \
    __m256i __acc03 = _mm256_setzero_si256();                                 \
    __m256i __acc04 = _mm256_setzero_si256();                                 \
    __m256i __acc05 = _mm256_setzero_si256();                                 \
    __m256i __acc06 = _mm256_setzero_si256();                                 \
    __m256i __acc07 = _mm256_setzero_si256();                                 \
    __m256i __acc10 = _mm256_setzero_si256();                                 \
    __m256i __acc11 = _mm256_setzero_si256();                                 \
    __m256i __acc12 = _mm256_setzero_si256();                                 \
    __m256i __acc13 = _mm256_setzero_si256();                                 \
    __m256i __acc14 = _mm256_setzero_si256();                                 \
    __m256i __acc15 = _mm256_setzero_si256();                                 \
    __m256i __acc16 = _mm256_setzero_si256();                                 \
    __m256i __acc17 = _mm256_setzero_si256();                                 \
    for (long __i = 0; __i < __len; __i += 32) {                              \
        __m256i __s0 = _mm256_load_si256((__m256i *)(__S0+__i));              \
        __m256i __s1 = _mm256_load_si256((__m256i *)(__S1+__i));              \
        __m256i __u0 = _mm256_load_si256((__m256i *)(__u+__i+0*__len));       \
        __m256i __u1 = _mm256_load_si256((__m256i *)(__u+__i+1*__len));       \
        __m256i __u2 = _mm256_load_si256((__m256i *)(__u+__i+2*__len));       \
        __m256i __u3 = _mm256_load_si256((__m256i *)(__u+__i+3*__len));       \
        __m256i __u4 = _mm256_load_si256((__m256i *)(__u+__i+4*__len));       \
        __m256i __u5 = _mm256_load_si256((__m256i *)(__u+__i+5*__len));       \
        __m256i __u6 = _mm256_load_si256((__m256i *)(__u+__i+6*__len));       \
        __m256i __u7 = _mm256_load_si256((__m256i *)(__u+__i+7*__len));       \
        __acc00 = _mm256_dpbusd_epi32(__acc00, __u0, __s0);                   \
        __acc01 = _mm256_dpbusd_epi32(__acc01, __u1, __s0);                   \
        __acc02 = _mm256_dpbusd_epi32(__acc02, __u2, __s0);                   \
        __acc03 = _mm256_dpbusd_epi32(__acc03, __u3, __s0);                   \
        __acc04 = _mm256_dpbusd_epi32(__acc04, __u4, __s0);                   \
        __acc05 = _mm256_dpbusd_epi32(__acc05, __u5, __s0);                   \
        __acc06 = _mm256_dpbusd_epi32(__acc06, __u6, __s0);                   \
        __acc07 = _mm256_dpbusd_epi32(__acc07, __u7, __s0);                   \
        __acc10 = _mm256_dpbusd_epi32(__acc10, __u0, __s1);                   \
        __acc11 = _mm256_dpbusd_epi32(__acc11, __u1, __s1);                   \
        __acc12 = _mm256_dpbusd_epi32(__acc12, __u2, __s1);                   \
        __acc13 = _mm256_dpbusd_epi32(__acc13, __u3, __s1);                   \
        __acc14 = _mm256_dpbusd_epi32(__acc14, __u4, __s1);                   \
        __acc15 = _mm256_dpbusd_epi32(__acc15, __u5, __s1);                   \
        __acc16 = _mm256_dpbusd_epi32(__acc16, __u6, __s1);                   \
        __acc17 = _mm256_dpbusd_epi32(__acc17, __u7, __s1);                   \
    }                                                                         \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc01);                            \
    __acc02 = _mm256_hadd_epi32(__acc02, __acc03);                            \
    __acc04 = _mm256_hadd_epi32(__acc04, __acc05);                            \
    __acc06 = _mm256_hadd_epi32(__acc06, __acc07);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc11);                            \
    __acc12 = _mm256_hadd_epi32(__acc12, __acc13);                            \
    __acc14 = _mm256_hadd_epi32(__acc14, __acc15);                            \
    __acc16 = _mm256_hadd_epi32(__acc16, __acc17);                            \
    __acc00 = _mm256_hadd_epi32(__acc00, __acc02);                            \
    __acc04 = _mm256_hadd_epi32(__acc04, __acc06);                            \
    __acc10 = _mm256_hadd_epi32(__acc10, __acc12);                            \
    __acc14 = _mm256_hadd_epi32(__acc14, __acc16);                            \
    __m256i __r0lo = _mm256_permute2f128_ps(__acc00, __acc04, 48);            \
    __m256i __r0hi = _mm256_permute2f128_ps(__acc00, __acc04, 33);            \
    __m256i __r1lo = _mm256_permute2f128_ps(__acc10, __acc14, 48);            \
    __m256i __r1hi = _mm256_permute2f128_ps(__acc10, __acc14, 33);            \
    __d0 = _mm256_add_epi32(__r0lo, __r0hi);                                  \
    __d1 = _mm256_add_epi32(__r1lo, __r1hi);                                  \
} while (0)


template <bool record_dp>
struct bucket_epi8_t;

template <uint32_t nb>
struct bgj_profile_data_t;

struct sol_list_epi8_t;

#if defined(__AMX_INT8__)
struct bucket_amx_t;

template <uint32_t nb>
struct bgj_amx_profile_data_t;

struct sol_list_amx_t;
#endif

struct nsh_mblock_t;
struct lsh_mblock_t;
struct nsh_profile_data_t;
struct lsh_profile_data_t;

void bgj_lsh_best_solution_reset();
int bgj_lsh_best_solution_get(double *length, double *vec, long capacity, long *dimension);
int bgj_lsh_best_solution_record(double length, const double *vec, long dimension);

#if defined(__AMX_INT8__) && BOOST_AMX_SIEVE
struct booster_amx160_t;
#endif

template<uint32_t nb>
struct Pool_epi8_t {
    // basis
        Lattice_QP *basis;
        float **_b_local = NULL;        // we don't use reversed index order here
        uint8_t *_b_dual = NULL;
        float _ratio;
        int32_t _dhalf;
        int32_t _dshift;
        
    // sieving Status
        long CSD;       //current sieving dimension
        long index_l;   //current sieving context = [index_l, index_r]
        long index_r;
        float gh2;      //gh^2 of L_{[index_l, index_r]}
        static constexpr long vec_length = nb * 32;
        long num_threads = 1;

    // uid
        UidHashTable *uid = NULL;

    // amx booster
        #if defined(__AMX_INT8__) && BOOST_AMX_SIEVE
        booster_amx160_t *booster = NULL;
        #endif
    
    // data
        long num_empty = 0;
        long num_vec = 0;
        uint64_t pool_epoch = 1;
        int8_t *vec = NULL;
        uint16_t *cvec = NULL;      // cvec_size = 3
        int32_t *vnorm = NULL;      // we store round(1/2 * ||v||^2) here
        int32_t *vsum = NULL;       // we store 128 * sum(round(v)) here
        uint64_t *vu = NULL;
        int last_lift_valid = 0;
        double last_lift_euclidean_norm = 0.0;
        double last_lift_lift_norm = 0.0;
        double last_lift_gh = 0.0;
        double last_lift_approx_factor = 0.0;

    // construction and distructions
        Pool_epi8_t();
        Pool_epi8_t(Lattice_QP *L);
        ~Pool_epi8_t();
        void clear_all();
        void clear_pool();

    // setup
        int set_num_threads(long n);
        int set_max_pool_size(long N);
        int set_basis(Lattice_QP *L);
        int set_sieving_context(long l, long r);
        
    
    // pool operations
        // do gaussian sampling and size reduction to collect N vectors in the pool
        int sampling(long N);
        // shrink the pool size to N
        int shrink(long N);
        // extend_left
        int extend_left();
        // shrink left
        int shrink_left();
        // sort the cvec list
        int sort_cvec();
        // check whether sieve is over
        int sieve_is_over(double saturation_radius, double saturation_ratio);
        // do insertion
        int insert(long index, double eta);
        // show the minimal lift to index
        int show_min_lift(long index);
        // do LLL in the last n pool indices, maintain the pool.
        int tail_LLL(double delta, long n);
        // store the pool to file
        int store(const char *file_name);
        // load the pool from file
        int load(const char *file_name);
        // return the pot of the lattice
        double pot();
        // check pool status
        bool check_pool_status(long q = 0, int supress_minor = 1);
        // check lose of dimension
        bool check_dim_lose(long q = 0);
        inline void mark_pool_dirty() {
            pool_epoch++;
            if (pool_epoch == 0) pool_epoch = 1;
        }

    // Sieving
        // bgj sieve
        int bgj1_Sieve(long log_level = 0, long lps_auto_adj = 1);
        #if defined(HAVE_CUDA)
        int bgj1_Sieve_cuda(long log_level = 0, long lps_auto_adj = 1);
        #endif
        int bgj2_Sieve(long log_level = 0, long lps_auto_adj = 1);
        #if defined(HAVE_CUDA)
        int bgj2_Sieve_cuda(long log_level = 0, long lps_auto_adj = 1);
        #endif
        int bgj3_Sieve(long log_level = 0, long lps_auto_adj = 1);
        #if defined(HAVE_CUDA)
        int bgj3_Sieve_cuda(long log_level = 0, long lps_auto_adj = 1);
        #endif
        #if defined(__AMX_INT8__)
        int bgj3_Sieve_amx(long log_level = 0, long max_epoch = -1, double goal_norm_scale = -1.0, long ESD = -1, double prefer_deep = -1.0);
        int bgj_amx_upsieve(long log_level = 0, long max_epoch = -1, double goal_norm_scale = -1.0, long ESD = -1, double prefer_deep = -1.0);
        int bgj_amx_downsieve(long log_level = 0, long max_epoch = -1, double goal_norm_scale = -1.0, long ESD = -1, double prefer_deep = -1.0);
        #endif
        // left progressive_sieve on L_{[ind_l, ind_r]}
        #if defined(__AMX_INT8__)
        int left_progressive_amx(long ind_l, long ind_r, long num_threads, long log_level, long ssd = 45);
        #endif
        int left_progressive_bgjfsieve(long ind_l, long ind_r, long num_threads, long log_level, long ssd = 45);
        int left_progressive_bgj1sieve(long ind_l, long ind_r, long num_threads, long log_level);
        #if defined(HAVE_CUDA)
        int left_progressive_bgjfsieve_cuda(long ind_l, long ind_r, long num_threads, long log_level, long ssd = 45);
        int left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
        #endif
        int left_progressive_bgj2sieve(long ind_l, long ind_r, long num_threads, long log_level);
        int left_progressive_bgj3sieve(long ind_l, long ind_r, long num_threads, long log_level);
        
        // the search process will stop when a vector of length stop_ratio * expect_length is found
        int naivedh_insert(long target_index, double eta, long log_level, double target_length = 0.0);
        int lsfdh_insert(long target_index, double eta, long log_level, double target_length = 0.0, double stop_ratio = 0.0);
        int naivesh_insert(long target_index, double eta, long log_level, double target_length = 0.0);
        int lsfsh_insert(long target_index, double eta, long log_level, double target_length = 0.0, double stop_ratio = 0.0, double qratio = 0.0);
        int show_lsfsh_insert(long target_index, double eta, long log_level, double target_length = 0.0, double stop_ratio = 0.0, double qratio = 0.0);

        friend struct bgj_profile_data_t<nb>;
        #if defined(__AMX_INT8__)
        friend struct bgj_amx_profile_data_t<nb>;
        friend struct booster_amx160_t;
        #endif
        inline int compute_coeff(int32_t *dst, uint32_t ind) {return _compute_coeff(dst, ind);}
        inline int compute_coeff_b8(int32_t *dst, uint32_t ind) {return _compute_coeff_b8(dst, ind);}
    private:
        long sorted_index = 0;
        long _pool_size = 0;
        thread_pool::thread_pool _threadpool;

        inline int32_t vdpss(uint32_t ind, uint32_t jnd) {
            __m256i acc = _mm256_setzero_si256();
            for (long i = 0; i < vec_length; i += 32) {
                __m256i uivec = _mm256_xor_si256(epi8_sign_bit, _mm256_load_si256((__m256i *)(vec + ind * vec_length + i)));
                acc = _mm256_dpbusd_epi32(acc, uivec, _mm256_load_si256((__m256i *)(vec + jnd * vec_length + i)));
            }
            __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 78));
            acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 177));
            return _mm_cvtsi128_si32(acc128) - vsum[jnd];
        }
        inline int32_t vdp(uint8_t *u, int8_t *s) {
            __m256i acc = _mm256_setzero_si256();
            for (long i = 0; i < vec_length; i += 32) {
                acc = _mm256_dpbusd_epi32(acc, _mm256_load_si256((__m256i *)(u+i)), _mm256_load_si256((__m256i *)(s+i)));
            }
            __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 78));
            acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 177));
            return _mm_cvtsi128_si32(acc128);
        }
        inline __m256i vdp1x8(uint8_t *u, int8_t *s) {
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256();
            __m256i acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256();
            __m256i acc7 = _mm256_setzero_si256();
            for (long i = 0; i < vec_length; i += 32) {
                __m256i uu = _mm256_load_si256((__m256i *)(u+i));
                acc0 = _mm256_dpbusd_epi32(acc0, uu, _mm256_load_si256((__m256i *)(s+i+0*vec_length)));
                acc1 = _mm256_dpbusd_epi32(acc1, uu, _mm256_load_si256((__m256i *)(s+i+1*vec_length)));
                acc2 = _mm256_dpbusd_epi32(acc2, uu, _mm256_load_si256((__m256i *)(s+i+2*vec_length)));
                acc3 = _mm256_dpbusd_epi32(acc3, uu, _mm256_load_si256((__m256i *)(s+i+3*vec_length)));
                acc4 = _mm256_dpbusd_epi32(acc4, uu, _mm256_load_si256((__m256i *)(s+i+4*vec_length)));
                acc5 = _mm256_dpbusd_epi32(acc5, uu, _mm256_load_si256((__m256i *)(s+i+5*vec_length)));
                acc6 = _mm256_dpbusd_epi32(acc6, uu, _mm256_load_si256((__m256i *)(s+i+6*vec_length)));
                acc7 = _mm256_dpbusd_epi32(acc7, uu, _mm256_load_si256((__m256i *)(s+i+7*vec_length)));
            }
            acc0 = _mm256_hadd_epi32(acc0, acc1);
            acc2 = _mm256_hadd_epi32(acc2, acc3);
            acc4 = _mm256_hadd_epi32(acc4, acc5);
            acc6 = _mm256_hadd_epi32(acc6, acc7);
            acc0 = _mm256_hadd_epi32(acc0, acc2);
            acc4 = _mm256_hadd_epi32(acc4, acc6);
            __m256i acclo = _mm256_permute2f128_si256(acc0, acc4, 48);
            __m256i acchi = _mm256_permute2f128_si256(acc0, acc4, 33);
            return _mm256_add_epi32(acclo, acchi);
        }
        inline __m256i vdp8x1(int8_t *s, uint8_t *u) {
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256();
            __m256i acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256();
            __m256i acc7 = _mm256_setzero_si256();
            for (long i = 0; i < vec_length; i += 32) {
                __m256i ss = _mm256_load_si256((__m256i *)(s+i));
                acc0 = _mm256_dpbusd_epi32(acc0, _mm256_load_si256((__m256i *)(u+i+0*vec_length)), ss);
                acc1 = _mm256_dpbusd_epi32(acc1, _mm256_load_si256((__m256i *)(u+i+1*vec_length)), ss);
                acc2 = _mm256_dpbusd_epi32(acc2, _mm256_load_si256((__m256i *)(u+i+2*vec_length)), ss);
                acc3 = _mm256_dpbusd_epi32(acc3, _mm256_load_si256((__m256i *)(u+i+3*vec_length)), ss);
                acc4 = _mm256_dpbusd_epi32(acc4, _mm256_load_si256((__m256i *)(u+i+4*vec_length)), ss);
                acc5 = _mm256_dpbusd_epi32(acc5, _mm256_load_si256((__m256i *)(u+i+5*vec_length)), ss);
                acc6 = _mm256_dpbusd_epi32(acc6, _mm256_load_si256((__m256i *)(u+i+6*vec_length)), ss);
                acc7 = _mm256_dpbusd_epi32(acc7, _mm256_load_si256((__m256i *)(u+i+7*vec_length)), ss);
            }
            acc0 = _mm256_hadd_epi32(acc0, acc1);
            acc2 = _mm256_hadd_epi32(acc2, acc3);
            acc4 = _mm256_hadd_epi32(acc4, acc5);
            acc6 = _mm256_hadd_epi32(acc6, acc7);
            acc0 = _mm256_hadd_epi32(acc0, acc2);
            acc4 = _mm256_hadd_epi32(acc4, acc6);
            __m256i acclo = _mm256_permute2f128_si256(acc0, acc4, 48);
            __m256i acchi = _mm256_permute2f128_si256(acc0, acc4, 33);
            return _mm256_add_epi32(acclo, acchi);
        }
        inline void vdp8x8(__m256i *dst, uint8_t *u, int8_t *s) {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31;
            AVX2_EDP_4x4(u, s, (s+vec_length), (s+vec_length*2), (s+vec_length*3), vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4(u, (s+vec_length*4), (s+vec_length*5), (s+vec_length*6), (s+vec_length*7), vec_length, d01, d11, d21, d31);
            dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
            AVX2_EDP_4x4((u+vec_length*4), s, (s+vec_length), (s+vec_length*2), (s+vec_length*3), vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4((u+vec_length*4), (s+vec_length*4), (s+vec_length*5), (s+vec_length*6), (s+vec_length*7), vec_length, d01, d11, d21, d31);
            dst[4] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[5] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[6] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[7] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
        }
        inline void vdpnx8(__m256i *dst, uint8_t *u, int8_t *s, long n) {
            if (n >= 4) {
                __m256i d00, d01, d10, d11, d20, d21, d30, d31;
                AVX2_EDP_4x4(u, s, (s+vec_length), (s+vec_length*2), (s+vec_length*3), vec_length, d00, d10, d20, d30);
                AVX2_EDP_4x4(u, (s+vec_length*4), (s+vec_length*5), (s+vec_length*6), (s+vec_length*7), vec_length, d01, d11, d21, d31);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
                n -= 4;
                dst += 4;
                u += vec_length * 4;
            }
            if (n == 1) {
                dst[0] = vdp1x8(u, s);
                return;
            } else if (n == 2) {
                AVX2_EDP_2x8(u, s, (s+vec_length), (s+vec_length*2), (s+vec_length*3), (s+vec_length*4), (s+vec_length*5), (s+vec_length*6), (s+vec_length*7), vec_length, dst[0], dst[1]);
                return;
            } else if (n == 3) {
                __m256i d00, d01, d10, d11, d20, d21;
                AVX2_EDP_3x4(u, s, (s+vec_length), (s+vec_length*2), (s+vec_length*3), vec_length, d00, d10, d20);
                AVX2_EDP_3x4(u, (s+vec_length*4), (s+vec_length*5), (s+vec_length*6), (s+vec_length*7), vec_length, d01, d11, d21);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                return;
            }
        }
        inline void vdp8xn(__m256i *dst, int8_t *s, uint8_t *u, long n) {
            if (n >= 4) {
                __m256i d00, d01, d10, d11, d20, d21, d30, d31;
                AVX2_EDP_4x4V(u, s, (s+vec_length), (s+vec_length*2), (s+vec_length*3), vec_length, d00, d10, d20, d30);
                AVX2_EDP_4x4V((u+vec_length*4), s, (s+vec_length), (s+vec_length*2), (s+vec_length*3), vec_length, d01, d11, d21, d31);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
                n -= 4;
                dst += 4;
                s += vec_length * 4;
            }
            if (n == 1) {
                dst[0] = vdp8x1(s, u);
                return;
            } else if (n == 2) {
                AVX2_EDP_2x8V(u, s, (s+vec_length), vec_length, dst[0], dst[1]);
                return;
            } else if (n == 3) {
                __m256i d00, d01, d10, d11, d20, d21;
                AVX2_EDP_3x4V(u, s, (s+vec_length), (s+vec_length*2), vec_length, d00, d10, d20);
                AVX2_EDP_3x4V((u+vec_length*4), s, (s+vec_length), (s+vec_length*2), vec_length, d01, d11, d21);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                return;
            }
        }
        
        inline __m256i vdp1x8(uint8_t *u, uint32_t *s_ind) {
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256();
            __m256i acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256();
            __m256i acc7 = _mm256_setzero_si256();
            for (long i = 0; i < vec_length; i += 32) {
                __m256i uu = _mm256_load_si256((__m256i *)(u+i));
                acc0 = _mm256_dpbusd_epi32(acc0, uu, _mm256_load_si256((__m256i *)(vec+s_ind[0]*vec_length + i)));
                acc1 = _mm256_dpbusd_epi32(acc1, uu, _mm256_load_si256((__m256i *)(vec+s_ind[1]*vec_length + i)));
                acc2 = _mm256_dpbusd_epi32(acc2, uu, _mm256_load_si256((__m256i *)(vec+s_ind[2]*vec_length + i)));
                acc3 = _mm256_dpbusd_epi32(acc3, uu, _mm256_load_si256((__m256i *)(vec+s_ind[3]*vec_length + i)));
                acc4 = _mm256_dpbusd_epi32(acc4, uu, _mm256_load_si256((__m256i *)(vec+s_ind[4]*vec_length + i)));
                acc5 = _mm256_dpbusd_epi32(acc5, uu, _mm256_load_si256((__m256i *)(vec+s_ind[5]*vec_length + i)));
                acc6 = _mm256_dpbusd_epi32(acc6, uu, _mm256_load_si256((__m256i *)(vec+s_ind[6]*vec_length + i)));
                acc7 = _mm256_dpbusd_epi32(acc7, uu, _mm256_load_si256((__m256i *)(vec+s_ind[7]*vec_length + i)));
            }
            acc0 = _mm256_hadd_epi32(acc0, acc1);
            acc2 = _mm256_hadd_epi32(acc2, acc3);
            acc4 = _mm256_hadd_epi32(acc4, acc5);
            acc6 = _mm256_hadd_epi32(acc6, acc7);
            acc0 = _mm256_hadd_epi32(acc0, acc2);
            acc4 = _mm256_hadd_epi32(acc4, acc6);
            __m256i acclo = _mm256_permute2f128_si256(acc0, acc4, 48);
            __m256i acchi = _mm256_permute2f128_si256(acc0, acc4, 33);
            return _mm256_add_epi32(acclo, acchi);
        }
        inline void vdp8x8h(__m256i *dst, uint8_t *u, uint32_t *s_ind) {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31;
            int8_t *S0 = vec + s_ind[0] * vec_length;
            int8_t *S1 = vec + s_ind[1] * vec_length;
            int8_t *S2 = vec + s_ind[2] * vec_length;
            int8_t *S3 = vec + s_ind[3] * vec_length;
            int8_t *S4 = vec + s_ind[4] * vec_length;
            int8_t *S5 = vec + s_ind[5] * vec_length;
            int8_t *S6 = vec + s_ind[6] * vec_length;
            int8_t *S7 = vec + s_ind[7] * vec_length;
            AVX2_EDP_4x4(u, S0, S1, S2, S3, vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4(u, S4, S5, S6, S7, vec_length, d01, d11, d21, d31);
            dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
            AVX2_EDP_4x4((u+vec_length*4), S0, S1, S2, S3, vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4((u+vec_length*4), S4, S5, S6, S7, vec_length, d01, d11, d21, d31);
            dst[4] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[5] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[6] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[7] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
        }
        inline void vdp8x8(__m256i *dst, uint8_t *u, uint32_t *s_ind) {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31;
            int8_t *S0 = vec + s_ind[0] * vec_length;
            int8_t *S1 = vec + s_ind[1] * vec_length;
            int8_t *S2 = vec + s_ind[2] * vec_length;
            int8_t *S3 = vec + s_ind[3] * vec_length;
            int8_t *S4 = vec + s_ind[4] * vec_length;
            int8_t *S5 = vec + s_ind[5] * vec_length;
            int8_t *S6 = vec + s_ind[6] * vec_length;
            int8_t *S7 = vec + s_ind[7] * vec_length;
            AVX2_EDP_4x4V(u, S0, S1, S2, S3, vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4V((u+vec_length*4), S0, S1, S2, S3, vec_length, d01, d11, d21, d31);
            dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
            AVX2_EDP_4x4V(u, S4, S5, S6, S7, vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4V((u+vec_length*4), S4, S5, S6, S7, vec_length, d01, d11, d21, d31);
            dst[4] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[5] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[6] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[7] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
        }
        inline void vdp8x8(__m256i *dst, uint8_t *u, uint32_t *s_ind, int32_t *sum) {
            __m256i d00, d01, d10, d11, d20, d21, d30, d31;
            int8_t *S0 = vec + s_ind[0] * vec_length;
            int8_t *S1 = vec + s_ind[1] * vec_length;
            int8_t *S2 = vec + s_ind[2] * vec_length;
            int8_t *S3 = vec + s_ind[3] * vec_length;
            int8_t *S4 = vec + s_ind[4] * vec_length;
            int8_t *S5 = vec + s_ind[5] * vec_length;
            int8_t *S6 = vec + s_ind[6] * vec_length;
            int8_t *S7 = vec + s_ind[7] * vec_length;
            AVX2_EDP_4x4V(u, S0, S1, S2, S3, vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4V((u+vec_length*4), S0, S1, S2, S3, vec_length, d01, d11, d21, d31);
            dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
            dst[0] = _mm256_sub_epi32(dst[0], (__m256i) _mm256_broadcast_ss((float *)&sum[0]));
            dst[1] = _mm256_sub_epi32(dst[1], (__m256i) _mm256_broadcast_ss((float *)&sum[1]));
            dst[2] = _mm256_sub_epi32(dst[2], (__m256i) _mm256_broadcast_ss((float *)&sum[2]));
            dst[3] = _mm256_sub_epi32(dst[3], (__m256i) _mm256_broadcast_ss((float *)&sum[3]));
            AVX2_EDP_4x4V(u, S4, S5, S6, S7, vec_length, d00, d10, d20, d30);
            AVX2_EDP_4x4V((u+vec_length*4), S4, S5, S6, S7, vec_length, d01, d11, d21, d31);
            dst[4] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
            dst[5] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
            dst[6] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
            dst[7] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
            dst[4] = _mm256_sub_epi32(dst[4], (__m256i) _mm256_broadcast_ss((float *)&sum[4]));
            dst[5] = _mm256_sub_epi32(dst[5], (__m256i) _mm256_broadcast_ss((float *)&sum[5]));
            dst[6] = _mm256_sub_epi32(dst[6], (__m256i) _mm256_broadcast_ss((float *)&sum[6]));
            dst[7] = _mm256_sub_epi32(dst[7], (__m256i) _mm256_broadcast_ss((float *)&sum[7]));
        }
        inline void vdpnx8(__m256i *dst, uint8_t *u, uint32_t *s_ind, long n) {
            int8_t *S0 = vec + s_ind[0] * vec_length;
            int8_t *S1 = vec + s_ind[1] * vec_length;
            int8_t *S2 = vec + s_ind[2] * vec_length;
            int8_t *S3 = vec + s_ind[3] * vec_length;
            int8_t *S4 = vec + s_ind[4] * vec_length;
            int8_t *S5 = vec + s_ind[5] * vec_length;
            int8_t *S6 = vec + s_ind[6] * vec_length;
            int8_t *S7 = vec + s_ind[7] * vec_length;
            if (n >= 4) {
                __m256i d00, d01, d10, d11, d20, d21, d30, d31;
                AVX2_EDP_4x4(u, S0, S1, S2, S3, vec_length, d00, d10, d20, d30);
                AVX2_EDP_4x4(u, S4, S5, S6, S7, vec_length, d01, d11, d21, d31);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
                n -= 4;
                dst += 4;
                u += vec_length * 4;
            }
            if (n == 1) {
                dst[0] = vdp1x8(u, s_ind);
                return;
            } else if (n == 2) {
                AVX2_EDP_2x8(u, S0, S1, S2, S3, S4, S5, S6, S7, vec_length, dst[0], dst[1]);
                return;
            } else if (n == 3) {
                __m256i d00, d01, d10, d11, d20, d21;
                AVX2_EDP_3x4(u, S0, S1, S2, S3, vec_length, d00, d10, d20);
                AVX2_EDP_3x4(u, S4, S5, S6, S7, vec_length, d01, d11, d21);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                return;
            }
        }
        inline void vdp8xn(__m256i *dst, uint32_t *s_ind, uint8_t *u, long n){
            if (n >= 4) {
                __m256i d00, d01, d10, d11, d20, d21, d30, d31;
                int8_t *S0 = vec + s_ind[0] * vec_length;
                int8_t *S1 = vec + s_ind[1] * vec_length;
                int8_t *S2 = vec + s_ind[2] * vec_length;
                int8_t *S3 = vec + s_ind[3] * vec_length;
                AVX2_EDP_4x4V(u, S0, S1, S2, S3, vec_length, d00, d10, d20, d30);
                AVX2_EDP_4x4V((u+vec_length*4), S0, S1, S2, S3, vec_length, d01, d11, d21, d31);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                dst[3] = _mm256_add_epi32(_mm256_permute2f128_si256(d30, d31, 48), _mm256_permute2f128_si256(d30, d31, 33));
                n -= 4;
                dst += 4;
                s_ind += 4;
            }
            if (n == 1) {
                dst[0] = vdp8x1(vec+vec_length * s_ind[0], u);
                return;
            } else if (n == 2) {
                AVX2_EDP_2x8V(u, (vec + s_ind[0] * vec_length), (vec + s_ind[1] * vec_length), vec_length, dst[0], dst[1]);
                return;
            } else if (n == 3) {
                __m256i d00, d01, d10, d11, d20, d21;
                AVX2_EDP_3x4V(u, (vec + s_ind[0] * vec_length), (vec + s_ind[1] * vec_length), (vec + s_ind[2] * vec_length), vec_length, d00, d10, d20);
                AVX2_EDP_3x4V((u+vec_length*4), (vec + s_ind[0] * vec_length), (vec + s_ind[1] * vec_length), (vec + s_ind[2] * vec_length), vec_length, d01, d11, d21);
                dst[0] = _mm256_add_epi32(_mm256_permute2f128_si256(d00, d01, 48), _mm256_permute2f128_si256(d00, d01, 33));
                dst[1] = _mm256_add_epi32(_mm256_permute2f128_si256(d10, d11, 48), _mm256_permute2f128_si256(d10, d11, 33));
                dst[2] = _mm256_add_epi32(_mm256_permute2f128_si256(d20, d21, 48), _mm256_permute2f128_si256(d20, d21, 33));
                return;
            }
        }

        int _update_b_local(float ratio = 0.0f);
        float **_compute_b_local(long ind_l, long ind_r);
        int _compute_gh2();
        int _sampling(int8_t *dst, uint16_t *cdst, int32_t *dst_norm, int32_t *dst_sum, uint64_t *dst_u, DGS1d *R);
        void _reconstruct_all_cvec();
        
        // when calling this, vnorm, vsum, vec should be correct, b_local data should be chosen correspondingly
        inline int _compute_coeff(int32_t *dst, uint32_t ind, long csd, uint8_t *b_dual, int32_t dhalf, int32_t dshift) {
            long dnd = 0;
            __m256i x = _mm256_set1_epi32(-vsum[ind]+dhalf);
            while (dnd < csd - 7) {
                __m256i y = vdp8x1(vec + ind * vec_length, b_dual + dnd * vec_length);
                __m256i c = _mm256_srai_epi32(_mm256_add_epi32(y, x), dshift);
                _mm256_store_si256((__m256i *)(&dst[dnd]), c);
                dnd += 8;
            }
            while (dnd < csd) {
                int32_t y = vdp(b_dual+dnd*vec_length, vec+ind*vec_length);
                dst[dnd] = (y+dhalf-vsum[ind]) >> dshift;
                dnd++;
            }
            return 0;
        }
        inline int _compute_coeff(int32_t *dst, uint32_t ind) {
            return _compute_coeff(dst, ind, CSD, _b_dual, _dhalf, _dshift);
        }
        inline int _compute_coeff(int32_t *dst, int8_t *src, int32_t sum) {
            long dnd = 0;
            __m256i x = _mm256_set1_epi32(-sum+_dhalf);
            while (dnd < CSD - 7) {
                __m256i y = vdp8x1(src, _b_dual + dnd * vec_length);
                __m256i c = _mm256_srai_epi32(_mm256_add_epi32(y, x), _dshift);
                _mm256_store_si256((__m256i *)(&dst[dnd]), c);
                dnd += 8;
            }
            while (dnd < CSD) {
                int32_t y = vdp(_b_dual+dnd*vec_length, src);
                dst[dnd] = (y+_dhalf-sum) >> _dshift;
                dnd++;
            }
            return 0;
        }
        // make sure that ind < num_vec - 7, we do not check it in this function
        inline int _compute_coeff_b8(int32_t *dst, uint32_t ind, long csd, uint8_t *b_dual, int32_t dhalf, int32_t dshift) {
            long dnd = 0;
            __m256i ymm[8];
            __m256i x = _mm256_sub_epi32(_mm256_set1_epi32(dhalf), _mm256_loadu_si256((__m256i *)(vsum+ind)));
            while (dnd < csd - 7) {
                vdp8x8(ymm, b_dual+dnd*vec_length, vec+ind*vec_length);
                _mm256_store_si256((__m256i *)(&dst[dnd*8+0*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[0], x), dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+1*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[1], x), dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+2*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[2], x), dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+3*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[3], x), dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+4*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[4], x), dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+5*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[5], x), dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+6*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[6], x), dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+7*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[7], x), dshift));
                dnd += 8;
            }
            if (dnd < csd) {
                vdpnx8(ymm, b_dual+dnd*vec_length, vec+ind*vec_length, csd-dnd);
                for (long i = 0; i < csd - dnd; i++) {
                    _mm256_store_si256((__m256i *)(&dst[dnd*8+i*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[i], x), dshift));
                }
            }
            return 0;
        }
        inline int _compute_coeff_b8(int32_t *dst, uint32_t ind) {
            return _compute_coeff_b8(dst, ind, CSD, _b_dual, _dhalf, _dshift);
        }
        inline int _compute_coeff_b8(int32_t *dst, int8_t *src, int32_t *sum) {
            long dnd = 0;
            __m256i ymm[8];
            __m256i x = _mm256_sub_epi32(_mm256_set1_epi32(_dhalf), _mm256_loadu_si256((__m256i *)sum));
            while (dnd < CSD - 7) {
                vdp8x8(ymm, _b_dual+dnd*vec_length, src);
                _mm256_store_si256((__m256i *)(&dst[dnd*8+0*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[0], x), _dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+1*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[1], x), _dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+2*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[2], x), _dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+3*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[3], x), _dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+4*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[4], x), _dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+5*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[5], x), _dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+6*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[6], x), _dshift));
                _mm256_store_si256((__m256i *)(&dst[dnd*8+7*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[7], x), _dshift));
                dnd += 8;
            }
            if (dnd < CSD) {
                vdpnx8(ymm, _b_dual+dnd*vec_length, src, CSD-dnd);
                for (long i = 0; i < CSD - dnd; i++) {
                    _mm256_store_si256((__m256i *)(&dst[dnd*8+i*8]), _mm256_srai_epi32(_mm256_add_epi32(ymm[i], x), _dshift));
                }
            }
            return 0;
        }
        inline void _compute_fvec(float *dst, int32_t *coeff) {
            set_zero_avx2(dst, vec_length);
            for (long i = 0; i < CSD; i++) {
                red_avx2(dst, _b_local[i], -coeff[i], i+1);
            }
        }
        inline void _compute_fvec_b8(float *dst, int32_t *coeff) {
            set_zero_avx2(dst, vec_length * 8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(dst+vec_length*0, _b_local[i], -coeff[i*8+0], i+1);
                red_avx2(dst+vec_length*1, _b_local[i], -coeff[i*8+1], i+1);
                red_avx2(dst+vec_length*2, _b_local[i], -coeff[i*8+2], i+1);
                red_avx2(dst+vec_length*3, _b_local[i], -coeff[i*8+3], i+1);
                red_avx2(dst+vec_length*4, _b_local[i], -coeff[i*8+4], i+1);
                red_avx2(dst+vec_length*5, _b_local[i], -coeff[i*8+5], i+1);
                red_avx2(dst+vec_length*6, _b_local[i], -coeff[i*8+6], i+1);
                red_avx2(dst+vec_length*7, _b_local[i], -coeff[i*8+7], i+1);
            }
        }
        inline void _compute_fnorm_b8(float *fnorm, float *fvec) {
            __m256 dst[8];
            for (long j = 0; j < 8; j++) dst[j] = _mm256_setzero_ps();
            for (long i = 0; i < vec_length; i += 8) {
                for (long j = 0; j < 8; j++) {
                    dst[j] = _mm256_fmadd_ps(_mm256_load_ps(fvec + j * vec_length + i), _mm256_load_ps(fvec + j * vec_length + i), dst[j]);
                }
            }
            dst[0] = _mm256_hadd_ps(dst[0], dst[1]);
            dst[2] = _mm256_hadd_ps(dst[2], dst[3]);
            dst[4] = _mm256_hadd_ps(dst[4], dst[5]);
            dst[6] = _mm256_hadd_ps(dst[6], dst[7]);
            dst[0] = _mm256_hadd_ps(dst[0], dst[2]);
            dst[4] = _mm256_hadd_ps(dst[4], dst[6]);
            __m256 dst_lo = _mm256_permute2f128_ps(dst[0], dst[4], 48);
            __m256 dst_hi = _mm256_permute2f128_ps(dst[0], dst[4], 33);
            _mm256_store_ps(fnorm, _mm256_mul_ps(_mm256_add_ps(dst_lo, dst_hi), _mm256_set1_ps(0.5)));
        }
        inline uint32_t _check_fvec_overflow(float *fvec) {
            __m256 upb = _mm256_set1_ps(127.4);
            __m256 lob = _mm256_set1_ps(-128.4);

            __m256 dst = _mm256_setzero_ps();
            for (long i = 0; i < vec_length; i += 8) {
                dst = _mm256_or_ps(dst, _mm256_cmp_ps(_mm256_load_ps(fvec + i), upb, 30));
                dst = _mm256_or_ps(dst, _mm256_cmp_ps(lob, _mm256_load_ps(fvec + i), 30));
            }

            return _mm256_movemask_ps(dst);
        }
        inline uint32_t _check_fvec_overflow_b8(float *fvec) {
            __m256 upb = _mm256_set1_ps(127.4);
            __m256 lob = _mm256_set1_ps(-128.4);

            __m256 dst[8];
            for (long j = 0; j < 8; j++) dst[j] = _mm256_setzero_ps();
            for (long i = 0; i < vec_length; i += 8) {
                for (long j = 0; j < 8; j++) {
                    dst[j] = _mm256_or_ps(dst[j], _mm256_cmp_ps(_mm256_load_ps(fvec + j * vec_length + i), upb, 30));
                    dst[j] = _mm256_or_ps(dst[j], _mm256_cmp_ps(lob, _mm256_load_ps(fvec + j * vec_length + i), 30));
                }
            }
            uint32_t ret = 0;
            for (long j = 0; j < 8; j++) {
                if (_mm256_movemask_ps(dst[j])) ret |= (1 << j);
            }
            return ret;
        }
        inline int32_t _compute_sum(int8_t *src) {
            __m256i all_one = _mm256_set1_epi8(0x80);
            __m256i acc = _mm256_setzero_si256();
            for (long i = 0; i < vec_length; i += 32) {
                acc = _mm256_dpbusd_epi32(acc, all_one, _mm256_load_si256((__m256i *)(src+i)));
            }
            __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 78));
            acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 177));
            return _mm_cvtsi128_si32(acc128);
        }
        inline void _compute_sum_b8(int32_t *dst, int8_t *src) {
            __m256i all_one = _mm256_set1_epi8(0x80);
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256();
            __m256i acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256();
            __m256i acc7 = _mm256_setzero_si256();
            for (long i = 0; i < vec_length; i += 32) {
                acc0 = _mm256_dpbusd_epi32(acc0, all_one, _mm256_load_si256((__m256i *)(src+i+0*vec_length)));
                acc1 = _mm256_dpbusd_epi32(acc1, all_one, _mm256_load_si256((__m256i *)(src+i+1*vec_length)));
                acc2 = _mm256_dpbusd_epi32(acc2, all_one, _mm256_load_si256((__m256i *)(src+i+2*vec_length)));
                acc3 = _mm256_dpbusd_epi32(acc3, all_one, _mm256_load_si256((__m256i *)(src+i+3*vec_length)));
                acc4 = _mm256_dpbusd_epi32(acc4, all_one, _mm256_load_si256((__m256i *)(src+i+4*vec_length)));
                acc5 = _mm256_dpbusd_epi32(acc5, all_one, _mm256_load_si256((__m256i *)(src+i+5*vec_length)));
                acc6 = _mm256_dpbusd_epi32(acc6, all_one, _mm256_load_si256((__m256i *)(src+i+6*vec_length)));
                acc7 = _mm256_dpbusd_epi32(acc7, all_one, _mm256_load_si256((__m256i *)(src+i+7*vec_length)));
            }
            acc0 = _mm256_hadd_epi32(acc0, acc1);
            acc2 = _mm256_hadd_epi32(acc2, acc3);
            acc4 = _mm256_hadd_epi32(acc4, acc5);
            acc6 = _mm256_hadd_epi32(acc6, acc7);
            acc0 = _mm256_hadd_epi32(acc0, acc2);
            acc4 = _mm256_hadd_epi32(acc4, acc6);
            __m256i acclo = _mm256_permute2f128_si256(acc0, acc4, 48);
            __m256i acchi = _mm256_permute2f128_si256(acc0, acc4, 33);
            _mm256_storeu_si256((__m256i *)dst, _mm256_add_epi32(acclo, acchi));
        }

        // low level operations, only used in bgjn sieves
        template <uint32_t batchsize, bool record_dp, bool faraway_center, bool for_bgj1, bool init_sieve>
        int _pool_bucketing(bucket_epi8_t<record_dp> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2);
        template <uint32_t batchsize, bool faraway_center, bool for_bgj2, bool init_sieve, bool profiling>
        int _sub_bucketing(bucket_epi8_t<0> *main_bucket, bucket_epi8_t<0> **dst3, bucket_epi8_t<0> **dst2, double alpha3, double alpha2, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        template <bool record_dp, bool profiling>
        int _search_cred(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        template <uint32_t l2_block, uint32_t l1_block, bool record_dp, bool profiling>
        int _search_np(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        template <uint32_t l2_block, uint32_t l1_block, bool record_dp, bool profiling>
        int _search_pp(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        template <uint32_t l2_block, uint32_t l1_block, bool record_dp, bool profiling>
        int _search_nn(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        #if defined(HAVE_CUDA)
        template <bool record_dp, bool profiling>
        int _search_bgj1_cuda(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        template <bool record_dp, bool profiling>
        int _search_bgj1_cuda_overlap(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        template <bool record_dp, bool profiling>
        int _search_bgj1_cuda_batch(bucket_epi8_t<record_dp> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<nb> *prof = NULL);
        template <bool record_dp, bool profiling>
        int _search_bgj1_cuda_batch_collect(bucket_epi8_t<record_dp> **buckets, long num_bucket, int32_t goal_norm, bgj_cuda_result_t *result_storage, uint32_t result_capacity, bgj_cuda_result_t **result_ptrs, uint32_t *result_counts, int *overflows);
        template <bool record_dp, bool profiling>
        int _search_bgj1_cuda_np_batch_collect(bucket_epi8_t<record_dp> **buckets, long num_bucket, int32_t goal_norm, int include_same_pairs, bgj_cuda_result_t *result_storage, uint32_t result_capacity, bgj_cuda_result_t **result_ptrs, uint32_t *result_counts, int *overflows);
        template <bool record_dp, bool profiling>
        int _consume_bgj1_cuda_results(bucket_epi8_t<record_dp> *bkt, sol_list_epi8_t *sol, bgj_profile_data_t<nb> *prof, bgj_cuda_result_t *results, uint32_t result_count);
        long _sol_list_to_desc(sol_list_epi8_t **sol_list, long num_sol_list, bgj_cuda_materialize_desc_t *desc, uint64_t *dst_vu);
        int _desc_to_vec_cpu(const bgj_cuda_materialize_desc_t *desc, long num_desc, long cpu_threads, int8_t *dst_vec, int32_t *dst_vnorm, int32_t *dst_vsum);
        int _sol_list_to_vec_cpu_parallel(sol_list_epi8_t **sol_list, long num_sol_list, int8_t *dst_vec, uint64_t *dst_vu, int32_t *dst_vnorm, int32_t *dst_vsum);
        int _sol_list_to_vec_cuda(sol_list_epi8_t **sol_list, long num_sol_list, int8_t *dst_vec, uint64_t *dst_vu, int32_t *dst_vnorm, int32_t *dst_vsum);
        int _sol_list_to_vec_cuda_staged(sol_list_epi8_t **sol_list, long num_sol_list, uint64_t *dst_vu, int32_t *dst_vnorm, int32_t *dst_vsum);
        #endif
        int _sol_list_to_vec(sol_list_epi8_t **sol_list, long num_sol_list, int8_t *dst_vec, uint64_t *dst_vu, int32_t *dst_vnorm, int32_t *dst_vsum);
        template <bool profiling>
        uint64_t _pool_insert(sol_list_epi8_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_profile_data_t<nb> *prof = NULL);
        
        // low level operations, only used in dh_insert
        template <uint32_t ndual, uint32_t dh_dim>
        void _compute_ndh_mblock(float *fvec, int8_t *dh, float **b_ext, float *dual_vec, long MInd, long num, long target_index);
        template <uint32_t ndual>
        void _process_ndhl1_triblock(int8_t *dh, long bias, long bound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
        template <uint32_t ndual>
        void _process_ndhl1_block(int8_t *dhi, int8_t *dhj, long ibias, long jbias, long ibound, long jbound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
        template <uint32_t ndual>
        inline void _vnpnorm_check8x8(uint64_t *cmpp, uint64_t *cmpn, int8_t *s1, int8_t *s2, __m256i th);
        template <uint32_t ndual>
        inline void _vnpnorm_check8xn(uint64_t *cmpp, uint64_t *cmpn, int8_t *s1, int8_t *s2, long n, __m256i th);
        template <uint32_t ndual>
        inline void _vnpnorm_checknx8(uint64_t *cmpp, uint64_t *cmpn, int8_t *s1, int8_t *s2, long n, __m256i th);
        template <uint32_t ndual>
        uint64_t _vnpnorm(int8_t *s1, int8_t *s2);
        template <uint32_t ndual>
        inline int32_t _compute_norm(int8_t *s);
        template <uint32_t ndual>
        inline __m256i _compute_norm_b8(int8_t *s);
        template <uint32_t shsize>
        void _process_nshl1_triblock(uint8_t *sh, long bias, long bound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
        template <uint32_t shsize>
        void _process_nshl1_block(uint8_t *shi, uint8_t *shj, long ibias, long jbias, long ibound, long jbound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
        template <uint32_t shsize>
        inline void _vshdot_check8x8(uint64_t *cmpp, uint64_t *cmpn, uint8_t *u1, uint8_t *u2, __m512i lb, __m512i ub);
        template <uint32_t shsize>
        inline void _vshdot_check8xn(uint64_t *cmpp, uint64_t *cmpn, uint8_t *u1, uint8_t *u2, long n, __m512i lb, __m512i ub);
        template <uint32_t shsize>
        inline int32_t _vshdot(uint8_t *u1, uint8_t *u2);


        template <uint32_t ndual, uint32_t dh_dim, uint32_t l1_block, uint32_t l2_block, uint32_t m_block>
        int _naivedh_insert(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length = 0.0);
        template <uint32_t shsize, uint32_t dh_dim, uint32_t l1_block, uint32_t l2_block, uint32_t m_block>
        int _naivesh_insert(long target_index, double eta, long log_level, 
                            float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_length = 0.0);
        template <uint32_t ndual, uint32_t dh_dim, uint32_t m_block>
        int _lsfdh_insert(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length = 0.0, double stop_length = 0.0);
        template <uint32_t shsize, uint32_t dh_dim>
        int _lsfsh_insert(long target_index, double eta, long log_level, float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_ratio, double target_length = 0.0, double qratio = 0.0);
        template <uint32_t shsize, uint32_t dh_dim>
        int _show_lsfsh_insert(long target_index, double eta, long log_level, float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_ratio, double target_length = 0.0, double qratio = 0.0);

        // only called in lsfdh_insert
        int32_t _opt_ldh_bucket_radius(long total_num, uint32_t ndual, long type);
        void _opt_ldh_threshold(float *dual_vec, uint32_t &ndual, int32_t &threshold, double &tail_alpha_bound,
                        double dual_exp_len, const double *tail_alpha_prob_list, Lattice_QP *L, double ratio, long log_level);
        void _opt_nsh_threshold(float *dual_vec, uint32_t *compress_pos, int32_t &num_hbits, int32_t &num_tbits, int32_t &threshold, 
                        Lattice_QP *b_mid, uint32_t shsize, double exp_length, double *tail_alpha_prob_list, long log_level);
        // only called in dh/sh insert
        int __basis_insert(long dst_index, float *v_fp, long FD, float **b_full_fp);
        template <uint32_t shsize, uint32_t l1_block, uint32_t l2_block>
        int __parallel_mblock_sh_search(nsh_mblock_t mb, int32_t threshold, uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, nsh_profile_data_t *profile_data);
        template <uint32_t shsize, uint32_t l1_block, uint32_t l2_block>
        int __mblock_sh_search(lsh_mblock_t mb, int32_t threshold, uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, lsh_profile_data_t *profile_data, 
                                long target_index, uint32_t dh_dim, float **b_full_fp, float *min_norm, float **min_vec, pthread_spinlock_t &min_lock);
        int __lift_buffer(nsh_mblock_t mb, long target_index, uint32_t dh_dim, float **b_full_fp, 
                        uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, 
                        float *min_norm, float **min_vec, nsh_profile_data_t *profile_data);
        int __lift_buffer(lsh_mblock_t mb, long target_index, uint32_t dh_dim, float **b_full_fp, 
                        uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, 
                        float *min_norm, float **min_vec, pthread_spinlock_t &min_lock, lsh_profile_data_t *profile_data);

        #if defined(__AMX_INT8__)
        template <uint32_t batchsize, bool faraway_center, bool reuse>
        int _pool_bucketing_amx(bucket_amx_t **rbucket0, bucket_amx_t **bucket0, double alpha_r0, double alpha_b0,  sol_list_amx_t **sol_list, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof = NULL);
        template <uint32_t batchsize, bool faraway_center, bool reuse>
        int _parallel_sub_bucketing_amx(bucket_amx_t *main_bucket, bucket_amx_t **rbucket, bucket_amx_t **bucket, double alpha_r, double alpha_b, sol_list_amx_t **sol_list, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof = NULL);
        template <uint32_t batchsize, bool faraway_center, bool reuse>
        int _sub_bucketing_amx(bucket_amx_t *main_bucket, bucket_amx_t **rbucket, bucket_amx_t **bucket, double alpha_r, double alpha_b, sol_list_amx_t *sol, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof = NULL);
        // for sieving dimension < 150, the whole bucket should be in L2 cache which is ~ 6000 vectors
        // pay attention to the size of rbuckets, it may larger than normal buckets
        int _search_amx(bucket_amx_t *bkt, sol_list_amx_t *sol, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof = NULL);
        template <bool profiling>
        uint64_t _pool_insert_amx(sol_list_amx_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_amx_profile_data_t<nb> *prof = NULL);
        int _sol_list_to_vec_amx(sol_list_amx_t **sol_list, long num_sol_list, int8_t *dst_vec, uint64_t *dst_vu, int32_t *dst_vnorm, int32_t *dst_vsum);
        #endif
        constexpr static const __m256i epi8_sign_bit = {-9187201950435737472, -9187201950435737472, -9187201950435737472, -9187201950435737472};
        constexpr static const __m256i epi32_01_bit = {0x000000ff000000ff, 0x000000ff000000ff, 0x000000ff000000ff, 0x000000ff000000ff};
        constexpr static const __m256i epi32_23_bit = {0x0000ff000000ff00, 0x0000ff000000ff00, 0x0000ff000000ff00, 0x0000ff000000ff00};
        constexpr static const __m256i epi32_45_bit = {0x00ff000000ff0000, 0x00ff000000ff0000, 0x00ff000000ff0000, 0x00ff000000ff0000};
        constexpr static const __m256i epi32_67_bit = {-72057589759737856, -72057589759737856, -72057589759737856, -72057589759737856};
};

#endif
