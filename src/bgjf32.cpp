/** 
 * \warning: old codes based on float32, do not read it
*/
#include "../include/pool.h"
#include <sys/time.h>

#include <immintrin.h>
#include <x86intrin.h>
#include <malloc.h>

/**
 * bgj2 stragety: 
 * pool --bucketing--> 0.35bucket --searching--> cred, 2red, 3red
 *                 --> 0.24bucket --bucketing--> 0.28subbucket --searching--> cred, 2red
 * 
 * bgj3 stragety: 
 * pool --bucketing--> 0.35bucket --searching--> cred, 2red, 3red
 *                 --> 0.22bucket --bucketing--> 0.28+subbucket --searching--> cred, 2red
 *                                               0.19subbucket  --bucketing--> 0.27subsubbucket --searching--> cred, 2red
*/

#if 1
#define dlog(_format, ...) do {                             \
    printf("[thread %d] ", omp_get_thread_num());           \
    printf(_format "\n", ##__VA_ARGS__);                    \
} while (0)
    
struct timeval _timer_start_bgjf32[16], _timer_end_bgjf32[16];
double _time_curr_bgjf32[16];

#define TIMER_START do {                                                        \
        gettimeofday(&_timer_start_bgjf32[omp_get_thread_num()], NULL);         \
    } while (0)

#define TIMER_END do {                                                                                                            \
        gettimeofday(&_timer_end_bgjf32[omp_get_thread_num()], NULL);                                                             \
        _time_curr_bgjf32[omp_get_thread_num()] =                                                                                 \
            (_timer_end_bgjf32[omp_get_thread_num()].tv_sec-_timer_start_bgjf32[omp_get_thread_num()].tv_sec)+                    \
            (double)(_timer_end_bgjf32[omp_get_thread_num()].tv_usec-_timer_start_bgjf32[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_time_curr_bgjf32[omp_get_thread_num()])
#endif

#define BGJ2_BUCKET0_REUSE 1
#define BGJ2_BUCKET1_USE_NONLATTICE_CENTER 0
#define BGJ2_BUCKET0_USE_FARAWAY_CENTER 1
#define BGJ2_BUCKET0_REUSE_ALPHA 0.365
#define BGJ2_BUCKET1_CCDP 0.5

#if BGJ2_BUCKET1_USE_NONLATTICE_CENTER
#define BGJ2_BUCKET0_BATCHSIZE 72
#define BGJ2_BUCKET1_BATCHSIZE 72
#define BGJ2_BUCKET1_ALPHA 0.325
#define BGJ2_BUCKET0_ALPHA 0.25
#else
#define BGJ2_BUCKET0_BATCHSIZE 72
#define BGJ2_BUCKET1_BATCHSIZE 72
#define BGJ2_BUCKET1_ALPHA 0.290
#define BGJ2_BUCKET0_ALPHA 0.262
#endif


#if BGJ2_BUCKET0_BATCHSIZE % 8
#error BGJ2 bucket0 batchsize must divided by 8
#endif



#define BGJ2_SEARCHING_DP_BLOCK 256

#define MAX_NTHREADS 16
#define MIN_LOG_CSD 40
#define MAX_STUCK_TIME 2


#pragma region
/** 
 * \brief compute the dot product of __ptr and __srci, and store 
 *      the result in __dsti for i = 0, 1, 2, 3.
 * \param[in] __src0, __src1, __src2, __src3    a pointer to aligened float vectors of length __len
 * \param[in] __ptr                             a pointer to aligened float vectors of length __len
 * \param[out] __dsti                           a float store the dot product of __srcci and __ptr
*/
#define AVX2_DOT_PRODUCT_1X4(__ptr, __src0, __src1, __src2, __src3, __len)                          \
                                                                   do {                             \
    __m256 __r0 = _mm256_setzero_ps();                                                              \
    __m256 __r1 = _mm256_setzero_ps();                                                              \
    __m256 __r2 = _mm256_setzero_ps();                                                              \
    __m256 __r3 = _mm256_setzero_ps();                                                              \
    long __i;                                                                                       \
    for (__i = 0; __i < __len - 31; __i += 32){                                                     \
        __m256 x0 = _mm256_load_ps(__ptr+__i+0);                                                    \
        __m256 x1 = _mm256_load_ps(__ptr+__i+8);                                                    \
        __m256 x2 = _mm256_load_ps(__ptr+__i+16);                                                   \
        __m256 x3 = _mm256_load_ps(__ptr+__i+24);                                                   \
        __r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src0+__i), __r0);                               \
        __r1 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src1+__i), __r1);                               \
        __r2 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src2+__i), __r2);                               \
        __r3 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src3+__i), __r3);                               \
        __r0 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src0+__i+8), __r0);                             \
        __r1 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src1+__i+8), __r1);                             \
        __r2 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src2+__i+8), __r2);                             \
        __r3 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src3+__i+8), __r3);                             \
        __r0 = _mm256_fmadd_ps(x2, _mm256_load_ps(__src0+__i+16), __r0);                            \
        __r1 = _mm256_fmadd_ps(x2, _mm256_load_ps(__src1+__i+16), __r1);                            \
        __r2 = _mm256_fmadd_ps(x2, _mm256_load_ps(__src2+__i+16), __r2);                            \
        __r3 = _mm256_fmadd_ps(x2, _mm256_load_ps(__src3+__i+16), __r3);                            \
        __r0 = _mm256_fmadd_ps(x3, _mm256_load_ps(__src0+__i+24), __r0);                            \
        __r1 = _mm256_fmadd_ps(x3, _mm256_load_ps(__src1+__i+24), __r1);                            \
        __r2 = _mm256_fmadd_ps(x3, _mm256_load_ps(__src2+__i+24), __r2);                            \
        __r3 = _mm256_fmadd_ps(x3, _mm256_load_ps(__src3+__i+24), __r3);                            \
    }                                                                                               \
    if (__i < __len - 15){                                                                          \
        __m256 x0 = _mm256_load_ps(__ptr+__i+0);                                                    \
        __m256 x1 = _mm256_load_ps(__ptr+__i+8);                                                    \
        __r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src0+__i), __r0);                               \
        __r1 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src1+__i), __r1);                               \
        __r2 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src2+__i), __r2);                               \
        __r3 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src3+__i), __r3);                               \
        __r0 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src0+__i+8), __r0);                             \
        __r1 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src1+__i+8), __r1);                             \
        __r2 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src2+__i+8), __r2);                             \
        __r3 = _mm256_fmadd_ps(x1, _mm256_load_ps(__src3+__i+8), __r3);                             \
        __i += 16;                                                                                  \
    }                                                                                               \
    if (__i < __len - 7) {                                                                          \
        __m256 x0 = _mm256_load_ps(__ptr+__i+0);                                                    \
        __r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src0+__i), __r0);                               \
        __r1 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src1+__i), __r1);                               \
        __r2 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src2+__i), __r2);                               \
        __r3 = _mm256_fmadd_ps(x0, _mm256_load_ps(__src3+__i), __r3);                               \
    }                                                                                               \
    __m128 __r0_128 = _mm_add_ps(_mm256_castps256_ps128(__r0), _mm256_extractf128_ps(__r0, 1));     \
    __m128 __r1_128 = _mm_add_ps(_mm256_castps256_ps128(__r1), _mm256_extractf128_ps(__r1, 1));     \
    __m128 __r2_128 = _mm_add_ps(_mm256_castps256_ps128(__r2), _mm256_extractf128_ps(__r2, 1));     \
    __m128 __r3_128 = _mm_add_ps(_mm256_castps256_ps128(__r3), _mm256_extractf128_ps(__r3, 1));     \
    __r0_128 = _mm_add_ps(__r0_128, _mm_permute_ps(__r0_128, 78));                                  \
    __r1_128 = _mm_add_ps(__r1_128, _mm_permute_ps(__r1_128, 78));                                  \
    __r2_128 = _mm_add_ps(__r2_128, _mm_permute_ps(__r2_128, 78));                                  \
    __r3_128 = _mm_add_ps(__r3_128, _mm_permute_ps(__r3_128, 78));                                  \
    __r0_128 = _mm_add_ps(__r0_128, _mm_shuffle_ps(__r0_128, __r0_128, 85));                        \
    __r1_128 = _mm_add_ps(__r1_128, _mm_shuffle_ps(__r1_128, __r1_128, 85));                        \
    __r2_128 = _mm_add_ps(__r2_128, _mm_shuffle_ps(__r2_128, __r2_128, 85));                        \
    __r3_128 = _mm_add_ps(__r3_128, _mm_shuffle_ps(__r3_128, __r3_128, 85));                        \
    __dst0 = _mm_cvtss_f32(__r0_128);                                                               \
    __dst1 = _mm_cvtss_f32(__r1_128);                                                               \
    __dst2 = _mm_cvtss_f32(__r2_128);                                                               \
    __dst3 = _mm_cvtss_f32(__r3_128);                                                               \
} while(0)


/** 
 * \brief compute the dot product of __c0, __c1 and 8 vectors __ptr[0], ..., __ptr[7], 
 *      store the result in 2 __m256 register, __dst0, __dst1.
*/
#define AVX2_DP_2X8(__c0, __c1, __ptr, __len, __dst0, __dst1)                                       \
                                                        do {                                        \
    __m256 __r00 = _mm256_setzero_ps();                                                             \
    __m256 __r01 = _mm256_setzero_ps();                                                             \
    __m256 __r02 = _mm256_setzero_ps();                                                             \
    __m256 __r03 = _mm256_setzero_ps();                                                             \
    __m256 __r04 = _mm256_setzero_ps();                                                             \
    __m256 __r05 = _mm256_setzero_ps();                                                             \
    __m256 __r06 = _mm256_setzero_ps();                                                             \
    __m256 __r07 = _mm256_setzero_ps();                                                             \
    __m256 __r10 = _mm256_setzero_ps();                                                             \
    __m256 __r11 = _mm256_setzero_ps();                                                             \
    __m256 __r12 = _mm256_setzero_ps();                                                             \
    __m256 __r13 = _mm256_setzero_ps();                                                             \
    __m256 __r14 = _mm256_setzero_ps();                                                             \
    __m256 __r15 = _mm256_setzero_ps();                                                             \
    __m256 __r16 = _mm256_setzero_ps();                                                             \
    __m256 __r17 = _mm256_setzero_ps();                                                             \
    long __i = 0;                                                                                   \
    while (__i < __len - 7) {                                                                       \
        __m256 __x0 = _mm256_load_ps(__c0 + __i);                                                   \
        __m256 __x1 = _mm256_load_ps(__c1 + __i);                                                   \
        __r00 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[0] + __i), __r00);                     \
        __r01 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[1] + __i), __r01);                     \
        __r02 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[2] + __i), __r02);                     \
        __r03 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[3] + __i), __r03);                     \
        __r04 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[4] + __i), __r04);                     \
        __r05 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[5] + __i), __r05);                     \
        __r06 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[6] + __i), __r06);                     \
        __r07 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr)[7] + __i), __r07);                     \
                                                                                                    \
        __r10 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[0] + __i), __r10);                     \
        __r11 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[1] + __i), __r11);                     \
        __r12 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[2] + __i), __r12);                     \
        __r13 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[3] + __i), __r13);                     \
        __r14 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[4] + __i), __r14);                     \
        __r15 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[5] + __i), __r15);                     \
        __r16 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[6] + __i), __r16);                     \
        __r17 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr)[7] + __i), __r17);                     \
        __i+=8;                                                                                     \
    }                                                                                               \
    __r00 = _mm256_hadd_ps(__r00, __r01);                                                           \
    __r02 = _mm256_hadd_ps(__r02, __r03);                                                           \
    __r04 = _mm256_hadd_ps(__r04, __r05);                                                           \
    __r06 = _mm256_hadd_ps(__r06, __r07);                                                           \
    __r10 = _mm256_hadd_ps(__r10, __r11);                                                           \
    __r12 = _mm256_hadd_ps(__r12, __r13);                                                           \
    __r14 = _mm256_hadd_ps(__r14, __r15);                                                           \
    __r16 = _mm256_hadd_ps(__r16, __r17);                                                           \
    __r10 = _mm256_hadd_ps(__r10, __r12);                                                           \
    __r14 = _mm256_hadd_ps(__r14, __r16);                                                           \
    __r00 = _mm256_hadd_ps(__r00, __r02);                                                           \
    __r04 = _mm256_hadd_ps(__r04, __r06);                                                           \
    __m256 __r0lo = _mm256_permute2f128_ps(__r00, __r04, 48);                                       \
    __m256 __r0hi = _mm256_permute2f128_ps(__r00, __r04, 33);                                       \
    __m256 __r1lo = _mm256_permute2f128_ps(__r10, __r14, 48);                                       \
    __m256 __r1hi = _mm256_permute2f128_ps(__r10, __r14, 33);                                       \
    __dst0 = _mm256_add_ps(__r0lo, __r0hi);                                                         \
    __dst1 = _mm256_add_ps(__r1lo, __r1hi);                                                         \
} while (0)


/**
 * \brief compute the dot product of __c0 and __pi, store the result in a __m256 register, __dst.
*/
#define AVX2_DP_1X8(__c0, __p0, __p1, __p2, __p3, __p4, __p5, __p6, __p7, __len, __dst)             \
                                                                                do {                \
    __m256 __r0 = _mm256_setzero_ps();                                                              \
    __m256 __r1 = _mm256_setzero_ps();                                                              \
    __m256 __r2 = _mm256_setzero_ps();                                                              \
    __m256 __r3 = _mm256_setzero_ps();                                                              \
    __m256 __r4 = _mm256_setzero_ps();                                                              \
    __m256 __r5 = _mm256_setzero_ps();                                                              \
    __m256 __r6 = _mm256_setzero_ps();                                                              \
    __m256 __r7 = _mm256_setzero_ps();                                                              \
    long __i = 0;                                                                                   \
    while (__i < __len - 15) {                                                                      \
        __m256 __x0 = _mm256_load_ps(__c0 + __i + 0);                                               \
        __m256 __x1 = _mm256_load_ps(__c0 + __i + 8);                                               \
        __r0 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p0 + __i), __r0);                             \
        __r1 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p1 + __i), __r1);                             \
        __r2 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p2 + __i), __r2);                             \
        __r3 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p3 + __i), __r3);                             \
        __r4 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p4 + __i), __r4);                             \
        __r5 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p5 + __i), __r5);                             \
        __r6 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p6 + __i), __r6);                             \
        __r7 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p7 + __i), __r7);                             \
                                                                                                    \
        __r0 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p0 + __i + 8), __r0);                         \
        __r1 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p1 + __i + 8), __r1);                         \
        __r2 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p2 + __i + 8), __r2);                         \
        __r3 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p3 + __i + 8), __r3);                         \
        __r4 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p4 + __i + 8), __r4);                         \
        __r5 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p5 + __i + 8), __r5);                         \
        __r6 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p6 + __i + 8), __r6);                         \
        __r7 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__p7 + __i + 8), __r7);                         \
        __i += 16;                                                                                  \
    }                                                                                               \
    if (__i < __len - 7) {                                                                          \
        __m256 __x0 = _mm256_load_ps(__c0 + __i + 0);                                               \
        __r0 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p0 + __i), __r0);                             \
        __r1 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p1 + __i), __r1);                             \
        __r2 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p2 + __i), __r2);                             \
        __r3 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p3 + __i), __r3);                             \
        __r4 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p4 + __i), __r4);                             \
        __r5 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p5 + __i), __r5);                             \
        __r6 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p6 + __i), __r6);                             \
        __r7 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__p7 + __i), __r7);                             \
    }                                                                                               \
    __r0 = _mm256_hadd_ps(__r0, __r1);                                                              \
    __r2 = _mm256_hadd_ps(__r2, __r3);                                                              \
    __r4 = _mm256_hadd_ps(__r4, __r5);                                                              \
    __r6 = _mm256_hadd_ps(__r6, __r7);                                                              \
    __r0 = _mm256_hadd_ps(__r0, __r2);                                                              \
    __r4 = _mm256_hadd_ps(__r4, __r6);                                                              \
    __m256 __rlo = _mm256_permute2f128_ps(__r0, __r4, 48);                                          \
    __m256 __rhi = _mm256_permute2f128_ps(__r0, __r4, 33);                                          \
    __dst = _mm256_add_ps(__rlo, __rhi);                                                            \
} while (0)


/** 
 * \brief compute the dot product of __ci and __ptrj, and store 
 *      the result in float __dst[16] = {__dst00, __dst01, ..., __dst33}, 
 *      also, we store the cmp results in __cmp.
 * \param[in] __c0, __c1, __c2, __c3            a pointer to aligened float vectors of length __len
 * \param[in] __ptr0, __ptr1, __ptr2, __ptr3    a pointer to aligened float vectors of length __len
 * \param[in] sign_bit                          = _mm256_set1_ps(-0.0f)
 * \param[in] __anptr                           a __m256 vector
 * \param[out] __dst                            a pointer to 16 aligened float store the dot product of __ci and __ptrj
 * \param[out] __cmp0, __cmp1                   two int, store the cmp results in lowest 8 bits, __cmp0&0x00 <--> __dst00, etc.
*/
#define AVX2_DP_CMP_4X4(__c0, __c1, __c2, __c3, __ptr0, __ptr1, __ptr2, __ptr3,                     \
                            __len, __anptr, __dst, __cmp0, __cmp1) do {                             \
    __m256 __r00 = _mm256_setzero_ps();                                                             \
    __m256 __r01 = _mm256_setzero_ps();                                                             \
    __m256 __r02 = _mm256_setzero_ps();                                                             \
    __m256 __r03 = _mm256_setzero_ps();                                                             \
    __m256 __r10 = _mm256_setzero_ps();                                                             \
    __m256 __r11 = _mm256_setzero_ps();                                                             \
    __m256 __r12 = _mm256_setzero_ps();                                                             \
    __m256 __r13 = _mm256_setzero_ps();                                                             \
    __m256 __r20 = _mm256_setzero_ps();                                                             \
    __m256 __r21 = _mm256_setzero_ps();                                                             \
    __m256 __r22 = _mm256_setzero_ps();                                                             \
    __m256 __r23 = _mm256_setzero_ps();                                                             \
    __m256 __r30 = _mm256_setzero_ps();                                                             \
    __m256 __r31 = _mm256_setzero_ps();                                                             \
    __m256 __r32 = _mm256_setzero_ps();                                                             \
    __m256 __r33 = _mm256_setzero_ps();                                                             \
    long __i;                                                                                       \
    for (__i = 0; __i < __len - 7; __i += 8){                                                       \
        __m256 __x0 = _mm256_load_ps(__c0 + __i);                                                   \
        __m256 __x1 = _mm256_load_ps(__c1 + __i);                                                   \
        __m256 __x2 = _mm256_load_ps(__c2 + __i);                                                   \
        __m256 __x3 = _mm256_load_ps(__c3 + __i);                                                   \
        __m256 __y0 = _mm256_load_ps(__ptr0 + __i);                                                 \
        __m256 __y1 = _mm256_load_ps(__ptr1 + __i);                                                 \
        __m256 __y2 = _mm256_load_ps(__ptr2 + __i);                                                 \
        __m256 __y3 = _mm256_load_ps(__ptr3 + __i);                                                 \
        __r00 = _mm256_fmadd_ps(__x0, __y0, __r00);                                                 \
        __r01 = _mm256_fmadd_ps(__x0, __y1, __r01);                                                 \
        __r02 = _mm256_fmadd_ps(__x0, __y2, __r02);                                                 \
        __r03 = _mm256_fmadd_ps(__x0, __y3, __r03);                                                 \
        __r10 = _mm256_fmadd_ps(__x1, __y0, __r10);                                                 \
        __r11 = _mm256_fmadd_ps(__x1, __y1, __r11);                                                 \
        __r12 = _mm256_fmadd_ps(__x1, __y2, __r12);                                                 \
        __r13 = _mm256_fmadd_ps(__x1, __y3, __r13);                                                 \
        __r20 = _mm256_fmadd_ps(__x2, __y0, __r20);                                                 \
        __r21 = _mm256_fmadd_ps(__x2, __y1, __r21);                                                 \
        __r22 = _mm256_fmadd_ps(__x2, __y2, __r22);                                                 \
        __r23 = _mm256_fmadd_ps(__x2, __y3, __r23);                                                 \
        __r30 = _mm256_fmadd_ps(__x3, __y0, __r30);                                                 \
        __r31 = _mm256_fmadd_ps(__x3, __y1, __r31);                                                 \
        __r32 = _mm256_fmadd_ps(__x3, __y2, __r32);                                                 \
        __r33 = _mm256_fmadd_ps(__x3, __y3, __r33);                                                 \
    }                                                                                               \
    __r00 = _mm256_hadd_ps(__r00, __r01);                                                           \
    __r02 = _mm256_hadd_ps(__r02, __r03);                                                           \
    __r10 = _mm256_hadd_ps(__r10, __r11);                                                           \
    __r12 = _mm256_hadd_ps(__r12, __r13);                                                           \
    __r20 = _mm256_hadd_ps(__r20, __r21);                                                           \
    __r22 = _mm256_hadd_ps(__r22, __r23);                                                           \
    __r30 = _mm256_hadd_ps(__r30, __r31);                                                           \
    __r32 = _mm256_hadd_ps(__r32, __r33);                                                           \
                                                                                                    \
    __r00 = _mm256_hadd_ps(__r00, __r02);                                                           \
    __r10 = _mm256_hadd_ps(__r10, __r12);                                                           \
    __r20 = _mm256_hadd_ps(__r20, __r22);                                                           \
    __r30 = _mm256_hadd_ps(__r30, __r32);                                                           \
                                                                                                    \
    __m256 __r01lo = _mm256_permute2f128_ps(__r00, __r10, 48);                                      \
    __m256 __r01hi = _mm256_permute2f128_ps(__r00, __r10, 33);                                      \
    __m256 __r23lo = _mm256_permute2f128_ps(__r20, __r30, 48);                                      \
    __m256 __r23hi = _mm256_permute2f128_ps(__r20, __r30, 33);                                      \
    __m256 __r01f = _mm256_add_ps(__r01lo, __r01hi);                                                \
    __m256 __r23f = _mm256_add_ps(__r23lo, __r23hi);                                                \
    _mm256_store_ps(&__dst[0], __r01f);                                                             \
    _mm256_store_ps(&__dst[8], __r23f);                                                             \
    __m256 __r01fabs = _mm256_andnot_ps(sign_bit, __r01f);                                          \
    __m256 __r23fabs = _mm256_andnot_ps(sign_bit, __r23f);                                          \
    __m256 __cmp01 = _mm256_cmp_ps(__r01fabs, __anptr, 30);                                         \
    __m256 __cmp23 = _mm256_cmp_ps(__r23fabs, __anptr, 30);                                         \
    __cmp0 = _mm256_movemask_ps(__cmp01);                                                           \
    __cmp1 = _mm256_movemask_ps(__cmp23);                                                           \
} while(0)


/** 
 * \brief compute the dot product of __ci and __ptrj, and store 
 *      the result in float __dst[16] = {__dst00, __dst01, ..., __dst33}, 
 *      also, we store the cmp results in __cmp.
 * \param[in] __c0, __c1, __c2, __c3            a pointer to aligened float vectors of length __len
 * \param[in] __ptr0, __ptr1, __ptr2, __ptr3    a pointer to aligened float vectors of length __len
 * \param[in] sign_bit                          = _mm256_set1_ps(-0.0f)
 * \param[in] __anptr                           a __m256 vector
 * \param[out] __dst                            a pointer to 16 aligened float store the dot product of __ci and __ptrj
 * \param[out] __cmp0, __cmp1                   two int, store the cmp results in lowest 8 bits, __cmp0&0x00 <--> __dst00, etc.
*/
#define AVX2_DP_CMP_RAW_4X4(__c0, __c1, __c2, __c3, __ptr0, __ptr1, __ptr2, __ptr3, __t,            \
                            __len, __anptr, __dst, __cmp0, __cmp1) do {                             \
    __m256 __r00 = _mm256_setzero_ps();                                                             \
    __m256 __r01 = _mm256_setzero_ps();                                                             \
    __m256 __r02 = _mm256_setzero_ps();                                                             \
    __m256 __r03 = _mm256_setzero_ps();                                                             \
    __m256 __r10 = _mm256_setzero_ps();                                                             \
    __m256 __r11 = _mm256_setzero_ps();                                                             \
    __m256 __r12 = _mm256_setzero_ps();                                                             \
    __m256 __r13 = _mm256_setzero_ps();                                                             \
    __m256 __r20 = _mm256_setzero_ps();                                                             \
    __m256 __r21 = _mm256_setzero_ps();                                                             \
    __m256 __r22 = _mm256_setzero_ps();                                                             \
    __m256 __r23 = _mm256_setzero_ps();                                                             \
    __m256 __r30 = _mm256_setzero_ps();                                                             \
    __m256 __r31 = _mm256_setzero_ps();                                                             \
    __m256 __r32 = _mm256_setzero_ps();                                                             \
    __m256 __r33 = _mm256_setzero_ps();                                                             \
    long __i;                                                                                       \
    for (__i = 0; __i < __len - 7; __i += 8){                                                       \
        __m256 __x0 = _mm256_load_ps(__c0 + __i);                                                   \
        __m256 __x1 = _mm256_load_ps(__c1 + __i);                                                   \
        __m256 __x2 = _mm256_load_ps(__c2 + __i);                                                   \
        __m256 __x3 = _mm256_load_ps(__c3 + __i);                                                   \
        __m256 __y0 = _mm256_load_ps(__ptr0 + __i);                                                 \
        __m256 __y1 = _mm256_load_ps(__ptr1 + __i);                                                 \
        __m256 __y2 = _mm256_load_ps(__ptr2 + __i);                                                 \
        __m256 __y3 = _mm256_load_ps(__ptr3 + __i);                                                 \
        __r00 = _mm256_fmadd_ps(__x0, __y0, __r00);                                                 \
        __r01 = _mm256_fmadd_ps(__x0, __y1, __r01);                                                 \
        __r02 = _mm256_fmadd_ps(__x0, __y2, __r02);                                                 \
        __r03 = _mm256_fmadd_ps(__x0, __y3, __r03);                                                 \
        __r10 = _mm256_fmadd_ps(__x1, __y0, __r10);                                                 \
        __r11 = _mm256_fmadd_ps(__x1, __y1, __r11);                                                 \
        __r12 = _mm256_fmadd_ps(__x1, __y2, __r12);                                                 \
        __r13 = _mm256_fmadd_ps(__x1, __y3, __r13);                                                 \
        __r20 = _mm256_fmadd_ps(__x2, __y0, __r20);                                                 \
        __r21 = _mm256_fmadd_ps(__x2, __y1, __r21);                                                 \
        __r22 = _mm256_fmadd_ps(__x2, __y2, __r22);                                                 \
        __r23 = _mm256_fmadd_ps(__x2, __y3, __r23);                                                 \
        __r30 = _mm256_fmadd_ps(__x3, __y0, __r30);                                                 \
        __r31 = _mm256_fmadd_ps(__x3, __y1, __r31);                                                 \
        __r32 = _mm256_fmadd_ps(__x3, __y2, __r32);                                                 \
        __r33 = _mm256_fmadd_ps(__x3, __y3, __r33);                                                 \
    }                                                                                               \
    __r00 = _mm256_hadd_ps(__r00, __r01);                                                           \
    __r02 = _mm256_hadd_ps(__r02, __r03);                                                           \
    __r10 = _mm256_hadd_ps(__r10, __r11);                                                           \
    __r12 = _mm256_hadd_ps(__r12, __r13);                                                           \
    __r20 = _mm256_hadd_ps(__r20, __r21);                                                           \
    __r22 = _mm256_hadd_ps(__r22, __r23);                                                           \
    __r30 = _mm256_hadd_ps(__r30, __r31);                                                           \
    __r32 = _mm256_hadd_ps(__r32, __r33);                                                           \
                                                                                                    \
    __r00 = _mm256_hadd_ps(__r00, __r02);                                                           \
    __r10 = _mm256_hadd_ps(__r10, __r12);                                                           \
    __r20 = _mm256_hadd_ps(__r20, __r22);                                                           \
    __r30 = _mm256_hadd_ps(__r30, __r32);                                                           \
                                                                                                    \
    __m256 __r01lo = _mm256_permute2f128_ps(__r00, __r10, 48);                                      \
    __m256 __r01hi = _mm256_permute2f128_ps(__r00, __r10, 33);                                      \
    __m256 __r23lo = _mm256_permute2f128_ps(__r20, __r30, 48);                                      \
    __m256 __r23hi = _mm256_permute2f128_ps(__r20, __r30, 33);                                      \
    __m256 __r01f = _mm256_add_ps(__r01lo, __r01hi);                                                \
    __m256 __r23f = _mm256_add_ps(__r23lo, __r23hi);                                                \
    _mm256_store_ps(&__dst[0], __r01f);                                                             \
    _mm256_store_ps(&__dst[8], __r23f);                                                             \
    if (__t == 0) {                                                                                 \
        __m256 __cmp01 = _mm256_cmp_ps(__r01f, __anptr, 30);                                        \
        __m256 __cmp23 = _mm256_cmp_ps(__r23f, __anptr, 30);                                        \
        __cmp0 = _mm256_movemask_ps(__cmp01);                                                       \
        __cmp1 = _mm256_movemask_ps(__cmp23);                                                       \
    } else {                                                                                        \
        __m256 __cmp01 = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_add_ps(__r01f, __anptr), 30);    \
        __m256 __cmp23 = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_add_ps(__r23f, __anptr), 30);    \
        __cmp0 = _mm256_movemask_ps(__cmp01);                                                       \
        __cmp1 = _mm256_movemask_ps(__cmp23);                                                       \
    }                                                                                               \
} while(0)
#pragma endregion


#if BGJ2_BUCKET1_USE_NONLATTICE_CENTER
float bgj2_lps_alpha0_adj[4][12] = {
    {0.04, 0.03, 0, 0, 0, 0,0,0,0,0,0,0},                                                       // 60~70
    {0.027, 0.022, 0.017, 0.014, 0.012, 0.012,0.012,0.012,0.012,0.012,0.012,0.012},             // 70~80
    {0.025, 0.025, 0.022, 0.02, 0.01, 0,0,0,0,0,0,0},                                           // 80~90
    {0.025, 0.025, 0.022, 0.02, 0.01, 0,0,0,0,0,0,0},                                           // 90~100, todo
};

float bgj2_lps_alpha1_adj[4][12] = {
    {0.05, 0.02, 0, 0, 0, 0,0,0,0,0,0,0},                                                       // 60~70
    {0.022, 0.018, 0.01, 0.000, -0.005, -0.005},                                                // 70~80
    {0.036, 0.037, 0.033, 0.022, 0.015, 0,0,0,0,0,0,0},                                         // 80~90
    {0.036, 0.037, 0.033, 0.022, 0.015, 0,0,0,0,0,0,0},                                         // 90~100, todo
};

float bgj2_lps_alphar_adj[4][12] = {
    {0.03, 0.0, 0, 0, 0, 0,0,0,0,0,0,0},                                                        // 60~70
    {0.03, 0.02, 0.0, 0.000, 0.000, 0,0,0,0,0,0,0,0},                                           // 70~80
    {0.03, 0.02, 0.0, 0.000, 0.000, 0,0,0,0,0,0,0,0},                                           // 80~90, todo
    {0.03, 0.02, 0.0, 0.000, 0.000, 0,0,0,0,0,0,0,0},                                           // 90~100, todo
};
#else 
float bgj2_lps_alpha0_adj[4][12] = {
    {-0.01, -0.005, 0.00, 0.00, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001},        // 60~70
    {-0.01, -0.005, 0.00, 0.00, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001},        // 70~80
    {0.02, 0.02, 0.02, 0.02, 0.013, 0.005, 0,0,0,0,0,0},                                        // 80~90
    {0.02, 0.02, 0.02, 0.02, 0.018, 0.016, 0.014,0.013,0.011,0.008,0.005,0},                    // 90~100
};

float bgj2_lps_alpha1_adj[4][12] = {
    {0.032, 0.02, 0.00, 0.000, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005}, // 60~70
    {0.032, 0.02, 0.00, 0.000, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005}, // 70~80
    {0.035, 0.033, 0.035, 0.032, 0.02, 0.01, 0,0,0,0,0,0},                                      // 80~90
    {0.035, 0.033, 0.032, 0.032, 0.028, 0.028, 0.025,0.02,0.015,0.01,0.005,0},                  // 90~100
};

float bgj2_lps_alphar_adj[4][12] = {
    {0.03, 0.0, 0, 0, 0, 0, 0,0,0,0,0,0},                                                       // 60~70
    {0.03, 0.02, 0.0, 0.000, 0.000,0,0,0,0,0,0,0},                                              // 70~80
    {0.03, 0.02, 0.02, 0.02, 0.000,0,0,0,0,0,0,0},                                              // 80~90
    {0.03, 0.02, 0.02, 0.02, 0.000,0,0,0,0,0,0,0},                                              // 90~100
};
#endif


// almost the same as bgj1_sol_t
struct bgj_sol_t {
    float **a_list = NULL;
    float **s_list = NULL;
    float **aa_list = NULL;
    float **sa_list = NULL;
    float **ss_list = NULL;
    long num_a = 0, num_s = 0;
    long num_aa = 0, num_sa = 0, num_ss = 0;


    bgj_sol_t() {}
    bgj_sol_t(long size2) { 
        _alloc(size2, 0); 
        _alloc(size2, 1);
        long size3 = size2/20;
        _alloc(size3, 2); 
        _alloc(size3, 3); 
        _alloc(size3, 4);
    }
    ~bgj_sol_t(){ _clear(); }
    
    inline void add_sol_a(float *ptr1, float *ptr2){
        if (num_a == _asize) _alloc(2 * _asize + 64, 0);
        a_list[num_a * 2] =  ptr1;
        a_list[num_a * 2 + 1] = ptr2;
        num_a++;
    }
    inline void add_sol_s(float *ptr1, float *ptr2){
        if (num_s == _ssize) _alloc(2 * _ssize + 64, 1);
        s_list[num_s * 2] =  ptr1;
        s_list[num_s * 2 + 1] = ptr2;
        num_s++;
    }
    inline void add_sol_aa(float *ctr, float *ptr1, float *ptr2){
        if (num_aa == _aasize) _alloc(2 * _aasize + 64, 2);
        aa_list[num_aa * 3] = ctr;
        aa_list[num_aa * 3 + 1] = ptr1;
        aa_list[num_aa * 3 + 2] = ptr2;
        num_aa++;
    }
    inline void add_sol_sa(float *ctr, float *ptr1, float *ptr2){
        if (num_sa == _sasize) _alloc(2 * _sasize + 64, 3);
        sa_list[num_sa * 3] = ctr;
        sa_list[num_sa * 3 + 1] = ptr1;
        sa_list[num_sa * 3 + 2] = ptr2;
        num_sa++;
    }
    inline void add_sol_ss(float *ctr, float *ptr1, float *ptr2){
        if (num_ss == _sssize) _alloc(2 * _sssize + 64, 4);
        ss_list[num_ss * 3] = ctr;
        ss_list[num_ss * 3 + 1] = ptr1;
        ss_list[num_ss * 3 + 2] = ptr2;
        num_ss++;
    }

    long num_sol() {
        return num_a + num_s + num_aa + num_sa + num_ss;
    }

    int _clear(){
        if (a_list) free(a_list);           // 0
        if (s_list) free(s_list);           // 1
        if (aa_list) free(aa_list);         // 2
        if (sa_list) free(sa_list);         // 3
        if (ss_list) free(ss_list);         // 4
        a_list = NULL; s_list = NULL;
        aa_list = NULL; sa_list = NULL; ss_list = NULL;
        return 0;
    }
    int _alloc(long size, long type) {
        #define _REALLOC_PTR(__ptr, __orgsize, __r) do {                        \
            if (size <= __orgsize) return 0;                                    \
            __ptr = (float **) realloc(__ptr, size * sizeof(float *) * __r);    \
            __orgsize = size;                                                   \
            return (__ptr != NULL);                                             \
        } while (0);

        if (type == 0) _REALLOC_PTR(a_list, _asize, 2);
        if (type == 1) _REALLOC_PTR(s_list, _ssize, 2);
        if (type == 2) _REALLOC_PTR(aa_list, _aasize, 3);
        if (type == 3) _REALLOC_PTR(sa_list, _sasize, 3);
        if (type == 4) _REALLOC_PTR(ss_list, _sssize, 3);
        #undef _REALLOC_PTR
        return -1;
    }


    long _asize = 0, _ssize = 0;
    long _aasize = 0, _sasize = 0, _sssize = 0;
};

template<bool use_3red>
struct bgj_sbucket_t {
    long num_pvec = 0;
    long num_nvec = 0;

    long try_add2 = 0;
    long succ_add2 = 0;
    long try_add3 = 0;
    long succ_add3 = 0;

    long vec_length;

    float **pvec = NULL;
    float **nvec = NULL;
    float *pdot = NULL;
    float *ndot = NULL;
    float *pnorm = NULL;
    float *nnorm = NULL;
    //uint64_t *pu = NULL;
    //uint64_t *nu = NULL;

    float *center = NULL;
    uint64_t center_u;
    float center_norm;

    bgj_sbucket_t() {}
    bgj_sbucket_t(long size) { _alloc(size, 0); _alloc(size, 1); }
    ~bgj_sbucket_t(){ _clear(); }


    #pragma region
    #define TRY_ADDA_TO_DST(__ptr0, __ptr1, __u0, __u1) do {    \
        try_add2++;                                             \
        uint64_t __u = __u0 + __u1;                             \
        if (uid->insert_uid(__u)) {                             \
            succ_add2++;                                        \
            sol->add_sol_a(__ptr0, __ptr1);                     \
        }                                                       \
    } while (0)

    #define TRY_ADDS_TO_DST(__ptr0, __ptr1, __u0, __u1) do {    \
        try_add2++;                                             \
        uint64_t __u = __u0 - __u1;                             \
        if (uid->insert_uid(__u)) {                             \
            succ_add2++;                                        \
            sol->add_sol_s(__ptr0, __ptr1);                     \
        }                                                       \
    } while (0)

    #define TRY_ADDAA_TO_DST(__ptr0, __ptr1, __u0, __u1) do {   \
        try_add3++;                                             \
        uint64_t __u = center_u + __u0 + __u1;                  \
        if (uid->insert_uid(__u)) {                             \
            succ_add3++;                                        \
            sol->add_sol_aa(center, __ptr0, __ptr1);            \
        }                                                       \
    } while (0)

    #define TRY_ADDSA_TO_DST(__ptr0, __ptr1, __u0, __u1) do {   \
        try_add3++;                                             \
        uint64_t __u = center_u - __u0 + __u1;                  \
        if (uid->insert_uid(__u)) {                             \
            succ_add3++;                                        \
            sol->add_sol_sa(center, __ptr0, __ptr1);            \
        }                                                       \
    } while (0)

    #define TRY_ADDAS_TO_DST(__ptr0, __ptr1, __u0, __u1) do {   \
        try_add3++;                                             \
        uint64_t __u = center_u + __u0 - __u1;                  \
        if (uid->insert_uid(__u)) {                             \
            succ_add3++;                                        \
            sol->add_sol_sa(center, __ptr1, __ptr0);            \
        }                                                       \
    } while (0)

    #define TRY_ADDSS_TO_DST(__ptr0, __ptr1, __u0, __u1) do {   \
        try_add3++;                                             \
        uint64_t __u = center_u - __u0 - __u1;                  \
        if (uid->insert_uid(__u)) {                             \
            succ_add3++;                                        \
            sol->add_sol_ss(center, __ptr0, __ptr1);            \
        }                                                       \
    } while (0)

    #define CHECK_AND_ADD2_1X8(__cmp, __ctr, __cu, __ptr, __pu, __add_func) do {                        \
        if (__cmp){                                                                                     \
            if (__cmp & 0xf){                                                                           \
                if (__cmp & 0x3) {                                                                      \
                    if (__cmp & 0x1) __add_func(__ctr, (__ptr)[0], __cu, (__pu)[0]);                    \
                    if (__cmp & 0x2) __add_func(__ctr, (__ptr)[1], __cu, (__pu)[1]);                    \
                }                                                                                       \
                if (__cmp & 0xc) {                                                                      \
                    if (__cmp & 0x4) __add_func(__ctr, (__ptr)[2], __cu, (__pu)[2]);                    \
                    if (__cmp & 0x8) __add_func(__ctr, (__ptr)[3], __cu, (__pu)[3]);                    \
                }                                                                                       \
            }                                                                                           \
            if (__cmp & 0xf0){                                                                          \
                if (__cmp & 0x30) {                                                                     \
                    if (__cmp & 0x10) __add_func(__ctr, (__ptr)[4], __cu, (__pu)[4]);                   \
                    if (__cmp & 0x20) __add_func(__ctr, (__ptr)[5], __cu, (__pu)[5]);                   \
                }                                                                                       \
                if (__cmp & 0xc0) {                                                                     \
                    if (__cmp & 0x40) __add_func(__ctr, (__ptr)[6], __cu, (__pu)[6]);                   \
                    if (__cmp & 0x80) __add_func(__ctr, (__ptr)[7], __cu, (__pu)[7]);                   \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
    } while (0)

    #define CHECK_AND_ADD2_1X8_NU(__cmp, __ctr, __ptr, __add_func) do {                                                                         \
        if (__cmp){                                                                                                                             \
            if (__cmp & 0xf){                                                                                                                   \
                if (__cmp & 0x3) {                                                                                                              \
                    if (__cmp & 0x1) __add_func(__ctr, (__ptr)[0], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[0])[-4])))));    \
                    if (__cmp & 0x2) __add_func(__ctr, (__ptr)[1], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[1])[-4])))));    \
                }                                                                                                                               \
                if (__cmp & 0xc) {                                                                                                              \
                    if (__cmp & 0x4) __add_func(__ctr, (__ptr)[2], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[2])[-4])))));    \
                    if (__cmp & 0x8) __add_func(__ctr, (__ptr)[3], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[3])[-4])))));    \
                }                                                                                                                               \
            }                                                                                                                                   \
            if (__cmp & 0xf0){                                                                                                                  \
                if (__cmp & 0x30) {                                                                                                             \
                    if (__cmp & 0x10) __add_func(__ctr, (__ptr)[4], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[4])[-4])))));   \
                    if (__cmp & 0x20) __add_func(__ctr, (__ptr)[5], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[5])[-4])))));   \
                }                                                                                                                               \
                if (__cmp & 0xc0) {                                                                                                             \
                    if (__cmp & 0x40) __add_func(__ctr, (__ptr)[6], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[6])[-4])))));   \
                    if (__cmp & 0x80) __add_func(__ctr, (__ptr)[7], (*((uint64_t *)(&(__ctr[-4])))), (*((uint64_t *)(&(((__ptr)[7])[-4])))));   \
                }                                                                                                                               \
            }                                                                                                                                   \
        }                                                                                                                                       \
    } while (0)
    #pragma endregion


    inline void set_center(float *ptr){
        if (center == NULL) {
            center = ptr;
            if (use_3red) {
                center_norm = ptr[-1];
                center_u = *((uint64_t *)(&ptr[-4]));
            }
        }
    }
    inline void add_pvec(float *ptr, float _n, float *_dot = NULL){
        if (num_pvec == _psize) _alloc(_psize * 2 + 64, 1);
        pvec[num_pvec] = ptr;
        pnorm[num_pvec] = _n;
        if (use_3red) pdot[num_pvec] = *_dot;
        num_pvec++;
    }
    inline void add_nvec(float *ptr, float _n, float *_dot = NULL){
        if (num_nvec == _nsize) _alloc(_nsize * 2 + 64, 0);
        nvec[num_nvec] = ptr;
        nnorm[num_nvec] = _n;
        if (use_3red) ndot[num_nvec] = *_dot;
        num_nvec++;
    }
    
    int search_naive(bgj_sol_t *sol, UidHashTable *uid, float goal_norm){
        // we have already remove center from pvec
        // we will multiply pdot and ndot by 0.5 in this function

        __m256 half = _mm256_set1_ps(0.5f);
        __attribute__ ((aligned (32))) int cmp[8];
        long *cmpl = (long *)cmp; 
        __m256 cn = _mm256_set1_ps(0.5 *(goal_norm - center_norm));

        // to optimize
        //pu = (uint64_t *) malloc(num_pvec * sizeof(uint64_t));
        //nu = (uint64_t *) malloc(num_nvec * sizeof(uint64_t));
        //for (long i = 0; i < num_pvec; i++) pu[i] = *((uint64_t *)(&pvec[i][-4]));
        //for (long i = 0; i < num_nvec; i++) nu[i] = *((uint64_t *)(&nvec[i][-4]));
        
        // p part
        long ind = 0;
        while (ind < num_pvec - 63) {
            __m256 pn0 = _mm256_loadu_ps(pnorm + ind+0);
            __m256 pn1 = _mm256_loadu_ps(pnorm + ind+8);
            __m256 pn2 = _mm256_loadu_ps(pnorm + ind+16);
            __m256 pn3 = _mm256_loadu_ps(pnorm + ind+24);
            __m256 pn4 = _mm256_loadu_ps(pnorm + ind+32);
            __m256 pn5 = _mm256_loadu_ps(pnorm + ind+40);
            __m256 pn6 = _mm256_loadu_ps(pnorm + ind+48);
            __m256 pn7 = _mm256_loadu_ps(pnorm + ind+56);
            pn0 = _mm256_mul_ps(pn0, half);
            pn1 = _mm256_mul_ps(pn1, half);
            pn2 = _mm256_mul_ps(pn2, half);
            pn3 = _mm256_mul_ps(pn3, half);
            pn4 = _mm256_mul_ps(pn4, half);
            pn5 = _mm256_mul_ps(pn5, half);
            pn6 = _mm256_mul_ps(pn6, half);
            pn7 = _mm256_mul_ps(pn7, half);
            if (use_3red) {
                cmp[0] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn0, _mm256_loadu_ps(pdot + ind+0)), 30));
                cmp[1] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn1, _mm256_loadu_ps(pdot + ind+8)), 30));
                cmp[2] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn2, _mm256_loadu_ps(pdot + ind+16)), 30));
                cmp[3] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn3, _mm256_loadu_ps(pdot + ind+24)), 30));
                cmp[4] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn4, _mm256_loadu_ps(pdot + ind+32)), 30));
                cmp[5] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn5, _mm256_loadu_ps(pdot + ind+40)), 30));
                cmp[6] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn6, _mm256_loadu_ps(pdot + ind+48)), 30));
                cmp[7] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn7, _mm256_loadu_ps(pdot + ind+56)), 30));
                for (long i = 0; i < 4; i++){
                    if (cmpl[i]){
                        CHECK_AND_ADD2_1X8_NU(cmp[2*i], center, pvec+ind+i*16+0, TRY_ADDS_TO_DST);
                        CHECK_AND_ADD2_1X8_NU(cmp[2*i+1], center, pvec+ind+i*16+8, TRY_ADDS_TO_DST);
                    }
                }
            }
            _mm256_storeu_ps(pnorm+ind+0, pn0);
            _mm256_storeu_ps(pnorm+ind+8, pn1);
            _mm256_storeu_ps(pnorm+ind+16, pn2);
            _mm256_storeu_ps(pnorm+ind+24, pn3);
            _mm256_storeu_ps(pnorm+ind+32, pn4);
            _mm256_storeu_ps(pnorm+ind+40, pn5);
            _mm256_storeu_ps(pnorm+ind+48, pn6);
            _mm256_storeu_ps(pnorm+ind+56, pn7);
            ind += 64;
        }
        while (ind < num_pvec - 7) {
            __m256 pn0 = _mm256_loadu_ps(pnorm+ind+0);
            pn0 = _mm256_mul_ps(pn0, half);
            if (use_3red) {
                int cmpp = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(pn0, _mm256_loadu_ps(pdot+ind)), 30));
                CHECK_AND_ADD2_1X8_NU(cmpp, center, pvec+ind, TRY_ADDS_TO_DST);
            }
            _mm256_storeu_ps(pnorm+ind, pn0);
            ind += 8;
        }
        while (ind < num_pvec) {
            pnorm[ind] *= 0.5f;
            if (use_3red){
                if (-pdot[ind] + pnorm[ind] < 0.5 * (goal_norm - center_norm)) TRY_ADDS_TO_DST(center, pvec[ind], center_u, *((uint64_t *)&(pvec[ind][-4])));
            }
            ind++;
        }

        
        // n part 
        ind = 0;
        while (ind < num_nvec - 63) {
            __m256 pn0 = _mm256_loadu_ps(nnorm+ind+0);
            __m256 pn1 = _mm256_loadu_ps(nnorm+ind+8);
            __m256 pn2 = _mm256_loadu_ps(nnorm+ind+16);
            __m256 pn3 = _mm256_loadu_ps(nnorm+ind+24);
            __m256 pn4 = _mm256_loadu_ps(nnorm+ind+32);
            __m256 pn5 = _mm256_loadu_ps(nnorm+ind+40);
            __m256 pn6 = _mm256_loadu_ps(nnorm+ind+48);
            __m256 pn7 = _mm256_loadu_ps(nnorm+ind+56);
            pn0 = _mm256_mul_ps(pn0, half);
            pn1 = _mm256_mul_ps(pn1, half);
            pn2 = _mm256_mul_ps(pn2, half);
            pn3 = _mm256_mul_ps(pn3, half);
            pn4 = _mm256_mul_ps(pn4, half);
            pn5 = _mm256_mul_ps(pn5, half);
            pn6 = _mm256_mul_ps(pn6, half);
            pn7 = _mm256_mul_ps(pn7, half);
            if (use_3red) {
                cmp[0] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn0, _mm256_loadu_ps(ndot+ind+0)), 30));
                cmp[1] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn1, _mm256_loadu_ps(ndot+ind+8)), 30));
                cmp[2] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn2, _mm256_loadu_ps(ndot+ind+16)), 30));
                cmp[3] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn3, _mm256_loadu_ps(ndot+ind+24)), 30));
                cmp[4] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn4, _mm256_loadu_ps(ndot+ind+32)), 30));
                cmp[5] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn5, _mm256_loadu_ps(ndot+ind+40)), 30));
                cmp[6] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn6, _mm256_loadu_ps(ndot+ind+48)), 30));
                cmp[7] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn7, _mm256_loadu_ps(ndot+ind+56)), 30));
                for (long i = 0; i < 4; i++){
                    if (cmpl[i]){
                        CHECK_AND_ADD2_1X8_NU(cmp[2*i], center, nvec+ind+i*16+0, TRY_ADDA_TO_DST);
                        CHECK_AND_ADD2_1X8_NU(cmp[2*i+1], center, nvec+ind+i*16+8, TRY_ADDA_TO_DST);
                    }
                }
            }
            _mm256_storeu_ps(nnorm+ind+0, pn0);
            _mm256_storeu_ps(nnorm+ind+8, pn1);
            _mm256_storeu_ps(nnorm+ind+16, pn2);
            _mm256_storeu_ps(nnorm+ind+24, pn3);
            _mm256_storeu_ps(nnorm+ind+32, pn4);
            _mm256_storeu_ps(nnorm+ind+40, pn5);
            _mm256_storeu_ps(nnorm+ind+48, pn6);
            _mm256_storeu_ps(nnorm+ind+56, pn7);
            ind += 64;
        }
        while (ind < num_nvec - 7) {
            __m256 pn0 = _mm256_loadu_ps(nnorm+ind+0);
            pn0 = _mm256_mul_ps(pn0, half);
            if (use_3red) {
                int cmpp = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(pn0, _mm256_loadu_ps(ndot+ind)), 30));
                CHECK_AND_ADD2_1X8_NU(cmpp, center, nvec+ind, TRY_ADDA_TO_DST);
            }
            _mm256_storeu_ps(nnorm+ind, pn0);
            ind += 8;
        }
        while (ind < num_nvec) {
            nnorm[ind] *= 0.5f;
            if (use_3red){
                if (ndot[ind] + nnorm[ind] < 0.5 * (goal_norm - center_norm)) TRY_ADDA_TO_DST(center, nvec[ind], center_u, *((uint64_t *)&(nvec[ind][-4])));
            }
            ind++;
        }

        return 0;
    }
    int search_pp(bgj_sol_t *sol, UidHashTable *uid, float goal_norm){
        for (long Ind = 0; Ind < num_pvec; Ind += BGJ2_SEARCHING_DP_BLOCK){
            for (long Jnd = Ind; Jnd < num_pvec; Jnd += BGJ2_SEARCHING_DP_BLOCK){
                const long Ibound = (Ind + BGJ2_SEARCHING_DP_BLOCK > num_pvec) ? num_pvec : Ind + BGJ2_SEARCHING_DP_BLOCK;
                const long Jbound = (Jnd + BGJ2_SEARCHING_DP_BLOCK > num_pvec) ? num_pvec : Jnd + BGJ2_SEARCHING_DP_BLOCK;

                __m256 gn, cn;
                gn = _mm256_set1_ps(0.5 * goal_norm);
                if (use_3red) cn = _mm256_set1_ps(0.5 * (goal_norm - center_norm));

                long ind = Ind;
                while (ind < Ibound - 7) {
                    __m256 b2, b3;
                    b2 = _mm256_sub_ps(gn, _mm256_loadu_ps(pnorm + ind));
                    if (use_3red) b3 = _mm256_sub_ps(_mm256_add_ps(cn, _mm256_loadu_ps(pdot + ind)), _mm256_loadu_ps(pnorm + ind));
                    long jnd = (Ind == Jnd) ? ind + 8: Jnd;
                    while (jnd < Jbound - 7) {
                        __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                        AVX2_DP_2X8(pvec[jnd+0], pvec[jnd+1], (&pvec[ind]), vec_length, dst0, dst1);
                        AVX2_DP_2X8(pvec[jnd+2], pvec[jnd+3], (&pvec[ind]), vec_length, dst2, dst3);
                        AVX2_DP_2X8(pvec[jnd+4], pvec[jnd+5], (&pvec[ind]), vec_length, dst4, dst5);
                        AVX2_DP_2X8(pvec[jnd+6], pvec[jnd+7], (&pvec[ind]), vec_length, dst6, dst7);

                        int cmp2[8], cmp3[8];
                        long *cmp2l = (long *) cmp2;
                        long *cmp3l = (long *) cmp3;
                        cmp2[0] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+0]), dst0), 30));
                        cmp2[1] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+1]), dst1), 30));
                        cmp2[2] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+2]), dst2), 30));
                        cmp2[3] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+3]), dst3), 30));
                        cmp2[4] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+4]), dst4), 30));
                        cmp2[5] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+5]), dst5), 30));
                        cmp2[6] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+6]), dst6), 30));
                        cmp2[7] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+7]), dst7), 30));
                        if (use_3red) {
                            cmp3[0] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+0]), _mm256_broadcast_ss(&pdot[jnd+0])), dst0), 30));
                            cmp3[1] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+1]), _mm256_broadcast_ss(&pdot[jnd+1])), dst1), 30));
                            cmp3[2] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+2]), _mm256_broadcast_ss(&pdot[jnd+2])), dst2), 30));
                            cmp3[3] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+3]), _mm256_broadcast_ss(&pdot[jnd+3])), dst3), 30));
                            cmp3[4] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+4]), _mm256_broadcast_ss(&pdot[jnd+4])), dst4), 30));
                            cmp3[5] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+5]), _mm256_broadcast_ss(&pdot[jnd+5])), dst5), 30));
                            cmp3[6] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+6]), _mm256_broadcast_ss(&pdot[jnd+6])), dst6), 30));
                            cmp3[7] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+7]), _mm256_broadcast_ss(&pdot[jnd+7])), dst7), 30));
                        }

                        for (long i = 0; i < 4; i++){
                            if (cmp2l[i]){
                                CHECK_AND_ADD2_1X8_NU(cmp2[2*i], pvec[jnd+2*i], pvec+ind, TRY_ADDS_TO_DST);
                                CHECK_AND_ADD2_1X8_NU(cmp2[2*i+1], pvec[jnd+2*i+1], pvec+ind, TRY_ADDS_TO_DST);
                            }
                            if (use_3red) {
                                if (cmp3l[i]){
                                    CHECK_AND_ADD2_1X8_NU(cmp3[2*i], pvec[jnd+2*i], pvec+ind, TRY_ADDSS_TO_DST);
                                    CHECK_AND_ADD2_1X8_NU(cmp3[2*i+1], pvec[jnd+2*i+1], pvec+ind, TRY_ADDSS_TO_DST);
                                }
                            }
                        } 
                        jnd += 8;
                    }

                    while (jnd < Jbound) {
                        __m256 dst;
                        AVX2_DP_1X8(pvec[jnd], pvec[ind+0], pvec[ind+1], pvec[ind+2], pvec[ind+3], pvec[ind+4], pvec[ind+5], pvec[ind+6], pvec[ind+7], vec_length, dst);
                        int cmp2, cmp3;
                        cmp2 = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd]), dst), 30));
                        if (use_3red) cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd]), _mm256_broadcast_ss(&pdot[jnd])), dst), 30));
                        CHECK_AND_ADD2_1X8_NU(cmp2, pvec[jnd], pvec+ind, TRY_ADDS_TO_DST);
                        if (use_3red) CHECK_AND_ADD2_1X8_NU(cmp3, pvec[jnd], pvec+ind, TRY_ADDSS_TO_DST);
                        jnd++;
                    }

                    ind += 8;
                }

                if (ind  < Ibound){
                    const long nrem = Ibound - ind;
                    long jnd = (Ind == Jnd) ? ind+8: Jnd;
                    while (jnd < Jbound - 7){
                        __m256 dst[7];
                        int cmp2[8], cmp3[8];
                        long *cmp2l = (long *) cmp2;
                        long *cmp3l = (long *) cmp3;
                        for (long j = 0; j < nrem; j++){
                            AVX2_DP_1X8(pvec[ind+j], pvec[jnd], pvec[jnd+1], pvec[jnd+2], pvec[jnd+3], pvec[jnd+4], pvec[jnd+5], pvec[jnd+6], pvec[jnd+7], vec_length, dst[j]);
                        }
                        __m256 nn, nd;
                        nn = _mm256_loadu_ps(pnorm + jnd);
                        if (use_3red) __m256 nd = _mm256_sub_ps(nn, _mm256_loadu_ps(pdot + jnd));
                        for (long j = 0; j < nrem; j++){
                            cmp2[j] = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(gn, _mm256_broadcast_ss(&pnorm[ind+j])), _mm256_sub_ps(nn, dst[j]), 30));
                            if (use_3red) {
                                __m256 b3 = _mm256_sub_ps(_mm256_add_ps(cn, _mm256_broadcast_ss(&pdot[ind+j])), _mm256_broadcast_ss(&pnorm[ind+j]));
                                cmp3[j] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(nd, dst[j]), 30));
                            }
                        }
                        for (long j = 0; j < nrem; j++){
                            CHECK_AND_ADD2_1X8_NU(cmp2[j], pvec[ind+j], pvec+jnd, TRY_ADDS_TO_DST);
                            if (use_3red) CHECK_AND_ADD2_1X8_NU(cmp3[j], pvec[ind+j], pvec+jnd, TRY_ADDSS_TO_DST);
                        }
                        jnd += 8;
                    }

                    if (jnd < Jbound) {
                        const long inrem = Ibound - ind;
                        const long jnrem = Jbound - jnd;
                        for (long i = 0; i < inrem; i++){
                            for (long j = 0; j < jnrem; j++){
                                float x = dot_avx2(pvec[ind+i], pvec[jnd+j], vec_length);
                                if (-x + pnorm[jnd+j] < 0.5 * goal_norm - pnorm[i+ind]) TRY_ADDS_TO_DST(pvec[ind+i], pvec[jnd+j], *((uint64_t *)(&(pvec[ind+i][-4]))), *((uint64_t *)(&(pvec[jnd+j][-4]))));
                                if (use_3red) {
                                    if (x + pnorm[jnd+j] - pdot[jnd+j] < 0.5 * (goal_norm - center_norm) - pnorm[ind+i]+pdot[ind+i]){
                                        TRY_ADDSS_TO_DST(pvec[ind+i], pvec[jnd+j], *((uint64_t *)(&(pvec[ind+i][-4]))), *((uint64_t *)(&(pvec[jnd+j][-4]))));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return ((num_pvec-8) * num_pvec) / 2;
    }
    int search_nn(bgj_sol_t *sol, UidHashTable *uid, float goal_norm){
        for (long Ind = 0; Ind < num_nvec; Ind += BGJ2_SEARCHING_DP_BLOCK){
            for (long Jnd = Ind; Jnd < num_nvec; Jnd += BGJ2_SEARCHING_DP_BLOCK){
                const long Ibound = (Ind + BGJ2_SEARCHING_DP_BLOCK > num_nvec) ? num_nvec : Ind + BGJ2_SEARCHING_DP_BLOCK;
                const long Jbound = (Jnd + BGJ2_SEARCHING_DP_BLOCK > num_nvec) ? num_nvec : Jnd + BGJ2_SEARCHING_DP_BLOCK;

                __m256 gn, cn;
                gn = _mm256_set1_ps(0.5 * goal_norm);
                if (use_3red) cn = _mm256_set1_ps(0.5 * (goal_norm - center_norm));

                long ind = Ind;
                while (ind < Ibound - 7) {
                    __m256 b2, b3;
                    b2 = _mm256_sub_ps(gn, _mm256_loadu_ps(nnorm + ind));
                    if (use_3red) b3 = _mm256_sub_ps(_mm256_sub_ps(cn, _mm256_loadu_ps(ndot + ind)), _mm256_loadu_ps(nnorm + ind));
                    long jnd = (Ind == Jnd) ? ind + 8: Jnd;
                    while (jnd < Jbound - 7) {
                        __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                        AVX2_DP_2X8(nvec[jnd+0], nvec[jnd+1], (&nvec[ind]), vec_length, dst0, dst1);
                        AVX2_DP_2X8(nvec[jnd+2], nvec[jnd+3], (&nvec[ind]), vec_length, dst2, dst3);
                        AVX2_DP_2X8(nvec[jnd+4], nvec[jnd+5], (&nvec[ind]), vec_length, dst4, dst5);
                        AVX2_DP_2X8(nvec[jnd+6], nvec[jnd+7], (&nvec[ind]), vec_length, dst6, dst7);

                        int cmp2[8], cmp3[8];
                        long *cmp2l = (long *) cmp2;
                        long *cmp3l = (long *) cmp3;

                        cmp2[0] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+0]), dst0), 30));
                        cmp2[1] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+1]), dst1), 30));
                        cmp2[2] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+2]), dst2), 30));
                        cmp2[3] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+3]), dst3), 30));
                        cmp2[4] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+4]), dst4), 30));
                        cmp2[5] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+5]), dst5), 30));
                        cmp2[6] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+6]), dst6), 30));
                        cmp2[7] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+7]), dst7), 30));
                        if (use_3red) {
                            cmp3[0] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+0]), _mm256_broadcast_ss(&ndot[jnd+0])), dst0), 30));
                            cmp3[1] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+1]), _mm256_broadcast_ss(&ndot[jnd+1])), dst1), 30));
                            cmp3[2] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+2]), _mm256_broadcast_ss(&ndot[jnd+2])), dst2), 30));
                            cmp3[3] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+3]), _mm256_broadcast_ss(&ndot[jnd+3])), dst3), 30));
                            cmp3[4] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+4]), _mm256_broadcast_ss(&ndot[jnd+4])), dst4), 30));
                            cmp3[5] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+5]), _mm256_broadcast_ss(&ndot[jnd+5])), dst5), 30));
                            cmp3[6] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+6]), _mm256_broadcast_ss(&ndot[jnd+6])), dst6), 30));
                            cmp3[7] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+7]), _mm256_broadcast_ss(&ndot[jnd+7])), dst7), 30));
                        }
                        for (long i = 0; i < 4; i++){
                            if (cmp2l[i]){
                                CHECK_AND_ADD2_1X8_NU(cmp2[2*i], nvec[jnd+2*i], nvec+ind, TRY_ADDS_TO_DST);
                                CHECK_AND_ADD2_1X8_NU(cmp2[2*i+1], nvec[jnd+2*i+1], nvec+ind, TRY_ADDS_TO_DST);
                            }
                            if (use_3red) {
                                if (cmp3l[i]){
                                    CHECK_AND_ADD2_1X8_NU(cmp3[2*i], nvec[jnd+2*i], nvec+ind, TRY_ADDAA_TO_DST);
                                    CHECK_AND_ADD2_1X8_NU(cmp3[2*i+1], nvec[jnd+2*i+1], nvec+ind, TRY_ADDAA_TO_DST);
                                }
                            }
                        } 
                        jnd += 8;
                    }
                    while (jnd < Jbound) {
                        __m256 dst;
                        AVX2_DP_1X8(nvec[jnd], nvec[ind+0], nvec[ind+1], nvec[ind+2], nvec[ind+3], nvec[ind+4], nvec[ind+5], nvec[ind+6], nvec[ind+7], vec_length, dst);
                        int cmp2, cmp3;
                        cmp2 = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd]), dst), 30));
                        if (use_3red) cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd]), _mm256_broadcast_ss(&ndot[jnd])), dst), 30));
                        CHECK_AND_ADD2_1X8_NU(cmp2, nvec[jnd], nvec+ind, TRY_ADDS_TO_DST);
                        if (use_3red) CHECK_AND_ADD2_1X8_NU(cmp3, nvec[jnd], nvec+ind, TRY_ADDAA_TO_DST);
                        jnd++;
                    }
                    ind += 8;
                }

                if (ind < Ibound) {
                    const long nrem = Ibound - ind;
                    long jnd = (Ind == Jnd) ? ind+8: Jnd;
                    while (jnd < Jbound - 7) {
                        __m256 dst[7];
                        int cmp2[8], cmp3[8];
                        long *cmp2l = (long *) cmp2;
                        long *cmp3l = (long *) cmp3;
                        for (long j = 0; j < nrem; j++){
                            AVX2_DP_1X8(nvec[ind+j], nvec[jnd], nvec[jnd+1], nvec[jnd+2], nvec[jnd+3], nvec[jnd+4], nvec[jnd+5], nvec[jnd+6], nvec[jnd+7], vec_length, dst[j]);
                        }
                        __m256 nn, nd;
                        nn = _mm256_loadu_ps(nnorm + jnd);
                        if (use_3red) nd = _mm256_add_ps(nn, _mm256_loadu_ps(ndot + jnd));
                        for (long j = 0; j < nrem; j++){
                            cmp2[j] = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(gn, _mm256_broadcast_ss(&nnorm[ind+j])), _mm256_sub_ps(nn, dst[j]), 30));
                            if (use_3red) {
                                __m256 b3 = _mm256_sub_ps(_mm256_sub_ps(cn, _mm256_broadcast_ss(&ndot[ind+j])), _mm256_broadcast_ss(&nnorm[ind+j]));
                                cmp3[j] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(nd, dst[j]), 30));
                            }
                        }
                        for (long j = 0; j < nrem; j++){
                            CHECK_AND_ADD2_1X8_NU(cmp2[j], nvec[ind+j], nvec+jnd, TRY_ADDS_TO_DST);
                            if (use_3red) CHECK_AND_ADD2_1X8_NU(cmp3[j], nvec[ind+j], nvec+jnd, TRY_ADDAA_TO_DST);
                        }
                        jnd+=8;
                    }

                    if (jnd < Jbound) {
                        const long inrem = Ibound - ind;
                        const long jnrem = Jbound - jnd;
                        for (long i = 0; i < inrem; i++){
                            for (long j = 0; j < jnrem; j++){
                                float x = dot(nvec[ind+i], nvec[jnd+j], vec_length);
                                if (-x + nnorm[jnd+j] < 0.5*goal_norm - nnorm[i+ind]) TRY_ADDS_TO_DST(nvec[ind+i], nvec[jnd+j], *((uint64_t *)(&(nvec[ind+i][-4]))), *((uint64_t *)(&(nvec[jnd+j][-4]))));
                                if (use_3red) {
                                    if (x + nnorm[jnd+j] + ndot[jnd+j] < 0.5*(goal_norm - center_norm) - nnorm[ind+i]-ndot[ind+i]){
                                        TRY_ADDAA_TO_DST(nvec[ind+i], nvec[jnd+j], *((uint64_t *)(&(nvec[ind+i][-4]))), *((uint64_t *)(&(nvec[jnd+j][-4]))));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return ((num_nvec-8) * num_nvec) / 2;
    }
    int search_np(bgj_sol_t *sol, UidHashTable *uid, float goal_norm){
        for (long Pnd = 0; Pnd < num_pvec; Pnd += BGJ2_SEARCHING_DP_BLOCK){
            for (long Nnd = 0; Nnd < num_nvec; Nnd += BGJ2_SEARCHING_DP_BLOCK){
                const long Pbound = (Pnd + BGJ2_SEARCHING_DP_BLOCK > num_pvec) ? num_pvec : Pnd + BGJ2_SEARCHING_DP_BLOCK;
                const long Nbound = (Nnd + BGJ2_SEARCHING_DP_BLOCK > num_nvec) ? num_nvec : Nnd + BGJ2_SEARCHING_DP_BLOCK;

                __m256 gn, cn;
                gn = _mm256_set1_ps(0.5 * goal_norm);
                if (use_3red) cn = _mm256_set1_ps(0.5 * (goal_norm - center_norm));

                long pnd = Pnd;
                while (pnd < Pbound - 7) {
                    __m256 b2, b3;
                    b2 = _mm256_sub_ps(gn, _mm256_loadu_ps(pnorm + pnd));
                    if (use_3red) b3 = _mm256_sub_ps(_mm256_add_ps(cn, _mm256_loadu_ps(pdot + pnd)), _mm256_loadu_ps(pnorm + pnd));
                    long nnd = Nnd;
                    while (nnd < Nbound - 7) {
                        __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                        AVX2_DP_2X8(nvec[nnd+0], nvec[nnd+1], (&pvec[pnd]), vec_length, dst0, dst1);
                        AVX2_DP_2X8(nvec[nnd+2], nvec[nnd+3], (&pvec[pnd]), vec_length, dst2, dst3);
                        AVX2_DP_2X8(nvec[nnd+4], nvec[nnd+5], (&pvec[pnd]), vec_length, dst4, dst5);
                        AVX2_DP_2X8(nvec[nnd+6], nvec[nnd+7], (&pvec[pnd]), vec_length, dst6, dst7);
                        
                        int cmp2[8], cmp3[8];
                        long *cmp2l = (long *) cmp2;
                        long *cmp3l = (long *) cmp3;
                        cmp2[0] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst0, _mm256_broadcast_ss(&nnorm[nnd+0])), 30));
                        cmp2[1] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst1, _mm256_broadcast_ss(&nnorm[nnd+1])), 30));
                        cmp2[2] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst2, _mm256_broadcast_ss(&nnorm[nnd+2])), 30));
                        cmp2[3] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst3, _mm256_broadcast_ss(&nnorm[nnd+3])), 30));
                        cmp2[4] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst4, _mm256_broadcast_ss(&nnorm[nnd+4])), 30));
                        cmp2[5] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst5, _mm256_broadcast_ss(&nnorm[nnd+5])), 30));
                        cmp2[6] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst6, _mm256_broadcast_ss(&nnorm[nnd+6])), 30));
                        cmp2[7] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst7, _mm256_broadcast_ss(&nnorm[nnd+7])), 30));
                        if (use_3red) {
                            cmp3[0] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+0]), _mm256_broadcast_ss(&ndot[nnd+0])), dst0), 30));
                            cmp3[1] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+1]), _mm256_broadcast_ss(&ndot[nnd+1])), dst1), 30));
                            cmp3[2] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+2]), _mm256_broadcast_ss(&ndot[nnd+2])), dst2), 30));
                            cmp3[3] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+3]), _mm256_broadcast_ss(&ndot[nnd+3])), dst3), 30));
                            cmp3[4] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+4]), _mm256_broadcast_ss(&ndot[nnd+4])), dst4), 30));
                            cmp3[5] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+5]), _mm256_broadcast_ss(&ndot[nnd+5])), dst5), 30));
                            cmp3[6] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+6]), _mm256_broadcast_ss(&ndot[nnd+6])), dst6), 30));
                            cmp3[7] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+7]), _mm256_broadcast_ss(&ndot[nnd+7])), dst7), 30));
                        }
                        for (long i = 0; i < 4; i++){
                            if (cmp2l[i]) {
                                CHECK_AND_ADD2_1X8_NU(cmp2[2*i], nvec[nnd+2*i], pvec+pnd, TRY_ADDA_TO_DST);
                                CHECK_AND_ADD2_1X8_NU(cmp2[2*i+1], nvec[nnd+2*i+1], pvec+pnd, TRY_ADDA_TO_DST);
                            }
                            if (use_3red) {
                                if (cmp3l[i]) {
                                    CHECK_AND_ADD2_1X8_NU(cmp3[2*i], nvec[nnd+2*i], pvec+pnd, TRY_ADDAS_TO_DST);
                                    CHECK_AND_ADD2_1X8_NU(cmp3[2*i+1], nvec[nnd+2*i+1], pvec+pnd, TRY_ADDAS_TO_DST);
                                }
                            }
                        }

                        nnd += 8;
                    }
                    while (nnd < Nbound) {
                        __m256 dst;
                        AVX2_DP_1X8(nvec[nnd], pvec[pnd+0], pvec[pnd+1], pvec[pnd+2], pvec[pnd+3], pvec[pnd+4], pvec[pnd+5], pvec[pnd+6], pvec[pnd+7], vec_length, dst);
                        int cmp2, cmp3;
                        cmp2 = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst, _mm256_broadcast_ss(&nnorm[nnd])), 30));
                        if (use_3red) cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd]), _mm256_broadcast_ss(&ndot[nnd])), dst), 30));
                        CHECK_AND_ADD2_1X8_NU(cmp2, nvec[nnd], pvec+pnd, TRY_ADDA_TO_DST);
                        if (use_3red) CHECK_AND_ADD2_1X8_NU(cmp3, nvec[nnd], pvec+pnd, TRY_ADDAS_TO_DST);
                        nnd++;
                    }
                    pnd += 8;
                }

                if (pnd < Pbound) {
                    const long nrem = Pbound - pnd;
                    long nnd = Nnd;
                    while (nnd < Nbound - 7) {
                        __m256 dst[7];
                        int cmp2[8], cmp3[8];
                        long *cmp2l = (long *) cmp2;
                        long *cmp3l = (long *) cmp3;
                        for (long j = 0; j < nrem; j++){
                            AVX2_DP_1X8(pvec[pnd+j], nvec[nnd], nvec[nnd+1], nvec[nnd+2], nvec[nnd+3], nvec[nnd+4], nvec[nnd+5], nvec[nnd+6], nvec[nnd+7], vec_length, dst[j]);
                        }
                        __m256 nn, nd;
                        nn = _mm256_loadu_ps(nnorm + nnd);
                        if (use_3red) nd = _mm256_add_ps(nn, _mm256_loadu_ps(ndot + nnd));
                        for (long j = 0; j < nrem; j++){
                            cmp2[j] = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(gn, _mm256_broadcast_ss(&pnorm[pnd+j])), _mm256_add_ps(dst[j], nn), 30));
                            if (use_3red) {
                                __m256 b3 = _mm256_sub_ps(_mm256_add_ps(cn, _mm256_broadcast_ss(&pdot[pnd+j])), _mm256_broadcast_ss(&pnorm[pnd+j]));
                                cmp3[j] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(nd, dst[j]), 30));
                            }
                        }
                        for (long j = 0; j < nrem; j++){
                            CHECK_AND_ADD2_1X8_NU(cmp2[j], pvec[pnd+j], nvec+nnd, TRY_ADDA_TO_DST);
                            if (use_3red) CHECK_AND_ADD2_1X8_NU(cmp3[j], pvec[pnd+j], nvec+nnd, TRY_ADDSA_TO_DST);
                        }
                        nnd+=8;
                    }

                    if (nnd < Nbound) {
                        const long pnrem = Pbound - pnd;
                        const long nnrem = Nbound - nnd;
                        for (long j = 0; j < pnrem; j++){
                            for (long i = 0; i < nnrem; i++){
                                float x = dot_avx2(pvec[pnd+j], nvec[nnd+i], vec_length);
                                if (x + nnorm[nnd+i] < 0.5*goal_norm - pnorm[j+pnd]) TRY_ADDA_TO_DST(pvec[pnd+j], nvec[nnd+i], *((uint64_t *)(&(pvec[pnd+j][-4]))), *((uint64_t *)(&(nvec[nnd+i][-4]))));
                                if (use_3red) {
                                    if (-x + nnorm[nnd+i] + ndot[nnd+i] < 0.5 * (goal_norm - center_norm) - pnorm[pnd+j]+pdot[pnd+j]){
                                        TRY_ADDSA_TO_DST(pvec[pnd+j], nvec[nnd+i], *((uint64_t *)(&(pvec[pnd+j][-4]))), *((uint64_t *)(&(nvec[nnd+i][-4]))));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return num_pvec * num_nvec;
    }


    int _clear(){
        if (pvec) free(pvec);
        if (nvec) free(nvec);
        if (use_3red && pdot) free(pdot);
        if (use_3red && ndot) free(ndot);
        if (pnorm) free(pnorm);
        if (nnorm) free(nnorm);
        //if (pu) free(pu);
        //if (nu) free(nu);
        //pu = NULL; nu = NULL;
        pnorm = NULL; nnorm = NULL;
        pvec = NULL; nvec = NULL;
        if (use_3red) { pdot = NULL; ndot = NULL; }
        return 0;
    }
    int _alloc(long size, bool p){
        if (p){
            if (size <= _psize) return 0;
            pvec = (float **) realloc(pvec, size * sizeof(float *));
            pnorm = (float *) realloc(pnorm, size * sizeof(float));
            if (use_3red) pdot = (float *) realloc(pdot, size * sizeof(float));
            _psize = size;
            return (pnorm && pvec && (!use_3red || pdot));
        } else {
            if (size <= _nsize) return 0;
            nvec = (float **) realloc(nvec, size * sizeof(float *));
            nnorm = (float *) realloc(nnorm, size * sizeof(float));
            if (use_3red) ndot = (float *) realloc(ndot, size * sizeof(float));
            _nsize = size;
            return (nnorm && nvec && (!use_3red || ndot));
        }
    }

    long _psize = 0;
    long _nsize = 0;
};

struct bgj_mbucket_t {
    /**
     * the variables and functions begin with "_" only used in bucket 
     * collecting stage, and should not be called after normalize()
    */
    long _num_pvec = 0;
    long _num_nvec = 0;
    uint32_t *_pvec = NULL;         // so we cannot sieve with more than (4/3)^(154/2) vectors!!!
    uint32_t *_nvec = NULL;         // I think this is not a problem.
    float *_pdot = NULL;
    float *_ndot = NULL;
    
    float *vec;
    long vec_length, vec_size;

    float *center = NULL;
    uint32_t center_ind;

    long try_add2 = 0;
    long succ_add2 = 0;

    long nvec = 0;

    bgj_mbucket_t() {}
    bgj_mbucket_t(long size) { _alloc(size, 0); _alloc(size, 1); }
    
    void _set_center(float *ptr, uint32_t ind){
        if (center == NULL) {
            center = ptr;
            center_ind = ind;
        }
    }
    inline void _add_vec(uint32_t ind, float _dot) {
        if (_dot > 0){
            if (_num_pvec == _psize) _alloc(_psize * 2 + 64, 1);
            _pvec[_num_pvec] = ind;
            _pdot[_num_pvec] = _dot;
            _num_pvec++;
        } else {
            if (_num_nvec == _nsize) _alloc(_nsize * 2 + 64, 0);
            _nvec[_num_nvec] = ind;
            _ndot[_num_nvec] = _dot;
            _num_nvec++;
        }
    }
    inline void _add_pvec(uint32_t ind, float _dot) {
        if (_num_pvec == _psize) _alloc(_psize * 2 + 64, 1);
        _pvec[_num_pvec] = ind;
        _pdot[_num_pvec] = _dot;
        _num_pvec++;
    }
    inline void _add_nvec(uint32_t ind, float _dot) {
        if (_num_nvec == _nsize) _alloc(_nsize * 2 + 64, 0);
        _nvec[_num_nvec] = ind;
        _ndot[_num_nvec] = _dot;
        _num_nvec++;
    }
    void _combine(bgj_mbucket_t **subbucket_list, long len){
        if (_num_pvec || _num_nvec) {
            fprintf(stderr, "[Error] bgj_mbucket_t::combine: combining subbuckets to nonempty buckets, aborted.\n");
            return;
        }
        long total_psize = 0, total_nsize = 0;
        for (long i = 0; i < len; i++){
            total_psize += subbucket_list[i]->_num_pvec;
            total_nsize += subbucket_list[i]->_num_nvec;   
        }
        _alloc(total_psize, 1);
        _alloc(total_nsize, 0);
        for (long i = 0; i < len; i++){
            memcpy(_pvec + _num_pvec, subbucket_list[i]->_pvec, subbucket_list[i]->_num_pvec * sizeof(uint32_t));
            memcpy(_nvec + _num_nvec, subbucket_list[i]->_nvec, subbucket_list[i]->_num_nvec * sizeof(uint32_t));
            memcpy(_pdot + _num_pvec, subbucket_list[i]->_pdot, subbucket_list[i]->_num_pvec * sizeof(float));
            memcpy(_ndot + _num_nvec, subbucket_list[i]->_ndot, subbucket_list[i]->_num_nvec * sizeof(float));
            _num_pvec += subbucket_list[i]->_num_pvec;
            _num_nvec += subbucket_list[i]->_num_nvec;
        }
    }

    // dst3 and dst2 need not to be allocated
    #if BGJ2_BUCKET1_USE_NONLATTICE_CENTER
    void bucket1(double alpha2, double alpha3, bgj_sbucket_t<1> **dst3, bgj_sbucket_t<0> **dst2) {
    #else
    void bucket1(double alpha2, double alpha3, bgj_sbucket_t<1> **dst3, bgj_sbucket_t<0> **dst2, UidHashTable *uid, bgj_sol_t *sol, float goal_norm){
    #endif
        /* remove the center from bucket */
        do {
            long low = 0;
            long high = _num_pvec-1;
            do {
                long mid = (low+high)/2;
                if (_pvec[mid] < center_ind){
                    low = mid;
                } else {
                    high = mid;
                }
            } while (high - low > 2);
            for (long i = low; i <= high; i++){
                if (_pvec[i] == center_ind) {
                    _num_pvec--;
                    _pvec[i] = _pvec[_num_pvec];
                    _pdot[i] = _pdot[_num_pvec];
                }
            }
        } while (0);

        nvec = _num_pvec + _num_nvec;

        dst3[0] = new bgj_sbucket_t<1>;
        for (long i = 0; i < BGJ2_BUCKET1_BATCHSIZE; i++) dst2[i] = new bgj_sbucket_t<0>;
        
        /* prepare the centers */
        dst3[0]->vec_length = vec_length;
        dst3[0]->set_center(center);
        long center1_ind[BGJ2_BUCKET1_BATCHSIZE];
        float **center1 = (float **) NEW_MAT(BGJ2_BUCKET1_BATCHSIZE, vec_length, sizeof(float));
        long count = 0;
        #if BGJ2_BUCKET1_USE_NONLATTICE_CENTER
        for (long i = 0; i < BGJ2_BUCKET1_BATCHSIZE; i++) {
            int pass;
            do {
                count++;
                pass = 1;
                center1_ind[i] = Uniform_long(_num_pvec);     // we always select positive centers!!!
                for (long j = 0; j < i; j++){
                    //if ((vec + vec_size * _pvec[center1_ind[i]])[-1] > 1.05 * goal_norm) pass = 0;
                    if (center1_ind[j] == center1_ind[i]) pass = 0;
                }
            } while(!pass && (count < 500));

            float *ptr = vec + vec_size * _pvec[center1_ind[i]];
            copy(center1[i], ptr, vec_length);

            float AA = (1-BGJ2_BUCKET1_CCDP*BGJ2_BUCKET1_CCDP) * center[-1] * center[-1];
            float BB = 2.0 * center[-1] * _pdot[center1_ind[i]] * (1-BGJ2_BUCKET1_CCDP*BGJ2_BUCKET1_CCDP);
            float CC = _pdot[center1_ind[i]] * _pdot[center1_ind[i]] - BGJ2_BUCKET1_CCDP*BGJ2_BUCKET1_CCDP * center[-1] * ptr[-1];
            float lambda = (-BB + sqrt(BB * BB - 4 * AA * CC))/(2 * AA);
            red(center1[i], center, -lambda, vec_length);
            float scal = sqrt(ptr[-1]/dot_avx2(center1[i], center1[i], vec_length));
            mul(center1[i], scal, vec_length);

            dst2[i]->vec_length = vec_length;
            dst2[i]->set_center(center1[i]);
        }
        #else
        float center1_norm[BGJ2_BUCKET1_BATCHSIZE];
        float *center1_ptr[BGJ2_BUCKET1_BATCHSIZE];
        uint64_t center1_u[BGJ2_BUCKET1_BATCHSIZE];
        for (long i = 0; i < BGJ2_BUCKET1_BATCHSIZE; i++){
            int pass;
            do {
                count++;
                pass = 1;
                center1_ind[i] = Uniform_long(_num_pvec);     // we always select positive centers!!!
                for (long j = 0; j < i; j++){
                    //if ((vec + vec_size * _pvec[center1_ind[i]])[-1] > 1.05 * goal_norm) pass = 0;
                    if (center1_ind[j] == center1_ind[i]) pass = 0;
                }
            } while(!pass && (count < 500));

            float *ptr = vec + vec_size * _pvec[center1_ind[i]];
            dst2[i]->vec_length = vec_length;
            dst2[i]->set_center(ptr);
            copy(center1[i], ptr, vec_length);
            center1_ptr[i] = ptr;
            center1_norm[i] = goal_norm - ptr[-1];
            center1_u[i] = *((uint64_t *)(&ptr[-4]));
        }
        for (long i = 0; i < BGJ2_BUCKET1_BATCHSIZE; i++) {
            center1_ind[i] = _pvec[center1_ind[i]];
        }
        #endif
        

        __m256 alpha2x8 = _mm256_set1_ps(alpha2);
        __m256 alpha3x8 = _mm256_set1_ps(alpha3);
        __m256 sign_bit = _mm256_set1_ps(-0.0f);

        const long expect_bucket1_size = pow(nvec, 0.6);
        long _cpn_size = expect_bucket1_size * 0.7 + 8 * BGJ2_BUCKET1_BATCHSIZE;
        long num_cpn = 0;
        uint32_t *cpn_buffer = (uint32_t *) malloc(_cpn_size * 3 * sizeof(uint32_t));

        /* main loop */
        do {
            long ind = 0;
            while (ind + 7 < _num_pvec) {
                float *ptr[8];
                ptr[0] = vec + vec_size * _pvec[ind+0];
                ptr[1] = vec + vec_size * _pvec[ind+1];
                ptr[2] = vec + vec_size * _pvec[ind+2];
                ptr[3] = vec + vec_size * _pvec[ind+3];
                ptr[4] = vec + vec_size * _pvec[ind+4];
                ptr[5] = vec + vec_size * _pvec[ind+5];
                ptr[6] = vec + vec_size * _pvec[ind+6];
                ptr[7] = vec + vec_size * _pvec[ind+7];

                __attribute__ ((aligned (32))) float nptr[8];
                __attribute__ ((aligned (32))) float dst[8 * BGJ2_BUCKET1_BATCHSIZE];
                __attribute__ ((aligned (32))) uint32_t cdst[BGJ2_BUCKET1_BATCHSIZE >> 2];
                __attribute__ ((aligned (32))) uint32_t cmp[8];
                nptr[0] = ptr[0][-1];
                nptr[1] = ptr[1][-1];
                nptr[2] = ptr[2][-1];
                nptr[3] = ptr[3][-1];
                nptr[4] = ptr[4][-1];
                nptr[5] = ptr[5][-1];
                nptr[6] = ptr[6][-1];
                nptr[7] = ptr[7][-1];

                __m128 nptr_half0 = _mm_load_ps(&nptr[0]);
                __m128 nptr_half1 = _mm_load_ps(&nptr[4]);
                __m256 anptr0 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(nptr_half0), nptr_half0, 1), alpha2x8);
                __m256 anptr1 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(nptr_half1), nptr_half1, 1), alpha2x8);

                // directly check and add to *dst3
                int cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(_pdot+ind), _mm256_mul_ps(alpha3x8, _mm256_load_ps(nptr)), 30));
                while (cmp3){
                    long r = __builtin_ctz(cmp3);
                    cmp3 -= (1 << r);
                    dst3[0]->add_pvec(ptr[r], nptr[r], &_pdot[ind+r]);
                }

                unsigned cind = 0;                
                while (cind + 7 < BGJ2_BUCKET1_BATCHSIZE){
                    AVX2_DP_CMP_RAW_4X4(center1[cind], center1[cind+1], center1[cind+2], center1[cind+3], ptr[0], ptr[1], ptr[2], ptr[3], 0, vec_length, anptr0, (&dst[cind*4]), cmp[0], cmp[1]);                   
                    AVX2_DP_CMP_RAW_4X4(center1[cind], center1[cind+1], center1[cind+2], center1[cind+3], ptr[4], ptr[5], ptr[6], ptr[7], 0, vec_length, anptr1, (&dst[cind*4+BGJ2_BUCKET1_BATCHSIZE*4]), cmp[2], cmp[3]);                   
                    AVX2_DP_CMP_RAW_4X4(center1[cind+4], center1[cind+5], center1[cind+6], center1[cind+7], ptr[0], ptr[1], ptr[2], ptr[3], 0, vec_length, anptr0, (&dst[cind*4+16]), cmp[4], cmp[5]);                   
                    AVX2_DP_CMP_RAW_4X4(center1[cind+4], center1[cind+5], center1[cind+6], center1[cind+7], ptr[4], ptr[5], ptr[6], ptr[7], 0, vec_length, anptr1, (&dst[cind*4+16+BGJ2_BUCKET1_BATCHSIZE*4]), cmp[6], cmp[7]);                   
                    cdst[cind >> 3] = ( cmp[0] | (cmp[1] << 8) | (cmp[4] << 16) | (cmp[5] << 24) );
                    cdst[(cind + BGJ2_BUCKET1_BATCHSIZE) >> 3] = ( cmp[2] | (cmp[3] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
                    cind += 8;
                }

                #define ADD_TO_CPN_BUFFER(__cind, __pind, __norm) do {                          \
                    cpn_buffer[3*num_cpn] = __cind;                                             \
                    cpn_buffer[3*num_cpn+1] = __pind;                                           \
                    *((float *)(&cpn_buffer[3*num_cpn+2])) = __norm;                            \
                    num_cpn++;                                                                  \
                } while (0);

                long idx = 0;
                for (long i = 0; i < 8; i += 4) {
                    for (long j = 0; j < BGJ2_BUCKET1_BATCHSIZE; j += 8){
                        while (cdst[idx]){
                            uint32_t r = __builtin_ctz(cdst[idx]);
                            cdst[idx] -= (1 << r);
                            int tt = (r&0x3)+i;
                            #if BGJ2_BUCKET1_USE_NONLATTICE_CENTER
                            ADD_TO_CPN_BUFFER(j+(r>>2), _pvec[ind+tt], nptr[tt]);
                            #else
                            if (_pvec[ind+tt] == center1_ind[j+(r>>2)]) continue;
                            ADD_TO_CPN_BUFFER(j+(r>>2), _pvec[ind+tt], nptr[tt]);
                            if (nptr[tt] - 2 * dst[i*BGJ2_BUCKET1_BATCHSIZE+j*4+r] < center1_norm[j+(r>>2)]){
                                float *ptr = (vec + vec_size * _pvec[ind+tt]);
                                TRY_ADDS_TO_DST(center1_ptr[j+(r>>2)], ptr, center1_u[j+(r>>2)], *((uint64_t *)(&ptr[-4])));
                            }
                            #endif
                        }
                        idx++;
                    }
                }
                
                if (num_cpn + 8 * BGJ2_BUCKET1_BATCHSIZE >= _cpn_size) {
                    _cpn_size *= 2;
                    cpn_buffer = (uint32_t *) realloc(cpn_buffer, _cpn_size * 3 * sizeof(uint32_t));
                }
                
                ind += 8;
            }

            for (long i = 0; i < num_cpn; i++){
                uint32_t cind = cpn_buffer[3*i];
                uint32_t pind = cpn_buffer[3*i+1];
                float _norm = *((float *)(&cpn_buffer[3*i+2]));
                dst2[cind]->add_pvec(vec + vec_size * pind, _norm);
            }
            

            while (ind < _num_pvec) {
                float *ptr = vec + vec_size * _pvec[ind];
                if (_pdot[ind] > alpha3 * ptr[-1]){
                    dst3[0]->add_pvec(ptr, ptr[-1], &_pdot[ind]);
                }
                long cind = 0;
                float alpha_nptr = ptr[-1] * alpha2;

                while (cind + 3 < BGJ2_BUCKET1_BATCHSIZE){
                    float __dst0, __dst1, __dst2, __dst3;
                    AVX2_DOT_PRODUCT_1X4(ptr, center1[cind], center1[cind+1], center1[cind+2], center1[cind+3], vec_length);
                    #if BGJ2_BUCKET1_USE_NONLATTICE_CENTER
                    if (__dst0 > alpha_nptr) dst2[cind+0]->add_pvec(ptr, ptr[-1]);
                    if (__dst1 > alpha_nptr) dst2[cind+1]->add_pvec(ptr, ptr[-1]);
                    if (__dst2 > alpha_nptr) dst2[cind+2]->add_pvec(ptr, ptr[-1]);
                    if (__dst3 > alpha_nptr) dst2[cind+3]->add_pvec(ptr, ptr[-1]);
                    #else
                    if (__dst0 > alpha_nptr) {
                        if (_pvec[ind] != center1_ind[cind+0]) dst2[cind+0]->add_pvec(ptr, ptr[-1]);
                    }
                    if (__dst1 > alpha_nptr) {
                        if (_pvec[ind] != center1_ind[cind+1]) dst2[cind+1]->add_pvec(ptr, ptr[-1]);
                    }
                    if (__dst2 > alpha_nptr) {
                        if (_pvec[ind] != center1_ind[cind+2]) dst2[cind+2]->add_pvec(ptr, ptr[-1]);
                    }
                    if (__dst3 > alpha_nptr) {
                        if (_pvec[ind] != center1_ind[cind+3]) dst2[cind+3]->add_pvec(ptr, ptr[-1]);
                    }
                    #endif
                    cind += 4;
                }
                ind++;
            }

            ind = 0;
            num_cpn = 0;

            while (ind + 7 < _num_nvec) {
                float *ptr[8];
                ptr[0] = vec + vec_size * _nvec[ind+0];
                ptr[1] = vec + vec_size * _nvec[ind+1];
                ptr[2] = vec + vec_size * _nvec[ind+2];
                ptr[3] = vec + vec_size * _nvec[ind+3];
                ptr[4] = vec + vec_size * _nvec[ind+4];
                ptr[5] = vec + vec_size * _nvec[ind+5];
                ptr[6] = vec + vec_size * _nvec[ind+6];
                ptr[7] = vec + vec_size * _nvec[ind+7];

                __attribute__ ((aligned (32))) float nptr[8];
                __attribute__ ((aligned (32))) float dst[8 * BGJ2_BUCKET1_BATCHSIZE];
                __attribute__ ((aligned (32))) uint32_t cdst[BGJ2_BUCKET1_BATCHSIZE >> 2];
                __attribute__ ((aligned (32))) uint32_t cmp[8];
                nptr[0] = ptr[0][-1];
                nptr[1] = ptr[1][-1];
                nptr[2] = ptr[2][-1];
                nptr[3] = ptr[3][-1];
                nptr[4] = ptr[4][-1];
                nptr[5] = ptr[5][-1];
                nptr[6] = ptr[6][-1];
                nptr[7] = ptr[7][-1];

                __m128 nptr_half0 = _mm_load_ps(&nptr[0]);
                __m128 nptr_half1 = _mm_load_ps(&nptr[4]);
                __m256 anptr0 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(nptr_half0), nptr_half0, 1), alpha2x8);
                __m256 anptr1 = _mm256_mul_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(nptr_half1), nptr_half1, 1), alpha2x8);

                // directly check and add to *dst3
                int cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_loadu_ps(_ndot+ind)), _mm256_mul_ps(alpha3x8, _mm256_load_ps(nptr)), 30));
                while (cmp3){
                    long r = __builtin_ctz(cmp3);
                    cmp3 -= (1 << r);
                    dst3[0]->add_nvec(ptr[r], nptr[r], &_ndot[ind+r]);
                }

                unsigned cind = 0;
                while (cind + 7 < BGJ2_BUCKET1_BATCHSIZE){
                    AVX2_DP_CMP_RAW_4X4(center1[cind], center1[cind+1], center1[cind+2], center1[cind+3], ptr[0], ptr[1], ptr[2], ptr[3], 1, vec_length, anptr0, (&dst[cind*4]), cmp[0], cmp[1]);                   
                    AVX2_DP_CMP_RAW_4X4(center1[cind], center1[cind+1], center1[cind+2], center1[cind+3], ptr[4], ptr[5], ptr[6], ptr[7], 1, vec_length, anptr1, (&dst[cind*4+BGJ2_BUCKET1_BATCHSIZE*4]), cmp[2], cmp[3]);                   
                    AVX2_DP_CMP_RAW_4X4(center1[cind+4], center1[cind+5], center1[cind+6], center1[cind+7], ptr[0], ptr[1], ptr[2], ptr[3], 1, vec_length, anptr0, (&dst[cind*4+16]), cmp[4], cmp[5]);                   
                    AVX2_DP_CMP_RAW_4X4(center1[cind+4], center1[cind+5], center1[cind+6], center1[cind+7], ptr[4], ptr[5], ptr[6], ptr[7], 1, vec_length, anptr1, (&dst[cind*4+16+BGJ2_BUCKET1_BATCHSIZE*4]), cmp[6], cmp[7]);                                     
                    cdst[cind >> 3] = ( cmp[0] | (cmp[1] << 8) | (cmp[4] << 16) | (cmp[5] << 24) );
                    cdst[(cind + BGJ2_BUCKET1_BATCHSIZE) >> 3] = ( cmp[2] | (cmp[3] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
                    cind += 8;
                }

                long idx = 0;
                for (long i = 0; i < 8; i += 4) {
                    for (long j = 0; j < BGJ2_BUCKET1_BATCHSIZE; j += 8){
                        while (cdst[idx]){
                            uint32_t r = __builtin_ctz(cdst[idx]);
                            cdst[idx] -= (1 << r);
                            int tt = (r&0x3)+i;
                            ADD_TO_CPN_BUFFER(j+(r>>2), _nvec[ind+tt], nptr[tt]);
                            #if !BGJ2_BUCKET1_USE_NONLATTICE_CENTER
                            if (nptr[tt] + 2 * dst[i*BGJ2_BUCKET1_BATCHSIZE+j*4+r] < center1_norm[j+(r>>2)]){
                                float *ptr = (vec + vec_size * _nvec[ind+tt]);
                                TRY_ADDA_TO_DST(center1_ptr[j+(r>>2)], ptr, center1_u[j+(r>>2)], *((uint64_t *)(&ptr[-4])));
                            }
                            #endif
                        }
                        idx++;
                    }
                }
                
                if (num_cpn + 8 * BGJ2_BUCKET1_BATCHSIZE >= _cpn_size) {
                    _cpn_size *= 2;
                    cpn_buffer = (uint32_t *) realloc(cpn_buffer, _cpn_size * 3 * sizeof(uint32_t));
                }
                
                ind += 8;
            }

            for (long i = 0; i < num_cpn; i++){
                uint32_t cind = cpn_buffer[3*i];
                uint32_t pind = cpn_buffer[3*i+1];
                float _norm = *((float *)(&cpn_buffer[3*i+2]));
                dst2[cind]->add_nvec(vec + vec_size * pind, _norm);
            }

            while (ind < _num_nvec) {
                float *ptr = vec + vec_size * _nvec[ind];
                if (-_ndot[ind] > alpha3 * ptr[-1]){
                    dst3[0]->add_nvec(ptr, ptr[-1], &_ndot[ind]);
                }
                long cind = 0;
                float alpha_nptr = ptr[-1] * alpha2;
                while (cind + 3 < BGJ2_BUCKET1_BATCHSIZE){
                    float __dst0, __dst1, __dst2, __dst3;
                    AVX2_DOT_PRODUCT_1X4(ptr, center1[cind], center1[cind+1], center1[cind+2], center1[cind+3], vec_length);
                    if (-__dst0 > alpha_nptr) dst2[cind+0]->add_nvec(ptr, ptr[-1]);
                    if (-__dst1 > alpha_nptr) dst2[cind+1]->add_nvec(ptr, ptr[-1]);
                    if (-__dst2 > alpha_nptr) dst2[cind+2]->add_nvec(ptr, ptr[-1]);
                    if (-__dst3 > alpha_nptr) dst2[cind+3]->add_nvec(ptr, ptr[-1]);
                    cind += 4;
                }
                ind++;
            }
        } while (0);

        for (long i = 0; i < BGJ2_BUCKET1_BATCHSIZE; i++) {
            for (long j = 0; j < i; j++) {
                if (center1_ind[i] == center1_ind[j]) {
                    dst2[i]->_clear();
                    delete dst2[i];
                    dst2[i] = NULL;
                    break;
                }
            }
        }

        /* free */
        free(cpn_buffer);
        FREE_MAT(center1);
        if (_pvec) free(_pvec);
        if (_nvec) free(_nvec);
        if (_pdot) free(_pdot);
        if (_ndot) free(_ndot);
        _pvec = NULL; _nvec = NULL;
        _pdot = NULL; _ndot = NULL;
    }
    ~bgj_mbucket_t(){ clear(); }


    int clear(){
        if (_pvec) free(_pvec);
        if (_nvec) free(_nvec);
        if (_pdot) free(_pdot);
        if (_ndot) free(_ndot);
        _pvec = NULL; _nvec = NULL;
        _pdot = NULL; _ndot = NULL;
        return 0;
    }
    int _alloc(long size, bool p){
        if (p){
            if (size <= _psize) return 0;
            _pvec = (uint32_t *) realloc(_pvec, size * sizeof(uint32_t));
            _pdot = (float *) realloc(_pdot, size * sizeof(float));
            _psize = size;
            return (_pvec && _pdot);
        } else {
            if (size <= _nsize) return 0;
            _nvec = (uint32_t *) realloc(_nvec, size * sizeof(uint32_t));
            _ndot = (float *) realloc(_ndot, size * sizeof(float));
            _nsize = size;
            return (_nvec && _ndot);
        }
    }
    long _psize = 0;
    long _nsize = 0;
};



int Pool::bgj2_Sieve(long log_level, long lps_auto_adj, long num_empty){
#if 1
    pthread_spinlock_t debug_lock;
    pthread_spin_init(&debug_lock, PTHREAD_PROCESS_SHARED);
    struct timeval bgj2_start_time, bgj2_end_time;
    gettimeofday(&bgj2_start_time, NULL);
    long num_epoch = 0;
    long num_bucket0 = 0;
    long num_bucket1 = 0;
    long sum_bucket0_size = 0;
    long sum_bucket1_size = 0;
    double bucket0_time = 0.0;
    double bucket1_time = 0.0;
    double search0_time = 0.0;
    double search1_time = 0.0;
    double sort_time = 0.0;
    double insert_time = 0.0;
    double combine_time = 0.0;
    uint64_t bucket0_ndp = 0;
    uint64_t bucket1_ndp = 0;
    uint64_t search0_ndp = 0;
    uint64_t search1_ndp = 0;
    uint64_t try_add2 = 0;
    uint64_t succ_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add3 = 0;
#endif

    /* params */
    double alpha_r0 = BGJ2_BUCKET0_REUSE_ALPHA;
    double alpha_b0 = BGJ2_BUCKET0_ALPHA;
    double alpha_b1 = BGJ2_BUCKET1_ALPHA;
    const double saturation_radius = 4.0/3.0;
    const double saturation_ratio = 0.43;
    const double one_epoch_ratio = 0.025;
    const double improve_ratio = 0.73;
    const double resort_ratio = 0.95;

    const long expect_bucket0_size = pow(1.0 - alpha_b0 * alpha_b0, CSD * 0.5) * num_vec;
    const long expect_bucket1_size = pow(1.0 - alpha_b1 * alpha_b1, CSD * 0.5) * expect_bucket0_size;

    /* sort the pool */
    TIMER_START;
    sort_cvec();
    TIMER_END;
    sort_time += CURRENT_TIME;

    /* global buffers */
    long global_bucket0_buffer_size[MAX_NTHREADS];
    uint32_t *global_bucket0_cdp_buffer[MAX_NTHREADS];
    for (long i = 0; i < num_threads; i++){
        global_bucket0_buffer_size[i] = expect_bucket0_size/num_threads * BGJ2_BUCKET0_BATCHSIZE;
        global_bucket0_cdp_buffer[i] = (uint32_t *) malloc(global_bucket0_buffer_size[i] * 3 * sizeof(uint32_t));
    }
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
    if (log_level == 0 && CSD > MIN_LOG_CSD){
        fprintf(stderr, "begin bgj2 sieve on context [%ld, %ld], gh = %.2f, pool size = %ld, %ld threads will be used",
                        index_l, index_r, sqrt(gh2), num_vec, num_threads);
    }
    if (log_level >= 1) {
        dlog("begin bgj2 sieve, sieving dimension = %ld, pool size = %ld", CSD, num_vec);
        if (log_level >= 3 && CSD > MIN_LOG_CSD){
            dlog("first bucket alpha = %f, batchsize = %d", BGJ2_BUCKET0_ALPHA, BGJ2_BUCKET0_BATCHSIZE);
            dlog("second bucket alpha = %f, batchsize = %d", BGJ2_BUCKET1_ALPHA, BGJ2_BUCKET1_BATCHSIZE);
            dlog("first bucket reuse = %d", BGJ2_BUCKET0_REUSE);
            if (BGJ2_BUCKET0_REUSE) dlog("first bucket reuse alpha = %f", BGJ2_BUCKET0_REUSE_ALPHA);
        }
    }

    long sieving_stucked = 0;
    /* main sieving procedure */
    while (!sieve_is_over(saturation_radius, saturation_ratio) && !sieving_stucked){
        if (lps_auto_adj) {
            long adj_row = (CSD-1)/10 - 6;
            if (adj_row < 0) adj_row = 0;
            if (adj_row > 3) adj_row = 3;
            long adj_col = (num_epoch > 11) ? 11 : num_epoch;
            alpha_r0 = BGJ2_BUCKET0_REUSE_ALPHA + bgj2_lps_alphar_adj[adj_row][adj_col];
            alpha_b0 = BGJ2_BUCKET0_ALPHA + bgj2_lps_alpha0_adj[adj_row][adj_col];
            alpha_b1 = BGJ2_BUCKET1_ALPHA + bgj2_lps_alpha1_adj[adj_row][adj_col];
        }
        num_epoch++;
        const long goal_index = (long)(improve_ratio * num_vec);
        const float goal_norm = ((float *)(cvec+goal_index*cvec_size+4))[0];
        if (log_level >= 2 && CSD > MIN_LOG_CSD) dlog("epoch %ld, goal_norm = %.2f", num_epoch-1, sqrt(goal_norm));

        #if 1
        long _num_bucket0 = 0;
        long _num_bucket1 = 0;
        long _sum_bucket0_size = 0;
        long _sum_bucket1_size = 0;
        double _bucket0_time = 0.0;
        double _bucket1_time = 0.0;
        double _search0_time = 0.0;
        double _search1_time = 0.0;
        uint64_t _bucket0_ndp = 0;
        uint64_t _bucket1_ndp = 0;
        uint64_t _search0_ndp = 0;
        uint64_t _search1_ndp = 0;
        uint64_t _try_add2 = 0;
        uint64_t _succ_add2 = 0;
        uint64_t _try_add3 = 0;
        uint64_t _succ_add3 = 0;
        #endif


        /* collect solutions */
        long stucktime = 0;
        long last_num_total_sol = 0;
        long num_total_sol = 0;
        bool rel_collection_stop = false;
        bgj_sol_t *local_buffer[MAX_NTHREADS];
        for (long i = 0; i < num_threads; i++) local_buffer[i] = new bgj_sol_t(2000);
        do {            
            bgj_mbucket_t main_bucket[BGJ2_BUCKET0_BATCHSIZE];
            bgj_mbucket_t **local_mbucket = new bgj_mbucket_t*[num_threads];

            /* choose centers */
            uint32_t ind[BGJ2_BUCKET0_BATCHSIZE];
            float **center = (float **) NEW_MAT(BGJ2_BUCKET0_BATCHSIZE, vec_length, sizeof(float));
            long num_try_find_center = 0;
            for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++){
                int pass;
                float *ptr;
                do {
                    pass = 1;
                    ind[i] = Uniform_long(goal_index);
                    for (long j = 0; j < i; j++){
                        if (ind[j] == ind[i]) pass = 0;
                    }
                    ptr = vec + ind[i] * vec_size;
                    num_try_find_center++;
                    #if BGJ2_BUCKET0_USE_FARAWAY_CENTER
                    if ((CSD * ptr[CSD-1] * ptr[CSD-1] > ptr[-1]) && (num_try_find_center < 500)) pass = 0;
                    #endif
                } while(!pass);

                //float *ptr = vec + ind[i] * vec_size;
                main_bucket[i].vec_length = vec_length;
                main_bucket[i].vec_size = vec_size;
                main_bucket[i].vec = vec;
                main_bucket[i]._set_center(ptr, ind[i]);
                copy(center[i], ptr, vec_length);
            }


            /* each thread collect vectors in local buckets */
            TIMER_START;
            #pragma omp parallel for 
            for (long thread = 0; thread < num_threads; thread++){
                /* prepare local buckets */
                local_mbucket[thread] = new bgj_mbucket_t[BGJ2_BUCKET0_BATCHSIZE];
                for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++){
                    local_mbucket[thread][i]._alloc(0.5 * expect_bucket0_size/num_threads, 1);
                    local_mbucket[thread][i]._alloc(0.5 * expect_bucket0_size/num_threads, 0);
                }

                const long begin_ind = (thread*num_vec)/num_threads;
                const long end_ind = (thread*num_vec+num_vec)/num_threads;
                long ind = begin_ind;

                /// prepare local buffer
                long _buffer_size = global_bucket0_buffer_size[thread];
                long num_buffer = 0;
                uint32_t *cdp_buffer = global_bucket0_cdp_buffer[thread];

                #define ADD_TO_LOCAL_BUFFER(__pind, __cind, __x) do {                                               \
                    cdp_buffer[num_buffer*3] = (uint32_t)(__cind);                                                  \
                    *((float *)(&cdp_buffer[num_buffer*3+1])) = __x;                                                \
                    cdp_buffer[num_buffer*3+2] = __pind;                                                            \
                    num_buffer++;                                                                                   \
                } while (0)
                /// prepare local buffer end

                /* process 8 vectors tegother */
                __m256 sign_bit = _mm256_set1_ps(-0.0f);
                while (ind + 7 < end_ind){
                    float *ptr[8];
                    __attribute__ ((aligned (32))) float alpha_nptr0[4];
                    __attribute__ ((aligned (32))) float alpha_nptr1[4];
                    __attribute__ ((aligned (32))) float dst[8 * BGJ2_BUCKET0_BATCHSIZE];
                    __attribute__ ((aligned (32))) uint32_t cdst[BGJ2_BUCKET0_BATCHSIZE >> 2];
                    __attribute__ ((aligned (32))) int cmp[8];

                    ptr[0] = vec + (0+ind) * vec_size;
                    ptr[1] = vec + (1+ind) * vec_size;
                    ptr[2] = vec + (2+ind) * vec_size;
                    ptr[3] = vec + (3+ind) * vec_size;
                    ptr[4] = vec + (4+ind) * vec_size;
                    ptr[5] = vec + (5+ind) * vec_size;
                    ptr[6] = vec + (6+ind) * vec_size;
                    ptr[7] = vec + (7+ind) * vec_size;

                    alpha_nptr0[0] = ptr[0][-1] * alpha_b0;
                    alpha_nptr0[1] = ptr[1][-1] * alpha_b0;
                    alpha_nptr0[2] = ptr[2][-1] * alpha_b0;
                    alpha_nptr0[3] = ptr[3][-1] * alpha_b0;
                    alpha_nptr1[0] = ptr[4][-1] * alpha_b0;
                    alpha_nptr1[1] = ptr[5][-1] * alpha_b0;
                    alpha_nptr1[2] = ptr[6][-1] * alpha_b0;
                    alpha_nptr1[3] = ptr[7][-1] * alpha_b0;

                    __m128 anptr_half0 = _mm_load_ps(alpha_nptr0);
                    __m128 anptr_half1 = _mm_load_ps(alpha_nptr1);
                    __m256 anptr0 = _mm256_insertf128_ps(_mm256_castps128_ps256(anptr_half0), anptr_half0, 1);
                    __m256 anptr1 = _mm256_insertf128_ps(_mm256_castps128_ps256(anptr_half1), anptr_half1, 1);
                    
                    unsigned cind = 0;
                    while (cind + 7 < BGJ2_BUCKET0_BATCHSIZE){
                        AVX2_DP_CMP_4X4(center[cind], center[cind+1], center[cind+2], center[cind+3], ptr[0], ptr[1], ptr[2], ptr[3], vec_length, anptr0, (&dst[cind*4]), cmp[0], cmp[1]);                   
                        AVX2_DP_CMP_4X4(center[cind], center[cind+1], center[cind+2], center[cind+3], ptr[4], ptr[5], ptr[6], ptr[7], vec_length, anptr1, (&dst[cind*4+BGJ2_BUCKET0_BATCHSIZE*4]), cmp[2], cmp[3]);                   
                        AVX2_DP_CMP_4X4(center[cind+4], center[cind+5], center[cind+6], center[cind+7], ptr[0], ptr[1], ptr[2], ptr[3], vec_length, anptr0, (&dst[cind*4+16]), cmp[4], cmp[5]);                   
                        AVX2_DP_CMP_4X4(center[cind+4], center[cind+5], center[cind+6], center[cind+7], ptr[4], ptr[5], ptr[6], ptr[7], vec_length, anptr1, (&dst[cind*4+16+BGJ2_BUCKET0_BATCHSIZE*4]), cmp[6], cmp[7]);                   
                        cdst[cind >> 3] = ( cmp[0] | (cmp[1] << 8) | (cmp[4] << 16) | (cmp[5] << 24) );
                        cdst[(cind + BGJ2_BUCKET0_BATCHSIZE) >> 3] = ( cmp[2] | (cmp[3] << 8) | (cmp[6] << 16) | (cmp[7] << 24) );
                        cind += 8;
                    }

                    long idx = 0;
                    for (long i = 0; i < 8; i += 4) {
                        for (long j = 0; j < BGJ2_BUCKET0_BATCHSIZE; j += 8){
                            while (cdst[idx]){
                                uint32_t r = __builtin_ctz(cdst[idx]);
                                cdst[idx] -= (1 << r);
                                ADD_TO_LOCAL_BUFFER((r&0x3)+i+ind, j+(r>>2), dst[i*BGJ2_BUCKET0_BATCHSIZE+j*4+r]);
                            }
                            idx++;
                        }
                    }

                    if (num_buffer + 8 * BGJ2_BUCKET0_BATCHSIZE > _buffer_size) {
                        mallopt(M_MMAP_MAX, 65536);
                        mallopt(M_TRIM_THRESHOLD, 128*1024);
                        _buffer_size *= 2;
                        cdp_buffer = (uint32_t *) realloc(cdp_buffer, _buffer_size * 3 * sizeof(uint32_t));
                        global_bucket0_cdp_buffer[thread] = cdp_buffer;
                        mallopt(M_MMAP_MAX, 0);
                        mallopt(M_TRIM_THRESHOLD, -1);
                    }

                    // we always choose BUCKETING_BATCH_SIZE divided by 4,
                    // so we do not optimize here
                    /*while (cind < BGJ2_BUCKET0_BATCHSIZE){
                    //    float __dst0, __dst1, __dst2, __dst3;
                    //    AVX2_DOT_PRODUCT_1X4(center[cind], ptr[0], ptr[1], ptr[2], ptr[3], vec_length);                    
                    //    if (fabs(__dst0) > alpha_nptr0[0]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[0], cind, __dst0);
                    //    if (fabs(__dst1) > alpha_nptr0[1]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[1], cind, __dst1);
                    //    if (fabs(__dst2) > alpha_nptr0[2]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[2], cind, __dst2);
                    //    if (fabs(__dst3) > alpha_nptr0[3]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[3], cind, __dst3);
                    //    AVX2_DOT_PRODUCT_1X4(center[cind], ptr[4], ptr[5], ptr[6], ptr[7], vec_length);                    
                    //    if (fabs(__dst0) > alpha_nptr1[0]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[4], cind, __dst0);
                    //    if (fabs(__dst1) > alpha_nptr1[1]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[5], cind, __dst1);
                    //    if (fabs(__dst2) > alpha_nptr1[2]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[6], cind, __dst2);
                    //    if (fabs(__dst3) > alpha_nptr1[3]) TAIL_ADD_TO_LOCAL_MBUCKET(ptr[7], cind, __dst3);
                    //    cind++;
                    //}*/

                    ind += 8;
                }

                for (long i = 0; i < num_buffer; i++){
                    local_mbucket[thread][cdp_buffer[i*3]]._add_vec(cdp_buffer[i*3+2], *((float *)(&cdp_buffer[i*3+1])));
                }

                global_bucket0_buffer_size[thread] = _buffer_size;

                #define ADD_TO_LOCAL_MBUCKET(__pind, __cind, __x) do {                                              \
                    local_mbucket[thread][__cind]._add_vec(__pind, __x);                                            \
                } while (0)

                /* the remaining part */
                while(ind < end_ind){
                    float *ptr = vec + ind * vec_size;
                    float alpha_nptr = ptr[-1] * alpha_b0;
                    long cind = 0;
                    while (cind + 3 < BGJ2_BUCKET0_BATCHSIZE){
                        float __dst0, __dst1, __dst2, __dst3;
                        AVX2_DOT_PRODUCT_1X4(ptr, center[cind], center[cind+1], center[cind+2], center[cind+3], vec_length);
                        if (fabs(__dst0) > alpha_nptr) ADD_TO_LOCAL_MBUCKET(ind, cind, __dst0);
                        if (fabs(__dst1) > alpha_nptr) ADD_TO_LOCAL_MBUCKET(ind, cind+1, __dst1);
                        if (fabs(__dst2) > alpha_nptr) ADD_TO_LOCAL_MBUCKET(ind, cind+2, __dst2);
                        if (fabs(__dst3) > alpha_nptr) ADD_TO_LOCAL_MBUCKET(ind, cind+3, __dst3);
                        cind += 4;
                    }
                    //while (cind < BGJ2_BUCKET0_BATCHSIZE){
                    //    float x = dot_avx2(ptr, center[cind], vec_length);
                    //    if (fabs(x) > alpha_nptr) ADD_TO_LOCAL_MBUCKET(ptr, cind, x);
                    //    cind++;
                    //}
                    ind++;
                }

                #undef ADD_TO_LOCAL_MBUCKET

                if (thread == 0) _bucket0_ndp += num_vec * BGJ2_BUCKET0_BATCHSIZE;
            }
            TIMER_END;
            _bucket0_time += CURRENT_TIME;


            /* combine the data to main buckets */
            TIMER_START;
            #pragma omp parallel for
            for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++){
                bgj_mbucket_t *tmp[MAX_NTHREADS];
                for (long j = 0; j < num_threads; j++) tmp[j] = &local_mbucket[j][i];
                main_bucket[i]._combine(tmp, num_threads);
            }
            TIMER_END;
            combine_time += CURRENT_TIME;

            if (log_level >= 4) {
                for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++){
                    printf("# main bucket %ld\n", i);
                    printf("# nsize = %ld, psize = %ld\n", main_bucket[i]._num_nvec, main_bucket[i]._num_pvec);
                    int alpha_pvec_pass = 1, alpha_nvec_pass = 1;
                    for (long j = 0; j < main_bucket[i]._num_pvec; j++){
                        float *ptr = vec + vec_size * main_bucket[i]._pvec[j];
                        float x = dot(ptr, main_bucket[i].center, vec_length);
                        if (fabs(x - main_bucket[i]._pdot[j]) > 10) {
                            alpha_pvec_pass = 0;
                        }
                        x /= ptr[-1];
                        if (fabs(x) < alpha_b0) {
                            alpha_pvec_pass = 0;
                        }
                    }
                    for (long j = 0; j < main_bucket[i]._num_nvec; j++){
                        float *ptr = vec + vec_size * main_bucket[i]._nvec[j];
                        float x = dot(ptr, main_bucket[i].center, vec_length);
                        if (fabs(x - main_bucket[i]._ndot[j]) > 10) {
                            alpha_nvec_pass = 0;
                        }
                        x /= ptr[-1];
                        if (fabs(x) < alpha_b0) {
                            alpha_nvec_pass = 0;
                        }
                    }
                    printf("# dot test of pvecs: %d\n", alpha_pvec_pass);
                    printf("# dot test of nvecs: %d\n", alpha_nvec_pass);
                }
            }


            /* free local buckets and centers */
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++){
                    local_mbucket[thread][i].clear();
                }
                delete[] local_mbucket[thread];
            }
            delete[] local_mbucket;
            FREE_MAT(center);

            _num_bucket0 += BGJ2_BUCKET0_BATCHSIZE;
            for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++) _sum_bucket0_size += main_bucket[i]._num_nvec + main_bucket[i]._num_pvec;
            if (log_level >= 3 && CSD > MIN_LOG_CSD) {
                double speed = CSD * 2 * _bucket0_ndp/_bucket0_time/1073741824.0;
                dlog("bucket0 done, avg bucket size = %ld, num_dp = %ld", _sum_bucket0_size/_num_bucket0, _bucket0_ndp);
                dlog("bucket0 speed: %f bucket/s, %f GFLOPS", _num_bucket0/_bucket0_time, speed);
            }
            
            /* bucket1 */
            TIMER_START;
            bgj_sbucket_t<1> *sbucket3[BGJ2_BUCKET0_BATCHSIZE];
            bgj_sbucket_t<0> *sbucket2[BGJ2_BUCKET0_BATCHSIZE * BGJ2_BUCKET1_BATCHSIZE];
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                const long begin_ind = (thread*BGJ2_BUCKET0_BATCHSIZE)/num_threads;
                const long end_ind = (thread*BGJ2_BUCKET0_BATCHSIZE+BGJ2_BUCKET0_BATCHSIZE)/num_threads;
                long ind = begin_ind;
                while (ind < end_ind){
                    #if BGJ2_BUCKET1_USE_NONLATTICE_CENTER
                    main_bucket[ind].bucket1(alpha_b1, alpha_r0, sbucket3+ind, &sbucket2[ind * BGJ2_BUCKET1_BATCHSIZE]);
                    #else 
                    main_bucket[ind].bucket1(alpha_b1, alpha_r0, sbucket3+ind, &sbucket2[ind * BGJ2_BUCKET1_BATCHSIZE], uid, local_buffer[thread], goal_norm);
                    #endif
                    ind++;
                }
            }
            TIMER_END;
            
            if (log_level >= 4) {
                for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++){
                    printf("# subbucket of main bucket %ld\n", i);
                    long sum = 0;
                    for (long ii = 0; ii < BGJ2_BUCKET1_BATCHSIZE; ii++) sum += sbucket2[BGJ2_BUCKET1_BATCHSIZE*i+ii]->num_nvec + sbucket2[BGJ2_BUCKET1_BATCHSIZE*i+ii]->num_pvec;
                    printf("# 3size = %ld, avg 2size = %ld\n", sbucket3[i]->num_pvec+sbucket3[i]->num_nvec, sum/BGJ2_BUCKET1_BATCHSIZE);
                    float *center0 = main_bucket[i].center;
                    // check dst3
                    int pass3n = 1, pass2n = 1;
                    int pass3d = 1, pass2d = 1;
                    int pass3c = 1, pass2c = 1;
                    
                    do {
                        float *cptr = sbucket3[i]->center;
                        if (fabs(cptr[-1] - sbucket3[i]->center_norm) > 5.0f) pass3c = 0;
                        if (*((uint64_t *)(&cptr[-4])) != sbucket3[i]->center_u) pass3c = 0;
                        if (dot(cptr, center0, vec_length) < (alpha_b0-0.01)*cptr[-1]) pass3c = 0;
                        for (long j = 0; j < sbucket3[i]->num_pvec; j++){
                            float *ptr = sbucket3[i]->pvec[j];
                            float d = dot(ptr, center0, vec_length);
                            if (fabs(ptr[-1] - sbucket3[i]->pnorm[j]) > 5.0f) pass3n = 0;
                            float dd = dot(sbucket3[i]->pvec[j], sbucket3[i]->center, vec_length);
                            if (fabs(dd - sbucket3[i]->pdot[j]) > 10.0f) {
                                printf("here\n\n");
                                pass3d = 0;
                            }
                            if (dd/ptr[-1] < alpha_b1-0.01) {
                                printf("here\n\n");
                                pass3d = 0;
                            }
                        }
                        for (long j = 0; j < sbucket3[i]->num_nvec; j++){
                            float *ptr = sbucket3[i]->nvec[j];
                            float d = dot(ptr, center0, vec_length);
                            if (fabs(ptr[-1] - sbucket3[i]->nnorm[j]) > 5.0f) pass3n = 0;
                            float dd = dot(sbucket3[i]->nvec[j], sbucket3[i]->center, vec_length);
                            if (fabs(dd - sbucket3[i]->ndot[j]) > 10.0f) {
                                printf("here\n\n");
                                pass3d = 0;
                            }
                            if (-dd/ptr[-1] < alpha_b1-0.01) {
                                printf("here\n\n");
                                pass3d = 0;
                            }
                        }
                    } while (0);

                    // check dst2
                    for (long k = 0; k < BGJ2_BUCKET1_BATCHSIZE; k++){
                        do {
                            float *cptr = sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->center;
                            if (fabs(cptr[-1] - sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->center_norm) > 5.0f) pass2c = 0;
                            if (*((uint64_t *)(&cptr[-4])) != sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->center_u) pass2c = 0;
                            if (dot(cptr, center0, vec_length) < (alpha_b0-0.01)*cptr[-1]) pass2c = 0;
                            for (long j = 0; j < sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->num_pvec; j++){
                                float *ptr = sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->pvec[j];
                                float d = dot(ptr, center0, vec_length);
                                if (fabs(ptr[-1] - sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->pnorm[j]) > 5.0f) pass2n = 0;
                                float dd = dot(sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->pvec[j], sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->center, vec_length);
                                if (dd/ptr[-1] < alpha_b1-0.01) pass2d = 0;
                            }
                            for (long j = 0; j < sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->num_nvec; j++){
                                float *ptr = sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->nvec[j];
                                float d = dot(ptr, center0, vec_length);
                                if (fabs(ptr[-1] - sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->nnorm[j]) > 5.0f) pass2n = 0;
                                float dd = dot(sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->nvec[j], sbucket2[k+i*BGJ2_BUCKET1_BATCHSIZE]->center, vec_length);
                                if (-dd/ptr[-1] < alpha_b1-0.01) pass2d = 0;
                            }
                        } while (0);
                    }
                    
                    printf("# ndc: pass3 = %d%d%d, pass2 = %d%d%d\n", pass3n, pass3d, pass3c, pass2n, pass2d, pass2c);
                    if (0) {
                        long count2[BGJ2_BUCKET1_BATCHSIZE] = {};
                        long count3 = 0;
                        for (long ii = 0; ii < main_bucket[i]._num_pvec; ii++){
                            if (dot(vec + vec_size * main_bucket[i]._pvec[ii], center0, vec_length) > alpha_r0 * (vec + vec_size * main_bucket[i]._pvec[ii])[-1]) count3++;
                            for (long j = 0; j < BGJ2_BUCKET1_BATCHSIZE; j++){
                                float *ptr = vec + vec_size * main_bucket[i]._pvec[ii];
                                if (dot(ptr, sbucket2[i*BGJ2_BUCKET1_BATCHSIZE+j]->center, vec_length) > alpha_b1 * ptr[-1]) count2[j]++;
                            }
                        }
                        
                        for (long ii = 0; ii < main_bucket[i]._num_nvec; ii++){
                            if (-dot(vec + vec_size * main_bucket[i]._nvec[ii], center0, vec_length) > alpha_r0 * (vec + vec_size * main_bucket[i]._nvec[ii])[-1]) count3++;
                            for (long j = 0; j < BGJ2_BUCKET1_BATCHSIZE; j++){
                                float *ptr = vec + vec_size * main_bucket[i]._nvec[ii];
                                if (-dot(ptr, sbucket2[i*BGJ2_BUCKET1_BATCHSIZE+j]->center, vec_length) > alpha_b1 * ptr[-1]) count2[j]++;
                            }
                        }
                        long ssum = 0;
                        for (long ss = 0; ss < BGJ2_BUCKET1_BATCHSIZE; ss++) ssum += count2[ss];
                        //for (long ss = 0; ss < BGJ2_BUCKET1_BATCHSIZE; ss++) {
                        //    printf("%ld ", sbucket2[i*BGJ2_BUCKET1_BATCHSIZE+ss]->nvec);
                        //}
                        //printf("\n");
                        //for (long ss = 0; ss < BGJ2_BUCKET1_BATCHSIZE; ss++) {
                        //    printf("%ld ", count2[ss]);
                        //}
                        //printf("\n");
                        printf("# 3bucket left %ld vec, 2bucket left %ld vec\n", count3 - sbucket3[i]->num_pvec - sbucket3[i]->num_nvec, ssum - sum);
                    }
                }
            }

            
            _bucket1_time += CURRENT_TIME;
            _num_bucket1 += BGJ2_BUCKET0_BATCHSIZE * (BGJ2_BUCKET1_BATCHSIZE+1);
            for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++) {
                _sum_bucket1_size += sbucket3[i]->num_pvec + sbucket3[i]->num_nvec;
                _bucket1_ndp += (main_bucket[i]._num_nvec + main_bucket[i]._num_pvec) * BGJ2_BUCKET1_BATCHSIZE;
                _try_add2 += main_bucket[i].try_add2;
                _succ_add2 += main_bucket[i].succ_add2;
            }
            for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE*BGJ2_BUCKET1_BATCHSIZE; i++){
                if (sbucket2[i]) _sum_bucket1_size += sbucket2[i]->num_pvec + sbucket2[i]->num_nvec;
            }
            if (log_level >= 3 && CSD > MIN_LOG_CSD) {
                double speed = CSD * 2 * _bucket1_ndp/_bucket1_time/1073741824.0;
                dlog("bucket1 done, avg bucket size = %ld, num_dp = %ld", _sum_bucket1_size/_num_bucket1, _bucket1_ndp);
                dlog("bucket1 speed: %f bucket/s, %f GFLOPS", _num_bucket1/_bucket1_time, speed);
                dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld", _try_add2, _try_add3, _succ_add2, _succ_add3);
            }


            /* search0 */
            TIMER_START;
            pthread_spinlock_t bucket_list_lock;
            pthread_spin_init(&bucket_list_lock, PTHREAD_PROCESS_SHARED);
            long nrem_bucket = BGJ2_BUCKET0_BATCHSIZE;
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                long __search0_ndp = 0;
                uint64_t __try_add2 = 0;
                uint64_t __succ_add2 = 0;
                uint64_t __try_add3 = 0;
                uint64_t __succ_add3 = 0;
                while (nrem_bucket) {
                    bgj_sbucket_t<1> *bkt = NULL;
                    // get a bucket
                    pthread_spin_lock(&bucket_list_lock);
                    if (nrem_bucket > 0) {
                        bkt = sbucket3[nrem_bucket-1];
                        nrem_bucket--;
                        if (_succ_add2 + _succ_add3 + (__succ_add2 + __succ_add3)*num_threads > num_empty + sorted_index - goal_index){
                            bkt = NULL;
                            if (log_level >= 1 && CSD > 40) dlog("search0: %ld buckets not use", nrem_bucket);
                            nrem_bucket = 0;
                        }
                    }
                    pthread_spin_unlock(&bucket_list_lock);
                    if (bkt == NULL) continue;
                    __search0_ndp += bkt->search_naive(local_buffer[thread], uid, goal_norm);
                    __search0_ndp += bkt->search_pp(local_buffer[thread], uid, goal_norm);
                    __search0_ndp += bkt->search_nn(local_buffer[thread], uid, goal_norm);
                    __search0_ndp += bkt->search_np(local_buffer[thread], uid, goal_norm);
                    __try_add2 += bkt->try_add2;
                    __try_add3 += bkt->try_add3;
                    __succ_add2 += bkt->succ_add2;
                    __succ_add3 += bkt->succ_add3;
                }
                pthread_spin_lock(&debug_lock);
                _search0_ndp += __search0_ndp;
                _try_add2 += __try_add2;
                _try_add3 += __try_add3;
                _succ_add2 += __succ_add2;
                _succ_add3 += __succ_add3;
                pthread_spin_unlock(&debug_lock);
            }
            TIMER_END;
            _search0_time += CURRENT_TIME;

            if (log_level >= 3 && CSD > MIN_LOG_CSD) {
                double speed = CSD * 2 * _search0_ndp/_search0_time/1073741824.0;
                dlog("search0 done, num_dp = %ld, %ld solutions found", _search0_ndp, _succ_add2 + _succ_add3);
                dlog("search0 speed: %f bucket/s, %f GFLOPS", _num_bucket0/_search0_time, speed);
                dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld", _try_add2, _try_add3, _succ_add2, _succ_add3);
            }


            /* search1 */
            TIMER_START;
            nrem_bucket = BGJ2_BUCKET0_BATCHSIZE*BGJ2_BUCKET1_BATCHSIZE;
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                long __search1_ndp = 0;
                uint64_t __try_add2 = 0;
                uint64_t __succ_add2 = 0;
                uint64_t __try_add3 = 0;
                uint64_t __succ_add3 = 0;
                while (nrem_bucket) {
                    bgj_sbucket_t<0> *bkt = NULL;
                    // get a bucket
                    pthread_spin_lock(&bucket_list_lock);
                    if (nrem_bucket > 0) {
                        bkt = sbucket2[nrem_bucket-1];
                        nrem_bucket--;
                        if (_succ_add2 + _succ_add3 + (__succ_add2 + __succ_add3)*num_threads > num_empty + sorted_index - goal_index){
                            bkt = NULL;
                            if (log_level >= 1 && CSD > 40) dlog("search1: %ld buckets not use", nrem_bucket);
                            nrem_bucket = 0;
                        }
                    }
                    pthread_spin_unlock(&bucket_list_lock);
                    if (bkt == NULL) continue;
                    __search1_ndp += bkt->search_naive(local_buffer[thread], uid, goal_norm);
                    __search1_ndp += bkt->search_pp(local_buffer[thread], uid, goal_norm);
                    __search1_ndp += bkt->search_nn(local_buffer[thread], uid, goal_norm);
                    __search1_ndp += bkt->search_np(local_buffer[thread], uid, goal_norm);
                    __try_add2 += bkt->try_add2;
                    __try_add3 += bkt->try_add3;
                    __succ_add2 += bkt->succ_add2;
                    __succ_add3 += bkt->succ_add3;
                }
                pthread_spin_lock(&debug_lock);
                _search1_ndp += __search1_ndp;
                _try_add2 += __try_add2;
                _try_add3 += __try_add3;
                _succ_add2 += __succ_add2;
                _succ_add3 += __succ_add3;
                pthread_spin_unlock(&debug_lock);
            }
            TIMER_END;
            _search1_time += CURRENT_TIME;

            if (log_level >= 3 && CSD > MIN_LOG_CSD) {
                double speed = CSD * 2 * _search1_ndp/_search1_time/1073741824.0;
                dlog("search1 done, num_dp = %ld, %ld solutions found", _search1_ndp, _succ_add2 + _succ_add3);
                dlog("search1 speed: %f bucket/s, %f GFLOPS", _num_bucket1/_search1_time, speed);
                dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld", _try_add2, _try_add3, _succ_add2, _succ_add3);
            }


            /* free sbuckets */
            for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE; i++) {
                sbucket3[i]->_clear();
                delete sbucket3[i];
            }
            for (long i = 0; i < BGJ2_BUCKET0_BATCHSIZE*BGJ2_BUCKET1_BATCHSIZE; i++) {
                if (sbucket2[i]) {
                    sbucket2[i]->_clear();
                    delete sbucket2[i];
                }
            }

            /* check if solution collection is done */
            num_total_sol = 0;
            for (long i = 0; i < num_threads; i++){
                num_total_sol += local_buffer[i]->num_sol();
            }
            if (num_total_sol <= last_num_total_sol + stucktime * BGJ2_BUCKET0_BATCHSIZE/2){
                stucktime++;
            } else {
                stucktime = 0;
                last_num_total_sol = num_total_sol;
            }
            if (num_total_sol > one_epoch_ratio * num_vec) rel_collection_stop = true;
            if (stucktime >= MAX_STUCK_TIME) {
                sieving_stucked = 1;
                rel_collection_stop = true;
            } 
        } while (!rel_collection_stop);


        if (log_level >= 1 && CSD > MIN_LOG_CSD) {
            double speed0, speed1;
            dlog("solution collect done, found %ld solutions, bucket-0 time = %fs, bucket-1 time = %fs, search-0 time = %fs, search-1 time = %fs",
                     num_total_sol, _bucket0_time, _bucket1_time, _search0_time, _search1_time);
            if (log_level >= 2) {
                speed0 = CSD * 2 * _bucket0_ndp/_bucket0_time/1073741824.0;
                speed1 = CSD * 2 * _bucket1_ndp/_bucket1_time/1073741824.0;
                dlog("bucket0 speed: %f bucket/s, %f GFLOPS", _num_bucket0/_bucket0_time, speed0);
                dlog("bucket1 speed: %f bucket/s, %f GFLOPS", _num_bucket1/_bucket1_time, speed1);        
                speed0 = CSD * 2 * _search0_ndp/_search0_time/1073741824.0;
                speed1 = CSD * 2 * _search1_ndp/_search1_time/1073741824.0;
                dlog("search0 speed: %f bucket/s, %f GFLOPS", _num_bucket0/_search0_time, speed0);
                dlog("search1 speed: %f bucket/s, %f GFLOPS", _num_bucket1/_search1_time, speed1);
                dlog("num bucket0 = %ld, avg bucket0 size = %ld, num bucket1 = %ld, avg bucket1 size = %ld", _num_bucket0, (long)(_sum_bucket0_size/(0.000001+_num_bucket0)), _num_bucket1, (long)(_sum_bucket1_size/(0.000001+_num_bucket1)));
                dlog("bucket cost = %ld dp/sol, search cost = %ld dp/sol", (_bucket0_ndp+_bucket1_ndp)/(_succ_add2 + _succ_add3), (_search0_ndp+_search1_ndp)/(_succ_add2 + _succ_add3));
                dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld", _try_add2, _try_add3, _succ_add2, _succ_add3);
            }
        }
        
        do {
            num_bucket0 += _num_bucket0;
            num_bucket1 += _num_bucket1;
            sum_bucket0_size += _sum_bucket0_size;
            sum_bucket1_size += _sum_bucket1_size;
            bucket0_time += _bucket0_time;
            bucket1_time += _bucket1_time;
            search0_time += _search0_time;
            search1_time += _search1_time;
            bucket0_ndp += _bucket0_ndp;
            bucket1_ndp += _bucket1_ndp;
            search0_ndp += _search0_ndp;
            search1_ndp += _search1_ndp;
            try_add2 += _try_add2;
            try_add3 += _try_add3;
            succ_add2 += _succ_add2;
            succ_add3 += _succ_add3;
        } while (0);

        /* check all uid of solutions have been inserted and the norm of solutinos is small enough */
        TIMER_START;
        if (0){
            for (long i = 0; i < num_threads; i++){
                for (long j = 0; j < local_buffer[i]->num_a; j++){
                    float *ptr1 = local_buffer[i]->a_list[2*j];
                    float *ptr2 = local_buffer[i]->a_list[2*j+1];
                    float x = ptr1[-1] + ptr2[-1] + 2 * dot(ptr1, ptr2, vec_length);
                    if (x >= goal_norm) fprintf(stderr, "ERROR!!!, norm = %f\n\n\n", sqrt(x));
                    if (!uid->check_uid(*(uint64_t *)(&ptr1[-4])+*(uint64_t *)(&ptr2[-4]))) fprintf(stderr, "ERROR!!!\n\n\n");
                }

                for (long j = 0; j < local_buffer[i]->num_s; j++){
                    float *ptr1 = local_buffer[i]->s_list[2*j];
                    float *ptr2 = local_buffer[i]->s_list[2*j+1];
                    float x = ptr1[-1] + ptr2[-1] - 2 * dot(ptr1, ptr2, vec_length);
                    if (x >= goal_norm) fprintf(stderr, "ERROR!!!, norm = %f\n\n\n", sqrt(x));
                    if (!uid->check_uid(*(uint64_t *)(&ptr1[-4])-*(uint64_t *)(&ptr2[-4]))) fprintf(stderr, "ERROR!!!\n\n\n");
                }

                for (long j = 0; j < local_buffer[i]->num_aa; j++){
                    float *ptr1 = local_buffer[i]->aa_list[3*j];
                    float *ptr2 = local_buffer[i]->aa_list[3*j+1];
                    float *ptr3 = local_buffer[i]->aa_list[3*j+2];
                    __attribute__ ((aligned (64))) float tmp[256] = {};
                    add(tmp, ptr1, vec_length);
                    add(tmp, ptr2, vec_length);
                    add(tmp, ptr3, vec_length);
                    float x = dot(tmp, tmp, vec_length);
                    if (x >= goal_norm) fprintf(stderr, "ERROR!!!, norm = %f\n\n\n", sqrt(x));
                    if (!uid->check_uid(*(uint64_t *)(&ptr1[-4])+*(uint64_t *)(&ptr2[-4])+*(uint64_t *)(&ptr3[-4]))) fprintf(stderr, "ERROR!!!\n\n\n");
                }

                for (long j = 0; j < local_buffer[i]->num_sa; j++){
                    float *ptr1 = local_buffer[i]->sa_list[3*j];
                    float *ptr2 = local_buffer[i]->sa_list[3*j+1];
                    float *ptr3 = local_buffer[i]->sa_list[3*j+2];
                    __attribute__ ((aligned (64))) float tmp[256] = {};
                    add(tmp, ptr1, vec_length);
                    sub(tmp, ptr2, vec_length);
                    add(tmp, ptr3, vec_length);
                    float x = dot(tmp, tmp, vec_length);
                    if (x >= goal_norm) fprintf(stderr, "ERROR!!!, norm = %f\n\n\n", sqrt(x));
                    if (!uid->check_uid(*(uint64_t *)(&ptr1[-4])-*(uint64_t *)(&ptr2[-4])+*(uint64_t *)(&ptr3[-4]))) fprintf(stderr, "ERROR!!!\n\n\n");
                }

                for (long j = 0; j < local_buffer[i]->num_ss; j++){
                    float *ptr1 = local_buffer[i]->ss_list[3*j];
                    float *ptr2 = local_buffer[i]->ss_list[3*j+1];
                    float *ptr3 = local_buffer[i]->ss_list[3*j+2];
                    __attribute__ ((aligned (64))) float tmp[256] = {};
                    add(tmp, ptr1, vec_length);
                    sub(tmp, ptr2, vec_length);
                    sub(tmp, ptr3, vec_length);
                    float x = dot(tmp, tmp, vec_length);
                    if (x >= goal_norm) fprintf(stderr, "ERROR!!!, norm = %f\n\n\n", sqrt(x));
                    if (!uid->check_uid(*(uint64_t *)(&ptr1[-4])-*(uint64_t *)(&ptr2[-4])-*(uint64_t *)(&ptr3[-4]))) fprintf(stderr, "ERROR!!!\n\n\n");
                }
            }
        }
        
        
        /* insert */
        long num_total_insert;
        do {
            long sol_too_much = 0;
            long start_ind[MAX_NTHREADS] = {};
            long end_ind[MAX_NTHREADS] = {};
            long buffer_nsol[MAX_NTHREADS];
            const long coeff_size = (int_bias-16)*2;
            for (long i = 0; i < num_threads; i++) buffer_nsol[i] = local_buffer[i]->num_sol();
            start_ind[0] = 0; end_ind[0] = buffer_nsol[0];
            for (long i = 1; i < num_threads; i++){
                start_ind[i] = end_ind[i-1];
                end_ind[i] = start_ind[i] + buffer_nsol[i];
            }
            num_total_insert = end_ind[num_threads-1];
            if (num_total_insert > num_empty + sorted_index - goal_index) {
                sol_too_much = 1;
                num_total_insert = num_empty + sorted_index - goal_index;
                for (long i = 0; i < num_threads; i++) {
                    if (buffer_nsol[i] > ((i+1) * num_total_insert)/num_threads - (i * num_total_insert)/num_threads){
                        buffer_nsol[i] = ((i+1) * num_total_insert)/num_threads - (i * num_total_insert)/num_threads;
                    }
                }
                start_ind[0] = 0; end_ind[0] = buffer_nsol[0];
                for (long i = 1; i < num_threads; i++){
                    start_ind[i] = end_ind[i-1];
                    end_ind[i] = start_ind[i] + buffer_nsol[i];
                }
                num_total_insert = end_ind[num_threads-1];
            }

            float *to_insert_vec_store = (float *) NEW_VEC((num_total_insert + 1) * vec_size, sizeof(float));
            float *to_insert_vec = to_insert_vec_store + vec_size;
            
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                long ind = start_ind[thread];
                if (!sol_too_much) {
                    for (long i = 0; i < local_buffer[thread]->num_a; i++){
                        float *dst = to_insert_vec + ind * vec_size;
                        short *coeff_dst = (short *)(dst - int_bias);
                        short *src1 = (short *)(local_buffer[thread]->a_list[i*2] - int_bias);
                        short *src2 = (short *)(local_buffer[thread]->a_list[i*2+1] - int_bias);
                        add(coeff_dst, src1, coeff_size);
                        add(coeff_dst, src2, coeff_size);
                        compute_vec(dst);
                        ind++;
                    }
                    for (long i = 0; i < local_buffer[thread]->num_s; i++){
                        float *dst = to_insert_vec + ind * vec_size;
                        short *coeff_dst = (short *)(dst - int_bias);
                        short *src1 = (short *)(local_buffer[thread]->s_list[i*2] - int_bias);
                        short *src2 = (short *)(local_buffer[thread]->s_list[i*2+1] - int_bias);
                        add(coeff_dst, src1, coeff_size);
                        sub(coeff_dst, src2, coeff_size);
                        compute_vec(dst);
                        ind++;
                    }
                    for (long i = 0; i < local_buffer[thread]->num_aa; i++){
                        float *dst = to_insert_vec + ind * vec_size;
                        short *coeff_dst = (short *)(dst - int_bias);
                        short *src1 = (short *)(local_buffer[thread]->aa_list[i*3] - int_bias);
                        short *src2 = (short *)(local_buffer[thread]->aa_list[i*3+1] - int_bias);
                        short *src3 = (short *)(local_buffer[thread]->aa_list[i*3+2] - int_bias);
                        add(coeff_dst, src1, coeff_size);
                        add(coeff_dst, src2, coeff_size);
                        add(coeff_dst, src3, coeff_size);
                        compute_vec(dst);
                        ind++;
                    }
                    for (long i = 0; i < local_buffer[thread]->num_sa; i++){
                        float *dst = to_insert_vec + ind * vec_size;
                        short *coeff_dst = (short *)(dst - int_bias);
                        short *src1 = (short *)(local_buffer[thread]->sa_list[i*3] - int_bias);
                        short *src2 = (short *)(local_buffer[thread]->sa_list[i*3+1] - int_bias);
                        short *src3 = (short *)(local_buffer[thread]->sa_list[i*3+2] - int_bias);
                        add(coeff_dst, src1, coeff_size);
                        sub(coeff_dst, src2, coeff_size);
                        add(coeff_dst, src3, coeff_size);
                        compute_vec(dst);
                        ind++;
                    }
                    for (long i = 0; i < local_buffer[thread]->num_ss; i++){
                        float *dst = to_insert_vec + ind * vec_size;
                        short *coeff_dst = (short *)(dst - int_bias);
                        short *src1 = (short *)(local_buffer[thread]->ss_list[i*3] - int_bias);
                        short *src2 = (short *)(local_buffer[thread]->ss_list[i*3+1] - int_bias);
                        short *src3 = (short *)(local_buffer[thread]->ss_list[i*3+2] - int_bias);
                        add(coeff_dst, src1, coeff_size);
                        sub(coeff_dst, src2, coeff_size);
                        sub(coeff_dst, src3, coeff_size);
                        compute_vec(dst);
                        ind++;
                    }
                } else {
                    for (long i = 0; i < local_buffer[thread]->num_a; i++){
                        if (ind >= end_ind[thread]){
                            uint64_t u1 = *((uint64_t *)&(local_buffer[thread]->a_list[i*2][-4]));
                            uint64_t u2 = *((uint64_t *)&(local_buffer[thread]->a_list[i*2+1][-4]));
                            if (!uid->erase_uid(u1 + u2)){
                                fprintf(stderr, "[Error] Pool::bgj2_Sieve: erase uid of non inserted vector failed, ignored.\n");
                            }
                        } else {
                            float *dst = to_insert_vec + ind * vec_size;
                            short *coeff_dst = (short *)(dst - int_bias);
                            short *src1 = (short *)(local_buffer[thread]->a_list[i*2] - int_bias);
                            short *src2 = (short *)(local_buffer[thread]->a_list[i*2+1] - int_bias);
                            add(coeff_dst, src1, coeff_size);
                            add(coeff_dst, src2, coeff_size);
                            compute_vec(dst);
                            ind++;
                        }
                    }

                    for (long i = 0; i < local_buffer[thread]->num_s; i++){
                        if (ind >= end_ind[thread]){
                            uint64_t u1 = *((uint64_t *)&(local_buffer[thread]->s_list[i*2][-4]));
                            uint64_t u2 = *((uint64_t *)&(local_buffer[thread]->s_list[i*2+1][-4]));
                            if (!uid->erase_uid(u1 - u2)){
                                fprintf(stderr, "[Error] Pool::bgj2_Sieve: erase uid of non inserted vector failed, ignored.\n");
                            }
                        } else {
                            float *dst = to_insert_vec + ind * vec_size;
                            short *coeff_dst = (short *)(dst - int_bias);
                            short *src1 = (short *)(local_buffer[thread]->s_list[i*2] - int_bias);
                            short *src2 = (short *)(local_buffer[thread]->s_list[i*2+1] - int_bias);
                            add(coeff_dst, src1, coeff_size);
                            sub(coeff_dst, src2, coeff_size);
                            compute_vec(dst);
                            ind++;
                        }
                    }

                    for (long i = 0; i < local_buffer[thread]->num_aa; i++){
                        if (ind >= end_ind[thread]){
                            uint64_t uc = *((uint64_t *)&(local_buffer[thread]->aa_list[i*3+0][-4]));
                            uint64_t u1 = *((uint64_t *)&(local_buffer[thread]->aa_list[i*3+1][-4]));
                            uint64_t u2 = *((uint64_t *)&(local_buffer[thread]->aa_list[i*3+2][-4]));
                            if (!uid->erase_uid(uc + u1 + u2)){
                                fprintf(stderr, "[Error] Pool::bgj2_Sieve: erase uid of non inserted vector failed, ignored.\n");
                            }
                        } else {
                            float *dst = to_insert_vec + ind * vec_size;
                            short *coeff_dst = (short *)(dst - int_bias);
                            short *src1 = (short *)(local_buffer[thread]->aa_list[i*3] - int_bias);
                            short *src2 = (short *)(local_buffer[thread]->aa_list[i*3+1] - int_bias);
                            short *src3 = (short *)(local_buffer[thread]->aa_list[i*3+2] - int_bias);
                            add(coeff_dst, src1, coeff_size);
                            add(coeff_dst, src2, coeff_size);
                            add(coeff_dst, src3, coeff_size);
                            compute_vec(dst);
                            ind++;
                        }
                    }

                    for (long i = 0; i < local_buffer[thread]->num_sa; i++){
                        if (ind >= end_ind[thread]){
                            uint64_t uc = *((uint64_t *)&(local_buffer[thread]->sa_list[i*3+0][-4]));
                            uint64_t u1 = *((uint64_t *)&(local_buffer[thread]->sa_list[i*3+1][-4]));
                            uint64_t u2 = *((uint64_t *)&(local_buffer[thread]->sa_list[i*3+2][-4]));
                            if (!uid->erase_uid(uc - u1 + u2)){
                                fprintf(stderr, "[Error] Pool::bgj2_Sieve: erase uid of non inserted vector failed, ignored.\n");
                            }
                        } else {
                            float *dst = to_insert_vec + ind * vec_size;
                            short *coeff_dst = (short *)(dst - int_bias);
                            short *src1 = (short *)(local_buffer[thread]->sa_list[i*3] - int_bias);
                            short *src2 = (short *)(local_buffer[thread]->sa_list[i*3+1] - int_bias);
                            short *src3 = (short *)(local_buffer[thread]->sa_list[i*3+2] - int_bias);
                            add(coeff_dst, src1, coeff_size);
                            sub(coeff_dst, src2, coeff_size);
                            add(coeff_dst, src3, coeff_size);
                            compute_vec(dst);
                            ind++;
                        }
                    }

                    for (long i = 0; i < local_buffer[thread]->num_ss; i++){
                        if (ind >= end_ind[thread]){
                            uint64_t uc = *((uint64_t *)&(local_buffer[thread]->ss_list[i*3+0][-4]));
                            uint64_t u1 = *((uint64_t *)&(local_buffer[thread]->ss_list[i*3+1][-4]));
                            uint64_t u2 = *((uint64_t *)&(local_buffer[thread]->ss_list[i*3+2][-4]));
                            if (!uid->erase_uid(uc - u1 - u2)){
                                fprintf(stderr, "[Error] Pool::bgj2_Sieve: erase uid of non inserted vector failed, ignored.\n");
                            }
                        } else {
                            float *dst = to_insert_vec + ind * vec_size;
                            short *coeff_dst = (short *)(dst - int_bias);
                            short *src1 = (short *)(local_buffer[thread]->ss_list[i*3] - int_bias);
                            short *src2 = (short *)(local_buffer[thread]->ss_list[i*3+1] - int_bias);
                            short *src3 = (short *)(local_buffer[thread]->ss_list[i*3+2] - int_bias);
                            add(coeff_dst, src1, coeff_size);
                            sub(coeff_dst, src2, coeff_size);
                            sub(coeff_dst, src3, coeff_size);
                            compute_vec(dst);
                            ind++;
                        }
                    }
                }
            }
            
            if (num_empty >= 0) {
                long insert_to_empty = (num_empty > num_total_insert) ? num_total_insert : num_empty;
                #pragma omp parallel for 
                for (long i = 0; i < insert_to_empty; i++){
                    long *cdst = cvec + cvec_size * (num_vec + i);
                    float *dst = vec + vec_size * (num_vec + i);
                    float *src = to_insert_vec + (num_total_insert - i - 1) * vec_size;
                    if (!uid->safely_check_uid(*((uint64_t *)(&(src[-4]))))){
                        fprintf(stderr, "[Error] Pool::bgj2_Sieve: uid of new pool vector not in the UidHashTable, ignored.\n");
                    }
                    copy(dst - vec_size + vec_length, src - vec_size + vec_length, vec_size);
                    *((uint64_t *)(&cdst[0])) = *((uint64_t *)(&dst[-16]));
                    *((uint64_t *)(&cdst[1])) = *((uint64_t *)(&dst[-14]));
                    *((uint64_t *)(&cdst[2])) = *((uint64_t *)(&dst[-12]));
                    *((uint64_t *)(&cdst[3])) = *((uint64_t *)(&dst[-10]));
                    *((float *)(&cdst[4])) = dst[-1];
                    *((float **)(&cdst[5])) = dst;
                }
                num_vec += insert_to_empty;
                num_total_insert -= insert_to_empty;
                num_empty -= insert_to_empty;
            }


            #pragma omp parallel for
            for (long i = 0; i < num_total_insert; i++){
                long *cdst = cvec + cvec_size * (sorted_index-i-1);
                float *dst = (float *)cdst[5];
                float *src = to_insert_vec + i * vec_size;
                if (!uid->erase_uid(*((uint64_t *)(&(dst[-4]))))){
                    fprintf(stderr, "[Error] Pool::bgj2_Sieve: erase uid of old pool vector failed, ignored.\n");
                }
                if (!uid->safely_check_uid(*((uint64_t *)(&(src[-4]))))){
                    fprintf(stderr, "[Error] Pool::bgj2_Sieve: uid of new pool vector not in the UidHashTable, ignored.\n");
                }
                copy(dst - vec_size + vec_length, src - vec_size + vec_length, vec_size);

                *((uint64_t *)(&cdst[0])) = *((uint64_t *)(&dst[-16]));
                *((uint64_t *)(&cdst[1])) = *((uint64_t *)(&dst[-14]));
                *((uint64_t *)(&cdst[2])) = *((uint64_t *)(&dst[-12]));
                *((uint64_t *)(&cdst[3])) = *((uint64_t *)(&dst[-10]));
                *((float *)(&cdst[4])) = dst[-1];
            }

            sorted_index = sorted_index - num_total_insert;
            FREE_VEC(to_insert_vec_store);
        } while (0);
        TIMER_END;
        insert_time += CURRENT_TIME;

        if (log_level >= 2 && CSD > MIN_LOG_CSD){
            dlog("insert %ld solutions in %fs", num_total_sol, CURRENT_TIME);
        }
        if (log_level == 0 && CSD > MIN_LOG_CSD){
            fprintf(stderr, ".");
        }

        if (resort_ratio * num_vec > sorted_index){
            TIMER_START;
            sort_cvec();
            TIMER_END;
            sort_time += CURRENT_TIME;
        }
        
        for (long i = 0; i < num_threads; i++) delete local_buffer[i];
    }


    mallopt(M_MMAP_MAX, 65536);
    mallopt(M_TRIM_THRESHOLD, 128*1024);
    for (long i = 0; i < num_threads; i++){
        free(global_bucket0_cdp_buffer[i]);
    }


    if (log_level >= 1) {
        if (sieving_stucked){
            dlog("sieving stucked, aborted.");
        } else {
            dlog("sieving done.");
        }
        double speed0, speed1;
        dlog("bucket-0 time = %fs, bucket-1 time = %fs, search-0 time = %fs, search-1 time = %fs", bucket0_time, bucket1_time, search0_time, search1_time);
        dlog("combine time = %fs, sort time = %fs, insert time = %fs", combine_time, sort_time, insert_time);
        speed0 = CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0;
        speed1 = CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0;
        dlog("bucket0 speed: %f bucket/s, %f GFLOPS", num_bucket0/bucket0_time, speed0);
        dlog("bucket1 speed: %f bucket/s, %f GFLOPS", num_bucket1/bucket1_time, speed1);        
        speed0 = CSD * 2 * search0_ndp/search0_time/1073741824.0;
        speed1 = CSD * 2 * search1_ndp/search1_time/1073741824.0;
        dlog("search0 speed: %f bucket/s, %f GFLOPS", num_bucket0/search0_time, speed0);
        dlog("search1 speed: %f bucket/s, %f GFLOPS", num_bucket1/search1_time, speed1);
        dlog("num bucket0 = %ld, avg bucket0 size = %ld, num bucket1 = %ld, avg bucket1 size = %ld", num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)), num_bucket1, (long)(sum_bucket1_size/(0.000001+num_bucket1)));
        dlog("bucket cost = %ld dp/sol, search cost = %ld dp/sol", (long)((bucket0_ndp+bucket1_ndp)/(succ_add2 + succ_add3 + 0.000001)), (long)((search0_ndp+search1_ndp)/(succ_add2 + succ_add3+0.000001)));
        dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld\n", try_add2, try_add3, succ_add2, succ_add3);
    }
    if (log_level == 0){
        gettimeofday(&bgj2_end_time, NULL);
        double tt = bgj2_end_time.tv_sec-bgj2_start_time.tv_sec+ (double)(bgj2_end_time.tv_usec-bgj2_start_time.tv_usec)/1000000.0;
        if (sieving_stucked) {
            fprintf(stderr, "get stucked.");
        }
        if (CSD > MIN_LOG_CSD) fprintf(stderr, "done, time = %.2fs\n", tt);
    }
    return sieving_stucked;
}

int Pool::bgj3_Sieve(long log_level, long lps_auto_adj, long num_empty){
    return 0;
}
