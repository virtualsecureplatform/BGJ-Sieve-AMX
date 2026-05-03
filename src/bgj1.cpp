/** 
 * \warning: old codes based on float32, do not read it
*/
#include "../include/pool.h"
#include <sys/time.h>

#include <immintrin.h>
#include <x86intrin.h>

#define likely(x)   __builtin_expect(!!(x),1)
#define unlikely(x) __builtin_expect(!!(x),0)

#if 1
#define dlog(_format, ...) do {                             \
    printf("[thread %d] ", omp_get_thread_num());           \
    printf(_format "\n", ##__VA_ARGS__);                    \
} while (0)
    
struct timeval _bgj1_timer_start[16], _bgj1_timer_end[16];
double _bgj1_time_curr[16];

#define TIMER_START do {                                                \
        gettimeofday(&_bgj1_timer_start[omp_get_thread_num()], NULL);        \
    } while (0)

#define TIMER_END do {                                                                                              \
        gettimeofday(&_bgj1_timer_end[omp_get_thread_num()], NULL);                                                      \
        _bgj1_time_curr[omp_get_thread_num()] =                                                                          \
            (_bgj1_timer_end[omp_get_thread_num()].tv_sec-_bgj1_timer_start[omp_get_thread_num()].tv_sec)+                    \
            (double)(_bgj1_timer_end[omp_get_thread_num()].tv_usec-_bgj1_timer_start[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_bgj1_time_curr[omp_get_thread_num()])
#endif

#define COLLECT_THREE_RED 1

// on my machine (Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz), even if 
// we process 0.5G simhash dot per second, the simhash based bucketing 
// will not outperform the current dot product based one (taking into 
// account the bucketing quality), so we will not further work on it 
// now. The current performance is about 0.177G simhash dot per second, 
// about half of the theoritical peak performance. And in the following 
// implementation of bgj2 and bgj3, we will not consider it for bucketing.
#define BUCKETING_USE_SIMHASH 0
// I only done part of np_search for simhash based search, so this option 
// should not turn on. If we only collect 2-red, and only do np_search, 
// simhash based implementation (570 buckets per second) will be faster
// than naive implementation (430 buckets per second). But in the naive
// implementation, we can get about 27% more solutions from searching 
// 3-red. And to collect these 3-red relations in simhash based 
// implementation, we need to do more dot product and the speed down to 
// 360 buckets per second. Though we may get speed up from simhash 
// by finetuning the parameters / in larger dimensions, we don't expect 
// a large gain and will not waste more time on it. In the following 
// implementations of bgj2/bgj3 based on float or integers, we will not 
// consider to use the simhash.
#define SEARCHING_USE_SIMHASH 0
#if SEARCHING_USE_SIMHASH
#error simhash based searching not implemneted yet
#endif

#define XPC_BGJ1_THRESHOLD 96
#define XPC_BGJ1_TRD_THRESHOLD 140
#define XPC_BGJ1_BUCKET_THRESHOLD 102

#define BGJ1_BUCKET_ALPHA 0.325
#define BGJ1_USE_FARAWAY_CENTER 1

#define BUCKETING_BATCH_SIZE 144
#if BUCKETING_USE_SIMHASH && (BUCKETING_BATCH_SIZE % 4)
#error bucketing batchsize should divided by 4 when bucketing use simhash
#endif
#define BUCKETING_DP_BLOCK 128
#define BUCKETING_DP_BUFFER 512

#define SEARCHING_DP_BLOCK 256
#define SEARCHING_SH_BLOCK 256

#define MAX_NTHREADS 16
#define MIN_LOG_CSD 40

#pragma region
/** 
 * \brief compute the dot product of __ptr and __srci, and store 
 *      the result in __dsti for i = 0, 1, 2, 3.
 * \param[in] __src0, __src1, __src2, __src3    a pointer to aligened float vectors of length __len
 * \param[in] __ptr                             a pointer to aligened float vectors of length __len
 * \param[out] __dsti                           a float store the dot product of __srcci and __ptr
*/
#define AVX2_DOT_PRODUCT_1X4(__ptr, __src0, __src1, __src2, __src3, __len) do {                     \
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
 * \brief compute the dot product of __ci and __ptrj, and store 
 *      the result in __dstij for i, j = 0, 1, 2, 3.
 * \param[in] __c0, __c1, __c2, __c3            a pointer to aligened float vectors of length __len
 * \param[in] __ptr0, __ptr1, __ptr2, __ptr3    a pointer to aligened float vectors of length __len
 * \param[out] __dstij                          a float store the dot product of __ci and __ptrj
*/
#define AVX2_DOT_PRODUCT_4X4(__c0, __c1, __c2, __c3, __ptr0, __ptr1, __ptr2, __ptr3, __len) do {    \
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
    __m128 __r00_128 = _mm_add_ps(_mm256_castps256_ps128(__r00), _mm256_extractf128_ps(__r00, 1));  \
    __m128 __r01_128 = _mm_add_ps(_mm256_castps256_ps128(__r01), _mm256_extractf128_ps(__r01, 1));  \
    __m128 __r02_128 = _mm_add_ps(_mm256_castps256_ps128(__r02), _mm256_extractf128_ps(__r02, 1));  \
    __m128 __r03_128 = _mm_add_ps(_mm256_castps256_ps128(__r03), _mm256_extractf128_ps(__r03, 1));  \
    __m128 __r10_128 = _mm_add_ps(_mm256_castps256_ps128(__r10), _mm256_extractf128_ps(__r10, 1));  \
    __m128 __r11_128 = _mm_add_ps(_mm256_castps256_ps128(__r11), _mm256_extractf128_ps(__r11, 1));  \
    __m128 __r12_128 = _mm_add_ps(_mm256_castps256_ps128(__r12), _mm256_extractf128_ps(__r12, 1));  \
    __m128 __r13_128 = _mm_add_ps(_mm256_castps256_ps128(__r13), _mm256_extractf128_ps(__r13, 1));  \
    __m128 __r20_128 = _mm_add_ps(_mm256_castps256_ps128(__r20), _mm256_extractf128_ps(__r20, 1));  \
    __m128 __r21_128 = _mm_add_ps(_mm256_castps256_ps128(__r21), _mm256_extractf128_ps(__r21, 1));  \
    __m128 __r22_128 = _mm_add_ps(_mm256_castps256_ps128(__r22), _mm256_extractf128_ps(__r22, 1));  \
    __m128 __r23_128 = _mm_add_ps(_mm256_castps256_ps128(__r23), _mm256_extractf128_ps(__r23, 1));  \
    __m128 __r30_128 = _mm_add_ps(_mm256_castps256_ps128(__r30), _mm256_extractf128_ps(__r30, 1));  \
    __m128 __r31_128 = _mm_add_ps(_mm256_castps256_ps128(__r31), _mm256_extractf128_ps(__r31, 1));  \
    __m128 __r32_128 = _mm_add_ps(_mm256_castps256_ps128(__r32), _mm256_extractf128_ps(__r32, 1));  \
    __m128 __r33_128 = _mm_add_ps(_mm256_castps256_ps128(__r33), _mm256_extractf128_ps(__r33, 1));  \
    __r00_128 = _mm_add_ps(__r00_128, _mm_permute_ps(__r00_128, 78));                               \
    __r01_128 = _mm_add_ps(__r01_128, _mm_permute_ps(__r01_128, 78));                               \
    __r02_128 = _mm_add_ps(__r02_128, _mm_permute_ps(__r02_128, 78));                               \
    __r03_128 = _mm_add_ps(__r03_128, _mm_permute_ps(__r03_128, 78));                               \
    __r10_128 = _mm_add_ps(__r10_128, _mm_permute_ps(__r10_128, 78));                               \
    __r11_128 = _mm_add_ps(__r11_128, _mm_permute_ps(__r11_128, 78));                               \
    __r12_128 = _mm_add_ps(__r12_128, _mm_permute_ps(__r12_128, 78));                               \
    __r13_128 = _mm_add_ps(__r13_128, _mm_permute_ps(__r13_128, 78));                               \
    __r20_128 = _mm_add_ps(__r20_128, _mm_permute_ps(__r20_128, 78));                               \
    __r21_128 = _mm_add_ps(__r21_128, _mm_permute_ps(__r21_128, 78));                               \
    __r22_128 = _mm_add_ps(__r22_128, _mm_permute_ps(__r22_128, 78));                               \
    __r23_128 = _mm_add_ps(__r23_128, _mm_permute_ps(__r23_128, 78));                               \
    __r30_128 = _mm_add_ps(__r30_128, _mm_permute_ps(__r30_128, 78));                               \
    __r31_128 = _mm_add_ps(__r31_128, _mm_permute_ps(__r31_128, 78));                               \
    __r32_128 = _mm_add_ps(__r32_128, _mm_permute_ps(__r32_128, 78));                               \
    __r33_128 = _mm_add_ps(__r33_128, _mm_permute_ps(__r33_128, 78));                               \
    __r00_128 = _mm_add_ps(__r00_128, _mm_shuffle_ps(__r00_128, __r00_128, 85));                    \
    __r01_128 = _mm_add_ps(__r01_128, _mm_shuffle_ps(__r01_128, __r01_128, 85));                    \
    __r02_128 = _mm_add_ps(__r02_128, _mm_shuffle_ps(__r02_128, __r02_128, 85));                    \
    __r03_128 = _mm_add_ps(__r03_128, _mm_shuffle_ps(__r03_128, __r03_128, 85));                    \
    __r10_128 = _mm_add_ps(__r10_128, _mm_shuffle_ps(__r10_128, __r10_128, 85));                    \
    __r11_128 = _mm_add_ps(__r11_128, _mm_shuffle_ps(__r11_128, __r11_128, 85));                    \
    __r12_128 = _mm_add_ps(__r12_128, _mm_shuffle_ps(__r12_128, __r12_128, 85));                    \
    __r13_128 = _mm_add_ps(__r13_128, _mm_shuffle_ps(__r13_128, __r13_128, 85));                    \
    __r20_128 = _mm_add_ps(__r20_128, _mm_shuffle_ps(__r20_128, __r20_128, 85));                    \
    __r21_128 = _mm_add_ps(__r21_128, _mm_shuffle_ps(__r21_128, __r21_128, 85));                    \
    __r22_128 = _mm_add_ps(__r22_128, _mm_shuffle_ps(__r22_128, __r22_128, 85));                    \
    __r23_128 = _mm_add_ps(__r23_128, _mm_shuffle_ps(__r23_128, __r23_128, 85));                    \
    __r30_128 = _mm_add_ps(__r30_128, _mm_shuffle_ps(__r30_128, __r30_128, 85));                    \
    __r31_128 = _mm_add_ps(__r31_128, _mm_shuffle_ps(__r31_128, __r31_128, 85));                    \
    __r32_128 = _mm_add_ps(__r32_128, _mm_shuffle_ps(__r32_128, __r32_128, 85));                    \
    __r33_128 = _mm_add_ps(__r33_128, _mm_shuffle_ps(__r33_128, __r33_128, 85));                    \
    __dst00 = _mm_cvtss_f32(__r00_128);                                                             \
    __dst01 = _mm_cvtss_f32(__r01_128);                                                             \
    __dst02 = _mm_cvtss_f32(__r02_128);                                                             \
    __dst03 = _mm_cvtss_f32(__r03_128);                                                             \
    __dst10 = _mm_cvtss_f32(__r10_128);                                                             \
    __dst11 = _mm_cvtss_f32(__r11_128);                                                             \
    __dst12 = _mm_cvtss_f32(__r12_128);                                                             \
    __dst13 = _mm_cvtss_f32(__r13_128);                                                             \
    __dst20 = _mm_cvtss_f32(__r20_128);                                                             \
    __dst21 = _mm_cvtss_f32(__r21_128);                                                             \
    __dst22 = _mm_cvtss_f32(__r22_128);                                                             \
    __dst23 = _mm_cvtss_f32(__r23_128);                                                             \
    __dst30 = _mm_cvtss_f32(__r30_128);                                                             \
    __dst31 = _mm_cvtss_f32(__r31_128);                                                             \
    __dst32 = _mm_cvtss_f32(__r32_128);                                                             \
    __dst33 = _mm_cvtss_f32(__r33_128);                                                             \
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
 * \brief compute the dot product of __c0 and __pi, store the result in a __m256 register, __dst.
*/
#define AVX2_DP_1X8(__c0, __p0, __p1, __p2, __p3, __p4, __p5, __p6, __p7, __len, __dst) do {        \
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
 * \brief compute the dot product of __c0, __c1 and 8 vectors __ptr[0], ..., __ptr[7], 
 *      store the result in 2 __m256 register, __dst0, __dst1.
*/
#define AVX2_DP_2X8(__c0, __c1, __ptr, __len, __dst0, __dst1) do {                                  \
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
 * \brief compute the dot product of __ci and __ptri, and store 
 *      the result in __dsti for i, j = 0, 1, 2, 3, 4, 5, 6, 7.
 * \param[in] __ci                              a pointer to aligened float vectors of length __len
 * \param[in] __ptri                            a pointer to aligened float vectors of length __len
 * \param[out] __dsti                           a float store the dot product of __ci and __ptri
*/
#define AVX2_DOT_PRODUCT_B8(__ptr0, __ptr1, __ptr2, __ptr3,                                         \
                            __ptr4, __ptr5, __ptr6, __ptr7,                                         \
                            __c0, __c1, __c2, __c3, __c4,                                           \
                            __c5, __c6, __c7, __len) do {                                           \
    __m256 __r0 = _mm256_setzero_ps();                                                              \
    __m256 __r1 = _mm256_setzero_ps();                                                              \
    __m256 __r2 = _mm256_setzero_ps();                                                              \
    __m256 __r3 = _mm256_setzero_ps();                                                              \
    __m256 __r4 = _mm256_setzero_ps();                                                              \
    __m256 __r5 = _mm256_setzero_ps();                                                              \
    __m256 __r6 = _mm256_setzero_ps();                                                              \
    __m256 __r7 = _mm256_setzero_ps();                                                              \
    long __i;                                                                                       \
    for (__i = 0; __i < __len - 15; __i += 16){                                                     \
        __r0 = _mm256_fmadd_ps(_mm256_load_ps(__ptr0+__i), _mm256_load_ps(__c0+__i), __r0);         \
        __r1 = _mm256_fmadd_ps(_mm256_load_ps(__ptr1+__i), _mm256_load_ps(__c1+__i), __r1);         \
        __r2 = _mm256_fmadd_ps(_mm256_load_ps(__ptr2+__i), _mm256_load_ps(__c2+__i), __r2);         \
        __r3 = _mm256_fmadd_ps(_mm256_load_ps(__ptr3+__i), _mm256_load_ps(__c3+__i), __r3);         \
        __r4 = _mm256_fmadd_ps(_mm256_load_ps(__ptr4+__i), _mm256_load_ps(__c4+__i), __r4);         \
        __r5 = _mm256_fmadd_ps(_mm256_load_ps(__ptr5+__i), _mm256_load_ps(__c5+__i), __r5);         \
        __r6 = _mm256_fmadd_ps(_mm256_load_ps(__ptr6+__i), _mm256_load_ps(__c6+__i), __r6);         \
        __r7 = _mm256_fmadd_ps(_mm256_load_ps(__ptr7+__i), _mm256_load_ps(__c7+__i), __r7);         \
        __r0 = _mm256_fmadd_ps(_mm256_load_ps(__ptr0+__i+8), _mm256_load_ps(__c0+__i+8), __r0);     \
        __r1 = _mm256_fmadd_ps(_mm256_load_ps(__ptr1+__i+8), _mm256_load_ps(__c1+__i+8), __r1);     \
        __r2 = _mm256_fmadd_ps(_mm256_load_ps(__ptr2+__i+8), _mm256_load_ps(__c2+__i+8), __r2);     \
        __r3 = _mm256_fmadd_ps(_mm256_load_ps(__ptr3+__i+8), _mm256_load_ps(__c3+__i+8), __r3);     \
        __r4 = _mm256_fmadd_ps(_mm256_load_ps(__ptr4+__i+8), _mm256_load_ps(__c4+__i+8), __r4);     \
        __r5 = _mm256_fmadd_ps(_mm256_load_ps(__ptr5+__i+8), _mm256_load_ps(__c5+__i+8), __r5);     \
        __r6 = _mm256_fmadd_ps(_mm256_load_ps(__ptr6+__i+8), _mm256_load_ps(__c6+__i+8), __r6);     \
        __r7 = _mm256_fmadd_ps(_mm256_load_ps(__ptr7+__i+8), _mm256_load_ps(__c7+__i+8), __r7);     \
    }                                                                                               \
    if (__i < __len - 7) {                                                                          \
        __r0 = _mm256_fmadd_ps(_mm256_load_ps(__ptr0+__i), _mm256_load_ps(__c0+__i), __r0);         \
        __r1 = _mm256_fmadd_ps(_mm256_load_ps(__ptr1+__i), _mm256_load_ps(__c1+__i), __r1);         \
        __r2 = _mm256_fmadd_ps(_mm256_load_ps(__ptr2+__i), _mm256_load_ps(__c2+__i), __r2);         \
        __r3 = _mm256_fmadd_ps(_mm256_load_ps(__ptr3+__i), _mm256_load_ps(__c3+__i), __r3);         \
        __r4 = _mm256_fmadd_ps(_mm256_load_ps(__ptr4+__i), _mm256_load_ps(__c4+__i), __r4);         \
        __r5 = _mm256_fmadd_ps(_mm256_load_ps(__ptr5+__i), _mm256_load_ps(__c5+__i), __r5);         \
        __r6 = _mm256_fmadd_ps(_mm256_load_ps(__ptr6+__i), _mm256_load_ps(__c6+__i), __r6);         \
        __r7 = _mm256_fmadd_ps(_mm256_load_ps(__ptr7+__i), _mm256_load_ps(__c7+__i), __r7);         \
    }                                                                                               \
    __m128 __r0_128 = _mm_add_ps(_mm256_castps256_ps128(__r0), _mm256_extractf128_ps(__r0, 1));     \
    __m128 __r1_128 = _mm_add_ps(_mm256_castps256_ps128(__r1), _mm256_extractf128_ps(__r1, 1));     \
    __m128 __r2_128 = _mm_add_ps(_mm256_castps256_ps128(__r2), _mm256_extractf128_ps(__r2, 1));     \
    __m128 __r3_128 = _mm_add_ps(_mm256_castps256_ps128(__r3), _mm256_extractf128_ps(__r3, 1));     \
    __m128 __r4_128 = _mm_add_ps(_mm256_castps256_ps128(__r4), _mm256_extractf128_ps(__r4, 1));     \
    __m128 __r5_128 = _mm_add_ps(_mm256_castps256_ps128(__r5), _mm256_extractf128_ps(__r5, 1));     \
    __m128 __r6_128 = _mm_add_ps(_mm256_castps256_ps128(__r6), _mm256_extractf128_ps(__r6, 1));     \
    __m128 __r7_128 = _mm_add_ps(_mm256_castps256_ps128(__r7), _mm256_extractf128_ps(__r7, 1));     \
    __r0_128 = _mm_add_ps(__r0_128, _mm_permute_ps(__r0_128, 78));                                  \
    __r1_128 = _mm_add_ps(__r1_128, _mm_permute_ps(__r1_128, 78));                                  \
    __r2_128 = _mm_add_ps(__r2_128, _mm_permute_ps(__r2_128, 78));                                  \
    __r3_128 = _mm_add_ps(__r3_128, _mm_permute_ps(__r3_128, 78));                                  \
    __r4_128 = _mm_add_ps(__r4_128, _mm_permute_ps(__r4_128, 78));                                  \
    __r5_128 = _mm_add_ps(__r5_128, _mm_permute_ps(__r5_128, 78));                                  \
    __r6_128 = _mm_add_ps(__r6_128, _mm_permute_ps(__r6_128, 78));                                  \
    __r7_128 = _mm_add_ps(__r7_128, _mm_permute_ps(__r7_128, 78));                                  \
    __r0_128 = _mm_add_ps(__r0_128, _mm_shuffle_ps(__r0_128, __r0_128, 85));                        \
    __r1_128 = _mm_add_ps(__r1_128, _mm_shuffle_ps(__r1_128, __r1_128, 85));                        \
    __r2_128 = _mm_add_ps(__r2_128, _mm_shuffle_ps(__r2_128, __r2_128, 85));                        \
    __r3_128 = _mm_add_ps(__r3_128, _mm_shuffle_ps(__r3_128, __r3_128, 85));                        \
    __r4_128 = _mm_add_ps(__r4_128, _mm_shuffle_ps(__r4_128, __r4_128, 85));                        \
    __r5_128 = _mm_add_ps(__r5_128, _mm_shuffle_ps(__r5_128, __r5_128, 85));                        \
    __r6_128 = _mm_add_ps(__r6_128, _mm_shuffle_ps(__r6_128, __r6_128, 85));                        \
    __r7_128 = _mm_add_ps(__r7_128, _mm_shuffle_ps(__r7_128, __r7_128, 85));                        \
    __dst0 = _mm_cvtss_f32(__r0_128);                                                               \
    __dst1 = _mm_cvtss_f32(__r1_128);                                                               \
    __dst2 = _mm_cvtss_f32(__r2_128);                                                               \
    __dst3 = _mm_cvtss_f32(__r3_128);                                                               \
    __dst4 = _mm_cvtss_f32(__r4_128);                                                               \
    __dst5 = _mm_cvtss_f32(__r5_128);                                                               \
    __dst6 = _mm_cvtss_f32(__r6_128);                                                               \
    __dst7 = _mm_cvtss_f32(__r7_128);                                                               \
} while (0)


#define POPCNT64_BATCHX4(__dst, __csh, __sh0, __sh1, __sh2, __sh3) do { \
    for (long __i = 0; __i < BUCKETING_BATCH_SIZE; __i++){              \
        __dst[__i*4+0] += _popcnt64(__sh0[0]^__csh[__i*4 + 0]);         \
        __dst[__i*4+0] += _popcnt64(__sh0[1]^__csh[__i*4 + 1]);         \
        __dst[__i*4+0] += _popcnt64(__sh0[2]^__csh[__i*4 + 2]);         \
        __dst[__i*4+0] += _popcnt64(__sh0[3]^__csh[__i*4 + 3]);         \
        __dst[__i*4+1] += _popcnt64(__sh1[0]^__csh[__i*4 + 0]);         \
        __dst[__i*4+1] += _popcnt64(__sh1[1]^__csh[__i*4 + 1]);         \
        __dst[__i*4+1] += _popcnt64(__sh1[2]^__csh[__i*4 + 2]);         \
        __dst[__i*4+1] += _popcnt64(__sh1[3]^__csh[__i*4 + 3]);         \
        __dst[__i*4+2] += _popcnt64(__sh2[0]^__csh[__i*4 + 0]);         \
        __dst[__i*4+2] += _popcnt64(__sh2[1]^__csh[__i*4 + 1]);         \
        __dst[__i*4+2] += _popcnt64(__sh2[2]^__csh[__i*4 + 2]);         \
        __dst[__i*4+2] += _popcnt64(__sh2[3]^__csh[__i*4 + 3]);         \
        __dst[__i*4+3] += _popcnt64(__sh3[0]^__csh[__i*4 + 0]);         \
        __dst[__i*4+3] += _popcnt64(__sh3[1]^__csh[__i*4 + 1]);         \
        __dst[__i*4+3] += _popcnt64(__sh3[2]^__csh[__i*4 + 2]);         \
        __dst[__i*4+3] += _popcnt64(__sh3[3]^__csh[__i*4 + 3]);         \
    }                                                                   \
} while (0)

#define POPCNT64_BATCHX1(__dst, __csh, __sh0) do {                      \
    long __i = 0;                                                       \
    while (__i < BUCKETING_BATCH_SIZE - 3) {                            \
        __dst[__i+0] += _popcnt64(__sh0[0]^__csh[__i*4+0]);             \
        __dst[__i+0] += _popcnt64(__sh0[1]^__csh[__i*4+1]);             \
        __dst[__i+0] += _popcnt64(__sh0[2]^__csh[__i*4+2]);             \
        __dst[__i+0] += _popcnt64(__sh0[3]^__csh[__i*4+3]);             \
        __dst[__i+1] += _popcnt64(__sh0[0]^__csh[__i*4+4]);             \
        __dst[__i+1] += _popcnt64(__sh0[1]^__csh[__i*4+5]);             \
        __dst[__i+1] += _popcnt64(__sh0[2]^__csh[__i*4+6]);             \
        __dst[__i+1] += _popcnt64(__sh0[3]^__csh[__i*4+7]);             \
        __dst[__i+2] += _popcnt64(__sh0[0]^__csh[__i*4+8]);             \
        __dst[__i+2] += _popcnt64(__sh0[1]^__csh[__i*4+9]);             \
        __dst[__i+2] += _popcnt64(__sh0[2]^__csh[__i*4+10]);            \
        __dst[__i+2] += _popcnt64(__sh0[3]^__csh[__i*4+11]);            \
        __dst[__i+3] += _popcnt64(__sh0[0]^__csh[__i*4+12]);            \
        __dst[__i+3] += _popcnt64(__sh0[1]^__csh[__i*4+13]);            \
        __dst[__i+3] += _popcnt64(__sh0[2]^__csh[__i*4+14]);            \
        __dst[__i+3] += _popcnt64(__sh0[3]^__csh[__i*4+15]);            \
        __i += 4;                                                       \
    }                                                                   \
    while (__i < BUCKETING_BATCH_SIZE) {                                \
        __dst[__i+0] += _popcnt64(__sh0[0]^__csh[__i*4+0]);             \
        __dst[__i+0] += _popcnt64(__sh0[1]^__csh[__i*4+1]);             \
        __dst[__i+0] += _popcnt64(__sh0[2]^__csh[__i*4+2]);             \
        __dst[__i+0] += _popcnt64(__sh0[3]^__csh[__i*4+3]);             \
        __i++;                                                          \
    }                                                                   \
} while (0)


#define AVX2_POPCNT64_BATCHX4(__dst, __csh, __sh0, __sh1, __sh2, __sh3, __ub, __lb) do {                            \
    __m256i __lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3,                                                     \
                                      1, 2, 2, 3, 2, 3, 3, 4,                                                       \
                                      0, 1, 1, 2, 1, 2, 2, 3,                                                       \
                                      1, 2, 2, 3, 2, 3, 3, 4);                                                      \
    __m256i __low_mask = _mm256_set1_epi8(0x0f);                                                                    \
    __m256i __zero = _mm256_setzero_si256();                                                                        \
                                                                                                                    \
    for (long __i = 0; __i < BUCKETING_BATCH_SIZE; __i += 4){                                                       \
        __m256i __csh0 = _mm256_load_si256((__m256i *)(&__csh[__i*4+0]));                                           \
        __m256i __csh1 = _mm256_load_si256((__m256i *)(&__csh[__i*4+4]));                                           \
        __m256i __csh2 = _mm256_load_si256((__m256i *)(&__csh[__i*4+8]));                                           \
        __m256i __csh3 = _mm256_load_si256((__m256i *)(&__csh[__i*4+12]));                                          \
                                                                                                                    \
        __m256i __x00 = _mm256_xor_si256(__sh0, __csh0);                                                            \
        __m256i __x01 = _mm256_xor_si256(__sh0, __csh1);                                                            \
        __m256i __x02 = _mm256_xor_si256(__sh0, __csh2);                                                            \
        __m256i __x03 = _mm256_xor_si256(__sh0, __csh3);                                                            \
        __m256i __x10 = _mm256_xor_si256(__sh1, __csh0);                                                            \
        __m256i __x11 = _mm256_xor_si256(__sh1, __csh1);                                                            \
        __m256i __x12 = _mm256_xor_si256(__sh1, __csh2);                                                            \
        __m256i __x13 = _mm256_xor_si256(__sh1, __csh3);                                                            \
        __m256i __x20 = _mm256_xor_si256(__sh2, __csh0);                                                            \
        __m256i __x21 = _mm256_xor_si256(__sh2, __csh1);                                                            \
        __m256i __x22 = _mm256_xor_si256(__sh2, __csh2);                                                            \
        __m256i __x23 = _mm256_xor_si256(__sh2, __csh3);                                                            \
        __m256i __x30 = _mm256_xor_si256(__sh3, __csh0);                                                            \
        __m256i __x31 = _mm256_xor_si256(__sh3, __csh1);                                                            \
        __m256i __x32 = _mm256_xor_si256(__sh3, __csh2);                                                            \
        __m256i __x33 = _mm256_xor_si256(__sh3, __csh3);                                                            \
                                                                                                                    \
        __m256i __r00lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x00, __low_mask));                       \
        __m256i __r01lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x01, __low_mask));                       \
        __m256i __r02lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x02, __low_mask));                       \
        __m256i __r03lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x03, __low_mask));                       \
        __m256i __r10lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x10, __low_mask));                       \
        __m256i __r11lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x11, __low_mask));                       \
        __m256i __r12lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x12, __low_mask));                       \
        __m256i __r13lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x13, __low_mask));                       \
        __m256i __r20lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x20, __low_mask));                       \
        __m256i __r21lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x21, __low_mask));                       \
        __m256i __r22lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x22, __low_mask));                       \
        __m256i __r23lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x23, __low_mask));                       \
        __m256i __r30lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x30, __low_mask));                       \
        __m256i __r31lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x31, __low_mask));                       \
        __m256i __r32lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x32, __low_mask));                       \
        __m256i __r33lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x33, __low_mask));                       \
                                                                                                                    \
        __m256i __r00hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x00, 4), __low_mask)); \
        __m256i __r01hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x01, 4), __low_mask)); \
        __m256i __r02hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x02, 4), __low_mask)); \
        __m256i __r03hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x03, 4), __low_mask)); \
        __m256i __r10hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x10, 4), __low_mask)); \
        __m256i __r11hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x11, 4), __low_mask)); \
        __m256i __r12hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x12, 4), __low_mask)); \
        __m256i __r13hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x13, 4), __low_mask)); \
        __m256i __r20hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x20, 4), __low_mask)); \
        __m256i __r21hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x21, 4), __low_mask)); \
        __m256i __r22hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x22, 4), __low_mask)); \
        __m256i __r23hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x23, 4), __low_mask)); \
        __m256i __r30hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x30, 4), __low_mask)); \
        __m256i __r31hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x31, 4), __low_mask)); \
        __m256i __r32hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x32, 4), __low_mask)); \
        __m256i __r33hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x33, 4), __low_mask)); \
                                                                                                                    \
        __m256i __r00 = _mm256_sad_epu8(_mm256_add_epi8(__r00lo, __r00hi), __zero);                                 \
        __m256i __r01 = _mm256_sad_epu8(_mm256_add_epi8(__r01lo, __r01hi), __zero);                                 \
        __m256i __r02 = _mm256_sad_epu8(_mm256_add_epi8(__r02lo, __r02hi), __zero);                                 \
        __m256i __r03 = _mm256_sad_epu8(_mm256_add_epi8(__r03lo, __r03hi), __zero);                                 \
        __m256i __r10 = _mm256_sad_epu8(_mm256_add_epi8(__r10lo, __r10hi), __zero);                                 \
        __m256i __r11 = _mm256_sad_epu8(_mm256_add_epi8(__r11lo, __r11hi), __zero);                                 \
        __m256i __r12 = _mm256_sad_epu8(_mm256_add_epi8(__r12lo, __r12hi), __zero);                                 \
        __m256i __r13 = _mm256_sad_epu8(_mm256_add_epi8(__r13lo, __r13hi), __zero);                                 \
        __m256i __r20 = _mm256_sad_epu8(_mm256_add_epi8(__r20lo, __r20hi), __zero);                                 \
        __m256i __r21 = _mm256_sad_epu8(_mm256_add_epi8(__r21lo, __r21hi), __zero);                                 \
        __m256i __r22 = _mm256_sad_epu8(_mm256_add_epi8(__r22lo, __r22hi), __zero);                                 \
        __m256i __r23 = _mm256_sad_epu8(_mm256_add_epi8(__r23lo, __r23hi), __zero);                                 \
        __m256i __r30 = _mm256_sad_epu8(_mm256_add_epi8(__r30lo, __r30hi), __zero);                                 \
        __m256i __r31 = _mm256_sad_epu8(_mm256_add_epi8(__r31lo, __r31hi), __zero);                                 \
        __m256i __r32 = _mm256_sad_epu8(_mm256_add_epi8(__r32lo, __r32hi), __zero);                                 \
        __m256i __r33 = _mm256_sad_epu8(_mm256_add_epi8(__r33lo, __r33hi), __zero);                                 \
                                                                                                                    \
        __r00 = _mm256_add_epi16(__r00, _mm256_slli_epi64(__r01, 16));                                              \
        __r02 = _mm256_add_epi16(__r02, _mm256_slli_epi64(__r03, 16));                                              \
        __r10 = _mm256_add_epi16(__r10, _mm256_slli_epi64(__r11, 16));                                              \
        __r12 = _mm256_add_epi16(__r12, _mm256_slli_epi64(__r13, 16));                                              \
        __r20 = _mm256_add_epi16(__r20, _mm256_slli_epi64(__r21, 16));                                              \
        __r22 = _mm256_add_epi16(__r22, _mm256_slli_epi64(__r23, 16));                                              \
        __r30 = _mm256_add_epi16(__r30, _mm256_slli_epi64(__r31, 16));                                              \
        __r32 = _mm256_add_epi16(__r32, _mm256_slli_epi64(__r33, 16));                                              \
                                                                                                                    \
        __r00 = _mm256_add_epi16(__r00, _mm256_slli_epi64(__r02, 32));                                              \
        __r10 = _mm256_add_epi16(__r10, _mm256_slli_epi64(__r12, 32));                                              \
        __r20 = _mm256_add_epi16(__r20, _mm256_slli_epi64(__r22, 32));                                              \
        __r30 = _mm256_add_epi16(__r30, _mm256_slli_epi64(__r32, 32));                                              \
                                                                                                                    \
        __r00 = _mm256_add_epi16(_mm256_shuffle_epi32(__r00, 78), __r00);                                           \
        __r10 = _mm256_add_epi16(_mm256_shuffle_epi32(__r10, 78), __r10);                                           \
        __r20 = _mm256_add_epi16(_mm256_shuffle_epi32(__r20, 78), __r20);                                           \
        __r30 = _mm256_add_epi16(_mm256_shuffle_epi32(__r30, 78), __r30);                                           \
                                                                                                                    \
        __r00 = _mm256_blend_epi32(__r00, __r10, 204);                                                              \
        __r20 = _mm256_blend_epi32(__r20, __r30, 204);                                                              \
        __r00 = _mm256_add_epi16(__r00, _mm256_permute4x64_epi64(__r00, 78));                                       \
        __r20 = _mm256_add_epi16(__r20, _mm256_permute4x64_epi64(__r20, 78));                                       \
        __r00 = _mm256_blend_epi32(__r00, __r20, 240);                                                              \
        __m256i __lc = _mm256_cmpgt_epi16(__lb, __r00);                                                             \
        __m256i __uc = _mm256_cmpgt_epi16(__r00, __ub);                                                             \
        __dst[__i>>2] = _mm256_movemask_epi8(_mm256_or_si256(__lc, __uc));                                          \
    }                                                                                                               \
} while(0)

#define AVX2_POPCNT64_4X4(__dst, __sh0, __sh1, __sh2, __sh3, __csh0, __csh1, __csh2, __csh3, __lookup, __lm) do {   \
    __m256i __zero = _mm256_setzero_si256();                                                                        \
    __m256i __x00 = _mm256_xor_si256(__sh0, __csh0);                                                                \
    __m256i __x01 = _mm256_xor_si256(__sh0, __csh1);                                                                \
    __m256i __x02 = _mm256_xor_si256(__sh0, __csh2);                                                                \
    __m256i __x03 = _mm256_xor_si256(__sh0, __csh3);                                                                \
    __m256i __x10 = _mm256_xor_si256(__sh1, __csh0);                                                                \
    __m256i __x11 = _mm256_xor_si256(__sh1, __csh1);                                                                \
    __m256i __x12 = _mm256_xor_si256(__sh1, __csh2);                                                                \
    __m256i __x13 = _mm256_xor_si256(__sh1, __csh3);                                                                \
    __m256i __x20 = _mm256_xor_si256(__sh2, __csh0);                                                                \
    __m256i __x21 = _mm256_xor_si256(__sh2, __csh1);                                                                \
    __m256i __x22 = _mm256_xor_si256(__sh2, __csh2);                                                                \
    __m256i __x23 = _mm256_xor_si256(__sh2, __csh3);                                                                \
    __m256i __x30 = _mm256_xor_si256(__sh3, __csh0);                                                                \
    __m256i __x31 = _mm256_xor_si256(__sh3, __csh1);                                                                \
    __m256i __x32 = _mm256_xor_si256(__sh3, __csh2);                                                                \
    __m256i __x33 = _mm256_xor_si256(__sh3, __csh3);                                                                \
                                                                                                                    \
    __m256i __r00lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x00, __lm));                                 \
    __m256i __r01lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x01, __lm));                                 \
    __m256i __r02lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x02, __lm));                                 \
    __m256i __r03lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x03, __lm));                                 \
    __m256i __r10lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x10, __lm));                                 \
    __m256i __r11lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x11, __lm));                                 \
    __m256i __r12lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x12, __lm));                                 \
    __m256i __r13lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x13, __lm));                                 \
    __m256i __r20lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x20, __lm));                                 \
    __m256i __r21lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x21, __lm));                                 \
    __m256i __r22lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x22, __lm));                                 \
    __m256i __r23lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x23, __lm));                                 \
    __m256i __r30lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x30, __lm));                                 \
    __m256i __r31lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x31, __lm));                                 \
    __m256i __r32lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x32, __lm));                                 \
    __m256i __r33lo = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(__x33, __lm));                                 \
                                                                                                                    \
    __m256i __r00hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x00, 4), __lm));           \
    __m256i __r01hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x01, 4), __lm));           \
    __m256i __r02hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x02, 4), __lm));           \
    __m256i __r03hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x03, 4), __lm));           \
    __m256i __r10hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x10, 4), __lm));           \
    __m256i __r11hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x11, 4), __lm));           \
    __m256i __r12hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x12, 4), __lm));           \
    __m256i __r13hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x13, 4), __lm));           \
    __m256i __r20hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x20, 4), __lm));           \
    __m256i __r21hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x21, 4), __lm));           \
    __m256i __r22hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x22, 4), __lm));           \
    __m256i __r23hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x23, 4), __lm));           \
    __m256i __r30hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x30, 4), __lm));           \
    __m256i __r31hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x31, 4), __lm));           \
    __m256i __r32hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x32, 4), __lm));           \
    __m256i __r33hi = _mm256_shuffle_epi8(__lookup, _mm256_and_si256(_mm256_srli_epi32(__x33, 4), __lm));           \
                                                                                                                    \
    __m256i __r00 = _mm256_sad_epu8(_mm256_add_epi8(__r00lo, __r00hi), __zero);                                     \
    __m256i __r01 = _mm256_sad_epu8(_mm256_add_epi8(__r01lo, __r01hi), __zero);                                     \
    __m256i __r02 = _mm256_sad_epu8(_mm256_add_epi8(__r02lo, __r02hi), __zero);                                     \
    __m256i __r03 = _mm256_sad_epu8(_mm256_add_epi8(__r03lo, __r03hi), __zero);                                     \
    __m256i __r10 = _mm256_sad_epu8(_mm256_add_epi8(__r10lo, __r10hi), __zero);                                     \
    __m256i __r11 = _mm256_sad_epu8(_mm256_add_epi8(__r11lo, __r11hi), __zero);                                     \
    __m256i __r12 = _mm256_sad_epu8(_mm256_add_epi8(__r12lo, __r12hi), __zero);                                     \
    __m256i __r13 = _mm256_sad_epu8(_mm256_add_epi8(__r13lo, __r13hi), __zero);                                     \
    __m256i __r20 = _mm256_sad_epu8(_mm256_add_epi8(__r20lo, __r20hi), __zero);                                     \
    __m256i __r21 = _mm256_sad_epu8(_mm256_add_epi8(__r21lo, __r21hi), __zero);                                     \
    __m256i __r22 = _mm256_sad_epu8(_mm256_add_epi8(__r22lo, __r22hi), __zero);                                     \
    __m256i __r23 = _mm256_sad_epu8(_mm256_add_epi8(__r23lo, __r23hi), __zero);                                     \
    __m256i __r30 = _mm256_sad_epu8(_mm256_add_epi8(__r30lo, __r30hi), __zero);                                     \
    __m256i __r31 = _mm256_sad_epu8(_mm256_add_epi8(__r31lo, __r31hi), __zero);                                     \
    __m256i __r32 = _mm256_sad_epu8(_mm256_add_epi8(__r32lo, __r32hi), __zero);                                     \
    __m256i __r33 = _mm256_sad_epu8(_mm256_add_epi8(__r33lo, __r33hi), __zero);                                     \
                                                                                                                    \
    __r00 = _mm256_add_epi16(__r00, _mm256_slli_epi64(__r01, 16));                                                  \
    __r02 = _mm256_add_epi16(__r02, _mm256_slli_epi64(__r03, 16));                                                  \
    __r10 = _mm256_add_epi16(__r10, _mm256_slli_epi64(__r11, 16));                                                  \
    __r12 = _mm256_add_epi16(__r12, _mm256_slli_epi64(__r13, 16));                                                  \
    __r20 = _mm256_add_epi16(__r20, _mm256_slli_epi64(__r21, 16));                                                  \
    __r22 = _mm256_add_epi16(__r22, _mm256_slli_epi64(__r23, 16));                                                  \
    __r30 = _mm256_add_epi16(__r30, _mm256_slli_epi64(__r31, 16));                                                  \
    __r32 = _mm256_add_epi16(__r32, _mm256_slli_epi64(__r33, 16));                                                  \
                                                                                                                    \
    __r00 = _mm256_add_epi16(__r00, _mm256_slli_epi64(__r02, 32));                                                  \
    __r10 = _mm256_add_epi16(__r10, _mm256_slli_epi64(__r12, 32));                                                  \
    __r20 = _mm256_add_epi16(__r20, _mm256_slli_epi64(__r22, 32));                                                  \
    __r30 = _mm256_add_epi16(__r30, _mm256_slli_epi64(__r32, 32));                                                  \
                                                                                                                    \
    __r00 = _mm256_add_epi16(_mm256_shuffle_epi32(__r00, 78), __r00);                                               \
    __r10 = _mm256_add_epi16(_mm256_shuffle_epi32(__r10, 78), __r10);                                               \
    __r20 = _mm256_add_epi16(_mm256_shuffle_epi32(__r20, 78), __r20);                                               \
    __r30 = _mm256_add_epi16(_mm256_shuffle_epi32(__r30, 78), __r30);                                               \
                                                                                                                    \
    __r00 = _mm256_blend_epi32(__r00, __r10, 204);                                                                  \
    __r20 = _mm256_blend_epi32(__r20, __r30, 204);                                                                  \
    __r00 = _mm256_add_epi16(__r00, _mm256_permute4x64_epi64(__r00, 78));                                           \
    __r20 = _mm256_add_epi16(__r20, _mm256_permute4x64_epi64(__r20, 78));                                           \
    __dst = _mm256_blend_epi32(__r00, __r20, 240);                                                                  \
} while (0)

#pragma endregion


float bgj1_lps_alpha_adj[4][5] = {
    {0.015, 0.00, 0, 0, 0},           // 60~70
    {0.017, 0.017, 0.012, 0.000, -0.005},           // 70~80
    {0.016, 0.014, 0.011, 0.000, 0.000},           // 80~90
    {0.016, 0.014, 0.011, 0.000, 0.000},           // 90~100
};

struct bgj1_sol_t {
    float **a_list = NULL;
    float **s_list = NULL;
    long num_a = 0, num_s = 0;
    #if COLLECT_THREE_RED
    float **aa_list = NULL;
    float **sa_list = NULL;
    float **ss_list = NULL;
    long num_aa = 0, num_sa = 0, num_ss = 0;
    #endif

    bgj1_sol_t() {}
    bgj1_sol_t(long size2) { 
        _alloc(size2, 0); 
        _alloc(size2, 1);
        #if COLLECT_THREE_RED
        long size3 = size2/5;
        _alloc(size3, 2); 
        _alloc(size3, 3); 
        _alloc(size3, 4);
        #endif
    }
    ~bgj1_sol_t(){ _clear(); }
    
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
    #if COLLECT_THREE_RED
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
    #endif
    long num_sol() {
        #if COLLECT_THREE_RED
        return num_a + num_s + num_aa + num_sa + num_ss;
        #else
        return num_a + num_s;
        #endif
    }

    int _clear(){
        if (a_list) free(a_list);           // 0
        if (s_list) free(s_list);           // 1
        a_list = NULL; s_list = NULL;
        #if COLLECT_THREE_RED
        if (aa_list) free(aa_list);         // 2
        if (sa_list) free(sa_list);         // 3
        if (ss_list) free(ss_list);         // 4
        aa_list = NULL; sa_list = NULL; ss_list = NULL;
        #endif
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
        #if COLLECT_THREE_RED
        if (type == 2) _REALLOC_PTR(aa_list, _aasize, 3);
        if (type == 3) _REALLOC_PTR(sa_list, _sasize, 3);
        if (type == 4) _REALLOC_PTR(ss_list, _sssize, 3);
        #endif
        #undef _REALLOC_PTR
        return -1;
    }


    long _asize = 0, _ssize = 0;
    #if COLLECT_THREE_RED
    long _aasize = 0, _sasize = 0, _sssize = 0;
    #endif
};

template<bool record_dp, bool record_sh>
struct bgj1_bucket_t {
    long num_pvec = 0;
    long num_nvec = 0;

    long try_add2 = 0;
    long succ_add2 = 0;
    long try_add3 = 0;
    long succ_add3 = 0;

    long vec_length;

    float **pvec = NULL;
    float **nvec = NULL;
    uint64_t *pu = NULL;
    uint64_t *nu = NULL;
    float *center = NULL;
    float center_norm;
    uint64_t center_u;
    uint64_t center_sh[4];

    float *pdot = NULL;
    float *ndot = NULL;
    uint64_t *psh = NULL;
    uint64_t *nsh = NULL;
    float *pnorm = NULL;
    float *nnorm = NULL;

    bgj1_bucket_t() {}
    bgj1_bucket_t(long size) { _alloc(size, 0); _alloc(size, 1); }
    ~bgj1_bucket_t(){ _clear(); }

    void set_center(float *ptr){
        if (center == NULL) {
            center = ptr;
            center_norm = ptr[-1];
            center_u = *((uint64_t *)(&ptr[-4]));
            center_sh[0] = *((uint64_t *)(&ptr[-16]));
            center_sh[1] = *((uint64_t *)(&ptr[-14]));
            center_sh[2] = *((uint64_t *)(&ptr[-12]));
            center_sh[3] = *((uint64_t *)(&ptr[-10]));
        }
    }
    inline void add_vec(float *ptr, float _dot, uint64_t u, uint64_t *_sh = NULL){
        if (_dot > 0){
            if (num_pvec == _psize) _alloc(_psize * 2 + 64, 1);
            pvec[num_pvec] = ptr;
            pu[num_pvec] = u;
            //pnorm[num_pvec] = ptr[-1];
            if (record_dp) pdot[num_pvec] = _dot;
            if (record_sh) {
                psh[num_pvec * 4 + 0] = _sh[0];
                psh[num_pvec * 4 + 1] = _sh[1];
                psh[num_pvec * 4 + 2] = _sh[2];
                psh[num_pvec * 4 + 3] = _sh[3];
            }
            num_pvec++;
        } else {
            if (num_nvec == _nsize) _alloc(_nsize * 2 + 64, 0);
            nvec[num_nvec] = ptr;
            nu[num_nvec] = u;
            //nnorm[num_nvec] = ptr[-1];
            if (record_dp) ndot[num_nvec] = _dot;
            if (record_sh) {
                nsh[num_nvec * 4 + 0] = _sh[0];
                nsh[num_nvec * 4 + 1] = _sh[1];
                nsh[num_nvec * 4 + 2] = _sh[2];
                nsh[num_nvec * 4 + 3] = _sh[3];
            }
            num_nvec++;
        }
    }

    void combine(bgj1_bucket_t **subbucket_list, long len){
        if (num_pvec || num_nvec) {
            fprintf(stderr, "[Error] bgj1_bucket_t::combine: combining subbuckets to nonempty buckets, aborted.\n");
            return;
        }
        long total_psize = 0, total_nsize = 0;
        for (long i = 0; i < len; i++){
            total_psize += subbucket_list[i]->num_pvec;
            total_nsize += subbucket_list[i]->num_nvec;   
        }
        _alloc(total_psize, 1);
        _alloc(total_nsize, 0);
        for (long i = 0; i < len; i++){
            memcpy(pvec + num_pvec, subbucket_list[i]->pvec, subbucket_list[i]->num_pvec * sizeof(float *));
            memcpy(nvec + num_nvec, subbucket_list[i]->nvec, subbucket_list[i]->num_nvec * sizeof(float *));
            memcpy(pu + num_pvec, subbucket_list[i]->pu, subbucket_list[i]->num_pvec * sizeof(uint64_t));
            memcpy(nu + num_nvec, subbucket_list[i]->nu, subbucket_list[i]->num_nvec * sizeof(uint64_t));
            //memcpy(pnorm + num_pvec, subbucket_list[i]->pnorm, subbucket_list[i]->num_pvec * sizeof(float));
            //memcpy(nnorm + num_nvec, subbucket_list[i]->nnorm, subbucket_list[i]->num_nvec * sizeof(float));
            if (record_dp){
                memcpy(pdot + num_pvec, subbucket_list[i]->pdot, subbucket_list[i]->num_pvec * sizeof(float));
                memcpy(ndot + num_nvec, subbucket_list[i]->ndot, subbucket_list[i]->num_nvec * sizeof(float));
            }
            if (record_sh){
                memcpy(psh + 4*num_pvec, subbucket_list[i]->psh, subbucket_list[i]->num_pvec * sizeof(uint64_t) * 4);
                memcpy(nsh + 4*num_nvec, subbucket_list[i]->nsh, subbucket_list[i]->num_nvec * sizeof(uint64_t) * 4);
            }
            num_pvec += subbucket_list[i]->num_pvec;
            num_nvec += subbucket_list[i]->num_nvec;
        }
    }
    
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
    
    // __ptr/__pu point to the pointers/uids
    // and __ctr/__cu are the pointer/uid itself
    #if 1
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
    #else
    // not fast in our case
    #define CHECK_AND_ADD2_1X8(__cmp, __ctr, __cu, __ptr, __pu, __add_func) do {                        \
        while (__cmp){                                                                                  \
            int __r = __builtin_ctz(__cmp);                                                             \
            __cmp -= (1 << __r);                                                                        \
            __add_func(__ctr, (__ptr)[__r], __cu, (__pu)[__r]);                                         \
        }                                                                                               \
    } while (0)
    #endif
    #pragma endregion

    // these 4 search functions return the number of dot 
    // product for debug.
    // after search, the bucket will be useless, so we will
    // multiply the stored dot products by 2 in search_naive,
    // and when searching for solutions, one should call it first
    int search_naive(bgj1_sol_t *sol, UidHashTable *uid, float goal_norm){
        //delete center from pvec
        do {
            long low = 0;
            long high = num_pvec-1;
            do {
                long mid = (low+high)/2;
                if ((long)pvec[mid] < (long)center){
                    low = mid;
                } else {
                    high = mid;
                }
            } while (high - low > 2);
            low -= 4;
            if (low < 0) low = 0;
            for (long i = low; i <= low+10; i++){
                if (pvec[i] == center) {
                    num_pvec--;
                    pvec[i] = pvec[num_pvec];
                    pu[i] = pu[num_pvec];
                    if (record_dp) pdot[i] = pdot[num_pvec];
                    if (record_sh) {
                        psh[i*4+0] = psh[4*num_pvec+0];
                        psh[i*4+1] = psh[4*num_pvec+1];
                        psh[i*4+2] = psh[4*num_pvec+2];
                        psh[i*4+3] = psh[4*num_pvec+3];
                    }
                }
            }
            // do we really need this ?
            if (pvec[num_pvec] != center){
                for (long i = 0; i < num_pvec; i++){
                    if (pvec[i] == center) {
                        num_pvec--;
                        pvec[i] = pvec[num_pvec];
                        pu[i] = pu[num_pvec];
                        if (record_dp) pdot[i] = pdot[num_pvec];
                        if (record_sh) {
                            psh[i*4+0] = psh[4*num_pvec+0];
                            psh[i*4+1] = psh[4*num_pvec+1];
                            psh[i*4+2] = psh[4*num_pvec+2];
                            psh[i*4+3] = psh[4*num_pvec+3];
                        }
                    }
                }
            }
        } while(0);
        if (!pnorm) pnorm = (float *) NEW_VEC(num_pvec, sizeof(float));
        if (!nnorm) nnorm = (float *) NEW_VEC(num_nvec, sizeof(float));
        for (long i = 0; i < num_pvec; i++) pnorm[i] = pvec[i][-1];
        for (long i = 0; i < num_nvec; i++) nnorm[i] = nvec[i][-1];
        long ind = 0;
        __attribute__ ((aligned (32))) int cmp[8];
        __m256 cn = _mm256_set1_ps(goal_norm - center_norm);
        long *cmpl = (long *)cmp; 
        
        while (ind < num_pvec - 63) {
            __m256 pd0 = _mm256_loadu_ps(pdot + ind+0);
            __m256 pd1 = _mm256_loadu_ps(pdot + ind+8);
            __m256 pd2 = _mm256_loadu_ps(pdot + ind+16);
            __m256 pd3 = _mm256_loadu_ps(pdot + ind+24);
            __m256 pd4 = _mm256_loadu_ps(pdot + ind+32);
            __m256 pd5 = _mm256_loadu_ps(pdot + ind+40);
            __m256 pd6 = _mm256_loadu_ps(pdot + ind+48);
            __m256 pd7 = _mm256_loadu_ps(pdot + ind+56);
            pd0 = _mm256_add_ps(pd0, pd0);
            pd1 = _mm256_add_ps(pd1, pd1);
            pd2 = _mm256_add_ps(pd2, pd2);
            pd3 = _mm256_add_ps(pd3, pd3);
            pd4 = _mm256_add_ps(pd4, pd4);
            pd5 = _mm256_add_ps(pd5, pd5);
            pd6 = _mm256_add_ps(pd6, pd6);
            pd7 = _mm256_add_ps(pd7, pd7);
            cmp[0] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 0), pd0), 30));
            cmp[1] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 8), pd1), 30));
            cmp[2] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 16), pd2), 30));
            cmp[3] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 24), pd3), 30));
            cmp[4] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 32), pd4), 30));
            cmp[5] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 40), pd5), 30));
            cmp[6] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 48), pd6), 30));
            cmp[7] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 56), pd7), 30));
            _mm256_storeu_ps(pdot+ind+0, pd0);
            _mm256_storeu_ps(pdot+ind+8, pd1);
            _mm256_storeu_ps(pdot+ind+16, pd2);
            _mm256_storeu_ps(pdot+ind+24, pd3);
            _mm256_storeu_ps(pdot+ind+32, pd4);
            _mm256_storeu_ps(pdot+ind+40, pd5);
            _mm256_storeu_ps(pdot+ind+48, pd6);
            _mm256_storeu_ps(pdot+ind+56, pd7);
            for (long i = 0; i < 4; i++){
                if (cmpl[i]){
                    CHECK_AND_ADD2_1X8(cmp[2*i], center, center_u, pvec+ind+i*16+0, pu+ind+i*16+0, TRY_ADDS_TO_DST);
                    CHECK_AND_ADD2_1X8(cmp[2*i+1], center, center_u, pvec+ind+i*16+8, pu+ind+i*16+8, TRY_ADDS_TO_DST);
                }
            }
            ind += 64;
        }
        while (ind < num_pvec - 7) {
            __m256 pd0 = _mm256_loadu_ps(pdot + ind+0);
            pd0 = _mm256_add_ps(pd0, pd0);
            int cmpp = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_sub_ps(_mm256_loadu_ps(pnorm + ind + 0), pd0), 30));
            _mm256_storeu_ps(pdot+ind+0, pd0);
            CHECK_AND_ADD2_1X8(cmpp, center, center_u, pvec+ind, pu+ind, TRY_ADDS_TO_DST);
            ind += 8;
        }
        while (ind < num_pvec) {
            pdot[ind] *= 2.0f;
            if (-pdot[ind] + pnorm[ind] < goal_norm - center_norm) TRY_ADDS_TO_DST(center, pvec[ind], center_u, pu[ind]);
            ind++;
        }

        ind = 0;
        while (ind < num_nvec - 63) {
            __m256 nd0 = _mm256_loadu_ps(ndot + ind+0);
            __m256 nd1 = _mm256_loadu_ps(ndot + ind+8);
            __m256 nd2 = _mm256_loadu_ps(ndot + ind+16);
            __m256 nd3 = _mm256_loadu_ps(ndot + ind+24);
            __m256 nd4 = _mm256_loadu_ps(ndot + ind+32);
            __m256 nd5 = _mm256_loadu_ps(ndot + ind+40);
            __m256 nd6 = _mm256_loadu_ps(ndot + ind+48);
            __m256 nd7 = _mm256_loadu_ps(ndot + ind+56);
            nd0 = _mm256_add_ps(nd0, nd0);
            nd1 = _mm256_add_ps(nd1, nd1);
            nd2 = _mm256_add_ps(nd2, nd2);
            nd3 = _mm256_add_ps(nd3, nd3);
            nd4 = _mm256_add_ps(nd4, nd4);
            nd5 = _mm256_add_ps(nd5, nd5);
            nd6 = _mm256_add_ps(nd6, nd6);
            nd7 = _mm256_add_ps(nd7, nd7);
            cmp[0] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 0), nd0), 30));
            cmp[1] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 8), nd1), 30));
            cmp[2] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 16), nd2), 30));
            cmp[3] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 24), nd3), 30));
            cmp[4] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 32), nd4), 30));
            cmp[5] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 40), nd5), 30));
            cmp[6] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 48), nd6), 30));
            cmp[7] = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 56), nd7), 30));
            _mm256_storeu_ps(ndot+ind+0, nd0);
            _mm256_storeu_ps(ndot+ind+8, nd1);
            _mm256_storeu_ps(ndot+ind+16, nd2);
            _mm256_storeu_ps(ndot+ind+24, nd3);
            _mm256_storeu_ps(ndot+ind+32, nd4);
            _mm256_storeu_ps(ndot+ind+40, nd5);
            _mm256_storeu_ps(ndot+ind+48, nd6);
            _mm256_storeu_ps(ndot+ind+56, nd7);
            for (long i = 0; i < 4; i++){
                if (cmpl[i]){
                    CHECK_AND_ADD2_1X8(cmp[2*i], center, center_u, nvec+ind+i*16+0, nu+ind+i*16+0, TRY_ADDA_TO_DST);
                    CHECK_AND_ADD2_1X8(cmp[2*i+1], center, center_u, nvec+ind+i*16+8, nu+ind+i*16+8, TRY_ADDA_TO_DST);
                }
            }
            ind += 64;
        }
        while (ind < num_nvec - 7) {
            __m256 nd0 = _mm256_loadu_ps(ndot + ind+0);
            nd0 = _mm256_add_ps(nd0, nd0);
            int cmpp = _mm256_movemask_ps(_mm256_cmp_ps(cn, _mm256_add_ps(_mm256_loadu_ps(nnorm + ind + 0), nd0), 30));
            _mm256_storeu_ps(ndot+ind+0, nd0);
            CHECK_AND_ADD2_1X8(cmpp, center, center_u, nvec+ind, nu+ind, TRY_ADDA_TO_DST);
            ind += 8;
        }
        while (ind < num_nvec) {
            ndot[ind] *= 2.0f;
            if (ndot[ind] + nnorm[ind] < goal_norm - center_norm) TRY_ADDA_TO_DST(center, nvec[ind], center_u, nu[ind]);
            ind++;
        }
        return 0;
    }
    int search_pp(bgj1_sol_t *sol, UidHashTable *uid, float goal_norm) {
        #if SEARCHING_USE_SIMHASH
        // todo
        return 0;
        #else
        /*do {
            long cnt2 = 0, cnt3 = 0;
            for (long i = 0; i < num_pvec; i++){
                for (long j = i+1; j < num_pvec; j++){
                    __attribute__ ((aligned (64))) float tmp[256] = {};
                    __attribute__ ((aligned (64))) float tmp1[256] = {};
                    add(tmp1, pvec[j], vec_length);
                    sub(tmp1, pvec[i], vec_length);
                    add(tmp, center, vec_length);
                    sub(tmp, pvec[j], vec_length);
                    sub(tmp, pvec[i], vec_length);
                    if (dot(tmp1, tmp1, vec_length) < goal_norm) cnt2++;
                    if (dot(tmp, tmp, vec_length) < goal_norm) {
                        cnt3++;
                    }
                }
            }
            try_add2 = -cnt2;
            try_add3 = -cnt3;
        } while (0);*/
        float **tmp = (float **)NEW_MAT(8, vec_length, sizeof(double));
        for (long Ind = 0; Ind < num_pvec; Ind += SEARCHING_DP_BLOCK){
            for (long Jnd = Ind; Jnd < num_pvec; Jnd += SEARCHING_DP_BLOCK){
                const long Ibound = (Ind + SEARCHING_DP_BLOCK > num_pvec) ? num_pvec : Ind + SEARCHING_DP_BLOCK;
                const long Jbound = (Jnd + SEARCHING_DP_BLOCK > num_pvec) ? num_pvec : Jnd + SEARCHING_DP_BLOCK;
                long ind = Ind;
                __m256 gn = _mm256_set1_ps(goal_norm);
                #if COLLECT_THREE_RED
                __m256 cn = _mm256_set1_ps(center_norm);
                #endif
                while (ind < Ibound - 7) {
                    for (long i = 0; i < vec_length; i+=8){
                        __m256 p0 = _mm256_load_ps(pvec[ind+0]+i);
                        __m256 p1 = _mm256_load_ps(pvec[ind+1]+i);
                        __m256 p2 = _mm256_load_ps(pvec[ind+2]+i);
                        __m256 p3 = _mm256_load_ps(pvec[ind+3]+i);
                        __m256 p4 = _mm256_load_ps(pvec[ind+4]+i);
                        __m256 p5 = _mm256_load_ps(pvec[ind+5]+i);
                        __m256 p6 = _mm256_load_ps(pvec[ind+6]+i);
                        __m256 p7 = _mm256_load_ps(pvec[ind+7]+i);
                        _mm256_store_ps(tmp[0] + i, _mm256_add_ps(p0, p0));
                        _mm256_store_ps(tmp[1] + i, _mm256_add_ps(p1, p1));
                        _mm256_store_ps(tmp[2] + i, _mm256_add_ps(p2, p2));
                        _mm256_store_ps(tmp[3] + i, _mm256_add_ps(p3, p3));
                        _mm256_store_ps(tmp[4] + i, _mm256_add_ps(p4, p4));
                        _mm256_store_ps(tmp[5] + i, _mm256_add_ps(p5, p5));
                        _mm256_store_ps(tmp[6] + i, _mm256_add_ps(p6, p6));
                        _mm256_store_ps(tmp[7] + i, _mm256_add_ps(p7, p7));
                    }
                    __m256 b2 = _mm256_sub_ps(gn, _mm256_load_ps(pnorm + ind));
                    #if COLLECT_THREE_RED
                    __m256 b3 = _mm256_sub_ps(_mm256_add_ps(gn, _mm256_loadu_ps(pdot + ind)), _mm256_add_ps(cn, _mm256_load_ps(pnorm + ind)));
                    #endif
                    long jnd = (Ind == Jnd) ? ind + 8: Jnd;
                    while (jnd < Jbound - 7) {
                        __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                        AVX2_DP_2X8(pvec[jnd+0], pvec[jnd+1], tmp, vec_length, dst0, dst1);
                        AVX2_DP_2X8(pvec[jnd+2], pvec[jnd+3], tmp, vec_length, dst2, dst3);
                        AVX2_DP_2X8(pvec[jnd+4], pvec[jnd+5], tmp, vec_length, dst4, dst5);
                        AVX2_DP_2X8(pvec[jnd+6], pvec[jnd+7], tmp, vec_length, dst6, dst7);
                        int cmp2[8];
                        long *cmp2l = (long *) cmp2;
                        #if COLLECT_THREE_RED
                        int cmp3[8];
                        long *cmp3l = (long *) cmp3;
                        #endif
                        cmp2[0] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+0]), dst0), 30));
                        cmp2[1] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+1]), dst1), 30));
                        cmp2[2] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+2]), dst2), 30));
                        cmp2[3] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+3]), dst3), 30));
                        cmp2[4] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+4]), dst4), 30));
                        cmp2[5] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+5]), dst5), 30));
                        cmp2[6] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+6]), dst6), 30));
                        cmp2[7] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+7]), dst7), 30));
                        #if COLLECT_THREE_RED
                        cmp3[0] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+0]), _mm256_broadcast_ss(&pdot[jnd+0])), dst0), 30));
                        cmp3[1] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+1]), _mm256_broadcast_ss(&pdot[jnd+1])), dst1), 30));
                        cmp3[2] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+2]), _mm256_broadcast_ss(&pdot[jnd+2])), dst2), 30));
                        cmp3[3] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+3]), _mm256_broadcast_ss(&pdot[jnd+3])), dst3), 30));
                        cmp3[4] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+4]), _mm256_broadcast_ss(&pdot[jnd+4])), dst4), 30));
                        cmp3[5] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+5]), _mm256_broadcast_ss(&pdot[jnd+5])), dst5), 30));
                        cmp3[6] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+6]), _mm256_broadcast_ss(&pdot[jnd+6])), dst6), 30));
                        cmp3[7] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd+7]), _mm256_broadcast_ss(&pdot[jnd+7])), dst7), 30));
                        #endif
                        // we skip the 8x8 upper triangular parts since the buckets are large enough
                        /*if (ind == jnd) {
                            cmp2[0] &= 0xfe;
                            cmp2[1] &= 0xfc;
                            cmp2[2] &= 0xf8;
                            cmp2[3] &= 0xf0;
                            cmp2[4] &= 0xe0;
                            cmp2[5] &= 0xc0;
                            cmp2[6] &= 0x80;
                            cmp2[7] &= 0x00;
                            cmp3[0] &= 0xfe;
                            cmp3[1] &= 0xfc;
                            cmp3[2] &= 0xf8;
                            cmp3[3] &= 0xf0;
                            cmp3[4] &= 0xe0;
                            cmp3[5] &= 0xc0;
                            cmp3[6] &= 0x80;
                            cmp3[7] &= 0x00;
                        }*/
                        for (long i = 0; i < 4; i++){
                            if (cmp2l[i]){
                                CHECK_AND_ADD2_1X8(cmp2[2*i], pvec[jnd+2*i], pu[jnd+2*i], pvec+ind, pu+ind, TRY_ADDS_TO_DST);
                                CHECK_AND_ADD2_1X8(cmp2[2*i+1], pvec[jnd+2*i+1], pu[jnd+2*i+1], pvec+ind, pu+ind, TRY_ADDS_TO_DST);
                            }
                            #if COLLECT_THREE_RED
                            if (cmp3l[i]){
                                CHECK_AND_ADD2_1X8(cmp3[2*i], pvec[jnd+2*i], pu[jnd+2*i], pvec+ind, pu+ind, TRY_ADDSS_TO_DST);
                                CHECK_AND_ADD2_1X8(cmp3[2*i+1], pvec[jnd+2*i+1], pu[jnd+2*i+1], pvec+ind, pu+ind, TRY_ADDSS_TO_DST);
                            }
                            #endif
                        } 
                        jnd += 8;
                    }
                    while (jnd < Jbound) {
                        __m256 dst;
                        AVX2_DP_1X8(pvec[jnd], tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], vec_length, dst);
                        int cmp2 = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd]), dst), 30));
                        #if COLLECT_THREE_RED
                        int cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_sub_ps(_mm256_broadcast_ss(&pnorm[jnd]), _mm256_broadcast_ss(&pdot[jnd])), dst), 30));
                        #endif
                        CHECK_AND_ADD2_1X8(cmp2, pvec[jnd], pu[jnd], pvec+ind, pu+ind, TRY_ADDS_TO_DST);
                        #if COLLECT_THREE_RED
                        CHECK_AND_ADD2_1X8(cmp3, pvec[jnd], pu[jnd], pvec+ind, pu+ind, TRY_ADDSS_TO_DST);
                        #endif
                        jnd++;
                    }

                    ind += 8;
                }

                if (ind  < Ibound){
                    const long nrem = Ibound - ind;
                    long jnd = (Ind == Jnd) ? ind+8: Jnd;
                    while (jnd < Jbound - 7) {
                        for (long i = 0; i < vec_length; i += 8){
                            for (long j = 0; j < nrem; j++){
                                __m256 p0 = _mm256_load_ps(pvec[ind+j]+i);
                                _mm256_store_ps(tmp[j] + i, _mm256_add_ps(p0, p0));
                            }
                        }
                        __m256 dst[7];
                        int cmp2[8];
                        long *cmp2l = (long *) cmp2;
                        #if COLLECT_THREE_RED
                        int cmp3[8];
                        long *cmp3l = (long *) cmp3;
                        #endif
                        for (long j = 0; j < nrem; j++){
                            AVX2_DP_1X8(tmp[j], pvec[jnd], pvec[jnd+1], pvec[jnd+2], pvec[jnd+3], pvec[jnd+4], pvec[jnd+5], pvec[jnd+6], pvec[jnd+7], vec_length, dst[j]);
                        }
                        __m256 nn = _mm256_loadu_ps(pnorm + jnd);
                        #if COLLECT_THREE_RED
                        __m256 nd = _mm256_sub_ps(nn, _mm256_loadu_ps(pdot + jnd));
                        #endif
                        for (long j = 0; j < nrem; j++){
                            cmp2[j] = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(gn, _mm256_broadcast_ss(&pnorm[ind+j])), _mm256_sub_ps(nn, dst[j]), 30));
                            #if COLLECT_THREE_RED
                            __m256 b3 = _mm256_sub_ps(_mm256_add_ps(gn, _mm256_broadcast_ss(&pdot[ind+j])), _mm256_add_ps(cn, _mm256_broadcast_ss(&pnorm[ind+j])));
                            cmp3[j] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(nd, dst[j]), 30));
                            #endif
                        }
                        for (long j = 0; j < nrem; j++){
                            CHECK_AND_ADD2_1X8(cmp2[j], pvec[ind+j], pu[ind+j], pvec+jnd, pu+jnd, TRY_ADDS_TO_DST);
                            #if COLLECT_THREE_RED
                            CHECK_AND_ADD2_1X8(cmp3[j], pvec[ind+j], pu[ind+j], pvec+jnd, pu+jnd, TRY_ADDSS_TO_DST);
                            #endif
                        }
                        jnd+=8;
                    }

                    if (jnd < Jbound) {
                        const long inrem = Ibound - ind;
                        const long jnrem = Jbound - jnd;
                        for (long i = 0; i < inrem; i++){
                            for (long j = 0; j < jnrem; j++){
                                float x = 2.0f * dot(pvec[ind+i], pvec[jnd+j], vec_length);
                                if (-x + pnorm[jnd+j] < goal_norm - pnorm[i+ind]) TRY_ADDS_TO_DST(pvec[ind+i], pvec[jnd+j], pu[ind+i], pu[jnd+j]);
                                #if COLLECT_THREE_RED
                                if (x + pnorm[jnd+j] - pdot[jnd+j] < goal_norm - center_norm - pnorm[ind+i]+pdot[ind+i]){
                                    TRY_ADDSS_TO_DST(pvec[ind+i], pvec[jnd+j], pu[ind+i], pu[jnd+j]);
                                }
                                #endif
                            }
                        }
                    }
                }
            }
        }
        FREE_MAT(tmp);
        return ((num_pvec-8) * num_pvec) / 2;
        #endif
    }
    int search_nn(bgj1_sol_t *sol, UidHashTable *uid, float goal_norm) {
        #if SEARCHING_USE_SIMHASH
        // todo
        return 0;
        #else
        /*do {
            long cnt2 = 0, cnt3 = 0;
            for (long i = 0; i < num_nvec; i++){
                for (long j = i+1; j < num_nvec; j++){
                    __attribute__ ((aligned (64))) float tmp[256] = {};
                    __attribute__ ((aligned (64))) float tmp1[256] = {};
                    add(tmp1, nvec[j], vec_length);
                    sub(tmp1, nvec[i], vec_length);
                    add(tmp, center, vec_length);
                    add(tmp, nvec[j], vec_length);
                    add(tmp, nvec[i], vec_length);
                    if (dot(tmp1, tmp1, vec_length) < goal_norm) cnt2++;
                    if (dot(tmp, tmp, vec_length) < goal_norm) {
                        cnt3++;
                    }
                }
            }
            try_add2 = -cnt2;
            try_add3 = -cnt3;
        } while (0);*/
        float **tmp = (float **)NEW_MAT(8, vec_length, sizeof(double));
        for (long Ind = 0; Ind < num_nvec; Ind += SEARCHING_DP_BLOCK){
            for (long Jnd = Ind; Jnd < num_nvec; Jnd += SEARCHING_DP_BLOCK){
                const long Ibound = (Ind + SEARCHING_DP_BLOCK > num_nvec) ? num_nvec : Ind + SEARCHING_DP_BLOCK;
                const long Jbound = (Jnd + SEARCHING_DP_BLOCK > num_nvec) ? num_nvec : Jnd + SEARCHING_DP_BLOCK;
                long ind = Ind;
                __m256 gn = _mm256_set1_ps(goal_norm);
                #if COLLECT_THREE_RED
                __m256 cn = _mm256_set1_ps(center_norm);
                #endif
                while (ind < Ibound - 7) {
                    for (long i = 0; i < vec_length; i+=8){
                        __m256 p0 = _mm256_load_ps(nvec[ind+0]+i);
                        __m256 p1 = _mm256_load_ps(nvec[ind+1]+i);
                        __m256 p2 = _mm256_load_ps(nvec[ind+2]+i);
                        __m256 p3 = _mm256_load_ps(nvec[ind+3]+i);
                        __m256 p4 = _mm256_load_ps(nvec[ind+4]+i);
                        __m256 p5 = _mm256_load_ps(nvec[ind+5]+i);
                        __m256 p6 = _mm256_load_ps(nvec[ind+6]+i);
                        __m256 p7 = _mm256_load_ps(nvec[ind+7]+i);
                        _mm256_store_ps(tmp[0] + i, _mm256_add_ps(p0, p0));
                        _mm256_store_ps(tmp[1] + i, _mm256_add_ps(p1, p1));
                        _mm256_store_ps(tmp[2] + i, _mm256_add_ps(p2, p2));
                        _mm256_store_ps(tmp[3] + i, _mm256_add_ps(p3, p3));
                        _mm256_store_ps(tmp[4] + i, _mm256_add_ps(p4, p4));
                        _mm256_store_ps(tmp[5] + i, _mm256_add_ps(p5, p5));
                        _mm256_store_ps(tmp[6] + i, _mm256_add_ps(p6, p6));
                        _mm256_store_ps(tmp[7] + i, _mm256_add_ps(p7, p7));
                    }
                    __m256 b2 = _mm256_sub_ps(gn, _mm256_load_ps(nnorm + ind));
                    #if COLLECT_THREE_RED
                    __m256 b3 = _mm256_sub_ps(_mm256_sub_ps(gn, _mm256_loadu_ps(ndot + ind)), _mm256_add_ps(cn, _mm256_load_ps(nnorm + ind)));
                    #endif
                    long jnd = (Ind == Jnd) ? ind + 8: Jnd;
                    while (jnd < Jbound - 7) {
                        __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                        AVX2_DP_2X8(nvec[jnd+0], nvec[jnd+1], tmp, vec_length, dst0, dst1);
                        AVX2_DP_2X8(nvec[jnd+2], nvec[jnd+3], tmp, vec_length, dst2, dst3);
                        AVX2_DP_2X8(nvec[jnd+4], nvec[jnd+5], tmp, vec_length, dst4, dst5);
                        AVX2_DP_2X8(nvec[jnd+6], nvec[jnd+7], tmp, vec_length, dst6, dst7);
                        int cmp2[8];
                        long *cmp2l = (long *) cmp2;
                        #if COLLECT_THREE_RED
                        int cmp3[8];
                        long *cmp3l = (long *) cmp3;
                        #endif
                        cmp2[0] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+0]), dst0), 30));
                        cmp2[1] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+1]), dst1), 30));
                        cmp2[2] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+2]), dst2), 30));
                        cmp2[3] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+3]), dst3), 30));
                        cmp2[4] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+4]), dst4), 30));
                        cmp2[5] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+5]), dst5), 30));
                        cmp2[6] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+6]), dst6), 30));
                        cmp2[7] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd+7]), dst7), 30));
                        #if COLLECT_THREE_RED
                        cmp3[0] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+0]), _mm256_broadcast_ss(&ndot[jnd+0])), dst0), 30));
                        cmp3[1] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+1]), _mm256_broadcast_ss(&ndot[jnd+1])), dst1), 30));
                        cmp3[2] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+2]), _mm256_broadcast_ss(&ndot[jnd+2])), dst2), 30));
                        cmp3[3] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+3]), _mm256_broadcast_ss(&ndot[jnd+3])), dst3), 30));
                        cmp3[4] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+4]), _mm256_broadcast_ss(&ndot[jnd+4])), dst4), 30));
                        cmp3[5] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+5]), _mm256_broadcast_ss(&ndot[jnd+5])), dst5), 30));
                        cmp3[6] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+6]), _mm256_broadcast_ss(&ndot[jnd+6])), dst6), 30));
                        cmp3[7] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd+7]), _mm256_broadcast_ss(&ndot[jnd+7])), dst7), 30));
                        #endif
                        // we skip the 8x8 upper triangular parts since the buckets are large enough
                        /*if (ind == jnd) {
                            cmp2[0] &= 0xfe;
                            cmp2[1] &= 0xfc;
                            cmp2[2] &= 0xf8;
                            cmp2[3] &= 0xf0;
                            cmp2[4] &= 0xe0;
                            cmp2[5] &= 0xc0;
                            cmp2[6] &= 0x80;
                            cmp2[7] &= 0x00;
                            cmp3[0] &= 0xfe;
                            cmp3[1] &= 0xfc;
                            cmp3[2] &= 0xf8;
                            cmp3[3] &= 0xf0;
                            cmp3[4] &= 0xe0;
                            cmp3[5] &= 0xc0;
                            cmp3[6] &= 0x80;
                            cmp3[7] &= 0x00;
                        }*/
                        for (long i = 0; i < 4; i++){
                            if (cmp2l[i]){
                                CHECK_AND_ADD2_1X8(cmp2[2*i], nvec[jnd+2*i], nu[jnd+2*i], nvec+ind, nu+ind, TRY_ADDS_TO_DST);
                                CHECK_AND_ADD2_1X8(cmp2[2*i+1], nvec[jnd+2*i+1], nu[jnd+2*i+1], nvec+ind, nu+ind, TRY_ADDS_TO_DST);
                            }
                            #if COLLECT_THREE_RED
                            if (cmp3l[i]){
                                CHECK_AND_ADD2_1X8(cmp3[2*i], nvec[jnd+2*i], nu[jnd+2*i], nvec+ind, nu+ind, TRY_ADDAA_TO_DST);
                                CHECK_AND_ADD2_1X8(cmp3[2*i+1], nvec[jnd+2*i+1], nu[jnd+2*i+1], nvec+ind, nu+ind, TRY_ADDAA_TO_DST);
                            }
                            #endif
                        } 
                        jnd += 8;
                    }
                    while (jnd < Jbound) {
                        __m256 dst;
                        AVX2_DP_1X8(nvec[jnd], tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], vec_length, dst);
                        int cmp2 = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_sub_ps(_mm256_broadcast_ss(&nnorm[jnd]), dst), 30));
                        #if COLLECT_THREE_RED
                        int cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[jnd]), _mm256_broadcast_ss(&ndot[jnd])), dst), 30));
                        #endif
                        CHECK_AND_ADD2_1X8(cmp2, nvec[jnd], nu[jnd], nvec+ind, nu+ind, TRY_ADDS_TO_DST);
                        #if COLLECT_THREE_RED
                        CHECK_AND_ADD2_1X8(cmp3, nvec[jnd], nu[jnd], nvec+ind, nu+ind, TRY_ADDAA_TO_DST);
                        #endif
                        jnd++;
                    }

                    ind += 8;
                }

                if (ind  < Ibound){
                    const long nrem = Ibound - ind;
                    long jnd = (Ind == Jnd) ? ind+8: Jnd;
                    while (jnd < Jbound - 7) {
                        for (long i = 0; i < vec_length; i += 8){
                            for (long j = 0; j < nrem; j++){
                                __m256 p0 = _mm256_load_ps(nvec[ind+j]+i);
                                _mm256_store_ps(tmp[j] + i, _mm256_add_ps(p0, p0));
                            }
                        }
                        __m256 dst[7];
                        int cmp2[8];
                        long *cmp2l = (long *) cmp2;
                        #if COLLECT_THREE_RED
                        int cmp3[8];
                        long *cmp3l = (long *) cmp3;
                        #endif
                        for (long j = 0; j < nrem; j++){
                            AVX2_DP_1X8(tmp[j], nvec[jnd], nvec[jnd+1], nvec[jnd+2], nvec[jnd+3], nvec[jnd+4], nvec[jnd+5], nvec[jnd+6], nvec[jnd+7], vec_length, dst[j]);
                        }
                        __m256 nn = _mm256_loadu_ps(nnorm + jnd);
                        #if COLLECT_THREE_RED
                        __m256 nd = _mm256_add_ps(nn, _mm256_loadu_ps(ndot + jnd));
                        #endif
                        for (long j = 0; j < nrem; j++){
                            cmp2[j] = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(gn, _mm256_broadcast_ss(&nnorm[ind+j])), _mm256_sub_ps(nn, dst[j]), 30));
                            #if COLLECT_THREE_RED
                            __m256 b3 = _mm256_sub_ps(_mm256_sub_ps(gn, _mm256_broadcast_ss(&ndot[ind+j])), _mm256_add_ps(cn, _mm256_broadcast_ss(&nnorm[ind+j])));
                            cmp3[j] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_add_ps(nd, dst[j]), 30));
                            #endif
                        }
                        for (long j = 0; j < nrem; j++){
                            CHECK_AND_ADD2_1X8(cmp2[j], nvec[ind+j], nu[ind+j], nvec+jnd, nu+jnd, TRY_ADDS_TO_DST);
                            #if COLLECT_THREE_RED
                            CHECK_AND_ADD2_1X8(cmp3[j], nvec[ind+j], nu[ind+j], nvec+jnd, nu+jnd, TRY_ADDAA_TO_DST);
                            #endif
                        }
                        jnd+=8;
                    }

                    if (jnd < Jbound) {
                        const long inrem = Ibound - ind;
                        const long jnrem = Jbound - jnd;
                        for (long i = 0; i < inrem; i++){
                            for (long j = 0; j < jnrem; j++){
                                float x = 2.0f * dot(nvec[ind+i], nvec[jnd+j], vec_length);
                                if (-x + nnorm[jnd+j] < goal_norm - nnorm[i+ind]) TRY_ADDS_TO_DST(nvec[ind+i], nvec[jnd+j], nu[ind+i], nu[jnd+j]);
                                #if COLLECT_THREE_RED
                                if (x + nnorm[jnd+j] + ndot[jnd+j] < goal_norm - center_norm - nnorm[ind+i]-ndot[ind+i]){
                                    TRY_ADDAA_TO_DST(nvec[ind+i], nvec[jnd+j], nu[ind+i], nu[jnd+j]);
                                }
                                #endif
                            }
                        }
                    }
                }
            }
        }
        FREE_MAT(tmp);
        return ((num_nvec-8) * num_nvec) / 2;
        #endif
    }
    int search_np(bgj1_sol_t *sol, UidHashTable *uid, float goal_norm) {
        #if SEARCHING_USE_SIMHASH
        // warning! not done yet
        uint64_t __bucket_ndp = 0;
        __m256i sh_lbound = _mm256_set1_epi16(256 - XPC_BGJ1_TRD_THRESHOLD);  // < lbound --> three reduction
        __m256i sh_ubound = _mm256_set1_epi16(256 - XPC_BGJ1_THRESHOLD);      // > hbound --> two reduction
        __m256i low_mask = _mm256_set1_epi8(0x0f);
        __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        long *to_dp_list2 = (long *) malloc(SEARCHING_SH_BLOCK * SEARCHING_SH_BLOCK * 2 * sizeof(long));
        long *to_dp_list3 = (long *) malloc(SEARCHING_SH_BLOCK * SEARCHING_SH_BLOCK * 2 * sizeof(long));
        for (long Pnd = 0; Pnd < num_pvec; Pnd += SEARCHING_SH_BLOCK) {
            for (long Nnd = 0; Nnd < num_nvec; Nnd += SEARCHING_SH_BLOCK) {
                const long Pbound = (Pnd + SEARCHING_SH_BLOCK > num_pvec) ? num_pvec : Pnd + SEARCHING_SH_BLOCK;
                const long Nbound = (Nnd + SEARCHING_SH_BLOCK > num_nvec) ? num_nvec : Nnd + SEARCHING_SH_BLOCK;
                long pnd = Pnd;
                long n_to_dp2 = 0;
                long n_to_dp3 = 0;
                while (pnd < Pbound - 7) {
                    __m256i psh0 = _mm256_loadu_si256((__m256i *)(psh+4*pnd));
                    __m256i psh1 = _mm256_loadu_si256((__m256i *)(psh+4*pnd+4));
                    __m256i psh2 = _mm256_loadu_si256((__m256i *)(psh+4*pnd+8));
                    __m256i psh3 = _mm256_loadu_si256((__m256i *)(psh+4*pnd+12));
                    __m256i psh4 = _mm256_loadu_si256((__m256i *)(psh+4*pnd+16));
                    __m256i psh5 = _mm256_loadu_si256((__m256i *)(psh+4*pnd+20));
                    __m256i psh6 = _mm256_loadu_si256((__m256i *)(psh+4*pnd+24));
                    __m256i psh7 = _mm256_loadu_si256((__m256i *)(psh+4*pnd+28));

                    long nnd = Nnd;
                    while (nnd < Nbound - 7) {
                        __m256i nsh0 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd));
                        __m256i nsh1 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd+4));
                        __m256i nsh2 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd+8));
                        __m256i nsh3 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd+12));
                        __m256i nsh4 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd+16));
                        __m256i nsh5 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd+20));
                        __m256i nsh6 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd+24));
                        __m256i nsh7 = _mm256_loadu_si256((__m256i *)(nsh+4*nnd+28));

                        __m256i dst0, dst1, dst2, dst3;
                        AVX2_POPCNT64_4X4(dst0, psh0, psh1, psh2, psh3, nsh0, nsh1, nsh2, nsh3, lookup, low_mask);
                        AVX2_POPCNT64_4X4(dst1, psh0, psh1, psh2, psh3, nsh4, nsh5, nsh6, nsh7, lookup, low_mask);
                        AVX2_POPCNT64_4X4(dst2, psh4, psh5, psh6, psh7, nsh0, nsh1, nsh2, nsh3, lookup, low_mask);
                        AVX2_POPCNT64_4X4(dst3, psh4, psh5, psh6, psh7, nsh4, nsh5, nsh6, nsh7, lookup, low_mask);
                        __m256i lc0 = _mm256_cmpgt_epi16(sh_lbound, dst0);
                        __m256i lc1 = _mm256_cmpgt_epi16(sh_lbound, dst1);
                        __m256i lc2 = _mm256_cmpgt_epi16(sh_lbound, dst2);
                        __m256i lc3 = _mm256_cmpgt_epi16(sh_lbound, dst3);
                        __m256i uc0 = _mm256_cmpgt_epi16(dst0, sh_ubound);
                        __m256i uc1 = _mm256_cmpgt_epi16(dst1, sh_ubound);
                        __m256i uc2 = _mm256_cmpgt_epi16(dst2, sh_ubound);
                        __m256i uc3 = _mm256_cmpgt_epi16(dst3, sh_ubound);

                        uint32_t lc320 = _mm256_movemask_epi8(lc0);
                        uint32_t lc321 = _mm256_movemask_epi8(lc1);
                        uint32_t lc322 = _mm256_movemask_epi8(lc2);
                        uint32_t lc323 = _mm256_movemask_epi8(lc3);
                        uint32_t uc320 = _mm256_movemask_epi8(uc0);
                        uint32_t uc321 = _mm256_movemask_epi8(uc1);
                        uint32_t uc322 = _mm256_movemask_epi8(uc2);
                        uint32_t uc323 = _mm256_movemask_epi8(uc3);


                        while (lc320) {
                            long r = __builtin_ctz(lc320);
                            lc320 -= (3U << r);
                            r >>= 1;
                            to_dp_list3[n_to_dp3 * 2] = pnd + (r >> 2);             // p first
                            to_dp_list3[n_to_dp3 * 2 + 1] = nnd + (r & 0x3);        // n next
                            n_to_dp3++;
                        }
                        while (lc321) {
                            long r = __builtin_ctz(lc321);
                            lc321 -= (3U << r);
                            r >>= 1;
                            to_dp_list3[n_to_dp3 * 2] = pnd + (r >> 2);             // p first
                            to_dp_list3[n_to_dp3 * 2 + 1] = nnd + 4 + (r & 0x3);    // n next
                            n_to_dp3++;
                        }
                        while (lc322) {
                            long r = __builtin_ctz(lc322);
                            lc322 -= (3U << r);
                            r >>= 1;
                            to_dp_list3[n_to_dp3 * 2] = pnd + 4 + (r >> 2);         // p first
                            to_dp_list3[n_to_dp3 * 2 + 1] = nnd + (r & 0x3);        // n next
                            n_to_dp3++;
                        }
                        while (lc323) {
                            long r = __builtin_ctz(lc323);
                            lc323 -= (3U << r);
                            r >>= 1;
                            to_dp_list3[n_to_dp3 * 2] = pnd + 4 + (r >> 2);         // p first
                            to_dp_list3[n_to_dp3 * 2 + 1] = nnd + 4 + (r & 0x3);    // n next
                            n_to_dp3++;
                        }
                        while (uc320) {
                            long r = __builtin_ctz(uc320);
                            uc320 -= (3U << r);
                            r >>= 1;
                            to_dp_list2[n_to_dp2 * 2] = pnd + (r >> 2);             // p first
                            to_dp_list2[n_to_dp2 * 2 + 1] = nnd + (r & 0x3);        // n next
                            n_to_dp2++;
                        }
                        while (uc321) {
                            long r = __builtin_ctz(uc321);
                            uc321 -= (3U << r);
                            r >>= 1;
                            to_dp_list2[n_to_dp2 * 2] = pnd + (r >> 2);             // p first
                            to_dp_list2[n_to_dp2 * 2 + 1] = nnd + 4 + (r & 0x3);    // n next
                            n_to_dp2++;
                        }
                        while (uc322) {
                            long r = __builtin_ctz(uc322);
                            uc322 -= (3U << r);
                            r >>= 1;
                            to_dp_list2[n_to_dp2 * 2] = pnd + 4 + (r >> 2);         // p first
                            to_dp_list2[n_to_dp2 * 2 + 1] = nnd + (r & 0x3);        // n next
                            n_to_dp2++;
                        }
                        while (uc323) {
                            long r = __builtin_ctz(uc323);
                            uc323 -= (3U << r);
                            r >>= 1;
                            to_dp_list2[n_to_dp2 * 2] = pnd + 4 + (r >> 2);         // p first
                            to_dp_list2[n_to_dp2 * 2 + 1] = nnd + 4 + (r & 0x3);    // n next
                            n_to_dp2++;
                        }

                        nnd += 8;
                    }

                    if (nnd < Nbound) {
                        // todo
                    }

                    pnd += 8;
                }

                if (pnd < Pbound) {
                    // todo
                }

                // process to_dp_list2
                do {
                    long ii = 0;
                    __bucket_ndp += n_to_dp2;
                    while (ii < n_to_dp2 - 7){
                        float *ptr0 = pvec[to_dp_list2[ii*2+0]];
                        float *ptr1 = pvec[to_dp_list2[ii*2+2]];
                        float *ptr2 = pvec[to_dp_list2[ii*2+4]];
                        float *ptr3 = pvec[to_dp_list2[ii*2+6]];
                        float *ptr4 = pvec[to_dp_list2[ii*2+8]];
                        float *ptr5 = pvec[to_dp_list2[ii*2+10]];
                        float *ptr6 = pvec[to_dp_list2[ii*2+12]];
                        float *ptr7 = pvec[to_dp_list2[ii*2+14]];

                        float *ntr0 = nvec[to_dp_list2[ii*2+1]];
                        float *ntr1 = nvec[to_dp_list2[ii*2+3]];
                        float *ntr2 = nvec[to_dp_list2[ii*2+5]];
                        float *ntr3 = nvec[to_dp_list2[ii*2+7]];
                        float *ntr4 = nvec[to_dp_list2[ii*2+9]];
                        float *ntr5 = nvec[to_dp_list2[ii*2+11]];
                        float *ntr6 = nvec[to_dp_list2[ii*2+13]];
                        float *ntr7 = nvec[to_dp_list2[ii*2+15]];

                        float __dst0, __dst1, __dst2, __dst3;
                        float __dst4, __dst5, __dst6, __dst7;
                        AVX2_DOT_PRODUCT_B8(ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ntr0, ntr1, ntr2, ntr3, ntr4, ntr5, ntr6, ntr7, vec_length);
                        if (2.0 * __dst0 + pnorm[to_dp_list2[ii*2+0]] + nnorm[to_dp_list2[ii*2+1]] < goal_norm) TRY_ADDA_TO_DST(ptr0, ntr0, pu[to_dp_list2[ii*2+0]], nu[to_dp_list2[ii*2+1]]);
                        if (2.0 * __dst1 + pnorm[to_dp_list2[ii*2+2]] + nnorm[to_dp_list2[ii*2+3]] < goal_norm) TRY_ADDA_TO_DST(ptr1, ntr1, pu[to_dp_list2[ii*2+2]], nu[to_dp_list2[ii*2+3]]);
                        if (2.0 * __dst2 + pnorm[to_dp_list2[ii*2+4]] + nnorm[to_dp_list2[ii*2+5]] < goal_norm) TRY_ADDA_TO_DST(ptr2, ntr2, pu[to_dp_list2[ii*2+4]], nu[to_dp_list2[ii*2+5]]);
                        if (2.0 * __dst3 + pnorm[to_dp_list2[ii*2+6]] + nnorm[to_dp_list2[ii*2+7]] < goal_norm) TRY_ADDA_TO_DST(ptr3, ntr3, pu[to_dp_list2[ii*2+6]], nu[to_dp_list2[ii*2+7]]);
                        if (2.0 * __dst4 + pnorm[to_dp_list2[ii*2+8]] + nnorm[to_dp_list2[ii*2+9]] < goal_norm) TRY_ADDA_TO_DST(ptr4, ntr4, pu[to_dp_list2[ii*2+8]], nu[to_dp_list2[ii*2+9]]);
                        if (2.0 * __dst5 + pnorm[to_dp_list2[ii*2+10]] + nnorm[to_dp_list2[ii*2+11]] < goal_norm) TRY_ADDA_TO_DST(ptr5, ntr5, pu[to_dp_list2[ii*2+10]], nu[to_dp_list2[ii*2+11]]);
                        if (2.0 * __dst6 + pnorm[to_dp_list2[ii*2+12]] + nnorm[to_dp_list2[ii*2+13]] < goal_norm) TRY_ADDA_TO_DST(ptr6, ntr6, pu[to_dp_list2[ii*2+12]], nu[to_dp_list2[ii*2+13]]);
                        if (2.0 * __dst7 + pnorm[to_dp_list2[ii*2+14]] + nnorm[to_dp_list2[ii*2+15]] < goal_norm) TRY_ADDA_TO_DST(ptr7, ntr7, pu[to_dp_list2[ii*2+14]], nu[to_dp_list2[ii*2+15]]);
                        ii += 8;
                    }
                    while (ii < n_to_dp2){
                        float *ptr0 = pvec[to_dp_list2[ii*2+0]];
                        float *ntr0 = nvec[to_dp_list2[ii*2+1]];
                        float x = dot(ptr0, ntr0, vec_length);
                        if (2.0 * x + pnorm[to_dp_list2[ii*2+0]] + nnorm[to_dp_list2[ii*2+1]] < goal_norm) TRY_ADDA_TO_DST(ptr0, ntr0, pu[to_dp_list2[ii*2+0]], nu[to_dp_list2[ii*2+1]]);
                        ii++;
                    }
                } while (0);

                // process to_dp_list3
                do {
                    long ii = 0;
                    __bucket_ndp += n_to_dp3;
                    while (ii < n_to_dp3 - 7) {
                        float *ptr0 = pvec[to_dp_list3[ii*2+0]];
                        float *ptr1 = pvec[to_dp_list3[ii*2+2]];
                        float *ptr2 = pvec[to_dp_list3[ii*2+4]];
                        float *ptr3 = pvec[to_dp_list3[ii*2+6]];
                        float *ptr4 = pvec[to_dp_list3[ii*2+8]];
                        float *ptr5 = pvec[to_dp_list3[ii*2+10]];
                        float *ptr6 = pvec[to_dp_list3[ii*2+12]];
                        float *ptr7 = pvec[to_dp_list3[ii*2+14]];

                        float *ntr0 = nvec[to_dp_list3[ii*2+1]];
                        float *ntr1 = nvec[to_dp_list3[ii*2+3]];
                        float *ntr2 = nvec[to_dp_list3[ii*2+5]];
                        float *ntr3 = nvec[to_dp_list3[ii*2+7]];
                        float *ntr4 = nvec[to_dp_list3[ii*2+9]];
                        float *ntr5 = nvec[to_dp_list3[ii*2+11]];
                        float *ntr6 = nvec[to_dp_list3[ii*2+13]];
                        float *ntr7 = nvec[to_dp_list3[ii*2+15]];

                        float __dst0, __dst1, __dst2, __dst3;
                        float __dst4, __dst5, __dst6, __dst7;
                        AVX2_DOT_PRODUCT_B8(ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ntr0, ntr1, ntr2, ntr3, ntr4, ntr5, ntr6, ntr7, vec_length);
                        if (-2.0 * __dst0 + pnorm[to_dp_list3[ii*2+0]] + nnorm[to_dp_list3[ii*2+1]] - pdot[to_dp_list3[ii*2+0]] + ndot[to_dp_list3[ii*2+1]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr0, ntr0, pu[to_dp_list3[ii*2+0]], nu[to_dp_list3[ii*2+1]]);
                        if (-2.0 * __dst1 + pnorm[to_dp_list3[ii*2+2]] + nnorm[to_dp_list3[ii*2+3]] - pdot[to_dp_list3[ii*2+2]] + ndot[to_dp_list3[ii*2+3]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr1, ntr1, pu[to_dp_list3[ii*2+2]], nu[to_dp_list3[ii*2+3]]);
                        if (-2.0 * __dst2 + pnorm[to_dp_list3[ii*2+4]] + nnorm[to_dp_list3[ii*2+5]] - pdot[to_dp_list3[ii*2+4]] + ndot[to_dp_list3[ii*2+5]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr2, ntr2, pu[to_dp_list3[ii*2+4]], nu[to_dp_list3[ii*2+5]]);
                        if (-2.0 * __dst3 + pnorm[to_dp_list3[ii*2+6]] + nnorm[to_dp_list3[ii*2+7]] - pdot[to_dp_list3[ii*2+6]] + ndot[to_dp_list3[ii*2+7]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr3, ntr3, pu[to_dp_list3[ii*2+6]], nu[to_dp_list3[ii*2+7]]);
                        if (-2.0 * __dst4 + pnorm[to_dp_list3[ii*2+8]] + nnorm[to_dp_list3[ii*2+9]] - pdot[to_dp_list3[ii*2+8]] + ndot[to_dp_list3[ii*2+9]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr4, ntr4, pu[to_dp_list3[ii*2+8]], nu[to_dp_list3[ii*2+9]]);
                        if (-2.0 * __dst5 + pnorm[to_dp_list3[ii*2+10]] + nnorm[to_dp_list3[ii*2+11]] - pdot[to_dp_list3[ii*2+10]] + ndot[to_dp_list3[ii*2+11]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr5, ntr5, pu[to_dp_list3[ii*2+10]], nu[to_dp_list3[ii*2+11]]);
                        if (-2.0 * __dst6 + pnorm[to_dp_list3[ii*2+12]] + nnorm[to_dp_list3[ii*2+13]] - pdot[to_dp_list3[ii*2+12]] + ndot[to_dp_list3[ii*2+13]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr6, ntr6, pu[to_dp_list3[ii*2+12]], nu[to_dp_list3[ii*2+13]]);
                        if (-2.0 * __dst7 + pnorm[to_dp_list3[ii*2+14]] + nnorm[to_dp_list3[ii*2+15]] - pdot[to_dp_list3[ii*2+14]] + ndot[to_dp_list3[ii*2+15]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr7, ntr7, pu[to_dp_list3[ii*2+14]], nu[to_dp_list3[ii*2+15]]);

                        ii += 8;
                    }

                    while (ii < n_to_dp3) {
                        float *ptr0 = pvec[to_dp_list3[ii*2+0]];
                        float *ntr0 = nvec[to_dp_list3[ii*2+1]];
                        float x = dot(ptr0, ntr0, vec_length);
                        if (-2.0 * x + pnorm[to_dp_list3[ii*2+0]] + nnorm[to_dp_list3[ii*2+1]] - pdot[to_dp_list3[ii*2+0]] + ndot[to_dp_list3[ii*2+1]] < goal_norm - center_norm) TRY_ADDSA_TO_DST(ptr0, ntr0, pu[to_dp_list3[ii*2+0]], nu[to_dp_list3[ii*2+1]]);
                        ii++;
                    }
                } while (0);
            }
        }
        free(to_dp_list2);
        free(to_dp_list3);
        return __bucket_ndp;
        #else
        /*do {
            long cnt2 = 0, cnt3 = 0;
            for (long i = 0; i < num_pvec; i++){
                for (long j = 0; j < num_nvec; j++){
                    __attribute__ ((aligned (64))) float tmp[256] = {};
                    __attribute__ ((aligned (64))) float tmp1[256] = {};
                    add(tmp1, nvec[j], vec_length);
                    add(tmp1, pvec[i], vec_length);
                    add(tmp, center, vec_length);
                    add(tmp, nvec[j], vec_length);
                    sub(tmp, pvec[i], vec_length);
                    if (dot(tmp1, tmp1, vec_length) < goal_norm) cnt2++;
                    if (dot(tmp, tmp, vec_length) < goal_norm) {
                        cnt3++;
                    }
                }
            }
            try_add2 = -cnt2;
            try_add3 = -cnt3;
        } while (0);*/
        float **tmp = (float **)NEW_MAT(8, vec_length, sizeof(double));
        for (long Pnd = 0; Pnd < num_pvec; Pnd += SEARCHING_DP_BLOCK){
            for (long Nnd = 0; Nnd < num_nvec; Nnd += SEARCHING_DP_BLOCK){
                const long Pbound = (Pnd + SEARCHING_DP_BLOCK > num_pvec) ? num_pvec : Pnd + SEARCHING_DP_BLOCK;
                const long Nbound = (Nnd + SEARCHING_DP_BLOCK > num_nvec) ? num_nvec : Nnd + SEARCHING_DP_BLOCK;
                long pnd = Pnd;
                
                __m256 gn = _mm256_set1_ps(goal_norm);
                #if COLLECT_THREE_RED
                __m256 cn = _mm256_set1_ps(center_norm);
                #endif
                while (pnd < Pbound - 7) {
                    for (long i = 0; i < vec_length; i+=8){
                        __m256 p0 = _mm256_load_ps(pvec[pnd+0]+i);
                        __m256 p1 = _mm256_load_ps(pvec[pnd+1]+i);
                        __m256 p2 = _mm256_load_ps(pvec[pnd+2]+i);
                        __m256 p3 = _mm256_load_ps(pvec[pnd+3]+i);
                        __m256 p4 = _mm256_load_ps(pvec[pnd+4]+i);
                        __m256 p5 = _mm256_load_ps(pvec[pnd+5]+i);
                        __m256 p6 = _mm256_load_ps(pvec[pnd+6]+i);
                        __m256 p7 = _mm256_load_ps(pvec[pnd+7]+i);
                        _mm256_store_ps(tmp[0] + i, _mm256_add_ps(p0, p0));
                        _mm256_store_ps(tmp[1] + i, _mm256_add_ps(p1, p1));
                        _mm256_store_ps(tmp[2] + i, _mm256_add_ps(p2, p2));
                        _mm256_store_ps(tmp[3] + i, _mm256_add_ps(p3, p3));
                        _mm256_store_ps(tmp[4] + i, _mm256_add_ps(p4, p4));
                        _mm256_store_ps(tmp[5] + i, _mm256_add_ps(p5, p5));
                        _mm256_store_ps(tmp[6] + i, _mm256_add_ps(p6, p6));
                        _mm256_store_ps(tmp[7] + i, _mm256_add_ps(p7, p7));
                    }
                    __m256 b2 = _mm256_sub_ps(gn, _mm256_load_ps(pnorm + pnd));
                    #if COLLECT_THREE_RED
                    __m256 b3 = _mm256_sub_ps(_mm256_add_ps(gn, _mm256_loadu_ps(pdot + pnd)), _mm256_add_ps(cn, _mm256_load_ps(pnorm + pnd)));
                    #endif
                    long nnd = Nnd;
                    while (nnd < Nbound - 7) {
                        __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                        AVX2_DP_2X8(nvec[nnd+0], nvec[nnd+1], tmp, vec_length, dst0, dst1);
                        AVX2_DP_2X8(nvec[nnd+2], nvec[nnd+3], tmp, vec_length, dst2, dst3);
                        AVX2_DP_2X8(nvec[nnd+4], nvec[nnd+5], tmp, vec_length, dst4, dst5);
                        AVX2_DP_2X8(nvec[nnd+6], nvec[nnd+7], tmp, vec_length, dst6, dst7);
                        int cmp2[8];
                        long *cmp2l = (long *) cmp2;
                        #if COLLECT_THREE_RED
                        int cmp3[8];
                        long *cmp3l = (long *) cmp3;
                        #endif
                        cmp2[0] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst0, _mm256_broadcast_ss(&nnorm[nnd+0])), 30));
                        cmp2[1] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst1, _mm256_broadcast_ss(&nnorm[nnd+1])), 30));
                        cmp2[2] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst2, _mm256_broadcast_ss(&nnorm[nnd+2])), 30));
                        cmp2[3] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst3, _mm256_broadcast_ss(&nnorm[nnd+3])), 30));
                        cmp2[4] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst4, _mm256_broadcast_ss(&nnorm[nnd+4])), 30));
                        cmp2[5] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst5, _mm256_broadcast_ss(&nnorm[nnd+5])), 30));
                        cmp2[6] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst6, _mm256_broadcast_ss(&nnorm[nnd+6])), 30));
                        cmp2[7] = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst7, _mm256_broadcast_ss(&nnorm[nnd+7])), 30));
                        #if COLLECT_THREE_RED
                        cmp3[0] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+0]), _mm256_broadcast_ss(&ndot[nnd+0])), dst0), 30));
                        cmp3[1] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+1]), _mm256_broadcast_ss(&ndot[nnd+1])), dst1), 30));
                        cmp3[2] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+2]), _mm256_broadcast_ss(&ndot[nnd+2])), dst2), 30));
                        cmp3[3] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+3]), _mm256_broadcast_ss(&ndot[nnd+3])), dst3), 30));
                        cmp3[4] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+4]), _mm256_broadcast_ss(&ndot[nnd+4])), dst4), 30));
                        cmp3[5] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+5]), _mm256_broadcast_ss(&ndot[nnd+5])), dst5), 30));
                        cmp3[6] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+6]), _mm256_broadcast_ss(&ndot[nnd+6])), dst6), 30));
                        cmp3[7] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd+7]), _mm256_broadcast_ss(&ndot[nnd+7])), dst7), 30));
                        #endif
                        for (long i = 0; i < 4; i++){
                            if (cmp2l[i]){
                                CHECK_AND_ADD2_1X8(cmp2[2*i], nvec[nnd+2*i], nu[nnd+2*i], pvec+pnd, pu+pnd, TRY_ADDA_TO_DST);
                                CHECK_AND_ADD2_1X8(cmp2[2*i+1], nvec[nnd+2*i+1], nu[nnd+2*i+1], pvec+pnd, pu+pnd, TRY_ADDA_TO_DST);
                            }
                            #if COLLECT_THREE_RED
                            if (cmp3l[i]){
                                CHECK_AND_ADD2_1X8(cmp3[2*i], nvec[nnd+2*i], nu[nnd+2*i], pvec+pnd, pu+pnd, TRY_ADDAS_TO_DST);
                                CHECK_AND_ADD2_1X8(cmp3[2*i+1], nvec[nnd+2*i+1], nu[nnd+2*i+1], pvec+pnd, pu+pnd, TRY_ADDAS_TO_DST);
                            }
                            #endif
                        }
                        ///
                        /*float rdst[8][8] = {};
                        int rcmp2[8] = {};
                        int rcmp3[8] = {};
                        for (long i = 0; i < 8; i++){
                            for (long j = 0; j < 8; j++){
                                rdst[i][j] = dot(tmp[j], nvec[nnd+i], vec_length);
                                if (pnorm[pnd+j] + rdst[i][j] + nnorm[nnd+i] < goal_norm) rcmp2[i] |= (1 << j);
                                if ((center_norm + pnorm[pnd+j] - rdst[i][j] + nnorm[nnd+i] - pdot[pnd+j] +ndot[nnd+i]) < goal_norm) rcmp3[i] |= (1 << j);
                            }
                        }
                        for (long i = 0; i < 8; i++){
                            if (cmp2[i] != rcmp2[i]){
                                fprintf(stderr, "here\n");
                                fprintf(stderr, "here\n");
                            }
                            if (cmp3[i] != rcmp3[i]){
                                fprintf(stderr, "here\n");
                                fprintf(stderr, "here\n");
                            }
                        }*/
                        ///    
                        nnd += 8;
                    }
                    while (nnd < Nbound) {
                        __m256 dst;
                        AVX2_DP_1X8(nvec[nnd], tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], vec_length, dst);
                        int cmp2 = _mm256_movemask_ps(_mm256_cmp_ps(b2, _mm256_add_ps(dst, _mm256_broadcast_ss(&nnorm[nnd])), 30));
                        #if COLLECT_THREE_RED
                        int cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(_mm256_add_ps(_mm256_broadcast_ss(&nnorm[nnd]), _mm256_broadcast_ss(&ndot[nnd])), dst), 30));
                        #endif
                        CHECK_AND_ADD2_1X8(cmp2, nvec[nnd], nu[nnd], pvec+pnd, pu+pnd, TRY_ADDA_TO_DST);
                        #if COLLECT_THREE_RED
                        CHECK_AND_ADD2_1X8(cmp3, nvec[nnd], nu[nnd], pvec+pnd, pu+pnd, TRY_ADDAS_TO_DST);
                        #endif
                        ///
                        /*float rdst[8] = {};
                        int rcmp2 = 0;
                        int rcmp3 = 0;
                        for (long j = 0; j < 8; j++){
                            rdst[j] = dot(tmp[j], nvec[nnd], vec_length);
                            if (pnorm[pnd+j] + rdst[j] + nnorm[nnd] < goal_norm) rcmp2 |= (1 << j);
                            if ((center_norm + pnorm[pnd+j] - rdst[j] + nnorm[nnd] - pdot[pnd+j] +ndot[nnd]) < goal_norm) rcmp3 |= (1 << j);
                        }

                        if (cmp2 != rcmp2){
                            fprintf(stderr, "here\n");
                            fprintf(stderr, "here\n");
                        }
                        if (cmp3 != rcmp3){
                            fprintf(stderr, "here\n");
                            fprintf(stderr, "here\n");
                        }*/
                        ///
                        nnd++;
                    }
                    pnd += 8;
                }

                if (pnd < Pbound) {
                    const long nrem = Pbound - pnd;
                    long nnd = Nnd;
                    while (nnd < Nbound - 7) {
                        for (long i = 0; i < vec_length; i += 8){
                            for (long j = 0; j < nrem; j++){
                                __m256 p0 = _mm256_load_ps(pvec[pnd+j]+i);
                                _mm256_store_ps(tmp[j] + i, _mm256_add_ps(p0, p0));
                            }
                        }
                        __m256 dst[7];
                        int cmp2[8];
                        long *cmp2l = (long *) cmp2;
                        #if COLLECT_THREE_RED
                        int cmp3[8];
                        long *cmp3l = (long *) cmp3;
                        #endif
                        for (long j = 0; j < nrem; j++){
                            AVX2_DP_1X8(tmp[j], nvec[nnd], nvec[nnd+1], nvec[nnd+2], nvec[nnd+3], nvec[nnd+4], nvec[nnd+5], nvec[nnd+6], nvec[nnd+7], vec_length, dst[j]);
                        }
                        __m256 nn = _mm256_load_ps(nnorm + nnd);
                        #if COLLECT_THREE_RED
                        __m256 nd = _mm256_add_ps(nn, _mm256_loadu_ps(ndot + nnd));
                        #endif
                        for (long j = 0; j < nrem; j++){
                            cmp2[j] = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(gn, _mm256_broadcast_ss(&pnorm[pnd+j])), _mm256_add_ps(dst[j], nn), 30));
                            #if COLLECT_THREE_RED
                            __m256 b3 = _mm256_sub_ps(_mm256_add_ps(gn, _mm256_broadcast_ss(&pdot[pnd+j])), _mm256_add_ps(cn, _mm256_broadcast_ss(&pnorm[pnd+j])));
                            cmp3[j] = _mm256_movemask_ps(_mm256_cmp_ps(b3, _mm256_sub_ps(nd, dst[j]), 30));
                            #endif
                        }
                        for (long j = 0; j < nrem; j++){
                            CHECK_AND_ADD2_1X8(cmp2[j], pvec[pnd+j], pu[pnd+j], nvec+nnd, nu+nnd, TRY_ADDA_TO_DST);
                            #if COLLECT_THREE_RED
                            CHECK_AND_ADD2_1X8(cmp3[j], pvec[pnd+j], pu[pnd+j], nvec+nnd, nu+nnd, TRY_ADDSA_TO_DST);
                            #endif
                        }
                        ///
                        /*float rdst[8][8] = {};
                        int rcmp2[8] = {};
                        int rcmp3[8] = {};
                        for (long i = 0; i < 8; i++){
                            for (long j = 0; j < nrem; j++){
                                rdst[j][i] = dot(tmp[j], nvec[nnd+i], vec_length);
                                if (pnorm[pnd+j] + rdst[j][i] + nnorm[nnd+i] < goal_norm) rcmp2[j] |= (1 << i);
                                if ((center_norm + pnorm[pnd+j] - rdst[j][i] + nnorm[nnd+i] - pdot[pnd+j] +ndot[nnd+i]) < goal_norm) rcmp3[j] |= (1 << i);
                            }
                        }
                        for (long j = 0; j < nrem; j++){
                            if (cmp2[j] != rcmp2[j]){
                                fprintf(stderr, "here\n");
                                fprintf(stderr, "here\n");
                            }
                            if (cmp3[j] != rcmp3[j]){
                                fprintf(stderr, "here\n");
                                fprintf(stderr, "here\n");
                            }
                        }*/
                        ///
                        nnd+=8;
                    }

                    if (nnd < Nbound) {
                        const long pnrem = Pbound - pnd;
                        const long nnrem = Nbound - nnd;
                        for (long j = 0; j < pnrem; j++){
                            for (long i = 0; i < nnrem; i++){
                                float x = 2.0 * dot(pvec[pnd+j], nvec[nnd+i], vec_length);
                                if (x + nnorm[nnd+i] < goal_norm - pnorm[j+pnd]) TRY_ADDA_TO_DST(pvec[pnd+j], nvec[nnd+i], pu[pnd+j], nu[nnd+i]);
                                #if COLLECT_THREE_RED
                                if (-x + nnorm[nnd+i] + ndot[nnd+i] < goal_norm - center_norm - pnorm[pnd+j]+pdot[pnd+j]){
                                    TRY_ADDSA_TO_DST(pvec[pnd+j], nvec[nnd+i], pu[pnd+j], nu[nnd+i]);
                                }
                                #endif
                            }
                        }
                    }
                }
            }
        }
        FREE_MAT(tmp);
        return num_pvec * num_nvec;
        #endif
    }

    
    int _clear(){
        if (pvec) free(pvec);
        if (nvec) free(nvec);
        if (pu) free(pu);
        if (nu) free(nu);
        if (record_dp && pdot) free(pdot);
        if (record_dp && ndot) free(ndot);
        if (record_sh && psh) free(psh);
        if (record_sh && nsh) free(nsh);
        if (pnorm) FREE_VEC(pnorm);
        if (nnorm) FREE_VEC(nnorm);
        pnorm = NULL; nnorm = NULL;
        pvec = NULL; nvec = NULL;
        pu = NULL; nu = NULL;
        if (record_dp) { pdot = NULL; ndot = NULL; }
        if (record_sh) { psh = NULL; nsh = NULL; }
        return 0;
    }
    int _alloc(long size, bool p){
        if (p){
            if (size <= _psize) return 0;
            pvec = (float **) realloc(pvec, size * sizeof(float *));
            pu = (uint64_t *) realloc(pu, size * sizeof(uint64_t));
            //pnorm = (float *) realloc(pnorm, size * sizeof(float));
            if (record_dp) pdot = (float *) realloc(pdot, size * sizeof(float));
            if (record_sh) psh = (uint64_t *) realloc(psh, 4 * size * sizeof(uint64_t));
            _psize = size;
            return (pu && pvec && pdot && psh);
        } else {
            if (size <= _nsize) return 0;
            nvec = (float **) realloc(nvec, size * sizeof(float *));
            nu = (uint64_t *) realloc(nu, size * sizeof(uint64_t));
            //nnorm = (float *) realloc(nnorm, size * sizeof(float));
            if (record_dp) ndot = (float *) realloc(ndot, size * sizeof(float));
            if (record_sh) nsh = (uint64_t *) realloc(nsh, 4 * size * sizeof(uint64_t));
            _nsize = size;
            return (nu && nvec && ndot && nsh);
        }
    }
    long _psize = 0;
    long _nsize = 0;
};


int Pool::bgj1_Sieve(long log_level, long lps_auto_adj, long num_empty){
#if 1
    pthread_spinlock_t debug_lock;
    pthread_spin_init(&debug_lock, PTHREAD_PROCESS_SHARED);
    struct timeval bgj1_start_time, bgj1_end_time;
    gettimeofday(&bgj1_start_time, NULL);
    long num_epoch = 0;
    long num_bucket = 0;
    long sum_bucket_size = 0;
    double sort_time = 0.0;
    double bucket_time = 0.0;
    double search_time = 0.0;
    double insert_time = 0.0;
    double combine_time = 0.0;
    uint64_t bucket_nsh = 0;
    uint64_t bucket_ndp = 0;
    uint64_t search_nsh = 0;
    uint64_t search_ndp = 0;
    uint64_t try_add2 = 0;
    uint64_t succ_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add3 = 0;
#endif

    /* params */
    double alpha = BGJ1_BUCKET_ALPHA;
    double saturation_radius = 4.0/3.0;
    double saturation_ratio = 0.43;
    double one_epoch_ratio = 0.025;
    double improve_ratio = 0.77;
    double resort_ratio = 0.95;

    /* sort the pool */
    TIMER_START;
    sort_cvec();
    TIMER_END;
    sort_time += CURRENT_TIME;

    if (log_level == 0 && CSD > MIN_LOG_CSD){
        fprintf(stderr, "begin bgj1 sieve on context [%ld, %ld], gh = %.2f, pool size = %ld, %ld threads will be used",
                        index_l, index_r, sqrt(gh2), num_vec, num_threads);
    }
    if (log_level >= 1) {
        dlog("begin bgj1 sieve, sieving dimension = %ld, pool size = %ld", CSD, num_vec);
        if (log_level >= 3 && CSD > MIN_LOG_CSD){
            dlog("collect_three_red = %d", COLLECT_THREE_RED);
            dlog("bucketing_use_simhash = %d", BUCKETING_USE_SIMHASH);
            dlog("searching_use_simhash = %d", SEARCHING_USE_SIMHASH);
            dlog("bucketing_batch_size = %d", BUCKETING_BATCH_SIZE);
        }
    }
    
    long sieving_stucked = 0;

    /* main sieving procedure */
    while (!sieve_is_over(saturation_radius, saturation_ratio) && !sieving_stucked){
        //do {
        //    long stat[1024] = {};
        //    double igh = 1.0/sqrt(gh2);
        //    for (long i = 0; i < num_vec; i++){
        //        float x = (vec + i * vec_size)[-1];
        //        x = sqrt(x);
        //        int e = round(x * igh * 100) - 100;
        //        if (e < 0) e = 0;
        //        stat[e]++;
        //    }
        //    if (num_epoch == 0) std::cout << "\n";
        //    do { std::cout << "["; for (long __i = 0; __i < 50-1; __i++){ std::cout << stat[__i] << " "; } std::cout << stat[50-1] << "]\n"; } while (0);
        //} while (0);
        if (lps_auto_adj) {
            long adj_row = (CSD-1)/10 - 6;
            if (adj_row < 0) adj_row = 0;
            if (adj_row > 3) adj_row = 3;
            long adj_col = (num_epoch > 4) ? 4 : num_epoch;
            alpha = BGJ1_BUCKET_ALPHA + bgj1_lps_alpha_adj[adj_row][adj_col];
        }
        num_epoch++;
        const long goal_index = (long)(improve_ratio * num_vec);
        const float goal_norm = ((float *)(cvec+goal_index*cvec_size+4))[0];
        if (log_level >= 2 && CSD > MIN_LOG_CSD) dlog("epoch %ld, goal_norm = %.2f", num_epoch-1, sqrt(goal_norm));

        #if 1
        long _num_bucket = 0;
        long _sum_bucket_size = 0;
        double _bucket_time = 0.0;
        double _search_time = 0.0;
        uint64_t _bucket_nsh = 0;
        uint64_t _bucket_ndp = 0;
        uint64_t _search_nsh = 0;
        uint64_t _search_ndp = 0;
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
        bgj1_sol_t *local_buffer[MAX_NTHREADS];
        for (long i = 0; i < num_threads; i++) local_buffer[i] = new bgj1_sol_t;
        do {
            /* bucketing */
            typedef bgj1_bucket_t<COLLECT_THREE_RED, SEARCHING_USE_SIMHASH> bgj1_bucket;

            #define ADD_TO_LOCAL_BUCKET(__ptr, __cind, __x) do {                                                                \
                if (SEARCHING_USE_SIMHASH) {                                                                                    \
                    local_bucket[thread][__cind].add_vec(__ptr, __x, *((uint64_t *)(&__ptr[-4])), (uint64_t *)(&__ptr[-16]));   \
                } else {                                                                                                        \
                    local_bucket[thread][__cind].add_vec(__ptr, __x, *((uint64_t *)(&__ptr[-4])));                              \
                }                                                                                                                                                                                                                   \
            } while (0)

            #define CHECK_AND_ADD_TO_LOCAL_BUCKET(__cmp, __ptr, __cind, __dst) do {                             \
                if (__cmp){                                                                                     \
                    if (__cmp & 0xf){                                                                           \
                        if (__cmp & 0x3) {                                                                      \
                            if (__cmp & 0x1) ADD_TO_LOCAL_BUCKET(__ptr[0], __cind, (__dst)[0]);                 \
                            if (__cmp & 0x2) ADD_TO_LOCAL_BUCKET(__ptr[1], __cind, (__dst)[1]);                 \
                        }                                                                                       \
                        if (__cmp & 0xc) {                                                                      \
                            if (__cmp & 0x4) ADD_TO_LOCAL_BUCKET(__ptr[2], __cind, (__dst)[2]);                 \
                            if (__cmp & 0x8) ADD_TO_LOCAL_BUCKET(__ptr[3], __cind, (__dst)[3]);                 \
                        }                                                                                       \
                    }                                                                                           \
                    if (__cmp & 0xf0){                                                                          \
                        if (__cmp & 0x30) {                                                                     \
                            if (__cmp & 0x10) ADD_TO_LOCAL_BUCKET(__ptr[0], __cind+1, (__dst)[4]);              \
                            if (__cmp & 0x20) ADD_TO_LOCAL_BUCKET(__ptr[1], __cind+1, (__dst)[5]);              \
                        }                                                                                       \
                        if (__cmp & 0xc0) {                                                                     \
                            if (__cmp & 0x40) ADD_TO_LOCAL_BUCKET(__ptr[2], __cind+1, (__dst)[6]);              \
                            if (__cmp & 0x80) ADD_TO_LOCAL_BUCKET(__ptr[3], __cind+1, (__dst)[7]);              \
                        }                                                                                       \
                    }                                                                                           \
                }                                                                                               \
            } while (0)

            // not fast in our case
            #define CHECK_AND_ADD_TO_LOCAL_BUCKET_CTZ(__cmp, __ptr, __cind, __dst) do {                         \
                while (__cmp){                                                                                  \
                    int __r = __builtin_ctz(__cmp);                                                             \
                    __cmp -= (1 << __r);                                                                        \
                    ADD_TO_LOCAL_BUCKET(__ptr[__r & 0x3], __cind+(__r >> 2), (__dst)[__r]);                     \
                }                                                                                               \
            } while (0)

            long expect_bucket_size = pow(1.0 - alpha * alpha, CSD * 0.5) * num_vec;
            bgj1_bucket main_bucket[BUCKETING_BATCH_SIZE];
            bgj1_bucket **local_bucket = new bgj1_bucket*[num_threads];

            /* choose centers */
            long ind[BUCKETING_BATCH_SIZE];
            #if BUCKETING_USE_SIMHASH
            __attribute__ ((aligned (32))) uint64_t center_sh[BUCKETING_BATCH_SIZE*4];
            #endif
            float **center = (float **) NEW_MAT(BUCKETING_BATCH_SIZE, vec_length, sizeof(float));
            long num_try_find_center = 500;
            for (long i = 0; i < BUCKETING_BATCH_SIZE; i++){
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
                    #if BGJ1_USE_FARAWAY_CENTER
                    if ((CSD * ptr[CSD-1] * ptr[CSD-1] > ptr[-1]) && (num_try_find_center < 500)) pass = 0;
                    #endif
                } while(!pass);

                //float *ptr = vec + ind[i] * vec_size;
                uint64_t *_sh =  (uint64_t *)(&ptr[-16]);
                main_bucket[i].set_center(ptr); 
                #if BUCKETING_USE_SIMHASH
                center_sh[i*4+0] = _sh[0];
                center_sh[i*4+1] = _sh[1];
                center_sh[i*4+2] = _sh[2];
                center_sh[i*4+3] = _sh[3];
                #endif
                copy(center[i], ptr, vec_length);
            }

            if (log_level >= 4) {
                printf("choose %d centers, norm = [", BUCKETING_BATCH_SIZE);
                for (long i = 0; i < BUCKETING_BATCH_SIZE-1; i++) printf("%.2f ", sqrt(main_bucket[i].center_norm));
                printf("%.2f]\n", sqrt(main_bucket[BUCKETING_BATCH_SIZE-1].center_norm));
            }

            /* each thread collect vectors in local buckets */
            TIMER_START;
            #pragma omp parallel for 
            for (long thread = 0; thread < num_threads; thread++){
                /* prepare local buckets */
                local_bucket[thread] = new bgj1_bucket[BUCKETING_BATCH_SIZE];
                for (long i = 0; i < BUCKETING_BATCH_SIZE; i++){
                    local_bucket[thread][i]._alloc(0.5 * expect_bucket_size/num_threads, 1);
                    local_bucket[thread][i]._alloc(0.5 * expect_bucket_size/num_threads, 0);
                }
                //dlog("local bucket prepared");

                const long begin_ind = (thread*num_vec)/num_threads;
                const long end_ind = (thread*num_vec+num_vec)/num_threads;
                long ind = begin_ind;
                #if BUCKETING_USE_SIMHASH
                uint64_t __bucket_ndp = 0;
                if (thread == 0) _bucket_nsh += num_vec * BUCKETING_BATCH_SIZE;
                /* process 4 vectors tegother */
                float *to_dp_list[BUCKETING_DP_BUFFER*2];
                __attribute__ ((aligned (32))) float nptr_list[BUCKETING_DP_BUFFER];
                long n_to_dp = 0;
                __m256i sh_lbound = _mm256_set1_epi16(XPC_BGJ1_BUCKET_THRESHOLD);
                __m256i sh_ubound = _mm256_set1_epi16(256 - XPC_BGJ1_BUCKET_THRESHOLD);
                while (ind + 3 < end_ind){
                    __m256i sh0 = _mm256_loadu_si256((__m256i *)(cvec + (0+ind) * cvec_size));
                    __m256i sh1 = _mm256_loadu_si256((__m256i *)(cvec + (1+ind) * cvec_size));
                    __m256i sh2 = _mm256_loadu_si256((__m256i *)(cvec + (2+ind) * cvec_size));
                    __m256i sh3 = _mm256_loadu_si256((__m256i *)(cvec + (3+ind) * cvec_size));
                    uint32_t cmp_result[BUCKETING_BATCH_SIZE >> 2];

                    AVX2_POPCNT64_BATCHX4(cmp_result, center_sh, sh0, sh1, sh2, sh3, sh_ubound, sh_lbound);

                    float alpha_nptr[4];
                    alpha_nptr[0] = *((float *)(&((cvec + (0+ind) * cvec_size)[4]))) * alpha;
                    alpha_nptr[1] = *((float *)(&((cvec + (1+ind) * cvec_size)[4]))) * alpha;
                    alpha_nptr[2] = *((float *)(&((cvec + (2+ind) * cvec_size)[4]))) * alpha;
                    alpha_nptr[3] = *((float *)(&((cvec + (3+ind) * cvec_size)[4]))) * alpha;
                    
                    for (long i = 0; i < (BUCKETING_BATCH_SIZE >> 2); i++){
                        while (cmp_result[i]){
                            long r = __builtin_ctz(cmp_result[i]);
                            cmp_result[i] -= (3 << r);
                            to_dp_list[n_to_dp*2] = (float *)((cvec + (ind+(r >> 3)) * cvec_size)[5]);
                            to_dp_list[n_to_dp*2+1] = (float *)((i << 2) + ((r >> 1) & 0x3));
                            nptr_list[n_to_dp] = alpha_nptr[r >> 3];
                            n_to_dp++;
                        }
                    }

                    // process blocks
                    // I tried vectorized cmp but got no improvement / even worse
                    while (n_to_dp >= BUCKETING_DP_BLOCK){
                        __bucket_ndp += BUCKETING_DP_BLOCK;
                        n_to_dp -= BUCKETING_DP_BLOCK;
                        for (long i = 0; i < BUCKETING_DP_BLOCK; i += 8){
                            float *ptr0 = to_dp_list[n_to_dp*2+i*2+0];
                            float *ptr1 = to_dp_list[n_to_dp*2+i*2+2];
                            float *ptr2 = to_dp_list[n_to_dp*2+i*2+4];
                            float *ptr3 = to_dp_list[n_to_dp*2+i*2+6];
                            float *ptr4 = to_dp_list[n_to_dp*2+i*2+8];
                            float *ptr5 = to_dp_list[n_to_dp*2+i*2+10];
                            float *ptr6 = to_dp_list[n_to_dp*2+i*2+12];
                            float *ptr7 = to_dp_list[n_to_dp*2+i*2+14];
                            long cind0 = (long)to_dp_list[n_to_dp*2+i*2+1];
                            long cind1 = (long)to_dp_list[n_to_dp*2+i*2+3];
                            long cind2 = (long)to_dp_list[n_to_dp*2+i*2+5];
                            long cind3 = (long)to_dp_list[n_to_dp*2+i*2+7];
                            long cind4 = (long)to_dp_list[n_to_dp*2+i*2+9];
                            long cind5 = (long)to_dp_list[n_to_dp*2+i*2+11];
                            long cind6 = (long)to_dp_list[n_to_dp*2+i*2+13];
                            long cind7 = (long)to_dp_list[n_to_dp*2+i*2+15];
                            float __dst0, __dst1, __dst2, __dst3;
                            float __dst4, __dst5, __dst6, __dst7;
                            AVX2_DOT_PRODUCT_B8(ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, 
                                                center[cind0], center[cind1], center[cind2], 
                                                center[cind3], center[cind4], center[cind5],
                                                center[cind6], center[cind7], vec_length);
                            if (fabs(__dst0) > nptr_list[n_to_dp+i+0]) ADD_TO_LOCAL_BUCKET(ptr0, cind0, __dst0);
                            if (fabs(__dst1) > nptr_list[n_to_dp+i+1]) ADD_TO_LOCAL_BUCKET(ptr1, cind1, __dst1);
                            if (fabs(__dst2) > nptr_list[n_to_dp+i+2]) ADD_TO_LOCAL_BUCKET(ptr2, cind2, __dst2);
                            if (fabs(__dst3) > nptr_list[n_to_dp+i+3]) ADD_TO_LOCAL_BUCKET(ptr3, cind3, __dst3);
                            if (fabs(__dst4) > nptr_list[n_to_dp+i+4]) ADD_TO_LOCAL_BUCKET(ptr4, cind4, __dst4);
                            if (fabs(__dst5) > nptr_list[n_to_dp+i+5]) ADD_TO_LOCAL_BUCKET(ptr5, cind5, __dst5);
                            if (fabs(__dst6) > nptr_list[n_to_dp+i+6]) ADD_TO_LOCAL_BUCKET(ptr6, cind6, __dst6);
                            if (fabs(__dst7) > nptr_list[n_to_dp+i+7]) ADD_TO_LOCAL_BUCKET(ptr7, cind7, __dst7);
                        }
                    }

                    ind += 4;
                }

                /* the remaining part */
                // this part only ~1/10000 of total computations, 
                // so I do not optimize it now
                while (ind < end_ind){
                    uint64_t *sh0 = (uint64_t *)(cvec+ind*cvec_size);
                    float alpha_nptr0 = *((float *)(&((cvec+ind*cvec_size)[4]))) * alpha;
                    uint64_t wt[BUCKETING_BATCH_SIZE] = {};
                    POPCNT64_BATCHX1(wt, center_sh, sh0);
                    for (long i = 0; i < BUCKETING_BATCH_SIZE; i++){
                        if (wt[i] < XPC_BGJ1_BUCKET_THRESHOLD || wt[i] > (256 - XPC_BGJ1_BUCKET_THRESHOLD)) {
                            to_dp_list[n_to_dp*2] = (float *)((cvec + ind * cvec_size)[5]) ;    // ptr
                            to_dp_list[n_to_dp*2+1] = (float *)i;                               // cind
                            nptr_list[n_to_dp] = alpha_nptr0;
                            n_to_dp++;
                        }
                    }
                    ind++;
                }

                // process blocks
                while (n_to_dp >= BUCKETING_DP_BLOCK){
                    n_to_dp -= BUCKETING_DP_BLOCK;
                    __bucket_ndp += BUCKETING_DP_BLOCK;
                    for (long i = 0; i < BUCKETING_DP_BLOCK; i += 8){
                        float *ptr0 = to_dp_list[n_to_dp*2+i*2+0];
                        float *ptr1 = to_dp_list[n_to_dp*2+i*2+2];
                        float *ptr2 = to_dp_list[n_to_dp*2+i*2+4];
                        float *ptr3 = to_dp_list[n_to_dp*2+i*2+6];
                        float *ptr4 = to_dp_list[n_to_dp*2+i*2+8];
                        float *ptr5 = to_dp_list[n_to_dp*2+i*2+10];
                        float *ptr6 = to_dp_list[n_to_dp*2+i*2+12];
                        float *ptr7 = to_dp_list[n_to_dp*2+i*2+14];
                        long cind0 = (long)to_dp_list[n_to_dp*2+i*2+1];
                        long cind1 = (long)to_dp_list[n_to_dp*2+i*2+3];
                        long cind2 = (long)to_dp_list[n_to_dp*2+i*2+5];
                        long cind3 = (long)to_dp_list[n_to_dp*2+i*2+7];
                        long cind4 = (long)to_dp_list[n_to_dp*2+i*2+9];
                        long cind5 = (long)to_dp_list[n_to_dp*2+i*2+11];
                        long cind6 = (long)to_dp_list[n_to_dp*2+i*2+13];
                        long cind7 = (long)to_dp_list[n_to_dp*2+i*2+15];
                        float __dst0, __dst1, __dst2, __dst3;
                        float __dst4, __dst5, __dst6, __dst7;
                        AVX2_DOT_PRODUCT_B8(ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, 
                                            center[cind0], center[cind1], center[cind2], 
                                            center[cind3], center[cind4], center[cind5],
                                            center[cind6], center[cind7], vec_length);
                        if (fabs(__dst0) > nptr_list[n_to_dp+i+0]) ADD_TO_LOCAL_BUCKET(ptr0, cind0, __dst0);
                        if (fabs(__dst1) > nptr_list[n_to_dp+i+1]) ADD_TO_LOCAL_BUCKET(ptr1, cind1, __dst1);
                        if (fabs(__dst2) > nptr_list[n_to_dp+i+2]) ADD_TO_LOCAL_BUCKET(ptr2, cind2, __dst2);
                        if (fabs(__dst3) > nptr_list[n_to_dp+i+3]) ADD_TO_LOCAL_BUCKET(ptr3, cind3, __dst3);
                        if (fabs(__dst4) > nptr_list[n_to_dp+i+4]) ADD_TO_LOCAL_BUCKET(ptr4, cind4, __dst4);
                        if (fabs(__dst5) > nptr_list[n_to_dp+i+5]) ADD_TO_LOCAL_BUCKET(ptr5, cind5, __dst5);
                        if (fabs(__dst6) > nptr_list[n_to_dp+i+6]) ADD_TO_LOCAL_BUCKET(ptr6, cind6, __dst6);
                        if (fabs(__dst7) > nptr_list[n_to_dp+i+7]) ADD_TO_LOCAL_BUCKET(ptr7, cind7, __dst7);
                    }
                }

                while (n_to_dp > 0) {
                    n_to_dp--;
                    __bucket_ndp++;
                    float *ptr = to_dp_list[n_to_dp*2];
                    long cind = (long)to_dp_list[n_to_dp*2 + 1];
                    float x = dot_avx2(ptr, center[cind], vec_length);
                    if (fabs(x) > nptr_list[n_to_dp]) ADD_TO_LOCAL_BUCKET(ptr, cind, x);
                }

                pthread_spin_lock(&debug_lock);
                _bucket_ndp += __bucket_ndp;
                pthread_spin_unlock(&debug_lock);

                #else
                /* process 8 vectors tegother */
                __m256 sign_bit = _mm256_set1_ps(-0.0f);
                while (ind + 7 < end_ind){
                    float *ptr[8];
                    ptr[0] = vec + (0+ind) * vec_size;
                    ptr[1] = vec + (1+ind) * vec_size;
                    ptr[2] = vec + (2+ind) * vec_size;
                    ptr[3] = vec + (3+ind) * vec_size;
                    ptr[4] = vec + (4+ind) * vec_size;
                    ptr[5] = vec + (5+ind) * vec_size;
                    ptr[6] = vec + (6+ind) * vec_size;
                    ptr[7] = vec + (7+ind) * vec_size;

                    __attribute__ ((aligned (32))) float alpha_nptr0[4];
                    __attribute__ ((aligned (32))) float alpha_nptr1[4];
                    alpha_nptr0[0] = ptr[0][-1] * alpha;
                    alpha_nptr0[1] = ptr[1][-1] * alpha;
                    alpha_nptr0[2] = ptr[2][-1] * alpha;
                    alpha_nptr0[3] = ptr[3][-1] * alpha;
                    alpha_nptr1[0] = ptr[4][-1] * alpha;
                    alpha_nptr1[1] = ptr[5][-1] * alpha;
                    alpha_nptr1[2] = ptr[6][-1] * alpha;
                    alpha_nptr1[3] = ptr[7][-1] * alpha;
                    __m128 anptr_half0 = _mm_load_ps(alpha_nptr0);
                    __m128 anptr_half1 = _mm_load_ps(alpha_nptr1);
                    __m256 anptr0 = _mm256_insertf128_ps(_mm256_castps128_ps256(anptr_half0), anptr_half0, 1);
                    __m256 anptr1 = _mm256_insertf128_ps(_mm256_castps128_ps256(anptr_half1), anptr_half1, 1);
                    
                    __attribute__ ((aligned (32))) float dst[8 * BUCKETING_BATCH_SIZE];
                    __attribute__ ((aligned (32))) int cmp[BUCKETING_BATCH_SIZE];
                    unsigned cind = 0;
                    while (cind + 3 < BUCKETING_BATCH_SIZE){
                        AVX2_DP_CMP_4X4(center[cind], center[cind+1], center[cind+2], center[cind+3], ptr[0], ptr[1], ptr[2], ptr[3], vec_length, anptr0, (&dst[cind*8]), cmp[cind], cmp[cind+1]);                   
                        AVX2_DP_CMP_4X4(center[cind], center[cind+1], center[cind+2], center[cind+3], ptr[4], ptr[5], ptr[6], ptr[7], vec_length, anptr1, (&dst[cind*8+16]), cmp[cind+2], cmp[cind+3]);                   
                        cind += 4;
                    }

                    // insert to the buckets, maybe we can optimize it furthur
                    for (unsigned _cind = 0; _cind < BUCKETING_BATCH_SIZE - 3; _cind+=4){
                        CHECK_AND_ADD_TO_LOCAL_BUCKET(cmp[_cind+0], ptr, _cind, &dst[_cind*8]);
                        CHECK_AND_ADD_TO_LOCAL_BUCKET(cmp[_cind+1], ptr, _cind+2, &dst[_cind*8+8]);
                        CHECK_AND_ADD_TO_LOCAL_BUCKET(cmp[_cind+2], (&ptr[4]), _cind, &dst[_cind*8+16]);
                        CHECK_AND_ADD_TO_LOCAL_BUCKET(cmp[_cind+3], (&ptr[4]), _cind+2, &dst[_cind*8+24]);
                    }
                    
                    // we always choose BUCKETING_BATCH_SIZE divided by 4,
                    // so we do not optimize here
                    while (cind < BUCKETING_BATCH_SIZE){
                        float __dst0, __dst1, __dst2, __dst3;
                        AVX2_DOT_PRODUCT_1X4(center[cind], ptr[0], ptr[1], ptr[2], ptr[3], vec_length);                    
                        if (fabs(__dst0) > alpha_nptr0[0]) ADD_TO_LOCAL_BUCKET(ptr[0], cind, __dst0);
                        if (fabs(__dst1) > alpha_nptr0[1]) ADD_TO_LOCAL_BUCKET(ptr[1], cind, __dst1);
                        if (fabs(__dst2) > alpha_nptr0[2]) ADD_TO_LOCAL_BUCKET(ptr[2], cind, __dst2);
                        if (fabs(__dst3) > alpha_nptr0[3]) ADD_TO_LOCAL_BUCKET(ptr[3], cind, __dst3);
                        AVX2_DOT_PRODUCT_1X4(center[cind], ptr[4], ptr[5], ptr[6], ptr[7], vec_length);                    
                        if (fabs(__dst0) > alpha_nptr1[0]) ADD_TO_LOCAL_BUCKET(ptr[4], cind, __dst0);
                        if (fabs(__dst1) > alpha_nptr1[1]) ADD_TO_LOCAL_BUCKET(ptr[5], cind, __dst1);
                        if (fabs(__dst2) > alpha_nptr1[2]) ADD_TO_LOCAL_BUCKET(ptr[6], cind, __dst2);
                        if (fabs(__dst3) > alpha_nptr1[3]) ADD_TO_LOCAL_BUCKET(ptr[7], cind, __dst3);
                        cind++;
                    }
                    ind += 8;
                }
                
                /* the remaining part */
                while(ind < end_ind){
                    float *ptr = (float *)((cvec + (0+ind) * cvec_size)[5]);
                    float alpha_nptr = *((float *)(&((cvec + (0+ind) * cvec_size)[4]))) * alpha;
                    long cind = 0;
                    while (cind + 3 < BUCKETING_BATCH_SIZE){
                        float __dst0, __dst1, __dst2, __dst3;
                        AVX2_DOT_PRODUCT_1X4(ptr, center[cind], center[cind+1], center[cind+2], center[cind+3], vec_length);
                        if (fabs(__dst0) > alpha_nptr) ADD_TO_LOCAL_BUCKET(ptr, cind, __dst0);
                        if (fabs(__dst1) > alpha_nptr) ADD_TO_LOCAL_BUCKET(ptr, cind+1, __dst1);
                        if (fabs(__dst2) > alpha_nptr) ADD_TO_LOCAL_BUCKET(ptr, cind+2, __dst2);
                        if (fabs(__dst3) > alpha_nptr) ADD_TO_LOCAL_BUCKET(ptr, cind+3, __dst3);
                        cind += 4;
                    }
                    while (cind < BUCKETING_BATCH_SIZE){
                        float x = dot_avx2(ptr, center[cind], vec_length);
                        if (fabs(x) > alpha_nptr) ADD_TO_LOCAL_BUCKET(ptr, cind, x);
                        cind++;
                    }
                    ind++;
                }

                /* update bucket_ndp */
                if (thread == 0) _bucket_ndp += num_vec * BUCKETING_BATCH_SIZE;
                #endif
            }
            TIMER_END;
            _bucket_time += CURRENT_TIME;

            /* combine the data to main buckets */
            TIMER_START;
            #pragma omp parallel for
            for (long i = 0; i < BUCKETING_BATCH_SIZE; i++){
                bgj1_bucket *tmp[MAX_NTHREADS];
                for (long j = 0; j < num_threads; j++) tmp[j] = &local_bucket[j][i];
                main_bucket[i].combine(tmp, num_threads);
            }
            TIMER_END;
            combine_time += CURRENT_TIME;

            if (log_level >= 4) {
                for (long i = 0; i < BUCKETING_BATCH_SIZE; i++){
                    printf("# main bucket %ld\n", i);
                    printf("# nsize = %ld, psize = %ld\n", main_bucket[i].num_nvec, main_bucket[i].num_pvec);
                    int alpha_pvec_pass = 1, alpha_nvec_pass = 1;
                    int simhash_pass = 1;
                    for (long j = 0; j < main_bucket[i].num_pvec; j++){
                        float *ptr = main_bucket[i].pvec[j];
                        uint64_t *rsh = (uint64_t *)(ptr - 16);
                        uint64_t ru = *(uint64_t *)(ptr - 4);
                        uint64_t *sh = &main_bucket[i].psh[4*j];
                        if (sh[0] != rsh[0] || sh[1] != rsh[1] || sh[2] != rsh[2] || sh[3] != rsh[3]) simhash_pass = 0;
                        if (ru != main_bucket[i].pu[j]) simhash_pass = 0;
                        float x = dot(ptr, main_bucket[i].center, vec_length);
                        if (fabs(x - main_bucket[i].pdot[j]) > 10) {
                            alpha_pvec_pass = 0;
                        }
                        x /= main_bucket[i].pvec[j][-1];
                        if (fabs(x) < 0.314) {
                            alpha_pvec_pass = 0;
                        }
                    }
                    for (long j = 0; j < main_bucket[i].num_nvec; j++){
                        float *ptr = main_bucket[i].nvec[j];
                        uint64_t *rsh = (uint64_t *)(ptr - 16);
                        uint64_t ru = *(uint64_t *)(ptr - 4);
                        uint64_t *sh = &main_bucket[i].nsh[4*j];
                        if (sh[0] != rsh[0] || sh[1] != rsh[1] || sh[2] != rsh[2] || sh[3] != rsh[3]) simhash_pass = 0;
                        if (ru != main_bucket[i].nu[j]) simhash_pass = 0;
                        float x = dot(ptr, main_bucket[i].center, vec_length);
                        if (fabs(x - main_bucket[i].ndot[j]) > 10) alpha_nvec_pass = 0;
                        x /= main_bucket[i].nvec[j][-1];
                        if (fabs(x) < 0.314) alpha_nvec_pass = 0;
                    }
                    printf("# simhash test: %d\n", simhash_pass);
                    printf("# dot test of pvecs: %d\n", alpha_pvec_pass);
                    printf("# dot test of nvecs: %d\n", alpha_nvec_pass);
                }
            }

            /* free local buckets and centers */
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                for (long i = 0; i < BUCKETING_BATCH_SIZE; i++){
                    local_bucket[thread][i]._clear();
                }
                delete[] local_bucket[thread];
            }
            delete[] local_bucket;
            FREE_MAT(center);
             
            _num_bucket += BUCKETING_BATCH_SIZE;
            for (long i = 0; i < BUCKETING_BATCH_SIZE; i++) _sum_bucket_size += main_bucket[i].num_nvec + main_bucket[i].num_pvec;
            if (log_level >= 4) {
                printf("bucket size = [");
                for (long i = 0; i < BUCKETING_BATCH_SIZE - 1; i++) printf("%ld ", main_bucket[i].num_nvec + main_bucket[i].num_pvec);
                printf("%ld]\n", main_bucket[BUCKETING_BATCH_SIZE-1].num_nvec + main_bucket[BUCKETING_BATCH_SIZE-1].num_pvec);
            }
            if (log_level >= 3 && CSD > MIN_LOG_CSD) {
                #if BUCKETING_USE_SIMHASH
                dlog("bucketing done, avg bucket size = %ld, num_sh = %ld, num_dp = %ld", _sum_bucket_size/_num_bucket, _bucket_nsh, _bucket_ndp);
                dlog("bucketing speed: %f bucket/s, %fG simhash/s, %fGFLOPS", _num_bucket/_bucket_time, _bucket_nsh/_bucket_time/1073741824.0, CSD * 2 * _bucket_ndp/_bucket_time/1073741824.0);
                #else
                double speed = CSD * 2 * _bucket_ndp/_bucket_time/1073741824.0;
                dlog("bucketing done, avg bucket size = %ld, num_dp = %ld", _sum_bucket_size/_num_bucket, _bucket_ndp);
                dlog("bucketing speed: %f bucket/s, %f GFLOPS", _num_bucket/_bucket_time, speed);
                #endif
            }

            /* searching */
            TIMER_START;
            pthread_spinlock_t bucket_list_lock;
            pthread_spin_init(&bucket_list_lock, PTHREAD_PROCESS_SHARED);
            long nrem_bucket = BUCKETING_BATCH_SIZE;
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++){
                long __search_nsh = 0;
                long __search_ndp = 0;
                uint64_t __try_add2 = 0;
                uint64_t __succ_add2 = 0;
                uint64_t __try_add3 = 0;
                uint64_t __succ_add3 = 0;
                while (nrem_bucket) {
                    bgj1_bucket *bkt = NULL;
                    // get a bucket
                    pthread_spin_lock(&bucket_list_lock);
                    if (nrem_bucket > 0) {
                        bkt = &main_bucket[nrem_bucket-1];
                        nrem_bucket--;
                        if (__succ_add2 + __succ_add3 > num_empty + sorted_index - goal_index){
                            bkt = NULL;
                            nrem_bucket = 0;
                        }
                    }
                    pthread_spin_unlock(&bucket_list_lock);
                    if (bkt == NULL) continue;
                    bkt->vec_length = vec_length;
                    __search_ndp += bkt->search_naive(local_buffer[thread], uid, goal_norm);
                    __search_ndp += bkt->search_pp(local_buffer[thread], uid, goal_norm);
                    __search_ndp += bkt->search_nn(local_buffer[thread], uid, goal_norm);
                    __search_ndp += bkt->search_np(local_buffer[thread], uid, goal_norm);
                    #if SEARCHING_USE_SIMHASH
                    //__search_nsh += (bkt->num_nvec + bkt->num_pvec) * (bkt->num_nvec + bkt->num_pvec - 1) / 2;
                    __search_nsh += bkt->num_nvec * bkt->num_pvec;
                    #endif
                    __try_add2 += bkt->try_add2;
                    __try_add3 += bkt->try_add3;
                    __succ_add2 += bkt->succ_add2;
                    __succ_add3 += bkt->succ_add3;
                }
                pthread_spin_lock(&debug_lock);
                _search_nsh += __search_nsh;
                _search_ndp += __search_ndp;
                _try_add2 += __try_add2;
                _try_add3 += __try_add3;
                _succ_add2 += __succ_add2;
                _succ_add3 += __succ_add3;
                pthread_spin_unlock(&debug_lock);
            }
            TIMER_END;
            _search_time += CURRENT_TIME;

            if (log_level >= 3 && CSD > MIN_LOG_CSD) {
                #if SEARCHING_USE_SIMHASH
                dlog("searching done, num_sh = %ld, num_dp = %ld, %ld solutions found", _search_nsh, _search_ndp, num_total_sol);
                dlog("searching speed: %f bucket/s, %fG simhash/s, %fGFLOPS", _num_bucket/_search_time, _search_nsh/_search_time/1073741824.0, CSD * 2 * _search_ndp/_search_time/1073741824.0);
                #else
                double speed = CSD * 2 * _search_ndp/_search_time/1073741824.0;
                dlog("searching done, num_dp = %ld, %ld solutions found", _search_ndp, _succ_add2 + _succ_add3);
                dlog("searching speed: %f bucket/s, %f GFLOPS", _num_bucket/_search_time, speed);
                #endif
                dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld", _try_add2, _try_add3, _succ_add2, _succ_add3);
            }

            // check if we get stucked or finished
            num_total_sol = 0;
            for (long i = 0; i < num_threads; i++){
                num_total_sol += local_buffer[i]->num_sol();
            }
            if (num_total_sol < last_num_total_sol + stucktime * BUCKETING_BATCH_SIZE/3){
                stucktime++;
            } else {
                stucktime = 0;
                last_num_total_sol = num_total_sol;
            }
            if (num_total_sol > one_epoch_ratio * num_vec) rel_collection_stop = true;
            if (stucktime > 3) {
                sieving_stucked = 1;
                rel_collection_stop = true;
            } 
        } while (!rel_collection_stop);

        if (log_level >= 1 && CSD > MIN_LOG_CSD) {
            double speed;
            dlog("solution collect done, found %ld solutions, bucketing time = %fs, searching time = %fs", num_total_sol, _bucket_time, _search_time);
            if (log_level >= 2) {
                #if BUCKETING_USE_SIMHASH
                dlog("bucketing speed: %f bucket/s, %fG simhash/s, %fGFLOPS", _num_bucket/_bucket_time, _bucket_nsh/_bucket_time/1073741824.0, CSD * 2 * _bucket_ndp/_bucket_time/1073741824.0);
                #else
                speed = CSD * 2 * _bucket_ndp/_bucket_time/1073741824.0;
                dlog("bucketing speed: %f bucket/s, %f GFLOPS", _num_bucket/_bucket_time, speed);
                #endif
                
                #if SEARCHING_USE_SIMHASH
                dlog("searching speed: %f bucket/s, %fG simhash/s, %fGFLOPS", _num_bucket/_search_time, _search_nsh/_search_time/1073741824.0, CSD * 2 * _search_ndp/_search_time/1073741824.0);
                #else
                speed = CSD * 2 * _search_ndp/_search_time/1073741824.0;
                dlog("searching speed: %f bucket/s, %f GFLOPS", _num_bucket/_search_time, speed);
                #endif
                dlog("num bucket = %ld, avg bucket size = %ld", _num_bucket, (long)(_sum_bucket_size/(0.000001+_num_bucket)));
                dlog("bucket cost = %ld dp/sol, search cost = %ld dp/sol", _bucket_ndp/num_total_sol, _search_ndp/num_total_sol);
                dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld", _try_add2, _try_add3, _succ_add2, _succ_add3);
            }
        }
        
        do {
            num_bucket += _num_bucket;
            sum_bucket_size += _sum_bucket_size;
            bucket_time += _bucket_time;
            search_time += _search_time;
            bucket_nsh += _bucket_nsh;
            bucket_ndp += _bucket_ndp;
            search_nsh += _search_nsh;
            search_ndp += _search_ndp;
            try_add2 += _try_add2;
            try_add3 += _try_add3;
            succ_add2 += _succ_add2;
            succ_add3 += _succ_add3;
        } while (0);

        
        /* check all uid of solutions have been inserted and the norm of solutinos is small enough */
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
        
        TIMER_START;
        /* insert solutions */
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
                        //if (CSD == 80 && Uniform_long(100) == 0) {
                        //    float n1 = sqrt(local_buffer[thread]->a_list[i*2][-1]/gh2) - 1;
                        //    float n2 = sqrt(local_buffer[thread]->a_list[i*2+1][-1]/gh2) - 1;
                        //    float nd = sqrt(dst[-1]/gh2) - 1;
                        //    float theta = dot_avx2(local_buffer[thread]->a_list[i*2], local_buffer[thread]->a_list[i*2+1], vec_length);
                        //    theta /= n1 * n2;
                        //    theta = fabs(theta);
                        //    printf("# 2red, [%ld, %ld] --> %ld\n", (long)(100*n1), (long)(100*n2), (long)(100*nd));
                        //}
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
                        //if (CSD == 80 && Uniform_long(100) == 0) {
                        //    float n1 = sqrt(local_buffer[thread]->aa_list[i*3][-1]/gh2) - 1;
                        //    float n2 = sqrt(local_buffer[thread]->aa_list[i*3+1][-1]/gh2) - 1;
                        //    float n3 = sqrt(local_buffer[thread]->aa_list[i*3+2][-1]/gh2) - 1;
                        //    float nd = sqrt(dst[-1]/gh2) - 1;
                        //    printf("# 3red, [%ld, %ld, %ld] --> %ld\n", (long)(100*n1), (long)(100*n2), (long)(100*n3), (long)(100*nd));
                        //}
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
                                fprintf(stderr, "[Error] Pool::bgj1_Sieve: erase uid of non inserted vector failed, ignored.\n");
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
                                fprintf(stderr, "[Error] Pool::bgj1_Sieve: erase uid of non inserted vector failed, ignored.\n");
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
                                fprintf(stderr, "[Error] Pool::bgj1_Sieve: erase uid of non inserted vector failed, ignored.\n");
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
                                fprintf(stderr, "[Error] Pool::bgj1_Sieve: erase uid of non inserted vector failed, ignored.\n");
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
                                fprintf(stderr, "[Error] Pool::bgj1_Sieve: erase uid of non inserted vector failed, ignored.\n");
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

            //if (num_total_insert > (num_vec * (1-improve_ratio))) num_total_insert = (num_vec * (1-improve_ratio));
            
            if (num_empty >= 0) {
                long insert_to_empty = (num_empty > num_total_insert) ? num_total_insert : num_empty;
                #pragma omp parallel for 
                for (long i = 0; i < insert_to_empty; i++){
                    long *cdst = cvec + cvec_size * (num_vec + i);
                    float *dst = vec + vec_size * (num_vec + i);
                    float *src = to_insert_vec + (num_total_insert - i - 1) * vec_size;
                    if (!uid->safely_check_uid(*((uint64_t *)(&(src[-4]))))){
                        fprintf(stderr, "[Error] Pool::bgj1_Sieve: uid of new pool vector not in the UidHashTable, ignored.\n");
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
                    fprintf(stderr, "[Error] Pool::bgj1_Sieve: erase uid of old pool vector failed, ignored.\n");
                }
                if (!uid->safely_check_uid(*((uint64_t *)(&(src[-4]))))){
                    fprintf(stderr, "[Error] Pool::bgj1_Sieve: uid of new pool vector not in the UidHashTable, ignored.\n");
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
            dlog("insert %ld solutions in %fs", num_total_insert, insert_time);
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
    if (log_level >= 1) {
        if (sieving_stucked){
            dlog("sieving stucked, aborted.");
        } else {
            dlog("sieving done.");
        }
        double speed;
        dlog("bucketing time = %fs, searching time = %fs, sort time = %fs, insert time = %fs", bucket_time, search_time, sort_time, insert_time);
        #if BUCKETING_USE_SIMHASH
        dlog("bucketing speed: %f bucket/s, %fG simhash/s, %fGFLOPS", num_bucket/bucket_time, bucket_nsh/bucket_time/1073741824.0, CSD * 2 * bucket_ndp/bucket_time/1073741824.0);
        #else
        speed = CSD * 2 * bucket_ndp/bucket_time/1073741824.0;
        dlog("bucketing speed: %f bucket/s, %f GFLOPS", num_bucket/bucket_time, speed);
        #endif
        
        #if SEARCHING_USE_SIMHASH
        dlog("searching speed: %f bucket/s, %fG simhash/s, %fGFLOPS", num_bucket/search_time, search_nsh/search_time/1073741824.0, CSD * 2 * search_ndp/search_time/1073741824.0);
        #else
        speed = CSD * 2 * search_ndp/search_time/1073741824.0;
        dlog("searching speed: %f bucket/s, %f GFLOPS", num_bucket/search_time, speed);
        #endif
        dlog("num bucket = %ld, avg bucket size = %ld", num_bucket, (long)(sum_bucket_size/(0.000001+num_bucket)));
        dlog("bucket cost = %ld dp/sol, search cost = %ld dp/sol", (long)(bucket_ndp/(succ_add2 + succ_add3 + 0.00001)), (long)(search_ndp/(succ_add2 + succ_add3+0.00001)));
        dlog("try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld, succ_add3 = %ld\n", try_add2, try_add3, succ_add2, succ_add3);
    }
    if (log_level == 0){
        gettimeofday(&bgj1_end_time, NULL);
        double tt = bgj1_end_time.tv_sec-bgj1_start_time.tv_sec+ (double)(bgj1_end_time.tv_usec-bgj1_start_time.tv_usec)/1000000.0;
        if (sieving_stucked) {
            fprintf(stderr, "get stucked.");
        }
        if (CSD > MIN_LOG_CSD) fprintf(stderr, "done, time = %.2fs\n", tt);
    }
    return sieving_stucked;
}
