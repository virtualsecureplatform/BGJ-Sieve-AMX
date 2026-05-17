#include "../include/pool_epi8.h"
#include "../include/bgj_epi8.h"

#include <sys/time.h>

#define NSH_DEFAULT_DHDIM 24
#define NSH_DEFAULT_SHSIZE 64
#define NSH_HALF_SHSIZE 32

#define NSH_L1_BLOCK 256
#define NSH_L2_BLOCK 8192
#define NSH_M_BLOCK 262144
#define NSH_DIAG_FIRST 1

#if 1
struct timeval _nsh_timer_start[MAX_NTHREADS], _nsh_timer_end[MAX_NTHREADS];
double _nsh_time_curr[MAX_NTHREADS];

#define TIMER_START do {                                                        \
        gettimeofday(&_nsh_timer_start[omp_get_thread_num()], NULL);            \
    } while (0)

#define TIMER_END do {                                                          \
        gettimeofday(&_nsh_timer_end[omp_get_thread_num()], NULL);              \
        _nsh_time_curr[omp_get_thread_num()] =                                                            \
            (_nsh_timer_end[omp_get_thread_num()].tv_sec-_nsh_timer_start[omp_get_thread_num()].tv_sec)+                    \
            (double)(_nsh_timer_end[omp_get_thread_num()].tv_usec-_nsh_timer_start[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_nsh_time_curr[omp_get_thread_num()])
#endif

#pragma region
#define ADDA_TO_BUFFER(__ind, __jnd) do {                                                           \
    if (*ptr_buffer_num == *ptr_buffer_size) {                                                      \
        *ptr_buffer_size *= 2;                                                                      \
        *ptr_buffer_size += 64;                                                                     \
        *ptr_buffer = (uint32_t *) realloc(*ptr_buffer, 3 * (*ptr_buffer_size) * sizeof(uint32_t)); \
    }                                                                                               \
    (*ptr_buffer)[(*ptr_buffer_num) * 3] = 0;                                                       \
    (*ptr_buffer)[(*ptr_buffer_num) * 3 + 1] = __ind;                                               \
    (*ptr_buffer)[(*ptr_buffer_num) * 3 + 2] = __jnd;                                               \
    (*ptr_buffer_num)++;                                                                            \
} while (0);
#define ADDS_TO_BUFFER(__ind, __jnd) do {                                                           \
    if (*ptr_buffer_num == *ptr_buffer_size) {                                                      \
        *ptr_buffer_size *= 2;                                                                      \
        *ptr_buffer_size += 64;                                                                     \
        *ptr_buffer = (uint32_t *) realloc(*ptr_buffer, 3 * (*ptr_buffer_size) * sizeof(uint32_t)); \
    }                                                                                               \
    (*ptr_buffer)[(*ptr_buffer_num) * 3] = 1;                                                       \
    (*ptr_buffer)[(*ptr_buffer_num) * 3 + 1] = __ind;                                               \
    (*ptr_buffer)[(*ptr_buffer_num) * 3 + 2] = __jnd;                                               \
    (*ptr_buffer_num)++;                                                                            \
} while (0);
#define CHECK_AND_ADD8x8(__cmp, __ibias, __jbias, __add_func) do {  \
    while (__cmp) {                                                 \
        int __r = __builtin_ctzl(__cmp);                            \
        __cmp -= (1UL << __r);                                      \
        __add_func(__ibias + (__r >> 3), __jbias + (__r & 0x7));    \
    }                                                               \
} while (0)
#pragma endregion

#pragma region
// if pop(u1[i] ^ u2[j]) >= ub, cmpp[i*8+j] = 1
// if pop(u1[i] ^ u2[j]) <= lb, cmpn[i*8+j] = 1
template <uint32_t nb>
template <uint32_t shsize>
inline void Pool_epi8_t<nb>::_vshdot_check8x8(uint64_t *cmpp, uint64_t *cmpn, uint8_t *u1, uint8_t *u2, __m512i lb, __m512i ub) {
    if (shsize == 64) {
        static const __m512i pf512_0 = _mm512_set_epi32(0xfe, 0x0e, 0xfc, 0x0c, 0xfa, 0x0a, 0xf8, 0x08, 
                                                      0xf6, 0x06, 0xf4, 0x04, 0xf2, 0x02, 0xf0, 0x00);
        static const __m512i pf512_1 = _mm512_set_epi64(0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01, 0x08, 0x00);
        static const __m512i pf512_2 = _mm512_set_epi64(0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04);
        static const __m512i pf512_3 = _mm512_set_epi64(0x0b, 0x0a, 0x03, 0x02, 0x09, 0x08, 0x01, 0x00);
        static const __m512i pf512_4 = _mm512_set_epi64(0x0f, 0x0e, 0x07, 0x06, 0x0d, 0x0c, 0x05, 0x04);
        static const __m512i pf512_5 = _mm512_set_epi64(0x0b, 0x0a, 0x09, 0x08, 0x03, 0x02, 0x01, 0x00);
        static const __m512i pf512_6 = _mm512_set_epi64(0x0f, 0x0e, 0x0d, 0x0c, 0x07, 0x06, 0x05, 0x04);
        #ifdef __AVX512_VNNI__
        __m512i acc[8];
        for (long i = 0; i < 8; i+= 2) {
            __m512i u10 = _mm512_load_si512((__m512i *)(u1 + i * shsize));
            __m512i u11 = _mm512_load_si512((__m512i *)(u1 + (i + 1) * shsize));
            __m512i u20 = _mm512_load_si512((__m512i *)(u2 + 0 * shsize));
            __m512i u21 = _mm512_load_si512((__m512i *)(u2 + 1 * shsize));
            __m512i u22 = _mm512_load_si512((__m512i *)(u2 + 2 * shsize));
            __m512i u23 = _mm512_load_si512((__m512i *)(u2 + 3 * shsize));
            __m512i u24 = _mm512_load_si512((__m512i *)(u2 + 4 * shsize));
            __m512i u25 = _mm512_load_si512((__m512i *)(u2 + 5 * shsize));
            __m512i u26 = _mm512_load_si512((__m512i *)(u2 + 6 * shsize));
            __m512i u27 = _mm512_load_si512((__m512i *)(u2 + 7 * shsize));
            __m512i acc00 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u20));
            __m512i acc01 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u21));
            __m512i acc02 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u22));
            __m512i acc03 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u23));
            __m512i acc04 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u24));
            __m512i acc05 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u25));
            __m512i acc06 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u26));
            __m512i acc07 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u27));
            __m512i acc10 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u20));
            __m512i acc11 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u21));
            __m512i acc12 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u22));
            __m512i acc13 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u23));
            __m512i acc14 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u24));
            __m512i acc15 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u25));
            __m512i acc16 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u26));
            __m512i acc17 = _mm512_popcnt_epi64(_mm512_xor_si512(u11, u27));
            acc00 = _mm512_permutex2var_epi32(acc00, pf512_0, acc02);
            acc01 = _mm512_permutex2var_epi32(acc01, pf512_0, acc03);
            acc04 = _mm512_permutex2var_epi32(acc04, pf512_0, acc06);
            acc05 = _mm512_permutex2var_epi32(acc05, pf512_0, acc07);
            acc10 = _mm512_permutex2var_epi32(acc10, pf512_0, acc12);
            acc11 = _mm512_permutex2var_epi32(acc11, pf512_0, acc13);
            acc14 = _mm512_permutex2var_epi32(acc14, pf512_0, acc16);
            acc15 = _mm512_permutex2var_epi32(acc15, pf512_0, acc17);
            acc00 = _mm512_xor_si512(acc00, _mm512_slli_epi64(acc01, 16));
            acc04 = _mm512_xor_si512(acc04, _mm512_slli_epi64(acc05, 16));
            acc10 = _mm512_xor_si512(acc10, _mm512_slli_epi64(acc11, 16));
            acc14 = _mm512_xor_si512(acc14, _mm512_slli_epi64(acc15, 16));
            acc[i+0] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc00, pf512_1, acc04), _mm512_permutex2var_epi64(acc00, pf512_2, acc04));
            acc[i+1] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc10, pf512_1, acc14), _mm512_permutex2var_epi64(acc10, pf512_2, acc14));
        }
        acc[0] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[0], pf512_3, acc[1]), _mm512_permutex2var_epi64(acc[0], pf512_4, acc[1]));
        acc[2] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[2], pf512_3, acc[3]), _mm512_permutex2var_epi64(acc[2], pf512_4, acc[3]));
        acc[4] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[4], pf512_3, acc[5]), _mm512_permutex2var_epi64(acc[4], pf512_4, acc[5]));
        acc[6] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[6], pf512_3, acc[7]), _mm512_permutex2var_epi64(acc[6], pf512_4, acc[7]));
        acc[0] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[0], pf512_5, acc[2]), _mm512_permutex2var_epi64(acc[0], pf512_6, acc[2]));
        acc[4] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[4], pf512_5, acc[6]), _mm512_permutex2var_epi64(acc[4], pf512_6, acc[6]));
        uint64_t cmpplo = _mm512_cmp_epi16_mask(ub, acc[0], _MM_CMPINT_LE);
        uint64_t cmpphi = _mm512_cmp_epi16_mask(ub, acc[4], _MM_CMPINT_LE);
        uint64_t cmpnlo = _mm512_cmp_epi16_mask(acc[0], lb, _MM_CMPINT_LE);
        uint64_t cmpnhi = _mm512_cmp_epi16_mask(acc[4], lb, _MM_CMPINT_LE);
        *cmpp = (cmpplo | (cmpphi << 32));
        *cmpn = (cmpnlo | (cmpnhi << 32));
        return;
        #else
        // TODO
        #endif
    }
    do {
        __attribute__ ((aligned (32))) int16_t dst[64];
        for (long i = 0; i < 8; i++) {
            for (long l = 0; l < shsize; l += 8) {
                dst[i * 8 + 0] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 0 * shsize + l));
                dst[i * 8 + 1] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 1 * shsize + l));
                dst[i * 8 + 2] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 2 * shsize + l));
                dst[i * 8 + 3] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 3 * shsize + l));
                dst[i * 8 + 4] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 4 * shsize + l));
                dst[i * 8 + 5] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 5 * shsize + l));
                dst[i * 8 + 6] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 6 * shsize + l));
                dst[i * 8 + 7] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 7 * shsize + l));
            }
        }
        *cmpp = 0;
        *cmpn = 0;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 0), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 0;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 1), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 16;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 2), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 32;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 3), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 48;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 0), _MM_CMPINT_LE)) << 0;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 1), _MM_CMPINT_LE)) << 16;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 2), _MM_CMPINT_LE)) << 32;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 3), _MM_CMPINT_LE)) << 48;
    } while (0);
}
template <uint32_t nb>
template <uint32_t shsize>
inline void Pool_epi8_t<nb>::_vshdot_check8xn(uint64_t *cmpp, uint64_t *cmpn, uint8_t *u1, uint8_t *u2, long n, __m512i lb, __m512i ub) {
    if (shsize == 64) {
        static const __m512i pf512_0 = _mm512_set_epi32(0xfe, 0x0e, 0xfc, 0x0c, 0xfa, 0x0a, 0xf8, 0x08, 
                                                      0xf6, 0x06, 0xf4, 0x04, 0xf2, 0x02, 0xf0, 0x00);
        static const __m512i pf512_1 = _mm512_set_epi64(0x0b, 0x03, 0x0a, 0x02, 0x09, 0x01, 0x08, 0x00);
        static const __m512i pf512_2 = _mm512_set_epi64(0x0f, 0x07, 0x0e, 0x06, 0x0d, 0x05, 0x0c, 0x04);
        static const __m512i pf512_3 = _mm512_set_epi64(0x0b, 0x0a, 0x03, 0x02, 0x09, 0x08, 0x01, 0x00);
        static const __m512i pf512_4 = _mm512_set_epi64(0x0f, 0x0e, 0x07, 0x06, 0x0d, 0x0c, 0x05, 0x04);
        static const __m512i pf512_5 = _mm512_set_epi64(0x0b, 0x0a, 0x09, 0x08, 0x03, 0x02, 0x01, 0x00);
        static const __m512i pf512_6 = _mm512_set_epi64(0x0f, 0x0e, 0x0d, 0x0c, 0x07, 0x06, 0x05, 0x04);
        #ifdef __AVX512_VNNI__
        __m512i acc[8] = {};
        for (long i = 0; i < n; i++) {
            __m512i u10 = _mm512_load_si512((__m512i *)(u1 + i * shsize));
            __m512i u20 = _mm512_load_si512((__m512i *)(u2 + 0 * shsize));
            __m512i u21 = _mm512_load_si512((__m512i *)(u2 + 1 * shsize));
            __m512i u22 = _mm512_load_si512((__m512i *)(u2 + 2 * shsize));
            __m512i u23 = _mm512_load_si512((__m512i *)(u2 + 3 * shsize));
            __m512i u24 = _mm512_load_si512((__m512i *)(u2 + 4 * shsize));
            __m512i u25 = _mm512_load_si512((__m512i *)(u2 + 5 * shsize));
            __m512i u26 = _mm512_load_si512((__m512i *)(u2 + 6 * shsize));
            __m512i u27 = _mm512_load_si512((__m512i *)(u2 + 7 * shsize));
            __m512i acc00 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u20));
            __m512i acc01 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u21));
            __m512i acc02 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u22));
            __m512i acc03 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u23));
            __m512i acc04 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u24));
            __m512i acc05 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u25));
            __m512i acc06 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u26));
            __m512i acc07 = _mm512_popcnt_epi64(_mm512_xor_si512(u10, u27));
            acc00 = _mm512_permutex2var_epi32(acc00, pf512_0, acc02);
            acc01 = _mm512_permutex2var_epi32(acc01, pf512_0, acc03);
            acc04 = _mm512_permutex2var_epi32(acc04, pf512_0, acc06);
            acc05 = _mm512_permutex2var_epi32(acc05, pf512_0, acc07);
            acc00 = _mm512_xor_si512(acc00, _mm512_slli_epi64(acc01, 16));
            acc04 = _mm512_xor_si512(acc04, _mm512_slli_epi64(acc05, 16));
            acc[i+0] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc00, pf512_1, acc04), _mm512_permutex2var_epi64(acc00, pf512_2, acc04));
        }
        acc[0] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[0], pf512_3, acc[1]), _mm512_permutex2var_epi64(acc[0], pf512_4, acc[1]));
        acc[2] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[2], pf512_3, acc[3]), _mm512_permutex2var_epi64(acc[2], pf512_4, acc[3]));
        acc[4] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[4], pf512_3, acc[5]), _mm512_permutex2var_epi64(acc[4], pf512_4, acc[5]));
        acc[6] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[6], pf512_3, acc[7]), _mm512_permutex2var_epi64(acc[6], pf512_4, acc[7]));
        acc[0] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[0], pf512_5, acc[2]), _mm512_permutex2var_epi64(acc[0], pf512_6, acc[2]));
        acc[4] = _mm512_add_epi16(_mm512_permutex2var_epi64(acc[4], pf512_5, acc[6]), _mm512_permutex2var_epi64(acc[4], pf512_6, acc[6]));
        uint64_t cmpplo = _mm512_cmp_epi16_mask(ub, acc[0], _MM_CMPINT_LE);
        uint64_t cmpphi = _mm512_cmp_epi16_mask(ub, acc[4], _MM_CMPINT_LE);
        uint64_t cmpnlo = _mm512_cmp_epi16_mask(acc[0], lb, _MM_CMPINT_LE);
        uint64_t cmpnhi = _mm512_cmp_epi16_mask(acc[4], lb, _MM_CMPINT_LE);
        *cmpp = (cmpplo | (cmpphi << 32));
        *cmpn = (cmpnlo | (cmpnhi << 32));
        *cmpp &= (1ULL << (n * 8)) - 1;
        *cmpn &= (1ULL << (n * 8)) - 1;
        return;
        #else
        // TODO
        #endif
    }
    do {
        __attribute__ ((aligned (32))) int16_t dst[64];
        for (long i = 0; i < n; i++) {
            for (long l = 0; l < shsize; l += 8) {
                dst[i * 8 + 0] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 0 * shsize + l));
                dst[i * 8 + 1] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 1 * shsize + l));
                dst[i * 8 + 2] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 2 * shsize + l));
                dst[i * 8 + 3] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 3 * shsize + l));
                dst[i * 8 + 4] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 4 * shsize + l));
                dst[i * 8 + 5] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 5 * shsize + l));
                dst[i * 8 + 6] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 6 * shsize + l));
                dst[i * 8 + 7] += _mm_popcnt_u64(*(uint64_t *)(u1 + i * shsize + l) ^ *(uint64_t *)(u2 + 7 * shsize + l));
            }
        }
        *cmpp = 0;
        *cmpn = 0;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 0), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 0;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 1), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 16;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 2), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 32;
        *cmpn |= ((uint64_t)_mm256_cmp_epi16_mask(_mm256_load_si256((__m256i *)dst + 3), _mm512_castsi512_si256(lb), _MM_CMPINT_LE)) << 48;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 0), _MM_CMPINT_LE)) << 0;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 1), _MM_CMPINT_LE)) << 16;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 2), _MM_CMPINT_LE)) << 32;
        *cmpp |= ((uint64_t)_mm256_cmp_epi16_mask(_mm512_castsi512_si256(ub), _mm256_load_si256((__m256i *)dst + 3), _MM_CMPINT_LE)) << 48;
        *cmpp &= (1ULL << (n * 8)) - 1;
        *cmpn &= (1ULL << (n * 8)) - 1;
    } while (0);
}
template <uint32_t nb>
template <uint32_t shsize>
inline int32_t Pool_epi8_t<nb>::_vshdot(uint8_t *u1, uint8_t *u2) {
    if (shsize == 64) {
        #ifdef __AVX512_VNNI__
        __m512i acc512 = _mm512_popcnt_epi64(_mm512_xor_si512(_mm512_load_si512((__m512i *)u1), _mm512_load_si512((__m512i *)u2)));
        __m256i acc256 = _mm256_add_epi64(_mm512_castsi512_si256(acc512), _mm512_extracti32x8_epi32(acc512, 1));
        #else
        __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3,
                                      1, 2, 2, 3, 2, 3, 3, 4, 
                                      0, 1, 1, 2, 1, 2, 2, 3,
                                      1, 2, 2, 3, 2, 3, 3, 4);
        __m256i low_mask = _mm256_set1_epi8(0x0f);
        __m256i d1 = _mm256_xor_si256(_mm256_load_si256((__m256i *)u1), _mm256_load_si256((__m256i *)u2));
        __m256i d2 = _mm256_xor_si256(_mm256_load_si256((__m256i *)(u1 + 32)), _mm256_load_si256((__m256i *)(u2 + 32)));
        __m256i d1lo = _mm256_shuffle_epi8(lookup, _mm256_and_si256(d1, low_mask));         
        __m256i d2lo = _mm256_shuffle_epi8(lookup, _mm256_and_si256(d2, low_mask));
        __m256i d1hi = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi32(d1, 4), low_mask));
        __m256i d2hi = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi32(d2, 4), low_mask));
        __m256i acc1 = _mm256_add_epi8(d1lo, d1hi);
        __m256i acc2 = _mm256_add_epi8(d2lo, d2hi);
        __m256i acc256 = _mm256_sad_epu8(_mm256_add_epi8(acc1, acc2), _mm256_setzero_si256());
        #endif
        acc256 = _mm256_add_epi64(acc256, _mm256_permute4x64_epi64(acc256, 0x4e));
        acc256 = _mm256_add_epi32(acc256, _mm256_permute4x64_epi64(acc256, 0xb1));
        return *((int64_t *)(&acc256));
    } else {
        int32_t ret = 0;
        for (long l = 0; l < shsize; l += 8) {
            ret += _mm_popcnt_u64(*(uint64_t *)(u1 + l) ^ *(uint64_t *)(u2 + l));
        }
        return ret;
    }
}

template <uint32_t nb>
template <uint32_t shsize>
void Pool_epi8_t<nb>::_process_nshl1_triblock(uint8_t *sh, long bias, long bound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num) {
    const __m512i lb = _mm512_set1_epi16(threshold);
    const __m512i ub = _mm512_set1_epi16(shsize * 8 - threshold);

    long ind = 0;
    while (ind < bound - 7) {
        long jnd = ind;
        while (jnd < bound - 7) {
            uint64_t cmpp = 0, cmpn = 0;
            _vshdot_check8x8<shsize>(&cmpp, &cmpn, sh + ind * shsize, sh + jnd * shsize, lb, ub);
            if (jnd == ind) {
                cmpp &= 0x7f3f1f0f07030100ULL;
                cmpn &= 0x7f3f1f0f07030100ULL;
            }
            CHECK_AND_ADD8x8(cmpp, (bias + ind), (bias + jnd), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (bias + ind), (bias + jnd), ADDS_TO_BUFFER);
            jnd += 8;
        }
        if (jnd < bound) {
            const long jrem = bound - jnd;
            uint64_t cmpp = 0, cmpn = 0;
            _vshdot_check8xn<shsize>(&cmpp, &cmpn, sh + jnd * shsize, sh + ind * shsize, jrem, lb, ub);
            CHECK_AND_ADD8x8(cmpp, (bias + jnd), (bias + ind), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (bias + jnd), (bias + ind), ADDS_TO_BUFFER);            
        }
        ind += 8;
    }
    if (ind < bound) {
        const long irem = bound - ind;
        for (long i = 0; i < irem; i++) {
            for (long j = i + 1; j < irem; j++) {
                uint64_t shdot = _vshdot<shsize>(sh + (ind + i) * shsize, sh + (ind + j) * shsize);         
                if (shdot <= threshold) ADDS_TO_BUFFER((bias + ind + i), (bias + ind + j));
                if (shdot >= shsize * 8 - threshold) ADDA_TO_BUFFER((bias + ind + i), (bias + ind + j));
            }
        }
    }
}

template <uint32_t nb>
template <uint32_t shsize>
void Pool_epi8_t<nb>::_process_nshl1_block(uint8_t *shi, uint8_t *shj, long ibias, long jbias, long ibound, long jbound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num) {
    const __m512i lb = _mm512_set1_epi16(threshold);
    const __m512i ub = _mm512_set1_epi16(shsize * 8 - threshold);

    long ind = 0;
    while (ind < ibound - 7) {
        long jnd = 0;
        while (jnd < jbound - 7) {
            uint64_t cmpp = 0, cmpn = 0;
            _vshdot_check8x8<shsize>(&cmpp, &cmpn, shi + ind * shsize, shj + jnd * shsize, lb, ub);
            CHECK_AND_ADD8x8(cmpp, (ibias + ind), (jbias + jnd), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (ibias + ind), (jbias + jnd), ADDS_TO_BUFFER);
            jnd += 8;
        }
        if (jnd < jbound) {
            const long jrem = jbound - jnd;
            uint64_t cmpp = 0, cmpn = 0;
            _vshdot_check8xn<shsize>(&cmpp, &cmpn, shj + jnd * shsize, shi + ind * shsize, jrem, lb, ub);
            CHECK_AND_ADD8x8(cmpp, (jbias + jnd), (ibias + ind), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (jbias + jnd), (ibias + ind), ADDS_TO_BUFFER);
        }
        ind += 8;
    }
    if (ind < ibound) {
        const long irem = ibound - ind;
        long jnd = 0;
        while (jnd < jbound) {
            uint64_t cmpp = 0, cmpn = 0;
            _vshdot_check8xn<shsize>(&cmpp, &cmpn, shj + jnd * shsize, shi + ind * shsize, irem, lb, ub);
            CHECK_AND_ADD8x8(cmpp, (jbias + jnd), (ibias + ind), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (jbias + jnd), (ibias + ind), ADDS_TO_BUFFER);
            jnd += 8;
        }
    }
}
#pragma endregion


// the implementation is in naivedh_epi8.cpp
int gen_dual_vec_list(float *dst, Lattice_QP *L, long log_level, long nlist, int rng_seed = -1);

static long lsh_param_seed_offset()
{
    const char *env = getenv("BGJ_LSH_PARAM_SEED_OFFSET");
    return (env && env[0]) ? atol(env) : 0;
}

template <uint32_t nb>
void Pool_epi8_t<nb>::_opt_nsh_threshold(float *dual_vec, uint32_t *compress_pos, int32_t &num_hbits, int32_t &num_tbits, int32_t &threshold,
                        Lattice_QP *b_mid, uint32_t shsize, double exp_length, double *tail_alpha_prob_list, long log_level, long target_index) {
    // number of samples
    const long N = (CSD >= 120) ? 1048576 : (1L << (CSD/5-4));
    const long dh_dim = b_mid->NumRows();
    const long tail_dim = CSD;
    const double tail_gh = sqrt(gh2);
    const double tail_exp_ratio = sqrt(cvec[3*(num_vec/2)+2] * 4.0) / _ratio / sqrt(gh2);
    // CHANGE_WITH_ARCH
    const double sh_cost = 0.4 / 1073741824.0;
    const double lift_cost = 100.0 / 1073741824.0;

    const int param_seed = (int)(0x5f3759dfu
                                 + (uint32_t)(CSD * 131)
                                 + (uint32_t)(index_l * 1009)
                                 + (uint32_t)(index_r * 9176)
                                 + (uint32_t)(target_index * 65537)
                                 + (uint32_t)(shsize * 8191)
                                 + (uint32_t)lsh_param_seed_offset());
    DGS1d param_R(param_seed);
    auto param_uniform_long = [&](long bound) -> long {
        if (bound <= 0) return 0;
        return (long)(param_R.Uniform_u64() % (uint64_t)bound);
    };

    // generate compress pos
    for (long i = 0; i < shsize * 8; i++) {
        if (i < tail_dim) {
            compress_pos[6 * i + 0] = i;
            compress_pos[6 * i + 1] = i;
            compress_pos[6 * i + 2] = i;
            compress_pos[6 * i + 3] = i;
            compress_pos[6 * i + 4] = i;
            compress_pos[6 * i + 5] = i;
        } else {
            compress_pos[6 * i + 0] = param_uniform_long(tail_dim);
            compress_pos[6 * i + 1] = param_uniform_long(tail_dim);
            compress_pos[6 * i + 2] = param_uniform_long(tail_dim);
            compress_pos[6 * i + 3] = param_uniform_long(tail_dim);
            compress_pos[6 * i + 4] = param_uniform_long(tail_dim);
            compress_pos[6 * i + 5] = param_uniform_long(tail_dim);
        }
    }
    // generate dual vector
    gen_dual_vec_list(dual_vec, b_mid, log_level, shsize * 8, param_seed + 1);
    float **b_local = (float **) NEW_MAT(dh_dim, _CEIL8(dh_dim), sizeof(float));
    float *b_local_idiag = (float *) NEW_VEC(dh_dim, sizeof(float));
    for (long i = 0; i < dh_dim; i++) {
        for (long j = 0; j < _CEIL8(dh_dim); j++) {
            b_local[i][j] = b_mid->get_b().hi[i][j];
        }
        b_local_idiag[i] = 1.0f / b_local[i][i];
    }

    uint64_t *nh_list = (uint64_t *) NEW_VEC(N + 1, NSH_DEFAULT_SHSIZE);
    uint64_t *nt_list = (uint64_t *) NEW_VEC(N + 1, NSH_DEFAULT_SHSIZE);
    uint64_t *nh_slist = (uint64_t *) NEW_VEC(N, NSH_DEFAULT_SHSIZE);
    uint64_t *nt_slist = (uint64_t *) NEW_VEC(N, NSH_DEFAULT_SHSIZE);
    // gen rsamples
    do {
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++) {
            __attribute__ ((aligned (32))) float ftmp[_CEIL8(NSH_DEFAULT_DHDIM)] = {};
            __attribute__ ((aligned (32))) float ttmp[256] = {};
            DGS1d R(param_seed + 4096 + thread);
            const long begin_ind = (thread * (N+1)) / num_threads;
            const long end_ind = ((thread + 1) * (N+1)) / num_threads;
            for (long ind = end_ind - 1; ind >= begin_ind; ind--) {
                // h part
                for (long i = 0; i < dh_dim; i++) ftmp[i] = R.Uniform_u64() & 0xffff;
                for (long i = dh_dim - 1; i >= 0; i--) {
                    red_avx2(ftmp, b_local[i], roundf(b_local_idiag[i] * ftmp[i]), dh_dim);
                }
                for (long i = 0; i < shsize * 8; i++) {
                    float x = dot_avx2(dual_vec + i * _CEIL8(dh_dim), ftmp, dh_dim);
                    int32_t xi = x;
                    int8_t xi8 = xi;
                    if (xi8 >= 0) nh_list[ind * NSH_DEFAULT_SHSIZE / 8 + i/64] |= (1ULL << (i % 64));
                }
                // t part
                for (long i = 0; i < tail_dim; i++) ttmp[i] = R.discrete_gaussian(0.0, 1048576.0);
                for (long i = 0; i < shsize * 8; i++) {
                    float x = (i < tail_dim) ? ttmp[compress_pos[6*i+0]] :
                                ttmp[compress_pos[6*i+0]] + ttmp[compress_pos[6*i+1]] + ttmp[compress_pos[6*i+2]] - 
                                ttmp[compress_pos[6*i+3]] - ttmp[compress_pos[6*i+4]] - ttmp[compress_pos[6*i+5]];
                    if (x > 0) nt_list[ind * NSH_DEFAULT_SHSIZE / 8 + i/64] |= (1ULL << (i % 64));
                }
            }
        }
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++) {
            const long begin_ind = (thread * N) / num_threads;
            const long end_ind = ((thread + 1) * N) / num_threads;
            for (long ind = end_ind - 1; ind >= begin_ind; ind--) {
                for (long j = 0; j < shsize * 8; j += 64) {
                    nh_list[ind * NSH_DEFAULT_SHSIZE / 8 + j/64] ^= nh_list[N * NSH_DEFAULT_SHSIZE / 8 + j/64];
                    nt_list[ind * NSH_DEFAULT_SHSIZE / 8 + j/64] ^= nt_list[N * NSH_DEFAULT_SHSIZE / 8 + j/64];
                }
            }
        }
    } while (0);
    #if 0
    for (long i = 0; i < 256; i++) {
        for (long j = 0; j < 8; j++) printf("%lx ", nh_list[i * 8 + j]);
        printf("\n");
    }
    printf("\n");
    for (long i = 0; i < 256; i++) {
        for (long j = 0; j < 8; j++) printf("%lx ", nt_list[i * 8 + j]);
        printf("\n");
    }
    #endif
    
    // gen ssamples
    do {
        int *ai_list = (int *) NEW_VEC(N, sizeof(int));
        for (int ai = 0; ai < 50; ai++) {
            for (long ind = (ai == 49) ? 0 : (long)(N * tail_alpha_prob_list[ai+1]); ind < (long)(N * tail_alpha_prob_list[ai]); ind++) {
                ai_list[ind] = ai;
            }
        }
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++) {
            const long begin_ind = (thread * N) / num_threads;
            const long end_ind = ((thread + 1) * N) / num_threads;
            __attribute__ ((aligned (32))) float ftmp1[_CEIL8(NSH_DEFAULT_DHDIM)] = {};
            __attribute__ ((aligned (32))) float ftmp2[_CEIL8(NSH_DEFAULT_DHDIM)] = {};
            __attribute__ ((aligned (32))) float ttmp1[256] = {};
            __attribute__ ((aligned (32))) float ttmp2[256] = {};
            DGS1d R(param_seed + 8192 + thread);

            for (long ind = end_ind - 1; ind >= begin_ind; ind--) {
                float alpha = 0.01 * ai_list[ind];
                float _tail_one_len = tail_exp_ratio * tail_gh;
                float _tail_sum_len = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * alpha);
                float _head_sum_len = sqrt(exp_length * exp_length - _tail_sum_len * _tail_sum_len);

                // hpart
                for (long i = 0; i < dh_dim; i++) ftmp1[i] = R.Uniform_u64() & 0xffff;
                for (long i = dh_dim - 1; i >= 0; i--) {
                    red_avx2(ftmp1, b_local[i], roundf(b_local_idiag[i] * ftmp1[i]), dh_dim);
                }
                for (long i = 0; i < dh_dim; i++) ftmp2[i] = R.discrete_gaussian(0.0, 1048576.0);
                mul_avx2(ftmp2, _head_sum_len/sqrt(dot_avx2(ftmp2, ftmp2, dh_dim)), dh_dim);
                add_avx2(ftmp2, ftmp1, ftmp2, dh_dim);
                for (long i = 0; i < shsize * 8; i++) {
                    float x1 = dot_avx2(dual_vec + i * _CEIL8(dh_dim), ftmp1, dh_dim);
                    float x2 = dot_avx2(dual_vec + i * _CEIL8(dh_dim), ftmp2, dh_dim);
                    int32_t x1i = x1;
                    int32_t x2i = x2;
                    int8_t x1i8 = x1i;
                    int8_t x2i8 = x2i;
                    if ((x1i8 >= 0 && x2i8 < 0) || (x1i8 < 0 && x2i8 >= 0)) nh_slist[ind * NSH_DEFAULT_SHSIZE/8 + i/64] |= (1ULL << (i % 64));
                }
                // tpart
                for (long i = 0; i < tail_dim; i++) ttmp1[i] = R.discrete_gaussian(0.0, 1048576.0);
                for (long i = 0; i < tail_dim; i++) ttmp2[i] = R.discrete_gaussian(0.0, 1048576.0);
                mul_avx2(ttmp1, _tail_one_len/sqrt(dot_avx2(ttmp1, ttmp1, tail_dim)), tail_dim);
                float lambda = dot_avx2(ttmp1, ttmp2, tail_dim) / dot_avx2(ttmp1, ttmp1, tail_dim);
                red_avx2(ttmp2, ttmp1, lambda, tail_dim);
                mul_avx2(ttmp2, sqrt(1-alpha * alpha) * _tail_one_len / sqrt(dot_avx2(ttmp2, ttmp2, tail_dim)), tail_dim);
                red_avx2(ttmp2, ttmp1, -alpha, tail_dim);
                for (long i = 0; i < shsize * 8; i++) {
                    float x1 = (i < tail_dim) ? ttmp1[compress_pos[6*i+0]] : 
                            ttmp1[compress_pos[6*i+0]] + ttmp1[compress_pos[6*i+1]] + ttmp1[compress_pos[6*i+2]] - 
                            ttmp1[compress_pos[6*i+3]] - ttmp1[compress_pos[6*i+4]] - ttmp1[compress_pos[6*i+5]];
                    float x2 = (i < tail_dim) ? ttmp2[compress_pos[6*i+0]] : 
                            ttmp2[compress_pos[6*i+0]] + ttmp2[compress_pos[6*i+1]] + ttmp2[compress_pos[6*i+2]] - 
                            ttmp2[compress_pos[6*i+3]] - ttmp2[compress_pos[6*i+4]] - ttmp2[compress_pos[6*i+5]];
                    if ((x1 > 0 && x2 < 0) || (x1 < 0 && x2 > 0)) nt_slist[ind * NSH_DEFAULT_SHSIZE / 8 + i/64] |= (1ULL << (i % 64));
                }
            }
        }
        FREE_VEC((void *)ai_list);
    } while (0);
    #if 0
    for (long i = 0; i < 256; i++) {
        for (long j = 0; j < 8; j++) printf("%lx ", nh_slist[i * 8 + j]);
        printf("\n");
    }
    printf("\n");
    for (long i = 0; i < 256; i++) {
        for (long j = 0; j < 8; j++) printf("%lx ", nt_slist[i * 8 + j]);
        printf("\n");
    }
    #endif
    
    uint64_t **r_stat = (uint64_t **) NEW_MAT(shsize + 1, shsize * 8 + 1, sizeof(uint64_t));
    uint64_t **s_stat = (uint64_t **) NEW_MAT(shsize + 1, shsize * 8 + 1, sizeof(uint64_t));
    for (long ind = 0; ind < N; ind++) {
        uint8_t *rhptr = (uint8_t *)(nh_list + ind * NSH_DEFAULT_SHSIZE / 8);
        uint8_t *rtptr = (uint8_t *)(nt_list + ind * NSH_DEFAULT_SHSIZE / 8);
        uint8_t *shptr = (uint8_t *)(nh_slist + ind * NSH_DEFAULT_SHSIZE / 8);
        uint8_t *stptr = (uint8_t *)(nt_slist + ind * NSH_DEFAULT_SHSIZE / 8);

        uint32_t rpop = 0;
        uint32_t spop = 0;
        for (long i = 0; i < shsize; i += 8) {
            rpop += _mm_popcnt_u64(*(uint64_t *)(rtptr + i));
            spop += _mm_popcnt_u64(*(uint64_t *)(stptr + i));
        }
        for (long k = 0; k <= shsize; k++) {
            r_stat[k][rpop]++;
            s_stat[k][spop]++;
            if (k == shsize) continue;
            rpop += _mm_popcnt_u64((uint64_t)(rhptr[k])) - _mm_popcnt_u64((uint64_t)(rtptr[k]));
            spop += _mm_popcnt_u64((uint64_t)(shptr[k])) - _mm_popcnt_u64((uint64_t)(stptr[k]));
        }
    }
    for (long i = 0; i <= shsize; i++) {
        for (long j = 1; j <= shsize * 8; j++) {
            r_stat[i][j] += r_stat[i][j-1];
            s_stat[i][j] += s_stat[i][j-1];
        }
    }
    #if 0
    for (long ind = 0; ind <= 512; ind += 8) {
        printf("mix512: h = %ld, t = %ld", ind, 512-ind);
        long j = 0;
        for (long i = 1; i <= 512; i++) {
            if (j == 0) printf("\n");
            else printf("\t");
            printf("r: %.6f/s: %.3f(%ld)", (r_stat[ind/8][i]+0.0) / N, (s_stat[ind/8][i]+0.0) / N, i);
            if (i == 512) printf("\n");
            j = (j + 1) & 0x7;
        }
    }
    #endif
    
    double max_speed = 0.0;
    for (long i = 0; i <= shsize; i++) {
        for (long j = 0; j <= shsize * 8; j++) {
            double speed = s_stat[i][j] / (sh_cost * N + lift_cost * (double)r_stat[i][j]);
            if (speed > max_speed && (lift_cost * (double)r_stat[i][j] < 0.6 * sh_cost * N)) {
                max_speed = speed;
                num_hbits = i * 8;
                num_tbits = shsize * 8 - i * 8;
                threshold = j;
            }
        }
    }

    #if 1
    double pmax_speed = 0.0;
    for (long i = 0; i <= shsize; i++) {
        for (long j = 0; j <= shsize * 8; j++) {
            double speed = s_stat[i][j] / (sh_cost * N + lift_cost * (double)r_stat[i][j]);
            if (speed > max_speed) {
                pmax_speed = speed;
            }
        }
    }
    if (pmax_speed > max_speed) {
        //fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_opt_nsh_threshold: allowing lifting time > 0.6 * searching time lead to faster speed?\n", nb);
    }
    #endif
    
    FREE_MAT(b_local);
    FREE_VEC(b_local_idiag);
    FREE_VEC((void *)nh_list);
    FREE_VEC((void *)nt_list);
    FREE_VEC((void *)nh_slist);
    FREE_VEC((void *)nt_slist);
    FREE_MAT((void **)r_stat);
    FREE_MAT((void **)s_stat);
}

struct nsh_profile_data_t {
    long log_level;
    uint64_t mblock_ops, dp_ops, lift_ops;
    double mblock_time, dp_time, lift_time;
    long num_mblock, total_mblock;
    pthread_spinlock_t profile_lock;

    void init(long _log_level) {
        log_level = _log_level;
        num_mblock = 0;
        mblock_ops = 0;
        dp_ops = 0;
        lift_ops = 0;
        mblock_time = 0.0;
        dp_time = 0.0;
        lift_time = 0.0;
        pthread_spin_init(&profile_lock, PTHREAD_PROCESS_SHARED);
    }

    void mblock_log(float *min_norm, long ID) {
        if (log_level >= 1) {
            fprintf(stdout, "mblock %lu / %lu, mblock_time = %fs(%.2f M), dp_time = %fs(%.2f G), lift_time = %fs(%.2f M)\n", num_mblock, total_mblock, 
                            mblock_time, mblock_ops/1048576.0, dp_time, dp_ops/1073741824.0, lift_time, lift_ops/1048576.0);
            fprintf(stdout, "dp_speed = %f Gops, lift_speed = %f Mops, mblock_speed = %f Mops\n",
                            dp_ops / dp_time / 1073741824.0, lift_ops / lift_time / 1048576.0, 
                            mblock_ops / mblock_time / 1048576.0);
            if (log_level >= 2) {
                fprintf(stdout, "min_norm = [");
                for (long i = 0; i < ID; i++) {
                    fprintf(stdout, "%f", sqrt(min_norm[i]));
                    if (i < ID - 1) fprintf(stdout, " "); 
                }
                fprintf(stdout, "]\n");
            }
        }
    }
    void final_log(float *min_norm, long ID) {
        if (log_level >= 0) {
            fprintf(stdout, "search done, mblock_time = %fs(%.2f M), dp_time = %fs(%.2f G), lift_time = %fs(%.2f M)\n", 
                            mblock_time, mblock_ops/1048576.0, dp_time, dp_ops/1073741824.0, lift_time, lift_ops/1048576.0);
            fprintf(stdout, "dp_speed = %f Gops, lift_speed = %f Mops, mblock_speed = %f Mops\n",
                            dp_ops / dp_time / 1073741824.0, lift_ops / lift_time / 1048576.0, 
                            mblock_ops / mblock_time / 1048576.0);
            if (log_level >= 1) {
                fprintf(stdout, "min_norm = [");
                for (long i = 0; i < ID; i++) {
                    fprintf(stdout, "%f", sqrt(min_norm[i]));
                    if (i < ID - 1) fprintf(stdout, " "); 
                }
                fprintf(stdout, "]\n");
            }
        }
    }
};

struct nsh_mblock_t {
    float *fveci = NULL, *fvecj = NULL;
    uint8_t *shi = NULL, *shj = NULL;
    long MIbound, MJbound;
};

template <uint32_t nb, uint32_t shsize, uint32_t dh_dim, uint32_t m_block>
struct nsh_mblock_provider_t {
    Pool_epi8_t<nb> *p;
    long target_index;
    float **b_ext_fp = NULL;

    int32_t num_hbits, num_tbits;
    uint32_t *compress_pos;
    float *dual_vec;

    long *MI_list = NULL, *MJ_list = NULL;
    float *_fvec0 = NULL, *_fvec1 = NULL;
    uint8_t *_sh0 = NULL, *_sh1 = NULL;
    long _m0, _m1;
    long *lifted = NULL;
    long num_poped = 0;
    long num_total_blocks;

    void init(Pool_epi8_t<nb> *pool, long _target_index, float *_dual_vec, uint32_t *_compress_pos, int32_t _num_hbits, float **&b_full_fp, nsh_profile_data_t *profile_data) {
        p = pool;
        target_index = _target_index;
        num_hbits = _num_hbits;
        num_tbits = shsize * 8 - num_hbits;
        dual_vec = _dual_vec;
        compress_pos = _compress_pos;
        // creating MI/J_list
        do {
            const long FD8 = _CEIL8(p->index_r - target_index);
            long num_total_parts = (p->num_vec + m_block - 1) / m_block;
            num_total_blocks = num_total_parts * (num_total_parts + 1) / 2;
            profile_data->total_mblock = num_total_blocks;
            MI_list = (long *) malloc(num_total_blocks * sizeof(long));
            MJ_list = (long *) malloc(num_total_blocks * sizeof(long));
            lifted = (long *) calloc(num_total_parts, sizeof(long));
            _fvec0 = (float *) NEW_VEC(m_block * FD8, sizeof(float));
            _fvec1 = (float *) NEW_VEC(m_block * FD8, sizeof(float));
            _sh0 = (uint8_t *) NEW_VEC(shsize * _CEIL8(m_block), sizeof(uint8_t));
            _sh1 = (uint8_t *) NEW_VEC(shsize * _CEIL8(m_block), sizeof(uint8_t));
            _m0 = -1;
            _m1 = -1;
            long curr_num = 0;
            #if NSH_DIAG_FIRST
            for (long DIFF = 0; DIFF < p->num_vec; DIFF += m_block) {
                for (long MInd = 0; MInd < p->num_vec - DIFF; MInd += m_block) {
                    long MJnd = MInd + DIFF;
            #else
            for (long MInd = 0; MInd < p->num_vec; MInd += m_block) {
                for (long MJnd = MInd; MJnd < p->num_vec; MJnd += m_block) {
            #endif
                    MI_list[curr_num] = MInd;
                    MJ_list[curr_num] = MJnd;
                    curr_num++;
            
                }
            }
        } while (0);
        // compute b_ext_fp and b_full_fp
        do {
            Lattice_QP *b_full = p->basis->b_loc_QP(target_index, p->index_r);
            b_full_fp = (float **) NEW_MAT(p->index_r - target_index, p->index_r - target_index, sizeof(float));
            for (long i = 0; i < p->index_r - target_index; i++) {
                for (long j = 0; j <= i; j++) b_full_fp[i][j] = b_full->get_b().hi[i][j];
            }
            delete b_full;
            b_ext_fp = (float **) NEW_MAT(p->CSD, p->index_r - target_index, sizeof(float));
            __attribute__ ((aligned (32))) float extmp[nb * 32] = {};
            for (long i = 0; i < p->CSD; i++) {
                copy_avx2(extmp, p->_b_local[i], p->CSD);
                mul_avx2(extmp, 1.0/p->_ratio, p->CSD);
                for (long j = p->CSD - 1; j >= 0; j--) {
                    float q = round(extmp[j] / b_full_fp[p->index_l - target_index + j][p->index_l - target_index + j]);
                    for (long l = 0; l < p->CSD; l++) extmp[l] -= q * b_full_fp[p->index_l - target_index+j][p->index_l - target_index+l];
                    red_avx2(b_ext_fp[i], b_full_fp[p->index_l - target_index+j], -q, p->index_r - target_index);
                }
            }
        } while(0);
    }
    void clear() {
        if (b_ext_fp) FREE_MAT(b_ext_fp);
        if (MI_list) free(MI_list);
        if (MJ_list) free(MJ_list);
        if (lifted) free(lifted);
        if (_fvec0) FREE_VEC(_fvec0);
        if (_fvec1) FREE_VEC(_fvec1);
        if (_sh0) FREE_VEC(_sh0);
        if (_sh1) FREE_VEC(_sh1);
        b_ext_fp = NULL;
        MI_list = NULL;
        MJ_list = NULL;
        lifted = NULL;
        _fvec0 = NULL;
        _fvec1 = NULL;
        _sh0 = NULL;
        _sh1 = NULL;
    }
    int pop(nsh_mblock_t *mb, float *min_norm, float **min_vec, float **b_full_fp, nsh_profile_data_t *profile_data) {
        if (num_poped >= num_total_blocks) return 0;
        
        const long MInd = MI_list[num_poped];
        const long MJnd = MJ_list[num_poped];
        num_poped++;
        const long MIbound = (MInd + m_block > p->num_vec) ? (p->num_vec - MInd) : m_block;
        const long MJbound = (MJnd + m_block > p->num_vec) ? (p->num_vec - MJnd) : m_block;
        
        nsh_mblock_t ret;
        ret.MIbound = MIbound;
        ret.MJbound = MJbound;
        TIMER_START;
        do {
            if (MInd == _m0) {
                ret.fveci = _fvec0;
                ret.shi = _sh0;
            } 
            if (MInd == _m1) {
                ret.fveci = _fvec1;
                ret.shi = _sh1;
            }
            if (MJnd == _m0) {
                ret.fvecj = _fvec0;
                ret.shj = _sh0;
            }
            if (MJnd == _m1) {
                ret.fvecj = _fvec1;
                ret.shj = _sh1;
            }
            if (ret.fveci == NULL) {
                if (_fvec0 == ret.fvecj) {
                    ret.fveci = _fvec1;
                    ret.shi = _sh1;
                    _m1 = MInd;
                } else {
                    ret.fveci = _fvec0;
                    ret.shi = _sh0;
                    _m0 = MInd;
                }
                if (lifted[MInd/m_block]) {
                    _compute_nsh_mblock(ret.fveci, ret.shi, MInd, MIbound);
                } else {
                    lifted[MInd/m_block] = 1;
                    _compute_nsh_mblock(ret.fveci, ret.shi, MInd, MIbound, b_full_fp, min_norm, min_vec);
                }
                
                profile_data->mblock_ops += MIbound;
            }
            if (ret.fvecj == NULL) {
                if (MInd == MJnd) {
                    ret.fvecj = ret.fveci;
                    ret.shj = ret.shi;
                } else {
                    if (ret.fveci == _fvec1) {
                        ret.fvecj = _fvec0;
                        ret.shj = _sh0;
                        _m0 = MJnd;
                    } else {
                        ret.fvecj = _fvec1;
                        ret.shj = _sh1;
                        _m1 = MJnd;
                    }
                    if (lifted[MJnd/m_block]) {
                        _compute_nsh_mblock(ret.fvecj, ret.shj, MJnd, MJbound);
                    } else {
                        lifted[MJnd/m_block] = 1;
                        _compute_nsh_mblock(ret.fvecj, ret.shj, MJnd, MJbound, b_full_fp, min_norm, min_vec);
                    }
                    profile_data->mblock_ops += MJbound;
                }
            }
        } while (0);
        *mb = ret;
        TIMER_END;
        profile_data->mblock_time += CURRENT_TIME;

        return 1;
    }
    void _compute_nsh_mblock(float *fvec, uint8_t *sh, long MInd, long MIbound, float **b_full_fp = NULL, float *min_norm = NULL, float **min_vec = NULL) {
        #pragma region
        /** 
         * \brief compute the dot product of __c0, __c1 and 8 vectors __ptr, ..., __ptr + __len * 7, 
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
                __r00 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 0 * __len) + __i), __r00);            \
                __r01 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 1 * __len) + __i), __r01);            \
                __r02 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 2 * __len) + __i), __r02);            \
                __r03 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 3 * __len) + __i), __r03);            \
                __r04 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 4 * __len) + __i), __r04);            \
                __r05 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 5 * __len) + __i), __r05);            \
                __r06 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 6 * __len) + __i), __r06);            \
                __r07 = _mm256_fmadd_ps(__x0, _mm256_load_ps((__ptr + 7 * __len) + __i), __r07);            \
                                                                                                            \
                __r10 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 0 * __len) + __i), __r10);            \
                __r11 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 1 * __len) + __i), __r11);            \
                __r12 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 2 * __len) + __i), __r12);            \
                __r13 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 3 * __len) + __i), __r13);            \
                __r14 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 4 * __len) + __i), __r14);            \
                __r15 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 5 * __len) + __i), __r15);            \
                __r16 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 6 * __len) + __i), __r16);            \
                __r17 = _mm256_fmadd_ps(__x1, _mm256_load_ps((__ptr + 7 * __len) + __i), __r17);            \
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
         * \brief compute the dot product of __c0 and __ptr + __len * i, store the result in a __m256 register, __dst.
        */
        #define AVX2_DP_1X8(__c0, __ptr, __len, __dst)                                                      \
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
                __r0 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 0 + __i), __r0);                \
                __r1 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 1 + __i), __r1);                \
                __r2 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 2 + __i), __r2);                \
                __r3 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 3 + __i), __r3);                \
                __r4 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 4 + __i), __r4);                \
                __r5 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 5 + __i), __r5);                \
                __r6 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 6 + __i), __r6);                \
                __r7 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 7 + __i), __r7);                \
                                                                                                            \
                __r0 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 0 + __i + 8), __r0);            \
                __r1 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 1 + __i + 8), __r1);            \
                __r2 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 2 + __i + 8), __r2);            \
                __r3 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 3 + __i + 8), __r3);            \
                __r4 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 4 + __i + 8), __r4);            \
                __r5 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 5 + __i + 8), __r5);            \
                __r6 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 6 + __i + 8), __r6);            \
                __r7 = _mm256_fmadd_ps(__x1, _mm256_load_ps(__ptr + __len * 7 + __i + 8), __r7);            \
                __i += 16;                                                                                  \
            }                                                                                               \
            if (__i < __len - 7) {                                                                          \
                __m256 __x0 = _mm256_load_ps(__c0 + __i + 0);                                               \
                __r0 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 0 + __i), __r0);                \
                __r1 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 1 + __i), __r1);                \
                __r2 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 2 + __i), __r2);                \
                __r3 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 3 + __i), __r3);                \
                __r4 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 4 + __i), __r4);                \
                __r5 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 5 + __i), __r5);                \
                __r6 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 6 + __i), __r6);                \
                __r7 = _mm256_fmadd_ps(__x0, _mm256_load_ps(__ptr + __len * 7 + __i), __r7);                \
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
        #pragma endregion
        if (shsize != NSH_DEFAULT_SHSIZE) {
            printf("[Error] nsh_mblock_provider_t::_compute_nsh_mblock: shsize(%u) != NSH_DEFAULT_SHSIZE(%u) not supported, nothing done.\n", shsize, NSH_DEFAULT_SHSIZE);
            return;
        }
        const long FD = p->index_r - target_index;
        const long LD = p->index_l - target_index;
        const long FD8 = _CEIL8(FD);
        const long ID = p->index_l - dh_dim - target_index;
        pthread_spinlock_t min_lock;
        pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);
        
        #pragma omp parallel for
        for (long thread = 0; thread < p->num_threads; thread++) {
            const long begin_ind = (MIbound * thread) / p->num_threads;
            const long end_ind = (MIbound * (thread+1)) / p->num_threads;
            long ind = begin_ind;
            __attribute__ ((aligned (32))) int32_t ctmp[8 * p->vec_length];
            __attribute__ ((aligned (32))) float ftmp[8 * _CEIL8(dh_dim)];
            float idiag[256];
            if (b_full_fp) for (long i = 0; i < FD; i++) idiag[i] = 1.0 / b_full_fp[i][i];
            while (ind < end_ind - 7) {
                p->compute_coeff_b8(ctmp, MInd + ind);
                set_zero_avx2(fvec + FD8 * ind, FD8 * 8);
                for (long i = 0; i < p->CSD; i++) {
                    red_avx2(fvec + FD8 * (ind+0), b_ext_fp[i], -ctmp[i*8+0], LD + i + 1);
                    red_avx2(fvec + FD8 * (ind+1), b_ext_fp[i], -ctmp[i*8+1], LD + i + 1);
                    red_avx2(fvec + FD8 * (ind+2), b_ext_fp[i], -ctmp[i*8+2], LD + i + 1);
                    red_avx2(fvec + FD8 * (ind+3), b_ext_fp[i], -ctmp[i*8+3], LD + i + 1);
                    red_avx2(fvec + FD8 * (ind+4), b_ext_fp[i], -ctmp[i*8+4], LD + i + 1);
                    red_avx2(fvec + FD8 * (ind+5), b_ext_fp[i], -ctmp[i*8+5], LD + i + 1);
                    red_avx2(fvec + FD8 * (ind+6), b_ext_fp[i], -ctmp[i*8+6], LD + i + 1);
                    red_avx2(fvec + FD8 * (ind+7), b_ext_fp[i], -ctmp[i*8+7], LD + i + 1);
                }
                if (b_full_fp) {
                    for (long i = LD - 1; i >= 0; i--) {
                        red_avx2(fvec + FD8 * (ind+0), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+0))[i]), i+1);
                        red_avx2(fvec + FD8 * (ind+1), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+1))[i]), i+1);
                        red_avx2(fvec + FD8 * (ind+2), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+2))[i]), i+1);
                        red_avx2(fvec + FD8 * (ind+3), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+3))[i]), i+1);
                        red_avx2(fvec + FD8 * (ind+4), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+4))[i]), i+1);
                        red_avx2(fvec + FD8 * (ind+5), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+5))[i]), i+1);
                        red_avx2(fvec + FD8 * (ind+6), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+6))[i]), i+1);
                        red_avx2(fvec + FD8 * (ind+7), b_full_fp[i], round(idiag[i] * (fvec + FD8 * (ind+7))[i]), i+1);
                    }
                    float current_norm[8];
                    for (long j = 0; j < 8; j++) current_norm[j] = dot_avx2(fvec + FD8 * (ind+j), fvec + FD8 * (ind+j), FD8);
                    for (long i = 0; i < ID; i++) {
                        for (long j = 0; j < 8; j++) {
                            if (current_norm[j] < min_norm[i]) {
                                pthread_spin_lock(&min_lock);
                                if (current_norm[j] < min_norm[i]) {
                                    int has_one = 0;
                                    __attribute__ ((aligned (32))) float ftmpp[256];
                                    copy_avx2(ftmpp, fvec + FD8 * (ind+j), FD);
                                    for (long l = FD - 1; l >= FD - p->CSD; l--) {
                                        float qq = round(ftmpp[l] * idiag[l]);
                                        if (fabs(qq)  == 1.0f) {
                                            has_one = 1;
                                            break;
                                        }
                                        red_avx2(ftmpp, b_full_fp[l], qq, l+1);
                                    }
                                    if (has_one) {
                                        min_norm[i] = current_norm[j];
                                        copy_avx2(min_vec[i], fvec + FD8 * (ind+j), FD);
                                    }
                                }
                                pthread_spin_unlock(&min_lock);
                            }
                            current_norm[j] -= (fvec + FD8 * (ind+j))[i] * (fvec + FD8 * (ind+j))[i];
                        }
                    }
                }
                for (long l = 0; l < dh_dim; l += 8) {
                    for (long i = 0; i < 8; i++) {
                        _mm256_store_ps(ftmp + i * _CEIL8(dh_dim) + l, _mm256_loadu_ps(fvec + (ind+i) * FD8 + ID + l));
                    }
                }
                // compute h part
                for (long k = 0; k < num_hbits; k += 8) {
                    __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                    AVX2_DP_2X8((ftmp+0*_CEIL8(dh_dim)), (ftmp+1*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst0, dst1);
                    AVX2_DP_2X8((ftmp+2*_CEIL8(dh_dim)), (ftmp+3*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst2, dst3);
                    AVX2_DP_2X8((ftmp+4*_CEIL8(dh_dim)), (ftmp+5*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst4, dst5);
                    AVX2_DP_2X8((ftmp+6*_CEIL8(dh_dim)), (ftmp+7*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst6, dst7);
                    sh[shsize * (ind+0) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst0), 24));
                    sh[shsize * (ind+1) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst1), 24));
                    sh[shsize * (ind+2) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst2), 24));
                    sh[shsize * (ind+3) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst3), 24));
                    sh[shsize * (ind+4) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst4), 24));
                    sh[shsize * (ind+5) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst5), 24));
                    sh[shsize * (ind+6) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst6), 24));
                    sh[shsize * (ind+7) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst7), 24));
                }
                // compute t part
                float *ptr0 = fvec + (ind+0) * FD8 + LD;
                float *ptr1 = fvec + (ind+1) * FD8 + LD;
                float *ptr2 = fvec + (ind+2) * FD8 + LD;
                float *ptr3 = fvec + (ind+3) * FD8 + LD;
                float *ptr4 = fvec + (ind+4) * FD8 + LD;
                float *ptr5 = fvec + (ind+5) * FD8 + LD;
                float *ptr6 = fvec + (ind+6) * FD8 + LD;
                float *ptr7 = fvec + (ind+7) * FD8 + LD;
                for (long k = 0; k < num_tbits; k += 8) {
                    __attribute__ ((aligned (32))) float ttmp[64];
                    for (long i = k; i < k+8; i++) {
                        ttmp[0*8+i-k] = (i < p->CSD) ? ptr0[compress_pos[6*i+0]] : 
                                    ptr0[compress_pos[6*i+0]] + ptr0[compress_pos[6*i+1]] + ptr0[compress_pos[6*i+2]] - 
                                    ptr0[compress_pos[6*i+3]] - ptr0[compress_pos[6*i+4]] - ptr0[compress_pos[6*i+5]];
                        ttmp[1*8+i-k] = (i < p->CSD) ? ptr1[compress_pos[6*i+0]] :
                                    ptr1[compress_pos[6*i+0]] + ptr1[compress_pos[6*i+1]] + ptr1[compress_pos[6*i+2]] - 
                                    ptr1[compress_pos[6*i+3]] - ptr1[compress_pos[6*i+4]] - ptr1[compress_pos[6*i+5]];
                        ttmp[2*8+i-k] = (i < p->CSD) ? ptr2[compress_pos[6*i+0]] :
                                    ptr2[compress_pos[6*i+0]] + ptr2[compress_pos[6*i+1]] + ptr2[compress_pos[6*i+2]] - 
                                    ptr2[compress_pos[6*i+3]] - ptr2[compress_pos[6*i+4]] - ptr2[compress_pos[6*i+5]];
                        ttmp[3*8+i-k] = (i < p->CSD) ? ptr3[compress_pos[6*i+0]] :
                                    ptr3[compress_pos[6*i+0]] + ptr3[compress_pos[6*i+1]] + ptr3[compress_pos[6*i+2]] - 
                                    ptr3[compress_pos[6*i+3]] - ptr3[compress_pos[6*i+4]] - ptr3[compress_pos[6*i+5]];
                        ttmp[4*8+i-k] = (i < p->CSD) ? ptr4[compress_pos[6*i+0]] :
                                    ptr4[compress_pos[6*i+0]] + ptr4[compress_pos[6*i+1]] + ptr4[compress_pos[6*i+2]] - 
                                    ptr4[compress_pos[6*i+3]] - ptr4[compress_pos[6*i+4]] - ptr4[compress_pos[6*i+5]];
                        ttmp[5*8+i-k] = (i < p->CSD) ? ptr5[compress_pos[6*i+0]] :
                                    ptr5[compress_pos[6*i+0]] + ptr5[compress_pos[6*i+1]] + ptr5[compress_pos[6*i+2]] - 
                                    ptr5[compress_pos[6*i+3]] - ptr5[compress_pos[6*i+4]] - ptr5[compress_pos[6*i+5]];
                        ttmp[6*8+i-k] = (i < p->CSD) ? ptr6[compress_pos[6*i+0]] :
                                    ptr6[compress_pos[6*i+0]] + ptr6[compress_pos[6*i+1]] + ptr6[compress_pos[6*i+2]] - 
                                    ptr6[compress_pos[6*i+3]] - ptr6[compress_pos[6*i+4]] - ptr6[compress_pos[6*i+5]];
                        ttmp[7*8+i-k] = (i < p->CSD) ? ptr7[compress_pos[6*i+0]] :
                                    ptr7[compress_pos[6*i+0]] + ptr7[compress_pos[6*i+1]] + ptr7[compress_pos[6*i+2]] - 
                                    ptr7[compress_pos[6*i+3]] - ptr7[compress_pos[6*i+4]] - ptr7[compress_pos[6*i+5]];
                    }

                    sh[shsize*(ind+0)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 0*8));
                    sh[shsize*(ind+1)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 1*8));
                    sh[shsize*(ind+2)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 2*8));
                    sh[shsize*(ind+3)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 3*8));
                    sh[shsize*(ind+4)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 4*8));
                    sh[shsize*(ind+5)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 5*8));
                    sh[shsize*(ind+6)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 6*8));
                    sh[shsize*(ind+7)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 7*8));
                }
                #if 0
                do {
                    for (long j = 0; j < 8; j++) {
                        float *ptr = fvec + (ind+j) * FD8 + LD;
                        for (long k = 0; k < num_hbits; k++) {
                            float x = dot_avx2(dual_vec + k * _CEIL8(dh_dim), ftmp + j * _CEIL8(dh_dim), dh_dim);
                            int32_t y = roundf(x);
                            int8_t z = y;
                            if ((z < 0) ^ ((sh[shsize * (ind+j) + k/8] & (1 << (k%8))) != 0)) {
                                printf("here, ind = %ld, j = %ld, k = %ld, x = %f, z = %x\n", ind, j, k, x, z);
                            }
                        }
                        for (long k = 0; k < num_tbits; k++) {
                            float x = (k < p->CSD) ? ptr[compress_pos[6*k+0]] : 
                                    ptr[compress_pos[6*k+0]] + ptr[compress_pos[6*k+1]] + ptr[compress_pos[6*k+2]] - 
                                    ptr[compress_pos[6*k+3]] - ptr[compress_pos[6*k+4]] - ptr[compress_pos[6*k+5]];
                            int32_t *y = (int32_t *)&x;
                            if ((*y < 0) ^ ((sh[shsize * (ind+j) + num_hbits/8 + k/8] & (1 << (k%8))) != 0)) {
                                printf("here, ind = %ld, j = %ld, k = %ld, x = %f, y = %x\n", ind, j, k, x, *y);
                            }
                        }
                    }
                } while (0);
                #endif
                ind += 8;
            }
            while (ind < end_ind) {
                p->compute_coeff(ctmp, MInd + ind);
                set_zero_avx2(fvec + FD8 * ind, FD8);
                for (long i = 0; i < p->CSD; i++) {
                    red_avx2(fvec + FD8 * ind, b_ext_fp[i], -ctmp[i], LD + i + 1);
                }
                // TODO: add lift here
                for (long l = 0; l < dh_dim; l += 8) {
                    _mm256_store_ps(ftmp + l, _mm256_loadu_ps(fvec + ind * FD8 + ID + l));
                }
                // compute h part
                for (long k = 0; k < num_hbits; k += 8) {
                    __m256 dst;
                    AVX2_DP_1X8(ftmp, (dual_vec + (k * _CEIL8(dh_dim))), _CEIL8(dh_dim), dst);
                    sh[shsize * ind + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst), 24));
                }
                // compute t part
                float *ptr0 = fvec + ind * FD8 + LD;
                for (long k = 0; k < num_tbits; k += 8) {
                    __attribute__ ((aligned (32))) float ttmp[8];
                    for (long i = k; i < k+8; i++) {
                        ttmp[i-k] = (k < p->CSD) ? ptr0[compress_pos[6*i+0]] : 
                                    ptr0[compress_pos[6*i+0]] + ptr0[compress_pos[6*i+1]] + ptr0[compress_pos[6*i+2]] - 
                                    ptr0[compress_pos[6*i+3]] - ptr0[compress_pos[6*i+4]] - ptr0[compress_pos[6*i+5]];
                        sh[shsize*ind+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp));
                    }
                }
                ind++;
            }
        }
    }
};

template <uint32_t nb>
template <uint32_t shsize, uint32_t l1_block, uint32_t l2_block>
int Pool_epi8_t<nb>::__parallel_mblock_sh_search(nsh_mblock_t mb, int32_t threshold, uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, nsh_profile_data_t *profile_data) {
    // create l2 block list
    long *l2_block_list = (long *) NEW_VEC(((mb.MIbound + l2_block - 1)/l2_block) * ((mb.MJbound+l2_block-1)/l2_block) * 2, sizeof(long));
    long n_l2_block = 0;
    if (mb.fveci != mb.fvecj) {
        for (long Ind = 0; Ind < mb.MIbound; Ind += l2_block) {
            for (long Jnd = 0; Jnd < mb.MJbound; Jnd += l2_block) {
                l2_block_list[n_l2_block*2] = Ind;
                l2_block_list[n_l2_block*2+1] = Jnd;
                n_l2_block++;
            }
        }
        pthread_spin_lock(&profile_data->profile_lock);
        profile_data->num_mblock++;
        profile_data->dp_ops += mb.MIbound * mb.MJbound;
        pthread_spin_unlock(&profile_data->profile_lock);
    } else {
        for (long Ind = 0; Ind < mb.MIbound; Ind += l2_block) {
            for (long Jnd = Ind; Jnd < mb.MJbound; Jnd += l2_block) {
                l2_block_list[n_l2_block*2] = Ind;
                l2_block_list[n_l2_block*2+1] = Jnd;
                n_l2_block++;
            }
        }
        pthread_spin_lock(&profile_data->profile_lock);
        profile_data->num_mblock++;
        profile_data->dp_ops += mb.MIbound * (mb.MIbound + 1) / 2;
        pthread_spin_unlock(&profile_data->profile_lock);
    }
    
    TIMER_START;
    pthread_spinlock_t buffer_lock;
    pthread_spin_init(&buffer_lock, PTHREAD_PROCESS_SHARED);
    long nrem = n_l2_block;
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long _ptr_buffer_num = ptr_buffer_num[thread];
        uint32_t *_ptr_buffer = ptr_buffer[thread];
        long _ptr_buffer_size = ptr_buffer_size[thread];
        while (nrem > 0) {
            // get a l2_block
            long Ind, Jnd = -1;
            pthread_spin_lock(&buffer_lock);
            if (nrem > 0) {
                Ind = l2_block_list[(n_l2_block - nrem) * 2];
                Jnd = l2_block_list[(n_l2_block - nrem) * 2 + 1];
                nrem--;
            }
            pthread_spin_unlock(&buffer_lock);
            if (Jnd == -1) break;
            
            if (Ind == Jnd && mb.fveci == mb.fvecj) {
                // a triangular block
                const long Ibound = (Ind + l2_block > mb.MIbound) ? mb.MIbound : Ind + l2_block;
                const long Jbound = (Jnd + l2_block > mb.MJbound) ? mb.MJbound : Jnd + l2_block;
                long ind = Ind;
                while (ind <= Ibound - l1_block) {
                    long jnd = ind;
                    // the first triangular block
                    _process_nshl1_triblock<shsize>(mb.shi + ind * shsize, ind, l1_block, threshold, 
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    jnd += l1_block;
                    // square blocks
                    while (jnd <= Jbound - l1_block) {
                        _process_nshl1_block<shsize>(mb.shi + ind * shsize, mb.shj + jnd * shsize, 
                                                ind, jnd, l1_block, l1_block, threshold,
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                        jnd += l1_block;
                    }
                    if (jnd < Jbound) {
                        _process_nshl1_block<shsize>(mb.shi + ind * shsize, mb.shj + jnd * shsize, 
                                                ind, jnd, l1_block, Jbound - jnd, threshold,
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    }
                    ind += l1_block;
                }
                if (ind < Ibound) {
                    long jnd = ind;
                    _process_nshl1_triblock<shsize>(mb.shi + ind * shsize, ind, Ibound - ind, threshold,
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                }
            } else {
                // normal block
                const long Ibound = (Ind + l2_block > mb.MIbound) ? mb.MIbound : Ind + l2_block;
                const long Jbound = (Jnd + l2_block > mb.MJbound) ? mb.MJbound : Jnd + l2_block;
                long ind = Ind;
                while (ind <= Ibound - l1_block) {
                    long jnd = Jnd;
                    while (jnd <= Jbound - l1_block) {
                        _process_nshl1_block<shsize>(mb.shi + ind * shsize, mb.shj + jnd * shsize,
                                                    ind, jnd, l1_block, l1_block, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                        jnd += l1_block;
                    }
                    if (jnd < Jbound) {
                        _process_nshl1_block<shsize>(mb.shi + ind * shsize, mb.shj + jnd * shsize,
                                                    ind, jnd, l1_block, Jbound - jnd, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    }
                    ind += l1_block;
                }
                if (ind < Ibound) {
                    long jnd = Jnd;
                    while (jnd <= Jbound - l1_block) {
                        _process_nshl1_block<shsize>(mb.shi + ind * shsize, mb.shj + jnd * shsize,
                                            ind, jnd, Ibound - ind, l1_block, threshold,
                                            &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                        jnd += l1_block;
                    }
                    if (jnd < Jbound) {
                        _process_nshl1_block<shsize>(mb.shi + ind * shsize, mb.shj + jnd * shsize,
                                            ind, jnd, Ibound - ind, Jbound - jnd, threshold,
                                            &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    }
                }
            }
        }
        ptr_buffer[thread] = _ptr_buffer;
        ptr_buffer_size[thread] = _ptr_buffer_size;
        ptr_buffer_num[thread] = _ptr_buffer_num;
    }
    TIMER_END;
    profile_data->dp_time += CURRENT_TIME;
    FREE_VEC((void *)l2_block_list);
    return 1;
}
template <uint32_t nb>
int Pool_epi8_t<nb>::__lift_buffer(nsh_mblock_t mb, long target_index, uint32_t dh_dim, float **b_full_fp, 
                                    uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, 
                                    float *min_norm, float **min_vec, nsh_profile_data_t *profile_data) {
    const long FD = index_r - target_index;
    const long ID = index_l - dh_dim - target_index;
    const long FD8 = _CEIL8(FD);

    pthread_spinlock_t min_lock;
    pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);

    TIMER_START;
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        // local min data initialization
        __attribute__ ((aligned (32))) float _min_norm[vec_length];
        float **_min_vec = (float **) NEW_MAT(ID, FD, sizeof(float));
        pthread_spin_lock(&min_lock);
        copy_avx2(_min_norm, min_norm, ID);
        pthread_spin_unlock(&min_lock);

        // main lifting
        TIMER_START;
        do {
            const long LD = FD - CSD;
            float *ftmp = (float *) NEW_VEC(FD8 * 8, sizeof(float));
            float idiag[256];
            for (long i = 0; i < FD; i++) idiag[i] = 1.0 / b_full_fp[i][i];
            __attribute__ ((aligned (32))) float curr_norm[8];
            long ind = 0;
            while (ind + 8 <= ptr_buffer_num[thread]) {
                for (long i = 0; i < 8; i++) {
                    if (ptr_buffer[thread][(ind+i) * 3]) {
                        sub_avx2(ftmp + FD8 * i, mb.fveci + ptr_buffer[thread][(ind+i)*3+1]*FD8, mb.fvecj + ptr_buffer[thread][(ind+i)*3+2]*FD8, FD8);
                    } else {
                        add_avx2(ftmp + FD8 * i, mb.fveci + ptr_buffer[thread][(ind+i)*3+1]*FD8, mb.fvecj + ptr_buffer[thread][(ind+i)*3+2]*FD8, FD8);
                    }
                }
                for (long i = 0; i < 8; i++){
                    curr_norm[i] = dot_aux2(ftmp + FD8 * i + LD, ftmp + FD8 * i + LD, CSD);
                }
                for (long i = LD - 1; i >= 0; i--) {
                    red_avx2(ftmp + 0 * FD8, b_full_fp[i], round(ftmp[0 * FD8 + i] * idiag[i]), i+1);
                    red_avx2(ftmp + 1 * FD8, b_full_fp[i], round(ftmp[1 * FD8 + i] * idiag[i]), i+1);
                    red_avx2(ftmp + 2 * FD8, b_full_fp[i], round(ftmp[2 * FD8 + i] * idiag[i]), i+1);
                    red_avx2(ftmp + 3 * FD8, b_full_fp[i], round(ftmp[3 * FD8 + i] * idiag[i]), i+1);
                    red_avx2(ftmp + 4 * FD8, b_full_fp[i], round(ftmp[4 * FD8 + i] * idiag[i]), i+1);
                    red_avx2(ftmp + 5 * FD8, b_full_fp[i], round(ftmp[5 * FD8 + i] * idiag[i]), i+1);
                    red_avx2(ftmp + 6 * FD8, b_full_fp[i], round(ftmp[6 * FD8 + i] * idiag[i]), i+1);
                    red_avx2(ftmp + 7 * FD8, b_full_fp[i], round(ftmp[7 * FD8 + i] * idiag[i]), i+1);
                    for (long j = 0; j < 8; j++) curr_norm[j] += ftmp[j * FD8 + i] * ftmp[j * FD8 + i];
                    if (i >= ID) continue;
                    int r = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_broadcast_ss(_min_norm+i), _mm256_load_ps(curr_norm), 30));
                    int s = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_load_ps(curr_norm), _mm256_broadcast_ss(_min_norm), 30));
                    while (r) {
                        int rr = __builtin_ctz(r);
                        r -= (1 << rr);
                        if (curr_norm[rr] < _min_norm[i]) {
                            int has_one = 0;
                            __attribute__ ((aligned (32))) float ftmpp[256];
                            copy_avx2(ftmpp, ftmp + rr * FD8, FD);
                            for (long j = FD - 1; j >= FD - CSD; j--) {
                                float qq = round(ftmpp[j] * idiag[j]);
                                if (fabs(qq)  == 1.0f) {
                                    has_one = 1;
                                    break;
                                }
                                red_avx2(ftmpp, b_full_fp[j], qq, j+1);
                            }
                            if (has_one) {
                                _min_norm[i] = curr_norm[rr];
                                copy_avx2(_min_vec[i], ftmp + rr * FD8, FD);
                            }
                        }
                    }
                    if (s == 255) break;
                }

                ind += 8;
            }
            while (ind < ptr_buffer_num[thread]) {
                if (ptr_buffer[thread][ind * 3]) {
                    sub_avx2(ftmp, mb.fveci + ptr_buffer[thread][ind*3+1]*FD8, mb.fvecj + ptr_buffer[thread][ind*3+2]*FD8, FD8);
                } else {
                    add_avx2(ftmp, mb.fveci + ptr_buffer[thread][ind*3+1]*FD8, mb.fvecj + ptr_buffer[thread][ind*3+2]*FD8, FD8);
                }
                curr_norm[0] = dot_aux2(ftmp + LD, ftmp + LD, CSD);
                for (long i = LD - 1; i >= 0; i--) {
                    float q = round(ftmp[i] * idiag[i]);
                    red_avx2(ftmp, b_full_fp[i], q, i+1);
                    curr_norm[0] += ftmp[i] * ftmp[i];
                    if (i >= ID) continue;
                    if (curr_norm[0] < _min_norm[i]) {
                        // check if there is an one in the coeff
                        int has_one = 0;
                        __attribute__ ((aligned (32))) float ftmpp[256];
                        copy_avx2(ftmpp, ftmp, FD);
                        for (long j = FD - 1; j >= FD - CSD; j--) {
                            float qq = round(ftmpp[j] * idiag[j]);
                            if (fabs(qq)  == 1.0f) {
                                has_one = 1;
                                break;
                            }
                            red_avx2(ftmpp, b_full_fp[j], qq, j+1);
                        }
                        if (has_one) {
                            _min_norm[i] = curr_norm[0];
                            copy_avx2(_min_vec[i], ftmp, FD);
                        }
                    }
                    if (curr_norm[0] > _min_norm[0]) break;
                }
                ind++;
            }
            FREE_VEC(ftmp);
        } while (0);
        

        // update profile data
        pthread_spin_lock(&profile_data->profile_lock);
        profile_data->lift_ops += ptr_buffer_num[thread];
        ptr_buffer_num[thread] = 0;
        pthread_spin_unlock(&profile_data->profile_lock);

        // update min data
        pthread_spin_lock(&min_lock);
        for (long i = 0; i < ID; i++) {
            if (_min_norm[i] < min_norm[i]) {
                min_norm[i] = _min_norm[i];
                copy_avx2(min_vec[i], _min_vec[i], FD);
            }
        }
        pthread_spin_unlock(&min_lock);
        FREE_MAT(_min_vec);
    }
    TIMER_END;
    profile_data->lift_time += CURRENT_TIME;
    return 0;
}



template <uint32_t nb>
template <uint32_t shsize, uint32_t dh_dim, uint32_t l1_block, uint32_t l2_block, uint32_t m_block>
int Pool_epi8_t<nb>::_naivesh_insert(long target_index, double eta, long log_level, 
                            float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_length) {
    /////// check params ///////
    if (target_index + dh_dim > index_l) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::naivesh_insert: target_index(%ld) + dh_dim(%u) > index_l(%ld), nothing done.\n", 
                nb, target_index, dh_dim, index_l);
        return -1;
    }
    if (target_index < 0) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::naivesh_insert: negetive target_index(%ld), nothing done.\n", nb, target_index);
        return -1;
    }
    
    const long FD = index_r - target_index;
    const long ID = index_l - dh_dim - target_index;

    /////// min data initialization ///////
    __attribute__ ((aligned (32))) float min_norm[vec_length];
    float **min_vec = (float **) NEW_MAT(ID, FD, sizeof(float));
    for (long i = 0; i < ID; i++) min_norm[i] = (float) (0.995 * basis->get_B().hi[i + target_index]);
    if (target_length != 0.0) {
        for (long i = 0; i < ID; i++) {
            if (min_norm[i] > 1.0001 * target_length * target_length) min_norm[i] = 1.0001 * target_length * target_length;
        }
    }


    /////// profiling data and buffer initialization ///////
    nsh_profile_data_t profile_data;
    profile_data.init(log_level);
    uint32_t **ptr_buffer = (uint32_t **) NEW_VEC(num_threads, sizeof(uint32_t *));
    uint64_t *ptr_buffer_size = (uint64_t *) NEW_VEC(num_threads, sizeof(uint64_t));
    uint64_t *ptr_buffer_num = (uint64_t *) NEW_VEC(num_threads, sizeof(uint64_t));
    for (long i = 0; i < num_threads; i++) {
        ptr_buffer[i] = (uint32_t *) malloc(8192 * 3 * sizeof(uint32_t));
        ptr_buffer_size[i] = 8192;
        ptr_buffer_num[i] = 0;
    }

    
    float **b_full_fp;
    nsh_mblock_provider_t<nb, shsize, dh_dim, m_block> mblock_provider;
    mblock_provider.init(this, target_index, dual_vec, compress_pos, num_hbits, b_full_fp, &profile_data);
    nsh_mblock_t mb;
    while (mblock_provider.pop(&mb, min_norm, min_vec, b_full_fp, &profile_data)) {
        __parallel_mblock_sh_search<shsize, l1_block, l2_block>(mb, threshold, ptr_buffer, ptr_buffer_size, ptr_buffer_num, &profile_data);
        __lift_buffer(mb, target_index, dh_dim, b_full_fp, 
                        ptr_buffer, ptr_buffer_size, ptr_buffer_num, 
                        min_norm, min_vec, &profile_data);
        profile_data.mblock_log(min_norm, ID);
    }
    profile_data.final_log(min_norm, ID);

    mblock_provider.clear();
    for (long i = 0; i < num_threads; i++) free(ptr_buffer[i]);
    FREE_VEC(ptr_buffer);
    FREE_VEC(ptr_buffer_size);
    FREE_VEC(ptr_buffer_num);

    long min_place = -1;
    do {
        double min_score = 1e100;
        for (long i = 0; i < ID; i++) {
            float old_norm = (float) (0.995 * basis->get_B().hi[i + target_index]);
            if (target_length != 0.0 && min_norm[i] > 1.00009 * target_length * target_length) continue;
            if (min_norm[i] < old_norm) {
                if (min_score > min_norm[i] / old_norm * pow(eta, i)) {
                    min_score = min_norm[i] / old_norm * pow(eta, i);
                    min_place = i + target_index;
                }
            }
        }
    } while (0);

    if (min_place == -1) {
        FREE_MAT(min_vec);
        FREE_MAT(b_full_fp);
        shrink_left();
        return 0;
    }

    __attribute__ ((aligned (32))) float v_fp[256];
    copy_avx2(v_fp, min_vec[min_place - target_index], FD);
    FREE_MAT(min_vec);
    int ret = __basis_insert(min_place, v_fp, FD, b_full_fp);
    FREE_MAT(b_full_fp); 

    return 0;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::naivesh_insert(long target_index, double eta, long log_level, double target_length) {
    /////// basic information of the lifting ///////
    const double tail_gh = sqrt(gh2);
    const double dual_gh = basis->gh(index_l - NSH_DEFAULT_DHDIM, index_l);
    const double lift_gh = basis->gh(target_index, index_l);
    const double tail_exp_ratio = sqrt(cvec[3*(num_vec/2)+2] * 4.0) / _ratio / tail_gh;

    double tail_exp_alpha = 0.0;
    do {
        double min_exp_length = lift_gh;
        for (double _alpha = 0.0; _alpha < 0.5; _alpha += 0.01) {
            uint64_t _num_lift = num_vec * (num_vec - 1) * pow(1 - _alpha * _alpha, CSD * 0.5);
            double _tail_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * _alpha);
            double _lift_length = lift_gh / pow(_num_lift, 1.0/(index_l - target_index));
            double _length = sqrt(_tail_length * _tail_length + _lift_length * _lift_length);
            if (_length < min_exp_length) {
                tail_exp_alpha = _alpha;
                min_exp_length = _length;
            }
        }
    } while (0);

    const uint64_t exp_num_lift = num_vec * (num_vec - 1) * pow(1 - tail_exp_alpha * tail_exp_alpha, CSD*0.5);
    const double tail_exp_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * tail_exp_alpha);
    const double lift_exp_length = lift_gh / pow(exp_num_lift, 1.0/(index_l - target_index));
    const double exp_length = sqrt(tail_exp_length * tail_exp_length + lift_exp_length * lift_exp_length);
    const bool unique_target = (target_length != 0.0) && (target_length < 0.96 * exp_length);

    // the probability that the solution with exp_length is filtered out
    double tail_alpha_prob_list[64] = {};
    do {
        if (!unique_target) {
            for (long i = 0; i < 50; i++) {
                double _alpha = 0.01 * i;
                double _tail_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * _alpha);
                double _num_lift = num_vec * (num_vec - 1) * pow(1 - _alpha * _alpha, CSD * 0.5);
                double _lift_length = lift_gh / pow(_num_lift, 1.0/(index_l - target_index));
                double _target_lift_length = sqrt(exp_length * exp_length - _tail_length * _tail_length);
                if (isnan(_target_lift_length)) _target_lift_length = 0.0;
                double _prob = pow(_target_lift_length / _lift_length, index_l - target_index);
                tail_alpha_prob_list[i] = _prob;
            }
            for (long i = 48; i >= 0; i--) {
                tail_alpha_prob_list[i] += tail_alpha_prob_list[i+1];
            }
            for (long i = 49; i >= 0; i--) {
                tail_alpha_prob_list[i] /= tail_alpha_prob_list[0];
            }
        } else {
            long N = 100000;
            DGS1d R;
            
            long count[1000] = {};
            __attribute__ ((aligned (32))) float tmp[256] = {};
            for (long i = 0; i < N; i++) {
                for (long j = 0; j < index_r - index_l; j++) {
                    tmp[j] = R.discrete_gaussian(0.0, 1048576.0);
                }
                for (long j = index_r - index_l; j < index_r - index_l + 8; j++) {
                    tmp[j] = 0.0f;
                }
                float x = dot_avx2(tmp, tmp, index_r - index_l);
                for (long j = index_r - index_l; j < index_r - target_index; j++) {
                    tmp[j] = R.discrete_gaussian(0.0, 1048576.0);
                }
                float y = dot_avx2(tmp, tmp, index_r - target_index);
                _ratio = x / y;
                long _index = floor(_ratio * 1000);
                count[_index]++;                
            }
            for (long i = 1; i < 1000; i++) {
                count[i] += count[i-1];
            }
            for (long i = 0; i < 50; i++) {
                double _alpha = 0.01 * i;
                double _tail_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * _alpha);
                double _ratio = _tail_length / target_length;
                _ratio = _ratio * _ratio;
                long _index = floor(_ratio * 1000);
                _index = (_index >= 1000) ? 999 : _index;
                double _prob = count[_index] / (double)N;
                tail_alpha_prob_list[i] = _prob;
            }
        }    
    } while (0);

    // a conservative estimation for normal case
    const double dual_exp_length = unique_target ? 
                        target_length * sqrt(NSH_DEFAULT_DHDIM) / sqrt(index_r - target_index) :
                        lift_exp_length * sqrt(NSH_DEFAULT_DHDIM) / sqrt(index_l - target_index);

    if (log_level >= 1) {
        printf("tail_gh = %.2f, dual_gh = %.2f, lift_gh = %.2f\n", tail_gh, dual_gh, lift_gh);
        printf("tail_exp_alpha = %.2f, tail_exp_ratio = %.2f, tail_exp_length = %.2f\n", tail_exp_alpha, tail_exp_ratio, tail_exp_length);
        printf("exp_num_lift = %lu = (%.2f)^{-%ld}, lift_exp_length = %.2f, exp_length = %.2f, dual_exp_length = %.2f(%.2f, type %s)\n",
                exp_num_lift, pow(exp_num_lift, -1.0/(index_l - target_index)), index_l-target_index, lift_exp_length, exp_length, 
                dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
    } else if (log_level == 0) {
        if (target_length != 0.0) {
            printf("exp_length = %.2f, target_length = %.2f(%.3f), dual_exp_length = %.2f(%.2f, type %s), ",exp_length, target_length, 
                    target_length / exp_length, dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
        } else {
            printf("exp_length = %.2f, dual_exp_length = %.2f(%.2f, type %s), ", exp_length, 
                    dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
        }
    }
    
    /////// choosing parameters /////// 
    // output values
    float *dual_vec = (float *) NEW_VEC(NSH_DEFAULT_SHSIZE * 8 * _CEIL8(NSH_DEFAULT_DHDIM), sizeof(float));
    uint32_t *compress_pos = (uint32_t *) NEW_VEC(NSH_DEFAULT_SHSIZE * 8 * 6, sizeof(uint32_t));
    int32_t num_hbits, num_tbits, threshold;
    // input values
    Lattice_QP *b_mid = basis->b_loc_QP(index_l - NSH_DEFAULT_DHDIM, index_l);
    TIMER_START;
    _opt_nsh_threshold(dual_vec, compress_pos, num_hbits, num_tbits, threshold,
                        b_mid, NSH_DEFAULT_SHSIZE, exp_length, tail_alpha_prob_list, log_level, target_index);
    TIMER_END;
    delete b_mid;
    if (log_level >= 0) printf("param_opt_time = %.2fs, num_hbits = %d, num_tbits = %d, threshold = %d\n", CURRENT_TIME, num_hbits, num_tbits, threshold);

    /////// call the lifting kernel ///////
    int ret = _naivesh_insert<NSH_DEFAULT_SHSIZE, NSH_DEFAULT_DHDIM, NSH_L1_BLOCK, NSH_L2_BLOCK, NSH_M_BLOCK>
                (target_index, eta, log_level, dual_vec, compress_pos, num_hbits, threshold, unique_target ? target_length : 0.0);
    
    FREE_VEC(dual_vec);
    FREE_VEC((void *)compress_pos);
    return ret;
}

#if COMPILE_POOL_EPI8_96
template void Pool_epi8_t<3>::_process_nshl1_triblock<NSH_DEFAULT_SHSIZE>(uint8_t *sh, long bias, long bound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
template void Pool_epi8_t<3>::_process_nshl1_block<NSH_DEFAULT_SHSIZE>(uint8_t *shi, uint8_t *shj, long ibias, long jbias, long ibound, long jbound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
template int Pool_epi8_t<3>::_naivesh_insert<NSH_DEFAULT_SHSIZE, NSH_DEFAULT_DHDIM, NSH_L1_BLOCK, NSH_L2_BLOCK, NSH_M_BLOCK>
                (long target_index, double eta, long log_level, float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_length);
template class Pool_epi8_t<3>;
#endif

#if COMPILE_POOL_EPI8_128
template void Pool_epi8_t<4>::_process_nshl1_triblock<NSH_DEFAULT_SHSIZE>(uint8_t *sh, long bias, long bound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
template void Pool_epi8_t<4>::_process_nshl1_block<NSH_DEFAULT_SHSIZE>(uint8_t *shi, uint8_t *shj, long ibias, long jbias, long ibound, long jbound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
template int Pool_epi8_t<4>::_naivesh_insert<NSH_DEFAULT_SHSIZE, NSH_DEFAULT_DHDIM, NSH_L1_BLOCK, NSH_L2_BLOCK, NSH_M_BLOCK>
                (long target_index, double eta, long log_level, float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_length);
template class Pool_epi8_t<4>;
#endif

#if COMPILE_POOL_EPI8_160
template void Pool_epi8_t<5>::_process_nshl1_triblock<NSH_DEFAULT_SHSIZE>(uint8_t *sh, long bias, long bound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
template void Pool_epi8_t<5>::_process_nshl1_block<NSH_DEFAULT_SHSIZE>(uint8_t *shi, uint8_t *shj, long ibias, long jbias, long ibound, long jbound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);
template int Pool_epi8_t<5>::_naivesh_insert<NSH_DEFAULT_SHSIZE, NSH_DEFAULT_DHDIM, NSH_L1_BLOCK, NSH_L2_BLOCK, NSH_M_BLOCK>
                (long target_index, double eta, long log_level, float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_length);
template class Pool_epi8_t<5>;
#endif
