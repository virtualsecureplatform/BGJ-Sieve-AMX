#include "../include/config.h"

#if defined(__AMX_INT8__)
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <immintrin.h>

#include "../include/bucket_amx.h"
#include "../include/bgj_amx.h"
#include "../include/pool_epi8.h"

///////////////// bucket_epi8_t /////////////////

// dst and src may not be aligned, size must divided by 4
static void memcpy_avx512(void *dst, void *src, long size) {
    long i = 0;
    while (i < size - 63) {
        _mm512_storeu_si512((uint8_t *)dst + i, _mm512_loadu_si512((uint8_t *)src + i));
    }
    if (i < size - 31) {
        _mm256_storeu_si256((__m256i *)((uint8_t *)dst + i), _mm256_loadu_si256((__m256i *)((uint8_t *)src + i)));
        i += 32;
    }
    if (i < size - 15) {
        _mm_storeu_si128((__m128i *)((uint8_t *)dst + i), _mm_loadu_si128((__m128i *)((uint8_t *)src + i)));
        i += 16;
    }
    if (i < size - 7) {
        *(uint64_t *)((uint8_t *)dst + i) = *(uint64_t *)((uint8_t *)src + i);
        i += 8;
    }
    if (i < size - 3) {
        *(uint32_t *)((uint8_t *)dst + i) = *(uint32_t *)((uint8_t *)src + i);
        i += 4;
    }
}

int bucket_amx_t::_clear() {
    if (pvec) free(pvec);
    if (nvec) free(nvec);
    if (pnorm) free(pnorm);
    if (nnorm) free(nnorm);
    pvec = NULL; nvec = NULL;
    pnorm = NULL; nnorm = NULL;
    return 0;
}

int bucket_amx_t::_alloc(long size, bool p) {
    if (p) {
        if (size <= _psize) return 0;
        pvec = (uint32_t *) realloc(pvec, size * sizeof(uint32_t)+64);
        pnorm = (int32_t *) realloc(pnorm, size * sizeof(int32_t)+64);
        _psize = size;
        return (pnorm && pvec);
    } else {
        if (size <= _nsize) return 0;
        nvec = (uint32_t *) realloc(nvec, size * sizeof(uint32_t)+64);
        nnorm = (int32_t *) realloc(nnorm, size * sizeof(int32_t)+64);
        _nsize = size;
        return (nnorm && nvec);
    }
}

void bucket_amx_t::combine(bucket_amx_t **subbucket_list, long len) {
    if (num_pvec || num_nvec) {
        fprintf(stderr, "[Error] bucket_amx_t::combine: combining subbuckets to nonempty buckets, aborted.\n");
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
        memcpy_avx512(pvec + num_pvec, subbucket_list[i]->pvec, subbucket_list[i]->num_pvec * sizeof(uint32_t));
        memcpy_avx512(nvec + num_nvec, subbucket_list[i]->nvec, subbucket_list[i]->num_nvec * sizeof(uint32_t));
        memcpy_avx512(pnorm + num_pvec, subbucket_list[i]->pnorm, subbucket_list[i]->num_pvec * sizeof(int32_t));
        memcpy_avx512(nnorm + num_nvec, subbucket_list[i]->nnorm, subbucket_list[i]->num_nvec * sizeof(int32_t));
        num_pvec += subbucket_list[i]->num_pvec;
        num_nvec += subbucket_list[i]->num_nvec;
    }
}

int bucket_amx_t::remove_center(int max_unordered) {
    if (pvec == NULL) return 0;
    long low = 0;
    long high = num_pvec - 1;
    long target_ind = -1;
    do {
        while (high - low > 1) {
            long mid = (high + low) / 2; 
            if (pvec[mid] >= center_ind) {
                high = mid;
            } else {
                low = mid;
            }
        }
        if (pvec[low] == center_ind) {
            target_ind = low;
            break;
        }
        if (pvec[high] == center_ind) {
            target_ind = high;
            break;
        }
        // not find ?
        if (target_ind == -1) {
            if (pvec[low] >= pvec[num_pvec - 1]) {
                if (pvec[high] >= pvec[num_pvec - 1]) {
                    if (low > 0) {
                        if (pvec[low-1] >= center_ind) {
                            high = low - 1;
                            low = 0;
                        } else if (high < num_pvec - 1) {
                            low = high + 1;
                            high = num_pvec - 1;
                        }
                    } else if (num_pvec > 1) {
                        low = 1;
                        high = num_pvec - 1;
                    }
                } else {
                    if (pvec[high] < center_ind) {
                        low = high;
                        high = num_pvec - 1;
                    } else if (low > 0) {
                        high = low - 1;
                        low = 0;
                    }
                }
            } else if (pvec[high] >= pvec[num_pvec - 1]) {
                if (high < num_pvec - 1) {
                    low = high + 1;
                    high = num_pvec - 1;
                }
            } else if (pvec[num_pvec - 1] == center_ind) {
                target_ind = num_pvec - 1;
                break;
            } else {
                low = 0;
                high = num_pvec - 2;
                return -1;
            }
        }
    } while (max_unordered -->= 0);
    if (target_ind == -1) {
        for (long i = 0; i < num_pvec; i++) {
            if (pvec[i] == center_ind) {
                target_ind = i;
                break;
            }
        }
        if (target_ind == -1) {
            fprintf(stderr, "# center not found in the bucket\n");
            return -1;
        } else {
            if (num_pvec >= 2) {
                if (pvec[target_ind] < pvec[num_pvec - 2] && pvec[target_ind] < pvec[num_pvec - 1]) {
                    fprintf(stderr, "# center finally found by a brute force search\n");
                }
            }
        }
    } 
    
    num_pvec--;
    pvec[target_ind] = pvec[num_pvec];
    pnorm[target_ind] = pnorm[num_pvec];

    return 1;
}


///////////////// sol_list_epi8_t /////////////////

int sol_list_amx_t::_clear(){
    if (a_list) free(a_list);           // 0
    if (s_list) free(s_list);           // 1
    a_list = NULL; s_list = NULL;
    return 0;
}

int sol_list_amx_t::_alloc(long size, long type) {
    #define _REALLOC_PTR(__ptr, __orgsize, __r) do {                        \
        if (size <= __orgsize) return 0;                                    \
        __ptr = (uint32_t *) realloc(__ptr, size * sizeof(uint32_t) * __r); \
        __orgsize = size;                                                   \
        return (__ptr != NULL);                                             \
    } while (0);

    if (type == 0) _REALLOC_PTR(a_list, _asize, 2);
    if (type == 1) _REALLOC_PTR(s_list, _ssize, 2);
    #undef _REALLOC_PTR
    return -1;
}

long sol_list_amx_t::num_sol() {
    return num_a + num_s;
}


///////////////// bgjn sieve operations /////////////////

#define TRY_ADDA_AMX(__ind0, __ind1)                        \
                                     do {                   \
    try_add2++;                                             \
    uint64_t __u = vu[__ind0] + vu[__ind1];                 \
    if (BOOST_AMX_SIEVE) {                                  \
        succ_add2++;                                        \
        sol->add_sol_a(__ind0, __ind1);                     \
    } else {                                                \
        if (uid->insert_uid(__u)) {                         \
            succ_add2++;                                    \
            sol->add_sol_a(__ind0, __ind1);                 \
        }                                                   \
    }                                                       \
} while (0)

#define TRY_ADDS_AMX(__ind0, __ind1)                        \
                                     do {                   \
    try_add2++;                                             \
    uint64_t __u = vu[__ind0] - vu[__ind1];                 \
    if (BOOST_AMX_SIEVE) {                                  \
        succ_add2++;                                        \
        sol->add_sol_s(__ind0, __ind1);                     \
    } else {                                                \
        if (uid->insert_uid(__u)) {                         \
            succ_add2++;                                    \
            sol->add_sol_s(__ind0, __ind1);                 \
        }                                                   \
    }                                                       \
} while (0)

template <uint32_t nb>
template <uint32_t batchsize, bool faraway_center, bool reuse>
int Pool_epi8_t<nb>::_pool_bucketing_amx(bucket_amx_t **rbucket0, bucket_amx_t **bucket0, double alpha_r0, double alpha_b0, 
                                        sol_list_amx_t **sol_list, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof) {
    /////// prepare dst and local buffer ///////
    for (long i = 0; i < batchsize; i++) {
        if (reuse) {
            if (rbucket0[i]) {
                if (rbucket0[i]->num_pvec || rbucket0[i]->num_nvec) {
                    fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_pool_bucketing_amx: nonempty input buckets, ignored.\n", nb);
                    rbucket0[i]->num_pvec = 0;
                    rbucket0[i]->num_nvec = 0;
                }
            } else {
                rbucket0[i] = new bucket_amx_t;
            }
            rbucket0[i]->_alloc(256, 1);
            rbucket0[i]->_alloc(256, 0);
        }
        if (bucket0[i]) {
            if (bucket0[i]->num_pvec || bucket0[i]->num_nvec) {
                fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_pool_bucketing_amx: nonempty input buckets, ignored.\n", nb);
                bucket0[i]->num_pvec = 0;
                bucket0[i]->num_nvec = 0;
            } 
        } else {
            bucket0[i] = new bucket_amx_t;
        }
        bucket0[i]->_alloc(256, 1);
        bucket0[i]->_alloc(256, 0);
    }
    const __m512 alpha_r0x2_ps = _mm512_set1_ps(alpha_r0 * 2.0);
    const __m512 alpha_b0x2_ps = _mm512_set1_ps(alpha_b0 * 2.0);
    const long expect_bucket0_size = pow(1.0 - alpha_b0 * alpha_b0, CSD * 0.5) * num_vec;
    const long init_buffer_size = 0.8 * expect_bucket0_size / num_threads + batchsize * 64;
    uint32_t *local_buf[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_num[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_pnum[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_nnum[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_rpnum[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_rnnum[batchsize * AMX_MAX_NTHREADS];
    for (long i = 0; i < batchsize * num_threads; i++) {
        local_buf[i] = (uint32_t *) malloc(init_buffer_size * 2 * sizeof(uint32_t));
    }
    uint64_t try_add2_stat[AMX_MAX_NTHREADS];
    uint64_t succ_add2_stat[AMX_MAX_NTHREADS];

    /////// choose centers ///////
    uint32_t center_ind_list[batchsize];
    int32_t center_norm[batchsize];
    int8_t *center0_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    int8_t *center1_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    int8_t *center2_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    long num_try_find_center = 0;
    for (long i = 0; i < batchsize; i++) {
        int pass;
        do {
            pass = 1;
            center_ind_list[i] = Uniform_long(num_vec);
            if (num_try_find_center < AMX_MAX_NUM_TRYCENTER) {
                for (long j = 0; j < i; j++){
                    if (center_ind_list[j] == center_ind_list[i]) pass = 0;
                }
            }
            int8_t *ptr = vec + center_ind_list[i] * vec_length;
            num_try_find_center++;
            if (faraway_center) {
                if ((CSD * (int)ptr[0] * (int)ptr[0] > 2 * vnorm[center_ind_list[i]]) && (num_try_find_center < (long)(0.8 * AMX_MAX_NUM_TRYCENTER))) pass = 0;
            }
            if (reuse) rbucket0[i]->center_ind = center_ind_list[i];
            bucket0[i]->center_ind = center_ind_list[i];
            center_norm[i] = vnorm[center_ind_list[i]] - goal_norm;
        } while(!pass);
    }
    if (num_try_find_center >= (long)(0.8 * AMX_MAX_NUM_TRYCENTER)) {
        if (CSD >= 108) fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_pool_bucketing_amx: fail to use faraway centers after %ld tries.\n", nb, num_try_find_center);
    }
    for (long i = 0; i < batchsize; i += 16) {
        __m512i z[16];
        __m256i y[16];
        for (long j = 0; j < 16; j++) {
            z[j] = _mm512_loadu_si512((__m512i *)(vec + center_ind_list[i+j] * vec_length));
        }
        AVX512_MATTR_16x16(NULL, center0_tr + i * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
        for (long j = 0; j < 16; j++) {
            z[j] = _mm512_loadu_si512((__m512i *)(vec + center_ind_list[i+j] * vec_length + 64));
        }
        AVX512_MATTR_16x16(NULL, center1_tr + i * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
        for (long j = 0; j < 16; j++) {
            y[j] = _mm256_load_si256((__m256i *)(vec + center_ind_list[i+j] * vec_length + 128));
        }
        AVX512_MATTR_16x8(NULL, center2_tr + i * 64, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]);
    }

    /////// each thread collect vectors in local buffers ///////
    #pragma omp parallel for 
    for (long thread = 0; thread < num_threads; thread++) {
        ///// tile configuration /////
        TILE_INITIALIZE;

        ///// prepare local buffers //////
        uint32_t _local_buf_size[batchsize];
        uint32_t _local_buf_num[batchsize] = {};
        uint32_t _local_buf_nnum[batchsize] = {};
        uint32_t _local_buf_rpnum[batchsize] = {};
        uint32_t _local_buf_rnnum[batchsize] = {};
        uint32_t **_local_buf = local_buf + thread * batchsize;
        for (long i = 0; i < batchsize; i++) _local_buf_size[i] = init_buffer_size;

        sol_list_amx_t *sol = sol_list[thread];
        uint64_t try_add2 = 0, succ_add2 = 0;

        const long nblocks = (num_vec + 15) / 16;
        const long begin_ind = (thread * nblocks / num_threads) * 16;
        const long end_ind = ((thread + 1) * nblocks / num_threads) * 16 > num_vec ? num_vec : ((thread + 1) * nblocks / num_threads) * 16;

        // real ind is __ind + ind, real __cind is __cind16 + r
        #define CHECK_MSK16_AND_PACK(__msk, __ind, __cind16)                                                    \
                                                            do {                                                \
            while (__msk) {                                                                                     \
                int r = __builtin_ctz(__msk);                                                                   \
                __msk &= ~(1 << r);                                                                             \
                _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r]] = ind + (__ind);                 \
                uint32_t nhint = (0x80000000 & dst[((__ind) + (__cind16)) * 16 + r]);                           \
                int norm_with_hint = norm_epi32[__ind] | nhint;                                                 \
                _local_buf_nnum[(__cind16) + r] += nhint >> 31;                                                 \
                _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r] + 1] = norm_with_hint;            \
                if (reuse) {                                                                                    \
                    if (abs(dst[((__ind) + (__cind16)) * 16 + r]) > rbound[__ind]) {                            \
                        _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r] + 1] |= 0x40000000;       \
                        if (nhint) {                                                                            \
                            _local_buf_rnnum[(__cind16) + r]++;                                                 \
                        } else {                                                                                \
                            _local_buf_rpnum[(__cind16) + r]++;                                                 \
                        }                                                                                       \
                        if (abs(dst[((__ind) + (__cind16)) * 16 + r]) > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                            if (dst[((__ind) + (__cind16)) * 16 + r] > 0) {                                     \
                                if (center_ind_list[(__cind16) + r] != ind + (__ind)) {                         \
                                    TRY_ADDS_AMX(center_ind_list[(__cind16) + r], ind + (__ind));               \
                                } else {                                                                        \
                                    _local_buf_num[(__cind16) + r]--;                                           \
                                    _local_buf_rpnum[(__cind16) + r]--;                                         \
                                }                                                                               \
                            } else {                                                                            \
                                TRY_ADDA_AMX(center_ind_list[(__cind16) + r], ind + (__ind));                   \
                            }                                                                                   \
                        }                                                                                       \
                    }                                                                                           \
                } else {                                                                                        \
                    if (abs(dst[((__ind) + (__cind16)) * 16 + r]) > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                        if (dst[((__ind) + (__cind16)) * 16 + r] > 0) {                                         \
                            if (center_ind_list[(__cind16) + r] != ind + (__ind)) {                             \
                                TRY_ADDS_AMX(center_ind_list[(__cind16) + r], ind + (__ind));                   \
                            } else _local_buf_num[(__cind16) + r]--;                                            \
                        } else {                                                                                \
                            TRY_ADDA_AMX(center_ind_list[(__cind16) + r], ind + (__ind));                       \
                        }                                                                                       \
                    }                                                                                           \
                }                                                                                               \
                _local_buf_num[(__cind16) + r]++;                                                               \
            }                                                                                                   \
        } while (0)

        #define CHECK_MSK16_PAIR_AND_PACK(__msk0, __msk1, __ind0, __ind1, __cind16)             \
                                                            do {                                \
            if (!_mm512_kortestz(__msk0, __msk1)) {                                             \
                CHECK_MSK16_AND_PACK(__msk0, __ind0, __cind16);                                 \
                CHECK_MSK16_AND_PACK(__msk1, __ind1, __cind16);                                 \
            }                                                                                   \
        } while (0)

        long ind = begin_ind;
        while (ind < end_ind - 127) {
            int32_t *norm_epi32 = vnorm + ind;
            __attribute__ ((aligned (64))) int32_t _bound[64];
            __attribute__ ((aligned (64))) int32_t _rbound[64];
            __m512i norm_si512_0 = _mm512_loadu_si512(norm_epi32 + 0);
            __m512i norm_si512_1 = _mm512_loadu_si512(norm_epi32 + 16);
            __m512i norm_si512_2 = _mm512_loadu_si512(norm_epi32 + 32);
            __m512i norm_si512_3 = _mm512_loadu_si512(norm_epi32 + 48);
            _mm512_store_si512(_bound + 0, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_b0x2_ps)));
            _mm512_store_si512(_bound + 16, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_b0x2_ps)));
            _mm512_store_si512(_bound + 32, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_b0x2_ps)));
            _mm512_store_si512(_bound + 48, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_b0x2_ps)));
            if (reuse) {
                _mm512_store_si512(_rbound + 0, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_r0x2_ps)));
                _mm512_store_si512(_rbound + 16, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_r0x2_ps)));
                _mm512_store_si512(_rbound + 32, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_r0x2_ps)));
                _mm512_store_si512(_rbound + 48, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_r0x2_ps)));
            }
            __attribute__ ((aligned (64))) int32_t _dst[64 * batchsize];
            int32_t *dst = _dst;
            int32_t *bound = _bound;
            int32_t *rbound = _rbound;

            int8_t *vec0 = vec + (ind+0) * vec_length;
            int8_t *vec1 = vec + (ind+16) * vec_length;
            int8_t *vec2 = vec + (ind+32) * vec_length;
            int8_t *vec3 = vec + (ind+48) * vec_length;
            int8_t *prefetch_start = vec + (ind+64) * vec_length;
            constexpr int32_t n_prefetch_line = vec_length;

            for (long i = 0; i < batchsize; i += 16) {
                for (long j = 0; j < 16 && i <= batchsize - 32; j++) {
                    _mm_prefetch(center0_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center1_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center2_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                }
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
                _tile_loadd(4, center0_tr + i * 64, 64);
                _tile_loadd(5, vec0, vec_length);
                _tile_loadd(6, vec1, vec_length);
                _tile_loadd(7, vec2, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3, vec_length);
                _tile_dpbssd(1, 6, 4);
                _tile_loadd(6, center1_tr + i * 64, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_loadd(7, vec0 + 64, vec_length);
                _tile_dpbssd(3, 5, 4);
                _tile_loadd(4, vec1 + 64, vec_length);
                _tile_loadd(5, vec2 + 64, vec_length);
                _tile_dpbssd(0, 7, 6);
                _tile_loadd(7, vec3 + 64, vec_length);
                _tile_dpbssd(1, 4, 6);
                _tile_loadd(4, center2_tr + i * 64, 64);
                _tile_dpbssd(2, 5, 6);
                _tile_loadd(5, vec0 + 128, vec_length);
                _tile_dpbssd(3, 7, 6);
                _tile_loadd(6, vec1 + 128, vec_length);
                _tile_loadd(7, vec2 + 128, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3 + 128, vec_length);
                _tile_stored(0, dst + 0 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(1, 6, 4);
                _tile_stored(1, dst + 1 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_stored(2, dst + 2 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(3, 5, 4);
                _tile_stored(3, dst + 3 * 16 * batchsize + i * 16, 64);
                for (long j = (n_prefetch_line * i) / batchsize; j < (n_prefetch_line * (i + 16)) / batchsize; j++) {
                    _mm_prefetch(prefetch_start + j * 64, _MM_HINT_T1);
                }
            }

            for (long kk = 0; kk < 4; kk++) {
                __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
                __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
                __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
                __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
                __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
                __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
                __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
                __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
                __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
                __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
                __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
                __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
                __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
                __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
                __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
                __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

                for (long i = 0; i < batchsize; i += 16) {
                    __mmask16 msk0 = _mm512_cmp_epi32_mask(bound0_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 0) * 16)), 1);
                    __mmask16 msk1 = _mm512_cmp_epi32_mask(bound1_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 1) * 16)), 1);
                    __mmask16 msk2 = _mm512_cmp_epi32_mask(bound2_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 2) * 16)), 1);
                    __mmask16 msk3 = _mm512_cmp_epi32_mask(bound3_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 3) * 16)), 1);
                    __mmask16 msk4 = _mm512_cmp_epi32_mask(bound4_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 4) * 16)), 1);
                    __mmask16 msk5 = _mm512_cmp_epi32_mask(bound5_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 5) * 16)), 1);
                    __mmask16 msk6 = _mm512_cmp_epi32_mask(bound6_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 6) * 16)), 1);
                    __mmask16 msk7 = _mm512_cmp_epi32_mask(bound7_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 7) * 16)), 1);
                    __mmask16 msk8 = _mm512_cmp_epi32_mask(bound8_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 8) * 16)), 1);
                    __mmask16 msk9 = _mm512_cmp_epi32_mask(bound9_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 9) * 16)), 1);
                    __mmask16 mskA = _mm512_cmp_epi32_mask(boundA_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 10) * 16)), 1);
                    __mmask16 mskB = _mm512_cmp_epi32_mask(boundB_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 11) * 16)), 1);
                    __mmask16 mskC = _mm512_cmp_epi32_mask(boundC_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 12) * 16)), 1);
                    __mmask16 mskD = _mm512_cmp_epi32_mask(boundD_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 13) * 16)), 1);
                    __mmask16 mskE = _mm512_cmp_epi32_mask(boundE_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 14) * 16)), 1);
                    __mmask16 mskF = _mm512_cmp_epi32_mask(boundF_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 15) * 16)), 1);

                    CHECK_MSK16_PAIR_AND_PACK(msk0, msk1, 0, 1, i);
                    CHECK_MSK16_PAIR_AND_PACK(msk2, msk3, 2, 3, i);
                    CHECK_MSK16_PAIR_AND_PACK(msk4, msk5, 4, 5, i);
                    CHECK_MSK16_PAIR_AND_PACK(msk6, msk7, 6, 7, i);
                    CHECK_MSK16_PAIR_AND_PACK(msk8, msk9, 8, 9, i);
                    CHECK_MSK16_PAIR_AND_PACK(mskA, mskB, 10, 11, i);
                    CHECK_MSK16_PAIR_AND_PACK(mskC, mskD, 12, 13, i);
                    CHECK_MSK16_PAIR_AND_PACK(mskE, mskF, 14, 15, i);
                }

                ind += 16;
                dst += 16 * batchsize;
                norm_epi32 += 16;
                bound += 16;
                if (reuse) rbound += 16;
            }
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 64) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 64 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }
        }
        while (ind < end_ind - 15) {
            int32_t *norm_epi32 = vnorm + ind;
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_b0x2_ps)));
            if (reuse) _mm512_store_si512(rbound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_r0x2_ps)));

            __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
            __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
            __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
            __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
            __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
            __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
            __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
            __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
            __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
            __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
            __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
            __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
            __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
            __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
            __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
            __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];

            // TODO: rewrite/optimize the amx part
            _tile_loadd(2, vec + ind * vec_length, vec_length);
            _tile_loadd(3, vec + ind * vec_length + 64, vec_length);
            _tile_loadd(4, vec + ind * vec_length + 128, vec_length);

            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            #if 1
            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk0 = _mm512_cmp_epi32_mask(bound0_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 0) * 16)), 1);
                __mmask16 msk1 = _mm512_cmp_epi32_mask(bound1_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 1) * 16)), 1);
                __mmask16 msk2 = _mm512_cmp_epi32_mask(bound2_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 2) * 16)), 1);
                __mmask16 msk3 = _mm512_cmp_epi32_mask(bound3_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 3) * 16)), 1);
                __mmask16 msk4 = _mm512_cmp_epi32_mask(bound4_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 4) * 16)), 1);
                __mmask16 msk5 = _mm512_cmp_epi32_mask(bound5_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 5) * 16)), 1);
                __mmask16 msk6 = _mm512_cmp_epi32_mask(bound6_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 6) * 16)), 1);
                __mmask16 msk7 = _mm512_cmp_epi32_mask(bound7_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 7) * 16)), 1);
                __mmask16 msk8 = _mm512_cmp_epi32_mask(bound8_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 8) * 16)), 1);
                __mmask16 msk9 = _mm512_cmp_epi32_mask(bound9_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 9) * 16)), 1);
                __mmask16 mskA = _mm512_cmp_epi32_mask(boundA_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 10) * 16)), 1);
                __mmask16 mskB = _mm512_cmp_epi32_mask(boundB_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 11) * 16)), 1);
                __mmask16 mskC = _mm512_cmp_epi32_mask(boundC_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 12) * 16)), 1);
                __mmask16 mskD = _mm512_cmp_epi32_mask(boundD_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 13) * 16)), 1);
                __mmask16 mskE = _mm512_cmp_epi32_mask(boundE_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 14) * 16)), 1);
                __mmask16 mskF = _mm512_cmp_epi32_mask(boundF_si512, _mm512_abs_epi32(_mm512_load_si512(dst + (i + 15) * 16)), 1);

                CHECK_MSK16_PAIR_AND_PACK(msk0, msk1, 0, 1, i);
                CHECK_MSK16_PAIR_AND_PACK(msk2, msk3, 2, 3, i);
                CHECK_MSK16_PAIR_AND_PACK(msk4, msk5, 4, 5, i);
                CHECK_MSK16_PAIR_AND_PACK(msk6, msk7, 6, 7, i);
                CHECK_MSK16_PAIR_AND_PACK(msk8, msk9, 8, 9, i);
                CHECK_MSK16_PAIR_AND_PACK(mskA, mskB, 10, 11, i);
                CHECK_MSK16_PAIR_AND_PACK(mskC, mskD, 12, 13, i);
                CHECK_MSK16_PAIR_AND_PACK(mskE, mskF, 14, 15, i);
            }
            #endif
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 16) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 16 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }

            ind += 16;
        }
        if (ind < end_ind) {
            __attribute__ ((aligned (64))) int32_t norm_epi32[16];
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            for (long i = ind; i < end_ind; i++) norm_epi32[i - ind] = vnorm[i];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_b0x2_ps)));
            if (reuse) _mm512_store_si512(rbound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_r0x2_ps)));

            __m512i bound_si512[16];
            for (long i = 0; i < end_ind - ind; i++) bound_si512[i] = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + i)));
            
            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];
            __attribute__ ((aligned (64))) int8_t tmp[1024 * 3 + 32];
            for (long i = 0; i < (end_ind - ind) * vec_length; i += 64) {
                _mm512_stream_si512((__m512i *)(tmp + i), _mm512_stream_load_si512(vec + ind * vec_length + i));
            }

            _tile_loadd(2, tmp, vec_length);
            _tile_loadd(3, tmp + 64, vec_length);
            _tile_loadd(4, tmp + 128, vec_length);
            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk[16];
                for (long j = 0; j < end_ind - ind; j++) {
                    msk[j] = _mm512_cmp_epi32_mask(bound_si512[j], _mm512_abs_epi32(_mm512_load_si512(dst + (i + j) * 16)), 1);
                }
                for (long j = 0; j < end_ind - ind; j++) {
                    CHECK_MSK16_AND_PACK(msk[j], j, i);
                }
            }
        }

        try_add2_stat[thread] = try_add2;
        succ_add2_stat[thread] = succ_add2;
        for (long i = 0; i < batchsize; i++) {
            local_buf_num[thread * batchsize + i] = _local_buf_num[i];
            local_buf_pnum[thread * batchsize + i] = _local_buf_num[i] - _local_buf_nnum[i];
            local_buf_nnum[thread * batchsize + i] = _local_buf_nnum[i];
            if (reuse) local_buf_rpnum[thread * batchsize + i] = _local_buf_rpnum[i];
            if (reuse) local_buf_rnnum[thread * batchsize + i] = _local_buf_rnnum[i];
        }
    }

    /////// combine the data to main buckets ///////
    #pragma omp parallel for
    for (long i = 0; i < batchsize; i++) {
        long total_pnum = 0;
        long total_nnum = 0;
        long total_rpnum = 0;
        long total_rnnum = 0;
        for (long j = 0; j < num_threads; j++) {
            total_pnum += local_buf_pnum[j * batchsize + i];
            total_nnum += local_buf_nnum[j * batchsize + i];
            if (reuse) {
                total_rpnum += local_buf_rpnum[j * batchsize + i];
                total_rnnum += local_buf_rnnum[j * batchsize + i];
            }
        }
        bucket0[i]->_alloc(total_pnum, 1);
        bucket0[i]->_alloc(total_nnum, 0);
        if (reuse) {
            rbucket0[i]->_alloc(total_rpnum, 1);
            rbucket0[i]->_alloc(total_rnnum, 0);
        }
        bucket0[i]->num_pvec = total_pnum;
        bucket0[i]->num_nvec = total_nnum;
        if (reuse) {
            rbucket0[i]->num_pvec = total_rpnum;
            rbucket0[i]->num_nvec = total_rnnum;
        }
    }

    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        uint32_t **_local_buf = local_buf + thread * batchsize;
        uint32_t *_local_buf_num = local_buf_num + thread * batchsize;
        for (long j = 0; j < batchsize; j++) {
            bucket_amx_t *dst, *rdst;
            long begin_pind = 0, begin_nind = 0, begin_rpind = 0, begin_rnind = 0;
            dst = bucket0[j];
            if (reuse) rdst = rbucket0[j];
            for (long i = 0; i < thread; i++) {
                begin_pind += local_buf_pnum[i * batchsize + j];
                begin_nind += local_buf_nnum[i * batchsize + j];
                if (reuse) {
                    begin_rpind += local_buf_rpnum[i * batchsize + j];
                    begin_rnind += local_buf_rnnum[i * batchsize + j];
                }
            }
            uint32_t *pind_dst = dst->pvec + begin_pind;
            uint32_t *nind_dst = dst->nvec + begin_nind;
            int32_t *pnorm_dst = dst->pnorm + begin_pind;
            int32_t *nnorm_dst = dst->nnorm + begin_nind;
            uint32_t *rpind_dst, *rnind_dst;
            int32_t *rpnorm_dst, *rnnorm_dst;
            if (reuse) {
                rpind_dst = rdst->pvec + begin_rpind;
                rnind_dst = rdst->nvec + begin_rnind;
                rpnorm_dst = rdst->pnorm + begin_rpind;
                rnnorm_dst = rdst->nnorm + begin_rnind;
            } 

            uint32_t *src = _local_buf[j];
            long src_num = _local_buf_num[j];
            
            // ASSUME THE OUTPUT OF MALLOC AND REALLOC IS 8-ALIGNED
            while ((long) src % 64L != 0 && src_num) {
                uint32_t norm_with_hint = src[1];
                int32_t norm = norm_with_hint & 0x3fffffff;
                if (norm_with_hint & 0x80000000) {
                    nind_dst[0] = src[0];
                    nnorm_dst[0] = norm;
                    nind_dst++;
                    nnorm_dst++;
                    if (reuse && norm_with_hint & 0x40000000) {
                        rnind_dst[0] = src[0];
                        rnnorm_dst[0] = norm;
                        rnind_dst++;
                        rnnorm_dst++;
                    }
                } else {
                    pind_dst[0] = src[0];
                    pnorm_dst[0] = norm;
                    pind_dst++;
                    pnorm_dst++;
                    if (reuse && norm_with_hint & 0x40000000) {
                        rpind_dst[0] = src[0];
                        rpnorm_dst[0] = norm;
                        rpind_dst++;
                        rpnorm_dst++;
                    }
                }
                src_num--;
                src += 2;
            }

            long ind = 0;
            while (ind < src_num - 255) {
                #define UNPACK_AND_ADD_TO_BUCKETS(__ind)   \
                                                    do {   \
                    __m512i src0 = _mm512_stream_load_si512(src + (__ind) * 2);   \
                    __m512i src1 = _mm512_stream_load_si512(src + (__ind) * 2 + 16);  \
                    __m512i ind_b16 = _mm512_mask_blend_epi32(0xaaaa, src0, _mm512_slli_epi64(src1, 32));    \
                    __m512i hnorm_b16 = _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64(src0, 32), src1);  \
                    __m512i norm_b16 = _mm512_and_si512(hnorm_b16, _mm512_set1_epi32(0x3fffffff)); \
                    __mmask16 pmsk = _mm512_cmp_epu32_mask(hnorm_b16, _mm512_set1_epi32(0x80000000), 1);    \
                    __mmask16 nmsk = ~pmsk; \
                    __mmask16 rmsk = _mm512_cmp_epu32_mask(_mm512_and_si512(hnorm_b16, _mm512_set1_epi32(0x40000000)), _mm512_set1_epi32(0x40000000), 0);   \
                    _mm512_mask_compressstoreu_epi32(pind_dst, pmsk, ind_b16); \
                    _mm512_mask_compressstoreu_epi32(nind_dst, nmsk, ind_b16);    \
                    _mm512_mask_compressstoreu_epi32(pnorm_dst, pmsk, norm_b16);   \
                    _mm512_mask_compressstoreu_epi32(nnorm_dst, nmsk, norm_b16);  \
                    pind_dst += __builtin_popcount(pmsk);  \
                    pnorm_dst += __builtin_popcount(pmsk); \
                    nind_dst += __builtin_popcount(nmsk);  \
                    nnorm_dst += __builtin_popcount(nmsk); \
                    if (reuse && rmsk) { \
                        if (rmsk & pmsk) _mm512_mask_compressstoreu_epi32(rpind_dst, rmsk & pmsk, ind_b16);   \
                        if (rmsk & nmsk) _mm512_mask_compressstoreu_epi32(rnind_dst, rmsk & nmsk, ind_b16); \
                        _mm512_mask_compressstoreu_epi32(rpnorm_dst, rmsk & pmsk, norm_b16);  \
                        _mm512_mask_compressstoreu_epi32(rnnorm_dst, rmsk & nmsk, norm_b16); \
                        rpind_dst += __builtin_popcount(rmsk & pmsk);  \
                        rpnorm_dst += __builtin_popcount(rmsk & pmsk); \
                        rnind_dst += __builtin_popcount(rmsk & nmsk);  \
                        rnnorm_dst += __builtin_popcount(rmsk & nmsk); \
                    }   \
                } while (0)
                
                UNPACK_AND_ADD_TO_BUCKETS(ind + 0);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 16);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 32);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 48);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 64);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 80);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 96);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 112);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 128);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 144);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 160);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 176);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 192);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 208);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 224);
                UNPACK_AND_ADD_TO_BUCKETS(ind + 240);
                ind += 256;
            }
            while (ind < src_num - 15) {
                UNPACK_AND_ADD_TO_BUCKETS(ind);
                ind += 16;
            }
            while (ind < src_num) {
                uint32_t norm_with_hint = src[ind * 2 + 1];
                int32_t norm = norm_with_hint & 0x3fffffff;
                if (norm_with_hint & 0x80000000) {
                    nind_dst[0] = src[ind * 2];
                    nnorm_dst[0] = norm;
                    nind_dst++;
                    nnorm_dst++;
                    if (reuse && norm_with_hint & 0x40000000) {
                        rnind_dst[0] = src[ind * 2];
                        rnnorm_dst[0] = norm;
                        rnind_dst++;
                        rnnorm_dst++;
                    }
                } else {
                    pind_dst[0] = src[ind * 2];
                    pnorm_dst[0] = norm;
                    pind_dst++;
                    pnorm_dst++;
                    if (reuse && norm_with_hint & 0x40000000) {
                        rpind_dst[0] = src[ind * 2];
                        rpnorm_dst[0] = norm;
                        rpind_dst++;
                        rpnorm_dst++;
                    }
                }
                ind++;
            }
            _local_buf_num[j] = 0;
        }
    }

    /////// free local buffer and center ///////
    for (long i = 0; i < batchsize * num_threads; i++) {
        free(local_buf[i]);
    }
    FREE_VEC((void *)center0_tr);
    FREE_VEC((void *)center1_tr);
    FREE_VEC((void *)center2_tr);
    
    if (prof) {
        for (long i = 1; i < num_threads; i++) {
            succ_add2_stat[0] += succ_add2_stat[i];
            try_add2_stat[0] += try_add2_stat[i];
        }
        // pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2_stat[0];
        prof->succ_add2 += succ_add2_stat[0];
        // pthread_spin_unlock(&prof->profile_lock);
    }
    return 0;
}

template <uint32_t nb>
template <uint32_t batchsize, bool faraway_center, bool reuse>
int Pool_epi8_t<nb>::_parallel_sub_bucketing_amx(bucket_amx_t *main_bucket, bucket_amx_t **rbucket, bucket_amx_t **bucket, 
            double alpha_r, double alpha_b, sol_list_amx_t **sol_list, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof) {
    /////// prepare dst and local buffer ///////
    for (long i = 0; i < batchsize; i++) {
        if (reuse) {
            if (rbucket[i]) {
                if (rbucket[i]->num_pvec || rbucket[i]->num_nvec) {
                    fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_parallel_sub_bucketing_amx: nonempty input buckets, ignored.\n", nb);
                    rbucket[i]->num_pvec = 0;
                    rbucket[i]->num_nvec = 0;
                }
            } else {
                rbucket[i] = new bucket_amx_t;
            }
            rbucket[i]->_alloc(256, 1);
            rbucket[i]->_alloc(256, 0);
        }
        if (bucket[i]) {
            if (bucket[i]->num_pvec || bucket[i]->num_nvec) {
                fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_parallel_sub_bucketing_amx: nonempty input buckets, ignored.\n", nb);
                bucket[i]->num_pvec = 0;
                bucket[i]->num_nvec = 0;
            } 
        } else {
            bucket[i] = new bucket_amx_t;
        }
        bucket[i]->_alloc(256, 1);
        bucket[i]->_alloc(256, 0);
    }
    const __m512 alpha_rx2_ps = _mm512_set1_ps(alpha_r * 2.0);
    const __m512 alpha_bx2_ps = _mm512_set1_ps(alpha_b * 2.0);
    const long expect_bucket_size = pow(1.0 - alpha_b * alpha_b, CSD * 0.5) * (main_bucket->num_pvec + main_bucket->num_nvec);
    const long init_buffer_size = 0.5 * expect_bucket_size / num_threads + batchsize * 64;
    uint32_t *local_buf[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_num[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_size[batchsize * AMX_MAX_NTHREADS];
    uint32_t local_buf_rnum[batchsize * AMX_MAX_NTHREADS];
    for (long i = 0; i < batchsize * num_threads; i++) {
        local_buf[i] = (uint32_t *) malloc(init_buffer_size * 2 * sizeof(uint32_t));
        local_buf_size[i] = init_buffer_size;
    }
    uint64_t try_add2_stat[AMX_MAX_NTHREADS] = {};
    uint64_t succ_add2_stat[AMX_MAX_NTHREADS] = {};

    /////// choose centers ///////
    uint32_t center_ind_list[batchsize];
    int32_t center_norm[batchsize];
    int8_t *center0_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    int8_t *center1_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    int8_t *center2_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    long num_try_find_center = 0;
    for (long i = 0; i < batchsize; i++) {
        int pass;
        do {
            pass = 1;
            center_ind_list[i] = main_bucket->pvec[Uniform_long(main_bucket->num_pvec)];
            if (num_try_find_center < AMX_MAX_NUM_TRYCENTER) {
                for (long j = 0; j < i; j++){
                    if (center_ind_list[j] == center_ind_list[i]) pass = 0;
                }
            }
            int8_t *ptr = vec + center_ind_list[i] * vec_length;
            num_try_find_center++;
            if (faraway_center) {
                if ((CSD * (int)ptr[0] * (int)ptr[0] > 2 * vnorm[center_ind_list[i]]) && (num_try_find_center < (long)(0.8 * AMX_MAX_NUM_TRYCENTER))) pass = 0;
            }
            if (reuse) rbucket[i]->center_ind = center_ind_list[i];
            bucket[i]->center_ind = center_ind_list[i];
            center_norm[i] = vnorm[center_ind_list[i]] - goal_norm;
        } while(!pass);
    }
    if (num_try_find_center >= (long)(0.8 * AMX_MAX_NUM_TRYCENTER)) {
        if (CSD >= 108) fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_parallel_sub_bucketing_amx: fail to use faraway centers after %ld tries.\n", nb, num_try_find_center);
    }
    for (long i = 0; i < batchsize; i += 16) {
        __m512i z[16];
        __m256i y[16];
        for (long j = 0; j < 16; j++) {
            z[j] = _mm512_loadu_si512((__m512i *)(vec + center_ind_list[i+j] * vec_length));
        }
        AVX512_MATTR_16x16(NULL, center0_tr + i * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
        for (long j = 0; j < 16; j++) {
            z[j] = _mm512_loadu_si512((__m512i *)(vec + center_ind_list[i+j] * vec_length + 64));
        }
        AVX512_MATTR_16x16(NULL, center1_tr + i * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
        for (long j = 0; j < 16; j++) {
            y[j] = _mm256_load_si256((__m256i *)(vec + center_ind_list[i+j] * vec_length + 128));
        }
        AVX512_MATTR_16x8(NULL, center2_tr + i * 64, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]);
    }

    /////// PPART: each thread collect vectors in local buffers ///////
    #pragma omp parallel for 
    for (long thread = 0; thread < num_threads; thread++) {
        ///// tile configuration /////
        TILE_INITIALIZE;

        ///// prepare local buffers //////
        uint32_t _local_buf_size[batchsize];
        uint32_t _local_buf_num[batchsize] = {};
        uint32_t _local_buf_rnum[batchsize] = {};
        uint32_t **_local_buf = local_buf + thread * batchsize;
        for (long i = 0; i < batchsize; i++) _local_buf_size[i] = local_buf_size[i + thread * batchsize];

        sol_list_amx_t *sol = sol_list[thread];
        uint64_t try_add2 = 0, succ_add2 = 0;

        const long nblocks = (main_bucket->num_pvec + 15) / 16;
        const long begin_ind = (thread * nblocks / num_threads) * 16;
        const long end_ind = ((thread + 1) * nblocks / num_threads) * 16 > main_bucket->num_pvec ? main_bucket->num_pvec : ((thread + 1) * nblocks / num_threads) * 16;

        // real ind is __ind + ind, real __cind is __cind16 + r
        #define CHECK_MSK16_AND_PACK_PPART(__msk, __ind, __cind16)                                              \
                                                            do {                                                \
            while (__msk) {                                                                                     \
                int r = __builtin_ctz(__msk);                                                                   \
                __msk &= ~(1 << r);                                                                             \
                _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r]] = main_bucket->pvec[ind+(__ind)];\
                int norm_with_hint = norm_epi32[__ind];                                                         \
                _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r] + 1] = norm_with_hint;            \
                if (reuse) {                                                                                    \
                    if (dst[((__ind) + (__cind16)) * 16 + r] > rbound[__ind]) {                                 \
                        _local_buf_rnum[(__cind16) + r]++;                                                      \
                        _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r] + 1] |= 0x80000000;       \
                        if (dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                            if (center_ind_list[(__cind16) + r] != main_bucket->pvec[ind + (__ind)]) {          \
                                TRY_ADDS_AMX(center_ind_list[(__cind16) + r], main_bucket->pvec[ind + (__ind)]);\
                            } else {                                                                            \
                                _local_buf_num[(__cind16) + r]--;                                               \
                                _local_buf_rnum[(__cind16) + r]--;                                              \
                            }                                                                                   \
                        }                                                                                       \
                    }                                                                                           \
                } else {                                                                                        \
                    if (dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                        if (center_ind_list[(__cind16) + r] != main_bucket->pvec[ind + (__ind)]) {              \
                            TRY_ADDS_AMX(center_ind_list[(__cind16) + r], main_bucket->pvec[ind + (__ind)]);    \
                        } else _local_buf_num[(__cind16) + r]--;                                                \
                    }                                                                                           \
                }                                                                                               \
                _local_buf_num[(__cind16) + r]++;                                                               \
            }                                                                                                   \
        } while (0)

        #define CHECK_MSK16_PAIR_AND_PACK_PPART(__msk0, __msk1, __ind0, __ind1, __cind16)             \
                                                            do {                                \
            if (!_mm512_kortestz(__msk0, __msk1)) {                                             \
                CHECK_MSK16_AND_PACK_PPART(__msk0, __ind0, __cind16);                                 \
                CHECK_MSK16_AND_PACK_PPART(__msk1, __ind1, __cind16);                                 \
            }                                                                                   \
        } while (0)

        long ind = begin_ind;
        while (ind < end_ind - 127) {
            int32_t *norm_epi32 = main_bucket->pnorm + ind;
            __attribute__ ((aligned (64))) int32_t _bound[64];
            __attribute__ ((aligned (64))) int32_t _rbound[64];
            __m512i norm_si512_0 = _mm512_loadu_si512(norm_epi32 + 0);
            __m512i norm_si512_1 = _mm512_loadu_si512(norm_epi32 + 16);
            __m512i norm_si512_2 = _mm512_loadu_si512(norm_epi32 + 32);
            __m512i norm_si512_3 = _mm512_loadu_si512(norm_epi32 + 48);
            _mm512_store_si512(_bound + 0, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_bx2_ps)));
            _mm512_store_si512(_bound + 16, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_bx2_ps)));
            _mm512_store_si512(_bound + 32, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_bx2_ps)));
            _mm512_store_si512(_bound + 48, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_bx2_ps)));
            if (reuse) {
                _mm512_store_si512(_rbound + 0, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_rx2_ps)));
                _mm512_store_si512(_rbound + 16, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_rx2_ps)));
                _mm512_store_si512(_rbound + 32, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_rx2_ps)));
                _mm512_store_si512(_rbound + 48, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_rx2_ps)));
            }
            __attribute__ ((aligned (64))) int32_t _dst[64 * batchsize];
            int32_t *dst = _dst;
            int32_t *bound = _bound;
            int32_t *rbound = _rbound;

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec1[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec2[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec3[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 0 + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 0 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 0 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec1 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 16 + j]) * vec_length)));
                _mm512_storeu_si512(vec1 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 16 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec1 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 16 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec2 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 32 + j]) * vec_length)));
                _mm512_storeu_si512(vec2 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 32 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec2 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 32 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec3 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 48 + j]) * vec_length)));
                _mm512_storeu_si512(vec3 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 48 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec3 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 48 + j]) * vec_length + 128)));
            }

            
            long prefetch_start_ind = (ind+64);
            constexpr int32_t n_prefetch_vec = 64;

            for (long i = 0; i < batchsize; i += 16) {
                for (long j = 0; j < 16 && i <= batchsize - 32; j++) {
                    _mm_prefetch(center0_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center1_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center2_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                }
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
                _tile_loadd(4, center0_tr + i * 64, 64);
                _tile_loadd(5, vec0, vec_length);
                _tile_loadd(6, vec1, vec_length);
                _tile_loadd(7, vec2, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3, vec_length);
                _tile_dpbssd(1, 6, 4);
                _tile_loadd(6, center1_tr + i * 64, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_loadd(7, vec0 + 64, vec_length);
                _tile_dpbssd(3, 5, 4);
                _tile_loadd(4, vec1 + 64, vec_length);
                _tile_loadd(5, vec2 + 64, vec_length);
                _tile_dpbssd(0, 7, 6);
                _tile_loadd(7, vec3 + 64, vec_length);
                _tile_dpbssd(1, 4, 6);
                _tile_loadd(4, center2_tr + i * 64, 64);
                _tile_dpbssd(2, 5, 6);
                _tile_loadd(5, vec0 + 128, vec_length);
                _tile_dpbssd(3, 7, 6);
                _tile_loadd(6, vec1 + 128, vec_length);
                _tile_loadd(7, vec2 + 128, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3 + 128, vec_length);
                _tile_stored(0, dst + 0 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(1, 6, 4);
                _tile_stored(1, dst + 1 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_stored(2, dst + 2 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(3, 5, 4);
                _tile_stored(3, dst + 3 * 16 * batchsize + i * 16, 64);
                for (long j = (n_prefetch_vec * i) / batchsize; j < (n_prefetch_vec * (i + 16)) / batchsize; j++) {
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->pvec[prefetch_start_ind + j], _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->pvec[prefetch_start_ind + j] + 64, _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->pvec[prefetch_start_ind + j] + 128, _MM_HINT_T1);
                }
            }

            for (long kk = 0; kk < 4; kk++) {
                __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
                __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
                __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
                __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
                __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
                __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
                __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
                __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
                __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
                __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
                __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
                __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
                __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
                __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
                __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
                __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

                for (long i = 0; i < batchsize; i += 16) {
                    __mmask16 msk0 = _mm512_cmp_epi32_mask(bound0_si512, _mm512_load_si512(dst + (i + 0) * 16), 1);
                    __mmask16 msk1 = _mm512_cmp_epi32_mask(bound1_si512, _mm512_load_si512(dst + (i + 1) * 16), 1);
                    __mmask16 msk2 = _mm512_cmp_epi32_mask(bound2_si512, _mm512_load_si512(dst + (i + 2) * 16), 1);
                    __mmask16 msk3 = _mm512_cmp_epi32_mask(bound3_si512, _mm512_load_si512(dst + (i + 3) * 16), 1);
                    __mmask16 msk4 = _mm512_cmp_epi32_mask(bound4_si512, _mm512_load_si512(dst + (i + 4) * 16), 1);
                    __mmask16 msk5 = _mm512_cmp_epi32_mask(bound5_si512, _mm512_load_si512(dst + (i + 5) * 16), 1);
                    __mmask16 msk6 = _mm512_cmp_epi32_mask(bound6_si512, _mm512_load_si512(dst + (i + 6) * 16), 1);
                    __mmask16 msk7 = _mm512_cmp_epi32_mask(bound7_si512, _mm512_load_si512(dst + (i + 7) * 16), 1);
                    __mmask16 msk8 = _mm512_cmp_epi32_mask(bound8_si512, _mm512_load_si512(dst + (i + 8) * 16), 1);
                    __mmask16 msk9 = _mm512_cmp_epi32_mask(bound9_si512, _mm512_load_si512(dst + (i + 9) * 16), 1);
                    __mmask16 mskA = _mm512_cmp_epi32_mask(boundA_si512, _mm512_load_si512(dst + (i + 10) * 16), 1);
                    __mmask16 mskB = _mm512_cmp_epi32_mask(boundB_si512, _mm512_load_si512(dst + (i + 11) * 16), 1);
                    __mmask16 mskC = _mm512_cmp_epi32_mask(boundC_si512, _mm512_load_si512(dst + (i + 12) * 16), 1);
                    __mmask16 mskD = _mm512_cmp_epi32_mask(boundD_si512, _mm512_load_si512(dst + (i + 13) * 16), 1);
                    __mmask16 mskE = _mm512_cmp_epi32_mask(boundE_si512, _mm512_load_si512(dst + (i + 14) * 16), 1);
                    __mmask16 mskF = _mm512_cmp_epi32_mask(boundF_si512, _mm512_load_si512(dst + (i + 15) * 16), 1);

                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk0, msk1, 0, 1, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk2, msk3, 2, 3, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk4, msk5, 4, 5, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk6, msk7, 6, 7, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk8, msk9, 8, 9, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(mskA, mskB, 10, 11, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(mskC, mskD, 12, 13, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(mskE, mskF, 14, 15, i);
                }

                ind += 16;
                dst += 16 * batchsize;
                norm_epi32 += 16;
                bound += 16;
                if (reuse) rbound += 16;
            }
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 64) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 64 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }
        }
        while (ind < end_ind - 15) {
            int32_t *norm_epi32 = main_bucket->pnorm + ind;
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps)));
            if (reuse) _mm512_store_si512(rbound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps)));

            __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
            __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
            __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
            __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
            __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
            __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
            __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
            __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
            __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
            __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
            __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
            __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
            __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
            __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
            __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
            __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + j]) * vec_length + 128)));
            }

            _tile_loadd(2, vec0, vec_length);
            _tile_loadd(3, vec0 + 64, vec_length);
            _tile_loadd(4, vec0 + 128, vec_length);

            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            
            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk0 = _mm512_cmp_epi32_mask(bound0_si512, _mm512_load_si512(dst + (i + 0) * 16), 1);
                __mmask16 msk1 = _mm512_cmp_epi32_mask(bound1_si512, _mm512_load_si512(dst + (i + 1) * 16), 1);
                __mmask16 msk2 = _mm512_cmp_epi32_mask(bound2_si512, _mm512_load_si512(dst + (i + 2) * 16), 1);
                __mmask16 msk3 = _mm512_cmp_epi32_mask(bound3_si512, _mm512_load_si512(dst + (i + 3) * 16), 1);
                __mmask16 msk4 = _mm512_cmp_epi32_mask(bound4_si512, _mm512_load_si512(dst + (i + 4) * 16), 1);
                __mmask16 msk5 = _mm512_cmp_epi32_mask(bound5_si512, _mm512_load_si512(dst + (i + 5) * 16), 1);
                __mmask16 msk6 = _mm512_cmp_epi32_mask(bound6_si512, _mm512_load_si512(dst + (i + 6) * 16), 1);
                __mmask16 msk7 = _mm512_cmp_epi32_mask(bound7_si512, _mm512_load_si512(dst + (i + 7) * 16), 1);
                __mmask16 msk8 = _mm512_cmp_epi32_mask(bound8_si512, _mm512_load_si512(dst + (i + 8) * 16), 1);
                __mmask16 msk9 = _mm512_cmp_epi32_mask(bound9_si512, _mm512_load_si512(dst + (i + 9) * 16), 1);
                __mmask16 mskA = _mm512_cmp_epi32_mask(boundA_si512, _mm512_load_si512(dst + (i + 10) * 16), 1);
                __mmask16 mskB = _mm512_cmp_epi32_mask(boundB_si512, _mm512_load_si512(dst + (i + 11) * 16), 1);
                __mmask16 mskC = _mm512_cmp_epi32_mask(boundC_si512, _mm512_load_si512(dst + (i + 12) * 16), 1);
                __mmask16 mskD = _mm512_cmp_epi32_mask(boundD_si512, _mm512_load_si512(dst + (i + 13) * 16), 1);
                __mmask16 mskE = _mm512_cmp_epi32_mask(boundE_si512, _mm512_load_si512(dst + (i + 14) * 16), 1);
                __mmask16 mskF = _mm512_cmp_epi32_mask(boundF_si512, _mm512_load_si512(dst + (i + 15) * 16), 1);

                CHECK_MSK16_PAIR_AND_PACK_PPART(msk0, msk1, 0, 1, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk2, msk3, 2, 3, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk4, msk5, 4, 5, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk6, msk7, 6, 7, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk8, msk9, 8, 9, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(mskA, mskB, 10, 11, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(mskC, mskD, 12, 13, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(mskE, mskF, 14, 15, i);
            }
            
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 16) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 16 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }

            ind += 16;
        }
        if (ind < end_ind) {
            __attribute__ ((aligned (64))) int32_t norm_epi32[16];
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            for (long i = ind; i < end_ind; i++) norm_epi32[i - ind] = main_bucket->pnorm[i];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps)));
            if (reuse) _mm512_store_si512(rbound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps)));

            __m512i bound_si512[16];
            for (long i = 0; i < end_ind - ind; i++) bound_si512[i] = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + i)));
            
            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];
            __attribute__ ((aligned (64))) int8_t tmp[16 * vec_length + 32];
            for (long i = ind; i < end_ind; i++) {
                _mm512_storeu_si512(tmp + (i-ind) * vec_length, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[i]) * vec_length)));
                _mm512_storeu_si512(tmp + (i-ind) * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[i]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(tmp + (i-ind) * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[i]) * vec_length + 128)));
            }

            _tile_loadd(2, tmp, vec_length);
            _tile_loadd(3, tmp + 64, vec_length);
            _tile_loadd(4, tmp + 128, vec_length);
            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk[16];
                for (long j = 0; j < end_ind - ind; j++) {
                    msk[j] = _mm512_cmp_epi32_mask(bound_si512[j], _mm512_load_si512(dst + (i + j) * 16), 1);
                }
                for (long j = 0; j < end_ind - ind; j++) {
                    CHECK_MSK16_AND_PACK_PPART(msk[j], j, i);
                }
            }
        }

        try_add2_stat[thread] += try_add2;
        succ_add2_stat[thread] += succ_add2;
        for (long i = 0; i < batchsize; i++) {
            local_buf_num[thread * batchsize + i] = _local_buf_num[i];
            if (reuse) local_buf_rnum[thread * batchsize + i] = _local_buf_rnum[i];
            local_buf_size[thread * batchsize + i] = _local_buf_size[i];
        }
    }

    /////// PPART: combine the data to main buckets ///////
    #pragma omp parallel for
    for (long i = 0; i < batchsize; i++) {
        long total_num = 0;
        long total_rnum = 0;
        for (long j = 0; j < num_threads; j++) {
            total_num += local_buf_num[j * batchsize + i];
            if (reuse) total_rnum += local_buf_rnum[j * batchsize + i];
        }
        bucket[i]->_alloc(total_num, 1);
        bucket[i]->num_pvec = total_num;
        if (reuse) {
            rbucket[i]->_alloc(total_rnum, 1);
            rbucket[i]->num_pvec = total_rnum;
        }
    }

    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        uint32_t **_local_buf = local_buf + thread * batchsize;
        uint32_t *_local_buf_num = local_buf_num + thread * batchsize;
        for (long j = 0; j < batchsize; j++) {
            bucket_amx_t *dst, *rdst;
            long begin_pind = 0, begin_rpind = 0;
            dst = bucket[j];
            if (reuse) rdst = rbucket[j];
            for (long i = 0; i < thread; i++) {
                begin_pind += local_buf_num[i * batchsize + j];
                if (reuse) begin_rpind += local_buf_rnum[i * batchsize + j];
            }
            uint32_t *pind_dst = dst->pvec + begin_pind;
            int32_t *pnorm_dst = dst->pnorm + begin_pind;
            uint32_t *rpind_dst;
            int32_t *rpnorm_dst;
            if (reuse) {
                rpind_dst = rdst->pvec + begin_rpind;
                rpnorm_dst = rdst->pnorm + begin_rpind;
            } 

            uint32_t *src = _local_buf[j];
            long src_num = _local_buf_num[j];
            
            // ASSUME THE OUTPUT OF MALLOC AND REALLOC IS 8-ALIGNED
            while ((long) src % 64L != 0 && src_num) {
                uint32_t norm_with_hint = src[1];
                int32_t norm = norm_with_hint & 0x3fffffff;
                pind_dst[0] = src[0];
                pnorm_dst[0] = norm;
                pind_dst++;
                pnorm_dst++;
                if (reuse && norm_with_hint & 0x80000000) {
                    rpind_dst[0] = src[0];
                    rpnorm_dst[0] = norm;
                    rpind_dst++;
                    rpnorm_dst++;
                }
                src_num--;
                src += 2;
            }

            long ind = 0;
            while (ind < src_num - 255) {
                #define UNPACK_AND_ADD_TO_BUCKETS_PPART(__ind)   \
                                                    do {   \
                    __m512i src0 = _mm512_stream_load_si512(src + (__ind) * 2);         \
                    __m512i src1 = _mm512_stream_load_si512(src + (__ind) * 2 + 16);    \
                    __m512i ind_b16 = _mm512_mask_blend_epi32(0xaaaa, src0, _mm512_slli_epi64(src1, 32));    \
                    __m512i hnorm_b16 = _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64(src0, 32), src1);  \
                    __m512i norm_b16 = _mm512_and_si512(hnorm_b16, _mm512_set1_epi32(0x7fffffff)); \
                    __mmask16 rmsk = _mm512_cmp_epu32_mask(_mm512_set1_epi32(0x80000000), hnorm_b16, 2);   \
                    _mm512_storeu_si512(pind_dst, ind_b16);     \
                    _mm512_storeu_si512(pnorm_dst, norm_b16);   \
                    pind_dst += 16;     \
                    pnorm_dst += 16;    \
                    if (reuse && rmsk) { \
                        _mm512_mask_compressstoreu_epi32(rpind_dst, rmsk, ind_b16);   \
                        _mm512_mask_compressstoreu_epi32(rpnorm_dst, rmsk, norm_b16);  \
                        rpind_dst += __builtin_popcount(rmsk);  \
                        rpnorm_dst += __builtin_popcount(rmsk); \
                    }   \
                } while (0)
                
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 0);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 16);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 32);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 48);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 64);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 80);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 96);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 112);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 128);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 144);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 160);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 176);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 192);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 208);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 224);
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 240);
                ind += 256;
            }
            while (ind < src_num - 15) {
                UNPACK_AND_ADD_TO_BUCKETS_PPART(ind);
                ind += 16;
            }
            while (ind < src_num) {
                uint32_t norm_with_hint = src[ind * 2 + 1];
                int32_t norm = norm_with_hint & 0x7fffffff;
                pind_dst[0] = src[ind * 2];
                pnorm_dst[0] = norm;
                pind_dst++;
                pnorm_dst++;
                if (reuse && norm_with_hint & 0x80000000) {
                    rpind_dst[0] = src[ind * 2];
                    rpnorm_dst[0] = norm;
                    rpind_dst++;
                    rpnorm_dst++;
                }
                ind++;
            }
        }
    }

    /////// NPART: each thread collect vectors in local buffers ///////
    #pragma omp parallel for 
    for (long thread = 0; thread < num_threads; thread++) {
        ///// tile configuration /////
        TILE_INITIALIZE;

        ///// prepare local buffers //////
        uint32_t _local_buf_size[batchsize];
        uint32_t _local_buf_num[batchsize] = {};
        uint32_t _local_buf_rnum[batchsize] = {};
        uint32_t **_local_buf = local_buf + thread * batchsize;
        for (long i = 0; i < batchsize; i++) _local_buf_size[i] = local_buf_size[i + thread * batchsize];

        sol_list_amx_t *sol = sol_list[thread];
        uint64_t try_add2 = 0, succ_add2 = 0;

        const long nblocks = (main_bucket->num_nvec + 15) / 16;
        const long begin_ind = (thread * nblocks / num_threads) * 16;
        const long end_ind = ((thread + 1) * nblocks / num_threads) * 16 > main_bucket->num_nvec ? main_bucket->num_nvec : ((thread + 1) * nblocks / num_threads) * 16;

        // real ind is __ind + ind, real __cind is __cind16 + r
        #define CHECK_MSK16_AND_PACK_NPART(__msk, __ind, __cind16)                                              \
                                                            do {                                                \
            while (__msk) {                                                                                     \
                int r = __builtin_ctz(__msk);                                                                   \
                __msk &= ~(1 << r);                                                                             \
                _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r]] = main_bucket->nvec[ind+(__ind)];\
                int norm_with_hint = norm_epi32[__ind];                                                         \
                _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r] + 1] = norm_with_hint;            \
                if (reuse) {                                                                                    \
                    if (dst[((__ind) + (__cind16)) * 16 + r] < rbound[__ind]) {                                 \
                        _local_buf_rnum[(__cind16) + r]++;                                                      \
                        _local_buf[(__cind16) + r][2 * _local_buf_num[(__cind16) + r] + 1] |= 0x80000000;       \
                        if (-dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                            TRY_ADDA_AMX(center_ind_list[(__cind16) + r], main_bucket->nvec[ind + (__ind)]);    \
                        }                                                                                       \
                    }                                                                                           \
                } else {                                                                                        \
                    if (-dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                        TRY_ADDA_AMX(center_ind_list[(__cind16) + r], main_bucket->nvec[ind + (__ind)]);        \
                    }                                                                                           \
                }                                                                                               \
                _local_buf_num[(__cind16) + r]++;                                                               \
            }                                                                                                   \
        } while (0)

        #define CHECK_MSK16_PAIR_AND_PACK_NPART(__msk0, __msk1, __ind0, __ind1, __cind16)             \
                                                            do {                                \
            if (!_mm512_kortestz(__msk0, __msk1)) {                                             \
                CHECK_MSK16_AND_PACK_NPART(__msk0, __ind0, __cind16);                                 \
                CHECK_MSK16_AND_PACK_NPART(__msk1, __ind1, __cind16);                                 \
            }                                                                                   \
        } while (0)

        long ind = begin_ind;
        while (ind < end_ind - 127) {
            int32_t *norm_epi32 = main_bucket->nnorm + ind;
            __attribute__ ((aligned (64))) int32_t _bound[64];
            __attribute__ ((aligned (64))) int32_t _rbound[64];
            __m512i norm_si512_0 = _mm512_loadu_si512(norm_epi32 + 0);
            __m512i norm_si512_1 = _mm512_loadu_si512(norm_epi32 + 16);
            __m512i norm_si512_2 = _mm512_loadu_si512(norm_epi32 + 32);
            __m512i norm_si512_3 = _mm512_loadu_si512(norm_epi32 + 48);
            _mm512_store_si512(_bound + 0, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_bx2_ps))));
            _mm512_store_si512(_bound + 16, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_bx2_ps))));
            _mm512_store_si512(_bound + 32, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_bx2_ps))));
            _mm512_store_si512(_bound + 48, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_bx2_ps))));
            if (reuse) {
                _mm512_store_si512(_rbound + 0, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_rx2_ps))));
                _mm512_store_si512(_rbound + 16, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_rx2_ps))));
                _mm512_store_si512(_rbound + 32, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_rx2_ps))));
                _mm512_store_si512(_rbound + 48, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_rx2_ps))));
            }
            __attribute__ ((aligned (64))) int32_t _dst[64 * batchsize];
            int32_t *dst = _dst;
            int32_t *bound = _bound;
            int32_t *rbound = _rbound;

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec1[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec2[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec3[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 0 + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 0 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 0 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec1 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 16 + j]) * vec_length)));
                _mm512_storeu_si512(vec1 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 16 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec1 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 16 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec2 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 32 + j]) * vec_length)));
                _mm512_storeu_si512(vec2 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 32 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec2 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 32 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec3 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 48 + j]) * vec_length)));
                _mm512_storeu_si512(vec3 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 48 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec3 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 48 + j]) * vec_length + 128)));
            }

            
            long prefetch_start_ind = (ind+64);
            constexpr int32_t n_prefetch_vec = 64;

            for (long i = 0; i < batchsize; i += 16) {
                for (long j = 0; j < 16 && i <= batchsize - 32; j++) {
                    _mm_prefetch(center0_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center1_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center2_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                }
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
                _tile_loadd(4, center0_tr + i * 64, 64);
                _tile_loadd(5, vec0, vec_length);
                _tile_loadd(6, vec1, vec_length);
                _tile_loadd(7, vec2, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3, vec_length);
                _tile_dpbssd(1, 6, 4);
                _tile_loadd(6, center1_tr + i * 64, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_loadd(7, vec0 + 64, vec_length);
                _tile_dpbssd(3, 5, 4);
                _tile_loadd(4, vec1 + 64, vec_length);
                _tile_loadd(5, vec2 + 64, vec_length);
                _tile_dpbssd(0, 7, 6);
                _tile_loadd(7, vec3 + 64, vec_length);
                _tile_dpbssd(1, 4, 6);
                _tile_loadd(4, center2_tr + i * 64, 64);
                _tile_dpbssd(2, 5, 6);
                _tile_loadd(5, vec0 + 128, vec_length);
                _tile_dpbssd(3, 7, 6);
                _tile_loadd(6, vec1 + 128, vec_length);
                _tile_loadd(7, vec2 + 128, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3 + 128, vec_length);
                _tile_stored(0, dst + 0 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(1, 6, 4);
                _tile_stored(1, dst + 1 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_stored(2, dst + 2 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(3, 5, 4);
                _tile_stored(3, dst + 3 * 16 * batchsize + i * 16, 64);
                for (long j = (n_prefetch_vec * i) / batchsize; j < (n_prefetch_vec * (i + 16)) / batchsize; j++) {
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->nvec[prefetch_start_ind + j], _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->nvec[prefetch_start_ind + j] + 64, _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->nvec[prefetch_start_ind + j] + 128, _MM_HINT_T1);
                }
            }

            for (long kk = 0; kk < 4; kk++) {
                __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
                __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
                __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
                __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
                __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
                __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
                __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
                __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
                __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
                __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
                __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
                __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
                __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
                __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
                __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
                __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

                for (long i = 0; i < batchsize; i += 16) {
                    __mmask16 msk0 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 0) * 16), bound0_si512, 1);
                    __mmask16 msk1 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 1) * 16), bound1_si512, 1);
                    __mmask16 msk2 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 2) * 16), bound2_si512, 1);
                    __mmask16 msk3 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 3) * 16), bound3_si512, 1);
                    __mmask16 msk4 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 4) * 16), bound4_si512, 1);
                    __mmask16 msk5 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 5) * 16), bound5_si512, 1);
                    __mmask16 msk6 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 6) * 16), bound6_si512, 1);
                    __mmask16 msk7 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 7) * 16), bound7_si512, 1);
                    __mmask16 msk8 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 8) * 16), bound8_si512, 1);
                    __mmask16 msk9 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 9) * 16), bound9_si512, 1);
                    __mmask16 mskA = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 10) * 16), boundA_si512, 1);
                    __mmask16 mskB = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 11) * 16), boundB_si512, 1);
                    __mmask16 mskC = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 12) * 16), boundC_si512, 1);
                    __mmask16 mskD = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 13) * 16), boundD_si512, 1);
                    __mmask16 mskE = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 14) * 16), boundE_si512, 1);
                    __mmask16 mskF = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 15) * 16), boundF_si512, 1);

                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk0, msk1, 0, 1, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk2, msk3, 2, 3, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk4, msk5, 4, 5, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk6, msk7, 6, 7, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk8, msk9, 8, 9, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(mskA, mskB, 10, 11, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(mskC, mskD, 12, 13, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(mskE, mskF, 14, 15, i);
                }

                ind += 16;
                dst += 16 * batchsize;
                norm_epi32 += 16;
                bound += 16;
                if (reuse) rbound += 16;
            }
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 64) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 64 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }
        }
        while (ind < end_ind - 15) {
            int32_t *norm_epi32 = main_bucket->nnorm + ind;
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps))));
            if (reuse) _mm512_store_si512(rbound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps))));

            __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
            __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
            __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
            __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
            __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
            __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
            __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
            __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
            __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
            __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
            __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
            __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
            __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
            __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
            __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
            __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + j]) * vec_length + 128)));
            }

            _tile_loadd(2, vec0, vec_length);
            _tile_loadd(3, vec0 + 64, vec_length);
            _tile_loadd(4, vec0 + 128, vec_length);

            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            
            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk0 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 0) * 16), bound0_si512, 1);
                __mmask16 msk1 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 1) * 16), bound1_si512, 1);
                __mmask16 msk2 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 2) * 16), bound2_si512, 1);
                __mmask16 msk3 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 3) * 16), bound3_si512, 1);
                __mmask16 msk4 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 4) * 16), bound4_si512, 1);
                __mmask16 msk5 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 5) * 16), bound5_si512, 1);
                __mmask16 msk6 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 6) * 16), bound6_si512, 1);
                __mmask16 msk7 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 7) * 16), bound7_si512, 1);
                __mmask16 msk8 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 8) * 16), bound8_si512, 1);
                __mmask16 msk9 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 9) * 16), bound9_si512, 1);
                __mmask16 mskA = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 10) * 16), boundA_si512, 1);
                __mmask16 mskB = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 11) * 16), boundB_si512, 1);
                __mmask16 mskC = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 12) * 16), boundC_si512, 1);
                __mmask16 mskD = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 13) * 16), boundD_si512, 1);
                __mmask16 mskE = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 14) * 16), boundE_si512, 1);
                __mmask16 mskF = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 15) * 16), boundF_si512, 1);

                CHECK_MSK16_PAIR_AND_PACK_NPART(msk0, msk1, 0, 1, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk2, msk3, 2, 3, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk4, msk5, 4, 5, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk6, msk7, 6, 7, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk8, msk9, 8, 9, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(mskA, mskB, 10, 11, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(mskC, mskD, 12, 13, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(mskE, mskF, 14, 15, i);
            }
            
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 16) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 16 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }

            ind += 16;
        }
        
        if (ind < end_ind) {
            __attribute__ ((aligned (64))) int32_t norm_epi32[16];
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            for (long i = ind; i < end_ind; i++) norm_epi32[i - ind] = main_bucket->nnorm[i];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps))));
            if (reuse) _mm512_store_si512(rbound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps))));

            __m512i bound_si512[16];
            for (long i = 0; i < end_ind - ind; i++) bound_si512[i] = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + i)));
            
            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];
            __attribute__ ((aligned (64))) int8_t tmp[16 * vec_length + 32];
            for (long i = ind; i < end_ind; i++) {
                _mm512_storeu_si512(tmp + (i-ind) * vec_length, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[i]) * vec_length)));
                _mm512_storeu_si512(tmp + (i-ind) * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[i]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(tmp + (i-ind) * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[i]) * vec_length + 128)));
            }

            _tile_loadd(2, tmp, vec_length);
            _tile_loadd(3, tmp + 64, vec_length);
            _tile_loadd(4, tmp + 128, vec_length);
            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk[16];
                for (long j = 0; j < end_ind - ind; j++) {
                    msk[j] = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + j) * 16), bound_si512[j], 1);
                }
                for (long j = 0; j < end_ind - ind; j++) {
                    CHECK_MSK16_AND_PACK_NPART(msk[j], j, i);
                }
            }
        }

        try_add2_stat[thread] += try_add2;
        succ_add2_stat[thread] += succ_add2;
        for (long i = 0; i < batchsize; i++) {
            local_buf_num[thread * batchsize + i] = _local_buf_num[i];
            if (reuse) local_buf_rnum[thread * batchsize + i] = _local_buf_rnum[i];
            local_buf_size[thread * batchsize + i] = _local_buf_size[i];
        }
    }

    /////// NPART: combine the data to main buckets ///////
    #pragma omp parallel for
    for (long i = 0; i < batchsize; i++) {
        long total_num = 0;
        long total_rnum = 0;
        for (long j = 0; j < num_threads; j++) {
            total_num += local_buf_num[j * batchsize + i];
            if (reuse) total_rnum += local_buf_rnum[j * batchsize + i];
        }
        bucket[i]->_alloc(total_num, 0);
        bucket[i]->num_nvec = total_num;
        if (reuse) {
            rbucket[i]->_alloc(total_rnum, 0);
            rbucket[i]->num_nvec = total_rnum;
        }
    }

    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        uint32_t **_local_buf = local_buf + thread * batchsize;
        uint32_t *_local_buf_num = local_buf_num + thread * batchsize;
        for (long j = 0; j < batchsize; j++) {
            bucket_amx_t *dst, *rdst;
            long begin_nind = 0, begin_rnind = 0;
            dst = bucket[j];
            if (reuse) rdst = rbucket[j];
            for (long i = 0; i < thread; i++) {
                begin_nind += local_buf_num[i * batchsize + j];
                if (reuse) begin_rnind += local_buf_rnum[i * batchsize + j];
            }
            uint32_t *nind_dst = dst->nvec + begin_nind;
            int32_t *nnorm_dst = dst->nnorm + begin_nind;
            uint32_t *rnind_dst; 
            int32_t *rnnorm_dst;
            if (reuse) {
                rnind_dst = rdst->nvec + begin_rnind;
                rnnorm_dst = rdst->nnorm + begin_rnind;
            } 

            uint32_t *src = _local_buf[j];
            long src_num = _local_buf_num[j];
            
            // ASSUME THE OUTPUT OF MALLOC AND REALLOC IS 8-ALIGNED
            while ((long) src % 64L != 0 && src_num) {
                uint32_t norm_with_hint = src[1];
                int32_t norm = norm_with_hint & 0x3fffffff;
                nind_dst[0] = src[0];
                nnorm_dst[0] = norm;
                nind_dst++;
                nnorm_dst++;
                if (reuse && norm_with_hint & 0x80000000) {
                    rnind_dst[0] = src[0];
                    rnnorm_dst[0] = norm;
                    rnind_dst++;
                    rnnorm_dst++;
                }
                src_num--;
                src += 2;
            }

            long ind = 0;
            while (ind < src_num - 255) {
                #define UNPACK_AND_ADD_TO_BUCKETS_NPART(__ind)   \
                                                    do {   \
                    __m512i src0 = _mm512_stream_load_si512(src + (__ind) * 2);         \
                    __m512i src1 = _mm512_stream_load_si512(src + (__ind) * 2 + 16);    \
                    __m512i ind_b16 = _mm512_mask_blend_epi32(0xaaaa, src0, _mm512_slli_epi64(src1, 32));    \
                    __m512i hnorm_b16 = _mm512_mask_blend_epi32(0xaaaa, _mm512_srli_epi64(src0, 32), src1);  \
                    __m512i norm_b16 = _mm512_and_si512(hnorm_b16, _mm512_set1_epi32(0x7fffffff)); \
                    __mmask16 rmsk = _mm512_cmp_epu32_mask(_mm512_set1_epi32(0x80000000), hnorm_b16, 2);   \
                    _mm512_storeu_si512(nind_dst, ind_b16);     \
                    _mm512_storeu_si512(nnorm_dst, norm_b16);   \
                    nind_dst += 16;     \
                    nnorm_dst += 16;    \
                    if (reuse && rmsk) { \
                        _mm512_mask_compressstoreu_epi32(rnind_dst, rmsk, ind_b16);   \
                        _mm512_mask_compressstoreu_epi32(rnnorm_dst, rmsk, norm_b16);  \
                        rnind_dst += __builtin_popcount(rmsk);  \
                        rnnorm_dst += __builtin_popcount(rmsk); \
                    }   \
                } while (0)
                
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 0);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 16);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 32);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 48);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 64);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 80);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 96);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 112);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 128);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 144);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 160);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 176);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 192);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 208);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 224);
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 240);
                ind += 256;
            }
            while (ind < src_num - 15) {
                UNPACK_AND_ADD_TO_BUCKETS_NPART(ind);
                ind += 16;
            }
            while (ind < src_num) {
                uint32_t norm_with_hint = src[ind * 2 + 1];
                int32_t norm = norm_with_hint & 0x7fffffff;
                nind_dst[0] = src[ind * 2];
                nnorm_dst[0] = norm;
                nind_dst++;
                nnorm_dst++;
                if (reuse && norm_with_hint & 0x80000000) {
                    rnind_dst[0] = src[ind * 2];
                    rnnorm_dst[0] = norm;
                    rnind_dst++;
                    rnnorm_dst++;
                }
                ind++;
            }
        }
    }

    /////// free local buffer and center ///////
    for (long i = 0; i < batchsize * num_threads; i++) {
        free(local_buf[i]);
    }
    FREE_VEC((void *)center0_tr);
    FREE_VEC((void *)center1_tr);
    FREE_VEC((void *)center2_tr);

    /////// remove duplicates ///////
    for (long i = 0; i < batchsize; i++) {
        for (long j = 0; j < i; j++) {
            if (center_ind_list[i] == center_ind_list[j]) {
                if (reuse) rbucket[i]->num_pvec = 0;
                if (reuse) rbucket[i]->num_nvec = 0;
                bucket[i]->num_pvec = 0;
                bucket[i]->num_nvec = 0;
                break;
            }
        }
    }

    if (prof) {
        for (long i = 1; i < num_threads; i++) {
            succ_add2_stat[0] += succ_add2_stat[i];
            try_add2_stat[0] += try_add2_stat[i];
        }
        // pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2_stat[0];
        prof->succ_add2 += succ_add2_stat[0];
        // pthread_spin_unlock(&prof->profile_lock);
    }
    return 0;
}

template <uint32_t nb>
template <uint32_t batchsize, bool faraway_center, bool reuse>
int Pool_epi8_t<nb>::_sub_bucketing_amx(bucket_amx_t *main_bucket, bucket_amx_t **rbucket, bucket_amx_t **bucket, 
            double alpha_r, double alpha_b, sol_list_amx_t *sol, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof) {
    /////// prepare dst and local buffer ///////
    for (long i = 0; i < batchsize; i++) {
        if (reuse) {
            if (rbucket[i]) {
                if (rbucket[i]->num_pvec || rbucket[i]->num_nvec) {
                    fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_sub_bucketing_amx: nonempty input buckets, ignored.\n", nb);
                    rbucket[i]->num_pvec = 0;
                    rbucket[i]->num_nvec = 0;
                }
            } else {
                rbucket[i] = new bucket_amx_t;
            }
            rbucket[i]->_alloc(256, 1);
            rbucket[i]->_alloc(256, 0);
        }
        if (bucket[i]) {
            if (bucket[i]->num_pvec || bucket[i]->num_nvec) {
                fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_sub_bucketing_amx: nonempty input buckets, ignored.\n", nb);
                bucket[i]->num_pvec = 0;
                bucket[i]->num_nvec = 0;
            } 
        } else {
            bucket[i] = new bucket_amx_t;
        }
        bucket[i]->_alloc(256, 1);
        bucket[i]->_alloc(256, 0);
    }
    const __m512 alpha_rx2_ps = _mm512_set1_ps(alpha_r * 2.0);
    const __m512 alpha_bx2_ps = _mm512_set1_ps(alpha_b * 2.0);
    #if AMX_BUCKET2_USE_BUFFER
    const long expect_bucket_size = pow(1.0 - alpha_b * alpha_b, CSD * 0.5) * (main_bucket->num_pvec + main_bucket->num_nvec);
    const long init_buffer_size = 0.5 * expect_bucket_size + batchsize * 64;
    uint32_t *local_buf[batchsize];
    uint32_t local_buf_num[batchsize];
    uint32_t local_buf_rnum[batchsize];
    uint32_t local_buf_size[batchsize];
    for (long i = 0; i < batchsize; i++) {
        local_buf[i] = (uint32_t *) malloc(init_buffer_size * 2 * sizeof(uint32_t));
        local_buf_size[i] = init_buffer_size;
    }
    #endif
    uint64_t try_add2 = 0;
    uint64_t succ_add2 = 0;

    /////// choose centers ///////
    uint32_t center_ind_list[batchsize];
    int32_t center_norm[batchsize];
    int8_t *center0_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    int8_t *center1_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    int8_t *center2_tr = (int8_t *) NEW_VEC(batchsize * 64, sizeof(int8_t));
    long num_try_find_center = 0;
    for (long i = 0; i < batchsize; i++) {
        int pass;
        do {
            int seed = std::chrono::steady_clock::now().time_since_epoch().count();
            seed += (long) main_bucket;
            seed *= sol->num_a;
            seed -= sol->num_s;
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<long> dist(0, main_bucket->num_pvec - 1);
            pass = 1;
            center_ind_list[i] = main_bucket->pvec[dist(rng)];
            if (num_try_find_center < AMX_MAX_NUM_TRYCENTER/2) {
                for (long j = 0; j < i; j++){
                    if (center_ind_list[j] == center_ind_list[i]) pass = 0;
                }
            }
            int8_t *ptr = vec + center_ind_list[i] * vec_length;
            num_try_find_center++;
            if (faraway_center) {
                if ((CSD * (int)ptr[0] * (int)ptr[0] > 2 * vnorm[center_ind_list[i]]) && (num_try_find_center < (long)(0.4 * AMX_MAX_NUM_TRYCENTER))) pass = 0;
            }
            if (reuse) rbucket[i]->center_ind = center_ind_list[i];
            bucket[i]->center_ind = center_ind_list[i];
            center_norm[i] = vnorm[center_ind_list[i]] - goal_norm;
        } while(!pass);
    }
    if (num_try_find_center >= (long)(0.4 * AMX_MAX_NUM_TRYCENTER)) {
        if (CSD >= 108) fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_sub_bucketing_amx: fail to use faraway centers after %ld tries.\n", nb, num_try_find_center);
    }
    for (long i = 0; i < batchsize; i += 16) {
        __m512i z[16];
        __m256i y[16];
        for (long j = 0; j < 16; j++) {
            z[j] = _mm512_loadu_si512((__m512i *)(vec + center_ind_list[i+j] * vec_length));
        }
        AVX512_MATTR_16x16(NULL, center0_tr + i * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
        for (long j = 0; j < 16; j++) {
            z[j] = _mm512_loadu_si512((__m512i *)(vec + center_ind_list[i+j] * vec_length + 64));
        }
        AVX512_MATTR_16x16(NULL, center1_tr + i * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
        for (long j = 0; j < 16; j++) {
            y[j] = _mm256_load_si256((__m256i *)(vec + center_ind_list[i+j] * vec_length + 128));
        }
        AVX512_MATTR_16x8(NULL, center2_tr + i * 64, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]);
    }

    /////// PPART: each thread collect vectors in local buffers /////// 
    do {
        ///// tile configuration /////
        TILE_INITIALIZE;

        ///// prepare local buffers //////
        #if AMX_BUCKET2_USE_BUFFER
        uint32_t _local_buf_size[batchsize];
        uint32_t _local_buf_num[batchsize] = {};
        uint32_t _local_buf_rnum[batchsize] = {};
        uint32_t **_local_buf = local_buf;
        for (long i = 0; i < batchsize; i++) _local_buf_size[i] = local_buf_size[i];
        #endif

        const long begin_ind = 0;
        const long end_ind = main_bucket->num_pvec;

        #define CHECK_MSK16_AND_ADD_PPART(__msk, __ind, __cind16)                                               \
                                                                  do {                                          \
            while (__msk) {                                                                                     \
                int r = __builtin_ctz(__msk);                                                                   \
                __msk &= ~(1 << r);                                                                             \
                bucket[(__cind16) + r]->add_pvec(main_bucket->pvec[ind+(__ind)], norm_epi32[__ind]);            \
                if (reuse) {                                                                                    \
                    if (dst[((__ind) + (__cind16)) * 16 + r] > rbound[__ind]) {                                 \
                        rbucket[(__cind16) + r]->add_pvec(main_bucket->pvec[ind+(__ind)], norm_epi32[__ind]);   \
                        if (dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                            if (center_ind_list[(__cind16) + r] != main_bucket->pvec[ind + (__ind)]) {          \
                                TRY_ADDS_AMX(center_ind_list[(__cind16) + r], main_bucket->pvec[ind + (__ind)]);\
                            } else {                                                                            \
                                bucket[(__cind16) + r]->num_pvec--;                                             \
                                rbucket[(__cind16) + r]->num_pvec--;                                            \
                            }                                                                                   \
                        }                                                                                       \
                    }                                                                                           \
                } else {                                                                                        \
                    if (dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                        if (center_ind_list[(__cind16) + r] != main_bucket->pvec[ind + (__ind)]) {              \
                            TRY_ADDS_AMX(center_ind_list[(__cind16) + r], main_bucket->pvec[ind + (__ind)]);    \
                        } else bucket[(__cind16) + r]->num_pvec--;                                              \
                    }                                                                                           \
                }                                                                                               \
            }                                                                                                   \
        } while (0)

        #define CHECK_MSK16_PAIR_AND_ADD_PPART(__msk0, __msk1, __ind0, __ind1, __cind16)                        \
                                                                            do {                                \
            if (!_mm512_kortestz(__msk0, __msk1)) {                                                             \
                CHECK_MSK16_AND_ADD_PPART(__msk0, __ind0, __cind16);                                            \
                CHECK_MSK16_AND_ADD_PPART(__msk1, __ind1, __cind16);                                            \
            }                                                                                                   \
        } while (0)

        long ind = begin_ind;
        while (ind < end_ind - 127) {
            int32_t *norm_epi32 = main_bucket->pnorm + ind;
            __attribute__ ((aligned (64))) int32_t _bound[64];
            __attribute__ ((aligned (64))) int32_t _rbound[64];
            __m512i norm_si512_0 = _mm512_loadu_si512(norm_epi32 + 0);
            __m512i norm_si512_1 = _mm512_loadu_si512(norm_epi32 + 16);
            __m512i norm_si512_2 = _mm512_loadu_si512(norm_epi32 + 32);
            __m512i norm_si512_3 = _mm512_loadu_si512(norm_epi32 + 48);
            _mm512_store_si512(_bound + 0, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_bx2_ps)));
            _mm512_store_si512(_bound + 16, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_bx2_ps)));
            _mm512_store_si512(_bound + 32, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_bx2_ps)));
            _mm512_store_si512(_bound + 48, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_bx2_ps)));
            if (reuse) {
                _mm512_store_si512(_rbound + 0, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_rx2_ps)));
                _mm512_store_si512(_rbound + 16, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_rx2_ps)));
                _mm512_store_si512(_rbound + 32, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_rx2_ps)));
                _mm512_store_si512(_rbound + 48, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_rx2_ps)));
            }
            __attribute__ ((aligned (64))) int32_t _dst[64 * batchsize];
            int32_t *dst = _dst;
            int32_t *bound = _bound;
            int32_t *rbound = _rbound;

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec1[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec2[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec3[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 0 + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 0 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 0 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec1 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 16 + j]) * vec_length)));
                _mm512_storeu_si512(vec1 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 16 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec1 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 16 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec2 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 32 + j]) * vec_length)));
                _mm512_storeu_si512(vec2 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 32 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec2 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 32 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec3 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 48 + j]) * vec_length)));
                _mm512_storeu_si512(vec3 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + 48 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec3 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + 48 + j]) * vec_length + 128)));
            }

            
            long prefetch_start_ind = (ind+64);
            constexpr int32_t n_prefetch_vec = 64;

            for (long i = 0; i < batchsize; i += 16) {
                for (long j = 0; j < 16 && i <= batchsize - 32; j++) {
                    _mm_prefetch(center0_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center1_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center2_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                }
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
                _tile_loadd(4, center0_tr + i * 64, 64);
                _tile_loadd(5, vec0, vec_length);
                _tile_loadd(6, vec1, vec_length);
                _tile_loadd(7, vec2, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3, vec_length);
                _tile_dpbssd(1, 6, 4);
                _tile_loadd(6, center1_tr + i * 64, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_loadd(7, vec0 + 64, vec_length);
                _tile_dpbssd(3, 5, 4);
                _tile_loadd(4, vec1 + 64, vec_length);
                _tile_loadd(5, vec2 + 64, vec_length);
                _tile_dpbssd(0, 7, 6);
                _tile_loadd(7, vec3 + 64, vec_length);
                _tile_dpbssd(1, 4, 6);
                _tile_loadd(4, center2_tr + i * 64, 64);
                _tile_dpbssd(2, 5, 6);
                _tile_loadd(5, vec0 + 128, vec_length);
                _tile_dpbssd(3, 7, 6);
                _tile_loadd(6, vec1 + 128, vec_length);
                _tile_loadd(7, vec2 + 128, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3 + 128, vec_length);
                _tile_stored(0, dst + 0 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(1, 6, 4);
                _tile_stored(1, dst + 1 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_stored(2, dst + 2 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(3, 5, 4);
                _tile_stored(3, dst + 3 * 16 * batchsize + i * 16, 64);
                for (long j = (n_prefetch_vec * i) / batchsize; j < (n_prefetch_vec * (i + 16)) / batchsize; j++) {
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->pvec[prefetch_start_ind + j], _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->pvec[prefetch_start_ind + j] + 64, _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->pvec[prefetch_start_ind + j] + 128, _MM_HINT_T1);
                }
            }

            for (long kk = 0; kk < 4; kk++) {
                __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
                __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
                __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
                __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
                __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
                __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
                __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
                __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
                __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
                __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
                __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
                __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
                __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
                __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
                __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
                __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

                for (long i = 0; i < batchsize; i += 16) {
                    __mmask16 msk0 = _mm512_cmp_epi32_mask(bound0_si512, _mm512_load_si512(dst + (i + 0) * 16), 1);
                    __mmask16 msk1 = _mm512_cmp_epi32_mask(bound1_si512, _mm512_load_si512(dst + (i + 1) * 16), 1);
                    __mmask16 msk2 = _mm512_cmp_epi32_mask(bound2_si512, _mm512_load_si512(dst + (i + 2) * 16), 1);
                    __mmask16 msk3 = _mm512_cmp_epi32_mask(bound3_si512, _mm512_load_si512(dst + (i + 3) * 16), 1);
                    __mmask16 msk4 = _mm512_cmp_epi32_mask(bound4_si512, _mm512_load_si512(dst + (i + 4) * 16), 1);
                    __mmask16 msk5 = _mm512_cmp_epi32_mask(bound5_si512, _mm512_load_si512(dst + (i + 5) * 16), 1);
                    __mmask16 msk6 = _mm512_cmp_epi32_mask(bound6_si512, _mm512_load_si512(dst + (i + 6) * 16), 1);
                    __mmask16 msk7 = _mm512_cmp_epi32_mask(bound7_si512, _mm512_load_si512(dst + (i + 7) * 16), 1);
                    __mmask16 msk8 = _mm512_cmp_epi32_mask(bound8_si512, _mm512_load_si512(dst + (i + 8) * 16), 1);
                    __mmask16 msk9 = _mm512_cmp_epi32_mask(bound9_si512, _mm512_load_si512(dst + (i + 9) * 16), 1);
                    __mmask16 mskA = _mm512_cmp_epi32_mask(boundA_si512, _mm512_load_si512(dst + (i + 10) * 16), 1);
                    __mmask16 mskB = _mm512_cmp_epi32_mask(boundB_si512, _mm512_load_si512(dst + (i + 11) * 16), 1);
                    __mmask16 mskC = _mm512_cmp_epi32_mask(boundC_si512, _mm512_load_si512(dst + (i + 12) * 16), 1);
                    __mmask16 mskD = _mm512_cmp_epi32_mask(boundD_si512, _mm512_load_si512(dst + (i + 13) * 16), 1);
                    __mmask16 mskE = _mm512_cmp_epi32_mask(boundE_si512, _mm512_load_si512(dst + (i + 14) * 16), 1);
                    __mmask16 mskF = _mm512_cmp_epi32_mask(boundF_si512, _mm512_load_si512(dst + (i + 15) * 16), 1);

                    #if AMX_BUCKET2_USE_BUFFER
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk0, msk1, 0, 1, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk2, msk3, 2, 3, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk4, msk5, 4, 5, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk6, msk7, 6, 7, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(msk8, msk9, 8, 9, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(mskA, mskB, 10, 11, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(mskC, mskD, 12, 13, i);
                    CHECK_MSK16_PAIR_AND_PACK_PPART(mskE, mskF, 14, 15, i);
                    #else
                    CHECK_MSK16_PAIR_AND_ADD_PPART(msk0, msk1, 0, 1, i);
                    CHECK_MSK16_PAIR_AND_ADD_PPART(msk2, msk3, 2, 3, i);
                    CHECK_MSK16_PAIR_AND_ADD_PPART(msk4, msk5, 4, 5, i);
                    CHECK_MSK16_PAIR_AND_ADD_PPART(msk6, msk7, 6, 7, i);
                    CHECK_MSK16_PAIR_AND_ADD_PPART(msk8, msk9, 8, 9, i);
                    CHECK_MSK16_PAIR_AND_ADD_PPART(mskA, mskB, 10, 11, i);
                    CHECK_MSK16_PAIR_AND_ADD_PPART(mskC, mskD, 12, 13, i);
                    CHECK_MSK16_PAIR_AND_ADD_PPART(mskE, mskF, 14, 15, i);
                    #endif
                }

                ind += 16;
                dst += 16 * batchsize;
                norm_epi32 += 16;
                bound += 16;
                if (reuse) rbound += 16;
            }
            #if AMX_BUCKET2_USE_BUFFER
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 64) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 64 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }
            #endif
        }
        while (ind < end_ind - 15) {
            int32_t *norm_epi32 = main_bucket->pnorm + ind;
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps)));
            if (reuse) _mm512_store_si512(rbound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps)));

            __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
            __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
            __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
            __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
            __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
            __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
            __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
            __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
            __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
            __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
            __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
            __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
            __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
            __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
            __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
            __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[ind + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[ind + j]) * vec_length + 128)));
            }

            _tile_loadd(2, vec0, vec_length);
            _tile_loadd(3, vec0 + 64, vec_length);
            _tile_loadd(4, vec0 + 128, vec_length);

            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            
            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk0 = _mm512_cmp_epi32_mask(bound0_si512, _mm512_load_si512(dst + (i + 0) * 16), 1);
                __mmask16 msk1 = _mm512_cmp_epi32_mask(bound1_si512, _mm512_load_si512(dst + (i + 1) * 16), 1);
                __mmask16 msk2 = _mm512_cmp_epi32_mask(bound2_si512, _mm512_load_si512(dst + (i + 2) * 16), 1);
                __mmask16 msk3 = _mm512_cmp_epi32_mask(bound3_si512, _mm512_load_si512(dst + (i + 3) * 16), 1);
                __mmask16 msk4 = _mm512_cmp_epi32_mask(bound4_si512, _mm512_load_si512(dst + (i + 4) * 16), 1);
                __mmask16 msk5 = _mm512_cmp_epi32_mask(bound5_si512, _mm512_load_si512(dst + (i + 5) * 16), 1);
                __mmask16 msk6 = _mm512_cmp_epi32_mask(bound6_si512, _mm512_load_si512(dst + (i + 6) * 16), 1);
                __mmask16 msk7 = _mm512_cmp_epi32_mask(bound7_si512, _mm512_load_si512(dst + (i + 7) * 16), 1);
                __mmask16 msk8 = _mm512_cmp_epi32_mask(bound8_si512, _mm512_load_si512(dst + (i + 8) * 16), 1);
                __mmask16 msk9 = _mm512_cmp_epi32_mask(bound9_si512, _mm512_load_si512(dst + (i + 9) * 16), 1);
                __mmask16 mskA = _mm512_cmp_epi32_mask(boundA_si512, _mm512_load_si512(dst + (i + 10) * 16), 1);
                __mmask16 mskB = _mm512_cmp_epi32_mask(boundB_si512, _mm512_load_si512(dst + (i + 11) * 16), 1);
                __mmask16 mskC = _mm512_cmp_epi32_mask(boundC_si512, _mm512_load_si512(dst + (i + 12) * 16), 1);
                __mmask16 mskD = _mm512_cmp_epi32_mask(boundD_si512, _mm512_load_si512(dst + (i + 13) * 16), 1);
                __mmask16 mskE = _mm512_cmp_epi32_mask(boundE_si512, _mm512_load_si512(dst + (i + 14) * 16), 1);
                __mmask16 mskF = _mm512_cmp_epi32_mask(boundF_si512, _mm512_load_si512(dst + (i + 15) * 16), 1);

                #if AMX_BUCKET2_USE_BUFFER
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk0, msk1, 0, 1, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk2, msk3, 2, 3, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk4, msk5, 4, 5, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk6, msk7, 6, 7, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(msk8, msk9, 8, 9, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(mskA, mskB, 10, 11, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(mskC, mskD, 12, 13, i);
                CHECK_MSK16_PAIR_AND_PACK_PPART(mskE, mskF, 14, 15, i);
                #else
                CHECK_MSK16_PAIR_AND_ADD_PPART(msk0, msk1, 0, 1, i);
                CHECK_MSK16_PAIR_AND_ADD_PPART(msk2, msk3, 2, 3, i);
                CHECK_MSK16_PAIR_AND_ADD_PPART(msk4, msk5, 4, 5, i);
                CHECK_MSK16_PAIR_AND_ADD_PPART(msk6, msk7, 6, 7, i);
                CHECK_MSK16_PAIR_AND_ADD_PPART(msk8, msk9, 8, 9, i);
                CHECK_MSK16_PAIR_AND_ADD_PPART(mskA, mskB, 10, 11, i);
                CHECK_MSK16_PAIR_AND_ADD_PPART(mskC, mskD, 12, 13, i);
                CHECK_MSK16_PAIR_AND_ADD_PPART(mskE, mskF, 14, 15, i);
                #endif
            }
            
            #if AMX_BUCKET2_USE_BUFFER
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 16) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 16 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }
            #endif

            ind += 16;
        }
        if (ind < end_ind) {
            __attribute__ ((aligned (64))) int32_t norm_epi32[16];
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            for (long i = ind; i < end_ind; i++) norm_epi32[i - ind] = main_bucket->pnorm[i];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps)));
            if (reuse) _mm512_store_si512(rbound, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps)));

            __m512i bound_si512[16];
            for (long i = 0; i < end_ind - ind; i++) bound_si512[i] = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + i)));
            
            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];
            __attribute__ ((aligned (64))) int8_t tmp[16 * vec_length + 32];
            for (long i = ind; i < end_ind; i++) {
                _mm512_storeu_si512(tmp + (i-ind) * vec_length, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[i]) * vec_length)));
                _mm512_storeu_si512(tmp + (i-ind) * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->pvec[i]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(tmp + (i-ind) * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->pvec[i]) * vec_length + 128)));
            }

            _tile_loadd(2, tmp, vec_length);
            _tile_loadd(3, tmp + 64, vec_length);
            _tile_loadd(4, tmp + 128, vec_length);
            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk[16];
                for (long j = 0; j < end_ind - ind; j++) {
                    msk[j] = _mm512_cmp_epi32_mask(bound_si512[j], _mm512_load_si512(dst + (i + j) * 16), 1);
                }
                for (long j = 0; j < end_ind - ind; j++) {
                    #if AMX_BUCKET2_USE_BUFFER
                    CHECK_MSK16_AND_PACK_PPART(msk[j], j, i);
                    #else
                    CHECK_MSK16_AND_ADD_PPART(msk[j], j, i);
                    #endif
                }
            }
        }

        #if AMX_BUCKET2_USE_BUFFER
        for (long i = 0; i < batchsize; i++) {
            local_buf_num[i] = _local_buf_num[i];
            if (reuse) local_buf_rnum[i] = _local_buf_rnum[i];
            local_buf_size[i] = _local_buf_size[i];
        }
        #endif
    } while (0);

    /////// PPART: combine the data to main buckets ///////
    #if AMX_BUCKET2_USE_BUFFER
    for (long i = 0; i < batchsize; i++) {
        bucket[i]->_alloc(local_buf_num[i], 1);
        bucket[i]->num_pvec = local_buf_num[i];
        if (reuse) {
            rbucket[i]->_alloc(local_buf_rnum[i], 1);
            rbucket[i]->num_pvec = local_buf_rnum[i];
        }
    }
    
    for (long j = 0; j < batchsize; j++) {
        bucket_amx_t *dst, *rdst;
        dst = bucket[j];
        if (reuse) rdst = rbucket[j];
        uint32_t *pind_dst = dst->pvec;
        int32_t *pnorm_dst = dst->pnorm;
        uint32_t *rpind_dst;
        int32_t *rpnorm_dst;
        if (reuse) {
            rpind_dst = rdst->pvec;
            rpnorm_dst = rdst->pnorm;
        } 

        uint32_t *src = local_buf[j];
        long src_num = local_buf_num[j];
        
        // ASSUME THE OUTPUT OF MALLOC AND REALLOC IS 8-ALIGNED
        while ((long) src % 64L != 0 && src_num) {
            uint32_t norm_with_hint = src[1];
            int32_t norm = norm_with_hint & 0x3fffffff;
            pind_dst[0] = src[0];
            pnorm_dst[0] = norm;
            pind_dst++;
            pnorm_dst++;
            if (reuse && norm_with_hint & 0x80000000) {
                rpind_dst[0] = src[0];
                rpnorm_dst[0] = norm;
                rpind_dst++;
                rpnorm_dst++;
            }
            src_num--;
            src += 2;
        }

        long ind = 0;
        while (ind < src_num - 255) {                
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 0);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 16);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 32);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 48);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 64);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 80);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 96);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 112);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 128);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 144);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 160);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 176);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 192);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 208);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 224);
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind + 240);
            ind += 256;
        }
        while (ind < src_num - 15) {
            UNPACK_AND_ADD_TO_BUCKETS_PPART(ind);
            ind += 16;
        }
        while (ind < src_num) {
            uint32_t norm_with_hint = src[ind * 2 + 1];
            int32_t norm = norm_with_hint & 0x7fffffff;
            pind_dst[0] = src[ind * 2];
            pnorm_dst[0] = norm;
            pind_dst++;
            pnorm_dst++;
            if (reuse && norm_with_hint & 0x80000000) {
                rpind_dst[0] = src[ind * 2];
                rpnorm_dst[0] = norm;
                rpind_dst++;
                rpnorm_dst++;
            }
            ind++;
        }
    }
    #endif

    /////// NPART: each thread collect vectors in local buffers ///////
    do {
        ///// prepare local buffers //////
        #if AMX_BUCKET2_USE_BUFFER
        uint32_t _local_buf_size[batchsize];
        uint32_t _local_buf_num[batchsize] = {};
        uint32_t _local_buf_rnum[batchsize] = {};
        uint32_t **_local_buf = local_buf;
        for (long i = 0; i < batchsize; i++) _local_buf_size[i] = local_buf_size[i];
        #endif

        const long begin_ind = 0;
        const long end_ind = main_bucket->num_nvec;

        #define CHECK_MSK16_AND_ADD_NPART(__msk, __ind, __cind16)                                               \
                                                                  do {                                          \
            while (__msk) {                                                                                     \
                int r = __builtin_ctz(__msk);                                                                   \
                __msk &= ~(1 << r);                                                                             \
                bucket[(__cind16) + r]->add_nvec(main_bucket->nvec[ind + (__ind)], norm_epi32[__ind]);          \
                if (reuse) {                                                                                    \
                    if (dst[((__ind) + (__cind16)) * 16 + r] < rbound[__ind]) {                                 \
                        rbucket[(__cind16) + r]->add_nvec(main_bucket->nvec[ind + (__ind)], norm_epi32[__ind]); \
                        if (-dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                            TRY_ADDA_AMX(center_ind_list[(__cind16) + r], main_bucket->nvec[ind + (__ind)]);    \
                        }                                                                                       \
                    }                                                                                           \
                } else {                                                                                        \
                    if (-dst[((__ind) + (__cind16)) * 16 + r] > center_norm[(__cind16) + r] + norm_epi32[__ind]) { \
                        TRY_ADDA_AMX(center_ind_list[(__cind16) + r], main_bucket->nvec[ind + (__ind)]);        \
                    }                                                                                           \
                }                                                                                               \
            }                                                                                                   \
        } while (0)

        #define CHECK_MSK16_PAIR_AND_ADD_NPART(__msk0, __msk1, __ind0, __ind1, __cind16) do {                   \
            if (!_mm512_kortestz(__msk0, __msk1)) {                                                             \
                CHECK_MSK16_AND_ADD_NPART(__msk0, __ind0, __cind16);                                            \
                CHECK_MSK16_AND_ADD_NPART(__msk1, __ind1, __cind16);                                            \
            }                                                                                                   \
        } while (0)

        long ind = begin_ind;
        while (ind < end_ind - 127) {
            int32_t *norm_epi32 = main_bucket->nnorm + ind;
            __attribute__ ((aligned (64))) int32_t _bound[64];
            __attribute__ ((aligned (64))) int32_t _rbound[64];
            __m512i norm_si512_0 = _mm512_loadu_si512(norm_epi32 + 0);
            __m512i norm_si512_1 = _mm512_loadu_si512(norm_epi32 + 16);
            __m512i norm_si512_2 = _mm512_loadu_si512(norm_epi32 + 32);
            __m512i norm_si512_3 = _mm512_loadu_si512(norm_epi32 + 48);
            _mm512_store_si512(_bound + 0, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_bx2_ps))));
            _mm512_store_si512(_bound + 16, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_bx2_ps))));
            _mm512_store_si512(_bound + 32, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_bx2_ps))));
            _mm512_store_si512(_bound + 48, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_bx2_ps))));
            if (reuse) {
                _mm512_store_si512(_rbound + 0, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_0), alpha_rx2_ps))));
                _mm512_store_si512(_rbound + 16, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_1), alpha_rx2_ps))));
                _mm512_store_si512(_rbound + 32, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_2), alpha_rx2_ps))));
                _mm512_store_si512(_rbound + 48, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512_3), alpha_rx2_ps))));
            }
            __attribute__ ((aligned (64))) int32_t _dst[64 * batchsize];
            int32_t *dst = _dst;
            int32_t *bound = _bound;
            int32_t *rbound = _rbound;

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec1[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec2[16 * vec_length + 32];
            __attribute__ ((aligned (64))) int8_t vec3[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 0 + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 0 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 0 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec1 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 16 + j]) * vec_length)));
                _mm512_storeu_si512(vec1 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 16 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec1 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 16 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec2 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 32 + j]) * vec_length)));
                _mm512_storeu_si512(vec2 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 32 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec2 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 32 + j]) * vec_length + 128)));
                _mm512_storeu_si512(vec3 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 48 + j]) * vec_length)));
                _mm512_storeu_si512(vec3 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + 48 + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec3 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + 48 + j]) * vec_length + 128)));
            }

            
            long prefetch_start_ind = (ind+64);
            constexpr int32_t n_prefetch_vec = 64;

            for (long i = 0; i < batchsize; i += 16) {
                for (long j = 0; j < 16 && i <= batchsize - 32; j++) {
                    _mm_prefetch(center0_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center1_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                    _mm_prefetch(center2_tr + (i + 16 + j) * 64, _MM_HINT_T0);
                }
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
                _tile_loadd(4, center0_tr + i * 64, 64);
                _tile_loadd(5, vec0, vec_length);
                _tile_loadd(6, vec1, vec_length);
                _tile_loadd(7, vec2, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3, vec_length);
                _tile_dpbssd(1, 6, 4);
                _tile_loadd(6, center1_tr + i * 64, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_loadd(7, vec0 + 64, vec_length);
                _tile_dpbssd(3, 5, 4);
                _tile_loadd(4, vec1 + 64, vec_length);
                _tile_loadd(5, vec2 + 64, vec_length);
                _tile_dpbssd(0, 7, 6);
                _tile_loadd(7, vec3 + 64, vec_length);
                _tile_dpbssd(1, 4, 6);
                _tile_loadd(4, center2_tr + i * 64, 64);
                _tile_dpbssd(2, 5, 6);
                _tile_loadd(5, vec0 + 128, vec_length);
                _tile_dpbssd(3, 7, 6);
                _tile_loadd(6, vec1 + 128, vec_length);
                _tile_loadd(7, vec2 + 128, vec_length);
                _tile_dpbssd(0, 5, 4);
                _tile_loadd(5, vec3 + 128, vec_length);
                _tile_stored(0, dst + 0 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(1, 6, 4);
                _tile_stored(1, dst + 1 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(2, 7, 4);
                _tile_stored(2, dst + 2 * 16 * batchsize + i * 16, 64);
                _tile_dpbssd(3, 5, 4);
                _tile_stored(3, dst + 3 * 16 * batchsize + i * 16, 64);
                for (long j = (n_prefetch_vec * i) / batchsize; j < (n_prefetch_vec * (i + 16)) / batchsize; j++) {
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->nvec[prefetch_start_ind + j], _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->nvec[prefetch_start_ind + j] + 64, _MM_HINT_T1);
                    _mm_prefetch(vec + (long)vec_length * (long)main_bucket->nvec[prefetch_start_ind + j] + 128, _MM_HINT_T1);
                }
            }

            for (long kk = 0; kk < 4; kk++) {
                __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
                __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
                __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
                __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
                __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
                __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
                __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
                __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
                __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
                __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
                __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
                __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
                __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
                __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
                __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
                __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

                for (long i = 0; i < batchsize; i += 16) {
                    __mmask16 msk0 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 0) * 16), bound0_si512, 1);
                    __mmask16 msk1 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 1) * 16), bound1_si512, 1);
                    __mmask16 msk2 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 2) * 16), bound2_si512, 1);
                    __mmask16 msk3 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 3) * 16), bound3_si512, 1);
                    __mmask16 msk4 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 4) * 16), bound4_si512, 1);
                    __mmask16 msk5 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 5) * 16), bound5_si512, 1);
                    __mmask16 msk6 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 6) * 16), bound6_si512, 1);
                    __mmask16 msk7 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 7) * 16), bound7_si512, 1);
                    __mmask16 msk8 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 8) * 16), bound8_si512, 1);
                    __mmask16 msk9 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 9) * 16), bound9_si512, 1);
                    __mmask16 mskA = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 10) * 16), boundA_si512, 1);
                    __mmask16 mskB = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 11) * 16), boundB_si512, 1);
                    __mmask16 mskC = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 12) * 16), boundC_si512, 1);
                    __mmask16 mskD = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 13) * 16), boundD_si512, 1);
                    __mmask16 mskE = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 14) * 16), boundE_si512, 1);
                    __mmask16 mskF = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 15) * 16), boundF_si512, 1);

                    #if AMX_BUCKET2_USE_BUFFER
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk0, msk1, 0, 1, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk2, msk3, 2, 3, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk4, msk5, 4, 5, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk6, msk7, 6, 7, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(msk8, msk9, 8, 9, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(mskA, mskB, 10, 11, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(mskC, mskD, 12, 13, i);
                    CHECK_MSK16_PAIR_AND_PACK_NPART(mskE, mskF, 14, 15, i);
                    #else
                    CHECK_MSK16_PAIR_AND_ADD_NPART(msk0, msk1, 0, 1, i);
                    CHECK_MSK16_PAIR_AND_ADD_NPART(msk2, msk3, 2, 3, i);
                    CHECK_MSK16_PAIR_AND_ADD_NPART(msk4, msk5, 4, 5, i);
                    CHECK_MSK16_PAIR_AND_ADD_NPART(msk6, msk7, 6, 7, i);
                    CHECK_MSK16_PAIR_AND_ADD_NPART(msk8, msk9, 8, 9, i);
                    CHECK_MSK16_PAIR_AND_ADD_NPART(mskA, mskB, 10, 11, i);
                    CHECK_MSK16_PAIR_AND_ADD_NPART(mskC, mskD, 12, 13, i);
                    CHECK_MSK16_PAIR_AND_ADD_NPART(mskE, mskF, 14, 15, i);
                    #endif
                }

                ind += 16;
                dst += 16 * batchsize;
                norm_epi32 += 16;
                bound += 16;
                if (reuse) rbound += 16;
            }
            #if AMX_BUCKET2_USE_BUFFER
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 64) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 64 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }
            #endif
        }
        while (ind < end_ind - 15) {
            int32_t *norm_epi32 = main_bucket->nnorm + ind;
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps))));
            if (reuse) _mm512_store_si512(rbound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps))));

            __m512i bound0_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 0)));
            __m512i bound1_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 1)));
            __m512i bound2_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 2)));
            __m512i bound3_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 3)));
            __m512i bound4_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 4)));
            __m512i bound5_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 5)));
            __m512i bound6_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 6)));
            __m512i bound7_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 7)));
            __m512i bound8_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 8)));
            __m512i bound9_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 9)));
            __m512i boundA_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 10)));
            __m512i boundB_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 11)));
            __m512i boundC_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 12)));
            __m512i boundD_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 13)));
            __m512i boundE_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 14)));
            __m512i boundF_si512 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + 15)));

            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];

            __attribute__ ((aligned (64))) int8_t vec0[16 * vec_length + 32];
            for (long j = 0; j < 16; j++) {
                _mm512_storeu_si512(vec0 + j * vec_length + 0, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + j]) * vec_length)));
                _mm512_storeu_si512(vec0 + j * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[ind + j]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(vec0 + j * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[ind + j]) * vec_length + 128)));
            }

            _tile_loadd(2, vec0, vec_length);
            _tile_loadd(3, vec0 + 64, vec_length);
            _tile_loadd(4, vec0 + 128, vec_length);

            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            
            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk0 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 0) * 16), bound0_si512, 1);
                __mmask16 msk1 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 1) * 16), bound1_si512, 1);
                __mmask16 msk2 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 2) * 16), bound2_si512, 1);
                __mmask16 msk3 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 3) * 16), bound3_si512, 1);
                __mmask16 msk4 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 4) * 16), bound4_si512, 1);
                __mmask16 msk5 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 5) * 16), bound5_si512, 1);
                __mmask16 msk6 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 6) * 16), bound6_si512, 1);
                __mmask16 msk7 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 7) * 16), bound7_si512, 1);
                __mmask16 msk8 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 8) * 16), bound8_si512, 1);
                __mmask16 msk9 = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 9) * 16), bound9_si512, 1);
                __mmask16 mskA = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 10) * 16), boundA_si512, 1);
                __mmask16 mskB = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 11) * 16), boundB_si512, 1);
                __mmask16 mskC = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 12) * 16), boundC_si512, 1);
                __mmask16 mskD = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 13) * 16), boundD_si512, 1);
                __mmask16 mskE = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 14) * 16), boundE_si512, 1);
                __mmask16 mskF = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + 15) * 16), boundF_si512, 1);

                #if AMX_BUCKET2_USE_BUFFER
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk0, msk1, 0, 1, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk2, msk3, 2, 3, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk4, msk5, 4, 5, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk6, msk7, 6, 7, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(msk8, msk9, 8, 9, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(mskA, mskB, 10, 11, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(mskC, mskD, 12, 13, i);
                CHECK_MSK16_PAIR_AND_PACK_NPART(mskE, mskF, 14, 15, i);
                #else
                CHECK_MSK16_PAIR_AND_ADD_NPART(msk0, msk1, 0, 1, i);
                CHECK_MSK16_PAIR_AND_ADD_NPART(msk2, msk3, 2, 3, i);
                CHECK_MSK16_PAIR_AND_ADD_NPART(msk4, msk5, 4, 5, i);
                CHECK_MSK16_PAIR_AND_ADD_NPART(msk6, msk7, 6, 7, i);
                CHECK_MSK16_PAIR_AND_ADD_NPART(msk8, msk9, 8, 9, i);
                CHECK_MSK16_PAIR_AND_ADD_NPART(mskA, mskB, 10, 11, i);
                CHECK_MSK16_PAIR_AND_ADD_NPART(mskC, mskD, 12, 13, i);
                CHECK_MSK16_PAIR_AND_ADD_NPART(mskE, mskF, 14, 15, i);
                #endif
            }
            
            #if AMX_BUCKET2_USE_BUFFER
            for (long i = 0; i < batchsize; i++) {
                if (_local_buf_num[i] > _local_buf_size[i] - batchsize * 16) {
                    _local_buf_size[i] *= 2;
                    _local_buf_size[i] += 16 * batchsize;
                    _local_buf[i] = (uint32_t *) realloc(_local_buf[i], _local_buf_size[i] * 2 * sizeof(uint32_t));
                }
            }
            #endif

            ind += 16;
        }
        
        if (ind < end_ind) {
            __attribute__ ((aligned (64))) int32_t norm_epi32[16];
            __attribute__ ((aligned (64))) int32_t bound[16];
            __attribute__ ((aligned (64))) int32_t rbound[16];
            for (long i = ind; i < end_ind; i++) norm_epi32[i - ind] = main_bucket->nnorm[i];
            __m512i norm_si512 = _mm512_loadu_si512(norm_epi32);
            _mm512_store_si512(bound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_bx2_ps))));
            if (reuse) _mm512_store_si512(rbound, _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(norm_si512), alpha_rx2_ps))));

            __m512i bound_si512[16];
            for (long i = 0; i < end_ind - ind; i++) bound_si512[i] = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(bound + i)));
            
            __attribute__ ((aligned (64))) int32_t dst[16 * batchsize];
            __attribute__ ((aligned (64))) int8_t tmp[16 * vec_length + 32];
            for (long i = ind; i < end_ind; i++) {
                _mm512_storeu_si512(tmp + (i-ind) * vec_length, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[i]) * vec_length)));
                _mm512_storeu_si512(tmp + (i-ind) * vec_length + 64, _mm512_loadu_si512((__m512i *)(vec + (long)(main_bucket->nvec[i]) * vec_length + 64)));
                _mm256_store_si256((__m256i *)(tmp + (i-ind) * vec_length + 128), _mm256_load_si256((__m256i *)(vec + (long)(main_bucket->nvec[i]) * vec_length + 128)));
            }

            _tile_loadd(2, tmp, vec_length);
            _tile_loadd(3, tmp + 64, vec_length);
            _tile_loadd(4, tmp + 128, vec_length);
            for (long i = 0; i < batchsize; i += 32) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_loadd(5, center0_tr + i * 64, 64);
                _tile_loadd(6, center0_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 2, 5);
                _tile_dpbssd(1, 2, 6);
                _tile_loadd(7, center1_tr + i * 64, 64);
                _tile_loadd(5, center1_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 3, 7);
                _tile_dpbssd(1, 3, 5);
                _tile_loadd(6, center2_tr + i * 64, 64);
                _tile_loadd(7, center2_tr + i * 64 + 1024, 64);
                _tile_dpbssd(0, 4, 6);
                _tile_dpbssd(1, 4, 7);
                _tile_stored(0, dst + i * 16, 64);
                _tile_stored(1, dst + i * 16 + 256, 64);
            }

            for (long i = 0; i < batchsize; i += 16) {
                __mmask16 msk[16];
                for (long j = 0; j < end_ind - ind; j++) {
                    msk[j] = _mm512_cmp_epi32_mask(_mm512_load_si512(dst + (i + j) * 16), bound_si512[j], 1);
                }
                for (long j = 0; j < end_ind - ind; j++) {
                    #if AMX_BUCKET2_USE_BUFFER
                    CHECK_MSK16_AND_PACK_NPART(msk[j], j, i);
                    #else
                    CHECK_MSK16_AND_ADD_NPART(msk[j], j, i);
                    #endif
                }
            }
        }

        #if AMX_BUCKET2_USE_BUFFER
        for (long i = 0; i < batchsize; i++) {
            local_buf_num[i] = _local_buf_num[i];
            if (reuse) local_buf_rnum[i] = _local_buf_rnum[i];
            local_buf_size[i] = _local_buf_size[i];
        }
        #endif
    } while (0);

    /////// NPART: combine the data to main buckets ///////
    #if AMX_BUCKET2_USE_BUFFER
    for (long i = 0; i < batchsize; i++) {
        bucket[i]->_alloc(local_buf_num[i], 0);
        bucket[i]->num_nvec = local_buf_num[i];
        if (reuse) {
            rbucket[i]->_alloc(local_buf_rnum[i], 0);
            rbucket[i]->num_nvec = local_buf_rnum[i];
        }
    }
    for (long j = 0; j < batchsize; j++) {
        bucket_amx_t *dst, *rdst;
        dst = bucket[j];
        if (reuse) rdst = rbucket[j];
        uint32_t *nind_dst = dst->nvec;
        int32_t *nnorm_dst = dst->nnorm;
        uint32_t *rnind_dst;
        int32_t *rnnorm_dst;
        if (reuse) {
            rnind_dst = rdst->nvec;
            rnnorm_dst = rdst->nnorm;
        } 

        uint32_t *src = local_buf[j];
        long src_num = local_buf_num[j];
        
        // ASSUME THE OUTPUT OF MALLOC AND REALLOC IS 8-ALIGNED
        while ((long) src % 64L != 0 && src_num) {
            uint32_t norm_with_hint = src[1];
            int32_t norm = norm_with_hint & 0x3fffffff;
            nind_dst[0] = src[0];
            nnorm_dst[0] = norm;
            nind_dst++;
            nnorm_dst++;
            if (reuse && norm_with_hint & 0x80000000) {
                rnind_dst[0] = src[0];
                rnnorm_dst[0] = norm;
                rnind_dst++;
                rnnorm_dst++;
            }
            src_num--;
            src += 2;
        }

        long ind = 0;
        while (ind < src_num - 255) {
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 0);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 16);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 32);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 48);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 64);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 80);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 96);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 112);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 128);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 144);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 160);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 176);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 192);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 208);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 224);
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind + 240);
            ind += 256;
        }
        while (ind < src_num - 15) {
            UNPACK_AND_ADD_TO_BUCKETS_NPART(ind);
            ind += 16;
        }
        while (ind < src_num) {
            uint32_t norm_with_hint = src[ind * 2 + 1];
            int32_t norm = norm_with_hint & 0x7fffffff;
            nind_dst[0] = src[ind * 2];
            nnorm_dst[0] = norm;
            nind_dst++;
            nnorm_dst++;
            if (reuse && norm_with_hint & 0x80000000) {
                rnind_dst[0] = src[ind * 2];
                rnnorm_dst[0] = norm;
                rnind_dst++;
                rnnorm_dst++;
            }
            ind++;
        }
    }

    /////// free local buffer and center ///////
    for (long i = 0; i < batchsize; i++) {
        free(local_buf[i]);
    }
    #endif
    FREE_VEC((void *)center0_tr);
    FREE_VEC((void *)center1_tr);
    FREE_VEC((void *)center2_tr);

    /////// remove duplicates ///////
    for (long i = 0; i < batchsize; i++) {
        for (long j = 0; j < i; j++) {
            if (center_ind_list[i] == center_ind_list[j]) {
                if (reuse) rbucket[i]->num_pvec = 0;
                if (reuse) rbucket[i]->num_nvec = 0;
                bucket[i]->num_pvec = 0;
                bucket[i]->num_nvec = 0;
                break;
            }
        }
    }

    if (prof) {
        // pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->succ_add2 += succ_add2;
        // pthread_spin_unlock(&prof->profile_lock);
    }
    return 0;
}


// WARNING! WE DO NOT FREE THESE MEMORIES
static int8_t *_amx_search_buf0[AMX_MAX_NTHREADS] = {};
static int8_t *_amx_search_buf1[AMX_MAX_NTHREADS] = {};
static int8_t *_amx_search_buf2[AMX_MAX_NTHREADS] = {};
static int8_t *_amx_search_buf0_tr[AMX_MAX_NTHREADS] = {};
static int8_t *_amx_search_buf1_tr[AMX_MAX_NTHREADS] = {};
static int8_t *_amx_search_buf2_tr[AMX_MAX_NTHREADS] = {};
static int32_t *_amx_search_norm[AMX_MAX_NTHREADS] = {};

static uint64_t padded_amx_search_buf_size[8 * AMX_MAX_NTHREADS] = {};

template <uint32_t nb>
int Pool_epi8_t<nb>::_search_amx(bucket_amx_t *bkt, sol_list_amx_t *sol, int32_t goal_norm, bgj_amx_profile_data_t<nb> *prof) {
    uint64_t try_add2 = 0;
    uint64_t succ_add2 = 0;
    const long nvec = bkt->num_nvec + bkt->num_pvec;
    TILE_INITIALIZE;

    long thread = omp_get_thread_num();
    if (thread < 0 || thread >= AMX_MAX_NTHREADS) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::_search_amx: thread id (%ld) is out of range [0, %d)\n", nb, thread, AMX_MAX_NTHREADS);
    }
    if (padded_amx_search_buf_size[thread * 8] < _CEIL16(nvec)) {
        long target_buf_size = 1.5 * (_CEIL16(nvec));
        target_buf_size = _CEIL16(target_buf_size);
        if (_amx_search_buf0[thread]) FREE_VEC((void *) _amx_search_buf0[thread]);
        if (_amx_search_buf1[thread]) FREE_VEC((void *) _amx_search_buf1[thread]);
        if (_amx_search_buf2[thread]) FREE_VEC((void *) _amx_search_buf2[thread]);
        if (_amx_search_buf0_tr[thread]) FREE_VEC((void *) _amx_search_buf0_tr[thread]);
        if (_amx_search_buf1_tr[thread]) FREE_VEC((void *) _amx_search_buf1_tr[thread]);
        if (_amx_search_buf2_tr[thread]) FREE_VEC((void *) _amx_search_buf2_tr[thread]);
        if (_amx_search_norm[thread]) FREE_VEC((void *) _amx_search_norm[thread]);
        _amx_search_buf0[thread] = (int8_t *) NEW_VEC(64 * target_buf_size, sizeof(int8_t));
        _amx_search_buf1[thread] = (int8_t *) NEW_VEC(64 * target_buf_size, sizeof(int8_t));
        _amx_search_buf2[thread] = (int8_t *) NEW_VEC(32 * target_buf_size + 32, sizeof(int8_t));
        _amx_search_buf0_tr[thread] = (int8_t *) NEW_VEC(64 * target_buf_size, sizeof(int8_t));
        _amx_search_buf1_tr[thread] = (int8_t *) NEW_VEC(64 * target_buf_size, sizeof(int8_t));
        _amx_search_buf2_tr[thread] = (int8_t *) NEW_VEC(64 * target_buf_size, sizeof(int8_t));
        _amx_search_norm[thread] = (int32_t *) NEW_VEC(target_buf_size, sizeof(int32_t));
        padded_amx_search_buf_size[thread * 8] = target_buf_size;
    }
    int8_t *buf0 = _amx_search_buf0[thread];
    int8_t *buf1 = _amx_search_buf1[thread];
    int8_t *buf2 = _amx_search_buf2[thread];
    int8_t *buf0_tr = _amx_search_buf0_tr[thread];
    int8_t *buf1_tr = _amx_search_buf1_tr[thread];
    int8_t *buf2_tr = _amx_search_buf2_tr[thread];
    int32_t *norm = _amx_search_norm[thread];

    __attribute__ ((aligned (64))) int32_t dst[1024];
    __attribute__ ((aligned (64))) int32_t th_tmp[16];
    __m512i gn_si512 = _mm512_set1_epi32(goal_norm);
    __m512i tail_th[15];
    __m512i th0, th1, th2, th3, th4, th5, th6, th7, th8, th9, tha, thb, thc, thd, the, thf;

    #define RELOAD_THRESHOLD(__jnptr)                                                           \
                                      do {                                                      \
        _mm512_store_si512(th_tmp, _mm512_sub_epi32(_mm512_load_si512((__jnptr)), gn_si512));   \
        th0 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 0)));      \
        th1 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 1)));      \
        th2 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 2)));      \
        th3 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 3)));      \
        th4 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 4)));      \
        th5 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 5)));      \
        th6 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 6)));      \
        th7 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 7)));      \
        th8 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 8)));      \
        th9 = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 9)));      \
        tha = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 10)));     \
        thb = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 11)));     \
        thc = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 12)));     \
        thd = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 13)));     \
        the = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 14)));     \
        thf = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + 15)));     \
    } while (0)

    #define TILE_DP160x1(__ind, __jnd) do {                     \
        _tile_zero(0);                                          \
        _tile_loadd(3, _buf0 + (__jnd) * 64, 64);               \
        _tile_loadd(4, _buf0_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_dpbssd(0, 3, 4);                                  \
        _tile_loadd(5, _buf1 + (__jnd) * 64, 64);               \
        _tile_loadd(6, _buf1_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_dpbssd(0, 5, 6);                                  \
        _tile_loadd(7, _buf2 + (__jnd) * 32, 32);               \
        _tile_loadd(3, _buf2_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_dpbssd(0, 7, 3);                                  \
        _tile_stored(0, dst + 0 * 256, 64);                     \
    } while (0)

    #define TILE_DP160x2(__ind, __jnd) do {                     \
        _tile_zero(1);                                          \
        _tile_zero(2);                                          \
        _tile_loadd(4, _buf0 + (__jnd) * 64, 64);               \
        _tile_loadd(5, _buf0_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_loadd(6, _buf0_tr + ((__ind) + 16 * 1) * 64, 64); \
        _tile_dpbssd(1, 4, 5);                                  \
        _tile_loadd(7, _buf1 + (__jnd) * 64, 64);               \
        _tile_loadd(3, _buf1_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_loadd(0, _buf1_tr + ((__ind) + 16 * 1) * 64, 64); \
        _tile_dpbssd(2, 4, 6);                                  \
        _tile_loadd(5, _buf2 + (__jnd) * 32, 32);               \
        _tile_dpbssd(1, 7, 3);                                  \
        _tile_loadd(6, _buf2_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_dpbssd(2, 7, 0);                                  \
        _tile_loadd(4, _buf2_tr + ((__ind) + 16 * 1) * 64, 64); \
        _tile_dpbssd(1, 5, 6);                                  \
        _tile_stored(1, dst + 1 * 256, 64);                     \
        _tile_dpbssd(2, 5, 4);                                  \
        _tile_stored(2, dst + 2 * 256, 64);                     \
    } while (0)

    #define TILE_DP160x3(__ind, __jnd) do {                     \
        _tile_zero(0);                                          \
        _tile_zero(1);                                          \
        _tile_zero(2);                                          \
        _tile_loadd(3, _buf0 + (__jnd) * 64, 64);               \
        _tile_loadd(4, _buf0_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_loadd(5, _buf0_tr + ((__ind) + 16 * 1) * 64, 64); \
        _tile_loadd(6, _buf0_tr + ((__ind) + 16 * 2) * 64, 64); \
        _tile_dpbssd(0, 3, 4);                                  \
        _tile_loadd(7, _buf1 + (__jnd) * 64, 64);               \
        _tile_dpbssd(1, 3, 5);                                  \
        _tile_loadd(4, _buf1_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_dpbssd(2, 3, 6);                                  \
        _tile_loadd(5, _buf1_tr + ((__ind) + 16 * 1) * 64, 64); \
        _tile_loadd(6, _buf1_tr + ((__ind) + 16 * 2) * 64, 64); \
        _tile_loadd(3, _buf2 + (__jnd) * 32, 32);               \
        _tile_dpbssd(0, 7, 4);                                  \
        _tile_loadd(4, _buf2_tr + ((__ind) + 16 * 0) * 64, 64); \
        _tile_dpbssd(1, 7, 5);                                  \
        _tile_loadd(5, _buf2_tr + ((__ind) + 16 * 1) * 64, 64); \
        _tile_dpbssd(2, 7, 6);                                  \
        _tile_loadd(6, _buf2_tr + ((__ind) + 16 * 2) * 64, 64); \
        _tile_dpbssd(0, 3, 4);                                  \
        _tile_stored(0, dst + 0 * 256, 64);                     \
        _tile_dpbssd(1, 3, 5);                                  \
        _tile_stored(1, dst + 1 * 256, 64);                     \
        _tile_dpbssd(2, 3, 6);                                  \
        _tile_stored(2, dst + 2 * 256, 64);                     \
    } while (0)

    #define TILE_DP160x4(__ind, __jnd)                              \
                                       do {                         \
        _tile_zero(0);                                              \
        _tile_zero(1);                                              \
        _tile_zero(2);                                              \
        _tile_zero(3);                                              \
        _tile_loadd(4, _buf0 + (__jnd) * 64, 64);                   \
        _tile_loadd(5, _buf0_tr + ((__ind) + 16 * 0) * 64, 64);     \
        _tile_loadd(6, _buf0_tr + ((__ind) + 16 * 1) * 64, 64);     \
        _tile_loadd(7, _buf0_tr + ((__ind) + 16 * 2) * 64, 64);     \
        _tile_dpbssd(0, 4, 5);                                      \
        _tile_loadd(5, _buf0_tr + ((__ind) + 16 * 3) * 64, 64);     \
        _tile_dpbssd(1, 4, 6);                                      \
        _tile_loadd(6, _buf1 + (__jnd) * 64, 64);                   \
        _tile_dpbssd(2, 4, 7);                                      \
        _tile_loadd(7, _buf1_tr + ((__ind) + 16 * 0) * 64, 64);     \
        _tile_dpbssd(3, 4, 5);                                      \
        _tile_loadd(4, _buf1_tr + ((__ind) + 16 * 1) * 64, 64);     \
        _tile_loadd(5, _buf1_tr + ((__ind) + 16 * 2) * 64, 64);     \
        _tile_dpbssd(0, 6, 7);                                      \
        _tile_loadd(7, _buf1_tr + ((__ind) + 16 * 3) * 64, 64);     \
        _tile_dpbssd(1, 6, 4);                                      \
        _tile_loadd(4, _buf2 + (__jnd) * 32, 32);                   \
        _tile_dpbssd(2, 6, 5);                                      \
        _tile_loadd(5, _buf2_tr + ((__ind) + 16 * 0) * 64, 64);     \
        _tile_dpbssd(3, 6, 7);                                      \
        _tile_loadd(6, _buf2_tr + ((__ind) + 16 * 1) * 64, 64);     \
        _tile_loadd(7, _buf2_tr + ((__ind) + 16 * 2) * 64, 64);     \
        _tile_dpbssd(0, 4, 5);                                      \
        _tile_loadd(5, _buf2_tr + ((__ind) + 16 * 3) * 64, 64);     \
        _tile_stored(0, dst + 0 * 256, 64);                         \
        _tile_dpbssd(1, 4, 6);                                      \
        _tile_stored(1, dst + 1 * 256, 64);                         \
        _tile_dpbssd(2, 4, 7);                                      \
        _tile_stored(2, dst + 2 * 256, 64);                         \
        _tile_dpbssd(3, 4, 5);                                      \
        _tile_stored(3, dst + 3 * 256, 64);                         \
    } while (0)

    #define CHECK_AND_ADD(_msk, _ind, _jnd)                                                     \
                                            do {                                                \
        while (_msk) {                                                                          \
            int r = __builtin_ctz(_msk);                                                        \
            _msk &= _msk - 1;                                                                   \
            int _jdx = (_jnd);                                                                  \
            int _idx = (_ind) + r;                                                              \
            if (_jdx < bkt->num_pvec) {                                                         \
                TRY_ADDS_AMX(bkt->pvec[_jdx], bkt->pvec[_idx]);                                 \
            } else if (_idx >= bkt->num_pvec) {                                                 \
                TRY_ADDS_AMX(bkt->nvec[_jdx - bkt->num_pvec], bkt->nvec[_idx - bkt->num_pvec]); \
            } else {                                                                            \
                TRY_ADDA_AMX(bkt->nvec[_jdx - bkt->num_pvec], bkt->pvec[_idx]);                 \
            }                                                                                   \
        }                                                                                       \
    } while (0)

    #define CHECK_AND_ADD_DIAG(_msk, _ind, _jnd)                                                \
                                            do {                                                \
        while (_msk) {                                                                          \
            int r = __builtin_ctz(_msk);                                                        \
            _msk &= _msk - 1;                                                                   \
            int _jdx = (_jnd);                                                                  \
            int _idx = (_ind) + r;                                                              \
            if (_idx < bkt->num_pvec) {                                                         \
                TRY_ADDS_AMX(bkt->pvec[_jdx], bkt->pvec[_idx]);                                 \
            } else if (_jdx >= bkt->num_pvec) {                                                 \
                TRY_ADDS_AMX(bkt->nvec[_jdx - bkt->num_pvec], bkt->nvec[_idx - bkt->num_pvec]); \
            } else {                                                                            \
                TRY_ADDA_AMX(bkt->nvec[_idx - bkt->num_pvec], bkt->pvec[_jdx]);                 \
            }                                                                                   \
        }                                                                                       \
    } while (0)

    #define CHECK_AND_MASKADD(__dst, __ind, __jnd, __msk)         \
                                                            do {  \
        __m512i __in_si512 = _mm512_load_si512(_inorm + (__ind)); \
        __m256i all_one = _mm256_cmpeq_epi16(_mm512_castsi512_si256(th0), _mm512_castsi512_si256(th0)); \
        __m256i all_zero = _mm256_setzero_si256();  \
        __m256i acc0 = all_zero;    \
        __m256i acc1 = all_zero;    \
        __m256i acc2 = all_zero;    \
        __m256i acc3 = all_zero;    \
                                    \
        if ((__msk) == 0xffff) {    \
            acc0 = _mm256_mask_add_epi16(acc0, _mm512_cmp_epi32_mask(_mm512_add_epi32(th0, __in_si512), _mm512_load_si512((__dst) + 0 * 16), 1), acc0, all_one);    \
            acc1 = _mm256_mask_add_epi16(acc1, _mm512_cmp_epi32_mask(_mm512_add_epi32(th1, __in_si512), _mm512_load_si512((__dst) + 1 * 16), 1), acc1, all_one);    \
            acc2 = _mm256_mask_add_epi16(acc2, _mm512_cmp_epi32_mask(_mm512_add_epi32(th2, __in_si512), _mm512_load_si512((__dst) + 2 * 16), 1), acc2, all_one);    \
            acc3 = _mm256_mask_add_epi16(acc3, _mm512_cmp_epi32_mask(_mm512_add_epi32(th3, __in_si512), _mm512_load_si512((__dst) + 3 * 16), 1), acc3, all_one);    \
            acc0 = _mm256_mask_add_epi16(acc0, _mm512_cmp_epi32_mask(_mm512_add_epi32(th4, __in_si512), _mm512_load_si512((__dst) + 4 * 16), 1), acc0, all_one);    \
            acc1 = _mm256_mask_add_epi16(acc1, _mm512_cmp_epi32_mask(_mm512_add_epi32(th5, __in_si512), _mm512_load_si512((__dst) + 5 * 16), 1), acc1, all_one);    \
            acc2 = _mm256_mask_add_epi16(acc2, _mm512_cmp_epi32_mask(_mm512_add_epi32(th6, __in_si512), _mm512_load_si512((__dst) + 6 * 16), 1), acc2, all_one);    \
            acc3 = _mm256_mask_add_epi16(acc3, _mm512_cmp_epi32_mask(_mm512_add_epi32(th7, __in_si512), _mm512_load_si512((__dst) + 7 * 16), 1), acc3, all_one);    \
            acc0 = _mm256_mask_add_epi16(acc0, _mm512_cmp_epi32_mask(_mm512_add_epi32(th8, __in_si512), _mm512_load_si512((__dst) + 8 * 16), 1), acc0, all_one);    \
            acc1 = _mm256_mask_add_epi16(acc1, _mm512_cmp_epi32_mask(_mm512_add_epi32(th9, __in_si512), _mm512_load_si512((__dst) + 9 * 16), 1), acc1, all_one);    \
            acc2 = _mm256_mask_add_epi16(acc2, _mm512_cmp_epi32_mask(_mm512_add_epi32(tha, __in_si512), _mm512_load_si512((__dst) + 10 * 16), 1), acc2, all_one);   \
            acc3 = _mm256_mask_add_epi16(acc3, _mm512_cmp_epi32_mask(_mm512_add_epi32(thb, __in_si512), _mm512_load_si512((__dst) + 11 * 16), 1), acc3, all_one);   \
            acc0 = _mm256_mask_add_epi16(acc0, _mm512_cmp_epi32_mask(_mm512_add_epi32(thc, __in_si512), _mm512_load_si512((__dst) + 12 * 16), 1), acc0, all_one);   \
            acc1 = _mm256_mask_add_epi16(acc1, _mm512_cmp_epi32_mask(_mm512_add_epi32(thd, __in_si512), _mm512_load_si512((__dst) + 13 * 16), 1), acc1, all_one);   \
            acc2 = _mm256_mask_add_epi16(acc2, _mm512_cmp_epi32_mask(_mm512_add_epi32(the, __in_si512), _mm512_load_si512((__dst) + 14 * 16), 1), acc2, all_one);   \
            acc3 = _mm256_mask_add_epi16(acc3, _mm512_cmp_epi32_mask(_mm512_add_epi32(thf, __in_si512), _mm512_load_si512((__dst) + 15 * 16), 1), acc3, all_one);   \
            acc0 = _mm256_or_si256(acc0, acc1); \
            acc2 = _mm256_or_si256(acc2, acc3); \
            acc0 = _mm256_or_si256(acc0, acc2); \
            if (!_mm256_testz_si256(acc0, acc0)) {   \
                __mmask16 msk0 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th0, __in_si512), _mm512_load_si512((__dst) + 0 * 16), 1);  \
                __mmask16 msk1 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th1, __in_si512), _mm512_load_si512((__dst) + 1 * 16), 1);  \
                __mmask16 msk2 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th2, __in_si512), _mm512_load_si512((__dst) + 2 * 16), 1);  \
                __mmask16 msk3 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th3, __in_si512), _mm512_load_si512((__dst) + 3 * 16), 1);  \
                CHECK_AND_ADD(msk0, (Ind + (__ind)), (Jnd + (__jnd) + 0));  \
                CHECK_AND_ADD(msk1, (Ind + (__ind)), (Jnd + (__jnd) + 1));  \
                CHECK_AND_ADD(msk2, (Ind + (__ind)), (Jnd + (__jnd) + 2));  \
                CHECK_AND_ADD(msk3, (Ind + (__ind)), (Jnd + (__jnd) + 3));  \
                __mmask16 msk4 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th4, __in_si512), _mm512_load_si512((__dst) + 4 * 16), 1);  \
                __mmask16 msk5 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th5, __in_si512), _mm512_load_si512((__dst) + 5 * 16), 1);  \
                __mmask16 msk6 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th6, __in_si512), _mm512_load_si512((__dst) + 6 * 16), 1);  \
                __mmask16 msk7 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th7, __in_si512), _mm512_load_si512((__dst) + 7 * 16), 1);  \
                CHECK_AND_ADD(msk4, (Ind + (__ind)), (Jnd + (__jnd) + 4));  \
                CHECK_AND_ADD(msk5, (Ind + (__ind)), (Jnd + (__jnd) + 5));  \
                CHECK_AND_ADD(msk6, (Ind + (__ind)), (Jnd + (__jnd) + 6));  \
                CHECK_AND_ADD(msk7, (Ind + (__ind)), (Jnd + (__jnd) + 7));  \
                __mmask16 msk8 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th8, __in_si512), _mm512_load_si512((__dst) + 8 * 16), 1);  \
                __mmask16 msk9 = _mm512_cmp_epi32_mask(_mm512_add_epi32(th9, __in_si512), _mm512_load_si512((__dst) + 9 * 16), 1);  \
                __mmask16 mska = _mm512_cmp_epi32_mask(_mm512_add_epi32(tha, __in_si512), _mm512_load_si512((__dst) + 10 * 16), 1); \
                __mmask16 mskb = _mm512_cmp_epi32_mask(_mm512_add_epi32(thb, __in_si512), _mm512_load_si512((__dst) + 11 * 16), 1); \
                CHECK_AND_ADD(msk8, (Ind + (__ind)), (Jnd + (__jnd) + 8));  \
                CHECK_AND_ADD(msk9, (Ind + (__ind)), (Jnd + (__jnd) + 9));  \
                CHECK_AND_ADD(mska, (Ind + (__ind)), (Jnd + (__jnd) + 10)); \
                CHECK_AND_ADD(mskb, (Ind + (__ind)), (Jnd + (__jnd) + 11)); \
                __mmask16 mskc = _mm512_cmp_epi32_mask(_mm512_add_epi32(thc, __in_si512), _mm512_load_si512((__dst) + 12 * 16), 1); \
                __mmask16 mskd = _mm512_cmp_epi32_mask(_mm512_add_epi32(thd, __in_si512), _mm512_load_si512((__dst) + 13 * 16), 1); \
                __mmask16 mske = _mm512_cmp_epi32_mask(_mm512_add_epi32(the, __in_si512), _mm512_load_si512((__dst) + 14 * 16), 1); \
                __mmask16 mskf = _mm512_cmp_epi32_mask(_mm512_add_epi32(thf, __in_si512), _mm512_load_si512((__dst) + 15 * 16), 1); \
                CHECK_AND_ADD(mskc, (Ind + (__ind)), (Jnd + (__jnd) + 12)); \
                CHECK_AND_ADD(mskd, (Ind + (__ind)), (Jnd + (__jnd) + 13)); \
                CHECK_AND_ADD(mske, (Ind + (__ind)), (Jnd + (__jnd) + 14)); \
                CHECK_AND_ADD(mskf, (Ind + (__ind)), (Jnd + (__jnd) + 15)); \
            }       \
        } else {    \
            const long _nrem = nvec - Jnd - (__jnd); \
            long __i = 0;   \
            while (__i < _nrem - 3) {    \
                __mmask16 msk0 = _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i], __in_si512), _mm512_load_si512((__dst) + __i * 16), 1);    \
                __mmask16 msk1 = _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i + 1], __in_si512), _mm512_load_si512((__dst) + (__i + 1) * 16), 1);    \
                __mmask16 msk2 = _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i + 2], __in_si512), _mm512_load_si512((__dst) + (__i + 2) * 16), 1);    \
                __mmask16 msk3 = _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i + 3], __in_si512), _mm512_load_si512((__dst) + (__i + 3) * 16), 1);    \
                CHECK_AND_ADD(msk0, (Ind + (__ind)), (Jnd + (__jnd) + __i + 0));  \
                CHECK_AND_ADD(msk1, (Ind + (__ind)), (Jnd + (__jnd) + __i + 1));  \
                CHECK_AND_ADD(msk2, (Ind + (__ind)), (Jnd + (__jnd) + __i + 2));  \
                CHECK_AND_ADD(msk3, (Ind + (__ind)), (Jnd + (__jnd) + __i + 3));  \
                __i += 4;   \
            }   \
            while (__i < _nrem) {    \
                __mmask16 msk0 = _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i], __in_si512), _mm512_load_si512((__dst) + __i * 16), 1);    \
                CHECK_AND_ADD(msk0, (Ind + (__ind)), (Jnd + (__jnd) + __i));  \
                __i++;  \
            }   \
        }   \
    } while (0)

    #define CHECK_AND_MASKADD_DIAG(__dst, __ind, __jnd, __msk)         \
                                                            do {  \
        __m512i __in_si512 = _mm512_load_si512(_inorm + (__ind)); \
        __m256i all_one = _mm256_cmpeq_epi16(_mm512_castsi512_si256(th0), _mm512_castsi512_si256(th0)); \
        __m256i all_zero = _mm256_setzero_si256();  \
        __m256i acc0 = all_zero;    \
        __m256i acc1 = all_zero;    \
        __m256i acc2 = all_zero;    \
        __m256i acc3 = all_zero;    \
                                    \
        if ((__msk) == 0xffff) {    \
            acc0 = _mm256_mask_add_epi16(acc0, 0xfffe & _mm512_cmp_epi32_mask(_mm512_add_epi32(th0, __in_si512), _mm512_load_si512((__dst) + 0 * 16), 1), acc0, all_one);    \
            acc1 = _mm256_mask_add_epi16(acc1, 0xfffc & _mm512_cmp_epi32_mask(_mm512_add_epi32(th1, __in_si512), _mm512_load_si512((__dst) + 1 * 16), 1), acc1, all_one);    \
            acc2 = _mm256_mask_add_epi16(acc2, 0xfff8 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th2, __in_si512), _mm512_load_si512((__dst) + 2 * 16), 1), acc2, all_one);    \
            acc3 = _mm256_mask_add_epi16(acc3, 0xfff0 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th3, __in_si512), _mm512_load_si512((__dst) + 3 * 16), 1), acc3, all_one);    \
            acc0 = _mm256_mask_add_epi16(acc0, 0xffe0 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th4, __in_si512), _mm512_load_si512((__dst) + 4 * 16), 1), acc0, all_one);    \
            acc1 = _mm256_mask_add_epi16(acc1, 0xffc0 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th5, __in_si512), _mm512_load_si512((__dst) + 5 * 16), 1), acc1, all_one);    \
            acc2 = _mm256_mask_add_epi16(acc2, 0xff80 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th6, __in_si512), _mm512_load_si512((__dst) + 6 * 16), 1), acc2, all_one);    \
            acc3 = _mm256_mask_add_epi16(acc3, 0xff00 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th7, __in_si512), _mm512_load_si512((__dst) + 7 * 16), 1), acc3, all_one);    \
            acc0 = _mm256_mask_add_epi16(acc0, 0xfe00 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th8, __in_si512), _mm512_load_si512((__dst) + 8 * 16), 1), acc0, all_one);    \
            acc1 = _mm256_mask_add_epi16(acc1, 0xfc00 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th9, __in_si512), _mm512_load_si512((__dst) + 9 * 16), 1), acc1, all_one);    \
            acc2 = _mm256_mask_add_epi16(acc2, 0xf800 & _mm512_cmp_epi32_mask(_mm512_add_epi32(tha, __in_si512), _mm512_load_si512((__dst) + 10 * 16), 1), acc2, all_one);   \
            acc3 = _mm256_mask_add_epi16(acc3, 0xf000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thb, __in_si512), _mm512_load_si512((__dst) + 11 * 16), 1), acc3, all_one);   \
            acc0 = _mm256_mask_add_epi16(acc0, 0xe000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thc, __in_si512), _mm512_load_si512((__dst) + 12 * 16), 1), acc0, all_one);   \
            acc1 = _mm256_mask_add_epi16(acc1, 0xc000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thd, __in_si512), _mm512_load_si512((__dst) + 13 * 16), 1), acc1, all_one);   \
            acc2 = _mm256_mask_add_epi16(acc2, 0x8000 &  _mm512_cmp_epi32_mask(_mm512_add_epi32(the, __in_si512), _mm512_load_si512((__dst) + 14 * 16), 1), acc2, all_one);   \
            acc3 = _mm256_mask_add_epi16(acc3, 0x0000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thf, __in_si512), _mm512_load_si512((__dst) + 15 * 16), 1), acc3, all_one);   \
            acc0 = _mm256_or_si256(acc0, acc1); \
            acc2 = _mm256_or_si256(acc2, acc3); \
            acc0 = _mm256_or_si256(acc0, acc2); \
            if (!_mm256_testz_si256(acc0, acc0)) {   \
                __mmask16 msk0 = 0xfffe & _mm512_cmp_epi32_mask(_mm512_add_epi32(th0, __in_si512), _mm512_load_si512((__dst) + 0 * 16), 1);  \
                __mmask16 msk1 = 0xfffc & _mm512_cmp_epi32_mask(_mm512_add_epi32(th1, __in_si512), _mm512_load_si512((__dst) + 1 * 16), 1);  \
                __mmask16 msk2 = 0xfff8 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th2, __in_si512), _mm512_load_si512((__dst) + 2 * 16), 1);  \
                __mmask16 msk3 = 0xfff0 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th3, __in_si512), _mm512_load_si512((__dst) + 3 * 16), 1);  \
                CHECK_AND_ADD_DIAG(msk0, (Ind + (__ind)), (Jnd + (__jnd) + 0));  \
                CHECK_AND_ADD_DIAG(msk1, (Ind + (__ind)), (Jnd + (__jnd) + 1));  \
                CHECK_AND_ADD_DIAG(msk2, (Ind + (__ind)), (Jnd + (__jnd) + 2));  \
                CHECK_AND_ADD_DIAG(msk3, (Ind + (__ind)), (Jnd + (__jnd) + 3));  \
                __mmask16 msk4 = 0xffe0 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th4, __in_si512), _mm512_load_si512((__dst) + 4 * 16), 1);  \
                __mmask16 msk5 = 0xffc0 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th5, __in_si512), _mm512_load_si512((__dst) + 5 * 16), 1);  \
                __mmask16 msk6 = 0xff80 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th6, __in_si512), _mm512_load_si512((__dst) + 6 * 16), 1);  \
                __mmask16 msk7 = 0xff00 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th7, __in_si512), _mm512_load_si512((__dst) + 7 * 16), 1);  \
                CHECK_AND_ADD_DIAG(msk4, (Ind + (__ind)), (Jnd + (__jnd) + 4));  \
                CHECK_AND_ADD_DIAG(msk5, (Ind + (__ind)), (Jnd + (__jnd) + 5));  \
                CHECK_AND_ADD_DIAG(msk6, (Ind + (__ind)), (Jnd + (__jnd) + 6));  \
                CHECK_AND_ADD_DIAG(msk7, (Ind + (__ind)), (Jnd + (__jnd) + 7));  \
                __mmask16 msk8 = 0xfe00 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th8, __in_si512), _mm512_load_si512((__dst) + 8 * 16), 1);  \
                __mmask16 msk9 = 0xfc00 & _mm512_cmp_epi32_mask(_mm512_add_epi32(th9, __in_si512), _mm512_load_si512((__dst) + 9 * 16), 1);  \
                __mmask16 mska = 0xf800 & _mm512_cmp_epi32_mask(_mm512_add_epi32(tha, __in_si512), _mm512_load_si512((__dst) + 10 * 16), 1); \
                __mmask16 mskb = 0xf000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thb, __in_si512), _mm512_load_si512((__dst) + 11 * 16), 1); \
                CHECK_AND_ADD_DIAG(msk8, (Ind + (__ind)), (Jnd + (__jnd) + 8));  \
                CHECK_AND_ADD_DIAG(msk9, (Ind + (__ind)), (Jnd + (__jnd) + 9));  \
                CHECK_AND_ADD_DIAG(mska, (Ind + (__ind)), (Jnd + (__jnd) + 10)); \
                CHECK_AND_ADD_DIAG(mskb, (Ind + (__ind)), (Jnd + (__jnd) + 11)); \
                __mmask16 mskc = 0xe000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thc, __in_si512), _mm512_load_si512((__dst) + 12 * 16), 1); \
                __mmask16 mskd = 0xc000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thd, __in_si512), _mm512_load_si512((__dst) + 13 * 16), 1); \
                __mmask16 mske = 0x8000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(the, __in_si512), _mm512_load_si512((__dst) + 14 * 16), 1); \
                __mmask16 mskf = 0x0000 & _mm512_cmp_epi32_mask(_mm512_add_epi32(thf, __in_si512), _mm512_load_si512((__dst) + 15 * 16), 1); \
                CHECK_AND_ADD_DIAG(mskc, (Ind + (__ind)), (Jnd + (__jnd) + 12)); \
                CHECK_AND_ADD_DIAG(mskd, (Ind + (__ind)), (Jnd + (__jnd) + 13)); \
                CHECK_AND_ADD_DIAG(mske, (Ind + (__ind)), (Jnd + (__jnd) + 14)); \
                CHECK_AND_ADD_DIAG(mskf, (Ind + (__ind)), (Jnd + (__jnd) + 15)); \
            }       \
        } else {    \
            const long _nrem = nvec - Jnd - (__jnd); \
            long __i = 0;   \
            while (__i < _nrem - 3) {    \
                __mmask16 msk0 = ((1 << _nrem) - 1) & (65536 - (2 << __i)) & _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i], __in_si512), _mm512_load_si512((__dst) + __i * 16), 1);    \
                __mmask16 msk1 = ((1 << _nrem) - 1) & (65536 - (4 << __i)) & _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i + 1], __in_si512), _mm512_load_si512((__dst) + (__i + 1) * 16), 1);    \
                __mmask16 msk2 = ((1 << _nrem) - 1) & (65536 - (8 << __i)) & _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i + 2], __in_si512), _mm512_load_si512((__dst) + (__i + 2) * 16), 1);    \
                __mmask16 msk3 = ((1 << _nrem) - 1) & (65536 - (16 << __i)) & _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i + 3], __in_si512), _mm512_load_si512((__dst) + (__i + 3) * 16), 1);    \
                CHECK_AND_ADD_DIAG(msk0, (Ind + (__ind)), (Jnd + (__jnd) + __i + 0));  \
                CHECK_AND_ADD_DIAG(msk1, (Ind + (__ind)), (Jnd + (__jnd) + __i + 1));  \
                CHECK_AND_ADD_DIAG(msk2, (Ind + (__ind)), (Jnd + (__jnd) + __i + 2));  \
                CHECK_AND_ADD_DIAG(msk3, (Ind + (__ind)), (Jnd + (__jnd) + __i + 3));  \
                __i += 4;   \
            }   \
            while (__i < _nrem) {    \
                __mmask16 msk0 = ((1 << _nrem) - 1) & (65536 - (2 << __i)) & _mm512_cmp_epi32_mask(_mm512_add_epi32(tail_th[__i], __in_si512), _mm512_load_si512((__dst) + __i * 16), 1);    \
                CHECK_AND_ADD_DIAG(msk0, (Ind + (__ind)), (Jnd + (__jnd) + __i));  \
                __i++;  \
            }   \
        }   \
    } while (0)

    /////// FIRSTSTEP: stream in all vectors while computing their transpose ///////
    do {
        long ind = 0;
        __m512i zero_si512 = _mm512_setzero_si512();
        __m256i zero_si256 = _mm256_setzero_si256();

        // p blocks
        while (ind < bkt->num_pvec - 15) {
            __m512i z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf;
            z0 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 0]));
            z1 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 1]));
            z2 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 2]));
            z3 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 3]));
            z4 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 4]));
            z5 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 5]));
            z6 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 6]));
            z7 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 7]));
            z8 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 8]));
            z9 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 9]));
            za = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 10]));
            zb = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 11]));
            zc = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 12]));
            zd = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 13]));
            ze = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 14]));
            zf = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 15]));
            AVX512_MATTR_16x16(buf0 + ind * 64, buf0_tr + ind * 64, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf);

            z0 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 0] + 64));
            z1 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 1] + 64));
            z2 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 2] + 64));
            z3 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 3] + 64));
            z4 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 4] + 64));
            z5 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 5] + 64));
            z6 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 6] + 64));
            z7 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 7] + 64));
            z8 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 8] + 64));
            z9 = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 9] + 64));
            za = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 10] + 64));
            zb = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 11] + 64));
            zc = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 12] + 64));
            zd = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 13] + 64));
            ze = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 14] + 64));
            zf = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 15] + 64));
            AVX512_MATTR_16x16(buf1 + ind * 64, buf1_tr + ind * 64, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf);

            __m256i y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf;
            y0 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 0] + 128));
            y1 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 1] + 128));
            y2 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 2] + 128));
            y3 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 3] + 128));
            y4 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 4] + 128));
            y5 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 5] + 128));
            y6 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 6] + 128));
            y7 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 7] + 128));
            y8 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 8] + 128));
            y9 = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 9] + 128));
            ya = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 10] + 128));
            yb = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 11] + 128));
            yc = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 12] + 128));
            yd = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 13] + 128));
            ye = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 14] + 128));
            yf = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + 15] + 128));
            AVX512_MATTR_16x8(buf2 + ind * 32, buf2_tr + ind * 64, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf);

            ind += 16;
        }
        if (ind < nvec - 15) {
            // the mixed block
            do {
                __m512i z[16];
                __m256i y[16];
                for (long i = 0; i < bkt->num_pvec - ind; i++) {
                    z[i] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + i]));
                }
                for (long i = bkt->num_pvec - ind; i < 16; i++) {
                    z[i] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind + i - bkt->num_pvec]));
                    z[i] = _mm512_sub_epi8(zero_si512, z[i]);
                }
                AVX512_MATTR_16x16(buf0 + ind * 64, buf0_tr + ind * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);

                for (long i = 0; i < bkt->num_pvec - ind; i++) {
                    z[i] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[ind + i] + 64));
                }
                for (long i = bkt->num_pvec - ind; i < 16; i++) {
                    z[i] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind + i - bkt->num_pvec] + 64));
                    z[i] = _mm512_sub_epi8(zero_si512, z[i]);
                }
                AVX512_MATTR_16x16(buf1 + ind * 64, buf1_tr + ind * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
                
                for (long i = 0; i < bkt->num_pvec - ind; i++) {
                    y[i] = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[ind + i] + 128));
                }
                for (long i = bkt->num_pvec - ind; i < 16; i++) {
                    y[i] = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind + i - bkt->num_pvec] + 128));
                    y[i] = _mm256_sub_epi8(zero_si256, y[i]);
                }
                AVX512_MATTR_16x8(buf2 + ind * 32, buf2_tr + ind * 64, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]);
                
                ind += 16;
            } while (0);
            // n blocks
            while (ind < nvec - 15) {
                __m512i z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf;
                z0 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 0])));
                z1 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 1])));
                z2 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 2])));
                z3 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 3])));
                z4 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 4])));
                z5 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 5])));
                z6 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 6])));
                z7 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 7])));
                z8 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 8])));
                z9 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 9])));
                za = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 10])));
                zb = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 11])));
                zc = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 12])));
                zd = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 13])));
                ze = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 14])));
                zf = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 15])));
                AVX512_MATTR_16x16(buf0 + ind * 64, buf0_tr + ind * 64, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf);

                z0 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 0] + 64)));
                z1 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 1] + 64)));
                z2 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 2] + 64)));
                z3 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 3] + 64)));
                z4 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 4] + 64)));
                z5 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 5] + 64)));
                z6 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 6] + 64)));
                z7 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 7] + 64)));
                z8 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 8] + 64)));
                z9 = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 9] + 64)));
                za = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 10] + 64)));
                zb = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 11] + 64)));
                zc = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 12] + 64)));
                zd = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 13] + 64)));
                ze = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 14] + 64)));
                zf = _mm512_sub_epi8(zero_si512, _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 15] + 64)));

                AVX512_MATTR_16x16(buf1 + ind * 64, buf1_tr + ind * 64, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, za, zb, zc, zd, ze, zf);

                __m256i y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf;
                y0 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 0] + 128)));
                y1 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 1] + 128)));
                y2 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 2] + 128)));
                y3 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 3] + 128)));
                y4 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 4] + 128)));
                y5 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 5] + 128)));
                y6 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 6] + 128)));
                y7 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 7] + 128)));
                y8 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 8] + 128)));
                y9 = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 9] + 128)));
                ya = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 10] + 128)));
                yb = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 11] + 128)));
                yc = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 12] + 128)));
                yd = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 13] + 128)));
                ye = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 14] + 128)));
                yf = _mm256_sub_epi8(zero_si256, _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[ind - bkt->num_pvec + 15] + 128)));

                AVX512_MATTR_16x8(buf2 + ind * 32, buf2_tr + ind * 64, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, ya, yb, yc, yd, ye, yf);
                ind += 16;
            }
        }
        // the last block
        if (ind < nvec) {
            __m512i z[16];
            __m256i y[16];
            long i = ind;
            for (; i < bkt->num_pvec; i++) {
                z[i-ind] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[i]));
            }
            for (; i < nvec; i++) {
                z[i-ind] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[i - bkt->num_pvec]));
                z[i-ind] = _mm512_sub_epi8(zero_si512, z[i-ind]);
            }
            for (; i < ind + 16; i++) {
                z[i-ind] = zero_si512;
            }
            AVX512_MATTR_16x16(buf0 + ind * 64, buf0_tr + ind * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);
        
            i = ind;
            for (; i < bkt->num_pvec; i++) {
                z[i-ind] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->pvec[i] + 64));
            }
            for (; i < nvec; i++) {
                z[i-ind] = _mm512_loadu_si512((__m512i *)(vec + (long)vec_length * (long)bkt->nvec[i - bkt->num_pvec] + 64));
                z[i-ind] = _mm512_sub_epi8(zero_si512, z[i-ind]);
            }
            for (; i < ind + 16; i++) {
                z[i-ind] = zero_si512;
            }
            AVX512_MATTR_16x16(buf1 + ind * 64, buf1_tr + ind * 64, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[10], z[11], z[12], z[13], z[14], z[15]);

            i = ind;
            for (; i < bkt->num_pvec; i++) {
                y[i-ind] = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->pvec[i] + 128));
            }
            for (; i < nvec; i++) {
                y[i-ind] = _mm256_load_si256((__m256i *)(vec + (long)vec_length * (long)bkt->nvec[i - bkt->num_pvec] + 128));
                y[i-ind] = _mm256_sub_epi8(_mm256_setzero_si256(), y[i-ind]);
            }
            for (; i < ind + 16; i++) {
                y[i-ind] = zero_si256;
            }
            AVX512_MATTR_16x8(buf2 + ind * 32, buf2_tr + ind * 64, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]);
        }
    } while (0);

    /////// SECONDSTEP: read in all norms ///////
    do {
        long ind = 0;
        while (ind < bkt->num_pvec) {
            _mm512_store_si512(norm + ind, _mm512_loadu_si512((__m512i *)(bkt->pnorm + ind)));
            ind += 16;
        }
        for (long i = 0; i < bkt->num_nvec && i < ind - bkt->num_pvec; i++) norm[bkt->num_pvec + i] = bkt->nnorm[i];
        while (ind < nvec) {
            _mm512_store_si512(norm + ind, _mm512_loadu_si512((__m512i *)(bkt->nnorm + ind - bkt->num_pvec)));
            ind += 16;
        }
    } while (0);

    /////// MAINSEARCH: search in each L1 blocks by amx ///////
    for (long Ind = 0; Ind < nvec; Ind += 128) {
        // the trianglar block
        do {
            long Jnd = Ind;
            if (Ind < nvec - 127) {
                // a full 8x8 triangular block
                const int8_t *_buf0 = buf0 + Jnd * 64;
                const int8_t *_buf1 = buf1 + Jnd * 64;
                const int8_t *_buf2 = buf2 + Jnd * 32;
                const int8_t *_buf0_tr = buf0_tr + Ind * 64;
                const int8_t *_buf1_tr = buf1_tr + Ind * 64;
                const int8_t *_buf2_tr = buf2_tr + Ind * 64;
                const int32_t *_inorm = norm + Ind;
                const int32_t *_jnorm = norm + Jnd;

                RELOAD_THRESHOLD(_jnorm + 0 * 16);
                TILE_DP160x1(0, 0);
                CHECK_AND_MASKADD_DIAG(dst, 0, 0, 0xffff);

                RELOAD_THRESHOLD(_jnorm + 1 * 16);
                TILE_DP160x2(0, 16);
                CHECK_AND_MASKADD(dst + 256, 0, 16, 0xffff);
                CHECK_AND_MASKADD_DIAG(dst + 512, 16, 16, 0xffff);

                RELOAD_THRESHOLD(_jnorm + 2 * 16);
                TILE_DP160x3(0, 32);
                for (long i = 0; i < 32; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i, 32, 0xffff);
                }
                CHECK_AND_MASKADD_DIAG(dst + 512, 32, 32, 0xffff);

                RELOAD_THRESHOLD(_jnorm + 3 * 16);
                TILE_DP160x4(0, 48);
                for (long i = 0; i < 48; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i, 48, 0xffff);
                }
                CHECK_AND_MASKADD_DIAG(dst + 768, 48, 48, 0xffff);

                RELOAD_THRESHOLD(_jnorm + 4 * 16);
                TILE_DP160x4(0, 64);
                for (long i = 0; i < 64; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i, 64, 0xffff);
                }
                TILE_DP160x1(64, 64);
                CHECK_AND_MASKADD_DIAG(dst, 64, 64, 0xffff);

                RELOAD_THRESHOLD(_jnorm + 5 * 16);
                TILE_DP160x2(64, 80);
                CHECK_AND_MASKADD(dst + 256, 64, 80, 0xffff);
                CHECK_AND_MASKADD_DIAG(dst + 512, 80, 80, 0xffff);
                TILE_DP160x4(0, 80);
                for (long i = 0; i < 64; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i, 80, 0xffff);
                }

                RELOAD_THRESHOLD(_jnorm + 6 * 16);
                TILE_DP160x4(0, 96);
                for (long i = 0; i < 64; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i, 96, 0xffff);
                }
                TILE_DP160x3(64, 96);
                for (long i = 0; i < 32; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i + 64, 96, 0xffff);
                }
                CHECK_AND_MASKADD_DIAG(dst + 512, 96, 96, 0xffff);

                RELOAD_THRESHOLD(_jnorm + 7 * 16);
                TILE_DP160x4(0, 112);
                for (long i = 0; i < 64; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i, 112, 0xffff);
                }
                TILE_DP160x4(64, 112);
                for (long i = 0; i < 48; i += 16) {
                    CHECK_AND_MASKADD(dst + 16 * i, i + 64, 112, 0xffff);
                }
                CHECK_AND_MASKADD_DIAG(dst + 768, 112, 112, 0xffff);
            } else {
                const int8_t *_buf0 = buf0 + Jnd * 64;
                const int8_t *_buf1 = buf1 + Jnd * 64;
                const int8_t *_buf2 = buf2 + Jnd * 32;
                const int8_t *_buf0_tr = buf0_tr + Ind * 64;
                const int8_t *_buf1_tr = buf1_tr + Ind * 64;
                const int8_t *_buf2_tr = buf2_tr + Ind * 64;
                const int32_t *_inorm = norm + Ind;
                const int32_t *_jnorm = norm + Jnd;

                const long nrem = nvec - Ind;
                long j = 0;
                for (; j < nrem - 15; j += 16) {
                    RELOAD_THRESHOLD(_jnorm + j);
                    for (long i = 0; i < j; i += 16) {
                        TILE_DP160x1(i, j);
                        CHECK_AND_MASKADD(dst, i, j, 0xffff);
                    }
                    TILE_DP160x1(j, j);
                    CHECK_AND_MASKADD_DIAG(dst, j, j, 0xffff);
                }
                if (j < nrem) {
                    _mm512_store_si512(th_tmp, _mm512_sub_epi32(_mm512_load_si512(_jnorm + j), gn_si512));
                    for (long i = 0; i < nvec - Jnd - j; i++) {
                        tail_th[i] = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + i)));
                    }
                    for (long i = 0; i < j; i += 16) {
                        TILE_DP160x1(i, j);
                        CHECK_AND_MASKADD(dst, i, j, 0);
                    }
                    TILE_DP160x1(j, j);
                    CHECK_AND_MASKADD_DIAG(dst, j, j, 0);
                }
            }
        } while (0);
        

        long Jnd = Ind + 128;
        while (Jnd < nvec - 127) {
            const int8_t *_buf0 = buf0 + Jnd * 64;
            const int8_t *_buf1 = buf1 + Jnd * 64;
            const int8_t *_buf2 = buf2 + Jnd * 32;
            const int8_t *_buf0_tr = buf0_tr + Ind * 64;
            const int8_t *_buf1_tr = buf1_tr + Ind * 64;
            const int8_t *_buf2_tr = buf2_tr + Ind * 64;
            const int32_t *_inorm = norm + Ind;
            const int32_t *_jnorm = norm + Jnd;

            for (long j = 0; j < 128; j += 16) {
                RELOAD_THRESHOLD(_jnorm + j);
                TILE_DP160x4(0, j);
                CHECK_AND_MASKADD(dst + 0 * 256, 0 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 1 * 256, 1 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 2 * 256, 2 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 3 * 256, 3 * 16, j, 0xffff);
                TILE_DP160x4(64, j);
                CHECK_AND_MASKADD(dst + 0 * 256, 4 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 1 * 256, 5 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 2 * 256, 6 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 3 * 256, 7 * 16, j, 0xffff);
            }
            Jnd += 128;
        }

        if (Jnd < nvec) {
            const int8_t *_buf0 = buf0 + Jnd * 64;
            const int8_t *_buf1 = buf1 + Jnd * 64;
            const int8_t *_buf2 = buf2 + Jnd * 32;
            const int8_t *_buf0_tr = buf0_tr + Ind * 64;
            const int8_t *_buf1_tr = buf1_tr + Ind * 64;
            const int8_t *_buf2_tr = buf2_tr + Ind * 64;
            const int32_t *_inorm = norm + Ind;
            const int32_t *_jnorm = norm + Jnd;

            long j = 0;
            for (; j < nvec - Jnd - 15; j += 16) {
                RELOAD_THRESHOLD(_jnorm + j);
                TILE_DP160x4(0, j);
                CHECK_AND_MASKADD(dst + 0 * 256, 0 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 1 * 256, 1 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 2 * 256, 2 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 3 * 256, 3 * 16, j, 0xffff);
                TILE_DP160x4(64, j);
                CHECK_AND_MASKADD(dst + 0 * 256, 4 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 1 * 256, 5 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 2 * 256, 6 * 16, j, 0xffff);
                CHECK_AND_MASKADD(dst + 3 * 256, 7 * 16, j, 0xffff);
            }

            if (j < nvec - Jnd) {
                _mm512_store_si512(th_tmp, _mm512_sub_epi32(_mm512_load_si512(_jnorm + j), gn_si512));
                for (long i = 0; i < nvec - Jnd - j; i++) {
                    tail_th[i] = _mm512_broadcast_i64x4((__m256i)_mm256_broadcast_ss((float *)(th_tmp + i)));
                }
                TILE_DP160x4(0, j);
                CHECK_AND_MASKADD(dst + 0 * 256, 0 * 16, j, 0);
                CHECK_AND_MASKADD(dst + 1 * 256, 1 * 16, j, 0);
                CHECK_AND_MASKADD(dst + 2 * 256, 2 * 16, j, 0);
                CHECK_AND_MASKADD(dst + 3 * 256, 3 * 16, j, 0);
                TILE_DP160x4(64, j);
                CHECK_AND_MASKADD(dst + 0 * 256, 4 * 16, j, 0);
                CHECK_AND_MASKADD(dst + 1 * 256, 5 * 16, j, 0);
                CHECK_AND_MASKADD(dst + 2 * 256, 6 * 16, j, 0);
                CHECK_AND_MASKADD(dst + 3 * 256, 7 * 16, j, 0);
            }
        }
    }

    if (prof) {
        // pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->succ_add2 += succ_add2;
        // pthread_spin_unlock(&prof->profile_lock);
    }
    return 0;
}


template <uint32_t nb>
int Pool_epi8_t<nb>::_sol_list_to_vec_amx(sol_list_amx_t **sol_list, long num_sol_list, int8_t *dst_vec, uint64_t *dst_vu, int32_t *dst_vnorm, int32_t *dst_vsum) {
    long *begin_ind = (long *) NEW_VEC(num_sol_list, sizeof(long));
    long *end_ind = (long *) NEW_VEC(num_sol_list, sizeof(long));
    end_ind[0] = sol_list[0]->num_sol();
    for (long i = 1; i < num_sol_list; i++) {
        begin_ind[i] = end_ind[i-1];
        end_ind[i] = begin_ind[i] + sol_list[i]->num_sol();
    }

    #pragma omp parallel for
    for (long thread = 0; thread < num_sol_list; thread++) {
        __attribute__ ((aligned (32))) int8_t tmp[vec_length * 8];
        __attribute__ ((aligned (32))) int16_t tck[vec_length * 8];
        __attribute__ ((aligned (32))) int32_t coeff[8 * vec_length];
        __attribute__ ((aligned (32))) float fvec[vec_length * 8];
        __attribute__ ((aligned (32))) float fnorm[8];
        __attribute__ ((aligned (32))) int32_t sum[8];

        const __m256i diff_bound = _mm256_set1_epi16(0x3);

        long ind = begin_ind[thread];
        long status, status_ind;
        sol_list[thread]->init(status, status_ind);
        while (ind < end_ind[thread] - 7) {
            for (long i = 0; i < 8; i++) {
                if (status == 0) {
                    int8_t *src1 = vec + sol_list[thread]->a_list[2*status_ind] * vec_length;
                    int8_t *src2 = vec + sol_list[thread]->a_list[2*status_ind+1] * vec_length;
                    copy_avx2(tmp + i * vec_length, src1, vec_length);
                    add_avx2(tmp + i * vec_length, src2, vec_length);
                    for (long l = 0; l < vec_length; l += 16) {
                        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                        __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                        _mm256_store_si256((__m256i *)(tck + i * vec_length + l), _mm256_add_epi16(x1, x2));
                    }
                    dst_vu[ind+i] = vu[sol_list[thread]->a_list[2*status_ind]] + vu[sol_list[thread]->a_list[2*status_ind+1]];
                } else if (status == 1) {
                    int8_t *src1 = vec + sol_list[thread]->s_list[2*status_ind] * vec_length;
                    int8_t *src2 = vec + sol_list[thread]->s_list[2*status_ind+1] * vec_length;
                    copy_avx2(tmp + i * vec_length, src1, vec_length);
                    sub_avx2(tmp + i * vec_length, src2, vec_length);
                    for (long l = 0; l < vec_length; l += 16) {
                        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                        __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                        _mm256_store_si256((__m256i *)(tck + i * vec_length + l), _mm256_sub_epi16(x1, x2));
                    }
                    dst_vu[ind+i] = vu[sol_list[thread]->s_list[2*status_ind]] - vu[sol_list[thread]->s_list[2*status_ind+1]];
                }
                sol_list[thread]->next(status, status_ind);
            }

            _compute_sum_b8(sum, tmp);
            _compute_coeff_b8(coeff, tmp, sum);
            _compute_fvec_b8(fvec, coeff);
            _compute_fnorm_b8(fnorm, fvec);
            _mm256_storeu_si256((__m256i *) (dst_vnorm + ind), _mm256_cvtps_epi32(_mm256_load_ps(fnorm)));            
            
            for (long i = 0; i < vec_length; i += 16) {
                for (long j = 0; j < 8; j++) {
                    __m128i lo = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + j * vec_length + i)));
                    __m128i hi = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + j * vec_length + i + 8)));
                    _mm_store_si128((__m128i *)(dst_vec + (ind + j) * vec_length + i), _mm_or_si128(_mm_slli_si128(hi, 8), lo));
                }                
            }


            uint32_t rej = 0;
            for (long i = 0; i < 8; i++) {
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i diff = _mm256_abs_epi16(_mm256_sub_epi16(_mm256_load_si256((__m256i *)(tck + i * vec_length + l)), _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(dst_vec + (ind + i) * vec_length + l)))));
                    if (_mm256_movemask_epi8(_mm256_cmpgt_epi16(diff, diff_bound))) {
                        rej |= (1 << i);
                        break;
                    }
                }
                #if REJ_ENTRY128
                do {
                    __m256i err0x80 = _mm256_setzero_si256();
                    __m256i all0x80 = _mm256_set1_epi32(0x80808080);
                    for (long l = 0; l < vec_length; l += 32) {
                        err0x80 = _mm256_or_si256(err0x80, _mm256_cmpeq_epi8(all0x80, _mm256_load_si256((__m256i *)(dst_vec + (ind + i) * vec_length + l))));
                    }
                    if (!_mm256_testz_si256(err0x80, err0x80)) rej |= (1 << i);
                } while (0);
                #endif
            }
            while (rej) {
                int32_t r = __builtin_ctz(rej);
                rej -= (1 << r);
                dst_vnorm[ind + r] = 2147483647;
            }

            _compute_sum_b8(&dst_vsum[ind], dst_vec + ind * vec_length);
            
            ind += 8;
        }

        while (ind < end_ind[thread]) {
            if (status == 0) {
                int8_t *src1 = vec + sol_list[thread]->a_list[2*status_ind] * vec_length;
                int8_t *src2 = vec + sol_list[thread]->a_list[2*status_ind+1] * vec_length;
                copy_avx2(tmp, src1, vec_length);
                add_avx2(tmp, src2, vec_length);
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                    _mm256_store_si256((__m256i *)(tck + l), _mm256_add_epi16(x1, x2));
                }
                dst_vu[ind] = vu[sol_list[thread]->a_list[2*status_ind]] + vu[sol_list[thread]->a_list[2*status_ind+1]];
            } else if (status == 1) {
                int8_t *src1 = vec + sol_list[thread]->s_list[2*status_ind] * vec_length;
                int8_t *src2 = vec + sol_list[thread]->s_list[2*status_ind+1] * vec_length;
                copy_avx2(tmp, src1, vec_length);
                sub_avx2(tmp, src2, vec_length);
                for (long l = 0; l < vec_length; l += 16) {
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src1 + l)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(src2 + l)));
                    _mm256_store_si256((__m256i *)(tck + l), _mm256_sub_epi16(x1, x2));
                }
                dst_vu[ind] = vu[sol_list[thread]->s_list[2*status_ind]] - vu[sol_list[thread]->s_list[2*status_ind+1]];
            }
            sol_list[thread]->next(status, status_ind);
            
            dst_vsum[ind] = _compute_sum(tmp);
            _compute_coeff(coeff, tmp, dst_vsum[ind]);
            _compute_fvec(fvec, coeff);
            fnorm[0] = 0.5 * dot_avx2(fvec, fvec, vec_length);
            dst_vnorm[ind] = round(fnorm[0]);
            for (long i = 0; i < vec_length; i += 16) {
                __m128i lo = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + i)));
                __m128i hi = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(_mm256_load_ps(fvec + i + 8)));
                _mm_store_si128((__m128i *)(dst_vec + ind * vec_length + i), _mm_or_si128(_mm_slli_si128(hi, 8), lo));          
            }

            for (long l = 0; l < vec_length; l += 16) {
                __m256i diff = _mm256_abs_epi16(_mm256_sub_epi16(_mm256_load_si256((__m256i *)(tck + l)), _mm256_cvtepi8_epi16(_mm_load_si128((__m128i *)(dst_vec + ind * vec_length + l)))));
                if (_mm256_movemask_epi8(_mm256_cmpgt_epi16(diff, diff_bound))) {
                    dst_vnorm[ind] = 2147483647;
                    break;
                }
            }
            #if REJ_ENTRY128
            do {
                __m256i err0x80 = _mm256_setzero_si256();
                __m256i all0x80 = _mm256_set1_epi32(0x80808080);
                for (long l = 0; l < vec_length; l += 32) {
                    err0x80 = _mm256_or_si256(err0x80, _mm256_cmpeq_epi8(all0x80, _mm256_load_si256((__m256i *)(dst_vec + ind * vec_length + l))));
                }
                if (!_mm256_testz_si256(err0x80, err0x80)) dst_vnorm[ind] = 2147483647;
            } while (0);
            #endif

            dst_vsum[ind] = _compute_sum(dst_vec + ind * vec_length);
            ind++;
        }
    }

    FREE_VEC((void *)begin_ind);
    FREE_VEC((void *)end_ind);
    return 0;
}       

template <uint32_t nb>
template <bool profiling>
uint64_t Pool_epi8_t<nb>::_pool_insert_amx(sol_list_amx_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_amx_profile_data_t<nb> *prof) {
    uint64_t num_total_insert = 0;

    uint64_t length_stat[256] = {};
    uint64_t num_linfty_failed = 0;
    uint64_t num_l2_failed = 0;
    uint64_t num_not_try = 0;

    long num_total_sol = 0;
    long num_total_empty;
    long num_total_nonempty;
    long num_sol[AMX_MAX_NTHREADS];
    long num_emptyy[AMX_MAX_NTHREADS];
    long num_nonemptyy[AMX_MAX_NTHREADS];
    long empty_begin[AMX_MAX_NTHREADS];
    long empty_end[AMX_MAX_NTHREADS];
    long nonempty_begin[AMX_MAX_NTHREADS];
    long nonempty_end[AMX_MAX_NTHREADS];

    for (long i = 0; i < num_threads; i++) {
        num_sol[i] = sol_list[i]->num_sol();
        num_total_sol += num_sol[i];
    }
    num_total_empty = ( num_total_sol > num_empty ) ? num_empty : num_total_sol;
    num_total_nonempty = ( num_total_sol - num_total_empty > sorted_index - goal_index) ? (sorted_index - goal_index) : (num_total_sol - num_total_empty);
    for (long i = 0; i < num_threads; i++) {
        num_emptyy[i] = (num_total_empty * (i+1)) / num_threads - (num_total_empty * i) / num_threads;
        num_nonemptyy[i] = (num_total_nonempty * (i+1)) / num_threads - (num_total_nonempty * i) / num_threads;
    }
    empty_begin[0] = num_vec;
    empty_end[0] = num_vec + num_emptyy[0];
    nonempty_begin[0] = sorted_index - num_nonemptyy[0];
    nonempty_end[0] = sorted_index;
    for (long i = 1; i < num_threads; i++) {
        empty_begin[i] = empty_end[i-1];
        empty_end[i] = empty_begin[i] + num_emptyy[i];
        nonempty_end[i] = nonempty_begin[i-1];
        nonempty_begin[i] = nonempty_end[i] - num_nonemptyy[i];
    }

    int8_t *vec_to_insert = (int8_t *) NEW_VEC(num_total_sol * vec_length, sizeof(int8_t));
    uint64_t *vu_to_insert = (uint64_t *) NEW_VEC(num_total_sol, sizeof(uint64_t));
    int32_t *vnorm_to_insert = (int32_t *) NEW_VEC(num_total_sol, sizeof(int32_t));
    int32_t *vsum_to_insert = (int32_t *) NEW_VEC(num_total_sol, sizeof(int32_t));
    _sol_list_to_vec_amx(sol_list, num_sol_list, vec_to_insert, vu_to_insert, vnorm_to_insert, vsum_to_insert);
    #if BOOST_AMX_SIEVE
    uint16_t *score_to_insert = (uint16_t *) NEW_VEC(num_total_sol, sizeof(uint16_t));
    booster->reconstruct_score(score_to_insert, vec_to_insert, vnorm_to_insert, num_total_sol);
    #endif

    long empty_final_ind[AMX_MAX_NTHREADS];
    long nonempty_final_ind[AMX_MAX_NTHREADS];
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        const long begin_ind = (num_total_sol * thread) / num_threads;
        const long end_ind = (num_total_sol * (thread+1)) / num_threads;
        long ind = begin_ind;
        long empty_ind = empty_begin[thread];
        long nonempty_ind = nonempty_end[thread] - 1;
        
        uint64_t _length_stat[256] = {};
        uint64_t _num_linfty_failed = 0;
        uint64_t _num_l2_failed = 0;

        const int32_t linfty_fail_bound = 1.2 * goal_norm;

        while (ind < end_ind && nonempty_ind >= nonempty_begin[thread]) {
            if (vnorm_to_insert[ind] > linfty_fail_bound) {
                if (profiling) _num_linfty_failed++;
                if (!uid->erase_uid(vu_to_insert[ind])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
                }
                ind++;
                continue;
            }
            
            uint32_t dst = *((uint32_t *)(cvec + 3LL * nonempty_ind));
            if (profiling) {
                int r = round(sqrt((10000.0 * vnorm_to_insert[ind]) / goal_norm));
                if (r > 255) r = 255;
                if (r < 0) r = 0;
                _length_stat[r]++;
            }
            #if BOOST_AMX_SIEVE
            if (score_to_insert[ind] < cvec[nonempty_ind*3LL+2LL]) {
            #else
            if (vnorm_to_insert[ind] < vnorm[dst]) {
            #endif
                if (!uid->erase_uid(vu[dst])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of pool vector failed, ignored.\n", nb);
                }
                copy_avx2(vec + dst * vec_length, vec_to_insert + ind * vec_length, vec_length);
                vnorm[dst] = vnorm_to_insert[ind];
                vu[dst] = vu_to_insert[ind];
                vsum[dst] = vsum_to_insert[ind];
                #if BOOST_AMX_SIEVE
                cvec[nonempty_ind*3LL+2LL] = score_to_insert[ind];
                #else
                int32_t cnorm = ((vnorm[dst] >> 1) > 65535) ? 65535 : (vnorm[dst] >> 1);
                cvec[nonempty_ind*3LL+2LL] = cnorm; 
                #endif
                nonempty_ind--;
                ind++;
                continue;
            } else if (empty_ind < empty_end[thread]) {
                copy_avx2(vec + empty_ind * vec_length, vec_to_insert + ind * vec_length, vec_length);
                vnorm[empty_ind] = vnorm_to_insert[ind];
                vsum[empty_ind] = vsum_to_insert[ind];
                vu[empty_ind] = vu_to_insert[ind];
                #if BOOST_AMX_SIEVE
                cvec[empty_ind * 3LL + 2LL] = score_to_insert[ind];
                #else
                int32_t cnorm = ((vnorm[empty_ind] >> 1) > 65535) ? 65535 : (vnorm[empty_ind] >> 1);
                cvec[empty_ind * 3LL + 2LL] = cnorm;
                #endif
                *((uint32_t *) (cvec + empty_ind * 3LL)) = empty_ind;
                ind++;
                empty_ind++;
                continue;
            } else {
                if (profiling) _num_l2_failed++;
                if (!uid->erase_uid(vu_to_insert[ind])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
                }
                ind++;
            }
        }

        while (ind < end_ind && empty_ind < empty_end[thread]) {
            if (vnorm_to_insert[ind] > linfty_fail_bound) {
                if (profiling) _num_linfty_failed++;
                if (!uid->erase_uid(vu_to_insert[ind])){
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
                }
                ind++;
                continue;
            }
            if (profiling) {
                int r = round(sqrt((10000.0 * vnorm_to_insert[ind]) / goal_norm));
                if (r > 255) r = 255;
                if (r < 0) r = 0;
                _length_stat[r]++;
            }
            copy_avx2(vec + empty_ind * vec_length, vec_to_insert + ind * vec_length, vec_length);
            vnorm[empty_ind] = vnorm_to_insert[ind];
            vsum[empty_ind] = vsum_to_insert[ind];
            vu[empty_ind] = vu_to_insert[ind];
            #if BOOST_AMX_SIEVE
            cvec[empty_ind * 3LL + 2LL] = score_to_insert[ind];
            #else
            int32_t cnorm = ((vnorm[empty_ind] >> 1) > 65535) ? 65535 : (vnorm[empty_ind] >> 1);
            cvec[empty_ind * 3LL + 2LL] = cnorm;
            #endif
            *((uint32_t *) (cvec + empty_ind * 3LL)) = empty_ind;
            ind++;
            empty_ind++;
        }

        while (ind < end_ind) {
            if (!uid->erase_uid(vu_to_insert[ind])){
                fprintf(stderr, "[Error] Pool_epi8_t<%u>::_pool_insert: erase uid of non inserted vector failed, ignored.\n", nb);
            }
            ind++;
        }
 
        if (prof) {
            uint64_t _num_total_insert = (empty_ind - empty_begin[thread]) + (nonempty_end[thread] - 1 - nonempty_ind);
            uint64_t _num_not_try = num_sol[thread] - _num_total_insert - _num_linfty_failed - _num_l2_failed;
            pthread_spin_lock(&prof->profile_lock);
            if (profiling) {
                for (long i = 0; i < 256; i++) length_stat[i] += _length_stat[i];
                num_linfty_failed += _num_linfty_failed;
                num_l2_failed += _num_l2_failed;
                num_not_try += _num_not_try;
            }
            num_total_insert += _num_total_insert;
            pthread_spin_unlock(&prof->profile_lock);
        }
        empty_final_ind[thread] = empty_ind;
        nonempty_final_ind[thread] = nonempty_ind;
    }
    
    FREE_VEC((void *)vec_to_insert);
    FREE_VEC((void *)vu_to_insert);
    FREE_VEC((void *)vnorm_to_insert);
    FREE_VEC((void *)vsum_to_insert);
    #if BOOST_AMX_SIEVE
    FREE_VEC((void *)score_to_insert);
    #endif

    for (long i = 0; i < num_threads - 1; i++) {
        for (long k = num_threads - 1; k > i; k--) {
            while ((empty_final_ind[i] < empty_end[i]) && (empty_final_ind[k] > empty_begin[k])) {
                uint32_t src = empty_final_ind[k] - 1;
                uint32_t dst = empty_final_ind[i];
                memcpy(vec + dst * vec_length, vec + src * vec_length, vec_length);
                vnorm[dst] = vnorm[src];
                vsum[dst] = vsum[src];
                vu[dst] = vu[src];
                cvec[(long)dst*3LL+2LL] = cvec[(long)src*3LL+2LL];
                ((uint32_t *)(cvec + (long)dst * 3LL))[0] = dst;
                empty_final_ind[i]++;
                empty_final_ind[k]--;
            }
        }
    }
    for (long i = 0; i < num_threads; i++) {
        num_vec += empty_final_ind[i] - empty_begin[i];
        num_empty -= empty_final_ind[i] - empty_begin[i];
    }
    sorted_index = nonempty_final_ind[num_threads-1];

    if (profiling && prof) {
        prof->insert_inner_log(length_stat, num_linfty_failed, num_l2_failed, num_not_try);
    }

    return num_total_insert;
}


template int Pool_epi8_t<5>::_pool_bucketing_amx<BGJ3_AMX_BUCKET0_BATCHSIZE, 1, 1>(bucket_amx_t **rbucket0, bucket_amx_t **bucket0, double alpha_r0, double alpha_b0, sol_list_amx_t **sol_list, int32_t goal_norm, bgj_amx_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_parallel_sub_bucketing_amx<BGJ3_AMX_BUCKET1_BATCHSIZE, 1, 1>(bucket_amx_t *main_bucket, bucket_amx_t **rbucket, bucket_amx_t **bucket, double alpha_r, double alpha_b, sol_list_amx_t **sol_list, int32_t goal_norm, bgj_amx_profile_data_t<5> *prof);
#if AMX_PARALLEL_BUCKET1 == 0
template int Pool_epi8_t<5>::_sub_bucketing_amx<BGJ3_AMX_BUCKET1_BATCHSIZE, 1, 1>(bucket_amx_t *main_bucket, bucket_amx_t **rbucket, bucket_amx_t **bucket, double alpha_r, double alpha_b, sol_list_amx_t *sol, int32_t goal_norm, bgj_amx_profile_data_t<5> *prof);
#endif
template int Pool_epi8_t<5>::_sub_bucketing_amx<BGJ3_AMX_BUCKET2_BATCHSIZE, 1, 0>(bucket_amx_t *main_bucket, bucket_amx_t **rbucket, bucket_amx_t **bucket, double alpha_r, double alpha_b, sol_list_amx_t *sol, int32_t goal_norm, bgj_amx_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_amx(bucket_amx_t *bkt, sol_list_amx_t *sol, int32_t goal_norm, bgj_amx_profile_data_t<5> *prof);
template uint64_t Pool_epi8_t<5>::_pool_insert_amx<0>(sol_list_amx_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_amx_profile_data_t<5> *prof);
template uint64_t Pool_epi8_t<5>::_pool_insert_amx<1>(sol_list_amx_t **sol_list, long num_sol_list, int32_t goal_norm, int32_t goal_index, bgj_amx_profile_data_t<5> *prof);
#endif
