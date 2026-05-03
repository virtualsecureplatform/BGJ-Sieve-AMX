#include "../include/pool_epi8.h"
#include "../include/bgj_epi8.h"

#include "../include/lsfsh_tables.hpp"

#include <sys/time.h>

#define LSH_DEFAULT_DHDIM 24
#define LSH_DEFAULT_SHSIZE 64
#define LSH_HALF_SHSIZE 32

#define LSH_L1_BLOCK 256
#define LSH_L2_BLOCK 8192
#define LSH_DIAG_FIRST 1
#define LSH_BUCKETING_OVERHEAD 25.0
#define LSH_MBLOCK_BATCH 112

#if 1
struct timeval _lsh_timer_start[MAX_NTHREADS], _lsh_timer_end[MAX_NTHREADS];
double _lsh_time_curr[MAX_NTHREADS];

#define TIMER_START do {                                                        \
        gettimeofday(&_lsh_timer_start[omp_get_thread_num()], NULL);            \
    } while (0)

#define TIMER_END do {                                                          \
        gettimeofday(&_lsh_timer_end[omp_get_thread_num()], NULL);              \
        _lsh_time_curr[omp_get_thread_num()] =                                                            \
            (_lsh_timer_end[omp_get_thread_num()].tv_sec-_lsh_timer_start[omp_get_thread_num()].tv_sec)+                    \
            (double)(_lsh_timer_end[omp_get_thread_num()].tv_usec-_lsh_timer_start[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_lsh_time_curr[omp_get_thread_num()])
#endif

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

struct lsh_profile_data_t {
    long log_level;
    uint64_t bucketing_ops, dp_ops, lift_ops, init_ops;
    double bucketing_time, dp_time, lift_time, init_time;
    uint64_t num_mblock, total_mblock;
    pthread_spinlock_t profile_lock;

    void init(long _log_level) {
        log_level = _log_level;
        num_mblock = 0;
        bucketing_ops = 0;
        dp_ops = 0;
        lift_ops = 0;
        init_ops = 0;
        bucketing_time = 0.0;
        dp_time = 0.0;
        lift_time = 0.0;
        init_time = 0.0;
        pthread_spin_init(&profile_lock, PTHREAD_PROCESS_SHARED);
    }
    void init_log(float *min_norm, long ID) {
        if (log_level >= 1) {
            fprintf(stdout, "init: pool lifting done, time = %fs(%.2f M), speed = %f Mops, total_num_buckets = %lu\n", 
                        init_time, init_ops/1048576.0, init_ops / init_time / 1048576.0, total_mblock);
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
    void mblock_log(float *min_norm, long ID) {
        if (log_level >= 1) {
            fprintf(stdout, "bucket %lu / %lu, bucketing_time = %.2fs(%.2f M, %.3f Mops), dp_time = %.2fs(%.2f G, %.3f Gops), lift_time = %.2fs(%.2f M, %.3f Mops)\n", num_mblock, total_mblock, 
                            bucketing_time, bucketing_ops/1048576.0, bucketing_ops / bucketing_time / 1048576.0, 
                            dp_time, dp_ops/1073741824.0, dp_ops / dp_time / 1073741824.0, 
                            lift_time, lift_ops/1048576.0, lift_ops / lift_time / 1048576.0);
        if (log_level >= 2) {
                fprintf(stdout, "min_norm = [");
                for (long i = 0; i < ID; i++) {
                    fprintf(stdout, "%f", sqrt(min_norm[i]));
                    if (i < ID - 1) fprintf(stdout, " "); 
                }
                fprintf(stdout, "]\n");
            }
        }
        fflush(stdout);
    }
    void final_log(float *min_norm, long ID) {
        if (log_level >= 0) {
            fprintf(stdout, "search done, bucketing_time = %.2fs(%.2f M, %.3f Mops), dp_time = %.2fs(%.2f G, %.3f Gops), lift_time = %.2fs(%.2f M, %.3f Mops)\n", 
                            bucketing_time, bucketing_ops/1048576.0, bucketing_ops / bucketing_time / 1048576.0, 
                            dp_time, dp_ops/1073741824.0, dp_ops / dp_time / 1073741824.0, 
                            lift_time, lift_ops/1048576.0, lift_ops / lift_time / 1048576.0);
            if (log_level >= 1) {
                fprintf(stdout, "min_norm = [");
                for (long i = 0; i < ID; i++) {
                    fprintf(stdout, "%f", sqrt(min_norm[i]));
                    if (i < ID - 1) fprintf(stdout, " "); 
                }
                fprintf(stdout, "]\n");
            }
            fflush(stdout);
        }
    }
};

struct lsh_mblock_t {
    float *fvec = NULL;
    uint8_t *sh = NULL;
    long Mbound;
};

template <uint32_t nb, uint32_t shsize, uint32_t dh_dim>
struct lsh_mblock_provider_t {
    Pool_epi8_t<nb> *p;
    long target_index;
    float **b_ext_fp = NULL;
    float **b_head_fp = NULL;

    int32_t num_hbits, num_tbits;
    uint32_t *compress_pos;
    float *dual_vec;

    float *_fvec_list[LSH_MBLOCK_BATCH] = {};
    uint8_t *_sh_list[LSH_MBLOCK_BATCH] = {};
    long num_poped = 0;
    long num_total_buckets;
    double exp_bucket_size;
    long max_bucket_size;
    float bucket_radius2;

    void init(Pool_epi8_t<nb> *pool, long _target_index, float *_dual_vec, uint32_t *_compress_pos, int32_t _num_hbits, 
            float **&b_full_fp, float *min_norm, float **min_vec, double target_ratio, double qratio, lsh_profile_data_t *profile_data) {
        p = pool;
        target_index = _target_index;
        num_hbits = _num_hbits;
        num_tbits = shsize * 8 - num_hbits;
        dual_vec = _dual_vec;
        compress_pos = _compress_pos;

        // compute b_ext_fp, b_head_fp and b_full_fp
        do {
            Lattice_QP *b_full = p->basis->b_loc_QP(target_index, p->index_r);
            b_full_fp = (float **) NEW_MAT(p->index_r - target_index, p->index_r - target_index, sizeof(float));
            b_head_fp = (float **) NEW_MAT(dh_dim, shsize, sizeof(float));
            for (long i = 0; i < p->index_r - target_index; i++) {
                for (long j = 0; j <= i; j++) b_full_fp[i][j] = b_full->get_b().hi[i][j];
            }
            for (long i = 0; i < dh_dim; i++) {
                for (long j = 0; j <= i; j++) 
                    b_head_fp[i][j] = b_full_fp[i + p->index_l - target_index - dh_dim][j + p->index_l - target_index - dh_dim];
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

        // compute optimal bucket_radius2, max_bucket_size and num_total_buckets
        do {
            double bucket_radius_alpha = 0.0;
            do {
                // use lsh_C to find bucket_radius_alpha 
                // such that the expect bucket size is 
                // LSH_BUCKETING_OVERHEAD * sqrt(num_vec)
                // then compute the max_bucket_size
                double high = 1.0, low = 0.0;
                while (high - low > 0.001) {
                    double mid = (high + low) / 2.0;
                    if (lsh_C(dh_dim, mid) * p->num_vec * 2.0 < sqrt(p->num_vec) * LSH_BUCKETING_OVERHEAD) low = mid;
                    else high = mid;
                }
                bucket_radius_alpha = 0.5 * (high + low);
                max_bucket_size = (long) (2.5 * lsh_C(dh_dim, bucket_radius_alpha) * p->num_vec);
            } while (0);
            bucket_radius2 = pow(p->basis->gh(p->index_l - dh_dim, p->index_l) * bucket_radius_alpha, 2.0); 

            double base_time = pow(2, p->CSD * 0.332 - 23);             // CHANGE_WITH_ARCH
            const double bucketing_cost = 150.0 / 1073741824.0;         // CHANGE_WITH_ARCH
            const double dp_cost = 0.875 / 1073741824.0;                // CHANGE_WITH_ARCH

            exp_bucket_size = lsh_C(dh_dim, bucket_radius_alpha) * (p->num_vec+0.0) * 2.0;
            const double one_search_buckets = pow((p->num_vec / exp_bucket_size), 2.0) * 
                                            lsh_W(dh_dim, 1.0, target_ratio) / lsh_W(dh_dim, bucket_radius_alpha, target_ratio);
            double speed = 0.0;

            double best_over_search = 0.0;
            for (double _over_search = 0.1; _over_search < 3.0; _over_search *= 1.01) {
                double _pass_prob = 1.0 - pow(2.718281828, -_over_search);
                double exp_bucketing_time = _over_search * one_search_buckets * p->num_vec * bucketing_cost;
                double exp_dp_time = _over_search * one_search_buckets * exp_bucket_size * (exp_bucket_size - 1.0) * 0.5 * dp_cost;
                if (_pass_prob / (base_time + exp_bucketing_time + exp_dp_time * 1.2) > speed) {
                    speed = _pass_prob / (base_time + exp_bucketing_time + exp_dp_time * 1.2);
                    num_total_buckets = _over_search * one_search_buckets;
                    best_over_search = _over_search;
                }
            }
            if (profile_data->log_level >= 2) {
                double _pass_prob = 1.0 - pow(2.718281828, -best_over_search);
                double exp_bucketing_time = best_over_search * one_search_buckets * p->num_vec * bucketing_cost;
                double exp_dp_time = one_search_buckets * exp_bucket_size * (exp_bucket_size - 1.0) * 0.5 * dp_cost;
                fprintf(stdout, "init: over_search = %.2f, pass_prob = %.2f, exp_bucketing_time = %.2fs, exp_dp_time = %.2fs\n", 
                            best_over_search, _pass_prob, exp_bucketing_time, exp_dp_time);
            }
            double expect_one_bucket_time = p->num_vec * bucketing_cost + exp_bucket_size * exp_bucket_size * 0.5 * dp_cost * 1.6;
            long max_num_total_buckets = qratio * base_time / expect_one_bucket_time;
            if (profile_data->log_level >= 1) {
                printf("exp_bucket_size = %ld, num_buckets = %ld, boost = %.3f\n", (long) exp_bucket_size, _CEIL8(num_total_buckets), 
                        lsh_W(dh_dim, bucket_radius_alpha, target_ratio) / lsh_W(dh_dim, 1.0, target_ratio));
            }

            if (qratio != 0.0 && num_total_buckets > max_num_total_buckets) num_total_buckets = max_num_total_buckets; 
            num_total_buckets = _CEIL8(num_total_buckets);
        } while (0);

        // prepare _fvec_list and _sh_list
        do {
            const long FD8 = _CEIL8(p->index_r - target_index);
            profile_data->total_mblock = num_total_buckets;
            for (long i = 0; i < LSH_MBLOCK_BATCH; i++) {
                _fvec_list[i] = (float *) NEW_VEC(max_bucket_size * FD8, sizeof(float));
                _sh_list[i] = (uint8_t *) NEW_VEC(shsize * _CEIL8(max_bucket_size), sizeof(uint8_t));
            }
        } while (0);

        // lift all pool vectors and update min_norm, min_vec correspondingly
        TIMER_START;
        do {
            const long LD = p->index_l - target_index;
            const long FD = p->index_r - target_index;
            const long ID = p->index_l - dh_dim - target_index;
            const long FD8 = _CEIL8(FD);
            pthread_spinlock_t min_lock;
            pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);

            #pragma omp parallel for
            for (long thread = 0; thread < p->num_threads; thread++) {
                const long begin_ind = thread * p->num_vec / p->num_threads;
                const long end_ind = (thread + 1) * p->num_vec / p->num_threads;
                long ind = begin_ind;
                __attribute__ ((aligned (32))) int32_t ctmp[8 * p->vec_length];
                __attribute__ ((aligned (32))) float fvec[8 * 256];
                float idiag[256];
                for (long i = 0; i < FD; i++) idiag[i] = 1.0 / b_full_fp[i][i];
                while (ind < end_ind - 7) {
                    p->compute_coeff_b8(ctmp, ind);
                    set_zero_avx2(fvec, 8 * FD8);
                    for (long i = 0; i < p->CSD; i++) {
                        red_avx2(fvec + FD8 * 0, b_ext_fp[i], -ctmp[i*8+0], LD + i + 1);
                        red_avx2(fvec + FD8 * 1, b_ext_fp[i], -ctmp[i*8+1], LD + i + 1);
                        red_avx2(fvec + FD8 * 2, b_ext_fp[i], -ctmp[i*8+2], LD + i + 1);
                        red_avx2(fvec + FD8 * 3, b_ext_fp[i], -ctmp[i*8+3], LD + i + 1);
                        red_avx2(fvec + FD8 * 4, b_ext_fp[i], -ctmp[i*8+4], LD + i + 1);
                        red_avx2(fvec + FD8 * 5, b_ext_fp[i], -ctmp[i*8+5], LD + i + 1);
                        red_avx2(fvec + FD8 * 6, b_ext_fp[i], -ctmp[i*8+6], LD + i + 1);
                        red_avx2(fvec + FD8 * 7, b_ext_fp[i], -ctmp[i*8+7], LD + i + 1);
                    }
                    for (long i = LD - 1; i >= 0; i--) {
                        red_avx2(fvec + FD8 * 0, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 0)[i]), i+1);
                        red_avx2(fvec + FD8 * 1, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 1)[i]), i+1);
                        red_avx2(fvec + FD8 * 2, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 2)[i]), i+1);
                        red_avx2(fvec + FD8 * 3, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 3)[i]), i+1);
                        red_avx2(fvec + FD8 * 4, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 4)[i]), i+1);
                        red_avx2(fvec + FD8 * 5, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 5)[i]), i+1);
                        red_avx2(fvec + FD8 * 6, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 6)[i]), i+1);
                        red_avx2(fvec + FD8 * 7, b_full_fp[i], round(idiag[i] * (fvec + FD8 * 7)[i]), i+1);
                    }
                    float current_norm[8];
                    for (long j = 0; j < 8; j++) current_norm[j] = dot_avx2(fvec + FD8 * j, fvec + FD8 * j, FD8);
                    for (long i = 0; i < ID; i++) {
                        for (long j = 0; j < 8; j++) {
                            if (current_norm[j] < min_norm[i]) {
                                pthread_spin_lock(&min_lock);
                                if (current_norm[j] < min_norm[i]) {
                                    int has_one = 0;
                                    __attribute__ ((aligned (32))) float ftmpp[256];
                                    copy_avx2(ftmpp, fvec + FD8 * j, FD);
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
                                        copy_avx2(min_vec[i], fvec + FD8 * j, FD);
                                    }
                                }
                                pthread_spin_unlock(&min_lock);
                            }
                            current_norm[j] -= (fvec + FD8 * j)[i] * (fvec + FD8 * j)[i];
                        }
                    }
                    ind += 8;
                }
                while (ind < end_ind) {
                    p->compute_coeff(ctmp, ind);
                    set_zero_avx2(fvec, FD8);
                    for (long i = 0; i < p->CSD; i++) {
                        red_avx2(fvec, b_ext_fp[i], -ctmp[i], LD + i + 1);
                    }
                    for (long i = LD - 1; i >= 0; i--) {
                        red_avx2(fvec, b_full_fp[i], round(idiag[i] * fvec[i]), i+1);
                    }
                    float current_norm = dot_avx2(fvec, fvec, FD8);
                    for (long i = 0; i < ID; i++) {
                        if (current_norm < min_norm[i]) {
                            pthread_spin_lock(&min_lock);
                            if (current_norm < min_norm[i]) {
                                int has_one = 0;
                                __attribute__ ((aligned (32))) float ftmpp[256];
                                copy_avx2(ftmpp, fvec, FD);
                                for (long l = FD - 1; l >= FD - p->CSD; l--) {
                                    float qq = round(ftmpp[l] * idiag[l]);
                                    if (fabs(qq)  == 1.0f) {
                                        has_one = 1;
                                        break;
                                    }
                                    red_avx2(ftmpp, b_full_fp[l], qq, l+1);
                                }
                                if (has_one) {
                                    min_norm[i] = current_norm;
                                    copy_avx2(min_vec[i], fvec, FD);
                                }
                            }
                            pthread_spin_unlock(&min_lock);
                        }
                        current_norm -= fvec[i] * fvec[i];
                    }
                    ind++;
                }
            }
        } while (0);
        TIMER_END;
        profile_data->init_time += CURRENT_TIME;
        profile_data->init_ops += pool->num_vec;
        profile_data->init_log(min_norm, p->index_l - target_index - dh_dim);
    }
    void clear() {
        if (b_ext_fp) {FREE_MAT(b_ext_fp); b_ext_fp = NULL;}
        if (b_head_fp) {FREE_MAT(b_head_fp); b_head_fp = NULL;}
        for (long i = 0; i < LSH_MBLOCK_BATCH; i++) {
            if (_fvec_list[i]) {FREE_VEC((void *)_fvec_list[i]); _fvec_list[i] = NULL;}
            if (_sh_list[i]) {FREE_VEC((void *)_sh_list[i]); _sh_list[i] = NULL;}
        }
    }
    int batch_pop(lsh_mblock_t *mb, long batch_size, lsh_profile_data_t *profile_data) {
        if (num_poped >= num_total_buckets) return 0;
        TIMER_START;
        for (long i = 0; i < LSH_MBLOCK_BATCH; i++) {
            mb[i].fvec = NULL;
            mb[i].sh = NULL;
            mb->Mbound = 0;
        }
        const long num_pop = ((num_total_buckets - num_poped) > batch_size) ? 
                                batch_size : (num_total_buckets - num_poped);
        num_poped += num_pop;
        float idiag[dh_dim];
        for (long i = 0; i < dh_dim; i++) idiag[i] = 1.0 / b_head_fp[i][i];
        float **center = (float **) NEW_MAT(num_pop, dh_dim, sizeof(float));
        do {
            for (long i = 0; i < num_pop; i++) {
                for (long j = 0; j < dh_dim; j++) center[i][j] = Uniform_long(65536);
                for (long j = dh_dim - 1; j >= 0; j--) {
                    red_avx2(center[i], b_head_fp[j], round(idiag[j] * center[i][j]), dh_dim);
                }
            }
        } while (0);


        // collect fvec and put them to local_fvec_dst
        const long FD = p->index_r - target_index;
        const long LD = p->index_l - target_index;
        const long ID = p->index_l - dh_dim - target_index;
        const long FD8 = _CEIL8(FD);
        uint64_t **local_Mbound_list = (uint64_t **) NEW_MAT(num_pop, p->num_threads, sizeof(uint64_t));
        #pragma omp parallel for
        for (long thread = 0; thread < p->num_threads; thread++) {
            float *local_fvec_dst[LSH_MBLOCK_BATCH];
            uint64_t local_Mbound[LSH_MBLOCK_BATCH] = {};
            const uint64_t max_bound = ((thread + 1) * max_bucket_size) / p->num_threads - (thread * max_bucket_size) / p->num_threads;
            for (long i = 0; i < num_pop; i++) {
                local_fvec_dst[i] = _fvec_list[i] + FD8 * ((thread * max_bucket_size) / p->num_threads);
            }

            const long begin_ind = (thread * p->num_vec) / p->num_threads;
            const long end_ind = ((thread+1) * p->num_vec) / p->num_threads;
            long ind = begin_ind;
            const long adj_ind = begin_ind + _CEIL8((end_ind - begin_ind) / 10);

            __attribute__ ((aligned (32))) int32_t ctmp[8 * p->vec_length];
            __attribute__ ((aligned (32))) float ftmp[8 * _CEIL8(dh_dim)];
            __attribute__ ((aligned (32))) float fvec[8 * 256];
            while (ind < end_ind - 7) {
                if (ind == adj_ind && thread == 0) {
                    long num_pass = 0;
                    double exp_pass = (ind - begin_ind) * max_bucket_size * 0.8 / p->num_vec * num_pop;
                    for (long i = 0; i < num_pop; i++) {
                        num_pass += local_Mbound[i];
                    }
                    bucket_radius2 *= pow(exp_pass / num_pass, 2.0 / dh_dim);
                }
                const __m512 fbound = _mm512_set1_ps(bucket_radius2);
                p->compute_coeff_b8(ctmp, ind);
                set_zero_avx2(fvec, 8 * FD8);
                for (long i = 0; i < p->CSD; i++) {
                    red_avx2(fvec + FD8 * 0, b_ext_fp[i], -ctmp[i*8+0], LD + i + 1);
                    red_avx2(fvec + FD8 * 1, b_ext_fp[i], -ctmp[i*8+1], LD + i + 1);
                    red_avx2(fvec + FD8 * 2, b_ext_fp[i], -ctmp[i*8+2], LD + i + 1);
                    red_avx2(fvec + FD8 * 3, b_ext_fp[i], -ctmp[i*8+3], LD + i + 1);
                    red_avx2(fvec + FD8 * 4, b_ext_fp[i], -ctmp[i*8+4], LD + i + 1);
                    red_avx2(fvec + FD8 * 5, b_ext_fp[i], -ctmp[i*8+5], LD + i + 1);
                    red_avx2(fvec + FD8 * 6, b_ext_fp[i], -ctmp[i*8+6], LD + i + 1);
                    red_avx2(fvec + FD8 * 7, b_ext_fp[i], -ctmp[i*8+7], LD + i + 1);
                }
                for (long l = 0; l < dh_dim; l += 8) {
                    for (long i = 0; i < 8; i++) {
                        _mm256_store_ps(ftmp + i * _CEIL8(dh_dim) + l, _mm256_loadu_ps(fvec + i * FD8 + ID + l));
                    }
                }
                uint64_t dst[_CEIL64(LSH_MBLOCK_BATCH)/8] = {};
                for (long i = 0; i < 8; i++) __size_red_npcheck(dst + i * (_CEIL64(LSH_MBLOCK_BATCH)/64), ftmp + i * _CEIL8(dh_dim), center, b_head_fp, idiag, num_pop, fbound);
                for (long i = 0; i < 8; i++) {
                    uint64_t *_dst = dst + i * (_CEIL64(LSH_MBLOCK_BATCH)/64);
                    for (long j = 0; j < _CEIL64(LSH_MBLOCK_BATCH)/64; j++) {
                        while (_dst[j]) {
                            uint64_t r = __builtin_ctzll(_dst[j]);
                            _dst[j] ^= (1ULL << r);
                            if (local_Mbound[r+j*64] < max_bound) {
                                copy_avx2(local_fvec_dst[r+j*64] + local_Mbound[r+j*64] * FD8, fvec + i * FD8, FD);
                                local_Mbound[r+j*64]++;
                            }
                        }
                    }
                }
                ind += 8;
            }
            while (ind < end_ind) {
                const __m512 fbound = _mm512_set1_ps(bucket_radius2);
                p->compute_coeff(ctmp, ind);
                set_zero_avx2(fvec, FD8);
                for (long i = 0; i < p->CSD; i++) {
                    red_avx2(fvec, b_ext_fp[i], -ctmp[i], LD + i + 1);
                }
                for (long l = 0; l < dh_dim; l += 8) {
                    _mm256_store_ps(ftmp + l, _mm256_loadu_ps(fvec + ID + l));
                }
                uint64_t dst[_CEIL64(LSH_MBLOCK_BATCH)/64] = {};
                __size_red_npcheck(dst, ftmp, center, b_head_fp, idiag, num_pop, fbound);
                for (long i = 0; i < _CEIL64(LSH_MBLOCK_BATCH)/64; i++) {
                    while (dst[i]) {
                        uint64_t r = __builtin_ctzll(dst[i]);
                        dst[i] ^= (1ULL << r);
                        if (local_Mbound[r+i*64] < max_bound) {
                            copy_avx2(local_fvec_dst[r+i*64] + local_Mbound[r+i*64] * FD8, fvec, FD);
                            local_Mbound[r+i*64]++;
                        }
                    }
                }
                ind++;
            }
            for (long i = 0; i < num_pop; i++) {
                local_Mbound_list[i][thread] = local_Mbound[i];
            }
        }
        
        // move fvec to the correct place
        #pragma omp parallel for
        for (long i = 0; i < num_pop; i++) {
            long right_ind = local_Mbound_list[i][0];
            for (long j = 1; j < p->num_threads; j++) {
                memmove(_fvec_list[i] + right_ind * FD8, _fvec_list[i] + (j * max_bucket_size) / p->num_threads * FD8, 
                        local_Mbound_list[i][j] * FD8 * sizeof(float));
                right_ind += local_Mbound_list[i][j];
            }
            mb[i].Mbound = right_ind;
            mb[i].fvec = _fvec_list[i];
            mb[i].sh = _sh_list[i];
        }

        // compute sh from fvec
        #pragma omp parallel for 
        for (long thread = 0; thread < p->num_threads; thread++) {
            for (long ip = 0; ip < num_pop; ip++) {
                const long begin_ind = (thread * mb[ip].Mbound) / p->num_threads;
                const long end_ind = ((thread+1) * mb[ip].Mbound) / p->num_threads;
                long ind = begin_ind;

                __attribute__ ((aligned (32))) float ftmp[8 * _CEIL8(dh_dim)];
                while (ind < end_ind - 7) {
                    for (long l = 0; l < dh_dim; l += 8) {
                        for (long i = 0; i < 8; i++) {
                            _mm256_store_ps(ftmp + i * _CEIL8(dh_dim) + l, _mm256_loadu_ps(mb[ip].fvec + (ind+i) * FD8 + ID + l));
                        }
                    }
                    // compute h part
                    for (long k = 0; k < num_hbits; k += 8) {
                        __m256 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
                        AVX2_DP_2X8((ftmp+0*_CEIL8(dh_dim)), (ftmp+1*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst0, dst1);
                        AVX2_DP_2X8((ftmp+2*_CEIL8(dh_dim)), (ftmp+3*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst2, dst3);
                        AVX2_DP_2X8((ftmp+4*_CEIL8(dh_dim)), (ftmp+5*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst4, dst5);
                        AVX2_DP_2X8((ftmp+6*_CEIL8(dh_dim)), (ftmp+7*_CEIL8(dh_dim)), (dual_vec+k*_CEIL8(dh_dim)), _CEIL8(dh_dim), dst6, dst7);
                        mb[ip].sh[shsize * (ind+0) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst0), 24));
                        mb[ip].sh[shsize * (ind+1) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst1), 24));
                        mb[ip].sh[shsize * (ind+2) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst2), 24));
                        mb[ip].sh[shsize * (ind+3) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst3), 24));
                        mb[ip].sh[shsize * (ind+4) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst4), 24));
                        mb[ip].sh[shsize * (ind+5) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst5), 24));
                        mb[ip].sh[shsize * (ind+6) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst6), 24));
                        mb[ip].sh[shsize * (ind+7) + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst7), 24));
                    }
                    // compute t part
                    float *ptr0 = mb[ip].fvec + (ind+0) * FD8 + LD;
                    float *ptr1 = mb[ip].fvec + (ind+1) * FD8 + LD;
                    float *ptr2 = mb[ip].fvec + (ind+2) * FD8 + LD;
                    float *ptr3 = mb[ip].fvec + (ind+3) * FD8 + LD;
                    float *ptr4 = mb[ip].fvec + (ind+4) * FD8 + LD;
                    float *ptr5 = mb[ip].fvec + (ind+5) * FD8 + LD;
                    float *ptr6 = mb[ip].fvec + (ind+6) * FD8 + LD;
                    float *ptr7 = mb[ip].fvec + (ind+7) * FD8 + LD;
                    for (long k = 0; k < num_tbits; k += 8) {
                        __attribute__ ((aligned (32))) float ttmp[64];
                        for (long i = k; i < k + 8; i++) {
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
                        mb[ip].sh[shsize*(ind+0)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 0*8));
                        mb[ip].sh[shsize*(ind+1)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 1*8));
                        mb[ip].sh[shsize*(ind+2)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 2*8));
                        mb[ip].sh[shsize*(ind+3)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 3*8));
                        mb[ip].sh[shsize*(ind+4)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 4*8));
                        mb[ip].sh[shsize*(ind+5)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 5*8));
                        mb[ip].sh[shsize*(ind+6)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 6*8));
                        mb[ip].sh[shsize*(ind+7)+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp + 7*8));
                    }
                    ind += 8;
                }
                while (ind < end_ind) {
                    for (long l = 0; l < dh_dim; l += 8) {
                        _mm256_store_ps(ftmp + l, _mm256_loadu_ps(mb[ip].fvec + ind * FD8 + ID + l));
                    }
                    // compute h part
                    for (long k = 0; k < num_hbits; k += 8) {
                        __m256 dst;
                        AVX2_DP_1X8(ftmp, (dual_vec + (k * _CEIL8(dh_dim))), _CEIL8(dh_dim), dst);
                        mb[ip].sh[shsize * ind + k/8] = _mm256_movemask_ps((__m256)_mm256_slli_epi32(_mm256_cvtps_epi32(dst), 24));
                    }
                    // compute t part
                    float *ptr0 = mb[ip].fvec + ind * FD8 + LD;
                    for (long k = 0; k < num_tbits; k += 8) {
                        __attribute__ ((aligned (32))) float ttmp[8];
                        for (long i = k; i < k+8; i++) {
                            ttmp[i-k] = (k < p->CSD) ? ptr0[compress_pos[6*i+0]] : 
                                        ptr0[compress_pos[6*i+0]] + ptr0[compress_pos[6*i+1]] + ptr0[compress_pos[6*i+2]] - 
                                        ptr0[compress_pos[6*i+3]] - ptr0[compress_pos[6*i+4]] - ptr0[compress_pos[6*i+5]];
                            mb[ip].sh[shsize*ind+num_hbits/8+k/8] = _mm256_movemask_ps(_mm256_load_ps(ttmp));
                        }
                    }
                    ind++;
                }
            }
        }

        #if 0
        // adjust bucket_radius2
        long avg_bucket_size = 0;
        for (long i = 0; i < num_pop; i++) avg_bucket_size += mb[i].Mbound;
        avg_bucket_size /= num_pop;
        bucket_radius2 *= pow(exp_bucket_size / avg_bucket_size, 2.0 / dh_dim);
        #endif

        FREE_MAT(center);
        FREE_MAT((void **)local_Mbound_list);
        TIMER_END;
        profile_data->bucketing_time += CURRENT_TIME;
        profile_data->bucketing_ops += p->num_vec * num_pop;

        return 1;
    }
    // num_c must be a multiple of 8
    inline void __size_red_npcheck(uint64_t *ret, float *v, float **c, float **b, float *idiag, long num_c, __m512 bound) {
        __attribute__ ((aligned (32))) float tmp[16 * dh_dim];
        __attribute__ ((aligned (32))) float q[16];
        for (long l = 0; l < num_c; l += 8) {
            for (long i = 0; i < 8; i++) {
                for (long j = 0; j < dh_dim; j += 8) {
                    _mm256_store_ps(tmp + (i * 2 + 0) * dh_dim + j, _mm256_add_ps(_mm256_load_ps(v + j), _mm256_load_ps(c[l+i] + j)));
                    _mm256_store_ps(tmp + (i * 2 + 1) * dh_dim + j, _mm256_sub_ps(_mm256_load_ps(v + j), _mm256_load_ps(c[l+i] + j)));
                }
            }
            for (long k = dh_dim - 1; k >= 0; k--) {
                for (long i = 0; i < 16; i++) q[i] = tmp[i * dh_dim + k];
                _mm256_store_ps(q+0, _mm256_round_ps(_mm256_mul_ps(_mm256_load_ps(q+0), _mm256_broadcast_ss(idiag+k)), 0x08));
                _mm256_store_ps(q+8, _mm256_round_ps(_mm256_mul_ps(_mm256_load_ps(q+8), _mm256_broadcast_ss(idiag+k)), 0x08));
                for (long j = 0; j < dh_dim; j += 8) {
                    _mm256_store_ps(tmp + 0x0 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x0), _mm256_load_ps(tmp + 0x0 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x1 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x1), _mm256_load_ps(tmp + 0x1 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x2 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x2), _mm256_load_ps(tmp + 0x2 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x3 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x3), _mm256_load_ps(tmp + 0x3 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x4 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x4), _mm256_load_ps(tmp + 0x4 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x5 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x5), _mm256_load_ps(tmp + 0x5 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x6 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x6), _mm256_load_ps(tmp + 0x6 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x7 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x7), _mm256_load_ps(tmp + 0x7 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x8 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x8), _mm256_load_ps(tmp + 0x8 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0x9 * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0x9), _mm256_load_ps(tmp + 0x9 * dh_dim + j)));
                    _mm256_store_ps(tmp + 0xA * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0xA), _mm256_load_ps(tmp + 0xA * dh_dim + j)));
                    _mm256_store_ps(tmp + 0xB * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0xB), _mm256_load_ps(tmp + 0xB * dh_dim + j)));
                    _mm256_store_ps(tmp + 0xC * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0xC), _mm256_load_ps(tmp + 0xC * dh_dim + j)));
                    _mm256_store_ps(tmp + 0xD * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0xD), _mm256_load_ps(tmp + 0xD * dh_dim + j)));
                    _mm256_store_ps(tmp + 0xE * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0xE), _mm256_load_ps(tmp + 0xE * dh_dim + j)));
                    _mm256_store_ps(tmp + 0xF * dh_dim + j, _mm256_fnmadd_ps(_mm256_load_ps(b[k]+j), _mm256_broadcast_ss(q + 0xF), _mm256_load_ps(tmp + 0xF * dh_dim + j)));
                }
            }
            __m256 accp0, accp1, accp2, accp3, accp4, accp5, accp6, accp7;
            __m256 accn0, accn1, accn2, accn3, accn4, accn5, accn6, accn7;
            const __m256i zero = _mm256_setzero_ps();
            accp0 = accp1 = accp2 = accp3 = accp4 = accp5 = accp6 = accp7 = zero;
            accn0 = accn1 = accn2 = accn3 = accn4 = accn5 = accn6 = accn7 = zero;
            for (long j = 0; j < dh_dim; j += 8) {
                accp0 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x0 * dh_dim + j), _mm256_load_ps(tmp + 0x0 * dh_dim + j), accp0);
                accn0 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x1 * dh_dim + j), _mm256_load_ps(tmp + 0x1 * dh_dim + j), accn0);
                accp1 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x2 * dh_dim + j), _mm256_load_ps(tmp + 0x2 * dh_dim + j), accp1);
                accn1 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x3 * dh_dim + j), _mm256_load_ps(tmp + 0x3 * dh_dim + j), accn1);
                accp2 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x4 * dh_dim + j), _mm256_load_ps(tmp + 0x4 * dh_dim + j), accp2);
                accn2 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x5 * dh_dim + j), _mm256_load_ps(tmp + 0x5 * dh_dim + j), accn2);
                accp3 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x6 * dh_dim + j), _mm256_load_ps(tmp + 0x6 * dh_dim + j), accp3);
                accn3 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x7 * dh_dim + j), _mm256_load_ps(tmp + 0x7 * dh_dim + j), accn3);
                accp4 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x8 * dh_dim + j), _mm256_load_ps(tmp + 0x8 * dh_dim + j), accp4);
                accn4 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0x9 * dh_dim + j), _mm256_load_ps(tmp + 0x9 * dh_dim + j), accn4);
                accp5 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0xA * dh_dim + j), _mm256_load_ps(tmp + 0xA * dh_dim + j), accp5);
                accn5 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0xB * dh_dim + j), _mm256_load_ps(tmp + 0xB * dh_dim + j), accn5);
                accp6 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0xC * dh_dim + j), _mm256_load_ps(tmp + 0xC * dh_dim + j), accp6);
                accn6 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0xD * dh_dim + j), _mm256_load_ps(tmp + 0xD * dh_dim + j), accn6);
                accp7 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0xE * dh_dim + j), _mm256_load_ps(tmp + 0xE * dh_dim + j), accp7);
                accn7 = _mm256_fmadd_ps(_mm256_load_ps(tmp + 0xF * dh_dim + j), _mm256_load_ps(tmp + 0xF * dh_dim + j), accn7);
            }
            accp0 = _mm256_hadd_ps(accp0, accp1);
            accp2 = _mm256_hadd_ps(accp2, accp3);
            accp4 = _mm256_hadd_ps(accp4, accp5);
            accp6 = _mm256_hadd_ps(accp6, accp7);
            accn0 = _mm256_hadd_ps(accn0, accn1);
            accn2 = _mm256_hadd_ps(accn2, accn3);
            accn4 = _mm256_hadd_ps(accn4, accn5);
            accn6 = _mm256_hadd_ps(accn6, accn7);
            accp0 = _mm256_hadd_ps(accp0, accp2);
            accp4 = _mm256_hadd_ps(accp4, accp6);
            accn0 = _mm256_hadd_ps(accn0, accn2);
            accn4 = _mm256_hadd_ps(accn4, accn6);
            __m256 accp = _mm256_add_ps(_mm256_permute2f128_ps(accp0, accp4, 48), _mm256_permute2f128_ps(accp0, accp4, 33));
            __m256 accn = _mm256_add_ps(_mm256_permute2f128_ps(accn0, accn4, 48), _mm256_permute2f128_ps(accn0, accn4, 33));
            uint64_t pos = _mm256_cmp_ps_mask(_mm512_castps512_ps256(bound), accp, 30) | _mm256_cmp_ps_mask(_mm512_castps512_ps256(bound), accn, 30);
            ret[l/64] |= pos << (l % 64);
        }
        return;
    }
};


template <uint32_t nb>
template <uint32_t shsize, uint32_t l1_block, uint32_t l2_block>
int Pool_epi8_t<nb>::__mblock_sh_search(lsh_mblock_t mb, int32_t threshold, uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, lsh_profile_data_t *profile_data,
                                        long target_index, uint32_t dh_dim, float **b_full_fp, float *min_norm, float **min_vec, pthread_spinlock_t &min_lock) {
    uint32_t *_ptr_buffer = *ptr_buffer;
    long _ptr_buffer_num = *ptr_buffer_num;
    long _ptr_buffer_size = *ptr_buffer_size;
    for (long Ind = 0; Ind < mb.Mbound; Ind += l2_block) {
        for (long Jnd = Ind; Jnd < mb.Mbound; Jnd += l2_block) {
            if (Ind == Jnd) {
                // a triangular block
                const long Ibound = (Ind + l2_block > mb.Mbound) ? mb.Mbound : Ind + l2_block;
                const long Jbound = (Jnd + l2_block > mb.Mbound) ? mb.Mbound : Jnd + l2_block;
                long ind = Ind;
                while (ind <= Ibound - l1_block) {
                    long jnd = ind;
                    // the first triangular block
                    _process_nshl1_triblock<shsize>(mb.sh + ind * shsize, ind, l1_block, threshold, 
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    jnd += l1_block;
                    // square blocks
                    while (jnd <= Jbound - l1_block) {
                        _process_nshl1_block<shsize>(mb.sh + ind * shsize, mb.sh + jnd * shsize, 
                                                ind, jnd, l1_block, l1_block, threshold,
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                        jnd += l1_block;
                    }
                    if (jnd < Jbound) {
                        _process_nshl1_block<shsize>(mb.sh + ind * shsize, mb.sh + jnd * shsize, 
                                                ind, jnd, l1_block, Jbound - jnd, threshold,
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    }
                    ind += l1_block;
                }
                if (ind < Ibound) {
                    long jnd = ind;
                    _process_nshl1_triblock<shsize>(mb.sh + ind * shsize, ind, Ibound - ind, threshold,
                                                &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                }
            } else {
                // normal block
                const long Ibound = (Ind + l2_block > mb.Mbound) ? mb.Mbound : Ind + l2_block;
                const long Jbound = (Jnd + l2_block > mb.Mbound) ? mb.Mbound : Jnd + l2_block;
                long ind = Ind;
                while (ind <= Ibound - l1_block) {
                    long jnd = Jnd;
                    while (jnd <= Jbound - l1_block) {
                        _process_nshl1_block<shsize>(mb.sh + ind * shsize, mb.sh + jnd * shsize,
                                                    ind, jnd, l1_block, l1_block, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                        jnd += l1_block;
                    }
                    if (jnd < Jbound) {
                        _process_nshl1_block<shsize>(mb.sh + ind * shsize, mb.sh + jnd * shsize,
                                                    ind, jnd, l1_block, Jbound - jnd, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    }
                    ind += l1_block;
                }
                if (ind < Ibound) {
                    long jnd = Jnd;
                    while (jnd <= Jbound - l1_block) {
                        _process_nshl1_block<shsize>(mb.sh + ind * shsize, mb.sh + jnd * shsize,
                                            ind, jnd, Ibound - ind, l1_block, threshold,
                                            &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                        jnd += l1_block;
                    }
                    if (jnd < Jbound) {
                        _process_nshl1_block<shsize>(mb.sh + ind * shsize, mb.sh + jnd * shsize,
                                            ind, jnd, Ibound - ind, Jbound - jnd, threshold,
                                            &_ptr_buffer, &_ptr_buffer_size, &_ptr_buffer_num);
                    }
                }
            }
            if (_ptr_buffer_num > 1048576) {
                __lift_buffer(mb, target_index, dh_dim, b_full_fp, 
                        &_ptr_buffer, (uint64_t *)(&_ptr_buffer_size), (uint64_t *)&_ptr_buffer_num, 
                        min_norm, min_vec, min_lock, profile_data);
            }
        }
    }
    pthread_spin_lock(&profile_data->profile_lock);
    profile_data->dp_ops += mb.Mbound * (mb.Mbound + 1) / 2;
    profile_data->num_mblock++;
    pthread_spin_unlock(&profile_data->profile_lock);
    *ptr_buffer = _ptr_buffer;
    *ptr_buffer_size = _ptr_buffer_size;
    *ptr_buffer_num = _ptr_buffer_num;
    
    return 1;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::__lift_buffer(lsh_mblock_t mb, long target_index, uint32_t dh_dim, float **b_full_fp, 
                                    uint32_t **ptr_buffer, uint64_t *ptr_buffer_size, uint64_t *ptr_buffer_num, 
                                    float *min_norm, float **min_vec, pthread_spinlock_t &min_lock, lsh_profile_data_t *profile_data) {
    const long FD = index_r - target_index;
    const long ID = index_l - dh_dim - target_index;
    const long FD8 = _CEIL8(FD);

    for (long thread = 0; thread < 1; thread++) {
        // local min data initialization
        __attribute__ ((aligned (32))) float _min_norm[vec_length];
        float **_min_vec = (float **) NEW_MAT(ID, FD, sizeof(float));
        pthread_spin_lock(&min_lock);
        copy_avx2(_min_norm, min_norm, ID);
        pthread_spin_unlock(&min_lock);

        // main lifting
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
                        sub_avx2(ftmp + FD8 * i, mb.fvec + ptr_buffer[thread][(ind+i)*3+1]*FD8, mb.fvec + ptr_buffer[thread][(ind+i)*3+2]*FD8, FD8);
                    } else {
                        add_avx2(ftmp + FD8 * i, mb.fvec + ptr_buffer[thread][(ind+i)*3+1]*FD8, mb.fvec + ptr_buffer[thread][(ind+i)*3+2]*FD8, FD8);
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
                    sub_avx2(ftmp, mb.fvec + ptr_buffer[thread][ind*3+1]*FD8, mb.fvec + ptr_buffer[thread][ind*3+2]*FD8, FD8);
                } else {
                    add_avx2(ftmp, mb.fvec + ptr_buffer[thread][ind*3+1]*FD8, mb.fvec + ptr_buffer[thread][ind*3+2]*FD8, FD8);
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
    return 0;
}

template <uint32_t nb>
template <uint32_t shsize, uint32_t dh_dim>
int Pool_epi8_t<nb>::_lsfsh_insert(long target_index, double eta, long log_level, 
                            float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_ratio, double target_length, double qratio) {
    /////// check params ///////
    if (target_index + dh_dim > index_l) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::lsfsh_insert: target_index(%ld) + dh_dim(%u) > index_l(%ld), nothing done.\n", 
                nb, target_index, dh_dim, index_l);
        return -1;
    }
    if (target_index < 0) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::lsfsh_insert: negetive target_index(%ld), nothing done.\n", nb, target_index);
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
    lsh_profile_data_t profile_data;
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
    lsh_mblock_provider_t<nb, shsize, dh_dim> mblock_provider;
    mblock_provider.init(this, target_index, dual_vec, compress_pos, num_hbits, b_full_fp, min_norm, min_vec, target_ratio, qratio, &profile_data);
    lsh_mblock_t mb[LSH_MBLOCK_BATCH];
    const long LSH_RBATCH = ((LSH_MBLOCK_BATCH / num_threads) * num_threads) > 0 ? ((LSH_MBLOCK_BATCH / num_threads) * num_threads) : LSH_MBLOCK_BATCH;
    while (mblock_provider.batch_pop(mb, LSH_RBATCH, &profile_data)) {
        int wild_mb = 0;
        long num_mb = 0;
        for (long i = 0; i < LSH_RBATCH; i++) {
            if (mb[i].sh) {
                num_mb++;
                if (mb[i].Mbound == mblock_provider.max_bucket_size) wild_mb += 1;
                if (2 * mb[i].Mbound < mblock_provider.max_bucket_size) wild_mb += 1;
            }
        }
        if (wild_mb > 0.5 * num_mb) {
            fprintf(stderr, "[Warning] Pool_epi8_t<%u>::lsfsh_insert: strange bucket size, exp_bucket_size = %ld, real size = \n", 
                    nb, (long)(mblock_provider.max_bucket_size * 0.8));
            for (long i = 0; i < num_mb; i++) {
                if (mb[i].sh) printf("%ld ", mb[i].Mbound);
            }
            printf("\n");
        }
        
        
        pthread_spinlock_t min_lock;
        pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);
        double lift_time = 0.0;
        double search_time = 0.0;
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++) {
            const long begin_ind = thread * num_mb / num_threads;
            const long end_ind = (thread + 1) * num_mb / num_threads;
            double _lift_time = 0.0;
            double _search_time = 0.0;
            for (long ind = begin_ind; ind < end_ind; ind++) {
                TIMER_START;
                __mblock_sh_search<shsize, LSH_L1_BLOCK, LSH_L2_BLOCK>(
                    mb[ind], threshold, ptr_buffer+thread, ptr_buffer_size+thread, ptr_buffer_num+thread, &profile_data,
                    target_index, dh_dim, b_full_fp, min_norm, min_vec, min_lock);
                TIMER_END;
                _search_time += CURRENT_TIME;
                TIMER_START;
                __lift_buffer(mb[ind], target_index, dh_dim, b_full_fp, 
                        ptr_buffer+thread, ptr_buffer_size+thread, ptr_buffer_num+thread, 
                        min_norm, min_vec, min_lock, &profile_data);
                TIMER_END;
                _lift_time += CURRENT_TIME;
            }
            pthread_spin_lock(&min_lock);
            if (_lift_time > lift_time) lift_time = _lift_time;
            if (_search_time > search_time) search_time = _search_time;
            pthread_spin_unlock(&min_lock);
        }
        profile_data.lift_time += lift_time;
        profile_data.dp_time += search_time;
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
int Pool_epi8_t<nb>::lsfsh_insert(long target_index, double eta, long log_level, double target_length, double stop_ratio, double qratio) {
    /////// basic information of the lifting ///////
    const double tail_gh = sqrt(gh2);
    const double dual_gh = basis->gh(index_l - LSH_DEFAULT_DHDIM, index_l);
    const double lift_gh = basis->gh(target_index, index_l);
    const double tail_exp_ratio = sqrt(cvec[3*(num_vec/2)+2] * 4.0) / _ratio / tail_gh;

    double tail_exp_alpha;
    do {
        double min_exp_length = lift_gh;
        for (double _alpha = 0.0; _alpha < 0.5; _alpha += 0.01) {
            uint64_t _num_lift = num_vec * (num_vec - 1.0) * pow(1 - _alpha * _alpha, CSD * 0.5);
            double _tail_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * _alpha);
            double _lift_length = lift_gh / pow(_num_lift, 1.0/(index_l - target_index));
            double _length = sqrt(_tail_length * _tail_length + _lift_length * _lift_length);
            if (_length < min_exp_length) {
                tail_exp_alpha = _alpha;
                min_exp_length = _length;
            }
        }
    } while (0);

    const uint64_t exp_num_lift = num_vec * (num_vec - 1.0) * pow(1 - tail_exp_alpha * tail_exp_alpha, CSD*0.5);
    const double tail_exp_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * tail_exp_alpha);
    const double lift_exp_length = lift_gh / pow(exp_num_lift, 1.0/(index_l - target_index));
    const double exp_length = sqrt(tail_exp_length * tail_exp_length + lift_exp_length * lift_exp_length);
    const bool unique_target = (target_length != 0.0) && (target_length < 0.96 * exp_length);

    // the probability that the solution with exp_length is filtered out
    double tail_alpha_prob_list[64];
    do {
        if (!unique_target) {
            for (long i = 0; i < 50; i++) {
                double _alpha = 0.01 * i;
                double _tail_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * _alpha);
                double _num_lift = num_vec * (num_vec - 1.0) * pow(1 - _alpha * _alpha, CSD * 0.5);
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
                        target_length * sqrt(LSH_DEFAULT_DHDIM) / sqrt(index_r - target_index) :
                        lift_exp_length * sqrt(LSH_DEFAULT_DHDIM) / sqrt(index_l - target_index);

    if (log_level >= 1) {
        printf("input: tail_gh = %.2f, dual_gh = %.2f, lift_gh = %.2f\n", tail_gh, dual_gh, lift_gh);
        printf("param: tail_exp_alpha = %.2f, tail_exp_ratio = %.2f, tail_exp_length = %.2f\n", tail_exp_alpha, tail_exp_ratio, tail_exp_length);
        printf("param: exp_num_lift = %lu = (%.2f)^{-%ld}, lift_exp_length = %.2f, exp_length = %.2f, dual_exp_length = %.2f(%.2f, type %s)\n",
                exp_num_lift, pow(exp_num_lift, -1.0/(index_l - target_index)), index_l-target_index, lift_exp_length, exp_length, 
                dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
    } else if (log_level == 0) {
        if (target_length != 0.0) {
            printf("param: exp_length = %.2f, target_length = %.2f(%.3f), dual_exp_length = %.2f(%.2f, type %s), ",exp_length, target_length, 
                    target_length / exp_length, dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
        } else {
            printf("param: exp_length = %.2f, dual_exp_length = %.2f(%.2f, type %s), ", exp_length, 
                    dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
        }
    }

    /////// choosing param ///////
    // output values
    float *dual_vec = (float *) NEW_VEC(LSH_DEFAULT_SHSIZE * 8 * _CEIL8(LSH_DEFAULT_DHDIM), sizeof(float));
    uint32_t *compress_pos = (uint32_t *) NEW_VEC(LSH_DEFAULT_SHSIZE * 8 * 6, sizeof(uint32_t));
    int32_t num_hbits, num_tbits, threshold;
    // input values
    Lattice_QP *b_mid = basis->b_loc_QP(index_l - LSH_DEFAULT_DHDIM, index_l);
    TIMER_START;
    if (log_level >= 2) printf("param: ");
    _opt_nsh_threshold(dual_vec, compress_pos, num_hbits, num_tbits, threshold, 
                        b_mid, LSH_DEFAULT_SHSIZE, exp_length, tail_alpha_prob_list, log_level);
    TIMER_END;
    delete b_mid;
    if (log_level >= 0) {
        if (log_level >= 1) printf("param: ");
        printf("opt_time = %.2fs, num_hbits = %d, num_tbits = %d, threshold = %d\n", CURRENT_TIME, num_hbits, num_tbits, threshold);
    }

    /////// call the lifting kernel ///////
    int ret = _lsfsh_insert<LSH_DEFAULT_SHSIZE, LSH_DEFAULT_DHDIM>
                (target_index, eta, log_level, dual_vec, compress_pos, num_hbits, threshold, dual_exp_length/dual_gh, unique_target ? target_length : 0.0, qratio);

    FREE_VEC(dual_vec);
    FREE_VEC((void *)compress_pos);
    return ret;
}

template <uint32_t nb>
template <uint32_t shsize, uint32_t dh_dim>
int Pool_epi8_t<nb>::_show_lsfsh_insert(long target_index, double eta, long log_level, 
                            float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_ratio, double target_length, double qratio) {
    /////// check params ///////
    if (target_index + dh_dim > index_l) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::lsfsh_insert: target_index(%ld) + dh_dim(%u) > index_l(%ld), nothing done.\n", 
                nb, target_index, dh_dim, index_l);
        return -1;
    }
    if (target_index < 0) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::lsfsh_insert: negetive target_index(%ld), nothing done.\n", nb, target_index);
        return -1;
    }

    const long FD = index_r - target_index;
    const long ID = index_l - dh_dim - target_index;


    /////// min data initialization ///////
    __attribute__ ((aligned (32))) float min_norm[vec_length];
    float **min_vec = (float **) NEW_MAT(ID, FD, sizeof(float));
    for (long i = 0; i < ID; i++) min_norm[i] = (float) (0.99995 * basis->get_B().hi[i + target_index]);
    if (target_length != 0.0) {
        for (long i = 0; i < ID; i++) {
            if (min_norm[i] > 1.00001 * target_length * target_length) min_norm[i] = 1.00001 * target_length * target_length;
        }
    }


    /////// profiling data and buffer initialization ///////
    lsh_profile_data_t profile_data;
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
    lsh_mblock_provider_t<nb, shsize, dh_dim> mblock_provider;
    mblock_provider.init(this, target_index, dual_vec, compress_pos, num_hbits, b_full_fp, min_norm, min_vec, target_ratio, qratio, &profile_data);
    lsh_mblock_t mb[LSH_MBLOCK_BATCH];
    const long LSH_RBATCH = ((LSH_MBLOCK_BATCH / num_threads) * num_threads) > 0 ? ((LSH_MBLOCK_BATCH / num_threads) * num_threads) : LSH_MBLOCK_BATCH;
    while (mblock_provider.batch_pop(mb, LSH_RBATCH, &profile_data)) {
        int wild_mb = 0;
        long num_mb = 0;
        for (long i = 0; i < LSH_RBATCH; i++) {
            if (mb[i].sh) {
                num_mb++;
                if (mb[i].Mbound == mblock_provider.max_bucket_size) wild_mb += 1;
                if (2 * mb[i].Mbound < mblock_provider.max_bucket_size) wild_mb += 1;
            }
        }
        if (wild_mb > 0.5 * num_mb) {
            fprintf(stderr, "[Warning] Pool_epi8_t<%u>::lsfsh_insert: strange bucket size, exp_bucket_size = %ld, real size = \n", 
                    nb, (long)(mblock_provider.max_bucket_size * 0.8));
            for (long i = 0; i < num_mb; i++) {
                if (mb[i].sh) printf("%ld ", mb[i].Mbound);
            }
            printf("\n");
        }
        
        pthread_spinlock_t min_lock;
        pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);
        double lift_time = 0.0;
        double search_time = 0.0;
        #pragma omp parallel for
        for (long thread = 0; thread < num_threads; thread++) {
            const long begin_ind = thread * num_mb / num_threads;
            const long end_ind = (thread + 1) * num_mb / num_threads;
            double _lift_time = 0.0;
            double _search_time = 0.0;
            for (long ind = begin_ind; ind < end_ind; ind++) {
                TIMER_START;
                __mblock_sh_search<shsize, LSH_L1_BLOCK, LSH_L2_BLOCK>(
                    mb[ind], threshold, ptr_buffer+thread, ptr_buffer_size+thread, ptr_buffer_num+thread, &profile_data,
                    target_index, dh_dim, b_full_fp, min_norm, min_vec, min_lock);
                TIMER_END;
                _search_time += CURRENT_TIME;
                TIMER_START;
                __lift_buffer(mb[ind], target_index, dh_dim, b_full_fp, 
                        ptr_buffer+thread, ptr_buffer_size+thread, ptr_buffer_num+thread, 
                        min_norm, min_vec, min_lock, &profile_data);
                TIMER_END;
                _lift_time += CURRENT_TIME;
            }
            pthread_spin_lock(&min_lock);
            if (_lift_time > lift_time) lift_time = _lift_time;
            if (_search_time > search_time) search_time = _search_time;
            pthread_spin_unlock(&min_lock);
        }
        profile_data.lift_time += lift_time;
        profile_data.dp_time += search_time;
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
            float old_norm = (float) (0.99995 * basis->get_B().hi[i + target_index]);
            if (target_length != 0.0 && min_norm[i] > 1.00004 * target_length * target_length) continue;
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
        return 0;
    }

    __attribute__ ((aligned (32))) float v_fp[256];
    copy_avx2(v_fp, min_vec[min_place - target_index], FD);
    FREE_MAT(min_vec);
    do {
        const long RD = index_l - min_place;
        VEC_QP v_QP = NEW_VEC_QP(basis->NumCols());
        MAT_QP b_QP = basis->get_b();

        do {
            for (long i = 0; i < RD+CSD; i++) {
                NTL::quad_float q(round(-v_fp[FD - i - 1] / b_full_fp[FD - i - 1][FD - i - 1]));
                red_avx2(v_fp, b_full_fp[FD-i-1], -q.hi, FD);
                red(v_QP.hi, v_QP.lo, b_QP.hi[index_r-i-1], b_QP.lo[index_r-i-1], q, basis->NumCols());
            }
            for (long i = min_place - 1; i >= 0; i--) {
                int32_t c = round(dot_avx2(v_QP.hi, basis->get_b_star().hi[i], basis->NumCols()) / basis->get_B().hi[i]);
                red(v_QP.hi, v_QP.lo, b_QP.hi[i], b_QP.lo[i], NTL::quad_float(c), basis->NumCols());
            }
        } while (0);

        if (target_index != 0 || (target_index == 0 && min_place == 0)) {
            fprintf(stderr, "\nlength = %f, vec = ", sqrt(dot_avx2(v_QP.hi, v_QP.hi, basis->NumCols())));
            do { 
                std::cerr << "["; 
                for (long __i = 0; __i < basis->NumCols()-1; __i++) { 
                    std::cerr << v_QP.hi[__i] << " "; 
                } 
                if (basis->NumCols() > 0) { 
                    std::cerr << v_QP.hi[basis->NumCols()-1] << "]\n"; 
                } else { 
                    std::cerr << "]\n"; 
                } 
            } while (0);
        }
        FREE_VEC_QP(v_QP);
    } while (0);
    FREE_MAT(b_full_fp);

    return 0;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::show_lsfsh_insert(long target_index, double eta, long log_level, double target_length, double stop_ratio, double qratio) {
    /////// basic information of the lifting ///////
    const double tail_gh = sqrt(gh2);
    const double dual_gh = basis->gh(index_l - LSH_DEFAULT_DHDIM, index_l);
    const double lift_gh = basis->gh(target_index, index_l);
    const double tail_exp_ratio = sqrt(cvec[3LL*(num_vec/2LL)+2LL] * 4.0) / _ratio / tail_gh;

    double tail_exp_alpha;
    do {
        double min_exp_length = lift_gh;
        for (double _alpha = 0.0; _alpha < 0.5; _alpha += 0.01) {
            uint64_t _num_lift = num_vec * (num_vec - 1.0) * pow(1 - _alpha * _alpha, CSD * 0.5);
            double _tail_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * _alpha);
            double _lift_length = lift_gh / pow(_num_lift, 1.0/(index_l - target_index));
            double _length = sqrt(_tail_length * _tail_length + _lift_length * _lift_length);
            if (_length < min_exp_length) {
                tail_exp_alpha = _alpha;
                min_exp_length = _length;
            }
        }
    } while (0);

    const uint64_t exp_num_lift = num_vec * (num_vec - 1.0) * pow(1 - tail_exp_alpha * tail_exp_alpha, CSD*0.5);
    const double tail_exp_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * tail_exp_alpha);
    const double lift_exp_length = lift_gh / pow(exp_num_lift, 1.0/(index_l - target_index));
    const double exp_length = sqrt(tail_exp_length * tail_exp_length + lift_exp_length * lift_exp_length);
    const bool unique_target = (target_length != 0.0) && (target_length < 0.96 * exp_length);

    // the probability that the solution with exp_length is filtered out
    double tail_alpha_prob_list[64];
    do {
        if (!unique_target) {
            for (long i = 0; i < 50; i++) {
                double _alpha = 0.01 * i;
                double _tail_length = tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * _alpha);
                double _num_lift = num_vec * (num_vec - 1.0) * pow(1 - _alpha * _alpha, CSD * 0.5);
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
                        target_length * sqrt(LSH_DEFAULT_DHDIM) / sqrt(index_r - target_index) :
                        lift_exp_length * sqrt(LSH_DEFAULT_DHDIM) / sqrt(index_l - target_index);

    if (log_level >= 1) {
        printf("input: tail_gh = %.2f, dual_gh = %.2f, lift_gh = %.2f\n", tail_gh, dual_gh, lift_gh);
        printf("param: tail_exp_alpha = %.2f, tail_exp_ratio = %.2f, tail_exp_length = %.2f\n", tail_exp_alpha, tail_exp_ratio, tail_exp_length);
        printf("param: exp_num_lift = %lu = (%.2f)^{-%ld}, lift_exp_length = %.2f, exp_length = %.2f, dual_exp_length = %.2f(%.2f, type %s)\n",
                exp_num_lift, pow(exp_num_lift, -1.0/(index_l - target_index)), index_l-target_index, lift_exp_length, exp_length, 
                dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
    } else if (log_level == 0) {
        if (target_length != 0.0) {
            printf("param: exp_length = %.2f, target_length = %.2f(%.3f), dual_exp_length = %.2f(%.2f, type %s), ",exp_length, target_length, 
                    target_length / exp_length, dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
        } else {
            printf("param: exp_length = %.2f, dual_exp_length = %.2f(%.2f, type %s), ", exp_length, 
                    dual_exp_length, dual_exp_length/dual_gh, unique_target ? "unique" : "normal");
        }
    }

    /////// choosing param ///////
    // output values
    float *dual_vec = (float *) NEW_VEC(LSH_DEFAULT_SHSIZE * 8 * _CEIL8(LSH_DEFAULT_DHDIM), sizeof(float));
    uint32_t *compress_pos = (uint32_t *) NEW_VEC(LSH_DEFAULT_SHSIZE * 8 * 6, sizeof(uint32_t));
    int32_t num_hbits, num_tbits, threshold;
    // input values
    Lattice_QP *b_mid = basis->b_loc_QP(index_l - LSH_DEFAULT_DHDIM, index_l);
    TIMER_START;
    if (log_level >= 2) printf("param: ");
    _opt_nsh_threshold(dual_vec, compress_pos, num_hbits, num_tbits, threshold, 
                        b_mid, LSH_DEFAULT_SHSIZE, exp_length, tail_alpha_prob_list, log_level);
    TIMER_END;
    delete b_mid;
    if (log_level >= 0) {
        if (log_level >= 1) printf("param: ");
        printf("opt_time = %.2fs, num_hbits = %d, num_tbits = %d, threshold = %d\n", CURRENT_TIME, num_hbits, num_tbits, threshold);
        fflush(stdout);
    }

    /////// call the lifting kernel ///////
    int ret = _show_lsfsh_insert<LSH_DEFAULT_SHSIZE, LSH_DEFAULT_DHDIM>
                (target_index, eta, log_level, dual_vec, compress_pos, num_hbits, threshold, dual_exp_length/dual_gh, unique_target ? target_length : 0.0, qratio);

    FREE_VEC(dual_vec);
    FREE_VEC((void *)compress_pos);
    return ret;
}


#if COMPILE_POOL_EPI8_96
template int Pool_epi8_t<3>::_lsfsh_insert<LSH_DEFAULT_SHSIZE, LSH_DEFAULT_DHDIM>(long target_index, double eta, long log_level, 
                float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_ratio, double target_length, double qratio);
template class Pool_epi8_t<3>;
#endif

#if COMPILE_POOL_EPI8_128
template int Pool_epi8_t<4>::_lsfsh_insert<LSH_DEFAULT_SHSIZE, LSH_DEFAULT_DHDIM>(long target_index, double eta, long log_level, 
                float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_ratio, double target_length, double qratio);
template class Pool_epi8_t<4>;
#endif

#if COMPILE_POOL_EPI8_160
template int Pool_epi8_t<5>::_lsfsh_insert<LSH_DEFAULT_SHSIZE, LSH_DEFAULT_DHDIM>(long target_index, double eta, long log_level, 
                float *dual_vec, uint32_t *compress_pos, int32_t num_hbits, int32_t threshold, double target_ratio, double target_length, double qratio);
template class Pool_epi8_t<5>;
#endif
