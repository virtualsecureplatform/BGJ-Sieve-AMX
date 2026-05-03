#include "../include/pool_epi8.h"
#include "../include/bgj_epi8.h"

#include <sys/time.h>
#include <algorithm>

#define NDH_DEFAULT_NDUAL 32
#define NDH_DOUBLE_NDUAL 64
#define NDH_DND_MAX_THRESHOLD 108000
#define NDH_DEFAULT_DHDIM 24
#if NDH_DEFAULT_NDUAL % 32
#error number of dual vector must divided by 32
#endif
#if NDH_DEFAULT_DHDIM % 8
#error dual hash dimension must divided by 8
#endif
#define NDH_L1_BLOCK 256
#define NDH_L2_BLOCK 8192
#define NDH_M_BLOCK 65536
#define NDH_DIAG_FIRST 1

#if 1
struct timeval _ndh_timer_start[MAX_NTHREADS], _ndh_timer_end[MAX_NTHREADS];
double _ndh_time_curr[MAX_NTHREADS];

#define TIMER_START do {                                                        \
        gettimeofday(&_ndh_timer_start[omp_get_thread_num()], NULL);            \
    } while (0)

#define TIMER_END do {                                                          \
        gettimeofday(&_ndh_timer_end[omp_get_thread_num()], NULL);              \
        _ndh_time_curr[omp_get_thread_num()] =                                                            \
            (_ndh_timer_end[omp_get_thread_num()].tv_sec-_ndh_timer_start[omp_get_thread_num()].tv_sec)+                    \
            (double)(_ndh_timer_end[omp_get_thread_num()].tv_usec-_ndh_timer_start[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_ndh_time_curr[omp_get_thread_num()])
#endif

// only called in gen_dual_vec_list
static float _compute_detn(float **D, float **Dt, long nlist, long d) {
    for (long i = 0; i < nlist; i++) {
        for (long j = 0; j < d; j++) Dt[j][i] = D[i][j];
    }

    float *dst_vec = (float *) NEW_VEC(d, sizeof(float));
    for (long i = 0; i < d; i++) {
        dst_vec[i] = dot_avx2(Dt[i], Dt[i], nlist);
        if (dst_vec[i] == 0.0f || ((i >= 1) && dst_vec[i] < 1e-6 * dst_vec[i-1])) {
            FREE_VEC(dst_vec);
            return 0.0f;
        }
        float r = 1.0 / dst_vec[i];
        for (long j = i + 1; j < d; j++) {
            float x = dot_avx2(Dt[i], Dt[j], nlist);
            red_avx2(Dt[j], Dt[i], x * r, nlist);
        }
    }

    float log_ret = 0.0;
    for (long i = 0; i < d; i++) log_ret += log(dst_vec[i]);
    log_ret /= d;
    FREE_VEC(dst_vec);
    return exp(log_ret);
}

int gen_dual_vec_list(float *dst, Lattice_QP *L, long log_level, long nlist) {
    // magic numbers
    const long expect_num_short_vec = 1000;
    const long max_iter = 1000;

    // profiling data
    struct timeval start, end;
    double enum_time, opt_time;
    
    // find short vectors by enumeration
    if (log_level >= 1) gettimeofday(&start, NULL);
    Lattice_QP *Ld = L->dual_QP();
    const long d = Ld->NumRows();
    const float R = pow(1.6 * expect_num_short_vec, 1.0 / L->NumRows()) * Ld->gh();
    float *short_vec_store = (float *) NEW_VEC(L->NumCols() * expect_num_short_vec, sizeof(float));
    float **short_vec_list = (float **) calloc(expect_num_short_vec, sizeof(float *));
    float *norm_list = (float *) malloc(expect_num_short_vec * sizeof(float));
    long num_short_vec = 0;
    do {
        const float R2 = R * R;
        float **b_fp = (float **) NEW_MAT(d, d, sizeof(float));
        float **miu_loc = (float **) NEW_MAT(d, d+1, sizeof(float));
        float *B_loc = (float *) NEW_VEC(d+1, sizeof(float));
        for (long i = 0; i < d; i++) {
            for (long j = 0; j < d-i-1; j++) miu_loc[i][j] = Ld->get_miu().hi[d-1-j][i];
            for (long j = i; j < d; j++) b_fp[i][j] = Ld->get_b().hi[i][j];
        }
        for (long i = 0; i < d; i++) B_loc[i] = Ld->get_B().hi[i];

        __attribute__ ((aligned (32))) float u[256] = {};
        __attribute__ ((aligned (32))) float l[256] = {};
        __attribute__ ((aligned (32))) float c[256] = {};
        long t = 0;
        long t_max = 0;

        while (1) {
            l[t] = l[t+1] + (u[d-1-t] + c[t]) * (u[d-1-t] + c[t]) * B_loc[t];
            if (l[t] < R2){
                if (t > 0){
                    t--;
                    c[t] = dot_avx2(u, miu_loc[t], d-t-1);
                    u[d-1-t] = round(-c[t]);
                }else{
                    if (l[0] != 0.0) {
                        short_vec_list[num_short_vec] = short_vec_store + num_short_vec * d;
                        for (long i = 0; i < d; i++) {
                            if (u[d-1-i] != 0.0) red_avx2(short_vec_list[num_short_vec], b_fp[i], -u[d-1-i], d);
                        }
                        norm_list[num_short_vec] = l[0];
                        num_short_vec++;
                        if (num_short_vec >= expect_num_short_vec) break;
                    }
                    
                    if (t >= t_max){
                        if (t > t_max) {
                            t_max = t;
                            if (t >= d) break;
                        }
                        u[d-1-t] += 1.0;
                    } else {
                        float c_tmp = round(-c[t]);
                        if (-c[t] >= c_tmp) {
                            if (u[d-1-t] <= c_tmp) {
                                u[d-1-t] = 2 * c_tmp - u[d-1-t] + 1;
                            } else {
                                u[d-1-t] = 2 * c_tmp - u[d-1-t];
                            }
                        } else {
                            if (u[d-1-t] < c_tmp) {
                                u[d-1-t] = 2 * c_tmp - u[d-1-t];
                            } else {
                                u[d-1-t] = 2 * c_tmp - u[d-1-t] - 1;
                            }
                        }
                    }
                }
            } else {
                t++;
                if (t >= t_max){
                    if (t > t_max) {
                        t_max = t;
                        if (t >= d) break;
                    }
                    u[d-1-t] += 1.0;
                } else {
                    float c_tmp = round(-c[t]);
                    if (-c[t] >= c_tmp){
                        if (u[d-1-t] <= c_tmp){
                            u[d-1-t] = 2 * c_tmp - u[d-1-t] + 1;
                        }else{
                            u[d-1-t] = 2 * c_tmp - u[d-1-t];
                        }
                    }else{
                        if (u[d-1-t] < c_tmp){
                            u[d-1-t] = 2 * c_tmp - u[d-1-t];
                        }else{
                            u[d-1-t] = 2 * c_tmp - u[d-1-t] - 1;
                        }
                    }
                }
            }
        }
        
        for (long i = 0; i < num_short_vec; i++) {
            for (long j = i; j > 0; j--) {
                if (norm_list[j] < norm_list[j-1]) {
                    float tmp = norm_list[j];
                    norm_list[j] = norm_list[j-1];
                    norm_list[j-1] = tmp;
                    float *tmpp = short_vec_list[j];
                    short_vec_list[j] = short_vec_list[j-1];
                    short_vec_list[j-1] = tmpp;
                } else break;
            }
        }
        
        FREE_MAT(miu_loc);
        FREE_MAT(b_fp);
        FREE_VEC(B_loc);
    } while (0);
    if (log_level >= 1) {
        gettimeofday(&end, NULL);
        enum_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    }


    // try to find good dual vector list
    if (log_level >= 1) gettimeofday(&start, NULL);
    long select_ind[nlist];
    float **dp_table = (float **) NEW_MAT(nlist, nlist, sizeof(float));
    float *dp_sum = (float *) NEW_VEC(nlist, sizeof(float));
    for (long i = 0; i < nlist; i++) {
        select_ind[i] = i;
        if (short_vec_list[i][i] == 0.0f) {
            for (long j = nlist; j < num_short_vec; j++) {
                if (short_vec_list[j][i] != 0.0f) {
                    int dup = 0;
                    for (long k = 0; k < i; k++) if (j == select_ind[k]) dup = 1;
                    if (!dup) {
                        select_ind[i] = j;
                        break;
                    }
                }
            }
        }
        if (short_vec_list[select_ind[i]][i] == 0.0f) {
            fprintf(stderr, "[Error] gen_dual_vec_list: dual vectors always in a subspace? aborted.\n");
        }
    }
    float **D = (float **) NEW_MAT(nlist, d, sizeof(float));
    float **Dt = (float **) NEW_MAT(d, nlist, sizeof(float));
    for (long i = 0; i < nlist; i++) copy_avx2(D[i], short_vec_list[select_ind[i]], d);
    for (long i = 0; i < nlist; i++) {
        for (long j = 0; j <= i; j++) {
            dp_table[i][j] = dot_avx2(D[i], D[j], d);
            dp_table[j][i] = dp_table[i][j];
        }
    }
    for (long i = 0; i < nlist; i++) {
        for (long j = 0; j < nlist; j++) dp_sum[i] += dp_table[i][j] * dp_table[i][j];
    }
    
    long iter = 0;
    double sum_dp = 0.0;
    double sum_norm = 0.0;
    double cond = 0.0;
    for (long i = 0; i < nlist; i++) cond += dp_table[i][i];
    cond /= _compute_detn(D, Dt, nlist, d);
    for (long i = 0; i < nlist; i++) sum_dp += dp_sum[i];
    for (long i = 0; i < nlist; i++) sum_norm += dp_table[i][i];
    if (log_level >= 3) printf("opt start: sum_dp = %e, sum_norm = %e, cond = %f\n", sum_dp, sum_norm, cond);
    
    while (iter < max_iter) {
        long dst_ind;
        long count = 0;
        do {
            dst_ind = Uniform_long(nlist);
            count++;
        } while (dp_sum[dst_ind] * nlist < sum_dp && count < 3);
        
        long src = 0;
        long nrem = 5;
        float new_dp[1024];
        float new_sum;
        float new_norm;
        while (nrem --> 0) {
            long pass = 1;
            src += Uniform_long(163) + 1;
            if (src >= num_short_vec) src -= num_short_vec;
            
            for (long i = 0; i < nlist; i++) if (select_ind[i] == src) pass = 0;
            if (!pass) continue;

            for (long i = 0; i < nlist; i++) {
                if (i != dst_ind) {
                    new_dp[i] = dot_avx2(short_vec_list[src], D[i], d);
                } else {
                    new_dp[i] = dot_avx2(short_vec_list[src], short_vec_list[src], d);
                }
            }
            new_sum = sum_dp;
            new_norm = sum_norm + new_dp[dst_ind] - dp_table[dst_ind][dst_ind];
            for (long i = 0; i < nlist; i++) {
                if (i == dst_ind) {
                    new_sum += new_dp[i] * new_dp[i] - dp_table[dst_ind][i] * dp_table[dst_ind][i];
                } else {
                    new_sum += 2 * (new_dp[i] * new_dp[i] - dp_table[dst_ind][i] * dp_table[dst_ind][i]);
                }
            }
            if (new_sum > sum_dp) pass = 0;

            if (pass) break;
        }
        iter++;

        long old_dst = select_ind[dst_ind];
        copy_avx2(D[dst_ind], short_vec_list[src], d);
        float new_cond = 0.0;
        for (long i = 0; i < nlist; i++) new_cond += dp_table[i][i];
        new_cond /= _compute_detn(D, Dt, nlist, d);
        if (new_cond < cond) {
            if (log_level >= 3) printf("iter %ld: dst_ind = %ld, src = %ld, sum_dp = %e, sum_norm = %e, cond = %f\n", iter - 1, dst_ind, src, new_sum, new_norm, new_cond);
            select_ind[dst_ind] = src;
            for (long i = 0; i < nlist; i++) {
                dp_sum[i] += new_dp[i] * new_dp[i] - dp_table[dst_ind][i] * dp_table[dst_ind][i];
            }
            dp_sum[dst_ind] = 0.0;
            for (long i = 0; i < nlist; i++) dp_sum[dst_ind] += new_dp[i] * new_dp[i];
            sum_dp = new_sum;
            cond = new_cond;
            for (long i = 0; i < nlist; i++) {
                dp_table[dst_ind][i] = new_dp[i];
                dp_table[i][dst_ind] = new_dp[i];
            }
        } else {
            if (log_level >= 4) printf("iter %ld: dst_ind = %ld, src = %ld, sum_dp = %e, cond = %f failed\n", iter - 1, dst_ind, src, new_sum, new_cond);
            copy_avx2(D[dst_ind], short_vec_list[old_dst], d);
        }
    }
    if (log_level >= 1) {
        gettimeofday(&end, NULL);
        opt_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
        printf("enum_time = %f, opt_time = %f, sum_dp = %e, sum_norm = %e, cond = %f * %ld\n", enum_time, opt_time, sum_dp, sum_norm, cond/d, d);
    }

    for (long i = 0; i < nlist; i++) {
        for (long j = 0; j < d; j++) dst[i*_CEIL8(d)+j] = 256.0 * D[i][j];
    }

    // free
    FREE_MAT(D);
    FREE_MAT(Dt);
    FREE_MAT(dp_table);
    FREE_VEC(dp_sum);
    free(norm_list);
    free(short_vec_list);
    FREE_VEC(short_vec_store);
    delete Ld;
    return 1;
}

int estimate_ndh_threshold(double len, float *dual_vec, int32_t dh_dim, int32_t ndual, double ratio) {
        static const float ndual32_overhead80140[64] = {
        0.013528, 0.015064, 0.018963, 0.021186, 0.025078, 0.028979, 0.034073, 0.040543, 
        0.046239, 0.054058, 0.066732, 0.072430, 0.085513, 0.097885, 0.113774, 0.140698, 
        0.147210, 0.174179, 0.194659, 0.224771, 0.256095, 0.292986, 0.331840, 0.374213, 
        0.417846, 0.472221, 0.510758, 0.581097, 0.646230, 0.724852, 0.784510, 0.874722, 
        0.972286, 1.045519, 1.164168, 1.260275, 1.389797, 1.478176, 1.634118, 1.785338, 
        1.911837, 2.058363, 2.225068, 2.406083, 2.569760, 2.726399, 2.922378, 3.095170, 
        3.289611, 3.473769, 3.718463, 3.840768, 4.063767, 4.235193, 4.487985, 4.661380, 
        4.910322, 5.043017, 5.298989, 5.460839, 5.689329
    };

    static const float ndual64_overhead180259[80] = {
        0.000190, 0.000218, 0.000266, 0.000276, 0.000336, 0.000341, 0.000376, 0.000439, 
        0.000517, 0.000599, 0.000652, 0.000730, 0.001086, 0.000953, 0.001111, 0.001279, 
        0.001474, 0.001545, 0.001985, 0.001933, 0.002078, 0.002373, 0.002909, 0.003149, 
        0.003334, 0.003782, 0.004160, 0.004583, 0.005391, 0.006105, 0.006585, 0.006999, 
        0.008194, 0.009584, 0.009708, 0.011195, 0.013009, 0.014017, 0.015829, 0.016450, 
        0.017733, 0.019799, 0.022014, 0.024924, 0.028826, 0.030009, 0.035585, 0.036210, 
        0.041620, 0.045106, 0.048645, 0.053988, 0.058054, 0.068668, 0.070919, 0.075734, 
        0.086182, 0.092791, 0.100726, 0.110717, 0.122000, 0.129198, 0.142617, 0.161505, 
        0.167860, 0.186555, 0.207070, 0.220216, 0.237436, 0.252718, 0.303881, 0.292986, 
        0.318739, 0.349796, 0.374722, 0.404118, 0.422350, 0.464339, 0.497339, 0.535689
    };
    const long N = 10000;
    const double base = 3;
    DGS1d R;
    
    int32_t *res = (int32_t *) NEW_VEC(N, sizeof(int32_t));
    __attribute__ ((aligned (32))) float tmp[256];
    for (long ind = 0; ind < N; ind++) {
        for (long l = 0; l < dh_dim; l++) tmp[l] = R.discrete_gaussian(0.0, 1048576.0);
        float x = dot_avx2(tmp, tmp, dh_dim);
        x = len / sqrt(x);
        mul_avx2(tmp, x, dh_dim);
        res[ind] = 0;
        for (long i = 0; i < ndual; i++) {
            float x = dot_avx2(tmp, dual_vec + i * dh_dim, dh_dim);
            int32_t xi = x;
            int8_t x8 = xi;
            res[ind] += (int)x8 * (int)x8;
        }
    }
    std::sort(res, res+N);
    
    double best_score = 0.0;
    int32_t ret = 0;
    if (ndual == 32) {
        for (double _ratio = ratio; _ratio < 0.998; _ratio += 0.003) {
            int th = 1000 * round(res[(int)(_ratio * N)] * 0.001);
            th = (th < 80000) ? 80000 : th;
            th = (th > 140000) ? 140000 : th;
            double score = _ratio / (base + ndual32_overhead80140[th/1000-80]);
            if (score > best_score) {
                best_score = score;
                ret = th;
            }
        }
    } else {
        int min_th = 1000 * round(res[(int)(ratio * N)] * 0.001);
        if (min_th > 259000) ret = min_th;
        for (double _ratio = ratio; _ratio < 0.998; _ratio += 0.003) {
            int th = 1000 * round(res[(int)(_ratio * N)] * 0.001);
            th = (th < 180000) ? 180000 : th;
            if (th > 259000) continue;
            double score = _ratio / (base + ndual64_overhead180259[th/1000-180]);
            if (score > best_score) {
                best_score = score;
                ret = th;
            }
        }
    }

    FREE_VEC((void *)res);
    return ret;
}

template <uint32_t nb>
template <uint32_t ndual, uint32_t dh_dim>
void Pool_epi8_t<nb>::_compute_ndh_mblock(float *fvec, int8_t *dh, float **b_ext, float *dual_vec, long MInd, long num, long target_index) {
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
    
    const long FD = index_r - target_index;
    const long LD = index_l - target_index;
    const long FD8 = ( (FD + 7) / 8 ) * 8;
    const long ID = index_l - dh_dim - target_index;
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        const long begin_ind = (num * thread) / num_threads;
        const long end_ind = (num * (thread+1)) / num_threads;
        long ind = begin_ind;
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        __attribute__ ((aligned (32))) float ftmp[8 * dh_dim];
        while (ind < end_ind - 7) {
            _compute_coeff_b8(ctmp, MInd + ind);
            set_zero_avx2(fvec + FD8 * ind, FD8 * 8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(fvec + FD8 * (ind+0), b_ext[i], -ctmp[i*8+0], LD + i + 1);
                red_avx2(fvec + FD8 * (ind+1), b_ext[i], -ctmp[i*8+1], LD + i + 1);
                red_avx2(fvec + FD8 * (ind+2), b_ext[i], -ctmp[i*8+2], LD + i + 1);
                red_avx2(fvec + FD8 * (ind+3), b_ext[i], -ctmp[i*8+3], LD + i + 1);
                red_avx2(fvec + FD8 * (ind+4), b_ext[i], -ctmp[i*8+4], LD + i + 1);
                red_avx2(fvec + FD8 * (ind+5), b_ext[i], -ctmp[i*8+5], LD + i + 1);
                red_avx2(fvec + FD8 * (ind+6), b_ext[i], -ctmp[i*8+6], LD + i + 1);
                red_avx2(fvec + FD8 * (ind+7), b_ext[i], -ctmp[i*8+7], LD + i + 1);
            }
            for (long l = 0; l < dh_dim; l += 8) {
                for (long i = 0; i < 8; i++) {
                    _mm256_store_ps(ftmp + i * dh_dim + l, _mm256_loadu_ps(fvec + (ind+i) * FD8 + ID + l));
                }
            }
            for (long k = 0; k < ndual; k += 16) {
                for (long i = 0; i < 8; i += 2) {
                    __m256 dst00, dst01, dst10, dst11;
                    __m128i dsti00, dsti01, dsti10, dsti11;
                    AVX2_DP_2X8((ftmp + (i+0) * dh_dim), (ftmp + (i+1) * dh_dim), (dual_vec + ((k+0) * dh_dim)), dh_dim, dst00, dst10);
                    AVX2_DP_2X8((ftmp + (i+0) * dh_dim), (ftmp + (i+1) * dh_dim), (dual_vec + ((k+8) * dh_dim)), dh_dim, dst01, dst11);
                    dsti00 = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(dst00));
                    dsti01 = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(dst01));
                    dsti10 = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(dst10));
                    dsti11 = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(dst11));
                    _mm_store_si128((__m128i *)(dh + (ind+i+0) * ndual + k), _mm_or_si128(dsti00, (__m128i)_mm_permute_pd((__m128d)dsti01, 0x1)));
                    _mm_store_si128((__m128i *)(dh + (ind+i+1) * ndual + k), _mm_or_si128(dsti10, (__m128i)_mm_permute_pd((__m128d)dsti11, 0x1)));
                }
            }
            ind += 8;
        }
        while (ind < end_ind) {
            _compute_coeff(ctmp, MInd + ind);
            for (long i = 0; i < CSD; i++) {
                red_avx2(fvec + FD8 * ind, b_ext[i], -ctmp[i], LD + i + 1);
            }
            for (long l = 0; l < dh_dim; l += 8) {
                _mm256_store_ps(ftmp + l, _mm256_loadu_ps(fvec + ind * FD8 + ID + l));
            }
            for (long k = 0; k < ndual; k += 16) {
                __m256 dst0, dst1;
                __m128i dsti0, dsti1;
                AVX2_DP_1X8(ftmp, (dual_vec + ((k + 0) * dh_dim)), dh_dim, dst0);
                AVX2_DP_1X8(ftmp, (dual_vec + ((k + 8) * dh_dim)), dh_dim, dst1);
                dsti0 = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(dst0));
                dsti1 = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(dst1));
                _mm_store_si128((__m128i *)(dh + ind * ndual + k), _mm_or_si128(dsti0, (__m128i)_mm_permute_pd((__m128d)dsti1, 0x1)));
            }
            ind++;
        }
    }
}

#pragma region
#define ADD_TO_BUFFER(__ind) do {                                                                   \
    if (*ptr_buffer_num == *ptr_buffer_size) {                                                      \
        *ptr_buffer_size *= 2;                                                                      \
        *ptr_buffer_size += 64;                                                                     \
        *ptr_buffer = (uint32_t *) realloc(*ptr_buffer, 3 * (*ptr_buffer_size) * sizeof(uint32_t)); \
    }                                                                                               \
    (*ptr_buffer)[(*ptr_buffer_num) * 3] = 2;                                                       \
    (*ptr_buffer)[(*ptr_buffer_num) * 3 + 1] = __ind;                                               \
    (*ptr_buffer_num)++;                                                                              \
} while (0);
#define ADDA_TO_BUFFER(__ind, __jnd) do {                                                           \
    if (*ptr_buffer_num == *ptr_buffer_size) {                                                      \
        *ptr_buffer_size *= 2;                                                                      \
        *ptr_buffer_size += 64;                                                                     \
        *ptr_buffer = (uint32_t *) realloc(*ptr_buffer, 3 * (*ptr_buffer_size) * sizeof(uint32_t)); \
    }                                                                                               \
    (*ptr_buffer)[(*ptr_buffer_num) * 3] = 0;                                                       \
    (*ptr_buffer)[(*ptr_buffer_num) * 3 + 1] = __ind;                                               \
    (*ptr_buffer)[(*ptr_buffer_num) * 3 + 2] = __jnd;                                               \
    (*ptr_buffer_num)++;                                                                              \
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
    (*ptr_buffer_num)++;                                                                              \
} while (0);
#define CHECK_AND_ADD8(__cmp, __bias) do {                          \
    while (__cmp) {                                                 \
        int __r = __builtin_ctzl(__cmp);                            \
        __cmp -= (1UL << __r);                                      \
        ADD_TO_BUFFER(__bias + __r);                                \
    }                                                               \
} while (0)
#define CHECK_AND_ADD8x8(__cmp, __ibias, __jbias, __add_func) do {  \
    while (__cmp) {                                                 \
        int __r = __builtin_ctzl(__cmp);                            \
        __cmp -= (1UL << __r);                                      \
        __add_func(__ibias + (__r >> 3), __jbias + (__r & 0x7));    \
    }                                                               \
} while (0)
#pragma endregion

#pragma region
///////////////////////////////////// npnorm check /////////////////////////////////////
template <uint32_t nb>
template <uint32_t ndual>
inline void Pool_epi8_t<nb>::_vnpnorm_check8x8(uint64_t *cmpp, uint64_t *cmpn, int8_t *s1, int8_t *s2, __m256i th) {
    __m256i accpn[8];
    __m256i accnn[8];
    for (long j = 0; j < 8; j++) {
        __m256i accpn0 = _mm256_setzero_si256();
        __m256i accpn1 = _mm256_setzero_si256();
        __m256i accpn2 = _mm256_setzero_si256();
        __m256i accpn3 = _mm256_setzero_si256();
        __m256i accpn4 = _mm256_setzero_si256();
        __m256i accpn5 = _mm256_setzero_si256();
        __m256i accpn6 = _mm256_setzero_si256();
        __m256i accpn7 = _mm256_setzero_si256();
        
        __m256i accnn0 = _mm256_setzero_si256();
        __m256i accnn1 = _mm256_setzero_si256();
        __m256i accnn2 = _mm256_setzero_si256();
        __m256i accnn3 = _mm256_setzero_si256();
        __m256i accnn4 = _mm256_setzero_si256();
        __m256i accnn5 = _mm256_setzero_si256();
        __m256i accnn6 = _mm256_setzero_si256();
        __m256i accnn7 = _mm256_setzero_si256();

        __m256i accps0 = _mm256_setzero_si256();
        __m256i accps1 = _mm256_setzero_si256();
        __m256i accps2 = _mm256_setzero_si256();
        __m256i accps3 = _mm256_setzero_si256();
        __m256i accps4 = _mm256_setzero_si256();
        __m256i accps5 = _mm256_setzero_si256();
        __m256i accps6 = _mm256_setzero_si256();
        __m256i accps7 = _mm256_setzero_si256();
        
        __m256i accns0 = _mm256_setzero_si256();
        __m256i accns1 = _mm256_setzero_si256();
        __m256i accns2 = _mm256_setzero_si256();
        __m256i accns3 = _mm256_setzero_si256();
        __m256i accns4 = _mm256_setzero_si256();
        __m256i accns5 = _mm256_setzero_si256();
        __m256i accns6 = _mm256_setzero_si256();
        __m256i accns7 = _mm256_setzero_si256();

        for (long l = 0; l < ndual; l += 32) {
            __m256i ya0 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 0 * ndual + l)));
            __m256i ya1 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 1 * ndual + l)));
            __m256i ya2 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 2 * ndual + l)));
            __m256i ya3 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 3 * ndual + l)));
            __m256i ya4 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 4 * ndual + l)));
            __m256i ya5 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 5 * ndual + l)));
            __m256i ya6 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 6 * ndual + l)));
            __m256i ya7 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 7 * ndual + l)));

            __m256i ys0 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 0 * ndual + l)));
            __m256i ys1 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 1 * ndual + l)));
            __m256i ys2 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 2 * ndual + l)));
            __m256i ys3 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 3 * ndual + l)));
            __m256i ys4 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 4 * ndual + l)));
            __m256i ys5 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 5 * ndual + l)));
            __m256i ys6 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 6 * ndual + l)));
            __m256i ys7 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 7 * ndual + l)));
        
            accpn0 = _mm256_dpbusd_epi32(accpn0, _mm256_xor_si256(ya0, epi8_sign_bit), ya0);
            accpn1 = _mm256_dpbusd_epi32(accpn1, _mm256_xor_si256(ya1, epi8_sign_bit), ya1);
            accpn2 = _mm256_dpbusd_epi32(accpn2, _mm256_xor_si256(ya2, epi8_sign_bit), ya2);
            accpn3 = _mm256_dpbusd_epi32(accpn3, _mm256_xor_si256(ya3, epi8_sign_bit), ya3);
            accpn4 = _mm256_dpbusd_epi32(accpn4, _mm256_xor_si256(ya4, epi8_sign_bit), ya4);
            accpn5 = _mm256_dpbusd_epi32(accpn5, _mm256_xor_si256(ya5, epi8_sign_bit), ya5);
            accpn6 = _mm256_dpbusd_epi32(accpn6, _mm256_xor_si256(ya6, epi8_sign_bit), ya6);
            accpn7 = _mm256_dpbusd_epi32(accpn7, _mm256_xor_si256(ya7, epi8_sign_bit), ya7);

            accps0 = _mm256_dpbusd_epi32(accps0, epi8_sign_bit, ya0);
            accps1 = _mm256_dpbusd_epi32(accps1, epi8_sign_bit, ya1);
            accps2 = _mm256_dpbusd_epi32(accps2, epi8_sign_bit, ya2);
            accps3 = _mm256_dpbusd_epi32(accps3, epi8_sign_bit, ya3);
            accps4 = _mm256_dpbusd_epi32(accps4, epi8_sign_bit, ya4);
            accps5 = _mm256_dpbusd_epi32(accps5, epi8_sign_bit, ya5);
            accps6 = _mm256_dpbusd_epi32(accps6, epi8_sign_bit, ya6);
            accps7 = _mm256_dpbusd_epi32(accps7, epi8_sign_bit, ya7);

            accnn0 = _mm256_dpbusd_epi32(accnn0, _mm256_xor_si256(ys0, epi8_sign_bit), ys0);
            accnn1 = _mm256_dpbusd_epi32(accnn1, _mm256_xor_si256(ys1, epi8_sign_bit), ys1);
            accnn2 = _mm256_dpbusd_epi32(accnn2, _mm256_xor_si256(ys2, epi8_sign_bit), ys2);
            accnn3 = _mm256_dpbusd_epi32(accnn3, _mm256_xor_si256(ys3, epi8_sign_bit), ys3);
            accnn4 = _mm256_dpbusd_epi32(accnn4, _mm256_xor_si256(ys4, epi8_sign_bit), ys4);
            accnn5 = _mm256_dpbusd_epi32(accnn5, _mm256_xor_si256(ys5, epi8_sign_bit), ys5);
            accnn6 = _mm256_dpbusd_epi32(accnn6, _mm256_xor_si256(ys6, epi8_sign_bit), ys6);
            accnn7 = _mm256_dpbusd_epi32(accnn7, _mm256_xor_si256(ys7, epi8_sign_bit), ys7);

            accns0 = _mm256_dpbusd_epi32(accns0, epi8_sign_bit, ys0);
            accns1 = _mm256_dpbusd_epi32(accns1, epi8_sign_bit, ys1);
            accns2 = _mm256_dpbusd_epi32(accns2, epi8_sign_bit, ys2);
            accns3 = _mm256_dpbusd_epi32(accns3, epi8_sign_bit, ys3);
            accns4 = _mm256_dpbusd_epi32(accns4, epi8_sign_bit, ys4);
            accns5 = _mm256_dpbusd_epi32(accns5, epi8_sign_bit, ys5);
            accns6 = _mm256_dpbusd_epi32(accns6, epi8_sign_bit, ys6);
            accns7 = _mm256_dpbusd_epi32(accns7, epi8_sign_bit, ys7);
        }
    
        accpn0 = _mm256_sub_epi32(accpn0, accps0);
        accpn1 = _mm256_sub_epi32(accpn1, accps1);
        accpn2 = _mm256_sub_epi32(accpn2, accps2);
        accpn3 = _mm256_sub_epi32(accpn3, accps3);
        accpn4 = _mm256_sub_epi32(accpn4, accps4);
        accpn5 = _mm256_sub_epi32(accpn5, accps5);
        accpn6 = _mm256_sub_epi32(accpn6, accps6);
        accpn7 = _mm256_sub_epi32(accpn7, accps7);

        accnn0 = _mm256_sub_epi32(accnn0, accns0);
        accnn1 = _mm256_sub_epi32(accnn1, accns1);
        accnn2 = _mm256_sub_epi32(accnn2, accns2);
        accnn3 = _mm256_sub_epi32(accnn3, accns3);
        accnn4 = _mm256_sub_epi32(accnn4, accns4);
        accnn5 = _mm256_sub_epi32(accnn5, accns5);
        accnn6 = _mm256_sub_epi32(accnn6, accns6);
        accnn7 = _mm256_sub_epi32(accnn7, accns7);

        accpn0 = _mm256_hadd_epi32(accpn0, accpn1);
        accpn2 = _mm256_hadd_epi32(accpn2, accpn3);
        accpn4 = _mm256_hadd_epi32(accpn4, accpn5);
        accpn6 = _mm256_hadd_epi32(accpn6, accpn7);
        accpn0 = _mm256_hadd_epi32(accpn0, accpn2);
        accpn4 = _mm256_hadd_epi32(accpn4, accpn6);

        accnn0 = _mm256_hadd_epi32(accnn0, accnn1);
        accnn2 = _mm256_hadd_epi32(accnn2, accnn3);
        accnn4 = _mm256_hadd_epi32(accnn4, accnn5);
        accnn6 = _mm256_hadd_epi32(accnn6, accnn7);
        accnn0 = _mm256_hadd_epi32(accnn0, accnn2);
        accnn4 = _mm256_hadd_epi32(accnn4, accnn6);

        __m256i accpnlo = _mm256_permute2f128_si256(accpn0, accpn4, 48);
        __m256i accpnhi = _mm256_permute2f128_si256(accpn0, accpn4, 33);
        __m256i accnnlo = _mm256_permute2f128_si256(accnn0, accnn4, 48);
        __m256i accnnhi = _mm256_permute2f128_si256(accnn0, accnn4, 33);

        accpn[j] = _mm256_add_epi32(accpnlo, accpnhi);
        accnn[j] = _mm256_add_epi32(accnnlo, accnnhi);
    }
    
    uint8_t *cmpp_epi8 = (uint8_t *) cmpp;
    uint8_t *cmpn_epi8 = (uint8_t *) cmpn;
    cmpp_epi8[0] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[0]));
    cmpp_epi8[1] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[1]));
    cmpp_epi8[2] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[2]));
    cmpp_epi8[3] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[3]));
    cmpp_epi8[4] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[4]));
    cmpp_epi8[5] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[5]));
    cmpp_epi8[6] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[6]));
    cmpp_epi8[7] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn[7]));

    cmpn_epi8[0] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[0]));
    cmpn_epi8[1] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[1]));
    cmpn_epi8[2] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[2]));
    cmpn_epi8[3] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[3]));
    cmpn_epi8[4] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[4]));
    cmpn_epi8[5] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[5]));
    cmpn_epi8[6] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[6]));
    cmpn_epi8[7] = _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn[7]));
}
template <uint32_t nb>
template <uint32_t ndual>
inline void Pool_epi8_t<nb>::_vnpnorm_check8xn(uint64_t *cmpp, uint64_t *cmpn, int8_t *s1, int8_t *s2, long n, __m256i th) {
    for (long j = 0; j < n; j++) {
        __m256i accpn0 = _mm256_setzero_si256();
        __m256i accpn1 = _mm256_setzero_si256();
        __m256i accpn2 = _mm256_setzero_si256();
        __m256i accpn3 = _mm256_setzero_si256();
        __m256i accpn4 = _mm256_setzero_si256();
        __m256i accpn5 = _mm256_setzero_si256();
        __m256i accpn6 = _mm256_setzero_si256();
        __m256i accpn7 = _mm256_setzero_si256();
        
        __m256i accnn0 = _mm256_setzero_si256();
        __m256i accnn1 = _mm256_setzero_si256();
        __m256i accnn2 = _mm256_setzero_si256();
        __m256i accnn3 = _mm256_setzero_si256();
        __m256i accnn4 = _mm256_setzero_si256();
        __m256i accnn5 = _mm256_setzero_si256();
        __m256i accnn6 = _mm256_setzero_si256();
        __m256i accnn7 = _mm256_setzero_si256();

        __m256i accps0 = _mm256_setzero_si256();
        __m256i accps1 = _mm256_setzero_si256();
        __m256i accps2 = _mm256_setzero_si256();
        __m256i accps3 = _mm256_setzero_si256();
        __m256i accps4 = _mm256_setzero_si256();
        __m256i accps5 = _mm256_setzero_si256();
        __m256i accps6 = _mm256_setzero_si256();
        __m256i accps7 = _mm256_setzero_si256();
        
        __m256i accns0 = _mm256_setzero_si256();
        __m256i accns1 = _mm256_setzero_si256();
        __m256i accns2 = _mm256_setzero_si256();
        __m256i accns3 = _mm256_setzero_si256();
        __m256i accns4 = _mm256_setzero_si256();
        __m256i accns5 = _mm256_setzero_si256();
        __m256i accns6 = _mm256_setzero_si256();
        __m256i accns7 = _mm256_setzero_si256();

        for (long l = 0; l < ndual; l += 32) {
            __m256i ya0 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 0 * ndual + l)));
            __m256i ya1 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 1 * ndual + l)));
            __m256i ya2 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 2 * ndual + l)));
            __m256i ya3 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 3 * ndual + l)));
            __m256i ya4 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 4 * ndual + l)));
            __m256i ya5 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 5 * ndual + l)));
            __m256i ya6 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 6 * ndual + l)));
            __m256i ya7 = _mm256_add_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 7 * ndual + l)));

            __m256i ys0 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 0 * ndual + l)));
            __m256i ys1 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 1 * ndual + l)));
            __m256i ys2 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 2 * ndual + l)));
            __m256i ys3 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 3 * ndual + l)));
            __m256i ys4 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 4 * ndual + l)));
            __m256i ys5 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 5 * ndual + l)));
            __m256i ys6 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 6 * ndual + l)));
            __m256i ys7 = _mm256_sub_epi8(_mm256_load_si256((__m256i *)(s1 + j * ndual + l)), _mm256_load_si256((__m256i *)(s2 + 7 * ndual + l)));
        
            accpn0 = _mm256_dpbusd_epi32(accpn0, _mm256_xor_si256(ya0, epi8_sign_bit), ya0);
            accpn1 = _mm256_dpbusd_epi32(accpn1, _mm256_xor_si256(ya1, epi8_sign_bit), ya1);
            accpn2 = _mm256_dpbusd_epi32(accpn2, _mm256_xor_si256(ya2, epi8_sign_bit), ya2);
            accpn3 = _mm256_dpbusd_epi32(accpn3, _mm256_xor_si256(ya3, epi8_sign_bit), ya3);
            accpn4 = _mm256_dpbusd_epi32(accpn4, _mm256_xor_si256(ya4, epi8_sign_bit), ya4);
            accpn5 = _mm256_dpbusd_epi32(accpn5, _mm256_xor_si256(ya5, epi8_sign_bit), ya5);
            accpn6 = _mm256_dpbusd_epi32(accpn6, _mm256_xor_si256(ya6, epi8_sign_bit), ya6);
            accpn7 = _mm256_dpbusd_epi32(accpn7, _mm256_xor_si256(ya7, epi8_sign_bit), ya7);

            accps0 = _mm256_dpbusd_epi32(accps0, epi8_sign_bit, ya0);
            accps1 = _mm256_dpbusd_epi32(accps1, epi8_sign_bit, ya1);
            accps2 = _mm256_dpbusd_epi32(accps2, epi8_sign_bit, ya2);
            accps3 = _mm256_dpbusd_epi32(accps3, epi8_sign_bit, ya3);
            accps4 = _mm256_dpbusd_epi32(accps4, epi8_sign_bit, ya4);
            accps5 = _mm256_dpbusd_epi32(accps5, epi8_sign_bit, ya5);
            accps6 = _mm256_dpbusd_epi32(accps6, epi8_sign_bit, ya6);
            accps7 = _mm256_dpbusd_epi32(accps7, epi8_sign_bit, ya7);

            accnn0 = _mm256_dpbusd_epi32(accnn0, _mm256_xor_si256(ys0, epi8_sign_bit), ys0);
            accnn1 = _mm256_dpbusd_epi32(accnn1, _mm256_xor_si256(ys1, epi8_sign_bit), ys1);
            accnn2 = _mm256_dpbusd_epi32(accnn2, _mm256_xor_si256(ys2, epi8_sign_bit), ys2);
            accnn3 = _mm256_dpbusd_epi32(accnn3, _mm256_xor_si256(ys3, epi8_sign_bit), ys3);
            accnn4 = _mm256_dpbusd_epi32(accnn4, _mm256_xor_si256(ys4, epi8_sign_bit), ys4);
            accnn5 = _mm256_dpbusd_epi32(accnn5, _mm256_xor_si256(ys5, epi8_sign_bit), ys5);
            accnn6 = _mm256_dpbusd_epi32(accnn6, _mm256_xor_si256(ys6, epi8_sign_bit), ys6);
            accnn7 = _mm256_dpbusd_epi32(accnn7, _mm256_xor_si256(ys7, epi8_sign_bit), ys7);

            accns0 = _mm256_dpbusd_epi32(accns0, epi8_sign_bit, ys0);
            accns1 = _mm256_dpbusd_epi32(accns1, epi8_sign_bit, ys1);
            accns2 = _mm256_dpbusd_epi32(accns2, epi8_sign_bit, ys2);
            accns3 = _mm256_dpbusd_epi32(accns3, epi8_sign_bit, ys3);
            accns4 = _mm256_dpbusd_epi32(accns4, epi8_sign_bit, ys4);
            accns5 = _mm256_dpbusd_epi32(accns5, epi8_sign_bit, ys5);
            accns6 = _mm256_dpbusd_epi32(accns6, epi8_sign_bit, ys6);
            accns7 = _mm256_dpbusd_epi32(accns7, epi8_sign_bit, ys7);
        }
    
        accpn0 = _mm256_sub_epi32(accpn0, accps0);
        accpn1 = _mm256_sub_epi32(accpn1, accps1);
        accpn2 = _mm256_sub_epi32(accpn2, accps2);
        accpn3 = _mm256_sub_epi32(accpn3, accps3);
        accpn4 = _mm256_sub_epi32(accpn4, accps4);
        accpn5 = _mm256_sub_epi32(accpn5, accps5);
        accpn6 = _mm256_sub_epi32(accpn6, accps6);
        accpn7 = _mm256_sub_epi32(accpn7, accps7);

        accnn0 = _mm256_sub_epi32(accnn0, accns0);
        accnn1 = _mm256_sub_epi32(accnn1, accns1);
        accnn2 = _mm256_sub_epi32(accnn2, accns2);
        accnn3 = _mm256_sub_epi32(accnn3, accns3);
        accnn4 = _mm256_sub_epi32(accnn4, accns4);
        accnn5 = _mm256_sub_epi32(accnn5, accns5);
        accnn6 = _mm256_sub_epi32(accnn6, accns6);
        accnn7 = _mm256_sub_epi32(accnn7, accns7);

        accpn0 = _mm256_hadd_epi32(accpn0, accpn1);
        accpn2 = _mm256_hadd_epi32(accpn2, accpn3);
        accpn4 = _mm256_hadd_epi32(accpn4, accpn5);
        accpn6 = _mm256_hadd_epi32(accpn6, accpn7);
        accpn0 = _mm256_hadd_epi32(accpn0, accpn2);
        accpn4 = _mm256_hadd_epi32(accpn4, accpn6);

        accnn0 = _mm256_hadd_epi32(accnn0, accnn1);
        accnn2 = _mm256_hadd_epi32(accnn2, accnn3);
        accnn4 = _mm256_hadd_epi32(accnn4, accnn5);
        accnn6 = _mm256_hadd_epi32(accnn6, accnn7);
        accnn0 = _mm256_hadd_epi32(accnn0, accnn2);
        accnn4 = _mm256_hadd_epi32(accnn4, accnn6);

        __m256i accpnlo = _mm256_permute2f128_si256(accpn0, accpn4, 48);
        __m256i accpnhi = _mm256_permute2f128_si256(accpn0, accpn4, 33);
        __m256i accnnlo = _mm256_permute2f128_si256(accnn0, accnn4, 48);
        __m256i accnnhi = _mm256_permute2f128_si256(accnn0, accnn4, 33);

        __m256i accpn = _mm256_add_epi32(accpnlo, accpnhi);
        __m256i accnn = _mm256_add_epi32(accnnlo, accnnhi);

        cmpp[0] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accpn))) << (j * 8);
        cmpn[0] |= ((uint64_t) _mm256_movemask_ps((__m256) _mm256_cmpgt_epi32(th, accnn))) << (j * 8);
    }
}

template <uint32_t nb>
template <uint32_t ndual>
uint64_t Pool_epi8_t<nb>::_vnpnorm(int8_t *s1, int8_t *s2) {
    __m256i accpn = _mm256_setzero_si256();
    __m256i accnn = _mm256_setzero_si256();
    __m256i accps = _mm256_setzero_si256();
    __m256i accns = _mm256_setzero_si256();
    for (long l = 0; l < ndual; l += 32) {
        __m256i y1 = _mm256_load_si256((__m256i *)(s1 + l));
        __m256i y2 = _mm256_load_si256((__m256i *)(s2 + l));
        __m256i ya = _mm256_add_epi8(y1, y2);
        __m256i ys = _mm256_sub_epi8(y1, y2);
        accpn = _mm256_dpbusd_epi32(accpn, _mm256_xor_si256(ya, epi8_sign_bit), ya);
        accnn = _mm256_dpbusd_epi32(accnn, _mm256_xor_si256(ys, epi8_sign_bit), ys);
        accps = _mm256_dpbusd_epi32(accps, epi8_sign_bit, ya);
        accns = _mm256_dpbusd_epi32(accns, epi8_sign_bit, ys);
    }
    accpn = _mm256_sub_epi32(accpn, accps);
    accnn = _mm256_sub_epi32(accnn, accns);

    __m128i accp128 = _mm_add_epi32(_mm256_castsi256_si128(accpn), _mm256_extracti128_si256(accpn, 1));
    __m128i accn128 = _mm_add_epi32(_mm256_castsi256_si128(accnn), _mm256_extracti128_si256(accnn, 1));
    accp128 = _mm_add_epi32(accp128, _mm_shuffle_epi32(accp128, 78));
    accn128 = _mm_add_epi32(accn128, _mm_shuffle_epi32(accn128, 78));
    accp128 = _mm_add_epi32(accp128, _mm_shuffle_epi32(accp128, 177));
    accn128 = _mm_add_epi32(accn128, _mm_shuffle_epi32(accn128, 177));
    uint64_t ret_lo = _mm_cvtsi128_si32(accp128);
    uint64_t ret_hi = _mm_cvtsi128_si32(accn128);
    return ( (ret_hi << 32) | ret_lo );
}
template <uint32_t nb>
template <uint32_t ndual>
inline int32_t Pool_epi8_t<nb>::_compute_norm(int8_t *s) {
    __m256i accn = _mm256_setzero_si256();
    __m256i accs = _mm256_setzero_si256();
    for (long l = 0; l < ndual; l += 32) {
        __m256i y = _mm256_load_si256((__m256i *)(s + l));
        accn = _mm256_dpbusd_epi32(accn, _mm256_xor_si256(epi8_sign_bit, y), y);
        accs = _mm256_dpbusd_epi32(accs, epi8_sign_bit, y);
    }
    accn = _mm256_sub_epi32(accn, accs);
    __m128i acc128 = _mm_add_epi32(_mm256_castsi256_si128(accn), _mm256_extracti128_si256(accn, 1));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 78));
    acc128 = _mm_add_epi32(acc128, _mm_shuffle_epi32(acc128, 177));
    return _mm_cvtsi128_si32(acc128);
}
template <uint32_t nb>
template <uint32_t ndual>
inline __m256i Pool_epi8_t<nb>::_compute_norm_b8(int8_t *s) {
    __m256i accn0 = _mm256_setzero_si256();
    __m256i accn1 = _mm256_setzero_si256();
    __m256i accn2 = _mm256_setzero_si256();
    __m256i accn3 = _mm256_setzero_si256();
    __m256i accn4 = _mm256_setzero_si256();
    __m256i accn5 = _mm256_setzero_si256();
    __m256i accn6 = _mm256_setzero_si256();
    __m256i accn7 = _mm256_setzero_si256();
    __m256i accs0 = _mm256_setzero_si256();
    __m256i accs1 = _mm256_setzero_si256();
    __m256i accs2 = _mm256_setzero_si256();
    __m256i accs3 = _mm256_setzero_si256();
    __m256i accs4 = _mm256_setzero_si256();
    __m256i accs5 = _mm256_setzero_si256();
    __m256i accs6 = _mm256_setzero_si256();
    __m256i accs7 = _mm256_setzero_si256();
    for (long l = 0; l < ndual; l += 32) {
        __m256i y0 = _mm256_load_si256((__m256i *)(s+l+0*ndual));
        __m256i y1 = _mm256_load_si256((__m256i *)(s+l+1*ndual));
        __m256i y2 = _mm256_load_si256((__m256i *)(s+l+2*ndual));
        __m256i y3 = _mm256_load_si256((__m256i *)(s+l+3*ndual));
        __m256i y4 = _mm256_load_si256((__m256i *)(s+l+4*ndual));
        __m256i y5 = _mm256_load_si256((__m256i *)(s+l+5*ndual));
        __m256i y6 = _mm256_load_si256((__m256i *)(s+l+6*ndual));
        __m256i y7 = _mm256_load_si256((__m256i *)(s+l+7*ndual));

        accn0 = _mm256_dpbusd_epi32(accn0, _mm256_xor_si256(y0, epi8_sign_bit), y0);
        accn1 = _mm256_dpbusd_epi32(accn1, _mm256_xor_si256(y1, epi8_sign_bit), y1);
        accn2 = _mm256_dpbusd_epi32(accn2, _mm256_xor_si256(y2, epi8_sign_bit), y2);
        accn3 = _mm256_dpbusd_epi32(accn3, _mm256_xor_si256(y3, epi8_sign_bit), y3);
        accn4 = _mm256_dpbusd_epi32(accn4, _mm256_xor_si256(y4, epi8_sign_bit), y4);
        accn5 = _mm256_dpbusd_epi32(accn5, _mm256_xor_si256(y5, epi8_sign_bit), y5);
        accn6 = _mm256_dpbusd_epi32(accn6, _mm256_xor_si256(y6, epi8_sign_bit), y6);
        accn7 = _mm256_dpbusd_epi32(accn7, _mm256_xor_si256(y7, epi8_sign_bit), y7);
        accs0 = _mm256_dpbusd_epi32(accs0, epi8_sign_bit, y0);
        accs1 = _mm256_dpbusd_epi32(accs1, epi8_sign_bit, y1);
        accs2 = _mm256_dpbusd_epi32(accs2, epi8_sign_bit, y2);
        accs3 = _mm256_dpbusd_epi32(accs3, epi8_sign_bit, y3);
        accs4 = _mm256_dpbusd_epi32(accs4, epi8_sign_bit, y4);
        accs5 = _mm256_dpbusd_epi32(accs5, epi8_sign_bit, y5);
        accs6 = _mm256_dpbusd_epi32(accs6, epi8_sign_bit, y6);
        accs7 = _mm256_dpbusd_epi32(accs7, epi8_sign_bit, y7);
    }
    accn0 = _mm256_sub_epi32(accn0, accs0);
    accn1 = _mm256_sub_epi32(accn1, accs1);
    accn2 = _mm256_sub_epi32(accn2, accs2);
    accn3 = _mm256_sub_epi32(accn3, accs3);
    accn4 = _mm256_sub_epi32(accn4, accs4);
    accn5 = _mm256_sub_epi32(accn5, accs5);
    accn6 = _mm256_sub_epi32(accn6, accs6);
    accn7 = _mm256_sub_epi32(accn7, accs7);

    accn0 = _mm256_hadd_epi32(accn0, accn1);
    accn2 = _mm256_hadd_epi32(accn2, accn3);
    accn4 = _mm256_hadd_epi32(accn4, accn5);
    accn6 = _mm256_hadd_epi32(accn6, accn7);
    accn0 = _mm256_hadd_epi32(accn0, accn2);
    accn4 = _mm256_hadd_epi32(accn4, accn6);
    __m256i accnlo = _mm256_permute2f128_si256(accn0, accn4, 48);
    __m256i accnhi = _mm256_permute2f128_si256(accn0, accn4, 33);
    return _mm256_add_epi32(accnlo, accnhi);
}


template <uint32_t nb>
template <uint32_t ndual>
void Pool_epi8_t<nb>::_process_ndhl1_triblock(int8_t *dh, long bias, long bound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num) {
    const __m256i th = _mm256_set1_epi32(threshold);

    long ind = 0;
    while (ind < bound - 7) {
        long jnd = ind;
        while (jnd < bound - 7) {
            uint64_t cmpp = 0, cmpn = 0;
            _vnpnorm_check8x8<ndual>(&cmpp, &cmpn, dh + ind * ndual, dh + jnd * ndual, th);
            if (jnd == ind) {
                cmpp &= 0x7f3f1f0f07030100ULL;
                cmpn &= 0x7f3f1f0f07030100ULL;
                uint64_t cmp = _mm256_movemask_ps(_mm256_cmpgt_epi32(th, _compute_norm_b8<ndual>(dh + jnd * ndual)));
                CHECK_AND_ADD8(cmp, (bias + ind));
            }
            CHECK_AND_ADD8x8(cmpp, (bias + ind), (bias + jnd), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (bias + ind), (bias + jnd), ADDS_TO_BUFFER);
            jnd += 8;
        }
        if (jnd < bound) {
            const long jrem = bound - jnd;
            uint64_t cmpp = 0, cmpn = 0;
            _vnpnorm_check8xn<ndual>(&cmpp, &cmpn, dh + jnd * ndual, dh + ind * ndual, jrem, th);
            CHECK_AND_ADD8x8(cmpp, (bias + jnd), (bias + ind), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (bias + jnd), (bias + ind), ADDS_TO_BUFFER);            
        }
        ind += 8;
    }
    if (ind < bound) {
        const long irem = bound - ind;
        for (long i = 0; i < irem; i++) {
            if (_compute_norm<ndual>(dh + (ind+i) * ndual) < threshold) ADD_TO_BUFFER(bias + ind + i);
            for (long j = i + 1; j < irem; j++) {
                uint64_t npnorm = _vnpnorm<ndual>(dh + (ind + i) * ndual, dh + (ind + j) * ndual);         
                int32_t *npnorm_epi32 = (int32_t *)(&npnorm);       
                if (npnorm_epi32[0] < threshold) ADDA_TO_BUFFER((bias + ind + i), (bias + ind + j));
                if (npnorm_epi32[1] < threshold) ADDS_TO_BUFFER((bias + ind + i), (bias + ind + j));
            }
        }
    }
}

template <uint32_t nb>
template <uint32_t ndual>
void Pool_epi8_t<nb>::_process_ndhl1_block(int8_t *dhi, int8_t *dhj, long ibias, long jbias, 
                                        long ibound, long jbound, int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num) {
    const __m256i th = _mm256_set1_epi32(threshold);

    long ind = 0;
    while (ind < ibound - 7) {
        long jnd = 0;
        while (jnd < jbound - 7) {
            uint64_t cmpp = 0, cmpn = 0;
            _vnpnorm_check8x8<ndual>(&cmpp, &cmpn, dhi + ind * ndual, dhj + jnd * ndual, th);
            CHECK_AND_ADD8x8(cmpp, (ibias + ind), (jbias + jnd), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (ibias + ind), (jbias + jnd), ADDS_TO_BUFFER);
            jnd += 8;            
        }
        if (jnd < jbound) {
            const long jrem = jbound - jnd;
            uint64_t cmpp = 0, cmpn = 0;
            _vnpnorm_check8xn<ndual>(&cmpp, &cmpn, dhi + ind * ndual, dhj + jnd * ndual, jrem, th);
            CHECK_AND_ADD8x8(cmpp, (ibias + ind), (jbias + jnd), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (ibias + ind), (jbias + jnd), ADDS_TO_BUFFER);
        }
        ind += 8;
    }
    if (ind < ibound) {
        const long irem = ibound - ind;
        long jnd = 0;
        while (jnd < jbound) {
            uint64_t cmpp = 0, cmpn = 0;
            _vnpnorm_check8xn<ndual>(&cmpp, &cmpn, dhj + jnd * ndual, dhi + ind * ndual, irem, th);
            CHECK_AND_ADD8x8(cmpp, (jbias + jnd), (ibias + ind), ADDA_TO_BUFFER);
            CHECK_AND_ADD8x8(cmpn, (jbias + jnd), (ibias + ind), ADDS_TO_BUFFER);
            jnd += 8;
        }
    }
}
#pragma endregion

template <uint32_t nb>
template <uint32_t ndual, uint32_t dh_dim, uint32_t l1_block, uint32_t l2_block, uint32_t m_block>
int Pool_epi8_t<nb>::_naivedh_insert(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length) {
    ///////////////// check input /////////////////
    if (target_index + dh_dim > index_l) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::naivedh_insert: target_index(%ld) + dh_dim(%u) > index_l(%ld), nothing done.\n", 
                nb, target_index, dh_dim, index_l);
        return -1;
    }
    if (target_index < 0) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::naivedh_insert: negetive target_index(%ld), nothing done.\n", nb, target_index);
        return -1;
    }

    ///////////////// prepare global data /////////////////
    const long FD = index_r - target_index;
    const long ID = index_l - dh_dim - target_index;
    const long FD8 = ( (FD + 7) / 8 ) * 8;
    Lattice_QP *b_full = basis->b_loc_QP(target_index, index_r);
    
    float **b_full_fp = (float **) NEW_MAT(FD, FD, sizeof(float));
    float **b_ext_fp = (float **) NEW_MAT(CSD, FD, sizeof(float));
    for (long i = 0; i < FD; i++) {
        for (long j = 0; j <= i; j++) b_full_fp[i][j] = b_full->get_b().hi[i][j];
    }
    do {
        __attribute__ ((aligned (32))) float extmp[vec_length] = {};
        for (long i = 0; i < CSD; i++) {
            copy_avx2(extmp, _b_local[i], CSD);
            mul_avx2(extmp, 1.0/_ratio, CSD);
            for (long j = CSD - 1; j >= 0; j--) {
                float q = round(extmp[j] / b_full_fp[FD - CSD + j][FD - CSD + j]);
                for (long l = 0; l < CSD; l++) extmp[l] -= q * b_full_fp[FD-CSD+j][FD-CSD+l];
                red_avx2(b_ext_fp[i], b_full_fp[FD-CSD+j], -q, FD);
            }
        }
    } while(0);


    ///////////////// profiling data /////////////////
    uint64_t num_mblock = 0;
    uint64_t total_mblock = (num_vec + NDH_M_BLOCK - 1) / NDH_M_BLOCK;
    total_mblock = total_mblock * (total_mblock + 1) / 2;
    uint64_t mblock_ops = 0;
    uint64_t dp_ops = 0;
    uint64_t lift_ops = 0;
    double mblock_time = 0.0;
    double dp_time = 0.0;
    double lift_time = 0.0;
    pthread_spinlock_t profile_lock;
    pthread_spin_init(&profile_lock, PTHREAD_PROCESS_SHARED);

    ///////////////// main loop /////////////////
    // mblock data
    float *_fvec0 = (float *) NEW_VEC(NDH_M_BLOCK * FD8, sizeof(float));
    float *_fvec1 = (float *) NEW_VEC(NDH_M_BLOCK * FD8, sizeof(float));
    int8_t *_dh0 = (int8_t *) NEW_VEC(NDH_M_BLOCK * ndual, sizeof(float));
    int8_t *_dh1 = (int8_t *) NEW_VEC(NDH_M_BLOCK * ndual, sizeof(float));
    long _m0 = -1;
    long _m1 = -1;
    // min data
    __attribute__ ((aligned (32))) float min_norm[vec_length];
    float **min_vec = (float **) NEW_MAT(ID, FD, sizeof(float));
    for (long i = 0; i < ID; i++) min_norm[i] = (float) (0.995 * basis->get_B().hi[i + target_index]);
    if (target_length != 0.0) {
        for (long i = 0; i < ID; i++) {
            if (min_norm[i] > 1.0001 * target_length * target_length) min_norm[i] = 1.0001 * target_length * target_length;
        }
    }
    pthread_spinlock_t min_lock;
    pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);
    // buffer and the size
    uint32_t **ptr_buffer = (uint32_t **) NEW_VEC(num_threads * 8, sizeof(uint32_t *));
    long *ptr_buffer_size = (long *) NEW_VEC(num_threads * 8, sizeof(long));
    for (long i = 0; i < num_threads; i++) {
        ptr_buffer[i*8] = (uint32_t *) malloc(8192 * 3 * sizeof(uint32_t));
        ptr_buffer_size[i*8] = 8192;
    }
    
    #if NDH_DIAG_FIRST
    for (long DIFF = 0; DIFF < num_vec; DIFF += NDH_M_BLOCK) {
        for (long MInd = 0; MInd < num_vec - DIFF; MInd += NDH_M_BLOCK) {
            long MJnd = MInd + DIFF;
    #else
    for (long MInd = 0; MInd < num_vec; MInd += NDH_M_BLOCK) {
        for (long MJnd = MInd; MJnd < num_vec; MJnd += NDH_M_BLOCK) {
    #endif  
            const long MIbound = (MInd + NDH_M_BLOCK > num_vec) ? (num_vec - MInd) : NDH_M_BLOCK;
            const long MJbound = (MJnd + NDH_M_BLOCK > num_vec) ? (num_vec - MJnd) : NDH_M_BLOCK;
            num_mblock++;

            // prepare mblock data
            TIMER_START;
            float *fveci = NULL, *fvecj = NULL;
            int8_t *dhi = NULL, *dhj = NULL;
            do {
                if (MInd == _m0) {
                    fveci = _fvec0;
                    dhi = _dh0;
                } 
                if (MInd == _m1) {
                    fveci = _fvec1;
                    dhi = _dh1;
                }
                if (MJnd == _m0) {
                    fvecj = _fvec0;
                    dhj = _dh0;
                }
                if (MJnd == _m1) {
                    fvecj = _fvec1;
                    dhj = _dh1;
                }
                if (fveci == NULL) {
                    if (_fvec0 == fvecj) {
                        fveci = _fvec1;
                        dhi = _dh1;
                        _m1 = MInd;
                    } else {
                        fveci = _fvec0;
                        dhi = _dh0;
                        _m0 = MInd;
                    }
                    _compute_ndh_mblock<ndual, dh_dim>(fveci, dhi, b_ext_fp, dual_vec, MInd, MIbound, target_index);
                    if (log_level >= 0) mblock_ops += MIbound;
                }
                if (fvecj == NULL) {
                    if (MInd == MJnd) {
                        fvecj = fveci;
                        dhj = dhi;
                    } else {
                        if (fveci == _fvec1) {
                            fvecj = _fvec0;
                            dhj = _dh0;
                            _m0 = MJnd;
                        } else {
                            fvecj = _fvec1;
                            dhj = _dh1;
                            _m1 = MJnd;
                        }
                        _compute_ndh_mblock<ndual, dh_dim>(fvecj, dhj, b_ext_fp, dual_vec, MJnd, MJbound, target_index);
                        if (log_level >= 0) mblock_ops += MJbound;
                    }
                }
            } while (0);
            TIMER_END;
            mblock_time += CURRENT_TIME;
            
            // create l2 block list
            long *l2_block_list = (long *) NEW_VEC((NDH_M_BLOCK/NDH_L2_BLOCK) * (NDH_M_BLOCK/NDH_L2_BLOCK) * 2, sizeof(long));
            long n_l2_block = 0;
            if (MInd != MJnd) {
                for (long Ind = 0; Ind < MIbound; Ind += NDH_L2_BLOCK) {
                    for (long Jnd = 0; Jnd < MJbound; Jnd += NDH_L2_BLOCK) {
                        l2_block_list[n_l2_block*2] = Ind;
                        l2_block_list[n_l2_block*2+1] = Jnd;
                        n_l2_block++;
                    }
                }
                dp_ops += MIbound * MJbound;
            } else {
                for (long Ind = 0; Ind < MIbound; Ind += NDH_L2_BLOCK) {
                    for (long Jnd = Ind; Jnd < MJbound; Jnd += NDH_L2_BLOCK) {
                        l2_block_list[n_l2_block*2] = Ind;
                        l2_block_list[n_l2_block*2+1] = Jnd;
                        n_l2_block++;
                    }
                }
                dp_ops += MIbound * (MIbound + 1) / 2;
            }

            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                // local min data
                __attribute__ ((aligned (32))) float _min_norm[vec_length];
                float **_min_vec = (float **) NEW_MAT(ID, FD, sizeof(float));
                pthread_spin_lock(&min_lock);
                copy_avx2(_min_norm, min_norm, ID);
                pthread_spin_unlock(&min_lock);

                #if 0
                __attribute__ ((aligned (32))) float th_norm[6] = {2900.0f, 2860.0f, 2820.0f, 2780.0f, 2740.0f, 2700.0f};
                for (long i = 0; i < 6; i++) th_norm[i] *= th_norm[i];
                #endif
                long ptr_buffer_num = 0;
                uint32_t *_ptr_buffer = ptr_buffer[thread * 8];
                long _ptr_buffer_size = ptr_buffer_size[thread * 8];

                TIMER_START;
                long n_begin = (n_l2_block * thread) / num_threads;
                long n_end = (n_l2_block * (thread + 1)) / num_threads;
                long n_curr = n_begin;
                while (n_curr < n_end) {
                    long Ind = l2_block_list[n_curr*2];
                    long Jnd = l2_block_list[n_curr*2+1];
                    if (Ind == Jnd && MInd == MJnd) {
                        const long Ibound = (Ind + NDH_L2_BLOCK > MIbound) ? MIbound : Ind + NDH_L2_BLOCK;
                        const long Jbound = (Jnd + NDH_L2_BLOCK > MJbound) ? MJbound : Jnd + NDH_L2_BLOCK;
                        long ind = Ind;
                        while (ind <= Ibound - NDH_L1_BLOCK) {
                            long jnd = ind;
                            // the first triangular block
                            _process_ndhl1_triblock<ndual>(dhi + ind * ndual, ind, NDH_L1_BLOCK, threshold, 
                                                        &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                            jnd += NDH_L1_BLOCK;
                            // square blocks
                            while (jnd <= Jbound - NDH_L1_BLOCK) {
                                _process_ndhl1_block<ndual>(dhi + ind * ndual, dhj + jnd * ndual, 
                                                        ind, jnd, NDH_L1_BLOCK, NDH_L1_BLOCK, threshold,
                                                        &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                                jnd += NDH_L1_BLOCK;
                            }
                            if (jnd < Jbound) {
                                _process_ndhl1_block<ndual>(dhi + ind * ndual, dhj + jnd * ndual, 
                                                        ind, jnd, NDH_L1_BLOCK, Jbound - jnd, threshold,
                                                        &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                            }
                            ind += NDH_L1_BLOCK;
                        }
                        if (ind < Ibound) {
                            long jnd = ind;
                            _process_ndhl1_triblock<ndual>(dhi + ind * ndual, ind, Ibound - ind, threshold,
                                                        &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                        }
                    } else {
                        const long Ibound = (Ind + NDH_L2_BLOCK > MIbound) ? MIbound : Ind + NDH_L2_BLOCK;
                        const long Jbound = (Jnd + NDH_L2_BLOCK > MJbound) ? MJbound : Jnd + NDH_L2_BLOCK;
                        long ind = Ind;
                        while (ind <= Ibound - NDH_L1_BLOCK) {
                            long jnd = Jnd;
                            while (jnd <= Jbound - NDH_L1_BLOCK) {
                                _process_ndhl1_block<ndual>(dhi + ind * ndual, dhj + jnd * ndual,
                                                    ind, jnd, NDH_L1_BLOCK, NDH_L1_BLOCK, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                                jnd += NDH_L1_BLOCK;
                            }
                            if (jnd < Jbound) {
                                _process_ndhl1_block<ndual>(dhi + ind * ndual, dhj + jnd * ndual,
                                                    ind, jnd, NDH_L1_BLOCK, Jbound - jnd, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                            }
                            ind += NDH_L1_BLOCK;
                        }
                        if (ind < Ibound) {
                            long jnd = Jnd;
                            while (jnd <= Jbound - NDH_L1_BLOCK) {
                                _process_ndhl1_block<ndual>(dhi + ind * ndual, dhj + jnd * ndual,
                                                    ind, jnd, Ibound - ind, NDH_L1_BLOCK, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                                jnd += NDH_L1_BLOCK;
                            }
                            if (jnd < Jbound) {
                                _process_ndhl1_block<ndual>(dhi + ind * ndual, dhj + jnd * ndual,
                                                    ind, jnd, Ibound - ind, Jbound - jnd, threshold,
                                                    &_ptr_buffer, &_ptr_buffer_size, &ptr_buffer_num);
                            }
                        }
                    }
                    n_curr++;
                }
                TIMER_END;
                double _dp_time = CURRENT_TIME;
                ptr_buffer[thread * 8] = _ptr_buffer;
                ptr_buffer_size[thread * 8] = _ptr_buffer_size;
                
                TIMER_START;
                do {
                    const long LD = FD - CSD;
                    float *ftmp = (float *) NEW_VEC(FD8 * 8, sizeof(float));
                    float idiag[256];
                    for (long i = 0; i < FD; i++) idiag[i] = 1.0 / b_full_fp[i][i];
                    __attribute__ ((aligned (32))) float curr_norm[8];
                    long ind = 0;
                    while (ind <= ptr_buffer_num - 8) {
                        for (long i = 0; i < 8; i++) {
                            if (_ptr_buffer[(ind+i) * 3] == 1) {
                                sub_avx2(ftmp + FD8 * i, fveci + _ptr_buffer[(ind+i)*3+1]*FD8, fvecj + _ptr_buffer[(ind+i)*3+2]*FD8, FD8);
                            } else if (_ptr_buffer[(ind+i) * 3] == 0) {
                                add_avx2(ftmp + FD8 * i, fveci + _ptr_buffer[(ind+i)*3+1]*FD8, fvecj + _ptr_buffer[(ind+i)*3+2]*FD8, FD8);
                            } else {
                                copy_avx2(ftmp + FD8 * i, fveci + _ptr_buffer[(ind+i)*3+1]*FD8, FD8);
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
                            #if 0
                            int thr = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_broadcast_ss(th_norm+i), _mm256_load_ps(curr_norm), 30));
                            while (thr) {
                                int rr = __builtin_ctz(thr);
                                thr -= (1 << rr);
                                int dn = 0;
                                float ln = 0, tn1 = 0, tn2 = 0, tn = 0, aa = 0;
                                int8_t *ptr1 = dhi + _ptr_buffer[(ind+rr)*3+1]*ndual;
                                int8_t *ptr2 = dhj + _ptr_buffer[(ind+rr)*3+2]*ndual;
                                float *fptr1 = fveci + _ptr_buffer[(ind+rr)*3+1]*FD8;
                                float *fptr2 = fvecj + _ptr_buffer[(ind+rr)*3+2]*FD8;
                                float *fptr = ftmp + rr * FD8;
                                if (_ptr_buffer[(ind+rr)*3] == 0) {
                                    for (long l = 0; l < ndual; l++) dn += (int)((int8_t)(ptr1[l] + ptr2[l])) * (int)((int8_t)(ptr1[l] + ptr2[l]));
                                } else if (_ptr_buffer[(ind+rr)*3] == 1) {
                                    for (long l = 0; l < ndual; l++) dn += (int)((int8_t)(ptr1[l] - ptr2[l])) * (int)((int8_t)(ptr1[l] - ptr2[l]));
                                } else {
                                    for (long l = 0; l < ndual; l++) dn += (int)(ptr1[l]) * (int)(ptr1[l]);
                                }
                                
                                for (long l = ID; l < ID + dh_dim; l++) ln += fptr[l] * fptr[l];
                                for (long l = FD - CSD; l < FD; l++) {
                                    tn1 += fptr1[l] * fptr1[l];
                                    tn2 += fptr2[l] * fptr2[l];
                                    tn += fptr[l] * fptr[l];
                                    aa += fptr1[l] * fptr2[l];
                                }
                                tn1 = sqrt(tn1);
                                tn2 = sqrt(tn2);
                                ln = sqrt(ln);
                                tn = sqrt(tn);
                                aa /= tn1 * tn2;
                                aa = fabs(aa);
                                printf("[%ld] norm = %.2f, dn = %d, ln = %.2f(%.2f*avg/%.2f), tn = %.2f(%.2f) & %.2f(%.2f) ==> %.2f(%.2f)\n", 
                                            i, sqrt(curr_norm[rr]), dn, ln, ln/3032.0, ln/sqrt((curr_norm[rr]*dh_dim)/(FD - i + 1)), tn1, tn1/1466, tn2, tn2/1466, tn, aa);
                                fflush(stdout);
                            }
                            #endif
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
                    while (ind < ptr_buffer_num) {
                        if (_ptr_buffer[ind * 3] == 1) {
                            sub_avx2(ftmp, fveci + _ptr_buffer[ind*3+1]*FD8, fvecj + _ptr_buffer[ind*3+2]*FD8, FD8);
                        } else if (_ptr_buffer[ind * 3] == 0) {
                            add_avx2(ftmp, fveci + _ptr_buffer[ind*3+1]*FD8, fvecj + _ptr_buffer[ind*3+2]*FD8, FD8);
                        } else {
                            copy_avx2(ftmp, fveci + _ptr_buffer[ind*3+1]*FD8, FD8);
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
                TIMER_END;
                double _lift_time = CURRENT_TIME;
                if (log_level >= 0) {
                    pthread_spin_lock(&profile_lock);
                    dp_time += _dp_time;
                    lift_time += _lift_time;
                    lift_ops += ptr_buffer_num;
                    pthread_spin_unlock(&profile_lock);
                }

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

            FREE_VEC(l2_block_list);
            
            if (log_level >= 1) {
                fprintf(stdout, "mblock %lu / %lu, mblock_time = %f, dp_time = %f, lift_time = %f\n", num_mblock, total_mblock, 
                                mblock_time, dp_time, lift_time);
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
    }
    if (log_level >= 0) {
        fprintf(stdout, "search done, mblock_time = %f, dp_time = %f, lift_time = %f\n", 
                        mblock_time, dp_time, lift_time);
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

    delete b_full;
    FREE_VEC(_fvec0);
    FREE_VEC(_fvec1);
    FREE_VEC((void *)_dh0);
    FREE_VEC((void *)_dh1);
    for (long i = 0; i < num_threads; i++) free(ptr_buffer[i*8]);
    FREE_VEC((void *)ptr_buffer);
    FREE_VEC((void *)ptr_buffer_size);
    FREE_MAT(b_ext_fp);

    long min_place = -1;
    do {
        double min_score = 1e100;
        for (long i = 0; i < ID; i++) {
            float old_norm = (float) (0.995 * basis->get_B().hi[i + target_index]);
            if (target_length != 0.0 && min_norm[i] > 1.00009 * target_length * target_length) continue;
            if (min_norm[i] < old_norm) {
                if (min_score > min_norm[i]/old_norm * pow(eta, i)) {
                    min_score = min_norm[i]/old_norm * pow(eta, i);
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
    return ret;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::naivedh_insert(long target_index, double eta, long log_level, double target_length) {
    const double tail_gh = sqrt(gh2);
    const double dual_gh = basis->gh(index_l - NDH_DEFAULT_DHDIM, index_l);
    const double lift_gh = basis->gh(target_index, index_l);
    const double tail_exp_ratio = sqrt(cvec[3*(num_vec/2)+2] * 4.0) / _ratio / tail_gh;

    double tail_exp_alpha;
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
    // a conservative estimation for normal case
    const double dual_exp_length = unique_target ? 
                        target_length * sqrt(NDH_DEFAULT_DHDIM) / sqrt(index_r - target_index) :
                        lift_exp_length * sqrt(NDH_DEFAULT_DHDIM) / sqrt(index_l - target_index);
                        

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

    float *dual_vec = (float *) NEW_VEC(NDH_DOUBLE_NDUAL * NDH_DEFAULT_DHDIM, sizeof(float));
    Lattice_QP *b_mid = basis->b_loc_QP(index_l - NDH_DEFAULT_DHDIM, index_l);
    gen_dual_vec_list(dual_vec, b_mid, log_level, NDH_DEFAULT_NDUAL);

    int32_t threshold = estimate_ndh_threshold(dual_exp_length, dual_vec, NDH_DEFAULT_DHDIM, NDH_DEFAULT_NDUAL, 0.9);

    int ret = 0;
    if (threshold >= NDH_DND_MAX_THRESHOLD) {
        gen_dual_vec_list(dual_vec, b_mid, log_level, NDH_DOUBLE_NDUAL);
        threshold = estimate_ndh_threshold(dual_exp_length, dual_vec, NDH_DEFAULT_DHDIM, NDH_DOUBLE_NDUAL, 0.9);
        if (log_level >= 0) printf("ndual = %d, threshold = %d\n", NDH_DOUBLE_NDUAL, threshold);
        ret = _naivedh_insert<NDH_DOUBLE_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(
                target_index, eta, log_level, dual_vec, threshold, unique_target ? target_length : 0.0);
    } else {
        if (log_level >= 0) printf("ndual = %d, threshold = %d\n", NDH_DEFAULT_NDUAL, threshold);
        ret = _naivedh_insert<NDH_DEFAULT_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(
                target_index, eta, log_level, dual_vec, threshold, unique_target ? target_length : 0.0);
    }

    FREE_VEC(dual_vec);
    delete b_mid;
    return ret;
}

#if COMPILE_POOL_EPI8_96
// default ndual
template void Pool_epi8_t<3>::_compute_ndh_mblock<NDH_DEFAULT_NDUAL, NDH_DEFAULT_DHDIM>(float *fvec, int8_t *dh, 
                                            float **b_ext, float *dual_vec, long MInd, long num, long target_index);

template void Pool_epi8_t<3>::_process_ndhl1_triblock<NDH_DEFAULT_NDUAL>(int8_t *dh, long bias, long bound, 
                                                                    int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template void Pool_epi8_t<3>::_process_ndhl1_block<NDH_DEFAULT_NDUAL>(int8_t *dhi, int8_t *dhj, long ibias, long jbias, long ibound, long jbound, 
                                                                int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template int Pool_epi8_t<3>::_naivedh_insert<NDH_DEFAULT_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length);

// double ndual
template void Pool_epi8_t<3>::_compute_ndh_mblock<NDH_DOUBLE_NDUAL, NDH_DEFAULT_DHDIM>(float *fvec, int8_t *dh, 
                                            float **b_ext, float *dual_vec, long MInd, long num, long target_index);

template void Pool_epi8_t<3>::_process_ndhl1_triblock<NDH_DOUBLE_NDUAL>(int8_t *dh, long bias, long bound, 
                                                                    int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template void Pool_epi8_t<3>::_process_ndhl1_block<NDH_DOUBLE_NDUAL>(int8_t *dhi, int8_t *dhj, long ibias, long jbias, long ibound, long jbound, 
                                                                int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template int Pool_epi8_t<3>::_naivedh_insert<NDH_DOUBLE_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length);

template class Pool_epi8_t<3>;
#endif

#if COMPILE_POOL_EPI8_128
// default ndual
template void Pool_epi8_t<4>::_compute_ndh_mblock<NDH_DEFAULT_NDUAL, NDH_DEFAULT_DHDIM>(float *fvec, int8_t *dh, 
                                            float **b_ext, float *dual_vec, long MInd, long num, long target_index);

template void Pool_epi8_t<4>::_process_ndhl1_triblock<NDH_DEFAULT_NDUAL>(int8_t *dh, long bias, long bound, 
                                                                    int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template void Pool_epi8_t<4>::_process_ndhl1_block<NDH_DEFAULT_NDUAL>(int8_t *dhi, int8_t *dhj, long ibias, long jbias, long ibound, long jbound, 
                                                                int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template int Pool_epi8_t<4>::_naivedh_insert<NDH_DEFAULT_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length);

// double ndual
template void Pool_epi8_t<4>::_compute_ndh_mblock<NDH_DOUBLE_NDUAL, NDH_DEFAULT_DHDIM>(float *fvec, int8_t *dh, 
                                            float **b_ext, float *dual_vec, long MInd, long num, long target_index);

template void Pool_epi8_t<4>::_process_ndhl1_triblock<NDH_DOUBLE_NDUAL>(int8_t *dh, long bias, long bound, 
                                                                    int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template void Pool_epi8_t<4>::_process_ndhl1_block<NDH_DOUBLE_NDUAL>(int8_t *dhi, int8_t *dhj, long ibias, long jbias, long ibound, long jbound, 
                                                                int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template int Pool_epi8_t<4>::_naivedh_insert<NDH_DOUBLE_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length);

template class Pool_epi8_t<4>;
#endif

#if COMPILE_POOL_EPI8_160
// default ndual
template void Pool_epi8_t<5>::_compute_ndh_mblock<NDH_DEFAULT_NDUAL, NDH_DEFAULT_DHDIM>(float *fvec, int8_t *dh, 
                                            float **b_ext, float *dual_vec, long MInd, long num, long target_index);

template void Pool_epi8_t<5>::_process_ndhl1_triblock<NDH_DEFAULT_NDUAL>(int8_t *dh, long bias, long bound, 
                                                                    int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template void Pool_epi8_t<5>::_process_ndhl1_block<NDH_DEFAULT_NDUAL>(int8_t *dhi, int8_t *dhj, long ibias, long jbias, long ibound, long jbound, 
                                                                int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template int Pool_epi8_t<5>::_naivedh_insert<NDH_DEFAULT_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length);

// double ndual
template void Pool_epi8_t<5>::_compute_ndh_mblock<NDH_DOUBLE_NDUAL, NDH_DEFAULT_DHDIM>(float *fvec, int8_t *dh, 
                                            float **b_ext, float *dual_vec, long MInd, long num, long target_index);

template void Pool_epi8_t<5>::_process_ndhl1_triblock<NDH_DOUBLE_NDUAL>(int8_t *dh, long bias, long bound, 
                                                                    int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template void Pool_epi8_t<5>::_process_ndhl1_block<NDH_DOUBLE_NDUAL>(int8_t *dhi, int8_t *dhj, long ibias, long jbias, long ibound, long jbound, 
                                                                int32_t threshold, uint32_t **ptr_buffer, long *ptr_buffer_size, long *ptr_buffer_num);

template int Pool_epi8_t<5>::_naivedh_insert<NDH_DOUBLE_NDUAL, NDH_DEFAULT_DHDIM, NDH_L1_BLOCK, NDH_L2_BLOCK, NDH_M_BLOCK>(long target_index, double eta, long log_level, float *dual_vec, int32_t threshold, double target_length);

template class Pool_epi8_t<5>;
#endif

