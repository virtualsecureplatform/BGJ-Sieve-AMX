#include "../include/pool_epi8.h"
#include "../include/bgj_epi8.h"

#include <sys/time.h>

/// features: 
/// 1. provide a parameter to control if we stop the 
///     search when current solution is good enough
/// 2. two ways to reduce the lfiting time: more dual vectors 
///     or check the tail norm
/// 3. mblock stragety: if time of bucketing and search is 1 : 3
///     as in the bgj1 sieve, naive mblock strgety will waste 
///     half of the bucketing result. We may latter use a better 
///     mblock algorithm to divide the pool into many pieces and 
///     make the probability of each two pieces meet in the same
///     mblock roughly to be the same. The choice of stragetgy 
///     should be controlled by a marco.
/// 4. if we want to collect at least 85% of the close pairs, 
///     then each pair will be find 7 times on average. So we
///     will use a uid hash table to remove duplicate solutions.
/// 5. the dual hash dimension can be numbers that do not divide
///     by 8.
/// 6. bucket threshold are chosen such that size of bucket is 
///     slightly larget than sqrt(mblock_size), then we choose 
///     a number of buckets by experimental results to make most 
///     of the solution to be found.
///
/// assumptions:
/// 1. the lifted part is uniform in the dh_dim dimensional space modulo L_mid
/// 2. the dualhash of the lifted part, thus their addition or subtraction, is
///     uniform in the ndual dimensional space


// we allow larger number of dual vectors for ldh, 
// you should search all the "change with LDH_MAX_NDUAL"
// and change them by hand if LDH_MAX_NDUAL is not 128
#define LDH_DEFAULT_NDUAL 32
#define LDH_MAX_NDUAL 128
#define LDH_DEFAULT_DHDIM 24
#if LDH_DEFAULT_NDUAL % 32
#error number of dual vector must divided by 32
#endif

#define LDH_L1_BLOCK 256
#define LDH_L2_BLOCK 8192
#define LDH_M_BLOCK 1048576

#define LDH_DIAG_FIRST 1

#if 1
struct timeval _ldh_timer_start[MAX_NTHREADS], _ldh_timer_end[MAX_NTHREADS];
double _ldh_time_curr[MAX_NTHREADS];

#define TIMER_START do {                                                        \
        gettimeofday(&_ldh_timer_start[omp_get_thread_num()], NULL);            \
    } while (0)

#define TIMER_END do {                                                          \
        gettimeofday(&_ldh_timer_end[omp_get_thread_num()], NULL);              \
        _ldh_time_curr[omp_get_thread_num()] =                                                            \
            (_ldh_timer_end[omp_get_thread_num()].tv_sec-_ldh_timer_start[omp_get_thread_num()].tv_sec)+                    \
            (double)(_ldh_timer_end[omp_get_thread_num()].tv_usec-_ldh_timer_start[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_ldh_time_curr[omp_get_thread_num()])
#endif

// the implementation is in naivedh_epi8.cpp
int gen_dual_vec_list(float *dst, Lattice_QP *L, long log_level, long nlist, int rng_seed = -1);

// change with LDH_MAX_NDUAL
#include "../include/lsfdh_tables.hpp"

// probality that random lifted vector is of length^2 less than radius
// change with LDH_MAX_NDUAL
double _ldh_C(uint32_t ndual, int32_t radius) {
    constexpr int32_t bound32 = 371;
    constexpr int32_t bound64 =  624;
    constexpr int32_t bound96 = 868;
    constexpr int32_t bound128 = 1094;
    
    if (ndual == 32) {
        double left = (radius / 1000 > bound32) ? 1.0 : _ldh_C32[radius / 1000];
        double right = (radius / 1000 + 1 > bound32) ? 1.0 : _ldh_C32[radius / 1000 + 1];
        return left + (right - left) * (radius % 1000) / 1000.0;
    }
    if (ndual == 64) {
        double left = (radius / 1000 > bound64) ? 1.0 : _ldh_C64[radius / 1000];
        double right = (radius / 1000 + 1 > bound64) ? 1.0 : _ldh_C64[radius / 1000 + 1];
        return left + (right - left) * (radius % 1000) / 1000.0;
    }
    if (ndual == 96) {
        double left = (radius / 1000 > bound96) ? 1.0 : _ldh_C96[radius / 1000];
        double right = (radius / 1000 + 1 > bound96) ? 1.0 : _ldh_C96[radius / 1000 + 1];
        return left + (right - left) * (radius % 1000) / 1000.0;
    }
    if (ndual == 128) {
        double left = (radius / 1000 > bound128) ? 1.0 : _ldh_C128[radius / 1000];
        double right = (radius / 1000 + 1 > bound128) ? 1.0 : _ldh_C128[radius / 1000 + 1];
        return left + (right - left) * (radius % 1000) / 1000.0;
    }
    fprintf(stderr, "[Warning] _ldh_C: ndual = %d is not supported, you may use gen_lsfdh_Ctable() in tool.cpp to generate it\n", ndual);
    return 0.0;
}

// probality that two random lifted vector of length^2 less than radius has distance^2 less than threshold
// the input radius must be an output of _opt_ldh_bucket_radius with the same ndual
// change with LDH_MAX_NDUAL
double _ldh_W(uint32_t ndual, int32_t radius, int32_t threshold) {
    constexpr int32_t lbound32 = _LDH_W_LBOUND32;
    constexpr int32_t lbound64 = _LDH_W_LBOUND64;
    constexpr int32_t lbound96 = _LDH_W_LBOUND96;
    constexpr int32_t lbound128 = _LDH_W_LBOUND128;
    constexpr int32_t ubound32 = _LDH_W_UBOUND32;
    constexpr int32_t ubound64 = _LDH_W_UBOUND64;
    constexpr int32_t ubound96 = _LDH_W_UBOUND96;
    constexpr int32_t ubound128 = _LDH_W_UBOUND128;

    constexpr int32_t nr32 = (ubound32 - lbound32) / 1000 + 1;
    constexpr int32_t nr64 = (ubound64 - lbound64) / 1000 + 1;
    constexpr int32_t nr96 = (ubound96 - lbound96) / 1000 + 1;
    constexpr int32_t nr128 = (ubound128 - lbound128) / 1000 + 1;
    constexpr int32_t nc32 = ubound32 / 1000 + 1;
    constexpr int32_t nc64 = ubound64 / 1000 + 1;
    constexpr int32_t nc96 = ubound96 / 1000 + 1;
    constexpr int32_t nc128 = ubound128 / 1000 + 1;

    if (ndual == 32) {
        if (radius < lbound32) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too small, use %d instead\n", radius, lbound32);
            radius = lbound32;
        }
        if (radius >= ubound32) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too large, use %d instead\n", radius, ubound32);
            radius = ubound32 - 1;
        }
        if (threshold >= ubound32) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too large, use %d instead\n", threshold, ubound32-1);
            threshold = ubound32 - 1;
        }
        if (threshold < lbound32) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too small, use %d instead\n", threshold, lbound32);
            threshold = lbound32;
        }

        double ul = _ldh_W32[(radius-lbound32) / 1000][(threshold-lbound32) / 1000];
        double ur = _ldh_W32[(radius-lbound32) / 1000][(threshold-lbound32) / 1000 + 1];
        double dl = _ldh_W32[(radius-lbound32) / 1000 + 1][(threshold-lbound32) / 1000];
        double dr = _ldh_W32[(radius-lbound32) / 1000][(threshold-lbound32) / 1000 + 1];
        double u = ul + (ur - ul) * (threshold % 1000) / 1000.0;
        double d = dl + (dr - dl) * (threshold % 1000) / 1000.0;
        return u + (d - u) * (radius % 1000) / 1000.0;
    }
    if (ndual == 64) {
        if (radius < lbound64) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too small, use %d instead\n", radius, lbound64);
            radius = lbound64;
        }
        if (radius >= ubound64) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too large, use %d instead\n", radius, ubound64);
            radius = ubound64 - 1;
        }
        if (threshold >= ubound64) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too large, use %d instead\n", threshold, ubound64-1);
            threshold = ubound64 - 1;
        }
        if (threshold < lbound64) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too small, use %d instead\n", threshold, lbound64);
            threshold = lbound64;
        }

        double ul = _ldh_W64[(radius-lbound64) / 1000][(threshold-lbound64) / 1000];
        double ur = _ldh_W64[(radius-lbound64) / 1000][(threshold-lbound64) / 1000 + 1];
        double dl = _ldh_W64[(radius-lbound64) / 1000 + 1][(threshold-lbound64) / 1000];
        double dr = _ldh_W64[(radius-lbound64) / 1000][(threshold-lbound64) / 1000 + 1];
        double u = ul + (ur - ul) * (threshold % 1000) / 1000.0;
        double d = dl + (dr - dl) * (threshold % 1000) / 1000.0;
        return u + (d - u) * (radius % 1000) / 1000.0;
    }
    if (ndual == 96) {
        if (radius < lbound96) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too small, use %d instead\n", radius, lbound96);
            radius = lbound96;
        }
        if (radius >= ubound96) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too large, use %d instead\n", radius, ubound96);
            radius = ubound96 - 1;
        }
        if (threshold >= ubound96) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too large, use %d instead\n", threshold, ubound96-1);
            threshold = ubound96 - 1;
        }
        if (threshold < lbound96) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too small, use %d instead\n", threshold, lbound96);
            threshold = lbound96;
        }

        double ul = _ldh_W96[(radius-lbound96) / 1000][(threshold-lbound96) / 1000];
        double ur = _ldh_W96[(radius-lbound96) / 1000][(threshold-lbound96) / 1000 + 1];
        double dl = _ldh_W96[(radius-lbound96) / 1000 + 1][(threshold-lbound96) / 1000];
        double dr = _ldh_W96[(radius-lbound96) / 1000][(threshold-lbound96) / 1000 + 1];
        double u = ul + (ur - ul) * (threshold % 1000) / 1000.0;
        double d = dl + (dr - dl) * (threshold % 1000) / 1000.0;
        return u + (d - u) * (radius % 1000) / 1000.0;
    }
    if (ndual == 128) {
        if (radius < lbound128) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too small, use %d instead\n", radius, lbound128);
            radius = lbound128;
        }
        if (radius >= ubound128) {
            fprintf(stderr, "[Warning] _ldh_W: radius = %d is too large, use %d instead\n", radius, ubound128);
            radius = ubound128 - 1;
        }
        if (threshold >= ubound128) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too large, use %d instead\n", threshold, ubound128-1);
            threshold = ubound128 - 1;
        }
        if (threshold < lbound128) {
            fprintf(stderr, "[Warning] _ldh_W: threshold = %d is too small, use %d instead\n", threshold, lbound128);
            threshold = lbound128;
        }

        double ul = _ldh_W128[(radius-lbound128) / 1000][(threshold-lbound128) / 1000];
        double ur = _ldh_W128[(radius-lbound128) / 1000][(threshold-lbound128) / 1000 + 1];
        double dl = _ldh_W128[(radius-lbound128) / 1000 + 1][(threshold-lbound128) / 1000];
        double dr = _ldh_W128[(radius-lbound128) / 1000][(threshold-lbound128) / 1000 + 1];
        double u = ul + (ur - ul) * (threshold % 1000) / 1000.0;
        double d = dl + (dr - dl) * (threshold % 1000) / 1000.0;
        return u + (d - u) * (radius % 1000) / 1000.0;
    }

    fprintf(stderr, "[Warning] _ldh_W: ndual = %d is not supported, you may use gen_lsfdh_Wtable() in tool.cpp to generate it\n", ndual);
    return 0.0;
}

// type 0: full search
// type 1: half part full search + np search
// type 2: np search only
template <uint32_t nb>
int32_t Pool_epi8_t<nb>::_opt_ldh_bucket_radius(long total_num, uint32_t ndual, long type) {
    constexpr long min_total_num = 1000;
    constexpr long max_total_num = 20000000000;     // larger than 2.0 * (4/3)^(160/2)
    if (total_num < min_total_num) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_opt_ldh_bucket_radius: total_num(%ld) too small, use %ld instead\n", nb, total_num, min_total_num);
        total_num = min_total_num;
    }
    if (total_num > max_total_num) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_opt_ldh_bucket_radius: total_num(%ld) too large, use %ld instead\n", nb, total_num, max_total_num);
        total_num = max_total_num;
    }

    // TODO
    constexpr double type0_opt_ratio = 3.0;
    constexpr double type1_opt_ratio = 2.5;
    constexpr double type2_opt_ratio = 2.0;

    double target_C;
    if (type != 0 && type != 1 && type != 2) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::_opt_ldh_bucket_radius: invalid type(%ld), use type 0 instead\n", nb, type);
    }
    if (type == 2) {
        target_C = sqrt(type0_opt_ratio / total_num / 0.5);
    } else if (type == 1) {
        target_C = sqrt(type1_opt_ratio / total_num / 0.75);
    } else {
        target_C = sqrt(type2_opt_ratio / total_num / 1.0);
    }

    int32_t left = 0, right = ndual * 128 * 128 / 1000;
    while (left < right - 1) {
        int32_t mid = (left + right) / 2;
        double C = _ldh_C(ndual, mid * 1000);
        if (C < target_C) {
            left = mid;
        } else if (C > target_C) {
            right = mid;
        } else {
            return mid * 1000;
        }
    }
    return left * 1000 + 1000 * (target_C - _ldh_C(ndual, left * 1000)) / (_ldh_C(ndual, right * 1000) - _ldh_C(ndual, left * 1000));

    return 0;
}

template <uint32_t nb>
void Pool_epi8_t<nb>::_opt_ldh_threshold(float *dual_vec, uint32_t &ndual, int32_t &threshold, double &tail_alpha_bound,
                        double dual_exp_len, const double *tail_alpha_prob_list, Lattice_QP *L, double ratio, long log_level) {
    #define SHOW_LDH_OPTTH_INFO 0
    const long N = 10000;
    // change with LDH_MAX_NDUAL
    // CHANGE_WITH_ARCH
    // the time/cpusec for each pnorm/nnorm computation
    const double dp_cost[LDH_MAX_NDUAL/32] = {1.0 / 1073741824.0, 1.5 / 1073741824.0, 2.0 / 1073741824.0, 2.5 / 1073741824.0};
    // the time/cpusec for each tail norm check
    const double tail_dp_cost = 10.0 / 1073741824.0;
    // the time/cpusec for each lift
    const double lift_cost = 100.0 / 1073741824.0;

    DGS1d R;
    const long dh_dim = L->NumRows();
    const long dh_dim8 = _CEIL8(dh_dim);
    float *_dual_vec = (float *) NEW_VEC(LDH_MAX_NDUAL * dh_dim8, sizeof(float));
    long count[2560];
    __attribute__ ((aligned (32))) float tmp[256] = {};

    // change with LDH_MAX_NDUAL
    const int32_t _threshold_lb[4] = {_LDH_W_LBOUND32, _LDH_W_LBOUND64, _LDH_W_LBOUND96, _LDH_W_LBOUND128};
    const int32_t _threshold_ub[4] = {_LDH_W_UBOUND32, _LDH_W_UBOUND64, _LDH_W_UBOUND96, _LDH_W_UBOUND128};

    // the time/cpusec for left progressive sieve of dim = CSD
    double base_time = pow(2, CSD * 0.332 - 23);    // CHANGE_WITH_ARCH
    double max_speed = 0.0;
    int max_speed_updated = 0;
    for (uint32_t _ndual = 32; _ndual <= LDH_MAX_NDUAL; _ndual += 32) {
        gen_dual_vec_list(_dual_vec, L, log_level, _ndual);
        memset(count, 0, sizeof(count));
        for (long ind = 0; ind < N; ind++) {
            for (long l = 0; l < dh_dim; l++) tmp[l] = R.discrete_gaussian(0.0, 1048576.0);
            float x = dot_avx2(tmp, tmp, dh_dim);
            x = dual_exp_len / sqrt(x);
            mul_avx2(tmp, x, dh_dim);
            int32_t res = 0;
            for (long i = 0; i < _ndual; i++) {
                float x = dot_avx2(tmp, _dual_vec + i * dh_dim8, dh_dim);
                int32_t xi = x;
                int8_t x8 = xi;
                res += (int)x8 * (int)x8;
            }
            count[(res + 500)/1000]++;
        }
        for (long i = 1; i < 2560; i++) {
            count[i] += count[i-1];
        }
        int32_t _bucket_radius;
        long _total_num = (num_vec < 2 * LDH_M_BLOCK) ? num_vec : 2 * LDH_M_BLOCK;
        _bucket_radius = _opt_ldh_bucket_radius(_total_num, _ndual, ((num_vec / (2 * LDH_M_BLOCK)) < 2) ? (num_vec / (2 * LDH_M_BLOCK)) : 2);
        #if SHOW_LDH_OPTTH_INFO
        int32_t best_threshold = 0;     
        double best_tail_alpha = 0.0;   
        double best_speed = 0.0;        
        double best_over_search = 0.0; 
        #endif 
        for (int32_t _threshold = _threshold_lb[_ndual/32-1]; _threshold < _threshold_ub[_ndual/32-1]; _threshold += 1000) {
            double _pass_prob0 = count[_threshold/1000] / (double)N;
            double bucket_size = 2.0 * _ldh_C(_ndual, _bucket_radius) * _total_num;
            double one_bucket_dp = 2.0 * _total_num + bucket_size * bucket_size / 2.0;
            double one_bucket_sol = bucket_size * bucket_size / 2.0 * _ldh_W(_ndual, _bucket_radius, _threshold);
            double dp_per_sol = one_bucket_dp / one_bucket_sol;
            double num_total_sol = _total_num * _total_num * _ldh_C(_ndual, _threshold);
            for (double _over_search = 0.25; _over_search < 3.01; _over_search *= 1.088) {
                double _pass_prob1 = 1.0 - pow(2.718281828, -_over_search);
                double _time0 = _over_search * dp_per_sol * num_total_sol * dp_cost[_ndual/32-1];
                if (isnan(_time0)) _time0 = 0.0;
                double num_pass = num_total_sol * _pass_prob1;
                for (double _tail_alpha = 0.0; _tail_alpha < 0.495; _tail_alpha += 0.01) {
                    double _pass_prob2 = tail_alpha_prob_list[(int)round(_tail_alpha * 100)];
                    double _pass_prob = _pass_prob0 * _pass_prob1 * _pass_prob2;
                    double _time1 = num_pass * tail_dp_cost + lift_cost * num_pass * (0.25 * pow(1-_tail_alpha * _tail_alpha, CSD * 0.5));
                    double _speed = _pass_prob / (base_time + _time0 + _time1 + 1e-30);
                    if (num_total_sol == 0.0) _speed = 0.0;
                    #if SHOW_LDH_OPTTH_INFO
                    if (_speed > best_speed) {          
                        best_speed = _speed;            
                        best_tail_alpha = _tail_alpha;  
                        best_over_search = _over_search;
                        best_threshold = _threshold;    
                    }
                    #endif                                   
                    if (_speed > max_speed) {
                        max_speed_updated = 1;
                        max_speed = _speed;
                        ndual = _ndual;
                        threshold = _threshold;
                        tail_alpha_bound = _tail_alpha;
                    }
                }
            }
        }
        #if SHOW_LDH_OPTTH_INFO
        do {
            double _pass_prob0 = count[best_threshold/1000] / (double)N;
            double bucket_size = 2.0 * _ldh_C(_ndual, _bucket_radius) * _total_num;
            double one_bucket_dp = 2.0 * _total_num + bucket_size * bucket_size / 2.0;
            double one_bucket_sol = bucket_size * bucket_size / 2.0 * _ldh_W(_ndual, _bucket_radius, best_threshold);
            double dp_per_sol = one_bucket_dp / one_bucket_sol;
            double num_total_sol = _total_num * _total_num * _ldh_C(_ndual, best_threshold);
            double _pass_prob1 = 1.0 - pow(2.718281828, -best_over_search);
            double _time0 = best_over_search * dp_per_sol * num_total_sol * dp_cost[_ndual/32-1];
            if (isnan(_time0)) _time0 = 0.0;
            double num_pass = num_total_sol * _pass_prob1;
            double _pass_prob2 = tail_alpha_prob_list[(int)round(best_tail_alpha * 100)];
            double _pass_prob = _pass_prob0 * _pass_prob1 * _pass_prob2;
            double _time1 = num_pass * tail_dp_cost + lift_cost * num_pass * (0.25 * pow(1-best_tail_alpha * best_tail_alpha, CSD * 0.5));
            double _speed = _pass_prob / (base_time + _time0 + _time1 + 1e-30);
            if (num_total_sol == 0.0) _speed = 0.0;
            printf("\n_ndual = %u, _bucket_radius = %d(%f), _threshold = %d(r:%f/b:%f), dp/sol = %f(total_sol: %f), prob_pass_th = %f\n", 
                    _ndual, _bucket_radius, 2.0 * _ldh_C(_ndual, _bucket_radius), best_threshold, _ldh_C(_ndual, best_threshold),
                    _ldh_W(_ndual, _bucket_radius, best_threshold), dp_per_sol, num_total_sol, _pass_prob0);
            printf("_over_search = %.2f, _prob_searched = %.3f, _tail_alpha_bound = %.2f, _prob_pass_alpha = %f\n",
                    best_over_search, _pass_prob1, best_tail_alpha, _pass_prob2);
            printf("time_search = %fs, time_tail = %fs, base_time = %fs, time = %fs, prob_pass = %f, speed = %f\n",
                    _time0, _time1, base_time, _time0 + _time1 + base_time, _pass_prob, _speed);
        } while (0);
        #endif
        if (max_speed_updated) {
            copy_avx2(dual_vec, _dual_vec, dh_dim8 * _ndual);
        }
    }
    FREE_VEC(_dual_vec);
}


/// the public interface lsfdh_insert
/// in this function we do the following things:
/// 1. compute the basic information of the lifting according to the basis and the pool
/// 2. choose the optimal ndual, threshold and tail bound (if needed) and the corresponding 
///     dual vector list.
/// 3. choose the mblock stragety, if needed.
/// 4. call the lifting kernel _lsfdh_insert.
template <uint32_t nb>
int Pool_epi8_t<nb>::lsfdh_insert(long target_index, double eta, long log_level, double target_length, double stop_ratio) {
    /////// basic information of the lifting ///////
    const double tail_gh = sqrt(gh2);
    const double dual_gh = basis->gh(index_l - LDH_DEFAULT_DHDIM, index_l);
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
    double tail_alpha_prob_list[64];
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
                        target_length * sqrt(LDH_DEFAULT_DHDIM) / sqrt(index_r - target_index) :
                        lift_exp_length * sqrt(LDH_DEFAULT_DHDIM) / sqrt(index_l - target_index);
    
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

    /////// compute ndual, threshold, tail bound, and the dual vec ///////
    float *dual_vec = (float *) NEW_VEC(LDH_MAX_NDUAL * _CEIL8(LDH_DEFAULT_DHDIM), sizeof(float));
    uint32_t ndual;
    int32_t threshold, tail_bound;
    double tail_alpha_bound;
    Lattice_QP *b_mid = basis->b_loc_QP(index_l - LDH_DEFAULT_DHDIM, index_l);
    _opt_ldh_threshold(dual_vec, ndual, threshold, tail_alpha_bound, 
                        dual_exp_length, tail_alpha_prob_list, b_mid, 0.9, log_level);
    delete b_mid;
    tail_bound = pow(tail_exp_ratio * tail_gh * sqrt(2.0 - 2.0 * tail_alpha_bound) * _ratio, 2.0);
    if (log_level >= 0) printf("ndual = %d, threshold = %d, tail_alpha_bound = %.2f, tail_bound = %d\n", 
                                ndual, threshold, tail_alpha_bound, tail_bound); 
    
    /////// choose mblock stragety ///////
    // nothing to do here now

    /////// call the lifting kernel ///////
    int ret = 0;
    double stop_length = unique_target ? stop_ratio * target_length : stop_ratio * exp_length;
    if (ndual == 32) {
        ret = _lsfdh_insert<32, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(target_index, eta, log_level, dual_vec, threshold, tail_bound, target_length, stop_length);
    } else if (ndual == 64) {
        ret = _lsfdh_insert<64, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(target_index, eta, log_level, dual_vec, threshold, tail_bound, target_length, stop_length);
    } else if (ndual == 96) {
        ret = _lsfdh_insert<96, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(target_index, eta, log_level, dual_vec, threshold, tail_bound, target_length, stop_length);
    } else {
        // change with LDH_MAX_NDUAL
        ret = _lsfdh_insert<128, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(target_index, eta, log_level, dual_vec, threshold, tail_bound, target_length, stop_length);
    }

    FREE_VEC(dual_vec);
    return ret;
}

template <uint32_t nb>
template <uint32_t ndual, uint32_t dh_dim, uint32_t mblock>
int Pool_epi8_t<nb>::_lsfdh_insert(long target_index, double eta, long log_level, 
                                    float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length) {
    printf("_lsfdh_insert<%u, %u, %u>(target_index = %ld, eta = %.2f, log_level = %ld, dual_vec = %p, threshold = %d, tail_bound = %d, target_length = %.2f, stop_length = %.2f) called\n",
            ndual, dh_dim, mblock, target_index, eta, log_level, dual_vec, threshold, tail_bound, target_length, stop_length);
    printf("_lsfdh_insert is not implemented yet\n");
    // TODO
    return 0;
}


#if COMPILE_POOL_EPI8_96
// change with LDH_MAX_NDUAL
template int Pool_epi8_t<3>::_lsfdh_insert<32, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<3>::_lsfdh_insert<64, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<3>::_lsfdh_insert<96, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<3>::_lsfdh_insert<128, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template class Pool_epi8_t<3>;
#endif

#if COMPILE_POOL_EPI8_128
// change with LDH_MAX_NDUAL
template int Pool_epi8_t<4>::_lsfdh_insert<32, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<4>::_lsfdh_insert<64, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<4>::_lsfdh_insert<96, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<4>::_lsfdh_insert<128, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template class Pool_epi8_t<4>;
#endif

#if COMPILE_POOL_EPI8_160
// change with LDH_MAX_NDUAL
template int Pool_epi8_t<5>::_lsfdh_insert<32, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<5>::_lsfdh_insert<64, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<5>::_lsfdh_insert<96, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template int Pool_epi8_t<5>::_lsfdh_insert<128, LDH_DEFAULT_DHDIM, LDH_M_BLOCK>(long target_index, double eta, long log_level, 
                            float *dual_vec, int32_t threshold, int32_t tail_bound, double target_length, double stop_length);
template class Pool_epi8_t<5>;
#endif
