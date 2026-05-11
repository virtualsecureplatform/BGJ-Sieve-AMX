#include "../include/svp.h"
#include "../include/pool.h"
#include "../include/lattice.h"
#include "../include/pool_epi8.h"
#include "../include/config.h"
#if defined(HAVE_CUDA)
#include "../include/bgj_cuda.h"
#endif
#include <sys/time.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <unistd.h>

#if defined(__AMX_INT8__)
#include "../include/bgj_amx.h"
#endif

/* svp kernels of bkz algorithm. */

static int svp_cuda_bgj1_enabled() {
#if defined(HAVE_CUDA)
    const char *env = std::getenv("BGJ_SVP_CUDA");
    return env && env[0] && env[0] != '0';
#else
    return 0;
#endif
}

template <uint32_t nb>
static int svp_bgj1_sieve(Pool_epi8_t<nb> &p, long log_level, long lps_auto_adj) {
#if defined(HAVE_CUDA)
    if (svp_cuda_bgj1_enabled()) return p.bgj1_Sieve_cuda(log_level, lps_auto_adj);
#endif
    return p.bgj1_Sieve(log_level, lps_auto_adj);
}

template <uint32_t nb>
static int svp_bgj2_sieve(Pool_epi8_t<nb> &p, long log_level, long lps_auto_adj) {
#if defined(HAVE_CUDA)
    if (svp_cuda_bgj1_enabled()) return p.bgj2_Sieve_cuda(log_level, lps_auto_adj);
#endif
    return p.bgj2_Sieve(log_level, lps_auto_adj);
}

template <uint32_t nb>
static int svp_bgj3_sieve(Pool_epi8_t<nb> &p, long log_level, long lps_auto_adj) {
#if defined(HAVE_CUDA)
    if (svp_cuda_bgj1_enabled()) return p.bgj3_Sieve_cuda(log_level, lps_auto_adj);
#endif
    return p.bgj3_Sieve(log_level, lps_auto_adj);
}

static int svp_pump_profile_enabled()
{
    const char *env = std::getenv("BGJ_PUMP_PROFILE");
    return env && env[0] && env[0] != '0';
}

static double svp_profile_now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int svp_lsh_best_at_or_below(long dimension, double stop_length, double *best_length)
{
    if (best_length) *best_length = 0.0;
    if (stop_length <= 0.0 || dimension <= 0) return 0;
    double length = 0.0;
    long record_dimension = 0;
    if (!bgj_lsh_best_solution_get(&length, NULL, dimension, &record_dimension)) return 0;
    if (record_dimension != dimension || length <= 0.0) return 0;
    if (best_length) *best_length = length;
    return length <= stop_length;
}

#define SVP_PROFILE_DO(acc, stmt) do {                             \
        if (pump_profile) {                                        \
            double __svp_profile_t0 = svp_profile_now();           \
            stmt;                                                  \
            acc += svp_profile_now() - __svp_profile_t0;           \
        } else {                                                   \
            stmt;                                                  \
        }                                                          \
    } while (0)

#define SVP_PROFILE_ASSIGN(acc, lhs, expr) do {                    \
        if (pump_profile) {                                        \
            double __svp_profile_t0 = svp_profile_now();           \
            lhs = (expr);                                          \
            acc += svp_profile_now() - __svp_profile_t0;           \
        } else {                                                   \
            lhs = (expr);                                          \
        }                                                          \
    } while (0)


/** svp kernel based on 3-Sieve.
 *  \param msd maximal sieving dimension
 *  \param f   maximal depth of dim for free
 *  \param ni  maximal number of insertion
 *  \param ne  maximal number of extend_left after the main sieve
 *  \param ns  maximal number of sieving after the main sieve
 */
void __pump_red(Lattice_QP *L, long num_threads, double eta, long msd, long f, long ni, long ne, long ns){
    long n = L->NumRows();
    if (min(msd, n) < 40){
        std::cerr << "[Warning] sieving dimension too small, may get stuck, aborted.\nYou can use Enum based algorithms instead.\n";
        return;
    }
    struct timeval start, end;
	gettimeofday(&start, NULL);
    double dPot = L->Pot();
    ni -= max(0, min(msd + f - n, f));
    ns -= max(0, min(msd + f - n, f));
    ne -= max(0, min(msd + f - n, f));
    ne = max(0, ne);

    bool GET_STUCKED = true;
    while (GET_STUCKED){
        GET_STUCKED = false;
        if (n > 110){
            L->tail_shuffle(80);
            L->compute_gso_QP(max(n-76, 0));
        } else if (n > 100) {
            L->tail_shuffle(74);
            L->compute_gso_QP(max(n-76, 0));
        } else if (n > 90){
            L->tail_shuffle(68);
            L->compute_gso_QP(max(n-68, 0));
        } else{
            L->tail_shuffle(60);
            L->compute_gso_QP(max(n-60, 0));
        }
        
        L->set_gso_status(GSO_COMPUTED_QP);

        Pool pool(L);
        long max_pool_size = (long)(pow(4./3., min(n, msd) * 0.5) * 3.2) + 1;
        pool.set_num_threads(num_threads);
        pool.set_MSD(min(n, msd));
        pool.set_max_pool_size(max_pool_size);
        pool.set_sieving_context(n-40, n);
        pool.gaussian_sampling((long)(pow(4./3., pool.CSD*0.5)*3.2));
        three_Sieve_params p;
        p.alpha = 0.315;

        //main sieve
        pool.three_Sieve(p, 0, 10000000);
        while ((pool.CSD < msd)&&(pool.CSD < n)){
            pool.extend_left();
            pool.gaussian_sampling((long)(pow(4./3., pool.CSD*0.5) * 3.2));
            if (pool.three_Sieve(p, 0, 15000) == THREE_SIEVE_GET_STUCKED){
                GET_STUCKED = true;
                break;
            }
        }

        //insertions
        long ind = max(pool.index_l - f, 0);
        while ((ind < n - 40)&&(!GET_STUCKED)){
            pool.insert(ind, eta);

            ni--;
            if (ni <= 0) break;
            ind++;
            
            if (ne > 0){
                ne--;
                pool.extend_left();
                ind--;
            }
            if (ns > 0){
                ns--;
                pool.shrink((long)(pow(4./3., pool.CSD * 0.5) * 3.2));
                pool.tail_LLL(0.99, 24);
                if (pool.three_Sieve(p, 0, 70000) == THREE_SIEVE_GET_STUCKED){
                    GET_STUCKED = true;
                    L->LLL_QP(0.99);
                    L->compute_gso_QP();
                    break;
                }
            }
        }
    }

    /*
        L->tail_shuffle(60);
        L->compute_gso_QP(max(n-60, 0));
        L->set_gso_status(GSO_COMPUTED_QP);


        Pool pool(L);
        long max_pool_size = (long)(pow(4./3., min(n, msd) * 0.5) * 3.2) + 1;
        pool.set_num_threads(num_threads);
        pool.set_MSD(min(n, msd));
        pool.set_max_pool_size(max_pool_size);
        pool.set_sieving_context(n-40, n);
        pool.gaussian_sampling((long)(pow(4./3., pool.CSD*0.5)*3.2));
        three_Sieve_params p;
        p.alpha = 0.315;

        //main sieve
        pool.three_Sieve(p, 0);
        while ((pool.CSD < msd)&&(pool.CSD < n)){
            pool.extend_left();
            pool.gaussian_sampling((long)(pow(4./3., pool.CSD*0.5) * 3.2));
            pool.three_Sieve(p, 0, 100000);
        }

        //insertions
        long ind = max(pool.index_l - f, 0);
        ni -= max(0, min(msd + f - n, f));
        ns -= max(0, min(msd + f - n, f));
        ne -= max(0, min(msd + f - n, f));
        ne = max(0, ne);
        while (ind < n - 40){
            pool.insert(ind, eta);

            ni--;
            if (ni <= 0) break;
            ind++;
            
            if (ne > 0){
                ne--;
                pool.extend_left();
                ind--;
            }
            if (ns > 0){
                ns--;
                pool.shrink((long)(pow(4./3., pool.CSD * 0.5) * 3.2));
                pool.tail_LLL(0.99, 24);
                pool.three_Sieve(p, 0, 100000);
            }
        }
    */
    L->LLL_QP(0.99);
    for (long i = 0; i < n; i++){
        dPot -= (n-i) * log2(L->get_B().hi[i]);
    }

    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    std::cout << "__pump_red_done: runtime = " << total_time << "s, num_threads = " << num_threads;
    std::cout << ", total_cost = " << (total_time * num_threads) << "Ts, dPot = " << dPot << "\n";
}

/** svp kernel based on bgjf_epi8 & dual hash lift disabled.
 *  \param msd maximal sieving dimension
 *  \param f   maximal depth of dim for free
 *  \param ni  maximal number of insertion
 *  \param ne  maximal number of extend_left after the main sieve
 *  \param ns  maximal number of sieving after the main sieve
 */
void __pump_red_epi8(Lattice_QP *L, long num_threads, double eta, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first, long minsd) {
    if (min(msd, L->NumRows()) < minsd){
        fprintf(stderr, "[Warning] sieving dimension too small, may get stuck, nothing done.\nYou can use Enum based algorithms instead.\n");
        return;
    }
    if (msd > COMPILE_POOL_EPI8_MAX_DIM) {
        fprintf(stderr, "[Warning] sieving dimension(%ld) >= %d, nothing done\n"
                "change COMPILE_POOL_EPI8_** in include/config.h to enable higher sieving dimension.\n",
                msd, COMPILE_POOL_EPI8_MAX_DIM);
        return;
    }
    
    struct timeval start, end;
    double dPot;
	if (log_level >= 2) {
        gettimeofday(&start, NULL);
        dPot = L->Pot();
    }
    const long n = L->NumRows();
    ni -= max(0, min(msd + f - n, f));
    ns -= max(0, min(msd + f - n, f));
    ne -= max(0, min(msd + f - n, f));
    ne = max(0, ne);

    constexpr double pool_size_ratio = 3.2;

    const int pump_profile = svp_pump_profile_enabled();
    static long pump_profile_seq = 0;
    const long profile_seq = pump_profile ? ++pump_profile_seq : 0;
    const int profile_cuda_bgj1 = svp_cuda_bgj1_enabled();
    const double profile_total_start = pump_profile ? svp_profile_now() : 0.0;
    double profile_shuffle_gso = 0.0;
    double profile_pool_setup = 0.0;
    double profile_sampling = 0.0;
    double profile_extend = 0.0;
    double profile_sieve_bgj1 = 0.0;
    double profile_sieve_bgj2 = 0.0;
    double profile_sieve_bgj3 = 0.0;
    double profile_insert = 0.0;
    double profile_tail_lll = 0.0;
    double profile_shrink = 0.0;
    double profile_final_lll = 0.0;
    long profile_bgj1_count = 0;
    long profile_bgj2_count = 0;
    long profile_bgj3_count = 0;
    long profile_insert_count = 0;
    long profile_tail_lll_count = 0;
    const long profile_orig_minsd = minsd;

    bool GET_STUCKED = shuffle_first;
    long num_stuck = 0;
    do {
        if (GET_STUCKED) {
            if (num_stuck == 0 || num_stuck == 3) {
                long shuf_dim = (n > 90) ? (0.6 * n + 14) : (n - 22);
                if (pump_profile) {
                    double __svp_profile_t0 = svp_profile_now();
                    L->tail_shuffle(shuf_dim);
                    L->compute_gso_QP(n-shuf_dim);
                    L->set_gso_status(GSO_COMPUTED_QP);
                    profile_shuffle_gso += svp_profile_now() - __svp_profile_t0;
                } else {
                    L->tail_shuffle(shuf_dim);
                    L->compute_gso_QP(n-shuf_dim);
                    L->set_gso_status(GSO_COMPUTED_QP);
                }
                if (num_stuck == 3) {
                    fprintf(stderr, "[Warning] __pump_red_epi8: get stucked, try again after shuffle.\n");
                }
            } else if (num_stuck < 3) {
                minsd += 10;
                if (minsd > n) minsd = n;
                fprintf(stderr, "[Warning] __pump_red_epi8: get stucked, try again with larger start sieving dim\n");
            }            
            GET_STUCKED = false;
        }

        long max_pool_size = (long)(pow(4./3., min(n, msd) * 0.5) * pool_size_ratio) + 1;
        if (msd <= 96) {
            Pool_epi8_t<3> p(L);
            p.set_num_threads(num_threads);
            #define __PUMP_RED_EPI8_ONE_TRY do {                                                                \
                double __svp_profile_t0 = pump_profile ? svp_profile_now() : 0.0;                                \
                long minps = (long)(pow(4./3., minsd * 0.5) * pool_size_ratio);                                 \
                minps *= 3;                                                                                     \
                p.set_max_pool_size((max_pool_size > minps) ? max_pool_size : minps);                           \
                p.set_sieving_context(n-minsd, n);                                                              \
                if (pump_profile) profile_pool_setup += svp_profile_now() - __svp_profile_t0;                    \
                int __svp_sampling_ret = 0;                                                                      \
                SVP_PROFILE_ASSIGN(profile_sampling, __svp_sampling_ret, p.sampling(minps));                     \
                if (__svp_sampling_ret == -1) return;                                                           \
                                                                                                                \
                int ret = 0;                                                                                    \
                SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level - 3, 1));               \
                profile_bgj1_count++;                                                                           \
                num_stuck = ret;                                                                                \
                while ((p.CSD < msd) && (p.CSD < n)) {                                                          \
                    SVP_PROFILE_DO(profile_extend, p.extend_left());                                             \
                    long target_num_vec = (long) (pow(4./3., p.CSD * 0.5) * pool_size_ratio);                   \
                    if (target_num_vec > p.num_vec + p.num_empty) p.num_empty = target_num_vec - p.num_vec;     \
                    if (p.CSD > 92) {                                                                           \
                        SVP_PROFILE_ASSIGN(profile_sieve_bgj3, ret, svp_bgj3_sieve(p, log_level-3, 1));         \
                        profile_bgj3_count++;                                                                   \
                    } else if (p.CSD > 80) {                                                                    \
                        SVP_PROFILE_ASSIGN(profile_sieve_bgj2, ret, svp_bgj2_sieve(p, log_level-3, 1));         \
                        profile_bgj2_count++;                                                                   \
                    } else {                                                                                    \
                        SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level-3, 1));         \
                        profile_bgj1_count++;                                                                   \
                    }                                                                                           \
                    if (ret) {                                                                                  \
                        num_stuck++;                                                                            \
                        GET_STUCKED = true;                                                                     \
                        break;                                                                                  \
                    }                                                                                           \
                }                                                                                               \
                                                                                                                \
                long ind = max(p.index_l - f, 0);                                                               \
                while ((ind < n - minsd) && (!GET_STUCKED)) {                                                   \
                    SVP_PROFILE_DO(profile_insert, p.insert(ind, eta));                                         \
                    profile_insert_count++;                                                                     \
                    int tlll_dim = 32;                                                                          \
                    for (long i = 32; i < p.CSD - 1; i++) {                                                     \
                        if (L->get_B().hi[p.CSD-i+p.index_l] <  0.49 * L->get_B().hi[p.CSD-i-1+p.index_l]) {    \
                            tlll_dim = i;                                                                       \
                        }                                                                                       \
                    }                                                                                           \
                    SVP_PROFILE_DO(profile_tail_lll, p.tail_LLL(0.99, p.CSD));                                  \
                    profile_tail_lll_count++;                                                                   \
                                                                                                                \
                    ni--;                                                                                       \
                    if (ni <= 0) break;                                                                         \
                    ind++;                                                                                      \
                                                                                                                \
                    if (ne > 0) {                                                                               \
                        ne--;                                                                                   \
                        SVP_PROFILE_DO(profile_extend, p.extend_left());                                        \
                        ind--;                                                                                  \
                    }                                                                                           \
                    if (ns > 0) {                                                                               \
                        ns--;                                                                                   \
                        if ((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio) < p.num_vec) {                    \
                            SVP_PROFILE_DO(profile_shrink, p.shrink((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio))); \
                        }                                                                                       \
                        if (p.CSD > 102) {                                                                      \
                            SVP_PROFILE_ASSIGN(profile_sieve_bgj3, ret, svp_bgj3_sieve(p, log_level-3, 1));     \
                            profile_bgj3_count++;                                                               \
                        } else if (p.CSD > 90) {                                                                \
                            SVP_PROFILE_ASSIGN(profile_sieve_bgj2, ret, svp_bgj2_sieve(p, log_level-3, 1));     \
                            profile_bgj2_count++;                                                               \
                        } else {                                                                                \
                            SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level-3, 1));     \
                            profile_bgj1_count++;                                                               \
                        }                                                                                       \
                        if (ret) {                                                                              \
                            num_stuck++;                                                                        \
                            GET_STUCKED = true;                                                                 \
                            break;                                                                              \
                        }                                                                                       \
                    }                                                                                           \
                }                                                                                               \
            } while (0)

            __PUMP_RED_EPI8_ONE_TRY;
        } else if (msd <= 128) {
            #if COMPILE_POOL_EPI8_128
            Pool_epi8_t<4> p(L);
            p.set_num_threads(num_threads);
            __PUMP_RED_EPI8_ONE_TRY;
            #endif
        } else if (msd <= 160) {
            #if COMPILE_POOL_EPI8_160
            Pool_epi8_t<5> p(L);
            p.set_num_threads(num_threads);
            __PUMP_RED_EPI8_ONE_TRY;
            #endif
        } else if (msd <= 192) {
            #if COMPILE_POOL_EPI8_192
            Pool_epi8_t<6> p(L);
            p.set_num_threads(num_threads);
            __PUMP_RED_EPI8_ONE_TRY;
            #endif
        } else {
            #if COMPILE_POOL_EPI8_224
            Pool_epi8_t<7> p(L);
            p.set_num_threads(num_threads);
            __PUMP_RED_EPI8_ONE_TRY;
            #endif
        }
    } while (GET_STUCKED && num_stuck <= 3);

    SVP_PROFILE_DO(profile_final_lll, L->LLL_QP(0.99));
    if (pump_profile) {
        const double profile_total = svp_profile_now() - profile_total_start;
        fprintf(stdout,
                "pump_profile: seq=%ld n=%ld msd=%ld f=%ld minsd=%ld cuda_bgj1=%d "
                "total=%.6fs shuffle_gso=%.6fs setup=%.6fs sampling=%.6fs "
                "extend=%.6fs bgj1=%.6fs/%ld bgj2=%.6fs/%ld bgj3=%.6fs/%ld "
                "insert=%.6fs/%ld tail_lll=%.6fs/%ld shrink=%.6fs final_lll=%.6fs\n",
                profile_seq, n, msd, f, profile_orig_minsd, profile_cuda_bgj1,
                profile_total, profile_shuffle_gso, profile_pool_setup, profile_sampling,
                profile_extend, profile_sieve_bgj1, profile_bgj1_count,
                profile_sieve_bgj2, profile_bgj2_count,
                profile_sieve_bgj3, profile_bgj3_count,
                profile_insert, profile_insert_count,
                profile_tail_lll, profile_tail_lll_count,
                profile_shrink, profile_final_lll);
        fflush(stdout);
    }
    if (log_level < 2) return;
    for (long i = 0; i < n; i++){
        dPot -= (n-i) * log2(L->get_B().hi[i]);
    }

    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    fprintf(stdout, "__pump_red_done: runtime = %fs, num_threads = %ld, total_cost = %fTs, dPot = %f\n",
             total_time, num_threads, (total_time * num_threads), dPot);
}

void __lsh_pump_red_epi8(Lattice_QP *L, long num_threads, double eta, double qratio, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first, long minsd, double stop_length) {
    if (min(msd, L->NumRows()) < minsd){
        fprintf(stderr, "[Warning] sieving dimension too small, may get stuck, nothing done.\nYou can use Enum based algorithms instead.\n");
        return;
    }
    if (msd > 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32) {
        fprintf(stderr, "[Warning] sieving dimension(%ld) >= %d, nothing done\n"
                "change COMPILE_POOL_EPI8_** in include/config.h to enable higher sieving dimension.\n", 
                msd, 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32);
        return;
    }
    
    struct timeval start, end;
    double dPot;
	if (log_level >= 2) {
        gettimeofday(&start, NULL);
        dPot = L->Pot();
    }
    const long n = L->NumRows();
    ni -= max(0, min(msd + f - n, f));
    ns -= max(0, min(msd + f - n, f));
    ne -= max(0, min(msd + f - n, f));
    ne = max(0, ne);

    constexpr double pool_size_ratio = 3.2;
    const int pump_profile = svp_pump_profile_enabled();
    static long lsh_pump_profile_seq = 0;
    const long profile_seq = pump_profile ? ++lsh_pump_profile_seq : 0;
    const int profile_cuda_bgj1 = svp_cuda_bgj1_enabled();
    const double profile_total_start = pump_profile ? svp_profile_now() : 0.0;
    double profile_shuffle_gso = 0.0;
    double profile_pool_setup = 0.0;
    double profile_sampling = 0.0;
    double profile_extend = 0.0;
    double profile_sieve_bgj1 = 0.0;
    double profile_sieve_bgj2 = 0.0;
    double profile_sieve_bgj3 = 0.0;
    double profile_insert = 0.0;
    double profile_tail_lll = 0.0;
    double profile_shrink = 0.0;
    double profile_final_lll = 0.0;
    long profile_bgj1_count = 0;
    long profile_bgj2_count = 0;
    long profile_bgj3_count = 0;
    long profile_insert_count = 0;
    long profile_tail_lll_count = 0;
    int profile_stop_hit = 0;
    double profile_stop_best_length = 0.0;


    bool GET_STUCKED = shuffle_first;
    long num_stuck = 0;
    do {
        if (GET_STUCKED) {
            if (num_stuck == 0 || num_stuck == 3) {
                long shuf_dim = (n > 90) ? (0.6 * n + 14) : (n - 22);
                if (pump_profile) {
                    double __svp_profile_t0 = svp_profile_now();
                    L->tail_shuffle(shuf_dim);
                    L->compute_gso_QP(n-shuf_dim);
                    L->set_gso_status(GSO_COMPUTED_QP);
                    profile_shuffle_gso += svp_profile_now() - __svp_profile_t0;
                } else {
                    L->tail_shuffle(shuf_dim);
                    L->compute_gso_QP(n-shuf_dim);
                    L->set_gso_status(GSO_COMPUTED_QP);
                }
                if (num_stuck == 3) {
                    fprintf(stderr, "[Warning] __lsh_pump_red_epi8: get stucked, try again after shuffle.\n");
                }
            } else if (num_stuck < 3) {
                minsd += 10;
                if (minsd > n) minsd = n;
                fprintf(stderr, "[Warning] __lsh_pump_red_epi8: get stucked, try again with larger start sieving dim\n");
            }            
            GET_STUCKED = false;
        }

        long max_pool_size = (long)(pow(4./3., min(n, msd) * 0.5) * pool_size_ratio) + 1;
        if (msd <= 96) {
            Pool_epi8_t<3> p(L);
            p.set_num_threads(num_threads);
            #define __LSH_PUMP_RED_EPI8_ONE_TRY do {                                                            \
                double __svp_profile_t0 = pump_profile ? svp_profile_now() : 0.0;                                \
                long minps = (long)(pow(4./3., minsd * 0.5) * pool_size_ratio);                                 \
                minps *= 3;                                                                                     \
                p.set_max_pool_size((max_pool_size > minps) ? max_pool_size : minps);                           \
                p.set_sieving_context(n-minsd, n);                                                              \
                if (pump_profile) profile_pool_setup += svp_profile_now() - __svp_profile_t0;                    \
                int __svp_sampling_ret = 0;                                                                      \
                SVP_PROFILE_ASSIGN(profile_sampling, __svp_sampling_ret, p.sampling(minps));                     \
                if (__svp_sampling_ret == -1) return;                                                           \
                                                                                                                \
                int ret = 0;                                                                                    \
                SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level - 3, 1));               \
                profile_bgj1_count++;                                                                           \
                num_stuck = ret;                                                                                \
                while ((p.CSD < msd) && (p.CSD < n)) {                                                          \
                    SVP_PROFILE_DO(profile_extend, p.extend_left());                                             \
                    long target_num_vec = (long) (pow(4./3., p.CSD * 0.5) * pool_size_ratio);                   \
                    if (target_num_vec > p.num_vec + p.num_empty) p.num_empty = target_num_vec - p.num_vec;     \
                    if (p.CSD > 92) {                                                                           \
                        SVP_PROFILE_ASSIGN(profile_sieve_bgj3, ret, svp_bgj3_sieve(p, log_level-3, 1));         \
                        profile_bgj3_count++;                                                                   \
                    } else if (p.CSD > 80) {                                                                    \
                        SVP_PROFILE_ASSIGN(profile_sieve_bgj2, ret, svp_bgj2_sieve(p, log_level-3, 1));         \
                        profile_bgj2_count++;                                                                   \
                    } else {                                                                                    \
                        SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level-3, 1));         \
                        profile_bgj1_count++;                                                                   \
                    }                                                                                           \
                    if (ret) {                                                                                  \
                        num_stuck++;                                                                            \
                        GET_STUCKED = true;                                                                     \
                        break;                                                                                  \
                    }                                                                                           \
                }                                                                                               \
                                                                                                                \
                long ind = max(p.index_l - f, 0);                                                               \
                while ((ind < n - minsd) && (!GET_STUCKED)) {                                                   \
                    if (p.CSD >= 80) {                                                                          \
                        SVP_PROFILE_DO(profile_insert, p.lsfsh_insert(ind, eta, log_level - 3, 0.0, 0.0, qratio)); \
                    } else {                                                                                    \
                        SVP_PROFILE_DO(profile_insert, p.insert(ind, eta));                                     \
                    }                                                                                           \
                    profile_insert_count++;                                                                     \
                    if (svp_lsh_best_at_or_below(L->NumCols(), stop_length, &profile_stop_best_length)) {        \
                        profile_stop_hit = 1;                                                                   \
                        break;                                                                                  \
                    }                                                                                           \
                    int tlll_dim = 32;                                                                          \
                    for (long i = 32; i < p.CSD; i++) {                                                         \
                        if (L->get_B().hi[p.CSD-i+p.index_l] <  0.49 * L->get_B().hi[p.CSD-i-1+p.index_l]) {    \
                            tlll_dim = i;                                                                       \
                        }                                                                                       \
                    }                                                                                           \
                    SVP_PROFILE_DO(profile_tail_lll, p.tail_LLL(0.99, p.CSD));                                  \
                    profile_tail_lll_count++;                                                                   \
                    if (pump_profile) {                                                                         \
                        fprintf(stderr, "lsh_pump_progress: seq=%ld n=%ld msd=%ld ind=%ld csd=%ld elapsed=%.6fs insert=%.6fs/%ld tail_lll=%.6fs/%ld bgj1=%.6fs/%ld bgj2=%.6fs/%ld bgj3=%.6fs/%ld\n", \
                                profile_seq, n, msd, ind, p.CSD, svp_profile_now() - profile_total_start,      \
                                profile_insert, profile_insert_count, profile_tail_lll, profile_tail_lll_count,  \
                                profile_sieve_bgj1, profile_bgj1_count, profile_sieve_bgj2, profile_bgj2_count, \
                                profile_sieve_bgj3, profile_bgj3_count);                                        \
                    }                                                                                           \
                                                                                                                \
                    ni--;                                                                                       \
                    if (ni <= 0) break;                                                                         \
                    ind++;                                                                                      \
                                                                                                                \
                    if (ne > 0) {                                                                               \
                        ne--;                                                                                   \
                        SVP_PROFILE_DO(profile_extend, p.extend_left());                                        \
                    }                                                                                           \
                    if (ns > 0) {                                                                               \
                        ns--;                                                                                   \
                        if ((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio) < p.num_vec) {                    \
                            SVP_PROFILE_DO(profile_shrink, p.shrink((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio))); \
                        }                                                                                       \
                        if (p.CSD > 102) {                                                                       \
                            SVP_PROFILE_ASSIGN(profile_sieve_bgj3, ret, svp_bgj3_sieve(p, log_level-3, 1));     \
                            profile_bgj3_count++;                                                               \
                        } else if (p.CSD > 90) {                                                                \
                            SVP_PROFILE_ASSIGN(profile_sieve_bgj2, ret, svp_bgj2_sieve(p, log_level-3, 1));     \
                            profile_bgj2_count++;                                                               \
                        } else {                                                                                \
                            SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level-3, 1));     \
                            profile_bgj1_count++;                                                               \
                        }                                                                                       \
                        if (ret) {                                                                              \
                            num_stuck++;                                                                        \
                            GET_STUCKED = true;                                                                 \
                            break;                                                                              \
                        }                                                                                       \
                    }                                                                                           \
                }                                                                                               \
            } while (0)

            __LSH_PUMP_RED_EPI8_ONE_TRY;
        } else if (msd <= 128) {
            #if COMPILE_POOL_EPI8_128
            Pool_epi8_t<4> p(L);
            p.set_num_threads(num_threads);
            __LSH_PUMP_RED_EPI8_ONE_TRY;
            #endif
        } else {
            #if COMPILE_POOL_EPI8_160
            Pool_epi8_t<5> p(L);
            p.set_num_threads(num_threads);
            __LSH_PUMP_RED_EPI8_ONE_TRY;
            #endif
        }
    } while (GET_STUCKED && num_stuck <= 3);

    SVP_PROFILE_DO(profile_final_lll, L->LLL_QP(0.99));
    if (pump_profile) {
        fprintf(stderr,
                "lsh_pump_profile: seq=%ld n=%ld msd=%ld f=%ld minsd=%ld cuda_bgj1=%d stop_length=%.9g early_stop=%d stop_best=%.9g total=%.6fs shuffle_gso=%.6fs setup=%.6fs sampling=%.6fs extend=%.6fs bgj1=%.6fs/%ld bgj2=%.6fs/%ld bgj3=%.6fs/%ld insert=%.6fs/%ld tail_lll=%.6fs/%ld shrink=%.6fs final_lll=%.6fs\n",
                profile_seq, n, msd, f, minsd, profile_cuda_bgj1, stop_length, profile_stop_hit,
                profile_stop_best_length, svp_profile_now() - profile_total_start,
                profile_shuffle_gso, profile_pool_setup, profile_sampling, profile_extend,
                profile_sieve_bgj1, profile_bgj1_count, profile_sieve_bgj2, profile_bgj2_count,
                profile_sieve_bgj3, profile_bgj3_count, profile_insert, profile_insert_count,
                profile_tail_lll, profile_tail_lll_count, profile_shrink, profile_final_lll);
    }
    if (log_level < 2) return;
    for (long i = 0; i < n; i++){
        dPot -= (n-i) * log2(L->get_B().hi[i]);
    }

    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    fprintf(stdout, "__pump_red_done: runtime = %fs, num_threads = %ld, total_cost = %fTs, dPot = %f\n",
             total_time, num_threads, (total_time * num_threads), dPot);
}

double __last_lsh_pump_epi8(Lattice_QP *L, long num_threads, double qratio, double ext_qratio, long msd, long ext_d, long log_level, long shuffle_first, long minsd, double stop_length, long lift_start_csd) {
    if (min(msd, L->NumRows()) < minsd){
        fprintf(stderr, "[Warning] sieving dimension too small, may get stuck, nothing done.\nYou can use Enum based algorithms instead.\n");
        return 0.0;
    }
    if (msd + ext_d > 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32) {
        fprintf(stderr, "[Warning] sieving dimension(%ld+%ld) >= %d, nothing done\n"
                "change COMPILE_POOL_EPI8_** in include/config.h to enable higher sieving dimension.\n",
                msd, ext_d, 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32);
        return 0.0;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    const long n = L->NumRows();
    if (msd > n) msd = n;
    if (msd + ext_d > n) ext_d = n - msd;
    if (lift_start_csd <= 0) lift_start_csd = msd - 12;
    if (lift_start_csd < minsd) lift_start_csd = minsd;
    if (lift_start_csd > msd + ext_d) lift_start_csd = msd + ext_d;
    double best_lift_length = 0.0;

    const int pump_profile = svp_pump_profile_enabled();
    const double profile_total_start = pump_profile ? svp_profile_now() : 0.0;
    double profile_pool_setup = 0.0;
    double profile_sampling = 0.0;
    double profile_extend = 0.0;
    double profile_sieve_bgj1 = 0.0;
    double profile_sieve_bgj2 = 0.0;
    double profile_sieve_bgj3 = 0.0;
    double profile_show_lift = 0.0;
    double profile_final_lll = 0.0;
    long profile_bgj1_count = 0;
    long profile_bgj2_count = 0;
    long profile_bgj3_count = 0;
    long profile_show_lift_count = 0;
    int profile_stop_hit = 0;

    constexpr double pool_size_ratio = 3.2;

    bool GET_STUCKED = shuffle_first;
    long num_stuck = 0;
    do {
        if (GET_STUCKED) {
            if (num_stuck == 0 || num_stuck == 3) {
                long shuf_dim = (n > 90) ? (0.6 * n + 14) : (n - 22);
                L->tail_shuffle(shuf_dim);
                L->compute_gso_QP(n-shuf_dim);
                L->set_gso_status(GSO_COMPUTED_QP);
                if (num_stuck == 3) {
                    fprintf(stderr, "[Warning] __pump_red_epi8: get stucked, try again after shuffle.\n");
                }
            } else if (num_stuck < 3) {
                minsd += 10;
                if (minsd > n) minsd = n;
                fprintf(stderr, "[Warning] __pump_red_epi8: get stucked, try again with larger start sieving dim\n");
            }            
            GET_STUCKED = false;
        }

        const long max_pool_size = (long)(pow(4./3., msd * 0.5) * pool_size_ratio);

        #define __LAST_LSH_PUMP_EPI8_ONE_TRY do {                                                           \
            double __svp_profile_t0 = pump_profile ? svp_profile_now() : 0.0;                                \
            p.set_num_threads(num_threads);                                                                 \
            long minps = (long)(pow(4./3., minsd * 0.5) * pool_size_ratio);                                 \
            minps *= 3;                                                                                     \
            p.set_max_pool_size((max_pool_size > minps) ? max_pool_size : minps);                           \
            p.set_sieving_context(n-minsd, n);                                                              \
            if (pump_profile) profile_pool_setup += svp_profile_now() - __svp_profile_t0;                    \
            int __svp_sampling_ret = 0;                                                                      \
            SVP_PROFILE_ASSIGN(profile_sampling, __svp_sampling_ret, p.sampling(minps));                     \
            if (__svp_sampling_ret == -1) return best_lift_length;                                           \
	                                                                                                            \
            int ret = 0;                                                                                    \
            SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level - 3, 1));               \
            profile_bgj1_count++;                                                                           \
            num_stuck = ret;                                                                                \
            while (p.CSD < msd + ext_d) {                                                                   \
                SVP_PROFILE_DO(profile_extend, p.extend_left());                                             \
                long target_num_vec = (long) (pow(4./3., min(p.CSD, msd) * 0.5) * pool_size_ratio);         \
                if (target_num_vec > p.num_vec + p.num_empty) p.num_empty = target_num_vec - p.num_vec;     \
	                                                                                                            \
                if (p.CSD > 92) {                                                                           \
                    SVP_PROFILE_ASSIGN(profile_sieve_bgj3, ret, svp_bgj3_sieve(p, log_level-3, 1));         \
                    profile_bgj3_count++;                                                                   \
                } else if (p.CSD > 80) {                                                                    \
                    SVP_PROFILE_ASSIGN(profile_sieve_bgj2, ret, svp_bgj2_sieve(p, log_level-3, 1));         \
                    profile_bgj2_count++;                                                                   \
                } else {                                                                                    \
                    SVP_PROFILE_ASSIGN(profile_sieve_bgj1, ret, svp_bgj1_sieve(p, log_level-3, 1));         \
                    profile_bgj1_count++;                                                                   \
                }                                                                                           \
                if (p.CSD >= lift_start_csd) {                                                               \
                    double __svp_lift_profile_t0 = pump_profile ? svp_profile_now() : 0.0;                  \
                    if (p.CSD < n - 24 && (((p.CSD >= msd) && ext_qratio != 0.0) || ((p.CSD < msd) && qratio != 0.0))) { \
                        p.show_lsfsh_insert(0, 10.0, log_level-3, 0, 0, (p.CSD>=msd)?ext_qratio:qratio);    \
                    } else {                                                                                \
                        p.show_min_lift(0);                                                                 \
                    }                                                                                       \
                    if (pump_profile) profile_show_lift += svp_profile_now() - __svp_lift_profile_t0;       \
                    profile_show_lift_count++;                                                              \
                    if (p.last_lift_valid &&                                                                \
                        (best_lift_length == 0.0 || p.last_lift_euclidean_norm < best_lift_length)) {       \
                        best_lift_length = p.last_lift_euclidean_norm;                                      \
                    }                                                                                       \
                    if (stop_length > 0.0 && best_lift_length > 0.0 && best_lift_length <= stop_length) {    \
                        profile_stop_hit = 1;                                                               \
                        break;                                                                              \
                    }                                                                                       \
                }                                                                                           \
                if (ret) {                                                                                  \
                    num_stuck++;                                                                            \
                    GET_STUCKED = true;                                                                     \
                    break;                                                                                  \
                }                                                                                           \
            }                                                                                               \
        } while (0)

        if (msd + ext_d <= 96) {
            Pool_epi8_t<3> p(L);
            __LAST_LSH_PUMP_EPI8_ONE_TRY;
        } else if (msd + ext_d <= 128) {
            #if COMPILE_POOL_EPI8_128
            Pool_epi8_t<4> p(L);
            __LAST_LSH_PUMP_EPI8_ONE_TRY;
            #endif
        } else {
            #if COMPILE_POOL_EPI8_160
            Pool_epi8_t<5> p(L);
            __LAST_LSH_PUMP_EPI8_ONE_TRY;
            #endif
        }
    } while (GET_STUCKED && num_stuck <= 3);



    SVP_PROFILE_DO(profile_final_lll, L->LLL_QP(0.99));
    if (pump_profile) {
        fprintf(stderr,
                "last_lsh_profile: n=%ld msd=%ld ext_d=%ld minsd=%ld lift_start_csd=%ld stop_length=%.9g early_stop=%d total=%.6fs setup=%.6fs sampling=%.6fs extend=%.6fs bgj1=%.6fs/%ld bgj2=%.6fs/%ld bgj3=%.6fs/%ld show_lift=%.6fs/%ld final_lll=%.6fs\n",
                n, msd, ext_d, minsd, lift_start_csd, stop_length, profile_stop_hit,
                svp_profile_now() - profile_total_start,
                profile_pool_setup, profile_sampling, profile_extend,
                profile_sieve_bgj1, profile_bgj1_count, profile_sieve_bgj2, profile_bgj2_count,
                profile_sieve_bgj3, profile_bgj3_count, profile_show_lift, profile_show_lift_count,
                profile_final_lll);
    }
    if (best_lift_length > 0.0) {
        fprintf(stdout, "__last_lsh_pump_epi8 best_lift_length = %.9g\n", best_lift_length);
    }
    if (log_level < 2) return best_lift_length;
    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    fprintf(stdout, "__last_lsh_pump_epi8 done: runtime = %fs, num_threads = %ld, total_cost = %fTs\n",
             total_time, num_threads, (total_time * num_threads));
    return best_lift_length;
}


#if defined(__AMX_INT8__)
void __pump_red_amx(Lattice_QP *L, long num_threads, double eta, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first, long minsd) {
    if (min(msd, L->NumRows()) < minsd){
        fprintf(stderr, "[Warning] sieving dimension too small, may get stuck, nothing done.\nYou can use Enum based algorithms instead.\n");
        return;
    }
    if (msd > 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32) {
        fprintf(stderr, "[Warning] sieving dimension(%ld) >= %d, nothing done\n"
                "change COMPILE_POOL_EPI8_** in include/config.h to enable higher sieving dimension.\n", 
                msd, 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32);
        return;
    }
    
    struct timeval start, end;
    double dPot;
	if (log_level >= 2) {
        gettimeofday(&start, NULL);
        dPot = L->Pot();
    }
    const long n = L->NumRows();
    ni -= max(0, min(msd + f - n, f));
    ns -= max(0, min(msd + f - n, f));
    ne -= max(0, min(msd + f - n, f));
    ne = max(0, ne);

    constexpr double pool_size_ratio = 3.2;
    

    bool GET_STUCKED = shuffle_first;
    long num_stuck = 0;
    do {
        if (GET_STUCKED) {
            if (num_stuck == 0 || num_stuck == 2) {
                long shuf_dim = (n > 90) ? (0.6 * n + 14) : (n - 22);
                L->tail_shuffle(shuf_dim);
                L->compute_gso_QP(n-shuf_dim);
                L->set_gso_status(GSO_COMPUTED_QP);
                if (num_stuck == 2) {
                    fprintf(stderr, "[Warning] __pump_red_amx: get stucked, try again after shuffle.\n");
                }
            } else if (num_stuck < 2) {
                minsd += 10;
                if (minsd > n) minsd = n;
                fprintf(stderr, "[Warning] __pump_red_amx: get stucked, try again with larger start sieving dim\n");
            }            
            GET_STUCKED = false;
        }

        long max_pool_size = (long)(pow(4./3., min(n, msd) * 0.5) * pool_size_ratio) + 1;
        do {
            Pool_epi8_t<5> p(L);
            p.set_num_threads(num_threads);
            long minps = (long)(pow(4./3., minsd * 0.5) * pool_size_ratio);                                 
            minps *= 3;                                                                                     
            p.set_max_pool_size((max_pool_size > minps) ? max_pool_size : minps);                           
            p.set_sieving_context(n-minsd, n);                                                              
            if (p.sampling(minps) == -1) return;
                                                                                                            
            int ret = svp_bgj1_sieve(p, log_level - 3, 1);
            if (ret) {
                num_stuck++;
                GET_STUCKED = true;
            }                                                                                                                                 
            while ((p.CSD < msd) && (p.CSD < n) && (!GET_STUCKED)) {                                                          
                p.extend_left();                                                                            
                long target_num_vec = (long) (pow(4./3., p.CSD * 0.5) * pool_size_ratio);                   
                if (target_num_vec > p.num_vec + p.num_empty) p.num_empty = target_num_vec - p.num_vec;     
                ret = p.bgj_amx_upsieve(log_level-3, -1, -1.0, -1, p.CSD <= msd - 8 ? -1.0 : -1.0 - AMX_BOOST_DOWNSIEVE_MASK_RATIO * (-msd + p.CSD + 8));                                                       
                if (ret) {                                                                                  
                    num_stuck++;                                                                            
                    GET_STUCKED = true;                                                                     
                    break;                                                                                  
                }                                                                                           
            }                                                                                               
                                                                                                            
            long ind = max(p.index_l - f, 0);                                                               
            while ((ind < n - minsd) && (!GET_STUCKED)) {                                                   
                p.insert(ind, eta);                                                                         
                int tlll_dim = 32;                                                                          
                for (long i = 32; i < p.CSD - 1; i++) {                                                     
                    if (L->get_B().hi[p.CSD-i+p.index_l] <  0.49 * L->get_B().hi[p.CSD-i-1+p.index_l]) {    
                        tlll_dim = i;                                                                       
                    }                                                                                       
                }                                                                                           
                p.tail_LLL(0.99, p.CSD);                                                                 
                                                                                                            
                ni--;                                                                                       
                if (ni <= 0) break;                                                                         
                ind++;                                                                                      
                                                                                                            
                if (ne > 0) {                                                                               
                    ne--;                                                                                   
                    p.extend_left();                                                                        
                    ind--;                                                                                  
                }                                                                                           
                if (ns > 0) {                                                                               
                    ns--;                                                                                   
                    if ((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio) < p.num_vec) {                    
                        p.shrink((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio));                        
                    }                                                                                       
                    ret = p.bgj_amx_downsieve(log_level-3, -1, -1.0, -1, -1.0 - AMX_BOOST_DOWNSIEVE_MASK_RATIO * (msd - p.CSD + 8));
                    if (ret) {                                                                              
                        num_stuck++;                                                                        
                        GET_STUCKED = true;                                                                 
                        break;                                                                              
                    }                                                                                       
                }                                                                                           
            }                                                                                               
        } while (0);
    } while (GET_STUCKED && num_stuck <= 2);

    L->LLL_QP(0.99);
    if (log_level < 2) return;
    L->compute_gso_QP();
    for (long i = 0; i < n; i++){
        dPot -= (n-i) * log2(L->get_B().hi[i]);
    }

    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    fprintf(stdout, "__pump_red_done: runtime = %fs, num_threads = %ld, total_cost = %fTs, dPot = %f(%.4f)\n",
             total_time, num_threads, (total_time * num_threads), dPot, sqrt(L->get_B().hi[0]) / L->gh());
}

void __lsh_pump_red_amx(Lattice_QP *L, long num_threads, double eta, double qratio, long msd, long f, long ni, long ne, long ns, long log_level, long shuffle_first, long minsd) {
    if (min(msd, L->NumRows()) < minsd){
        fprintf(stderr, "[Warning] sieving dimension too small, may get stuck, nothing done.\nYou can use Enum based algorithms instead.\n");
        return;
    }
    if (msd > 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32) {
        fprintf(stderr, "[Warning] sieving dimension(%ld) >= %d, nothing done\n"
                "change COMPILE_POOL_EPI8_** in include/config.h to enable higher sieving dimension.\n", 
                msd, 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32);
        return;
    }
    
    struct timeval start, end;
    double dPot;
	if (log_level >= 2) {
        gettimeofday(&start, NULL);
        dPot = L->Pot();
    }
    const long n = L->NumRows();
    ni -= max(0, min(msd + f - n, f));
    ns -= max(0, min(msd + f - n, f));
    ne -= max(0, min(msd + f - n, f));
    ne = max(0, ne);

    constexpr double pool_size_ratio = 3.2;
    

    bool GET_STUCKED = shuffle_first;
    long num_stuck = 0;
    do {
        if (GET_STUCKED) {
            if (num_stuck == 0 || num_stuck == 3) {
                long shuf_dim = (n > 90) ? (0.6 * n + 14) : (n - 22);
                L->tail_shuffle(shuf_dim);
                L->compute_gso_QP(n-shuf_dim);
                L->set_gso_status(GSO_COMPUTED_QP);
                if (num_stuck == 3) {
                    fprintf(stderr, "[Warning] __lsh_pump_red_amx: get stucked, try again after shuffle.\n");
                }
            } else if (num_stuck < 3) {
                minsd += 10;
                if (minsd > n) minsd = n;
                fprintf(stderr, "[Warning] __lsh_pump_red_amx: get stucked, try again with larger start sieving dim\n");
            }            
            GET_STUCKED = false;
        }

        long max_pool_size = (long)(pow(4./3., min(n, msd) * 0.5) * pool_size_ratio) + 1;
        do {
            Pool_epi8_t<5> p(L);
            p.set_num_threads(num_threads);
            long minps = (long)(pow(4./3., minsd * 0.5) * pool_size_ratio);                                 
            minps *= 3;                                                                                     
            p.set_max_pool_size((max_pool_size > minps) ? max_pool_size : minps);                           
            p.set_sieving_context(n-minsd, n);                                                              
            p.sampling(minps);                                                                              
                                                                                                            
            int ret = svp_bgj1_sieve(p, log_level - 3, 1);
            num_stuck = ret;                                                                                
            while ((p.CSD < msd) && (p.CSD < n)) {                                                          
                p.extend_left();                                                                            
                long target_num_vec = (long) (pow(4./3., p.CSD * 0.5) * pool_size_ratio);                   
                if (target_num_vec > p.num_vec + p.num_empty) p.num_empty = target_num_vec - p.num_vec;     
                ret = p.bgj_amx_upsieve(log_level-3, -1, -1.0, -1, p.CSD <= msd - 8 ? -1.0 : -1.0 - AMX_BOOST_DOWNSIEVE_MASK_RATIO * (-msd + p.CSD + 8));                                                        
                if (ret) {                                                                                  
                    num_stuck++;                                                                            
                    GET_STUCKED = true;                                                                     
                    break;                                                                                  
                }                                                                                           
            }                                                                                               
                                                                                                            
            long ind = max(p.index_l - f, 0);                                                               
            while ((ind < n - minsd) && (!GET_STUCKED)) {                                                   
                if (p.CSD >= 80) p.lsfsh_insert(ind, eta, log_level - 3, 0.0, 0.0, qratio);                 
                else p.insert(ind, eta);                                                                    
                int tlll_dim = 32;                                                                          
                for (long i = 32; i < p.CSD; i++) {                                                         
                    if (L->get_B().hi[p.CSD-i+p.index_l] <  0.49 * L->get_B().hi[p.CSD-i-1+p.index_l]) {    
                        tlll_dim = i;                                                                       
                    }                                                                                       
                }                                                                                           
                p.tail_LLL(0.99, p.CSD);                                                                    
                                                                                                            
                ni--;                                                                                       
                if (ni <= 0) break;                                                                         
                ind++;                                                                                      
                                                                                                            
                if (ne > 0) {                                                                               
                    ne--;                                                                                   
                    p.extend_left();                                                                        
                }                                                                                           
                if (ns > 0) {                                                                               
                    ns--;                                                                                   
                    if ((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio) < p.num_vec) {                    
                        p.shrink((long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio));                        
                    }                                                                                       
                    ret = p.bgj_amx_downsieve(log_level-3, -1, -1.0, -1, -1.0 - AMX_BOOST_DOWNSIEVE_MASK_RATIO * (msd - p.CSD + 8));
                    if (ret) {                                                                              
                        num_stuck++;                                                                        
                        GET_STUCKED = true;                                                                 
                        break;                                                                              
                    }                                                                                       
                }                                                                                           
            }                                                                                               
        } while (0);
    } while (GET_STUCKED && num_stuck <= 3);

    L->LLL_QP(0.99);
    if (log_level < 2) return;
    for (long i = 0; i < n; i++){
        dPot -= (n-i) * log2(L->get_B().hi[i]);
    }

    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    fprintf(stdout, "__lsh_pump_red_done: runtime = %fs, num_threads = %ld, total_cost = %fTs, dPot = %f\n",
             total_time, num_threads, (total_time * num_threads), dPot);
}

void __last_lsh_pump_amx(Lattice_QP *L, long num_threads, double qratio, double ext_qratio, long msd, long ext_d, long log_level, long shuffle_first, long minsd) {
    if (min(msd, L->NumRows()) < minsd){
        fprintf(stderr, "[Warning] sieving dimension too small, may get stuck, nothing done.\nYou can use Enum based algorithms instead.\n");
        return;
    }
    if (msd + ext_d > 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32) {
        fprintf(stderr, "[Warning] sieving dimension(%ld+%ld) >= %d, nothing done\n"
                "change COMPILE_POOL_EPI8_** in include/config.h to enable higher sieving dimension.\n", 
                msd, ext_d, 96 + COMPILE_POOL_EPI8_128 * 32 + COMPILE_POOL_EPI8_160 * 32);
        return;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    const long n = L->NumRows();
    if (msd > n) msd = n;
    if (msd + ext_d > n) ext_d = n - msd;

    constexpr double pool_size_ratio = 3.2;

    bool GET_STUCKED = shuffle_first;
    long num_stuck = 0;
    do {
        if (GET_STUCKED) {
            if (num_stuck == 0 || num_stuck == 3) {
                long shuf_dim = (n > 90) ? (0.6 * n + 14) : (n - 22);
                L->tail_shuffle(shuf_dim);
                L->compute_gso_QP(n-shuf_dim);
                L->set_gso_status(GSO_COMPUTED_QP);
                if (num_stuck == 3) {
                    fprintf(stderr, "[Warning] __last_lsh_pump_red_amx: get stucked, try again after shuffle.\n");
                }
            } else if (num_stuck < 3) {
                minsd += 10;
                if (minsd > n) minsd = n;
                fprintf(stderr, "[Warning] __last_lsh_pump_red_amx: get stucked, try again with larger start sieving dim\n");
            }            
            GET_STUCKED = false;
        }

        const long max_pool_size = (long)(pow(4./3., msd * 0.5) * pool_size_ratio);
        do {
            Pool_epi8_t<5> p(L);
            p.set_num_threads(num_threads);                                                                 
            long minps = (long)(pow(4./3., minsd * 0.5) * pool_size_ratio);                                 
            minps *= 3;                                                                                     
            p.set_max_pool_size((max_pool_size > minps) ? max_pool_size : minps);                           
            p.set_sieving_context(n-minsd, n);                                                              
            p.sampling(minps);                                                                              
                                                                                                            
            int ret = svp_bgj1_sieve(p, log_level - 3, 1);
            num_stuck = ret;                                                                                
            while (p.CSD < msd + ext_d) {                                                                   
                p.extend_left();                                                                            
                long target_num_vec = (long) (pow(4./3., min(p.CSD, msd) * 0.5) * pool_size_ratio);         
                if (target_num_vec > p.num_vec + p.num_empty) p.num_empty = target_num_vec - p.num_vec;     
                                                                                                            
                ret = p.bgj_amx_upsieve(log_level-3, -1, -1.0, -1, p.CSD <= msd - 8 ? -1.0 : -1.0 - AMX_BOOST_DOWNSIEVE_MASK_RATIO * (-msd + p.CSD + 8));                                                                                
                if (p.CSD >= msd - 12) {                                                                    
                    if (p.CSD < n - 24 && (((p.CSD >= msd) && ext_qratio != 0.0) || ((p.CSD < msd) && qratio != 0.0))) {                           
                        p.show_lsfsh_insert(0, 10.0, log_level-3, 0, 0, (p.CSD>=msd)?ext_qratio:qratio);    
                    } else {                                                                                
                        p.show_min_lift(0);                                                                 
                    }                                                                                       
                }                                                                                           
                if (ret) {                                                                                  
                    num_stuck++;                                                                            
                    GET_STUCKED = true;                                                                     
                    break;                                                                                  
                }                                                                                           
            } 
            for (;;) {
                std::ifstream INstream(".in");
                if (!INstream) break;

                char instruction;
                INstream >> instruction;
                if (instruction == 's') {
                    p.bgj_amx_upsieve(log_level-3, -1, -1.0, -1, -10.0);
                    p.show_min_lift(0);
                } else if (instruction == 'h') {
                    std::ifstream VINstream(".vin");
                    double qr;
                    VINstream >> qr;
                    if (qr > 0.01 && qr < 1.0) {
                        p.show_lsfsh_insert(0, 10.0, log_level-3, 0, 0, qr);
                    }
                    VINstream.close();
                } else if (instruction == 'r') {
                    for (long i = 0; i < 16; i++) {
                        p.insert(i, 10.0);
                        p.tail_LLL(0.99, p.CSD);
                    }
                    p.basis->store("ftmp");
                    break;
                } else {
                    sleep(60);
                }
                INstream.close();
            }                                                                 
        } while (0);
    } while (GET_STUCKED && num_stuck <= 3);

    L->LLL_QP(0.99);
    if (log_level < 2) return;
    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    fprintf(stdout, "__last_lsh_pump_epi8 done: runtime = %fs, num_threads = %ld, total_cost = %fTs\n",
             total_time, num_threads, (total_time * num_threads));
}

#endif
/** hkz-reduce the lattice, based on 3-Sieve.
 */
void __hkz_red(Lattice_QP *L, long num_threads){
    long n = L->NumRows();
    if (n < 30){
        L->LLL_DEEP_QP(0.99);
        return;
    }
    struct timeval start, end;
	gettimeofday(&start, NULL);
    double dPot = L->Pot();


    Pool pool(L);
    long max_pool_size = (long)(pow(4./3., n * 0.5) * 3.2) + 1;
    pool.set_num_threads(num_threads);
    pool.set_MSD(n);
    pool.set_max_pool_size(max(1009, max_pool_size));
    pool.set_sieving_context(max(0, n-40), n);
    pool.gaussian_sampling(max(1009, (long)(pow(4./3., pool.CSD*0.5)*3.2)));
    three_Sieve_params p;

    //main sieve
    pool.three_Sieve(p, 0, 10000000);
    while (pool.CSD < n - 1){
        pool.extend_left();
        pool.gaussian_sampling(max(1009, (long)(pow(4./3., pool.CSD*0.5) * 3.2)));
        pool.three_Sieve(p, 0, 100000);
    }

    //insertions
    long ind = 0;
    while (ind <= n - 30){
        pool.insert(ind, 10.0);

        ind++;
        pool.shrink(max(1009, (long)(pow(4./3., pool.CSD * 0.5) * 3.2)));
        pool.tail_LLL(0.99, 16);
        pool.three_Sieve(p, 0, 30000);
    }
    L->LLL_DEEP_QP(0.99, n - 30, n);

    for (long i = 0; i < n; i++){
        dPot -= (n-i) * log2(L->get_B().hi[i]);
    }
    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    std::cout << "__hkz_red_done: runtime = " << total_time << "s, num_threads = " << num_threads;
    std::cout << ", total_cost = " << (total_time * num_threads) << "Ts, dPot = " << dPot << "\n";
    L->compute_gso_QP();
}

void __hkz_red_epi8(Lattice_QP *L, long num_threads, long log_level) {
    long n = L->NumRows();
    if (n < 30){
        L->LLL_DEEP_QP(0.99);
        return;
    }
    struct timeval start, end;
	gettimeofday(&start, NULL);
    double dPot = L->Pot();

    constexpr double pool_size_ratio = 3.2;

    
    #define __HKZ_RED_EPI8_ONE_TRY do {                                                                 \
        long max_pool_size = (long)(pow(4./3., n * 0.5) * pool_size_ratio) + 1;                         \
        p.set_num_threads(num_threads);                                                                 \
        p.set_max_pool_size((max_pool_size > 2071) ? max_pool_size : 2071);                             \
        p.set_sieving_context(max(0, n-40), n);                                                         \
        p.sampling(2071);                                                                               \
        svp_bgj1_sieve(p, log_level - 3, 1); \
        while (p.CSD < n - 1){                                                                          \
            p.extend_left();                                                                            \
            long target_num_vec = (long) (pow(4./3., p.CSD * 0.5) * 3.2);                               \
            if (target_num_vec > p.num_vec + p.num_empty) p.num_empty = target_num_vec - p.num_vec;     \
            if (p.CSD > 92) {                                                                           \
                svp_bgj3_sieve(p, log_level-3, 1);                                                       \
            } else if (p.CSD > 80) {                                                                    \
                svp_bgj2_sieve(p, log_level-3, 1);                                                       \
            } else {                                                                                    \
                svp_bgj1_sieve(p, log_level-3, 1); \
            }                                                                                           \
        }                                                                                               \
        long ind = 0;                                                                                   \
        while (ind <= n - 30){                                                                          \
            p.insert(ind, 10.0);                                                                        \
            p.tail_LLL(0.99, 16);                                                                       \
            p.shrink(max(1009, (long)(pow(4./3., p.CSD * 0.5) * pool_size_ratio)));                     \
            ind++;                                                                                      \
            if (p.CSD > 92) {                                                                           \
                svp_bgj3_sieve(p, log_level-3, 1);                                                       \
            } else if (p.CSD > 80) {                                                                    \
                svp_bgj2_sieve(p, log_level-3, 1);                                                       \
            } else {                                                                                    \
                svp_bgj1_sieve(p, log_level-3, 1); \
            }                                                                                           \
        }                                                                                               \
    } while (0);
    if (n <= 96) {
        Pool_epi8_t<3> p(L);
        __HKZ_RED_EPI8_ONE_TRY;
    } else if (n <= 128) {
        #if COMPILE_POOL_EPI8_128
        Pool_epi8_t<4> p(L);
        __HKZ_RED_EPI8_ONE_TRY;
        #else 
        fprintf(stderr, "[Error] __hkz_red_epi8: n = %ld > 96, nothing done\n", n);
        #endif
    } else if (n <= 160) {
        #if COMPILE_POOL_EPI8_160
        Pool_epi8_t<5> p(L);
        __HKZ_RED_EPI8_ONE_TRY;
        #else 
        fprintf(stderr, "[Error] __hkz_red_epi8: n = %ld > 128, nothing done\n", n);
        #endif
    } else {
        fprintf(stderr, "[Error] __hkz_red_epi8: n = %ld > 160, nothing done\n", n);
        return;
    }

    L->LLL_DEEP_QP(0.99, n - 30, n);
    if (log_level < 2) return;
    for (long i = 0; i < n; i++){
        dPot -= (n-i) * log2(L->get_B().hi[i]);
    }
    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec-start.tv_sec)+(double)(end.tv_usec-start.tv_usec)/1000000.0;
    std::cout << "__hkz_red_done: runtime = " << total_time << "s, num_threads = " << num_threads;
    std::cout << ", total_cost = " << (total_time * num_threads) << "Ts, dPot = " << dPot << "\n";
    L->compute_gso_QP();
}



long _red_60(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 60, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_61(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 61, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_62(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 62, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_63(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 63, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_64(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 64, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_65(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 65, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_66(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 66, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_67(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 67, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_68(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 68, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_69(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 69, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_70(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 70, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_71(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 71, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_72(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 72, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_73(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 73, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_74(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 74, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_75(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 75, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_76(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 76, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_77(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 77, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_78(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 78, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_79(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 79, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_80(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 80, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_81(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 81, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_82(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 82, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_83(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 83, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_84(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 84, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_85(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 85, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_86(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 86, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_87(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 87, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_88(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 88, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_89(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 89, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_90(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 90, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_91(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 91, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_92(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 92, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_93(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 93, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_94(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 94, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_95(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 95, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_96(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 96, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_97(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 97, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_98(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 98, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_99(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 99, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_100(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 100, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_101(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 101, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_102(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 102, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_103(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 103, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_104(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 104, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_105(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 105, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_106(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 106, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_107(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 107, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_108(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 108, 18, 24, 0, 24);
    return JUMPING_STEP;
}
long _red_109(Lattice_QP *L, long num_threads){
    L->compute_gso_QP();
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;
    if (L->NumRows() < 60) {
        __hkz_red(L, num_threads);
        return 60;
    }
    __pump_red(L, num_threads, 1.10, 109, 18, 24, 0, 24);
    return JUMPING_STEP;
}

#define DEFINE_PUMP_RED_EPI8(MSD)                                           \
long _pump_red_epi8_##MSD(Lattice_QP *L, long num_threads){                 \
    long d4f = (((MSD) + 8) / 16 + 13);                                     \
    L->compute_gso_QP();                                                    \
    if (L->gh() > 1.1 * sqrt(L->get_B().hi[0])) return 1;                   \
    if (L->NumRows() < 60) {                                                \
        __hkz_red_epi8(L, num_threads, 2);                                  \
        return 60;                                                          \
    }                                                                       \
    __pump_red_epi8(L, num_threads, 1.1, MSD, d4f, d4f+6, 0, d4f+6, 2, 1);  \
    return JUMPING_STEP;                                                    \
}


DEFINE_PUMP_RED_EPI8(60)
DEFINE_PUMP_RED_EPI8(61)
DEFINE_PUMP_RED_EPI8(62)
DEFINE_PUMP_RED_EPI8(63)
DEFINE_PUMP_RED_EPI8(64)
DEFINE_PUMP_RED_EPI8(65)
DEFINE_PUMP_RED_EPI8(66)
DEFINE_PUMP_RED_EPI8(67)
DEFINE_PUMP_RED_EPI8(68)
DEFINE_PUMP_RED_EPI8(69)
DEFINE_PUMP_RED_EPI8(70)
DEFINE_PUMP_RED_EPI8(71)
DEFINE_PUMP_RED_EPI8(72)
DEFINE_PUMP_RED_EPI8(73)
DEFINE_PUMP_RED_EPI8(74)
DEFINE_PUMP_RED_EPI8(75)
DEFINE_PUMP_RED_EPI8(76)
DEFINE_PUMP_RED_EPI8(77)
DEFINE_PUMP_RED_EPI8(78)
DEFINE_PUMP_RED_EPI8(79)
DEFINE_PUMP_RED_EPI8(80)
DEFINE_PUMP_RED_EPI8(81)
DEFINE_PUMP_RED_EPI8(82)
DEFINE_PUMP_RED_EPI8(83)
DEFINE_PUMP_RED_EPI8(84)
DEFINE_PUMP_RED_EPI8(85)
DEFINE_PUMP_RED_EPI8(86)
DEFINE_PUMP_RED_EPI8(87)
DEFINE_PUMP_RED_EPI8(88)
DEFINE_PUMP_RED_EPI8(89)
DEFINE_PUMP_RED_EPI8(90)
DEFINE_PUMP_RED_EPI8(91)
DEFINE_PUMP_RED_EPI8(92)
DEFINE_PUMP_RED_EPI8(93)
DEFINE_PUMP_RED_EPI8(94)
DEFINE_PUMP_RED_EPI8(95)
DEFINE_PUMP_RED_EPI8(96)
#if COMPILE_POOL_EPI8_128
DEFINE_PUMP_RED_EPI8(97)
DEFINE_PUMP_RED_EPI8(98)
DEFINE_PUMP_RED_EPI8(99)
DEFINE_PUMP_RED_EPI8(100)
DEFINE_PUMP_RED_EPI8(101)
DEFINE_PUMP_RED_EPI8(102)
DEFINE_PUMP_RED_EPI8(103)
DEFINE_PUMP_RED_EPI8(104)
DEFINE_PUMP_RED_EPI8(105)
DEFINE_PUMP_RED_EPI8(106)
DEFINE_PUMP_RED_EPI8(107)
DEFINE_PUMP_RED_EPI8(108)
DEFINE_PUMP_RED_EPI8(109)
DEFINE_PUMP_RED_EPI8(110)
DEFINE_PUMP_RED_EPI8(111)
DEFINE_PUMP_RED_EPI8(112)
DEFINE_PUMP_RED_EPI8(113)
DEFINE_PUMP_RED_EPI8(114)
DEFINE_PUMP_RED_EPI8(115)
DEFINE_PUMP_RED_EPI8(116)
DEFINE_PUMP_RED_EPI8(117)
DEFINE_PUMP_RED_EPI8(118)
DEFINE_PUMP_RED_EPI8(119)
DEFINE_PUMP_RED_EPI8(120)
DEFINE_PUMP_RED_EPI8(121)
DEFINE_PUMP_RED_EPI8(122)
DEFINE_PUMP_RED_EPI8(123)
DEFINE_PUMP_RED_EPI8(124)
DEFINE_PUMP_RED_EPI8(125)
DEFINE_PUMP_RED_EPI8(126)
DEFINE_PUMP_RED_EPI8(127)
DEFINE_PUMP_RED_EPI8(128)
#endif
#if COMPILE_POOL_EPI8_160
DEFINE_PUMP_RED_EPI8(129)
DEFINE_PUMP_RED_EPI8(130)
DEFINE_PUMP_RED_EPI8(131)
DEFINE_PUMP_RED_EPI8(132)
DEFINE_PUMP_RED_EPI8(133)
DEFINE_PUMP_RED_EPI8(134)
DEFINE_PUMP_RED_EPI8(135)
DEFINE_PUMP_RED_EPI8(136)
DEFINE_PUMP_RED_EPI8(137)
DEFINE_PUMP_RED_EPI8(138)
DEFINE_PUMP_RED_EPI8(139)
DEFINE_PUMP_RED_EPI8(140)
DEFINE_PUMP_RED_EPI8(141)
DEFINE_PUMP_RED_EPI8(142)
DEFINE_PUMP_RED_EPI8(143)
DEFINE_PUMP_RED_EPI8(144)
DEFINE_PUMP_RED_EPI8(145)
DEFINE_PUMP_RED_EPI8(146)
DEFINE_PUMP_RED_EPI8(147)
DEFINE_PUMP_RED_EPI8(148)
DEFINE_PUMP_RED_EPI8(149)
DEFINE_PUMP_RED_EPI8(150)
DEFINE_PUMP_RED_EPI8(151)
DEFINE_PUMP_RED_EPI8(152)
DEFINE_PUMP_RED_EPI8(153)
DEFINE_PUMP_RED_EPI8(154)
DEFINE_PUMP_RED_EPI8(155)
DEFINE_PUMP_RED_EPI8(156)
DEFINE_PUMP_RED_EPI8(157)
DEFINE_PUMP_RED_EPI8(158)
DEFINE_PUMP_RED_EPI8(159)
DEFINE_PUMP_RED_EPI8(160)
#endif

