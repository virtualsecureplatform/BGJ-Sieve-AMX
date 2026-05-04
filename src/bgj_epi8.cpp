#include "../include/pool_epi8.h"
#include "../include/bgj_epi8.h"
#include "../include/bucket_epi8.h"
#include "../include/bgj_cuda.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>


#if 1
struct timeval _bgj_timer_start[MAX_NTHREADS], _bgj_timer_end[MAX_NTHREADS];
double _time_curr[MAX_NTHREADS];

#define TIMER_START do {                                                        \
        gettimeofday(&_bgj_timer_start[omp_get_thread_num()], NULL);                                  \
    } while (0)

#define TIMER_END do {                                                          \
        gettimeofday(&_bgj_timer_end[omp_get_thread_num()], NULL);                                    \
        _time_curr[omp_get_thread_num()] =                                                            \
            (_bgj_timer_end[omp_get_thread_num()].tv_sec-_bgj_timer_start[omp_get_thread_num()].tv_sec)+                    \
            (double)(_bgj_timer_end[omp_get_thread_num()].tv_usec-_bgj_timer_start[omp_get_thread_num()].tv_usec)/1000000.0;\
    } while (0)

#define CURRENT_TIME (_time_curr[omp_get_thread_num()])
#endif

static uint64_t bgj_epi8_bucket_pair_dots(long num_p, long num_n)
{
    const uint64_t p = num_p > 0 ? (uint64_t)num_p : 0;
    const uint64_t n = num_n > 0 ? (uint64_t)num_n : 0;
    return p * n + p * (p > 1 ? p - 1 : 0) / 2 + n * (n > 1 ? n - 1 : 0) / 2;
}

static long bgj_epi8_search0_profile_ndp(long num_p, long num_n)
{
    return num_p * num_n + (num_p - 8) * num_p / 2 + (num_n - 8) * num_n / 2;
}

static double bgj_epi8_wall_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int bgj_epi8_parse_env_double(const char *name, double *value)
{
    const char *env = getenv(name);
    if (!env || !env[0]) return 0;

    errno = 0;
    char *end = NULL;
    const double parsed = strtod(env, &end);
    if (errno || end == env || (end && *end) || !isfinite(parsed)) return 0;

    *value = parsed;
    return 1;
}

static double bgj1_epi8_default_cuda_bucket_target(long csd)
{
    (void)csd;
    return 12288.0;
}

static long bgj1_epi8_search_threads_runtime(long threads)
{
#if defined(HAVE_CUDA)
    if (bgj_cuda_search_requested()) {
        const char *env = getenv("BGJ_CUDA_SEARCH_THREADS");
        if (env && env[0]) {
            char *end = NULL;
            long parsed = strtol(env, &end, 10);
            if (end != env && parsed > 0) {
                if (parsed > threads) return threads;
                return parsed;
            }
        }
        return threads;
    }
#endif
    return threads;
}

static double bgj1_epi8_bucket_alpha_runtime(long csd, long num_vec)
{
    double alpha = BGJ1_EPI8_BUCKET_ALPHA;
    double value = 0.0;

    if (bgj_epi8_parse_env_double("BGJ1_EPI8_BUCKET_ALPHA", &value) ||
        bgj_epi8_parse_env_double("BGJ_EPI8_BUCKET_ALPHA", &value)) {
        if (value > 0.0 && value < 1.0) alpha = value;
    } else if (bgj_epi8_parse_env_double("BGJ1_EPI8_BUCKET_TARGET_SIZE", &value) ||
               bgj_epi8_parse_env_double("BGJ_EPI8_BUCKET_TARGET_SIZE", &value)
#if defined(HAVE_CUDA)
               || (bgj_cuda_search_requested() && csd >= 40 &&
                   (value = bgj1_epi8_default_cuda_bucket_target(csd)) > 0.0)
#endif
    ) {
        if (value > 0.0 && csd > 0 && num_vec > 0) {
            double ratio = value / (double)num_vec;
            if (ratio > 0.95) ratio = 0.95;
            if (ratio < 1e-9) ratio = 1e-9;

            const double alpha2 = 1.0 - pow(ratio, 2.0 / (double)csd);
            if (alpha2 > 0.0 && alpha2 < 1.0) alpha = sqrt(alpha2);
        }
    }

    if (alpha < 0.01) alpha = 0.01;
    if (alpha > 0.95) alpha = 0.95;
    return alpha;
}

///////////////// bgj_profile_data_t impl /////////////////

template <uint32_t nb>
void bgj_profile_data_t<nb>::init(Pool_epi8_t<nb> *_p, long _log_level) {
    p = _p;
    log_level = _log_level;
    pthread_spin_init(&profile_lock, PTHREAD_PROCESS_SHARED);
    gettimeofday(&bgj_start_time, NULL);
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::initial_log(int bgj) {
    if (log_level == 0 && p->CSD > MIN_LOG_CSD){
    fprintf(log_err, "begin bgj%d sieve on context [%ld, %ld], gh = %.2f, pool size = %ld, %ld threads will be used",
                    bgj, p->index_l, p->index_r, sqrt(p->gh2), p->num_vec, p->num_threads);
    }
    if (log_level >= 1 && p->CSD > MIN_LOG_CSD) {
        fprintf(log_out, "begin bgj%d sieve, sieving dimension = %ld, pool size = %ld\n", bgj, p->CSD, p->num_vec);
        if (log_level >= 3 && p->CSD > MIN_LOG_CSD){
            if (bgj == 1) {
                fprintf(log_out, "bucket_alpha = %f\n", BGJ1_EPI8_BUCKET_ALPHA);
                fprintf(log_out, "bucketing_batch_size = %d\n", BGJ1_EPI8_BUCKET_BATCHSIZE);
            }
            if (bgj == 2) {
                fprintf(log_out, "bucket0_batchsize = %d\n", BGJ2_EPI8_BUCKET0_BATCHSIZE);
                fprintf(log_out, "bucket1_batchsize = %d\n", BGJ2_EPI8_BUCKET1_BATCHSIZE);
                fprintf(log_out, "bucket0_alpha = %f\n", BGJ2_EPI8_BUCKET0_ALPHA);
                fprintf(log_out, "bucket1_alpha = %f\n", BGJ2_EPI8_BUCKET1_ALPHA);
                fprintf(log_out, "bucket0_reuse_alpha = %f\n", BGJ2_EPI8_REUSE_ALPHA);
            }
            if (bgj == 3) {
                fprintf(log_out, "bucket0_batchsize = %d\n", BGJ3_EPI8_BUCKET0_BATCHSIZE);
                fprintf(log_out, "bucket1_batchsize = %d\n", BGJ3_EPI8_BUCKET1_BATCHSIZE);
                fprintf(log_out, "bucket2_batchsize = %d\n", BGJ3_EPI8_BUCKET2_BATCHSIZE);
                fprintf(log_out, "bucket0_alpha = %f\n", BGJ3_EPI8_BUCKET0_ALPHA);
                fprintf(log_out, "bucket1_alpha = %f\n", BGJ3_EPI8_BUCKET1_ALPHA);
                fprintf(log_out, "bucket2_alpha = %f\n", BGJ3_EPI8_BUCKET2_ALPHA);
                fprintf(log_out, "bucket0_reuse0_alpha = %f\n", BGJ3_EPI8_REUSE0_ALPHA);
                fprintf(log_out, "bucket1_reuse1_alpha = %f\n", BGJ3_EPI8_REUSE1_ALPHA);
            }
        }
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::epoch_initial_log(int32_t goal_norm) {
    if (log_level >= 2 && p->CSD > MIN_LOG_CSD) fprintf(log_out, "epoch %ld, goal_norm = %.2f\n", num_epoch-1, sqrt(2.0 * goal_norm));
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::bucket0_log() {
    if (log_level >= 3 && p->CSD > MIN_LOG_CSD) {
        double speed = p->CSD * 2 * bucket0_ndp / bucket0_time / 1073741824.0;
        fprintf(log_out, "bucket0 done, avg bucket size = %ld, num_dp = %ld\n", sum_bucket0_size/num_bucket0, bucket0_ndp);
        fprintf(log_out, "bucket0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/bucket0_time, speed);
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::bucket1_log() {
    if (log_level >= 3 && p->CSD > MIN_LOG_CSD) {
        double speed = p->CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0;
        fprintf(log_out, "bucket1 done, avg bucket size = %ld, num_dp = %ld\n", sum_bucket1_size/num_bucket1, bucket1_ndp);
        fprintf(log_out, "bucket1 speed: %f bucket/s, %f GFLOPS\n", num_bucket1/bucket1_time, speed);
        fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n", 
                try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::search0_log() {
    if (log_level >= 3 && p->CSD > MIN_LOG_CSD) {
        double speed = p->CSD * 2 * search0_ndp / search0_time / 1073741824.0;
        fprintf(log_out, "search0 done, num_dp = %ld, %ld solutions found\n", search0_ndp, succ_add2 + succ_add3);
        fprintf(log_out, "search0 speed: %.2f bucket/s, %.2f GFLOPS\n", num_bucket0/search0_time, speed);
        fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n",
                try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
        if (cuda_single_bucket0 || cuda_batch_bucket0 || cuda_fallback_bucket0) {
            fprintf(log_out, "cuda search0: single=%lu buckets/%lu dp in %.6fs, batch=%lu calls/%lu buckets/%lu dp in %.6fs, cred=%.6fs, fallback=%lu buckets/%lu dp in %.6fs\n",
                    cuda_single_bucket0, cuda_single_ndp0, cuda_single_time0,
                    cuda_batch_call0, cuda_batch_bucket0, cuda_batch_ndp0, cuda_batch_time0,
                    cuda_cred_time0, cuda_fallback_bucket0, cuda_fallback_ndp0, cuda_fallback_time0);
        }
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::search1_log() {
    if (log_level >= 3 && p->CSD > MIN_LOG_CSD) {
        double speed = p->CSD * 2 * search1_ndp/search1_time/1073741824.0;
        fprintf(log_out, "search1 done, num_dp = %ld, %ld solutions found\n", search1_ndp, succ_add2 + succ_add3);
        fprintf(log_out, "search1 speed: %f bucket/s, %f GFLOPS\n", num_bucket1/search1_time, speed);
        fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n", 
                try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::one_epoch_log(int bgj) {
    if (log_level >= 1 && p->CSD > MIN_LOG_CSD) {
        if (bgj == 1) {
            fprintf(log_out, "solution collect done, found %ld solutions in %ld buckets, bucket time = %fs, search time = %fs\n",
                            succ_add2 + succ_add3, num_bucket0, bucket0_time, search0_time);
            if (log_level >= 2) {
                double speed0;
                speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0;
                fprintf(log_out, "bucket0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/bucket0_time, speed0);
                speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0;
                fprintf(log_out, "search0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/search0_time, speed0);
                fprintf(log_out, "num bucket = %ld, avg bucket size = %ld\n", num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)));
                fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol\n", (long)(bucket0_ndp/(0.000001+succ_add2+succ_add3)), (long)(search0_ndp/(0.00001+succ_add2+succ_add3)));
                fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n",
                        try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
                if (cuda_single_bucket0 || cuda_batch_bucket0 || cuda_fallback_bucket0) {
                    fprintf(log_out, "cuda search0: single=%lu buckets/%lu dp in %.6fs, batch=%lu calls/%lu buckets/%lu dp in %.6fs, cred=%.6fs, fallback=%lu buckets/%lu dp in %.6fs\n",
                            cuda_single_bucket0, cuda_single_ndp0, cuda_single_time0,
                            cuda_batch_call0, cuda_batch_bucket0, cuda_batch_ndp0, cuda_batch_time0,
                            cuda_cred_time0, cuda_fallback_bucket0, cuda_fallback_ndp0, cuda_fallback_time0);
                }
            }
        }
        if (bgj == 2) {
            double speed0, speed1;
            fprintf(log_out, "solution collect done, found %ld solutions in %ld buckets, "
                            "bucket-0 time = %fs, bucket-1 time = %fs, search-0 time = %fs, search-1 time = %fs\n",
                     succ_add2 + succ_add3, num_bucket0 + num_bucket1, bucket0_time, bucket1_time, search0_time, search1_time);
            if (log_level >= 2) {
                speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0;
                speed1 = p->CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0;
                fprintf(log_out, "bucket0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/bucket0_time, speed0);
                fprintf(log_out, "bucket1 speed: %f bucket/s, %f GFLOPS\n", num_bucket1/bucket1_time, speed1);        
                speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0;
                speed1 = p->CSD * 2 * search1_ndp/search1_time/1073741824.0;
                fprintf(log_out, "search0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/search0_time, speed0);
                fprintf(log_out, "search1 speed: %f bucket/s, %f GFLOPS\n", num_bucket1/search1_time, speed1);
                fprintf(log_out, "num bucket0 = %ld, avg bucket0 size = %ld, num bucket1 = %ld, avg bucket1 size = %ld\n", 
                                num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)), 
                                num_bucket1, (long)(sum_bucket1_size/(0.000001+num_bucket1)));
                fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol\n", 
                                (long) ((bucket0_ndp + bucket1_ndp) / (0.000001 + succ_add2 + succ_add3)), 
                                (long) ((search0_ndp + search1_ndp) / (0.000001 + succ_add2 + succ_add3)));
                fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n", 
                        try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
            }
        }
        if (bgj == 3) {
            double speed0, speed1, speed2;
            fprintf(log_out, "solution collect done, found %ld solutions in %ld buckets, "
                            "b0: %.3fs, b1: %.3fs, b2: %.3fs, s0: %.3fs, s1: %.3fs, s2: %.3fs\n",
                     succ_add2 + succ_add3, num_bucket0 + num_bucket1 + num_bucket2 + num_r0 + num_r1, 
                     bucket0_time, bucket1_time, bucket2_time, search0_time, search1_time, search2_time);
            if (log_level >= 2) {
                speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0;
                speed1 = p->CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0;
                speed2 = p->CSD * 2 * bucket2_ndp/bucket2_time/1073741824.0;
                fprintf(log_out, "b0: %.2f bucket/s (%.3f GFLOPS), b1: %.2f bucket/s (%.3f GFLOPS), b2: %.2f bucket/s (%.3f GFLOPS)\n",
                            num_bucket0/bucket0_time, speed0, num_bucket1/bucket1_time, speed1, num_bucket2/bucket2_time, speed2);
                speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0;
                speed1 = p->CSD * 2 * search1_ndp/search1_time/1073741824.0;
                speed2 = p->CSD * 2 * search2_ndp/search2_time/1073741824.0;
                fprintf(log_out, "s0: %.2f bucket/s (%.3f GFLOPS), s1: %.2f bucket/s (%.3f GFLOPS), "
                                "s2: %.2f bucket/s (%.3f GFLOPS)\n",
                                num_bucket0/search0_time, speed0, num_bucket1/search1_time, speed1, 
                                num_bucket2/search2_time, speed2);
                fprintf(log_out, "nb0 = %ld(%ld), nb1 = %ld(%ld), nb2 = %ld(%ld), nbr0 = %ld(%ld), nbr1 = %ld(%ld)\n", 
                                num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)), 
                                num_bucket1, (long)(sum_bucket1_size/(0.000001+num_bucket1)), 
                                num_bucket2, (long)(sum_bucket2_size/(0.000001+num_bucket2)),
                                num_r0, (long)(sum_r0_size/(0.000001+num_r0)),
                                num_r1, (long)(sum_r1_size/(0.000001+num_r1)));
                fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol\n",
                                (long) ((bucket0_ndp + bucket1_ndp + bucket2_ndp) / (0.000001 + succ_add2 + succ_add3)),
                                (long) ((search0_ndp + search1_ndp + search2_ndp) / (0.000001 + succ_add2 + succ_add3)));
                fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n",
                        try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
            }
        }
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::insert_log(uint64_t num_total_sol, double insert_time) {
    if (log_level >= 2 && p->CSD > MIN_LOG_CSD){
        fprintf(log_out, "insert %ld solutions in %fs\n", num_total_sol, insert_time);
    }
    if (log_level == 0 && p->CSD > MIN_LOG_CSD){
        fprintf(log_err, ".");
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::insert_inner_log(uint64_t *length_stat, uint64_t num_linfty_failed, uint64_t num_l2_failed, uint64_t num_not_try) {
    if (p->CSD <= MIN_LOG_CSD) return;
    fprintf(log_out, "length_stat = [");
    for (long i = 90; i < 110; i++) fprintf(log_out, "%lu ", length_stat[i]);
    fprintf(log_out, "%lu]\n", length_stat[110]);
    fprintf(log_out, "num_linfty_failed = %lu, num_l2_failed = %lu, num_not_try = %lu\n", num_linfty_failed, num_l2_failed, num_not_try);
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::final_log(int bgj, long sieving_stucked) {
    if (log_level >= 1 && p->CSD > MIN_LOG_CSD) {
        if (sieving_stucked){
            fprintf(log_out, "sieving stucked, aborted.\n");
        } else {
            fprintf(log_out, "sieving done.\n");
        }
        if (bgj == 1) {
            double speed0;
            fprintf(log_out, "bucket time = %fs, search time = %fs, sort time = %fs, insert time = %fs\n",
                     bucket0_time, search0_time, sort_time, insert_time);
            speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0;
            fprintf(log_out, "bucket0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/bucket0_time, speed0);
            speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0;
            fprintf(log_out, "search0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/search0_time, speed0);
            fprintf(log_out, "num bucket = %ld, avg bucket size = %ld\n", num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)));
            fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol\n", (long)(bucket0_ndp/(0.000001+succ_add2+succ_add3)), (long)(search0_ndp/(0.00001+succ_add2+succ_add3)));
            fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n\n\n",
                    try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
            if (cuda_single_bucket0 || cuda_batch_bucket0 || cuda_fallback_bucket0) {
                fprintf(log_out, "cuda search0: single=%lu buckets/%lu dp in %.6fs, batch=%lu calls/%lu buckets/%lu dp in %.6fs, cred=%.6fs, fallback=%lu buckets/%lu dp in %.6fs\n\n",
                        cuda_single_bucket0, cuda_single_ndp0, cuda_single_time0,
                        cuda_batch_call0, cuda_batch_bucket0, cuda_batch_ndp0, cuda_batch_time0,
                        cuda_cred_time0, cuda_fallback_bucket0, cuda_fallback_ndp0, cuda_fallback_time0);
            }
        }
        if (bgj == 2) {
            double speed0, speed1;
            fprintf(log_out, "bucket-0 time = %fs, bucket-1 time = %fs, search-0 time = %fs, search-1 time = %fs\n", bucket0_time, bucket1_time, search0_time, search1_time);
            fprintf(log_out, "sort time = %fs, insert time = %fs\n", sort_time, insert_time);
            speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0;
            speed1 = p->CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0;
            fprintf(log_out, "bucket0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/bucket0_time, speed0);
            fprintf(log_out, "bucket1 speed: %f bucket/s, %f GFLOPS\n", num_bucket1/bucket1_time, speed1);  
            speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0;
            speed1 = p->CSD * 2 * search1_ndp/search1_time/1073741824.0;
            fprintf(log_out, "search0 speed: %f bucket/s, %f GFLOPS\n", num_bucket0/search0_time, speed0);
            fprintf(log_out, "search1 speed: %f bucket/s, %f GFLOPS\n", num_bucket1/search1_time, speed1);
            fprintf(log_out, "num bucket0 = %ld, avg bucket0 size = %ld, num bucket1 = %ld, avg bucket1 size = %ld\n", 
                    num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)), num_bucket1, (long)(sum_bucket1_size/(0.000001+num_bucket1)));
            fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol\n", 
                (long)((bucket0_ndp+bucket1_ndp) / (succ_add2 + succ_add3 + 0.000001)), 
                (long)((search0_ndp+search1_ndp) / (succ_add2 + succ_add3 + 0.000001)));
            fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n\n\n", 
                    try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
        }
        if (bgj == 3) {
            double speed0, speed1, speed2, speedr0, speedr1;
            fprintf(log_out, "solution collect done, found %ld solutions in %ld buckets, "
                            "b0: %.3fs, b1: %.3fs, b2: %.3fs, s0: %.3fs, s1: %.3fs, s2: %.3fs, sort: %.3fs, insert: %.3fs\n",
                     succ_add2 + succ_add3, num_bucket0 + num_bucket1 + num_bucket2 + num_r0 + num_r1, 
                     bucket0_time, bucket1_time, bucket2_time, search0_time, search1_time, search2_time, 
                     sort_time, insert_time);
            speed0 = p->CSD * 2 * bucket0_ndp/bucket0_time/1073741824.0;
            speed1 = p->CSD * 2 * bucket1_ndp/bucket1_time/1073741824.0;
            speed2 = p->CSD * 2 * bucket2_ndp/bucket2_time/1073741824.0;
            fprintf(log_out, "b0: %.2f bucket/s (%.3f GFLOPS), b1: %.2f bucket/s (%.3f GFLOPS), b2: %.2f bucket/s (%.3f GFLOPS)\n",
                            num_bucket0/bucket0_time, speed0, num_bucket1/bucket1_time, speed1, num_bucket2/bucket2_time, speed2);
            speed0 = p->CSD * 2 * search0_ndp/search0_time/1073741824.0;
            speed1 = p->CSD * 2 * search1_ndp/search1_time/1073741824.0;
            speed2 = p->CSD * 2 * search2_ndp/search2_time/1073741824.0;
            fprintf(log_out, "s0: %.2f bucket/s (%.3f GFLOPS), s1: %.2f bucket/s (%.3f GFLOPS), "
                            "s2: %.2f bucket/s (%.3f GFLOPS)\n",
                            num_bucket0/search0_time, speed0, num_bucket1/search1_time, speed1, 
                            num_bucket2/search2_time, speed2);
            fprintf(log_out, "nb0 = %ld(%ld), nb1 = %ld(%ld), nb2 = %ld(%ld), nbr0 = %ld(%ld), nbr1 = %ld(%ld)\n", 
                                num_bucket0, (long)(sum_bucket0_size/(0.000001+num_bucket0)), 
                                num_bucket1, (long)(sum_bucket1_size/(0.000001+num_bucket1)), 
                                num_bucket2, (long)(sum_bucket2_size/(0.000001+num_bucket2)),
                                num_r0, (long)(sum_r0_size/(0.000001+num_r0)),
                                num_r1, (long)(sum_r1_size/(0.000001+num_r1)));
            fprintf(log_out, "bucket cost = %ld dp/sol, search cost = %ld dp/sol\n",
                    (long)((bucket0_ndp+bucket1_ndp+bucket2_ndp) / (succ_add2 + succ_add3 + 0.000001)),
                    (long)((search0_ndp+search1_ndp+search2_ndp) / (succ_add2 + succ_add3 + 0.000001)));
            fprintf(log_out, "try_add2 = %ld, try_add3 = %ld, succ_add2 = %ld(1/%.2f), succ_add3 = %ld(1/%.2f)\n\n\n",
                    try_add2, try_add3, succ_add2, (try_add2+1e-20)/succ_add2, succ_add3, (try_add3+1e-20)/succ_add3);
        }
    }
    if (log_level == 0 && p->CSD > MIN_LOG_CSD){
        gettimeofday(&bgj_end_time, NULL);
        double tt = bgj_end_time.tv_sec-bgj_start_time.tv_sec+ (double)(bgj_end_time.tv_usec-bgj_start_time.tv_usec)/1000000.0;
        if (sieving_stucked) {
            fprintf(log_err, "get stucked.\n");
        } else if (p->CSD > MIN_LOG_CSD) {
            fprintf(log_err, "done, time = %.2fs\n", tt);
        }
    }
}

template <uint32_t nb>
template <bool record_dp>
void bgj_profile_data_t<nb>::pool_bucket_check(bucket_epi8_t<record_dp> **bucket_list, long num_bucket, double alpha) {
    if (log_level >= 4 && p->CSD > MIN_LOG_CSD) {
        int pass = 1;
        int64_t alpha_epi32 = round(65536.0 * 2.0 * alpha);
        for (long i = 0; i < num_bucket; i++) {
            int passn = 1, passs = 1, passdp = 1;
            bucket_epi8_t<record_dp> *bkt = bucket_list[i];
            for (long j = 0; j < bkt->num_pvec; j++) {
                uint32_t ind = bkt->pvec[j];
                uint32_t norm = bkt->pnorm[j];
                uint32_t sum = bkt->psum[j];
                int32_t rdp = 0;
                for (long l = 0; l < p->CSD; l++) 
                    rdp += (int) (p->vec + bkt->center_ind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                if (abs(rdp) <= ((alpha_epi32 * (int64_t)norm)>>16)) passdp = 0;
                if (record_dp) {
                    int32_t dp = bkt->pdot[j];
                    if (rdp != dp) {
                        //PRINT_VEC((int)(p->vec + bkt->center_ind * p->vec_length), p->CSD);
                        //PRINT_VEC((int)(p->vec + ind * p->vec_length), p->CSD);
                        //printf("rdp = %d, dp = %d\n", rdp, dp);
                        passdp = 0;
                    }
                }
                if (p->vnorm[ind] != norm) passn = 0;
                if (p->vsum[ind] != sum) passs = 0;
            }
            for (long j = 0; j < bkt->num_nvec; j++) {
                uint32_t ind = bkt->nvec[j];
                uint32_t norm = bkt->nnorm[j];
                uint32_t sum = bkt->nsum[j];
                int32_t rdp = 0;
                for (long l = 0; l < p->CSD; l++) 
                    rdp += (int) (p->vec + bkt->center_ind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                if (abs(rdp) <= ((alpha_epi32 * (int64_t)norm)>>16)) passdp = 0;
                if (record_dp) {
                    int32_t dp = bkt->ndot[j];
                    if (rdp != dp) {
                        //PRINT_VEC((int)(p->vec + bkt->center_ind * p->vec_length), p->CSD);
                        //PRINT_VEC((int)(p->vec + ind * p->vec_length), p->CSD);
                        //printf("rdp = %d, dp = %d\n", rdp, dp);
                        passdp = 0;
                    }
                }
                if (p->vnorm[ind] != norm) passn = 0;
                if (p->vsum[ind] != sum) passs = 0;
            }
            if (!passn || !passs || !passdp) {
                pass = 0;
                fprintf(log_out, "# bucket %ld: %s%s%s\n", i, (passn ? "" : "norm failed "), (passs ? "" : "sum failed "), (passdp ? "" : "dp failed "));
            }
        }
        long *count = (long *) NEW_VEC(num_bucket, sizeof(long));
        for (long i = 0; i < p->num_vec; i++) {
            for (long j = 0; j < num_bucket; j++) {
                int32_t rdp = 0;
                for (long l = 0; l < p->vec_length; l++) 
                    rdp += (int) (p->vec + bucket_list[j]->center_ind * p->vec_length)[l] * 
                            (int) (p->vec + i * p->vec_length)[l];
                if (abs(rdp) > ((alpha_epi32 * (int64_t)p->vnorm[i])>>16)) count[j]++;
            }
        }
        for (long i = 0; i < num_bucket; i++) {
            if (count[i] - 1 > bucket_list[i]->num_pvec + bucket_list[i]->num_nvec) {
                pass = 0;
                fprintf(log_out, "# bucket %ld, found %ld / %ld\n", i, bucket_list[i]->num_pvec + bucket_list[i]->num_nvec, count[i]);
            }
        }
        FREE_VEC((void *)count);
        
        if (pass) fprintf(log_out, "# buckets verified.\n");
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::subbucket_check(bucket_epi8_t<0> **bucket_list, bucket_epi8_t<0> **subbucket_list, long num_bucket, long num_subbucket, double alpha) {
    if (log_level >= 4 && p->CSD > MIN_LOG_CSD) {
        int pass = 1;
        long num_null = 0;
        int64_t alpha_epi32 = round(65536.0 * 2.0 * alpha);
        for (long k = 0; k < num_bucket; k++) {
            for (long i = k * num_subbucket; i < (k + 1) * num_subbucket; i++) {
                bucket_epi8_t<0> *bkt = bucket_list[k];
                bucket_epi8_t<0> *sbkt = subbucket_list[i];
                if (sbkt == NULL) {
                    num_null++;
                    continue;
                }
                int passn = 1, passs = 1, passdp = 1;
                uint32_t cind = sbkt->center_ind;
                for (long j = 0; j < sbkt->num_pvec; j++) {
                    uint32_t ind = sbkt->pvec[j];
                    uint32_t norm = sbkt->pnorm[j];
                    uint32_t sum = sbkt->psum[j];
                    int32_t rdp = 0;
                    for (long l = 0; l < p->CSD; l++) 
                        rdp += (int) (p->vec + cind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                    if (abs(rdp) <= ((alpha_epi32 * (int64_t)norm)>>16)) passdp = 0;
                    if (p->vnorm[ind] != norm) passn = 0;
                    if (p->vsum[ind] != sum) passs = 0;
                }
                for (long j = 0; j < sbkt->num_nvec; j++) {
                    uint32_t ind = sbkt->nvec[j];
                    uint32_t norm = sbkt->nnorm[j];
                    uint32_t sum = sbkt->nsum[j];
                    int32_t rdp = 0;
                    for (long l = 0; l < p->CSD; l++) 
                        rdp += (int) (p->vec + cind * p->vec_length)[l] * (int) (p->vec + ind * p->vec_length)[l];
                    if (abs(rdp) <= ((alpha_epi32 * (int64_t)norm)>>16)) passdp = 0;
                    if (p->vnorm[ind] != norm) passn = 0;
                    if (p->vsum[ind] != sum) passs = 0;
                }
                
                if (!passn || !passs || !passdp) {
                    pass = 0;
                    fprintf(log_out, "# bucket %ld: %s%s%s\n", i, (passn ? "" : "norm failed "), (passs ? "" : "sum failed "), (passdp ? "" : "dp failed "));
                }

                long countp = 0, countn = 0;
                for (long i = 0; i < bkt->num_pvec; i++) {
                    int32_t rdp = 0;
                    for (long l = 0; l < p->vec_length; l++) 
                        rdp += (int) (p->vec + sbkt->center_ind * p->vec_length)[l] * 
                                (int) (p->vec + bkt->pvec[i] * p->vec_length)[l];
                    if (rdp > ((alpha_epi32 * (int64_t)bkt->pnorm[i])>>16)) countp++;
                }
                for (long i = 0; i < bkt->num_nvec; i++) {
                    int32_t rdp = 0;
                    for (long l = 0; l < p->vec_length; l++) 
                        rdp += (int) (p->vec + sbkt->center_ind * p->vec_length)[l] * 
                                (int) (p->vec + bkt->nvec[i] * p->vec_length)[l];
                    if (-rdp > ((alpha_epi32 * (int64_t)bkt->nnorm[i])>>16)) countn++;
                }
                if (countp - 1 > sbkt->num_pvec) {
                    pass = 0;
                    fprintf(log_out, "# subbucket %ld of bucket %ld, found %ld / %ld pvec\n", i, k, sbkt->num_pvec, countp-1);
                }
                if (countn > sbkt->num_nvec) {
                    pass = 0;
                    fprintf(log_out, "# subbucket %ld of bucket %ld, found %ld / %ld pvec\n", i, k, sbkt->num_nvec, countn);
                }
            }
        }
        if (num_null) fprintf(log_out, "# warning: %ld of %ld buckets duplicated\n", num_null, num_subbucket * num_bucket);
        if (pass) fprintf(log_out, "# buckets verified.\n");
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::sol_check(sol_list_epi8_t **sol_list, long num, int32_t goal_norm, UidHashTable *uid) {
    if (log_level >= 4 && p->CSD > MIN_LOG_CSD) {
        int pass = 1;
        for (long ind = 0; ind < num; ind++) {
            int passu = 1, passn = 1;
            sol_list_epi8_t *sol = sol_list[ind];
            for (long i = 0; i < sol->num_a; i++) {
                uint32_t src1 = sol_list[ind]->a_list[2*i];
                uint32_t src2 = sol_list[ind]->a_list[2*i+1];
                int32_t dp = p->vdpss(src1, src2);
                int32_t ss = p->vnorm[src1] + p->vnorm[src2] + dp;
                uint64_t uu = p->vu[src1] + p->vu[src2];
                if (ss >= goal_norm) {
                    passn = 0;
                }
                if (!uid->check_uid(uu)) passu = 0;
            }
            for (long i = 0; i < sol->num_s; i++) {
                uint32_t src1 = sol_list[ind]->s_list[2*i];
                uint32_t src2 = sol_list[ind]->s_list[2*i+1];
                int32_t dp = p->vdpss(src1, src2);
                int32_t ss = p->vnorm[src1] + p->vnorm[src2] - dp;
                uint64_t uu = p->vu[src1] - p->vu[src2];
                if (ss >= goal_norm) {
                    passn = 0;
                }
                if (!uid->check_uid(uu)) passu = 0;
            }
            for (long i = 0; i < sol->num_aa; i++) {
                uint32_t src1 = sol_list[ind]->aa_list[3*i];
                uint32_t src2 = sol_list[ind]->aa_list[3*i+1];
                uint32_t src3 = sol_list[ind]->aa_list[3*i+2];
                int32_t dp1 = p->vdpss(src1, src2);
                int32_t dp2 = p->vdpss(src1, src3);
                int32_t dp3 = p->vdpss(src2, src3);
                int32_t ss = p->vnorm[src1] + p->vnorm[src2] + p->vnorm[src3] + dp1 + dp2 + dp3;
                uint64_t uu = p->vu[src1] + p->vu[src2] + p->vu[src3];
                if (ss >= goal_norm) passn = 0;
                if (!uid->check_uid(uu)) passu = 0;
            }
            for (long i = 0; i < sol->num_ss; i++) {
                uint32_t src1 = sol_list[ind]->ss_list[3*i];
                uint32_t src2 = sol_list[ind]->ss_list[3*i+1];
                uint32_t src3 = sol_list[ind]->ss_list[3*i+2];
                int32_t dp1 = p->vdpss(src1, src2);
                int32_t dp2 = p->vdpss(src1, src3);
                int32_t dp3 = p->vdpss(src2, src3);
                int32_t ss = p->vnorm[src1] + p->vnorm[src2] + p->vnorm[src3] - dp1 - dp2 + dp3;
                uint64_t uu = p->vu[src1] - p->vu[src2] - p->vu[src3];
                if (ss >= goal_norm) passn = 0;
                if (!uid->check_uid(uu)) passu = 0;
            }
            for (long i = 0; i < sol->num_sa; i++) {
                uint32_t src1 = sol_list[ind]->sa_list[3*i];
                uint32_t src2 = sol_list[ind]->sa_list[3*i+1];
                uint32_t src3 = sol_list[ind]->sa_list[3*i+2];
                int32_t dp1 = p->vdpss(src1, src2);
                int32_t dp2 = p->vdpss(src1, src3);
                int32_t dp3 = p->vdpss(src2, src3);
                int32_t ss = p->vnorm[src1] + p->vnorm[src2] + p->vnorm[src3] - dp1 + dp2 - dp3;
                uint64_t uu = p->vu[src1] - p->vu[src2] + p->vu[src3];
                if (ss >= goal_norm) passn = 0;
                if (!uid->check_uid(uu)) passu = 0;
            }
            if (!passu || !passn) {
                pass = 0;
                fprintf(log_out, "# sol list %ld: %s%s\n", ind, (passn ? "" : "norm failed "), (passu ? "" : "uid failed "));
            }
        }
        long count = p->num_vec + p->CSD + 1;
        for (long i = 0; i < num; i++) count += sol_list[i]->num_sol();
        long rcount = uid->size();
        if (count != uid->size()) {
            pass = 0;
            fprintf(log_out, "# uid error, %ld vectors, %ld uids in the table\n", count, rcount);
        }
        if (pass) fprintf(log_out, "# sol lists verified.\n");
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::report_bucket_not_used(int bgj, long num_rrem, long num_rem) {
    if (log_level >= 1 && p->CSD > MIN_LOG_CSD) {
        if (bgj == 1) fprintf(log_out, "search0: %ld buckets not used\n", num_rrem);
        if (bgj == 2) {
            if (num_rrem) fprintf(log_out, "search0: %ld buckets not used\n", num_rrem);
            if (num_rem) fprintf(log_out, "search1: %ld buckets not used\n", num_rem);
        }
    }
}

template <uint32_t nb>
void bgj_profile_data_t<nb>::combine(bgj_profile_data_t<nb> *prof) {
    num_bucket0 += prof->num_bucket0;
    num_bucket1 += prof->num_bucket1;
    num_bucket2 += prof->num_bucket2;
    sum_bucket0_size += prof->sum_bucket0_size;
    sum_bucket1_size += prof->sum_bucket1_size;
    sum_bucket2_size += prof->sum_bucket2_size;
    num_r0 += prof->num_r0;
    num_r1 += prof->num_r1;
    sum_r0_size += prof->sum_r0_size;
    sum_r1_size += prof->sum_r1_size;

    bucket0_time += prof->bucket0_time;
    bucket1_time += prof->bucket1_time;
    bucket2_time += prof->bucket2_time;
    search0_time += prof->search0_time;
    search1_time += prof->search1_time;
    search2_time += prof->search2_time;
    cuda_cred_time0 += prof->cuda_cred_time0;
    cuda_single_time0 += prof->cuda_single_time0;
    cuda_batch_time0 += prof->cuda_batch_time0;
    cuda_fallback_time0 += prof->cuda_fallback_time0;

    bucket0_ndp += prof->bucket0_ndp;
    bucket1_ndp += prof->bucket1_ndp;
    bucket2_ndp += prof->bucket2_ndp;
    search0_ndp += prof->search0_ndp;
    search1_ndp += prof->search1_ndp;
    search2_ndp += prof->search2_ndp;
    cuda_single_ndp0 += prof->cuda_single_ndp0;
    cuda_batch_ndp0 += prof->cuda_batch_ndp0;
    cuda_fallback_ndp0 += prof->cuda_fallback_ndp0;
    cuda_single_bucket0 += prof->cuda_single_bucket0;
    cuda_batch_bucket0 += prof->cuda_batch_bucket0;
    cuda_batch_call0 += prof->cuda_batch_call0;
    cuda_fallback_bucket0 += prof->cuda_fallback_bucket0;

    try_add2 += prof->try_add2;
    succ_add2 += prof->succ_add2;
    try_add3 += prof->try_add3;
    succ_add3 += prof->succ_add3;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj1_Sieve(long log_level, long lps_auto_adj){
    bgj_profile_data_t<nb> main_profile;
    main_profile.init(this, log_level);

    ///////////////// params /////////////////
    const double saturation_radius = 4.0/3.0;
    const double saturation_ratio = 0.375;
    const double one_epoch_ratio = 0.025;
    const double improve_ratio = 0.77;
    const double resort_ratio = 0.95;

    ///////////////// sort before sieve /////////////////
    TIMER_START;
    sort_cvec();
    TIMER_END;
    main_profile.sort_time += CURRENT_TIME;

    main_profile.initial_log(1);
    if (log_level >= 3 && CSD > MIN_LOG_CSD) {
        fprintf(main_profile.log_out, "runtime_bucket_alpha = %f\n",
                bgj1_epi8_bucket_alpha_runtime(CSD, num_vec));
    }

    long sieving_stucked = 0;
    double first_collect_sol = -1.0;
    int32_t max_norm = (((int32_t ) cvec[(num_vec - 1) * 3LL + 2LL]) << 1) + 1;

    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);

    ///////////////// main sieving procedure /////////////////
    while (!sieve_is_over(saturation_radius, saturation_ratio) && !sieving_stucked) {
        if (lps_auto_adj) {
            //...
        }

        main_profile.num_epoch++;
        const long goal_index = (long)(improve_ratio * num_vec);
        const int32_t goal_norm = vnorm[*((uint32_t *)(cvec + 3LL * goal_index))];
        main_profile.epoch_initial_log(goal_norm);

        bgj_profile_data_t<nb> local_profile;
        local_profile.init(this, log_level);

        ///////////////// collect solutions /////////////////
        long stucktime = 0;
        long num_total_sol = 0;
        long last_num_total_sol = 0;
        sol_list_epi8_t *sol_list[MAX_NTHREADS];
        for (long i = 0; i < num_threads; i++) sol_list[i] = new sol_list_epi8_t;
        bool rel_collection_stop = false;
        do {
            ///////////////// bucketing /////////////////
            const double alpha = bgj1_epi8_bucket_alpha_runtime(CSD, num_vec);
            TIMER_START;
            bucket_epi8_t<BGJ1_EPI8_USE_3RED> *main_bucket[BGJ1_EPI8_BUCKET_BATCHSIZE];
            if (max_norm > 65535) {
                _pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 1>(main_bucket, NULL, alpha, 0.0);
            } else {
                _pool_bucketing<BGJ1_EPI8_BUCKET_BATCHSIZE, BGJ1_EPI8_USE_3RED, BGJ1_EPI8_USE_FARAWAY_CENTER, 1, 0>(main_bucket, NULL, alpha, 0.0);
            }
            TIMER_END;
            local_profile.bucket0_time += CURRENT_TIME;
            local_profile.num_bucket0 += BGJ1_EPI8_BUCKET_BATCHSIZE;
            local_profile.bucket0_ndp += num_vec * BGJ1_EPI8_BUCKET_BATCHSIZE;
            for (long i = 0; i < BGJ1_EPI8_BUCKET_BATCHSIZE; i++) {
                local_profile.sum_bucket0_size += main_bucket[i]->num_nvec+main_bucket[i]->num_pvec;
            }
            local_profile.bucket0_log();
            local_profile.template pool_bucket_check<BGJ1_EPI8_USE_3RED>(main_bucket, BGJ1_EPI8_BUCKET_BATCHSIZE, alpha);

            ///////////////// searching /////////////////
            pthread_spinlock_t bucket_list_lock;
            pthread_spin_init(&bucket_list_lock, PTHREAD_PROCESS_SHARED);
            long nrem_bucket = BGJ1_EPI8_BUCKET_BATCHSIZE;
            #if defined(HAVE_CUDA)
            const uint32_t cuda_batch_capacity = bgj_cuda_batch_size((uint32_t)num_threads);
            const uint32_t cuda_min_batch = cuda_batch_capacity < 4 ? cuda_batch_capacity : 4;
            const uint64_t cuda_batch_min_dots = bgj_cuda_batch_min_dots();
            #endif
            const long search_threads = bgj1_epi8_search_threads_runtime(num_threads);
            TIMER_START;
            #pragma omp parallel for
            for (long thread = 0; thread < search_threads; thread++){
                long __search0_ndp = 0;
                double __cuda_cred_time0 = 0.0;
                double __cuda_single_time0 = 0.0;
                double __cuda_batch_time0 = 0.0;
                double __cuda_fallback_time0 = 0.0;
                uint64_t __cuda_single_ndp0 = 0;
                uint64_t __cuda_batch_ndp0 = 0;
                uint64_t __cuda_fallback_ndp0 = 0;
                uint64_t __cuda_single_bucket0 = 0;
                uint64_t __cuda_batch_bucket0 = 0;
                uint64_t __cuda_batch_call0 = 0;
                uint64_t __cuda_fallback_bucket0 = 0;
                while (nrem_bucket) {
                    bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt = NULL;
                    pthread_spin_lock(&bucket_list_lock);
                    if (nrem_bucket > 0) {
                        bkt = main_bucket[nrem_bucket-1];
                        nrem_bucket--;
                        if (local_profile.succ_add2 + local_profile.succ_add3 > num_empty + sorted_index - goal_index) {
                            bkt = NULL;
                            local_profile.report_bucket_not_used(1, nrem_bucket);
                            nrem_bucket = 0;
                        }
                    }
                    pthread_spin_unlock(&bucket_list_lock);
                    if (bkt == NULL) continue;
                    bool cred_done = false;
                    bool cuda_done = false;
                    #if defined(HAVE_CUDA)
                    if (bgj_cuda_search_requested()) {
                        if (cuda_batch_capacity > 1 &&
                            bgj_epi8_bucket_pair_dots(bkt->num_pvec, bkt->num_nvec) >= cuda_batch_min_dots) {
                            bucket_epi8_t<BGJ1_EPI8_USE_3RED> *cuda_batch[BGJ1_EPI8_BUCKET_BATCHSIZE];
                            long cuda_num_bucket = 1;
                            cuda_batch[0] = bkt;

                            pthread_spin_lock(&bucket_list_lock);
                            while (cuda_num_bucket < (long)cuda_batch_capacity &&
                                   cuda_num_bucket < BGJ1_EPI8_BUCKET_BATCHSIZE &&
                                   nrem_bucket > 0) {
                                if (local_profile.succ_add2 + local_profile.succ_add3 >
                                    num_empty + sorted_index - goal_index) {
                                    local_profile.report_bucket_not_used(1, nrem_bucket);
                                    nrem_bucket = 0;
                                    break;
                                }
                                cuda_batch[cuda_num_bucket] = main_bucket[nrem_bucket - 1];
                                nrem_bucket--;
                                cuda_num_bucket++;
                            }
                            pthread_spin_unlock(&bucket_list_lock);

                            if (cuda_num_bucket >= (long)cuda_min_batch) {
                                long batch_ndp_long = 0;
                                uint64_t batch_ndp = 0;
                                for (long i = 0; i < cuda_num_bucket; i++) {
                                    const long ndp = bgj_epi8_search0_profile_ndp(cuda_batch[i]->num_pvec,
                                                                                  cuda_batch[i]->num_nvec);
                                    batch_ndp_long += ndp;
                                    if (ndp > 0) batch_ndp += (uint64_t)ndp;
                                }
                                double t0 = bgj_epi8_wall_time();
                                for (long i = 0; i < cuda_num_bucket; i++) {
                                    _search_cred<BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                }
                                __cuda_cred_time0 += bgj_epi8_wall_time() - t0;
                                t0 = bgj_epi8_wall_time();
                                cuda_done = _search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(cuda_batch,
                                                                                           cuda_num_bucket,
                                                                                           sol_list[thread],
                                                                                           goal_norm,
                                                                                           &local_profile) > 0;
                                __cuda_batch_time0 += bgj_epi8_wall_time() - t0;
                                __cuda_batch_call0++;
                                if (!cuda_done) {
                                    t0 = bgj_epi8_wall_time();
                                    for (long i = 0; i < cuda_num_bucket; i++) {
                                        _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                        _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                        _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                    }
                                    __cuda_fallback_time0 += bgj_epi8_wall_time() - t0;
                                    __cuda_fallback_bucket0 += (uint64_t)cuda_num_bucket;
                                    __cuda_fallback_ndp0 += batch_ndp;
                                } else {
                                    __cuda_batch_bucket0 += (uint64_t)cuda_num_bucket;
                                    __cuda_batch_ndp0 += batch_ndp;
                                }
                                __search0_ndp += batch_ndp_long;
                                continue;
                            } else if (cuda_num_bucket > 1) {
                                for (long i = 0; i < cuda_num_bucket; i++) {
                                    const long ndp_long = bgj_epi8_search0_profile_ndp(cuda_batch[i]->num_pvec,
                                                                                       cuda_batch[i]->num_nvec);
                                    const uint64_t ndp = ndp_long > 0 ? (uint64_t)ndp_long : 0;
                                    double t0 = bgj_epi8_wall_time();
                                    _search_cred<BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                    __cuda_cred_time0 += bgj_epi8_wall_time() - t0;
                                    t0 = bgj_epi8_wall_time();
                                    const bool one_cuda_done =
                                        _search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i],
                                                                                  sol_list[thread],
                                                                                  goal_norm,
                                                                                  &local_profile) > 0;
                                    __cuda_single_time0 += bgj_epi8_wall_time() - t0;
                                    if (!one_cuda_done) {
                                        t0 = bgj_epi8_wall_time();
                                        _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                        _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                        _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(cuda_batch[i], sol_list[thread], goal_norm, &local_profile);
                                        __cuda_fallback_time0 += bgj_epi8_wall_time() - t0;
                                        __cuda_fallback_bucket0++;
                                        __cuda_fallback_ndp0 += ndp;
                                    } else {
                                        __cuda_single_bucket0++;
                                        __cuda_single_ndp0 += ndp;
                                    }
                                    __search0_ndp += ndp_long;
                                }
                                continue;
                            }
                        }
                        double t0 = bgj_epi8_wall_time();
                        _search_cred<BGJ1_EPI8_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                        __cuda_cred_time0 += bgj_epi8_wall_time() - t0;
                        cred_done = true;
                        t0 = bgj_epi8_wall_time();
                        cuda_done = _search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile) > 0;
                        __cuda_single_time0 += bgj_epi8_wall_time() - t0;
                        if (cuda_done) {
                            const long ndp_long = bgj_epi8_search0_profile_ndp(bkt->num_pvec, bkt->num_nvec);
                            __cuda_single_bucket0++;
                            if (ndp_long > 0) __cuda_single_ndp0 += (uint64_t)ndp_long;
                        }
                    }
                    #endif
                    if (!cuda_done) {
                        const long ndp_long = bgj_epi8_search0_profile_ndp(bkt->num_pvec, bkt->num_nvec);
                        if (!cred_done) {
                            const double t0 = bgj_epi8_wall_time();
                            _search_cred<BGJ1_EPI8_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                            #if defined(HAVE_CUDA)
                            if (bgj_cuda_search_requested()) __cuda_cred_time0 += bgj_epi8_wall_time() - t0;
                            #endif
                        }
                        const double t0 = bgj_epi8_wall_time();
                        _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                        _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                        _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ1_EPI8_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                        #if defined(HAVE_CUDA)
                        if (bgj_cuda_search_requested()) {
                            __cuda_fallback_time0 += bgj_epi8_wall_time() - t0;
                            __cuda_fallback_bucket0++;
                            if (ndp_long > 0) __cuda_fallback_ndp0 += (uint64_t)ndp_long;
                        }
                        #endif
                    }
                    __search0_ndp += bgj_epi8_search0_profile_ndp(bkt->num_pvec, bkt->num_nvec);
                }
                pthread_spin_lock(&local_profile.profile_lock);
                local_profile.search0_ndp += __search0_ndp;
                local_profile.cuda_cred_time0 += __cuda_cred_time0;
                local_profile.cuda_single_time0 += __cuda_single_time0;
                local_profile.cuda_batch_time0 += __cuda_batch_time0;
                local_profile.cuda_fallback_time0 += __cuda_fallback_time0;
                local_profile.cuda_single_ndp0 += __cuda_single_ndp0;
                local_profile.cuda_batch_ndp0 += __cuda_batch_ndp0;
                local_profile.cuda_fallback_ndp0 += __cuda_fallback_ndp0;
                local_profile.cuda_single_bucket0 += __cuda_single_bucket0;
                local_profile.cuda_batch_bucket0 += __cuda_batch_bucket0;
                local_profile.cuda_batch_call0 += __cuda_batch_call0;
                local_profile.cuda_fallback_bucket0 += __cuda_fallback_bucket0;
                pthread_spin_unlock(&local_profile.profile_lock);
            }
            TIMER_END;
            local_profile.search0_time += CURRENT_TIME;
            local_profile.search0_log();
            local_profile.sol_check(sol_list, num_threads, goal_norm, uid);

            // free buckets
            #pragma omp parallel for
            for (long i = 0; i < BGJ1_EPI8_BUCKET_BATCHSIZE; i++) {
                delete main_bucket[i];
            }

            // check if we get stucked or finished
            do {
                num_total_sol = 0;
                for (long i = 0; i < num_threads; i++){
                    num_total_sol += sol_list[i]->num_sol();
                }
                if (first_collect_sol == -1.0) first_collect_sol = (num_total_sol + 0.0);
                if (num_total_sol - last_num_total_sol <= (1 + stucktime) * first_collect_sol * 0.01){
                    stucktime++;
                } else {
                    stucktime = 0;
                    last_num_total_sol = num_total_sol;
                }
                if (num_total_sol > one_epoch_ratio * num_vec) rel_collection_stop = true;
                if (stucktime > MAX_STUCK_TIME) {
                    sieving_stucked = 1;
                    rel_collection_stop = true;
                } 
            } while (0);
        } while (!rel_collection_stop);
        
        local_profile.one_epoch_log(1);
        local_profile.sol_check(sol_list, num_threads, goal_norm, uid);
        main_profile.combine(&local_profile);

        ///////////////// inserting /////////////////
        TIMER_START;
        uint64_t num_total_insert;
        if (log_level >= 3) {
            num_total_insert = _pool_insert<1>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        } else {
            num_total_insert = _pool_insert<0>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        }
        TIMER_END;
        main_profile.insert_time += CURRENT_TIME;
        main_profile.insert_log(num_total_insert, CURRENT_TIME);

        if (log_level >= 4) check_pool_status(0, 1);

        if (resort_ratio * num_vec > sorted_index){
            TIMER_START;
            sort_cvec();
            TIMER_END;
            main_profile.sort_time += CURRENT_TIME;
            max_norm = (((int32_t ) cvec[(num_vec - 1) * 3LL + 2LL]) << 1) + 1;
        }

        for (long i = 0; i < num_threads; i++) delete sol_list[i];
    }

    mallopt(M_MMAP_MAX, 65536);
    mallopt(M_TRIM_THRESHOLD, 128*1024);
    main_profile.final_log(1, sieving_stucked);
    if (sieving_stucked || CSD == 80) {
        if (check_dim_lose()) {
            sieving_stucked = 0;
        } else {
            sieving_stucked = 1;
        }
    }

    return sieving_stucked;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj2_Sieve(long log_level, long lps_auto_adj){
    bgj_profile_data_t<nb> main_profile;
    main_profile.init(this, log_level);

    ///////////////// params /////////////////
    double alpha_r0 = BGJ2_EPI8_REUSE_ALPHA;
    double alpha_b0 = BGJ2_EPI8_BUCKET0_ALPHA;
    double alpha_b1 = BGJ2_EPI8_BUCKET1_ALPHA;
    const double saturation_radius = 4.0/3.0;
    const double saturation_ratio = 0.375;
    const double one_epoch_ratio = 0.025;
    const double improve_ratio = 0.73;
    const double resort_ratio = 0.95;

    ///////////////// sort before sieve /////////////////
    TIMER_START;
    sort_cvec();
    TIMER_END;
    main_profile.sort_time += CURRENT_TIME;

    main_profile.initial_log(2);

    long sieving_stucked = 0;
    double first_collect_sol = -1.0;
    int32_t max_norm = (((int32_t ) cvec[(num_vec - 1) * 3LL + 2LL]) << 1) + 1;

    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);

    ///////////////// main sieving procedure /////////////////
    while (!sieve_is_over(saturation_radius, saturation_ratio) && !sieving_stucked) {
        if (lps_auto_adj) {
            //...
        }

        main_profile.num_epoch++;
        const long goal_index = (long)(improve_ratio * num_vec);
        const int32_t goal_norm = vnorm[*((uint32_t *)(cvec + 3LL * goal_index))];
        main_profile.epoch_initial_log(goal_norm);

        bgj_profile_data_t<nb> local_profile;
        local_profile.init(this, log_level);

        ///////////////// collect solutions /////////////////
        long stucktime = 0;
        long num_total_sol = 0;
        long last_num_total_sol = 0;
        sol_list_epi8_t *sol_list[MAX_NTHREADS];
        for (long i = 0; i < num_threads; i++) sol_list[i] = new sol_list_epi8_t;
        bool rel_collection_stop = false;
        do {
            ///////////////// bucket0 /////////////////
            TIMER_START;
            bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> *rbucket[BGJ2_EPI8_BUCKET0_BATCHSIZE];
            bucket_epi8_t<0> *bucket0[BGJ2_EPI8_BUCKET0_BATCHSIZE];
            if (max_norm > 65535) {
                _pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 1>(rbucket, bucket0, alpha_r0, alpha_b0);
            } else {
                _pool_bucketing<BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_REUSE_USE_3RED, BGJ2_EPI8_USE_FARAWAY_CENTER, 0, 0>(rbucket, bucket0, alpha_r0, alpha_b0);
            }
            TIMER_END;
            local_profile.bucket0_time += CURRENT_TIME;
            local_profile.num_bucket0 += BGJ2_EPI8_BUCKET0_BATCHSIZE;
            local_profile.bucket0_ndp += num_vec * BGJ2_EPI8_BUCKET0_BATCHSIZE;
            for (long i = 0; i < BGJ2_EPI8_BUCKET0_BATCHSIZE; i++) {
                local_profile.sum_bucket0_size += bucket0[i]->num_nvec+bucket0[i]->num_pvec;
            }
            local_profile.bucket0_log();
            local_profile.template pool_bucket_check<BGJ2_EPI8_REUSE_USE_3RED>(rbucket, BGJ2_EPI8_BUCKET0_BATCHSIZE, alpha_r0);
            local_profile.template pool_bucket_check<0>(bucket0, BGJ2_EPI8_BUCKET0_BATCHSIZE, alpha_b0);

            ///////////////// bucket1 /////////////////
            TIMER_START;
            bucket_epi8_t<0> *bucket1[BGJ2_EPI8_BUCKET0_BATCHSIZE * BGJ2_EPI8_BUCKET1_BATCHSIZE] = {};
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                const long begin_ind = (thread * BGJ2_EPI8_BUCKET0_BATCHSIZE) / num_threads;
                const long end_ind = ((thread + 1) * BGJ2_EPI8_BUCKET0_BATCHSIZE) / num_threads;
                for (long ind = begin_ind; ind < end_ind; ind++) {
                    if (max_norm > 65535) {
                        _sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(
                            bucket0[ind], &bucket1[ind * BGJ2_EPI8_BUCKET1_BATCHSIZE], NULL, alpha_b1, 0.0, sol_list[thread], goal_norm, &local_profile);
                    } else {
                        _sub_bucketing<BGJ2_EPI8_BUCKET1_BATCHSIZE, BGJ2_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(
                            bucket0[ind], &bucket1[ind * BGJ2_EPI8_BUCKET1_BATCHSIZE], NULL, alpha_b1, 0.0, sol_list[thread], goal_norm, &local_profile);
                    }
                }
            }
            TIMER_END;
            local_profile.bucket1_time += CURRENT_TIME;
            local_profile.num_bucket1 += BGJ2_EPI8_BUCKET0_BATCHSIZE * BGJ2_EPI8_BUCKET1_BATCHSIZE;
            for (long i = 0; i < BGJ2_EPI8_BUCKET0_BATCHSIZE; i++) {
                local_profile.bucket1_ndp += BGJ2_EPI8_BUCKET1_BATCHSIZE * (bucket0[i]->num_pvec + bucket0[i]->num_nvec);
                for (long j = 0; j < BGJ2_EPI8_BUCKET1_BATCHSIZE; j++) {
                    if (bucket1[i * BGJ2_EPI8_BUCKET1_BATCHSIZE + j]) local_profile.sum_bucket1_size += bucket1[i * BGJ2_EPI8_BUCKET1_BATCHSIZE + j]->num_pvec + bucket1[i * BGJ2_EPI8_BUCKET1_BATCHSIZE + j]->num_nvec;
                }
            }
            local_profile.bucket1_log();
            local_profile.subbucket_check(bucket0, bucket1, BGJ2_EPI8_BUCKET0_BATCHSIZE, BGJ2_EPI8_BUCKET1_BATCHSIZE, alpha_b1);
            local_profile.sol_check(sol_list, num_threads, goal_norm, uid);

            ///////////////// search0 /////////////////
            pthread_spinlock_t rbucket_list_lock;
            pthread_spin_init(&rbucket_list_lock, PTHREAD_PROCESS_SHARED);
            long nrem_rbucket = BGJ2_EPI8_BUCKET0_BATCHSIZE;
            TIMER_START;
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                long __search0_ndp = 0;
                while (nrem_rbucket) {
                    bucket_epi8_t<BGJ2_EPI8_REUSE_USE_3RED> *bkt = NULL;
                    pthread_spin_lock(&rbucket_list_lock);
                    if (nrem_rbucket > 0) {
                        if (local_profile.succ_add2 + local_profile.succ_add3 > num_empty + sorted_index - goal_index) {
                            bkt = NULL;
                            local_profile.report_bucket_not_used(2, nrem_rbucket, 0);
                            nrem_rbucket = 0;
                        } else {
                            bkt = rbucket[nrem_rbucket-1];
                            nrem_rbucket--;
                        }
                    }
                    pthread_spin_unlock(&rbucket_list_lock);
                    if (bkt == NULL) continue;
                    _search_cred<BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    __search0_ndp += bkt->num_pvec * bkt->num_nvec;
                    __search0_ndp += (bkt->num_pvec - 8) * bkt->num_pvec / 2;
                    __search0_ndp += (bkt->num_nvec - 8) * bkt->num_nvec / 2;
                }
                pthread_spin_lock(&local_profile.profile_lock);
                local_profile.search0_ndp += __search0_ndp;
                pthread_spin_unlock(&local_profile.profile_lock);
            }
            TIMER_END;
            local_profile.search0_time += CURRENT_TIME;
            local_profile.search0_log();
            local_profile.sol_check(sol_list, num_threads, goal_norm, uid);

            ///////////////// search1 /////////////////
            pthread_spinlock_t bucket1_list_lock;
            pthread_spin_init(&bucket1_list_lock, PTHREAD_PROCESS_SHARED);
            long nrem_bucket1 = BGJ2_EPI8_BUCKET0_BATCHSIZE * BGJ2_EPI8_BUCKET1_BATCHSIZE;
            TIMER_START;
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                long __search1_ndp = 0;
                while (nrem_bucket1) {
                    bucket_epi8_t<0> *bkt = NULL;
                    pthread_spin_lock(&bucket1_list_lock);
                    if (nrem_bucket1 > 0) {
                        if (local_profile.succ_add2 + local_profile.succ_add3 > num_empty + sorted_index - goal_index) {
                            bkt = NULL;
                            local_profile.report_bucket_not_used(2, 0, nrem_bucket1);
                            nrem_bucket1 = 0;
                        } else {
                            bkt = bucket1[nrem_bucket1-1];
                            nrem_bucket1--;
                        }
                    }
                    pthread_spin_unlock(&bucket1_list_lock);
                    if (bkt == NULL) continue;
                    _search_cred<0, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bkt, sol_list[thread], goal_norm, &local_profile);
                    __search1_ndp += bkt->num_pvec * bkt->num_nvec;
                    __search1_ndp += (bkt->num_pvec - 8) * bkt->num_pvec / 2;
                    __search1_ndp += (bkt->num_nvec - 8) * bkt->num_nvec / 2;
                }
                pthread_spin_lock(&local_profile.profile_lock);
                local_profile.search1_ndp += __search1_ndp;
                pthread_spin_unlock(&local_profile.profile_lock);
            }
            TIMER_END;
            local_profile.search1_time += CURRENT_TIME;
            local_profile.search1_log();
            local_profile.sol_check(sol_list, num_threads, goal_norm, uid);

            // free buckets
            #pragma omp parallel for
            for (long i = 0; i < BGJ2_EPI8_BUCKET0_BATCHSIZE; i++) {
                delete rbucket[i];
                delete bucket0[i];
                for (long j = 0; j < BGJ2_EPI8_BUCKET1_BATCHSIZE; j++) delete bucket1[i*BGJ2_EPI8_BUCKET1_BATCHSIZE+j];
            }

            // check if we get stucked or finished
            do {
                num_total_sol = 0;
                for (long i = 0; i < num_threads; i++){
                    num_total_sol += sol_list[i]->num_sol();
                }
                if (first_collect_sol == -1.0) first_collect_sol = num_total_sol + 0.0;
                if (num_total_sol - last_num_total_sol <= (1 + stucktime) * first_collect_sol * 0.01){
                    stucktime++;
                } else {
                    stucktime = 0;
                    last_num_total_sol = num_total_sol;
                }
                if (num_total_sol > one_epoch_ratio * num_vec) rel_collection_stop = true;
                if (stucktime > MAX_STUCK_TIME) {
                    sieving_stucked = 1;
                    rel_collection_stop = true;
                } 
            } while (0);
        } while (!rel_collection_stop);

        local_profile.one_epoch_log(2);
        local_profile.sol_check(sol_list, num_threads, goal_norm, uid);
        main_profile.combine(&local_profile);

        ///////////////// inserting /////////////////
        TIMER_START;
        uint64_t num_total_insert;
        if (log_level >= 3) {
            num_total_insert = _pool_insert<1>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        } else {
            num_total_insert = _pool_insert<0>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        }
        TIMER_END;
        main_profile.insert_time += CURRENT_TIME;
        main_profile.insert_log(num_total_insert, CURRENT_TIME);

        if (log_level >= 4) check_pool_status(0, 1);

        if (resort_ratio * num_vec > sorted_index){
            TIMER_START;
            sort_cvec();
            TIMER_END;
            main_profile.sort_time += CURRENT_TIME;
            max_norm = (((int32_t ) cvec[(num_vec - 1) * 3LL + 2LL]) << 1) + 1;
        }

        for (long i = 0; i < num_threads; i++) delete sol_list[i];
    }

    mallopt(M_MMAP_MAX, 65536);
    mallopt(M_TRIM_THRESHOLD, 128*1024);
    main_profile.final_log(2, sieving_stucked);
    if (sieving_stucked || CSD == 80) {
        if (check_dim_lose()) {
            sieving_stucked = 0;
        } else {
            sieving_stucked = 1;
        }
    }

    return sieving_stucked;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj3_Sieve(long log_level, long lps_auto_adj){
    bgj_profile_data_t<nb> main_profile;
    main_profile.init(this, log_level);

    ///////////////// params /////////////////
    double alpha_r0 = BGJ3_EPI8_REUSE0_ALPHA;
    double alpha_r1 = BGJ3_EPI8_REUSE1_ALPHA;
    double alpha_b0 = BGJ3_EPI8_BUCKET0_ALPHA;
    double alpha_b1 = BGJ3_EPI8_BUCKET1_ALPHA;
    double alpha_b2 = BGJ3_EPI8_BUCKET2_ALPHA;
    const double saturation_radius = 4.0/3.0;
    const double saturation_ratio = 0.375;
    const double one_epoch_ratio = 0.025;
    const double improve_ratio = 0.73;
    const double resort_ratio = 0.95;

    ///////////////// sort before sieve /////////////////
    TIMER_START;
    sort_cvec();
    TIMER_END;
    main_profile.sort_time += CURRENT_TIME;

    main_profile.initial_log(3);

    long sieving_stucked = 0;
    double first_collect_sol = -1.0;
    int32_t max_norm = (((int32_t ) cvec[(num_vec - 1) * 3LL + 2LL]) << 1) + 1;

    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);

    ///////////////// main sieving procedure /////////////////
    while (!sieve_is_over(saturation_radius, saturation_ratio) && !sieving_stucked) {
        if (lps_auto_adj) {
            if (CSD > 105 && main_profile.num_epoch == 0) {
                alpha_r1 += 0.005;
                alpha_b0 += 0.005;
                alpha_b1 += 0.003;
                alpha_b2 += 0.007;
            }
        }

        main_profile.num_epoch++;
        const long goal_index = (long)(improve_ratio * num_vec);
        const int32_t goal_norm = vnorm[*((uint32_t *)(cvec + 3LL * goal_index))];
        main_profile.epoch_initial_log(goal_norm);

        bgj_profile_data_t<nb> local_profile;
        local_profile.init(this, log_level);

        ///////////////// collect solutions /////////////////
        long stucktime = 0;
        long num_total_sol = 0;
        long last_num_total_sol = 0;
        sol_list_epi8_t *sol_list[MAX_NTHREADS];
        for (long i = 0; i < num_threads; i++) sol_list[i] = new sol_list_epi8_t;
        bool rel_collection_stop = false;
        do {
            ///////////// bucket0 //////////////
            TIMER_START;
            bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> *rbucket0[BGJ3_EPI8_BUCKET0_BATCHSIZE];
            bucket_epi8_t<0> *bucket0[BGJ3_EPI8_BUCKET0_BATCHSIZE];
            if (max_norm > 65535) {
                _pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1>(rbucket0, bucket0, alpha_r0, alpha_b0);
            } else {
                _pool_bucketing<BGJ3_EPI8_BUCKET0_BATCHSIZE, BGJ3_EPI8_REUSE0_USE_3RED, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0>(rbucket0, bucket0, alpha_r0, alpha_b0);
            }
            TIMER_END;
            local_profile.bucket0_time += CURRENT_TIME;
            local_profile.num_bucket0 += BGJ3_EPI8_BUCKET0_BATCHSIZE;
            local_profile.num_r0 += BGJ3_EPI8_BUCKET0_BATCHSIZE;
            local_profile.bucket0_ndp += num_vec * BGJ3_EPI8_BUCKET0_BATCHSIZE;
            for (long i = 0; i < BGJ3_EPI8_BUCKET0_BATCHSIZE; i++) {
                local_profile.sum_bucket0_size += bucket0[i]->num_nvec+bucket0[i]->num_pvec;
                local_profile.sum_r0_size += rbucket0[i]->num_nvec+rbucket0[i]->num_pvec;
            }
            local_profile.template pool_bucket_check<BGJ3_EPI8_REUSE0_USE_3RED>(rbucket0, BGJ3_EPI8_BUCKET0_BATCHSIZE, alpha_r0);
            local_profile.template pool_bucket_check<0>(bucket0, BGJ3_EPI8_BUCKET0_BATCHSIZE, alpha_b0);

            ///////////// bucket1 --> search while bucket2 //////////////
            bucket_epi8_t<0> *rbucket1[BGJ3_EPI8_BUCKET0_BATCHSIZE * BGJ3_EPI8_BUCKET1_BATCHSIZE] = {};
            bucket_epi8_t<0> *bucket1[BGJ3_EPI8_BUCKET0_BATCHSIZE * BGJ3_EPI8_BUCKET1_BATCHSIZE] = {};
            bucket_epi8_t<0> *bucket2[MAX_NTHREADS * BGJ3_EPI8_BUCKET2_BATCHSIZE] = {};
            int rbucket1_nrem[BGJ3_EPI8_BUCKET0_BATCHSIZE];
            int rbucket0_nrem = BGJ3_EPI8_BUCKET0_BATCHSIZE;
            int too_many_sol = 0;
            for (long i = 0; i < BGJ3_EPI8_BUCKET0_BATCHSIZE; i++) rbucket1_nrem[i] = -1;
            pthread_spinlock_t bucket_list_lock;
            pthread_spin_init(&bucket_list_lock, PTHREAD_PROCESS_SHARED);
            #pragma omp parallel for
            for (long thread = 0; thread < num_threads; thread++) {
                // bucket1
                do {
                    const long begin_ind = (thread * BGJ3_EPI8_BUCKET0_BATCHSIZE) / num_threads;
                    const long end_ind = ((thread + 1) * BGJ3_EPI8_BUCKET0_BATCHSIZE) / num_threads;
                    long ndp = 0;
                    TIMER_START;
                    for (long ind = begin_ind; ind < end_ind; ind++) {
                        if (max_norm > 65535) {
                            _sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 1, 1>(
                                bucket0[ind], &rbucket1[ind * BGJ3_EPI8_BUCKET1_BATCHSIZE], &bucket1[ind * BGJ3_EPI8_BUCKET1_BATCHSIZE], alpha_r1, alpha_b1, sol_list[thread], goal_norm, &local_profile);
                        } else {
                            _sub_bucketing<BGJ3_EPI8_BUCKET1_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 0, 0, 1>(
                                bucket0[ind], &rbucket1[ind * BGJ3_EPI8_BUCKET1_BATCHSIZE], &bucket1[ind * BGJ3_EPI8_BUCKET1_BATCHSIZE], alpha_r1, alpha_b1, sol_list[thread], goal_norm, &local_profile);
                        }
                        ndp += BGJ3_EPI8_BUCKET1_BATCHSIZE * (bucket0[ind]->num_pvec + bucket0[ind]->num_nvec);
                        if (local_profile.log_level <= 3) delete bucket0[ind];
                        long num_bucket1_done = 0;
                        for (long j = 0; j < BGJ3_EPI8_BUCKET1_BATCHSIZE; j++) {
                            if (bucket1[ind * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]) num_bucket1_done++;
                        }
                        rbucket1_nrem[ind] = num_bucket1_done;
                    }
                    TIMER_END;
                    pthread_spin_lock(&local_profile.profile_lock);
                    local_profile.bucket1_time += CURRENT_TIME;
                    local_profile.bucket1_ndp += ndp;
                    for (long i = begin_ind; i < end_ind; i++) {
                        for (long j = 0; j < BGJ3_EPI8_BUCKET1_BATCHSIZE; j++) {
                            if (rbucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]) {
                                local_profile.num_r1++;
                                local_profile.sum_r1_size += rbucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]->num_pvec + rbucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]->num_nvec;
                            }
                            if (bucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]) {
                                local_profile.num_bucket1++;
                                local_profile.sum_bucket1_size += bucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]->num_pvec + bucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]->num_nvec;
                            }
                        }
                    }
                    local_profile.subbucket_check(bucket0+begin_ind, bucket1 + begin_ind * BGJ3_EPI8_BUCKET1_BATCHSIZE, 
                                end_ind - begin_ind, BGJ3_EPI8_BUCKET1_BATCHSIZE, alpha_b1);
                    local_profile.subbucket_check(bucket0+begin_ind, rbucket1 + begin_ind * BGJ3_EPI8_BUCKET1_BATCHSIZE, 
                                end_ind - begin_ind, BGJ3_EPI8_BUCKET1_BATCHSIZE, alpha_r1);
                    local_profile.sol_check(sol_list, num_threads, goal_norm, uid);
                    pthread_spin_unlock(&local_profile.profile_lock);
                    if (local_profile.log_level >= 4) {
                        for (long ind = begin_ind; ind < end_ind; ind++) delete bucket0[ind];
                    }
                } while (0);
                
                // now rbucket0 must be ready, but some (r)bucket1 may not be ready
                // we need to do:   search0 (process rbucket0)                                  0
                //                  search1 & bucket2 & search2 (process rbucket1 & bucket1)    1
                // each time, we get and process an rbucket0 or a pair (rbucket1, bucket1)
                for (;;) {
                    int type = -1;
                    bucket_epi8_t<BGJ3_EPI8_REUSE0_USE_3RED> *bkt3 = NULL;
                    bucket_epi8_t<0> *bkt2 = NULL;
                    bucket_epi8_t<0> *bkt = NULL;
                    pthread_spin_lock(&bucket_list_lock);
                    if (rbucket0_nrem > 0) {
                        bkt3 = rbucket0[rbucket0_nrem-1];
                        rbucket0[rbucket0_nrem-1] = NULL;
                        rbucket0_nrem--;
                        type = 0;
                    } else {
                        for (long i = 0; i < BGJ3_EPI8_BUCKET0_BATCHSIZE; i++) {
                            if (rbucket1_nrem[i] > 0) {
                                for (long j = 0; j < BGJ3_EPI8_BUCKET1_BATCHSIZE; j++) {
                                    if (bucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j]) {
                                        bkt2 = rbucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j];
                                        bkt = bucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j];
                                        bucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j] = NULL;
                                        rbucket1[i * BGJ3_EPI8_BUCKET1_BATCHSIZE + j] = NULL;
                                        rbucket1_nrem[i]--;
                                        type = 1;
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                        if (type == -1) {
                            int finished = 1;
                            for (long i = 0; i < BGJ3_EPI8_BUCKET0_BATCHSIZE; i++) {
                                if (rbucket1_nrem[i] == -1) {
                                    finished = 0;
                                    break;
                                }
                            }
                            if (finished) {
                                pthread_spin_unlock(&bucket_list_lock);
                                break;
                            }
                        }
                    }
                    pthread_spin_unlock(&bucket_list_lock);
                    if (type == -1) continue;
                    if (type == 0) {
                        TIMER_START;
                        long __search0_ndp = 0;
                        _search_cred<BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt3, sol_list[thread], goal_norm, &local_profile);
                        _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt3, sol_list[thread], goal_norm, &local_profile);
                        _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt3, sol_list[thread], goal_norm, &local_profile);
                        _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, BGJ2_EPI8_REUSE_USE_3RED, 1>(bkt3, sol_list[thread], goal_norm, &local_profile);
                        __search0_ndp += bkt3->num_pvec * bkt3->num_nvec;
                        __search0_ndp += (bkt3->num_pvec - 8) * bkt3->num_pvec / 2;
                        __search0_ndp += (bkt3->num_nvec - 8) * bkt3->num_nvec / 2;
                        TIMER_END;
                        pthread_spin_lock(&local_profile.profile_lock);
                        local_profile.search0_time += CURRENT_TIME;
                        local_profile.search0_ndp += __search0_ndp;
                        pthread_spin_unlock(&local_profile.profile_lock);
                        delete bkt3;
                    } else {
                        long __search1_ndp = 0;
                        double __search1_time = 0.0;

                        double __bucket2_time = 0.0;
                        long __bucket2_ndp = 0;
                        long __sum_bucket2_size = 0;
                        long __num_bucket2 = 0;
                        long __search2_ndp = 0;
                        double __search2_time = 0.0;
                        
                        TIMER_START;
                        _search_cred<0, 1>(bkt2, sol_list[thread], goal_norm, &local_profile);
                        _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bkt2, sol_list[thread], goal_norm, &local_profile);
                        _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bkt2, sol_list[thread], goal_norm, &local_profile);
                        _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(bkt2, sol_list[thread], goal_norm, &local_profile);
                        __search1_ndp += bkt2->num_pvec * bkt2->num_nvec;
                        __search1_ndp += (bkt2->num_pvec - 8) * bkt2->num_pvec / 2;
                        __search1_ndp += (bkt2->num_nvec - 8) * bkt2->num_nvec / 2;
                        TIMER_END;
                        __search1_time += CURRENT_TIME;
                        delete bkt2;

                        TIMER_START;
                        if (max_norm > 65535) {
                            _sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 1, 1>(
                            bkt, &bucket2[thread * BGJ3_EPI8_BUCKET2_BATCHSIZE], NULL, alpha_b2, 0.0, sol_list[thread], goal_norm, &local_profile);
                        } else {
                            _sub_bucketing<BGJ3_EPI8_BUCKET2_BATCHSIZE, BGJ3_EPI8_USE_FARAWAY_CENTER, 1, 0, 1>(
                            bkt, &bucket2[thread * BGJ3_EPI8_BUCKET2_BATCHSIZE], NULL, alpha_b2, 0.0, sol_list[thread], goal_norm, &local_profile);
                        }
                        TIMER_END;
                        __bucket2_time += CURRENT_TIME;
                        __bucket2_ndp = BGJ3_EPI8_BUCKET2_BATCHSIZE * (bkt->num_pvec + bkt->num_nvec);
                        for (long i = 0; i < BGJ3_EPI8_BUCKET2_BATCHSIZE; i++) {
                            if (bucket2[thread * BGJ3_EPI8_BUCKET2_BATCHSIZE + i]) {
                                __sum_bucket2_size += bucket2[thread * BGJ3_EPI8_BUCKET2_BATCHSIZE + i]->num_pvec + bucket2[thread * BGJ3_EPI8_BUCKET2_BATCHSIZE + i]->num_nvec;
                                __num_bucket2++;
                            }
                        }
                        TIMER_START;
                        for (long i = 0; i < BGJ3_EPI8_BUCKET2_BATCHSIZE; i++) {
                            bucket_epi8_t<0> *_bkt = bucket2[thread * BGJ3_EPI8_BUCKET2_BATCHSIZE + i];
                            if (_bkt == NULL) continue;
                            _search_cred<0, 1>(_bkt, sol_list[thread], goal_norm, &local_profile);
                            _search_np<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(_bkt, sol_list[thread], goal_norm, &local_profile);
                            _search_pp<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(_bkt, sol_list[thread], goal_norm, &local_profile);
                            _search_nn<SEARCH_L2_BLOCK, SEARCH_L1_BLOCK, 0, 1>(_bkt, sol_list[thread], goal_norm, &local_profile);
                            __search2_ndp += _bkt->num_pvec * _bkt->num_nvec;
                            __search2_ndp += (_bkt->num_pvec - 8) * _bkt->num_pvec / 2;
                            __search2_ndp += (_bkt->num_nvec - 8) * _bkt->num_nvec / 2;
                        }
                        TIMER_END;
                        __search2_time += CURRENT_TIME;

                        pthread_spin_lock(&local_profile.profile_lock);
                        local_profile.search1_time += __search1_time;
                        local_profile.search2_time += __search2_time;
                        local_profile.bucket2_time += __bucket2_time;
                        local_profile.search1_ndp += __search1_ndp;
                        local_profile.search2_ndp += __search2_ndp;
                        local_profile.bucket2_ndp += __bucket2_ndp;
                        local_profile.sum_bucket2_size += __sum_bucket2_size;
                        local_profile.num_bucket2 += __num_bucket2;
                        pthread_spin_unlock(&local_profile.profile_lock);
                        delete bkt;
                    }
                    if (local_profile.succ_add2 + local_profile.succ_add3 > num_empty + sorted_index - goal_index) {
                        pthread_spin_lock(&local_profile.profile_lock);
                        if (too_many_sol) {
                            pthread_spin_unlock(&local_profile.profile_lock);
                            break;
                        }
                        too_many_sol = 1;
                        long rb0_not_used = 0;
                        long rb1_not_used = 0;
                        for (long i = 0; i < BGJ3_EPI8_BUCKET0_BATCHSIZE; i++) {
                            if (rbucket1_nrem[i] > 0) rb1_not_used += rbucket1_nrem[i];
                            if (rbucket1_nrem[i] == -1) rb1_not_used += BGJ3_EPI8_BUCKET1_BATCHSIZE;
                        }
                        if (rbucket0_nrem > 0) rb0_not_used = rbucket0_nrem;
                        local_profile.report_bucket_not_used(2, rb0_not_used, rb1_not_used);
                        pthread_spin_unlock(&local_profile.profile_lock);
                        break;
                    }
                }
            }

            #pragma omp parallel for
            for (long i = 0; i < num_threads * BGJ3_EPI8_BUCKET2_BATCHSIZE; i++) {
                delete bucket2[i];
            }
            #pragma omp parallel for schedule (guided)
            for (long i = 0; i < BGJ3_EPI8_BUCKET0_BATCHSIZE * BGJ3_EPI8_BUCKET1_BATCHSIZE; i++) {
                if (bucket1[i]) delete bucket1[i];
                if (rbucket1[i]) delete rbucket1[i];
            }
            #pragma omp parallel for schedule (guided)
            for (long i = 0; i < BGJ3_EPI8_BUCKET0_BATCHSIZE; i++) {
                if (rbucket0[i]) delete rbucket0[i];
            }
            // check if we get stucked or finished
            do {
                num_total_sol = 0;
                for (long i = 0; i < num_threads; i++){
                    num_total_sol += sol_list[i]->num_sol();
                }
                if (first_collect_sol == -1.0) first_collect_sol = num_total_sol + 0.0;
                if (num_total_sol - last_num_total_sol <= (1 + stucktime) * first_collect_sol * 0.01){
                    stucktime++;
                } else {
                    stucktime = 0;
                    last_num_total_sol = num_total_sol;
                }
                if (num_total_sol > one_epoch_ratio * num_vec) rel_collection_stop = true;
                if (stucktime > MAX_STUCK_TIME) {
                    sieving_stucked = 1;
                    rel_collection_stop = true;
                } 
            } while (0);
        } while (!rel_collection_stop);
        local_profile.bucket1_time /= num_threads;
        local_profile.bucket2_time /= num_threads;
        local_profile.search0_time /= num_threads;
        local_profile.search1_time /= num_threads;
        local_profile.search2_time /= num_threads;
        local_profile.one_epoch_log(3);
        local_profile.sol_check(sol_list, num_threads, goal_norm, uid);
        main_profile.combine(&local_profile);

        ///////////////// inserting /////////////////
        TIMER_START;
        uint64_t num_total_insert;
        if (log_level >= 3) {
            num_total_insert = _pool_insert<1>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        } else {
            num_total_insert = _pool_insert<0>(sol_list, num_threads, goal_norm, goal_index, &main_profile);
        }
        TIMER_END;
        main_profile.insert_time += CURRENT_TIME;
        main_profile.insert_log(num_total_insert, CURRENT_TIME);

        if (log_level >= 4) check_pool_status(0, 1);

        if (resort_ratio * num_vec > sorted_index){
            TIMER_START;
            sort_cvec();
            TIMER_END;
            main_profile.sort_time += CURRENT_TIME;
            max_norm = (((int32_t ) cvec[(num_vec - 1) * 3LL + 2LL]) << 1) + 1;
        }

        for (long i = 0; i < num_threads; i++) delete sol_list[i];
    }
    mallopt(M_MMAP_MAX, 65536);
    mallopt(M_TRIM_THRESHOLD, 128*1024);
    main_profile.final_log(3, sieving_stucked);
    if (sieving_stucked || CSD == 80) {
        if (check_dim_lose()) {
            sieving_stucked = 0;
        } else {
            sieving_stucked = 1;
        }
    }
    
    return sieving_stucked;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::left_progressive_bgjfsieve(long ind_l, long ind_r, long num_threads, long log_level, long ssd) {
    if (ind_r - ind_l < ssd) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgjfsieve: sieving dim too small, aborted.\n", nb);
        return 0;
    }
    if (ind_r - ind_l > nb * 32) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgjfsieve: sieving dim(%ld) > nb(%u) * 32, aborted.\n", nb, ind_r - ind_l, nb);
        return -1;
    }

    int show_lift = 0;
    if (abs(log_level - 16384) < 10) {
        log_level -= 16384;
        show_lift = 1;
    }
    
    clear_pool();
    set_num_threads(num_threads);
    set_max_pool_size((long)(pow(4./3., (ind_r - ind_l) *0.5) * 3.2) + 1);
    set_sieving_context(ind_r - ssd, ind_r);
    sampling((pow(4./3., ssd * 0.5) * 6.0) > _pool_size ? _pool_size : (long)(pow(4./3., ssd * 0.5) * 6.0));
    bgj1_Sieve(log_level, 1);
    while(index_l > ind_l) {
        extend_left();
        long target_num_vec = (long) (pow(4./3., CSD * 0.5) * 3.2);
        if (target_num_vec > num_vec + num_empty) num_empty = target_num_vec - num_vec;
        if (CSD >= 92) {
            bgj3_Sieve(log_level, 1);
        } else if (CSD > 80) {
            bgj2_Sieve(log_level, 1);
        } else {
            bgj1_Sieve(log_level, 1);
        }
        if (show_lift) show_min_lift(0);
    }
    return 1;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::left_progressive_bgj1sieve(long ind_l, long ind_r, long num_threads, long log_level) {
    if (ind_r - ind_l < 40) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj1sieve: sieving dim too small, aborted.\n", nb);
        return 0;
    }
    if (ind_r - ind_l > nb * 32) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj1sieve: sieving dim(%ld) > nb(%u) * 32, aborted.\n", nb, ind_r - ind_l, nb);
        return -1;
    }

    int show_lift = 0;
    if (abs(log_level - 16384) < 10) {
        log_level -= 16384;
        show_lift = 1;
    }
    
    clear_pool();
    set_num_threads(num_threads);
    set_max_pool_size((long)(pow(4./3., (ind_r - ind_l) *0.5) * 3.2) + 1);
    set_sieving_context(ind_r - 40, ind_r);
    sampling(2071);
    bgj1_Sieve(log_level, 1);
    while(index_l > ind_l) {
        extend_left();
        long target_num_vec = (long) (pow(4./3., CSD * 0.5) * 3.2);
        if (target_num_vec > num_vec + num_empty) num_empty = target_num_vec - num_vec;
        bgj1_Sieve(log_level, 1);
        if (show_lift) show_min_lift(0);
    }
    return 1;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::left_progressive_bgj2sieve(long ind_l, long ind_r, long num_threads, long log_level) {
    if (ind_r - ind_l < 40) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj2sieve: sieving dim too small, aborted.\n", nb);
        return 0;
    }
    if (ind_r - ind_l > nb * 32) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj2sieve: sieving dim(%ld) > nb(%u) * 32, aborted.\n", nb, ind_r - ind_l, nb);
        return -1;
    }

    int show_lift = 0;
    if (abs(log_level - 16384) < 10) {
        log_level -= 16384;
        show_lift = 1;
    }
    
    clear_pool();
    set_num_threads(num_threads);
    set_max_pool_size((long)(pow(4./3., (ind_r - ind_l) *0.5) * 3.2) + 1);
    set_sieving_context(ind_r - 40, ind_r);
    sampling(2071);
    bgj2_Sieve(log_level, 1);
    while(index_l > ind_l) {
        extend_left();
        long target_num_vec = (long) (pow(4./3., CSD * 0.5) * 3.2);
        if (target_num_vec > num_vec + num_empty) num_empty = target_num_vec - num_vec;
        bgj2_Sieve(log_level, 1);
        if (show_lift) show_min_lift(0);
    }
    return 1;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::left_progressive_bgj3sieve(long ind_l, long ind_r, long num_threads, long log_level) {
    if (ind_r - ind_l < 40) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj3sieve: sieving dim too small, aborted.\n", nb);
        return 0;
    }
    if (ind_r - ind_l > nb * 32) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj3sieve: sieving dim(%ld) > nb(%u) * 32, aborted.\n", nb, ind_r - ind_l, nb);
        return -1;
    }

    int show_lift = 0;
    if (abs(log_level - 16384) < 10) {
        log_level -= 16384;
        show_lift = 1;
    }
    
    clear_pool();
    set_num_threads(num_threads);
    set_max_pool_size((long)(pow(4./3., (ind_r - ind_l) *0.5) * 3.2) + 1);
    set_sieving_context(ind_r - 40, ind_r);
    sampling(2071);
    bgj2_Sieve(log_level, 1);
    while(index_l > ind_l) {
        extend_left();
        long target_num_vec = (long) (pow(4./3., CSD * 0.5) * 3.2);
        if (target_num_vec > num_vec + num_empty) num_empty = target_num_vec - num_vec;
        bgj3_Sieve(log_level, 1);
        if (show_lift) show_min_lift(0);
    }
    return 1;
}

#if COMPILE_POOL_EPI8_96
template struct bgj_profile_data_t<3>;
template void bgj_profile_data_t<3>::pool_bucket_check<0>(bucket_epi8_t<0> **bucket_list, long num_bucket, double alpha);
template void bgj_profile_data_t<3>::pool_bucket_check<1>(bucket_epi8_t<1> **bucket_list, long num_bucket, double alpha);

template class Pool_epi8_t<3>;
#endif

#if COMPILE_POOL_EPI8_128
template struct bgj_profile_data_t<4>;
template void bgj_profile_data_t<4>::pool_bucket_check<0>(bucket_epi8_t<0> **bucket_list, long num_bucket, double alpha);
template void bgj_profile_data_t<4>::pool_bucket_check<1>(bucket_epi8_t<1> **bucket_list, long num_bucket, double alpha);

template class Pool_epi8_t<4>;
#endif

#if COMPILE_POOL_EPI8_160
template struct bgj_profile_data_t<5>;
template void bgj_profile_data_t<5>::pool_bucket_check<0>(bucket_epi8_t<0> **bucket_list, long num_bucket, double alpha);
template void bgj_profile_data_t<5>::pool_bucket_check<1>(bucket_epi8_t<1> **bucket_list, long num_bucket, double alpha);

template class Pool_epi8_t<5>;
#endif

#if COMPILE_POOL_EPI8_192
template struct bgj_profile_data_t<6>;
template void bgj_profile_data_t<6>::pool_bucket_check<0>(bucket_epi8_t<0> **bucket_list, long num_bucket, double alpha);
template void bgj_profile_data_t<6>::pool_bucket_check<1>(bucket_epi8_t<1> **bucket_list, long num_bucket, double alpha);

template class Pool_epi8_t<6>;
#endif

#if COMPILE_POOL_EPI8_224
template struct bgj_profile_data_t<7>;
template void bgj_profile_data_t<7>::pool_bucket_check<0>(bucket_epi8_t<0> **bucket_list, long num_bucket, double alpha);
template void bgj_profile_data_t<7>::pool_bucket_check<1>(bucket_epi8_t<1> **bucket_list, long num_bucket, double alpha);

template class Pool_epi8_t<7>;
#endif
