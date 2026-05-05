/** \file bgj_epi8.h
 *  \brief configuration of sieving parameters
*/
#ifndef __BGJ_EPI8_H
#define __BGJ_EPI8_H


///////////////// BGJ1 parameters /////////////////
#define BGJ1_EPI8_BUCKET_ALPHA 0.325
#define BGJ1_EPI8_USE_FARAWAY_CENTER 1
#define BGJ1_EPI8_USE_3RED 1
#define BGJ1_EPI8_BUCKET_BATCHSIZE 32
#if BGJ1_EPI8_BUCKET_BATCHSIZE % 8
#error pool bucketing batchsize should always divided by 8
#endif

///////////////// BGJ2 parameters /////////////////
#define BGJ2_EPI8_BUCKET0_ALPHA 0.257
#define BGJ2_EPI8_BUCKET1_ALPHA 0.280
#define BGJ2_EPI8_BUCKET0_BATCHSIZE 72
#define BGJ2_EPI8_BUCKET1_BATCHSIZE 72
#define BGJ2_EPI8_REUSE_ALPHA 0.365
#define BGJ2_EPI8_REUSE_USE_3RED 1
#define BGJ2_EPI8_USE_FARAWAY_CENTER 1
#if BGJ2_EPI8_BUCKET0_BATCHSIZE % 8
#error pool bucketing batchsize should always divided by 8
#endif

///////////////// BGJ3 parameters /////////////////
#define BGJ3_EPI8_BUCKET0_ALPHA 0.20
#define BGJ3_EPI8_BUCKET1_ALPHA 0.21
#define BGJ3_EPI8_BUCKET2_ALPHA 0.28
#define BGJ3_EPI8_BUCKET0_BATCHSIZE 56
#define BGJ3_EPI8_BUCKET1_BATCHSIZE 72
#define BGJ3_EPI8_BUCKET2_BATCHSIZE 48
#define BGJ3_EPI8_REUSE0_ALPHA 0.375
#define BGJ3_EPI8_REUSE1_ALPHA 0.310
#define BGJ3_EPI8_REUSE0_USE_3RED 1
#define BGJ3_EPI8_USE_FARAWAY_CENTER 1

///////////////// common parameters /////////////////
#define MAX_NTHREADS 112
#define MIN_LOG_CSD 45
#define MAX_STUCK_TIME 2
#define PBUCKET_USE_BUFFER 0
#define SBUCKET_USE_BUFFER 0
#define PBUCKET_BUFFER_SIZE_DYNAMIC 1
#define SBUCKET_BUFFER_SIZE_DYNAMIC 1
#define PBUCKET_BUFFER_INIT_SIZE 65536
#define SBUCKET_BUFFER_INIT_SIZE 65536
#define SEARCH_L1_BLOCK 64
#define SEARCH_L2_BLOCK 512
#define BGJ_NEED_SEARCH_3RED (BGJ1_EPI8_USE_3RED || BGJ2_EPI8_REUSE_USE_3RED)
#if SEARCH_L1_BLOCK % 8
#error search l1 block must divided by 8
#endif

///////////////// bgj profiling /////////////////
template <unsigned nb>
struct bgj_profile_data_t {
    Pool_epi8_t<nb> *p;
    pthread_spinlock_t profile_lock;
    struct timeval bgj_start_time, bgj_end_time;

    void init(Pool_epi8_t<nb> *_p, long _log_level);
    void initial_log(int bgj);
    void epoch_initial_log(int32_t goal_norm);
    void bucket0_log();
    void bucket1_log();
    void search0_log();
    void search1_log();
    void one_epoch_log(int bgj);
    void insert_log(uint64_t num_total_sol, double insert_time);
    void insert_inner_log(uint64_t *length_stat, uint64_t num_linfty_failed, uint64_t num_l2_failed, uint64_t num_not_try);
    void final_log(int bgj, long sieving_stucked);
    template <bool record_dp>
    void pool_bucket_check(bucket_epi8_t<record_dp> **bucket_list, long num_bucket, double alpha);
    void subbucket_check(bucket_epi8_t<0> **bucket_list, bucket_epi8_t<0> **subbucket_list, long num_bucket, long num_subbucket, double alpha);
    void sol_check(sol_list_epi8_t **sol_list, long num, int32_t goal_norm, UidHashTable *uid);
    void report_bucket_not_used(int bgj, long num_rrem, long num_rem = 0);
    void combine(bgj_profile_data_t<nb> *prof);

    long log_level = -1;
    FILE *log_out = stdout;
    FILE *log_err = stderr;

    long num_epoch = 0;
    long num_bucket0 = 0;
    long num_bucket1 = 0;
    long num_bucket2 = 0;
    long num_r0 = 0;
    long num_r1 = 0;
    long sum_bucket0_size = 0;
    long sum_bucket1_size = 0;
    long sum_bucket2_size = 0;
    long sum_r0_size = 0;
    long sum_r1_size = 0;

    double bucket0_time = 0.0;
    double bucket1_time = 0.0;
    double bucket2_time = 0.0;
    double search0_time = 0.0;
    double search1_time = 0.0;
    double search2_time = 0.0;
    double sort_time = 0.0;
    double insert_time = 0.0;
    double cuda_cred_time0 = 0.0;
    double cuda_single_time0 = 0.0;
    double cuda_batch_time0 = 0.0;
    double cuda_fallback_time0 = 0.0;
    double cuda_cred_time1 = 0.0;
    double cuda_single_time1 = 0.0;
    double cuda_batch_time1 = 0.0;
    double cuda_fallback_time1 = 0.0;
    double materialize_time = 0.0;
    double materialize_gpu_time = 0.0;
    double materialize_cpu_time = 0.0;
    double materialize_scalar_time = 0.0;
    double materialize_cuda_failed_time = 0.0;
    double materialize_cuda_pool_time = 0.0;
    double materialize_cuda_basis_time = 0.0;
    double materialize_cuda_desc_time = 0.0;
    double materialize_cuda_build_time = 0.0;
    double materialize_cuda_gemm_time = 0.0;
    double materialize_cuda_coeff_time = 0.0;
    double materialize_cuda_reconstruct_time = 0.0;
    double materialize_cuda_copy_time = 0.0;
    double insert_scan_time = 0.0;
    double insert_uid_erase_time = 0.0;
    double insert_uid_batch_time = 0.0;
    double insert_copy_time = 0.0;
    double insert_compact_time = 0.0;

    uint64_t bucket0_ndp = 0;
    uint64_t bucket1_ndp = 0;
    uint64_t bucket2_ndp = 0;
    uint64_t search0_ndp = 0;
    uint64_t search1_ndp = 0;
    uint64_t search2_ndp = 0;
    uint64_t cuda_single_ndp0 = 0;
    uint64_t cuda_batch_ndp0 = 0;
    uint64_t cuda_fallback_ndp0 = 0;
    uint64_t cuda_single_bucket0 = 0;
    uint64_t cuda_batch_bucket0 = 0;
    uint64_t cuda_batch_call0 = 0;
    uint64_t cuda_fallback_bucket0 = 0;
    uint64_t cuda_single_ndp1 = 0;
    uint64_t cuda_batch_ndp1 = 0;
    uint64_t cuda_fallback_ndp1 = 0;
    uint64_t cuda_single_bucket1 = 0;
    uint64_t cuda_batch_bucket1 = 0;
    uint64_t cuda_batch_call1 = 0;
    uint64_t cuda_fallback_bucket1 = 0;
    uint64_t materialize_call = 0;
    uint64_t materialize_candidate = 0;
    uint64_t materialize_gpu_call = 0;
    uint64_t materialize_gpu_candidate = 0;
    uint64_t materialize_cpu_call = 0;
    uint64_t materialize_cpu_candidate = 0;
    uint64_t materialize_scalar_call = 0;
    uint64_t materialize_scalar_candidate = 0;
    uint64_t materialize_cuda_failed_call = 0;
    uint64_t materialize_cuda_failed_candidate = 0;
    uint64_t materialize_cuda_phase_chunk = 0;
    uint64_t insert_uid_erase_count = 0;
    uint64_t insert_uid_erase_fail = 0;
    uint64_t insert_copy_count = 0;
    uint64_t insert_compact_move = 0;

    uint64_t try_add2 = 0;
    uint64_t succ_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add3 = 0;
};

#endif
