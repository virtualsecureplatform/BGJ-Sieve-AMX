#include "../include/bgj_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int bgj_cuda_requested_flag = 0;

int bgj_cuda_search_requested()
{
    const char *env = getenv("BGJ_CUDA_SEARCH");
    if (env && env[0] && env[0] != '0') return 1;
    return bgj_cuda_requested_flag;
}

void bgj_cuda_set_search_requested(int enabled)
{
    bgj_cuda_requested_flag = enabled ? 1 : 0;
}

#if !defined(HAVE_CUDA)

uint32_t bgj_cuda_batch_size(uint32_t)
{
    return 1;
}

uint64_t bgj_cuda_batch_min_dots()
{
    return 0xffffffffffffffffULL;
}

int bgj_cuda_materialize_requested()
{
    return 0;
}

int bgj_cuda_device_count()
{
    return 0;
}

const char *bgj_cuda_last_error()
{
    return "CUDA support was not compiled in";
}

#else

#include "../include/pool_epi8.h"
#include "../include/bgj_epi8.h"
#include "../include/bucket_epi8.h"

#include <algorithm>
#include <vector>

extern "C" int bgj_cuda_raw_device_count();
extern "C" const char *bgj_cuda_raw_last_error();
extern "C" int bgj_cuda_search_bucket_raw(const int8_t *p_vecs,
                                           const int8_t *n_vecs,
                                           const uint32_t *p_ids,
                                           const uint32_t *n_ids,
                                           const int32_t *p_norm,
                                           const int32_t *n_norm,
                                           const int32_t *p_dot,
                                           const int32_t *n_dot,
                                           uint32_t num_p,
                                           uint32_t num_n,
                                           uint32_t vec_length,
                                           int32_t goal_norm,
                                           int32_t center_norm,
                                           int record_dp,
                                           bgj_cuda_result_t *results,
                                           uint32_t result_capacity,
                                           uint32_t *result_count,
                                           int *overflow);
extern "C" int bgj_cuda_search_bucket_pool_raw(const int8_t *pool_vecs,
                                                uint64_t pool_epoch,
                                                uint32_t pool_size,
                                                const uint32_t *p_ids,
                                                const uint32_t *n_ids,
                                                const int32_t *p_norm,
                                                const int32_t *n_norm,
                                                const int32_t *p_dot,
                                                const int32_t *n_dot,
                                                uint32_t num_p,
                                                uint32_t num_n,
                                                uint32_t vec_length,
                                                int32_t goal_norm,
                                                int32_t center_norm,
                                                int record_dp,
                                                bgj_cuda_result_t *results,
                                                uint32_t result_capacity,
                                                uint32_t *result_count,
                                                int *overflow);
extern "C" int bgj_cuda_search_bucket_pool_batch_raw(const int8_t *pool_vecs,
                                                      uint64_t pool_epoch,
                                                      uint32_t pool_size,
                                                      const uint32_t *const *p_ids,
                                                      const uint32_t *const *n_ids,
                                                      const int32_t *const *p_norm,
                                                      const int32_t *const *n_norm,
                                                      const int32_t *const *p_dot,
                                                      const int32_t *const *n_dot,
                                                      const uint32_t *num_p,
                                                      const uint32_t *num_n,
                                                      uint32_t batch_size,
                                                      uint32_t vec_length,
                                                      const int32_t *goal_norm,
                                                      const int32_t *center_norm,
                                                      int record_dp,
                                                      bgj_cuda_result_t *const *results,
                                                      const uint32_t *result_capacity,
                                                      uint32_t *result_count,
                                                      int *overflow);

int bgj_cuda_device_count()
{
    static int count = -1;
    if (count < 0) count = bgj_cuda_raw_device_count();
    return count;
}

const char *bgj_cuda_last_error()
{
    return bgj_cuda_raw_last_error();
}

static uint32_t bgj_cuda_max_results()
{
    const char *env = getenv("BGJ_CUDA_MAX_RESULTS");
    if (env && env[0]) {
        unsigned long value = strtoul(env, NULL, 10);
        if (value > 0 && value <= 0xffffffffUL) return (uint32_t)value;
    }
    return 1u << 24;
}

static int bgj_cuda_pool_cache_requested()
{
    const char *env = getenv("BGJ_CUDA_POOL_CACHE");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static int bgj_cuda_overflow_fallback_requested()
{
    const char *env = getenv("BGJ_CUDA_OVERFLOW_FALLBACK");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

static int bgj_cuda_sort_results_requested()
{
    const char *env = getenv("BGJ_CUDA_SORT_RESULTS");
    return env && env[0] && env[0] != '0';
}

int bgj_cuda_materialize_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

uint32_t bgj_cuda_batch_size(uint32_t host_threads)
{
    if (!bgj_cuda_pool_cache_requested()) return 1;

    const char *enabled = getenv("BGJ_CUDA_BATCH");
    if (enabled && enabled[0] && enabled[0] == '0') return 1;

    const char *size_env = getenv("BGJ_CUDA_BATCH_SIZE");
    if (!enabled && !size_env) return 1;

    uint32_t value = 8;
    if (size_env && size_env[0]) {
        unsigned long parsed = strtoul(size_env, NULL, 10);
        if (parsed > 0 && parsed <= 0xffffffffUL) value = (uint32_t)parsed;
    }
    if (value < 1) value = 1;
    if (value > 32) value = 32;
    return value;
}

uint64_t bgj_cuda_batch_min_dots()
{
    const char *env = getenv("BGJ_CUDA_BATCH_MIN_DOTS");
    if (env && env[0]) {
        unsigned long long value = strtoull(env, NULL, 10);
        if (value > 0) return (uint64_t)value;
    }
    return 128ULL * 1024ULL;
}

static int bgj_cuda_rank_of(const std::vector<int32_t> &rank,
                            const std::vector<uint32_t> &stamp,
                            uint32_t epoch,
                            uint32_t id)
{
    return id < rank.size() && stamp[id] == epoch ? rank[id] : -1;
}

static int bgj_cuda_result_phase(const bgj_cuda_result_t &result,
                                 const std::vector<int32_t> &p_rank,
                                 const std::vector<uint32_t> &p_stamp,
                                 const std::vector<int32_t> &n_rank,
                                 const std::vector<uint32_t> &n_stamp,
                                 uint32_t epoch)
{
    switch (result.type) {
    case BGJ_CUDA_SOL_A:
        return 0;
    case BGJ_CUDA_SOL_SA:
        return 1;
    case BGJ_CUDA_SOL_S:
        if (bgj_cuda_rank_of(p_rank, p_stamp, epoch, result.x) >= 0 &&
            bgj_cuda_rank_of(p_rank, p_stamp, epoch, result.y) >= 0) {
            return 2;
        }
        if (bgj_cuda_rank_of(n_rank, n_stamp, epoch, result.x) >= 0 &&
            bgj_cuda_rank_of(n_rank, n_stamp, epoch, result.y) >= 0) {
            return 4;
        }
        return 6;
    case BGJ_CUDA_SOL_SS:
        return 3;
    case BGJ_CUDA_SOL_AA:
        return 5;
    default:
        return 7;
    }
}

static int bgj_cuda_local_rank(const bgj_cuda_result_t &result,
                               const std::vector<int32_t> &p_rank,
                               const std::vector<uint32_t> &p_stamp,
                               const std::vector<int32_t> &n_rank,
                               const std::vector<uint32_t> &n_stamp,
                               uint32_t epoch,
                               int first)
{
    const uint32_t id = first ? result.x : result.y;
    switch (result.type) {
    case BGJ_CUDA_SOL_A:
    case BGJ_CUDA_SOL_SA:
        return first ? bgj_cuda_rank_of(p_rank, p_stamp, epoch, id) :
                       bgj_cuda_rank_of(n_rank, n_stamp, epoch, id);
    case BGJ_CUDA_SOL_SS:
        return bgj_cuda_rank_of(p_rank, p_stamp, epoch, id);
    case BGJ_CUDA_SOL_AA:
        return bgj_cuda_rank_of(n_rank, n_stamp, epoch, id);
    case BGJ_CUDA_SOL_S:
        if (bgj_cuda_rank_of(p_rank, p_stamp, epoch, result.x) >= 0 &&
            bgj_cuda_rank_of(p_rank, p_stamp, epoch, result.y) >= 0) {
            return bgj_cuda_rank_of(p_rank, p_stamp, epoch, id);
        }
        return bgj_cuda_rank_of(n_rank, n_stamp, epoch, id);
    default:
        return -1;
    }
}

template <uint32_t nb, bool record_dp, bool profiling>
static int bgj_cuda_consume_bgj1_results(Pool_epi8_t<nb> *pool,
                                         bucket_epi8_t<record_dp> *bkt,
                                         sol_list_epi8_t *sol,
                                         bgj_profile_data_t<nb> *prof,
                                         bgj_cuda_result_t *results,
                                         uint32_t result_count)
{
    const uint32_t num_p = (uint32_t)bkt->num_pvec;
    const uint32_t num_n = (uint32_t)bkt->num_nvec;

    if (result_count > 1 && bgj_cuda_sort_results_requested()) {
        static thread_local std::vector<int32_t> p_rank;
        static thread_local std::vector<int32_t> n_rank;
        static thread_local std::vector<uint32_t> p_rank_stamp;
        static thread_local std::vector<uint32_t> n_rank_stamp;
        static thread_local uint32_t rank_epoch = 1;

        rank_epoch++;
        if (rank_epoch == 0) {
            std::fill(p_rank_stamp.begin(), p_rank_stamp.end(), 0);
            std::fill(n_rank_stamp.begin(), n_rank_stamp.end(), 0);
            rank_epoch = 1;
        }
        try {
            if (p_rank.size() < (size_t)pool->num_vec) {
                p_rank.resize((size_t)pool->num_vec);
                p_rank_stamp.resize((size_t)pool->num_vec, 0);
            }
            if (n_rank.size() < (size_t)pool->num_vec) {
                n_rank.resize((size_t)pool->num_vec);
                n_rank_stamp.resize((size_t)pool->num_vec, 0);
            }
        } catch (...) {
            return 0;
        }
        for (uint32_t i = 0; i < num_p; i++) {
            if (bkt->pvec[i] < (uint32_t)p_rank.size()) {
                p_rank[bkt->pvec[i]] = (int32_t)i;
                p_rank_stamp[bkt->pvec[i]] = rank_epoch;
            }
        }
        for (uint32_t i = 0; i < num_n; i++) {
            if (bkt->nvec[i] < (uint32_t)n_rank.size()) {
                n_rank[bkt->nvec[i]] = (int32_t)i;
                n_rank_stamp[bkt->nvec[i]] = rank_epoch;
            }
        }
        std::sort(results, results + result_count,
                  [](const bgj_cuda_result_t &a, const bgj_cuda_result_t &b) {
                      const int phase_a = bgj_cuda_result_phase(a, p_rank, p_rank_stamp, n_rank, n_rank_stamp, rank_epoch);
                      const int phase_b = bgj_cuda_result_phase(b, p_rank, p_rank_stamp, n_rank, n_rank_stamp, rank_epoch);
                      if (phase_a != phase_b) return phase_a < phase_b;
                      const int a0 = bgj_cuda_local_rank(a, p_rank, p_rank_stamp, n_rank, n_rank_stamp, rank_epoch, 1);
                      const int b0 = bgj_cuda_local_rank(b, p_rank, p_rank_stamp, n_rank, n_rank_stamp, rank_epoch, 1);
                      if (a0 != b0) return a0 < b0;
                      const int a1 = bgj_cuda_local_rank(a, p_rank, p_rank_stamp, n_rank, n_rank_stamp, rank_epoch, 0);
                      const int b1 = bgj_cuda_local_rank(b, p_rank, p_rank_stamp, n_rank, n_rank_stamp, rank_epoch, 0);
                      if (a1 != b1) return a1 < b1;
                      if (a.type != b.type) return a.type < b.type;
                      if (a.x != b.x) return a.x < b.x;
                      return a.y < b.y;
                  });
    }

    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;

    for (uint32_t i = 0; i < result_count; i++) {
        const uint32_t x = results[i].x;
        const uint32_t y = results[i].y;
        switch (results[i].type) {
        case BGJ_CUDA_SOL_A: {
            try_add2++;
            const uint64_t u = pool->vu[x] + pool->vu[y];
            if (pool->uid->insert_uid(u)) {
                succ_add2++;
                sol->add_sol_a(x, y);
            }
            break;
        }
        case BGJ_CUDA_SOL_S: {
            try_add2++;
            const uint64_t u = pool->vu[x] - pool->vu[y];
            if (pool->uid->insert_uid(u)) {
                succ_add2++;
                sol->add_sol_s(x, y);
            }
            break;
        }
        case BGJ_CUDA_SOL_AA: {
            try_add3++;
            const uint64_t u = bkt->center_u + pool->vu[x] + pool->vu[y];
            if (pool->uid->insert_uid(u)) {
                succ_add3++;
                sol->add_sol_aa(bkt->center_ind, x, y);
            }
            break;
        }
        case BGJ_CUDA_SOL_SA: {
            try_add3++;
            const uint64_t u = bkt->center_u - pool->vu[x] + pool->vu[y];
            if (pool->uid->insert_uid(u)) {
                succ_add3++;
                sol->add_sol_sa(bkt->center_ind, x, y);
            }
            break;
        }
        case BGJ_CUDA_SOL_SS: {
            try_add3++;
            const uint64_t u = bkt->center_u - pool->vu[x] - pool->vu[y];
            if (pool->uid->insert_uid(u)) {
                succ_add3++;
                sol->add_sol_ss(bkt->center_ind, x, y);
            }
            break;
        }
        default:
            break;
        }
    }

    if (profiling && prof) {
        pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->try_add3 += try_add3;
        prof->succ_add2 += succ_add2;
        prof->succ_add3 += succ_add3;
        pthread_spin_unlock(&prof->profile_lock);
    }

    return 1;
}

template <uint32_t nb>
template <bool record_dp, bool profiling>
int Pool_epi8_t<nb>::_search_bgj1_cuda(bucket_epi8_t<record_dp> *bkt,
                                       sol_list_epi8_t *sol,
                                       int32_t goal_norm,
                                       bgj_profile_data_t<nb> *prof)
{
    if (bgj_cuda_device_count() <= 0) return 0;
    if (bkt->num_pvec < 0 || bkt->num_nvec < 0) return 0;
    if (bkt->num_pvec > 0xffffffffL || bkt->num_nvec > 0xffffffffL) return 0;
    if (num_vec < 0 || num_vec > 0xffffffffL) return 0;

    const uint32_t num_p = (uint32_t)bkt->num_pvec;
    const uint32_t num_n = (uint32_t)bkt->num_nvec;
    if (num_p + num_n == 0) return 1;

    static thread_local std::vector<int8_t> p_vec_storage;
    static thread_local std::vector<int8_t> n_vec_storage;
    static thread_local std::vector<bgj_cuda_result_t> result_storage;
    const int use_pool_cache = bgj_cuda_pool_cache_requested();
    int8_t *p_vecs = NULL;
    int8_t *n_vecs = NULL;

    if (!use_pool_cache) {
        const size_t p_bytes = (size_t)num_p * (size_t)vec_length * sizeof(int8_t);
        const size_t n_bytes = (size_t)num_n * (size_t)vec_length * sizeof(int8_t);
        try {
            p_vec_storage.resize(p_bytes);
            n_vec_storage.resize(n_bytes);
        } catch (...) {
            return 0;
        }
        p_vecs = p_bytes ? p_vec_storage.data() : NULL;
        n_vecs = n_bytes ? n_vec_storage.data() : NULL;

        for (uint32_t i = 0; i < num_p; i++) {
            memcpy(p_vecs + (size_t)i * vec_length, vec + (size_t)bkt->pvec[i] * vec_length, vec_length);
        }
        for (uint32_t i = 0; i < num_n; i++) {
            memcpy(n_vecs + (size_t)i * vec_length, vec + (size_t)bkt->nvec[i] * vec_length, vec_length);
        }
    }

    const uint32_t result_capacity = bgj_cuda_max_results();
    try {
        result_storage.resize((size_t)result_capacity);
    } catch (...) {
        return 0;
    }
    bgj_cuda_result_t *results = result_storage.data();

    uint32_t result_count = 0;
    int overflow = 0;
    const int ok = use_pool_cache ?
                   bgj_cuda_search_bucket_pool_raw(vec,
                                                   pool_epoch,
                                                   (uint32_t)num_vec,
                                                   bkt->pvec,
                                                   bkt->nvec,
                                                   bkt->pnorm,
                                                   bkt->nnorm,
                                                   record_dp ? bkt->pdot : NULL,
                                                   record_dp ? bkt->ndot : NULL,
                                                   num_p,
                                                   num_n,
                                                   (uint32_t)vec_length,
                                                   goal_norm,
                                                   bkt->center_norm,
                                                   record_dp ? 1 : 0,
                                                   results,
                                                   result_capacity,
                                                   &result_count,
                                                   &overflow) :
                   bgj_cuda_search_bucket_raw(p_vecs,
                                              n_vecs,
                                              bkt->pvec,
                                              bkt->nvec,
                                              bkt->pnorm,
                                              bkt->nnorm,
                                              record_dp ? bkt->pdot : NULL,
                                              record_dp ? bkt->ndot : NULL,
                                              num_p,
                                              num_n,
                                              (uint32_t)vec_length,
                                              goal_norm,
                                              bkt->center_norm,
                                              record_dp ? 1 : 0,
                                              results,
                                              result_capacity,
                                              &result_count,
                                              &overflow);

    if (!ok || (overflow && bgj_cuda_overflow_fallback_requested())) {
        return 0;
    }

    return bgj_cuda_consume_bgj1_results<nb, record_dp, profiling>(this,
                                                                    bkt,
                                                                    sol,
                                                                    prof,
                                                                    results,
                                                                    result_count);
}

template <uint32_t nb>
template <bool record_dp, bool profiling>
int Pool_epi8_t<nb>::_search_bgj1_cuda_batch(bucket_epi8_t<record_dp> **buckets,
                                             long num_bucket,
                                             sol_list_epi8_t *sol,
                                             int32_t goal_norm,
                                             bgj_profile_data_t<nb> *prof)
{
    if (num_bucket <= 0) return 1;
    if (bgj_cuda_device_count() <= 0) return 0;
    if (!bgj_cuda_pool_cache_requested()) return 0;
    if (num_vec < 0 || num_vec > 0xffffffffL) return 0;
    if (num_bucket > 0xffffffffL) return 0;

    const uint32_t batch_size = (uint32_t)num_bucket;
    const uint32_t result_capacity = bgj_cuda_max_results();

    static thread_local std::vector<const uint32_t *> p_ids;
    static thread_local std::vector<const uint32_t *> n_ids;
    static thread_local std::vector<const int32_t *> p_norm;
    static thread_local std::vector<const int32_t *> n_norm;
    static thread_local std::vector<const int32_t *> p_dot;
    static thread_local std::vector<const int32_t *> n_dot;
    static thread_local std::vector<uint32_t> num_p;
    static thread_local std::vector<uint32_t> num_n;
    static thread_local std::vector<int32_t> goal_norms;
    static thread_local std::vector<int32_t> center_norms;
    static thread_local std::vector<bgj_cuda_result_t *> result_ptrs;
    static thread_local std::vector<uint32_t> result_capacities;
    static thread_local std::vector<uint32_t> result_counts;
    static thread_local std::vector<int> overflows;
    static thread_local std::vector<std::vector<bgj_cuda_result_t> > result_storage;

    try {
        p_ids.resize(batch_size);
        n_ids.resize(batch_size);
        p_norm.resize(batch_size);
        n_norm.resize(batch_size);
        p_dot.resize(batch_size);
        n_dot.resize(batch_size);
        num_p.resize(batch_size);
        num_n.resize(batch_size);
        goal_norms.resize(batch_size);
        center_norms.resize(batch_size);
        result_ptrs.resize(batch_size);
        result_capacities.resize(batch_size);
        result_counts.resize(batch_size);
        overflows.resize(batch_size);
        result_storage.resize(batch_size);
    } catch (...) {
        return 0;
    }

    for (uint32_t i = 0; i < batch_size; i++) {
        bucket_epi8_t<record_dp> *bkt = buckets[i];
        if (!bkt) return 0;
        if (bkt->num_pvec < 0 || bkt->num_nvec < 0) return 0;
        if (bkt->num_pvec > 0xffffffffL || bkt->num_nvec > 0xffffffffL) return 0;

        num_p[i] = (uint32_t)bkt->num_pvec;
        num_n[i] = (uint32_t)bkt->num_nvec;
        p_ids[i] = bkt->pvec;
        n_ids[i] = bkt->nvec;
        p_norm[i] = bkt->pnorm;
        n_norm[i] = bkt->nnorm;
        p_dot[i] = record_dp ? bkt->pdot : NULL;
        n_dot[i] = record_dp ? bkt->ndot : NULL;
        goal_norms[i] = goal_norm;
        center_norms[i] = bkt->center_norm;
        result_capacities[i] = result_capacity;
        try {
            result_storage[i].resize((size_t)result_capacity);
        } catch (...) {
            return 0;
        }
        result_ptrs[i] = result_storage[i].data();
        result_counts[i] = 0;
        overflows[i] = 0;
    }

    const int ok = bgj_cuda_search_bucket_pool_batch_raw(vec,
                                                         pool_epoch,
                                                         (uint32_t)num_vec,
                                                         p_ids.data(),
                                                         n_ids.data(),
                                                         p_norm.data(),
                                                         n_norm.data(),
                                                         p_dot.data(),
                                                         n_dot.data(),
                                                         num_p.data(),
                                                         num_n.data(),
                                                         batch_size,
                                                         (uint32_t)vec_length,
                                                         goal_norms.data(),
                                                         center_norms.data(),
                                                         record_dp ? 1 : 0,
                                                         result_ptrs.data(),
                                                         result_capacities.data(),
                                                         result_counts.data(),
                                                         overflows.data());
    if (!ok) return 0;

    for (uint32_t i = 0; i < batch_size; i++) {
        if (overflows[i] && bgj_cuda_overflow_fallback_requested()) return 0;
    }
    for (uint32_t i = 0; i < batch_size; i++) {
        if (!bgj_cuda_consume_bgj1_results<nb, record_dp, profiling>(this,
                                                                      buckets[i],
                                                                      sol,
                                                                      prof,
                                                                      result_ptrs[i],
                                                                      result_counts[i])) {
            return 0;
        }
    }
    return 1;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj)
{
    if (bgj_cuda_device_count() <= 0) {
        fprintf(stderr, "[Warning] CUDA requested but no CUDA device is available: %s. Falling back to CPU bgj1.\n",
                bgj_cuda_last_error());
        return bgj1_Sieve(log_level, lps_auto_adj);
    }

    const int old = bgj_cuda_search_requested();
    bgj_cuda_set_search_requested(1);
    const int ret = bgj1_Sieve(log_level, lps_auto_adj);
    bgj_cuda_set_search_requested(old);
    return ret;
}

static long bgj_cuda_host_threads(long requested, long sieving_dim)
{
    const char *env = getenv("BGJ_CUDA_HOST_THREADS");
    if (env && env[0]) {
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            return parsed > MAX_NTHREADS ? MAX_NTHREADS : parsed;
        }
    }

    if (requested < 1) requested = 1;
    if (requested < 2 && sieving_dim >= 80) return 2;
    return requested > MAX_NTHREADS ? MAX_NTHREADS : requested;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level)
{
    if (ind_r - ind_l < 40) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj1sieve_cuda: sieving dim too small, aborted.\n", nb);
        return 0;
    }
    if (ind_r - ind_l > nb * 32) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgj1sieve_cuda: sieving dim(%ld) > nb(%u) * 32, aborted.\n", nb, ind_r - ind_l, nb);
        return -1;
    }

    int show_lift = 0;
    if (abs(log_level - 16384) < 10) {
        log_level -= 16384;
        show_lift = 1;
    }

    clear_pool();
    set_num_threads(bgj_cuda_host_threads(num_threads, ind_r - ind_l));
    set_max_pool_size((long)(pow(4. / 3., (ind_r - ind_l) * 0.5) * 3.2) + 1);
    set_sieving_context(ind_r - 40, ind_r);
    sampling(2071);
    bgj1_Sieve_cuda(log_level, 1);
    while (index_l > ind_l) {
        extend_left();
        long target_num_vec = (long)(pow(4. / 3., CSD * 0.5) * 3.2);
        if (target_num_vec > num_vec + num_empty) num_empty = target_num_vec - num_vec;
        bgj1_Sieve_cuda(log_level, 1);
        if (show_lift) show_min_lift(0);
    }
    return 1;
}

#if COMPILE_POOL_EPI8_96
template int Pool_epi8_t<3>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<3>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_128
template int Pool_epi8_t<4>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<4>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_160
template int Pool_epi8_t<5>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<5>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_192
template int Pool_epi8_t<6>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<6>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_224
template int Pool_epi8_t<7>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<7>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#endif
