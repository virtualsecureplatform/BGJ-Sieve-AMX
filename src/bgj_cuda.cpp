#include "../include/bgj_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

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

int bgj_cuda_cred_transform_requested()
{
    return 0;
}

int bgj_cuda_overlap_cred_requested()
{
    return 0;
}

int bgj_cuda_bucket_requested()
{
    return 0;
}

int bgj_cuda_device_count()
{
    return 0;
}

int bgj_cuda_execution_device_count()
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
#include <limits>
#include <unordered_set>
#include <vector>

static_assert((UidHashTable::NUM_UID_LOCK & (UidHashTable::NUM_UID_LOCK - 1)) == 0,
              "CUDA UID fast path assumes power-of-two UID lock count");

struct bgj_cuda_uid_candidate_t {
    uint64_t uid;
    uint32_t x;
    uint32_t y;
    uint32_t center;
    uint32_t type;
    uint16_t slot;
    uint8_t accepted;
};

static inline void bgj_cuda_normalize_uid(uint64_t &uid)
{
    if (uid > std::numeric_limits<uint64_t>::max() / 2 + 1) uid = -uid;
}

static inline bool bgj_cuda_insert_uid_fast(UidHashTable *uid_table, uint64_t uid)
{
    bgj_cuda_normalize_uid(uid);
    const unsigned slot = (unsigned)(uid & (UidHashTable::NUM_UID_LOCK - 1));
    pthread_spin_lock(&uid_table->uid_lock[slot].a);
    const bool success = uid_table->uid_table[slot].a.insert(uid).second;
    pthread_spin_unlock(&uid_table->uid_lock[slot].a);
    return success;
}

static int bgj_cuda_uid_batch_requested()
{
    static const int requested = []() {
        const char *env = getenv("BGJ_CUDA_UID_BATCH");
        if (env && env[0]) return env[0] != '0' ? 1 : 0;
        return 1;
    }();
    return requested;
}

static uint32_t bgj_cuda_uid_batch_min_results()
{
    static const uint32_t min_results = []() {
        const char *env = getenv("BGJ_CUDA_UID_BATCH_MIN_RESULTS");
        if (env && env[0]) {
            char *end = NULL;
            const unsigned long parsed = strtoul(env, &end, 10);
            if (end != env && parsed > 0 && parsed <= 0xfffffffful) {
                return (uint32_t)parsed;
            }
        }
        return (uint32_t)4096;
    }();
    return min_results;
}

static uint32_t bgj_cuda_sorted_uid_batch_min_results()
{
    static const uint32_t min_results = []() {
        const char *env = getenv("BGJ_CUDA_SORTED_UID_BATCH_MIN_RESULTS");
        if (env && env[0]) {
            char *end = NULL;
            const unsigned long parsed = strtoul(env, &end, 10);
            if (end != env && parsed > 0 && parsed <= 0xfffffffful) {
                return (uint32_t)parsed;
            }
        }
        return (uint32_t)1024;
    }();
    return min_results;
}

static int bgj_cuda_uid_coalesce_requested()
{
    static const int requested = []() {
        const char *env = getenv("BGJ_CUDA_UID_COALESCE");
        if (env && env[0]) return env[0] != '0' ? 1 : 0;
        return 0;
    }();
    return requested;
}

static uint64_t bgj_cuda_uid_coalesce_min_results()
{
    static const uint64_t min_results = []() {
        const char *env = getenv("BGJ_CUDA_UID_COALESCE_MIN_RESULTS");
        if (env && env[0]) {
            char *end = NULL;
            const unsigned long long parsed = strtoull(env, &end, 10);
            if (end != env && parsed > 0) return (uint64_t)parsed;
        }
        return (uint64_t)4096;
    }();
    return min_results;
}

static inline void bgj_cuda_add_uid_candidate(std::vector<bgj_cuda_uid_candidate_t> &candidates,
                                              std::vector<uint32_t> &touched_slots,
                                              std::vector<uint32_t> &slot_counts,
                                              uint64_t uid,
                                              uint32_t type,
                                              uint32_t center,
                                              uint32_t x,
                                              uint32_t y)
{
    bgj_cuda_normalize_uid(uid);
    const uint16_t slot = (uint16_t)(uid & (UidHashTable::NUM_UID_LOCK - 1));
    if (slot_counts[slot]++ == 0) touched_slots.push_back(slot);

    bgj_cuda_uid_candidate_t candidate;
    candidate.uid = uid;
    candidate.x = x;
    candidate.y = y;
    candidate.center = center;
    candidate.type = type;
    candidate.slot = slot;
    candidate.accepted = 0;
    candidates.push_back(candidate);
}

static inline void bgj_cuda_add_accepted_solution(sol_list_epi8_t *sol,
                                                  const bgj_cuda_uid_candidate_t &candidate)
{
    switch (candidate.type) {
    case BGJ_CUDA_SOL_A:
        sol->add_sol_a(candidate.x, candidate.y);
        break;
    case BGJ_CUDA_SOL_S:
        sol->add_sol_s(candidate.x, candidate.y);
        break;
    case BGJ_CUDA_SOL_AA:
        sol->add_sol_aa(candidate.center, candidate.x, candidate.y);
        break;
    case BGJ_CUDA_SOL_SA:
        sol->add_sol_sa(candidate.center, candidate.x, candidate.y);
        break;
    case BGJ_CUDA_SOL_SS:
        sol->add_sol_ss(candidate.center, candidate.x, candidate.y);
        break;
    default:
        break;
    }
}

struct bgj_cuda_host_profile_state_t {
    pthread_mutex_t lock;
    int registered;
    uint64_t buckets;
    uint64_t results;
    uint64_t try_add2;
    uint64_t try_add3;
    uint64_t succ_add2;
    uint64_t succ_add3;
    uint64_t type_count[5];
    uint64_t bucket_uid_unique;
    uint64_t bucket_uid_dup;
    double sort_sec;
    double bucket_uid_sec;
    double uid_sec;
    double sol_sec;
    double consume_sec;

    bgj_cuda_host_profile_state_t()
        : lock(PTHREAD_MUTEX_INITIALIZER),
          registered(0),
          buckets(0),
          results(0),
          try_add2(0),
          try_add3(0),
          succ_add2(0),
          succ_add3(0),
          type_count(),
          bucket_uid_unique(0),
          bucket_uid_dup(0),
          sort_sec(0.0),
          bucket_uid_sec(0.0),
          uid_sec(0.0),
          sol_sec(0.0),
          consume_sec(0.0)
    {
    }
};

static bgj_cuda_host_profile_state_t bgj_cuda_host_profile;

struct bgj_cuda_uid_batch_profile_state_t {
    pthread_mutex_t lock;
    int registered;
    uint64_t buckets;
    uint64_t batched_buckets;
    uint64_t scalar_buckets;
    uint64_t results;
    uint64_t candidates;
    uint64_t batched_candidates;
    uint64_t scalar_candidates;
    uint64_t accepted;
    uint64_t touched_locks;
    uint64_t max_results;
    uint64_t max_candidates;
    uint64_t max_touched_locks;
    double build_sec;
    double group_sec;
    double insert_sec;
    double sol_sec;
    double scalar_sec;
    double total_sec;

    bgj_cuda_uid_batch_profile_state_t()
        : lock(PTHREAD_MUTEX_INITIALIZER),
          registered(0),
          buckets(0),
          batched_buckets(0),
          scalar_buckets(0),
          results(0),
          candidates(0),
          batched_candidates(0),
          scalar_candidates(0),
          accepted(0),
          touched_locks(0),
          max_results(0),
          max_candidates(0),
          max_touched_locks(0),
          build_sec(0.0),
          group_sec(0.0),
          insert_sec(0.0),
          sol_sec(0.0),
          scalar_sec(0.0),
          total_sec(0.0)
    {
    }
};

static bgj_cuda_uid_batch_profile_state_t bgj_cuda_uid_batch_profile;

struct bgj_cuda_consume_profile_state_t {
    pthread_mutex_t lock;
    int registered;
    uint64_t buckets;
    uint64_t sorted_buckets;
    uint64_t radix_buckets;
    uint64_t stdsort_buckets;
    uint64_t batched_uid_buckets;
    uint64_t scalar_uid_buckets;
    uint64_t results;
    uint64_t sorted_results;
    uint64_t linear_rank_buckets;
    uint64_t table_rank_buckets;
    uint64_t uid_candidates;
    uint64_t uid_accepted;
    uint64_t uid_touched_locks;
    uint64_t max_results;
    double rank_sec;
    double key_sec;
    double radix_sec;
    double stdsort_sec;
    double write_sec;
    double sort_sec;
    double uid_build_sec;
    double uid_group_sec;
    double uid_insert_sec;
    double uid_sol_sec;
    double uid_scalar_sec;
    double uid_total_sec;
    double total_sec;

    bgj_cuda_consume_profile_state_t()
        : lock(PTHREAD_MUTEX_INITIALIZER),
          registered(0),
          buckets(0),
          sorted_buckets(0),
          radix_buckets(0),
          stdsort_buckets(0),
          batched_uid_buckets(0),
          scalar_uid_buckets(0),
          results(0),
          sorted_results(0),
          linear_rank_buckets(0),
          table_rank_buckets(0),
          uid_candidates(0),
          uid_accepted(0),
          uid_touched_locks(0),
          max_results(0),
          rank_sec(0.0),
          key_sec(0.0),
          radix_sec(0.0),
          stdsort_sec(0.0),
          write_sec(0.0),
          sort_sec(0.0),
          uid_build_sec(0.0),
          uid_group_sec(0.0),
          uid_insert_sec(0.0),
          uid_sol_sec(0.0),
          uid_scalar_sec(0.0),
          uid_total_sec(0.0),
          total_sec(0.0)
    {
    }
};

static bgj_cuda_consume_profile_state_t bgj_cuda_consume_profile;

static double bgj_cuda_host_wall_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static int bgj_cuda_host_profile_requested()
{
    const char *env = getenv("BGJ_CUDA_HOST_PROFILE");
    return env && env[0] && env[0] != '0';
}

static int bgj_cuda_uid_batch_profile_requested()
{
    static const int requested = []() {
        const char *env = getenv("BGJ_CUDA_UID_BATCH_PROFILE");
        return env && env[0] && env[0] != '0';
    }();
    return requested;
}

static int bgj_cuda_consume_profile_requested()
{
    static const int requested = []() {
        const char *env = getenv("BGJ_CUDA_CONSUME_PROFILE");
        return env && env[0] && env[0] != '0';
    }();
    return requested;
}

static void bgj_cuda_host_profile_dump()
{
    if (!bgj_cuda_host_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_host_profile.lock);
    fprintf(stderr,
            "cuda_host_profile: buckets=%lu results=%lu try_add2=%lu try_add3=%lu "
            "succ_add2=%lu succ_add3=%lu type=[%lu,%lu,%lu,%lu,%lu] "
            "bucket_uid_unique=%lu bucket_uid_dup=%lu "
            "sort=%.6fs bucket_uid=%.6fs uid=%.6fs sol=%.6fs consume=%.6fs\n",
            (unsigned long)bgj_cuda_host_profile.buckets,
            (unsigned long)bgj_cuda_host_profile.results,
            (unsigned long)bgj_cuda_host_profile.try_add2,
            (unsigned long)bgj_cuda_host_profile.try_add3,
            (unsigned long)bgj_cuda_host_profile.succ_add2,
            (unsigned long)bgj_cuda_host_profile.succ_add3,
            (unsigned long)bgj_cuda_host_profile.type_count[0],
            (unsigned long)bgj_cuda_host_profile.type_count[1],
            (unsigned long)bgj_cuda_host_profile.type_count[2],
            (unsigned long)bgj_cuda_host_profile.type_count[3],
            (unsigned long)bgj_cuda_host_profile.type_count[4],
            (unsigned long)bgj_cuda_host_profile.bucket_uid_unique,
            (unsigned long)bgj_cuda_host_profile.bucket_uid_dup,
            bgj_cuda_host_profile.sort_sec,
            bgj_cuda_host_profile.bucket_uid_sec,
            bgj_cuda_host_profile.uid_sec,
            bgj_cuda_host_profile.sol_sec,
            bgj_cuda_host_profile.consume_sec);
    pthread_mutex_unlock(&bgj_cuda_host_profile.lock);
}

static void bgj_cuda_uid_batch_profile_dump()
{
    if (!bgj_cuda_uid_batch_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_uid_batch_profile.lock);
    fprintf(stderr,
            "cuda_uid_batch_profile: buckets=%lu batched_buckets=%lu scalar_buckets=%lu "
            "results=%lu candidates=%lu batched_candidates=%lu scalar_candidates=%lu "
            "accepted=%lu touched_locks=%lu max_results=%lu max_candidates=%lu "
            "max_touched_locks=%lu build=%.6fs group=%.6fs insert=%.6fs "
            "sol=%.6fs scalar=%.6fs total=%.6fs\n",
            (unsigned long)bgj_cuda_uid_batch_profile.buckets,
            (unsigned long)bgj_cuda_uid_batch_profile.batched_buckets,
            (unsigned long)bgj_cuda_uid_batch_profile.scalar_buckets,
            (unsigned long)bgj_cuda_uid_batch_profile.results,
            (unsigned long)bgj_cuda_uid_batch_profile.candidates,
            (unsigned long)bgj_cuda_uid_batch_profile.batched_candidates,
            (unsigned long)bgj_cuda_uid_batch_profile.scalar_candidates,
            (unsigned long)bgj_cuda_uid_batch_profile.accepted,
            (unsigned long)bgj_cuda_uid_batch_profile.touched_locks,
            (unsigned long)bgj_cuda_uid_batch_profile.max_results,
            (unsigned long)bgj_cuda_uid_batch_profile.max_candidates,
            (unsigned long)bgj_cuda_uid_batch_profile.max_touched_locks,
            bgj_cuda_uid_batch_profile.build_sec,
            bgj_cuda_uid_batch_profile.group_sec,
            bgj_cuda_uid_batch_profile.insert_sec,
            bgj_cuda_uid_batch_profile.sol_sec,
            bgj_cuda_uid_batch_profile.scalar_sec,
            bgj_cuda_uid_batch_profile.total_sec);
    pthread_mutex_unlock(&bgj_cuda_uid_batch_profile.lock);
}

static void bgj_cuda_consume_profile_dump()
{
    if (!bgj_cuda_consume_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_consume_profile.lock);
    fprintf(stderr,
            "cuda_consume_profile: buckets=%lu sorted_buckets=%lu radix_buckets=%lu "
            "stdsort_buckets=%lu batched_uid_buckets=%lu scalar_uid_buckets=%lu "
            "results=%lu sorted_results=%lu linear_rank_buckets=%lu table_rank_buckets=%lu "
            "uid_candidates=%lu uid_accepted=%lu "
            "uid_touched_locks=%lu max_results=%lu rank=%.6fs key=%.6fs "
            "radix=%.6fs stdsort=%.6fs write=%.6fs sort=%.6fs "
            "uid_build=%.6fs uid_group=%.6fs uid_insert=%.6fs uid_sol=%.6fs "
            "uid_scalar=%.6fs uid_total=%.6fs total=%.6fs\n",
            (unsigned long)bgj_cuda_consume_profile.buckets,
            (unsigned long)bgj_cuda_consume_profile.sorted_buckets,
            (unsigned long)bgj_cuda_consume_profile.radix_buckets,
            (unsigned long)bgj_cuda_consume_profile.stdsort_buckets,
            (unsigned long)bgj_cuda_consume_profile.batched_uid_buckets,
            (unsigned long)bgj_cuda_consume_profile.scalar_uid_buckets,
            (unsigned long)bgj_cuda_consume_profile.results,
            (unsigned long)bgj_cuda_consume_profile.sorted_results,
            (unsigned long)bgj_cuda_consume_profile.linear_rank_buckets,
            (unsigned long)bgj_cuda_consume_profile.table_rank_buckets,
            (unsigned long)bgj_cuda_consume_profile.uid_candidates,
            (unsigned long)bgj_cuda_consume_profile.uid_accepted,
            (unsigned long)bgj_cuda_consume_profile.uid_touched_locks,
            (unsigned long)bgj_cuda_consume_profile.max_results,
            bgj_cuda_consume_profile.rank_sec,
            bgj_cuda_consume_profile.key_sec,
            bgj_cuda_consume_profile.radix_sec,
            bgj_cuda_consume_profile.stdsort_sec,
            bgj_cuda_consume_profile.write_sec,
            bgj_cuda_consume_profile.sort_sec,
            bgj_cuda_consume_profile.uid_build_sec,
            bgj_cuda_consume_profile.uid_group_sec,
            bgj_cuda_consume_profile.uid_insert_sec,
            bgj_cuda_consume_profile.uid_sol_sec,
            bgj_cuda_consume_profile.uid_scalar_sec,
            bgj_cuda_consume_profile.uid_total_sec,
            bgj_cuda_consume_profile.total_sec);
    pthread_mutex_unlock(&bgj_cuda_consume_profile.lock);
}

static void bgj_cuda_host_profile_register()
{
    if (!bgj_cuda_host_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_host_profile.lock);
    if (!bgj_cuda_host_profile.registered) {
        bgj_cuda_host_profile.registered = 1;
        atexit(bgj_cuda_host_profile_dump);
    }
    pthread_mutex_unlock(&bgj_cuda_host_profile.lock);
}

static void bgj_cuda_uid_batch_profile_register()
{
    if (!bgj_cuda_uid_batch_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_uid_batch_profile.lock);
    if (!bgj_cuda_uid_batch_profile.registered) {
        bgj_cuda_uid_batch_profile.registered = 1;
        atexit(bgj_cuda_uid_batch_profile_dump);
    }
    pthread_mutex_unlock(&bgj_cuda_uid_batch_profile.lock);
}

static void bgj_cuda_consume_profile_register()
{
    if (!bgj_cuda_consume_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_consume_profile.lock);
    if (!bgj_cuda_consume_profile.registered) {
        bgj_cuda_consume_profile.registered = 1;
        atexit(bgj_cuda_consume_profile_dump);
    }
    pthread_mutex_unlock(&bgj_cuda_consume_profile.lock);
}

static void bgj_cuda_host_profile_record(uint32_t result_count,
                                         uint64_t try_add2,
                                         uint64_t try_add3,
                                         uint64_t succ_add2,
                                         uint64_t succ_add3,
                                         const uint64_t *type_count,
                                         uint64_t bucket_uid_unique,
                                         uint64_t bucket_uid_dup,
                                         double sort_sec,
                                         double bucket_uid_sec,
                                         double uid_sec,
                                         double sol_sec,
                                         double consume_sec)
{
    if (!bgj_cuda_host_profile_requested()) return;
    bgj_cuda_host_profile_register();

    pthread_mutex_lock(&bgj_cuda_host_profile.lock);
    bgj_cuda_host_profile.buckets++;
    bgj_cuda_host_profile.results += result_count;
    bgj_cuda_host_profile.try_add2 += try_add2;
    bgj_cuda_host_profile.try_add3 += try_add3;
    bgj_cuda_host_profile.succ_add2 += succ_add2;
    bgj_cuda_host_profile.succ_add3 += succ_add3;
    for (uint32_t i = 0; i < 5; i++) {
        bgj_cuda_host_profile.type_count[i] += type_count ? type_count[i] : 0;
    }
    bgj_cuda_host_profile.bucket_uid_unique += bucket_uid_unique;
    bgj_cuda_host_profile.bucket_uid_dup += bucket_uid_dup;
    bgj_cuda_host_profile.sort_sec += sort_sec;
    bgj_cuda_host_profile.bucket_uid_sec += bucket_uid_sec;
    bgj_cuda_host_profile.uid_sec += uid_sec;
    bgj_cuda_host_profile.sol_sec += sol_sec;
    bgj_cuda_host_profile.consume_sec += consume_sec;
    pthread_mutex_unlock(&bgj_cuda_host_profile.lock);
}

static void bgj_cuda_uid_batch_profile_record(int used_batch,
                                              uint64_t bucket_count,
                                              uint64_t result_count,
                                              uint64_t candidates,
                                              uint64_t accepted,
                                              uint64_t touched_locks,
                                              double build_sec,
                                              double group_sec,
                                              double insert_sec,
                                              double sol_sec,
                                              double scalar_sec,
                                              double total_sec)
{
    if (!bgj_cuda_uid_batch_profile_requested()) return;
    bgj_cuda_uid_batch_profile_register();

    pthread_mutex_lock(&bgj_cuda_uid_batch_profile.lock);
    bgj_cuda_uid_batch_profile.buckets += bucket_count;
    if (used_batch) {
        bgj_cuda_uid_batch_profile.batched_buckets += bucket_count;
        bgj_cuda_uid_batch_profile.batched_candidates += candidates;
    } else {
        bgj_cuda_uid_batch_profile.scalar_buckets += bucket_count;
        bgj_cuda_uid_batch_profile.scalar_candidates += candidates;
    }
    bgj_cuda_uid_batch_profile.results += result_count;
    bgj_cuda_uid_batch_profile.candidates += candidates;
    bgj_cuda_uid_batch_profile.accepted += accepted;
    bgj_cuda_uid_batch_profile.touched_locks += touched_locks;
    if (result_count > bgj_cuda_uid_batch_profile.max_results) {
        bgj_cuda_uid_batch_profile.max_results = result_count;
    }
    if (candidates > bgj_cuda_uid_batch_profile.max_candidates) {
        bgj_cuda_uid_batch_profile.max_candidates = candidates;
    }
    if (touched_locks > bgj_cuda_uid_batch_profile.max_touched_locks) {
        bgj_cuda_uid_batch_profile.max_touched_locks = touched_locks;
    }
    bgj_cuda_uid_batch_profile.build_sec += build_sec;
    bgj_cuda_uid_batch_profile.group_sec += group_sec;
    bgj_cuda_uid_batch_profile.insert_sec += insert_sec;
    bgj_cuda_uid_batch_profile.sol_sec += sol_sec;
    bgj_cuda_uid_batch_profile.scalar_sec += scalar_sec;
    bgj_cuda_uid_batch_profile.total_sec += total_sec;
    pthread_mutex_unlock(&bgj_cuda_uid_batch_profile.lock);
}

static void bgj_cuda_consume_profile_record(uint32_t result_count,
                                            int sorted,
                                            int used_radix,
                                            int used_linear_rank,
                                            int used_uid_batch,
                                            uint64_t uid_candidates,
                                            uint64_t uid_accepted,
                                            uint64_t uid_touched_locks,
                                            double rank_sec,
                                            double key_sec,
                                            double radix_sec,
                                            double stdsort_sec,
                                            double write_sec,
                                            double sort_sec,
                                            double uid_build_sec,
                                            double uid_group_sec,
                                            double uid_insert_sec,
                                            double uid_sol_sec,
                                            double uid_scalar_sec,
                                            double uid_total_sec,
                                            double total_sec)
{
    if (!bgj_cuda_consume_profile_requested()) return;
    bgj_cuda_consume_profile_register();

    pthread_mutex_lock(&bgj_cuda_consume_profile.lock);
    bgj_cuda_consume_profile.buckets++;
    if (sorted) {
        bgj_cuda_consume_profile.sorted_buckets++;
        bgj_cuda_consume_profile.sorted_results += result_count;
        if (used_linear_rank) {
            bgj_cuda_consume_profile.linear_rank_buckets++;
        } else {
            bgj_cuda_consume_profile.table_rank_buckets++;
        }
        if (used_radix) {
            bgj_cuda_consume_profile.radix_buckets++;
        } else {
            bgj_cuda_consume_profile.stdsort_buckets++;
        }
    }
    if (used_uid_batch) {
        bgj_cuda_consume_profile.batched_uid_buckets++;
    } else {
        bgj_cuda_consume_profile.scalar_uid_buckets++;
    }
    bgj_cuda_consume_profile.results += result_count;
    bgj_cuda_consume_profile.uid_candidates += uid_candidates;
    bgj_cuda_consume_profile.uid_accepted += uid_accepted;
    bgj_cuda_consume_profile.uid_touched_locks += uid_touched_locks;
    if (result_count > bgj_cuda_consume_profile.max_results) {
        bgj_cuda_consume_profile.max_results = result_count;
    }
    bgj_cuda_consume_profile.rank_sec += rank_sec;
    bgj_cuda_consume_profile.key_sec += key_sec;
    bgj_cuda_consume_profile.radix_sec += radix_sec;
    bgj_cuda_consume_profile.stdsort_sec += stdsort_sec;
    bgj_cuda_consume_profile.write_sec += write_sec;
    bgj_cuda_consume_profile.sort_sec += sort_sec;
    bgj_cuda_consume_profile.uid_build_sec += uid_build_sec;
    bgj_cuda_consume_profile.uid_group_sec += uid_group_sec;
    bgj_cuda_consume_profile.uid_insert_sec += uid_insert_sec;
    bgj_cuda_consume_profile.uid_sol_sec += uid_sol_sec;
    bgj_cuda_consume_profile.uid_scalar_sec += uid_scalar_sec;
    bgj_cuda_consume_profile.uid_total_sec += uid_total_sec;
    bgj_cuda_consume_profile.total_sec += total_sec;
    pthread_mutex_unlock(&bgj_cuda_consume_profile.lock);
}

extern "C" int bgj_cuda_raw_device_count();
extern "C" int bgj_cuda_raw_execution_device_count();
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
                                           uint32_t center_id,
                                           int32_t center_norm,
                                           int record_dp,
                                           int transform_dp,
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
                                                uint32_t center_id,
                                                int32_t center_norm,
                                                int record_dp,
                                                int transform_dp,
                                                bgj_cuda_result_t *results,
                                                uint32_t result_capacity,
                                                uint32_t *result_count,
                                                int *overflow);
extern "C" int bgj_cuda_search_bucket_pool_raw_submit_async(const int8_t *pool_vecs,
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
                                                            uint32_t center_id,
                                                            int32_t center_norm,
                                                            int record_dp,
                                                            int transform_dp,
                                                            int device_transform_dp,
                                                            int raw_center_dp,
                                                            int record_dot_copy_event,
                                                            uint32_t result_capacity,
                                                            uint32_t *submitted_result_count,
                                                            int *submitted_overflow);
extern "C" int bgj_cuda_search_bucket_raw_wait_dot_copy_async();
extern "C" int bgj_cuda_search_bucket_raw_finish_async(bgj_cuda_result_t *results,
                                                       uint32_t result_capacity,
                                                       uint32_t *submitted_result_count,
                                                       int *submitted_overflow,
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
                                                      const uint32_t *center_id,
                                                      const int32_t *center_norm,
                                                      int record_dp,
                                                      int transform_dp,
                                                      bgj_cuda_result_t *const *results,
                                                      const uint32_t *result_capacity,
                                                      uint32_t *result_count,
                                                      int *overflow);
extern "C" void *bgj_cuda_alloc_pinned_host_raw(size_t bytes);
extern "C" void bgj_cuda_free_pinned_host_raw(void *ptr);

static int bgj_cuda_pinned_results_requested()
{
    static const int requested = []() {
        const char *env = getenv("BGJ_CUDA_PINNED_RESULTS");
        if (env && env[0]) return env[0] != '0' ? 1 : 0;
        return 1;
    }();
    return requested;
}

struct bgj_cuda_result_storage_t {
    bgj_cuda_result_t *ptr;
    size_t capacity;
    int pinned;

    bgj_cuda_result_storage_t()
        : ptr(NULL),
          capacity(0),
          pinned(0)
    {
    }

    ~bgj_cuda_result_storage_t()
    {
        release();
    }

    void release()
    {
        if (ptr) {
            if (pinned) {
                bgj_cuda_free_pinned_host_raw(ptr);
            } else {
                free(ptr);
            }
        }
        ptr = NULL;
        capacity = 0;
        pinned = 0;
    }

    int reserve(size_t requested)
    {
        if (requested <= capacity) return 1;
        if (requested > std::numeric_limits<size_t>::max() / sizeof(bgj_cuda_result_t)) {
            return 0;
        }
        release();
        if (requested == 0) return 1;

        const size_t bytes = requested * sizeof(bgj_cuda_result_t);
        if (bgj_cuda_pinned_results_requested()) {
            ptr = (bgj_cuda_result_t *)bgj_cuda_alloc_pinned_host_raw(bytes);
            if (ptr) pinned = 1;
        }
        if (!ptr) {
            ptr = (bgj_cuda_result_t *)malloc(bytes);
            pinned = 0;
        }
        if (!ptr) {
            capacity = 0;
            return 0;
        }
        capacity = requested;
        return 1;
    }
};

int bgj_cuda_device_count()
{
    static int count = -1;
    if (count < 0) count = bgj_cuda_raw_device_count();
    return count;
}

int bgj_cuda_execution_device_count()
{
    return bgj_cuda_raw_execution_device_count();
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
    return 1u << 20;
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
    if (env && env[0]) return env[0] != '0';
    return bgj_cuda_pinned_results_requested();
}

static int bgj_cuda_radix_sort_requested()
{
    const char *env = getenv("BGJ_CUDA_RADIX_SORT");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static uint32_t bgj_cuda_radix_sort_min_results()
{
    static const uint32_t min_results = []() {
        const char *env = getenv("BGJ_CUDA_RADIX_SORT_MIN_RESULTS");
        if (env && env[0]) {
            char *end = NULL;
            const unsigned long parsed = strtoul(env, &end, 10);
            if (end != env && parsed > 0 && parsed <= 0xfffffffful) {
                return (uint32_t)parsed;
            }
        }
        return (uint32_t)1024;
    }();
    return min_results;
}

static uint32_t bgj_cuda_linear_rank_max_results()
{
    static const uint32_t max_results = []() {
        const char *env = getenv("BGJ_CUDA_LINEAR_RANK_MAX_RESULTS");
        if (env && env[0]) {
            char *end = NULL;
            const unsigned long parsed = strtoul(env, &end, 10);
            if (end != env && parsed <= 0xfffffffful) {
                return (uint32_t)parsed;
            }
        }
        return (uint32_t)64;
    }();
    return max_results;
}

int bgj_cuda_materialize_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE");
    if (env && env[0]) return env[0] != '0';
    return bgj_cuda_search_requested();
}

int bgj_cuda_bucket_requested()
{
    const char *env = getenv("BGJ_CUDA_BUCKET");
    if (env && env[0]) return env[0] != '0';
    env = getenv("BGJ_CUDA_BUCKETING");
    if (env && env[0]) return env[0] != '0';
    return bgj_cuda_search_requested();
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

int bgj_cuda_cred_transform_requested()
{
    const char *env = getenv("BGJ_CUDA_CRED");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

int bgj_cuda_overlap_cred_requested()
{
    const char *env = getenv("BGJ_CUDA_OVERLAP_CRED");
    if (env && env[0]) return env[0] != '0';
    return 1;
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

static inline void bgj_cuda_result_sort_keys_from_ranks(const bgj_cuda_result_t &result,
                                                        int32_t px,
                                                        int32_t py,
                                                        int32_t nx,
                                                        int32_t ny,
                                                        int32_t &phase,
                                                        int32_t &rank0,
                                                        int32_t &rank1)
{
    switch (result.type) {
    case BGJ_CUDA_SOL_A:
        phase = 0;
        rank0 = px;
        rank1 = ny;
        return;
    case BGJ_CUDA_SOL_SA:
        phase = 1;
        rank0 = px;
        rank1 = ny;
        return;
    case BGJ_CUDA_SOL_S:
        if (px >= 0 && py >= 0) {
            phase = 2;
            rank0 = px;
            rank1 = py;
            return;
        }
        phase = (nx >= 0 && ny >= 0) ? 4 : 6;
        rank0 = nx;
        rank1 = ny;
        return;
    case BGJ_CUDA_SOL_SS:
        phase = 3;
        rank0 = px;
        rank1 = py;
        return;
    case BGJ_CUDA_SOL_AA:
        phase = 5;
        rank0 = nx;
        rank1 = ny;
        return;
    default:
        phase = 7;
        rank0 = -1;
        rank1 = -1;
        return;
    }
}

static inline int32_t bgj_cuda_target_rank_of(const std::vector<uint32_t> &target_stamp,
                                              const std::vector<uint32_t> &target_slot,
                                              const std::vector<int32_t> &target_rank,
                                              uint32_t epoch,
                                              uint32_t id)
{
    if (id >= target_stamp.size() || target_stamp[id] != epoch) return -1;
    return target_rank[target_slot[id]];
}

struct bgj_cuda_result_sort_item_t {
    int32_t phase;
    int32_t rank0;
    int32_t rank1;
    bgj_cuda_result_t result;
};

enum bgj_cuda_radix_sort_field_t {
    BGJ_CUDA_SORT_FIELD_Y = 0,
    BGJ_CUDA_SORT_FIELD_X = 1,
    BGJ_CUDA_SORT_FIELD_TYPE = 2,
    BGJ_CUDA_SORT_FIELD_RANK1 = 3,
    BGJ_CUDA_SORT_FIELD_RANK0 = 4,
    BGJ_CUDA_SORT_FIELD_PHASE = 5
};

struct bgj_cuda_radix_sort_pass_t {
    uint8_t field;
    uint8_t shift;
};

static inline uint32_t bgj_cuda_sort_rank_key(int32_t rank)
{
    // Ranks are either -1 or local nonnegative indices; +1 preserves that order.
    return (uint32_t)((int64_t)rank + 1);
}

static inline uint32_t bgj_cuda_sort_field_key(const bgj_cuda_result_sort_item_t &item,
                                               uint8_t field)
{
    switch (field) {
    case BGJ_CUDA_SORT_FIELD_Y:
        return item.result.y;
    case BGJ_CUDA_SORT_FIELD_X:
        return item.result.x;
    case BGJ_CUDA_SORT_FIELD_TYPE:
        return item.result.type;
    case BGJ_CUDA_SORT_FIELD_RANK1:
        return bgj_cuda_sort_rank_key(item.rank1);
    case BGJ_CUDA_SORT_FIELD_RANK0:
        return bgj_cuda_sort_rank_key(item.rank0);
    case BGJ_CUDA_SORT_FIELD_PHASE:
        return (uint32_t)item.phase;
    default:
        return 0;
    }
}

static inline uint32_t bgj_cuda_sort_key_byte(const bgj_cuda_result_sort_item_t &item,
                                              const bgj_cuda_radix_sort_pass_t &pass)
{
    return (bgj_cuda_sort_field_key(item, pass.field) >> pass.shift) & 0xffu;
}

static inline void bgj_cuda_radix_add_field_passes(bgj_cuda_radix_sort_pass_t *passes,
                                                   uint32_t *num_passes,
                                                   uint8_t field,
                                                   uint32_t mask)
{
    for (uint8_t shift = 0; mask; shift += 8, mask >>= 8) {
        passes[(*num_passes)++] = {field, shift};
    }
}

static void bgj_cuda_radix_sort_items(std::vector<bgj_cuda_result_sort_item_t> &items,
                                      std::vector<bgj_cuda_result_sort_item_t> &scratch,
                                      uint32_t count,
                                      uint32_t y_mask,
                                      uint32_t x_mask,
                                      uint32_t type_mask,
                                      uint32_t rank1_mask,
                                      uint32_t rank0_mask,
                                      uint32_t phase_mask)
{
    bgj_cuda_radix_sort_pass_t passes[21];
    uint32_t num_passes = 0;
    bgj_cuda_radix_add_field_passes(passes, &num_passes, BGJ_CUDA_SORT_FIELD_Y, y_mask);
    bgj_cuda_radix_add_field_passes(passes, &num_passes, BGJ_CUDA_SORT_FIELD_X, x_mask);
    bgj_cuda_radix_add_field_passes(passes, &num_passes, BGJ_CUDA_SORT_FIELD_TYPE, type_mask);
    bgj_cuda_radix_add_field_passes(passes, &num_passes, BGJ_CUDA_SORT_FIELD_RANK1, rank1_mask);
    bgj_cuda_radix_add_field_passes(passes, &num_passes, BGJ_CUDA_SORT_FIELD_RANK0, rank0_mask);
    bgj_cuda_radix_add_field_passes(passes, &num_passes, BGJ_CUDA_SORT_FIELD_PHASE, phase_mask);

    bgj_cuda_result_sort_item_t *src = items.data();
    bgj_cuda_result_sort_item_t *dst = scratch.data();
    bool in_scratch = false;

    for (uint32_t pass = 0; pass < num_passes; pass++) {
        uint32_t counts[256];
        uint32_t offsets[256];
        memset(counts, 0, sizeof(counts));

        for (uint32_t i = 0; i < count; i++) {
            counts[bgj_cuda_sort_key_byte(src[i], passes[pass])]++;
        }

        uint32_t offset = 0;
        for (uint32_t i = 0; i < 256; i++) {
            offsets[i] = offset;
            offset += counts[i];
        }

        for (uint32_t i = 0; i < count; i++) {
            const uint32_t key = bgj_cuda_sort_key_byte(src[i], passes[pass]);
            dst[offsets[key]++] = src[i];
        }

        bgj_cuda_result_sort_item_t *tmp = src;
        src = dst;
        dst = tmp;
        in_scratch = !in_scratch;
    }

    if (in_scratch) {
        std::copy(scratch.begin(), scratch.begin() + count, items.begin());
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
    const int host_profile = bgj_cuda_host_profile_requested();
    const int consume_profile = !host_profile && bgj_cuda_consume_profile_requested();
    const int uid_batch_profile = !host_profile && bgj_cuda_uid_batch_profile_requested();
    const int uid_timing_profile = uid_batch_profile || consume_profile;
    const double consume_t0 = host_profile ? bgj_cuda_host_wall_time() : 0.0;
    const double consume_profile_t0 = consume_profile ? bgj_cuda_host_wall_time() : 0.0;
    double sort_sec = 0.0;
    double consume_rank_sec = 0.0;
    double consume_key_sec = 0.0;
    double consume_radix_sec = 0.0;
    double consume_stdsort_sec = 0.0;
    double consume_write_sec = 0.0;
    int consume_sorted = 0;
    int consume_used_radix = 0;
    int consume_used_linear_rank = 0;
    const uint32_t num_p = (uint32_t)bkt->num_pvec;
    const uint32_t num_n = (uint32_t)bkt->num_nvec;
    const int sort_results = bgj_cuda_sort_results_requested();

    if (result_count > 1 && sort_results) {
        consume_sorted = 1;
        const double sort_t0 = bgj_cuda_host_wall_time();
        const double rank_t0 = consume_profile ? bgj_cuda_host_wall_time() : 0.0;
        static thread_local std::vector<int32_t> p_rank;
        static thread_local std::vector<int32_t> n_rank;
        static thread_local std::vector<uint32_t> p_rank_stamp;
        static thread_local std::vector<uint32_t> n_rank_stamp;
        static thread_local std::vector<bgj_cuda_result_sort_item_t> sort_items;
        static thread_local std::vector<bgj_cuda_result_sort_item_t> sort_phase_items;
        static thread_local std::vector<uint32_t> target_stamp;
        static thread_local std::vector<uint32_t> target_slot;
        static thread_local std::vector<int32_t> target_p_rank;
        static thread_local std::vector<int32_t> target_n_rank;
        static thread_local uint32_t rank_epoch = 1;
        static thread_local uint32_t target_epoch = 1;

        const uint32_t linear_rank_max_results = bgj_cuda_linear_rank_max_results();
        const int use_linear_rank = linear_rank_max_results != 0 &&
                                    result_count <= linear_rank_max_results;
        consume_used_linear_rank = use_linear_rank;
        if (use_linear_rank) {
            target_epoch++;
            if (target_epoch == 0) {
                std::fill(target_stamp.begin(), target_stamp.end(), 0);
                target_epoch = 1;
            }
            try {
                if (target_stamp.size() < (size_t)pool->num_vec) {
                    target_stamp.resize((size_t)pool->num_vec, 0);
                    target_slot.resize((size_t)pool->num_vec, 0);
                }
                target_p_rank.clear();
                target_n_rank.clear();
                target_p_rank.reserve((size_t)result_count * 2u);
                target_n_rank.reserve((size_t)result_count * 2u);
            } catch (...) {
                return 0;
            }
            auto add_target = [&](uint32_t id) {
                if (id >= (uint32_t)target_stamp.size()) return;
                if (target_stamp[id] == target_epoch) return;
                target_stamp[id] = target_epoch;
                target_slot[id] = (uint32_t)target_p_rank.size();
                target_p_rank.push_back(-1);
                target_n_rank.push_back(-1);
            };
            for (uint32_t i = 0; i < result_count; i++) {
                add_target(results[i].x);
                add_target(results[i].y);
            }
            for (uint32_t i = 0; i < num_p; i++) {
                const uint32_t id = bkt->pvec[i];
                if (id < (uint32_t)target_stamp.size() && target_stamp[id] == target_epoch) {
                    target_p_rank[target_slot[id]] = (int32_t)i;
                }
            }
            for (uint32_t i = 0; i < num_n; i++) {
                const uint32_t id = bkt->nvec[i];
                if (id < (uint32_t)target_stamp.size() && target_stamp[id] == target_epoch) {
                    target_n_rank[target_slot[id]] = (int32_t)i;
                }
            }
        } else {
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
        }
        if (consume_profile) {
            consume_rank_sec += bgj_cuda_host_wall_time() - rank_t0;
        }
        try {
            sort_items.resize((size_t)result_count);
            sort_phase_items.resize((size_t)result_count);
        } catch (...) {
            return 0;
        }
        uint32_t phase_count[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t y_mask = 0;
        uint32_t x_mask = 0;
        uint32_t type_mask = 0;
        uint32_t rank1_mask = 0;
        uint32_t rank0_mask = 0;
        uint32_t phase_mask = 0;
        const double key_t0 = consume_profile ? bgj_cuda_host_wall_time() : 0.0;
        for (uint32_t i = 0; i < result_count; i++) {
            bgj_cuda_result_sort_item_t &item = sort_items[i];
            item.result = results[i];
            if (use_linear_rank) {
                const int32_t px = bgj_cuda_target_rank_of(target_stamp,
                                                           target_slot,
                                                           target_p_rank,
                                                           target_epoch,
                                                           item.result.x);
                const int32_t py = bgj_cuda_target_rank_of(target_stamp,
                                                           target_slot,
                                                           target_p_rank,
                                                           target_epoch,
                                                           item.result.y);
                const int32_t nx = bgj_cuda_target_rank_of(target_stamp,
                                                           target_slot,
                                                           target_n_rank,
                                                           target_epoch,
                                                           item.result.x);
                const int32_t ny = bgj_cuda_target_rank_of(target_stamp,
                                                           target_slot,
                                                           target_n_rank,
                                                           target_epoch,
                                                           item.result.y);
                bgj_cuda_result_sort_keys_from_ranks(item.result,
                                                     px,
                                                     py,
                                                     nx,
                                                     ny,
                                                     item.phase,
                                                     item.rank0,
                                                     item.rank1);
            } else {
                item.phase = bgj_cuda_result_phase(item.result,
                                                   p_rank,
                                                   p_rank_stamp,
                                                   n_rank,
                                                   n_rank_stamp,
                                                   rank_epoch);
                item.rank0 = bgj_cuda_local_rank(item.result,
                                                 p_rank,
                                                 p_rank_stamp,
                                                 n_rank,
                                                 n_rank_stamp,
                                                 rank_epoch,
                                                 1);
                item.rank1 = bgj_cuda_local_rank(item.result,
                                                 p_rank,
                                                 p_rank_stamp,
                                                 n_rank,
                                                 n_rank_stamp,
                                                 rank_epoch,
                                                 0);
            }
            phase_count[(uint32_t)item.phase]++;
            y_mask |= item.result.y;
            x_mask |= item.result.x;
            type_mask |= item.result.type;
            rank1_mask |= bgj_cuda_sort_rank_key(item.rank1);
            rank0_mask |= bgj_cuda_sort_rank_key(item.rank0);
            phase_mask |= (uint32_t)item.phase;
        }
        if (consume_profile) {
            consume_key_sec += bgj_cuda_host_wall_time() - key_t0;
        }

        const int use_radix_sort = bgj_cuda_radix_sort_requested() &&
                                   result_count >= bgj_cuda_radix_sort_min_results();
        if (use_radix_sort) {
            consume_used_radix = 1;
            const double radix_t0 = consume_profile ? bgj_cuda_host_wall_time() : 0.0;
            bgj_cuda_radix_sort_items(sort_items,
                                      sort_phase_items,
                                      result_count,
                                      y_mask,
                                      x_mask,
                                      type_mask,
                                      rank1_mask,
                                      rank0_mask,
                                      phase_mask);
            if (consume_profile) {
                consume_radix_sec += bgj_cuda_host_wall_time() - radix_t0;
            }
        } else {
            const double stdsort_t0 = consume_profile ? bgj_cuda_host_wall_time() : 0.0;
            uint32_t phase_begin[9];
            phase_begin[0] = 0;
            for (uint32_t phase = 0; phase < 8; phase++) {
                phase_begin[phase + 1] = phase_begin[phase] + phase_count[phase];
            }
            uint32_t phase_cursor[8];
            for (uint32_t phase = 0; phase < 8; phase++) {
                phase_cursor[phase] = phase_begin[phase];
            }
            for (uint32_t i = 0; i < result_count; i++) {
                const uint32_t phase = (uint32_t)sort_items[i].phase;
                sort_phase_items[phase_cursor[phase]++] = sort_items[i];
            }
            for (uint32_t phase = 0; phase < 8; phase++) {
                bgj_cuda_result_sort_item_t *begin = sort_phase_items.data() + phase_begin[phase];
                bgj_cuda_result_sort_item_t *end = sort_phase_items.data() + phase_begin[phase + 1];
                if (end - begin <= 1) continue;
                std::sort(begin, end,
                          [](const bgj_cuda_result_sort_item_t &a,
                             const bgj_cuda_result_sort_item_t &b) {
                              if (a.rank0 != b.rank0) return a.rank0 < b.rank0;
                              if (a.rank1 != b.rank1) return a.rank1 < b.rank1;
                              if (a.result.type != b.result.type) return a.result.type < b.result.type;
                              if (a.result.x != b.result.x) return a.result.x < b.result.x;
                              return a.result.y < b.result.y;
                          });
            }
            sort_items.swap(sort_phase_items);
            if (consume_profile) {
                consume_stdsort_sec += bgj_cuda_host_wall_time() - stdsort_t0;
            }
        }
        const double write_t0 = consume_profile ? bgj_cuda_host_wall_time() : 0.0;
        for (uint32_t i = 0; i < result_count; i++) {
            results[i] = sort_items[i].result;
        }
        if (consume_profile) {
            consume_write_sec += bgj_cuda_host_wall_time() - write_t0;
        }
        sort_sec = bgj_cuda_host_wall_time() - sort_t0;
    }

    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;
    uint64_t type_count[5] = {0, 0, 0, 0, 0};
    uint64_t bucket_uid_unique = 0;
    uint64_t bucket_uid_dup = 0;
    double bucket_uid_sec = 0.0;
    double uid_sec = 0.0;
    double sol_sec = 0.0;
    int uid_profile_used_batch = 0;
    uint64_t uid_profile_candidates = 0;
    uint64_t uid_profile_accepted = 0;
    uint64_t uid_profile_touched_locks = 0;
    double uid_profile_build_sec = 0.0;
    double uid_profile_group_sec = 0.0;
    double uid_profile_insert_sec = 0.0;
    double uid_profile_sol_sec = 0.0;
    double uid_profile_scalar_sec = 0.0;
    double uid_profile_total_sec = 0.0;

    if (host_profile) {
        static thread_local std::unordered_set<uint64_t> bucket_uid_seen;
        int bucket_uid_profile = result_count > 0;
        if (bucket_uid_profile) {
            try {
                bucket_uid_seen.clear();
                if (bucket_uid_seen.bucket_count() < (size_t)result_count) {
                    bucket_uid_seen.reserve((size_t)result_count);
                }
            } catch (...) {
                bucket_uid_seen.clear();
                bucket_uid_profile = 0;
            }
        }
        auto profile_bucket_uid = [&](uint64_t u) {
            if (!bucket_uid_profile) return;
            const double t0 = bgj_cuda_host_wall_time();
            pool->uid->normalize_uid(u);
            if (bucket_uid_seen.insert(u).second) {
                bucket_uid_unique++;
            } else {
                bucket_uid_dup++;
            }
            bucket_uid_sec += bgj_cuda_host_wall_time() - t0;
        };

        for (uint32_t i = 0; i < result_count; i++) {
            const uint32_t x = results[i].x;
            const uint32_t y = results[i].y;
            const uint32_t type = results[i].type;
            if (type < 5) type_count[type]++;
            switch (type) {
            case BGJ_CUDA_SOL_A: {
                try_add2++;
                const uint64_t u = pool->vu[x] + pool->vu[y];
                profile_bucket_uid(u);
                double t0 = bgj_cuda_host_wall_time();
                const int inserted = bgj_cuda_insert_uid_fast(pool->uid, u);
                uid_sec += bgj_cuda_host_wall_time() - t0;
                if (inserted) {
                    succ_add2++;
                    t0 = bgj_cuda_host_wall_time();
                    sol->add_sol_a(x, y);
                    sol_sec += bgj_cuda_host_wall_time() - t0;
                }
                break;
            }
            case BGJ_CUDA_SOL_S: {
                try_add2++;
                const uint64_t u = pool->vu[x] - pool->vu[y];
                profile_bucket_uid(u);
                double t0 = bgj_cuda_host_wall_time();
                const int inserted = bgj_cuda_insert_uid_fast(pool->uid, u);
                uid_sec += bgj_cuda_host_wall_time() - t0;
                if (inserted) {
                    succ_add2++;
                    t0 = bgj_cuda_host_wall_time();
                    sol->add_sol_s(x, y);
                    sol_sec += bgj_cuda_host_wall_time() - t0;
                }
                break;
            }
            case BGJ_CUDA_SOL_AA: {
                try_add3++;
                const uint64_t u = bkt->center_u + pool->vu[x] + pool->vu[y];
                profile_bucket_uid(u);
                double t0 = bgj_cuda_host_wall_time();
                const int inserted = bgj_cuda_insert_uid_fast(pool->uid, u);
                uid_sec += bgj_cuda_host_wall_time() - t0;
                if (inserted) {
                    succ_add3++;
                    t0 = bgj_cuda_host_wall_time();
                    sol->add_sol_aa(bkt->center_ind, x, y);
                    sol_sec += bgj_cuda_host_wall_time() - t0;
                }
                break;
            }
            case BGJ_CUDA_SOL_SA: {
                try_add3++;
                const uint64_t u = bkt->center_u - pool->vu[x] + pool->vu[y];
                profile_bucket_uid(u);
                double t0 = bgj_cuda_host_wall_time();
                const int inserted = bgj_cuda_insert_uid_fast(pool->uid, u);
                uid_sec += bgj_cuda_host_wall_time() - t0;
                if (inserted) {
                    succ_add3++;
                    t0 = bgj_cuda_host_wall_time();
                    sol->add_sol_sa(bkt->center_ind, x, y);
                    sol_sec += bgj_cuda_host_wall_time() - t0;
                }
                break;
            }
            case BGJ_CUDA_SOL_SS: {
                try_add3++;
                const uint64_t u = bkt->center_u - pool->vu[x] - pool->vu[y];
                profile_bucket_uid(u);
                double t0 = bgj_cuda_host_wall_time();
                const int inserted = bgj_cuda_insert_uid_fast(pool->uid, u);
                uid_sec += bgj_cuda_host_wall_time() - t0;
                if (inserted) {
                    succ_add3++;
                    t0 = bgj_cuda_host_wall_time();
                    sol->add_sol_ss(bkt->center_ind, x, y);
                    sol_sec += bgj_cuda_host_wall_time() - t0;
                }
                break;
            }
            default:
                break;
            }
        }
    } else {
        const double uid_profile_total_t0 = uid_timing_profile ? bgj_cuda_host_wall_time() : 0.0;
        const uint32_t uid_batch_min_results =
            sort_results ? bgj_cuda_sorted_uid_batch_min_results() : bgj_cuda_uid_batch_min_results();
        int use_uid_batch = bgj_cuda_uid_batch_requested() &&
                            result_count >= uid_batch_min_results;
        if (use_uid_batch) {
            static thread_local std::vector<bgj_cuda_uid_candidate_t> candidates;
            static thread_local std::vector<uint32_t> order;
            static thread_local std::vector<uint32_t> touched_slots;
            static thread_local std::vector<uint32_t> slot_counts;
            static thread_local std::vector<uint32_t> slot_offsets;
            static thread_local std::vector<uint32_t> slot_cursor;

            try {
                if (slot_counts.size() != UidHashTable::NUM_UID_LOCK) {
                    slot_counts.assign(UidHashTable::NUM_UID_LOCK, 0);
                    slot_offsets.assign(UidHashTable::NUM_UID_LOCK, 0);
                    slot_cursor.assign(UidHashTable::NUM_UID_LOCK, 0);
                }
                candidates.clear();
                touched_slots.clear();
                candidates.reserve(result_count);
                touched_slots.reserve(UidHashTable::NUM_UID_LOCK);
                order.resize(result_count);
            } catch (...) {
                use_uid_batch = 0;
            }

            if (use_uid_batch) {
                uid_profile_used_batch = 1;
                double uid_profile_phase_t0 = uid_timing_profile ? bgj_cuda_host_wall_time() : 0.0;
                for (uint32_t i = 0; i < result_count; i++) {
                    const uint32_t x = results[i].x;
                    const uint32_t y = results[i].y;
                    switch (results[i].type) {
                    case BGJ_CUDA_SOL_A:
                        try_add2++;
                        bgj_cuda_add_uid_candidate(candidates,
                                                   touched_slots,
                                                   slot_counts,
                                                   pool->vu[x] + pool->vu[y],
                                                   BGJ_CUDA_SOL_A,
                                                   bkt->center_ind,
                                                   x,
                                                   y);
                        break;
                    case BGJ_CUDA_SOL_S:
                        try_add2++;
                        bgj_cuda_add_uid_candidate(candidates,
                                                   touched_slots,
                                                   slot_counts,
                                                   pool->vu[x] - pool->vu[y],
                                                   BGJ_CUDA_SOL_S,
                                                   bkt->center_ind,
                                                   x,
                                                   y);
                        break;
                    case BGJ_CUDA_SOL_AA:
                        try_add3++;
                        bgj_cuda_add_uid_candidate(candidates,
                                                   touched_slots,
                                                   slot_counts,
                                                   bkt->center_u + pool->vu[x] + pool->vu[y],
                                                   BGJ_CUDA_SOL_AA,
                                                   bkt->center_ind,
                                                   x,
                                                   y);
                        break;
                    case BGJ_CUDA_SOL_SA:
                        try_add3++;
                        bgj_cuda_add_uid_candidate(candidates,
                                                   touched_slots,
                                                   slot_counts,
                                                   bkt->center_u - pool->vu[x] + pool->vu[y],
                                                   BGJ_CUDA_SOL_SA,
                                                   bkt->center_ind,
                                                   x,
                                                   y);
                        break;
                    case BGJ_CUDA_SOL_SS:
                        try_add3++;
                        bgj_cuda_add_uid_candidate(candidates,
                                                   touched_slots,
                                                   slot_counts,
                                                   bkt->center_u - pool->vu[x] - pool->vu[y],
                                                   BGJ_CUDA_SOL_SS,
                                                   bkt->center_ind,
                                                   x,
                                                   y);
                        break;
                    default:
                        break;
                    }
                }
                if (uid_timing_profile) {
                    uid_profile_build_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
                }
                uid_profile_candidates = candidates.size();
                uid_profile_touched_locks = touched_slots.size();

                uid_profile_phase_t0 = uid_timing_profile ? bgj_cuda_host_wall_time() : 0.0;
                uint32_t offset = 0;
                for (uint32_t i = 0; i < touched_slots.size(); i++) {
                    const uint32_t slot = touched_slots[i];
                    slot_offsets[slot] = offset;
                    slot_cursor[slot] = offset;
                    offset += slot_counts[slot];
                }
                for (uint32_t i = 0; i < candidates.size(); i++) {
                    const uint32_t slot = candidates[i].slot;
                    order[slot_cursor[slot]++] = i;
                }
                if (uid_timing_profile) {
                    uid_profile_group_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
                }

                uid_profile_phase_t0 = uid_timing_profile ? bgj_cuda_host_wall_time() : 0.0;
                for (uint32_t i = 0; i < touched_slots.size(); i++) {
                    const uint32_t slot = touched_slots[i];
                    const uint32_t begin = slot_offsets[slot];
                    const uint32_t end = begin + slot_counts[slot];
                    pthread_spin_lock(&pool->uid->uid_lock[slot].a);
                    for (uint32_t j = begin; j < end; j++) {
                        bgj_cuda_uid_candidate_t &candidate = candidates[order[j]];
                        candidate.accepted =
                            pool->uid->uid_table[slot].a.insert(candidate.uid).second ? 1 : 0;
                    }
                    pthread_spin_unlock(&pool->uid->uid_lock[slot].a);
                }
                if (uid_timing_profile) {
                    uid_profile_insert_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
                }

                uid_profile_phase_t0 = uid_timing_profile ? bgj_cuda_host_wall_time() : 0.0;
                uint64_t uid_profile_batch_accepted = 0;
                for (uint32_t i = 0; i < candidates.size(); i++) {
                    if (!candidates[i].accepted) continue;
                    uid_profile_batch_accepted++;
                    if (candidates[i].type == BGJ_CUDA_SOL_A ||
                        candidates[i].type == BGJ_CUDA_SOL_S) {
                        succ_add2++;
                    } else {
                        succ_add3++;
                    }
                    bgj_cuda_add_accepted_solution(sol, candidates[i]);
                }
                uid_profile_accepted = uid_profile_batch_accepted;
                if (uid_timing_profile) {
                    uid_profile_sol_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
                }

                for (uint32_t i = 0; i < touched_slots.size(); i++) {
                    const uint32_t slot = touched_slots[i];
                    slot_counts[slot] = 0;
                    slot_offsets[slot] = 0;
                    slot_cursor[slot] = 0;
                }
            }
        }

        if (!use_uid_batch) {
            const double uid_profile_scalar_t0 = uid_timing_profile ? bgj_cuda_host_wall_time() : 0.0;
            for (uint32_t i = 0; i < result_count; i++) {
                const uint32_t x = results[i].x;
                const uint32_t y = results[i].y;
                switch (results[i].type) {
                case BGJ_CUDA_SOL_A: {
                    try_add2++;
                    uid_profile_candidates++;
                    const uint64_t u = pool->vu[x] + pool->vu[y];
                    if (bgj_cuda_insert_uid_fast(pool->uid, u)) {
                        uid_profile_accepted++;
                        succ_add2++;
                        sol->add_sol_a(x, y);
                    }
                    break;
                }
                case BGJ_CUDA_SOL_S: {
                    try_add2++;
                    uid_profile_candidates++;
                    const uint64_t u = pool->vu[x] - pool->vu[y];
                    if (bgj_cuda_insert_uid_fast(pool->uid, u)) {
                        uid_profile_accepted++;
                        succ_add2++;
                        sol->add_sol_s(x, y);
                    }
                    break;
                }
                case BGJ_CUDA_SOL_AA: {
                    try_add3++;
                    uid_profile_candidates++;
                    const uint64_t u = bkt->center_u + pool->vu[x] + pool->vu[y];
                    if (bgj_cuda_insert_uid_fast(pool->uid, u)) {
                        uid_profile_accepted++;
                        succ_add3++;
                        sol->add_sol_aa(bkt->center_ind, x, y);
                    }
                    break;
                }
                case BGJ_CUDA_SOL_SA: {
                    try_add3++;
                    uid_profile_candidates++;
                    const uint64_t u = bkt->center_u - pool->vu[x] + pool->vu[y];
                    if (bgj_cuda_insert_uid_fast(pool->uid, u)) {
                        uid_profile_accepted++;
                        succ_add3++;
                        sol->add_sol_sa(bkt->center_ind, x, y);
                    }
                    break;
                }
                case BGJ_CUDA_SOL_SS: {
                    try_add3++;
                    uid_profile_candidates++;
                    const uint64_t u = bkt->center_u - pool->vu[x] - pool->vu[y];
                    if (bgj_cuda_insert_uid_fast(pool->uid, u)) {
                        uid_profile_accepted++;
                        succ_add3++;
                        sol->add_sol_ss(bkt->center_ind, x, y);
                    }
                    break;
                }
                default:
                    break;
                }
            }
            if (uid_timing_profile) {
                uid_profile_scalar_sec += bgj_cuda_host_wall_time() - uid_profile_scalar_t0;
            }
        }
        if (uid_timing_profile) {
            uid_profile_total_sec = bgj_cuda_host_wall_time() - uid_profile_total_t0;
        }
        if (uid_batch_profile) {
            bgj_cuda_uid_batch_profile_record(uid_profile_used_batch,
                                              1,
                                              result_count,
                                              uid_profile_candidates,
                                              uid_profile_accepted,
                                              uid_profile_touched_locks,
                                              uid_profile_build_sec,
                                              uid_profile_group_sec,
                                              uid_profile_insert_sec,
                                              uid_profile_sol_sec,
                                              uid_profile_scalar_sec,
                                              uid_profile_total_sec);
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

    if (host_profile) {
        bgj_cuda_host_profile_record(result_count,
                                     try_add2,
                                     try_add3,
                                     succ_add2,
                                     succ_add3,
                                     type_count,
                                     bucket_uid_unique,
                                     bucket_uid_dup,
                                     sort_sec,
                                     bucket_uid_sec,
                                     uid_sec,
                                     sol_sec,
                                     bgj_cuda_host_wall_time() - consume_t0);
    }
    if (consume_profile) {
        bgj_cuda_consume_profile_record(result_count,
                                        consume_sorted,
                                        consume_used_radix,
                                        consume_used_linear_rank,
                                        uid_profile_used_batch,
                                        uid_profile_candidates,
                                        uid_profile_accepted,
                                        uid_profile_touched_locks,
                                        consume_rank_sec,
                                        consume_key_sec,
                                        consume_radix_sec,
                                        consume_stdsort_sec,
                                        consume_write_sec,
                                        sort_sec,
                                        uid_profile_build_sec,
                                        uid_profile_group_sec,
                                        uid_profile_insert_sec,
                                        uid_profile_sol_sec,
                                        uid_profile_scalar_sec,
                                        uid_profile_total_sec,
                                        bgj_cuda_host_wall_time() - consume_profile_t0);
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
    static thread_local bgj_cuda_result_storage_t result_storage;
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
    if (!result_storage.reserve((size_t)result_capacity)) return 0;
    bgj_cuda_result_t *results = result_storage.ptr;

    uint32_t result_count = 0;
    int overflow = 0;
    const int transform_dp = record_dp && bgj_cuda_cred_transform_requested();
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
                                                   bkt->center_ind,
                                                   bkt->center_norm,
                                                   record_dp ? 1 : 0,
                                                   transform_dp,
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
                                              bkt->center_ind,
                                              bkt->center_norm,
                                              record_dp ? 1 : 0,
                                              transform_dp,
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
int Pool_epi8_t<nb>::_search_bgj1_cuda_overlap(bucket_epi8_t<record_dp> *bkt,
                                               sol_list_epi8_t *sol,
                                               int32_t goal_norm,
                                               bgj_profile_data_t<nb> *prof)
{
    if (!record_dp) return _search_bgj1_cuda<record_dp, profiling>(bkt, sol, goal_norm, prof);
    if (bkt->num_pvec < 0 || bkt->num_nvec < 0) return 0;

    const uint32_t num_p = (uint32_t)bkt->num_pvec;
    const uint32_t num_n = (uint32_t)bkt->num_nvec;
    if (num_p + num_n == 0) {
        _search_cred<record_dp, profiling>(bkt, sol, goal_norm, prof);
        return 1;
    }

    const int can_submit = bgj_cuda_device_count() > 0 &&
                           bgj_cuda_pool_cache_requested() &&
                           !bgj_cuda_cred_transform_requested() &&
                           num_vec >= 0 &&
                           num_vec <= 0xffffffffL &&
                           bkt->num_pvec <= 0xffffffffL &&
                           bkt->num_nvec <= 0xffffffffL;
    if (!can_submit) {
        _search_cred<record_dp, profiling>(bkt, sol, goal_norm, prof);
        return 0;
    }

    const uint32_t result_capacity = bgj_cuda_max_results();
    static thread_local bgj_cuda_result_storage_t result_storage;

    if (!result_storage.reserve((size_t)result_capacity)) {
        _search_cred<record_dp, profiling>(bkt, sol, goal_norm, prof);
        return 0;
    }

    uint32_t submitted_result_count = 0;
    int submitted_overflow = 0;
    const int submitted = bgj_cuda_search_bucket_pool_raw_submit_async(vec,
                                                                       pool_epoch,
                                                                       (uint32_t)num_vec,
                                                                       bkt->pvec,
                                                                       bkt->nvec,
                                                                       bkt->pnorm,
                                                                       bkt->nnorm,
                                                                       num_p ? bkt->pdot : NULL,
                                                                       num_n ? bkt->ndot : NULL,
                                                                       num_p,
                                                                       num_n,
                                                                       (uint32_t)vec_length,
                                                                       goal_norm,
                                                                       bkt->center_ind,
                                                                       bkt->center_norm,
                                                                       1,
                                                                       0,
                                                                       0,
                                                                       1,
                                                                       1,
                                                                       result_capacity,
                                                                       &submitted_result_count,
                                                                       &submitted_overflow);

    const int dot_ready = submitted ? bgj_cuda_search_bucket_raw_wait_dot_copy_async() : 0;
    _search_cred<record_dp, profiling>(bkt, sol, goal_norm, prof);
    if (!submitted || !dot_ready) return 0;

    uint32_t result_count = 0;
    int overflow = 0;
    bgj_cuda_result_t *results = result_storage.ptr;
    const int finished = bgj_cuda_search_bucket_raw_finish_async(results,
                                                                 result_capacity,
                                                                 &submitted_result_count,
                                                                 &submitted_overflow,
                                                                 &result_count,
                                                                 &overflow);
    if (!finished || (overflow && bgj_cuda_overflow_fallback_requested())) return 0;

    return bgj_cuda_consume_bgj1_results<nb, record_dp, profiling>(this,
                                                                    bkt,
                                                                    sol,
                                                                    prof,
                                                                    results,
                                                                    result_count);
}

template <uint32_t nb, bool record_dp, bool profiling>
static int bgj_cuda_consume_bgj1_result_batch_coalesced(Pool_epi8_t<nb> *pool,
                                                        bucket_epi8_t<record_dp> **buckets,
                                                        long num_bucket,
                                                        sol_list_epi8_t *sol,
                                                        bgj_profile_data_t<nb> *prof,
                                                        bgj_cuda_result_t **result_ptrs,
                                                        const uint32_t *result_counts)
{
    if (num_bucket <= 1) return 0;
    if (!bgj_cuda_uid_batch_requested() || !bgj_cuda_uid_coalesce_requested()) return 0;
    if (bgj_cuda_host_profile_requested() || bgj_cuda_sort_results_requested()) return 0;

    uint64_t total_results = 0;
    for (long i = 0; i < num_bucket; i++) {
        total_results += result_counts[i];
    }
    if (total_results < bgj_cuda_uid_coalesce_min_results()) return 0;
    if (total_results > 0xffffffffULL) return 0;

    static thread_local std::vector<bgj_cuda_uid_candidate_t> candidates;
    static thread_local std::vector<uint32_t> order;
    static thread_local std::vector<uint32_t> touched_slots;
    static thread_local std::vector<uint32_t> slot_counts;
    static thread_local std::vector<uint32_t> slot_offsets;
    static thread_local std::vector<uint32_t> slot_cursor;

    try {
        if (slot_counts.size() != UidHashTable::NUM_UID_LOCK) {
            slot_counts.assign(UidHashTable::NUM_UID_LOCK, 0);
            slot_offsets.assign(UidHashTable::NUM_UID_LOCK, 0);
            slot_cursor.assign(UidHashTable::NUM_UID_LOCK, 0);
        }
        candidates.clear();
        touched_slots.clear();
        candidates.reserve((size_t)total_results);
        touched_slots.reserve(UidHashTable::NUM_UID_LOCK);
        order.resize((size_t)total_results);
    } catch (...) {
        return 0;
    }

    const int uid_batch_profile = bgj_cuda_uid_batch_profile_requested();
    const double uid_profile_total_t0 = uid_batch_profile ? bgj_cuda_host_wall_time() : 0.0;
    double uid_profile_build_sec = 0.0;
    double uid_profile_group_sec = 0.0;
    double uid_profile_insert_sec = 0.0;
    double uid_profile_sol_sec = 0.0;
    uint64_t try_add2 = 0;
    uint64_t try_add3 = 0;
    uint64_t succ_add2 = 0;
    uint64_t succ_add3 = 0;

    double uid_profile_phase_t0 = uid_batch_profile ? bgj_cuda_host_wall_time() : 0.0;
    for (long bucket_index = 0; bucket_index < num_bucket; bucket_index++) {
        bucket_epi8_t<record_dp> *bkt = buckets[bucket_index];
        bgj_cuda_result_t *results = result_ptrs[bucket_index];
        const uint32_t result_count = result_counts[bucket_index];
        const uint32_t center = bkt->center_ind;
        const uint64_t center_u = bkt->center_u;
        for (uint32_t i = 0; i < result_count; i++) {
            const uint32_t x = results[i].x;
            const uint32_t y = results[i].y;
            switch (results[i].type) {
            case BGJ_CUDA_SOL_A:
                try_add2++;
                bgj_cuda_add_uid_candidate(candidates,
                                           touched_slots,
                                           slot_counts,
                                           pool->vu[x] + pool->vu[y],
                                           BGJ_CUDA_SOL_A,
                                           center,
                                           x,
                                           y);
                break;
            case BGJ_CUDA_SOL_S:
                try_add2++;
                bgj_cuda_add_uid_candidate(candidates,
                                           touched_slots,
                                           slot_counts,
                                           pool->vu[x] - pool->vu[y],
                                           BGJ_CUDA_SOL_S,
                                           center,
                                           x,
                                           y);
                break;
            case BGJ_CUDA_SOL_AA:
                try_add3++;
                bgj_cuda_add_uid_candidate(candidates,
                                           touched_slots,
                                           slot_counts,
                                           center_u + pool->vu[x] + pool->vu[y],
                                           BGJ_CUDA_SOL_AA,
                                           center,
                                           x,
                                           y);
                break;
            case BGJ_CUDA_SOL_SA:
                try_add3++;
                bgj_cuda_add_uid_candidate(candidates,
                                           touched_slots,
                                           slot_counts,
                                           center_u - pool->vu[x] + pool->vu[y],
                                           BGJ_CUDA_SOL_SA,
                                           center,
                                           x,
                                           y);
                break;
            case BGJ_CUDA_SOL_SS:
                try_add3++;
                bgj_cuda_add_uid_candidate(candidates,
                                           touched_slots,
                                           slot_counts,
                                           center_u - pool->vu[x] - pool->vu[y],
                                           BGJ_CUDA_SOL_SS,
                                           center,
                                           x,
                                           y);
                break;
            default:
                break;
            }
        }
    }
    if (uid_batch_profile) {
        uid_profile_build_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
    }

    uid_profile_phase_t0 = uid_batch_profile ? bgj_cuda_host_wall_time() : 0.0;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < touched_slots.size(); i++) {
        const uint32_t slot = touched_slots[i];
        slot_offsets[slot] = offset;
        slot_cursor[slot] = offset;
        offset += slot_counts[slot];
    }
    for (uint32_t i = 0; i < candidates.size(); i++) {
        const uint32_t slot = candidates[i].slot;
        order[slot_cursor[slot]++] = i;
    }
    if (uid_batch_profile) {
        uid_profile_group_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
    }

    uid_profile_phase_t0 = uid_batch_profile ? bgj_cuda_host_wall_time() : 0.0;
    for (uint32_t i = 0; i < touched_slots.size(); i++) {
        const uint32_t slot = touched_slots[i];
        const uint32_t begin = slot_offsets[slot];
        const uint32_t end = begin + slot_counts[slot];
        pthread_spin_lock(&pool->uid->uid_lock[slot].a);
        for (uint32_t j = begin; j < end; j++) {
            bgj_cuda_uid_candidate_t &candidate = candidates[order[j]];
            candidate.accepted =
                pool->uid->uid_table[slot].a.insert(candidate.uid).second ? 1 : 0;
        }
        pthread_spin_unlock(&pool->uid->uid_lock[slot].a);
    }
    if (uid_batch_profile) {
        uid_profile_insert_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
    }

    uid_profile_phase_t0 = uid_batch_profile ? bgj_cuda_host_wall_time() : 0.0;
    uint64_t accepted = 0;
    for (uint32_t i = 0; i < candidates.size(); i++) {
        if (!candidates[i].accepted) continue;
        accepted++;
        if (candidates[i].type == BGJ_CUDA_SOL_A ||
            candidates[i].type == BGJ_CUDA_SOL_S) {
            succ_add2++;
        } else {
            succ_add3++;
        }
        bgj_cuda_add_accepted_solution(sol, candidates[i]);
    }
    if (uid_batch_profile) {
        uid_profile_sol_sec += bgj_cuda_host_wall_time() - uid_profile_phase_t0;
    }

    for (uint32_t i = 0; i < touched_slots.size(); i++) {
        const uint32_t slot = touched_slots[i];
        slot_counts[slot] = 0;
        slot_offsets[slot] = 0;
        slot_cursor[slot] = 0;
    }

    if (profiling && prof) {
        pthread_spin_lock(&prof->profile_lock);
        prof->try_add2 += try_add2;
        prof->try_add3 += try_add3;
        prof->succ_add2 += succ_add2;
        prof->succ_add3 += succ_add3;
        pthread_spin_unlock(&prof->profile_lock);
    }

    if (uid_batch_profile) {
        bgj_cuda_uid_batch_profile_record(1,
                                          (uint64_t)num_bucket,
                                          total_results,
                                          (uint64_t)candidates.size(),
                                          accepted,
                                          (uint64_t)touched_slots.size(),
                                          uid_profile_build_sec,
                                          uid_profile_group_sec,
                                          uid_profile_insert_sec,
                                          uid_profile_sol_sec,
                                          0.0,
                                          bgj_cuda_host_wall_time() - uid_profile_total_t0);
    }
    return 1;
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
    static thread_local std::vector<uint32_t> center_ids;
    static thread_local std::vector<int32_t> goal_norms;
    static thread_local std::vector<int32_t> center_norms;
    static thread_local std::vector<bgj_cuda_result_t *> result_ptrs;
    static thread_local std::vector<uint32_t> result_capacities;
    static thread_local std::vector<uint32_t> result_counts;
    static thread_local std::vector<int> overflows;
    static thread_local bgj_cuda_result_storage_t result_storage;

    try {
        p_ids.resize(batch_size);
        n_ids.resize(batch_size);
        p_norm.resize(batch_size);
        n_norm.resize(batch_size);
        p_dot.resize(batch_size);
        n_dot.resize(batch_size);
        num_p.resize(batch_size);
        num_n.resize(batch_size);
        center_ids.resize(batch_size);
        goal_norms.resize(batch_size);
        center_norms.resize(batch_size);
        result_ptrs.resize(batch_size);
        result_capacities.resize(batch_size);
        result_counts.resize(batch_size);
        overflows.resize(batch_size);
    } catch (...) {
        return 0;
    }

    if (batch_size != 0 &&
        (size_t)batch_size > std::numeric_limits<size_t>::max() / (size_t)result_capacity) {
        return 0;
    }
    const size_t result_slab_capacity = (size_t)batch_size * (size_t)result_capacity;
    if (!result_storage.reserve(result_slab_capacity)) return 0;

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
        center_ids[i] = bkt->center_ind;
        goal_norms[i] = goal_norm;
        center_norms[i] = bkt->center_norm;
        result_capacities[i] = result_capacity;
        result_ptrs[i] = result_storage.ptr + (size_t)i * (size_t)result_capacity;
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
                                                         center_ids.data(),
                                                         center_norms.data(),
                                                         record_dp ? 1 : 0,
                                                         record_dp && bgj_cuda_cred_transform_requested(),
                                                         result_ptrs.data(),
                                                         result_capacities.data(),
                                                         result_counts.data(),
                                                         overflows.data());
    if (!ok) return 0;

    for (uint32_t i = 0; i < batch_size; i++) {
        if (overflows[i] && bgj_cuda_overflow_fallback_requested()) return 0;
    }
    if (bgj_cuda_consume_bgj1_result_batch_coalesced<nb, record_dp, profiling>(this,
                                                                                buckets,
                                                                                num_bucket,
                                                                                sol,
                                                                                prof,
                                                                                result_ptrs.data(),
                                                                                result_counts.data())) {
        return 1;
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

static long bgj_cuda_host_threads(long requested, long sieving_dim);

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj)
{
    const char *min_csd_env = getenv("BGJ_CUDA_MIN_CSD");
    if (min_csd_env && min_csd_env[0]) {
        char *end = NULL;
        const long min_csd = strtol(min_csd_env, &end, 10);
        if (end != min_csd_env && min_csd > 0 && CSD < min_csd) {
            return bgj1_Sieve(log_level, lps_auto_adj);
        }
    }

    if (bgj_cuda_device_count() <= 0) {
        fprintf(stderr, "[Warning] CUDA requested but no CUDA device is available: %s. Falling back to CPU bgj1.\n",
                bgj_cuda_last_error());
        return bgj1_Sieve(log_level, lps_auto_adj);
    }

    set_num_threads(bgj_cuda_host_threads(num_threads, CSD));
    const int old = bgj_cuda_search_requested();
    bgj_cuda_set_search_requested(1);
    const int ret = bgj1_Sieve(log_level, lps_auto_adj);
    bgj_cuda_set_search_requested(old);
    return ret;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj2_Sieve_cuda(long log_level, long lps_auto_adj)
{
    if (bgj_cuda_device_count() <= 0) {
        fprintf(stderr, "[Warning] CUDA requested but no CUDA device is available: %s. Falling back to CPU bgj2.\n",
                bgj_cuda_last_error());
        return bgj2_Sieve(log_level, lps_auto_adj);
    }

    set_num_threads(bgj_cuda_host_threads(num_threads, CSD));
    const int old = bgj_cuda_search_requested();
    bgj_cuda_set_search_requested(1);
    const int ret = bgj2_Sieve(log_level, lps_auto_adj);
    bgj_cuda_set_search_requested(old);
    return ret;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::bgj3_Sieve_cuda(long log_level, long lps_auto_adj)
{
    if (bgj_cuda_device_count() <= 0) {
        fprintf(stderr, "[Warning] CUDA requested but no CUDA device is available: %s. Falling back to CPU bgj3.\n",
                bgj_cuda_last_error());
        return bgj3_Sieve(log_level, lps_auto_adj);
    }

    set_num_threads(bgj_cuda_host_threads(num_threads, CSD));
    const int old = bgj_cuda_search_requested();
    bgj_cuda_set_search_requested(1);
    const int ret = bgj3_Sieve(log_level, lps_auto_adj);
    bgj_cuda_set_search_requested(old);
    return ret;
}

static long bgj_cuda_host_threads(long requested, long sieving_dim)
{
    const int active_devices = bgj_cuda_execution_device_count();
    const char *env = getenv("BGJ_CUDA_HOST_THREADS");
    if (env && env[0]) {
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env && parsed > 0) {
            return parsed > MAX_NTHREADS ? MAX_NTHREADS : parsed;
        }
    }

    if (requested < 1) requested = 1;
    if (active_devices > requested) requested = active_devices;
    if (requested < 2 && sieving_dim >= 80) return 2;
    return requested > MAX_NTHREADS ? MAX_NTHREADS : requested;
}

static int bgj_cuda_lift_print_mode(long *log_level)
{
    if (abs(*log_level - 32768) < 10) {
        *log_level -= 32768;
        return 2;
    }
    if (abs(*log_level - 16384) < 10) {
        *log_level -= 16384;
        return 1;
    }
    return 0;
}

template <uint32_t nb>
int Pool_epi8_t<nb>::left_progressive_bgjfsieve_cuda(long ind_l, long ind_r, long num_threads, long log_level, long ssd)
{
    if (ind_r - ind_l < ssd) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgjfsieve_cuda: sieving dim too small, aborted.\n", nb);
        return 0;
    }
    if (ind_r - ind_l > nb * 32) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::left_progressive_bgjfsieve_cuda: sieving dim(%ld) > nb(%u) * 32, aborted.\n", nb, ind_r - ind_l, nb);
        return -1;
    }

    const int show_lift = bgj_cuda_lift_print_mode(&log_level);

    clear_pool();
    set_num_threads(bgj_cuda_host_threads(num_threads, ind_r - ind_l));
    set_max_pool_size((long)(pow(4. / 3., (ind_r - ind_l) * 0.5) * 3.2) + 1);
    set_sieving_context(ind_r - ssd, ind_r);
    long sample_size = (long)(pow(4. / 3., ssd * 0.5) * 6.0);
    if (sample_size > _pool_size) sample_size = _pool_size;
    sampling(sample_size);

    bgj1_Sieve_cuda(log_level, 1);
    while (index_l > ind_l) {
        extend_left();
        long target_num_vec = (long)(pow(4. / 3., CSD * 0.5) * 3.2);
        if (target_num_vec > num_vec + num_empty) num_empty = target_num_vec - num_vec;
        if (CSD >= 92) {
            bgj3_Sieve_cuda(log_level, 1);
        } else if (CSD > 80) {
            bgj2_Sieve_cuda(log_level, 1);
        } else {
            bgj1_Sieve_cuda(log_level, 1);
        }
        if (show_lift == 1) show_min_lift(0);
    }
    if (show_lift == 2) show_min_lift(0);
    return 1;
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

    const int show_lift = bgj_cuda_lift_print_mode(&log_level);

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
        if (show_lift == 1) show_min_lift(0);
    }
    if (show_lift == 2) show_min_lift(0);
    return 1;
}

#if COMPILE_POOL_EPI8_96
template int Pool_epi8_t<3>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_bgj1_cuda_overlap<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_bgj1_cuda<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_bgj1_cuda_overlap<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::_search_bgj1_cuda_batch<0, 1>(bucket_epi8_t<0> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<3> *prof);
template int Pool_epi8_t<3>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<3>::bgj2_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<3>::bgj3_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<3>::left_progressive_bgjfsieve_cuda(long ind_l, long ind_r, long num_threads, long log_level, long ssd);
template int Pool_epi8_t<3>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_128
template int Pool_epi8_t<4>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_bgj1_cuda_overlap<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_bgj1_cuda<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_bgj1_cuda_overlap<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::_search_bgj1_cuda_batch<0, 1>(bucket_epi8_t<0> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<4> *prof);
template int Pool_epi8_t<4>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<4>::bgj2_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<4>::bgj3_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<4>::left_progressive_bgjfsieve_cuda(long ind_l, long ind_r, long num_threads, long log_level, long ssd);
template int Pool_epi8_t<4>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_160
template int Pool_epi8_t<5>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_bgj1_cuda_overlap<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_bgj1_cuda<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_bgj1_cuda_overlap<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::_search_bgj1_cuda_batch<0, 1>(bucket_epi8_t<0> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<5> *prof);
template int Pool_epi8_t<5>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<5>::bgj2_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<5>::bgj3_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<5>::left_progressive_bgjfsieve_cuda(long ind_l, long ind_r, long num_threads, long log_level, long ssd);
template int Pool_epi8_t<5>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_192
template int Pool_epi8_t<6>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::_search_bgj1_cuda_overlap<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::_search_bgj1_cuda<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::_search_bgj1_cuda_overlap<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::_search_bgj1_cuda_batch<0, 1>(bucket_epi8_t<0> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<6> *prof);
template int Pool_epi8_t<6>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<6>::bgj2_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<6>::bgj3_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<6>::left_progressive_bgjfsieve_cuda(long ind_l, long ind_r, long num_threads, long log_level, long ssd);
template int Pool_epi8_t<6>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#if COMPILE_POOL_EPI8_224
template int Pool_epi8_t<7>::_search_bgj1_cuda<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::_search_bgj1_cuda_overlap<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::_search_bgj1_cuda_batch<BGJ1_EPI8_USE_3RED, 1>(bucket_epi8_t<BGJ1_EPI8_USE_3RED> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::_search_bgj1_cuda<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::_search_bgj1_cuda_overlap<0, 1>(bucket_epi8_t<0> *bkt, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::_search_bgj1_cuda_batch<0, 1>(bucket_epi8_t<0> **buckets, long num_bucket, sol_list_epi8_t *sol, int32_t goal_norm, bgj_profile_data_t<7> *prof);
template int Pool_epi8_t<7>::bgj1_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<7>::bgj2_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<7>::bgj3_Sieve_cuda(long log_level, long lps_auto_adj);
template int Pool_epi8_t<7>::left_progressive_bgjfsieve_cuda(long ind_l, long ind_r, long num_threads, long log_level, long ssd);
template int Pool_epi8_t<7>::left_progressive_bgj1sieve_cuda(long ind_l, long ind_r, long num_threads, long log_level);
#endif

#endif
