#include "../include/bgj_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <pthread.h>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

using namespace nvcuda;

static thread_local char bgj_cuda_error[512] = "no CUDA error";

#define BGJ_CUDA_MAX_EXEC_DEVICES 16
#define BGJ_CUDA_WARP_SIZE 32u
#define BGJ_CUDA_TENSOR_WARPS_PER_BLOCK 4u
#define BGJ_CUDA_TENSOR_THREADS_PER_BLOCK (BGJ_CUDA_WARP_SIZE * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK)
#define BGJ_CUDA_TENSOR_NP_WIDE_TILES 4u
#define BGJ_CUDA_TENSOR_NP_MULTI_P_TILES 2u
#define BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP 4u
#define BGJ_CUDA_TENSOR_NP_MULTI_N_TILES (BGJ_CUDA_TENSOR_WARPS_PER_BLOCK * BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP)
#define BGJ_CUDA_BUCKET_DET_THREADS 256u
#define BGJ_CUDA_SEARCH_DET_THREADS 256u
#define BGJ_CUDA_LSH_TILE 16u
#define BGJ_CUDA_LSH_THREADS (BGJ_CUDA_LSH_TILE * BGJ_CUDA_LSH_TILE)
#define BGJ_CUDA_LSH_LIFT_THREADS 128u
#define BGJ_CUDA_LSH_LIFT_MAX_FD 160u

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> bgj_cuda_i8_a_frag_t;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bgj_cuda_i8_b_frag_t;
static const uint32_t BGJ_CUDA_I8_A_FRAG_WORDS =
    (sizeof(bgj_cuda_i8_a_frag_t) + sizeof(uint64_t) - 1u) / sizeof(uint64_t);
static const uint32_t BGJ_CUDA_I8_B_FRAG_WORDS =
    (sizeof(bgj_cuda_i8_b_frag_t) + sizeof(uint64_t) - 1u) / sizeof(uint64_t);

union bgj_cuda_i8_a_frag_u {
    bgj_cuda_i8_a_frag_t frag;
    uint64_t word[BGJ_CUDA_I8_A_FRAG_WORDS];
};

union bgj_cuda_i8_b_frag_u {
    bgj_cuda_i8_b_frag_t frag;
    uint64_t word[BGJ_CUDA_I8_B_FRAG_WORDS];
};

enum bgj_cuda_search_phase_t {
    BGJ_CUDA_SEARCH_PHASE_H2D = 0,
    BGJ_CUDA_SEARCH_PHASE_PACK = 1,
    BGJ_CUDA_SEARCH_PHASE_CRED = 2,
    BGJ_CUDA_SEARCH_PHASE_DET_COUNT = 3,
    BGJ_CUDA_SEARCH_PHASE_DET_FILL = 4,
    BGJ_CUDA_SEARCH_PHASE_TENSOR_NP = 5,
    BGJ_CUDA_SEARCH_PHASE_TENSOR_PP = 6,
    BGJ_CUDA_SEARCH_PHASE_TENSOR_NN = 7,
    BGJ_CUDA_SEARCH_PHASE_SCALAR = 8,
    BGJ_CUDA_SEARCH_PHASE_COUNT_COPY = 9,
    BGJ_CUDA_SEARCH_PHASE_RESULT_COPY = 10,
    BGJ_CUDA_SEARCH_PHASE_COUNT = 11
};

struct bgj_cuda_search_phase_profile_t {
    double sec[BGJ_CUDA_SEARCH_PHASE_COUNT];

    bgj_cuda_search_phase_profile_t()
        : sec()
    {
    }
};

struct bgj_cuda_raw_scratch_t {
    int8_t *p_vecs;
    int8_t *n_vecs;
    uint64_t *p_tensor_a_frags;
    uint64_t *p_tensor_b_frags;
    uint64_t *n_tensor_a_frags;
    uint64_t *n_tensor_b_frags;
    uint32_t *p_ids;
    uint32_t *n_ids;
    int32_t *p_norm;
    int32_t *n_norm;
    int32_t *p_dot;
    int32_t *n_dot;
    bgj_cuda_result_t *results;
    uint32_t *result_count;
    uint32_t *det_counts;
    uint32_t *det_offsets;
    int *overflow;
    uint32_t *host_result_count;
    int *host_overflow;
    bgj_cuda_result_t *host_results;
    bgj_cuda_result_t *pending_result_dst;
    uint32_t pending_result_count;
    cudaStream_t stream;
    cudaEvent_t dot_copy_done;
    cudaEvent_t profile_start[BGJ_CUDA_SEARCH_PHASE_COUNT];
    cudaEvent_t profile_stop[BGJ_CUDA_SEARCH_PHASE_COUNT];
    unsigned char profile_ran[BGJ_CUDA_SEARCH_PHASE_COUNT];
    bgj_cuda_search_phase_profile_t last_phase_profile;
    int device;

    size_t p_vec_capacity;
    size_t n_vec_capacity;
    size_t p_tensor_a_frag_capacity;
    size_t p_tensor_b_frag_capacity;
    size_t n_tensor_a_frag_capacity;
    size_t n_tensor_b_frag_capacity;
    size_t p_id_capacity;
    size_t n_id_capacity;
    size_t p_i32_capacity;
    size_t n_i32_capacity;
    size_t p_dot_capacity;
    size_t n_dot_capacity;
    size_t result_capacity;
    size_t result_count_capacity;
    size_t det_count_capacity;
    size_t det_offset_capacity;
    size_t overflow_capacity;
    size_t host_result_count_capacity;
    size_t host_overflow_capacity;
    size_t host_result_capacity;
    int stream_ready;
    int dot_event_ready;
    int profile_events_ready;
    int phase_profile_active;

    bgj_cuda_raw_scratch_t()
        : p_vecs(NULL),
          n_vecs(NULL),
          p_tensor_a_frags(NULL),
          p_tensor_b_frags(NULL),
          n_tensor_a_frags(NULL),
          n_tensor_b_frags(NULL),
          p_ids(NULL),
          n_ids(NULL),
          p_norm(NULL),
          n_norm(NULL),
          p_dot(NULL),
          n_dot(NULL),
          results(NULL),
          result_count(NULL),
          det_counts(NULL),
          det_offsets(NULL),
          overflow(NULL),
          host_result_count(NULL),
          host_overflow(NULL),
          host_results(NULL),
          pending_result_dst(NULL),
          pending_result_count(0),
          stream(NULL),
          dot_copy_done(NULL),
          profile_start(),
          profile_stop(),
          profile_ran(),
          last_phase_profile(),
          device(-1),
          p_vec_capacity(0),
          n_vec_capacity(0),
          p_tensor_a_frag_capacity(0),
          p_tensor_b_frag_capacity(0),
          n_tensor_a_frag_capacity(0),
          n_tensor_b_frag_capacity(0),
          p_id_capacity(0),
          n_id_capacity(0),
          p_i32_capacity(0),
          n_i32_capacity(0),
          p_dot_capacity(0),
          n_dot_capacity(0),
          result_capacity(0),
          result_count_capacity(0),
          det_count_capacity(0),
          det_offset_capacity(0),
          overflow_capacity(0),
          host_result_count_capacity(0),
          host_overflow_capacity(0),
          host_result_capacity(0),
          stream_ready(0),
          dot_event_ready(0),
          profile_events_ready(0),
          phase_profile_active(0)
    {
    }

    void release()
    {
        if (device >= 0) cudaSetDevice(device);
        if (stream_ready) cudaStreamSynchronize(stream);
        cudaFree(p_vecs);
        cudaFree(n_vecs);
        cudaFree(p_tensor_a_frags);
        cudaFree(p_tensor_b_frags);
        cudaFree(n_tensor_a_frags);
        cudaFree(n_tensor_b_frags);
        cudaFree(p_ids);
        cudaFree(n_ids);
        cudaFree(p_norm);
        cudaFree(n_norm);
        cudaFree(p_dot);
        cudaFree(n_dot);
        cudaFree(results);
        cudaFree(result_count);
        cudaFree(det_counts);
        cudaFree(det_offsets);
        cudaFree(overflow);
        if (host_result_count) cudaFreeHost(host_result_count);
        if (host_overflow) cudaFreeHost(host_overflow);
        if (host_results) cudaFreeHost(host_results);
        if (dot_event_ready) cudaEventDestroy(dot_copy_done);
        if (profile_events_ready) {
            for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
                cudaEventDestroy(profile_start[i]);
                cudaEventDestroy(profile_stop[i]);
            }
        }
        if (stream_ready) cudaStreamDestroy(stream);
        p_vecs = NULL;
        n_vecs = NULL;
        p_tensor_a_frags = NULL;
        p_tensor_b_frags = NULL;
        n_tensor_a_frags = NULL;
        n_tensor_b_frags = NULL;
        p_ids = NULL;
        n_ids = NULL;
        p_norm = NULL;
        n_norm = NULL;
        p_dot = NULL;
        n_dot = NULL;
        results = NULL;
        result_count = NULL;
        det_counts = NULL;
        det_offsets = NULL;
        overflow = NULL;
        host_result_count = NULL;
        host_overflow = NULL;
        host_results = NULL;
        pending_result_dst = NULL;
        pending_result_count = 0;
        stream = NULL;
        dot_copy_done = NULL;
        for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
            profile_start[i] = NULL;
            profile_stop[i] = NULL;
            profile_ran[i] = 0;
            last_phase_profile.sec[i] = 0.0;
        }
        p_vec_capacity = 0;
        n_vec_capacity = 0;
        p_tensor_a_frag_capacity = 0;
        p_tensor_b_frag_capacity = 0;
        n_tensor_a_frag_capacity = 0;
        n_tensor_b_frag_capacity = 0;
        p_id_capacity = 0;
        n_id_capacity = 0;
        p_i32_capacity = 0;
        n_i32_capacity = 0;
        p_dot_capacity = 0;
        n_dot_capacity = 0;
        result_capacity = 0;
        result_count_capacity = 0;
        det_count_capacity = 0;
        det_offset_capacity = 0;
        overflow_capacity = 0;
        host_result_count_capacity = 0;
        host_overflow_capacity = 0;
        host_result_capacity = 0;
        stream_ready = 0;
        dot_event_ready = 0;
        profile_events_ready = 0;
        phase_profile_active = 0;
        device = -1;
    }

    ~bgj_cuda_raw_scratch_t()
    {
        release();
    }
};

static thread_local bgj_cuda_raw_scratch_t bgj_cuda_raw_scratch;

struct bgj_cuda_raw_batch_scratch_t {
    bgj_cuda_raw_scratch_t **items;
    size_t size;
    size_t capacity;

    bgj_cuda_raw_batch_scratch_t()
        : items(NULL),
          size(0),
          capacity(0)
    {
    }

    ~bgj_cuda_raw_batch_scratch_t()
    {
        for (size_t i = 0; i < size; i++) {
            if (items[i]) {
                items[i]->~bgj_cuda_raw_scratch_t();
                free(items[i]);
            }
        }
        free(items);
    }

    bgj_cuda_raw_scratch_t *get(uint32_t index)
    {
        if ((size_t)index >= capacity) {
            size_t new_capacity = capacity ? capacity : 8;
            while (new_capacity <= (size_t)index) new_capacity *= 2;
            bgj_cuda_raw_scratch_t **new_items =
                (bgj_cuda_raw_scratch_t **)realloc(items, new_capacity * sizeof(bgj_cuda_raw_scratch_t *));
            if (!new_items) return NULL;
            items = new_items;
            for (size_t i = capacity; i < new_capacity; i++) items[i] = NULL;
            capacity = new_capacity;
        }
        while (size <= (size_t)index) {
            void *memory = malloc(sizeof(bgj_cuda_raw_scratch_t));
            if (!memory) return NULL;
            bgj_cuda_raw_scratch_t *scratch = new (memory) bgj_cuda_raw_scratch_t;
            items[size] = scratch;
            size++;
            if (!scratch) {
                return NULL;
            }
        }
        return items[index];
    }
};

static thread_local bgj_cuda_raw_batch_scratch_t bgj_cuda_raw_batch_scratch;

struct bgj_cuda_lsh_lift_scratch_t {
    float *fvec;
    uint32_t *candidates;
    float *b_full;
    float *idiag;
    float *min_norm;
    bgj_cuda_lsh_lift_result_t *results;
    uint32_t *result_count;
    int *overflow;
    uint32_t *host_result_count;
    int *host_overflow;
    cudaStream_t stream;
    int device;
    size_t fvec_capacity;
    size_t candidate_capacity;
    size_t b_full_capacity;
    size_t idiag_capacity;
    size_t min_norm_capacity;
    size_t result_capacity;
    size_t result_count_capacity;
    size_t overflow_capacity;
    size_t host_result_count_capacity;
    size_t host_overflow_capacity;
    int stream_ready;

    bgj_cuda_lsh_lift_scratch_t()
        : fvec(NULL),
          candidates(NULL),
          b_full(NULL),
          idiag(NULL),
          min_norm(NULL),
          results(NULL),
          result_count(NULL),
          overflow(NULL),
          host_result_count(NULL),
          host_overflow(NULL),
          stream(NULL),
          device(-1),
          fvec_capacity(0),
          candidate_capacity(0),
          b_full_capacity(0),
          idiag_capacity(0),
          min_norm_capacity(0),
          result_capacity(0),
          result_count_capacity(0),
          overflow_capacity(0),
          host_result_count_capacity(0),
          host_overflow_capacity(0),
          stream_ready(0)
    {
    }

    void release()
    {
        if (device >= 0) cudaSetDevice(device);
        if (stream_ready) cudaStreamSynchronize(stream);
        cudaFree(fvec);
        cudaFree(candidates);
        cudaFree(b_full);
        cudaFree(idiag);
        cudaFree(min_norm);
        cudaFree(results);
        cudaFree(result_count);
        cudaFree(overflow);
        if (host_result_count) cudaFreeHost(host_result_count);
        if (host_overflow) cudaFreeHost(host_overflow);
        if (stream_ready) cudaStreamDestroy(stream);
        fvec = NULL;
        candidates = NULL;
        b_full = NULL;
        idiag = NULL;
        min_norm = NULL;
        results = NULL;
        result_count = NULL;
        overflow = NULL;
        host_result_count = NULL;
        host_overflow = NULL;
        stream = NULL;
        device = -1;
        fvec_capacity = 0;
        candidate_capacity = 0;
        b_full_capacity = 0;
        idiag_capacity = 0;
        min_norm_capacity = 0;
        result_capacity = 0;
        result_count_capacity = 0;
        overflow_capacity = 0;
        host_result_count_capacity = 0;
        host_overflow_capacity = 0;
        stream_ready = 0;
    }

    ~bgj_cuda_lsh_lift_scratch_t()
    {
        release();
    }
};

static thread_local bgj_cuda_lsh_lift_scratch_t bgj_cuda_lsh_lift_scratch;

struct bgj_cuda_raw_async_state_t {
    int active;
    int split_active;
    int split_tensor;
    int split_device_count;
    int split_devices[BGJ_CUDA_MAX_EXEC_DEVICES];
    uint64_t split_work_begin[BGJ_CUDA_MAX_EXEC_DEVICES];
    uint64_t split_work_count[BGJ_CUDA_MAX_EXEC_DEVICES];
    uint32_t split_submitted_result_count[BGJ_CUDA_MAX_EXEC_DEVICES];
    uint32_t split_copied_result_count[BGJ_CUDA_MAX_EXEC_DEVICES];
    int split_submitted_overflow[BGJ_CUDA_MAX_EXEC_DEVICES];
    int split_copied_overflow[BGJ_CUDA_MAX_EXEC_DEVICES];
    uint64_t estimated_dots;
    uint64_t full_work;
    uint32_t num_p;
    uint32_t num_n;
    const int8_t *pool_vecs_host;
    uint64_t pool_epoch;
    uint32_t pool_size;
    const uint32_t *p_ids;
    const uint32_t *n_ids;
    const int32_t *p_norm;
    const int32_t *n_norm;
    const int32_t *p_dot;
    const int32_t *n_dot;
    int32_t *p_dot_snapshot;
    int32_t *n_dot_snapshot;
    size_t p_dot_snapshot_capacity;
    size_t n_dot_snapshot_capacity;
    uint32_t vec_length;
    int32_t goal_norm;
    uint32_t center_id;
    int32_t center_norm;
    int record_dp;
    int raw_center_dp;
    double total_t0;
    double submit_sec;

    bgj_cuda_raw_async_state_t()
        : active(0),
          split_active(0),
          split_tensor(0),
          split_device_count(0),
          split_devices(),
          split_work_begin(),
          split_work_count(),
          split_submitted_result_count(),
          split_copied_result_count(),
          split_submitted_overflow(),
          split_copied_overflow(),
          estimated_dots(0),
          full_work(0),
          num_p(0),
          num_n(0),
          pool_vecs_host(NULL),
          pool_epoch(0),
          pool_size(0),
          p_ids(NULL),
          n_ids(NULL),
          p_norm(NULL),
          n_norm(NULL),
          p_dot(NULL),
          n_dot(NULL),
          p_dot_snapshot(NULL),
          n_dot_snapshot(NULL),
          p_dot_snapshot_capacity(0),
          n_dot_snapshot_capacity(0),
          vec_length(0),
          goal_norm(0),
          center_id(0),
          center_norm(0),
          record_dp(0),
          raw_center_dp(0),
          total_t0(0.0),
          submit_sec(0.0)
    {
    }

    void reset()
    {
        free(p_dot_snapshot);
        free(n_dot_snapshot);
        active = 0;
        split_active = 0;
        split_tensor = 0;
        split_device_count = 0;
        estimated_dots = 0;
        full_work = 0;
        num_p = 0;
        num_n = 0;
        pool_vecs_host = NULL;
        pool_epoch = 0;
        pool_size = 0;
        p_ids = NULL;
        n_ids = NULL;
        p_norm = NULL;
        n_norm = NULL;
        p_dot = NULL;
        n_dot = NULL;
        p_dot_snapshot = NULL;
        n_dot_snapshot = NULL;
        p_dot_snapshot_capacity = 0;
        n_dot_snapshot_capacity = 0;
        vec_length = 0;
        goal_norm = 0;
        center_id = 0;
        center_norm = 0;
        record_dp = 0;
        raw_center_dp = 0;
        total_t0 = 0.0;
        submit_sec = 0.0;
        for (int i = 0; i < BGJ_CUDA_MAX_EXEC_DEVICES; i++) {
            split_devices[i] = 0;
            split_work_begin[i] = 0;
            split_work_count[i] = 0;
            split_submitted_result_count[i] = 0;
            split_copied_result_count[i] = 0;
            split_submitted_overflow[i] = 0;
            split_copied_overflow[i] = 0;
        }
    }
};

static thread_local bgj_cuda_raw_async_state_t bgj_cuda_raw_async_state;

struct bgj_cuda_bucket_scratch_t {
    uint32_t *center_ids;
    int8_t *center_vecs;
    int32_t *vnorm;
    bgj_cuda_bucket_entry_t *entries;
    uint32_t *entry_count;
    uint32_t *bucket_counts;
    uint32_t *bucket_offsets;
    int *overflow;
    cudaStream_t stream;
    int device;

    size_t center_id_capacity;
    size_t center_vec_capacity;
    size_t vnorm_capacity;
    size_t entry_capacity;
    size_t entry_count_capacity;
    size_t bucket_count_capacity;
    size_t bucket_offset_capacity;
    size_t overflow_capacity;
    int stream_ready;

    bgj_cuda_bucket_scratch_t()
        : center_ids(NULL),
          center_vecs(NULL),
          vnorm(NULL),
          entries(NULL),
          entry_count(NULL),
          bucket_counts(NULL),
          bucket_offsets(NULL),
          overflow(NULL),
          stream(NULL),
          device(-1),
          center_id_capacity(0),
          center_vec_capacity(0),
          vnorm_capacity(0),
          entry_capacity(0),
          entry_count_capacity(0),
          bucket_count_capacity(0),
          bucket_offset_capacity(0),
          overflow_capacity(0),
          stream_ready(0)
    {
    }

    void release()
    {
        if (device >= 0) cudaSetDevice(device);
        if (stream_ready) cudaStreamSynchronize(stream);
        cudaFree(center_ids);
        cudaFree(center_vecs);
        cudaFree(vnorm);
        cudaFree(entries);
        cudaFree(entry_count);
        cudaFree(bucket_counts);
        cudaFree(bucket_offsets);
        cudaFree(overflow);
        if (stream_ready) cudaStreamDestroy(stream);
        center_ids = NULL;
        center_vecs = NULL;
        vnorm = NULL;
        entries = NULL;
        entry_count = NULL;
        bucket_counts = NULL;
        bucket_offsets = NULL;
        overflow = NULL;
        stream = NULL;
        center_id_capacity = 0;
        center_vec_capacity = 0;
        vnorm_capacity = 0;
        entry_capacity = 0;
        entry_count_capacity = 0;
        bucket_count_capacity = 0;
        bucket_offset_capacity = 0;
        overflow_capacity = 0;
        stream_ready = 0;
        device = -1;
    }

    ~bgj_cuda_bucket_scratch_t()
    {
        release();
    }
};

static thread_local bgj_cuda_bucket_scratch_t bgj_cuda_bucket_scratch;

static void set_cuda_error(const char *context, cudaError_t err)
{
    snprintf(bgj_cuda_error, sizeof(bgj_cuda_error), "%s: %s", context, cudaGetErrorString(err));
}

static const char *cublas_status_name(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

static void set_cublas_error(const char *context, cublasStatus_t status)
{
    snprintf(bgj_cuda_error, sizeof(bgj_cuda_error), "%s: %s", context, cublas_status_name(status));
}

static void set_plain_error(const char *message)
{
    snprintf(bgj_cuda_error, sizeof(bgj_cuda_error), "%s", message);
}

extern "C" const char *bgj_cuda_raw_last_error()
{
    return bgj_cuda_error;
}

extern "C" int bgj_cuda_raw_device_count()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        set_cuda_error("cudaGetDeviceCount", err);
        return 0;
    }
    return count;
}

struct bgj_cuda_execution_device_state_t {
    pthread_mutex_t lock;
    int initialized;
    int count;
    int devices[BGJ_CUDA_MAX_EXEC_DEVICES];
    unsigned next_slot;

    bgj_cuda_execution_device_state_t()
        : initialized(0),
          count(0),
          devices(),
          next_slot(0)
    {
        pthread_mutex_init(&lock, NULL);
    }

    ~bgj_cuda_execution_device_state_t()
    {
        pthread_mutex_destroy(&lock);
    }
};

static bgj_cuda_execution_device_state_t bgj_cuda_execution_devices;
static thread_local int bgj_cuda_thread_device = -1;

static int bgj_cuda_env_flag(const char *name, int default_value)
{
    const char *env = getenv(name);
    if (env && env[0]) return env[0] != '0';
    return default_value;
}

static uint64_t bgj_cuda_env_u64(const char *name, uint64_t default_value)
{
    const char *env = getenv(name);
    if (!env || !env[0]) return default_value;
    char *end = NULL;
    const unsigned long long value = strtoull(env, &end, 10);
    if (end == env || value == 0) return default_value;
    return (uint64_t)value;
}

static int bgj_cuda_env_has_multiple_devices(const char *env)
{
    if (!env || !env[0]) return 0;
    int count = 0;
    const char *p = env;
    while (*p) {
        char *end = NULL;
        (void)strtol(p, &end, 10);
        if (end != p) {
            ++count;
            if (count > 1) return 1;
            p = end;
        } else {
            ++p;
        }
        while (*p == ',' || *p == ';' || *p == ':' ||
               *p == ' ' || *p == '\t') {
            ++p;
        }
    }
    return 0;
}

static int bgj_cuda_explicit_multi_gpu_requested()
{
    if (bgj_cuda_env_has_multiple_devices(getenv("BGJ_CUDA_DEVICES"))) {
        return 1;
    }
    const char *num_env = getenv("BGJ_CUDA_NUM_DEVICES");
    if (num_env && num_env[0]) {
        char *end = NULL;
        const long value = strtol(num_env, &end, 10);
        if (end != num_env && value > 1) return 1;
    }
    return 0;
}

static int bgj_cuda_stable_multi_gpu_requested()
{
    const char *env = getenv("BGJ_CUDA_STABLE_MULTI_GPU");
    if (env && env[0]) return env[0] != '0';
    return bgj_cuda_explicit_multi_gpu_requested();
}

static int bgj_cuda_primary_thread_device_requested(int device_count)
{
    const char *env = getenv("BGJ_CUDA_PRIMARY_THREAD_DEVICE");
    if (env && env[0]) return env[0] != '0';
    return device_count > 1 && bgj_cuda_stable_multi_gpu_requested();
}

struct bgj_cuda_device_profile_entry_t {
    uint64_t calls;
    uint64_t buckets;
    uint64_t estimated_dots;
    uint64_t results;
    uint64_t overflows;
    uint64_t failures;
    double submit_sec;
    double wait_sec;
    double copy_sec;
    double total_sec;
};

struct bgj_cuda_device_profile_state_t {
    pthread_mutex_t lock;
    int registered;
    bgj_cuda_device_profile_entry_t devices[BGJ_CUDA_MAX_EXEC_DEVICES];

    bgj_cuda_device_profile_state_t()
        : registered(0), devices()
    {
        pthread_mutex_init(&lock, NULL);
    }

    ~bgj_cuda_device_profile_state_t()
    {
        pthread_mutex_destroy(&lock);
    }
};

static bgj_cuda_device_profile_state_t bgj_cuda_device_profile;

static int bgj_cuda_device_profile_requested()
{
    const char *env = getenv("BGJ_CUDA_DEVICE_PROFILE");
    return env && env[0] && env[0] != '0';
}

static double bgj_cuda_wall_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static const char *bgj_cuda_search_phase_name(int phase)
{
    switch (phase) {
    case BGJ_CUDA_SEARCH_PHASE_H2D:
        return "h2d";
    case BGJ_CUDA_SEARCH_PHASE_PACK:
        return "pack";
    case BGJ_CUDA_SEARCH_PHASE_CRED:
        return "cred";
    case BGJ_CUDA_SEARCH_PHASE_DET_COUNT:
        return "det_count";
    case BGJ_CUDA_SEARCH_PHASE_DET_FILL:
        return "det_fill";
    case BGJ_CUDA_SEARCH_PHASE_TENSOR_NP:
        return "tensor_np";
    case BGJ_CUDA_SEARCH_PHASE_TENSOR_PP:
        return "tensor_pp";
    case BGJ_CUDA_SEARCH_PHASE_TENSOR_NN:
        return "tensor_nn";
    case BGJ_CUDA_SEARCH_PHASE_SCALAR:
        return "scalar";
    case BGJ_CUDA_SEARCH_PHASE_COUNT_COPY:
        return "count_copy";
    case BGJ_CUDA_SEARCH_PHASE_RESULT_COPY:
        return "result_copy";
    default:
        return "unknown";
    }
}

struct bgj_cuda_phase_profile_state_t {
    pthread_mutex_t lock;
    int registered;
    uint64_t calls;
    uint64_t buckets;
    uint64_t estimated_dots;
    uint64_t results;
    uint64_t overflows;
    uint64_t failures;
    double sec[BGJ_CUDA_SEARCH_PHASE_COUNT];

    bgj_cuda_phase_profile_state_t()
        : registered(0),
          calls(0),
          buckets(0),
          estimated_dots(0),
          results(0),
          overflows(0),
          failures(0),
          sec()
    {
        pthread_mutex_init(&lock, NULL);
    }

    ~bgj_cuda_phase_profile_state_t()
    {
        pthread_mutex_destroy(&lock);
    }
};

static bgj_cuda_phase_profile_state_t bgj_cuda_phase_profile;

static int bgj_cuda_phase_profile_requested()
{
    const char *env = getenv("BGJ_CUDA_PHASE_PROFILE");
    return env && env[0] && env[0] != '0';
}

static int bgj_cuda_split_profile_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_SPLIT_PROFILE", 0);
}

static uint64_t bgj_cuda_split_profile_min_dots()
{
    return bgj_cuda_env_u64("BGJ_CUDA_SPLIT_PROFILE_MIN_DOTS", 0);
}

static void bgj_cuda_phase_profile_dump()
{
    if (!bgj_cuda_phase_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_phase_profile.lock);
    fprintf(stderr,
            "cuda_phase_profile: calls=%lu buckets=%lu dots=%lu results=%lu "
            "overflows=%lu failures=%lu",
            (unsigned long)bgj_cuda_phase_profile.calls,
            (unsigned long)bgj_cuda_phase_profile.buckets,
            (unsigned long)bgj_cuda_phase_profile.estimated_dots,
            (unsigned long)bgj_cuda_phase_profile.results,
            (unsigned long)bgj_cuda_phase_profile.overflows,
            (unsigned long)bgj_cuda_phase_profile.failures);
    for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
        fprintf(stderr, " %s=%.6fs",
                bgj_cuda_search_phase_name(i),
                bgj_cuda_phase_profile.sec[i]);
    }
    fprintf(stderr, "\n");
    pthread_mutex_unlock(&bgj_cuda_phase_profile.lock);
}

static void bgj_cuda_phase_profile_register()
{
    if (!bgj_cuda_phase_profile_requested()) return;
    pthread_mutex_lock(&bgj_cuda_phase_profile.lock);
    if (!bgj_cuda_phase_profile.registered) {
        bgj_cuda_phase_profile.registered = 1;
        atexit(bgj_cuda_phase_profile_dump);
    }
    pthread_mutex_unlock(&bgj_cuda_phase_profile.lock);
}

static void bgj_cuda_phase_profile_record(uint64_t buckets,
                                          uint64_t estimated_dots,
                                          uint64_t results,
                                          uint64_t overflows,
                                          uint64_t failures,
                                          const bgj_cuda_search_phase_profile_t *profile)
{
    if (!bgj_cuda_phase_profile_requested() || !profile) return;
    bgj_cuda_phase_profile_register();

    pthread_mutex_lock(&bgj_cuda_phase_profile.lock);
    bgj_cuda_phase_profile.calls++;
    bgj_cuda_phase_profile.buckets += buckets;
    bgj_cuda_phase_profile.estimated_dots += estimated_dots;
    bgj_cuda_phase_profile.results += results;
    bgj_cuda_phase_profile.overflows += overflows;
    bgj_cuda_phase_profile.failures += failures;
    for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
        bgj_cuda_phase_profile.sec[i] += profile->sec[i];
    }
    pthread_mutex_unlock(&bgj_cuda_phase_profile.lock);
}

static uint64_t bgj_cuda_estimated_pair_dots(uint32_t num_p, uint32_t num_n)
{
    return (uint64_t)num_p * (uint64_t)num_n +
           ((uint64_t)num_p * (uint64_t)(num_p > 1 ? num_p - 1u : 0u)) / 2u +
           ((uint64_t)num_n * (uint64_t)(num_n > 1 ? num_n - 1u : 0u)) / 2u;
}

static void bgj_cuda_device_profile_dump()
{
    if (!bgj_cuda_device_profile_requested()) return;

    pthread_mutex_lock(&bgj_cuda_device_profile.lock);
    for (int device = 0; device < BGJ_CUDA_MAX_EXEC_DEVICES; device++) {
        const bgj_cuda_device_profile_entry_t *entry =
            &bgj_cuda_device_profile.devices[device];
        if (!entry->calls && !entry->failures) continue;
        fprintf(stderr,
                "cuda_device_profile: device=%d calls=%lu buckets=%lu dots=%lu "
                "results=%lu overflows=%lu failures=%lu submit=%.6fs "
                "wait=%.6fs copy=%.6fs total=%.6fs\n",
                device,
                (unsigned long)entry->calls,
                (unsigned long)entry->buckets,
                (unsigned long)entry->estimated_dots,
                (unsigned long)entry->results,
                (unsigned long)entry->overflows,
                (unsigned long)entry->failures,
                entry->submit_sec,
                entry->wait_sec,
                entry->copy_sec,
                entry->total_sec);
    }
    pthread_mutex_unlock(&bgj_cuda_device_profile.lock);
}

static void bgj_cuda_device_profile_register()
{
    if (!bgj_cuda_device_profile_requested()) return;
    pthread_mutex_lock(&bgj_cuda_device_profile.lock);
    if (!bgj_cuda_device_profile.registered) {
        bgj_cuda_device_profile.registered = 1;
        atexit(bgj_cuda_device_profile_dump);
    }
    pthread_mutex_unlock(&bgj_cuda_device_profile.lock);
}

static void bgj_cuda_device_profile_record(int device,
                                           uint64_t buckets,
                                           uint64_t estimated_dots,
                                           uint64_t results,
                                           uint64_t overflows,
                                           uint64_t failures,
                                           double submit_sec,
                                           double wait_sec,
                                           double copy_sec,
                                           double total_sec)
{
    if (!bgj_cuda_device_profile_requested()) return;
    if (device < 0 || device >= BGJ_CUDA_MAX_EXEC_DEVICES) return;
    bgj_cuda_device_profile_register();

    pthread_mutex_lock(&bgj_cuda_device_profile.lock);
    bgj_cuda_device_profile_entry_t *entry = &bgj_cuda_device_profile.devices[device];
    entry->calls++;
    entry->buckets += buckets;
    entry->estimated_dots += estimated_dots;
    entry->results += results;
    entry->overflows += overflows;
    entry->failures += failures;
    entry->submit_sec += submit_sec;
    entry->wait_sec += wait_sec;
    entry->copy_sec += copy_sec;
    entry->total_sec += total_sec;
    pthread_mutex_unlock(&bgj_cuda_device_profile.lock);
}

enum bgj_cuda_bucket_profile_kind_t {
    BGJ_CUDA_BUCKET_PROFILE_SINGLE = 0,
    BGJ_CUDA_BUCKET_PROFILE_ASYNC = 1,
    BGJ_CUDA_BUCKET_PROFILE_BATCH = 2,
    BGJ_CUDA_BUCKET_PROFILE_BATCH_ITEM = 3
};

struct bgj_cuda_bucket_profile_entry_t {
    uint64_t seq;
    uint64_t score;
    uint64_t estimated_dots;
    uint64_t results;
    uint64_t overflows;
    uint64_t failures;
    uint32_t num_p;
    uint32_t num_n;
    uint32_t batch_size;
    uint32_t batch_index;
    int device;
    int kind;
    double submit_sec;
    double wait_sec;
    double copy_sec;
    double total_sec;
    double phase_sec[BGJ_CUDA_SEARCH_PHASE_COUNT];
};

struct bgj_cuda_bucket_profile_state_t {
    pthread_mutex_t lock;
    int enabled;
    int limit;
    uint64_t dump_every;
    uint64_t next_dump_seq;
    uint64_t next_seq;
    int slow_count;
    int work_count;
    bgj_cuda_bucket_profile_entry_t slow[256];
    bgj_cuda_bucket_profile_entry_t work[256];

    bgj_cuda_bucket_profile_state_t()
        : enabled(0),
          limit(64),
          dump_every(0),
          next_dump_seq(0),
          next_seq(0),
          slow_count(0),
          work_count(0),
          slow(),
          work()
    {
        pthread_mutex_init(&lock, NULL);
    }

    ~bgj_cuda_bucket_profile_state_t()
    {
        pthread_mutex_destroy(&lock);
    }
};

static bgj_cuda_bucket_profile_state_t bgj_cuda_bucket_profile;
static pthread_once_t bgj_cuda_bucket_profile_once = PTHREAD_ONCE_INIT;

static void bgj_cuda_bucket_profile_dump();

static long bgj_cuda_bucket_profile_parse_long(const char *name, long default_value)
{
    const char *env = getenv(name);
    if (!env || !env[0]) return default_value;
    char *end = NULL;
    long parsed = strtol(env, &end, 10);
    if (end == env) return default_value;
    return parsed;
}

static void bgj_cuda_bucket_profile_init_once()
{
    const char *env = getenv("BGJ_CUDA_BUCKET_PROFILE");
    bgj_cuda_bucket_profile.enabled = env && env[0] && env[0] != '0';
    if (!bgj_cuda_bucket_profile.enabled) return;

    long limit = bgj_cuda_bucket_profile_parse_long("BGJ_CUDA_BUCKET_PROFILE_TOP", 64);
    if (limit < 1) limit = 1;
    if (limit > 256) limit = 256;
    bgj_cuda_bucket_profile.limit = (int)limit;

    long dump_every =
        bgj_cuda_bucket_profile_parse_long("BGJ_CUDA_BUCKET_PROFILE_DUMP_EVERY", 0);
    if (dump_every < 0) dump_every = 0;
    bgj_cuda_bucket_profile.dump_every = (uint64_t)dump_every;
    bgj_cuda_bucket_profile.next_dump_seq = (uint64_t)dump_every;

    atexit(bgj_cuda_bucket_profile_dump);
}

static int bgj_cuda_bucket_profile_requested()
{
    pthread_once(&bgj_cuda_bucket_profile_once, bgj_cuda_bucket_profile_init_once);
    return bgj_cuda_bucket_profile.enabled;
}

static int bgj_cuda_bucket_profile_limit()
{
    pthread_once(&bgj_cuda_bucket_profile_once, bgj_cuda_bucket_profile_init_once);
    return bgj_cuda_bucket_profile.limit;
}

static const char *bgj_cuda_bucket_profile_kind_name(int kind)
{
    switch (kind) {
    case BGJ_CUDA_BUCKET_PROFILE_SINGLE:
        return "single";
    case BGJ_CUDA_BUCKET_PROFILE_ASYNC:
        return "async";
    case BGJ_CUDA_BUCKET_PROFILE_BATCH:
        return "batch";
    case BGJ_CUDA_BUCKET_PROFILE_BATCH_ITEM:
        return "batch_item";
    default:
        return "unknown";
    }
}

static void bgj_cuda_bucket_profile_dump_list(const char *label,
                                              const bgj_cuda_bucket_profile_entry_t *entries,
                                              int count)
{
    bgj_cuda_bucket_profile_entry_t sorted[256];
    for (int i = 0; i < count; i++) sorted[i] = entries[i];
    std::sort(sorted, sorted + count,
              [](const bgj_cuda_bucket_profile_entry_t &a,
                 const bgj_cuda_bucket_profile_entry_t &b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.seq < b.seq;
              });
    for (int i = 0; i < count; i++) {
        const bgj_cuda_bucket_profile_entry_t *e = &sorted[i];
        fprintf(stderr,
                "%s: rank=%d seq=%lu kind=%s device=%d batch=%u/%u "
                "num_p=%u num_n=%u dots=%lu results=%lu overflows=%lu "
                "failures=%lu submit=%.6fs wait=%.6fs copy=%.6fs total=%.6fs "
                "score=%lu",
                label,
                i + 1,
                (unsigned long)e->seq,
                bgj_cuda_bucket_profile_kind_name(e->kind),
                e->device,
                e->batch_index,
                e->batch_size,
                e->num_p,
                e->num_n,
                (unsigned long)e->estimated_dots,
                (unsigned long)e->results,
                (unsigned long)e->overflows,
                (unsigned long)e->failures,
                e->submit_sec,
                e->wait_sec,
                e->copy_sec,
                e->total_sec,
                (unsigned long)e->score);
        for (int phase = 0; phase < BGJ_CUDA_SEARCH_PHASE_COUNT; phase++) {
            if (e->phase_sec[phase] > 0.0) {
                fprintf(stderr, " %s=%.6fs",
                        bgj_cuda_search_phase_name(phase),
                        e->phase_sec[phase]);
            }
        }
        fprintf(stderr, "\n");
    }
}

static void bgj_cuda_bucket_profile_dump_snapshot(const char *reason)
{
    if (!bgj_cuda_bucket_profile_requested()) return;
    pthread_mutex_lock(&bgj_cuda_bucket_profile.lock);
    fprintf(stderr,
            "cuda_bucket_profile_summary: reason=%s records=%lu top=%d dump_every=%lu\n",
            reason,
            (unsigned long)bgj_cuda_bucket_profile.next_seq,
            bgj_cuda_bucket_profile.limit,
            (unsigned long)bgj_cuda_bucket_profile.dump_every);
    bgj_cuda_bucket_profile_dump_list("cuda_bucket_profile_slow",
                                      bgj_cuda_bucket_profile.slow,
                                      bgj_cuda_bucket_profile.slow_count);
    bgj_cuda_bucket_profile_dump_list("cuda_bucket_profile_work",
                                      bgj_cuda_bucket_profile.work,
                                      bgj_cuda_bucket_profile.work_count);
    fflush(stderr);
    pthread_mutex_unlock(&bgj_cuda_bucket_profile.lock);
}

static void bgj_cuda_bucket_profile_dump()
{
    bgj_cuda_bucket_profile_dump_snapshot("exit");
}

static void bgj_cuda_bucket_profile_insert_top(bgj_cuda_bucket_profile_entry_t *entries,
                                               int *count,
                                               const bgj_cuda_bucket_profile_entry_t &entry,
                                               int limit)
{
    if (*count < limit) {
        entries[*count] = entry;
        (*count)++;
        return;
    }
    int min_index = 0;
    for (int i = 1; i < *count; i++) {
        if (entries[i].score < entries[min_index].score) min_index = i;
    }
    if (entry.score > entries[min_index].score) entries[min_index] = entry;
}

static void bgj_cuda_bucket_profile_record(int kind,
                                           int device,
                                           uint32_t batch_size,
                                           uint32_t batch_index,
                                           uint32_t num_p,
                                           uint32_t num_n,
                                           uint64_t estimated_dots,
                                           uint64_t results,
                                           uint64_t overflows,
                                           uint64_t failures,
                                           double submit_sec,
                                           double wait_sec,
                                           double copy_sec,
                                           double total_sec,
                                           const bgj_cuda_search_phase_profile_t *phase_profile = NULL)
{
    if (!bgj_cuda_bucket_profile_requested()) return;

    bgj_cuda_bucket_profile_entry_t entry = {};
    entry.kind = kind;
    entry.device = device;
    entry.batch_size = batch_size;
    entry.batch_index = batch_index;
    entry.num_p = num_p;
    entry.num_n = num_n;
    entry.estimated_dots = estimated_dots;
    entry.results = results;
    entry.overflows = overflows;
    entry.failures = failures;
    entry.submit_sec = submit_sec;
    entry.wait_sec = wait_sec;
    entry.copy_sec = copy_sec;
    entry.total_sec = total_sec;
    if (phase_profile) {
        for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
            entry.phase_sec[i] = phase_profile->sec[i];
        }
    }

    int should_dump = 0;
    const int limit = bgj_cuda_bucket_profile_limit();
    pthread_mutex_lock(&bgj_cuda_bucket_profile.lock);
    entry.seq = ++bgj_cuda_bucket_profile.next_seq;
    entry.score = (uint64_t)(total_sec * 1000000.0);
    bgj_cuda_bucket_profile_insert_top(bgj_cuda_bucket_profile.slow,
                                       &bgj_cuda_bucket_profile.slow_count,
                                       entry,
                                       limit);
    entry.score = estimated_dots;
    bgj_cuda_bucket_profile_insert_top(bgj_cuda_bucket_profile.work,
                                       &bgj_cuda_bucket_profile.work_count,
                                       entry,
                                       limit);
    if (bgj_cuda_bucket_profile.dump_every &&
        entry.seq >= bgj_cuda_bucket_profile.next_dump_seq) {
        bgj_cuda_bucket_profile.next_dump_seq = entry.seq + bgj_cuda_bucket_profile.dump_every;
        should_dump = 1;
    }
    pthread_mutex_unlock(&bgj_cuda_bucket_profile.lock);

    if (should_dump) bgj_cuda_bucket_profile_dump_snapshot("periodic");
}

static int bgj_cuda_add_execution_device(int *devices, int *count, int runtime_count, long device)
{
    if (device < 0 || device >= runtime_count) return 0;
    for (int i = 0; i < *count; i++) {
        if (devices[i] == (int)device) return 1;
    }
    if (*count >= BGJ_CUDA_MAX_EXEC_DEVICES) return 0;
    devices[*count] = (int)device;
    (*count)++;
    return 1;
}

static int bgj_cuda_parse_execution_device_list(const char *env,
                                                int runtime_count,
                                                int *devices)
{
    int count = 0;
    const char *p = env;
    while (p && *p) {
        char *end = NULL;
        long device = strtol(p, &end, 10);
        if (end != p) {
            bgj_cuda_add_execution_device(devices, &count, runtime_count, device);
            p = end;
        } else {
            p++;
        }
        while (*p == ',' || *p == ';' || *p == ':' || *p == ' ' || *p == '\t') p++;
    }
    return count;
}

static void bgj_cuda_init_execution_devices_locked()
{
    bgj_cuda_execution_device_state_t *state = &bgj_cuda_execution_devices;
    if (state->initialized) return;
    state->initialized = 1;

    int runtime_count = 0;
    cudaError_t err = cudaGetDeviceCount(&runtime_count);
    if (err != cudaSuccess || runtime_count <= 0) {
        if (err != cudaSuccess) set_cuda_error("cudaGetDeviceCount", err);
        return;
    }

    const char *list_env = getenv("BGJ_CUDA_DEVICES");
    if (list_env && list_env[0]) {
        state->count =
            bgj_cuda_parse_execution_device_list(list_env, runtime_count, state->devices);
        if (state->count <= 0) {
            set_plain_error("BGJ_CUDA_DEVICES selected no valid CUDA devices");
        }
        return;
    }

    const char *num_env = getenv("BGJ_CUDA_NUM_DEVICES");
    if (num_env && num_env[0]) {
        char *end = NULL;
        long requested = strtol(num_env, &end, 10);
        if (end != num_env && requested > 0) {
            if (requested > runtime_count) requested = runtime_count;
            if (requested > BGJ_CUDA_MAX_EXEC_DEVICES) requested = BGJ_CUDA_MAX_EXEC_DEVICES;
            for (long i = 0; i < requested; i++) {
                state->devices[state->count++] = (int)i;
            }
            return;
        }
    }

    int current_device = 0;
    err = cudaGetDevice(&current_device);
    if (err == cudaSuccess && current_device >= 0 && current_device < runtime_count) {
        state->devices[state->count++] = current_device;
    } else {
        state->devices[state->count++] = 0;
    }
}

extern "C" int bgj_cuda_raw_execution_device_count()
{
    bgj_cuda_execution_device_state_t *state = &bgj_cuda_execution_devices;
    pthread_mutex_lock(&state->lock);
    bgj_cuda_init_execution_devices_locked();
    const int count = state->count;
    pthread_mutex_unlock(&state->lock);
    return count;
}

static int bgj_cuda_copy_execution_devices(int *devices, int capacity)
{
    if (!devices || capacity <= 0) return 0;
    bgj_cuda_execution_device_state_t *state = &bgj_cuda_execution_devices;
    pthread_mutex_lock(&state->lock);
    bgj_cuda_init_execution_devices_locked();
    int count = state->count;
    if (count > capacity) count = capacity;
    for (int i = 0; i < count; i++) devices[i] = state->devices[i];
    pthread_mutex_unlock(&state->lock);
    return count;
}

static int bgj_cuda_current_device(int *device)
{
    cudaError_t err = cudaGetDevice(device);
    if (err != cudaSuccess) {
        set_cuda_error("cudaGetDevice", err);
        return 0;
    }
    return 1;
}

static int bgj_cuda_set_current_device(int device)
{
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        set_cuda_error("cudaSetDevice", err);
        return 0;
    }
    return 1;
}

static int bgj_cuda_select_thread_device()
{
    if (bgj_cuda_thread_device < 0) {
        bgj_cuda_execution_device_state_t *state = &bgj_cuda_execution_devices;
        pthread_mutex_lock(&state->lock);
        bgj_cuda_init_execution_devices_locked();
        if (state->count <= 0) {
            pthread_mutex_unlock(&state->lock);
            set_plain_error("no CUDA execution devices are configured");
            return 0;
        }
        const unsigned slot = bgj_cuda_primary_thread_device_requested(state->count) ?
                              0u : state->next_slot++;
        bgj_cuda_thread_device = state->devices[slot % (unsigned)state->count];
        pthread_mutex_unlock(&state->lock);
    }
    return bgj_cuda_set_current_device(bgj_cuda_thread_device);
}

__device__ __forceinline__ int bgj_cuda_dot_i8_scalar(const int8_t *a,
                                                      const int8_t *b,
                                                      uint32_t vec_length)
{
    int acc = 0;
    for (uint32_t i = 0; i < vec_length; i++) {
        acc += (int)a[i] * (int)b[i];
    }
    return acc;
}

__device__ __forceinline__ int bgj_cuda_pack_i8x4(const int8_t *x)
{
    return (int)(uint8_t)x[0] |
           ((int)(uint8_t)x[1] << 8) |
           ((int)(uint8_t)x[2] << 16) |
           ((int)(uint8_t)x[3] << 24);
}

__device__ __forceinline__ int bgj_cuda_dot_i8(const int8_t *a,
                                               const int8_t *b,
                                               uint32_t vec_length)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    int acc = 0;
    uint32_t i = 0;

    if ((((uintptr_t)a | (uintptr_t)b) & 3u) == 0) {
        const int *a4 = (const int *)a;
        const int *b4 = (const int *)b;
        for (; i + 4 <= vec_length; i += 4) {
            acc = __dp4a(a4[i >> 2], b4[i >> 2], acc);
        }
    } else {
        for (; i + 4 <= vec_length; i += 4) {
            acc = __dp4a(bgj_cuda_pack_i8x4(a + i), bgj_cuda_pack_i8x4(b + i), acc);
        }
    }

    for (; i < vec_length; i++) {
        acc += (int)a[i] * (int)b[i];
    }
    return acc;
#else
    return bgj_cuda_dot_i8_scalar(a, b, vec_length);
#endif
}

__device__ void bgj_cuda_push_result(bgj_cuda_result_t *results,
                                     uint32_t *result_count,
                                     int *overflow,
                                     uint32_t capacity,
                                     uint32_t type,
                                     uint32_t x,
                                     uint32_t y)
{
    uint32_t out = atomicAdd(result_count, 1u);
    if (out < capacity) {
        results[out].type = type;
        results[out].x = x;
        results[out].y = y;
    } else {
        *overflow = 1;
    }
}

__global__ void bgj_cuda_transform_dp_and_search_cred_kernel(uint32_t center_id,
                                                             const uint32_t *p_ids,
                                                             const uint32_t *n_ids,
                                                             const int32_t *p_norm,
                                                             const int32_t *n_norm,
                                                             int32_t *p_dot,
                                                             int32_t *n_dot,
                                                             uint32_t num_p,
                                                             uint32_t num_n,
                                                             int32_t center_goal_norm,
                                                             bgj_cuda_result_t *results,
                                                             uint32_t result_capacity,
                                                             uint32_t *result_count,
                                                             int *overflow)
{
    const uint64_t total = (uint64_t)num_p + (uint64_t)num_n;
    const uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    uint64_t work = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    for (; work < total; work += stride) {
        if (work < num_p) {
            const uint32_t i = (uint32_t)work;
            const int32_t transformed = p_norm[i] - p_dot[i];
            p_dot[i] = transformed;
            if (transformed < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_S, center_id, p_ids[i]);
            }
        } else {
            const uint32_t i = (uint32_t)(work - num_p);
            const int32_t transformed = n_norm[i] + n_dot[i];
            n_dot[i] = transformed;
            if (transformed < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_A, center_id, n_ids[i]);
            }
        }
    }
}

__global__ void bgj_cuda_transform_dp_kernel(const int32_t *p_norm,
                                             const int32_t *n_norm,
                                             int32_t *p_dot,
                                             int32_t *n_dot,
                                             uint32_t num_p,
                                             uint32_t num_n)
{
    const uint64_t total = (uint64_t)num_p + (uint64_t)num_n;
    const uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    uint64_t work = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    for (; work < total; work += stride) {
        if (work < num_p) {
            const uint32_t i = (uint32_t)work;
            p_dot[i] = p_norm[i] - p_dot[i];
        } else {
            const uint32_t i = (uint32_t)(work - num_p);
            n_dot[i] = n_norm[i] + n_dot[i];
        }
    }
}

__device__ __forceinline__ int32_t bgj_cuda_p_center_dot(const int32_t *p_norm,
                                                         const int32_t *p_dot,
                                                         uint32_t i,
                                                         int raw_center_dp)
{
    return raw_center_dp ? p_norm[i] - p_dot[i] : p_dot[i];
}

__device__ __forceinline__ int32_t bgj_cuda_n_center_dot(const int32_t *n_norm,
                                                         const int32_t *n_dot,
                                                         uint32_t i,
                                                         int raw_center_dp)
{
    return raw_center_dp ? n_norm[i] + n_dot[i] : n_dot[i];
}

__device__ __forceinline__ int32_t bgj_cuda_same_center_dot(const int32_t *norm,
                                                            const int32_t *dot,
                                                            uint32_t i,
                                                            int raw_center_dp,
                                                            int negative_bucket)
{
    if (!raw_center_dp) return dot[i];
    return negative_bucket ? norm[i] + dot[i] : norm[i] - dot[i];
}

__device__ __forceinline__ uint64_t bgj_cuda_lsh_tile_prefix(uint64_t row, uint32_t num_tiles)
{
    return row * (uint64_t)num_tiles - (row * (row - 1u)) / 2u;
}

__device__ __forceinline__ void bgj_cuda_lsh_tile_pair(uint64_t slot,
                                                       uint32_t num_tiles,
                                                       uint32_t *tile_i,
                                                       uint32_t *tile_j)
{
    const double m = (double)(2u * num_tiles + 1u);
    const double d = m * m - 8.0 * (double)slot;
    uint64_t row = (uint64_t)((m - sqrt(d > 0.0 ? d : 0.0)) * 0.5);
    while (row + 1u < (uint64_t)num_tiles &&
           bgj_cuda_lsh_tile_prefix(row + 1u, num_tiles) <= slot) {
        row++;
    }
    while (row > 0u && bgj_cuda_lsh_tile_prefix(row, num_tiles) > slot) {
        row--;
    }
    const uint64_t col = row + slot - bgj_cuda_lsh_tile_prefix(row, num_tiles);
    *tile_i = (uint32_t)row;
    *tile_j = (uint32_t)col;
}

__global__ __launch_bounds__(BGJ_CUDA_LSH_THREADS, 2)
void bgj_cuda_lsh_search_kernel(const uint64_t *sh,
                                uint32_t mbound,
                                int32_t threshold,
                                uint64_t tile_slot_begin,
                                uint64_t tile_slots,
                                bgj_cuda_result_t *results,
                                uint32_t result_capacity,
                                uint32_t *result_count,
                                int *overflow)
{
    __shared__ uint64_t tile_a[BGJ_CUDA_LSH_TILE][8];
    __shared__ uint64_t tile_b[BGJ_CUDA_LSH_TILE][8];
    const uint32_t num_tiles = (mbound + BGJ_CUDA_LSH_TILE - 1u) / BGJ_CUDA_LSH_TILE;
    const int32_t upper_threshold = 512 - threshold;

    for (uint64_t tile_slot_offset = blockIdx.x; tile_slot_offset < tile_slots; tile_slot_offset += gridDim.x) {
        const uint64_t tile_slot = tile_slot_begin + tile_slot_offset;
        uint32_t tile_i = 0;
        uint32_t tile_j = 0;
        bgj_cuda_lsh_tile_pair(tile_slot, num_tiles, &tile_i, &tile_j);

        for (uint32_t k = threadIdx.x; k < BGJ_CUDA_LSH_TILE * 8u; k += blockDim.x) {
            const uint32_t row = k >> 3;
            const uint32_t word = k & 7u;
            const uint32_t ai = tile_i * BGJ_CUDA_LSH_TILE + row;
            const uint32_t bj = tile_j * BGJ_CUDA_LSH_TILE + row;
            tile_a[row][word] = ai < mbound ? sh[(uint64_t)ai * 8u + word] : 0u;
            tile_b[row][word] = bj < mbound ? sh[(uint64_t)bj * 8u + word] : 0u;
        }
        __syncthreads();

        const uint32_t ai_local = threadIdx.x >> 4;
        const uint32_t bj_local = threadIdx.x & 15u;
        const uint32_t i = tile_i * BGJ_CUDA_LSH_TILE + ai_local;
        const uint32_t j = tile_j * BGJ_CUDA_LSH_TILE + bj_local;
        if (i < mbound && j < mbound && (tile_i != tile_j || ai_local < bj_local)) {
            uint32_t dist = 0;
            #pragma unroll
            for (uint32_t word = 0; word < 8u; word++) {
                dist += __popcll(tile_a[ai_local][word] ^ tile_b[bj_local][word]);
            }
            if ((int32_t)dist <= threshold) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity, BGJ_CUDA_SOL_S, i, j);
            } else if ((int32_t)dist >= upper_threshold) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity, BGJ_CUDA_SOL_A, i, j);
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(BGJ_CUDA_LSH_LIFT_THREADS, 2)
void bgj_cuda_lsh_lift_kernel(const float *fvec,
                              uint32_t mbound,
                              uint32_t fd,
                              uint32_t fd8,
                              const uint32_t *candidates,
                              uint32_t candidate_count,
                              const float *b_full,
                              const float *idiag,
                              const float *min_norm,
                              uint32_t id,
                              uint32_t csd,
                              float threshold_scale,
                              bgj_cuda_lsh_lift_result_t *results,
                              uint32_t result_capacity,
                              uint32_t *result_count,
                              int *overflow)
{
    const uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;
    if (fd == 0u || fd > BGJ_CUDA_LSH_LIFT_MAX_FD || fd8 > BGJ_CUDA_LSH_LIFT_MAX_FD ||
        csd == 0u || csd > fd || id == 0u || id > fd) {
        return;
    }

    const uint32_t base = candidate * 3u;
    const uint32_t type = candidates[base];
    const uint32_t x = candidates[base + 1u];
    const uint32_t y = candidates[base + 2u];
    if (x >= mbound || y >= mbound) return;

    float tmp[BGJ_CUDA_LSH_LIFT_MAX_FD];
    const float *xvec = fvec + (uint64_t)x * fd8;
    const float *yvec = fvec + (uint64_t)y * fd8;
    for (uint32_t i = 0; i < fd; i++) {
        tmp[i] = type ? (xvec[i] - yvec[i]) : (xvec[i] + yvec[i]);
    }

    const uint32_t ld = fd - csd;
    float curr_norm = 0.0f;
    for (uint32_t i = ld; i < fd; i++) {
        curr_norm += tmp[i] * tmp[i];
    }
    for (int row = (int)ld - 1; row >= 0; row--) {
        const float q = roundf(tmp[row] * idiag[row]);
        const float *basis_row = b_full + (uint64_t)row * fd8;
        for (int col = 0; col <= row; col++) {
            tmp[col] -= basis_row[col] * q;
        }
        curr_norm += tmp[row] * tmp[row];
        if ((uint32_t)row >= id) continue;
        if (curr_norm < min_norm[row] * threshold_scale) {
            const uint32_t pos = atomicAdd(result_count, 1u);
            if (pos < result_capacity) {
                results[pos].candidate = candidate;
                results[pos].place = (uint32_t)row;
                results[pos].norm = curr_norm;
            } else {
                *overflow = 1;
            }
        }
        if (curr_norm > min_norm[0] * threshold_scale) {
            break;
        }
    }
}

__device__ __forceinline__ void bgj_cuda_tensor_accumulator_coord(uint32_t lane_id,
                                                                  uint32_t element,
                                                                  uint32_t *row,
                                                                  uint32_t *col)
{
    *row = lane_id / 4u + ((element & 2u) ? 8u : 0u);
    *col = 2u * (lane_id & 3u) + (element & 1u) + ((element & 4u) ? 8u : 0u);
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_pack_a_frag_kernel(const int8_t *vecs,
                                 uint32_t num_blocks,
                                 uint32_t k_blocks,
                                 uint32_t vec_length,
                                 uint64_t *out_frags)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t frag_index = blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    const uint32_t total_frags = num_blocks * k_blocks;
    if (frag_index >= total_frags) return;

    const uint32_t row_block = frag_index / k_blocks;
    const uint32_t k_block = frag_index - row_block * k_blocks;
    bgj_cuda_i8_a_frag_u frag;
    wmma::load_matrix_sync(frag.frag,
                           (const signed char *)(vecs + (uint64_t)row_block * 16u * vec_length + k_block * 16u),
                           vec_length);

    uint64_t *dst = out_frags +
                    ((uint64_t)frag_index * BGJ_CUDA_WARP_SIZE + lane_id) *
                    BGJ_CUDA_I8_A_FRAG_WORDS;
    #pragma unroll
    for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
        dst[word] = frag.word[word];
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_pack_b_frag_kernel(const int8_t *vecs,
                                 uint32_t num_blocks,
                                 uint32_t k_blocks,
                                 uint32_t vec_length,
                                 uint64_t *out_frags)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t frag_index = blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    const uint32_t total_frags = num_blocks * k_blocks;
    if (frag_index >= total_frags) return;

    const uint32_t row_block = frag_index / k_blocks;
    const uint32_t k_block = frag_index - row_block * k_blocks;
    bgj_cuda_i8_b_frag_u frag;
    wmma::load_matrix_sync(frag.frag,
                           (const signed char *)(vecs + (uint64_t)row_block * 16u * vec_length + k_block * 16u),
                           vec_length);

    uint64_t *dst = out_frags +
                    ((uint64_t)frag_index * BGJ_CUDA_WARP_SIZE + lane_id) *
                    BGJ_CUDA_I8_B_FRAG_WORDS;
    #pragma unroll
    for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
        dst[word] = frag.word[word];
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_np_tensor_kernel(const int8_t *p_vecs,
                                      const int8_t *n_vecs,
                                      const uint32_t *p_ids,
                                      const uint32_t *n_ids,
                                      const int32_t *p_norm,
                                      const int32_t *n_norm,
                                      const int32_t *p_dot,
                                      const int32_t *n_dot,
                                      uint32_t tensor_blocks_n,
                                      uint32_t tensor_tiles,
                                      uint32_t vec_length,
                                      int32_t goal_norm,
                                      int32_t center_goal_norm,
                                      int record_dp,
                                      int raw_center_dp,
                                      bgj_cuda_result_t *results,
                                      uint32_t result_capacity,
                                      uint32_t *result_count,
                                      int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t tensor_tile = blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    if (tensor_tile >= tensor_tiles) return;

    const uint32_t tile_p = (tensor_tile / tensor_blocks_n) * 16u;
    const uint32_t tile_n = (tensor_tile - (tensor_tile / tensor_blocks_n) * tensor_blocks_n) * 16u;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k = 0; k < vec_length; k += 16) {
        wmma::load_matrix_sync(a_frag, (const signed char *)(p_vecs + (uint64_t)tile_p * vec_length + k), vec_length);
        wmma::load_matrix_sync(b_frag, (const signed char *)(n_vecs + (uint64_t)tile_n * vec_length + k), vec_length);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t p = tile_p + row;
        const uint32_t n = tile_n + col;
        const int32_t dp = c_frag.x[element];

        if (p_norm[p] + n_norm[n] + dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
        }
        if (record_dp &&
            bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
            bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_np_tensor_reordered_kernel(const uint64_t *p_a_frags,
                                                const uint64_t *n_b_frags,
                                                const uint32_t *p_ids,
                                                const uint32_t *n_ids,
                                                const int32_t *p_norm,
                                                const int32_t *n_norm,
                                                const int32_t *p_dot,
                                                const int32_t *n_dot,
                                                uint32_t tensor_blocks_n,
                                                uint32_t tensor_tiles,
                                                uint32_t k_blocks,
                                                int32_t goal_norm,
                                                int32_t center_goal_norm,
                                                int record_dp,
                                                int raw_center_dp,
                                                bgj_cuda_result_t *results,
                                                uint32_t result_capacity,
                                                uint32_t *result_count,
                                                int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t tensor_tile = blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    if (tensor_tile >= tensor_tiles) return;

    const uint32_t tile_block_p = tensor_tile / tensor_blocks_n;
    const uint32_t tile_block_n = tensor_tile - tile_block_p * tensor_blocks_n;
    const uint32_t tile_p = tile_block_p * 16u;
    const uint32_t tile_n = tile_block_n * 16u;

    bgj_cuda_i8_a_frag_u a_frag;
    bgj_cuda_i8_b_frag_u b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        const uint64_t *a_src = p_a_frags +
                                (((uint64_t)tile_block_p * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_A_FRAG_WORDS;
        const uint64_t *b_src = n_b_frags +
                                (((uint64_t)tile_block_n * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_B_FRAG_WORDS;
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
            a_frag.word[word] = __ldg(a_src + word);
        }
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
            b_frag.word[word] = __ldg(b_src + word);
        }
        wmma::mma_sync(c_frag, a_frag.frag, b_frag.frag, c_frag);
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t p = tile_p + row;
        const uint32_t n = tile_n + col;
        const int32_t dp = c_frag.x[element];

        if (p_norm[p] + n_norm[n] + dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
        }
        if (record_dp &&
            bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
            bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_np_tensor_reordered_range_kernel(const uint64_t *p_a_frags,
                                                      const uint64_t *n_b_frags,
                                                      const uint32_t *p_ids,
                                                      const uint32_t *n_ids,
                                                      const int32_t *p_norm,
                                                      const int32_t *n_norm,
                                                      const int32_t *p_dot,
                                                      const int32_t *n_dot,
                                                      uint32_t tensor_blocks_n,
                                                      uint64_t tile_begin,
                                                      uint64_t tile_count,
                                                      uint32_t k_blocks,
                                                      int32_t goal_norm,
                                                      int32_t center_goal_norm,
                                                      int record_dp,
                                                      int raw_center_dp,
                                                      bgj_cuda_result_t *results,
                                                      uint32_t result_capacity,
                                                      uint32_t *result_count,
                                                      int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint64_t local_tile =
        (uint64_t)blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    if (local_tile >= tile_count) return;

    const uint64_t tensor_tile = tile_begin + local_tile;
    const uint32_t tile_block_p = (uint32_t)(tensor_tile / tensor_blocks_n);
    const uint32_t tile_block_n =
        (uint32_t)(tensor_tile - (uint64_t)tile_block_p * tensor_blocks_n);
    const uint32_t tile_p = tile_block_p * 16u;
    const uint32_t tile_n = tile_block_n * 16u;

    bgj_cuda_i8_a_frag_u a_frag;
    bgj_cuda_i8_b_frag_u b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        const uint64_t *a_src = p_a_frags +
                                (((uint64_t)tile_block_p * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_A_FRAG_WORDS;
        const uint64_t *b_src = n_b_frags +
                                (((uint64_t)tile_block_n * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_B_FRAG_WORDS;
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
            a_frag.word[word] = __ldg(a_src + word);
        }
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
            b_frag.word[word] = __ldg(b_src + word);
        }
        wmma::mma_sync(c_frag, a_frag.frag, b_frag.frag, c_frag);
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t p = tile_p + row;
        const uint32_t n = tile_n + col;
        const int32_t dp = c_frag.x[element];

        if (p_norm[p] + n_norm[n] + dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
        }
        if (record_dp &&
            bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
            bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 1)
void bgj_cuda_search_np_tensor_multi_kernel(const uint64_t *p_a_frags,
                                            const uint64_t *n_b_frags,
                                            const uint32_t *p_ids,
                                            const uint32_t *n_ids,
                                            const int32_t *p_norm,
                                            const int32_t *n_norm,
                                            const int32_t *p_dot,
                                            const int32_t *n_dot,
                                            uint32_t group_blocks_n,
                                            uint32_t k_blocks,
                                            int32_t goal_norm,
                                            int32_t center_goal_norm,
                                            int record_dp,
                                            int raw_center_dp,
                                            bgj_cuda_result_t *results,
                                            uint32_t result_capacity,
                                            uint32_t *result_count,
                                            int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t group_p = blockIdx.x / group_blocks_n;
    const uint32_t group_n = blockIdx.x - group_p * group_blocks_n;
    const uint32_t base_p_block = group_p * BGJ_CUDA_TENSOR_NP_MULTI_P_TILES;
    const uint32_t base_n_block =
        group_n * BGJ_CUDA_TENSOR_NP_MULTI_N_TILES +
        warp_id * BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP;

    bgj_cuda_i8_a_frag_u a_frag[BGJ_CUDA_TENSOR_NP_MULTI_P_TILES];
    bgj_cuda_i8_b_frag_u b_frag[BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int>
        c_frag[BGJ_CUDA_TENSOR_NP_MULTI_P_TILES][BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP];

    #pragma unroll
    for (uint32_t row_tile = 0; row_tile < BGJ_CUDA_TENSOR_NP_MULTI_P_TILES; row_tile++) {
        #pragma unroll
        for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP; col_tile++) {
            wmma::fill_fragment(c_frag[row_tile][col_tile], 0);
        }
    }

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        #pragma unroll
        for (uint32_t row_tile = 0; row_tile < BGJ_CUDA_TENSOR_NP_MULTI_P_TILES; row_tile++) {
            const uint64_t *a_src = p_a_frags +
                                    (((uint64_t)(base_p_block + row_tile) * k_blocks + k_block) *
                                     BGJ_CUDA_WARP_SIZE + lane_id) *
                                    BGJ_CUDA_I8_A_FRAG_WORDS;
            #pragma unroll
            for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
                a_frag[row_tile].word[word] = __ldg(a_src + word);
            }
        }
        #pragma unroll
        for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP; col_tile++) {
            const uint64_t *b_src = n_b_frags +
                                    (((uint64_t)(base_n_block + col_tile) * k_blocks + k_block) *
                                     BGJ_CUDA_WARP_SIZE + lane_id) *
                                    BGJ_CUDA_I8_B_FRAG_WORDS;
            #pragma unroll
            for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
                b_frag[col_tile].word[word] = __ldg(b_src + word);
            }
        }

        #pragma unroll
        for (uint32_t row_tile = 0; row_tile < BGJ_CUDA_TENSOR_NP_MULTI_P_TILES; row_tile++) {
            #pragma unroll
            for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP; col_tile++) {
                wmma::mma_sync(c_frag[row_tile][col_tile],
                               a_frag[row_tile].frag,
                               b_frag[col_tile].frag,
                               c_frag[row_tile][col_tile]);
            }
        }
    }

    #pragma unroll
    for (uint32_t row_tile = 0; row_tile < BGJ_CUDA_TENSOR_NP_MULTI_P_TILES; row_tile++) {
        #pragma unroll
        for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP; col_tile++) {
            const uint32_t tile_p = (base_p_block + row_tile) * 16u;
            const uint32_t tile_n = (base_n_block + col_tile) * 16u;
            #pragma unroll
            for (uint32_t element = 0; element < c_frag[row_tile][col_tile].num_elements; element++) {
                uint32_t row;
                uint32_t col;
                bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
                const uint32_t p = tile_p + row;
                const uint32_t n = tile_n + col;
                const int32_t dp = c_frag[row_tile][col_tile].x[element];

                if (p_norm[p] + n_norm[n] + dp < goal_norm) {
                    bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                         BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
                }
                if (record_dp &&
                    bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
                    bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
                    bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                         BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
                }
            }
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 1)
void bgj_cuda_search_np_tensor_wide_reordered_kernel(const uint64_t *p_a_frags,
                                                     const uint64_t *n_b_frags,
                                                     const uint32_t *p_ids,
                                                     const uint32_t *n_ids,
                                                     const int32_t *p_norm,
                                                     const int32_t *n_norm,
                                                     const int32_t *p_dot,
                                                     const int32_t *n_dot,
                                                     uint32_t group_blocks_n,
                                                     uint32_t k_blocks,
                                                     int32_t goal_norm,
                                                     int32_t center_goal_norm,
                                                     int record_dp,
                                                     int raw_center_dp,
                                                     bgj_cuda_result_t *results,
                                                     uint32_t result_capacity,
                                                     uint32_t *result_count,
                                                     int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t group_p = blockIdx.x / group_blocks_n;
    const uint32_t group_n = blockIdx.x - group_p * group_blocks_n;
    const uint32_t tile_block_p = group_p * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    const uint32_t base_n_block = group_n * BGJ_CUDA_TENSOR_NP_WIDE_TILES;

    bgj_cuda_i8_a_frag_u a_frag;
    bgj_cuda_i8_b_frag_u b_frag[BGJ_CUDA_TENSOR_NP_WIDE_TILES];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[BGJ_CUDA_TENSOR_NP_WIDE_TILES];

    #pragma unroll
    for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col_tile++) {
        wmma::fill_fragment(c_frag[col_tile], 0);
    }

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        const uint64_t *a_src = p_a_frags +
                                (((uint64_t)tile_block_p * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_A_FRAG_WORDS;
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
            a_frag.word[word] = __ldg(a_src + word);
        }
        #pragma unroll
        for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col_tile++) {
            const uint64_t *b_src = n_b_frags +
                                    (((uint64_t)(base_n_block + col_tile) * k_blocks + k_block) *
                                     BGJ_CUDA_WARP_SIZE + lane_id) *
                                    BGJ_CUDA_I8_B_FRAG_WORDS;
            #pragma unroll
            for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
                b_frag[col_tile].word[word] = __ldg(b_src + word);
            }
            wmma::mma_sync(c_frag[col_tile], a_frag.frag, b_frag[col_tile].frag, c_frag[col_tile]);
        }
    }

    #pragma unroll
    for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col_tile++) {
        const uint32_t tile_p = tile_block_p * 16u;
        const uint32_t tile_n = (base_n_block + col_tile) * 16u;
        #pragma unroll
        for (uint32_t element = 0; element < c_frag[col_tile].num_elements; element++) {
            uint32_t row;
            uint32_t col;
            bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
            const uint32_t p = tile_p + row;
            const uint32_t n = tile_n + col;
            const int32_t dp = c_frag[col_tile].x[element];

            if (p_norm[p] + n_norm[n] + dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
            }
            if (record_dp &&
                bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
                bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
            }
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 1)
void bgj_cuda_search_np_tensor_wide_reordered_range_kernel(const uint64_t *p_a_frags,
                                                           const uint64_t *n_b_frags,
                                                           const uint32_t *p_ids,
                                                           const uint32_t *n_ids,
                                                           const int32_t *p_norm,
                                                           const int32_t *n_norm,
                                                           const int32_t *p_dot,
                                                           const int32_t *n_dot,
                                                           uint32_t group_blocks_n,
                                                           uint64_t group_begin,
                                                           uint64_t group_count,
                                                           uint32_t k_blocks,
                                                           int32_t goal_norm,
                                                           int32_t center_goal_norm,
                                                           int record_dp,
                                                           int raw_center_dp,
                                                           bgj_cuda_result_t *results,
                                                           uint32_t result_capacity,
                                                           uint32_t *result_count,
                                                           int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint64_t local_group = (uint64_t)blockIdx.x;
    if (local_group >= group_count) return;

    const uint64_t tensor_group = group_begin + local_group;
    const uint32_t group_p = (uint32_t)(tensor_group / group_blocks_n);
    const uint32_t group_n =
        (uint32_t)(tensor_group - (uint64_t)group_p * group_blocks_n);
    const uint32_t tile_block_p = group_p * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    const uint32_t base_n_block = group_n * BGJ_CUDA_TENSOR_NP_WIDE_TILES;

    bgj_cuda_i8_a_frag_u a_frag;
    bgj_cuda_i8_b_frag_u b_frag[BGJ_CUDA_TENSOR_NP_WIDE_TILES];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[BGJ_CUDA_TENSOR_NP_WIDE_TILES];

    #pragma unroll
    for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col_tile++) {
        wmma::fill_fragment(c_frag[col_tile], 0);
    }

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        const uint64_t *a_src = p_a_frags +
                                (((uint64_t)tile_block_p * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_A_FRAG_WORDS;
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
            a_frag.word[word] = __ldg(a_src + word);
        }
        #pragma unroll
        for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col_tile++) {
            const uint64_t *b_src = n_b_frags +
                                    (((uint64_t)(base_n_block + col_tile) * k_blocks + k_block) *
                                     BGJ_CUDA_WARP_SIZE + lane_id) *
                                    BGJ_CUDA_I8_B_FRAG_WORDS;
            #pragma unroll
            for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
                b_frag[col_tile].word[word] = __ldg(b_src + word);
            }
            wmma::mma_sync(c_frag[col_tile], a_frag.frag, b_frag[col_tile].frag, c_frag[col_tile]);
        }
    }

    #pragma unroll
    for (uint32_t col_tile = 0; col_tile < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col_tile++) {
        const uint32_t tile_p = tile_block_p * 16u;
        const uint32_t tile_n = (base_n_block + col_tile) * 16u;
        #pragma unroll
        for (uint32_t element = 0; element < c_frag[col_tile].num_elements; element++) {
            uint32_t row;
            uint32_t col;
            bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
            const uint32_t p = tile_p + row;
            const uint32_t n = tile_n + col;
            const int32_t dp = c_frag[col_tile].x[element];

            if (p_norm[p] + n_norm[n] + dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
            }
            if (record_dp &&
                bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
                bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
            }
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_np_tensor_wide_kernel(const int8_t *p_vecs,
                                           const int8_t *n_vecs,
                                           const uint32_t *p_ids,
                                           const uint32_t *n_ids,
                                           const int32_t *p_norm,
                                           const int32_t *n_norm,
                                           const int32_t *p_dot,
                                           const int32_t *n_dot,
                                           uint32_t group_blocks_n,
                                           uint32_t vec_length,
                                           int32_t goal_norm,
                                           int32_t center_goal_norm,
                                           int record_dp,
                                           int raw_center_dp,
                                           bgj_cuda_result_t *results,
                                           uint32_t result_capacity,
                                           uint32_t *result_count,
                                           int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t group_p = blockIdx.x / group_blocks_n;
    const uint32_t group_n = blockIdx.x - group_p * group_blocks_n;
    const uint32_t tile_p = (group_p * BGJ_CUDA_TENSOR_NP_WIDE_TILES + warp_id) * 16u;
    const uint32_t tile_n = group_n * BGJ_CUDA_TENSOR_NP_WIDE_TILES * 16u;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[BGJ_CUDA_TENSOR_NP_WIDE_TILES];

    for (uint32_t col = 0; col < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col++) {
        wmma::fill_fragment(c_frag[col], 0);
    }

    for (uint32_t k = 0; k < vec_length; k += 16) {
        wmma::load_matrix_sync(a_frag, (const signed char *)(p_vecs + (uint64_t)tile_p * vec_length + k), vec_length);
        for (uint32_t col = 0; col < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col++) {
            const uint32_t col_tile_n = tile_n + col * 16u;
            wmma::load_matrix_sync(b_frag, (const signed char *)(n_vecs + (uint64_t)col_tile_n * vec_length + k), vec_length);
            wmma::mma_sync(c_frag[col], a_frag, b_frag, c_frag[col]);
        }
    }

    #pragma unroll
    for (uint32_t col = 0; col < BGJ_CUDA_TENSOR_NP_WIDE_TILES; col++) {
        const uint32_t col_tile_n = tile_n + col * 16u;
        #pragma unroll
        for (uint32_t element = 0; element < c_frag[col].num_elements; element++) {
            uint32_t row;
            uint32_t tile_col;
            bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &tile_col);
            const uint32_t p = tile_p + row;
            const uint32_t n = col_tile_n + tile_col;
            const int32_t dp = c_frag[col].x[element];

            if (p_norm[p] + n_norm[n] + dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
            }
            if (record_dp &&
                bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
                bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
            }
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_np_tensor_shared_a_kernel(const int8_t *p_vecs,
                                               const int8_t *n_vecs,
                                               const uint32_t *p_ids,
                                               const uint32_t *n_ids,
                                               const int32_t *p_norm,
                                               const int32_t *n_norm,
                                               const int32_t *p_dot,
                                               const int32_t *n_dot,
                                               uint32_t group_blocks_n,
                                               uint32_t vec_length,
                                               int32_t goal_norm,
                                               int32_t center_goal_norm,
                                               int record_dp,
                                               int raw_center_dp,
                                               bgj_cuda_result_t *results,
                                               uint32_t result_capacity,
                                               uint32_t *result_count,
                                               int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t group_p = blockIdx.x / group_blocks_n;
    const uint32_t group_n = blockIdx.x - group_p * group_blocks_n;
    const uint32_t tile_p = group_p * 16u;
    const uint32_t tile_n = (group_n * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id) * 16u;

    __shared__ int8_t a_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k = 0; k < vec_length; k += 16) {
        for (uint32_t idx = threadIdx.x; idx < 16u * 16u; idx += BGJ_CUDA_TENSOR_THREADS_PER_BLOCK) {
            const uint32_t row = idx / 16u;
            const uint32_t col = idx % 16u;
            a_tile[idx] = p_vecs[(uint64_t)(tile_p + row) * vec_length + k + col];
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, (const signed char *)a_tile, 16);
        wmma::load_matrix_sync(b_frag, (const signed char *)(n_vecs + (uint64_t)tile_n * vec_length + k), vec_length);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t p = tile_p + row;
        const uint32_t n = tile_n + col;
        const int32_t dp = c_frag.x[element];

        if (p_norm[p] + n_norm[n] + dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
        }
        if (record_dp &&
            bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
            bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
        }
    }
}

__device__ __forceinline__ void bgj_cuda_upper_tile_from_linear(uint32_t tile_index,
                                                                uint32_t num_tiles,
                                                                uint32_t *tile_i,
                                                                uint32_t *tile_j)
{
    uint32_t lo = 0;
    uint32_t hi = num_tiles;
    while (lo + 1 < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const uint64_t first = (uint64_t)mid * num_tiles - ((uint64_t)mid * (mid - 1u)) / 2u;
        if (first <= tile_index) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const uint64_t first = (uint64_t)lo * num_tiles - ((uint64_t)lo * (lo - 1u)) / 2u;
    *tile_i = lo;
    *tile_j = lo + (uint32_t)((uint64_t)tile_index - first);
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_same_tensor_kernel(const int8_t *vecs,
                                        const uint32_t *ids,
                                        const int32_t *norm,
                                        const int32_t *center_dot,
                                        uint32_t num_tiles,
                                        uint32_t tensor_tiles,
                                        uint32_t vec_length,
                                        int32_t goal_norm,
                                        int32_t center_goal_norm,
                                        int record_dp,
                                        int raw_center_dp,
                                        int negative_bucket,
                                        bgj_cuda_result_t *results,
                                        uint32_t result_capacity,
                                        uint32_t *result_count,
                                        int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t tensor_tile = blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    if (tensor_tile >= tensor_tiles) return;

    uint32_t tile_i;
    uint32_t tile_j;
    bgj_cuda_upper_tile_from_linear(tensor_tile, num_tiles, &tile_i, &tile_j);
    const uint32_t base_i = tile_i * 16u;
    const uint32_t base_j = tile_j * 16u;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k = 0; k < vec_length; k += 16) {
        wmma::load_matrix_sync(a_frag, (const signed char *)(vecs + (uint64_t)base_i * vec_length + k), vec_length);
        wmma::load_matrix_sync(b_frag, (const signed char *)(vecs + (uint64_t)base_j * vec_length + k), vec_length);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t i = base_i + row;
        const uint32_t j = base_j + col;
        if (j <= i) continue;

        const int32_t dp = c_frag.x[element];
        if (norm[i] + norm[j] - dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_S, ids[i], ids[j]);
        }
        if (record_dp &&
            bgj_cuda_same_center_dot(norm, center_dot, i, raw_center_dp, negative_bucket) +
            bgj_cuda_same_center_dot(norm, center_dot, j, raw_center_dp, negative_bucket) +
            dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 negative_bucket ? BGJ_CUDA_SOL_AA : BGJ_CUDA_SOL_SS,
                                 ids[i], ids[j]);
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_same_tensor_reordered_kernel(const uint64_t *a_frags,
                                                  const uint64_t *b_frags,
                                                  const uint32_t *ids,
                                                  const int32_t *norm,
                                                  const int32_t *center_dot,
                                                  uint32_t num_tiles,
                                                  uint32_t tensor_tiles,
                                                  uint32_t k_blocks,
                                                  int32_t goal_norm,
                                                  int32_t center_goal_norm,
                                                  int record_dp,
                                                  int raw_center_dp,
                                                  int negative_bucket,
                                                  bgj_cuda_result_t *results,
                                                  uint32_t result_capacity,
                                                  uint32_t *result_count,
                                                  int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t tensor_tile = blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    if (tensor_tile >= tensor_tiles) return;

    uint32_t tile_i;
    uint32_t tile_j;
    bgj_cuda_upper_tile_from_linear(tensor_tile, num_tiles, &tile_i, &tile_j);
    const uint32_t base_i = tile_i * 16u;
    const uint32_t base_j = tile_j * 16u;

    bgj_cuda_i8_a_frag_u a_frag;
    bgj_cuda_i8_b_frag_u b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        const uint64_t *a_src = a_frags +
                                (((uint64_t)tile_i * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_A_FRAG_WORDS;
        const uint64_t *b_src = b_frags +
                                (((uint64_t)tile_j * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_B_FRAG_WORDS;
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
            a_frag.word[word] = __ldg(a_src + word);
        }
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
            b_frag.word[word] = __ldg(b_src + word);
        }
        wmma::mma_sync(c_frag, a_frag.frag, b_frag.frag, c_frag);
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t i = base_i + row;
        const uint32_t j = base_j + col;
        if (j <= i) continue;

        const int32_t dp = c_frag.x[element];
        if (norm[i] + norm[j] - dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_S, ids[i], ids[j]);
        }
        if (record_dp &&
            bgj_cuda_same_center_dot(norm, center_dot, i, raw_center_dp, negative_bucket) +
            bgj_cuda_same_center_dot(norm, center_dot, j, raw_center_dp, negative_bucket) +
            dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 negative_bucket ? BGJ_CUDA_SOL_AA : BGJ_CUDA_SOL_SS,
                                 ids[i], ids[j]);
        }
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_search_same_tensor_reordered_range_kernel(const uint64_t *a_frags,
                                                        const uint64_t *b_frags,
                                                        const uint32_t *ids,
                                                        const int32_t *norm,
                                                        const int32_t *center_dot,
                                                        uint32_t num_tiles,
                                                        uint64_t tile_begin,
                                                        uint64_t tile_count,
                                                        uint32_t k_blocks,
                                                        int32_t goal_norm,
                                                        int32_t center_goal_norm,
                                                        int record_dp,
                                                        int raw_center_dp,
                                                        int negative_bucket,
                                                        bgj_cuda_result_t *results,
                                                        uint32_t result_capacity,
                                                        uint32_t *result_count,
                                                        int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint64_t local_tile =
        (uint64_t)blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    if (local_tile >= tile_count) return;

    const uint64_t tensor_tile = tile_begin + local_tile;
    uint32_t tile_i;
    uint32_t tile_j;
    bgj_cuda_upper_tile_from_linear((uint32_t)tensor_tile, num_tiles, &tile_i, &tile_j);
    const uint32_t base_i = tile_i * 16u;
    const uint32_t base_j = tile_j * 16u;

    bgj_cuda_i8_a_frag_u a_frag;
    bgj_cuda_i8_b_frag_u b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k_block = 0; k_block < k_blocks; k_block++) {
        const uint64_t *a_src = a_frags +
                                (((uint64_t)tile_i * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_A_FRAG_WORDS;
        const uint64_t *b_src = b_frags +
                                (((uint64_t)tile_j * k_blocks + k_block) *
                                 BGJ_CUDA_WARP_SIZE + lane_id) *
                                BGJ_CUDA_I8_B_FRAG_WORDS;
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_A_FRAG_WORDS; word++) {
            a_frag.word[word] = __ldg(a_src + word);
        }
        #pragma unroll
        for (uint32_t word = 0; word < BGJ_CUDA_I8_B_FRAG_WORDS; word++) {
            b_frag.word[word] = __ldg(b_src + word);
        }
        wmma::mma_sync(c_frag, a_frag.frag, b_frag.frag, c_frag);
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t i = base_i + row;
        const uint32_t j = base_j + col;
        if (j <= i) continue;

        const int32_t dp = c_frag.x[element];
        if (norm[i] + norm[j] - dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_S, ids[i], ids[j]);
        }
        if (record_dp &&
            bgj_cuda_same_center_dot(norm, center_dot, i, raw_center_dp, negative_bucket) +
            bgj_cuda_same_center_dot(norm, center_dot, j, raw_center_dp, negative_bucket) +
            dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 negative_bucket ? BGJ_CUDA_SOL_AA : BGJ_CUDA_SOL_SS,
                                 ids[i], ids[j]);
        }
    }
}

__global__ void bgj_cuda_search_bucket_kernel(const int8_t *p_vecs,
                                              const int8_t *n_vecs,
                                              const uint32_t *p_ids,
                                              const uint32_t *n_ids,
                                              const int32_t *p_norm,
                                              const int32_t *n_norm,
                                              const int32_t *p_dot,
                                              const int32_t *n_dot,
                                              uint32_t num_p,
                                              uint32_t num_n,
                                              uint32_t tensor_np_num_p,
                                              uint32_t tensor_np_num_n,
                                              uint32_t tensor_same_num_p,
                                              uint32_t tensor_same_num_n,
                                              uint32_t vec_length,
                                              int32_t goal_norm,
                                              int32_t center_goal_norm,
                                              int record_dp,
                                              int raw_center_dp,
                                              bgj_cuda_result_t *results,
                                              uint32_t result_capacity,
                                              uint32_t *result_count,
                                              int *overflow)
{
    const int tensor_active = tensor_np_num_p != 0 && tensor_np_num_n != 0;
    const uint32_t tensor_n_tail = num_n - tensor_np_num_n;
    const uint64_t np_tensor_n_tail = tensor_active ? (uint64_t)tensor_np_num_p * (uint64_t)tensor_n_tail : 0;
    const uint64_t np_tensor_p_tail = tensor_active ? (uint64_t)(num_p - tensor_np_num_p) * (uint64_t)num_n : 0;
    const uint64_t np_total = tensor_active ? np_tensor_n_tail + np_tensor_p_tail :
                                             (uint64_t)num_p * (uint64_t)num_n;
    const int tensor_p_same_active = tensor_same_num_p != 0;
    const int tensor_n_same_active = tensor_same_num_n != 0;
    const uint32_t p_tail = num_p - tensor_same_num_p;
    const uint32_t n_tail = num_n - tensor_same_num_n;
    const uint64_t pp_head_tail = tensor_p_same_active ? (uint64_t)tensor_same_num_p * (uint64_t)p_tail : 0;
    const uint64_t nn_head_tail = tensor_n_same_active ? (uint64_t)tensor_same_num_n * (uint64_t)n_tail : 0;
    const uint64_t pp_tail_rows = tensor_p_same_active ? (uint64_t)p_tail * (uint64_t)num_p : 0;
    const uint64_t nn_tail_rows = tensor_n_same_active ? (uint64_t)n_tail * (uint64_t)num_n : 0;
    const uint64_t pp_total = num_p > 1 ? (tensor_p_same_active ? pp_head_tail + pp_tail_rows :
                                                                  (uint64_t)num_p * (uint64_t)num_p) : 0;
    const uint64_t nn_total = num_n > 1 ? (tensor_n_same_active ? nn_head_tail + nn_tail_rows :
                                                                  (uint64_t)num_n * (uint64_t)num_n) : 0;
    const uint64_t total = np_total + pp_total + nn_total;
    const uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    uint64_t work = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    for (; work < total; work += stride) {
        if (work < np_total) {
            uint32_t p;
            uint32_t n;
            if (tensor_active && work < np_tensor_n_tail) {
                p = (uint32_t)(work / tensor_n_tail);
                n = tensor_np_num_n + (uint32_t)(work - (uint64_t)p * tensor_n_tail);
            } else if (tensor_active) {
                const uint64_t local = work - np_tensor_n_tail;
                p = tensor_np_num_p + (uint32_t)(local / num_n);
                n = (uint32_t)(local - (uint64_t)(p - tensor_np_num_p) * num_n);
            } else {
                p = (uint32_t)(work / num_n);
                n = (uint32_t)(work - (uint64_t)p * num_n);
            }

            const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)p * vec_length,
                                               n_vecs + (uint64_t)n * vec_length,
                                               vec_length);

            if (p_norm[p] + n_norm[n] + dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
            }
            if (record_dp &&
                bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
                bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
            }
        } else if (work < np_total + pp_total) {
            const uint64_t local = work - np_total;
            uint32_t i;
            uint32_t j;
            if (tensor_p_same_active && local < pp_head_tail) {
                i = (uint32_t)(local / p_tail);
                j = tensor_same_num_p + (uint32_t)(local - (uint64_t)i * p_tail);
            } else if (tensor_p_same_active) {
                const uint64_t tail_local = local - pp_head_tail;
                i = tensor_same_num_p + (uint32_t)(tail_local / num_p);
                j = (uint32_t)(tail_local - (uint64_t)(i - tensor_same_num_p) * num_p);
            } else {
                i = (uint32_t)(local / num_p);
                j = (uint32_t)(local - (uint64_t)i * num_p);
            }
            if (j <= i) continue;

            const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)i * vec_length,
                                               p_vecs + (uint64_t)j * vec_length,
                                               vec_length);

            if (p_norm[i] + p_norm[j] - dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_S, p_ids[i], p_ids[j]);
            }
            if (record_dp &&
                bgj_cuda_p_center_dot(p_norm, p_dot, i, raw_center_dp) +
                bgj_cuda_p_center_dot(p_norm, p_dot, j, raw_center_dp) + dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_SS, p_ids[i], p_ids[j]);
            }
        } else {
            const uint64_t local = work - np_total - pp_total;
            uint32_t i;
            uint32_t j;
            if (tensor_n_same_active && local < nn_head_tail) {
                i = (uint32_t)(local / n_tail);
                j = tensor_same_num_n + (uint32_t)(local - (uint64_t)i * n_tail);
            } else if (tensor_n_same_active) {
                const uint64_t tail_local = local - nn_head_tail;
                i = tensor_same_num_n + (uint32_t)(tail_local / num_n);
                j = (uint32_t)(tail_local - (uint64_t)(i - tensor_same_num_n) * num_n);
            } else {
                i = (uint32_t)(local / num_n);
                j = (uint32_t)(local - (uint64_t)i * num_n);
            }
            if (j <= i) continue;

            const int32_t dp = bgj_cuda_dot_i8(n_vecs + (uint64_t)i * vec_length,
                                               n_vecs + (uint64_t)j * vec_length,
                                               vec_length);

            if (n_norm[i] + n_norm[j] - dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_S, n_ids[i], n_ids[j]);
            }
            if (record_dp &&
                bgj_cuda_n_center_dot(n_norm, n_dot, i, raw_center_dp) +
                bgj_cuda_n_center_dot(n_norm, n_dot, j, raw_center_dp) + dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_AA, n_ids[i], n_ids[j]);
            }
        }
    }
}

__global__ void bgj_cuda_search_bucket_range_kernel(const int8_t *p_vecs,
                                                    const int8_t *n_vecs,
                                                    const uint32_t *p_ids,
                                                    const uint32_t *n_ids,
                                                    const int32_t *p_norm,
                                                    const int32_t *n_norm,
                                                    const int32_t *p_dot,
                                                    const int32_t *n_dot,
                                                    uint32_t num_p,
                                                    uint32_t num_n,
                                                    uint32_t tensor_np_num_p,
                                                    uint32_t tensor_np_num_n,
                                                    uint32_t tensor_same_num_p,
                                                    uint32_t tensor_same_num_n,
                                                    uint32_t vec_length,
                                                    int32_t goal_norm,
                                                    int32_t center_goal_norm,
                                                    int record_dp,
                                                    int raw_center_dp,
                                                    uint64_t work_offset,
                                                    uint64_t work_count,
                                                    bgj_cuda_result_t *results,
                                                    uint32_t result_capacity,
                                                    uint32_t *result_count,
                                                    int *overflow)
{
    const int tensor_active = tensor_np_num_p != 0 && tensor_np_num_n != 0;
    const uint32_t tensor_n_tail = num_n - tensor_np_num_n;
    const uint64_t np_tensor_n_tail = tensor_active ? (uint64_t)tensor_np_num_p * (uint64_t)tensor_n_tail : 0;
    const uint64_t np_tensor_p_tail = tensor_active ? (uint64_t)(num_p - tensor_np_num_p) * (uint64_t)num_n : 0;
    const uint64_t np_total = tensor_active ? np_tensor_n_tail + np_tensor_p_tail :
                                             (uint64_t)num_p * (uint64_t)num_n;
    const int tensor_p_same_active = tensor_same_num_p != 0;
    const int tensor_n_same_active = tensor_same_num_n != 0;
    const uint32_t p_tail = num_p - tensor_same_num_p;
    const uint32_t n_tail = num_n - tensor_same_num_n;
    const uint64_t pp_head_tail = tensor_p_same_active ? (uint64_t)tensor_same_num_p * (uint64_t)p_tail : 0;
    const uint64_t nn_head_tail = tensor_n_same_active ? (uint64_t)tensor_same_num_n * (uint64_t)n_tail : 0;
    const uint64_t pp_tail_rows = tensor_p_same_active ? (uint64_t)p_tail * (uint64_t)num_p : 0;
    const uint64_t nn_tail_rows = tensor_n_same_active ? (uint64_t)n_tail * (uint64_t)num_n : 0;
    const uint64_t pp_total = num_p > 1 ? (tensor_p_same_active ? pp_head_tail + pp_tail_rows :
                                                                  (uint64_t)num_p * (uint64_t)num_p) : 0;
    const uint64_t nn_total = num_n > 1 ? (tensor_n_same_active ? nn_head_tail + nn_tail_rows :
                                                                  (uint64_t)num_n * (uint64_t)num_n) : 0;
    const uint64_t full_total = np_total + pp_total + nn_total;
    if (work_offset >= full_total || work_count == 0) return;
    if (work_count > full_total - work_offset) work_count = full_total - work_offset;

    const uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    uint64_t local = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    for (; local < work_count; local += stride) {
        const uint64_t work = work_offset + local;
        if (work < np_total) {
            uint32_t p;
            uint32_t n;
            if (tensor_active && work < np_tensor_n_tail) {
                p = (uint32_t)(work / tensor_n_tail);
                n = tensor_np_num_n + (uint32_t)(work - (uint64_t)p * tensor_n_tail);
            } else if (tensor_active) {
                const uint64_t tail_local = work - np_tensor_n_tail;
                p = tensor_np_num_p + (uint32_t)(tail_local / num_n);
                n = (uint32_t)(tail_local - (uint64_t)(p - tensor_np_num_p) * num_n);
            } else {
                p = num_n ? (uint32_t)(work / num_n) : 0;
                n = num_n ? (uint32_t)(work - (uint64_t)p * num_n) : 0;
            }
            if (p >= num_p || n >= num_n) continue;

            const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)p * vec_length,
                                               n_vecs + (uint64_t)n * vec_length,
                                               vec_length);

            if (p_norm[p] + n_norm[n] + dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
            }
            if (record_dp &&
                bgj_cuda_p_center_dot(p_norm, p_dot, p, raw_center_dp) +
                bgj_cuda_n_center_dot(n_norm, n_dot, n, raw_center_dp) - dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
            }
        } else if (work < np_total + pp_total) {
            const uint64_t w = work - np_total;
            uint32_t i;
            uint32_t j;
            if (tensor_p_same_active && w < pp_head_tail) {
                i = (uint32_t)(w / p_tail);
                j = tensor_same_num_p + (uint32_t)(w - (uint64_t)i * p_tail);
            } else if (tensor_p_same_active) {
                const uint64_t tail_local = w - pp_head_tail;
                i = tensor_same_num_p + (uint32_t)(tail_local / num_p);
                j = (uint32_t)(tail_local - (uint64_t)(i - tensor_same_num_p) * num_p);
            } else {
                i = num_p ? (uint32_t)(w / num_p) : 0;
                j = num_p ? (uint32_t)(w - (uint64_t)i * num_p) : 0;
            }
            if (i >= num_p || j >= num_p || j <= i) continue;

            const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)i * vec_length,
                                               p_vecs + (uint64_t)j * vec_length,
                                               vec_length);

            if (p_norm[i] + p_norm[j] - dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_S, p_ids[i], p_ids[j]);
            }
            if (record_dp &&
                bgj_cuda_p_center_dot(p_norm, p_dot, i, raw_center_dp) +
                bgj_cuda_p_center_dot(p_norm, p_dot, j, raw_center_dp) + dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_SS, p_ids[i], p_ids[j]);
            }
        } else {
            const uint64_t w = work - np_total - pp_total;
            uint32_t i;
            uint32_t j;
            if (tensor_n_same_active && w < nn_head_tail) {
                i = (uint32_t)(w / n_tail);
                j = tensor_same_num_n + (uint32_t)(w - (uint64_t)i * n_tail);
            } else if (tensor_n_same_active) {
                const uint64_t tail_local = w - nn_head_tail;
                i = tensor_same_num_n + (uint32_t)(tail_local / num_n);
                j = (uint32_t)(tail_local - (uint64_t)(i - tensor_same_num_n) * num_n);
            } else {
                i = num_n ? (uint32_t)(w / num_n) : 0;
                j = num_n ? (uint32_t)(w - (uint64_t)i * num_n) : 0;
            }
            if (i >= num_n || j >= num_n || j <= i) continue;

            const int32_t dp = bgj_cuda_dot_i8(n_vecs + (uint64_t)i * vec_length,
                                               n_vecs + (uint64_t)j * vec_length,
                                               vec_length);

            if (n_norm[i] + n_norm[j] - dp < goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_S, n_ids[i], n_ids[j]);
            }
            if (record_dp &&
                bgj_cuda_n_center_dot(n_norm, n_dot, i, raw_center_dp) +
                bgj_cuda_n_center_dot(n_norm, n_dot, j, raw_center_dp) + dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_AA, n_ids[i], n_ids[j]);
            }
        }
    }
}

__device__ __forceinline__ uint32_t bgj_cuda_search_det_phase(const uint64_t slot,
                                                              const uint64_t phase1_base,
                                                              const uint64_t phase2_base,
                                                              const uint64_t phase3_base,
                                                              const uint64_t phase4_base,
                                                              const uint64_t phase5_base,
                                                              const uint64_t phase6_base,
                                                              uint64_t *segment)
{
    if (slot < phase1_base) {
        *segment = slot;
        return 0u;
    }
    if (slot < phase2_base) {
        *segment = slot - phase1_base;
        return 1u;
    }
    if (slot < phase3_base) {
        *segment = slot - phase2_base;
        return 2u;
    }
    if (slot < phase4_base) {
        *segment = slot - phase3_base;
        return 3u;
    }
    if (slot < phase5_base) {
        *segment = slot - phase4_base;
        return 4u;
    }
    if (slot < phase6_base) {
        *segment = slot - phase5_base;
        return 5u;
    }
    *segment = 0;
    return 6u;
}

__device__ __forceinline__ int bgj_cuda_search_det_candidate(const int8_t *p_vecs,
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
                                                             int32_t center_goal_norm,
                                                             uint32_t phase,
                                                             uint64_t work,
                                                             bgj_cuda_result_t *result)
{
    if (phase == 0u || phase == 1u) {
        if (num_n == 0) return 0;
        const uint32_t p = (uint32_t)(work / num_n);
        const uint32_t n = (uint32_t)(work - (uint64_t)p * num_n);
        if (p >= num_p || n >= num_n) return 0;
        const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)p * vec_length,
                                           n_vecs + (uint64_t)n * vec_length,
                                           vec_length);
        if (phase == 0u) {
            if (p_norm[p] + n_norm[n] + dp >= goal_norm) return 0;
            result->type = BGJ_CUDA_SOL_A;
        } else {
            if (p_dot[p] + n_dot[n] - dp >= center_goal_norm) return 0;
            result->type = BGJ_CUDA_SOL_SA;
        }
        result->x = p_ids[p];
        result->y = n_ids[n];
        return 1;
    }

    if (phase == 2u || phase == 3u) {
        if (num_p == 0) return 0;
        const uint32_t i = (uint32_t)(work / num_p);
        const uint32_t j = (uint32_t)(work - (uint64_t)i * num_p);
        if (i >= num_p || j >= num_p || j <= i) return 0;
        const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)i * vec_length,
                                           p_vecs + (uint64_t)j * vec_length,
                                           vec_length);
        if (phase == 2u) {
            if (p_norm[i] + p_norm[j] - dp >= goal_norm) return 0;
            result->type = BGJ_CUDA_SOL_S;
        } else {
            if (p_dot[i] + p_dot[j] + dp >= center_goal_norm) return 0;
            result->type = BGJ_CUDA_SOL_SS;
        }
        result->x = p_ids[i];
        result->y = p_ids[j];
        return 1;
    }

    if (phase == 4u || phase == 5u) {
        if (num_n == 0) return 0;
        const uint32_t i = (uint32_t)(work / num_n);
        const uint32_t j = (uint32_t)(work - (uint64_t)i * num_n);
        if (i >= num_n || j >= num_n || j <= i) return 0;
        const int32_t dp = bgj_cuda_dot_i8(n_vecs + (uint64_t)i * vec_length,
                                           n_vecs + (uint64_t)j * vec_length,
                                           vec_length);
        if (phase == 4u) {
            if (n_norm[i] + n_norm[j] - dp >= goal_norm) return 0;
            result->type = BGJ_CUDA_SOL_S;
        } else {
            if (n_dot[i] + n_dot[j] + dp >= center_goal_norm) return 0;
            result->type = BGJ_CUDA_SOL_AA;
        }
        result->x = n_ids[i];
        result->y = n_ids[j];
        return 1;
    }

    return 0;
}

__global__ void bgj_cuda_search_det_count_kernel(const int8_t *p_vecs,
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
                                                 int32_t center_goal_norm,
                                                 uint64_t phase1_base,
                                                 uint64_t phase2_base,
                                                 uint64_t phase3_base,
                                                 uint64_t phase4_base,
                                                 uint64_t phase5_base,
                                                 uint64_t phase6_base,
                                                 uint64_t count_slots,
                                                 uint32_t *counts)
{
    for (uint64_t slot = blockIdx.x; slot < count_slots; slot += gridDim.x) {
        uint64_t segment;
        const uint32_t phase =
            bgj_cuda_search_det_phase(slot,
                                      phase1_base,
                                      phase2_base,
                                      phase3_base,
                                      phase4_base,
                                      phase5_base,
                                      phase6_base,
                                      &segment);
        const uint64_t work = segment * BGJ_CUDA_SEARCH_DET_THREADS + threadIdx.x;
        bgj_cuda_result_t result;
        const uint32_t pass =
            bgj_cuda_search_det_candidate(p_vecs,
                                          n_vecs,
                                          p_ids,
                                          n_ids,
                                          p_norm,
                                          n_norm,
                                          p_dot,
                                          n_dot,
                                          num_p,
                                          num_n,
                                          vec_length,
                                          goal_norm,
                                          center_goal_norm,
                                          phase,
                                          work,
                                          &result) ? 1u : 0u;

        __shared__ uint32_t scan[BGJ_CUDA_SEARCH_DET_THREADS];
        scan[threadIdx.x] = pass;
        __syncthreads();

        for (uint32_t step = 1u; step < BGJ_CUDA_SEARCH_DET_THREADS; step <<= 1u) {
            const uint32_t add = threadIdx.x >= step ? scan[threadIdx.x - step] : 0u;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }

        if (threadIdx.x == BGJ_CUDA_SEARCH_DET_THREADS - 1u) {
            counts[slot] = scan[threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void bgj_cuda_search_det_fill_kernel(const int8_t *p_vecs,
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
                                                int32_t center_goal_norm,
                                                uint64_t phase1_base,
                                                uint64_t phase2_base,
                                                uint64_t phase3_base,
                                                uint64_t phase4_base,
                                                uint64_t phase5_base,
                                                uint64_t phase6_base,
                                                uint64_t count_slots,
                                                const uint32_t *offsets,
                                                bgj_cuda_result_t *results,
                                                uint32_t result_capacity,
                                                int *overflow)
{
    for (uint64_t slot = blockIdx.x; slot < count_slots; slot += gridDim.x) {
        uint64_t segment;
        const uint32_t phase =
            bgj_cuda_search_det_phase(slot,
                                      phase1_base,
                                      phase2_base,
                                      phase3_base,
                                      phase4_base,
                                      phase5_base,
                                      phase6_base,
                                      &segment);
        const uint64_t work = segment * BGJ_CUDA_SEARCH_DET_THREADS + threadIdx.x;
        bgj_cuda_result_t result;
        const uint32_t pass =
            bgj_cuda_search_det_candidate(p_vecs,
                                          n_vecs,
                                          p_ids,
                                          n_ids,
                                          p_norm,
                                          n_norm,
                                          p_dot,
                                          n_dot,
                                          num_p,
                                          num_n,
                                          vec_length,
                                          goal_norm,
                                          center_goal_norm,
                                          phase,
                                          work,
                                          &result) ? 1u : 0u;

        __shared__ uint32_t scan[BGJ_CUDA_SEARCH_DET_THREADS];
        scan[threadIdx.x] = pass;
        __syncthreads();

        for (uint32_t step = 1u; step < BGJ_CUDA_SEARCH_DET_THREADS; step <<= 1u) {
            const uint32_t add = threadIdx.x >= step ? scan[threadIdx.x - step] : 0u;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }

        if (pass) {
            const uint32_t out = offsets[slot] + scan[threadIdx.x] - 1u;
            if (out < result_capacity) {
                results[out] = result;
            } else {
                *overflow = 1;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t bgj_cuda_search_det_pair_space(uint64_t slot,
                                                                   uint64_t np_segments,
                                                                   uint64_t pp_segments,
                                                                   uint64_t *segment)
{
    if (slot < np_segments) {
        *segment = slot;
        return 0u;
    }
    slot -= np_segments;
    if (slot < pp_segments) {
        *segment = slot;
        return 1u;
    }
    *segment = slot - pp_segments;
    return 2u;
}

__global__ void bgj_cuda_search_det_pair_count_kernel(const int8_t *p_vecs,
                                                      const int8_t *n_vecs,
                                                      const int32_t *p_norm,
                                                      const int32_t *n_norm,
                                                      const int32_t *p_dot,
                                                      const int32_t *n_dot,
                                                      uint32_t num_p,
                                                      uint32_t num_n,
                                                      uint32_t vec_length,
                                                      int32_t goal_norm,
                                                      int32_t center_goal_norm,
                                                      int record_dp,
                                                      uint64_t np_segments,
                                                      uint64_t pp_segments,
                                                      uint64_t pair_slots,
                                                      uint64_t phase1_base,
                                                      uint64_t phase2_base,
                                                      uint64_t phase3_base,
                                                      uint64_t phase4_base,
                                                      uint64_t phase5_base,
                                                      uint32_t *counts)
{
    for (uint64_t slot = blockIdx.x; slot < pair_slots; slot += gridDim.x) {
        uint64_t segment;
        const uint32_t space = bgj_cuda_search_det_pair_space(slot, np_segments, pp_segments, &segment);
        const uint64_t work = segment * BGJ_CUDA_SEARCH_DET_THREADS + threadIdx.x;
        uint32_t pass0 = 0;
        uint32_t pass1 = 0;

        if (space == 0u) {
            if (num_n) {
                const uint32_t p = (uint32_t)(work / num_n);
                const uint32_t n = (uint32_t)(work - (uint64_t)p * num_n);
                if (p < num_p && n < num_n) {
                    const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)p * vec_length,
                                                       n_vecs + (uint64_t)n * vec_length,
                                                       vec_length);
                    pass0 = p_norm[p] + n_norm[n] + dp < goal_norm ? 1u : 0u;
                    pass1 = record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm ? 1u : 0u;
                }
            }
        } else if (space == 1u) {
            if (num_p) {
                const uint32_t i = (uint32_t)(work / num_p);
                const uint32_t j = (uint32_t)(work - (uint64_t)i * num_p);
                if (i < num_p && j < num_p && j > i) {
                    const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)i * vec_length,
                                                       p_vecs + (uint64_t)j * vec_length,
                                                       vec_length);
                    pass0 = p_norm[i] + p_norm[j] - dp < goal_norm ? 1u : 0u;
                    pass1 = record_dp && p_dot[i] + p_dot[j] + dp < center_goal_norm ? 1u : 0u;
                }
            }
        } else {
            if (num_n) {
                const uint32_t i = (uint32_t)(work / num_n);
                const uint32_t j = (uint32_t)(work - (uint64_t)i * num_n);
                if (i < num_n && j < num_n && j > i) {
                    const int32_t dp = bgj_cuda_dot_i8(n_vecs + (uint64_t)i * vec_length,
                                                       n_vecs + (uint64_t)j * vec_length,
                                                       vec_length);
                    pass0 = n_norm[i] + n_norm[j] - dp < goal_norm ? 1u : 0u;
                    pass1 = record_dp && n_dot[i] + n_dot[j] + dp < center_goal_norm ? 1u : 0u;
                }
            }
        }

        __shared__ uint32_t scan0[BGJ_CUDA_SEARCH_DET_THREADS];
        __shared__ uint32_t scan1[BGJ_CUDA_SEARCH_DET_THREADS];
        scan0[threadIdx.x] = pass0;
        scan1[threadIdx.x] = pass1;
        __syncthreads();

        for (uint32_t step = 1u; step < BGJ_CUDA_SEARCH_DET_THREADS; step <<= 1u) {
            const uint32_t add0 = threadIdx.x >= step ? scan0[threadIdx.x - step] : 0u;
            const uint32_t add1 = threadIdx.x >= step ? scan1[threadIdx.x - step] : 0u;
            __syncthreads();
            scan0[threadIdx.x] += add0;
            scan1[threadIdx.x] += add1;
            __syncthreads();
        }

        if (threadIdx.x == BGJ_CUDA_SEARCH_DET_THREADS - 1u) {
            if (space == 0u) {
                counts[segment] = scan0[threadIdx.x];
                if (record_dp) counts[phase1_base + segment] = scan1[threadIdx.x];
            } else if (space == 1u) {
                counts[phase2_base + segment] = scan0[threadIdx.x];
                if (record_dp) counts[phase3_base + segment] = scan1[threadIdx.x];
            } else {
                counts[phase4_base + segment] = scan0[threadIdx.x];
                if (record_dp) counts[phase5_base + segment] = scan1[threadIdx.x];
            }
        }
        __syncthreads();
    }
}

__global__ void bgj_cuda_search_det_pair_fill_kernel(const int8_t *p_vecs,
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
                                                     int32_t center_goal_norm,
                                                     int record_dp,
                                                     uint64_t np_segments,
                                                     uint64_t pp_segments,
                                                     uint64_t pair_slots,
                                                     uint64_t phase1_base,
                                                     uint64_t phase2_base,
                                                     uint64_t phase3_base,
                                                     uint64_t phase4_base,
                                                     uint64_t phase5_base,
                                                     const uint32_t *offsets,
                                                     bgj_cuda_result_t *results,
                                                     uint32_t result_capacity,
                                                     int *overflow)
{
    for (uint64_t slot = blockIdx.x; slot < pair_slots; slot += gridDim.x) {
        uint64_t segment;
        const uint32_t space = bgj_cuda_search_det_pair_space(slot, np_segments, pp_segments, &segment);
        const uint64_t work = segment * BGJ_CUDA_SEARCH_DET_THREADS + threadIdx.x;
        uint32_t pass0 = 0;
        uint32_t pass1 = 0;
        bgj_cuda_result_t result0;
        bgj_cuda_result_t result1;

        if (space == 0u) {
            if (num_n) {
                const uint32_t p = (uint32_t)(work / num_n);
                const uint32_t n = (uint32_t)(work - (uint64_t)p * num_n);
                if (p < num_p && n < num_n) {
                    const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)p * vec_length,
                                                       n_vecs + (uint64_t)n * vec_length,
                                                       vec_length);
                    pass0 = p_norm[p] + n_norm[n] + dp < goal_norm ? 1u : 0u;
                    pass1 = record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm ? 1u : 0u;
                    result0.type = BGJ_CUDA_SOL_A;
                    result0.x = p_ids[p];
                    result0.y = n_ids[n];
                    result1.type = BGJ_CUDA_SOL_SA;
                    result1.x = p_ids[p];
                    result1.y = n_ids[n];
                }
            }
        } else if (space == 1u) {
            if (num_p) {
                const uint32_t i = (uint32_t)(work / num_p);
                const uint32_t j = (uint32_t)(work - (uint64_t)i * num_p);
                if (i < num_p && j < num_p && j > i) {
                    const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)i * vec_length,
                                                       p_vecs + (uint64_t)j * vec_length,
                                                       vec_length);
                    pass0 = p_norm[i] + p_norm[j] - dp < goal_norm ? 1u : 0u;
                    pass1 = record_dp && p_dot[i] + p_dot[j] + dp < center_goal_norm ? 1u : 0u;
                    result0.type = BGJ_CUDA_SOL_S;
                    result0.x = p_ids[i];
                    result0.y = p_ids[j];
                    result1.type = BGJ_CUDA_SOL_SS;
                    result1.x = p_ids[i];
                    result1.y = p_ids[j];
                }
            }
        } else {
            if (num_n) {
                const uint32_t i = (uint32_t)(work / num_n);
                const uint32_t j = (uint32_t)(work - (uint64_t)i * num_n);
                if (i < num_n && j < num_n && j > i) {
                    const int32_t dp = bgj_cuda_dot_i8(n_vecs + (uint64_t)i * vec_length,
                                                       n_vecs + (uint64_t)j * vec_length,
                                                       vec_length);
                    pass0 = n_norm[i] + n_norm[j] - dp < goal_norm ? 1u : 0u;
                    pass1 = record_dp && n_dot[i] + n_dot[j] + dp < center_goal_norm ? 1u : 0u;
                    result0.type = BGJ_CUDA_SOL_S;
                    result0.x = n_ids[i];
                    result0.y = n_ids[j];
                    result1.type = BGJ_CUDA_SOL_AA;
                    result1.x = n_ids[i];
                    result1.y = n_ids[j];
                }
            }
        }

        __shared__ uint32_t scan0[BGJ_CUDA_SEARCH_DET_THREADS];
        __shared__ uint32_t scan1[BGJ_CUDA_SEARCH_DET_THREADS];
        scan0[threadIdx.x] = pass0;
        scan1[threadIdx.x] = pass1;
        __syncthreads();

        for (uint32_t step = 1u; step < BGJ_CUDA_SEARCH_DET_THREADS; step <<= 1u) {
            const uint32_t add0 = threadIdx.x >= step ? scan0[threadIdx.x - step] : 0u;
            const uint32_t add1 = threadIdx.x >= step ? scan1[threadIdx.x - step] : 0u;
            __syncthreads();
            scan0[threadIdx.x] += add0;
            scan1[threadIdx.x] += add1;
            __syncthreads();
        }

        uint64_t offset0;
        uint64_t offset1;
        if (space == 0u) {
            offset0 = segment;
            offset1 = phase1_base + segment;
        } else if (space == 1u) {
            offset0 = phase2_base + segment;
            offset1 = phase3_base + segment;
        } else {
            offset0 = phase4_base + segment;
            offset1 = phase5_base + segment;
        }

        if (pass0) {
            const uint64_t out = (uint64_t)offsets[offset0] + (uint64_t)scan0[threadIdx.x] - 1ull;
            if (out < (uint64_t)result_capacity) {
                results[(uint32_t)out] = result0;
            } else {
                *overflow = 1;
            }
        }
        if (pass1) {
            const uint64_t out = (uint64_t)offsets[offset1] + (uint64_t)scan1[threadIdx.x] - 1ull;
            if (out < (uint64_t)result_capacity) {
                results[(uint32_t)out] = result1;
            } else {
                *overflow = 1;
            }
        }
        __syncthreads();
    }
}

__global__ void bgj_cuda_pack_pool_vecs_kernel(const int8_t *pool_vecs,
                                               const uint32_t *ids,
                                               uint32_t num_vecs,
                                               uint32_t vec_length,
                                               int8_t *out_vecs)
{
    const uint64_t total = (uint64_t)num_vecs * vec_length;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t pos = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pos < total;
         pos += stride) {
        const uint32_t row = (uint32_t)(pos / vec_length);
        const uint32_t col = (uint32_t)(pos - (uint64_t)row * vec_length);
        out_vecs[pos] = pool_vecs[(uint64_t)ids[row] * vec_length + col];
    }
}

__device__ void bgj_cuda_push_bucket_entry(bgj_cuda_bucket_entry_t *entries,
                                           uint32_t *entry_count,
                                           int *overflow,
                                           uint32_t capacity,
                                           uint32_t bucket,
                                           uint32_t id,
                                           int32_t dot)
{
    const uint32_t out = atomicAdd(entry_count, 1u);
    if (out < capacity) {
        entries[out].bucket = bucket;
        entries[out].id = id;
        entries[out].dot = dot;
    } else {
        *overflow = 1;
    }
}

__global__ void bgj_cuda_bucket_bgj1_kernel(const int8_t *pool_vecs,
                                            const uint32_t *center_ids,
                                            uint32_t num_centers,
                                            const int32_t *vnorm,
                                            uint32_t pool_size,
                                            uint32_t start_id,
                                            uint32_t candidate_count,
                                            uint32_t vec_length,
                                            uint32_t alpha_x2_u16,
                                            bgj_cuda_bucket_entry_t *entries,
                                            uint32_t entry_capacity,
                                            uint32_t *entry_count,
                                            int *overflow)
{
    const uint64_t total = (uint64_t)candidate_count * num_centers;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t pos = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pos < total;
         pos += stride) {
        const uint32_t bucket = (uint32_t)(pos / candidate_count);
        const uint32_t id = start_id + (uint32_t)(pos - (uint64_t)bucket * candidate_count);
        const uint32_t center_id = center_ids[bucket];
        if (center_id >= pool_size) continue;

        const int32_t dot = bgj_cuda_dot_i8(pool_vecs + (uint64_t)center_id * vec_length,
                                            pool_vecs + (uint64_t)id * vec_length,
                                            vec_length);
        const int32_t norm = vnorm[id];
        const int32_t bound = (int32_t)(((int64_t)norm * (int64_t)alpha_x2_u16) >> 16);
        const int32_t abs_dot = dot < 0 ? -dot : dot;
        if (abs_dot > bound) {
            bgj_cuda_push_bucket_entry(entries,
                                       entry_count,
                                       overflow,
                                       entry_capacity,
                                       bucket,
                                       id,
                                       dot);
        }
    }
}

__global__ void bgj_cuda_bucket_bgj1_block_append_kernel(const int8_t *pool_vecs,
                                                         const uint32_t *center_ids,
                                                         uint32_t num_centers,
                                                         const int32_t *vnorm,
                                                         uint32_t pool_size,
                                                         uint32_t start_id,
                                                         uint32_t candidate_count,
                                                         uint32_t vec_length,
                                                         uint32_t alpha_x2_u16,
                                                         bgj_cuda_bucket_entry_t *entries,
                                                         uint32_t entry_capacity,
                                                         uint32_t *entry_count,
                                                         int *overflow)
{
    const uint64_t total = (uint64_t)candidate_count * num_centers;
    const uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;

    for (uint64_t block_work = (uint64_t)blockIdx.x * blockDim.x;
         block_work < total;
         block_work += stride) {
        const uint64_t pos = block_work + threadIdx.x;
        uint32_t bucket = 0;
        uint32_t id = 0;
        int32_t dot = 0;
        int pass = 0;

        if (pos < total) {
            bucket = (uint32_t)(pos / candidate_count);
            id = start_id + (uint32_t)(pos - (uint64_t)bucket * candidate_count);
            const uint32_t center_id = center_ids[bucket];
            if (center_id < pool_size) {
                dot = bgj_cuda_dot_i8(pool_vecs + (uint64_t)center_id * vec_length,
                                      pool_vecs + (uint64_t)id * vec_length,
                                      vec_length);
                const int32_t norm = vnorm[id];
                const int32_t bound = (int32_t)(((int64_t)norm * (int64_t)alpha_x2_u16) >> 16);
                const int32_t abs_dot = dot < 0 ? -dot : dot;
                pass = abs_dot > bound;
            }
        }

        __shared__ uint32_t block_write;
        __shared__ uint32_t block_base;
        if (threadIdx.x == 0) block_write = 0;
        __syncthreads();

        uint32_t local = 0;
        if (pass) {
            local = atomicAdd(&block_write, 1u);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            block_base = block_write ? atomicAdd(entry_count, block_write) : 0;
        }
        __syncthreads();

        if (pass) {
            const uint32_t out = block_base + local;
            if (out < entry_capacity) {
                entries[out].bucket = bucket;
                entries[out].id = id;
                entries[out].dot = dot;
            } else {
                *overflow = 1;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ uint64_t bgj_cuda_bucket_det_count_index(uint32_t bucket,
                                                                    uint32_t sign,
                                                                    uint32_t segment,
                                                                    uint32_t num_segments)
{
    return ((uint64_t)bucket * 2u + (uint64_t)sign) * (uint64_t)num_segments + segment;
}

__device__ __forceinline__ int bgj_cuda_bucket_bgj1_candidate(const int8_t *pool_vecs,
                                                              const uint32_t *center_ids,
                                                              const int32_t *vnorm,
                                                              uint32_t pool_size,
                                                              uint32_t bucket,
                                                              uint32_t id,
                                                              uint32_t vec_length,
                                                              uint32_t alpha_x2_u16,
                                                              int32_t *dot_out)
{
    const uint32_t center_id = center_ids[bucket];
    if (center_id >= pool_size || id >= pool_size) return 0;

    const int32_t dot = bgj_cuda_dot_i8(pool_vecs + (uint64_t)center_id * vec_length,
                                        pool_vecs + (uint64_t)id * vec_length,
                                        vec_length);
    const int32_t norm = vnorm[id];
    const int32_t bound = (int32_t)(((int64_t)norm * (int64_t)alpha_x2_u16) >> 16);
    const int32_t abs_dot = dot < 0 ? -dot : dot;
    *dot_out = dot;
    return abs_dot > bound;
}

__global__ void bgj_cuda_bucket_bgj1_count_kernel(const int8_t *pool_vecs,
                                                  const uint32_t *center_ids,
                                                  uint32_t num_centers,
                                                  const int32_t *vnorm,
                                                  uint32_t pool_size,
                                                  uint32_t num_segments,
                                                  uint32_t vec_length,
                                                  uint32_t alpha_x2_u16,
                                                  uint32_t *counts)
{
    const uint64_t total_blocks = (uint64_t)num_centers * (uint64_t)num_segments;
    for (uint64_t linear = blockIdx.x; linear < total_blocks; linear += gridDim.x) {
        const uint32_t bucket = (uint32_t)(linear / num_segments);
        const uint32_t segment = (uint32_t)(linear - (uint64_t)bucket * num_segments);
        const uint32_t id = segment * BGJ_CUDA_BUCKET_DET_THREADS + threadIdx.x;

        __shared__ uint32_t pos_count;
        __shared__ uint32_t neg_count;
        if (threadIdx.x == 0) {
            pos_count = 0;
            neg_count = 0;
        }
        __syncthreads();

        int32_t dot = 0;
        if (id < pool_size &&
            bgj_cuda_bucket_bgj1_candidate(pool_vecs,
                                           center_ids,
                                           vnorm,
                                           pool_size,
                                           bucket,
                                           id,
                                           vec_length,
                                           alpha_x2_u16,
                                           &dot)) {
            if (dot > 0) {
                atomicAdd(&pos_count, 1u);
            } else {
                atomicAdd(&neg_count, 1u);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            counts[bgj_cuda_bucket_det_count_index(bucket, 0u, segment, num_segments)] = pos_count;
            counts[bgj_cuda_bucket_det_count_index(bucket, 1u, segment, num_segments)] = neg_count;
        }
        __syncthreads();
    }
}

__global__ void bgj_cuda_bucket_bgj1_fill_kernel(const int8_t *pool_vecs,
                                                 const uint32_t *center_ids,
                                                 uint32_t num_centers,
                                                 const int32_t *vnorm,
                                                 uint32_t pool_size,
                                                 uint32_t num_segments,
                                                 uint32_t vec_length,
                                                 uint32_t alpha_x2_u16,
                                                 const uint32_t *offsets,
                                                 bgj_cuda_bucket_entry_t *entries,
                                                 uint32_t entry_capacity,
                                                 int *overflow)
{
    const uint64_t total_blocks = (uint64_t)num_centers * (uint64_t)num_segments;
    for (uint64_t linear = blockIdx.x; linear < total_blocks; linear += gridDim.x) {
        const uint32_t bucket = (uint32_t)(linear / num_segments);
        const uint32_t segment = (uint32_t)(linear - (uint64_t)bucket * num_segments);
        const uint32_t id = segment * BGJ_CUDA_BUCKET_DET_THREADS + threadIdx.x;

        __shared__ uint32_t pos_scan[BGJ_CUDA_BUCKET_DET_THREADS];
        __shared__ uint32_t neg_scan[BGJ_CUDA_BUCKET_DET_THREADS];

        int32_t dot = 0;
        const int pass =
            id < pool_size &&
            bgj_cuda_bucket_bgj1_candidate(pool_vecs,
                                           center_ids,
                                           vnorm,
                                           pool_size,
                                           bucket,
                                           id,
                                           vec_length,
                                           alpha_x2_u16,
                                           &dot);
        pos_scan[threadIdx.x] = (pass && dot > 0) ? 1u : 0u;
        neg_scan[threadIdx.x] = (pass && dot <= 0) ? 1u : 0u;
        __syncthreads();

        for (uint32_t step = 1u; step < BGJ_CUDA_BUCKET_DET_THREADS; step <<= 1u) {
            const uint32_t pos_add = threadIdx.x >= step ? pos_scan[threadIdx.x - step] : 0u;
            const uint32_t neg_add = threadIdx.x >= step ? neg_scan[threadIdx.x - step] : 0u;
            __syncthreads();
            pos_scan[threadIdx.x] += pos_add;
            neg_scan[threadIdx.x] += neg_add;
            __syncthreads();
        }

        if (pass) {
            const uint32_t sign = dot > 0 ? 0u : 1u;
            const uint32_t local_rank = sign == 0u ? pos_scan[threadIdx.x] - 1u
                                                   : neg_scan[threadIdx.x] - 1u;
            const uint32_t out =
                offsets[bgj_cuda_bucket_det_count_index(bucket, sign, segment, num_segments)] +
                local_rank;
            if (out < entry_capacity) {
                entries[out].bucket = bucket;
                entries[out].id = id;
                entries[out].dot = dot;
                entries[out].reserved = 0;
            } else {
                *overflow = 1;
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(BGJ_CUDA_TENSOR_THREADS_PER_BLOCK, 2)
void bgj_cuda_bucket_bgj1_tensor_kernel(const int8_t *center_vecs,
                                        const int8_t *pool_vecs,
                                        uint32_t num_center_tiles,
                                        uint32_t num_pool_tiles,
                                        const int32_t *vnorm,
                                        uint32_t vec_length,
                                        uint32_t alpha_x2_u16,
                                        bgj_cuda_bucket_entry_t *entries,
                                        uint32_t entry_capacity,
                                        uint32_t *entry_count,
                                        int *overflow)
{
    const uint32_t warp_id = threadIdx.x / BGJ_CUDA_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % BGJ_CUDA_WARP_SIZE;
    const uint32_t tile = blockIdx.x * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK + warp_id;
    const uint32_t total_tiles = num_center_tiles * num_pool_tiles;
    if (tile >= total_tiles) return;

    const uint32_t center_tile = tile / num_pool_tiles;
    const uint32_t pool_tile = tile - center_tile * num_pool_tiles;
    const uint32_t bucket_base = center_tile * 16u;
    const uint32_t id_base = pool_tile * 16u;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k = 0; k < vec_length; k += 16) {
        wmma::load_matrix_sync(a_frag,
                               (const signed char *)(center_vecs + (uint64_t)bucket_base * vec_length + k),
                               vec_length);
        wmma::load_matrix_sync(b_frag,
                               (const signed char *)(pool_vecs + (uint64_t)id_base * vec_length + k),
                               vec_length);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    #pragma unroll
    for (uint32_t element = 0; element < c_frag.num_elements; element++) {
        uint32_t row;
        uint32_t col;
        bgj_cuda_tensor_accumulator_coord(lane_id, element, &row, &col);
        const uint32_t bucket = bucket_base + row;
        const uint32_t id = id_base + col;
        const int32_t dot = c_frag.x[element];
        const int32_t norm = vnorm[id];
        const int32_t bound = (int32_t)(((int64_t)norm * (int64_t)alpha_x2_u16) >> 16);
        const int32_t abs_dot = dot < 0 ? -dot : dot;
        if (abs_dot > bound) {
            bgj_cuda_push_bucket_entry(entries,
                                       entry_count,
                                       overflow,
                                       entry_capacity,
                                       bucket,
                                       id,
                                       dot);
        }
    }
}

#define CUDA_TRY(call)                                      \
    do {                                                    \
        cudaError_t err__ = (call);                         \
        if (err__ != cudaSuccess) {                         \
            set_cuda_error(#call, err__);                   \
            goto fail;                                      \
        }                                                   \
    } while (0)

#define CUBLAS_TRY(call)                                    \
    do {                                                    \
        cublasStatus_t status__ = (call);                   \
        if (status__ != CUBLAS_STATUS_SUCCESS) {            \
            set_cublas_error(#call, status__);              \
            goto fail;                                      \
        }                                                   \
    } while (0)

static int ensure_cuda_capacity(void **ptr, size_t *capacity, size_t requested)
{
    if (requested <= *capacity) return 1;
    cudaError_t err = cudaFree(*ptr);
    if (err != cudaSuccess) {
        set_cuda_error("cudaFree", err);
        return 0;
    }
    *ptr = NULL;
    *capacity = 0;
    err = cudaMalloc(ptr, requested);
    if (err != cudaSuccess) {
        set_cuda_error("cudaMalloc", err);
        return 0;
    }
    *capacity = requested;
    return 1;
}

static int ensure_cuda_host_capacity(void **ptr, size_t *capacity, size_t requested)
{
    if (requested <= *capacity) return 1;
    cudaError_t err = cudaFreeHost(*ptr);
    if (err != cudaSuccess) {
        set_cuda_error("cudaFreeHost", err);
        return 0;
    }
    *ptr = NULL;
    *capacity = 0;
    err = cudaHostAlloc(ptr, requested, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        set_cuda_error("cudaHostAlloc", err);
        return 0;
    }
    *capacity = requested;
    return 1;
}

static void bgj_cuda_get_submitted_counts(bgj_cuda_raw_scratch_t *scratch,
                                          uint32_t *submitted_result_count,
                                          int *submitted_overflow)
{
    if (submitted_result_count) {
        *submitted_result_count = scratch->host_result_count ?
                                  *scratch->host_result_count : 0;
    }
    if (submitted_overflow) {
        *submitted_overflow = scratch->host_overflow ?
                              *scratch->host_overflow : 1;
    }
}

extern "C" void *bgj_cuda_alloc_pinned_host_raw(size_t bytes)
{
    if (bytes == 0) return NULL;
    void *ptr = NULL;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        set_cuda_error("cudaHostAlloc", err);
        return NULL;
    }
    set_plain_error("no CUDA error");
    return ptr;
}

extern "C" void bgj_cuda_free_pinned_host_raw(void *ptr)
{
    if (!ptr) return;
    cudaFreeHost(ptr);
}

static uint64_t bgj_cuda_hash_bytes(const void *data, size_t size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ULL;
    }
    hash ^= (uint64_t)size;
    hash *= 1099511628211ULL;
    return hash;
}

struct bgj_cuda_shared_pool_cache_t {
    int8_t *vecs;
    size_t capacity;
    const int8_t *host_key;
    uint64_t epoch;
    uint32_t pool_size;
    uint32_t vec_length;
    int device;
    pthread_mutex_t lock;

    bgj_cuda_shared_pool_cache_t()
        : vecs(NULL),
          capacity(0),
          host_key(NULL),
          epoch(0),
          pool_size(0),
          vec_length(0),
          device(-1)
    {
        pthread_mutex_init(&lock, NULL);
    }

    ~bgj_cuda_shared_pool_cache_t()
    {
        if (device >= 0) cudaSetDevice(device);
        cudaFree(vecs);
        pthread_mutex_destroy(&lock);
    }
};

static bgj_cuda_shared_pool_cache_t bgj_cuda_shared_pool_cache[BGJ_CUDA_MAX_EXEC_DEVICES];

static bgj_cuda_shared_pool_cache_t *bgj_cuda_current_shared_pool_cache()
{
    int device = 0;
    if (!bgj_cuda_current_device(&device)) return NULL;
    if (device < 0 || device >= BGJ_CUDA_MAX_EXEC_DEVICES) {
        set_plain_error("CUDA device index exceeds BGJ_CUDA_MAX_EXEC_DEVICES");
        return NULL;
    }
    bgj_cuda_shared_pool_cache_t *cache = &bgj_cuda_shared_pool_cache[device];
    if (cache->device < 0) cache->device = device;
    return cache;
}

static int bgj_cuda_prepare_shared_pool_cache(const int8_t *pool_vecs_host,
                                              uint64_t pool_epoch,
                                              uint32_t pool_size,
                                              uint32_t vec_length,
                                              size_t pool_vec_bytes,
                                              cudaStream_t stream,
                                              int8_t **pool_vecs_device)
{
    bgj_cuda_shared_pool_cache_t *cache = bgj_cuda_current_shared_pool_cache();
    if (!cache) return 0;
    pthread_mutex_lock(&cache->lock);

    if (cache->host_key != pool_vecs_host ||
        cache->epoch != pool_epoch ||
        cache->pool_size != pool_size ||
        cache->vec_length != vec_length) {
        if (!ensure_cuda_capacity((void **)&cache->vecs, &cache->capacity, pool_vec_bytes)) {
            pthread_mutex_unlock(&cache->lock);
            return 0;
        }
        if (pool_vec_bytes) {
            cudaError_t err = cudaMemcpyAsync(cache->vecs,
                                              pool_vecs_host,
                                              pool_vec_bytes,
                                              cudaMemcpyHostToDevice,
                                              stream);
            if (err != cudaSuccess) {
                set_cuda_error("cudaMemcpyAsync shared pool", err);
                pthread_mutex_unlock(&cache->lock);
                return 0;
            }
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                set_cuda_error("cudaStreamSynchronize shared pool", err);
                pthread_mutex_unlock(&cache->lock);
                return 0;
            }
        }
        cache->host_key = pool_vecs_host;
        cache->epoch = pool_epoch;
        cache->pool_size = pool_size;
        cache->vec_length = vec_length;
    }

    *pool_vecs_device = cache->vecs;
    pthread_mutex_unlock(&cache->lock);
    return 1;
}

static int bgj_cuda_prepare_stream(bgj_cuda_raw_scratch_t *scratch)
{
    int device = 0;
    if (!bgj_cuda_current_device(&device)) return 0;
    if (scratch->device >= 0 && scratch->device != device) {
        scratch->release();
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            set_cuda_error("cudaSetDevice prepare stream", err);
            return 0;
        }
    }
    if (scratch->stream_ready) return 1;
    cudaError_t err = cudaStreamCreateWithFlags(&scratch->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        set_cuda_error("cudaStreamCreateWithFlags", err);
        return 0;
    }
    scratch->device = device;
    scratch->stream_ready = 1;
    return 1;
}

static int bgj_cuda_prepare_lsh_lift_stream(bgj_cuda_lsh_lift_scratch_t *scratch)
{
    int device = 0;
    if (!bgj_cuda_current_device(&device)) return 0;
    if (scratch->device >= 0 && scratch->device != device) {
        scratch->release();
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            set_cuda_error("cudaSetDevice prepare LSH lift stream", err);
            return 0;
        }
    }
    if (scratch->stream_ready) return 1;
    cudaError_t err = cudaStreamCreateWithFlags(&scratch->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        set_cuda_error("cudaStreamCreateWithFlags LSH lift", err);
        return 0;
    }
    scratch->device = device;
    scratch->stream_ready = 1;
    return 1;
}

static int bgj_cuda_prepare_dot_event(bgj_cuda_raw_scratch_t *scratch)
{
    if (scratch->dot_event_ready) return 1;
    cudaError_t err = cudaEventCreateWithFlags(&scratch->dot_copy_done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventCreateWithFlags dot copy", err);
        return 0;
    }
    scratch->dot_event_ready = 1;
    return 1;
}

static int bgj_cuda_prepare_search_profile_events(bgj_cuda_raw_scratch_t *scratch)
{
    if (scratch->profile_events_ready) return 1;
    for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
        cudaError_t err = cudaEventCreate(&scratch->profile_start[i]);
        if (err != cudaSuccess) {
            set_cuda_error("cudaEventCreate search profile start", err);
            for (int j = 0; j < i; j++) {
                cudaEventDestroy(scratch->profile_start[j]);
                cudaEventDestroy(scratch->profile_stop[j]);
                scratch->profile_start[j] = NULL;
                scratch->profile_stop[j] = NULL;
            }
            return 0;
        }
        err = cudaEventCreate(&scratch->profile_stop[i]);
        if (err != cudaSuccess) {
            set_cuda_error("cudaEventCreate search profile stop", err);
            cudaEventDestroy(scratch->profile_start[i]);
            scratch->profile_start[i] = NULL;
            for (int j = 0; j < i; j++) {
                cudaEventDestroy(scratch->profile_start[j]);
                cudaEventDestroy(scratch->profile_stop[j]);
                scratch->profile_start[j] = NULL;
                scratch->profile_stop[j] = NULL;
            }
            return 0;
        }
    }
    scratch->profile_events_ready = 1;
    return 1;
}

static int bgj_cuda_search_profile_reset(bgj_cuda_raw_scratch_t *scratch)
{
    scratch->phase_profile_active =
        bgj_cuda_phase_profile_requested() || bgj_cuda_split_profile_requested();
    for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
        scratch->profile_ran[i] = 0;
        scratch->last_phase_profile.sec[i] = 0.0;
    }
    if (!scratch->phase_profile_active) return 1;
    return bgj_cuda_prepare_search_profile_events(scratch);
}

static int bgj_cuda_search_profile_begin(bgj_cuda_raw_scratch_t *scratch,
                                         int phase)
{
    if (!scratch->phase_profile_active) return 1;
    cudaError_t err = cudaEventRecord(scratch->profile_start[phase], scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventRecord search profile begin", err);
        return 0;
    }
    return 1;
}

static int bgj_cuda_search_profile_end(bgj_cuda_raw_scratch_t *scratch,
                                       int phase)
{
    if (!scratch->phase_profile_active) return 1;
    cudaError_t err = cudaEventRecord(scratch->profile_stop[phase], scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventRecord search profile end", err);
        return 0;
    }
    scratch->profile_ran[phase] = 1;
    return 1;
}

static int bgj_cuda_search_profile_collect(bgj_cuda_raw_scratch_t *scratch)
{
    if (!scratch->phase_profile_active) return 1;
    for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
        if (!scratch->profile_ran[i]) continue;
        float ms = 0.0f;
        cudaError_t err =
            cudaEventElapsedTime(&ms, scratch->profile_start[i], scratch->profile_stop[i]);
        if (err != cudaSuccess) {
            set_cuda_error("cudaEventElapsedTime search profile", err);
            return 0;
        }
        scratch->last_phase_profile.sec[i] = (double)ms * 0.001;
    }
    return 1;
}

static void bgj_cuda_search_phase_profile_add(bgj_cuda_search_phase_profile_t *dst,
                                              const bgj_cuda_search_phase_profile_t *src)
{
    if (!dst || !src) return;
    for (int i = 0; i < BGJ_CUDA_SEARCH_PHASE_COUNT; i++) {
        dst->sec[i] += src->sec[i];
    }
}

static int bgj_cuda_prepare_bucket_stream(bgj_cuda_bucket_scratch_t *scratch)
{
    int device = 0;
    if (!bgj_cuda_current_device(&device)) return 0;
    if (scratch->device >= 0 && scratch->device != device) {
        scratch->release();
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            set_cuda_error("cudaSetDevice prepare bucket stream", err);
            return 0;
        }
    }
    if (scratch->stream_ready) return 1;
    cudaError_t err = cudaStreamCreateWithFlags(&scratch->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        set_cuda_error("cudaStreamCreateWithFlags bucket", err);
        return 0;
    }
    scratch->device = device;
    scratch->stream_ready = 1;
    return 1;
}

#define CUDA_ENSURE(ptr, capacity, requested)                         \
    do {                                                              \
        if (!ensure_cuda_capacity((void **)&(ptr), &(capacity), requested)) goto fail; \
    } while (0)

static int bgj_cuda_finish_submitted_bucket(bgj_cuda_raw_scratch_t *scratch,
                                            bgj_cuda_result_t *results,
                                            uint32_t result_capacity,
                                            uint32_t submitted_result_count,
                                            int submitted_overflow,
                                            uint32_t *result_count,
                                            int *overflow);

extern "C" uint64_t bgj_cuda_lsh_total_tile_slots(uint32_t mbound)
{
    const uint32_t num_tiles = (mbound + BGJ_CUDA_LSH_TILE - 1u) / BGJ_CUDA_LSH_TILE;
    return (uint64_t)num_tiles * (uint64_t)(num_tiles + 1u) / 2u;
}

static int bgj_cuda_lsh_multi_gpu_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_LSH_MULTI_GPU",
                             bgj_cuda_stable_multi_gpu_requested());
}

static uint64_t bgj_cuda_lsh_multi_gpu_min_slots()
{
    return bgj_cuda_env_u64("BGJ_CUDA_LSH_MULTI_GPU_MIN_SLOTS",
                            1024ULL * 1024ULL);
}

static uint64_t bgj_cuda_lsh_stream_default_chunk_slots()
{
    return bgj_cuda_env_u64("BGJ_CUDA_LSH_STREAM_CHUNK_SLOTS", 65536ull);
}

static int bgj_cuda_lsh_pinned_results_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_LSH_PINNED_RESULTS", 0);
}

static int bgj_cuda_lsh_upload_signatures(bgj_cuda_raw_scratch_t *scratch,
                                          const uint8_t *sh,
                                          uint32_t mbound,
                                          uint32_t shsize)
{
    if (!scratch) return 0;
    if (!bgj_cuda_prepare_stream(scratch)) return 0;

    cudaStream_t stream = scratch->stream;
    const size_t sh_bytes = (size_t)mbound * (size_t)shsize;
    CUDA_ENSURE(scratch->p_vecs, scratch->p_vec_capacity, sh_bytes);
    CUDA_TRY(cudaMemcpyAsync(scratch->p_vecs,
                             sh,
                             sh_bytes,
                             cudaMemcpyHostToDevice,
                             stream));
    return 1;

fail:
    return 0;
}

static int bgj_cuda_complete_lsh_pinned_results(bgj_cuda_raw_scratch_t *scratch)
{
    if (!scratch) return 0;
    if (scratch->pending_result_dst && scratch->pending_result_count) {
        memcpy(scratch->pending_result_dst,
               scratch->host_results,
               (size_t)scratch->pending_result_count * sizeof(bgj_cuda_result_t));
    }
    scratch->pending_result_dst = NULL;
    scratch->pending_result_count = 0;
    return 1;
}

static int bgj_cuda_finish_submitted_lsh_bucket(bgj_cuda_raw_scratch_t *scratch,
                                                bgj_cuda_result_t *results,
                                                uint32_t result_capacity,
                                                uint32_t submitted_result_count,
                                                int submitted_overflow,
                                                uint32_t *result_count,
                                                int *overflow)
{
    if (!bgj_cuda_lsh_pinned_results_requested()) {
        return bgj_cuda_finish_submitted_bucket(scratch,
                                                results,
                                                result_capacity,
                                                submitted_result_count,
                                                submitted_overflow,
                                                result_count,
                                                overflow);
    }
    cudaStream_t stream = scratch->stream;

    scratch->pending_result_dst = NULL;
    scratch->pending_result_count = 0;
    *overflow = submitted_overflow || (submitted_result_count > result_capacity);
    *result_count = submitted_result_count > result_capacity ? result_capacity : submitted_result_count;
    if (*result_count) {
        const size_t bytes = (size_t)(*result_count) * sizeof(bgj_cuda_result_t);
        if (!ensure_cuda_host_capacity((void **)&scratch->host_results,
                                       &scratch->host_result_capacity,
                                       bytes)) {
            goto fail;
        }
        CUDA_TRY(cudaMemcpyAsync(scratch->host_results,
                                 scratch->results,
                                 bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream));
        scratch->pending_result_dst = results;
        scratch->pending_result_count = *result_count;
    }

    set_plain_error("no CUDA error");
    return 1;

fail:
    scratch->pending_result_dst = NULL;
    scratch->pending_result_count = 0;
    return 0;
}

static int bgj_cuda_lsh_search_range_submit(bgj_cuda_raw_scratch_t *scratch,
                                            const uint8_t *sh,
                                            uint32_t mbound,
                                            uint32_t shsize,
                                            int32_t threshold,
                                            uint64_t tile_slot_begin,
                                            uint64_t tile_slot_count,
                                            uint32_t result_capacity,
                                            int copy_signatures)
{
    if (!scratch) return 0;
    if (!bgj_cuda_prepare_stream(scratch)) return 0;

    cudaStream_t stream = scratch->stream;
    const size_t sh_bytes = (size_t)mbound * (size_t)shsize;
    CUDA_ENSURE(scratch->p_vecs, scratch->p_vec_capacity, sh_bytes);
    CUDA_ENSURE(scratch->results, scratch->result_capacity, (size_t)result_capacity * sizeof(bgj_cuda_result_t));
    CUDA_ENSURE(scratch->result_count, scratch->result_count_capacity, sizeof(uint32_t));
    CUDA_ENSURE(scratch->overflow, scratch->overflow_capacity, sizeof(int));
    if (!ensure_cuda_host_capacity((void **)&scratch->host_result_count,
                                   &scratch->host_result_count_capacity,
                                   sizeof(uint32_t)) ||
        !ensure_cuda_host_capacity((void **)&scratch->host_overflow,
                                   &scratch->host_overflow_capacity,
                                   sizeof(int))) {
        goto fail;
    }
    *scratch->host_result_count = 0;
    *scratch->host_overflow = 0;

    if (copy_signatures) {
        CUDA_TRY(cudaMemcpyAsync(scratch->p_vecs,
                                 sh,
                                 sh_bytes,
                                 cudaMemcpyHostToDevice,
                                 stream));
    }
    CUDA_TRY(cudaMemsetAsync(scratch->result_count, 0, sizeof(uint32_t), stream));
    CUDA_TRY(cudaMemsetAsync(scratch->overflow, 0, sizeof(int), stream));

    {
        uint32_t grid = tile_slot_count > 65535ull ? 65535u : (uint32_t)tile_slot_count;
        if (grid == 0) grid = 1;
        bgj_cuda_lsh_search_kernel<<<grid,
                                      BGJ_CUDA_LSH_THREADS,
                                      0,
                                      stream>>>((const uint64_t *)scratch->p_vecs,
                                                mbound,
                                                threshold,
                                                tile_slot_begin,
                                                tile_slot_count,
                                                scratch->results,
                                                result_capacity,
                                                scratch->result_count,
                                                scratch->overflow);
        CUDA_TRY(cudaGetLastError());
    }

    CUDA_TRY(cudaMemcpyAsync(scratch->host_result_count,
                             scratch->result_count,
                             sizeof(uint32_t),
                             cudaMemcpyDeviceToHost,
                             stream));
    CUDA_TRY(cudaMemcpyAsync(scratch->host_overflow,
                             scratch->overflow,
                             sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));

    set_plain_error("no CUDA error");
    return 1;

fail:
    return 0;
}

static int bgj_cuda_lsh_search_split_raw(const uint8_t *sh,
                                         uint32_t mbound,
                                         uint32_t shsize,
                                         int32_t threshold,
                                         bgj_cuda_result_t *results,
                                         uint32_t result_capacity,
                                         uint32_t *result_count,
                                         int *overflow,
                                         uint64_t total_tile_slots)
{
    if (!bgj_cuda_select_thread_device()) return 0;

    int selected_device = -1;
    bgj_cuda_current_device(&selected_device);

    int devices[BGJ_CUDA_MAX_EXEC_DEVICES];
    int device_count =
        bgj_cuda_copy_execution_devices(devices, BGJ_CUDA_MAX_EXEC_DEVICES);
    if (device_count <= 1) return 0;
    if ((uint64_t)device_count > total_tile_slots) {
        device_count = (int)total_tile_slots;
    }
    if (device_count <= 1) return 0;

    uint64_t tile_begin[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint64_t tile_count[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint32_t submitted_counts[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint32_t copied_counts[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    int submitted_overflows[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    int copied_overflows[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint32_t out_count = 0;
    int out_overflow = 0;
    uint64_t submitted_total = 0;
    int submitted_any_overflow = 0;
    uint32_t submitted = 0;
    const int split_profile = bgj_cuda_split_profile_requested();
    const double total_t0 = bgj_cuda_wall_time();
    double submit_sec = 0.0;
    double wait_sec = 0.0;
    double copy_sec = 0.0;
    double submit_t0 = 0.0;
    double wait_t0 = 0.0;
    double copy_t0 = 0.0;

    for (int i = 0; i < device_count; i++) {
        const uint64_t begin =
            ((uint64_t)i * total_tile_slots) / (uint64_t)device_count;
        const uint64_t end =
            ((uint64_t)(i + 1) * total_tile_slots) / (uint64_t)device_count;
        tile_begin[i] = begin;
        tile_count[i] = end > begin ? end - begin : 0ull;
    }

    submit_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!tile_count[i]) continue;
        if (!bgj_cuda_set_current_device(devices[i])) goto fail_sync;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch) {
            set_plain_error("out of host memory");
            goto fail_sync;
        }
        if (!bgj_cuda_lsh_search_range_submit(scratch,
                                              sh,
                                              mbound,
                                              shsize,
                                              threshold,
                                              tile_begin[i],
                                              tile_count[i],
                                              result_capacity,
                                              1)) {
            goto fail_sync;
        }
        submitted++;
    }
    if (!submitted) return 0;
    submit_sec = bgj_cuda_wall_time() - submit_t0;

    wait_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!tile_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize LSH split counts", err);
            goto fail_sync;
        }
        bgj_cuda_get_submitted_counts(scratch,
                                      &submitted_counts[i],
                                      &submitted_overflows[i]);
        submitted_total += submitted_counts[i];
        if (submitted_overflows[i]) submitted_any_overflow = 1;
    }
    wait_sec = bgj_cuda_wall_time() - wait_t0;

    copy_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!tile_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        const uint32_t remaining =
            out_count < result_capacity ? result_capacity - out_count : 0u;
        if (!bgj_cuda_finish_submitted_lsh_bucket(scratch,
                                                  results + out_count,
                                                  remaining,
                                                  submitted_counts[i],
                                                  submitted_overflows[i],
                                                  &copied_counts[i],
                                                  &copied_overflows[i])) {
            goto fail_sync;
        }
        out_count += copied_counts[i];
        if (copied_overflows[i]) out_overflow = 1;
    }

    for (int i = 0; i < device_count; i++) {
        if (!tile_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize LSH split results", err);
            goto fail_sync;
        }
        if (!bgj_cuda_complete_lsh_pinned_results(scratch)) goto fail_sync;
    }
    copy_sec = bgj_cuda_wall_time() - copy_t0;

    *result_count = out_count;
    *overflow =
        out_overflow || submitted_any_overflow ||
        submitted_total > (uint64_t)result_capacity;
    set_plain_error("no CUDA error");
    if (split_profile) {
        fprintf(stderr,
                "cuda_lsh_split_profile: devices=%d mbound=%u threshold=%d "
                "slots=%lu results=%u overflow=%d submit=%.6fs wait=%.6fs "
                "copy=%.6fs total=%.6fs\n",
                device_count,
                mbound,
                threshold,
                (unsigned long)total_tile_slots,
                out_count,
                *overflow,
                submit_sec,
                wait_sec,
                copy_sec,
                bgj_cuda_wall_time() - total_t0);
        for (int i = 0; i < device_count; i++) {
            if (!tile_count[i]) continue;
            fprintf(stderr,
                    "cuda_lsh_split_device_profile: slot=%d/%d device=%d "
                    "tile_begin=%lu tile_count=%lu submitted=%u copied=%u "
                    "submitted_overflow=%d copied_overflow=%d\n",
                    i,
                    device_count,
                    devices[i],
                    (unsigned long)tile_begin[i],
                    (unsigned long)tile_count[i],
                    submitted_counts[i],
                    copied_counts[i],
                    submitted_overflows[i],
                    copied_overflows[i]);
        }
        fflush(stderr);
    }
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 1;

fail_sync:
    for (int i = 0; i < device_count; i++) {
        if (!tile_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (scratch && scratch->stream_ready) {
            if (scratch->device >= 0) cudaSetDevice(scratch->device);
            cudaStreamSynchronize(scratch->stream);
        }
    }
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 0;
}

extern "C" int bgj_cuda_lsh_search_stream_raw(const uint8_t *sh,
                                               uint32_t mbound,
                                               uint32_t shsize,
                                               int32_t threshold,
                                               uint64_t chunk_tile_slots,
                                               bgj_cuda_result_t *results,
                                               uint32_t result_capacity,
                                               uint64_t *total_result_count,
                                               int *overflow,
                                               bgj_cuda_lsh_result_callback_t callback,
                                               void *callback_ctx)
{
    if (total_result_count) *total_result_count = 0;
    if (overflow) *overflow = 0;
    if (!callback) {
        set_plain_error("missing CUDA LSH stream callback");
        return 0;
    }
    if (!results || result_capacity == 0u) {
        set_plain_error("invalid CUDA LSH stream result buffer");
        return 0;
    }
    if (shsize != 64u) {
        set_plain_error("CUDA LSH stream only supports 64-byte signatures");
        return 0;
    }
    if (mbound < 2u) {
        set_plain_error("no CUDA error");
        return 1;
    }
    if (!bgj_cuda_select_thread_device()) return 0;

    int selected_device = -1;
    bgj_cuda_current_device(&selected_device);

    const uint64_t total_tile_slots = bgj_cuda_lsh_total_tile_slots(mbound);
    if (total_tile_slots == 0u) {
        set_plain_error("no CUDA error");
        if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
        return 1;
    }
    if (chunk_tile_slots == 0u) {
        chunk_tile_slots = bgj_cuda_lsh_stream_default_chunk_slots();
    }
    if (chunk_tile_slots == 0u) chunk_tile_slots = total_tile_slots;

    int devices[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    int device_count = 1;
    devices[0] = selected_device >= 0 ? selected_device : 0;
    if (bgj_cuda_lsh_multi_gpu_requested() &&
        total_tile_slots >= bgj_cuda_lsh_multi_gpu_min_slots()) {
        device_count = bgj_cuda_copy_execution_devices(devices, BGJ_CUDA_MAX_EXEC_DEVICES);
        if (device_count <= 0) {
            devices[0] = selected_device >= 0 ? selected_device : 0;
            device_count = 1;
        }
    }
    if ((uint64_t)device_count > total_tile_slots) device_count = (int)total_tile_slots;
    if (device_count <= 0) device_count = 1;

    uint64_t tile_begin[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint64_t tile_count[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint32_t submitted_counts[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint32_t copied_counts[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    int submitted_overflows[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    int copied_overflows[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint64_t stream_results = 0;
    uint64_t next_tile = 0;
    uint64_t waves = 0;
    uint64_t chunks = 0;
    const int split_profile = bgj_cuda_split_profile_requested();
    const double total_t0 = bgj_cuda_wall_time();
    double submit_sec = 0.0;
    double wait_sec = 0.0;
    double copy_sec = 0.0;
    double upload_sec = 0.0;

    const double upload_t0 = bgj_cuda_wall_time();
    for (int slot = 0; slot < device_count; slot++) {
        if (!bgj_cuda_set_current_device(devices[slot])) goto fail_sync;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)slot);
        if (!scratch) {
            set_plain_error("out of host memory");
            goto fail_sync;
        }
        if (!bgj_cuda_lsh_upload_signatures(scratch, sh, mbound, shsize)) {
            goto fail_sync;
        }
    }
    upload_sec = bgj_cuda_wall_time() - upload_t0;

    while (next_tile < total_tile_slots) {
        int submitted = 0;
        const double submit_t0 = bgj_cuda_wall_time();
        for (int slot = 0; slot < device_count && next_tile < total_tile_slots; slot++) {
            uint64_t count = chunk_tile_slots;
            if (count > total_tile_slots - next_tile) count = total_tile_slots - next_tile;
            if (count == 0u) break;
            if (!bgj_cuda_set_current_device(devices[slot])) goto fail_sync;
            bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)slot);
            if (!scratch) {
                set_plain_error("out of host memory");
                goto fail_sync;
            }
            if (!bgj_cuda_lsh_search_range_submit(scratch,
                                                  sh,
                                                  mbound,
                                                  shsize,
                                                  threshold,
                                                  next_tile,
                                                  count,
                                                  result_capacity,
                                                  0)) {
                goto fail_sync;
            }
            tile_begin[slot] = next_tile;
            tile_count[slot] = count;
            submitted++;
            chunks++;
            next_tile += count;
        }
        if (!submitted) break;
        submit_sec += bgj_cuda_wall_time() - submit_t0;
        waves++;

        for (int slot = 0; slot < submitted; slot++) {
            bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)slot);
            if (!scratch || !scratch->stream_ready) goto fail_sync;
            if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
                goto fail_sync;
            }

            const double wait_t0 = bgj_cuda_wall_time();
            cudaError_t err = cudaStreamSynchronize(scratch->stream);
            if (err != cudaSuccess) {
                set_cuda_error("cudaStreamSynchronize LSH stream counts", err);
                goto fail_sync;
            }
            bgj_cuda_get_submitted_counts(scratch,
                                          &submitted_counts[slot],
                                          &submitted_overflows[slot]);
            wait_sec += bgj_cuda_wall_time() - wait_t0;

            const double copy_t0 = bgj_cuda_wall_time();
            if (!bgj_cuda_finish_submitted_lsh_bucket(scratch,
                                                      results,
                                                      result_capacity,
                                                      submitted_counts[slot],
                                                      submitted_overflows[slot],
                                                      &copied_counts[slot],
                                                      &copied_overflows[slot])) {
                goto fail_sync;
            }
            err = cudaStreamSynchronize(scratch->stream);
            if (err != cudaSuccess) {
                set_cuda_error("cudaStreamSynchronize LSH stream results", err);
                goto fail_sync;
            }
            if (!bgj_cuda_complete_lsh_pinned_results(scratch)) goto fail_sync;
            copy_sec += bgj_cuda_wall_time() - copy_t0;

            if (copied_overflows[slot]) {
                if (overflow) *overflow = 1;
                set_plain_error("CUDA LSH stream chunk overflow");
                goto done;
            }
            if (copied_counts[slot] > 0u) {
                if (!callback(callback_ctx,
                              results,
                              copied_counts[slot],
                              tile_begin[slot],
                              tile_count[slot])) {
                    set_plain_error("CUDA LSH stream callback failed");
                    goto fail_sync;
                }
                stream_results += copied_counts[slot];
            }
        }
    }

done:
    if (total_result_count) *total_result_count = stream_results;
    set_plain_error("no CUDA error");
    if (split_profile) {
        fprintf(stderr,
                "cuda_lsh_stream_profile: devices=%d mbound=%u threshold=%d "
                "slots=%lu chunks=%lu waves=%lu chunk_slots=%lu results=%lu "
                "overflow=%d upload=%.6fs submit=%.6fs wait=%.6fs copy=%.6fs total=%.6fs\n",
                device_count,
                mbound,
                threshold,
                (unsigned long)total_tile_slots,
                (unsigned long)chunks,
                (unsigned long)waves,
                (unsigned long)chunk_tile_slots,
                (unsigned long)stream_results,
                overflow ? *overflow : 0,
                upload_sec,
                submit_sec,
                wait_sec,
                copy_sec,
                bgj_cuda_wall_time() - total_t0);
        fflush(stderr);
    }
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 1;

fail_sync:
    for (int slot = 0; slot < device_count; slot++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)slot);
        if (scratch && scratch->stream_ready) {
            if (scratch->device >= 0) cudaSetDevice(scratch->device);
            cudaStreamSynchronize(scratch->stream);
        }
    }
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 0;
}

extern "C" int bgj_cuda_lsh_search_range_raw(const uint8_t *sh,
                                              uint32_t mbound,
                                              uint32_t shsize,
                                              int32_t threshold,
                                              uint64_t tile_slot_begin,
                                              uint64_t tile_slot_count,
                                              bgj_cuda_result_t *results,
                                              uint32_t result_capacity,
                                              uint32_t *result_count,
                                              int *overflow)
{
    if (!bgj_cuda_select_thread_device()) return 0;
    bgj_cuda_raw_scratch_t *scratch = &bgj_cuda_raw_scratch;
    if (shsize != 64u) {
        set_plain_error("CUDA LSH search only supports 64-byte signatures");
        return 0;
    }
    if (result_capacity == 0) {
        set_plain_error("result capacity is zero");
        return 0;
    }
    if (mbound < 2u) {
        *result_count = 0;
        *overflow = 0;
        set_plain_error("no CUDA error");
        return 1;
    }
    const uint64_t total_tile_slots = bgj_cuda_lsh_total_tile_slots(mbound);
    if (tile_slot_begin >= total_tile_slots || tile_slot_count == 0u) {
        *result_count = 0;
        *overflow = 0;
        set_plain_error("no CUDA error");
        return 1;
    }
    if (tile_slot_count > total_tile_slots - tile_slot_begin) {
        tile_slot_count = total_tile_slots - tile_slot_begin;
    }
    if (!bgj_cuda_lsh_search_range_submit(scratch,
                                          sh,
                                          mbound,
                                          shsize,
                                          threshold,
                                          tile_slot_begin,
                                          tile_slot_count,
                                          result_capacity,
                                          1)) {
        return 0;
    }
    cudaStream_t stream = scratch->stream;
    CUDA_TRY(cudaStreamSynchronize(stream));
    bgj_cuda_get_submitted_counts(scratch, result_count, overflow);
    if (*result_count > result_capacity) {
        *result_count = result_capacity;
        *overflow = 1;
    }
    if (!bgj_cuda_finish_submitted_lsh_bucket(scratch,
                                              results,
                                              result_capacity,
                                              *result_count,
                                              *overflow,
                                              result_count,
                                              overflow)) {
        return 0;
    }
    CUDA_TRY(cudaStreamSynchronize(stream));
    if (!bgj_cuda_complete_lsh_pinned_results(scratch)) return 0;

    set_plain_error("no CUDA error");
    return 1;

fail:
    return 0;
}

extern "C" int bgj_cuda_lsh_search_raw(const uint8_t *sh,
                                        uint32_t mbound,
                                        uint32_t shsize,
                                        int32_t threshold,
                                        bgj_cuda_result_t *results,
                                        uint32_t result_capacity,
                                        uint32_t *result_count,
                                        int *overflow)
{
    const uint64_t total_tile_slots = bgj_cuda_lsh_total_tile_slots(mbound);
    if (shsize == 64u &&
        result_capacity > 0 &&
        mbound >= 2u &&
        bgj_cuda_lsh_multi_gpu_requested() &&
        total_tile_slots >= bgj_cuda_lsh_multi_gpu_min_slots()) {
        if (bgj_cuda_lsh_search_split_raw(sh,
                                          mbound,
                                          shsize,
                                          threshold,
                                          results,
                                          result_capacity,
                                          result_count,
                                          overflow,
                                          total_tile_slots)) {
            return 1;
        }
    }
    return bgj_cuda_lsh_search_range_raw(sh,
                                         mbound,
                                         shsize,
                                         threshold,
                                         0u,
                                         total_tile_slots,
                                         results,
                                         result_capacity,
                                         result_count,
                                         overflow);
}

extern "C" int bgj_cuda_lsh_lift_raw(const float *fvec,
                                      uint32_t mbound,
                                      uint32_t fd,
                                      uint32_t fd8,
                                      const uint32_t *candidates,
                                      uint32_t candidate_count,
                                      const float *b_full,
                                      const float *idiag,
                                      const float *min_norm,
                                      uint32_t id,
                                      uint32_t csd,
                                      float threshold_scale,
                                      bgj_cuda_lsh_lift_result_t *results,
                                      uint32_t result_capacity,
                                      uint32_t *result_count,
                                      int *overflow)
{
    if (!result_count || !overflow) {
        set_plain_error("invalid LSH lift output pointers");
        return 0;
    }
    *result_count = 0;
    *overflow = 0;
    if (candidate_count == 0u) {
        set_plain_error("no CUDA error");
        return 1;
    }
    if (!fvec || !candidates || !b_full || !idiag || !min_norm || !results) {
        set_plain_error("invalid LSH lift input pointers");
        return 0;
    }
    if (result_capacity == 0u) {
        set_plain_error("LSH lift result capacity is zero");
        return 0;
    }
    if (fd == 0u || fd > BGJ_CUDA_LSH_LIFT_MAX_FD ||
        fd8 < fd || fd8 > BGJ_CUDA_LSH_LIFT_MAX_FD ||
        id == 0u || id > fd || csd == 0u || csd > fd || mbound == 0u) {
        set_plain_error("unsupported CUDA LSH lift dimensions");
        return 0;
    }
    if (threshold_scale <= 0.0f) threshold_scale = 1.0f;

    if (!bgj_cuda_select_thread_device()) return 0;
    bgj_cuda_lsh_lift_scratch_t *scratch = &bgj_cuda_lsh_lift_scratch;
    if (!bgj_cuda_prepare_lsh_lift_stream(scratch)) return 0;
    cudaStream_t stream = scratch->stream;

    {
        const size_t fvec_bytes = (size_t)mbound * (size_t)fd8 * sizeof(float);
        const size_t candidate_bytes = (size_t)candidate_count * 3u * sizeof(uint32_t);
        const size_t b_full_bytes = (size_t)fd * (size_t)fd8 * sizeof(float);
        const size_t idiag_bytes = (size_t)fd * sizeof(float);
        const size_t min_norm_bytes = (size_t)id * sizeof(float);
        const size_t result_bytes = (size_t)result_capacity * sizeof(bgj_cuda_lsh_lift_result_t);

        CUDA_ENSURE(scratch->fvec, scratch->fvec_capacity, fvec_bytes);
        CUDA_ENSURE(scratch->candidates, scratch->candidate_capacity, candidate_bytes);
        CUDA_ENSURE(scratch->b_full, scratch->b_full_capacity, b_full_bytes);
        CUDA_ENSURE(scratch->idiag, scratch->idiag_capacity, idiag_bytes);
        CUDA_ENSURE(scratch->min_norm, scratch->min_norm_capacity, min_norm_bytes);
        CUDA_ENSURE(scratch->results, scratch->result_capacity, result_bytes);
        CUDA_ENSURE(scratch->result_count, scratch->result_count_capacity, sizeof(uint32_t));
        CUDA_ENSURE(scratch->overflow, scratch->overflow_capacity, sizeof(int));
        if (!ensure_cuda_host_capacity((void **)&scratch->host_result_count,
                                       &scratch->host_result_count_capacity,
                                       sizeof(uint32_t)) ||
            !ensure_cuda_host_capacity((void **)&scratch->host_overflow,
                                       &scratch->host_overflow_capacity,
                                       sizeof(int))) {
            goto fail;
        }
        *scratch->host_result_count = 0;
        *scratch->host_overflow = 0;

        CUDA_TRY(cudaMemcpyAsync(scratch->fvec, fvec, fvec_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->candidates, candidates, candidate_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->b_full, b_full, b_full_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->idiag, idiag, idiag_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->min_norm, min_norm, min_norm_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemsetAsync(scratch->result_count, 0, sizeof(uint32_t), stream));
        CUDA_TRY(cudaMemsetAsync(scratch->overflow, 0, sizeof(int), stream));

        {
            const uint32_t grid = (candidate_count + BGJ_CUDA_LSH_LIFT_THREADS - 1u) /
                                  BGJ_CUDA_LSH_LIFT_THREADS;
            bgj_cuda_lsh_lift_kernel<<<grid,
                                        BGJ_CUDA_LSH_LIFT_THREADS,
                                        0,
                                        stream>>>(scratch->fvec,
                                                  mbound,
                                                  fd,
                                                  fd8,
                                                  scratch->candidates,
                                                  candidate_count,
                                                  scratch->b_full,
                                                  scratch->idiag,
                                                  scratch->min_norm,
                                                  id,
                                                  csd,
                                                  threshold_scale,
                                                  scratch->results,
                                                  result_capacity,
                                                  scratch->result_count,
                                                  scratch->overflow);
            CUDA_TRY(cudaGetLastError());
        }

        CUDA_TRY(cudaMemcpyAsync(scratch->host_result_count,
                                 scratch->result_count,
                                 sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost,
                                 stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->host_overflow,
                                 scratch->overflow,
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 stream));
        CUDA_TRY(cudaStreamSynchronize(stream));

        uint32_t count = *scratch->host_result_count;
        int did_overflow = *scratch->host_overflow;
        if (count > result_capacity) {
            count = result_capacity;
            did_overflow = 1;
        }
        if (count > 0u) {
            CUDA_TRY(cudaMemcpyAsync(results,
                                     scratch->results,
                                     (size_t)count * sizeof(bgj_cuda_lsh_lift_result_t),
                                     cudaMemcpyDeviceToHost,
                                     stream));
            CUDA_TRY(cudaStreamSynchronize(stream));
        }
        *result_count = count;
        *overflow = did_overflow;
    }

    set_plain_error("no CUDA error");
    return 1;

fail:
    return 0;
}

struct bgj_cuda_materialize_scratch_t {
    bgj_cuda_materialize_desc_t *desc;
    uint8_t *b_dual;
    int8_t *b_dual_i8;
    float *b_local;
    int8_t *tmp_vec;
    int16_t *exact_vec;
    int32_t *coeff_i32;
    float *coeff_f32;
    float *fvec;
    int8_t *dst_vec;
    int32_t *dst_vnorm;
    int32_t *dst_vsum;
    uint32_t *gather_indices;
    int8_t *gather_vec;
    int8_t *host_dst_vec;
    int32_t *host_dst_vnorm;
    int32_t *host_dst_vsum;
    cudaStream_t stream;
    cublasHandle_t handle;
    int device;
    size_t desc_capacity;
    size_t b_dual_capacity;
    size_t b_dual_i8_capacity;
    size_t b_local_capacity;
    size_t tmp_vec_capacity;
    size_t exact_vec_capacity;
    size_t coeff_i32_capacity;
    size_t coeff_f32_capacity;
    size_t fvec_capacity;
    size_t dst_vec_capacity;
    size_t dst_vnorm_capacity;
    size_t dst_vsum_capacity;
    size_t gather_indices_capacity;
    size_t gather_vec_capacity;
    size_t host_dst_vec_capacity;
    size_t host_dst_vnorm_capacity;
    size_t host_dst_vsum_capacity;
    const uint8_t *basis_b_dual_host;
    const float *basis_b_local_host;
    uint32_t basis_vec_length;
    uint32_t basis_csd;
    uint64_t basis_b_dual_hash;
    uint64_t basis_b_local_hash;
    int basis_ready;
    int b_dual_i8_ready;
    int stream_ready;
    int handle_ready;
    int profile_events_ready;
    int stage_active;
    int phase_profile_active;
    uint32_t stage_count;
    uint32_t stage_vec_length;
    cudaEvent_t profile_start;
    cudaEvent_t profile_stop;
    bgj_cuda_materialize_phase_profile_t last_profile;
    pthread_mutex_t lock;

    bgj_cuda_materialize_scratch_t()
        : desc(NULL),
          b_dual(NULL),
          b_dual_i8(NULL),
          b_local(NULL),
          tmp_vec(NULL),
          exact_vec(NULL),
          coeff_i32(NULL),
          coeff_f32(NULL),
          fvec(NULL),
          dst_vec(NULL),
          dst_vnorm(NULL),
          dst_vsum(NULL),
          gather_indices(NULL),
          gather_vec(NULL),
          host_dst_vec(NULL),
          host_dst_vnorm(NULL),
          host_dst_vsum(NULL),
          stream(NULL),
          handle(NULL),
          device(-1),
          desc_capacity(0),
          b_dual_capacity(0),
          b_dual_i8_capacity(0),
          b_local_capacity(0),
          tmp_vec_capacity(0),
          exact_vec_capacity(0),
          coeff_i32_capacity(0),
          coeff_f32_capacity(0),
          fvec_capacity(0),
          dst_vec_capacity(0),
          dst_vnorm_capacity(0),
          dst_vsum_capacity(0),
          gather_indices_capacity(0),
          gather_vec_capacity(0),
          host_dst_vec_capacity(0),
          host_dst_vnorm_capacity(0),
          host_dst_vsum_capacity(0),
          basis_b_dual_host(NULL),
          basis_b_local_host(NULL),
          basis_vec_length(0),
          basis_csd(0),
          basis_b_dual_hash(0),
          basis_b_local_hash(0),
          basis_ready(0),
          b_dual_i8_ready(0),
          stream_ready(0),
          handle_ready(0),
          profile_events_ready(0),
          stage_active(0),
          phase_profile_active(0),
          stage_count(0),
          stage_vec_length(0),
          profile_start(NULL),
          profile_stop(NULL),
          last_profile()
    {
        memset(&last_profile, 0, sizeof(last_profile));
        pthread_mutex_init(&lock, NULL);
    }

    void release()
    {
        if (device >= 0) cudaSetDevice(device);
        if (stream_ready) cudaStreamSynchronize(stream);
        if (handle_ready) cublasDestroy(handle);
        cudaFree(desc);
        cudaFree(b_dual);
        cudaFree(b_dual_i8);
        cudaFree(b_local);
        cudaFree(tmp_vec);
        cudaFree(exact_vec);
        cudaFree(coeff_i32);
        cudaFree(coeff_f32);
        cudaFree(fvec);
        cudaFree(dst_vec);
        cudaFree(dst_vnorm);
        cudaFree(dst_vsum);
        cudaFree(gather_indices);
        cudaFree(gather_vec);
        cudaFreeHost(host_dst_vec);
        cudaFreeHost(host_dst_vnorm);
        cudaFreeHost(host_dst_vsum);
        if (profile_events_ready) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        if (stream_ready) cudaStreamDestroy(stream);
        desc = NULL;
        b_dual = NULL;
        b_dual_i8 = NULL;
        b_local = NULL;
        tmp_vec = NULL;
        exact_vec = NULL;
        coeff_i32 = NULL;
        coeff_f32 = NULL;
        fvec = NULL;
        dst_vec = NULL;
        dst_vnorm = NULL;
        dst_vsum = NULL;
        gather_indices = NULL;
        gather_vec = NULL;
        host_dst_vec = NULL;
        host_dst_vnorm = NULL;
        host_dst_vsum = NULL;
        stream = NULL;
        handle = NULL;
        device = -1;
        desc_capacity = 0;
        b_dual_capacity = 0;
        b_dual_i8_capacity = 0;
        b_local_capacity = 0;
        tmp_vec_capacity = 0;
        exact_vec_capacity = 0;
        coeff_i32_capacity = 0;
        coeff_f32_capacity = 0;
        fvec_capacity = 0;
        dst_vec_capacity = 0;
        dst_vnorm_capacity = 0;
        dst_vsum_capacity = 0;
        gather_indices_capacity = 0;
        gather_vec_capacity = 0;
        host_dst_vec_capacity = 0;
        host_dst_vnorm_capacity = 0;
        host_dst_vsum_capacity = 0;
        basis_b_dual_host = NULL;
        basis_b_local_host = NULL;
        basis_vec_length = 0;
        basis_csd = 0;
        basis_b_dual_hash = 0;
        basis_b_local_hash = 0;
        basis_ready = 0;
        b_dual_i8_ready = 0;
        stream_ready = 0;
        handle_ready = 0;
        profile_events_ready = 0;
        stage_active = 0;
        phase_profile_active = 0;
        stage_count = 0;
        stage_vec_length = 0;
        profile_start = NULL;
        profile_stop = NULL;
        memset(&last_profile, 0, sizeof(last_profile));
    }

    ~bgj_cuda_materialize_scratch_t()
    {
        release();
        pthread_mutex_destroy(&lock);
    }
};

static bgj_cuda_materialize_scratch_t bgj_cuda_materialize_scratch;

static int bgj_cuda_prepare_materialize_stream(bgj_cuda_materialize_scratch_t *scratch)
{
    int device = 0;
    if (!bgj_cuda_current_device(&device)) return 0;
    if (scratch->device >= 0 && scratch->device != device) {
        scratch->release();
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            set_cuda_error("cudaSetDevice prepare materialize stream", err);
            return 0;
        }
    }
    if (!scratch->stream_ready) {
        cudaError_t err = cudaStreamCreateWithFlags(&scratch->stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamCreateWithFlags materialize", err);
            return 0;
        }
        scratch->device = device;
        scratch->stream_ready = 1;
    }
    return 1;
}

static int bgj_cuda_prepare_materialize_handle(bgj_cuda_materialize_scratch_t *scratch)
{
    if (!scratch->handle_ready) {
        cublasStatus_t status = cublasCreate(&scratch->handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            set_cublas_error("cublasCreate materialize", status);
            return 0;
        }
        scratch->handle_ready = 1;
    }
    cublasMath_t math_mode =
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
        CUBLAS_PEDANTIC_MATH;
#else
        CUBLAS_DEFAULT_MATH;
#endif
    const char *tf32_env = getenv("BGJ_CUDA_MATERIALIZE_TF32");
    if (tf32_env && tf32_env[0] && tf32_env[0] != '0') {
        math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
    }
    cublasStatus_t status = cublasSetMathMode(scratch->handle, math_mode);
    if (status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cublasSetMathMode materialize", status);
        return 0;
    }
    status = cublasSetStream(scratch->handle, scratch->stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        set_cublas_error("cublasSetStream materialize", status);
        return 0;
    }
    return 1;
}

static int bgj_cuda_materialize_phase_profile_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_PHASE_PROFILE");
    return env && env[0] && env[0] != '0';
}

static int bgj_cuda_prepare_materialize_profile_events(bgj_cuda_materialize_scratch_t *scratch)
{
    if (scratch->profile_events_ready) return 1;
    cudaError_t err = cudaEventCreate(&scratch->profile_start);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventCreate materialize profile start", err);
        return 0;
    }
    err = cudaEventCreate(&scratch->profile_stop);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventCreate materialize profile stop", err);
        cudaEventDestroy(scratch->profile_start);
        scratch->profile_start = NULL;
        return 0;
    }
    scratch->profile_events_ready = 1;
    return 1;
}

static int bgj_cuda_materialize_profile_begin(bgj_cuda_materialize_scratch_t *scratch,
                                              int enabled)
{
    if (!enabled) return 1;
    cudaError_t err = cudaEventRecord(scratch->profile_start, scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventRecord materialize profile begin", err);
        return 0;
    }
    return 1;
}

static int bgj_cuda_materialize_profile_end(bgj_cuda_materialize_scratch_t *scratch,
                                            int enabled,
                                            double *dst_sec)
{
    if (!enabled) return 1;
    cudaError_t err = cudaEventRecord(scratch->profile_stop, scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventRecord materialize profile end", err);
        return 0;
    }
    err = cudaEventSynchronize(scratch->profile_stop);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventSynchronize materialize profile", err);
        return 0;
    }
    float ms = 0.0f;
    err = cudaEventElapsedTime(&ms, scratch->profile_start, scratch->profile_stop);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventElapsedTime materialize profile", err);
        return 0;
    }
    *dst_sec += (double)ms * 0.001;
    return 1;
}

extern "C" void bgj_cuda_materialize_last_profile(bgj_cuda_materialize_phase_profile_t *profile)
{
    if (!profile) return;
    *profile = bgj_cuda_materialize_scratch.last_profile;
}

static int bgj_cuda_materialize_pinned_host_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_PINNED_HOST");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

static int bgj_cuda_materialize_copy_outputs(bgj_cuda_materialize_scratch_t *scratch,
                                             uint32_t count,
                                             uint32_t vec_length,
                                             int8_t *dst_vec,
                                             int32_t *dst_vnorm,
                                             int32_t *dst_vsum)
{
    const size_t vec_bytes = (size_t)count * vec_length * sizeof(int8_t);
    const size_t i32_bytes = (size_t)count * sizeof(int32_t);
    cudaError_t err;

    if (!bgj_cuda_materialize_pinned_host_requested()) {
        err = cudaMemcpyAsync(dst_vec,
                              scratch->dst_vec,
                              vec_bytes,
                              cudaMemcpyDeviceToHost,
                              scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaMemcpyAsync materialize dst_vec", err);
            return 0;
        }
        err = cudaMemcpyAsync(dst_vnorm,
                              scratch->dst_vnorm,
                              i32_bytes,
                              cudaMemcpyDeviceToHost,
                              scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaMemcpyAsync materialize dst_vnorm", err);
            return 0;
        }
        err = cudaMemcpyAsync(dst_vsum,
                              scratch->dst_vsum,
                              i32_bytes,
                              cudaMemcpyDeviceToHost,
                              scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaMemcpyAsync materialize dst_vsum", err);
            return 0;
        }
        err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize materialize output", err);
            return 0;
        }
        return 1;
    }

    if (!ensure_cuda_host_capacity((void **)&scratch->host_dst_vec,
                                   &scratch->host_dst_vec_capacity,
                                   vec_bytes) ||
        !ensure_cuda_host_capacity((void **)&scratch->host_dst_vnorm,
                                   &scratch->host_dst_vnorm_capacity,
                                   i32_bytes) ||
        !ensure_cuda_host_capacity((void **)&scratch->host_dst_vsum,
                                   &scratch->host_dst_vsum_capacity,
                                   i32_bytes)) {
        return 0;
    }

    err = cudaMemcpyAsync(scratch->host_dst_vec,
                          scratch->dst_vec,
                          vec_bytes,
                          cudaMemcpyDeviceToHost,
                          scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaMemcpyAsync pinned materialize dst_vec", err);
        return 0;
    }
    err = cudaMemcpyAsync(scratch->host_dst_vnorm,
                          scratch->dst_vnorm,
                          i32_bytes,
                          cudaMemcpyDeviceToHost,
                          scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaMemcpyAsync pinned materialize dst_vnorm", err);
        return 0;
    }
    err = cudaMemcpyAsync(scratch->host_dst_vsum,
                          scratch->dst_vsum,
                          i32_bytes,
                          cudaMemcpyDeviceToHost,
                          scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaMemcpyAsync pinned materialize dst_vsum", err);
        return 0;
    }
    err = cudaStreamSynchronize(scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaStreamSynchronize pinned materialize output", err);
        return 0;
    }

    memcpy(dst_vec, scratch->host_dst_vec, vec_bytes);
    memcpy(dst_vnorm, scratch->host_dst_vnorm, i32_bytes);
    memcpy(dst_vsum, scratch->host_dst_vsum, i32_bytes);
    return 1;
}

static int bgj_cuda_materialize_copy_meta_outputs(bgj_cuda_materialize_scratch_t *scratch,
                                                  uint32_t count,
                                                  int32_t *dst_vnorm,
                                                  int32_t *dst_vsum)
{
    const size_t i32_bytes = (size_t)count * sizeof(int32_t);
    cudaError_t err = cudaMemcpyAsync(dst_vnorm,
                                      scratch->dst_vnorm,
                                      i32_bytes,
                                      cudaMemcpyDeviceToHost,
                                      scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaMemcpyAsync staged materialize dst_vnorm", err);
        return 0;
    }
    err = cudaMemcpyAsync(dst_vsum,
                          scratch->dst_vsum,
                          i32_bytes,
                          cudaMemcpyDeviceToHost,
                          scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaMemcpyAsync staged materialize dst_vsum", err);
        return 0;
    }
    err = cudaStreamSynchronize(scratch->stream);
    if (err != cudaSuccess) {
        set_cuda_error("cudaStreamSynchronize staged materialize meta", err);
        return 0;
    }
    return 1;
}

#define BGJ_CUDA_MATERIALIZE_MAX_DIM 256u
#define BGJ_CUDA_MATERIALIZE_THREADS 256u

__device__ __forceinline__ int bgj_cuda_wrap_i8(int x)
{
    int y = x & 255;
    return y >= 128 ? y - 256 : y;
}

__global__ void bgj_cuda_materialize_center_dual_kernel(const uint8_t *src,
                                                        uint64_t count,
                                                        int8_t *dst)
{
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t pos = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pos < count;
         pos += stride) {
        dst[pos] = (int8_t)((int)src[pos] - 128);
    }
}

__global__ void bgj_cuda_materialize_mask_b_local_kernel(float *b_local,
                                                         uint32_t csd,
                                                         uint32_t vec_length)
{
    const uint64_t total = (uint64_t)csd * vec_length;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t pos = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pos < total;
         pos += stride) {
        const uint32_t row = (uint32_t)(pos / vec_length);
        const uint32_t col = (uint32_t)(pos - (uint64_t)row * vec_length);
        if (col > row) b_local[pos] = 0.0f;
    }
}

__global__ void bgj_cuda_materialize_gather_vectors_kernel(const int8_t *src_vec,
                                                           uint32_t src_count,
                                                           uint32_t vec_length,
                                                           const uint32_t *indices,
                                                           uint32_t count,
                                                           int8_t *dst_vec)
{
    const uint64_t total = (uint64_t)count * vec_length;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t pos = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pos < total;
         pos += stride) {
        const uint32_t cand = (uint32_t)(pos / vec_length);
        const uint32_t j = (uint32_t)(pos - (uint64_t)cand * vec_length);
        const uint32_t src = indices[cand];
        dst_vec[pos] = src < src_count ? src_vec[(uint64_t)src * vec_length + j] : 0;
    }
}

__global__ __launch_bounds__(BGJ_CUDA_MATERIALIZE_THREADS, 1)
static void bgj_cuda_materialize_build_kernel(const int8_t *pool_vecs,
                                              uint32_t vec_length,
                                              const bgj_cuda_materialize_desc_t *desc,
                                              uint32_t count,
                                              int8_t *tmp_vec,
                                              int16_t *exact_vec)
{
    const uint32_t cand = blockIdx.x;
    if (cand >= count) return;

    const uint32_t tid = threadIdx.x;
    const bgj_cuda_materialize_desc_t d = desc[cand];

    for (uint32_t j = tid; j < vec_length; j += blockDim.x) {
        const int x = (int)pool_vecs[(uint64_t)d.x * vec_length + j];
        const int y = (int)pool_vecs[(uint64_t)d.y * vec_length + j];
        int v = 0;
        switch (d.type) {
        case BGJ_CUDA_SOL_A:
            v = x + y;
            break;
        case BGJ_CUDA_SOL_S:
            v = x - y;
            break;
        case BGJ_CUDA_SOL_AA:
            v = x + y + (int)pool_vecs[(uint64_t)d.z * vec_length + j];
            break;
        case BGJ_CUDA_SOL_SA:
            v = x - y + (int)pool_vecs[(uint64_t)d.z * vec_length + j];
            break;
        case BGJ_CUDA_SOL_SS:
            v = x - y - (int)pool_vecs[(uint64_t)d.z * vec_length + j];
            break;
        default:
            v = 0;
            break;
        }
        const int wrapped = bgj_cuda_wrap_i8(v);
        tmp_vec[(uint64_t)cand * vec_length + j] = (int8_t)wrapped;
        exact_vec[(uint64_t)cand * vec_length + j] = (int16_t)v;
    }
}

__global__ __launch_bounds__(256, 1)
static void bgj_cuda_materialize_coeff_kernel(const int32_t *coeff_i32,
                                              uint32_t csd,
                                              uint32_t coeff_ld,
                                              uint32_t count,
                                              int32_t dhalf,
                                              int32_t dshift,
                                              float *coeff_f32)
{
    const uint64_t total = (uint64_t)csd * count;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t pos = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pos < total;
         pos += stride) {
        const uint32_t i = (uint32_t)(pos % csd);
        const uint32_t cand = (uint32_t)(pos / csd);
        const uint64_t padded_pos = (uint64_t)cand * coeff_ld + i;
        const int32_t c = (coeff_i32[padded_pos] + dhalf) >> dshift;
        coeff_f32[padded_pos] = (float)c;
    }
}

__global__ __launch_bounds__(BGJ_CUDA_MATERIALIZE_THREADS, 1)
static void bgj_cuda_materialize_finish_kernel(const float *fvec,
                                               const int16_t *exact_vec,
                                               uint32_t vec_length,
                                               uint32_t count,
                                               int8_t *dst_vec,
                                               int32_t *dst_vnorm,
                                               int32_t *dst_vsum)
{
    const uint32_t cand = blockIdx.x;
    if (cand >= count) return;

    __shared__ int32_t ireduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ float freduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ int reject_flag;
    const uint32_t tid = threadIdx.x;
    if (tid == 0) reject_flag = 0;
    __syncthreads();

    int local_sum = 0;
    float local_norm = 0.0f;
    int local_reject = 0;
    for (uint32_t j = tid; j < vec_length; j += blockDim.x) {
        const float value = fvec[cand + (uint64_t)count * j];
        const int rounded = __float2int_rn(value);
        const int wrapped = bgj_cuda_wrap_i8(rounded);
        dst_vec[(uint64_t)cand * vec_length + j] = (int8_t)wrapped;
        local_sum += wrapped;
        local_norm += value * value;
        const int diff = (int)exact_vec[(uint64_t)cand * vec_length + j] - wrapped;
        if ((diff < -3 || diff > 3) || wrapped == -128) local_reject = 1;
    }

    if (local_reject) atomicExch(&reject_flag, 1);
    ireduce[tid] = local_sum;
    freduce[tid] = local_norm;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ireduce[tid] += ireduce[tid + stride];
            freduce[tid] += freduce[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst_vsum[cand] = 128 * ireduce[0];
        dst_vnorm[cand] = reject_flag ? 2147483647 : __float2int_rn(0.5f * freduce[0]);
    }
}

__global__ __launch_bounds__(BGJ_CUDA_MATERIALIZE_THREADS, 1)
static void bgj_cuda_materialize_exact_finish_kernel(const int32_t *coeff_i32,
                                                     uint32_t coeff_ld,
                                                     const float *b_local,
                                                     const int16_t *exact_vec,
                                                     uint32_t vec_length,
                                                     uint32_t csd,
                                                     uint32_t count,
                                                     int32_t dhalf,
                                                     int32_t dshift,
                                                     int8_t *dst_vec,
                                                     int32_t *dst_vnorm,
                                                     int32_t *dst_vsum)
{
    const uint32_t cand = blockIdx.x;
    if (cand >= count) return;

    __shared__ int32_t ireduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ float freduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ int reject_flag;
    const uint32_t tid = threadIdx.x;
    if (tid == 0) reject_flag = 0;
    __syncthreads();

    int local_sum = 0;
    float local_norm = 0.0f;
    int local_reject = 0;
    const int32_t *cand_coeff = coeff_i32 + (uint64_t)cand * coeff_ld;
    for (uint32_t j = tid; j < vec_length; j += blockDim.x) {
        float value = 0.0f;
        for (uint32_t i = j; i < csd; i++) {
            const int32_t coeff = (cand_coeff[i] + dhalf) >> dshift;
            value = fmaf(b_local[(uint64_t)i * vec_length + j], (float)coeff, value);
        }
        const int rounded = __float2int_rn(value);
        const int wrapped = bgj_cuda_wrap_i8(rounded);
        dst_vec[(uint64_t)cand * vec_length + j] = (int8_t)wrapped;
        local_sum += wrapped;
        local_norm = fmaf(value, value, local_norm);
        const int diff = (int)exact_vec[(uint64_t)cand * vec_length + j] - wrapped;
        if ((diff < -3 || diff > 3) || wrapped == -128) local_reject = 1;
    }

    if (local_reject) atomicExch(&reject_flag, 1);
    ireduce[tid] = local_sum;
    freduce[tid] = local_norm;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ireduce[tid] += ireduce[tid + stride];
            freduce[tid] += freduce[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst_vsum[cand] = 128 * ireduce[0];
        dst_vnorm[cand] = reject_flag ? 2147483647 : __float2int_rn(0.5f * freduce[0]);
    }
}

__global__ __launch_bounds__(BGJ_CUDA_MATERIALIZE_THREADS, 1)
static void bgj_cuda_materialize_exact_finish_f32_kernel(const float *coeff_f32,
                                                         uint32_t coeff_ld,
                                                         const float *b_local,
                                                         const int16_t *exact_vec,
                                                         uint32_t vec_length,
                                                         uint32_t csd,
                                                         uint32_t count,
                                                         int8_t *dst_vec,
                                                         int32_t *dst_vnorm,
                                                         int32_t *dst_vsum)
{
    const uint32_t cand = blockIdx.x;
    if (cand >= count) return;

    __shared__ int32_t ireduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ float freduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ int reject_flag;
    const uint32_t tid = threadIdx.x;
    if (tid == 0) reject_flag = 0;
    __syncthreads();

    int local_sum = 0;
    float local_norm = 0.0f;
    int local_reject = 0;
    const float *cand_coeff = coeff_f32 + (uint64_t)cand * coeff_ld;
    for (uint32_t j = tid; j < vec_length; j += blockDim.x) {
        float value = 0.0f;
        for (uint32_t i = j; i < csd; i++) {
            value = fmaf(b_local[(uint64_t)i * vec_length + j], cand_coeff[i], value);
        }
        const int rounded = __float2int_rn(value);
        const int wrapped = bgj_cuda_wrap_i8(rounded);
        dst_vec[(uint64_t)cand * vec_length + j] = (int8_t)wrapped;
        local_sum += wrapped;
        local_norm = fmaf(value, value, local_norm);
        const int diff = (int)exact_vec[(uint64_t)cand * vec_length + j] - wrapped;
        if ((diff < -3 || diff > 3) || wrapped == -128) local_reject = 1;
    }

    if (local_reject) atomicExch(&reject_flag, 1);
    ireduce[tid] = local_sum;
    freduce[tid] = local_norm;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ireduce[tid] += ireduce[tid + stride];
            freduce[tid] += freduce[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst_vsum[cand] = 128 * ireduce[0];
        dst_vnorm[cand] = reject_flag ? 2147483647 : __float2int_rn(0.5f * freduce[0]);
    }
}

__global__ __launch_bounds__(BGJ_CUDA_MATERIALIZE_THREADS, 1)
static void bgj_cuda_materialize_fused_kernel(const int8_t *pool_vecs,
                                              uint32_t vec_length,
                                              const bgj_cuda_materialize_desc_t *desc,
                                              uint32_t count,
                                              const uint8_t *b_dual,
                                              const float *b_local,
                                              uint32_t csd,
                                              int32_t dhalf,
                                              int32_t dshift,
                                              int8_t *dst_vec,
                                              int32_t *dst_vnorm,
                                              int32_t *dst_vsum)
{
    const uint32_t cand = blockIdx.x;
    if (cand >= count) return;

    __shared__ int8_t tmp[BGJ_CUDA_MATERIALIZE_MAX_DIM];
    __shared__ int16_t exact[BGJ_CUDA_MATERIALIZE_MAX_DIM];
    __shared__ int32_t coeff[BGJ_CUDA_MATERIALIZE_MAX_DIM];
    __shared__ int32_t ireduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ float freduce[BGJ_CUDA_MATERIALIZE_THREADS];
    __shared__ int reject_flag;

    const uint32_t tid = threadIdx.x;
    const bgj_cuda_materialize_desc_t d = desc[cand];

    for (uint32_t j = tid; j < vec_length; j += blockDim.x) {
        const int x = (int)pool_vecs[(uint64_t)d.x * vec_length + j];
        const int y = (int)pool_vecs[(uint64_t)d.y * vec_length + j];
        int v = 0;
        switch (d.type) {
        case BGJ_CUDA_SOL_A:
            v = x + y;
            break;
        case BGJ_CUDA_SOL_S:
            v = x - y;
            break;
        case BGJ_CUDA_SOL_AA:
            v = x + y + (int)pool_vecs[(uint64_t)d.z * vec_length + j];
            break;
        case BGJ_CUDA_SOL_SA:
            v = x - y + (int)pool_vecs[(uint64_t)d.z * vec_length + j];
            break;
        case BGJ_CUDA_SOL_SS:
            v = x - y - (int)pool_vecs[(uint64_t)d.z * vec_length + j];
            break;
        default:
            v = 0;
            break;
        }
        tmp[j] = (int8_t)bgj_cuda_wrap_i8(v);
        exact[j] = (int16_t)v;
    }
    __syncthreads();

    for (uint32_t i = tid; i < csd; i += blockDim.x) {
        int32_t dot = 0;
        const uint8_t *dual_row = b_dual + (uint64_t)i * vec_length;
        for (uint32_t j = 0; j < vec_length; j++) {
            dot += ((int32_t)dual_row[j] - 128) * (int32_t)tmp[j];
        }
        coeff[i] = (dot + dhalf) >> dshift;
    }
    if (tid == 0) reject_flag = 0;
    __syncthreads();

    int local_sum = 0;
    float local_norm = 0.0f;
    int local_reject = 0;
    for (uint32_t j = tid; j < vec_length; j += blockDim.x) {
        float value = 0.0f;
        for (uint32_t i = j; i < csd; i++) {
            value = fmaf(b_local[(uint64_t)i * vec_length + j], (float)coeff[i], value);
        }
        const int rounded = __float2int_rn(value);
        const int wrapped = bgj_cuda_wrap_i8(rounded);
        dst_vec[(uint64_t)cand * vec_length + j] = (int8_t)wrapped;
        local_sum += wrapped;
        local_norm += value * value;
        const int diff = (int)exact[j] - wrapped;
        if ((diff < -3 || diff > 3) || wrapped == -128) local_reject = 1;
    }

    if (local_reject) atomicExch(&reject_flag, 1);
    ireduce[tid] = local_sum;
    freduce[tid] = local_norm;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ireduce[tid] += ireduce[tid + stride];
            freduce[tid] += freduce[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst_vsum[cand] = 128 * ireduce[0];
        dst_vnorm[cand] = reject_flag ? 2147483647 : __float2int_rn(0.5f * freduce[0]);
    }
}

static uint32_t bgj_cuda_materialize_chunk_size()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_CHUNK");
    unsigned long value = 1048576UL;
    if (env && env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed > 0) value = parsed;
    }
    if (value < 1024UL) value = 1024UL;
    if (value > 1048576UL) value = 1048576UL;
    return (uint32_t)value;
}

static int bgj_cuda_materialize_basis_cache_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_BASIS_CACHE");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static int bgj_cuda_materialize_fused_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_FUSED");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

static int bgj_cuda_materialize_sgemm_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_SGEMM");
    return env && env[0] && env[0] != '0';
}

static int bgj_cuda_materialize_fused_coeff_requested()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_FUSED_COEFF");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

static uint32_t bgj_cuda_materialize_align_up(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint32_t bgj_cuda_materialize_thread_count()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_THREADS");
    unsigned long value = 32UL;
    if (env && env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed > 0) value = parsed;
    }
    if (value <= 32UL) return 32u;
    if (value <= 64UL) return 64u;
    if (value <= 128UL) return 128u;
    return 256u;
}

static uint32_t bgj_cuda_materialize_fused_max_count()
{
    const char *env = getenv("BGJ_CUDA_MATERIALIZE_FUSED_MAX");
    unsigned long value = 8192UL;
    if (env && env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed > 0) value = parsed;
    }
    if (value > 1048576UL) value = 1048576UL;
    return (uint32_t)value;
}

static int bgj_cuda_materialize_sol_list_impl(const int8_t *pool_vecs,
                                              uint64_t pool_epoch,
                                              uint32_t pool_size,
                                              uint32_t vec_length,
                                              const bgj_cuda_materialize_desc_t *desc,
                                              uint32_t count,
                                              const uint8_t *b_dual,
                                              const float *b_local,
                                              uint32_t csd,
                                              int32_t dhalf,
                                              int32_t dshift,
                                              int8_t *dst_vec,
                                              int32_t *dst_vnorm,
                                              int32_t *dst_vsum,
                                              int staged)
{
    if (count == 0) {
        set_plain_error("no CUDA error");
        return 1;
    }
    if (!pool_vecs || !desc || !b_dual || !b_local || (!staged && !dst_vec) ||
        !dst_vnorm || !dst_vsum) {
        set_plain_error("invalid materialize pointer");
        return 0;
    }
    if (vec_length == 0 || csd == 0 ||
        vec_length > BGJ_CUDA_MATERIALIZE_MAX_DIM ||
        csd > BGJ_CUDA_MATERIALIZE_MAX_DIM) {
        set_plain_error("unsupported materialize dimension");
        return 0;
    }
    if (pool_size == 0) {
        set_plain_error("invalid materialize pool size");
        return 0;
    }
    if (!bgj_cuda_select_thread_device()) return 0;

    bgj_cuda_materialize_scratch_t *scratch = &bgj_cuda_materialize_scratch;
    pthread_mutex_lock(&scratch->lock);
    if (scratch->stage_active) {
        pthread_mutex_unlock(&scratch->lock);
        set_plain_error("staged materialize session already active");
        return 0;
    }

    int8_t *device_pool_vecs = NULL;
    uint32_t chunk_limit = 0;
    uint32_t chunk_limit_gemm = 0;
    uint32_t offset = 0;
    int alpha_i = 1;
    int beta_i = 0;
    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    int use_fused = 0;
    int use_sgemm_reconstruct = 0;
    int use_fused_coeff = 0;
    int phase_profile = 0;
    const size_t pool_vec_bytes = (size_t)pool_size * vec_length * sizeof(int8_t);
    const size_t b_dual_bytes = (size_t)csd * vec_length * sizeof(uint8_t);
    const size_t b_local_bytes = (size_t)csd * vec_length * sizeof(float);
    const uint32_t gemm_csd = bgj_cuda_materialize_align_up(csd, 16u);
    const int basis_cache = bgj_cuda_materialize_basis_cache_requested();
    uint64_t b_dual_hash = 0;
    uint64_t b_local_hash = 0;
    int upload_basis = 1;
    memset(&scratch->last_profile, 0, sizeof(scratch->last_profile));
    scratch->last_profile.candidates = count;
    phase_profile = bgj_cuda_materialize_phase_profile_requested();
    scratch->phase_profile_active = phase_profile;
    if (phase_profile && !bgj_cuda_prepare_materialize_profile_events(scratch)) goto fail;
    if (!bgj_cuda_prepare_materialize_stream(scratch)) goto fail;
    if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
    if (!bgj_cuda_prepare_shared_pool_cache(pool_vecs,
                                            pool_epoch,
                                            pool_size,
                                            vec_length,
                                            pool_vec_bytes,
                                            scratch->stream,
                                            &device_pool_vecs)) {
        goto fail;
    }
    if (!bgj_cuda_materialize_profile_end(scratch,
                                          phase_profile,
                                          &scratch->last_profile.pool_sec)) {
        goto fail;
    }

    if (b_dual_bytes > scratch->b_dual_capacity || b_local_bytes > scratch->b_local_capacity) {
        scratch->basis_ready = 0;
        scratch->b_dual_i8_ready = 0;
    }
    CUDA_ENSURE(scratch->b_dual, scratch->b_dual_capacity, b_dual_bytes);
    CUDA_ENSURE(scratch->b_local, scratch->b_local_capacity, b_local_bytes);

    if (basis_cache) {
        b_dual_hash = bgj_cuda_hash_bytes(b_dual, b_dual_bytes);
        b_local_hash = bgj_cuda_hash_bytes(b_local, b_local_bytes);
        upload_basis =
            !scratch->basis_ready ||
            scratch->basis_b_dual_host != b_dual ||
            scratch->basis_b_local_host != b_local ||
            scratch->basis_vec_length != vec_length ||
            scratch->basis_csd != csd ||
            scratch->basis_b_dual_hash != b_dual_hash ||
            scratch->basis_b_local_hash != b_local_hash;
    }
    if (upload_basis) {
        if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
        CUDA_TRY(cudaMemcpyAsync(scratch->b_dual, b_dual,
                                 b_dual_bytes,
                                 cudaMemcpyHostToDevice, scratch->stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->b_local, b_local,
                                 b_local_bytes,
                                 cudaMemcpyHostToDevice, scratch->stream));
        scratch->basis_b_dual_host = b_dual;
        scratch->basis_b_local_host = b_local;
        scratch->basis_vec_length = vec_length;
        scratch->basis_csd = csd;
        scratch->basis_b_dual_hash = b_dual_hash;
        scratch->basis_b_local_hash = b_local_hash;
        scratch->basis_ready = basis_cache ? 1 : 0;
        scratch->b_dual_i8_ready = 0;

        {
            const uint64_t local_total = (uint64_t)csd * vec_length;
            uint32_t local_blocks = (uint32_t)((local_total + 255u) / 256u);
            if (local_blocks > 65535u) local_blocks = 65535u;
            bgj_cuda_materialize_mask_b_local_kernel<<<local_blocks,
                                                       256,
                                                       0,
                                                       scratch->stream>>>(scratch->b_local,
                                                                          csd,
                                                                          vec_length);
            CUDA_TRY(cudaGetLastError());
        }
        if (!bgj_cuda_materialize_profile_end(scratch,
                                              phase_profile,
                                              &scratch->last_profile.basis_sec)) {
            goto fail;
        }
    }

    use_fused = bgj_cuda_materialize_fused_requested() &&
                count <= bgj_cuda_materialize_fused_max_count();
    use_sgemm_reconstruct = bgj_cuda_materialize_sgemm_requested();
    use_fused_coeff = bgj_cuda_materialize_fused_coeff_requested();
    if (use_fused) {
        CUDA_ENSURE(scratch->desc, scratch->desc_capacity, (size_t)count * sizeof(bgj_cuda_materialize_desc_t));
        CUDA_ENSURE(scratch->dst_vec, scratch->dst_vec_capacity, (size_t)count * vec_length * sizeof(int8_t));
        CUDA_ENSURE(scratch->dst_vnorm, scratch->dst_vnorm_capacity, (size_t)count * sizeof(int32_t));
        CUDA_ENSURE(scratch->dst_vsum, scratch->dst_vsum_capacity, (size_t)count * sizeof(int32_t));
        CUDA_TRY(cudaMemcpyAsync(scratch->desc, desc,
                                 (size_t)count * sizeof(bgj_cuda_materialize_desc_t),
                                 cudaMemcpyHostToDevice,
                                 scratch->stream));
        bgj_cuda_materialize_fused_kernel<<<count,
                                            BGJ_CUDA_MATERIALIZE_THREADS,
                                            0,
                                            scratch->stream>>>(device_pool_vecs,
                                                               vec_length,
                                                               scratch->desc,
                                                               count,
                                                               scratch->b_dual,
                                                               scratch->b_local,
                                                               csd,
                                                               dhalf,
                                                               dshift,
                                                               scratch->dst_vec,
                                                               scratch->dst_vnorm,
                                                               scratch->dst_vsum);
        CUDA_TRY(cudaGetLastError());
        if (staged) {
            if (!bgj_cuda_materialize_copy_meta_outputs(scratch,
                                                        count,
                                                        dst_vnorm,
                                                        dst_vsum)) {
                goto fail;
            }
            scratch->stage_active = 1;
            scratch->stage_count = count;
            scratch->stage_vec_length = vec_length;
            set_plain_error("no CUDA error");
            return 1;
        } else {
            if (!bgj_cuda_materialize_copy_outputs(scratch,
                                                   count,
                                                   vec_length,
                                                   dst_vec,
                                                   dst_vnorm,
                                                   dst_vsum)) {
                goto fail;
            }
        }
        pthread_mutex_unlock(&scratch->lock);
        set_plain_error("no CUDA error");
        return 1;
    }

    if (!bgj_cuda_prepare_materialize_handle(scratch)) goto fail;

    if ((size_t)gemm_csd * vec_length * sizeof(int8_t) > scratch->b_dual_i8_capacity) {
        scratch->b_dual_i8_ready = 0;
    }
    CUDA_ENSURE(scratch->b_dual_i8,
                scratch->b_dual_i8_capacity,
                (size_t)gemm_csd * vec_length * sizeof(int8_t));
    if (!scratch->b_dual_i8_ready) {
        const uint64_t dual_total = (uint64_t)csd * vec_length;
        CUDA_TRY(cudaMemsetAsync(scratch->b_dual_i8,
                                 0,
                                 (size_t)gemm_csd * vec_length * sizeof(int8_t),
                                 scratch->stream));
        uint32_t dual_blocks = (uint32_t)((dual_total + 255u) / 256u);
        if (dual_blocks > 65535u) dual_blocks = 65535u;
        bgj_cuda_materialize_center_dual_kernel<<<dual_blocks, 256, 0, scratch->stream>>>(
            scratch->b_dual,
            dual_total,
            scratch->b_dual_i8);
        CUDA_TRY(cudaGetLastError());
        scratch->b_dual_i8_ready = 1;
    }

    chunk_limit = bgj_cuda_materialize_chunk_size();
    if (chunk_limit > count) chunk_limit = count;
    chunk_limit_gemm = bgj_cuda_materialize_align_up(chunk_limit, 16u);
    CUDA_ENSURE(scratch->desc, scratch->desc_capacity, (size_t)chunk_limit * sizeof(bgj_cuda_materialize_desc_t));
    CUDA_ENSURE(scratch->tmp_vec, scratch->tmp_vec_capacity, (size_t)chunk_limit_gemm * vec_length * sizeof(int8_t));
    CUDA_ENSURE(scratch->exact_vec, scratch->exact_vec_capacity, (size_t)chunk_limit * vec_length * sizeof(int16_t));
    CUDA_ENSURE(scratch->coeff_i32, scratch->coeff_i32_capacity, (size_t)chunk_limit_gemm * gemm_csd * sizeof(int32_t));
    if (use_sgemm_reconstruct || !use_fused_coeff) {
        CUDA_ENSURE(scratch->coeff_f32, scratch->coeff_f32_capacity, (size_t)chunk_limit_gemm * gemm_csd * sizeof(float));
    }
    if (use_sgemm_reconstruct) {
        CUDA_ENSURE(scratch->fvec, scratch->fvec_capacity, (size_t)chunk_limit * vec_length * sizeof(float));
    }
    CUDA_ENSURE(scratch->dst_vec,
                scratch->dst_vec_capacity,
                (size_t)(staged ? count : chunk_limit) * vec_length * sizeof(int8_t));
    CUDA_ENSURE(scratch->dst_vnorm, scratch->dst_vnorm_capacity, (size_t)chunk_limit * sizeof(int32_t));
    CUDA_ENSURE(scratch->dst_vsum, scratch->dst_vsum_capacity, (size_t)chunk_limit * sizeof(int32_t));

    for (offset = 0; offset < count; offset += chunk_limit) {
        const uint32_t chunk_count =
            (count - offset < chunk_limit) ? (count - offset) : chunk_limit;
        const uint32_t gemm_chunk_count =
            bgj_cuda_materialize_align_up(chunk_count, 16u);
        int8_t *chunk_dst_vec =
            staged ? scratch->dst_vec + (uint64_t)offset * vec_length : scratch->dst_vec;
        if (phase_profile) scratch->last_profile.chunks++;
        if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
        CUDA_TRY(cudaMemcpyAsync(scratch->desc, desc + offset,
                                 (size_t)chunk_count * sizeof(bgj_cuda_materialize_desc_t),
                                 cudaMemcpyHostToDevice, scratch->stream));
        if (!bgj_cuda_materialize_profile_end(scratch,
                                              phase_profile,
                                              &scratch->last_profile.desc_sec)) {
            goto fail;
        }

        if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
        bgj_cuda_materialize_build_kernel<<<chunk_count,
                                            BGJ_CUDA_MATERIALIZE_THREADS,
                                            0,
                                            scratch->stream>>>(device_pool_vecs,
                                                               vec_length,
                                                               scratch->desc,
                                                               chunk_count,
                                                               scratch->tmp_vec,
                                                               scratch->exact_vec);
        CUDA_TRY(cudaGetLastError());
        if (gemm_chunk_count > chunk_count) {
            CUDA_TRY(cudaMemsetAsync(scratch->tmp_vec + (uint64_t)chunk_count * vec_length,
                                     0,
                                     (size_t)(gemm_chunk_count - chunk_count) *
                                         vec_length * sizeof(int8_t),
                                     scratch->stream));
        }
        if (!bgj_cuda_materialize_profile_end(scratch,
                                              phase_profile,
                                              &scratch->last_profile.build_sec)) {
            goto fail;
        }

        if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
        CUBLAS_TRY(cublasGemmEx(scratch->handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                (int)gemm_csd,
                                (int)gemm_chunk_count,
                                (int)vec_length,
                                &alpha_i,
                                scratch->b_dual_i8,
                                CUDA_R_8I,
                                (int)vec_length,
                                scratch->tmp_vec,
                                CUDA_R_8I,
                                (int)vec_length,
                                &beta_i,
                                scratch->coeff_i32,
                                CUDA_R_32I,
                                (int)gemm_csd,
                                CUBLAS_COMPUTE_32I,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (!bgj_cuda_materialize_profile_end(scratch,
                                              phase_profile,
                                              &scratch->last_profile.gemm_sec)) {
            goto fail;
        }

        if (use_sgemm_reconstruct || !use_fused_coeff) {
            const uint64_t coeff_total = (uint64_t)chunk_count * csd;
            uint32_t coeff_blocks = (uint32_t)((coeff_total + 255u) / 256u);
            if (coeff_blocks > 65535u) coeff_blocks = 65535u;
            if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
            bgj_cuda_materialize_coeff_kernel<<<coeff_blocks, 256, 0, scratch->stream>>>(
                scratch->coeff_i32,
                csd,
                gemm_csd,
                chunk_count,
                dhalf,
                dshift,
                scratch->coeff_f32);
            CUDA_TRY(cudaGetLastError());
            if (!bgj_cuda_materialize_profile_end(scratch,
                                                  phase_profile,
                                                  &scratch->last_profile.coeff_sec)) {
                goto fail;
            }
        }

        if (use_sgemm_reconstruct) {
            if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
            CUBLAS_TRY(cublasSgemm(scratch->handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_T,
                                   (int)chunk_count,
                                   (int)vec_length,
                                   (int)csd,
                                   &alpha_f,
                                   scratch->coeff_f32,
                                   (int)gemm_csd,
                                   scratch->b_local,
                                   (int)vec_length,
                                   &beta_f,
                                   scratch->fvec,
                                   (int)chunk_count));

            bgj_cuda_materialize_finish_kernel<<<chunk_count,
                                                 BGJ_CUDA_MATERIALIZE_THREADS,
                                                 0,
                                                 scratch->stream>>>(scratch->fvec,
                                                                    scratch->exact_vec,
                                                                    vec_length,
                                                                    chunk_count,
                                                                    chunk_dst_vec,
                                                                    scratch->dst_vnorm,
                                                                    scratch->dst_vsum);
        } else if (!use_fused_coeff) {
            if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
            bgj_cuda_materialize_exact_finish_f32_kernel<<<chunk_count,
                                                           bgj_cuda_materialize_thread_count(),
                                                           0,
                                                           scratch->stream>>>(scratch->coeff_f32,
                                                                              gemm_csd,
                                                                              scratch->b_local,
                                                                              scratch->exact_vec,
                                                                              vec_length,
                                                                              csd,
                                                                              chunk_count,
                                                                              chunk_dst_vec,
                                                                              scratch->dst_vnorm,
                                                                              scratch->dst_vsum);
        } else {
            if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
            bgj_cuda_materialize_exact_finish_kernel<<<chunk_count,
                                                       bgj_cuda_materialize_thread_count(),
                                                       0,
                                                       scratch->stream>>>(scratch->coeff_i32,
                                                                          gemm_csd,
                                                                          scratch->b_local,
                                                                          scratch->exact_vec,
                                                                          vec_length,
                                                                          csd,
                                                                          chunk_count,
                                                                          dhalf,
                                                                          dshift,
                                                                          chunk_dst_vec,
                                                                          scratch->dst_vnorm,
                                                                          scratch->dst_vsum);
        }
        CUDA_TRY(cudaGetLastError());
        if (!bgj_cuda_materialize_profile_end(scratch,
                                              phase_profile,
                                              &scratch->last_profile.reconstruct_sec)) {
            goto fail;
        }

        if (!bgj_cuda_materialize_profile_begin(scratch, phase_profile)) goto fail;
        if (staged) {
            if (!bgj_cuda_materialize_copy_meta_outputs(scratch,
                                                        chunk_count,
                                                        dst_vnorm + offset,
                                                        dst_vsum + offset)) {
                goto fail;
            }
        } else {
            if (!bgj_cuda_materialize_copy_outputs(scratch,
                                                   chunk_count,
                                                   vec_length,
                                                   dst_vec + (uint64_t)offset * vec_length,
                                                   dst_vnorm + offset,
                                                   dst_vsum + offset)) {
                goto fail;
            }
        }
        if (!bgj_cuda_materialize_profile_end(scratch,
                                              phase_profile,
                                              &scratch->last_profile.copy_sec)) {
            goto fail;
        }
    }

    if (staged) {
        scratch->stage_active = 1;
        scratch->stage_count = count;
        scratch->stage_vec_length = vec_length;
        set_plain_error("no CUDA error");
        return 1;
    }

    pthread_mutex_unlock(&scratch->lock);
    set_plain_error("no CUDA error");
    return 1;

fail:
    if (scratch->stream_ready) cudaStreamSynchronize(scratch->stream);
    scratch->stage_active = 0;
    scratch->stage_count = 0;
    scratch->stage_vec_length = 0;
    pthread_mutex_unlock(&scratch->lock);
    return 0;
}

extern "C" int bgj_cuda_materialize_sol_list_raw(const int8_t *pool_vecs,
                                                  uint64_t pool_epoch,
                                                  uint32_t pool_size,
                                                  uint32_t vec_length,
                                                  const bgj_cuda_materialize_desc_t *desc,
                                                  uint32_t count,
                                                  const uint8_t *b_dual,
                                                  const float *b_local,
                                                  uint32_t csd,
                                                  int32_t dhalf,
                                                  int32_t dshift,
                                                  int8_t *dst_vec,
                                                  int32_t *dst_vnorm,
                                                  int32_t *dst_vsum)
{
    return bgj_cuda_materialize_sol_list_impl(pool_vecs,
                                              pool_epoch,
                                              pool_size,
                                              vec_length,
                                              desc,
                                              count,
                                              b_dual,
                                              b_local,
                                              csd,
                                              dhalf,
                                              dshift,
                                              dst_vec,
                                              dst_vnorm,
                                              dst_vsum,
                                              0);
}

extern "C" int bgj_cuda_materialize_sol_list_staged_raw(const int8_t *pool_vecs,
                                                         uint64_t pool_epoch,
                                                         uint32_t pool_size,
                                                         uint32_t vec_length,
                                                         const bgj_cuda_materialize_desc_t *desc,
                                                         uint32_t count,
                                                         const uint8_t *b_dual,
                                                         const float *b_local,
                                                         uint32_t csd,
                                                         int32_t dhalf,
                                                         int32_t dshift,
                                                         int32_t *dst_vnorm,
                                                         int32_t *dst_vsum)
{
    return bgj_cuda_materialize_sol_list_impl(pool_vecs,
                                              pool_epoch,
                                              pool_size,
                                              vec_length,
                                              desc,
                                              count,
                                              b_dual,
                                              b_local,
                                              csd,
                                              dhalf,
                                              dshift,
                                              NULL,
                                              dst_vnorm,
                                              dst_vsum,
                                              1);
}

extern "C" int bgj_cuda_materialize_copy_staged_vectors_raw(const uint32_t *indices,
                                                             uint32_t count,
                                                             uint32_t vec_length,
                                                             int8_t *dst_vec)
{
    bgj_cuda_materialize_scratch_t *scratch = &bgj_cuda_materialize_scratch;
    if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) return 0;
    if (!scratch->stage_active) {
        set_plain_error("no staged materialize session active");
        return 0;
    }
    if (count == 0) {
        set_plain_error("no CUDA error");
        return 1;
    }
    if (!indices || !dst_vec || vec_length != scratch->stage_vec_length) {
        set_plain_error("invalid staged materialize gather argument");
        return 0;
    }

    CUDA_ENSURE(scratch->gather_indices,
                scratch->gather_indices_capacity,
                (size_t)count * sizeof(uint32_t));
    CUDA_ENSURE(scratch->gather_vec,
                scratch->gather_vec_capacity,
                (size_t)count * vec_length * sizeof(int8_t));

    if (!bgj_cuda_materialize_profile_begin(scratch, scratch->phase_profile_active)) goto fail;
    CUDA_TRY(cudaMemcpyAsync(scratch->gather_indices,
                             indices,
                             (size_t)count * sizeof(uint32_t),
                             cudaMemcpyHostToDevice,
                             scratch->stream));
    {
        const uint64_t total = (uint64_t)count * vec_length;
        uint32_t blocks = (uint32_t)((total + 255u) / 256u);
        if (blocks > 65535u) blocks = 65535u;
        bgj_cuda_materialize_gather_vectors_kernel<<<blocks,
                                                     256,
                                                     0,
                                                     scratch->stream>>>(scratch->dst_vec,
                                                                        scratch->stage_count,
                                                                        vec_length,
                                                                        scratch->gather_indices,
                                                                        count,
                                                                        scratch->gather_vec);
        CUDA_TRY(cudaGetLastError());
    }
    CUDA_TRY(cudaMemcpyAsync(dst_vec,
                             scratch->gather_vec,
                             (size_t)count * vec_length * sizeof(int8_t),
                             cudaMemcpyDeviceToHost,
                             scratch->stream));
    CUDA_TRY(cudaStreamSynchronize(scratch->stream));
    if (!bgj_cuda_materialize_profile_end(scratch,
                                          scratch->phase_profile_active,
                                          &scratch->last_profile.copy_sec)) {
        goto fail;
    }
    set_plain_error("no CUDA error");
    return 1;

fail:
    if (scratch->stream_ready) cudaStreamSynchronize(scratch->stream);
    return 0;
}

extern "C" void bgj_cuda_materialize_finish_staged_raw()
{
    bgj_cuda_materialize_scratch_t *scratch = &bgj_cuda_materialize_scratch;
    if (!scratch->stage_active) return;
    if (scratch->device >= 0) cudaSetDevice(scratch->device);
    scratch->stage_active = 0;
    scratch->stage_count = 0;
    scratch->stage_vec_length = 0;
    pthread_mutex_unlock(&scratch->lock);
}

static int bgj_cuda_tensor_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static int bgj_cuda_deterministic_results_requested()
{
    const char *env = getenv("BGJ_CUDA_DETERMINISTIC_RESULTS");
    return env && env[0] && env[0] != '0';
}

static int bgj_cuda_multi_gpu_batch_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_MULTI_GPU_BATCH",
                             bgj_cuda_stable_multi_gpu_requested());
}

static uint64_t bgj_cuda_multi_gpu_batch_min_dots()
{
    // Smaller batch thresholds have reproduced CUDA state corruption on SVP-120.
    const uint64_t floor = 64ULL * 1024ULL * 1024ULL;
    const uint64_t value =
        bgj_cuda_env_u64("BGJ_CUDA_MULTI_GPU_BATCH_MIN_DOTS", floor);
    if (value < floor) {
        return bgj_cuda_env_flag("BGJ_CUDA_MULTI_GPU_BATCH_ALLOW_SMALL", 0) ?
               value : floor;
    }
    return value;
}

static int bgj_cuda_multi_gpu_batch_split_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_MULTI_GPU_BATCH_SPLIT", 0);
}

static uint64_t bgj_cuda_multi_gpu_batch_split_min_dots()
{
    return bgj_cuda_env_u64("BGJ_CUDA_MULTI_GPU_BATCH_SPLIT_MIN_DOTS",
                            bgj_cuda_multi_gpu_batch_min_dots());
}

static uint32_t bgj_cuda_multi_gpu_batch_split_min_share_pct()
{
    uint64_t value = bgj_cuda_env_u64("BGJ_CUDA_MULTI_GPU_BATCH_SPLIT_MIN_SHARE_PCT", 0);
    if (value > 100) value = 100;
    return (uint32_t)value;
}

static int bgj_cuda_multi_gpu_batch_tensor_split_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_MULTI_GPU_BATCH_TENSOR_SPLIT", 1);
}

static int bgj_cuda_single_bucket_split_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_SINGLE_BUCKET_SPLIT",
                             bgj_cuda_stable_multi_gpu_requested());
}

static int bgj_cuda_single_bucket_tensor_split_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_SINGLE_BUCKET_TENSOR_SPLIT", 1);
}

static int bgj_cuda_single_bucket_split_verify_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_SINGLE_BUCKET_SPLIT_VERIFY", 0);
}

static int bgj_cuda_single_bucket_split_verify_order_requested()
{
    return bgj_cuda_env_flag("BGJ_CUDA_SINGLE_BUCKET_SPLIT_VERIFY_ORDER", 0);
}

static uint64_t bgj_cuda_single_bucket_split_min_dots()
{
    const uint64_t default_value = bgj_cuda_stable_multi_gpu_requested() ?
                                   16000000ULL :
                                   64ULL * 1024ULL * 1024ULL;
    return bgj_cuda_env_u64("BGJ_CUDA_SINGLE_BUCKET_SPLIT_MIN_DOTS",
                            default_value);
}

static int bgj_cuda_tensor_same_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_SAME");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static int bgj_cuda_sm80_device()
{
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return 0;
    }
    if (device < 0 || device >= BGJ_CUDA_MAX_EXEC_DEVICES) return 0;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return 0;
    }
    return (prop.major == 8 && prop.minor == 0) ? 1 : 0;
}

static int bgj_cuda_tensor_np_wide_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_NP_WIDE");
    if (env && env[0]) return env[0] != '0';
    return bgj_cuda_sm80_device();
}

static int bgj_cuda_tensor_np_shared_a_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_NP_SHARED_A");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

static int bgj_cuda_tensor_reorder_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_REORDER");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static int bgj_cuda_tensor_np_multi_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_NP_MULTI");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

static uint32_t bgj_cuda_tensor_np_min_tiles()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_NP_MIN_TILES");
    if (env && env[0]) {
        unsigned long value = strtoul(env, NULL, 10);
        if (value > 0 && value <= 0xffffffffUL) return (uint32_t)value;
    }
    return 512u;
}

static uint32_t bgj_cuda_tensor_same_min_tiles()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_SAME_MIN_TILES");
    if (env && env[0]) {
        unsigned long value = strtoul(env, NULL, 10);
        if (value > 0 && value <= 0xffffffffUL) return (uint32_t)value;
    }
    return 32u;
}

static int bgj_cuda_tensor_capable()
{
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return 0;
    }
    if (device < 0 || device >= BGJ_CUDA_MAX_EXEC_DEVICES) return 0;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return 0;
    }
    return prop.major >= 8 ? 1 : 0;
}

static int bgj_cuda_bucket_tensor_requested()
{
    const char *env = getenv("BGJ_CUDA_BUCKET_TENSOR");
    if (env && env[0]) return env[0] != '0';
    return bgj_cuda_sm80_device();
}

static int bgj_cuda_bucket_deterministic_requested()
{
    const char *env = getenv("BGJ_CUDA_BUCKET_DETERMINISTIC");
    if (env && env[0]) return env[0] != '0';
    return 0;
}

static int bgj_cuda_bucket_block_append_requested()
{
    const char *env = getenv("BGJ_CUDA_BUCKET_BLOCK_APPEND");
    if (env && env[0]) return env[0] != '0';
    return bgj_cuda_sm80_device();
}

extern "C" int bgj_cuda_bucket_bgj1_raw(const int8_t *pool_vecs,
                                         uint64_t pool_epoch,
                                         uint32_t pool_size,
                                         const uint32_t *center_ids,
                                         uint32_t num_centers,
                                         const int32_t *vnorm,
                                         uint32_t vec_length,
                                         uint32_t alpha_x2_u16,
                                         bgj_cuda_bucket_entry_t *entries,
                                         uint32_t entry_capacity,
                                         uint32_t *entry_count,
                                         int *overflow)
{
    if (!bgj_cuda_select_thread_device()) return 0;
    if (entry_count) *entry_count = 0;
    if (overflow) *overflow = 0;
    if (!pool_vecs || !center_ids || !vnorm || !entries ||
        !entry_count || !overflow) {
        set_plain_error("invalid bucket pointer");
        return 0;
    }
    if (pool_size == 0 || num_centers == 0 || vec_length == 0) {
        set_plain_error("no CUDA error");
        return 1;
    }
    if (entry_capacity == 0) {
        set_plain_error("bucket entry capacity is zero");
        return 0;
    }

    bgj_cuda_bucket_scratch_t *scratch = &bgj_cuda_bucket_scratch;
    if (!bgj_cuda_prepare_bucket_stream(scratch)) return 0;
    cudaStream_t stream = scratch->stream;
    int8_t *device_pool_vecs = NULL;
    uint32_t h_entry_count = 0;
    int h_overflow = 0;
    uint32_t *h_det_counts = NULL;
    uint32_t *h_det_offsets = NULL;

    const size_t pool_vec_bytes = (size_t)pool_size * (size_t)vec_length * sizeof(int8_t);
    const size_t center_id_bytes = (size_t)num_centers * sizeof(uint32_t);
    const size_t center_vec_bytes = (size_t)num_centers * (size_t)vec_length * sizeof(int8_t);
    const size_t i32_bytes = (size_t)pool_size * sizeof(int32_t);
    const size_t entry_bytes = (size_t)entry_capacity * sizeof(bgj_cuda_bucket_entry_t);
    const uint32_t det_num_segments =
        (pool_size + BGJ_CUDA_BUCKET_DET_THREADS - 1u) / BGJ_CUDA_BUCKET_DET_THREADS;
    const uint64_t det_count_slots_u64 =
        (uint64_t)num_centers * 2ull * (uint64_t)det_num_segments;
    const size_t det_count_bytes = (size_t)det_count_slots_u64 * sizeof(uint32_t);

    if (!bgj_cuda_prepare_shared_pool_cache(pool_vecs,
                                            pool_epoch,
                                            pool_size,
                                            vec_length,
                                            pool_vec_bytes,
                                            stream,
                                            &device_pool_vecs)) {
        goto fail;
    }

    CUDA_ENSURE(scratch->center_ids, scratch->center_id_capacity, center_id_bytes);
    CUDA_ENSURE(scratch->vnorm, scratch->vnorm_capacity, i32_bytes);
    CUDA_ENSURE(scratch->entries, scratch->entry_capacity, entry_bytes);
    CUDA_ENSURE(scratch->entry_count, scratch->entry_count_capacity, sizeof(uint32_t));
    CUDA_ENSURE(scratch->overflow, scratch->overflow_capacity, sizeof(int));

    CUDA_TRY(cudaMemcpyAsync(scratch->center_ids,
                             center_ids,
                             center_id_bytes,
                             cudaMemcpyHostToDevice,
                             stream));
    CUDA_TRY(cudaMemcpyAsync(scratch->vnorm,
                             vnorm,
                             i32_bytes,
                             cudaMemcpyHostToDevice,
                             stream));
    CUDA_TRY(cudaMemsetAsync(scratch->entry_count, 0, sizeof(uint32_t), stream));
    CUDA_TRY(cudaMemsetAsync(scratch->overflow, 0, sizeof(int), stream));

    if (bgj_cuda_bucket_deterministic_requested()) {
        if (det_count_slots_u64 == 0 ||
            det_count_slots_u64 > (uint64_t)((size_t)-1 / sizeof(uint32_t))) {
            set_plain_error("bucket deterministic count buffer is too large");
            goto fail;
        }

        CUDA_ENSURE(scratch->bucket_counts,
                    scratch->bucket_count_capacity,
                    det_count_bytes);
        CUDA_ENSURE(scratch->bucket_offsets,
                    scratch->bucket_offset_capacity,
                    det_count_bytes);

        {
            const uint64_t det_blocks_u64 =
                (uint64_t)num_centers * (uint64_t)det_num_segments;
            uint32_t det_grid = det_blocks_u64 > 65535ull ? 65535u : (uint32_t)det_blocks_u64;
            if (det_grid == 0) det_grid = 1;
            bgj_cuda_bucket_bgj1_count_kernel<<<det_grid,
                                                 BGJ_CUDA_BUCKET_DET_THREADS,
                                                 0,
                                                 stream>>>(device_pool_vecs,
                                                           scratch->center_ids,
                                                           num_centers,
                                                           scratch->vnorm,
                                                           pool_size,
                                                           det_num_segments,
                                                           vec_length,
                                                           alpha_x2_u16,
                                                           scratch->bucket_counts);
            CUDA_TRY(cudaGetLastError());
        }

        h_det_counts = (uint32_t *)malloc(det_count_bytes);
        h_det_offsets = (uint32_t *)malloc(det_count_bytes);
        if (!h_det_counts || !h_det_offsets) {
            free(h_det_counts);
            free(h_det_offsets);
            h_det_counts = NULL;
            h_det_offsets = NULL;
            set_plain_error("bucket deterministic host count allocation failed");
            goto fail;
        }

        CUDA_TRY(cudaMemcpyAsync(h_det_counts,
                                 scratch->bucket_counts,
                                 det_count_bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream));
        CUDA_TRY(cudaStreamSynchronize(stream));

        uint64_t total_entries = 0;
        for (size_t i = 0; i < (size_t)det_count_slots_u64; i++) {
            h_det_offsets[i] = total_entries > 0xffffffffull ? 0xffffffffu : (uint32_t)total_entries;
            total_entries += (uint64_t)h_det_counts[i];
        }

        if (total_entries > (uint64_t)entry_capacity) {
            free(h_det_counts);
            free(h_det_offsets);
            h_det_counts = NULL;
            h_det_offsets = NULL;
            h_entry_count = entry_capacity;
            h_overflow = 1;
            *entry_count = h_entry_count;
            *overflow = h_overflow;
            set_plain_error("no CUDA error");
            return 1;
        }

        h_entry_count = (uint32_t)total_entries;
        if (h_entry_count) {
            CUDA_TRY(cudaMemcpyAsync(scratch->bucket_offsets,
                                     h_det_offsets,
                                     det_count_bytes,
                                     cudaMemcpyHostToDevice,
                                     stream));
            CUDA_TRY(cudaMemsetAsync(scratch->overflow, 0, sizeof(int), stream));

            const uint64_t det_blocks_u64 =
                (uint64_t)num_centers * (uint64_t)det_num_segments;
            uint32_t det_grid = det_blocks_u64 > 65535ull ? 65535u : (uint32_t)det_blocks_u64;
            if (det_grid == 0) det_grid = 1;
            bgj_cuda_bucket_bgj1_fill_kernel<<<det_grid,
                                                BGJ_CUDA_BUCKET_DET_THREADS,
                                                0,
                                                stream>>>(device_pool_vecs,
                                                          scratch->center_ids,
                                                          num_centers,
                                                          scratch->vnorm,
                                                          pool_size,
                                                          det_num_segments,
                                                          vec_length,
                                                          alpha_x2_u16,
                                                          scratch->bucket_offsets,
                                                          scratch->entries,
                                                          entry_capacity,
                                                          scratch->overflow);
            CUDA_TRY(cudaGetLastError());
            CUDA_TRY(cudaMemcpyAsync(&h_overflow,
                                     scratch->overflow,
                                     sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream));
            CUDA_TRY(cudaStreamSynchronize(stream));
            CUDA_TRY(cudaMemcpyAsync(entries,
                                     scratch->entries,
                                     (size_t)h_entry_count * sizeof(bgj_cuda_bucket_entry_t),
                                     cudaMemcpyDeviceToHost,
                                     stream));
            CUDA_TRY(cudaStreamSynchronize(stream));
        }

        free(h_det_counts);
        free(h_det_offsets);
        h_det_counts = NULL;
        h_det_offsets = NULL;
        *entry_count = h_entry_count;
        *overflow = h_overflow;
        set_plain_error("no CUDA error");
        return 1;
    }

    {
        const int tensor_requested = bgj_cuda_bucket_tensor_requested();
        const int use_block_append = !tensor_requested &&
                                     bgj_cuda_bucket_block_append_requested();
        const int use_tensor = tensor_requested &&
                               bgj_cuda_tensor_capable() &&
                               (vec_length % 16u) == 0 &&
                               num_centers >= 16u &&
                               (num_centers % 16u) == 0 &&
                               pool_size >= 16u;
        const uint32_t threads = 256;
        uint32_t scalar_start = 0;
        uint32_t scalar_count = pool_size;

        if (use_tensor) {
            const uint32_t tensor_pool_size = (pool_size / 16u) * 16u;
            const uint32_t center_pack_blocks =
                (uint32_t)((center_vec_bytes + threads - 1u) / threads);
            uint32_t bounded_center_pack_blocks = center_pack_blocks;
            if (bounded_center_pack_blocks == 0) bounded_center_pack_blocks = 1;
            if (bounded_center_pack_blocks > 65535u) bounded_center_pack_blocks = 65535u;

            CUDA_ENSURE(scratch->center_vecs, scratch->center_vec_capacity, center_vec_bytes);
            bgj_cuda_pack_pool_vecs_kernel<<<bounded_center_pack_blocks,
                                             threads,
                                             0,
                                             stream>>>(device_pool_vecs,
                                                       scratch->center_ids,
                                                       num_centers,
                                                       vec_length,
                                                       scratch->center_vecs);
            CUDA_TRY(cudaGetLastError());

            const uint32_t num_center_tiles = num_centers / 16u;
            const uint32_t num_pool_tiles = tensor_pool_size / 16u;
            const uint64_t tensor_tiles = (uint64_t)num_center_tiles * num_pool_tiles;
            uint32_t tensor_grid =
                (uint32_t)((tensor_tiles + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                           BGJ_CUDA_TENSOR_WARPS_PER_BLOCK);
            if (tensor_grid == 0) tensor_grid = 1;
            bgj_cuda_bucket_bgj1_tensor_kernel<<<tensor_grid,
                                                 BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                 0,
                                                 stream>>>(scratch->center_vecs,
                                                           device_pool_vecs,
                                                           num_center_tiles,
                                                           num_pool_tiles,
                                                           scratch->vnorm,
                                                           vec_length,
                                                           alpha_x2_u16,
                                                           scratch->entries,
                                                           entry_capacity,
                                                           scratch->entry_count,
                                                           scratch->overflow);
            CUDA_TRY(cudaGetLastError());
            scalar_start = tensor_pool_size;
            scalar_count = pool_size - tensor_pool_size;
        }

        if (scalar_count) {
            const uint64_t total_pairs = (uint64_t)scalar_count * num_centers;
            uint32_t blocks = (uint32_t)((total_pairs + threads - 1u) / threads);
            if (blocks == 0) blocks = 1;
            if (blocks > 65535u) blocks = 65535u;
            if (use_block_append) {
                bgj_cuda_bucket_bgj1_block_append_kernel<<<blocks,
                                                           threads,
                                                           0,
                                                           stream>>>(device_pool_vecs,
                                                                     scratch->center_ids,
                                                                     num_centers,
                                                                     scratch->vnorm,
                                                                     pool_size,
                                                                     scalar_start,
                                                                     scalar_count,
                                                                     vec_length,
                                                                     alpha_x2_u16,
                                                                     scratch->entries,
                                                                     entry_capacity,
                                                                     scratch->entry_count,
                                                                     scratch->overflow);
            } else {
                bgj_cuda_bucket_bgj1_kernel<<<blocks, threads, 0, stream>>>(device_pool_vecs,
                                                                            scratch->center_ids,
                                                                            num_centers,
                                                                            scratch->vnorm,
                                                                            pool_size,
                                                                            scalar_start,
                                                                            scalar_count,
                                                                            vec_length,
                                                                            alpha_x2_u16,
                                                                            scratch->entries,
                                                                            entry_capacity,
                                                                            scratch->entry_count,
                                                                            scratch->overflow);
            }
            CUDA_TRY(cudaGetLastError());
        }
    }
    CUDA_TRY(cudaMemcpyAsync(&h_entry_count,
                             scratch->entry_count,
                             sizeof(uint32_t),
                             cudaMemcpyDeviceToHost,
                             stream));
    CUDA_TRY(cudaMemcpyAsync(&h_overflow,
                             scratch->overflow,
                             sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    if (h_entry_count > entry_capacity) {
        h_entry_count = entry_capacity;
        h_overflow = 1;
    }
    if (h_entry_count) {
        CUDA_TRY(cudaMemcpyAsync(entries,
                                 scratch->entries,
                                 (size_t)h_entry_count * sizeof(bgj_cuda_bucket_entry_t),
                                 cudaMemcpyDeviceToHost,
                                 stream));
        CUDA_TRY(cudaStreamSynchronize(stream));
    }

    *entry_count = h_entry_count;
    *overflow = h_overflow;
    set_plain_error("no CUDA error");
    return 1;

fail:
    free(h_det_counts);
    free(h_det_offsets);
    if (scratch->stream_ready) cudaStreamSynchronize(scratch->stream);
    return 0;
}

static int bgj_cuda_search_bucket_raw_submit(bgj_cuda_raw_scratch_t *scratch,
                                             const int8_t *p_vecs,
                                             const int8_t *n_vecs,
                                             const int8_t *pool_vecs_host,
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
                                             uint64_t work_offset,
                                             uint64_t work_count,
                                             uint32_t tensor_split_slot,
                                             uint32_t tensor_split_count,
                                             uint32_t result_capacity,
                                             uint32_t *submitted_result_count,
                                             int *submitted_overflow)
{
    uint32_t tensor_np_num_p = 0;
    uint32_t tensor_np_num_n = 0;
    uint32_t tensor_same_num_p = 0;
    uint32_t tensor_same_num_n = 0;
    uint32_t tensor_np_min_tiles = 0;
    uint32_t tensor_same_min_tiles = 0;
    const int use_pool = pool_vecs_host != NULL;
    const int range_active = work_offset != 0 || work_count != 0;
    const int tensor_split_active = tensor_split_count > 1;
    int8_t *pool_vecs_device = NULL;
    uint32_t *h_det_counts = NULL;
    uint32_t *h_det_offsets = NULL;

    if (result_capacity == 0) {
        set_plain_error("result capacity is zero");
        return 0;
    }
    if (record_dp && (transform_dp + device_transform_dp + raw_center_dp) > 1) {
        set_plain_error("conflicting CUDA dot transforms");
        return 0;
    }
    if (range_active && tensor_split_active) {
        set_plain_error("conflicting CUDA bucket split modes");
        return 0;
    }
    if (tensor_split_active && tensor_split_slot >= tensor_split_count) {
        set_plain_error("invalid CUDA tensor split slot");
        return 0;
    }
    if (tensor_split_active && transform_dp) {
        set_plain_error("tensor CUDA bucket split does not support device CRED transform");
        return 0;
    }
    if (range_active && transform_dp) {
        set_plain_error("range CUDA bucket split does not support device CRED transform");
        return 0;
    }
    if (range_active && bgj_cuda_deterministic_results_requested()) {
        set_plain_error("range CUDA bucket split does not support deterministic results");
        return 0;
    }
    if (tensor_split_active && bgj_cuda_deterministic_results_requested()) {
        set_plain_error("tensor CUDA bucket split does not support deterministic results");
        return 0;
    }
    if (raw_center_dp && bgj_cuda_deterministic_results_requested()) {
        set_plain_error("raw CUDA center dots are not supported with deterministic results");
        return 0;
    }
    if (use_pool && pool_size == 0 && (num_p || num_n)) {
        set_plain_error("pool size is zero");
        return 0;
    }
    if (!bgj_cuda_prepare_stream(scratch)) return 0;
    cudaStream_t stream = scratch->stream;
    if (!bgj_cuda_search_profile_reset(scratch)) return 0;

    const size_t p_vec_bytes = (size_t)num_p * (size_t)vec_length * sizeof(int8_t);
    const size_t n_vec_bytes = (size_t)num_n * (size_t)vec_length * sizeof(int8_t);
    const size_t p_id_bytes = (size_t)num_p * sizeof(uint32_t);
    const size_t n_id_bytes = (size_t)num_n * sizeof(uint32_t);
    const size_t p_i32_bytes = (size_t)num_p * sizeof(int32_t);
    const size_t n_i32_bytes = (size_t)num_n * sizeof(int32_t);

    if (use_pool) {
        const size_t pool_vec_bytes = (size_t)pool_size * (size_t)vec_length * sizeof(int8_t);
        if (!bgj_cuda_prepare_shared_pool_cache(pool_vecs_host,
                                                pool_epoch,
                                                pool_size,
                                                vec_length,
                                                pool_vec_bytes,
                                                stream,
                                                &pool_vecs_device)) goto fail;
    }

    if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_H2D)) goto fail;
    if (num_p) {
        CUDA_ENSURE(scratch->p_vecs, scratch->p_vec_capacity, p_vec_bytes);
        CUDA_ENSURE(scratch->p_ids, scratch->p_id_capacity, p_id_bytes);
        CUDA_ENSURE(scratch->p_norm, scratch->p_i32_capacity, p_i32_bytes);
        CUDA_TRY(cudaMemcpyAsync(scratch->p_ids, p_ids, p_id_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->p_norm, p_norm, p_i32_bytes, cudaMemcpyHostToDevice, stream));
        if (record_dp) {
            CUDA_ENSURE(scratch->p_dot, scratch->p_dot_capacity, p_i32_bytes);
            CUDA_TRY(cudaMemcpyAsync(scratch->p_dot, p_dot, p_i32_bytes, cudaMemcpyHostToDevice, stream));
        }
    }

    if (num_n) {
        CUDA_ENSURE(scratch->n_vecs, scratch->n_vec_capacity, n_vec_bytes);
        CUDA_ENSURE(scratch->n_ids, scratch->n_id_capacity, n_id_bytes);
        CUDA_ENSURE(scratch->n_norm, scratch->n_i32_capacity, n_i32_bytes);
        CUDA_TRY(cudaMemcpyAsync(scratch->n_ids, n_ids, n_id_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->n_norm, n_norm, n_i32_bytes, cudaMemcpyHostToDevice, stream));
        if (record_dp) {
            CUDA_ENSURE(scratch->n_dot, scratch->n_dot_capacity, n_i32_bytes);
            CUDA_TRY(cudaMemcpyAsync(scratch->n_dot, n_dot, n_i32_bytes, cudaMemcpyHostToDevice, stream));
        }
    }
    if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_H2D)) goto fail;

    if (record_dot_copy_event && record_dp) {
        if (!bgj_cuda_prepare_dot_event(scratch)) goto fail;
        CUDA_TRY(cudaEventRecord(scratch->dot_copy_done, stream));
    }

    if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_PACK)) goto fail;
    if (num_p) {
        if (use_pool) {
            const uint32_t threads = 256;
            uint32_t blocks = (uint32_t)((p_vec_bytes + threads - 1) / threads);
            if (blocks == 0) blocks = 1;
            if (blocks > 65535) blocks = 65535;
            bgj_cuda_pack_pool_vecs_kernel<<<blocks, threads, 0, stream>>>(pool_vecs_device,
                                                                           scratch->p_ids,
                                                                           num_p,
                                                                           vec_length,
                                                                           scratch->p_vecs);
            CUDA_TRY(cudaGetLastError());
        } else {
            CUDA_TRY(cudaMemcpyAsync(scratch->p_vecs, p_vecs, p_vec_bytes, cudaMemcpyHostToDevice, stream));
        }
    }

    if (num_n) {
        if (use_pool) {
            const uint32_t threads = 256;
            uint32_t blocks = (uint32_t)((n_vec_bytes + threads - 1) / threads);
            if (blocks == 0) blocks = 1;
            if (blocks > 65535) blocks = 65535;
            bgj_cuda_pack_pool_vecs_kernel<<<blocks, threads, 0, stream>>>(pool_vecs_device,
                                                                           scratch->n_ids,
                                                                           num_n,
                                                                           vec_length,
                                                                           scratch->n_vecs);
            CUDA_TRY(cudaGetLastError());
        } else {
            CUDA_TRY(cudaMemcpyAsync(scratch->n_vecs, n_vecs, n_vec_bytes, cudaMemcpyHostToDevice, stream));
        }
    }
    if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_PACK)) goto fail;

    CUDA_ENSURE(scratch->results, scratch->result_capacity, (size_t)result_capacity * sizeof(bgj_cuda_result_t));
    CUDA_ENSURE(scratch->result_count, scratch->result_count_capacity, sizeof(uint32_t));
    CUDA_ENSURE(scratch->overflow, scratch->overflow_capacity, sizeof(int));
    if (!ensure_cuda_host_capacity((void **)&scratch->host_result_count,
                                   &scratch->host_result_count_capacity,
                                   sizeof(uint32_t)) ||
        !ensure_cuda_host_capacity((void **)&scratch->host_overflow,
                                   &scratch->host_overflow_capacity,
                                   sizeof(int))) {
        goto fail;
    }
    *scratch->host_result_count = 0;
    *scratch->host_overflow = 0;
    if (submitted_result_count) *submitted_result_count = 0;
    if (submitted_overflow) *submitted_overflow = 0;
    CUDA_TRY(cudaMemsetAsync(scratch->result_count, 0, sizeof(uint32_t), stream));
    CUDA_TRY(cudaMemsetAsync(scratch->overflow, 0, sizeof(int), stream));

    if (transform_dp && record_dp && (num_p || num_n)) {
        if (bgj_cuda_deterministic_results_requested()) {
            set_plain_error("CUDA cred transform is not supported with deterministic results");
            goto fail;
        }
        if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_CRED)) goto fail;
        const uint32_t threads = 256;
        const uint64_t total = (uint64_t)num_p + (uint64_t)num_n;
        uint32_t blocks = (uint32_t)((total + threads - 1u) / threads);
        if (blocks == 0) blocks = 1;
        if (blocks > 65535u) blocks = 65535u;
        bgj_cuda_transform_dp_and_search_cred_kernel<<<blocks, threads, 0, stream>>>(center_id,
                                                                                    scratch->p_ids,
                                                                                    scratch->n_ids,
                                                                                    scratch->p_norm,
                                                                                    scratch->n_norm,
                                                                                    scratch->p_dot,
                                                                                    scratch->n_dot,
                                                                                    num_p,
                                                                                    num_n,
                                                                                    goal_norm - center_norm,
                                                                                    scratch->results,
                                                                                    result_capacity,
                                                                                    scratch->result_count,
                                                                                    scratch->overflow);
        CUDA_TRY(cudaGetLastError());
        if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_CRED)) goto fail;
    }

    if (device_transform_dp && record_dp && (num_p || num_n)) {
        if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_CRED)) goto fail;
        const uint32_t threads = 256;
        const uint64_t total = (uint64_t)num_p + (uint64_t)num_n;
        uint32_t blocks = (uint32_t)((total + threads - 1u) / threads);
        if (blocks == 0) blocks = 1;
        if (blocks > 65535u) blocks = 65535u;
        bgj_cuda_transform_dp_kernel<<<blocks, threads, 0, stream>>>(scratch->p_norm,
                                                                     scratch->n_norm,
                                                                     scratch->p_dot,
                                                                     scratch->n_dot,
                                                                     num_p,
                                                                     num_n);
        CUDA_TRY(cudaGetLastError());
        if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_CRED)) goto fail;
    }

    if (!range_active && !tensor_split_active && bgj_cuda_deterministic_results_requested()) {
        const uint64_t np_pairs = (uint64_t)num_p * (uint64_t)num_n;
        const uint64_t pp_pairs = num_p > 1 ? (uint64_t)num_p * (uint64_t)num_p : 0ull;
        const uint64_t nn_pairs = num_n > 1 ? (uint64_t)num_n * (uint64_t)num_n : 0ull;
        const uint64_t np_segments =
            (np_pairs + BGJ_CUDA_SEARCH_DET_THREADS - 1u) / BGJ_CUDA_SEARCH_DET_THREADS;
        const uint64_t pp_segments =
            (pp_pairs + BGJ_CUDA_SEARCH_DET_THREADS - 1u) / BGJ_CUDA_SEARCH_DET_THREADS;
        const uint64_t nn_segments =
            (nn_pairs + BGJ_CUDA_SEARCH_DET_THREADS - 1u) / BGJ_CUDA_SEARCH_DET_THREADS;
        const uint64_t phase1_base = np_segments;
        const uint64_t phase2_base = phase1_base + (record_dp ? np_segments : 0ull);
        const uint64_t phase3_base = phase2_base + pp_segments;
        const uint64_t phase4_base = phase3_base + (record_dp ? pp_segments : 0ull);
        const uint64_t phase5_base = phase4_base + nn_segments;
        const uint64_t phase6_base = phase5_base + (record_dp ? nn_segments : 0ull);
        const uint64_t count_slots = phase6_base;
        const uint64_t pair_slots = np_segments + pp_segments + nn_segments;
        const int32_t center_goal_norm = goal_norm - center_norm;

        if (count_slots == 0) {
            *submitted_result_count = 0;
            *submitted_overflow = 0;
            set_plain_error("no CUDA error");
            return 1;
        }
        if (count_slots > (uint64_t)((size_t)-1 / sizeof(uint32_t))) {
            set_plain_error("deterministic result count buffer is too large");
            goto fail;
        }

        const size_t count_bytes = (size_t)count_slots * sizeof(uint32_t);
        CUDA_ENSURE(scratch->det_counts, scratch->det_count_capacity, count_bytes);
        CUDA_ENSURE(scratch->det_offsets, scratch->det_offset_capacity, count_bytes);

        {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_DET_COUNT)) goto fail;
            uint32_t grid = pair_slots > 65535ull ? 65535u : (uint32_t)pair_slots;
            if (grid == 0) grid = 1;
            bgj_cuda_search_det_pair_count_kernel<<<grid,
                                                     BGJ_CUDA_SEARCH_DET_THREADS,
                                                     0,
                                                     stream>>>(scratch->p_vecs,
                                                               scratch->n_vecs,
                                                               scratch->p_norm,
                                                               scratch->n_norm,
                                                               scratch->p_dot,
                                                               scratch->n_dot,
                                                               num_p,
                                                               num_n,
                                                               vec_length,
                                                               goal_norm,
                                                               center_goal_norm,
                                                               record_dp,
                                                               np_segments,
                                                               pp_segments,
                                                               pair_slots,
                                                               phase1_base,
                                                               phase2_base,
                                                               phase3_base,
                                                               phase4_base,
                                                               phase5_base,
                                                               scratch->det_counts);
            CUDA_TRY(cudaGetLastError());
        }

        h_det_counts = (uint32_t *)malloc(count_bytes);
        h_det_offsets = (uint32_t *)malloc(count_bytes);
        if (!h_det_counts || !h_det_offsets) {
            set_plain_error("deterministic result host count allocation failed");
            goto fail;
        }

        CUDA_TRY(cudaMemcpyAsync(h_det_counts,
                                 scratch->det_counts,
                                 count_bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream));
        if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_DET_COUNT)) goto fail;
        CUDA_TRY(cudaStreamSynchronize(stream));

        uint64_t total_results = 0;
        for (size_t i = 0; i < (size_t)count_slots; i++) {
            h_det_offsets[i] = total_results > 0xffffffffull ? 0xffffffffu : (uint32_t)total_results;
            total_results += (uint64_t)h_det_counts[i];
        }

        if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_DET_FILL)) goto fail;
        CUDA_TRY(cudaMemcpyAsync(scratch->det_offsets,
                                 h_det_offsets,
                                 count_bytes,
                                 cudaMemcpyHostToDevice,
                                 stream));
        CUDA_TRY(cudaMemsetAsync(scratch->overflow, 0, sizeof(int), stream));

        if (total_results) {
            uint32_t grid = pair_slots > 65535ull ? 65535u : (uint32_t)pair_slots;
            if (grid == 0) grid = 1;
            bgj_cuda_search_det_pair_fill_kernel<<<grid,
                                                    BGJ_CUDA_SEARCH_DET_THREADS,
                                                    0,
                                                    stream>>>(scratch->p_vecs,
                                                              scratch->n_vecs,
                                                              scratch->p_ids,
                                                              scratch->n_ids,
                                                              scratch->p_norm,
                                                              scratch->n_norm,
                                                              scratch->p_dot,
                                                              scratch->n_dot,
                                                              num_p,
                                                              num_n,
                                                              vec_length,
                                                              goal_norm,
                                                              center_goal_norm,
                                                              record_dp,
                                                              np_segments,
                                                              pp_segments,
                                                              pair_slots,
                                                              phase1_base,
                                                              phase2_base,
                                                              phase3_base,
                                                              phase4_base,
                                                              phase5_base,
                                                              scratch->det_offsets,
                                                              scratch->results,
                                                              result_capacity,
                                                              scratch->overflow);
            CUDA_TRY(cudaGetLastError());
        }

        {
            int device_overflow = 0;
            CUDA_TRY(cudaMemcpyAsync(&device_overflow,
                                     scratch->overflow,
                                     sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream));
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_DET_FILL)) goto fail;
            CUDA_TRY(cudaStreamSynchronize(stream));
            *submitted_result_count = total_results > 0xffffffffull ? 0xffffffffu : (uint32_t)total_results;
            *submitted_overflow =
                device_overflow || total_results > (uint64_t)result_capacity ||
                total_results > 0xffffffffull;
            *scratch->host_result_count = *submitted_result_count;
            *scratch->host_overflow = *submitted_overflow;
        }

        free(h_det_counts);
        free(h_det_offsets);
        h_det_counts = NULL;
        h_det_offsets = NULL;
        set_plain_error("no CUDA error");
        return 1;
    }

    if (tensor_split_active &&
        bgj_cuda_tensor_requested() &&
        bgj_cuda_tensor_capable() &&
        bgj_cuda_tensor_reorder_requested() &&
        vec_length % 16 == 0) {
        const uint32_t k_blocks = vec_length / 16u;
        tensor_np_min_tiles = bgj_cuda_tensor_np_min_tiles();
        tensor_same_min_tiles = bgj_cuda_tensor_same_min_tiles();

        if (num_p >= 16 && num_n >= 16) {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NP)) goto fail;
            tensor_np_num_p = (num_p / 16u) * 16u;
            tensor_np_num_n = (num_n / 16u) * 16u;
            const uint32_t tensor_blocks_p = tensor_np_num_p / 16u;
            const uint32_t tensor_blocks_n = tensor_np_num_n / 16u;
            const uint64_t tensor_tiles =
                (uint64_t)tensor_blocks_p * (uint64_t)tensor_blocks_n;
            int tensor_np_launched = 0;
            if (tensor_tiles >= tensor_np_min_tiles &&
                bgj_cuda_tensor_np_wide_requested() &&
                tensor_blocks_p >= BGJ_CUDA_TENSOR_NP_WIDE_TILES &&
                tensor_blocks_n >= BGJ_CUDA_TENSOR_NP_WIDE_TILES) {
                const uint32_t wide_blocks_p =
                    (tensor_blocks_p / BGJ_CUDA_TENSOR_NP_WIDE_TILES) *
                    BGJ_CUDA_TENSOR_NP_WIDE_TILES;
                const uint32_t wide_blocks_n =
                    (tensor_blocks_n / BGJ_CUDA_TENSOR_NP_WIDE_TILES) *
                    BGJ_CUDA_TENSOR_NP_WIDE_TILES;
                const uint32_t wide_group_blocks_n = wide_blocks_n / BGJ_CUDA_TENSOR_NP_WIDE_TILES;
                const uint64_t wide_groups =
                    (uint64_t)(wide_blocks_p / BGJ_CUDA_TENSOR_NP_WIDE_TILES) *
                    (uint64_t)wide_group_blocks_n;
                const uint64_t p_frag_words =
                    (uint64_t)wide_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_A_FRAG_WORDS;
                const uint64_t n_frag_words =
                    (uint64_t)wide_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_B_FRAG_WORDS;
                const uint64_t p_pack_grid =
                    ((uint64_t)wide_blocks_p * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint64_t n_pack_grid =
                    ((uint64_t)wide_blocks_n * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint64_t group_begin =
                    ((uint64_t)tensor_split_slot * wide_groups) / (uint64_t)tensor_split_count;
                const uint64_t group_end =
                    ((uint64_t)(tensor_split_slot + 1u) * wide_groups) / (uint64_t)tensor_split_count;
                const uint64_t group_count = group_end > group_begin ? group_end - group_begin : 0;
                if (wide_groups <= 0x7fffffffu &&
                    group_count <= 0x7fffffffu &&
                    p_pack_grid <= 0x7fffffffu &&
                    n_pack_grid <= 0x7fffffffu) {
                    CUDA_ENSURE(scratch->p_tensor_a_frags,
                                scratch->p_tensor_a_frag_capacity,
                                (size_t)p_frag_words * sizeof(uint64_t));
                    CUDA_ENSURE(scratch->n_tensor_b_frags,
                                scratch->n_tensor_b_frag_capacity,
                                (size_t)n_frag_words * sizeof(uint64_t));
                    bgj_cuda_pack_a_frag_kernel<<<(uint32_t)p_pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->p_vecs,
                                                            wide_blocks_p,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->p_tensor_a_frags);
                    CUDA_TRY(cudaGetLastError());
                    bgj_cuda_pack_b_frag_kernel<<<(uint32_t)n_pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->n_vecs,
                                                            wide_blocks_n,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->n_tensor_b_frags);
                    CUDA_TRY(cudaGetLastError());
                    if (group_count) {
                        bgj_cuda_search_np_tensor_wide_reordered_range_kernel<<<(uint32_t)group_count,
                                                                                BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                                0,
                                                                                stream>>>(scratch->p_tensor_a_frags,
                                                                                          scratch->n_tensor_b_frags,
                                                                                          scratch->p_ids,
                                                                                          scratch->n_ids,
                                                                                          scratch->p_norm,
                                                                                          scratch->n_norm,
                                                                                          scratch->p_dot,
                                                                                          scratch->n_dot,
                                                                                          wide_group_blocks_n,
                                                                                          group_begin,
                                                                                          group_count,
                                                                                          k_blocks,
                                                                                          goal_norm,
                                                                                          goal_norm - center_norm,
                                                                                          record_dp,
                                                                                          raw_center_dp,
                                                                                          scratch->results,
                                                                                          result_capacity,
                                                                                          scratch->result_count,
                                                                                          scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    }
                    tensor_np_num_p = wide_blocks_p * 16u;
                    tensor_np_num_n = wide_blocks_n * 16u;
                    tensor_np_launched = 1;
                }
            }
            if (!tensor_np_launched &&
                tensor_tiles >= tensor_np_min_tiles &&
                tensor_tiles <= 0xffffffffULL) {
                const uint64_t p_frag_words =
                    (uint64_t)tensor_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_A_FRAG_WORDS;
                const uint64_t n_frag_words =
                    (uint64_t)tensor_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_B_FRAG_WORDS;
                const uint64_t p_pack_grid =
                    ((uint64_t)tensor_blocks_p * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint64_t n_pack_grid =
                    ((uint64_t)tensor_blocks_n * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint64_t tile_begin =
                    ((uint64_t)tensor_split_slot * tensor_tiles) / (uint64_t)tensor_split_count;
                const uint64_t tile_end =
                    ((uint64_t)(tensor_split_slot + 1u) * tensor_tiles) / (uint64_t)tensor_split_count;
                const uint64_t tile_count = tile_end > tile_begin ? tile_end - tile_begin : 0;
                const uint64_t tensor_grid =
                    (tile_count + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                if (p_pack_grid <= 0x7fffffffu &&
                    n_pack_grid <= 0x7fffffffu &&
                    tensor_grid <= 0x7fffffffu) {
                    CUDA_ENSURE(scratch->p_tensor_a_frags,
                                scratch->p_tensor_a_frag_capacity,
                                (size_t)p_frag_words * sizeof(uint64_t));
                    CUDA_ENSURE(scratch->n_tensor_b_frags,
                                scratch->n_tensor_b_frag_capacity,
                                (size_t)n_frag_words * sizeof(uint64_t));
                    bgj_cuda_pack_a_frag_kernel<<<(uint32_t)p_pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->p_vecs,
                                                            tensor_blocks_p,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->p_tensor_a_frags);
                    CUDA_TRY(cudaGetLastError());
                    bgj_cuda_pack_b_frag_kernel<<<(uint32_t)n_pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->n_vecs,
                                                            tensor_blocks_n,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->n_tensor_b_frags);
                    CUDA_TRY(cudaGetLastError());
                    if (tile_count) {
                        bgj_cuda_search_np_tensor_reordered_range_kernel<<<(uint32_t)tensor_grid,
                                                                           BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                           0,
                                                                           stream>>>(scratch->p_tensor_a_frags,
                                                                                     scratch->n_tensor_b_frags,
                                                                                     scratch->p_ids,
                                                                                     scratch->n_ids,
                                                                                     scratch->p_norm,
                                                                                     scratch->n_norm,
                                                                                     scratch->p_dot,
                                                                                     scratch->n_dot,
                                                                                     tensor_blocks_n,
                                                                                     tile_begin,
                                                                                     tile_count,
                                                                                     k_blocks,
                                                                                     goal_norm,
                                                                                     goal_norm - center_norm,
                                                                                     record_dp,
                                                                                     raw_center_dp,
                                                                                     scratch->results,
                                                                                     result_capacity,
                                                                                     scratch->result_count,
                                                                                     scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    }
                } else {
                    tensor_np_num_p = 0;
                    tensor_np_num_n = 0;
                }
            } else {
                if (!tensor_np_launched) {
                    tensor_np_num_p = 0;
                    tensor_np_num_n = 0;
                }
            }
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NP)) goto fail;
        }

        if (bgj_cuda_tensor_same_requested() && num_p >= 16) {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_PP)) goto fail;
            tensor_same_num_p = (num_p / 16u) * 16u;
            const uint32_t tensor_blocks_p = tensor_same_num_p / 16u;
            const uint64_t tensor_tiles_p =
                ((uint64_t)tensor_blocks_p * (tensor_blocks_p + 1u)) / 2u;
            if (tensor_blocks_p >= tensor_same_min_tiles &&
                tensor_tiles_p <= 0x7fffffffu) {
                const uint64_t a_frag_words =
                    (uint64_t)tensor_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_A_FRAG_WORDS;
                const uint64_t b_frag_words =
                    (uint64_t)tensor_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_B_FRAG_WORDS;
                const uint64_t pack_grid =
                    ((uint64_t)tensor_blocks_p * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint64_t tile_begin =
                    ((uint64_t)tensor_split_slot * tensor_tiles_p) / (uint64_t)tensor_split_count;
                const uint64_t tile_end =
                    ((uint64_t)(tensor_split_slot + 1u) * tensor_tiles_p) / (uint64_t)tensor_split_count;
                const uint64_t tile_count = tile_end > tile_begin ? tile_end - tile_begin : 0;
                const uint64_t tensor_grid =
                    (tile_count + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                if (pack_grid <= 0x7fffffffu && tensor_grid <= 0x7fffffffu) {
                    CUDA_ENSURE(scratch->p_tensor_a_frags,
                                scratch->p_tensor_a_frag_capacity,
                                (size_t)a_frag_words * sizeof(uint64_t));
                    CUDA_ENSURE(scratch->p_tensor_b_frags,
                                scratch->p_tensor_b_frag_capacity,
                                (size_t)b_frag_words * sizeof(uint64_t));
                    bgj_cuda_pack_a_frag_kernel<<<(uint32_t)pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->p_vecs,
                                                            tensor_blocks_p,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->p_tensor_a_frags);
                    CUDA_TRY(cudaGetLastError());
                    bgj_cuda_pack_b_frag_kernel<<<(uint32_t)pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->p_vecs,
                                                            tensor_blocks_p,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->p_tensor_b_frags);
                    CUDA_TRY(cudaGetLastError());
                    if (tile_count) {
                        bgj_cuda_search_same_tensor_reordered_range_kernel<<<(uint32_t)tensor_grid,
                                                                             BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                             0,
                                                                             stream>>>(scratch->p_tensor_a_frags,
                                                                                       scratch->p_tensor_b_frags,
                                                                                       scratch->p_ids,
                                                                                       scratch->p_norm,
                                                                                       scratch->p_dot,
                                                                                       tensor_blocks_p,
                                                                                       tile_begin,
                                                                                       tile_count,
                                                                                       k_blocks,
                                                                                       goal_norm,
                                                                                       goal_norm - center_norm,
                                                                                       record_dp,
                                                                                       raw_center_dp,
                                                                                       0,
                                                                                       scratch->results,
                                                                                       result_capacity,
                                                                                       scratch->result_count,
                                                                                       scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    }
                } else {
                    tensor_same_num_p = 0;
                }
            } else {
                tensor_same_num_p = 0;
            }
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_PP)) goto fail;
        }

        if (bgj_cuda_tensor_same_requested() && num_n >= 16) {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NN)) goto fail;
            tensor_same_num_n = (num_n / 16u) * 16u;
            const uint32_t tensor_blocks_n = tensor_same_num_n / 16u;
            const uint64_t tensor_tiles_n =
                ((uint64_t)tensor_blocks_n * (tensor_blocks_n + 1u)) / 2u;
            if (tensor_blocks_n >= tensor_same_min_tiles &&
                tensor_tiles_n <= 0x7fffffffu) {
                const uint64_t a_frag_words =
                    (uint64_t)tensor_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_A_FRAG_WORDS;
                const uint64_t b_frag_words =
                    (uint64_t)tensor_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                    BGJ_CUDA_I8_B_FRAG_WORDS;
                const uint64_t pack_grid =
                    ((uint64_t)tensor_blocks_n * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint64_t tile_begin =
                    ((uint64_t)tensor_split_slot * tensor_tiles_n) / (uint64_t)tensor_split_count;
                const uint64_t tile_end =
                    ((uint64_t)(tensor_split_slot + 1u) * tensor_tiles_n) / (uint64_t)tensor_split_count;
                const uint64_t tile_count = tile_end > tile_begin ? tile_end - tile_begin : 0;
                const uint64_t tensor_grid =
                    (tile_count + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                if (pack_grid <= 0x7fffffffu && tensor_grid <= 0x7fffffffu) {
                    CUDA_ENSURE(scratch->n_tensor_a_frags,
                                scratch->n_tensor_a_frag_capacity,
                                (size_t)a_frag_words * sizeof(uint64_t));
                    CUDA_ENSURE(scratch->n_tensor_b_frags,
                                scratch->n_tensor_b_frag_capacity,
                                (size_t)b_frag_words * sizeof(uint64_t));
                    bgj_cuda_pack_a_frag_kernel<<<(uint32_t)pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->n_vecs,
                                                            tensor_blocks_n,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->n_tensor_a_frags);
                    CUDA_TRY(cudaGetLastError());
                    bgj_cuda_pack_b_frag_kernel<<<(uint32_t)pack_grid,
                                                  BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                  0,
                                                  stream>>>(scratch->n_vecs,
                                                            tensor_blocks_n,
                                                            k_blocks,
                                                            vec_length,
                                                            scratch->n_tensor_b_frags);
                    CUDA_TRY(cudaGetLastError());
                    if (tile_count) {
                        bgj_cuda_search_same_tensor_reordered_range_kernel<<<(uint32_t)tensor_grid,
                                                                             BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                             0,
                                                                             stream>>>(scratch->n_tensor_a_frags,
                                                                                       scratch->n_tensor_b_frags,
                                                                                       scratch->n_ids,
                                                                                       scratch->n_norm,
                                                                                       scratch->n_dot,
                                                                                       tensor_blocks_n,
                                                                                       tile_begin,
                                                                                       tile_count,
                                                                                       k_blocks,
                                                                                       goal_norm,
                                                                                       goal_norm - center_norm,
                                                                                       record_dp,
                                                                                       raw_center_dp,
                                                                                       1,
                                                                                       scratch->results,
                                                                                       result_capacity,
                                                                                       scratch->result_count,
                                                                                       scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    }
                } else {
                    tensor_same_num_n = 0;
                }
            } else {
                tensor_same_num_n = 0;
            }
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NN)) goto fail;
        }
    }

    if (!range_active &&
        !tensor_split_active &&
        bgj_cuda_tensor_requested() &&
        bgj_cuda_tensor_capable() &&
        vec_length % 16 == 0) {
        tensor_np_min_tiles = bgj_cuda_tensor_np_min_tiles();
        tensor_same_min_tiles = bgj_cuda_tensor_same_min_tiles();
        if (num_p >= 16 && num_n >= 16) {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NP)) goto fail;
            tensor_np_num_p = (num_p / 16u) * 16u;
            tensor_np_num_n = (num_n / 16u) * 16u;
            const uint32_t tensor_blocks_p = tensor_np_num_p / 16u;
            const uint32_t tensor_blocks_n = tensor_np_num_n / 16u;
            const uint64_t tensor_tiles = (uint64_t)tensor_blocks_p * (uint64_t)tensor_blocks_n;
            if (tensor_tiles >= tensor_np_min_tiles &&
                bgj_cuda_tensor_np_wide_requested() &&
                tensor_blocks_p >= BGJ_CUDA_TENSOR_NP_WIDE_TILES &&
                tensor_blocks_n >= BGJ_CUDA_TENSOR_NP_WIDE_TILES) {
                const uint32_t wide_blocks_p =
                    (tensor_blocks_p / BGJ_CUDA_TENSOR_NP_WIDE_TILES) *
                    BGJ_CUDA_TENSOR_NP_WIDE_TILES;
                const uint32_t wide_blocks_n =
                    (tensor_blocks_n / BGJ_CUDA_TENSOR_NP_WIDE_TILES) *
                    BGJ_CUDA_TENSOR_NP_WIDE_TILES;
                const uint32_t wide_group_blocks_n = wide_blocks_n / BGJ_CUDA_TENSOR_NP_WIDE_TILES;
                const uint64_t wide_groups =
                    (uint64_t)(wide_blocks_p / BGJ_CUDA_TENSOR_NP_WIDE_TILES) *
                    (uint64_t)wide_group_blocks_n;
                if (wide_groups <= 0x7fffffffu) {
                    int wide_launch_ok = 1;
                    if (bgj_cuda_tensor_reorder_requested()) {
                        const uint32_t k_blocks = vec_length / 16u;
                        const uint64_t p_frag_words =
                            (uint64_t)wide_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                            BGJ_CUDA_I8_A_FRAG_WORDS;
                        const uint64_t n_frag_words =
                            (uint64_t)wide_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                            BGJ_CUDA_I8_B_FRAG_WORDS;
                        const uint64_t p_pack_grid =
                            ((uint64_t)wide_blocks_p * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                            BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                        const uint64_t n_pack_grid =
                            ((uint64_t)wide_blocks_n * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                            BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                        if (p_pack_grid <= 0x7fffffffu && n_pack_grid <= 0x7fffffffu) {
                            CUDA_ENSURE(scratch->p_tensor_a_frags,
                                        scratch->p_tensor_a_frag_capacity,
                                        (size_t)p_frag_words * sizeof(uint64_t));
                            CUDA_ENSURE(scratch->n_tensor_b_frags,
                                        scratch->n_tensor_b_frag_capacity,
                                        (size_t)n_frag_words * sizeof(uint64_t));
                            bgj_cuda_pack_a_frag_kernel<<<(uint32_t)p_pack_grid,
                                                          BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                          0,
                                                          stream>>>(scratch->p_vecs,
                                                                    wide_blocks_p,
                                                                    k_blocks,
                                                                    vec_length,
                                                                    scratch->p_tensor_a_frags);
                            CUDA_TRY(cudaGetLastError());
                            bgj_cuda_pack_b_frag_kernel<<<(uint32_t)n_pack_grid,
                                                          BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                          0,
                                                          stream>>>(scratch->n_vecs,
                                                                    wide_blocks_n,
                                                                    k_blocks,
                                                                    vec_length,
                                                                    scratch->n_tensor_b_frags);
                            CUDA_TRY(cudaGetLastError());
                            bgj_cuda_search_np_tensor_wide_reordered_kernel<<<(uint32_t)wide_groups,
                                                                              BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                              0,
                                                                              stream>>>(scratch->p_tensor_a_frags,
                                                                                        scratch->n_tensor_b_frags,
                                                                                        scratch->p_ids,
                                                                                        scratch->n_ids,
                                                                                        scratch->p_norm,
                                                                                        scratch->n_norm,
                                                                                        scratch->p_dot,
                                                                                        scratch->n_dot,
                                                                                        wide_group_blocks_n,
                                                                                        k_blocks,
                                                                                        goal_norm,
                                                                                        goal_norm - center_norm,
                                                                                        record_dp,
                                                                                        raw_center_dp,
                                                                                        scratch->results,
                                                                                        result_capacity,
                                                                                        scratch->result_count,
                                                                                        scratch->overflow);
                            CUDA_TRY(cudaGetLastError());
                        } else {
                            wide_launch_ok = 0;
                        }
                    } else {
                        bgj_cuda_search_np_tensor_wide_kernel<<<(uint32_t)wide_groups,
                                                                BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                0,
                                                                stream>>>(scratch->p_vecs,
                                                                          scratch->n_vecs,
                                                                          scratch->p_ids,
                                                                          scratch->n_ids,
                                                                          scratch->p_norm,
                                                                          scratch->n_norm,
                                                                          scratch->p_dot,
                                                                          scratch->n_dot,
                                                                          wide_group_blocks_n,
                                                                          vec_length,
                                                                          goal_norm,
                                                                          goal_norm - center_norm,
                                                                          record_dp,
                                                                          raw_center_dp,
                                                                          scratch->results,
                                                                          result_capacity,
                                                                          scratch->result_count,
                                                                          scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    }
                    if (wide_launch_ok) {
                        tensor_np_num_p = wide_blocks_p * 16u;
                        tensor_np_num_n = wide_blocks_n * 16u;
                    } else {
                        tensor_np_num_p = 0;
                        tensor_np_num_n = 0;
                    }
                } else {
                    tensor_np_num_p = 0;
                    tensor_np_num_n = 0;
                }
            } else if (tensor_tiles >= tensor_np_min_tiles &&
                       bgj_cuda_tensor_np_shared_a_requested() &&
                       tensor_blocks_n >= BGJ_CUDA_TENSOR_WARPS_PER_BLOCK) {
                const uint32_t shared_blocks_n =
                    (tensor_blocks_n / BGJ_CUDA_TENSOR_WARPS_PER_BLOCK) *
                    BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint32_t shared_group_blocks_n = shared_blocks_n / BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                const uint64_t shared_groups = (uint64_t)tensor_blocks_p * (uint64_t)shared_group_blocks_n;
                if (shared_groups <= 0x7fffffffu) {
                    bgj_cuda_search_np_tensor_shared_a_kernel<<<(uint32_t)shared_groups,
                                                                BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                0,
                                                                stream>>>(scratch->p_vecs,
                                                                          scratch->n_vecs,
                                                                          scratch->p_ids,
                                                                          scratch->n_ids,
                                                                          scratch->p_norm,
                                                                          scratch->n_norm,
                                                                          scratch->p_dot,
                                                                          scratch->n_dot,
                                                                          shared_group_blocks_n,
                                                                          vec_length,
                                                                          goal_norm,
                                                                          goal_norm - center_norm,
                                                                          record_dp,
                                                                          raw_center_dp,
                                                                          scratch->results,
                                                                          result_capacity,
                                                                          scratch->result_count,
                                                                          scratch->overflow);
                    CUDA_TRY(cudaGetLastError());
                    tensor_np_num_p = tensor_blocks_p * 16u;
                    tensor_np_num_n = shared_blocks_n * 16u;
                } else {
                    tensor_np_num_p = 0;
                    tensor_np_num_n = 0;
                }
            } else if (tensor_tiles >= tensor_np_min_tiles &&
                       bgj_cuda_tensor_reorder_requested()) {
                const uint32_t k_blocks = vec_length / 16u;
                if (bgj_cuda_tensor_np_multi_requested() &&
                    tensor_blocks_p >= BGJ_CUDA_TENSOR_NP_MULTI_P_TILES &&
                    tensor_blocks_n >= BGJ_CUDA_TENSOR_NP_MULTI_N_TILES) {
                    const uint32_t multi_blocks_p =
                        (tensor_blocks_p / BGJ_CUDA_TENSOR_NP_MULTI_P_TILES) *
                        BGJ_CUDA_TENSOR_NP_MULTI_P_TILES;
                    const uint32_t multi_blocks_n =
                        (tensor_blocks_n / BGJ_CUDA_TENSOR_NP_MULTI_N_TILES) *
                        BGJ_CUDA_TENSOR_NP_MULTI_N_TILES;
                    const uint32_t multi_group_blocks_n =
                        multi_blocks_n / BGJ_CUDA_TENSOR_NP_MULTI_N_TILES;
                    const uint64_t multi_groups =
                        (uint64_t)(multi_blocks_p / BGJ_CUDA_TENSOR_NP_MULTI_P_TILES) *
                        (uint64_t)multi_group_blocks_n;
                    const uint64_t p_frag_words =
                        (uint64_t)multi_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_A_FRAG_WORDS;
                    const uint64_t n_frag_words =
                        (uint64_t)multi_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_B_FRAG_WORDS;
                    const uint64_t p_pack_grid =
                        ((uint64_t)multi_blocks_p * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                        BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                    const uint64_t n_pack_grid =
                        ((uint64_t)multi_blocks_n * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                        BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                    if (multi_groups <= 0x7fffffffu &&
                        p_pack_grid <= 0x7fffffffu &&
                        n_pack_grid <= 0x7fffffffu) {
                        CUDA_ENSURE(scratch->p_tensor_a_frags,
                                    scratch->p_tensor_a_frag_capacity,
                                    (size_t)p_frag_words * sizeof(uint64_t));
                        CUDA_ENSURE(scratch->n_tensor_b_frags,
                                    scratch->n_tensor_b_frag_capacity,
                                    (size_t)n_frag_words * sizeof(uint64_t));
                        bgj_cuda_pack_a_frag_kernel<<<(uint32_t)p_pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->p_vecs,
                                                                multi_blocks_p,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->p_tensor_a_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_pack_b_frag_kernel<<<(uint32_t)n_pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->n_vecs,
                                                                multi_blocks_n,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->n_tensor_b_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_search_np_tensor_multi_kernel<<<(uint32_t)multi_groups,
                                                                 BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                 0,
                                                                 stream>>>(scratch->p_tensor_a_frags,
                                                                           scratch->n_tensor_b_frags,
                                                                           scratch->p_ids,
                                                                           scratch->n_ids,
                                                                           scratch->p_norm,
                                                                           scratch->n_norm,
                                                                           scratch->p_dot,
                                                                           scratch->n_dot,
                                                                           multi_group_blocks_n,
                                                                           k_blocks,
                                                                           goal_norm,
                                                                           goal_norm - center_norm,
                                                                           record_dp,
                                                                           raw_center_dp,
                                                                           scratch->results,
                                                                           result_capacity,
                                                                           scratch->result_count,
                                                                           scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                        tensor_np_num_p = multi_blocks_p * 16u;
                        tensor_np_num_n = multi_blocks_n * 16u;
                    } else {
                        tensor_np_num_p = 0;
                        tensor_np_num_n = 0;
                    }
                } else {
                    const uint64_t p_frag_words =
                        (uint64_t)tensor_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_A_FRAG_WORDS;
                    const uint64_t n_frag_words =
                        (uint64_t)tensor_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_B_FRAG_WORDS;
                    const uint64_t tensor_grid = (tensor_tiles + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                                                 BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                    const uint64_t p_pack_grid =
                        ((uint64_t)tensor_blocks_p * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                        BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                    const uint64_t n_pack_grid =
                        ((uint64_t)tensor_blocks_n * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                        BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                    if (tensor_tiles <= 0xffffffffULL &&
                        tensor_grid <= 0x7fffffffu &&
                        p_pack_grid <= 0x7fffffffu &&
                        n_pack_grid <= 0x7fffffffu) {
                        CUDA_ENSURE(scratch->p_tensor_a_frags,
                                    scratch->p_tensor_a_frag_capacity,
                                    (size_t)p_frag_words * sizeof(uint64_t));
                        CUDA_ENSURE(scratch->n_tensor_b_frags,
                                    scratch->n_tensor_b_frag_capacity,
                                    (size_t)n_frag_words * sizeof(uint64_t));
                        bgj_cuda_pack_a_frag_kernel<<<(uint32_t)p_pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->p_vecs,
                                                                tensor_blocks_p,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->p_tensor_a_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_pack_b_frag_kernel<<<(uint32_t)n_pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->n_vecs,
                                                                tensor_blocks_n,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->n_tensor_b_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_search_np_tensor_reordered_kernel<<<(uint32_t)tensor_grid,
                                                                     BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                     0,
                                                                     stream>>>(scratch->p_tensor_a_frags,
                                                                               scratch->n_tensor_b_frags,
                                                                               scratch->p_ids,
                                                                               scratch->n_ids,
                                                                               scratch->p_norm,
                                                                               scratch->n_norm,
                                                                               scratch->p_dot,
                                                                               scratch->n_dot,
                                                                               tensor_blocks_n,
                                                                               (uint32_t)tensor_tiles,
                                                                               k_blocks,
                                                                               goal_norm,
                                                                               goal_norm - center_norm,
                                                                               record_dp,
                                                                               raw_center_dp,
                                                                               scratch->results,
                                                                               result_capacity,
                                                                               scratch->result_count,
                                                                               scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    } else {
                        tensor_np_num_p = 0;
                        tensor_np_num_n = 0;
                    }
                }
            } else if (tensor_tiles >= tensor_np_min_tiles) {
                const uint64_t tensor_grid = (tensor_tiles + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                                             BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                if (tensor_tiles <= 0xffffffffULL && tensor_grid <= 0x7fffffffu) {
                    bgj_cuda_search_np_tensor_kernel<<<(uint32_t)tensor_grid,
                                                        BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                        0,
                                                        stream>>>(scratch->p_vecs,
                                                                  scratch->n_vecs,
                                                                  scratch->p_ids,
                                                                  scratch->n_ids,
                                                                  scratch->p_norm,
                                                                  scratch->n_norm,
                                                                  scratch->p_dot,
                                                                  scratch->n_dot,
                                                                  tensor_blocks_n,
                                                                  (uint32_t)tensor_tiles,
                                                                  vec_length,
                                                                  goal_norm,
                                                                  goal_norm - center_norm,
                                                                  record_dp,
                                                                  raw_center_dp,
                                                                  scratch->results,
                                                                  result_capacity,
                                                                  scratch->result_count,
                                                                  scratch->overflow);
                    CUDA_TRY(cudaGetLastError());
                } else {
                    tensor_np_num_p = 0;
                    tensor_np_num_n = 0;
                }
            } else {
                tensor_np_num_p = 0;
                tensor_np_num_n = 0;
            }
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NP)) goto fail;
        }

        if (bgj_cuda_tensor_same_requested() && num_p >= 16) {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_PP)) goto fail;
            tensor_same_num_p = (num_p / 16u) * 16u;
            const uint32_t tensor_blocks_p = tensor_same_num_p / 16u;
            const uint64_t tensor_tiles_p = ((uint64_t)tensor_blocks_p * (tensor_blocks_p + 1u)) / 2u;
            if (tensor_blocks_p >= tensor_same_min_tiles && tensor_tiles_p <= 0x7fffffffu) {
                const uint32_t tensor_grid_p =
                    (uint32_t)((tensor_tiles_p + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                               BGJ_CUDA_TENSOR_WARPS_PER_BLOCK);
                if (bgj_cuda_tensor_reorder_requested()) {
                    const uint32_t k_blocks = vec_length / 16u;
                    const uint64_t a_frag_words =
                        (uint64_t)tensor_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_A_FRAG_WORDS;
                    const uint64_t b_frag_words =
                        (uint64_t)tensor_blocks_p * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_B_FRAG_WORDS;
                    const uint64_t pack_grid =
                        ((uint64_t)tensor_blocks_p * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                        BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                    if (pack_grid <= 0x7fffffffu) {
                        CUDA_ENSURE(scratch->p_tensor_a_frags,
                                    scratch->p_tensor_a_frag_capacity,
                                    (size_t)a_frag_words * sizeof(uint64_t));
                        CUDA_ENSURE(scratch->p_tensor_b_frags,
                                    scratch->p_tensor_b_frag_capacity,
                                    (size_t)b_frag_words * sizeof(uint64_t));
                        bgj_cuda_pack_a_frag_kernel<<<(uint32_t)pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->p_vecs,
                                                                tensor_blocks_p,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->p_tensor_a_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_pack_b_frag_kernel<<<(uint32_t)pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->p_vecs,
                                                                tensor_blocks_p,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->p_tensor_b_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_search_same_tensor_reordered_kernel<<<tensor_grid_p,
                                                                       BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                       0,
                                                                       stream>>>(scratch->p_tensor_a_frags,
                                                                                 scratch->p_tensor_b_frags,
                                                                                 scratch->p_ids,
                                                                                 scratch->p_norm,
                                                                                 scratch->p_dot,
                                                                                 tensor_blocks_p,
                                                                                 (uint32_t)tensor_tiles_p,
                                                                                 k_blocks,
                                                                                 goal_norm,
                                                                                 goal_norm - center_norm,
                                                                                 record_dp,
                                                                                 raw_center_dp,
                                                                                 0,
                                                                                 scratch->results,
                                                                                 result_capacity,
                                                                                 scratch->result_count,
                                                                                 scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    } else {
                        tensor_same_num_p = 0;
                    }
                } else {
                    bgj_cuda_search_same_tensor_kernel<<<tensor_grid_p,
                                                         BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                         0,
                                                         stream>>>(scratch->p_vecs,
                                                                   scratch->p_ids,
                                                                   scratch->p_norm,
                                                                   scratch->p_dot,
                                                                   tensor_blocks_p,
                                                                   (uint32_t)tensor_tiles_p,
                                                                   vec_length,
                                                                   goal_norm,
                                                                   goal_norm - center_norm,
                                                                   record_dp,
                                                                   raw_center_dp,
                                                                   0,
                                                                   scratch->results,
                                                                   result_capacity,
                                                                   scratch->result_count,
                                                                   scratch->overflow);
                    CUDA_TRY(cudaGetLastError());
                }
            } else {
                tensor_same_num_p = 0;
            }
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_PP)) goto fail;
        }

        if (bgj_cuda_tensor_same_requested() && num_n >= 16) {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NN)) goto fail;
            tensor_same_num_n = (num_n / 16u) * 16u;
            const uint32_t tensor_blocks_n = tensor_same_num_n / 16u;
            const uint64_t tensor_tiles_n = ((uint64_t)tensor_blocks_n * (tensor_blocks_n + 1u)) / 2u;
            if (tensor_blocks_n >= tensor_same_min_tiles && tensor_tiles_n <= 0x7fffffffu) {
                const uint32_t tensor_grid_n =
                    (uint32_t)((tensor_tiles_n + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                               BGJ_CUDA_TENSOR_WARPS_PER_BLOCK);
                if (bgj_cuda_tensor_reorder_requested()) {
                    const uint32_t k_blocks = vec_length / 16u;
                    const uint64_t a_frag_words =
                        (uint64_t)tensor_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_A_FRAG_WORDS;
                    const uint64_t b_frag_words =
                        (uint64_t)tensor_blocks_n * k_blocks * BGJ_CUDA_WARP_SIZE *
                        BGJ_CUDA_I8_B_FRAG_WORDS;
                    const uint64_t pack_grid =
                        ((uint64_t)tensor_blocks_n * k_blocks + BGJ_CUDA_TENSOR_WARPS_PER_BLOCK - 1u) /
                        BGJ_CUDA_TENSOR_WARPS_PER_BLOCK;
                    if (pack_grid <= 0x7fffffffu) {
                        CUDA_ENSURE(scratch->n_tensor_a_frags,
                                    scratch->n_tensor_a_frag_capacity,
                                    (size_t)a_frag_words * sizeof(uint64_t));
                        CUDA_ENSURE(scratch->n_tensor_b_frags,
                                    scratch->n_tensor_b_frag_capacity,
                                    (size_t)b_frag_words * sizeof(uint64_t));
                        bgj_cuda_pack_a_frag_kernel<<<(uint32_t)pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->n_vecs,
                                                                tensor_blocks_n,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->n_tensor_a_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_pack_b_frag_kernel<<<(uint32_t)pack_grid,
                                                      BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                      0,
                                                      stream>>>(scratch->n_vecs,
                                                                tensor_blocks_n,
                                                                k_blocks,
                                                                vec_length,
                                                                scratch->n_tensor_b_frags);
                        CUDA_TRY(cudaGetLastError());
                        bgj_cuda_search_same_tensor_reordered_kernel<<<tensor_grid_n,
                                                                       BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                                       0,
                                                                       stream>>>(scratch->n_tensor_a_frags,
                                                                                 scratch->n_tensor_b_frags,
                                                                                 scratch->n_ids,
                                                                                 scratch->n_norm,
                                                                                 scratch->n_dot,
                                                                                 tensor_blocks_n,
                                                                                 (uint32_t)tensor_tiles_n,
                                                                                 k_blocks,
                                                                                 goal_norm,
                                                                                 goal_norm - center_norm,
                                                                                 record_dp,
                                                                                 raw_center_dp,
                                                                                 1,
                                                                                 scratch->results,
                                                                                 result_capacity,
                                                                                 scratch->result_count,
                                                                                 scratch->overflow);
                        CUDA_TRY(cudaGetLastError());
                    } else {
                        tensor_same_num_n = 0;
                    }
                } else {
                    bgj_cuda_search_same_tensor_kernel<<<tensor_grid_n,
                                                         BGJ_CUDA_TENSOR_THREADS_PER_BLOCK,
                                                         0,
                                                         stream>>>(scratch->n_vecs,
                                                                   scratch->n_ids,
                                                                   scratch->n_norm,
                                                                   scratch->n_dot,
                                                                   tensor_blocks_n,
                                                                   (uint32_t)tensor_tiles_n,
                                                                   vec_length,
                                                                   goal_norm,
                                                                   goal_norm - center_norm,
                                                                   record_dp,
                                                                   raw_center_dp,
                                                                   1,
                                                                   scratch->results,
                                                                   result_capacity,
                                                                   scratch->result_count,
                                                                   scratch->overflow);
                    CUDA_TRY(cudaGetLastError());
                }
            } else {
                tensor_same_num_n = 0;
            }
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_TENSOR_NN)) goto fail;
        }
    }

    {
        const uint64_t np_total = tensor_np_num_p && tensor_np_num_n ?
                                  (uint64_t)tensor_np_num_p * (uint64_t)(num_n - tensor_np_num_n) +
                                  (uint64_t)(num_p - tensor_np_num_p) * (uint64_t)num_n :
                                  (uint64_t)num_p * (uint64_t)num_n;
        const uint64_t pp_total = tensor_same_num_p ?
                                  (uint64_t)tensor_same_num_p * (uint64_t)(num_p - tensor_same_num_p) +
                                  (uint64_t)(num_p - tensor_same_num_p) * (uint64_t)num_p :
                                  (num_p > 1 ? (uint64_t)num_p * (uint64_t)num_p : 0);
        const uint64_t nn_total = tensor_same_num_n ?
                                  (uint64_t)tensor_same_num_n * (uint64_t)(num_n - tensor_same_num_n) +
                                  (uint64_t)(num_n - tensor_same_num_n) * (uint64_t)num_n :
                                  (num_n > 1 ? (uint64_t)num_n * (uint64_t)num_n : 0);
        const uint64_t total = np_total +
                               pp_total +
                               nn_total;
        if (range_active || tensor_split_active) {
            uint64_t active_offset = 0;
            uint64_t active_count = 0;
            if (range_active && work_offset < total) {
                active_offset = work_offset;
                active_count = work_count;
                if (active_count == 0 || active_count > total - work_offset) {
                    active_count = total - work_offset;
                }
            } else if (tensor_split_active && total) {
                active_offset =
                    ((uint64_t)tensor_split_slot * total) / (uint64_t)tensor_split_count;
                const uint64_t active_end =
                    ((uint64_t)(tensor_split_slot + 1u) * total) / (uint64_t)tensor_split_count;
                active_count = active_end > active_offset ? active_end - active_offset : 0;
            }
            const uint32_t threads = 256;
            uint32_t blocks = (uint32_t)((active_count + threads - 1) / threads);
            if (blocks == 0) blocks = 1;
            if (blocks > 65535) blocks = 65535;
            if (active_count) {
                if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_SCALAR)) goto fail;
                bgj_cuda_search_bucket_range_kernel<<<blocks, threads, 0, stream>>>(scratch->p_vecs,
                                                                                    scratch->n_vecs,
                                                                                    scratch->p_ids,
                                                                                    scratch->n_ids,
                                                                                    scratch->p_norm,
                                                                                    scratch->n_norm,
                                                                                    scratch->p_dot,
                                                                                    scratch->n_dot,
                                                                                    num_p,
                                                                                    num_n,
                                                                                    tensor_np_num_p,
                                                                                    tensor_np_num_n,
                                                                                    tensor_same_num_p,
                                                                                    tensor_same_num_n,
                                                                                    vec_length,
                                                                                    goal_norm,
                                                                                    goal_norm - center_norm,
                                                                                    record_dp,
                                                                                    raw_center_dp,
                                                                                    active_offset,
                                                                                    active_count,
                                                                                    scratch->results,
                                                                                    result_capacity,
                                                                                    scratch->result_count,
                                                                                    scratch->overflow);
                CUDA_TRY(cudaGetLastError());
                if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_SCALAR)) goto fail;
            }
            goto scalar_done;
        }
        const uint32_t threads = 256;
        uint32_t blocks = (uint32_t)((total + threads - 1) / threads);
        if (blocks == 0) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        if (total) {
            if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_SCALAR)) goto fail;
            bgj_cuda_search_bucket_kernel<<<blocks, threads, 0, stream>>>(scratch->p_vecs,
                                                                          scratch->n_vecs,
                                                                          scratch->p_ids,
                                                                          scratch->n_ids,
                                                                          scratch->p_norm,
                                                                          scratch->n_norm,
                                                                          scratch->p_dot,
                                                                          scratch->n_dot,
                                                                          num_p,
                                                                          num_n,
                                                                          tensor_np_num_p,
                                                                          tensor_np_num_n,
                                                                          tensor_same_num_p,
                                                                          tensor_same_num_n,
                                                                          vec_length,
                                                                          goal_norm,
                                                                          goal_norm - center_norm,
                                                                          record_dp,
                                                                          raw_center_dp,
                                                                          scratch->results,
                                                                          result_capacity,
                                                                          scratch->result_count,
                                                                          scratch->overflow);
            CUDA_TRY(cudaGetLastError());
            if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_SCALAR)) goto fail;
        }
    }

scalar_done:
    if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_COUNT_COPY)) goto fail;
    CUDA_TRY(cudaMemcpyAsync(scratch->host_result_count, scratch->result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaMemcpyAsync(scratch->host_overflow, scratch->overflow, sizeof(int), cudaMemcpyDeviceToHost, stream));
    if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_COUNT_COPY)) goto fail;
    set_plain_error("no CUDA error");
    return 1;

fail:
    free(h_det_counts);
    free(h_det_offsets);
    return 0;
}

static int bgj_cuda_finish_submitted_bucket(bgj_cuda_raw_scratch_t *scratch,
                                            bgj_cuda_result_t *results,
                                            uint32_t result_capacity,
                                            uint32_t submitted_result_count,
                                            int submitted_overflow,
                                            uint32_t *result_count,
                                            int *overflow)
{
    cudaStream_t stream = scratch->stream;

    *overflow = submitted_overflow || (submitted_result_count > result_capacity);
    *result_count = submitted_result_count > result_capacity ? result_capacity : submitted_result_count;
    if (*result_count) {
        if (!bgj_cuda_search_profile_begin(scratch, BGJ_CUDA_SEARCH_PHASE_RESULT_COPY)) goto fail;
        CUDA_TRY(cudaMemcpyAsync(results,
                                 scratch->results,
                                 (size_t)(*result_count) * sizeof(bgj_cuda_result_t),
                                 cudaMemcpyDeviceToHost,
                                 stream));
        if (!bgj_cuda_search_profile_end(scratch, BGJ_CUDA_SEARCH_PHASE_RESULT_COPY)) goto fail;
    }

    set_plain_error("no CUDA error");
    return 1;

fail:
    return 0;
}

static int bgj_cuda_result_same_host(const bgj_cuda_result_t &a,
                                     const bgj_cuda_result_t &b)
{
    return a.type == b.type && a.x == b.x && a.y == b.y;
}

static bool bgj_cuda_result_less_host(const bgj_cuda_result_t &a,
                                      const bgj_cuda_result_t &b)
{
    if (a.type != b.type) return a.type < b.type;
    if (a.x != b.x) return a.x < b.x;
    return a.y < b.y;
}

static int bgj_cuda_verify_split_bucket_results(const char *mode,
                                                int verify_device,
                                                const int8_t *pool_vecs_host,
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
                                                int raw_center_dp,
                                                const bgj_cuda_result_t *split_results,
                                                uint32_t split_result_count,
                                                int split_overflow,
                                                uint32_t result_capacity)
{
    if (!bgj_cuda_single_bucket_split_verify_requested()) return 1;
    if (split_overflow) return 1;
    if (!pool_vecs_host || !split_results) return 1;

    int previous_device = -1;
    bgj_cuda_current_device(&previous_device);
    if (verify_device >= 0 && !bgj_cuda_set_current_device(verify_device)) {
        return 0;
    }

    bgj_cuda_result_t *single_results =
        (bgj_cuda_result_t *)malloc((size_t)result_capacity * sizeof(bgj_cuda_result_t));
    bgj_cuda_result_t *split_sorted =
        (bgj_cuda_result_t *)malloc((size_t)split_result_count * sizeof(bgj_cuda_result_t));
    bgj_cuda_result_t *single_sorted = NULL;
    if (!single_results || (!split_sorted && split_result_count)) {
        free(single_results);
        free(split_sorted);
        set_plain_error("CUDA split verifier allocation failed");
        if (previous_device >= 0) bgj_cuda_set_current_device(previous_device);
        return 0;
    }

    bgj_cuda_raw_scratch_t *scratch = &bgj_cuda_raw_scratch;
    uint32_t single_submitted_count = 0;
    uint32_t single_result_count = 0;
    int single_submitted_overflow = 0;
    int single_overflow = 0;
    int ok = 0;

    if (!bgj_cuda_search_bucket_raw_submit(scratch,
                                           NULL,
                                           NULL,
                                           pool_vecs_host,
                                           pool_epoch,
                                           pool_size,
                                           p_ids,
                                           n_ids,
                                           p_norm,
                                           n_norm,
                                           p_dot,
                                           n_dot,
                                           num_p,
                                           num_n,
                                           vec_length,
                                           goal_norm,
                                           center_id,
                                           center_norm,
                                           record_dp,
                                           0,
                                           0,
                                           raw_center_dp,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           result_capacity,
                                           &single_submitted_count,
                                           &single_submitted_overflow)) {
        goto done;
    }
    if (scratch->stream_ready) {
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize split verifier counts", err);
            goto done;
        }
    }
    bgj_cuda_get_submitted_counts(scratch,
                                  &single_submitted_count,
                                  &single_submitted_overflow);
    if (!bgj_cuda_finish_submitted_bucket(scratch,
                                          single_results,
                                          result_capacity,
                                          single_submitted_count,
                                          single_submitted_overflow,
                                          &single_result_count,
                                          &single_overflow)) {
        goto done;
    }
    if (scratch->stream_ready) {
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize split verifier results", err);
            goto done;
        }
    }

    if (single_overflow) {
        ok = 1;
        goto done;
    }

    if (single_result_count != split_result_count) {
        fprintf(stderr,
                "cuda_split_verify: mode=%s count mismatch split=%u single=%u\n",
                mode ? mode : "unknown",
                split_result_count,
                single_result_count);
        set_plain_error("CUDA split verifier count mismatch");
        goto done;
    }

    for (uint32_t i = 0; i < split_result_count; i++) {
        if (!bgj_cuda_result_same_host(split_results[i], single_results[i])) {
            if (bgj_cuda_single_bucket_split_verify_order_requested()) {
                fprintf(stderr,
                        "cuda_split_verify: mode=%s order mismatch index=%u "
                        "split=(%u,%u,%u) single=(%u,%u,%u)\n",
                        mode ? mode : "unknown",
                        i,
                        split_results[i].type,
                        split_results[i].x,
                        split_results[i].y,
                        single_results[i].type,
                        single_results[i].x,
                        single_results[i].y);
                set_plain_error("CUDA split verifier order mismatch");
                goto done;
            }
            break;
        }
    }

    if (split_result_count) {
        memcpy(split_sorted,
               split_results,
               (size_t)split_result_count * sizeof(bgj_cuda_result_t));
        single_sorted =
            (bgj_cuda_result_t *)malloc((size_t)single_result_count * sizeof(bgj_cuda_result_t));
        if (!single_sorted) {
            set_plain_error("CUDA split verifier allocation failed");
            goto done;
        }
        memcpy(single_sorted,
               single_results,
               (size_t)single_result_count * sizeof(bgj_cuda_result_t));
        std::sort(split_sorted,
                  split_sorted + split_result_count,
                  bgj_cuda_result_less_host);
        std::sort(single_sorted,
                  single_sorted + single_result_count,
                  bgj_cuda_result_less_host);
        for (uint32_t i = 0; i < split_result_count; i++) {
            if (!bgj_cuda_result_same_host(split_sorted[i], single_sorted[i])) {
                fprintf(stderr,
                        "cuda_split_verify: mode=%s multiset mismatch index=%u "
                        "split=(%u,%u,%u) single=(%u,%u,%u)\n",
                        mode ? mode : "unknown",
                        i,
                        split_sorted[i].type,
                        split_sorted[i].x,
                        split_sorted[i].y,
                        single_sorted[i].type,
                        single_sorted[i].x,
                        single_sorted[i].y);
                set_plain_error("CUDA split verifier multiset mismatch");
                goto done;
            }
        }
    }

    ok = 1;

done:
    free(single_results);
    free(split_sorted);
    free(single_sorted);
    if (previous_device >= 0) bgj_cuda_set_current_device(previous_device);
    return ok;
}

static uint64_t bgj_cuda_full_pair_work(uint32_t num_p, uint32_t num_n)
{
    return (uint64_t)num_p * (uint64_t)num_n +
           (num_p > 1 ? (uint64_t)num_p * (uint64_t)num_p : 0ull) +
           (num_n > 1 ? (uint64_t)num_n * (uint64_t)num_n : 0ull);
}

static uint64_t bgj_cuda_scale_work(uint64_t total, uint64_t part, uint64_t full)
{
    if (!full || !part || !total) return 0;
    long double value = (long double)total * (long double)part / (long double)full;
    if (value < 1.0L) return 1;
    if (value > (long double)0xffffffffffffffffull) return 0xffffffffffffffffull;
    return (uint64_t)value;
}

static void bgj_cuda_split_profile_print_phases(FILE *out,
                                                const bgj_cuda_search_phase_profile_t *profile)
{
    if (!out || !profile) return;
    for (int phase = 0; phase < BGJ_CUDA_SEARCH_PHASE_COUNT; phase++) {
        if (profile->sec[phase] > 0.0) {
            fprintf(out, " %s=%.6fs",
                    bgj_cuda_search_phase_name(phase),
                    profile->sec[phase]);
        }
    }
}

static int bgj_cuda_search_bucket_pool_raw_split_impl(const int8_t *pool_vecs_host,
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
                                                      int raw_center_dp,
                                                      int tensor_split,
                                                      bgj_cuda_result_t *results,
                                                      uint32_t result_capacity,
                                                      uint32_t *result_count,
                                                      int *overflow)
{
    if (!pool_vecs_host || !results || !result_count || !overflow) return 0;
    if (bgj_cuda_deterministic_results_requested()) return 0;

    int selected_device = -1;
    bgj_cuda_current_device(&selected_device);

    int devices[BGJ_CUDA_MAX_EXEC_DEVICES];
    int device_count =
        bgj_cuda_copy_execution_devices(devices, BGJ_CUDA_MAX_EXEC_DEVICES);
    if (device_count <= 1) return 0;

    const uint64_t estimated_dots = bgj_cuda_estimated_pair_dots(num_p, num_n);
    const uint64_t full_work = bgj_cuda_full_pair_work(num_p, num_n);
    if (!full_work || estimated_dots < bgj_cuda_single_bucket_split_min_dots()) {
        return 0;
    }
    const int split_profile =
        bgj_cuda_split_profile_requested() &&
        estimated_dots >= bgj_cuda_split_profile_min_dots();
    if ((uint64_t)device_count > full_work) device_count = (int)full_work;
    if (device_count <= 1) return 0;

    uint64_t work_begin[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint64_t work_count[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint32_t submitted_result_count[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    uint32_t copied_result_count[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    int submitted_overflow[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    int copied_overflow[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
    bgj_cuda_search_phase_profile_t phase_profile_sum = {};
    double submit_sec = 0.0;
    double wait_sec = 0.0;
    double copy_sec = 0.0;
    double total_sec = 0.0;
    double wait_t0 = 0.0;
    double copy_t0 = 0.0;
    uint32_t out_count = 0;
    int out_overflow = 0;

    for (int i = 0; i < device_count; i++) {
        const uint64_t begin = ((uint64_t)i * full_work) / (uint64_t)device_count;
        const uint64_t end = ((uint64_t)(i + 1) * full_work) / (uint64_t)device_count;
        work_begin[i] = begin;
        work_count[i] = end > begin ? end - begin : 0;
    }

    const double total_t0 = bgj_cuda_wall_time();
    const double submit_t0 = bgj_cuda_wall_time();
    uint32_t submitted = 0;
    for (int i = 0; i < device_count; i++) {
        if (!work_count[i]) continue;
        if (!bgj_cuda_set_current_device(devices[i])) goto fail_sync;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch) {
            set_plain_error("out of host memory");
            goto fail_sync;
        }
        const uint64_t split_work_begin = tensor_split ? 0 : work_begin[i];
        const uint64_t split_work_count = tensor_split ? 0 : work_count[i];
        const uint32_t split_tensor_slot = tensor_split ? (uint32_t)i : 0u;
        const uint32_t split_tensor_count = tensor_split ? (uint32_t)device_count : 0u;
        if (!bgj_cuda_search_bucket_raw_submit(scratch,
                                               NULL,
                                               NULL,
                                               pool_vecs_host,
                                               pool_epoch,
                                               pool_size,
                                               p_ids,
                                               n_ids,
                                               p_norm,
                                               n_norm,
                                               p_dot,
                                               n_dot,
                                               num_p,
                                               num_n,
                                               vec_length,
                                               goal_norm,
                                               center_id,
                                               center_norm,
                                               record_dp,
                                               0,
                                               0,
                                               raw_center_dp,
                                               0,
                                               split_work_begin,
                                               split_work_count,
                                               split_tensor_slot,
                                               split_tensor_count,
                                               result_capacity,
                                               &submitted_result_count[i],
                                               &submitted_overflow[i])) {
            goto fail_sync;
        }
        submitted++;
    }
    submit_sec = bgj_cuda_wall_time() - submit_t0;
    if (!submitted) return 0;

    wait_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize split bucket counts", err);
            goto fail_sync;
        }
        bgj_cuda_get_submitted_counts(scratch,
                                      &submitted_result_count[i],
                                      &submitted_overflow[i]);
    }
    wait_sec = bgj_cuda_wall_time() - wait_t0;

    copy_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        const uint32_t remaining =
            out_count < result_capacity ? result_capacity - out_count : 0u;
        if (!bgj_cuda_finish_submitted_bucket(scratch,
                                              results + out_count,
                                              remaining,
                                              submitted_result_count[i],
                                              submitted_overflow[i],
                                              &copied_result_count[i],
                                              &copied_overflow[i])) {
            goto fail_sync;
        }
        out_count += copied_result_count[i];
        if (copied_overflow[i]) out_overflow = 1;
    }

    for (int i = 0; i < device_count; i++) {
        if (!work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize split bucket results", err);
            goto fail_sync;
        }
        if (!bgj_cuda_search_profile_collect(scratch)) goto fail_sync;
        if (scratch->phase_profile_active) {
            bgj_cuda_search_phase_profile_add(&phase_profile_sum,
                                              &scratch->last_phase_profile);
        }
    }
    copy_sec = bgj_cuda_wall_time() - copy_t0;

    if (!bgj_cuda_verify_split_bucket_results(tensor_split ? "tensor" : "range",
                                              selected_device >= 0 ? selected_device : devices[0],
                                              pool_vecs_host,
                                              pool_epoch,
                                              pool_size,
                                              p_ids,
                                              n_ids,
                                              p_norm,
                                              n_norm,
                                              p_dot,
                                              n_dot,
                                              num_p,
                                              num_n,
                                              vec_length,
                                              goal_norm,
                                              center_id,
                                              center_norm,
                                              record_dp,
                                              raw_center_dp,
                                              results,
                                              out_count,
                                              out_overflow,
                                              result_capacity)) {
        goto fail_sync;
    }

    *result_count = out_count;
    *overflow = out_overflow;
    set_plain_error("no CUDA error");

    total_sec = bgj_cuda_wall_time() - total_t0;
    if (split_profile) {
        fprintf(stderr,
                "cuda_split_profile: mode=%s devices=%d num_p=%u num_n=%u dots=%lu "
                "full_work=%lu results=%u overflow=%d submit=%.6fs wait=%.6fs "
                "copy=%.6fs total=%.6fs",
                tensor_split ? "tensor" : "range",
                device_count,
                num_p,
                num_n,
                (unsigned long)estimated_dots,
                (unsigned long)full_work,
                out_count,
                out_overflow,
                submit_sec,
                wait_sec,
                copy_sec,
                total_sec);
        bgj_cuda_split_profile_print_phases(stderr, &phase_profile_sum);
        fprintf(stderr, "\n");
        for (int i = 0; i < device_count; i++) {
            if (!work_count[i]) continue;
            bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
            const uint64_t device_dots =
                bgj_cuda_scale_work(estimated_dots, work_count[i], full_work);
            fprintf(stderr,
                    "cuda_split_device_profile: mode=%s slot=%d/%d device=%d "
                    "work_begin=%lu work_count=%lu dots=%lu submitted=%u copied=%u "
                    "submitted_overflow=%d copied_overflow=%d",
                    tensor_split ? "tensor" : "range",
                    i,
                    device_count,
                    devices[i],
                    (unsigned long)work_begin[i],
                    (unsigned long)work_count[i],
                    (unsigned long)device_dots,
                    submitted_result_count[i],
                    copied_result_count[i],
                    submitted_overflow[i],
                    copied_overflow[i]);
            if (scratch && scratch->phase_profile_active) {
                bgj_cuda_split_profile_print_phases(stderr, &scratch->last_phase_profile);
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }

    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_SINGLE,
                                   selected_device >= 0 ? selected_device : devices[0],
                                   1,
                                   0,
                                   num_p,
                                   num_n,
                                   estimated_dots,
                                   out_count,
                                   out_overflow,
                                   0,
                                   submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   total_sec,
                                   &phase_profile_sum);
    for (int i = 0; i < device_count; i++) {
        if (!work_count[i]) continue;
        const uint64_t device_dots =
            bgj_cuda_scale_work(estimated_dots, work_count[i], full_work);
        bgj_cuda_device_profile_record(devices[i],
                                       1,
                                       device_dots,
                                       submitted_result_count[i],
                                       copied_overflow[i] ? 1 : 0,
                                       0,
                                       submit_sec,
                                       wait_sec,
                                       copy_sec,
                                       total_sec);
    }
    bgj_cuda_phase_profile_record(1,
                                  estimated_dots,
                                  out_count,
                                  out_overflow,
                                  0,
                                  &phase_profile_sum);
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 1;

fail_sync:
    for (int i = 0; i < device_count; i++) {
        if (!work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (scratch && scratch->stream_ready) {
            if (scratch->device >= 0) cudaSetDevice(scratch->device);
            cudaStreamSynchronize(scratch->stream);
        }
    }
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 0;
}

static int bgj_cuda_search_bucket_pool_raw_split_submit_async(const int8_t *pool_vecs_host,
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
                                                              int tensor_split,
                                                              uint32_t result_capacity,
                                                              uint32_t *submitted_result_count,
                                                              int *submitted_overflow)
{
    if (!pool_vecs_host || !submitted_result_count || !submitted_overflow) return 0;
    if (bgj_cuda_deterministic_results_requested()) return 0;
    if (transform_dp || device_transform_dp) return 0;

    int selected_device = -1;
    bgj_cuda_current_device(&selected_device);

    int devices[BGJ_CUDA_MAX_EXEC_DEVICES];
    int device_count =
        bgj_cuda_copy_execution_devices(devices, BGJ_CUDA_MAX_EXEC_DEVICES);
    if (device_count <= 1) return 0;

    const uint64_t estimated_dots = bgj_cuda_estimated_pair_dots(num_p, num_n);
    const uint64_t full_work = bgj_cuda_full_pair_work(num_p, num_n);
    if (!full_work || estimated_dots < bgj_cuda_single_bucket_split_min_dots()) {
        return 0;
    }
    if ((uint64_t)device_count > full_work) device_count = (int)full_work;
    if (device_count <= 1) return 0;

    bgj_cuda_raw_async_state_t *state = &bgj_cuda_raw_async_state;
    double submit_t0 = 0.0;
    uint32_t submitted = 0;
    state->reset();
    state->split_active = 1;
    state->split_tensor = tensor_split ? 1 : 0;
    state->split_device_count = device_count;
    state->estimated_dots = estimated_dots;
    state->full_work = full_work;
    state->num_p = num_p;
    state->num_n = num_n;
    state->pool_vecs_host = pool_vecs_host;
    state->pool_epoch = pool_epoch;
    state->pool_size = pool_size;
    state->p_ids = p_ids;
    state->n_ids = n_ids;
    state->p_norm = p_norm;
    state->n_norm = n_norm;
    state->p_dot = p_dot;
    state->n_dot = n_dot;
    state->vec_length = vec_length;
    state->goal_norm = goal_norm;
    state->center_id = center_id;
    state->center_norm = center_norm;
    state->record_dp = record_dp;
    state->raw_center_dp = raw_center_dp;
    state->total_t0 = bgj_cuda_wall_time();

    const int snapshot_center_dots =
        record_dp && raw_center_dp && bgj_cuda_single_bucket_split_verify_requested();
    const int32_t *submit_p_dot = p_dot;
    const int32_t *submit_n_dot = n_dot;
    if (snapshot_center_dots) {
        const size_t p_dot_bytes = (size_t)num_p * sizeof(int32_t);
        const size_t n_dot_bytes = (size_t)num_n * sizeof(int32_t);
        if ((num_p && !p_dot) || (num_n && !n_dot)) {
            set_plain_error("missing CUDA center dots for split verifier");
            goto fail_sync;
        }
        if (p_dot_bytes) {
            state->p_dot_snapshot = (int32_t *)malloc(p_dot_bytes);
            if (!state->p_dot_snapshot) {
                set_plain_error("CUDA split verifier center dot allocation failed");
                goto fail_sync;
            }
            state->p_dot_snapshot_capacity = p_dot_bytes;
            memcpy(state->p_dot_snapshot, p_dot, p_dot_bytes);
            submit_p_dot = state->p_dot_snapshot;
            state->p_dot = state->p_dot_snapshot;
        }
        if (n_dot_bytes) {
            state->n_dot_snapshot = (int32_t *)malloc(n_dot_bytes);
            if (!state->n_dot_snapshot) {
                set_plain_error("CUDA split verifier center dot allocation failed");
                goto fail_sync;
            }
            state->n_dot_snapshot_capacity = n_dot_bytes;
            memcpy(state->n_dot_snapshot, n_dot, n_dot_bytes);
            submit_n_dot = state->n_dot_snapshot;
            state->n_dot = state->n_dot_snapshot;
        }
    }

    for (int i = 0; i < device_count; i++) {
        const uint64_t begin = ((uint64_t)i * full_work) / (uint64_t)device_count;
        const uint64_t end = ((uint64_t)(i + 1) * full_work) / (uint64_t)device_count;
        state->split_devices[i] = devices[i];
        state->split_work_begin[i] = begin;
        state->split_work_count[i] = end > begin ? end - begin : 0;
    }

    submit_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!state->split_work_count[i]) continue;
        if (!bgj_cuda_set_current_device(devices[i])) goto fail_sync;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch) {
            set_plain_error("out of host memory");
            goto fail_sync;
        }
        const uint64_t split_work_begin = tensor_split ? 0 : state->split_work_begin[i];
        const uint64_t split_work_count = tensor_split ? 0 : state->split_work_count[i];
        const uint32_t split_tensor_slot = tensor_split ? (uint32_t)i : 0u;
        const uint32_t split_tensor_count = tensor_split ? (uint32_t)device_count : 0u;
        if (!bgj_cuda_search_bucket_raw_submit(scratch,
                                               NULL,
                                               NULL,
                                               pool_vecs_host,
                                               pool_epoch,
                                               pool_size,
                                               p_ids,
                                               n_ids,
                                               p_norm,
                                               n_norm,
                                               submit_p_dot,
                                               submit_n_dot,
                                               num_p,
                                               num_n,
                                               vec_length,
                                               goal_norm,
                                               center_id,
                                               center_norm,
                                               record_dp,
                                               0,
                                               0,
                                               raw_center_dp,
                                               record_dot_copy_event,
                                               split_work_begin,
                                               split_work_count,
                                               split_tensor_slot,
                                               split_tensor_count,
                                               result_capacity,
                                               &state->split_submitted_result_count[i],
                                               &state->split_submitted_overflow[i])) {
            goto fail_sync;
        }
        submitted++;
    }
    if (!submitted) {
        state->reset();
        if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
        return 0;
    }

    state->submit_sec = bgj_cuda_wall_time() - submit_t0;
    state->active = 1;
    *submitted_result_count = 0;
    *submitted_overflow = 0;
    set_plain_error("no CUDA error");
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 1;

fail_sync:
    for (int i = 0; i < device_count; i++) {
        if (!state->split_work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (scratch && scratch->stream_ready) {
            if (scratch->device >= 0) cudaSetDevice(scratch->device);
            cudaStreamSynchronize(scratch->stream);
        }
    }
    state->reset();
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 0;
}

static int bgj_cuda_search_bucket_pool_raw_split_finish_async(bgj_cuda_result_t *results,
                                                              uint32_t result_capacity,
                                                              uint32_t *submitted_result_count,
                                                              int *submitted_overflow,
                                                              uint32_t *result_count,
                                                              int *overflow)
{
    bgj_cuda_raw_async_state_t *state = &bgj_cuda_raw_async_state;
    if (!state->active || !state->split_active) {
        set_plain_error("no active async split CUDA bucket");
        return 0;
    }

    int selected_device = -1;
    bgj_cuda_current_device(&selected_device);

    const int device_count = state->split_device_count;
    const int split_profile =
        bgj_cuda_split_profile_requested() &&
        state->estimated_dots >= bgj_cuda_split_profile_min_dots();
    bgj_cuda_search_phase_profile_t phase_profile_sum = {};
    double wait_sec = 0.0;
    double copy_sec = 0.0;
    double copy_t0 = 0.0;
    double total_sec = 0.0;
    const char *mode = state->split_tensor ? "tensor_async" : "range_async";
    uint32_t out_count = 0;
    int out_overflow = 0;
    uint64_t submitted_total = 0;
    int submitted_any_overflow = 0;

    const double wait_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!state->split_work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize async split bucket counts", err);
            goto fail_sync;
        }
        bgj_cuda_get_submitted_counts(scratch,
                                      &state->split_submitted_result_count[i],
                                      &state->split_submitted_overflow[i]);
        submitted_total += state->split_submitted_result_count[i];
        if (state->split_submitted_overflow[i]) submitted_any_overflow = 1;
    }
    wait_sec = bgj_cuda_wall_time() - wait_t0;
    if (submitted_result_count) {
        *submitted_result_count = submitted_total > 0xffffffffull ?
                                  0xffffffffu : (uint32_t)submitted_total;
    }
    if (submitted_overflow) {
        *submitted_overflow = submitted_any_overflow ||
                              submitted_total > (uint64_t)result_capacity;
    }

    copy_t0 = bgj_cuda_wall_time();
    for (int i = 0; i < device_count; i++) {
        if (!state->split_work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        const uint32_t remaining =
            out_count < result_capacity ? result_capacity - out_count : 0u;
        if (!bgj_cuda_finish_submitted_bucket(scratch,
                                              results + out_count,
                                              remaining,
                                              state->split_submitted_result_count[i],
                                              state->split_submitted_overflow[i],
                                              &state->split_copied_result_count[i],
                                              &state->split_copied_overflow[i])) {
            goto fail_sync;
        }
        out_count += state->split_copied_result_count[i];
        if (state->split_copied_overflow[i]) out_overflow = 1;
    }

    for (int i = 0; i < device_count; i++) {
        if (!state->split_work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (!scratch || !scratch->stream_ready) goto fail_sync;
        if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize async split bucket results", err);
            goto fail_sync;
        }
        if (!bgj_cuda_search_profile_collect(scratch)) goto fail_sync;
        if (scratch->phase_profile_active) {
            bgj_cuda_search_phase_profile_add(&phase_profile_sum,
                                              &scratch->last_phase_profile);
        }
    }
    copy_sec = bgj_cuda_wall_time() - copy_t0;

    if (!bgj_cuda_verify_split_bucket_results(mode,
                                              selected_device >= 0 ? selected_device : state->split_devices[0],
                                              state->pool_vecs_host,
                                              state->pool_epoch,
                                              state->pool_size,
                                              state->p_ids,
                                              state->n_ids,
                                              state->p_norm,
                                              state->n_norm,
                                              state->p_dot,
                                              state->n_dot,
                                              state->num_p,
                                              state->num_n,
                                              state->vec_length,
                                              state->goal_norm,
                                              state->center_id,
                                              state->center_norm,
                                              state->record_dp,
                                              state->raw_center_dp,
                                              results,
                                              out_count,
                                              out_overflow,
                                              result_capacity)) {
        goto fail_sync;
    }

    if (result_count) *result_count = out_count;
    if (overflow) *overflow = out_overflow;
    set_plain_error("no CUDA error");

    total_sec = bgj_cuda_wall_time() - state->total_t0;
    if (split_profile) {
        fprintf(stderr,
                "cuda_split_profile: mode=%s devices=%d num_p=%u num_n=%u dots=%lu "
                "full_work=%lu results=%u overflow=%d submit=%.6fs wait=%.6fs "
                "copy=%.6fs total=%.6fs",
                mode,
                device_count,
                state->num_p,
                state->num_n,
                (unsigned long)state->estimated_dots,
                (unsigned long)state->full_work,
                out_count,
                out_overflow,
                state->submit_sec,
                wait_sec,
                copy_sec,
                total_sec);
        bgj_cuda_split_profile_print_phases(stderr, &phase_profile_sum);
        fprintf(stderr, "\n");
        for (int i = 0; i < device_count; i++) {
            if (!state->split_work_count[i]) continue;
            bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
            const uint64_t device_dots =
                bgj_cuda_scale_work(state->estimated_dots,
                                    state->split_work_count[i],
                                    state->full_work);
            fprintf(stderr,
                    "cuda_split_device_profile: mode=%s slot=%d/%d device=%d "
                    "work_begin=%lu work_count=%lu dots=%lu submitted=%u copied=%u "
                    "submitted_overflow=%d copied_overflow=%d",
                    mode,
                    i,
                    device_count,
                    state->split_devices[i],
                    (unsigned long)state->split_work_begin[i],
                    (unsigned long)state->split_work_count[i],
                    (unsigned long)device_dots,
                    state->split_submitted_result_count[i],
                    state->split_copied_result_count[i],
                    state->split_submitted_overflow[i],
                    state->split_copied_overflow[i]);
            if (scratch && scratch->phase_profile_active) {
                bgj_cuda_split_profile_print_phases(stderr, &scratch->last_phase_profile);
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }

    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_ASYNC,
                                   device_count > 0 ? state->split_devices[0] : selected_device,
                                   (uint32_t)device_count,
                                   0,
                                   state->num_p,
                                   state->num_n,
                                   state->estimated_dots,
                                   out_count,
                                   out_overflow,
                                   0,
                                   state->submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   total_sec,
                                   &phase_profile_sum);
    for (int i = 0; i < device_count; i++) {
        if (!state->split_work_count[i]) continue;
        const uint64_t device_dots =
            bgj_cuda_scale_work(state->estimated_dots,
                                state->split_work_count[i],
                                state->full_work);
        bgj_cuda_device_profile_record(state->split_devices[i],
                                       1,
                                       device_dots,
                                       state->split_copied_result_count[i],
                                       state->split_copied_overflow[i] ? 1 : 0,
                                       0,
                                       state->submit_sec,
                                       wait_sec,
                                       copy_sec,
                                       total_sec);
    }
    bgj_cuda_phase_profile_record(1,
                                  state->estimated_dots,
                                  out_count,
                                  out_overflow,
                                  0,
                                  &phase_profile_sum);
    state->reset();
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 1;

fail_sync:
    for (int i = 0; i < device_count; i++) {
        if (!state->split_work_count[i]) continue;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get((uint32_t)i);
        if (scratch && scratch->stream_ready) {
            if (scratch->device >= 0) cudaSetDevice(scratch->device);
            cudaStreamSynchronize(scratch->stream);
        }
    }
    state->reset();
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    return 0;
}

static int bgj_cuda_search_bucket_raw_impl(const int8_t *p_vecs,
                                           const int8_t *n_vecs,
                                           const int8_t *pool_vecs_host,
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
                                           int raw_center_dp,
                                           int allow_split,
                                           bgj_cuda_result_t *results,
                                           uint32_t result_capacity,
                                           uint32_t *result_count,
                                           int *overflow)
{
    if (!bgj_cuda_select_thread_device()) return 0;
    bgj_cuda_raw_scratch_t *scratch = &bgj_cuda_raw_scratch;
    uint32_t h_result_count = 0;
    int h_overflow = 0;
    const uint64_t estimated_dots = bgj_cuda_estimated_pair_dots(num_p, num_n);
    const double total_t0 = bgj_cuda_wall_time();
    double submit_sec = 0.0;
    double wait_sec = 0.0;
    double copy_sec = 0.0;
    double wait_t0 = 0.0;
    double copy_t0 = 0.0;
    const bgj_cuda_search_phase_profile_t *phase_profile = NULL;

    if (allow_split &&
        bgj_cuda_single_bucket_split_requested() &&
        pool_vecs_host &&
        !transform_dp) {
        if (bgj_cuda_search_bucket_pool_raw_split_impl(pool_vecs_host,
                                                       pool_epoch,
                                                       pool_size,
                                                       p_ids,
                                                       n_ids,
                                                       p_norm,
                                                       n_norm,
                                                       p_dot,
                                                       n_dot,
                                                       num_p,
                                                       num_n,
                                                       vec_length,
                                                       goal_norm,
                                                       center_id,
                                                       center_norm,
                                                       record_dp,
                                                       raw_center_dp,
                                                       bgj_cuda_single_bucket_tensor_split_requested(),
                                                       results,
                                                       result_capacity,
                                                       result_count,
                                                       overflow)) {
            return 1;
        }
    }

    const double submit_t0 = bgj_cuda_wall_time();
    if (!bgj_cuda_search_bucket_raw_submit(scratch,
                                           p_vecs,
                                           n_vecs,
                                           pool_vecs_host,
                                           pool_epoch,
                                           pool_size,
                                           p_ids,
                                           n_ids,
                                           p_norm,
                                           n_norm,
                                           p_dot,
                                           n_dot,
                                           num_p,
                                           num_n,
                                           vec_length,
                                           goal_norm,
                                           center_id,
                                           center_norm,
                                           record_dp,
                                           transform_dp,
                                           0,
                                           raw_center_dp,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           result_capacity,
                                           &h_result_count,
                                           &h_overflow)) {
        bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_SINGLE,
                                       scratch->device,
                                       1,
                                       0,
                                       num_p,
                                       num_n,
                                       estimated_dots,
                                       0,
                                       0,
                                       1,
                                       bgj_cuda_wall_time() - submit_t0,
                                       0.0,
                                       0.0,
                                       bgj_cuda_wall_time() - total_t0);
        bgj_cuda_device_profile_record(scratch->device,
                                       1,
                                       estimated_dots,
                                       0,
                                       0,
                                       1,
                                       bgj_cuda_wall_time() - submit_t0,
                                       0.0,
                                       0.0,
                                       bgj_cuda_wall_time() - total_t0);
        return 0;
    }
    submit_sec = bgj_cuda_wall_time() - submit_t0;
    cudaStream_t stream = scratch->stream;
    wait_t0 = bgj_cuda_wall_time();
    CUDA_TRY(cudaStreamSynchronize(stream));
    wait_sec = bgj_cuda_wall_time() - wait_t0;
    bgj_cuda_get_submitted_counts(scratch, &h_result_count, &h_overflow);

    copy_t0 = bgj_cuda_wall_time();
    if (!bgj_cuda_finish_submitted_bucket(scratch,
                                          results,
                                          result_capacity,
                                          h_result_count,
                                          h_overflow,
                                          result_count,
                                          overflow)) {
        bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_SINGLE,
                                       scratch->device,
                                       1,
                                       0,
                                       num_p,
                                       num_n,
                                       estimated_dots,
                                       h_result_count,
                                       h_overflow ? 1 : 0,
                                       1,
                                       submit_sec,
                                       wait_sec,
                                       bgj_cuda_wall_time() - copy_t0,
                                       bgj_cuda_wall_time() - total_t0);
        bgj_cuda_device_profile_record(scratch->device,
                                       1,
                                       estimated_dots,
                                       h_result_count,
                                       h_overflow ? 1 : 0,
                                       1,
                                       submit_sec,
                                       wait_sec,
                                       bgj_cuda_wall_time() - copy_t0,
                                       bgj_cuda_wall_time() - total_t0);
        return 0;
    }
    CUDA_TRY(cudaStreamSynchronize(stream));
    copy_sec = bgj_cuda_wall_time() - copy_t0;
    if (!bgj_cuda_search_profile_collect(scratch)) goto fail;
    phase_profile = scratch->phase_profile_active ? &scratch->last_phase_profile : NULL;

    set_plain_error("no CUDA error");
    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_SINGLE,
                                   scratch->device,
                                   1,
                                   0,
                                   num_p,
                                   num_n,
                                   estimated_dots,
                                   *result_count,
                                   *overflow ? 1 : 0,
                                   0,
                                   submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - total_t0,
                                   phase_profile);
    bgj_cuda_device_profile_record(scratch->device,
                                   1,
                                   estimated_dots,
                                   *result_count,
                                   *overflow ? 1 : 0,
                                   0,
                                   submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - total_t0);
    bgj_cuda_phase_profile_record(1,
                                  estimated_dots,
                                  *result_count,
                                  *overflow ? 1 : 0,
                                  0,
                                  phase_profile);
    return 1;

fail:
    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_SINGLE,
                                   scratch->device,
                                   1,
                                   0,
                                   num_p,
                                   num_n,
                                   estimated_dots,
                                   h_result_count,
                                   h_overflow ? 1 : 0,
                                   1,
                                   submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - total_t0);
    bgj_cuda_device_profile_record(scratch->device,
                                   1,
                                   estimated_dots,
                                   h_result_count,
                                   h_overflow ? 1 : 0,
                                   1,
                                   submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - total_t0);
    return 0;
}

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
                                                            int *submitted_overflow)
{
    bgj_cuda_raw_async_state.reset();
    if (!bgj_cuda_select_thread_device()) return 0;
    if (bgj_cuda_single_bucket_split_requested() &&
        pool_vecs &&
        !transform_dp &&
        !device_transform_dp) {
        if (bgj_cuda_search_bucket_pool_raw_split_submit_async(pool_vecs,
                                                               pool_epoch,
                                                               pool_size,
                                                               p_ids,
                                                               n_ids,
                                                               p_norm,
                                                               n_norm,
                                                               p_dot,
                                                               n_dot,
                                                               num_p,
                                                               num_n,
                                                               vec_length,
                                                               goal_norm,
                                                               center_id,
                                                               center_norm,
                                                               record_dp,
                                                               transform_dp,
                                                               device_transform_dp,
                                                               raw_center_dp,
                                                               record_dot_copy_event,
                                                               bgj_cuda_single_bucket_tensor_split_requested(),
                                                               result_capacity,
                                                               submitted_result_count,
                                                               submitted_overflow)) {
            return 1;
        }
    }
    bgj_cuda_raw_scratch_t *scratch = &bgj_cuda_raw_scratch;
    const uint64_t estimated_dots = bgj_cuda_estimated_pair_dots(num_p, num_n);
    const double total_t0 = bgj_cuda_wall_time();
    const double submit_t0 = bgj_cuda_wall_time();

    if (!bgj_cuda_search_bucket_raw_submit(scratch,
                                           NULL,
                                           NULL,
                                           pool_vecs,
                                           pool_epoch,
                                           pool_size,
                                           p_ids,
                                           n_ids,
                                           p_norm,
                                           n_norm,
                                           p_dot,
                                           n_dot,
                                           num_p,
                                           num_n,
                                           vec_length,
                                           goal_norm,
                                           center_id,
                                           center_norm,
                                           record_dp,
                                           transform_dp,
                                           device_transform_dp,
                                           raw_center_dp,
                                           record_dot_copy_event,
                                           0,
                                           0,
                                           0,
                                           0,
                                           result_capacity,
                                           submitted_result_count,
                                           submitted_overflow)) {
        if (scratch->stream_ready) cudaStreamSynchronize(scratch->stream);
        bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_ASYNC,
                                       scratch->device,
                                       1,
                                       0,
                                       num_p,
                                       num_n,
                                       estimated_dots,
                                       0,
                                       0,
                                       1,
                                       bgj_cuda_wall_time() - submit_t0,
                                       0.0,
                                       0.0,
                                       bgj_cuda_wall_time() - total_t0);
        bgj_cuda_device_profile_record(scratch->device,
                                       1,
                                       estimated_dots,
                                       0,
                                       0,
                                       1,
                                       bgj_cuda_wall_time() - submit_t0,
                                       0.0,
                                       0.0,
                                       bgj_cuda_wall_time() - total_t0);
        return 0;
    }

    bgj_cuda_raw_async_state.active = 1;
    bgj_cuda_raw_async_state.estimated_dots = estimated_dots;
    bgj_cuda_raw_async_state.num_p = num_p;
    bgj_cuda_raw_async_state.num_n = num_n;
    bgj_cuda_raw_async_state.total_t0 = total_t0;
    bgj_cuda_raw_async_state.submit_sec = bgj_cuda_wall_time() - submit_t0;
    set_plain_error("no CUDA error");
    return 1;
}

extern "C" int bgj_cuda_search_bucket_raw_wait_dot_copy_async()
{
    bgj_cuda_raw_scratch_t *scratch = &bgj_cuda_raw_scratch;
    if (!bgj_cuda_raw_async_state.active) {
        set_plain_error("no active async CUDA bucket");
        return 0;
    }
    if (bgj_cuda_raw_async_state.split_active) {
        bgj_cuda_raw_async_state_t *state = &bgj_cuda_raw_async_state;
        int selected_device = -1;
        bgj_cuda_current_device(&selected_device);
        for (int i = 0; i < state->split_device_count; i++) {
            if (!state->split_work_count[i]) continue;
            bgj_cuda_raw_scratch_t *split_scratch =
                bgj_cuda_raw_batch_scratch.get((uint32_t)i);
            if (!split_scratch || !split_scratch->stream_ready) {
                set_plain_error("missing async split CUDA bucket stream");
                if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
                return 0;
            }
            if (split_scratch->device >= 0 &&
                !bgj_cuda_set_current_device(split_scratch->device)) {
                if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
                return 0;
            }
            if (!split_scratch->dot_event_ready) {
                set_plain_error("no async split CUDA dot-copy event");
                if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
                return 0;
            }
            cudaError_t err = cudaEventSynchronize(split_scratch->dot_copy_done);
            if (err != cudaSuccess) {
                set_cuda_error("cudaEventSynchronize split dot copy", err);
                if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
                return 0;
            }
        }
        set_plain_error("no CUDA error");
        if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
        return 1;
    }
    if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) return 0;
    if (!scratch->dot_event_ready) {
        set_plain_error("no async CUDA dot-copy event");
        return 0;
    }
    cudaError_t err = cudaEventSynchronize(scratch->dot_copy_done);
    if (err != cudaSuccess) {
        set_cuda_error("cudaEventSynchronize dot copy", err);
        return 0;
    }
    set_plain_error("no CUDA error");
    return 1;
}

extern "C" int bgj_cuda_search_bucket_raw_finish_async(bgj_cuda_result_t *results,
                                                       uint32_t result_capacity,
                                                       uint32_t *submitted_result_count,
                                                       int *submitted_overflow,
                                                       uint32_t *result_count,
                                                       int *overflow)
{
    if (bgj_cuda_raw_async_state.active && bgj_cuda_raw_async_state.split_active) {
        return bgj_cuda_search_bucket_pool_raw_split_finish_async(results,
                                                                  result_capacity,
                                                                  submitted_result_count,
                                                                  submitted_overflow,
                                                                  result_count,
                                                                  overflow);
    }
    bgj_cuda_raw_scratch_t *scratch = &bgj_cuda_raw_scratch;
    if (!bgj_cuda_raw_async_state.active) {
        set_plain_error("no active async CUDA bucket");
        return 0;
    }
    if (scratch->device >= 0 && !bgj_cuda_set_current_device(scratch->device)) return 0;

    cudaStream_t stream = scratch->stream;
    double wait_sec = 0.0;
    double copy_sec = 0.0;
    double copy_t0 = 0.0;
    uint32_t h_result_count = 0;
    int h_overflow = 1;
    const bgj_cuda_search_phase_profile_t *phase_profile = NULL;

    const double wait_t0 = bgj_cuda_wall_time();
    CUDA_TRY(cudaStreamSynchronize(stream));
    wait_sec = bgj_cuda_wall_time() - wait_t0;
    bgj_cuda_get_submitted_counts(scratch, &h_result_count, &h_overflow);
    if (submitted_result_count) *submitted_result_count = h_result_count;
    if (submitted_overflow) *submitted_overflow = h_overflow;

    copy_t0 = bgj_cuda_wall_time();
    if (!bgj_cuda_finish_submitted_bucket(scratch,
                                          results,
                                          result_capacity,
                                          h_result_count,
                                          h_overflow,
                                          result_count,
                                          overflow)) {
        bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_ASYNC,
                                       scratch->device,
                                       1,
                                       0,
                                       bgj_cuda_raw_async_state.num_p,
                                       bgj_cuda_raw_async_state.num_n,
                                       bgj_cuda_raw_async_state.estimated_dots,
                                       h_result_count,
                                       h_overflow ? 1 : 0,
                                       1,
                                       bgj_cuda_raw_async_state.submit_sec,
                                       wait_sec,
                                       bgj_cuda_wall_time() - copy_t0,
                                       bgj_cuda_wall_time() - bgj_cuda_raw_async_state.total_t0);
        bgj_cuda_device_profile_record(scratch->device,
                                       1,
                                       bgj_cuda_raw_async_state.estimated_dots,
                                       h_result_count,
                                       h_overflow ? 1 : 0,
                                       1,
                                       bgj_cuda_raw_async_state.submit_sec,
                                       wait_sec,
                                       bgj_cuda_wall_time() - copy_t0,
                                       bgj_cuda_wall_time() - bgj_cuda_raw_async_state.total_t0);
        bgj_cuda_raw_async_state.active = 0;
        return 0;
    }
    CUDA_TRY(cudaStreamSynchronize(stream));
    copy_sec = bgj_cuda_wall_time() - copy_t0;
    if (!bgj_cuda_search_profile_collect(scratch)) goto fail;
    phase_profile = scratch->phase_profile_active ? &scratch->last_phase_profile : NULL;

    set_plain_error("no CUDA error");
    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_ASYNC,
                                   scratch->device,
                                   1,
                                   0,
                                   bgj_cuda_raw_async_state.num_p,
                                   bgj_cuda_raw_async_state.num_n,
                                   bgj_cuda_raw_async_state.estimated_dots,
                                   result_count ? *result_count : 0,
                                   overflow && *overflow ? 1 : 0,
                                   0,
                                   bgj_cuda_raw_async_state.submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - bgj_cuda_raw_async_state.total_t0,
                                   phase_profile);
    bgj_cuda_device_profile_record(scratch->device,
                                   1,
                                   bgj_cuda_raw_async_state.estimated_dots,
                                   result_count ? *result_count : 0,
                                   overflow && *overflow ? 1 : 0,
                                   0,
                                   bgj_cuda_raw_async_state.submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - bgj_cuda_raw_async_state.total_t0);
    bgj_cuda_phase_profile_record(1,
                                  bgj_cuda_raw_async_state.estimated_dots,
                                  result_count ? *result_count : 0,
                                  overflow && *overflow ? 1 : 0,
                                  0,
                                  phase_profile);
    bgj_cuda_raw_async_state.active = 0;
    return 1;

fail:
    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_ASYNC,
                                   scratch->device,
                                   1,
                                   0,
                                   bgj_cuda_raw_async_state.num_p,
                                   bgj_cuda_raw_async_state.num_n,
                                   bgj_cuda_raw_async_state.estimated_dots,
                                   submitted_result_count ? *submitted_result_count : 0,
                                   submitted_overflow && *submitted_overflow ? 1 : 0,
                                   1,
                                   bgj_cuda_raw_async_state.submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - bgj_cuda_raw_async_state.total_t0);
    bgj_cuda_device_profile_record(scratch->device,
                                   1,
                                   bgj_cuda_raw_async_state.estimated_dots,
                                   submitted_result_count ? *submitted_result_count : 0,
                                   submitted_overflow && *submitted_overflow ? 1 : 0,
                                   1,
                                   bgj_cuda_raw_async_state.submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - bgj_cuda_raw_async_state.total_t0);
    bgj_cuda_raw_async_state.active = 0;
    return 0;
}

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
                                           int *overflow)
{
    return bgj_cuda_search_bucket_raw_impl(p_vecs,
                                           n_vecs,
                                           NULL,
                                           0,
                                           0,
                                           p_ids,
                                           n_ids,
                                           p_norm,
                                           n_norm,
                                           p_dot,
                                           n_dot,
                                           num_p,
                                           num_n,
                                           vec_length,
                                           goal_norm,
                                           center_id,
                                           center_norm,
                                           record_dp,
                                           transform_dp,
                                           0,
                                           1,
                                           results,
                                           result_capacity,
                                           result_count,
                                           overflow);
}

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
                                                int *overflow)
{
    return bgj_cuda_search_bucket_raw_impl(NULL,
                                           NULL,
                                           pool_vecs,
                                           pool_epoch,
                                           pool_size,
                                           p_ids,
                                           n_ids,
                                           p_norm,
                                           n_norm,
                                           p_dot,
                                           n_dot,
                                           num_p,
                                           num_n,
                                           vec_length,
                                           goal_norm,
                                           center_id,
                                           center_norm,
                                           record_dp,
                                           transform_dp,
                                           0,
                                           1,
                                           results,
                                           result_capacity,
                                           result_count,
                                           overflow);
}

extern "C" int bgj_cuda_search_bucket_pool_raw_single(const int8_t *pool_vecs,
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
                                                       int raw_center_dp,
                                                       bgj_cuda_result_t *results,
                                                       uint32_t result_capacity,
                                                       uint32_t *result_count,
                                                       int *overflow)
{
    return bgj_cuda_search_bucket_raw_impl(NULL,
                                           NULL,
                                           pool_vecs,
                                           pool_epoch,
                                           pool_size,
                                           p_ids,
                                           n_ids,
                                           p_norm,
                                           n_norm,
                                           p_dot,
                                           n_dot,
                                           num_p,
                                           num_n,
                                           vec_length,
                                           goal_norm,
                                           center_id,
                                           center_norm,
                                           record_dp,
                                           transform_dp,
                                           raw_center_dp,
                                           0,
                                           results,
                                           result_capacity,
                                           result_count,
                                           overflow);
}

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
                                                      int *overflow)
{
    if (!bgj_cuda_select_thread_device()) return 0;
    if (batch_size == 0) {
        set_plain_error("no CUDA error");
        return 1;
    }
    if (!pool_vecs || !p_ids || !n_ids || !p_norm || !n_norm ||
        !num_p || !num_n || !goal_norm || !center_id || !center_norm ||
        !results || !result_capacity || !result_count || !overflow) {
        set_plain_error("invalid batch pointer");
        return 0;
    }
    if (record_dp && (!p_dot || !n_dot)) {
        set_plain_error("invalid batch dot pointer");
        return 0;
    }

    int selected_device = -1;
    bgj_cuda_current_device(&selected_device);
    const int multi_gpu_batch_enabled = bgj_cuda_multi_gpu_batch_requested();
    int execution_devices[BGJ_CUDA_MAX_EXEC_DEVICES] = {selected_device};
    int execution_device_count = 1;
    if (multi_gpu_batch_enabled) {
        execution_device_count =
            bgj_cuda_copy_execution_devices(execution_devices, BGJ_CUDA_MAX_EXEC_DEVICES);
        if (execution_device_count <= 0) {
            execution_devices[0] = selected_device;
            execution_device_count = 1;
        }
    }
    uint64_t item_dots_stack[64];
    int item_devices_stack[64];
    uint32_t scratch_items_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    uint32_t scratch_split_slots_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    uint32_t scratch_split_counts_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    int scratch_devices_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    uint64_t scratch_work_begin_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    uint64_t scratch_work_count_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    uint32_t scratch_result_counts_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    uint32_t scratch_copied_counts_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    int scratch_overflows_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    int scratch_copied_overflows_stack[64 + BGJ_CUDA_MAX_EXEC_DEVICES];
    uint64_t *item_dots = batch_size <= 64 ? item_dots_stack :
                          (uint64_t *)malloc((size_t)batch_size * sizeof(uint64_t));
    int *item_devices = batch_size <= 64 ? item_devices_stack :
                        (int *)malloc((size_t)batch_size * sizeof(int));
    const uint32_t scratch_stack_capacity = 64u + BGJ_CUDA_MAX_EXEC_DEVICES;
    const uint32_t max_scratch_count =
        batch_size + (execution_device_count > 1 ? (uint32_t)execution_device_count - 1u : 0u);
    uint32_t *scratch_items = max_scratch_count <= scratch_stack_capacity ?
                              scratch_items_stack :
                              (uint32_t *)malloc((size_t)max_scratch_count * sizeof(uint32_t));
    uint32_t *scratch_split_slots = max_scratch_count <= scratch_stack_capacity ?
                                    scratch_split_slots_stack :
                                    (uint32_t *)malloc((size_t)max_scratch_count * sizeof(uint32_t));
    uint32_t *scratch_split_counts = max_scratch_count <= scratch_stack_capacity ?
                                     scratch_split_counts_stack :
                                     (uint32_t *)malloc((size_t)max_scratch_count * sizeof(uint32_t));
    int *scratch_devices = max_scratch_count <= scratch_stack_capacity ?
                           scratch_devices_stack :
                           (int *)malloc((size_t)max_scratch_count * sizeof(int));
    uint64_t *scratch_work_begin = max_scratch_count <= scratch_stack_capacity ?
                                   scratch_work_begin_stack :
                                   (uint64_t *)malloc((size_t)max_scratch_count * sizeof(uint64_t));
    uint64_t *scratch_work_count = max_scratch_count <= scratch_stack_capacity ?
                                   scratch_work_count_stack :
                                   (uint64_t *)malloc((size_t)max_scratch_count * sizeof(uint64_t));
    uint32_t *scratch_result_counts = max_scratch_count <= scratch_stack_capacity ?
                                      scratch_result_counts_stack :
                                      (uint32_t *)malloc((size_t)max_scratch_count * sizeof(uint32_t));
    uint32_t *scratch_copied_counts = max_scratch_count <= scratch_stack_capacity ?
                                      scratch_copied_counts_stack :
                                      (uint32_t *)malloc((size_t)max_scratch_count * sizeof(uint32_t));
    int *scratch_overflows = max_scratch_count <= scratch_stack_capacity ?
                             scratch_overflows_stack :
                             (int *)malloc((size_t)max_scratch_count * sizeof(int));
    int *scratch_copied_overflows = max_scratch_count <= scratch_stack_capacity ?
                                    scratch_copied_overflows_stack :
                                    (int *)malloc((size_t)max_scratch_count * sizeof(int));
    if (!item_dots || !item_devices) {
        if (item_dots != item_dots_stack) free(item_dots);
        if (item_devices != item_devices_stack) free(item_devices);
        if (scratch_items != scratch_items_stack) free(scratch_items);
        if (scratch_split_slots != scratch_split_slots_stack) free(scratch_split_slots);
        if (scratch_split_counts != scratch_split_counts_stack) free(scratch_split_counts);
        if (scratch_devices != scratch_devices_stack) free(scratch_devices);
        if (scratch_work_begin != scratch_work_begin_stack) free(scratch_work_begin);
        if (scratch_work_count != scratch_work_count_stack) free(scratch_work_count);
        if (scratch_result_counts != scratch_result_counts_stack) free(scratch_result_counts);
        if (scratch_copied_counts != scratch_copied_counts_stack) free(scratch_copied_counts);
        if (scratch_overflows != scratch_overflows_stack) free(scratch_overflows);
        if (scratch_copied_overflows != scratch_copied_overflows_stack) free(scratch_copied_overflows);
        set_plain_error("out of host memory");
        return 0;
    }
    if (!scratch_items || !scratch_split_slots || !scratch_split_counts ||
        !scratch_devices || !scratch_work_begin || !scratch_work_count ||
        !scratch_result_counts || !scratch_copied_counts || !scratch_overflows ||
        !scratch_copied_overflows) {
        if (item_dots != item_dots_stack) free(item_dots);
        if (item_devices != item_devices_stack) free(item_devices);
        if (scratch_items != scratch_items_stack) free(scratch_items);
        if (scratch_split_slots != scratch_split_slots_stack) free(scratch_split_slots);
        if (scratch_split_counts != scratch_split_counts_stack) free(scratch_split_counts);
        if (scratch_devices != scratch_devices_stack) free(scratch_devices);
        if (scratch_work_begin != scratch_work_begin_stack) free(scratch_work_begin);
        if (scratch_work_count != scratch_work_count_stack) free(scratch_work_count);
        if (scratch_result_counts != scratch_result_counts_stack) free(scratch_result_counts);
        if (scratch_copied_counts != scratch_copied_counts_stack) free(scratch_copied_counts);
        if (scratch_overflows != scratch_overflows_stack) free(scratch_overflows);
        if (scratch_copied_overflows != scratch_copied_overflows_stack) free(scratch_copied_overflows);
        set_plain_error("out of host memory");
        return 0;
    }
    uint32_t submitted = 0;
    uint64_t estimated_dots = 0;
    uint64_t total_results = 0;
    uint64_t total_overflows = 0;
    uint32_t max_num_p = 0;
    uint32_t max_num_n = 0;
    const double total_t0 = bgj_cuda_wall_time();
    double submit_sec = 0.0;
    double wait_sec = 0.0;
    double copy_sec = 0.0;
    double wait_t0 = 0.0;
    double copy_t0 = 0.0;
    bgj_cuda_search_phase_profile_t phase_profile_sum = {};
    bgj_cuda_search_phase_profile_t split_item_phase_profile = {};
    const bgj_cuda_search_phase_profile_t *batch_phase_profile = NULL;
    int multi_gpu_batch = 0;
    int batch_split = 0;
    uint32_t split_item = batch_size;
    uint64_t split_full_work = 0;
    uint32_t scratch_count = 0;

    for (uint32_t i = 0; i < batch_size; i++) {
        result_count[i] = 0;
        overflow[i] = 0;
        item_dots[i] = bgj_cuda_estimated_pair_dots(num_p[i], num_n[i]);
        item_devices[i] = selected_device;
        estimated_dots += item_dots[i];
        if (num_p[i] > max_num_p) max_num_p = num_p[i];
        if (num_n[i] > max_num_n) max_num_n = num_n[i];
    }

    if (execution_device_count > 1 &&
        estimated_dots >= bgj_cuda_multi_gpu_batch_min_dots() &&
        multi_gpu_batch_enabled) {
        uint64_t device_work[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        if (bgj_cuda_multi_gpu_batch_split_requested() &&
            !transform_dp &&
            !bgj_cuda_deterministic_results_requested()) {
            uint32_t max_item = 0;
            for (uint32_t i = 1; i < batch_size; i++) {
                if (item_dots[i] > item_dots[max_item]) max_item = i;
            }
            const uint64_t min_split_dots = bgj_cuda_multi_gpu_batch_split_min_dots();
            const uint32_t min_share_pct =
                bgj_cuda_multi_gpu_batch_split_min_share_pct();
            split_full_work = bgj_cuda_full_pair_work(num_p[max_item], num_n[max_item]);
            const int share_ok =
                !min_share_pct ||
                (estimated_dots &&
                 (long double)item_dots[max_item] * 100.0L >=
                 (long double)estimated_dots * (long double)min_share_pct);
            if (item_dots[max_item] >= min_split_dots &&
                split_full_work >= (uint64_t)execution_device_count &&
                share_ok) {
                batch_split = 1;
                split_item = max_item;
                item_devices[split_item] = execution_devices[0];
                for (int slot = 0; slot < execution_device_count; slot++) {
                    const uint64_t begin =
                        ((uint64_t)slot * split_full_work) /
                        (uint64_t)execution_device_count;
                    const uint64_t end =
                        ((uint64_t)(slot + 1) * split_full_work) /
                        (uint64_t)execution_device_count;
                    const uint64_t count = end > begin ? end - begin : 0;
                    device_work[slot] =
                        bgj_cuda_scale_work(item_dots[split_item],
                                            count,
                                            split_full_work);
                }
            }
        }
        if (batch_size < (uint32_t)execution_device_count && !batch_split) {
            goto build_batch_scratch_plan;
        }
        for (uint32_t i = 0; i < batch_size; i++) {
            if (batch_split && i == split_item) continue;
            int best_slot = 0;
            for (int slot = 1; slot < execution_device_count; slot++) {
                if (device_work[slot] < device_work[best_slot]) best_slot = slot;
            }
            item_devices[i] = execution_devices[best_slot];
            device_work[best_slot] += item_dots[i];
        }
        multi_gpu_batch = 1;
    }

build_batch_scratch_plan:
    for (uint32_t i = 0; i < batch_size; i++) {
        if (batch_split && i == split_item) {
            const int tensor_split =
                bgj_cuda_multi_gpu_batch_tensor_split_requested();
            for (int slot = 0; slot < execution_device_count; slot++) {
                const uint64_t begin =
                    ((uint64_t)slot * split_full_work) /
                    (uint64_t)execution_device_count;
                const uint64_t end =
                    ((uint64_t)(slot + 1) * split_full_work) /
                    (uint64_t)execution_device_count;
                scratch_items[scratch_count] = i;
                scratch_split_slots[scratch_count] = tensor_split ? (uint32_t)slot : 0u;
                scratch_split_counts[scratch_count] =
                    tensor_split ? (uint32_t)execution_device_count : 0u;
                scratch_devices[scratch_count] = execution_devices[slot];
                scratch_work_begin[scratch_count] = begin;
                scratch_work_count[scratch_count] = end > begin ? end - begin : 0ull;
                scratch_result_counts[scratch_count] = 0;
                scratch_copied_counts[scratch_count] = 0;
                scratch_overflows[scratch_count] = 0;
                scratch_copied_overflows[scratch_count] = 0;
                scratch_count++;
            }
        } else {
            scratch_items[scratch_count] = i;
            scratch_split_slots[scratch_count] = 0;
            scratch_split_counts[scratch_count] = 0;
            scratch_devices[scratch_count] = item_devices[i];
            scratch_work_begin[scratch_count] = 0;
            scratch_work_count[scratch_count] = 0;
            scratch_result_counts[scratch_count] = 0;
            scratch_copied_counts[scratch_count] = 0;
            scratch_overflows[scratch_count] = 0;
            scratch_copied_overflows[scratch_count] = 0;
            scratch_count++;
        }
    }

    const double submit_t0 = bgj_cuda_wall_time();
    for (uint32_t s = 0; s < scratch_count; s++) {
        const uint32_t i = scratch_items[s];
        if (multi_gpu_batch && !bgj_cuda_set_current_device(scratch_devices[s])) {
            goto fail_sync;
        }
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(s);
        if (!scratch) {
            set_plain_error("out of host memory");
            goto fail_sync;
        }
        if (!bgj_cuda_search_bucket_raw_submit(scratch,
                                               NULL,
                                               NULL,
                                               pool_vecs,
                                               pool_epoch,
                                               pool_size,
                                               p_ids[i],
                                               n_ids[i],
                                               p_norm[i],
                                               n_norm[i],
                                               record_dp ? p_dot[i] : NULL,
                                               record_dp ? n_dot[i] : NULL,
                                               num_p[i],
                                               num_n[i],
                                               vec_length,
                                               goal_norm[i],
                                               center_id[i],
                                               center_norm[i],
                                               record_dp,
                                               transform_dp,
                                               0,
                                               0,
                                               0,
                                               scratch_split_counts[s] ? 0ull : scratch_work_begin[s],
                                               scratch_split_counts[s] ? 0ull : scratch_work_count[s],
                                               scratch_split_slots[s],
                                               scratch_split_counts[s],
                                               result_capacity[i],
                                               &scratch_result_counts[s],
                                               &scratch_overflows[s])) {
            goto fail_sync;
        }
        submitted++;
    }
    submit_sec = bgj_cuda_wall_time() - submit_t0;

    wait_t0 = bgj_cuda_wall_time();
    for (uint32_t i = 0; i < submitted; i++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
        if (scratch && scratch->device >= 0 &&
            !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize batch counts", err);
            goto fail_sync;
        }
        bgj_cuda_get_submitted_counts(scratch,
                                      &scratch_result_counts[i],
                                      &scratch_overflows[i]);
    }
    wait_sec = bgj_cuda_wall_time() - wait_t0;

    copy_t0 = bgj_cuda_wall_time();
    for (uint32_t s = 0; s < submitted; s++) {
        const uint32_t i = scratch_items[s];
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(s);
        if (scratch && scratch->device >= 0 &&
            !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        const uint32_t submitted_count = scratch_result_counts[s];
        const int submitted_overflow = scratch_overflows[s];
        const uint32_t existing_count = result_count[i];
        const uint32_t remaining =
            existing_count < result_capacity[i] ? result_capacity[i] - existing_count : 0u;
        if (!bgj_cuda_finish_submitted_bucket(scratch,
                                              results[i] + existing_count,
                                              remaining,
                                              submitted_count,
                                              submitted_overflow,
                                              &scratch_copied_counts[s],
                                              &scratch_copied_overflows[s])) {
            goto fail_sync;
        }
        result_count[i] += scratch_copied_counts[s];
        if (scratch_copied_overflows[s]) overflow[i] = 1;
    }

    for (uint32_t i = 0; i < submitted; i++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
        if (scratch && scratch->device >= 0 &&
            !bgj_cuda_set_current_device(scratch->device)) {
            goto fail_sync;
        }
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize batch results", err);
            goto fail_sync;
        }
        if (!bgj_cuda_search_profile_collect(scratch)) goto fail_sync;
        if (scratch->phase_profile_active) {
            bgj_cuda_search_phase_profile_add(&phase_profile_sum,
                                              &scratch->last_phase_profile);
            if (batch_split && scratch_items[i] == split_item) {
                bgj_cuda_search_phase_profile_add(&split_item_phase_profile,
                                                  &scratch->last_phase_profile);
            }
        }
    }
    copy_sec = bgj_cuda_wall_time() - copy_t0;
    for (uint32_t i = 0; i < batch_size; i++) {
        total_results += result_count[i];
        if (overflow[i]) total_overflows++;
    }
    batch_phase_profile = bgj_cuda_phase_profile_requested() ? &phase_profile_sum : NULL;

    set_plain_error("no CUDA error");
    for (uint32_t i = 0; i < batch_size; i++) {
        bgj_cuda_raw_scratch_t *scratch = NULL;
        for (uint32_t s = 0; s < submitted; s++) {
            if (scratch_items[s] == i) {
                scratch = bgj_cuda_raw_batch_scratch.get(s);
                break;
            }
        }
        const bgj_cuda_search_phase_profile_t *item_phase_profile = NULL;
        if (batch_split && i == split_item) {
            item_phase_profile = bgj_cuda_phase_profile_requested() ?
                                 &split_item_phase_profile : NULL;
        } else {
            item_phase_profile =
                scratch && scratch->phase_profile_active ? &scratch->last_phase_profile : NULL;
        }
        bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_BATCH_ITEM,
                                       scratch ? scratch->device : selected_device,
                                       batch_size,
                                       i,
                                       num_p[i],
                                       num_n[i],
                                       bgj_cuda_estimated_pair_dots(num_p[i], num_n[i]),
                                       result_count[i],
                                       overflow[i] ? 1 : 0,
                                       0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       item_phase_profile);
    }
    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_BATCH,
                                   selected_device,
                                   batch_size,
                                   0,
                                   max_num_p,
                                   max_num_n,
                                   estimated_dots,
                                   total_results,
                                   total_overflows,
                                   0,
                                   submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - total_t0,
                                   batch_phase_profile);
    if (multi_gpu_batch) {
        uint64_t device_buckets[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        uint64_t device_dots[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        uint64_t device_results[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        uint64_t device_overflows[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        for (uint32_t s = 0; s < submitted; s++) {
            const uint32_t item = scratch_items[s];
            bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(s);
            const int device = scratch ? scratch->device : scratch_devices[s];
            if (device < 0 || device >= BGJ_CUDA_MAX_EXEC_DEVICES) continue;
            device_buckets[device]++;
            if (batch_split && item == split_item) {
                device_dots[device] +=
                    bgj_cuda_scale_work(item_dots[item],
                                        scratch_work_count[s],
                                        split_full_work);
            } else {
                device_dots[device] += item_dots[item];
            }
            device_results[device] += scratch_copied_counts[s];
            if (scratch_copied_overflows[s]) device_overflows[device]++;
        }
        for (int device = 0; device < BGJ_CUDA_MAX_EXEC_DEVICES; device++) {
            if (!device_buckets[device]) continue;
            bgj_cuda_device_profile_record(device,
                                           device_buckets[device],
                                           device_dots[device],
                                           device_results[device],
                                           device_overflows[device],
                                           0,
                                           submit_sec,
                                           wait_sec,
                                           copy_sec,
                                           bgj_cuda_wall_time() - total_t0);
        }
    } else {
        bgj_cuda_device_profile_record(selected_device,
                                       submitted,
                                       estimated_dots,
                                       total_results,
                                       total_overflows,
                                       0,
                                       submit_sec,
                                       wait_sec,
                                       copy_sec,
                                       bgj_cuda_wall_time() - total_t0);
    }
    bgj_cuda_phase_profile_record(batch_size,
                                  estimated_dots,
                                  total_results,
                                  total_overflows,
                                  0,
                                  batch_phase_profile);
    if (batch_split && bgj_cuda_split_profile_requested() &&
        item_dots[split_item] >= bgj_cuda_split_profile_min_dots()) {
        fprintf(stderr,
                "cuda_batch_split_profile: item=%u/%u devices=%d num_p=%u num_n=%u "
                "item_dots=%lu batch_dots=%lu results=%u overflow=%d "
                "submit=%.6fs wait=%.6fs copy=%.6fs total=%.6fs",
                split_item,
                batch_size,
                execution_device_count,
                num_p[split_item],
                num_n[split_item],
                (unsigned long)item_dots[split_item],
                (unsigned long)estimated_dots,
                result_count[split_item],
                overflow[split_item],
                submit_sec,
                wait_sec,
                copy_sec,
                bgj_cuda_wall_time() - total_t0);
        bgj_cuda_split_profile_print_phases(stderr, &split_item_phase_profile);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    if (item_dots != item_dots_stack) free(item_dots);
    if (item_devices != item_devices_stack) free(item_devices);
    if (scratch_items != scratch_items_stack) free(scratch_items);
    if (scratch_split_slots != scratch_split_slots_stack) free(scratch_split_slots);
    if (scratch_split_counts != scratch_split_counts_stack) free(scratch_split_counts);
    if (scratch_devices != scratch_devices_stack) free(scratch_devices);
    if (scratch_work_begin != scratch_work_begin_stack) free(scratch_work_begin);
    if (scratch_work_count != scratch_work_count_stack) free(scratch_work_count);
    if (scratch_result_counts != scratch_result_counts_stack) free(scratch_result_counts);
    if (scratch_copied_counts != scratch_copied_counts_stack) free(scratch_copied_counts);
    if (scratch_overflows != scratch_overflows_stack) free(scratch_overflows);
    if (scratch_copied_overflows != scratch_copied_overflows_stack) free(scratch_copied_overflows);
    return 1;

fail_sync:
    for (uint32_t i = 0; i < submitted; i++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
        if (scratch && scratch->stream_ready) {
            if (scratch->device >= 0) cudaSetDevice(scratch->device);
            cudaStreamSynchronize(scratch->stream);
        }
    }
    for (uint32_t i = 0; i < batch_size; i++) {
        total_results += result_count[i];
        if (overflow[i]) total_overflows++;
    }
    for (uint32_t i = 0; i < batch_size; i++) {
        bgj_cuda_raw_scratch_t *scratch = NULL;
        for (uint32_t s = 0; s < submitted; s++) {
            if (scratch_items[s] == i) {
                scratch = bgj_cuda_raw_batch_scratch.get(s);
                break;
            }
        }
        bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_BATCH_ITEM,
                                       scratch ? scratch->device : selected_device,
                                       batch_size,
                                       i,
                                       num_p[i],
                                       num_n[i],
                                       bgj_cuda_estimated_pair_dots(num_p[i], num_n[i]),
                                       result_count[i],
                                       overflow[i] ? 1 : 0,
                                       1,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0);
    }
    bgj_cuda_bucket_profile_record(BGJ_CUDA_BUCKET_PROFILE_BATCH,
                                   selected_device,
                                   batch_size,
                                   0,
                                   max_num_p,
                                   max_num_n,
                                   estimated_dots,
                                   total_results,
                                   total_overflows,
                                   1,
                                   submit_sec,
                                   wait_sec,
                                   copy_sec,
                                   bgj_cuda_wall_time() - total_t0);
    if (multi_gpu_batch) {
        uint64_t device_buckets[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        uint64_t device_dots[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        uint64_t device_results[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        uint64_t device_overflows[BGJ_CUDA_MAX_EXEC_DEVICES] = {};
        for (uint32_t s = 0; s < submitted; s++) {
            const uint32_t item = scratch_items[s];
            bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(s);
            const int device = scratch ? scratch->device : scratch_devices[s];
            if (device < 0 || device >= BGJ_CUDA_MAX_EXEC_DEVICES) continue;
            device_buckets[device]++;
            if (batch_split && item == split_item) {
                device_dots[device] +=
                    bgj_cuda_scale_work(item_dots[item],
                                        scratch_work_count[s],
                                        split_full_work);
            } else {
                device_dots[device] += item_dots[item];
            }
            device_results[device] += scratch_copied_counts[s];
            if (scratch_copied_overflows[s]) device_overflows[device]++;
        }
        for (int device = 0; device < BGJ_CUDA_MAX_EXEC_DEVICES; device++) {
            if (!device_buckets[device]) continue;
            bgj_cuda_device_profile_record(device,
                                           device_buckets[device],
                                           device_dots[device],
                                           device_results[device],
                                           device_overflows[device],
                                           1,
                                           submit_sec,
                                           wait_sec,
                                           copy_sec,
                                           bgj_cuda_wall_time() - total_t0);
        }
    } else {
        bgj_cuda_device_profile_record(selected_device,
                                       batch_size,
                                       estimated_dots,
                                       total_results,
                                       total_overflows,
                                       1,
                                       submit_sec,
                                       wait_sec,
                                       copy_sec,
                                       bgj_cuda_wall_time() - total_t0);
    }
    if (selected_device >= 0) bgj_cuda_set_current_device(selected_device);
    if (item_dots != item_dots_stack) free(item_dots);
    if (item_devices != item_devices_stack) free(item_devices);
    if (scratch_items != scratch_items_stack) free(scratch_items);
    if (scratch_split_slots != scratch_split_slots_stack) free(scratch_split_slots);
    if (scratch_split_counts != scratch_split_counts_stack) free(scratch_split_counts);
    if (scratch_devices != scratch_devices_stack) free(scratch_devices);
    if (scratch_work_begin != scratch_work_begin_stack) free(scratch_work_begin);
    if (scratch_work_count != scratch_work_count_stack) free(scratch_work_count);
    if (scratch_result_counts != scratch_result_counts_stack) free(scratch_result_counts);
    if (scratch_copied_counts != scratch_copied_counts_stack) free(scratch_copied_counts);
    if (scratch_overflows != scratch_overflows_stack) free(scratch_overflows);
    if (scratch_copied_overflows != scratch_copied_overflows_stack) free(scratch_copied_overflows);
    return 0;
}

#undef CUDA_ENSURE
#undef CUDA_TRY
#undef BGJ_CUDA_TENSOR_THREADS_PER_BLOCK
#undef BGJ_CUDA_TENSOR_WARPS_PER_BLOCK
#undef BGJ_CUDA_WARP_SIZE
