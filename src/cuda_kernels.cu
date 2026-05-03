#include "../include/bgj_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static char bgj_cuda_error[512] = "no CUDA error";

struct bgj_cuda_raw_scratch_t {
    int8_t *p_vecs;
    int8_t *n_vecs;
    uint32_t *p_ids;
    uint32_t *n_ids;
    int32_t *p_norm;
    int32_t *n_norm;
    int32_t *p_dot;
    int32_t *n_dot;
    bgj_cuda_result_t *results;
    uint32_t *result_count;
    int *overflow;

    size_t p_vec_capacity;
    size_t n_vec_capacity;
    size_t p_id_capacity;
    size_t n_id_capacity;
    size_t p_i32_capacity;
    size_t n_i32_capacity;
    size_t p_dot_capacity;
    size_t n_dot_capacity;
    size_t result_capacity;
    size_t result_count_capacity;
    size_t overflow_capacity;

    bgj_cuda_raw_scratch_t()
        : p_vecs(NULL),
          n_vecs(NULL),
          p_ids(NULL),
          n_ids(NULL),
          p_norm(NULL),
          n_norm(NULL),
          p_dot(NULL),
          n_dot(NULL),
          results(NULL),
          result_count(NULL),
          overflow(NULL),
          p_vec_capacity(0),
          n_vec_capacity(0),
          p_id_capacity(0),
          n_id_capacity(0),
          p_i32_capacity(0),
          n_i32_capacity(0),
          p_dot_capacity(0),
          n_dot_capacity(0),
          result_capacity(0),
          result_count_capacity(0),
          overflow_capacity(0)
    {
    }

    ~bgj_cuda_raw_scratch_t()
    {
        cudaFree(p_vecs);
        cudaFree(n_vecs);
        cudaFree(p_ids);
        cudaFree(n_ids);
        cudaFree(p_norm);
        cudaFree(n_norm);
        cudaFree(p_dot);
        cudaFree(n_dot);
        cudaFree(results);
        cudaFree(result_count);
        cudaFree(overflow);
    }
};

static thread_local bgj_cuda_raw_scratch_t bgj_cuda_raw_scratch;

static void set_cuda_error(const char *context, cudaError_t err)
{
    snprintf(bgj_cuda_error, sizeof(bgj_cuda_error), "%s: %s", context, cudaGetErrorString(err));
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

__device__ int bgj_cuda_dot_i8(const int8_t *a, const int8_t *b, uint32_t vec_length)
{
    int acc = 0;
    for (uint32_t i = 0; i < vec_length; i++) {
        acc += (int)a[i] * (int)b[i];
    }
    return acc;
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

__global__ void bgj_cuda_search_np_kernel(const int8_t *p_vecs,
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
                                          bgj_cuda_result_t *results,
                                          uint32_t result_capacity,
                                          uint32_t *result_count,
                                          int *overflow)
{
    const uint64_t total = (uint64_t)num_p * (uint64_t)num_n;
    const uint64_t stride = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    uint64_t pair = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;

    for (; pair < total; pair += stride) {
        const uint32_t p = (uint32_t)(pair / num_n);
        const uint32_t n = (uint32_t)(pair - (uint64_t)p * num_n);
        const int32_t dp = bgj_cuda_dot_i8(p_vecs + (uint64_t)p * vec_length,
                                           n_vecs + (uint64_t)n * vec_length,
                                           vec_length);

        if (p_norm[p] + n_norm[n] + dp < goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_A, p_ids[p], n_ids[n]);
        }
        if (record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm) {
            bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                 BGJ_CUDA_SOL_SA, p_ids[p], n_ids[n]);
        }
    }
}

__global__ void bgj_cuda_search_same_kernel(const int8_t *vecs,
                                            const uint32_t *ids,
                                            const int32_t *norm,
                                            const int32_t *center_dot,
                                            uint32_t num_vecs,
                                            uint32_t vec_length,
                                            int32_t goal_norm,
                                            int32_t center_goal_norm,
                                            int record_dp,
                                            int negative_bucket,
                                            bgj_cuda_result_t *results,
                                            uint32_t result_capacity,
                                            uint32_t *result_count,
                                            int *overflow)
{
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_vecs || j >= num_vecs || j <= i) return;

    const int32_t dp = bgj_cuda_dot_i8(vecs + (uint64_t)i * vec_length,
                                       vecs + (uint64_t)j * vec_length,
                                       vec_length);

    if (norm[i] + norm[j] - dp < goal_norm) {
        bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                             BGJ_CUDA_SOL_S, ids[i], ids[j]);
    }
    if (record_dp && center_dot[i] + center_dot[j] + dp < center_goal_norm) {
        bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                             negative_bucket ? BGJ_CUDA_SOL_AA : BGJ_CUDA_SOL_SS,
                             ids[i], ids[j]);
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

#define CUDA_ENSURE(ptr, capacity, requested)                         \
    do {                                                              \
        if (!ensure_cuda_capacity((void **)&(ptr), &(capacity), requested)) goto fail; \
    } while (0)

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
                                           int *overflow)
{
    bgj_cuda_raw_scratch_t *scratch = &bgj_cuda_raw_scratch;
    uint32_t h_result_count = 0;
    int h_overflow = 0;

    if (result_capacity == 0) {
        set_plain_error("result capacity is zero");
        return 0;
    }

    const size_t p_vec_bytes = (size_t)num_p * (size_t)vec_length * sizeof(int8_t);
    const size_t n_vec_bytes = (size_t)num_n * (size_t)vec_length * sizeof(int8_t);
    const size_t p_id_bytes = (size_t)num_p * sizeof(uint32_t);
    const size_t n_id_bytes = (size_t)num_n * sizeof(uint32_t);
    const size_t p_i32_bytes = (size_t)num_p * sizeof(int32_t);
    const size_t n_i32_bytes = (size_t)num_n * sizeof(int32_t);

    if (num_p) {
        CUDA_ENSURE(scratch->p_vecs, scratch->p_vec_capacity, p_vec_bytes);
        CUDA_ENSURE(scratch->p_ids, scratch->p_id_capacity, p_id_bytes);
        CUDA_ENSURE(scratch->p_norm, scratch->p_i32_capacity, p_i32_bytes);
        CUDA_TRY(cudaMemcpy(scratch->p_vecs, p_vecs, p_vec_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(scratch->p_ids, p_ids, p_id_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(scratch->p_norm, p_norm, p_i32_bytes, cudaMemcpyHostToDevice));
        if (record_dp) {
            CUDA_ENSURE(scratch->p_dot, scratch->p_dot_capacity, p_i32_bytes);
            CUDA_TRY(cudaMemcpy(scratch->p_dot, p_dot, p_i32_bytes, cudaMemcpyHostToDevice));
        }
    }

    if (num_n) {
        CUDA_ENSURE(scratch->n_vecs, scratch->n_vec_capacity, n_vec_bytes);
        CUDA_ENSURE(scratch->n_ids, scratch->n_id_capacity, n_id_bytes);
        CUDA_ENSURE(scratch->n_norm, scratch->n_i32_capacity, n_i32_bytes);
        CUDA_TRY(cudaMemcpy(scratch->n_vecs, n_vecs, n_vec_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(scratch->n_ids, n_ids, n_id_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(scratch->n_norm, n_norm, n_i32_bytes, cudaMemcpyHostToDevice));
        if (record_dp) {
            CUDA_ENSURE(scratch->n_dot, scratch->n_dot_capacity, n_i32_bytes);
            CUDA_TRY(cudaMemcpy(scratch->n_dot, n_dot, n_i32_bytes, cudaMemcpyHostToDevice));
        }
    }

    CUDA_ENSURE(scratch->results, scratch->result_capacity, (size_t)result_capacity * sizeof(bgj_cuda_result_t));
    CUDA_ENSURE(scratch->result_count, scratch->result_count_capacity, sizeof(uint32_t));
    CUDA_ENSURE(scratch->overflow, scratch->overflow_capacity, sizeof(int));
    CUDA_TRY(cudaMemset(scratch->result_count, 0, sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(scratch->overflow, 0, sizeof(int)));

    if (num_p && num_n) {
        const uint64_t total = (uint64_t)num_p * (uint64_t)num_n;
        const uint32_t threads = 256;
        uint32_t blocks = (uint32_t)((total + threads - 1) / threads);
        if (blocks == 0) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        bgj_cuda_search_np_kernel<<<blocks, threads>>>(scratch->p_vecs,
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
                                                       goal_norm - center_norm,
                                                       record_dp,
                                                       scratch->results,
                                                       result_capacity,
                                                       scratch->result_count,
                                                       scratch->overflow);
        CUDA_TRY(cudaGetLastError());
    }

    if (num_p > 1) {
        dim3 threads(16, 16);
        dim3 blocks((num_p + threads.x - 1) / threads.x,
                    (num_p + threads.y - 1) / threads.y);
        if (blocks.x > 65535 || blocks.y > 65535) {
            set_plain_error("bucket too large for same-side CUDA grid");
            goto fail;
        }
        bgj_cuda_search_same_kernel<<<blocks, threads>>>(scratch->p_vecs,
                                                         scratch->p_ids,
                                                         scratch->p_norm,
                                                         scratch->p_dot,
                                                         num_p,
                                                         vec_length,
                                                         goal_norm,
                                                         goal_norm - center_norm,
                                                         record_dp,
                                                         0,
                                                         scratch->results,
                                                         result_capacity,
                                                         scratch->result_count,
                                                         scratch->overflow);
        CUDA_TRY(cudaGetLastError());
    }

    if (num_n > 1) {
        dim3 threads(16, 16);
        dim3 blocks((num_n + threads.x - 1) / threads.x,
                    (num_n + threads.y - 1) / threads.y);
        if (blocks.x > 65535 || blocks.y > 65535) {
            set_plain_error("bucket too large for same-side CUDA grid");
            goto fail;
        }
        bgj_cuda_search_same_kernel<<<blocks, threads>>>(scratch->n_vecs,
                                                         scratch->n_ids,
                                                         scratch->n_norm,
                                                         scratch->n_dot,
                                                         num_n,
                                                         vec_length,
                                                         goal_norm,
                                                         goal_norm - center_norm,
                                                         record_dp,
                                                         1,
                                                         scratch->results,
                                                         result_capacity,
                                                         scratch->result_count,
                                                         scratch->overflow);
        CUDA_TRY(cudaGetLastError());
    }

    CUDA_TRY(cudaMemcpy(&h_result_count, scratch->result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(&h_overflow, scratch->overflow, sizeof(int), cudaMemcpyDeviceToHost));

    *overflow = h_overflow || (h_result_count > result_capacity);
    *result_count = h_result_count > result_capacity ? result_capacity : h_result_count;
    if (!*overflow && *result_count) {
        CUDA_TRY(cudaMemcpy(results,
                            scratch->results,
                            (size_t)(*result_count) * sizeof(bgj_cuda_result_t),
                            cudaMemcpyDeviceToHost));
    }

    set_plain_error("no CUDA error");
    return 1;

fail:
    return 0;
}

#undef CUDA_ENSURE
#undef CUDA_TRY
