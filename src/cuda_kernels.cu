#include "../include/bgj_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static char bgj_cuda_error[512] = "no CUDA error";

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
    int8_t *d_p_vecs = NULL;
    int8_t *d_n_vecs = NULL;
    uint32_t *d_p_ids = NULL;
    uint32_t *d_n_ids = NULL;
    int32_t *d_p_norm = NULL;
    int32_t *d_n_norm = NULL;
    int32_t *d_p_dot = NULL;
    int32_t *d_n_dot = NULL;
    bgj_cuda_result_t *d_results = NULL;
    uint32_t *d_result_count = NULL;
    int *d_overflow = NULL;
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
        CUDA_TRY(cudaMalloc((void **)&d_p_vecs, p_vec_bytes));
        CUDA_TRY(cudaMalloc((void **)&d_p_ids, p_id_bytes));
        CUDA_TRY(cudaMalloc((void **)&d_p_norm, p_i32_bytes));
        CUDA_TRY(cudaMemcpy(d_p_vecs, p_vecs, p_vec_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_p_ids, p_ids, p_id_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_p_norm, p_norm, p_i32_bytes, cudaMemcpyHostToDevice));
        if (record_dp) {
            CUDA_TRY(cudaMalloc((void **)&d_p_dot, p_i32_bytes));
            CUDA_TRY(cudaMemcpy(d_p_dot, p_dot, p_i32_bytes, cudaMemcpyHostToDevice));
        }
    }

    if (num_n) {
        CUDA_TRY(cudaMalloc((void **)&d_n_vecs, n_vec_bytes));
        CUDA_TRY(cudaMalloc((void **)&d_n_ids, n_id_bytes));
        CUDA_TRY(cudaMalloc((void **)&d_n_norm, n_i32_bytes));
        CUDA_TRY(cudaMemcpy(d_n_vecs, n_vecs, n_vec_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_n_ids, n_ids, n_id_bytes, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(d_n_norm, n_norm, n_i32_bytes, cudaMemcpyHostToDevice));
        if (record_dp) {
            CUDA_TRY(cudaMalloc((void **)&d_n_dot, n_i32_bytes));
            CUDA_TRY(cudaMemcpy(d_n_dot, n_dot, n_i32_bytes, cudaMemcpyHostToDevice));
        }
    }

    CUDA_TRY(cudaMalloc((void **)&d_results, (size_t)result_capacity * sizeof(bgj_cuda_result_t)));
    CUDA_TRY(cudaMalloc((void **)&d_result_count, sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc((void **)&d_overflow, sizeof(int)));
    CUDA_TRY(cudaMemset(d_result_count, 0, sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(d_overflow, 0, sizeof(int)));

    if (num_p && num_n) {
        const uint64_t total = (uint64_t)num_p * (uint64_t)num_n;
        const uint32_t threads = 256;
        uint32_t blocks = (uint32_t)((total + threads - 1) / threads);
        if (blocks == 0) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        bgj_cuda_search_np_kernel<<<blocks, threads>>>(d_p_vecs,
                                                       d_n_vecs,
                                                       d_p_ids,
                                                       d_n_ids,
                                                       d_p_norm,
                                                       d_n_norm,
                                                       d_p_dot,
                                                       d_n_dot,
                                                       num_p,
                                                       num_n,
                                                       vec_length,
                                                       goal_norm,
                                                       goal_norm - center_norm,
                                                       record_dp,
                                                       d_results,
                                                       result_capacity,
                                                       d_result_count,
                                                       d_overflow);
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
        bgj_cuda_search_same_kernel<<<blocks, threads>>>(d_p_vecs,
                                                         d_p_ids,
                                                         d_p_norm,
                                                         d_p_dot,
                                                         num_p,
                                                         vec_length,
                                                         goal_norm,
                                                         goal_norm - center_norm,
                                                         record_dp,
                                                         0,
                                                         d_results,
                                                         result_capacity,
                                                         d_result_count,
                                                         d_overflow);
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
        bgj_cuda_search_same_kernel<<<blocks, threads>>>(d_n_vecs,
                                                         d_n_ids,
                                                         d_n_norm,
                                                         d_n_dot,
                                                         num_n,
                                                         vec_length,
                                                         goal_norm,
                                                         goal_norm - center_norm,
                                                         record_dp,
                                                         1,
                                                         d_results,
                                                         result_capacity,
                                                         d_result_count,
                                                         d_overflow);
        CUDA_TRY(cudaGetLastError());
    }

    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(&h_result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));

    *overflow = h_overflow || (h_result_count > result_capacity);
    *result_count = h_result_count > result_capacity ? result_capacity : h_result_count;
    if (!*overflow && *result_count) {
        CUDA_TRY(cudaMemcpy(results,
                            d_results,
                            (size_t)(*result_count) * sizeof(bgj_cuda_result_t),
                            cudaMemcpyDeviceToHost));
    }

    cudaFree(d_p_vecs);
    cudaFree(d_n_vecs);
    cudaFree(d_p_ids);
    cudaFree(d_n_ids);
    cudaFree(d_p_norm);
    cudaFree(d_n_norm);
    cudaFree(d_p_dot);
    cudaFree(d_n_dot);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaFree(d_overflow);
    set_plain_error("no CUDA error");
    return 1;

fail:
    cudaFree(d_p_vecs);
    cudaFree(d_n_vecs);
    cudaFree(d_p_ids);
    cudaFree(d_n_ids);
    cudaFree(d_p_norm);
    cudaFree(d_n_norm);
    cudaFree(d_p_dot);
    cudaFree(d_n_dot);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaFree(d_overflow);
    return 0;
}

#undef CUDA_TRY
