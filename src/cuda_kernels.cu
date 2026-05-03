#include "../include/bgj_cuda.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace nvcuda;

static char bgj_cuda_error[512] = "no CUDA error";

struct bgj_cuda_raw_scratch_t {
    int8_t *pool_vecs;
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
    cudaStream_t stream;

    size_t pool_vec_capacity;
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
    const int8_t *pool_host_key;
    uint64_t pool_epoch;
    uint32_t pool_size;
    uint32_t pool_vec_length;
    int stream_ready;

    bgj_cuda_raw_scratch_t()
        : pool_vecs(NULL),
          p_vecs(NULL),
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
          stream(NULL),
          pool_vec_capacity(0),
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
          overflow_capacity(0),
          pool_host_key(NULL),
          pool_epoch(0),
          pool_size(0),
          pool_vec_length(0),
          stream_ready(0)
    {
    }

    ~bgj_cuda_raw_scratch_t()
    {
        if (stream_ready) cudaStreamSynchronize(stream);
        cudaFree(pool_vecs);
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
        if (stream_ready) cudaStreamDestroy(stream);
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

__global__ void bgj_cuda_search_np_tensor_kernel(const int8_t *p_vecs,
                                                 const int8_t *n_vecs,
                                                 const uint32_t *p_ids,
                                                 const uint32_t *n_ids,
                                                 const int32_t *p_norm,
                                                 const int32_t *n_norm,
                                                 const int32_t *p_dot,
                                                 const int32_t *n_dot,
                                                 uint32_t vec_length,
                                                 int32_t goal_norm,
                                                 int32_t center_goal_norm,
                                                 int record_dp,
                                                 bgj_cuda_result_t *results,
                                                 uint32_t result_capacity,
                                                 uint32_t *result_count,
                                                 int *overflow)
{
    const uint32_t tile_p = blockIdx.y * 16u;
    const uint32_t tile_n = blockIdx.x * 16u;
    __shared__ int32_t dp_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k = 0; k < vec_length; k += 16) {
        wmma::load_matrix_sync(a_frag, (const signed char *)(p_vecs + (uint64_t)tile_p * vec_length + k), vec_length);
        wmma::load_matrix_sync(b_frag, (const signed char *)(n_vecs + (uint64_t)tile_n * vec_length + k), vec_length);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(dp_tile, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    for (uint32_t idx = threadIdx.x; idx < 16u * 16u; idx += blockDim.x) {
        const uint32_t p = tile_p + idx / 16u;
        const uint32_t n = tile_n + idx % 16u;
        const int32_t dp = dp_tile[idx];

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

__global__ void bgj_cuda_search_same_tensor_kernel(const int8_t *vecs,
                                                   const uint32_t *ids,
                                                   const int32_t *norm,
                                                   const int32_t *center_dot,
                                                   uint32_t num_tiles,
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
    uint32_t tile_i;
    uint32_t tile_j;
    bgj_cuda_upper_tile_from_linear(blockIdx.x, num_tiles, &tile_i, &tile_j);
    const uint32_t base_i = tile_i * 16u;
    const uint32_t base_j = tile_j * 16u;
    __shared__ int32_t dp_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (uint32_t k = 0; k < vec_length; k += 16) {
        wmma::load_matrix_sync(a_frag, (const signed char *)(vecs + (uint64_t)base_i * vec_length + k), vec_length);
        wmma::load_matrix_sync(b_frag, (const signed char *)(vecs + (uint64_t)base_j * vec_length + k), vec_length);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(dp_tile, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    for (uint32_t idx = threadIdx.x; idx < 16u * 16u; idx += blockDim.x) {
        const uint32_t i = base_i + idx / 16u;
        const uint32_t j = base_j + idx % 16u;
        if (j <= i) continue;

        const int32_t dp = dp_tile[idx];
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
            if (record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm) {
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
            if (record_dp && p_dot[i] + p_dot[j] + dp < center_goal_norm) {
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
            if (record_dp && n_dot[i] + n_dot[j] + dp < center_goal_norm) {
                bgj_cuda_push_result(results, result_count, overflow, result_capacity,
                                     BGJ_CUDA_SOL_AA, n_ids[i], n_ids[j]);
            }
        }
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

static int bgj_cuda_prepare_stream(bgj_cuda_raw_scratch_t *scratch)
{
    if (scratch->stream_ready) return 1;
    cudaError_t err = cudaStreamCreateWithFlags(&scratch->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        set_cuda_error("cudaStreamCreateWithFlags", err);
        return 0;
    }
    scratch->stream_ready = 1;
    return 1;
}

#define CUDA_ENSURE(ptr, capacity, requested)                         \
    do {                                                              \
        if (!ensure_cuda_capacity((void **)&(ptr), &(capacity), requested)) goto fail; \
    } while (0)

static int bgj_cuda_tensor_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR");
    if (env && env[0]) return env[0] != '0';
    return 1;
}

static int bgj_cuda_tensor_same_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_SAME");
    if (env && env[0]) return env[0] != '0';
    return 1;
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
    static int capable = -1;
    if (capable >= 0) return capable;

    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        capable = 0;
        return capable;
    }
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        capable = 0;
        return capable;
    }
    capable = prop.major >= 8 ? 1 : 0;
    return capable;
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
    uint32_t tensor_np_num_p = 0;
    uint32_t tensor_np_num_n = 0;
    uint32_t tensor_same_num_p = 0;
    uint32_t tensor_same_num_n = 0;
    uint32_t tensor_same_min_tiles = 0;
    const int use_pool = pool_vecs_host != NULL;

    if (result_capacity == 0) {
        set_plain_error("result capacity is zero");
        return 0;
    }
    if (use_pool && pool_size == 0 && (num_p || num_n)) {
        set_plain_error("pool size is zero");
        return 0;
    }
    if (!bgj_cuda_prepare_stream(scratch)) return 0;
    cudaStream_t stream = scratch->stream;

    const size_t p_vec_bytes = (size_t)num_p * (size_t)vec_length * sizeof(int8_t);
    const size_t n_vec_bytes = (size_t)num_n * (size_t)vec_length * sizeof(int8_t);
    const size_t p_id_bytes = (size_t)num_p * sizeof(uint32_t);
    const size_t n_id_bytes = (size_t)num_n * sizeof(uint32_t);
    const size_t p_i32_bytes = (size_t)num_p * sizeof(int32_t);
    const size_t n_i32_bytes = (size_t)num_n * sizeof(int32_t);

    if (use_pool) {
        const size_t pool_vec_bytes = (size_t)pool_size * (size_t)vec_length * sizeof(int8_t);
        if (scratch->pool_host_key != pool_vecs_host ||
            scratch->pool_epoch != pool_epoch ||
            scratch->pool_size != pool_size ||
            scratch->pool_vec_length != vec_length) {
            CUDA_ENSURE(scratch->pool_vecs, scratch->pool_vec_capacity, pool_vec_bytes);
            CUDA_TRY(cudaMemcpyAsync(scratch->pool_vecs,
                                     pool_vecs_host,
                                     pool_vec_bytes,
                                     cudaMemcpyHostToDevice,
                                     stream));
            scratch->pool_host_key = pool_vecs_host;
            scratch->pool_epoch = pool_epoch;
            scratch->pool_size = pool_size;
            scratch->pool_vec_length = vec_length;
        }
    }

    if (num_p) {
        CUDA_ENSURE(scratch->p_vecs, scratch->p_vec_capacity, p_vec_bytes);
        CUDA_ENSURE(scratch->p_ids, scratch->p_id_capacity, p_id_bytes);
        CUDA_ENSURE(scratch->p_norm, scratch->p_i32_capacity, p_i32_bytes);
        CUDA_TRY(cudaMemcpyAsync(scratch->p_ids, p_ids, p_id_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaMemcpyAsync(scratch->p_norm, p_norm, p_i32_bytes, cudaMemcpyHostToDevice, stream));
        if (use_pool) {
            const uint32_t threads = 256;
            uint32_t blocks = (uint32_t)((p_vec_bytes + threads - 1) / threads);
            if (blocks == 0) blocks = 1;
            if (blocks > 65535) blocks = 65535;
            bgj_cuda_pack_pool_vecs_kernel<<<blocks, threads, 0, stream>>>(scratch->pool_vecs,
                                                                           scratch->p_ids,
                                                                           num_p,
                                                                           vec_length,
                                                                           scratch->p_vecs);
            CUDA_TRY(cudaGetLastError());
        } else {
            CUDA_TRY(cudaMemcpyAsync(scratch->p_vecs, p_vecs, p_vec_bytes, cudaMemcpyHostToDevice, stream));
        }
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
        if (use_pool) {
            const uint32_t threads = 256;
            uint32_t blocks = (uint32_t)((n_vec_bytes + threads - 1) / threads);
            if (blocks == 0) blocks = 1;
            if (blocks > 65535) blocks = 65535;
            bgj_cuda_pack_pool_vecs_kernel<<<blocks, threads, 0, stream>>>(scratch->pool_vecs,
                                                                           scratch->n_ids,
                                                                           num_n,
                                                                           vec_length,
                                                                           scratch->n_vecs);
            CUDA_TRY(cudaGetLastError());
        } else {
            CUDA_TRY(cudaMemcpyAsync(scratch->n_vecs, n_vecs, n_vec_bytes, cudaMemcpyHostToDevice, stream));
        }
        if (record_dp) {
            CUDA_ENSURE(scratch->n_dot, scratch->n_dot_capacity, n_i32_bytes);
            CUDA_TRY(cudaMemcpyAsync(scratch->n_dot, n_dot, n_i32_bytes, cudaMemcpyHostToDevice, stream));
        }
    }

    CUDA_ENSURE(scratch->results, scratch->result_capacity, (size_t)result_capacity * sizeof(bgj_cuda_result_t));
    CUDA_ENSURE(scratch->result_count, scratch->result_count_capacity, sizeof(uint32_t));
    CUDA_ENSURE(scratch->overflow, scratch->overflow_capacity, sizeof(int));
    CUDA_TRY(cudaMemsetAsync(scratch->result_count, 0, sizeof(uint32_t), stream));
    CUDA_TRY(cudaMemsetAsync(scratch->overflow, 0, sizeof(int), stream));

    if (bgj_cuda_tensor_requested() &&
        bgj_cuda_tensor_capable() &&
        vec_length % 16 == 0) {
        tensor_same_min_tiles = bgj_cuda_tensor_same_min_tiles();
        if (num_p >= 16 && num_n >= 16) {
            tensor_np_num_p = (num_p / 16u) * 16u;
            tensor_np_num_n = (num_n / 16u) * 16u;
            const uint32_t tensor_blocks_p = tensor_np_num_p / 16u;
            const uint32_t tensor_blocks_n = tensor_np_num_n / 16u;
            if (tensor_blocks_p <= 65535u && tensor_blocks_n <= 65535u) {
                dim3 blocks(tensor_blocks_n, tensor_blocks_p);
                bgj_cuda_search_np_tensor_kernel<<<blocks, 32, 0, stream>>>(scratch->p_vecs,
                                                                            scratch->n_vecs,
                                                                            scratch->p_ids,
                                                                            scratch->n_ids,
                                                                            scratch->p_norm,
                                                                            scratch->n_norm,
                                                                            scratch->p_dot,
                                                                            scratch->n_dot,
                                                                            vec_length,
                                                                            goal_norm,
                                                                            goal_norm - center_norm,
                                                                            record_dp,
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

        if (bgj_cuda_tensor_same_requested() && num_p >= 16) {
            tensor_same_num_p = (num_p / 16u) * 16u;
            const uint32_t tensor_blocks_p = tensor_same_num_p / 16u;
            const uint64_t tensor_tiles_p = ((uint64_t)tensor_blocks_p * (tensor_blocks_p + 1u)) / 2u;
            if (tensor_blocks_p >= tensor_same_min_tiles && tensor_tiles_p <= 0x7fffffffu) {
                bgj_cuda_search_same_tensor_kernel<<<(uint32_t)tensor_tiles_p, 32, 0, stream>>>(scratch->p_vecs,
                                                                                               scratch->p_ids,
                                                                                               scratch->p_norm,
                                                                                               scratch->p_dot,
                                                                                               tensor_blocks_p,
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
            } else {
                tensor_same_num_p = 0;
            }
        }

        if (bgj_cuda_tensor_same_requested() && num_n >= 16) {
            tensor_same_num_n = (num_n / 16u) * 16u;
            const uint32_t tensor_blocks_n = tensor_same_num_n / 16u;
            const uint64_t tensor_tiles_n = ((uint64_t)tensor_blocks_n * (tensor_blocks_n + 1u)) / 2u;
            if (tensor_blocks_n >= tensor_same_min_tiles && tensor_tiles_n <= 0x7fffffffu) {
                bgj_cuda_search_same_tensor_kernel<<<(uint32_t)tensor_tiles_n, 32, 0, stream>>>(scratch->n_vecs,
                                                                                               scratch->n_ids,
                                                                                               scratch->n_norm,
                                                                                               scratch->n_dot,
                                                                                               tensor_blocks_n,
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
            } else {
                tensor_same_num_n = 0;
            }
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
        const uint32_t threads = 256;
        uint32_t blocks = (uint32_t)((total + threads - 1) / threads);
        if (blocks == 0) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        if (total) {
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
                                                                          scratch->results,
                                                                          result_capacity,
                                                                          scratch->result_count,
                                                                          scratch->overflow);
            CUDA_TRY(cudaGetLastError());
        }
    }

    CUDA_TRY(cudaMemcpyAsync(&h_result_count, scratch->result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaMemcpyAsync(&h_overflow, scratch->overflow, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    *overflow = h_overflow || (h_result_count > result_capacity);
    *result_count = h_result_count > result_capacity ? result_capacity : h_result_count;
    if (!*overflow && *result_count) {
        CUDA_TRY(cudaMemcpyAsync(results,
                                 scratch->results,
                                 (size_t)(*result_count) * sizeof(bgj_cuda_result_t),
                                 cudaMemcpyDeviceToHost,
                                 stream));
        CUDA_TRY(cudaStreamSynchronize(stream));
    }

    set_plain_error("no CUDA error");
    return 1;

fail:
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
                                           int32_t center_norm,
                                           int record_dp,
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
                                           center_norm,
                                           record_dp,
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
                                                int32_t center_norm,
                                                int record_dp,
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
                                           center_norm,
                                           record_dp,
                                           results,
                                           result_capacity,
                                           result_count,
                                           overflow);
}

#undef CUDA_ENSURE
#undef CUDA_TRY
