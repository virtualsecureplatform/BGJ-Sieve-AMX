#include "../include/bgj_cuda.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <pthread.h>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace nvcuda;

static char bgj_cuda_error[512] = "no CUDA error";

#define BGJ_CUDA_WARP_SIZE 32u
#define BGJ_CUDA_TENSOR_WARPS_PER_BLOCK 4u
#define BGJ_CUDA_TENSOR_THREADS_PER_BLOCK (BGJ_CUDA_WARP_SIZE * BGJ_CUDA_TENSOR_WARPS_PER_BLOCK)
#define BGJ_CUDA_TENSOR_NP_WIDE_TILES 4u
#define BGJ_CUDA_TENSOR_NP_MULTI_P_TILES 2u
#define BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP 4u
#define BGJ_CUDA_TENSOR_NP_MULTI_N_TILES (BGJ_CUDA_TENSOR_WARPS_PER_BLOCK * BGJ_CUDA_TENSOR_NP_MULTI_N_TILES_PER_WARP)

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
    int *overflow;
    cudaStream_t stream;

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
    size_t overflow_capacity;
    int stream_ready;

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
          overflow(NULL),
          stream(NULL),
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
          overflow_capacity(0),
          stream_ready(0)
    {
    }

    ~bgj_cuda_raw_scratch_t()
    {
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
        cudaFree(overflow);
        if (stream_ready) cudaStreamDestroy(stream);
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
        if (record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm) {
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
        if (record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm) {
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
                if (record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm) {
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
            if (record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm) {
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
            if (record_dp && p_dot[p] + n_dot[n] - dp < center_goal_norm) {
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
        if (record_dp && center_dot[i] + center_dot[j] + dp < center_goal_norm) {
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

struct bgj_cuda_shared_pool_cache_t {
    int8_t *vecs;
    size_t capacity;
    const int8_t *host_key;
    uint64_t epoch;
    uint32_t pool_size;
    uint32_t vec_length;
    pthread_mutex_t lock;

    bgj_cuda_shared_pool_cache_t()
        : vecs(NULL),
          capacity(0),
          host_key(NULL),
          epoch(0),
          pool_size(0),
          vec_length(0)
    {
        pthread_mutex_init(&lock, NULL);
    }

    ~bgj_cuda_shared_pool_cache_t()
    {
        cudaFree(vecs);
        pthread_mutex_destroy(&lock);
    }
};

static bgj_cuda_shared_pool_cache_t bgj_cuda_shared_pool_cache;

static int bgj_cuda_prepare_shared_pool_cache(const int8_t *pool_vecs_host,
                                              uint64_t pool_epoch,
                                              uint32_t pool_size,
                                              uint32_t vec_length,
                                              size_t pool_vec_bytes,
                                              cudaStream_t stream,
                                              int8_t **pool_vecs_device)
{
    bgj_cuda_shared_pool_cache_t *cache = &bgj_cuda_shared_pool_cache;
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

static int bgj_cuda_tensor_np_wide_requested()
{
    const char *env = getenv("BGJ_CUDA_TENSOR_NP_WIDE");
    if (env && env[0]) return env[0] != '0';
    return 0;
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
                                             int32_t center_norm,
                                             int record_dp,
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
    int8_t *pool_vecs_device = NULL;

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
        if (!bgj_cuda_prepare_shared_pool_cache(pool_vecs_host,
                                                pool_epoch,
                                                pool_size,
                                                vec_length,
                                                pool_vec_bytes,
                                                stream,
                                                &pool_vecs_device)) goto fail;
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
            bgj_cuda_pack_pool_vecs_kernel<<<blocks, threads, 0, stream>>>(pool_vecs_device,
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
            bgj_cuda_pack_pool_vecs_kernel<<<blocks, threads, 0, stream>>>(pool_vecs_device,
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
        tensor_np_min_tiles = bgj_cuda_tensor_np_min_tiles();
        tensor_same_min_tiles = bgj_cuda_tensor_same_min_tiles();
        if (num_p >= 16 && num_n >= 16) {
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
        }

        if (bgj_cuda_tensor_same_requested() && num_p >= 16) {
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
        }

        if (bgj_cuda_tensor_same_requested() && num_n >= 16) {
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

    CUDA_TRY(cudaMemcpyAsync(submitted_result_count, scratch->result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaMemcpyAsync(submitted_overflow, scratch->overflow, sizeof(int), cudaMemcpyDeviceToHost, stream));
    set_plain_error("no CUDA error");
    return 1;

fail:
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
    if (!*overflow && *result_count) {
        CUDA_TRY(cudaMemcpyAsync(results,
                                 scratch->results,
                                 (size_t)(*result_count) * sizeof(bgj_cuda_result_t),
                                 cudaMemcpyDeviceToHost,
                                 stream));
    }

    set_plain_error("no CUDA error");
    return 1;

fail:
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
                                           center_norm,
                                           record_dp,
                                           result_capacity,
                                           &h_result_count,
                                           &h_overflow)) {
        return 0;
    }
    cudaStream_t stream = scratch->stream;
    CUDA_TRY(cudaStreamSynchronize(stream));

    if (!bgj_cuda_finish_submitted_bucket(scratch,
                                          results,
                                          result_capacity,
                                          h_result_count,
                                          h_overflow,
                                          result_count,
                                          overflow)) {
        return 0;
    }
    CUDA_TRY(cudaStreamSynchronize(stream));

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
                                                      int *overflow)
{
    if (batch_size == 0) {
        set_plain_error("no CUDA error");
        return 1;
    }
    if (!pool_vecs || !p_ids || !n_ids || !p_norm || !n_norm ||
        !num_p || !num_n || !goal_norm || !center_norm ||
        !results || !result_capacity || !result_count || !overflow) {
        set_plain_error("invalid batch pointer");
        return 0;
    }
    if (record_dp && (!p_dot || !n_dot)) {
        set_plain_error("invalid batch dot pointer");
        return 0;
    }

    uint32_t submitted = 0;
    for (uint32_t i = 0; i < batch_size; i++) {
        result_count[i] = 0;
        overflow[i] = 0;
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
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
                                               center_norm[i],
                                               record_dp,
                                               result_capacity[i],
                                               &result_count[i],
                                               &overflow[i])) {
            goto fail_sync;
        }
        submitted++;
    }

    for (uint32_t i = 0; i < submitted; i++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize batch counts", err);
            goto fail_sync;
        }
    }

    for (uint32_t i = 0; i < submitted; i++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
        const uint32_t submitted_count = result_count[i];
        const int submitted_overflow = overflow[i];
        if (!bgj_cuda_finish_submitted_bucket(scratch,
                                              results[i],
                                              result_capacity[i],
                                              submitted_count,
                                              submitted_overflow,
                                              &result_count[i],
                                              &overflow[i])) {
            goto fail_sync;
        }
    }

    for (uint32_t i = 0; i < submitted; i++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
        cudaError_t err = cudaStreamSynchronize(scratch->stream);
        if (err != cudaSuccess) {
            set_cuda_error("cudaStreamSynchronize batch results", err);
            goto fail_sync;
        }
    }

    set_plain_error("no CUDA error");
    return 1;

fail_sync:
    for (uint32_t i = 0; i < submitted; i++) {
        bgj_cuda_raw_scratch_t *scratch = bgj_cuda_raw_batch_scratch.get(i);
        if (scratch && scratch->stream_ready) cudaStreamSynchronize(scratch->stream);
    }
    return 0;
}

#undef CUDA_ENSURE
#undef CUDA_TRY
#undef BGJ_CUDA_TENSOR_THREADS_PER_BLOCK
#undef BGJ_CUDA_TENSOR_WARPS_PER_BLOCK
#undef BGJ_CUDA_WARP_SIZE
