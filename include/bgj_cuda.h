#ifndef __BGJ_CUDA_H
#define __BGJ_CUDA_H

#include <stdint.h>

enum bgj_cuda_sol_type {
    BGJ_CUDA_SOL_A = 0,
    BGJ_CUDA_SOL_S = 1,
    BGJ_CUDA_SOL_AA = 2,
    BGJ_CUDA_SOL_SA = 3,
    BGJ_CUDA_SOL_SS = 4
};

struct bgj_cuda_result_t {
    uint32_t type;
    uint32_t x;
    uint32_t y;
};

struct bgj_cuda_materialize_desc_t {
    uint32_t type;
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct bgj_cuda_materialize_phase_profile_t {
    uint64_t chunks;
    uint64_t candidates;
    double pool_sec;
    double basis_sec;
    double desc_sec;
    double build_sec;
    double gemm_sec;
    double coeff_sec;
    double reconstruct_sec;
    double copy_sec;
};

struct alignas(16) bgj_cuda_bucket_entry_t {
    uint32_t bucket;
    uint32_t id;
    int32_t dot;
    uint32_t reserved;
};

int bgj_cuda_device_count();
const char *bgj_cuda_last_error();
int bgj_cuda_search_requested();
void bgj_cuda_set_search_requested(int enabled);
uint32_t bgj_cuda_batch_size(uint32_t host_threads);
uint64_t bgj_cuda_batch_min_dots();
int bgj_cuda_materialize_requested();
int bgj_cuda_bucket_requested();

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
                                         int *overflow);

extern "C" int bgj_cuda_lsh_search_raw(const uint8_t *sh,
                                        uint32_t mbound,
                                        uint32_t shsize,
                                        int32_t threshold,
                                        bgj_cuda_result_t *results,
                                        uint32_t result_capacity,
                                        uint32_t *result_count,
                                        int *overflow);

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
                                                  int32_t *dst_vsum);
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
                                                         int32_t *dst_vsum);
extern "C" int bgj_cuda_materialize_copy_staged_vectors_raw(const uint32_t *indices,
                                                             uint32_t count,
                                                             uint32_t vec_length,
                                                             int8_t *dst_vec);
extern "C" void bgj_cuda_materialize_finish_staged_raw();
extern "C" void bgj_cuda_materialize_last_profile(bgj_cuda_materialize_phase_profile_t *profile);

#endif
