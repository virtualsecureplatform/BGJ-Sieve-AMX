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

int bgj_cuda_device_count();
const char *bgj_cuda_last_error();
int bgj_cuda_search_requested();
void bgj_cuda_set_search_requested(int enabled);
uint32_t bgj_cuda_batch_size(uint32_t host_threads);
uint64_t bgj_cuda_batch_min_dots();
int bgj_cuda_materialize_requested();

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

#endif
