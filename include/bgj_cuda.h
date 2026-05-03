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

int bgj_cuda_device_count();
const char *bgj_cuda_last_error();
int bgj_cuda_search_requested();
void bgj_cuda_set_search_requested(int enabled);

#endif
