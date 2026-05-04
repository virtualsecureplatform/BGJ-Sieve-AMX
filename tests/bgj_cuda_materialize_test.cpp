#include "bgj_cuda.h"

#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <climits>
#include <iostream>
#include <vector>

static int wrap_i8(int x)
{
    int y = x & 255;
    return y >= 128 ? y - 256 : y;
}

static void materialize_oracle(const std::vector<int8_t> &pool,
                               uint32_t pool_size,
                               uint32_t dim,
                               const std::vector<bgj_cuda_materialize_desc_t> &desc,
                               const std::vector<uint8_t> &b_dual,
                               const std::vector<float> &b_local,
                               uint32_t csd,
                               int32_t dhalf,
                               int32_t dshift,
                               std::vector<int8_t> &dst_vec,
                               std::vector<int32_t> &dst_vnorm,
                               std::vector<int32_t> &dst_vsum)
{
    (void)pool_size;
    std::vector<int8_t> tmp(dim);
    std::vector<int16_t> exact(dim);
    std::vector<int32_t> coeff(csd);
    dst_vec.assign(desc.size() * dim, 0);
    dst_vnorm.assign(desc.size(), 0);
    dst_vsum.assign(desc.size(), 0);

    for (size_t cand = 0; cand < desc.size(); cand++) {
        const bgj_cuda_materialize_desc_t d = desc[cand];
        int32_t sum128 = 0;
        for (uint32_t j = 0; j < dim; j++) {
            const int x = pool[(uint64_t)d.x * dim + j];
            const int y = pool[(uint64_t)d.y * dim + j];
            int v = 0;
            switch (d.type) {
            case BGJ_CUDA_SOL_A:
                v = x + y;
                break;
            case BGJ_CUDA_SOL_S:
                v = x - y;
                break;
            case BGJ_CUDA_SOL_AA:
                v = x + y + pool[(uint64_t)d.z * dim + j];
                break;
            case BGJ_CUDA_SOL_SA:
                v = x - y + pool[(uint64_t)d.z * dim + j];
                break;
            case BGJ_CUDA_SOL_SS:
                v = x - y - pool[(uint64_t)d.z * dim + j];
                break;
            default:
                v = 0;
                break;
            }
            exact[j] = (int16_t)v;
            tmp[j] = (int8_t)wrap_i8(v);
            sum128 += 128 * (int32_t)tmp[j];
        }

        for (uint32_t i = 0; i < csd; i++) {
            int32_t dot = 0;
            for (uint32_t j = 0; j < dim; j++) {
                dot += (int32_t)b_dual[(uint64_t)i * dim + j] * (int32_t)tmp[j];
            }
            coeff[i] = (dot + dhalf - sum128) >> dshift;
        }

        bool reject = false;
        float norm = 0.0f;
        int32_t out_sum128 = 0;
        for (uint32_t j = 0; j < dim; j++) {
            float value = 0.0f;
            for (uint32_t i = j; i < csd; i++) {
                value = std::fma(b_local[(uint64_t)i * dim + j], (float)coeff[i], value);
            }
            const int rounded = (int)lrintf(value);
            const int wrapped = wrap_i8(rounded);
            dst_vec[(uint64_t)cand * dim + j] = (int8_t)wrapped;
            out_sum128 += 128 * wrapped;
            norm += value * value;
            const int diff = (int)exact[j] - wrapped;
            if (diff < -3 || diff > 3 || wrapped == -128) reject = true;
        }
        dst_vsum[cand] = out_sum128;
        dst_vnorm[cand] = reject ? INT_MAX : (int32_t)lrintf(0.5f * norm);
    }
}

static bool run_case(uint32_t dim, uint32_t csd, uint32_t count)
{
    const uint32_t pool_size = 41;
    std::vector<int8_t> pool((size_t)pool_size * dim);
    std::vector<bgj_cuda_materialize_desc_t> desc(count);
    std::vector<uint8_t> b_dual((size_t)csd * dim);
    std::vector<float> b_local((size_t)csd * dim, 0.0f);

    for (uint32_t i = 0; i < pool_size; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            pool[(uint64_t)i * dim + j] = (int8_t)(((int)(i * 17 + j * 5 + 42) % 33) - 16);
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        desc[i].type = i % 5;
        desc[i].x = (i * 7 + 1) % pool_size;
        desc[i].y = (i * 11 + 3) % pool_size;
        desc[i].z = (i * 13 + 5) % pool_size;
    }
    for (uint32_t i = 0; i < csd; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            b_dual[(uint64_t)i * dim + j] =
                (uint8_t)((i * 37 + j * 101 + 42) & 255);
            if (i == j) {
                b_local[(uint64_t)i * dim + j] = 0.25f;
            } else if (j < i && ((i + j) % 17 == 0)) {
                b_local[(uint64_t)i * dim + j] = 0.125f;
            } else if (j > i) {
                b_local[(uint64_t)i * dim + j] =
                    (float)(((i * 11 + j * 5 + 3) % 7) - 3) * 0.0625f;
            }
        }
    }

    std::vector<int8_t> expected_vec;
    std::vector<int32_t> expected_norm;
    std::vector<int32_t> expected_sum;
    materialize_oracle(pool, pool_size, dim, desc, b_dual, b_local, csd, 128, 8,
                       expected_vec, expected_norm, expected_sum);

    std::vector<int8_t> gpu_vec((size_t)count * dim);
    std::vector<int32_t> gpu_norm(count);
    std::vector<int32_t> gpu_sum(count);
    const int ok = bgj_cuda_materialize_sol_list_raw(pool.data(),
                                                     1234,
                                                     pool_size,
                                                     dim,
                                                     desc.data(),
                                                     count,
                                                     b_dual.data(),
                                                     b_local.data(),
                                                     csd,
                                                     128,
                                                     8,
                                                     gpu_vec.data(),
                                                     gpu_norm.data(),
                                                     gpu_sum.data());
    if (!ok) {
        std::cerr << "materialize raw failed: " << bgj_cuda_last_error() << "\n";
        return false;
    }
    if (gpu_vec != expected_vec || gpu_norm != expected_norm || gpu_sum != expected_sum) {
        std::cerr << "materialize mismatch for dim=" << dim << " count=" << count << "\n";
        return false;
    }

    std::vector<uint32_t> indices(count);
    std::vector<int8_t> staged_vec((size_t)count * dim);
    std::vector<int32_t> staged_norm(count);
    std::vector<int32_t> staged_sum(count);
    for (uint32_t i = 0; i < count; i++) indices[i] = i;
    const int staged_ok = bgj_cuda_materialize_sol_list_staged_raw(pool.data(),
                                                                   1235,
                                                                   pool_size,
                                                                   dim,
                                                                   desc.data(),
                                                                   count,
                                                                   b_dual.data(),
                                                                   b_local.data(),
                                                                   csd,
                                                                   128,
                                                                   8,
                                                                   staged_norm.data(),
                                                                   staged_sum.data());
    if (!staged_ok) {
        std::cerr << "staged materialize raw failed: " << bgj_cuda_last_error() << "\n";
        return false;
    }
    const int gather_ok = bgj_cuda_materialize_copy_staged_vectors_raw(indices.data(),
                                                                       count,
                                                                       dim,
                                                                       staged_vec.data());
    bgj_cuda_materialize_finish_staged_raw();
    if (!gather_ok) {
        std::cerr << "staged materialize gather failed: " << bgj_cuda_last_error() << "\n";
        return false;
    }
    if (staged_vec != expected_vec ||
        staged_norm != expected_norm ||
        staged_sum != expected_sum) {
        std::cerr << "staged materialize mismatch for dim=" << dim << " count=" << count << "\n";
        return false;
    }
    return true;
}

int main()
{
    if (bgj_cuda_device_count() <= 0) {
        std::cerr << "SKIP: no CUDA devices: " << bgj_cuda_last_error() << "\n";
        return 0;
    }
    setenv("BGJ_CUDA_MATERIALIZE_CHUNK", "17", 1);
    if (!run_case(32, 32, 19)) return 1;
    if (!run_case(64, 64, 73)) return 1;
    if (!run_case(96, 46, 19)) return 1;
    if (!run_case(64, 49, 19)) return 1;
    std::cout << "CUDA materialize tests passed\n";
    return 0;
}
