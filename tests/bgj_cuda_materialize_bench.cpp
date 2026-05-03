#include "bgj_cuda.h"

#include <stdint.h>
#include <stdlib.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

static int wrap_i8(int x)
{
    int y = x & 255;
    return y >= 128 ? y - 256 : y;
}

static uint32_t parse_u32(const char *value, const char *name)
{
    char *end = NULL;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (!end || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "invalid " << name << ": " << value << "\n";
        std::exit(2);
    }
    return (uint32_t)parsed;
}

static void fill_case(uint32_t dim,
                      uint32_t count,
                      uint32_t pool_size,
                      std::vector<int8_t> &pool,
                      std::vector<bgj_cuda_materialize_desc_t> &desc,
                      std::vector<uint8_t> &b_dual,
                      std::vector<float> &b_local)
{
    pool.resize((size_t)pool_size * dim);
    desc.resize(count);
    b_dual.resize((size_t)dim * dim);
    b_local.assign((size_t)dim * dim, 0.0f);

    for (uint32_t i = 0; i < pool_size; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            pool[(uint64_t)i * dim + j] =
                (int8_t)(((int)(i * 17 + j * 5 + 42) % 33) - 16);
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        desc[i].type = i % 5;
        desc[i].x = (i * 7 + 1) % pool_size;
        desc[i].y = (i * 11 + 3) % pool_size;
        desc[i].z = (i * 13 + 5) % pool_size;
    }
    for (uint32_t i = 0; i < dim; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            b_dual[(uint64_t)i * dim + j] =
                (uint8_t)(128 + ((int)((i * 3 + j * 7 + 42) % 5) - 2));
            if (j <= i) {
                b_local[(uint64_t)i * dim + j] =
                    (i == j) ? 0.25f : (((i + j) % 17 == 0) ? 0.125f : 0.0f);
            }
        }
    }
}

static uint64_t checksum(const std::vector<int8_t> &vec,
                         const std::vector<int32_t> &norm,
                         const std::vector<int32_t> &sum)
{
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < vec.size(); i++) {
        h ^= (uint8_t)vec[i];
        h *= 1099511628211ULL;
    }
    for (size_t i = 0; i < norm.size(); i++) {
        h ^= (uint32_t)norm[i];
        h *= 1099511628211ULL;
        h ^= (uint32_t)sum[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t materialize_cpu(const std::vector<int8_t> &pool,
                                uint32_t dim,
                                const std::vector<bgj_cuda_materialize_desc_t> &desc,
                                const std::vector<uint8_t> &b_dual,
                                const std::vector<float> &b_local,
                                std::vector<int8_t> &dst_vec,
                                std::vector<int32_t> &dst_vnorm,
                                std::vector<int32_t> &dst_vsum)
{
    const uint32_t count = (uint32_t)desc.size();
    std::vector<int8_t> tmp(dim);
    std::vector<int16_t> exact(dim);
    std::vector<int32_t> coeff(dim);
    dst_vec.assign((size_t)count * dim, 0);
    dst_vnorm.assign(count, 0);
    dst_vsum.assign(count, 0);

    for (uint32_t cand = 0; cand < count; cand++) {
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
                break;
            }
            exact[j] = (int16_t)v;
            tmp[j] = (int8_t)wrap_i8(v);
            sum128 += 128 * (int32_t)tmp[j];
        }

        for (uint32_t i = 0; i < dim; i++) {
            int32_t dot = 0;
            for (uint32_t j = 0; j < dim; j++) {
                dot += (int32_t)b_dual[(uint64_t)i * dim + j] * (int32_t)tmp[j];
            }
            coeff[i] = (dot + 128 - sum128) >> 8;
        }

        bool reject = false;
        float norm = 0.0f;
        int32_t out_sum128 = 0;
        for (uint32_t j = 0; j < dim; j++) {
            float value = 0.0f;
            for (uint32_t i = j; i < dim; i++) {
                value += (float)coeff[i] * b_local[(uint64_t)i * dim + j];
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
    return checksum(dst_vec, dst_vnorm, dst_vsum);
}

static double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char **argv)
{
    uint32_t dim = argc > 1 ? parse_u32(argv[1], "dim") : 128;
    uint32_t count = argc > 2 ? parse_u32(argv[2], "count") : 16384;
    uint32_t repeats = argc > 3 ? parse_u32(argv[3], "repeats") : 5;
    uint32_t pool_size = argc > 4 ? parse_u32(argv[4], "pool_size") : 8192;
    int run_cpu = argc > 5 ? (parse_u32(argv[5], "run_cpu") != 0) : 1;

    if (dim == 0 || dim > 256 || (dim % 32) != 0) {
        std::cerr << "dim must be a positive multiple of 32 and <= 256\n";
        return 2;
    }
    if (count == 0 || repeats == 0 || pool_size < 3) {
        std::cerr << "count, repeats, and pool_size are invalid\n";
        return 2;
    }

    std::vector<int8_t> pool;
    std::vector<bgj_cuda_materialize_desc_t> desc;
    std::vector<uint8_t> b_dual;
    std::vector<float> b_local;
    fill_case(dim, count, pool_size, pool, desc, b_dual, b_local);

    std::vector<int8_t> cpu_vec;
    std::vector<int32_t> cpu_norm;
    std::vector<int32_t> cpu_sum;
    uint64_t cpu_hash = 0;
    double cpu_sec = 0.0;
    if (run_cpu) {
        auto start = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < repeats; r++) {
            cpu_hash = materialize_cpu(pool, dim, desc, b_dual, b_local,
                                       cpu_vec, cpu_norm, cpu_sum);
        }
        cpu_sec = seconds_since(start);
    }

    if (bgj_cuda_device_count() <= 0) {
        std::cerr << "SKIP: no CUDA devices: " << bgj_cuda_last_error() << "\n";
        return 0;
    }

    std::vector<int8_t> gpu_vec((size_t)count * dim);
    std::vector<int32_t> gpu_norm(count);
    std::vector<int32_t> gpu_sum(count);
    for (uint32_t r = 0; r < 2; r++) {
        if (!bgj_cuda_materialize_sol_list_raw(pool.data(), 42, pool_size, dim,
                                               desc.data(), count, b_dual.data(),
                                               b_local.data(), dim, 128, 8,
                                               gpu_vec.data(), gpu_norm.data(),
                                               gpu_sum.data())) {
            std::cerr << "warmup failed: " << bgj_cuda_last_error() << "\n";
            return 1;
        }
    }

    auto start = std::chrono::steady_clock::now();
    for (uint32_t r = 0; r < repeats; r++) {
        if (!bgj_cuda_materialize_sol_list_raw(pool.data(), 42, pool_size, dim,
                                               desc.data(), count, b_dual.data(),
                                               b_local.data(), dim, 128, 8,
                                               gpu_vec.data(), gpu_norm.data(),
                                               gpu_sum.data())) {
            std::cerr << "benchmark failed: " << bgj_cuda_last_error() << "\n";
            return 1;
        }
    }
    const double gpu_sec = seconds_since(start);
    const uint64_t gpu_hash = checksum(gpu_vec, gpu_norm, gpu_sum);

    std::cout << "dim,count,repeats,pool_size,cpu_sec,gpu_sec,cpu_candidates_per_sec,gpu_candidates_per_sec,speedup,cpu_hash,gpu_hash\n";
    const double total = (double)count * repeats;
    const double cpu_cps = run_cpu && cpu_sec > 0.0 ? total / cpu_sec : 0.0;
    const double gpu_cps = gpu_sec > 0.0 ? total / gpu_sec : 0.0;
    const double speedup = run_cpu && gpu_sec > 0.0 ? cpu_sec / gpu_sec : 0.0;
    std::cout << dim << ","
              << count << ","
              << repeats << ","
              << pool_size << ","
              << cpu_sec << ","
              << gpu_sec << ","
              << cpu_cps << ","
              << gpu_cps << ","
              << speedup << ","
              << cpu_hash << ","
              << gpu_hash << "\n";
    if (run_cpu && cpu_hash != gpu_hash) {
        std::cerr << "warning: CPU/GPU checksum mismatch\n";
    }
    return 0;
}
