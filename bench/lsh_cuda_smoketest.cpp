#include "../include/bgj_cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static int result_less(const bgj_cuda_result_t &a, const bgj_cuda_result_t &b)
{
    if (a.type != b.type) return a.type < b.type;
    if (a.x != b.x) return a.x < b.x;
    return a.y < b.y;
}

static int result_equal(const bgj_cuda_result_t &a, const bgj_cuda_result_t &b)
{
    return a.type == b.type && a.x == b.x && a.y == b.y;
}

static uint64_t load_u64_le(const uint8_t *ptr)
{
    uint64_t value = 0;
    for (uint32_t i = 0; i < 8; i++) value |= (uint64_t)ptr[i] << (8u * i);
    return value;
}

int main()
{
    const uint32_t mbound = 257;
    const uint32_t shsize = 64;
    const int32_t threshold = 230;
    std::vector<uint8_t> sh((size_t)mbound * shsize);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < sh.size(); i += 8) {
        uint64_t word = rng();
        for (uint32_t j = 0; j < 8; j++) sh[i + j] = (uint8_t)(word >> (8u * j));
    }

    std::vector<bgj_cuda_result_t> cpu;
    for (uint32_t i = 0; i < mbound; i++) {
        for (uint32_t j = i + 1; j < mbound; j++) {
            uint32_t dist = 0;
            for (uint32_t w = 0; w < 8; w++) {
                const uint64_t a = load_u64_le(sh.data() + ((uint64_t)i * shsize) + w * 8u);
                const uint64_t b = load_u64_le(sh.data() + ((uint64_t)j * shsize) + w * 8u);
                dist += __builtin_popcountll(a ^ b);
            }
            if ((int32_t)dist <= threshold) cpu.push_back({BGJ_CUDA_SOL_S, i, j});
            else if ((int32_t)dist >= 512 - threshold) cpu.push_back({BGJ_CUDA_SOL_A, i, j});
        }
    }

    std::vector<bgj_cuda_result_t> gpu(cpu.size() + 1024);
    uint32_t gpu_count = 0;
    int overflow = 0;
    int ok = bgj_cuda_lsh_search_raw(sh.data(), mbound, shsize, threshold,
                                     gpu.data(), (uint32_t)gpu.size(), &gpu_count, &overflow);
    if (!ok || overflow) {
        std::fprintf(stderr, "CUDA LSH call failed: ok=%d overflow=%d error=%s\n",
                     ok, overflow, bgj_cuda_last_error());
        return 1;
    }
    gpu.resize(gpu_count);
    std::sort(cpu.begin(), cpu.end(), result_less);
    std::sort(gpu.begin(), gpu.end(), result_less);
    if (cpu.size() != gpu.size() || !std::equal(cpu.begin(), cpu.end(), gpu.begin(), result_equal)) {
        std::fprintf(stderr, "mismatch: cpu=%zu gpu=%zu\n", cpu.size(), gpu.size());
        return 2;
    }
    std::printf("lsh_cuda_smoketest ok: results=%zu\n", cpu.size());
    return 0;
}
