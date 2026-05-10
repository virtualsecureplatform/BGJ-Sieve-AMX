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
    if (cpu.size() > 1) {
        std::vector<bgj_cuda_result_t> tiny(1);
        uint32_t tiny_count = 0;
        int tiny_overflow = 0;
        ok = bgj_cuda_lsh_search_raw(sh.data(), mbound, shsize, threshold,
                                     tiny.data(), (uint32_t)tiny.size(), &tiny_count, &tiny_overflow);
        if (!ok || !tiny_overflow) {
            std::fprintf(stderr, "expected overflow: ok=%d overflow=%d count=%u error=%s\n",
                         ok, tiny_overflow, tiny_count, bgj_cuda_last_error());
            return 3;
        }
    }

    std::vector<bgj_cuda_result_t> gpu_chunked;
    const uint64_t total_slots = bgj_cuda_lsh_total_tile_slots(mbound);
    const uint64_t chunk_slots = 17;
    for (uint64_t begin = 0; begin < total_slots; begin += chunk_slots) {
        const uint64_t count = std::min(chunk_slots, total_slots - begin);
        std::vector<bgj_cuda_result_t> chunk(cpu.size() + 1024);
        uint32_t chunk_count = 0;
        int chunk_overflow = 0;
        ok = bgj_cuda_lsh_search_range_raw(sh.data(), mbound, shsize, threshold,
                                           begin, count, chunk.data(), (uint32_t)chunk.size(),
                                           &chunk_count, &chunk_overflow);
        if (!ok || chunk_overflow) {
            std::fprintf(stderr, "CUDA LSH range call failed: ok=%d overflow=%d begin=%llu count=%llu error=%s\n",
                         ok, chunk_overflow, (unsigned long long)begin,
                         (unsigned long long)count, bgj_cuda_last_error());
            return 4;
        }
        chunk.resize(chunk_count);
        gpu_chunked.insert(gpu_chunked.end(), chunk.begin(), chunk.end());
    }
    std::sort(gpu_chunked.begin(), gpu_chunked.end(), result_less);
    if (cpu.size() != gpu_chunked.size() || !std::equal(cpu.begin(), cpu.end(), gpu_chunked.begin(), result_equal)) {
        std::fprintf(stderr, "range mismatch: cpu=%zu gpu=%zu\n", cpu.size(), gpu_chunked.size());
        return 5;
    }

    std::printf("lsh_cuda_smoketest ok: results=%zu tile_slots=%llu\n",
                cpu.size(), (unsigned long long)total_slots);
    return 0;
}
