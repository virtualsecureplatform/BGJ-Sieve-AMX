#include "bgj_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int32_t row_dot(const std::vector<int8_t> &pool, uint32_t a, uint32_t b, uint32_t dim)
{
    int32_t acc = 0;
    for (uint32_t i = 0; i < dim; i++) {
        acc += (int32_t)pool[(uint64_t)a * dim + i] *
               (int32_t)pool[(uint64_t)b * dim + i];
    }
    return acc;
}

bool entry_less(const bgj_cuda_bucket_entry_t &a, const bgj_cuda_bucket_entry_t &b)
{
    if (a.bucket != b.bucket) return a.bucket < b.bucket;
    if (a.id != b.id) return a.id < b.id;
    return a.dot < b.dot;
}

bool same_entries(std::vector<bgj_cuda_bucket_entry_t> a,
                  std::vector<bgj_cuda_bucket_entry_t> b)
{
    std::sort(a.begin(), a.end(), entry_less);
    std::sort(b.begin(), b.end(), entry_less);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].bucket != b[i].bucket ||
            a[i].id != b[i].id ||
            a[i].dot != b[i].dot) {
            return false;
        }
    }
    return true;
}

bool same_ordered_entries(const std::vector<bgj_cuda_bucket_entry_t> &a,
                          const std::vector<bgj_cuda_bucket_entry_t> &b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].bucket != b[i].bucket ||
            a[i].id != b[i].id ||
            a[i].dot != b[i].dot) {
            return false;
        }
    }
    return true;
}

bool run_case(uint32_t dim, uint32_t pool_size)
{
    const uint32_t alpha_x2_u16 = (uint32_t)std::llround(65536.0 * 2.0 * 0.35);
    const uint32_t center_ids[] = {
        0, 5, 9, 13, 19, 24, 31, 42,
        47, 53, 61, 70, 79, 83, 89, 96
    };
    const uint32_t num_centers = sizeof(center_ids) / sizeof(center_ids[0]);

    std::vector<int8_t> pool((uint64_t)pool_size * dim);
    std::vector<int32_t> vnorm(pool_size);
    std::vector<int32_t> vsum(pool_size);
    for (uint32_t i = 0; i < pool_size; i++) {
        int32_t norm = 0;
        int32_t sum = 0;
        for (uint32_t j = 0; j < dim; j++) {
            const int value = ((int)(i * 17 + j * 11 + 42) % 17) - 8;
            pool[(uint64_t)i * dim + j] = (int8_t)value;
            norm += value * value;
            sum += 128 * value;
        }
        vnorm[i] = norm / 2;
        vsum[i] = sum;
    }

    std::vector<bgj_cuda_bucket_entry_t> expected;
    for (uint32_t b = 0; b < num_centers; b++) {
        for (uint32_t id = 0; id < pool_size; id++) {
            const int32_t dot = row_dot(pool, center_ids[b], id, dim);
            const int32_t bound = (int32_t)(((int64_t)vnorm[id] * alpha_x2_u16) >> 16);
            if ((dot < 0 ? -dot : dot) <= bound) continue;
            bgj_cuda_bucket_entry_t entry;
            entry.bucket = b;
            entry.id = id;
            entry.dot = dot;
            expected.push_back(entry);
        }
    }
    std::vector<bgj_cuda_bucket_entry_t> expected_ordered;
    for (uint32_t b = 0; b < num_centers; b++) {
        for (int sign = 0; sign < 2; sign++) {
            for (uint32_t id = 0; id < pool_size; id++) {
                const int32_t dot = row_dot(pool, center_ids[b], id, dim);
                if ((sign == 0 && dot <= 0) || (sign == 1 && dot > 0)) continue;
                const int32_t bound = (int32_t)(((int64_t)vnorm[id] * alpha_x2_u16) >> 16);
                if ((dot < 0 ? -dot : dot) <= bound) continue;
                bgj_cuda_bucket_entry_t entry;
                entry.bucket = b;
                entry.id = id;
                entry.dot = dot;
                expected_ordered.push_back(entry);
            }
        }
    }

    std::vector<bgj_cuda_bucket_entry_t> got(expected.size() + 16);
    uint32_t count = 0;
    int overflow = 0;
    setenv("BGJ_CUDA_BUCKET_DETERMINISTIC", "1", 1);
    const int ok = bgj_cuda_bucket_bgj1_raw(pool.data(),
                                            99,
                                            pool_size,
                                            center_ids,
                                            num_centers,
                                            vnorm.data(),
                                            dim,
                                            alpha_x2_u16,
                                            got.data(),
                                            (uint32_t)got.size(),
                                            &count,
                                            &overflow);
    unsetenv("BGJ_CUDA_BUCKET_DETERMINISTIC");
    if (!ok || overflow) {
        std::cerr << "CUDA bucket raw failed: " << bgj_cuda_last_error()
                  << " overflow=" << overflow << "\n";
        return false;
    }
    got.resize(count);
    if (!same_entries(expected, got)) {
        std::cerr << "CUDA bucket mismatch for dim=" << dim
                  << " pool_size=" << pool_size
                  << " expected=" << expected.size()
                  << " got=" << got.size() << "\n";
        return false;
    }
    if (!same_ordered_entries(expected_ordered, got)) {
        std::cerr << "CUDA deterministic bucket order mismatch for dim=" << dim
                  << " pool_size=" << pool_size << "\n";
        return false;
    }

    unsetenv("BGJ_CUDA_BUCKET_DETERMINISTIC");
    unsetenv("BGJ_CUDA_BUCKET_TENSOR");
    unsetenv("BGJ_CUDA_BUCKET_BLOCK_APPEND");
    got.assign(expected.size() + 16, bgj_cuda_bucket_entry_t());
    count = 0;
    overflow = 0;
    const int default_ok = bgj_cuda_bucket_bgj1_raw(pool.data(),
                                                   100,
                                                   pool_size,
                                                   center_ids,
                                                   num_centers,
                                                   vnorm.data(),
                                                   dim,
                                                   alpha_x2_u16,
                                                   got.data(),
                                                   (uint32_t)got.size(),
                                                   &count,
                                                   &overflow);
    if (!default_ok || overflow) {
        std::cerr << "CUDA default bucket raw failed: " << bgj_cuda_last_error()
                  << " overflow=" << overflow << "\n";
        return false;
    }
    got.resize(count);
    if (!same_entries(expected, got)) {
        std::cerr << "CUDA default bucket mismatch for dim=" << dim
                  << " pool_size=" << pool_size
                  << " expected=" << expected.size()
                  << " got=" << got.size() << "\n";
        return false;
    }

    setenv("BGJ_CUDA_BUCKET_DETERMINISTIC", "0", 1);
    setenv("BGJ_CUDA_BUCKET_TENSOR", "0", 1);
    setenv("BGJ_CUDA_BUCKET_BLOCK_APPEND", "1", 1);
    got.assign(expected.size() + 16, bgj_cuda_bucket_entry_t());
    count = 0;
    overflow = 0;
    const int legacy_block_append_ok = bgj_cuda_bucket_bgj1_raw(pool.data(),
                                                                101,
                                                                pool_size,
                                                                center_ids,
                                                                num_centers,
                                                                vnorm.data(),
                                                                dim,
                                                                alpha_x2_u16,
                                                                got.data(),
                                                                (uint32_t)got.size(),
                                                                &count,
                                                                &overflow);
    unsetenv("BGJ_CUDA_BUCKET_DETERMINISTIC");
    unsetenv("BGJ_CUDA_BUCKET_BLOCK_APPEND");
    unsetenv("BGJ_CUDA_BUCKET_TENSOR");
    if (!legacy_block_append_ok || overflow) {
        std::cerr << "CUDA legacy block-append bucket raw failed: " << bgj_cuda_last_error()
                  << " overflow=" << overflow << "\n";
        return false;
    }
    got.resize(count);
    if (!same_entries(expected, got)) {
        std::cerr << "CUDA legacy block-append bucket mismatch for dim=" << dim
                  << " pool_size=" << pool_size
                  << " expected=" << expected.size()
                  << " got=" << got.size() << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (bgj_cuda_device_count() <= 0) {
        std::cerr << "SKIP: no CUDA devices: " << bgj_cuda_last_error() << "\n";
        return 0;
    }
    if (!run_case(32, 97)) return 1;
    if (!run_case(64, 131)) return 1;
    std::cout << "CUDA bucket tests passed\n";
    return 0;
}
