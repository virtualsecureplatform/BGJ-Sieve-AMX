// Oracle test for bgj_cuda_search_bucket_pool_batch_np_flat_raw.
// Covers the tensor tile path (BGJ_CUDA_BATCH_TENSOR=1, default) and the
// scalar path (BGJ_CUDA_BATCH_TENSOR=0) against a host reference over random
// pools, bucket shapes, vec lengths 96/128/160, include_same_pairs 0/1, and a
// deliberate result-capacity overflow case.

#include "bgj_cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>

extern "C" int bgj_cuda_raw_device_count();
extern "C" const char *bgj_cuda_raw_last_error();

typedef std::tuple<uint32_t, uint32_t, uint32_t> sol_tuple_t;

static int32_t dot_i8(const int8_t *a, const int8_t *b, uint32_t vec_length)
{
    int32_t acc = 0;
    for (uint32_t i = 0; i < vec_length; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

struct flat_case_t {
    std::vector<uint32_t> p_ids;
    std::vector<uint32_t> n_ids;
    std::vector<int32_t> p_norm;
    std::vector<int32_t> n_norm;
    std::vector<uint32_t> p_offsets;
    std::vector<uint32_t> n_offsets;
    std::vector<uint32_t> num_p;
    std::vector<uint32_t> num_n;
    uint32_t batch_size;
};

int main()
{
    if (bgj_cuda_raw_device_count() <= 0) {
        printf("no CUDA device: %s\n", bgj_cuda_raw_last_error());
        return 0;
    }

    std::mt19937 rng(20260717u);
    uint64_t pool_epoch = 1;
    int failures = 0;

    const uint32_t vec_lengths[3] = {96u, 128u, 160u};
    for (int vl_case = 0; vl_case < 3; vl_case++) {
        const uint32_t vec_length = vec_lengths[vl_case];
        const uint32_t pool_size = 4096;
        std::vector<int8_t> pool((size_t)pool_size * vec_length);
        std::uniform_int_distribution<int> vdist(-4, 4);
        for (size_t i = 0; i < pool.size(); i++) pool[i] = (int8_t)vdist(rng);
        std::vector<int32_t> pool_norm(pool_size);
        for (uint32_t i = 0; i < pool_size; i++) {
            pool_norm[i] = dot_i8(pool.data() + (size_t)i * vec_length,
                                  pool.data() + (size_t)i * vec_length,
                                  vec_length);
        }
        pool_epoch++;

        for (int same = 0; same <= 1; same++) {
            for (int shape = 0; shape < 3; shape++) {
                flat_case_t c;
                if (shape == 0) {
                    const uint32_t sizes_p[6] = {0, 1, 63, 64, 65, 200};
                    const uint32_t sizes_n[6] = {17, 0, 64, 1, 127, 513};
                    c.batch_size = 6;
                    for (uint32_t i = 0; i < c.batch_size; i++) {
                        c.num_p.push_back(sizes_p[i]);
                        c.num_n.push_back(sizes_n[i]);
                    }
                } else if (shape == 1) {
                    c.batch_size = 48;
                    std::uniform_int_distribution<uint32_t> sdist(0, 300);
                    for (uint32_t i = 0; i < c.batch_size; i++) {
                        c.num_p.push_back(sdist(rng));
                        c.num_n.push_back(sdist(rng));
                    }
                } else {
                    c.batch_size = 2;
                    c.num_p.push_back(1000);
                    c.num_p.push_back(3);
                    c.num_n.push_back(999);
                    c.num_n.push_back(2);
                }
                std::uniform_int_distribution<uint32_t> iddist(0, pool_size - 1);
                for (uint32_t i = 0; i < c.batch_size; i++) {
                    c.p_offsets.push_back((uint32_t)c.p_ids.size());
                    c.n_offsets.push_back((uint32_t)c.n_ids.size());
                    for (uint32_t k = 0; k < c.num_p[i]; k++) {
                        const uint32_t id = iddist(rng);
                        c.p_ids.push_back(id);
                        c.p_norm.push_back(pool_norm[id]);
                    }
                    for (uint32_t k = 0; k < c.num_n[i]; k++) {
                        const uint32_t id = iddist(rng);
                        c.n_ids.push_back(id);
                        c.n_norm.push_back(pool_norm[id]);
                    }
                }

                // Oracle pass 1: gather every candidate sum to place goal_norm at
                // a quantile giving a few hundred accepts.
                std::vector<int32_t> sums;
                for (uint32_t i = 0; i < c.batch_size; i++) {
                    const uint32_t *pi = c.p_ids.data() + c.p_offsets[i];
                    const uint32_t *ni = c.n_ids.data() + c.n_offsets[i];
                    const int32_t *pn = c.p_norm.data() + c.p_offsets[i];
                    const int32_t *nn = c.n_norm.data() + c.n_offsets[i];
                    for (uint32_t a = 0; a < c.num_p[i]; a++) {
                        for (uint32_t b = 0; b < c.num_n[i]; b++) {
                            sums.push_back(pn[a] + nn[b] +
                                           dot_i8(pool.data() + (size_t)pi[a] * vec_length,
                                                  pool.data() + (size_t)ni[b] * vec_length,
                                                  vec_length));
                        }
                    }
                    if (same) {
                        for (uint32_t a = 0; a < c.num_p[i] && c.num_p[i] > 1; a++) {
                            for (uint32_t b = a + 1; b < c.num_p[i]; b++) {
                                sums.push_back(pn[a] + pn[b] -
                                               dot_i8(pool.data() + (size_t)pi[a] * vec_length,
                                                      pool.data() + (size_t)pi[b] * vec_length,
                                                      vec_length));
                            }
                        }
                        for (uint32_t a = 0; a < c.num_n[i] && c.num_n[i] > 1; a++) {
                            for (uint32_t b = a + 1; b < c.num_n[i]; b++) {
                                sums.push_back(nn[a] + nn[b] -
                                               dot_i8(pool.data() + (size_t)ni[a] * vec_length,
                                                      pool.data() + (size_t)ni[b] * vec_length,
                                                      vec_length));
                            }
                        }
                    }
                }
                if (sums.empty()) continue;
                std::vector<int32_t> sorted_sums = sums;
                std::sort(sorted_sums.begin(), sorted_sums.end());
                const size_t target = sorted_sums.size() < 800 ? sorted_sums.size() / 2 :
                                      400;
                const int32_t goal_norm = sorted_sums[target] + 1;

                // Oracle pass 2: expected multisets per bucket.
                std::vector<std::vector<sol_tuple_t> > expected(c.batch_size);
                for (uint32_t i = 0; i < c.batch_size; i++) {
                    const uint32_t *pi = c.p_ids.data() + c.p_offsets[i];
                    const uint32_t *ni = c.n_ids.data() + c.n_offsets[i];
                    const int32_t *pn = c.p_norm.data() + c.p_offsets[i];
                    const int32_t *nn = c.n_norm.data() + c.n_offsets[i];
                    for (uint32_t a = 0; a < c.num_p[i]; a++) {
                        for (uint32_t b = 0; b < c.num_n[i]; b++) {
                            const int32_t dp = dot_i8(pool.data() + (size_t)pi[a] * vec_length,
                                                      pool.data() + (size_t)ni[b] * vec_length,
                                                      vec_length);
                            if (pn[a] + nn[b] + dp < goal_norm) {
                                expected[i].push_back(sol_tuple_t(BGJ_CUDA_SOL_A, pi[a], ni[b]));
                            }
                        }
                    }
                    if (same && c.num_p[i] > 1) {
                        for (uint32_t a = 0; a < c.num_p[i]; a++) {
                            for (uint32_t b = a + 1; b < c.num_p[i]; b++) {
                                const int32_t dp = dot_i8(pool.data() + (size_t)pi[a] * vec_length,
                                                          pool.data() + (size_t)pi[b] * vec_length,
                                                          vec_length);
                                if (pn[a] + pn[b] - dp < goal_norm) {
                                    expected[i].push_back(sol_tuple_t(BGJ_CUDA_SOL_S, pi[a], pi[b]));
                                }
                            }
                        }
                    }
                    if (same && c.num_n[i] > 1) {
                        for (uint32_t a = 0; a < c.num_n[i]; a++) {
                            for (uint32_t b = a + 1; b < c.num_n[i]; b++) {
                                const int32_t dp = dot_i8(pool.data() + (size_t)ni[a] * vec_length,
                                                          pool.data() + (size_t)ni[b] * vec_length,
                                                          vec_length);
                                if (nn[a] + nn[b] - dp < goal_norm) {
                                    expected[i].push_back(sol_tuple_t(BGJ_CUDA_SOL_S, ni[a], ni[b]));
                                }
                            }
                        }
                    }
                    std::sort(expected[i].begin(), expected[i].end());
                }

                const uint32_t result_capacity = 65536;
                std::vector<bgj_cuda_result_t> storage((size_t)c.batch_size * result_capacity);
                std::vector<bgj_cuda_result_t *> result_ptrs(c.batch_size);
                std::vector<uint32_t> result_caps(c.batch_size, result_capacity);
                std::vector<uint32_t> result_counts(c.batch_size, 0);
                std::vector<int> overflows(c.batch_size, 0);
                for (uint32_t i = 0; i < c.batch_size; i++) {
                    result_ptrs[i] = storage.data() + (size_t)i * result_capacity;
                }

                const int ok = bgj_cuda_search_bucket_pool_batch_np_flat_raw(
                    pool.data(),
                    pool_epoch,
                    pool_size,
                    c.p_ids.empty() ? NULL : c.p_ids.data(),
                    c.n_ids.empty() ? NULL : c.n_ids.data(),
                    c.p_norm.empty() ? NULL : c.p_norm.data(),
                    c.n_norm.empty() ? NULL : c.n_norm.data(),
                    c.p_offsets.data(),
                    c.n_offsets.data(),
                    c.num_p.data(),
                    c.num_n.data(),
                    c.batch_size,
                    vec_length,
                    goal_norm,
                    same,
                    result_ptrs.data(),
                    result_caps.data(),
                    result_counts.data(),
                    overflows.data());
                if (!ok) {
                    printf("FAIL vl=%u same=%d shape=%d: call failed: %s\n",
                           vec_length, same, shape, bgj_cuda_raw_last_error());
                    failures++;
                    continue;
                }

                for (uint32_t i = 0; i < c.batch_size; i++) {
                    if (overflows[i]) {
                        printf("FAIL vl=%u same=%d shape=%d bucket=%u: unexpected overflow\n",
                               vec_length, same, shape, i);
                        failures++;
                        continue;
                    }
                    std::vector<sol_tuple_t> got;
                    for (uint32_t k = 0; k < result_counts[i]; k++) {
                        got.push_back(sol_tuple_t(result_ptrs[i][k].type,
                                                  result_ptrs[i][k].x,
                                                  result_ptrs[i][k].y));
                    }
                    std::sort(got.begin(), got.end());
                    if (got != expected[i]) {
                        printf("FAIL vl=%u same=%d shape=%d bucket=%u: got %zu expected %zu\n",
                               vec_length, same, shape, i, got.size(), expected[i].size());
                        failures++;
                    }
                }

                // Overflow behavior: rerun bucket 0 alone with a tiny capacity.
                if (!expected[0].empty()) {
                    const uint32_t small_cap = expected[0].size() > 2 ?
                                               (uint32_t)(expected[0].size() - 1) : 1u;
                    std::vector<bgj_cuda_result_t> small_storage(small_cap);
                    bgj_cuda_result_t *small_ptr = small_storage.data();
                    uint32_t small_count = 0;
                    int small_overflow = 0;
                    const int ok2 = bgj_cuda_search_bucket_pool_batch_np_flat_raw(
                        pool.data(),
                        pool_epoch,
                        pool_size,
                        c.p_ids.empty() ? NULL : c.p_ids.data(),
                        c.n_ids.empty() ? NULL : c.n_ids.data(),
                        c.p_norm.empty() ? NULL : c.p_norm.data(),
                        c.n_norm.empty() ? NULL : c.n_norm.data(),
                        c.p_offsets.data(),
                        c.n_offsets.data(),
                        c.num_p.data(),
                        c.num_n.data(),
                        1,
                        vec_length,
                        goal_norm,
                        same,
                        &small_ptr,
                        &small_cap,
                        &small_count,
                        &small_overflow);
                    if (!ok2 || !small_overflow || small_count != small_cap) {
                        printf("FAIL vl=%u same=%d shape=%d: overflow case ok=%d overflow=%d count=%u cap=%u\n",
                               vec_length, same, shape, ok2, small_overflow, small_count, small_cap);
                        failures++;
                    }
                }
            }
        }
    }

    if (failures) {
        printf("np_flat test FAILED: %d failures\n", failures);
        return 1;
    }
    printf("np_flat test passed\n");
    return 0;
}
