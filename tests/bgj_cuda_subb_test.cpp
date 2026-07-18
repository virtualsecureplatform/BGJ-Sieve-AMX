// Oracle test for bgj_cuda_sub_bucketing_raw: replays the acceptance rules of
// Pool_epi8_t::_sub_bucketing on the host (bound_x(v) = (norm_v*alpha_x2)>>16,
// p passes on dp > bound, n on dp < -bound; entries2/entries3 nesting in dual
// mode; 2-reduction sol candidates against center_b2) and checks the device
// output matches as a multiset; entry order within the streams is
// unspecified.

#include "bgj_cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" int bgj_cuda_raw_device_count();
extern "C" const char *bgj_cuda_raw_last_error();

static int32_t dot_i8(const int8_t *a, const int8_t *b, uint32_t vec_length)
{
    int32_t acc = 0;
    for (uint32_t i = 0; i < vec_length; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

static uint64_t entry_key(const bgj_cuda_subb_entry_t &e)
{
    return ((uint64_t)e.meta << 32) | e.index;
}

static int run_bench()
{
    std::mt19937 rng(7u);
    const uint32_t vec_length = 128;
    const uint32_t pool_size = 400000;
    std::vector<int8_t> pool((size_t)pool_size * vec_length);
    std::uniform_int_distribution<int> vdist(-5, 5);
    for (size_t i = 0; i < pool.size(); i++) pool[i] = (int8_t)vdist(rng);
    std::vector<int32_t> pool_norm(pool_size);
    for (uint32_t i = 0; i < pool_size; i++) {
        pool_norm[i] = dot_i8(pool.data() + (size_t)i * vec_length,
                              pool.data() + (size_t)i * vec_length, vec_length) / 2;
    }
    const uint32_t num_p = 160000, num_n = 160000, num_centers = 72;
    std::uniform_int_distribution<uint32_t> iddist(0, pool_size - 1);
    std::vector<uint32_t> p_ids(num_p), n_ids(num_n), center_ids(num_centers);
    std::vector<int32_t> p_norm(num_p), n_norm(num_n), center_b2(num_centers);
    for (uint32_t i = 0; i < num_p; i++) { p_ids[i] = iddist(rng); p_norm[i] = pool_norm[p_ids[i]]; }
    for (uint32_t i = 0; i < num_n; i++) { n_ids[i] = iddist(rng); n_norm[i] = pool_norm[n_ids[i]]; }
    for (uint32_t i = 0; i < num_centers; i++) { center_ids[i] = iddist(rng); center_b2[i] = pool_norm[center_ids[i]]; }
    const uint32_t a3 = (uint32_t)(uint16_t)(65536.0 * 2.0 * 0.36 + 0.5);
    const uint32_t a2 = (uint32_t)(uint16_t)(65536.0 * 2.0 * 0.24 + 0.5);
    const uint32_t cap3 = 4u << 20, cap2 = 24u << 20, sol_cap = 1u << 20;
    std::vector<bgj_cuda_subb_entry_t> e3(cap3), e2(cap2), es(sol_cap);
    for (int rep = 0; rep < 12; rep++) {
        uint32_t c3 = 0, c2 = 0, cs = 0; int ov = 0;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const int ok = bgj_cuda_sub_bucketing_raw(
            pool.data(), 424242, pool_size,
            p_ids.data(), p_norm.data(), num_p,
            n_ids.data(), n_norm.data(), num_n,
            center_ids.data(), center_b2.data(), num_centers,
            vec_length, a3, a2,
            e3.data(), cap3, &c3, e2.data(), cap2, &c2,
            es.data(), sol_cap, &cs, &ov);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        printf("rep %d: ok=%d ov=%d c3=%u c2=%u cs=%u  %.3f ms\n", rep, ok, ov, c3, c2, cs, ms);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && argv[1][0] == 'b') {
        if (bgj_cuda_raw_device_count() <= 0) return 0;
        return run_bench();
    }
    if (bgj_cuda_raw_device_count() <= 0) {
        printf("no CUDA device: %s\n", bgj_cuda_raw_last_error());
        return 0;
    }

    std::mt19937 rng(20260718u);
    uint64_t pool_epoch = 100;
    int failures = 0;

    const uint32_t vec_lengths[3] = {96u, 128u, 160u};
    for (int vl_case = 0; vl_case < 3; vl_case++) {
        const uint32_t vec_length = vec_lengths[vl_case];
        const uint32_t pool_size = 4096;
        std::vector<int8_t> pool((size_t)pool_size * vec_length);
        std::uniform_int_distribution<int> vdist(-5, 5);
        for (size_t i = 0; i < pool.size(); i++) pool[i] = (int8_t)vdist(rng);
        std::vector<int32_t> pool_norm(pool_size);
        for (uint32_t i = 0; i < pool_size; i++) {
            pool_norm[i] = dot_i8(pool.data() + (size_t)i * vec_length,
                                  pool.data() + (size_t)i * vec_length,
                                  vec_length) / 2;
        }
        pool_epoch++;

        const uint32_t shapes[4][3] = {
            {3000, 2900, 48}, {500, 0, 72}, {0, 700, 96}, {65, 63, 48}};
        for (int shape = 0; shape < 4; shape++) {
            for (int dual = 0; dual <= 1; dual++) {
                const uint32_t num_p = shapes[shape][0];
                const uint32_t num_n = shapes[shape][1];
                const uint32_t num_centers = shapes[shape][2];
                if (num_p == 0 && num_n == 0) continue;

                std::uniform_int_distribution<uint32_t> iddist(0, pool_size - 1);
                std::vector<uint32_t> p_ids(num_p), n_ids(num_n), center_ids(num_centers);
                std::vector<int32_t> p_norm(num_p), n_norm(num_n), center_b2(num_centers);
                for (uint32_t i = 0; i < num_p; i++) {
                    p_ids[i] = iddist(rng);
                    p_norm[i] = pool_norm[p_ids[i]];
                }
                for (uint32_t i = 0; i < num_n; i++) {
                    n_ids[i] = iddist(rng);
                    n_norm[i] = pool_norm[n_ids[i]];
                }
                int64_t norm_acc = 0;
                for (uint32_t i = 0; i < num_centers; i++) {
                    center_ids[i] = iddist(rng);
                    norm_acc += pool_norm[center_ids[i]];
                }
                const int32_t goal_norm = (int32_t)(2 * norm_acc / num_centers);
                for (uint32_t i = 0; i < num_centers; i++) {
                    center_b2[i] = goal_norm - pool_norm[center_ids[i]];
                }

                const double alpha3 = 0.36;
                const double alpha2 = dual ? 0.24 : 0.0;
                const uint32_t a3 = (uint32_t)(uint16_t)(65536.0 * 2.0 * alpha3 + 0.5);
                const uint32_t a2 = dual ? (uint32_t)(uint16_t)(65536.0 * 2.0 * alpha2 + 0.5) : 0;

                // Host oracle.
                std::vector<bgj_cuda_subb_entry_t> exp3, exp2, exps;
                for (int side = 1; side >= 0; side--) {
                    const uint32_t num = side ? num_p : num_n;
                    const uint32_t *ids = side ? p_ids.data() : n_ids.data();
                    const int32_t *norms = side ? p_norm.data() : n_norm.data();
                    for (uint32_t i = 0; i < num; i++) {
                        const int8_t *v = pool.data() + (size_t)ids[i] * vec_length;
                        const int32_t b3 = (int32_t)(((int64_t)norms[i] * (int64_t)a3) >> 16);
                        const int32_t b2b = dual ?
                            (int32_t)(((int64_t)norms[i] * (int64_t)a2) >> 16) : 0;
                        for (uint32_t c = 0; c < num_centers; c++) {
                            const int32_t dp = dot_i8(v,
                                                      pool.data() + (size_t)center_ids[c] * vec_length,
                                                      vec_length);
                            const int pass3 = side ? (dp > b3) : (dp < -b3);
                            const int pass2 = dual ? (side ? (dp > b2b) : (dp < -b2b)) : pass3;
                            if (!pass2) continue;
                            const uint32_t meta = c | (side ? (1u << 16) : 0u);
                            bgj_cuda_subb_entry_t e;
                            e.index = i;
                            e.meta = meta;
                            if (dual) {
                                exp2.push_back(e);
                                if (pass3) exp3.push_back(e);
                            } else {
                                exp3.push_back(e);
                            }
                            if (pass3) {
                                const int32_t reduced = side ? norms[i] - dp : norms[i] + dp;
                                if (reduced < center_b2[c]) exps.push_back(e);
                            }
                        }
                    }
                }
                auto by_key = [](const bgj_cuda_subb_entry_t &x, const bgj_cuda_subb_entry_t &y) {
                    return entry_key(x) < entry_key(y);
                };
                std::sort(exp3.begin(), exp3.end(), by_key);
                std::sort(exp2.begin(), exp2.end(), by_key);
                std::sort(exps.begin(), exps.end(), by_key);

                const uint32_t cap3 = (uint32_t)exp3.size() + 4096;
                const uint32_t cap2 = (uint32_t)exp2.size() + 4096;
                const uint32_t sol_cap = (uint32_t)exps.size() + 4096;
                std::vector<bgj_cuda_subb_entry_t> got3(cap3), got2(cap2), gots(sol_cap);
                uint32_t count3 = 0, count2 = 0, sol_count = 0;
                int overflow = 0;
                const int ok = bgj_cuda_sub_bucketing_raw(
                    pool.data(), pool_epoch, pool_size,
                    num_p ? p_ids.data() : NULL, num_p ? p_norm.data() : NULL, num_p,
                    num_n ? n_ids.data() : NULL, num_n ? n_norm.data() : NULL, num_n,
                    center_ids.data(), center_b2.data(), num_centers,
                    vec_length, a3, a2,
                    got3.data(), cap3, &count3,
                    dual ? got2.data() : NULL, dual ? cap2 : 0, dual ? &count2 : NULL,
                    gots.data(), sol_cap, &sol_count,
                    &overflow);
                if (!ok || overflow) {
                    printf("FAIL vl=%u shape=%d dual=%d: ok=%d overflow=%d (%s)\n",
                           vec_length, shape, dual, ok, overflow, bgj_cuda_raw_last_error());
                    failures++;
                    continue;
                }
                got3.resize(count3);
                got2.resize(count2);
                gots.resize(sol_count);
                std::sort(got3.begin(), got3.end(), by_key);
                std::sort(got2.begin(), got2.end(), by_key);
                std::sort(gots.begin(), gots.end(), by_key);
                int bad = 0;
                if (got3.size() != exp3.size() ||
                    !std::equal(got3.begin(), got3.end(), exp3.begin(),
                                [](const bgj_cuda_subb_entry_t &x, const bgj_cuda_subb_entry_t &y) {
                                    return entry_key(x) == entry_key(y);
                                })) bad |= 1;
                if (dual && (got2.size() != exp2.size() ||
                    !std::equal(got2.begin(), got2.end(), exp2.begin(),
                                [](const bgj_cuda_subb_entry_t &x, const bgj_cuda_subb_entry_t &y) {
                                    return entry_key(x) == entry_key(y);
                                }))) bad |= 2;
                if (gots.size() != exps.size() ||
                    !std::equal(gots.begin(), gots.end(), exps.begin(),
                                [](const bgj_cuda_subb_entry_t &x, const bgj_cuda_subb_entry_t &y) {
                                    return entry_key(x) == entry_key(y);
                                })) bad |= 4;
                if (bad) {
                    printf("FAIL vl=%u shape=%d dual=%d: mismatch mask=%d got3=%zu exp3=%zu got2=%zu exp2=%zu gots=%zu exps=%zu\n",
                           vec_length, shape, dual, bad, got3.size(), exp3.size(),
                           got2.size(), exp2.size(), gots.size(), exps.size());
                    failures++;
                }

                // Overflow behavior on a tiny entries3 cap.
                if (exp3.size() > 4) {
                    const uint32_t small_cap = (uint32_t)exp3.size() - 1;
                    std::vector<bgj_cuda_subb_entry_t> small3(small_cap);
                    uint32_t c3 = 0, c2 = 0, cs = 0;
                    int ov = 0;
                    const int ok2 = bgj_cuda_sub_bucketing_raw(
                        pool.data(), pool_epoch, pool_size,
                        num_p ? p_ids.data() : NULL, num_p ? p_norm.data() : NULL, num_p,
                        num_n ? n_ids.data() : NULL, num_n ? n_norm.data() : NULL, num_n,
                        center_ids.data(), center_b2.data(), num_centers,
                        vec_length, a3, a2,
                        small3.data(), small_cap, &c3,
                        dual ? got2.data() : NULL, dual ? cap2 : 0, dual ? &c2 : NULL,
                        gots.data(), sol_cap, &cs,
                        &ov);
                    if (!ok2 || !ov) {
                        printf("FAIL vl=%u shape=%d dual=%d: overflow case ok=%d ov=%d\n",
                               vec_length, shape, dual, ok2, ov);
                        failures++;
                    }
                }
            }
        }
    }

    if (failures) {
        printf("sub-bucketing test FAILED: %d failures\n", failures);
        return 1;
    }
    printf("sub-bucketing test passed\n");
    return 0;
}
// Invoke with "bench" to time the raw call on a large synthetic bucket.
