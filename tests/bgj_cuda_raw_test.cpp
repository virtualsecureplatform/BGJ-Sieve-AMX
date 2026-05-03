#include "bgj_cuda.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

extern "C" int bgj_cuda_raw_device_count();
extern "C" const char *bgj_cuda_raw_last_error();
extern "C" int bgj_cuda_search_bucket_raw(const int8_t *p_vecs,
                                           const int8_t *n_vecs,
                                           const uint32_t *p_ids,
                                           const uint32_t *n_ids,
                                           const int32_t *p_norm,
                                           const int32_t *n_norm,
                                           const int32_t *p_dot,
                                           const int32_t *n_dot,
                                           uint32_t num_p,
                                           uint32_t num_n,
                                           uint32_t vec_length,
                                           int32_t goal_norm,
                                           int32_t center_norm,
                                           int record_dp,
                                           bgj_cuda_result_t *results,
                                           uint32_t result_capacity,
                                           uint32_t *result_count,
                                           int *overflow);
extern "C" int bgj_cuda_search_bucket_pool_raw(const int8_t *pool_vecs,
                                                uint64_t pool_epoch,
                                                uint32_t pool_size,
                                                const uint32_t *p_ids,
                                                const uint32_t *n_ids,
                                                const int32_t *p_norm,
                                                const int32_t *n_norm,
                                                const int32_t *p_dot,
                                                const int32_t *n_dot,
                                                uint32_t num_p,
                                                uint32_t num_n,
                                                uint32_t vec_length,
                                                int32_t goal_norm,
                                                int32_t center_norm,
                                                int record_dp,
                                                bgj_cuda_result_t *results,
                                                uint32_t result_capacity,
                                                uint32_t *result_count,
                                                int *overflow);

namespace {

struct Bucket {
    std::vector<int8_t> p_vecs;
    std::vector<int8_t> n_vecs;
    std::vector<uint32_t> p_ids;
    std::vector<uint32_t> n_ids;
    std::vector<int32_t> p_norm;
    std::vector<int32_t> n_norm;
    std::vector<int32_t> p_dot;
    std::vector<int32_t> n_dot;
};

struct Scores {
    std::vector<int32_t> two_red;
    std::vector<int32_t> three_red;
};

enum class ThresholdMode {
    Low,
    Selective,
    High,
};

int32_t row_dot(const std::vector<int8_t> &a,
                uint32_t ai,
                const std::vector<int8_t> &b,
                uint32_t bi,
                uint32_t vec_length)
{
    int32_t acc = 0;
    for (uint32_t k = 0; k < vec_length; k++) {
        acc += (int32_t)a[(size_t)ai * vec_length + k] *
               (int32_t)b[(size_t)bi * vec_length + k];
    }
    return acc;
}

void fill_side(uint32_t count,
               uint32_t vec_length,
               uint32_t id_base,
               std::vector<int8_t> &vecs,
               std::vector<uint32_t> &ids,
               std::vector<int32_t> &norms,
               std::vector<int32_t> &center_dots,
               std::mt19937 &rng)
{
    std::uniform_int_distribution<int> coeff_dist(-6, 6);
    std::uniform_int_distribution<int> center_dot_dist(-80, 80);

    vecs.resize((size_t)count * vec_length);
    ids.resize(count);
    norms.resize(count);
    center_dots.resize(count);

    for (uint32_t i = 0; i < count; i++) {
        ids[i] = id_base + i;
        int32_t norm = 0;
        for (uint32_t k = 0; k < vec_length; k++) {
            const int value = coeff_dist(rng);
            vecs[(size_t)i * vec_length + k] = (int8_t)value;
            norm += value * value;
        }
        norms[i] = norm;
        center_dots[i] = center_dot_dist(rng);
    }
}

Bucket make_bucket(uint32_t num_p, uint32_t num_n, uint32_t vec_length, uint32_t seed)
{
    std::mt19937 rng(seed);
    Bucket bucket;
    fill_side(num_p, vec_length, 100000u + seed * 1000u, bucket.p_vecs,
              bucket.p_ids, bucket.p_norm, bucket.p_dot, rng);
    fill_side(num_n, vec_length, 200000u + seed * 1000u, bucket.n_vecs,
              bucket.n_ids, bucket.n_norm, bucket.n_dot, rng);
    return bucket;
}

Bucket make_compact_bucket(uint32_t num_p, uint32_t num_n, uint32_t vec_length, uint32_t seed)
{
    std::mt19937 rng(seed);
    Bucket bucket;
    fill_side(num_p, vec_length, 0, bucket.p_vecs,
              bucket.p_ids, bucket.p_norm, bucket.p_dot, rng);
    fill_side(num_n, vec_length, num_p, bucket.n_vecs,
              bucket.n_ids, bucket.n_norm, bucket.n_dot, rng);
    return bucket;
}

void collect_scores(const Bucket &bucket, uint32_t vec_length, Scores &scores)
{
    const uint32_t num_p = (uint32_t)bucket.p_ids.size();
    const uint32_t num_n = (uint32_t)bucket.n_ids.size();

    for (uint32_t p = 0; p < num_p; p++) {
        for (uint32_t n = 0; n < num_n; n++) {
            const int32_t dp = row_dot(bucket.p_vecs, p, bucket.n_vecs, n, vec_length);
            scores.two_red.push_back(bucket.p_norm[p] + bucket.n_norm[n] + dp);
            scores.three_red.push_back(bucket.p_dot[p] + bucket.n_dot[n] - dp);
        }
    }
    for (uint32_t i = 0; i < num_p; i++) {
        for (uint32_t j = i + 1; j < num_p; j++) {
            const int32_t dp = row_dot(bucket.p_vecs, i, bucket.p_vecs, j, vec_length);
            scores.two_red.push_back(bucket.p_norm[i] + bucket.p_norm[j] - dp);
            scores.three_red.push_back(bucket.p_dot[i] + bucket.p_dot[j] + dp);
        }
    }
    for (uint32_t i = 0; i < num_n; i++) {
        for (uint32_t j = i + 1; j < num_n; j++) {
            const int32_t dp = row_dot(bucket.n_vecs, i, bucket.n_vecs, j, vec_length);
            scores.two_red.push_back(bucket.n_norm[i] + bucket.n_norm[j] - dp);
            scores.three_red.push_back(bucket.n_dot[i] + bucket.n_dot[j] + dp);
        }
    }
}

int32_t threshold_for(std::vector<int32_t> scores, ThresholdMode mode)
{
    if (scores.empty()) return 0;
    std::sort(scores.begin(), scores.end());
    if (mode == ThresholdMode::Low) return scores.front() - 1;
    if (mode == ThresholdMode::High) return scores.back() + 1;
    return scores[scores.size() / 2] + 1;
}

void push(std::vector<bgj_cuda_result_t> &out, uint32_t type, uint32_t x, uint32_t y)
{
    bgj_cuda_result_t result;
    result.type = type;
    result.x = x;
    result.y = y;
    out.push_back(result);
}

std::vector<bgj_cuda_result_t> cpu_oracle(const Bucket &bucket,
                                          uint32_t vec_length,
                                          int32_t goal_norm,
                                          int32_t center_norm,
                                          bool record_dp)
{
    const uint32_t num_p = (uint32_t)bucket.p_ids.size();
    const uint32_t num_n = (uint32_t)bucket.n_ids.size();
    const int32_t center_goal_norm = goal_norm - center_norm;
    std::vector<bgj_cuda_result_t> out;

    for (uint32_t p = 0; p < num_p; p++) {
        for (uint32_t n = 0; n < num_n; n++) {
            const int32_t dp = row_dot(bucket.p_vecs, p, bucket.n_vecs, n, vec_length);
            if (bucket.p_norm[p] + bucket.n_norm[n] + dp < goal_norm) {
                push(out, BGJ_CUDA_SOL_A, bucket.p_ids[p], bucket.n_ids[n]);
            }
            if (record_dp && bucket.p_dot[p] + bucket.n_dot[n] - dp < center_goal_norm) {
                push(out, BGJ_CUDA_SOL_SA, bucket.p_ids[p], bucket.n_ids[n]);
            }
        }
    }
    for (uint32_t i = 0; i < num_p; i++) {
        for (uint32_t j = i + 1; j < num_p; j++) {
            const int32_t dp = row_dot(bucket.p_vecs, i, bucket.p_vecs, j, vec_length);
            if (bucket.p_norm[i] + bucket.p_norm[j] - dp < goal_norm) {
                push(out, BGJ_CUDA_SOL_S, bucket.p_ids[i], bucket.p_ids[j]);
            }
            if (record_dp && bucket.p_dot[i] + bucket.p_dot[j] + dp < center_goal_norm) {
                push(out, BGJ_CUDA_SOL_SS, bucket.p_ids[i], bucket.p_ids[j]);
            }
        }
    }
    for (uint32_t i = 0; i < num_n; i++) {
        for (uint32_t j = i + 1; j < num_n; j++) {
            const int32_t dp = row_dot(bucket.n_vecs, i, bucket.n_vecs, j, vec_length);
            if (bucket.n_norm[i] + bucket.n_norm[j] - dp < goal_norm) {
                push(out, BGJ_CUDA_SOL_S, bucket.n_ids[i], bucket.n_ids[j]);
            }
            if (record_dp && bucket.n_dot[i] + bucket.n_dot[j] + dp < center_goal_norm) {
                push(out, BGJ_CUDA_SOL_AA, bucket.n_ids[i], bucket.n_ids[j]);
            }
        }
    }
    return out;
}

bool result_less(const bgj_cuda_result_t &a, const bgj_cuda_result_t &b)
{
    return std::make_tuple(a.type, a.x, a.y) < std::make_tuple(b.type, b.x, b.y);
}

bool result_equal(const bgj_cuda_result_t &a, const bgj_cuda_result_t &b)
{
    return a.type == b.type && a.x == b.x && a.y == b.y;
}

void sort_results(std::vector<bgj_cuda_result_t> &results)
{
    std::sort(results.begin(), results.end(), result_less);
}

bool compare_results(const std::string &name,
                     std::vector<bgj_cuda_result_t> gpu,
                     std::vector<bgj_cuda_result_t> oracle)
{
    sort_results(gpu);
    sort_results(oracle);
    if (gpu.size() != oracle.size()) {
        std::cerr << name << ": result count mismatch, gpu=" << gpu.size()
                  << " cpu=" << oracle.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < gpu.size(); i++) {
        if (!result_equal(gpu[i], oracle[i])) {
            std::cerr << name << ": first mismatch at " << i
                      << " gpu=(" << gpu[i].type << "," << gpu[i].x << "," << gpu[i].y << ")"
                      << " cpu=(" << oracle[i].type << "," << oracle[i].x << "," << oracle[i].y << ")\n";
            return false;
        }
    }
    return true;
}

bool run_case(const std::string &name,
              uint32_t num_p,
              uint32_t num_n,
              uint32_t vec_length,
              bool record_dp,
              uint32_t seed,
              ThresholdMode mode)
{
    Bucket bucket = make_bucket(num_p, num_n, vec_length, seed);
    Scores scores;
    collect_scores(bucket, vec_length, scores);

    const int32_t goal_norm = threshold_for(scores.two_red, mode);
    const int32_t three_red_goal = threshold_for(scores.three_red, mode);
    const int32_t center_norm = goal_norm - three_red_goal;
    const std::vector<bgj_cuda_result_t> oracle =
        cpu_oracle(bucket, vec_length, goal_norm, center_norm, record_dp);

    const uint32_t capacity = std::max<uint32_t>(16u, (uint32_t)oracle.size() + 16u);
    std::vector<bgj_cuda_result_t> gpu(capacity);
    uint32_t result_count = 0;
    int overflow = 0;
    const int ok = bgj_cuda_search_bucket_raw(bucket.p_vecs.data(), bucket.n_vecs.data(),
                                              bucket.p_ids.data(), bucket.n_ids.data(),
                                              bucket.p_norm.data(), bucket.n_norm.data(),
                                              bucket.p_dot.data(), bucket.n_dot.data(),
                                              num_p, num_n, vec_length, goal_norm, center_norm,
                                              record_dp ? 1 : 0, gpu.data(), capacity,
                                              &result_count, &overflow);
    if (!ok) {
        std::cerr << name << ": CUDA call failed: " << bgj_cuda_raw_last_error() << "\n";
        return false;
    }
    if (overflow) {
        std::cerr << name << ": unexpected overflow\n";
        return false;
    }
    gpu.resize(result_count);
    const bool matched = compare_results(name, gpu, oracle);
    if (matched) {
        std::cout << name << ": ok, results=" << result_count << "\n";
    }
    return matched;
}

bool run_overflow_case()
{
    const uint32_t num_p = 8;
    const uint32_t num_n = 7;
    const uint32_t vec_length = 16;
    Bucket bucket = make_bucket(num_p, num_n, vec_length, 99);
    Scores scores;
    collect_scores(bucket, vec_length, scores);
    const int32_t goal_norm = threshold_for(scores.two_red, ThresholdMode::High);
    const int32_t three_red_goal = threshold_for(scores.three_red, ThresholdMode::High);
    const int32_t center_norm = goal_norm - three_red_goal;

    bgj_cuda_result_t result;
    uint32_t result_count = 0;
    int overflow = 0;
    const int ok = bgj_cuda_search_bucket_raw(bucket.p_vecs.data(), bucket.n_vecs.data(),
                                              bucket.p_ids.data(), bucket.n_ids.data(),
                                              bucket.p_norm.data(), bucket.n_norm.data(),
                                              bucket.p_dot.data(), bucket.n_dot.data(),
                                              num_p, num_n, vec_length, goal_norm, center_norm,
                                              1, &result, 1, &result_count, &overflow);
    if (!ok) {
        std::cerr << "overflow: CUDA call failed: " << bgj_cuda_raw_last_error() << "\n";
        return false;
    }
    if (!overflow || result_count != 1) {
        std::cerr << "overflow: expected overflow with one retained result, got overflow="
                  << overflow << " count=" << result_count << "\n";
        return false;
    }
    std::cout << "overflow: ok\n";
    return true;
}

bool run_pool_case()
{
    const uint32_t num_p = 21;
    const uint32_t num_n = 18;
    const uint32_t vec_length = 32;
    Bucket bucket = make_compact_bucket(num_p, num_n, vec_length, 123);
    Scores scores;
    collect_scores(bucket, vec_length, scores);

    const int32_t goal_norm = threshold_for(scores.two_red, ThresholdMode::Selective);
    const int32_t three_red_goal = threshold_for(scores.three_red, ThresholdMode::Selective);
    const int32_t center_norm = goal_norm - three_red_goal;
    const std::vector<bgj_cuda_result_t> oracle =
        cpu_oracle(bucket, vec_length, goal_norm, center_norm, true);

    std::vector<int8_t> pool((size_t)(num_p + num_n) * vec_length);
    for (uint32_t i = 0; i < num_p; i++) {
        std::copy(bucket.p_vecs.begin() + (size_t)i * vec_length,
                  bucket.p_vecs.begin() + (size_t)(i + 1) * vec_length,
                  pool.begin() + (size_t)bucket.p_ids[i] * vec_length);
    }
    for (uint32_t i = 0; i < num_n; i++) {
        std::copy(bucket.n_vecs.begin() + (size_t)i * vec_length,
                  bucket.n_vecs.begin() + (size_t)(i + 1) * vec_length,
                  pool.begin() + (size_t)bucket.n_ids[i] * vec_length);
    }

    const uint32_t capacity = std::max<uint32_t>(16u, (uint32_t)oracle.size() + 16u);
    std::vector<bgj_cuda_result_t> gpu(capacity);
    uint32_t result_count = 0;
    int overflow = 0;
    const int ok = bgj_cuda_search_bucket_pool_raw(pool.data(), 1, num_p + num_n,
                                                   bucket.p_ids.data(), bucket.n_ids.data(),
                                                   bucket.p_norm.data(), bucket.n_norm.data(),
                                                   bucket.p_dot.data(), bucket.n_dot.data(),
                                                   num_p, num_n, vec_length, goal_norm, center_norm,
                                                   1, gpu.data(), capacity,
                                                   &result_count, &overflow);
    if (!ok) {
        std::cerr << "pool-cache: CUDA call failed: " << bgj_cuda_raw_last_error() << "\n";
        return false;
    }
    if (overflow) {
        std::cerr << "pool-cache: unexpected overflow\n";
        return false;
    }
    gpu.resize(result_count);
    const bool matched = compare_results("pool-cache", gpu, oracle);
    if (matched) {
        std::cout << "pool-cache: ok, results=" << result_count << "\n";
    }
    return matched;
}

}  // namespace

int main()
{
    const int devices = bgj_cuda_raw_device_count();
    if (devices <= 0) {
        std::cerr << "SKIP: no CUDA devices: " << bgj_cuda_raw_last_error() << "\n";
        return 77;
    }
    std::cout << "CUDA devices=" << devices << "\n";

    bool ok = true;
    ok = run_case("empty", 0, 0, 32, false, 1, ThresholdMode::Selective) && ok;
    ok = run_case("no-candidates", 4, 3, 16, true, 2, ThresholdMode::Low) && ok;
    ok = run_case("p-only-all", 5, 0, 32, true, 3, ThresholdMode::High) && ok;
    ok = run_case("n-only-all", 0, 6, 32, true, 4, ThresholdMode::High) && ok;
    ok = run_case("np-odd-len", 4, 5, 7, true, 5, ThresholdMode::Selective) && ok;
    ok = run_case("np-no-dp-len1", 9, 7, 1, false, 6, ThresholdMode::Selective) && ok;
    ok = run_case("mixed-32", 17, 19, 32, true, 7, ThresholdMode::Selective) && ok;
    ok = run_case("mixed-96", 6, 5, 96, true, 8, ThresholdMode::Selective) && ok;
    ok = run_pool_case() && ok;
    ok = run_overflow_case() && ok;

    if (!ok) return 1;
    std::cout << "all CUDA raw bucket tests passed\n";
    return 0;
}
