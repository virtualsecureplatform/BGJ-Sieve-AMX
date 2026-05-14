#include "bgj_cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
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
                                           uint32_t center_id,
                                           int32_t center_norm,
                                           int record_dp,
                                           int transform_dp,
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
                                                uint32_t center_id,
                                                int32_t center_norm,
                                                int record_dp,
                                                int transform_dp,
                                                bgj_cuda_result_t *results,
                                                uint32_t result_capacity,
                                                uint32_t *result_count,
                                                int *overflow);
extern "C" int bgj_cuda_search_bucket_pool_batch_raw(const int8_t *pool_vecs,
                                                      uint64_t pool_epoch,
                                                      uint32_t pool_size,
                                                      const uint32_t *const *p_ids,
                                                      const uint32_t *const *n_ids,
                                                      const int32_t *const *p_norm,
                                                      const int32_t *const *n_norm,
                                                      const int32_t *const *p_dot,
                                                      const int32_t *const *n_dot,
                                                      const uint32_t *num_p,
                                                      const uint32_t *num_n,
                                                      uint32_t batch_size,
                                                      uint32_t vec_length,
                                                      const int32_t *goal_norm,
                                                      const uint32_t *center_id,
                                                      const int32_t *center_norm,
                                                      int record_dp,
                                                      int transform_dp,
                                                      bgj_cuda_result_t *const *results,
                                                      const uint32_t *result_capacity,
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
    std::vector<int8_t> pool_vecs;
};

uint32_t parse_u32(const char *value, const char *name)
{
    char *end = NULL;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "invalid " << name << ": " << value << "\n";
        std::exit(2);
    }
    return (uint32_t)parsed;
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
    fill_side(num_p, vec_length, 0, bucket.p_vecs, bucket.p_ids,
              bucket.p_norm, bucket.p_dot, rng);
    fill_side(num_n, vec_length, num_p, bucket.n_vecs, bucket.n_ids,
              bucket.n_norm, bucket.n_dot, rng);

    bucket.pool_vecs.resize((size_t)(num_p + num_n) * vec_length);
    if (num_p) {
        std::copy(bucket.p_vecs.begin(), bucket.p_vecs.end(), bucket.pool_vecs.begin());
    }
    if (num_n) {
        std::copy(bucket.n_vecs.begin(), bucket.n_vecs.end(),
                  bucket.pool_vecs.begin() + (size_t)num_p * vec_length);
    }
    return bucket;
}

void append_bucket_to_pool(Bucket &bucket, uint32_t vec_length, std::vector<int8_t> &pool)
{
    const uint32_t base = (uint32_t)(pool.size() / vec_length);
    const uint32_t num_p = (uint32_t)bucket.p_ids.size();
    const uint32_t num_n = (uint32_t)bucket.n_ids.size();
    pool.resize(pool.size() + (size_t)(num_p + num_n) * vec_length);

    for (uint32_t i = 0; i < num_p; i++) {
        const uint32_t id = base + i;
        std::copy(bucket.p_vecs.begin() + (size_t)i * vec_length,
                  bucket.p_vecs.begin() + (size_t)(i + 1) * vec_length,
                  pool.begin() + (size_t)id * vec_length);
        bucket.p_ids[i] = id;
    }
    for (uint32_t i = 0; i < num_n; i++) {
        const uint32_t id = base + num_p + i;
        std::copy(bucket.n_vecs.begin() + (size_t)i * vec_length,
                  bucket.n_vecs.begin() + (size_t)(i + 1) * vec_length,
                  pool.begin() + (size_t)id * vec_length);
        bucket.n_ids[i] = id;
    }
}

uint64_t pair_dots(uint32_t num_p, uint32_t num_n)
{
    const uint64_t p = num_p;
    const uint64_t n = num_n;
    return p * n + p * (p - 1u) / 2u + n * (n - 1u) / 2u;
}

uint32_t default_repeats(uint32_t num_p, uint32_t num_n)
{
    const uint64_t pairs = std::max<uint64_t>(pair_dots(num_p, num_n), 1u);
    uint64_t repeats = 200000000ULL / pairs;
    if (repeats < 5u) repeats = 5u;
    if (repeats > 250u) repeats = 250u;
    return (uint32_t)repeats;
}

bool run_once(const Bucket &bucket,
              uint32_t num_p,
              uint32_t num_n,
              uint32_t vec_length,
              bool use_pool,
              int record_dp,
              bgj_cuda_result_t *results,
              uint32_t result_capacity,
              uint32_t *result_count,
              int *overflow)
{
    const int32_t goal_norm = std::numeric_limits<int32_t>::min() / 4;
    const int32_t center_norm = 0;
    if (use_pool) {
        return bgj_cuda_search_bucket_pool_raw(bucket.pool_vecs.data(),
                                               1,
                                               num_p + num_n,
                                               bucket.p_ids.data(),
                                               bucket.n_ids.data(),
                                               bucket.p_norm.data(),
                                               bucket.n_norm.data(),
                                               bucket.p_dot.data(),
                                               bucket.n_dot.data(),
                                               num_p,
                                               num_n,
                                               vec_length,
                                               goal_norm,
                                               0,
                                               center_norm,
                                               record_dp,
                                               0,
                                               results,
                                               result_capacity,
                                               result_count,
                                               overflow) != 0;
    }
    return bgj_cuda_search_bucket_raw(bucket.p_vecs.data(),
                                      bucket.n_vecs.data(),
                                      bucket.p_ids.data(),
                                      bucket.n_ids.data(),
                                      bucket.p_norm.data(),
                                      bucket.n_norm.data(),
                                      bucket.p_dot.data(),
                                      bucket.n_dot.data(),
                                      num_p,
                                      num_n,
                                      vec_length,
                                      goal_norm,
                                      0,
                                      center_norm,
                                      record_dp,
                                      0,
                                      results,
                                      result_capacity,
                                      result_count,
                                      overflow) != 0;
}

bool bench_case(uint32_t vec_length,
                uint32_t num_p,
                uint32_t num_n,
                uint32_t repeats,
                bool use_pool,
                int record_dp,
                uint32_t threads)
{
    Bucket bucket = make_bucket(num_p, num_n, vec_length,
                                0xC001D00Du ^ vec_length ^ (num_p << 8) ^ (num_n << 16));
    if (threads == 0) threads = 1;

    bool warmup_ok = true;
    for (uint32_t i = 0; i < 3 && warmup_ok; i++) {
        std::vector<std::thread> workers;
        std::vector<int> ok(threads, 1);
        for (uint32_t t = 0; t < threads; t++) {
            workers.push_back(std::thread([&, t]() {
                std::vector<bgj_cuda_result_t> results(16);
                uint32_t result_count = 0;
                int overflow = 0;
                if (!run_once(bucket, num_p, num_n, vec_length, use_pool, record_dp,
                              results.data(), (uint32_t)results.size(), &result_count, &overflow)) {
                    ok[t] = 0;
                    return;
                }
                if (overflow || result_count != 0) {
                    ok[t] = 0;
                }
            }));
        }
        for (uint32_t t = 0; t < workers.size(); t++) workers[t].join();
        for (uint32_t t = 0; t < ok.size(); t++) warmup_ok = warmup_ok && ok[t] != 0;
    }
    if (!warmup_ok) {
        std::cerr << "CUDA warmup failed: " << bgj_cuda_raw_last_error() << "\n";
        return false;
    }

    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    std::vector<int> ok(threads, 1);
    for (uint32_t t = 0; t < threads; t++) {
        workers.push_back(std::thread([&, t]() {
            std::vector<bgj_cuda_result_t> results(16);
            for (uint32_t i = 0; i < repeats; i++) {
                uint32_t result_count = 0;
                int overflow = 0;
                if (!run_once(bucket, num_p, num_n, vec_length, use_pool, record_dp,
                              results.data(), (uint32_t)results.size(), &result_count, &overflow)) {
                    ok[t] = 0;
                    return;
                }
                if (overflow || result_count != 0) {
                    ok[t] = 0;
                    return;
                }
            }
        }));
    }
    for (uint32_t t = 0; t < workers.size(); t++) workers[t].join();
    const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    for (uint32_t t = 0; t < ok.size(); t++) {
        if (!ok[t]) {
            std::cerr << "CUDA benchmark failed: " << bgj_cuda_raw_last_error() << "\n";
            return false;
        }
    }

    const double seconds = std::chrono::duration<double>(stop - start).count();
    const double ms_per_call = seconds * 1000.0 / ((double)repeats * (double)threads);
    const double dots = (double)pair_dots(num_p, num_n) * (double)repeats * (double)threads;
    const double gdot_per_second = dots / seconds / 1.0e9;
    const double gint8_mac_per_second = dots * (double)vec_length / seconds / 1.0e9;

    std::cout << (use_pool ? "pool" : "raw") << ','
              << vec_length << ','
              << num_p << ','
              << num_n << ','
              << repeats << ','
              << record_dp << ','
              << threads << ','
              << std::fixed << std::setprecision(4) << ms_per_call << ','
              << std::setprecision(6) << gdot_per_second << ','
              << std::setprecision(3) << gint8_mac_per_second << "\n";
    return true;
}

bool bench_batch_case(uint32_t vec_length,
                      uint32_t num_p,
                      uint32_t num_n,
                      uint32_t repeats,
                      int record_dp,
                      uint32_t threads,
                      uint32_t batch_size)
{
    if (threads == 0) threads = 1;
    if (batch_size == 0) batch_size = 1;

    std::vector<Bucket> buckets;
    std::vector<int8_t> pool;
    buckets.reserve(batch_size);
    for (uint32_t i = 0; i < batch_size; i++) {
        Bucket bucket = make_bucket(num_p, num_n, vec_length,
                                    0xBADC0DEu ^ vec_length ^ (num_p << 8) ^
                                    (num_n << 16) ^ (i * 7919u));
        append_bucket_to_pool(bucket, vec_length, pool);
        buckets.push_back(bucket);
    }

    std::vector<const uint32_t *> p_ids(batch_size);
    std::vector<const uint32_t *> n_ids(batch_size);
    std::vector<const int32_t *> p_norm(batch_size);
    std::vector<const int32_t *> n_norm(batch_size);
    std::vector<const int32_t *> p_dot(batch_size);
    std::vector<const int32_t *> n_dot(batch_size);
    std::vector<uint32_t> p_counts(batch_size, num_p);
    std::vector<uint32_t> n_counts(batch_size, num_n);
    std::vector<int32_t> goal_norm(batch_size, std::numeric_limits<int32_t>::min() / 4);
    std::vector<uint32_t> center_id(batch_size, 0);
    std::vector<int32_t> center_norm(batch_size, 0);
    std::vector<uint32_t> result_capacity(batch_size, 16);

    for (uint32_t i = 0; i < batch_size; i++) {
        p_ids[i] = buckets[i].p_ids.data();
        n_ids[i] = buckets[i].n_ids.data();
        p_norm[i] = buckets[i].p_norm.data();
        n_norm[i] = buckets[i].n_norm.data();
        p_dot[i] = buckets[i].p_dot.data();
        n_dot[i] = buckets[i].n_dot.data();
    }

    auto run_worker = [&](uint32_t calls, int &ok) {
        std::vector<std::vector<bgj_cuda_result_t> > result_storage(batch_size);
        std::vector<bgj_cuda_result_t *> result_ptrs(batch_size);
        std::vector<uint32_t> result_count(batch_size);
        std::vector<int> overflow(batch_size);
        for (uint32_t i = 0; i < batch_size; i++) {
            result_storage[i].resize(result_capacity[i]);
            result_ptrs[i] = result_storage[i].data();
        }

        for (uint32_t call = 0; call < calls; call++) {
            if (!bgj_cuda_search_bucket_pool_batch_raw(pool.data(),
                                                       1,
                                                       (uint32_t)(pool.size() / vec_length),
                                                       p_ids.data(),
                                                       n_ids.data(),
                                                       p_norm.data(),
                                                       n_norm.data(),
                                                       p_dot.data(),
                                                       n_dot.data(),
                                                       p_counts.data(),
                                                       n_counts.data(),
                                                       batch_size,
                                                       vec_length,
                                                       goal_norm.data(),
                                                       center_id.data(),
                                                       center_norm.data(),
                                                       record_dp,
                                                       0,
                                                       result_ptrs.data(),
                                                       result_capacity.data(),
                                                       result_count.data(),
                                                       overflow.data())) {
                ok = 0;
                return;
            }
            for (uint32_t i = 0; i < batch_size; i++) {
                if (overflow[i] || result_count[i] != 0) {
                    ok = 0;
                    return;
                }
            }
        }
    };

    bool warmup_ok = true;
    for (uint32_t i = 0; i < 3 && warmup_ok; i++) {
        std::vector<std::thread> workers;
        std::vector<int> ok(threads, 1);
        for (uint32_t t = 0; t < threads; t++) {
            workers.push_back(std::thread([&, t]() {
                run_worker(1, ok[t]);
            }));
        }
        for (uint32_t t = 0; t < workers.size(); t++) workers[t].join();
        for (uint32_t t = 0; t < ok.size(); t++) warmup_ok = warmup_ok && ok[t] != 0;
    }
    if (!warmup_ok) {
        std::cerr << "CUDA batch warmup failed: " << bgj_cuda_raw_last_error() << "\n";
        return false;
    }

    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    std::vector<int> ok(threads, 1);
    for (uint32_t t = 0; t < threads; t++) {
        workers.push_back(std::thread([&, t]() {
            run_worker(repeats, ok[t]);
        }));
    }
    for (uint32_t t = 0; t < workers.size(); t++) workers[t].join();
    const std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    for (uint32_t t = 0; t < ok.size(); t++) {
        if (!ok[t]) {
            std::cerr << "CUDA batch benchmark failed: " << bgj_cuda_raw_last_error() << "\n";
            return false;
        }
    }

    const double seconds = std::chrono::duration<double>(stop - start).count();
    const double bucket_calls = (double)repeats * (double)threads * (double)batch_size;
    const double ms_per_call = seconds * 1000.0 / bucket_calls;
    const double dots = (double)pair_dots(num_p, num_n) * bucket_calls;
    const double gdot_per_second = dots / seconds / 1.0e9;
    const double gint8_mac_per_second = dots * (double)vec_length / seconds / 1.0e9;

    std::cout << "pool-batch" << batch_size << ','
              << vec_length << ','
              << num_p << ','
              << num_n << ','
              << repeats << ','
              << record_dp << ','
              << threads << ','
              << std::fixed << std::setprecision(4) << ms_per_call << ','
              << std::setprecision(6) << gdot_per_second << ','
              << std::setprecision(3) << gint8_mac_per_second << "\n";
    return true;
}

void print_header()
{
    std::cout << "mode,vec_length,num_p,num_n,repeats,record_dp,threads,ms_per_call,gdot_per_s,gint8_mac_per_s\n";
}

bool run_sweep()
{
    const uint32_t dims[] = {96u, 160u, 224u};
    const uint32_t sizes[] = {64u, 128u, 256u, 512u, 1024u, 2048u};

    print_header();
    for (uint32_t dim_i = 0; dim_i < sizeof(dims) / sizeof(dims[0]); dim_i++) {
        for (uint32_t size_i = 0; size_i < sizeof(sizes) / sizeof(sizes[0]); size_i++) {
            const uint32_t repeats = default_repeats(sizes[size_i], sizes[size_i]);
            if (!bench_case(dims[dim_i], sizes[size_i], sizes[size_i],
                            repeats, false, 1, 1)) {
                return false;
            }
        }
    }
    return true;
}

void usage(const char *argv0)
{
    std::cerr << "usage: " << argv0 << " [vec_length num_p num_n [repeats [pool [record_dp [threads [batch]]]]]]\n"
              << "       no arguments runs a bounded default sweep\n";
}

}  // namespace

int main(int argc, char **argv)
{
    const int devices = bgj_cuda_raw_device_count();
    if (devices <= 0) {
        std::cerr << "SKIP: no CUDA devices: " << bgj_cuda_raw_last_error() << "\n";
        return 77;
    }

    if (argc == 1) {
        return run_sweep() ? 0 : 1;
    }
    if (argc < 4 || argc > 9) {
        usage(argv[0]);
        return 2;
    }

    const uint32_t vec_length = parse_u32(argv[1], "vec_length");
    const uint32_t num_p = parse_u32(argv[2], "num_p");
    const uint32_t num_n = parse_u32(argv[3], "num_n");
    const uint32_t repeats = argc >= 5 ? parse_u32(argv[4], "repeats") :
                                         default_repeats(num_p, num_n);
    const bool use_pool = argc >= 6 ? parse_u32(argv[5], "pool") != 0 : false;
    const int record_dp = argc >= 7 ? (parse_u32(argv[6], "record_dp") != 0 ? 1 : 0) : 1;
    const uint32_t threads = argc >= 8 ? parse_u32(argv[7], "threads") : 1;
    const uint32_t batch = argc >= 9 ? parse_u32(argv[8], "batch") : 1;

    print_header();
    if (batch > 1) {
        if (!use_pool) {
            std::cerr << "batch benchmark currently uses the pool-cache raw API; pass pool=1\n";
            return 2;
        }
        return bench_batch_case(vec_length, num_p, num_n, repeats, record_dp, threads, batch) ? 0 : 1;
    }
    return bench_case(vec_length, num_p, num_n, repeats, use_pool, record_dp, threads) ? 0 : 1;
}
