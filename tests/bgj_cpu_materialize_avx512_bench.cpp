#include "bgj_cuda.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
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

template <typename T>
static T *aligned_alloc_items(size_t count)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, count * sizeof(T)) != 0) std::abort();
    return (T *)ptr;
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

static int hsum256_epi32(__m256i x)
{
    __m128i lo = _mm256_castsi256_si128(x);
    __m128i hi = _mm256_extracti128_si256(x, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 78));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 177));
    return _mm_cvtsi128_si32(sum);
}

static int hsum512_epi32(__m512i x)
{
    __m256i lo = _mm512_castsi512_si256(x);
    __m256i hi = _mm512_extracti64x4_epi64(x, 1);
    return hsum256_epi32(_mm256_add_epi32(lo, hi));
}

static float hsum256_ps(__m256 x)
{
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 85));
    return _mm_cvtss_f32(sum);
}

static float hsum512_ps(__m512 x)
{
    __m256 lo = _mm512_castps512_ps256(x);
    __m256 hi = _mm512_extractf32x8_ps(x, 1);
    return hsum256_ps(_mm256_add_ps(lo, hi));
}

static int dot_u8_i8_avx2(const uint8_t *u, const int8_t *s, uint32_t dim)
{
    __m256i acc = _mm256_setzero_si256();
    for (uint32_t i = 0; i < dim; i += 32) {
        acc = _mm256_dpbusd_epi32(acc,
                                  _mm256_loadu_si256((const __m256i *)(u + i)),
                                  _mm256_load_si256((const __m256i *)(s + i)));
    }
    return hsum256_epi32(acc);
}

static int sum_i8_avx2(const int8_t *s, uint32_t dim)
{
    const __m256i all_128 = _mm256_set1_epi8((char)0x80);
    __m256i acc = _mm256_setzero_si256();
    for (uint32_t i = 0; i < dim; i += 32) {
        acc = _mm256_dpbusd_epi32(acc, all_128, _mm256_load_si256((const __m256i *)(s + i)));
    }
    return hsum256_epi32(acc);
}

static int dot_u8_i8_avx512(const uint8_t *u, const int8_t *s, uint32_t dim)
{
    const __mmask64 tail32 = 0xffffffffULL;
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        acc = _mm512_dpbusd_epi32(acc,
                                  _mm512_loadu_si512((const __m512i *)(u + i)),
                                  _mm512_load_si512((const __m512i *)(s + i)));
    }
    if (i < dim) {
        acc = _mm512_dpbusd_epi32(acc,
                                  _mm512_maskz_loadu_epi8(tail32, u + i),
                                  _mm512_maskz_loadu_epi8(tail32, s + i));
    }
    return hsum512_epi32(acc);
}

static int sum_i8_avx512(const int8_t *s, uint32_t dim)
{
    const __mmask64 tail32 = 0xffffffffULL;
    const __m512i all_128 = _mm512_set1_epi8((char)0x80);
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        acc = _mm512_dpbusd_epi32(acc, all_128, _mm512_load_si512((const __m512i *)(s + i)));
    }
    if (i < dim) {
        acc = _mm512_dpbusd_epi32(acc, all_128, _mm512_maskz_loadu_epi8(tail32, s + i));
    }
    return hsum512_epi32(acc);
}

static void build_candidate(const std::vector<int8_t> &pool,
                            uint32_t dim,
                            const bgj_cuda_materialize_desc_t &d,
                            int8_t *tmp,
                            int16_t *exact)
{
    const int8_t *xv = pool.data() + (uint64_t)d.x * dim;
    const int8_t *yv = pool.data() + (uint64_t)d.y * dim;
    const int8_t *zv = pool.data() + (uint64_t)d.z * dim;
    for (uint32_t j = 0; j < dim; j++) {
        int v = 0;
        switch (d.type) {
        case BGJ_CUDA_SOL_A:
            v = xv[j] + yv[j];
            break;
        case BGJ_CUDA_SOL_S:
            v = xv[j] - yv[j];
            break;
        case BGJ_CUDA_SOL_AA:
            v = xv[j] + yv[j] + zv[j];
            break;
        case BGJ_CUDA_SOL_SA:
            v = xv[j] - yv[j] + zv[j];
            break;
        case BGJ_CUDA_SOL_SS:
            v = xv[j] - yv[j] - zv[j];
            break;
        default:
            break;
        }
        exact[j] = (int16_t)v;
        tmp[j] = (int8_t)wrap_i8(v);
    }
}

static uint64_t materialize_avx2(const std::vector<int8_t> &pool,
                                 uint32_t dim,
                                 const std::vector<bgj_cuda_materialize_desc_t> &desc,
                                 const std::vector<uint8_t> &b_dual,
                                 const std::vector<float> &b_local,
                                 std::vector<int8_t> &dst_vec,
                                 std::vector<int32_t> &dst_vnorm,
                                 std::vector<int32_t> &dst_vsum)
{
    int8_t *tmp = aligned_alloc_items<int8_t>(dim);
    int16_t *exact = aligned_alloc_items<int16_t>(dim);
    int32_t *coeff = aligned_alloc_items<int32_t>(dim);
    float *fvec = aligned_alloc_items<float>(dim);
    dst_vec.assign(desc.size() * dim, 0);
    dst_vnorm.assign(desc.size(), 0);
    dst_vsum.assign(desc.size(), 0);

    for (size_t cand = 0; cand < desc.size(); cand++) {
        build_candidate(pool, dim, desc[cand], tmp, exact);
        const int sum128 = sum_i8_avx2(tmp, dim);
        for (uint32_t i = 0; i < dim; i++) {
            const int dot = dot_u8_i8_avx2(b_dual.data() + (uint64_t)i * dim, tmp, dim);
            coeff[i] = (dot + 128 - sum128) >> 8;
        }
        for (uint32_t j = 0; j < dim; j += 8) _mm256_store_ps(fvec + j, _mm256_setzero_ps());
        for (uint32_t i = 0; i < dim; i++) {
            const __m256 q = _mm256_set1_ps((float)coeff[i]);
            const uint32_t rounded = (i + 8) & ~7u;
            for (uint32_t j = 0; j < rounded; j += 8) {
                __m256 x = _mm256_load_ps(fvec + j);
                __m256 b = _mm256_loadu_ps(b_local.data() + (uint64_t)i * dim + j);
                _mm256_store_ps(fvec + j, _mm256_fmadd_ps(q, b, x));
            }
        }

        __m256 norm = _mm256_setzero_ps();
        int local_sum = 0;
        bool reject = false;
        for (uint32_t j = 0; j < dim; j += 8) {
            __m256 fv = _mm256_load_ps(fvec + j);
            norm = _mm256_fmadd_ps(fv, fv, norm);
            __m128i packed = _mm256_cvtepi32_epi8(_mm256_cvtps_epi32(fv));
            _mm_storel_epi64((__m128i *)(dst_vec.data() + (uint64_t)cand * dim + j), packed);
            for (uint32_t k = 0; k < 8; k++) {
                const int wrapped = dst_vec[(uint64_t)cand * dim + j + k];
                const int diff = exact[j + k] - wrapped;
                local_sum += wrapped;
                if (diff < -3 || diff > 3 || wrapped == -128) reject = true;
            }
        }
        dst_vsum[cand] = 128 * local_sum;
        dst_vnorm[cand] = reject ? INT_MAX : (int32_t)lrintf(0.5f * hsum256_ps(norm));
    }

    free(tmp);
    free(exact);
    free(coeff);
    free(fvec);
    return checksum(dst_vec, dst_vnorm, dst_vsum);
}

static uint64_t materialize_avx512(const std::vector<int8_t> &pool,
                                   uint32_t dim,
                                   const std::vector<bgj_cuda_materialize_desc_t> &desc,
                                   const std::vector<uint8_t> &b_dual,
                                   const std::vector<float> &b_local,
                                   std::vector<int8_t> &dst_vec,
                                   std::vector<int32_t> &dst_vnorm,
                                   std::vector<int32_t> &dst_vsum)
{
    int8_t *tmp = aligned_alloc_items<int8_t>(dim);
    int16_t *exact = aligned_alloc_items<int16_t>(dim);
    int32_t *coeff = aligned_alloc_items<int32_t>(dim);
    float *fvec = aligned_alloc_items<float>(dim);
    dst_vec.assign(desc.size() * dim, 0);
    dst_vnorm.assign(desc.size(), 0);
    dst_vsum.assign(desc.size(), 0);

    for (size_t cand = 0; cand < desc.size(); cand++) {
        build_candidate(pool, dim, desc[cand], tmp, exact);
        const int sum128 = sum_i8_avx512(tmp, dim);
        for (uint32_t i = 0; i < dim; i++) {
            const int dot = dot_u8_i8_avx512(b_dual.data() + (uint64_t)i * dim, tmp, dim);
            coeff[i] = (dot + 128 - sum128) >> 8;
        }
        for (uint32_t j = 0; j < dim; j += 16) _mm512_store_ps(fvec + j, _mm512_setzero_ps());
        for (uint32_t i = 0; i < dim; i++) {
            const __m512 q = _mm512_set1_ps((float)coeff[i]);
            const uint32_t rounded = (i + 16) & ~15u;
            for (uint32_t j = 0; j < rounded; j += 16) {
                __m512 x = _mm512_load_ps(fvec + j);
                __m512 b = _mm512_loadu_ps(b_local.data() + (uint64_t)i * dim + j);
                _mm512_store_ps(fvec + j, _mm512_fmadd_ps(q, b, x));
            }
        }

        __m512 norm = _mm512_setzero_ps();
        int local_sum = 0;
        bool reject = false;
        for (uint32_t j = 0; j < dim; j += 16) {
            __m512 fv = _mm512_load_ps(fvec + j);
            norm = _mm512_fmadd_ps(fv, fv, norm);
            __m128i packed = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(fv));
            _mm_storeu_si128((__m128i *)(dst_vec.data() + (uint64_t)cand * dim + j), packed);
            for (uint32_t k = 0; k < 16; k++) {
                const int wrapped = dst_vec[(uint64_t)cand * dim + j + k];
                const int diff = exact[j + k] - wrapped;
                local_sum += wrapped;
                if (diff < -3 || diff > 3 || wrapped == -128) reject = true;
            }
        }
        dst_vsum[cand] = 128 * local_sum;
        dst_vnorm[cand] = reject ? INT_MAX : (int32_t)lrintf(0.5f * hsum512_ps(norm));
    }

    free(tmp);
    free(exact);
    free(coeff);
    free(fvec);
    return checksum(dst_vec, dst_vnorm, dst_vsum);
}

static double seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char **argv)
{
    uint32_t dim = argc > 1 ? parse_u32(argv[1], "dim") : 128;
    uint32_t count = argc > 2 ? parse_u32(argv[2], "count") : 8192;
    uint32_t repeats = argc > 3 ? parse_u32(argv[3], "repeats") : 3;
    uint32_t pool_size = argc > 4 ? parse_u32(argv[4], "pool_size") : 8192;

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

    std::vector<int8_t> avx2_vec;
    std::vector<int32_t> avx2_norm;
    std::vector<int32_t> avx2_sum;
    std::vector<int8_t> avx512_vec;
    std::vector<int32_t> avx512_norm;
    std::vector<int32_t> avx512_sum;

    uint64_t avx2_hash = materialize_avx2(pool, dim, desc, b_dual, b_local,
                                          avx2_vec, avx2_norm, avx2_sum);
    uint64_t avx512_hash = materialize_avx512(pool, dim, desc, b_dual, b_local,
                                              avx512_vec, avx512_norm, avx512_sum);
    if (avx2_hash != avx512_hash) {
        std::cerr << "warning: AVX2/AVX512 checksum mismatch\n";
    }

    auto start = std::chrono::steady_clock::now();
    for (uint32_t r = 0; r < repeats; r++) {
        avx2_hash = materialize_avx2(pool, dim, desc, b_dual, b_local,
                                     avx2_vec, avx2_norm, avx2_sum);
    }
    const double avx2_sec = seconds_since(start);

    start = std::chrono::steady_clock::now();
    for (uint32_t r = 0; r < repeats; r++) {
        avx512_hash = materialize_avx512(pool, dim, desc, b_dual, b_local,
                                         avx512_vec, avx512_norm, avx512_sum);
    }
    const double avx512_sec = seconds_since(start);

    const double total = (double)count * repeats;
    std::cout << "dim,count,repeats,pool_size,avx2_sec,avx512_sec,avx2_candidates_per_sec,avx512_candidates_per_sec,speedup,avx2_hash,avx512_hash\n";
    std::cout << dim << ","
              << count << ","
              << repeats << ","
              << pool_size << ","
              << avx2_sec << ","
              << avx512_sec << ","
              << total / avx2_sec << ","
              << total / avx512_sec << ","
              << avx2_sec / avx512_sec << ","
              << avx2_hash << ","
              << avx512_hash << "\n";
    return 0;
}
