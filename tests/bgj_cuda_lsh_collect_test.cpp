// Oracle test for bgj_cuda_lsh_collect_raw: replays the coefficient
// reconstruction, float expansion, and +/-center Babai membership test on the
// host with the same FMA order (std::fmaf, rintf) and compares the per-center
// outputs as multisets of rows. Row order within a center is unspecified.

#include "bgj_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

extern "C" int bgj_cuda_raw_device_count();
extern "C" const char *bgj_cuda_raw_last_error();
extern "C" void *bgj_cuda_alloc_pinned_host_raw(size_t bytes);

static int run_bench()
{
    std::mt19937 rng(9u);
    const uint32_t vec_length = 128, num_vec = 4000000, csd = 120;
    const uint32_t fd = 152, fd8 = 152, dh_dim = 24, dh_stride = 64;
    const uint32_t ld = fd - csd, id_off = ld - dh_dim + 16;
    const uint32_t num_c = 40;
    std::vector<int8_t> pool((size_t)num_vec * vec_length);
    for (size_t i = 0; i < pool.size(); i += 4) *(uint32_t *)&pool[i] = rng();
    std::vector<int32_t> vsum(num_vec, 0);
    std::vector<uint8_t> b_dual((size_t)csd * vec_length);
    for (size_t i = 0; i < b_dual.size(); i++) b_dual[i] = (uint8_t)(rng() & 0xff);
    std::vector<float> b_ext((size_t)csd * fd8, 0.01f);
    std::vector<float> b_head((size_t)dh_dim * dh_stride, 0.0f);
    std::vector<float> idiag(dh_dim);
    for (uint32_t i = 0; i < dh_dim; i++) { b_head[(size_t)i*dh_stride+i] = 40.0f; idiag[i] = 1.0f/40.0f; }
    std::vector<float> centers((size_t)num_c * dh_dim, 1.0f);
    const uint32_t cap = 40000;
    float *out = (float *)bgj_cuda_alloc_pinned_host_raw((size_t)num_c * cap * fd8 * sizeof(float));
    if (!out) return 1;
    std::vector<float *> ptrs(num_c);
    for (uint32_t c = 0; c < num_c; c++) ptrs[c] = out + (size_t)c * cap * fd8;
    std::vector<uint32_t> mbound(num_c);
    for (int rep = 0; rep < 6; rep++) {
        float r2 = 1000.0f, rout = 0;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const int ok = bgj_cuda_lsh_collect_raw(pool.data(), 999, num_vec, vsum.data(), num_vec,
            vec_length, csd, b_dual.data(), 1<<14, 15, b_ext.data(), fd, fd8,
            b_head.data(), idiag.data(), dh_dim, dh_stride, id_off,
            centers.data(), num_c, r2, 16384, (double)cap*0.8/num_vec*num_c, &rout,
            cap, ptrs.data(), mbound.data());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        uint64_t total = 0; for (uint32_t c = 0; c < num_c; c++) total += mbound[c];
        printf("rep %d: ok=%d radius=%.3f total_mbound=%llu  %.1f ms\n", rep, ok, rout,
               (unsigned long long)total, (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6);
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
    int failures = 0;

    const uint32_t vec_lengths[2] = {96u, 128u};
    for (int vcase = 0; vcase < 2; vcase++) {
        const uint32_t vec_length = vec_lengths[vcase];
        const uint32_t num_vec = 30000;
        const uint32_t csd = vec_length - 8;
        const uint32_t fd = csd + 40;
        const uint32_t fd8 = (fd + 7u) / 8u * 8u;
        const uint32_t ld = fd - csd;
        const uint32_t dh_dim = 24;
        const uint32_t dh_stride = 64;
        const uint32_t id_off = ld - dh_dim;
        const uint32_t num_c = 40;
        const int32_t dhalf = 1 << 14;
        const int32_t dshift = 15;

        std::vector<int8_t> pool((size_t)num_vec * vec_length);
        std::uniform_int_distribution<int> vdist(-40, 40);
        for (size_t i = 0; i < pool.size(); i++) pool[i] = (int8_t)vdist(rng);
        std::vector<int32_t> vsum(num_vec);
        std::uniform_int_distribution<int> sdist(-2000, 2000);
        for (uint32_t i = 0; i < num_vec; i++) vsum[i] = sdist(rng);

        std::vector<uint8_t> b_dual((size_t)csd * vec_length);
        for (size_t i = 0; i < b_dual.size(); i++) b_dual[i] = (uint8_t)(rng() & 0xff);

        std::vector<float> b_ext((size_t)csd * fd8, 0.0f);
        std::uniform_real_distribution<float> fdist(-0.4f, 0.4f);
        for (uint32_t i = 0; i < csd; i++) {
            const uint32_t prefix = ld + i + 1 < fd ? ld + i + 1 : fd;
            for (uint32_t j = 0; j < prefix; j++) b_ext[(size_t)i * fd8 + j] = fdist(rng);
        }

        std::vector<float> b_head((size_t)dh_dim * dh_stride, 0.0f);
        std::vector<float> idiag(dh_dim);
        for (uint32_t i = 0; i < dh_dim; i++) {
            for (uint32_t j = 0; j < i; j++) b_head[(size_t)i * dh_stride + j] = fdist(rng) * 4.0f;
            const float diag = 3.0f + 2.0f * std::abs(fdist(rng)) * 10.0f;
            b_head[(size_t)i * dh_stride + i] = diag;
            idiag[i] = 1.0f / diag;
        }

        std::vector<float> centers((size_t)num_c * dh_dim);
        for (size_t i = 0; i < centers.size(); i++) centers[i] = fdist(rng) * 8.0f;

        // Host oracle with matching FMA order.
        std::vector<int32_t> coeff(csd);
        std::vector<float> fvec(fd8);
        float wp[64], wn[64];
        auto lift_vec = [&](uint32_t v) {
            for (uint32_t i = 0; i < csd; i++) {
                int acc = 0;
                const int8_t *vp = pool.data() + (size_t)v * vec_length;
                const uint8_t *dr = b_dual.data() + (size_t)i * vec_length;
                for (uint32_t k = 0; k < vec_length; k++) acc += (int)dr[k] * (int)vp[k];
                coeff[i] = (acc + dhalf - vsum[v]) >> dshift;
            }
            for (uint32_t j = 0; j < fd8; j++) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < csd; i++) {
                    acc = std::fmaf((float)coeff[i], b_ext[(size_t)i * fd8 + j], acc);
                }
                fvec[j] = acc;
            }
        };
        auto npcheck = [&](uint32_t c, float radius2) {
            for (uint32_t j = 0; j < dh_dim; j++) {
                const float h = fvec[id_off + j];
                const float cc = centers[(size_t)c * dh_dim + j];
                wp[j] = h + cc;
                wn[j] = h - cc;
            }
            for (int k = (int)dh_dim - 1; k >= 0; k--) {
                const float qp = std::rintf(wp[k] * idiag[k]);
                const float qn = std::rintf(wn[k] * idiag[k]);
                const float *br = b_head.data() + (size_t)k * dh_stride;
                for (uint32_t j = 0; j < dh_dim; j++) {
                    wp[j] = std::fmaf(-qp, br[j], wp[j]);
                    wn[j] = std::fmaf(-qn, br[j], wn[j]);
                }
            }
            float np = 0.0f, nn = 0.0f;
            for (uint32_t j = 0; j < dh_dim; j++) {
                np = std::fmaf(wp[j], wp[j], np);
                nn = std::fmaf(wn[j], wn[j], nn);
            }
            return (radius2 > np) || (radius2 > nn);
        };

        // Choose a radius giving a moderate acceptance on a sample.
        std::vector<float> sample_norms;
        for (uint32_t v = 0; v < 400; v++) {
            lift_vec(v);
            for (uint32_t c = 0; c < num_c; c += 8) {
                for (uint32_t j = 0; j < dh_dim; j++) {
                    wp[j] = fvec[id_off + j] + centers[(size_t)c * dh_dim + j];
                }
                for (int k = (int)dh_dim - 1; k >= 0; k--) {
                    const float q = std::rintf(wp[k] * idiag[k]);
                    const float *br = b_head.data() + (size_t)k * dh_stride;
                    for (uint32_t j = 0; j < dh_dim; j++) wp[j] = std::fmaf(-q, br[j], wp[j]);
                }
                float n2 = 0.0f;
                for (uint32_t j = 0; j < dh_dim; j++) n2 = std::fmaf(wp[j], wp[j], n2);
                sample_norms.push_back(n2);
            }
        }
        std::sort(sample_norms.begin(), sample_norms.end());
        const float radius2 = sample_norms[sample_norms.size() / 50];

        // Expected per-center row multisets.
        std::vector<std::map<std::vector<uint8_t>, int> > expected(num_c);
        std::vector<uint32_t> expected_counts(num_c, 0);
        for (uint32_t v = 0; v < num_vec; v++) {
            lift_vec(v);
            for (uint32_t c = 0; c < num_c; c++) {
                if (!npcheck(c, radius2)) continue;
                expected_counts[c]++;
                std::vector<uint8_t> key((const uint8_t *)fvec.data(),
                                         (const uint8_t *)fvec.data() + fd8 * sizeof(float));
                expected[c][key]++;
            }
        }

        const uint32_t cap = 16384;
        std::vector<float> out_storage((size_t)num_c * cap * fd8);
        std::vector<float *> out_ptrs(num_c);
        for (uint32_t c = 0; c < num_c; c++) out_ptrs[c] = out_storage.data() + (size_t)c * cap * fd8;
        std::vector<uint32_t> mbound(num_c, 0);
        float radius_out = 0.0f;
        const int ok = bgj_cuda_lsh_collect_raw(
            pool.data(), 777000 + vcase, num_vec,
            vsum.data(), num_vec, vec_length, csd,
            b_dual.data(), dhalf, dshift,
            b_ext.data(), fd, fd8,
            b_head.data(), idiag.data(), dh_dim, dh_stride, id_off,
            centers.data(), num_c,
            radius2, 0, 0.0, &radius_out,
            cap, out_ptrs.data(), mbound.data());
        if (!ok) {
            printf("FAIL vcase=%d: call failed: %s\n", vcase, bgj_cuda_raw_last_error());
            failures++;
            continue;
        }
        int bad = 0;
        for (uint32_t c = 0; c < num_c; c++) {
            if (mbound[c] != expected_counts[c]) {
                printf("FAIL vcase=%d center=%u: mbound=%u expected=%u\n",
                       vcase, c, mbound[c], expected_counts[c]);
                bad = 1;
                continue;
            }
            std::map<std::vector<uint8_t>, int> got;
            for (uint32_t r = 0; r < mbound[c]; r++) {
                const uint8_t *row = (const uint8_t *)(out_ptrs[c] + (size_t)r * fd8);
                got[std::vector<uint8_t>(row, row + fd8 * sizeof(float))]++;
            }
            if (got != expected[c]) {
                printf("FAIL vcase=%d center=%u: row multiset mismatch\n", vcase, c);
                bad = 1;
            }
        }
        if (bad) failures++;

        // Calibration path: just check it runs and reports a positive radius.
        float radius_out2 = 0.0f;
        const int ok2 = bgj_cuda_lsh_collect_raw(
            pool.data(), 777000 + vcase, num_vec,
            vsum.data(), num_vec, vec_length, csd,
            b_dual.data(), dhalf, dshift,
            b_ext.data(), fd, fd8,
            b_head.data(), idiag.data(), dh_dim, dh_stride, id_off,
            centers.data(), num_c,
            radius2, 2048, (double)cap * 0.8 / num_vec * num_c, &radius_out2,
            cap, out_ptrs.data(), mbound.data());
        if (!ok2 || !(radius_out2 > 0.0f)) {
            printf("FAIL vcase=%d: calibration ok=%d radius=%f\n", vcase, ok2, radius_out2);
            failures++;
        }
    }

    if (failures) {
        printf("lsh collect test FAILED: %d failures\n", failures);
        return 1;
    }
    printf("lsh collect test passed\n");
    return 0;
}
