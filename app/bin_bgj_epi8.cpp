#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config.h"
#include "../include/lattice.h"
#include "../include/pool_epi8.h"
#include "../include/bgj_cuda.h"
#include "../include/sampler.h"


int main(int argc, char** argv) {
    long help = 0;
    long algo = -2;
    long num_threads = -2;
    long log_level = -2;
    long seed = -2;
    long print_min_lift = 0;
    long require_quality = 0;
    double quality_gamma = 1.05;

    for (long i = 0; i < argc; i++) {
        if (!strcasecmp(argv[i], "-h") || !strcasecmp(argv[i], "--help")) help = 1;
        if (!strcasecmp(argv[i], "-a") || !strcasecmp(argv[i], "--algo")) {
            if (i < argc - 1) {
                if (!strcasecmp(argv[i+1], "bgjf")) algo = 0;
                if (!strcasecmp(argv[i+1], "bgj1")) algo = 1;
                if (!strcasecmp(argv[i+1], "bgj2")) algo = 2;
                if (!strcasecmp(argv[i+1], "bgj3")) algo = 3;
                #if defined(HAVE_CUDA)
                if (!strcasecmp(argv[i+1], "cuda") ||
                    !strcasecmp(argv[i+1], "bgjf-cuda") ||
                    !strcasecmp(argv[i+1], "cuda-bgjf") ||
                    !strcasecmp(argv[i+1], "cuda-f")) algo = 7;
                if (!strcasecmp(argv[i+1], "bgj1-cuda") ||
                    !strcasecmp(argv[i+1], "cuda-bgj1")) algo = 8;
                #endif
                #if defined(__AMX_INT8__)
                if (!strcasecmp(argv[i+1], "amx")) algo = 9;
                #endif
            }
        }
        if (!strcasecmp(argv[i], "-t") || !strcasecmp(argv[i], "--threads")) {
            if (i < argc - 1) num_threads = atol(argv[i+1]);
        }
        if (!strcasecmp(argv[i], "-l") || !strcasecmp(argv[i], "--log")) {
            if (i < argc - 1) log_level = atol(argv[i+1]);
        }
        if (!strcasecmp(argv[i], "-s") || !strcasecmp(argv[i], "--seed")) {
            if (i < argc - 1) seed = atol(argv[i+1]);
        }
        if (!strcasecmp(argv[i], "-p") || !strcasecmp(argv[i], "--print")) {
            print_min_lift = 1;
        }
        if (!strcasecmp(argv[i], "--print-final")) {
            print_min_lift = 2;
        }
        if (!strcasecmp(argv[i], "--require-quality") ||
            !strcasecmp(argv[i], "--require-svp-quality") ||
            !strcasecmp(argv[i], "--svp-challenge")) {
            require_quality = 1;
        }
        if (!strcasecmp(argv[i], "--quality-gamma") || !strcasecmp(argv[i], "--gamma")) {
            if (i >= argc - 1) {
                fprintf(stderr, "Error: --quality-gamma requires a value.\n");
                return 2;
            }
            quality_gamma = atof(argv[++i]);
        }
    }
    if (argc < 2 || help) {
        #if defined(__AMX_INT8__) && defined(HAVE_CUDA)
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3|cuda|bgj1-cuda|amx] [num_threads] [log_level] [seed] [-p|--print|--print-final] [--require-quality] [--quality-gamma gamma]\n", argv[0]);
        #elif defined(__AMX_INT8__)
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3|amx] [num_threads] [log_level] [seed] [-p|--print|--print-final] [--require-quality] [--quality-gamma gamma]\n", argv[0]);
        #elif defined(HAVE_CUDA)
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3|cuda|bgj1-cuda] [num_threads] [log_level] [seed] [-p|--print|--print-final] [--require-quality] [--quality-gamma gamma]\n", argv[0]);
        #else
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3] [num_threads] [log_level] [seed] [-p|--print|--print-final] [--require-quality] [--quality-gamma gamma]\n", argv[0]);
        #endif
        return 0;
    }
    
    if (algo == -2) {
        algo = 0;
        if (argc > 2) {
            if (!strcasecmp(argv[2], "bgj1")) algo = 1;
            if (!strcasecmp(argv[2], "bgj2")) algo = 2;
            if (!strcasecmp(argv[2], "bgj3")) algo = 3;
            #if defined(HAVE_CUDA)
            if (!strcasecmp(argv[2], "cuda") ||
                !strcasecmp(argv[2], "bgjf-cuda") ||
                !strcasecmp(argv[2], "cuda-bgjf") ||
                !strcasecmp(argv[2], "cuda-f")) algo = 7;
            if (!strcasecmp(argv[2], "bgj1-cuda") ||
                !strcasecmp(argv[2], "cuda-bgj1")) algo = 8;
            #endif
            #if defined(__AMX_INT8__)
            if (!strcasecmp(argv[2], "amx")) algo = 9;
            #endif
        }
    }
    if (num_threads == -2) num_threads = (argc > 3) ? atol(argv[3]) : 1;
    if (log_level == -2) log_level = (argc > 4) ? atol(argv[4]) : 0;
    if (seed == -2) {
        const char *seed_env = getenv("BGJ_SEED");
        if (seed_env && seed_env[0]) seed = atol(seed_env);
    }
    if (seed == -2 && argc > 5) seed = atol(argv[5]);
    if (seed != -2) SetSamplerSeed((uint64_t)seed);
    if (require_quality && print_min_lift == 0) print_min_lift = 2;
    if (quality_gamma <= 0.0) {
        fprintf(stderr, "Error: quality gamma must be positive.\n");
        return 2;
    }
    if (print_min_lift == 1) {
        log_level += 16384;
    } else if (print_min_lift == 2 && (algo == 7 || algo == 8)) {
        log_level += 32768;
    }

    Lattice_QP L(argv[1]);
    Pool_epi8_t<3> p3(&L);
    #if COMPILE_POOL_EPI8_128
    Pool_epi8_t<4> p4(&L);
    #endif
    #if COMPILE_POOL_EPI8_160
    Pool_epi8_t<5> p5(&L);
    #endif
    #if COMPILE_POOL_EPI8_192
    Pool_epi8_t<6> p6(&L);
    #endif
    #if COMPILE_POOL_EPI8_224
    Pool_epi8_t<7> p7(&L);
    #endif

    int quality_valid = 0;
    double quality_norm = 0.0;
    double quality_gh = 0.0;
    double quality_approx = 0.0;

#define CAPTURE_QUALITY(POOL) do {                                             \
        if ((print_min_lift == 2 || require_quality) && !(POOL).last_lift_valid) { \
            (POOL).show_min_lift(0);                                           \
        }                                                                      \
        quality_valid = (POOL).last_lift_valid;                                \
        quality_norm = (POOL).last_lift_euclidean_norm;                        \
        quality_gh = (POOL).last_lift_gh;                                      \
        quality_approx = (POOL).last_lift_approx_factor;                       \
    } while (0)

#if COMPILE_POOL_EPI8_128
#define RUN_EPI8_128_OR_FALLBACK(METHOD, DIM) do {                             \
        p4.METHOD(0, (DIM), num_threads, log_level);                           \
        CAPTURE_QUALITY(p4);                                                   \
    } while (0)
#else
#define RUN_EPI8_128_OR_FALLBACK(METHOD, DIM) do {                             \
        p3.METHOD((DIM) - 96, (DIM), num_threads, log_level);                  \
        CAPTURE_QUALITY(p3);                                                   \
    } while (0)
#endif

#if COMPILE_POOL_EPI8_160
#define RUN_EPI8_160_OR_FALLBACK(METHOD, DIM) do {                             \
        p5.METHOD(0, (DIM), num_threads, log_level);                           \
        CAPTURE_QUALITY(p5);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_128
#define RUN_EPI8_160_OR_FALLBACK(METHOD, DIM) do {                             \
        p4.METHOD((DIM) - 128, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p4);                                                   \
    } while (0)
#else
#define RUN_EPI8_160_OR_FALLBACK(METHOD, DIM) do {                             \
        p3.METHOD((DIM) - 96, (DIM), num_threads, log_level);                  \
        CAPTURE_QUALITY(p3);                                                   \
    } while (0)
#endif

#if COMPILE_POOL_EPI8_192
#define RUN_EPI8_192_OR_FALLBACK(METHOD, DIM) do {                             \
        p6.METHOD(0, (DIM), num_threads, log_level);                           \
        CAPTURE_QUALITY(p6);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_160
#define RUN_EPI8_192_OR_FALLBACK(METHOD, DIM) do {                             \
        p5.METHOD((DIM) - 160, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p5);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_128
#define RUN_EPI8_192_OR_FALLBACK(METHOD, DIM) do {                             \
        p4.METHOD((DIM) - 128, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p4);                                                   \
    } while (0)
#else
#define RUN_EPI8_192_OR_FALLBACK(METHOD, DIM) do {                             \
        p3.METHOD((DIM) - 96, (DIM), num_threads, log_level);                  \
        CAPTURE_QUALITY(p3);                                                   \
    } while (0)
#endif

#if COMPILE_POOL_EPI8_224
#define RUN_EPI8_224_OR_FALLBACK(METHOD, DIM) do {                             \
        p7.METHOD(0, (DIM), num_threads, log_level);                           \
        CAPTURE_QUALITY(p7);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_192
#define RUN_EPI8_224_OR_FALLBACK(METHOD, DIM) do {                             \
        p6.METHOD((DIM) - 192, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p6);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_160
#define RUN_EPI8_224_OR_FALLBACK(METHOD, DIM) do {                             \
        p5.METHOD((DIM) - 160, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p5);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_128
#define RUN_EPI8_224_OR_FALLBACK(METHOD, DIM) do {                             \
        p4.METHOD((DIM) - 128, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p4);                                                   \
    } while (0)
#else
#define RUN_EPI8_224_OR_FALLBACK(METHOD, DIM) do {                             \
        p3.METHOD((DIM) - 96, (DIM), num_threads, log_level);                  \
        CAPTURE_QUALITY(p3);                                                   \
    } while (0)
#endif

#if COMPILE_POOL_EPI8_224
#define RUN_EPI8_MAX_OR_FALLBACK(METHOD, DIM) do {                             \
        p7.METHOD((DIM) - 224, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p7);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_192
#define RUN_EPI8_MAX_OR_FALLBACK(METHOD, DIM) do {                             \
        p6.METHOD((DIM) - 192, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p6);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_160
#define RUN_EPI8_MAX_OR_FALLBACK(METHOD, DIM) do {                             \
        p5.METHOD((DIM) - 160, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p5);                                                   \
    } while (0)
#elif COMPILE_POOL_EPI8_128
#define RUN_EPI8_MAX_OR_FALLBACK(METHOD, DIM) do {                             \
        p4.METHOD((DIM) - 128, (DIM), num_threads, log_level);                 \
        CAPTURE_QUALITY(p4);                                                   \
    } while (0)
#else
#define RUN_EPI8_MAX_OR_FALLBACK(METHOD, DIM) do {                             \
        p3.METHOD((DIM) - 96, (DIM), num_threads, log_level);                  \
        CAPTURE_QUALITY(p3);                                                   \
    } while (0)
#endif

#define RUN_EPI8_PROGRESSIVE(METHOD) do {                                      \
        const long dim__ = L.NumRows();                                        \
        if (dim__ <= 96) {                                                     \
            p3.METHOD(0, dim__, num_threads, log_level);                       \
            CAPTURE_QUALITY(p3);                                               \
        } else if (dim__ <= 128) {                                             \
            RUN_EPI8_128_OR_FALLBACK(METHOD, dim__);                          \
        } else if (dim__ <= 160) {                                             \
            RUN_EPI8_160_OR_FALLBACK(METHOD, dim__);                          \
        } else if (dim__ <= 192) {                                             \
            RUN_EPI8_192_OR_FALLBACK(METHOD, dim__);                          \
        } else if (dim__ <= 224) {                                             \
            RUN_EPI8_224_OR_FALLBACK(METHOD, dim__);                          \
        } else {                                                               \
            RUN_EPI8_MAX_OR_FALLBACK(METHOD, dim__);                          \
        }                                                                      \
    } while (0)

    if (algo == 0){
        RUN_EPI8_PROGRESSIVE(left_progressive_bgjfsieve);
    } else if (algo == 1) {
        RUN_EPI8_PROGRESSIVE(left_progressive_bgj1sieve);
    } else if (algo == 2) {
        RUN_EPI8_PROGRESSIVE(left_progressive_bgj2sieve);
    } else if (algo == 3) {
        RUN_EPI8_PROGRESSIVE(left_progressive_bgj3sieve);
    } else if (algo == 7) {
        #if defined(HAVE_CUDA)
        RUN_EPI8_PROGRESSIVE(left_progressive_bgjfsieve_cuda);
        #else
        fprintf(stderr, "Error: CUDA support is not compiled in. Rebuild with `make CUDA=1`.\n");
        return 1;
        #endif
    } else if (algo == 8) {
        #if defined(HAVE_CUDA)
        RUN_EPI8_PROGRESSIVE(left_progressive_bgj1sieve_cuda);
        #else
        fprintf(stderr, "Error: CUDA support is not compiled in. Rebuild with `make CUDA=1`.\n");
        return 1;
        #endif
    } else if (algo == 9) {
        #if defined(__AMX_INT8__)
        p5.left_progressive_amx(0, L.NumRows(), num_threads, log_level);
        CAPTURE_QUALITY(p5);
        #endif
    }
#undef RUN_EPI8_PROGRESSIVE
#undef RUN_EPI8_MAX_OR_FALLBACK
#undef RUN_EPI8_224_OR_FALLBACK
#undef RUN_EPI8_192_OR_FALLBACK
#undef RUN_EPI8_160_OR_FALLBACK
#undef RUN_EPI8_128_OR_FALLBACK
#undef CAPTURE_QUALITY
    if (require_quality) {
        if (!quality_valid || quality_gh <= 0.0) {
            fprintf(stderr, "quality check failed: final lifted quality is unavailable\n");
            return 3;
        }
        const double quality_bound = quality_gamma * quality_gh;
        if (quality_norm > quality_bound) {
            fprintf(stderr,
                    "quality check failed: norm=%.9g bound=%.9g gh=%.9g approx=%.9g gamma=%.9g\n",
                    quality_norm, quality_bound, quality_gh, quality_approx, quality_gamma);
            return 3;
        }
        fprintf(stderr,
                "quality check passed: norm=%.9g bound=%.9g gh=%.9g approx=%.9g gamma=%.9g\n",
                quality_norm, quality_bound, quality_gh, quality_approx, quality_gamma);
    }
    return 0;
}
