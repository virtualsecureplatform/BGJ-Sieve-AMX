#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config.h"
#include "../include/lattice.h"
#include "../include/pool_epi8.h"
#include "../include/bgj_cuda.h"


int main(int argc, char** argv) {
    long help = 0;
    long algo = -2;
    long num_threads = -2;
    long log_level = -2;
    long print_min_lift = 0;

    for (long i = 0; i < argc; i++) {
        if (!strcasecmp(argv[i], "-h") || !strcasecmp(argv[i], "--help")) help = 1;
        if (!strcasecmp(argv[i], "-a") || !strcasecmp(argv[i], "--algo")) {
            if (i < argc - 1) {
                if (!strcasecmp(argv[i+1], "bgjf")) algo = 0;
                if (!strcasecmp(argv[i+1], "bgj1")) algo = 1;
                if (!strcasecmp(argv[i+1], "bgj2")) algo = 2;
                if (!strcasecmp(argv[i+1], "bgj3")) algo = 3;
                #if defined(HAVE_CUDA)
                if (!strcasecmp(argv[i+1], "cuda") || !strcasecmp(argv[i+1], "bgj1-cuda")) algo = 7;
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
        if (!strcasecmp(argv[i], "-p") || !strcasecmp(argv[i], "--print")) {
            print_min_lift = 1;
        }
    }
    if (argc < 2 || help) {
        #if defined(__AMX_INT8__) && defined(HAVE_CUDA)
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3|cuda|amx] [num_threads] [log_level]\n", argv[0]);
        #elif defined(__AMX_INT8__)
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3|amx] [num_threads] [log_level]\n", argv[0]);
        #elif defined(HAVE_CUDA)
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3|cuda] [num_threads] [log_level]\n", argv[0]);
        #else
        printf("Usage: %s <lattice_file> [bgjf|bgj1|bgj2|bgj3] [num_threads] [log_level]\n", argv[0]);
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
            if (!strcasecmp(argv[2], "cuda") || !strcasecmp(argv[2], "bgj1-cuda")) algo = 7;
            #endif
            #if defined(__AMX_INT8__)
            if (!strcasecmp(argv[2], "amx")) algo = 9;
            #endif
        }
    }
    if (num_threads == -2) num_threads = (argc > 3) ? atol(argv[3]) : 1;
    if (log_level == -2) log_level = (argc > 4) ? atol(argv[4]) : 0;
    log_level += print_min_lift * 16384;

    Lattice_QP L(argv[1]);
    Pool_epi8_t<3> p3(&L);
    #if COMPILE_POOL_EPI8_128
    Pool_epi8_t<4> p4(&L);
    #endif
    #if COMPILE_POOL_EPI8_160
    Pool_epi8_t<5> p5(&L);
    #endif

    if (algo == 0){
        if (L.NumRows() <= 96) {
            p3.left_progressive_bgjfsieve(0, L.NumRows(), num_threads, log_level);
        } else if (L.NumRows() <= 128) {
            #if COMPILE_POOL_EPI8_128
            p4.left_progressive_bgjfsieve(0, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgjfsieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        } else {
            #if COMPILE_POOL_EPI8_160
            p5.left_progressive_bgjfsieve(0, L.NumRows(), num_threads, log_level);
            #elif COMPILE_POOL_EPI8_128
            p4.left_progressive_bgjfsieve(L.NumRows() - 128, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgjfsieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        }    
    } else if (algo == 1) {
        if (L.NumRows() <= 96) {
            p3.left_progressive_bgj1sieve(0, L.NumRows(), num_threads, log_level);
        } else if (L.NumRows() <= 128) {
            #if COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj1sieve(0, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj1sieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        } else {
            #if COMPILE_POOL_EPI8_160
            p5.left_progressive_bgj1sieve(0, L.NumRows(), num_threads, log_level);
            #elif COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj1sieve(L.NumRows() - 128, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj1sieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        }    
    } else if (algo == 2) {
        if (L.NumRows() <= 96) {
            p3.left_progressive_bgj2sieve(0, L.NumRows(), num_threads, log_level);
        } else if (L.NumRows() <= 128) {
            #if COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj2sieve(0, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj2sieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        } else {
            #if COMPILE_POOL_EPI8_160
            p5.left_progressive_bgj2sieve(0, L.NumRows(), num_threads, log_level);
            #elif COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj2sieve(L.NumRows() - 128, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj2sieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        }    
    } else if (algo == 3) {
        if (L.NumRows() <= 96) {
            p3.left_progressive_bgj3sieve(0, L.NumRows(), num_threads, log_level);
        } else if (L.NumRows() <= 128) {
            #if COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj3sieve(0, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj3sieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        } else {
            #if COMPILE_POOL_EPI8_160
            p5.left_progressive_bgj3sieve(0, L.NumRows(), num_threads, log_level);
            #elif COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj3sieve(L.NumRows() - 128, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj3sieve(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        }
    } else if (algo == 7) {
        #if defined(HAVE_CUDA)
        if (L.NumRows() <= 96) {
            p3.left_progressive_bgj1sieve_cuda(0, L.NumRows(), num_threads, log_level);
        } else if (L.NumRows() <= 128) {
            #if COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj1sieve_cuda(0, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj1sieve_cuda(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        } else {
            #if COMPILE_POOL_EPI8_160
            p5.left_progressive_bgj1sieve_cuda(0, L.NumRows(), num_threads, log_level);
            #elif COMPILE_POOL_EPI8_128
            p4.left_progressive_bgj1sieve_cuda(L.NumRows() - 128, L.NumRows(), num_threads, log_level);
            #else
            p3.left_progressive_bgj1sieve_cuda(L.NumRows() - 96, L.NumRows(), num_threads, log_level);
            #endif
        }
        #else
        fprintf(stderr, "Error: CUDA support is not compiled in. Rebuild with `make CUDA=1`.\n");
        return 1;
        #endif
    } else if (algo == 9) {
        #if defined(__AMX_INT8__)
        p5.left_progressive_amx(0, L.NumRows(), num_threads, log_level);
        #endif
    }
    return 0;
}
