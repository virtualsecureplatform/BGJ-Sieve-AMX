#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>

#include "../include/svp.h"
#include "../include/utils.h"
#include "../include/lattice.h"
#include "../include/pool_epi8.h"
#include "../include/sampler.h"


#if 1
struct timeval _svptool_timer_start, _svptool_timer_end;
double _svptool_time_curr;

#define TIMER_START do {                                                                \
        gettimeofday(&_svptool_timer_start, NULL);                                       \
    } while (0)

#define TIMER_END do {                                                                  \
        gettimeofday(&_svptool_timer_end, NULL);                                         \
        _svptool_time_curr =                                                             \
            (_svptool_timer_end.tv_sec-_svptool_timer_start.tv_sec)+                      \
            (double)(_svptool_timer_end.tv_usec-_svptool_timer_start.tv_usec)/1000000.0;  \
    } while (0)

#define CURRENT_TIME (_svptool_time_curr)
#endif

template <uint32_t nb>
static int svptool_bgj1_sieve(Pool_epi8_t<nb> &p, int use_cuda, long log_level, long lps_auto_adj) {
#if defined(HAVE_CUDA)
    if (use_cuda) return p.bgj1_Sieve_cuda(log_level, lps_auto_adj);
#else
    (void)use_cuda;
#endif
    return p.bgj1_Sieve(log_level, lps_auto_adj);
}

template <uint32_t nb>
static int svptool_left_progressive_bgjf(Pool_epi8_t<nb> &p, int use_cuda, long ind_l, long ind_r,
                                         long num_threads, long log_level, long ssd) {
#if defined(HAVE_CUDA)
    if (use_cuda) return p.left_progressive_bgjfsieve_cuda(ind_l, ind_r, num_threads, log_level, ssd);
#else
    (void)use_cuda;
#endif
    return p.left_progressive_bgjfsieve(ind_l, ind_r, num_threads, log_level, ssd);
}

int show_help(int argc, char **argv) {
    std::cout << "Usage: " << argv[0] << " <inputfile> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help\t\t\tShow this help message" << std::endl;
    std::cout << "  -l, --ind_l\t\t\tLeft index of local processing" << std::endl;
    std::cout << "  -r, --ind_r\t\t\tRight index of local processing" << std::endl;
    std::cout << "  -s, --seed\t\t\tSeed for random number generator" << std::endl;
    std::cout << "  -t, --threads\t\t\tNumber of threads" << std::endl;
    std::cout << "  -v, --verbose\t\t\tVerbosity level, default = 3" << std::endl;
    std::cout << "  -f, --final\t\t\tFinal run" << std::endl;
    std::cout << "  -lsh, --lsh\t\t\tUse lsh lifting and specify qratio and ext_qratio" << std::endl;
    std::cout << "  -ssd, --ssd\t\t\tStart sieving dimension" << std::endl;
    std::cout << "  -msd, --msd\t\t\tMaximal sieving dimension" << std::endl;
    std::cout << "  -esd, --esd\t\t\tExtended sieving dimension" << std::endl;
    std::cout << "  -ds, --down_sieve\t\tDown sieve" << std::endl;
    std::cout << "  -cuda, --cuda\t\t\tUse CUDA for BGJ1 phases where available" << std::endl;
    #if defined(__AMX_INT8__)
    std::cout << "  -amx, --amx\t\t\tUse amx" << std::endl;
    #endif
    std::cout << std::endl;
    return 0;

}

int main(int argc, char** argv) {
    long help = 0;
    long down_sieve = 1;
    long sr = 0;
    long shuf = 0;
    long ind_l = 0;
    long ind_r = 0;
    long lsh = 0;
    long fin = 0;
    long ssd = 45;
    long msd = 0;
    long esd = 0;
    long log_level = 3;
    long num_threads = 1;
    long seed = time(NULL);
    double lsh_qr = 0.0;
    double lsh_er = 0.0;
    int cuda = 0;
    int amx = 0;

    for (long i = 1; i < argc; i++) {
        if (!strcasecmp(argv[i], "-h") || !strcasecmp(argv[i], "--help")) {
            help = 1;
        } else if (!strcasecmp(argv[i], "-l") || !strcasecmp(argv[i], "--ind_l")) {
            if (i < argc - 1) ind_l = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-r") || !strcasecmp(argv[i], "--ind_r")) {
            if (i < argc - 1) ind_r = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-s") || !strcasecmp(argv[i], "--seed")) {
            if (i < argc - 1) seed = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-t") || !strcasecmp(argv[i], "--threads")) {
            if (i < argc - 1) num_threads = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-v") || !strcasecmp(argv[i], "--verbose")) {
            if (i < argc - 1) log_level = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-lsh") || !strcasecmp(argv[i], "--lsh")) {
            lsh = 1;
            if (i < argc - 1) lsh_qr = atof(argv[i+1]);
            if (i < argc - 2) lsh_er = atof(argv[i+2]);
        } else if (!strcasecmp(argv[i], "-f") || !strcasecmp(argv[i], "--final")) {
            fin = 1;
        } else if (!strcasecmp(argv[i], "-msd") || !strcasecmp(argv[i], "--msd")) {
            if (i < argc - 1) msd = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-esd") || !strcasecmp(argv[i], "--esd")) {
            if (i < argc - 1) esd = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-ssd") || !strcasecmp(argv[i], "--ssd")) {
            if (i < argc - 1) ssd = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-ds") || !strcasecmp(argv[i], "--down_sieve")) {
            if (i < argc - 1) down_sieve = atol(argv[++i]);
        } else if (!strcasecmp(argv[i], "-sr") || !strcasecmp(argv[i], "--sr")) {
            sr = 1;
        } else if (!strcasecmp(argv[i], "-shuf") || !strcasecmp(argv[i], "--shuf")) {
            shuf = 1;
        } else if (!strcasecmp(argv[i], "-cuda") || !strcasecmp(argv[i], "--cuda")) {
            cuda = 1;
        } else if (!strcasecmp(argv[i], "-amx") || !strcasecmp(argv[i], "--amx")) {
            amx = 1;
        }
    }

    if (cuda) {
        #if !defined(HAVE_CUDA)
        printf("Error: CUDA is not supported, disabled.\n");
        cuda = 0;
        #endif
    }
    if (amx) {
        #if !defined(__AMX_INT8__)
        printf("Error: AMX is not supported, disabled.\n");
        amx = 0;
        #endif
    }
    if (msd == 0) help = 1;
    if (lsh && lsh_qr == 0.0 && !fin) lsh_qr = amx ? 0.1 : 0.2;
    if (lsh && lsh_er == 0.0 && !fin) lsh_er = amx ? 0.1 : 0.2;
    if (help) { show_help(argc, argv); return 0; }
    int suppress_cuda_bucket_target = 0;
    if (cuda) {
        setenv("BGJ_SVP_CUDA", "1", 1);
        if (getenv("BGJ_CUDA_MIN_CSD") == NULL) setenv("BGJ_CUDA_MIN_CSD", "80", 0);
        if (getenv("BGJ_CUDA_BUCKET_TARGET_SIZE") == NULL &&
            getenv("BGJ1_EPI8_BUCKET_TARGET_SIZE") == NULL &&
            getenv("BGJ_EPI8_BUCKET_TARGET_SIZE") == NULL &&
            getenv("BGJ1_EPI8_BUCKET_ALPHA") == NULL &&
            getenv("BGJ_EPI8_BUCKET_ALPHA") == NULL) {
            setenv("BGJ_CUDA_BUCKET_TARGET_SIZE", "0", 0);
            suppress_cuda_bucket_target = 1;
        }
        if (suppress_cuda_bucket_target && getenv("BGJ_CUDA_MATERIALIZE") == NULL) {
            setenv("BGJ_CUDA_MATERIALIZE", "0", 0);
        }
    }
    
    SetSamplerSeed((uint64_t)seed);
    
    printf("command: ");
    for (long i = 0; i < argc; i++) {
        printf("%s ", argv[i]);
    }
    printf("\n");
    fflush(stdout);

    TIMER_START;
    Lattice_QP L(argv[1]);
    if (L.NumRows() == 0) return 0;

    if (ind_l != 0 || ind_r != 0) {
        if (ind_l != 0 && ind_r == 0) ind_r = L.NumRows();
        Lattice_QP *L_loc = L.b_loc_QP(ind_l, ind_r);
        if (lsh) {
            if (amx) {
                #if defined(__AMX_INT8__)
                __lsh_pump_red_amx(L_loc, num_threads, 1.2, lsh_qr, msd, L_loc->NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, shuf, ssd);
                #endif
            } else {
                __lsh_pump_red_epi8(L_loc, num_threads, 1.2, lsh_qr, msd, L_loc->NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, shuf, ssd);
            }
        } else {
            if (amx) {
                #if defined(__AMX_INT8__)
                __pump_red_amx(L_loc, num_threads, 1.2, msd, L_loc->NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, shuf, ssd);
                #endif
            } else {
                __pump_red_epi8(L_loc, num_threads, 1.2, msd, L_loc->NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, shuf, ssd);
            }
        }
        L.trans_to(ind_l, ind_r, L_loc);
        L.compute_gso_QP();
        L.size_reduce();
        L.LLL_QP();
    } else {
        if (fin) {
            if (lsh) {
                if (amx) {
                    #if defined(__AMX_INT8__)
                    __last_lsh_pump_amx(&L, num_threads, lsh_qr, lsh_er, msd, esd, log_level, 0, ssd);
                    #endif
                } else {
                    __last_lsh_pump_epi8(&L, num_threads, lsh_qr, lsh_er, msd, esd, log_level, 0, ssd);
                }
            } else {
#define FINAL_PUMP do {                                                                             \
                    svptool_left_progressive_bgjf(p, cuda, L.NumRows() - msd, L.NumRows(), num_threads, 16384+log_level - 3, ssd); \
                    while (p.CSD < msd + esd) {                                                                 \
                        p.extend_left();                                                                        \
                        if (p.CSD >= 92) {                                                                      \
                            p.bgj3_Sieve(log_level - 3, 1);                                                     \
                        } else if (p.CSD > 80) {                                                                \
                            p.bgj2_Sieve(log_level - 3, 1);                                                     \
                        } else {                                                                                \
                            svptool_bgj1_sieve(p, cuda, log_level - 3, 1);                                      \
                        }                                                                                       \
                        p.show_min_lift(0);                                                                     \
                    }                                                                                           \
                } while (0)
                if (amx) {
                    #if defined(__AMX_INT8__)
                    Pool_epi8_t<5> p(&L);
                    p.left_progressive_amx(L.NumRows() - msd, L.NumRows(), num_threads, 16384+log_level - 3, ssd);
                    while (p.CSD < msd + esd) {
                        p.extend_left();
                        p.bgj_amx_upsieve(log_level - 3, -1, -1.0, -1, -10.0);
                        p.show_min_lift(0);
                    }
                    for (;;) {
                        std::ifstream INstream(".in");
                        if (!INstream) break;

                        char instruction;
                        INstream >> instruction;
                        if (instruction == 's') {
                            p.bgj_amx_upsieve(log_level-3, 6, -1.0, -1, -10.0);
                            p.show_min_lift(0);
                        } else if (instruction == 'h') {
                            std::ifstream VINstream(".vin");
                            double qr;
                            VINstream >> qr;
                            if (qr > 0.01 && qr < 1.0) {
                                p.show_lsfsh_insert(0, 10.0, log_level-3, 0, 0, qr);
                            }
                            VINstream.close();
                        } else if (instruction == 'r') {
                            std::ifstream VINstream(".vin");
                            double vinval = 0.0;
                            long begin_ind = 0;
                            long ds = 0;
                            VINstream >> vinval;
                            begin_ind = round(vinval);
                            vinval -= begin_ind;
                            ds = round(vinval * 1000.0);
                            VINstream.close();
                            for (long i = begin_ind; i < begin_ind + 16; i++) {
                                p.insert(i, 10.0);
                                p.tail_LLL(0.99, p.CSD);
                                if (ds > 0) {
                                    if (pow(4./3., p.CSD *0.5) * 6.4 < p.num_vec) {
                                        p.shrink(pow(4./3., p.CSD *0.5) * 6.4);
                                    }
                                    p.bgj_amx_upsieve(log_level - 3, -1, -1.0, -1, -10.0);
                                    ds--;
                                }
                            }
                            p.basis->store("ftmp");
                            break;
                        } else {
                            sleep(60);
                        }
                        INstream.close();
                    }
                    #endif
                } else if (msd + esd > 224) {
                    printf("Error: msd + esd > 224, please recompile with a larger Pool_epi8_t\n");
                } else if (msd + esd > 192) {
                    #if COMPILE_POOL_EPI8_224
                    Pool_epi8_t<7> p(&L);
                    FINAL_PUMP;
                    #else
                    printf("Error: msd + esd > 192, please recompile with COMPILE_POOL_EPI8_224 = 1\n");
                    #endif
                } else if (msd + esd > 160) {
                    #if COMPILE_POOL_EPI8_192
                    Pool_epi8_t<6> p(&L);
                    FINAL_PUMP;
                    #else
                    printf("Error: msd + esd > 160, please recompile with COMPILE_POOL_EPI8_192 = 1\n");
                    #endif
                } else if (msd + esd > 128) {
                    #if COMPILE_POOL_EPI8_160
                    Pool_epi8_t<5> p(&L);
                    FINAL_PUMP;
                    #else
                    printf("Error: msd + esd > 128, please recompile with COMPILE_POOL_EPI8_160 = 1\n");
                    #endif
                } else if (msd + esd > 96) {
                    #if COMPILE_POOL_EPI8_128
                    Pool_epi8_t<4> p(&L);
                    FINAL_PUMP;
                    #else
                    printf("Error: msd + esd > 96, please recompile with COMPILE_POOL_EPI8_128 = 1\n");
                    #endif
                } else {
                    Pool_epi8_t<3> p(&L);
                    FINAL_PUMP;
                }
#undef FINAL_PUMP
            }
        } else {
            if (lsh) {
                if (amx) {
                    #if defined(__AMX_INT8__)
                    __lsh_pump_red_amx(&L, num_threads, 1.2, lsh_qr, msd, L.NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, 0, ssd);
                    #endif
                } else {
                    __lsh_pump_red_epi8(&L, num_threads, 1.2, lsh_qr, msd, L.NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, 0, ssd);
                }
            } else {
                if (amx) {
                    #if defined(__AMX_INT8__)
                    __pump_red_amx(&L, num_threads, 1.2, msd, L.NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, 0, ssd);
                    #endif
                } else {
                    __pump_red_epi8(&L, num_threads, 1.2, msd, L.NumRows() - msd, (msd-50), 0, down_sieve ? (msd-50) : 0, log_level, 0, ssd);
                }
            }
        }
    }
    TIMER_END;

    printf("time: %fs\n", CURRENT_TIME);

    char output_file[256];
    sprintf(output_file, "%s%s", argv[1], sr ? "" : "r");
    L.store(output_file);
    return 0;
}
