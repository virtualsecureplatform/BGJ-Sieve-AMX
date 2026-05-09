#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#include <sys/time.h>

#include "NTL/LLL.h"

#include "../include/svp.h"
#include "../include/utils.h"
#include "../include/lattice.h"
#include "../include/pool_epi8.h"
#include "../include/sampler.h"
#include "../include/fplll_bridge.h"

#if 1
struct timeval _solver_timer_start, _solver_timer_end;
double _solver_time_curr;

#define TIMER_START do {                                                                \
        gettimeofday(&_solver_timer_start, NULL);                                       \
    } while (0)

#define TIMER_END do {                                                                  \
        gettimeofday(&_solver_timer_end, NULL);                                         \
        _solver_time_curr =                                                             \
            (_solver_timer_end.tv_sec-_solver_timer_start.tv_sec)+                      \
            (double)(_solver_timer_end.tv_usec-_solver_timer_start.tv_usec)/1000000.0;  \
    } while (0)

#define CURRENT_TIME (_solver_time_curr)
#endif


#define SVPALGO_NULL        0
#define SVPALGO_100T90      1
#define SVPALGO_110T90      2
#define SVPALGO_120T95      3
#define SVPALGO_130T98      4
#define SVPALGO_140T100     5
#define SVPALGO_150T102     6

long num_threads = 1;
long profile = 0;

static double solver_now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void solver_profile_line(const char *name, double seconds)
{
    if (profile) {
        printf("profile: %-32s %.6fs\n", name, seconds);
        fflush(stdout);
    }
}

static int solver_use_fplll_initial_lll()
{
    if (!bgj_fplll_is_available()) return 0;
    const char *backend = getenv("BGJ_LLL_BACKEND");
    if (backend == NULL || backend[0] == '\0') return 1;
    if (!strcasecmp(backend, "fplll") || !strcasecmp(backend, "1") ||
        !strcasecmp(backend, "true") || !strcasecmp(backend, "yes")) return 1;
    return 0;
}

#define SOLVER_PROFILE_DO(label, stmt) do {               \
        double __solver_profile_t0 = solver_now();        \
        stmt;                                             \
        solver_profile_line(label, solver_now() - __solver_profile_t0); \
    } while (0)

int __progressive_LLL(Lattice_QP *L) {
    L->LLL_QP(0.5);
    L->LLL_QP(0.75);
    L->LLL_QP(0.9);
    L->LLL_QP(0.99);
    return 1;
}
int __deep40_tour(Lattice_QP *L) {
    for (long i = 0; i <= L->NumRows() - 40; i += 10) L->LLL_DEEP_QP(0.99, i, i + 40);
    return 1;
}
int __local_pump(Lattice_QP *L, long msd, long d4f, long ind_l, long ind_r) {
    Lattice_QP *L_loc = L->b_loc_QP(ind_l, ind_r);
    __pump_red_epi8(L_loc, num_threads, 1.1, msd, d4f, 24, 0, 24, 0, 0, 45);
    L->trans_to(ind_l, ind_r, L_loc);
    L->compute_gso_QP();
    L->size_reduce();
    delete L_loc;
    return 1;
}
int __local_lsh_pump(Lattice_QP *L, long msd, long d4f, long ind_l, long ind_r) {
    Lattice_QP *L_loc = L->b_loc_QP(ind_l, ind_r);
    __lsh_pump_red_epi8(L_loc, num_threads, 1.1, 0.2, msd, d4f, 24, 0, 24, 0, 0, 45);
    L->trans_to(ind_l, ind_r, L_loc);
    L->compute_gso_QP();
    L->size_reduce();
    delete L_loc;
    return 1;
}
int __local_dual_pump(Lattice_QP *L, long msd, long d4f, long ind_l, long ind_r) {
    Lattice_QP *L_loc = L->b_loc_QP(ind_l, ind_r);
    Lattice_QP *L_dual = L_loc->dual_QP();

    L_dual->usd();
    L_dual->size_reduce();
    __pump_red_epi8(L_dual, num_threads, 1.1, msd, d4f, 24, 0, 24, 0);
    L_dual->usd();
    Lattice_QP *L_dual_dual = L_dual->dual_QP();

    L->trans_to(ind_l, ind_r, L_dual_dual);
    L->size_reduce();
    delete L_loc;
    delete L_dual;
    delete L_dual_dual;
    L->LLL_QP(0.99);
    return 1;
}

int _svp_solver_red(Lattice_QP* L, long algo) {
    if (algo == SVPALGO_NULL) {
        return 0;
    }
    if (algo == SVPALGO_100T90) {
        double t0 = solver_now();
        __progressive_LLL(L);
        solver_profile_line("100t90 progressive_LLL", solver_now() - t0);
        for (long l = 0; l < 3; l++) {
            char label[64];
            snprintf(label, sizeof(label), "100t90 deep40_tour_%ld", l + 1);
            t0 = solver_now();
            __deep40_tour(L);
            solver_profile_line(label, solver_now() - t0);
        }
        t0 = solver_now();
        __local_pump(L, 65, 15, 0, 80);
        solver_profile_line("100t90 local_pump_65_0_80", solver_now() - t0);
        t0 = solver_now();
        __local_dual_pump(L, 65, 15, 20, 100);
        solver_profile_line("100t90 local_dual_pump_65_20_100", solver_now() - t0);
        t0 = solver_now();
        __local_pump(L, 72, 15, 0, 87);
        solver_profile_line("100t90 local_pump_72_0_87", solver_now() - t0);
        t0 = solver_now();
        __pump_red_epi8(L, 1, 100, 82, 18, 1, 0, 0, 0, 0);
        solver_profile_line("100t90 final_pump_82_d4f18", solver_now() - t0);

        return 1;
    }
    if (algo == SVPALGO_110T90) {
        __progressive_LLL(L);
        for (long l = 0; l < 4; l++) __deep40_tour(L);
        __local_pump(L, 60, 15, 0, 75);
        __local_pump(L, 60, 15, 15, 90);
        __local_dual_pump(L, 60, 15, 35, 110);
        __local_pump(L, 65, 15, 0, 80);
        __local_pump(L, 65, 15, 15, 95);
        __local_dual_pump(L, 65, 15, 30, 110);
        __local_pump(L, 70, 15, 0, 85);
        __local_pump(L, 70, 15, 10, 95);
        __local_pump(L, 82, 16, 0, 98);
        __local_pump(L, 78, 15, 8, 100);
        __pump_red_epi8(L, 1, 100, 90, 20, 1, 0, 0, 0, 0);

        return 1;
    }
    if (algo == SVPALGO_120T95) {
        SOLVER_PROFILE_DO("120t95 progressive_LLL", __progressive_LLL(L));
        for (long l = 0; l < 5; l++) {
            char label[64];
            snprintf(label, sizeof(label), "120t95 deep40_tour_%ld", l + 1);
            SOLVER_PROFILE_DO(label, __deep40_tour(L));
        }
        for (long l = 0; l < 60; l += 15) {
            char label[64];
            snprintf(label, sizeof(label), "120t95 local_pump_60_%ld_%ld", l, l + 75);
            SOLVER_PROFILE_DO(label, __local_pump(L, 60, 15, l, l + 75));
        }
        SOLVER_PROFILE_DO("120t95 LLL_after_pump60", L->LLL_QP());
        for (long l = 0; l < 50; l += 10) {
            char label[64];
            snprintf(label, sizeof(label), "120t95 local_pump_65_%ld_%ld", l, l + 80);
            SOLVER_PROFILE_DO(label, __local_pump(L, 65, 15, l, l + 80));
        }
        SOLVER_PROFILE_DO("120t95 LLL_after_pump65", L->LLL_QP());
        for (long l = 0; l < 40; l += 10) {
            char label[64];
            snprintf(label, sizeof(label), "120t95 local_pump_75_%ld_%ld", l, l + 90);
            SOLVER_PROFILE_DO(label, __local_pump(L, 75, 15, l, l + 90));
        }
        SOLVER_PROFILE_DO("120t95 LLL_after_pump75", L->LLL_QP());
        SOLVER_PROFILE_DO("120t95 lsh_pump_85_35_q02", __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 85, 35, 24, 0, 24, 0, 0, 45));
        SOLVER_PROFILE_DO("120t95 local_lsh_80_10_120", __local_lsh_pump(L, 80, 30, 10, 120));
        SOLVER_PROFILE_DO("120t95 local_lsh_75_20_120", __local_lsh_pump(L, 75, 25, 20, 120));
        SOLVER_PROFILE_DO("120t95 lsh_pump_88_32_q04", __lsh_pump_red_epi8(L, num_threads, 1.1, 0.4, 88, 32, 24, 0, 24, 0, 0, 45));
        SOLVER_PROFILE_DO("120t95 local_lsh_82_13_120", __local_lsh_pump(L, 82, 25, 13, 120));
        SOLVER_PROFILE_DO("120t95 local_lsh_75_20_120_b", __local_lsh_pump(L, 75, 25, 20, 120));
        SOLVER_PROFILE_DO("120t95 lsh_pump_90_30_q035", __lsh_pump_red_epi8(L, num_threads, 1.1, 0.35, 90, 30, 24, 0, 24, 0, 0, 45));
        SOLVER_PROFILE_DO("120t95 local_pump_85_15_120", __local_pump(L, 85, 20, 15, 120));
        SOLVER_PROFILE_DO("120t95 lsh_pump_92_28_q035", __lsh_pump_red_epi8(L, num_threads, 1.1, 0.35, 92, 28, 24, 0, 24, 0, 0, 45));
        SOLVER_PROFILE_DO("120t95 local_pump_85_15_120_b", __local_pump(L, 85, 20, 15, 120));
        // the final sieve do not insert to the basis, it only print the short vectors
        // now we run this part manually
        // __last_lsh_pump_epi8(L, num_threads, 0.6, 0.6, 102, 0, 3, 0, 40);

        return 1;
    }
    if (algo == SVPALGO_130T98) {
        __progressive_LLL(L);
        for (long l = 0; l < 5; l++) __deep40_tour(L);
        for (long l = 0; l < 60; l += 15) { __local_pump(L, 60, 15, l, l + 75); L->LLL_QP(); }
        for (long l = 0; l < 60; l += 10) { __local_pump(L, 65, 15, l, l + 80); L->LLL_QP(); }
        __pump_red_epi8(L, num_threads, 1.1, 80, 50, 30, 0, 30, 0, 0, 45);
        __pump_red_epi8(L, num_threads, 1.1, 80, 50, 30, 0, 30, 0, 0, 45);
        __local_lsh_pump(L, 75, 30, 25, 130);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 83, 47, 24, 0, 24, 0, 0, 45);
        __local_pump(L, 80, 30, 20, 130);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 86, 44, 24, 0, 24, 0, 0, 45);
        __local_pump(L, 83, 27, 20, 130);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 89, 41, 24, 0, 24, 0, 0, 45);
        __local_lsh_pump(L, 87, 33, 10, 130);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 92, 38, 24, 0, 24, 0, 0, 45);
        __local_lsh_pump(L, 89, 31, 10, 130);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 95, 35, 24, 0, 24, 0, 0, 45);
        __local_lsh_pump(L, 92, 28, 10, 130);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 98, 32, 24, 0, 24, 0, 0, 45);
        __local_lsh_pump(L, 95, 27, 8, 130);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 101, 29, 24, 0, 24, 0, 0, 45);
        __local_lsh_pump(L, 95, 27, 8, 130);

        // the final sieve do not insert to the basis, it only print the short vectors
        // now we run this part manually
        // __last_lsh_pump_epi8(L, num_threads, 0.4, 0.4, 112, 0, 3, 0, 40);

        return 1;
    }
    if (algo == SVPALGO_140T100) {
        __progressive_LLL(L);
        for (long l = 0; l < 5; l++) __deep40_tour(L);
        for (long l = 0; l < 60; l += 15) { __local_pump(L, 60, 15, l, l + 75); L->LLL_QP(); }
        for (long l = 0; l < 70; l += 10) { __local_pump(L, 65, 15, l, l + 80); L->LLL_QP(); }
        __pump_red_epi8(L, num_threads, 1.1, 80, 60, 40, 0, 40, 0, 0, 45);
        __local_pump(L, 72, 28, 40, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 85, 55, 40, 0, 40, 0, 0, 45);
        __local_pump(L, 79, 31, 30, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 88, 52, 40, 0, 40, 0, 0, 45);
        __local_pump(L, 82, 30, 28, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 91, 49, 40, 0, 40, 0, 0, 45);
        __local_pump(L, 85, 30, 25, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 94, 46, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 88, 32, 20, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 97, 43, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 90, 31, 19, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 100, 40, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 92, 31, 17, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 103, 37, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 95, 28, 17, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 106, 34, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 96, 28, 16, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 109, 31, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 98, 30, 12, 140);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 111, 29, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 100, 30, 10, 140);

        // the final sieve do not insert to the basis, it only print the short vectors
        // now we run this part manually
        // __last_lsh_pump_epi8(L, num_threads, 0.0, 0.0, 122, 0, 0, 0, 40);
        return 1;
    }
    if (algo == SVPALGO_150T102) {
        __progressive_LLL(L);
        for (long l = 0; l < 5; l++) __deep40_tour(L);
        for (long l = 0; l < 60; l += 15) { __local_pump(L, 60, 15, l, l + 75); L->LLL_QP(); }
        for (long l = 0; l < 70; l += 10) { __local_pump(L, 65, 15, l, l + 80); L->LLL_QP(); }  // 90s in total
        __pump_red_epi8(L, num_threads, 1.1, 80, 70, 40, 0, 40, 0, 0, 45);  // 55s
        __pump_red_epi8(L, num_threads, 1.1, 90, 60, 40, 0, 40, 0, 0, 45);  // 460s
        __local_pump(L, 80, 30, 40, 150);  // 48s
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 90, 60, 40, 0, 40, 0, 0, 45);
        __local_pump(L, 85, 35, 30, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 93, 57, 40, 0, 40, 0, 0, 45);
        __local_pump(L, 87, 33, 30, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 96, 54, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 90, 35, 25, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 99, 51, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 92, 34, 24, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 102, 48, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 95, 33, 22, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 105, 45, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 97, 33, 20, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 108, 42, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 100, 31, 19, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 111, 39, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 103, 30, 17, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 114, 36, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 106, 29, 15, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 117, 33, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 109, 27, 14, 150);
        __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 119, 30, 40, 0, 40, 0, 0, 45);
        __local_lsh_pump(L, 111, 27, 12, 150);

        // the final sieve do not insert to the basis, it only print the short vectors
        // now we run this part manually
        //__last_lsh_pump_epi8(L, num_threads, 0.2, 0.2, 128, 0, 0, 0, 40);
        return 1;
    }
    printf("Unknown algorithm: %ld, nothing done\n", algo);
    return 0;
}

int show_help(int argc, char** argv) {
    printf("Usage: %s <lattice_file> [--algo algorithm] [--threads num_threads] [--goal goal_length] [--cuda] [--profile] [--bkz-pre msd d4f]\n", argv[0]);
    printf("./red/<lattice_file> will be read and the reducted basis will be written to ./red/<lattice_file>r\n");
    printf("Options:\n");
    printf("   -s, --seed          sampler seed\n");
    printf("   -t, --threads       number of CPU threads\n");
    printf("   -g, --goal          print possible sol only below this length\n");
    printf("   -cuda, --cuda       use CUDA-assisted BGJ sieve phases\n");
    printf("   -profile, --profile print coarse stage timings\n");
    printf("   -bkz-pre, --bkz-pre run one pump-BKZ preprocessing tour with msd and d4f\n");
    printf("Supported algorithms:\n");
    printf( "   SVPALGO_NULL        0\n"
            "   SVPALGO_100T90      1\n"
            "   SVPALGO_110T90      2\n"
            "   SVPALGO_120T95      3\n"
            "   SVPALGO_130T98      4\n"
            "   SVPALGO_140T100     5\n"
            "   SVPALGO_150T102     6\n");
    return 1;
}

int main(int argc, char** argv) {
    long help = 0;
    long algo = 0;
    long filename_place = 0;
    long seed = time(NULL);
    long cuda = 0;
    long bkz_pre = 0;
    long bkz_pre_msd = 0;
    long bkz_pre_d4f = 0;
    long bkz_pre_minsd = 45;
    double goal_length = 0.0;

    for (long i = 0; i < argc; i++) {
        if (!strcasecmp(argv[i], "-h") || !strcasecmp(argv[i], "--help")) help = 1;
        if (!strcasecmp(argv[i], "-a") || !strcasecmp(argv[i], "--algo")) {
            if (i < argc - 1) {
                i++;
                if (!strcasecmp(argv[i], "100t90") || !strcasecmp(argv[i], "1")) {
                    algo = SVPALGO_100T90;
                    continue;
                }
                if (!strcasecmp(argv[i], "110t90") || !strcasecmp(argv[i], "2")) {
                    algo = SVPALGO_110T90;
                    continue;
                }
                if (!strcasecmp(argv[i], "120t95") || !strcasecmp(argv[i], "3")) {
                    algo = SVPALGO_120T95;
                    continue;
                }
                if (!strcasecmp(argv[i], "130t98") || !strcasecmp(argv[i], "4")) {
                    algo = SVPALGO_130T98;
                    continue;
                }
                if (!strcasecmp(argv[i], "140t100") || !strcasecmp(argv[i], "5")) {
                    algo = SVPALGO_140T100;
                    continue;
                }
                if (!strcasecmp(argv[i], "150t102") || !strcasecmp(argv[i], "6")) {
                    algo = SVPALGO_150T102;
                    continue;
                }
                printf("Unknown algorithm: %s\n", argv[i]);
                help = 1;
            }
        }
        if (!strcasecmp(argv[i], "-cuda") || !strcasecmp(argv[i], "--cuda")) {
            cuda = 1;
            continue;
        }
        if (!strcasecmp(argv[i], "-profile") || !strcasecmp(argv[i], "--profile")) {
            profile = 1;
            continue;
        }
        if (!strcasecmp(argv[i], "-bkz-pre") || !strcasecmp(argv[i], "--bkz-pre")) {
            if (i < argc - 2) {
                bkz_pre = 1;
                bkz_pre_msd = atol(argv[++i]);
                bkz_pre_d4f = atol(argv[++i]);
                continue;
            }
            help = 1;
        }
        if (!strcasecmp(argv[i], "--bkz-pre-minsd")) {
            if (i < argc - 1) {
                bkz_pre_minsd = atol(argv[++i]);
                continue;
            }
            help = 1;
        }
        if (!strcasecmp(argv[i], "-g") || !strcasecmp(argv[i], "--goal")) {
            if (i < argc - 1) {
                i++;
                goal_length = atof(argv[i]);
                continue;
            }
        }
        if (!strcasecmp(argv[i], "-t") || !strcasecmp(argv[i], "--threads")) {
            if (i < argc - 1) {
                i++;
                num_threads = atol(argv[i]);
                continue;
            }
        }
        if (!strcasecmp(argv[i], "-s") || !strcasecmp(argv[i], "--seed")) {
            if (i < argc - 1) {
                i++;
                seed = atol(argv[i]);
                continue;
            }
        }
        if (filename_place == 0) filename_place = i;
    }

    if (help || argc < 3 || filename_place == 0) {
        show_help(argc, argv);
        return 0;
    }

    if (cuda) {
        #if !defined(HAVE_CUDA)
        fprintf(stderr, "Error: --cuda requires a CUDA build.\n");
        return 2;
        #else
        setenv("BGJ_SVP_CUDA", "1", 1);
        #endif
    }
    if (profile) {
        setenv("BGJ_PUMP_PROFILE", "1", 1);
    }

    char input[256];
    char output[256];
    snprintf(input, sizeof(input), "./raw/%s", argv[filename_place]);
    snprintf(output, sizeof(output), "./red/%sr", argv[filename_place]);

    TIMER_START;
    double profile_t0 = solver_now();
    NTL::Mat<NTL::ZZ> L_ZZ;
    std::ifstream data(input, std::ios::in);
    data >> L_ZZ;
    solver_profile_line("read_raw_basis", solver_now() - profile_t0);
    profile_t0 = solver_now();
    if (solver_use_fplll_initial_lll()) {
        int fplll_status = bgj_fplll_lll(L_ZZ, 1.0 / 3.0, 0);
        solver_profile_line(fplll_status == 0 ? "initial_fplll_LLL" : "initial_fplll_LLL_failed",
                            solver_now() - profile_t0);
        if (fplll_status != 0) {
            fprintf(stderr, "Warning: fplll initial LLL failed (%s); falling back to NTL LLL.\n",
                    bgj_fplll_status_string(fplll_status));
            profile_t0 = solver_now();
            NTL::ZZ det2;
            LLL(det2, L_ZZ, 1, 3, 0);
            solver_profile_line("initial_NTL_LLL", solver_now() - profile_t0);
        }
    } else {
        NTL::ZZ det2;
        LLL(det2, L_ZZ, 1, 3, 0);
        solver_profile_line("initial_NTL_LLL", solver_now() - profile_t0);
    }

    profile_t0 = solver_now();
    Lattice_QP L(L_ZZ);
    solver_profile_line("lattice_init", solver_now() - profile_t0);
    SetSamplerSeed((uint64_t)seed);
    if (bkz_pre) {
        if (bkz_pre_msd <= 0 || bkz_pre_d4f < 0 || bkz_pre_msd + bkz_pre_d4f >= L.NumRows()) {
            fprintf(stderr, "Error: --bkz-pre requires msd > 0, d4f >= 0, and msd + d4f < dimension.\n");
            return 2;
        }
        profile_t0 = solver_now();
        L.BKZ_tour_pump_epi8(bkz_pre_msd, bkz_pre_d4f, num_threads, 0, L.NumRows(), profile ? 1 : 0, bkz_pre_minsd);
        solver_profile_line("bkz_pre_total", solver_now() - profile_t0);
    }
    profile_t0 = solver_now();
    _svp_solver_red(&L, algo);
    solver_profile_line("solver_red_total", solver_now() - profile_t0);
    TIMER_END;
    
    double min_len = sqrt(dot_avx2(L.get_b().hi[0], L.get_b().hi[0], L.NumRows()));
    if (goal_length == 0.0) goal_length = L.gh() * 1.00;
    if (min_len < goal_length) {
        printf("possible sol: %s, length = %.2f, time = %.2fs, vec = ", argv[filename_place], min_len, CURRENT_TIME);
        PRINT_VEC(L.get_b().hi[0], L.NumRows());
    }
    profile_t0 = solver_now();
    L.store(output);
    solver_profile_line("store_reduced_basis", solver_now() - profile_t0);


    return 0;
}
