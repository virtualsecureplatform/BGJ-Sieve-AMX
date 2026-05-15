#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <strings.h>
#include <vector>
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
static Lattice_QP *solver_profile_lattice_current = NULL;

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

static int solver_env_is_set(const char *name)
{
    const char *env = getenv(name);
    return env != NULL && env[0] != '\0';
}

static long solver_env_long(const char *name, long default_value)
{
    const char *env = getenv(name);
    if (env == NULL || env[0] == '\0') return default_value;
    char *end = NULL;
    long value = strtol(env, &end, 10);
    return (end != env) ? value : default_value;
}

static double solver_env_double(const char *name, double default_value)
{
    const char *env = getenv(name);
    if (env == NULL || env[0] == '\0') return default_value;
    char *end = NULL;
    double value = strtod(env, &end);
    return (end != env) ? value : default_value;
}

static int solver_env_flag(const char *name, int default_value)
{
    const char *env = getenv(name);
    if (env == NULL || env[0] == '\0') return default_value;
    if (!strcasecmp(env, "0") || !strcasecmp(env, "false") ||
        !strcasecmp(env, "no") || !strcasecmp(env, "off")) return 0;
    return 1;
}

struct solver_best_row_t {
    long index;
    double length;
};

struct solver_best_candidate_t {
    double length;
    const char *source;
    long basis_index;
    double basis_length;
    double lsh_length;
    int has_lsh;
};

static solver_best_row_t solver_find_best_basis_row(Lattice_QP *L)
{
    solver_best_row_t best;
    best.index = 0;
    best.length = 0.0;
    for (long i = 0; i < L->NumRows(); ++i) {
        const double len = sqrt(dot_avx2(L->get_b().hi[i], L->get_b().hi[i], L->NumCols()));
        if (i == 0 || len < best.length) {
            best.index = i;
            best.length = len;
        }
    }
    return best;
}

static solver_best_candidate_t solver_find_best_candidate(Lattice_QP *L)
{
    const solver_best_row_t basis = solver_find_best_basis_row(L);
    solver_best_candidate_t best;
    best.length = basis.length;
    best.source = "basis";
    best.basis_index = basis.index;
    best.basis_length = basis.length;
    best.lsh_length = 0.0;
    best.has_lsh = 0;

    long lsh_dim = 0;
    double lsh_length = 0.0;
    if (bgj_lsh_best_solution_get(&lsh_length, NULL, L->NumCols(), &lsh_dim) &&
        lsh_dim == L->NumCols() && lsh_length > 0.0) {
        best.has_lsh = 1;
        best.lsh_length = lsh_length;
        if (lsh_length < best.length) {
            best.length = lsh_length;
            best.source = "lsh_best";
        }
    }
    return best;
}

static int solver_env_flag_value(const char *env, int default_value)
{
    if (env == NULL || env[0] == '\0') return default_value;
    if (!strcasecmp(env, "0") || !strcasecmp(env, "false") ||
        !strcasecmp(env, "no") || !strcasecmp(env, "off")) return 0;
    return 1;
}

class solver_scoped_env_override_t {
public:
    solver_scoped_env_override_t(const char *name, const char *value)
        : name_(name), had_old_(0), old_value_()
    {
        const char *old = getenv(name_);
        if (old != NULL) {
            had_old_ = 1;
            old_value_ = old;
        }
        setenv(name_, value, 1);
    }

    ~solver_scoped_env_override_t()
    {
        if (had_old_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char *name_;
    int had_old_;
    std::string old_value_;
};

static int solver_svp120_final_lsh_single_gpu()
{
    return solver_env_flag_value(getenv("BGJ_120T95_FINAL_LSH_SINGLE_GPU"), 0);
}

static long solver_svp120_final_lsh_threads()
{
    if (solver_env_is_set("BGJ_120T95_FINAL_LSH_THREADS")) {
        long value = solver_env_long("BGJ_120T95_FINAL_LSH_THREADS", num_threads);
        if (value > 0) return value;
    }
    if (num_threads > 8) return 8;
    return num_threads;
}

static int solver_svp120_final_lsh_mode()
{
    const char *env = getenv("BGJ_SVP_FINAL_LSH");
    if (env == NULL || env[0] == '\0') return 2;
    if (!strcasecmp(env, "auto")) return 2;
    if (!strcasecmp(env, "default")) return 2;
    if (!strcasecmp(env, "force")) return 1;
    if (!strcasecmp(env, "always")) return 1;
    return solver_env_flag_value(env, 0);
}

static double solver_svp120_final_lsh_target(Lattice_QP *L)
{
    double target = solver_env_double("BGJ_120T95_FINAL_LSH_TARGET", 0.0);
    if (target <= 0.0) {
        const double target_factor = solver_env_double("BGJ_120T95_FINAL_LSH_TARGET_FACTOR", 0.97);
        target = L->gh() * target_factor;
    }
    return target;
}

static int solver_svp120_should_run_final_lsh(Lattice_QP *L, double *target_out)
{
    const int mode = solver_svp120_final_lsh_mode();
    const double target = solver_svp120_final_lsh_target(L);
    if (target_out) *target_out = target;
    if (mode == 0) {
        if (profile) {
            printf("final_lsh_policy: algo=120t95 mode=off final_lsh=skip\n");
            fflush(stdout);
        }
        return 0;
    }
    if (mode == 1) {
        if (profile) {
            printf("final_lsh_policy: algo=120t95 mode=force target=%.9g final_lsh=run\n", target);
            fflush(stdout);
        }
        return 1;
    }

    const solver_best_candidate_t best = solver_find_best_candidate(L);
    const int run = best.length > target;
    if (profile) {
        if (best.has_lsh) {
            printf("final_lsh_policy: algo=120t95 mode=auto basis_best=%.9g lsh_best=%.9g current_best=%.9g source=%s target=%.9g final_lsh=%s\n",
                   best.basis_length, best.lsh_length, best.length, best.source,
                   target, run ? "run" : "skip");
        } else {
            printf("final_lsh_policy: algo=120t95 mode=auto basis_best=%.9g lsh_best=none current_best=%.9g source=%s target=%.9g final_lsh=%s\n",
                   best.basis_length, best.length, best.source,
                   target, run ? "run" : "skip");
        }
        fflush(stdout);
    }
    return run;
}

static int solver_svp120_quality_satisfied(Lattice_QP *L, double target, solver_best_candidate_t *candidate_out)
{
    const solver_best_candidate_t candidate = solver_find_best_candidate(L);
    if (candidate_out) *candidate_out = candidate;
    return target > 0.0 && candidate.length > 0.0 && candidate.length <= target;
}

static int solver_profile_best_enabled()
{
    return solver_env_flag("BGJ_SOLVER_PROFILE_BEST", 0);
}

static void solver_profile_best_line(const char *stage)
{
    if (!profile || !solver_profile_best_enabled() || solver_profile_lattice_current == NULL) return;
    solver_best_candidate_t best = solver_find_best_candidate(solver_profile_lattice_current);
    if (best.has_lsh) {
        printf("solver_stage_best: stage=\"%s\" basis_best=%.9g lsh_best=%.9g current_best=%.9g source=%s row=%ld\n",
               stage, best.basis_length, best.lsh_length, best.length,
               best.source, best.basis_index);
    } else {
        printf("solver_stage_best: stage=\"%s\" basis_best=%.9g lsh_best=none current_best=%.9g source=%s row=%ld\n",
               stage, best.basis_length, best.length, best.source,
               best.basis_index);
    }
    fflush(stdout);
}

#define SOLVER_PROFILE_DO(label, stmt) do {               \
        double __solver_profile_t0 = solver_now();        \
        stmt;                                             \
        solver_profile_line(label, solver_now() - __solver_profile_t0); \
        solver_profile_best_line(label);                  \
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
    solver_profile_lattice_current = L;
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
        if (solver_env_is_set("BGJ_120T95_FIRST_LSH_QRATIO") ||
            solver_env_is_set("BGJ_120T95_FIRST_LSH_MSD") ||
            solver_env_is_set("BGJ_120T95_FIRST_LSH_F") ||
            solver_env_is_set("BGJ_120T95_FIRST_LSH_MINSD")) {
            const double first_lsh_qratio = solver_env_double("BGJ_120T95_FIRST_LSH_QRATIO", 0.2);
            const long first_lsh_msd = solver_env_long("BGJ_120T95_FIRST_LSH_MSD", 85);
            const long first_lsh_f = solver_env_long("BGJ_120T95_FIRST_LSH_F",
                                                     solver_env_long("BGJ_120T95_FIRST_LSH_MINSD", 35));
            char first_lsh_label[96];
            snprintf(first_lsh_label, sizeof(first_lsh_label), "120t95 lsh_pump_env_%ld_%ld_q%.3f",
                     first_lsh_msd, first_lsh_f, first_lsh_qratio);
            SOLVER_PROFILE_DO(first_lsh_label, __lsh_pump_red_epi8(L, num_threads, 1.1, first_lsh_qratio, first_lsh_msd, first_lsh_f, 24, 0, 24, 0, 0, 45));
        } else {
            SOLVER_PROFILE_DO("120t95 lsh_pump_85_35_q02", __lsh_pump_red_epi8(L, num_threads, 1.1, 0.2, 85, 35, 24, 0, 24, 0, 0, 45));
        }
        SOLVER_PROFILE_DO("120t95 local_lsh_80_10_120", __local_lsh_pump(L, 80, 30, 10, 120));
        SOLVER_PROFILE_DO("120t95 local_lsh_75_20_120", __local_lsh_pump(L, 75, 25, 20, 120));
        SOLVER_PROFILE_DO("120t95 lsh_pump_88_32_q04", __lsh_pump_red_epi8(L, num_threads, 1.1, 0.4, 88, 32, 24, 0, 24, 0, 0, 45));
        SOLVER_PROFILE_DO("120t95 local_lsh_82_13_120", __local_lsh_pump(L, 82, 25, 13, 120));
        SOLVER_PROFILE_DO("120t95 local_lsh_75_20_120_b", __local_lsh_pump(L, 75, 25, 20, 120));
        SOLVER_PROFILE_DO("120t95 lsh_pump_90_30_q035", __lsh_pump_red_epi8(L, num_threads, 1.1, 0.35, 90, 30, 24, 0, 24, 0, 0, 45));
        SOLVER_PROFILE_DO("120t95 local_pump_85_15_120", __local_pump(L, 85, 20, 15, 120));
        double final_lsh_target = solver_svp120_final_lsh_target(L);
        const int final_lsh_mode = solver_svp120_final_lsh_mode();
        const double lsh_pump_stop_length = (final_lsh_mode != 0) ?
            solver_env_double("BGJ_120T95_LSH_PUMP_STOP_LENGTH", final_lsh_target) : 0.0;
        SOLVER_PROFILE_DO("120t95 lsh_pump_92_28_q035",
                          __lsh_pump_red_epi8(L, num_threads, 1.1, 0.35, 92, 28, 24, 0, 24, 0, 0, 45,
                                              lsh_pump_stop_length));
        solver_best_candidate_t quality_candidate;
        long retry_count = 0;
        int final_lsh_decided = 0;
        int run_final_lsh = 0;
        if (final_lsh_mode == 2 && solver_svp120_quality_satisfied(L, final_lsh_target, &quality_candidate)) {
            if (profile) {
                printf("quality_stop: algo=120t95 after=lsh_pump_92 current_best=%.9g source=%s target=%.9g skip_local_pump_b=1\n",
                       quality_candidate.length, quality_candidate.source, final_lsh_target);
                fflush(stdout);
            }
        } else {
            retry_count = (final_lsh_mode == 2) ? solver_env_long("BGJ_120T95_LSH_RETRY", 0) : 0;
            const long retry_msd = solver_env_long("BGJ_120T95_LSH_RETRY_MSD", 92);
            const long retry_f = solver_env_long("BGJ_120T95_LSH_RETRY_F", 28);
            const long retry_ni = solver_env_long("BGJ_120T95_LSH_RETRY_NI", 1);
            const long retry_ne = solver_env_long("BGJ_120T95_LSH_RETRY_NE", 0);
            const long retry_ns = solver_env_long("BGJ_120T95_LSH_RETRY_NS", 0);
            const long retry_minsd = solver_env_long("BGJ_120T95_LSH_RETRY_MINSD", 45);
            const double retry_qratio = solver_env_double("BGJ_120T95_LSH_RETRY_QRATIO", 0.35);
            for (long retry = 0; retry < retry_count; ++retry) {
                char retry_label[128];
                snprintf(retry_label, sizeof(retry_label),
                         "120t95 lsh_pump_92_retry_%ld", retry + 1);
                SOLVER_PROFILE_DO(retry_label,
                                  __lsh_pump_red_epi8(L, num_threads, 1.1, retry_qratio,
                                                      retry_msd, retry_f, retry_ni, retry_ne,
                                                      retry_ns, 0, 0, retry_minsd,
                                                      lsh_pump_stop_length));
                if (solver_svp120_quality_satisfied(L, final_lsh_target, &quality_candidate)) {
                    if (profile) {
                        printf("quality_stop: algo=120t95 after=lsh_pump_92_retry_%ld current_best=%.9g source=%s target=%.9g skip_local_pump_b=1\n",
                               retry + 1, quality_candidate.length,
                               quality_candidate.source, final_lsh_target);
                        fflush(stdout);
                    }
                    break;
                }
            }
            if (!solver_svp120_quality_satisfied(L, final_lsh_target, &quality_candidate)) {
                const int keep_local_b = solver_env_long("BGJ_120T95_LOCAL_PUMP_B_BEFORE_RESCUE", 1) != 0;
                if (final_lsh_mode == 2 && !keep_local_b) {
                    run_final_lsh = solver_svp120_should_run_final_lsh(L, &final_lsh_target);
                    final_lsh_decided = 1;
                    if (run_final_lsh && profile) {
                        printf("local_pump_skip: algo=120t95 after=lsh_pump_92 current_best=%.9g source=%s target=%.9g skip_local_pump_b=1 reason=final_lsh_rescue\n",
                               quality_candidate.length, quality_candidate.source, final_lsh_target);
                        fflush(stdout);
                    }
                }
                if (!run_final_lsh) {
                    SOLVER_PROFILE_DO("120t95 local_pump_85_15_120_b", __local_pump(L, 85, 20, 15, 120));
                }
            }
        }
        if (!final_lsh_decided) {
            run_final_lsh = solver_svp120_should_run_final_lsh(L, &final_lsh_target);
        }
        if (run_final_lsh) {
            const long final_threads = solver_svp120_final_lsh_threads();
            const double final_qratio = solver_env_double("BGJ_120T95_FINAL_LSH_QRATIO", 0.6);
            const double final_ext_qratio = solver_env_double("BGJ_120T95_FINAL_LSH_EXT_QRATIO", final_qratio);
            const long final_msd = solver_env_long("BGJ_120T95_FINAL_LSH_MSD", 102);
            const long final_ext_d = solver_env_long("BGJ_120T95_FINAL_LSH_EXT_D", 0);
            const long final_log = solver_env_long("BGJ_120T95_FINAL_LSH_LOG", 3);
            const long final_shuffle = solver_env_long("BGJ_120T95_FINAL_LSH_SHUFFLE_FIRST", 0);
            const long final_minsd = solver_env_long("BGJ_120T95_FINAL_LSH_MINSD", 40);
            const long final_lift_margin = solver_env_long("BGJ_120T95_FINAL_LSH_LIFT_MARGIN", 12);
            const long final_lift_start_default = final_msd - final_lift_margin;
            const long final_lift_start = solver_env_long("BGJ_120T95_FINAL_LSH_LIFT_START_CSD",
                                                          final_lift_start_default);
            const double final_stop_length = solver_env_double("BGJ_120T95_FINAL_LSH_STOP_LENGTH",
                                                               final_lsh_target);
            double final_lsh_length = 0.0;
            char final_lsh_label[128];
            snprintf(final_lsh_label, sizeof(final_lsh_label),
                     "120t95 final_lsh_%ld_%ld_q%.3f",
                     final_msd, final_ext_d, final_qratio);
            if (profile && final_threads != num_threads) {
                printf("final_lsh_thread_policy: algo=120t95 solver_threads=%ld final_lsh_threads=%ld\n",
                       num_threads, final_threads);
                fflush(stdout);
            }
            if (solver_svp120_final_lsh_single_gpu()) {
                if (profile) {
                    printf("final_lsh_cuda_policy: algo=120t95 primary_thread_device=1 multi_gpu_batch=0\n");
                    fflush(stdout);
                }
                solver_scoped_env_override_t primary_device("BGJ_CUDA_PRIMARY_THREAD_DEVICE", "1");
                solver_scoped_env_override_t no_multi_batch("BGJ_CUDA_MULTI_GPU_BATCH", "0");
                solver_scoped_env_override_t no_batch_split("BGJ_CUDA_MULTI_GPU_BATCH_SPLIT", "0");
                SOLVER_PROFILE_DO(final_lsh_label,
                                  final_lsh_length = __last_lsh_pump_epi8(L, final_threads,
                                                                           final_qratio,
                                                                           final_ext_qratio,
                                                                           final_msd,
                                                                           final_ext_d,
                                                                           final_log,
                                                                           final_shuffle,
                                                                           final_minsd,
                                                                           final_stop_length,
                                                                           final_lift_start));
            } else {
                if (profile) {
                    printf("final_lsh_cuda_policy: algo=120t95 inherited\n");
                    fflush(stdout);
                }
                SOLVER_PROFILE_DO(final_lsh_label,
                                  final_lsh_length = __last_lsh_pump_epi8(L, final_threads,
                                                                           final_qratio,
                                                                           final_ext_qratio,
                                                                           final_msd,
                                                                           final_ext_d,
                                                                           final_log,
                                                                           final_shuffle,
                                                                           final_minsd,
                                                                           final_stop_length,
                                                                           final_lift_start));
            }
            if (final_lsh_length > 0.0) {
                printf("final_lsh_best: algo=120t95 length=%.9g\n", final_lsh_length);
            }
        }

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
        if (!solver_env_is_set("BGJ_CUDA_LSH_SEARCH")) {
            setenv("BGJ_CUDA_LSH_SEARCH", "1", 0);
        }
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
    bgj_lsh_best_solution_reset();
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
    
    const solver_best_row_t best_row = solver_find_best_basis_row(&L);
    std::vector<double> lsh_best_vec((size_t)L.NumCols());
    double lsh_best_len = 0.0;
    long lsh_best_dim = 0;
    const int has_lsh_best = bgj_lsh_best_solution_get(&lsh_best_len,
                                                       lsh_best_vec.data(),
                                                       L.NumCols(),
                                                       &lsh_best_dim) &&
                             lsh_best_dim == L.NumCols();
    double min_len = best_row.length;
    const int use_lsh_best = has_lsh_best && lsh_best_len < min_len;
    if (use_lsh_best) min_len = lsh_best_len;
    if (goal_length == 0.0) goal_length = L.gh() * 1.00;
    if (min_len < goal_length) {
        if (use_lsh_best) {
            printf("possible sol: %s, source = lsh_best, length = %.2f, time = %.2fs, vec = ",
                   argv[filename_place], min_len, CURRENT_TIME);
            PRINT_VEC(lsh_best_vec.data(), lsh_best_dim);
        } else {
            printf("possible sol: %s, row = %ld, length = %.2f, time = %.2fs, vec = ",
                   argv[filename_place], best_row.index, min_len, CURRENT_TIME);
            PRINT_VEC(L.get_b().hi[best_row.index], L.NumCols());
        }
    }
    profile_t0 = solver_now();
    L.store(output);
    solver_profile_line("store_reduced_basis", solver_now() - profile_t0);


    return 0;
}
