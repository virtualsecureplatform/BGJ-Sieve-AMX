#include "../include/pool_epi8.h"
#include "../include/fplll_bridge.h"

#if defined(__AMX_INT8__)
#include "../include/bgj_amx.h"
#endif

#include <cmath>
#include <cstdlib>
#include <sys/time.h>

#define POOL_EPI8_RATIO_ADJ ( CSD < 80 ? pow(1.01, CSD - 80) : 1.0)

static double bgj_uid_reserve_factor() {
    const char *env = getenv("BGJ_UID_RESERVE_FACTOR");
    if (env) {
        const double parsed = atof(env);
        if (parsed > 0.0) return parsed;
    }
#if USE_SPARSEPP
    return 8.0;
#else
    return 1.5;
#endif
}

static long bgj_uid_reserve_max() {
    const char *env = getenv("BGJ_UID_RESERVE_MAX");
    if (env) return atol(env);
    return 268435456L;
}

static long bgj_uid_reserve_total_hint(long pool_size) {
    const double reserve = (double)pool_size * bgj_uid_reserve_factor();
    long total_hint = reserve > (double)pool_size ? (long)reserve : pool_size;
    const long max_hint = bgj_uid_reserve_max();
    if (max_hint > 0 && total_hint > max_hint) total_hint = max_hint;
    return total_hint;
}

static double pool_epi8_wall_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int pool_epi8_lift_profile_enabled()
{
    const char *env = getenv("BGJ_LIFT_PROFILE");
    if (env == NULL) env = getenv("BGJ_PUMP_PROFILE");
    return env != NULL && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
}

static int bgj_tail_lll_use_fplll()
{
    if (!bgj_fplll_is_available()) return 0;
    const char *backend = getenv("BGJ_TAIL_LLL_BACKEND");
    if (backend == NULL || backend[0] == '\0') return 1;
    if (!strcasecmp(backend, "fplll") || !strcasecmp(backend, "1") ||
        !strcasecmp(backend, "true") || !strcasecmp(backend, "yes")) return 1;
    return 0;
}

static int bgj_tail_deep_lll_use_custom()
{
    const char *backend = getenv("BGJ_TAIL_DEEP_LLL_BACKEND");
    if (backend == NULL || backend[0] == '\0') return 0;
    if (!strcasecmp(backend, "custom") || !strcasecmp(backend, "1") ||
        !strcasecmp(backend, "true") || !strcasecmp(backend, "yes")) return 1;
    return 0;
}

static void bgj_reserve_uid_table(UidHashTable *uid, long pool_size) {
    if (uid) uid->reserve_total(bgj_uid_reserve_total_hint(pool_size));
}

struct cvec_for_sort_epi8{
    __attribute__ ((packed)) uint16_t _data[3];
};
inline bool cmp_cvec_epi8(cvec_for_sort_epi8 x, cvec_for_sort_epi8 y){
    return (x._data[2] < y._data[2]);
}

////////////////////// construction and distructions //////////////////////

template<uint32_t nb>
Pool_epi8_t<nb>::Pool_epi8_t() {
    _threadpool.resize(num_threads);
}

template<uint32_t nb>
Pool_epi8_t<nb>::Pool_epi8_t(Lattice_QP *L) {
    basis = L;
    if (basis->get_gso_status() == 0) basis->compute_gso_QP();
    _threadpool.resize(num_threads);
}

template<uint32_t nb>
Pool_epi8_t<nb>::~Pool_epi8_t() {
    clear_all();
}

template<uint32_t nb>
void Pool_epi8_t<nb>::clear_all() {
    clear_pool();
    if (_b_local) {
        FREE_MAT((void **)_b_local);
        _b_local = NULL;
    }
    if (_b_dual) {
        FREE_VEC((void *)_b_dual);
        _b_dual = NULL;
    }
    if (uid) {
        delete uid;
        uid = NULL;
    }
    #if defined(__AMX_INT8__) && BOOST_AMX_SIEVE
    if (booster) {
        delete booster;
        booster = NULL;
    }
    #endif
}

template<uint32_t nb>
void Pool_epi8_t<nb>::clear_pool() {
    if (cvec) {
        FREE_VEC((void *)cvec);
        cvec = NULL;
    }
    if (vec) {
        FREE_VEC((void *)vec);
        vec = NULL;
    }
    if (vnorm) {
        FREE_VEC((void *)vnorm);
        vnorm = NULL;
    }
    if (vsum) {
        FREE_VEC((void *)vsum);
        vsum = NULL;
    }
    if (vu) {
        FREE_VEC((void *)vu);
        vu = NULL;
    }
    num_vec = 0;
    num_empty = 0;
    sorted_index = 0;
    _pool_size = 0;
    mark_pool_dirty();
}


////////////////////// setup //////////////////////

template<uint32_t nb>
int Pool_epi8_t<nb>::set_num_threads(long n) {
    num_threads = n;
    _threadpool.resize(num_threads);
    omp_set_num_threads(num_threads);
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::set_max_pool_size(long N) {
    clear_pool();
    _pool_size = N;
    vec = (int8_t *) NEW_VEC(N * vec_length + 32, sizeof(int8_t));
    cvec = (uint16_t *) NEW_VEC(N * 3, sizeof(uint16_t));
    vnorm = (int32_t *) NEW_VEC(N, sizeof(int32_t));
    vsum = (int32_t *) NEW_VEC(N, sizeof(int32_t));
    vu = (uint64_t *) NEW_VEC(N, sizeof(uint64_t));

    if (!vec || !cvec || !vnorm || !vsum || !vu){
        clear_pool();
        return 0;
    }
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::set_basis(Lattice_QP *L) {
    clear_pool();
    basis = L;
    if (basis->get_gso_status() == 0) basis->compute_gso_QP();
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::set_sieving_context(long l, long r) {
    if (uid == NULL){
        uid = new UidHashTable();
        if (uid == NULL) return 0;
    }
    num_vec = 0;
    sorted_index = 0;
    index_l = l;
    index_r = r;
    CSD = r - l;
    mark_pool_dirty();
    _compute_gh2();
    uid->reset_hash_function(CSD);
    bgj_reserve_uid_table(uid, _pool_size);
    for (long i = 0; i < CSD; i++) uid->insert_uid(uid->uid_coeffs[i]);
    _update_b_local();
    return 1;
}


////////////////////// pool operations //////////////////////

template<uint32_t nb>
int Pool_epi8_t<nb>::sampling(long N) {
    if (N > _pool_size){
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::sampling: N is larger than _pool_size, set N = _pool_size instead.\n", nb);
        N = _pool_size;
    }
    if (N < num_vec) {
        fprintf(stderr, "[Info] Pool_epi8_t<%u>::sampling: N is smaller than current pool size, ignored.\n", nb);
        return 0;
    }

    long num_insert = N - num_vec;
    int succ = 1;
    #pragma omp parallel for 
    for (long thread = 0; thread < num_threads; thread++) {
        DGS1d R(thread);
        long begin_ind = num_vec + (thread * num_insert)/num_threads;
        long end_ind = num_vec + ((thread+1) * num_insert)/num_threads;
        for (long ind = begin_ind; ind < end_ind; ind++) {
            int8_t *dst = vec + ind * vec_length;
            uint16_t *cdst = cvec + ind * 3LL;
            if (_sampling(dst, cdst, &vnorm[ind], &vsum[ind], &vu[ind], &R) == -1) {
                succ = 0;
                break;
            }
        }
    }

    // all succ or do nothing
    if (succ) {
        num_empty -= num_insert;
        if (num_empty < 0) num_empty = 0;
        num_vec += num_insert;
        mark_pool_dirty();
        return 0;
    }
    return -1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::shrink(long N) {
    if (N >= num_vec + num_empty) {
        if (N > num_vec + num_empty + 1000){
            fprintf(stderr, "Pool_epi8_t<%u>::shrink: [Warning] N(%ld) >> num_vec(%ld) + num_empty(%ld), nothing done.\n", 
                    nb, N, num_vec, num_empty);
        }
        return 0;
    }
    if (N >= num_vec) {
        num_empty = N - num_vec;
        return 0;
    }
    sort_cvec();

    long tail_ind = N;
    for (long i = 0; i < N; i++){
        uint32_t ind = ((uint32_t *)(cvec + i * 3LL))[0];
        if (ind < N) continue;
        uint32_t dst_ind;
        do{ 
            if (tail_ind >= num_vec) {
                fprintf(stderr, "[Error] Pool_epi8_t<%u>::shrink: shrink failed! something must be wrong, pool clobbered!\n", nb);
                return 0;
            }
            dst_ind = ((uint32_t *)(cvec + tail_ind * 3LL))[0];
            if (!uid->erase_uid(vu[dst_ind])){
                fprintf(stderr, "[Error] Pool_epi8_t<%u>::shrink: erase pool vec uid failed, ignored.\n", nb);
            }
            tail_ind++;
        } while(dst_ind >= N);
        memcpy(vec + dst_ind * vec_length, vec + ind * vec_length, vec_length);
        vu[dst_ind] = vu[ind];
        vsum[dst_ind] = vsum[ind];
        vnorm[dst_ind] = vnorm[ind];
        ((uint32_t *)(cvec + i * 3LL))[0] = dst_ind;
    }
    while (tail_ind < num_vec) {
        uint32_t dst_ind = ((uint32_t *)(cvec + tail_ind * 3LL))[0];
        if (!uid->erase_uid(vu[dst_ind])){
            fprintf(stderr, "[Error] Pool_epi8_t<%u>::shrink: erase pool vec uid failed, ignored.\n", nb);
        }
        tail_ind++;
    }

    memset(vec+N*vec_length, 0, (num_vec-N)*vec_length);

    num_vec = N;
    sorted_index = 0;
    mark_pool_dirty();
    
    
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::extend_left() {
    // be careful when using it!
    #define INIT_COLLISION_LIST                                         \
    uint32_t *collision_list = NULL;                                    \
    uint32_t num_collision = 0;                                         \
    uint32_t _collision_list_size = 0;                                  \
    pthread_spinlock_t collision_list_lock;                             \
    pthread_spin_init(&collision_list_lock, PTHREAD_PROCESS_SHARED)    

    #define ADD_TO_COLLISION_LIST(__ind) do {                                                                           \
        pthread_spin_lock(&collision_list_lock);                                                                        \
        if (num_collision == _collision_list_size) {                                                                    \
            collision_list = (uint32_t *) realloc(collision_list, (2 * _collision_list_size + 64)*sizeof(uint32_t));    \
            _collision_list_size = 2 * _collision_list_size + 64;                                                       \
        }                                                                                                               \
        collision_list[num_collision] = __ind;                                                                          \
        num_collision++;                                                                                                \
        pthread_spin_unlock(&collision_list_lock);                                                                      \
    } while (0)

    #define PROCESS_COLLISION_LIST                                                              \
                                    do {                                                        \
        if (num_collision) {                                                                    \
            parallel_algorithms::sort(collision_list, collision_list+num_collision,_threadpool);\
        }                                                                                       \
        while (num_collision) {                                                                 \
            uint32_t coind = collision_list[num_collision-1];                                   \
                                                                                                \
            if (coind != num_vec - 1) {                                                         \
                vnorm[coind] = vnorm[num_vec-1];                                                \
                vu[coind] = vu[num_vec-1];                                                      \
                vsum[coind] = vsum[num_vec-1];                                                  \
                memcpy(vec + coind*vec_length, vec + (num_vec - 1)*vec_length, vec_length);     \
            }                                                                                   \
            num_vec--;                                                                          \
            num_empty++;                                                                        \
            memset(vec+num_vec*vec_length, 0, vec_length);                                      \
            num_collision--;                                                                    \
        }                                                                                       \
                                                                                                \
        if (collision_list) free(collision_list);                                               \
    } while (0)


    if (index_l == 0) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::extend_left: index_l = 0, cannot extend_left, nothing done.\n", nb);
        return 0;
    }

    int32_t _dhalf_old = _dhalf;
    int32_t _dshift_old = _dshift;
    uint8_t *_b_dual_old = _b_dual;
    _b_dual = NULL;

    INIT_COLLISION_LIST;

    index_l--;
    CSD++;
    _update_b_local();
    _compute_gh2();
    uid->reset_hash_function(CSD);
    bgj_reserve_uid_table(uid, _pool_size);
    for (long i = 0; i < CSD; i++) uid->insert_uid(uid->uid_coeffs[i]);

    // compute new vec from old vec
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        __attribute__ ((aligned (32))) float ftmp[vec_length * 8];
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        float Bis0 = 1.0 / _b_local[0][0];

        long ind = begin_ind;
        while (ind < end_ind - 7) {
            // recover old coeff
            _compute_coeff_b8(ctmp, ind, CSD-1, _b_dual_old, _dhalf_old, _dshift_old);
            
            // compute u
            uint64_t u[8];
            for (long i = 0; i < 8; i++) u[i] = 0;
            for (long i = 1; i < CSD; i++) {
                u[0] += ctmp[(i-1)*8+0]*uid->uid_coeffs[i];
                u[1] += ctmp[(i-1)*8+1]*uid->uid_coeffs[i];
                u[2] += ctmp[(i-1)*8+2]*uid->uid_coeffs[i];
                u[3] += ctmp[(i-1)*8+3]*uid->uid_coeffs[i];
                u[4] += ctmp[(i-1)*8+4]*uid->uid_coeffs[i];
                u[5] += ctmp[(i-1)*8+5]*uid->uid_coeffs[i];
                u[6] += ctmp[(i-1)*8+6]*uid->uid_coeffs[i];
                u[7] += ctmp[(i-1)*8+7]*uid->uid_coeffs[i];
            }

            // compute new vec
            set_zero_avx2(ftmp, vec_length * 8);
            for (long i = 0; i < CSD-1; i++) {
                red_avx2(ftmp+vec_length*0, _b_local[i+1], -ctmp[i*8+0], i+2);
                red_avx2(ftmp+vec_length*1, _b_local[i+1], -ctmp[i*8+1], i+2);
                red_avx2(ftmp+vec_length*2, _b_local[i+1], -ctmp[i*8+2], i+2);
                red_avx2(ftmp+vec_length*3, _b_local[i+1], -ctmp[i*8+3], i+2);
                red_avx2(ftmp+vec_length*4, _b_local[i+1], -ctmp[i*8+4], i+2);
                red_avx2(ftmp+vec_length*5, _b_local[i+1], -ctmp[i*8+5], i+2);
                red_avx2(ftmp+vec_length*6, _b_local[i+1], -ctmp[i*8+6], i+2);
                red_avx2(ftmp+vec_length*7, _b_local[i+1], -ctmp[i*8+7], i+2);
            }
            int32_t c[8];
            for (long i = 0; i < 8; i++) {
                int overflow = 0;
                c[i] = round(-ftmp[i*vec_length]*Bis0);
                u[i] += c[i] * uid->uid_coeffs[0];
                ftmp[i*vec_length] += c[i] * _b_local[0][0];
                for (long j = 0; j < CSD; j++) {
                    (vec + vec_length * (ind+i))[j] = roundf(ftmp[i*vec_length+j]);
                    if (abs(ftmp[i*vec_length+j]) > 127) {
                        overflow = 1;
                        break;
                    }
                }
                if (overflow) {
                    ADD_TO_COLLISION_LIST(ind+i);
                    continue;
                } else if (!uid->insert_uid(u[i])) {
                    fprintf(stderr, "[Error] Pool_epi8_t<%u>::extend_left: insert failed, rejected.\n", nb);
                    ADD_TO_COLLISION_LIST(ind+i);
                } else {
                    vu[ind+i] = u[i];
                }
            }

            #define COMPUTE_VNORM_AND_VSUM_B8 do {                                                  \
                vnorm[ind+0] = roundf(0.5 * dot_avx2(ftmp+vec_length*0, ftmp+vec_length*0, CSD));   \
                vnorm[ind+1] = roundf(0.5 * dot_avx2(ftmp+vec_length*1, ftmp+vec_length*1, CSD));   \
                vnorm[ind+2] = roundf(0.5 * dot_avx2(ftmp+vec_length*2, ftmp+vec_length*2, CSD));   \
                vnorm[ind+3] = roundf(0.5 * dot_avx2(ftmp+vec_length*3, ftmp+vec_length*3, CSD));   \
                vnorm[ind+4] = roundf(0.5 * dot_avx2(ftmp+vec_length*4, ftmp+vec_length*4, CSD));   \
                vnorm[ind+5] = roundf(0.5 * dot_avx2(ftmp+vec_length*5, ftmp+vec_length*5, CSD));   \
                vnorm[ind+6] = roundf(0.5 * dot_avx2(ftmp+vec_length*6, ftmp+vec_length*6, CSD));   \
                vnorm[ind+7] = roundf(0.5 * dot_avx2(ftmp+vec_length*7, ftmp+vec_length*7, CSD));   \
                                                                                                    \
                int32_t sum[8] = {};                                                                \
                for (long i = 0; i < 8; i++) {                                                      \
                    for (long j = 0; j < CSD; j++) {                                                \
                        sum[i] += (vec+vec_length*(ind+i))[j];                                      \
                    }                                                                               \
                }                                                                                   \
                for (long i = 0; i < 8; i++) vsum[ind+i] = 128*sum[i];                              \
            } while (0)

            // compute norm and sum
            COMPUTE_VNORM_AND_VSUM_B8;

            ind += 8;
        }
        while (ind < end_ind) {
            // recover old coeff
            _compute_coeff(ctmp, ind, CSD-1, _b_dual_old, _dhalf_old, _dshift_old);
            
            // compute new vec
            set_zero_avx2(ftmp, vec_length);
            for (long i = 0; i < CSD-1; i++) {
                red_avx2(ftmp, _b_local[i+1], -ctmp[i], vec_length);
            }
            int overflow = 0;
            int c0 = round(-ftmp[0]*Bis0);
            ftmp[0] += c0 * _b_local[0][0];
            for (long i = 0; i < CSD; i++) {
                (vec + vec_length * ind)[i] = roundf(ftmp[i]);
                if (abs(ftmp[i]) > 127) {
                    overflow = 1;
                    break;
                }
            }
            if (overflow) {
                ADD_TO_COLLISION_LIST(ind);
                ind++;
                continue;
            }
            
            // compute u
            uint64_t u = c0 * uid->uid_coeffs[0];
            for (long i = 1; i < CSD; i++) {
                u += ctmp[i-1]*uid->uid_coeffs[i];
            }
            vu[ind] = u;
            if (!uid->insert_uid(u)) {
                fprintf(stderr, "[Error] Pool_epi8_t<%u>::extend_left: insert failed, rejected.\n", nb);
                ADD_TO_COLLISION_LIST(ind);
            }

            #define COMPUTE_VNORM_AND_VSUM do {                                     \
                vnorm[ind] = roundf(0.5 * dot_avx2(ftmp, ftmp, CSD));               \
                                                                                    \
                int32_t sum = 0;                                                    \
                for (long i = 0; i < CSD; i++) sum += (vec + vec_length * ind)[i];  \
                sum *= 128;                                                         \
                vsum[ind] = sum;                                                    \
            } while (0)

            // compute norm and sum
            COMPUTE_VNORM_AND_VSUM;

            ind++;
        }
    }

    PROCESS_COLLISION_LIST;
    _reconstruct_all_cvec();
    FREE_VEC((void *)_b_dual_old);
    mark_pool_dirty();
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::shrink_left() {
    // we do not check sieving context here

    INIT_COLLISION_LIST;

    int32_t _dhalf_old = _dhalf;
    int32_t _dshift_old = _dshift;
    uint8_t *_b_dual_old = _b_dual;
    _b_dual = NULL;

    index_l++;
    CSD--;
    _update_b_local();
    _compute_gh2();
    uid->reset_hash_function(CSD);
    bgj_reserve_uid_table(uid, _pool_size);
    for (long i = 0; i < CSD; i++) uid->insert_uid(uid->uid_coeffs[i]);

    // compute new vec from old vec
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        __attribute__ ((aligned (32))) float ftmp[vec_length * 8];
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        long ind = begin_ind;
        while (ind < end_ind - 7) {
            // recover old coeff
            _compute_coeff_b8(ctmp, ind, CSD+1, _b_dual_old, _dhalf_old, _dshift_old);

            // compute new vec
            set_zero_avx2(ftmp, vec_length * 8);
            for (long i = 1; i < CSD+1; i++) {
                red_avx2(ftmp+vec_length*0, _b_local[i-1], -ctmp[i*8+0], i);
                red_avx2(ftmp+vec_length*1, _b_local[i-1], -ctmp[i*8+1], i);
                red_avx2(ftmp+vec_length*2, _b_local[i-1], -ctmp[i*8+2], i);
                red_avx2(ftmp+vec_length*3, _b_local[i-1], -ctmp[i*8+3], i);
                red_avx2(ftmp+vec_length*4, _b_local[i-1], -ctmp[i*8+4], i);
                red_avx2(ftmp+vec_length*5, _b_local[i-1], -ctmp[i*8+5], i);
                red_avx2(ftmp+vec_length*6, _b_local[i-1], -ctmp[i*8+6], i);
                red_avx2(ftmp+vec_length*7, _b_local[i-1], -ctmp[i*8+7], i);
            }

            // compute u
            uint64_t u[8] = {};
            for (long i = 0; i < CSD; i++) {
                u[0] += ctmp[(i+1)*8+0]*uid->uid_coeffs[i];
                u[1] += ctmp[(i+1)*8+1]*uid->uid_coeffs[i];
                u[2] += ctmp[(i+1)*8+2]*uid->uid_coeffs[i];
                u[3] += ctmp[(i+1)*8+3]*uid->uid_coeffs[i];
                u[4] += ctmp[(i+1)*8+4]*uid->uid_coeffs[i];
                u[5] += ctmp[(i+1)*8+5]*uid->uid_coeffs[i];
                u[6] += ctmp[(i+1)*8+6]*uid->uid_coeffs[i];
                u[7] += ctmp[(i+1)*8+7]*uid->uid_coeffs[i];
            }
            for (long i = 0; i < 8; i++) vu[ind+i] = u[i];
            for (long i = 0; i < 8; i++) {
                int overflow = 0;
                for (long j = 0; j < CSD; j++) {
                    int32_t val = roundf(ftmp[i*vec_length+j]);
                    (vec + vec_length * (ind+i))[j] = val;
                    if (abs(val) > 127) overflow = 1;
                }
                (vec + vec_length * (ind+i))[CSD] = 0;
                if (overflow) {
                    ADD_TO_COLLISION_LIST(ind+i);
                    continue;
                }
                if (!uid->insert_uid(u[i])) ADD_TO_COLLISION_LIST(ind+i);
            }

            COMPUTE_VNORM_AND_VSUM_B8;

            ind += 8;
        }
        while (ind < end_ind) {
            // recover old coeff
            _compute_coeff(ctmp, ind, CSD+1, _b_dual_old, _dhalf_old, _dshift_old);

            // compute new vec
            set_zero_avx2(ftmp, vec_length);
            for (long i = 1; i < CSD+1; i++) {
                red_avx2(ftmp, _b_local[i-1], -ctmp[i], vec_length);
            }
            int overflow = 0;
            for (long i = 0; i < CSD; i++) {
                int32_t val = roundf(ftmp[i]);
                (vec + vec_length * ind)[i] = val;
                if (fabs(val) > 127) overflow = 1;
            }
            (vec + vec_length * ind)[CSD] = 0;
            if (overflow) ADD_TO_COLLISION_LIST(ind);

            // compute u
            uint64_t u = 0;
            for (long i = 0; i < CSD; i++) {
                u += ctmp[i+1]*uid->uid_coeffs[i];
            }
            vu[ind] = u;
            if (!overflow) {
                if (!uid->insert_uid(u)) ADD_TO_COLLISION_LIST(ind);
            }

            // compute norm and sum
            COMPUTE_VNORM_AND_VSUM;

            ind++;
        }
    }

    PROCESS_COLLISION_LIST;

    _reconstruct_all_cvec();
    FREE_VEC((void *)_b_dual_old);
    mark_pool_dirty();

    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::sort_cvec() {
    cvec_for_sort_epi8 *start = (cvec_for_sort_epi8 *) cvec;
    cvec_for_sort_epi8 *middle = (cvec_for_sort_epi8 *) (cvec + sorted_index * 3LL);
    cvec_for_sort_epi8 *end = (cvec_for_sort_epi8 *)(cvec + num_vec * 3LL);
    if (sorted_index == num_vec) return 1;
    if (sorted_index > num_vec / 4) {
        parallel_algorithms::sort(middle, end, cmp_cvec_epi8, _threadpool);
        cvec_for_sort_epi8 *tmp = new cvec_for_sort_epi8[num_vec];
        parallel_algorithms::merge(start, middle, middle, end, tmp, cmp_cvec_epi8, _threadpool);
        parallel_algorithms::copy(tmp, tmp + num_vec, start, _threadpool);
        delete[] tmp;
    }else{
        parallel_algorithms::sort(start, end, cmp_cvec_epi8, _threadpool);
    }
    sorted_index = num_vec;
    // for (long i = 0; i < num_vec; i++) {
    //     uint16_t *cptr = cvec + 3 * i;
    //     uint32_t ind = ((uint32_t *) cptr)[0];
    //     printf("[cvec %ld] ind = %u, cnorm = %u, norm = %u\n", i, ind, cptr[2], vnorm[ind]);
    // }
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::sieve_is_over(double saturation_radius, double saturation_ratio) {
    int32_t goal_norm = round(0.5 * gh2 * _ratio * _ratio * saturation_radius);
    goal_norm >>= 1;
    long goal_num = saturation_ratio * 0.5 * pow(saturation_radius, CSD/2.0);
    long already_found = 0;
    
    // sorted part
    long up = sorted_index;
    long down = 0;
    while (up > down + 1){
        long mid = (up+down)/2;
        if (cvec[mid * 3LL+2LL] <= goal_norm){
            down = mid;
        }else{
            up = mid;
        }
    }
    while (cvec[(down+1) * 3LL+2LL] <= goal_norm) down++;
    already_found = down;

    if (already_found >= goal_num) return 1;
    for (long i = sorted_index; i < num_vec; i++) {
        if (i % 128 == 0) {
            if (already_found + num_vec - i < goal_num) return 0;
            if (already_found >= goal_num) return 1;
        }
        if (cvec[i * 3LL+2LL] <= goal_norm) already_found++;
    }
    if (already_found >= goal_num) return 1;
    return 0;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::insert(long index, double eta) {
    if (index > index_l) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::insert: index(%ld) > index_l(%ld), nothing done.\n", nb, index, index_l);
        return -1;
    }
    if (index < 0) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::insert: negetive index(%ld), nothing done.\n", nb, index);
        return -1;
    }

    // global data
    const long ID = index_l - index;
    const long FD = index_r - index;
    const long FD8 = ((FD + 7) / 8) * 8;
    float **b_ext = _compute_b_local(index, index_r);
    float **b_normal_old = (float **) NEW_MAT(CSD, vec_length, sizeof(float));
    for (long j = 0; j < CSD; j++) {
        float x = sqrt((basis->get_B()).hi[j+index_l]) * _ratio;
        for (long i = j; i < CSD; i++) {
            b_normal_old[i][j] = (basis->get_miu()).hi[i+index_l][j+index_l] * x;
        }
    }
    float Bi[256];
    float Bis[256];
    float Ci[256];
    for (long i = index; i <= index_l; i++) {
        Bi[i] = 1.0 / basis->get_B().hi[i] / _ratio / _ratio;
        Bis[i] = 1.0 / sqrt(basis->get_B().hi[i]) / _ratio;
        Ci[i] = Bi[i] * pow(eta, i-index);
    }

    // min related data
    uint32_t min_ind;
    long min_place = -1;
    float min_score = 0.995 * pow(eta, ID);
    pthread_spinlock_t min_lock;
    pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);
    // require: cur_place
    #define TRY_UPDATE_MIN_INSERT(__ind, __norm, __c) do {                      \
        if (__norm * Bi[cur_place] < 0.995) {                                   \
            float __score = __norm * Ci[cur_place];                             \
            if (__score < min_score) {                                          \
                pthread_spin_lock(&min_lock);                                   \
                if (__score < min_score) {                                      \
                    int __has_one = 0;                                          \
                    __attribute__ ((aligned (32))) float __tmp[256] = {};\
                    for (long __i = 0; __i < CSD; __i++) {                      \
                        float _q = *((int32_t *)(&(__c[__i])));                 \
                        red_avx2(__tmp, b_ext[ID+__i], -_q, FD8);        \
                    }                                                           \
                    for (long __i = 0; __i < CSD; __i++) {                      \
                        __tmp[__i] = __tmp[__i+ID];                             \
                    }                                                           \
                    for (long __i = CSD-1; __i >= 0; __i--) {                   \
                        int32_t _c = round(__tmp[__i]/b_normal_old[__i][__i]);  \
                        if (_c == 1) __has_one = 1;                             \
                        if (_c == -1) __has_one = 1;                            \
                        if (__has_one) break;                                   \
                        red_avx2(__tmp, b_normal_old[__i], _c, FD8);     \
                    }                                                           \
                    if (__has_one) {                                            \
                        min_score = __score;                                    \
                        min_ind = __ind;                                        \
                        min_place = cur_place;                                  \
                    }                                                           \
                }                                                               \
                pthread_spin_unlock(&min_lock);                                 \
            }                                                                   \
        }                                                                       \
    } while (0);

    // search for minimal insertion
    #pragma omp parallel for 
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        __attribute__ ((aligned (32))) float ftmp[256 * 8];
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        long ind = begin_ind;
        while (ind < end_ind - 7) {
            // recover coeff
            _compute_coeff_b8(ctmp, ind);
            
            // compute fvec
            set_zero_avx2(ftmp, 256 * 8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp+FD8*0, b_ext[i+ID], -ctmp[i*8+0], FD8);
                red_avx2(ftmp+FD8*1, b_ext[i+ID], -ctmp[i*8+1], FD8);
                red_avx2(ftmp+FD8*2, b_ext[i+ID], -ctmp[i*8+2], FD8);
                red_avx2(ftmp+FD8*3, b_ext[i+ID], -ctmp[i*8+3], FD8);
                red_avx2(ftmp+FD8*4, b_ext[i+ID], -ctmp[i*8+4], FD8);
                red_avx2(ftmp+FD8*5, b_ext[i+ID], -ctmp[i*8+5], FD8);
                red_avx2(ftmp+FD8*6, b_ext[i+ID], -ctmp[i*8+6], FD8);
                red_avx2(ftmp+FD8*7, b_ext[i+ID], -ctmp[i*8+7], FD8);
            }

            // check scores while lifting to left
            float cur_norm[8] = {};
            long cur_place = index_l;
            for (long i = 0; i < 8; i++) {
                for (long j = ID; j < ID+CSD; j++) cur_norm[i] += (ftmp+FD8*i)[j] * (ftmp+FD8*i)[j];
                TRY_UPDATE_MIN_INSERT(ind+i, cur_norm[i], ((__m256i *)(ctmp + i)));
            }
            while (cur_place > index) {
                cur_place--;
                int c[8];
                for (long i = 0; i < 8; i++) {
                    c[i] = round(ftmp[i*FD8+cur_place-index]*Bis[cur_place]);
                }
                red_avx2(ftmp+FD8*0, b_ext[cur_place-index], c[0], FD8);
                red_avx2(ftmp+FD8*1, b_ext[cur_place-index], c[1], FD8);
                red_avx2(ftmp+FD8*2, b_ext[cur_place-index], c[2], FD8);
                red_avx2(ftmp+FD8*3, b_ext[cur_place-index], c[3], FD8);
                red_avx2(ftmp+FD8*4, b_ext[cur_place-index], c[4], FD8);
                red_avx2(ftmp+FD8*5, b_ext[cur_place-index], c[5], FD8);
                red_avx2(ftmp+FD8*6, b_ext[cur_place-index], c[6], FD8);
                red_avx2(ftmp+FD8*7, b_ext[cur_place-index], c[7], FD8);
                for (long i = 0; i < 8; i++) {
                    cur_norm[i] += (ftmp+FD8*i)[cur_place-index] * (ftmp+FD8*i)[cur_place-index];
                    TRY_UPDATE_MIN_INSERT(ind+i, cur_norm[i], ((__m256i *)(ctmp + i)));
                }
            }
            ind += 8;
        }

        while (ind < end_ind) {
            // recover coeff
            _compute_coeff(ctmp, ind);

            // compute fvec
            set_zero_avx2(ftmp, FD8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp, b_ext[i+ID], -ctmp[i], FD8);
            }

            // check scores while lifting to left
            float cur_norm = 0.0f;
            long cur_place = index_l;
            for (long j = ID; j < ID+CSD; j++) cur_norm += ftmp[j] * ftmp[j];
            TRY_UPDATE_MIN_INSERT(ind, cur_norm, ctmp);
            while (cur_place > index) {
                cur_place--;
                int c = round(ftmp[cur_place-index]*Bis[cur_place]);
                red_avx2(ftmp, b_ext[cur_place-index], c, FD8);
                cur_norm += ftmp[cur_place-index] * ftmp[cur_place-index];
                TRY_UPDATE_MIN_INSERT(ind, cur_norm, ctmp);
            }

            ind++;
        }
    }

    if (min_place == -1) {
        // insertion failed
        shrink_left();
        FREE_MAT(b_ext);
        FREE_MAT(b_normal_old);
        return 0;
    }

    // prepare the new vec
    const long RD = index_l - min_place;
    VEC_QP v_QP = NEW_VEC_QP(basis->NumCols());
    MAT_QP b_QP = basis->get_b();
    int32_t v_normal_coeff[vec_length];
    
    long rm_index;
    do {
        __attribute__ ((aligned (32))) float v_fp[256] = {};
        int32_t v_coeff[256];

        // compute coeff and v_fp
        for (long i = 0; i < CSD; i++) {
            v_coeff[i+RD] = (vdp(_b_dual+i*vec_length, vec+min_ind*vec_length) + _dhalf - vsum[min_ind]) >> _dshift;
        }
        for (long i = 0; i < CSD; i++) {
            red_avx2(v_fp, b_ext[i+ID], -v_coeff[i+RD], FD8);
        }
        for (long i = 0; i < RD; i++) {
            v_coeff[RD - i - 1] = round(-v_fp[ID - i - 1] * Bis[index_l - i - 1]);
            red_avx2(v_fp, b_ext[ID - i - 1], -v_coeff[RD - i - 1], FD8);
        }
        for (long i = 0; i < CSD+RD; i++) v_fp[i] = v_fp[i+ID-RD];
        for (long i = CSD+RD; i < FD; i++) v_fp[i] = 0.0f;
        FREE_MAT((void **)b_ext);

        // compute v_QP from v_fp
        float **tmp1 = (float **) NEW_MAT(CSD+RD, FD8, sizeof(float));
        for (long j = 0; j < CSD+RD; j++){
            double x = sqrt((basis->get_B()).hi[j+min_place]) * _ratio;
            for (long i = j; i < CSD+RD; i++){
                tmp1[i][j] = (basis->get_miu()).hi[i+min_place][j+min_place] * x;
            }
        }
        for (long i = 0; i < RD+CSD; i++) {
            if (i < CSD) v_normal_coeff[CSD-i-1] = round(v_fp[RD+CSD - i - 1] / tmp1[RD+CSD - i - 1][RD+CSD - i - 1]);
            NTL::quad_float q(round(-v_fp[RD+CSD - i - 1] / tmp1[RD+CSD - i - 1][RD+CSD - i - 1]));
            red_avx2(v_fp, tmp1[RD+CSD-i-1], -q.hi, FD8);
            red(v_QP.hi, v_QP.lo, b_QP.hi[index_r-i-1], b_QP.lo[index_r-i-1], q, basis->NumCols());
        }
        FREE_MAT(tmp1);
        
        // compute rm_index
        for (long i = 0; i < CSD; i++) {
            if (v_normal_coeff[i] == 1) {
                rm_index = i + index_l;
                break;
            }
            if (v_normal_coeff[i] == -1) {
                rm_index = i + index_l;
                mul(v_QP, NTL::quad_float(-1.0), basis->NumCols());
                for (long j = 0; j < CSD; j++) v_normal_coeff[j] = -v_normal_coeff[j];
                break;
            }
            if (i == CSD - 1) {
                fprintf(stderr, "[Error] Pool_epi8_t<%u>::insert: no 1 or -1 in coeff, something must be wrong, aborted.\n", nb);
                shrink_left();
                FREE_VEC_QP(v_QP);
                return -1;
            }
        }
    } while (0);

    // process the basis
    for (long i = rm_index; i > min_place; i--) {
        copy(b_QP.hi[i], b_QP.lo[i], b_QP.hi[i-1], b_QP.lo[i-1], basis->NumCols());
    }
    copy(b_QP.hi[min_place], b_QP.lo[min_place], v_QP.hi, v_QP.lo, basis->NumCols());
    FREE_VEC_QP(v_QP);
    basis->compute_gso_QP();
    basis->size_reduce(min_place);

    // keeping old b_local data
    uint8_t *_b_dual_old = _b_dual;
    float **_b_local_old = _b_local;
    int32_t _dhalf_old = _dhalf;
    int32_t _dshift_old = _dshift;
    _b_dual = NULL;
    _b_local = NULL;

    // update b_local data
    index_l++;
    CSD--;
    _update_b_local();
    _compute_gh2();
    uid->reset_hash_function(CSD);
    bgj_reserve_uid_table(uid, _pool_size);
    for (long i = 0; i < CSD; i++) uid->insert_uid(uid->uid_coeffs[i]);

    // b_trans and u_trans
    float **b_trans = (float **) NEW_MAT(CSD+1, vec_length, sizeof(float));
    uint64_t u_trans[vec_length]={};
    do {
        float **b_normal_new = (float **) NEW_MAT(CSD, vec_length, sizeof(float));
        for (long j = 0; j < CSD; j++) {
            float x = sqrt((basis->get_B()).hi[j+index_l]) * _ratio;
            for (long i = j; i < CSD; i++) {
                b_normal_new[i][j] = (basis->get_miu()).hi[i+index_l][j+index_l] * x;
            }
        }

        int32_t old_coeff[vec_length];
        __attribute__ ((aligned (32))) float tmp[vec_length];
        for (long i = 0; i < CSD+1; i++) {
            copy_avx2(tmp, _b_local_old[i], vec_length);
            for (long j = CSD; j >= 0; j--) {
                old_coeff[j] = round(tmp[j]/b_normal_old[j][j]);
                red_avx2(tmp, b_normal_old[j], old_coeff[j], vec_length);
            }

            long c = old_coeff[rm_index - (index_l-1)];
            for (long j = 0; j <= CSD; j++) {
                old_coeff[j] -= c * v_normal_coeff[j];
            }
            for (long j = rm_index - (index_l-1); j < CSD; j++) old_coeff[j] = old_coeff[j+1];
            for (long j = 0; j < CSD; j++) {
                red_avx2(b_trans[i], b_normal_new[j], -old_coeff[j], vec_length);
            }

            copy_avx2(tmp, b_trans[i], vec_length);
            for (long j = CSD-1; j >=0; j--) {
                int32_t c = round(tmp[j]/_b_local[j][j]);
                red_avx2(tmp, _b_local[j], c, vec_length);
                u_trans[i] += c * uid->uid_coeffs[j];
            }
        }
        FREE_MAT(b_normal_old);
        FREE_MAT(b_normal_new);
    } while (0);


    // some vector will fail, so we record them here
    INIT_COLLISION_LIST;

    // compute new vec from old vec
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        __attribute__ ((aligned (32))) float ftmp[vec_length * 8];
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        long ind = begin_ind;
        while (ind < end_ind - 7) {
            // recover old coeff
            _compute_coeff_b8(ctmp, ind, CSD+1, _b_dual_old, _dhalf_old, _dshift_old);

            // compute new vec
            set_zero_avx2(ftmp, vec_length * 8);
            for (long i = 0; i < CSD+1; i++) {
                red_avx2(ftmp+vec_length*0, b_trans[i], -ctmp[i*8+0], vec_length);
                red_avx2(ftmp+vec_length*1, b_trans[i], -ctmp[i*8+1], vec_length);
                red_avx2(ftmp+vec_length*2, b_trans[i], -ctmp[i*8+2], vec_length);
                red_avx2(ftmp+vec_length*3, b_trans[i], -ctmp[i*8+3], vec_length);
                red_avx2(ftmp+vec_length*4, b_trans[i], -ctmp[i*8+4], vec_length);
                red_avx2(ftmp+vec_length*5, b_trans[i], -ctmp[i*8+5], vec_length);
                red_avx2(ftmp+vec_length*6, b_trans[i], -ctmp[i*8+6], vec_length);
                red_avx2(ftmp+vec_length*7, b_trans[i], -ctmp[i*8+7], vec_length);
            }

            // compute u
            uint64_t u[8] = {};
            for (long i = 0; i < CSD+1; i++) {
                u[0] += ctmp[i*8+0]*u_trans[i];
                u[1] += ctmp[i*8+1]*u_trans[i];
                u[2] += ctmp[i*8+2]*u_trans[i];
                u[3] += ctmp[i*8+3]*u_trans[i];
                u[4] += ctmp[i*8+4]*u_trans[i];
                u[5] += ctmp[i*8+5]*u_trans[i];
                u[6] += ctmp[i*8+6]*u_trans[i];
                u[7] += ctmp[i*8+7]*u_trans[i];
            }
            for (long i = 0; i < 8; i++) vu[ind+i] = u[i];
            for (long i = 0; i < 8; i++) {
                int overflow = 0;
                for (long j = 0; j < CSD; j++) {
                    int32_t val = roundf(ftmp[i*vec_length+j]);
                    (vec + vec_length * (ind+i))[j] = val;
                    if (abs(val) > 127) overflow = 1;
                }
                (vec + vec_length * (ind+i))[CSD] = 0;
                if (overflow) {
                    ADD_TO_COLLISION_LIST(ind+i);
                    continue;
                }
                if (!uid->insert_uid(u[i])) ADD_TO_COLLISION_LIST(ind+i);
            }

            COMPUTE_VNORM_AND_VSUM_B8;

            ind += 8;
        }

        while (ind < end_ind) {
            // recover old coeff
            _compute_coeff(ctmp, ind, CSD+1, _b_dual_old, _dhalf_old, _dshift_old);

            // compute new vec
            set_zero_avx2(ftmp, vec_length * 8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp, b_trans[i], -ctmp[i], vec_length);
            }
            int overflow = 0;
            for (long i = 0; i < CSD; i++) {
                int32_t val = roundf(ftmp[i]);
                (vec + vec_length * ind)[i] = val;
                if (fabs(val) > 127) overflow = 1;
            }
            (vec + vec_length * ind)[CSD] = 0;
            if (overflow) ADD_TO_COLLISION_LIST(ind);

            // compute u
            uint64_t u = 0;
            for (long i = 0; i < CSD+1; i++) {
                u += ctmp[i]*u_trans[i];
            }
            vu[ind] = u;
            if (!overflow) {
                if (!uid->insert_uid(u)) ADD_TO_COLLISION_LIST(ind);
            }

            // compute norm and sum
            COMPUTE_VNORM_AND_VSUM;

            ind++;
        }
    }

    // process uid failed and overflowed vectors
    PROCESS_COLLISION_LIST;

    _reconstruct_all_cvec();
    FREE_VEC((void *)_b_dual_old);
    FREE_MAT(_b_local_old);
    mark_pool_dirty();

    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::show_min_lift(long index) {
    last_lift_valid = 0;
    last_lift_euclidean_norm = 0.0;
    last_lift_lift_norm = 0.0;
    last_lift_gh = 0.0;
    last_lift_approx_factor = 0.0;
    const int lift_profile = pool_epi8_lift_profile_enabled();
    const double profile_total_t0 = lift_profile ? pool_epi8_wall_time() : 0.0;
    double profile_setup = 0.0;
    double profile_scan = 0.0;
    double profile_reconstruct = 0.0;
    double profile_qp_build = 0.0;
    double profile_norm_record = 0.0;

    if (index > index_l) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::show_min_lift: index(%ld) > index_l(%ld), nothing done.\n", nb, index, index_l);
        return -1;
    }
    if (index < 0) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::show_min_lift: negetive index(%ld), nothing done.\n", nb, index);
        return -1;
    }

    // global data
    const long ID = index_l - index;
    const long FD = index_r - index;
    const long FD8 = ((FD + 7) / 8) * 8;
    double profile_t0 = lift_profile ? pool_epi8_wall_time() : 0.0;
    float **b_ext = _compute_b_local(index, index_r);
    float Bi[256];
    float Bis[256];
    for (long i = index; i <= index_l; i++) {
        Bi[i] = 1.0 / basis->get_B().hi[i] / _ratio / _ratio;
        Bis[i] = 1.0 / sqrt(basis->get_B().hi[i]) / _ratio;
    }

    // min related data
    uint32_t min_ind = (uint32_t) -1;
    float min_norm = b_ext[0][0] * b_ext[0][0] * 0.995;
    pthread_spinlock_t min_lock;
    pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);
    if (lift_profile) profile_setup = pool_epi8_wall_time() - profile_t0;

    // search for minimal insertion
    profile_t0 = lift_profile ? pool_epi8_wall_time() : 0.0;
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        __attribute__ ((aligned (32))) float ftmp[256 * 8];
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        long ind = begin_ind;
        while (ind < end_ind - 7) {
            // recover coeff
            _compute_coeff_b8(ctmp, ind);
            
            // compute fvec
            set_zero_avx2(ftmp, 256 * 8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp+FD8*0, b_ext[i+ID], -ctmp[i*8+0], FD8);
                red_avx2(ftmp+FD8*1, b_ext[i+ID], -ctmp[i*8+1], FD8);
                red_avx2(ftmp+FD8*2, b_ext[i+ID], -ctmp[i*8+2], FD8);
                red_avx2(ftmp+FD8*3, b_ext[i+ID], -ctmp[i*8+3], FD8);
                red_avx2(ftmp+FD8*4, b_ext[i+ID], -ctmp[i*8+4], FD8);
                red_avx2(ftmp+FD8*5, b_ext[i+ID], -ctmp[i*8+5], FD8);
                red_avx2(ftmp+FD8*6, b_ext[i+ID], -ctmp[i*8+6], FD8);
                red_avx2(ftmp+FD8*7, b_ext[i+ID], -ctmp[i*8+7], FD8);
            }

            // check scores while lifting to left
            float cur_norm[8] = {};
            long cur_place = index_l;
            while (cur_place > index) {
                cur_place--;
                int c[8];
                for (long i = 0; i < 8; i++) {
                    c[i] = round(ftmp[i*FD8+cur_place-index]*Bis[cur_place]);
                }
                red_avx2(ftmp+FD8*0, b_ext[cur_place-index], c[0], FD8);
                red_avx2(ftmp+FD8*1, b_ext[cur_place-index], c[1], FD8);
                red_avx2(ftmp+FD8*2, b_ext[cur_place-index], c[2], FD8);
                red_avx2(ftmp+FD8*3, b_ext[cur_place-index], c[3], FD8);
                red_avx2(ftmp+FD8*4, b_ext[cur_place-index], c[4], FD8);
                red_avx2(ftmp+FD8*5, b_ext[cur_place-index], c[5], FD8);
                red_avx2(ftmp+FD8*6, b_ext[cur_place-index], c[6], FD8);
                red_avx2(ftmp+FD8*7, b_ext[cur_place-index], c[7], FD8);
            }
            for (long i = 0; i < 8; i++) {
                cur_norm[i] = dot_avx2(ftmp+i*FD8, ftmp+i*FD8, FD8);
                if (cur_norm[i] < min_norm) {
                    pthread_spin_lock(&min_lock);
                    if (cur_norm[i] < min_norm) {
                        min_norm = cur_norm[i];
                        min_ind = ind+i;
                    }
                    pthread_spin_unlock(&min_lock);
                }
            }
            ind += 8;
        }

        while (ind < end_ind) {
            // recover coeff
            _compute_coeff(ctmp, ind);

            // compute fvec
            set_zero_avx2(ftmp, FD8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp, b_ext[i+ID], -ctmp[i], FD8);
            }

            // check scores while lifting to left
            float cur_norm = 0.0f;
            long cur_place = index_l;
            while (cur_place > index) {
                cur_place--;
                int c = round(ftmp[cur_place-index]*Bis[cur_place]);
                red_avx2(ftmp, b_ext[cur_place-index], c, FD8);
            }
            cur_norm = dot_avx2(ftmp, ftmp, FD8);
            if (cur_norm < min_norm) {
                pthread_spin_lock(&min_lock);
                if (cur_norm < min_norm) {
                    min_norm = cur_norm;
                    min_ind = ind;
                }
                pthread_spin_unlock(&min_lock);
            }
            ind++;
        }
    }
    if (lift_profile) profile_scan = pool_epi8_wall_time() - profile_t0;

    if (min_ind == (uint32_t) -1) {
        // insertion failed
        FREE_MAT(b_ext);
        if (lift_profile) {
            fprintf(stderr,
                    "show_min_lift_profile: nb=%u index=%ld csd=%ld nvec=%ld min_ind=none "
                    "setup=%.6fs scan=%.6fs total=%.6fs\n",
                    nb, index, CSD, num_vec, profile_setup, profile_scan,
                    pool_epi8_wall_time() - profile_total_t0);
        }
        return 0;
    }

    // prepare the new vec
    VEC_QP v_QP = NEW_VEC_QP(basis->NumCols());
    MAT_QP b_QP = basis->get_b();
    
    do {
        __attribute__ ((aligned (32))) float v_fp[256] = {};
        int32_t v_coeff[256];

        // compute coeff and v_fp
        profile_t0 = lift_profile ? pool_epi8_wall_time() : 0.0;
        for (long i = 0; i < CSD; i++) {
            v_coeff[i+ID] = (vdp(_b_dual+i*vec_length, vec+min_ind*vec_length) + _dhalf - vsum[min_ind]) >> _dshift;
        }
        for (long i = 0; i < CSD; i++) {
            red_avx2(v_fp, b_ext[i+ID], -v_coeff[i+ID], FD8);
        }
        for (long i = 0; i < ID; i++) {
            v_coeff[ID - i - 1] = round(-v_fp[ID - i - 1] * Bis[index_l - i - 1]);
            red_avx2(v_fp, b_ext[ID - i - 1], -v_coeff[ID - i - 1], FD8);
        }
        for (long i = 0; i < CSD+ID; i++) v_fp[i] = v_fp[i];
        for (long i = CSD+ID; i < FD; i++) v_fp[i] = 0.0f;
        FREE_MAT((void **)b_ext);
        if (lift_profile) profile_reconstruct = pool_epi8_wall_time() - profile_t0;

        // compute v_QP from v_fp
        profile_t0 = lift_profile ? pool_epi8_wall_time() : 0.0;
        float **tmp1 = (float **) NEW_MAT(CSD+ID, FD8, sizeof(float));
        for (long j = 0; j < CSD+ID; j++){
            double x = sqrt((basis->get_B()).hi[j+index]) * _ratio;
            for (long i = j; i < CSD+ID; i++){
                tmp1[i][j] = (basis->get_miu()).hi[i+index][j+index] * x;
            }
        }
        for (long i = 0; i < ID+CSD; i++) {
            NTL::quad_float q(round(-v_fp[ID+CSD - i - 1] / tmp1[ID+CSD - i - 1][ID+CSD - i - 1]));
            red_avx2(v_fp, tmp1[ID+CSD-i-1], -q.hi, FD8);
            red(v_QP.hi, v_QP.lo, b_QP.hi[index_r-i-1], b_QP.lo[index_r-i-1], q, basis->NumCols());
        }
        FREE_MAT(tmp1);

        for (long i = index - 1; i >= 0; i--) {
            int32_t c = round(dot_avx2(v_QP.hi, basis->get_b_star().hi[i], basis->NumCols()) / basis->get_B().hi[i]);
            red(v_QP.hi, v_QP.lo, b_QP.hi[i], b_QP.lo[i], NTL::quad_float(c), basis->NumCols());
        }
        if (lift_profile) profile_qp_build = pool_epi8_wall_time() - profile_t0;
    } while (0);

    profile_t0 = lift_profile ? pool_epi8_wall_time() : 0.0;
    const double euclidean_norm = sqrt(dot_avx2(v_QP.hi, v_QP.hi, basis->NumCols()));
    const double lift_norm = sqrt(min_norm);
    const double gh = basis->gh(index, index_r);
    const double approx = (gh > 0.0) ? euclidean_norm / gh : 0.0;
    last_lift_valid = 1;
    last_lift_euclidean_norm = euclidean_norm;
    last_lift_lift_norm = lift_norm;
    last_lift_gh = gh;
    last_lift_approx_factor = approx;
    if (index == 0) {
        bgj_lsh_best_solution_record(euclidean_norm, v_QP.hi, basis->NumCols());
    }
    printf("\nlength = %.9g(%.9g), gh = %.9g, approx = %.9g, vec = ",
           euclidean_norm, lift_norm, gh, approx);
    PRINT_VEC(v_QP.hi, basis->NumCols());
    fflush(stdout);
    if (lift_profile) {
        profile_norm_record = pool_epi8_wall_time() - profile_t0;
        fprintf(stderr,
                "show_min_lift_profile: nb=%u index=%ld csd=%ld fd=%ld id=%ld nvec=%ld "
                "min_ind=%u length=%.9g setup=%.6fs scan=%.6fs reconstruct=%.6fs "
                "qp_build=%.6fs norm_record=%.6fs total=%.6fs\n",
                nb, index, CSD, FD, ID, num_vec, min_ind, euclidean_norm,
                profile_setup, profile_scan, profile_reconstruct, profile_qp_build,
                profile_norm_record, pool_epi8_wall_time() - profile_total_t0);
    }
    FREE_VEC_QP(v_QP);
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::tail_LLL(double delta, long n) {
    if (n > CSD) {
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::tail_LLL: n(%ld) > CSD(%ld), adjust to CSD.", nb, n, CSD);
        n = CSD;
    }

    // create a new Lattice_QP and LLL
    uint8_t *_b_dual_old = (uint8_t *) NEW_VEC(CSD * vec_length, sizeof(uint8_t));
    if (!_b_dual_old) return 0;
    int32_t _dhalf_old = 0;
    int32_t _dshift_old = 0;
    const int tail_use_fplll = bgj_tail_lll_use_fplll();
    const int tail_use_custom_deep = bgj_tail_deep_lll_use_custom();
    const double tail_min_row_abs = 1.0e-30;

    auto lattice_entries_are_sane = [&](Lattice_QP *L, const char **reason) -> int {
        if (L == NULL) {
            if (reason) *reason = "null lattice";
            return 0;
        }
        MAT_QP b = L->get_b();
        for (long i = 0; i < L->NumRows(); i++) {
            double row_max = 0.0;
            for (long j = 0; j < L->NumCols(); j++) {
                const double hi = b.hi[i][j];
                const double lo = b.lo[i][j];
                if (!std::isfinite(hi) || !std::isfinite(lo)) {
                    if (reason) *reason = "non-finite local basis entry";
                    return 0;
                }
                const double abs_hi = fabs(hi);
                if (abs_hi > row_max) row_max = abs_hi;
            }
            if (row_max <= tail_min_row_abs) {
                if (reason) *reason = "zero local basis row";
                return 0;
            }
        }
        return 1;
    };

    auto prepare_tail_basis = [&](int force_custom_tail, const char **reason) -> int {
        Lattice_QP *L_tmp = new Lattice_QP(CSD, CSD);
        Lattice_QP *L_src = NULL;
        Lattice_QP *L_tmp_dual = NULL;
        if (!L_tmp) {
            if (reason) *reason = "failed to allocate local lattice";
            return 0;
        }
        for (long i = 0; i < CSD; i++) {
            for (long j = 0; j < CSD; j++) {
                L_tmp->get_b().hi[i][j] = _b_local[i][j];
            }
        }
        L_src = this->basis->b_loc_QP(this->index_l, this->index_r);
        if (!L_src) {
            if (reason) *reason = "failed to create projected source lattice";
            delete L_tmp;
            return 0;
        }
        do {
            NTL::quad_float _ratio_qp(_ratio);
            for (long i = 0; i < CSD; i++) {
                for (long j = 0; j < CSD; j++) {
                    NTL::quad_float x_qp(L_src->get_b().hi[i][j], L_src->get_b().lo[i][j]);
                    x_qp *= _ratio_qp;
                    L_src->get_b().hi[i][j] = x_qp.hi;
                    L_src->get_b().lo[i][j] = x_qp.lo;
                }
            }
        } while (0);

        if (!L_tmp->reconstruct(L_src)) {
            if (reason) *reason = "reconstruct failed";
            delete L_src;
            delete L_tmp;
            return 0;
        }
        delete L_src;
        L_src = NULL;
        int fplll_tail_status = -1;
        if (!force_custom_tail && tail_use_fplll) {
            fplll_tail_status = bgj_fplll_lll_qp(*L_tmp, delta, CSD-n, CSD, 0);
        }
        if (force_custom_tail || fplll_tail_status != 0) {
            L_tmp->LLL_QP(delta, CSD-n, CSD);
        }
        if (force_custom_tail || fplll_tail_status != 0 || tail_use_custom_deep) {
            L_tmp->LLL_DEEP_QP(delta-0.03, CSD-n, CSD);
        }
        if (!lattice_entries_are_sane(L_tmp, reason)) {
            delete L_tmp;
            return 0;
        }
        if (L_tmp->dual_size_red() != 1) {
            if (reason) *reason = "dual size reduction failed";
            delete L_tmp;
            return 0;
        }
        if (!lattice_entries_are_sane(L_tmp, reason)) {
            delete L_tmp;
            return 0;
        }
        L_tmp_dual = L_tmp->dual_QP();
        if (!lattice_entries_are_sane(L_tmp_dual, reason)) {
            delete L_tmp;
            delete L_tmp_dual;
            return 0;
        }

        // prepare b_dual_old before changing the global basis.
        double mdd = 0.0;
        for (long i = 0; i < CSD; i++) {
            for (long j = 0; j < CSD; j++) {
                const double v = L_tmp_dual->get_b().hi[i][j];
                if (!std::isfinite(v)) {
                    if (reason) *reason = "non-finite dual basis entry";
                    delete L_tmp;
                    delete L_tmp_dual;
                    return 0;
                }
                if (fabs(v) > mdd) mdd = fabs(v);
            }
        }
        const double dhalf_floor = (mdd > 0.0) ? floor(127.0 / mdd) : 0.0;
        if (!std::isfinite(mdd) || !std::isfinite(dhalf_floor) ||
            dhalf_floor < 2.0 || dhalf_floor > 2147483647.0) {
            if (reason) *reason = "dual basis too coarse for int8 scaling";
            delete L_tmp;
            delete L_tmp_dual;
            return 0;
        }
        const uint32_t dhalf_candidate = (uint32_t)dhalf_floor;
        _dshift_old = 31 - __builtin_clz(dhalf_candidate);
        if (_dshift_old <= 0) {
            if (reason) *reason = "invalid dual scaling shift";
            delete L_tmp;
            delete L_tmp_dual;
            return 0;
        }
        _dhalf_old = 1 << (_dshift_old-1);
        if (_dhalf_old < 1) {
            if (reason) *reason = "invalid dual scaling half-width";
            delete L_tmp;
            delete L_tmp_dual;
            return 0;
        }
        const double dual_scale = ((_dhalf_old << 1) + 0.0);
        for (long i = 0; i < CSD; i++) {
            mul_avx2(L_tmp_dual->get_b().hi[i], dual_scale, CSD);
            for (long j = 0; j < CSD; j++) {
                const double scaled = L_tmp_dual->get_b().hi[i][j] + 128.0;
                const long rounded = (long)round(scaled);
                if (!std::isfinite(scaled) || rounded < 0 || rounded > 255) {
                    if (reason) *reason = "dual basis int8 scaling overflow";
                    delete L_tmp;
                    delete L_tmp_dual;
                    return 0;
                }
                _b_dual_old[i*vec_length+j] = (uint8_t)rounded;
            }
        }

        // process basis
        NTL::quad_float q(1.0/_ratio);
        for (long i = 0; i < CSD; i++){
            mul(L_tmp->get_b().hi[i], L_tmp->get_b().lo[i], q, CSD);
        }
        if (!lattice_entries_are_sane(L_tmp, reason)) {
            delete L_tmp;
            delete L_tmp_dual;
            return 0;
        }
        if (!basis->trans_to(index_l, index_r, L_tmp)) {
            if (reason) *reason = "basis transfer failed";
            delete L_tmp;
            delete L_tmp_dual;
            return 0;
        }
        basis->compute_gso_QP();

        delete L_tmp;
        delete L_tmp_dual;
        return 1;
    };

    const char *tail_fail_reason = "unknown failure";
    if (!prepare_tail_basis(0, &tail_fail_reason)) {
        if (tail_use_fplll) {
            fprintf(stderr,
                    "[Warning] Pool_epi8_t<%u>::tail_LLL: fast tail reduction failed on [%ld,%ld) (%s), retrying with custom deep-LLL.\n",
                    nb, index_l, index_r, tail_fail_reason);
            const char *fallback_fail_reason = "unknown failure";
            if (!prepare_tail_basis(1, &fallback_fail_reason)) {
                fprintf(stderr,
                        "[Error] Pool_epi8_t<%u>::tail_LLL: custom deep-LLL fallback failed on [%ld,%ld) (%s), keeping current basis and pool.\n",
                        nb, index_l, index_r, fallback_fail_reason);
                FREE_VEC(_b_dual_old);
                return 0;
            }
        } else {
            fprintf(stderr,
                    "[Error] Pool_epi8_t<%u>::tail_LLL: tail reduction failed on [%ld,%ld) (%s), keeping current basis and pool.\n",
                    nb, index_l, index_r, tail_fail_reason);
            FREE_VEC(_b_dual_old);
            return 0;
        }
    }

    float **b_normal = (float **) NEW_MAT(CSD, vec_length, sizeof(float));
    do {
        for (long j = 0; j < CSD; j++) {
            float x = sqrt((basis->get_B()).hi[j+index_l]) * _ratio;
            for (long i = j; i < CSD; i++) {
                b_normal[i][j] = (basis->get_miu()).hi[i+index_l][j+index_l] * x;
            }
        }
    } while (0);

    basis->size_reduce();
    _update_b_local(_ratio);
    uid->reset_hash_function(CSD);
    bgj_reserve_uid_table(uid, _pool_size);
    for (long i = 0; i < CSD; i++) uid->insert_uid(uid->uid_coeffs[i]);

    uint64_t u_normal[vec_length] = {};
    do {
        for (long i = 0; i < CSD; i++) {
            __attribute__ ((aligned (32))) float tmp[vec_length];
            copy_avx2(tmp, b_normal[i], vec_length);
            for (long j = CSD - 1; j >= 0; j--) {
                int32_t c = round(tmp[j]/_b_local[j][j]);
                red_avx2(tmp, _b_local[j], c, vec_length);
                u_normal[i] += c * uid->uid_coeffs[j];
            }
        }
    } while (0);

    INIT_COLLISION_LIST;

    // update all uid and remove vectors we cannot recover
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        __attribute__ ((aligned (32))) float ftmp[vec_length * 8];
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        long ind = begin_ind;
        while (ind < end_ind - 7) {
            // compute coeff
            _compute_coeff_b8(ctmp, ind, CSD, _b_dual_old, _dhalf_old, _dshift_old);

            // compute vec
            set_zero_avx2(ftmp, vec_length * 8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp+vec_length*0, b_normal[i], -ctmp[i*8+0], i+1);
                red_avx2(ftmp+vec_length*1, b_normal[i], -ctmp[i*8+1], i+1);
                red_avx2(ftmp+vec_length*2, b_normal[i], -ctmp[i*8+2], i+1);
                red_avx2(ftmp+vec_length*3, b_normal[i], -ctmp[i*8+3], i+1);
                red_avx2(ftmp+vec_length*4, b_normal[i], -ctmp[i*8+4], i+1);
                red_avx2(ftmp+vec_length*5, b_normal[i], -ctmp[i*8+5], i+1);
                red_avx2(ftmp+vec_length*6, b_normal[i], -ctmp[i*8+6], i+1);
                red_avx2(ftmp+vec_length*7, b_normal[i], -ctmp[i*8+7], i+1);
            }

            uint64_t u[8] = {};
            for (long i = 0; i < CSD; i++) {
                u[0] += ctmp[i*8+0]*u_normal[i];
                u[1] += ctmp[i*8+1]*u_normal[i];
                u[2] += ctmp[i*8+2]*u_normal[i];
                u[3] += ctmp[i*8+3]*u_normal[i];
                u[4] += ctmp[i*8+4]*u_normal[i];
                u[5] += ctmp[i*8+5]*u_normal[i];
                u[6] += ctmp[i*8+6]*u_normal[i];
                u[7] += ctmp[i*8+7]*u_normal[i];
            }
            for (long i = 0; i < 8; i++) vu[ind+i] = u[i];
            for (long i = 0; i < 8; i++) {
                int overflow = 0;
                for (long j = 0; j < CSD; j++) {
                    int32_t val = roundf(ftmp[i*vec_length+j]);
                    (vec + vec_length * (ind+i))[j] = val;
                    if (abs(val) > 127) overflow = 1;
                }
                if (overflow) {
                    ADD_TO_COLLISION_LIST(ind+i);
                    continue;
                }
                if (!uid->insert_uid(u[i])) ADD_TO_COLLISION_LIST(ind+i);
            }

            COMPUTE_VNORM_AND_VSUM_B8;
            
            ind += 8;
        }
        while (ind < end_ind) {
            // compute coeff
            _compute_coeff(ctmp, ind, CSD, _b_dual_old, _dhalf_old, _dshift_old);

            set_zero_avx2(ftmp, vec_length);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp, b_normal[i], -ctmp[i], i+1);
            }

            int overflow = 0;
            for (long i = 0; i < CSD; i++) {
                int32_t val = roundf(ftmp[i]);
                (vec + vec_length * ind)[i] = val;
                if (fabs(val) > 127) overflow = 1;
            }
            if (overflow) ADD_TO_COLLISION_LIST(ind);

            // compute u
            uint64_t u = 0;
            for (long i = 0; i < CSD; i++) {
                u += ctmp[i]*u_normal[i];
            }
            vu[ind] = u;

            if (!overflow) {
                if (!uid->insert_uid(u)) ADD_TO_COLLISION_LIST(ind);
            }

            COMPUTE_VNORM_AND_VSUM;

            ind++;
        }
    }

    PROCESS_COLLISION_LIST;                                              

    _reconstruct_all_cvec();
    FREE_VEC(_b_dual_old);
    FREE_MAT(b_normal);

    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::store(const char *file_name) {
    FILE *file = fopen(file_name, "w");
    if (!file) {
        fprintf(stderr, "[Warning] pool_epi8_t<%u>::store: can not open %s, aborted.\n", nb, file_name);
        return -1;
    }
    
    fprintf(file, "%ld\n%ld\n%ld\n", index_l, index_r, _pool_size);
    for (long i = 0; i < num_vec; i++) {
        fprintf(file, "[");
        for (long j = 0; j < CSD; j++) {
            fprintf(file, "%d", (int)vec[i*vec_length+j]);
            if (j < CSD - 1) {
                fprintf(file, " ");
            } else {
                fprintf(file, "]\n");
            }
        }
    }

    // store uid->uid_coeffs
    fprintf(file, "#\n[");
    for (long j = 0; j < CSD; j++) {
        fprintf(file, "%lu", uid->uid_coeffs[j]);
        if (j < CSD - 1) {
            fprintf(file, " ");
        } else {
            fprintf(file, "]\n");
        }
    }

    // store vu
    fprintf(file, "[");
    for (long j = 0; j < num_vec; j++) {
        fprintf(file, "%lu", vu[j]);
        if (j < num_vec - 1) {
            fprintf(file, " ");
        } else {
            fprintf(file, "]\n");
        }
    }

    // store vnorm
    fprintf(file, "[");
    for (long j = 0; j < num_vec; j++) {
        fprintf(file, "%d", vnorm[j]);
        if (j < num_vec - 1) {
            fprintf(file, " ");
        } else {
            fprintf(file, "]\n");
        }
    }

    // store vsum
    fprintf(file, "[");
    for (long j = 0; j < num_vec; j++) {
        fprintf(file, "%d", vsum[j]);
        if (j < num_vec - 1) {
            fprintf(file, " ");
        } else {
            fprintf(file, "]\n");
        }
    }

    fclose(file);
    return 0;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::load(const char *file_name) {
    FILE *file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "[Warning] pool_epi8_t<%u>::load: can not open %s, aborted.\n", nb, file_name);
        return -1;
    }

    clear_pool();
    if (EOF == fscanf(file, "%ld\n%ld\n%ld\n", &index_l, &index_r, &_pool_size)) goto err;
    set_max_pool_size(_pool_size);
    CSD = index_r - index_l;
    while (getc(file) == '[') {
        int tmp;
        for (long i = 0; i < CSD - 1; i++) {
            if (EOF == fscanf(file, "%d ", &tmp)) goto err;
            vec[num_vec*vec_length + i] = tmp;
        }
        if (EOF == fscanf(file, "%d]\n", &tmp)) goto err;
        vec[num_vec*vec_length+CSD-1] = tmp;
        num_vec++;
    }


    uid = new UidHashTable();
    if (uid == NULL) return -1;
    uid->uid_coeffs = new uint64_t[CSD+16];
    if (uid->uid_coeffs == NULL) return -1;
    if (EOF == fscanf(file, "\n[%lu ", &(uid->uid_coeffs[0]))) goto err;
    for (long i = 1; i < CSD-1; i++) if (EOF == fscanf(file, "%lu ", &(uid->uid_coeffs[i]))) goto err;
    if (EOF == fscanf(file, "%lu]\n", &(uid->uid_coeffs[CSD-1]))) goto err;
    uid->insert_uid(0);
    for (long i = 0; i < CSD; i++) uid->insert_uid(uid->uid_coeffs[i]);


    _compute_gh2();
    _update_b_local();

    // vu
    if (EOF == fscanf(file, "[%lu ", vu)) goto err;
    for (long i = 1; i < num_vec-1; i++) if (EOF == fscanf(file, "%lu ", vu+i)) goto err;
    if (EOF == fscanf(file, "%lu]\n", vu+num_vec-1)) goto err;

    // vnorm
    if (EOF == fscanf(file, "[%d ", vnorm)) goto err;
    for (long i = 1; i < num_vec-1; i++) if (EOF == fscanf(file, "%d ", vnorm+i)) goto err;
    if (EOF == fscanf(file, "%d]\n", vnorm+num_vec-1)) goto err;

    // vsum
    if (EOF == fscanf(file, "[%d ", vsum)) goto err;
    for (long i = 1; i < num_vec-1; i++) if (EOF == fscanf(file, "%d ", vsum+i)) goto err;
    if (EOF == fscanf(file, "%d]\n", vsum+num_vec-1)) goto err;


    fclose(file);
    
    // reconstruct cvec
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        for (long cind = begin_ind; cind < end_ind; cind++) {
            int32_t cnorm = vnorm[cind] >> 1;
            *((uint32_t *)(cvec+3LL*cind)) = cind;
            cvec[3LL*cind+2LL] = (cnorm > 65535) ? 65535 : cnorm;
        }
    }
    mark_pool_dirty();

    return 0;
    
    err:
        fprintf(stderr, "[Warning] Pool_epi8_t<%u>::load: format error, pool cleared.\n", nb);
        clear_pool();
        fclose(file);
        return -1;
}

template<uint32_t nb>
double Pool_epi8_t<nb>::pot() {
    double *B = basis->get_B().hi;
    double pot = 0.0;
    for (long i = 0; i < basis->NumRows(); i++){
        pot += (basis->NumRows()-i) * log2(B[i]);
    }
    return pot;
}

template<uint32_t nb>
bool Pool_epi8_t<nb>::check_pool_status(long q, int supress_minor) {
    int pass = 1;
    long num_err = 0;
    __attribute__ ((aligned (32))) int32_t coeff[vec_length];
    __attribute__ ((aligned (32))) float tmp[vec_length];
    if (num_vec < 100) {
        pass = 0;
        num_err++;
        fprintf(stderr, "pool_size = %ld.\n", num_vec);
    }
    for (long ind = 0; ind < num_vec; ind++) {
        if (q && (Uniform_long(q) != 0)) continue;
        // check sum
        int32_t sum = 0;
        for (long i = 0; i < CSD; i++) sum += (vec+ind*vec_length)[i];
        if (128 * sum != vsum[ind]) {
            pass = 0;
            num_err++;
            if (num_err <= 100) fprintf(stderr, "[vec %ld] real sum = %d, vsum = %d\n", ind, 128*sum, vsum[ind]);
        }

        // check vec
        _compute_coeff(coeff, ind);
        _compute_fvec(tmp, coeff);
        for (long i = 0; i < vec_length; i++) {
            if (roundf(tmp[i]) != vec[ind*vec_length+i]) {
                if (supress_minor && (fabs(roundf(tmp[i]) - vec[ind*vec_length+i])) < 1.01) continue;
                pass = 0;
                num_err++;
                if (num_err <= 100) fprintf(stderr, "[vec %ld] reconstruct failed, oringin vec = [", ind);
                for (long i = 0; i < CSD; i++) {
                    if (num_err <= 100) fprintf(stderr, "%d", (int)vec[ind*vec_length+i]);
                    if (i < CSD - 1) {
                        if (num_err <= 100) fprintf(stderr, " ");
                    } else {
                        if (num_err <= 100) fprintf(stderr, "]\nnew vec = [");
                    }
                }
                for (long i = 0; i < CSD; i++) {
                    if (num_err <= 100) fprintf(stderr, "%d", (int)roundf(tmp[i]));
                    if (i < CSD - 1) {
                        if (num_err <= 100) fprintf(stderr, " ");
                    } else {
                        if (num_err <= 100) fprintf(stderr, "]\n");
                    }
                }
                break;
            }
        }
        
        // check u
        uint64_t u = 0;
        for (long i = 0; i < CSD; i++) u += coeff[i] * uid->uid_coeffs[i];
        if (u != vu[ind]) {
            pass = 0;
            num_err++;
            if (num_err <= 100) fprintf(stderr, "[vec %ld] real u = %lu, vu = %lu\n", ind, u, vu[ind]);
        }

        // check norm
        int32_t nn = roundf(0.5 * dot_avx2(tmp, tmp, CSD));
        if (vnorm[ind] != nn) {
            if (supress_minor && (abs(vnorm[ind] - nn) <= 1)) continue;
            pass = 0;
            num_err++;
            if (num_err <= 100) fprintf(stderr, "[vec %ld] real norm = %d, vnorm = %d\n", ind, nn, vnorm[ind]);
        }
    }

    #if defined(__AMX_INT8__) && BOOST_AMX_SIEVE
    #else
    for (long cind = 0; cind < num_vec; cind++) {
        if (q && (Uniform_long(q) != 0)) continue;
        long ind = *((uint32_t *)(cvec+3LL*cind));
        int32_t nn = vnorm[ind] >> 1;
        uint16_t cn = (nn > 65535) ? 65535 : nn;
        if (cn != cvec[3LL*cind+2LL]) {
            if (supress_minor && abs(cn - cvec[3LL*cind+2LL]) <= 1) continue;
            pass = 0;
            num_err++;
            if (num_err <= 100) fprintf(stderr, "[cvec %ld --> vec %ld] real cnorm = %u, recorded cnorm = %u\n", cind, ind, cn, cvec[3LL*cind+2LL]);
        }
    }
    #endif
    if (uid == NULL) return pass;
    long not_in = 0;
    for (long ind = 0; ind < num_vec; ind++) {
        if (!uid->check_uid(vu[ind])) not_in++;
    }
    if (not_in) {
        pass = 0;
        num_err++;
        if (num_err <= 100) fprintf(stderr, "uid of %ld pool vec not in\n", not_in);
    }
    if (uid->size() - 1 - CSD != num_vec) {
        pass = 0;
        num_err++;
        if (num_err <= 100) fprintf(stderr, "UidHashTable size = %ld, num_vec = %ld\n", uid->size(), num_vec);
    }
    if (num_err > 100) {
        fprintf(stderr, "too much errors... %ld errors not report\n", num_err - 100);
    }
    return pass;
}

template<uint32_t nb>
bool Pool_epi8_t<nb>::check_dim_lose(long q) {
    int pass = 1;
    long num_checked = 0;
    long count[vec_length] = {};

    __attribute__ ((aligned (32))) int32_t coeff[vec_length];
    for (long ind = 0; ind < num_vec; ind++) {
        if (q && (Uniform_long(q) != 0)) continue;
        num_checked++;
        _compute_coeff(coeff, ind);
        for (long i = 0; i < CSD; i++) {
            if (coeff[i]) count[i]++;
        }
    }
    for (long i = 0; i < CSD; i++) {
        if (count[i] == 0 && q == 0 && num_vec) {
            pass = 0;
            printf("%ld-th basis vec definitely lost\n", i);
        } else if (count[i] * 100 < num_checked && count[i] < 100) {
            pass = 0;
            printf("%ld-th basis vec may lost, %ld/%ld nonzero\n", i, count[i], num_checked);
        }
    }
    return pass;
}


////////////////////// private methods //////////////////////

template<uint32_t nb>
int Pool_epi8_t<nb>::_update_b_local(float ratio) {
    const long FD8 = (vec_length < CSD) ? ((CSD + 7) / 8) * 8 : vec_length;
    // prepare the space
    if (_b_local) {
        FREE_MAT((void **)_b_local);
        _b_local = NULL;
    }
    if (_b_dual) {
        FREE_VEC((void *)_b_dual);
        _b_dual = NULL;
    }
    double **tmp1 = (double **) NEW_MAT(CSD, CSD, sizeof(double));
    double **tmp2 = (double **) NEW_MAT(CSD, CSD, sizeof(double));
    _b_local = (float **) NEW_MAT(CSD, FD8, sizeof(float));
    _b_dual = (uint8_t *) NEW_VEC(CSD * FD8, sizeof(uint8_t)); 
    if (!tmp1 || !tmp2 || !_b_local || !_b_dual) return -1;

    // prepare b_local_tr
    _ratio = -1.0;
    double **miu = (basis->get_miu()).hi;
    double *B = (basis->get_B()).hi;
    for (long j = 0; j < CSD; j++){
        double x = sqrt(B[j+index_l]);
        if (x > _ratio) _ratio = x;
        for (long i = j; i < CSD; i++){
            tmp1[j][i] = miu[i+index_l][j+index_l] * x;
        }
    }
    
    // compute the dual of tmp1
    for (long i = 0; i < CSD; i++) {
        tmp2[i][i] = 1.0 / tmp1[i][i];
        mul_avx2(tmp1[i], tmp2[i][i], CSD);
    }
    for (long i = CSD - 1; i > 0; i--) {
        for (long j = 0; j < i; j++) {
            red_avx2(tmp2[j], tmp2[i], tmp1[j][i], CSD);
        }
    }

    // size reduce tmp2 while computing its dual
    for (long i = 0; i < CSD; i++) {
        mul_avx2(tmp1[i], 1.0 / tmp2[i][i], CSD);
    }
    for (long i = 0; i < CSD; i++) {
        for (long j = 0; j < i; j++) {
            double x = tmp1[i][j];
            tmp1[i][j] = tmp1[j][i];
            tmp1[j][i] = x;
        }
    }
    for (long i = 1; i < CSD; i++) {
        float x = 1.0 / tmp2[i][i];
        for (long j = 0; j < i; j++) {
            double y = round(tmp2[j][i] * x);
            if (fabs(y) > 1e-3) {
                red_avx2(tmp2[j], tmp2[i], y, CSD);
                red_avx2(tmp1[i], tmp1[j], -y, CSD);
            }
        }
    }

    /*uint32_t h = 1;
    for (long i = 1; i <= 30; i++) {
        for (long j = i; j <= 30; j++) {
            h += *((uint32_t*)(&tmp1[CSD-i][CSD-j]));
            h *= *((uint32_t*)(&tmp1[CSD-i][CSD-j]));
        }
    }
    printf("%x\n", h);*/

    /*for (long i = 0; i < CSD; i++) {
        printf("%f ", dot_avx2(tmp2[i], tmp2[i], CSD));
    }
    printf("\n");
    for (long i = 0; i < CSD; i++) {
        for (long j = 0; j < CSD; j++){
            _b_local[i][j] = dot_avx2(tmp1[i], tmp2[j], CSD);
            if (fabs(_b_local[i][j]) < 1e-14) _b_local[i][j] = 0.0;
        }
    }
    PRINT_MAT(_b_local, CSD, CSD);*/
    
    // adjust b_local and b_dual according to _ratio
    double mdd = 0.0;
    // so we don't allow basis vectors to enter the pool, the uid
    // of these vectors should always be inserted, same as 0.
    _ratio = 254.0 / _ratio * POOL_EPI8_RATIO_ADJ;
    if (ratio != 0.0f) _ratio = ratio;
    for (long i = 0; i < CSD; i++) {
        mul_avx2(tmp1[i], _ratio, CSD);
        mul_avx2(tmp2[i], 1.0/_ratio, CSD);
        for (long j = 0; j < CSD; j++) _b_local[i][j] = tmp1[i][j];
        if (tmp2[i][i] > mdd) mdd = tmp2[i][i];
    }
    _dhalf = floor(127.0 / mdd);
    _dshift = 31 - __builtin_clz(_dhalf);
    _dhalf = 1 << (_dshift-1);
    if (_dhalf < 1) {
        fprintf(stderr, "[Error] Pool_epi8_t<%u>::_update_b_local: _dhalf < 1, the basis too coarse?", nb);
    }
    mdd = ((_dhalf << 1) + 0.0);

    for (long i = 0; i < CSD; i++) {
        mul_avx2(tmp2[i], mdd, CSD);
        for (long j = 0; j < CSD; j++) {
            _b_dual[i*FD8+j] = round(tmp2[i][j] + 128.0);
        }
    }

    /*PRINT_MAT(tmp1, CSD, CSD);
    PRINT_MAT(tmp2, CSD, CSD);
    PRINT_MAT((int)_b_dual, CSD, CSD);
    for (long i = 0; i < CSD; i++) {
        printf("%f ", sqrt(dot_avx2(tmp1[i], tmp1[i], CSD)));
    }
    printf("\n");
    for (long i = 0; i < CSD; i++) {
        printf("%f ", sqrt(dot_avx2(tmp2[i], tmp2[i], CSD)));
    }
    printf("\n");*/

    FREE_MAT(tmp1);
    FREE_MAT(tmp2);
    return 1;
}

template<uint32_t nb>
float **Pool_epi8_t<nb>::_compute_b_local(long ind_l, long ind_r) {
    // protect old values
    long _tmp_index_l = index_l;
    long _tmp_index_r = index_r;
    long _tmp_CSD = CSD;
    float **_tmp_b_local = _b_local;
    uint8_t *_tmp_b_dual = _b_dual;
    float _tmp_ratio = _ratio;
    int32_t _tmp_dhalf = _dhalf;
    int32_t _tmp_dshift = _dshift;
    _b_local = NULL;
    _b_dual = NULL;
    
    // so we are safe if ind_r = index_r
    index_l = ind_l;
    index_r = ind_r;
    CSD = ind_r - ind_l;
    _update_b_local(_tmp_ratio);

    FREE_VEC((void *) _b_dual);
    float **ret = _b_local;

    // recover all the values
    index_l = _tmp_index_l;
    index_r = _tmp_index_r;
    CSD = _tmp_CSD;
    _b_local = _tmp_b_local;
    _b_dual = _tmp_b_dual;
    _ratio = _tmp_ratio;
    _dhalf = _tmp_dhalf;
    _dshift = _tmp_dshift;
    
    // we should be very careful because we use O3 and every thing 
    // in template class is inline, see the story of quad_float
    if ((ind_r == index_r) && (ind_l <= index_l)) {
        uint32_t h1 = 0;
        uint32_t h2 = 0;
        long b = index_l - ind_l;
        for (long i = 0; i < CSD; i++) {
            for (long j = 0; j <= i; j++) {
                h1 += *((uint32_t *)(&_b_local[i][j]));
                h1 *= *((uint32_t *)(&_b_local[i][j]));
                h2 += *((uint32_t *)(&ret[i+b][j+b]));
                h2 *= *((uint32_t *)(&ret[i+b][j+b]));
            }
        }
        if (h1 != h2) {
            fprintf(stderr, "[Error] Pool_epi8_t<%u>::_compute_b_local: Hash value check fail, unexpected things may happen, ignored.\n", nb);
        }
    }

    return ret;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::_compute_gh2() {
    double *B = basis->get_B().hi;
    double detn2 = 1.0;
    for (long i = index_l; i < index_r; i++){
        detn2 *= pow(B[i], 1.0/CSD);
    }
    gh2 = detn2 * pow(gh_coeff(CSD), 2.0);
    return 1;
}

template<uint32_t nb>
int Pool_epi8_t<nb>::_sampling(int8_t *dst, uint16_t *cdst, int32_t *dst_norm, int32_t *dst_sum, uint64_t *dst_u, DGS1d *R) {
    __attribute__ ((aligned (32))) float res[vec_length];
    int coeff[vec_length];
    double sigma2 = _b_local[CSD/2][CSD/2] * _b_local[CSD/2][CSD/2];
    long cnt = 0;
    do {
        set_zero_avx2(res, vec_length);
        for (long i = CSD - 1; i >= 0; i--) {
            do {
                coeff[i] = R->discrete_gaussian(-res[i]/_b_local[i][i],sigma2/(_b_local[i][i] * _b_local[i][i]) + 0.1);
            } while (fabsf(res[i] + coeff[i] * _b_local[i][i]) > 127.4f);
            red_avx2(res, _b_local[i], -coeff[i], CSD);
        }
        *dst_u = 0;
        for (long i = 0; i < CSD; i++){
            *dst_u += coeff[i] * uid->uid_coeffs[i];
        }
        cnt++;
        if (cnt > 30) {
            fprintf(stderr, "[Error] Pool_epi8_t<%u>::_sampling: sampling always get collision, aborted.\n", nb);
            return -1;
        }
    } while (!uid->insert_uid(*dst_u));
    for (long i = 0; i < CSD; i++) dst[i] = roundf(res[i]);
    *dst_sum = 0;
    for (long i = 0; i < CSD; i++) *dst_sum += dst[i];
    *dst_sum *= 128;
    *dst_norm = roundf(0.5 * dot_avx2(res, res, CSD));
    ((uint32_t *)cdst)[0] = (long)(dst - vec) / vec_length;
    uint32_t x = (*dst_norm) >> 1;
    cdst[2] = (x > 65535) ? 65535 : x;
    //printf("[vec %u] u = %lu, 128sum = %d, norm = %u, cnorm = %u\n", ((uint32_t *)cdst)[0], *dst_u, *dst_sum, *dst_norm, cdst[2]);
    //PRINT_VEC((int)dst, CSD);
    return 1;
}

template<uint32_t nb>
void Pool_epi8_t<nb>::_reconstruct_all_cvec() {
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        for (long cind = begin_ind; cind < end_ind; cind++) {
            int32_t cnorm = vnorm[cind] >> 1;
            *((uint32_t *)(cvec+3LL*cind)) = cind;
            cvec[3LL*cind+2LL] = (cnorm > 65535) ? 65535 : cnorm;
        }
    }

    sorted_index = 0;
}


template<uint32_t nb>
int Pool_epi8_t<nb>::__basis_insert(long dst_index, float *v_fp, long FD, float **b_full_fp) {
    // prepare the new vec
    const long RD = index_l - dst_index;
    VEC_QP v_QP = NEW_VEC_QP(basis->NumCols());
    MAT_QP b_QP = basis->get_b();
    int32_t v_normal_coeff[vec_length];

    long rm_index;
    do {
        for (long i = 0; i < RD+CSD; i++) {
            if (i < CSD) v_normal_coeff[CSD-i-1] = round( v_fp[FD - i - 1] / b_full_fp[FD - i - 1][FD - i - 1] );
            NTL::quad_float q(round(-v_fp[FD - i - 1] / b_full_fp[FD - i - 1][FD - i - 1]));
            red_avx2(v_fp, b_full_fp[FD-i-1], -q.hi, FD);
            red(v_QP.hi, v_QP.lo, b_QP.hi[index_r-i-1], b_QP.lo[index_r-i-1], q, basis->NumCols());
        }
        
        
        // compute rm_index
        for (long i = 0; i < CSD; i++) {
            if (v_normal_coeff[i] == 1) {
                rm_index = i + index_l;
                break;
            }
            if (v_normal_coeff[i] == -1) {
                rm_index = i + index_l;
                mul(v_QP, NTL::quad_float(-1.0), basis->NumCols());
                for (long j = 0; j < CSD; j++) v_normal_coeff[j] = -v_normal_coeff[j];
                break;
            }
            if (i == CSD - 1) {
                fprintf(stderr, "[Error] Pool_epi8_t<%u>::__pool_insert: no 1 or -1 in coeff, something must be wrong, aborted.\n", nb);
                shrink_left();
                FREE_VEC_QP(v_QP);
                return -1;
            }
        }
    } while (0);

    // process the basis
    for (long i = rm_index; i > dst_index; i--) {
        copy(b_QP.hi[i], b_QP.lo[i], b_QP.hi[i-1], b_QP.lo[i-1], basis->NumCols());
    }
    copy(b_QP.hi[dst_index], b_QP.lo[dst_index], v_QP.hi, v_QP.lo, basis->NumCols());
    FREE_VEC_QP(v_QP);
    basis->compute_gso_QP();
    basis->size_reduce(dst_index);

    // keeping old b_local data
    uint8_t *_b_dual_old = _b_dual;
    float **_b_local_old = _b_local;
    int32_t _dhalf_old = _dhalf;
    int32_t _dshift_old = _dshift;
    _b_dual = NULL;
    _b_local = NULL;
    float old_ratio = _ratio;

    // update b_local data
    index_l++;
    CSD--;
    _update_b_local();
    _compute_gh2();
    uid->reset_hash_function(CSD);
    bgj_reserve_uid_table(uid, _pool_size);
    for (long i = 0; i < CSD; i++) uid->insert_uid(uid->uid_coeffs[i]);

    // b_trans and u_trans
    float **b_trans = (float **) NEW_MAT(CSD+1, vec_length, sizeof(float));
    uint64_t u_trans[vec_length]={};
    do {
        float **b_normal_new = (float **) NEW_MAT(CSD, vec_length, sizeof(float));
        float **b_normal_old = (float **) NEW_MAT(CSD+1, vec_length, sizeof(float));
        for (long j = 0; j < CSD; j++) {
            float x = sqrt((basis->get_B()).hi[j+index_l]) * _ratio;
            for (long i = j; i < CSD; i++) {
                b_normal_new[i][j] = (basis->get_miu()).hi[i+index_l][j+index_l] * x;
            }
        }
        for (long j = 0; j < CSD + 1; j++) {
            for (long i = 0; i <= j; i++) {
                b_normal_old[j][i] = b_full_fp[j+FD-CSD-1][i+FD-CSD-1] * old_ratio;
            }
        }

        int32_t old_coeff[vec_length];
        __attribute__ ((aligned (32))) float tmp[vec_length];
        for (long i = 0; i < CSD+1; i++) {
            copy_avx2(tmp, _b_local_old[i], vec_length);
            for (long j = CSD; j >= 0; j--) {
                old_coeff[j] = round(tmp[j]/b_normal_old[j][j]);
                red_avx2(tmp, b_normal_old[j], old_coeff[j], vec_length);
            }

            long c = old_coeff[rm_index - (index_l-1)];
            for (long j = 0; j <= CSD; j++) {
                old_coeff[j] -= c * v_normal_coeff[j];
            }
            for (long j = rm_index - (index_l-1); j < CSD; j++) old_coeff[j] = old_coeff[j+1];
            for (long j = 0; j < CSD; j++) {
                red_avx2(b_trans[i], b_normal_new[j], -old_coeff[j], vec_length);
            }

            copy_avx2(tmp, b_trans[i], vec_length);
            for (long j = CSD-1; j >=0; j--) {
                int32_t c = round(tmp[j]/_b_local[j][j]);
                red_avx2(tmp, _b_local[j], c, vec_length);
                u_trans[i] += c * uid->uid_coeffs[j];
            }
        }
        FREE_MAT(b_normal_old);
        FREE_MAT(b_normal_new);
    } while (0);

    // some vector will fail, so we record them here
    INIT_COLLISION_LIST;

    // compute new vec from old vec
    #pragma omp parallel for
    for (long thread = 0; thread < num_threads; thread++) {
        long begin_ind = (thread * num_vec) / num_threads;
        long end_ind = ((thread+1) * num_vec) / num_threads;

        __attribute__ ((aligned (32))) float ftmp[vec_length * 8];
        __attribute__ ((aligned (32))) int32_t ctmp[8 * vec_length];
        long ind = begin_ind;
        while (ind < end_ind - 7) {
            // recover old coeff
            _compute_coeff_b8(ctmp, ind, CSD+1, _b_dual_old, _dhalf_old, _dshift_old);

            // compute new vec
            set_zero_avx2(ftmp, vec_length * 8);
            for (long i = 0; i < CSD+1; i++) {
                red_avx2(ftmp+vec_length*0, b_trans[i], -ctmp[i*8+0], vec_length);
                red_avx2(ftmp+vec_length*1, b_trans[i], -ctmp[i*8+1], vec_length);
                red_avx2(ftmp+vec_length*2, b_trans[i], -ctmp[i*8+2], vec_length);
                red_avx2(ftmp+vec_length*3, b_trans[i], -ctmp[i*8+3], vec_length);
                red_avx2(ftmp+vec_length*4, b_trans[i], -ctmp[i*8+4], vec_length);
                red_avx2(ftmp+vec_length*5, b_trans[i], -ctmp[i*8+5], vec_length);
                red_avx2(ftmp+vec_length*6, b_trans[i], -ctmp[i*8+6], vec_length);
                red_avx2(ftmp+vec_length*7, b_trans[i], -ctmp[i*8+7], vec_length);
            }

            // compute u
            uint64_t u[8] = {};
            for (long i = 0; i < CSD+1; i++) {
                u[0] += ctmp[i*8+0]*u_trans[i];
                u[1] += ctmp[i*8+1]*u_trans[i];
                u[2] += ctmp[i*8+2]*u_trans[i];
                u[3] += ctmp[i*8+3]*u_trans[i];
                u[4] += ctmp[i*8+4]*u_trans[i];
                u[5] += ctmp[i*8+5]*u_trans[i];
                u[6] += ctmp[i*8+6]*u_trans[i];
                u[7] += ctmp[i*8+7]*u_trans[i];
            }
            for (long i = 0; i < 8; i++) vu[ind+i] = u[i];
            for (long i = 0; i < 8; i++) {
                int overflow = 0;
                for (long j = 0; j < CSD; j++) {
                    int32_t val = roundf(ftmp[i*vec_length+j]);
                    (vec + vec_length * (ind+i))[j] = val;
                    if (abs(val) > 127) overflow = 1;
                }
                (vec + vec_length * (ind+i))[CSD] = 0;
                if (overflow) {
                    ADD_TO_COLLISION_LIST(ind+i);
                    continue;
                }
                if (!uid->insert_uid(u[i])) ADD_TO_COLLISION_LIST(ind+i);
            }

            COMPUTE_VNORM_AND_VSUM_B8;

            ind += 8;
        }

        while (ind < end_ind) {
            // recover old coeff
            _compute_coeff(ctmp, ind, CSD+1, _b_dual_old, _dhalf_old, _dshift_old);

            // compute new vec
            set_zero_avx2(ftmp, vec_length * 8);
            for (long i = 0; i < CSD; i++) {
                red_avx2(ftmp, b_trans[i], -ctmp[i], vec_length);
            }
            int overflow = 0;
            for (long i = 0; i < CSD; i++) {
                int32_t val = roundf(ftmp[i]);
                (vec + vec_length * ind)[i] = val;
                if (fabs(val) > 127) overflow = 1;
            }
            (vec + vec_length * ind)[CSD] = 0;
            if (overflow) ADD_TO_COLLISION_LIST(ind);

            // compute u
            uint64_t u = 0;
            for (long i = 0; i < CSD+1; i++) {
                u += ctmp[i]*u_trans[i];
            }
            vu[ind] = u;
            if (!overflow) {
                if (!uid->insert_uid(u)) ADD_TO_COLLISION_LIST(ind);
            }

            // compute norm and sum
            COMPUTE_VNORM_AND_VSUM;

            ind++;
        }
    }

    // process uid failed and overflowed vectors
    PROCESS_COLLISION_LIST;

    _reconstruct_all_cvec();
    FREE_VEC((void *)_b_dual_old);
    FREE_MAT(_b_local_old);
    return 1;
}

#if COMPILE_POOL_EPI8_96
template class Pool_epi8_t<3>;
#endif

#if COMPILE_POOL_EPI8_128
template class Pool_epi8_t<4>;
#endif

#if COMPILE_POOL_EPI8_160
template class Pool_epi8_t<5>;
#endif

#if COMPILE_POOL_EPI8_192
template class Pool_epi8_t<6>;
#endif

#if COMPILE_POOL_EPI8_224
template class Pool_epi8_t<7>;
#endif
