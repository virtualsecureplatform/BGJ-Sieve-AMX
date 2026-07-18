#ifndef __BGJ_CUDA_H
#define __BGJ_CUDA_H

#include <stdint.h>

enum bgj_cuda_sol_type {
    BGJ_CUDA_SOL_A = 0,
    BGJ_CUDA_SOL_S = 1,
    BGJ_CUDA_SOL_AA = 2,
    BGJ_CUDA_SOL_SA = 3,
    BGJ_CUDA_SOL_SS = 4
};

struct bgj_cuda_result_t {
    uint32_t type;
    uint32_t x;
    uint32_t y;
};

struct bgj_cuda_lsh_lift_result_t {
    uint32_t candidate;
    uint32_t place;
    float norm;
};

typedef int (*bgj_cuda_lsh_result_callback_t)(void *ctx,
                                              const bgj_cuda_result_t *results,
                                              uint32_t result_count,
                                              uint64_t tile_slot_begin,
                                              uint64_t tile_slot_count);

struct bgj_cuda_materialize_desc_t {
    uint32_t type;
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct bgj_cuda_materialize_phase_profile_t {
    uint64_t chunks;
    uint64_t candidates;
    double pool_sec;
    double basis_sec;
    double desc_sec;
    double build_sec;
    double gemm_sec;
    double coeff_sec;
    double reconstruct_sec;
    double copy_sec;
};

struct alignas(16) bgj_cuda_bucket_entry_t {
    uint32_t bucket;
    uint32_t id;
    int32_t dot;
    uint32_t reserved;
};

// Sub-bucketing output entry: index is the position in the main bucket's
// p-list or n-list; meta packs the center slot in bits 0..15 and the side
// (1 = positive list) in bit 16.
struct bgj_cuda_subb_entry_t {
    uint32_t index;
    uint32_t meta;
};

#define BGJ_CUDA_SUBB_META_CIND(meta) ((meta) & 0xffffu)
#define BGJ_CUDA_SUBB_META_P(meta) (((meta) >> 16) & 1u)

int bgj_cuda_device_count();
int bgj_cuda_execution_device_count();
const char *bgj_cuda_last_error();
int bgj_cuda_search_requested();
void bgj_cuda_set_search_requested(int enabled);
uint32_t bgj_cuda_batch_size(uint32_t host_threads);
uint64_t bgj_cuda_batch_min_dots();
uint32_t bgj_cuda_result_capacity();
int bgj_cuda_cred_transform_requested();
int bgj_cuda_overlap_cred_requested();
int bgj_cuda_materialize_requested();
int bgj_cuda_bucket_requested();

extern "C" int bgj_cuda_bucket_bgj1_raw(const int8_t *pool_vecs,
                                         uint64_t pool_epoch,
                                         uint32_t pool_size,
                                         const uint32_t *center_ids,
                                         uint32_t num_centers,
                                         const int32_t *vnorm,
                                         uint32_t vec_length,
                                         uint32_t alpha_x2_u16,
                                         bgj_cuda_bucket_entry_t *entries,
                                         uint32_t entry_capacity,
                                         uint32_t *entry_count,
                                         int *overflow);

extern "C" int bgj_cuda_lsh_search_raw(const uint8_t *sh,
                                        uint32_t mbound,
                                        uint32_t shsize,
                                        int32_t threshold,
                                        bgj_cuda_result_t *results,
                                        uint32_t result_capacity,
                                        uint32_t *result_count,
                                        int *overflow);
extern "C" uint64_t bgj_cuda_lsh_total_tile_slots(uint32_t mbound);
extern "C" int bgj_cuda_lsh_search_range_raw(const uint8_t *sh,
                                              uint32_t mbound,
                                              uint32_t shsize,
                                              int32_t threshold,
                                              uint64_t tile_slot_begin,
                                              uint64_t tile_slot_count,
                                              bgj_cuda_result_t *results,
                                              uint32_t result_capacity,
                                              uint32_t *result_count,
                                              int *overflow);
extern "C" int bgj_cuda_lsh_search_stream_raw(const uint8_t *sh,
                                               uint32_t mbound,
                                               uint32_t shsize,
                                               int32_t threshold,
                                               uint64_t chunk_tile_slots,
                                               bgj_cuda_result_t *results,
                                               uint32_t result_capacity,
                                               uint64_t *total_result_count,
                                               int *overflow,
                                               bgj_cuda_lsh_result_callback_t callback,
                                               void *callback_ctx);
extern "C" int bgj_cuda_lsh_lift_raw(const float *fvec,
                                      uint32_t mbound,
                                      uint32_t fd,
                                      uint32_t fd8,
                                      const uint32_t *candidates,
                                      uint32_t candidate_count,
                                      const float *b_full,
                                      const float *idiag,
                                      const float *min_norm,
                                      uint32_t id,
                                      uint32_t csd,
                                      float threshold_scale,
                                      bgj_cuda_lsh_lift_result_t *results,
                                      uint32_t result_capacity,
                                      uint32_t *result_count,
                                      int *overflow);
extern "C" int bgj_cuda_search_bucket_pool_batch_np_flat_raw(const int8_t *pool_vecs,
                                                              uint64_t pool_epoch,
                                                              uint32_t pool_size,
                                                              const uint32_t *p_ids,
                                                              const uint32_t *n_ids,
                                                              const int32_t *p_norm,
                                                              const int32_t *n_norm,
                                                              const uint32_t *p_offsets,
                                                              const uint32_t *n_offsets,
                                                              const uint32_t *num_p,
                                                              const uint32_t *num_n,
                                          uint32_t batch_size,
                                          uint32_t vec_length,
                                          int32_t goal_norm,
                                          int include_same_pairs,
                                          bgj_cuda_result_t *const *results,
                                          const uint32_t *result_capacity,
                                          uint32_t *result_count,
                                          int *overflow);

// GPU LSH m-block collect: for every pool vector, reconstruct the integer
// coefficients against b_dual, expand to the float representation with b_ext,
// Babai-reduce head +/- center against b_head, and append passing vectors'
// fvec rows to the per-center host buffers. Reproduces the arithmetic of
// lsh_mblock_provider_t::batch_pop's collect loop (same FMA order).
extern "C" int bgj_cuda_lsh_collect_raw(const int8_t *pool_vecs,
                                         uint64_t pool_epoch,
                                         uint32_t pool_size,
                                         const int32_t *vsum,
                                         uint32_t num_vec,
                                         uint32_t vec_length,
                                         uint32_t csd,
                                         const uint8_t *b_dual,
                                         int32_t dhalf,
                                         int32_t dshift,
                                         const float *b_ext,
                                         uint32_t fd,
                                         uint32_t fd8,
                                         const float *b_head,
                                         const float *idiag,
                                         uint32_t dh_dim,
                                         uint32_t dh_stride,
                                         uint32_t id_off,
                                         const float *centers,
                                         uint32_t num_c,
                                         float radius2,
                                         uint32_t adjust_sample,
                                         double adjust_expect,
                                         float *radius2_out,
                                         uint32_t max_per_center,
                                         float *const *fvec_out,
                                         uint32_t *mbound_out);

extern "C" int bgj_cuda_sub_bucketing_raw(const int8_t *pool_vecs,
                                           uint64_t pool_epoch,
                                           uint32_t pool_size,
                                           const uint32_t *p_ids,
                                           const int32_t *p_norm,
                                           uint32_t num_p,
                                           const uint32_t *n_ids,
                                           const int32_t *n_norm,
                                           uint32_t num_n,
                                           const uint32_t *center_ids,
                                           const int32_t *center_b2,
                                           uint32_t num_centers,
                                           uint32_t vec_length,
                                           uint32_t alpha3x2_u16,
                                           uint32_t alpha2x2_u16,
                                           bgj_cuda_subb_entry_t *entries3,
                                           uint32_t cap3,
                                           uint32_t *count3,
                                           bgj_cuda_subb_entry_t *entries2,
                                           uint32_t cap2,
                                           uint32_t *count2,
                                           bgj_cuda_subb_entry_t *sols,
                                           uint32_t sol_cap,
                                           uint32_t *sol_count,
                                           int *overflow);

extern "C" int bgj_cuda_materialize_sol_list_raw(const int8_t *pool_vecs,
                                                  uint64_t pool_epoch,
                                                  uint32_t pool_size,
                                                  uint32_t vec_length,
                                                  const bgj_cuda_materialize_desc_t *desc,
                                                  uint32_t count,
                                                  const uint8_t *b_dual,
                                                  const float *b_local,
                                                  uint32_t csd,
                                                  int32_t dhalf,
                                                  int32_t dshift,
                                                  int8_t *dst_vec,
                                                  int32_t *dst_vnorm,
                                                  int32_t *dst_vsum);
extern "C" int bgj_cuda_materialize_sol_list_staged_raw(const int8_t *pool_vecs,
                                                         uint64_t pool_epoch,
                                                         uint32_t pool_size,
                                                         uint32_t vec_length,
                                                         const bgj_cuda_materialize_desc_t *desc,
                                                         uint32_t count,
                                                         const uint8_t *b_dual,
                                                         const float *b_local,
                                                         uint32_t csd,
                                                         int32_t dhalf,
                                                         int32_t dshift,
                                                         int32_t *dst_vnorm,
                                                         int32_t *dst_vsum);
extern "C" int bgj_cuda_materialize_copy_staged_vectors_raw(const uint32_t *indices,
                                                             uint32_t count,
                                                             uint32_t vec_length,
                                                             int8_t *dst_vec);
extern "C" void bgj_cuda_materialize_finish_staged_raw();
extern "C" void bgj_cuda_materialize_last_profile(bgj_cuda_materialize_phase_profile_t *profile);

#endif
