#include "../include/quad.h"
#include "../include/vec.h"
#include <immintrin.h>


/* naive vec operations, n can be any integer, data need not to be aligned. */
NTL::quad_float dot(NTL::quad_float *src1, NTL::quad_float *src2, long n){
    NTL::quad_float ret(0.0);
    for (long i = 0; i < n; i++){
        ret += src1[i] * src2[i];
    }
    return ret;
}
void red(NTL::quad_float *dst, NTL::quad_float *src, NTL::quad_float q, long n){
    for (long i = 0; i < n; i++){
        dst[i] -= q * src[i];
    }
}
void copy(NTL::quad_float *dst, NTL::quad_float *src, long n){
    for (long i = 0; i < n; i++){
        dst[i] = src[i];
    }
}
void mul(NTL::quad_float *dst, NTL::quad_float q, long n){
    for (long i = 0; i < n; i++){
        dst[i] *= q;
    }
}
void sub(NTL::quad_float *dst, NTL::quad_float q, long n){
    for (long i = 0; i < n; i++){
        dst[i] -= q;
    }
}


/* aligned vec operations, n should be divisible by 8 and pointers 64-byte
 * aligned. The AVX-512 versions transcribe NTL quad_float's operator
 * sequences (Dekker split multiply, Knuth-style addition) lane for lane, so
 * every element matches the scalar path bit for bit; only dot's summation
 * order differs (eight partial sums reduced pairwise), as in any vector dot.
 * The previous inline-assembly versions assumed argument registers survive
 * to the asm block, which modern optimizing builds do not guarantee. */
#if defined(__AVX512F__)

#define BGJ_QF_SPLIT 134217729.0    /* 2^27 + 1, == NTL_QUAD_FLOAT_SPLIT */

static inline void bgj_qf_add8(__m512d &zh, __m512d &zl,
                               __m512d xh, __m512d xl,
                               __m512d yh, __m512d yl)
{
    __m512d S = _mm512_add_pd(xh, yh);
    __m512d T = _mm512_add_pd(xl, yl);
    __m512d e = _mm512_sub_pd(S, xh);
    __m512d f = _mm512_sub_pd(T, xl);
    __m512d t1 = _mm512_sub_pd(S, e);
    t1 = _mm512_sub_pd(xh, t1);
    __m512d s = _mm512_sub_pd(yh, e);
    s = _mm512_add_pd(s, t1);
    t1 = _mm512_sub_pd(T, f);
    t1 = _mm512_sub_pd(xl, t1);
    __m512d t = _mm512_sub_pd(yl, f);
    t = _mm512_add_pd(t, t1);
    s = _mm512_add_pd(s, T);
    __m512d H = _mm512_add_pd(S, s);
    __m512d h = _mm512_sub_pd(S, H);
    h = _mm512_add_pd(h, s);
    h = _mm512_add_pd(h, t);
    zh = _mm512_add_pd(H, h);
    __m512d f2 = _mm512_sub_pd(H, zh);
    zl = _mm512_add_pd(f2, h);
}

static inline void bgj_qf_sub8(__m512d &zh, __m512d &zl,
                               __m512d xh, __m512d xl,
                               __m512d yh, __m512d yl)
{
    const __m512d sign = _mm512_set1_pd(-0.0);
    bgj_qf_add8(zh, zl, xh, xl,
                _mm512_xor_pd(yh, sign), _mm512_xor_pd(yl, sign));
}

static inline void bgj_qf_mul8(__m512d &zh, __m512d &zl,
                               __m512d xh, __m512d xl,
                               __m512d yh, __m512d yl)
{
    const __m512d SPL = _mm512_set1_pd(BGJ_QF_SPLIT);
    __m512d C = _mm512_mul_pd(SPL, xh);
    __m512d hx = _mm512_sub_pd(C, xh);
    __m512d c = _mm512_mul_pd(SPL, yh);
    hx = _mm512_sub_pd(C, hx);
    __m512d tx = _mm512_sub_pd(xh, hx);
    __m512d hy = _mm512_sub_pd(c, yh);
    C = _mm512_mul_pd(xh, yh);
    hy = _mm512_sub_pd(c, hy);
    __m512d ty = _mm512_sub_pd(yh, hy);
    __m512d t1 = _mm512_mul_pd(hx, hy);
    t1 = _mm512_sub_pd(t1, C);
    __m512d t2 = _mm512_mul_pd(hx, ty);
    t1 = _mm512_add_pd(t1, t2);
    t2 = _mm512_mul_pd(tx, hy);
    t1 = _mm512_add_pd(t1, t2);
    t2 = _mm512_mul_pd(tx, ty);
    __m512d cc = _mm512_add_pd(t1, t2);
    t1 = _mm512_mul_pd(xh, yl);
    t2 = _mm512_mul_pd(xl, yh);
    t1 = _mm512_add_pd(t1, t2);
    cc = _mm512_add_pd(cc, t1);
    zh = _mm512_add_pd(C, cc);
    __m512d rl = _mm512_sub_pd(C, zh);
    zl = _mm512_add_pd(rl, cc);
}

#endif

NTL::quad_float dot(double *src1_hi, double *src1_lo, double *src2_hi, double *src2_lo, long n){
#if defined(__AVX512F__)
    const long n8 = n & ~7L;
    NTL::quad_float ret(0.0);
    if (n8) {
        __m512d acc_hi = _mm512_setzero_pd();
        __m512d acc_lo = _mm512_setzero_pd();
        for (long i = 0; i < n8; i += 8) {
            __m512d ph, pl;
            bgj_qf_mul8(ph, pl,
                        _mm512_load_pd(src1_hi + i), _mm512_load_pd(src1_lo + i),
                        _mm512_load_pd(src2_hi + i), _mm512_load_pd(src2_lo + i));
            bgj_qf_add8(acc_hi, acc_lo, acc_hi, acc_lo, ph, pl);
        }
        __attribute__ ((aligned (64))) double lane_hi[8];
        __attribute__ ((aligned (64))) double lane_lo[8];
        _mm512_store_pd(lane_hi, acc_hi);
        _mm512_store_pd(lane_lo, acc_lo);
        for (long i = 0; i < 8; i++) {
            ret += NTL::quad_float(lane_hi[i], lane_lo[i]);
        }
    }
    for (long i = n8; i < n; i++) {
        ret += NTL::quad_float(src1_hi[i], src1_lo[i]) * NTL::quad_float(src2_hi[i], src2_lo[i]);
    }
    return ret;
#else
    NTL::quad_float ret(0.0);
    for (long i = 0; i < n; i++){
        ret += NTL::quad_float(src1_hi[i], src1_lo[i]) * NTL::quad_float(src2_hi[i], src2_lo[i]);
    }
    return ret;
#endif
}
void red(double *dst_hi, double *dst_lo, double *src_hi, double *src_lo, NTL::quad_float q, long n){
#if defined(__AVX512F__)
    const long n8 = n & ~7L;
    const __m512d qh = _mm512_set1_pd(q.hi);
    const __m512d ql = _mm512_set1_pd(q.lo);
    for (long i = 0; i < n8; i += 8) {
        __m512d ph, pl, rh, rl;
        bgj_qf_mul8(ph, pl, qh, ql,
                    _mm512_load_pd(src_hi + i), _mm512_load_pd(src_lo + i));
        bgj_qf_sub8(rh, rl,
                    _mm512_load_pd(dst_hi + i), _mm512_load_pd(dst_lo + i), ph, pl);
        _mm512_store_pd(dst_hi + i, rh);
        _mm512_store_pd(dst_lo + i, rl);
    }
    for (long i = n8; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x -= q * NTL::quad_float(src_hi[i], src_lo[i]);
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
#else
    for (long i = 0; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x -= q * NTL::quad_float(src_hi[i], src_lo[i]);
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
#endif
}
void copy(double *dst_hi, double *dst_lo, double *src_hi, double *src_lo, long n){
    copy(dst_hi, src_hi, n);
    copy(dst_lo, src_lo, n);
}
void mul(double *dst_hi, double *dst_lo, NTL::quad_float q, long n){
#if defined(__AVX512F__)
    const long n8 = n & ~7L;
    const __m512d qh = _mm512_set1_pd(q.hi);
    const __m512d ql = _mm512_set1_pd(q.lo);
    for (long i = 0; i < n8; i += 8) {
        __m512d rh, rl;
        bgj_qf_mul8(rh, rl,
                    _mm512_load_pd(dst_hi + i), _mm512_load_pd(dst_lo + i), qh, ql);
        _mm512_store_pd(dst_hi + i, rh);
        _mm512_store_pd(dst_lo + i, rl);
    }
    for (long i = n8; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x *= q;
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
#else
    for (long i = 0; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x *= q;
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
#endif
}
void sub(double *dst_hi, double *dst_lo, NTL::quad_float q, long n){
#if defined(__AVX512F__)
    const long n8 = n & ~7L;
    const __m512d qh = _mm512_set1_pd(q.hi);
    const __m512d ql = _mm512_set1_pd(q.lo);
    for (long i = 0; i < n8; i += 8) {
        __m512d rh, rl;
        bgj_qf_sub8(rh, rl,
                    _mm512_load_pd(dst_hi + i), _mm512_load_pd(dst_lo + i), qh, ql);
        _mm512_store_pd(dst_hi + i, rh);
        _mm512_store_pd(dst_lo + i, rl);
    }
    for (long i = n8; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x -= q;
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
#else
    for (long i = 0; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x -= q;
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
#endif
}

/* interleaved VEC_QP wrappers */

NTL::quad_float dot(VEC_QP src1, VEC_QP src2, long n){
    return dot(src1.hi, src1.lo, src2.hi, src2.lo, n);
}
void red(VEC_QP dst, VEC_QP src, NTL::quad_float q, long n){
    red(dst.hi, dst.lo, src.hi, src.lo, q, n);
}
void copy(VEC_QP dst, VEC_QP src, long n){
    copy(dst.hi, dst.lo, src.hi, src.lo, n);
}
void mul(VEC_QP dst, NTL::quad_float q, long n){
    mul(dst.hi, dst.lo, q, n);
}
void sub(VEC_QP dst, NTL::quad_float q, long n){
    sub(dst.hi, dst.lo, q, n);
}