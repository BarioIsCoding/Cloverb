// Fused attention for 64-token sequences (one chess board) on AVX-512 CPUs.
//
// For every (batch, head) pair the 64x64 score matrix lives in L1: scores, softmax and the
// weighted sum of values are computed without writing a B*H*64*64 tensor. The bias is
//     bias[b,h] = static[h] + sum_t coef[b,h,t] * templates[t]          (64x64 each)
// i.e. a learned per-head bias plus Cloverb's dynamic template mix (GAB-lite), fused in.
// The backward pass recomputes the scores and returns dq, dk, dv and the score gradient dS
// (fp32, [BH, 4096]); the bias gradients are then two dense GEMMs / one reduction in torch:
//     dstatic = sum_b dS,   dcoef = dS @ templates^T,   dtemplates = coef^T @ dS.
// q, k, v, o, do, dq, dk, dv are bf16 (uint16) or fp32 (flag `bf16`).
// Build: gcc -O3 -march=native -fopenmp -shared -fPIC -o libattn64.so attn64.c -lm
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define L 64

static inline __m512 exp512(__m512 x) {
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(1.5353362e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3398874e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.6181291e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504109e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022651e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

static inline __m512 load16(const void *base, int64_t idx, int bf16) {
    if (bf16) {
        __m256i a = _mm256_loadu_si256((const __m256i *)((const uint16_t *)base + idx));
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16));
    }
    return _mm512_loadu_ps((const float *)base + idx);
}

static inline void store16(void *base, int64_t idx, __m512 v, int bf16) {
    if (bf16) {  // round-to-nearest-even fp32 -> bf16
        __m512i u = _mm512_castps_si512(v);
        __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
        u = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb));
        _mm256_storeu_si256((__m256i *)((uint16_t *)base + idx), _mm512_cvtepi32_epi16(_mm512_srli_epi32(u, 16)));
    } else {
        _mm512_storeu_ps((float *)base + idx, v);
    }
}

// Y[i,:] = alpha * sum_j A[i,j] X[j,:]    A 64x64, X 64xD, Y 64xD (4-row register blocking)
static void mm_AX(float *Y, const float *A, const float *X, int D, float alpha) {
    int nv = D / 16;
    __m512 al = _mm512_set1_ps(alpha);
    for (int i = 0; i < L; i += 4) {
        __m512 acc[4][4];
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) acc[r][c] = _mm512_setzero_ps();
        for (int j = 0; j < L; j++) {
            __m512 a0 = _mm512_set1_ps(A[i * L + j]), a1 = _mm512_set1_ps(A[(i + 1) * L + j]);
            __m512 a2 = _mm512_set1_ps(A[(i + 2) * L + j]), a3 = _mm512_set1_ps(A[(i + 3) * L + j]);
            for (int c = 0; c < nv; c++) {
                __m512 x = _mm512_load_ps(X + j * D + 16 * c);
                acc[0][c] = _mm512_fmadd_ps(a0, x, acc[0][c]); acc[1][c] = _mm512_fmadd_ps(a1, x, acc[1][c]);
                acc[2][c] = _mm512_fmadd_ps(a2, x, acc[2][c]); acc[3][c] = _mm512_fmadd_ps(a3, x, acc[3][c]);
            }
        }
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < nv; c++) _mm512_store_ps(Y + (i + r) * D + 16 * c, _mm512_mul_ps(acc[r][c], al));
    }
}

// Y[j,:] = alpha * sum_i A[i,j] X[i,:]    (A^T X without a transpose: 4 output rows at a time)
static void mm_AtX(float *Y, const float *A, const float *X, int D, float alpha) {
    int nv = D / 16;
    __m512 al = _mm512_set1_ps(alpha);
    for (int j = 0; j < L; j += 4) {
        __m512 acc[4][4];
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) acc[r][c] = _mm512_setzero_ps();
        for (int i = 0; i < L; i++) {
            const float *ar = A + i * L + j;
            __m512 a0 = _mm512_set1_ps(ar[0]), a1 = _mm512_set1_ps(ar[1]), a2 = _mm512_set1_ps(ar[2]), a3 = _mm512_set1_ps(ar[3]);
            for (int c = 0; c < nv; c++) {
                __m512 x = _mm512_load_ps(X + i * D + 16 * c);
                acc[0][c] = _mm512_fmadd_ps(a0, x, acc[0][c]); acc[1][c] = _mm512_fmadd_ps(a1, x, acc[1][c]);
                acc[2][c] = _mm512_fmadd_ps(a2, x, acc[2][c]); acc[3][c] = _mm512_fmadd_ps(a3, x, acc[3][c]);
            }
        }
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < nv; c++) _mm512_store_ps(Y + (j + r) * D + 16 * c, _mm512_mul_ps(acc[r][c], al));
    }
}

// S[i,:] = alpha * sum_d P[i,d] Qt[d,:]  where Qt is D x 64 (i.e. S = alpha * P Q^T), 2 rows per pass
static void mm_XYt(float *S, const float *P, const float *Qt, int D, float alpha) {
    __m512 al = _mm512_set1_ps(alpha);
    for (int i = 0; i < L; i += 2) {
        __m512 a[2][4];
        for (int r = 0; r < 2; r++) for (int c = 0; c < 4; c++) a[r][c] = _mm512_setzero_ps();
        for (int d = 0; d < D; d++) {
            __m512 p0 = _mm512_set1_ps(P[i * D + d]), p1 = _mm512_set1_ps(P[(i + 1) * D + d]);
            const float *qr = Qt + d * L;
            for (int c = 0; c < 4; c++) {
                __m512 x = _mm512_load_ps(qr + 16 * c);
                a[0][c] = _mm512_fmadd_ps(p0, x, a[0][c]); a[1][c] = _mm512_fmadd_ps(p1, x, a[1][c]);
            }
        }
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 4; c++) _mm512_store_ps(S + (i + r) * L + 16 * c, _mm512_mul_ps(a[r][c], al));
    }
}

// load a 64xD block, optionally also its transpose (D x 64)
static void load_mat(float *dst, float *dstT, const void *src, int64_t off, int D, int bf16) {
    for (int i = 0; i < L; i++)
        for (int c = 0; c < D; c += 16) _mm512_store_ps(dst + i * D + c, load16(src, off + i * D + c, bf16));
    if (dstT) {
        // transpose via 16-element gathers on the freshly loaded row-major copy
        __m512i idx = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                         _mm512_set1_epi32(D));
        for (int d = 0; d < D; d++)
            for (int i = 0; i < L; i += 16)
                _mm512_store_ps(dstT + d * L + i, _mm512_i32gather_ps(idx, dst + i * D + d, 4));
    }
}

// bias + softmax in place on S (rows), lse optional
static void bias_softmax(float *S, const float *stat, const float *coef, const float *tmpl, int T) {
    for (int i = 0; i < L; i++) {
        float *sr = S + i * L;
        __m512 a[4];
        for (int c = 0; c < 4; c++) {
            a[c] = _mm512_load_ps(sr + 16 * c);
            if (stat) a[c] = _mm512_add_ps(a[c], _mm512_loadu_ps(stat + i * L + 16 * c));
        }
        for (int t = 0; t < T; t++) {
            __m512 ct = _mm512_set1_ps(coef[t]);
            const float *tp = tmpl + (int64_t)t * L * L + i * L;
            for (int c = 0; c < 4; c++) a[c] = _mm512_fmadd_ps(ct, _mm512_loadu_ps(tp + 16 * c), a[c]);
        }
        __m512 mx = _mm512_max_ps(_mm512_max_ps(a[0], a[1]), _mm512_max_ps(a[2], a[3]));
        __m512 m = _mm512_set1_ps(_mm512_reduce_max_ps(mx));
        __m512 sum = _mm512_setzero_ps();
        for (int c = 0; c < 4; c++) { a[c] = exp512(_mm512_sub_ps(a[c], m)); sum = _mm512_add_ps(sum, a[c]); }
        __m512 inv = _mm512_set1_ps(1.0f / _mm512_reduce_add_ps(sum));
        for (int c = 0; c < 4; c++) _mm512_store_ps(sr + 16 * c, _mm512_mul_ps(a[c], inv));
    }
}

typedef struct { float *Q, *Qt, *K, *Kt, *V, *Vt, *S, *dO, *O, *dP, *dQ, *dK, *dV; } Scratch;

static float *falloc(void) { float *p = aligned_alloc(64, sizeof(float) * L * L); memset(p, 0, sizeof(float) * L * L); return p; }

static Scratch scratch_new(void) {
    Scratch s = {falloc(), falloc(), falloc(), falloc(), falloc(), falloc(), falloc(), falloc(), falloc(), falloc(),
                 falloc(), falloc(), falloc()};
    return s;
}

static void scratch_free(Scratch *s) {
    float **p = (float **)s;
    for (int i = 0; i < 13; i++) free(p[i]);
}

int attn64_fwd(const void *q, const void *k, const void *v, const float *stat, const float *coef, const float *tmpl,
               int BH, int H, int D, int T, float scale, int bf16, void *o) {
    if (D % 16 || D > 64) return 1;
#pragma omp parallel
    {
        Scratch s = scratch_new();
#pragma omp for schedule(static)
        for (int bh = 0; bh < BH; bh++) {
            int64_t off = (int64_t)bh * L * D;
            load_mat(s.Q, NULL, q, off, D, bf16);
            load_mat(s.K, s.Kt, k, off, D, bf16);
            load_mat(s.V, NULL, v, off, D, bf16);
            mm_XYt(s.S, s.Q, s.Kt, D, scale);
            bias_softmax(s.S, stat ? stat + (int64_t)(bh % H) * L * L : NULL, coef + (int64_t)bh * T, tmpl, T);
            mm_AX(s.O, s.S, s.V, D, 1.0f);
            for (int i = 0; i < L; i++)
                for (int c = 0; c < D; c += 16) store16(o, off + i * D + c, _mm512_load_ps(s.O + i * D + c), bf16);
        }
        scratch_free(&s);
    }
    return 0;
}

// Backward with in-kernel bias gradients. Per-thread partial sums live in `partial`
// (omp_get_max_threads() * (H + T) * 4096 floats): dstatic[h] += dS, dtmpl[t] += coef[t] * dS, and
// dcoef[bh, t] = <dS, tmpl[t]>. dstat / dtmpl are overwritten with the reduced sums.
int attn64_bwd(const void *q, const void *k, const void *v, const void *o, const void *dout, const float *stat,
               const float *coef, const float *tmpl, int BH, int H, int D, int T, float scale, int bf16,
               void *dq, void *dk, void *dv, float *dstat, float *dcoef, float *dtmpl, float *partial) {
    if (D % 16 || D > 64) return 1;
    int nt = omp_get_max_threads();
    int64_t per = (int64_t)(H + T) * L * L;
#pragma omp parallel
    {
        Scratch s = scratch_new();
        float Dv[L] __attribute__((aligned(64)));
        float *my = partial + per * omp_get_thread_num();
        memset(my, 0, sizeof(float) * per);
#pragma omp for schedule(static)
        for (int bh = 0; bh < BH; bh++) {
            int64_t off = (int64_t)bh * L * D;
            const float *cf = coef + (int64_t)bh * T;
            load_mat(s.Q, NULL, q, off, D, bf16);
            load_mat(s.K, s.Kt, k, off, D, bf16);
            load_mat(s.V, s.Vt, v, off, D, bf16);
            load_mat(s.O, NULL, o, off, D, bf16);
            load_mat(s.dO, NULL, dout, off, D, bf16);
            mm_XYt(s.S, s.Q, s.Kt, D, scale);                                         // recompute P
            bias_softmax(s.S, stat ? stat + (int64_t)(bh % H) * L * L : NULL, cf, tmpl, T);
            for (int i = 0; i < L; i++) {
                __m512 acc = _mm512_setzero_ps();
                for (int c = 0; c < D; c += 16)
                    acc = _mm512_fmadd_ps(_mm512_load_ps(s.dO + i * D + c), _mm512_load_ps(s.O + i * D + c), acc);
                Dv[i] = _mm512_reduce_add_ps(acc);
            }
            mm_XYt(s.dP, s.dO, s.Vt, D, 1.0f);                                        // dP = dO V^T
            float *ms = my + (int64_t)(bh % H) * L * L;
            for (int i = 0; i < L; i++) {                                             // dS = P (dP - D)
                __m512 dd = _mm512_set1_ps(Dv[i]);
                for (int c = 0; c < L; c += 16) {
                    int x = i * L + c;
                    __m512 g = _mm512_mul_ps(_mm512_load_ps(s.S + x), _mm512_sub_ps(_mm512_load_ps(s.dP + x), dd));
                    _mm512_store_ps(s.dP + x, g);
                    _mm512_storeu_ps(ms + x, _mm512_add_ps(_mm512_loadu_ps(ms + x), g));
                }
            }
            for (int t = 0; t < T; t++) {
                const float *tp = tmpl + (int64_t)t * L * L;
                float *mt = my + (int64_t)(H + t) * L * L;
                __m512 c = _mm512_set1_ps(cf[t]), acc = _mm512_setzero_ps();
                for (int x = 0; x < L * L; x += 16) {
                    __m512 g = _mm512_load_ps(s.dP + x);
                    acc = _mm512_fmadd_ps(g, _mm512_loadu_ps(tp + x), acc);
                    _mm512_storeu_ps(mt + x, _mm512_fmadd_ps(c, g, _mm512_loadu_ps(mt + x)));
                }
                dcoef[(int64_t)bh * T + t] = _mm512_reduce_add_ps(acc);
            }
            mm_AtX(s.dV, s.S, s.dO, D, 1.0f);                                         // dV = P^T dO
            mm_AX(s.dQ, s.dP, s.K, D, scale);                                         // dQ = scale dS K
            mm_AtX(s.dK, s.dP, s.Q, D, scale);                                        // dK = scale dS^T Q
            for (int i = 0; i < L; i++)
                for (int c = 0; c < D; c += 16) {
                    int64_t ix = off + i * D + c;
                    store16(dq, ix, _mm512_load_ps(s.dQ + i * D + c), bf16);
                    store16(dk, ix, _mm512_load_ps(s.dK + i * D + c), bf16);
                    store16(dv, ix, _mm512_load_ps(s.dV + i * D + c), bf16);
                }
        }
        scratch_free(&s);
    }
    // reduce per-thread partials (parallel over the output elements)
    int64_t nH = (int64_t)H * L * L, nT = (int64_t)T * L * L;
#pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < nH + nT; x++) {
        float acc = 0.f;
        for (int th = 0; th < nt; th++) acc += partial[per * th + x];
        if (x < nH) dstat[x] = acc; else dtmpl[x - nH] = acc;
    }
    return 0;
}

int attn64_threads(void) { return omp_get_max_threads(); }
